// Host侧代码：算子注册、输出信息推导和Tiling切分。
//
// Ascend C 自定义算子通常分成两部分：
// 1. Host侧：运行在CPU侧的CANN框架中，负责告诉框架“这个算子叫什么、支持什么输入输出、
//    输出shape和dtype是什么、AI Core要启动多少个核、每个核处理多少数据”。
// 2. Kernel侧：运行在NPU的AI Core上，真正执行GM到UB的数据搬运和向量计算。
//
// 本文件就是第1部分。这里不直接计算GELU数值，而是准备Kernel执行所需的元信息。
#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/gelu_tiling.h"
#include "../op_kernel/tiling_key_gelu.h"

namespace {
// Ascend AI Core的数据搬运以32字节为基本对齐粒度。输入长度不是32字节整数倍时，
// Kernel侧使用DataCopyPad处理尾块；Host侧仍尽量把切分长度对齐，减少尾块数量。
constexpr int64_t BLOCK_BYTES = 32;

// Vector指令一次通常适合处理一批连续元素。这里使用64个元素作为计算对齐粒度，
// 既覆盖float16的32B对齐，也让tile/block大小更规整。
constexpr int64_t COMPUTE_ALIGN = 64;

// 小/中shape每个AI Vector Core至少分到约512个元素时再增加并行核数。
// 数据太小时强行开很多核，启动和同步成本可能比计算本身还高。
// 线上实测显示512对1、2、4号点更友好，因此默认保持这个激进阈值。
constexpr int64_t MIN_ELEMS_PER_CORE = 512;

// 大shape使用稍粗的每核任务量。上一轮从4096开始切换会误伤1、4号点；
// 这里把切换点推到8192之后，只尝试影响更大规模的3、5号点。
constexpr int64_t LARGE_MIN_ELEMS_PER_CORE = 768;
constexpr int64_t LARGE_CORE_SPLIT_TOTAL = 8192;

// 80755显示把大shape每核最少元素提高到1024能明显改善3号点，
// 但会拖慢4/5号点。4/5历史上更像 totalLength > 262144 的超大shape，
// 因此这里只在中等大shape区间试用1024，超大shape仍保持768。
constexpr int64_t MID_LARGE_MIN_ELEMS_PER_CORE = 1024;
constexpr int64_t MID_LARGE_CORE_SPLIT_TOTAL = 262144;

// Kernel侧x5-tanh路径的总长度阈值。这里和Kernel里的判别条件保持一致，
// 目的是让Host直接选择不同模板实例，默认路径和x5路径在编译后互不影响。
constexpr int64_t X5_SINGLE_TILE_TOTAL_THRESHOLD = 262144;

// float16单次进入UB计算的最大元素数。上一轮把它提到16384后，5号大shape变慢，
// 说明更大tile虽然减少了DataCopy次数，但会降低当前GELU流水/UB布局效率。
// 回到8192，让大shape继续使用更稳定的16KB输入块。
constexpr int64_t MAX_TILE_LENGTH = 8192;

// float32改用较小的tile，目的是把每个核的大block拆成多个tile，
// 配合Kernel侧的自适应双缓冲，让搬运与计算在流水线上重叠以隐藏访存延迟(主攻大张量测试点)。
// 4096个float32 = 16KB，符合“单次搬运>=16KB”的DMA效率建议；block<=4096时仍是单tile、单缓冲，
// 所以中小float32张量不受影响。
constexpr int64_t F32_PIPELINE_TILE = 4096;

// UB是AI Core上的片上高速缓存。这里预留16KB给编译器、队列元数据和潜在对齐开销，
// 避免把UB算得太满导致运行时申请buffer失败。
constexpr int64_t UB_RESERVE_BYTES = 16 * 1024;

// 本算子不需要workspace，但CANN接口要求设置workspace数组。
constexpr size_t WORKSPACE_NUM = 1;

// 向上整除。例如CeilDiv(10, 4) = 3。
// 在tiling里经常用于“总长度 / 核数”或“总长度 / tile长度”。
static int64_t CeilDiv(int64_t a, int64_t b)
{
    return (a + b - 1) / b;
}

// 把value向上对齐到align的整数倍。例如CeilAlign(130, 64) = 192。
// 对齐后的长度可能略大于真实数据长度，Kernel侧会用真实total_length防止越界。
static int64_t CeilAlign(int64_t value, int64_t align)
{
    return CeilDiv(value, align) * align;
}

// 把value向下对齐到align的整数倍。例如FloorAlign(130, 64) = 128。
// 计算UB能容纳多少元素时，向下对齐可以保证不会超过UB容量。
static int64_t FloorAlign(int64_t value, int64_t align)
{
    return value / align * align;
}

// 根据dtype计算“多少个元素”对应至少32字节，同时不小于COMPUTE_ALIGN。
// float32是4字节，32B对应8个元素；float16是2字节，32B对应16个元素。
// 最后统一至少64元素，让block/tile长度更适合向量化处理。
static int64_t GetAlignElements(ge::DataType dtype)
{
    int64_t typeSize = ge::GetSizeByDataType(dtype);
    int64_t blockAlign = BLOCK_BYTES / typeSize;
    return std::max(blockAlign, COMPUTE_ALIGN);
}

// 估算每个元素在UB里需要多少字节，用于Host侧计算tile大小。
//
// float16和float32都用各自的类型直接计算(不升精度)。最坏情况下(多tile)输入/输出队列
// 各开2块UB做双缓冲。当前rational-tanh实验需要额外1块tmpBuf保存x^2和分母，
// 因此按最坏情况估算为：
// - 输入队列：2个T（双缓冲）
// - 输出队列：2个T（双缓冲）
// - tmpBuf：1个T
// 合计 5 * sizeof(T)：float32为20字节，float16为10字节。
// 用最坏情况(双缓冲)估算，能保证即使开了双缓冲也不会超UB；单缓冲场景只会更省，绝不溢出。
static int64_t GetUbBytesPerElement(ge::DataType dtype)
{
    int64_t typeSize = ge::GetSizeByDataType(dtype);
    // (输入队列2 + 输出队列2)双缓冲 + 1个临时buffer = 5份，全部是输入dtype本身的类型。
    return 5 * typeSize;
}

// 根据总元素数选择开核阈值。
// 小shape继续用512保护短耗时测试点；中等大shape用1024主攻3号点；
// 超大shape回到768，避免把4/5号点也带进1024策略导致退化。
static int64_t GetMinElemsPerCore(int64_t totalLength)
{
    if (totalLength > LARGE_CORE_SPLIT_TOTAL && totalLength <= MID_LARGE_CORE_SPLIT_TOTAL) {
        return MID_LARGE_MIN_ELEMS_PER_CORE;
    }
    return totalLength > LARGE_CORE_SPLIT_TOTAL ? LARGE_MIN_ELEMS_PER_CORE : MIN_ELEMS_PER_CORE;
}
}  // namespace

namespace optiling {
// TilingFunc是Host侧最关键的函数之一。
// CANN框架在真正启动AI Core Kernel前会调用它。它的任务是：
// 1. 读取输入tensor的shape和dtype；
// 2. 查询硬件有多少个AI Vector Core、每个Core有多大UB；
// 3. 计算每个Core负责多少元素，以及每次搬进UB计算多少元素；
// 4. 把这些数字写入GeluTilingData，Kernel启动后会读取这份数据。
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    // PlatformAscendC封装了目标NPU的信息。这里主要用到AI Vector Core数量和UB大小。
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    int64_t coreNum = platform.GetCoreNumAiv();
    if (coreNum <= 0) {
        coreNum = 1;
    }

    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    if (ubSize <= UB_RESERVE_BYTES) {
        // UB比预留空间还小，说明无法安全分配输入、输出和中间buffer。
        return ge::GRAPH_FAILED;
    }

    // 获取第0个输入tensor，也就是GELU的input_x。
    const gert::Tensor *inputTensor = context->GetRequiredInputTensor(0);
    if (inputTensor == nullptr) {
        return ge::GRAPH_FAILED;
    }

    // 赛题只要求支持float16和float32。其他dtype直接返回失败，避免Kernel模板选型错误。
    ge::DataType inputDtype = inputTensor->GetDataType();
    if (inputDtype != ge::DT_FLOAT16 && inputDtype != ge::DT_FLOAT) {
        return ge::GRAPH_FAILED;
    }

    // GELU是逐元素算子，不关心原始是1维、2维还是8维。
    // 因此Host侧把shape展平成一个totalLength，Kernel按一段连续内存处理。
    int64_t totalLength = static_cast<int64_t>(inputTensor->GetShapeSize());
    auto inputShape = context->GetInputShape(0);
    if (inputShape != nullptr) {
        auto storageShape = inputShape->GetStorageShape();
        // 0维标量的shape维度数为0，但它仍然有1个元素。
        totalLength = storageShape.GetDimNum() == 0 ? 1 : static_cast<int64_t>(storageShape.GetShapeSize());
    }
    if (totalLength < 0) {
        return ge::GRAPH_FAILED;
    }

    int64_t alignElements = GetAlignElements(inputDtype);
    // 当单核任务量超过一个tile时，把每核block按tile粒度对齐。
    // 这样大shape下大多数AI Core都处理完整tile，只把非对齐尾块集中到最后少数Core，
    // 可以减少每个Core最后一轮都触发小块DataCopyPad的次数，主攻5号大shape。
    int64_t preferredTileAlign = (inputDtype == ge::DT_FLOAT) ? F32_PIPELINE_TILE : MAX_TILE_LENGTH;
    int64_t usedCoreNum = 1;
    if (totalLength > 0) {
        // 小数据少开核，大数据多开核。这样可以兼顾启动开销和并行度。
        int64_t minElemsPerCore = GetMinElemsPerCore(totalLength);
        usedCoreNum = std::min(coreNum, std::max<int64_t>(1, CeilDiv(totalLength, minElemsPerCore)));
    }

    // blockLength表示“一个AI Core最多处理多少元素”。
    // 它是对齐后的长度，所以最后一个Core可能拿到的真实数据少一些。
    int64_t baseBlockLength = (totalLength == 0) ? 0 : CeilDiv(totalLength, usedCoreNum);
    int64_t blockAlign = (baseBlockLength > preferredTileAlign) ? preferredTileAlign : alignElements;
    int64_t blockLength = (totalLength == 0) ? 0 : CeilAlign(baseBlockLength, blockAlign);

    // realCoreNum表示这次Kernel真正启动多少个AI Core。
    // SetBlockDim(realCoreNum)后，Kernel里AscendC::GetBlockIdx()会返回0到realCoreNum-1。
    int64_t realCoreNum = (totalLength == 0) ? 1 : CeilDiv(totalLength, blockLength);
    realCoreNum = std::max<int64_t>(1, std::min(coreNum, realCoreNum));

    // tileLength表示“一个Core每轮从GM搬到UB计算多少元素”。
    // 它受三件事限制：MAX_TILE_LENGTH、UB容量、当前Core分到的blockLength。
    int64_t bytesPerElement = GetUbBytesPerElement(inputDtype);
    int64_t maxTileByUb = FloorAlign((static_cast<int64_t>(ubSize) - UB_RESERVE_BYTES) / bytesPerElement,
                                     alignElements);
    // float32用较小的tile(F32_PIPELINE_TILE)把大block拆成多tile以便流水重叠；
    // float16使用MAX_TILE_LENGTH对应的稳定16KB输入块。
    int64_t maxTile = (inputDtype == ge::DT_FLOAT) ? F32_PIPELINE_TILE : MAX_TILE_LENGTH;
    int64_t tileLength = std::max<int64_t>(alignElements, std::min<int64_t>(maxTile, maxTileByUb));
    if (blockLength > 0) {
        tileLength = std::min(tileLength, CeilAlign(blockLength, alignElements));
        tileLength = std::max<int64_t>(alignElements, FloorAlign(tileLength, alignElements));
    }

    // 把Host侧算出的切分结果写入tiling区。Kernel启动后会通过GET_TILING_DATA_WITH_STRUCT读取。
    GeluTilingData *tiling = context->GetTilingData<GeluTilingData>();
    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tiling->total_length = totalLength;
    tiling->block_length = blockLength;
    tiling->tile_length = tileLength;

    context->SetBlockDim(static_cast<uint32_t>(realCoreNum));

    // 本算子所有中间数据都放在UB中，不需要额外GM workspace。
    size_t *workspace = context->GetWorkspaceSizes(WORKSPACE_NUM);
    if (workspace == nullptr) {
        return ge::GRAPH_FAILED;
    }
    workspace[0] = 0;

    // 根据输入dtype和shape形态选择Kernel模板实例：
    // - float16输入会实例化KernelGelu<half, ...>
    // - float32输入会实例化KernelGelu<float, ...>
    // - useX5Template=1时，编译期选择x5-tanh路径；否则选择默认rational-tanh路径。
    // - useStdTanhTemplate=1时，编译期选择float32长循环标准tanh路径，尝试主攻5号点。
    //
    // 之前在Kernel Init里用运行时bool判断x5路径。现在把这个判断前移到Host侧，
    // 让非x5测试点不携带x5分支代码，减少指令布局和分支预测对短耗时shape的影响。
    uint32_t useX5Template =
        (totalLength > X5_SINGLE_TILE_TOTAL_THRESHOLD && blockLength <= tileLength) ? 1U : 0U;
    int64_t loopCount = (blockLength + tileLength - 1) / tileLength;
    uint32_t useStdTanhTemplate =
        (inputDtype == ge::DT_FLOAT && useX5Template == 0U && loopCount >= 4) ? 1U : 0U;
    ASCENDC_TPL_SEL_PARAM(context, static_cast<uint32_t>(inputDtype), useX5Template, useStdTanhTemplate);
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
// InferShape告诉框架输出shape是什么。
// GELU逐元素计算，不改变形状，所以直接把输入shape复制给输出。
static graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *inputShape = context->GetInputShape(0);
    gert::Shape *outputShape = context->GetOutputShape(0);
    if (inputShape == nullptr || outputShape == nullptr) {
        return GRAPH_FAILED;
    }
    *outputShape = *inputShape;
    return GRAPH_SUCCESS;
}

// InferDataType告诉框架输出dtype是什么。
// 赛题要求输出类型和输入类型一致，所以直接透传。
static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
// OpDef是CANN自定义算子的“声明书”。
// 框架通过这里知道算子名、输入输出名字、支持的数据类型/格式，以及Host侧推导函数。
class Gelu : public OpDef {
public:
    explicit Gelu(const char *name) : OpDef(name)
    {
        this->Input("input_x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
            // AutoContiguous要求输入在内存中尽量是连续布局，便于Kernel按扁平数组读取。
            .AutoContiguous();
        this->Output("output")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore()
            // 把上面的TilingFunc绑定到AI Core Kernel启动流程。
            .SetTiling(optiling::TilingFunc)
            // 本赛题评测环境使用ascend910b。
            .AddConfig("ascend910b");
    }
};

// OP_ADD把Gelu注册进CANN算子包。没有这一行，框架找不到这个自定义算子。
OP_ADD(Gelu);
}  // namespace ops
