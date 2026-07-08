#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/gelu_tiling.h"
#include "../op_kernel/tiling_key_gelu.h"

namespace {
constexpr int64_t BLOCK_BYTES = 32;
constexpr int64_t COMPUTE_ALIGN = 64;
constexpr int64_t MIN_ELEMS_PER_CORE = 512;
constexpr int64_t LARGE_MIN_ELEMS_PER_CORE = 768;
constexpr int64_t LARGE_CORE_SPLIT_TOTAL = 8192;
constexpr int64_t MID_LARGE_MIN_ELEMS_PER_CORE = 1024;
constexpr int64_t MID_LARGE_CORE_SPLIT_TOTAL = 262144;
constexpr int64_t X5_SINGLE_TILE_TOTAL_THRESHOLD = 262144;
constexpr int64_t MAX_TILE_LENGTH = 8192;
constexpr int64_t F32_PIPELINE_TILE = 4096;
constexpr int64_t UB_RESERVE_BYTES = 16 * 1024;
constexpr size_t WORKSPACE_NUM = 1;
static int64_t CeilDiv(int64_t a, int64_t b)
{
    return (a + b - 1) / b;
}
static int64_t CeilAlign(int64_t value, int64_t align)
{
    return CeilDiv(value, align) * align;
}
static int64_t FloorAlign(int64_t value, int64_t align)
{
    return value / align * align;
}
static int64_t GetAlignElements(ge::DataType dtype)
{
    int64_t typeSize = ge::GetSizeByDataType(dtype);
    int64_t blockAlign = BLOCK_BYTES / typeSize;
    return std::max(blockAlign, COMPUTE_ALIGN);
}
static int64_t GetUbBytesPerElement(ge::DataType dtype)
{
    int64_t typeSize = ge::GetSizeByDataType(dtype);
    return 5 * typeSize;
}
static int64_t GetMinElemsPerCore(int64_t totalLength)
{
    if (totalLength > LARGE_CORE_SPLIT_TOTAL && totalLength <= MID_LARGE_CORE_SPLIT_TOTAL) {
        return MID_LARGE_MIN_ELEMS_PER_CORE;
    }
    return totalLength > LARGE_CORE_SPLIT_TOTAL ? LARGE_MIN_ELEMS_PER_CORE : MIN_ELEMS_PER_CORE;
}
} optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    int64_t coreNum = platform.GetCoreNumAiv();
    if (coreNum <= 0) {
        coreNum = 1;
    }

    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    if (ubSize <= UB_RESERVE_BYTES) {
        return ge::GRAPH_FAILED;
    }
    const gert::Tensor *inputTensor = context->GetRequiredInputTensor(0);
    if (inputTensor == nullptr) {
        return ge::GRAPH_FAILED;
    }
    ge::DataType inputDtype = inputTensor->GetDataType();
    if (inputDtype != ge::DT_FLOAT16 && inputDtype != ge::DT_FLOAT) {
        return ge::GRAPH_FAILED;
    }
    int64_t totalLength = static_cast<int64_t>(inputTensor->GetShapeSize());
    auto inputShape = context->GetInputShape(0);
    if (inputShape != nullptr) {
        auto storageShape = inputShape->GetStorageShape();
        totalLength = storageShape.GetDimNum() == 0 ? 1 : static_cast<int64_t>(storageShape.GetShapeSize());
    }
    if (totalLength < 0) {
        return ge::GRAPH_FAILED;
    }

    int64_t alignElements = GetAlignElements(inputDtype);
    int64_t preferredTileAlign = (inputDtype == ge::DT_FLOAT) ? F32_PIPELINE_TILE : MAX_TILE_LENGTH;
    int64_t usedCoreNum = 1;
    if (totalLength > 0) {
        int64_t minElemsPerCore = GetMinElemsPerCore(totalLength);
        usedCoreNum = std::min(coreNum, std::max<int64_t>(1, CeilDiv(totalLength, minElemsPerCore)));
    }
    int64_t baseBlockLength = (totalLength == 0) ? 0 : CeilDiv(totalLength, usedCoreNum);
    int64_t blockAlign = (baseBlockLength > preferredTileAlign) ? preferredTileAlign : alignElements;
    int64_t blockLength = (totalLength == 0) ? 0 : CeilAlign(baseBlockLength, blockAlign);
    int64_t realCoreNum = (totalLength == 0) ? 1 : CeilDiv(totalLength, blockLength);
    realCoreNum = std::max<int64_t>(1, std::min(coreNum, realCoreNum));
    int64_t bytesPerElement = GetUbBytesPerElement(inputDtype);
    int64_t maxTileByUb = FloorAlign((static_cast<int64_t>(ubSize) - UB_RESERVE_BYTES) / bytesPerElement,
                                     alignElements);
    int64_t maxTile = (inputDtype == ge::DT_FLOAT) ? F32_PIPELINE_TILE : MAX_TILE_LENGTH;
    int64_t tileLength = std::max<int64_t>(alignElements, std::min<int64_t>(maxTile, maxTileByUb));
    if (blockLength > 0) {
        tileLength = std::min(tileLength, CeilAlign(blockLength, alignElements));
        tileLength = std::max<int64_t>(alignElements, FloorAlign(tileLength, alignElements));
    }
    GeluTilingData *tiling = context->GetTilingData<GeluTilingData>();
    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tiling->total_length = totalLength;
    tiling->block_length = blockLength;
    tiling->tile_length = tileLength;

    context->SetBlockDim(static_cast<uint32_t>(realCoreNum));
    size_t *workspace = context->GetWorkspaceSizes(WORKSPACE_NUM);
    if (workspace == nullptr) {
        return ge::GRAPH_FAILED;
    }
    workspace[0] = 0;
    uint32_t useX5Template =
        (totalLength > X5_SINGLE_TILE_TOTAL_THRESHOLD && blockLength <= tileLength) ? 1U : 0U;
    int64_t loopCount = (blockLength + tileLength - 1) / tileLength;
    uint32_t useStdTanhTemplate =
        (inputDtype == ge::DT_FLOAT && useX5Template == 0U && loopCount >= 4) ? 1U : 0U;
    ASCENDC_TPL_SEL_PARAM(context, static_cast<uint32_t>(inputDtype), useX5Template, useStdTanhTemplate);
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
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
static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
}

namespace ops {
class Gelu : public OpDef {
public:
    explicit Gelu(const char *name) : OpDef(name)
    {
        this->Input("input_x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("output")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};
OP_ADD(Gelu);
}
