#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/gelu_tiling.h"
#include "../op_kernel/tiling_key_gelu.h"

namespace {
constexpr int64_t kHalfTileCap = 8192;
constexpr int64_t kFloatTileCap = 4096;
constexpr int64_t kUbGuardBytes = 16 * 1024;
constexpr size_t kWorkspaceSlots = 1;
constexpr int64_t kDmaBlockBytes = 32;
constexpr int64_t kVectorPackAlign = 64;
constexpr int64_t kSmallCoreGrain = 512;
constexpr int64_t kLargeCoreGrain = 768;
constexpr int64_t kLargeSplitPoint = 8192;
constexpr int64_t kMiddleCoreGrain = 1024;
constexpr int64_t kMiddleSplitLimit = 262144;
constexpr int64_t kX5SingleTileLimit = 262144;




static int64_t EstimateUbStride(ge::DataType dtype)
{
    int64_t typeSize = ge::GetSizeByDataType(dtype);
    return 5 * typeSize;
}

static int64_t RoundUpTo(int64_t value, int64_t align)
{
    return RoundUpDiv(value, align) * align;
}
static int64_t RoundDownTo(int64_t value, int64_t align)
{
    return value / align * align;
}

static int64_t RoundUpDiv(int64_t a, int64_t b)
{
    return (a + b - 1) / b;
}

static int64_t ChooseVectorAlign(ge::DataType dtype)
{
    int64_t typeSize = ge::GetSizeByDataType(dtype);
    int64_t coreAlign = kDmaBlockBytes / typeSize;
    return std::max(coreAlign, kVectorPackAlign);
}

static int64_t PickCoreGrain(int64_t elemCount)
{
    if (elemCount > kLargeSplitPoint && elemCount <= kMiddleSplitLimit) {
        return kMiddleCoreGrain;
    }
    return elemCount > kLargeSplitPoint ? kLargeCoreGrain : kSmallCoreGrain;
}
}

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    int64_t hwCores = platform.GetCoreNumAiv();
    if (hwCores <= 0) {
        hwCores = 1;
    }

    uint64_t ubBytes = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubBytes);
    if (ubBytes <= kUbGuardBytes) {
        return ge::GRAPH_FAILED;
    }
    const gert::Tensor *xTensor = context->GetRequiredInputTensor(0);
    if (xTensor == nullptr) {
        return ge::GRAPH_FAILED;
    }
    ge::DataType xDtype = xTensor->GetDataType();
    if (xDtype != ge::DT_FLOAT16 && xDtype != ge::DT_FLOAT) {
        return ge::GRAPH_FAILED;
    }
    int64_t elemCount = static_cast<int64_t>(xTensor->GetShapeSize());
    auto shapeView = context->GetInputShape(0);
    if (shapeView != nullptr) {
        auto storageView = shapeView->GetStorageShape();
        elemCount = storageView.GetDimNum() == 0 ? 1 : static_cast<int64_t>(storageView.GetShapeSize());
    }
    if (elemCount < 0) {
        return ge::GRAPH_FAILED;
    }

    int64_t packAlign = ChooseVectorAlign(xDtype);
    int64_t tileAlignHint = (xDtype == ge::DT_FLOAT) ? kFloatTileCap : kHalfTileCap;
    int64_t plannedCores = 1;
    if (elemCount > 0) {
        int64_t coreGrain = PickCoreGrain(elemCount);
        plannedCores = std::min(hwCores, std::max<int64_t>(1, RoundUpDiv(elemCount, coreGrain)));
    }
    int64_t rawCoreSpan = (elemCount == 0) ? 0 : RoundUpDiv(elemCount, plannedCores);
    int64_t coreAlign = (rawCoreSpan > tileAlignHint) ? tileAlignHint : packAlign;
    int64_t coreSpan = (elemCount == 0) ? 0 : RoundUpTo(rawCoreSpan, coreAlign);
    int64_t launchCores = (elemCount == 0) ? 1 : RoundUpDiv(elemCount, coreSpan);
    launchCores = std::max<int64_t>(1, std::min(hwCores, launchCores));

    int64_t ubBytesPerElem = EstimateUbStride(xDtype);
    int64_t ubTileLimit = RoundDownTo((static_cast<int64_t>(ubBytes) - kUbGuardBytes) / ubBytesPerElem,
                                      packAlign);
    int64_t typeTileCap = (xDtype == ge::DT_FLOAT) ? kFloatTileCap : kHalfTileCap;
    int64_t tileSpan = std::max<int64_t>(packAlign, std::min<int64_t>(typeTileCap, ubTileLimit));
    if (coreSpan > 0) {
        tileSpan = std::min(tileSpan, RoundUpTo(coreSpan, packAlign));
        tileSpan = std::max<int64_t>(packAlign, RoundDownTo(tileSpan, packAlign));
    }

    GeluTilingData *tiling = context->GetTilingData<GeluTilingData>();
    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tiling->total_elems = elemCount;
    tiling->core_elems = coreSpan;
    tiling->tile_elems = tileSpan;

    context->SetBlockDim(static_cast<uint32_t>(launchCores));
    size_t *workspace = context->GetWorkspaceSizes(kWorkspaceSlots);
    if (workspace == nullptr) {
        return ge::GRAPH_FAILED;
    }
    workspace[0] = 0;

    uint32_t chooseX5 =
        (elemCount > kX5SingleTileLimit && coreSpan <= tileSpan) ? 1U : 0U;
    int64_t coreLoops = (coreSpan + tileSpan - 1) / tileSpan;
    uint32_t chooseLogistic =
        (xDtype == ge::DT_FLOAT && chooseX5 == 0U && coreLoops >= 4) ? 1U : 0U;
    ASCENDC_TPL_SEL_PARAM(context, static_cast<uint32_t>(xDtype), chooseX5, chooseLogistic);
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *shapeView = context->GetInputShape(0);
    gert::Shape *yShape = context->GetOutputShape(0);
    if (shapeView == nullptr || yShape == nullptr) {
        return GRAPH_FAILED;
    }
    *yShape = *shapeView;
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
