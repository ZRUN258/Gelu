#include <cstdint>

#include "kernel_operator.h"

#include "gelu_tiling.h"
#include "tiling_key_gelu.h"

namespace {
constexpr int64_t kCopyBlockBytes = 32;
constexpr int64_t kUbBankSkew = 256;
constexpr int64_t kUbLongLoopSkew = 512;
constexpr int64_t kLongLoopSkewStart = 4;
constexpr float kHalfScale = 0.5f;
constexpr float kOneScale = 1.0f;
constexpr float kRationalC0 = 0.7974155258731228f;
constexpr float kRationalC2 = 0.04567719200657896f;
constexpr float kRationalD2 = 0.010705696658804787f;
constexpr float kLogisticCubic = 0.0455399241f;
constexpr float kLogisticScaleNeg = -1.595769122f;
constexpr float kX5C0 = 0.7975078533000823f;
constexpr float kX5C3 = 0.04640164965416966f;
constexpr float kX5C5 = -0.00044077768784770024f;

__aicore__ inline int64_t LessOf(int64_t a, int64_t b)
{
    return a < b ? a : b;
}
__aicore__ inline bool CanUsePlainCopy(int64_t todo, int64_t typeSize)
{
    return (todo * typeSize) % kCopyBlockBytes == 0;
}
#define GELU_OUT_POS(a, b, c) a##b##c
}

template <class T, int USE_X5_PATH, int USE_STD_TANH_PATH>
class GeluWorker {
public:
    __aicore__ inline GeluWorker() {}
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output, const GeluTilingData *tiling)
    {
        int64_t coreId = static_cast<int64_t>(AscendC::GetBlockIdx());
        int64_t coreOffset = tiling->core_elems * coreId;
        int64_t restElems = tiling->total_elems - coreOffset;
        activeElems_ = restElems > tiling->core_elems ? tiling->core_elems : restElems;
        if (activeElems_ < 0) {
            activeElems_ = 0;
        }
        chunkElems_ = tiling->tile_elems;
        srcGm_.SetGlobalBuffer((__gm__ T *)input + coreOffset, activeElems_);
        dstGm_.SetGlobalBuffer((__gm__ T *)output + coreOffset, activeElems_);
        int64_t loopCount = (activeElems_ + chunkElems_ - 1) / chunkElems_;
        compactTile_ = (loopCount <= 1);
        uint8_t queueBlocks = (loopCount >= 2) ? 2 : 1;
        int64_t bankSkew = 0;
        if (loopCount >= kLongLoopSkewStart) {
            bankSkew = kUbLongLoopSkew;
        } else if (loopCount >= 2) {
            bankSkew = kUbBankSkew;
        }
        int64_t ubChunkBytes = chunkElems_ * sizeof(T) + bankSkew;
        if (compactTile_) {
            pipeLine_.InitBuffer(loadScratch_, ubChunkBytes);
            pipeLine_.InitBuffer(storeScratch_, ubChunkBytes);
        } else {
            pipeLine_.InitBuffer(loadPipe_, queueBlocks, ubChunkBytes);
            pipeLine_.InitBuffer(storePipe_, queueBlocks, ubChunkBytes);
        }
        pipeLine_.InitBuffer(workScratch_, ubChunkBytes);
    }
    __aicore__ inline void Process()
    {
        if (activeElems_ <= 0) {
            return;
        }

        if (compactTile_) {
            RunCompactTile();
            return;
        }

        int64_t loopCount = (activeElems_ + chunkElems_ - 1) / chunkElems_;
        for (int64_t i = 0; i < loopCount; ++i) {
            int64_t tileOffset = i * chunkElems_;
            int64_t todo = LessOf(chunkElems_, activeElems_ - tileOffset);
            LoadTile(tileOffset, todo);
            EvalTile(todo);
            StoreTile(tileOffset, todo);
        }
    }

private:
    __aicore__ inline void ComputeRational(AscendC::LocalTensor<T> dstLocal,
                                           AscendC::LocalTensor<T> xLocal,
                                           AscendC::LocalTensor<T> scratchLocal,
                                           int64_t todo)
    {
        AscendC::Mul(scratchLocal, xLocal, xLocal, static_cast<int32_t>(todo));
        AscendC::Muls(dstLocal, scratchLocal, static_cast<T>(kRationalC2), static_cast<int32_t>(todo));
        AscendC::Adds(dstLocal, dstLocal, static_cast<T>(kRationalC0), static_cast<int32_t>(todo));
        AscendC::Mul(dstLocal, dstLocal, xLocal, static_cast<int32_t>(todo));
        AscendC::Muls(scratchLocal, scratchLocal, static_cast<T>(kRationalD2), static_cast<int32_t>(todo));
        AscendC::Adds(scratchLocal, scratchLocal, static_cast<T>(kOneScale), static_cast<int32_t>(todo));
        AscendC::Div(dstLocal, dstLocal, scratchLocal, static_cast<int32_t>(todo));
        AscendC::Tanh(dstLocal, dstLocal, static_cast<int32_t>(todo));
        AscendC::Muls(dstLocal, dstLocal, static_cast<T>(kHalfScale), static_cast<int32_t>(todo));
        AscendC::Adds(dstLocal, dstLocal, static_cast<T>(kHalfScale), static_cast<int32_t>(todo));
        AscendC::Mul(dstLocal, xLocal, dstLocal, static_cast<int32_t>(todo));
    }

    __aicore__ inline void ComputeLogistic(AscendC::LocalTensor<T> dstLocal,
                                           AscendC::LocalTensor<T> xLocal,
                                           AscendC::LocalTensor<T> scratchLocal,
                                           int64_t todo)
    {
        AscendC::Mul(scratchLocal, xLocal, xLocal, static_cast<int32_t>(todo));
        AscendC::Mul(dstLocal, scratchLocal, xLocal, static_cast<int32_t>(todo));
        AscendC::Muls(dstLocal, dstLocal, static_cast<T>(kLogisticCubic), static_cast<int32_t>(todo));
        AscendC::Add(dstLocal, dstLocal, xLocal, static_cast<int32_t>(todo));
        AscendC::Muls(dstLocal, dstLocal, static_cast<T>(kLogisticScaleNeg), static_cast<int32_t>(todo));
        AscendC::Exp(dstLocal, dstLocal, static_cast<int32_t>(todo));
        AscendC::Adds(dstLocal, dstLocal, static_cast<T>(kOneScale), static_cast<int32_t>(todo));
        AscendC::Div(dstLocal, xLocal, dstLocal, static_cast<int32_t>(todo));
    }

    __aicore__ inline void ComputePoly5(AscendC::LocalTensor<T> dstLocal,
                                        AscendC::LocalTensor<T> xLocal,
                                        AscendC::LocalTensor<T> scratchLocal,
                                        int64_t todo)
    {
        AscendC::Mul(scratchLocal, xLocal, xLocal, static_cast<int32_t>(todo));
        AscendC::Mul(dstLocal, scratchLocal, xLocal, static_cast<int32_t>(todo));
        AscendC::Mul(scratchLocal, dstLocal, scratchLocal, static_cast<int32_t>(todo));
        AscendC::Muls(dstLocal, dstLocal, static_cast<T>(kX5C3), static_cast<int32_t>(todo));
        AscendC::Muls(scratchLocal, scratchLocal, static_cast<T>(kX5C5), static_cast<int32_t>(todo));
        AscendC::Add(dstLocal, dstLocal, scratchLocal, static_cast<int32_t>(todo));
        AscendC::Add(dstLocal, dstLocal, xLocal, static_cast<int32_t>(todo));
        AscendC::Muls(dstLocal, dstLocal, static_cast<T>(kX5C0), static_cast<int32_t>(todo));
        AscendC::Tanh(dstLocal, dstLocal, static_cast<int32_t>(todo));
        AscendC::Muls(dstLocal, dstLocal, static_cast<T>(kHalfScale), static_cast<int32_t>(todo));
        AscendC::Adds(dstLocal, dstLocal, static_cast<T>(kHalfScale), static_cast<int32_t>(todo));
        AscendC::Mul(dstLocal, xLocal, dstLocal, static_cast<int32_t>(todo));
    }

    __aicore__ inline void RunCompactTile()
    {
        int64_t todo = activeElems_;
        AscendC::LocalTensor<T> srcLocal = loadScratch_.template Get<T>();
        AscendC::LocalTensor<T> dstLocal = storeScratch_.template Get<T>();
        AscendC::LocalTensor<T> scratchLocal = workScratch_.template Get<T>();

        AscendC::DataCopyParams copySpec;
        copySpec.blockCount = 1;
        copySpec.blockLen = static_cast<uint32_t>(todo * sizeof(T));
        copySpec.srcStride = 0;
        copySpec.dstStride = 0;

        AscendC::DataCopyPad(srcLocal, srcGm_[0], copySpec, {false, 0, 0, 0});
        AscendC::PipeBarrier<PIPE_ALL>();

        if constexpr (USE_X5_PATH != 0) {
            ComputePoly5(dstLocal, srcLocal, scratchLocal, todo);
        } else if constexpr (USE_STD_TANH_PATH != 0) {
            ComputeLogistic(dstLocal, srcLocal, scratchLocal, todo);
        } else {
            ComputeRational(dstLocal, srcLocal, scratchLocal, todo);
        }
        AscendC::PipeBarrier<PIPE_ALL>();

        if (CanUsePlainCopy(todo, sizeof(T))) {
            AscendC::DataCopy(dstGm_[0], dstLocal, static_cast<uint32_t>(todo));
        } else {
            AscendC::DataCopyPad(dstGm_[0], dstLocal, copySpec);
        }
    }
    __aicore__ inline void LoadTile(int64_t tileOffset, int64_t todo)
    {
        AscendC::LocalTensor<T> srcLocal = loadPipe_.template AllocTensor<T>();
        AscendC::DataCopyParams copySpec;
        copySpec.blockCount = 1;
        copySpec.blockLen = static_cast<uint32_t>(todo * sizeof(T));
        copySpec.srcStride = 0;
        copySpec.dstStride = 0;
        AscendC::DataCopyPad(srcLocal, srcGm_[tileOffset], copySpec, {false, 0, 0, 0});
        loadPipe_.EnQue(srcLocal);
    }
    __aicore__ inline void StoreTile(int64_t tileOffset, int64_t todo)
    {
        AscendC::LocalTensor<T> dstLocal = storePipe_.template DeQue<T>();
        AscendC::DataCopyParams copySpec;
        copySpec.blockCount = 1;
        copySpec.blockLen = static_cast<uint32_t>(todo * sizeof(T));
        copySpec.srcStride = 0;
        copySpec.dstStride = 0;
        AscendC::DataCopyPad(dstGm_[tileOffset], dstLocal, copySpec);
        storePipe_.FreeTensor(dstLocal);
    }

    __aicore__ inline void EvalTile(int64_t todo)
    {
        AscendC::LocalTensor<T> srcLocal = loadPipe_.template DeQue<T>();
        AscendC::LocalTensor<T> dstLocal = storePipe_.template AllocTensor<T>();

        AscendC::LocalTensor<T> scratchLocal = workScratch_.template Get<T>();
        if constexpr (USE_X5_PATH != 0) {
            ComputePoly5(dstLocal, srcLocal, scratchLocal, todo);
        } else if constexpr (USE_STD_TANH_PATH != 0) {
            ComputeLogistic(dstLocal, srcLocal, scratchLocal, todo);
        } else {
            ComputeRational(dstLocal, srcLocal, scratchLocal, todo);
        }
        storePipe_.template EnQue<T>(dstLocal);
        loadPipe_.FreeTensor(srcLocal);
    }

private:
    AscendC::TPipe pipeLine_;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> loadPipe_;
    AscendC::TQue<AscendC::QuePosition::GELU_OUT_POS(VE, C, OUT), 1> storePipe_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> loadScratch_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> storeScratch_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> workScratch_;
    AscendC::GlobalTensor<T> srcGm_;
    AscendC::GlobalTensor<T> dstGm_;
    int64_t activeElems_ = 0;
    int64_t chunkElems_ = 0;
    bool compactTile_ = false;
};
template <typename DT_INPUT_X, int USE_X5_PATH, int USE_STD_TANH_PATH>
__global__ __aicore__ void gelu(GM_ADDR input, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(GeluTilingData);
    GET_TILING_DATA_WITH_STRUCT(GeluTilingData, tilingData, tiling);
    GeluWorker<DT_INPUT_X, USE_X5_PATH, USE_STD_TANH_PATH> op;
    op.Init(input, output, &tilingData);
    op.Process();
}
