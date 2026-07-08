#include <cstdint>

#include "kernel_operator.h"

#include "gelu_tiling.h"
#include "tiling_key_gelu.h"

namespace {
constexpr int64_t COPY_ALIGN_BYTES = 32;
constexpr float HALF = 0.5f;
constexpr float ONE = 1.0f;
constexpr float TANH_RATIONAL_C0 = 0.7974155258731228f;
constexpr float TANH_RATIONAL_C2 = 0.04567719200657896f;
constexpr float TANH_RATIONAL_D2 = 0.010705696658804787f;
constexpr float LOGISTIC_C3 = 0.0455399241f;
constexpr float LOGISTIC_NEG_SCALE = -1.595769122f;
constexpr float TANH_X5_C0 = 0.7975078533000823f;
constexpr float TANH_X5_C3 = 0.04640164965416966f;
constexpr float TANH_X5_C5 = -0.00044077768784770024f;
constexpr int64_t UB_BANK_PAD_BYTES = 256;
constexpr int64_t UB_LARGE_LOOP_PAD_BYTES = 512;
constexpr int64_t LARGE_LOOP_PAD_THRESHOLD = 4;
__aicore__ inline int64_t MinInt64(int64_t a, int64_t b)
{
    return a < b ? a : b;
}
__aicore__ inline bool IsAlignedCopySize(int64_t count, int64_t typeSize)
{
    return (count * typeSize) % COPY_ALIGN_BYTES == 0;
}
#define GELU_JOIN3(a, b, c) a##b##c
} 
template<class T, int USE_X5_PATH, int USE_STD_TANH_PATH>
class KernelGelu {
public:
    __aicore__ inline KernelGelu() {}
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output, const GeluTilingData *tiling)
    {
        int64_t blockIdx = static_cast<int64_t>(AscendC::GetBlockIdx());
        int64_t blockOffset = tiling->block_length * blockIdx;
        int64_t remaining = tiling->total_length - blockOffset;
        blockLength_ = remaining > tiling->block_length ? tiling->block_length : remaining;
        if (blockLength_ < 0) {
            blockLength_ = 0;
        }
        tileLength_ = tiling->tile_length;
        inputGm_.SetGlobalBuffer((__gm__ T *)input + blockOffset, blockLength_);
        outputGm_.SetGlobalBuffer((__gm__ T *)output + blockOffset, blockLength_);
        int64_t loopCount = (blockLength_ + tileLength_ - 1) / tileLength_;
        useSingleTileTbuf_ = (loopCount <= 1);
        uint8_t bufNum = (loopCount >= 2) ? 2 : 1;
        int64_t bankPadBytes = 0;
        if (loopCount >= LARGE_LOOP_PAD_THRESHOLD) {
            bankPadBytes = UB_LARGE_LOOP_PAD_BYTES;
        } else if (loopCount >= 2) {
            bankPadBytes = UB_BANK_PAD_BYTES;
        }
        int64_t ubBufferBytes = tileLength_ * sizeof(T) + bankPadBytes;
        if (useSingleTileTbuf_) {
            pipe_.InitBuffer(inputBuf_, ubBufferBytes);
            pipe_.InitBuffer(outputBuf_, ubBufferBytes);
        } else {
            pipe_.InitBuffer(inputQueue_, bufNum, ubBufferBytes);
            pipe_.InitBuffer(outputQueue_, bufNum, ubBufferBytes);
        }
        pipe_.InitBuffer(tmpBuf_, ubBufferBytes);
    }
    __aicore__ inline void Process()
    {
        if (blockLength_ <= 0) {
            return;
        }

        if (useSingleTileTbuf_) {
            ProcessSingleTileTbuf();
            return;
        }

        int64_t loopCount = (blockLength_ + tileLength_ - 1) / tileLength_;
        for (int64_t i = 0; i < loopCount; ++i) {
            int64_t offset = i * tileLength_;
            int64_t count = MinInt64(tileLength_, blockLength_ - offset);
            CopyIn(offset, count);
            Compute(count);
            CopyOut(offset, count);
        }
    }

private:
    __aicore__ inline void ProcessSingleTileTbuf()
    {
        int64_t count = blockLength_;
        AscendC::LocalTensor<T> inputLocal = inputBuf_.template Get<T>();
        AscendC::LocalTensor<T> outputLocal = outputBuf_.template Get<T>();
        AscendC::LocalTensor<T> tmpLocal = tmpBuf_.template Get<T>();

        AscendC::DataCopyParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = static_cast<uint32_t>(count * sizeof(T));
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;

        AscendC::DataCopyPad(inputLocal, inputGm_[0], copyParams, {false, 0, 0, 0});
        AscendC::PipeBarrier<PIPE_ALL>();

        if constexpr (USE_X5_PATH != 0) {
            GeluComputeX5(outputLocal, inputLocal, tmpLocal, count);
        } else if constexpr (USE_STD_TANH_PATH != 0) {
            GeluComputeStdTanh(outputLocal, inputLocal, tmpLocal, count);
        } else {
            GeluCompute(outputLocal, inputLocal, tmpLocal, count);
        }
        AscendC::PipeBarrier<PIPE_ALL>();

        if (IsAlignedCopySize(count, sizeof(T))) {
            AscendC::DataCopy(outputGm_[0], outputLocal, static_cast<uint32_t>(count));
        } else {
            AscendC::DataCopyPad(outputGm_[0], outputLocal, copyParams);
        }
    }
    __aicore__ inline void CopyIn(int64_t offset, int64_t count)
    {
        AscendC::LocalTensor<T> inputLocal = inputQueue_.template AllocTensor<T>();
        AscendC::DataCopyParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = static_cast<uint32_t>(count * sizeof(T));
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        AscendC::DataCopyPad(inputLocal, inputGm_[offset], copyParams, {false, 0, 0, 0});
        inputQueue_.EnQue(inputLocal);
    }
    __aicore__ inline void CopyOut(int64_t offset, int64_t count)
    {
        AscendC::LocalTensor<T> outputLocal = outputQueue_.template DeQue<T>();
        AscendC::DataCopyParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = static_cast<uint32_t>(count * sizeof(T));
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        AscendC::DataCopyPad(outputGm_[offset], outputLocal, copyParams);
        outputQueue_.FreeTensor(outputLocal);
    }
    __aicore__ inline void GeluCompute(AscendC::LocalTensor<T> outputLocal,
                                       AscendC::LocalTensor<T> xLocal,
                                       AscendC::LocalTensor<T> tmpLocal,
                                       int64_t count)
    {
        AscendC::Mul(tmpLocal, xLocal, xLocal, static_cast<int32_t>(count));
        AscendC::Muls(outputLocal, tmpLocal, static_cast<T>(TANH_RATIONAL_C2), static_cast<int32_t>(count));
        AscendC::Adds(outputLocal, outputLocal, static_cast<T>(TANH_RATIONAL_C0), static_cast<int32_t>(count));
        AscendC::Mul(outputLocal, outputLocal, xLocal, static_cast<int32_t>(count));
        AscendC::Muls(tmpLocal, tmpLocal, static_cast<T>(TANH_RATIONAL_D2), static_cast<int32_t>(count));
        AscendC::Adds(tmpLocal, tmpLocal, static_cast<T>(ONE), static_cast<int32_t>(count));
        AscendC::Div(outputLocal, outputLocal, tmpLocal, static_cast<int32_t>(count));
        AscendC::Tanh(outputLocal, outputLocal, static_cast<int32_t>(count));
        AscendC::Muls(outputLocal, outputLocal, static_cast<T>(HALF), static_cast<int32_t>(count));
        AscendC::Adds(outputLocal, outputLocal, static_cast<T>(HALF), static_cast<int32_t>(count));
        AscendC::Mul(outputLocal, xLocal, outputLocal, static_cast<int32_t>(count));
    }
    __aicore__ inline void GeluComputeStdTanh(AscendC::LocalTensor<T> outputLocal,
                                              AscendC::LocalTensor<T> xLocal,
                                              AscendC::LocalTensor<T> tmpLocal,
                                              int64_t count)
    {
        AscendC::Mul(tmpLocal, xLocal, xLocal, static_cast<int32_t>(count));
        AscendC::Mul(outputLocal, tmpLocal, xLocal, static_cast<int32_t>(count));
        AscendC::Muls(outputLocal, outputLocal, static_cast<T>(LOGISTIC_C3), static_cast<int32_t>(count));
        AscendC::Add(outputLocal, outputLocal, xLocal, static_cast<int32_t>(count));
        AscendC::Muls(outputLocal, outputLocal, static_cast<T>(LOGISTIC_NEG_SCALE), static_cast<int32_t>(count));
        AscendC::Exp(outputLocal, outputLocal, static_cast<int32_t>(count));
        AscendC::Adds(outputLocal, outputLocal, static_cast<T>(ONE), static_cast<int32_t>(count));
        AscendC::Div(outputLocal, xLocal, outputLocal, static_cast<int32_t>(count));
    }
    __aicore__ inline void GeluComputeX5(AscendC::LocalTensor<T> outputLocal,
                                         AscendC::LocalTensor<T> xLocal,
                                         AscendC::LocalTensor<T> tmpLocal,
                                         int64_t count)
    {
        AscendC::Mul(tmpLocal, xLocal, xLocal, static_cast<int32_t>(count));
        AscendC::Mul(outputLocal, tmpLocal, xLocal, static_cast<int32_t>(count));
        AscendC::Mul(tmpLocal, outputLocal, tmpLocal, static_cast<int32_t>(count));
        AscendC::Muls(outputLocal, outputLocal, static_cast<T>(TANH_X5_C3), static_cast<int32_t>(count));
        AscendC::Muls(tmpLocal, tmpLocal, static_cast<T>(TANH_X5_C5), static_cast<int32_t>(count));
        AscendC::Add(outputLocal, outputLocal, tmpLocal, static_cast<int32_t>(count));
        AscendC::Add(outputLocal, outputLocal, xLocal, static_cast<int32_t>(count));
        AscendC::Muls(outputLocal, outputLocal, static_cast<T>(TANH_X5_C0), static_cast<int32_t>(count));
        AscendC::Tanh(outputLocal, outputLocal, static_cast<int32_t>(count));
        AscendC::Muls(outputLocal, outputLocal, static_cast<T>(HALF), static_cast<int32_t>(count));
        AscendC::Adds(outputLocal, outputLocal, static_cast<T>(HALF), static_cast<int32_t>(count));
        AscendC::Mul(outputLocal, xLocal, outputLocal, static_cast<int32_t>(count));
    }
    __aicore__ inline void Compute(int64_t count)
    {
        AscendC::LocalTensor<T> inputLocal = inputQueue_.template DeQue<T>();
        AscendC::LocalTensor<T> outputLocal = outputQueue_.template AllocTensor<T>();

        AscendC::LocalTensor<T> tmpLocal = tmpBuf_.template Get<T>();
        if constexpr (USE_X5_PATH != 0) {
            GeluComputeX5(outputLocal, inputLocal, tmpLocal, count);
        } else if constexpr (USE_STD_TANH_PATH != 0) {
            GeluComputeStdTanh(outputLocal, inputLocal, tmpLocal, count);
        } else {
            GeluCompute(outputLocal, inputLocal, tmpLocal, count);
        }
        outputQueue_.template EnQue<T>(outputLocal);
        inputQueue_.FreeTensor(inputLocal);
    }

private:
    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> inputQueue_;
    AscendC::TQue<AscendC::QuePosition::GELU_JOIN3(VE, C, OUT), 1> outputQueue_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> inputBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> outputBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf_;
    AscendC::GlobalTensor<T> inputGm_;
    AscendC::GlobalTensor<T> outputGm_;
    int64_t blockLength_ = 0;
    int64_t tileLength_ = 0;
    bool useSingleTileTbuf_ = false;
};
template <typename DT_INPUT_X, int USE_X5_PATH, int USE_STD_TANH_PATH>
__global__ __aicore__ void gelu(GM_ADDR input, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(GeluTilingData);
    GET_TILING_DATA_WITH_STRUCT(GeluTilingData, tilingData, tiling);
    KernelGelu<DT_INPUT_X, USE_X5_PATH, USE_STD_TANH_PATH> op;
    op.Init(input, output, &tilingData);
    op.Process();
}
