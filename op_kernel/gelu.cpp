// Kernel侧核函数实现 (优化版: 双缓冲 + UB感知分块)
#include "kernel_operator.h"
#include <type_traits>

#include "gelu_tiling.h"
#include "tiling_key_gelu.h"

using namespace AscendC;

template <class DT_INPUT_X>
class KernelGelu {
public:
    __aicore__ inline KernelGelu() {}
    __aicore__ inline void Init(GM_ADDR input_x, GM_ADDR output, const GeluTilingData& tiling) {
        totalLength = tiling.length;

        uint32_t blockNum = GetBlockNum();
        uint32_t blockIdx = GetBlockIdx();
        uint32_t alignElements = 32 / sizeof(DT_INPUT_X);

        uint32_t avgLength = AlignUp((totalLength + blockNum - 1) / blockNum, alignElements);
        startOffset = avgLength * blockIdx;
        coreLength = (startOffset < totalLength) ? totalLength - startOffset : 0;
        if (coreLength > avgLength) coreLength = avgLength;

        inputGm.SetGlobalBuffer((__gm__ DT_INPUT_X*)input_x + startOffset, coreLength);
        outputGm.SetGlobalBuffer((__gm__ DT_INPUT_X*)output + startOffset, coreLength);

        // UB感知分块大小
        // 双缓冲: 2*TILE input + 2*TILE output + 2*TILE float calc = perElem * TILE
        uint32_t perElemUb = 2 * sizeof(DT_INPUT_X) + 2 * sizeof(DT_INPUT_X) + 2 * sizeof(float);
        // 使用 85% UB 作为安全余量
        uint32_t safeUb = tiling.ubSize * 85 / 100;
        tileLength = safeUb / perElemUb;
        tileLength = AlignUp(tileLength, alignElements);
        if (tileLength < 256) tileLength = 256;
        if (tileLength > 65536) tileLength = 65536;

        pipe.InitBuffer(inputQueue, BUFFER_NUM, tileLength * sizeof(DT_INPUT_X));
        pipe.InitBuffer(outputQueue, BUFFER_NUM, tileLength * sizeof(DT_INPUT_X));
        pipe.InitBuffer(calcBuf, tileLength * sizeof(float) * FLOAT_BUF_NUM);
    }

    __aicore__ inline void Process() {
        if (coreLength == 0) return;

        uint32_t alignElements = 32 / sizeof(DT_INPUT_X);

        // === 预取第一个分块到 inputQueue[0] ===
        uint32_t firstLen = (coreLength > tileLength) ? tileLength : coreLength;
        uint32_t firstAligned = AlignUp(firstLen, alignElements);
        {
            LocalTensor<DT_INPUT_X> inLocal = inputQueue.AllocTensor<DT_INPUT_X>();
            CopyIn(inLocal, inputGm[0], firstLen, firstAligned);
            inputQueue.EnQue(inLocal);
        }

        // === 双缓冲主循环 ===
        // 流水线: DMA_In(tile N+1) || Compute(tile N) || DMA_Out(tile N-1)
        for (uint32_t offset = 0; offset < coreLength; offset += tileLength) {
            uint32_t calcLength = coreLength - offset;
            if (calcLength > tileLength) calcLength = tileLength;
            uint32_t alignedLength = AlignUp(calcLength, alignElements);

            // 等待当前输入 DMA 完成并取出
            LocalTensor<DT_INPUT_X> inLocal = inputQueue.DeQue<DT_INPUT_X>();

            // 异步预取下一分块 (与当前计算重叠)
            if (offset + tileLength < coreLength) {
                uint32_t nextLen = coreLength - offset - tileLength;
                if (nextLen > tileLength) nextLen = tileLength;
                uint32_t nextAligned = AlignUp(nextLen, alignElements);
                LocalTensor<DT_INPUT_X> nextIn = inputQueue.AllocTensor<DT_INPUT_X>();
                CopyIn(nextIn, inputGm[offset + tileLength], nextLen, nextAligned);
                inputQueue.EnQue(nextIn);
            }

            // === 计算 GELU ===
            LocalTensor<DT_INPUT_X> outLocal = outputQueue.AllocTensor<DT_INPUT_X>();
            ComputeGelu(inLocal, outLocal, alignedLength);

            outputQueue.EnQue(outLocal);
            inputQueue.FreeTensor(inLocal);

            // 写出上一个分块的结果 (与下一次计算重叠)
            if (offset > 0) {
                LocalTensor<DT_INPUT_X> prevOut = outputQueue.DeQue<DT_INPUT_X>();
                uint32_t prevOffset = offset - tileLength;
                uint32_t prevLen = coreLength - prevOffset;
                if (prevLen > tileLength) prevLen = tileLength;
                CopyOut(outputGm[prevOffset], prevOut, prevLen);
                outputQueue.FreeTensor(prevOut);
            }
        }

        // === 写出最后一个分块 ===
        uint32_t lastOffset = ((coreLength - 1) / tileLength) * tileLength;
        LocalTensor<DT_INPUT_X> lastOut = outputQueue.DeQue<DT_INPUT_X>();
        uint32_t lastLen = coreLength - lastOffset;
        CopyOut(outputGm[lastOffset], lastOut, lastLen);
        outputQueue.FreeTensor(lastOut);
    }

private:
    __aicore__ inline uint32_t AlignUp(uint32_t value, uint32_t align) {
        return (value + align - 1) / align * align;
    }

    // DMA 搬入 (带 padding 处理非对齐)
    __aicore__ inline void CopyIn(LocalTensor<DT_INPUT_X>& dst,
                                  const GlobalTensor<DT_INPUT_X>& src,
                                  uint32_t validLen, uint32_t alignedLen) {
        DataCopyExtParams params{1, static_cast<uint32_t>(validLen * sizeof(DT_INPUT_X)), 0, 0, 0};
        if (validLen == alignedLen) {
            DataCopy(dst, src, params);
        } else {
            DataCopyPadExtParams<DT_INPUT_X> padParams{true, 0,
                static_cast<uint8_t>(alignedLen - validLen), 0};
            DataCopyPad(dst, src, params, padParams);
        }
    }

    // DMA 搬出 (精确长度, 不 padding)
    __aicore__ inline void CopyOut(GlobalTensor<DT_INPUT_X>& dst,
                                   LocalTensor<DT_INPUT_X>& src,
                                   uint32_t validLen) {
        DataCopyExtParams params{1, static_cast<uint32_t>(validLen * sizeof(DT_INPUT_X)), 0, 0, 0};
        DataCopy(dst, src, params);
    }

    // GELU 核心计算: x * 0.5 * (1 + erf(x / sqrt(2)))
    __aicore__ inline void ComputeGelu(const LocalTensor<DT_INPUT_X>& input,
                                       LocalTensor<DT_INPUT_X>& output,
                                       uint32_t count) {
        constexpr float INV_SQRT2 = 0.70710678118654752440f;
        constexpr float ONE = 1.0f;
        constexpr float HALF = 0.5f;

        LocalTensor<float> xFloat = calcBuf.Get<float>();
        LocalTensor<float> erfFloat = xFloat[tileLength];

        // 转 float32 计算, 保证精度
        if constexpr (std::is_same<DT_INPUT_X, float>::value) {
            Adds(xFloat, input, 0.0f, count);
        } else {
            Cast(xFloat, input, RoundMode::CAST_NONE, count);
        }

        // GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
        Muls(erfFloat, xFloat, INV_SQRT2, count);  // erfFloat = x / sqrt(2)
        Erf(erfFloat, erfFloat, count);             // erfFloat = erf(x / sqrt(2))
        Adds(erfFloat, erfFloat, ONE, count);       // erfFloat = 1 + erf(...)
        Mul(xFloat, xFloat, erfFloat, count);       // xFloat = x * (1 + erf(...))
        Muls(xFloat, xFloat, HALF, count);          // xFloat = x * 0.5 * (1 + erf(...))

        // 转回原始类型
        if constexpr (std::is_same<DT_INPUT_X, float>::value) {
            Adds(output, xFloat, 0.0f, count);
        } else {
            Cast(output, xFloat, RoundMode::CAST_NONE, count);
        }
    }

    static constexpr uint32_t BUFFER_NUM = 2;
    static constexpr uint32_t FLOAT_BUF_NUM = 2;

    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueue;
    TBuf<QuePosition::VECCALC> calcBuf;
    GlobalTensor<DT_INPUT_X> inputGm;
    GlobalTensor<DT_INPUT_X> outputGm;
    uint32_t totalLength;
    uint32_t startOffset;
    uint32_t coreLength;
    uint32_t tileLength;
};

template <typename DT_INPUT_X>
__global__ __aicore__ void gelu(GM_ADDR input_x, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(GeluTilingData);
    GET_TILING_DATA_WITH_STRUCT(GeluTilingData, tiling_data, tiling);
    KernelGelu<DT_INPUT_X> op;
    op.Init(input_x, output, tiling_data);
    op.Process();
}
