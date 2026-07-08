// Tiling结构体定义的头文件。
//
#pragma once

#include <cstdint>

struct GeluTilingData {
    // 整个输入tensor展平后的元素总数。
    // GELU是逐元素算子，所以不需要保留原始多维shape，只需要知道总共有多少元素。
    int64_t total_length;

    // 每个AI Core最多处理多少元素。
    // 例如total_length=10000，block_length=2048，则第0个Core处理[0, 2048)，
    // 第1个Core处理[2048, 4096)，依此类推，最后一个Core可能不足block_length。
    int64_t block_length;

    // 每个AI Core每一轮搬进UB计算多少元素。
    // block_length可能比UB能容纳的元素更多，因此Kernel会按tile_length循环处理。
    int64_t tile_length;
};
