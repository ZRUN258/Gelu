#pragma once

#include <cstdint>

struct GeluTilingData {
    int64_t total_elems;
    int64_t core_elems;
    int64_t tile_elems;
};
