#pragma once
// Pre-flattened Arrow-style GPU string column buffers.
// Uses GpuBuffer RAII — compiler-generated copy/move/dtor handle retain/release.

#include "GpuBuffer.hpp"
#include <Metal/Metal.hpp>
#include <cstdint>

namespace engine {

struct FlatStringCol {
    GpuBuffer chars;    // raw character bytes
    GpuBuffer offsets;  // uint32_t[rowCount] start offset per string
    GpuBuffer lengths;  // uint32_t[rowCount] length per string
    uint32_t rowCount   = 0;
    uint32_t totalBytes = 0;

    // Assign from a FlatStringGatherResult (takes ownership, releases old)
    void takeFrom(GpuBuffer c, GpuBuffer off, GpuBuffer len,
                  uint32_t rc, uint32_t tb) {
        chars = std::move(c); offsets = std::move(off); lengths = std::move(len);
        rowCount = rc; totalBytes = tb;
    }

    void release() {
        chars = nullptr; offsets = nullptr; lengths = nullptr;
        rowCount = 0;
        totalBytes = 0;
    }
};

} // namespace engine
