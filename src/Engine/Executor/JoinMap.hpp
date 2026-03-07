#pragma once

#include <cstdint>

#include "GpuBuffer.hpp"

namespace engine {

// For each output row: left row index and right row index.
// Note: indices are into the *base* column buffers (after selection remap).
struct GpuJoinMap {
    GpuBuffer leftRow;
    GpuBuffer rightRow;
    uint32_t count = 0;
};

} // namespace engine
