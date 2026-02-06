#pragma once

#include <cstdint>

#include <Metal/Metal.hpp>

namespace engine {

// For each output row: left row index and right row index.
// Note: indices are into the *base* column buffers (after selection remap).
struct JoinMapGPU {
    MTL::Buffer* leftRow = nullptr;
    MTL::Buffer* rightRow = nullptr;
    uint32_t count = 0;
};

} // namespace engine
