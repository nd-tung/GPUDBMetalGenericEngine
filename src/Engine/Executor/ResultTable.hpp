#pragma once

#include "GpuBuffer.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_set>
#include <vector>

namespace engine {

// CPU-visible table container for hybrid/GPU outputs.
// Supports u32 and f32 columns.
struct TableResult {
    struct ColRef {
        enum class Kind { U32, F32, String } kind;
        std::size_t index = 0;
        std::string name;
    };

    std::vector<std::string> u32Names;
    std::vector<std::vector<uint32_t>> u32Cols;

    std::vector<std::string> f32Names;
    std::vector<std::vector<float>> f32Cols;

    std::vector<std::string> stringNames;
    std::vector<std::vector<std::string>> stringCols;

    // Explicit output column order (can interleave u32/f32). When empty, callers may
    // fall back to u32Names followed by f32Names.
    std::vector<ColRef> order;

    // Column names that store single-char strings (should be decoded as char on output)
    std::unordered_set<std::string> singleCharCols;

    std::size_t rowCount = 0;

    // GPU buffer mirrors (RAII — auto-retains on copy, auto-releases on destroy).
    // Indexed in parallel with u32Cols/f32Cols (same order).
    std::vector<GpuBuffer> u32ColsGPU;
    std::vector<GpuBuffer> f32ColsGPU;

    double uploadMs = 0.0;
    double gpuMs = 0.0;
    double cpuPostMs = 0.0;
};

} // namespace engine
