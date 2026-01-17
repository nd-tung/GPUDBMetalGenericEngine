#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_set>
#include <vector>

namespace engine {

// Minimal CPU-visible table container for hybrid/GPU outputs.
// v0 supports u32 and f32 columns only.
struct TableResult {
    struct ColRef {
        enum class Kind { U32, F32, String } kind;
        std::size_t index = 0;
        std::string name;
    };

    std::vector<std::string> u32_names;
    std::vector<std::vector<uint32_t>> u32_cols;

    std::vector<std::string> f32_names;
    std::vector<std::vector<float>> f32_cols;

    std::vector<std::string> string_names;
    std::vector<std::vector<std::string>> string_cols;

    // Explicit output column order (can interleave u32/f32). When empty, callers may
    // fall back to u32_names followed by f32_names.
    std::vector<ColRef> order;

    // Column names that store single-char strings (should be decoded as char on output)
    std::unordered_set<std::string> singleCharCols;

    std::size_t rowCount = 0;

    double upload_ms = 0.0;
    double gpu_ms = 0.0;
    double cpu_post_ms = 0.0;
};

} // namespace engine
