#pragma once
// ============================================================================
// Configuration constants (previously magic numbers scattered across the codebase)
// ============================================================================
#include <cstdint>
#include <cstddef>

namespace engine::config {
    // Maximum number of suffix variations to try when resolving column names
    // (e.g., col_1, col_2, ... col_N for multi-instance table columns)
    constexpr int kMaxColumnSuffixSearch = 9;

    // Threshold row count below which a date value is treated as days-since-epoch
    // rather than YYYYMMDD format
    constexpr uint32_t kDateFormatThreshold = 100000;

    // Sample size for detecting whether a column has varying values
    constexpr size_t kColumnSampleSize = 100;

    // Thread spawn threshold for parallel CPU string gather in joins
    constexpr uint32_t kParallelStringGatherThreshold = 10000;

    // Maximum keys and aggregates per group in GPU hash table layout
    constexpr uint32_t kMaxGroupByKeys = 8;
    constexpr uint32_t kMaxGroupByAggs = 16;

    // GPU block sort threshold — elements <= this use shared-memory bitonic sort
    constexpr uint32_t kBlockSortThreshold = 1024;
} // namespace engine::config
