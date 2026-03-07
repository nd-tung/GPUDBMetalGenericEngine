#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>

#include "GpuBuffer.hpp"

namespace engine {

// GPU relation abstraction: typed columns + rowCount.
// Data is materialized (no selection vector).
// Uses GpuBuffer RAII — compiler-generated move/dtor handle retain/release.
struct GpuRelation {
    uint32_t rowCount = 0;

    // Columns stored as GpuBuffer (shared memory, RAII lifetime).
    std::unordered_map<std::string, GpuBuffer> u32cols;
    std::unordered_map<std::string, GpuBuffer> f32cols;

    GpuRelation() = default;
    GpuRelation(const GpuRelation&) = delete;
    GpuRelation& operator=(const GpuRelation&) = delete;
    GpuRelation(GpuRelation&&) noexcept = default;
    GpuRelation& operator=(GpuRelation&&) noexcept = default;
    ~GpuRelation() = default;
};

} // namespace engine
