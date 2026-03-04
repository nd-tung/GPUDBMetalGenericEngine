#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>

#include <Metal/Metal.hpp>

namespace engine {

// GPU relation abstraction: typed columns + rowCount.
// Data is materialized (no selection vector).
struct GpuRelation {
    uint32_t rowCount = 0;

    // Columns stored as MTL::Buffer* in shared memory.
    // Lifetime is owned by GpuRelation (release in destructor).
    std::unordered_map<std::string, MTL::Buffer*> u32cols;
    std::unordered_map<std::string, MTL::Buffer*> f32cols;

    GpuRelation() = default;
    GpuRelation(const GpuRelation&) = delete;
    GpuRelation& operator=(const GpuRelation&) = delete;

    GpuRelation(GpuRelation&& other) noexcept {
        rowCount = other.rowCount;
        u32cols = std::move(other.u32cols);
        f32cols = std::move(other.f32cols);
        other.rowCount = 0;
        other.u32cols.clear();
        other.f32cols.clear();
    }

    GpuRelation& operator=(GpuRelation&& other) noexcept {
        if (this == &other) return *this;
        releaseAll();
        rowCount = other.rowCount;
        u32cols = std::move(other.u32cols);
        f32cols = std::move(other.f32cols);
        other.rowCount = 0;
        other.u32cols.clear();
        other.f32cols.clear();
        return *this;
    }

    ~GpuRelation() { releaseAll(); }

    void releaseAll() {
        for (auto& [_, b] : u32cols) if (b) b->release();
        for (auto& [_, b] : f32cols) if (b) b->release();
        u32cols.clear();
        f32cols.clear();
        rowCount = 0;
    }


};

} // namespace engine
