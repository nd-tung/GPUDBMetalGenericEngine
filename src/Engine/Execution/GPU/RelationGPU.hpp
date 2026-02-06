#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>

#include <Metal/Metal.hpp>

#include "JoinMapGPU.hpp"

namespace engine {

// Minimal GPU relation abstraction: typed columns + rowCount.
// v0 keeps data materialized (no selection vector) to keep the refactor small.
struct RelationGPU {
    uint32_t rowCount = 0;

    // Optional selection vector into the base columns.
    // When present, it contains `rowCount` u32 indices (into the base column buffers).
    // This enables lazy filtering without immediately gathering every column.
    MTL::Buffer* indices = nullptr;

    // Columns stored as MTL::Buffer* in shared memory.
    // Lifetime is owned by RelationGPU (release in destructor).
    std::unordered_map<std::string, MTL::Buffer*> u32cols;
    std::unordered_map<std::string, MTL::Buffer*> f32cols;

    struct JoinMapping {
        std::unique_ptr<RelationGPU> left;
        std::unique_ptr<RelationGPU> right;
        JoinMapGPU map;
    };
    std::unique_ptr<JoinMapping> join;

    RelationGPU() = default;
    RelationGPU(const RelationGPU&) = delete;
    RelationGPU& operator=(const RelationGPU&) = delete;

    RelationGPU(RelationGPU&& other) noexcept {
        rowCount = other.rowCount;
        indices = other.indices;
        u32cols = std::move(other.u32cols);
        f32cols = std::move(other.f32cols);
        join = std::move(other.join);
        other.rowCount = 0;
        other.indices = nullptr;
        other.u32cols.clear();
        other.f32cols.clear();
    }

    RelationGPU& operator=(RelationGPU&& other) noexcept {
        if (this == &other) return *this;
        releaseAll();
        rowCount = other.rowCount;
        indices = other.indices;
        u32cols = std::move(other.u32cols);
        f32cols = std::move(other.f32cols);
        join = std::move(other.join);
        other.rowCount = 0;
        other.indices = nullptr;
        other.u32cols.clear();
        other.f32cols.clear();
        return *this;
    }

    ~RelationGPU() { releaseAll(); }

    void releaseAll() {
        if (join) {
            if (join->map.leftRow) join->map.leftRow->release();
            if (join->map.rightRow) join->map.rightRow->release();
            join.reset();
        }
        if (indices) indices->release();
        indices = nullptr;
        for (auto& [_, b] : u32cols) if (b) b->release();
        for (auto& [_, b] : f32cols) if (b) b->release();
        u32cols.clear();
        f32cols.clear();
        rowCount = 0;
    }

    bool hasU32(const std::string& name) const { return u32cols.find(name) != u32cols.end(); }
    bool hasF32(const std::string& name) const { return f32cols.find(name) != f32cols.end(); }
};

} // namespace engine
