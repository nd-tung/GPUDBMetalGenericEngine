#pragma once
// EvalContext: per-operator execution context holding column data (CPU + GPU).
// Split from GpuExecutorDetail.hpp for focused compilation.

#include "EngineConfig.hpp"
#include "FlatStringCol.hpp"
#include "DictEncoded.hpp"
#include "GpuBuffer.hpp"
#include "Operators.hpp"   // GpuOps

#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <string>
#include <cstring>

namespace engine {

// Move EvalContext definition here so it can be shared across translation units.
// NOTE: EvalContext uses implicit (default) copy/move for raw MTL::Buffer* in u32/f32 maps.
// FlatStringCol/DictEncoded maps handle their own retain/release via their RAII semantics.
// Call releaseGPU() manually when an EvalContext is no longer needed.
struct EvalContext {
    // Column data keyed by column name
    std::unordered_map<std::string, std::vector<uint32_t>> u32Cols;
    std::unordered_map<std::string, std::vector<float>> f32Cols;
    
    // GPU storage - Metal buffers (RAII — auto-retains on copy, auto-releases on destroy)
    std::unordered_map<std::string, GpuBuffer> u32ColsGPU;
    std::unordered_map<std::string, GpuBuffer> f32ColsGPU;
    
    // Raw string columns for pattern matching (LIKE, CONTAINS)
    // NOTE: With GPU-native dictionary encoding, stringCols is now a LAZY CACHE.
    // Primary string data lives in dictCols. stringCols is populated on-demand
    // when pattern matching needs raw strings (LIKE, CONTAINS) or at final output.
    std::unordered_map<std::string, std::vector<std::string>> stringCols;

    // Pre-flattened string columns (Arrow-style GPU buffers, created at load time)
    // Uses standalone FlatStringCol struct above.
    // NOTE: Built lazily from dictCols when GPU string pattern matching is needed.
    std::unordered_map<std::string, FlatStringCol> flatStringCols;
    
    // Dictionary-encoded string columns — PRIMARY string representation.
    // Dict IDs are GPU-resident u32 values that propagate through the pipeline
    // like normal u32 columns (compact, gather, join, groupby all work on IDs).
    // Strings are only materialized from dict at output time.
    std::unordered_map<std::string, DictEncoded> dictCols;
    
    // Column aliases: maps alias -> canonical name
    // e.g., "supplier_no" -> "l_suppkey" for CTE aliasing
    std::unordered_map<std::string, std::string> columnAliases;
    
    // Active row indices (selection vector)
    std::vector<uint32_t> activeRows;
    
    // GPU selection vector (RAII — auto-retains on copy, auto-releases on destroy)
    GpuBuffer activeRowsGPU;
    uint32_t activeRowsCountGPU = 0;
    
    // Row count
    size_t rowCount = 0;

    // Lazy sync: download activeRowsGPU to CPU activeRows on demand.
    // Call this before any code path that reads activeRows (the CPU vector).
    void ensureActiveRowsCPU() {
        if (activeRowsGPU && activeRows.size() != activeRowsCountGPU) {
            activeRows.resize(activeRowsCountGPU);
            if (activeRowsCountGPU > 0) {
                std::memcpy(activeRows.data(), activeRowsGPU->contents(),
                            activeRowsCountGPU * sizeof(uint32_t));
            }
        }
    }

    // Flag to indicate if this context represents a scalar result (even if broadcasted)
    bool isScalarResult = false;

    // ========== Column resolution helper ==========
    // Resolve a column name to the actual key present in this context.
    // Searches u32ColsGPU, f32ColsGPU, u32Cols, f32Cols, dictCols, stringCols, flatStringCols.
    // Tries: 1) exact, 2) suffix _1.._9, 3) _rhs_ prefix, 4) columnAliases.
    // Returns the resolved key, or empty string if not found.
    std::string resolveColName(const std::string& name) const {
        // Check if col name exists in any map
        auto inAnyMap = [this](const std::string& n) -> bool {
            return u32ColsGPU.count(n) || f32ColsGPU.count(n)
                || u32Cols.count(n)    || f32Cols.count(n)
                || dictCols.count(n)   || stringCols.count(n)
                || flatStringCols.count(n);
        };
        // 1. Exact match
        if (inAnyMap(name)) return name;
        // 2. Suffixed _1 through _9
        for (int i = 1; i <= engine::config::kMaxColumnSuffixSearch; ++i) {
            std::string s = name + "_" + std::to_string(i);
            if (inAnyMap(s)) return s;
        }
        // 3. _rhs_ prefix match (e.g., name_rhs_7)
        std::string rhsPfx = name + "_rhs_";
        for (const auto& [k, _] : u32ColsGPU) if (k.rfind(rhsPfx, 0) == 0) return k;
        for (const auto& [k, _] : f32ColsGPU) if (k.rfind(rhsPfx, 0) == 0) return k;
        for (const auto& [k, _] : u32Cols)    if (k.rfind(rhsPfx, 0) == 0) return k;
        for (const auto& [k, _] : f32Cols)    if (k.rfind(rhsPfx, 0) == 0) return k;
        for (const auto& [k, _] : dictCols)   if (k.rfind(rhsPfx, 0) == 0) return k;
        for (const auto& [k, _] : stringCols) if (k.rfind(rhsPfx, 0) == 0) return k;
        // 4. Column alias
        auto aliasIt = columnAliases.find(name);
        if (aliasIt != columnAliases.end() && inAnyMap(aliasIt->second))
            return aliasIt->second;
        return "";
    }

    // Check if a column exists (exact, suffixed, or aliased) in GPU u32 or f32 maps only.
    std::string resolveGpuColName(const std::string& name) const {
        auto inGpu = [this](const std::string& n) -> bool {
            return u32ColsGPU.count(n) || f32ColsGPU.count(n);
        };
        if (inGpu(name)) return name;
        for (int i = 1; i <= engine::config::kMaxColumnSuffixSearch; ++i) {
            std::string s = name + "_" + std::to_string(i);
            if (inGpu(s)) return s;
        }
        std::string rhsPfx = name + "_rhs_";
        for (const auto& [k, _] : u32ColsGPU) if (k.rfind(rhsPfx, 0) == 0) return k;
        for (const auto& [k, _] : f32ColsGPU) if (k.rfind(rhsPfx, 0) == 0) return k;
        auto aliasIt = columnAliases.find(name);
        if (aliasIt != columnAliases.end() && inGpu(aliasIt->second))
            return aliasIt->second;
        return "";
    }

    // ========== GPU buffer lifecycle helpers ==========
    // All GPU buffer maps use GpuBuffer RAII (retain-on-copy, release-on-destroy).
    // FlatStringCol/DictEncoded maps use their own RAII.
    // Call releaseGPU() explicitly for cleanup when an EvalContext is no longer needed.

    void collectRawGPUBuffers(std::unordered_set<MTL::Buffer*>& out) const {
        for (const auto& [_, buf] : u32ColsGPU) if (buf) out.insert(buf.get());
        for (const auto& [_, buf] : f32ColsGPU) if (buf) out.insert(buf.get());
        if (activeRowsGPU) out.insert(activeRowsGPU.get());
    }

    // Release all GPU buffers held by this struct.
    // All GPU buffer maps use RAII — clearing triggers destructors.
    void releaseGPU() {
        u32ColsGPU.clear();
        f32ColsGPU.clear();
        activeRowsGPU = nullptr;
        activeRowsCountGPU = 0;
        flatStringCols.clear();
        dictCols.clear();
    }

    // Safely release and remove a single GPU buffer from the u32/f32 maps.
    void releaseU32Col(const std::string& col) {
        u32ColsGPU.erase(col);  // GpuBuffer destructor releases
    }
    void releaseF32Col(const std::string& col) {
        f32ColsGPU.erase(col);  // GpuBuffer destructor releases
    }

    // ========== End RAII ==========

    // Ensure stringCols[colName] is populated from dictCols (lazy materialization).
    // Call before any code path that needs raw string data (LIKE, CONTAINS).
    void ensureStringCol(const std::string& colName) {
        if (stringCols.count(colName) && !stringCols[colName].empty()) return;
        auto dit = dictCols.find(colName);
        if (dit != dictCols.end() && dit->second.valid()) {
            stringCols[colName] = dit->second.materialize();
        }
    }

    // Ensure flatStringCols[colName] is built from stringCols (lazy).
    // Needs forward-declared flattenStringCol — implemented externally.
    // Callers should check flatStringCols.count(colName) first if possible.

    // Check if a column has dictionary encoding available
    bool hasDictCol(const std::string& colName) const {
        auto it = dictCols.find(colName);
        return it != dictCols.end() && it->second.valid();
    }

    // Compact dictCols using activeRowsGPU (GPU gather of dict IDs)
    void compactDictCols(uint32_t compactCount) {
        for (auto& [name, dict] : dictCols) {
            if (dict.idsGPU) {
                uint32_t bufRows = (uint32_t)(dict.idsGPU->length() / sizeof(uint32_t));
                if (bufRows > compactCount) {
                    MTL::Buffer* compacted = GpuOps::gatherU32(dict.idsGPU, activeRowsGPU, compactCount, true);
                    if (compacted) {
                        dict.idsGPU.reset(compacted);
                        dict.rowCount = compactCount;
                        dict.ids.clear();  // Invalidate CPU mirror (lazy sync)
                    }
                }
            }
        }
    }

    // Compact dictCols using an explicit index buffer (GPU gather of dict IDs)
    void compactDictCols(MTL::Buffer* indexBuf, uint32_t newCount) {
        for (auto& [name, dict] : dictCols) {
            if (dict.idsGPU) {
                MTL::Buffer* gathered = GpuOps::gatherU32(dict.idsGPU, indexBuf, newCount, false);
                if (gathered) {
                    dict.idsGPU.reset(gathered);
                    dict.rowCount = newCount;
                    dict.ids.clear();
                }
            }
        }
    }

    // Compact flatStringCols using activeRowsGPU (GPU gather of chars/offsets/lengths)
    void compactFlatStringCols(uint32_t compactCount) {
        for (auto& [name, flat] : flatStringCols) {
            if (flat.chars && flat.offsets && flat.lengths && flat.rowCount > compactCount) {
                auto r = GpuOps::gatherFlatString(
                    flat.chars, flat.offsets, flat.lengths,
                    activeRowsGPU, compactCount, true);
                if (r.chars) {
                    flat.takeFrom(r.chars, r.offsets, r.lengths, r.rowCount, r.totalBytes);
                }
            }
        }
    }

    // Compact flatStringCols using an explicit index buffer
    void compactFlatStringCols(MTL::Buffer* indexBuf, uint32_t newCount) {
        for (auto& [name, flat] : flatStringCols) {
            if (flat.chars && flat.offsets && flat.lengths) {
                auto r = GpuOps::gatherFlatString(
                    flat.chars, flat.offsets, flat.lengths,
                    indexBuf, newCount, true);
                if (r.chars) {
                    flat.takeFrom(r.chars, r.offsets, r.lengths, r.rowCount, r.totalBytes);
                }
            }
        }
    }

    // Ensure flatStringCols[colName] is built from dictCols or stringCols (lazy).
    // Implementation uses flattenStringCol() free function (declared below struct).
    void ensureFlatStringCol(const std::string& colName);

    // Safely erase a flat string column (RAII destructor releases GPU buffers)
    void eraseFlatStringCol(const std::string& colName) {
        flatStringCols.erase(colName);
    }

    // Gather all GPU-side columns (u32, f32, dict, flat string) by index array.
    // Releases old GPU buffers and replaces with gathered versions.
    void gatherAllGPU(MTL::Buffer* indices, uint32_t count) {
        for (auto& [name, buf] : u32ColsGPU) {
            if (!buf) continue;
            MTL::Buffer* gathered = GpuOps::gatherU32(buf, indices, count);
            buf.reset(gathered);
        }
        for (auto& [name, buf] : f32ColsGPU) {
            if (!buf) continue;
            MTL::Buffer* gathered = GpuOps::gatherF32(buf, indices, count);
            buf.reset(gathered);
        }
        compactDictCols(indices, count);
        compactFlatStringCols(indices, count);
    }

    // Invalidate stringCols entries that have dict or flat-string equivalents.
    void invalidateStringColsForDictFlat() {
        for (const auto& [name, dc] : dictCols)
            stringCols.erase(name);
        for (const auto& [name, fc] : flatStringCols)
            stringCols.erase(name);
    }

    // Reset active rows tracking (releases activeRowsGPU if set).
    void clearActiveRows() {
        activeRows.clear();
        activeRowsGPU = nullptr; // GpuBuffer RAII releases automatically
        activeRowsCountGPU = 0;
    }

    // Which table is "current" for column lookups
    std::string currentTable;
    
    // Columns originating from DELIM_SCAN (correlation keys)
    // These should be prioritized during join column name collision resolution
    std::unordered_set<std::string> isDelimCorrelation;

    // Sequential counter for positional aggregate output columns (#0, #1, ...)
    size_t aggregateCounter = 0;
};

} // namespace engine
