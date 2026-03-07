#include "GpuExecutor.hpp"
#include "GpuExecutorDetail.hpp"
#include "TypedExpr.hpp"
#include "Operators.hpp"
#include "GpuColumnStore.hpp"
#include <Metal/Metal.hpp>
#include <future>
#include <thread>

#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <map>
#include <numeric>

namespace engine {

// Helper: extract all equi-join key pairs from a condition expression,
// and collect non-equality conditions as residual post-join filters.
static void extractJoinKeyPairs(const TypedExprPtr& expr, 
                                std::vector<std::pair<std::string, std::string>>& keyPairs,
                                std::vector<TypedExprPtr>* residuals = nullptr) {
    if (!expr) return;
    
    if (expr->kind == TypedExpr::Kind::Compare) {
        const auto& cmp = expr->asCompare();
        if (cmp.op == CompareOp::Eq && cmp.left && cmp.right) {
            if (cmp.left->kind == TypedExpr::Kind::Column && 
                cmp.right->kind == TypedExpr::Kind::Column) {
                keyPairs.emplace_back(cmp.left->asColumn().column, 
                                      cmp.right->asColumn().column);
                return;
            }
        }
        // Non-equality compare or non-column operands → residual
        if (residuals) residuals->push_back(expr);
    } else if (expr->kind == TypedExpr::Kind::Binary) {
        const auto& bin = expr->asBinary();
        if (bin.op == BinaryOp::And) {
            extractJoinKeyPairs(bin.left, keyPairs, residuals);
            extractJoinKeyPairs(bin.right, keyPairs, residuals);
        } else {
            // Other binary ops → residual
            if (residuals) residuals->push_back(expr);
        }
    } else {
        if (residuals) residuals->push_back(expr);
    }
}


// ============================================================================
// Extracted helpers for executeJoin()
// ============================================================================

static void materializeJoinContext(EvalContext& ctx, const char* label, bool debug) {
    if (!ctx.activeRowsGPU) return;  // No filter applied — nothing to materialize
    uint32_t count = ctx.activeRowsCountGPU;
    if (debug) std::cerr << "[Exec] Join: materializing " << label 
                         << " ctx (" << count << " active rows from " << ctx.rowCount << ")\n";
    // If the filter matched 0 rows, clear everything and set rowCount=0
    if (count == 0) {
        for (auto& [name, vec] : ctx.u32Cols) vec.clear();
        for (auto& [name, vec] : ctx.f32Cols) vec.clear();
        for (auto& [name, vec] : ctx.stringCols) vec.clear();
        ctx.u32ColsGPU.clear();
        ctx.f32ColsGPU.clear();
        ctx.activeRowsGPU = nullptr;
        ctx.activeRowsCountGPU = 0;
        ctx.activeRows.clear();
        ctx.rowCount = 0;
        return;
    }

    // Compact GPU u32 columns (async — no per-column sync)
    for (auto& [name, buf] : ctx.u32ColsGPU) {
        if (!buf) continue;
        uint32_t bufRows = (uint32_t)(buf->length() / sizeof(uint32_t));
        if (bufRows > count) {
            MTL::Buffer* compacted = GpuOps::gatherU32(buf, ctx.activeRowsGPU, count, false);
            if (compacted) {
                buf.reset(compacted);
            }
        }
    }
    // Compact GPU f32 columns (async — no per-column sync)
    for (auto& [name, buf] : ctx.f32ColsGPU) {
        if (!buf) continue;
        uint32_t bufRows = (uint32_t)(buf->length() / sizeof(float));
        if (bufRows > count) {
            MTL::Buffer* compacted = GpuOps::gatherF32(buf, ctx.activeRowsGPU, count, false);
            if (compacted) {
                buf.reset(compacted);
            }
        }
    }
    // GPU-native dict compaction: gather dict IDs on GPU (async)
    for (auto& [name, dict] : ctx.dictCols) {
        if (dict.idsGPU) {
            uint32_t bufRows = (uint32_t)(dict.idsGPU->length() / sizeof(uint32_t));
            if (bufRows > count) {
                MTL::Buffer* compacted = GpuOps::gatherU32(dict.idsGPU, ctx.activeRowsGPU, count, false);
                if (compacted) {
                    dict.idsGPU.reset(compacted);
                    dict.rowCount = count;
                    dict.ids.clear();  // Invalidate CPU mirror (lazy sync)
                }
            }
        } else if (dict.ids.size() > count) {
            // CPU-only fallback (no GPU buffer) — needs sync first
            GpuOps::sync();
            uint32_t* indices2 = (uint32_t*)ctx.activeRowsGPU->contents();
            std::vector<uint32_t> c;
            c.reserve(count);
            for (uint32_t i = 0; i < count; ++i)
                c.push_back(indices2[i] < (uint32_t)dict.ids.size() ? dict.ids[indices2[i]] : 0u);
            dict.ids = std::move(c);
            dict.rowCount = count;
        }
    }
    // Sync all async GPU gathers before reading results on CPU
    GpuOps::sync();
    {
        auto& s = GpuColumnStore::instance();
        for (auto& [name, vec] : ctx.u32Cols) {
            if (vec.size() > count) {
                auto itGpu = ctx.u32ColsGPU.find(name);
                if (itGpu != ctx.u32ColsGPU.end() && itGpu->second) {
                    // GPU buffer already compacted above — sync CPU from it (zero-cost on unified mem)
                    vec.resize(count);
                    std::memcpy(vec.data(), itGpu->second->contents(), count * sizeof(uint32_t));
                } else {
                    MTL::Buffer* src = s.device()->newBuffer(vec.data(), vec.size() * sizeof(uint32_t), MTL::ResourceStorageModeShared);
                    MTL::Buffer* dst = GpuOps::gatherU32(src, ctx.activeRowsGPU, count, true);
                    vec.resize(count);
                    std::memcpy(vec.data(), dst->contents(), count * sizeof(uint32_t));
                    src->release(); dst->release();
                }
            }
        }
        // Compact CPU f32 columns via GPU gather
        for (auto& [name, vec] : ctx.f32Cols) {
            if (vec.size() > count) {
                auto itGpu = ctx.f32ColsGPU.find(name);
                if (itGpu != ctx.f32ColsGPU.end() && itGpu->second) {
                    vec.resize(count);
                    std::memcpy(vec.data(), itGpu->second->contents(), count * sizeof(float));
                } else {
                    MTL::Buffer* src = s.device()->newBuffer(vec.data(), vec.size() * sizeof(float), MTL::ResourceStorageModeShared);
                    MTL::Buffer* dst = GpuOps::gatherF32(src, ctx.activeRowsGPU, count, true);
                    vec.resize(count);
                    std::memcpy(vec.data(), dst->contents(), count * sizeof(float));
                    src->release(); dst->release();
                }
            }
        }
    }
    // Compact CPU string columns: prefer invalidation if dictCol exists, else GPU/CPU gather
    {
        for (auto& [name, vec] : ctx.stringCols) {
            if (ctx.dictCols.count(name) && ctx.dictCols[name].valid()) {
                vec.clear();  // Will be rebuilt from dict on demand
            } else if (vec.size() > count) {
                // Try GPU gather via flatStringCols
                auto fit = ctx.flatStringCols.find(name);
                if (fit != ctx.flatStringCols.end() && fit->second.chars && ctx.activeRowsGPU) {
                    auto r = GpuOps::gatherFlatString(
                        fit->second.chars, fit->second.offsets, fit->second.lengths,
                        ctx.activeRowsGPU, count, true);
                    if (r.chars) {
                        const uint32_t* offs = static_cast<const uint32_t*>(r.offsets->contents());
                        const uint32_t* lens = static_cast<const uint32_t*>(r.lengths->contents());
                        const char* ch = static_cast<const char*>(r.chars->contents());
                        vec.resize(count);
                        for (uint32_t i = 0; i < count; ++i) vec[i].assign(ch + offs[i], lens[i]);
                        // Update flatStringCols to compacted version
                        fit->second.takeFrom(r.chars, r.offsets, r.lengths,
                                             r.rowCount, r.totalBytes);
                        continue;
                    }
                }
                // CPU fallback
                uint32_t* indices = (uint32_t*)ctx.activeRowsGPU->contents();
                std::vector<std::string> c;
                c.reserve(count);
                for (uint32_t i = 0; i < count; ++i)
                    c.push_back(indices[i] < (uint32_t)vec.size() ? vec[indices[i]] : std::string());
                vec = std::move(c);
            }
        }
    }
    // Compact flatStringCols that don't have matching stringCols (handled above)
    if (ctx.activeRowsGPU) {
        for (auto& [name, flat] : ctx.flatStringCols) {
            if (!ctx.stringCols.count(name) && flat.chars && flat.rowCount > count) {
                auto r = GpuOps::gatherFlatString(flat.chars, flat.offsets, flat.lengths,
                                                   ctx.activeRowsGPU, count, true);
                if (r.chars) {
                    flat.takeFrom(r.chars, r.offsets, r.lengths,
                                  r.rowCount, r.totalBytes);
                }
            }
        }
    }
    // Clear selection vector — data is now dense
    ctx.activeRowsGPU = nullptr;
    ctx.activeRowsCountGPU = 0;
    ctx.activeRows.clear();
    ctx.rowCount = count;
}

static void appendUnmatchedLeftRows(
    EvalContext& leftCtx, EvalContext& outCtx,
    const JoinResult& jRes, uint32_t& resCount, uint32_t lCount,
    const std::unordered_map<std::string, std::string>& rightColumnMapping,
    bool debug
) {
    auto& store = GpuColumnStore::instance();
    (void)rightColumnMapping; // reserved for future use
    // Use GpuOps to find unmatched left (probe) indices via scatter→flip→compact
    auto unmatched = GpuOps::findUnmatchedIndices(jRes.probeIndices, resCount, lCount);
    uint32_t unmatchedCount = unmatched.count;

    // Download unmatched indices for string gather (CPU)
    std::vector<uint32_t> unmatchedIndices(unmatchedCount);
    if (unmatchedCount > 0) {
        std::memcpy(unmatchedIndices.data(), unmatched.indices->contents(),
                    unmatchedCount * sizeof(uint32_t));
    }
    MTL::Buffer* unmatchedBuf = unmatched.indices; // reuse for GPU gather

    if (debug) std::cerr << "[Exec] Left Join: " << unmatchedCount << " unmatched left rows to append\n";

    if (unmatchedCount > 0) {
        uint32_t totalCount = resCount + unmatchedCount;

        // Append left columns: gather unmatched rows and concatenate with matched
        for (auto& [name, buf] : outCtx.u32ColsGPU) {
            if (leftCtx.u32Cols.count(name) || leftCtx.u32ColsGPU.count(name)) {
                MTL::Buffer* leftSrc = nullptr;
                bool leftSrcAllocated = false;
                if (leftCtx.u32ColsGPU.count(name)) leftSrc = leftCtx.u32ColsGPU.at(name);
                else if (leftCtx.u32Cols.count(name) && !leftCtx.u32Cols.at(name).empty()) {
                    const auto& vec = leftCtx.u32Cols.at(name);
                    leftSrc = store.device()->newBuffer(vec.data(), vec.size() * sizeof(uint32_t), MTL::ResourceStorageModeShared);
                    leftSrcAllocated = true;
                }
                if (leftSrc) {
                    MTL::Buffer* g = GpuOps::gatherU32(leftSrc, unmatchedBuf, unmatchedCount, false);
                    MTL::Buffer* combined = store.device()->newBuffer(totalCount * sizeof(uint32_t), MTL::ResourceStorageModeShared);
                    std::memcpy(combined->contents(), buf->contents(), resCount * sizeof(uint32_t));
                    std::memcpy((uint8_t*)combined->contents() + resCount * sizeof(uint32_t),
                                g->contents(), unmatchedCount * sizeof(uint32_t));
                    if (leftSrcAllocated) leftSrc->release();
                    buf.reset(combined); g->release();
                }
            } else {
                MTL::Buffer* combined = store.device()->newBuffer(totalCount * sizeof(uint32_t), MTL::ResourceStorageModeShared);
                std::memcpy(combined->contents(), buf->contents(), resCount * sizeof(uint32_t));
                std::memset((uint8_t*)combined->contents() + resCount * sizeof(uint32_t), 0, unmatchedCount * sizeof(uint32_t));
                buf.reset(combined);
            }
        }
        for (auto& [name, buf] : outCtx.f32ColsGPU) {
            if (leftCtx.f32Cols.count(name) || leftCtx.f32ColsGPU.count(name)) {
                MTL::Buffer* leftSrc = nullptr;
                bool leftSrcAllocated = false;
                if (leftCtx.f32ColsGPU.count(name)) leftSrc = leftCtx.f32ColsGPU.at(name);
                else if (leftCtx.f32Cols.count(name) && !leftCtx.f32Cols.at(name).empty()) {
                    const auto& vec = leftCtx.f32Cols.at(name);
                    leftSrc = store.device()->newBuffer(vec.data(), vec.size() * sizeof(float), MTL::ResourceStorageModeShared);
                    leftSrcAllocated = true;
                }
                if (leftSrc) {
                    MTL::Buffer* g = GpuOps::gatherF32(leftSrc, unmatchedBuf, unmatchedCount, false);
                    MTL::Buffer* combined = store.device()->newBuffer(totalCount * sizeof(float), MTL::ResourceStorageModeShared);
                    std::memcpy(combined->contents(), buf->contents(), resCount * sizeof(float));
                    std::memcpy((uint8_t*)combined->contents() + resCount * sizeof(float),
                                g->contents(), unmatchedCount * sizeof(float));
                    if (leftSrcAllocated) leftSrc->release();
                    g->release(); buf.reset(combined);
                }
            } else {
                MTL::Buffer* combined = store.device()->newBuffer(totalCount * sizeof(float), MTL::ResourceStorageModeShared);
                std::memcpy(combined->contents(), buf->contents(), resCount * sizeof(float));
                std::memset((uint8_t*)combined->contents() + resCount * sizeof(float), 0, unmatchedCount * sizeof(float));
                buf.reset(combined);
            }
        }

        // String columns: append unmatched left values + empty for right
        for (auto& [name, vec] : outCtx.stringCols) {
            if (leftCtx.stringCols.count(name)) {
                const auto& leftVec = leftCtx.stringCols.at(name);
                for (uint32_t idx : unmatchedIndices) {
                    vec.push_back(idx < leftVec.size() ? leftVec[idx] : "");
                }
            } else {
                for (uint32_t i = 0; i < unmatchedCount; ++i) vec.push_back("");
            }
        }

        // Dict columns: GPU gather unmatched left dict IDs + append zeros for right
        for (auto& [name, dc] : outCtx.dictCols) {
            if (!dc.idsGPU) continue;
            // Check if this is a left-side column
            auto leftDictIt = leftCtx.dictCols.find(name);
            if (leftDictIt != leftCtx.dictCols.end() && leftDictIt->second.idsGPU) {
                // GPU gather unmatched left dict IDs
                MTL::Buffer* g = GpuOps::gatherU32(leftDictIt->second.idsGPU, unmatchedBuf, unmatchedCount, false);
                // Concatenate matched + unmatched
                MTL::Buffer* combined = store.device()->newBuffer(totalCount * sizeof(uint32_t), MTL::ResourceStorageModeShared);
                std::memcpy(combined->contents(), dc.idsGPU->contents(), resCount * sizeof(uint32_t));
                std::memcpy((uint8_t*)combined->contents() + resCount * sizeof(uint32_t),
                            g->contents(), unmatchedCount * sizeof(uint32_t));
                g->release();
                dc.idsGPU.reset(combined);
            } else {
                // Right-side column: pad with sentinel (0) for unmatched left rows
                MTL::Buffer* combined = store.device()->newBuffer(totalCount * sizeof(uint32_t), MTL::ResourceStorageModeShared);
                std::memcpy(combined->contents(), dc.idsGPU->contents(), resCount * sizeof(uint32_t));
                std::memset((uint8_t*)combined->contents() + resCount * sizeof(uint32_t), 0, unmatchedCount * sizeof(uint32_t));
                dc.idsGPU.reset(combined);
            }
            dc.ids.clear(); // invalidate CPU mirror
            dc.rowCount = totalCount;
            // Invalidate stale stringCols for this column
            outCtx.stringCols.erase(name);
            outCtx.flatStringCols.erase(name);
        }

        // Sync CPU-side u32/f32 cols via GPU gather or GPU buffer download
        for (auto& [name, vec] : outCtx.u32Cols) {
            if (!vec.empty()) {
                if (outCtx.u32ColsGPU.count(name)) {
                    // GPU buffer already has combined data — sync from it
                    vec.resize(totalCount);
                    std::memcpy(vec.data(), outCtx.u32ColsGPU.at(name)->contents(), totalCount * sizeof(uint32_t));
                } else if (leftCtx.u32ColsGPU.count(name) && leftCtx.u32ColsGPU.at(name)) {
                    // Prefer existing GPU buffer from left context
                    MTL::Buffer* g = GpuOps::gatherU32(leftCtx.u32ColsGPU.at(name), unmatchedBuf, unmatchedCount, false);
                    size_t oldSz = vec.size();
                    vec.resize(oldSz + unmatchedCount);
                    std::memcpy(vec.data() + oldSz, g->contents(), unmatchedCount * sizeof(uint32_t));
                    g->release();
                } else if (leftCtx.u32Cols.count(name)) {
                    const auto& leftVec = leftCtx.u32Cols.at(name);
                    MTL::Buffer* src = store.device()->newBuffer(leftVec.data(), leftVec.size() * sizeof(uint32_t), MTL::ResourceStorageModeShared);
                    MTL::Buffer* g = GpuOps::gatherU32(src, unmatchedBuf, unmatchedCount, false);
                    size_t oldSz = vec.size();
                    vec.resize(oldSz + unmatchedCount);
                    std::memcpy(vec.data() + oldSz, g->contents(), unmatchedCount * sizeof(uint32_t));
                    src->release(); g->release();
                } else {
                    vec.resize(vec.size() + unmatchedCount, 0);
                }
            }
        }
        for (auto& [name, vec] : outCtx.f32Cols) {
            if (!vec.empty()) {
                if (outCtx.f32ColsGPU.count(name)) {
                    vec.resize(totalCount);
                    std::memcpy(vec.data(), outCtx.f32ColsGPU.at(name)->contents(), totalCount * sizeof(float));
                } else if (leftCtx.f32ColsGPU.count(name) && leftCtx.f32ColsGPU.at(name)) {
                    MTL::Buffer* g = GpuOps::gatherF32(leftCtx.f32ColsGPU.at(name), unmatchedBuf, unmatchedCount, false);
                    size_t oldSz = vec.size();
                    vec.resize(oldSz + unmatchedCount);
                    std::memcpy(vec.data() + oldSz, g->contents(), unmatchedCount * sizeof(float));
                    g->release();
                } else if (leftCtx.f32Cols.count(name)) {
                    const auto& leftVec = leftCtx.f32Cols.at(name);
                    MTL::Buffer* src = store.device()->newBuffer(leftVec.data(), leftVec.size() * sizeof(float), MTL::ResourceStorageModeShared);
                    MTL::Buffer* g = GpuOps::gatherF32(src, unmatchedBuf, unmatchedCount, false);
                    size_t oldSz = vec.size();
                    vec.resize(oldSz + unmatchedCount);
                    std::memcpy(vec.data() + oldSz, g->contents(), unmatchedCount * sizeof(float));
                    src->release(); g->release();
                } else {
                    vec.resize(vec.size() + unmatchedCount, 0.0f);
                }
            }
        }

        outCtx.rowCount = totalCount;
        resCount = totalCount;
        // Dict IDs already updated with unmatched rows above
        if (debug) std::cerr << "[Exec] Left Join: total output rows = " << totalCount << "\n";
    }
}

static void appendUnmatchedRightRows(
    EvalContext& rightCtx, EvalContext& outCtx,
    const JoinResult& jRes, uint32_t& resCount, uint32_t rCount,
    const std::unordered_map<std::string, std::string>& rightColumnMapping,
    bool debug
) {
    auto& store = GpuColumnStore::instance();
    auto getRightColumnName = [&](const std::string& name) -> std::string {
        auto it = rightColumnMapping.find(name);
        if (it != rightColumnMapping.end()) return it->second;
        return name;
    };
    uint32_t matchedCount = jRes.count;
    // Use GpuOps to find unmatched right (build) indices via scatter→flip→compact
    auto unmatched = GpuOps::findUnmatchedIndices(jRes.buildIndices, matchedCount, rCount);
    uint32_t unmatchedCount = unmatched.count;

    // Download unmatched indices for string gather (CPU)
    std::vector<uint32_t> unmatchedIndices(unmatchedCount);
    if (unmatchedCount > 0) {
        std::memcpy(unmatchedIndices.data(), unmatched.indices->contents(),
                    unmatchedCount * sizeof(uint32_t));
    }
    MTL::Buffer* unmatchedBuf = unmatched.indices;

    if (debug) std::cerr << "[Exec] Right Join: " << unmatchedCount << " unmatched right rows to append\n";

    if (unmatchedCount > 0) {
        uint32_t totalCount = resCount + unmatchedCount;

        // For RIGHT columns: gather unmatched rows and append
        // For LEFT columns: extend with zeros (NULL)
        for (auto& [name, buf] : outCtx.u32ColsGPU) {
            if (rightCtx.u32Cols.count(name) || rightCtx.u32ColsGPU.count(name) ||
                rightCtx.u32Cols.count(getRightColumnName(name)) || rightCtx.u32ColsGPU.count(getRightColumnName(name))) {
                std::string srcName = name;
                for (const auto& [origName, mappedName] : rightColumnMapping) {
                    if (mappedName == name) { srcName = origName; break; }
                }
                MTL::Buffer* rightSrc = nullptr;
                bool rightSrcAllocated = false;
                if (rightCtx.u32ColsGPU.count(srcName)) rightSrc = rightCtx.u32ColsGPU.at(srcName);
                else if (rightCtx.u32Cols.count(srcName) && !rightCtx.u32Cols.at(srcName).empty()) {
                    const auto& vec = rightCtx.u32Cols.at(srcName);
                    rightSrc = store.device()->newBuffer(vec.data(), vec.size() * sizeof(uint32_t), MTL::ResourceStorageModeShared);
                    rightSrcAllocated = true;
                }
                if (rightSrc) {
                    MTL::Buffer* g = GpuOps::gatherU32(rightSrc, unmatchedBuf, unmatchedCount, false);
                    MTL::Buffer* combined = store.device()->newBuffer(totalCount * sizeof(uint32_t), MTL::ResourceStorageModeShared);
                    std::memcpy(combined->contents(), buf->contents(), resCount * sizeof(uint32_t));
                    std::memcpy((uint8_t*)combined->contents() + resCount * sizeof(uint32_t),
                                g->contents(), unmatchedCount * sizeof(uint32_t));
                    if (rightSrcAllocated) rightSrc->release();
                    buf.reset(combined); g->release();
                } else {
                    MTL::Buffer* combined = store.device()->newBuffer(totalCount * sizeof(uint32_t), MTL::ResourceStorageModeShared);
                    std::memcpy(combined->contents(), buf->contents(), resCount * sizeof(uint32_t));
                    std::memset((uint8_t*)combined->contents() + resCount * sizeof(uint32_t), 0, unmatchedCount * sizeof(uint32_t));
                    buf.reset(combined);
                }
            } else {
                MTL::Buffer* combined = store.device()->newBuffer(totalCount * sizeof(uint32_t), MTL::ResourceStorageModeShared);
                std::memcpy(combined->contents(), buf->contents(), resCount * sizeof(uint32_t));
                std::memset((uint8_t*)combined->contents() + resCount * sizeof(uint32_t), 0, unmatchedCount * sizeof(uint32_t));
                buf.reset(combined);
            }
        }
        for (auto& [name, buf] : outCtx.f32ColsGPU) {
            if (rightCtx.f32Cols.count(name) || rightCtx.f32ColsGPU.count(name)) {
                MTL::Buffer* rightSrc = nullptr;
                bool rightSrcAllocated = false;
                std::string srcName = name;
                for (const auto& [origName, mappedName] : rightColumnMapping) {
                    if (mappedName == name) { srcName = origName; break; }
                }
                if (rightCtx.f32ColsGPU.count(srcName)) rightSrc = rightCtx.f32ColsGPU.at(srcName);
                else if (rightCtx.f32Cols.count(srcName) && !rightCtx.f32Cols.at(srcName).empty()) {
                    const auto& vec = rightCtx.f32Cols.at(srcName);
                    rightSrc = store.device()->newBuffer(vec.data(), vec.size() * sizeof(float), MTL::ResourceStorageModeShared);
                    rightSrcAllocated = true;
                }
                if (rightSrc) {
                    MTL::Buffer* g = GpuOps::gatherF32(rightSrc, unmatchedBuf, unmatchedCount, false);
                    MTL::Buffer* combined = store.device()->newBuffer(totalCount * sizeof(float), MTL::ResourceStorageModeShared);
                    std::memcpy(combined->contents(), buf->contents(), resCount * sizeof(float));
                    std::memcpy((uint8_t*)combined->contents() + resCount * sizeof(float),
                                g->contents(), unmatchedCount * sizeof(float));
                    if (rightSrcAllocated) rightSrc->release();
                    g->release(); buf.reset(combined);
                } else {
                    MTL::Buffer* combined = store.device()->newBuffer(totalCount * sizeof(float), MTL::ResourceStorageModeShared);
                    std::memcpy(combined->contents(), buf->contents(), resCount * sizeof(float));
                    std::memset((uint8_t*)combined->contents() + resCount * sizeof(float), 0, unmatchedCount * sizeof(float));
                    buf.reset(combined);
                }
            } else {
                MTL::Buffer* combined = store.device()->newBuffer(totalCount * sizeof(float), MTL::ResourceStorageModeShared);
                std::memcpy(combined->contents(), buf->contents(), resCount * sizeof(float));
                std::memset((uint8_t*)combined->contents() + resCount * sizeof(float), 0, unmatchedCount * sizeof(float));
                buf.reset(combined);
            }
        }

        // String columns
        for (auto& [name, vec] : outCtx.stringCols) {
            std::string srcName = name;
            for (const auto& [origName, mappedName] : rightColumnMapping) {
                if (mappedName == name) { srcName = origName; break; }
            }
            if (rightCtx.stringCols.count(srcName)) {
                const auto& rightVec = rightCtx.stringCols.at(srcName);
                for (uint32_t idx : unmatchedIndices) {
                    vec.push_back(idx < rightVec.size() ? rightVec[idx] : "");
                }
            } else {
                for (uint32_t i = 0; i < unmatchedCount; ++i) vec.push_back("");
            }
        }

        // Dict columns: GPU gather unmatched right dict IDs + append zeros for left
        for (auto& [name, dc] : outCtx.dictCols) {
            if (!dc.idsGPU) continue;
            std::string srcName = name;
            for (const auto& [origName, mappedName] : rightColumnMapping) {
                if (mappedName == name) { srcName = origName; break; }
            }
            auto rightDictIt = rightCtx.dictCols.find(srcName);
            if (rightDictIt != rightCtx.dictCols.end() && rightDictIt->second.idsGPU) {
                // GPU gather unmatched right dict IDs
                MTL::Buffer* g = GpuOps::gatherU32(rightDictIt->second.idsGPU, unmatchedBuf, unmatchedCount, false);
                MTL::Buffer* combined = store.device()->newBuffer(totalCount * sizeof(uint32_t), MTL::ResourceStorageModeShared);
                std::memcpy(combined->contents(), dc.idsGPU->contents(), resCount * sizeof(uint32_t));
                std::memcpy((uint8_t*)combined->contents() + resCount * sizeof(uint32_t),
                            g->contents(), unmatchedCount * sizeof(uint32_t));
                g->release();
                dc.idsGPU.reset(combined);
            } else {
                // Left-side column: pad with sentinel (0) for unmatched right rows
                MTL::Buffer* combined = store.device()->newBuffer(totalCount * sizeof(uint32_t), MTL::ResourceStorageModeShared);
                std::memcpy(combined->contents(), dc.idsGPU->contents(), resCount * sizeof(uint32_t));
                std::memset((uint8_t*)combined->contents() + resCount * sizeof(uint32_t), 0, unmatchedCount * sizeof(uint32_t));
                dc.idsGPU.reset(combined);
            }
            dc.ids.clear();
            dc.rowCount = totalCount;
            outCtx.stringCols.erase(name);
            outCtx.flatStringCols.erase(name);
        }

        // Sync CPU-side u32/f32 cols via GPU gather or GPU buffer download
        for (auto& [name, vec] : outCtx.u32Cols) {
            if (!vec.empty()) {
                if (outCtx.u32ColsGPU.count(name)) {
                    vec.resize(totalCount);
                    std::memcpy(vec.data(), outCtx.u32ColsGPU.at(name)->contents(), totalCount * sizeof(uint32_t));
                } else {
                    std::string srcName = name;
                    for (const auto& [origName, mappedName] : rightColumnMapping) {
                        if (mappedName == name) { srcName = origName; break; }
                    }
                    // Prefer existing GPU buffer from right context
                    MTL::Buffer* rightGpu = nullptr;
                    if (rightCtx.u32ColsGPU.count(srcName)) rightGpu = rightCtx.u32ColsGPU.at(srcName);
                    if (rightGpu) {
                        MTL::Buffer* g = GpuOps::gatherU32(rightGpu, unmatchedBuf, unmatchedCount, false);
                        size_t oldSz = vec.size();
                        vec.resize(oldSz + unmatchedCount);
                        std::memcpy(vec.data() + oldSz, g->contents(), unmatchedCount * sizeof(uint32_t));
                        g->release();
                    } else if (rightCtx.u32Cols.count(srcName)) {
                        const auto& rightVec = rightCtx.u32Cols.at(srcName);
                        MTL::Buffer* src = store.device()->newBuffer(rightVec.data(), rightVec.size() * sizeof(uint32_t), MTL::ResourceStorageModeShared);
                        MTL::Buffer* g = GpuOps::gatherU32(src, unmatchedBuf, unmatchedCount, false);
                        size_t oldSz = vec.size();
                        vec.resize(oldSz + unmatchedCount);
                        std::memcpy(vec.data() + oldSz, g->contents(), unmatchedCount * sizeof(uint32_t));
                        src->release(); g->release();
                    } else {
                        vec.resize(vec.size() + unmatchedCount, 0);
                    }
                }
            }
        }
        for (auto& [name, vec] : outCtx.f32Cols) {
            if (!vec.empty()) {
                if (outCtx.f32ColsGPU.count(name)) {
                    vec.resize(totalCount);
                    std::memcpy(vec.data(), outCtx.f32ColsGPU.at(name)->contents(), totalCount * sizeof(float));
                } else {
                    std::string srcName = name;
                    for (const auto& [origName, mappedName] : rightColumnMapping) {
                        if (mappedName == name) { srcName = origName; break; }
                    }
                    MTL::Buffer* rightGpu = nullptr;
                    if (rightCtx.f32ColsGPU.count(srcName)) rightGpu = rightCtx.f32ColsGPU.at(srcName);
                    if (rightGpu) {
                        MTL::Buffer* g = GpuOps::gatherF32(rightGpu, unmatchedBuf, unmatchedCount, false);
                        size_t oldSz = vec.size();
                        vec.resize(oldSz + unmatchedCount);
                        std::memcpy(vec.data() + oldSz, g->contents(), unmatchedCount * sizeof(float));
                        g->release();
                    } else if (rightCtx.f32Cols.count(srcName)) {
                        const auto& rightVec = rightCtx.f32Cols.at(srcName);
                        MTL::Buffer* src = store.device()->newBuffer(rightVec.data(), rightVec.size() * sizeof(float), MTL::ResourceStorageModeShared);
                        MTL::Buffer* g = GpuOps::gatherF32(src, unmatchedBuf, unmatchedCount, false);
                        size_t oldSz = vec.size();
                        vec.resize(oldSz + unmatchedCount);
                        std::memcpy(vec.data() + oldSz, g->contents(), unmatchedCount * sizeof(float));
                        src->release(); g->release();
                    } else {
                        vec.resize(vec.size() + unmatchedCount, 0.0f);
                    }
                }
            }
        }

        outCtx.rowCount = totalCount;
        resCount = totalCount;
        // Dict IDs already updated with unmatched rows above
        if (debug) std::cerr << "[Exec] Right Join: total output rows = " << totalCount << "\n";
    }
}

static void applyPostJoinFilter(
    const TypedExprPtr& postJoinFilter,
    const EvalContext& leftCtx, const EvalContext& rightCtx,
    const std::unordered_map<std::string, std::string>& rightColumnMapping,
    EvalContext& outCtx, bool debug
) {
    if (debug) {
        std::cerr << "[Exec] Join: applying post-join filter, current rows=" << outCtx.rowCount << "\n";
    }

    // Rewrite column references in the residual filter to resolve to actual outCtx columns.
    // When a condition like "l_suppkey != l_suppkey" appears, both sides have the same name
    // but refer to columns from different contexts (left vs right).
    // After join, outCtx may have e.g. l_suppkey_2 (from left) and l_suppkey (from right).
    // We resolve by finding which actual column in leftCtx/rightCtx matches the base name.

    // Helper: find a column in a context matching a base name (with possible instance suffix)
    auto findColInCtx = [](const EvalContext& ctx, const std::string& baseName) -> std::string {
        // Exact match first
        if (ctx.u32Cols.count(baseName) || ctx.f32Cols.count(baseName)) return baseName;
        if (ctx.stringCols.count(baseName)) return baseName;
        // Try instance-suffixed versions (e.g., l_suppkey_1, l_suppkey_2, ...)
        for (const auto& [name, _] : ctx.u32Cols) {
            auto pos = name.rfind('_');
            if (pos != std::string::npos) {
                std::string suffix = name.substr(pos + 1);
                bool allDigits = !suffix.empty() && std::all_of(suffix.begin(), suffix.end(), ::isdigit);
                if (allDigits && name.substr(0, pos) == baseName) return name;
            }
        }
        for (const auto& [name, _] : ctx.f32Cols) {
            auto pos = name.rfind('_');
            if (pos != std::string::npos) {
                std::string suffix = name.substr(pos + 1);
                bool allDigits = !suffix.empty() && std::all_of(suffix.begin(), suffix.end(), ::isdigit);
                if (allDigits && name.substr(0, pos) == baseName) return name;
            }
        }
        return "";
    };

    std::function<TypedExprPtr(const TypedExprPtr&)> rewriteResidualCols;
    rewriteResidualCols = [&](const TypedExprPtr& expr) -> TypedExprPtr {
        if (!expr) return expr;
        if (expr->kind == TypedExpr::Kind::Compare) {
            auto& cmp = expr->asCompare();
            bool bothCols = (cmp.left && cmp.right &&
                             cmp.left->kind == TypedExpr::Kind::Column &&
                             cmp.right->kind == TypedExpr::Kind::Column);
            if (bothCols) {
                const std::string& lName = cmp.left->asColumn().column;
                const std::string& rName = cmp.right->asColumn().column;

                // Find actual column names in left/right contexts
                std::string leftActual = findColInCtx(leftCtx, lName);
                std::string rightActual = findColInCtx(rightCtx, rName);

                // Map right-side column through rightColumnMapping (rename after join)
                std::string rightInOut = rightActual;
                if (!rightActual.empty() && rightColumnMapping.count(rightActual)) {
                    rightInOut = rightColumnMapping.at(rightActual);
                }

                // Use the actual names if we found them
                std::string newLName = !leftActual.empty() ? leftActual : lName;
                std::string newRName = !rightInOut.empty() ? rightInOut : rName;

                if (newLName != lName || newRName != rName) {
                    if (debug) {
                        std::cerr << "[Exec] Join: rewrote residual col: " 
                                  << lName << " -> " << newLName << ", "
                                  << rName << " -> " << newRName << "\n";
                    }
                    return TypedExpr::compare(cmp.op, 
                                              TypedExpr::column(newLName), 
                                              TypedExpr::column(newRName));
                }
            }
            return expr;
        }
        if (expr->kind == TypedExpr::Kind::Binary) {
            auto& bin = expr->asBinary();
            auto newLeft = rewriteResidualCols(bin.left);
            auto newRight = rewriteResidualCols(bin.right);
            if (newLeft != bin.left || newRight != bin.right) {
                return TypedExpr::binary(bin.op, newLeft, newRight);
            }
            return expr;
        }
        return expr;
    };

    TypedExprPtr rewrittenFilter = rewriteResidualCols(postJoinFilter);
    if (debug && rewrittenFilter != postJoinFilter) {
        std::cerr << "[Exec] Join: rewrote post-join filter column references\n";
    }

    // Upload CPU columns to GPU for filtering (they were gathered from join)
    for (const auto& [name, vec] : outCtx.u32Cols) {
        if (!vec.empty() && outCtx.u32ColsGPU.find(name) == outCtx.u32ColsGPU.end()) {
            MTL::Buffer* buf = GpuOps::createBuffer(vec.data(), vec.size() * sizeof(uint32_t));
            if (buf) outCtx.u32ColsGPU[name].reset(buf);
        }
    }
    for (const auto& [name, vec] : outCtx.f32Cols) {
        if (!vec.empty() && outCtx.f32ColsGPU.find(name) == outCtx.f32ColsGPU.end()) {
            MTL::Buffer* buf = GpuOps::createBuffer(vec.data(), vec.size() * sizeof(float));
            if (buf) outCtx.f32ColsGPU[name].reset(buf);
        }
    }

    // Use the regular filter infrastructure
    IRFilter postFilter{rewrittenFilter, ""};
    bool filterOk = GpuExecutor::executeFilter(postFilter, outCtx);
    if (debug) {
        std::cerr << "[Exec] Join: post-join filter " << (filterOk?"ok":"failed") 
                  << ", rows after=" << outCtx.rowCount << "\n";
    }
}

// -- Extracted: ensureColumnOnGPU --
// Uploads a u32 column to GPU, compacting via activeRows gather if needed.
static MTL::Buffer* ensureColumnOnGPU(EvalContext& ctx, const std::string& col, bool debug) {
    auto& store = GpuColumnStore::instance();
    uint32_t expectedSize = ctx.activeRowsGPU ? ctx.activeRowsCountGPU : (uint32_t)ctx.rowCount;
    if (ctx.u32ColsGPU.count(col)) {
        MTL::Buffer* existing = ctx.u32ColsGPU.at(col);
        if (ctx.activeRowsGPU && ctx.activeRowsCountGPU > 0 &&
            existing->length() / sizeof(uint32_t) > expectedSize) {
            if (debug) std::cerr << "[Exec] ensureGPU: compacting GPU buf " << col << " from " << (existing->length()/sizeof(uint32_t)) << " to " << expectedSize << "\n";
            auto compactedBuf = GpuOps::gatherU32(existing, ctx.activeRowsGPU, ctx.activeRowsCountGPU, true);
            if (compactedBuf) {
                if (debug) {
                    uint32_t* p = (uint32_t*)compactedBuf->contents();
                    std::cerr << "[Exec] ensureGPU: compacted " << col << " first 5:";
                    for (uint32_t i = 0; i < std::min(expectedSize, 5u); i++) std::cerr << " " << p[i];
                    if (debug) std::cerr << "\n";
                }
                ctx.u32ColsGPU[col].reset(compactedBuf);
                return compactedBuf;
            }
        }
        return existing;
    }
    if (ctx.u32Cols.count(col)) {
         const auto& vec = ctx.u32Cols.at(col);
         if (ctx.activeRowsGPU && ctx.activeRowsCountGPU > 0 && vec.size() > expectedSize) {
             auto fullBuf = store.device()->newBuffer(vec.data(), vec.size() * sizeof(uint32_t), MTL::ResourceStorageModeShared);
             if (fullBuf) {
                 auto compactedBuf = GpuOps::gatherU32(fullBuf, ctx.activeRowsGPU, ctx.activeRowsCountGPU, true);
                 fullBuf->release();
                 if (compactedBuf) {
                     ctx.u32ColsGPU[col].reset(compactedBuf);
                     return compactedBuf;
                 }
             }
         }
         auto buf = store.device()->newBuffer(vec.data(), vec.size() * sizeof(uint32_t), MTL::ResourceStorageModeShared);
         ctx.u32ColsGPU[col].reset(buf);
         return buf;
    }
    return nullptr;
}

// -- Extracted: scatterJoinOutputColumns --
// Gathers/scatters columns from left and right join contexts into the output context
// using the probe/build index arrays from the join result.
// Returns true on success (including the resCount==0 fast-path).
static bool scatterJoinOutputColumns(
    EvalContext&       leftCtx,
    EvalContext&       rightCtx,
    EvalContext&       outCtx,
    const JoinResult&  jRes,
    uint32_t           resCount,
    uint32_t           lCount,
    uint32_t           rCount,
    bool               isAntiJoin,
    bool               isSemiJoin,
    bool               rightAntiGather,
    std::unordered_map<std::string, std::string>& rightColumnMappingOut,
    bool               debug)
{
    auto& store = GpuColumnStore::instance();
    
    outCtx.rowCount = resCount;
    outCtx.activeRowsGPU = nullptr;
    outCtx.activeRowsCountGPU = 0; // Materialized

    // Collect LEFT column names (these are the "primary" names in the output)
    std::unordered_set<std::string> leftColumnNames;
    if (!rightAntiGather) {
        for (const auto& [name, _] : leftCtx.u32Cols) leftColumnNames.insert(name);
        for (const auto& [name, _] : leftCtx.f32Cols) leftColumnNames.insert(name);
        for (const auto& [name, _] : leftCtx.stringCols) leftColumnNames.insert(name);
        for (const auto& [name, _] : leftCtx.dictCols) leftColumnNames.insert(name);
    }
    
    // Pre-compute the rename mapping for ALL right column names
    std::unordered_set<std::string> rightColumnNames;
    for (const auto& [name, _] : rightCtx.u32Cols) rightColumnNames.insert(name);
    for (const auto& [name, _] : rightCtx.f32Cols) rightColumnNames.insert(name);
    for (const auto& [name, _] : rightCtx.stringCols) rightColumnNames.insert(name);
    for (const auto& [name, _] : rightCtx.dictCols) rightColumnNames.insert(name);
    
    // Map from original right column name to output column name
    auto& rightColumnMapping = rightColumnMappingOut;
    rightColumnMapping.clear();
    std::unordered_set<std::string> usedNames;
    for (const auto& name : leftColumnNames) usedNames.insert(name);
    
    for (const auto& name : rightColumnNames) {
        if (leftColumnNames.count(name) == 0) {
            rightColumnMapping[name] = name;
            usedNames.insert(name);
        } else {
            for (int suffix = 1; suffix <= 10; ++suffix) {
                std::string newName = name + "_" + std::to_string(suffix);
                if (usedNames.count(newName) == 0) {
                    rightColumnMapping[name] = newName;
                    usedNames.insert(newName);
                    if (debug) {
                        std::cerr << "[Exec] Join: Renaming duplicate column " << name << " -> " << newName << "\n";
                    }
                    break;
                }
            }
            if (rightColumnMapping.count(name) == 0) {
                std::string fallback = name + "_r";
                rightColumnMapping[name] = fallback;
                usedNames.insert(fallback);
            }
        }
    }
    
    auto getRightColumnName = [&](const std::string& name) -> std::string {
        auto it = rightColumnMapping.find(name);
        if (it != rightColumnMapping.end()) return it->second;
        return name;
    };

    if (resCount == 0) {
        for (const auto& [name, _] : leftCtx.u32Cols) { 
            outCtx.u32Cols[name] = {};
        }
        for (const auto& [name, _] : leftCtx.f32Cols) {
            outCtx.f32Cols[name] = {};
        }
        for (const auto& [name, _] : leftCtx.stringCols) {
            outCtx.stringCols[name] = {};
        }
        for (const auto& [name, dict] : leftCtx.dictCols) {
            DictEncoded emptyDict;
            emptyDict.dictionary = dict.dictionary;
            emptyDict.rowCount = 0;
            outCtx.dictCols[name] = std::move(emptyDict);
        }
        for (const auto& [name, _] : rightCtx.u32Cols) {
            std::string outName = getRightColumnName(name);
            outCtx.u32Cols[outName] = {};
        }
        for (const auto& [name, _] : rightCtx.f32Cols) {
            std::string outName = getRightColumnName(name);
            outCtx.f32Cols[outName] = {};
        }
        for (const auto& [name, _] : rightCtx.stringCols) {
            std::string outName = getRightColumnName(name);
            outCtx.stringCols[outName] = {};
        }
        for (const auto& [name, dict] : rightCtx.dictCols) {
            std::string outName = getRightColumnName(name);
            DictEncoded emptyDict;
            emptyDict.dictionary = dict.dictionary;
            emptyDict.rowCount = 0;
            outCtx.dictCols[outName] = std::move(emptyDict);
        }
        return true;
    }

    // Gather Left Columns
    if (!rightAntiGather) {
    if (debug && jRes.probeIndices) {
        uint32_t* probePtr = (uint32_t*)jRes.probeIndices->contents();
        std::cerr << "[Exec] Join: probeIndices first 5: ";
        for (uint32_t i = 0; i < std::min(5u, resCount); ++i) std::cerr << probePtr[i] << " ";
        if (debug) std::cerr << "\n";
    }
    for (const auto& [name, valid] : leftCtx.u32Cols) {
        if (debug) std::cerr << "[Exec] Join: gathering L_U32 " << name << " srcSize=" << valid.size() << "\n";
        MTL::Buffer* src = ensureColumnOnGPU(leftCtx, name, debug);
        if (src) {
             MTL::Buffer* gathered = GpuOps::gatherU32(src, jRes.probeIndices, resCount, false);
             outCtx.u32ColsGPU[name].reset(gathered);
             outCtx.u32Cols[name].clear();
        }
    }
    for (const auto& [name, valid] : leftCtx.f32Cols) {
        if (debug) std::cerr << "[Exec] Join: gathering L_F32 " << name << " srcSize=" << valid.size() << "\n";
        MTL::Buffer* src = nullptr;
        if (leftCtx.f32ColsGPU.count(name)) src = leftCtx.f32ColsGPU.at(name);
        else if (leftCtx.f32Cols.count(name)) {
             const auto& vec = leftCtx.f32Cols.at(name);
             src = store.device()->newBuffer(vec.data(), vec.size() * sizeof(float), MTL::ResourceStorageModeShared);
             leftCtx.f32ColsGPU[name].reset(src);
        }
        
        if (src) {
             MTL::Buffer* gathered = GpuOps::gatherF32(src, jRes.probeIndices, resCount, false);
             outCtx.f32ColsGPU[name].reset(gathered);
             outCtx.f32Cols[name].clear();
        }
    }
    } // end if (!rightAntiGather)
    
    // Gather Right Columns
    if (rCount > 0 && !isSemiJoin && (!isAntiJoin || rightAntiGather)) {
        for (const auto& [name, valid] : rightCtx.u32Cols) {
            std::string outName = getRightColumnName(name);
            if (debug) std::cerr << "[Exec] Join: gathering R_U32 " << name << " -> " << outName << "\n";
            MTL::Buffer* src = ensureColumnOnGPU(rightCtx, name, debug);
            if (src) {
                 MTL::Buffer* gathered = GpuOps::gatherU32(src, jRes.buildIndices, resCount, false);
                 outCtx.u32ColsGPU[outName].reset(gathered);
                 outCtx.u32Cols[outName].clear();
            }
        }
        for (const auto& [name, valid] : rightCtx.f32Cols) {
            std::string outName = getRightColumnName(name);
            if (debug) std::cerr << "[Exec] Join: gathering R_F32 " << name << " -> " << outName << "\n";
            MTL::Buffer* src = nullptr;
            if (rightCtx.f32ColsGPU.count(name)) src = rightCtx.f32ColsGPU.at(name);
            else if (rightCtx.f32Cols.count(name)) {
                 const auto& vec = rightCtx.f32Cols.at(name);
                 src = store.device()->newBuffer(vec.data(), vec.size() * sizeof(float), MTL::ResourceStorageModeShared);
                 rightCtx.f32ColsGPU[name].reset(src);
            }
    
            if (src) {
                 MTL::Buffer* gathered = GpuOps::gatherF32(src, jRes.buildIndices, resCount, false);
                 outCtx.f32ColsGPU[outName].reset(gathered);
                 outCtx.f32Cols[outName].clear();
            }
        }
    } else if (resCount > 0) {
         for (const auto& [name, valid] : rightCtx.u32Cols) {
             std::string outName = getRightColumnName(name);
             MTL::Buffer* buf = store.device()->newBuffer(resCount * sizeof(uint32_t), MTL::ResourceStorageModeShared);
             std::memset(buf->contents(), 0, resCount * sizeof(uint32_t));
             outCtx.u32ColsGPU[outName].reset(buf);
             std::vector<uint32_t> cpuVec(resCount, 0);
             outCtx.u32Cols[outName] = std::move(cpuVec);
        }
         for (const auto& [name, valid] : rightCtx.f32Cols) {
             std::string outName = getRightColumnName(name);
             MTL::Buffer* buf = store.device()->newBuffer(resCount * sizeof(float), MTL::ResourceStorageModeShared);
             std::memset(buf->contents(), 0, resCount * sizeof(float));
             outCtx.f32ColsGPU[outName].reset(buf);
             std::vector<float> cpuVec(resCount, 0.0f);
             outCtx.f32Cols[outName] = std::move(cpuVec);
        }
    }

    // CPU Gather for String Columns (or GPU dict-id gather when available)
    if (!leftCtx.stringCols.empty() || !rightCtx.stringCols.empty() ||
        !leftCtx.dictCols.empty() || !rightCtx.dictCols.empty()) {
        std::vector<uint32_t> cpuProbeIndices(resCount);
        std::vector<uint32_t> cpuBuildIndices(resCount);
        std::memcpy(cpuProbeIndices.data(), jRes.probeIndices->contents(), resCount * sizeof(uint32_t));
        std::memcpy(cpuBuildIndices.data(), jRes.buildIndices->contents(), resCount * sizeof(uint32_t));
        
        auto parallelGather = [&](const std::vector<uint32_t>& indices, const std::vector<std::string>& srcVec, std::vector<std::string>& dstVec) {
             dstVec.resize(resCount);
             size_t numThreads = std::thread::hardware_concurrency();
             if (numThreads == 0) numThreads = 4;
             if (resCount < 10000) numThreads = 1;
             size_t chunkSize = (resCount + numThreads - 1) / numThreads;
             std::vector<std::future<void>> futures;
             for (size_t t = 0; t < numThreads; ++t) {
                 size_t start = t * chunkSize;
                 size_t end = std::min(start + chunkSize, (size_t)resCount);
                 if (start >= end) break;
                 futures.push_back(std::async(std::launch::async, [&, start, end]() {
                     for (size_t i = start; i < end; ++i) {
                         uint32_t idx = indices[i];
                         if (idx < srcVec.size()) dstVec[i] = srcVec[idx];
                     }
                 }));
             }
             for (auto& f : futures) f.wait();
        };

        auto dictGather = [&](const std::string& name, const EvalContext& srcCtx,
                              MTL::Buffer* indexBuf, const std::string& outName) {
            auto dictIt = srcCtx.dictCols.find(name);
            if (dictIt == srcCtx.dictCols.end() || !dictIt->second.idsGPU) return false;
            const auto& srcDict = dictIt->second;
            MTL::Buffer* gatheredIds = GpuOps::gatherU32(srcDict.idsGPU, indexBuf, resCount, false);
            if (!gatheredIds) return false;
            DictEncoded outDict;
            outDict.dictionary = srcDict.dictionary;
            outDict.idsGPU.reset(gatheredIds);
            outDict.rowCount = resCount;
            outCtx.dictCols[outName] = std::move(outDict);
            outCtx.stringCols.erase(outName);
            outCtx.flatStringCols.erase(outName);
            if (debug) std::cerr << "[Exec] Join: GPU dict gather " << name << " -> " << outName
                                 << " (" << srcDict.dictionary.size() << " unique, " << resCount << " rows)\n";
            return true;
        };

        auto flatGather = [&](const EvalContext& srcCtx, const std::string& name,
                              MTL::Buffer* indexBuf, const std::string& outName) -> bool {
            auto fit = srcCtx.flatStringCols.find(name);
            if (fit == srcCtx.flatStringCols.end() || !fit->second.chars) return false;
            auto& flat = fit->second;
            auto r = GpuOps::gatherFlatString(flat.chars, flat.offsets, flat.lengths,
                                               indexBuf, resCount, true);
            if (!r.chars) return false;
            FlatStringCol outFlat;
            outFlat.chars.reset(r.chars); outFlat.offsets.reset(r.offsets); outFlat.lengths.reset(r.lengths);
            outFlat.rowCount = r.rowCount; outFlat.totalBytes = r.totalBytes;
            outCtx.flatStringCols[outName] = outFlat;
            outCtx.stringCols.erase(outName);
            if (debug) std::cerr << "[Exec] Join: GPU flat string gather " << name << " -> " << outName
                                 << " (" << resCount << " rows, " << r.totalBytes << " bytes)\n";
            return true;
        };

        for (const auto& [name, vec] : leftCtx.stringCols) {
            if (rightAntiGather) continue;
            if (dictGather(name, leftCtx, jRes.probeIndices, name)) continue;
            if (flatGather(leftCtx, name, jRes.probeIndices, name)) continue;
            if (debug) std::cerr << "[Exec] Join: gathering L_STR " << name << " srcSize=" << vec.size() << " resCount=" << resCount << "\n";
            std::vector<std::string> newVec;
            parallelGather(cpuProbeIndices, vec, newVec);
            if (debug) std::cerr << "[Exec] Join: gathered L_STR " << name << " newVec.size=" << newVec.size() << "\n";
            outCtx.stringCols[name] = std::move(newVec);
        }
        for (const auto& [name, vec] : rightCtx.stringCols) {
             if (isSemiJoin || (isAntiJoin && !rightAntiGather)) continue;
             std::string outName = getRightColumnName(name);
             if (dictGather(name, rightCtx, jRes.buildIndices, outName)) continue;
             if (flatGather(rightCtx, name, jRes.buildIndices, outName)) continue;
             if (debug) std::cerr << "[Exec] Join: gathering R_STR " << name << " -> " << outName << " srcSize=" << vec.size() << " resCount=" << resCount << "\n";
             std::vector<std::string> newVec;
             parallelGather(cpuBuildIndices, vec, newVec);
             if (debug) std::cerr << "[Exec] Join: gathered R_STR " << name << " newVec.size=" << newVec.size() << "\n";
             outCtx.stringCols[outName] = std::move(newVec);
        }
        for (const auto& [name, dc] : leftCtx.dictCols) {
            if (rightAntiGather) continue;
            if (leftCtx.stringCols.count(name)) continue;
            dictGather(name, leftCtx, jRes.probeIndices, name);
        }
        for (const auto& [name, dc] : rightCtx.dictCols) {
            if (isSemiJoin || (isAntiJoin && !rightAntiGather)) continue;
            if (rightCtx.stringCols.count(name)) continue;
            std::string outName = getRightColumnName(name);
            dictGather(name, rightCtx, jRes.buildIndices, outName);
        }
        for (const auto& [name, flat] : leftCtx.flatStringCols) {
            if (rightAntiGather) continue;
            if (leftCtx.stringCols.count(name) || leftCtx.dictCols.count(name)) continue;
            flatGather(leftCtx, name, jRes.probeIndices, name);
        }
        for (const auto& [name, flat] : rightCtx.flatStringCols) {
            if (isSemiJoin || (isAntiJoin && !rightAntiGather)) continue;
            if (rightCtx.stringCols.count(name) || rightCtx.dictCols.count(name)) continue;
            std::string outName = getRightColumnName(name);
            flatGather(rightCtx, name, jRes.buildIndices, outName);
        }
    }
    
    GpuOps::sync(); // Ensure all async gathers complete
    return false; // Not an early return — caller continues
}

bool GpuExecutor::executeJoin(const IRJoin& join,
                                   EvalContext& leftCtx, EvalContext& rightCtx, EvalContext& outCtx) {
    const bool debug = env_truthy("GPUDB_DEBUG_OPS");
    
    // Supported: INNER, LEFT, RIGHT, SEMI, ANTI, MARK (MARK treated as SEMI)
    if (join.type != JoinType::Inner && join.type != JoinType::Left &&
        join.type != JoinType::Right && join.type != JoinType::Semi && 
        join.type != JoinType::Anti && join.type != JoinType::Mark) {
        if (debug) std::cerr << "[Exec] Join: unsupported join type\n";
        return false;
    }
    
    const bool isLeftJoin = (join.type == JoinType::Left);
    const bool isRightJoin = (join.type == JoinType::Right);
    const bool isSemiJoin = (join.type == JoinType::Semi || join.type == JoinType::Mark);
    const bool isAntiJoin = (join.type == JoinType::Anti);
    
    if (debug) {
        std::cerr << "[Exec] Join: type=" << static_cast<int>(join.type) 
                  << " isLeft=" << isLeftJoin << " isRight=" << isRightJoin 
                  << " isSemi=" << isSemiJoin << " isAnti=" << isAntiJoin << "\n";
        if (debug) std::cerr << "[Exec] Join: leftCtx has " << leftCtx.u32Cols.size() << " u32 cols, "
                  << leftCtx.f32Cols.size() << " f32 cols, " << leftCtx.rowCount << " rows";
        if (debug) for (const auto& [k, v] : leftCtx.u32Cols) std::cerr << " " << k;
        if (debug) std::cerr << "\n";
        if (debug) std::cerr << "[Exec] Join: rightCtx has " << rightCtx.u32Cols.size() << " u32 cols, "
                  << rightCtx.f32Cols.size() << " f32 cols, " << rightCtx.rowCount << " rows";
        if (debug) for (const auto& [k, v] : rightCtx.u32Cols) std::cerr << " " << k;
        if (debug) std::cerr << "\n";
    }
    
    // Extract all join key pairs from the condition
    std::vector<std::pair<std::string, std::string>> keyPairs;
    
    bool isCrossJoin = false;
    bool hasPostJoinFilter = false;  // For non-equality conditions (>, <, etc.)
    TypedExprPtr postJoinFilter = nullptr;
    
    if (join.conditionStr == "1=1") {
        isCrossJoin = true;
    } else if (join.condition && join.condition->kind == TypedExpr::Kind::Compare) {
        const auto& cmp = join.condition->asCompare();
        if (cmp.op == CompareOp::Eq &&
            cmp.left->kind == TypedExpr::Kind::Literal &&
            cmp.right->kind == TypedExpr::Kind::Literal) {
            isCrossJoin = true;
        }
        // Check for non-equality comparison (treat as cross join + filter)
        else if (cmp.op != CompareOp::Eq) {
            // This is a non-equality join condition (e.g., value > threshold from HAVING subquery)
            // Treat as cross-join with post-filter
            isCrossJoin = true;
            hasPostJoinFilter = true;
            postJoinFilter = join.condition;
            if (debug) {
                std::cerr << "[Exec] Join: detected non-equality condition, treating as cross-join + filter\n";
                std::cerr << "[Exec] Join: conditionStr=" << join.conditionStr << "\n";
            }
        }
    }

    if (join.condition && !isCrossJoin) {
        std::vector<TypedExprPtr> residuals;
        extractJoinKeyPairs(join.condition, keyPairs, &residuals);
        // If we have equality keys + residual non-eq conditions, apply residuals as post-join filter
        if (!keyPairs.empty() && !residuals.empty()) {
            hasPostJoinFilter = true;
            if (residuals.size() == 1) {
                postJoinFilter = residuals[0];
            } else {
                // Chain residuals with AND
                TypedExprPtr combined = residuals[0];
                for (size_t ri = 1; ri < residuals.size(); ++ri) {
                    combined = TypedExpr::binary(BinaryOp::And, combined, residuals[ri]);
                }
                postJoinFilter = combined;
            }
            if (debug) {
                std::cerr << "[Exec] Join: extracted " << residuals.size() 
                          << " residual condition(s) as post-join filter\n";
            }
        }
    }
    
    // Fallback: parse from condition string (single pair only)
    if (!isCrossJoin && keyPairs.empty()) {
        std::string cond = join.conditionStr;
        // Split by " AND " (case-insensitive) to extract each equality
        size_t pos = 0;
        while (pos < cond.size()) {
            // Find next " AND " delimiter (case-insensitive)
            size_t andPos = std::string::npos;
            for (size_t j = pos; j + 5 <= cond.size(); ++j) {
                if ((cond[j] == ' ') &&
                    (cond[j+1] == 'A' || cond[j+1] == 'a') &&
                    (cond[j+2] == 'N' || cond[j+2] == 'n') &&
                    (cond[j+3] == 'D' || cond[j+3] == 'd') &&
                    (cond[j+4] == ' ')) {
                    andPos = j;
                    break;
                }
            }
            std::string part;
            if (andPos != std::string::npos) {
                part = cond.substr(pos, andPos - pos);
                pos = andPos + 5; // skip " AND "
            } else {
                part = cond.substr(pos);
                pos = cond.size();
            }
            auto eq = part.find('=');
            if (eq != std::string::npos) {
                std::string left = base_ident(part.substr(0, eq));
                std::string right = base_ident(part.substr(eq + 1));
                if (!left.empty() && !right.empty()) {
                    keyPairs.emplace_back(left, right);
                }
            }
        }
    }
    
    // If we still have no key pairs and there's a condition, it might be a complex non-equality join
    // (e.g., NESTED_LOOP_JOIN with > comparison from HAVING subquery)
    if (!isCrossJoin && keyPairs.empty() && join.condition) {
        // Treat as cross-join with post-filter
        if (debug) std::cerr << "[Exec] Join: no equi-join keys found but has condition, treating as cross-join + filter\n";
        isCrossJoin = true;
        hasPostJoinFilter = true;
        postJoinFilter = join.condition;
    }
    
    if (!isCrossJoin && keyPairs.empty()) {
        if (debug) std::cerr << "[Exec] Join: no key pairs found\n";
        return false;
    }
    
    if (debug) {
        std::cerr << "[Exec] Join: " << keyPairs.size() << " key pair(s):\n";
        for (const auto& [l, r] : keyPairs) {
            std::cerr << "[Exec] Join:   " << l << " = " << r << std::endl;
        }
        if (debug) std::cerr << "[Exec] Join: leftCtx has " << leftCtx.u32Cols.size() << " u32 cols, " 
                  << leftCtx.f32Cols.size() << " f32 cols, " << leftCtx.rowCount << " rows";
        if (debug) for (const auto& [n,_] : leftCtx.u32Cols) std::cerr << " " << n;
        if (debug) std::cerr << std::endl;
        if (debug) std::cerr << "[Exec] Join: rightCtx has " << rightCtx.u32Cols.size() << " u32 cols, "
                  << rightCtx.f32Cols.size() << " f32 cols, " << rightCtx.rowCount << " rows";
        if (debug) for (const auto& [n,_] : rightCtx.u32Cols) std::cerr << " " << n;
        if (debug) std::cerr << std::endl;
    }
    
    // Helper to find column with suffix fallback for multi-instance tables
    // Checks f32Cols and auto-converts to u32Cols (bitwise) if found
    auto findColWithSuffix = [](EvalContext& ctx, 
                                 const std::string& col) -> std::string {
        // Check U32 Direct
        if (ctx.u32Cols.find(col) != ctx.u32Cols.end()) return col;
        
        // Check F32 Direct -> Convert
        if (ctx.f32Cols.find(col) != ctx.f32Cols.end()) {
             const auto& fVec = ctx.f32Cols.at(col);
             std::vector<uint32_t> uVec(fVec.size());
             if (!fVec.empty()) std::memcpy(uVec.data(), fVec.data(), fVec.size() * sizeof(uint32_t));
             ctx.u32Cols[col] = std::move(uVec);
             return col;
        }

        // Try suffixed versions
        for (int suffix = 1; suffix <= 9; ++suffix) {
            std::string suffixedCol = col + "_" + std::to_string(suffix);
            if (ctx.u32Cols.find(suffixedCol) != ctx.u32Cols.end()) return suffixedCol;
            if (ctx.f32Cols.find(suffixedCol) != ctx.f32Cols.end()) {
                 const auto& fVec = ctx.f32Cols.at(suffixedCol);
                 std::vector<uint32_t> uVec(fVec.size());
                 if (!fVec.empty()) std::memcpy(uVec.data(), fVec.data(), fVec.size() * sizeof(uint32_t));
                 ctx.u32Cols[suffixedCol] = std::move(uVec);
                 return suffixedCol;
            }
        }
        return "";
    };

    // Helper to attempt fuzzy resolution of a column in a context
    // excludeCols: set of column names already used by prior key resolutions (for multi-key joins)
    auto fuzzyResolve = [&](EvalContext& ctx, const std::string& colName,
                            const std::unordered_set<std::string>& excludeCols = {}) -> std::string {
        // 1. Try suffixed versions (e.g. name_1)
        std::string s = findColWithSuffix(ctx, colName); // Use updated helper
        if (!s.empty() && excludeCols.find(s) == excludeCols.end()) return s;

        // 2. Try prefix aliases BEFORE positional refs (l_ -> o_, etc)
        if (colName.size() > 2 && colName[1] == '_') {
            std::string suffix = colName.substr(2);
            static const std::vector<std::string> prefixes = {"l_", "o_", "c_", "p_", "s_", "ps_", "n_", "r_"};
            for (const auto& p : prefixes) {
                std::string alt = p + suffix;
                std::string res = findColWithSuffix(ctx, alt); // Re-use helper to handle conversion
                if (!res.empty() && excludeCols.find(res) == excludeCols.end()) return res;
            }
        }

        // 3. Try positional refs (#0..#9) - skip already-used refs
        for (int i = 0; i < 10; ++i) {
            std::string posRef = "#" + std::to_string(i);
            if (excludeCols.find(posRef) != excludeCols.end()) continue; // Skip already used
            if (ctx.u32Cols.count(posRef)) return posRef;
            // Check F32 #N
            if (ctx.f32Cols.count(posRef)) {
                 const auto& fVec = ctx.f32Cols.at(posRef);
                 std::vector<uint32_t> uVec(fVec.size());
                 if (!fVec.empty()) std::memcpy(uVec.data(), fVec.data(), fVec.size() * sizeof(uint32_t));
                 ctx.u32Cols[posRef] = std::move(uVec);
                 return posRef;
            }
        }
        
        // 4. Try suffix match (Iterate both u32 and f32)
        auto underscorePos = colName.find('_');
        if (underscorePos != std::string::npos) {
             std::string suffix = colName.substr(underscorePos); 
             for (const auto& [n, _] : ctx.u32Cols) {
                 if (n.size() >= suffix.size() && 
                     n.rfind(suffix) == n.size() - suffix.size()) {
                     return n;
                 }
             }
             // F32 Loop
             for (const auto& [n, _] : ctx.f32Cols) {
                 if (n.size() >= suffix.size() && 
                     n.rfind(suffix) == n.size() - suffix.size()) {
                     // Check if not already in u32 (optimization)
                     findColWithSuffix(ctx, n); // Ensure converted
                     return n;
                 }
             }
        }

        // 5. Try stripping explicit aliases (_rhs_N, _lhs_N)
        size_t rhsPos = colName.find("_rhs_");
        if (rhsPos != std::string::npos) {
            std::string base = colName.substr(0, rhsPos);
            // Search exact base
            if (ctx.u32Cols.count(base) || ctx.f32Cols.count(base)) return base;
            // Recurse fuzzy on base
            // (Use simple direct check of prefixes to avoid infinite recursion if implemented recursively)
            if (base.size() > 2 && base[1] == '_') {
                std::string suffix = base.substr(2);
                static const std::vector<std::string> prefixes = {"l_", "o_", "c_", "p_", "s_", "ps_", "n_", "r_"};
                for (const auto& p : prefixes) {
                     std::string alt = p + suffix;
                     if (ctx.u32Cols.count(alt) || ctx.f32Cols.count(alt)) return alt;
                }
            }
        }
        
        return "";
    };
    
    // Resolve key columns - figure out which column is in left vs right
    std::vector<std::pair<std::string, std::string>> resolvedKeys; // (leftCol, rightCol)
    std::unordered_set<std::string> usedLeftCols, usedRightCols; // Track used cols for multi-key joins
    for (auto& [k1, k2] : keyPairs) {
        if (k1 == "supplier_no") k1 = "l_suppkey";
        if (k2 == "supplier_no") k2 = "l_suppkey";
        // Check if k1 is in left and k2 is in right (with suffix fallback)
        std::string k1Left = findColWithSuffix(leftCtx, k1);
        std::string k2Right = findColWithSuffix(rightCtx, k2);
        std::string k2Left = findColWithSuffix(leftCtx, k2);
        std::string k1Right = findColWithSuffix(rightCtx, k1);
        
        bool k1InLeft = !k1Left.empty();
        bool k2InRight = !k2Right.empty();
        bool k2InLeft = !k2Left.empty();
        bool k1InRight = !k1Right.empty();
        
        if (k1InLeft && k2InRight) {
            resolvedKeys.emplace_back(k1Left, k2Right);
            usedLeftCols.insert(k1Left);
            usedRightCols.insert(k2Right);
        } else if (k2InLeft && k1InRight) {
            resolvedKeys.emplace_back(k2Left, k1Right);
            usedLeftCols.insert(k2Left);
            usedRightCols.insert(k1Right);
        } else {
             // Try to fuzzy resolve missing left key if right key exists
             std::string leftResolved, rightResolved;
             
             if (k1InRight) {
                 // Right has k1. We need k2 in Left.
                 rightResolved = k1Right;
                 leftResolved = fuzzyResolve(leftCtx, k2, usedLeftCols);
             } else if (k2InRight) {
                 // Right has k2. We need k1 in Left.
                 rightResolved = k2Right;
                 leftResolved = fuzzyResolve(leftCtx, k1, usedLeftCols);
             }
             
             // Try to fuzzy resolve missing right key if left key exists
             if (leftResolved.empty() && rightResolved.empty()) {
                  if (k1InLeft) {
                      leftResolved = k1Left;
                      rightResolved = fuzzyResolve(rightCtx, k2, usedRightCols);
                  } else if (k2InLeft) {
                      leftResolved = k2Left;
                      rightResolved = fuzzyResolve(rightCtx, k1, usedRightCols);
                  }
             }
             
             if (!leftResolved.empty() && !rightResolved.empty()) {
                  resolvedKeys.emplace_back(leftResolved, rightResolved);
                  usedLeftCols.insert(leftResolved);
                  usedRightCols.insert(rightResolved);
                   if (debug) {
                       std::cerr << "[Exec] Join: fuzzy resolved " << k1 << "=" << k2 << " to (" 
                                 << leftResolved << ", " << rightResolved << ")\n";
                   }
             } else {
                if (debug) {
                    std::cerr << "[Exec] Join: cannot resolve key pair " << k1 << "=" << k2 
                            << " k1InLeft=" << k1InLeft << " k2InRight=" << k2InRight
                            << " k2InLeft=" << k2InLeft << " k1InRight=" << k1InRight << "\n";
                }
                return false;
             }
        }
    }
    
    if (debug) {
        std::cerr << "[Exec] Join: resolved " << resolvedKeys.size() << " key pair(s)\n";
    }
    
    // Get vectors for all keys
    std::vector<const std::vector<uint32_t>*> leftKeyVecs, rightKeyVecs;
    for (const auto& [lk, rk] : resolvedKeys) {
        leftKeyVecs.push_back(&leftCtx.u32Cols.at(lk));
        rightKeyVecs.push_back(&rightCtx.u32Cols.at(rk));
    }
    
    if (!isCrossJoin && resolvedKeys.empty()) return false;

    if (!isCrossJoin && resolvedKeys.size() > 2) throw std::runtime_error("GPU Join > 2 columns not implemented");

    auto& store = GpuColumnStore::instance();
    
    // ensureGPU is now the static function ensureColumnOnGPU (extracted above)
    auto ensureGPU = [&](EvalContext& ctx, const std::string& col) -> MTL::Buffer* {
        return ensureColumnOnGPU(ctx, col, debug);
    };
    
    // --- Materialize contexts before join (compact to dense form) ---
    materializeJoinContext(leftCtx, "left", debug);
    materializeJoinContext(rightCtx, "right", debug);
    
    uint32_t rCount = (uint32_t)rightCtx.rowCount;
    uint32_t lCount = (uint32_t)leftCtx.rowCount;

    MTL::Buffer* lBuf = nullptr;
    MTL::Buffer* rBuf = nullptr;
    
    JoinResult jRes;
    
    if (lCount > 0 && rCount > 0) {
        if (isCrossJoin) {
            if (debug) std::cerr << "[Exec] GPU Join: Cross Join 1=1 (" << lCount << " x " << rCount << ")\n";
            uint64_t totalCount = (uint64_t)lCount * (uint64_t)rCount;
             
            auto device = GpuColumnStore::instance().device();
            if (totalCount > UINT32_MAX) {
                std::cerr << "[Exec] WARNING: Cross join produces " << totalCount 
                          << " rows, exceeding uint32_t max. Clamping to " << UINT32_MAX << ".\n";
                totalCount = UINT32_MAX;
            }
            jRes.count = (uint32_t)totalCount;
            jRes.probeIndices.reset(device->newBuffer(totalCount * sizeof(uint32_t), MTL::ResourceStorageModeShared));
            jRes.buildIndices.reset(device->newBuffer(totalCount * sizeof(uint32_t), MTL::ResourceStorageModeShared));
            
            // Get or create left/right index buffers on GPU
            MTL::Buffer* lIndicesGPU = leftCtx.activeRowsGPU;
            MTL::Buffer* rIndicesGPU = rightCtx.activeRowsGPU;
            bool createdL = false, createdR = false;
            
            if (!lIndicesGPU) {
                lIndicesGPU = GpuOps::iotaU32(lCount);
                createdL = true;
            }
            if (!rIndicesGPU) {
                rIndicesGPU = GpuOps::iotaU32(rCount);
                createdR = true;
            }
            
            // Cross product on GPU
            GpuOps::crossProduct(lIndicesGPU, rIndicesGPU,
                                       jRes.probeIndices, jRes.buildIndices,
                                       lCount, rCount);
            
            if (createdL) lIndicesGPU->release();
            if (createdR) rIndicesGPU->release();
        } else if (resolvedKeys.size() == 2) {
            if (debug) {
                std::cerr << "[Exec] Multi-Key Join (2 keys)\n";
                std::cerr << "[Exec] Multi-Key Join: key0=(" << resolvedKeys[0].first << ", " << resolvedKeys[0].second << ")\n";
                std::cerr << "[Exec] Multi-Key Join: key1=(" << resolvedKeys[1].first << ", " << resolvedKeys[1].second << ")\n";
            }
            MTL::Buffer* l1 = ensureGPU(leftCtx, resolvedKeys[0].first);
            MTL::Buffer* r1 = ensureGPU(rightCtx, resolvedKeys[0].second);
            MTL::Buffer* l2 = ensureGPU(leftCtx, resolvedKeys[1].first);
            MTL::Buffer* r2 = ensureGPU(rightCtx, resolvedKeys[1].second);
            if(!l1||!r1||!l2||!r2) throw std::runtime_error("Missing GPU col data for multi-key join");
            
            uint32_t lSize = (uint32_t)leftCtx.rowCount;
            uint32_t rSize = (uint32_t)rightCtx.rowCount;
            
            if (debug) std::cerr << "[Exec] Multi-Key Join: packing left (" << lSize << " rows)...\n" << std::flush;
            lBuf = GpuOps::packU32ToU64(l1, l2, lSize);
            if (debug) std::cerr << "[Exec] Multi-Key Join: packing right (" << rSize << " rows)...\n" << std::flush;
            rBuf = GpuOps::packU32ToU64(r1, r2, rSize);
            if (debug) std::cerr << "[Exec] Multi-Key Join: packing done.\n" << std::flush;
        } else {
            lBuf = ensureGPU(leftCtx, resolvedKeys[0].first);
            rBuf = ensureGPU(rightCtx, resolvedKeys[0].second);
        }

        if (!isCrossJoin && (!lBuf || !rBuf)) throw std::runtime_error("Missing GPU column data for Join");
        
        // Multi-match hash join; right=build, left=probe.
        
        if (debug) if (!isCrossJoin && debug) std::cerr << "[Exec] GPU Join: Build (" << rCount << "), Probe (" << lCount << ")\n";
        if (debug) {
            std::cerr << "[Exec] GPU Join: leftCtx.activeRowsGPU=" << (leftCtx.activeRowsGPU ? "SET" : "NULL") << " rightCtx.activeRowsGPU=" << (rightCtx.activeRowsGPU ? "SET" : "NULL") << "\n";
            if (leftCtx.activeRowsGPU) {
                uint32_t* leftIndices = (uint32_t*)leftCtx.activeRowsGPU->contents();
                if (debug) std::cerr << "[Exec] GPU Join: leftActiveIndices first 5: ";
                if (debug) for (uint32_t i = 0; i < std::min(5u, leftCtx.activeRowsCountGPU); ++i) std::cerr << leftIndices[i] << " ";
                if (debug) std::cerr << "\n";
            }
        }
        
        if (!isCrossJoin) {
            MTL::Buffer* buildActiveRows = rightCtx.activeRowsGPU;
            MTL::Buffer* probeActiveRows = leftCtx.activeRowsGPU;
            
            // GPU ANTI join shortcut: build HT from right, probe left, invert mask
            if (isAntiJoin && resolvedKeys.size() == 1) {
                if (join.rightVariant) {
                    // RIGHT ANTI: find right rows NOT matching left
                    auto antiRes = GpuOps::hashJoinAntiU32(rBuf, rCount, lBuf, lCount);
                    if (!antiRes) throw std::runtime_error("GPU hashJoinAntiU32 failed");
                    jRes.count = antiRes->count;
                    jRes.buildIndices = std::move(antiRes->indices);
                    jRes.probeIndices.reset(store.device()->newBuffer(
                        std::max(antiRes->count, 1u) * sizeof(uint32_t), MTL::ResourceStorageModeShared));
                    std::memset(jRes.probeIndices->contents(), 0,
                                std::max(antiRes->count, 1u) * sizeof(uint32_t));
                } else {
                    // LEFT ANTI: find left rows NOT matching right
                    auto antiRes = GpuOps::hashJoinAntiU32(lBuf, lCount, rBuf, rCount);
                    if (!antiRes) throw std::runtime_error("GPU hashJoinAntiU32 failed");
                    jRes.count = antiRes->count;
                    jRes.probeIndices = std::move(antiRes->indices);
                    jRes.buildIndices.reset(store.device()->newBuffer(
                        std::max(antiRes->count, 1u) * sizeof(uint32_t), MTL::ResourceStorageModeShared));
                    std::memset(jRes.buildIndices->contents(), 0,
                                std::max(antiRes->count, 1u) * sizeof(uint32_t));
                }
                if (debug) std::cerr << "[Exec] GPU Anti Join (direct): " << jRes.count << " non-matching rows\n";
            }
            // GPU SEMI join shortcut: build HT from right, probe left, compact matches
            else if (isSemiJoin && resolvedKeys.size() == 1) {
                auto semiRes = GpuOps::hashJoinSemiU32(lBuf, lCount, rBuf, rCount);
                if (!semiRes) throw std::runtime_error("GPU hashJoinSemiU32 failed");
                jRes.count = semiRes->count;
                jRes.probeIndices = std::move(semiRes->indices);
                // Build indices: use iota as placeholder (not needed for semi-only output,
                // but downstream might gather right columns for MARK join)
                jRes.buildIndices.reset(store.device()->newBuffer(
                    std::max(semiRes->count, 1u) * sizeof(uint32_t), MTL::ResourceStorageModeShared));
                std::memset(jRes.buildIndices->contents(), 0,
                            std::max(semiRes->count, 1u) * sizeof(uint32_t));
                if (debug) std::cerr << "[Exec] GPU Semi Join (direct): " << jRes.count << " matched left rows\n";
            }
            else if (resolvedKeys.size() == 2) {
                 jRes = GpuOps::joinHashU64(rBuf, buildActiveRows, rCount, lBuf, probeActiveRows, lCount);
                 lBuf->release(); rBuf->release();
            } else {
                 jRes = GpuOps::joinHash(rBuf, buildActiveRows, rCount, lBuf, probeActiveRows, lCount);
            }
            

        }
    } else {
        if (lCount > 0 && rCount == 0 && (isAntiJoin || isLeftJoin)) {
             if (debug) std::cerr << "[Exec] GPU Join: Empty Build side for Anti/Left Join -> Returning all " << lCount << " left rows.\n";
             jRes.count = lCount;
             
             if (leftCtx.activeRowsGPU) {
                 MTL::Buffer* src = leftCtx.activeRowsGPU;
                 jRes.probeIndices.reset(store.device()->newBuffer(src->contents(), src->length(), MTL::ResourceStorageModeShared));
             } else {
                 std::vector<uint32_t> seq(lCount);
                 std::iota(seq.begin(), seq.end(), 0);
                 jRes.probeIndices.reset(store.device()->newBuffer(seq.data(), seq.size() * sizeof(uint32_t), MTL::ResourceStorageModeShared));
             }
             
             // Placeholder build indices (required non-null)
             jRes.buildIndices.reset(store.device()->newBuffer(lCount * sizeof(uint32_t), MTL::ResourceStorageModeShared));
             std::memset(jRes.buildIndices->contents(), 0, lCount * sizeof(uint32_t));
        } else {
            if (debug) std::cerr << "[Exec] GPU Join: Skipping (Build=" << rCount << ", Probe=" << lCount << ")\n";
            jRes.count = 0;
            jRes.buildIndices = nullptr;
            jRes.probeIndices = nullptr;
        }
    }
                                       
    if ((lCount > 0 && rCount > 0) && !jRes.buildIndices) throw std::runtime_error("GPU Join Kernel Failed");
    
    if (debug) std::cerr << "[Exec] GPU Join Success: Result " << jRes.count << " rows.\n";
    
    uint32_t resCount = jRes.count;
    
    // Flag: skip CPU-based SEMI/ANTI post-processing if GPU shortcut was used
    bool gpuSemiAntiDone = (lCount > 0 && rCount > 0 && !isCrossJoin && resolvedKeys.size() == 1 &&
                            (isAntiJoin || isSemiJoin));
    
    // SEMI JOIN: deduplicate probeIndices, keeping first match per probe row.
    if (isSemiJoin && resCount > 0 && jRes.probeIndices && !gpuSemiAntiDone) {
        if (debug) std::cerr << "[Exec] Semi Join: Deduplicating " << resCount << " probe indices\n";
        
        // GPU dedup: dedupByKeys on probeIndices returns gather indices for unique probe values
        std::vector<MTL::Buffer*> dedupKeys = { jRes.probeIndices };
        uint32_t uniqueCount = 0;
        MTL::Buffer* uniqueIdx = GpuOps::dedupByKeys(dedupKeys, resCount, uniqueCount);
        
        if (uniqueIdx && uniqueCount > 0) {
            // Gather both probe and build indices using the dedup gather indices
            auto newProbe = GpuOps::gatherU32(jRes.probeIndices, uniqueIdx, uniqueCount);
            auto newBuild = GpuOps::gatherU32(jRes.buildIndices, uniqueIdx, uniqueCount);
            uniqueIdx->release();
            
            if (newProbe && newBuild) {
                jRes.probeIndices.reset(newProbe);
                jRes.buildIndices.reset(newBuild);
                resCount = uniqueCount;
                jRes.count = uniqueCount;
            }
        } else if (uniqueIdx) {
            uniqueIdx->release();
        }
        
        if (debug) std::cerr << "[Exec] Semi Join: After dedup: " << resCount << " unique rows\n";
    }
    
    // ANTI JOIN: Find rows that did NOT match
    // For LEFT ANTI (default): keep LHS rows not matching RHS
    // For RIGHT ANTI (rightVariant): keep RHS rows not matching LHS
    // Skip when rCount==0: the empty-build fast path already returns all left rows directly.
    // Skip when gpuSemiAntiDone: GPU anti join already computed the result.
    if (isAntiJoin && rCount > 0 && jRes.probeIndices && !gpuSemiAntiDone) {
        if (join.rightVariant) {
            // RIGHT ANTI: find build (right) rows that were NOT matched
            if (debug) std::cerr << "[Exec] Right Anti Join: Finding non-matching rows from " << rCount << " right rows, " << resCount << " matches\n";
            
            // GPU: findUnmatchedIndices on build side
            auto antiRes = GpuOps::findUnmatchedIndices(jRes.buildIndices, resCount, rCount);
            uint32_t antiCount = antiRes.count;
            if (debug) std::cerr << "[Exec] Right Anti Join: " << antiCount << " non-matching right rows\n";
            
            // For RIGHT ANTI, we only gather from the RIGHT (build) side
            jRes.buildIndices = std::move(antiRes.indices);
            
            jRes.probeIndices.reset(store.device()->newBuffer(
                std::max(antiCount, 1u) * sizeof(uint32_t), MTL::ResourceStorageModeShared));
            std::memset(jRes.probeIndices->contents(), 0, std::max(antiCount, 1u) * sizeof(uint32_t));
            
            resCount = antiCount;
            jRes.count = antiCount;
        } else {
            // LEFT ANTI (default): find probe (left) rows that were NOT matched
            if (debug) std::cerr << "[Exec] Anti Join: Finding non-matching rows from " << lCount << " left rows, " << resCount << " matches\n";
        
        // GPU: findUnmatchedIndices on probe side
        auto antiRes = GpuOps::findUnmatchedIndices(jRes.probeIndices, resCount, lCount);
        uint32_t antiCount = antiRes.count;
        if (debug) std::cerr << "[Exec] Anti Join: " << antiCount << " non-matching rows\n";
        
        // Replace with anti-join result
        jRes.probeIndices = std::move(antiRes.indices);
        
        jRes.buildIndices.reset(store.device()->newBuffer(
            std::max(antiCount, 1u) * sizeof(uint32_t), MTL::ResourceStorageModeShared));
        std::memset(jRes.buildIndices->contents(), 0, std::max(antiCount, 1u) * sizeof(uint32_t));
        
        resCount = antiCount;
        jRes.count = antiCount;
        } // end LEFT ANTI else
    }
    
    // RIGHT ANTI: Only gather from right side (the build side)
    bool rightAntiGather = (isAntiJoin && join.rightVariant);
    
    // Scatter/gather all columns into output context
    std::unordered_map<std::string, std::string> rightColumnMapping;
    bool earlyReturn = scatterJoinOutputColumns(
        leftCtx, rightCtx, outCtx, jRes, resCount, lCount, rCount,
        isAntiJoin, isSemiJoin, rightAntiGather, rightColumnMapping, debug);
    if (earlyReturn) return true;

    if (debug) {
        std::cerr << "[Exec] Join: After string gather, outCtx.stringCols sizes:\n";
        for (const auto& [name, vec] : outCtx.stringCols) {
            std::cerr << "[Exec]   stringCol " << name << " size=" << vec.size() << "\n";
        }
    }

    // LEFT JOIN: Append unmatched left rows with NULL/0 for right columns
    if (isLeftJoin && rCount > 0 && resCount > 0) {
        appendUnmatchedLeftRows(leftCtx, outCtx, jRes, resCount, lCount, rightColumnMapping, debug);
    }

    // RIGHT JOIN: Append unmatched right (build) rows with NULL/0 for left columns
    if (isRightJoin && rCount > 0 && resCount > 0) {
        appendUnmatchedRightRows(rightCtx, outCtx, jRes, resCount, rCount, rightColumnMapping, debug);
    }

    // jRes goes out of scope — GpuBuffer handles release
    
    // If the right side had DELIM correlation columns that were renamed due to collision,
    // swap the data so the correlation columns keep their primary (un-suffixed) names.
    // This ensures subsequent projections pick up the DELIM_SCAN correlation keys
    // instead of the lineitem instance columns.
    if (!rightCtx.isDelimCorrelation.empty()) {
        for (const auto& col : rightCtx.isDelimCorrelation) {
            auto rmIt = rightColumnMapping.find(col);
            if (rmIt != rightColumnMapping.end() && rmIt->second != col) {
                const std::string& renamedCol = rmIt->second;
                // Swap u32 CPU data
                if (outCtx.u32Cols.count(col) && outCtx.u32Cols.count(renamedCol)) {
                    std::swap(outCtx.u32Cols[col], outCtx.u32Cols[renamedCol]);
                    if (debug) std::cerr << "[Exec] Join: DELIM priority swap: " << col << " <-> " << renamedCol << "\n";
                }
                // Swap u32 GPU buffers
                if (outCtx.u32ColsGPU.count(col) && outCtx.u32ColsGPU.count(renamedCol)) {
                    std::swap(outCtx.u32ColsGPU[col], outCtx.u32ColsGPU[renamedCol]);
                }
                // Swap f32 CPU data
                if (outCtx.f32Cols.count(col) && outCtx.f32Cols.count(renamedCol)) {
                    std::swap(outCtx.f32Cols[col], outCtx.f32Cols[renamedCol]);
                }
                // Swap f32 GPU buffers
                if (outCtx.f32ColsGPU.count(col) && outCtx.f32ColsGPU.count(renamedCol)) {
                    std::swap(outCtx.f32ColsGPU[col], outCtx.f32ColsGPU[renamedCol]);
                }
                // Swap dict columns
                if (outCtx.dictCols.count(col) && outCtx.dictCols.count(renamedCol)) {
                    std::swap(outCtx.dictCols[col], outCtx.dictCols[renamedCol]);
                }
                // Swap string columns
                if (outCtx.stringCols.count(col) && outCtx.stringCols.count(renamedCol)) {
                    std::swap(outCtx.stringCols[col], outCtx.stringCols[renamedCol]);
                }
            }
        }
        // Clear the marker so subsequent joins don't re-swap
        outCtx.isDelimCorrelation.clear();
    }
    
    // Apply post-join filter for non-equality conditions (e.g., l_suppkey != l_suppkey)
    // Use the regular filter pipeline which handles col-vs-col, col-vs-scalar, etc.
    if (hasPostJoinFilter && postJoinFilter && outCtx.rowCount > 0) {
        applyPostJoinFilter(postJoinFilter, leftCtx, rightCtx, rightColumnMapping, outCtx, debug);
    }
    
    // Rebuild flat string buffers and dictionary encoding for downstream operators
    // Skip columns that already have a valid dictCol (GPU-native path)
    if (outCtx.rowCount > 0) {
        for (const auto& [name, vec] : outCtx.stringCols) {
            if (!vec.empty() && !outCtx.hasDictCol(name)) {
                flattenStringCol(outCtx, name);
                buildDictCol(outCtx, name);
            }
        }
    }
    
    return true;
}


// -- Extracted: handleScalarSubquerySavedPipelines --
// Handles scalar SUBQUERY join when savedPipelines contains main data.
// Returns true if handled (caller should return).
static bool handleScalarSubquerySavedPipelines(
    const IRJoin& join, EvalContext& currentCtx,
    std::vector<EvalContext>& savedPipelines,
    std::vector<std::set<std::string>>& savedPipelineTables,
    std::set<std::string>& joinedTables,
    GpuExecutor::ExecutionResult& result, bool debug) {
    if (debug) {
        std::cerr << "[Exec] Join: detected scalar subquery pattern\n";
        std::cerr << "[Exec]   Current context rows: " << currentCtx.rowCount << "\n";
        std::cerr << "[Exec]   Saved pipelines: " << savedPipelines.size() << "\n";
    }

    // Determine if scalar is in currentCtx or savedPipelines
    double scalarValue = 0.0;
    bool foundScalar = false;
    bool scalarIsInCurrent = (currentCtx.rowCount == 1);

    int groupedPipelineIdx = -1;
    int scalarPipelineIdx = -1;

    if (!scalarIsInCurrent) {
        // Check if scalar is in savedPipelines
        for (size_t pi = 0; pi < savedPipelines.size(); ++pi) {
            if (savedPipelines[pi].rowCount == 1 || savedPipelines[pi].isScalarResult) {
                scalarPipelineIdx = static_cast<int>(pi);
                if (debug && savedPipelines[pi].isScalarResult) {
                    std::cerr << "[Exec]   Found scalar pipeline via flag (rowCount=" << savedPipelines[pi].rowCount << ")\n";
                }
                break;
            }
        }
    } else {
        // Scalar is current. Find grouped pipeline in saved
        for (size_t pi = 0; pi < savedPipelineTables.size(); ++pi) {
             if (savedPipelineTables[pi].count("__GROUPED__") > 0 || savedPipelines[pi].rowCount > 1) {
                 groupedPipelineIdx = static_cast<int>(pi);
                 break;
             }
        }
    }

    const EvalContext* scalarCtx = nullptr;
    if (scalarIsInCurrent) {
         scalarCtx = &currentCtx;
    } else if (scalarPipelineIdx >= 0) {
         scalarCtx = &savedPipelines[scalarPipelineIdx];
    }

    if (!scalarCtx) {
        result.error = "Scalar subquery join: could not locate scalar value source (neither current inputs nor saved pipelines seem correct)";
        return false;
    }

    // Extract scalar from scalarCtx
    // Priority: #0, then SUM/AVG, then any
    auto tryExtract = [&](const std::string& pattern, bool exact) -> bool {
         // Search f32
         for (const auto& [name, values] : scalarCtx->f32Cols) {
             if (values.empty()) continue;
             bool match = exact ? (name == pattern) : (name.find(pattern) != std::string::npos);
             if (match) {
                 scalarValue = values[0];
                 if (debug) std::cerr << "[Exec]   Scalar value from f32 col '" << name << "': " << scalarValue << "\n";
                 return true;
             }
         }
         // Search u32
         for (const auto& [name, values] : scalarCtx->u32Cols) {
             if (values.empty()) continue;
             bool match = exact ? (name == pattern) : (name.find(pattern) != std::string::npos);
             if (match) {
                 scalarValue = static_cast<double>(values[0]);
                 if (debug) std::cerr << "[Exec]   Scalar value from u32 col '" << name << "': " << scalarValue << "\n";
                 return true;
             }
         }
         return false;
    };

    if (!foundScalar) foundScalar = tryExtract("#0", true);
    // Also check for #0 in u32 (some DBs output integer counts)
    if (!foundScalar) foundScalar = tryExtract("SUM", false);
    if (!foundScalar) foundScalar = tryExtract("AVG", false);
    if (!foundScalar) foundScalar = tryExtract("first", false);

    // Fallback to any
    if (!foundScalar) foundScalar = tryExtract("", false);

    if (!foundScalar) {
        result.error = "Scalar subquery join: could not find scalar value";
        return false;
    }

    // Capture input scalars (e.g. CASE, Aggregates) to broadcast.
    std::map<std::string, float> scalarF32s;
    std::map<std::string, uint32_t> scalarU32s;
    if (scalarCtx) {
         for(auto& [n, v] : scalarCtx->f32Cols) if(!v.empty()) scalarF32s[n] = v[0];
         for(auto& [n, v] : scalarCtx->u32Cols) if(!v.empty()) scalarU32s[n] = v[0];
    }

    // Prepare the Data (Grouped) Pipeline
    if (scalarIsInCurrent) {
        if (groupedPipelineIdx < 0) {
            result.error = "Scalar subquery join: could not find grouped pipeline";
            return false;
        }
        // Restore saved pipeline
        currentCtx = savedPipelines[groupedPipelineIdx];
        joinedTables = savedPipelineTables[groupedPipelineIdx];
        joinedTables.erase("__GROUPED__");

        savedPipelines.erase(savedPipelines.begin() + groupedPipelineIdx);
        savedPipelineTables.erase(savedPipelineTables.begin() + groupedPipelineIdx);

        if (debug) {
            std::cerr << "[Exec]   Restored saved pipeline with " << currentCtx.rowCount << " rows\n";
        }
    } else {
        // Data is already currentCtx. Just remove the scalar pipeline from saved.
        if (scalarPipelineIdx >= 0) {
            savedPipelines.erase(savedPipelines.begin() + scalarPipelineIdx);
            savedPipelineTables.erase(savedPipelineTables.begin() + scalarPipelineIdx);
        }
        if (debug) {
            std::cerr << "[Exec]   Using current context as data table with " << currentCtx.rowCount << " rows\n";
        }
    }

    // Inject broadcasted scalars into the data context
    for(auto& [n, v] : scalarF32s) {
        if (currentCtx.f32Cols.find(n) == currentCtx.f32Cols.end() && currentCtx.f32ColsGPU.find(n) == currentCtx.f32ColsGPU.end()) {
             currentCtx.f32Cols[n] = {v}; // Size 1 vector (scalar broadcast)
             if (debug) std::cerr << "[Exec]   Broadcasted scalar F32col: " << n << "\n";
        }
    }
    for(auto& [n, v] : scalarU32s) {
        if (currentCtx.u32Cols.find(n) == currentCtx.u32Cols.end() && currentCtx.u32ColsGPU.find(n) == currentCtx.u32ColsGPU.end()) {
             currentCtx.u32Cols[n] = {v};
             if (debug) std::cerr << "[Exec]   Broadcasted scalar U32col: " << n << "\n";
        }
    }

    // Parse condition to extract comparison column and operator
    std::string condStr = join.conditionStr;

    // Find the comparison operator
    size_t opPos = std::string::npos;
    std::string opStr;
    engine::GpuFilterOp compOp = engine::GpuFilterOp::EQ;
    if ((opPos = condStr.find(" > SUBQUERY")) != std::string::npos) {
        opStr = ">";
        compOp = engine::GpuFilterOp::GT;
    } else if ((opPos = condStr.find(" >= SUBQUERY")) != std::string::npos) {
        opStr = ">=";
        compOp = engine::GpuFilterOp::GE;
    } else if ((opPos = condStr.find(" < SUBQUERY")) != std::string::npos) {
        opStr = "<";
        compOp = engine::GpuFilterOp::LT;
    } else if ((opPos = condStr.find(" <= SUBQUERY")) != std::string::npos) {
        opStr = "<=";
        compOp = engine::GpuFilterOp::LE;
    } else if ((opPos = condStr.find(" = SUBQUERY")) != std::string::npos) {
        opStr = "=";
        compOp = engine::GpuFilterOp::EQ;
    }

    if (opPos == std::string::npos) {
        result.error = "Scalar subquery join: unsupported comparison operator in condition: " + condStr;
        return false;
    }

    // Extract the column/expression being compared
    std::string leftExpr = condStr.substr(0, opPos);
    // Trim
    while (!leftExpr.empty() && std::isspace(leftExpr.back())) leftExpr.pop_back();

    // Find matching aggregate column in context
    std::string aggColName;

    // First check if we have #1 (typical aggregate position)
    if (currentCtx.f32Cols.find("#1") != currentCtx.f32Cols.end()) {
        aggColName = "#1";
    } else if (currentCtx.f32Cols.find("SUM_#1") != currentCtx.f32Cols.end()) {
        aggColName = "SUM_#1";
    } else if (currentCtx.u32Cols.find("#1") != currentCtx.u32Cols.end()) {
        aggColName = "#1";
    } else {
        // Look for any aggregate column
        for (const auto& [name, vals] : currentCtx.f32Cols) {
            if (name.find("SUM") != std::string::npos || 
                name.find("AVG") != std::string::npos ||
                name.find("COUNT") != std::string::npos ||
                name[0] == '#') {
                aggColName = name;
                break;
            }
        }
    }

    if (aggColName.empty()) {
        result.error = "Scalar subquery join: could not find aggregate column";
        return false;
    }

    if (debug) {
        std::cerr << "[Exec]   Filtering: " << aggColName << " " << opStr << " " << scalarValue << "\n";
    }

    // Apply scalar subquery filter on GPU
    // 1. Ensure data columns are uploaded to GPU
    auto device = GpuColumnStore::instance().device();
    for (auto& [name, vec] : currentCtx.f32Cols) {
        if (currentCtx.f32ColsGPU.find(name) == currentCtx.f32ColsGPU.end()) {
            auto buf = device->newBuffer(vec.data(), vec.size() * sizeof(float), MTL::ResourceStorageModeShared);
            if (buf) currentCtx.f32ColsGPU[name].reset(buf);
        }
    }
    for (auto& [name, vec] : currentCtx.u32Cols) {
        if (currentCtx.u32ColsGPU.find(name) == currentCtx.u32ColsGPU.end()) {
            auto buf = device->newBuffer(vec.data(), vec.size() * sizeof(uint32_t), MTL::ResourceStorageModeShared);
            if (buf) currentCtx.u32ColsGPU[name].reset(buf);
        }
    }

    // 2. Build a TypedExpr comparison predicate: aggColName <op> scalarValue
    CompareOp typedOp = CompareOp::Eq;
    switch (compOp) {
        case engine::GpuFilterOp::GT: typedOp = CompareOp::Gt; break;
        case engine::GpuFilterOp::GE: typedOp = CompareOp::Ge; break;
        case engine::GpuFilterOp::LT: typedOp = CompareOp::Lt; break;
        case engine::GpuFilterOp::LE: typedOp = CompareOp::Le; break;
        case engine::GpuFilterOp::EQ: typedOp = CompareOp::Eq; break;
        case engine::GpuFilterOp::NE: typedOp = CompareOp::Ne; break;
        default: break;
    }
    auto filterPred = TypedExpr::compare(
        typedOp,
        TypedExpr::column(aggColName),
        TypedExpr::literal(static_cast<double>(scalarValue))
    );

    // 3. Execute GPU filter
    if (!GpuExecutor::executeFilterRecursive(filterPred, currentCtx)) {
        result.error = "Scalar subquery join: GPU filter failed for " + aggColName;
        return false;
    }

    // 4. Materialize: compact all columns using activeRowsGPU
    if (currentCtx.activeRowsGPU && currentCtx.activeRowsCountGPU > 0) {
        uint32_t count = currentCtx.activeRowsCountGPU;
        uint32_t* indices = (uint32_t*)currentCtx.activeRowsGPU->contents();

        // Compact GPU columns
        for (auto& [name, buf] : currentCtx.u32ColsGPU) {
            if (!buf) continue;
            uint32_t bufRows = (uint32_t)(buf->length() / sizeof(uint32_t));
            if (bufRows > count) {
                auto compacted = GpuOps::gatherU32(buf, currentCtx.activeRowsGPU, count, true);
                if (compacted) buf.reset(compacted);
            }
        }
        for (auto& [name, buf] : currentCtx.f32ColsGPU) {
            if (!buf) continue;
            uint32_t bufRows = (uint32_t)(buf->length() / sizeof(float));
            if (bufRows > count) {
                auto compacted = GpuOps::gatherF32(buf, currentCtx.activeRowsGPU, count, true);
                if (compacted) buf.reset(compacted);
            }
        }
        // Compact CPU columns: sync from GPU if possible, else CPU gather
        for (auto& [name, vec] : currentCtx.u32Cols) {
            if (vec.size() > count) {
                if (currentCtx.u32ColsGPU.count(name) && currentCtx.u32ColsGPU[name]) {
                    vec.resize(count);
                    std::memcpy(vec.data(), currentCtx.u32ColsGPU[name]->contents(), count * sizeof(uint32_t));
                } else {
                    std::vector<uint32_t> c;
                    c.reserve(count);
                    for (uint32_t i = 0; i < count; ++i)
                        c.push_back(indices[i] < (uint32_t)vec.size() ? vec[indices[i]] : 0u);
                    vec = std::move(c);
                }
            }
        }
        for (auto& [name, vec] : currentCtx.f32Cols) {
            if (vec.size() > count) {
                if (currentCtx.f32ColsGPU.count(name) && currentCtx.f32ColsGPU[name]) {
                    vec.resize(count);
                    std::memcpy(vec.data(), currentCtx.f32ColsGPU[name]->contents(), count * sizeof(float));
                } else {
                    std::vector<float> c;
                    c.reserve(count);
                    for (uint32_t i = 0; i < count; ++i)
                        c.push_back(indices[i] < (uint32_t)vec.size() ? vec[indices[i]] : 0.0f);
                    vec = std::move(c);
                }
            }
        }

        if (currentCtx.activeRowsGPU) { currentCtx.activeRowsGPU = nullptr; }
        currentCtx.activeRowsCountGPU = 0;
        currentCtx.activeRows.clear();
        currentCtx.rowCount = count;
    } else {
        // No rows matched
        currentCtx.rowCount = 0;
        currentCtx.activeRows.clear();
        currentCtx.activeRowsGPU = nullptr;
        currentCtx.activeRowsCountGPU = 0;
    }

    // Reset scalar aggregate flag - we now have a proper table result
    result.isScalarAggregate = false;

    if (debug) {
        std::cerr << "[Exec]   After scalar filter: " << currentCtx.rowCount << " rows\n";
    }

    // Don't do the normal join - we've handled this specially
    return true;
    return false;
}

// -- Extracted: handleScalarSubqueryTableContexts --
// Handles scalar SUBQUERY join via tableContexts (theta-comparison).
// Returns true if handled (caller should return).
static bool handleScalarSubqueryTableContexts(
    const IRJoin& join, EvalContext& currentCtx,
    std::unordered_map<std::string, EvalContext>& tableContexts,
    std::set<std::string>& joinedTables, bool& hasPipeline,
    GpuExecutor::ExecutionResult& result, bool debug) {
    // Check if this is a theta-comparison (>, <, >=, <=) with SUBQUERY
    std::string condStr = join.conditionStr;
    size_t opPos = std::string::npos;
    std::string opStr;
    bool isTheta = false;

    if ((opPos = condStr.find(" > SUBQUERY")) != std::string::npos) {
        opStr = ">"; isTheta = true;
    } else if ((opPos = condStr.find(" >= SUBQUERY")) != std::string::npos) {
        opStr = ">="; isTheta = true;
    } else if ((opPos = condStr.find(" < SUBQUERY")) != std::string::npos) {
        opStr = "<"; isTheta = true;
    } else if ((opPos = condStr.find(" <= SUBQUERY")) != std::string::npos) {
        opStr = "<="; isTheta = true;
    } else if ((opPos = condStr.find(" = SUBQUERY")) != std::string::npos) {
        opStr = "="; isTheta = true;
    }

    if (isTheta && currentCtx.rowCount <= 1) {
        if (debug) {
            std::cerr << "[Exec] Join: scalar SUBQUERY theta-join (tableContexts path)\n";
            std::cerr << "[Exec]   Current context rows: " << currentCtx.rowCount << "\n";
        }

        // Extract scalar value from currentCtx
        double scalarValue = 0.0;
        bool foundScalar = false;

        // Priority 1: Explicit AVG column (common for scalar subquery).
        auto avgIt = currentCtx.f32Cols.find("AVG");
        if (avgIt != currentCtx.f32Cols.end() && !avgIt->second.empty()) {
            scalarValue = avgIt->second[0];
            foundScalar = true;
            if (debug) {
                std::cerr << "[Exec]   Scalar value from 'AVG': " << scalarValue << "\n";
            }
        }

        // Priority 2: Look for SUM column
        if (!foundScalar) {
            auto sumIt = currentCtx.f32Cols.find("SUM");
            if (sumIt != currentCtx.f32Cols.end() && !sumIt->second.empty()) {
                scalarValue = sumIt->second[0];
                foundScalar = true;
                if (debug) {
                    std::cerr << "[Exec]   Scalar value from 'SUM': " << scalarValue << "\n";
                }
            }
        }

        // Priority 3: #0 (first computed column, scalar result).
        if (!foundScalar) {
            auto numIt = currentCtx.f32Cols.find("#0");
            if (numIt != currentCtx.f32Cols.end() && !numIt->second.empty()) {
                scalarValue = numIt->second[0];
                foundScalar = true;
                if (debug) {
                    std::cerr << "[Exec]   Scalar value from '#0': " << scalarValue << "\n";
                }
            }
        }

        // Fallback: any f32 column except COUNT
        if (!foundScalar) {
            for (const auto& [name, values] : currentCtx.f32Cols) {
                if (!values.empty() && name.find("COUNT") == std::string::npos) {
                    scalarValue = values[0];
                    foundScalar = true;
                    if (debug) {
                        std::cerr << "[Exec]   Scalar value fallback from '" << name << "': " << scalarValue << "\n";
                    }
                    break;
                }
            }
        }

        if (!foundScalar) {
            if (debug) std::cerr << "[Exec]   Could not find scalar value\n";
            result.error = "Scalar SUBQUERY join: could not extract scalar value";
            return false;
        }

        // Find the data table - the one containing the comparison column
        // Parse column from condition (e.g., "CAST(c_acctbal AS DOUBLE)" -> c_acctbal)
        std::string leftExpr = condStr.substr(0, opPos);
        // Extract column name from CAST or direct reference
        std::string filterCol;
        if (leftExpr.find("CAST(") != std::string::npos) {
            size_t start = leftExpr.find("CAST(") + 5;
            size_t end = leftExpr.find(" AS", start);
            if (end != std::string::npos) {
                filterCol = leftExpr.substr(start, end - start);
                // Trim
                while (!filterCol.empty() && std::isspace(filterCol.front())) filterCol.erase(0, 1);
                while (!filterCol.empty() && std::isspace(filterCol.back())) filterCol.pop_back();
            }
        }
        if (filterCol.empty()) {
            filterCol = leftExpr;
            while (!filterCol.empty() && std::isspace(filterCol.front())) filterCol.erase(0, 1);
            while (!filterCol.empty() && std::isspace(filterCol.back())) filterCol.pop_back();
        }

        if (debug) {
            std::cerr << "[Exec]   Filter column: " << filterCol << "\n";
        }

        // Find the table with this column in tableContexts
        std::string dataTable;
        for (const auto& [tname, tctx] : tableContexts) {
            if (tctx.f32Cols.find(filterCol) != tctx.f32Cols.end() ||
                tctx.u32Cols.find(filterCol) != tctx.u32Cols.end()) {
                // Check for suffixed versions too
                if (joinedTables.find(tname) == joinedTables.end()) {
                    dataTable = tname;
                    break;
                }
            }
            // Try with suffix
            for (const auto& [cname, cvals] : tctx.f32Cols) {
                if ((cname == filterCol || cname.find(filterCol + "_") == 0 || 
                     cname.rfind("_" + filterCol) == cname.size() - filterCol.size() - 1) &&
                    joinedTables.find(tname) == joinedTables.end()) {
                    dataTable = tname;
                    filterCol = cname;  // Use actual column name
                    break;
                }
            }
            if (!dataTable.empty()) break;
        }

        if (dataTable.empty()) {
            if (debug) std::cerr << "[Exec]   Could not find data table\n";
            result.error = "Scalar SUBQUERY join: could not find data table";
            return false;
        }

        if (debug) {
            std::cerr << "[Exec]   Data table: " << dataTable << " with " 
                      << tableContexts[dataTable].rowCount << " rows\n";
        }

        // Apply the filter: col <op> scalarValue
        EvalContext& dataCtx = tableContexts[dataTable];
        std::vector<uint32_t> passingIndices;

        auto it = dataCtx.f32Cols.find(filterCol);
        if (it == dataCtx.f32Cols.end()) {
            // Try suffixed versions
            for (const auto& [cname, cvals] : dataCtx.f32Cols) {
                if (cname.find(filterCol) != std::string::npos) {
                    it = dataCtx.f32Cols.find(cname);
                    filterCol = cname;
                    break;
                }
            }
        }

        if (it != dataCtx.f32Cols.end()) {
            // Valid column to filter
            auto& store = GpuColumnStore::instance();

            // Ensure column is on GPU
            MTL::Buffer* colBuf = nullptr;
            if (dataCtx.f32ColsGPU.count(filterCol)) {
                colBuf = dataCtx.f32ColsGPU[filterCol];
            } else {
                // Upload (Lazy)
                const auto& vec = it->second;
                colBuf = store.device()->newBuffer(vec.data(), vec.size() * sizeof(float), MTL::ResourceStorageModeShared);
                dataCtx.f32ColsGPU[filterCol].reset(colBuf);
            }

            // Map Op
            engine::GpuFilterOp op = engine::GpuFilterOp::EQ;
            if (opStr == ">") op = engine::GpuFilterOp::GT;
            else if (opStr == ">=") op = engine::GpuFilterOp::GE;
            else if (opStr == "<") op = engine::GpuFilterOp::LT;
            else if (opStr == "<=") op = engine::GpuFilterOp::LE;
            else if (opStr == "=") op = engine::GpuFilterOp::EQ;

            std::optional<FilterResult> filterRes;
            if (dataCtx.activeRowsGPU) {
                 filterRes = GpuOps::filterF32Indexed(filterCol, colBuf, dataCtx.activeRowsGPU, dataCtx.activeRowsCountGPU, op, static_cast<float>(scalarValue));
            } else {
                 filterRes = GpuOps::filterF32(filterCol, colBuf, dataCtx.rowCount, op, static_cast<float>(scalarValue));
            }

            if (!filterRes) throw std::runtime_error("GPU Scalar Filter failed");

            MTL::Buffer* indices = filterRes->indices;
            uint32_t newCount = filterRes->count;

            // Download indices for CPU String sync
            std::vector<uint32_t> passingIndices(newCount);
            if (newCount > 0) {
                std::memcpy(passingIndices.data(), indices->contents(), newCount * sizeof(uint32_t));
            }

            // Safe Gather for U32 (preserving aliases and avoiding double-free)
            std::unordered_map<MTL::Buffer*, MTL::Buffer*> u32Replacements;
            for (auto& [name, buf] : dataCtx.u32ColsGPU) {
                if (buf && u32Replacements.find(buf) == u32Replacements.end()) {
                    u32Replacements[buf] = GpuOps::gatherU32(buf, indices, newCount);
                }
            }
            // Update map with new buffers
            for (auto& [name, buf] : dataCtx.u32ColsGPU) {
                if (buf) {
                    MTL::Buffer* newBuf = u32Replacements[buf];
                    newBuf->retain(); 
                    buf.reset(newBuf); 
                }
            }
            // Consume creation refs of new buffers (old buffers already released by GpuBuffer::reset)
            for (auto& [_, newBuf] : u32Replacements) {
                newBuf->release(); 
            }

            // Safe Gather for F32
            std::unordered_map<MTL::Buffer*, MTL::Buffer*> f32Replacements;
            for (auto& [name, buf] : dataCtx.f32ColsGPU) {
                if (buf && f32Replacements.find(buf.get()) == f32Replacements.end()) {
                    f32Replacements[buf.get()] = GpuOps::gatherF32(buf, indices, newCount);
                }
            }
            for (auto& [name, buf] : dataCtx.f32ColsGPU) {
                if (buf) {
                    MTL::Buffer* newBuf = f32Replacements[buf.get()];
                    newBuf->retain();
                    buf.reset(newBuf);
                }
            }
            for (auto& [_, newBuf] : f32Replacements) {
                newBuf->release();
            }

            // Handle strings on CPU (fallback when dict/flat not available)
            for (auto& [name, vals] : dataCtx.stringCols) {
                if (dataCtx.dictCols.count(name)) continue; // dict path below
                if (dataCtx.flatStringCols.count(name)) continue; // flat path below
                std::vector<std::string> compacted;
                compacted.reserve(passingIndices.size());
                for (uint32_t idx : passingIndices) {
                    if (idx < vals.size()) compacted.push_back(vals[idx]);
                    else compacted.push_back("");
                }
                vals = std::move(compacted);
            }

            // GPU gather for dict and flat string columns
            dataCtx.compactDictCols(indices, newCount);
            dataCtx.compactFlatStringCols(indices, newCount);
            dataCtx.invalidateStringColsForDictFlat();

            // Update Context
            dataCtx.rowCount = newCount;

            dataCtx.clearActiveRows();

            // Clear CPU vectors to enforce GPU usage
            for(auto& [n, v] : dataCtx.u32Cols) v.clear(); 
            for(auto& [n, v] : dataCtx.f32Cols) v.clear();
        }

        // Switch currentCtx to the filtered data table
        currentCtx = dataCtx;
        joinedTables.clear();
        joinedTables.insert(dataTable);
        hasPipeline = true;

        return true;  // Handled this join
    }
    return false;
}

// -- Extracted: dedupDelimJoinRHS --
// Deduplicates RHS for DELIM_JOIN self-comparison patterns.
static void dedupDelimJoinRHS(
    const IRJoin& join, EvalContext& currentCtx, EvalContext& rightCtx, bool debug) {
    // Extract all self-comparison key columns from the condition
    std::vector<std::string> delimDedupKeys;
    bool hasINDFKey = false; // Has "IS NOT DISTINCT FROM" pattern

    auto parts = splitConditionByAnd(join.conditionStr);
    for (const auto& part : parts) {
        bool isINDF = false;
        std::string col = parseSelfComparison(part, &isINDF);
        if (!col.empty()) {
            delimDedupKeys.push_back(col);
            hasINDFKey = hasINDFKey || isINDF;
        }
    }

    // Only apply RHS dedup when:
    // 1. IS NOT DISTINCT FROM conditions (standard DELIM_JOIN marker), OR
    // 2. For "=" self-comparison: only when the left side has fewer rows
    //    (indicating it's a DELIM correlation join, not a regular inner
    //    join inside a DELIM subquery). Without this check, regular joins
    //    like "l_orderkey = l_orderkey AND l_suppkey != l_suppkey" inside
    //    EXISTS subqueries would incorrectly dedup the build side by a
    //    subset of keys, losing correlation information (Q21 bug).
    bool shouldDedup = !delimDedupKeys.empty() && !join.rightTable.empty() && rightCtx.rowCount > 1;
    if (shouldDedup && !hasINDFKey) {
        // Non-INDF self-comparison: only dedup when left has fewer rows
        // (DELIM correlation pattern: small subquery result on left,
        // large original context on right needing dedup)
        shouldDedup = (currentCtx.rowCount < rightCtx.rowCount);
    }
    if (shouldDedup) {
        // Compact rightCtx GPU buffers if activeRowsGPU is set (e.g., from
        // post-join filter). Without this, GPU buffers have more elements
        // than rowCount and the first rowCount elements are NOT the valid
        // filtered rows — the dedup would operate on wrong data.
        if (rightCtx.activeRowsGPU && rightCtx.activeRowsCountGPU > 0) {
            uint32_t compactCount = rightCtx.activeRowsCountGPU;
            if (debug) {
                std::cerr << "[Exec] Join: DELIM dedup: compacting rightCtx GPU buffers via activeRowsGPU ("
                          << compactCount << " active rows)\n";
            }
            rightCtx.gatherAllGPU(rightCtx.activeRowsGPU, compactCount);
            // NOTE: syncCPUFromGPU removed — unified memory means
            // ->contents() already provides zero-cost CPU access.
            // Downstream code uses GPU buffers directly.
            rightCtx.clearActiveRows();
            rightCtx.rowCount = compactCount;
        }
        // Resolve key columns from GPU buffers (avoid CPU materialization)
        std::vector<std::string> resolvedKeys;
        std::vector<MTL::Buffer*> gpuKeys;
        for (const auto& k : delimDedupKeys) {
            // Try direct name in GPU
            if (rightCtx.u32ColsGPU.count(k) && rightCtx.u32ColsGPU[k]) {
                resolvedKeys.push_back(k);
                gpuKeys.push_back(rightCtx.u32ColsGPU[k]);
                continue;
            }
            bool found = false;
            // Try suffixed names
            for (int s = 1; s <= 9; ++s) {
                std::string sk = k + "_" + std::to_string(s);
                if (rightCtx.u32ColsGPU.count(sk) && rightCtx.u32ColsGPU[sk]) {
                    resolvedKeys.push_back(sk);
                    gpuKeys.push_back(rightCtx.u32ColsGPU[sk]);
                    found = true;
                    break;
                }
            }
            // Try prefix swap
            if (!found && k.size() > 2 && k[1] == '_') {
                std::string suffix = k.substr(2);
                for (const auto& p : {"l_", "o_", "c_", "p_", "s_", "ps_", "n_", "r_"}) {
                    std::string alt = std::string(p) + suffix;
                    if (rightCtx.u32ColsGPU.count(alt) && rightCtx.u32ColsGPU[alt]) {
                        resolvedKeys.push_back(alt);
                        gpuKeys.push_back(rightCtx.u32ColsGPU[alt]);
                        found = true;
                        break;
                    }
                }
            }
            // Fallback: upload from CPU
            if (!found) {
                auto tryUpload = [&](const std::string& name) -> bool {
                    if (rightCtx.u32Cols.count(name) && rightCtx.u32Cols.at(name).size() == rightCtx.rowCount) {
                        MTL::Buffer* buf = GpuOps::createBuffer(rightCtx.u32Cols[name].data(),
                                                                rightCtx.rowCount * sizeof(uint32_t));
                        rightCtx.u32ColsGPU[name].reset(buf);
                        resolvedKeys.push_back(name);
                        gpuKeys.push_back(buf);
                        return true;
                    }
                    return false;
                };
                if (!tryUpload(k)) {
                    for (int s = 1; s <= 9 && !found; ++s) {
                        std::string sk = k + "_" + std::to_string(s);
                        if (tryUpload(sk)) { found = true; break; }
                    }
                }
            }
        }

        if (!resolvedKeys.empty()) {
            // GPU-based dedup using dedupByKeys (radix sort + mark unique)
            uint32_t newCount = 0;
            MTL::Buffer* uniqueIdx = GpuOps::dedupByKeys(gpuKeys, rightCtx.rowCount, newCount);

            if (newCount < rightCtx.rowCount) {
                if (debug) {
                    std::cerr << "[Exec] Join: DELIM dedup RHS by [";
                    for (size_t ri=0; ri<resolvedKeys.size(); ++ri) { if (ri) std::cerr << ","; std::cerr << resolvedKeys[ri]; }
                    std::cerr << "]: " << rightCtx.rowCount << " -> " << newCount << "\n";
                }
                // GPU gather all GPU columns (u32, f32, dict, flat string)
                rightCtx.gatherAllGPU(uniqueIdx, newCount);
                // NOTE: syncCPUFromGPU removed — unified memory means
                // ->contents() already provides zero-cost CPU access.
                // CPU-only columns below are gathered explicitly.
                // CPU-only columns: download uniqueIdx and gather
                std::vector<uint32_t> keepIdx(newCount);
                memcpy(keepIdx.data(), uniqueIdx->contents(), newCount * sizeof(uint32_t));
                for (auto& [name, col] : rightCtx.u32Cols) {
                    if (!rightCtx.u32ColsGPU.count(name) && col.size() == rightCtx.rowCount) {
                        std::vector<uint32_t> compact(newCount);
                        for (uint32_t i = 0; i < newCount; ++i) compact[i] = col[keepIdx[i]];
                        col = std::move(compact);
                    }
                }
                for (auto& [name, col] : rightCtx.f32Cols) {
                    if (!rightCtx.f32ColsGPU.count(name) && col.size() == rightCtx.rowCount) {
                        std::vector<float> compact(newCount);
                        for (uint32_t i = 0; i < newCount; ++i) compact[i] = col[keepIdx[i]];
                        col = std::move(compact);
                    }
                }
                for (auto& [name, vec] : rightCtx.stringCols) {
                    if (vec.size() == rightCtx.rowCount && !rightCtx.dictCols.count(name) && !rightCtx.flatStringCols.count(name)) {
                        std::vector<std::string> compact(newCount);
                        for (uint32_t i = 0; i < newCount; ++i) compact[i] = vec[keepIdx[i]];
                        vec = std::move(compact);
                    }
                }
                rightCtx.invalidateStringColsForDictFlat();
                uniqueIdx->release();

                rightCtx.rowCount = newCount;
                rightCtx.activeRows.clear();
                rightCtx.activeRowsGPU = nullptr;
                rightCtx.activeRowsCountGPU = 0;

                // Strip right-side columns that already exist on the left side
                {
                    std::set<std::string> keepU32(resolvedKeys.begin(), resolvedKeys.end());
                    // Check both CPU and GPU column maps (CPU may be absent
                    // when syncCPUFromGPU is skipped — unified memory)
                    for (const auto& [name, _] : rightCtx.u32Cols) {
                        if (currentCtx.u32Cols.find(name) == currentCtx.u32Cols.end() &&
                            currentCtx.u32ColsGPU.find(name) == currentCtx.u32ColsGPU.end())
                            keepU32.insert(name);
                    }
                    for (const auto& [name, _] : rightCtx.u32ColsGPU) {
                        if (currentCtx.u32Cols.find(name) == currentCtx.u32Cols.end() &&
                            currentCtx.u32ColsGPU.find(name) == currentCtx.u32ColsGPU.end())
                            keepU32.insert(name);
                    }
                    for (auto it2 = rightCtx.u32Cols.begin(); it2 != rightCtx.u32Cols.end(); ) {
                        if (keepU32.find(it2->first) == keepU32.end()) it2 = rightCtx.u32Cols.erase(it2);
                        else ++it2;
                    }
                    for (auto it2 = rightCtx.u32ColsGPU.begin(); it2 != rightCtx.u32ColsGPU.end(); ) {
                        if (keepU32.find(it2->first) == keepU32.end())
                            it2 = rightCtx.u32ColsGPU.erase(it2);
                        else ++it2;
                    }
                    for (auto it2 = rightCtx.f32Cols.begin(); it2 != rightCtx.f32Cols.end(); ) {
                        if (currentCtx.f32Cols.find(it2->first) != currentCtx.f32Cols.end() ||
                            currentCtx.f32ColsGPU.find(it2->first) != currentCtx.f32ColsGPU.end())
                            it2 = rightCtx.f32Cols.erase(it2);
                        else ++it2;
                    }
                    for (auto it2 = rightCtx.f32ColsGPU.begin(); it2 != rightCtx.f32ColsGPU.end(); ) {
                        if (currentCtx.f32ColsGPU.find(it2->first) != currentCtx.f32ColsGPU.end() ||
                            currentCtx.f32Cols.find(it2->first) != currentCtx.f32Cols.end())
                            it2 = rightCtx.f32ColsGPU.erase(it2);
                        else ++it2;
                    }
                    if (debug) {
                        std::cerr << "[Exec] Join: stripped RHS to " << rightCtx.u32Cols.size()
                                  << " u32, " << rightCtx.f32Cols.size() << " f32, "
                                  << rightCtx.stringCols.size() << " string cols\n";
                    }
                }
            }
        }
    }
}

bool GpuExecutor::executeJoinPipeline(
    const IRJoin& join,
    EvalContext& currentCtx,
    std::unordered_map<std::string, EvalContext>& tableContexts,
    std::vector<EvalContext>& savedPipelines,
    std::vector<std::set<std::string>>& savedPipelineTables,
    std::set<std::string>& joinedTables,
    bool& hasPipeline,
    ExecutionResult& result
) {
    const bool debug = env_truthy("GPUDB_DEBUG_OPS");
                result.isScalarAggregate = false;
                
                // Collect all columns referenced in the join condition
                std::set<std::string> condCols;
                collectColumnsFromExpr(join.condition, condCols);
                
                if (debug) {
                    std::cerr << "[Exec] Join: conditionStr=" << join.conditionStr << "\n";
                    std::cerr << "[Exec] Join: type=";
                    switch (join.type) {
                        if (debug) case JoinType::Inner: std::cerr << "Inner"; break;
                        if (debug) case JoinType::Left: std::cerr << "Left"; break;
                        if (debug) case JoinType::Semi: std::cerr << "Semi"; break;
                        if (debug) case JoinType::Anti: std::cerr << "Anti"; break;
                        if (debug) case JoinType::Mark: std::cerr << "Mark"; break;
                        if (debug) default: std::cerr << "Unknown(" << static_cast<int>(join.type) << ")"; break;
                    }
                    if (debug) std::cerr << "\n";
                    if (debug) std::cerr << "[Exec] Join: condCols extracted: ";
                    if (debug) for (const auto& c : condCols) std::cerr << c << " ";
                    if (debug) std::cerr << "(total=" << condCols.size() << ")\n";
                }
                
                // Skip trivial self-joins from DELIM_SCAN correlation markers.
                bool isTrivialSelfJoin = false;
                
                // Check for IS NOT DISTINCT FROM pattern (DuckDB's DELIM_SCAN correlation marker)
                if (join.conditionStr.find("IS NOT DISTINCT FROM") != std::string::npos) {
                    std::string selfCol = parseSelfComparison(join.conditionStr);
                    if (!selfCol.empty()) {
                            // Verify column is in currentCtx (may be hidden by GroupBy).
                            bool colInContext = (currentCtx.u32Cols.find(selfCol) != currentCtx.u32Cols.end() ||
                                                 currentCtx.f32Cols.find(selfCol) != currentCtx.f32Cols.end());
                            
                            if (colInContext) {
                                // Don't skip LEFT joins (needed for DELIM joins)
                                if (join.type != JoinType::Left) {
                                    // Don't skip if explicit right table is specified
                                    if (join.rightTable.empty()) {
                                        isTrivialSelfJoin = true;
                                    } else if (debug) {
                                        std::cerr << "[Exec] Join: IS NOT DISTINCT FROM self-comparison BUT explicit right table specified (" << join.rightTable << "). Not skipping.\n";
                                    }
                                }
                                if (debug && isTrivialSelfJoin) {
                                    std::cerr << "[Exec] Join: IS NOT DISTINCT FROM self-comparison: '" 
                                              << selfCol << "' (col in context)\n";
                                }
                            } else if (debug) {
                                std::cerr << "[Exec] Join: IS NOT DISTINCT FROM self-comparison: '" 
                                          << selfCol << "' BUT col not in context, may need re-join\n";
                            }
                    }
                }
                
                // Also check for self-comparison patterns: col = col
                if (!isTrivialSelfJoin && condCols.size() == 1) {
                    // Only one unique column - it's a self-comparison
                    const std::string& col = *condCols.begin();
                    
                    // First check if the column is actually in the current context
                    bool colInContext = (currentCtx.u32Cols.find(col) != currentCtx.u32Cols.end() ||
                                         currentCtx.f32Cols.find(col) != currentCtx.f32Cols.end());
                    
                    if (colInContext) {
                        std::string baseTable = tableForColumn(col);
                        // Check if base table or any of its instances are already joined
                        bool alreadyJoined = false;
                        for (const auto& jt : joinedTables) {
                            if (jt == baseTable || jt.rfind(baseTable + "_", 0) == 0) {
                                alreadyJoined = true;
                                break;
                            }
                        }
                        if (alreadyJoined) {
                            // The column's table is already joined AND the column is in context
                            // Don't skip LEFT joins
                            if (join.type != JoinType::Left) {
                                if (join.rightTable.empty()) {
                                    isTrivialSelfJoin = true;
                                } else if (debug) {
                                    std::cerr << "[Exec] Join: self-comparison BUT explicit right table specified (" << join.rightTable << "). Not skipping.\n";
                                }
                            }
                            if (debug && isTrivialSelfJoin) {
                                std::cerr << "[Exec] Join: self-comparison detected for " << col << " (table " << baseTable << " already joined, col in context)\n";
                            }
                        }
                    } else if (debug) {
                        std::cerr << "[Exec] Join: self-comparison for " << col << " but col not in context, may need re-join\n";
                    }
                }
                
                if (isTrivialSelfJoin) {
                    if (debug) {
                        std::cerr << "[Exec] Join: skipping trivial self-join (all columns already in pipeline)\n";
                    }
                    return true; // Skip this join
                }
                
                // Check for scalar subquery pattern (join condition contains SUBQUERY).
                if (join.conditionStr.find("SUBQUERY") != std::string::npos && !savedPipelines.empty()) {
                    if (handleScalarSubquerySavedPipelines(join, currentCtx, savedPipelines, savedPipelineTables, joinedTables, result, debug))
                        return true;
                }
                
                // Alt: scalar SUBQUERY join (savedPipelines empty, data in tableContexts).
                if (join.conditionStr.find("SUBQUERY") != std::string::npos && savedPipelines.empty()) {
                    if (handleScalarSubqueryTableContexts(join, currentCtx, tableContexts, joinedTables, hasPipeline, result, debug))
                        return true;
                }
                
                // Check for malformed joins where both condition columns are from the same table
                // and are DIFFERENT columns (e.g., "p_size = p_partkey" in Q16).
                // Do NOT skip valid self-comparisons (e.g., "col = col") which indicate self-joins.
                if (condCols.size() == 2) {
                    std::string firstTable;
                    bool allColsFromSameTable = true;
                    bool hasOrphanColumn = false;
                    std::vector<std::string> colsList(condCols.begin(), condCols.end());
                    
                    for (const auto& col : condCols) {
                        std::string baseTable = tableForColumn(col);
                        if (baseTable.empty()) {
                            hasOrphanColumn = true;
                        } else {
                            if (firstTable.empty()) {
                                firstTable = baseTable;
                            } else if (baseTable != firstTable) {
                                allColsFromSameTable = false;
                            }
                        }
                    }
                    
                    // Check for self-comparison patterns (e.g., "l_k = l_k").
                    // These indicate valid self-joins between table instances.
                    bool hasSelfComparisonInCondition = false;
                    for (const auto& col : condCols) {
                        // Check for "col = col" pattern
                        std::string pattern1 = col + " = " + col;
                        std::string pattern2 = col + " IS NOT DISTINCT FROM " + col;
                        if (join.conditionStr.find(pattern1) != std::string::npos ||
                            join.conditionStr.find(pattern2) != std::string::npos) {
                            hasSelfComparisonInCondition = true;
                            break;
                        }
                    }
                    
                    // "p_size = p_partkey" (same table, different col) -> skip.
                    // "col = col" -> valid self-join.
                    // Also check for suffixed aliases (e.g. p_partkey_rhs_9) which imply distinct instances
                    bool hasAlias = false;
                    for (const auto& col : condCols) {
                        if (col.find("_rhs_") != std::string::npos || col.find("_lhs_") != std::string::npos) {
                            hasAlias = true;
                            break;
                        }
                    }

                    if (allColsFromSameTable && !firstTable.empty() && !hasOrphanColumn && !hasSelfComparisonInCondition && !hasAlias) {
                        if (debug) {
                            std::cerr << "[Exec] Join: skipping malformed join (both columns from " 
                                      << firstTable << ", different cols: " << colsList[0] << " vs " << colsList[1] << ")\n";
                        }
                        return true; // Skip this join
                    }
                    
                    // Check orphan columns (no prefix). Only skip if genuinely not found anywhere (CTX/CTE).
                    if (hasOrphanColumn) {
                        bool orphanFoundSomewhere = false;
                        for (const auto& col : condCols) {
                            if (tableForColumn(col).empty()) {
                                // Check if this orphan column exists in currentCtx, tableContexts, or savedPipelines
                                if (currentCtx.u32Cols.find(col) != currentCtx.u32Cols.end() ||
                                    currentCtx.f32Cols.find(col) != currentCtx.f32Cols.end()) {
                                    orphanFoundSomewhere = true;
                                    break;
                                }
                                for (const auto& [tname, tctx] : tableContexts) {
                                    if (tctx.u32Cols.find(col) != tctx.u32Cols.end() ||
                                        tctx.f32Cols.find(col) != tctx.f32Cols.end()) {
                                        orphanFoundSomewhere = true;
                                        break;
                                    }
                                }
                                for (const auto& sp : savedPipelines) {
                                    if (sp.u32Cols.find(col) != sp.u32Cols.end() ||
                                        sp.f32Cols.find(col) != sp.f32Cols.end()) {
                                        orphanFoundSomewhere = true;
                                        break;
                                    }
                                }
                            }
                        }
                        
                        // Only skip if orphan column is truly not found
                        if (!orphanFoundSomewhere) {
                            static const std::unordered_map<std::string, std::string> knownAliases = {
                                {"supplier_no", "l_suppkey"}
                            };
                            
                            for (const auto& col : condCols) {
                                if (knownAliases.count(col)) {
                                    std::string mapped = knownAliases.at(col);
                                    // Check if mapped column exists anywhere
                                    bool mappedFound = false;
                                    auto checkCtx = [&](const EvalContext& ctx) {
                                        return ctx.u32Cols.count(mapped) || ctx.f32Cols.count(mapped);
                                    };
                                    
                                    if (checkCtx(currentCtx)) mappedFound = true;
                                    else {
                                        for (const auto& [t, c] : tableContexts) if (checkCtx(c)) mappedFound = true;
                                        if (!mappedFound) for (const auto& sp : savedPipelines) if (checkCtx(sp)) mappedFound = true;
                                    }
                                    
                                    if (mappedFound) {
                                        if (debug) std::cerr << "[Exec] Join: resolved orphan '" << col << "' -> '" << mapped << "'\n";
                                        // Found via alias. Proceed to 'matchesRHS' check; subsequent logic handles lookup.
                                        orphanFoundSomewhere = true;
                                    }
                                }
                            }

                            if (!orphanFoundSomewhere) {
                                // Special case: SUBQUERY token with a scalar (1-row) right table
                                // This is a NESTED_LOOP_JOIN with a scalar subquery result
                                bool handleAsScalarSubquery = false;
                                if (condCols.count("SUBQUERY") && !join.rightTable.empty()) {
                                    // Check if right table has 1 row (scalar subquery)
                                    EvalContext* scalarCtx = nullptr;
                                    if (tableContexts.count(join.rightTable)) {
                                        scalarCtx = &tableContexts[join.rightTable];
                                    }
                                    if (!scalarCtx) {
                                        for (auto& sp : savedPipelines) {
                                            if (savedPipelineTables[&sp - &savedPipelines[0]].count(join.rightTable)) {
                                                scalarCtx = &sp;
                                                break;
                                            }
                                        }
                                    }
                                    if (scalarCtx && scalarCtx->rowCount <= 1) {
                                        // Find the scalar value - prefer avg() column or matching column name
                                        float scalarVal = 0.0f;
                                        bool foundScalar = false;
                                        // First try to find avg() column
                                        for (const auto& [name, vec] : scalarCtx->f32Cols) {
                                            if (!vec.empty() && (name.find("avg") != std::string::npos || 
                                                                  name.find("first") != std::string::npos)) {
                                                scalarVal = vec[0];
                                                foundScalar = true;
                                                if (debug) std::cerr << "[Exec] Join: SUBQUERY scalar from '" << name << "' = " << scalarVal << "\n";
                                                break;
                                            }
                                        }
                                        // Fallback: first f32 column that isn't count_star
                                        if (!foundScalar) {
                                            for (const auto& [name, vec] : scalarCtx->f32Cols) {
                                                if (!vec.empty() && name.find("count") == std::string::npos) {
                                                    scalarVal = vec[0];
                                                    foundScalar = true;
                                                    if (debug) std::cerr << "[Exec] Join: SUBQUERY scalar from '" << name << "' = " << scalarVal << "\n";
                                                    break;
                                                }
                                            }
                                        }
                                        if (foundScalar) {
                                            // Apply as filter: replace SUBQUERY with the scalar value
                                            // Parse condition: "CAST(c_acctbal AS DOUBLE) > SUBQUERY"
                                            // → filter "c_acctbal > <scalar>"
                                            std::string filterCol;
                                            std::string filterOp;
                                            std::string cond = join.conditionStr;
                                            // Extract the LHS column (strip CAST if present)
                                            size_t castPos = cond.find("CAST(");
                                            if (castPos != std::string::npos) {
                                                size_t asPos = cond.find(" AS ", castPos);
                                                if (asPos != std::string::npos) {
                                                    filterCol = cond.substr(castPos + 5, asPos - castPos - 5);
                                                }
                                            }
                                            // Find the operator
                                            for (const auto& op : {">", "<", ">=", "<=", "="}) {
                                                size_t opPos = cond.find(std::string(" ") + op + " ");
                                                if (opPos != std::string::npos) {
                                                    filterOp = op;
                                                    if (filterCol.empty()) {
                                                        filterCol = base_ident(cond.substr(0, opPos));
                                                    }
                                                    break;
                                                }
                                            }
                                            if (!filterCol.empty() && !filterOp.empty()) {
                                                if (debug) {
                                                    std::cerr << "[Exec] Join: SUBQUERY scalar cross-join: " 
                                                              << filterCol << " " << filterOp << " " << scalarVal << "\n";
                                                }
                                                // Apply the filter directly on currentCtx
                                                // Always prefer the highest-suffixed version (latest scan instance)
                                                {
                                                    std::string bestMatch;
                                                    int bestSuffix = -1;
                                                    for (const auto& [n, v] : currentCtx.f32Cols) {
                                                        if (n == filterCol) {
                                                            if (bestSuffix < 0) { bestMatch = n; bestSuffix = 0; }
                                                        } else {
                                                            auto pos = n.rfind('_');
                                                            if (pos != std::string::npos) {
                                                                std::string sfx = n.substr(pos + 1);
                                                                if (!sfx.empty() && std::all_of(sfx.begin(), sfx.end(), ::isdigit) 
                                                                    && n.substr(0, pos) == filterCol) {
                                                                    int sfxNum = std::stoi(sfx);
                                                                    if (sfxNum > bestSuffix) {
                                                                        bestMatch = n;
                                                                        bestSuffix = sfxNum;
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                    if (bestMatch.empty()) {
                                                        goto skip_scalar_filter;
                                                    }
                                                    filterCol = bestMatch;
                                                    if (debug) std::cerr << "[Exec] Join: SUBQUERY resolved filterCol to '" << filterCol << "'\n";
                                                }
                                                {
                                                    // --- GPU scalar subquery filter ---
                                                    // Map filterOp string to GpuFilterOp enum
                                                    engine::GpuFilterOp compOp = engine::GpuFilterOp::EQ;
                                                    if (filterOp == ">") compOp = engine::GpuFilterOp::GT;
                                                    else if (filterOp == ">=") compOp = engine::GpuFilterOp::GE;
                                                    else if (filterOp == "<") compOp = engine::GpuFilterOp::LT;
                                                    else if (filterOp == "<=") compOp = engine::GpuFilterOp::LE;
                                                    else if (filterOp == "=") compOp = engine::GpuFilterOp::EQ;

                                                    // Ensure the filter column has a GPU buffer
                                                    MTL::Buffer* filterColGPU = nullptr;
                                                    if (currentCtx.f32ColsGPU.count(filterCol)) {
                                                        filterColGPU = currentCtx.f32ColsGPU[filterCol];
                                                    } else if (!currentCtx.f32Cols[filterCol].empty()) {
                                                        auto& vec = currentCtx.f32Cols[filterCol];
                                                        filterColGPU = GpuOps::createBuffer(vec.data(), vec.size() * sizeof(float));
                                                        currentCtx.f32ColsGPU[filterCol].reset(filterColGPU);
                                                    }
                                                    if (!filterColGPU) {
                                                        if (debug) std::cerr << "[Exec] Join: SUBQUERY GPU filter: no GPU buffer for '" << filterCol << "'\n";
                                                        goto skip_scalar_filter;
                                                    }

                                                    // Run GPU filter (indexed if activeRows present)
                                                    std::optional<FilterResult> gpuFilterRes;
                                                    if (currentCtx.activeRowsGPU && currentCtx.activeRowsCountGPU > 0) {
                                                        gpuFilterRes = GpuOps::filterF32Indexed(filterCol, filterColGPU,
                                                                                                 currentCtx.activeRowsGPU, currentCtx.activeRowsCountGPU, compOp, scalarVal);
                                                    } else if (!currentCtx.activeRows.empty()) {
                                                        // CPU activeRows without GPU buffer (rare fallback)
                                                        MTL::Buffer* arGPU = GpuOps::createBuffer(currentCtx.activeRows.data(),
                                                                                         currentCtx.activeRows.size() * sizeof(uint32_t));
                                                        uint32_t arCount = (uint32_t)currentCtx.activeRows.size();
                                                        gpuFilterRes = GpuOps::filterF32Indexed(filterCol, filterColGPU,
                                                                                                 arGPU, arCount, compOp, scalarVal);
                                                        if (arGPU) arGPU->release();
                                                    } else {
                                                        uint32_t fullRowCount = (uint32_t)currentCtx.f32Cols[filterCol].size();
                                                        if (fullRowCount == 0) fullRowCount = currentCtx.rowCount;
                                                        gpuFilterRes = GpuOps::filterF32(filterCol, filterColGPU,
                                                                                          fullRowCount, compOp, scalarVal);
                                                    }
                                                    if (!gpuFilterRes) {
                                                        if (debug) std::cerr << "[Exec] Join: SUBQUERY GPU filter failed, skipping\n";
                                                        goto skip_scalar_filter;
                                                    }

                                                    {
                                                        MTL::Buffer* keepIndicesGPU = gpuFilterRes->indices;
                                                        uint32_t keepCount = gpuFilterRes->count;

                                                        if (debug) {
                                                            std::cerr << "[Exec] Join: SUBQUERY GPU scalar filter: " 
                                                                      << keepCount << " rows after\n";
                                                        }

                                                        // GPU gather for u32 columns
                                                        for (auto& [name, buf] : currentCtx.u32ColsGPU) {
                                                            if (buf) {
                                                                MTL::Buffer* gathered = GpuOps::gatherU32(buf, keepIndicesGPU, keepCount, false);
                                                                buf.reset(gathered);
                                                            }
                                                        }
                                                        // GPU gather for f32 columns
                                                        for (auto& [name, buf] : currentCtx.f32ColsGPU) {
                                                            if (buf) {
                                                                MTL::Buffer* gathered = GpuOps::gatherF32(buf, keepIndicesGPU, keepCount, false);
                                                                buf.reset(gathered);
                                                            }
                                                        }
                                                        GpuOps::sync();

                                                        // CPU gather for string columns (fallback when dict/flat not available)
                                                        // Read keepIndices back to CPU for string compaction
                                                        std::vector<uint32_t> keepIdx(keepCount);
                                                        if (keepCount > 0) {
                                                            memcpy(keepIdx.data(), keepIndicesGPU->contents(), keepCount * sizeof(uint32_t));
                                                        }
                                                        for (auto& [name, vec] : currentCtx.stringCols) {
                                                            if (!vec.empty() && !currentCtx.dictCols.count(name) && !currentCtx.flatStringCols.count(name)) {
                                                                std::vector<std::string> compact(keepCount);
                                                                for (uint32_t i = 0; i < keepCount; ++i)
                                                                    compact[i] = vec[keepIdx[i]];
                                                                vec = std::move(compact);
                                                            }
                                                        }

                                                        // GPU gather for dict and flat string columns
                                                        currentCtx.compactDictCols(keepIndicesGPU, keepCount);
                                                        currentCtx.compactFlatStringCols(keepIndicesGPU, keepCount);
                                                        currentCtx.invalidateStringColsForDictFlat();

                                                        // Materialize GPU back to CPU vectors for downstream use
                                                        for (auto& [name, vec] : currentCtx.u32Cols) {
                                                            if (currentCtx.u32ColsGPU.count(name) && currentCtx.u32ColsGPU[name]) {
                                                                vec.resize(keepCount);
                                                                memcpy(vec.data(), currentCtx.u32ColsGPU[name]->contents(),
                                                                       keepCount * sizeof(uint32_t));
                                                            } else if (!vec.empty()) {
                                                                std::vector<uint32_t> compact(keepCount);
                                                                for (uint32_t i = 0; i < keepCount; ++i)
                                                                    compact[i] = vec[keepIdx[i]];
                                                                vec = std::move(compact);
                                                            }
                                                        }
                                                        for (auto& [name, vec] : currentCtx.f32Cols) {
                                                            if (currentCtx.f32ColsGPU.count(name) && currentCtx.f32ColsGPU[name]) {
                                                                vec.resize(keepCount);
                                                                memcpy(vec.data(), currentCtx.f32ColsGPU[name]->contents(),
                                                                       keepCount * sizeof(float));
                                                            } else if (!vec.empty()) {
                                                                std::vector<float> compact(keepCount);
                                                                for (uint32_t i = 0; i < keepCount; ++i)
                                                                    compact[i] = vec[keepIdx[i]];
                                                                vec = std::move(compact);
                                                            }
                                                        }

                                                        currentCtx.activeRows.clear();
                                                        currentCtx.activeRowsGPU = nullptr;
                                                        currentCtx.rowCount = keepCount;

                                                        handleAsScalarSubquery = true;
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                                skip_scalar_filter:
                                if (!handleAsScalarSubquery) {
                                    if (debug) {
                                        std::cerr << "[Exec] Join: skipping join with orphan column (not found anywhere)\n";
                                    }
                                    return true; // Skip this join
                                } else {
                                    return true; // Already handled as scalar filter, skip normal join
                                }
                            }
                        } else if (debug) {
                            std::cerr << "[Exec] Join: orphan column found in some context, proceeding\n";
                        }
                    }
                }
                
                // Check for unjoined table instances (priority over saved pipelines for multi-instance tables).
                // Ensure table is NOT already in a saved pipeline.
                std::string unjoinedTableForJoin;
                
                // Lambda to check if column (or its suffixed version) exists in a context
                auto hasColumnOrSuffixed = [](const EvalContext& ctx, const std::string& colName) -> bool {
                    if (ctx.u32Cols.find(colName) != ctx.u32Cols.end()) return true;
                    if (ctx.f32Cols.find(colName) != ctx.f32Cols.end()) return true;
                    // Try numeric suffixes
                    for (int suffix = 1; suffix <= 9; ++suffix) {
                        std::string suffixedCol = colName + "_" + std::to_string(suffix);
                        if (ctx.u32Cols.find(suffixedCol) != ctx.u32Cols.end()) return true;
                        if (ctx.f32Cols.find(suffixedCol) != ctx.f32Cols.end()) return true;
                    }
                    // Try rhs suffixes (e.g. col_rhs_10)
                    std::string rhsPattern = colName + "_rhs_";
                    for (const auto& [name, _] : ctx.u32Cols) {
                        if (name.find(rhsPattern) == 0) return true;
                    }
                    for (const auto& [name, _] : ctx.f32Cols) {
                        if (name.find(rhsPattern) == 0) return true;
                    }
                    return false;
                };
                
// Check if we need to use a saved pipeline for this join
                // Check this FIRST to prefer connecting to existing intermediate results (e.g. filtered sub-joins)
                // over creating new Cartesian products with fresh table instances.
                int savedPipelineIdx = -1;

                // PRIORITY: Explicit right table check (for DELIM joins)
                if (!join.rightTable.empty()) {
                    bool specificTableFound = false;
                    
                    // For base table names (not tmpl_ prefixes), check tableContexts FIRST
                    // This prevents incorrectly using a saved pipeline that contains a table
                    // when we actually need a fresh instance (e.g., nation table for multiple
                    // different nation joins in Q7)
                    bool isBaseTable = (join.rightTable.find("tmpl_") != 0);
                    
                    if (isBaseTable && tableContexts.count(join.rightTable)) {
                        unjoinedTableForJoin = join.rightTable;
                        specificTableFound = true;
                        if (debug) std::cerr << "[Exec] Join: found explicit right table '" << join.rightTable << "' in tableContexts (base table priority)\n";
                    }
                    
                    // For tmpl_ tables, check saved pipelines first
                    if (!specificTableFound) {
                        for (int pi = (int)savedPipelines.size() - 1; pi >= 0; --pi) {
                            if (savedPipelineTables[pi].count(join.rightTable)) {
                                savedPipelineIdx = pi;
                                specificTableFound = true;
                                if (debug) std::cerr << "[Exec] Join: found explicit right table '" << join.rightTable << "' in saved pipeline #" << pi << "\n";
                                break;
                            }
                        }
                    }
                    
                    // Check table contexts if not found in saved
                    if (!specificTableFound && tableContexts.count(join.rightTable)) {
                        unjoinedTableForJoin = join.rightTable;
                        specificTableFound = true;
                        if (debug) std::cerr << "[Exec] Join: found explicit right table '" << join.rightTable << "' in tableContexts\n";
                    }
                    
                    // VALIDATE: If the explicit right table doesn't contain any condition columns
                    // that are missing from the current context, it's likely a misidentification
                    // (e.g., planner captured a saved pipeline instead of the actual join target).
                    // Fall through to heuristic search in that case.
                    if (specificTableFound && !unjoinedTableForJoin.empty()) {
                        const EvalContext& rightCandidate = tableContexts[unjoinedTableForJoin];
                        bool hasNewColumn = false;
                        for (const auto& col : condCols) {
                            if (!hasColumnOrSuffixed(currentCtx, col) && hasColumnOrSuffixed(rightCandidate, col)) {
                                hasNewColumn = true;
                                break;
                            }
                        }
                        if (!hasNewColumn) {
                            if (debug) std::cerr << "[Exec] Join: explicit right table '" << unjoinedTableForJoin
                                                  << "' has no new condition columns, falling through to heuristic\n";
                            unjoinedTableForJoin.clear();
                            specificTableFound = false;
                        }
                    }
                }

                // If explicit lookup didn't set anything, run legacy heuristic
                if (savedPipelineIdx < 0 && unjoinedTableForJoin.empty())
                // Prefer LATEST pipeline (reverse search) to ensure we get the most accumulated state
                for (int pi = (int)savedPipelines.size() - 1; pi >= 0; --pi) {
                    const auto& savedCtx = savedPipelines[pi];
                    // Check if this saved pipeline has columns needed for the join
                    for (const auto& col : condCols) {
                        if (hasColumnOrSuffixed(savedCtx, col)) {
                            // Check that current pipeline doesn't have this column (or suffixed version)
                            if (!hasColumnOrSuffixed(currentCtx, col)) {
                                // Before accepting, check if the saved pipeline has been aggregated
                                // (very few rows compared to the fresh table in tableContexts).
                                // If so, prefer the fresh table from tableContexts instead.
                                std::string baseTable = tableForColumn(col);
                                bool isAggregatedPipeline = false;
                                if (!baseTable.empty() && savedCtx.rowCount <= 10) {
                                    for (const auto& [key, freshCtx] : tableContexts) {
                                        bool isInstanceOf = (key == baseTable || 
                                                            key.rfind(baseTable + "_", 0) == 0);
                                        if (isInstanceOf && freshCtx.rowCount > 10 && 
                                            joinedTables.find(key) == joinedTables.end()) {
                                            isAggregatedPipeline = true;
                                            if (debug) std::cerr << "[Exec] Join: savedPipeline " << pi 
                                                << " has " << savedCtx.rowCount << " rows but fresh table '" 
                                                << key << "' has " << freshCtx.rowCount 
                                                << " rows — skipping aggregated pipeline\n";
                                            break;
                                        }
                                    }
                                }
                                if (!isAggregatedPipeline) {
                                    savedPipelineIdx = pi;
                                }
                            }
                        }
                    }
                    if (savedPipelineIdx >= 0) break;
                }

                if (savedPipelineIdx < 0 && unjoinedTableForJoin.empty()) {
                    for (const auto& col : condCols) {
                        // Skip if column (or its suffixed version) is already in current context
                        if (hasColumnOrSuffixed(currentCtx, col)) {
                            continue;  // Column already in current context
                        }
                        
                        std::string baseTable = tableForColumn(col);
                        if (baseTable.empty()) continue;
                        
                        // Check for unjoined table instances
                        for (const auto& [key, ctx] : tableContexts) {
                            bool isInstanceOf = (key == baseTable || 
                                                key.rfind(baseTable + "_", 0) == 0);
                            if (isInstanceOf && joinedTables.find(key) == joinedTables.end()) {
                                // Check if this table is in a saved pipeline.
                                // If the saved pipeline has been aggregated to a very small row count
                                // (e.g. scalar subquery result), prefer the fresh table from tableContexts
                                // over the aggregated saved pipeline.
                                bool inSavedPipeline = false;
                                bool savedPipelineIsAggregated = false;
                                for (size_t spi = 0; spi < savedPipelineTables.size(); ++spi) {
                                    if (savedPipelineTables[spi].find(key) != savedPipelineTables[spi].end()) {
                                        inSavedPipeline = true;
                                        // Check if the saved pipeline was aggregated down 
                                        // (much fewer rows than the original table)
                                        size_t savedRows = savedPipelines[spi].rowCount;
                                        size_t freshRows = ctx.rowCount;
                                        if (freshRows > 10 && savedRows <= 10) {
                                            savedPipelineIsAggregated = true;
                                        }
                                        break;
                                    }
                                }
                                if (inSavedPipeline && !savedPipelineIsAggregated) {
                                    if (debug) {
                                        std::cerr << "[Exec] Join: table " << key 
                                                  << " is in saved pipeline, skipping\n";
                                    }
                                    continue;  // Skip - use saved pipeline instead
                                }
                                if (savedPipelineIsAggregated && debug) {
                                    if (debug) std::cerr << "[Exec] Join: table " << key 
                                              << " is in saved pipeline but pipeline was aggregated, using fresh table\n";
                                }
                                
                                if (hasColumnOrSuffixed(ctx, col)) {
                                    unjoinedTableForJoin = key;
                                    if (debug) {
                                        std::cerr << "[Exec] Join: found unjoined table " << key 
                                                  << " with column " << col << "\n";
                                    }
                                    break;
                                }
                            }
                        }
                        if (!unjoinedTableForJoin.empty()) break;
                    }
                }
                
                EvalContext rightCtx;
                std::set<std::string> rightJoinedTables;
                
                if (savedPipelineIdx >= 0) {
                    // Use saved pipeline as right context (multi-pipeline merge join)
                    rightCtx = savedPipelines[savedPipelineIdx];
                    rightJoinedTables = savedPipelineTables[savedPipelineIdx];
                    if (debug) {
                        std::cerr << "[Exec] Join: using saved pipeline " << savedPipelineIdx 
                                  << " with " << rightCtx.rowCount << " rows as right side\n";
                        std::cerr << "[Exec] Join: saved pipeline tables: ";
                        if (debug) for (const auto& t : rightJoinedTables) std::cerr << t << " ";
                        if (debug) std::cerr << "\n";
                    }
                } else if (!unjoinedTableForJoin.empty()) {
                    // Use the unjoined table we found earlier (priority over other inference)
                    // BUT: Skip if this is a spurious ANTI join with a scalar subquery table after GroupBy
                    // This pattern appears when DuckDB decorrelates scalar subqueries and creates
                    // both a theta-join (for the comparison) and an ANTI join (which is redundant)
                    bool skipSpuriousAntiJoin = false;
                    if ((join.type == JoinType::Anti || join.type == JoinType::Mark) &&
                        joinedTables.find("__GROUPED__") != joinedTables.end()) {
                        const EvalContext& potentialRight = tableContexts[unjoinedTableForJoin];
                        // If the potential right table has rowCount=1 (scalar subquery result),
                        // and this is a self-comparison (same col = same col), skip it
                        if (potentialRight.rowCount <= 1 && 
                            join.conditionStr.find("IS NOT DISTINCT FROM") != std::string::npos) {
                            std::string selfCol = parseSelfComparison(join.conditionStr);
                            if (!selfCol.empty()) {
                                if (debug) {
                                    std::cerr << "[Exec] Join: skipping spurious ANTI join with scalar table "
                                              << unjoinedTableForJoin << " after GroupBy\n";
                                }
                                skipSpuriousAntiJoin = true;
                            }
                        }
                    }
                    
                    if (skipSpuriousAntiJoin) {
                        return true; // Skip this join entirely
                    }
                    
                    rightCtx = tableContexts[unjoinedTableForJoin];
                    rightJoinedTables.insert(unjoinedTableForJoin);
                    if (debug) {
                        std::cerr << "[Exec] Join: using pre-found unjoined table " << unjoinedTableForJoin
                                  << " with " << rightCtx.rowCount << " rows as right side\n";
                    }
                } else {
                    // Find the right table from join condition - look for a table not yet joined
                    std::string rightTable;
                    if (!join.rightTable.empty()) {
                        rightTable = join.rightTable;
                    } else {
                        // Infer from condition - find a table not already in joinedTables
                        // Two-pass strategy:
                        // Pass 1: Find columns not in currentCtx (cleaner case)
                        // Pass 2: Find unjoined instances even if base column is in ctx
                        
                        // Pass 1: columns not in currentCtx
                        for (const auto& col : condCols) {
                            std::string baseTable = tableForColumn(col);
                            if (baseTable.empty()) continue;
                            
                            // Skip if column is already in currentCtx
                            bool colInCurrentCtx = (currentCtx.u32Cols.find(col) != currentCtx.u32Cols.end() ||
                                                   currentCtx.f32Cols.find(col) != currentCtx.f32Cols.end());
                            if (colInCurrentCtx) continue;
                            
                            // Find an unjoined instance of this table that contains this column
                            for (const auto& [key, ctx] : tableContexts) {
                                bool isInstanceOf = (key == baseTable || 
                                                    key.rfind(baseTable + "_", 0) == 0);
                                if (isInstanceOf && joinedTables.find(key) == joinedTables.end()) {
                                    // Check for column - try exact match first, then suffixed versions
                                    bool hasCol = (ctx.u32Cols.find(col) != ctx.u32Cols.end() ||
                                                  ctx.f32Cols.find(col) != ctx.f32Cols.end());
                                    // If not found, try suffixed versions (e.g., n_nationkey_2 for nation_2)
                                    if (!hasCol) {
                                        for (int suffix = 1; suffix <= 9; ++suffix) {
                                            std::string suffixedCol = col + "_" + std::to_string(suffix);
                                            if (ctx.u32Cols.find(suffixedCol) != ctx.u32Cols.end() ||
                                                ctx.f32Cols.find(suffixedCol) != ctx.f32Cols.end()) {
                                                hasCol = true;
                                                break;
                                            }
                                        }
                                    }
                                    if (hasCol) {
                                        rightTable = key;
                                        if (debug) {
                                            std::cerr << "[Exec] Join: found unjoined instance " << key 
                                                      << " for base table " << baseTable 
                                                      << " (has column " << col << " or suffixed variant)\n";
                                        }
                                        break;
                                    }
                                }
                            }
                            if (!rightTable.empty()) break;
                        }
                        
                        // Pass 2: if no table found, look for unjoined instances of multi-instance tables
                        if (rightTable.empty()) {
                            for (const auto& col : condCols) {
                                std::string baseTable = tableForColumn(col);
                                if (baseTable.empty()) continue;
                                
                                // Find an unjoined instance, even if column is in ctx from another instance
                                for (const auto& [key, ctx] : tableContexts) {
                                    bool isInstanceOf = (key == baseTable || 
                                                        key.rfind(baseTable + "_", 0) == 0);
                                    if (isInstanceOf && joinedTables.find(key) == joinedTables.end()) {
                                        // Check for column - try exact match first, then suffixed versions
                                        bool hasCol = (ctx.u32Cols.find(col) != ctx.u32Cols.end() ||
                                                      ctx.f32Cols.find(col) != ctx.f32Cols.end());
                                        // If not found, try suffixed versions (e.g., n_nationkey_2 for nation_2)
                                        if (!hasCol) {
                                            for (int suffix = 1; suffix <= 9; ++suffix) {
                                                std::string suffixedCol = col + "_" + std::to_string(suffix);
                                                if (ctx.u32Cols.find(suffixedCol) != ctx.u32Cols.end() ||
                                                    ctx.f32Cols.find(suffixedCol) != ctx.f32Cols.end()) {
                                                    hasCol = true;
                                                    break;
                                                }
                                            }
                                        }
                                        if (hasCol) {
                                            rightTable = key;
                                            if (debug) {
                                                std::cerr << "[Exec] Join: pass2 found unjoined instance " << key 
                                                          << " for base table " << baseTable 
                                                          << " (has column " << col << " or suffixed variant)\n";
                                            }
                                            break;
                                        }
                                    }
                                }
                                if (!rightTable.empty()) break;
                            }
                        }
                    }
                    
                    if (rightTable.empty() || tableContexts.find(rightTable) == tableContexts.end()) {
                        if (debug) {
                            std::cerr << "[Exec] Join: cannot determine right table. joinedTables=";
                            for (const auto& t : joinedTables) std::cerr << t << " ";
                            std::cerr << "\n";
                            if (debug) std::cerr << "[Exec] Join: available tableContexts=";
                            if (debug) for (const auto& [k, v] : tableContexts) std::cerr << k << " ";
                            if (debug) std::cerr << "\n";
                        }
                        result.error = "Cannot determine right table for join";
                        return false;
                    }
                    
                    rightCtx = tableContexts[rightTable];
                    rightJoinedTables.insert(rightTable);
                }
                
                EvalContext joinCtx;
                

                // Apply right filter if present (e.g. pushed down predicates)
                if (join.rightFilter) {
                    if (debug) std::cerr << "[Exec] Join: Applying right filter to right side (GPU)\n";
                    
                    if (!executeFilterRecursive(join.rightFilter, rightCtx)) {
                         throw std::runtime_error("GPU Join Right Filter failed.");
                    }
                }

                dedupDelimJoinRHS(join, currentCtx, rightCtx, debug);

                // SEMI join with self-comparison: swap so outer table is the probe side.
                if (join.type == JoinType::Semi && !join.rightTable.empty()) {
                    // Check for self-comparison condition (col = col)
                    auto& cond = join.conditionStr;
                    size_t eqPos = cond.find(" = ");
                    if (eqPos != std::string::npos) {
                        std::string lhs = cond.substr(0, eqPos);
                        std::string rhs = cond.substr(eqPos + 3);
                        while (!lhs.empty() && std::isspace(lhs.back())) lhs.pop_back();
                        while (!rhs.empty() && std::isspace(rhs.front())) rhs.erase(0, 1);
                        if (lhs == rhs) {
                            if (debug) std::cerr << "[Exec] SEMI join: swapping sides (right table becomes probe)\n";
                            std::swap(currentCtx, rightCtx);
                        }
                    }
                }

                if (!executeJoin(join, currentCtx, rightCtx, joinCtx)) {
                    result.error = "Join execution failed";
                    return false;
                }
                
                currentCtx = std::move(joinCtx);
                if (debug) {
                    std::cerr << "[Exec] Join: currentCtx after move: rowCount=" << currentCtx.rowCount << " u32ColsGPU.size=" << currentCtx.u32ColsGPU.size() << "\n";
                    std::cerr << "[Exec] Join: currentCtx.stringCols after move:\n";
                    for (const auto& [n, v] : currentCtx.stringCols) {
                        if (debug) std::cerr << "[Exec]   " << n << " size=" << v.size() << "\n";
                    }
                    if (debug) std::cerr << "[Exec] Join: currentCtx.currentTable='" << currentCtx.currentTable << "'\n";
                }
                // Merge all joined tables from both sides
                for (const auto& t : rightJoinedTables) {
                    joinedTables.insert(t);
                }
                hasPipeline = true;  // We now have a joined result in the pipeline
                if (debug) {
                    std::cerr << "[Exec] Join: " << currentCtx.rowCount << " rows after. joinedTables=";
                    for (const auto& t : joinedTables) std::cerr << t << " ";
                    std::cerr << "\n";
                }

    return true;
}

} // namespace engine
