// ============================================================================
// JoinOutput.cpp — Scatter/gather output columns and unmatched row handling
// ============================================================================
#include "JoinInternal.hpp"
#include <future>
#include <thread>

namespace engine {

void appendUnmatchedLeftRows(
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

void appendUnmatchedRightRows(
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

bool scatterJoinOutputColumns(
    EvalContext&       leftCtx,
    EvalContext&       rightCtx,
    EvalContext&       outCtx,
    const JoinResult&  jRes,
    uint32_t           resCount,
    uint32_t           /*lCount*/,
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
             if (resCount < engine::config::kParallelStringGatherThreshold) numThreads = 1;
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

} // namespace engine
