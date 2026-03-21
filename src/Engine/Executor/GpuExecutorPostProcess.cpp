#include "GpuExecutor.hpp"
#include "GpuExecutorDetail.hpp"
#include "Operators.hpp"
#include "GpuColumnStore.hpp"
#include "StringMaterializer.hpp"
#include "EngineConfig.hpp"

#include <algorithm>
#include <iostream>
#include <map>
#include <set>
#include <unordered_set>
#include "Logger.hpp"
#include "KernelTimer.hpp"

namespace engine {

// ============================================================================
// Post-processing helpers extracted from GpuExecutor.cpp
// Called from GpuExecutor::execute() after the main pipeline loop.
// ============================================================================

// --- GPU Feasibility Checking ---

std::vector<std::string> GpuExecutor::getUnsupportedFeatures(const Plan& plan) {
    std::vector<std::string> blockers;

    size_t joinCount = 0;
    bool hasEmptyScan = false;
    bool hasOuterJoin = false;
    bool hasSubqueryInCondition = false;
    bool hasIsNotDistinctFrom = false;
    std::map<std::string, int> tableScanCounts;

    bool hasUngroupedAggregate = false;
    bool hasSubqueryInFilter = false;
    for (const auto& node : plan.nodes) {
        if (node.duckdbName == "UNGROUPED_AGGREGATE") {
            hasUngroupedAggregate = true;
        }
        if (node.type == IRNode::Type::Filter) {
            const auto& filter = node.asFilter();
            if (filter.predicateStr.find("SUBQUERY") != std::string::npos) {
                hasSubqueryInFilter = true;
            }
        }
    }

    for (const auto& node : plan.nodes) {
        switch (node.type) {
            case IRNode::Type::Scan: {
                const auto& scan = node.asScan();
                if (!scan.table.empty()) {
                    tableScanCounts[scan.table]++;
                } else {
                    hasEmptyScan = true;
                }
                break;
            }
            case IRNode::Type::Join: {
                joinCount++;
                const auto& join = node.asJoin();
                if (join.type != JoinType::Inner && join.type != JoinType::Left &&
                    join.type != JoinType::Right && join.type != JoinType::Semi && 
                    join.type != JoinType::Anti && join.type != JoinType::Mark) {
                    hasOuterJoin = true;
                }
                if (join.conditionStr.find("SUBQUERY") != std::string::npos) {
                    hasSubqueryInCondition = true;
                }
                if (join.conditionStr.find("IS NOT DISTINCT FROM") != std::string::npos) {
                    hasIsNotDistinctFrom = true;
                }
                break;
            }
            case IRNode::Type::Filter: {
                const auto& filter = node.asFilter();
                if (filter.predicateStr.find("SUBQUERY") != std::string::npos) {
                    hasSubqueryInCondition = true;
                }
                break;
            }
            case IRNode::Type::Distinct:
                break;
            default:
                break;
        }
    }

    if (joinCount > 20) {
        blockers.push_back("Multi-way JOIN (>20 tables)");
    }
    if (hasOuterJoin) {
        blockers.push_back("FULL OUTER/CROSS JOIN not supported (INNER/LEFT/RIGHT/SEMI/ANTI supported)");
    }
    if (hasSubqueryInCondition) {
        bool isScalarSubqueryOK = hasUngroupedAggregate && !hasSubqueryInFilter && !hasIsNotDistinctFrom;
        
        bool hasDecorrelatedJoin = false;
        for (const auto& n : plan.nodes) {
            if (n.type == IRNode::Type::Join) {
                const auto& j = n.asJoin();
                if (j.type == JoinType::Semi || j.type == JoinType::Anti || j.type == JoinType::Mark) {
                    hasDecorrelatedJoin = true;
                    break;
                }
            }
        }
        
        if (!isScalarSubqueryOK && !hasDecorrelatedJoin && !hasEmptyScan) {
            blockers.push_back("Correlated subquery in condition not supported");
        }
    }

    for (const auto& node : plan.nodes) {
        if (node.type == IRNode::Type::GroupBy) {
            const auto& gb = node.asGroupBy();
            if (gb.keys.size() > engine::config::kMaxGroupByKeys) {
                blockers.push_back("GROUP BY with >8 keys");
            }
        }
    }

    return blockers;
}

uint32_t deduplicateContext(EvalContext& ctx,
                                const std::vector<std::string>& dedupCols,
                                bool debug) {
    if (dedupCols.empty() || ctx.rowCount <= 1) return 0;

    // Collect GPU key buffers for dedup columns (upload from CPU if needed)
    std::vector<MTL::Buffer*> gpuKeys;
    for (const auto& col : dedupCols) {
        auto it = ctx.u32ColsGPU.find(col);
        if (it != ctx.u32ColsGPU.end() && it->second) {
            gpuKeys.push_back(it->second);
        } else {
            // Try CPU column and upload to GPU
            auto cit = ctx.u32Cols.find(col);
            if (cit != ctx.u32Cols.end() && cit->second.size() >= ctx.rowCount) {
                GpuBuffer buf = GpuOps::createBuffer(cit->second.data(), ctx.rowCount * sizeof(uint32_t));
                if (buf) {
                    ctx.u32ColsGPU[col] = std::move(buf);
                    gpuKeys.push_back(ctx.u32ColsGPU[col]);
                } else {
                    return 0;
                }
            } else {
                return 0;
            }
        }
    }

    uint32_t uniqueCount = 0;
    GpuBuffer uniqueIdx = GpuOps::dedupByKeys(gpuKeys, ctx.rowCount, uniqueCount);
    if (!uniqueIdx || uniqueCount == 0 || uniqueCount >= ctx.rowCount) {
        return 0;  // No duplicates found
    }

    if (debug) {
        LOG_INFO("Exec", "GPU dedup: " << ctx.rowCount << " -> " << uniqueCount << " rows\n");
    }

    // GPU gather u32 columns (async)
    for (auto& [name, buf] : ctx.u32ColsGPU) {
        if (buf) {
            auto compacted = GpuOps::gatherU32(buf, uniqueIdx, uniqueCount, false);
            if (compacted) buf = std::move(compacted);
        }
    }
    // GPU gather f32 columns (async)
    for (auto& [name, buf] : ctx.f32ColsGPU) {
        if (buf) {
            auto compacted = GpuOps::gatherF32(buf, uniqueIdx, uniqueCount, false);
            if (compacted) buf = std::move(compacted);
        }
    }

    // Invalidate stale CPU mirrors — unified memory means GPU buffers
    // are directly accessible via ->contents().
    for (auto& [name, buf] : ctx.u32ColsGPU) {
        if (buf) ctx.u32Cols[name].clear();
    }
    for (auto& [name, buf] : ctx.f32ColsGPU) {
        if (buf) ctx.f32Cols[name].clear();
    }
    // CPU-only u32/f32 columns: upload → GPU gather (async) → keep as GPU buffer
    {
        auto& s = GpuColumnStore::instance();
        std::vector<MTL::Buffer*> tempSrcs;
        for (auto& [name, col] : ctx.u32Cols) {
            if (!ctx.u32ColsGPU.count(name) && col.size() >= ctx.rowCount) {
                MTL::Buffer* src = s.device()->newBuffer(col.data(), col.size() * sizeof(uint32_t), MTL::ResourceStorageModeShared);
                GpuBuffer dst = GpuOps::gatherU32(src, uniqueIdx, uniqueCount, false);
                if (dst) {
                    ctx.u32ColsGPU[name] = std::move(dst);
                    col.clear();
                }
                tempSrcs.push_back(src);
            }
        }
        for (auto& [name, col] : ctx.f32Cols) {
            if (!ctx.f32ColsGPU.count(name) && col.size() >= ctx.rowCount) {
                MTL::Buffer* src = s.device()->newBuffer(col.data(), col.size() * sizeof(float), MTL::ResourceStorageModeShared);
                GpuBuffer dst = GpuOps::gatherF32(src, uniqueIdx, uniqueCount, false);
                if (dst) {
                    ctx.f32ColsGPU[name] = std::move(dst);
                    col.clear();
                }
                tempSrcs.push_back(src);
            }
        }
        // Sync + release temp sources after all gathers dispatched
        GpuOps::sync();
        for (auto* s : tempSrcs) s->release();
    }
    // String columns: GPU gather dict IDs (async — GPU-native dictionary encoding)
    for (auto& [name, dict] : ctx.dictCols) {
        if (dict.idsGPU) {
            auto compacted = GpuOps::gatherU32(dict.idsGPU, uniqueIdx, uniqueCount, false);
            if (compacted) {
                dict.idsGPU = std::move(compacted);
                dict.rowCount = uniqueCount;
                dict.ids.clear();
            }
        }
    }
    // Sync stringCols: skip those covered by dict or flat, CPU gather orphans only
    for (auto& [name, col] : ctx.stringCols) {
        if (ctx.dictCols.count(name) || ctx.flatStringCols.count(name)) {
            continue;
        } else if (col.size() >= ctx.rowCount) {
            std::vector<uint32_t> keepIdx(uniqueCount);
            memcpy(keepIdx.data(), uniqueIdx->contents(), uniqueCount * sizeof(uint32_t));
            std::vector<std::string> compact(uniqueCount);
            for (uint32_t i = 0; i < uniqueCount; ++i) compact[i] = col[keepIdx[i]];
            col = std::move(compact);
        }
    }
    // GPU gather for flat string columns
    ctx.compactFlatStringCols(uniqueIdx, uniqueCount);
    ctx.invalidateStringColsForDictFlat();

    ctx.rowCount = uniqueCount;
    ctx.activeRows.clear();
    ctx.activeRowsGPU = nullptr;
    ctx.activeRowsCountGPU = 0;
    return uniqueCount;
}

void materializeContextToResult(
    EvalContext& currentCtx, TableResult& tableResult,
    bool isScalarAggregate, bool debug
) {
    // If tableResult is empty but currentCtx has data (scan-only pipeline),
    // materialize ctx columns into tableResult so the output path can render them.
    if (tableResult.u32Names.empty() && tableResult.f32Names.empty() &&
        tableResult.stringNames.empty() && !isScalarAggregate &&
        currentCtx.rowCount > 0) {

        // Materialize GPU buffers to CPU vectors
        auto materialize = [&]() {
            // Determine active row indices (if filtered)
            uint32_t arCount = currentCtx.rowCount;
            bool hasAR = (currentCtx.activeRowsGPU && currentCtx.activeRowsCountGPU > 0);
            if (hasAR) arCount = currentCtx.activeRowsCountGPU;

            // Phase 1: Dispatch all u32/f32 gathers async
            struct PendU32 { std::string name; GpuBuffer buf; };
            struct PendF32 { std::string name; GpuBuffer buf; };
            std::vector<PendU32> pendU32;
            std::vector<PendF32> pendF32;
            std::vector<std::string> directU32Names, directF32Names;
            std::vector<std::vector<uint32_t>> directU32Cols;
            std::vector<std::vector<float>> directF32Cols;

            for (const auto& [name, buf] : currentCtx.u32ColsGPU) {
                if (!buf) continue;
                if (hasAR) {
                    GpuBuffer gathered = GpuOps::gatherU32(buf, currentCtx.activeRowsGPU, arCount, false);
                    pendU32.push_back({name, std::move(gathered)});
                } else {
                    std::vector<uint32_t> col(arCount);
                    std::memcpy(col.data(), buf->contents(), arCount * sizeof(uint32_t));
                    directU32Names.push_back(name);
                    directU32Cols.push_back(std::move(col));
                }
            }
            for (const auto& [name, buf] : currentCtx.f32ColsGPU) {
                if (!buf) continue;
                if (hasAR) {
                    GpuBuffer gathered = GpuOps::gatherF32(buf, currentCtx.activeRowsGPU, arCount, false);
                    pendF32.push_back({name, std::move(gathered)});
                } else {
                    std::vector<float> col(arCount);
                    std::memcpy(col.data(), buf->contents(), arCount * sizeof(float));
                    directF32Names.push_back(name);
                    directF32Cols.push_back(std::move(col));
                }
            }
            // Also dispatch dict ID gathers async
            struct PendDict { std::string name; GpuBuffer ids; const DictEncoded* dict; };
            std::vector<PendDict> pendDict;
            for (const auto& [name, dict] : currentCtx.dictCols) {
                if (!dict.valid()) continue;
                if (hasAR) {
                    GpuBuffer gatheredIds = GpuOps::gatherU32(dict.idsGPU, currentCtx.activeRowsGPU, arCount, false);
                    PendDict pd;
                    pd.name = name;
                    pd.ids = std::move(gatheredIds);
                    pd.dict = &dict;
                    pendDict.push_back(std::move(pd));
                }
            }

            // Phase 2: Sync all async gathers
            if (hasAR && (!pendU32.empty() || !pendF32.empty() || !pendDict.empty()))
                GpuOps::sync();

            // Phase 3: Download results
            for (auto& p : pendU32) {
                std::vector<uint32_t> col(arCount);
                std::memcpy(col.data(), p.buf->contents(), arCount * sizeof(uint32_t));
                tableResult.u32Names.push_back(p.name);
                tableResult.u32Cols.push_back(std::move(col));
            }
            for (size_t i = 0; i < directU32Names.size(); ++i) {
                tableResult.u32Names.push_back(directU32Names[i]);
                tableResult.u32Cols.push_back(std::move(directU32Cols[i]));
            }
            for (auto& p : pendF32) {
                std::vector<float> col(arCount);
                std::memcpy(col.data(), p.buf->contents(), arCount * sizeof(float));
                tableResult.f32Names.push_back(p.name);
                tableResult.f32Cols.push_back(std::move(col));
            }
            for (size_t i = 0; i < directF32Names.size(); ++i) {
                tableResult.f32Names.push_back(directF32Names[i]);
                tableResult.f32Cols.push_back(std::move(directF32Cols[i]));
            }

            for (const auto& [name, vec] : currentCtx.stringCols) {
                // Skip if dict or flat alternative available — let those loops handle it
                if (currentCtx.dictCols.count(name) || currentCtx.flatStringCols.count(name)) continue;
                if (hasAR) {
                    const uint32_t* arPtr = static_cast<const uint32_t*>(currentCtx.activeRowsGPU->contents());
                    std::vector<std::string> col(arCount);
                    for (uint32_t i = 0; i < arCount; ++i) {
                        if (arPtr[i] < vec.size()) col[i] = vec[arPtr[i]];
                    }
                    tableResult.stringNames.push_back(name);
                    tableResult.stringCols.push_back(std::move(col));
                } else {
                    tableResult.stringNames.push_back(name);
                    tableResult.stringCols.push_back(
                        std::vector<std::string>(vec.begin(), vec.begin() + std::min((size_t)arCount, vec.size())));
                }
            }
            // Materialize dict-only string columns
            for (auto& p : pendDict) {
                std::vector<std::string> col(arCount);
                const uint32_t* ids = static_cast<const uint32_t*>(p.ids->contents());
                for (uint32_t i = 0; i < arCount; ++i) {
                    col[i] = p.dict->lookupString(ids[i]);
                }
                tableResult.stringNames.push_back(p.name);
                tableResult.stringCols.push_back(std::move(col));
            }
            // Non-AR dict columns
            if (!hasAR) {
                for (const auto& [name, dict] : currentCtx.dictCols) {
                    if (!dict.valid()) continue;
                    tableResult.stringNames.push_back(name);
                    tableResult.stringCols.push_back(dict.materialize());
                }
            }
            // Materialize flat-only string columns not yet handled above
            for (const auto& [name, fc] : currentCtx.flatStringCols) {
                if (!fc.chars) continue;
                if (currentCtx.dictCols.count(name)) continue; // Already handled by dict loop
                // Already handled by stringCols loop if it wasn't skipped
                bool alreadyHandled = false;
                for (const auto& sn : tableResult.stringNames) {
                    if (sn == name) { alreadyHandled = true; break; }
                }
                if (alreadyHandled) continue;
                if (hasAR) {
                    auto r = GpuOps::gatherFlatString(fc.chars, fc.offsets, fc.lengths,
                                                       currentCtx.activeRowsGPU, arCount, true);
                    if (r.chars) {
                        const uint32_t* offs = static_cast<const uint32_t*>(r.offsets->contents());
                        const uint32_t* lens = static_cast<const uint32_t*>(r.lengths->contents());
                        const char* ch = static_cast<const char*>(r.chars->contents());
                        std::vector<std::string> col(arCount);
                        for (uint32_t i = 0; i < arCount; ++i) col[i].assign(ch + offs[i], lens[i]);
                        tableResult.stringNames.push_back(name);
                        tableResult.stringCols.push_back(std::move(col));
                    }
                } else {
                    const uint32_t* offs = static_cast<const uint32_t*>(fc.offsets->contents());
                    const uint32_t* lens = static_cast<const uint32_t*>(fc.lengths->contents());
                    const char* ch = static_cast<const char*>(fc.chars->contents());
                    uint32_t cnt = std::min((uint32_t)arCount, fc.rowCount);
                    std::vector<std::string> col(cnt);
                    for (uint32_t i = 0; i < cnt; ++i) col[i].assign(ch + offs[i], lens[i]);
                    tableResult.stringNames.push_back(name);
                    tableResult.stringCols.push_back(std::move(col));
                }
            }
            tableResult.rowCount = arCount;
        };
        materialize();
        if (debug) {
            LOG_INFO("Exec", "Scan-only pipeline: materialized " << tableResult.rowCount << " rows (" << tableResult.u32Names.size() << " u32, " << tableResult.f32Names.size() << " f32, " << tableResult.stringNames.size() << " string cols)\n");
        }
    }

}

void resolveOutputColumnNames(
    EvalContext& currentCtx, TableResult& tableResult, bool debug
) {
    // Post-process: Clean up column names and mark single-char columns
    const auto& schema = SchemaRegistry::instance();

    // Track positional ref -> original column name mappings from context
    // The GroupBy stores both "l_returnflag" and "#0" -> same data
    std::map<std::string, std::string> posToOriginal;

    bool debugCleanup = env_truthy("GPUDB_DEBUG_OPS");
    if (debugCleanup) {
        LOG_INFO("Exec", "Cleanup: currentCtx.u32Cols=");
        for (const auto& [n,v] : currentCtx.u32Cols) std::cerr << n << "(" << v.size() << ") ";
        LOG_INFO("ENGINE", "\n");
    }

    // Track which positional refs have been assigned (to avoid double-assignment when columns have same data)
    std::set<std::string> assignedPosRefs;
    std::set<std::string> usedOriginalNames;

    // Build mapping for u32 columns
    for (const auto& [name, vec] : currentCtx.u32Cols) {
        if (name.size() >= 2 && name[0] == '#' && std::isdigit(name[1])) continue;
        if (name.find("__internal_") != std::string::npos) continue;  // Skip internal wrapper names
        if (usedOriginalNames.count(name)) continue;  // Already used this name for another pos ref
        // Find if there's a positional ref with same data
        for (const auto& [pos, posVec] : currentCtx.u32Cols) {
            if (pos.size() >= 2 && pos[0] == '#' && std::isdigit(pos[1])) {
                if (assignedPosRefs.count(pos)) continue;  // Already assigned
                // Fast path: GPU buffer pointer identity (O(1) instead of O(N) vector compare)
                bool match = false;
                auto gpuPos = currentCtx.u32ColsGPU.find(pos);
                auto gpuName = currentCtx.u32ColsGPU.find(name);
                if (gpuPos != currentCtx.u32ColsGPU.end() && gpuName != currentCtx.u32ColsGPU.end()
                    && gpuPos->second && gpuName->second
                    && gpuPos->second.get() == gpuName->second.get()) {
                    match = true;
                } else {
                    match = posVec.size() == vec.size() && posVec == vec;
                }
                if (match) {
                    posToOriginal[pos] = name;
                    assignedPosRefs.insert(pos);
                    usedOriginalNames.insert(name);
                    break;  // Found a match for this name, move to next
                }
            }
        }
    }

    // Build mapping for f32 columns
    for (const auto& [name, vec] : currentCtx.f32Cols) {
        if (name.size() >= 2 && name[0] == '#' && std::isdigit(name[1])) continue;
        if (name.rfind("SUM_", 0) == 0) continue;  // Skip SUM_#N variants
        if (name.find("__internal_") != std::string::npos) continue;  // Skip internal wrapper names
        if (usedOriginalNames.count(name)) continue;  // Already used this name for another pos ref
        // Find if there's a positional ref with same data
        for (const auto& [pos, posVec] : currentCtx.f32Cols) {
            if (pos.size() >= 2 && pos[0] == '#' && std::isdigit(pos[1])) {
                if (assignedPosRefs.count(pos)) continue;  // Already assigned
                // Fast path: GPU buffer pointer identity (O(1) instead of O(N) vector compare)
                bool match = false;
                auto gpuPos = currentCtx.f32ColsGPU.find(pos);
                auto gpuName = currentCtx.f32ColsGPU.find(name);
                if (gpuPos != currentCtx.f32ColsGPU.end() && gpuName != currentCtx.f32ColsGPU.end()
                    && gpuPos->second && gpuName->second
                    && gpuPos->second.get() == gpuName->second.get()) {
                    match = true;
                } else {
                    match = posVec.size() == vec.size() && posVec == vec;
                }
                if (match) {
                    posToOriginal[pos] = name;
                    assignedPosRefs.insert(pos);
                    usedOriginalNames.insert(name);
                    break;  // Found a match for this name, move to next
                }
            }
        }
    }

    if (debugCleanup) {
        LOG_INFO("Exec", "Cleanup: posToOriginal mappings:\n");
        for (const auto& [pos, orig] : posToOriginal) {
            LOG_INFO("ENGINE", "  " << pos << " -> " << orig);
        }
        LOG_DEBUG("Exec", "Cleanup: tableResult.u32Names=");
        if (debug) for (const auto& n : tableResult.u32Names) std::cerr << "'" << n << "' ";
        LOG_DEBUG("ENGINE", "\n");
    }

    // Clean up u32 column names
    for (size_t i = 0; i < tableResult.u32Names.size(); ++i) {
        std::string& name = tableResult.u32Names[i];
        name = cleanupColumnName(name);

        // If it's a positional ref, try to map to original name
        if (name.size() >= 2 && name[0] == '#' && std::isdigit(name[1])) {
            auto it = posToOriginal.find(name);
            if (it != posToOriginal.end()) {
                name = it->second;
            }
        }

        // Check if single-char column
        std::string table = tableForColumn(name);
        if (schema.isSingleCharColumn(table, name)) {
            tableResult.singleCharCols.insert(name);
        }
    }

    // Clean up f32 column names
    for (size_t i = 0; i < tableResult.f32Names.size(); ++i) {
        std::string& name = tableResult.f32Names[i];
        name = cleanupColumnName(name);

        // DuckDB scalar subquery CASE wrapper cleanup:
        // Names like "')) ELSE "first"(max(total_revenue)) END" should become "total_revenue"
        if (name.find("CASE") != std::string::npos || name.find("ELSE") != std::string::npos) {
            if (name.find("\"error\"(") != std::string::npos || name.find("\"first\"(") != std::string::npos) {
                // Extract innermost meaningful identifier from ELSE branch
                size_t elsePos = name.find("ELSE");
                size_t endPos = name.rfind("END");
                if (elsePos == std::string::npos) { elsePos = 0; }
                if (endPos == std::string::npos) { endPos = name.size(); }
                std::string tail = name.substr(elsePos, endPos - elsePos);
                // Look for the deepest parenthesized identifier: max(total_revenue) -> total_revenue
                // Find the innermost '(' and matching ')'
                size_t lastOpen = tail.rfind('(');
                if (lastOpen != std::string::npos) {
                    size_t close = tail.find(')', lastOpen);
                    if (close != std::string::npos && close > lastOpen + 1) {
                        name = tail.substr(lastOpen + 1, close - lastOpen - 1);
                    }
                }
            }
        }

        // Map positional refs to actual names
        if (name.size() >= 2 && name[0] == '#' && std::isdigit(name[1])) {
            auto it = posToOriginal.find(name);
            if (it != posToOriginal.end()) {
                name = it->second;
            }
        }
    }

}

void recoverStringColumns(
    EvalContext& currentCtx, TableResult& tableResult,
    const std::string& datasetPath, bool debug
) {
    const auto& schema = SchemaRegistry::instance();
    bool debugCleanup = debug;
    // --- Convert U32 hashes back to strings ---
    std::vector<std::string> new_u32_names;
    std::vector<std::vector<uint32_t>> new_u32_cols;
    std::vector<size_t> u32_remap(tableResult.u32Names.size());
    std::vector<bool> is_converted(tableResult.u32Names.size(), false);
    std::vector<size_t> string_converted_idx(tableResult.u32Names.size(), 0);

    for (size_t i = 0; i < tableResult.u32Names.size(); ++i) {
        std::string colName = tableResult.u32Names[i];
        std::string tableName = tableForColumn(colName);
        bool converted = false;

        // Lazy-fetch from GPU if CPU vector is empty (scan lazy-fetch pattern)
        if (tableResult.u32Cols[i].empty() &&
            i < tableResult.u32ColsGPU.size() && tableResult.u32ColsGPU[i]) {
            uint32_t rc = tableResult.rowCount;
            tableResult.u32Cols[i].resize(rc);
            std::memcpy(tableResult.u32Cols[i].data(),
                        tableResult.u32ColsGPU[i]->contents(), rc * sizeof(uint32_t));
        }

        // Check if a string column with this name already exists (e.g. from GroupBy string recovery)
        bool alreadyHasString = false;
        for (size_t si = 0; si < tableResult.stringNames.size(); ++si) {
            if (tableResult.stringNames[si] == colName) {
                // String column already exists — just mark u32 for removal, keep existing string
                is_converted[i] = true;
                string_converted_idx[i] = si;
                converted = true;
                alreadyHasString = true;
            }
        }
        if (alreadyHasString) continue;

        if (!tableName.empty()) {
            auto tSchema = schema.getTable(tableName);
            if (tSchema) {
                auto cSchema = tSchema->getColumn(colName);
                if (cSchema && cSchema->type == ColumnType::StringHash) {
                    // GPU-native dict: try direct dict-ID→string lookup first (O(1), no hashing)
                    auto dictIt = currentCtx.dictCols.find(colName);
                    if (dictIt != currentCtx.dictCols.end() && dictIt->second.valid()) {
                        const auto& dict = dictIt->second;
                        const auto& u32col = tableResult.u32Cols[i];

                        // Check if the u32 values are dictionary IDs (max val < dict size)
                        // vs FNV1a hashes (values are hash codes)
                        bool isDictIds = true;
                        uint32_t maxVal = 0;
                        for (uint32_t val : u32col) { if (val > maxVal) maxVal = val; }
                        if (maxVal >= dict.dictionary.size()) isDictIds = false;

                        std::vector<std::string> strCol;
                        strCol.reserve(u32col.size());

                        if (isDictIds) {
                            // Direct dictionary ID lookup — fastest path
                            for (uint32_t val : u32col) {
                                strCol.push_back(dict.lookupString(val));
                            }
                        } else {
                            // FNV1a hash reverse-mapping (legacy path)
                            std::unordered_map<uint32_t, std::string> hashMap;
                            hashMap.reserve(dict.dictionary.size());
                            for (const auto& s : dict.dictionary) {
                                hashMap[GpuOps::fnv1a32(s)] = s;
                            }
                            for (uint32_t val : u32col) {
                                auto hit = hashMap.find(val);
                                if (hit != hashMap.end()) strCol.push_back(hit->second);
                                else strCol.push_back(std::to_string(val));
                            }
                        }

                        tableResult.stringNames.push_back(colName);
                        tableResult.stringCols.push_back(std::move(strCol));
                        is_converted[i] = true;
                        string_converted_idx[i] = tableResult.stringNames.size() - 1;
                        converted = true;
                    } else {
                        // Fallback: try ctx.stringCols
                        auto strIt = currentCtx.stringCols.find(colName);
                        if (strIt != currentCtx.stringCols.end() && !strIt->second.empty()) {
                            std::unordered_map<uint32_t, std::string> hashMap;
                            hashMap.reserve(strIt->second.size());
                            for (const auto& s : strIt->second) {
                                hashMap[GpuOps::fnv1a32(s)] = s;
                            }

                            std::vector<std::string> strCol;
                            strCol.reserve(tableResult.u32Cols[i].size());
                            for (uint32_t val : tableResult.u32Cols[i]) {
                                auto hit = hashMap.find(val);
                                if (hit != hashMap.end()) strCol.push_back(hit->second);
                                else strCol.push_back(std::to_string(val));
                            }

                            tableResult.stringNames.push_back(colName);
                            tableResult.stringCols.push_back(std::move(strCol));
                            is_converted[i] = true;
                            string_converted_idx[i] = tableResult.stringNames.size() - 1;
                            converted = true;
                        } else {
                            // Last resort: re-read from disk (should be rare with always-flatten)
                            if (debugCleanup) {
                                LOG_WARN("Exec", "WARNING: disk re-read for output reverse-map of " << colName);
                            }
                            auto raw = GpuOps::loadStringColumnRaw(datasetPath, tableName, colName);
                            std::unordered_map<uint32_t, std::string> map;
                            map.reserve(raw.size());
                            for (const auto& s : raw) {
                                map[GpuOps::fnv1a32(s)] = s;
                            }

                            std::vector<std::string> strCol;
                            strCol.reserve(tableResult.u32Cols[i].size());
                            for (uint32_t val : tableResult.u32Cols[i]) {
                                if (map.count(val)) strCol.push_back(map[val]);
                                else strCol.push_back(std::to_string(val));
                            }

                            tableResult.stringNames.push_back(colName);
                            tableResult.stringCols.push_back(std::move(strCol));
                            is_converted[i] = true;
                            string_converted_idx[i] = tableResult.stringNames.size() - 1;
                            converted = true;
                        }
                    }
                } else if (cSchema && cSchema->type == ColumnType::Float32) {
                    // GroupBy bit-reinterprets f32 keys as u32. Restore to f32.
                    std::vector<float> f32Col(tableResult.u32Cols[i].size());
                    std::memcpy(f32Col.data(), tableResult.u32Cols[i].data(), f32Col.size() * sizeof(float));
                    tableResult.f32Names.push_back(colName);
                    tableResult.f32Cols.push_back(std::move(f32Col));
                    is_converted[i] = true;
                    // Mark as f32-converted (use a high sentinel so it doesn't collide with string index)
                    string_converted_idx[i] = SIZE_MAX;
                    converted = true;
                }
            }
        }

        if (!converted) {
            new_u32_names.push_back(colName);
            new_u32_cols.push_back(std::move(tableResult.u32Cols[i]));
            u32_remap[i] = new_u32_names.size() - 1;
        }
    }

    // Apply strict update to u32 columns
    tableResult.u32Names = std::move(new_u32_names);
    tableResult.u32Cols = std::move(new_u32_cols);

    // Update the order refs to use cleaned names
    for (auto& ref : tableResult.order) {
        ref.name = cleanupColumnName(ref.name);
        if (ref.kind == TableResult::ColRef::Kind::U32) {
            if (ref.index < is_converted.size()) {
                if (is_converted[ref.index]) {
                    if (string_converted_idx[ref.index] == SIZE_MAX) {
                        // Converted to f32 (was bit-reinterpreted u32)
                        ref.kind = TableResult::ColRef::Kind::F32;
                        // Find the f32 index by name
                        for (size_t fi = 0; fi < tableResult.f32Names.size(); ++fi) {
                            if (tableResult.f32Names[fi] == ref.name) {
                                ref.index = fi;
                            }
                        }
                    } else {
                        ref.kind = TableResult::ColRef::Kind::String;
                        ref.index = string_converted_idx[ref.index];
                        if (ref.index < tableResult.stringNames.size()) {
                             ref.name = tableResult.stringNames[ref.index];
                        }
                    }
                } else {
                    ref.index = u32_remap[ref.index];
                    if (ref.index < tableResult.u32Names.size()) {
                        ref.name = tableResult.u32Names[ref.index];
                    }
                }
            }
        } else if (ref.kind == TableResult::ColRef::Kind::F32 && ref.index < tableResult.f32Names.size()) {
            ref.name = tableResult.f32Names[ref.index];
        } else if (ref.kind == TableResult::ColRef::Kind::String && ref.index < tableResult.stringNames.size()) {
            ref.name = tableResult.stringNames[ref.index];
        }
    }

    // Update GPU timing from kernel timer
    tableResult.gpuMs = KernelTimer::instance().totalGpuMs();
}

void filterOutputColumns(
    const Plan& plan, TableResult& tableResult,
    bool isScalarAggregate, bool debug
) {
    // Filter final output to only include columns from the plan's outputColumns
    // This strips intermediate join keys and sort-only columns
    if (!plan.outputColumns.empty() && !isScalarAggregate) {
        // Build set of expected output column names (lowercased for fuzzy match)
        std::set<std::string> expectedCols;
        for (const auto& c : plan.outputColumns) {
            std::string lower = c;
            std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
            expectedCols.insert(lower);
        }

        auto isExpected = [&](const std::string& name) -> bool {
            std::string lower = name;
            std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
            if (expectedCols.count(lower)) return true;
            // Also try base_ident (strip table prefix)
            std::string base = base_ident(lower);
            if (expectedCols.count(base)) return true;
            // Check if any expected col matches the base
            for (const auto& ec : expectedCols) {
                if (base_ident(ec) == base) return true;
            }
            return false;
        };

        if (tableResult.order.empty()) {
            // No order vector - filter u32/f32/string columns directly
            TableResult filtered;
            filtered.rowCount = tableResult.rowCount;
            filtered.singleCharCols = tableResult.singleCharCols;

            for (size_t i = 0; i < tableResult.u32Names.size(); ++i) {
                if (isExpected(tableResult.u32Names[i])) {
                    filtered.u32Names.push_back(tableResult.u32Names[i]);
                    filtered.u32Cols.push_back(tableResult.u32Cols[i]);
                }
            }
            for (size_t i = 0; i < tableResult.f32Names.size(); ++i) {
                if (isExpected(tableResult.f32Names[i])) {
                    filtered.f32Names.push_back(tableResult.f32Names[i]);
                    filtered.f32Cols.push_back(tableResult.f32Cols[i]);
                }
            }
            for (size_t i = 0; i < tableResult.stringNames.size(); ++i) {
                if (isExpected(tableResult.stringNames[i])) {
                    filtered.stringNames.push_back(tableResult.stringNames[i]);
                    filtered.stringCols.push_back(tableResult.stringCols[i]);
                }
            }

            tableResult.u32Cols = std::move(filtered.u32Cols);
            tableResult.u32Names = std::move(filtered.u32Names);
            tableResult.f32Cols = std::move(filtered.f32Cols);
            tableResult.f32Names = std::move(filtered.f32Names);
            tableResult.stringCols = std::move(filtered.stringCols);
            tableResult.stringNames = std::move(filtered.stringNames);
        } else {
            // Filter order vector
            std::vector<TableResult::ColRef> filteredOrder;
            for (const auto& ref : tableResult.order) {
                if (isExpected(ref.name)) {
                    filteredOrder.push_back(ref);
                }
            }
            tableResult.order = std::move(filteredOrder);
        }

        if (debug) {
            LOG_INFO("Exec", "Final output filter: " << plan.outputColumns.size() << " expected cols, " << "result has " << tableResult.u32Names.size() << " u32, " << tableResult.f32Names.size() << " f32, " << tableResult.stringNames.size() << " string cols\n");
        }
    }

}

// ── Extract DELIM correlation columns from self-comparison join conditions ──
std::unordered_map<std::string, std::vector<std::string>>
extractDelimCorrelationCols(const Plan& plan, bool debug) {
    std::unordered_map<std::string, std::vector<std::string>> result;

    for (size_t ni = 0; ni < plan.nodes.size(); ++ni) {
        if (plan.nodes[ni].type != IRNode::Type::Join) continue;

        const auto& join = plan.nodes[ni].asJoin();
        const std::string& cond = join.conditionStr;

        std::vector<std::string> corrCols;
        auto condParts = splitConditionByAnd(cond);
        for (const auto& part : condParts) {
            std::string col = parseSelfComparison(part);
            if (!col.empty()) corrCols.push_back(col);
        }
        if (corrCols.empty()) continue;

        // Find the delim group by looking backward for tmpl_delim_lhs_* scans
        std::string delimGroup;
        for (size_t si = 0; si < ni; ++si) {
            if (plan.nodes[si].type == IRNode::Type::Scan) {
                const std::string& tbl = plan.nodes[si].asScan().table;
                if (tbl.find("tmpl_delim_lhs_") == 0) delimGroup = tbl;
            }
        }
        if (delimGroup.empty()) continue;

        auto& existing = result[delimGroup];
        for (const auto& c : corrCols) {
            if (std::find(existing.begin(), existing.end(), c) == existing.end())
                existing.push_back(c);
        }
    }

    if (debug && !result.empty()) {
        for (const auto& [group, cols] : result) {
            LOG_INFO("Exec", "DELIM correlation: " << group << " -> [");
            for (size_t i = 0; i < cols.size(); ++i) {
                if (i) std::cerr << ", ";
                LOG_INFO("ENGINE", cols[i]);
            }
            LOG_INFO("ENGINE", "]\n");
        }
    }
    return result;
}

} // namespace engine
