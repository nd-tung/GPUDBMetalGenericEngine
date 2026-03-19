#include "GpuExecutor.hpp"
#include "GpuExecutorDetail.hpp"
#include "QueryExecutionContext.hpp"
#include "Operators.hpp"
#include "GpuColumnStore.hpp"

#include "Planner.hpp"
#include "KernelTimer.hpp"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <iostream>
#include <map>
#include <set>
#include <unordered_set>
#include "Logger.hpp"

namespace engine {

static void delimScanDedup(
    EvalContext& ctx, const std::string& tableKey,
    const std::unordered_map<std::string, std::vector<std::string>>& delimCorrelationCols,
    const std::vector<std::string>& scanColumns,
    bool markCorrelation, bool debug)
{
    if (ctx.rowCount <= 1) return;

    // No bulk GPU→CPU download — deduplicateContext reads from GPU directly.
    // We only need column name discovery here, which works with empty sentinels
    // or GPU buffer presence.

    // Helper: check if a u32 column exists (either CPU or GPU)
    auto hasU32Col = [&](const std::string& c) -> bool {
        if (ctx.u32ColsGPU.count(c) && ctx.u32ColsGPU.at(c)) return true;
        if (ctx.u32Cols.count(c) && !ctx.u32Cols.at(c).empty()) return true;
        // Empty sentinel with GPU buffer counts
        if (ctx.u32Cols.count(c) && ctx.u32ColsGPU.count(c) && ctx.u32ColsGPU.at(c)) return true;
        return false;
    };

    // Prefer correlation columns from DELIM_JOIN if available
    std::vector<std::string> dedupCols;
    for (const auto& [grp, cols] : delimCorrelationCols) {
        if (tableKey.find(grp) == 0 || grp.find(tableKey) == 0) {
            for (const auto& c : cols) {
                if (hasU32Col(c))
                    dedupCols.push_back(c);
            }
        }
    }
    // Fallback: use all u32 scan columns
    if (dedupCols.empty()) {
        for (const auto& c : scanColumns) {
            if (hasU32Col(c))
                dedupCols.push_back(c);
        }
    }

    if (debug) {
        LOG_INFO("Exec", "DELIM_SCAN dedup: dedupCols=[");
        for (size_t ci = 0; ci < dedupCols.size(); ++ci) { if (ci) std::cerr << ","; std::cerr << dedupCols[ci]; }
        LOG_INFO("ENGINE", "]\n");
    }
    if (dedupCols.empty()) return;

    uint32_t newCount = deduplicateContext(ctx, dedupCols, debug);
    if (newCount > 0) {
        ctx.f32Cols.clear(); ctx.f32ColsGPU.clear(); ctx.stringCols.clear();
        std::set<std::string> keepCols(dedupCols.begin(), dedupCols.end());
        for (auto it2 = ctx.u32Cols.begin(); it2 != ctx.u32Cols.end(); ) {
            if (keepCols.find(it2->first) == keepCols.end()) it2 = ctx.u32Cols.erase(it2);
            else ++it2;
        }
        for (auto it2 = ctx.u32ColsGPU.begin(); it2 != ctx.u32ColsGPU.end(); ) {
            if (keepCols.find(it2->first) == keepCols.end()) it2 = ctx.u32ColsGPU.erase(it2);
            else ++it2;
        }
        if (markCorrelation) {
            LOG_DEBUG("Exec", "DELIM_SCAN: stripped to correlation cols only: [" << dedupCols.size() << " cols]\n");
            for (const auto& dc : dedupCols)
                ctx.isDelimCorrelation.insert(dc);
        }
    }
}

// Helper: Filter an EvalContext to only keep columns referenced by a scan
// node's projection list and pushed-down filter, supporting instance suffixes.
static void filterContextColumns(
    EvalContext& ctx, const std::vector<std::string>& scanColumns,
    const TypedExprPtr& scanFilter, bool debug)
{
    if (scanColumns.empty()) return;

    std::set<std::string> keepCols(scanColumns.begin(), scanColumns.end());
    if (scanFilter) collectColumnsFromExpr(scanFilter, keepCols);

    auto shouldKeep = [&](const std::string& colName) -> bool {
        if (keepCols.count(colName)) return true;
        auto lastUnderscore = colName.rfind('_');
        if (lastUnderscore != std::string::npos) {
            std::string suffix = colName.substr(lastUnderscore + 1);
            bool allDigits = !suffix.empty() && std::all_of(suffix.begin(), suffix.end(), ::isdigit);
            if (allDigits && keepCols.count(colName.substr(0, lastUnderscore)))
                return true;
        }
        return false;
    };

    for (auto cit = ctx.u32Cols.begin(); cit != ctx.u32Cols.end(); )
        if (!shouldKeep(cit->first)) cit = ctx.u32Cols.erase(cit); else ++cit;
    for (auto cit = ctx.u32ColsGPU.begin(); cit != ctx.u32ColsGPU.end(); )
        if (!shouldKeep(cit->first)) cit = ctx.u32ColsGPU.erase(cit); else ++cit;
    for (auto cit = ctx.f32Cols.begin(); cit != ctx.f32Cols.end(); )
        if (!shouldKeep(cit->first)) cit = ctx.f32Cols.erase(cit); else ++cit;
    for (auto cit = ctx.f32ColsGPU.begin(); cit != ctx.f32ColsGPU.end(); )
        if (!shouldKeep(cit->first)) cit = ctx.f32ColsGPU.erase(cit); else ++cit;
    for (auto cit = ctx.stringCols.begin(); cit != ctx.stringCols.end(); )
        if (!shouldKeep(cit->first)) cit = ctx.stringCols.erase(cit); else ++cit;
    for (auto cit = ctx.flatStringCols.begin(); cit != ctx.flatStringCols.end(); )
        if (!shouldKeep(cit->first)) cit = ctx.flatStringCols.erase(cit); else ++cit;
    for (auto cit = ctx.dictCols.begin(); cit != ctx.dictCols.end(); )
        if (!shouldKeep(cit->first)) cit = ctx.dictCols.erase(cit); else ++cit;

    if (debug) {
        LOG_INFO("Exec", "Scan column filter: kept cols:");
        for (const auto& c : keepCols) std::cerr << " " << c;
        LOG_INFO("ENGINE", "\n");
    }
}

void handleExecScanNode(const IRScan& scan, size_t nodeIdx, QueryExecutionContext& qctx) {
    auto& currentCtx = qctx.currentCtx;
    const auto& plan = qctx.plan;
    bool debug = qctx.debug;
    const auto& scanInstanceMap = qctx.scanInstanceMap;
    const auto& delimCorrelationCols = qctx.delimCorrelationCols;
    auto& tableContexts = qctx.joinState.tableContexts;
    auto& savedPipelines = qctx.joinState.savedPipelines;
    auto& savedPipelineTables = qctx.joinState.savedPipelineTables;
    auto& joinedTables = qctx.joinState.joinedTables;
    auto& hasPipeline = qctx.joinState.hasPipeline;

    // Skip empty scans (DELIM_SCAN markers)
    if (scan.table.empty()) {
        if (debug) {
            LOG_INFO("Exec", "Skipping empty scan (DELIM_SCAN marker)\n");
        }
    }

    // Check if this scan has an instance key (for multi-instance tables)
    std::string tableKey = scan.table;
    auto instIt = scanInstanceMap.find(nodeIdx);
    if (instIt != scanInstanceMap.end()) {
        tableKey = instIt->second.instanceKey;
    }

    auto it = tableContexts.find(tableKey);

    // Fallback for runtime tables (tmpl_) that were saved without instance suffixes
    if (it == tableContexts.end() && instIt != scanInstanceMap.end()) {
         auto baseIt = tableContexts.find(instIt->second.baseTable);
         if (baseIt != tableContexts.end()) {
             LOG_DEBUG("Exec", "Scan fallback: using base table " << instIt->second.baseTable << " for " << tableKey);
             it = baseIt;
         }
    }

    // Fallback: tmpl_delim_lhs_N -> tmpl_join_N aliasing
    if (it == tableContexts.end()) {
        // Check for raw key first
        if (tableKey.find("tmpl_delim_lhs_") == 0) {
            std::string suffix = tableKey.substr(15); 
            // Remove instance suffix _X if present (e.g. _1, _2)
            size_t instParams = suffix.find('_');
            if (instParams != std::string::npos) {
                suffix = suffix.substr(0, instParams);
            }

            std::string altKey = "tmpl_join_" + suffix;
            auto altIt = tableContexts.find(altKey);
            if (altIt != tableContexts.end()) {
                 LOG_DEBUG("Exec", "Scan fallback (DELIM aliasing): using " << altKey << " for " << tableKey);
                 it = altIt;
            }
        } 
        // Also check baseTable if instance key failed
        else if (instIt != scanInstanceMap.end()) {
            std::string base = instIt->second.baseTable;
             if (base.find("tmpl_delim_lhs_") == 0) {
                 std::string suffix = base.substr(15);
                 std::string altKey = "tmpl_join_" + suffix;
                 auto altIt = tableContexts.find(altKey);
                 if (altIt != tableContexts.end()) {
                      LOG_DEBUG("Exec", "Scan fallback (DELIM aliasing base): using " << altKey << " for " << tableKey);
                      it = altIt;
                 }
             }
        }
    }

    // Fallback: try any available tmpl_delim_lhs table
    if (it == tableContexts.end() && tableKey.find("tmpl_delim_lhs_") == 0) {
         // Generic fallback: find any available delim LHS table
         for(auto rit = tableContexts.begin(); rit != tableContexts.end(); ++rit) {
             if (rit->first.find("tmpl_delim_lhs_") == 0) {
                  LOG_DEBUG("Exec", "Scan fallback (DELIM Find): using " << rit->first << " for " << tableKey);
                  it = rit;
             }
         }
    }

    LOG_DEBUG("Exec", "Scan Loop lookup: " << tableKey << " found=" << (it != tableContexts.end()));
    if (it != tableContexts.end()) LOG_DEBUG("Exec", "Scan Table Size: " << it->second.rowCount);
    if (debug) {
        LOG_INFO("Exec", "Scan isDelimScan=" << scan.isDelimScan << " columns=[");
        for (size_t ci=0; ci<scan.columns.size(); ++ci) { if (ci) std::cerr << ","; std::cerr << scan.columns[ci]; }
        LOG_INFO("ENGINE", "]\n");
    }
    if (it != tableContexts.end()) {
        // Check if this Scan is followed by a Join - if so, this is loading
        // the build side, so we should NOT clobber the pipeline context
        bool isJoinBuildSide = false;
        if (hasPipeline && nodeIdx + 1 < plan.nodes.size()) {
            // Look ahead to see if the next non-Filter/Project node is a Join
            for (size_t ahead = nodeIdx + 1; ahead < plan.nodes.size(); ++ahead) {
                auto aheadType = plan.nodes[ahead].type;
                if (aheadType == IRNode::Type::Join) {
                    const auto& joinNode = plan.nodes[ahead].asJoin();
                    // If the join explicitly specifies a different right table,
                    // then this scan is NOT the build side for that join (it's likely a new LHS).
                    if (!joinNode.rightTable.empty() && joinNode.rightTable != tableKey) {
                        isJoinBuildSide = false;
                    } else {
                        isJoinBuildSide = true;
                    }
                } else if (aheadType != IRNode::Type::Filter && 
                           aheadType != IRNode::Type::Project) {
                    break; // Not a join, not building for join
                }
            }
        }

        if (isJoinBuildSide) {
            // Don't clobber the pipeline - just make sure this table is in tableContexts
            // Apply pushed-down filter to the table's context (these are pre-filtered
            // in the planner to only include precise filters, not optimizer-derived ones)
            if (scan.filter) {
                if (debug) {
                    LOG_INFO("Exec", "Applying scan filter for build-side table " << tableKey);
                }
                EvalContext& tableCtx = tableContexts[tableKey];
                GpuExecutor::executeFilter(IRFilter{scan.filter, ""}, tableCtx);
                if (debug) {
                    LOG_INFO("Exec", "After filter: " << tableCtx.rowCount << " rows\n");
                }
            }
            // DELIM_SCAN dedup for build-side tables too
            if (scan.isDelimScan && !scan.columns.empty()) {
                EvalContext& tableCtx = tableContexts[tableKey];
                delimScanDedup(tableCtx, tableKey, delimCorrelationCols, scan.columns, false, debug);
            }
            if (debug) {
                LOG_INFO("Exec", "Scan " << tableKey << " (for join build): "  << tableContexts[tableKey].rowCount << " rows\n");
            }
        } else {
            // Starting a new pipeline - save previous pipeline if it has joined data
            if (hasPipeline && !joinedTables.empty() && currentCtx.rowCount > 0) {
                savedPipelines.push_back(currentCtx);
                savedPipelineTables.push_back(joinedTables);
                if (debug) {
                    LOG_INFO("Exec", "Saved pipeline with tables: ");
                    for (const auto& t : joinedTables) std::cerr << t << " ";
                    LOG_INFO("ENGINE", "(" << currentCtx.rowCount << " rows)\n");
                }
            }

            // Start/continue current context with this table.
            // Move string data (raw strings + dict encodings) instead of
            // deep-copying. Source retains flatStringCols (GPU buffers, O(1)
            // shared_ptr copy) and can rebuild on demand via ensureStringCol().
            {
                auto& src = it->second;
                currentCtx.u32Cols = src.u32Cols;
                currentCtx.f32Cols = src.f32Cols;
                currentCtx.u32ColsGPU = src.u32ColsGPU;
                currentCtx.f32ColsGPU = src.f32ColsGPU;
                currentCtx.stringCols = std::move(src.stringCols);
                currentCtx.flatStringCols = src.flatStringCols;
                currentCtx.dictCols = src.dictCols;
                currentCtx.columnAliases = src.columnAliases;
                currentCtx.activeRows = src.activeRows;
                currentCtx.activeRowsGPU = src.activeRowsGPU;
                currentCtx.activeRowsCountGPU = src.activeRowsCountGPU;
                currentCtx.rowCount = src.rowCount;
                currentCtx.isScalarResult = src.isScalarResult;
                currentCtx.isDelimCorrelation = src.isDelimCorrelation;
                currentCtx.aggregateCounter = src.aggregateCounter;
            }
            currentCtx.currentTable = tableKey;
            joinedTables.clear();
            joinedTables.insert(tableKey);

            // Filter context to only include columns needed by this scan
            // This prevents extra columns (e.g., string match cols) from
            // leaking into the pipeline and causing name collisions later
            filterContextColumns(currentCtx, scan.columns, scan.filter, debug);

            // DELIM_SCAN deduplication: In DuckDB's decorrelated plans,
            // DELIM_SCAN produces the DISTINCT set of correlated keys,
            // while COLUMN_DATA_SCAN produces the full original data.
            // Deduplicate by the scan's projected columns to get distinct keys.
            if (scan.isDelimScan && !scan.columns.empty()) {
                delimScanDedup(currentCtx, tableKey, delimCorrelationCols, scan.columns, true, debug);
            }

            // Alias ps_partkey -> p_partkey for correlated subquery contexts
            if (currentCtx.currentTable.find("tmpl_") == 0) {
                bool hasPS = currentCtx.u32Cols.count("ps_partkey");
                bool hasP = currentCtx.u32Cols.count("p_partkey");
                if (hasPS && !hasP) {
                    LOG_DEBUG("Exec", "Patch: Aliasing ps_partkey -> p_partkey in " << currentCtx.currentTable);
                    currentCtx.u32Cols["p_partkey"] = currentCtx.u32Cols["ps_partkey"];
                    if (currentCtx.u32ColsGPU.count("ps_partkey")) {
                        currentCtx.u32ColsGPU["p_partkey"] = currentCtx.u32ColsGPU["ps_partkey"]; // GpuBuffer copy retains
                    }
                } else if (!hasP && !hasPS) {
                    // Inject p_partkey from global 'part' table as placeholder
                    auto partIt = tableContexts.find("part");
                    if (partIt != tableContexts.end() && partIt->second.u32Cols.count("p_partkey")) {
                         LOG_DEBUG("Exec", "Patch: Injecting global p_partkey from 'part' into " << currentCtx.currentTable);

                         // Create a buffer of correct size
                         std::vector<uint32_t> dummy(currentCtx.rowCount, 0); 

                         // Copy the first N IDs from part table if available to act as placeholder
                         const auto& src = partIt->second.u32Cols.at("p_partkey");
                         for(size_t i=0; i<currentCtx.rowCount && i<src.size(); ++i) {
                             dummy[i] = src[i];
                         }

                         currentCtx.u32Cols["p_partkey"] = dummy;
                         currentCtx.u32ColsGPU["p_partkey"]= GpuOps::createBuffer(dummy.data(), dummy.size() * sizeof(uint32_t));
                    }
                }
            }

            // Apply pushed-down filter if present (these are pre-filtered
            // in the planner to only include precise filters)
            if (scan.filter) {
                if (debug) {
                    LOG_INFO("Exec", "Applying scan filter for pipeline table " << tableKey);
                }
                GpuExecutor::executeFilter(IRFilter{scan.filter, ""}, currentCtx);
                // Update tableContexts with filtered data for joins
                tableContexts[tableKey] = currentCtx;
            }

            if (debug) {
                LOG_INFO("Exec", "Scan " << tableKey << ": " << currentCtx.rowCount << " rows, u32cols=");
                for (const auto& [k, v] : currentCtx.u32Cols) std::cerr << k << " ";
                LOG_INFO("ENGINE", "f32cols=");
                if (debug) for (const auto& [k, v] : currentCtx.f32Cols) std::cerr << k << " ";
                LOG_DEBUG("ENGINE", "\n");
            }
        }
    }
}

// -- Extracted: compactTableResultAfterFilter --
// Compact tableResult columns via GPU gather using activeRows indices.
// Only compacts when tableResult row count matches the pre-filter context size.
static void compactTableResultAfterFilter(EvalContext& ctx, TableResult& tableResult, bool /*debug*/) {
    if (tableResult.u32Cols.empty() && tableResult.f32Cols.empty()) return;

    // Find the physical buffer size (pre-filter row count)
    size_t physicalRows = 0;
    for (const auto& [name, buf] : ctx.u32ColsGPU) {
        if (buf) { physicalRows = buf->length() / sizeof(uint32_t); break; }
    }
    if (physicalRows == 0) {
        for (const auto& [name, buf] : ctx.f32ColsGPU) {
            if (buf) { physicalRows = buf->length() / sizeof(float); break; }
        }
    }
    bool sizeMatch = (tableResult.rowCount == physicalRows) ||
                     (physicalRows == 0 && !tableResult.u32Cols.empty() &&
                      tableResult.u32Cols[0].size() == ctx.activeRowsCountGPU);

    if (sizeMatch && ctx.activeRowsCountGPU > 0 && ctx.activeRowsGPU) {
        uint32_t arCount = ctx.activeRowsCountGPU;
        auto& s = GpuColumnStore::instance();
        for (size_t ci = 0; ci < tableResult.u32Cols.size(); ++ci) {
            auto& col = tableResult.u32Cols[ci];
            MTL::Buffer* gpuBuf = nullptr;
            if (ci < tableResult.u32Names.size()) {
                auto it = ctx.u32ColsGPU.find(tableResult.u32Names[ci]);
                if (it != ctx.u32ColsGPU.end()) gpuBuf = it->second;
            }
            if (gpuBuf) {
                GpuBuffer dst = GpuOps::gatherU32(gpuBuf, ctx.activeRowsGPU, arCount);
                col.resize(arCount);
                std::memcpy(col.data(), dst->contents(), arCount * sizeof(uint32_t));
            } else {
                MTL::Buffer* src = s.device()->newBuffer(col.data(), col.size() * sizeof(uint32_t), MTL::ResourceStorageModeShared);
                GpuBuffer dst = GpuOps::gatherU32(src, ctx.activeRowsGPU, arCount);
                col.resize(arCount);
                std::memcpy(col.data(), dst->contents(), arCount * sizeof(uint32_t));
                src->release();
            }
        }
        for (size_t ci = 0; ci < tableResult.f32Cols.size(); ++ci) {
            auto& col = tableResult.f32Cols[ci];
            MTL::Buffer* gpuBuf = nullptr;
            if (ci < tableResult.f32Names.size()) {
                auto it = ctx.f32ColsGPU.find(tableResult.f32Names[ci]);
                if (it != ctx.f32ColsGPU.end()) gpuBuf = it->second;
            }
            if (gpuBuf) {
                GpuBuffer dst = GpuOps::gatherF32(gpuBuf, ctx.activeRowsGPU, arCount);
                col.resize(arCount);
                std::memcpy(col.data(), dst->contents(), arCount * sizeof(float));
            } else {
                MTL::Buffer* src = s.device()->newBuffer(col.data(), col.size() * sizeof(float), MTL::ResourceStorageModeShared);
                GpuBuffer dst = GpuOps::gatherF32(src, ctx.activeRowsGPU, arCount);
                col.resize(arCount);
                std::memcpy(col.data(), dst->contents(), arCount * sizeof(float));
                src->release();
            }
        }
        for (auto& col : tableResult.stringCols) {
            if (col.empty()) continue;  // deferred — compacted via ctx
            bool gpuDone = false;
            size_t colIdx = &col - &tableResult.stringCols[0];
            if (colIdx < tableResult.stringNames.size()) {
                const auto& colName = tableResult.stringNames[colIdx];
                auto fit = ctx.flatStringCols.find(colName);
                if (fit != ctx.flatStringCols.end() && fit->second.chars) {
                    auto r = GpuOps::gatherFlatString(
                        fit->second.chars, fit->second.offsets, fit->second.lengths,
                        ctx.activeRowsGPU, arCount, true);
                    if (r.chars) {
                        const uint32_t* offs = static_cast<const uint32_t*>(r.offsets->contents());
                        const uint32_t* lens = static_cast<const uint32_t*>(r.lengths->contents());
                        const char* ch = static_cast<const char*>(r.chars->contents());
                        col.resize(arCount);
                        for (uint32_t i = 0; i < arCount; ++i) {
                            col[i].assign(ch + offs[i], lens[i]);
                        }
                        gpuDone = true;
                    }
                }
            }
            if (!gpuDone) {
                const uint32_t* arPtr = static_cast<const uint32_t*>(ctx.activeRowsGPU->contents());
                std::vector<std::string> filtered;
                filtered.reserve(arCount);
                for (uint32_t i = 0; i < arCount; ++i) {
                    uint32_t idx = arPtr[i];
                    if (idx < col.size()) filtered.push_back(col[idx]);
                }
                col = std::move(filtered);
            }
        }
        tableResult.rowCount = arCount;
    } else if (!sizeMatch) {
        LOG_DEBUG("Exec", "Filter: clearing stale tableResult (size " << tableResult.rowCount << " != physical " << physicalRows << ")\n");
        tableResult.u32Cols.clear();
        tableResult.u32Names.clear();
        tableResult.f32Cols.clear();
        tableResult.f32Names.clear();
        tableResult.stringCols.clear();
        tableResult.stringNames.clear();
        tableResult.order.clear();
        tableResult.clearDeferredStrings();
        tableResult.rowCount = 0;
    }
}

// -- Extracted: compactContextAfterFilter --
// Compact all context columns (GPU + CPU) using activeRows, then
// invalidate stale CPU mirrors and clear activeRows state.
static void compactContextAfterFilter(EvalContext& ctx) {
    if (ctx.activeRowsCountGPU == 0 || !ctx.activeRowsGPU) return;

    uint32_t compactCount = ctx.activeRowsCountGPU;

    // GPU-direct compaction: gather u32/f32 GPU columns
    for (auto& [name, buf] : ctx.u32ColsGPU) {
        if (buf) {
            auto compacted = GpuOps::gatherU32(buf, ctx.activeRowsGPU, compactCount);
            if (compacted) { buf = std::move(compacted); }
        }
    }
    for (auto& [name, buf] : ctx.f32ColsGPU) {
        if (buf) {
            auto compacted = GpuOps::gatherF32(buf, ctx.activeRowsGPU, compactCount);
            if (compacted) { buf = std::move(compacted); }
        }
    }

    // Compact CPU-only columns via GPU gather — retain gathered GPU buffer
    {
        auto& s = GpuColumnStore::instance();
        for (auto& [name, col] : ctx.u32Cols) {
            if (col.size() > compactCount) {
                auto itGpu = ctx.u32ColsGPU.find(name);
                if (itGpu != ctx.u32ColsGPU.end() && itGpu->second) {
                    col.resize(compactCount);
                    std::memcpy(col.data(), itGpu->second->contents(), compactCount * sizeof(uint32_t));
                } else {
                    MTL::Buffer* src = s.device()->newBuffer(col.data(), col.size() * sizeof(uint32_t), MTL::ResourceStorageModeShared);
                    GpuBuffer dst = GpuOps::gatherU32(src, ctx.activeRowsGPU, compactCount);
                    col.resize(compactCount);
                    std::memcpy(col.data(), dst->contents(), compactCount * sizeof(uint32_t));
                    ctx.u32ColsGPU[name] = std::move(dst);
                    src->release();
                }
            }
        }
        for (auto& [name, col] : ctx.f32Cols) {
            if (col.size() > compactCount) {
                auto itGpu = ctx.f32ColsGPU.find(name);
                if (itGpu != ctx.f32ColsGPU.end() && itGpu->second) {
                    col.resize(compactCount);
                    std::memcpy(col.data(), itGpu->second->contents(), compactCount * sizeof(float));
                } else {
                    MTL::Buffer* src = s.device()->newBuffer(col.data(), col.size() * sizeof(float), MTL::ResourceStorageModeShared);
                    GpuBuffer dst = GpuOps::gatherF32(src, ctx.activeRowsGPU, compactCount);
                    col.resize(compactCount);
                    std::memcpy(col.data(), dst->contents(), compactCount * sizeof(float));
                    ctx.f32ColsGPU[name] = std::move(dst);
                    src->release();
                }
            }
        }
    }
    // GPU-native dict compaction
    ctx.compactDictCols(compactCount);
    // GPU-native flat string compaction
    ctx.compactFlatStringCols(compactCount);
    // Invalidate stale stringCols
    for (auto& [name, col] : ctx.stringCols) {
        auto dit = ctx.dictCols.find(name);
        auto fit = ctx.flatStringCols.find(name);
        if ((dit != ctx.dictCols.end() && dit->second.valid()) ||
            (fit != ctx.flatStringCols.end() && fit->second.chars)) {
            col.clear();
        } else if (col.size() > compactCount) {
            const uint32_t* arPtr = static_cast<const uint32_t*>(ctx.activeRowsGPU->contents());
            std::vector<std::string> filtered;
            filtered.reserve(compactCount);
            for (uint32_t i = 0; i < compactCount; ++i) {
                uint32_t idx = arPtr[i];
                if (idx < col.size()) filtered.push_back(col[idx]);
            }
            col = std::move(filtered);
        }
    }

    // Invalidate stale CPU mirrors
    for (auto& [name, buf] : ctx.u32ColsGPU) {
        if (buf) ctx.u32Cols[name].clear();
    }
    for (auto& [name, buf] : ctx.f32ColsGPU) {
        if (buf) ctx.f32Cols[name].clear();
    }

    ctx.activeRows.clear();
    ctx.activeRowsGPU = nullptr;
    ctx.activeRowsCountGPU = 0;
}

bool handleExecFilterNode(const IRFilter& filter, QueryExecutionContext& qctx) {
    auto& currentCtx = qctx.currentCtx;
    auto& tableContexts = qctx.tableContexts;
    auto& tableResult = qctx.tableResult;
    auto& result = qctx.result;
    bool debug = qctx.debug;
    if (debug) {
        LOG_INFO("Exec", "Filter: BEFORE filter, currentCtx.stringCols:\n");
        for (const auto& [n, v] : currentCtx.stringCols) {
            LOG_INFO("Exec", n << " size=" << v.size());
        }
    }
    if (!GpuExecutor::executeFilter(filter, currentCtx)) {
        result.error = "Filter execution failed";
        return false;
    }
    if (debug) {
        LOG_INFO("Exec", "Filter: AFTER filter, currentCtx.stringCols:\n");
        for (const auto& [n, v] : currentCtx.stringCols) {
            LOG_INFO("Exec", n << " size=" << v.size());
        }
    }
    // Update tableContexts with filtered data for joins to use
    if (!currentCtx.currentTable.empty()) {
        tableContexts[currentCtx.currentTable] = currentCtx;
    }

    compactTableResultAfterFilter(currentCtx, tableResult, debug);
    compactContextAfterFilter(currentCtx);

    if (debug) {
        LOG_INFO("Exec", "Filter: " << currentCtx.rowCount << " rows after\n");
    }
    return true;
}

bool handleExecGroupByNode(const IRGroupBy& groupBy, QueryExecutionContext& qctx) {
    auto& currentCtx = qctx.currentCtx;
    auto& tableResult = qctx.tableResult;
    auto& joinedTables = qctx.joinState.joinedTables;
    auto& hasPipeline = qctx.joinState.hasPipeline;
    auto& result = qctx.result;
    bool debug = qctx.debug;
    if (!GpuExecutor::executeGroupBy(groupBy, currentCtx, tableResult)) {
        result.error = "GroupBy execution failed";
        return false;
    }

    LOG_DEBUG("Exec", "DEBUG: GroupBy returned, clearing old context\n");

    // If GroupBy produces multiple rows, this is NOT a scalar result
    if (tableResult.rowCount > 1) {
        result.isScalarAggregate = false;
    }

    // Clear old columns and update with GroupBy output
    currentCtx.rowCount = tableResult.rowCount;
    currentCtx.activeRows.clear();

    LOG_DEBUG("Exec", "DEBUG: Clearing activeRowsGPU\n");
    // Release old GPU buffers with dedup (multiple map keys may alias same buffer)
    {
        std::unordered_set<MTL::Buffer*> released;
        if (currentCtx.activeRowsGPU)
            released.insert(currentCtx.activeRowsGPU);  // track for dedup, RAII releases
        currentCtx.activeRowsGPU = nullptr;
        currentCtx.activeRowsCountGPU = 0;

        LOG_DEBUG("Exec", "DEBUG: Clearing u32ColsGPU\n");
        // u32ColsGPU uses GpuBuffer RAII — clearing triggers destructors.
        for (auto& [_, buf] : currentCtx.u32ColsGPU) {
            if (buf) released.insert(buf.get());  // track for f32 dedup
        }
        currentCtx.u32ColsGPU.clear();

        LOG_DEBUG("Exec", "DEBUG: Clearing f32ColsGPU\n");
        // f32ColsGPU uses GpuBuffer RAII — clearing triggers destructors.
        currentCtx.f32ColsGPU.clear();
    }

    currentCtx.u32Cols.clear();
    currentCtx.f32Cols.clear();
    currentCtx.stringCols.clear();
    currentCtx.dictCols.clear();
    currentCtx.flatStringCols.clear();
    currentCtx.currentTable.clear();

    // Reset joinedTables for SEMI join decorrelation pattern
    joinedTables.clear();
    joinedTables.insert("__GROUPED__");

    // Build set of f32 column names to detect float keys restored from u32
    std::set<std::string> f32NameSet;
    for (const auto& fn : tableResult.f32Names) f32NameSet.insert(fn);

    for (size_t i = 0; i < tableResult.u32Cols.size(); ++i) {
        if (i < tableResult.u32Names.size()) {
            const std::string& name = tableResult.u32Names[i];
            // Skip named registration if this column was restored to f32
            // (the u32 version contains raw IEEE 754 bits, not the actual value)
            bool restoredToF32 = f32NameSet.count(name) > 0;
            if (!restoredToF32) {
                currentCtx.u32Cols[name] = tableResult.u32Cols[i];
            }
            // Register positional key only if not restored to f32
            if (!restoredToF32) {
                std::string posKey = "#" + std::to_string(i);
                currentCtx.u32Cols[posKey] = tableResult.u32Cols[i];
            }
            // Re-register columns under their aliases (for CTE support)
            if (!restoredToF32) {
                for (const auto& [alias, canonical] : currentCtx.columnAliases) {
                    if (canonical == name) {
                        currentCtx.u32Cols[alias] = tableResult.u32Cols[i];
                        LOG_DEBUG("Exec", "GroupBy: re-registering alias " << alias << " -> " << canonical);
                    }
                }
            }
        }
    }
    // Recount non-skipped u32 columns for f32 positional offset
    size_t u32RegisteredCount = 0;
    for (size_t i = 0; i < tableResult.u32Cols.size(); ++i) {
        if (i < tableResult.u32Names.size() && !f32NameSet.count(tableResult.u32Names[i]))
            u32RegisteredCount++;
    }
    for (size_t i = 0; i < tableResult.f32Cols.size(); ++i) {
        if (i < tableResult.f32Names.size()) {
            currentCtx.f32Cols[tableResult.f32Names[i]] = tableResult.f32Cols[i];
            // Also register under positional name for #N references (offset by registered u32 count)
            std::string posKey = "#" + std::to_string(i + u32RegisteredCount);
            currentCtx.f32Cols[posKey] = tableResult.f32Cols[i];
            // Re-register columns under their aliases (for CTE support)
            for (const auto& [alias, canonical] : currentCtx.columnAliases) {
                if (canonical == tableResult.f32Names[i]) {
                    currentCtx.f32Cols[alias] = tableResult.f32Cols[i];
                    LOG_DEBUG("Exec", "GroupBy: re-registering f32 alias " << alias << " -> " << canonical);
                }
            }
        }
    }

    // Populate stringCols from GroupBy result and build dictCols
    for (size_t i = 0; i < tableResult.stringCols.size(); ++i) {
        if (i < tableResult.stringNames.size()) {
            const std::string& sname = tableResult.stringNames[i];
            currentCtx.stringCols[sname] = tableResult.stringCols[i];
            // Build dictionary encoding for downstream operators
            buildDictCol(currentCtx, sname);
            LOG_DEBUG("Exec", "GroupBy: setting stringCol+dictCol " << sname << " with " << tableResult.stringCols[i].size() << " rows\n");
        }
    }

    // Use GPU buffers from GroupBy output directly (avoid CPU→GPU re-upload).
    // For each unique column in TableResult, transfer the pre-created GPU buffer;
    // for positional keys and aliases, share the same buffer with retain().
    LOG_DEBUG("Exec", "Transferring GroupBy GPU buffers directly (zero-copy)\n");

    for (size_t i = 0; i < tableResult.u32Cols.size(); ++i) {
        if (i >= tableResult.u32Names.size()) continue;
        const std::string& name = tableResult.u32Names[i];
        bool restoredToF32 = f32NameSet.count(name) > 0;
        if (restoredToF32) continue;

        // Transfer GPU buffer from TableResult (or create if missing)
        GpuBuffer buf;
        if (i < tableResult.u32ColsGPU.size()) buf = std::move(tableResult.u32ColsGPU[i]);
        if (!buf && !tableResult.u32Cols[i].empty()) {
            buf = GpuOps::createBuffer(tableResult.u32Cols[i].data(),
                                       tableResult.u32Cols[i].size() * sizeof(uint32_t));
        }
        if (buf) {
            currentCtx.u32ColsGPU[name] = buf;  // GpuBuffer takes ownership
            // Positional key shares the same buffer (GpuBuffer copy retains)
            std::string posKey = "#" + std::to_string(i);
            currentCtx.u32ColsGPU[posKey] = currentCtx.u32ColsGPU[name];
            // Aliases share the same buffer (GpuBuffer copy retains)
            for (const auto& [alias, canonical] : currentCtx.columnAliases) {
                if (canonical == name) {
                    currentCtx.u32ColsGPU[alias] = currentCtx.u32ColsGPU[name];
                }
            }
        }
    }
    for (size_t i = 0; i < tableResult.f32Cols.size(); ++i) {
        if (i >= tableResult.f32Names.size()) continue;
        const std::string& name = tableResult.f32Names[i];

        // Transfer GPU buffer from TableResult (or create if missing)
        GpuBuffer buf;
        if (i < tableResult.f32ColsGPU.size()) buf = std::move(tableResult.f32ColsGPU[i]);
        if (!buf && !tableResult.f32Cols[i].empty()) {
            buf = GpuOps::createBuffer(tableResult.f32Cols[i].data(),
                                       tableResult.f32Cols[i].size() * sizeof(float));
        }
        if (buf) {
            currentCtx.f32ColsGPU[name] = buf;
            // Positional key shares the same buffer
            std::string posKey = "#" + std::to_string(i + u32RegisteredCount);
            currentCtx.f32ColsGPU[posKey] = currentCtx.f32ColsGPU[name];  // GpuBuffer copy auto-retains
            // Aliases share the same buffer
            for (const auto& [alias, canonical] : currentCtx.columnAliases) {
                if (canonical == name) {
                    currentCtx.f32ColsGPU[alias] = currentCtx.f32ColsGPU[name];  // GpuBuffer copy auto-retains
                }
            }
        }
    } 

    if (debug) {

        LOG_INFO("Exec", "GroupBy: " << tableResult.rowCount << " groups\n");
        LOG_INFO("Exec", "GroupBy: ctx updated with u32Cols=");
        if (debug) for (const auto& [k, v] : currentCtx.u32Cols) std::cerr << k << "(" << v.size() << ") ";
        LOG_DEBUG("ENGINE", "f32Cols=");
        if (debug) for (const auto& [k, v] : currentCtx.f32Cols) std::cerr << k << "(" << v.size() << ") ";
        LOG_DEBUG("ENGINE", "\n");
    }

    // Mark pipeline active so a new scan can trigger pipeline save
    hasPipeline = true;

    // Clear stale tableResult to avoid misaligned filter compaction
    tableResult.u32Cols.clear();
    tableResult.u32Names.clear();
    tableResult.u32ColsGPU.clear();
    tableResult.f32Cols.clear();
    tableResult.f32Names.clear();
    tableResult.f32ColsGPU.clear();
    tableResult.stringCols.clear();
    tableResult.stringNames.clear();
    tableResult.order.clear();
    tableResult.clearDeferredStrings();
    tableResult.rowCount = 0;

    return true;
}

bool handleExecAggregateNode(const IRAggregate& agg, QueryExecutionContext& qctx) {
    auto& currentCtx = qctx.currentCtx;
    auto& result = qctx.result;
    bool debug = qctx.debug;
    double value;
    std::string name;
    if (!GpuExecutor::executeAggregate(agg, currentCtx, value, name)) {
        result.error = "Aggregate execution failed";
        return false;
    }
    result.isScalarAggregate = true;
    result.scalarValue = value;
    result.scalarName = name;

    // Mark context as scalar result ONLY if this is the last aggregate in the block
    // This prevents sibling aggregates (e.g. sum(a), count(b)) from confusing the row count
    // (sum(a) sets scalar=true, then count(b) sees true and returns 1 -> WRONG)
    if (agg.isLastAgg) {
        currentCtx.isScalarResult = true;
        currentCtx.rowCount = 1;
        // Clear stale activeRowsGPU so projections use rowCount=1
        currentCtx.activeRowsGPU = nullptr;
        currentCtx.activeRowsCountGPU = 0;
        LOG_DEBUG("Exec", "Aggregate: isLastAgg=true, setting rowCount=1 (scalar result)\n");
    }

    // Store aggregate result in context for later projection reference
    // Multiple aggregates get stored as #0, #1, etc. based on aggIndex
    // But DON'T change rowCount yet - other aggregates may still need original data
    std::string posKey = "#" + std::to_string(agg.aggIndex);
    currentCtx.f32Cols[posKey] = std::vector<float>{static_cast<float>(value)};

    // Create GPU buffer for the scalar result
    GpuBuffer aggBuf = GpuOps::createBuffer(currentCtx.f32Cols[posKey].data(), sizeof(float));
    currentCtx.f32ColsGPU[posKey] = std::move(aggBuf);
    // No extra retain needed — createBuffer returns refcount=1 which covers the posKey map entry

    // Also store by name
    if (!name.empty()) {
        currentCtx.f32Cols[name] = std::vector<float>{static_cast<float>(value)};
        currentCtx.f32ColsGPU[name] = currentCtx.f32ColsGPU[posKey]; // GpuBuffer copy auto-retains
    }
    if (!agg.exprStr.empty() && agg.exprStr != name) {
         currentCtx.f32Cols[agg.exprStr] = std::vector<float>{static_cast<float>(value)};
         currentCtx.f32ColsGPU[agg.exprStr] = currentCtx.f32ColsGPU[posKey]; // GpuBuffer copy auto-retains
    }
    if (debug) {
        LOG_INFO("Exec", "Aggregate " << name << ": " << value  << " (stored as " << posKey << ")\n");
    }
    return true;
}

bool handleExecOrderByNode(const IROrderBy& orderBy, QueryExecutionContext& qctx) {
    auto& currentCtx = qctx.currentCtx;
    auto& tableResult = qctx.tableResult;
    auto& result = qctx.result;
    bool debug = qctx.debug;
    // If tableResult is out of sync with currentCtx (e.g. a Join happened
    // after the last Project), materialize currentCtx into tableResult first.
    if (tableResult.rowCount != currentCtx.rowCount && currentCtx.rowCount > 0) {
        if (debug) {
            LOG_INFO("Exec", "OrderBy: syncing tableResult from currentCtx (" << currentCtx.rowCount << " rows, tableResult had " << tableResult.rowCount << ")\n");
        }
        tableResult.u32Cols.clear();
        tableResult.u32Names.clear();
        tableResult.u32ColsGPU.clear();
        tableResult.f32Cols.clear();
        tableResult.f32Names.clear();
        tableResult.f32ColsGPU.clear();
        tableResult.stringCols.clear();
        tableResult.stringNames.clear();
        tableResult.order.clear();
        tableResult.clearDeferredStrings();

        // Download GPU columns to CPU if needed, and populate GPU buffer vectors
        for (auto& [name, buf] : currentCtx.u32ColsGPU) {
            if (!buf) continue;
            if (name.find("__internal_") != std::string::npos) continue;
            if (name.size() >= 2 && name[0] == '#' && std::isdigit(name[1])) continue;
            uint32_t count = (uint32_t)(buf->length() / sizeof(uint32_t));
            if (count < currentCtx.rowCount) continue;
            // Ensure CPU vector exists (download from GPU via ->contents())
            if (currentCtx.u32Cols.find(name) == currentCtx.u32Cols.end() ||
                currentCtx.u32Cols.at(name).empty()) {
                std::vector<uint32_t> v(currentCtx.rowCount);
                std::memcpy(v.data(), buf->contents(), currentCtx.rowCount * sizeof(uint32_t));
                currentCtx.u32Cols[name] = std::move(v);
            }
            tableResult.u32Names.push_back(name);
            tableResult.u32Cols.push_back(currentCtx.u32Cols[name]);
            tableResult.u32ColsGPU.push_back(buf);
        }
        for (auto& [name, buf] : currentCtx.f32ColsGPU) {
            if (!buf) continue;
            if (name.size() >= 2 && name[0] == '#' && std::isdigit(name[1])) continue;
            uint32_t count = (uint32_t)(buf->length() / sizeof(float));
            if (count < currentCtx.rowCount) continue;
            if (currentCtx.f32Cols.find(name) == currentCtx.f32Cols.end() ||
                currentCtx.f32Cols.at(name).empty()) {
                std::vector<float> v(currentCtx.rowCount);
                std::memcpy(v.data(), buf->contents(), currentCtx.rowCount * sizeof(float));
                currentCtx.f32Cols[name] = std::move(v);
            }
            tableResult.f32Names.push_back(name);
            tableResult.f32Cols.push_back(currentCtx.f32Cols[name]);
            tableResult.f32ColsGPU.push_back(buf);
        }
        // Also pick up CPU-only columns that have no GPU buffer
        for (const auto& [name, vec] : currentCtx.u32Cols) {
            if (vec.empty()) continue;
            if (name.find("__internal_") != std::string::npos) continue;
            if (name.size() >= 2 && name[0] == '#' && std::isdigit(name[1])) continue;
            if (currentCtx.u32ColsGPU.count(name)) continue;  // already handled above
            tableResult.u32Names.push_back(name);
            tableResult.u32Cols.push_back(vec);
            tableResult.u32ColsGPU.emplace_back();  // null GpuBuffer
        }
        for (const auto& [name, vec] : currentCtx.f32Cols) {
            if (vec.empty()) continue;
            if (name.size() >= 2 && name[0] == '#' && std::isdigit(name[1])) continue;
            if (currentCtx.f32ColsGPU.count(name)) continue;  // already handled above
            tableResult.f32Names.push_back(name);
            tableResult.f32Cols.push_back(vec);
            tableResult.f32ColsGPU.emplace_back();  // null GpuBuffer
        }
        for (const auto& [name, vec] : currentCtx.stringCols) {
            if (!vec.empty()) {
                tableResult.stringNames.push_back(name);
                tableResult.stringCols.push_back(vec);
            }
        }
        // Pass through dict/flat string columns without materializing (deferred)
        for (const auto& [name, dict] : currentCtx.dictCols) {
            if (dict.valid() && !currentCtx.stringCols.count(name)) {
                tableResult.stringNames.push_back(name);
                tableResult.stringCols.push_back({});  // empty placeholder (deferred)
                tableResult.dictStringResults[name] = dict;
            }
        }
        for (const auto& [name, flat] : currentCtx.flatStringCols) {
            if (!flat.chars) continue;
            if (currentCtx.dictCols.count(name)) continue;  // handled by dict loop
            if (currentCtx.stringCols.count(name)) continue; // handled by stringCols loop
            bool found = false;
            for (const auto& sn : tableResult.stringNames) if (sn == name) { found = true; break; }
            if (found) continue;
            tableResult.stringNames.push_back(name);
            tableResult.stringCols.push_back({});  // empty placeholder (deferred)
            tableResult.flatStringResults[name] = flat;
        }
        tableResult.rowCount = currentCtx.rowCount;
    }
    if (!GpuExecutor::executeOrderBy(orderBy, tableResult, currentCtx.dictCols, currentCtx.flatStringCols)) {
        result.error = "OrderBy execution failed";
        return false;
    }
    // Sync ctx with sorted tableResult so that a subsequent Project
    // does not re-read unsorted data from the old context.
    for (size_t i = 0; i < tableResult.u32Cols.size(); ++i) {
        if (i < tableResult.u32Names.size()) {
            const auto& name = tableResult.u32Names[i];
            if (!tableResult.u32Cols[i].empty()) {
                currentCtx.u32Cols[name] = tableResult.u32Cols[i];
            } else if (i < tableResult.u32ColsGPU.size() && tableResult.u32ColsGPU[i]) {
                // GPU-only after sort gather — sync from GPU buffer
                uint32_t rc = tableResult.rowCount;
                currentCtx.u32Cols[name].resize(rc);
                std::memcpy(currentCtx.u32Cols[name].data(),
                            tableResult.u32ColsGPU[i]->contents(), rc * sizeof(uint32_t));
                // Also sync the tableResult CPU vec for output
                tableResult.u32Cols[i] = currentCtx.u32Cols[name];
            }
            if (i < tableResult.u32ColsGPU.size() && tableResult.u32ColsGPU[i]) {
                currentCtx.u32ColsGPU[name].reset(tableResult.u32ColsGPU[i].get());
                tableResult.u32ColsGPU[i]->retain();
            }
        }
    }
    for (size_t i = 0; i < tableResult.f32Cols.size(); ++i) {
        if (i < tableResult.f32Names.size()) {
            const auto& name = tableResult.f32Names[i];
            if (!tableResult.f32Cols[i].empty()) {
                currentCtx.f32Cols[name] = tableResult.f32Cols[i];
            } else if (i < tableResult.f32ColsGPU.size() && tableResult.f32ColsGPU[i]) {
                // GPU-only after sort gather — sync from GPU buffer
                uint32_t rc = tableResult.rowCount;
                currentCtx.f32Cols[name].resize(rc);
                std::memcpy(currentCtx.f32Cols[name].data(),
                            tableResult.f32ColsGPU[i]->contents(), rc * sizeof(float));
                // Also sync the tableResult CPU vec for output
                tableResult.f32Cols[i] = currentCtx.f32Cols[name];
            }
            if (i < tableResult.f32ColsGPU.size() && tableResult.f32ColsGPU[i]) {
                currentCtx.f32ColsGPU[name].reset(tableResult.f32ColsGPU[i].get());
                tableResult.f32ColsGPU[i]->retain();
            }
        }
    }
    for (size_t i = 0; i < tableResult.stringCols.size(); ++i) {
        if (i < tableResult.stringNames.size()) {
            const std::string& sname = tableResult.stringNames[i];
            if (!tableResult.stringCols[i].empty()) {
                // Materialized — use as before
                currentCtx.stringCols[sname] = tableResult.stringCols[i];
                buildDictCol(currentCtx, sname);
                currentCtx.flatStringCols.erase(sname);
            } else {
                // Deferred — copy flat/dict from tableResult to ctx (keep in tableResult for Limit)
                auto fit = tableResult.flatStringResults.find(sname);
                if (fit != tableResult.flatStringResults.end() && fit->second.chars) {
                    currentCtx.flatStringCols[sname] = fit->second;  // copy, not move
                }
                auto dit = tableResult.dictStringResults.find(sname);
                if (dit != tableResult.dictStringResults.end() && dit->second.valid()) {
                    currentCtx.dictCols[sname] = dit->second;  // copy, not move
                    // If dict was reordered but flat was erased (by OrderBy reorder),
                    // erase the stale flat from ctx so downstream reads the reordered dict.
                    if (fit == tableResult.flatStringResults.end() || !fit->second.chars) {
                        currentCtx.flatStringCols.erase(sname);
                    }
                }
                // Clear any stale materialized strings
                currentCtx.stringCols.erase(sname);
            }
        }
    }
    if (debug) {
        LOG_INFO("Exec", "OrderBy applied\n");
    }
    return true;
}

bool handleExecProjectNode(const IRProject& project, QueryExecutionContext& qctx) {
    auto& currentCtx = qctx.currentCtx;
    auto& tableResult = qctx.tableResult;
    auto& tableContexts = qctx.tableContexts;
    auto& result = qctx.result;
    bool debug = qctx.debug;
    if (!GpuExecutor::executeProject(project, currentCtx, tableResult, &tableContexts)) {
        result.error = "Project execution failed";
        return false;
    }

    // If this is a projection after aggregates (e.g., 100.0 * sum(...) / sum(...)),
    // the projection output is the final result, not the raw aggregate value
    // Note: We relax the rowCount==1 check because vector broadcasting might have produced N rows
    if (result.isScalarAggregate && !tableResult.f32Cols.empty()) {
        // Update the scalar result with the projection output
        result.scalarValue = tableResult.f32Cols[0][0];
        result.scalarName = tableResult.f32Names.empty() ? "result" : tableResult.f32Names[0];
        if (debug) {
            LOG_INFO("Exec", "Project after Aggregate: updated scalar to "  << result.scalarValue << " (" << result.scalarName << ")\n");
        }
    }

    // Update tableContexts if we're still working with a single table
    if (!currentCtx.currentTable.empty()) {
        tableContexts[currentCtx.currentTable] = currentCtx;
        if (debug) {
            LOG_INFO("Exec", "Project: updated tableContexts[" << currentCtx.currentTable << "] with "  << currentCtx.rowCount << " rows, u32cols=");
            for (const auto& [k,v] : currentCtx.u32Cols) std::cerr << k << " ";
            LOG_DEBUG("ENGINE", "f32cols=");
            if (debug) for (const auto& [k,v] : currentCtx.f32Cols) std::cerr << k << " ";
            LOG_DEBUG("ENGINE", "\n");
        }
    }
    return true;
}

} // namespace engine
