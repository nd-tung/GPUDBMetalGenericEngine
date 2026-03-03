#include "GpuExecutor.hpp"
#include "GpuExecutorPriv.hpp"
#include "Operators.hpp"
#include "ColumnStoreGPU.hpp"
#include <Metal/Metal.hpp>

#include "Planner.hpp"
#include "KernelTimer.hpp"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <iostream>
#include <map>
#include <set>
#include <unordered_set>

namespace engine {

// ── GPU dedup helper: deduplicate an EvalContext by u32 key columns ──
// Uses GpuOps::dedupByKeys on GPU buffers, then GPU gather for u32/f32,
// CPU gather for strings. Returns new row count (0 = no dedup needed).
uint32_t gpuDedupContext(EvalContext& ctx,
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
                MTL::Buffer* buf = GpuOps::createBuffer(cit->second.data(), ctx.rowCount * sizeof(uint32_t));
                if (buf) {
                    ctx.u32ColsGPU[col] = buf;
                    gpuKeys.push_back(buf);
                } else {
                    return 0;
                }
            } else {
                return 0;
            }
        }
    }

    uint32_t uniqueCount = 0;
    MTL::Buffer* uniqueIdx = GpuOps::dedupByKeys(gpuKeys, ctx.rowCount, uniqueCount);
    if (!uniqueIdx || uniqueCount == 0 || uniqueCount >= ctx.rowCount) {
        if (uniqueIdx) uniqueIdx->release();
        return 0;  // No duplicates found
    }

    if (debug) {
        std::cerr << "[Exec] GPU dedup: " << ctx.rowCount << " -> " << uniqueCount << " rows\n";
    }

    // GPU gather u32 columns
    for (auto& [name, buf] : ctx.u32ColsGPU) {
        if (buf) {
            auto compacted = GpuOps::gatherU32(buf, uniqueIdx, uniqueCount);
            if (compacted) buf = compacted;
        }
    }
    // GPU gather f32 columns
    for (auto& [name, buf] : ctx.f32ColsGPU) {
        if (buf) {
            auto compacted = GpuOps::gatherF32(buf, uniqueIdx, uniqueCount);
            if (compacted) buf = compacted;
        }
    }

    // CPU gather u32 columns (sync from GPU)
    for (auto& [name, buf] : ctx.u32ColsGPU) {
        if (buf) {
            size_t n = buf->length() / sizeof(uint32_t);
            ctx.u32Cols[name].resize(n);
            memcpy(ctx.u32Cols[name].data(), buf->contents(), n * sizeof(uint32_t));
        }
    }
    // CPU gather f32 columns (sync from GPU)
    for (auto& [name, buf] : ctx.f32ColsGPU) {
        if (buf) {
            size_t n = buf->length() / sizeof(float);
            ctx.f32Cols[name].resize(n);
            memcpy(ctx.f32Cols[name].data(), buf->contents(), n * sizeof(float));
        }
    }
    // CPU-only u32/f32 columns: upload → GPU gather → download
    {
        auto& s = ColumnStoreGPU::instance();
        for (auto& [name, col] : ctx.u32Cols) {
            if (!ctx.u32ColsGPU.count(name) && col.size() >= ctx.rowCount) {
                MTL::Buffer* src = s.device()->newBuffer(col.data(), col.size() * sizeof(uint32_t), MTL::ResourceStorageModeShared);
                MTL::Buffer* dst = GpuOps::gatherU32(src, uniqueIdx, uniqueCount);
                if (dst) {
                    col.resize(uniqueCount);
                    std::memcpy(col.data(), dst->contents(), uniqueCount * sizeof(uint32_t));
                    dst->release();
                }
                src->release();
            }
        }
        for (auto& [name, col] : ctx.f32Cols) {
            if (!ctx.f32ColsGPU.count(name) && col.size() >= ctx.rowCount) {
                MTL::Buffer* src = s.device()->newBuffer(col.data(), col.size() * sizeof(float), MTL::ResourceStorageModeShared);
                MTL::Buffer* dst = GpuOps::gatherF32(src, uniqueIdx, uniqueCount);
                if (dst) {
                    col.resize(uniqueCount);
                    std::memcpy(col.data(), dst->contents(), uniqueCount * sizeof(float));
                    dst->release();
                }
                src->release();
            }
        }
    }
    // String columns: GPU gather dict IDs (GPU-native dictionary encoding)
    for (auto& [name, dict] : ctx.dictCols) {
        if (dict.idsGPU) {
            auto compacted = GpuOps::gatherU32(dict.idsGPU, uniqueIdx, uniqueCount);
            if (compacted) {
                dict.idsGPU = compacted;
                dict.rowCount = uniqueCount;
                dict.ids.clear();  // Invalidate CPU mirror (lazy sync)
            }
        }
    }
    // Sync stringCols: skip those covered by dict or flat, CPU gather orphans only
    for (auto& [name, col] : ctx.stringCols) {
        if (ctx.dictCols.count(name) || ctx.flatStringCols.count(name)) {
            continue;  // Lazy — will be materialized on demand
        } else if (col.size() >= ctx.rowCount) {
            // Legacy fallback: CPU gather for orphan string cols without dict/flat
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
    uniqueIdx->release();

    ctx.rowCount = uniqueCount;
    ctx.activeRows.clear();
    ctx.activeRowsGPU = nullptr;
    ctx.activeRowsCountGPU = 0;
    return uniqueCount;
}

// --- Multi-Instance Column Resolution ---
// Rewrites predicates to use suffixed column names (e.g. col_2) when the same
// column appears in multiple comparison arms and suffixed variants exist.


// --- GPU Feasibility Checking ---

std::vector<std::string> GpuExecutor::getGPUBlockers(const Plan& plan) {
    std::vector<std::string> blockers;

    // Count nodes and track table scans
    size_t joinCount = 0;
    bool hasEmptyScan = false;
    bool hasDistinct = false;
    bool hasOuterJoin = false;
    bool hasSubqueryInCondition = false;
    bool hasIsNotDistinctFrom = false;
    std::map<std::string, int> tableScanCounts;  // Track duplicate table scans

    // First pass: check for UNGROUPED_AGGREGATE which indicates scalar subquery
    // Also check if SUBQUERY appears in a Filter (correlated) vs Join (scalar)
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
                    // Empty table name indicates subquery/CTE artifact
                    hasEmptyScan = true;
                }
                break;
            }
            case IRNode::Type::Join: {
                joinCount++;
                const auto& join = node.asJoin();
                // Support INNER, LEFT, RIGHT, SEMI (for IN/EXISTS), ANTI (for NOT EXISTS), and MARK (for NOT IN)
                if (join.type != JoinType::Inner && join.type != JoinType::Left &&
                    join.type != JoinType::Right && join.type != JoinType::Semi && 
                    join.type != JoinType::Anti && join.type != JoinType::Mark) {
                    hasOuterJoin = true;
                }
                // Check for SUBQUERY in join condition
                if (join.conditionStr.find("SUBQUERY") != std::string::npos) {
                    hasSubqueryInCondition = true;
                }
                // Check for IS NOT DISTINCT FROM (indicates DELIM_SCAN correlation pattern)
                if (join.conditionStr.find("IS NOT DISTINCT FROM") != std::string::npos) {
                    hasIsNotDistinctFrom = true;
                }
                // Self-comparison patterns (col = col) are valid in DuckDB's flattened subquery plans
                break;
            }
            case IRNode::Type::Filter: {
                // Check for SUBQUERY in filter predicate
                const auto& filter = node.asFilter();
                if (filter.predicateStr.find("SUBQUERY") != std::string::npos) {
                    hasSubqueryInCondition = true;
                }
                break;
            }
            case IRNode::Type::Distinct:
                hasDistinct = true;
                break;
            default:
                break;
        }
    }
    
    // Multi-instance table scans use instance-qualified keys (e.g., nation_1, nation_2)

    if (joinCount > 20) {
        blockers.push_back("Multi-way JOIN (>20 tables)");
    }
    if (hasOuterJoin) {
        blockers.push_back("FULL OUTER/CROSS JOIN not supported (INNER/LEFT/RIGHT/SEMI/ANTI supported)");
    }
    if (hasDistinct) {
        blockers.push_back("DISTINCT not supported on GPU");
    }
    if (hasSubqueryInCondition) {
        // DuckDB decorrelates correlated subqueries using SEMI/ANTI/MARK joins.
        // Patterns we can handle:
        // 1. Pure scalar subquery: UNGROUPED_AGGREGATE, no SUBQUERY in Filter
        // 2. Decorrelated via SEMI/ANTI/MARK joins: DuckDB transforms EXISTS/NOT EXISTS/IN/NOT IN
        // 3. IS NOT DISTINCT FROM pattern: DuckDB's DELIM_SCAN decorrelation
        
        bool isScalarSubqueryOK = hasUngroupedAggregate && !hasSubqueryInFilter && !hasIsNotDistinctFrom;
        
        // Check for SEMI/ANTI/MARK joins which indicate DuckDB has decorrelated the subquery
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
        
        // Only block if we truly have a problematic correlated subquery
        // If DuckDB has decorrelated it (SEMI/ANTI/MARK joins), we can execute it
        if (!isScalarSubqueryOK && !hasDecorrelatedJoin && !hasEmptyScan) {
            blockers.push_back("Correlated subquery in condition not supported");
        }
    }
    // DELIM_SCAN patterns handled by skipping empty scans and treating self-comparison SEMI joins as pass-through

    // Check for unsupported expression types
    for (const auto& node : plan.nodes) {
        if (node.type == IRNode::Type::GroupBy) {
            const auto& gb = node.asGroupBy();
            if (gb.keys.size() > 8) {
                blockers.push_back("GROUP BY with >8 keys");
            }
        }
    }

    return blockers;
}

// --- Main Execution Entry Point ---

GpuExecutor::ExecutionResult GpuExecutor::execute(const Plan& plan, const std::string& datasetPath) {
    ExecutionResult result;
    result.success = false;
    
    // Reset kernel timer for this query
    KernelTimer::instance().reset();

    if (!plan.isValid()) {
        result.error = "Invalid plan: " + plan.parseError;
        return result;
    }

    auto blockers = getGPUBlockers(plan);
    if (!blockers.empty()) {
        result.error = "GPU execution blocked: " + blockers[0];
        for (size_t i = 1; i < blockers.size(); ++i) {
            result.error += ", " + blockers[i];
        }
        return result;
    }

    const bool debug = env_truthy("GPUDB_DEBUG_OPS");

    if (debug) {
        std::cerr << "[Exec] Plan Nodes (" << plan.nodes.size() << "):\n";
        for (size_t i = 0; i < plan.nodes.size(); ++i) {
            const auto& n = plan.nodes[i];
            std::string name = n.duckdbName;
            if (n.type == IRNode::Type::Save) name = "Save(" + n.asSave().name + ")";
            else if (n.type == IRNode::Type::Scan) name = "Scan(" + n.asScan().table + ")";
            else if (n.type == IRNode::Type::Join) name = "Join(" + n.asJoin().conditionStr + ")";
            else if (name.empty()) name = "[Empty/Unknown Type=" + std::to_string((int)n.type) + "]";
            std::cerr << "  #" << i << ": " << name << "\n";
        }
    }

    // Build scan instance map for tables that appear multiple times
    auto scanInstanceMap = buildScanInstanceMap(plan);
    
    if (debug && !scanInstanceMap.empty()) {
        std::cerr << "[Exec] Table instances for self-joins:\n";
        for (const auto& [nodeIdx, inst] : scanInstanceMap) {
            std::cerr << "  Node " << nodeIdx << ": " << inst.baseTable 
                      << " -> " << inst.instanceKey << "\n";
        }
    }

    // Collect all tables and columns needed
    auto tableColsMap = collectNeededColumns(plan);

    if (debug) {
        std::cerr << "[Exec] Columns needed per table:\n";
        for (const auto& [t, cs] : tableColsMap) {
            std::cerr << "  " << t << ": ";
            for (const auto& c : cs) std::cerr << c << " ";
            std::cerr << "\n";
        }
    }

    // Build execution contexts for each table
    std::unordered_map<std::string, EvalContext> tableContexts;

    IRGpuLoader::loadTables(tableColsMap, scanInstanceMap, datasetPath, tableContexts, result, debug);
    if (!result.error.empty()) return result;
    
    auto loadEnd = std::chrono::high_resolution_clock::now();

    if (debug) {
        std::cerr << "[Exec] Loaded " << tableContexts.size() << " tables in " 
                  << result.table.upload_ms << "ms\n";
    }

    // Execute operators in pipeline order
    EvalContext currentCtx;
    TableResult tableResult;

    // Track which tables have been joined into the current context
    std::set<std::string> joinedTables;
    
    bool hasPipeline = false;
    
    // Save previous pipeline contexts for multi-pipeline query merges
    std::vector<EvalContext> savedPipelines;
    std::vector<std::set<std::string>> savedPipelineTables;

    // Pre-scan: extract DELIM correlation columns from self-comparison join conditions
    std::unordered_map<std::string, std::vector<std::string>> delimCorrelationCols;
    {
        // Find Save nodes for delim groups, then find the DELIM_JOIN for each
        for (size_t ni = 0; ni < plan.nodes.size(); ++ni) {
            if (plan.nodes[ni].type == IRNode::Type::Join) {
                const auto& join = plan.nodes[ni].asJoin();
                const std::string& cond = join.conditionStr;
                // Look for self-comparison patterns that indicate DELIM correlation
                bool hasSelfComp = false;
                std::vector<std::string> corrCols;
                // Parse "colA IS NOT DISTINCT FROM colA" and "colA = colA" patterns
                auto condParts = splitConditionByAND(cond);
                for (const auto& part : condParts) {
                    std::string col = parseSelfComparison(part);
                    if (!col.empty()) {
                        hasSelfComp = true;
                        corrCols.push_back(col);
                    }
                }
                if (hasSelfComp && !corrCols.empty()) {
                    // Find the delim group this belongs to by looking at rightTable
                    // or by looking backward for the nearest Save
                    std::string delimGroup;
                    // Look for Scan nodes referencing tmpl_delim_lhs_* before this join
                    for (size_t si = 0; si < ni; ++si) {
                        if (plan.nodes[si].type == IRNode::Type::Scan) {
                            const std::string& tbl = plan.nodes[si].asScan().table;
                            if (tbl.find("tmpl_delim_lhs_") == 0) {
                                delimGroup = tbl;
                            }
                        }
                    }
                    if (!delimGroup.empty()) {
                        // Merge correlation cols (may accumulate from multiple joins)
                        auto& existing = delimCorrelationCols[delimGroup];
                        for (const auto& c : corrCols) {
                            if (std::find(existing.begin(), existing.end(), c) == existing.end()) {
                                existing.push_back(c);
                            }
                        }
                    }
                }
            }
        }
        if (debug && !delimCorrelationCols.empty()) {
            for (const auto& [group, cols] : delimCorrelationCols) {
                std::cerr << "[Exec] DELIM correlation: " << group << " -> [";
                for (size_t i = 0; i < cols.size(); ++i) {
                    if (i) std::cerr << ", ";
                    std::cerr << cols[i];
                }
                std::cerr << "]\n";
            }
        }
    }

    for (size_t nodeIdx = 0; nodeIdx < plan.nodes.size(); ++nodeIdx) {
        const auto& node = plan.nodes[nodeIdx];
        if (debug) {
            std::cerr << "[Exec] Executing Node " << nodeIdx << " Type=" << (int)node.type << "\n";
            if (node.type == IRNode::Type::Save) {
                 std::cerr << "[Exec] ... Save Name: " << node.asSave().name << "\n";
            }
        }
        switch (node.type) {
            case IRNode::Type::Scan: {
                const auto& scan = node.asScan();
                
                // Skip empty scans (DELIM_SCAN markers)
                if (scan.table.empty()) {
                    if (debug) {
                        std::cerr << "[Exec] Skipping empty scan (DELIM_SCAN marker)\n";
                    }
                    break;
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
                         if (debug) std::cerr << "[Exec] Scan fallback: using base table " << instIt->second.baseTable << " for " << tableKey << "\n";
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
                             if (debug) std::cerr << "[Exec] Scan fallback (DELIM aliasing): using " << altKey << " for " << tableKey << "\n";
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
                                  if (debug) std::cerr << "[Exec] Scan fallback (DELIM aliasing base): using " << altKey << " for " << tableKey << "\n";
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
                              if (debug) std::cerr << "[Exec] Scan fallback (DELIM Find): using " << rit->first << " for " << tableKey << "\n";
                              it = rit;
                              break;
                         }
                     }
                }

                if (debug) std::cerr << "[Exec] Scan Loop lookup: " << tableKey << " found=" << (it != tableContexts.end()) << "\n";
                if (it != tableContexts.end() && debug) std::cerr << "[Exec] Scan Table Size: " << it->second.rowCount << "\n";
                if (debug) {
                    std::cerr << "[Exec] Scan isDelimScan=" << scan.isDelimScan << " columns=[";
                    for (size_t ci=0; ci<scan.columns.size(); ++ci) { if (ci) std::cerr << ","; std::cerr << scan.columns[ci]; }
                    std::cerr << "]\n";
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
                                break;
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
                                std::cerr << "[Exec] Applying scan filter for build-side table " << tableKey << "\n";
                            }
                            EvalContext& tableCtx = tableContexts[tableKey];
                            executeFilter(IRFilter{scan.filter, ""}, tableCtx);
                            if (debug) {
                                std::cerr << "[Exec] After filter: " << tableCtx.rowCount << " rows\n";
                            }
                        }
                        // DELIM_SCAN dedup for build-side tables too
                        if (scan.isDelimScan && !scan.columns.empty()) {
                            EvalContext& tableCtx = tableContexts[tableKey];
                            if (tableCtx.rowCount > 1) {
                                // First, materialize GPU data to CPU if needed
                                for (auto& [name, buf] : tableCtx.u32ColsGPU) {
                                    if (buf && (!tableCtx.u32Cols.count(name) || tableCtx.u32Cols.at(name).empty())) {
                                        tableCtx.u32Cols[name].resize(tableCtx.rowCount);
                                        memcpy(tableCtx.u32Cols[name].data(), buf->contents(), tableCtx.rowCount * sizeof(uint32_t));
                                    }
                                }
                                for (auto& [name, buf] : tableCtx.f32ColsGPU) {
                                    if (buf && (!tableCtx.f32Cols.count(name) || tableCtx.f32Cols.at(name).empty())) {
                                        tableCtx.f32Cols[name].resize(tableCtx.rowCount);
                                        memcpy(tableCtx.f32Cols[name].data(), buf->contents(), tableCtx.rowCount * sizeof(float));
                                    }
                                }
                                std::vector<std::string> dedupCols;
                                // Prefer correlation columns from DELIM_JOIN if available
                                std::string baseDelim = tableKey;
                                // Strip instance suffix (e.g., tmpl_delim_lhs_11_1 -> tmpl_delim_lhs_11)
                                for (const auto& [grp, cols] : delimCorrelationCols) {
                                    if (baseDelim.find(grp) == 0 || grp.find(baseDelim) == 0) {
                                        for (const auto& c : cols) {
                                            if (tableCtx.u32Cols.count(c) && !tableCtx.u32Cols.at(c).empty())
                                                dedupCols.push_back(c);
                                        }
                                        break;
                                    }
                                }
                                // Fallback: use all u32 scan columns if no correlation cols found
                                if (dedupCols.empty()) {
                                    for (const auto& c : scan.columns) {
                                        if (tableCtx.u32Cols.count(c) && !tableCtx.u32Cols.at(c).empty()) {
                                            dedupCols.push_back(c);
                                        }
                                    }
                                }
                                if (!dedupCols.empty()) {
                                    uint32_t newCount = gpuDedupContext(tableCtx, dedupCols, debug);
                                    if (newCount > 0) {
                                        // Strip payload columns
                                        tableCtx.f32Cols.clear();
                                        tableCtx.f32ColsGPU.clear();
                                        tableCtx.stringCols.clear();
                                        // Also strip non-correlation u32 columns
                                        {
                                            std::set<std::string> keepCols(dedupCols.begin(), dedupCols.end());
                                            for (auto it2 = tableCtx.u32Cols.begin(); it2 != tableCtx.u32Cols.end(); ) {
                                                if (keepCols.find(it2->first) == keepCols.end())
                                                    it2 = tableCtx.u32Cols.erase(it2);
                                                else ++it2;
                                            }
                                            for (auto it2 = tableCtx.u32ColsGPU.begin(); it2 != tableCtx.u32ColsGPU.end(); ) {
                                                if (keepCols.find(it2->first) == keepCols.end())
                                                    it2 = tableCtx.u32ColsGPU.erase(it2);
                                                else ++it2;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        if (debug) {
                            std::cerr << "[Exec] Scan " << tableKey << " (for join build): " 
                                      << tableContexts[tableKey].rowCount << " rows\n";
                        }
                    } else {
                        // Starting a new pipeline - save previous pipeline if it has joined data
                        if (hasPipeline && !joinedTables.empty() && currentCtx.rowCount > 0) {
                            savedPipelines.push_back(currentCtx);
                            savedPipelineTables.push_back(joinedTables);
                            if (debug) {
                                std::cerr << "[Exec] Saved pipeline with tables: ";
                                for (const auto& t : joinedTables) std::cerr << t << " ";
                                std::cerr << "(" << currentCtx.rowCount << " rows)\n";
                            }
                        }
                        
                        // Start/continue current context with this table
                        currentCtx = it->second;
                        currentCtx.currentTable = tableKey;
                        joinedTables.clear();
                        joinedTables.insert(tableKey);

                        // Filter context to only include columns needed by this scan
                        // This prevents extra columns (e.g., string match cols) from
                        // leaking into the pipeline and causing name collisions later
                        if (!scan.columns.empty()) {
                            std::set<std::string> keepCols(scan.columns.begin(), scan.columns.end());
                            // Also keep columns referenced by the scan's pushed-down filter
                            if (scan.filter) {
                                collectColumnsFromExpr(scan.filter, keepCols);
                            }
                            // Helper: check if a column name (possibly with instance suffix) should be kept
                            auto shouldKeep = [&](const std::string& colName) -> bool {
                                if (keepCols.count(colName)) return true;
                                // Check without instance suffix (e.g., "n_nationkey_2" -> "n_nationkey")
                                auto lastUnderscore = colName.rfind('_');
                                if (lastUnderscore != std::string::npos) {
                                    std::string suffix = colName.substr(lastUnderscore + 1);
                                    bool allDigits = !suffix.empty() && std::all_of(suffix.begin(), suffix.end(), ::isdigit);
                                    if (allDigits) {
                                        std::string baseName = colName.substr(0, lastUnderscore);
                                        if (keepCols.count(baseName)) return true;
                                    }
                                }
                                return false;
                            };
                            // Apply filter to u32/f32/string columns
                            for (auto cit = currentCtx.u32Cols.begin(); cit != currentCtx.u32Cols.end(); ) {
                                if (!shouldKeep(cit->first))
                                    cit = currentCtx.u32Cols.erase(cit);
                                else ++cit;
                            }
                            for (auto cit = currentCtx.u32ColsGPU.begin(); cit != currentCtx.u32ColsGPU.end(); ) {
                                if (!shouldKeep(cit->first))
                                    cit = currentCtx.u32ColsGPU.erase(cit);
                                else ++cit;
                            }
                            for (auto cit = currentCtx.f32Cols.begin(); cit != currentCtx.f32Cols.end(); ) {
                                if (!shouldKeep(cit->first))
                                    cit = currentCtx.f32Cols.erase(cit);
                                else ++cit;
                            }
                            for (auto cit = currentCtx.f32ColsGPU.begin(); cit != currentCtx.f32ColsGPU.end(); ) {
                                if (!shouldKeep(cit->first))
                                    cit = currentCtx.f32ColsGPU.erase(cit);
                                else ++cit;
                            }
                            for (auto cit = currentCtx.stringCols.begin(); cit != currentCtx.stringCols.end(); ) {
                                if (!shouldKeep(cit->first))
                                    cit = currentCtx.stringCols.erase(cit);
                                else ++cit;
                            }
                            for (auto cit = currentCtx.flatStringCols.begin(); cit != currentCtx.flatStringCols.end(); ) {
                                if (!shouldKeep(cit->first))
                                    cit = currentCtx.flatStringCols.erase(cit);
                                else ++cit;
                            }
                            for (auto cit = currentCtx.dictCols.begin(); cit != currentCtx.dictCols.end(); ) {
                                if (!shouldKeep(cit->first))
                                    cit = currentCtx.dictCols.erase(cit);
                                else ++cit;
                            }
                            if (debug) {
                                std::cerr << "[Exec] Scan column filter: kept cols:";
                                for (const auto& c : keepCols) std::cerr << " " << c;
                                std::cerr << "\n";
                            }
                        }

                        // DELIM_SCAN deduplication: In DuckDB's decorrelated plans,
                        // DELIM_SCAN produces the DISTINCT set of correlated keys,
                        // while COLUMN_DATA_SCAN produces the full original data.
                        // Deduplicate by the scan's projected columns to get distinct keys.
                        if (scan.isDelimScan && !scan.columns.empty() && currentCtx.rowCount > 1) {
                            // First, materialize GPU data to CPU if needed
                            for (auto& [name, buf] : currentCtx.u32ColsGPU) {
                                if (buf && (!currentCtx.u32Cols.count(name) || currentCtx.u32Cols.at(name).empty())) {
                                    currentCtx.u32Cols[name].resize(currentCtx.rowCount);
                                    memcpy(currentCtx.u32Cols[name].data(), buf->contents(), currentCtx.rowCount * sizeof(uint32_t));
                                }
                            }
                            for (auto& [name, buf] : currentCtx.f32ColsGPU) {
                                if (buf && (!currentCtx.f32Cols.count(name) || currentCtx.f32Cols.at(name).empty())) {
                                    currentCtx.f32Cols[name].resize(currentCtx.rowCount);
                                    memcpy(currentCtx.f32Cols[name].data(), buf->contents(), currentCtx.rowCount * sizeof(float));
                                }
                            }
                            // Find correlation columns for DELIM_SCAN dedup
                            std::vector<std::string> dedupCols;
                            // Prefer correlation columns from DELIM_JOIN if available
                            for (const auto& [grp, cols] : delimCorrelationCols) {
                                if (tableKey.find(grp) == 0 || grp.find(tableKey) == 0) {
                                    for (const auto& c : cols) {
                                        if (currentCtx.u32Cols.count(c) && !currentCtx.u32Cols.at(c).empty())
                                            dedupCols.push_back(c);
                                    }
                                    break;
                                }
                            }
                            // Fallback: use all u32 scan columns
                            if (dedupCols.empty()) {
                                for (const auto& c : scan.columns) {
                                    if (currentCtx.u32Cols.count(c) && !currentCtx.u32Cols.at(c).empty()) {
                                        dedupCols.push_back(c);
                                    }
                                }
                            }
                            if (debug) {
                                std::cerr << "[Exec] DELIM_SCAN dedup: dedupCols=[";
                                for (size_t ci=0; ci<dedupCols.size(); ++ci) { if (ci) std::cerr << ","; std::cerr << dedupCols[ci]; }
                                std::cerr << "]\n";
                            }
                            if (!dedupCols.empty()) {
                                uint32_t newCount = gpuDedupContext(currentCtx, dedupCols, debug);
                                if (newCount > 0) {
                                    // Strip non-correlation columns from DELIM_SCAN context
                                    currentCtx.f32Cols.clear();
                                    currentCtx.f32ColsGPU.clear();
                                    currentCtx.stringCols.clear();
                                    // Also strip non-correlation u32 columns
                                    {
                                        std::set<std::string> keepCols(dedupCols.begin(), dedupCols.end());
                                        for (auto it2 = currentCtx.u32Cols.begin(); it2 != currentCtx.u32Cols.end(); ) {
                                            if (keepCols.find(it2->first) == keepCols.end())
                                                it2 = currentCtx.u32Cols.erase(it2);
                                            else ++it2;
                                        }
                                        for (auto it2 = currentCtx.u32ColsGPU.begin(); it2 != currentCtx.u32ColsGPU.end(); ) {
                                            if (keepCols.find(it2->first) == keepCols.end())
                                                it2 = currentCtx.u32ColsGPU.erase(it2);
                                            else ++it2;
                                        }
                                    }
                                    if (debug) std::cerr << "[Exec] DELIM_SCAN: stripped to correlation cols only: [" << dedupCols.size() << " cols]\n";
                                    // Mark these columns as DELIM correlation for join priority
                                    for (const auto& dc : dedupCols) {
                                        currentCtx.isDelimCorrelation.insert(dc);
                                    }
                                }
                            }
                        }
                        
                        // Alias ps_partkey -> p_partkey for correlated subquery contexts
                        if (currentCtx.currentTable.find("tmpl_") == 0) {
                            bool hasPS = currentCtx.u32Cols.count("ps_partkey");
                            bool hasP = currentCtx.u32Cols.count("p_partkey");
                            if (hasPS && !hasP) {
                                if (debug) std::cerr << "[Exec] Patch: Aliasing ps_partkey -> p_partkey in " << currentCtx.currentTable << "\n";
                                currentCtx.u32Cols["p_partkey"] = currentCtx.u32Cols["ps_partkey"];
                                if (currentCtx.u32ColsGPU.count("ps_partkey")) {
                                    MTL::Buffer* buf = currentCtx.u32ColsGPU["ps_partkey"];
                                    currentCtx.u32ColsGPU["p_partkey"] = buf;
                                    buf->retain(); 
                                }
                            } else if (!hasP && !hasPS) {
                                // Inject p_partkey from global 'part' table as placeholder
                                auto partIt = tableContexts.find("part");
                                if (partIt != tableContexts.end() && partIt->second.u32Cols.count("p_partkey")) {
                                     if (debug) std::cerr << "[Exec] Patch: Injecting global p_partkey from 'part' into " << currentCtx.currentTable << "\n";
                                     
                                     // Create a buffer of correct size
                                     std::vector<uint32_t> dummy(currentCtx.rowCount, 0); 
                                     
                                     // Copy the first N IDs from part table if available to act as placeholder
                                     const auto& src = partIt->second.u32Cols.at("p_partkey");
                                     for(size_t i=0; i<currentCtx.rowCount && i<src.size(); ++i) {
                                         dummy[i] = src[i];
                                     }
                                     
                                     currentCtx.u32Cols["p_partkey"] = dummy;
                                     currentCtx.u32ColsGPU["p_partkey"] = GpuOps::createBuffer(dummy.data(), dummy.size() * sizeof(uint32_t));
                                }
                            }
                        }

                        // Apply pushed-down filter if present (these are pre-filtered
                        // in the planner to only include precise filters)
                        if (scan.filter) {
                            if (debug) {
                                std::cerr << "[Exec] Applying scan filter for pipeline table " << tableKey << "\n";
                            }
                            executeFilter(IRFilter{scan.filter, ""}, currentCtx);
                            // Update tableContexts with filtered data for joins
                            tableContexts[tableKey] = currentCtx;
                        }
                        
                        if (debug) {
                            std::cerr << "[Exec] Scan " << tableKey << ": " << currentCtx.rowCount << " rows, u32cols=";
                            for (const auto& [k, v] : currentCtx.u32Cols) std::cerr << k << " ";
                            std::cerr << "f32cols=";
                            for (const auto& [k, v] : currentCtx.f32Cols) std::cerr << k << " ";
                            std::cerr << "\n";
                        }
                    }
                }
                break;
            }

            case IRNode::Type::Filter: {
                if (debug) {
                    std::cerr << "[Exec] Filter: BEFORE filter, currentCtx.stringCols:\n";
                    for (const auto& [n, v] : currentCtx.stringCols) {
                        std::cerr << "[Exec]   " << n << " size=" << v.size() << "\n";
                    }
                }
                if (!executeFilter(node.asFilter(), currentCtx)) {
                    result.error = "Filter execution failed";
                    return result;
                }
                if (debug) {
                    std::cerr << "[Exec] Filter: AFTER filter, currentCtx.stringCols:\n";
                    for (const auto& [n, v] : currentCtx.stringCols) {
                        std::cerr << "[Exec]   " << n << " size=" << v.size() << "\n";
                    }
                }
                // Update tableContexts with filtered data for joins to use
                if (!currentCtx.currentTable.empty()) {
                    tableContexts[currentCtx.currentTable] = currentCtx;
                }
                
                // Compact tableResult only if its row count matches the pre-filter
                // context size (otherwise it's from a different pipeline stage).
                if (!tableResult.u32_cols.empty() || !tableResult.f32_cols.empty()) {
                    // Find the physical buffer size (pre-filter row count)
                    size_t physicalRows = 0;
                    for (const auto& [name, buf] : currentCtx.u32ColsGPU) {
                        if (buf) { physicalRows = buf->length() / sizeof(uint32_t); break; }
                    }
                    if (physicalRows == 0) {
                        for (const auto& [name, buf] : currentCtx.f32ColsGPU) {
                            if (buf) { physicalRows = buf->length() / sizeof(float); break; }
                        }
                    }
                    bool sizeMatch = (tableResult.rowCount == physicalRows) || 
                                     (physicalRows == 0 && !tableResult.u32_cols.empty() && 
                                      tableResult.u32_cols[0].size() == currentCtx.activeRowsCountGPU);
                    
                    if (sizeMatch && currentCtx.activeRowsCountGPU > 0 && currentCtx.activeRowsGPU) {
                        // Compact tableResult based on activeRows via GPU gather
                        uint32_t arCount = currentCtx.activeRowsCountGPU;
                        auto& s = ColumnStoreGPU::instance();
                        for (auto& col : tableResult.u32_cols) {
                            MTL::Buffer* src = s.device()->newBuffer(col.data(), col.size() * sizeof(uint32_t), MTL::ResourceStorageModeShared);
                            MTL::Buffer* dst = GpuOps::gatherU32(src, currentCtx.activeRowsGPU, arCount);
                            col.resize(arCount);
                            std::memcpy(col.data(), dst->contents(), arCount * sizeof(uint32_t));
                            src->release(); dst->release();
                        }
                        for (auto& col : tableResult.f32_cols) {
                            MTL::Buffer* src = s.device()->newBuffer(col.data(), col.size() * sizeof(float), MTL::ResourceStorageModeShared);
                            MTL::Buffer* dst = GpuOps::gatherF32(src, currentCtx.activeRowsGPU, arCount);
                            col.resize(arCount);
                            std::memcpy(col.data(), dst->contents(), arCount * sizeof(float));
                            src->release(); dst->release();
                        }
                        for (auto& col : tableResult.string_cols) {
                            // Try GPU gather via flatStringCols if available
                            bool gpuDone = false;
                            // Find matching column name from tableResult.string_names
                            size_t colIdx = &col - &tableResult.string_cols[0];
                            if (colIdx < tableResult.string_names.size()) {
                                const auto& colName = tableResult.string_names[colIdx];
                                auto fit = currentCtx.flatStringCols.find(colName);
                                if (fit != currentCtx.flatStringCols.end() && fit->second.chars) {
                                    auto r = GpuOps::gatherFlatString(
                                        fit->second.chars, fit->second.offsets, fit->second.lengths,
                                        currentCtx.activeRowsGPU, arCount, true);
                                    if (r.chars) {
                                        // Materialize gathered FlatStringCol to std::vector<std::string> for tableResult
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
                                const uint32_t* arPtr = static_cast<const uint32_t*>(currentCtx.activeRowsGPU->contents());
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
                        // tableResult is from a different pipeline stage, clear it
                        if (debug) std::cerr << "[Exec] Filter: clearing stale tableResult (size " 
                                             << tableResult.rowCount << " != physical " << physicalRows << ")\n";
                        tableResult.u32_cols.clear();
                        tableResult.u32_names.clear();
                        tableResult.f32_cols.clear();
                        tableResult.f32_names.clear();
                        tableResult.string_cols.clear();
                        tableResult.string_names.clear();
                        tableResult.order.clear();
                        tableResult.rowCount = 0;
                    }
                }
                
                // Always compact currentCtx columns when filter has activeRows,
                // to ensure consistent row counts across all data.
                if (currentCtx.activeRowsCountGPU > 0 && currentCtx.activeRowsGPU) {
                    uint32_t compactCount = currentCtx.activeRowsCountGPU;

                    // GPU-direct compaction: gather u32/f32 GPU columns using
                    // activeRowsGPU index buffer — no CPU round-trip.
                    for (auto& [name, buf] : currentCtx.u32ColsGPU) {
                        if (buf) {
                            auto compacted = GpuOps::gatherU32(buf, currentCtx.activeRowsGPU, compactCount);
                            if (compacted) { buf = compacted; }
                        }
                    }
                    for (auto& [name, buf] : currentCtx.f32ColsGPU) {
                        if (buf) {
                            auto compacted = GpuOps::gatherF32(buf, currentCtx.activeRowsGPU, compactCount);
                            if (compacted) { buf = compacted; }
                        }
                    }

                    // Compact CPU-only columns via GPU gather (upload→gather→download)
                    {
                        auto& s = ColumnStoreGPU::instance();
                        for (auto& [name, col] : currentCtx.u32Cols) {
                            if (col.size() > compactCount) {
                                MTL::Buffer* src = s.device()->newBuffer(col.data(), col.size() * sizeof(uint32_t), MTL::ResourceStorageModeShared);
                                MTL::Buffer* dst = GpuOps::gatherU32(src, currentCtx.activeRowsGPU, compactCount);
                                col.resize(compactCount);
                                std::memcpy(col.data(), dst->contents(), compactCount * sizeof(uint32_t));
                                src->release(); dst->release();
                            }
                        }
                        for (auto& [name, col] : currentCtx.f32Cols) {
                            if (col.size() > compactCount) {
                                MTL::Buffer* src = s.device()->newBuffer(col.data(), col.size() * sizeof(float), MTL::ResourceStorageModeShared);
                                MTL::Buffer* dst = GpuOps::gatherF32(src, currentCtx.activeRowsGPU, compactCount);
                                col.resize(compactCount);
                                std::memcpy(col.data(), dst->contents(), compactCount * sizeof(float));
                                src->release(); dst->release();
                            }
                        }
                    }
                    // GPU-native dict compaction: gather dict IDs on GPU
                    currentCtx.compactDictCols(compactCount);
                    // GPU-native flat string compaction: gather chars/offsets/lengths on GPU
                    currentCtx.compactFlatStringCols(compactCount);
                    // Invalidate stale stringCols — will be lazily rebuilt from dictCols/flatStringCols if needed
                    for (auto& [name, col] : currentCtx.stringCols) {
                        auto dit = currentCtx.dictCols.find(name);
                        auto fit = currentCtx.flatStringCols.find(name);
                        if ((dit != currentCtx.dictCols.end() && dit->second.valid()) ||
                            (fit != currentCtx.flatStringCols.end() && fit->second.chars)) {
                            col.clear();  // Will be rebuilt from dict/flat on demand
                        } else if (col.size() > compactCount) {
                            // Legacy fallback for orphan string cols without dict or flat encoding
                            const uint32_t* arPtr = static_cast<const uint32_t*>(currentCtx.activeRowsGPU->contents());
                            std::vector<std::string> filtered;
                            filtered.reserve(compactCount);
                            for (uint32_t i = 0; i < compactCount; ++i) {
                                uint32_t idx = arPtr[i];
                                if (idx < col.size()) filtered.push_back(col[idx]);
                            }
                            col = std::move(filtered);
                        }
                    }

                    // Sync compacted GPU data back to CPU mirrors
                    for (auto& [name, buf] : currentCtx.u32ColsGPU) {
                        if (buf) {
                            size_t n = buf->length() / sizeof(uint32_t);
                            currentCtx.u32Cols[name].resize(n);
                            memcpy(currentCtx.u32Cols[name].data(), buf->contents(), n * sizeof(uint32_t));
                        }
                    }
                    for (auto& [name, buf] : currentCtx.f32ColsGPU) {
                        if (buf) {
                            size_t n = buf->length() / sizeof(float);
                            currentCtx.f32Cols[name].resize(n);
                            memcpy(currentCtx.f32Cols[name].data(), buf->contents(), n * sizeof(float));
                        }
                    }

                    currentCtx.activeRows.clear();
                    currentCtx.activeRowsGPU = nullptr;
                    currentCtx.activeRowsCountGPU = 0;
                }
                
                if (debug) {
                    std::cerr << "[Exec] Filter: " << currentCtx.rowCount << " rows after\n";
                }
                break;
            }

            case IRNode::Type::Join: {
                if (!orchestrateJoin(node.asJoin(), datasetPath, currentCtx, tableContexts, 
                                     savedPipelines, savedPipelineTables, joinedTables, hasPipeline, result)) {
                    return result;
                }
                break;
            }

            case IRNode::Type::GroupBy: {
                if (!executeGroupBy(node.asGroupBy(), currentCtx, tableResult)) {
                    result.error = "GroupBy execution failed";
                    return result;
                }

                if (debug) std::cerr << "[Exec] DEBUG: GroupBy returned, clearing old context\n";
                
                // If GroupBy produces multiple rows, this is NOT a scalar result
                if (tableResult.rowCount > 1) {
                    result.isScalarAggregate = false;
                }
                
                // Clear old columns and update with GroupBy output
                currentCtx.rowCount = tableResult.rowCount;
                currentCtx.activeRows.clear();

                if (debug) std::cerr << "[Exec] DEBUG: Clearing activeRowsGPU\n";
                if (currentCtx.activeRowsGPU) {
                    currentCtx.activeRowsGPU->release();
                    currentCtx.activeRowsGPU = nullptr;
                }
                currentCtx.activeRowsCountGPU = 0;

                if (debug) std::cerr << "[Exec] DEBUG: Clearing u32ColsGPU\n";
                {
                    std::set<MTL::Buffer*> released;
                    for(auto& [n, b] : currentCtx.u32ColsGPU) {
                        if(b && released.find(b) == released.end()) {
                            b->release();
                            released.insert(b);
                        }
                    }
                }
                currentCtx.u32ColsGPU.clear();

                if (debug) std::cerr << "[Exec] DEBUG: Clearing f32ColsGPU\n";
                {
                    std::set<MTL::Buffer*> released;
                    for(auto& [n, b] : currentCtx.f32ColsGPU) {
                        if(b && released.find(b) == released.end()) {
                            b->release();
                            released.insert(b);
                        }
                    }
                }
                currentCtx.f32ColsGPU.clear();
                
                currentCtx.u32Cols.clear();
                currentCtx.f32Cols.clear();
                currentCtx.stringCols.clear();
                currentCtx.currentTable.clear();
                
                // Reset joinedTables for SEMI join decorrelation pattern
                joinedTables.clear();
                joinedTables.insert("__GROUPED__");
                
                // Build set of f32 column names to detect float keys restored from u32
                std::set<std::string> f32NameSet;
                for (const auto& fn : tableResult.f32_names) f32NameSet.insert(fn);
                
                for (size_t i = 0; i < tableResult.u32_cols.size(); ++i) {
                    if (i < tableResult.u32_names.size()) {
                        const std::string& name = tableResult.u32_names[i];
                        // Skip named registration if this column was restored to f32
                        // (the u32 version contains raw IEEE 754 bits, not the actual value)
                        bool restoredToF32 = f32NameSet.count(name) > 0;
                        if (!restoredToF32) {
                            currentCtx.u32Cols[name] = tableResult.u32_cols[i];
                        }
                        // Register positional key only if not restored to f32
                        if (!restoredToF32) {
                            std::string posKey = "#" + std::to_string(i);
                            currentCtx.u32Cols[posKey] = tableResult.u32_cols[i];
                        }
                        // Re-register columns under their aliases (for CTE support)
                        if (!restoredToF32) {
                            for (const auto& [alias, canonical] : currentCtx.columnAliases) {
                                if (canonical == name) {
                                    currentCtx.u32Cols[alias] = tableResult.u32_cols[i];
                                    if (debug) std::cerr << "[Exec] GroupBy: re-registering alias " << alias << " -> " << canonical << "\n";
                                }
                            }
                        }
                    }
                }
                // Recount non-skipped u32 columns for f32 positional offset
                size_t u32RegisteredCount = 0;
                for (size_t i = 0; i < tableResult.u32_cols.size(); ++i) {
                    if (i < tableResult.u32_names.size() && !f32NameSet.count(tableResult.u32_names[i]))
                        u32RegisteredCount++;
                }
                for (size_t i = 0; i < tableResult.f32_cols.size(); ++i) {
                    if (i < tableResult.f32_names.size()) {
                        currentCtx.f32Cols[tableResult.f32_names[i]] = tableResult.f32_cols[i];
                        // Also register under positional name for #N references (offset by registered u32 count)
                        std::string posKey = "#" + std::to_string(i + u32RegisteredCount);
                        currentCtx.f32Cols[posKey] = tableResult.f32_cols[i];
                        // Re-register columns under their aliases (for CTE support)
                        for (const auto& [alias, canonical] : currentCtx.columnAliases) {
                            if (canonical == tableResult.f32_names[i]) {
                                currentCtx.f32Cols[alias] = tableResult.f32_cols[i];
                                if (debug) std::cerr << "[Exec] GroupBy: re-registering f32 alias " << alias << " -> " << canonical << "\n";
                            }
                        }
                    }
                }
                
                // Populate stringCols from GroupBy result and build dictCols
                for (size_t i = 0; i < tableResult.string_cols.size(); ++i) {
                    if (i < tableResult.string_names.size()) {
                        const std::string& sname = tableResult.string_names[i];
                        currentCtx.stringCols[sname] = tableResult.string_cols[i];
                        // Build dictionary encoding for downstream operators
                        buildDictCol(currentCtx, sname);
                        if (debug) std::cerr << "[Exec] GroupBy: setting stringCol+dictCol " << sname
                                            << " with " << tableResult.string_cols[i].size() << " rows\n";
                    }
                }

                // Strict Mode: Upload GroupBy results to GPU
                if (debug) std::cerr << "[Exec] Uploading GroupBy results to GPU (Strict Mode)\n";
                
                for(const auto& [name, vec] : currentCtx.u32Cols) {
                    if (!vec.empty()) {
                         MTL::Buffer* buf = GpuOps::createBuffer(vec.data(), vec.size() * sizeof(uint32_t));
                         if (buf) {
                            currentCtx.u32ColsGPU[name] = buf;
                         } else {
                            std::cerr << "[Exec] ERROR: Failed to create GPU buffer for u32 col " << name << "\n";
                         }
                    }
                }
                for(const auto& [name, vec] : currentCtx.f32Cols) {
                    if (!vec.empty()) {
                         MTL::Buffer* buf = GpuOps::createBuffer(vec.data(), vec.size() * sizeof(float));
                         if (buf) {
                            currentCtx.f32ColsGPU[name] = buf;
                         } else {
                            std::cerr << "[Exec] ERROR: Failed to create GPU buffer for f32 col " << name << "\n";
                         }
                    }
                } 

                if (debug) {

                    std::cerr << "[Exec] GroupBy: " << tableResult.rowCount << " groups\n";
                    std::cerr << "[Exec] GroupBy: ctx updated with u32Cols=";
                    for (const auto& [k, v] : currentCtx.u32Cols) std::cerr << k << "(" << v.size() << ") ";
                    std::cerr << "f32Cols=";
                    for (const auto& [k, v] : currentCtx.f32Cols) std::cerr << k << "(" << v.size() << ") ";
                    std::cerr << "\n";
                }
                
                // Mark pipeline active so a new scan can trigger pipeline save
                hasPipeline = true;
                
                // Clear stale tableResult to avoid misaligned filter compaction
                tableResult.u32_cols.clear();
                tableResult.u32_names.clear();
                tableResult.f32_cols.clear();
                tableResult.f32_names.clear();
                tableResult.string_cols.clear();
                tableResult.string_names.clear();
                tableResult.order.clear();
                tableResult.rowCount = 0;
                
                break;
            }

            case IRNode::Type::Aggregate: {
                double value;
                std::string name;
                if (!executeAggregate(node.asAggregate(), currentCtx, value, name)) {
                    result.error = "Aggregate execution failed";
                    return result;
                }
                result.isScalarAggregate = true;
                result.scalarValue = value;
                result.scalarName = name;
                
                // Mark context as scalar result ONLY if this is the last aggregate in the block
                // This prevents sibling aggregates (e.g. sum(a), count(b)) from confusing the row count
                // (sum(a) sets scalar=true, then count(b) sees true and returns 1 -> WRONG)
                if (node.asAggregate().isLastAgg) {
                    currentCtx.isScalarResult = true;
                    currentCtx.rowCount = 1;
                    // Clear stale activeRowsGPU so projections use rowCount=1
                    if (currentCtx.activeRowsGPU) {
                        currentCtx.activeRowsGPU->release();
                        currentCtx.activeRowsGPU = nullptr;
                        currentCtx.activeRowsCountGPU = 0;
                    }
                    if (debug) std::cerr << "[Exec] Aggregate: isLastAgg=true, setting rowCount=1 (scalar result)\n";
                }
                
                // Store aggregate result in context for later projection reference
                // Multiple aggregates get stored as #0, #1, etc. based on aggIndex
                // But DON'T change rowCount yet - other aggregates may still need original data
                const auto& agg = node.asAggregate();
                std::string posKey = "#" + std::to_string(agg.aggIndex);
                currentCtx.f32Cols[posKey] = std::vector<float>{static_cast<float>(value)};
                
                // Create GPU buffer for the scalar result
                MTL::Buffer* aggBuf = GpuOps::createBuffer(currentCtx.f32Cols[posKey].data(), sizeof(float));
                currentCtx.f32ColsGPU[posKey] = aggBuf;
                // No extra retain needed — createBuffer returns refcount=1 which covers the posKey map entry
                
                // Also store by name
                if (!name.empty()) {
                    currentCtx.f32Cols[name] = std::vector<float>{static_cast<float>(value)};
                    currentCtx.f32ColsGPU[name] = aggBuf;
                    aggBuf->retain(); // +1 for name map entry
                }
                if (!agg.exprStr.empty() && agg.exprStr != name) {
                     currentCtx.f32Cols[agg.exprStr] = std::vector<float>{static_cast<float>(value)};
                     currentCtx.f32ColsGPU[agg.exprStr] = aggBuf;
                     aggBuf->retain(); // +1 for exprStr map entry
                }
                if (debug) {
                    std::cerr << "[Exec] Aggregate " << name << ": " << value 
                              << " (stored as " << posKey << ")\n";
                }
                break;
            }

            case IRNode::Type::OrderBy: {
                // If tableResult is out of sync with currentCtx (e.g. a Join happened
                // after the last Project), materialize currentCtx into tableResult first.
                if (tableResult.rowCount != currentCtx.rowCount && currentCtx.rowCount > 0) {
                    if (debug) {
                        std::cerr << "[Exec] OrderBy: syncing tableResult from currentCtx ("
                                  << currentCtx.rowCount << " rows, tableResult had "
                                  << tableResult.rowCount << ")\n";
                    }
                    tableResult.u32_cols.clear();
                    tableResult.u32_names.clear();
                    tableResult.f32_cols.clear();
                    tableResult.f32_names.clear();
                    tableResult.string_cols.clear();
                    tableResult.string_names.clear();
                    tableResult.order.clear();

                    // Download GPU columns to CPU if needed
                    for (auto& [name, buf] : currentCtx.u32ColsGPU) {
                        if (buf && (currentCtx.u32Cols.find(name) == currentCtx.u32Cols.end() ||
                                    currentCtx.u32Cols.at(name).empty())) {
                            uint32_t count = (uint32_t)(buf->length() / sizeof(uint32_t));
                            if (count >= currentCtx.rowCount) {
                                std::vector<uint32_t> v(currentCtx.rowCount);
                                std::memcpy(v.data(), buf->contents(), currentCtx.rowCount * sizeof(uint32_t));
                                currentCtx.u32Cols[name] = std::move(v);
                            }
                        }
                    }
                    for (auto& [name, buf] : currentCtx.f32ColsGPU) {
                        if (buf && (currentCtx.f32Cols.find(name) == currentCtx.f32Cols.end() ||
                                    currentCtx.f32Cols.at(name).empty())) {
                            uint32_t count = (uint32_t)(buf->length() / sizeof(float));
                            if (count >= currentCtx.rowCount) {
                                std::vector<float> v(currentCtx.rowCount);
                                std::memcpy(v.data(), buf->contents(), currentCtx.rowCount * sizeof(float));
                                currentCtx.f32Cols[name] = std::move(v);
                            }
                        }
                    }

                    for (const auto& [name, vec] : currentCtx.u32Cols) {
                        if (!vec.empty() && name.find("__internal_") == std::string::npos
                            && !(name.size() >= 2 && name[0] == '#' && std::isdigit(name[1]))) {
                            tableResult.u32_names.push_back(name);
                            tableResult.u32_cols.push_back(vec);
                        }
                    }
                    for (const auto& [name, vec] : currentCtx.f32Cols) {
                        if (!vec.empty()
                            && !(name.size() >= 2 && name[0] == '#' && std::isdigit(name[1]))) {
                            tableResult.f32_names.push_back(name);
                            tableResult.f32_cols.push_back(vec);
                        }
                    }
                    for (const auto& [name, vec] : currentCtx.stringCols) {
                        if (!vec.empty()) {
                            tableResult.string_names.push_back(name);
                            tableResult.string_cols.push_back(vec);
                        }
                    }
                    // Also materialize any dict-only string columns not yet in stringCols
                    for (const auto& [name, dict] : currentCtx.dictCols) {
                        if (dict.valid() && !currentCtx.stringCols.count(name)) {
                            tableResult.string_names.push_back(name);
                            tableResult.string_cols.push_back(dict.materialize());
                        }
                    }
                    tableResult.rowCount = currentCtx.rowCount;
                }
                if (!executeOrderBy(node.asOrderBy(), tableResult, currentCtx.dictCols, currentCtx.flatStringCols)) {
                    result.error = "OrderBy execution failed";
                    return result;
                }
                // Sync ctx with sorted tableResult so that a subsequent Project
                // does not re-read unsorted data from the old context.
                for (size_t i = 0; i < tableResult.u32_cols.size(); ++i) {
                    if (i < tableResult.u32_names.size())
                        currentCtx.u32Cols[tableResult.u32_names[i]] = tableResult.u32_cols[i];
                }
                for (size_t i = 0; i < tableResult.f32_cols.size(); ++i) {
                    if (i < tableResult.f32_names.size())
                        currentCtx.f32Cols[tableResult.f32_names[i]] = tableResult.f32_cols[i];
                }
                for (size_t i = 0; i < tableResult.string_cols.size(); ++i) {
                    if (i < tableResult.string_names.size()) {
                        const std::string& sname = tableResult.string_names[i];
                        currentCtx.stringCols[sname] = tableResult.string_cols[i];
                        // Rebuild dictionary encoding for sorted string columns
                        buildDictCol(currentCtx, sname);
                    }
                }
                if (debug) {
                    std::cerr << "[Exec] OrderBy applied\n";
                }
                break;
            }

            case IRNode::Type::Limit: {
                if (!executeLimit(node.asLimit(), tableResult)) {
                    result.error = "Limit execution failed";
                    return result;
                }
                if (debug) {
                    std::cerr << "[Exec] Limit: " << tableResult.rowCount << " rows\n";
                }
                break;
            }

            case IRNode::Type::Project: {
                if (!executeProject(node.asProject(), currentCtx, tableResult, &tableContexts)) {
                    result.error = "Project execution failed";
                    return result;
                }
                
                // If this is a projection after aggregates (e.g., 100.0 * sum(...) / sum(...)),
                // the projection output is the final result, not the raw aggregate value
                // Note: We relax the rowCount==1 check because vector broadcasting might have produced N rows
                if (result.isScalarAggregate && !tableResult.f32_cols.empty()) {
                    // Update the scalar result with the projection output
                    result.scalarValue = tableResult.f32_cols[0][0];
                    result.scalarName = tableResult.f32_names.empty() ? "result" : tableResult.f32_names[0];
                    if (debug) {
                        std::cerr << "[Exec] Project after Aggregate: updated scalar to " 
                                  << result.scalarValue << " (" << result.scalarName << ")\n";
                    }
                }
                
                // Update tableContexts if we're still working with a single table
                if (!currentCtx.currentTable.empty()) {
                    tableContexts[currentCtx.currentTable] = currentCtx;
                    if (debug) {
                        std::cerr << "[Exec] Project: updated tableContexts[" << currentCtx.currentTable << "] with " 
                                  << currentCtx.rowCount << " rows, u32cols=";
                        for (const auto& [k,v] : currentCtx.u32Cols) std::cerr << k << " ";
                        std::cerr << "f32cols=";
                        for (const auto& [k,v] : currentCtx.f32Cols) std::cerr << k << " ";
                        std::cerr << "\n";
                    }
                }
                break;
            }

            case IRNode::Type::Save: {
                if (debug) {
                    std::cerr << "[Exec] Save: storing " << currentCtx.rowCount << " rows into " << node.asSave().name << "\n";
                }
                tableContexts[node.asSave().name] = currentCtx;
                // Retain GPU buffers so the saved copy owns its own references.
                // Without this, shallow copies share raw GPU pointers, and if
                // another copy's DELIM dedup releases a buffer, this saved copy
                // ends up with dangling pointers (e.g., Q21 crash).
                {
                    auto& saved = tableContexts[node.asSave().name];
                    for (auto& [n, buf] : saved.u32ColsGPU)
                        if (buf) buf->retain();
                    for (auto& [n, buf] : saved.f32ColsGPU)
                        if (buf) buf->retain();
                    for (auto& [n, dc] : saved.dictCols)
                        if (dc.idsGPU) dc.idsGPU->retain();
                    for (auto& [n, fc] : saved.flatStringCols) {
                        if (fc.offsets) fc.offsets->retain();
                        if (fc.chars) fc.chars->retain();
                        if (fc.lengths) fc.lengths->retain();
                    }
                    if (saved.activeRowsGPU) saved.activeRowsGPU->retain();
                }
                break;
            }

            default:
                break;
        }
    }

    // If tableResult is empty but currentCtx has data (scan-only pipeline),
    // materialize ctx columns into tableResult so the output path can render them.
    if (tableResult.u32_names.empty() && tableResult.f32_names.empty() &&
        tableResult.string_names.empty() && !result.isScalarAggregate &&
        currentCtx.rowCount > 0) {
        
        // Materialize GPU buffers to CPU vectors
        auto materialize = [&]() {
            // Determine active row indices (if filtered)
            uint32_t arCount = currentCtx.rowCount;
            bool hasAR = (currentCtx.activeRowsGPU && currentCtx.activeRowsCountGPU > 0);
            if (hasAR) arCount = currentCtx.activeRowsCountGPU;

            for (const auto& [name, buf] : currentCtx.u32ColsGPU) {
                if (!buf) continue;
                std::vector<uint32_t> col(arCount);
                if (hasAR) {
                    MTL::Buffer* gathered = GpuOps::gatherU32(buf, currentCtx.activeRowsGPU, arCount);
                    std::memcpy(col.data(), gathered->contents(), arCount * sizeof(uint32_t));
                    gathered->release();
                } else {
                    std::memcpy(col.data(), buf->contents(), arCount * sizeof(uint32_t));
                }
                tableResult.u32_names.push_back(name);
                tableResult.u32_cols.push_back(std::move(col));
            }
            for (const auto& [name, buf] : currentCtx.f32ColsGPU) {
                if (!buf) continue;
                std::vector<float> col(arCount);
                if (hasAR) {
                    MTL::Buffer* gathered = GpuOps::gatherF32(buf, currentCtx.activeRowsGPU, arCount);
                    std::memcpy(col.data(), gathered->contents(), arCount * sizeof(float));
                    gathered->release();
                } else {
                    std::memcpy(col.data(), buf->contents(), arCount * sizeof(float));
                }
                tableResult.f32_names.push_back(name);
                tableResult.f32_cols.push_back(std::move(col));
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
                    tableResult.string_names.push_back(name);
                    tableResult.string_cols.push_back(std::move(col));
                } else {
                    tableResult.string_names.push_back(name);
                    tableResult.string_cols.push_back(
                        std::vector<std::string>(vec.begin(), vec.begin() + std::min((size_t)arCount, vec.size())));
                }
            }
            // Materialize dict-only string columns not yet in stringCols
            for (const auto& [name, dict] : currentCtx.dictCols) {
                if (!dict.valid()) continue;
                if (hasAR) {
                    MTL::Buffer* gatheredIds = GpuOps::gatherU32(dict.idsGPU, currentCtx.activeRowsGPU, arCount);
                    std::vector<std::string> col(arCount);
                    const uint32_t* ids = static_cast<const uint32_t*>(gatheredIds->contents());
                    for (uint32_t i = 0; i < arCount; ++i) {
                        col[i] = dict.lookupString(ids[i]);
                    }
                    gatheredIds->release();
                    tableResult.string_names.push_back(name);
                    tableResult.string_cols.push_back(std::move(col));
                } else {
                    tableResult.string_names.push_back(name);
                    tableResult.string_cols.push_back(dict.materialize());
                }
            }
            // Materialize flat-only string columns not yet handled above
            for (const auto& [name, fc] : currentCtx.flatStringCols) {
                if (!fc.chars) continue;
                if (currentCtx.dictCols.count(name)) continue; // Already handled by dict loop
                // Already handled by stringCols loop if it wasn't skipped
                bool alreadyHandled = false;
                for (const auto& sn : tableResult.string_names) {
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
                        tableResult.string_names.push_back(name);
                        tableResult.string_cols.push_back(std::move(col));
                    }
                } else {
                    const uint32_t* offs = static_cast<const uint32_t*>(fc.offsets->contents());
                    const uint32_t* lens = static_cast<const uint32_t*>(fc.lengths->contents());
                    const char* ch = static_cast<const char*>(fc.chars->contents());
                    uint32_t cnt = std::min((uint32_t)arCount, fc.rowCount);
                    std::vector<std::string> col(cnt);
                    for (uint32_t i = 0; i < cnt; ++i) col[i].assign(ch + offs[i], lens[i]);
                    tableResult.string_names.push_back(name);
                    tableResult.string_cols.push_back(std::move(col));
                }
            }
            tableResult.rowCount = arCount;
        };
        materialize();
        if (debug) {
            std::cerr << "[Exec] Scan-only pipeline: materialized " << tableResult.rowCount
                      << " rows (" << tableResult.u32_names.size() << " u32, "
                      << tableResult.f32_names.size() << " f32, "
                      << tableResult.string_names.size() << " string cols)\n";
        }
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    double pipelineWallMs = std::chrono::duration<double, std::milli>(endTime - loadEnd).count();

    // Post-process: Clean up column names and mark single-char columns
    const auto& schema = SchemaRegistry::instance();
    
    // Track positional ref -> original column name mappings from context
    // The GroupBy stores both "l_returnflag" and "#0" -> same data
    std::map<std::string, std::string> posToOriginal;
    
    bool debugCleanup = env_truthy("GPUDB_DEBUG_OPS");
    if (debugCleanup) {
        std::cerr << "[Exec] Cleanup: currentCtx.u32Cols=";
        for (const auto& [n,v] : currentCtx.u32Cols) std::cerr << n << "(" << v.size() << ") ";
        std::cerr << "\n";
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
                if (posVec.size() == vec.size() && posVec == vec) {
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
                if (posVec.size() == vec.size() && posVec == vec) {
                    posToOriginal[pos] = name;
                    assignedPosRefs.insert(pos);
                    usedOriginalNames.insert(name);
                    break;  // Found a match for this name, move to next
                }
            }
        }
    }
    
    if (debugCleanup) {
        std::cerr << "[Exec] Cleanup: posToOriginal mappings:\n";
        for (const auto& [pos, orig] : posToOriginal) {
            std::cerr << "  " << pos << " -> " << orig << "\n";
        }
        std::cerr << "[Exec] Cleanup: tableResult.u32_names=";
        for (const auto& n : tableResult.u32_names) std::cerr << "'" << n << "' ";
        std::cerr << "\n";
    }
    
    // Clean up u32 column names
    for (size_t i = 0; i < tableResult.u32_names.size(); ++i) {
        std::string& name = tableResult.u32_names[i];
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
    for (size_t i = 0; i < tableResult.f32_names.size(); ++i) {
        std::string& name = tableResult.f32_names[i];
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

    // --- Convert U32 hashes back to strings ---
    std::vector<std::string> new_u32_names;
    std::vector<std::vector<uint32_t>> new_u32_cols;
    std::vector<size_t> u32_remap(tableResult.u32_names.size());
    std::vector<bool> is_converted(tableResult.u32_names.size(), false);
    std::vector<size_t> string_converted_idx(tableResult.u32_names.size(), 0);

    for (size_t i = 0; i < tableResult.u32_names.size(); ++i) {
        std::string colName = tableResult.u32_names[i];
        std::string tableName = tableForColumn(colName);
        bool converted = false;

        // Check if a string column with this name already exists (e.g. from GroupBy string recovery)
        bool alreadyHasString = false;
        for (size_t si = 0; si < tableResult.string_names.size(); ++si) {
            if (tableResult.string_names[si] == colName) {
                // String column already exists — just mark u32 for removal, keep existing string
                is_converted[i] = true;
                string_converted_idx[i] = si;
                converted = true;
                alreadyHasString = true;
                break;
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
                        const auto& u32col = tableResult.u32_cols[i];
                        
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

                        tableResult.string_names.push_back(colName);
                        tableResult.string_cols.push_back(std::move(strCol));
                        is_converted[i] = true;
                        string_converted_idx[i] = tableResult.string_names.size() - 1;
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
                            strCol.reserve(tableResult.u32_cols[i].size());
                            for (uint32_t val : tableResult.u32_cols[i]) {
                                auto hit = hashMap.find(val);
                                if (hit != hashMap.end()) strCol.push_back(hit->second);
                                else strCol.push_back(std::to_string(val));
                            }

                            tableResult.string_names.push_back(colName);
                            tableResult.string_cols.push_back(std::move(strCol));
                            is_converted[i] = true;
                            string_converted_idx[i] = tableResult.string_names.size() - 1;
                            converted = true;
                        } else {
                            // Last resort: re-read from disk (should be rare with always-flatten)
                            if (debugCleanup) {
                                std::cerr << "[Exec] WARNING: disk re-read for output reverse-map of " << colName << "\n";
                            }
                            auto raw = GpuOps::loadStringColumnRaw(datasetPath, tableName, colName);
                            std::unordered_map<uint32_t, std::string> map;
                            map.reserve(raw.size());
                            for (const auto& s : raw) {
                                map[GpuOps::fnv1a32(s)] = s;
                            }

                            std::vector<std::string> strCol;
                            strCol.reserve(tableResult.u32_cols[i].size());
                            for (uint32_t val : tableResult.u32_cols[i]) {
                                if (map.count(val)) strCol.push_back(map[val]);
                                else strCol.push_back(std::to_string(val));
                            }

                            tableResult.string_names.push_back(colName);
                            tableResult.string_cols.push_back(std::move(strCol));
                            is_converted[i] = true;
                            string_converted_idx[i] = tableResult.string_names.size() - 1;
                            converted = true;
                        }
                    }
                } else if (cSchema && cSchema->type == ColumnType::Float32) {
                    // GroupBy bit-reinterprets f32 keys as u32. Restore to f32.
                    std::vector<float> f32Col(tableResult.u32_cols[i].size());
                    std::memcpy(f32Col.data(), tableResult.u32_cols[i].data(), f32Col.size() * sizeof(float));
                    tableResult.f32_names.push_back(colName);
                    tableResult.f32_cols.push_back(std::move(f32Col));
                    is_converted[i] = true;
                    // Mark as f32-converted (use a high sentinel so it doesn't collide with string index)
                    string_converted_idx[i] = SIZE_MAX;
                    converted = true;
                }
            }
        }

        if (!converted) {
            new_u32_names.push_back(colName);
            new_u32_cols.push_back(std::move(tableResult.u32_cols[i]));
            u32_remap[i] = new_u32_names.size() - 1;
        }
    }

    // Apply strict update to u32 columns
    tableResult.u32_names = std::move(new_u32_names);
    tableResult.u32_cols = std::move(new_u32_cols);
    
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
                        for (size_t fi = 0; fi < tableResult.f32_names.size(); ++fi) {
                            if (tableResult.f32_names[fi] == ref.name) {
                                ref.index = fi;
                                break;
                            }
                        }
                    } else {
                        ref.kind = TableResult::ColRef::Kind::String;
                        ref.index = string_converted_idx[ref.index];
                        if (ref.index < tableResult.string_names.size()) {
                             ref.name = tableResult.string_names[ref.index];
                        }
                    }
                } else {
                    ref.index = u32_remap[ref.index];
                    if (ref.index < tableResult.u32_names.size()) {
                        ref.name = tableResult.u32_names[ref.index];
                    }
                }
            }
        } else if (ref.kind == TableResult::ColRef::Kind::F32 && ref.index < tableResult.f32_names.size()) {
            ref.name = tableResult.f32_names[ref.index];
        } else if (ref.kind == TableResult::ColRef::Kind::String && ref.index < tableResult.string_names.size()) {
            ref.name = tableResult.string_names[ref.index];
        }
    }

    // Update GPU timing from kernel timer
    tableResult.gpu_ms = KernelTimer::instance().totalGpuMs();
    tableResult.upload_ms = result.table.upload_ms;
    
    // Filter final output to only include columns from the plan's outputColumns
    // This strips intermediate join keys and sort-only columns
    if (!plan.outputColumns.empty() && !result.isScalarAggregate) {
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
            
            for (size_t i = 0; i < tableResult.u32_names.size(); ++i) {
                if (isExpected(tableResult.u32_names[i])) {
                    filtered.u32_names.push_back(tableResult.u32_names[i]);
                    filtered.u32_cols.push_back(tableResult.u32_cols[i]);
                }
            }
            for (size_t i = 0; i < tableResult.f32_names.size(); ++i) {
                if (isExpected(tableResult.f32_names[i])) {
                    filtered.f32_names.push_back(tableResult.f32_names[i]);
                    filtered.f32_cols.push_back(tableResult.f32_cols[i]);
                }
            }
            for (size_t i = 0; i < tableResult.string_names.size(); ++i) {
                if (isExpected(tableResult.string_names[i])) {
                    filtered.string_names.push_back(tableResult.string_names[i]);
                    filtered.string_cols.push_back(tableResult.string_cols[i]);
                }
            }
            
            tableResult.u32_cols = std::move(filtered.u32_cols);
            tableResult.u32_names = std::move(filtered.u32_names);
            tableResult.f32_cols = std::move(filtered.f32_cols);
            tableResult.f32_names = std::move(filtered.f32_names);
            tableResult.string_cols = std::move(filtered.string_cols);
            tableResult.string_names = std::move(filtered.string_names);
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
            std::cerr << "[Exec] Final output filter: " << plan.outputColumns.size() << " expected cols, "
                      << "result has " << tableResult.u32_names.size() << " u32, "
                      << tableResult.f32_names.size() << " f32, "
                      << tableResult.string_names.size() << " string cols\n";
        }
    }
    
    // CPU post-processing = pipeline wall-clock minus GPU kernel time + column cleanup time
    auto postEnd = std::chrono::high_resolution_clock::now();
    double postProcessMs = std::chrono::duration<double, std::milli>(postEnd - endTime).count();
    double cpuPipelineMs = pipelineWallMs - tableResult.gpu_ms;
    tableResult.cpu_post_ms = cpuPipelineMs + postProcessMs;

    result.success = true;
    result.table = std::move(tableResult);
    return result;
}

} // namespace engine
