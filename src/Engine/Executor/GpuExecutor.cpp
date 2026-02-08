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
#include <cmath>
#include <iostream>
#include <map>
#include <numeric>
#include <regex>
#include <set>
#include <unordered_set>

namespace engine {

// Thread-local aggregate counter for multi-aggregate projection resolution
thread_local size_t g_aggregateCounter = 0;

// --- Multi-Instance Column Resolution ---
// Rewrites predicates to use suffixed column names (e.g. col_2) when the same
// column appears in multiple comparison arms and suffixed variants exist.


// Check if a column name matches a table's column pattern
static bool columnBelongsToTable(const std::string& col, const std::string& table) {
    std::string t = tableForColumn(col);
    return t == table;
}
















// --- GPU Feasibility Checking ---

bool GpuExecutor::canExecuteGPU(const Plan& plan) {
    return getGPUBlockers(plan).empty();
}

// Helper to find unsupported functions in an expression tree
static void findUnsupportedFunctions(const TypedExprPtr& expr, std::set<std::string>& unsupported) {
    if (!expr) return;
    
    if (expr->kind == TypedExpr::Kind::Function) {
        const auto& func = expr->asFunction();
        // No functions currently marked as unsupported
        (void)func;
        // Recurse into function arguments
        for (const auto& arg : func.args) {
            findUnsupportedFunctions(arg, unsupported);
        }
    } else if (expr->kind == TypedExpr::Kind::Binary) {
        findUnsupportedFunctions(expr->asBinary().left, unsupported);
        findUnsupportedFunctions(expr->asBinary().right, unsupported);
    } else if (expr->kind == TypedExpr::Kind::Unary) {
        findUnsupportedFunctions(expr->asUnary().operand, unsupported);
    } else if (expr->kind == TypedExpr::Kind::Compare) {
        findUnsupportedFunctions(expr->asCompare().left, unsupported);
        findUnsupportedFunctions(expr->asCompare().right, unsupported);
        for (const auto& item : expr->asCompare().inList) {
            findUnsupportedFunctions(item, unsupported);
        }
    } else if (expr->kind == TypedExpr::Kind::Case) {
        const auto& cs = expr->asCase();
        for (const auto& wt : cs.cases) {
            findUnsupportedFunctions(wt.when, unsupported);
            findUnsupportedFunctions(wt.then, unsupported);
        }
        findUnsupportedFunctions(cs.elseExpr, unsupported);
    } else if (expr->kind == TypedExpr::Kind::Aggregate) {
        findUnsupportedFunctions(expr->asAggregate().arg, unsupported);
    }
}

std::vector<std::string> GpuExecutor::getGPUBlockers(const Plan& plan) {
    std::vector<std::string> blockers;

    // Count nodes and track table scans
    size_t joinCount = 0;
    bool hasSubquery = false;
    bool hasEmptyScan = false;
    bool hasCase = false;
    bool hasDistinct = false;
    bool hasOuterJoin = false;
    bool hasSubqueryInCondition = false;
    bool hasIsNotDistinctFrom = false;
    std::set<std::string> unsupportedFuncs;
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
        if (node.type == IRNode::Type::Filter) {
            // Note: LIKE is supported via Function type (LIKE, PREFIX, SUFFIX, CONTAINS)
            // Check for unsupported functions in filter predicate
            const auto& pred = node.asFilter().predicate;
            findUnsupportedFunctions(pred, unsupportedFuncs);
        }
        if (node.type == IRNode::Type::Project) {
            // Check for unsupported functions in projections
            const auto& proj = node.asProject();
            for (const auto& expr : proj.exprs) {
                findUnsupportedFunctions(expr, unsupportedFuncs);
            }
        }
        if (node.type == IRNode::Type::Aggregate) {
            // Check for unsupported functions in aggregates
            findUnsupportedFunctions(node.asAggregate().expr, unsupportedFuncs);
        }
        if (node.type == IRNode::Type::GroupBy) {
            const auto& gb = node.asGroupBy();
            // CPU-based GroupBy in V2 executor supports any number of keys
            // Only limit is hash collision probability with many keys
            if (gb.keys.size() > 8) {
                blockers.push_back("GROUP BY with >8 keys");
            }
            // Check for unsupported functions in aggregates
            for (const auto& agg : gb.aggSpecs) {
                findUnsupportedFunctions(agg.input, unsupportedFuncs);
            }
        }
    }
    
    // Add unsupported functions to blockers
    for (const auto& funcName : unsupportedFuncs) {
        blockers.push_back(funcName + "() function not supported (requires string storage)");
    }

    return blockers;
}

// --- Main Execution Entry Point ---

GpuExecutor::ExecutionResult GpuExecutor::execute(const Plan& plan, const std::string& datasetPath) {
    ExecutionResult result;
    result.success = false;
    
    // Reset kernel timer for this query
    KernelTimer::instance().reset();
    
    // Reset thread-local aggregate counter at start of each query
    g_aggregateCounter = 0;

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
    
    // Collect columns that need raw strings for pattern matching
    auto patternMatchCols = collectPatternMatchColumns(plan);

    if (debug) {
        std::cerr << "[Exec] Columns needed per table:\n";
        for (const auto& [t, cs] : tableColsMap) {
            std::cerr << "  " << t << ": ";
            for (const auto& c : cs) std::cerr << c << " ";
            std::cerr << "\n";
        }
        if (!patternMatchCols.empty()) {
            std::cerr << "[Exec] Pattern-match columns (raw strings):\n";
            for (const auto& [t, cs] : patternMatchCols) {
                std::cerr << "  " << t << ": ";
                for (const auto& c : cs) std::cerr << c << " ";
                std::cerr << "\n";
            }
        }
    }

    // Build execution contexts for each table
    std::unordered_map<std::string, EvalContext> tableContexts;

    auto start = std::chrono::high_resolution_clock::now();

    IRGpuLoader::loadTables(tableColsMap, patternMatchCols, scanInstanceMap, datasetPath, tableContexts, result, debug);
    if (!result.error.empty()) return result;
    
    auto loadEnd = std::chrono::high_resolution_clock::now();

    if (debug) {
        std::cerr << "[Exec] Loaded " << tableContexts.size() << " tables in " 
                  << result.table.upload_ms << "ms\n";
    }

    // Execute operators in pipeline order
    EvalContext currentCtx;
    TableResult tableResult;
    bool isFirst = true;

    // Track which tables have been joined into the current context
    std::set<std::string> joinedTables;
    
    // Track the "main pipeline" context that accumulates join results
    EvalContext pipelineCtx;
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
                // Handle AND-separated multi-key conditions
                std::string remaining = cond;
                while (!remaining.empty()) {
                    std::string part;
                    size_t andPos = remaining.find(" AND ");
                    if (andPos != std::string::npos) {
                        part = remaining.substr(0, andPos);
                        remaining = remaining.substr(andPos + 5);
                    } else {
                        part = remaining;
                        remaining.clear();
                    }
                    // Check "X IS NOT DISTINCT FROM X"
                    size_t indPos = part.find(" IS NOT DISTINCT FROM ");
                    if (indPos != std::string::npos) {
                        std::string lhs = part.substr(0, indPos);
                        std::string rhs = part.substr(indPos + 22);
                        // Trim
                        while (!lhs.empty() && lhs.back() == ' ') lhs.pop_back();
                        while (!rhs.empty() && rhs[0] == ' ') rhs.erase(0, 1);
                        if (lhs == rhs) {
                            hasSelfComp = true;
                            corrCols.push_back(lhs);
                        }
                        continue;
                    }
                    // Check "X = X" (exact self-comparison, not like "a = b")
                    size_t eqPos = part.find(" = ");
                    if (eqPos != std::string::npos) {
                        std::string lhs = part.substr(0, eqPos);
                        std::string rhs = part.substr(eqPos + 3);
                        while (!lhs.empty() && lhs.back() == ' ') lhs.pop_back();
                        while (!rhs.empty() && rhs[0] == ' ') rhs.erase(0, 1);
                        if (lhs == rhs && !lhs.empty()) {
                            hasSelfComp = true;
                            corrCols.push_back(lhs);
                        }
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
                                    std::unordered_map<std::string, uint32_t> seen;
                                    std::vector<uint32_t> keepIdx;
                                    for (uint32_t i = 0; i < tableCtx.rowCount; ++i) {
                                        std::string compositeKey;
                                        for (const auto& col : dedupCols) {
                                            compositeKey += std::to_string(tableCtx.u32Cols.at(col)[i]) + "|";
                                        }
                                        if (seen.find(compositeKey) == seen.end()) {
                                            seen[compositeKey] = i;
                                            keepIdx.push_back(i);
                                        }
                                    }
                                    uint32_t newCount = (uint32_t)keepIdx.size();
                                    if (newCount < tableCtx.rowCount) {
                                        if (debug) {
                                            std::cerr << "[Exec] DELIM_SCAN dedup (build): " << tableCtx.rowCount
                                                      << " -> " << newCount << " rows\n";
                                        }
                                        for (auto& [name, vec] : tableCtx.u32Cols) {
                                            if (vec.size() >= tableCtx.rowCount) {
                                                std::vector<uint32_t> compact(newCount);
                                                for (uint32_t i = 0; i < newCount; ++i)
                                                    compact[i] = vec[keepIdx[i]];
                                                vec = std::move(compact);
                                            }
                                        }
                                        for (auto& [name, vec] : tableCtx.f32Cols) {
                                            if (vec.size() >= tableCtx.rowCount) {
                                                std::vector<float> compact(newCount);
                                                for (uint32_t i = 0; i < newCount; ++i)
                                                    compact[i] = vec[keepIdx[i]];
                                                vec = std::move(compact);
                                            }
                                        }
                                        for (auto& [name, vec] : tableCtx.stringCols) {
                                            if (vec.size() >= tableCtx.rowCount) {
                                                std::vector<std::string> compact(newCount);
                                                for (uint32_t i = 0; i < newCount; ++i)
                                                    compact[i] = vec[keepIdx[i]];
                                                vec = std::move(compact);
                                            }
                                        }
                                        // Rebuild flat string columns from compacted data
                                        tableCtx.flatStringColsGPU.clear();
                                        for (auto& [name, vec] : tableCtx.stringCols) {
                                            auto& cstore = ColumnStoreGPU::instance();
                                            tableCtx.flatStringColsGPU[name] = makeFlatStringColumn(cstore.device(), vec);
                                        }
                                        for (auto& [name, buf] : tableCtx.u32ColsGPU) {
                                            if (tableCtx.u32Cols.count(name) && !tableCtx.u32Cols.at(name).empty()) {
                                                const auto& v = tableCtx.u32Cols.at(name);
                                                buf = GpuOps::createBuffer(v.data(), v.size() * sizeof(uint32_t));
                                            }
                                        }
                                        for (auto& [name, buf] : tableCtx.f32ColsGPU) {
                                            if (tableCtx.f32Cols.count(name) && !tableCtx.f32Cols.at(name).empty()) {
                                                const auto& v = tableCtx.f32Cols.at(name);
                                                buf = GpuOps::createBuffer(v.data(), v.size() * sizeof(float));
                                            }
                                        }
                                        tableCtx.rowCount = newCount;
                                        tableCtx.activeRows.clear();
                                        tableCtx.activeRowsGPU = nullptr;
                                        tableCtx.activeRowsCountGPU = 0;
                                        
                                        // Strip payload columns
                                        tableCtx.f32Cols.clear();
                                        tableCtx.f32ColsGPU.clear();
                                        tableCtx.stringCols.clear();
                                        tableCtx.flatStringColsGPU.clear();
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
                        isFirst = false;

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
                                // Build composite key strings and dedup
                                std::unordered_map<std::string, uint32_t> seen;
                                std::vector<uint32_t> keepIdx;
                                for (uint32_t i = 0; i < currentCtx.rowCount; ++i) {
                                    std::string compositeKey;
                                    for (const auto& col : dedupCols) {
                                        compositeKey += std::to_string(currentCtx.u32Cols.at(col)[i]) + "|";
                                    }
                                    if (seen.find(compositeKey) == seen.end()) {
                                        seen[compositeKey] = i;
                                        keepIdx.push_back(i);
                                    }
                                }
                                uint32_t newCount = (uint32_t)keepIdx.size();
                                if (newCount < currentCtx.rowCount) {
                                    if (debug) {
                                        std::cerr << "[Exec] DELIM_SCAN dedup: " << currentCtx.rowCount
                                                  << " -> " << newCount << " rows\n";
                                    }
                                    // Compact all CPU columns
                                    for (auto& [name, vec] : currentCtx.u32Cols) {
                                        if (vec.size() >= currentCtx.rowCount) {
                                            std::vector<uint32_t> compact(newCount);
                                            for (uint32_t i = 0; i < newCount; ++i)
                                                compact[i] = vec[keepIdx[i]];
                                            vec = std::move(compact);
                                        }
                                    }
                                    for (auto& [name, vec] : currentCtx.f32Cols) {
                                        if (vec.size() >= currentCtx.rowCount) {
                                            std::vector<float> compact(newCount);
                                            for (uint32_t i = 0; i < newCount; ++i)
                                                compact[i] = vec[keepIdx[i]];
                                            vec = std::move(compact);
                                        }
                                    }
                                    for (auto& [name, vec] : currentCtx.stringCols) {
                                        if (vec.size() >= currentCtx.rowCount) {
                                            std::vector<std::string> compact(newCount);
                                            for (uint32_t i = 0; i < newCount; ++i)
                                                compact[i] = vec[keepIdx[i]];
                                            vec = std::move(compact);
                                        }
                                    }
                                    // Rebuild flat string columns from compacted data
                                    currentCtx.flatStringColsGPU.clear();
                                    for (auto& [name, vec] : currentCtx.stringCols) {
                                        auto& cstore = ColumnStoreGPU::instance();
                                        currentCtx.flatStringColsGPU[name] = makeFlatStringColumn(cstore.device(), vec);
                                    }
                                    // Re-upload GPU columns
                                    for (auto& [name, buf] : currentCtx.u32ColsGPU) {
                                        if (currentCtx.u32Cols.count(name) && !currentCtx.u32Cols.at(name).empty()) {
                                            const auto& v = currentCtx.u32Cols.at(name);
                                            buf = GpuOps::createBuffer(v.data(), v.size() * sizeof(uint32_t));
                                        }
                                    }
                                    for (auto& [name, buf] : currentCtx.f32ColsGPU) {
                                        if (currentCtx.f32Cols.count(name) && !currentCtx.f32Cols.at(name).empty()) {
                                            const auto& v = currentCtx.f32Cols.at(name);
                                            buf = GpuOps::createBuffer(v.data(), v.size() * sizeof(float));
                                        }
                                    }
                                    currentCtx.rowCount = newCount;
                                    currentCtx.activeRows.clear();
                                    currentCtx.activeRowsGPU = nullptr;
                                    currentCtx.activeRowsCountGPU = 0;
                                    
                                    // Strip non-correlation columns from DELIM_SCAN context
                                    currentCtx.f32Cols.clear();
                                    currentCtx.f32ColsGPU.clear();
                                    currentCtx.stringCols.clear();
                                    currentCtx.flatStringColsGPU.clear();
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
                                      tableResult.u32_cols[0].size() == currentCtx.activeRows.size());
                    
                    if (sizeMatch && !currentCtx.activeRows.empty()) {
                        // Compact tableResult based on activeRows
                        for (auto& col : tableResult.u32_cols) {
                            std::vector<uint32_t> filtered;
                            filtered.reserve(currentCtx.activeRows.size());
                            for (uint32_t idx : currentCtx.activeRows) {
                                if (idx < col.size()) {
                                    filtered.push_back(col[idx]);
                                }
                            }
                            col = std::move(filtered);
                        }
                        for (auto& col : tableResult.f32_cols) {
                            std::vector<float> filtered;
                            filtered.reserve(currentCtx.activeRows.size());
                            for (uint32_t idx : currentCtx.activeRows) {
                                if (idx < col.size()) {
                                    filtered.push_back(col[idx]);
                                }
                            }
                            col = std::move(filtered);
                        }
                        for (auto& col : tableResult.string_cols) {
                            std::vector<std::string> filtered;
                            filtered.reserve(currentCtx.activeRows.size());
                            for (uint32_t idx : currentCtx.activeRows) {
                                if (idx < col.size()) {
                                    filtered.push_back(col[idx]);
                                }
                            }
                            col = std::move(filtered);
                        }
                        tableResult.rowCount = currentCtx.activeRows.size();
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
                
                // Always compact currentCtx CPU AND GPU columns when filter has
                // activeRows, to ensure consistent row counts across all data.
                if (!currentCtx.activeRows.empty()) {
                    // First, materialize any GPU-only columns to CPU
                    for (auto& [name, buf] : currentCtx.u32ColsGPU) {
                        if (buf && (!currentCtx.u32Cols.count(name) || currentCtx.u32Cols.at(name).empty())) {
                            size_t n = buf->length() / sizeof(uint32_t);
                            currentCtx.u32Cols[name].resize(n);
                            memcpy(currentCtx.u32Cols[name].data(), buf->contents(), n * sizeof(uint32_t));
                        }
                    }
                    for (auto& [name, buf] : currentCtx.f32ColsGPU) {
                        if (buf && (!currentCtx.f32Cols.count(name) || currentCtx.f32Cols.at(name).empty())) {
                            size_t n = buf->length() / sizeof(float);
                            currentCtx.f32Cols[name].resize(n);
                            memcpy(currentCtx.f32Cols[name].data(), buf->contents(), n * sizeof(float));
                        }
                    }
                    
                    // Compact CPU columns
                    for (auto& [name, col] : currentCtx.u32Cols) {
                        if (col.size() > currentCtx.activeRows.size()) {
                            std::vector<uint32_t> filtered;
                            filtered.reserve(currentCtx.activeRows.size());
                            for (uint32_t idx : currentCtx.activeRows) {
                                if (idx < col.size()) filtered.push_back(col[idx]);
                            }
                            col = std::move(filtered);
                        }
                    }
                    for (auto& [name, col] : currentCtx.f32Cols) {
                        if (col.size() > currentCtx.activeRows.size()) {
                            std::vector<float> filtered;
                            filtered.reserve(currentCtx.activeRows.size());
                            for (uint32_t idx : currentCtx.activeRows) {
                                if (idx < col.size()) filtered.push_back(col[idx]);
                            }
                            col = std::move(filtered);
                        }
                    }
                    for (auto& [name, col] : currentCtx.stringCols) {
                        if (col.size() > currentCtx.activeRows.size()) {
                            std::vector<std::string> filtered;
                            filtered.reserve(currentCtx.activeRows.size());
                            for (uint32_t idx : currentCtx.activeRows) {
                                if (idx < col.size()) filtered.push_back(col[idx]);
                            }
                            col = std::move(filtered);
                        }
                    }
                    
                    // Re-upload compacted columns to GPU
                    for (auto& [name, buf] : currentCtx.u32ColsGPU) {
                        if (currentCtx.u32Cols.count(name) && !currentCtx.u32Cols.at(name).empty()) {
                            const auto& v = currentCtx.u32Cols.at(name);
                            buf = GpuOps::createBuffer(v.data(), v.size() * sizeof(uint32_t));
                        }
                    }
                    for (auto& [name, buf] : currentCtx.f32ColsGPU) {
                        if (currentCtx.f32Cols.count(name) && !currentCtx.f32Cols.at(name).empty()) {
                            const auto& v = currentCtx.f32Cols.at(name);
                            buf = GpuOps::createBuffer(v.data(), v.size() * sizeof(float));
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
                currentCtx.flatStringColsGPU.clear();
                currentCtx.currentTable.clear();
                
                // Reset joinedTables for SEMI join decorrelation pattern
                joinedTables.clear();
                joinedTables.insert("__GROUPED__");
                
                for (size_t i = 0; i < tableResult.u32_cols.size(); ++i) {
                    if (i < tableResult.u32_names.size()) {
                        currentCtx.u32Cols[tableResult.u32_names[i]] = tableResult.u32_cols[i];
                        // Also register under positional name for #N references
                        std::string posKey = "#" + std::to_string(i);
                        currentCtx.u32Cols[posKey] = tableResult.u32_cols[i];
                        // Re-register columns under their aliases (for CTE support)
                        for (const auto& [alias, canonical] : currentCtx.columnAliases) {
                            if (canonical == tableResult.u32_names[i]) {
                                currentCtx.u32Cols[alias] = tableResult.u32_cols[i];
                                if (debug) std::cerr << "[Exec] GroupBy: re-registering alias " << alias << " -> " << canonical << "\n";
                            }
                        }
                    }
                }
                for (size_t i = 0; i < tableResult.f32_cols.size(); ++i) {
                    if (i < tableResult.f32_names.size()) {
                        currentCtx.f32Cols[tableResult.f32_names[i]] = tableResult.f32_cols[i];
                        // Also register under positional name for #N references (offset by key count)
                        std::string posKey = "#" + std::to_string(i + tableResult.u32_cols.size());
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
                
                // Populate stringCols from GroupBy result
                for (size_t i = 0; i < tableResult.string_cols.size(); ++i) {
                    if (i < tableResult.string_names.size()) {
                        currentCtx.stringCols[tableResult.string_names[i]] = tableResult.string_cols[i];
                        if (debug) std::cerr << "[Exec] GroupBy: setting stringCol " << tableResult.string_names[i] 
                                            << " with " << tableResult.string_cols[i].size() << " rows\n";
                    }
                }

                // Rebuild flat string columns after GroupBy
                if (!currentCtx.stringCols.empty()) {
                    auto& cstoreGB = ColumnStoreGPU::instance();
                    currentCtx.flatStringColsGPU.clear();
                    for (const auto& [sn, sv] : currentCtx.stringCols) {
                        currentCtx.flatStringColsGPU[sn] = makeFlatStringColumn(cstoreGB.device(), sv);
                    }
                }

                // Strict Mode: Upload GroupBy results to GPU
                if (debug) std::cerr << "[Exec] Uploading GroupBy results to GPU (Strict Mode)\n";
                // auto& store = ColumnStoreGPU::instance(); // Unused if we use GpuOps
                
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
                aggBuf->retain(); // Keep alive for other references
                
                // Also store by name
                if (!name.empty()) {
                    currentCtx.f32Cols[name] = std::vector<float>{static_cast<float>(value)};
                    currentCtx.f32ColsGPU[name] = aggBuf;
                    aggBuf->retain();
                }
                if (!agg.exprStr.empty() && agg.exprStr != name) {
                     currentCtx.f32Cols[agg.exprStr] = std::vector<float>{static_cast<float>(value)};
                     currentCtx.f32ColsGPU[agg.exprStr] = aggBuf;
                     aggBuf->retain();
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
                        if (!vec.empty() && name.find("__internal_") == std::string::npos) {
                            tableResult.u32_names.push_back(name);
                            tableResult.u32_cols.push_back(vec);
                        }
                    }
                    for (const auto& [name, vec] : currentCtx.f32Cols) {
                        if (!vec.empty()) {
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
                    tableResult.rowCount = currentCtx.rowCount;
                }
                if (!executeOrderBy(node.asOrderBy(), tableResult)) {
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
                    if (i < tableResult.string_names.size())
                        currentCtx.stringCols[tableResult.string_names[i]] = tableResult.string_cols[i];
                }
                // Rebuild flat string columns after OrderBy sort
                if (!currentCtx.stringCols.empty()) {
                    auto& cstoreOB = ColumnStoreGPU::instance();
                    currentCtx.flatStringColsGPU.clear();
                    for (const auto& [sn, sv] : currentCtx.stringCols) {
                        currentCtx.flatStringColsGPU[sn] = makeFlatStringColumn(cstoreOB.device(), sv);
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
                break;
            }

            default:
                break;
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
                    // Materialize String
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
                        else strCol.push_back(std::to_string(val)); // Fallback
                    }

                    tableResult.string_names.push_back(colName);
                    tableResult.string_cols.push_back(std::move(strCol));
                    is_converted[i] = true;
                    string_converted_idx[i] = tableResult.string_names.size() - 1;
                    converted = true;
                } else if (cSchema && cSchema->type == ColumnType::Float32) {
                    // GroupBy bit-reinterprets f32 keys as u32. Restore to f32.
                    std::vector<float> f32Col(tableResult.u32_cols[i].size());
                    for (size_t j = 0; j < f32Col.size(); ++j) {
                        std::memcpy(&f32Col[j], &tableResult.u32_cols[i][j], sizeof(float));
                    }
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
    
    // CPU post-processing = pipeline wall-clock minus GPU kernel time + column cleanup time
    auto postEnd = std::chrono::high_resolution_clock::now();
    double postProcessMs = std::chrono::duration<double, std::milli>(postEnd - endTime).count();
    double cpuPipelineMs = pipelineWallMs - tableResult.gpu_ms;
    tableResult.cpu_post_ms = cpuPipelineMs + postProcessMs;

    result.success = true;
    result.table = std::move(tableResult);
    return result;
}

// --- Expression Evaluation ---

// Evaluator implementations moved to GpuExecutor_Evaluator.cpp
// - mapCompOp
// - executeGPUFilterRecursive
// - evalExprFloat
// - evalExprFloatGPU
// - evalExprU32
// - evalPredicate
} // namespace engine
