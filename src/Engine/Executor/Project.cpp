#include "GpuExecutor.hpp"
#include "GpuExecutorDetail.hpp"
#include "Operators.hpp"
#include "GpuColumnStore.hpp"
#include "EngineError.hpp"
#include "EngineConfig.hpp"

#include <iostream>
#include <vector>
#include <set>
#include <cstring>
#include <algorithm>
#include <cctype>
#include <functional>
#include <map>
#include "Logger.hpp"

namespace engine {

// Forward declarations (implemented in ProjectHelpers.cpp)
bool projectSubstring(const FunctionCall& func,
    EvalContext& ctx, TableResult& out, const std::string& outName,
    const std::string& posName, bool debug);
bool projectExtractYear(const FunctionCall& func,
    const std::string& funcLower,
    EvalContext& ctx, TableResult& out, const std::string& outName,
    const std::string& posName, bool debug);
bool projectCrossTableLookup(const std::string& lookupCol,
    EvalContext& ctx, TableResult& out, const std::string& outName,
    const std::string& posName, size_t& projectedRowCount, bool& rowCountInitialized,
    std::unordered_map<std::string, EvalContext>* tableContexts, bool debug);
bool projectComputedExpression(
    const TypedExprPtr& expr,
    size_t exprIndex,
    const std::string& outName,
    const std::string& posName,
    EvalContext& ctx,
    TableResult& out,
    const std::function<void(size_t)>& updateRowCount,
    bool debug);

// Forward declarations (implemented in ProjectColumns.cpp)
std::string fuzzyFindColumn(const std::string& name, const EvalContext& ctx, bool debug);
bool projectStringColumn(const std::string& col, const std::string& outName,
    const std::string& posName, EvalContext& ctx, TableResult& out,
    size_t& projectedRowCount, bool& rowCountInitialized, bool debug);
std::string resolveProjectColumn(const std::string& col, const std::string& outName,
    EvalContext& ctx, const std::set<std::string>& usedColumns, bool debug);
bool projectU32Column(const std::string& col, const std::string& lookupCol,
    const std::string& outName, const std::string& posName,
    EvalContext& ctx, TableResult& out, std::set<std::string>& usedColumns,
    size_t& projectedRowCount, bool& rowCountInitialized, bool debug);
bool projectF32Column(const std::string& col, const std::string& lookupCol,
    const std::string& outName, const std::string& posName,
    EvalContext& ctx, TableResult& out, std::set<std::string>& usedColumns,
    size_t& projectedRowCount, bool& rowCountInitialized, bool debug);

// ── Copy prior output columns into context for chained projections ──
// When a Project follows another Project (no table context), columns from
// the prior output need to be available in the EvalContext.
static void copyPriorOutputToCtx(TableResult& out, EvalContext& ctx,
                                  bool hasExistingOutput, bool /*debug*/) {
    if (!hasExistingOutput) return;

    bool shouldCopy = ctx.currentTable.empty();
    if (shouldCopy) {
        for (size_t i = 0; i < out.u32Names.size() && i < out.u32Cols.size(); ++i) {
            if (ctx.u32Cols.find(out.u32Names[i]) == ctx.u32Cols.end())
                ctx.u32Cols[out.u32Names[i]] = out.u32Cols[i];
        }
        for (size_t i = 0; i < out.f32Names.size() && i < out.f32Cols.size(); ++i) {
            if (ctx.f32Cols.find(out.f32Names[i]) == ctx.f32Cols.end())
                ctx.f32Cols[out.f32Names[i]] = out.f32Cols[i];
        }
        for (size_t i = 0; i < out.stringNames.size() && i < out.stringCols.size(); ++i) {
            if (ctx.stringCols.find(out.stringNames[i]) == ctx.stringCols.end())
                ctx.stringCols[out.stringNames[i]] = out.stringCols[i];
        }
    }
    out.u32Cols.clear();  out.u32Names.clear();
    out.f32Cols.clear();  out.f32Names.clear();
    out.stringCols.clear(); out.stringNames.clear();
    out.order.clear();
}

bool GpuExecutor::executeProject(const IRProject& project, EvalContext& ctx, TableResult& out, std::unordered_map<std::string, EvalContext>* tableContexts) {
    const bool debug = env_truthy("GPUDB_DEBUG_OPS");
    
    if (debug) {
        LOG_INFO("Exec", "Project START: currentTable=" << ctx.currentTable << " ctx.u32Cols=");
        for (const auto& [k, v] : ctx.u32Cols) std::cerr << k << " ";
        LOG_INFO("PROJECT", "\n");
    }

    const size_t originalRowCount = ctx.rowCount;
    size_t projectedRowCount = ctx.rowCount;
    bool rowCountInitialized = false;
    auto updateRowCount = [&](size_t size) {
        if (!rowCountInitialized) {
            projectedRowCount = size;
            rowCountInitialized = true;
        } else if (size > 0 && projectedRowCount != size) {
            // Prefer the new size when encountering differing column lengths (e.g., scalar aggregates)
            if (size > projectedRowCount) projectedRowCount = size;
        }
    };
    
    bool hasExistingOutput = !out.u32Cols.empty() || !out.f32Cols.empty();
    
    if (debug && hasExistingOutput) {
        LOG_INFO("Exec", "Project: hasExistingOutput=true, out.u32Names=");
        for (const auto& n : out.u32Names) std::cerr << n << " ";
        LOG_INFO("PROJECT", "\n");
    }
    
    // Copy prior output columns into context (for chained projections)
    copyPriorOutputToCtx(out, ctx, hasExistingOutput, debug);
    
    std::set<std::string> usedColumns;
    
    for (size_t i = 0; i < project.exprs.size(); ++i) {
        const auto& expr = project.exprs[i];
        std::string outName = i < project.outputNames.size() ? project.outputNames[i] : "";
        
        // Generate a name for positional reference (#N)
        std::string posName = "#" + std::to_string(i);
        
        if (debug) {
            LOG_INFO("Exec", "Project: expr[" << i << "] outName=" << outName);
            if (expr) {
                LOG_INFO("PROJECT", " kind=" << static_cast<int>(expr->kind));
                if (expr->kind == TypedExpr::Kind::Column) {
                    LOG_DEBUG("PROJECT", " col=" << expr->asColumn().column);
                }
            }
            LOG_DEBUG("PROJECT", "\n");
        }
        
        if (!expr) continue;
        
        // Handle DuckDB internal functions like __internal_decompress_string(#0)
        // These are essentially passthrough - extract the inner column
        // BUT NOT for computation functions like EXTRACT or SUBSTRING
        if (expr->kind == TypedExpr::Kind::Function) {
            const auto& func = expr->asFunction();
            std::string funcLower = func.name;
            std::transform(funcLower.begin(), funcLower.end(), funcLower.begin(), ::tolower);
            
            if (debug) {
                LOG_INFO("Exec", "Project: function '" << func.name << "' (lower: '" << funcLower  << "') args=" << func.args.size());
            }
            
            // Handle SUBSTRING/SUBSTR as a string computation function
            if ((funcLower == "substring" || funcLower == "substr") && func.args.size() >= 1) {
                if (projectSubstring(func, ctx, out, outName, posName, debug)) continue;
            }
            
            if (projectExtractYear(func, funcLower, ctx, out, outName, posName, debug)) continue;

            // Skip passthrough for computation functions that need actual evaluation
            bool isComputation = (funcLower == "extract" || funcLower == "year" ||
                                  funcLower == "month" || funcLower == "day" ||
                                  funcLower == "substring" || funcLower == "substr");
            
            // One-arg column function -> treat as column reference (unless computation).
            if (!isComputation && func.args.size() == 1 && func.args[0] && 
                func.args[0]->kind == TypedExpr::Kind::Column) {
                std::string col = func.args[0]->asColumn().column;
                // Try to find the column or its equivalent (e.g., #0 -> first group key)
                auto itU = ctx.u32Cols.find(col);
                if (itU != ctx.u32Cols.end()) {
                    // Lazy-fetch from GPU if CPU vector is empty sentinel
                    if (itU->second.empty() && ctx.u32ColsGPU.count(col) && ctx.u32ColsGPU[col]) {
                        uint32_t rc = ctx.rowCount;
                        itU->second.resize(rc);
                        std::memcpy(itU->second.data(), ctx.u32ColsGPU[col]->contents(), rc * sizeof(uint32_t));
                    }
                    LOG_DEBUG("Exec", "Project: function passthrough " << col);
                    ctx.u32Cols[posName] = itU->second;
                    out.u32Cols.push_back(itU->second);
                    out.u32Names.push_back(outName.empty() ? col : outName);
                    continue;
                }
                // For #N positional references, look up directly (they should exist in context)
                if (col.size() >= 2 && col[0] == '#' && std::isdigit(static_cast<unsigned char>(col[1]))) {
                    auto itU2 = ctx.u32Cols.find(col);
                    if (itU2 != ctx.u32Cols.end()) {
                        LOG_DEBUG("Exec", "Project: function passthrough positional " << col);
                        ctx.u32Cols[posName] = itU2->second;
                        out.u32Cols.push_back(itU2->second);
                        out.u32Names.push_back(outName.empty() ? col : outName);
                        continue;
                    }
                    // Also try f32
                    auto itF = ctx.f32Cols.find(col);
                    if (itF != ctx.f32Cols.end()) {
                        LOG_DEBUG("Exec", "Project: function passthrough positional " << col << " (f32)\n");
                        ctx.f32Cols[posName] = itF->second;
                        out.f32Cols.push_back(itF->second);
                        out.f32Names.push_back(outName.empty() ? col : outName);
                        continue;
                    }
                }
            }
        }
        
        if (expr->kind == TypedExpr::Kind::Column) {
            // Simple column reference - copy to context with new name if needed
            std::string col = expr->asColumn().column;
            
            if (debug) {
                 LOG_INFO("Exec", "Project: Looking for col '" << col << "'\n");
            }
            
            // String column pass-through (alias/prefix/dict resolution + GPU/CPU compaction)
            if (projectStringColumn(col, outName, posName, ctx, out, projectedRowCount, rowCountInitialized, debug)) continue;

            
            // Handle post-GroupBy positional references: #N might be SUM_#N or COUNT_#N
            if (col.size() >= 2 && col[0] == '#') {
                // Try SUM_#N first for aggregate outputs
                std::string sumName = "SUM_" + col;
                auto itSum = ctx.f32Cols.find(sumName);
                if (itSum != ctx.f32Cols.end()) {
                    LOG_DEBUG("Exec", "Project: mapping " << col << " -> " << sumName);
                    ctx.f32Cols[posName] = itSum->second;
                    if (!outName.empty()) ctx.f32Cols[outName] = itSum->second;
                    out.f32Cols.push_back(itSum->second);
                    out.f32Names.push_back(outName.empty() ? col : outName);
                    continue;
                }
                // Try COUNT_#N
                std::string countName = "COUNT_" + col;
                auto itCount = ctx.f32Cols.find(countName);
                if (itCount != ctx.f32Cols.end()) {
                    LOG_DEBUG("Exec", "Project: mapping " << col << " -> " << countName);
                    ctx.f32Cols[posName] = itCount->second;
                    if (!outName.empty()) ctx.f32Cols[outName] = itCount->second;
                    out.f32Cols.push_back(itCount->second);
                    out.f32Names.push_back(outName.empty() ? col : outName);
                    continue;
                }
            }
            
            // Resolve column name through multi-instance, alias, CTE fallback, fuzzy matching
            std::string lookupCol = resolveProjectColumn(col, outName, ctx, usedColumns, debug);

            // U32 column download/compact/output
            if (projectU32Column(col, lookupCol, outName, posName, ctx, out, usedColumns, projectedRowCount, rowCountInitialized, debug)) continue;

            // F32 column download/compact/output
            if (projectF32Column(col, lookupCol, outName, posName, ctx, out, usedColumns, projectedRowCount, rowCountInitialized, debug)) continue;

            if (projectCrossTableLookup(lookupCol, ctx, out, outName, posName, projectedRowCount, rowCountInitialized, tableContexts, debug)) continue;
            
            // Column not found in context - might be an alias for aggregate output
            // Try to find aggregate column by positional key (#N) first, then by name pattern
            bool foundAggregate = false;
            
            // First: try positional match using the projection index (#0, #1, etc.)
            if (ctx.f32Cols.count(posName)) {
                LOG_DEBUG("Exec", "Project: mapping unknown alias '" << col << "' to positional aggregate " << posName);
                auto& aggData = ctx.f32Cols[posName];
                ctx.f32Cols[col] = aggData;
                if (!outName.empty()) ctx.f32Cols[outName] = aggData;
                out.f32Cols.push_back(aggData);
                out.f32Names.push_back(outName.empty() ? col : outName);
                foundAggregate = true;
            }
            
            // Fallback: try aggregate-prefixed keys matching the projection index
            if (!foundAggregate) {
                std::string idxSuffix = posName; // "#N"
                for (const auto& [aggName, aggData] : ctx.f32Cols) {
                    if ((aggName.find("COUNT_") == 0 || aggName.find("SUM_") == 0 || 
                         aggName.find("AVG_") == 0 || aggName.find("MIN_") == 0 || aggName.find("MAX_") == 0) &&
                        aggName.find(idxSuffix) != std::string::npos) {
                        LOG_DEBUG("Exec", "Project: mapping unknown alias '" << col << "' to aggregate " << aggName);
                        ctx.f32Cols[col] = aggData;
                        ctx.f32Cols[posName] = aggData;
                        if (!outName.empty()) ctx.f32Cols[outName] = aggData;
                        out.f32Cols.push_back(aggData);
                        out.f32Names.push_back(outName.empty() ? col : outName);
                        foundAggregate = true;
                        break;
                    }
                }
            }
            if (foundAggregate) continue;
        } else {
            if (projectComputedExpression(expr, i, outName, posName, ctx, out, updateRowCount, debug)) continue;
        }
    }
    
    if (rowCountInitialized) {
        out.rowCount = projectedRowCount;
        ctx.rowCount = projectedRowCount;
    } else {
        // Prefer GPU activeRows count if available
        size_t fallbackCount = originalRowCount;
        if (ctx.activeRowsGPU && ctx.activeRowsCountGPU > 0) {
            fallbackCount = ctx.activeRowsCountGPU;
        } else if (!ctx.activeRows.empty()) {
            fallbackCount = ctx.activeRows.size();
        }
        out.rowCount = fallbackCount;
        ctx.rowCount = fallbackCount;
    }

    // Populate GPU buffer mirrors in TableResult for downstream operators
    out.u32ColsGPU.resize(out.u32Names.size());
    for (size_t i = 0; i < out.u32Names.size(); ++i) {
        if (ctx.u32ColsGPU.count(out.u32Names[i])) {
            out.u32ColsGPU[i] = ctx.u32ColsGPU[out.u32Names[i]];
        }
    }
    out.f32ColsGPU.resize(out.f32Names.size());
    for (size_t i = 0; i < out.f32Names.size(); ++i) {
        if (ctx.f32ColsGPU.count(out.f32Names[i])) {
            out.f32ColsGPU[i] = ctx.f32ColsGPU[out.f32Names[i]];
        }
    }

    return true;
}

} // namespace engine
