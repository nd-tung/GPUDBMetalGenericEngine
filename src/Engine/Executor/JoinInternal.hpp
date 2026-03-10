// ============================================================================
// JoinInternal.hpp — Shared declarations for the Join split files
// ============================================================================
#pragma once

#include "GpuExecutor.hpp"
#include "GpuExecutorDetail.hpp"
#include "GpuColumnStore.hpp"
#include "EngineError.hpp"
#include <Metal/Metal.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <map>
#include <numeric>

namespace engine {

// ---- JoinUtils functions ----
MTL::Buffer* ensureColumnOnGPU(EvalContext& ctx, const std::string& col, bool debug);
std::string findColWithSuffix(EvalContext& ctx, const std::string& col);
std::string fuzzyResolveColumn(EvalContext& ctx, const std::string& colName,
                                const std::unordered_set<std::string>& excludeCols = {});
bool hasColumnOrSuffixed(const EvalContext& ctx, const std::string& colName);

// ---- JoinOutput functions ----
void appendUnmatchedLeftRows(EvalContext& leftCtx, EvalContext& outCtx,
    const JoinResult& jRes, uint32_t& resCount, uint32_t lCount,
    const std::unordered_map<std::string, std::string>& rightColumnMapping, bool debug);
void appendUnmatchedRightRows(EvalContext& rightCtx, EvalContext& outCtx,
    const JoinResult& jRes, uint32_t& resCount, uint32_t rCount,
    const std::unordered_map<std::string, std::string>& rightColumnMapping, bool debug);
bool scatterJoinOutputColumns(EvalContext& leftCtx, EvalContext& rightCtx,
    EvalContext& outCtx, const JoinResult& jRes, uint32_t resCount,
    uint32_t lCount, uint32_t rCount, bool isAntiJoin, bool isSemiJoin,
    bool rightAntiGather, std::unordered_map<std::string, std::string>& rightColumnMappingOut,
    bool debug);

// ---- JoinScalarSubquery functions ----
bool handleScalarSubquerySavedPipelines(const IRJoin& join, EvalContext& currentCtx,
    std::vector<EvalContext>& savedPipelines,
    std::vector<std::set<std::string>>& savedPipelineTables,
    std::set<std::string>& joinedTables,
    GpuExecutor::ExecutionResult& result, bool debug);
bool handleScalarSubqueryTableContexts(const IRJoin& join, EvalContext& currentCtx,
    std::unordered_map<std::string, EvalContext>& tableContexts,
    std::set<std::string>& joinedTables, bool& hasPipeline,
    GpuExecutor::ExecutionResult& result, bool debug);
bool applyScalarSubqueryCrossJoinFilter(const std::set<std::string>& condCols,
    const IRJoin& join, EvalContext& currentCtx,
    std::unordered_map<std::string, EvalContext>& tableContexts,
    std::vector<EvalContext>& savedPipelines,
    const std::vector<std::set<std::string>>& savedPipelineTables, bool debug);

// ---- JoinPipeline helper functions ----
bool detectTrivialSelfJoin(const IRJoin& join, const EvalContext& currentCtx,
    const std::set<std::string>& condCols, const std::set<std::string>& joinedTables, bool debug);
std::string inferRightTableForJoin(const IRJoin& join, const std::set<std::string>& condCols,
    const EvalContext& currentCtx,
    const std::unordered_map<std::string, EvalContext>& tableContexts,
    const std::set<std::string>& joinedTables, bool debug);
void dedupDelimJoinRHS(const IRJoin& join, EvalContext& currentCtx, EvalContext& rightCtx, bool debug);

} // namespace engine
