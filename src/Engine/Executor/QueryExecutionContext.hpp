#pragma once
// QueryExecutionContext: bundles all mutable state threaded through node handlers.
// Replaces 6-8 separate parameters with a single reference, improving readability
// and making it easier to add new shared state without touching every handler signature.

#include "GpuExecutor.hpp"
#include "ResultTable.hpp"

#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

namespace engine {

struct EvalContext;
struct ScanInstance;

struct QueryExecutionContext {
    EvalContext& currentCtx;
    std::unordered_map<std::string, EvalContext>& tableContexts;
    TableResult& tableResult;
    GpuExecutor::JoinPipelineState& joinState;
    GpuExecutor::ExecutionResult& result;
    const Plan& plan;
    const std::map<size_t, ScanInstance>& scanInstanceMap;
    const std::unordered_map<std::string, std::vector<std::string>>& delimCorrelationCols;
    bool debug;
};

} // namespace engine
