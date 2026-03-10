#pragma once

#include <optional>
#include <string>
#include <vector>
#include <unordered_map>
#include <set>

#include "IR.hpp"
#include "TypedExpr.hpp"
#include "Schema.hpp"
#include "ResultTable.hpp"

namespace MTL { class Buffer; }

namespace engine {

struct EvalContext;
struct DictEncoded;
struct FlatStringCol;

// ============================================================================
// GpuExecutor: Generic GPU executor using V2 IR (no regex fallbacks)
// Uses TypedExpr trees for expression evaluation and SchemaRegistry for types
// ============================================================================

class GpuExecutor {
public:
    struct ExecutionResult {
        bool success = false;
        std::string error;
        TableResult table;
        
        // Scalar aggregate result (for queries like Q6)
        bool isScalarAggregate = false;
        double scalarValue = 0.0;
        std::string scalarName;
    };

    // Main entry point: execute a V2 plan and return results
    static ExecutionResult execute(const Plan& plan, const std::string& datasetPath);

    // Get list of unsupported features preventing GPU execution
    static std::vector<std::string> getUnsupportedFeatures(const Plan& plan);

    // Execute individual operators (public for use by extracted node-handler helpers)
    static bool executeFilter(const IRFilter& filter, EvalContext& ctx);
    static bool executeJoin(const IRJoin& join,
                            EvalContext& leftCtx, EvalContext& rightCtx, EvalContext& outCtx);
    static bool executeGroupBy(const IRGroupBy& groupBy, EvalContext& ctx, TableResult& out);
    static bool executeAggregate(const IRAggregate& agg, EvalContext& ctx, 
                                  double& outValue, std::string& outName);
    static bool executeOrderBy(const IROrderBy& order, TableResult& table,
                                const std::unordered_map<std::string, DictEncoded>& dictCols = {},
                                const std::unordered_map<std::string, FlatStringCol>& flatStringCols = {});
    static bool executeLimit(const IRLimit& limit, TableResult& table);
    static bool executeDistinct(const IRDistinct& distinct, EvalContext& ctx);
    static bool executeProject(const IRProject& project, EvalContext& ctx, TableResult& out, std::unordered_map<std::string, EvalContext>* tableContexts = nullptr);

    // Evaluate a TypedExpr tree into a GPU float buffer
    static MTL::Buffer* evaluateExpression(const TypedExprPtr& expr, EvalContext& ctx);
    
    // Recursively evaluate filter predicates on GPU
    static bool executeFilterRecursive(const TypedExprPtr& expr, EvalContext& ctx);

    // ========================================================================
    // JoinPipelineState: bundles mutable state threaded through join execution.
    // Avoids passing 5+ separate mutable references between executeJoinPipeline
    // and its helper functions.
    // ========================================================================
    struct JoinPipelineState {
        std::unordered_map<std::string, EvalContext>& tableContexts;
        std::vector<EvalContext>&                     savedPipelines;
        std::vector<std::set<std::string>>&           savedPipelineTables;
        std::set<std::string>&                        joinedTables;
        bool&                                         hasPipeline;
    };
    
    // Execute full join pipeline: resolve tables, handle scalar subqueries, and dispatch join
    static bool executeJoinPipeline(
        const IRJoin& join,
        EvalContext& currentCtx,
        JoinPipelineState& state,
        ExecutionResult& result
    );
};

} // namespace engine
