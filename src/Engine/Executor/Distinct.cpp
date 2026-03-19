#include "GpuExecutor.hpp"
#include "GpuExecutorDetail.hpp"
#include "Operators.hpp"
#include "TypedExpr.hpp"
#include "EnvUtil.hpp"

#include <iostream>
#include "Logger.hpp"

namespace engine {

bool GpuExecutor::executeDistinct(const IRDistinct& distinct, EvalContext& ctx) {
    if (ctx.rowCount <= 1) return true;

    bool debug = env_truthy("GPUDB_DEBUG_OPS");

    // Collect all column names to deduplicate on.
    // If IRDistinct has explicit expressions, use those column references.
    // Otherwise, deduplicate on ALL u32/f32 columns in the context.
    std::vector<std::string> dedupCols;

    for (const auto& expr : distinct.exprs) {
        if (expr && expr->kind == TypedExpr::Kind::Column) {
            const auto& colRef = std::get<ColumnRef>(expr->data);
            dedupCols.push_back(colRef.column);
        }
    }

    // If no explicit columns specified, deduplicate on all available columns
    if (dedupCols.empty()) {
        for (const auto& [name, buf] : ctx.u32ColsGPU) {
            if (buf && name.find("__internal_") == std::string::npos) {
                dedupCols.push_back(name);
            }
        }
        for (const auto& [name, col] : ctx.u32Cols) {
            if (!col.empty() && !ctx.u32ColsGPU.count(name) &&
                name.find("__internal_") == std::string::npos) {
                dedupCols.push_back(name);
            }
        }
    }

    if (dedupCols.empty()) {
        if (debug) {
            LOG_INFO("Exec", "Distinct: no columns to deduplicate, skipping\n");
        }
        return true;
    }

    if (debug) {
        LOG_INFO("Exec", "Distinct: dedup on " << dedupCols.size() << " cols:");
        for (const auto& c : dedupCols) std::cerr << " " << c;
        LOG_INFO("DISTINCT", "\n");
    }

    uint32_t newCount = deduplicateContext(ctx, dedupCols, debug);
    if (newCount == 0) {
        // No duplicates found — all rows already unique
        if (debug) {
            LOG_INFO("Exec", "Distinct: all " << ctx.rowCount << " rows already unique\n");
        }
    }

    return true;
}

} // namespace engine
