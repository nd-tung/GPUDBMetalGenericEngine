#include "GpuExecutor.hpp"
#include "GpuExecutorDetail.hpp"
#include "Operators.hpp"
#include "TypedExpr.hpp"

#include <iostream>

namespace engine {

// Forward declaration — defined in GpuExecutor.cpp
uint32_t deduplicateContext(EvalContext& ctx,
                            const std::vector<std::string>& dedupCols,
                            bool debug);

bool GpuExecutor::executeDistinct(const IRDistinct& distinct, EvalContext& ctx) {
    if (ctx.rowCount <= 1) return true;

    bool debug = (std::getenv("GPUDB_DEBUG") != nullptr);

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
            std::cerr << "[Exec] Distinct: no columns to deduplicate, skipping\n";
        }
        return true;
    }

    if (debug) {
        std::cerr << "[Exec] Distinct: dedup on " << dedupCols.size() << " cols:";
        for (const auto& c : dedupCols) std::cerr << " " << c;
        std::cerr << "\n";
    }

    uint32_t newCount = deduplicateContext(ctx, dedupCols, debug);
    if (newCount == 0) {
        // No duplicates found — all rows already unique
        if (debug) {
            std::cerr << "[Exec] Distinct: all " << ctx.rowCount << " rows already unique\n";
        }
    }

    return true;
}

} // namespace engine
