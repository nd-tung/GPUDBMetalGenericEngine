#include "GpuExecutor.hpp"
#include "GpuExecutorPriv.hpp"
#include "Operators.hpp"

#include <iostream>
#include <algorithm>
#include <cmath>
#include <limits>

namespace engine {

bool GpuExecutor::executeAggregate(const IRAggregate& agg, EvalContext& ctx,
                                        double& outValue, std::string& outName) {
    outName = agg.alias.empty() ? aggFuncName(agg.func) : agg.alias;
    
    // Account for active rows
    uint32_t count = (ctx.activeRowsGPU != nullptr) ? ctx.activeRowsCountGPU : ctx.rowCount;
    // Clamp if activeRowsCount exceeds rowCount
    if (ctx.activeRowsGPU != nullptr && ctx.activeRowsCountGPU > ctx.rowCount) {
        if (env_truthy("GPUDB_DEBUG_OPS")) std::cerr << "[Exec] CountStar: activeRowsCountGPU " << ctx.activeRowsCountGPU << " > rowCount " << ctx.rowCount << ", ignoring GPU selection\n";
        count = ctx.rowCount;
    }
    if (count == 0 && ctx.rowCount > 0 && !ctx.activeRowsGPU) count = ctx.rowCount;

    // Scalar context: treat as single row
    if (ctx.isScalarResult) {
        count = 1;
    }

    // COUNT(*) can be done without evaluating expression
    if (agg.func == AggFunc::CountStar) {
        if(env_truthy("GPUDB_DEBUG_OPS")) std::cerr << "[Exec] CountStar: ctx.rowCount=" << ctx.rowCount << " count=" << count << "\n";
        outValue = static_cast<double>(count);
        return true;
    }

    // Handle 0-row context: return identity values for aggregates
    if (count == 0) {
        if (agg.func == AggFunc::Sum || agg.func == AggFunc::Avg || agg.func == AggFunc::Count) {
            outValue = 0.0;
            return true;
        } else if (agg.func == AggFunc::Min) {
            outValue = std::numeric_limits<double>::infinity();
            return true;
        } else if (agg.func == AggFunc::Max) {
            outValue = -std::numeric_limits<double>::infinity();
            return true;
        }
    }

    MTL::Buffer* gpuInput = evalExprFloatGPU(agg.expr, ctx);
    if (gpuInput) {
        // Check if buffer is an intermediate (owned) vs borrowed from context
        bool isOwned = true;
        for (const auto& [name, buf] : ctx.f32ColsGPU) {
            if (buf == gpuInput) { isOwned = false; break; }
        }
        if (isOwned) {
            for (const auto& [name, buf] : ctx.u32ColsGPU) {
                if (buf == gpuInput) { isOwned = false; break; }
            }
        }
        // Also check against activeRowsGPU
        if (ctx.activeRowsGPU == gpuInput) isOwned = false;

        bool success = true;
        if (agg.func == AggFunc::Sum) {
            outValue = (double)GpuOps::reduceSumF32(gpuInput, count);
        } else if (agg.func == AggFunc::Min) {
            outValue = (double)GpuOps::reduceMinF32(gpuInput, count);
        } else if (agg.func == AggFunc::Max) {
            outValue = (double)GpuOps::reduceMaxF32(gpuInput, count);
        } else if (agg.func == AggFunc::Avg) {
            double sum = (double)GpuOps::reduceSumF32(gpuInput, count);
            outValue = count > 0 ? sum / count : 0.0;
        } else if (agg.func == AggFunc::Count) {
             outValue = (double)count;
             // Treated as count(*); NULL handling not implemented
        } else if (agg.func == AggFunc::First) {
             if (count > 0) {
                 float* ptr = (float*)gpuInput->contents();
                 outValue = (double)ptr[0];
             } else {
                 outValue = 0.0;
             }
        } else {
             success = false;
        }

        if (isOwned) gpuInput->release();
        
        if (success) return true;
    }

    throw std::runtime_error("GPU Aggregate failed: operation not supported on GPU (CPU fallback disabled).");
}

} // namespace engine
