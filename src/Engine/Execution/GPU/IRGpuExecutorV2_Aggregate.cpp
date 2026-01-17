#include "IRGpuExecutorV2.hpp"
#include "IRGpuExecutorV2_Priv.hpp"
#include "OperatorsGPU.hpp"
// #include "../CPU/Operators.hpp" // For helper functions if needed

#include <iostream>
#include <algorithm>
#include <cmath>
#include <limits>

namespace engine {

bool IRGpuExecutorV2::executeAggregate(const IRAggregateV2& agg, EvalContext& ctx,
                                        double& outValue, std::string& outName) {
    outName = agg.alias.empty() ? aggFuncName(agg.func) : agg.alias;
    
    // Try GPU Path first
    // Note: count must account for activeRows
    uint32_t count = (ctx.activeRowsGPU != nullptr) ? ctx.activeRowsCountGPU : ctx.rowCount;
    // Patch for Q15: sanity check active rows
    if (ctx.activeRowsGPU != nullptr && ctx.activeRowsCountGPU > ctx.rowCount) {
        if (env_truthy("GPUDB_DEBUG_OPS")) std::cerr << "[V2] CountStar: activeRowsCountGPU " << ctx.activeRowsCountGPU << " > rowCount " << ctx.rowCount << ", ignoring GPU selection\n";
        count = ctx.rowCount;
    }
    if (count == 0 && ctx.rowCount > 0 && !ctx.activeRowsGPU) count = ctx.rowCount;

    // Fix for Scalar Subqueries: If the context is already a scalar result (e.g. from a previous 
    // ungrouped aggregate that logic dictates is a single row), treat count as 1.
    // This allows count(*) to return 1, and sum(scalar) to return scalar.
    if (ctx.isScalarResult) {
        count = 1;
    }

    // COUNT(*) can be done without evaluating expression
    if (agg.func == AggFunc::CountStar) {
        if(env_truthy("GPUDB_DEBUG_OPS")) std::cerr << "[V2] CountStar: ctx.rowCount=" << ctx.rowCount << " count=" << count << "\n";
        outValue = static_cast<double>(count);
        return true;
    }

    MTL::Buffer* gpuInput = evalExprFloatGPU(agg.expr, ctx);
    if (gpuInput) {
        // Determine if we own this buffer (intermediate result) or if it's borrowed from context
        bool isOwned = true;
        for (const auto& [name, buf] : ctx.f32ColsGPU) {
            if (buf == gpuInput) { isOwned = false; break; }
        }
        if (isOwned) {
            for (const auto& [name, buf] : ctx.u32ColsGPU) {
                if (buf == gpuInput) { isOwned = false; break; }
            }
        }
        // Also check if it matches activeRowsGPU (unlikely but safe)
        if (ctx.activeRowsGPU == gpuInput) isOwned = false;

        bool success = true;
        if (agg.func == AggFunc::Sum) {
            outValue = (double)OperatorsGPU::reduceSumF32(gpuInput, count);
        } else if (agg.func == AggFunc::Min) {
            outValue = (double)OperatorsGPU::reduceMinF32(gpuInput, count);
        } else if (agg.func == AggFunc::Max) {
            outValue = (double)OperatorsGPU::reduceMaxF32(gpuInput, count);
        } else if (agg.func == AggFunc::Avg) {
            double sum = (double)OperatorsGPU::reduceSumF32(gpuInput, count);
            outValue = count > 0 ? sum / count : 0.0;
        } else if (agg.func == AggFunc::Count) {
             outValue = (double)count;
             // Treating as count(*), ignoring NULLs for now as simplified model
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
