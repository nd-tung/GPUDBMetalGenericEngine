#include "GpuExecutor.hpp"
#include "GpuExecutorDetail.hpp"
#include "Operators.hpp"
#include "GpuColumnStore.hpp"
#include "KernelTimer.hpp"
#include "EngineError.hpp"
#include "Schema.hpp"

#include <algorithm>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include "Logger.hpp"

namespace engine {

// Forward declarations (implemented in GroupByKeys.cpp)
void postProcessStringKeys(
    const std::vector<std::vector<uint32_t>>& keyVecs,
    const std::vector<std::vector<std::string>>& outputStringMaps,
    const std::vector<std::unordered_map<uint32_t, std::string>>& hashToStringMaps,
    const std::vector<std::shared_ptr<std::vector<std::string>>>& dictRefMaps,
    TableResult& out, bool debug);
void restoreF32Keys(
    const std::vector<bool>& keyFromF32,
    TableResult& out, bool debug);
void buildGroupByOutputOrder(
    const std::vector<std::vector<uint32_t>>& keyVecs,
    const std::vector<bool>& keyFromF32,
    const std::vector<std::vector<std::string>>& outputStringMaps,
    const std::vector<std::unordered_map<uint32_t, std::string>>& hashToStringMaps,
    const std::vector<std::shared_ptr<std::vector<std::string>>>& dictRefMaps,
    TableResult& out);
GroupByKeyData buildGroupByKeys(
    const IRGroupBy& groupBy, EvalContext& ctx,
    size_t expectedKeyRows, bool debug);

struct GroupByAggData {
    std::vector<std::vector<float>> aggInputs;
    std::vector<MTL::Buffer*> aggBufsGPU;
    std::vector<AggFunc> aggFuncs;
    std::vector<std::string> aggNames;
};

// -- Extracted: handleCountDistinct --
// 2-stage GPU: stage1 groups by {keys+distinct_col}, stage2 COUNT(*).
static bool handleCountDistinct(const IRGroupBy& groupBy, EvalContext& ctx, TableResult& out,
                                const std::vector<AggFunc>& aggFuncs, int countDistinctIdx, bool /*debug*/) {
    LOG_DEBUG("Exec", "GroupBy: Detected CountDistinct, attempting 2-stage GPU execution.\n");

    const auto& distinctSpec = groupBy.aggSpecs[countDistinctIdx];
    std::string distinctInputStr = distinctSpec.inputExpr;

    // Verify multiple CountDistincts
    for (size_t i = 0; i < aggFuncs.size(); ++i) {
        if (aggFuncs[i] == AggFunc::CountDistinct) {
            if (groupBy.aggSpecs[i].inputExpr != distinctInputStr) {
                ENGINE_THROW("Multiple different CountDistinct columns not supported on GPU yet.");
            }
        }
    }

    // Stage 1: Group By {Keys + DistinctCol}
    IRGroupBy stage1Spec;
    stage1Spec.keys = groupBy.keys;
    stage1Spec.keyNames = groupBy.keyNames;

    // Add DistinctCol to keys
    if (distinctSpec.input) {
        stage1Spec.keys.push_back(distinctSpec.input);
        // Use inputExpr string as name? or a temp name
        std::string dName = "distinct_col_stage1";
        if (distinctSpec.input->kind == TypedExpr::Kind::Column) {
             dName = distinctSpec.input->asColumn().column;
        }
        stage1Spec.keyNames.push_back(dName);
    } else {
         ENGINE_THROW("CountDistinct missing input expression node");
    }

    // Stage 1 Aggregates: Add dummy COUNT(*) because GPU kernel requires at least 1 agg
    IRGroupBy::AggSpec dummyAgg;
    dummyAgg.func = AggFunc::CountStar; 
    dummyAgg.outputName = "dummy_cnt";
    stage1Spec.aggSpecs.push_back(dummyAgg);

    TableResult stage1Res;
    bool s1Ok = GpuExecutor::executeGroupBy(stage1Spec, ctx, stage1Res);
    ENGINE_ASSERT(s1Ok, "Stage 1 GroupBy failed (CountDistinct pre-pass)");

    LOG_DEBUG("Exec", "GroupBy: Stage 1 complete. Rows=" << stage1Res.rowCount);

    // Stage 2: Group By {Keys} on stage1Res, with COUNT(*)
    EvalContext stage2Ctx;
    stage2Ctx.rowCount = stage1Res.rowCount;

    // Populate context from Stage 1 result.
    // Skip u32 columns for string keys so the next GroupBy re-encodes them,
    // preserving string maps in the final result.

    std::set<std::string> strColNames;
    for(size_t i=0; i<stage1Res.stringNames.size(); ++i) {
        stage2Ctx.stringCols[stage1Res.stringNames[i]] = stage1Res.stringCols[i];
        strColNames.insert(stage1Res.stringNames[i]);
        // Build dictionary encoding so Stage 2 uses fast dict-ID path
        buildDictCol(stage2Ctx, stage1Res.stringNames[i]);
    }

    for(size_t i=0; i<stage1Res.u32Names.size(); ++i) {
         // Only copy as u32 if it's NOT a string column
         if (strColNames.find(stage1Res.u32Names[i]) == strColNames.end()) {
            stage2Ctx.u32Cols[stage1Res.u32Names[i]] = stage1Res.u32Cols[i];
         }
    }

    for(size_t i=0; i<stage1Res.f32Names.size(); ++i) {
        stage2Ctx.f32Cols[stage1Res.f32Names[i]] = stage1Res.f32Cols[i];
    }

    IRGroupBy stage2Spec;
    // Reconstruct keys for Stage 2 (Columns referencing stage1 outputs)
    for(const auto& kn : groupBy.keyNames) {
         auto col = std::make_shared<TypedExpr>();
         col->kind = TypedExpr::Kind::Column;
         col->asColumn().column = kn; 
         stage2Spec.keys.push_back(col);
         stage2Spec.keyNames.push_back(kn);
    }

    // Reconstruct aggregates
    for(size_t i=0; i<groupBy.aggSpecs.size(); ++i) {
        const auto& spec = groupBy.aggSpecs[i];
        IRGroupBy::AggSpec s2Agg;
        s2Agg.outputName = spec.outputName;

        if (spec.func == AggFunc::CountDistinct) {
            s2Agg.func = AggFunc::CountStar; 
        } else {
             s2Agg.func = spec.func; 
        }
        stage2Spec.aggSpecs.push_back(s2Agg);
    }

    return GpuExecutor::executeGroupBy(stage2Spec, stage2Ctx, out);
    return false;
}

struct AggColumnResult {
    std::vector<float> values;
    MTL::Buffer* gpuBuf = nullptr;
    bool gpuOwned = false;
};

static AggColumnResult resolveAggColumnAsF32(
    EvalContext& ctx, const std::string& col,
    size_t expectedKeyRows, bool /*debug*/)
{
    AggColumnResult result;

    // Lazy fetch u32 from GPU if CPU vector is missing
    if ((ctx.u32Cols.find(col) == ctx.u32Cols.end() || ctx.u32Cols[col].empty()) && ctx.u32ColsGPU.count(col)) {
        MTL::Buffer* buf = ctx.u32ColsGPU.at(col);
        size_t count = buf->length() / sizeof(uint32_t);
        if (count > 0) {
            std::vector<uint32_t> down(count);
            std::memcpy(down.data(), buf->contents(), count * sizeof(uint32_t));
            ctx.u32Cols[col] = std::move(down);
        }
    }

    // Try u32→f32 GPU cast (with optional activeRows gather)
    auto itU = ctx.u32Cols.find(col);
    if (itU != ctx.u32Cols.end()) {
        auto& s = GpuColumnStore::instance();
        MTL::Buffer* u32Buf = ctx.u32ColsGPU.count(col) ? ctx.u32ColsGPU[col]
            : s.device()->newBuffer(itU->second.data(), itU->second.size() * sizeof(uint32_t), MTL::ResourceStorageModeShared);
        bool ownU32 = !ctx.u32ColsGPU.count(col);
        MTL::Buffer* src = u32Buf;
        bool ownSrc = false;
        if (!ctx.activeRows.empty() && itU->second.size() > expectedKeyRows && ctx.activeRowsGPU) {
            src = GpuOps::gatherU32(u32Buf, ctx.activeRowsGPU, (uint32_t)expectedKeyRows, false).detach();
            ownSrc = true;
        }
        uint32_t castCount = ownSrc ? (uint32_t)expectedKeyRows : (uint32_t)itU->second.size();
        GpuBuffer f32Buf = GpuOps::castU32ToF32(src, castCount);
        // Skip CPU download — GPU buffer passed directly to kernel
        result.gpuBuf = f32Buf.detach();
        result.gpuOwned = true;
        if (ownSrc) src->release();
        if (ownU32) u32Buf->release();
        LOG_DEBUG("Exec", "GroupBy: resolved u32→f32 col " << col << " size=" << result.values.size() << " (GPU cast)\n");
        return result;
    }

    // Lazy fetch f32 from GPU if CPU vector is missing
    if ((ctx.f32Cols.find(col) == ctx.f32Cols.end() || ctx.f32Cols[col].empty()) && ctx.f32ColsGPU.count(col)) {
        MTL::Buffer* buf = ctx.f32ColsGPU.at(col);
        size_t count = buf->length() / sizeof(float);
        if (count > 0) {
            std::vector<float> down(count);
            std::memcpy(down.data(), buf->contents(), count * sizeof(float));
            ctx.f32Cols[col] = std::move(down);
        }
    }

    // Try f32 col (with optional activeRows gather)
    auto itF = ctx.f32Cols.find(col);
    if (itF != ctx.f32Cols.end()) {
        if (!ctx.activeRows.empty() && itF->second.size() > expectedKeyRows && ctx.activeRowsGPU) {
            auto& s = GpuColumnStore::instance();
            MTL::Buffer* src = ctx.f32ColsGPU.count(col) ? ctx.f32ColsGPU[col]
                : s.device()->newBuffer(itF->second.data(), itF->second.size() * sizeof(float), MTL::ResourceStorageModeShared);
            bool ownSrc = !ctx.f32ColsGPU.count(col);
            GpuBuffer dst = GpuOps::gatherF32(src, ctx.activeRowsGPU, (uint32_t)expectedKeyRows, false);
            // Skip CPU download — GPU buffer passed directly to kernel
            result.gpuBuf = dst.detach();
            result.gpuOwned = true;
            if (ownSrc) src->release();
        } else {
            result.values = itF->second;
            if (ctx.f32ColsGPU.count(col) && ctx.f32ColsGPU[col] &&
                ctx.f32ColsGPU[col]->length() >= itF->second.size() * sizeof(float)) {
                result.gpuBuf = ctx.f32ColsGPU[col];
                result.gpuOwned = false;
            }
        }
        LOG_DEBUG("Exec", "GroupBy: resolved f32 col " << col << " size=" << result.values.size());
        return result;
    }

    LOG_DEBUG("Exec", "GroupBy: col " << col << " not found in u32 or f32\n");
    return result; // empty = not found
}

// -- Extracted: buildAggInputs --
// Builds aggregate input vectors and GPU buffers for each IRGroupBy aggregate spec.
static GroupByAggData buildAggInputs(
    const IRGroupBy& groupBy, EvalContext& ctx,
    size_t expectedKeyRows, bool debug)
{
    GroupByAggData ad;
    
    for (const auto& spec : groupBy.aggSpecs) {
        ad.aggFuncs.push_back(spec.func);
        // Use outputName if provided, otherwise generate one from function and input
        std::string name = spec.outputName;
        if (name.empty()) {
            name = aggFuncName(spec.func);
            if (!spec.inputExpr.empty()) {
                name += "_" + spec.inputExpr;
            }
        }
        ad.aggNames.push_back(name);
        
        if (debug) {
            LOG_INFO("Exec", "GroupBy: agg func=" << static_cast<int>(spec.func)  << " outputName=" << name << " inputExpr=" << spec.inputExpr);
        }
        
        if (spec.func == AggFunc::CountStar) {
            // COUNT(*) doesn't need input - counts all rows
            ad.aggInputs.push_back({});
            ad.aggBufsGPU.push_back(nullptr);
        } else if (spec.func == AggFunc::Count || spec.func == AggFunc::CountDistinct) {
            // COUNT(column) / COUNT(DISTINCT column) - need to track which rows have non-NULL values
            // We'll store the column values so we can check for NULLs (0 = NULL sentinel)
            std::string col = spec.inputExpr;
            while (!col.empty() && col.front() == '(' && col.back() == ')') {
                col = col.substr(1, col.size() - 2);
            }
            col = trim_copy(col);
            
            // Get the u32 column values (most columns are u32) - prefer u32Cols for correct row count
            std::vector<float> input;
            // Get the column values as f32 via shared helper
            auto colRes = resolveAggColumnAsF32(ctx, col, expectedKeyRows, debug);
            input = std::move(colRes.values);
            MTL::Buffer* inputGPU = colRes.gpuBuf;

            ad.aggInputs.push_back(std::move(input));
            ad.aggBufsGPU.push_back(inputGPU);
        } else {
            // Evaluate input expression
            std::vector<float> input;
            MTL::Buffer* inputGPU = nullptr;  // Track GPU buffer to avoid re-upload
            bool inputGPUOwned = false;

            // Use pre-computed column if available (avoids double-gather)
            bool foundPrecomputed = false;
            if (!spec.inputExpr.empty()) {
                // Try f32Cols (CPU) for exact match with correct size
                auto itPreF = ctx.f32Cols.find(spec.inputExpr);
                if (itPreF != ctx.f32Cols.end() && itPreF->second.size() == expectedKeyRows) {
                    input = itPreF->second;
                    foundPrecomputed = true;
                    LOG_DEBUG("Exec", "GroupBy: using pre-computed f32Col '" << spec.inputExpr << "' (" << input.size() << " values)\n");
                }
                // Try f32ColsGPU for exact match
                if (!foundPrecomputed && ctx.f32ColsGPU.count(spec.inputExpr)) {
                    MTL::Buffer* preBuf = ctx.f32ColsGPU[spec.inputExpr];
                    size_t bufElems = preBuf->length() / sizeof(float);
                    if (bufElems == expectedKeyRows) {
                        // Skip CPU download — GPU buffer passed directly to kernel
                        inputGPU = preBuf;  // ctx-owned, don't release
                        inputGPUOwned = false;
                        foundPrecomputed = true;
                        LOG_DEBUG("Exec", "GroupBy: using pre-computed f32ColGPU '" << spec.inputExpr << "' (GPU-only, " << expectedKeyRows << " rows)\n");
                    }
                }
            }

            if (!foundPrecomputed) {
                GpuBuffer buf = GpuExecutor::evaluateExpression(spec.input, ctx);
                if (buf) {
                     uint32_t count = (ctx.activeRowsGPU) ? ctx.activeRowsCountGPU : ctx.rowCount;
                     if (ctx.activeRowsGPU && ctx.activeRowsCountGPU == 0) count = 0;
                     input.resize(count);
                     if (count > 0) std::memcpy(input.data(), buf->contents(), count * sizeof(float));
                     inputGPU = buf.detach();  // transfer ownership to raw ptr for aggBufsGPU
                     inputGPUOwned = true;
                }
            }

            if (debug) {
                LOG_INFO("Exec", "GroupBy: evaluateExpression returned " << input.size() << " values\n");
                if (!input.empty()) {
                    float sum = 0, minV = input[0], maxV = input[0];
                    size_t zeroCount = 0;
                    for (float v : input) { 
                        sum += v; 
                        minV = std::min(minV, v); 
                        maxV = std::max(maxV, v); 
                        if (v == 0) zeroCount++;
                    }
                    LOG_DEBUG("Exec", "GroupBy: evalExprFloat stats: sum=" << sum << " min=" << minV << " max=" << maxV << " avg=" << (sum/input.size()) << " zeros=" << zeroCount);
                    // Print first 10 values
                    LOG_DEBUG("Exec", "GroupBy: first 10 values: ");
                    for (size_t i = 0; i < std::min(input.size(), size_t(10)); ++i) {
                        LOG_DEBUG("GROUPBY", input[i] << " ");
                    }
                    LOG_DEBUG("GROUPBY", "\n");
                }
            }
            if (input.empty()) {
                // Try from inputExpr string (might be positional ref like #3)
                std::string col = spec.inputExpr;
                while (!col.empty() && col.front() == '(' && col.back() == ')') {
                    col = col.substr(1, col.size() - 2);
                }
                col = trim_copy(col);
                
                LOG_DEBUG("Exec", "GroupBy: trying col=" << col);
                
                // Infer column name if empty
                if (col.empty()) {
                    for (const auto& [cname, vals] : ctx.f32Cols) {
                        if (cname[0] == '#' || cname == "SUM" || cname == "AVG" || 
                            cname == "COUNT(*)" || cname == "MIN" || cname == "MAX") continue;
                        bool hasSuffix = cname.size() > 2 && cname[cname.size()-2] == '_' && 
                                        std::isdigit(cname[cname.size()-1]);
                        if (!hasSuffix) { col = cname; break; }
                    }
                    if (col.empty() && !ctx.f32Cols.empty()) {
                        col = ctx.f32Cols.begin()->first;
                    }
                    if (debug && !col.empty()) {
                        LOG_INFO("Exec", "GroupBy: inferred col=" << col << " for empty inputExpr\n");
                    }
                }
                
                auto colRes = resolveAggColumnAsF32(ctx, col, expectedKeyRows, debug);
                input = std::move(colRes.values);
                inputGPU = colRes.gpuBuf;
                inputGPUOwned = colRes.gpuOwned;
            }
            ad.aggInputs.push_back(std::move(input));
            // Retain ctx-owned GPU buffers so they can be safely released later
            if (inputGPU && !inputGPUOwned) inputGPU->retain();
            ad.aggBufsGPU.push_back(inputGPU);
        }
    }
    
    return ad;
}

// Debug-only: verify GPU GROUP BY hash table against CPU reference computation.
static void verifyGroupByGPUvsCPU(
    const std::vector<std::vector<uint32_t>>& keyVecs,
    const std::vector<std::vector<float>>& aggInputs,
    const std::vector<MTL::Buffer*>& keyBufs,
    const std::vector<MTL::Buffer*>& aggBufs,
    const std::vector<AggFunc>& aggFuncs,
    const uint32_t* keyWords,
    const uint32_t* aggWords,
    uint32_t cap,
    size_t gpuRowCount)
{
    // Check GPU key buffer vs CPU keyVecs row-by-row
    if (!keyBufs.empty() && !keyVecs.empty() && !keyVecs[0].empty()) {
        const uint32_t* gpuKeys = static_cast<const uint32_t*>(keyBufs[0]->contents());
        size_t mismatches = 0;
        for (size_t i = 0; i < std::min(gpuRowCount, keyVecs[0].size()); ++i) {
            uint32_t gpuKey = gpuKeys[i];          // biased
            uint32_t cpuKey = keyVecs[0][i] + 1;   // bias CPU for comparison
            if (gpuKey != cpuKey) {
                if (mismatches < 10) {
                    LOG_WARN("Exec", "GroupBy VERIFY: KEY MISMATCH at row " << i  << " GPU=" << gpuKey << " CPU=" << cpuKey);
                }
                mismatches++;
            }
        }
        LOG_WARN("Exec", "GroupBy VERIFY: key mismatches = " << mismatches << " / " << std::min(gpuRowCount, keyVecs[0].size()));
    }

    // Check GPU agg buffer vs CPU aggInputs row-by-row
    if (!aggBufs.empty() && !aggInputs.empty() && !aggInputs[0].empty()) {
        const float* gpuAggs = static_cast<const float*>(aggBufs[0]->contents());
        size_t mismatches = 0;
        for (size_t i = 0; i < std::min(gpuRowCount, aggInputs[0].size()); ++i) {
            if (gpuAggs[i] != aggInputs[0][i]) {
                if (mismatches < 10) {
                    LOG_WARN("Exec", "GroupBy VERIFY: AGG MISMATCH at row " << i  << " GPU=" << gpuAggs[i] << " CPU=" << aggInputs[0][i]);
                }
                mismatches++;
            }
        }
        LOG_WARN("Exec", "GroupBy VERIFY: agg mismatches = " << mismatches << " / " << std::min(gpuRowCount, aggInputs[0].size()));
    }

    // Compute CPU GROUP BY using double precision for accuracy
    // Skip CPU verification when keyVecs are empty (GPU-only path)
    bool hasAllKeyData = !keyVecs.empty();
    for (size_t k = 0; k < keyVecs.size() && hasAllKeyData; ++k)
        if (keyVecs[k].empty()) hasAllKeyData = false;
    if (!hasAllKeyData) return;

    std::unordered_map<uint64_t, double> cpuGroupSums;
    size_t numK = keyVecs.size();
    for (size_t i = 0; i < gpuRowCount; ++i) {
        uint64_t compositeKey = 0;
        for (size_t k = 0; k < numK; ++k) {
            compositeKey ^= (uint64_t(keyVecs[k][i]) + 1) * (2654435761ULL + k * 31);
        }
        double val = (aggFuncs.size() > 0 && !aggInputs[0].empty() && i < aggInputs[0].size())
            ? static_cast<double>(aggInputs[0][i]) : 0.0;
        cpuGroupSums[compositeKey] += val;
    }

    // CPU GROUP BY keyed on first key value
    std::unordered_map<uint32_t, double> cpuByKey0;
    for (size_t i = 0; i < gpuRowCount; ++i) {
        uint32_t k0 = keyVecs[0][i];
        double val = (aggFuncs.size() > 0 && !aggInputs[0].empty() && i < aggInputs[0].size())
            ? static_cast<double>(aggInputs[0][i]) : 0.0;
        cpuByKey0[k0] += val;
    }

    // Find top CPU group
    uint32_t cpuMaxKey = 0; double cpuMaxVal = -1e30;
    for (auto& [k, v] : cpuByKey0) {
        if (v > cpuMaxVal) { cpuMaxVal = v; cpuMaxKey = k; }
    }

    // Compute GPU grand total and find top GPU group
    double gpuGrandTotal = 0.0;
    uint32_t gpuMaxKey = 0; float gpuMaxVal = -1e30f;
    for (uint32_t s = 0; s < cap; ++s) {
        uint32_t k0 = keyWords[s * 8 + 0];
        if (k0 == 0) continue;
        uint32_t raw = aggWords[s * 16 + 0];
        float fval = *reinterpret_cast<const float*>(&raw);
        gpuGrandTotal += static_cast<double>(fval);
        if (fval > gpuMaxVal) {
            gpuMaxVal = fval;
            gpuMaxKey = k0 - 1;
        }
    }

    // Compute CPU grand total
    double cpuGrandTotal = 0.0;
    if (!aggInputs.empty() && !aggInputs[0].empty()) {
        for (size_t i = 0; i < std::min(aggInputs[0].size(), gpuRowCount); ++i) {
            cpuGrandTotal += static_cast<double>(aggInputs[0][i]);
        }
    }

    LOG_INFO("Exec", "GroupBy VERIFY: CPU grand total = " << std::fixed << std::setprecision(2) << cpuGrandTotal);
    LOG_INFO("Exec", "GroupBy VERIFY: GPU grand total = " << std::fixed << std::setprecision(2) << gpuGrandTotal);
    LOG_INFO("Exec", "GroupBy VERIFY: difference = " << std::fixed << std::setprecision(2) << (gpuGrandTotal - cpuGrandTotal));
    LOG_INFO("Exec", "GroupBy VERIFY: CPU max key=" << cpuMaxKey << " val=" << std::fixed << std::setprecision(2) << cpuMaxVal);
    LOG_INFO("Exec", "GroupBy VERIFY: GPU max key=" << gpuMaxKey << " val=" << std::fixed << std::setprecision(2) << static_cast<double>(gpuMaxVal));

    // Check GPU group for the CPU max key
    for (uint32_t s = 0; s < cap; ++s) {
        uint32_t k0 = keyWords[s * 8 + 0];
        if (k0 == cpuMaxKey + 1) {
            uint32_t raw = aggWords[s * 16 + 0];
            float fval = *reinterpret_cast<const float*>(&raw);
            LOG_INFO("Exec", "GroupBy VERIFY: GPU value for CPU max key " << cpuMaxKey << " = " << std::fixed << std::setprecision(2) << static_cast<double>(fval));
            break;
        }
    }

    // Count GPU groups exceeding CPU max
    size_t higherCount = 0;
    for (uint32_t s = 0; s < cap; ++s) {
        uint32_t k0 = keyWords[s * 8 + 0];
        if (k0 == 0) continue;
        uint32_t raw = aggWords[s * 16 + 0];
        float fval = *reinterpret_cast<const float*>(&raw);
        if (static_cast<double>(fval) > cpuMaxVal * 1.001) {
            higherCount++;
            LOG_INFO("Exec", "GroupBy VERIFY: GPU group key=" << (k0-1) << " val=" << std::fixed << std::setprecision(2) << static_cast<double>(fval) << " EXCEEDS CPU max!\n");
        }
    }
    LOG_INFO("Exec", "GroupBy VERIFY: " << higherCount << " GPU groups exceed CPU max revenue\n");
}

// Extract GPU hash table results into output TableResult columns.
static void processGroupByHTResults(
    GroupByHashTable& htResult,
    const std::vector<std::vector<uint32_t>>& keyVecs,
    const std::vector<AggFunc>& aggFuncs,
    const std::vector<std::string>& keyNames,
    const std::vector<std::string>& aggNames,
    TableResult& out)
{
    // Clear and resize for fresh extraction
    out.u32Cols.clear();
    out.u32Cols.resize(keyVecs.size());
    out.u32ColsGPU.clear();
    out.u32ColsGPU.resize(keyVecs.size());
    out.u32Names = keyNames;
    out.f32Cols.clear();
    out.f32Cols.resize(aggFuncs.size());
    out.f32ColsGPU.clear();
    out.f32ColsGPU.resize(aggFuncs.size());
    out.f32Names = aggNames;
    out.stringCols.clear();
    out.stringNames.clear();
    out.rowCount = 0;

    // GPU Stream Compaction: Mark → Prefix Sum → Compact
    uint32_t numKeysHT = static_cast<uint32_t>(keyVecs.size());
    uint32_t numAvgExtra = 0;
    for (auto& af : aggFuncs) if (af == AggFunc::Avg) numAvgExtra++;
    uint32_t numAggsTotal = static_cast<uint32_t>(aggFuncs.size()) + numAvgExtra;

    auto extractResult = GpuOps::extractGroupByHT(htResult, numKeysHT, numAggsTotal);
    if (extractResult && extractResult->rowCount > 0) {
        out.rowCount = extractResult->rowCount;

        // Move extracted keys into output
        for (size_t k = 0; k < keyVecs.size(); ++k) {
            out.u32Cols[k] = std::move(extractResult->keyCols[k]);
            out.u32ColsGPU[k] = std::move(extractResult->keyColsGPU[k]);
        }

        // Process raw aggregate words with correct type conversion
        for (size_t a = 0; a < aggFuncs.size(); ++a) {
            out.f32Cols[a].resize(extractResult->rowCount);
        }
        size_t avgCount = 0;
        for (size_t a = 0; a < aggFuncs.size(); ++a) {
            uint32_t rc = extractResult->rowCount;
            MTL::Buffer* aggGPU = (a < extractResult->aggColsGPU.size()) ? extractResult->aggColsGPU[a].get() : nullptr;

            if (aggFuncs[a] == AggFunc::CountStar) {
                if (aggGPU) {
                    GpuBuffer f32Buf = GpuOps::castU32ToF32(aggGPU, rc);
                    if (f32Buf) {
                        // Skip CPU memcpy — GPU buffer is authoritative; lazy-fetch at output.
                        out.f32Cols[a].clear();
                        out.f32ColsGPU[a] = std::move(f32Buf);
                        extractResult->aggColsGPU[a] = nullptr;
                    }
                } else {
                    const auto& rawWords = extractResult->aggWords[a];
                    for (uint32_t r = 0; r < rc; ++r)
                        out.f32Cols[a][r] = static_cast<float>(rawWords[r]);
                }
            } else if (aggFuncs[a] == AggFunc::Avg) {
                size_t countIdx = aggFuncs.size() + avgCount;
                MTL::Buffer* countGPU = (countIdx < extractResult->aggColsGPU.size()) ? extractResult->aggColsGPU[countIdx].get() : nullptr;
                if (aggGPU && countGPU) {
                    GpuBuffer countF32 = GpuOps::castU32ToF32(countGPU, rc);
                    if (countF32) {
                        GpuBuffer avgBuf = GpuOps::arithDivF32ColCol(aggGPU, countF32, rc);
                        if (avgBuf) {
                            // Skip CPU memcpy — GPU buffer is authoritative; lazy-fetch at output.
                            out.f32Cols[a].clear();
                            out.f32ColsGPU[a] = std::move(avgBuf);
                        }
                    }
                } else {
                    const auto& rawWords = extractResult->aggWords[a];
                    for (uint32_t r = 0; r < rc; ++r) {
                        uint32_t w = rawWords[r];
                        float sum = *reinterpret_cast<const float*>(&w);
                        uint32_t cw = extractResult->aggWords[countIdx][r];
                        float count = static_cast<float>(cw);
                        out.f32Cols[a][r] = count > 0 ? sum / count : 0.0f;
                    }
                }
                avgCount++;
            } else if (aggFuncs[a] == AggFunc::Count) {
                if (aggGPU) {
                    // Skip CPU memcpy — GPU buffer is authoritative; lazy-fetch at output.
                    out.f32Cols[a].clear();
                    out.f32ColsGPU[a] = std::move(extractResult->aggColsGPU[a]);
                } else {
                    const auto& rawWords = extractResult->aggWords[a];
                    for (uint32_t r = 0; r < rc; ++r) {
                        uint32_t w = rawWords[r];
                        out.f32Cols[a][r] = *reinterpret_cast<const float*>(&w);
                    }
                }
            } else {
                // SUM/MIN/MAX: raw bits are f32
                if (aggGPU) {
                    // Skip CPU memcpy — GPU buffer is authoritative; lazy-fetch at output.
                    out.f32Cols[a].clear();
                    out.f32ColsGPU[a] = std::move(extractResult->aggColsGPU[a]);
                } else {
                    const auto& rawWords = extractResult->aggWords[a];
                    for (uint32_t r = 0; r < rc; ++r) {
                        uint32_t w = rawWords[r];
                        out.f32Cols[a][r] = *reinterpret_cast<const float*>(&w);
                    }
                }
            }
        }
        // Create GPU buffers for aggregates that don't already have one
        for (size_t a = 0; a < aggFuncs.size(); ++a) {
            if (!out.f32ColsGPU[a] && !out.f32Cols[a].empty()) {
                out.f32ColsGPU[a] = GpuOps::createBuffer(
                    out.f32Cols[a].data(),
                    out.f32Cols[a].size() * sizeof(float));
            }
        }
    }
}

// -- Extracted: prepareGroupByGpuBuffers --
// Uploads key and aggregate vectors to GPU with +1 key bias, COUNT(col) indicator,
// and AVG→SUM+COUNT expansion. Returns all GPU buffers needed for the HT kernel.
struct GroupByGpuBuffers {
    std::vector<MTL::Buffer*> keyBufs;
    std::vector<MTL::Buffer*> aggBufs;
    std::vector<uint32_t> aggTypesGpu;
    std::vector<size_t> avgIndices;
    std::vector<MTL::Buffer*> toRelease;
    bool ok = true;
};

static GroupByGpuBuffers prepareGroupByGpuBuffers(
    const std::vector<std::vector<uint32_t>>& keyVecs,
    std::vector<MTL::Buffer*>& keyBufsGPU,
    const std::vector<AggFunc>& aggFuncs,
    const std::vector<std::vector<float>>& aggInputs,
    std::vector<MTL::Buffer*>& aggBufsGPU,
    size_t gpuRowCount, bool debug) {

    auto& store = GpuColumnStore::instance();
    GroupByGpuBuffers gb;

    // ── Upload key vectors with +1 bias ──
    for (size_t k = 0; k < keyVecs.size() && gb.ok; ++k) {
        MTL::Buffer* srcBuf = nullptr;
        bool usedGpuBuf = false;
        bool alreadyBiased = false;
        if (k < keyBufsGPU.size() && keyBufsGPU[k] &&
            keyBufsGPU[k]->length() == gpuRowCount * sizeof(uint32_t)) {
            srcBuf = keyBufsGPU[k];
            keyBufsGPU[k] = nullptr;
            usedGpuBuf = true;
        } else {
            // CPU upload: apply +1 bias inline, avoiding separate GPU dispatch
            srcBuf = store.device()->newBuffer(gpuRowCount * sizeof(uint32_t), MTL::ResourceStorageModeShared);
            if (!srcBuf) { gb.ok = false; break; }
            uint32_t* ptr = static_cast<uint32_t*>(srcBuf->contents());
            size_t copyLen = std::min(keyVecs[k].size(), gpuRowCount);
            for (size_t i = 0; i < copyLen; ++i) ptr[i] = keyVecs[k][i] + 1;
            if (copyLen < gpuRowCount) memset(ptr + copyLen, 0, (gpuRowCount - copyLen) * sizeof(uint32_t));
            alreadyBiased = true;
        }

        if (debug && k == 0) {
            const uint32_t* sp = static_cast<const uint32_t*>(srcBuf->contents());
            size_t srcMismatches = 0;
            for (size_t i = 0; i < std::min(gpuRowCount, keyVecs[k].size()); ++i) {
                if (sp[i] != keyVecs[k][i]) {
                    if (srcMismatches < 5)
                        LOG_WARN("Exec", "GroupBy: SRC MISMATCH[" << i << "] GPU=" << sp[i] << " CPU=" << keyVecs[k][i]);
                    srcMismatches++;
                }
            }
            LOG_WARN("Exec", "GroupBy: srcBuf vs keyVecs[0]: " << srcMismatches << " mismatches" << " (usedGpuBuf=" << usedGpuBuf << " srcLen=" << srcBuf->length()/4  << " keyVecLen=" << keyVecs[k].size() << " gpuRowCount=" << gpuRowCount << ")\n");
        }

        if (alreadyBiased) {
            // Bias applied during CPU copy — no GPU dispatch needed
            gb.keyBufs.push_back(srcBuf);
            gb.toRelease.push_back(srcBuf);
        } else {
            auto biased = GpuOps::arithAddConstU32(srcBuf, 1, static_cast<uint32_t>(gpuRowCount)).detach();
            srcBuf->release();
            if (!biased) { gb.ok = false; break; }

            if (debug) {
                uint32_t* bp = static_cast<uint32_t*>(biased->contents());
                std::map<uint32_t, size_t> dist;
                for (size_t i = 0; i < gpuRowCount; ++i) dist[bp[i]]++;
                LOG_INFO("Exec", "GroupBy: GPU key buf[" << k << "] distribution (biased):");
                for (auto& [v,c] : dist) std::cerr << " " << v << ":" << c;
                LOG_INFO("GROUPBY", "\n");
            }
            gb.keyBufs.push_back(biased);
            gb.toRelease.push_back(biased);
        }
    }

    // ── Upload aggregate input buffers ──
    for (size_t a = 0; a < aggFuncs.size() && gb.ok; ++a) {
        uint32_t gpuType = 0;
        switch (aggFuncs[a]) {
            case AggFunc::Sum: gpuType = 0; break;
            case AggFunc::Avg: gpuType = 0; break;
            case AggFunc::Count: gpuType = 0; break; // SUM of non-null indicator
            case AggFunc::CountStar: gpuType = 1; break;
            case AggFunc::Min: gpuType = 2; break;
            case AggFunc::Max: gpuType = 3; break;
            default: gpuType = 0; break;
        }
        gb.aggTypesGpu.push_back(gpuType);

        MTL::Buffer* aggBuf = nullptr;
        if (aggFuncs[a] == AggFunc::Count) {
            MTL::Buffer* srcBuf = nullptr;
            if (a < aggBufsGPU.size() && aggBufsGPU[a] &&
                aggBufsGPU[a]->length() >= gpuRowCount * sizeof(float)) {
                srcBuf = aggBufsGPU[a];
                aggBufsGPU[a] = nullptr;
            } else {
                srcBuf = store.device()->newBuffer(gpuRowCount * sizeof(float), MTL::ResourceStorageModeShared);
                if (srcBuf) {
                    float* ptr = static_cast<float*>(srcBuf->contents());
                    size_t copyLen = std::min(aggInputs[a].size(), gpuRowCount);
                    if (copyLen > 0) memcpy(ptr, aggInputs[a].data(), copyLen * sizeof(float));
                    if (copyLen < gpuRowCount) memset(ptr + copyLen, 0, (gpuRowCount - copyLen) * sizeof(float));
                }
            }
            if (srcBuf) {
                aggBuf = GpuOps::nonNullIndicatorF32(srcBuf, static_cast<uint32_t>(gpuRowCount)).detach();
                srcBuf->release();
                if (aggBuf) gb.toRelease.push_back(aggBuf);
            }
        } else if (gpuType == 1) {
            aggBuf = store.device()->newBuffer(gpuRowCount * sizeof(float), MTL::ResourceStorageModeShared);
            if (aggBuf) {
                std::memset(aggBuf->contents(), 0, gpuRowCount * sizeof(float));
                gb.toRelease.push_back(aggBuf);
            }
        } else if (!aggInputs[a].empty() ||
                   (a < aggBufsGPU.size() && aggBufsGPU[a] &&
                    aggBufsGPU[a]->length() >= gpuRowCount * sizeof(float))) {
            if (a < aggBufsGPU.size() && aggBufsGPU[a] &&
                aggBufsGPU[a]->length() >= gpuRowCount * sizeof(float)) {
                aggBuf = aggBufsGPU[a];
                aggBufsGPU[a] = nullptr;
                gb.toRelease.push_back(aggBuf);
            } else {
                aggBuf = store.device()->newBuffer(gpuRowCount * sizeof(float), MTL::ResourceStorageModeShared);
                if (aggBuf) {
                    float* ptr = static_cast<float*>(aggBuf->contents());
                    for (size_t i = 0; i < gpuRowCount; ++i)
                        ptr[i] = (i < aggInputs[a].size()) ? aggInputs[a][i] : 0.0f;
                    gb.toRelease.push_back(aggBuf);
                }
            }
        } else {
            aggBuf = store.device()->newBuffer(gpuRowCount * sizeof(float), MTL::ResourceStorageModeShared);
            if (aggBuf) {
                std::memset(aggBuf->contents(), 0, gpuRowCount * sizeof(float));
                gb.toRelease.push_back(aggBuf);
            }
        }
        if (!aggBuf) { gb.ok = false; break; }
        gb.aggBufs.push_back(aggBuf);
    }

    // AVG → append extra COUNT aggregate
    for (size_t a = 0; a < aggFuncs.size(); ++a) {
        if (aggFuncs[a] == AggFunc::Avg) {
            gb.avgIndices.push_back(a);
            gb.aggTypesGpu.push_back(1);
            MTL::Buffer* countBuf = store.device()->newBuffer(gpuRowCount * sizeof(float), MTL::ResourceStorageModeShared);
            if (countBuf) {
                std::memset(countBuf->contents(), 0, gpuRowCount * sizeof(float));
                gb.toRelease.push_back(countBuf);
                gb.aggBufs.push_back(countBuf);
            }
        }
    }

    return gb;
}

bool GpuExecutor::executeGroupBy(const IRGroupBy& groupBy, EvalContext& ctx, TableResult& out) {
    const bool debug = env_truthy("GPUDB_DEBUG_OPS");
    
    if (debug) {
        LOG_INFO("Exec", "GroupBy: ctx.rowCount=" << ctx.rowCount);
        LOG_INFO("Exec", "GroupBy: ctx.u32Cols.size=" << ctx.u32Cols.size() << ":");
        for (const auto& [n,v] : ctx.u32Cols) std::cerr << " " << n << "(" << v.size() << ")";
        LOG_DEBUG("GROUPBY", "\n");
        LOG_DEBUG("Exec", "GroupBy: ctx.f32Cols.size=" << ctx.f32Cols.size() << ":");
        if (debug) for (const auto& [n,v] : ctx.f32Cols) std::cerr << " " << n << "(" << v.size() << ")";
        LOG_DEBUG("GROUPBY", "\n");
        LOG_DEBUG("Exec", "GroupBy: keys.size=" << groupBy.keys.size());
        for (size_t i = 0; i < groupBy.keys.size(); ++i) {
            if (groupBy.keys[i] && groupBy.keys[i]->kind == TypedExpr::Kind::Column) {
                LOG_DEBUG("Exec", "GroupBy:   key[" << i << "]=" << groupBy.keys[i]->asColumn().column);
            }
        }
    }
    
    // Expected row count for this GroupBy - prefer GPU activeRows count
    size_t expectedKeyRows = ctx.rowCount;
    if (ctx.activeRowsGPU && ctx.activeRowsCountGPU > 0) {
        expectedKeyRows = ctx.activeRowsCountGPU;
    } else if (!ctx.activeRows.empty()) {
        expectedKeyRows = ctx.activeRows.size();
    }
    
    // Build key vectors (dict ID, GPU FNV1a hash, CPU fallback, f32 bitcast)
    auto kd = buildGroupByKeys(groupBy, ctx, expectedKeyRows, debug);
    auto& keyVecs = kd.keyVecs;
    auto& keyNames = kd.keyNames;
    auto& outputStringMaps = kd.outputStringMaps;
    auto& hashToStringMaps = kd.hashToStringMaps;
    auto& dictRefMaps = kd.dictRefMaps;
    auto& keyFromF32 = kd.keyFromF32;
    auto& keyBufsGPU = kd.keyBufsGPU;
    
    if (keyVecs.empty()) return false;
    
    // Check if any key vector is empty (0 rows) — also check GPU buffers
    bool hasEmptyKeys = false;
    for (size_t ki = 0; ki < keyVecs.size(); ++ki) {
        if (keyVecs[ki].empty() && !(ki < keyBufsGPU.size() && keyBufsGPU[ki])) {
            hasEmptyKeys = true;
            break;
        }
    }
    
    // Empty keys: return 0 groups
    if (hasEmptyKeys) {
        if (debug) {
            LOG_WARN("Exec", "GroupBy: empty key vectors, returning 0 groups\n");
        }
        out.rowCount = 0;
        out.u32Cols.clear();
        out.u32Cols.resize(keyVecs.size());
        out.u32Names = keyNames;
        out.f32Cols.clear();
        out.f32Names.clear();
        out.order.clear();
        // Still need to populate aggNames
        for (const auto& spec : groupBy.aggSpecs) {
            std::string name = spec.outputName;
            if (name.empty()) {
                name = aggFuncName(spec.func);
            }
            out.f32Names.push_back(name);
            out.f32Cols.push_back({});
        }
        for (size_t i = 0; i < out.u32Names.size(); ++i) {
            out.order.push_back({TableResult::ColRef::Kind::U32, i, out.u32Names[i]});
        }
        for (size_t i = 0; i < out.f32Names.size(); ++i) {
            out.order.push_back({TableResult::ColRef::Kind::F32, i, out.f32Names[i]});
        }
        return true;
    }
    
    // Build aggregate input vectors
    auto ad = buildAggInputs(groupBy, ctx, expectedKeyRows, debug);
    auto& aggInputs = ad.aggInputs;
    auto& aggBufsGPU = ad.aggBufsGPU;
    auto& aggFuncs = ad.aggFuncs;
    auto& aggNames = ad.aggNames;
    
    // --- CountDistinct Handling (2-Stage GPU) ---
    // Check if we have any CountDistinct aggregates
    int countDistinctIdx = -1;
    for (size_t i = 0; i < aggFuncs.size(); ++i) {
        if (aggFuncs[i] == AggFunc::CountDistinct) {
            countDistinctIdx = static_cast<int>(i);
            break;
        }
    }

    if (countDistinctIdx >= 0) {
        return handleCountDistinct(groupBy, ctx, out, aggFuncs, countDistinctIdx, debug);
    }
    
    // Try GPU GroupBy if all aggregates are GPU-compatible (no CountDistinct)
    bool useGpu = true;
    
    // Also require at most 8 keys (GPU kernel limit)
    if (keyVecs.size() > engine::config::kMaxGroupByKeys) {
        useGpu = false;
    }
    
    // Check for size consistency - all keyVecs and aggInputs should have same size
    // Prefer GPU activeRows count over CPU
    size_t expectedRowCount = ctx.rowCount;
    if (ctx.activeRowsGPU && ctx.activeRowsCountGPU > 0) {
        expectedRowCount = ctx.activeRowsCountGPU;
    } else if (!ctx.activeRows.empty()) {
        expectedRowCount = ctx.activeRows.size();
    }

    // Determine consensus row count from keys if possible
    size_t consensusRowCount = expectedRowCount;
    bool keysConsistent = true;
    if (!keyVecs.empty()) {
        // Prefer GPU buffer length over CPU vec for row count
        size_t firstSize = 0;
        if (!keyBufsGPU.empty() && keyBufsGPU[0])
            firstSize = keyBufsGPU[0]->length() / sizeof(uint32_t);
        else if (!keyVecs[0].empty())
            firstSize = keyVecs[0].size();
        for (size_t ki = 0; ki < keyVecs.size(); ++ki) {
            size_t ksz = 0;
            if (ki < keyBufsGPU.size() && keyBufsGPU[ki])
                ksz = keyBufsGPU[ki]->length() / sizeof(uint32_t);
            else if (!keyVecs[ki].empty())
                ksz = keyVecs[ki].size();
            if (ksz != firstSize) {
                keysConsistent = false;
                break;
            }
        }
        if (keysConsistent && firstSize > 0 && firstSize != expectedRowCount) {
            if (debug) {
                LOG_WARN("Exec", "GroupBy: Warning: ctx.rowCount (" << expectedRowCount  << ") differs from consistent key size (" << firstSize << "). Using key size.\n");
            }
            consensusRowCount = firstSize;
        }
    }
    
    // Verify key sizes match consensus
    for (size_t i = 0; i < keyVecs.size(); ++i) {
        size_t kvSize = 0;
        if (i < keyBufsGPU.size() && keyBufsGPU[i])
            kvSize = keyBufsGPU[i]->length() / sizeof(uint32_t);
        else
            kvSize = keyVecs[i].size();
        if (kvSize != consensusRowCount) {
            if (debug) {
                LOG_WARN("Exec", "GroupBy: key size mismatch for key index " << i << " (name: "  << (i < keyNames.size() ? keyNames[i] : "?") << "), expected " << consensusRowCount  << " but got " << kvSize);
                LOG_DEBUG("Exec", "ctx.rowCount=" << ctx.rowCount);
            }
            ENGINE_THROW("GroupBy Key size mismatch. CPU fallback disabled.");
        }
    }
    
    // GPU GroupBy path
    if (useGpu && !keyVecs.empty()) {
        auto& store = GpuColumnStore::instance();
        if (store.device() && store.library() && store.queue()) {
            

            if (debug) {
                LOG_INFO("Exec", "GroupBy: Using GPU path with " << keyVecs.size() << " keys and " << aggFuncs.size() << " aggregates\n");
            }
            
            // Determine row count from key vectors - prefer GPU activeRows
            size_t gpuRowCount = ctx.rowCount;
            if (ctx.activeRowsGPU && ctx.activeRowsCountGPU > 0) {
                gpuRowCount = ctx.activeRowsCountGPU;
            } else if (!ctx.activeRows.empty()) {
                gpuRowCount = ctx.activeRows.size();
            }
            if (!keyVecs.empty() && !keyVecs[0].empty()) {
                gpuRowCount = keyVecs[0].size();
            } else if (!keyBufsGPU.empty() && keyBufsGPU[0]) {
                gpuRowCount = keyBufsGPU[0]->length() / sizeof(uint32_t);
            }
            
            // Prepare all GPU buffers (keys with +1 bias, agg inputs, AVG expansion)
            auto gb = prepareGroupByGpuBuffers(keyVecs, keyBufsGPU, aggFuncs, aggInputs, aggBufsGPU, gpuRowCount, debug);
            auto& keyBufs = gb.keyBufs;
            auto& aggBufs = gb.aggBufs;
            auto& aggTypesGpu = gb.aggTypesGpu;
            auto& toRelease = gb.toRelease;
            bool gpuOk = gb.ok;
            
            if (gpuOk) {
                auto htOpt = GpuOps::groupByAggMultiKeyTyped(keyBufs, aggBufs, aggTypesGpu, static_cast<uint32_t>(gpuRowCount));
                
                if (htOpt.has_value()) {
                    const uint32_t cap = htOpt->capacity;
                    const auto* keyWords = reinterpret_cast<const uint32_t*>(htOpt->htKeys->contents());
                    const auto* aggWords = reinterpret_cast<const uint32_t*>(htOpt->htAggs->contents());
                    
                    if (debug) {
                        LOG_INFO("Exec", "GroupBy: GPU hash table capacity=" << cap << " gpuRowCount=" << gpuRowCount);
                        // Count non-empty slots
                        size_t nonEmpty = 0;
                        for (uint32_t s = 0; s < cap; ++s) {
                            if (keyWords[s * 8 + 0] != 0) nonEmpty++;
                        }
                        LOG_DEBUG("Exec", "GroupBy: GPU hash table non-empty slots=" << nonEmpty);
                    }
                    
                    // ── CPU verification: recompute GROUP BY from input data, compare with GPU ──
                    if (debug) {
                        verifyGroupByGPUvsCPU(keyVecs, aggInputs, keyBufs, aggBufs, aggFuncs,
                                              keyWords, aggWords, cap, gpuRowCount);
                    }
                    
                    processGroupByHTResults(*htOpt, keyVecs, aggFuncs, keyNames, aggNames, out);
                    
                    postProcessStringKeys(keyVecs, outputStringMaps, hashToStringMaps, dictRefMaps, out, debug);

                    restoreF32Keys(keyFromF32, out, debug);

                    buildGroupByOutputOrder(keyVecs, keyFromF32, outputStringMaps, hashToStringMaps, dictRefMaps, out);

                    // Release GPU resources
                    GpuOps::release(*htOpt);
                    for (auto* buf : toRelease) buf->release();
                    for (auto* buf : keyBufsGPU) { if (buf) buf->release(); }
                    for (size_t i = 0; i < aggBufsGPU.size(); ++i) { if (aggBufsGPU[i]) aggBufsGPU[i]->release(); }
                    
                    if (debug) {
                        LOG_INFO("Exec", "GroupBy: GPU completed with " << out.rowCount << " groups\n");
                        LOG_INFO("Exec", "GroupBy: GPU output u32_cols.size=" << out.u32Cols.size());
                        for (size_t i = 0; i < out.u32Cols.size(); ++i) {
                            LOG_DEBUG("GROUPBY", " " << out.u32Names[i] << "(" << out.u32Cols[i].size() << ")");
                        }
                        LOG_DEBUG("GROUPBY", "\n[Exec] GroupBy: GPU output f32_cols.size=" << out.f32Cols.size());
                        for (size_t i = 0; i < out.f32Cols.size(); ++i) {
                            LOG_DEBUG("GROUPBY", " " << out.f32Names[i] << "(" << out.f32Cols[i].size() << ")");
                            if (!out.f32Cols[i].empty()) {
                                LOG_DEBUG("GROUPBY", "[" << out.f32Cols[i][0]);
                                if (out.f32Cols[i].size() > 1) LOG_DEBUG("GROUPBY", "," << out.f32Cols[i][1]);
                                LOG_DEBUG("GROUPBY", "]");
                            }
                        }
                        LOG_DEBUG("GROUPBY", "\n");
                    }
                    return true;
                }
            }
            
            
            // GPU failed, release buffers and fall through to CPU
            for (auto* buf : toRelease) buf->release();
            for (auto* buf : keyBufsGPU) { if (buf) buf->release(); }
            for (size_t i = 0; i < aggBufsGPU.size(); ++i) { if (aggBufsGPU[i]) aggBufsGPU[i]->release(); }
            if (debug) {
                LOG_WARN("Exec", "GroupBy: GPU path failed, falling back to CPU\n");
            }
        }
    }
    
    for (auto* buf : keyBufsGPU) { if (buf) buf->release(); }
    for (size_t i = 0; i < aggBufsGPU.size(); ++i) { if (aggBufsGPU[i]) aggBufsGPU[i]->release(); }
    ENGINE_THROW("GPU GroupBy failed: conditions not met for any kernel (and CPU fallback is disabled).");
}

} // namespace engine
