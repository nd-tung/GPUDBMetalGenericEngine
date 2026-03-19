#include "Operators.hpp"
#include "OperatorsInternal.hpp"
#include "EnvUtil.hpp"
#include "KernelTimer.hpp"
#include "EngineConfig.hpp"

#include <cstring>
#include <iostream>
#include "Logger.hpp"

namespace engine {

std::optional<GroupByHashTable> GpuOps::groupByAggMultiKeyTyped(const std::vector<MTL::Buffer*>& keyColsU32,
                                                                         const std::vector<MTL::Buffer*>& aggInputsF32,
                                                                         const std::vector<uint32_t>& aggTypes,
                                                                         uint32_t rowCount) {
    auto& store = GpuColumnStore::instance();
    if (!store.device() || !store.library() || !store.queue()) return std::nullopt;

    auto p = makePSO(store.device(), store.library(), "ops::groupby_agg_multi_key_typed");
    if (!p) return std::nullopt;

    uint32_t numKeys = static_cast<uint32_t>(keyColsU32.size());
    if (numKeys == 0 || numKeys > engine::config::kMaxGroupByKeys) return std::nullopt;

    const uint32_t numAggs = static_cast<uint32_t>(aggTypes.size());
    if (numAggs == 0 || numAggs > engine::config::kMaxGroupByAggs) return std::nullopt;
    if (aggInputsF32.size() < numAggs) return std::nullopt;

    // Cap hash table capacity to avoid enormous Metal allocations.
    // htKeys = cap*32 bytes, htAggs = cap*64 bytes.  Cap at 2^26 (~6.1 GB total).
    constexpr uint32_t kMaxHTCap = 1u << 26; // 67M slots
    uint64_t desired = static_cast<uint64_t>(rowCount) * 2u;
    uint32_t cap = nextPow2(std::max<uint32_t>(128u, static_cast<uint32_t>(std::min<uint64_t>(desired, kMaxHTCap))));
    if (cap > kMaxHTCap) cap = kMaxHTCap;

    // Stride increased from 4 to 8, size is cap * 8 * sizeof(uint32_t)
    auto htKeys = store.device()->newBuffer(static_cast<size_t>(cap) * 8 * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    auto htAggs = store.device()->newBuffer(static_cast<size_t>(cap) * 16 * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    if (!htKeys || !htAggs) {
        if (htKeys) htKeys->release();
        if (htAggs) htAggs->release();
        return std::nullopt; // fall back to CPU path
    }
    std::memset(htKeys->contents(), 0, static_cast<size_t>(cap) * 8 * sizeof(uint32_t));
    std::memset(htAggs->contents(), 0, static_cast<size_t>(cap) * 16 * sizeof(uint32_t));

    auto agg_types_buf = store.device()->newBuffer(static_cast<size_t>(numAggs) * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    std::memcpy(agg_types_buf->contents(), aggTypes.data(), static_cast<size_t>(numAggs) * sizeof(uint32_t));

    // Always bind non-null agg buffers (kernel ignores them for COUNT slots).
    MTL::Buffer* dummyAgg = nullptr;
    for (uint32_t a = 0; a < numAggs; ++a) {
        if (aggTypes[a] == 0u && aggInputsF32[a] == nullptr) {
            dummyAgg = store.device()->newBuffer(static_cast<size_t>(rowCount) * sizeof(float), MTL::ResourceStorageModeShared);
            std::memset(dummyAgg->contents(), 0, static_cast<size_t>(rowCount) * sizeof(float));
            break;
        }
    }
    if (!dummyAgg) {
        // Even if no SUM slots exist, bind a small dummy buffer to satisfy setBuffer calls.
        dummyAgg = store.device()->newBuffer(sizeof(float), MTL::ResourceStorageModeShared);
        *reinterpret_cast<float*>(dummyAgg->contents()) = 0.0f;
    }

    MTL::Buffer* k0 = keyColsU32[0];
    MTL::Buffer* k1 = keyColsU32.size() > 1 ? keyColsU32[1] : keyColsU32[0];
    MTL::Buffer* k2 = keyColsU32.size() > 2 ? keyColsU32[2] : keyColsU32[0];
    MTL::Buffer* k3 = keyColsU32.size() > 3 ? keyColsU32[3] : keyColsU32[0];
    MTL::Buffer* k4 = keyColsU32.size() > 4 ? keyColsU32[4] : keyColsU32[0];
    MTL::Buffer* k5 = keyColsU32.size() > 5 ? keyColsU32[5] : keyColsU32[0];
    MTL::Buffer* k6 = keyColsU32.size() > 6 ? keyColsU32[6] : keyColsU32[0];
    MTL::Buffer* k7 = keyColsU32.size() > 7 ? keyColsU32[7] : keyColsU32[0];

    // ── Diagnostic: verify key-aggregate alignment ──
    if (env_truthy("GPUDB_DEBUG_OPS")) {
        const uint32_t* kp = static_cast<const uint32_t*>(k0->contents());
        const float* ap = (numAggs > 0 && aggInputsF32[0]) ? static_cast<const float*>(aggInputsF32[0]->contents()) : nullptr;
        LOG_INFO("Ops", "groupByAgg: GPU key buf len=" << k0->length()/4 << " agg buf len=" << (ap ? aggInputsF32[0]->length()/4 : 0) << " rowCount=" << rowCount);
        if (kp && ap) {
            // Print first 10 key-value pairs
            LOG_INFO("Ops", "groupByAgg: first 10 GPU key-value pairs: ");
            for (uint32_t i = 0; i < std::min(rowCount, 10u); ++i) {
                LOG_INFO("GROUPBY", "[k=" << kp[i] << " v=" << ap[i] << "] ");
            }
            LOG_INFO("GROUPBY", "\n");
            // Compute per-key sums using double precision
            std::unordered_map<uint32_t, double> gpuBufSums;
            for (uint32_t i = 0; i < rowCount; ++i) {
                gpuBufSums[kp[i]] += static_cast<double>(ap[i]);
            }
            // Find max from GPU buffers directly
            uint32_t maxK = 0; double maxV = -1e30;
            for (auto& [k, v] : gpuBufSums) {
                if (v > maxV) { maxV = v; maxK = k; }
            }
            LOG_INFO("Ops", "groupByAgg: GPU BUFFER (pre-kernel) max key=" << maxK << " (debiased=" << (maxK-1) << ") val=" << std::fixed << std::setprecision(2) << maxV);
            LOG_INFO("Ops", "groupByAgg: GPU BUFFER unique keys=" << gpuBufSums.size());
        }
    }

    {
        auto cmd = store.queue()->commandBuffer();
        auto enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(p);

        enc->setBuffer(k0, 0, 0);
        enc->setBuffer(k1, 0, 1);
        enc->setBuffer(k2, 0, 2);
        enc->setBuffer(k3, 0, 3);

        // agg buffers 0..15 are always bound at indices 4..19.
        for (uint32_t a = 0; a < 16; ++a) {
            MTL::Buffer* buf = dummyAgg;
            if (a < numAggs && aggInputsF32[a] != nullptr) buf = aggInputsF32[a];
            enc->setBuffer(buf, 0, 4 + a);
        }

        enc->setBuffer(htKeys, 0, 20);
        enc->setBuffer(htAggs, 0, 21);
        enc->setBytes(&cap, sizeof(cap), 22);
        enc->setBytes(&rowCount, sizeof(rowCount), 23);
        enc->setBytes(&numKeys, sizeof(numKeys), 24);
        enc->setBytes(&numAggs, sizeof(numAggs), 25);
        enc->setBuffer(agg_types_buf, 0, 26);
        enc->setBuffer(k4, 0, 27);
        enc->setBuffer(k5, 0, 28);
        enc->setBuffer(k6, 0, 29);
        enc->setBuffer(k7, 0, 30);

        dispatch1D(enc, rowCount);
        enc->endEncoding();
        
        auto t0 = std::chrono::high_resolution_clock::now();
        cmd->commit();
        cmd->waitUntilCompleted();
        checkGpuStatus(cmd, "groupby_agg_multi_key_typed");
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        KernelTimer::instance().record("ops::groupby_agg_multi_key_typed", "groupby", ms, rowCount);
    }

    dummyAgg->release();
    agg_types_buf->release();

    GroupByHashTable g;
    g.htKeys.reset(htKeys);
    g.htAggs.reset(htAggs);
    g.capacity = cap;
    return g;
}

std::optional<GroupByExtractResult> GpuOps::extractGroupByHT(
    const GroupByHashTable& ht,
    uint32_t numKeys,
    uint32_t numAggsTotal)
{
    auto& store = GpuColumnStore::instance();
    if (!store.device() || !store.library() || !store.queue()) return std::nullopt;

    auto p_mark    = makePSO(store.device(), store.library(), "ops::ht_mark_valid");
    auto p_extract = makePSO(store.device(), store.library(), "ops::ht_extract_compact");
    if (!p_mark || !p_extract) return std::nullopt;

    uint32_t cap = ht.capacity;
    if (cap == 0) return GroupByExtractResult{{}, {}, {}, {}, 0};

    // Step 1 (Mark): GPU writes 1 for valid slots, 0 for empty.
    auto markBuf = store.device()->newBuffer(
        static_cast<size_t>(cap) * sizeof(uint32_t), MTL::ResourceStorageModeShared);

    // Step 2 (Prefix Sum): Blit mark → offsets on GPU, then run exclusive prefix sum.
    auto offsetsBuf = store.device()->newBuffer(
        static_cast<size_t>(cap) * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    {
        auto cmd = store.queue()->commandBuffer();
        // Compute encoder: mark valid slots
        auto enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(p_mark);
        enc->setBuffer(ht.htKeys, 0, 0);
        enc->setBuffer(markBuf, 0, 1);
        enc->setBytes(&cap, sizeof(cap), 2);
        dispatch1D(enc, cap);
        enc->endEncoding();
        // Blit encoder: copy mark → offsets (replaces CPU memcpy, avoids GPU→CPU round-trip)
        auto blit = cmd->blitCommandEncoder();
        blit->copyFromBuffer(markBuf, 0, offsetsBuf, 0,
                             static_cast<NS::UInteger>(cap) * sizeof(uint32_t));
        blit->endEncoding();
        cmd->commit();
    }

    uint64_t totalSum = scanInPlace(offsetsBuf, cap);
    uint32_t totalCount = static_cast<uint32_t>(totalSum);

    if (totalCount == 0) {
        markBuf->release();
        offsetsBuf->release();
        return GroupByExtractResult{{}, {}, {}, {}, 0};
    }

    // Step 3 (Compact): GPU writes valid keys/aggs to dense output.
    auto outKeysBuf = store.device()->newBuffer(
        static_cast<size_t>(totalCount) * numKeys * sizeof(uint32_t),
        MTL::ResourceStorageModeShared);
    auto outAggsBuf = store.device()->newBuffer(
        static_cast<size_t>(totalCount) * numAggsTotal * sizeof(uint32_t),
        MTL::ResourceStorageModeShared);
    {
        auto cmd = store.queue()->commandBuffer();
        auto enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(p_extract);
        enc->setBuffer(ht.htKeys, 0, 0);
        enc->setBuffer(ht.htAggs, 0, 1);
        enc->setBuffer(markBuf, 0, 2);
        enc->setBuffer(offsetsBuf, 0, 3);
        enc->setBuffer(outKeysBuf, 0, 4);
        enc->setBuffer(outAggsBuf, 0, 5);
        enc->setBytes(&cap, sizeof(cap), 6);
        enc->setBytes(&numKeys, sizeof(numKeys), 7);
        enc->setBytes(&numAggsTotal, sizeof(numAggsTotal), 8);
        enc->setBytes(&totalCount, sizeof(totalCount), 9);
        dispatch1D(enc, cap);
        enc->endEncoding();
        auto t0 = std::chrono::high_resolution_clock::now();
        cmd->commit();
        cmd->waitUntilCompleted();
        checkGpuStatus(cmd, "ht_extract_compact");
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        KernelTimer::instance().record("ops::ht_extract_compact", "groupby", ms, totalCount);
    }

    // GPU already produced SoA output — direct memcpy per column.
    GroupByExtractResult result;
    result.rowCount = totalCount;
    result.keyCols.resize(numKeys);
    result.aggWords.resize(numAggsTotal);
    result.keyColsGPU.resize(numKeys);
    result.aggColsGPU.resize(numAggsTotal);

    auto* keyPtr = reinterpret_cast<const uint32_t*>(outKeysBuf->contents());
    auto* aggPtr = reinterpret_cast<const uint32_t*>(outAggsBuf->contents());

    for (uint32_t k = 0; k < numKeys; ++k) {
        result.keyCols[k].resize(totalCount);
        std::memcpy(result.keyCols[k].data(), keyPtr + k * totalCount, totalCount * sizeof(uint32_t));
        // Create per-column GPU buffer from SoA slice (avoids re-upload downstream)
        result.keyColsGPU[k].reset(store.device()->newBuffer(
            keyPtr + k * totalCount, totalCount * sizeof(uint32_t), MTL::ResourceStorageModeShared));
    }
    for (uint32_t a = 0; a < numAggsTotal; ++a) {
        // Skip CPU memcpy — GPU buffer is authoritative; lazy-fetch at output if needed.
        // CPU vector stays empty; processGroupByHTResults uses aggColsGPU exclusively.
        result.aggColsGPU[a].reset(store.device()->newBuffer(
            aggPtr + a * totalCount, totalCount * sizeof(uint32_t), MTL::ResourceStorageModeShared));
    }

    markBuf->release();
    offsetsBuf->release();
    outKeysBuf->release();
    outAggsBuf->release();

    return result;
}

void GpuOps::release(GroupByHashTable& g) {
    g.htKeys = nullptr;
    g.htAggs = nullptr;
    g.capacity = 0;
}

GpuBuffer GpuOps::dedupByKeys(const std::vector<MTL::Buffer*>& keys, uint32_t count,
                                  uint32_t& uniqueCount) {
    uniqueCount = 0;
    if (count == 0 || keys.empty()) return {};
    auto& store = GpuColumnStore::instance();
    auto dev = store.device();
    auto lib = store.library();
    if (!dev || !lib || !store.queue()) return {};

    // Build sort key: single u32 or packed u64
    bool useU64 = (keys.size() >= 2);
    GpuBuffer sortKeys(nullptr);

    if (keys.size() == 1) {
        // Copy key to avoid mutating original
        sortKeys.reset(dev->newBuffer(count * sizeof(uint32_t), MTL::ResourceStorageModeShared));
        std::memcpy(sortKeys->contents(), keys[0]->contents(), count * sizeof(uint32_t));
    } else if (keys.size() == 2) {
        sortKeys = packU32ToU64(keys[0], keys[1], count);
    } else {
        // 3+ keys: pack first two into u64, then GPU-fold remaining via hash
        sortKeys = packU32ToU64(keys[0], keys[1], count);
        auto pHash = makePSO(dev, lib, "ops::hash_combine_u64_u32");
        for (size_t k = 2; k < keys.size(); ++k) {
            if (pHash) {
                auto cmd = store.queue()->commandBuffer();
                auto enc = cmd->computeCommandEncoder();
                enc->setComputePipelineState(pHash);
                enc->setBuffer(sortKeys, 0, 0);
                enc->setBuffer(keys[k], 0, 1);
                enc->setBytes(&count, sizeof(count), 2);
                dispatch1D(enc, count);
                enc->endEncoding();
                cmd->commit();
                cmd->waitUntilCompleted();
                checkGpuStatus(cmd, "hash_combine_u64_u32");
            } else {
                // CPU fallback if kernel not found
                auto* ptr = static_cast<uint64_t*>(sortKeys->contents());
                auto* kp = static_cast<const uint32_t*>(keys[k]->contents());
                for (uint32_t i = 0; i < count; ++i) {
                    ptr[i] = ptr[i] * 0x9E3779B97F4A7C15ULL + kp[i];
                }
            }
        }
    }

    // GPU iota index array [0, 1, 2, ...]
    GpuBuffer indices = iotaU32(count);

    // Radix sort
    if (useU64) {
        radixSortU64(sortKeys, indices, count);
    } else {
        radixSortU32(sortKeys, indices, count);
    }

    // Mark unique positions after sort
    GpuBuffer mask(dev->newBuffer(count * sizeof(uint32_t), MTL::ResourceStorageModeShared));
    const char* kernelName = useU64 ? "ops::mark_unique_sorted_u64" : "ops::mark_unique_sorted_u32";
    auto pso = makePSO(dev, lib, kernelName);
    if (!pso) {
        return GpuBuffer(nullptr);
    }

    {
        auto cmd = store.queue()->commandBuffer();
        auto enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(pso);
        enc->setBuffer(sortKeys, 0, 0);
        enc->setBuffer(mask, 0, 1);
        enc->setBytes(&count, sizeof(count), 2);
        dispatch1D(enc, count);
        enc->endEncoding();
        cmd->commit();
        cmd->waitUntilCompleted();
        checkGpuStatus(cmd, "mark_unique_sorted");
    }

    // sortKeys no longer needed after marking unique positions
    sortKeys = nullptr;

    // Compact mask → positions where mask[i]==1
    auto [maskIdx, uCount] = compactU32Mask(mask, count);
    mask = nullptr;

    if (!maskIdx || uCount == 0) {
        return GpuBuffer(nullptr);
    }

    if (uCount == count) {
        // All unique — no dedup needed
        uniqueCount = count;
        return GpuBuffer(nullptr);
    }

    // Gather original indices at unique positions
    GpuBuffer uniqueIdx = gatherU32(indices, maskIdx, uCount);

    uniqueCount = uCount;
    return uniqueIdx;
}


} // namespace engine
