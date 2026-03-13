#include "Operators.hpp"
#include "OperatorsInternal.hpp"
#include "EnvUtil.hpp"
#include "KernelTimer.hpp"

#include <cstring>
#include <iostream>
#include "Logger.hpp"

namespace engine {

JoinResult GpuOps::joinHash(MTL::Buffer* buildKeys, 
                                     uint32_t buildCount,
                                     MTL::Buffer* probeKeys,
                                     uint32_t probeCount) {
    auto& store = GpuColumnStore::instance();
    if (buildCount == 0 || probeCount == 0 || !store.device()) return JoinResult{};

    // Use multi-match join to correctly handle duplicate keys on the build side.
    // The hash table uses linked lists so multiple build rows per key are preserved.
    
    // 1. Setup Hash Table for multi-match
    uint32_t capacity = 1024;
    uint64_t minCap = static_cast<uint64_t>(buildCount) * 2;
    while (capacity < minCap && capacity < (1u << 30)) capacity <<= 1;
    
    GpuBuffer bufHTKeys(store.device()->newBuffer(capacity * sizeof(uint32_t), MTL::ResourceStorageModeShared));
    GpuBuffer bufHTHead(store.device()->newBuffer(capacity * sizeof(uint32_t), MTL::ResourceStorageModeShared));
    GpuBuffer bufNext  (store.device()->newBuffer(buildCount * sizeof(uint32_t), MTL::ResourceStorageModeShared));
    
    std::memset(bufHTKeys->contents(), 0, capacity * sizeof(uint32_t)); // 0 = empty sentinel
    std::memset(bufHTHead->contents(), 0, capacity * sizeof(uint32_t)); // 0 = null pointer
    
    // 2. Build Phase — build linked lists per key
    auto p_build = makePSO(store.device(), store.library(), "ops::hash_join_build_multi");
    if (!p_build) {
        return JoinResult{};
    }
    
    // 3. Count Phase — count matches per probe row
    GpuBuffer bufCounts(store.device()->newBuffer(probeCount * sizeof(uint32_t), MTL::ResourceStorageModeShared));
    auto p_count = makePSO(store.device(), store.library(), "ops::hash_join_probe_count_multi");
    if (!p_count) {
        return JoinResult{};
    }
    
    // Fused: BUILD → COUNT in one command buffer (2 encoders)
    {
        auto cmd = store.queue()->commandBuffer();
        // Encoder 1: build
        auto enc1 = cmd->computeCommandEncoder();
        enc1->setComputePipelineState(p_build);
        enc1->setBuffer(buildKeys, 0, 0);
        enc1->setBuffer(bufHTKeys, 0, 1);
        enc1->setBuffer(bufHTHead, 0, 2);
        enc1->setBuffer(bufNext, 0, 3);
        enc1->setBytes(&capacity, 4, 4);
        enc1->setBytes(&buildCount, 4, 5);
        dispatch1D(enc1, buildCount);
        enc1->endEncoding();
        // Encoder 2: count
        auto enc2 = cmd->computeCommandEncoder();
        enc2->setComputePipelineState(p_count);
        enc2->setBuffer(probeKeys, 0, 0);
        enc2->setBuffer(bufHTKeys, 0, 1);
        enc2->setBuffer(bufHTHead, 0, 2);
        enc2->setBuffer(bufNext, 0, 3);
        enc2->setBuffer(bufCounts, 0, 4);
        enc2->setBytes(&capacity, 4, 5);
        enc2->setBytes(&probeCount, 4, 6);
        dispatch1D(enc2, probeCount);
        enc2->endEncoding();
        cmd->commit();
        cmd->waitUntilCompleted();
        checkGpuStatus(cmd, "joinHash_count");
    }
    
    // 4. GPU exclusive prefix sum on counts → offsets, returns total
    uint64_t totalPairs64 = scanInPlace(bufCounts, probeCount);
    uint32_t totalPairs = static_cast<uint32_t>(totalPairs64);
    
    LOG_DEBUG("GPU", "joinHashMulti: buildCount=" << buildCount  << " probeCount=" << probeCount << " totalPairs=" << totalPairs);
    
    if (totalPairs == 0) {
        auto emptyBuf = store.device()->newBuffer(sizeof(uint32_t), MTL::ResourceStorageModeShared);
        return {GpuBuffer(emptyBuf), GpuBuffer(store.device()->newBuffer(sizeof(uint32_t), MTL::ResourceStorageModeShared)), 0};
    }
    
    // 5. Write Phase — write matched pairs (bufCounts now holds exclusive prefix sums = offsets)
    MTL::Buffer* bufOffsets = bufCounts;  // reuse in-place — scanInPlace converted counts → offsets
    GpuBuffer outProbeIndices(store.device()->newBuffer(totalPairs * sizeof(uint32_t), MTL::ResourceStorageModeShared));
    GpuBuffer outBuildIndices(store.device()->newBuffer(totalPairs * sizeof(uint32_t), MTL::ResourceStorageModeShared));
    
    auto p_write = makePSO(store.device(), store.library(), "ops::hash_join_probe_write_multi");
    if (!p_write) {
        return JoinResult{};
    }
    
    {
        auto start = std::chrono::high_resolution_clock::now();
        auto cmd = store.queue()->commandBuffer();
        auto enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(p_write);
        enc->setBuffer(probeKeys, 0, 0);
        enc->setBuffer(bufHTKeys, 0, 1);
        enc->setBuffer(bufHTHead, 0, 2);
        enc->setBuffer(bufNext, 0, 3);
        enc->setBuffer(bufOffsets, 0, 4);
        enc->setBuffer(outProbeIndices, 0, 5);
        enc->setBuffer(outBuildIndices, 0, 6);
        enc->setBytes(&capacity, 4, 7);
        enc->setBytes(&probeCount, 4, 8);
        dispatch1D(enc, probeCount);
        enc->endEncoding();
        cmd->commit();
        cmd->waitUntilCompleted();
        checkGpuStatus(cmd, "joinHash_write");
        auto end = std::chrono::high_resolution_clock::now();
        KernelTimer::instance().record("hash_join_probe_write_multi", "hash_join_probe_u32", 
            std::chrono::duration<double, std::milli>(end - start).count(), probeCount);
    }
    
    return {std::move(outBuildIndices), std::move(outProbeIndices), totalPairs};
}

JoinResult GpuOps::joinHashU64(MTL::Buffer* buildKeys, 
                                        MTL::Buffer* buildIndices, 
                                        uint32_t buildCount,
                                        MTL::Buffer* probeKeys,
                                        MTL::Buffer* probeIndices,
                                        uint32_t probeCount) {
    auto& store = GpuColumnStore::instance();
    LOG_DEBUG("GPU", "joinHashU64: buildCount=" << buildCount << " probeCount=" << probeCount);
    if (buildCount == 0 || probeCount == 0 || !store.device()) return JoinResult{};

    uint32_t capacity = 1024;
    {
        uint64_t minCap = static_cast<uint64_t>(buildCount) * 2;
        while (capacity < minCap && capacity < (1u << 30)) capacity <<= 1;
    }
    LOG_DEBUG("GPU", "joinHashU64: hash table capacity=" << capacity);
    
    // Split hash table: separate buffers for low and high 32 bits of keys
    // This avoids 64-bit atomics which are not well supported on all Metal devices
    GpuBuffer bufHTKeysLow(store.device()->newBuffer(capacity * sizeof(uint32_t), MTL::ResourceStorageModeShared));
    GpuBuffer bufHTKeysHigh(store.device()->newBuffer(capacity * sizeof(uint32_t), MTL::ResourceStorageModeShared));
    GpuBuffer bufHTVals(store.device()->newBuffer(capacity * sizeof(uint32_t), MTL::ResourceStorageModeShared));
    
    // Init Keys to EMPTY (0xFFFFFFFF for both parts = 64-bit EMPTY)
    std::memset(bufHTKeysLow->contents(), 0xFF, capacity * sizeof(uint32_t));
    std::memset(bufHTKeysHigh->contents(), 0xFF, capacity * sizeof(uint32_t));
    
    auto p_build = makePSO(store.device(), store.library(), "ops::join_build_u64");
    if (!p_build) {
        return JoinResult{};
    }
    
    LOG_DEBUG("GPU", "joinHashU64: starting build phase...");
    {
        auto cmd = store.queue()->commandBuffer();
        auto enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(p_build);
        enc->setBuffer(buildKeys, 0, 0);
        enc->setBuffer(buildIndices, 0, 1);
        enc->setBuffer(bufHTKeysLow, 0, 2);   // Low 32 bits of key
        enc->setBuffer(bufHTVals, 0, 3);
        enc->setBytes(&capacity, 4, 4);
        enc->setBytes(&buildCount, 4, 5);
        enc->setBuffer(bufHTKeysHigh, 0, 6);  // High 32 bits of key
        dispatch1D(enc, buildCount);
        enc->endEncoding();
        cmd->commit();
    }
    LOG_DEBUG("GPU", "joinHashU64: build phase done.");
    
    // Deterministic probe: mark matches in mask + build map, then compact via prefix-sum
    GpuBuffer probeMask(store.device()->newBuffer(probeCount * sizeof(uint8_t), MTL::ResourceStorageModeShared));
    GpuBuffer buildMap(store.device()->newBuffer(probeCount * sizeof(uint32_t), MTL::ResourceStorageModeShared));
    
    auto p_probe = makePSO(store.device(), store.library(), "ops::join_probe_u64_mark");
    if (!p_probe) {
        return JoinResult{};
    }

    LOG_DEBUG("GPU", "joinHashU64: starting probe phase...");
    {
        auto cmd = store.queue()->commandBuffer();
        auto enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(p_probe);
        enc->setBuffer(probeKeys, 0, 0);
        enc->setBuffer(probeIndices, 0, 1);
        enc->setBuffer(bufHTKeysLow, 0, 2);
        enc->setBuffer(bufHTVals, 0, 3);
        enc->setBytes(&capacity, 4, 4);
        enc->setBytes(&probeCount, 4, 5);
        enc->setBuffer(probeMask, 0, 6);
        enc->setBuffer(buildMap, 0, 7);
        enc->setBuffer(bufHTKeysHigh, 0, 8);
        dispatch1D(enc, probeCount);
        enc->endEncoding();
        cmd->commit();
    }
    LOG_DEBUG("GPU", "joinHashU64: probe phase done.");
    
    // Compute prefix sums from mask
    auto [offsets, totalPairs] = prefixSumFromU8Mask(probeMask, probeCount);
    LOG_DEBUG("GPU", "joinHashU64: result count=" << totalPairs);
    
    if (totalPairs == 0) {
        auto emptyBuf = store.device()->newBuffer(sizeof(uint32_t), MTL::ResourceStorageModeShared);
        return {GpuBuffer(emptyBuf), GpuBuffer(store.device()->newBuffer(sizeof(uint32_t), MTL::ResourceStorageModeShared)), 0};
    }
    
    // Scatter build indices and probe indices using shared prefix sums
    GpuBuffer outBuildIndices(store.device()->newBuffer(static_cast<size_t>(totalPairs) * sizeof(uint32_t), MTL::ResourceStorageModeShared));
    GpuBuffer outProbeIndices(store.device()->newBuffer(static_cast<size_t>(totalPairs) * sizeof(uint32_t), MTL::ResourceStorageModeShared));
    
    auto p_scatter_indexed = makePSO(store.device(), store.library(), "ops::scatter_by_prefix_u8_indexed");
    auto p_scatter = makePSO(store.device(), store.library(), "ops::scatter_by_prefix_u8");
    {
        auto cmd = store.queue()->commandBuffer();
        // Scatter build indices (from buildMap)
        auto enc1 = cmd->computeCommandEncoder();
        enc1->setComputePipelineState(p_scatter_indexed);
        enc1->setBuffer(probeMask, 0, 0);
        enc1->setBuffer(buildMap, 0, 1);
        enc1->setBuffer(offsets, 0, 2);
        enc1->setBuffer(outBuildIndices, 0, 3);
        enc1->setBytes(&probeCount, 4, 4);
        dispatch1D(enc1, probeCount);
        enc1->endEncoding();
        // Scatter probe indices
        if (probeIndices) {
            auto enc2 = cmd->computeCommandEncoder();
            enc2->setComputePipelineState(p_scatter_indexed);
            enc2->setBuffer(probeMask, 0, 0);
            enc2->setBuffer(probeIndices, 0, 1);
            enc2->setBuffer(offsets, 0, 2);
            enc2->setBuffer(outProbeIndices, 0, 3);
            enc2->setBytes(&probeCount, 4, 4);
            dispatch1D(enc2, probeCount);
            enc2->endEncoding();
        } else {
            auto enc2 = cmd->computeCommandEncoder();
            enc2->setComputePipelineState(p_scatter);
            enc2->setBuffer(probeMask, 0, 0);
            enc2->setBuffer(offsets, 0, 1);
            enc2->setBuffer(outProbeIndices, 0, 2);
            enc2->setBytes(&probeCount, 4, 3);
            dispatch1D(enc2, probeCount);
            enc2->endEncoding();
        }
        cmd->commit();
        cmd->waitUntilCompleted();
        checkGpuStatus(cmd, "joinHashU64_scatter");
    }
    
    return {std::move(outBuildIndices), std::move(outProbeIndices), totalPairs};
}

GpuBuffer GpuOps::packU32ToU64(MTL::Buffer* c1, MTL::Buffer* c2, uint32_t count) {
    auto& store = GpuColumnStore::instance();
    auto p = makePSO(store.device(), store.library(), "ops::pack_u32_to_u64");
    if (!p) return {};
    auto out = store.device()->newBuffer(static_cast<NS::UInteger>(count) * 8, MTL::ResourceStorageModeShared);
    if (!out) return {};
    {
        auto cmd = store.queue()->commandBuffer();
        auto enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(p);
        enc->setBuffer(c1, 0, 0);
        enc->setBuffer(c2, 0, 1);
        enc->setBuffer(out, 0, 2);
        enc->setBytes(&count, 4, 3);
        dispatch1D(enc, count);
        enc->endEncoding();
        cmd->commit();
        cmd->waitUntilCompleted();
        checkGpuStatus(cmd, "packU32ToU64");
    }
    return GpuBuffer(out);
}

void GpuOps::crossProduct(MTL::Buffer* left, MTL::Buffer* right,
                                MTL::Buffer* outLeft, MTL::Buffer* outRight,
                                uint32_t leftCount, uint32_t rightCount) {
    auto* dev = GpuColumnStore::instance().device();
    auto* lib = GpuColumnStore::instance().library();
    auto* cmd = GpuColumnStore::instance().queue()->commandBuffer();
    auto* enc = cmd->computeCommandEncoder();
    
    auto* pso = makePSO(dev, lib, "ops::cross_product");
    if(!pso) { enc->endEncoding(); return; }
    
    enc->setComputePipelineState(pso);
    enc->setBuffer(left, 0, 0);
    enc->setBuffer(right, 0, 1);
    enc->setBuffer(outLeft, 0, 2);
    enc->setBuffer(outRight, 0, 3);
    enc->setBytes(&leftCount, sizeof(uint32_t), 4);
    enc->setBytes(&rightCount, sizeof(uint32_t), 5);
    
    uint64_t totalCount64 = static_cast<uint64_t>(leftCount) * rightCount;
    uint32_t totalCount = static_cast<uint32_t>(std::min<uint64_t>(totalCount64, UINT32_MAX));
    enc->setBytes(&totalCount, sizeof(uint32_t), 6);
    dispatch1D(enc, totalCount);
    enc->endEncoding();
    cmd->commit();
    cmd->waitUntilCompleted();
    checkGpuStatus(cmd, "crossProduct");
}

std::optional<FilterResult> GpuOps::hashJoinSemiU32(MTL::Buffer* leftKey,
                                                             uint32_t leftCount,
                                                             MTL::Buffer* rightKey,
                                                             uint32_t rightCount) {
    auto& store = GpuColumnStore::instance();
    if (!store.device() || !store.library()) return std::nullopt;

    auto p_build = makePSO(store.device(), store.library(), "ops::hash_join_build_multi");
    auto p_probe = makePSO(store.device(), store.library(), "ops::hash_join_probe_semi");
    if (!p_build || !p_probe) return std::nullopt;

    uint32_t cap = nextPow2(std::max<uint32_t>(8u, static_cast<uint32_t>(std::min<uint64_t>(static_cast<uint64_t>(rightCount) * 2u, UINT32_MAX))));
    GpuBuffer htKeys(store.device()->newBuffer(cap * sizeof(uint32_t), MTL::ResourceStorageModeShared));
    GpuBuffer ht_head(store.device()->newBuffer(cap * sizeof(uint32_t), MTL::ResourceStorageModeShared));
    GpuBuffer next(store.device()->newBuffer(static_cast<size_t>(rightCount) * sizeof(uint32_t), MTL::ResourceStorageModeShared));
    
    std::memset(htKeys->contents(), 0, cap * sizeof(uint32_t));
    std::memset(ht_head->contents(), 0, cap * sizeof(uint32_t));
    if (rightCount > 0) std::memset(next->contents(), 0, static_cast<size_t>(rightCount) * sizeof(uint32_t));

    // BUILD → PROBE in one command buffer
    GpuBuffer mask(store.device()->newBuffer(leftCount * sizeof(uint8_t), MTL::ResourceStorageModeShared));

    {
        auto cmd = store.queue()->commandBuffer();
        // Encoder 1: BUILD
        auto enc1 = cmd->computeCommandEncoder();
        enc1->setComputePipelineState(p_build);
        enc1->setBuffer(rightKey, 0, 0);
        enc1->setBuffer(htKeys, 0, 1);
        enc1->setBuffer(ht_head, 0, 2);
        enc1->setBuffer(next, 0, 3);
        enc1->setBytes(&cap, sizeof(cap), 4);
        enc1->setBytes(&rightCount, sizeof(rightCount), 5);
        dispatch1D(enc1, rightCount);
        enc1->endEncoding();
        // Encoder 2: PROBE → mask
        auto enc2 = cmd->computeCommandEncoder();
        enc2->setComputePipelineState(p_probe);
        enc2->setBuffer(leftKey, 0, 0);
        enc2->setBuffer(htKeys, 0, 1);
        enc2->setBytes(&cap, sizeof(cap), 2);
        enc2->setBytes(&leftCount, sizeof(leftCount), 3);
        enc2->setBuffer(mask, 0, 4);
        dispatch1D(enc2, leftCount);
        enc2->endEncoding();
        cmd->commit();
    }
    
    // Deterministic compaction
    auto [outIdx, validCount] = compactU8Deterministic(mask, leftCount);
    
    FilterResult res;
    res.indices = std::move(outIdx);
    res.count = validCount;
    return res;
}

std::optional<FilterResult> GpuOps::hashJoinAntiU32(MTL::Buffer* leftKey,
                                                             uint32_t leftCount,
                                                             MTL::Buffer* rightKey,
                                                             uint32_t rightCount) {
    auto& store = GpuColumnStore::instance();
    if (!store.device() || !store.library()) return std::nullopt;

    // If no right rows, every left row is unmatched
    if (rightCount == 0) {
        auto idx = iotaU32(leftCount);
        FilterResult res;
        res.indices = std::move(idx);
        res.count = leftCount;
        return res;
    }

    auto p_build   = makePSO(store.device(), store.library(), "ops::hash_join_build_multi");
    auto p_probe   = makePSO(store.device(), store.library(), "ops::hash_join_probe_semi");
    auto p_flip    = makePSO(store.device(), store.library(), "ops::flip_mask_u8");
    if (!p_build || !p_probe || !p_flip) return std::nullopt;

    uint32_t cap = nextPow2(std::max<uint32_t>(8u, static_cast<uint32_t>(std::min<uint64_t>(static_cast<uint64_t>(rightCount) * 2u, UINT32_MAX))));
    GpuBuffer htKeys(store.device()->newBuffer(cap * sizeof(uint32_t), MTL::ResourceStorageModeShared));
    GpuBuffer ht_head(store.device()->newBuffer(cap * sizeof(uint32_t), MTL::ResourceStorageModeShared));
    GpuBuffer next   (store.device()->newBuffer(static_cast<size_t>(rightCount) * sizeof(uint32_t), MTL::ResourceStorageModeShared));
    
    std::memset(htKeys->contents(), 0, cap * sizeof(uint32_t));
    std::memset(ht_head->contents(), 0, cap * sizeof(uint32_t));
    if (rightCount > 0) std::memset(next->contents(), 0, static_cast<size_t>(rightCount) * sizeof(uint32_t));

    // BUILD → PROBE → FLIP in one command buffer
    GpuBuffer mask(store.device()->newBuffer(leftCount * sizeof(uint8_t), MTL::ResourceStorageModeShared));

    {
        auto cmd = store.queue()->commandBuffer();
        // Encoder 1: BUILD
        auto enc1 = cmd->computeCommandEncoder();
        enc1->setComputePipelineState(p_build);
        enc1->setBuffer(rightKey, 0, 0);
        enc1->setBuffer(htKeys, 0, 1);
        enc1->setBuffer(ht_head, 0, 2);
        enc1->setBuffer(next, 0, 3);
        enc1->setBytes(&cap, sizeof(cap), 4);
        enc1->setBytes(&rightCount, sizeof(rightCount), 5);
        dispatch1D(enc1, rightCount);
        enc1->endEncoding();
        // Encoder 2: PROBE → mask (1 = matched)
        auto enc2 = cmd->computeCommandEncoder();
        enc2->setComputePipelineState(p_probe);
        enc2->setBuffer(leftKey, 0, 0);
        enc2->setBuffer(htKeys, 0, 1);
        enc2->setBytes(&cap, sizeof(cap), 2);
        enc2->setBytes(&leftCount, sizeof(leftCount), 3);
        enc2->setBuffer(mask, 0, 4);
        dispatch1D(enc2, leftCount);
        enc2->endEncoding();
        // Encoder 3: FLIP mask (1→0 matched, 0→1 unmatched)
        auto enc3 = cmd->computeCommandEncoder();
        enc3->setComputePipelineState(p_flip);
        enc3->setBuffer(mask, 0, 0);
        enc3->setBytes(&leftCount, sizeof(leftCount), 1);
        dispatch1D(enc3, leftCount);
        enc3->endEncoding();
        cmd->commit();
    }

    // Deterministic compaction
    auto [outIdx, validCount] = compactU8Deterministic(mask, leftCount);

    FilterResult res;
    res.indices = std::move(outIdx);
    res.count = validCount;
    return res;
}

FilterResult GpuOps::findUnmatchedIndices(MTL::Buffer* matchedIndices,
                                                    uint32_t matchedCount,
                                                    uint32_t totalRows) {
    auto& store = GpuColumnStore::instance();

    // Edge case: no matches → every row is unmatched
    if (matchedCount == 0) {
        FilterResult res;
        res.indices = GpuOps::iotaU32(totalRows);
        res.count = totalRows;
        return res;
    }
    if (totalRows == 0) {
        FilterResult res;
        res.indices.reset(store.device()->newBuffer(sizeof(uint32_t), MTL::ResourceStorageModeShared));
        res.count = 0;
        return res;
    }

    auto p_scatter = makePSO(store.device(), store.library(), "ops::scatter_one_u8");
    auto p_flip    = makePSO(store.device(), store.library(), "ops::flip_mask_u8");

    // Create u8 mask, zero-initialized
    GpuBuffer mask(store.device()->newBuffer(totalRows * sizeof(uint8_t), MTL::ResourceStorageModeShared));
    std::memset(mask->contents(), 0, totalRows * sizeof(uint8_t));

    if (p_scatter && p_flip) {
        // SCATTER → FLIP in one command buffer
        {
            auto cmd = store.queue()->commandBuffer();
            // Encoder 1: scatter 1 at matched indices
            auto enc1 = cmd->computeCommandEncoder();
            enc1->setComputePipelineState(p_scatter);
            enc1->setBuffer(matchedIndices, 0, 0);
            enc1->setBuffer(mask, 0, 1);
            enc1->setBytes(&matchedCount, sizeof(matchedCount), 2);
            dispatch1D(enc1, matchedCount);
            enc1->endEncoding();
            // Encoder 2: flip mask (1→0 matched, 0→1 unmatched)
            auto enc2 = cmd->computeCommandEncoder();
            enc2->setComputePipelineState(p_flip);
            enc2->setBuffer(mask, 0, 0);
            enc2->setBytes(&totalRows, sizeof(totalRows), 1);
            dispatch1D(enc2, totalRows);
            enc2->endEncoding();
            cmd->commit();
        }
        // Deterministic compaction
        auto [outIdx, cnt] = compactU8Deterministic(mask, totalRows);
        FilterResult res;
        res.indices = std::move(outIdx);
        res.count = cnt;
        return res;
    }

    // CPU fallback
    uint8_t* maskPtr = static_cast<uint8_t*>(mask->contents());
    uint32_t* matchPtr = static_cast<uint32_t*>(matchedIndices->contents());
    for (uint32_t i = 0; i < matchedCount; ++i) {
        if (matchPtr[i] < totalRows) maskPtr[matchPtr[i]] = 1;
    }
    std::vector<uint32_t> result;
    for (uint32_t i = 0; i < totalRows; ++i) {
        if (!maskPtr[i]) result.push_back(i);
    }
    uint32_t cnt = static_cast<uint32_t>(result.size());
    GpuBuffer outIdx(store.device()->newBuffer(
        result.empty() ? sizeof(uint32_t) : result.size() * sizeof(uint32_t),
        MTL::ResourceStorageModeShared));
    if (!result.empty()) std::memcpy(outIdx->contents(), result.data(), result.size() * sizeof(uint32_t));
    FilterResult fRes;
    fRes.indices = std::move(outIdx);
    fRes.count = cnt;
    return fRes;
}


} // namespace engine
