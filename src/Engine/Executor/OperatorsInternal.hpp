#pragma once
// Shared infrastructure for GPU operator implementations.
// These inline helpers are used across FilterOps, JoinOps, SortOps, GroupByOps.

#include "GpuColumnStore.hpp"
#include "GpuBuffer.hpp"
#include <unordered_map>
#include <iostream>
#include <cstring>
#include <mutex>
#include "Logger.hpp"

namespace engine {

// Check GPU command buffer status after waitUntilCompleted().
// Logs error details to stderr if the command buffer failed.
// Returns true if the command completed successfully.
inline bool checkGpuStatus(MTL::CommandBuffer* cmd, const char* context = nullptr) {
    if (cmd->status() == MTL::CommandBufferStatusError) {
        auto err = cmd->error();
        if (context && err)
            LOG_ERROR("GPU", "Command buffer error in " << context << ": " << err->localizedDescription()->utf8String());
        else if (context)
            LOG_ERROR("GPU", "Command buffer error in " << context);
        else if (err)
            LOG_ERROR("GPU", "Command buffer error: " << err->localizedDescription()->utf8String());
        else
            LOG_ERROR("GPU", "Command buffer error");
        return false;
    }
    return true;
}

inline MTL::ComputePipelineState* makePSO(MTL::Device* dev, MTL::Library* lib, const char* fn) {
    // Cache PSOs for the lifetime of the process to avoid repeated compilation.
    // Returned PSOs are owned by the cache; callers must NOT release them.
    static std::unordered_map<std::string, MTL::ComputePipelineState*> cache;
    static std::mutex cacheMutex;

    {
        std::lock_guard<std::mutex> lock(cacheMutex);
        auto it = cache.find(fn);
        if (it != cache.end()) return it->second;
    }

    auto name = NS::String::alloc()->init(fn, NS::UTF8StringEncoding);
    NS::Error* error = nullptr;
    MTL::Function* f = lib->newFunction(name);
    name->release();
    if (!f) {
        LOG_ERROR("GPU", "function not found: " << fn);
        return nullptr;
    }
    auto pso = dev->newComputePipelineState(f, &error);
    f->release();
    if (!pso) {
        LOG_ERROR("GPU", "Failed to create PSO for " << fn);
        if (error) {
            LOG_ERROR("GPU", "pipeline error for " << fn << ": " << error->localizedDescription()->utf8String());
        }
        return nullptr;
    }

    {
        std::lock_guard<std::mutex> lock(cacheMutex);
        cache.emplace(std::string(fn), pso);
    }
    return pso;
}

inline uint64_t scanInPlace(MTL::Buffer* data, uint32_t count) {
    if (count == 0 || !data) return 0;
    auto& store = GpuColumnStore::instance();
    auto lib = store.library();
    auto p_scan = makePSO(store.device(), lib, "ops::scan_exclusive_subblock_u32");
    auto p_add = makePSO(store.device(), lib, "ops::scan_add_base_u32");
    if (!p_scan || !p_add) return 0; 

    uint32_t blockSize = 256;
    uint32_t blocks = (count + blockSize - 1) / blockSize;

    auto partials = store.device()->newBuffer(blocks * sizeof(uint32_t), MTL::ResourceStorageModePrivate);
    if (!partials) return 0;

    if (blocks > 1) {
        // Sub-block scan — commit without wait (serial queue ordering)
        {
            auto cmd = store.queue()->commandBuffer();
            auto enc = cmd->computeCommandEncoder();
            enc->setComputePipelineState(p_scan);
            enc->setBuffer(data, 0, 0);
            enc->setBuffer(partials, 0, 1);
            enc->setBytes(&count, sizeof(count), 2);
            enc->dispatchThreadgroups(MTL::Size::Make(blocks, 1, 1), MTL::Size::Make(blockSize, 1, 1));
            enc->endEncoding();
            cmd->commit();
        }
        
        uint64_t totalSum = scanInPlace(partials, blocks);
        
        // Add base — commit without wait (callers use data on GPU via serial queue)
        {
            auto cmd = store.queue()->commandBuffer();
            auto enc = cmd->computeCommandEncoder();
            enc->setComputePipelineState(p_add);
            enc->setBuffer(data, 0, 0);
            enc->setBuffer(partials, 0, 1);
            enc->setBytes(&count, sizeof(count), 2);
            enc->dispatchThreadgroups(MTL::Size::Make(blocks, 1, 1), MTL::Size::Make(blockSize, 1, 1));
            enc->endEncoding();
            cmd->commit();
        }
        
        partials->release();
        return totalSum;
    } else {
        // Single-block: combine sub-block scan + blit in one command buffer
        static thread_local GpuBuffer s_readBuf;
        if (!s_readBuf) {
            s_readBuf.reset(store.device()->newBuffer(sizeof(uint32_t), MTL::ResourceStorageModeShared));
        }
        auto cmd = store.queue()->commandBuffer();
        auto enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(p_scan);
        enc->setBuffer(data, 0, 0);
        enc->setBuffer(partials, 0, 1);
        enc->setBytes(&count, sizeof(count), 2);
        enc->dispatchThreadgroups(MTL::Size::Make(blocks, 1, 1), MTL::Size::Make(blockSize, 1, 1));
        enc->endEncoding();
        auto blit = cmd->blitCommandEncoder();
        blit->copyFromBuffer(partials, 0, s_readBuf, 0, sizeof(uint32_t));
        blit->endEncoding();
        cmd->commit();
        cmd->waitUntilCompleted();
        uint32_t val;
        std::memcpy(&val, s_readBuf->contents(), sizeof(uint32_t));
        partials->release();
        return val;
    }
}

// Async variant: dispatches prefix-sum without waiting or reading back the total.
// Use when the total sum is not needed (e.g. radix sort histograms).
inline void scanInPlaceAsync(MTL::Buffer* data, uint32_t count) {
    if (count == 0 || !data) return;
    auto& store = GpuColumnStore::instance();
    auto lib = store.library();
    auto p_scan = makePSO(store.device(), lib, "ops::scan_exclusive_subblock_u32");
    auto p_add  = makePSO(store.device(), lib, "ops::scan_add_base_u32");
    if (!p_scan || !p_add) return;

    uint32_t blockSize = 256;
    uint32_t blocks = (count + blockSize - 1) / blockSize;

    auto partials = store.device()->newBuffer(blocks * sizeof(uint32_t),
                                               MTL::ResourceStorageModePrivate);
    if (blocks > 1) {
        {
            auto cmd = store.queue()->commandBuffer();
            auto enc = cmd->computeCommandEncoder();
            enc->setComputePipelineState(p_scan);
            enc->setBuffer(data, 0, 0);
            enc->setBuffer(partials, 0, 1);
            enc->setBytes(&count, sizeof(count), 2);
            enc->dispatchThreadgroups(MTL::Size::Make(blocks, 1, 1),
                                      MTL::Size::Make(blockSize, 1, 1));
            enc->endEncoding();
            cmd->commit();
        }

        scanInPlaceAsync(partials, blocks);

        {
            auto cmd = store.queue()->commandBuffer();
            auto enc = cmd->computeCommandEncoder();
            enc->setComputePipelineState(p_add);
            enc->setBuffer(data, 0, 0);
            enc->setBuffer(partials, 0, 1);
            enc->setBytes(&count, sizeof(count), 2);
            enc->dispatchThreadgroups(MTL::Size::Make(blocks, 1, 1),
                                      MTL::Size::Make(blockSize, 1, 1));
            enc->endEncoding();
            cmd->commit();
        }

        partials->release();
    } else {
        auto cmd = store.queue()->commandBuffer();
        auto enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(p_scan);
        enc->setBuffer(data, 0, 0);
        enc->setBuffer(partials, 0, 1);
        enc->setBytes(&count, sizeof(count), 2);
        enc->dispatchThreadgroups(MTL::Size::Make(1, 1, 1),
                                  MTL::Size::Make(blockSize, 1, 1));
        enc->endEncoding();
        cmd->commit();
        partials->release();  // safe: command buffer retains the reference
    }
}

inline void dispatch1D(MTL::ComputeCommandEncoder* enc, uint32_t count) {
    const uint32_t tg = 256;
    MTL::Size grid = MTL::Size::Make(count, 1, 1);
    MTL::Size tgsz = MTL::Size::Make(tg, 1, 1);
    enc->dispatchThreads(grid, tgsz);
}

inline uint32_t nextPow2(uint32_t v) {
    if (v == 0) return 1;
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    return v + 1;
}

// ============================================================================
// Deterministic compaction helpers (prefix-sum based, replaces atomic compact)
// ============================================================================

// Compute exclusive prefix sums from a u8 mask, returns {offsets buffer, total count}
inline std::pair<GpuBuffer, uint32_t> prefixSumFromU8Mask(MTL::Buffer* mask, uint32_t count) {
    auto& store = GpuColumnStore::instance();
    if (count == 0) {
        return {GpuBuffer(store.device()->newBuffer(sizeof(uint32_t), MTL::ResourceStorageModeShared)), 0};
    }
    GpuBuffer offsets(store.device()->newBuffer(static_cast<size_t>(count) * sizeof(uint32_t),
                                                MTL::ResourceStorageModeShared));
    auto p = makePSO(store.device(), store.library(), "ops::mask_to_offsets_u8");
    if (!p) return {std::move(offsets), 0};
    {
        auto cmd = store.queue()->commandBuffer();
        auto enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(p);
        enc->setBuffer(mask, 0, 0);
        enc->setBuffer(offsets, 0, 1);
        enc->setBytes(&count, 4, 2);
        dispatch1D(enc, count);
        enc->endEncoding();
        cmd->commit();
    }
    uint64_t total = scanInPlace(offsets, count);
    return {std::move(offsets), static_cast<uint32_t>(total)};
}

// Compact u8 mask → deterministic index array
inline std::pair<GpuBuffer, uint32_t> compactU8Deterministic(MTL::Buffer* mask, uint32_t count) {
    auto& store = GpuColumnStore::instance();
    auto [offsets, total] = prefixSumFromU8Mask(mask, count);
    if (total == 0) {
        return {GpuBuffer(store.device()->newBuffer(sizeof(uint32_t), MTL::ResourceStorageModeShared)), 0};
    }
    GpuBuffer out(store.device()->newBuffer(static_cast<size_t>(total) * sizeof(uint32_t),
                                            MTL::ResourceStorageModeShared));
    auto p = makePSO(store.device(), store.library(), "ops::scatter_by_prefix_u8");
    if (!p) return {GpuBuffer(), 0};
    {
        auto cmd = store.queue()->commandBuffer();
        auto enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(p);
        enc->setBuffer(mask, 0, 0);
        enc->setBuffer(offsets, 0, 1);
        enc->setBuffer(out, 0, 2);
        enc->setBytes(&count, 4, 3);
        dispatch1D(enc, count);
        enc->endEncoding();
        cmd->commit();
        cmd->waitUntilCompleted();
    }
    return {std::move(out), total};
}

// Compact u8 mask + existing indices → deterministic index array
inline std::pair<GpuBuffer, uint32_t> compactU8DeterministicIndexed(MTL::Buffer* mask,
                                                                     MTL::Buffer* inIndices,
                                                                     uint32_t count) {
    auto& store = GpuColumnStore::instance();
    auto [offsets, total] = prefixSumFromU8Mask(mask, count);
    if (total == 0) {
        return {GpuBuffer(store.device()->newBuffer(sizeof(uint32_t), MTL::ResourceStorageModeShared)), 0};
    }
    GpuBuffer out(store.device()->newBuffer(static_cast<size_t>(total) * sizeof(uint32_t),
                                            MTL::ResourceStorageModeShared));
    auto p = makePSO(store.device(), store.library(), "ops::scatter_by_prefix_u8_indexed");
    if (!p) return {GpuBuffer(), 0};
    {
        auto cmd = store.queue()->commandBuffer();
        auto enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(p);
        enc->setBuffer(mask, 0, 0);
        enc->setBuffer(inIndices, 0, 1);
        enc->setBuffer(offsets, 0, 2);
        enc->setBuffer(out, 0, 3);
        enc->setBytes(&count, 4, 4);
        dispatch1D(enc, count);
        enc->endEncoding();
        cmd->commit();
        cmd->waitUntilCompleted();
    }
    return {std::move(out), total};
}

// Compact u32 mask → deterministic index array
inline std::pair<GpuBuffer, uint32_t> compactU32Deterministic(MTL::Buffer* mask, uint32_t count) {
    auto& store = GpuColumnStore::instance();
    if (count == 0) {
        return {GpuBuffer(store.device()->newBuffer(sizeof(uint32_t), MTL::ResourceStorageModeShared)), 0};
    }
    GpuBuffer offsets(store.device()->newBuffer(static_cast<size_t>(count) * sizeof(uint32_t),
                                                MTL::ResourceStorageModeShared));
    auto pMask = makePSO(store.device(), store.library(), "ops::mask_to_offsets_u32");
    if (!pMask) return {GpuBuffer(), 0};
    {
        auto cmd = store.queue()->commandBuffer();
        auto enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(pMask);
        enc->setBuffer(mask, 0, 0);
        enc->setBuffer(offsets, 0, 1);
        enc->setBytes(&count, 4, 2);
        dispatch1D(enc, count);
        enc->endEncoding();
        cmd->commit();
    }
    uint64_t total64 = scanInPlace(offsets, count);
    uint32_t total = static_cast<uint32_t>(total64);
    if (total == 0) {
        return {GpuBuffer(store.device()->newBuffer(sizeof(uint32_t), MTL::ResourceStorageModeShared)), 0};
    }
    GpuBuffer out(store.device()->newBuffer(static_cast<size_t>(total) * sizeof(uint32_t),
                                            MTL::ResourceStorageModeShared));
    auto pScatter = makePSO(store.device(), store.library(), "ops::scatter_by_prefix_u32");
    if (!pScatter) return {GpuBuffer(), 0};
    {
        auto cmd = store.queue()->commandBuffer();
        auto enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(pScatter);
        enc->setBuffer(mask, 0, 0);
        enc->setBuffer(offsets, 0, 1);
        enc->setBuffer(out, 0, 2);
        enc->setBytes(&count, 4, 3);
        dispatch1D(enc, count);
        enc->endEncoding();
        cmd->commit();
        cmd->waitUntilCompleted();
    }
    return {std::move(out), total};
}

} // namespace engine
