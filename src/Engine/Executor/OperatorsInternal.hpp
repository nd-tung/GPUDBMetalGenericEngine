#pragma once
// Shared infrastructure for GPU operator implementations.
// These inline helpers are used across FilterOps, JoinOps, SortOps, GroupByOps.

#include "GpuColumnStore.hpp"
#include <unordered_map>
#include <iostream>
#include <cstring>
#include <mutex>

namespace engine {

// Check GPU command buffer status after waitUntilCompleted().
// Logs error details to stderr if the command buffer failed.
// Returns true if the command completed successfully.
inline bool checkGpuStatus(MTL::CommandBuffer* cmd, const char* context = nullptr) {
    if (cmd->status() == MTL::CommandBufferStatusError) {
        std::cerr << "[GPU] Command buffer error";
        if (context) std::cerr << " in " << context;
        auto err = cmd->error();
        if (err) std::cerr << ": " << err->localizedDescription()->utf8String();
        std::cerr << "\n";
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
        std::cerr << "[GPU] function not found: " << fn << "\n";
        return nullptr;
    }
    auto pso = dev->newComputePipelineState(f, &error);
    f->release();
    if (!pso) {
        std::cerr << "[GPU] Failed to create PSO for " << fn << "\n";
        if (error) {
            std::cerr << "[GPU] pipeline error for " << fn << ": " << error->localizedDescription()->utf8String() << "\n";
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
        cmd->waitUntilCompleted();
    }
    
    uint64_t totalSum = 0;
    if (blocks > 1) {
        totalSum = scanInPlace(partials, blocks);
        
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
            cmd->waitUntilCompleted();
        }
    } else {
        // Cached 4-byte readBuf — avoids repeated alloc/release for single-block scans
        static thread_local MTL::Buffer* s_readBuf = nullptr;
        if (!s_readBuf) {
            s_readBuf = store.device()->newBuffer(sizeof(uint32_t), MTL::ResourceStorageModeShared);
        }
        auto cmd = store.queue()->commandBuffer();
        auto blit = cmd->blitCommandEncoder();
        blit->copyFromBuffer(partials, 0, s_readBuf, 0, sizeof(uint32_t));
        blit->endEncoding();
        cmd->commit();
        cmd->waitUntilCompleted();
        uint32_t val;
        std::memcpy(&val, s_readBuf->contents(), sizeof(uint32_t));
        totalSum = val;
    }
    
    partials->release();
    return totalSum;
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

} // namespace engine
