#include "Operators.hpp"
#include "OperatorsInternal.hpp"
#include "KernelTimer.hpp"
#include "EngineConfig.hpp"

#include <cstring>
#include <iostream>

namespace engine {

GpuBuffer GpuOps::floatToSortKeyU32(MTL::Buffer* in, uint32_t count, bool desc) {
    auto& store = GpuColumnStore::instance();
    auto out = store.device()->newBuffer(count * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    auto p = makePSO(store.device(), store.library(), "ops::float_to_sort_key_u32");
    if (!p) {
        // CPU fallback
        const float* src = (const float*)in->contents();
        uint32_t* dst = static_cast<uint32_t*>(out->contents());
        for (uint32_t i = 0; i < count; ++i) {
            uint32_t bits;
            std::memcpy(&bits, &src[i], sizeof(bits));
            if (bits & 0x80000000u) bits = ~bits;
            else bits ^= 0x80000000u;
            dst[i] = desc ? ~bits : bits;
        }
        return GpuBuffer(out);
    }
    uint32_t descFlag = desc ? 1 : 0;
    auto cmd = store.queue()->commandBuffer();
    auto enc = cmd->computeCommandEncoder();
    enc->setComputePipelineState(p);
    enc->setBuffer(in, 0, 0);
    enc->setBuffer(out, 0, 1);
    enc->setBytes(&count, sizeof(count), 2);
    enc->setBytes(&descFlag, sizeof(descFlag), 3);
    dispatch1D(enc, count);
    enc->endEncoding();
    cmd->commit();
    cmd->waitUntilCompleted();
    checkGpuStatus(cmd);
    return GpuBuffer(out);
}

GpuBuffer GpuOps::invertU32(MTL::Buffer* in, uint32_t count) {
    auto& store = GpuColumnStore::instance();
    auto out = store.device()->newBuffer(count * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    auto p = makePSO(store.device(), store.library(), "ops::invert_u32");
    if (!p) {
        const uint32_t* src = static_cast<const uint32_t*>(in->contents());
        uint32_t* dst = static_cast<uint32_t*>(out->contents());
        for (uint32_t i = 0; i < count; ++i) dst[i] = ~src[i];
        return GpuBuffer(out);
    }
    auto cmd = store.queue()->commandBuffer();
    auto enc = cmd->computeCommandEncoder();
    enc->setComputePipelineState(p);
    enc->setBuffer(in, 0, 0);
    enc->setBuffer(out, 0, 1);
    enc->setBytes(&count, sizeof(count), 2);
    dispatch1D(enc, count);
    enc->endEncoding();
    cmd->commit();
    cmd->waitUntilCompleted();
    checkGpuStatus(cmd);
    return GpuBuffer(out);
}

static void blockSortU32(MTL::Buffer* keys, MTL::Buffer* indices, uint32_t count) {
    auto& store = GpuColumnStore::instance();
    auto p = makePSO(store.device(), store.library(), "ops::block_sort_kv_u32");
    if (!p) return;

    uint32_t tg = 1;
    while (tg < count) tg <<= 1;
    if (tg > 1024) tg = 1024; // safety cap

    auto cmd = store.queue()->commandBuffer();
    auto enc = cmd->computeCommandEncoder();
    enc->setComputePipelineState(p);
    enc->setBuffer(keys, 0, 0);
    enc->setBuffer(indices, 0, 1);
    enc->setBytes(&count, sizeof(count), 2);
    enc->dispatchThreads(MTL::Size::Make(tg, 1, 1), MTL::Size::Make(tg, 1, 1));
    enc->endEncoding();
    cmd->commit();
    cmd->waitUntilCompleted();
    checkGpuStatus(cmd);
}

static void blockSortU64(MTL::Buffer* keys, MTL::Buffer* indices, uint32_t count) {
    auto& store = GpuColumnStore::instance();
    auto p = makePSO(store.device(), store.library(), "ops::block_sort_kv_u64");
    if (!p) return;

    uint32_t tg = 1;
    while (tg < count) tg <<= 1;
    if (tg > 1024) tg = 1024;

    auto cmd = store.queue()->commandBuffer();
    auto enc = cmd->computeCommandEncoder();
    enc->setComputePipelineState(p);
    enc->setBuffer(keys, 0, 0);
    enc->setBuffer(indices, 0, 1);
    enc->setBytes(&count, sizeof(count), 2);
    enc->dispatchThreads(MTL::Size::Make(tg, 1, 1), MTL::Size::Make(tg, 1, 1));
    enc->endEncoding();
    cmd->commit();
    cmd->waitUntilCompleted();
    checkGpuStatus(cmd);
}

void GpuOps::radixSortU32(MTL::Buffer* keys, MTL::Buffer* indices, uint32_t count) {
    if (count <= 1) return;

    if (count <= engine::config::kBlockSortThreshold) {
        blockSortU32(keys, indices, count);
        KernelTimer::instance().record("block_sort_kv_u32", "sort", 0, count);
        return;
    }

    auto& store = GpuColumnStore::instance();
    auto* dev = store.device();

    constexpr uint32_t BLK = 256;
    uint32_t numBlocks = (count + BLK - 1) / BLK;
    uint32_t histSize  = 256 * numBlocks;

    auto* keysAlt = dev->newBuffer(count * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    auto* valsAlt = dev->newBuffer(count * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    auto* histBuf = dev->newBuffer(histSize * sizeof(uint32_t), MTL::ResourceStorageModeShared);

    auto p_hist    = makePSO(dev, store.library(), "ops::radix_histogram_u32");
    auto p_scatter = makePSO(dev, store.library(), "ops::radix_scatter_u32");
    if (!p_hist || !p_scatter) { keysAlt->release(); valsAlt->release(); histBuf->release(); return; }

    MTL::Buffer* srcK = keys;
    MTL::Buffer* srcV = indices;
    MTL::Buffer* dstK = keysAlt;
    MTL::Buffer* dstV = valsAlt;

    for (uint32_t pass = 0; pass < 4; ++pass) {
        uint32_t shift = pass * 8;

        // Histogram
        {
            auto cmd = store.queue()->commandBuffer();
            auto enc = cmd->computeCommandEncoder();
            enc->setComputePipelineState(p_hist);
            enc->setBuffer(srcK, 0, 0);
            enc->setBuffer(histBuf, 0, 1);
            enc->setBytes(&count, sizeof(count), 2);
            enc->setBytes(&shift, sizeof(shift), 3);
            enc->setBytes(&numBlocks, sizeof(numBlocks), 4);
            enc->dispatchThreadgroups(MTL::Size::Make(numBlocks, 1, 1),
                                      MTL::Size::Make(BLK, 1, 1));
            enc->endEncoding();
            cmd->commit();
        }

        // Prefix sum (async — no GPU sync)
        scanInPlaceAsync(histBuf, histSize);

        // Scatter (commit without wait — serial queue guarantees ordering)
        {
            auto cmd = store.queue()->commandBuffer();
            auto enc = cmd->computeCommandEncoder();
            enc->setComputePipelineState(p_scatter);
            enc->setBuffer(srcK, 0, 0);
            enc->setBuffer(srcV, 0, 1);
            enc->setBuffer(dstK, 0, 2);
            enc->setBuffer(dstV, 0, 3);
            enc->setBuffer(histBuf, 0, 4);
            enc->setBytes(&count, sizeof(count), 5);
            enc->setBytes(&shift, sizeof(shift), 6);
            enc->setBytes(&numBlocks, sizeof(numBlocks), 7);
            enc->dispatchThreadgroups(MTL::Size::Make(numBlocks, 1, 1),
                                      MTL::Size::Make(BLK, 1, 1));
            enc->endEncoding();
            cmd->commit();
        }

        std::swap(srcK, dstK);
        std::swap(srcV, dstV);
    }

    // Single GPU sync point after all passes (was 8 syncs before)
    {
        auto cmd = store.queue()->commandBuffer();
        cmd->commit();
        cmd->waitUntilCompleted();
        checkGpuStatus(cmd);
    }
    // After 4 passes (even), result is back in original (keys, indices) buffers.

    keysAlt->release();
    valsAlt->release();
    histBuf->release();

    KernelTimer::instance().record("radix_sort_u32", "sort", 0, count);
}

void GpuOps::radixSortU64(MTL::Buffer* keys, MTL::Buffer* indices, uint32_t count) {
    if (count <= 1) return;

    if (count <= engine::config::kBlockSortThreshold) {
        blockSortU64(keys, indices, count);
        KernelTimer::instance().record("block_sort_kv_u64", "sort", 0, count);
        return;
    }

    auto& store = GpuColumnStore::instance();
    auto* dev = store.device();

    constexpr uint32_t BLK = 256;
    uint32_t numBlocks = (count + BLK - 1) / BLK;
    uint32_t histSize  = 256 * numBlocks;

    auto* keysAlt = dev->newBuffer(count * sizeof(uint64_t), MTL::ResourceStorageModeShared);
    auto* valsAlt = dev->newBuffer(count * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    auto* histBuf = dev->newBuffer(histSize * sizeof(uint32_t), MTL::ResourceStorageModeShared);

    auto p_hist    = makePSO(dev, store.library(), "ops::radix_histogram_u64");
    auto p_scatter = makePSO(dev, store.library(), "ops::radix_scatter_u64");
    if (!p_hist || !p_scatter) { keysAlt->release(); valsAlt->release(); histBuf->release(); return; }

    MTL::Buffer* srcK = keys;
    MTL::Buffer* srcV = indices;
    MTL::Buffer* dstK = keysAlt;
    MTL::Buffer* dstV = valsAlt;

    for (uint32_t pass = 0; pass < 8; ++pass) {
        uint32_t shift = pass * 8;

        {
            auto cmd = store.queue()->commandBuffer();
            auto enc = cmd->computeCommandEncoder();
            enc->setComputePipelineState(p_hist);
            enc->setBuffer(srcK, 0, 0);
            enc->setBuffer(histBuf, 0, 1);
            enc->setBytes(&count, sizeof(count), 2);
            enc->setBytes(&shift, sizeof(shift), 3);
            enc->setBytes(&numBlocks, sizeof(numBlocks), 4);
            enc->dispatchThreadgroups(MTL::Size::Make(numBlocks, 1, 1),
                                      MTL::Size::Make(BLK, 1, 1));
            enc->endEncoding();
            cmd->commit();
        }

        scanInPlaceAsync(histBuf, histSize);

        {
            auto cmd = store.queue()->commandBuffer();
            auto enc = cmd->computeCommandEncoder();
            enc->setComputePipelineState(p_scatter);
            enc->setBuffer(srcK, 0, 0);
            enc->setBuffer(srcV, 0, 1);
            enc->setBuffer(dstK, 0, 2);
            enc->setBuffer(dstV, 0, 3);
            enc->setBuffer(histBuf, 0, 4);
            enc->setBytes(&count, sizeof(count), 5);
            enc->setBytes(&shift, sizeof(shift), 6);
            enc->setBytes(&numBlocks, sizeof(numBlocks), 7);
            enc->dispatchThreadgroups(MTL::Size::Make(numBlocks, 1, 1),
                                      MTL::Size::Make(BLK, 1, 1));
            enc->endEncoding();
            cmd->commit();
        }

        std::swap(srcK, dstK);
        std::swap(srcV, dstV);
    }

    // Single GPU sync point after all passes (was 16 syncs before)
    {
        auto cmd = store.queue()->commandBuffer();
        cmd->commit();
        cmd->waitUntilCompleted();
        checkGpuStatus(cmd);
    }
    // After 8 passes (even), result is back in original (keys, indices) buffers.

    keysAlt->release();
    valsAlt->release();
    histBuf->release();

    KernelTimer::instance().record("radix_sort_u64", "sort", 0, count);
}

} // namespace engine
