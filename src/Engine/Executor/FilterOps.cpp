#include "Operators.hpp"
#include "OperatorsInternal.hpp"
#include "EnvUtil.hpp"
#include "KernelTimer.hpp"

#include <cstring>
#include <iostream>

namespace engine {

std::optional<FilterResult> GpuOps::filterU32(const std::string& colName,
                                                      MTL::Buffer* col,
                                                      uint32_t rowCount,
                                                      engine::GpuFilterOp op,
                                                      uint32_t literal) {
    auto& store = GpuColumnStore::instance();
    if (!store.device() || !store.library() || !store.queue()) return std::nullopt;

    if (env_truthy("GPUDB_DEBUG_OPS")) {
        std::cerr << "[Exec] GPU filterU32: col=" << colName << " rowCount=" << rowCount << " val=" << literal << "\n";
    }

    const char* fn = nullptr;
    switch (op) {
        case engine::GpuFilterOp::EQ: fn = "ops::filter_eq_u32"; break;
        case engine::GpuFilterOp::LT: fn = "ops::filter_lt_u32"; break;
        case engine::GpuFilterOp::GT: fn = "ops::filter_gt_u32"; break;
        case engine::GpuFilterOp::LE: fn = "ops::filter_le_u32"; break;
        case engine::GpuFilterOp::GE: fn = "ops::filter_ge_u32"; break;
        case engine::GpuFilterOp::NE: fn = "ops::filter_ne_u32"; break;
        default: break;
    }

    auto p_filter = makePSO(store.device(), store.library(), fn);
    auto p_compact = makePSO(store.device(), store.library(), "ops::compact_indices");
    if (!p_filter || !p_compact) {
        return std::nullopt;
    }

    auto mask = store.device()->newBuffer(rowCount * sizeof(uint8_t), MTL::ResourceStorageModeShared);
    std::memset(mask->contents(), 0, rowCount * sizeof(uint8_t));

    auto outIdx = store.device()->newBuffer(rowCount * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    auto outCnt = store.device()->newBuffer(sizeof(uint32_t), MTL::ResourceStorageModeShared);
    std::memset(outIdx->contents(), 0, rowCount * sizeof(uint32_t));
    std::memset(outCnt->contents(), 0, sizeof(uint32_t));

    auto filterStart = std::chrono::high_resolution_clock::now();
    {
        auto cmd = store.queue()->commandBuffer();
        // Encoder 1: filter kernel → mask
        auto enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(p_filter);
        enc->setBuffer(col, 0, 0);
        enc->setBuffer(mask, 0, 1);
        enc->setBytes(&literal, sizeof(literal), 2);
        if (op != engine::GpuFilterOp::EQ) {
            enc->setBytes(&rowCount, sizeof(rowCount), 3);
        }
        dispatch1D(enc, rowCount);
        enc->endEncoding();
        // Encoder 2: compact mask → indices (same cmd buffer, sequential execution)
        auto enc2 = cmd->computeCommandEncoder();
        enc2->setComputePipelineState(p_compact);
        enc2->setBuffer(mask, 0, 0);
        enc2->setBuffer(outIdx, 0, 1);
        enc2->setBuffer(outCnt, 0, 2);
        enc2->setBytes(&rowCount, sizeof(rowCount), 3);
        dispatch1D(enc2, rowCount);
        enc2->endEncoding();
        cmd->commit();
        cmd->waitUntilCompleted();
        checkGpuStatus(cmd);
    }
    auto filterEnd = std::chrono::high_resolution_clock::now();
    KernelTimer::instance().record(fn, "filter",
        std::chrono::duration<double, std::milli>(filterEnd - filterStart).count(), rowCount);

    FilterResult res;
    res.indices.reset(outIdx);
    res.count = *reinterpret_cast<uint32_t*>(outCnt->contents());

    mask->release();
    outCnt->release();
    return res;
}

std::optional<FilterResult> GpuOps::filterString(const std::string& /*colName*/,
                                                          const std::vector<std::string>& data,
                                                          engine::GpuFilterOp op,
                                                          const std::string& pattern,
                                                          MTL::Buffer* preChars,
                                                          MTL::Buffer* preOffsets,
                                                          MTL::Buffer* preLengths) {
    auto& store = GpuColumnStore::instance();
    if (!store.device() || !store.library() || !store.queue()) return std::nullopt;

    // 1. Prepare data
    size_t rowCount = data.size();
    if (rowCount == 0) return FilterResult{};
    if (env_truthy("GPUDB_DEBUG_OPS")) {
        std::cerr << "[Exec] GPU filterString: rowCount=" << rowCount << " pattern=" << pattern << "\n";
    }
    
    // Use pre-flattened GPU buffers (always built at scan time, rebuilt after join)
    MTL::Buffer* bufChars   = preChars;
    MTL::Buffer* bufOffsets = preOffsets;
    MTL::Buffer* bufLengths = preLengths;
    bool ownBufs = (preChars == nullptr);

    if (ownBufs) {
        // Fallback: flatten on-the-fly (should be rare after join/project rebuild)
        std::vector<uint32_t> offsets(rowCount), lengths(rowCount);
        size_t totalChars = 0;
        for (const auto& s : data) totalChars += s.size();
        std::vector<char> chars;
        chars.reserve(totalChars);
        size_t cur = 0;
        for (size_t i = 0; i < rowCount; ++i) {
            offsets[i] = static_cast<uint32_t>(cur);
            lengths[i] = static_cast<uint32_t>(data[i].size());
            chars.insert(chars.end(), data[i].begin(), data[i].end());
            cur += data[i].size();
        }
        bufChars = chars.empty()
            ? store.device()->newBuffer(1, MTL::ResourceStorageModeShared)
            : store.device()->newBuffer(chars.data(), chars.size(), MTL::ResourceStorageModeShared);
        bufOffsets = store.device()->newBuffer(offsets.data(), offsets.size() * sizeof(uint32_t), MTL::ResourceStorageModeShared);
        bufLengths = store.device()->newBuffer(lengths.data(), lengths.size() * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    }
    
    // 3. Process Pattern — detect multi-wildcard LIKE (e.g. %Customer%Complaints%)
    std::string rawPattern = pattern;
    if (rawPattern.size() > 0 && rawPattern.front() == '%') rawPattern.erase(0, 1);
    if (rawPattern.size() > 0 && rawPattern.back() == '%') rawPattern.pop_back();

    // Split by '%' to detect multi-segment patterns
    std::vector<std::string> segments;
    {
        std::string seg;
        for (char c : rawPattern) {
            if (c == '%') {
                if (!seg.empty()) { segments.push_back(seg); seg.clear(); }
            } else {
                seg += c;
            }
        }
        if (!seg.empty()) segments.push_back(seg);
    }

    bool useMultiContains = (segments.size() > 1);
    
    // Build GPU buffers for pattern(s)
    MTL::Buffer* bufPattern = nullptr;
    MTL::Buffer* bufPatOffsets = nullptr;
    MTL::Buffer* bufPatLengths = nullptr;
    uint32_t patternLen = 0;
    uint32_t numSegments = static_cast<uint32_t>(segments.size());

    if (useMultiContains) {
        // Pack all segments into one buffer with offset/length arrays
        std::vector<char> packedPat;
        std::vector<uint32_t> patOffsets, patLens;
        for (const auto& seg : segments) {
            patOffsets.push_back(static_cast<uint32_t>(packedPat.size()));
            patLens.push_back(static_cast<uint32_t>(seg.size()));
            packedPat.insert(packedPat.end(), seg.begin(), seg.end());
        }
        bufPattern = store.device()->newBuffer(packedPat.data(), packedPat.size(), MTL::ResourceStorageModeShared);
        bufPatOffsets = store.device()->newBuffer(patOffsets.data(), patOffsets.size() * sizeof(uint32_t), MTL::ResourceStorageModeShared);
        bufPatLengths = store.device()->newBuffer(patLens.data(), patLens.size() * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    } else {
        // Single segment (or empty)
        std::string singlePat = segments.empty() ? "" : segments[0];
        patternLen = static_cast<uint32_t>(singlePat.size());
        if (patternLen > 0) {
            bufPattern = store.device()->newBuffer(singlePat.data(), patternLen, MTL::ResourceStorageModeShared);
        } else {
            bufPattern = store.device()->newBuffer(1, MTL::ResourceStorageModeShared);
        }
    }
    
    uint32_t rc = static_cast<uint32_t>(rowCount);

    // 4. Dispatch Kernel
    if (env_truthy("GPUDB_DEBUG_OPS")) std::cerr << "[Exec] GPU filterString: dispatching kernel rowCount=" << rowCount
                                                   << (useMultiContains ? " (multi-contains, " + std::to_string(numSegments) + " segments)" : "")
                                                   << "\n";
    
    const char* kernelName = useMultiContains ? "ops::filter_string_multi_contains" : "ops::filter_string_contains";
    if (!useMultiContains) {
        switch(op) {
            case engine::GpuFilterOp::EQ: kernelName = "ops::filter_string_eq"; break;
            case engine::GpuFilterOp::NE: kernelName = "ops::filter_string_ne"; break;
            case engine::GpuFilterOp::LT: kernelName = "ops::filter_string_lt"; break;
            case engine::GpuFilterOp::LE: kernelName = "ops::filter_string_le"; break;
            case engine::GpuFilterOp::GT: kernelName = "ops::filter_string_gt"; break;
            case engine::GpuFilterOp::GE: kernelName = "ops::filter_string_ge"; break;
            default: break;
        }
    }

    auto p_filter = makePSO(store.device(), store.library(), kernelName);
    auto p_compact = makePSO(store.device(), store.library(), "ops::compact_indices");
    
    if (!p_filter || !p_compact) {
        if (ownBufs) { bufChars->release(); bufOffsets->release(); bufLengths->release(); }
        bufPattern->release();
        if (bufPatOffsets) bufPatOffsets->release();
        if (bufPatLengths) bufPatLengths->release();
        return std::nullopt;
    }
    
    auto mask = store.device()->newBuffer(rowCount * sizeof(uint8_t), MTL::ResourceStorageModeShared);
    std::memset(mask->contents(), 0, rowCount * sizeof(uint8_t));

    // Prepare compact output
    auto outIdx = store.device()->newBuffer(rowCount * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    auto outCnt = store.device()->newBuffer(sizeof(uint32_t), MTL::ResourceStorageModeShared);
    std::memset(outCnt->contents(), 0, sizeof(uint32_t));

    // Check if we need mask flip (NOTLIKE with multi-segment)
    bool needFlip = (useMultiContains && op == engine::GpuFilterOp::NE);
    MTL::ComputePipelineState* p_flip = nullptr;
    if (needFlip) {
        p_flip = makePSO(store.device(), store.library(), "ops::flip_mask_u8");
    }
    
    auto filterStart = std::chrono::high_resolution_clock::now();

    // Fused: FILTER [→ FLIP] → COMPACT in one command buffer
    {
        auto cmd = store.queue()->commandBuffer();
        // Encoder 1: filter
        auto enc1 = cmd->computeCommandEncoder();
        enc1->setComputePipelineState(p_filter);
        enc1->setBuffer(bufChars, 0, 0);
        enc1->setBuffer(bufOffsets, 0, 1);
        enc1->setBuffer(bufLengths, 0, 2);
        enc1->setBuffer(mask, 0, 3);
        if (useMultiContains) {
            enc1->setBuffer(bufPattern, 0, 4);
            enc1->setBuffer(bufPatOffsets, 0, 5);
            enc1->setBuffer(bufPatLengths, 0, 6);
            enc1->setBytes(&numSegments, sizeof(numSegments), 7);
            enc1->setBytes(&rc, sizeof(rc), 8);
        } else {
            enc1->setBuffer(bufPattern, 0, 4);
            enc1->setBytes(&patternLen, sizeof(patternLen), 5);
            enc1->setBytes(&rc, sizeof(rc), 6);
        }
        dispatch1D(enc1, rowCount);
        enc1->endEncoding();

        // Encoder 2 (optional): flip mask for NOTLIKE
        if (needFlip && p_flip) {
            auto encFlip = cmd->computeCommandEncoder();
            encFlip->setComputePipelineState(p_flip);
            encFlip->setBuffer(mask, 0, 0);
            encFlip->setBytes(&rc, sizeof(rc), 1);
            dispatch1D(encFlip, rowCount);
            encFlip->endEncoding();
            if (env_truthy("GPUDB_DEBUG_OPS"))
                std::cerr << "[Exec] GPU filterString: flipped mask for NOTLIKE multi-contains\n";
        }

        // Final encoder: compact
        auto enc2 = cmd->computeCommandEncoder();
        enc2->setComputePipelineState(p_compact);
        enc2->setBuffer(mask, 0, 0);
        enc2->setBuffer(outIdx, 0, 1);
        enc2->setBuffer(outCnt, 0, 2);
        enc2->setBytes(&rc, sizeof(rc), 3);
        dispatch1D(enc2, rowCount);
        enc2->endEncoding();
        cmd->commit();
        cmd->waitUntilCompleted();
        checkGpuStatus(cmd);
    }
    
    auto filterEnd = std::chrono::high_resolution_clock::now();
    KernelTimer::instance().record(kernelName, "filter", 
        std::chrono::duration<double, std::milli>(filterEnd - filterStart).count(), rowCount);

    if (ownBufs) { bufChars->release(); bufOffsets->release(); bufLengths->release(); }
    bufPattern->release();
    if (bufPatOffsets) bufPatOffsets->release();
    if (bufPatLengths) bufPatLengths->release();
    
    mask->release();
    
    FilterResult res;
    res.indices.reset(outIdx);
    res.count = *reinterpret_cast<uint32_t*>(outCnt->contents());

    outCnt->release();

    return res;
}

std::optional<FilterResult> GpuOps::filterStringPrefix(const std::string& /*colName*/,
                                                          const std::vector<std::string>& data,
                                                          const std::string& pattern,
                                                          bool invert,
                                                          MTL::Buffer* preChars,
                                                          MTL::Buffer* preOffsets,
                                                          MTL::Buffer* preLengths) {
    auto& store = GpuColumnStore::instance();
    if (!store.device() || !store.library() || !store.queue()) {
        std::cerr << "[GpuOps] Device/Lib/Queue invalid\n";
        return std::nullopt;
    }

    size_t rowCount = data.size();
    if (rowCount == 0) return FilterResult{};
    
    if (env_truthy("GPUDB_DEBUG_OPS")) {
        std::cerr << "[GpuOps] filterStringPrefix pattern='" << pattern << "' invert=" << invert << " rowCount=" << rowCount << "\n";
    }
    
    // Use pre-flattened GPU buffers (always built at scan time, rebuilt after join)
    MTL::Buffer* bufChars   = preChars;
    MTL::Buffer* bufOffsets = preOffsets;
    MTL::Buffer* bufLengths = preLengths;
    bool ownBufs = (preChars == nullptr);

    if (ownBufs) {
        // Fallback: flatten on-the-fly (should be rare after join/project rebuild)
        std::vector<uint32_t> offsets(rowCount), lengths(rowCount);
        size_t totalChars = 0;
        for (const auto& s : data) totalChars += s.size();
        std::vector<char> chars;
        chars.reserve(totalChars);
        size_t cur = 0;
        for (size_t i = 0; i < rowCount; ++i) {
            offsets[i] = static_cast<uint32_t>(cur);
            lengths[i] = static_cast<uint32_t>(data[i].size());
            chars.insert(chars.end(), data[i].begin(), data[i].end());
            cur += data[i].size();
        }
        bufChars = chars.empty()
            ? store.device()->newBuffer(1, MTL::ResourceStorageModeShared)
            : store.device()->newBuffer(chars.data(), chars.size(), MTL::ResourceStorageModeShared);
        bufOffsets = store.device()->newBuffer(offsets.data(), offsets.size() * sizeof(uint32_t), MTL::ResourceStorageModeShared);
        bufLengths = store.device()->newBuffer(lengths.data(), lengths.size() * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    }
    
    std::string rawPattern = pattern;
    if (!rawPattern.empty() && rawPattern.back() == '%') rawPattern.pop_back();
    
    uint32_t patternLen = static_cast<uint32_t>(rawPattern.size());
    MTL::Buffer* bufPattern = nullptr;
    if (patternLen > 0) bufPattern = store.device()->newBuffer(rawPattern.data(), patternLen, MTL::ResourceStorageModeShared);
    else bufPattern = store.device()->newBuffer(1, MTL::ResourceStorageModeShared);
    
    uint32_t rc = static_cast<uint32_t>(rowCount);

    const char* kernelName = invert ? "ops::filter_string_not_prefix" : "ops::filter_string_prefix";

    if (env_truthy("GPUDB_DEBUG_OPS")) {
        std::cerr << "[GpuOps] Requesting kernel: " << kernelName << "\n";
    }

    auto p_filter = makePSO(store.device(), store.library(), kernelName);
    auto p_compact = makePSO(store.device(), store.library(), "ops::compact_indices");
    
    if (!p_filter || !p_compact) {
        if(!p_filter) std::cerr << "[GpuOps] Failed to make PSO for " << kernelName << "\n";
        if(!p_compact) std::cerr << "[GpuOps] Failed to make PSO for compact\n";
        if (ownBufs) { bufChars->release(); bufOffsets->release(); bufLengths->release(); }
        bufPattern->release();
        return std::nullopt;
    }
    
    auto mask = store.device()->newBuffer(rowCount * sizeof(uint8_t), MTL::ResourceStorageModeShared);
    std::memset(mask->contents(), 0, rowCount);

    auto outIdx = store.device()->newBuffer(rowCount * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    auto outCnt = store.device()->newBuffer(sizeof(uint32_t), MTL::ResourceStorageModeShared);
    std::memset(outCnt->contents(), 0, sizeof(uint32_t));

    // Fused: FILTER → COMPACT in one command buffer (2 encoders)
    {
        auto cmd = store.queue()->commandBuffer();
        // Encoder 1: filter
        auto enc1 = cmd->computeCommandEncoder();
        enc1->setComputePipelineState(p_filter);
        enc1->setBuffer(bufChars, 0, 0);
        enc1->setBuffer(bufOffsets, 0, 1);
        enc1->setBuffer(bufLengths, 0, 2);
        enc1->setBuffer(mask, 0, 3);
        enc1->setBuffer(bufPattern, 0, 4);
        enc1->setBytes(&patternLen, sizeof(patternLen), 5);
        enc1->setBytes(&rc, sizeof(rc), 6);
        dispatch1D(enc1, rowCount);
        enc1->endEncoding();
        // Encoder 2: compact
        auto enc2 = cmd->computeCommandEncoder();
        enc2->setComputePipelineState(p_compact);
        enc2->setBuffer(mask, 0, 0);
        enc2->setBuffer(outIdx, 0, 1);
        enc2->setBuffer(outCnt, 0, 2);
        enc2->setBytes(&rc, sizeof(rc), 3);
        dispatch1D(enc2, rowCount);
        enc2->endEncoding();
        cmd->commit();
        cmd->waitUntilCompleted();
        checkGpuStatus(cmd);
    }

    if (ownBufs) { bufChars->release(); bufOffsets->release(); bufLengths->release(); }
    bufPattern->release();
    
    mask->release();
    
    FilterResult res;
    res.indices.reset(outIdx);
    res.count = *reinterpret_cast<uint32_t*>(outCnt->contents());

    outCnt->release();
    return res;
}

std::optional<FilterResult> GpuOps::filterU32Indexed(const std::string& colName,
                                                              MTL::Buffer* col,
                                                              MTL::Buffer* indices,
                                                              uint32_t count,
                                                              engine::GpuFilterOp op,
                                                              uint32_t literal) {
    auto& store = GpuColumnStore::instance();
    if (!store.device() || !store.library() || !store.queue()) return std::nullopt;

    const char* fn = nullptr;
    switch (op) {
        case engine::GpuFilterOp::EQ: fn = "ops::filter_eq_u32_indexed"; break;
        case engine::GpuFilterOp::LT: fn = "ops::filter_lt_u32_indexed"; break;
        case engine::GpuFilterOp::GT: fn = "ops::filter_gt_u32_indexed"; break;
        case engine::GpuFilterOp::LE: fn = "ops::filter_le_u32_indexed"; break;
        case engine::GpuFilterOp::GE: fn = "ops::filter_ge_u32_indexed"; break;
        case engine::GpuFilterOp::NE: fn = "ops::filter_ne_u32_indexed"; break;
        default: break;
    }

    auto p_filter = makePSO(store.device(), store.library(), fn);
    auto p_compact = makePSO(store.device(), store.library(), "ops::compact_indices_indexed");
    if (!p_filter || !p_compact) {
        return std::nullopt;
    }

    auto mask = store.device()->newBuffer(static_cast<size_t>(count) * sizeof(uint8_t), MTL::ResourceStorageModeShared);
    std::memset(mask->contents(), 0, static_cast<size_t>(count) * sizeof(uint8_t));

    auto outIdx = store.device()->newBuffer(static_cast<size_t>(count) * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    auto outCnt = store.device()->newBuffer(sizeof(uint32_t), MTL::ResourceStorageModeShared);
    std::memset(outCnt->contents(), 0, sizeof(uint32_t));

    auto filterStart = std::chrono::high_resolution_clock::now();
    // Fused: FILTER → COMPACT in one command buffer (2 encoders)
    {
        auto cmd = store.queue()->commandBuffer();
        // Encoder 1: filter
        auto enc1 = cmd->computeCommandEncoder();
        enc1->setComputePipelineState(p_filter);
        enc1->setBuffer(col, 0, 0);
        enc1->setBuffer(indices, 0, 1);
        enc1->setBuffer(mask, 0, 2);
        enc1->setBytes(&literal, sizeof(literal), 3);
        enc1->setBytes(&count, sizeof(count), 4);
        dispatch1D(enc1, count);
        enc1->endEncoding();
        // Encoder 2: compact
        auto enc2 = cmd->computeCommandEncoder();
        enc2->setComputePipelineState(p_compact);
        enc2->setBuffer(mask, 0, 0);
        enc2->setBuffer(indices, 0, 1);
        enc2->setBuffer(outIdx, 0, 2);
        enc2->setBuffer(outCnt, 0, 3);
        enc2->setBytes(&count, sizeof(count), 4);
        dispatch1D(enc2, count);
        enc2->endEncoding();
        cmd->commit();
        cmd->waitUntilCompleted();
        checkGpuStatus(cmd);
    }
    auto filterEnd = std::chrono::high_resolution_clock::now();
    KernelTimer::instance().record(fn, "filter", 
        std::chrono::duration<double, std::milli>(filterEnd - filterStart).count(), count);

    FilterResult res;
    res.indices.reset(outIdx);
    res.count = *reinterpret_cast<uint32_t*>(outCnt->contents());

    bool debug = env_truthy("GPUDB_DEBUG_OPS");
    if (debug) std::cerr << "[Exec] GPU filterU32Indexed: col=" << colName << " rowCount=" << count << " val=" << literal << " result=" << res.count << "\n";

    mask->release();
    outCnt->release();
    return res;
}

std::optional<FilterResult> GpuOps::filterF32(const std::string& /*colName*/,
                                                      MTL::Buffer* col,
                                                      uint32_t rowCount,
                                                      engine::GpuFilterOp op,
                                                      float literal) {
    auto& store = GpuColumnStore::instance();
    if (!store.device() || !store.library() || !store.queue()) return std::nullopt;

    const char* fn = nullptr;
    switch (op) {
        case engine::GpuFilterOp::EQ: fn = "ops::filter_eq_f32"; break;
        case engine::GpuFilterOp::LT: fn = "ops::filter_lt_f32"; break;
        case engine::GpuFilterOp::GT: fn = "ops::filter_gt_f32"; break;
        case engine::GpuFilterOp::LE: fn = "ops::filter_le_f32"; break;
        case engine::GpuFilterOp::GE: fn = "ops::filter_ge_f32"; break;
        case engine::GpuFilterOp::NE: fn = "ops::filter_ne_f32"; break;
        default: break;
    }

    auto p_filter = makePSO(store.device(), store.library(), fn);
    auto p_compact = makePSO(store.device(), store.library(), "ops::compact_indices");
    if (!p_filter || !p_compact) {
        return std::nullopt;
    }

    auto mask = store.device()->newBuffer(rowCount * sizeof(uint8_t), MTL::ResourceStorageModeShared);
    std::memset(mask->contents(), 0, rowCount * sizeof(uint8_t));

    auto outIdx = store.device()->newBuffer(rowCount * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    auto outCnt = store.device()->newBuffer(sizeof(uint32_t), MTL::ResourceStorageModeShared);
    std::memset(outCnt->contents(), 0, sizeof(uint32_t));

    auto filterStart = std::chrono::high_resolution_clock::now();
    {
        auto cmd = store.queue()->commandBuffer();
        auto enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(p_filter);
        enc->setBuffer(col, 0, 0);
        enc->setBuffer(mask, 0, 1);
        enc->setBytes(&literal, sizeof(literal), 2);
        enc->setBytes(&rowCount, sizeof(rowCount), 3);
        dispatch1D(enc, rowCount);
        enc->endEncoding();
        auto enc2 = cmd->computeCommandEncoder();
        enc2->setComputePipelineState(p_compact);
        enc2->setBuffer(mask, 0, 0);
        enc2->setBuffer(outIdx, 0, 1);
        enc2->setBuffer(outCnt, 0, 2);
        enc2->setBytes(&rowCount, sizeof(rowCount), 3);
        dispatch1D(enc2, rowCount);
        enc2->endEncoding();
        cmd->commit();
        cmd->waitUntilCompleted();
        checkGpuStatus(cmd);
    }
    auto filterEnd = std::chrono::high_resolution_clock::now();
    KernelTimer::instance().record(fn, "filter",
        std::chrono::duration<double, std::milli>(filterEnd - filterStart).count(), rowCount);

    FilterResult res;
    res.indices.reset(outIdx);
    res.count = *reinterpret_cast<uint32_t*>(outCnt->contents());

    mask->release();
    outCnt->release();
    return res;
}

std::optional<FilterResult> GpuOps::filterF32Indexed(const std::string& colName,
                                                              MTL::Buffer* col,
                                                              MTL::Buffer* indices,
                                                              uint32_t count,
                                                              engine::GpuFilterOp op,
                                                              float literal) {
    auto& store = GpuColumnStore::instance();
    if (!store.device() || !store.library() || !store.queue()) return std::nullopt;

    const char* fn = nullptr;
    switch (op) {
        case engine::GpuFilterOp::EQ: fn = "ops::filter_eq_f32_indexed"; break;
        case engine::GpuFilterOp::LT: fn = "ops::filter_lt_f32_indexed"; break;
        case engine::GpuFilterOp::GT: fn = "ops::filter_gt_f32_indexed"; break;
        case engine::GpuFilterOp::LE: fn = "ops::filter_le_f32_indexed"; break;
        case engine::GpuFilterOp::GE: fn = "ops::filter_ge_f32_indexed"; break;
        case engine::GpuFilterOp::NE: fn = "ops::filter_ne_f32_indexed"; break;
        default: break;
    }

    auto p_filter = makePSO(store.device(), store.library(), fn);
    auto p_compact = makePSO(store.device(), store.library(), "ops::compact_indices_indexed");
    if (!p_filter || !p_compact) {
        return std::nullopt;
    }

    auto mask = store.device()->newBuffer(static_cast<size_t>(count) * sizeof(uint8_t), MTL::ResourceStorageModeShared);
    std::memset(mask->contents(), 0, static_cast<size_t>(count) * sizeof(uint8_t));

    auto outIdx = store.device()->newBuffer(static_cast<size_t>(count) * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    auto outCnt = store.device()->newBuffer(sizeof(uint32_t), MTL::ResourceStorageModeShared);
    std::memset(outCnt->contents(), 0, sizeof(uint32_t));

    auto filterStart = std::chrono::high_resolution_clock::now();
    {
        auto cmd = store.queue()->commandBuffer();
        auto enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(p_filter);
        enc->setBuffer(col, 0, 0);
        enc->setBuffer(indices, 0, 1);
        enc->setBuffer(mask, 0, 2);
        enc->setBytes(&literal, sizeof(literal), 3);
        enc->setBytes(&count, sizeof(count), 4);
        dispatch1D(enc, count);
        enc->endEncoding();
        auto enc2 = cmd->computeCommandEncoder();
        enc2->setComputePipelineState(p_compact);
        enc2->setBuffer(mask, 0, 0);
        enc2->setBuffer(indices, 0, 1);
        enc2->setBuffer(outIdx, 0, 2);
        enc2->setBuffer(outCnt, 0, 3);
        enc2->setBytes(&count, sizeof(count), 4);
        dispatch1D(enc2, count);
        enc2->endEncoding();
        cmd->commit();
        cmd->waitUntilCompleted();
        checkGpuStatus(cmd);
    }
    auto filterEnd = std::chrono::high_resolution_clock::now();
    KernelTimer::instance().record(fn, "filter",
        std::chrono::duration<double, std::milli>(filterEnd - filterStart).count(), count);

    FilterResult res;
    res.indices.reset(outIdx);
    res.count = *reinterpret_cast<uint32_t*>(outCnt->contents());

    mask->release();
    outCnt->release();
    (void)colName;
    return res;
}

std::optional<FilterResult> GpuOps::filterColColU32(
    MTL::Buffer* colA,
    MTL::Buffer* colB,
    uint32_t count,
    int opInt) {
    auto& store = GpuColumnStore::instance();
    if (!store.device() || !store.library() || !store.queue()) return std::nullopt;
    
    engine::GpuFilterOp op = static_cast<engine::GpuFilterOp>(opInt);
    const char* kernelName = nullptr;
    switch(op) {
        case engine::GpuFilterOp::EQ: kernelName = "ops::filter_u32_col_col_eq"; break;
        case engine::GpuFilterOp::NE: kernelName = "ops::filter_u32_col_col_ne"; break;
        case engine::GpuFilterOp::LT: kernelName = "ops::filter_u32_col_col_lt"; break;
        case engine::GpuFilterOp::LE: kernelName = "ops::filter_u32_col_col_le"; break;
        case engine::GpuFilterOp::GT: kernelName = "ops::filter_u32_col_col_gt"; break;
        case engine::GpuFilterOp::GE: kernelName = "ops::filter_u32_col_col_ge"; break;
        default: return std::nullopt;
    }

    auto p_filter = makePSO(store.device(), store.library(), kernelName);
    auto p_compact = makePSO(store.device(), store.library(), "ops::compact_indices");
    if (!p_filter || !p_compact) return std::nullopt;

    auto mask = store.device()->newBuffer(count * sizeof(uint8_t), MTL::ResourceStorageModeShared);
    auto outIdx = store.device()->newBuffer(count * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    auto outCnt = store.device()->newBuffer(sizeof(uint32_t), MTL::ResourceStorageModeShared);
    *(uint32_t*)outCnt->contents() = 0;

    {
        auto cmd = store.queue()->commandBuffer();
        auto enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(p_filter);
        enc->setBuffer(colA, 0, 0);
        enc->setBuffer(colB, 0, 1);
        enc->setBuffer(mask, 0, 2);
        enc->setBytes(&count, sizeof(count), 3);
        dispatch1D(enc, count);
        enc->endEncoding();
        auto enc2 = cmd->computeCommandEncoder();
        enc2->setComputePipelineState(p_compact);
        enc2->setBuffer(mask, 0, 0);
        enc2->setBuffer(outIdx, 0, 1);
        enc2->setBuffer(outCnt, 0, 2);
        enc2->setBytes(&count, sizeof(count), 3);
        dispatch1D(enc2, count);
        enc2->endEncoding();
        cmd->commit();
        cmd->waitUntilCompleted();
        checkGpuStatus(cmd);
    }
    
    mask->release();

    FilterResult res;
    res.indices.reset(outIdx);
    res.count = *reinterpret_cast<uint32_t*>(outCnt->contents());
    outCnt->release();
    return res;
}

std::optional<FilterResult> GpuOps::filterColColF32(
    MTL::Buffer* colA,
    MTL::Buffer* colB,
    uint32_t count,
    int opInt) {
    auto& store = GpuColumnStore::instance();
    if (!store.device() || !store.library() || !store.queue()) return std::nullopt;

    engine::GpuFilterOp op = static_cast<engine::GpuFilterOp>(opInt);
    const char* kernelName = nullptr;
    switch(op) {
        case engine::GpuFilterOp::EQ: kernelName = "ops::filter_f32_col_col_eq"; break;
        case engine::GpuFilterOp::NE: kernelName = "ops::filter_f32_col_col_ne"; break;
        case engine::GpuFilterOp::LT: kernelName = "ops::filter_f32_col_col_lt"; break;
        case engine::GpuFilterOp::LE: kernelName = "ops::filter_f32_col_col_le"; break;
        case engine::GpuFilterOp::GT: kernelName = "ops::filter_f32_col_col_gt"; break;
        case engine::GpuFilterOp::GE: kernelName = "ops::filter_f32_col_col_ge"; break;
        default: return std::nullopt;
    }

    auto p_filter = makePSO(store.device(), store.library(), kernelName);
    auto p_compact = makePSO(store.device(), store.library(), "ops::compact_indices");
    if (!p_filter || !p_compact) return std::nullopt;

    auto mask = store.device()->newBuffer(count * sizeof(uint8_t), MTL::ResourceStorageModeShared);
    auto outIdx = store.device()->newBuffer(count * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    auto outCnt = store.device()->newBuffer(sizeof(uint32_t), MTL::ResourceStorageModeShared);
    *(uint32_t*)outCnt->contents() = 0;

    {
        auto cmd = store.queue()->commandBuffer();
        auto enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(p_filter);
        enc->setBuffer(colA, 0, 0);
        enc->setBuffer(colB, 0, 1);
        enc->setBuffer(mask, 0, 2);
        enc->setBytes(&count, sizeof(count), 3);
        dispatch1D(enc, count);
        enc->endEncoding();
        auto enc2 = cmd->computeCommandEncoder();
        enc2->setComputePipelineState(p_compact);
        enc2->setBuffer(mask, 0, 0);
        enc2->setBuffer(outIdx, 0, 1);
        enc2->setBuffer(outCnt, 0, 2);
        enc2->setBytes(&count, sizeof(count), 3);
        dispatch1D(enc2, count);
        enc2->endEncoding();
        cmd->commit();
        cmd->waitUntilCompleted();
        checkGpuStatus(cmd);
    }
    
    mask->release();

    FilterResult res;
    res.indices.reset(outIdx);
    res.count = *reinterpret_cast<uint32_t*>(outCnt->contents());
    outCnt->release();
    return res;
}


} // namespace engine
