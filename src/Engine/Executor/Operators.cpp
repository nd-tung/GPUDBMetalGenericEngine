#include "Operators.hpp"
#include "OperatorsInternal.hpp"
#include "EnvUtil.hpp"

#include "GpuColumnStore.hpp"
#include "KernelTimer.hpp"
#include "Schema.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <unordered_map>
#include "Logger.hpp"

namespace engine {

// ── Scan / Data Loading ────────────────────────────────────────────────

uint32_t GpuOps::fnv1a32(std::string_view s) {
    uint32_t hash = 2166136261u;
    for (unsigned char c : s) {
        hash ^= static_cast<uint32_t>(c);
        hash *= 16777619u;
    }
    if (hash == 0) hash = 1;
    if (hash == 0xFFFFFFFFu) hash = 0xFFFFFFFEu;
    return hash;
}

// Single-pass multi-column loader: reads a .tbl file ONCE and extracts all
// requested columns in a single sweep. Replaces per-column loaders that each
// re-read the entire file.
// ============================================================================

struct ColLoadSpec {
    int columnIndex;           // 0-based column in the pipe-delimited file
    std::string outputName;    // key for the result maps
    enum Type { kU32, kF32, kDateU32, kStrCharU32, kStringRaw } type;
};

struct OnePassResults {
    std::unordered_map<std::string, std::vector<uint32_t>> u32Data;   // U32, DateU32, StrCharU32
    std::unordered_map<std::string, std::vector<float>>    f32Data;   // F32
    std::unordered_map<std::string, std::vector<std::string>> strData; // StringRaw
};

static OnePassResults loadColumnsOnePass(const std::string& filePath,
                                          const std::vector<ColLoadSpec>& specs) {
    OnePassResults result;
    if (specs.empty()) return result;

    std::ifstream file(filePath);
    if (!file.is_open()) return result;

    // Find the maximum column index we need to scan to
    int maxColIdx = 0;
    for (const auto& s : specs)
        if (s.columnIndex > maxColIdx) maxColIdx = s.columnIndex;

    // Build lookup: column_index -> list of spec indices
    std::vector<std::vector<size_t>> idxToSpecs(maxColIdx + 1);
    for (size_t i = 0; i < specs.size(); ++i)
        idxToSpecs[specs[i].columnIndex].push_back(i);

    // Pre-allocate result maps
    for (const auto& s : specs) {
        switch (s.type) {
            case ColLoadSpec::kF32:
                result.f32Data[s.outputName].reserve(1 << 20); break;
            case ColLoadSpec::kStringRaw:
                result.strData[s.outputName].reserve(1 << 20); break;
            default:
                result.u32Data[s.outputName].reserve(1 << 20); break;
        }
    }

    const int totalSpecs = static_cast<int>(specs.size());
    std::string line;
    while (std::getline(file, line)) {
        int col = 0;
        size_t start = 0;
        size_t end = line.find('|');
        int found = 0;

        while (end != std::string::npos && col <= maxColIdx) {
            if (!idxToSpecs[col].empty()) {
                // Trim whitespace from token
                size_t ts = start, te = end;
                while (ts < te && (line[ts] == ' ' || line[ts] == '\t')) ++ts;
                while (te > ts && (line[te-1] == ' ' || line[te-1] == '\t' ||
                                   line[te-1] == '\n' || line[te-1] == '\r')) --te;

                for (size_t si : idxToSpecs[col]) {
                    const auto& spec = specs[si];
                    switch (spec.type) {
                        case ColLoadSpec::kU32: {
                            uint32_t val = 0;
                            for (size_t i = ts; i < te; ++i) {
                                char c = line[i];
                                if (c >= '0' && c <= '9') val = val * 10 + (c - '0');
                            }
                            result.u32Data[spec.outputName].push_back(val);
                            break;
                        }
                        case ColLoadSpec::kF32: {
                            float val = 0.0f;
                            try { val = std::stof(std::string(line, ts, te - ts)); }
                            catch (...) { /* malformed float in .tbl — defaults to 0.0f */ }
                            result.f32Data[spec.outputName].push_back(val);
                            break;
                        }
                        case ColLoadSpec::kDateU32: {
                            // Parse YYYY-MM-DD → YYYYMMDD (skip '-' chars)
                            uint32_t val = 0;
                            for (size_t i = ts; i < te; ++i) {
                                char c = line[i];
                                if (c >= '0' && c <= '9') val = val * 10 + (c - '0');
                            }
                            result.u32Data[spec.outputName].push_back(val);
                            break;
                        }
                        case ColLoadSpec::kStrCharU32: {
                            uint32_t val = (ts < te)
                                ? static_cast<uint32_t>(static_cast<unsigned char>(line[ts])) : 0;
                            result.u32Data[spec.outputName].push_back(val);
                            break;
                        }
                        case ColLoadSpec::kStringRaw: {
                            result.strData[spec.outputName].emplace_back(line, ts, te - ts);
                            break;
                        }
                    }
                    ++found;
                }
            }

            if (found >= totalSpecs) break;  // All columns found for this row

            start = end + 1;
            end = line.find('|', start);
            ++col;
        }
    }

    return result;
}

// Cache for raw string columns, populated by one-pass loader.
// Key: "filePath:columnIndex"
static std::unordered_map<std::string, std::vector<std::string>> s_rawStringCache;
static std::mutex s_rawStringCacheMutex;

static std::string rawStringCacheKey(const std::string& filePath, int columnIndex) {
    return filePath + ":" + std::to_string(columnIndex);
}

// Load raw string column (for LIKE/CONTAINS pattern matching)
// Checks the cache first; falls back to file read.
static std::vector<std::string> loadStringColumnRawImpl(const std::string& filePath, int columnIndex) {
    // Check cache (populated by single-pass loader)
    std::string cKey = rawStringCacheKey(filePath, columnIndex);
    {
        std::lock_guard<std::mutex> lock(s_rawStringCacheMutex);
        auto cit = s_rawStringCache.find(cKey);
        if (cit != s_rawStringCache.end()) return cit->second;
    }

    // Cache miss: read this single column from file
    std::vector<std::string> data;
    std::ifstream file(filePath);
    if (!file.is_open()) return data;

    std::string line;
    while (std::getline(file, line)) {
        int col = 0;
        size_t s = 0;
        size_t e = line.find('|');
        while (e != std::string::npos) {
            if (col == columnIndex) {
                std::string token = line.substr(s, e - s);
                token.erase(0, token.find_first_not_of(" \t\n\r"));
                token.erase(token.find_last_not_of(" \t\n\r") + 1);
                data.push_back(std::move(token));
                break;
            }
            s = e + 1;
            e = line.find('|', s);
            ++col;
        }
    }
    return data;
}

// ── Batch Control ──────────────────────────────────────────────────────
static thread_local int s_gpuBatchDepth = 0;

bool GpuOps::isBatchActive() { return s_gpuBatchDepth > 0; }

void GpuOps::beginBatch() { s_gpuBatchDepth++; }

void GpuOps::endBatch() {
    if (--s_gpuBatchDepth <= 0) {
        s_gpuBatchDepth = 0;
        // Flush: ensure all submitted command buffers have completed
        sync();
    }
}

GpuRelation GpuOps::scanTable(const std::string& datasetPath,
                                   const std::string& table,
                                   const std::vector<std::string>& neededCols) {
    GpuRelation rel;

    auto& store = GpuColumnStore::instance();
    store.initialize();
    if (!store.device()) return rel;

    const auto& schemaReg = engine::SchemaRegistry::instance();
    const auto* tblSchema = schemaReg.getTable(table);
    if (!tblSchema) return rel;

    std::string path = datasetPath + table + ".tbl";

    auto cache_key = [&](const std::string& colName) {
        return datasetPath + table + "." + colName;
    };

    bool sizeSet = false;
    uint32_t rowCount = 0;

    // Phase 1: Check GpuColumnStore cache for each column; collect uncached ones
    std::vector<ColLoadSpec> uncached;

    for (const auto& c : neededCols) {
        auto gpuInfo = schemaReg.getGpuColInfo(table, c);
        if (!gpuInfo) {
            // StringHash column — handle below
            const auto* colSchema = tblSchema->getColumn(c);
            if (colSchema && colSchema->type == ColumnType::StringHash) {
                std::string cKey = rawStringCacheKey(path, colSchema->index);
                bool found;
                {
                    std::lock_guard<std::mutex> lock(s_rawStringCacheMutex);
                    found = s_rawStringCache.find(cKey) != s_rawStringCache.end();
                }
                if (!found) {
                    ColLoadSpec spec;
                    spec.columnIndex = colSchema->index;
                    spec.outputName = c;
                    spec.type = ColLoadSpec::kStringRaw;
                    uncached.push_back(spec);
                }
            }
            continue;
        }

        const std::string key = cache_key(c);
        GpuColumn* staged = store.getColumn(key);
        if (staged && staged->buffer) {
            // Already cached — use directly
            if (!sizeSet) { rowCount = static_cast<uint32_t>(staged->count); sizeSet = true; }
            if (static_cast<uint32_t>(staged->count) != rowCount) continue;
            staged->buffer->retain();
            if (gpuInfo->kind == SchemaRegistry::GpuColKind::F32)
                rel.f32cols[c].reset(staged->buffer);
            else
                rel.u32cols[c].reset(staged->buffer);
            continue;
        }

        // Not cached — will load in one-pass
        ColLoadSpec spec;
        spec.columnIndex = gpuInfo->index;
        spec.outputName = c;
        switch (gpuInfo->kind) {
            case SchemaRegistry::GpuColKind::U32:       spec.type = ColLoadSpec::kU32; break;
            case SchemaRegistry::GpuColKind::F32:       spec.type = ColLoadSpec::kF32; break;
            case SchemaRegistry::GpuColKind::DateU32:   spec.type = ColLoadSpec::kDateU32; break;
            case SchemaRegistry::GpuColKind::StrCharU32:spec.type = ColLoadSpec::kStrCharU32; break;
        }
        uncached.push_back(spec);
    }

    // Phase 2: Load ALL uncached columns in a SINGLE file pass
    if (!uncached.empty()) {
        auto loaded = loadColumnsOnePass(path, uncached);

        for (const auto& spec : uncached) {
            if (spec.type == ColLoadSpec::kStringRaw) {
                // Cache raw strings for later loadStringColumnRaw calls
                auto it = loaded.strData.find(spec.outputName);
                if (it != loaded.strData.end() && !it->second.empty()) {
                    const auto* cs = tblSchema->getColumn(spec.outputName);
                    if (cs) {
                        std::lock_guard<std::mutex> lock(s_rawStringCacheMutex);
                        s_rawStringCache[rawStringCacheKey(path, cs->index)] = std::move(it->second);
                    }
                }
                continue;
            }

            const std::string key = cache_key(spec.outputName);
            GpuColumn* staged = nullptr;

            if (spec.type == ColLoadSpec::kF32) {
                auto it = loaded.f32Data.find(spec.outputName);
                if (it == loaded.f32Data.end() || it->second.empty()) continue;
                staged = store.stageFloatColumn(key, it->second);
            } else {
                auto it = loaded.u32Data.find(spec.outputName);
                if (it == loaded.u32Data.end() || it->second.empty()) continue;
                staged = store.stageU32Column(key, it->second);
            }

            if (!staged || !staged->buffer) continue;
            if (!sizeSet) { rowCount = static_cast<uint32_t>(staged->count); sizeSet = true; }
            if (static_cast<uint32_t>(staged->count) != rowCount) continue;
            staged->buffer->retain();

            if (spec.type == ColLoadSpec::kF32)
                rel.f32cols[spec.outputName].reset(staged->buffer);
            else
                rel.u32cols[spec.outputName].reset(staged->buffer);
        }
    }

    rel.rowCount = rowCount;
    return rel;
}











// ── Gather & Scatter ───────────────────────────────────────────────────

GpuBuffer GpuOps::gatherU32(MTL::Buffer* in, MTL::Buffer* indices, uint32_t count, bool sync) {
    auto& store = GpuColumnStore::instance();
    auto p_g = makePSO(store.device(), store.library(), "ops::gather_col_u32");
    if (!p_g) {
        LOG_ERROR("GPU", "GpuOps::gatherU32: Failed to create PSO ops::gather_col_u32\n");
        return GpuBuffer(nullptr);
    }

    if (!in || !indices) {
        LOG_ERROR("GPU", "GpuOps::gatherU32: Input or Indices buffer is NULL\n");
        return GpuBuffer(nullptr);
    }

    auto out = store.device()->newBuffer(count * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    if (!out) {
        LOG_ERROR("GPU", "GpuOps::gatherU32: Failed to allocate output buffer size " << (count * sizeof(uint32_t)));
        return GpuBuffer(nullptr);
    }
    auto start = std::chrono::high_resolution_clock::now();
    {
        auto cmd = store.queue()->commandBuffer();
        auto enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(p_g);
        enc->setBuffer(in, 0, 0);
        enc->setBuffer(indices, 0, 1);
        enc->setBuffer(out, 0, 2);
        enc->setBytes(&count, sizeof(count), 3);
        dispatch1D(enc, count);
        enc->endEncoding();
        cmd->commit();
        if (sync) {
            cmd->waitUntilCompleted();
            checkGpuStatus(cmd, "gatherU32");
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    KernelTimer::instance().record("ops::gather_col_u32", sync ? "gather" : "gather_dispatch",
        std::chrono::duration<double, std::milli>(end - start).count(), count);
    return GpuBuffer(out);
}

GpuBuffer GpuOps::gatherF32(MTL::Buffer* in, MTL::Buffer* indices, uint32_t count, bool sync) {
    auto& store = GpuColumnStore::instance();
    auto p_g = makePSO(store.device(), store.library(), "ops::gather_col_f32");
    if (!p_g) return {};

    auto out = store.device()->newBuffer(count * sizeof(float), MTL::ResourceStorageModeShared);
    auto start = std::chrono::high_resolution_clock::now();
    {
        auto cmd = store.queue()->commandBuffer();
        auto enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(p_g);
        enc->setBuffer(in, 0, 0);
        enc->setBuffer(indices, 0, 1);
        enc->setBuffer(out, 0, 2);
        enc->setBytes(&count, sizeof(count), 3);
        dispatch1D(enc, count);
        enc->endEncoding();
        cmd->commit();
        if (sync) {
            cmd->waitUntilCompleted();
            checkGpuStatus(cmd, "gatherF32");
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    KernelTimer::instance().record("ops::gather_col_f32", sync ? "gather" : "gather_dispatch",
        std::chrono::duration<double, std::milli>(end - start).count(), count);
    return GpuBuffer(out);
}

// ── Utility ────────────────────────────────────────────────────────────

void GpuOps::sync() {
    auto& store = GpuColumnStore::instance();
    auto cmd = store.queue()->commandBuffer();
    auto start = std::chrono::high_resolution_clock::now();
    cmd->commit();
    cmd->waitUntilCompleted();
    auto end = std::chrono::high_resolution_clock::now();
    KernelTimer::instance().record("sync", "batch_flush",
        std::chrono::duration<double, std::milli>(end - start).count(), 0);
}

// ── Conversion & Bitwise ───────────────────────────────────────────────

GpuBuffer GpuOps::castU32ToF32(MTL::Buffer* in, uint32_t count) {
    auto& store = GpuColumnStore::instance();
    auto p = makePSO(store.device(), store.library(), "ops::cast_u32_to_f32");
    if (!p) return {};

    auto out = store.device()->newBuffer(static_cast<size_t>(count) * sizeof(float), MTL::ResourceStorageModeShared);
    auto start = std::chrono::high_resolution_clock::now();
    {
        auto cmd = store.queue()->commandBuffer();
        auto enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(p);
        enc->setBuffer(in, 0, 0);
        enc->setBuffer(out, 0, 1);
        enc->setBytes(&count, sizeof(count), 2);
        dispatch1D(enc, count);
        enc->endEncoding();
        cmd->commit();
        if (!isBatchActive()) {
            cmd->waitUntilCompleted();
            checkGpuStatus(cmd, "castU32ToF32");
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    KernelTimer::instance().record("ops::cast_u32_to_f32", "cast",
        std::chrono::duration<double, std::milli>(end - start).count(), count);
    return GpuBuffer(out);
}



std::vector<std::string> GpuOps::loadStringColumnRaw(const std::string& datasetPath,
                                                           const std::string& table,
                                                           const std::string& column) {
    const auto& schema = engine::SchemaRegistry::instance();
    const auto* tblSchema = schema.getTable(table);
    if (!tblSchema) return {};
    
    const auto* colSchema = tblSchema->getColumn(column);
    if (!colSchema) return {};
    
    std::string path = datasetPath + table + ".tbl";
    return loadStringColumnRawImpl(path, colSchema->index);
}




// ── Mask & Index Operations ────────────────────────────────────────────

GpuBuffer GpuOps::logicOrU32(MTL::Buffer* colA, MTL::Buffer* colB, uint32_t count) {
    auto* dev = GpuColumnStore::instance().device();
    auto* lib = GpuColumnStore::instance().library();
    auto* cmd = GpuColumnStore::instance().queue()->commandBuffer();
    auto* enc = cmd->computeCommandEncoder();
    
    auto* pso = makePSO(dev, lib, "ops::logic_or_u32");
    if(!pso) { enc->endEncoding(); return {}; }
    
    enc->setComputePipelineState(pso);
    enc->setBuffer(colA, 0, 0);
    enc->setBuffer(colB, 0, 1);
    
    auto* outMask = dev->newBuffer(count * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    enc->setBuffer(outMask, 0, 2);
    
    enc->setBytes(&count, sizeof(uint32_t), 3);
    
    dispatch1D(enc, count);
    enc->endEncoding();
    cmd->commit();
    
    return GpuBuffer(outMask);
}

GpuBuffer GpuOps::logicAndNotU32(MTL::Buffer* colA, MTL::Buffer* colB, uint32_t count) {
    auto* dev = GpuColumnStore::instance().device();
    auto* lib = GpuColumnStore::instance().library();
    auto* cmd = GpuColumnStore::instance().queue()->commandBuffer();
    auto* enc = cmd->computeCommandEncoder();
    
    auto* pso = makePSO(dev, lib, "ops::logic_andnot_u32");
    if(!pso) { enc->endEncoding(); return {}; }
    
    enc->setComputePipelineState(pso);
    enc->setBuffer(colA, 0, 0);
    enc->setBuffer(colB, 0, 1);
    
    auto* outMask = dev->newBuffer(count * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    enc->setBuffer(outMask, 0, 2);
    
    enc->setBytes(&count, sizeof(uint32_t), 3);
    
    dispatch1D(enc, count);
    enc->endEncoding();
    cmd->commit();
    
    return GpuBuffer(outMask);
}

GpuBuffer GpuOps::indicesToMask(MTL::Buffer* indices, uint32_t indexCount, uint32_t totalRows) {
    auto* dev = GpuColumnStore::instance().device();
    auto* lib = GpuColumnStore::instance().library();
    auto* cmd = GpuColumnStore::instance().queue()->commandBuffer();
    
    // Create and clear the mask buffer
    auto* mask = dev->newBuffer(totalRows * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    
    // Clear mask to zeros
    auto* enc1 = cmd->computeCommandEncoder();
    auto* psoClear = makePSO(dev, lib, "ops::clear_mask");
    if(!psoClear) { enc1->endEncoding(); return {}; }
    enc1->setComputePipelineState(psoClear);
    enc1->setBuffer(mask, 0, 0);
    enc1->setBytes(&totalRows, sizeof(uint32_t), 1);
    dispatch1D(enc1, totalRows);
    enc1->endEncoding();
    
    // Set mask[indices[i]] = 1
    auto* enc2 = cmd->computeCommandEncoder();
    auto* psoSet = makePSO(dev, lib, "ops::indices_to_mask");
    if(!psoSet) { enc2->endEncoding(); return GpuBuffer(mask); }
    enc2->setComputePipelineState(psoSet);
    enc2->setBuffer(indices, 0, 0);
    enc2->setBuffer(mask, 0, 1);
    enc2->setBytes(&indexCount, sizeof(uint32_t), 2);
    dispatch1D(enc2, indexCount);
    enc2->endEncoding();
    
    cmd->commit();
    
    return GpuBuffer(mask);
}

std::pair<GpuBuffer, uint32_t> GpuOps::compactU32Mask(MTL::Buffer* mask, uint32_t totalRows) {
    return compactU32Deterministic(mask, totalRows);
}

// ── Fill & Initialize ──────────────────────────────────────────────────

void GpuOps::fillU32(MTL::Buffer* buf, uint32_t val, uint32_t count) {
    auto* dev = GpuColumnStore::instance().device();
    auto* lib = GpuColumnStore::instance().library();
    auto* cmd = GpuColumnStore::instance().queue()->commandBuffer();
    auto* enc = cmd->computeCommandEncoder();
    
    auto* pso = makePSO(dev, lib, "ops::fill_u32");
    if(!pso) { enc->endEncoding(); return; }
    
    enc->setComputePipelineState(pso);
    enc->setBuffer(buf, 0, 0);
    enc->setBytes(&val, sizeof(uint32_t), 1);
    enc->setBytes(&count, sizeof(uint32_t), 2);
    
    dispatch1D(enc, count);
    enc->endEncoding();
    cmd->commit();
}

GpuBuffer GpuOps::createFilledU32(uint32_t val, uint32_t count) {
    auto* dev = GpuColumnStore::instance().device();
    auto* buf = dev->newBuffer(count * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    fillU32(buf, val, count);
    return GpuBuffer(buf);
}

void GpuOps::fillF32(MTL::Buffer* buf, float val, uint32_t count) {
    auto* dev = GpuColumnStore::instance().device();
    auto* lib = GpuColumnStore::instance().library();
    auto* cmd = GpuColumnStore::instance().queue()->commandBuffer();
    auto* enc = cmd->computeCommandEncoder();
    
    auto* pso = makePSO(dev, lib, "ops::fill_f32");
    if(!pso) { enc->endEncoding(); return; }
    
    enc->setComputePipelineState(pso);
    enc->setBuffer(buf, 0, 0);
    enc->setBytes(&val, sizeof(float), 1);
    enc->setBytes(&count, sizeof(uint32_t), 2);
    
    dispatch1D(enc, count);
    enc->endEncoding();
    cmd->commit();
}

GpuBuffer GpuOps::createFilledF32(float val, uint32_t count) {
    auto* dev = GpuColumnStore::instance().device();
    auto* buf = dev->newBuffer(count * sizeof(float), MTL::ResourceStorageModeShared);
    fillF32(buf, val, count);
    return GpuBuffer(buf);
}

GpuBuffer GpuOps::iotaU32(uint32_t count) {
    auto* dev = GpuColumnStore::instance().device();
    auto* lib = GpuColumnStore::instance().library();
    auto* cmd = GpuColumnStore::instance().queue()->commandBuffer();
    auto* enc = cmd->computeCommandEncoder();
    
    auto* pso = makePSO(dev, lib, "ops::iota_u32");
    if(!pso) { enc->endEncoding(); return {}; }
    
    auto* buf = dev->newBuffer(count * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    
    enc->setComputePipelineState(pso);
    enc->setBuffer(buf, 0, 0);
    enc->setBytes(&count, sizeof(uint32_t), 1);
    
    dispatch1D(enc, count);
    enc->endEncoding();
    cmd->commit();
    
    return GpuBuffer(buf);
}




// (see also: bitcastF32ToU32 below in Conversion & Bitwise)

GpuBuffer GpuOps::bitcastF32ToU32(MTL::Buffer* in, uint32_t count) {
    auto& store = GpuColumnStore::instance();
    auto out = store.device()->newBuffer(count * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    auto p = makePSO(store.device(), store.library(), "ops::bitcast_f32_to_u32");
    if (!p) {
        // CPU fallback: memcpy bitcast
        const float* src = (const float*)in->contents();
        uint32_t* dst = static_cast<uint32_t*>(out->contents());
        for (uint32_t i = 0; i < count; ++i) std::memcpy(&dst[i], &src[i], sizeof(uint32_t));
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
    return GpuBuffer(out);
}




// ── Arithmetic ─────────────────────────────────────────────────────────

GpuBuffer GpuOps::arithAddConstU32(MTL::Buffer* in, uint32_t val, uint32_t count) {
    auto& store = GpuColumnStore::instance();
    auto p = makePSO(store.device(), store.library(), "ops::arith_add_const_u32");
    if (!p) return {};

    auto out = store.device()->newBuffer(static_cast<size_t>(count) * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    {
        auto cmd = store.queue()->commandBuffer();
        auto enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(p);
        enc->setBuffer(in, 0, 0);
        enc->setBytes(&val, sizeof(val), 1);
        enc->setBuffer(out, 0, 2);
        enc->setBytes(&count, sizeof(count), 3);
        dispatch1D(enc, count);
        enc->endEncoding();
        cmd->commit();
        if (!isBatchActive()) {
            cmd->waitUntilCompleted();
            checkGpuStatus(cmd, "arithAddConstU32");
        }
    }
    return GpuBuffer(out);
}

GpuBuffer GpuOps::nonNullIndicatorF32(MTL::Buffer* in, uint32_t count) {
    auto& store = GpuColumnStore::instance();
    auto p = makePSO(store.device(), store.library(), "ops::nonnull_indicator_f32");
    if (!p) return {};

    auto out = store.device()->newBuffer(static_cast<size_t>(count) * sizeof(float), MTL::ResourceStorageModeShared);
    {
        auto cmd = store.queue()->commandBuffer();
        auto enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(p);
        enc->setBuffer(in, 0, 0);
        enc->setBuffer(out, 0, 1);
        enc->setBytes(&count, sizeof(count), 2);
        dispatch1D(enc, count);
        enc->endEncoding();
        cmd->commit();
        if (!isBatchActive()) {
            cmd->waitUntilCompleted();
            checkGpuStatus(cmd, "nonNullIndicatorF32");
        }
    }
    return GpuBuffer(out);
}

// Shared helper for binary f32 arithmetic kernels (ColCol / ColScalar / ScalarCol).
enum class ArithBindKind { ColCol, ColScalar, ScalarCol };

static GpuBuffer arithF32Dispatch(
    const char* kernelName, ArithBindKind kind,
    MTL::Buffer* a, MTL::Buffer* b, float scalarVal, uint32_t count)
{
    auto& store = GpuColumnStore::instance();
    auto p = makePSO(store.device(), store.library(), kernelName);
    if (!p) return {};

    auto out = store.device()->newBuffer(
        static_cast<size_t>(count) * sizeof(float), MTL::ResourceStorageModeShared);
    {
        auto cmd = store.queue()->commandBuffer();
        auto enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(p);
        switch (kind) {
            case ArithBindKind::ColCol:
                enc->setBuffer(a, 0, 0);
                enc->setBuffer(b, 0, 1);
                break;
            case ArithBindKind::ColScalar:
                enc->setBuffer(a, 0, 0);
                enc->setBytes(&scalarVal, sizeof(scalarVal), 1);
                break;
            case ArithBindKind::ScalarCol:
                enc->setBytes(&scalarVal, sizeof(scalarVal), 0);
                enc->setBuffer(b, 0, 1);
                break;
        }
        enc->setBuffer(out, 0, 2);
        enc->setBytes(&count, sizeof(count), 3);
        dispatch1D(enc, count);
        enc->endEncoding();
        cmd->commit();
        if (!GpuOps::isBatchActive()) {
            cmd->waitUntilCompleted();
            checkGpuStatus(cmd, kernelName);
        }
    }
    return GpuBuffer(out);
}

GpuBuffer GpuOps::arithMulF32ColCol(MTL::Buffer* colA, MTL::Buffer* colB, uint32_t count) {
    return arithF32Dispatch("ops::arith_mul_f32_col_col", ArithBindKind::ColCol, colA, colB, 0, count);
}
GpuBuffer GpuOps::arithMulF32ColScalar(MTL::Buffer* colA, float valB, uint32_t count) {
    return arithF32Dispatch("ops::arith_mul_f32_col_scalar", ArithBindKind::ColScalar, colA, nullptr, valB, count);
}
GpuBuffer GpuOps::arithDivF32ColCol(MTL::Buffer* colA, MTL::Buffer* colB, uint32_t count) {
    return arithF32Dispatch("ops::arith_div_f32_col_col", ArithBindKind::ColCol, colA, colB, 0, count);
}
GpuBuffer GpuOps::arithDivF32ColScalar(MTL::Buffer* colA, float valB, uint32_t count) {
    return arithF32Dispatch("ops::arith_div_f32_col_scalar", ArithBindKind::ColScalar, colA, nullptr, valB, count);
}
GpuBuffer GpuOps::arithDivF32ScalarCol(float valA, MTL::Buffer* colB, uint32_t count) {
    return arithF32Dispatch("ops::arith_div_f32_scalar_col", ArithBindKind::ScalarCol, nullptr, colB, valA, count);
}
GpuBuffer GpuOps::arithSubF32ColCol(MTL::Buffer* colA, MTL::Buffer* colB, uint32_t count) {
    return arithF32Dispatch("ops::arith_sub_f32_col_col", ArithBindKind::ColCol, colA, colB, 0, count);
}
GpuBuffer GpuOps::arithSubF32ColScalar(MTL::Buffer* colA, float valB, uint32_t count) {
    return arithF32Dispatch("ops::arith_sub_f32_col_scalar", ArithBindKind::ColScalar, colA, nullptr, valB, count);
}
GpuBuffer GpuOps::arithSubF32ScalarCol(float valA, MTL::Buffer* colB, uint32_t count) {
    return arithF32Dispatch("ops::arith_sub_f32_scalar_col", ArithBindKind::ScalarCol, nullptr, colB, valA, count);
}

GpuBuffer GpuOps::createBuffer(const void* data, size_t size) {
    auto& store = GpuColumnStore::instance();
    if (!store.device()) return {};
    if (data) {
        return GpuBuffer(store.device()->newBuffer(data, size, MTL::ResourceStorageModeShared));
    } else {
        return GpuBuffer(store.device()->newBuffer(size, MTL::ResourceStorageModeShared));
    }
}

// ── Reduction ──────────────────────────────────────────────────────────

static MTL::Buffer* getReduceOutBuf() {
    static std::mutex mu;
    std::lock_guard<std::mutex> lk(mu);
    static MTL::Buffer* s_reduceOut = nullptr;
    if (!s_reduceOut) {
        auto& store = GpuColumnStore::instance();
        s_reduceOut = store.device()->newBuffer(sizeof(float), MTL::ResourceStorageModeShared);
    }
    return s_reduceOut;
}

float GpuOps::reduceSumF32(MTL::Buffer* in, uint32_t count) {
    auto& store = GpuColumnStore::instance();
    auto p = makePSO(store.device(), store.library(), "ops::reduce_sum_f32");
    if (!p) return 0.0f;

    auto out = getReduceOutBuf();
    std::memset(out->contents(), 0, sizeof(float));

    {
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
        checkGpuStatus(cmd, "reduceSumF32");
    }
    
    float res = *static_cast<float*>(out->contents());
    return res;
}

float GpuOps::reduceMinF32(MTL::Buffer* in, uint32_t count) {
    auto& store = GpuColumnStore::instance();
    auto p = makePSO(store.device(), store.library(), "ops::reduce_min_f32");
    if (!p) return 0.0f;

    auto out = getReduceOutBuf();
    float init = std::numeric_limits<float>::max();
    std::memcpy(out->contents(), &init, sizeof(float));

    {
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
        checkGpuStatus(cmd, "reduceMinF32");
    }
    
    float res = *static_cast<float*>(out->contents());
    return res;
}

float GpuOps::reduceMaxF32(MTL::Buffer* in, uint32_t count) {
    auto& store = GpuColumnStore::instance();
    auto p = makePSO(store.device(), store.library(), "ops::reduce_max_f32");
    if (!p) return 0.0f;

    auto out = getReduceOutBuf();
    float init = std::numeric_limits<float>::lowest();
    std::memcpy(out->contents(), &init, sizeof(float));

    {
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
        checkGpuStatus(cmd, "reduceMaxF32");
    }
    
    float res = *static_cast<float*>(out->contents());
    return res;
}

// ── Date ───────────────────────────────────────────────────────────────

GpuBuffer GpuOps::extractYearU32(MTL::Buffer* dateCol, uint32_t count) {
    if (count == 0) return {};
    auto& store = GpuColumnStore::instance();
    auto dev = store.device();
    auto lib = store.library();
    if (!dev || !lib || !store.queue()) return {};

    MTL::Buffer* outBuf = dev->newBuffer(count * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    if (!outBuf) return {};

    auto pso = makePSO(dev, lib, "ops::extract_year_u32_to_u32");
    if (!pso) { outBuf->release(); return {}; }

    auto cmd = store.queue()->commandBuffer();
    auto enc = cmd->computeCommandEncoder();
    enc->setComputePipelineState(pso);
    enc->setBuffer(dateCol, 0, 0);
    enc->setBuffer(outBuf, 0, 1);
    enc->setBytes(&count, sizeof(count), 2);
    dispatch1D(enc, count);
    enc->endEncoding();
    cmd->commit();
    if (!isBatchActive()) {
        cmd->waitUntilCompleted();
        checkGpuStatus(cmd, "extractYearU32");
    }

    return GpuBuffer(outBuf);
}

// ── String Operations ──────────────────────────────────────────────────

std::pair<GpuBuffer, GpuBuffer> GpuOps::substringFlat(
    MTL::Buffer* inOffsets, MTL::Buffer* inLengths,
    uint32_t startPos, uint32_t substrLen, uint32_t rowCount) {
    if (rowCount == 0) return {{}, {}};
    auto& store = GpuColumnStore::instance();
    auto dev = store.device();
    auto lib = store.library();
    if (!dev || !lib || !store.queue()) return {{}, {}};

    auto outOffsets = dev->newBuffer(rowCount * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    auto outLengths = dev->newBuffer(rowCount * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    if (!outOffsets || !outLengths) {
        if (outOffsets) outOffsets->release();
        if (outLengths) outLengths->release();
        return {{}, {}};
    }

    auto pso = makePSO(dev, lib, "ops::substring_flat");
    if (!pso) { outOffsets->release(); outLengths->release(); return {{}, {}}; }

    auto cmd = store.queue()->commandBuffer();
    auto enc = cmd->computeCommandEncoder();
    enc->setComputePipelineState(pso);
    enc->setBuffer(inOffsets, 0, 0);
    enc->setBuffer(inLengths, 0, 1);
    enc->setBuffer(outOffsets, 0, 2);
    enc->setBuffer(outLengths, 0, 3);
    enc->setBytes(&startPos, sizeof(startPos), 4);
    enc->setBytes(&substrLen, sizeof(substrLen), 5);
    enc->setBytes(&rowCount, sizeof(rowCount), 6);
    dispatch1D(enc, rowCount);
    enc->endEncoding();
    cmd->commit();
    cmd->waitUntilCompleted();
    checkGpuStatus(cmd, "substringFlat");

    return {GpuBuffer(outOffsets), GpuBuffer(outLengths)};
}

GpuBuffer GpuOps::stringHashEncodeU32(
    MTL::Buffer* chars, MTL::Buffer* offsets, MTL::Buffer* lengths, uint32_t rowCount) {
    if (rowCount == 0) return {};
    auto& store = GpuColumnStore::instance();
    auto dev = store.device();
    auto lib = store.library();
    if (!dev || !lib || !store.queue()) return {};

    auto outBuf = dev->newBuffer(rowCount * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    if (!outBuf) return {};

    auto pso = makePSO(dev, lib, "ops::string_hash_encode_u32");
    if (!pso) { outBuf->release(); return {}; }

    auto cmd = store.queue()->commandBuffer();
    auto enc = cmd->computeCommandEncoder();
    enc->setComputePipelineState(pso);
    enc->setBuffer(chars, 0, 0);
    enc->setBuffer(offsets, 0, 1);
    enc->setBuffer(lengths, 0, 2);
    enc->setBuffer(outBuf, 0, 3);
    enc->setBytes(&rowCount, sizeof(rowCount), 4);
    dispatch1D(enc, rowCount);
    enc->endEncoding();
    cmd->commit();

    return GpuBuffer(outBuf);
}

GpuBuffer GpuOps::stringFnv1aU32(
    MTL::Buffer* chars, MTL::Buffer* offsets, MTL::Buffer* lengths, uint32_t rowCount) {
    if (rowCount == 0) return {};
    auto& store = GpuColumnStore::instance();
    auto dev = store.device();
    auto lib = store.library();
    if (!dev || !lib || !store.queue()) return {};

    auto outBuf = dev->newBuffer(rowCount * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    if (!outBuf) return {};

    auto pso = makePSO(dev, lib, "ops::string_fnv1a_u32");
    if (!pso) { outBuf->release(); return {}; }

    auto cmd = store.queue()->commandBuffer();
    auto enc = cmd->computeCommandEncoder();
    enc->setComputePipelineState(pso);
    enc->setBuffer(chars, 0, 0);
    enc->setBuffer(offsets, 0, 1);
    enc->setBuffer(lengths, 0, 2);
    enc->setBuffer(outBuf, 0, 3);
    enc->setBytes(&rowCount, sizeof(rowCount), 4);
    dispatch1D(enc, rowCount);
    enc->endEncoding();
    cmd->commit();

    return GpuBuffer(outBuf);
}

GpuOps::FlatStringGatherResult GpuOps::gatherFlatString(
    MTL::Buffer* srcChars, MTL::Buffer* srcOffsets, MTL::Buffer* srcLengths,
    MTL::Buffer* indices, uint32_t count, bool doSync) {
    FlatStringGatherResult r;
    if (count == 0 || !srcChars || !srcOffsets || !srcLengths || !indices) return r;

    auto& store = GpuColumnStore::instance();
    auto dev = store.device();
    if (!dev || !store.queue()) return r;

    auto start = std::chrono::high_resolution_clock::now();

    // Step 1: Gather lengths  (outLengths[i] = srcLengths[indices[i]])
    GpuBuffer outLengths = gatherU32(srcLengths, indices, count, true);
    if (!outLengths) return r;

    // Step 2: Prefix-sum on lengths → outOffsets (exclusive scan)
    // We need a copy for the scan since scanInPlace modifies in-place.
    GpuBuffer outOffsets(dev->newBuffer(count * sizeof(uint32_t), MTL::ResourceStorageModeShared));
    if (!outOffsets) { return r; }
    std::memcpy(outOffsets->contents(), outLengths->contents(), count * sizeof(uint32_t));
    uint64_t totalBytes = scanInPlace(outOffsets, count);

    // Step 3: Allocate output chars and dispatch char-copy kernel
    // Guard against >4GB overflow (uint32_t max)
    if (totalBytes > (uint64_t)UINT32_MAX) return r;
    uint32_t totalBytesU32 = static_cast<uint32_t>(totalBytes);
    size_t allocBytes = (totalBytesU32 > 0) ? totalBytesU32 : 1;
    GpuBuffer outChars(dev->newBuffer(allocBytes, MTL::ResourceStorageModeShared));
    if (!outChars) { return r; }

    if (totalBytesU32 > 0) {
        auto pso = makePSO(dev, store.library(), "ops::gather_flat_string_chars");
        if (!pso) { return r; }

        auto cmd = store.queue()->commandBuffer();
        auto enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(pso);
        enc->setBuffer(srcChars, 0, 0);
        enc->setBuffer(srcOffsets, 0, 1);
        enc->setBuffer(indices, 0, 2);
        enc->setBuffer(outOffsets, 0, 3);
        enc->setBuffer(outLengths, 0, 4);
        enc->setBuffer(outChars, 0, 5);
        enc->setBytes(&count, sizeof(count), 6);
        dispatch1D(enc, count);
        enc->endEncoding();
        cmd->commit();
        if (doSync) {
            cmd->waitUntilCompleted();
            checkGpuStatus(cmd, "gatherFlatString");
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    KernelTimer::instance().record("ops::gather_flat_string", "gather",
        std::chrono::duration<double, std::milli>(end - start).count(), count);

    r.chars = std::move(outChars);
    r.offsets = std::move(outOffsets);
    r.lengths = std::move(outLengths);
    r.rowCount = count;
    r.totalBytes = totalBytesU32;
    return r;
}

GpuBuffer GpuOps::stringFnv1aU64Fold32(
    MTL::Buffer* chars, MTL::Buffer* offsets, MTL::Buffer* lengths, uint32_t rowCount) {
    if (rowCount == 0) return {};
    auto& store = GpuColumnStore::instance();
    auto dev = store.device();
    auto lib = store.library();
    if (!dev || !lib || !store.queue()) return {};

    // Step 1: compute 64-bit hashes, Step 2: XOR-fold to u32
    // Both kernels in a single command buffer (serial queue ordering).
    auto hashBuf64 = dev->newBuffer(rowCount * sizeof(uint64_t), MTL::ResourceStorageModeShared);
    if (!hashBuf64) return {};

    auto pso1 = makePSO(dev, lib, "ops::string_fnv1a_u64");
    if (!pso1) { hashBuf64->release(); return {}; }

    auto outBuf = dev->newBuffer(rowCount * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    if (!outBuf) { hashBuf64->release(); return {}; }

    auto pso2 = makePSO(dev, lib, "ops::fold_u64_to_u32");
    if (!pso2) { hashBuf64->release(); outBuf->release(); return {}; }

    auto cmd = store.queue()->commandBuffer();

    auto enc1 = cmd->computeCommandEncoder();
    enc1->setComputePipelineState(pso1);
    enc1->setBuffer(chars, 0, 0);
    enc1->setBuffer(offsets, 0, 1);
    enc1->setBuffer(lengths, 0, 2);
    enc1->setBuffer(hashBuf64, 0, 3);
    enc1->setBytes(&rowCount, sizeof(rowCount), 4);
    dispatch1D(enc1, rowCount);
    enc1->endEncoding();

    auto enc2 = cmd->computeCommandEncoder();
    enc2->setComputePipelineState(pso2);
    enc2->setBuffer(hashBuf64, 0, 0);
    enc2->setBuffer(outBuf, 0, 1);
    enc2->setBytes(&rowCount, sizeof(rowCount), 2);
    dispatch1D(enc2, rowCount);
    enc2->endEncoding();

    cmd->commit();

    hashBuf64->release();
    return GpuBuffer(outBuf);
}

// Helper: dispatch string_prefix_u64_gathered kernel
static MTL::Buffer* dispatchPrefixGathered(
    MTL::Device* dev, MTL::Library* lib, MTL::CommandQueue* queue,
    MTL::Buffer* chars, MTL::Buffer* offsets, MTL::Buffer* lengths,
    MTL::Buffer* indices, uint32_t rowCount, uint32_t byteOffset) {
    auto buf = dev->newBuffer(rowCount * sizeof(uint64_t), MTL::ResourceStorageModeShared);
    if (!buf) return nullptr;
    auto pso = makePSO(dev, lib, "ops::string_prefix_u64_gathered");
    if (!pso) { buf->release(); return nullptr; }
    auto cmd = queue->commandBuffer();
    auto enc = cmd->computeCommandEncoder();
    enc->setComputePipelineState(pso);
    enc->setBuffer(chars, 0, 0);
    enc->setBuffer(offsets, 0, 1);
    enc->setBuffer(lengths, 0, 2);
    enc->setBuffer(indices, 0, 3);
    enc->setBuffer(buf, 0, 4);
    enc->setBytes(&rowCount, sizeof(rowCount), 5);
    enc->setBytes(&byteOffset, sizeof(byteOffset), 6);
    dispatch1D(enc, rowCount);
    enc->endEncoding();
    cmd->commit();
    return buf;
}

GpuBuffer GpuOps::stringRankByPrefix(
    MTL::Buffer* chars, MTL::Buffer* offsets, MTL::Buffer* lengths,
    uint32_t rowCount, bool ascending) {
    if (rowCount == 0) return {};
    auto& store = GpuColumnStore::instance();
    auto dev = store.device();
    auto lib = store.library();
    if (!dev || !lib || !store.queue()) return {};

    // 16-byte two-pass stable radix sort:
    // Sort by secondary key (bytes 8-15) first, then primary key (bytes 0-7).
    // Stable sort preserves secondary ordering within equal primary keys.

    // Step 1: compute lo = prefix bytes 8-15 (identity indices)
    GpuBuffer idxBuf = iotaU32(rowCount);
    uint32_t byteOff8 = 8;
    MTL::Buffer* loBuf = dispatchPrefixGathered(
        dev, lib, store.queue(), chars, offsets, lengths, idxBuf, rowCount, byteOff8);
    if (!loBuf) return {};

    // Step 2: radix sort by secondary key (lo) — stable
    radixSortU64(loBuf, idxBuf, rowCount);

    // Step 3: compute hi = prefix bytes 0-7 gathered by sorted indices
    uint32_t byteOff0 = 0;
    MTL::Buffer* hiBuf = dispatchPrefixGathered(
        dev, lib, store.queue(), chars, offsets, lengths, idxBuf, rowCount, byteOff0);
    if (!hiBuf) { loBuf->release(); return {}; }

    // Step 4: radix sort by primary key (hi) — stable preserves lo ordering
    radixSortU64(hiBuf, idxBuf, rowCount);

    // Step 5: recompute lo gathered by final indices (for boundary marking)
    loBuf->release();
    loBuf = dispatchPrefixGathered(
        dev, lib, store.queue(), chars, offsets, lengths, idxBuf, rowCount, byteOff8);
    if (!loBuf) { hiBuf->release(); return {}; }

    // Step 6: mark boundaries using both hi and lo keys
    auto markBuf = dev->newBuffer(rowCount * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    if (!markBuf) { hiBuf->release(); loBuf->release(); return {}; }

    {
        auto pso = makePSO(dev, lib, "ops::mark_sorted_boundaries_2xu64");
        if (!pso) { hiBuf->release(); loBuf->release(); markBuf->release(); return {}; }
        auto cmd = store.queue()->commandBuffer();
        auto enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(pso);
        enc->setBuffer(hiBuf, 0, 0);
        enc->setBuffer(loBuf, 0, 1);
        enc->setBuffer(markBuf, 0, 2);
        enc->setBytes(&rowCount, sizeof(rowCount), 3);
        dispatch1D(enc, rowCount);
        enc->endEncoding();
        cmd->commit();
    }

    hiBuf->release();
    loBuf->release();

    // Step 7: prefix sum on boundary marks → cumulative ranks
    uint64_t uniqueCount = scanInPlace(markBuf, rowCount);

    // Check for ties: if uniqueCount < rowCount-1, 16-byte prefixes still tied
    if (rowCount > 1 && uniqueCount < rowCount - 1) {
        markBuf->release();
        return {};
    }

    // Step 8: scatter ranks to original row positions
    auto rankBuf = dev->newBuffer(rowCount * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    if (!rankBuf) { markBuf->release(); return {}; }

    {
        auto pso = makePSO(dev, lib, "ops::scatter_rank_by_index");
        if (!pso) { markBuf->release(); rankBuf->release(); return {}; }
        auto cmd = store.queue()->commandBuffer();
        auto enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(pso);
        enc->setBuffer(markBuf, 0, 0);
        enc->setBuffer(idxBuf, 0, 1);
        enc->setBuffer(rankBuf, 0, 2);
        enc->setBytes(&rowCount, sizeof(rowCount), 3);
        dispatch1D(enc, rowCount);
        enc->endEncoding();
        cmd->commit();
    }

    markBuf->release();

    // Step 9: if descending, invert ranks
    if (!ascending) {
        GpuBuffer inv = invertU32(rankBuf, rowCount);
        rankBuf->release();
        return inv;
    }

    return GpuBuffer(rankBuf);
}

GpuOps::FlatStringGatherResult GpuOps::concatFlatStrings(
    const FlatStringGatherResult& a, const FlatStringGatherResult& b) {
    FlatStringGatherResult r;
    if (!a.chars && !b.chars) return r;
    if (!a.chars || a.rowCount == 0) {
        // Only b — copy it
        if (b.chars) b.chars->retain();
        if (b.offsets) b.offsets->retain();
        if (b.lengths) b.lengths->retain();
        r.chars.reset(b.chars.get());
        r.offsets.reset(b.offsets.get());
        r.lengths.reset(b.lengths.get());
        r.rowCount = b.rowCount;
        r.totalBytes = b.totalBytes;
        return r;
    }
    if (!b.chars || b.rowCount == 0) {
        if (a.chars) a.chars->retain();
        if (a.offsets) a.offsets->retain();
        if (a.lengths) a.lengths->retain();
        r.chars.reset(a.chars.get());
        r.offsets.reset(a.offsets.get());
        r.lengths.reset(a.lengths.get());
        r.rowCount = a.rowCount;
        r.totalBytes = a.totalBytes;
        return r;
    }

    auto& store = GpuColumnStore::instance();
    auto dev = store.device();
    auto lib = store.library();
    if (!dev || !lib || !store.queue()) return r;

    uint32_t totalRows = a.rowCount + b.rowCount;
    uint32_t totalBytes = a.totalBytes + b.totalBytes;

    // Concatenate chars: memcpy a then b (shared memory, zero-copy safe)
    size_t allocBytes = totalBytes > 0 ? totalBytes : 1;
    GpuBuffer combinedChars(dev->newBuffer(allocBytes, MTL::ResourceStorageModeShared));
    if (a.totalBytes > 0)
        std::memcpy(combinedChars->contents(), a.chars->contents(), a.totalBytes);
    if (b.totalBytes > 0)
        std::memcpy(static_cast<uint8_t*>(combinedChars->contents()) + a.totalBytes,
                    b.chars->contents(), b.totalBytes);

    // Concatenate lengths: memcpy a then b
    GpuBuffer combinedLengths(dev->newBuffer(totalRows * sizeof(uint32_t), MTL::ResourceStorageModeShared));
    std::memcpy(combinedLengths->contents(), a.lengths->contents(), a.rowCount * sizeof(uint32_t));
    std::memcpy(static_cast<uint8_t*>(combinedLengths->contents()) + a.rowCount * sizeof(uint32_t),
                b.lengths->contents(), b.rowCount * sizeof(uint32_t));

    // Concatenate offsets: a offsets unchanged, b offsets shifted by a.totalBytes
    GpuBuffer combinedOffsets(dev->newBuffer(totalRows * sizeof(uint32_t), MTL::ResourceStorageModeShared));
    std::memcpy(combinedOffsets->contents(), a.offsets->contents(), a.rowCount * sizeof(uint32_t));

    if (b.rowCount > 0) {
        // Shift b offsets by a.totalBytes using GPU kernel
        auto pso = makePSO(dev, lib, "ops::offset_shift_u32");
        if (pso) {
            // Write shifted offsets directly to the second half of combinedOffsets
            auto tmpShifted = dev->newBuffer(b.rowCount * sizeof(uint32_t), MTL::ResourceStorageModeShared);
            auto cmd = store.queue()->commandBuffer();
            auto enc = cmd->computeCommandEncoder();
            enc->setComputePipelineState(pso);
            enc->setBuffer(b.offsets, 0, 0);
            enc->setBuffer(tmpShifted, 0, 1);
            enc->setBytes(&a.totalBytes, sizeof(a.totalBytes), 2);
            enc->setBytes(&b.rowCount, sizeof(b.rowCount), 3);
            dispatch1D(enc, b.rowCount);
            enc->endEncoding();
            cmd->commit();
            cmd->waitUntilCompleted();
            checkGpuStatus(cmd, "offsetShift");
            std::memcpy(static_cast<uint8_t*>(combinedOffsets->contents()) + a.rowCount * sizeof(uint32_t),
                        tmpShifted->contents(), b.rowCount * sizeof(uint32_t));
            tmpShifted->release();
        } else {
            // CPU fallback for offset shift
            const auto* bOffs = static_cast<const uint32_t*>(b.offsets->contents());
            auto* dst = static_cast<uint32_t*>(combinedOffsets->contents()) + a.rowCount;
            for (uint32_t i = 0; i < b.rowCount; i++)
                dst[i] = bOffs[i] + a.totalBytes;
        }
    }

    r.chars = std::move(combinedChars);
    r.offsets = std::move(combinedOffsets);
    r.lengths = std::move(combinedLengths);
    r.rowCount = totalRows;
    r.totalBytes = totalBytes;
    return r;
}

// (continued: Arithmetic — arithAdd, scatter, math)

GpuBuffer GpuOps::arithAddF32ColCol(MTL::Buffer* colA, MTL::Buffer* colB, uint32_t count) {
    return arithF32Dispatch("ops::arith_add_f32_col_col", ArithBindKind::ColCol, colA, colB, 0, count);
}
GpuBuffer GpuOps::arithAddF32ColScalar(MTL::Buffer* colA, float valB, uint32_t count) {
    return arithF32Dispatch("ops::arith_add_f32_col_scalar", ArithBindKind::ColScalar, colA, nullptr, valB, count);
}

void GpuOps::scatterConstantF32(MTL::Buffer* output, MTL::Buffer* indices, uint32_t indexCount, float val) {
    if (indexCount == 0 || !output || !indices) return;

    auto& store = GpuColumnStore::instance();

    auto p = makePSO(store.device(), store.library(), "ops::scatter_constant_f32");
    if (!p) {
        // Fallback or debug
        LOG_ERROR("GPU", "function not found: ops::scatter_constant_f32");
        return;
    }

    auto cmd = store.queue()->commandBuffer();
    auto enc = cmd->computeCommandEncoder();
    enc->setComputePipelineState(p);
    enc->setBuffer(output, 0, 0);
    enc->setBuffer(indices, 0, 1);
    enc->setBytes(&val, sizeof(val), 2);
    enc->setBytes(&indexCount, sizeof(indexCount), 3);
    
    MTL::Size grp = MTL::Size::Make(256, 1, 1);
    MTL::Size grd = MTL::Size::Make((indexCount + 255) / 256, 1, 1);
    enc->dispatchThreadgroups(grd, grp);
    
    enc->endEncoding();
    cmd->commit();
}

void GpuOps::scatterF32(MTL::Buffer* input, MTL::Buffer* output, MTL::Buffer* indices, uint32_t count) {
    if (count == 0 || !input || !output || !indices) return;

    auto& store = GpuColumnStore::instance();
    auto p = makePSO(store.device(), store.library(), "ops::scatter_f32_indexed");
    if (!p) {
        LOG_ERROR("GPU", "function not found: ops::scatter_f32_indexed");
        return;
    }

    auto cmd = store.queue()->commandBuffer();
    auto enc = cmd->computeCommandEncoder();
    enc->setComputePipelineState(p);
    enc->setBuffer(input, 0, 0);
    enc->setBuffer(indices, 0, 1);
    enc->setBuffer(output, 0, 2);
    enc->setBytes(&count, sizeof(count), 3);
    
    MTL::Size grp = MTL::Size::Make(256, 1, 1);
    MTL::Size grd = MTL::Size::Make((count + 255) / 256, 1, 1);
    enc->dispatchThreadgroups(grd, grp);
    
    enc->endEncoding();
    cmd->commit();
}

GpuBuffer GpuOps::mathFloorF32(MTL::Buffer* col, uint32_t count) {
    auto& store = GpuColumnStore::instance();
    auto p = makePSO(store.device(), store.library(), "ops::arith_floor_f32");
    if (!p) return {};

    auto out = store.device()->newBuffer(static_cast<size_t>(count) * sizeof(float), MTL::ResourceStorageModeShared);
    {
        auto cmd = store.queue()->commandBuffer();
        auto enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(p);
        enc->setBuffer(col, 0, 0);
        enc->setBuffer(out, 0, 1);
        enc->setBytes(&count, sizeof(count), 2);
        dispatch1D(enc, count);
        enc->endEncoding();
        cmd->commit();
        if (!isBatchActive()) {
            cmd->waitUntilCompleted();
            checkGpuStatus(cmd, "mathFloorF32");
        }
    }
    return GpuBuffer(out);
}

} // namespace engine

