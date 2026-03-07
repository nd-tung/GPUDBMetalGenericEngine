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
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <unordered_map>

namespace engine {


// Performs in-place exclusive scan on 'data' (u32). 
// Returns the total sum (reduction).


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

// --- File loaders ---

// ============================================================================
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
                            catch (...) {}
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

// ============================================================================
// GPU Arithmetic Batch Mode
// When active, arithmetic ops skip waitUntilCompleted() — the serial command
// queue guarantees ordering. A sync is performed at the end of the batch.
// ============================================================================
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

// --- Schema ---

struct ColMeta {
    int idx;
    enum class Kind { U32, F32, DateU32, StrCharU32 } kind;  // StrCharU32 = single-char reversible
};

static const std::map<std::string, std::map<std::string, ColMeta>> kTpchSchema = {
    {"customer", {
        {"c_custkey", {0, ColMeta::Kind::U32}},
        {"c_nationkey", {3, ColMeta::Kind::U32}},
        {"c_acctbal", {5, ColMeta::Kind::F32}},
    }},
    {"orders", {
        {"o_orderkey", {0, ColMeta::Kind::U32}},
        {"o_custkey", {1, ColMeta::Kind::U32}},
        {"o_orderstatus", {2, ColMeta::Kind::StrCharU32}},  // Single char: F/O/P
        {"o_totalprice", {3, ColMeta::Kind::F32}},
        {"o_orderdate", {4, ColMeta::Kind::DateU32}},
        {"o_shippriority", {7, ColMeta::Kind::U32}},
    }},
    {"lineitem", {
        {"l_orderkey", {0, ColMeta::Kind::U32}},
        {"l_partkey", {1, ColMeta::Kind::U32}},
        {"l_suppkey", {2, ColMeta::Kind::U32}},
        {"l_linenumber", {3, ColMeta::Kind::U32}},
        {"l_quantity", {4, ColMeta::Kind::F32}},
        {"l_extendedprice", {5, ColMeta::Kind::F32}},
        {"l_discount", {6, ColMeta::Kind::F32}},
        {"l_tax", {7, ColMeta::Kind::F32}},
        {"l_returnflag", {8, ColMeta::Kind::StrCharU32}},   // Single char: A/N/R
        {"l_linestatus", {9, ColMeta::Kind::StrCharU32}},   // Single char: F/O
        {"l_shipdate", {10, ColMeta::Kind::DateU32}},
        {"l_commitdate", {11, ColMeta::Kind::DateU32}},
        {"l_receiptdate", {12, ColMeta::Kind::DateU32}},
    }},
    {"supplier", {
        {"s_suppkey", {0, ColMeta::Kind::U32}},
        {"s_nationkey", {3, ColMeta::Kind::U32}},
        {"s_acctbal", {5, ColMeta::Kind::F32}},
    }},
    {"part", {
        {"p_partkey", {0, ColMeta::Kind::U32}},
        {"p_size", {5, ColMeta::Kind::U32}},
        {"p_retailprice", {7, ColMeta::Kind::F32}},
    }},
    {"partsupp", {
        {"ps_partkey", {0, ColMeta::Kind::U32}},
        {"ps_suppkey", {1, ColMeta::Kind::U32}},
        {"ps_availqty", {2, ColMeta::Kind::U32}},
        {"ps_supplycost", {3, ColMeta::Kind::F32}},
    }},
    {"nation", {
        {"n_nationkey", {0, ColMeta::Kind::U32}},
        {"n_regionkey", {2, ColMeta::Kind::U32}},
    }},
    {"region", {
        {"r_regionkey", {0, ColMeta::Kind::U32}},
    }},
};

GpuRelation GpuOps::scanTable(const std::string& datasetPath,
                                   const std::string& table,
                                   const std::vector<std::string>& neededCols) {
    GpuRelation rel;

    auto& store = GpuColumnStore::instance();
    store.initialize();
    if (!store.device()) return rel;

    const auto itT = kTpchSchema.find(table);
    if (itT == kTpchSchema.end()) return rel;

    std::string path = datasetPath + table + ".tbl";

    auto cache_key = [&](const std::string& colName) {
        return datasetPath + table + "." + colName;
    };

    bool sizeSet = false;
    uint32_t rowCount = 0;

    // Phase 1: Check GpuColumnStore cache for each column; collect uncached ones
    std::vector<ColLoadSpec> uncached;
    // Track which columns are already cached (name -> ColMeta::Kind)
    struct CachedCol { std::string name; ColMeta::Kind kind; };

    for (const auto& c : neededCols) {
        const auto itC = itT->second.find(c);
        if (itC == itT->second.end()) continue;

        const std::string key = cache_key(c);
        GpuColumn* staged = store.getColumn(key);
        if (staged && staged->buffer) {
            // Already cached — use directly
            if (!sizeSet) { rowCount = static_cast<uint32_t>(staged->count); sizeSet = true; }
            if (static_cast<uint32_t>(staged->count) != rowCount) continue;
            staged->buffer->retain();
            if (itC->second.kind == ColMeta::Kind::F32)
                rel.f32cols[c].reset(staged->buffer);
            else
                rel.u32cols[c].reset(staged->buffer);
            continue;
        }

        // Not cached — will load in one-pass
        ColLoadSpec spec;
        spec.columnIndex = itC->second.idx;
        spec.outputName = c;
        switch (itC->second.kind) {
            case ColMeta::Kind::U32:       spec.type = ColLoadSpec::kU32; break;
            case ColMeta::Kind::F32:       spec.type = ColLoadSpec::kF32; break;
            case ColMeta::Kind::DateU32:   spec.type = ColLoadSpec::kDateU32; break;
            case ColMeta::Kind::StrCharU32:spec.type = ColLoadSpec::kStrCharU32; break;
        }
        uncached.push_back(spec);
    }

    // Also opportunistically load StringHash columns from SchemaRegistry
    // in the same file pass, caching them for later loadStringColumnRaw calls.
    const auto& schemaReg = engine::SchemaRegistry::instance();
    const auto* tblSchema = schemaReg.getTable(table);
    if (tblSchema) {
        for (const auto& c : neededCols) {
            // Skip columns already handled via kTpchSchema
            if (itT->second.count(c)) continue;
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
        }
    }

    // Phase 2: Load ALL uncached columns in a SINGLE file pass
    if (!uncached.empty()) {
        auto loaded = loadColumnsOnePass(path, uncached);

        for (const auto& spec : uncached) {
            if (spec.type == ColLoadSpec::kStringRaw) {
                // Cache raw strings for later loadStringColumnRaw calls
                auto it = loaded.strData.find(spec.outputName);
                if (it != loaded.strData.end() && !it->second.empty()) {
                    const auto* cs = tblSchema ? tblSchema->getColumn(spec.outputName) : nullptr;
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











MTL::Buffer* GpuOps::gatherU32(MTL::Buffer* in, MTL::Buffer* indices, uint32_t count, bool sync) {
    auto& store = GpuColumnStore::instance();
    auto p_g = makePSO(store.device(), store.library(), "ops::gather_col_u32");
    if (!p_g) {
        std::cerr << "GpuOps::gatherU32: Failed to create PSO ops::gather_col_u32\n";
        return nullptr;
    }

    if (!in || !indices) {
        std::cerr << "GpuOps::gatherU32: Input or Indices buffer is NULL\n";
        return nullptr;
    }

    auto out = store.device()->newBuffer(count * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    if (!out) {
        std::cerr << "GpuOps::gatherU32: Failed to allocate output buffer size " << (count * sizeof(uint32_t)) << "\n";
        return nullptr;
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
    if (sync) {
        KernelTimer::instance().record("ops::gather_col_u32", "gather",
            std::chrono::duration<double, std::milli>(end - start).count(), count);
    }
    return out;
}

MTL::Buffer* GpuOps::gatherF32(MTL::Buffer* in, MTL::Buffer* indices, uint32_t count, bool sync) {
    auto& store = GpuColumnStore::instance();
    auto p_g = makePSO(store.device(), store.library(), "ops::gather_col_f32");
    if (!p_g) return nullptr;

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
    if (sync) {
        KernelTimer::instance().record("ops::gather_col_f32", "gather",
            std::chrono::duration<double, std::milli>(end - start).count(), count);
    }
    return out;
}

void GpuOps::sync() {
    auto& store = GpuColumnStore::instance();
    auto cmd = store.queue()->commandBuffer();
    cmd->commit();
    cmd->waitUntilCompleted();
}

MTL::Buffer* GpuOps::castU32ToF32(MTL::Buffer* in, uint32_t count) {
    auto& store = GpuColumnStore::instance();
    auto p = makePSO(store.device(), store.library(), "ops::cast_u32_to_f32");
    if (!p) return nullptr;

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
        if (!isBatchActive()) cmd->waitUntilCompleted();
    }
    auto end = std::chrono::high_resolution_clock::now();
    KernelTimer::instance().record("ops::cast_u32_to_f32", "cast",
        std::chrono::duration<double, std::milli>(end - start).count(), count);
    return out;
}



// ── GPU Stream Compaction: extract valid entries from GroupBy hash table ──
// Mark → Prefix Sum → Compact pipeline.


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




MTL::Buffer* GpuOps::logicOrU32(MTL::Buffer* colA, MTL::Buffer* colB, uint32_t count) {
    auto* dev = GpuColumnStore::instance().device();
    auto* lib = GpuColumnStore::instance().library();
    auto* cmd = GpuColumnStore::instance().queue()->commandBuffer();
    auto* enc = cmd->computeCommandEncoder();
    
    auto* pso = makePSO(dev, lib, "ops::logic_or_u32");
    if(!pso) { enc->endEncoding(); return nullptr; }
    
    enc->setComputePipelineState(pso);
    enc->setBuffer(colA, 0, 0);
    enc->setBuffer(colB, 0, 1);
    
    auto* outMask = dev->newBuffer(count * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    enc->setBuffer(outMask, 0, 2);
    
    enc->setBytes(&count, sizeof(uint32_t), 3);
    
    dispatch1D(enc, count);
    enc->endEncoding();
    cmd->commit();
    cmd->waitUntilCompleted();
    checkGpuStatus(cmd, "logicOrU32");
    
    return outMask;
}

MTL::Buffer* GpuOps::logicAndNotU32(MTL::Buffer* colA, MTL::Buffer* colB, uint32_t count) {
    auto* dev = GpuColumnStore::instance().device();
    auto* lib = GpuColumnStore::instance().library();
    auto* cmd = GpuColumnStore::instance().queue()->commandBuffer();
    auto* enc = cmd->computeCommandEncoder();
    
    auto* pso = makePSO(dev, lib, "ops::logic_andnot_u32");
    if(!pso) { enc->endEncoding(); return nullptr; }
    
    enc->setComputePipelineState(pso);
    enc->setBuffer(colA, 0, 0);
    enc->setBuffer(colB, 0, 1);
    
    auto* outMask = dev->newBuffer(count * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    enc->setBuffer(outMask, 0, 2);
    
    enc->setBytes(&count, sizeof(uint32_t), 3);
    
    dispatch1D(enc, count);
    enc->endEncoding();
    cmd->commit();
    cmd->waitUntilCompleted();
    
    return outMask;
}

MTL::Buffer* GpuOps::indicesToMask(MTL::Buffer* indices, uint32_t indexCount, uint32_t totalRows) {
    auto* dev = GpuColumnStore::instance().device();
    auto* lib = GpuColumnStore::instance().library();
    auto* cmd = GpuColumnStore::instance().queue()->commandBuffer();
    
    // Create and clear the mask buffer
    auto* mask = dev->newBuffer(totalRows * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    
    // Clear mask to zeros
    auto* enc1 = cmd->computeCommandEncoder();
    auto* psoClear = makePSO(dev, lib, "ops::clear_mask");
    if(!psoClear) { enc1->endEncoding(); return nullptr; }
    enc1->setComputePipelineState(psoClear);
    enc1->setBuffer(mask, 0, 0);
    enc1->setBytes(&totalRows, sizeof(uint32_t), 1);
    dispatch1D(enc1, totalRows);
    enc1->endEncoding();
    
    // Set mask[indices[i]] = 1
    auto* enc2 = cmd->computeCommandEncoder();
    auto* psoSet = makePSO(dev, lib, "ops::indices_to_mask");
    if(!psoSet) { enc2->endEncoding(); return mask; }
    enc2->setComputePipelineState(psoSet);
    enc2->setBuffer(indices, 0, 0);
    enc2->setBuffer(mask, 0, 1);
    enc2->setBytes(&indexCount, sizeof(uint32_t), 2);
    dispatch1D(enc2, indexCount);
    enc2->endEncoding();
    
    cmd->commit();
    cmd->waitUntilCompleted();
    
    return mask;
}

std::pair<MTL::Buffer*, uint32_t> GpuOps::compactU32Mask(MTL::Buffer* mask, uint32_t totalRows) {
    auto* dev = GpuColumnStore::instance().device();
    auto* lib = GpuColumnStore::instance().library();
    auto* cmd = GpuColumnStore::instance().queue()->commandBuffer();
    auto* enc = cmd->computeCommandEncoder();
    
    auto* pso = makePSO(dev, lib, "ops::compact_u32_mask");
    if(!pso) { enc->endEncoding(); return {nullptr, 0}; }
    
    enc->setComputePipelineState(pso);
    enc->setBuffer(mask, 0, 0);
    
    auto* outIdx = dev->newBuffer(totalRows * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    auto* outCnt = dev->newBuffer(sizeof(uint32_t), MTL::ResourceStorageModeShared);
    std::memset(outCnt->contents(), 0, sizeof(uint32_t));
    
    enc->setBuffer(outIdx, 0, 1);
    enc->setBuffer(outCnt, 0, 2);
    enc->setBytes(&totalRows, sizeof(uint32_t), 3);
    
    dispatch1D(enc, totalRows);
    enc->endEncoding();
    cmd->commit();
    cmd->waitUntilCompleted();
    
    uint32_t count = *reinterpret_cast<uint32_t*>(outCnt->contents());
    outCnt->release();
    
    return {outIdx, count};
}

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
    cmd->waitUntilCompleted();
}

MTL::Buffer* GpuOps::createFilledU32(uint32_t val, uint32_t count) {
    auto* dev = GpuColumnStore::instance().device();
    auto* buf = dev->newBuffer(count * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    fillU32(buf, val, count);
    return buf;
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
    cmd->waitUntilCompleted();
}

MTL::Buffer* GpuOps::createFilledF32(float val, uint32_t count) {
    auto* dev = GpuColumnStore::instance().device();
    auto* buf = dev->newBuffer(count * sizeof(float), MTL::ResourceStorageModeShared);
    fillF32(buf, val, count);
    return buf;
}

MTL::Buffer* GpuOps::iotaU32(uint32_t count) {
    auto* dev = GpuColumnStore::instance().device();
    auto* lib = GpuColumnStore::instance().library();
    auto* cmd = GpuColumnStore::instance().queue()->commandBuffer();
    auto* enc = cmd->computeCommandEncoder();
    
    auto* pso = makePSO(dev, lib, "ops::iota_u32");
    if(!pso) { enc->endEncoding(); return nullptr; }
    
    auto* buf = dev->newBuffer(count * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    
    enc->setComputePipelineState(pso);
    enc->setBuffer(buf, 0, 0);
    enc->setBytes(&count, sizeof(uint32_t), 1);
    
    dispatch1D(enc, count);
    enc->endEncoding();
    cmd->commit();
    cmd->waitUntilCompleted();
    
    return buf;
}




MTL::Buffer* GpuOps::bitcastF32ToU32(MTL::Buffer* in, uint32_t count) {
    auto& store = GpuColumnStore::instance();
    auto out = store.device()->newBuffer(count * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    auto p = makePSO(store.device(), store.library(), "ops::bitcast_f32_to_u32");
    if (!p) {
        // CPU fallback: memcpy bitcast
        const float* src = (const float*)in->contents();
        uint32_t* dst = (uint32_t*)out->contents();
        for (uint32_t i = 0; i < count; ++i) std::memcpy(&dst[i], &src[i], sizeof(uint32_t));
        return out;
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
    return out;
}




// ── GPU arithAddConstU32: out[i] = in[i] + val ──
MTL::Buffer* GpuOps::arithAddConstU32(MTL::Buffer* in, uint32_t val, uint32_t count) {
    auto& store = GpuColumnStore::instance();
    auto p = makePSO(store.device(), store.library(), "ops::arith_add_const_u32");
    if (!p) return nullptr;

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
        if (!isBatchActive()) cmd->waitUntilCompleted();
    }
    return out;
}

// ── GPU nonNullIndicatorF32: out[i] = (in[i] != 0) ? 1.0 : 0.0 ──
MTL::Buffer* GpuOps::nonNullIndicatorF32(MTL::Buffer* in, uint32_t count) {
    auto& store = GpuColumnStore::instance();
    auto p = makePSO(store.device(), store.library(), "ops::nonnull_indicator_f32");
    if (!p) return nullptr;

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
        if (!isBatchActive()) cmd->waitUntilCompleted();
    }
    return out;
}

// ── Shared helper for binary f32 arithmetic GPU kernels ──
// Handles ColCol (2 buffers), ColScalar (buf + scalar), ScalarCol (scalar + buf).
// Avoids ~180 lines of near-identical boilerplate.
enum class ArithBindKind { ColCol, ColScalar, ScalarCol };

static MTL::Buffer* arithF32Dispatch(
    const char* kernelName, ArithBindKind kind,
    MTL::Buffer* a, MTL::Buffer* b, float scalarVal, uint32_t count)
{
    auto& store = GpuColumnStore::instance();
    auto p = makePSO(store.device(), store.library(), kernelName);
    if (!p) return nullptr;

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
        if (!GpuOps::isBatchActive()) cmd->waitUntilCompleted();
    }
    return out;
}

MTL::Buffer* GpuOps::arithMulF32ColCol(MTL::Buffer* colA, MTL::Buffer* colB, uint32_t count) {
    return arithF32Dispatch("ops::arith_mul_f32_col_col", ArithBindKind::ColCol, colA, colB, 0, count);
}
MTL::Buffer* GpuOps::arithMulF32ColScalar(MTL::Buffer* colA, float valB, uint32_t count) {
    return arithF32Dispatch("ops::arith_mul_f32_col_scalar", ArithBindKind::ColScalar, colA, nullptr, valB, count);
}
MTL::Buffer* GpuOps::arithDivF32ColCol(MTL::Buffer* colA, MTL::Buffer* colB, uint32_t count) {
    return arithF32Dispatch("ops::arith_div_f32_col_col", ArithBindKind::ColCol, colA, colB, 0, count);
}
MTL::Buffer* GpuOps::arithDivF32ColScalar(MTL::Buffer* colA, float valB, uint32_t count) {
    return arithF32Dispatch("ops::arith_div_f32_col_scalar", ArithBindKind::ColScalar, colA, nullptr, valB, count);
}
MTL::Buffer* GpuOps::arithDivF32ScalarCol(float valA, MTL::Buffer* colB, uint32_t count) {
    return arithF32Dispatch("ops::arith_div_f32_scalar_col", ArithBindKind::ScalarCol, nullptr, colB, valA, count);
}
MTL::Buffer* GpuOps::arithSubF32ColCol(MTL::Buffer* colA, MTL::Buffer* colB, uint32_t count) {
    return arithF32Dispatch("arith_sub_f32_col_col", ArithBindKind::ColCol, colA, colB, 0, count);
}
MTL::Buffer* GpuOps::arithSubF32ColScalar(MTL::Buffer* colA, float valB, uint32_t count) {
    return arithF32Dispatch("arith_sub_f32_col_scalar", ArithBindKind::ColScalar, colA, nullptr, valB, count);
}
MTL::Buffer* GpuOps::arithSubF32ScalarCol(float valA, MTL::Buffer* colB, uint32_t count) {
    return arithF32Dispatch("arith_sub_f32_scalar_col", ArithBindKind::ScalarCol, nullptr, colB, valA, count);
}

MTL::Buffer* GpuOps::createBuffer(const void* data, size_t size) {
    auto& store = GpuColumnStore::instance();
    if (!store.device()) return nullptr;
    if (data) {
        return store.device()->newBuffer(data, size, MTL::ResourceStorageModeShared);
    } else {
        return store.device()->newBuffer(size, MTL::ResourceStorageModeShared);
    }
}

// Cached 4-byte output buffer for reduce operations (avoids repeated alloc/release)
static MTL::Buffer* getReduceOutBuf() {
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
    }
    
    float res = *(float*)out->contents();
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
    }
    
    float res = *(float*)out->contents();
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
    }
    
    float res = *(float*)out->contents();
    return res;
}

// ── M11: Extract YEAR from u32 date → u32 year ─────────────────────────
MTL::Buffer* GpuOps::extractYearU32(MTL::Buffer* dateCol, uint32_t count) {
    if (count == 0) return nullptr;
    auto& store = GpuColumnStore::instance();
    auto dev = store.device();
    auto lib = store.library();
    if (!dev || !lib || !store.queue()) return nullptr;

    MTL::Buffer* outBuf = dev->newBuffer(count * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    if (!outBuf) return nullptr;

    auto pso = makePSO(dev, lib, "ops::extract_year_u32_to_u32");
    if (!pso) { outBuf->release(); return nullptr; }

    auto cmd = store.queue()->commandBuffer();
    auto enc = cmd->computeCommandEncoder();
    enc->setComputePipelineState(pso);
    enc->setBuffer(dateCol, 0, 0);
    enc->setBuffer(outBuf, 0, 1);
    enc->setBytes(&count, sizeof(count), 2);
    dispatch1D(enc, count);
    enc->endEncoding();
    cmd->commit();
    if (!isBatchActive()) cmd->waitUntilCompleted();

    return outBuf;
}

// ── T7: Flat-string SUBSTRING (zero-copy offset/length adjustment) ──
std::pair<MTL::Buffer*, MTL::Buffer*> GpuOps::substringFlat(
    MTL::Buffer* inOffsets, MTL::Buffer* inLengths,
    uint32_t startPos, uint32_t substrLen, uint32_t rowCount) {
    if (rowCount == 0) return {nullptr, nullptr};
    auto& store = GpuColumnStore::instance();
    auto dev = store.device();
    auto lib = store.library();
    if (!dev || !lib || !store.queue()) return {nullptr, nullptr};

    auto outOffsets = dev->newBuffer(rowCount * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    auto outLengths = dev->newBuffer(rowCount * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    if (!outOffsets || !outLengths) {
        if (outOffsets) outOffsets->release();
        if (outLengths) outLengths->release();
        return {nullptr, nullptr};
    }

    auto pso = makePSO(dev, lib, "ops::substring_flat");
    if (!pso) { outOffsets->release(); outLengths->release(); return {nullptr, nullptr}; }

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

    return {outOffsets, outLengths};
}

// ── T7: Hash-encode flat string to u32 (first 8 chars packed big-endian) ──
MTL::Buffer* GpuOps::stringHashEncodeU32(
    MTL::Buffer* chars, MTL::Buffer* offsets, MTL::Buffer* lengths, uint32_t rowCount) {
    if (rowCount == 0) return nullptr;
    auto& store = GpuColumnStore::instance();
    auto dev = store.device();
    auto lib = store.library();
    if (!dev || !lib || !store.queue()) return nullptr;

    auto outBuf = dev->newBuffer(rowCount * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    if (!outBuf) return nullptr;

    auto pso = makePSO(dev, lib, "ops::string_hash_encode_u32");
    if (!pso) { outBuf->release(); return nullptr; }

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
    cmd->waitUntilCompleted();

    return outBuf;
}

// ── T8: FNV1a-32 hash of flat string columns ──
MTL::Buffer* GpuOps::stringFnv1aU32(
    MTL::Buffer* chars, MTL::Buffer* offsets, MTL::Buffer* lengths, uint32_t rowCount) {
    if (rowCount == 0) return nullptr;
    auto& store = GpuColumnStore::instance();
    auto dev = store.device();
    auto lib = store.library();
    if (!dev || !lib || !store.queue()) return nullptr;

    auto outBuf = dev->newBuffer(rowCount * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    if (!outBuf) return nullptr;

    auto pso = makePSO(dev, lib, "ops::string_fnv1a_u32");
    if (!pso) { outBuf->release(); return nullptr; }

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
    cmd->waitUntilCompleted();

    return outBuf;
}

// ── GPU FlatStringCol gather ─────────────────────────────────────────────
// Gathers chars/offsets/lengths by an index buffer to produce a compacted FlatStringCol.
// Steps: 1) gather lengths via gatherU32, 2) prefix-sum → new offsets,
//        3) gather_flat_string_chars kernel copies chars.
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
    MTL::Buffer* outLengths = gatherU32(srcLengths, indices, count, true);
    if (!outLengths) return r;

    // Step 2: Prefix-sum on lengths → outOffsets (exclusive scan)
    // We need a copy for the scan since scanInPlace modifies in-place.
    MTL::Buffer* outOffsets = dev->newBuffer(count * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    if (!outOffsets) { outLengths->release(); return r; }
    std::memcpy(outOffsets->contents(), outLengths->contents(), count * sizeof(uint32_t));
    uint64_t totalBytes = scanInPlace(outOffsets, count);

    // Step 3: Allocate output chars and dispatch char-copy kernel
    uint32_t totalBytesU32 = static_cast<uint32_t>(totalBytes);
    size_t allocBytes = (totalBytesU32 > 0) ? totalBytesU32 : 1;
    MTL::Buffer* outChars = dev->newBuffer(allocBytes, MTL::ResourceStorageModeShared);
    if (!outChars) { outLengths->release(); outOffsets->release(); return r; }

    if (totalBytesU32 > 0) {
        auto pso = makePSO(dev, store.library(), "ops::gather_flat_string_chars");
        if (!pso) { outChars->release(); outLengths->release(); outOffsets->release(); return r; }

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
        if (doSync) cmd->waitUntilCompleted();
    }

    auto end = std::chrono::high_resolution_clock::now();
    KernelTimer::instance().record("ops::gather_flat_string", "gather",
        std::chrono::duration<double, std::milli>(end - start).count(), count);

    r.chars = outChars;
    r.offsets = outOffsets;
    r.lengths = outLengths;
    r.rowCount = count;
    r.totalBytes = totalBytesU32;
    return r;
}

// ── H4: GPU dedup by sorted keys ────────────────────────────────────────

MTL::Buffer* GpuOps::arithAddF32ColCol(MTL::Buffer* colA, MTL::Buffer* colB, uint32_t count) {
    return arithF32Dispatch("arith_add_f32_col_col", ArithBindKind::ColCol, colA, colB, 0, count);
}
MTL::Buffer* GpuOps::arithAddF32ColScalar(MTL::Buffer* colA, float valB, uint32_t count) {
    return arithF32Dispatch("arith_add_f32_col_scalar", ArithBindKind::ColScalar, colA, nullptr, valB, count);
}

void GpuOps::scatterConstantF32(MTL::Buffer* output, MTL::Buffer* indices, uint32_t indexCount, float val) {
    if (indexCount == 0 || !output || !indices) return;

    auto& store = GpuColumnStore::instance();

    auto p = makePSO(store.device(), store.library(), "ops::scatter_constant_f32");
    if (!p) {
        // Fallback or debug
        std::cerr << "[GPU] function not found: ops::scatter_constant_f32" << std::endl;
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
    cmd->waitUntilCompleted();
}

void GpuOps::scatterF32(MTL::Buffer* input, MTL::Buffer* output, MTL::Buffer* indices, uint32_t count) {
    if (count == 0 || !input || !output || !indices) return;

    auto& store = GpuColumnStore::instance();
    auto p = makePSO(store.device(), store.library(), "ops::scatter_f32_indexed");
    if (!p) {
        std::cerr << "[GPU] function not found: ops::scatter_f32_indexed" << std::endl;
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
    cmd->waitUntilCompleted();
}

MTL::Buffer* GpuOps::mathFloorF32(MTL::Buffer* col, uint32_t count) {
    auto& store = GpuColumnStore::instance();
    auto p = makePSO(store.device(), store.library(), "ops::arith_floor_f32");
    if (!p) return nullptr;

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
        if (!isBatchActive()) cmd->waitUntilCompleted();
    }
    return out;
}

} // namespace engine

