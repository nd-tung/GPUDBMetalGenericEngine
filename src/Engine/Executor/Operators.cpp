#include "Operators.hpp"
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
#include <unordered_map>

namespace engine {

static MTL::ComputePipelineState* makePSO(MTL::Device* dev, MTL::Library* lib, const char* fn) {
    // Cache PSOs for the lifetime of the process to avoid repeated compilation.
    // Returned PSOs are owned by the cache; callers must NOT release them.
    static std::unordered_map<std::string, MTL::ComputePipelineState*> cache;

    auto it = cache.find(fn);
    if (it != cache.end()) return it->second;

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

    cache.emplace(std::string(fn), pso);
    return pso;
}

// Performs in-place exclusive scan on 'data' (u32). 
// Returns the total sum (reduction).
static uint64_t scanInPlace(MTL::Buffer* data, uint32_t count) {
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
        static MTL::Buffer* s_readBuf = nullptr;
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

static void dispatch1D(MTL::ComputeCommandEncoder* enc, uint32_t count) {
    const uint32_t tg = 256;
    MTL::Size grid = MTL::Size::Make(count, 1, 1);
    MTL::Size tgsz = MTL::Size::Make(tg, 1, 1);
    enc->dispatchThreads(grid, tgsz);
}

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

static std::string rawStringCacheKey(const std::string& filePath, int columnIndex) {
    return filePath + ":" + std::to_string(columnIndex);
}

// Load raw string column (for LIKE/CONTAINS pattern matching)
// Checks the cache first; falls back to file read.
static std::vector<std::string> loadStringColumnRawImpl(const std::string& filePath, int columnIndex) {
    // Check cache (populated by single-pass loader)
    std::string cKey = rawStringCacheKey(filePath, columnIndex);
    auto cit = s_rawStringCache.find(cKey);
    if (cit != s_rawStringCache.end()) return cit->second;

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
                if (s_rawStringCache.find(cKey) == s_rawStringCache.end()) {
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

JoinResult GpuOps::joinHash(MTL::Buffer* buildKeys, 
                                     MTL::Buffer* /*buildIndices*/, 
                                     uint32_t buildCount,
                                     MTL::Buffer* probeKeys,
                                     MTL::Buffer* /*probeIndices*/,
                                     uint32_t probeCount) {
    auto& store = GpuColumnStore::instance();
    const bool debug = env_truthy("GPUDB_DEBUG_OPS");
    if (buildCount == 0 || probeCount == 0 || !store.device()) return JoinResult{};

    // Use multi-match join to correctly handle duplicate keys on the build side.
    // The hash table uses linked lists so multiple build rows per key are preserved.
    
    // 1. Setup Hash Table for multi-match
    uint32_t capacity = 1024;
    while (capacity < buildCount * 2) capacity <<= 1;
    
    auto bufHTKeys = store.device()->newBuffer(capacity * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    auto bufHTHead = store.device()->newBuffer(capacity * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    auto bufNext   = store.device()->newBuffer(buildCount * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    
    std::memset(bufHTKeys->contents(), 0, capacity * sizeof(uint32_t)); // 0 = empty sentinel
    std::memset(bufHTHead->contents(), 0, capacity * sizeof(uint32_t)); // 0 = null pointer
    
    // 2. Build Phase — build linked lists per key
    auto p_build = makePSO(store.device(), store.library(), "ops::hash_join_build_multi");
    if (!p_build) {
        bufHTKeys->release(); bufHTHead->release(); bufNext->release();
        return JoinResult{};
    }
    
    // 3. Count Phase — count matches per probe row
    auto bufCounts = store.device()->newBuffer(probeCount * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    auto p_count = makePSO(store.device(), store.library(), "ops::hash_join_probe_count_multi");
    if (!p_count) {
        bufHTKeys->release(); bufHTHead->release(); bufNext->release(); bufCounts->release();
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
    }
    
    // 4. GPU exclusive prefix sum on counts → offsets, returns total
    uint64_t totalPairs64 = scanInPlace(bufCounts, probeCount);
    uint32_t totalPairs = static_cast<uint32_t>(totalPairs64);
    
    if (debug) std::cerr << "[GPU] joinHashMulti: buildCount=" << buildCount 
                         << " probeCount=" << probeCount << " totalPairs=" << totalPairs << "\n";
    
    if (totalPairs == 0) {
        bufHTKeys->release(); bufHTHead->release(); bufNext->release(); bufCounts->release();
        auto emptyBuf = store.device()->newBuffer(sizeof(uint32_t), MTL::ResourceStorageModeShared);
        return {GpuBuffer(emptyBuf), GpuBuffer(store.device()->newBuffer(sizeof(uint32_t), MTL::ResourceStorageModeShared)), 0};
    }
    
    // 5. Write Phase — write matched pairs (bufCounts now holds exclusive prefix sums = offsets)
    MTL::Buffer* bufOffsets = bufCounts;  // reuse in-place — scanInPlace converted counts → offsets
    auto outProbeIndices = store.device()->newBuffer(totalPairs * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    auto outBuildIndices = store.device()->newBuffer(totalPairs * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    
    auto p_write = makePSO(store.device(), store.library(), "ops::hash_join_probe_write_multi");
    if (!p_write) {
        bufHTKeys->release(); bufHTHead->release(); bufNext->release(); bufOffsets->release();
        outProbeIndices->release(); outBuildIndices->release();
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
        auto end = std::chrono::high_resolution_clock::now();
        KernelTimer::instance().record("hash_join_probe_write_multi", "hash_join_probe_u32", 
            std::chrono::duration<double, std::milli>(end - start).count(), probeCount);
    }
    
    bufHTKeys->release();
    bufHTHead->release();
    bufNext->release();
    bufOffsets->release();  // bufCounts was reused as bufOffsets
    
    return {GpuBuffer(outBuildIndices), GpuBuffer(outProbeIndices), totalPairs};
}

JoinResult GpuOps::joinHashU64(MTL::Buffer* buildKeys, 
                                        MTL::Buffer* buildIndices, 
                                        uint32_t buildCount,
                                        MTL::Buffer* probeKeys,
                                        MTL::Buffer* probeIndices,
                                        uint32_t probeCount) {
    auto& store = GpuColumnStore::instance();
    const bool debug = env_truthy("GPUDB_DEBUG_OPS");
    if (debug) std::cerr << "[GPU] joinHashU64: buildCount=" << buildCount << " probeCount=" << probeCount << std::endl << std::flush;
    if (buildCount == 0 || probeCount == 0 || !store.device()) return JoinResult{};

    uint32_t capacity = 1024;
    while (capacity < buildCount * 2) capacity <<= 1;
    if (debug) std::cerr << "[GPU] joinHashU64: hash table capacity=" << capacity << std::endl << std::flush;
    
    // Split hash table: separate buffers for low and high 32 bits of keys
    // This avoids 64-bit atomics which are not well supported on all Metal devices
    auto bufHTKeysLow = store.device()->newBuffer(capacity * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    auto bufHTKeysHigh = store.device()->newBuffer(capacity * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    auto bufHTVals = store.device()->newBuffer(capacity * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    
    // Init Keys to EMPTY (0xFFFFFFFF for both parts = 64-bit EMPTY)
    std::memset(bufHTKeysLow->contents(), 0xFF, capacity * sizeof(uint32_t));
    std::memset(bufHTKeysHigh->contents(), 0xFF, capacity * sizeof(uint32_t));
    
    auto p_build = makePSO(store.device(), store.library(), "ops::join_build_u64");
    if (!p_build) {
        bufHTKeysLow->release(); bufHTKeysHigh->release(); bufHTVals->release();
        return JoinResult{};
    }
    
    if (debug) std::cerr << "[GPU] joinHashU64: starting build phase..." << std::endl << std::flush;
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
        cmd->waitUntilCompleted();
    }
    if (debug) std::cerr << "[GPU] joinHashU64: build phase done." << std::endl << std::flush;
    
    auto outBuildIndices = store.device()->newBuffer(probeCount * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    auto outProbeIndices = store.device()->newBuffer(probeCount * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    auto outCount = store.device()->newBuffer(sizeof(uint32_t), MTL::ResourceStorageModeShared);
    std::memset(outCount->contents(), 0, 4);
    
    auto p_probe = makePSO(store.device(), store.library(), "ops::join_probe_u64");
    if (!p_probe) {
        bufHTKeysLow->release(); bufHTKeysHigh->release(); bufHTVals->release();
        outBuildIndices->release(); outProbeIndices->release(); outCount->release();
        return JoinResult{};
    }

    if (debug) std::cerr << "[GPU] joinHashU64: starting probe phase..." << std::endl << std::flush;
    {
        auto start = std::chrono::high_resolution_clock::now();
        auto cmd = store.queue()->commandBuffer();
        auto enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(p_probe);
        enc->setBuffer(probeKeys, 0, 0);
        enc->setBuffer(probeIndices, 0, 1);
        enc->setBuffer(bufHTKeysLow, 0, 2);   // Low 32 bits
        enc->setBuffer(bufHTVals, 0, 3);
        enc->setBytes(&capacity, 4, 4);
        enc->setBytes(&probeCount, 4, 5);
        enc->setBuffer(outCount, 0, 6);
        enc->setBuffer(outBuildIndices, 0, 7);
        enc->setBuffer(outProbeIndices, 0, 8);
        enc->setBuffer(bufHTKeysHigh, 0, 9);  // High 32 bits
        dispatch1D(enc, probeCount);
        enc->endEncoding();
        cmd->commit();
        cmd->waitUntilCompleted();
        auto end = std::chrono::high_resolution_clock::now();
        KernelTimer::instance().record("ops::join_probe_u64", "hash_join_probe_u64", 
            std::chrono::duration<double, std::milli>(end - start).count(), probeCount);
    }
    if (debug) std::cerr << "[GPU] joinHashU64: probe phase done." << std::endl << std::flush;
    
    uint32_t totalPairs = *reinterpret_cast<uint32_t*>(outCount->contents());
    if (debug) std::cerr << "[GPU] joinHashU64: result count=" << totalPairs << std::endl << std::flush;
    
    bufHTKeysLow->release();
    bufHTKeysHigh->release();
    bufHTVals->release();
    outCount->release();
    
    return {GpuBuffer(outBuildIndices), GpuBuffer(outProbeIndices), totalPairs};
}

MTL::Buffer* GpuOps::packU32ToU64(MTL::Buffer* c1, MTL::Buffer* c2, uint32_t count) {
    auto& store = GpuColumnStore::instance();
    auto p = makePSO(store.device(), store.library(), "ops::pack_u32_to_u64");
    if (!p) return nullptr;
    auto out = store.device()->newBuffer(static_cast<NS::UInteger>(count) * 8, MTL::ResourceStorageModeShared);
    if (!out) return nullptr;
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
    }
    return out;
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

std::optional<FilterResult> GpuOps::filterF32(const std::string& colName,
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
        if (sync) cmd->waitUntilCompleted();
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
        if (sync) cmd->waitUntilCompleted();
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

static uint32_t nextPow2(uint32_t v) {
    if (v == 0) return 1;
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    return v + 1;
}

std::optional<GroupByHashTable> GpuOps::groupByAggMultiKeyTyped(const std::vector<MTL::Buffer*>& keyColsU32,
                                                                         const std::vector<MTL::Buffer*>& aggInputsF32,
                                                                         const std::vector<uint32_t>& aggTypes,
                                                                         uint32_t rowCount) {
    auto& store = GpuColumnStore::instance();
    if (!store.device() || !store.library() || !store.queue()) return std::nullopt;

    auto p = makePSO(store.device(), store.library(), "ops::groupby_agg_multi_key_typed");
    if (!p) return std::nullopt;

    uint32_t numKeys = static_cast<uint32_t>(keyColsU32.size());
    if (numKeys == 0 || numKeys > 8) return std::nullopt;

    const uint32_t numAggs = static_cast<uint32_t>(aggTypes.size());
    if (numAggs == 0 || numAggs > 16) return std::nullopt;
    if (aggInputsF32.size() < numAggs) return std::nullopt;

    uint32_t cap = nextPow2(std::max<uint32_t>(128u, rowCount * 2u));
    // Stride increased from 4 to 8, size is cap * 8 * sizeof(uint32_t)
    auto htKeys = store.device()->newBuffer(static_cast<size_t>(cap) * 8 * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    auto htAggs = store.device()->newBuffer(static_cast<size_t>(cap) * 16 * sizeof(uint32_t), MTL::ResourceStorageModeShared);
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
        std::cerr << "[Ops] groupByAgg: GPU key buf len=" << k0->length()/4 << " agg buf len=" << (ap ? aggInputsF32[0]->length()/4 : 0) << " rowCount=" << rowCount << "\n";
        if (kp && ap) {
            // Print first 10 key-value pairs
            std::cerr << "[Ops] groupByAgg: first 10 GPU key-value pairs: ";
            for (uint32_t i = 0; i < std::min(rowCount, 10u); ++i) {
                std::cerr << "[k=" << kp[i] << " v=" << ap[i] << "] ";
            }
            std::cerr << "\n";
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
            std::cerr << "[Ops] groupByAgg: GPU BUFFER (pre-kernel) max key=" << maxK << " (debiased=" << (maxK-1) << ") val=" << std::fixed << std::setprecision(2) << maxV << "\n";
            std::cerr << "[Ops] groupByAgg: GPU BUFFER unique keys=" << gpuBufSums.size() << "\n";
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

// ── GPU Stream Compaction: extract valid entries from GroupBy hash table ──
// Mark → Prefix Sum → Compact pipeline.
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
    {
        auto cmd = store.queue()->commandBuffer();
        auto enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(p_mark);
        enc->setBuffer(ht.htKeys, 0, 0);
        enc->setBuffer(markBuf, 0, 1);
        enc->setBytes(&cap, sizeof(cap), 2);
        dispatch1D(enc, cap);
        enc->endEncoding();
        auto t0 = std::chrono::high_resolution_clock::now();
        cmd->commit();
        cmd->waitUntilCompleted();
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        KernelTimer::instance().record("ops::ht_mark_valid", "groupby", ms, cap);
    }

    // Step 2 (Prefix Sum): Copy mark → offsets, run exclusive prefix sum.
    auto offsetsBuf = store.device()->newBuffer(
        static_cast<size_t>(cap) * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    std::memcpy(offsetsBuf->contents(), markBuf->contents(),
                static_cast<size_t>(cap) * sizeof(uint32_t));

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
        result.aggWords[a].resize(totalCount);
        std::memcpy(result.aggWords[a].data(), aggPtr + a * totalCount, totalCount * sizeof(uint32_t));
        // Create per-column GPU buffer from SoA slice
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
    }
    
    mask->release();

    FilterResult res;
    res.indices.reset(outIdx);
    res.count = *reinterpret_cast<uint32_t*>(outCnt->contents());
    outCnt->release();
    return res;
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
    
    uint32_t totalCount = leftCount * rightCount;
    dispatch1D(enc, totalCount);
    enc->endEncoding();
    cmd->commit();
    cmd->waitUntilCompleted();
}

std::optional<FilterResult> GpuOps::hashJoinSemiU32(MTL::Buffer* leftKey,
                                                             uint32_t leftCount,
                                                             MTL::Buffer* rightKey,
                                                             uint32_t rightCount) {
    auto& store = GpuColumnStore::instance();
    if (!store.device() || !store.library()) return std::nullopt;

    auto p_build = makePSO(store.device(), store.library(), "ops::hash_join_build_multi");
    auto p_probe = makePSO(store.device(), store.library(), "ops::hash_join_probe_semi");
    auto p_compact = makePSO(store.device(), store.library(), "ops::compact_indices");
    if (!p_build || !p_probe || !p_compact) return std::nullopt;

    uint32_t cap = nextPow2(std::max<uint32_t>(8u, rightCount * 2u));
    auto htKeys = store.device()->newBuffer(cap * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    auto ht_head = store.device()->newBuffer(cap * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    auto next = store.device()->newBuffer(static_cast<size_t>(rightCount) * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    
    std::memset(htKeys->contents(), 0, cap * sizeof(uint32_t));
    std::memset(ht_head->contents(), 0, cap * sizeof(uint32_t));
    if (rightCount > 0) std::memset(next->contents(), 0, static_cast<size_t>(rightCount) * sizeof(uint32_t));

    // Fused: BUILD → PROBE → COMPACT in one command buffer (3 encoders)
    auto mask = store.device()->newBuffer(leftCount * sizeof(uint8_t), MTL::ResourceStorageModeShared);
    auto outIdx = store.device()->newBuffer(leftCount * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    auto outCnt = store.device()->newBuffer(sizeof(uint32_t), MTL::ResourceStorageModeShared);
    *(uint32_t*)outCnt->contents() = 0;

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
        // Encoder 3: COMPACT
        auto enc3 = cmd->computeCommandEncoder();
        enc3->setComputePipelineState(p_compact);
        enc3->setBuffer(mask, 0, 0);
        enc3->setBuffer(outIdx, 0, 1);
        enc3->setBuffer(outCnt, 0, 2);
        enc3->setBytes(&leftCount, sizeof(leftCount), 3);
        dispatch1D(enc3, leftCount);
        enc3->endEncoding();
        cmd->commit();
        cmd->waitUntilCompleted();
    }
    ht_head->release();
    next->release();
    htKeys->release();
    mask->release();
    
    uint32_t validCount = *(uint32_t*)outCnt->contents();
    outCnt->release();
    
    FilterResult res;
    res.indices.reset(outIdx);
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
        res.indices.reset(idx);
        res.count = leftCount;
        return res;
    }

    auto p_build   = makePSO(store.device(), store.library(), "ops::hash_join_build_multi");
    auto p_probe   = makePSO(store.device(), store.library(), "ops::hash_join_probe_semi");
    auto p_flip    = makePSO(store.device(), store.library(), "ops::flip_mask_u8");
    auto p_compact = makePSO(store.device(), store.library(), "ops::compact_indices");
    if (!p_build || !p_probe || !p_flip || !p_compact) return std::nullopt;

    uint32_t cap = nextPow2(std::max<uint32_t>(8u, rightCount * 2u));
    auto htKeys = store.device()->newBuffer(cap * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    auto ht_head = store.device()->newBuffer(cap * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    auto next    = store.device()->newBuffer(static_cast<size_t>(rightCount) * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    
    std::memset(htKeys->contents(), 0, cap * sizeof(uint32_t));
    std::memset(ht_head->contents(), 0, cap * sizeof(uint32_t));
    if (rightCount > 0) std::memset(next->contents(), 0, static_cast<size_t>(rightCount) * sizeof(uint32_t));

    // Fused: BUILD → PROBE → FLIP → COMPACT in one command buffer (4 encoders)
    auto mask = store.device()->newBuffer(leftCount * sizeof(uint8_t), MTL::ResourceStorageModeShared);
    auto outIdx = store.device()->newBuffer(leftCount * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    auto outCnt = store.device()->newBuffer(sizeof(uint32_t), MTL::ResourceStorageModeShared);
    *(uint32_t*)outCnt->contents() = 0;

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
        // Encoder 4: COMPACT
        auto enc4 = cmd->computeCommandEncoder();
        enc4->setComputePipelineState(p_compact);
        enc4->setBuffer(mask, 0, 0);
        enc4->setBuffer(outIdx, 0, 1);
        enc4->setBuffer(outCnt, 0, 2);
        enc4->setBytes(&leftCount, sizeof(leftCount), 3);
        dispatch1D(enc4, leftCount);
        enc4->endEncoding();
        cmd->commit();
        cmd->waitUntilCompleted();
    }
    ht_head->release();
    next->release();
    htKeys->release();
    mask->release();

    uint32_t validCount = *(uint32_t*)outCnt->contents();
    outCnt->release();

    FilterResult res;
    res.indices.reset(outIdx);
    res.count = validCount;
    return res;
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

MTL::Buffer* GpuOps::floatToSortKeyU32(MTL::Buffer* in, uint32_t count, bool desc) {
    auto& store = GpuColumnStore::instance();
    auto out = store.device()->newBuffer(count * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    auto p = makePSO(store.device(), store.library(), "ops::float_to_sort_key_u32");
    if (!p) {
        // CPU fallback
        const float* src = (const float*)in->contents();
        uint32_t* dst = (uint32_t*)out->contents();
        for (uint32_t i = 0; i < count; ++i) {
            uint32_t bits;
            std::memcpy(&bits, &src[i], sizeof(bits));
            if (bits & 0x80000000u) bits = ~bits;
            else bits ^= 0x80000000u;
            dst[i] = desc ? ~bits : bits;
        }
        return out;
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
    return out;
}

MTL::Buffer* GpuOps::invertU32(MTL::Buffer* in, uint32_t count) {
    auto& store = GpuColumnStore::instance();
    auto out = store.device()->newBuffer(count * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    auto p = makePSO(store.device(), store.library(), "ops::invert_u32");
    if (!p) {
        const uint32_t* src = (const uint32_t*)in->contents();
        uint32_t* dst = (uint32_t*)out->contents();
        for (uint32_t i = 0; i < count; ++i) dst[i] = ~src[i];
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

void GpuOps::flipMaskU8(MTL::Buffer* mask, uint32_t count) {
    auto& store = GpuColumnStore::instance();
    auto p = makePSO(store.device(), store.library(), "ops::flip_mask_u8");
    if (!p) {
        // CPU fallback
        uint8_t* ptr = (uint8_t*)mask->contents();
        for (uint32_t i = 0; i < count; ++i) ptr[i] = ptr[i] ? 0 : 1;
        return;
    }
    auto cmd = store.queue()->commandBuffer();
    auto enc = cmd->computeCommandEncoder();
    enc->setComputePipelineState(p);
    enc->setBuffer(mask, 0, 0);
    enc->setBytes(&count, sizeof(count), 1);
    dispatch1D(enc, count);
    enc->endEncoding();
    cmd->commit();
    cmd->waitUntilCompleted();
}

FilterResult GpuOps::findUnmatchedIndices(MTL::Buffer* matchedIndices,
                                                    uint32_t matchedCount,
                                                    uint32_t totalRows) {
    auto& store = GpuColumnStore::instance();

    // Edge case: no matches → every row is unmatched
    if (matchedCount == 0) {
        FilterResult res;
        res.indices.reset(iotaU32(totalRows));
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
    auto p_compact = makePSO(store.device(), store.library(), "ops::compact_indices");

    // Create u8 mask, zero-initialized
    auto mask = store.device()->newBuffer(totalRows * sizeof(uint8_t), MTL::ResourceStorageModeShared);
    std::memset(mask->contents(), 0, totalRows * sizeof(uint8_t));

    if (p_scatter && p_flip && p_compact) {
        // Fused: SCATTER → FLIP → COMPACT in one command buffer (3 encoders)
        auto outIdx = store.device()->newBuffer(totalRows * sizeof(uint32_t), MTL::ResourceStorageModeShared);
        auto outCnt = store.device()->newBuffer(sizeof(uint32_t), MTL::ResourceStorageModeShared);
        *(uint32_t*)outCnt->contents() = 0;
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
            // Encoder 3: compact
            auto enc3 = cmd->computeCommandEncoder();
            enc3->setComputePipelineState(p_compact);
            enc3->setBuffer(mask, 0, 0);
            enc3->setBuffer(outIdx, 0, 1);
            enc3->setBuffer(outCnt, 0, 2);
            enc3->setBytes(&totalRows, sizeof(totalRows), 3);
            dispatch1D(enc3, totalRows);
            enc3->endEncoding();
            cmd->commit();
            cmd->waitUntilCompleted();
        }
        mask->release();
        uint32_t cnt = *(uint32_t*)outCnt->contents();
        outCnt->release();
        FilterResult res;
        res.indices.reset(outIdx);
        res.count = cnt;
        return res;
    }

    // CPU fallback
    uint8_t* maskPtr = (uint8_t*)mask->contents();
    uint32_t* matchPtr = (uint32_t*)matchedIndices->contents();
    for (uint32_t i = 0; i < matchedCount; ++i) {
        if (matchPtr[i] < totalRows) maskPtr[matchPtr[i]] = 1;
    }
    std::vector<uint32_t> result;
    for (uint32_t i = 0; i < totalRows; ++i) {
        if (!maskPtr[i]) result.push_back(i);
    }
    mask->release();
    uint32_t cnt = (uint32_t)result.size();
    auto outIdx = store.device()->newBuffer(
        result.empty() ? sizeof(uint32_t) : result.size() * sizeof(uint32_t),
        MTL::ResourceStorageModeShared);
    if (!result.empty()) std::memcpy(outIdx->contents(), result.data(), result.size() * sizeof(uint32_t));
    FilterResult fRes;
    fRes.indices.reset(outIdx);
    fRes.count = cnt;
    return fRes;
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

MTL::Buffer* GpuOps::arithMulF32ColCol(MTL::Buffer* colA, MTL::Buffer* colB, uint32_t count) {
    auto& store = GpuColumnStore::instance();
    auto p = makePSO(store.device(), store.library(), "ops::arith_mul_f32_col_col");
    if (!p) return nullptr;

    auto out = store.device()->newBuffer(static_cast<size_t>(count) * sizeof(float), MTL::ResourceStorageModeShared);
    {
        auto cmd = store.queue()->commandBuffer();
        auto enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(p);
        enc->setBuffer(colA, 0, 0);
        enc->setBuffer(colB, 0, 1);
        enc->setBuffer(out, 0, 2);
        enc->setBytes(&count, sizeof(count), 3);
        dispatch1D(enc, count);
        enc->endEncoding();
        cmd->commit();
        if (!isBatchActive()) cmd->waitUntilCompleted();
    }
    return out;
}

MTL::Buffer* GpuOps::arithMulF32ColScalar(MTL::Buffer* colA, float valB, uint32_t count) {
    auto& store = GpuColumnStore::instance();
    auto p = makePSO(store.device(), store.library(), "ops::arith_mul_f32_col_scalar");
    if (!p) return nullptr;

    auto out = store.device()->newBuffer(static_cast<size_t>(count) * sizeof(float), MTL::ResourceStorageModeShared);
    {
        auto cmd = store.queue()->commandBuffer();
        auto enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(p);
        enc->setBuffer(colA, 0, 0);
        enc->setBytes(&valB, sizeof(valB), 1);
        enc->setBuffer(out, 0, 2);
        enc->setBytes(&count, sizeof(count), 3);
        dispatch1D(enc, count);
        enc->endEncoding();
        cmd->commit();
        if (!isBatchActive()) cmd->waitUntilCompleted();
    }
    return out;
}

MTL::Buffer* GpuOps::arithDivF32ColCol(MTL::Buffer* colA, MTL::Buffer* colB, uint32_t count) {
    auto& store = GpuColumnStore::instance();
    auto p = makePSO(store.device(), store.library(), "ops::arith_div_f32_col_col");
    if (!p) return nullptr;

    auto out = store.device()->newBuffer(static_cast<size_t>(count) * sizeof(float), MTL::ResourceStorageModeShared);
    {
        auto cmd = store.queue()->commandBuffer();
        auto enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(p);
        enc->setBuffer(colA, 0, 0);
        enc->setBuffer(colB, 0, 1);
        enc->setBuffer(out, 0, 2);
        enc->setBytes(&count, sizeof(count), 3);
        dispatch1D(enc, count);
        enc->endEncoding();
        cmd->commit();
        if (!isBatchActive()) cmd->waitUntilCompleted();
    }
    return out;
}

MTL::Buffer* GpuOps::arithDivF32ColScalar(MTL::Buffer* colA, float valB, uint32_t count) {
    auto& store = GpuColumnStore::instance();
    auto p = makePSO(store.device(), store.library(), "ops::arith_div_f32_col_scalar");
    if (!p) return nullptr;

    auto out = store.device()->newBuffer(static_cast<size_t>(count) * sizeof(float), MTL::ResourceStorageModeShared);
    {
        auto cmd = store.queue()->commandBuffer();
        auto enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(p);
        enc->setBuffer(colA, 0, 0);
        enc->setBytes(&valB, sizeof(valB), 1);
        enc->setBuffer(out, 0, 2);
        enc->setBytes(&count, sizeof(count), 3);
        dispatch1D(enc, count);
        enc->endEncoding();
        cmd->commit();
        if (!isBatchActive()) cmd->waitUntilCompleted();
    }
    return out;
}

MTL::Buffer* GpuOps::arithDivF32ScalarCol(float valA, MTL::Buffer* colB, uint32_t count) {
    auto& store = GpuColumnStore::instance();
    auto p = makePSO(store.device(), store.library(), "ops::arith_div_f32_scalar_col");
    if (!p) return nullptr;

    auto out = store.device()->newBuffer(static_cast<size_t>(count) * sizeof(float), MTL::ResourceStorageModeShared);
    {
        auto cmd = store.queue()->commandBuffer();
        auto enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(p);
        enc->setBytes(&valA, sizeof(valA), 0);
        enc->setBuffer(colB, 0, 1);
        enc->setBuffer(out, 0, 2);
        enc->setBytes(&count, sizeof(count), 3);
        dispatch1D(enc, count);
        enc->endEncoding();
        cmd->commit();
        if (!isBatchActive()) cmd->waitUntilCompleted();
    }
    return out;
}

MTL::Buffer* GpuOps::arithSubF32ColCol(MTL::Buffer* colA, MTL::Buffer* colB, uint32_t count) {
    auto& store = GpuColumnStore::instance();
    auto p = makePSO(store.device(), store.library(), "arith_sub_f32_col_col");
    if (!p) return nullptr;

    auto out = store.device()->newBuffer(static_cast<size_t>(count) * sizeof(float), MTL::ResourceStorageModeShared);
    {
        auto cmd = store.queue()->commandBuffer();
        auto enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(p);
        enc->setBuffer(colA, 0, 0);
        enc->setBuffer(colB, 0, 1);
        enc->setBuffer(out, 0, 2);
        enc->setBytes(&count, sizeof(count), 3);
        dispatch1D(enc, count);
        enc->endEncoding();
        cmd->commit();
        if (!isBatchActive()) cmd->waitUntilCompleted();
    }
    return out;
}

MTL::Buffer* GpuOps::arithSubF32ColScalar(MTL::Buffer* colA, float valB, uint32_t count) {
    auto& store = GpuColumnStore::instance();
    auto p = makePSO(store.device(), store.library(), "arith_sub_f32_col_scalar");
    if (!p) return nullptr;

    auto out = store.device()->newBuffer(static_cast<size_t>(count) * sizeof(float), MTL::ResourceStorageModeShared);
    {
        auto cmd = store.queue()->commandBuffer();
        auto enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(p);
        enc->setBuffer(colA, 0, 0);
        enc->setBytes(&valB, sizeof(valB), 1);
        enc->setBuffer(out, 0, 2);
        enc->setBytes(&count, sizeof(count), 3);
        dispatch1D(enc, count);
        enc->endEncoding();
        cmd->commit();
        if (!isBatchActive()) cmd->waitUntilCompleted();
    }
    return out;
}

MTL::Buffer* GpuOps::arithSubF32ScalarCol(float valA, MTL::Buffer* colB, uint32_t count) {
    auto& store = GpuColumnStore::instance();
    auto p = makePSO(store.device(), store.library(), "arith_sub_f32_scalar_col");
    if (!p) return nullptr;

    auto out = store.device()->newBuffer(static_cast<size_t>(count) * sizeof(float), MTL::ResourceStorageModeShared);
    {
        auto cmd = store.queue()->commandBuffer();
        auto enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(p);
        enc->setBytes(&valA, sizeof(valA), 0);
        enc->setBuffer(colB, 0, 1);
        enc->setBuffer(out, 0, 2);
        enc->setBytes(&count, sizeof(count), 3);
        dispatch1D(enc, count);
        enc->endEncoding();
        cmd->commit();
        if (!isBatchActive()) cmd->waitUntilCompleted();
    }
    return out;
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

// ── T8b: 8-byte prefix extraction for sort-compatible string keys ──
MTL::Buffer* GpuOps::stringPrefixU64(
    MTL::Buffer* chars, MTL::Buffer* offsets, MTL::Buffer* lengths, uint32_t rowCount) {
    if (rowCount == 0) return nullptr;
    auto& store = GpuColumnStore::instance();
    auto dev = store.device();
    auto lib = store.library();
    if (!dev || !lib || !store.queue()) return nullptr;

    auto outBuf = dev->newBuffer(rowCount * sizeof(uint64_t), MTL::ResourceStorageModeShared);
    if (!outBuf) return nullptr;

    auto pso = makePSO(dev, lib, "ops::string_prefix_u64");
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
MTL::Buffer* GpuOps::dedupByKeys(const std::vector<MTL::Buffer*>& keys, uint32_t count,
                                  uint32_t& uniqueCount) {
    uniqueCount = 0;
    if (count == 0 || keys.empty()) return nullptr;
    auto& store = GpuColumnStore::instance();
    auto dev = store.device();
    auto lib = store.library();
    if (!dev || !lib || !store.queue()) return nullptr;

    // Build sort key: single u32 or packed u64
    bool useU64 = (keys.size() >= 2);
    MTL::Buffer* sortKeys = nullptr;
    bool ownSortKeys = false;

    if (keys.size() == 1) {
        // Copy key to avoid mutating original
        sortKeys = dev->newBuffer(count * sizeof(uint32_t), MTL::ResourceStorageModeShared);
        std::memcpy(sortKeys->contents(), keys[0]->contents(), count * sizeof(uint32_t));
        ownSortKeys = true;
    } else if (keys.size() == 2) {
        sortKeys = packU32ToU64(keys[0], keys[1], count);
        ownSortKeys = true;
    } else {
        // 3+ keys: pack first two into u64, then GPU-fold remaining via hash
        sortKeys = packU32ToU64(keys[0], keys[1], count);
        ownSortKeys = true;
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
            } else {
                // CPU fallback if kernel not found
                auto* ptr = (uint64_t*)sortKeys->contents();
                auto* kp = (const uint32_t*)keys[k]->contents();
                for (uint32_t i = 0; i < count; ++i) {
                    ptr[i] = ptr[i] * 0x9E3779B97F4A7C15ULL + kp[i];
                }
            }
        }
    }

    // GPU iota index array [0, 1, 2, ...]
    MTL::Buffer* indices = iotaU32(count);

    // Radix sort
    if (useU64) {
        radixSortU64(sortKeys, indices, count);
    } else {
        radixSortU32(sortKeys, indices, count);
    }

    // Mark unique positions after sort
    MTL::Buffer* mask = dev->newBuffer(count * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    const char* kernelName = useU64 ? "ops::mark_unique_sorted_u64" : "ops::mark_unique_sorted_u32";
    auto pso = makePSO(dev, lib, kernelName);
    if (!pso) {
        if (ownSortKeys) sortKeys->release();
        indices->release();
        mask->release();
        return nullptr;
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
    }

    if (ownSortKeys) { sortKeys->release(); sortKeys = nullptr; }

    // Compact mask → positions where mask[i]==1
    auto [maskIdx, uCount] = compactU32Mask(mask, count);
    mask->release();

    if (!maskIdx || uCount == 0) {
        indices->release();
        if (maskIdx) maskIdx->release();
        return nullptr;
    }

    if (uCount == count) {
        // All unique — no dedup needed
        indices->release();
        maskIdx->release();
        uniqueCount = count;
        return nullptr;
    }

    // Gather original indices at unique positions
    MTL::Buffer* uniqueIdx = gatherU32(indices, maskIdx, uCount);

    indices->release();
    maskIdx->release();

    uniqueCount = uCount;
    return uniqueIdx;
}

MTL::Buffer* GpuOps::arithAddF32ColCol(MTL::Buffer* colA, MTL::Buffer* colB, uint32_t count) {
    auto& store = GpuColumnStore::instance();
    auto p = makePSO(store.device(), store.library(), "arith_add_f32_col_col");
    if (!p) return nullptr;

    auto out = store.device()->newBuffer(static_cast<size_t>(count) * sizeof(float), MTL::ResourceStorageModeShared);
    {
        auto cmd = store.queue()->commandBuffer();
        auto enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(p);
        enc->setBuffer(colA, 0, 0);
        enc->setBuffer(colB, 0, 1);
        enc->setBuffer(out, 0, 2);
        enc->setBytes(&count, sizeof(count), 3);
        dispatch1D(enc, count);
        enc->endEncoding();
        cmd->commit();
        if (!isBatchActive()) cmd->waitUntilCompleted();
    }
    return out;
}

MTL::Buffer* GpuOps::arithAddF32ColScalar(MTL::Buffer* colA, float valB, uint32_t count) {
    auto& store = GpuColumnStore::instance();
    auto p = makePSO(store.device(), store.library(), "arith_add_f32_col_scalar");
    if (!p) return nullptr;

    auto out = store.device()->newBuffer(static_cast<size_t>(count) * sizeof(float), MTL::ResourceStorageModeShared);
    {
        auto cmd = store.queue()->commandBuffer();
        auto enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(p);
        enc->setBuffer(colA, 0, 0);
        enc->setBytes(&valB, sizeof(valB), 1);
        enc->setBuffer(out, 0, 2);
        enc->setBytes(&count, sizeof(count), 3);
        dispatch1D(enc, count);
        enc->endEncoding();
        cmd->commit();
        if (!isBatchActive()) cmd->waitUntilCompleted();
    }
    return out;
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

// ============================================================================
// GPU Radix Sort (stable, 8-bit radix)
// ============================================================================
// For ≤1024 elements: single-dispatch block sort (shared-memory bitonic).
// For >1024 elements: multi-pass LSD radix sort (histogram → scan → scatter).

static void blockSortU32(MTL::Buffer* keys, MTL::Buffer* indices, uint32_t count) {
    auto& store = GpuColumnStore::instance();
    auto p = makePSO(store.device(), store.library(), "ops::block_sort_kv_u32");

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
}

static void blockSortU64(MTL::Buffer* keys, MTL::Buffer* indices, uint32_t count) {
    auto& store = GpuColumnStore::instance();
    auto p = makePSO(store.device(), store.library(), "ops::block_sort_kv_u64");

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
}

void GpuOps::radixSortU32(MTL::Buffer* keys, MTL::Buffer* indices, uint32_t count) {
    if (count <= 1) return;

    if (count <= 1024) {
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
            cmd->waitUntilCompleted();
        }

        // Prefix sum
        scanInPlace(histBuf, histSize);

        // Scatter
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
            cmd->waitUntilCompleted();
        }

        std::swap(srcK, dstK);
        std::swap(srcV, dstV);
    }
    // After 4 passes (even), result is back in original (keys, indices) buffers.

    keysAlt->release();
    valsAlt->release();
    histBuf->release();

    KernelTimer::instance().record("radix_sort_u32", "sort", 0, count);
}

void GpuOps::radixSortU64(MTL::Buffer* keys, MTL::Buffer* indices, uint32_t count) {
    if (count <= 1) return;

    if (count <= 1024) {
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
            cmd->waitUntilCompleted();
        }

        scanInPlace(histBuf, histSize);

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
            cmd->waitUntilCompleted();
        }

        std::swap(srcK, dstK);
        std::swap(srcV, dstV);
    }
    // After 8 passes (even), result is back in original (keys, indices) buffers.

    keysAlt->release();
    valsAlt->release();
    histBuf->release();

    KernelTimer::instance().record("radix_sort_u64", "sort", 0, count);
}

} // namespace engine
