#include "Operators.hpp"

#include "ColumnStoreGPU.hpp"
#include "KernelTimer.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
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
    auto& store = ColumnStoreGPU::instance();
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
        auto readBuf = store.device()->newBuffer(sizeof(uint32_t), MTL::ResourceStorageModeShared);
        auto cmd = store.queue()->commandBuffer();
        auto blit = cmd->blitCommandEncoder();
        blit->copyFromBuffer(partials, 0, readBuf, 0, sizeof(uint32_t));
        blit->endEncoding();
        cmd->commit();
        cmd->waitUntilCompleted();
        uint32_t val;
        std::memcpy(&val, readBuf->contents(), sizeof(uint32_t));
        totalSum = val;
        readBuf->release();
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

static std::vector<uint32_t> loadU32Column(const std::string& filePath, int columnIndex) {
    std::vector<uint32_t> data;
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
                try { data.push_back(static_cast<uint32_t>(std::stoul(token))); }
                catch (...) { data.push_back(0); }
                break;
            }
            s = e + 1;
            e = line.find('|', s);
            ++col;
        }
    }
    return data;
}

static std::vector<uint32_t> loadDateAsU32YYYYMMDD(const std::string& filePath, int columnIndex) {
    std::vector<uint32_t> data;
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
                token.erase(std::remove(token.begin(), token.end(), '-'), token.end());
                token.erase(0, token.find_first_not_of(" \t\n\r"));
                token.erase(token.find_last_not_of(" \t\n\r") + 1);
                try { data.push_back(static_cast<uint32_t>(std::stoul(token))); }
                catch (...) { data.push_back(0); }
                break;
            }
            s = e + 1;
            e = line.find('|', s);
            ++col;
        }
    }
    return data;
}

static std::vector<float> loadF32Column(const std::string& filePath, int columnIndex) {
    std::vector<float> data;
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
                try { data.push_back(std::stof(token)); }
                catch (...) { data.push_back(0.0f); }
                break;
            }
            s = e + 1;
            e = line.find('|', s);
            ++col;
        }
    }
    return data;
}

static std::vector<uint32_t> loadStringHashU32(const std::string& filePath, int columnIndex) {
    std::vector<uint32_t> data;
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
                data.push_back(GpuOps::fnv1a32(token));
                break;
            }
            s = e + 1;
            e = line.find('|', s);
            ++col;
        }
    }
    return data;
}

// Load single-character string column as char code (reversible)
static std::vector<uint32_t> loadStringCharU32(const std::string& filePath, int columnIndex) {
    std::vector<uint32_t> data;
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
                // Store first character code (or 0 if empty)
                uint32_t val = token.empty() ? 0 : static_cast<uint32_t>(static_cast<unsigned char>(token[0]));
                data.push_back(val);
                break;
            }
            s = e + 1;
            e = line.find('|', s);
            ++col;
        }
    }
    return data;
}

// Load raw string column (for LIKE/CONTAINS pattern matching)
static std::vector<std::string> loadStringColumnRawImpl(const std::string& filePath, int columnIndex) {
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

// --- Schema ---

struct ColMeta {
    int idx;
    enum class Kind { U32, F32, DateU32, StrHashU32, StrCharU32 } kind;  // StrCharU32 = single-char reversible
};

static const std::map<std::string, std::map<std::string, ColMeta>> kSchema = {
    {"customer", {
        {"c_custkey", {0, ColMeta::Kind::U32}},
        {"c_name", {1, ColMeta::Kind::StrHashU32}},
        {"c_address", {2, ColMeta::Kind::StrHashU32}},
        {"c_nationkey", {3, ColMeta::Kind::U32}},
        {"c_phone", {4, ColMeta::Kind::StrHashU32}},
        {"c_acctbal", {5, ColMeta::Kind::F32}},
        {"c_mktsegment", {6, ColMeta::Kind::StrHashU32}},
        {"c_comment", {7, ColMeta::Kind::StrHashU32}},
    }},
    {"orders", {
        {"o_orderkey", {0, ColMeta::Kind::U32}},
        {"o_custkey", {1, ColMeta::Kind::U32}},
        {"o_orderstatus", {2, ColMeta::Kind::StrCharU32}},  // Single char: F/O/P
        {"o_totalprice", {3, ColMeta::Kind::F32}},
        {"o_orderdate", {4, ColMeta::Kind::DateU32}},
        {"o_orderpriority", {5, ColMeta::Kind::StrHashU32}},
        {"o_clerk", {6, ColMeta::Kind::StrHashU32}},
        {"o_shippriority", {7, ColMeta::Kind::U32}},
        {"o_comment", {8, ColMeta::Kind::StrHashU32}},
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
        {"l_shipinstruct", {13, ColMeta::Kind::StrHashU32}},
        {"l_shipmode", {14, ColMeta::Kind::StrHashU32}},
        {"l_comment", {15, ColMeta::Kind::StrHashU32}},
    }},
    {"supplier", {
        {"s_suppkey", {0, ColMeta::Kind::U32}},
        {"s_name", {1, ColMeta::Kind::StrHashU32}},
        {"s_address", {2, ColMeta::Kind::StrHashU32}},
        {"s_nationkey", {3, ColMeta::Kind::U32}},
        {"s_phone", {4, ColMeta::Kind::StrHashU32}},
        {"s_acctbal", {5, ColMeta::Kind::F32}},
        {"s_comment", {6, ColMeta::Kind::StrHashU32}},
    }},
    {"part", {
        {"p_partkey", {0, ColMeta::Kind::U32}},
        {"p_name", {1, ColMeta::Kind::StrHashU32}},
        {"p_mfgr", {2, ColMeta::Kind::StrHashU32}},
        {"p_brand", {3, ColMeta::Kind::StrHashU32}},
        {"p_type", {4, ColMeta::Kind::StrHashU32}},
        {"p_size", {5, ColMeta::Kind::U32}},
        {"p_container", {6, ColMeta::Kind::StrHashU32}},
        {"p_retailprice", {7, ColMeta::Kind::F32}},
        {"p_comment", {8, ColMeta::Kind::StrHashU32}},
    }},
    {"partsupp", {
        {"ps_partkey", {0, ColMeta::Kind::U32}},
        {"ps_suppkey", {1, ColMeta::Kind::U32}},
        {"ps_availqty", {2, ColMeta::Kind::U32}},
        {"ps_supplycost", {3, ColMeta::Kind::F32}},
        {"ps_comment", {4, ColMeta::Kind::StrHashU32}},
    }},
    {"nation", {
        {"n_nationkey", {0, ColMeta::Kind::U32}},
        {"n_name", {1, ColMeta::Kind::StrHashU32}},
        {"n_regionkey", {2, ColMeta::Kind::U32}},
        {"n_comment", {3, ColMeta::Kind::StrHashU32}},
    }},
    {"region", {
        {"r_regionkey", {0, ColMeta::Kind::U32}},
        {"r_name", {1, ColMeta::Kind::StrHashU32}},
        {"r_comment", {2, ColMeta::Kind::StrHashU32}},
    }},
};

RelationGPU GpuOps::scanTable(const std::string& dataset_path,
                                   const std::string& table,
                                   const std::vector<std::string>& neededCols) {
    RelationGPU rel;

    auto& store = ColumnStoreGPU::instance();
    store.initialize();
    if (!store.device()) return rel;

    const auto itT = kSchema.find(table);
    if (itT == kSchema.end()) return rel;

    std::string path = dataset_path + table + ".tbl";

    auto cache_key = [&](const std::string& colName) {
        // Ensure SF-1 vs SF-10 (and other datasets) don't collide.
        return dataset_path + table + "." + colName;
    };

    bool sizeSet = false;
    uint32_t rowCount = 0;

    for (const auto& c : neededCols) {
        const auto itC = itT->second.find(c);
        if (itC == itT->second.end()) continue;

        const ColMeta meta = itC->second;
        if (meta.kind == ColMeta::Kind::F32) {
            const std::string key = cache_key(c);
            GPUColumn* staged = store.getColumn(key);
            if (!staged) {
                auto host = loadF32Column(path, meta.idx);
                if (host.empty()) continue;
                staged = store.stageFloatColumn(key, host);
            }
            if (!staged || !staged->buffer) continue;
            if (!sizeSet) { rowCount = static_cast<uint32_t>(staged->count); sizeSet = true; }
            if (static_cast<uint32_t>(staged->count) != rowCount) continue;
            staged->buffer->retain();
            rel.f32cols[c] = staged->buffer;
        } else if (meta.kind == ColMeta::Kind::U32) {
            const std::string key = cache_key(c);
            GPUColumn* staged = store.getColumn(key);
            if (!staged) {
                auto host = loadU32Column(path, meta.idx);
                if (host.empty()) continue;
                staged = store.stageU32Column(key, host);
            }
            if (!staged || !staged->buffer) continue;
            if (!sizeSet) { rowCount = static_cast<uint32_t>(staged->count); sizeSet = true; }
            if (static_cast<uint32_t>(staged->count) != rowCount) continue;
            staged->buffer->retain();
            rel.u32cols[c] = staged->buffer;
        } else if (meta.kind == ColMeta::Kind::DateU32) {
            const std::string key = cache_key(c);
            GPUColumn* staged = store.getColumn(key);
            if (!staged) {
                auto host = loadDateAsU32YYYYMMDD(path, meta.idx);
                if (host.empty()) continue;
                staged = store.stageU32Column(key, host);
            }
            if (!staged || !staged->buffer) continue;
            if (!sizeSet) { rowCount = static_cast<uint32_t>(staged->count); sizeSet = true; }
            if (static_cast<uint32_t>(staged->count) != rowCount) continue;
            staged->buffer->retain();
            rel.u32cols[c] = staged->buffer;
        } else if (meta.kind == ColMeta::Kind::StrHashU32) {
            const std::string key = cache_key(c);
            GPUColumn* staged = store.getColumn(key);
            if (!staged) {
                auto host = loadStringHashU32(path, meta.idx);
                if (host.empty()) continue;
                staged = store.stageU32Column(key, host);
            }
            if (!staged || !staged->buffer) continue;
            if (!sizeSet) { rowCount = static_cast<uint32_t>(staged->count); sizeSet = true; }
            if (static_cast<uint32_t>(staged->count) != rowCount) continue;
            staged->buffer->retain();
            rel.u32cols[c] = staged->buffer;
        } else if (meta.kind == ColMeta::Kind::StrCharU32) {
            const std::string key = cache_key(c);
            GPUColumn* staged = store.getColumn(key);
            if (!staged) {
                auto host = loadStringCharU32(path, meta.idx);
                if (host.empty()) continue;
                staged = store.stageU32Column(key, host);
            }
            if (!staged || !staged->buffer) continue;
            if (!sizeSet) { rowCount = static_cast<uint32_t>(staged->count); sizeSet = true; }
            if (static_cast<uint32_t>(staged->count) != rowCount) continue;
            staged->buffer->retain();
            rel.u32cols[c] = staged->buffer;
        }
    }

    rel.rowCount = rowCount;
    return rel;
}

std::optional<FilterResult> GpuOps::filterU32(const std::string& colName,
                                                      MTL::Buffer* col,
                                                      uint32_t rowCount,
                                                      engine::expr::CompOp op,
                                                      uint32_t literal) {
    auto& store = ColumnStoreGPU::instance();
    if (!store.device() || !store.library() || !store.queue()) return std::nullopt;

    if (std::getenv("GPUDB_DEBUG_OPS")) {
        std::cerr << "[Exec] GPU filterU32: col=" << colName << " rowCount=" << rowCount << " val=" << literal << "\n";
    }

    const char* fn = nullptr;
    switch (op) {
        case engine::expr::CompOp::EQ: fn = "ops::filter_eq_u32"; break;
        case engine::expr::CompOp::LT: fn = "ops::filter_lt_u32"; break;
        case engine::expr::CompOp::GT: fn = "ops::filter_gt_u32"; break;
        case engine::expr::CompOp::LE: fn = "ops::filter_le_u32"; break;
        case engine::expr::CompOp::GE: fn = "ops::filter_ge_u32"; break;
        case engine::expr::CompOp::NE: fn = "ops::filter_ne_u32"; break;
    }

    auto p_filter = makePSO(store.device(), store.library(), fn);
    auto p_compact = makePSO(store.device(), store.library(), "ops::compact_indices");
    if (!p_filter || !p_compact) {
        return std::nullopt;
    }

    auto mask = store.device()->newBuffer(rowCount * sizeof(uint8_t), MTL::ResourceStorageModeShared);
    std::memset(mask->contents(), 0, rowCount * sizeof(uint8_t));

    auto filterStart = std::chrono::high_resolution_clock::now();
    {
        auto cmd = store.queue()->commandBuffer();
        auto enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(p_filter);
        enc->setBuffer(col, 0, 0);
        enc->setBuffer(mask, 0, 1);
        enc->setBytes(&literal, sizeof(literal), 2);
        if (op != engine::expr::CompOp::EQ) {
            enc->setBytes(&rowCount, sizeof(rowCount), 3);
        }
        dispatch1D(enc, rowCount);
        enc->endEncoding();
        cmd->commit();
        cmd->waitUntilCompleted();
    }
    auto filterEnd = std::chrono::high_resolution_clock::now();
    KernelTimer::instance().record(fn, "filter", 
        std::chrono::duration<double, std::milli>(filterEnd - filterStart).count(), rowCount);

    auto outIdx = store.device()->newBuffer(rowCount * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    auto outCnt = store.device()->newBuffer(sizeof(uint32_t), MTL::ResourceStorageModeShared);
    std::memset(outIdx->contents(), 0, rowCount * sizeof(uint32_t));
    std::memset(outCnt->contents(), 0, sizeof(uint32_t));

    auto compactStart = std::chrono::high_resolution_clock::now();
    {
        auto cmd = store.queue()->commandBuffer();
        auto enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(p_compact);
        enc->setBuffer(mask, 0, 0);
        enc->setBuffer(outIdx, 0, 1);
        enc->setBuffer(outCnt, 0, 2);
        enc->setBytes(&rowCount, sizeof(rowCount), 3);
        dispatch1D(enc, rowCount);
        enc->endEncoding();
        cmd->commit();
        cmd->waitUntilCompleted();
    }
    auto compactEnd = std::chrono::high_resolution_clock::now();
    KernelTimer::instance().record("ops::compact_indices", "compact", 
        std::chrono::duration<double, std::milli>(compactEnd - compactStart).count(), rowCount);

    FilterResult res;
    res.indices = outIdx;
    res.count = *reinterpret_cast<uint32_t*>(outCnt->contents());

    mask->release();
    outCnt->release();
    (void)colName;
    return res;
}

std::optional<FilterResult> GpuOps::filterString(const std::string& colName,
                                                          const std::vector<std::string>& data,
                                                          engine::expr::CompOp op,
                                                          const std::string& pattern,
                                                          MTL::Buffer* preChars,
                                                          MTL::Buffer* preOffsets,
                                                          MTL::Buffer* preLengths) {
    auto& store = ColumnStoreGPU::instance();
    if (!store.device() || !store.library() || !store.queue()) return std::nullopt;

    // 1. Prepare data
    size_t rowCount = data.size();
    if (rowCount == 0) return FilterResult{nullptr, 0};
    if (std::getenv("GPUDB_DEBUG_OPS")) {
        std::cerr << "[Exec] GPU filterString: rowCount=" << rowCount << " pattern=" << pattern << "\n";
    }
    
    // Use pre-flattened GPU buffers when available (Phase 1 Arrow-style)
    bool ownBufs = (preChars == nullptr);
    MTL::Buffer* bufChars   = preChars;
    MTL::Buffer* bufOffsets = preOffsets;
    MTL::Buffer* bufLengths = preLengths;

    if (ownBufs) {
        // Flatten
        std::vector<uint32_t> offsets(rowCount);
        std::vector<uint32_t> lengths(rowCount);
        
        // First pass: calculate total size
        size_t totalChars = 0;
        for (const auto& s : data) {
            totalChars += s.size();
        }
        
        std::vector<char> chars;
        chars.reserve(totalChars);
        
        // Second pass: fill
        size_t currentOffset = 0;
        for (size_t i = 0; i < rowCount; ++i) {
            offsets[i] = static_cast<uint32_t>(currentOffset);
            lengths[i] = static_cast<uint32_t>(data[i].size());
            if (!data[i].empty()) {
                chars.insert(chars.end(), data[i].begin(), data[i].end());
            }
            currentOffset += data[i].size();
        }
        
        // 2. Upload to GPU
        if (!chars.empty()) {
            bufChars = store.device()->newBuffer(chars.data(), chars.size(), MTL::ResourceStorageModeShared);
        } else {
            bufChars = store.device()->newBuffer(1, MTL::ResourceStorageModeShared);
        }

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
    if (std::getenv("GPUDB_DEBUG_OPS")) std::cerr << "[Exec] GPU filterString: dispatching kernel rowCount=" << rowCount
                                                   << (useMultiContains ? " (multi-contains, " + std::to_string(numSegments) + " segments)" : "")
                                                   << (ownBufs ? " [flatten]" : " [flat-precomp]")
                                                   << "\n";
    
    const char* kernelName = useMultiContains ? "ops::filter_string_multi_contains" : "ops::filter_string_contains";
    if (!useMultiContains) {
        switch(op) {
            case engine::expr::CompOp::EQ: kernelName = "ops::filter_string_eq"; break;
            case engine::expr::CompOp::NE: kernelName = "ops::filter_string_ne"; break;
            case engine::expr::CompOp::LT: kernelName = "ops::filter_string_lt"; break;
            case engine::expr::CompOp::LE: kernelName = "ops::filter_string_le"; break;
            case engine::expr::CompOp::GT: kernelName = "ops::filter_string_gt"; break;
            case engine::expr::CompOp::GE: kernelName = "ops::filter_string_ge"; break;
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
    
    auto filterStart = std::chrono::high_resolution_clock::now();

    {
        if (std::getenv("GPUDB_DEBUG_OPS")) std::cerr << "Encoding...\n";
        auto cmd = store.queue()->commandBuffer();
        auto enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(p_filter);
        enc->setBuffer(bufChars, 0, 0);
        enc->setBuffer(bufOffsets, 0, 1);
        enc->setBuffer(bufLengths, 0, 2);
        enc->setBuffer(mask, 0, 3);
        if (useMultiContains) {
            enc->setBuffer(bufPattern, 0, 4);
            enc->setBuffer(bufPatOffsets, 0, 5);
            enc->setBuffer(bufPatLengths, 0, 6);
            enc->setBytes(&numSegments, sizeof(numSegments), 7);
            enc->setBytes(&rc, sizeof(rc), 8);
        } else {
            enc->setBuffer(bufPattern, 0, 4);
            enc->setBytes(&patternLen, sizeof(patternLen), 5);
            enc->setBytes(&rc, sizeof(rc), 6);
        }
        if (std::getenv("GPUDB_DEBUG_OPS")) std::cerr << "Dispatching 1D...\n";
        dispatch1D(enc, rowCount);
        enc->endEncoding();
        cmd->commit();
        if (std::getenv("GPUDB_DEBUG_OPS")) std::cerr << "Waiting...\n";
        cmd->waitUntilCompleted();
        if (std::getenv("GPUDB_DEBUG_OPS")) std::cerr << "Done waiting.\n";
    }
    
    auto filterEnd = std::chrono::high_resolution_clock::now();
    KernelTimer::instance().record(kernelName, "filter", 
        std::chrono::duration<double, std::milli>(filterEnd - filterStart).count(), rowCount);

    if (ownBufs) { bufChars->release(); bufOffsets->release(); bufLengths->release(); }
    bufPattern->release();
    if (bufPatOffsets) bufPatOffsets->release();
    if (bufPatLengths) bufPatLengths->release();
    
    // 4b. Flip mask for NOTLIKE with multi-segment pattern
    if (useMultiContains && op == engine::expr::CompOp::NE) {
        flipMaskU8(mask, (uint32_t)rowCount);
        if (std::getenv("GPUDB_DEBUG_OPS"))
            std::cerr << "[Exec] GPU filterString: flipped mask for NOTLIKE multi-contains\n";
    }
    
    // 5. Compact Results
    auto outIdx = store.device()->newBuffer(rowCount * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    auto outCnt = store.device()->newBuffer(sizeof(uint32_t), MTL::ResourceStorageModeShared);
    std::memset(outIdx->contents(), 0, rowCount * sizeof(uint32_t));
    std::memset(outCnt->contents(), 0, sizeof(uint32_t));
    
    auto compactStart = std::chrono::high_resolution_clock::now();
    {
        if (std::getenv("GPUDB_DEBUG_OPS")) std::cerr << "Compacting...\n";
        auto cmd = store.queue()->commandBuffer();
        auto enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(p_compact);
        enc->setBuffer(mask, 0, 0);
        enc->setBuffer(outIdx, 0, 1);
        enc->setBuffer(outCnt, 0, 2);
        enc->setBytes(&rc, sizeof(rc), 3);
        dispatch1D(enc, rowCount);
        enc->endEncoding();
        cmd->commit();
        cmd->waitUntilCompleted();
        if (std::getenv("GPUDB_DEBUG_OPS")) std::cerr << "Compact done.\n";
    }
    auto compactEnd = std::chrono::high_resolution_clock::now();
    KernelTimer::instance().record("ops::compact_indices", "compact", 
        std::chrono::duration<double, std::milli>(compactEnd - compactStart).count(), rowCount);
    
    mask->release();
    
    FilterResult res;
    res.indices = outIdx;
    res.count = *reinterpret_cast<uint32_t*>(outCnt->contents());

    outCnt->release();

    return res;
}

std::optional<FilterResult> GpuOps::filterStringPrefix(const std::string& colName,
                                                          const std::vector<std::string>& data,
                                                          const std::string& pattern,
                                                          bool invert,
                                                          MTL::Buffer* preChars,
                                                          MTL::Buffer* preOffsets,
                                                          MTL::Buffer* preLengths) {
    auto& store = ColumnStoreGPU::instance();
    if (!store.device() || !store.library() || !store.queue()) {
        std::cerr << "[GpuOps] Device/Lib/Queue invalid\n";
        return std::nullopt;
    }

    size_t rowCount = data.size();
    if (rowCount == 0) return FilterResult{nullptr, 0};
    
    if (std::getenv("GPUDB_DEBUG_OPS")) {
        std::cerr << "[GpuOps] filterStringPrefix pattern='" << pattern << "' invert=" << invert << " rowCount=" << rowCount << "\n";
    }
    
    // Use pre-flattened GPU buffers when available (Phase 1 Arrow-style)
    bool ownBufs = (preChars == nullptr);
    MTL::Buffer* bufChars   = preChars;
    MTL::Buffer* bufOffsets = preOffsets;
    MTL::Buffer* bufLengths = preLengths;

    if (ownBufs) {
        std::vector<uint32_t> offsets(rowCount);
        std::vector<uint32_t> lengths(rowCount);
        size_t totalChars = 0;
        for (const auto& s : data) totalChars += s.size();
        
        std::vector<char> chars;
        chars.reserve(totalChars);
        
        size_t currentOffset = 0;
        for (size_t i = 0; i < rowCount; ++i) {
            offsets[i] = static_cast<uint32_t>(currentOffset);
            lengths[i] = static_cast<uint32_t>(data[i].size());
            if (!data[i].empty()) {
                chars.insert(chars.end(), data[i].begin(), data[i].end());
            }
            currentOffset += data[i].size();
        }
        
        if (!chars.empty()) bufChars = store.device()->newBuffer(chars.data(), chars.size(), MTL::ResourceStorageModeShared);
        else bufChars = store.device()->newBuffer(1, MTL::ResourceStorageModeShared);

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

    if (std::getenv("GPUDB_DEBUG_OPS")) {
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

    {
        auto cmd = store.queue()->commandBuffer();
        auto enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(p_filter);
        enc->setBuffer(bufChars, 0, 0);
        enc->setBuffer(bufOffsets, 0, 1);
        enc->setBuffer(bufLengths, 0, 2);
        enc->setBuffer(mask, 0, 3);
        enc->setBuffer(bufPattern, 0, 4);
        enc->setBytes(&patternLen, sizeof(patternLen), 5);
        enc->setBytes(&rc, sizeof(rc), 6);
        dispatch1D(enc, rowCount);
        enc->endEncoding();
        cmd->commit();
        cmd->waitUntilCompleted();
    }

    if (ownBufs) { bufChars->release(); bufOffsets->release(); bufLengths->release(); }
    bufPattern->release();
    
    auto outIdx = store.device()->newBuffer(rowCount * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    auto outCnt = store.device()->newBuffer(sizeof(uint32_t), MTL::ResourceStorageModeShared);
    std::memset(outIdx->contents(), 0, rowCount * sizeof(uint32_t));
    std::memset(outCnt->contents(), 0, sizeof(uint32_t));

    {
        auto cmd = store.queue()->commandBuffer();
        auto enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(p_compact);
        enc->setBuffer(mask, 0, 0);
        enc->setBuffer(outIdx, 0, 1);
        enc->setBuffer(outCnt, 0, 2);
        enc->setBytes(&rc, sizeof(rc), 3);
        dispatch1D(enc, rowCount);
        enc->endEncoding();
        cmd->commit();
        cmd->waitUntilCompleted();
    }
    
    mask->release();
    
    FilterResult res;
    res.indices = outIdx;
    res.count = *reinterpret_cast<uint32_t*>(outCnt->contents());

    outCnt->release();
    return res;
}

JoinResult GpuOps::joinHash(MTL::Buffer* buildKeys, 
                                     MTL::Buffer* buildIndices, 
                                     uint32_t buildCount,
                                     MTL::Buffer* probeKeys,
                                     MTL::Buffer* probeIndices,
                                     uint32_t probeCount) {
    auto& store = ColumnStoreGPU::instance();
    const char* dbgEnv = std::getenv("GPUDB_DEBUG_OPS");
    bool debug = dbgEnv && (std::string(dbgEnv) == "1" || std::string(dbgEnv) == "true");
    if (buildCount == 0 || probeCount == 0 || !store.device()) return {nullptr, nullptr, 0};

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
        return {nullptr, nullptr, 0};
    }
    
    {
        auto cmd = store.queue()->commandBuffer();
        auto enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(p_build);
        enc->setBuffer(buildKeys, 0, 0);
        enc->setBuffer(bufHTKeys, 0, 1);
        enc->setBuffer(bufHTHead, 0, 2);
        enc->setBuffer(bufNext, 0, 3);
        enc->setBytes(&capacity, 4, 4);
        enc->setBytes(&buildCount, 4, 5);
        dispatch1D(enc, buildCount);
        enc->endEncoding();
        cmd->commit();
        cmd->waitUntilCompleted();
    }
    
    // 3. Count Phase — count matches per probe row
    auto bufCounts = store.device()->newBuffer(probeCount * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    auto p_count = makePSO(store.device(), store.library(), "ops::hash_join_probe_count_multi");
    if (!p_count) {
        bufHTKeys->release(); bufHTHead->release(); bufNext->release(); bufCounts->release();
        return {nullptr, nullptr, 0};
    }
    
    {
        auto cmd = store.queue()->commandBuffer();
        auto enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(p_count);
        enc->setBuffer(probeKeys, 0, 0);
        enc->setBuffer(bufHTKeys, 0, 1);
        enc->setBuffer(bufHTHead, 0, 2);
        enc->setBuffer(bufNext, 0, 3);
        enc->setBuffer(bufCounts, 0, 4);
        enc->setBytes(&capacity, 4, 5);
        enc->setBytes(&probeCount, 4, 6);
        dispatch1D(enc, probeCount);
        enc->endEncoding();
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
        return {emptyBuf, store.device()->newBuffer(sizeof(uint32_t), MTL::ResourceStorageModeShared), 0};
    }
    
    // 5. Write Phase — write matched pairs (bufCounts now holds exclusive prefix sums = offsets)
    MTL::Buffer* bufOffsets = bufCounts;  // reuse in-place — scanInPlace converted counts → offsets
    auto outProbeIndices = store.device()->newBuffer(totalPairs * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    auto outBuildIndices = store.device()->newBuffer(totalPairs * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    
    auto p_write = makePSO(store.device(), store.library(), "ops::hash_join_probe_write_multi");
    if (!p_write) {
        bufHTKeys->release(); bufHTHead->release(); bufNext->release(); bufOffsets->release();
        outProbeIndices->release(); outBuildIndices->release();
        return {nullptr, nullptr, 0};
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
    
    return {outBuildIndices, outProbeIndices, totalPairs};
}

JoinResult GpuOps::joinHashU64(MTL::Buffer* buildKeys, 
                                        MTL::Buffer* buildIndices, 
                                        uint32_t buildCount,
                                        MTL::Buffer* probeKeys,
                                        MTL::Buffer* probeIndices,
                                        uint32_t probeCount) {
    auto& store = ColumnStoreGPU::instance();
    const char* dbgEnv = std::getenv("GPUDB_DEBUG_OPS");
    bool debug = dbgEnv && (std::string(dbgEnv) == "1" || std::string(dbgEnv) == "true");
    if (debug) std::cerr << "[GPU] joinHashU64: buildCount=" << buildCount << " probeCount=" << probeCount << std::endl << std::flush;
    if (buildCount == 0 || probeCount == 0 || !store.device()) return {nullptr, nullptr, 0};

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
        return {nullptr, nullptr, 0};
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
        return {nullptr, nullptr, 0};
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
    
    return {outBuildIndices, outProbeIndices, totalPairs};
}

MTL::Buffer* GpuOps::packU32ToU64(MTL::Buffer* c1, MTL::Buffer* c2, uint32_t count) {
    auto& store = ColumnStoreGPU::instance();
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
                                                              engine::expr::CompOp op,
                                                              uint32_t literal) {
    auto& store = ColumnStoreGPU::instance();
    if (!store.device() || !store.library() || !store.queue()) return std::nullopt;

    const char* fn = nullptr;
    switch (op) {
        case engine::expr::CompOp::EQ: fn = "ops::filter_eq_u32_indexed"; break;
        case engine::expr::CompOp::LT: fn = "ops::filter_lt_u32_indexed"; break;
        case engine::expr::CompOp::GT: fn = "ops::filter_gt_u32_indexed"; break;
        case engine::expr::CompOp::LE: fn = "ops::filter_le_u32_indexed"; break;
        case engine::expr::CompOp::GE: fn = "ops::filter_ge_u32_indexed"; break;
        case engine::expr::CompOp::NE: fn = "ops::filter_ne_u32_indexed"; break;
    }

    auto p_filter = makePSO(store.device(), store.library(), fn);
    auto p_compact = makePSO(store.device(), store.library(), "ops::compact_indices_indexed");
    if (!p_filter || !p_compact) {
        return std::nullopt;
    }

    auto mask = store.device()->newBuffer(static_cast<size_t>(count) * sizeof(uint8_t), MTL::ResourceStorageModeShared);
    std::memset(mask->contents(), 0, static_cast<size_t>(count) * sizeof(uint8_t));

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
        cmd->commit();
        cmd->waitUntilCompleted();
    }
    auto filterEnd = std::chrono::high_resolution_clock::now();
    KernelTimer::instance().record(fn, "filter", 
        std::chrono::duration<double, std::milli>(filterEnd - filterStart).count(), count);

    auto outIdx = store.device()->newBuffer(static_cast<size_t>(count) * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    auto outCnt = store.device()->newBuffer(sizeof(uint32_t), MTL::ResourceStorageModeShared);
    std::memset(outIdx->contents(), 0, static_cast<size_t>(count) * sizeof(uint32_t));
    std::memset(outCnt->contents(), 0, sizeof(uint32_t));

    auto compactStart = std::chrono::high_resolution_clock::now();
    {
        auto cmd = store.queue()->commandBuffer();
        auto enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(p_compact);
        enc->setBuffer(mask, 0, 0);
        enc->setBuffer(indices, 0, 1);    // Pass input indices for indexed compact
        enc->setBuffer(outIdx, 0, 2);
        enc->setBuffer(outCnt, 0, 3);
        enc->setBytes(&count, sizeof(count), 4);
        dispatch1D(enc, count);
        enc->endEncoding();
        cmd->commit();
        cmd->waitUntilCompleted();
    }
    auto compactEnd = std::chrono::high_resolution_clock::now();
    KernelTimer::instance().record("ops::compact_indices_indexed", "compact", 
        std::chrono::duration<double, std::milli>(compactEnd - compactStart).count(), count);

    FilterResult res;
    res.indices = outIdx;
    res.count = *reinterpret_cast<uint32_t*>(outCnt->contents());

    bool debug = (std::getenv("GPUDB_DEBUG_OPS") != nullptr);
    if (debug) std::cerr << "[Exec] GPU filterU32Indexed: col=" << colName << " rowCount=" << count << " val=" << literal << " result=" << res.count << "\n";

    mask->release();
    outCnt->release();
    (void)colName;
    return res;
}

std::optional<FilterResult> GpuOps::filterF32(const std::string& colName,
                                                      MTL::Buffer* col,
                                                      uint32_t rowCount,
                                                      engine::expr::CompOp op,
                                                      float literal) {
    auto& store = ColumnStoreGPU::instance();
    if (!store.device() || !store.library() || !store.queue()) return std::nullopt;

    const char* fn = nullptr;
    switch (op) {
        case engine::expr::CompOp::EQ: fn = "ops::filter_eq_f32"; break;
        case engine::expr::CompOp::LT: fn = "ops::filter_lt_f32"; break;
        case engine::expr::CompOp::GT: fn = "ops::filter_gt_f32"; break;
        case engine::expr::CompOp::LE: fn = "ops::filter_le_f32"; break;
        case engine::expr::CompOp::GE: fn = "ops::filter_ge_f32"; break;
        case engine::expr::CompOp::NE: fn = "ops::filter_ne_f32"; break;
    }

    auto p_filter = makePSO(store.device(), store.library(), fn);
    auto p_compact = makePSO(store.device(), store.library(), "ops::compact_indices");
    if (!p_filter || !p_compact) {
        return std::nullopt;
    }

    auto mask = store.device()->newBuffer(rowCount * sizeof(uint8_t), MTL::ResourceStorageModeShared);
    std::memset(mask->contents(), 0, rowCount * sizeof(uint8_t));

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
        cmd->commit();
        cmd->waitUntilCompleted();
    }
    auto filterEnd = std::chrono::high_resolution_clock::now();
    KernelTimer::instance().record(fn, "filter", 
        std::chrono::duration<double, std::milli>(filterEnd - filterStart).count(), rowCount);

    auto outIdx = store.device()->newBuffer(rowCount * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    auto outCnt = store.device()->newBuffer(sizeof(uint32_t), MTL::ResourceStorageModeShared);
    std::memset(outIdx->contents(), 0, rowCount * sizeof(uint32_t));
    std::memset(outCnt->contents(), 0, sizeof(uint32_t));

    auto compactStart = std::chrono::high_resolution_clock::now();
    {
        auto cmd = store.queue()->commandBuffer();
        auto enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(p_compact);
        enc->setBuffer(mask, 0, 0);
        enc->setBuffer(outIdx, 0, 1);
        enc->setBuffer(outCnt, 0, 2);
        enc->setBytes(&rowCount, sizeof(rowCount), 3);
        dispatch1D(enc, rowCount);
        enc->endEncoding();
        cmd->commit();
        cmd->waitUntilCompleted();
    }
    auto compactEnd = std::chrono::high_resolution_clock::now();
    KernelTimer::instance().record("ops::compact_indices", "compact", 
        std::chrono::duration<double, std::milli>(compactEnd - compactStart).count(), rowCount);

    FilterResult res;
    res.indices = outIdx;
    res.count = *reinterpret_cast<uint32_t*>(outCnt->contents());

    mask->release();
    outCnt->release();
    (void)colName;
    return res;
}

std::optional<FilterResult> GpuOps::filterF32Indexed(const std::string& colName,
                                                              MTL::Buffer* col,
                                                              MTL::Buffer* indices,
                                                              uint32_t count,
                                                              engine::expr::CompOp op,
                                                              float literal) {
    auto& store = ColumnStoreGPU::instance();
    if (!store.device() || !store.library() || !store.queue()) return std::nullopt;

    const char* fn = nullptr;
    switch (op) {
        case engine::expr::CompOp::EQ: fn = "ops::filter_eq_f32_indexed"; break;
        case engine::expr::CompOp::LT: fn = "ops::filter_lt_f32_indexed"; break;
        case engine::expr::CompOp::GT: fn = "ops::filter_gt_f32_indexed"; break;
        case engine::expr::CompOp::LE: fn = "ops::filter_le_f32_indexed"; break;
        case engine::expr::CompOp::GE: fn = "ops::filter_ge_f32_indexed"; break;
        case engine::expr::CompOp::NE: fn = "ops::filter_ne_f32_indexed"; break;
    }

    auto p_filter = makePSO(store.device(), store.library(), fn);
    auto p_compact = makePSO(store.device(), store.library(), "ops::compact_indices_indexed");
    if (!p_filter || !p_compact) {
        return std::nullopt;
    }

    auto mask = store.device()->newBuffer(static_cast<size_t>(count) * sizeof(uint8_t), MTL::ResourceStorageModeShared);
    std::memset(mask->contents(), 0, static_cast<size_t>(count) * sizeof(uint8_t));

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
        cmd->commit();
        cmd->waitUntilCompleted();
    }
    auto filterEnd = std::chrono::high_resolution_clock::now();
    KernelTimer::instance().record(fn, "filter", 
        std::chrono::duration<double, std::milli>(filterEnd - filterStart).count(), count);

    auto outIdx = store.device()->newBuffer(static_cast<size_t>(count) * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    auto outCnt = store.device()->newBuffer(sizeof(uint32_t), MTL::ResourceStorageModeShared);
    std::memset(outIdx->contents(), 0, static_cast<size_t>(count) * sizeof(uint32_t));
    std::memset(outCnt->contents(), 0, sizeof(uint32_t));

    auto compactStart = std::chrono::high_resolution_clock::now();
    {
        auto cmd = store.queue()->commandBuffer();
        auto enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(p_compact);
        enc->setBuffer(mask, 0, 0);
        enc->setBuffer(indices, 0, 1);    // Pass input indices for indexed compact
        enc->setBuffer(outIdx, 0, 2);
        enc->setBuffer(outCnt, 0, 3);
        enc->setBytes(&count, sizeof(count), 4);
        dispatch1D(enc, count);
        enc->endEncoding();
        cmd->commit();
        cmd->waitUntilCompleted();
    }
    auto compactEnd = std::chrono::high_resolution_clock::now();
    KernelTimer::instance().record("ops::compact_indices_indexed", "compact", 
        std::chrono::duration<double, std::milli>(compactEnd - compactStart).count(), count);

    FilterResult res;
    res.indices = outIdx;
    res.count = *reinterpret_cast<uint32_t*>(outCnt->contents());

    mask->release();
    outCnt->release();
    (void)colName;
    return res;
}

MTL::Buffer* GpuOps::gatherU32(MTL::Buffer* in, MTL::Buffer* indices, uint32_t count, bool sync) {
    auto& store = ColumnStoreGPU::instance();
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
    auto& store = ColumnStoreGPU::instance();
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
    auto& store = ColumnStoreGPU::instance();
    auto cmd = store.queue()->commandBuffer();
    cmd->commit();
    cmd->waitUntilCompleted();
}

MTL::Buffer* GpuOps::castU32ToF32(MTL::Buffer* in, uint32_t count) {
    auto& store = ColumnStoreGPU::instance();
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
        cmd->waitUntilCompleted();
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
    auto& store = ColumnStoreGPU::instance();
    if (!store.device() || !store.library() || !store.queue()) return std::nullopt;

    auto p = makePSO(store.device(), store.library(), "ops::groupby_agg_multi_key_typed");
    if (!p) return std::nullopt;

    uint32_t numKeys = static_cast<uint32_t>(keyColsU32.size());
    if (numKeys == 0 || numKeys > 8) return std::nullopt;

    const uint32_t numAggs = static_cast<uint32_t>(aggTypes.size());
    if (numAggs == 0 || numAggs > 16) return std::nullopt;
    if (aggInputsF32.size() < numAggs) return std::nullopt;

    uint32_t cap = nextPow2(std::max<uint32_t>(1024u, rowCount * 2u));
    // Stride increased from 4 to 8, size is cap * 8 * sizeof(uint32_t)
    auto ht_keys = store.device()->newBuffer(static_cast<size_t>(cap) * 8 * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    auto ht_aggs = store.device()->newBuffer(static_cast<size_t>(cap) * 16 * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    std::memset(ht_keys->contents(), 0, static_cast<size_t>(cap) * 8 * sizeof(uint32_t));
    std::memset(ht_aggs->contents(), 0, static_cast<size_t>(cap) * 16 * sizeof(uint32_t));

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

    MTL::Buffer* k0 = keyColsU32.size() > 0 ? keyColsU32[0] : keyColsU32[0];
    MTL::Buffer* k1 = keyColsU32.size() > 1 ? keyColsU32[1] : keyColsU32[0];
    MTL::Buffer* k2 = keyColsU32.size() > 2 ? keyColsU32[2] : keyColsU32[0];
    MTL::Buffer* k3 = keyColsU32.size() > 3 ? keyColsU32[3] : keyColsU32[0];
    MTL::Buffer* k4 = keyColsU32.size() > 4 ? keyColsU32[4] : keyColsU32[0];
    MTL::Buffer* k5 = keyColsU32.size() > 5 ? keyColsU32[5] : keyColsU32[0];
    MTL::Buffer* k6 = keyColsU32.size() > 6 ? keyColsU32[6] : keyColsU32[0];
    MTL::Buffer* k7 = keyColsU32.size() > 7 ? keyColsU32[7] : keyColsU32[0];

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

        enc->setBuffer(ht_keys, 0, 20);
        enc->setBuffer(ht_aggs, 0, 21);
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
    g.ht_keys = ht_keys;
    g.ht_aggs = ht_aggs;
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
    auto& store = ColumnStoreGPU::instance();
    if (!store.device() || !store.library() || !store.queue()) return std::nullopt;

    auto p_mark    = makePSO(store.device(), store.library(), "ops::ht_mark_valid");
    auto p_extract = makePSO(store.device(), store.library(), "ops::ht_extract_compact");
    if (!p_mark || !p_extract) return std::nullopt;

    uint32_t cap = ht.capacity;
    if (cap == 0) return GroupByExtractResult{{}, {}, 0};

    // Step 1 (Mark): GPU writes 1 for valid slots, 0 for empty.
    auto markBuf = store.device()->newBuffer(
        static_cast<size_t>(cap) * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    {
        auto cmd = store.queue()->commandBuffer();
        auto enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(p_mark);
        enc->setBuffer(ht.ht_keys, 0, 0);
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
        return GroupByExtractResult{{}, {}, 0};
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
        enc->setBuffer(ht.ht_keys, 0, 0);
        enc->setBuffer(ht.ht_aggs, 0, 1);
        enc->setBuffer(markBuf, 0, 2);
        enc->setBuffer(offsetsBuf, 0, 3);
        enc->setBuffer(outKeysBuf, 0, 4);
        enc->setBuffer(outAggsBuf, 0, 5);
        enc->setBytes(&cap, sizeof(cap), 6);
        enc->setBytes(&numKeys, sizeof(numKeys), 7);
        enc->setBytes(&numAggsTotal, sizeof(numAggsTotal), 8);
        dispatch1D(enc, cap);
        enc->endEncoding();
        auto t0 = std::chrono::high_resolution_clock::now();
        cmd->commit();
        cmd->waitUntilCompleted();
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        KernelTimer::instance().record("ops::ht_extract_compact", "groupby", ms, totalCount);
    }

    // Deinterleave compacted GPU output to per-column CPU vectors.
    GroupByExtractResult result;
    result.rowCount = totalCount;
    result.keyCols.resize(numKeys);
    result.aggWords.resize(numAggsTotal);

    auto* keyPtr = reinterpret_cast<const uint32_t*>(outKeysBuf->contents());
    auto* aggPtr = reinterpret_cast<const uint32_t*>(outAggsBuf->contents());

    for (uint32_t k = 0; k < numKeys; ++k) result.keyCols[k].resize(totalCount);
    for (uint32_t a = 0; a < numAggsTotal; ++a) result.aggWords[a].resize(totalCount);

    for (uint32_t r = 0; r < totalCount; ++r) {
        for (uint32_t k = 0; k < numKeys; ++k) {
            result.keyCols[k][r] = keyPtr[r * numKeys + k];
        }
        for (uint32_t a = 0; a < numAggsTotal; ++a) {
            result.aggWords[a][r] = aggPtr[r * numAggsTotal + a];
        }
    }

    markBuf->release();
    offsetsBuf->release();
    outKeysBuf->release();
    outAggsBuf->release();

    return result;
}

void GpuOps::release(FilterResult& r) {
    if (r.indices) r.indices->release();
    r.indices = nullptr;
    r.count = 0;
}

void GpuOps::release(JoinMapGPU& j) {
    if (j.leftRow) j.leftRow->release();
    if (j.rightRow) j.rightRow->release();
    j.leftRow = nullptr;
    j.rightRow = nullptr;
    j.count = 0;
}

void GpuOps::release(GroupByHashTable& g) {
    if (g.ht_keys) g.ht_keys->release();
    if (g.ht_aggs) g.ht_aggs->release();
    g.ht_keys = nullptr;
    g.ht_aggs = nullptr;
    g.capacity = 0;
}

std::vector<std::string> GpuOps::loadStringColumnRaw(const std::string& dataset_path,
                                                           const std::string& table,
                                                           const std::string& column) {
    const auto itT = kSchema.find(table);
    if (itT == kSchema.end()) return {};
    
    const auto itC = itT->second.find(column);
    if (itC == itT->second.end()) return {};
    
    std::string path = dataset_path + table + ".tbl";
    return loadStringColumnRawImpl(path, itC->second.idx);
}

std::optional<FilterResult> GpuOps::filterColColU32(
    MTL::Buffer* colA,
    MTL::Buffer* colB,
    uint32_t count,
    int opInt) {
    auto& store = ColumnStoreGPU::instance();
    if (!store.device() || !store.library() || !store.queue()) return std::nullopt;
    
    engine::expr::CompOp op = static_cast<engine::expr::CompOp>(opInt);
    const char* kernelName = nullptr;
    switch(op) {
        case engine::expr::CompOp::EQ: kernelName = "ops::filter_u32_col_col_eq"; break;
        case engine::expr::CompOp::NE: kernelName = "ops::filter_u32_col_col_ne"; break;
        case engine::expr::CompOp::LT: kernelName = "ops::filter_u32_col_col_lt"; break;
        case engine::expr::CompOp::LE: kernelName = "ops::filter_u32_col_col_le"; break;
        case engine::expr::CompOp::GT: kernelName = "ops::filter_u32_col_col_gt"; break;
        case engine::expr::CompOp::GE: kernelName = "ops::filter_u32_col_col_ge"; break;
        default: return std::nullopt;
    }

    auto p_filter = makePSO(store.device(), store.library(), kernelName);
    auto p_compact = makePSO(store.device(), store.library(), "ops::compact_indices");
    if (!p_filter || !p_compact) return std::nullopt;

    auto mask = store.device()->newBuffer(count * sizeof(uint8_t), MTL::ResourceStorageModeShared);
    
    // Filter
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
        cmd->commit();
        cmd->waitUntilCompleted();
    }

    // Compact
    auto outIdx = store.device()->newBuffer(count * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    auto outCnt = store.device()->newBuffer(sizeof(uint32_t), MTL::ResourceStorageModeShared);
    *(uint32_t*)outCnt->contents() = 0;

    {
        auto cmd = store.queue()->commandBuffer();
        auto enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(p_compact);
        enc->setBuffer(mask, 0, 0);
        enc->setBuffer(outIdx, 0, 1);
        enc->setBuffer(outCnt, 0, 2);
        enc->setBytes(&count, sizeof(count), 3);
        dispatch1D(enc, count);
        enc->endEncoding();
        cmd->commit();
        cmd->waitUntilCompleted();
    }
    
    mask->release();

    FilterResult res;
    res.indices = outIdx;
    res.count = *reinterpret_cast<uint32_t*>(outCnt->contents());
    outCnt->release();
    return res;
}

std::optional<FilterResult> GpuOps::filterColColF32(
    MTL::Buffer* colA,
    MTL::Buffer* colB,
    uint32_t count,
    int opInt) {
    auto& store = ColumnStoreGPU::instance();
    if (!store.device() || !store.library() || !store.queue()) return std::nullopt;

    engine::expr::CompOp op = static_cast<engine::expr::CompOp>(opInt);
    const char* kernelName = nullptr;
    switch(op) {
        case engine::expr::CompOp::EQ: kernelName = "ops::filter_f32_col_col_eq"; break;
        case engine::expr::CompOp::NE: kernelName = "ops::filter_f32_col_col_ne"; break;
        case engine::expr::CompOp::LT: kernelName = "ops::filter_f32_col_col_lt"; break;
        case engine::expr::CompOp::LE: kernelName = "ops::filter_f32_col_col_le"; break;
        case engine::expr::CompOp::GT: kernelName = "ops::filter_f32_col_col_gt"; break;
        case engine::expr::CompOp::GE: kernelName = "ops::filter_f32_col_col_ge"; break;
        default: return std::nullopt;
    }

    auto p_filter = makePSO(store.device(), store.library(), kernelName);
    auto p_compact = makePSO(store.device(), store.library(), "ops::compact_indices");
    if (!p_filter || !p_compact) return std::nullopt;

    auto mask = store.device()->newBuffer(count * sizeof(uint8_t), MTL::ResourceStorageModeShared);
    
    // Filter
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
        cmd->commit();
        cmd->waitUntilCompleted();
    }

    // Compact
    auto outIdx = store.device()->newBuffer(count * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    auto outCnt = store.device()->newBuffer(sizeof(uint32_t), MTL::ResourceStorageModeShared);
    *(uint32_t*)outCnt->contents() = 0;

    {
        auto cmd = store.queue()->commandBuffer();
        auto enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(p_compact);
        enc->setBuffer(mask, 0, 0);
        enc->setBuffer(outIdx, 0, 1);
        enc->setBuffer(outCnt, 0, 2);
        enc->setBytes(&count, sizeof(count), 3);
        dispatch1D(enc, count);
        enc->endEncoding();
        cmd->commit();
        cmd->waitUntilCompleted();
    }
    
    mask->release();

    FilterResult res;
    res.indices = outIdx;
    res.count = *reinterpret_cast<uint32_t*>(outCnt->contents());
    outCnt->release();
    return res;
}


MTL::Buffer* GpuOps::logicOrU32(MTL::Buffer* colA, MTL::Buffer* colB, uint32_t count) {
    auto* dev = ColumnStoreGPU::instance().device();
    auto* lib = ColumnStoreGPU::instance().library();
    auto* cmd = ColumnStoreGPU::instance().queue()->commandBuffer();
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
    auto* dev = ColumnStoreGPU::instance().device();
    auto* lib = ColumnStoreGPU::instance().library();
    auto* cmd = ColumnStoreGPU::instance().queue()->commandBuffer();
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
    auto* dev = ColumnStoreGPU::instance().device();
    auto* lib = ColumnStoreGPU::instance().library();
    auto* cmd = ColumnStoreGPU::instance().queue()->commandBuffer();
    
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
    auto* dev = ColumnStoreGPU::instance().device();
    auto* lib = ColumnStoreGPU::instance().library();
    auto* cmd = ColumnStoreGPU::instance().queue()->commandBuffer();
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
    auto* dev = ColumnStoreGPU::instance().device();
    auto* lib = ColumnStoreGPU::instance().library();
    auto* cmd = ColumnStoreGPU::instance().queue()->commandBuffer();
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

void GpuOps::fillF32(MTL::Buffer* buf, float val, uint32_t count) {
    auto* dev = ColumnStoreGPU::instance().device();
    auto* lib = ColumnStoreGPU::instance().library();
    auto* cmd = ColumnStoreGPU::instance().queue()->commandBuffer();
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

MTL::Buffer* GpuOps::createFilledU32(uint32_t val, uint32_t count) {
    auto* dev = ColumnStoreGPU::instance().device();
    auto* buf = dev->newBuffer(count * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    fillU32(buf, val, count);
    return buf;
}

MTL::Buffer* GpuOps::iotaU32(uint32_t count) {
    auto* dev = ColumnStoreGPU::instance().device();
    auto* lib = ColumnStoreGPU::instance().library();
    auto* cmd = ColumnStoreGPU::instance().queue()->commandBuffer();
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
    auto* dev = ColumnStoreGPU::instance().device();
    auto* lib = ColumnStoreGPU::instance().library();
    auto* cmd = ColumnStoreGPU::instance().queue()->commandBuffer();
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

std::optional<FilterResult> GpuOps::filterF32Between(const std::string& colName,
                                                              MTL::Buffer* col,
                                                              uint32_t rowCount,
                                                              float minVal,
                                                              float maxVal) {
    auto& store = ColumnStoreGPU::instance();
    if (!store.device() || !store.library() || !store.queue()) return std::nullopt;

    const char* fn = "ops::filter_range_to_mask_f32";
    auto p_filter = makePSO(store.device(), store.library(), fn);
    auto p_compact = makePSO(store.device(), store.library(), "ops::compact_indices");
    if (!p_filter || !p_compact) return std::nullopt;

    auto mask = store.device()->newBuffer(rowCount * sizeof(uint8_t), MTL::ResourceStorageModeShared);
    auto outIdx = store.device()->newBuffer(rowCount * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    auto outCnt = store.device()->newBuffer(sizeof(uint32_t), MTL::ResourceStorageModeShared);

    {
        auto cmd = store.queue()->commandBuffer();
        auto enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(p_filter);
        enc->setBuffer(col, 0, 0);
        enc->setBuffer(mask, 0, 1);
        enc->setBytes(&minVal, sizeof(float), 2);
        enc->setBytes(&maxVal, sizeof(float), 3);
        enc->setBytes(&rowCount, sizeof(uint32_t), 4);
        dispatch1D(enc, rowCount);
        enc->endEncoding();
        cmd->commit();
        cmd->waitUntilCompleted();
    }
    
    {
        auto cmd = store.queue()->commandBuffer();
        auto enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(p_compact);
        enc->setBuffer(mask, 0, 0);
        enc->setBuffer(outIdx, 0, 1);
        enc->setBuffer(outCnt, 0, 2);
        enc->setBytes(&rowCount, sizeof(uint32_t), 3);
        dispatch1D(enc, rowCount);
        enc->endEncoding();
        cmd->commit();
        cmd->waitUntilCompleted();
    }

    FilterResult res;
    res.indices = outIdx;
    res.count = *reinterpret_cast<uint32_t*>(outCnt->contents());
    mask->release();
    outCnt->release();
    return res;
}

std::optional<FilterResult> GpuOps::filterU32Between(const std::string& colName,
                                                              MTL::Buffer* col,
                                                              uint32_t rowCount,
                                                              uint32_t minVal,
                                                              uint32_t maxVal) {
    auto& store = ColumnStoreGPU::instance();
    if (!store.device() || !store.library() || !store.queue()) return std::nullopt;

    // Use int32 variant (safe for dates)
    const char* fn = "ops::filter_range_to_mask_int32";
    auto p_filter = makePSO(store.device(), store.library(), fn);
    auto p_compact = makePSO(store.device(), store.library(), "ops::compact_indices");
    if (!p_filter || !p_compact) return std::nullopt;

    auto mask = store.device()->newBuffer(rowCount * sizeof(uint8_t), MTL::ResourceStorageModeShared);
    auto outIdx = store.device()->newBuffer(rowCount * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    auto outCnt = store.device()->newBuffer(sizeof(uint32_t), MTL::ResourceStorageModeShared);

    {
        auto cmd = store.queue()->commandBuffer();
        auto enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(p_filter);
        enc->setBuffer(col, 0, 0);
        enc->setBuffer(mask, 0, 1);
        enc->setBytes(&minVal, sizeof(int32_t), 2);
        enc->setBytes(&maxVal, sizeof(int32_t), 3);
        enc->setBytes(&rowCount, sizeof(uint32_t), 4);
        dispatch1D(enc, rowCount);
        enc->endEncoding();
        cmd->commit();
        cmd->waitUntilCompleted();
    }
    
    {
        auto cmd = store.queue()->commandBuffer();
        auto enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(p_compact);
        enc->setBuffer(mask, 0, 0);
        enc->setBuffer(outIdx, 0, 1);
        enc->setBuffer(outCnt, 0, 2);
        enc->setBytes(&rowCount, sizeof(uint32_t), 3);
        dispatch1D(enc, rowCount);
        enc->endEncoding();
        cmd->commit();
        cmd->waitUntilCompleted();
    }

    FilterResult res;
    res.indices = outIdx;
    res.count = *reinterpret_cast<uint32_t*>(outCnt->contents());
    mask->release();
    outCnt->release();
    return res;
}

std::optional<FilterResult> GpuOps::hashJoinSemiU32(MTL::Buffer* leftKey,
                                                             uint32_t leftCount,
                                                             MTL::Buffer* rightKey,
                                                             uint32_t rightCount) {
    auto& store = ColumnStoreGPU::instance();
    if (!store.device() || !store.library()) return std::nullopt;

    auto p_build = makePSO(store.device(), store.library(), "ops::hash_join_build_multi");
    auto p_probe = makePSO(store.device(), store.library(), "ops::hash_join_probe_semi");
    auto p_compact = makePSO(store.device(), store.library(), "ops::compact_indices");
    if (!p_build || !p_probe || !p_compact) return std::nullopt;

    uint32_t cap = nextPow2(std::max<uint32_t>(8u, rightCount * 2u));
    auto ht_keys = store.device()->newBuffer(cap * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    auto ht_head = store.device()->newBuffer(cap * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    auto next = store.device()->newBuffer(static_cast<size_t>(rightCount) * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    
    std::memset(ht_keys->contents(), 0, cap * sizeof(uint32_t));
    std::memset(ht_head->contents(), 0, cap * sizeof(uint32_t));
    if (rightCount > 0) std::memset(next->contents(), 0, static_cast<size_t>(rightCount) * sizeof(uint32_t));

    // BUILD
    {
        auto cmd = store.queue()->commandBuffer();
        auto enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(p_build);
        enc->setBuffer(rightKey, 0, 0);
        enc->setBuffer(ht_keys, 0, 1);
        enc->setBuffer(ht_head, 0, 2);
        enc->setBuffer(next, 0, 3);
        enc->setBytes(&cap, sizeof(cap), 4);
        enc->setBytes(&rightCount, sizeof(rightCount), 5);
        dispatch1D(enc, rightCount);
        enc->endEncoding();
        cmd->commit();
        cmd->waitUntilCompleted();
    }
    ht_head->release();
    next->release();

    // PROBE -> MASK
    auto mask = store.device()->newBuffer(leftCount * sizeof(uint8_t), MTL::ResourceStorageModeShared);
    
    {
        auto cmd = store.queue()->commandBuffer();
        auto enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(p_probe);
        enc->setBuffer(leftKey, 0, 0);
        enc->setBuffer(ht_keys, 0, 1);
        enc->setBytes(&cap, sizeof(cap), 2);
        enc->setBytes(&leftCount, sizeof(leftCount), 3);
        enc->setBuffer(mask, 0, 4);
        dispatch1D(enc, leftCount);
        enc->endEncoding();
        cmd->commit();
        cmd->waitUntilCompleted();
    }
    ht_keys->release();

    // COMPACT
    auto outIdx = store.device()->newBuffer(leftCount * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    auto outCnt = store.device()->newBuffer(sizeof(uint32_t), MTL::ResourceStorageModeShared);
    *(uint32_t*)outCnt->contents() = 0;

    {
        auto cmd = store.queue()->commandBuffer();
        auto enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(p_compact);
        enc->setBuffer(mask, 0, 0);
        enc->setBuffer(outIdx, 0, 1);
        enc->setBuffer(outCnt, 0, 2);
        enc->setBytes(&leftCount, sizeof(leftCount), 3);
        dispatch1D(enc, leftCount);
        enc->endEncoding();
        cmd->commit();
        cmd->waitUntilCompleted();
    }
    mask->release();
    
    uint32_t validCount = *(uint32_t*)outCnt->contents();
    outCnt->release();
    
    FilterResult res;
    res.indices = outIdx;
    res.count = validCount;
    return res;
}

std::optional<FilterResult> GpuOps::hashJoinAntiU32(MTL::Buffer* leftKey,
                                                             uint32_t leftCount,
                                                             MTL::Buffer* rightKey,
                                                             uint32_t rightCount) {
    auto& store = ColumnStoreGPU::instance();
    if (!store.device() || !store.library()) return std::nullopt;

    // If no right rows, every left row is unmatched
    if (rightCount == 0) {
        auto idx = iotaU32(leftCount);
        return FilterResult{idx, leftCount};
    }

    auto p_build   = makePSO(store.device(), store.library(), "ops::hash_join_build_multi");
    auto p_probe   = makePSO(store.device(), store.library(), "ops::hash_join_probe_semi");
    auto p_flip    = makePSO(store.device(), store.library(), "ops::flip_mask_u8");
    auto p_compact = makePSO(store.device(), store.library(), "ops::compact_indices");
    if (!p_build || !p_probe || !p_flip || !p_compact) return std::nullopt;

    uint32_t cap = nextPow2(std::max<uint32_t>(8u, rightCount * 2u));
    auto ht_keys = store.device()->newBuffer(cap * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    auto ht_head = store.device()->newBuffer(cap * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    auto next    = store.device()->newBuffer(static_cast<size_t>(rightCount) * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    
    std::memset(ht_keys->contents(), 0, cap * sizeof(uint32_t));
    std::memset(ht_head->contents(), 0, cap * sizeof(uint32_t));
    if (rightCount > 0) std::memset(next->contents(), 0, static_cast<size_t>(rightCount) * sizeof(uint32_t));

    // BUILD
    {
        auto cmd = store.queue()->commandBuffer();
        auto enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(p_build);
        enc->setBuffer(rightKey, 0, 0);
        enc->setBuffer(ht_keys, 0, 1);
        enc->setBuffer(ht_head, 0, 2);
        enc->setBuffer(next, 0, 3);
        enc->setBytes(&cap, sizeof(cap), 4);
        enc->setBytes(&rightCount, sizeof(rightCount), 5);
        dispatch1D(enc, rightCount);
        enc->endEncoding();
        cmd->commit();
        cmd->waitUntilCompleted();
    }
    ht_head->release();
    next->release();

    // PROBE → u8 mask (1 = matched)
    auto mask = store.device()->newBuffer(leftCount * sizeof(uint8_t), MTL::ResourceStorageModeShared);
    {
        auto cmd = store.queue()->commandBuffer();
        auto enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(p_probe);
        enc->setBuffer(leftKey, 0, 0);
        enc->setBuffer(ht_keys, 0, 1);
        enc->setBytes(&cap, sizeof(cap), 2);
        enc->setBytes(&leftCount, sizeof(leftCount), 3);
        enc->setBuffer(mask, 0, 4);
        dispatch1D(enc, leftCount);
        enc->endEncoding();
        cmd->commit();
        cmd->waitUntilCompleted();
    }
    ht_keys->release();

    // FLIP mask: 1→0 (matched→discard), 0→1 (unmatched→keep)
    {
        auto cmd = store.queue()->commandBuffer();
        auto enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(p_flip);
        enc->setBuffer(mask, 0, 0);
        enc->setBytes(&leftCount, sizeof(leftCount), 1);
        dispatch1D(enc, leftCount);
        enc->endEncoding();
        cmd->commit();
        cmd->waitUntilCompleted();
    }

    // COMPACT
    auto outIdx = store.device()->newBuffer(leftCount * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    auto outCnt = store.device()->newBuffer(sizeof(uint32_t), MTL::ResourceStorageModeShared);
    *(uint32_t*)outCnt->contents() = 0;
    {
        auto cmd = store.queue()->commandBuffer();
        auto enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(p_compact);
        enc->setBuffer(mask, 0, 0);
        enc->setBuffer(outIdx, 0, 1);
        enc->setBuffer(outCnt, 0, 2);
        enc->setBytes(&leftCount, sizeof(leftCount), 3);
        dispatch1D(enc, leftCount);
        enc->endEncoding();
        cmd->commit();
        cmd->waitUntilCompleted();
    }
    mask->release();

    uint32_t validCount = *(uint32_t*)outCnt->contents();
    outCnt->release();

    FilterResult res;
    res.indices = outIdx;
    res.count = validCount;
    return res;
}

MTL::Buffer* GpuOps::bitcastF32ToU32(MTL::Buffer* in, uint32_t count) {
    auto& store = ColumnStoreGPU::instance();
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
    auto& store = ColumnStoreGPU::instance();
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
    auto& store = ColumnStoreGPU::instance();
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
    auto& store = ColumnStoreGPU::instance();
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
    auto& store = ColumnStoreGPU::instance();

    // Edge case: no matches → every row is unmatched
    if (matchedCount == 0) {
        auto idx = iotaU32(totalRows);
        return {idx, totalRows};
    }
    if (totalRows == 0) {
        return {store.device()->newBuffer(sizeof(uint32_t), MTL::ResourceStorageModeShared), 0};
    }

    auto p_scatter = makePSO(store.device(), store.library(), "ops::scatter_one_u8");
    auto p_flip    = makePSO(store.device(), store.library(), "ops::flip_mask_u8");
    auto p_compact = makePSO(store.device(), store.library(), "ops::compact_indices");

    // Create u8 mask, zero-initialized
    auto mask = store.device()->newBuffer(totalRows * sizeof(uint8_t), MTL::ResourceStorageModeShared);
    std::memset(mask->contents(), 0, totalRows * sizeof(uint8_t));

    if (p_scatter && p_flip && p_compact) {
        // GPU: scatter 1 at matched indices
        {
            auto cmd = store.queue()->commandBuffer();
            auto enc = cmd->computeCommandEncoder();
            enc->setComputePipelineState(p_scatter);
            enc->setBuffer(matchedIndices, 0, 0);
            enc->setBuffer(mask, 0, 1);
            enc->setBytes(&matchedCount, sizeof(matchedCount), 2);
            dispatch1D(enc, matchedCount);
            enc->endEncoding();
            cmd->commit();
            cmd->waitUntilCompleted();
        }
        // GPU: flip mask (1→0 matched, 0→1 unmatched)
        {
            auto cmd = store.queue()->commandBuffer();
            auto enc = cmd->computeCommandEncoder();
            enc->setComputePipelineState(p_flip);
            enc->setBuffer(mask, 0, 0);
            enc->setBytes(&totalRows, sizeof(totalRows), 1);
            dispatch1D(enc, totalRows);
            enc->endEncoding();
            cmd->commit();
            cmd->waitUntilCompleted();
        }
        // GPU: compact
        auto outIdx = store.device()->newBuffer(totalRows * sizeof(uint32_t), MTL::ResourceStorageModeShared);
        auto outCnt = store.device()->newBuffer(sizeof(uint32_t), MTL::ResourceStorageModeShared);
        *(uint32_t*)outCnt->contents() = 0;
        {
            auto cmd = store.queue()->commandBuffer();
            auto enc = cmd->computeCommandEncoder();
            enc->setComputePipelineState(p_compact);
            enc->setBuffer(mask, 0, 0);
            enc->setBuffer(outIdx, 0, 1);
            enc->setBuffer(outCnt, 0, 2);
            enc->setBytes(&totalRows, sizeof(totalRows), 3);
            dispatch1D(enc, totalRows);
            enc->endEncoding();
            cmd->commit();
            cmd->waitUntilCompleted();
        }
        mask->release();
        uint32_t cnt = *(uint32_t*)outCnt->contents();
        outCnt->release();
        return {outIdx, cnt};
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
    return {outIdx, cnt};
}

MTL::Buffer* GpuOps::arithMulF32ColCol(MTL::Buffer* colA, MTL::Buffer* colB, uint32_t count) {
    auto& store = ColumnStoreGPU::instance();
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
        cmd->waitUntilCompleted();
    }
    return out;
}

MTL::Buffer* GpuOps::arithMulF32ColScalar(MTL::Buffer* colA, float valB, uint32_t count) {
    auto& store = ColumnStoreGPU::instance();
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
        cmd->waitUntilCompleted();
    }
    return out;
}

MTL::Buffer* GpuOps::arithDivF32ColCol(MTL::Buffer* colA, MTL::Buffer* colB, uint32_t count) {
    auto& store = ColumnStoreGPU::instance();
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
        cmd->waitUntilCompleted();
    }
    return out;
}

MTL::Buffer* GpuOps::arithDivF32ColScalar(MTL::Buffer* colA, float valB, uint32_t count) {
    auto& store = ColumnStoreGPU::instance();
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
        cmd->waitUntilCompleted();
    }
    return out;
}

MTL::Buffer* GpuOps::arithDivF32ScalarCol(float valA, MTL::Buffer* colB, uint32_t count) {
    auto& store = ColumnStoreGPU::instance();
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
        cmd->waitUntilCompleted();
    }
    return out;
}

MTL::Buffer* GpuOps::arithSubF32ColCol(MTL::Buffer* colA, MTL::Buffer* colB, uint32_t count) {
    auto& store = ColumnStoreGPU::instance();
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
        cmd->waitUntilCompleted();
    }
    return out;
}

MTL::Buffer* GpuOps::arithSubF32ColScalar(MTL::Buffer* colA, float valB, uint32_t count) {
    auto& store = ColumnStoreGPU::instance();
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
        cmd->waitUntilCompleted();
    }
    return out;
}

MTL::Buffer* GpuOps::arithSubF32ScalarCol(float valA, MTL::Buffer* colB, uint32_t count) {
    auto& store = ColumnStoreGPU::instance();
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
        cmd->waitUntilCompleted();
    }
    return out;
}

MTL::Buffer* GpuOps::createBuffer(const void* data, size_t size) {
    auto& store = ColumnStoreGPU::instance();
    if (!store.device()) return nullptr;
    if (data) {
        return store.device()->newBuffer(data, size, MTL::ResourceStorageModeShared);
    } else {
        return store.device()->newBuffer(size, MTL::ResourceStorageModeShared);
    }
}

float GpuOps::reduceSumF32(MTL::Buffer* in, uint32_t count) {
    auto& store = ColumnStoreGPU::instance();
    auto p = makePSO(store.device(), store.library(), "ops::reduce_sum_f32");
    if (!p) return 0.0f;

    auto out = store.device()->newBuffer(sizeof(float), MTL::ResourceStorageModeShared);
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
    out->release();
    return res;
}

float GpuOps::reduceMinF32(MTL::Buffer* in, uint32_t count) {
    auto& store = ColumnStoreGPU::instance();
    auto p = makePSO(store.device(), store.library(), "ops::reduce_min_f32");
    if (!p) return 0.0f;

    auto out = store.device()->newBuffer(sizeof(float), MTL::ResourceStorageModeShared);
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
    out->release();
    return res;
}

float GpuOps::reduceMaxF32(MTL::Buffer* in, uint32_t count) {
    auto& store = ColumnStoreGPU::instance();
    auto p = makePSO(store.device(), store.library(), "ops::reduce_max_f32");
    if (!p) return 0.0f;

    auto out = store.device()->newBuffer(sizeof(float), MTL::ResourceStorageModeShared);
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
    out->release();
    return res;
}

// ── M11: Extract YEAR from u32 date → u32 year ─────────────────────────
MTL::Buffer* GpuOps::extractYearU32(MTL::Buffer* dateCol, uint32_t count) {
    if (count == 0) return nullptr;
    auto& store = ColumnStoreGPU::instance();
    auto dev = store.device();
    auto lib = store.library();
    if (!dev || !lib || !store.queue()) return nullptr;

    MTL::Buffer* outBuf = dev->newBuffer(count * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    if (!outBuf) return nullptr;

    auto pso = makePSO(dev, lib, "extract_year_u32_to_u32");
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
    cmd->waitUntilCompleted();

    return outBuf;
}

// ── H4: GPU dedup by sorted keys ────────────────────────────────────────
MTL::Buffer* GpuOps::dedupByKeys(const std::vector<MTL::Buffer*>& keys, uint32_t count,
                                  uint32_t& uniqueCount) {
    uniqueCount = 0;
    if (count == 0 || keys.empty()) return nullptr;
    auto& store = ColumnStoreGPU::instance();
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
        // 3+ keys: pack first two into u64, then fold remaining via hash
        sortKeys = packU32ToU64(keys[0], keys[1], count);
        ownSortKeys = true;
        auto* ptr = (uint64_t*)sortKeys->contents();
        for (size_t k = 2; k < keys.size(); ++k) {
            auto* kp = (const uint32_t*)keys[k]->contents();
            for (uint32_t i = 0; i < count; ++i) {
                ptr[i] = ptr[i] * 0x9E3779B97F4A7C15ULL + kp[i];
            }
        }
    }

    // Create index array [0, 1, 2, ...]
    MTL::Buffer* indices = dev->newBuffer(count * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    {
        auto* ip = (uint32_t*)indices->contents();
        for (uint32_t i = 0; i < count; ++i) ip[i] = i;
    }

    // Radix sort
    if (useU64) {
        radixSortU64(sortKeys, indices, count);
    } else {
        radixSortU32(sortKeys, indices, count);
    }

    // Mark unique positions after sort
    MTL::Buffer* mask = dev->newBuffer(count * sizeof(uint8_t), MTL::ResourceStorageModeShared);
    const char* kernelName = useU64 ? "mark_unique_sorted_u64" : "mark_unique_sorted_u32";
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
    auto& store = ColumnStoreGPU::instance();
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
        cmd->waitUntilCompleted();
    }
    return out;
}

MTL::Buffer* GpuOps::arithAddF32ColScalar(MTL::Buffer* colA, float valB, uint32_t count) {
    auto& store = ColumnStoreGPU::instance();
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
        cmd->waitUntilCompleted();
    }
    return out;
}

void GpuOps::scatterConstantF32(MTL::Buffer* output, MTL::Buffer* indices, uint32_t indexCount, float val) {
    if (indexCount == 0 || !output || !indices) return;

    auto& store = ColumnStoreGPU::instance();

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

    auto& store = ColumnStoreGPU::instance();
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
    auto& store = ColumnStoreGPU::instance();
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
        cmd->waitUntilCompleted();
    }
    return out;
}

// ============================================================================
// GPU Radix Sort (stable, 8-bit radix)
// ============================================================================
// For ≤1024 elements: single-dispatch block sort (shared-memory bitonic).
// For >1024 elements: multi-pass LSD radix sort (histogram → scan → scatter).

static void blockSortU32(MTL::Buffer* keys, MTL::Buffer* indices, uint32_t count) {
    auto& store = ColumnStoreGPU::instance();
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
    auto& store = ColumnStoreGPU::instance();
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

    auto& store = ColumnStoreGPU::instance();
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

    auto& store = ColumnStoreGPU::instance();
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
