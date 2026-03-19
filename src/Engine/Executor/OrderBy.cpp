#include "GpuExecutor.hpp"
#include "GpuExecutorDetail.hpp"
#include "Operators.hpp"
#include "GpuColumnStore.hpp"
#include "KernelTimer.hpp"

#include <chrono>
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <unordered_map>
#include <cstring>
#include "Logger.hpp"

namespace engine {

// -- Extracted: resolveSortColumns --
struct SortCol {
    int type;      // 0=u32, 1=f32, 2=string
    bool ascending;
    size_t colIdx;
};

static std::vector<SortCol> resolveSortColumns(
    const IROrderBy& order, const TableResult& table, bool /*debug*/)
{
    std::vector<SortCol> sortCols;

    for (size_t i = 0; i < order.columns.size(); ++i) {
        const std::string& colName = order.columns[i];
        bool asc = i < order.ascending.size() ? order.ascending[i] : true;
        bool found = false;

        // Check string_names FIRST — u32 columns for strings store hash values
        // that sort numerically (wrong). String sort is always correct.
        for (size_t j = 0; j < table.stringNames.size() && !found; ++j) {
            if (table.stringNames[j] == colName || base_ident(table.stringNames[j]) == base_ident(colName)) {
                sortCols.push_back({2, asc, j});
                found = true;
            }
        }
        for (size_t j = 0; j < table.u32Names.size() && !found; ++j) {
            if (table.u32Names[j] == colName || base_ident(table.u32Names[j]) == base_ident(colName)) {
                sortCols.push_back({0, asc, j});
                found = true;
            }
        }
        for (size_t j = 0; j < table.f32Names.size() && !found; ++j) {
            if (table.f32Names[j] == colName || base_ident(table.f32Names[j]) == base_ident(colName)) {
                sortCols.push_back({1, asc, j});
                found = true;
            }
        }

        if (!found) {
            std::string colLower = colName;
            std::transform(colLower.begin(), colLower.end(), colLower.begin(), ::tolower);
            bool isSum = colLower.find("sum(") != std::string::npos || colLower.find("sum_no_overflow(") != std::string::npos;
            bool isAvg = colLower.find("avg(") != std::string::npos;
            bool isCount = colLower.find("count(") != std::string::npos || colLower.find("count_star(") != std::string::npos;
            bool isMin = colLower.find("min(") != std::string::npos;
            bool isMax = colLower.find("max(") != std::string::npos;

            if (isSum || isAvg || isCount || isMin || isMax) {
                for (size_t j = 0; j < table.f32Names.size() && !found; ++j) {
                    const std::string& name = table.f32Names[j];
                    std::string nameLower = name;
                    std::transform(nameLower.begin(), nameLower.end(), nameLower.begin(), ::tolower);
                    if ((isSum && (nameLower.find("revenue") != std::string::npos || 
                                   nameLower.find("sum") != std::string::npos ||
                                   nameLower.find("price") != std::string::npos ||
                                   nameLower.find("charge") != std::string::npos)) ||
                        (isCount && (nameLower.find("count") != std::string::npos ||
                                     nameLower.find("_cnt") != std::string::npos ||
                                     nameLower.find("dist") != std::string::npos)) ||
                        (isAvg && nameLower.find("avg") != std::string::npos) ||
                        (isMin && nameLower.find("min") != std::string::npos) ||
                        (isMax && nameLower.find("max") != std::string::npos)) {
                        sortCols.push_back({1, asc, j});
                        found = true;
                        LOG_DEBUG("Exec", "OrderBy: matched '" << colName << "' to '" << name << "'\n");
                    }
                }

                if (!found && table.f32Names.size() == 1) {
                    sortCols.push_back({1, asc, 0});
                    found = true;
                    LOG_DEBUG("Exec", "OrderBy: fallback '" << colName << "' to single f32 '" << table.f32Names[0] << "'\n");
                }

                if (!found && colLower.find("distinct") != std::string::npos) {
                    for (size_t j = 0; j < table.f32Names.size() && !found; ++j) {
                        const std::string& name = table.f32Names[j];
                        if (name.size() >= 2 && name[0] == '#') {
                            sortCols.push_back({1, asc, j});
                            found = true;
                            LOG_DEBUG("Exec", "OrderBy: matched COUNT(DISTINCT) to positional '" << name << "'\n");
                        }
                    }
                }
            }
        }
    }

    return sortCols;
}

// -- Extracted: buildRankStringStatic --
// Builds a GPU buffer of uint32 rank values for a string sort column.
// Uses dictionary IDs when available, then GPU prefix-u64 sort, then CPU fallback.
static MTL::Buffer* buildRankStringStatic(
    const std::vector<std::string>& col, bool asc, const std::string& colName,
    uint32_t n, const std::unordered_map<std::string, DictEncoded>& dictCols,
    const std::unordered_map<std::string, FlatStringCol>& flatStringCols,
    bool debug) {
    auto& s = GpuColumnStore::instance();
    auto dictIt = dictCols.find(colName);
    if (dictIt != dictCols.end() && dictIt->second.rowCount == n) {
        const auto& dict = dictIt->second;
        if (asc) {
            if (dict.idsGPU) {
                dict.idsGPU->retain();
                return dict.idsGPU.get();  // reuse existing GPU buffer
            } else {
                return s.device()->newBuffer(dict.ids.data(), n * sizeof(uint32_t), MTL::ResourceStorageModeShared);
            }
        } else {
            if (dict.idsGPU) {
                return GpuOps::invertU32(dict.idsGPU, n).detach();
            } else {
                MTL::Buffer* buf = s.device()->newBuffer(dict.ids.data(), n * sizeof(uint32_t), MTL::ResourceStorageModeShared);
                GpuBuffer inv = GpuOps::invertU32(buf, n);
                buf->release();
                return inv.detach();
            }
        }
    }

    // GPU path: prefix-u64 sort + rank assignment (no CPU string ops)
    auto flatIt = flatStringCols.find(colName);
    if (flatIt != flatStringCols.end() && flatIt->second.chars && flatIt->second.rowCount == n) {
        const auto& flat = flatIt->second;
        GpuBuffer rankBuf = GpuOps::stringRankByPrefix(
            flat.chars, flat.offsets, flat.lengths, n, asc);
        if (rankBuf) {
            if (debug)
                LOG_INFO("Exec", "OrderBy: GPU prefix-u64 rank for '" << colName << "' (all-GPU, no ties)\n");
            return rankBuf.detach();
        }
        // Ties detected — fall through to CPU
        if (debug)
            LOG_INFO("Exec", "OrderBy: GPU prefix-u64 ties for '" << colName << "', CPU fallback\n");
    }

    // CPU fallback: prefix-u64 accelerated ranking
    auto cpuPrefix = [](const std::string& str) -> uint64_t {
        uint64_t val = 0;
        size_t m = std::min(str.size(), size_t(8));
        for (size_t i = 0; i < m; ++i) val = (val << 8) | uint8_t(str[i]);
        val <<= (8 - m) * 8;
        return val;
    };

    uint32_t rc = static_cast<uint32_t>(col.size());
    std::vector<std::string> uniq(col.begin(), col.end());
    std::sort(uniq.begin(), uniq.end(), [&](const std::string& a, const std::string& b) {
        uint64_t pa = cpuPrefix(a), pb = cpuPrefix(b);
        if (pa != pb) return pa < pb;
        return a < b;
    });
    uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());

    std::unordered_map<std::string, uint32_t> rankMap;
    rankMap.reserve(uniq.size());
    for (uint32_t r = 0; r < (uint32_t)uniq.size(); ++r) rankMap[uniq[r]] = r;

    uint32_t maxRank = (uint32_t)uniq.size();
    std::vector<uint32_t> rank(n);
    for (uint32_t i = 0; i < rc; ++i) {
        auto it2 = rankMap.find(col[i]);
        uint32_t r = (it2 != rankMap.end()) ? it2->second : maxRank;
        rank[i] = asc ? r : (maxRank - 1 - r);
    }

    if (debug) {
        LOG_INFO("Exec", "OrderBy: Prefix-u64 rank for '" << colName << "' (" << uniq.size() << " unique)\n");
    }
    return s.device()->newBuffer(rank.data(), n * sizeof(uint32_t), MTL::ResourceStorageModeShared);
}

// -- Extracted: gpuRadixSortComposite --
// Packs rank GPU buffers into composite u64 keys and runs GPU radix sort.
// Returns the sorted index buffer (caller must release).
static MTL::Buffer* gpuRadixSortComposite(
    std::vector<MTL::Buffer*>& rankBufs,
    const std::vector<SortCol>& sortCols,
    uint32_t n, bool debug) {
    auto& store = GpuColumnStore::instance();

    if (debug) {
        LOG_INFO("Exec", "OrderBy: sortCols.size()=" << sortCols.size() << " rankBufs.size()=" << rankBufs.size() << " n=" << n);
        for (size_t k = 0; k < rankBufs.size(); ++k) {
            LOG_INFO("Exec", "OrderBy: rankBufs[" << k << "] first few = [");
            auto* p = static_cast<const uint32_t*>(rankBufs[k]->contents());
            for (uint32_t i = 0; i < std::min(n, 20u); ++i)
                LOG_INFO("SORT", p[i] << (i+1<n?",":""));
            LOG_INFO("SORT", "]\n");
        }
    }

    GpuBuffer idxBuf = GpuOps::iotaU32(n);

    if (sortCols.size() <= 2) {
        MTL::Buffer* rank0Buf = rankBufs[0];
        MTL::Buffer* rank1Buf = nullptr;
        if (sortCols.size() > 1) {
            rank1Buf = rankBufs[1];
        } else {
            rank1Buf = store.device()->newBuffer(n * sizeof(uint32_t), MTL::ResourceStorageModeShared);
            std::memset(rank1Buf->contents(), 0, n * sizeof(uint32_t));
        }
        GpuBuffer keyBuf = GpuOps::packU32ToU64(rank0Buf, rank1Buf, n);
        rank0Buf->release();
        if (sortCols.size() <= 1) rank1Buf->release();
        else rank1Buf->release();

        GpuOps::radixSortU64(keyBuf, idxBuf, n);
    } else {
        for (int k = (int)sortCols.size() - 1; k >= 0; --k) {
            GpuBuffer gatheredRank = GpuOps::gatherU32(rankBufs[k], idxBuf, n, true);
            rankBufs[k]->release();
            GpuBuffer posBuf = GpuOps::iotaU32(n);
            GpuBuffer keyBuf = GpuOps::packU32ToU64(gatheredRank, posBuf, n);
            GpuOps::radixSortU64(keyBuf, idxBuf, n);
        }
    }

    if (debug) {
        std::vector<uint32_t> dbgIdx(std::min(n, 20u));
        std::memcpy(dbgIdx.data(), idxBuf->contents(), dbgIdx.size() * sizeof(uint32_t));
        LOG_INFO("Exec", "OrderBy: sortedIdx = [");
        for (uint32_t i = 0; i < (uint32_t)dbgIdx.size(); ++i)
            LOG_INFO("SORT", dbgIdx[i] << (i+1<n?",":""));
        LOG_INFO("SORT", "]\n");
    }

    return idxBuf.detach();
}

// -- Extracted: gatherNumericColumns --
// GPU-gathers all u32 and f32 columns by sorted index, syncs once, copies back to CPU.
static void gatherNumericColumns(TableResult& table, MTL::Buffer* idxBuf, uint32_t n, bool /*debug*/) {
    auto& store = GpuColumnStore::instance();
    uint32_t totalGatherElements = (uint32_t)(table.u32Cols.size() + table.f32Cols.size()) * n;
    auto gatherStart = std::chrono::high_resolution_clock::now();

    std::vector<GpuBuffer> gatheredU32;
    std::vector<MTL::Buffer*> srcU32Bufs;
    gatheredU32.reserve(table.u32Cols.size());
    srcU32Bufs.reserve(table.u32Cols.size());
    for (size_t ci = 0; ci < table.u32Cols.size(); ++ci) {
        MTL::Buffer* srcBuf = (ci < table.u32ColsGPU.size() && table.u32ColsGPU[ci])
            ? table.u32ColsGPU[ci] : nullptr;
        bool ownSrc = false;
        if (!srcBuf) {
            srcBuf = store.device()->newBuffer(
                table.u32Cols[ci].data(), n * sizeof(uint32_t), MTL::ResourceStorageModeShared);
            ownSrc = true;
        }
        gatheredU32.push_back(GpuOps::gatherU32(srcBuf, idxBuf, n, false));
        srcU32Bufs.push_back(ownSrc ? srcBuf : nullptr);
    }

    std::vector<GpuBuffer> gatheredF32;
    std::vector<MTL::Buffer*> srcF32Bufs;
    gatheredF32.reserve(table.f32Cols.size());
    srcF32Bufs.reserve(table.f32Cols.size());
    for (size_t ci = 0; ci < table.f32Cols.size(); ++ci) {
        MTL::Buffer* srcBuf = (ci < table.f32ColsGPU.size() && table.f32ColsGPU[ci])
            ? table.f32ColsGPU[ci] : nullptr;
        bool ownSrc = false;
        if (!srcBuf) {
            srcBuf = store.device()->newBuffer(
                table.f32Cols[ci].data(), n * sizeof(float), MTL::ResourceStorageModeShared);
            ownSrc = true;
        }
        gatheredF32.push_back(GpuOps::gatherF32(srcBuf, idxBuf, n, false));
        srcF32Bufs.push_back(ownSrc ? srcBuf : nullptr);
    }

    GpuOps::sync();
    auto gatherEnd = std::chrono::high_resolution_clock::now();
    double gatherMs = std::chrono::duration<double, std::milli>(gatherEnd - gatherStart).count();
    KernelTimer::instance().record("orderby_gpu_gather", "sort", gatherMs, totalGatherElements);

    table.u32ColsGPU.resize(table.u32Cols.size());
    for (size_t i = 0; i < table.u32Cols.size(); ++i) {
        // Skip CPU download — GPU buffer is authoritative; lazy-fetch at output
        table.u32Cols[i].clear();
        table.u32ColsGPU[i] = std::move(gatheredU32[i]);
        if (srcU32Bufs[i]) srcU32Bufs[i]->release();
    }
    table.f32ColsGPU.resize(table.f32Cols.size());
    for (size_t i = 0; i < table.f32Cols.size(); ++i) {
        // Skip CPU download — GPU buffer is authoritative; lazy-fetch at output
        table.f32Cols[i].clear();
        table.f32ColsGPU[i] = std::move(gatheredF32[i]);
        if (srcF32Bufs[i]) srcF32Bufs[i]->release();
    }
}

// -- Extracted: reorderStringColumns --
// Reorders string columns using GPU dict-ID gather, GPU flat-string gather, or CPU fallback.
static void reorderStringColumns(
    TableResult& table, MTL::Buffer* idxBuf, uint32_t n,
    const std::unordered_map<std::string, DictEncoded>& dictCols,
    const std::unordered_map<std::string, FlatStringCol>& flatStringCols,
    bool debug) {
    if (table.stringCols.empty()) return;

    std::vector<uint32_t> sortedIdx(n);
    std::memcpy(sortedIdx.data(), idxBuf->contents(), n * sizeof(uint32_t));

    for (size_t ci = 0; ci < table.stringCols.size(); ++ci) {
        const std::string& colName = table.stringNames[ci];
        bool isDeferred = table.stringCols[ci].empty();

        // --- Deferred string path: keep GPU-resident after reorder ---
        if (isDeferred) {
            // Try dict (from ctx or tableResult)
            auto dictIt = dictCols.find(colName);
            if (dictIt == dictCols.end() || !dictIt->second.idsGPU)
                dictIt = table.dictStringResults.find(colName) != table.dictStringResults.end()
                    ? (void(0), dictCols.find("__NEVER__"))  // won't match, fall through
                    : dictCols.end();
            // Check tableResult.dictStringResults
            auto tDictIt = table.dictStringResults.find(colName);
            if (tDictIt != table.dictStringResults.end() && tDictIt->second.idsGPU && tDictIt->second.rowCount == n) {
                GpuBuffer gatheredIds = GpuOps::gatherU32(tDictIt->second.idsGPU, idxBuf, n);
                DictEncoded reordered;
                reordered.dictionary = tDictIt->second.dictionary;
                reordered.idsGPU = std::move(gatheredIds);
                reordered.rowCount = n;
                table.dictStringResults[colName] = std::move(reordered);
                table.flatStringResults.erase(colName);
                if (debug) LOG_INFO("Exec", "OrderBy: DEFERRED dict reorder '" << colName << "'\n");
                continue;
            }
            if (dictIt != dictCols.end() && dictIt->second.idsGPU && dictIt->second.rowCount == n) {
                GpuBuffer gatheredIds = GpuOps::gatherU32(dictIt->second.idsGPU, idxBuf, n);
                DictEncoded reordered;
                reordered.dictionary = dictIt->second.dictionary;
                reordered.idsGPU = std::move(gatheredIds);
                reordered.rowCount = n;
                table.dictStringResults[colName] = std::move(reordered);
                table.flatStringResults.erase(colName);
                if (debug) LOG_INFO("Exec", "OrderBy: DEFERRED dict reorder (ctx) '" << colName << "'\n");
                continue;
            }
            // Try flat (from tableResult or ctx)
            FlatStringCol const* flatSrc = nullptr;
            auto tFlatIt = table.flatStringResults.find(colName);
            if (tFlatIt != table.flatStringResults.end() && tFlatIt->second.chars && tFlatIt->second.rowCount == n)
                flatSrc = &tFlatIt->second;
            if (!flatSrc) {
                auto fit = flatStringCols.find(colName);
                if (fit != flatStringCols.end() && fit->second.chars && fit->second.rowCount == n)
                    flatSrc = &fit->second;
            }
            if (flatSrc) {
                auto r = GpuOps::gatherFlatString(
                    flatSrc->chars, flatSrc->offsets, flatSrc->lengths,
                    idxBuf, n, true);
                if (r.chars) {
                    FlatStringCol reordered;
                    reordered.takeFrom(std::move(r.chars), std::move(r.offsets),
                                       std::move(r.lengths), r.rowCount, r.totalBytes);
                    table.flatStringResults[colName] = std::move(reordered);
                    if (debug) LOG_INFO("Exec", "OrderBy: DEFERRED flat reorder '" << colName << "'\n");
                    continue;
                }
            }
            // No deferred source found — skip (shouldn't happen)
            continue;
        }

        // --- Materialized string path (existing behavior) ---
        auto dictIt = dictCols.find(colName);
        if (dictIt != dictCols.end() && dictIt->second.idsGPU && dictIt->second.rowCount == n) {
            auto gatheredIds = gatherToVector<uint32_t>(dictIt->second.idsGPU, idxBuf, n);
            const auto& dict = dictIt->second.dictionary;
            auto& col = table.stringCols[ci];
            for (uint32_t i = 0; i < n; ++i) col[i] = dict[gatheredIds[i]];
            if (debug)
                LOG_INFO("Exec", "OrderBy: GPU dict reorder '" << colName << "'\n");
        } else {
            auto fit = flatStringCols.find(colName);
            if (fit != flatStringCols.end() && fit->second.chars && fit->second.rowCount == n) {
                auto r = GpuOps::gatherFlatString(
                    fit->second.chars, fit->second.offsets, fit->second.lengths,
                    idxBuf, n, true);
                if (r.chars) {
                    const uint32_t* offs = static_cast<const uint32_t*>(r.offsets->contents());
                    const uint32_t* lens = static_cast<const uint32_t*>(r.lengths->contents());
                    const char* ch = static_cast<const char*>(r.chars->contents());
                    auto& col = table.stringCols[ci];
                    for (uint32_t i = 0; i < n; ++i) col[i].assign(ch + offs[i], lens[i]);
                    if (debug)
                        LOG_INFO("Exec", "OrderBy: GPU flat string reorder '" << colName << "'\n");
                    continue;
                }
            }
            auto& col = table.stringCols[ci];
            std::vector<std::string> tmp(n);
            for (uint32_t i = 0; i < n; ++i) tmp[i] = std::move(col[sortedIdx[i]]);
            col = std::move(tmp);
        }
    }
}

bool GpuExecutor::executeOrderBy(const IROrderBy& order, TableResult& table,
                                 const std::unordered_map<std::string, DictEncoded>& dictCols,
                                 const std::unordered_map<std::string, FlatStringCol>& flatStringCols) {
    const bool debug = env_truthy("GPUDB_DEBUG_OPS");
    
    if (debug) {
        LOG_INFO("Exec", "OrderBy: columns=[");
        for (size_t i = 0; i < order.columns.size(); ++i) {
            LOG_INFO("SORT", order.columns[i]);
            if (i < order.ascending.size()) LOG_DEBUG("SORT", (order.ascending[i] ? " ASC" : " DESC"));
            if (i + 1 < order.columns.size()) LOG_DEBUG("SORT", ", ");
        }
        LOG_DEBUG("SORT", "]\n");
        LOG_DEBUG("Exec", "OrderBy: table.u32Names=[");
        if (debug) for (const auto& n : table.u32Names) std::cerr << n << ", ";
        LOG_DEBUG("SORT", "]\n");
        LOG_DEBUG("Exec", "OrderBy: table.f32Names=[");
        if (debug) for (const auto& n : table.f32Names) std::cerr << n << ", ";
        LOG_DEBUG("SORT", "]\n");
        LOG_DEBUG("Exec", "OrderBy: table.stringNames=[");
        if (debug) for (const auto& n : table.stringNames) std::cerr << n << ", ";
        LOG_DEBUG("SORT", "]\n");
    }
    
    if (table.rowCount == 0) return true;
    
    std::vector<SortCol> sortCols = resolveSortColumns(order, table, debug);
    
    uint32_t n = (uint32_t)table.rowCount;

    if (debug) {
        LOG_INFO("Exec", "OrderBy: GPU bitonic sort with " << sortCols.size() << " sort col(s), " << n << " rows\n");
    }

    // Guard: if no valid sort columns were resolved, nothing to sort
    if (sortCols.empty()) {
        LOG_DEBUG("Exec", "OrderBy: no sort columns resolved, skipping sort\n");
        return true;
    }

    auto buildRankU32 = [&](const std::vector<uint32_t>& col, bool asc, MTL::Buffer* gpuBuf) -> MTL::Buffer* {
        // GPU path: use existing GPU buffer or upload, optionally invert for DESC
        auto& s = GpuColumnStore::instance();
        MTL::Buffer* srcBuf = gpuBuf;
        bool ownSrc = false;
        if (!srcBuf) {
            srcBuf = s.device()->newBuffer(col.data(), n * sizeof(uint32_t), MTL::ResourceStorageModeShared);
            ownSrc = true;
        }
        if (asc) {
            if (ownSrc) return srcBuf;
            // Need a copy since we shouldn't modify the original
            MTL::Buffer* copy = s.device()->newBuffer(srcBuf->contents(), n * sizeof(uint32_t), MTL::ResourceStorageModeShared);
            return copy;
        } else {
            GpuBuffer inv = GpuOps::invertU32(srcBuf, n);
            if (ownSrc) srcBuf->release();
            return inv.detach();
        }
    };

    auto buildRankF32 = [&](const std::vector<float>& col, bool asc, MTL::Buffer* gpuBuf) -> MTL::Buffer* {
        // GPU path: use existing GPU buffer or upload, convert to sort key u32 on GPU
        auto& s = GpuColumnStore::instance();
        MTL::Buffer* srcBuf = gpuBuf;
        bool ownSrc = false;
        if (!srcBuf) {
            srcBuf = s.device()->newBuffer(col.data(), n * sizeof(float), MTL::ResourceStorageModeShared);
            ownSrc = true;
        }
        GpuBuffer keyBuf = GpuOps::floatToSortKeyU32(srcBuf, n, !asc);
        if (ownSrc) srcBuf->release();
        return keyBuf.detach();
    };

    auto buildRankString = [&](const std::vector<std::string>& col, bool asc,
                               const std::string& colName) -> MTL::Buffer* {
        return buildRankStringStatic(col, asc, colName, n, dictCols, flatStringCols, debug);
    };

    // Build rank GPU buffers — stay on GPU throughout (no CPU round-trip)
    std::vector<MTL::Buffer*> rankBufs;
    for (const auto& sc : sortCols) {
        MTL::Buffer* gpuBuf = nullptr;
        if (sc.type == 0) {
            // Check for pre-existing GPU buffer in TableResult
            gpuBuf = (sc.colIdx < table.u32ColsGPU.size()) ? table.u32ColsGPU[sc.colIdx] : nullptr;
            rankBufs.push_back(buildRankU32(table.u32Cols[sc.colIdx], sc.ascending, gpuBuf));
        } else if (sc.type == 1) {
            gpuBuf = (sc.colIdx < table.f32ColsGPU.size()) ? table.f32ColsGPU[sc.colIdx] : nullptr;
            rankBufs.push_back(buildRankF32(table.f32Cols[sc.colIdx], sc.ascending, gpuBuf));
        } else {
            rankBufs.push_back(buildRankString(table.stringCols[sc.colIdx], sc.ascending,
                                            table.stringNames[sc.colIdx]));
        }
    }

    // GPU sort: composite-key radix sort, then gather all columns
    {
        MTL::Buffer* idxBuf = gpuRadixSortComposite(rankBufs, sortCols, n, debug);
        gatherNumericColumns(table, idxBuf, n, debug);
        reorderStringColumns(table, idxBuf, n, dictCols, flatStringCols, debug);
        idxBuf->release();
    }

    if (debug) {
        LOG_INFO("Exec", "OrderBy: GPU sort complete, " << n << " rows sorted\n");
    }
    
    return true;
}

} // namespace engine
