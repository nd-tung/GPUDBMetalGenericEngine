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

namespace engine {

bool GpuExecutor::executeOrderBy(const IROrderBy& order, TableResult& table,
                                 const std::unordered_map<std::string, DictEncoded>& dictCols,
                                 const std::unordered_map<std::string, FlatStringCol>& flatStringCols) {
    const bool debug = env_truthy("GPUDB_DEBUG_OPS");
    
    if (debug) {
        std::cerr << "[Exec] OrderBy: columns=[";
        for (size_t i = 0; i < order.columns.size(); ++i) {
            std::cerr << order.columns[i];
            if (i < order.ascending.size()) std::cerr << (order.ascending[i] ? " ASC" : " DESC");
            if (i + 1 < order.columns.size()) std::cerr << ", ";
        }
        std::cerr << "]\n";
        std::cerr << "[Exec] OrderBy: table.u32Names=[";
        for (const auto& n : table.u32Names) std::cerr << n << ", ";
        std::cerr << "]\n";
        std::cerr << "[Exec] OrderBy: table.f32Names=[";
        for (const auto& n : table.f32Names) std::cerr << n << ", ";
        std::cerr << "]\n";
        std::cerr << "[Exec] OrderBy: table.stringNames=[";
        for (const auto& n : table.stringNames) std::cerr << n << ", ";
        std::cerr << "]\n";
    }
    
    if (table.rowCount == 0) return true;
    
    struct SortCol {
        int type;
        bool ascending;
        size_t colIdx;
    };
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
                        if (debug) std::cerr << "[Exec] OrderBy: matched '" << colName << "' to '" << name << "'\n";
                    }
                }
                
                if (!found && table.f32Names.size() == 1) {
                    sortCols.push_back({1, asc, 0});
                    found = true;
                    if (debug) std::cerr << "[Exec] OrderBy: fallback '" << colName << "' to single f32 '" << table.f32Names[0] << "'\n";
                }
                
                if (!found && colLower.find("distinct") != std::string::npos) {
                    for (size_t j = 0; j < table.f32Names.size() && !found; ++j) {
                        const std::string& name = table.f32Names[j];
                        if (name.size() >= 2 && name[0] == '#') {
                            sortCols.push_back({1, asc, j});
                            found = true;
                            if (debug) std::cerr << "[Exec] OrderBy: matched COUNT(DISTINCT) to positional '" << name << "'\n";
                        }
                    }
                }
            }
        }
    }
    
    uint32_t n = (uint32_t)table.rowCount;

    if (debug) {
        std::cerr << "[Exec] OrderBy: GPU bitonic sort with " << sortCols.size() << " sort col(s), " << n << " rows\n";
    }

    // Guard: if no valid sort columns were resolved, nothing to sort
    if (sortCols.empty()) {
        if (debug) std::cerr << "[Exec] OrderBy: no sort columns resolved, skipping sort\n";
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
            MTL::Buffer* inv = GpuOps::invertU32(srcBuf, n);
            if (ownSrc) srcBuf->release();
            return inv;
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
        MTL::Buffer* keyBuf = GpuOps::floatToSortKeyU32(srcBuf, n, !asc);
        if (ownSrc) srcBuf->release();
        return keyBuf;
    };

    auto buildRankString = [&](const std::vector<std::string>& col, bool asc,
                               const std::string& colName) -> MTL::Buffer* {
        // Path 0: Dictionary ID path — dict IDs are already lexicographic ranks
        auto& s = GpuColumnStore::instance();
        auto dictIt = dictCols.find(colName);
        if (dictIt != dictCols.end() && dictIt->second.rowCount == n) {
            const auto& dict = dictIt->second;
            if (asc) {
                // ASC: dict IDs are already lexicographic ranks — use GPU buffer directly
                if (dict.idsGPU) {
                    MTL::Buffer* copy = s.device()->newBuffer(dict.idsGPU->contents(), n * sizeof(uint32_t), MTL::ResourceStorageModeShared);
                    return copy;
                } else {
                    return s.device()->newBuffer(dict.ids.data(), n * sizeof(uint32_t), MTL::ResourceStorageModeShared);
                }
            } else {
                // DESC: invert on GPU
                if (dict.idsGPU) {
                    MTL::Buffer* inv = GpuOps::invertU32(dict.idsGPU, n);
                    return inv;
                } else {
                    MTL::Buffer* buf = s.device()->newBuffer(dict.ids.data(), n * sizeof(uint32_t), MTL::ResourceStorageModeShared);
                    MTL::Buffer* inv = GpuOps::invertU32(buf, n);
                    buf->release();
                    return inv;
                }
            }
        }

        // Path 1: Prefix-u64 accelerated ranking.
        // Sort unique strings using 8-byte prefix comparison (resolves >99%
        // of pairs without full string compare). Eliminates hash collision risk.
        auto cpuPrefix = [](const std::string& s) -> uint64_t {
            uint64_t val = 0;
            size_t m = std::min(s.size(), size_t(8));
            for (size_t i = 0; i < m; ++i) val = (val << 8) | uint8_t(s[i]);
            val <<= (8 - m) * 8;
            return val;
        };

        // Build sorted unique strings with prefix-accelerated comparator
        uint32_t rc = static_cast<uint32_t>(col.size());
        std::vector<std::string> uniq(col.begin(), col.end());
        std::sort(uniq.begin(), uniq.end(), [&](const std::string& a, const std::string& b) {
            uint64_t pa = cpuPrefix(a), pb = cpuPrefix(b);
            if (pa != pb) return pa < pb;
            return a < b; // Full compare only for equal 8-byte prefixes
        });
        uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());

        // Build string → rank map
        std::unordered_map<std::string, uint32_t> rankMap;
        rankMap.reserve(uniq.size());
        for (uint32_t r = 0; r < (uint32_t)uniq.size(); ++r) {
            rankMap[uniq[r]] = r;
        }

        // Assign ranks to all rows
        uint32_t maxRank = (uint32_t)uniq.size();
        std::vector<uint32_t> rank(n);
        for (uint32_t i = 0; i < rc; ++i) {
            auto it2 = rankMap.find(col[i]);
            uint32_t r = (it2 != rankMap.end()) ? it2->second : maxRank;
            rank[i] = asc ? r : (maxRank - 1 - r);
        }

        if (debug) {
            std::cerr << "[Exec] OrderBy: Prefix-u64 rank for '" << colName
                      << "' (" << uniq.size() << " unique)\n";
        }
        return s.device()->newBuffer(rank.data(), n * sizeof(uint32_t), MTL::ResourceStorageModeShared);
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

    auto& store = GpuColumnStore::instance();

    // GPU sort: pack all rank GPU buffers into a composite key and use
    // GPU radix sort (block sort for ≤1024, multi-pass radix for larger).
    // Rank buffers stay on GPU throughout — no CPU round-trip.
    {
        if (debug) {
            std::cerr << "[Exec] OrderBy: sortCols.size()=" << sortCols.size()
                      << " rankBufs.size()=" << rankBufs.size() << " n=" << n << "\n";
            for (size_t k = 0; k < rankBufs.size(); ++k) {
                std::cerr << "[Exec] OrderBy: rankBufs[" << k << "] first few = [";
                auto* p = static_cast<const uint32_t*>(rankBufs[k]->contents());
                for (uint32_t i = 0; i < std::min(n, 20u); ++i)
                    std::cerr << p[i] << (i+1<n?",":"");
                std::cerr << "]\n";
            }
        }

        MTL::Buffer* idxBuf = GpuOps::iotaU32(n);

        if (sortCols.size() <= 2) {
            // Pack into u64: primary key in upper 32 bits, secondary in lower 32.
            // Use rank GPU buffers directly — no upload needed
            MTL::Buffer* rank0Buf = rankBufs[0];
            MTL::Buffer* rank1Buf = nullptr;
            if (sortCols.size() > 1) {
                rank1Buf = rankBufs[1];
            } else {
                // Zero-filled secondary key
                rank1Buf = store.device()->newBuffer(n * sizeof(uint32_t), MTL::ResourceStorageModeShared);
                std::memset(rank1Buf->contents(), 0, n * sizeof(uint32_t));
            }
            MTL::Buffer* keyBuf = GpuOps::packU32ToU64(rank0Buf, rank1Buf, n);
            rank0Buf->release();
            if (sortCols.size() <= 1) rank1Buf->release();  // only release zero-fill, not rankBufs[1]
            else rank1Buf->release();  // rankBufs[1] ownership transferred

            GpuOps::radixSortU64(keyBuf, idxBuf, n);
            keyBuf->release();
        } else {
            // 3+ keys: stable LSD radix sort (least-significant-digit first).
            for (int k = (int)sortCols.size() - 1; k >= 0; --k) {
                // Use rank GPU buffer directly — no upload needed
                MTL::Buffer* gatheredRank = GpuOps::gatherU32(rankBufs[k], idxBuf, n, true);
                rankBufs[k]->release();  // ownership transferred
                MTL::Buffer* posBuf = GpuOps::iotaU32(n);
                MTL::Buffer* keyBuf = GpuOps::packU32ToU64(gatheredRank, posBuf, n);
                gatheredRank->release();
                posBuf->release();
                GpuOps::radixSortU64(keyBuf, idxBuf, n);
                keyBuf->release();
            }
        }

        if (debug) {
            std::vector<uint32_t> dbgIdx(std::min(n, 20u));
            std::memcpy(dbgIdx.data(), idxBuf->contents(), dbgIdx.size() * sizeof(uint32_t));
            std::cerr << "[Exec] OrderBy: sortedIdx = [";
            for (uint32_t i = 0; i < (uint32_t)dbgIdx.size(); ++i)
                std::cerr << dbgIdx[i] << (i+1<n?",":"");
            std::cerr << "]\n";
        }

        // --- GPU Gather: reorder u32 and f32 columns on GPU ---
        // Use pre-existing GPU buffers when available (from GroupBy output);
        // otherwise upload from CPU. Dispatch all gathers without sync for max throughput.
        uint32_t totalGatherElements = (uint32_t)(table.u32Cols.size() + table.f32Cols.size()) * n;
        auto gatherStart = std::chrono::high_resolution_clock::now();

        std::vector<MTL::Buffer*> gatheredU32;
        std::vector<MTL::Buffer*> srcU32Bufs;      // track for release (only if we uploaded)
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
            MTL::Buffer* dstBuf = GpuOps::gatherU32(srcBuf, idxBuf, n, /*sync=*/false);
            srcU32Bufs.push_back(ownSrc ? srcBuf : nullptr);
            gatheredU32.push_back(dstBuf);
        }

        std::vector<MTL::Buffer*> gatheredF32;
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
            MTL::Buffer* dstBuf = GpuOps::gatherF32(srcBuf, idxBuf, n, /*sync=*/false);
            srcF32Bufs.push_back(ownSrc ? srcBuf : nullptr);
            gatheredF32.push_back(dstBuf);
        }

        // Single sync point — wait for all gather kernels to complete
        GpuOps::sync();
        auto gatherEnd = std::chrono::high_resolution_clock::now();
        double gatherMs = std::chrono::duration<double, std::milli>(gatherEnd - gatherStart).count();
        KernelTimer::instance().record("orderby_gpu_gather", "sort",
            gatherMs, totalGatherElements);

        // Sync gathered GPU buffers to CPU vectors; keep GPU buffers alive for downstream reuse
        table.u32ColsGPU.resize(table.u32Cols.size());
        for (size_t i = 0; i < table.u32Cols.size(); ++i) {
            std::memcpy(table.u32Cols[i].data(), gatheredU32[i]->contents(), n * sizeof(uint32_t));
            table.u32ColsGPU[i].reset(gatheredU32[i]);  // GpuBuffer takes ownership
            if (srcU32Bufs[i]) srcU32Bufs[i]->release();
        }
        table.f32ColsGPU.resize(table.f32Cols.size());
        for (size_t i = 0; i < table.f32Cols.size(); ++i) {
            std::memcpy(table.f32Cols[i].data(), gatheredF32[i]->contents(), n * sizeof(float));
            table.f32ColsGPU[i].reset(gatheredF32[i]);  // GpuBuffer takes ownership
            if (srcF32Bufs[i]) srcF32Bufs[i]->release();
        }

        // String columns: reorder via GPU dict ID gather, GPU flat string gather,
        // or CPU random-access move (in that priority order)
        if (!table.stringCols.empty()) {
            std::vector<uint32_t> sortedIdx(n);
            std::memcpy(sortedIdx.data(), idxBuf->contents(), n * sizeof(uint32_t));
            for (size_t ci = 0; ci < table.stringCols.size(); ++ci) {
                const std::string& colName = table.stringNames[ci];
                auto dictIt = dictCols.find(colName);
                if (dictIt != dictCols.end() && dictIt->second.idsGPU && dictIt->second.rowCount == n) {
                    // GPU path: gather dict IDs by sorted index, then sequential dictionary lookup
                    MTL::Buffer* gathered = GpuOps::gatherU32(dictIt->second.idsGPU, idxBuf, n);
                    std::vector<uint32_t> gatheredIds(n);
                    std::memcpy(gatheredIds.data(), gathered->contents(), n * sizeof(uint32_t));
                    gathered->release();
                    const auto& dict = dictIt->second.dictionary;
                    auto& col = table.stringCols[ci];
                    for (uint32_t i = 0; i < n; ++i) {
                        col[i] = dict[gatheredIds[i]];
                    }
                    if (debug)
                        std::cerr << "[Exec] OrderBy: GPU dict reorder '" << colName << "'\n";
                } else {
                    // Try GPU flat string gather
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
                            // Release gathered GPU buffers
                            r.chars->release();
                            r.offsets->release();
                            r.lengths->release();
                            if (debug)
                                std::cerr << "[Exec] OrderBy: GPU flat string reorder '" << colName << "'\n";
                            continue;
                        }
                    }
                    // CPU fallback: random-access string move
                    auto& col = table.stringCols[ci];
                    std::vector<std::string> tmp(n);
                    for (uint32_t i = 0; i < n; ++i) tmp[i] = std::move(col[sortedIdx[i]]);
                    col = std::move(tmp);
                }
            }
        }

        idxBuf->release();
    }

    if (debug) {
        std::cerr << "[Exec] OrderBy: GPU sort complete, " << n << " rows sorted\n";
    }
    
    return true;
}

} // namespace engine
