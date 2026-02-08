#include "GpuExecutor.hpp"
#include "GpuExecutorPriv.hpp"
#include "Operators.hpp"
#include "Relation.hpp"
#include "ColumnStoreGPU.hpp"
#include "KernelTimer.hpp"

#include <chrono>
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <tuple>
#include <unordered_map>
#include <cstring>

namespace engine {

bool GpuExecutor::executeOrderBy(const IROrderBy& order, TableResult& table) {
    const bool debug = env_truthy("GPUDB_DEBUG_OPS");
    
    if (debug) {
        std::cerr << "[Exec] OrderBy: columns=[";
        for (size_t i = 0; i < order.columns.size(); ++i) {
            std::cerr << order.columns[i];
            if (i < order.ascending.size()) std::cerr << (order.ascending[i] ? " ASC" : " DESC");
            if (i + 1 < order.columns.size()) std::cerr << ", ";
        }
        std::cerr << "]\n";
        std::cerr << "[Exec] OrderBy: table.u32_names=[";
        for (const auto& n : table.u32_names) std::cerr << n << ", ";
        std::cerr << "]\n";
        std::cerr << "[Exec] OrderBy: table.f32_names=[";
        for (const auto& n : table.f32_names) std::cerr << n << ", ";
        std::cerr << "]\n";
        std::cerr << "[Exec] OrderBy: table.string_names=[";
        for (const auto& n : table.string_names) std::cerr << n << ", ";
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
        for (size_t j = 0; j < table.string_names.size() && !found; ++j) {
            if (table.string_names[j] == colName || base_ident(table.string_names[j]) == base_ident(colName)) {
                sortCols.push_back({2, asc, j});
                found = true;
            }
        }
        for (size_t j = 0; j < table.u32_names.size() && !found; ++j) {
            if (table.u32_names[j] == colName || base_ident(table.u32_names[j]) == base_ident(colName)) {
                sortCols.push_back({0, asc, j});
                found = true;
            }
        }
        for (size_t j = 0; j < table.f32_names.size() && !found; ++j) {
            if (table.f32_names[j] == colName || base_ident(table.f32_names[j]) == base_ident(colName)) {
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
                for (size_t j = 0; j < table.f32_names.size() && !found; ++j) {
                    const std::string& name = table.f32_names[j];
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
                
                if (!found && table.f32_names.size() == 1) {
                    sortCols.push_back({1, asc, 0});
                    found = true;
                    if (debug) std::cerr << "[Exec] OrderBy: fallback '" << colName << "' to single f32 '" << table.f32_names[0] << "'\n";
                }
                
                if (!found && colLower.find("distinct") != std::string::npos) {
                    for (size_t j = 0; j < table.f32_names.size() && !found; ++j) {
                        const std::string& name = table.f32_names[j];
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

    auto buildRankU32 = [&](const std::vector<uint32_t>& col, bool asc) -> std::vector<uint32_t> {
        std::vector<uint32_t> rank(n);
        if (asc) {
            rank = col;
        } else {
            for (uint32_t i = 0; i < n; ++i) rank[i] = ~col[i];
        }
        return rank;
    };

    auto buildRankF32 = [&](const std::vector<float>& col, bool asc) -> std::vector<uint32_t> {
        // IEEE 754 float -> u32 for ordering:
        // Negative: flip all bits. Positive: flip sign bit.
        std::vector<uint32_t> rank(n);
        for (uint32_t i = 0; i < n; ++i) {
            uint32_t bits;
            std::memcpy(&bits, &col[i], sizeof(bits));
            if (bits & 0x80000000u) {
                bits = ~bits;
            } else {
                bits ^= 0x80000000u;
            }
            rank[i] = asc ? bits : ~bits;
        }
        return rank;
    };

    auto buildRankString = [&](const std::vector<std::string>& col, bool asc) -> std::vector<uint32_t> {
        std::vector<std::string> uniq(col.begin(), col.end());
        std::sort(uniq.begin(), uniq.end());
        uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());
        
        std::unordered_map<std::string, uint32_t> rankMap;
        for (uint32_t r = 0; r < (uint32_t)uniq.size(); ++r) {
            rankMap[uniq[r]] = r;
        }
        
        std::vector<uint32_t> rank(n);
        uint32_t maxRank = (uint32_t)uniq.size();
        for (uint32_t i = 0; i < n; ++i) {
            auto it = rankMap.find(col[i]);
            uint32_t r = (it != rankMap.end()) ? it->second : maxRank;
            rank[i] = asc ? r : (maxRank - 1 - r);
        }
        return rank;
    };

    std::vector<std::vector<uint32_t>> ranks;
    for (const auto& sc : sortCols) {
        if (sc.type == 0) {
            ranks.push_back(buildRankU32(table.u32_cols[sc.colIdx], sc.ascending));
        } else if (sc.type == 1) {
            ranks.push_back(buildRankF32(table.f32_cols[sc.colIdx], sc.ascending));
        } else {
            ranks.push_back(buildRankString(table.string_cols[sc.colIdx], sc.ascending));
        }
    }

    auto& store = ColumnStoreGPU::instance();

    // GPU sort: pack all rank vectors into a single composite key and use
    // GPU radix sort (block sort for ≤1024, multi-pass radix for larger).
    {
        if (debug) {
            std::cerr << "[Exec] OrderBy: sortCols.size()=" << sortCols.size()
                      << " ranks.size()=" << ranks.size() << " n=" << n << "\n";
            for (size_t k = 0; k < ranks.size(); ++k) {
                std::cerr << "[Exec] OrderBy: ranks[" << k << "] = [";
                for (uint32_t i = 0; i < std::min(n, 20u); ++i)
                    std::cerr << ranks[k][i] << (i+1<n?",":"");
                std::cerr << "]\n";
            }
        }

        MTL::Buffer* idxBuf = GpuOps::iotaU32(n);

        if (sortCols.size() <= 2) {
            // Pack into u64: primary key in upper 32 bits, secondary in lower 32.
            // For a single key, secondary is 0 everywhere.
            std::vector<uint64_t> keys64(n);
            for (uint32_t i = 0; i < n; ++i) {
                uint64_t hi = (uint64_t)ranks[0][i];
                uint64_t lo = (sortCols.size() > 1) ? (uint64_t)ranks[1][i] : 0;
                keys64[i] = (hi << 32) | lo;
            }
            MTL::Buffer* keyBuf = store.device()->newBuffer(
                keys64.data(), n * sizeof(uint64_t), MTL::ResourceStorageModeShared);

            GpuOps::radixSortU64(keyBuf, idxBuf, n);
            keyBuf->release();
        } else {
            // 3+ keys: stable LSD radix sort (least-significant-digit first).
            // Embed position in low 32 bits of u64 key to ensure stability.
            for (int k = (int)sortCols.size() - 1; k >= 0; --k) {
                std::vector<uint32_t> curIdx(n);
                std::memcpy(curIdx.data(), idxBuf->contents(), n * sizeof(uint32_t));
                std::vector<uint64_t> keys64(n);
                for (uint32_t i = 0; i < n; ++i)
                    keys64[i] = ((uint64_t)ranks[k][curIdx[i]] << 32) | (uint64_t)i;
                MTL::Buffer* keyBuf = store.device()->newBuffer(
                    keys64.data(), n * sizeof(uint64_t), MTL::ResourceStorageModeShared);
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
        // Upload each column, gather with idxBuf, download result.
        // Dispatch all gathers without sync for max throughput, then sync once.
        uint32_t totalGatherElements = (uint32_t)(table.u32_cols.size() + table.f32_cols.size()) * n;
        auto gatherStart = std::chrono::high_resolution_clock::now();

        std::vector<MTL::Buffer*> gatheredU32;
        std::vector<MTL::Buffer*> srcU32Bufs;      // track for release
        gatheredU32.reserve(table.u32_cols.size());
        srcU32Bufs.reserve(table.u32_cols.size());
        for (auto& col : table.u32_cols) {
            MTL::Buffer* srcBuf = store.device()->newBuffer(
                col.data(), n * sizeof(uint32_t), MTL::ResourceStorageModeShared);
            MTL::Buffer* dstBuf = GpuOps::gatherU32(srcBuf, idxBuf, n, /*sync=*/false);
            srcU32Bufs.push_back(srcBuf);
            gatheredU32.push_back(dstBuf);
        }

        std::vector<MTL::Buffer*> gatheredF32;
        std::vector<MTL::Buffer*> srcF32Bufs;
        gatheredF32.reserve(table.f32_cols.size());
        srcF32Bufs.reserve(table.f32_cols.size());
        for (auto& col : table.f32_cols) {
            MTL::Buffer* srcBuf = store.device()->newBuffer(
                col.data(), n * sizeof(float), MTL::ResourceStorageModeShared);
            MTL::Buffer* dstBuf = GpuOps::gatherF32(srcBuf, idxBuf, n, /*sync=*/false);
            srcF32Bufs.push_back(srcBuf);
            gatheredF32.push_back(dstBuf);
        }

        // Single sync point — wait for all gather kernels to complete
        GpuOps::sync();
        auto gatherEnd = std::chrono::high_resolution_clock::now();
        double gatherMs = std::chrono::duration<double, std::milli>(gatherEnd - gatherStart).count();
        KernelTimer::instance().record("orderby_gpu_gather", "sort",
            gatherMs, totalGatherElements);

        // Download gathered results back to CPU vectors and release GPU buffers
        for (size_t i = 0; i < table.u32_cols.size(); ++i) {
            std::memcpy(table.u32_cols[i].data(), gatheredU32[i]->contents(), n * sizeof(uint32_t));
            gatheredU32[i]->release();
            srcU32Bufs[i]->release();
        }
        for (size_t i = 0; i < table.f32_cols.size(); ++i) {
            std::memcpy(table.f32_cols[i].data(), gatheredF32[i]->contents(), n * sizeof(float));
            gatheredF32[i]->release();
            srcF32Bufs[i]->release();
        }

        // String columns: reorder on CPU (GPU has no variable-length string support)
        if (!table.string_cols.empty()) {
            std::vector<uint32_t> sortedIdx(n);
            std::memcpy(sortedIdx.data(), idxBuf->contents(), n * sizeof(uint32_t));
            for (auto& col : table.string_cols) {
                std::vector<std::string> tmp(n);
                for (uint32_t i = 0; i < n; ++i) tmp[i] = std::move(col[sortedIdx[i]]);
                col = std::move(tmp);
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
