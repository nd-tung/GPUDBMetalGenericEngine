#include "GpuExecutor.hpp"
#include "GpuExecutorDetail.hpp"
#include "Operators.hpp"
#include "GpuColumnStore.hpp"
#include "KernelTimer.hpp"
#include "EngineError.hpp"
#include "Schema.hpp"

#include <algorithm>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <map>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include "Logger.hpp"

namespace engine {

// -- Extracted: postProcessStringKeys --
// Reverses hash/ID to string mapping for string groupby keys.
void postProcessStringKeys(
    const std::vector<std::vector<uint32_t>>& keyVecs,
    const std::vector<std::vector<std::string>>& outputStringMaps,
    const std::vector<std::unordered_map<uint32_t, std::string>>& hashToStringMaps,
    const std::vector<std::shared_ptr<std::vector<std::string>>>& dictRefMaps,
    TableResult& out, bool /*debug*/) {
    // Post-process string columns
    for (size_t k = 0; k < keyVecs.size(); ++k) {
        // Fast path: direct dict array indexing (0-based, no hash lookup)
        if (k < dictRefMaps.size() && dictRefMaps[k]) {
            const auto& dict = *dictRefMaps[k];
            std::vector<std::string> strCol;
            strCol.reserve(out.rowCount);
            for (uint32_t val : out.u32Cols[k]) {
                strCol.push_back(val < dict.size() ? dict[val] : "");
            }
            out.stringCols.push_back(std::move(strCol));
            out.stringNames.push_back(out.u32Names[k]);
        } else if (k < hashToStringMaps.size() && !hashToStringMaps[k].empty()) {
            // Use hash lookup
            std::vector<std::string> strCol;
            strCol.reserve(out.rowCount);
            const auto& hashMap = hashToStringMaps[k];

            LOG_DEBUG("Exec", "GroupBy: Post-proc string col " << k  << " via hash lookup, hashMap.size=" << hashMap.size());

            for (uint32_t hashVal : out.u32Cols[k]) {
                auto it = hashMap.find(hashVal);
                if (it != hashMap.end()) {
                    strCol.push_back(it->second);
                } else {
                    strCol.push_back("");
                }
            }
            LOG_DEBUG("Exec", "GroupBy: Built strCol with " << strCol.size() << " strings via hash lookup\n");
            out.stringCols.push_back(std::move(strCol));
            out.stringNames.push_back(out.u32Names[k]);
        } else if (!outputStringMaps[k].empty()) {
            // Convert IDs back to strings (1-based index)
            std::vector<std::string> strCol;
            strCol.reserve(out.rowCount);
            const auto& map = outputStringMaps[k];

            LOG_DEBUG("Exec", "GroupBy: Post-proc string col " << k  << " u32_cols[k].size=" << out.u32Cols[k].size()  << " map.size=" << map.size());

            for (uint32_t val : out.u32Cols[k]) {
                if (val > 0 && (val - 1) < map.size()) {
                    strCol.push_back(map[val - 1]);
                } else {
                    strCol.push_back(""); 
                }
            }
            LOG_DEBUG("Exec", "GroupBy: Built strCol with " << strCol.size() << " strings\n");
            out.stringCols.push_back(std::move(strCol));
            out.stringNames.push_back(out.u32Names[k]);
        }
    }

}

// -- Extracted: restoreF32Keys --
// Bitcasts u32 groupby keys back to f32 where originally f32.
void restoreF32Keys(
    const std::vector<bool>& keyFromF32,
    TableResult& out, bool /*debug*/) {
    // Restore f32 keys that were bit-reinterpreted to u32
    for (size_t k = 0; k < keyFromF32.size(); ++k) {
        if (k < keyFromF32.size() && keyFromF32[k]) {
            // Convert u32 bits back to float and move to f32_cols
            std::vector<float> restored(out.u32Cols[k].size());
            for (size_t j = 0; j < restored.size(); ++j) {
                std::memcpy(&restored[j], &out.u32Cols[k][j], sizeof(float));
            }
            LOG_DEBUG("Exec", "GroupBy: restoring f32 key " << out.u32Names[k]  << " (" << restored.size() << " values)\n");
            // Create GPU buffer for the restored f32 key (same bits as u32)
            // GpuBuffer copy auto-retains (shared ownership with u32ColsGPU[k])
            // Add to f32 output (prepend before aggregates)
            out.f32Names.insert(out.f32Names.begin(), out.u32Names[k]);
            out.f32Cols.insert(out.f32Cols.begin(), std::move(restored));
            out.f32ColsGPU.insert(out.f32ColsGPU.begin(), out.u32ColsGPU[k]);
            // Mark u32 slot as converted (will be handled in order building)
        }
    }

}

// -- Extracted: buildGroupByOutputOrder --
// Sets column ordering and marks single-char columns.
void buildGroupByOutputOrder(
    const std::vector<std::vector<uint32_t>>& keyVecs,
    const std::vector<bool>& keyFromF32,
    const std::vector<std::vector<std::string>>& outputStringMaps,
    const std::vector<std::unordered_map<uint32_t, std::string>>& hashToStringMaps,
    const std::vector<std::shared_ptr<std::vector<std::string>>>& dictRefMaps,
    TableResult& out) {
    // Build output order - check if any string column was produced
    out.order.clear();
    size_t strIdx = 0;
    // Count how many f32-restored keys were prepended (they shift agg f32 indices)
    size_t f32KeyCount = 0;
    for (size_t k = 0; k < keyVecs.size(); ++k) {
        if (k < keyFromF32.size() && keyFromF32[k]) f32KeyCount++;
    }
    for (size_t i = 0; i < out.u32Names.size(); ++i) {
        bool hasStrings = (!outputStringMaps[i].empty()) || 
                          (i < hashToStringMaps.size() && !hashToStringMaps[i].empty()) ||
                          (i < dictRefMaps.size() && dictRefMaps[i]);
        bool wasF32 = (i < keyFromF32.size() && keyFromF32[i]);
        if (hasStrings) {
            out.order.push_back({TableResult::ColRef::Kind::String, strIdx++, out.u32Names[i]});
        } else if (wasF32) {
            // Find the f32 index for this key (prepended before aggregates)
            size_t f32Idx = 0;
            for (size_t fi = 0; fi < out.f32Names.size(); ++fi) {
                if (out.f32Names[fi] == out.u32Names[i]) { f32Idx = fi; break; }
            }
            out.order.push_back({TableResult::ColRef::Kind::F32, f32Idx, out.u32Names[i]});
        } else {
            out.order.push_back({TableResult::ColRef::Kind::U32, i, out.u32Names[i]});
        }
    }
    for (size_t i = f32KeyCount; i < out.f32Names.size(); ++i) {
        out.order.push_back({TableResult::ColRef::Kind::F32, i, out.f32Names[i]});
    }

    // Mark single-char columns
    const auto& schema = SchemaRegistry::instance();
    for (const auto& name : out.u32Names) {
        std::string table = tableForColumn(name);
        if (schema.isSingleCharColumn(table, name)) {
            out.singleCharCols.insert(name);
        }
    }

}

// -- Extracted: buildDictIdKey --
// Dictionary ID path — collision-free, no hashing needed.
static bool buildDictIdKey(EvalContext& ctx, const std::string& col,
                           const std::string& keyName, size_t expectedKeyRows,
                           GroupByKeyData& kd, bool /*debug*/) {
    if (!ctx.dictCols.count(col) || ctx.dictCols.at(col).dictionary.empty())
        return false;

    auto& dict = ctx.dictCols[col];
    ctx.ensureActiveRowsCPU();

    std::vector<uint32_t> ids;
    GpuBuffer idsBufOwner;
    if (!ctx.activeRows.empty() && dict.ids.size() != expectedKeyRows) {
        if (dict.idsGPU && ctx.activeRowsGPU) {
            GpuBuffer gathered = GpuOps::gatherU32(dict.idsGPU, ctx.activeRowsGPU, ctx.activeRowsCountGPU, false);
            // Skip CPU download — GPU buffer passed directly to kernel
            idsBufOwner = std::move(gathered);
        } else {
            dict.ensureIdsCPU();
            ids.reserve(expectedKeyRows);
            for (uint32_t r : ctx.activeRows) {
                ids.push_back(r < dict.ids.size() ? dict.ids[r] : 0);
            }
        }
    } else {
        if (dict.idsGPU && dict.ids.size() != expectedKeyRows) {
            // Skip CPU download — GPU buffer passed directly to kernel
            idsBufOwner = dict.idsGPU; // copy retains
        } else {
            dict.ensureIdsCPU();
            ids = dict.ids;
            if (ids.size() > expectedKeyRows) ids.resize(expectedKeyRows);
        }
    }

    kd.keyVecs.push_back(std::move(ids));
    kd.keyBufsGPU.push_back(std::move(idsBufOwner));
    kd.keyNames.push_back(keyName.empty() ? col : keyName);
    kd.keyFromF32.push_back(false);
    kd.outputStringMaps.push_back({});
    kd.hashToStringMaps.push_back({});
    kd.dictRefMaps.push_back(dict.dictionary.p);
    LOG_DEBUG("Exec", "GroupBy: Dict ID key for " << col << " (" << dict.dictionary.size() << " unique, collision-free)\n");
    return true;
}

// -- Extracted: buildStringHashKey --
// GPU FNV1a hash with CPU collision check, or CPU sequential ID fallback.
static bool buildStringHashKey(EvalContext& ctx, const std::string& col,
                               const std::string& keyName, size_t expectedKeyRows,
                               GroupByKeyData& kd, bool /*debug*/) {
    // Locate flatStringCols key early to avoid expensive ensureStringCol
    std::string flatKey = col;
    if (!ctx.flatStringCols.count(flatKey)) {
        for (int sfx = 1; sfx <= 9; ++sfx) {
            std::string sfxKey = col + "_" + std::to_string(sfx);
            if (ctx.flatStringCols.count(sfxKey)) { flatKey = sfxKey; break; }
        }
    }
    bool hasFlat = ctx.flatStringCols.count(flatKey) > 0;

    if (!hasFlat && !ctx.stringCols.count(col) && !ctx.hasDictCol(col))
        return false;

    // If u32 data exists with correct row count, let the u32 path handle it
    // (avoids expensive CPU sequential encoding; u32 values serve as direct keys)
    {
        auto u32It = ctx.u32Cols.find(col);
        if (u32It != ctx.u32Cols.end() && u32It->second.size() == expectedKeyRows)
            return false;
        if (ctx.u32ColsGPU.count(col) && ctx.activeRowsGPU)
            return false;
    }

    // Get source row count WITHOUT materializing strings when flat exists
    uint32_t sourceRowCount = 0;
    if (hasFlat) {
        sourceRowCount = ctx.flatStringCols[flatKey].rowCount;
    } else {
        ctx.ensureStringCol(col);
        if (!ctx.stringCols.count(col) || ctx.stringCols.at(col).empty())
            return false;
        sourceRowCount = (uint32_t)ctx.stringCols.at(col).size();
    }

    ctx.ensureActiveRowsCPU();

    if (sourceRowCount != expectedKeyRows && ctx.activeRows.empty())
        return false;

    // --- GPU FNV1a hash path ---
    bool gpuHashOk = false;

    MTL::Buffer* hashBuf = nullptr;
    if (hasFlat) {
        auto& flat = ctx.flatStringCols[flatKey];
        GpuBuffer gOff, gLen;
        if (!ctx.activeRows.empty() && sourceRowCount != expectedKeyRows && ctx.activeRowsGPU) {
            gOff = GpuOps::gatherU32(flat.offsets, ctx.activeRowsGPU, ctx.activeRowsCountGPU, false);
            gLen = GpuOps::gatherU32(flat.lengths, ctx.activeRowsGPU, ctx.activeRowsCountGPU, false);
            GpuOps::sync();
            hashBuf = GpuOps::stringFnv1aU64Fold32(flat.chars, gOff, gLen, expectedKeyRows).detach();
        } else {
            hashBuf = GpuOps::stringFnv1aU64Fold32(flat.chars, flat.offsets, flat.lengths, expectedKeyRows).detach();
        }

        // Retain pointers to offsets/lengths for zero-copy string reads below
        if (hashBuf) {
            MTL::Buffer* effOff = gOff ? gOff.get() : flat.offsets.get();
            MTL::Buffer* effLen = gLen ? gLen.get() : flat.lengths.get();
            GpuOps::sync(); // ensure hash kernel completes before CPU read
            const uint32_t* hashes = static_cast<const uint32_t*>(hashBuf->contents());
            const char* chars = static_cast<const char*>(flat.chars->contents());
            const uint32_t* offs = static_cast<const uint32_t*>(effOff->contents());
            const uint32_t* lens = static_cast<const uint32_t*>(effLen->contents());

            // Build hash→string map for output reconstruction (first occurrence wins)
            std::unordered_map<uint32_t, std::string> hashMap;
            hashMap.reserve(256);
            for (uint32_t i = 0; i < expectedKeyRows; ++i) {
                uint32_t h = hashes[i];
                if (hashMap.find(h) == hashMap.end()) {
                    hashMap.emplace(h, std::string(chars + offs[i], lens[i]));
                }
            }

            gpuHashOk = true;
            kd.keyVecs.push_back({});
            kd.keyBufsGPU.push_back(GpuBuffer(hashBuf));
            kd.keyNames.push_back(keyName.empty() ? col : keyName);
            kd.keyFromF32.push_back(false);
            kd.outputStringMaps.push_back({});
            kd.hashToStringMaps.push_back(std::move(hashMap));
            kd.dictRefMaps.push_back(nullptr);
            LOG_DEBUG("Exec", "GroupBy: GPU FNV1a-u64fold32 encoded string key " << col << " (" << kd.hashToStringMaps.back().size() << " unique)\n");
        }
    } else {
        LOG_DEBUG("Exec", "GroupBy: WARN no flatStringCols for " << col << ", skipping GPU hash\n");
    }

    // GPU hash path covers all cases — no CPU fallback needed
    if (!gpuHashOk) {
        LOG_DEBUG("Exec", "GroupBy: no flat buffers for string key " << col << ", cannot encode\n");
        return false;
    }
    return true;
}

// -- Extracted: buildGroupByF32Key --
// Builds a group-by key from an f32 column via bitcast to u32.
static bool buildGroupByF32Key(EvalContext& ctx, const std::string& col,
                               const std::string& keyName, size_t expectedKeyRows,
                               GroupByKeyData& kd, bool /*debug*/) {
    // GPU fast path: if f32 is on GPU, bitcast directly without downloading
    if (ctx.f32ColsGPU.count(col)) {
        MTL::Buffer* gpuF32 = ctx.f32ColsGPU.at(col);
        uint32_t gpuCount = (uint32_t)(gpuF32->length() / sizeof(float));
        if (gpuCount > 0) {
            MTL::Buffer* gpuU32 = nullptr;
            if (ctx.activeRowsGPU && ctx.activeRowsCountGPU > 0 && gpuCount != (uint32_t)expectedKeyRows) {
                GpuBuffer gathered = GpuOps::gatherF32(gpuF32, ctx.activeRowsGPU, ctx.activeRowsCountGPU, false);
                gpuU32 = GpuOps::bitcastF32ToU32(gathered, ctx.activeRowsCountGPU).detach();
                gpuCount = ctx.activeRowsCountGPU;
            } else {
                gpuU32 = GpuOps::bitcastF32ToU32(gpuF32, gpuCount).detach();
            }
            // Skip CPU download — GPU buffer passed directly to kernel
            LOG_DEBUG("Exec", "GroupBy: GPU bitcast f32 key " << col << " to u32 (" << gpuCount << " rows)\n");
            kd.keyVecs.push_back({});
            kd.keyBufsGPU.push_back(GpuBuffer(gpuU32));
            kd.keyNames.push_back(keyName.empty() ? col : keyName);
            kd.keyFromF32.push_back(true);
            kd.outputStringMaps.push_back({});
            kd.hashToStringMaps.push_back({});
            kd.dictRefMaps.push_back(nullptr);
            return true;
        }
    }

    // CPU fallback: download f32, bitcast element-by-element
    if ((ctx.f32Cols.find(col) == ctx.f32Cols.end() || ctx.f32Cols[col].empty()) && ctx.f32ColsGPU.count(col)) {
        MTL::Buffer* buf = ctx.f32ColsGPU.at(col);
        size_t count = buf->length() / sizeof(float);
        if (count > 0) {
            std::vector<float> down(count);
            std::memcpy(down.data(), buf->contents(), count * sizeof(float));
            ctx.f32Cols[col] = std::move(down);
            LOG_DEBUG("Exec", "GroupBy: Lazy fetch F32 key " << col << " from GPU\n");
        }
    }

    auto itF = ctx.f32Cols.find(col);
    if (itF == ctx.f32Cols.end()) {
        for (int suffix = 1; suffix <= 9 && itF == ctx.f32Cols.end(); ++suffix) {
            std::string suffixedCol = col + "_" + std::to_string(suffix);
            itF = ctx.f32Cols.find(suffixedCol);
        }
    }
    if (itF != ctx.f32Cols.end()) {
        std::vector<uint32_t> converted;
        converted.reserve(expectedKeyRows);
        if (!ctx.activeRows.empty() && itF->second.size() != expectedKeyRows) {
            for(uint32_t r : ctx.activeRows) {
                if (r < itF->second.size()) {
                    uint32_t bits; std::memcpy(&bits, &itF->second[r], sizeof(bits));
                    converted.push_back(bits);
                }
                else converted.push_back(0);
            }
        } else {
            for (float f : itF->second) {
                uint32_t bits; std::memcpy(&bits, &f, sizeof(bits));
                converted.push_back(bits);
            }
        }
        LOG_DEBUG("Exec", "GroupBy: converted f32 key " << col << " to u32\n");
        kd.keyVecs.push_back(std::move(converted));
        kd.keyBufsGPU.emplace_back();
        kd.keyNames.push_back(keyName.empty() ? col : keyName);
        kd.keyFromF32.push_back(true);
        kd.outputStringMaps.push_back({});
        kd.hashToStringMaps.push_back({});
        kd.dictRefMaps.push_back(nullptr);
        return true;
    }
    return false;
}

// -- Extracted: buildGroupByKeys --
// Builds key vectors from ctx columns for each IRGroupBy key expression.
// Handles dict ID path, GPU FNV1a hash path, CPU fallback, positional references,
// and f32→u32 bitcast for float groupby keys.
GroupByKeyData buildGroupByKeys(
    const IRGroupBy& groupBy, EvalContext& ctx,
    size_t expectedKeyRows, bool debug)
{
    GroupByKeyData kd;

    // Zero rows: return empty key vectors (one per key) so caller
    // can produce a valid 0-row result instead of failing.
    if (expectedKeyRows == 0) {
        for (size_t i = 0; i < groupBy.keys.size(); ++i) {
            std::string keyName = i < groupBy.keyNames.size() ? groupBy.keyNames[i] : "";
            kd.keyVecs.push_back({});
            kd.keyBufsGPU.emplace_back();
            kd.keyNames.push_back(keyName);
            kd.keyFromF32.push_back(false);
            kd.outputStringMaps.push_back({});
            kd.hashToStringMaps.push_back({});
            kd.dictRefMaps.push_back(nullptr);
        }
        return kd;
    }
    
    for (size_t i = 0; i < groupBy.keys.size(); ++i) {
        const auto& keyExpr = groupBy.keys[i];
        std::string keyName = i < groupBy.keyNames.size() ? groupBy.keyNames[i] : "";
        
        if (keyExpr && keyExpr->kind == TypedExpr::Kind::Column) {
            const std::string& col = keyExpr->asColumn().column;

            // LAZY FETCH: If vector is empty but on GPU, bring it back
            // Skip if dictCols exist — buildDictIdKey uses dict IDs directly
            if (!ctx.dictCols.count(col) &&
                (ctx.u32Cols.find(col) == ctx.u32Cols.end() || ctx.u32Cols[col].empty()) && ctx.u32ColsGPU.count(col)) {
                 MTL::Buffer* buf = ctx.u32ColsGPU.at(col);
                 size_t count = buf->length() / sizeof(uint32_t);
                 if (count > 0) {
                     std::vector<uint32_t> down(count);
                     std::memcpy(down.data(), buf->contents(), count * sizeof(uint32_t));
                     ctx.u32Cols[col] = std::move(down);
                     LOG_DEBUG("Exec", "GroupBy: Lazy fetch key " << col << " from GPU (" << count << " rows)\n");
                 }
            }
            
            // ── Resolve u32 column iterator ──
            auto it = ctx.u32Cols.find(col);
            
            // Prefer column with matching row count (in case of duplicates with different sizes)
            if (it != ctx.u32Cols.end() && it->second.size() != expectedKeyRows) {
                if (debug) {
                    LOG_INFO("Exec", "GroupBy: key " << col << " has wrong size (" << it->second.size()  << " vs expected " << expectedKeyRows << "), looking for positional ref\n");
                }
                auto origIt = it;
                it = ctx.u32Cols.end();
                
                for (size_t pos = 0; pos < 20 && it == ctx.u32Cols.end(); ++pos) {
                    std::string posKey = "#" + std::to_string(pos);
                    auto posIt = ctx.u32Cols.find(posKey);
                    if (posIt != ctx.u32Cols.end() && posIt->second.size() == expectedKeyRows) {
                        const auto& origData = origIt->second;
                        const auto& posData = posIt->second;
                        if (!origData.empty() && !posData.empty()) {
                            size_t firstIdx = 0;
                            ctx.ensureActiveRowsCPU();
                            if (!ctx.activeRows.empty()) firstIdx = ctx.activeRows[0];
                            
                            if (firstIdx < origData.size() && origData[firstIdx] == posData[0]) {
                                bool matchConfirmed = true;
                                if (posData.size() > 1) {
                                  size_t lastIdx = posData.size() - 1;
                                  size_t origLastIdx = lastIdx;
                                  if (!ctx.activeRows.empty() && lastIdx < ctx.activeRows.size())
                                      origLastIdx = ctx.activeRows[lastIdx];
                                  if (origLastIdx < origData.size() && origData[origLastIdx] != posData[lastIdx])
                                      matchConfirmed = false;
                                }
                                if (matchConfirmed) {
                                    it = posIt;
                                    LOG_DEBUG("Exec", "GroupBy: using positional " << posKey  << " for key " << col << " (matched via value sampling)\n");
                                }
                            }
                        }
                    }
                }
                if (it == ctx.u32Cols.end()) it = origIt;
            }
            
            // Positional reference (#N) lookup
            if (it == ctx.u32Cols.end() && col.size() >= 2 && col[0] == '#') {
                try {
                    size_t pos = std::stoul(col.substr(1));
                    size_t idx = 0;
                    for (auto& [name, data] : ctx.u32Cols) {
                        if (idx == pos) {
                            it = ctx.u32Cols.find(name);
                            if (keyName.empty()) keyName = name;
                            LOG_DEBUG("Exec", "GroupBy: resolved positional " << col << " to " << name);
                            break;
                        }
                        idx++;
                    }
                } catch (...) {
                    LOG_DEBUG("Exec", "GroupBy: positional col parse failed for '" << col << "'\n");
                }
            }
            
            // Try keyName as fallback
            if (it == ctx.u32Cols.end() && !keyName.empty() && keyName != col) {
                it = ctx.u32Cols.find(keyName);
                if (debug && it != ctx.u32Cols.end())
                    LOG_INFO("Exec", "GroupBy: found key using keyName " << keyName);
            }
            
            // Try suffixed versions for multi-instance tables
            if (it == ctx.u32Cols.end()) {
                for (int suffix = 1; suffix <= 9 && it == ctx.u32Cols.end(); ++suffix)
                    it = ctx.u32Cols.find(col + "_" + std::to_string(suffix));
            }
            
            // ── Dispatch to key-type-specific builders ──
            if (buildDictIdKey(ctx, col, keyName, expectedKeyRows, kd, debug))
                continue;
            if (buildStringHashKey(ctx, col, keyName, expectedKeyRows, kd, debug))
                continue;
            
            // Numeric key paths
            if (it != ctx.u32Cols.end()) {
                // ── U32 key ──
                if (!ctx.activeRows.empty() && it->second.size() != expectedKeyRows) {
                    if (ctx.activeRowsGPU && ctx.u32ColsGPU.count(col) && ctx.u32ColsGPU[col]) {
                        GpuBuffer gathered = GpuOps::gatherU32(ctx.u32ColsGPU[col], ctx.activeRowsGPU, (uint32_t)expectedKeyRows, false);
                        // Skip CPU download — GPU buffer passed directly to kernel
                        kd.keyVecs.push_back({});
                        kd.keyBufsGPU.push_back(std::move(gathered));
                    } else {
                        std::vector<uint32_t> filtered;
                        filtered.reserve(expectedKeyRows);
                        for (uint32_t r : ctx.activeRows) {
                            if (r < it->second.size()) filtered.push_back(it->second[r]);
                            else filtered.push_back(0);
                        }
                        kd.keyVecs.push_back(std::move(filtered));
                        kd.keyBufsGPU.emplace_back();
                    }
                } else {
                    kd.keyVecs.push_back(it->second);
                    if (ctx.u32ColsGPU.count(col) && ctx.u32ColsGPU[col]) {
                        size_t gpuElems = ctx.u32ColsGPU[col]->length() / sizeof(uint32_t);
                        if (gpuElems == it->second.size()) {
                            kd.keyBufsGPU.push_back(ctx.u32ColsGPU[col]); // copy retains
                        } else {
                            LOG_DEBUG("Exec", "GroupBy: SKIP stale GPU buf for " << col << " (gpu=" << gpuElems << " vs cpu=" << it->second.size() << ")\n");
                            kd.keyBufsGPU.emplace_back();
                        }
                    } else {
                        kd.keyBufsGPU.emplace_back();
                    }
                }
                kd.keyNames.push_back(keyName.empty() ? col : keyName);
                kd.keyFromF32.push_back(false);
                
                // Build hash->string map for string output
                ctx.ensureStringCol(col);
                if (ctx.stringCols.count(col)) {
                    const auto& strData = ctx.stringCols.at(col);
                    std::unordered_map<uint32_t, std::string> hashToStr;
                    const auto& u32Data = it->second;
                    LOG_DEBUG("Exec", "GroupBy: building hash->string map for " << col  << " u32Data.size=" << u32Data.size()  << " strData.size=" << strData.size());
                    for (size_t r = 0; r < std::min(u32Data.size(), strData.size()); ++r) {
                        uint32_t hash = u32Data[r];
                        if (hashToStr.find(hash) == hashToStr.end())
                            hashToStr[hash] = strData[r];
                    }
                    LOG_DEBUG("Exec", "GroupBy: built hash->string map with " << hashToStr.size() << " entries\n");
                    kd.hashToStringMaps.push_back(std::move(hashToStr));
                    kd.outputStringMaps.push_back({});
                    kd.dictRefMaps.push_back(nullptr);
                } else {
                    kd.hashToStringMaps.push_back({});
                    kd.outputStringMaps.push_back({});
                    kd.dictRefMaps.push_back(nullptr);
                }
            } else {
                // ── F32 key via bitcast ──
                buildGroupByF32Key(ctx, col, keyName, expectedKeyRows, kd, debug);
            }
        }
    }
    
    // dictRefMaps is populated inline in each push site (dict path pushes dict ptr,
    // all other paths push nullptr). No padding needed.
    
    return kd;
}

} // namespace engine
