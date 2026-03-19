#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "Relation.hpp"

namespace engine {

// GPU-side filter comparison operator. Values map directly to GPU kernel parameters.
// This enum is separate from CompareOp (planner-level) because it only covers
// operations that the GPU filter kernels support, plus LIKE_PATTERN for string matching.
enum class GpuFilterOp { LT, LE, GT, GE, EQ, NE, LIKE_PATTERN = 999 };

struct FilterResult {
    GpuBuffer indices;    // u32 indices, length == count
    uint32_t count = 0;
};

struct GroupByHashTable {
    GpuBuffer htKeys;   // capacity * 4 u32
    GpuBuffer htAggs;   // capacity * 8 u32 (float bits)
    uint32_t capacity = 0;
};

// Result of GPU stream compaction on a GroupBy hash table.
struct GroupByExtractResult {
    std::vector<std::vector<uint32_t>> keyCols;  // [keyIdx][row], bias removed
    std::vector<std::vector<uint32_t>> aggWords; // [aggSlot][row], raw u32
    // GPU buffers for each key/agg column (per-column slices of the SoA output).
    // These stay alive so downstream can avoid re-uploading CPU vectors to GPU.
    std::vector<GpuBuffer> keyColsGPU;           // [keyIdx] -> GpuBuffer
    std::vector<GpuBuffer> aggColsGPU;           // [aggSlot] -> GpuBuffer
    uint32_t rowCount = 0;
};


struct JoinResult {
    GpuBuffer buildIndices;
    GpuBuffer probeIndices;
    uint32_t count = 0;
};

// Reusable GPU operator helpers.
//
// Implementations are split across:
//   Operators.cpp  — scan, gather/scatter, arithmetic, reduction, string, date, utility
//   FilterOps.cpp  — all filter* methods
//   JoinOps.cpp    — join, semi/anti join, cross product
//   GroupByOps.cpp — groupBy, extractHT, dedup
//   SortOps.cpp    — radix sort, sort-key conversion
class GpuOps {
public:

    // ── Batch Control ──────────────────────────────────────────────────
    // When active, arithmetic ops skip waitUntilCompleted() — the serial
    // command queue guarantees ordering. Call endBatch() to flush.
    static void beginBatch();
    static void endBatch();
    static bool isBatchActive();

    // ── Scan / Data Loading ────────────────────────────────────────────
    static GpuRelation scanTable(const std::string& datasetPath,
                                 const std::string& table,
                                 const std::vector<std::string>& neededCols);
    static std::vector<std::string> loadStringColumnRaw(const std::string& datasetPath,
                                                        const std::string& table,
                                                        const std::string& column);
    static uint32_t fnv1a32(std::string_view s);

    // ── Filter (FilterOps.cpp) ─────────────────────────────────────────
    static std::optional<FilterResult> filterU32(const std::string& colName,
                                                 MTL::Buffer* col, uint32_t rowCount,
                                                 engine::GpuFilterOp op, uint32_t literal);
    static std::optional<FilterResult> filterU32Indexed(const std::string& colName,
                                                        MTL::Buffer* col, MTL::Buffer* indices,
                                                        uint32_t count, engine::GpuFilterOp op,
                                                        uint32_t literal);
    static std::optional<FilterResult> filterF32(const std::string& colName,
                                                 MTL::Buffer* col, uint32_t rowCount,
                                                 engine::GpuFilterOp op, float literal);
    static std::optional<FilterResult> filterF32Indexed(const std::string& colName,
                                                        MTL::Buffer* col, MTL::Buffer* indices,
                                                        uint32_t count, engine::GpuFilterOp op,
                                                        float literal);
    // String filter: when preChars/preOffsets/preLengths are non-null, skip CPU→GPU flatten.
    static std::optional<FilterResult> filterString(const std::string& colName,
                                                    const std::vector<std::string>& data,
                                                    engine::GpuFilterOp op, const std::string& pattern,
                                                    MTL::Buffer* preChars = nullptr,
                                                    MTL::Buffer* preOffsets = nullptr,
                                                    MTL::Buffer* preLengths = nullptr);
    static std::optional<FilterResult> filterStringPrefix(const std::string& colName,
                                                          const std::vector<std::string>& data,
                                                          const std::string& pattern, bool invert = false,
                                                          MTL::Buffer* preChars = nullptr,
                                                          MTL::Buffer* preOffsets = nullptr,
                                                          MTL::Buffer* preLengths = nullptr);
    // Column-vs-column filter (op: 0=LT, 1=LE, 2=GT, 3=GE, 4=EQ, 5=NE)
    static std::optional<FilterResult> filterColColU32(MTL::Buffer* colA, MTL::Buffer* colB,
                                                       uint32_t count, int op);
    static std::optional<FilterResult> filterColColF32(MTL::Buffer* colA, MTL::Buffer* colB,
                                                       uint32_t count, int op);

    // ── Join (JoinOps.cpp) ─────────────────────────────────────────────
    static JoinResult joinHash(MTL::Buffer* buildKeys, uint32_t buildCount,
                               MTL::Buffer* probeKeys, uint32_t probeCount);
    static JoinResult joinHashU64(MTL::Buffer* buildKeys, MTL::Buffer* buildIndices,
                                  uint32_t buildCount, MTL::Buffer* probeKeys,
                                  MTL::Buffer* probeIndices, uint32_t probeCount);
    static GpuBuffer packU32ToU64(MTL::Buffer* c1, MTL::Buffer* c2, uint32_t count);
    static std::optional<FilterResult> hashJoinSemiU32(MTL::Buffer* leftKey, uint32_t leftCount,
                                                       MTL::Buffer* rightKey, uint32_t rightCount);
    static std::optional<FilterResult> hashJoinAntiU32(MTL::Buffer* leftKey, uint32_t leftCount,
                                                       MTL::Buffer* rightKey, uint32_t rightCount);
    // Returns indices in [0, totalRows) NOT in matchedIndices (GPU scatter→flip→compact).
    static FilterResult findUnmatchedIndices(MTL::Buffer* matchedIndices,
                                             uint32_t matchedCount, uint32_t totalRows);
    static void crossProduct(MTL::Buffer* left, MTL::Buffer* right,
                             MTL::Buffer* outLeft, MTL::Buffer* outRight,
                             uint32_t leftCount, uint32_t rightCount);

    // ── GroupBy & Dedup (GroupByOps.cpp) ────────────────────────────────
    // aggTypes[a]: 0 = SUM(f32), 1 = COUNT(*). Results in htAggs[slot*8+a].
    static std::optional<GroupByHashTable> groupByAggMultiKeyTyped(
        const std::vector<MTL::Buffer*>& keyColsU32,
        const std::vector<MTL::Buffer*>& aggInputsF32,
        const std::vector<uint32_t>& aggTypes, uint32_t rowCount);
    // GPU stream compaction: Mark → PrefixSum → Compact on hash table.
    static std::optional<GroupByExtractResult> extractGroupByHT(
        const GroupByHashTable& ht, uint32_t numKeys, uint32_t numAggsTotal);
    static void release(GroupByHashTable& g);
    // GPU dedup — returns gather indices for unique rows (1-2 keys; CPU fallback for 3+).
    static GpuBuffer dedupByKeys(const std::vector<MTL::Buffer*>& keys, uint32_t count,
                                 uint32_t& uniqueCount);

    // ── Sort (SortOps.cpp) ─────────────────────────────────────────────
    // Stable 8-bit LSD radix sort. ≤1024: shared-memory bitonic; >1024: multi-pass.
    static void radixSortU32(MTL::Buffer* keys, MTL::Buffer* indices, uint32_t count);
    static void radixSortU64(MTL::Buffer* keys, MTL::Buffer* indices, uint32_t count);
    // IEEE 754 f32 → order-preserving u32 sort key (desc flips for descending).
    static GpuBuffer floatToSortKeyU32(MTL::Buffer* in, uint32_t count, bool desc);
    // Bitwise NOT for DESC ordering.
    static GpuBuffer invertU32(MTL::Buffer* in, uint32_t count);

    // ── Gather & Scatter ───────────────────────────────────────────────
    static GpuBuffer gatherU32(MTL::Buffer* in, MTL::Buffer* indices, uint32_t count, bool sync = true);
    static GpuBuffer gatherF32(MTL::Buffer* in, MTL::Buffer* indices, uint32_t count, bool sync = true);
    static void scatterConstantF32(MTL::Buffer* output, MTL::Buffer* indices, uint32_t indexCount, float val);
    static void scatterF32(MTL::Buffer* input, MTL::Buffer* output, MTL::Buffer* indices, uint32_t count);

    // ── Mask & Index Operations ────────────────────────────────────────
    static GpuBuffer logicOrU32(MTL::Buffer* colA, MTL::Buffer* colB, uint32_t count);
    static GpuBuffer logicAndNotU32(MTL::Buffer* colA, MTL::Buffer* colB, uint32_t count);
    static GpuBuffer indicesToMask(MTL::Buffer* indices, uint32_t indexCount, uint32_t totalRows);
    static std::pair<GpuBuffer, uint32_t> compactU32Mask(MTL::Buffer* mask, uint32_t totalRows);

    // ── Fill & Initialize ──────────────────────────────────────────────
    static void fillU32(MTL::Buffer* buf, uint32_t val, uint32_t count);
    static void fillF32(MTL::Buffer* buf, float val, uint32_t count);
    static GpuBuffer createFilledU32(uint32_t val, uint32_t count);
    static GpuBuffer createFilledF32(float val, uint32_t count);
    static GpuBuffer iotaU32(uint32_t count);

    // ── Arithmetic ─────────────────────────────────────────────────────
    static GpuBuffer arithAddConstU32(MTL::Buffer* in, uint32_t val, uint32_t count);
    static GpuBuffer arithAddF32ColCol(MTL::Buffer* colA, MTL::Buffer* colB, uint32_t count);
    static GpuBuffer arithAddF32ColScalar(MTL::Buffer* colA, float valB, uint32_t count);
    static GpuBuffer arithSubF32ColCol(MTL::Buffer* colA, MTL::Buffer* colB, uint32_t count);
    static GpuBuffer arithSubF32ColScalar(MTL::Buffer* colA, float valB, uint32_t count);
    static GpuBuffer arithSubF32ScalarCol(float valA, MTL::Buffer* colB, uint32_t count);
    static GpuBuffer arithMulF32ColCol(MTL::Buffer* colA, MTL::Buffer* colB, uint32_t count);
    static GpuBuffer arithMulF32ColScalar(MTL::Buffer* colA, float valB, uint32_t count);
    static GpuBuffer arithDivF32ColCol(MTL::Buffer* colA, MTL::Buffer* colB, uint32_t count);
    static GpuBuffer arithDivF32ColScalar(MTL::Buffer* colA, float valB, uint32_t count);
    static GpuBuffer arithDivF32ScalarCol(float valA, MTL::Buffer* colB, uint32_t count);
    static GpuBuffer nonNullIndicatorF32(MTL::Buffer* in, uint32_t count);
    static GpuBuffer mathFloorF32(MTL::Buffer* col, uint32_t count);

    // ── Conversion & Bitwise ───────────────────────────────────────────
    static GpuBuffer bitcastF32ToU32(MTL::Buffer* in, uint32_t count);
    static GpuBuffer castU32ToF32(MTL::Buffer* in, uint32_t count);

    // ── Reduction ──────────────────────────────────────────────────────
    static float reduceSumF32(MTL::Buffer* in, uint32_t count);
    static float reduceMinF32(MTL::Buffer* in, uint32_t count);
    static float reduceMaxF32(MTL::Buffer* in, uint32_t count);

    // ── Date ───────────────────────────────────────────────────────────
    // Extract YEAR from u32 date → u32 year (handles YYYYMMDD and day-count formats).
    static GpuBuffer extractYearU32(MTL::Buffer* dateCol, uint32_t count);

    // ── String Operations ──────────────────────────────────────────────
    struct FlatStringGatherResult {
        GpuBuffer chars;
        GpuBuffer offsets;
        GpuBuffer lengths;
        uint32_t rowCount    = 0;
        uint32_t totalBytes  = 0;
    };
    // SUBSTRING: adjusts offsets/lengths (zero-copy). startPos is 1-based (SQL).
    static std::pair<GpuBuffer, GpuBuffer> substringFlat(
        MTL::Buffer* inOffsets, MTL::Buffer* inLengths,
        uint32_t startPos, uint32_t substrLen, uint32_t rowCount);
    // Gather chars/offsets/lengths by index buffer into a compacted FlatStringCol.
    static FlatStringGatherResult gatherFlatString(
        MTL::Buffer* srcChars, MTL::Buffer* srcOffsets, MTL::Buffer* srcLengths,
        MTL::Buffer* indices, uint32_t count, bool sync = true);
    static FlatStringGatherResult concatFlatStrings(
        const FlatStringGatherResult& a, const FlatStringGatherResult& b);
    // Hash-encode flat strings to u32 (first 8 chars, big-endian packed).
    static GpuBuffer stringHashEncodeU32(
        MTL::Buffer* chars, MTL::Buffer* offsets, MTL::Buffer* lengths, uint32_t rowCount);
    // FNV1a-32 hash (full string, better distribution).
    static GpuBuffer stringFnv1aU32(
        MTL::Buffer* chars, MTL::Buffer* offsets, MTL::Buffer* lengths, uint32_t rowCount);
    // FNV1a-64 hash XOR-folded to u32 (collision-resistant, for group-by keys).
    static GpuBuffer stringFnv1aU64Fold32(
        MTL::Buffer* chars, MTL::Buffer* offsets, MTL::Buffer* lengths, uint32_t rowCount);
    // GPU string rank by prefix-u64 sort. Empty GpuBuffer if ties detected (CPU fallback).
    static GpuBuffer stringRankByPrefix(
        MTL::Buffer* chars, MTL::Buffer* offsets, MTL::Buffer* lengths,
        uint32_t rowCount, bool ascending);

    // ── Utility ────────────────────────────────────────────────────────
    static void sync();
    static GpuBuffer createBuffer(const void* data, size_t size);
};

} // namespace engine
