#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "Predicate.hpp"
#include "Relation.hpp"
#include "JoinMap.hpp"

namespace engine {

struct FlatStringColumn; // forward decl (defined in GpuExecutorPriv.hpp)

struct FilterResult {
    MTL::Buffer* indices = nullptr; // u32 indices, length == count
    uint32_t count = 0;
};

struct GroupByHashTable {
    MTL::Buffer* ht_keys = nullptr;  // capacity * 4 u32
    MTL::Buffer* ht_aggs = nullptr;  // capacity * 8 u32 (float bits)
    uint32_t capacity = 0;
};

// Result of GPU stream compaction on a GroupBy hash table.
struct GroupByExtractResult {
    std::vector<std::vector<uint32_t>> keyCols;  // [keyIdx][row], bias removed
    std::vector<std::vector<uint32_t>> aggWords; // [aggSlot][row], raw u32
    uint32_t rowCount = 0;
};


struct JoinResult {
    MTL::Buffer* buildIndices = nullptr;
    MTL::Buffer* probeIndices = nullptr;
    uint32_t count = 0;
};

// Reusable GPU operator helpers.
class GpuOps {
public:
    static JoinResult joinHash(
        MTL::Buffer* buildKeys, 
        MTL::Buffer* buildIndices, // Optional: if null, implied 0..N
        uint32_t buildCount,
        MTL::Buffer* probeKeys,
        MTL::Buffer* probeIndices, // Optional: if null, implied 0..N
        uint32_t probeCount
    );

    // Join on u64 keys (multi-column)
    static JoinResult joinHashU64(
        MTL::Buffer* buildKeys, 
        MTL::Buffer* buildIndices, 
        uint32_t buildCount,
        MTL::Buffer* probeKeys,
        MTL::Buffer* probeIndices,
        uint32_t probeCount
    );

    static MTL::Buffer* packU32ToU64(MTL::Buffer* c1, MTL::Buffer* c2, uint32_t count);

    static uint32_t fnv1a32(std::string_view s);

    // Encode a string literal to the u32 representation used by the column (hash or char)
    static uint32_t encodeStringForColumn(const std::string& table, const std::string& col, const std::string& val);


    // Load raw strings from a table column (for pattern matching: LIKE, CONTAINS)
    static std::vector<std::string> loadStringColumnRaw(const std::string& dataset_path,
                                                        const std::string& table,
                                                        const std::string& column);

    // Load a table into a RelationGPU with only the requested columns.
    static RelationGPU scanTable(const std::string& dataset_path,
                                 const std::string& table,
                                 const std::vector<std::string>& neededCols);

    // Filter a u32 column with (op, literal) and return compacted indices.
    static std::optional<FilterResult> filterU32(const std::string& colName,
                                                    MTL::Buffer* col,
                                                    uint32_t rowCount,
                                                    engine::expr::CompOp op,
                                                    uint32_t literal);

    // Filter a raw string column. 
    // Uses GPU kernel for pattern matching.
    static std::optional<FilterResult> filterString(const std::string& colName,
                                                       const std::vector<std::string>& data,
                                                       engine::expr::CompOp op,
                                                       const std::string& pattern);

    static std::optional<FilterResult> filterStringPrefix(const std::string& colName,
                                                      const std::vector<std::string>& data,
                                                      const std::string& pattern,
                                                      bool invert = false);

    // Arrow-style flat string filter overloads — use persistent FlatStringColumn buffers
    // directly, avoiding the re-flatten from vector<string>.
    static std::optional<FilterResult> filterStringFlat(const std::string& colName,
                                                        const FlatStringColumn& flat,
                                                        engine::expr::CompOp op,
                                                        const std::string& pattern);

    static std::optional<FilterResult> filterStringPrefixFlat(const std::string& colName,
                                                               const FlatStringColumn& flat,
                                                               const std::string& pattern,
                                                               bool invert = false);

    static std::optional<FilterResult> filterU32Indexed(const std::string& colName,
                                                           MTL::Buffer* col,
                                                           MTL::Buffer* indices,
                                                           uint32_t count,
                                                           engine::expr::CompOp op,
                                                           uint32_t literal);

    // Filter a f32 column with (op, literal) and return compacted indices.
    static std::optional<FilterResult> filterF32(const std::string& colName,
                                                    MTL::Buffer* col,
                                                    uint32_t rowCount,
                                                    engine::expr::CompOp op,
                                                    float literal);

    // Scatter constant value to output buffer at specified indices
    static void scatterConstantF32(MTL::Buffer* output, MTL::Buffer* indices, uint32_t indexCount, float val);
    
    // Scatter values from input vector to output buffer at specified indices
    static void scatterF32(MTL::Buffer* input, MTL::Buffer* output, MTL::Buffer* indices, uint32_t count);


    // Filter a f32 column through an indices vector (no predicate-column gather).
    // Returns indices into the provided `indices` array.
    static std::optional<FilterResult> filterF32Indexed(const std::string& colName,
                                                           MTL::Buffer* col,
                                                           MTL::Buffer* indices,
                                                           uint32_t count,
                                                           engine::expr::CompOp op,
                                                           float literal);

    // Filter a f32 column with BETWEEN [min, max] (inclusive) and return compacted indices.
    static std::optional<FilterResult> filterF32Between(const std::string& colName,
                                                           MTL::Buffer* col,
                                                           uint32_t rowCount,
                                                           float minVal,
                                                           float maxVal);

    // Filter a f32 column through an indices vector using BETWEEN.
    static std::optional<FilterResult> filterF32BetweenIndexed(const std::string& colName,
                                                                  MTL::Buffer* col,
                                                                  MTL::Buffer* indices,
                                                                  uint32_t count,
                                                                  float minVal,
                                                                  float maxVal);

    // Filter a u32 column with BETWEEN [min, max] (inclusive) and return compacted indices.
    static std::optional<FilterResult> filterU32Between(const std::string& colName,
                                                           MTL::Buffer* col,
                                                           uint32_t rowCount,
                                                           uint32_t minVal,
                                                           uint32_t maxVal);

    // Filter a u32 column through an indices vector using BETWEEN.
    static std::optional<FilterResult> filterU32BetweenIndexed(const std::string& colName,
                                                                  MTL::Buffer* col,
                                                                  MTL::Buffer* indices,
                                                                  uint32_t count,
                                                                  uint32_t minVal,
                                                                  uint32_t maxVal);

    // Materialize relation by gathering each present column using indices.
    static RelationGPU applySelection(RelationGPU&& rel, const FilterResult& sel);

    // Build hash table on rightKey (payload is right row index) and probe leftKey.
    static std::optional<JoinMapGPU> hashJoinU32(MTL::Buffer* leftKey,
                                                 uint32_t leftCount,
                                                 MTL::Buffer* rightKey,
                                                 uint32_t rightCount);

    static std::optional<FilterResult> hashJoinSemiU32(MTL::Buffer* leftKey,
                                                          uint32_t leftCount,
                                                          MTL::Buffer* rightKey,
                                                          uint32_t rightCount);

    // Materialize joined columns: gather from left/right using JoinMapGPU.
    static RelationGPU materializeJoin(RelationGPU&& left,
                                       RelationGPU&& right,
                                       const JoinMapGPU& map,
                                       const std::vector<std::string>& keepLeftU32,
                                       const std::vector<std::string>& keepLeftF32,
                                       const std::vector<std::string>& keepRightU32,
                                       const std::vector<std::string>& keepRightF32);

    // Compute revenue = extendedprice * (1-discount)
    static MTL::Buffer* computeRevenue(MTL::Buffer* extendedprice,
                                       MTL::Buffer* discount,
                                       uint32_t rowCount);

    // Compute charge = extendedprice * (1-discount) * (1+tax)
    static MTL::Buffer* computeCharge(MTL::Buffer* extendedprice,
                                      MTL::Buffer* discount,
                                      MTL::Buffer* tax,
                                      uint32_t rowCount);

    // GroupBy multi-key SUM over one float agg.
    static std::optional<GroupByHashTable> groupBySumMultiKey(const std::vector<MTL::Buffer*>& keyColsU32,
                                                                 MTL::Buffer* aggF32,
                                                                 uint32_t rowCount);

    // GroupBy multi-key SUM+COUNT over one float agg.
    // COUNT is accumulated as u32 in ht_aggs[slot*8 + 1].
    static std::optional<GroupByHashTable> groupBySumCountMultiKey(const std::vector<MTL::Buffer*>& keyColsU32,
                                                                      MTL::Buffer* aggF32,
                                                                      uint32_t rowCount);

    // GroupBy multi-key COUNT(*).
    // COUNT is accumulated as u32 in ht_aggs[slot*8 + 0].
    static std::optional<GroupByHashTable> groupByCountMultiKey(const std::vector<MTL::Buffer*>& keyColsU32,
                                                                   uint32_t rowCount);

    // GroupBy multi-key with typed aggregates (up to 8).
    // aggTypes[a]: 0 = SUM(f32) using aggInputsF32[a], 1 = COUNT(*) (u32).
    // Results are written into ht_aggs[slot*8 + a].
    static std::optional<GroupByHashTable> groupByAggMultiKeyTyped(const std::vector<MTL::Buffer*>& keyColsU32,
                                                                      const std::vector<MTL::Buffer*>& aggInputsF32,
                                                                      const std::vector<uint32_t>& aggTypes,
                                                                      uint32_t rowCount);

    // Utility: gather column buffers
    static MTL::Buffer* gatherU32(MTL::Buffer* in, MTL::Buffer* indices, uint32_t count, bool sync = true);
    static MTL::Buffer* gatherF32(MTL::Buffer* in, MTL::Buffer* indices, uint32_t count, bool sync = true);

    // Helper to synchronize command queue
    static void sync();

    // Utility: cast a u32 buffer into a f32 buffer (StorageModeShared).
    // Used to treat integer columns as float aggregate inputs in v0 group-by.
    static MTL::Buffer* castU32ToF32(MTL::Buffer* in, uint32_t count);

    // Filter Col vs Col (U32)
    static std::optional<FilterResult> filterColColU32(
        MTL::Buffer* colA,
        MTL::Buffer* colB,
        uint32_t count,
        int op);

    // Filter Col vs Col (F32)
    static std::optional<FilterResult> filterColColF32(
        MTL::Buffer* colA,
        MTL::Buffer* colB,
        uint32_t count,
        int op);

    // Generic Mask Operations
    static MTL::Buffer* cmpColColU32Mask(MTL::Buffer* colA, MTL::Buffer* colB, uint32_t count, int op);
    static MTL::Buffer* cmpColLitU32Mask(MTL::Buffer* colA, uint32_t valB, uint32_t count, int op);
    static MTL::Buffer* logicOrU32(MTL::Buffer* colA, MTL::Buffer* colB, uint32_t count);
    static MTL::Buffer* logicAndU32(MTL::Buffer* colA, MTL::Buffer* colB, uint32_t count);
    static MTL::Buffer* logicNotU32(MTL::Buffer* mask, uint32_t count);
    static MTL::Buffer* logicAndNotU32(MTL::Buffer* colA, MTL::Buffer* colB, uint32_t count);
    
    // Index array to mask conversion
    static MTL::Buffer* indicesToMask(MTL::Buffer* indices, uint32_t indexCount, uint32_t totalRows);
    static void clearMask(MTL::Buffer* mask, uint32_t count);
    
    // Compact u32 mask to indices
    static std::pair<MTL::Buffer*, uint32_t> compactU32Mask(MTL::Buffer* mask, uint32_t totalRows);
    
    // Fill operations
    static void fillU32(MTL::Buffer* buf, uint32_t val, uint32_t count);
    static void fillF32(MTL::Buffer* buf, float val, uint32_t count);
    static MTL::Buffer* createFilledU32(uint32_t val, uint32_t count);
    static MTL::Buffer* createFilledF32(float val, uint32_t count);
    
    // Generate sequence 0, 1, 2, ... (iota)
    static MTL::Buffer* iotaU32(uint32_t count);
    
    // Cross product
    static void crossProduct(MTL::Buffer* left, MTL::Buffer* right, 
                             MTL::Buffer* outLeft, MTL::Buffer* outRight,
                             uint32_t leftCount, uint32_t rightCount);
    
    // Select
    static MTL::Buffer* selectU32(MTL::Buffer* mask, MTL::Buffer* t, MTL::Buffer* f, uint32_t count);

    // Utility: CPU-side copy with +constant bias (buffers are StorageModeShared).
    // Used to avoid 0 being interpreted as an empty key in hash tables.
    static MTL::Buffer* copyAddU32(MTL::Buffer* in, uint32_t count, uint32_t add);

    static void release(FilterResult& r);
    static void release(JoinMapGPU& j);
    static void release(GroupByHashTable& g);

    // GPU stream compaction: extract valid entries from GroupBy hash table.
    // Mark → Prefix Sum → Compact pipeline on GPU.
    // Returns deinterleaved key columns (bias removed) and raw agg words.
    static std::optional<GroupByExtractResult> extractGroupByHT(
        const GroupByHashTable& ht,
        uint32_t numKeys,
        uint32_t numAggsTotal);

    // Arithmetic Ops
    static MTL::Buffer* arithMulF32ColCol(MTL::Buffer* colA, MTL::Buffer* colB, uint32_t count);
    static MTL::Buffer* arithMulF32ColColIndexed(MTL::Buffer* colA, MTL::Buffer* colB, MTL::Buffer* indices, uint32_t count);
    static MTL::Buffer* arithMulF32ColScalar(MTL::Buffer* colA, float valB, uint32_t count);
    static MTL::Buffer* arithMulF32ColScalarIndexed(MTL::Buffer* colA, float valB, MTL::Buffer* indices, uint32_t count);

    static MTL::Buffer* arithDivF32ColCol(MTL::Buffer* colA, MTL::Buffer* colB, uint32_t count);
    static MTL::Buffer* arithDivF32ScalarCol(float valA, MTL::Buffer* colB, uint32_t count);
    static MTL::Buffer* arithDivF32ColScalar(MTL::Buffer* colA, float valB, uint32_t count);

    static MTL::Buffer* arithSubF32ColCol(MTL::Buffer* colA, MTL::Buffer* colB, uint32_t count);
    static MTL::Buffer* arithSubF32ScalarCol(float valA, MTL::Buffer* colB, uint32_t count);
    static MTL::Buffer* arithSubF32ColScalar(MTL::Buffer* colA, float valB, uint32_t count);

    static MTL::Buffer* arithAddF32ColCol(MTL::Buffer* colA, MTL::Buffer* colB, uint32_t count);
    static MTL::Buffer* arithAddF32ScalarCol(float valA, MTL::Buffer* colB, uint32_t count);
    static MTL::Buffer* arithAddF32ColScalar(MTL::Buffer* colA, float valB, uint32_t count);

    // Math Ops
    static MTL::Buffer* mathFloorF32(MTL::Buffer* col, uint32_t count);

    // Helpers
    static MTL::Buffer* createBuffer(const void* data, size_t size);

    // Reduction Ops (returns scalar on CPU)
    static float reduceSumF32(MTL::Buffer* in, uint32_t count);
    static float reduceMinF32(MTL::Buffer* in, uint32_t count);
    static float reduceMaxF32(MTL::Buffer* in, uint32_t count);

    // Date Extract
    static MTL::Buffer* extractYearFromDate(MTL::Buffer* dateCol, uint32_t count);

    // GPU Bitonic Sort
    // Sorts an index array by a u32 key array in-place.
    // After sort, indices[i] gives the original row index of the i-th sorted element.
    // Keys are also sorted in-place. Both buffers must be at least `count` elements.
    static void bitonicSortU32(MTL::Buffer* keys, MTL::Buffer* indices, uint32_t count);
    // Same but with u64 keys (for multi-column composite sort keys)
    static void bitonicSortU64(MTL::Buffer* keys, MTL::Buffer* indices, uint32_t count);

    // GPU Radix Sort (stable, 8-bit radix)
    // For ≤1024 elements: single-dispatch block sort (shared-memory bitonic).
    // For >1024 elements: 4-pass (u32) or 8-pass (u64) LSD radix sort.
    static void radixSortU32(MTL::Buffer* keys, MTL::Buffer* indices, uint32_t count);
    static void radixSortU64(MTL::Buffer* keys, MTL::Buffer* indices, uint32_t count);
};

} // namespace engine
