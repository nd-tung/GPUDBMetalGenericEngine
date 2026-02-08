#include <metal_stdlib>
using namespace metal;

namespace ops {

struct ColumnViewUInt32 {
    device const uint32_t* data;
    uint32_t count;
};

struct ColumnViewFloat {
    device const float* data;
    uint32_t count;
};

struct ColumnOutUInt32 {
    device uint32_t* data;
    uint32_t count;
};

struct RowMask {
    device uint8_t* mask; // 0/1 per row
    uint32_t count;
};

// ============================================================================
// HASH JOIN KERNELS (Linear Probing)
// ============================================================================

constant uint32_t MAX_HASH_STEPS = 128;
constant uint32_t EMPTY_KEY = 0xFFFFFFFF; // Sentinel value

// Murmur3-style hash mixer
inline uint32_t hash_u32(uint32_t k) {
    k ^= k >> 16;
    k *= 0x85ebca6b;
    k ^= k >> 13;
    k *= 0xc2b2ae35;
    k ^= k >> 16;
    return k;
}

// Build: Insert (key, row_id) pairs into hash table
kernel void join_build_u32(
    const device uint32_t* build_keys [[buffer(0)]],
    const device uint32_t* build_indices [[buffer(1)]], // Original row IDs (optional, else use gid)
    device uint32_t* ht_keys [[buffer(2)]],    // Hash Table Keys
    device uint32_t* ht_vals [[buffer(3)]],    // Hash Table Values (Row IDs)
    constant uint32_t& ht_capacity [[buffer(4)]],
    constant uint32_t& row_count [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) 
{
    if (gid >= row_count) return;

    // Use indices for indirect access if provided
    uint32_t actual_idx = gid;
    if (build_indices) actual_idx = build_indices[gid];
    
    uint32_t key = build_keys[actual_idx];
    uint32_t payload = actual_idx;

    uint32_t h = hash_u32(key);
    uint32_t idx = h % ht_capacity;
    
    // Linear probing
    for (uint32_t i = 0; i < MAX_HASH_STEPS; ++i) {
        uint32_t expected = EMPTY_KEY;
        device atomic_uint* a_keys = (device atomic_uint*)ht_keys;
        
        bool won = atomic_compare_exchange_weak_explicit(
            &a_keys[idx], &expected, key,
            memory_order_relaxed, memory_order_relaxed
        );
        
        if (won) {
            ht_vals[idx] = payload;
            return;
        }
        
        if (expected == key) {
             // Duplicate key on build side.
             ht_vals[idx] = payload;
             return;
        }
        
        // Collision, move next
        idx = (idx + 1) % ht_capacity;
    }
}

// Probe (Atomic Materialize, unique build keys)
kernel void join_probe_u32(
    const device uint32_t* probe_keys [[buffer(0)]],
    const device uint32_t* probe_indices [[buffer(1)]], // Original probe row IDs (optional)
    const device uint32_t* ht_keys [[buffer(2)]],
    const device uint32_t* ht_vals [[buffer(3)]],
    constant uint32_t& ht_capacity [[buffer(4)]],
    constant uint32_t& row_count [[buffer(5)]],
    device atomic_uint* out_counter [[buffer(6)]],
    device uint32_t* out_build_indices [[buffer(7)]],
    device uint32_t* out_probe_indices [[buffer(8)]],
    uint gid [[thread_position_in_grid]]) 
{
    if (gid >= row_count) return;
    
    // Use indices for indirect access if provided
    uint32_t actual_idx = gid;
    if (probe_indices) actual_idx = probe_indices[gid];
    uint32_t key = probe_keys[actual_idx];
    // if key is 0? TPC-H has valid keys.
    
    uint32_t h = hash_u32(key);
    uint32_t idx = h % ht_capacity;
    
    for (uint32_t i = 0; i < MAX_HASH_STEPS; ++i) {
        uint32_t k_at = ht_keys[idx];
        if (k_at == EMPTY_KEY) {
            // Not found
            break; 
        }
        if (k_at == key) {
            // Found
            uint32_t build_idx = ht_vals[idx];
            
            uint32_t write_pos = atomic_fetch_add_explicit(out_counter, 1, memory_order_relaxed);
            
            out_build_indices[write_pos] = build_idx;
            out_probe_indices[write_pos] = actual_idx;  // Use actual row index
            
            break;
        }
        idx = (idx + 1) % ht_capacity;
    }
}



kernel void filter_range_to_mask_f32(const device float* col [[buffer(0)]],
                                     device uint8_t* out_mask [[buffer(1)]],
                                     constant float& min_val [[buffer(2)]],
                                     constant float& max_val [[buffer(3)]],
                                     constant uint& row_count [[buffer(4)]],
                                     uint gid [[thread_position_in_grid]]) {
    if (gid >= row_count) return;
    float val = col[gid];
    out_mask[gid] = (val >= min_val && val <= max_val) ? 1 : 0;
}

kernel void filter_range_to_mask_int32(const device int* col [[buffer(0)]],
                                       device uint8_t* out_mask [[buffer(1)]],
                                       constant int& min_val [[buffer(2)]],
                                       constant int& max_val [[buffer(3)]],
                                       constant uint& row_count [[buffer(4)]],
                                       uint gid [[thread_position_in_grid]]) {
    if (gid >= row_count) return;
    int val = col[gid];
    out_mask[gid] = (val >= min_val && val <= max_val) ? 1 : 0;
}

kernel void filter_eq_u32(const device uint32_t* in,
                          device uint8_t* out_mask,
                          constant uint32_t& eq_value,
                          uint gid [[thread_position_in_grid]],
                          uint grid_size [[threads_per_grid]]) {
    if (gid >= grid_size) return;
    out_mask[gid] = (in[gid] == eq_value) ? 1 : 0;
}

kernel void filter_eq_u32_indexed(const device uint32_t* in [[buffer(0)]],
                                  const device uint32_t* indices [[buffer(1)]],
                                  device uint8_t* out_mask [[buffer(2)]],
                                  constant uint32_t& eq_value [[buffer(3)]],
                                  constant uint& count [[buffer(4)]],
                                  uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    const uint32_t idx = indices[gid];
    out_mask[gid] = (in[idx] == eq_value) ? 1 : 0;
}

kernel void filter_lt_u32(const device uint32_t* in,
                          device uint8_t* out_mask,
                          constant uint32_t& lt_value,
                          constant uint& row_count,
                          uint gid [[thread_position_in_grid]]) {
    if (gid >= row_count) return;
    out_mask[gid] = (in[gid] < lt_value) ? 1 : 0;
}

kernel void filter_lt_u32_indexed(const device uint32_t* in [[buffer(0)]],
                                  const device uint32_t* indices [[buffer(1)]],
                                  device uint8_t* out_mask [[buffer(2)]],
                                  constant uint32_t& lt_value [[buffer(3)]],
                                  constant uint& count [[buffer(4)]],
                                  uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    const uint32_t idx = indices[gid];
    out_mask[gid] = (in[idx] < lt_value) ? 1 : 0;
}

kernel void filter_gt_u32(const device uint32_t* in,
                          device uint8_t* out_mask,
                          constant uint32_t& gt_value,
                          constant uint& row_count,
                          uint gid [[thread_position_in_grid]]) {
    if (gid >= row_count) return;
    out_mask[gid] = (in[gid] > gt_value) ? 1 : 0;
}

kernel void filter_gt_u32_indexed(const device uint32_t* in [[buffer(0)]],
                                  const device uint32_t* indices [[buffer(1)]],
                                  device uint8_t* out_mask [[buffer(2)]],
                                  constant uint32_t& gt_value [[buffer(3)]],
                                  constant uint& count [[buffer(4)]],
                                  uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    const uint32_t idx = indices[gid];
    out_mask[gid] = (in[idx] > gt_value) ? 1 : 0;
}

kernel void filter_le_u32(const device uint32_t* in,
                          device uint8_t* out_mask,
                          constant uint32_t& le_value,
                          constant uint& row_count,
                          uint gid [[thread_position_in_grid]]) {
    if (gid >= row_count) return;
    out_mask[gid] = (in[gid] <= le_value) ? 1 : 0;
}

kernel void filter_le_u32_indexed(const device uint32_t* in [[buffer(0)]],
                                  const device uint32_t* indices [[buffer(1)]],
                                  device uint8_t* out_mask [[buffer(2)]],
                                  constant uint32_t& le_value [[buffer(3)]],
                                  constant uint& count [[buffer(4)]],
                                  uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    const uint32_t idx = indices[gid];
    out_mask[gid] = (in[idx] <= le_value) ? 1 : 0;
}

kernel void filter_ge_u32(const device uint32_t* in,
                          device uint8_t* out_mask,
                          constant uint32_t& ge_value,
                          constant uint& row_count,
                          uint gid [[thread_position_in_grid]]) {
    if (gid >= row_count) return;
    out_mask[gid] = (in[gid] >= ge_value) ? 1 : 0;
}

kernel void filter_ge_u32_indexed(const device uint32_t* in [[buffer(0)]],
                                  const device uint32_t* indices [[buffer(1)]],
                                  device uint8_t* out_mask [[buffer(2)]],
                                  constant uint32_t& ge_value [[buffer(3)]],
                                  constant uint& count [[buffer(4)]],
                                  uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    const uint32_t idx = indices[gid];
    out_mask[gid] = (in[idx] >= ge_value) ? 1 : 0;
}

kernel void filter_eq_f32(const device float* in,
                          device uint8_t* out_mask,
                          constant float& eq_value,
                          constant uint& row_count,
                          uint gid [[thread_position_in_grid]]) {
    if (gid >= row_count) return;
    out_mask[gid] = (in[gid] == eq_value) ? 1 : 0;
}

kernel void filter_eq_f32_indexed(const device float* in [[buffer(0)]],
                                  const device uint32_t* indices [[buffer(1)]],
                                  device uint8_t* out_mask [[buffer(2)]],
                                  constant float& eq_value [[buffer(3)]],
                                  constant uint& count [[buffer(4)]],
                                  uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    const uint32_t idx = indices[gid];
    out_mask[gid] = (in[idx] == eq_value) ? 1 : 0;
}

kernel void filter_lt_f32(const device float* in,
                          device uint8_t* out_mask,
                          constant float& lt_value,
                          constant uint& row_count,
                          uint gid [[thread_position_in_grid]]) {
    if (gid >= row_count) return;
    out_mask[gid] = (in[gid] < lt_value) ? 1 : 0;
}

kernel void filter_lt_f32_indexed(const device float* in [[buffer(0)]],
                                  const device uint32_t* indices [[buffer(1)]],
                                  device uint8_t* out_mask [[buffer(2)]],
                                  constant float& lt_value [[buffer(3)]],
                                  constant uint& count [[buffer(4)]],
                                  uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    const uint32_t idx = indices[gid];
    out_mask[gid] = (in[idx] < lt_value) ? 1 : 0;
}

kernel void filter_gt_f32(const device float* in,
                          device uint8_t* out_mask,
                          constant float& gt_value,
                          constant uint& row_count,
                          uint gid [[thread_position_in_grid]]) {
    if (gid >= row_count) return;
    out_mask[gid] = (in[gid] > gt_value) ? 1 : 0;
}

kernel void filter_gt_f32_indexed(const device float* in [[buffer(0)]],
                                  const device uint32_t* indices [[buffer(1)]],
                                  device uint8_t* out_mask [[buffer(2)]],
                                  constant float& gt_value [[buffer(3)]],
                                  constant uint& count [[buffer(4)]],
                                  uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    const uint32_t idx = indices[gid];
    out_mask[gid] = (in[idx] > gt_value) ? 1 : 0;
}

kernel void filter_le_f32(const device float* in,
                          device uint8_t* out_mask,
                          constant float& le_value,
                          constant uint& row_count,
                          uint gid [[thread_position_in_grid]]) {
    if (gid >= row_count) return;
    out_mask[gid] = (in[gid] <= le_value) ? 1 : 0;
}

kernel void filter_le_f32_indexed(const device float* in [[buffer(0)]],
                                  const device uint32_t* indices [[buffer(1)]],
                                  device uint8_t* out_mask [[buffer(2)]],
                                  constant float& le_value [[buffer(3)]],
                                  constant uint& count [[buffer(4)]],
                                  uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    const uint32_t idx = indices[gid];
    out_mask[gid] = (in[idx] <= le_value) ? 1 : 0;
}

kernel void filter_ge_f32(const device float* in,
                          device uint8_t* out_mask,
                          constant float& ge_value,
                          constant uint& row_count,
                          uint gid [[thread_position_in_grid]]) {
    if (gid >= row_count) return;
    out_mask[gid] = (in[gid] >= ge_value) ? 1 : 0;
}

kernel void filter_ge_f32_indexed(const device float* in [[buffer(0)]],
                                  const device uint32_t* indices [[buffer(1)]],
                                  device uint8_t* out_mask [[buffer(2)]],
                                  constant float& ge_value [[buffer(3)]],
                                  constant uint& count [[buffer(4)]],
                                  uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    const uint32_t idx = indices[gid];
    out_mask[gid] = (in[idx] >= ge_value) ? 1 : 0;
}

kernel void u32mask_to_u8(const device uint* in_mask [[buffer(0)]],
                          device uint8_t* out_mask [[buffer(1)]],
                          constant uint& row_count [[buffer(2)]],
                          uint gid [[thread_position_in_grid]]) {
    if (gid >= row_count) return;
    out_mask[gid] = in_mask[gid] ? 1 : 0;
}

kernel void project_select_u32(const device uint32_t* in,
                               const device uint8_t* mask,
                               device uint32_t* out,
                               uint gid [[thread_position_in_grid]],
                               uint grid_size [[threads_per_grid]]) {
    if (gid >= grid_size) return;
    // Pass-through respecting mask (non-matching entries zeroed)
    out[gid] = mask[gid] ? in[gid] : 0u;
}

kernel void compact_indices(const device uint8_t* mask [[buffer(0)]],
                            device uint32_t* out_indices [[buffer(1)]],
                            device atomic_uint* out_count [[buffer(2)]],
                            constant uint& row_count [[buffer(3)]],
                            uint gid [[thread_position_in_grid]]) {
    if (gid >= row_count) return;
    if (mask[gid]) {
        uint idx = atomic_fetch_add_explicit(out_count, 1, memory_order_relaxed);
        out_indices[idx] = gid;
    }
}

// Compact with original indices - for indexed filters
kernel void compact_indices_indexed(const device uint8_t* mask [[buffer(0)]],
                                    const device uint32_t* in_indices [[buffer(1)]],
                                    device uint32_t* out_indices [[buffer(2)]],
                                    device atomic_uint* out_count [[buffer(3)]],
                                    constant uint& row_count [[buffer(4)]],
                                    uint gid [[thread_position_in_grid]]) {
    if (gid >= row_count) return;
    if (mask[gid]) {
        uint idx = atomic_fetch_add_explicit(out_count, 1, memory_order_relaxed);
        out_indices[idx] = in_indices[gid];  // Output the ORIGINAL index
    }
}

kernel void gather_col_f32(const device float* in_col [[buffer(0)]],
                           const device uint32_t* indices [[buffer(1)]],
                           device float* out_col [[buffer(2)]],
                           constant uint& count [[buffer(3)]],
                           uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    out_col[gid] = in_col[indices[gid]];
}

kernel void gather_col_u32(const device uint32_t* in_col [[buffer(0)]],
                           const device uint32_t* indices [[buffer(1)]],
                           device uint32_t* out_col [[buffer(2)]],
                           constant uint& count [[buffer(3)]],
                           uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    out_col[gid] = in_col[indices[gid]];
}

kernel void cast_u32_to_f32(const device uint32_t* in_col [[buffer(0)]],
                            device float* out_col [[buffer(1)]],
                            constant uint& count [[buffer(2)]],
                            uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    out_col[gid] = static_cast<float>(in_col[gid]);
}

kernel void compute_revenue_ep_disc(device const float* extendedprice [[buffer(0)]],
                                    device const float* discount [[buffer(1)]],
                                    device float* revenue [[buffer(2)]],
                                    constant uint& row_count [[buffer(3)]],
                                    uint gid [[thread_position_in_grid]]) {
    if (gid >= row_count) return;
    revenue[gid] = extendedprice[gid] * (1.0f - discount[gid]);
}

// Computes charge = l_extendedprice * (1 - l_discount) * (1 + l_tax)
kernel void compute_charge_ep_disc_tax(device const float* extendedprice [[buffer(0)]],
                                       device const float* discount [[buffer(1)]],
                                       device const float* tax [[buffer(2)]],
                                       device float* charge [[buffer(3)]],
                                       constant uint& row_count [[buffer(4)]],
                                       uint gid [[thread_position_in_grid]]) {
    if (gid >= row_count) return;
    charge[gid] = extendedprice[gid] * (1.0f - discount[gid]) * (1.0f + tax[gid]);
}

kernel void gather_col_int32(const device int* in_col [[buffer(0)]],
                             const device uint32_t* indices [[buffer(1)]],
                             device int* out_col [[buffer(2)]],
                             constant uint& count [[buffer(3)]],
                             uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    out_col[gid] = in_col[indices[gid]];
}

kernel void hash_build_u32(const device uint32_t* keys,
                           const device uint32_t* payloads,
                           device uint32_t* ht_keys,
                           device uint32_t* ht_vals,
                           constant uint32_t& capacity,
                           uint gid [[thread_position_in_grid]],
                           uint grid_size [[threads_per_grid]]) {
    if (gid >= grid_size) return;
    // Direct modulo placement, no collision handling
    uint32_t k = keys[gid];
    uint32_t v = payloads[gid];
    uint32_t slot = k % capacity;
    ht_keys[slot] = k;
    ht_vals[slot] = v;
}

kernel void hash_probe_u32(const device uint32_t* probe_keys,
                           const device uint32_t* ht_keys,
                           const device uint32_t* ht_vals,
                           device uint32_t* out_payload,
                           constant uint32_t& capacity,
                           uint gid [[thread_position_in_grid]],
                           uint grid_size [[threads_per_grid]]) {
    if (gid >= grid_size) return;
    // Single-slot probe
    uint32_t k = probe_keys[gid];
    uint32_t slot = k % capacity;
    out_payload[gid] = (ht_keys[slot] == k) ? ht_vals[slot] : 0u;
}

struct GroupByBucketF32 {
    atomic_uint key;
    atomic_uint count;
    atomic_uint sum_bits; // reinterpret float
};

inline float atomicLoadF32Bits(const device atomic_uint* a) {
    return as_type<float>(atomic_load_explicit((device atomic_uint*)a, memory_order_relaxed));
}

inline void atomicAddF32Bits(device atomic_uint* a, float v) {
    uint expected = atomic_load_explicit(a, memory_order_relaxed);
    while (true) {
        float cur = as_type<float>(expected);
        float nxt = cur + v;
        uint desired = as_type<uint>(nxt);
        if (atomic_compare_exchange_weak_explicit(a, &expected, desired, memory_order_relaxed, memory_order_relaxed)) {
            break;
        }
    }
}

inline void atomicMinF32Bits(device atomic_uint* a, float v) {
    uint expected = atomic_load_explicit(a, memory_order_relaxed);
    while (true) {
        float cur = as_type<float>(expected);
        if (!(v < cur)) break;
        uint desired = as_type<uint>(v);
        if (atomic_compare_exchange_weak_explicit(a, &expected, desired, memory_order_relaxed, memory_order_relaxed)) {
            break;
        }
    }
}

inline void atomicMaxF32Bits(device atomic_uint* a, float v) {
    uint expected = atomic_load_explicit(a, memory_order_relaxed);
    while (true) {
        float cur = as_type<float>(expected);
        if (!(v > cur)) break;
        uint desired = as_type<uint>(v);
        if (atomic_compare_exchange_weak_explicit(a, &expected, desired, memory_order_relaxed, memory_order_relaxed)) {
            break;
        }
    }
}

kernel void groupby_sum_f32(const device uint32_t* keys,
                            const device float* vals,
                            device atomic_uint* bucket_keys,
                            device atomic_uint* bucket_counts,
                            device atomic_uint* bucket_sumbits,
                            constant uint32_t& bucket_mask,
                            uint gid [[thread_position_in_grid]],
                            uint grid_size [[threads_per_grid]]) {
    if (gid >= grid_size) return;
    uint32_t k = keys[gid];
    float v = vals[gid];
    uint32_t slot = k & bucket_mask; // power-of-two buckets
    // Atomic insert/update: set key, increment count, accumulate sum
    atomic_store_explicit(&bucket_keys[slot], k, memory_order_relaxed);
    atomic_fetch_add_explicit(&bucket_counts[slot], 1u, memory_order_relaxed);
    atomicAddF32Bits(&bucket_sumbits[slot], v);
}

// Packed predicate clause (host must ensure alignment)
struct PredicateClause {
    uint colIndex;   // Column index among provided buffers
    uint op;         // 0:LT 1:LE 2:GT 3:GE 4:EQ
    uint isDate;     // 0 numeric/string, 1 date/int
    uint isString;   // 0 numeric/date, 1 string (hash comparison)
    uint isOrNext;   // 0 next clause is AND'd, 1 next clause is OR'd
    uint _pad;       // padding for alignment
    int64_t value;   // encoded literal (date as YYYYMMDD, float bits, or string hash)
};

// RPN expression token for arithmetic evaluation
struct ExprToken {
    uint type;       // 0:column_ref 1:literal 2:operator
    uint colIndex;   // if type==0, column index
    float literal;   // if type==1, literal value
    uint op;         // if type==2, operator: 0:+ 1:- 2:* 3:/
};

// Multi-column scan+filter+sum kernel
// Accepts up to 8 column buffers via [[buffer(0..7)]]
// Predicates reference columns by index (0..7)
// Target aggregation column is always buffer(0)
kernel void scan_filter_sum_f32(const device float* col0 [[buffer(0)]],
                                const device float* col1 [[buffer(1)]],
                                const device float* col2 [[buffer(2)]],
                                const device float* col3 [[buffer(3)]],
                                const device float* col4 [[buffer(4)]],
                                const device float* col5 [[buffer(5)]],
                                const device float* col6 [[buffer(6)]],
                                const device float* col7 [[buffer(7)]],
                                constant PredicateClause* clauses [[buffer(8)]],
                                constant uint& col_count [[buffer(9)]],
                                constant uint& clause_count [[buffer(10)]],
                                constant uint& row_count [[buffer(11)]],
                                device atomic_uint* out_sum_bits [[buffer(12)]],
                                uint gid [[thread_position_in_grid]],
                                uint tid [[thread_index_in_threadgroup]],
                                uint tgSize [[threads_per_threadgroup]]) {
    if (gid >= row_count) return;
    if (tgSize > 1024) tgSize = 1024;
    threadgroup float localVals[1024];

    // Build local column pointer array for dynamic indexing
    const device float* cols[8] = {col0, col1, col2, col3, col4, col5, col6, col7};
    
    // Target column for aggregation is always col0
    float target_val = cols[0][gid];
    
    // Evaluate predicates with dynamic column access and OR/AND logic
    bool passes = true;
    bool groupResult = true;
    
    for (uint c = 0; c < clause_count; ++c) {
        PredicateClause pc = clauses[c];
        if (pc.colIndex >= col_count) { passes = false; break; }
        
        float col_val = cols[pc.colIndex][gid];
        
        bool clauseResult;
        if (pc.isDate) {
            // Date stored as YYYYMMDD integer in column, compare as integers
            int date_val = as_type<int>(col_val);  // reinterpret float bits as int
            int date_lit = (int)(pc.value & 0xFFFFFFFFull);  // lower 32 bits
            switch (pc.op) {
                case 0: clauseResult = date_val < date_lit; break;
                case 1: clauseResult = date_val <= date_lit; break;
                case 2: clauseResult = date_val > date_lit; break;
                case 3: clauseResult = date_val >= date_lit; break;
                case 4: clauseResult = date_val == date_lit; break;
                default: clauseResult = false; break;
            }
        } else if (pc.isString) {
            // String comparison via hash - only equality supported
            uint col_hash = as_type<uint>(col_val);  // reinterpret float as hash
            uint lit_hash = (uint)(pc.value & 0xFFFFFFFFull);
            switch (pc.op) {
                case 4: clauseResult = (col_hash == lit_hash); break;  // Equal
                default: clauseResult = false; break;  // Other ops not supported for strings
            }
        } else {
            // Numeric comparison
            union { uint32_t u; float f; } conv; 
            conv.u = (uint32_t)(pc.value & 0xFFFFFFFFull);
            float lit = conv.f;
            switch (pc.op) {
                case 0: clauseResult = col_val < lit; break;
                case 1: clauseResult = col_val <= lit; break;
                case 2: clauseResult = col_val > lit; break;
                case 3: clauseResult = col_val >= lit; break;
                case 4: clauseResult = col_val == lit; break;
                default: clauseResult = false; break;
            }
        }
        
        if (c == 0) {
            groupResult = clauseResult;
        } else if (clauses[c-1].isOrNext) {
            groupResult = groupResult || clauseResult;
        } else {
            passes = passes && groupResult;
            if (!passes) break;
            groupResult = clauseResult;
        }
    }
    if (clause_count > 0) passes = passes && groupResult;
    
    localVals[tid] = (passes ? target_val : 0.0f);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Parallel reduction
    for (uint stride = tgSize >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            localVals[tid] += localVals[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        atomicAddF32Bits(out_sum_bits, localVals[0]);
    }
}

// Kernel: scan + filter + evaluate RPN expression + sum
// Supports arithmetic expressions like l_extendedprice * (1 - l_discount)
kernel void scan_filter_eval_sum(const device float* col0 [[buffer(0)]],
                                  const device float* col1 [[buffer(1)]],
                                  const device float* col2 [[buffer(2)]],
                                  const device float* col3 [[buffer(3)]],
                                  const device float* col4 [[buffer(4)]],
                                  const device float* col5 [[buffer(5)]],
                                  const device float* col6 [[buffer(6)]],
                                  const device float* col7 [[buffer(7)]],
                                  constant PredicateClause* clauses [[buffer(8)]],
                                  constant ExprToken* expr_rpn [[buffer(9)]],
                                  constant uint& col_count [[buffer(10)]],
                                  constant uint& clause_count [[buffer(11)]],
                                  constant uint& expr_length [[buffer(12)]],
                                  constant uint& row_count [[buffer(13)]],
                                  device atomic_uint* out_sum_bits [[buffer(14)]],
                                  uint gid [[thread_position_in_grid]],
                                  uint tid [[thread_index_in_threadgroup]],
                                  uint tgSize [[threads_per_threadgroup]]) {
    if (gid >= row_count) return;
    if (tgSize > 1024) tgSize = 1024;
    threadgroup float localVals[1024];
    
    const device float* cols[8] = {col0, col1, col2, col3, col4, col5, col6, col7};
    
    // Evaluate predicates first with OR/AND logic
    bool passes = true;
    bool groupResult = true;
    
    for (uint c = 0; c < clause_count; ++c) {
        PredicateClause pc = clauses[c];
        if (pc.colIndex >= col_count) { passes = false; break; }
        
        float col_val = cols[pc.colIndex][gid];
        
        bool clauseResult;
        if (pc.isDate) {
            int date_val = as_type<int>(col_val);
            int date_lit = (int)(pc.value & 0xFFFFFFFFull);
            switch (pc.op) {
                case 0: clauseResult = date_val < date_lit; break;
                case 1: clauseResult = date_val <= date_lit; break;
                case 2: clauseResult = date_val > date_lit; break;
                case 3: clauseResult = date_val >= date_lit; break;
                case 4: clauseResult = date_val == date_lit; break;
                default: clauseResult = false; break;
            }
        } else if (pc.isString) {
            uint col_hash = as_type<uint>(col_val);
            uint lit_hash = (uint)(pc.value & 0xFFFFFFFFull);
            switch (pc.op) {
                case 4: clauseResult = (col_hash == lit_hash); break;
                default: clauseResult = false; break;
            }
        } else {
            union { uint32_t u; float f; } conv;
            conv.u = (uint32_t)(pc.value & 0xFFFFFFFFull);
            float lit = conv.f;
            switch (pc.op) {
                case 0: clauseResult = col_val < lit; break;
                case 1: clauseResult = col_val <= lit; break;
                case 2: clauseResult = col_val > lit; break;
                case 3: clauseResult = col_val >= lit; break;
                case 4: clauseResult = col_val == lit; break;
                default: clauseResult = false; break;
            }
        }
        
        if (c == 0) {
            groupResult = clauseResult;
        } else if (clauses[c-1].isOrNext) {
            groupResult = groupResult || clauseResult;
        } else {
            passes = passes && groupResult;
            if (!passes) break;
            groupResult = clauseResult;
        }
    }
    if (clause_count > 0) passes = passes && groupResult;
    
    float result_val = 0.0f;
    if (passes) {
        // Evaluate RPN expression using stack
        float stack[32];  // Support expressions up to 32 tokens deep
        uint sp = 0;
        
        for (uint i = 0; i < expr_length; ++i) {
            ExprToken tok = expr_rpn[i];
            if (tok.type == 0) {
                // Column reference
                if (tok.colIndex < col_count && sp < 32) {
                    stack[sp++] = cols[tok.colIndex][gid];
                }
            } else if (tok.type == 1) {
                // Literal
                if (sp < 32) {
                    stack[sp++] = tok.literal;
                }
            } else if (tok.type == 2) {
                // Operator - pop two operands, apply, push result
                if (sp >= 2) {
                    float b = stack[--sp];
                    float a = stack[--sp];
                    float res = 0.0f;
                    switch (tok.op) {
                        case 0: res = a + b; break;  // ADD
                        case 1: res = a - b; break;  // SUB
                        case 2: res = a * b; break;  // MUL
                        case 3: res = (b != 0.0f) ? a / b : 0.0f; break;  // DIV
                    }
                    if (sp < 32) {
                        stack[sp++] = res;
                    }
                }
            }
        }
        
        // Final result is top of stack
        if (sp > 0) {
            result_val = stack[sp - 1];
        }
    }
    
    localVals[tid] = result_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Parallel reduction
    for (uint stride = tgSize >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            localVals[tid] += localVals[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        atomicAddF32Bits(out_sum_bits, localVals[0]);
    }
}

// Kernel: Evaluate RPN expression for each row and output result column
kernel void eval_expression_f32(const device float* col0 [[buffer(0)]],
                                const device float* col1 [[buffer(1)]],
                                const device float* col2 [[buffer(2)]],
                                const device float* col3 [[buffer(3)]],
                                constant ExprToken* expr_rpn [[buffer(4)]],
                                constant uint& col_count [[buffer(5)]],
                                constant uint& expr_length [[buffer(6)]],
                                constant uint& row_count [[buffer(7)]],
                                device float* out_col [[buffer(8)]],
                                uint gid [[thread_position_in_grid]]) {
    if (gid >= row_count) return;
    
    const device float* cols[4] = {col0, col1, col2, col3};
    
    float stack[16];
    uint sp = 0;
    
    for (uint i = 0; i < expr_length; ++i) {
        ExprToken tok = expr_rpn[i];
        if (tok.type == 0) { // Column
            if (tok.colIndex < col_count && sp < 16) {
                stack[sp++] = cols[tok.colIndex][gid];
            }
        } else if (tok.type == 1) { // Literal
            if (sp < 16) {
                stack[sp++] = tok.literal;
            }
        } else if (tok.type == 2) { // Operator
            if (sp >= 2) {
                float b = stack[--sp];
                float a = stack[--sp];
                float res = 0.0f;
                switch (tok.op) {
                    case 0: res = a + b; break;
                    case 1: res = a - b; break;
                    case 2: res = a * b; break;
                    case 3: res = (b != 0.0f) ? a / b : 0.0f; break;
                }
                if (sp < 16) stack[sp++] = res;
            }
        }
    }
    
    if (sp > 0) out_col[gid] = stack[sp - 1];
    else out_col[gid] = 0.0f;
}

// Bitonic sort kernel for ORDER BY support
kernel void bitonic_sort_step(device float* data [[buffer(0)]],
                               device uint* indices [[buffer(1)]],
                               constant uint& stage [[buffer(2)]],
                               constant uint& pass [[buffer(3)]],
                               constant uint& count [[buffer(4)]],
                               uint gid [[thread_position_in_grid]]) {
    uint pairDist = 1 << (stage - pass);
    uint blockWidth = 2 * pairDist;
    uint leftId = (gid % pairDist) + (gid / pairDist) * blockWidth;
    uint rightId = leftId + pairDist;
    
    if (rightId >= count) return;
    
    float leftVal = data[leftId];
    float rightVal = data[rightId];
    bool ascending = ((leftId / (1 << stage)) % 2) == 0;
    
    if ((leftVal > rightVal) == ascending) {
        // Swap
        data[leftId] = rightVal;
        data[rightId] = leftVal;
        uint tmpIdx = indices[leftId];
        indices[leftId] = indices[rightId];
        indices[rightId] = tmpIdx;
    }
}

// LIMIT kernel: copy first N elements
kernel void limit_copy(const device float* input [[buffer(0)]],
                       device float* output [[buffer(1)]],
                       constant uint& limit [[buffer(2)]],
                       uint gid [[thread_position_in_grid]]) {
    if (gid < limit) {
        output[gid] = input[gid];
    }
}

// Helper for ctz (count trailing zeros)
inline uint ctz(uint mask) {
    return __builtin_ctz(mask);
}

// Multi-column GROUP BY with multiple aggregates
// Supports up to 4 group key columns and 8 aggregate columns
kernel void groupby_agg_multi_key(const device uint* key_col0 [[buffer(0)]],
                                   const device uint* key_col1 [[buffer(1)]],
                                   const device uint* key_col2 [[buffer(2)]],
                                   const device uint* key_col3 [[buffer(3)]],
                                   const device float* agg_col0 [[buffer(4)]],
                                   const device float* agg_col1 [[buffer(5)]],
                                   const device float* agg_col2 [[buffer(6)]],
                                   const device float* agg_col3 [[buffer(7)]],
                                   const device float* agg_col4 [[buffer(8)]],
                                   const device float* agg_col5 [[buffer(9)]],
                                   const device float* agg_col6 [[buffer(10)]],
                                   const device float* agg_col7 [[buffer(11)]],
                                   device atomic_uint* ht_keys [[buffer(12)]],   // Flattened: capacity * 4 uint32s
                                   device atomic_uint* ht_agg_bits [[buffer(13)]], // Flattened: capacity * 8 floats as uint32
                                   constant uint& capacity [[buffer(14)]],
                                   constant uint& row_count [[buffer(15)]],
                                   constant uint& num_keys [[buffer(16)]],      // Number of group keys (1-4)
                                   constant uint& num_aggs [[buffer(17)]],      // Number of aggregates (1-8)
                                   uint gid [[thread_position_in_grid]],
                                   uint simd_lane_id [[thread_index_in_simdgroup]]) {
    if (gid >= row_count) return;

    constexpr uint IN_PROGRESS = 0xFFFFFFFFu;
    
    // Read composite key
    uint keys[4];
    keys[0] = (num_keys > 0) ? key_col0[gid] : 0;
    keys[1] = (num_keys > 1) ? key_col1[gid] : 0;
    keys[2] = (num_keys > 2) ? key_col2[gid] : 0;
    keys[3] = (num_keys > 3) ? key_col3[gid] : 0;
    
    // Read aggregate values
    float aggs[8];
    aggs[0] = (num_aggs > 0) ? agg_col0[gid] : 0.0f;
    aggs[1] = (num_aggs > 1) ? agg_col1[gid] : 0.0f;
    aggs[2] = (num_aggs > 2) ? agg_col2[gid] : 0.0f;
    aggs[3] = (num_aggs > 3) ? agg_col3[gid] : 0.0f;
    aggs[4] = (num_aggs > 4) ? agg_col4[gid] : 0.0f;
    aggs[5] = (num_aggs > 5) ? agg_col5[gid] : 0.0f;
    aggs[6] = (num_aggs > 6) ? agg_col6[gid] : 0.0f;
    aggs[7] = (num_aggs > 7) ? agg_col7[gid] : 0.0f;
    
    // SIMD Reduction Loop
    // Process lanes with identical keys together to reduce atomic contention
    bool done = false;
    ulong active_mask = (ulong)simd_ballot(true);
    
    // Iterate through all potential lanes in the SIMD group
    for (uint i = 0; i < 32; ++i) {
        // Lane i must be active to be a leader
        if (!((active_mask >> i) & 1)) continue;

        // Optimization: If all threads in the group are done, stop
        if (simd_all(done)) break;
        
        // Check if lane i is a valid leader candidate
        // It must be within the grid (gid < row_count) and not yet done
        uint leader_gid = simd_broadcast(gid, i);
        if (leader_gid >= row_count) continue; // Lane i is inactive/out of bounds
        
        bool leader_done = (bool)simd_broadcast((uint)done, i);
        if (leader_done) continue; // Lane i is already processed
        
        // Lane i is the leader for this round
        uint leader_keys[4];
        for(uint k=0; k<4; ++k) leader_keys[k] = simd_broadcast(keys[k], i);
        
        // Check if I match the leader
        bool match = !done; // Only participate if I'm not done
        for(uint k=0; k<num_keys; ++k) {
            if (keys[k] != leader_keys[k]) match = false;
        }
        
        // Sum aggregates for this group
        float group_sums[8];
        for (uint a=0; a<num_aggs; ++a) {
            float contribution = match ? aggs[a] : 0.0f;
            group_sums[a] = simd_sum(contribution);
        }

            // Optionally accumulate COUNT in ht_agg_bits[agg_base + 1] as a u32.
            // This avoids float precision issues for large counts.
            const uint group_count = (uint)simd_sum(match ? 1.0f : 0.0f);
        
        // Leader writes to global memory
        if (simd_lane_id == i) {
            // Compute hash for leader keys
            uint hash = 2166136261u;
            for (uint k = 0; k < num_keys; ++k) {
                hash ^= leader_keys[k];
                hash *= 16777619u;
            }
            uint slot = hash % capacity;
            
            // Linear probing
            for (uint probe = 0; probe < capacity; ++probe) {
                uint probe_slot = (slot + probe) % capacity;
                uint base_idx = probe_slot * 4;

                uint ht_k0 = atomic_load_explicit(&ht_keys[base_idx + 0], memory_order_relaxed);

                // Spin-wait if the slot is being initialized by another thread
                while (ht_k0 == IN_PROGRESS) {
                    ht_k0 = atomic_load_explicit(&ht_keys[base_idx + 0], memory_order_relaxed);
                }

                if (ht_k0 == 0u) {
                    uint expected = 0u;
                    if (atomic_compare_exchange_weak_explicit(&ht_keys[base_idx + 0], &expected, IN_PROGRESS,
                                                              memory_order_relaxed, memory_order_relaxed)) {
                        for (uint k = 1; k < num_keys; ++k) {
                            atomic_store_explicit(&ht_keys[base_idx + k], leader_keys[k], memory_order_relaxed);
                        }
                        // Store key0 last to signal completion (if we were using a flag, but here we use IN_PROGRESS)
                        // Actually we need to update key0 from IN_PROGRESS to real key
                        atomic_store_explicit(&ht_keys[base_idx + 0], leader_keys[0], memory_order_relaxed);

                        uint agg_base = probe_slot * 8;
                        for (uint a = 0; a < num_aggs; ++a) {
                            atomicAddF32Bits(&ht_agg_bits[agg_base + a], group_sums[a]);
                        }
                        if (num_aggs >= 2u) {
                            atomic_fetch_add_explicit(&ht_agg_bits[agg_base + 1], group_count, memory_order_relaxed);
                        }
                        break; // Done
                    }
                    // Lost race, retry slot
                    ht_k0 = atomic_load_explicit(&ht_keys[base_idx + 0], memory_order_relaxed);
                    while (ht_k0 == IN_PROGRESS) {
                        ht_k0 = atomic_load_explicit(&ht_keys[base_idx + 0], memory_order_relaxed);
                    }
                }

                if (ht_k0 == leader_keys[0]) {
                    bool key_match = true;
                    for (uint k = 1; k < num_keys; ++k) {
                        uint ht_k = atomic_load_explicit(&ht_keys[base_idx + k], memory_order_relaxed);
                        if (ht_k != leader_keys[k]) {
                            key_match = false;
                            break;
                        }
                    }
                    
                    if (key_match) {
                        uint agg_base = probe_slot * 8;
                        for (uint a = 0; a < num_aggs; ++a) {
                            atomicAddF32Bits(&ht_agg_bits[agg_base + a], group_sums[a]);
                        }
                        if (num_aggs >= 2u) {
                            atomic_fetch_add_explicit(&ht_agg_bits[agg_base + 1], group_count, memory_order_relaxed);
                        }
                        break; // Done
                    }
                }
            }
        }
        
        // Mark matching lanes as done
        if (match) done = true;
    }
}

// Multi-column GROUP BY COUNT(*)
// - Writes COUNT into ht_agg_bits[slot*16 + 0] as a u32 (16-slot stride to match groupby_agg_multi_key_typed).
// - Expects keys to already be biased such that key0!=0 (0 reserved as empty).
kernel void groupby_count_multi_key(const device uint* key_col0 [[buffer(0)]],
                                   const device uint* key_col1 [[buffer(1)]],
                                   const device uint* key_col2 [[buffer(2)]],
                                   const device uint* key_col3 [[buffer(3)]],
                                   device atomic_uint* ht_keys [[buffer(4)]],
                                   device atomic_uint* ht_agg_bits [[buffer(5)]],
                                   constant uint& capacity [[buffer(6)]],
                                   constant uint& row_count [[buffer(7)]],
                                   constant uint& num_keys [[buffer(8)]],
                                   uint gid [[thread_position_in_grid]],
                                   uint simd_lane_id [[thread_index_in_simdgroup]]) {
    if (gid >= row_count) return;

    constexpr uint IN_PROGRESS = 0xFFFFFFFFu;

    uint keys[4];
    keys[0] = (num_keys > 0) ? key_col0[gid] : 0;
    keys[1] = (num_keys > 1) ? key_col1[gid] : 0;
    keys[2] = (num_keys > 2) ? key_col2[gid] : 0;
    keys[3] = (num_keys > 3) ? key_col3[gid] : 0;

    bool done = false;
    for (uint i = 0; i < 32; ++i) {
        if (simd_all(done)) break;

        uint leader_gid = simd_broadcast(gid, i);
        if (leader_gid >= row_count) continue;
        bool leader_done = (bool)simd_broadcast((uint)done, i);
        if (leader_done) continue;

        uint leader_keys[4];
        for (uint k = 0; k < 4; ++k) leader_keys[k] = simd_broadcast(keys[k], i);

        bool match = !done;
        for (uint k = 0; k < num_keys; ++k) {
            if (keys[k] != leader_keys[k]) match = false;
        }

        const uint group_count = (uint)simd_sum(match ? 1.0f : 0.0f);

        if (simd_lane_id == i) {
            uint hash = 2166136261u;
            for (uint k = 0; k < num_keys; ++k) {
                hash ^= leader_keys[k];
                hash *= 16777619u;
            }
            uint slot = hash % capacity;

            for (uint probe = 0; probe < capacity; ++probe) {
                uint probe_slot = (slot + probe) % capacity;
                uint base_idx = probe_slot * 4;

                uint ht_k0 = atomic_load_explicit(&ht_keys[base_idx + 0], memory_order_relaxed);
                while (ht_k0 == IN_PROGRESS) {
                    ht_k0 = atomic_load_explicit(&ht_keys[base_idx + 0], memory_order_relaxed);
                }

                if (ht_k0 == 0u) {
                    uint expected = 0u;
                    if (atomic_compare_exchange_weak_explicit(&ht_keys[base_idx + 0], &expected, IN_PROGRESS,
                                                              memory_order_relaxed, memory_order_relaxed)) {
                        for (uint k = 1; k < num_keys; ++k) {
                            atomic_store_explicit(&ht_keys[base_idx + k], leader_keys[k], memory_order_relaxed);
                        }
                        atomic_store_explicit(&ht_keys[base_idx + 0], leader_keys[0], memory_order_relaxed);
                        uint agg_base = probe_slot * 16;
                        atomic_fetch_add_explicit(&ht_agg_bits[agg_base + 0], group_count, memory_order_relaxed);
                        break;
                    }
                    ht_k0 = atomic_load_explicit(&ht_keys[base_idx + 0], memory_order_relaxed);
                    while (ht_k0 == IN_PROGRESS) {
                        ht_k0 = atomic_load_explicit(&ht_keys[base_idx + 0], memory_order_relaxed);
                    }
                }

                if (ht_k0 == leader_keys[0]) {
                    bool key_match = true;
                    for (uint k = 1; k < num_keys; ++k) {
                        uint ht_k = atomic_load_explicit(&ht_keys[base_idx + k], memory_order_relaxed);
                        if (ht_k != leader_keys[k]) { key_match = false; break; }
                    }
                    if (key_match) {
                        uint agg_base = probe_slot * 16;
                        atomic_fetch_add_explicit(&ht_agg_bits[agg_base + 0], group_count, memory_order_relaxed);
                        break;
                    }
                }
            }
        }

        if (match) done = true;
    }
}

// Multi-column GROUP BY with typed aggregates.
// agg_types[a]:
//   0 = SUM(f32) using agg_col[a]
//   1 = COUNT(*) (u32) (ignores agg_col[a])
//   2 = MIN(f32) using agg_col[a]
//   3 = MAX(f32) using agg_col[a]
// Supports up to 8 group key columns and 16 aggregate columns (for queries like TPC-H Q1 with AVG)
kernel void groupby_agg_multi_key_typed(const device uint* key_col0 [[buffer(0)]],
                                       const device uint* key_col1 [[buffer(1)]],
                                       const device uint* key_col2 [[buffer(2)]],
                                       const device uint* key_col3 [[buffer(3)]],
                                       const device float* agg_col0 [[buffer(4)]],
                                       const device float* agg_col1 [[buffer(5)]],
                                       const device float* agg_col2 [[buffer(6)]],
                                       const device float* agg_col3 [[buffer(7)]],
                                       const device float* agg_col4 [[buffer(8)]],
                                       const device float* agg_col5 [[buffer(9)]],
                                       const device float* agg_col6 [[buffer(10)]],
                                       const device float* agg_col7 [[buffer(11)]],
                                       const device float* agg_col8 [[buffer(12)]],
                                       const device float* agg_col9 [[buffer(13)]],
                                       const device float* agg_col10 [[buffer(14)]],
                                       const device float* agg_col11 [[buffer(15)]],
                                       const device float* agg_col12 [[buffer(16)]],
                                       const device float* agg_col13 [[buffer(17)]],
                                       const device float* agg_col14 [[buffer(18)]],
                                       const device float* agg_col15 [[buffer(19)]],
                                       device atomic_uint* ht_keys [[buffer(20)]],
                                       device atomic_uint* ht_agg_bits [[buffer(21)]],
                                       constant uint& capacity [[buffer(22)]],
                                       constant uint& row_count [[buffer(23)]],
                                       constant uint& num_keys [[buffer(24)]],
                                       constant uint& num_aggs [[buffer(25)]],
                                       constant uint* agg_types [[buffer(26)]],
                                       const device uint* key_col4 [[buffer(27)]],
                                       const device uint* key_col5 [[buffer(28)]],
                                       const device uint* key_col6 [[buffer(29)]],
                                       const device uint* key_col7 [[buffer(30)]],
                                       uint gid [[thread_position_in_grid]],
                                       uint simd_lane_id [[thread_index_in_simdgroup]]) {
    if (gid >= row_count) return;

    constexpr uint IN_PROGRESS = 0xFFFFFFFFu;

    uint keys[8];
    keys[0] = (num_keys > 0) ? key_col0[gid] : 0;
    keys[1] = (num_keys > 1) ? key_col1[gid] : 0;
    keys[2] = (num_keys > 2) ? key_col2[gid] : 0;
    keys[3] = (num_keys > 3) ? key_col3[gid] : 0;
    keys[4] = (num_keys > 4) ? key_col4[gid] : 0;
    keys[5] = (num_keys > 5) ? key_col5[gid] : 0;
    keys[6] = (num_keys > 6) ? key_col6[gid] : 0;
    keys[7] = (num_keys > 7) ? key_col7[gid] : 0;

    const device float* agg_cols[16] = {agg_col0, agg_col1, agg_col2, agg_col3,
                                        agg_col4, agg_col5, agg_col6, agg_col7,
                                        agg_col8, agg_col9, agg_col10, agg_col11,
                                        agg_col12, agg_col13, agg_col14, agg_col15};

    float aggs[16];
    for (uint a = 0; a < 16; ++a) {
        if (a < num_aggs) {
            uint t = agg_types[a];
            if (t == 0u || t == 2u || t == 3u) {
                aggs[a] = agg_cols[a][gid];
            } else {
                aggs[a] = 0.0f;
            }
        } else {
            aggs[a] = 0.0f;
        }
    }

    bool done = false;
    for (uint i = 0; i < 32; ++i) {
        if (simd_all(done)) break;

        uint leader_gid = simd_broadcast(gid, i);
        if (leader_gid >= row_count) continue;
        bool leader_done = (bool)simd_broadcast((uint)done, i);
        if (leader_done) continue;

        uint leader_keys[8];
        for (uint k = 0; k < 8; ++k) leader_keys[k] = simd_broadcast(keys[k], i);

        bool match = !done;
        for (uint k = 0; k < num_keys; ++k) {
            if (keys[k] != leader_keys[k]) match = false;
        }

        const uint group_count = (uint)simd_sum(match ? 1.0f : 0.0f);

        float group_vals[16];
        for (uint a = 0; a < num_aggs; ++a) {
            uint t = agg_types[a];
            if (t == 0u) {
                float contribution = match ? aggs[a] : 0.0f;
                group_vals[a] = simd_sum(contribution);
            } else if (t == 2u) {
                float contribution = match ? aggs[a] : INFINITY;
                group_vals[a] = simd_min(contribution);
            } else if (t == 3u) {
                float contribution = match ? aggs[a] : -INFINITY;
                group_vals[a] = simd_max(contribution);
            } else {
                group_vals[a] = 0.0f;
            }
        }

        if (simd_lane_id == i) {
            uint hash = 2166136261u;
            for (uint k = 0; k < num_keys; ++k) {
                hash ^= leader_keys[k];
                hash *= 16777619u;
            }
            uint slot = hash % capacity;

            for (uint probe = 0; probe < capacity; ++probe) {
                uint probe_slot = (slot + probe) % capacity;
                uint base_idx = probe_slot * 8; // Stride 8

                uint ht_k0 = atomic_load_explicit(&ht_keys[base_idx + 0], memory_order_relaxed);
                while (ht_k0 == IN_PROGRESS) {
                    ht_k0 = atomic_load_explicit(&ht_keys[base_idx + 0], memory_order_relaxed);
                }

                if (ht_k0 == 0u) {
                    uint expected = 0u;
                    if (atomic_compare_exchange_weak_explicit(&ht_keys[base_idx + 0], &expected, IN_PROGRESS,
                                                              memory_order_relaxed, memory_order_relaxed)) {
                        for (uint k = 1; k < num_keys; ++k) {
                            atomic_store_explicit(&ht_keys[base_idx + k], leader_keys[k], memory_order_relaxed);
                        }
                        atomic_store_explicit(&ht_keys[base_idx + 0], leader_keys[0], memory_order_relaxed);

                        uint agg_base = probe_slot * 16;
                        for (uint a = 0; a < num_aggs; ++a) {
                            uint t = agg_types[a];
                            if (t == 0u) {
                                atomicAddF32Bits(&ht_agg_bits[agg_base + a], group_vals[a]);
                            } else if (t == 1u) {
                                atomic_fetch_add_explicit(&ht_agg_bits[agg_base + a], group_count, memory_order_relaxed);
                            } else if (t == 2u) {
                                atomic_store_explicit(&ht_agg_bits[agg_base + a], as_type<uint>(group_vals[a]), memory_order_relaxed);
                            } else if (t == 3u) {
                                atomic_store_explicit(&ht_agg_bits[agg_base + a], as_type<uint>(group_vals[a]), memory_order_relaxed);
                            }
                        }
                        break;
                    }
                    ht_k0 = atomic_load_explicit(&ht_keys[base_idx + 0], memory_order_relaxed);
                    while (ht_k0 == IN_PROGRESS) {
                        ht_k0 = atomic_load_explicit(&ht_keys[base_idx + 0], memory_order_relaxed);
                    }
                }

                if (ht_k0 == leader_keys[0]) {
                    bool key_match = true;
                    for (uint k = 1; k < num_keys; ++k) {
                        uint ht_k = atomic_load_explicit(&ht_keys[base_idx + k], memory_order_relaxed);
                        if (ht_k != leader_keys[k]) { key_match = false; break; }
                    }
                    if (key_match) {
                        uint agg_base = probe_slot * 16;
                        for (uint a = 0; a < num_aggs; ++a) {
                            uint t = agg_types[a];
                            if (t == 0u) {
                                atomicAddF32Bits(&ht_agg_bits[agg_base + a], group_vals[a]);
                            } else if (t == 1u) {
                                atomic_fetch_add_explicit(&ht_agg_bits[agg_base + a], group_count, memory_order_relaxed);
                            } else if (t == 2u) {
                                atomicMinF32Bits(&ht_agg_bits[agg_base + a], group_vals[a]);
                            } else if (t == 3u) {
                                atomicMaxF32Bits(&ht_agg_bits[agg_base + a], group_vals[a]);
                            }
                        }
                        break;
                    }
                }
            }
        }

        if (match) done = true;
    }
}

// Multi-key GROUP BY with aggregation
// Simplified version for single uint32 key
kernel void groupby_agg_single_key(const device uint* keys [[buffer(0)]],
                                    const device float* values [[buffer(1)]],
                                    device atomic_uint* ht_keys [[buffer(2)]],
                                    device atomic_uint* ht_counts [[buffer(3)]],
                                    device atomic_uint* ht_sum_bits [[buffer(4)]],
                                    constant uint& capacity [[buffer(5)]],
                                    constant uint& row_count [[buffer(6)]],
                                    uint gid [[thread_position_in_grid]]) {
    if (gid >= row_count) return;
    
    uint key = keys[gid];
    float val = values[gid];
    
    // Hash to slot
    uint slot = key % capacity;
    
    // Atomic insert/update (simplified, no collision handling)
    atomic_store_explicit(&ht_keys[slot], key, memory_order_relaxed);
    atomic_fetch_add_explicit(&ht_counts[slot], 1u, memory_order_relaxed);
    atomicAddF32Bits(&ht_sum_bits[slot], val);
}


// Hash join: Build phase
kernel void hash_join_build(const device uint* keys [[buffer(0)]],
                             const device uint* payloads [[buffer(1)]],
                             device atomic_uint* ht_keys [[buffer(2)]],
                             device atomic_uint* ht_payloads [[buffer(3)]],
                             constant uint& capacity [[buffer(4)]],
                             constant uint& build_count [[buffer(5)]],
                             uint gid [[thread_position_in_grid]]) {
    if (gid >= build_count) return;
    
    uint key = keys[gid];
    uint payload = payloads[gid];
    uint slot = key % capacity;
    
    // Linear probing to find empty slot
    for (uint i = 0; i < capacity; ++i) {
        uint probe_slot = (slot + i) % capacity;
        uint expected = 0;  // Empty slot marker
        
        // Try to claim this slot atomically
        if (atomic_compare_exchange_weak_explicit(&ht_keys[probe_slot], &expected, key,
                                                   memory_order_relaxed, memory_order_relaxed)) {
            // Successfully claimed slot, write payload
            atomic_store_explicit(&ht_payloads[probe_slot], payload, memory_order_relaxed);
            return;
        }
        
        // If slot has same key (duplicate), just update payload and return
        if (atomic_load_explicit(&ht_keys[probe_slot], memory_order_relaxed) == key) {
            atomic_store_explicit(&ht_payloads[probe_slot], payload, memory_order_relaxed);
            return;
        }
    }
}

// Hash join: Probe phase
kernel void hash_join_probe(const device uint* probe_keys [[buffer(0)]],
                             const device uint* ht_keys [[buffer(1)]],
                             const device uint* ht_payloads [[buffer(2)]],
                             device uint* output_matches [[buffer(3)]],
                             device uint* output_payloads [[buffer(4)]],
                             constant uint& capacity [[buffer(5)]],
                             constant uint& probe_count [[buffer(6)]],
                             uint gid [[thread_position_in_grid]]) {
    if (gid >= probe_count) return;
    
    uint key = probe_keys[gid];
    uint slot = key % capacity;
    
    // Linear probing to find matching key
    for (uint i = 0; i < capacity; ++i) {
        uint probe_slot = (slot + i) % capacity;
        uint ht_key = atomic_load_explicit((device atomic_uint*)&ht_keys[probe_slot], memory_order_relaxed);
        
        if (ht_key == key) {
            // Match found
            output_matches[gid] = 1;
            output_payloads[gid] = atomic_load_explicit((device atomic_uint*)&ht_payloads[probe_slot], memory_order_relaxed);
            return;
        }
        
        if (ht_key == 0) {
            // Empty slot found, key doesn't exist in hash table
            output_matches[gid] = 0;
            output_payloads[gid] = 0;
            return;
        }
    }
    
    // Shouldn't reach here unless table is completely full
    output_matches[gid] = 0;
    output_payloads[gid] = 0;
}

// Hash join (multi-match): build key->linked-list of build row indices.
// - Uses 0 as empty sentinel in ht_keys; stores (key+1) so key==0 is supported.
// - Uses 0 as null pointer in ht_head/next; stores (rowIndex+1) pointers.
kernel void hash_join_build_multi(const device uint* keys [[buffer(0)]],
                                 device atomic_uint* ht_keys [[buffer(1)]],
                                 device atomic_uint* ht_head [[buffer(2)]],
                                 device uint* next [[buffer(3)]],
                                 constant uint& capacity [[buffer(4)]],
                                 constant uint& build_count [[buffer(5)]],
                                 uint gid [[thread_position_in_grid]]) {
    if (gid >= build_count) return;

    const uint key_store = keys[gid] + 1u;
    uint slot = key_store % capacity;

    while (true) {
        uint expected = 0u;

        // Try to claim empty slot with this key (retry on spurious CAS failure).
        if (atomic_compare_exchange_weak_explicit(
                &ht_keys[slot], &expected, key_store,
                memory_order_relaxed, memory_order_relaxed)) {
            // We just claimed this empty slot for our key.
            // Push row onto the linked-list head.
            const uint old = atomic_exchange_explicit(
                &ht_head[slot], gid + 1u, memory_order_relaxed);
            next[gid] = old;
            return;
        }

        // CAS failed — expected now holds the current value of ht_keys[slot].
        if (expected == key_store) {
            // Same key — push onto the chain.
            const uint old = atomic_exchange_explicit(
                &ht_head[slot], gid + 1u, memory_order_relaxed);
            next[gid] = old;
            return;
        }

        if (expected == 0u) {
            // Spurious CAS failure on an empty slot — retry same slot.
            continue;
        }

        // Slot occupied by a different key — linear probe to next slot.
        slot = (slot + 1u) % capacity;
    }
}

kernel void hash_join_probe_count_multi(const device uint* probe_keys [[buffer(0)]],
                                       const device atomic_uint* ht_keys [[buffer(1)]],
                                       const device atomic_uint* ht_head [[buffer(2)]],
                                       const device uint* next [[buffer(3)]],
                                       device uint* out_counts [[buffer(4)]],
                                       constant uint& capacity [[buffer(5)]],
                                       constant uint& probe_count [[buffer(6)]],
                                       uint gid [[thread_position_in_grid]]) {
    if (gid >= probe_count) return;

    const uint key_store = probe_keys[gid] + 1u;
    const uint slot0 = key_store % capacity;
    uint count = 0u;

    for (uint i = 0; i < capacity; ++i) {
        const uint slot = (slot0 + i) % capacity;
        const uint ht_key = atomic_load_explicit(&ht_keys[slot], memory_order_relaxed);

        if (ht_key == key_store) {
            uint head = atomic_load_explicit(&ht_head[slot], memory_order_relaxed);
            while (head != 0u) {
                ++count;
                head = next[head - 1u];
            }
            break;
        }
        if (ht_key == 0u) break;
    }

    out_counts[gid] = count;
}

kernel void hash_join_probe_semi(const device uint* probe_keys [[buffer(0)]],
                                 const device atomic_uint* ht_keys [[buffer(1)]],
                                 constant uint& capacity [[buffer(2)]],
                                 constant uint& probe_count [[buffer(3)]],
                                 device uint8_t* out_mask [[buffer(4)]],
                                 uint gid [[thread_position_in_grid]]) {
    if (gid >= probe_count) return;
    const uint key_store = probe_keys[gid] + 1u;
    const uint slot0 = key_store % capacity;
    bool found = false;
    for (uint i = 0; i < capacity; ++i) {
        const uint slot = (slot0 + i) % capacity;
        const uint ht_key = atomic_load_explicit(&ht_keys[slot], memory_order_relaxed);
        if (ht_key == key_store) {
            found = true;
            break;
        }
        if (ht_key == 0u) break;
    }
    out_mask[gid] = found ? 1 : 0;
}

kernel void hash_join_probe_write_multi(const device uint* probe_keys [[buffer(0)]],
                                       const device atomic_uint* ht_keys [[buffer(1)]],
                                       const device atomic_uint* ht_head [[buffer(2)]],
                                       const device uint* next [[buffer(3)]],
                                       const device uint* offsets [[buffer(4)]],
                                       device uint* out_left [[buffer(5)]],
                                       device uint* out_right [[buffer(6)]],
                                       constant uint& capacity [[buffer(7)]],
                                       constant uint& probe_count [[buffer(8)]],
                                       uint gid [[thread_position_in_grid]]) {
    if (gid >= probe_count) return;

    const uint key_store = probe_keys[gid] + 1u;
    const uint slot0 = key_store % capacity;
    uint base = offsets[gid];
    uint k = 0u;

    for (uint i = 0; i < capacity; ++i) {
        const uint slot = (slot0 + i) % capacity;
        const uint ht_key = atomic_load_explicit(&ht_keys[slot], memory_order_relaxed);

        if (ht_key == key_store) {
            uint head = atomic_load_explicit(&ht_head[slot], memory_order_relaxed);
            while (head != 0u) {
                out_left[base + k] = gid;
                out_right[base + k] = head - 1u;
                ++k;
                head = next[head - 1u];
            }
            break;
        }
        if (ht_key == 0u) break;
    }
}

// Generic aggregation kernel supporting COUNT, SUM, AVG, MIN, MAX
// aggType: 0=COUNT, 1=SUM, 2=AVG, 3=MIN, 4=MAX
kernel void scan_filter_aggregate(const device float* col0 [[buffer(0)]],
                                  const device float* col1 [[buffer(1)]],
                                  const device float* col2 [[buffer(2)]],
                                  const device float* col3 [[buffer(3)]],
                                  const device float* col4 [[buffer(4)]],
                                  const device float* col5 [[buffer(5)]],
                                  const device float* col6 [[buffer(6)]],
                                  const device float* col7 [[buffer(7)]],
                                  constant PredicateClause* clauses [[buffer(8)]],
                                  constant uint& col_count [[buffer(9)]],
                                  constant uint& clause_count [[buffer(10)]],
                                  constant uint& row_count [[buffer(11)]],
                                  constant uint& aggType [[buffer(12)]],
                                  device atomic_uint* out_result_bits [[buffer(13)]],
                                  device atomic_uint* out_count [[buffer(14)]],
                                  uint gid [[thread_position_in_grid]],
                                  uint tid [[thread_index_in_threadgroup]],
                                  uint tgSize [[threads_per_threadgroup]]) {
    if (gid >= row_count) return;
    if (tgSize > 1024) tgSize = 1024;
    
    threadgroup float localVals[1024];
    threadgroup uint localCounts[1024];
    
    const device float* cols[8] = {col0, col1, col2, col3, col4, col5, col6, col7};
    float target_val = cols[0][gid];
    
    // Evaluate predicates with OR/AND logic
    // Group consecutive clauses connected by OR, evaluate groups with AND
    bool passes = true;
    bool groupResult = true;
    
    for (uint c = 0; c < clause_count; ++c) {
        PredicateClause pc = clauses[c];
        if (pc.colIndex >= col_count) { passes = false; break; }
        float col_val = cols[pc.colIndex][gid];
        
        bool clauseResult;
        if (pc.isDate) {
            int date_val = as_type<int>(col_val);
            int date_lit = (int)(pc.value & 0xFFFFFFFFull);
            switch (pc.op) {
                case 0: clauseResult = date_val < date_lit; break;
                case 1: clauseResult = date_val <= date_lit; break;
                case 2: clauseResult = date_val > date_lit; break;
                case 3: clauseResult = date_val >= date_lit; break;
                case 4: clauseResult = date_val == date_lit; break;
                default: clauseResult = false; break;
            }
        } else {
            union { uint32_t u; float f; } conv;
            conv.u = (uint32_t)(pc.value & 0xFFFFFFFFull);
            float lit = conv.f;
            switch (pc.op) {
                case 0: clauseResult = col_val < lit; break;
                case 1: clauseResult = col_val <= lit; break;
                case 2: clauseResult = col_val > lit; break;
                case 3: clauseResult = col_val >= lit; break;
                case 4: clauseResult = col_val == lit; break;
                default: clauseResult = false; break;
            }
        }
        
        if (c == 0) {
            groupResult = clauseResult;
        } else if (clauses[c-1].isOrNext) {
            // Previous clause was OR'd with this one
            groupResult = groupResult || clauseResult;
        } else {
            // Previous clause was AND'd - finish previous group
            passes = passes && groupResult;
            if (!passes) break; // Short circuit
            groupResult = clauseResult;
        }
    }
    // Don't forget the last group
    if (clause_count > 0) passes = passes && groupResult;
    
    // Initialize local values based on aggregation type
    if (aggType == 0) {
        // COUNT
        localVals[tid] = passes ? 1.0f : 0.0f;
        localCounts[tid] = passes ? 1 : 0;
    } else if (aggType == 3) {
        // MIN
        localVals[tid] = passes ? target_val : FLT_MAX;
    } else if (aggType == 4) {
        // MAX
        localVals[tid] = passes ? target_val : -FLT_MAX;
    } else {
        // SUM or AVG
        localVals[tid] = passes ? target_val : 0.0f;
        localCounts[tid] = passes ? 1 : 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Parallel reduction
    for (uint stride = tgSize >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (aggType == 3) {
                // MIN
                localVals[tid] = min(localVals[tid], localVals[tid + stride]);
            } else if (aggType == 4) {
                // MAX
                localVals[tid] = max(localVals[tid], localVals[tid + stride]);
            } else {
                // COUNT, SUM, AVG
                localVals[tid] += localVals[tid + stride];
                if (aggType == 0 || aggType == 2) {
                    localCounts[tid] += localCounts[tid + stride];
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Thread 0 in each threadgroup atomically updates global result
    if (tid == 0) {
        union { uint32_t u; float f; } conv;
        conv.f = localVals[0];
        
        if (aggType == 3) {
            // MIN: atomic min
            uint expected = atomic_load_explicit(out_result_bits, memory_order_relaxed);
            while (true) {
                union { uint32_t u; float f; } current;
                current.u = expected;
                float new_val = min(current.f, conv.f);
                union { uint32_t u; float f; } new_conv;
                new_conv.f = new_val;
                if (atomic_compare_exchange_weak_explicit(out_result_bits, &expected, new_conv.u,
                                                          memory_order_relaxed, memory_order_relaxed)) {
                    break;
                }
            }
        } else if (aggType == 4) {
            // MAX: atomic max
            uint expected = atomic_load_explicit(out_result_bits, memory_order_relaxed);
            while (true) {
                union { uint32_t u; float f; } current;
                current.u = expected;
                float new_val = max(current.f, conv.f);
                union { uint32_t u; float f; } new_conv;
                new_conv.f = new_val;
                if (atomic_compare_exchange_weak_explicit(out_result_bits, &expected, new_conv.u,
                                                          memory_order_relaxed, memory_order_relaxed)) {
                    break;
                }
            }
        } else {
            // COUNT, SUM, AVG: atomic add
            atomicAddF32Bits(out_result_bits, conv.f);
        }
        
        if (aggType == 0 || aggType == 2) {
            atomic_fetch_add_explicit(out_count, localCounts[0], memory_order_relaxed);
        }
    }
}

// ============================================================================
// Col-vs-Col Kernels
// ============================================================================

kernel void filter_col_col_u32(const device uint32_t* a [[buffer(0)]],
                               const device uint32_t* b [[buffer(1)]],
                               device uint8_t* out_mask [[buffer(2)]],
                               constant int& op [[buffer(3)]],
                               constant uint& count [[buffer(4)]],
                               uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    uint32_t va = a[gid];
    uint32_t vb = b[gid];
    bool pass = false;
    switch(op) {
        case 0: pass = (va == vb); break; // Eq
        case 1: pass = (va != vb); break; // Ne
        case 2: pass = (va < vb); break;  // Lt
        case 3: pass = (va <= vb); break; // Le
        case 4: pass = (va > vb); break;  // Gt
        case 5: pass = (va >= vb); break; // Ge
    }
    out_mask[gid] = pass ? 1 : 0;
}

kernel void filter_col_col_f32(const device float* a [[buffer(0)]],
                               const device float* b [[buffer(1)]],
                               device uint8_t* out_mask [[buffer(2)]],
                               constant int& op [[buffer(3)]],
                               constant uint& count [[buffer(4)]],
                               uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    float va = a[gid];
    float vb = b[gid];
    bool pass = false;
    switch(op) {
        case 0: pass = (va == vb); break; // Eq
        case 1: pass = (va != vb); break; // Ne
        case 2: pass = (va < vb); break;  // Lt
        case 3: pass = (va <= vb); break; // Le
        case 4: pass = (va > vb); break;  // Gt
        case 5: pass = (va >= vb); break; // Ge
    }
    out_mask[gid] = pass ? 1 : 0;
}

// NE Kernels
kernel void filter_ne_u32(const device uint32_t* in [[buffer(0)]],
                          device uint8_t* out_mask [[buffer(1)]],
                          constant uint32_t& ne_value [[buffer(2)]],
                          constant uint& row_count [[buffer(3)]],
                          uint gid [[thread_position_in_grid]]) {
    if (gid >= row_count) return;
    out_mask[gid] = (in[gid] != ne_value) ? 1 : 0;
}

kernel void filter_ne_u32_indexed(const device uint32_t* in [[buffer(0)]],
                                  const device uint32_t* indices [[buffer(1)]],
                                  device uint8_t* out_mask [[buffer(2)]],
                                  constant uint32_t& ne_value [[buffer(3)]],
                                  constant uint& count [[buffer(4)]],
                                  uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    const uint32_t idx = indices[gid];
    out_mask[gid] = (in[idx] != ne_value) ? 1 : 0;
}

kernel void filter_ne_f32(const device float* in [[buffer(0)]],
                          device uint8_t* out_mask [[buffer(1)]],
                          constant float& ne_value [[buffer(2)]],
                          constant uint& row_count [[buffer(3)]],
                          uint gid [[thread_position_in_grid]]) {
    if (gid >= row_count) return;
    out_mask[gid] = (in[gid] != ne_value) ? 1 : 0;
}

kernel void filter_ne_f32_indexed(const device float* in [[buffer(0)]],
                                  const device uint32_t* indices [[buffer(1)]],
                                  device uint8_t* out_mask [[buffer(2)]],
                                  constant float& ne_value [[buffer(3)]],
                                  constant uint& count [[buffer(4)]],
                                  uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    const uint32_t idx = indices[gid];
    out_mask[gid] = (in[idx] != ne_value) ? 1 : 0;
}

kernel void select_u32(const device uint32_t* mask [[buffer(0)]],
                       const device uint32_t* t [[buffer(1)]],
                       const device uint32_t* f [[buffer(2)]],
                       device uint32_t* out [[buffer(3)]],
                       constant uint& count [[buffer(4)]],
                       uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    out[gid] = mask[gid] ? t[gid] : f[gid];
}

// --- New Generic Kernels [Copilot] ---

kernel void cmp_col_col_u32_mask(
    const device uint32_t* colA [[buffer(0)]],
    const device uint32_t* colB [[buffer(1)]],
    device uint32_t* outMask [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    constant int& op [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    uint32_t a = colA[gid];
    uint32_t b = colB[gid];
    bool p = false;
    switch(op) {
        case 0: p = (a < b); break;  // LT
        case 1: p = (a <= b); break; // LE
        case 2: p = (a > b); break;  // GT
        case 3: p = (a >= b); break; // GE
        case 4: p = (a == b); break; // EQ
        case 5: p = (a != b); break; // NE
    }
    outMask[gid] = p ? 1 : 0;
}

kernel void cmp_col_lit_u32_mask(
    const device uint32_t* colA [[buffer(0)]],
    constant uint32_t& valB [[buffer(1)]],
    device uint32_t* outMask [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    constant int& op [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    uint32_t a = colA[gid];
    uint32_t b = valB;
    bool p = false;
    switch(op) {
        case 0: p = (a < b); break;
        case 1: p = (a <= b); break;
        case 2: p = (a > b); break;
        case 3: p = (a >= b); break;
        case 4: p = (a == b); break;
        case 5: p = (a != b); break;
    }
    outMask[gid] = p ? 1 : 0;
}

kernel void logic_or_u32(
    const device uint32_t* colA [[buffer(0)]],
    const device uint32_t* colB [[buffer(1)]],
    device uint32_t* outMask [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    outMask[gid] = (colA[gid] || colB[gid]) ? 1 : 0;
}

kernel void logic_and_u32(
    const device uint32_t* colA [[buffer(0)]],
    const device uint32_t* colB [[buffer(1)]],
    device uint32_t* outMask [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    outMask[gid] = (colA[gid] && colB[gid]) ? 1 : 0;
}

// Convert index array to bitmask: mask[indices[i]] = 1
kernel void indices_to_mask(
    const device uint32_t* indices [[buffer(0)]],
    device uint32_t* mask [[buffer(1)]],
    constant uint& indexCount [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= indexCount) return;
    uint32_t idx = indices[gid];
    mask[idx] = 1;
}

// Fill buffer with constant u32 value
kernel void fill_u32(
    device uint32_t* buf [[buffer(0)]],
    constant uint32_t& val [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    buf[gid] = val;
}

// Fill buffer with constant f32 value
kernel void fill_f32(
    device float* buf [[buffer(0)]],
    constant float& val [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    buf[gid] = val;
}

// Generate sequence 0, 1, 2, ... (iota)
kernel void iota_u32(
    device uint32_t* buf [[buffer(0)]],
    constant uint& count [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    buf[gid] = gid;
}

// Cross product: generate pairs (left[i], right[j]) for all i, j
kernel void cross_product(
    const device uint32_t* left [[buffer(0)]],
    const device uint32_t* right [[buffer(1)]],
    device uint32_t* outLeft [[buffer(2)]],
    device uint32_t* outRight [[buffer(3)]],
    constant uint& leftCount [[buffer(4)]],
    constant uint& rightCount [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
    uint totalCount = leftCount * rightCount;
    if (gid >= totalCount) return;
    
    uint i = gid / rightCount;  // left index
    uint j = gid % rightCount;  // right index
    
    outLeft[gid] = left[i];
    outRight[gid] = right[j];
}

// Clear mask buffer to zeros
kernel void clear_mask(
    device uint32_t* mask [[buffer(0)]],
    constant uint& count [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    mask[gid] = 0;
}

// Invert a mask: outMask[i] = !inMask[i]
kernel void logic_not_u32(
    const device uint32_t* inMask [[buffer(0)]],
    device uint32_t* outMask [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    outMask[gid] = inMask[gid] ? 0 : 1;
}

// AND-NOT: outMask[i] = colA[i] && !colB[i] (for set difference: A - B)
kernel void logic_andnot_u32(
    const device uint32_t* colA [[buffer(0)]],
    const device uint32_t* colB [[buffer(1)]],
    device uint32_t* outMask [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    outMask[gid] = (colA[gid] && !colB[gid]) ? 1 : 0;
}

// Compact u32 mask to indices: for each mask[i] != 0, output i
kernel void compact_u32_mask(
    const device uint32_t* mask [[buffer(0)]],
    device uint32_t* out_indices [[buffer(1)]],
    device atomic_uint* out_count [[buffer(2)]],
    constant uint& row_count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= row_count) return;
    if (mask[gid]) {
        uint idx = atomic_fetch_add_explicit(out_count, 1, memory_order_relaxed);
        out_indices[idx] = gid;
    }
}

kernel void filter_range_to_mask_f32_indexed(const device float* col [[buffer(0)]],
                                             const device uint32_t* indices [[buffer(1)]],
                                             device uint8_t* out_mask [[buffer(2)]],
                                             constant float& min_val [[buffer(3)]],
                                             constant float& max_val [[buffer(4)]],
                                             constant uint& count [[buffer(5)]],
                                             uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    const uint32_t idx = indices[gid];
    float val = col[idx];
    out_mask[gid] = (val >= min_val && val <= max_val) ? 1 : 0;
}

kernel void filter_range_to_mask_u32_indexed(const device uint32_t* col [[buffer(0)]],
                                             const device uint32_t* indices [[buffer(1)]],
                                             device uint8_t* out_mask [[buffer(2)]],
                                             constant uint32_t& min_val [[buffer(3)]],
                                             constant uint32_t& max_val [[buffer(4)]],
                                             constant uint& count [[buffer(5)]],
                                             uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    const uint32_t idx = indices[gid];
    uint32_t val = col[idx];
    out_mask[gid] = (val >= min_val && val <= max_val) ? 1 : 0;
}

// Generic Arithmetic Kernels
kernel void arith_mul_f32_col_col(const device float* colA [[buffer(0)]],
                                  const device float* colB [[buffer(1)]],
                                  device float* out [[buffer(2)]],
                                  constant uint& count [[buffer(3)]],
                                  uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    out[gid] = colA[gid] * colB[gid];
}

kernel void arith_mul_f32_col_col_indexed(const device float* colA [[buffer(0)]],
                                          const device float* colB [[buffer(1)]],
                                          const device uint32_t* indices [[buffer(2)]],
                                          device float* out [[buffer(3)]],
                                          constant uint& count [[buffer(4)]],
                                          uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    uint32_t idx = indices[gid];
    out[gid] = colA[idx] * colB[idx];
}

kernel void arith_mul_f32_col_scalar(const device float* colA [[buffer(0)]],
                                     constant float& valB [[buffer(1)]],
                                     device float* out [[buffer(2)]],
                                     constant uint& count [[buffer(3)]],
                                     uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    out[gid] = colA[gid] * valB;
}

kernel void arith_mul_f32_col_scalar_indexed(const device float* colA [[buffer(0)]],
                                             constant float& valB [[buffer(1)]],
                                             const device uint32_t* indices [[buffer(2)]],
                                             device float* out [[buffer(3)]],
                                             constant uint& count [[buffer(4)]],
                                             uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    uint32_t idx = indices[gid];
    out[gid] = colA[idx] * valB;
}

kernel void arith_div_f32_col_col(const device float* colA [[buffer(0)]],
                                  const device float* colB [[buffer(1)]],
                                  device float* out [[buffer(2)]],
                                  constant uint& count [[buffer(3)]],
                                  uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    float denom = colB[gid];
    out[gid] = (abs(denom) > 1e-9) ? (colA[gid] / denom) : 0.0f; // Handle DivByZero?
}

kernel void arith_div_f32_col_scalar(const device float* colA [[buffer(0)]],
                                     constant float& valB [[buffer(1)]],
                                     device float* out [[buffer(2)]],
                                     constant uint& count [[buffer(3)]],
                                     uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    float denom = valB;
    out[gid] = (abs(denom) > 1e-9) ? (colA[gid] / denom) : 0.0f;
}

kernel void arith_div_f32_scalar_col(const device float* valA [[buffer(0)]],
                                     const device float* colB [[buffer(1)]],
                                     device float* out [[buffer(2)]],
                                     constant uint& count [[buffer(3)]],
                                     uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    float denom = colB[gid];
    out[gid] = (abs(denom) > 1e-9) ? (*valA / denom) : 0.0f;
}

// Reduction Kernels
kernel void reduce_sum_f32(const device float* in [[buffer(0)]],
                           device atomic_uint* out [[buffer(1)]],
                           constant uint& count [[buffer(2)]],
                           uint gid [[thread_position_in_grid]],
                           uint simd_lane_id [[thread_index_in_simdgroup]]) {
    float val = (gid < count) ? in[gid] : 0.0f;
    float sum = simd_sum(val);
    
    if (simd_lane_id == 0) {
        atomicAddF32Bits(out, sum);
    }
}

kernel void reduce_max_f32(const device float* in [[buffer(0)]],
                           device atomic_uint* out [[buffer(1)]],
                           constant uint& count [[buffer(2)]],
                           uint gid [[thread_position_in_grid]],
                           uint simd_lane_id [[thread_index_in_simdgroup]]) {
    float val = (gid < count) ? in[gid] : -MAXFLOAT;
    float m = simd_max(val);
    
    if (simd_lane_id == 0) {
        atomicMaxF32Bits(out, m);
    }
}

kernel void reduce_min_f32(const device float* in [[buffer(0)]],
                           device atomic_uint* out [[buffer(1)]],
                           constant uint& count [[buffer(2)]],
                           uint gid [[thread_position_in_grid]],
                           uint simd_lane_id [[thread_index_in_simdgroup]]) {
    float val = (gid < count) ? in[gid] : MAXFLOAT;
    float m = simd_min(val);
    
    if (simd_lane_id == 0) {
        atomicMinF32Bits(out, m);
    }
}


// ---- Arrow-style string kernels ----
// All string kernels use Arrow-style offsets: offsets[gid+1] - offsets[gid] = length.
// offsets buffer has (row_count + 1) elements. No separate lengths buffer.

kernel void filter_string_prefix(const device char* chars [[buffer(0)]],
                                   const device uint32_t* offsets [[buffer(1)]],
                                   device uint8_t* out_mask [[buffer(2)]],
                                   const device char* pattern [[buffer(3)]],
                                   constant uint& pattern_len [[buffer(4)]],
                                   constant uint& row_count [[buffer(5)]],
                                   uint gid [[thread_position_in_grid]]) {
    if (gid >= row_count) return;
    
    uint start = offsets[gid];
    uint len = offsets[gid + 1] - start;
    const device char* str = chars + start;
    
    if (pattern_len == 0) {
        out_mask[gid] = 1;
        return;
    }

    if (pattern_len > len) {
        out_mask[gid] = 0;
        return;
    }
    
    bool match = true;
    for (uint j = 0; j < pattern_len; ++j) {
        if (str[j] != pattern[j]) {
            match = false;
            break;
        }
    }
    out_mask[gid] = match ? 1 : 0;
}

kernel void filter_string_not_prefix(const device char* chars [[buffer(0)]],
                                   const device uint32_t* offsets [[buffer(1)]],
                                   device uint8_t* out_mask [[buffer(2)]],
                                   const device char* pattern [[buffer(3)]],
                                   constant uint& pattern_len [[buffer(4)]],
                                   constant uint& row_count [[buffer(5)]],
                                   uint gid [[thread_position_in_grid]]) {
    if (gid >= row_count) return;
    
    uint start = offsets[gid];
    uint len = offsets[gid + 1] - start;
    const device char* str = chars + start;
    
    if (pattern_len == 0) {
        out_mask[gid] = 0;
        return;
    }

    if (pattern_len > len) {
        out_mask[gid] = 1;
        return;
    }
    
    bool match = true;
    for (uint j = 0; j < pattern_len; ++j) {
        if (str[j] != pattern[j]) {
            match = false;
            break;
        }
    }
    out_mask[gid] = match ? 0 : 1;
}

kernel void filter_string_contains(const device char* chars [[buffer(0)]],
                                   const device uint32_t* offsets [[buffer(1)]],
                                   device uint8_t* out_mask [[buffer(2)]],
                                   const device char* pattern [[buffer(3)]],
                                   constant uint& pattern_len [[buffer(4)]],
                                   constant uint& row_count [[buffer(5)]],
                                   uint gid [[thread_position_in_grid]]) {
    if (gid >= row_count) return;
    
    uint start = offsets[gid];
    uint len = offsets[gid + 1] - start;
    const device char* str = chars + start;
    
    if (pattern_len == 0) {
        out_mask[gid] = 1; // Empty pattern matches everything
        return;
    }

    if (pattern_len > len) {
        out_mask[gid] = 0;
        return;
    }
    
    // Brute-force substring search
    bool found = false;
    for (uint i = 0; i <= len - pattern_len; ++i) {
        bool match = true;
        for (uint j = 0; j < pattern_len; ++j) {
            if (str[i + j] != pattern[j]) {
                match = false;
                break;
            }
        }
        if (match) {
            found = true;
            break;
        }
    }
    out_mask[gid] = found ? 1 : 0;
}

// Multi-wildcard contains: pattern segments are packed into one buffer,
// with their offsets/lengths in separate arrays.
// Matches %seg0%seg1%...%segN% — each segment must be found in order.
// Data offsets are Arrow-style (N+1). Pattern offsets/lengths are NOT Arrow-style.
kernel void filter_string_multi_contains(const device char* chars [[buffer(0)]],
                                         const device uint32_t* offsets [[buffer(1)]],
                                         device uint8_t* out_mask [[buffer(2)]],
                                         const device char* patterns [[buffer(3)]],      // packed segments
                                         const device uint32_t* pat_offsets [[buffer(4)]],
                                         const device uint32_t* pat_lengths [[buffer(5)]],
                                         constant uint& num_segments [[buffer(6)]],
                                         constant uint& row_count [[buffer(7)]],
                                         uint gid [[thread_position_in_grid]]) {
    if (gid >= row_count) return;

    uint str_start = offsets[gid];
    uint str_len   = offsets[gid + 1] - str_start;
    const device char* str = chars + str_start;

    uint search_from = 0;
    bool all_found = true;

    for (uint s = 0; s < num_segments; ++s) {
        uint poff = pat_offsets[s];
        uint plen = pat_lengths[s];
        if (plen == 0) continue;

        if (search_from + plen > str_len) { all_found = false; break; }

        bool seg_found = false;
        for (uint i = search_from; i <= str_len - plen; ++i) {
            bool match = true;
            for (uint j = 0; j < plen; ++j) {
                if (str[i + j] != patterns[poff + j]) { match = false; break; }
            }
            if (match) {
                search_from = i + plen;   // next segment must appear after this one
                seg_found = true;
                break;
            }
        }
        if (!seg_found) { all_found = false; break; }
    }
    out_mask[gid] = all_found ? 1 : 0;
}

// Str compare: < 0 if s1 < s2, 0 if eq, > 0 if s1 > s2
inline int compare_str(const device char* s1, uint len1, const device char* s2, uint len2) {
    uint len = len1 < len2 ? len1 : len2;
    for (uint i = 0; i < len; ++i) {
        if (s1[i] < s2[i]) return -1;
        if (s1[i] > s2[i]) return 1;
    }
    if (len1 < len2) return -1;
    if (len1 > len2) return 1;
    return 0;
}

kernel void filter_string_eq(const device char* chars [[buffer(0)]],
                                   const device uint32_t* offsets [[buffer(1)]],
                                   device uint8_t* out_mask [[buffer(2)]],
                                   const device char* pattern [[buffer(3)]],
                                   constant uint& pattern_len [[buffer(4)]],
                                   constant uint& row_count [[buffer(5)]],
                                   uint gid [[thread_position_in_grid]]) {
    if (gid >= row_count) return;
    uint start = offsets[gid];
    uint len = offsets[gid + 1] - start;
    const device char* str = chars + start;
    out_mask[gid] = (compare_str(str, len, pattern, pattern_len) == 0) ? 1 : 0;
}

kernel void filter_string_ne(const device char* chars [[buffer(0)]],
                                   const device uint32_t* offsets [[buffer(1)]],
                                   device uint8_t* out_mask [[buffer(2)]],
                                   const device char* pattern [[buffer(3)]],
                                   constant uint& pattern_len [[buffer(4)]],
                                   constant uint& row_count [[buffer(5)]],
                                   uint gid [[thread_position_in_grid]]) {
    if (gid >= row_count) return;
    uint start = offsets[gid];
    uint len = offsets[gid + 1] - start;
    const device char* str = chars + start;
    out_mask[gid] = (compare_str(str, len, pattern, pattern_len) != 0) ? 1 : 0;
}

kernel void filter_string_lt(const device char* chars [[buffer(0)]],
                                   const device uint32_t* offsets [[buffer(1)]],
                                   device uint8_t* out_mask [[buffer(2)]],
                                   const device char* pattern [[buffer(3)]],
                                   constant uint& pattern_len [[buffer(4)]],
                                   constant uint& row_count [[buffer(5)]],
                                   uint gid [[thread_position_in_grid]]) {
    if (gid >= row_count) return;
    uint start = offsets[gid];
    uint len = offsets[gid + 1] - start;
    const device char* str = chars + start;
    out_mask[gid] = (compare_str(str, len, pattern, pattern_len) < 0) ? 1 : 0;
}

kernel void filter_string_le(const device char* chars [[buffer(0)]],
                                   const device uint32_t* offsets [[buffer(1)]],
                                   device uint8_t* out_mask [[buffer(2)]],
                                   const device char* pattern [[buffer(3)]],
                                   constant uint& pattern_len [[buffer(4)]],
                                   constant uint& row_count [[buffer(5)]],
                                   uint gid [[thread_position_in_grid]]) {
    if (gid >= row_count) return;
    uint start = offsets[gid];
    uint len = offsets[gid + 1] - start;
    const device char* str = chars + start;
    out_mask[gid] = (compare_str(str, len, pattern, pattern_len) <= 0) ? 1 : 0;
}

kernel void filter_string_gt(const device char* chars [[buffer(0)]],
                                   const device uint32_t* offsets [[buffer(1)]],
                                   device uint8_t* out_mask [[buffer(2)]],
                                   const device char* pattern [[buffer(3)]],
                                   constant uint& pattern_len [[buffer(4)]],
                                   constant uint& row_count [[buffer(5)]],
                                   uint gid [[thread_position_in_grid]]) {
    if (gid >= row_count) return;
    uint start = offsets[gid];
    uint len = offsets[gid + 1] - start;
    const device char* str = chars + start;
    out_mask[gid] = (compare_str(str, len, pattern, pattern_len) > 0) ? 1 : 0;
}

kernel void filter_string_ge(const device char* chars [[buffer(0)]],
                                   const device uint32_t* offsets [[buffer(1)]],
                                   device uint8_t* out_mask [[buffer(2)]],
                                   const device char* pattern [[buffer(3)]],
                                   constant uint& pattern_len [[buffer(4)]],
                                   constant uint& row_count [[buffer(5)]],
                                   uint gid [[thread_position_in_grid]]) {
    if (gid >= row_count) return;
    uint start = offsets[gid];
    uint len = offsets[gid + 1] - start;
    const device char* str = chars + start;
    out_mask[gid] = (compare_str(str, len, pattern, pattern_len) >= 0) ? 1 : 0;
}

// ---- GPU string gather kernel (Arrow-style) ----
// Phase 1: Compute output lengths from gathered source offsets.
// Phase 2: Copy chars using prefix-summed output offsets.

kernel void gather_flat_string_chars(const device char* src_chars [[buffer(0)]],
                                     const device uint32_t* src_offsets [[buffer(1)]],
                                     const device uint32_t* indices [[buffer(2)]],
                                     const device uint32_t* dst_offsets [[buffer(3)]],
                                     device char* dst_chars [[buffer(4)]],
                                     constant uint& num_indices [[buffer(5)]],
                                     uint gid [[thread_position_in_grid]]) {
    if (gid >= num_indices) return;
    uint src_idx = indices[gid];
    uint src_start = src_offsets[src_idx];
    uint src_len   = src_offsets[src_idx + 1] - src_start;
    uint dst_start = dst_offsets[gid];
    for (uint i = 0; i < src_len; ++i) {
        dst_chars[dst_start + i] = src_chars[src_start + i];
    }
}

// ----------------------------------------------------------------------------
// Multi-Column Support Helpers
// ----------------------------------------------------------------------------

kernel void pack_u32_to_u64(const device uint32_t* c1 [[buffer(0)]],
                            const device uint32_t* c2 [[buffer(1)]],
                            device uint64_t* out [[buffer(2)]],
                            constant uint32_t& count [[buffer(3)]],
                            uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    out[gid] = (static_cast<uint64_t>(c1[gid]) << 32) | c2[gid];
}

constant uint64_t EMPTY_KEY64 = 0xFFFFFFFFFFFFFFFF;

inline uint32_t hash_u64(uint64_t k) {
    k ^= k >> 33;
    k *= 0xff51afd7ed558ccd;
    k ^= k >> 33;
    k *= 0xc4ceb9fe1a85ec53;
    k ^= k >> 33;
    return (uint32_t)k; 
}

// Lock-free hash table build using linear probing (first-writer-wins)
kernel void join_build_u64(
    const device uint64_t* build_keys [[buffer(0)]],
    const device uint32_t* build_indices [[buffer(1)]],
    device atomic_uint* ht_keys_low [[buffer(2)]],  // Lower 32 bits of key
    device uint32_t* ht_vals [[buffer(3)]],
    constant uint32_t& ht_capacity [[buffer(4)]],
    constant uint32_t& row_count [[buffer(5)]],
    device atomic_uint* ht_keys_high [[buffer(6)]],  // Upper 32 bits of key
    uint gid [[thread_position_in_grid]]) 
{
    if (gid >= row_count) return;

    uint64_t key = build_keys[gid];
    uint32_t payload = gid; 
    if (build_indices) payload = build_indices[gid];

    uint32_t key_low = (uint32_t)(key & 0xFFFFFFFF);
    uint32_t key_high = (uint32_t)(key >> 32);
    
    uint32_t h = hash_u64(key);
    uint32_t idx = h % ht_capacity;
    
    // Linear probing with atomic CAS
    for (uint32_t i = 0; i < MAX_HASH_STEPS; ++i) {
        // Try to claim this slot by setting the low bits first
        uint32_t expected_low = 0xFFFFFFFF; // EMPTY marker
        
        // Atomically try to set key_low
        bool claimed = atomic_compare_exchange_weak_explicit(
            &ht_keys_low[idx], &expected_low, key_low,
            memory_order_relaxed, memory_order_relaxed);
        
        if (claimed) {
            // We successfully claimed this slot
            // Now set the high bits and value (we own the slot)
            atomic_store_explicit(&ht_keys_high[idx], key_high, memory_order_relaxed);
            ht_vals[idx] = payload;
            return;
        }
        
        // Slot already taken - check if it's our key
        uint32_t existing_low = atomic_load_explicit(&ht_keys_low[idx], memory_order_relaxed);
        uint32_t existing_high = atomic_load_explicit(&ht_keys_high[idx], memory_order_relaxed);
        
        if (existing_low == key_low && existing_high == key_high) {
            // Same key - update value
            ht_vals[idx] = payload;
            return;
        }
        
        // Different key - linear probe to next slot
        idx = (idx + 1) % ht_capacity;
    }
}

kernel void join_probe_u64(
    const device uint64_t* probe_keys [[buffer(0)]],
    const device uint32_t* probe_indices [[buffer(1)]],
    const device atomic_uint* ht_keys_low [[buffer(2)]],  // Lower 32 bits
    const device uint32_t* ht_vals [[buffer(3)]],
    constant uint32_t& ht_capacity [[buffer(4)]],
    constant uint32_t& row_count [[buffer(5)]],
    device atomic_uint* out_counter [[buffer(6)]],
    device uint32_t* out_build_indices [[buffer(7)]],
    device uint32_t* out_probe_indices [[buffer(8)]],
    const device atomic_uint* ht_keys_high [[buffer(9)]],  // Upper 32 bits
    uint gid [[thread_position_in_grid]]) 
{
    if (gid >= row_count) return;
    
    uint64_t key = probe_keys[gid];
    uint32_t key_low = (uint32_t)(key & 0xFFFFFFFF);
    uint32_t key_high = (uint32_t)(key >> 32);
    
    uint32_t h = hash_u64(key);
    uint32_t idx = h % ht_capacity;
    
    for (uint32_t i = 0; i < MAX_HASH_STEPS; ++i) {
        uint32_t existing_low = atomic_load_explicit(&ht_keys_low[idx], memory_order_relaxed);
        
        // Check for empty slot (EMPTY marker = 0xFFFFFFFF)
        if (existing_low == 0xFFFFFFFF) break;
        
        // Check if this key matches
        uint32_t existing_high = atomic_load_explicit(&ht_keys_high[idx], memory_order_relaxed);
        if (existing_low == key_low && existing_high == key_high) {
            uint32_t build_idx = ht_vals[idx];
            uint32_t write_pos = atomic_fetch_add_explicit(out_counter, 1, memory_order_relaxed);
            out_build_indices[write_pos] = build_idx;
            uint32_t p_idx = gid;
            if (probe_indices) p_idx = probe_indices[gid];
            out_probe_indices[write_pos] = p_idx;
            break;
        }
        idx = (idx + 1) % ht_capacity;
    }
}

} // namespace ops

// Extract YEAR from YYYYMMDD u32
kernel void extract_year_u32_to_f32(
    const device uint32_t* in [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint32_t& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return;
    // in[gid] is YYYYMMDD. Integer division by 10000 gives YYYY.
    out[gid] = (float)(in[gid] / 10000);
}

kernel void arith_sub_f32_col_col(
    const device float* a [[buffer(0)]],
    const device float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint32_t& count [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return;
    out[gid] = a[gid] - b[gid];
}

kernel void arith_sub_f32_scalar_col(
    constant float& a [[buffer(0)]],
    const device float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint32_t& count [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return;
    out[gid] = a - b[gid];
}

kernel void arith_sub_f32_col_scalar(
    const device float* a [[buffer(0)]],
    constant float& b [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint32_t& count [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return;
    out[gid] = a[gid] - b;
}

kernel void arith_add_f32_col_col(
    const device float* a [[buffer(0)]],
    const device float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint32_t& count [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return;
    out[gid] = a[gid] + b[gid];
}

kernel void arith_add_f32_scalar_col(
    constant float& a [[buffer(0)]],
    const device float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint32_t& count [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return;
    out[gid] = a + b[gid];
}

kernel void arith_add_f32_col_scalar(
    const device float* a [[buffer(0)]],
    constant float& b [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint32_t& count [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return;
    out[gid] = a[gid] + b;
}

namespace ops {
// ============================================================================
// Col-Col Comparison Kernels
// ============================================================================

kernel void filter_u32_col_col_eq(const device uint32_t* colA [[buffer(0)]], const device uint32_t* colB [[buffer(1)]], device uint8_t* mask [[buffer(2)]], constant uint32_t& count [[buffer(3)]], uint gid [[thread_position_in_grid]]) { if (gid < count) mask[gid] = (colA[gid] == colB[gid]); }
kernel void filter_u32_col_col_ne(const device uint32_t* colA [[buffer(0)]], const device uint32_t* colB [[buffer(1)]], device uint8_t* mask [[buffer(2)]], constant uint32_t& count [[buffer(3)]], uint gid [[thread_position_in_grid]]) { if (gid < count) mask[gid] = (colA[gid] != colB[gid]); }
kernel void filter_u32_col_col_lt(const device uint32_t* colA [[buffer(0)]], const device uint32_t* colB [[buffer(1)]], device uint8_t* mask [[buffer(2)]], constant uint32_t& count [[buffer(3)]], uint gid [[thread_position_in_grid]]) { if (gid < count) mask[gid] = (colA[gid] < colB[gid]); }
kernel void filter_u32_col_col_le(const device uint32_t* colA [[buffer(0)]], const device uint32_t* colB [[buffer(1)]], device uint8_t* mask [[buffer(2)]], constant uint32_t& count [[buffer(3)]], uint gid [[thread_position_in_grid]]) { if (gid < count) mask[gid] = (colA[gid] <= colB[gid]); }
kernel void filter_u32_col_col_gt(const device uint32_t* colA [[buffer(0)]], const device uint32_t* colB [[buffer(1)]], device uint8_t* mask [[buffer(2)]], constant uint32_t& count [[buffer(3)]], uint gid [[thread_position_in_grid]]) { if (gid < count) mask[gid] = (colA[gid] > colB[gid]); }
kernel void filter_u32_col_col_ge(const device uint32_t* colA [[buffer(0)]], const device uint32_t* colB [[buffer(1)]], device uint8_t* mask [[buffer(2)]], constant uint32_t& count [[buffer(3)]], uint gid [[thread_position_in_grid]]) { if (gid < count) mask[gid] = (colA[gid] >= colB[gid]); }

kernel void filter_f32_col_col_eq(const device float* colA [[buffer(0)]], const device float* colB [[buffer(1)]], device uint8_t* mask [[buffer(2)]], constant uint32_t& count [[buffer(3)]], uint gid [[thread_position_in_grid]]) { if (gid < count) mask[gid] = (colA[gid] == colB[gid]); }
kernel void filter_f32_col_col_ne(const device float* colA [[buffer(0)]], const device float* colB [[buffer(1)]], device uint8_t* mask [[buffer(2)]], constant uint32_t& count [[buffer(3)]], uint gid [[thread_position_in_grid]]) { if (gid < count) mask[gid] = (colA[gid] != colB[gid]); }
kernel void filter_f32_col_col_lt(const device float* colA [[buffer(0)]], const device float* colB [[buffer(1)]], device uint8_t* mask [[buffer(2)]], constant uint32_t& count [[buffer(3)]], uint gid [[thread_position_in_grid]]) { if (gid < count) mask[gid] = (colA[gid] < colB[gid]); }
kernel void filter_f32_col_col_le(const device float* colA [[buffer(0)]], const device float* colB [[buffer(1)]], device uint8_t* mask [[buffer(2)]], constant uint32_t& count [[buffer(3)]], uint gid [[thread_position_in_grid]]) { if (gid < count) mask[gid] = (colA[gid] <= colB[gid]); }
kernel void filter_f32_col_col_gt(const device float* colA [[buffer(0)]], const device float* colB [[buffer(1)]], device uint8_t* mask [[buffer(2)]], constant uint32_t& count [[buffer(3)]], uint gid [[thread_position_in_grid]]) { if (gid < count) mask[gid] = (colA[gid] > colB[gid]); }
kernel void filter_f32_col_col_ge(const device float* colA [[buffer(0)]], const device float* colB [[buffer(1)]], device uint8_t* mask [[buffer(2)]], constant uint32_t& count [[buffer(3)]], uint gid [[thread_position_in_grid]]) { if (gid < count) mask[gid] = (colA[gid] >= colB[gid]); }

}

namespace ops {
kernel void scatter_constant_f32(
    device float* output [[buffer(0)]],
    const device uint32_t* indices [[buffer(1)]],
    constant float& val [[buffer(2)]],
    constant uint32_t& count [[buffer(3)]],
    uint index [[thread_position_in_grid]])
{
    if (index < count) {
        output[indices[index]] = val;
    }
}

kernel void scatter_f32_indexed(
    const device float* input [[buffer(0)]],
    const device uint32_t* indices [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint32_t& count [[buffer(3)]],
    uint index [[thread_position_in_grid]])
{
    if (index < count) {
        output[indices[index]] = input[index];
    }
}

kernel void arith_floor_f32(const device float* col [[buffer(0)]],
                            device float* out [[buffer(1)]],
                            constant uint& count [[buffer(2)]],
                            uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    out[gid] = floor(col[gid]);
}

// ============================================================================
// SCAN / PREFIX SUM KERNELS
// ============================================================================

// Basic Hillis-Steele exclusive scan for a single threadgroup.
// Writes total sum of the block to partial_sums[group_id] if partial_sums is not null.
// Output 'data' becomes the exclusive scan of input 'data' within the block.
kernel void scan_exclusive_subblock_u32(
    device uint32_t* data [[buffer(0)]],
    device uint32_t* partial_sums [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint group_id [[threadgroup_position_in_grid]],
    uint threads_per_group [[threads_per_threadgroup]]
) {
    // threads_per_group must be power of 2, max 1024
    threadgroup uint32_t temp[1024]; 

    // Load input
    uint32_t val = (gid < n) ? data[gid] : 0;
    
    temp[tid] = val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Hillis-Steele
    for (uint offset = 1; offset < threads_per_group; offset <<= 1) {
        uint32_t t = 0;
        if (tid >= offset) {
            t = temp[tid - offset];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid >= offset) {
            temp[tid] += t;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Save total for this block
    if (tid == threads_per_group - 1) {
        if (partial_sums) {
             partial_sums[group_id] = temp[tid];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Exclusive scan result
    uint32_t res = (tid > 0) ? temp[tid - 1] : 0;

    if (gid < n) {
        data[gid] = res;
    }
}

// Adds base value to each element in the block
kernel void scan_add_base_u32(
    device uint32_t* data [[buffer(0)]],
    device const uint32_t* bases [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint group_id [[threadgroup_position_in_grid]]
) {
    if (gid >= n) return;
    // bases array should be the exclusive scan of partial sums.
    // So bases[group_id] is the start value for this block.
    data[gid] += bases[group_id];
}

// ============================================================================
// GPU SORT KERNELS — Block Sort (shared-memory bitonic) + Radix Sort
// ============================================================================

// ---------- Block Sort: in-threadgroup bitonic sort for ≤1024 elements ------
// Sorts (key, index) pairs entirely in shared memory with a single dispatch.
// Host must launch exactly one threadgroup of nextPow2(n) threads.

kernel void block_sort_kv_u32(
    device uint32_t* keys [[buffer(0)]],
    device uint32_t* vals [[buffer(1)]],
    constant uint&   n    [[buffer(2)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tpg [[threads_per_threadgroup]]
) {
    threadgroup uint32_t lk[1024];
    threadgroup uint32_t lv[1024];

    lk[tid] = (tid < n) ? keys[tid] : 0xFFFFFFFFu;
    lv[tid] = (tid < n) ? vals[tid] : tid;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint blk = 2; blk <= tpg; blk <<= 1) {
        for (uint stp = blk >> 1; stp >= 1; stp >>= 1) {
            uint partner = tid ^ stp;
            if (partner > tid && partner < tpg) {
                bool asc = ((tid & blk) == 0);
                if (asc ? (lk[tid] > lk[partner]) : (lk[tid] < lk[partner])) {
                    uint32_t tk = lk[tid]; lk[tid] = lk[partner]; lk[partner] = tk;
                    uint32_t tv = lv[tid]; lv[tid] = lv[partner]; lv[partner] = tv;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    if (tid < n) { keys[tid] = lk[tid]; vals[tid] = lv[tid]; }
}

kernel void block_sort_kv_u64(
    device ulong*    keys [[buffer(0)]],
    device uint32_t* vals [[buffer(1)]],
    constant uint&   n    [[buffer(2)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tpg [[threads_per_threadgroup]]
) {
    threadgroup ulong    lk[1024];
    threadgroup uint32_t lv[1024];

    lk[tid] = (tid < n) ? keys[tid] : 0xFFFFFFFFFFFFFFFFul;
    lv[tid] = (tid < n) ? vals[tid] : tid;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint blk = 2; blk <= tpg; blk <<= 1) {
        for (uint stp = blk >> 1; stp >= 1; stp >>= 1) {
            uint partner = tid ^ stp;
            if (partner > tid && partner < tpg) {
                bool asc = ((tid & blk) == 0);
                if (asc ? (lk[tid] > lk[partner]) : (lk[tid] < lk[partner])) {
                    ulong  tk = lk[tid]; lk[tid] = lk[partner]; lk[partner] = tk;
                    uint32_t tv = lv[tid]; lv[tid] = lv[partner]; lv[partner] = tv;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    if (tid < n) { keys[tid] = lk[tid]; vals[tid] = lv[tid]; }
}

// ---------- Radix Sort: 8-bit radix, stable, for >1024 elements -------------
// Histogram: per-threadgroup 256-bin histogram.
// Layout: histograms[digit * numBlocks + group_id]  (digit-major for linear prefix-sum)

kernel void radix_histogram_u32(
    device const uint32_t* keys  [[buffer(0)]],
    device uint32_t* histograms  [[buffer(1)]],
    constant uint& n             [[buffer(2)]],
    constant uint& shift         [[buffer(3)]],
    constant uint& numBlocks     [[buffer(4)]],
    uint gid      [[thread_position_in_grid]],
    uint tid      [[thread_position_in_threadgroup]],
    uint group_id [[threadgroup_position_in_grid]]
) {
    threadgroup atomic_uint lh[256];
    if (tid < 256) atomic_store_explicit(&lh[tid], 0, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (gid < n) {
        uint d = (keys[gid] >> shift) & 0xFFu;
        atomic_fetch_add_explicit(&lh[d], 1, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < 256) histograms[tid * numBlocks + group_id] =
                       atomic_load_explicit(&lh[tid], memory_order_relaxed);
}

kernel void radix_histogram_u64(
    device const ulong* keys     [[buffer(0)]],
    device uint32_t* histograms  [[buffer(1)]],
    constant uint& n             [[buffer(2)]],
    constant uint& shift         [[buffer(3)]],
    constant uint& numBlocks     [[buffer(4)]],
    uint gid      [[thread_position_in_grid]],
    uint tid      [[thread_position_in_threadgroup]],
    uint group_id [[threadgroup_position_in_grid]]
) {
    threadgroup atomic_uint lh[256];
    if (tid < 256) atomic_store_explicit(&lh[tid], 0, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (gid < n) {
        uint d = (uint)((keys[gid] >> shift) & 0xFFul);
        atomic_fetch_add_explicit(&lh[d], 1, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < 256) histograms[tid * numBlocks + group_id] =
                       atomic_load_explicit(&lh[tid], memory_order_relaxed);
}

// Scatter: stable scatter using per-thread local ranking (preserves input order
// within each digit bucket for sort stability, critical for LSB radix sort).

kernel void radix_scatter_u32(
    device const uint32_t* keys_in   [[buffer(0)]],
    device const uint32_t* vals_in   [[buffer(1)]],
    device uint32_t*       keys_out  [[buffer(2)]],
    device uint32_t*       vals_out  [[buffer(3)]],
    device const uint32_t* scan_hist [[buffer(4)]],
    constant uint& n                 [[buffer(5)]],
    constant uint& shift             [[buffer(6)]],
    constant uint& numBlocks         [[buffer(7)]],
    uint gid      [[thread_position_in_grid]],
    uint tid      [[thread_position_in_threadgroup]],
    uint group_id [[threadgroup_position_in_grid]],
    uint tpg      [[threads_per_threadgroup]]
) {
    threadgroup uint offsets[256];
    if (tid < 256) offsets[tid] = scan_hist[tid * numBlocks + group_id];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup uint ld[256];   // local digits for this block
    bool valid = gid < n;
    uint myDigit = 0;
    uint32_t myKey = 0, myVal = 0;
    if (valid) {
        myKey = keys_in[gid];
        myVal = vals_in[gid];
        myDigit = (myKey >> shift) & 0xFFu;
    }
    ld[tid] = valid ? myDigit : 0xFFFFu;    // sentinel
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (valid) {
        uint rank = 0;
        for (uint j = 0; j < tid; j++) { if (ld[j] == myDigit) rank++; }
        uint pos = offsets[myDigit] + rank;
        keys_out[pos] = myKey;
        vals_out[pos] = myVal;
    }
}

kernel void radix_scatter_u64(
    device const ulong*    keys_in   [[buffer(0)]],
    device const uint32_t* vals_in   [[buffer(1)]],
    device ulong*          keys_out  [[buffer(2)]],
    device uint32_t*       vals_out  [[buffer(3)]],
    device const uint32_t* scan_hist [[buffer(4)]],
    constant uint& n                 [[buffer(5)]],
    constant uint& shift             [[buffer(6)]],
    constant uint& numBlocks         [[buffer(7)]],
    uint gid      [[thread_position_in_grid]],
    uint tid      [[thread_position_in_threadgroup]],
    uint group_id [[threadgroup_position_in_grid]],
    uint tpg      [[threads_per_threadgroup]]
) {
    threadgroup uint offsets[256];
    if (tid < 256) offsets[tid] = scan_hist[tid * numBlocks + group_id];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup uint ld[256];
    bool valid = gid < n;
    uint myDigit = 0;
    ulong myKey = 0;
    uint32_t myVal = 0;
    if (valid) {
        myKey = keys_in[gid];
        myVal = vals_in[gid];
        myDigit = (uint)((myKey >> shift) & 0xFFul);
    }
    ld[tid] = valid ? myDigit : 0xFFFFu;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (valid) {
        uint rank = 0;
        for (uint j = 0; j < tid; j++) { if (ld[j] == myDigit) rank++; }
        uint pos = offsets[myDigit] + rank;
        keys_out[pos] = myKey;
        vals_out[pos] = myVal;
    }
}

// ── GroupBy Hash Table Stream Compaction ───────────────────────────

// Step 1 (Mark): Write 1 if slot is valid (key[0] != 0), else 0.
kernel void ht_mark_valid(
    device const uint32_t* ht_keys [[buffer(0)]],
    device uint32_t*       mark    [[buffer(1)]],
    constant uint32_t&     cap     [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= cap) return;
    mark[gid] = (ht_keys[gid * 8] != 0) ? 1u : 0u;
}

// Step 3 (Compact): Write valid keys and agg words to dense output arrays.
// offsets[] is the exclusive prefix sum of mark[].
kernel void ht_extract_compact(
    device const uint32_t* ht_keys  [[buffer(0)]],
    device const uint32_t* ht_aggs  [[buffer(1)]],
    device const uint32_t* mark     [[buffer(2)]],
    device const uint32_t* offsets  [[buffer(3)]],
    device uint32_t*       out_keys [[buffer(4)]],
    device uint32_t*       out_aggs [[buffer(5)]],
    constant uint32_t&     cap      [[buffer(6)]],
    constant uint32_t&     numKeys  [[buffer(7)]],
    constant uint32_t&     numAggs  [[buffer(8)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= cap) return;
    if (mark[gid] == 0) return;
    uint dest = offsets[gid];
    for (uint k = 0; k < numKeys; ++k) {
        uint keyVal = ht_keys[gid * 8 + k];
        out_keys[dest * numKeys + k] = (keyVal > 0) ? (keyVal - 1) : 0;
    }
    for (uint a = 0; a < numAggs; ++a) {
        out_aggs[dest * numAggs + a] = ht_aggs[gid * 16 + a];
    }
}

}
