// FilterKernels.h — Filter, mask, compact, and logic kernels
#ifndef FILTERKERNELS_H
#define FILTERKERNELS_H

// ============================================================================
// Range-to-Mask Filters
// ============================================================================

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

// ============================================================================
// Scalar Comparison Filters (u32)
// ============================================================================

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

// ============================================================================
// Scalar Comparison Filters (f32)
// ============================================================================

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

// ============================================================================
// Indexed Range Filters
// ============================================================================

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

// ============================================================================
// Col-vs-Col Filters
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
        case 0: pass = (va == vb); break;
        case 1: pass = (va != vb); break;
        case 2: pass = (va < vb); break;
        case 3: pass = (va <= vb); break;
        case 4: pass = (va > vb); break;
        case 5: pass = (va >= vb); break;
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
        case 0: pass = (va == vb); break;
        case 1: pass = (va != vb); break;
        case 2: pass = (va < vb); break;
        case 3: pass = (va <= vb); break;
        case 4: pass = (va > vb); break;
        case 5: pass = (va >= vb); break;
    }
    out_mask[gid] = pass ? 1 : 0;
}

// Typed col-col comparison kernels (one-liners)
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

// ============================================================================
// Generic Comparison + Logic Kernels
// ============================================================================

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
        case 0: p = (a < b); break;
        case 1: p = (a <= b); break;
        case 2: p = (a > b); break;
        case 3: p = (a >= b); break;
        case 4: p = (a == b); break;
        case 5: p = (a != b); break;
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

kernel void logic_not_u32(
    const device uint32_t* inMask [[buffer(0)]],
    device uint32_t* outMask [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    outMask[gid] = inMask[gid] ? 0 : 1;
}

kernel void logic_andnot_u32(
    const device uint32_t* colA [[buffer(0)]],
    const device uint32_t* colB [[buffer(1)]],
    device uint32_t* outMask [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    outMask[gid] = (colA[gid] && !colB[gid]) ? 1 : 0;
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

// ============================================================================
// Compact / Mask Conversion
// ============================================================================

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

kernel void compact_indices_indexed(const device uint8_t* mask [[buffer(0)]],
                                    const device uint32_t* in_indices [[buffer(1)]],
                                    device uint32_t* out_indices [[buffer(2)]],
                                    device atomic_uint* out_count [[buffer(3)]],
                                    constant uint& row_count [[buffer(4)]],
                                    uint gid [[thread_position_in_grid]]) {
    if (gid >= row_count) return;
    if (mask[gid]) {
        uint idx = atomic_fetch_add_explicit(out_count, 1, memory_order_relaxed);
        out_indices[idx] = in_indices[gid];
    }
}

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

// ============================================================================
// Deterministic compaction via prefix-sum (replaces atomic compact kernels)
// ============================================================================

// Convert u8 mask to u32 0/1 for prefix-sum scan
kernel void mask_to_offsets_u8(const device uint8_t* mask [[buffer(0)]],
                               device uint32_t* offsets [[buffer(1)]],
                               constant uint& count [[buffer(2)]],
                               uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    offsets[gid] = mask[gid] ? 1u : 0u;
}

// Convert u32 mask to u32 0/1 for prefix-sum scan
kernel void mask_to_offsets_u32(const device uint32_t* mask [[buffer(0)]],
                                device uint32_t* offsets [[buffer(1)]],
                                constant uint& count [[buffer(2)]],
                                uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    offsets[gid] = mask[gid] ? 1u : 0u;
}

// Scatter gid to output at position prefix[gid], when u8 mask is set
kernel void scatter_by_prefix_u8(const device uint8_t* mask [[buffer(0)]],
                                 const device uint32_t* prefix [[buffer(1)]],
                                 device uint32_t* out [[buffer(2)]],
                                 constant uint& count [[buffer(3)]],
                                 uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    if (mask[gid]) out[prefix[gid]] = gid;
}

// Scatter in_indices[gid] to output at position prefix[gid], when u8 mask is set
kernel void scatter_by_prefix_u8_indexed(const device uint8_t* mask [[buffer(0)]],
                                         const device uint32_t* in_indices [[buffer(1)]],
                                         const device uint32_t* prefix [[buffer(2)]],
                                         device uint32_t* out [[buffer(3)]],
                                         constant uint& count [[buffer(4)]],
                                         uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    if (mask[gid]) out[prefix[gid]] = in_indices[gid];
}

// Scatter gid to output at position prefix[gid], when u32 mask is set
kernel void scatter_by_prefix_u32(const device uint32_t* mask [[buffer(0)]],
                                  const device uint32_t* prefix [[buffer(1)]],
                                  device uint32_t* out [[buffer(2)]],
                                  constant uint& count [[buffer(3)]],
                                  uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    if (mask[gid]) out[prefix[gid]] = gid;
}

kernel void indices_to_mask(
    const device uint32_t* indices [[buffer(0)]],
    device uint32_t* mask [[buffer(1)]],
    constant uint& indexCount [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= indexCount) return;
    uint32_t idx = indices[gid];
    mask[idx] = 1;
}

kernel void clear_mask(
    device uint32_t* mask [[buffer(0)]],
    constant uint& count [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    mask[gid] = 0;
}

kernel void flip_mask_u8(
    device uint8_t*     mask  [[buffer(0)]],
    constant uint32_t&  count [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    mask[gid] = mask[gid] ? 0 : 1;
}

#endif // FILTERKERNELS_H
