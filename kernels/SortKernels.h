// SortKernels.h — Bitonic sort, radix sort, prefix scan, and sort-key utility kernels
#ifndef SORTKERNELS_H
#define SORTKERNELS_H

// ============================================================================
// Bitonic Sort
// ============================================================================

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
        data[leftId] = rightVal;
        data[rightId] = leftVal;
        uint tmpIdx = indices[leftId];
        indices[leftId] = indices[rightId];
        indices[rightId] = tmpIdx;
    }
}

// ============================================================================
// SCAN / PREFIX SUM KERNELS
// ============================================================================

kernel void scan_exclusive_subblock_u32(
    device uint32_t* data [[buffer(0)]],
    device uint32_t* partial_sums [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint group_id [[threadgroup_position_in_grid]],
    uint threads_per_group [[threads_per_threadgroup]]
) {
    threadgroup uint32_t temp[1024]; 

    uint32_t val = (gid < n) ? data[gid] : 0;
    
    temp[tid] = val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

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
    
    if (tid == threads_per_group - 1) {
        if (partial_sums) {
             partial_sums[group_id] = temp[tid];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint32_t res = (tid > 0) ? temp[tid - 1] : 0;

    if (gid < n) {
        data[gid] = res;
    }
}

kernel void scan_add_base_u32(
    device uint32_t* data [[buffer(0)]],
    device const uint32_t* bases [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint group_id [[threadgroup_position_in_grid]]
) {
    if (gid >= n) return;
    data[gid] += bases[group_id];
}

// ============================================================================
// Block Sort (shared-memory bitonic) for ≤1024 elements
// ============================================================================

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

// ============================================================================
// Radix Sort (8-bit radix, stable)
// ============================================================================

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

    threadgroup uint ld[256];
    bool valid = gid < n;
    uint myDigit = 0;
    uint32_t myKey = 0, myVal = 0;
    if (valid) {
        myKey = keys_in[gid];
        myVal = vals_in[gid];
        myDigit = (myKey >> shift) & 0xFFu;
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

// ============================================================================
// Sort Key Transform + Dedup
// ============================================================================

kernel void float_to_sort_key_u32(
    const device float*    in    [[buffer(0)]],
    device uint32_t*       out   [[buffer(1)]],
    constant uint32_t&     count [[buffer(2)]],
    constant uint32_t&     desc  [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    uint32_t bits = as_type<uint32_t>(in[gid]);
    if (bits & 0x80000000u)
        bits = ~bits;
    else
        bits ^= 0x80000000u;
    out[gid] = desc ? ~bits : bits;
}

kernel void invert_u32(
    const device uint32_t* in    [[buffer(0)]],
    device uint32_t*       out   [[buffer(1)]],
    constant uint32_t&     count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    out[gid] = ~in[gid];
}

kernel void mark_unique_sorted_u32(
    const device uint32_t* sortedKeys [[buffer(0)]],
    device uint32_t*       mask       [[buffer(1)]],
    constant uint32_t&     count      [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    if (gid == 0) { mask[gid] = 1; return; }
    mask[gid] = (sortedKeys[gid] != sortedKeys[gid - 1]) ? 1 : 0;
}

kernel void mark_unique_sorted_u64(
    const device uint64_t* sortedKeys [[buffer(0)]],
    device uint32_t*       mask       [[buffer(1)]],
    constant uint32_t&     count      [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    if (gid == 0) { mask[gid] = 1; return; }
    mask[gid] = (sortedKeys[gid] != sortedKeys[gid - 1]) ? 1 : 0;
}

#endif // SORTKERNELS_H
