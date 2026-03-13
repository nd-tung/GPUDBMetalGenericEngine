// JoinKernels.h — Hash join build/probe kernels (u32, u64, multi-match)
#ifndef JOINKERNELS_H
#define JOINKERNELS_H

// ============================================================================
// Multi-Match Hash Join (linked-list chaining)
// ============================================================================

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

        if (atomic_compare_exchange_weak_explicit(
                &ht_keys[slot], &expected, key_store,
                memory_order_relaxed, memory_order_relaxed)) {
            const uint old = atomic_exchange_explicit(
                &ht_head[slot], gid + 1u, memory_order_relaxed);
            next[gid] = old;
            return;
        }

        if (expected == key_store) {
            const uint old = atomic_exchange_explicit(
                &ht_head[slot], gid + 1u, memory_order_relaxed);
            next[gid] = old;
            return;
        }

        if (expected == 0u) {
            continue;
        }

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

// ============================================================================
// Cross Product
// ============================================================================

kernel void cross_product(
    const device uint32_t* left [[buffer(0)]],
    const device uint32_t* right [[buffer(1)]],
    device uint32_t* outLeft [[buffer(2)]],
    device uint32_t* outRight [[buffer(3)]],
    constant uint& leftCount [[buffer(4)]],
    constant uint& rightCount [[buffer(5)]],
    constant uint& totalCount [[buffer(6)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= totalCount) return;
    
    uint i = gid / rightCount;
    uint j = gid % rightCount;
    
    outLeft[gid] = left[i];
    outRight[gid] = right[j];
}

// ============================================================================
// Multi-Column Key Support (u64 pack + u64 join)
// ============================================================================

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

kernel void join_build_u64(
    const device uint64_t* build_keys [[buffer(0)]],
    const device uint32_t* build_indices [[buffer(1)]],
    device atomic_uint* ht_keys_low [[buffer(2)]],
    device uint32_t* ht_vals [[buffer(3)]],
    constant uint32_t& ht_capacity [[buffer(4)]],
    constant uint32_t& row_count [[buffer(5)]],
    device atomic_uint* ht_keys_high [[buffer(6)]],
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
    
    for (uint32_t i = 0; i < MAX_HASH_STEPS; ++i) {
        uint32_t expected_low = 0xFFFFFFFF;
        
        bool claimed = atomic_compare_exchange_weak_explicit(
            &ht_keys_low[idx], &expected_low, key_low,
            memory_order_relaxed, memory_order_relaxed);
        
        if (claimed) {
            atomic_store_explicit(&ht_keys_high[idx], key_high, memory_order_relaxed);
            ht_vals[idx] = payload;
            return;
        }
        
        uint32_t existing_low = atomic_load_explicit(&ht_keys_low[idx], memory_order_relaxed);
        uint32_t existing_high = atomic_load_explicit(&ht_keys_high[idx], memory_order_relaxed);
        
        if (existing_low == key_low && existing_high == key_high) {
            ht_vals[idx] = payload;
            return;
        }
        
        idx = (idx + 1) % ht_capacity;
    }
}

kernel void join_probe_u64(
    const device uint64_t* probe_keys [[buffer(0)]],
    const device uint32_t* probe_indices [[buffer(1)]],
    const device atomic_uint* ht_keys_low [[buffer(2)]],
    const device uint32_t* ht_vals [[buffer(3)]],
    constant uint32_t& ht_capacity [[buffer(4)]],
    constant uint32_t& row_count [[buffer(5)]],
    device atomic_uint* out_counter [[buffer(6)]],
    device uint32_t* out_build_indices [[buffer(7)]],
    device uint32_t* out_probe_indices [[buffer(8)]],
    const device atomic_uint* ht_keys_high [[buffer(9)]],
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
        
        if (existing_low == 0xFFFFFFFF) break;
        
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

// Deterministic version: writes mask + build map instead of atomic append
kernel void join_probe_u64_mark(
    const device uint64_t* probe_keys [[buffer(0)]],
    const device uint32_t* probe_indices [[buffer(1)]],
    const device atomic_uint* ht_keys_low [[buffer(2)]],
    const device uint32_t* ht_vals [[buffer(3)]],
    constant uint32_t& ht_capacity [[buffer(4)]],
    constant uint32_t& row_count [[buffer(5)]],
    device uint8_t* out_mask [[buffer(6)]],
    device uint32_t* out_build_map [[buffer(7)]],
    const device atomic_uint* ht_keys_high [[buffer(8)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= row_count) return;
    out_mask[gid] = 0;

    uint64_t key = probe_keys[gid];
    uint32_t key_low = (uint32_t)(key & 0xFFFFFFFF);
    uint32_t key_high = (uint32_t)(key >> 32);

    uint32_t h = hash_u64(key);
    uint32_t idx = h % ht_capacity;

    for (uint32_t i = 0; i < MAX_HASH_STEPS; ++i) {
        uint32_t existing_low = atomic_load_explicit(&ht_keys_low[idx], memory_order_relaxed);
        if (existing_low == 0xFFFFFFFF) break;
        uint32_t existing_high = atomic_load_explicit(&ht_keys_high[idx], memory_order_relaxed);
        if (existing_low == key_low && existing_high == key_high) {
            out_mask[gid] = 1;
            out_build_map[gid] = ht_vals[idx];
            break;
        }
        idx = (idx + 1) % ht_capacity;
    }
}

#endif // JOINKERNELS_H
