// AggregateKernels.h — GroupBy, reduction, fused scan/filter/aggregate, HT compaction
#ifndef AGGREGATEKERNELS_H
#define AGGREGATEKERNELS_H

// ============================================================================
// GroupBy Structures
// ============================================================================

struct GroupByBucketF32 {
    atomic_uint key;
    atomic_uint count;
    atomic_uint sum_bits;
};

// Helper for ctz (count trailing zeros)
inline uint ctz(uint mask) {
    return __builtin_ctz(mask);
}

// ============================================================================
// Simple GroupBy (single-key, single-agg)
// ============================================================================

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
    uint32_t slot = k & bucket_mask;
    atomic_store_explicit(&bucket_keys[slot], k, memory_order_relaxed);
    atomic_fetch_add_explicit(&bucket_counts[slot], 1u, memory_order_relaxed);
    atomicAddF32Bits(&bucket_sumbits[slot], v);
}

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
    
    uint slot = key % capacity;
    
    atomic_store_explicit(&ht_keys[slot], key, memory_order_relaxed);
    atomic_fetch_add_explicit(&ht_counts[slot], 1u, memory_order_relaxed);
    atomicAddF32Bits(&ht_sum_bits[slot], val);
}

// ============================================================================
// Fused Scan/Filter/Aggregate Kernels
// ============================================================================

// Packed predicate clause
struct PredicateClause {
    uint colIndex;
    uint op;         // 0:LT 1:LE 2:GT 3:GE 4:EQ
    uint isDate;
    uint isString;
    uint isOrNext;
    uint _pad;
    int64_t value;
};

// RPN expression token for arithmetic evaluation
struct ExprToken {
    uint type;       // 0:column_ref 1:literal 2:operator
    uint colIndex;
    float literal;
    uint op;         // if type==2: 0:+ 1:- 2:* 3:/
};

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

    const device float* cols[8] = {col0, col1, col2, col3, col4, col5, col6, col7};
    float target_val = cols[0][gid];
    
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
    
    localVals[tid] = (passes ? target_val : 0.0f);
    threadgroup_barrier(mem_flags::mem_threadgroup);

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
        float stack[32];
        uint sp = 0;
        
        for (uint i = 0; i < expr_length; ++i) {
            ExprToken tok = expr_rpn[i];
            if (tok.type == 0) {
                if (tok.colIndex < col_count && sp < 32) {
                    stack[sp++] = cols[tok.colIndex][gid];
                }
            } else if (tok.type == 1) {
                if (sp < 32) {
                    stack[sp++] = tok.literal;
                }
            } else if (tok.type == 2) {
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
                    if (sp < 32) {
                        stack[sp++] = res;
                    }
                }
            }
        }
        
        if (sp > 0) {
            result_val = stack[sp - 1];
        }
    }
    
    localVals[tid] = result_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
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
        if (tok.type == 0) {
            if (tok.colIndex < col_count && sp < 16) {
                stack[sp++] = cols[tok.colIndex][gid];
            }
        } else if (tok.type == 1) {
            if (sp < 16) {
                stack[sp++] = tok.literal;
            }
        } else if (tok.type == 2) {
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
            groupResult = groupResult || clauseResult;
        } else {
            passes = passes && groupResult;
            if (!passes) break;
            groupResult = clauseResult;
        }
    }
    if (clause_count > 0) passes = passes && groupResult;
    
    if (aggType == 0) {
        localVals[tid] = passes ? 1.0f : 0.0f;
        localCounts[tid] = passes ? 1 : 0;
    } else if (aggType == 3) {
        localVals[tid] = passes ? target_val : FLT_MAX;
    } else if (aggType == 4) {
        localVals[tid] = passes ? target_val : -FLT_MAX;
    } else {
        localVals[tid] = passes ? target_val : 0.0f;
        localCounts[tid] = passes ? 1 : 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint stride = tgSize >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (aggType == 3) {
                localVals[tid] = min(localVals[tid], localVals[tid + stride]);
            } else if (aggType == 4) {
                localVals[tid] = max(localVals[tid], localVals[tid + stride]);
            } else {
                localVals[tid] += localVals[tid + stride];
                if (aggType == 0 || aggType == 2) {
                    localCounts[tid] += localCounts[tid + stride];
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (tid == 0) {
        union { uint32_t u; float f; } conv;
        conv.f = localVals[0];
        
        if (aggType == 3) {
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
            atomicAddF32Bits(out_result_bits, conv.f);
        }
        
        if (aggType == 0 || aggType == 2) {
            atomic_fetch_add_explicit(out_count, localCounts[0], memory_order_relaxed);
        }
    }
}

// ============================================================================
// Multi-Column GroupBy (SIMD-optimized)
// ============================================================================

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
                                   device atomic_uint* ht_keys [[buffer(12)]],
                                   device atomic_uint* ht_agg_bits [[buffer(13)]],
                                   constant uint& capacity [[buffer(14)]],
                                   constant uint& row_count [[buffer(15)]],
                                   constant uint& num_keys [[buffer(16)]],
                                   constant uint& num_aggs [[buffer(17)]],
                                   uint gid [[thread_position_in_grid]],
                                   uint simd_lane_id [[thread_index_in_simdgroup]]) {
    if (gid >= row_count) return;

    constexpr uint IN_PROGRESS = 0xFFFFFFFFu;
    
    uint keys[4];
    keys[0] = (num_keys > 0) ? key_col0[gid] : 0;
    keys[1] = (num_keys > 1) ? key_col1[gid] : 0;
    keys[2] = (num_keys > 2) ? key_col2[gid] : 0;
    keys[3] = (num_keys > 3) ? key_col3[gid] : 0;
    
    float aggs[8];
    aggs[0] = (num_aggs > 0) ? agg_col0[gid] : 0.0f;
    aggs[1] = (num_aggs > 1) ? agg_col1[gid] : 0.0f;
    aggs[2] = (num_aggs > 2) ? agg_col2[gid] : 0.0f;
    aggs[3] = (num_aggs > 3) ? agg_col3[gid] : 0.0f;
    aggs[4] = (num_aggs > 4) ? agg_col4[gid] : 0.0f;
    aggs[5] = (num_aggs > 5) ? agg_col5[gid] : 0.0f;
    aggs[6] = (num_aggs > 6) ? agg_col6[gid] : 0.0f;
    aggs[7] = (num_aggs > 7) ? agg_col7[gid] : 0.0f;
    
    bool done = false;
    ulong active_mask = (ulong)simd_ballot(true);
    
    for (uint i = 0; i < 32; ++i) {
        if (!((active_mask >> i) & 1)) continue;
        if (simd_all(done)) break;
        
        uint leader_gid = simd_broadcast(gid, i);
        if (leader_gid >= row_count) continue;
        
        bool leader_done = (bool)simd_broadcast((uint)done, i);
        if (leader_done) continue;
        
        uint leader_keys[4];
        for(uint k=0; k<4; ++k) leader_keys[k] = simd_broadcast(keys[k], i);
        
        bool match = !done;
        for(uint k=0; k<num_keys; ++k) {
            if (keys[k] != leader_keys[k]) match = false;
        }
        
        float group_sums[8];
        for (uint a=0; a<num_aggs; ++a) {
            float contribution = match ? aggs[a] : 0.0f;
            group_sums[a] = simd_sum(contribution);
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

                        uint agg_base = probe_slot * 8;
                        for (uint a = 0; a < num_aggs; ++a) {
                            atomicAddF32Bits(&ht_agg_bits[agg_base + a], group_sums[a]);
                        }
                        if (num_aggs >= 2u) {
                            atomic_fetch_add_explicit(&ht_agg_bits[agg_base + 1], group_count, memory_order_relaxed);
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
                        break;
                    }
                }
            }
        }
        
        if (match) done = true;
    }
}

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
                uint base_idx = probe_slot * 8;

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

// ============================================================================
// Reduction Kernels
// ============================================================================

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

// ============================================================================
// GroupBy Hash Table Stream Compaction
// ============================================================================

kernel void ht_mark_valid(
    device const uint32_t* ht_keys [[buffer(0)]],
    device uint32_t*       mark    [[buffer(1)]],
    constant uint32_t&     cap     [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= cap) return;
    mark[gid] = (ht_keys[gid * 8] != 0) ? 1u : 0u;
}

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
    constant uint32_t&     totalRows [[buffer(9)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= cap) return;
    if (mark[gid] == 0) return;
    uint dest = offsets[gid];
    for (uint k = 0; k < numKeys; ++k) {
        uint keyVal = ht_keys[gid * 8 + k];
        out_keys[k * totalRows + dest] = (keyVal > 0) ? (keyVal - 1) : 0;
    }
    for (uint a = 0; a < numAggs; ++a) {
        out_aggs[a * totalRows + dest] = ht_aggs[gid * 16 + a];
    }
}

#endif // AGGREGATEKERNELS_H
