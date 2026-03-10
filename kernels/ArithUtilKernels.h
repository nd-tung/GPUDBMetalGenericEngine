// ArithUtilKernels.h — Arithmetic, gather/scatter, cast, and utility kernels
#ifndef ARITHUTILKERNELS_H
#define ARITHUTILKERNELS_H

// ============================================================================
// Gather / Cast
// ============================================================================

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

// ============================================================================
// Domain-Specific Arithmetic (TPC-H revenue/charge)
// ============================================================================

kernel void compute_revenue_ep_disc(device const float* extendedprice [[buffer(0)]],
                                    device const float* discount [[buffer(1)]],
                                    device float* revenue [[buffer(2)]],
                                    constant uint& row_count [[buffer(3)]],
                                    uint gid [[thread_position_in_grid]]) {
    if (gid >= row_count) return;
    revenue[gid] = extendedprice[gid] * (1.0f - discount[gid]);
}

kernel void compute_charge_ep_disc_tax(device const float* extendedprice [[buffer(0)]],
                                       device const float* discount [[buffer(1)]],
                                       device const float* tax [[buffer(2)]],
                                       device float* charge [[buffer(3)]],
                                       constant uint& row_count [[buffer(4)]],
                                       uint gid [[thread_position_in_grid]]) {
    if (gid >= row_count) return;
    charge[gid] = extendedprice[gid] * (1.0f - discount[gid]) * (1.0f + tax[gid]);
}

// ============================================================================
// Generic Arithmetic (col-col, col-scalar, scalar-col)
// ============================================================================

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
    out[gid] = (abs(denom) > 1e-9) ? (colA[gid] / denom) : 0.0f;
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

kernel void arith_sub_f32_col_col(
    const device float* a [[buffer(0)]],
    const device float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint32_t& count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    out[gid] = a[gid] - b[gid];
}

kernel void arith_sub_f32_scalar_col(
    constant float& a [[buffer(0)]],
    const device float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint32_t& count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    out[gid] = a - b[gid];
}

kernel void arith_sub_f32_col_scalar(
    const device float* a [[buffer(0)]],
    constant float& b [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint32_t& count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    out[gid] = a[gid] - b;
}

kernel void arith_add_f32_col_col(
    const device float* a [[buffer(0)]],
    const device float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint32_t& count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    out[gid] = a[gid] + b[gid];
}

kernel void arith_add_f32_scalar_col(
    constant float& a [[buffer(0)]],
    const device float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint32_t& count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    out[gid] = a + b[gid];
}

kernel void arith_add_f32_col_scalar(
    const device float* a [[buffer(0)]],
    constant float& b [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint32_t& count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    out[gid] = a[gid] + b;
}

kernel void arith_floor_f32(const device float* col [[buffer(0)]],
                            device float* out [[buffer(1)]],
                            constant uint& count [[buffer(2)]],
                            uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    out[gid] = floor(col[gid]);
}

// ============================================================================
// Utility Kernels
// ============================================================================

kernel void limit_copy(const device float* input [[buffer(0)]],
                       device float* output [[buffer(1)]],
                       constant uint& limit [[buffer(2)]],
                       uint gid [[thread_position_in_grid]]) {
    if (gid < limit) {
        output[gid] = input[gid];
    }
}

kernel void fill_u32(
    device uint32_t* buf [[buffer(0)]],
    constant uint32_t& val [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    buf[gid] = val;
}

kernel void fill_f32(
    device float* buf [[buffer(0)]],
    constant float& val [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    buf[gid] = val;
}

kernel void iota_u32(
    device uint32_t* buf [[buffer(0)]],
    constant uint& count [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    buf[gid] = gid;
}

kernel void arith_add_const_u32(
    const device uint32_t* in [[buffer(0)]],
    constant uint32_t& val [[buffer(1)]],
    device uint32_t* out [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    out[gid] = in[gid] + val;
}

kernel void nonnull_indicator_f32(
    const device float* in [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    out[gid] = (in[gid] != 0.0f) ? 1.0f : 0.0f;
}

kernel void hash_combine_u64_u32(
    device uint64_t* u64buf [[buffer(0)]],
    const device uint32_t* u32key [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    u64buf[gid] = u64buf[gid] * 0x9E3779B97F4A7C15ULL + u32key[gid];
}

// ============================================================================
// Scatter / Copy / Bitcast
// ============================================================================

kernel void scatter_constant_f32(
    device float* output [[buffer(0)]],
    const device uint32_t* indices [[buffer(1)]],
    constant float& val [[buffer(2)]],
    constant uint32_t& count [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    if (index < count) {
        output[indices[index]] = val;
    }
}

kernel void scatter_f32_indexed(
    const device float* input [[buffer(0)]],
    const device uint32_t* indices [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint32_t& count [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    if (index < count) {
        output[indices[index]] = input[index];
    }
}

kernel void copy_add_u32(
    const device uint32_t* in  [[buffer(0)]],
    device uint32_t*       out [[buffer(1)]],
    constant uint32_t&     add [[buffer(2)]],
    constant uint32_t&     count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    out[gid] = in[gid] + add;
}

kernel void bitcast_f32_to_u32(
    const device float*    in  [[buffer(0)]],
    device uint32_t*       out [[buffer(1)]],
    constant uint32_t&     count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    out[gid] = as_type<uint32_t>(in[gid]);
}

kernel void bitcast_u32_to_f32(
    const device uint32_t* in  [[buffer(0)]],
    device float*          out [[buffer(1)]],
    constant uint32_t&     count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    out[gid] = as_type<float>(in[gid]);
}

kernel void scatter_one_u8(
    const device uint32_t* indices [[buffer(0)]],
    device uint8_t*        mask    [[buffer(1)]],
    constant uint32_t&     count   [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    mask[indices[gid]] = 1;
}

// ============================================================================
// Date/Year Extraction
// ============================================================================

kernel void extract_year_u32_to_f32(
    const device uint32_t* in [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint32_t& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    out[gid] = (float)(in[gid] / 10000);
}

kernel void extract_year_u32_to_u32(
    const device uint32_t* in  [[buffer(0)]],
    device uint32_t*       out [[buffer(1)]],
    constant uint32_t&     count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    uint32_t val = in[gid];
    if (val > 19000000) {
        out[gid] = val / 10000;
    } else {
        out[gid] = 1970 + uint32_t(float(val) / 365.25f);
    }
}

#endif // ARITHUTILKERNELS_H
