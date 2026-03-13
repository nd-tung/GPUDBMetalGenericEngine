// StringKernels.h — String filter, comparison, hash, and utility kernels
#ifndef STRINGKERNELS_H
#define STRINGKERNELS_H

// ============================================================================
// String Pattern Filters
// ============================================================================

kernel void filter_string_prefix(const device char* chars [[buffer(0)]],
                                   const device uint32_t* offsets [[buffer(1)]],
                                   const device uint32_t* lengths [[buffer(2)]],
                                   device uint8_t* out_mask [[buffer(3)]],
                                   const device char* pattern [[buffer(4)]],
                                   constant uint& pattern_len [[buffer(5)]],
                                   constant uint& row_count [[buffer(6)]],
                                   uint gid [[thread_position_in_grid]]) {
    if (gid >= row_count) return;
    
    uint start = offsets[gid];
    uint len = lengths[gid];
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
                                   const device uint32_t* lengths [[buffer(2)]],
                                   device uint8_t* out_mask [[buffer(3)]],
                                   const device char* pattern [[buffer(4)]],
                                   constant uint& pattern_len [[buffer(5)]],
                                   constant uint& row_count [[buffer(6)]],
                                   uint gid [[thread_position_in_grid]]) {
    if (gid >= row_count) return;
    
    uint start = offsets[gid];
    uint len = lengths[gid];
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
                                   const device uint32_t* lengths [[buffer(2)]],
                                   device uint8_t* out_mask [[buffer(3)]],
                                   const device char* pattern [[buffer(4)]],
                                   constant uint& pattern_len [[buffer(5)]],
                                   constant uint& row_count [[buffer(6)]],
                                   uint gid [[thread_position_in_grid]]) {
    if (gid >= row_count) return;
    
    uint start = offsets[gid];
    uint len = lengths[gid];
    const device char* str = chars + start;
    
    if (pattern_len == 0) {
        out_mask[gid] = 1;
        return;
    }

    if (pattern_len > len) {
        out_mask[gid] = 0;
        return;
    }
    
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

kernel void filter_string_multi_contains(const device char* chars [[buffer(0)]],
                                         const device uint32_t* offsets [[buffer(1)]],
                                         const device uint32_t* lengths [[buffer(2)]],
                                         device uint8_t* out_mask [[buffer(3)]],
                                         const device char* patterns [[buffer(4)]],
                                         const device uint32_t* pat_offsets [[buffer(5)]],
                                         const device uint32_t* pat_lengths [[buffer(6)]],
                                         constant uint& num_segments [[buffer(7)]],
                                         constant uint& row_count [[buffer(8)]],
                                         uint gid [[thread_position_in_grid]]) {
    if (gid >= row_count) return;

    uint str_start = offsets[gid];
    uint str_len   = lengths[gid];
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
                search_from = i + plen;
                seg_found = true;
                break;
            }
        }
        if (!seg_found) { all_found = false; break; }
    }
    out_mask[gid] = all_found ? 1 : 0;
}

// ============================================================================
// String Comparison Filters
// ============================================================================

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
                                   const device uint32_t* lengths [[buffer(2)]],
                                   device uint8_t* out_mask [[buffer(3)]],
                                   const device char* pattern [[buffer(4)]],
                                   constant uint& pattern_len [[buffer(5)]],
                                   constant uint& row_count [[buffer(6)]],
                                   uint gid [[thread_position_in_grid]]) {
    if (gid >= row_count) return;
    uint start = offsets[gid];
    uint len = lengths[gid];
    const device char* str = chars + start;
    out_mask[gid] = (compare_str(str, len, pattern, pattern_len) == 0) ? 1 : 0;
}

kernel void filter_string_ne(const device char* chars [[buffer(0)]],
                                   const device uint32_t* offsets [[buffer(1)]],
                                   const device uint32_t* lengths [[buffer(2)]],
                                   device uint8_t* out_mask [[buffer(3)]],
                                   const device char* pattern [[buffer(4)]],
                                   constant uint& pattern_len [[buffer(5)]],
                                   constant uint& row_count [[buffer(6)]],
                                   uint gid [[thread_position_in_grid]]) {
    if (gid >= row_count) return;
    uint start = offsets[gid];
    uint len = lengths[gid];
    const device char* str = chars + start;
    out_mask[gid] = (compare_str(str, len, pattern, pattern_len) != 0) ? 1 : 0;
}

kernel void filter_string_lt(const device char* chars [[buffer(0)]],
                                   const device uint32_t* offsets [[buffer(1)]],
                                   const device uint32_t* lengths [[buffer(2)]],
                                   device uint8_t* out_mask [[buffer(3)]],
                                   const device char* pattern [[buffer(4)]],
                                   constant uint& pattern_len [[buffer(5)]],
                                   constant uint& row_count [[buffer(6)]],
                                   uint gid [[thread_position_in_grid]]) {
    if (gid >= row_count) return;
    uint start = offsets[gid];
    uint len = lengths[gid];
    const device char* str = chars + start;
    out_mask[gid] = (compare_str(str, len, pattern, pattern_len) < 0) ? 1 : 0;
}

kernel void filter_string_le(const device char* chars [[buffer(0)]],
                                   const device uint32_t* offsets [[buffer(1)]],
                                   const device uint32_t* lengths [[buffer(2)]],
                                   device uint8_t* out_mask [[buffer(3)]],
                                   const device char* pattern [[buffer(4)]],
                                   constant uint& pattern_len [[buffer(5)]],
                                   constant uint& row_count [[buffer(6)]],
                                   uint gid [[thread_position_in_grid]]) {
    if (gid >= row_count) return;
    uint start = offsets[gid];
    uint len = lengths[gid];
    const device char* str = chars + start;
    out_mask[gid] = (compare_str(str, len, pattern, pattern_len) <= 0) ? 1 : 0;
}

kernel void filter_string_gt(const device char* chars [[buffer(0)]],
                                   const device uint32_t* offsets [[buffer(1)]],
                                   const device uint32_t* lengths [[buffer(2)]],
                                   device uint8_t* out_mask [[buffer(3)]],
                                   const device char* pattern [[buffer(4)]],
                                   constant uint& pattern_len [[buffer(5)]],
                                   constant uint& row_count [[buffer(6)]],
                                   uint gid [[thread_position_in_grid]]) {
    if (gid >= row_count) return;
    uint start = offsets[gid];
    uint len = lengths[gid];
    const device char* str = chars + start;
    out_mask[gid] = (compare_str(str, len, pattern, pattern_len) > 0) ? 1 : 0;
}

kernel void filter_string_ge(const device char* chars [[buffer(0)]],
                                   const device uint32_t* offsets [[buffer(1)]],
                                   const device uint32_t* lengths [[buffer(2)]],
                                   device uint8_t* out_mask [[buffer(3)]],
                                   const device char* pattern [[buffer(4)]],
                                   constant uint& pattern_len [[buffer(5)]],
                                   constant uint& row_count [[buffer(6)]],
                                   uint gid [[thread_position_in_grid]]) {
    if (gid >= row_count) return;
    uint start = offsets[gid];
    uint len = lengths[gid];
    const device char* str = chars + start;
    out_mask[gid] = (compare_str(str, len, pattern, pattern_len) >= 0) ? 1 : 0;
}

// ============================================================================
// String Utility Kernels
// ============================================================================

kernel void substring_flat(
    const device uint32_t* inOffsets  [[buffer(0)]],
    const device uint32_t* inLengths  [[buffer(1)]],
    device uint32_t*       outOffsets [[buffer(2)]],
    device uint32_t*       outLengths [[buffer(3)]],
    constant uint32_t&     startPos   [[buffer(4)]],
    constant uint32_t&     substrLen  [[buffer(5)]],
    constant uint32_t&     rowCount   [[buffer(6)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= rowCount) return;
    uint32_t origOff = inOffsets[tid];
    uint32_t origLen = inLengths[tid];
    uint32_t cStart = (startPos > 0) ? (startPos - 1) : 0;
    if (cStart >= origLen) {
        outOffsets[tid] = origOff + origLen;
        outLengths[tid] = 0;
    } else {
        outOffsets[tid] = origOff + cStart;
        uint32_t remaining = origLen - cStart;
        outLengths[tid] = (substrLen < remaining) ? substrLen : remaining;
    }
}

kernel void string_hash_encode_u32(
    const device uint8_t*  chars    [[buffer(0)]],
    const device uint32_t* offsets  [[buffer(1)]],
    const device uint32_t* lengths  [[buffer(2)]],
    device uint32_t*       encoded  [[buffer(3)]],
    constant uint32_t&     rowCount [[buffer(4)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= rowCount) return;
    uint32_t off = offsets[tid];
    uint32_t len = lengths[tid];
    uint32_t hashLen = min(len, 8u);
    uint32_t val = 0;
    for (uint32_t i = 0; i < hashLen; i++) {
        val = val * 256 + chars[off + i];
    }
    encoded[tid] = val;
}

kernel void string_fnv1a_u32(
    const device uint8_t*  chars    [[buffer(0)]],
    const device uint32_t* offsets  [[buffer(1)]],
    const device uint32_t* lengths  [[buffer(2)]],
    device uint32_t*       hashes   [[buffer(3)]],
    constant uint32_t&     rowCount [[buffer(4)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= rowCount) return;
    uint32_t off = offsets[tid];
    uint32_t len = lengths[tid];
    uint32_t h = 2166136261u;
    for (uint32_t i = 0; i < len; i++) {
        h ^= chars[off + i];
        h *= 16777619u;
    }
    if (h == 0u) h = 1u;
    if (h == 0xFFFFFFFFu) h = 0xFFFFFFFEu;
    hashes[tid] = h;
}

kernel void string_prefix_u64(
    const device uint8_t*  chars    [[buffer(0)]],
    const device uint32_t* offsets  [[buffer(1)]],
    const device uint32_t* lengths  [[buffer(2)]],
    device ulong*          prefixes [[buffer(3)]],
    constant uint32_t&     rowCount [[buffer(4)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= rowCount) return;
    uint32_t off = offsets[tid];
    uint32_t len = lengths[tid];
    uint32_t n = min(len, 8u);
    ulong val = 0;
    for (uint32_t i = 0; i < n; i++) {
        val = (val << 8) | ulong(chars[off + i]);
    }
    val <<= (8u - n) * 8u;
    prefixes[tid] = val;
}

kernel void gather_flat_string_chars(
    const device uint8_t*  srcChars   [[buffer(0)]],
    const device uint32_t* srcOffsets [[buffer(1)]],
    const device uint32_t* indices    [[buffer(2)]],
    const device uint32_t* dstOffsets [[buffer(3)]],
    const device uint32_t* dstLengths [[buffer(4)]],
    device uint8_t*        dstChars   [[buffer(5)]],
    constant uint32_t&     count      [[buffer(6)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= count) return;
    uint32_t srcRow = indices[tid];
    uint32_t srcOff = srcOffsets[srcRow];
    uint32_t dstOff = dstOffsets[tid];
    uint32_t len    = dstLengths[tid];
    for (uint32_t i = 0; i < len; i++) {
        dstChars[dstOff + i] = srcChars[srcOff + i];
    }
}

// ============================================================================
// E1a: FNV1a-64 hash of flat string columns (collision-resistant)
// ============================================================================
kernel void string_fnv1a_u64(
    const device uint8_t*  chars    [[buffer(0)]],
    const device uint32_t* offsets  [[buffer(1)]],
    const device uint32_t* lengths  [[buffer(2)]],
    device ulong*          hashes   [[buffer(3)]],
    constant uint32_t&     rowCount [[buffer(4)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= rowCount) return;
    uint32_t off = offsets[tid];
    uint32_t len = lengths[tid];
    ulong h = 14695981039346656037UL; // FNV1a-64 offset basis
    for (uint32_t i = 0; i < len; i++) {
        h ^= ulong(chars[off + i]);
        h *= 1099511628211UL;         // FNV1a-64 prime
    }
    if (h == 0UL) h = 1UL;
    hashes[tid] = h;
}

// ============================================================================
// E1a helper: XOR-fold 64-bit hashes to 32-bit
// ============================================================================
kernel void fold_u64_to_u32(
    const device ulong*    in      [[buffer(0)]],
    device uint32_t*       out     [[buffer(1)]],
    constant uint32_t&     count   [[buffer(2)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= count) return;
    ulong h = in[tid];
    uint32_t folded = uint32_t(h) ^ uint32_t(h >> 32);
    if (folded == 0u) folded = 1u;
    if (folded == 0xFFFFFFFFu) folded = 0xFFFFFFFEu;
    out[tid] = folded;
}

// ============================================================================
// E2a: Mark group boundaries in sorted key array.
// out[0] = 0, out[i] = 1 if keys[i] != keys[i-1], else 0.
// Works for both ulong prefix keys and uint32_t keys.
// ============================================================================
kernel void mark_sorted_boundaries_u64(
    const device ulong*    keys    [[buffer(0)]],
    device uint32_t*       marks   [[buffer(1)]],
    constant uint32_t&     count   [[buffer(2)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= count) return;
    if (tid == 0) {
        marks[0] = 0;
    } else {
        marks[tid] = (keys[tid] != keys[tid - 1]) ? 1 : 0;
    }
}

// ============================================================================
// E2a: Scatter ranks back to original row positions.
// rank[sortedIndices[tid]] = cumulativeRanks[tid]
// ============================================================================
kernel void scatter_rank_by_index(
    const device uint32_t* cumulativeRanks [[buffer(0)]],
    const device uint32_t* sortedIndices   [[buffer(1)]],
    device uint32_t*       rank            [[buffer(2)]],
    constant uint32_t&     count           [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= count) return;
    rank[sortedIndices[tid]] = cumulativeRanks[tid];
}

// ============================================================================
// E3a: Offset-shift kernel — add a constant to each offset value.
// Used for adjusting offsets when concatenating flat-string buffers.
// ============================================================================
kernel void offset_shift_u32(
    const device uint32_t* inOffsets  [[buffer(0)]],
    device uint32_t*       outOffsets [[buffer(1)]],
    constant uint32_t&     shift     [[buffer(2)]],
    constant uint32_t&     count     [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= count) return;
    outOffsets[tid] = inOffsets[tid] + shift;
}

#endif // STRINGKERNELS_H
