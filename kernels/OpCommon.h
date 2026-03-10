// OpCommon.h — Shared constants and inline helpers for Metal GPU kernels
#ifndef OPCOMMON_H
#define OPCOMMON_H

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

// Atomic float helpers (CAS-based)
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

#endif // OPCOMMON_H
