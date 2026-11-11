#pragma once

#include <bit>
#include <cmath>
#include <cstdint>

namespace gemm {

// f16 нужен потому, что ggml vulkan умеет только его
// также cublas ядра в основном умножают f16 матрицы с аккумуляцией в f32
// для текущего бенчмарка или для дальнейшего расширения в adept это будет полезно
     
struct Half {
    uint16_t bits = 0;
};

inline float toFloat(Half h) {
    const uint32_t sign = static_cast<uint32_t>(h.bits & 0x8000u) << 16;
    const uint32_t exponent = (h.bits >> 10) & 0x1Fu;
    const uint32_t mantissa = h.bits & 0x3FFu;
    if (exponent == 0) {
        const float subnormal = std::ldexp(static_cast<float>(mantissa), -24);
        return sign ? -subnormal : subnormal;
    }
    if (exponent == 0x1F) return std::bit_cast<float>(sign | 0x7F800000u | (mantissa << 13));
    return std::bit_cast<float>(sign | ((exponent + 112) << 23) | (mantissa << 13));
}

inline Half toHalf(float f) {
    const uint32_t x = std::bit_cast<uint32_t>(f);
    const auto sign = static_cast<uint16_t>((x >> 16) & 0x8000u);
    const uint32_t abs = x & 0x7FFFFFFFu;
    if (abs > 0x7F800000u) return {static_cast<uint16_t>(sign | 0x7E00u)};
    if (abs >= 0x477FF000u) return {static_cast<uint16_t>(sign | 0x7C00u)};
    if (abs < 0x33000000u) return {sign};

    uint32_t result;
    uint32_t rest;
    uint32_t halfway;
    if (abs < 0x38800000u) {
        const uint32_t shift = 126 - (abs >> 23);
        const uint32_t mantissa = (abs & 0x7FFFFFu) | 0x800000u;
        result = mantissa >> shift;
        rest = mantissa & ((1u << shift) - 1);
        halfway = 1u << (shift - 1);
    } else {
        result = (abs - 0x38000000u) >> 13;
        rest = abs & 0x1FFFu;
        halfway = 0x1000u;
    }
    if (rest > halfway || (rest == halfway && (result & 1u))) ++result;
    return {static_cast<uint16_t>(sign | result)};
}

}
