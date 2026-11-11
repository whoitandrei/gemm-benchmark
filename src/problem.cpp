#include "problem.hpp"

#include "half.hpp"

#include <cstring>
#include <random>

namespace gemm {

std::string toString(DType t) { return t == DType::F16 ? "f16" : "f32"; }

std::size_t sizeOf(DType t) { return t == DType::F16 ? sizeof(Half) : sizeof(float); }

std::string GemmProblem::shape() const {
    return std::to_string(batch) + "x" + std::to_string(M) + "x" + std::to_string(K) + "x" + std::to_string(N);
}

std::string GemmProblem::toString() const {
    return shape() + " " + gemm::toString(dtype) + "/" + gemm::toString(acc) + (broadcastB ? " shared B" : "");
}

namespace {

template <class T, class Gen>
std::vector<std::byte> randomBytes(int64_t count, Gen next) {
    std::vector<std::byte> bytes(static_cast<std::size_t>(count) * sizeof(T));
    for (int64_t i = 0; i < count; ++i) {
        const T v = next();
        std::memcpy(bytes.data() + i * sizeof(T), &v, sizeof(T));
    }
    return bytes;
}

}

GemmData makeData(const GemmProblem& p, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> real(-1.0f, 1.0f);
    const int64_t countA = p.batch * p.elemsA();
    const int64_t countB = p.batchesB() * p.elemsB();
    if (p.dtype == DType::F16) {
        const auto f16 = [&] { return toHalf(real(rng)); };
        return {randomBytes<Half>(countA, f16), randomBytes<Half>(countB, f16)};
    }
    const auto f32 = [&] { return real(rng); };
    return {randomBytes<float>(countA, f32), randomBytes<float>(countB, f32)};
}

std::vector<std::byte> transposedB(const GemmProblem& p, const GemmData& in) {
    const std::size_t size = sizeOf(p.dtype);
    std::vector<std::byte> t(in.b.size());
    for (int64_t b = 0; b < p.batchesB(); ++b)
        for (int64_t k = 0; k < p.K; ++k)
            for (int64_t n = 0; n < p.N; ++n)
                std::memcpy(&t[static_cast<std::size_t>(b * p.elemsB() + n * p.K + k) * size],
                            &in.b[static_cast<std::size_t>(b * p.elemsB() + k * p.N + n) * size], size);
    return t;
}

}
