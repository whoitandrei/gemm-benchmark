#include "reference.hpp"

#include "half.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>

namespace gemm {
namespace {

struct Index {
    int64_t b, i, j;
};

std::vector<Index> pickElements(const GemmProblem& p, int64_t samples) {
    std::vector<Index> idx;
    if (samples == 0) {
        idx.reserve(static_cast<std::size_t>(p.batch * p.elemsC()));
        for (int64_t b = 0; b < p.batch; ++b)
            for (int64_t i = 0; i < p.M; ++i)
                for (int64_t j = 0; j < p.N; ++j) idx.push_back({b, i, j});
        return idx;
    }
    for (int64_t b : {int64_t{0}, p.batch - 1})
        for (int64_t i : {int64_t{0}, p.M - 1})
            for (int64_t j : {int64_t{0}, p.N - 1}) idx.push_back({b, i, j});
    std::mt19937_64 rng(7);
    std::uniform_int_distribution<int64_t> db(0, p.batch - 1), di(0, p.M - 1), dj(0, p.N - 1);
    for (int64_t s = 0; s < samples; ++s) idx.push_back({db(rng), di(rng), dj(rng)});
    return idx;
}

double value(float v) { return v; }
double value(Half v) { return toFloat(v); }

template <class TIn, class TOut>
VerifyResult verifyTyped(const GemmProblem& p, const GemmData& in, std::span<const std::byte> c,
                         int64_t samples) {
    const auto* A = reinterpret_cast<const TIn*>(in.a.data());
    const auto* B = reinterpret_cast<const TIn*>(in.b.data());
    const auto* C = reinterpret_cast<const TOut*>(c.data());

    VerifyResult r;
    for (const Index& e : pickElements(p, samples)) {
        const TIn* a = A + e.b * p.elemsA() + e.i * p.K;
        const TIn* b = B + (p.broadcastB ? 0 : e.b) * p.elemsB() + e.j;
        double ref = 0.0;
        double absSum = 0.0;
        for (int64_t k = 0; k < p.K; ++k) {
            const double prod = value(a[k]) * value(b[k * p.N]);
            ref += prod;
            absSum += std::abs(prod);
        }
        const double got = value(C[e.b * p.elemsC() + e.i * p.N + e.j]);
        const double err = std::abs(got - ref) / std::max(absSum, 1e-30);
        r.maxError = std::max(r.maxError, err);
        ++r.checked;
    }
    return r;
}

}

VerifyResult verify(const GemmProblem& p, const GemmData& in, std::span<const std::byte> c,
                    int64_t samples) {
    VerifyResult r;
    if (p.dtype == DType::F32 && p.acc == DType::F32)
        r = verifyTyped<float, float>(p, in, c, samples);
    else if (p.dtype == DType::F16 && p.acc == DType::F32)
        r = verifyTyped<Half, float>(p, in, c, samples);
    else if (p.dtype == DType::F16 && p.acc == DType::F16)
        r = verifyTyped<Half, Half>(p, in, c, samples);
    else
        throw std::invalid_argument("unsupported types " + toString(p.dtype) + "/" + toString(p.acc));

    const int exponent = p.acc == DType::F16 ? -11 : -24;
    r.tolerance = static_cast<double>(p.K + 2) * std::ldexp(1.0, exponent);
    return r;
}

}
