#include "problem.hpp"

#include <cstring>
#include <random>
#include <stdexcept>

namespace gemm {

std::string toString(DType t) {
    switch (t) {
    case DType::F32: return "f32";
    case DType::I8: return "i8";
    }
    return "?";
}

DType parseDType(const std::string& s) {
    if (s == "f32") return DType::F32;
    if (s == "i8") return DType::I8;
    throw std::invalid_argument("unknown dtype '" + s + "' (expected f32 or i8)");
}

std::size_t inputBytes(DType t) {
    switch (t) {
    case DType::F32: return sizeof(float);
    case DType::I8: return sizeof(int8_t);
    }
    return 0;
}

std::size_t outputBytes(DType t) {
    switch (t) {
    case DType::F32: return sizeof(float);
    case DType::I8: return sizeof(int32_t);
    }
    return 0;
}

double GemmProblem::bytes() const {
    const double in = static_cast<double>(inputBytes(dtype));
    const double out = static_cast<double>(outputBytes(dtype));
    return in * static_cast<double>(batch * elemsA() + batchesB() * elemsB()) +
           out * static_cast<double>(batch * elemsC());
}

std::string GemmProblem::toString() const {
    std::string s = gemm::toString(dtype) + " " + std::to_string(M) + "x" + std::to_string(N) +
                    "x" + std::to_string(K) + " b" + std::to_string(batch) + " " +
                    (transA ? "t" : "n") + (transB ? "t" : "n");
    if (broadcastB) s += " bcastB";
    return s;
}

namespace {

template <class T, class Dist>
std::vector<std::byte> randomBytes(int64_t count, std::mt19937& rng, Dist dist) {
    std::vector<std::byte> bytes(static_cast<std::size_t>(count) * sizeof(T));
    for (int64_t i = 0; i < count; ++i) {
        const T v = static_cast<T>(dist(rng));
        std::memcpy(bytes.data() + i * sizeof(T), &v, sizeof(T));
    }
    return bytes;
}

}

GemmData makeData(const GemmProblem& p, uint32_t seed) {
    std::mt19937 rng(seed);
    const int64_t countA = p.batch * p.elemsA();
    const int64_t countB = p.batchesB() * p.elemsB();
    GemmData d;
    switch (p.dtype) {
    case DType::F32: {
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        d.a = randomBytes<float>(countA, rng, dist);
        d.b = randomBytes<float>(countB, rng, dist);
        break;
    }
    case DType::I8: {
        std::uniform_int_distribution<int> dist(-127, 127);
        d.a = randomBytes<int8_t>(countA, rng, dist);
        d.b = randomBytes<int8_t>(countB, rng, dist);
        break;
    }
    }
    return d;
}

}
