#include "problem.hpp"

#include "half.hpp"

#include <cmath>
#include <cstring>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>

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

constexpr float kConstValue = 0.01f;
constexpr float kNormalStd = 0.02f;

std::optional<float> parseFloat(const std::string& text) {
    try {
        std::size_t used = 0;
        const float v = std::stof(text, &used);
        if (used == text.size() && std::isfinite(v)) return v;
    } catch (const std::logic_error&) {
    }
    return std::nullopt;
}

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

std::string Init::toString() const {
    std::ostringstream out;
    switch (kind) {
        case Kind::Const:
            out << "const " << kConstValue;
            break;
        case Kind::Normal:
            out << "normal mean 0 std " << kNormalStd;
            break;
        case Kind::Uniform:
            out << "uniform [" << low << ", " << high << ")";
            break;
    }
    return out.str();
}

Init parseInit(const std::string& text) {
    if (text == "const") return {};
    if (text == "normal") return {.kind = Init::Kind::Normal};
    const std::string prefix = "uniform:";
    if (text.starts_with(prefix)) {
        const std::string range = text.substr(prefix.size());
        const std::size_t comma = range.find(',');
        if (comma != std::string::npos) {
            const auto low = parseFloat(range.substr(0, comma));
            const auto high = parseFloat(range.substr(comma + 1));
            if (low && high && *low < *high && std::isfinite(*high - *low))
                return {.kind = Init::Kind::Uniform, .low = *low, .high = *high};
        }
    }
    throw std::invalid_argument("--init must be const, normal or uniform:LOW,HIGH with LOW < HIGH, got '" + text +
                                "'");
}

GemmData makeData(const GemmProblem& p, const Init& init, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> normal(0.0f, kNormalStd);
    std::uniform_real_distribution<float> uniform(init.low, init.high);
    const auto next = [&] {
        switch (init.kind) {
            case Init::Kind::Normal:
                return normal(rng);
            case Init::Kind::Uniform:
                return uniform(rng);
            case Init::Kind::Const:
                break;
        }
        return kConstValue;
    };
    const int64_t countA = p.batch * p.elemsA();
    const int64_t countB = p.batchesB() * p.elemsB();
    if (p.dtype == DType::F16) {
        const auto f16 = [&] { return toHalf(next()); };
        return {randomBytes<Half>(countA, f16), randomBytes<Half>(countB, f16)};
    }
    return {randomBytes<float>(countA, next), randomBytes<float>(countB, next)};
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
