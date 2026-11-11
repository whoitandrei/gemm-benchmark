#include "half.hpp"
#include "kernel.hpp"
#include "measure.hpp"
#include "reference.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstring>
#include <stdexcept>

using namespace gemm;

TEST_CASE("reference accepts the right product and rejects a wrong one", "[reference]") {
    const GemmProblem p{.M = 2, .K = 2, .N = 2};
    const float a[] = {1, 2, 3, 4}, b[] = {5, 6, 7, 8};
    GemmData d{std::vector<std::byte>(sizeof a), std::vector<std::byte>(sizeof b)};
    std::memcpy(d.a.data(), a, sizeof a);
    std::memcpy(d.b.data(), b, sizeof b);

    float c[] = {19, 22, 43, 50};
    const std::span<const std::byte> view(reinterpret_cast<const std::byte*>(c), sizeof c);
    CHECK(verify(p, d, view).ok());
    c[3] = 51;
    CHECK_FALSE(verify(p, d, view).ok());
}

TEST_CASE("f16 conversion is exact and rounds to nearest even", "[half]") {
    int errors = 0;
    for (uint32_t bits = 0; bits < 0x10000; ++bits) {
        const Half h{static_cast<uint16_t>(bits)};
        const bool nan = (bits & 0x7C00u) == 0x7C00u && (bits & 0x3FFu) != 0;
        if (nan ? !std::isnan(toFloat(h)) : toHalf(toFloat(h)).bits != h.bits) ++errors;
    }
    CHECK(errors == 0);
    CHECK(toHalf(1.0f + std::ldexp(1.0f, -11)).bits == 0x3C00);
    CHECK(toHalf(1.0f + 3 * std::ldexp(1.0f, -11)).bits == 0x3C02);
}

TEST_CASE("kernels match the reference after queued runs", "[correctness]") {
    const GemmProblem shapes[] = {
        {.batch = 1, .M = 7, .K = 5, .N = 13},
        {.batch = 3, .M = 65, .K = 67, .N = 129},
        {.batch = 3, .M = 65, .K = 67, .N = 129, .broadcastB = true},
    };
    const std::pair<DType, DType> types[] = {
        {DType::F32, DType::F32}, {DType::F16, DType::F32}, {DType::F16, DType::F16}};

    for (const auto& kernel : allKernels())
        for (GemmProblem p : shapes)
            for (const auto& [dtype, acc] : types) {
                p.dtype = dtype;
                p.acc = acc;
                if (!kernel->supports(p)) continue;
                INFO(kernel->name() << " " << p.toString());

                const GemmData data = makeData(p);
                auto instance = kernel->prepare(p, data);
                for (int i = 0; i < 3; ++i) instance->run();
                instance->sync();
                std::vector<std::byte> c(p.bytesC());
                instance->readC(c);
                const VerifyResult r = verify(p, data, c);
                CHECK(r.maxError <= r.tolerance);
            }
}

TEST_CASE("only kernels with a GPU timer report gpu_ms", "[gpu-time]") {
    const GemmProblem p{.batch = 3, .M = 64, .K = 64, .N = 64};
    const GemmData data = makeData(p);
    for (const auto& kernel : allKernels()) {
        if (!kernel->supports(p)) continue;
        INFO(kernel->name());
        auto instance = kernel->prepare(p, data);
        instance->beginGpuTiming();
        instance->run();
        const double ms = instance->endGpuTimingMs();
        if (kernel->backend() == "cuda")
            CHECK(ms > 0.0);
        else
            CHECK(std::isnan(ms));
    }
}

namespace {

class Counting : public GemmInstance {
public:
    int runs = 0;
    int syncs = 0;
    void run() override { ++runs; }
    void sync() override { ++syncs; }
    void readC(std::span<std::byte>) override {}
};

class Timed final : public Counting {
public:
    int blocks = 0;
    void beginGpuTiming() override {}
    double endGpuTimingMs() override { return 10.0 * ++blocks; }
};

}

TEST_CASE("measure follows Adept op_bench", "[measure]") {
    const MeasureOptions options{.warmup = 3, .iterations = 4, .rounds = 5};

    Timed timed;
    const Timing t = measure(timed, options);
    CHECK(timed.runs == 3 + 5 * 4);
    CHECK(timed.blocks == 5);
    CHECK(t.gpuMs == 30.0 / 4);
    CHECK(t.wallMs >= t.cpuMs);

    Counting plain;
    const Timing p = measure(plain, options);
    CHECK(plain.syncs == 1 + 5 + 5);
    CHECK(std::isnan(p.gpuMs));

    CHECK(median({4.0, 1.0, 3.0, 2.0}) == 2.5);
    CHECK_THROWS_AS(measure(plain, {.iterations = 0}), std::invalid_argument);
}
