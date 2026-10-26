#include "config.hpp"
#include "kernel.hpp"
#include "reference.hpp"
#include "results.hpp"

#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <iostream>
#include <vector>

using namespace gemm;
using namespace gemm::bench;

namespace {

int warmUp(GemmInstance& instance, int minRuns, int minMs) {
    using Clock = std::chrono::steady_clock;
    const auto start = Clock::now();
    const auto minTime = std::chrono::milliseconds(minMs);
    int runs = 0;
    while (runs < minRuns || Clock::now() - start < minTime) {
        instance.run();
        ++runs;
    }
    return runs;
}

void runCase(const GemmKernel& kernel, const KernelParams& params, const GemmProblem& problem,
             const GemmData& data) {
    auto instance = kernel.prepare(problem, params, data);

    std::vector<std::byte> c(static_cast<std::size_t>(problem.batch * problem.elemsC()) *
                             outputBytes(problem.dtype));
    instance->run();
    instance->readC(c);

    BenchCase bc{kernel.name(), kernel.backend(), params, problem, {}};
    bc.check = verify(problem, data, c, benchConfig().verifySamples);

    const std::string name = caseName(bc);
    INFO(name);
    CHECK(bc.check.maxError <= bc.check.tolerance);
    if (!bc.check.ok()) return;

    bc.warmupRuns = warmUp(*instance, benchConfig().warmupRuns, benchConfig().warmupMs);
    setCurrentCase(bc);
    BENCHMARK_ADVANCED(std::string(name))(Catch::Benchmark::Chronometer meter) {
        meter.measure([&] { instance->run(); });
    };
}

}

TEST_CASE("GEMM", "[benchmark]") {
    const BenchConfig& cfg = benchConfig();
    for (const auto& kernel : allKernels()) {
        if (!cfg.kernelSelected(kernel->name())) continue;
        const auto paramSets = expand(applyOverrides(kernel->paramSpace(), cfg.paramOverrides));
        std::size_t skipped = 0;
        for (const auto& problem : cfg.problems) {
            std::vector<KernelParams> supported;
            for (const auto& params : paramSets)
                if (kernel->supports(problem, params)) supported.push_back(params);
            skipped += paramSets.size() - supported.size();
            if (supported.empty()) continue;
            const GemmData data = makeData(problem);
            for (const auto& params : supported) runCase(*kernel, params, problem, data);
        }
        if (skipped > 0)
            std::cerr << kernel->name() << ": " << skipped << " of " << paramSets.size() * cfg.problems.size()
                      << " combinations are not supported by the kernel and were skipped"
                         " (check --dtype, --trans, --params)\n";
    }
}
