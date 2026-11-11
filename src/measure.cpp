#include "measure.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace gemm {

double median(std::vector<double> values) {
    if (values.empty()) throw std::invalid_argument("median requires at least one sample");
    if (std::any_of(values.begin(), values.end(), [](double v) { return std::isnan(v); }))
        return std::numeric_limits<double>::quiet_NaN();
    std::sort(values.begin(), values.end());
    const std::size_t mid = values.size() / 2;
    if (values.size() % 2 != 0) return values[mid];
    return (values[mid - 1] + values[mid]) * 0.5;
}

Timing measure(GemmInstance& instance, const MeasureOptions& options) {
    if (options.warmup < 0) throw std::invalid_argument("warmup must not be negative");
    if (options.iterations <= 0) throw std::invalid_argument("iterations must be positive");
    if (options.rounds <= 0) throw std::invalid_argument("rounds must be positive");

    using Clock = std::chrono::steady_clock;
    using Ms = std::chrono::duration<double, std::milli>;
    const double iterations = static_cast<double>(options.iterations);

    for (int i = 0; i < options.warmup; ++i) instance.run();
    instance.sync();

    std::vector<double> wall, gpu, cpu;
    for (int round = 0; round < options.rounds; ++round) {
        instance.sync();
        instance.beginGpuTiming();
        const auto cpuStart = Clock::now();
        for (int i = 0; i < options.iterations; ++i) instance.run();
        const auto cpuEnd = Clock::now();
        const double gpuBlockMs = instance.endGpuTimingMs();
        const auto wallEnd = Clock::now();

        wall.push_back(Ms(wallEnd - cpuStart).count() / iterations);
        cpu.push_back(Ms(cpuEnd - cpuStart).count() / iterations);
        gpu.push_back(gpuBlockMs / iterations);
    }
    return {median(std::move(wall)), median(std::move(gpu)), median(std::move(cpu))};
}

}
