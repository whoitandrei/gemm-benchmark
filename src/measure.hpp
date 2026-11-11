#pragma once

#include "kernel.hpp"

#include <vector>

namespace gemm {

struct MeasureOptions {
    int warmup = 5;
    int iterations = 20;
    int rounds = 5;
};

struct Timing {
    double wallMs = 0.0;
    double gpuMs = 0.0;
    double cpuMs = 0.0;
};

double median(std::vector<double> values);

Timing measure(GemmInstance& instance, const MeasureOptions& options);

}
