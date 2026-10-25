#pragma once

#include "params.hpp"
#include "problem.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace gemm::bench {

struct BenchOptions {
    std::string kernels;
    std::string shapes = "128,256,512";
    std::string batches = "1";
    std::string dtypes = "f32,i8";
    std::string trans = "nn";
    bool broadcastB = false;
    std::string params;
    std::string csv;
    int verifySamples = 1024;
    int warmupRuns = 3;
    int warmupMs = 200;
};

struct BenchConfig {
    std::vector<std::string> kernels;
    std::vector<GemmProblem> problems;
    ParamSpace paramOverrides;
    std::string csvPath;
    int64_t verifySamples = 1024;
    int warmupRuns = 3;
    int warmupMs = 200;

    bool kernelSelected(const std::string& name) const;
};

BenchConfig parseConfig(const BenchOptions& o);

BenchConfig& benchConfig();

}
