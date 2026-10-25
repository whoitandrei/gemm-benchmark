#pragma once

#include "params.hpp"
#include "problem.hpp"
#include "reference.hpp"

#include <string>

namespace gemm::bench {

struct BenchCase {
    std::string kernel;
    std::string backend;
    KernelParams params;
    GemmProblem problem;
    VerifyResult check;
    int warmupRuns = 0;
};

void setCurrentCase(const BenchCase& c);

std::string caseName(const BenchCase& c);

}
