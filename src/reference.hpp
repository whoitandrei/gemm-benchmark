#pragma once

#include "problem.hpp"

#include <cstddef>
#include <cstdint>
#include <span>

namespace gemm {

struct VerifyResult {
    double maxError = 0.0;
    double tolerance = 0.0;
    int64_t checked = 0;

    bool ok() const { return maxError <= tolerance; }
};

VerifyResult verify(const GemmProblem& p, const GemmData& in, std::span<const std::byte> c,
                    int64_t samples = 0);

}
