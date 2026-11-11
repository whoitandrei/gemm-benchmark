#pragma once

#include "problem.hpp"

#include <cstddef>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace gemm {

class GemmInstance {
public:
    virtual ~GemmInstance() = default;

    virtual void run() = 0;

    virtual void sync() {}

    virtual void beginGpuTiming() {}

    virtual double endGpuTimingMs() {
        sync();
        return std::numeric_limits<double>::quiet_NaN();
    }

    virtual void readC(std::span<std::byte> out) = 0;
};

class GemmKernel {
public:
    virtual ~GemmKernel() = default;

    virtual std::string name() const = 0;
    virtual std::string backend() const = 0;
    virtual std::string device() const = 0;
    virtual bool supports(const GemmProblem& p) const = 0;
    virtual std::unique_ptr<GemmInstance> prepare(const GemmProblem& p, const GemmData& in) const = 0;
};

std::vector<std::unique_ptr<GemmKernel>> allKernels();

}
