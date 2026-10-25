#pragma once

#include "params.hpp"
#include "problem.hpp"

#include <cstddef>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace gemm {

class GemmInstance {
public:
    virtual ~GemmInstance() = default;

    virtual void run() = 0;

    virtual void readC(std::span<std::byte> out) = 0;
};

class GemmKernel {
public:
    virtual ~GemmKernel() = default;

    virtual std::string name() const = 0;
    virtual std::string backend() const = 0;
    virtual ParamSpace paramSpace() const { return {}; }
    virtual bool supports(const GemmProblem& p, const KernelParams& params) const = 0;
    virtual std::unique_ptr<GemmInstance> prepare(const GemmProblem& p, const KernelParams& params,
                                                  const GemmData& in) const = 0;
};

std::vector<std::unique_ptr<GemmKernel>> allKernels();

}
