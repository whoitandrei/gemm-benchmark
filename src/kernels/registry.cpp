#include "kernel.hpp"

namespace gemm {

std::unique_ptr<GemmKernel> makeCpuTiledKernel();

std::vector<std::unique_ptr<GemmKernel>> allKernels() {
    std::vector<std::unique_ptr<GemmKernel>> kernels;
    kernels.push_back(makeCpuTiledKernel());
    return kernels;
}

}
