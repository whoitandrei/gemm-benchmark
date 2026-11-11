#include "kernel.hpp"

namespace gemm {

#ifdef GEMM_WITH_GGML
std::unique_ptr<GemmKernel> makeGgmlCpuKernel();
#endif
#ifdef GEMM_WITH_XNNPACK
std::unique_ptr<GemmKernel> makeXnnpackKernel();
#endif
#ifdef GEMM_WITH_GGML_VULKAN
std::unique_ptr<GemmKernel> makeGgmlVulkanKernel();
#endif
#ifdef GEMM_WITH_CUBLAS
std::unique_ptr<GemmKernel> makeCublasKernel(int tc);
#endif

std::vector<std::unique_ptr<GemmKernel>> allKernels() {
    std::vector<std::unique_ptr<GemmKernel>> kernels;
#ifdef GEMM_WITH_GGML
    kernels.push_back(makeGgmlCpuKernel());
#endif
#ifdef GEMM_WITH_XNNPACK
    kernels.push_back(makeXnnpackKernel());
#endif
#ifdef GEMM_WITH_GGML_VULKAN
    kernels.push_back(makeGgmlVulkanKernel());
#endif
#ifdef GEMM_WITH_CUBLAS
    kernels.push_back(makeCublasKernel(0));
    kernels.push_back(makeCublasKernel(1));
#endif
    return kernels;
}

}
