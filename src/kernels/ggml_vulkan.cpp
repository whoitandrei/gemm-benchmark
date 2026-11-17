#include "half.hpp"
#include "kernel.hpp"

#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-vulkan.h>
#include <ggml.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace gemm {
namespace {

constexpr double kF16Rounding = 1.0 / 1024.0;

bool f16Mode() {
    static const bool f16 = std::getenv("GEMM_GGML_VULKAN_F16") != nullptr;
    return f16;
}

std::map<std::string, std::string>& matrixCoresByDevice() {
    static std::map<std::string, std::string> cores;
    return cores;
}

void forwardLog(ggml_log_level, const char* text, void*) {
    std::fputs(text, stderr);
    const std::string line(text);
    const std::string prefix = "ggml_vulkan: ";
    const std::string key = " | matrix cores: ";
    const std::size_t equals = line.find(" = ");
    const std::size_t cores = line.find(key);
    if (!line.starts_with(prefix) || equals == std::string::npos || cores == std::string::npos) return;
    std::string value = line.substr(cores + key.size());
    value.erase(value.find_last_not_of(" \r\n") + 1);
    matrixCoresByDevice()["Vulkan" + line.substr(prefix.size(), equals - prefix.size())] = value;
}

void enableEnv(const char* name) {
#ifdef _WIN32
    _putenv_s(name, "1");
#else
    setenv(name, "1", 1);
#endif
}

ggml_backend_dev_t pickDevice() {
    static const ggml_backend_dev_t device = [] {
        if (!f16Mode())
            for (const char* name : {"GGML_VK_DISABLE_F16", "GGML_VK_DISABLE_COOPMAT", "GGML_VK_DISABLE_COOPMAT2"})
                enableEnv(name);
        ggml_log_set(forwardLog, nullptr);
        ggml_backend_reg_t reg = ggml_backend_vk_reg();
        if (!reg) return ggml_backend_dev_t{nullptr};
        for (enum ggml_backend_dev_type type : {GGML_BACKEND_DEVICE_TYPE_GPU, GGML_BACKEND_DEVICE_TYPE_IGPU})
            for (std::size_t i = 0; i < ggml_backend_reg_dev_count(reg); ++i) {
                ggml_backend_dev_t dev = ggml_backend_reg_dev_get(reg, i);
                if (ggml_backend_dev_type(dev) == type) return dev;
            }
        return ggml_backend_dev_t{nullptr};
    }();
    return device;
}

class GgmlVulkanInstance final : public GemmInstance {
public:
    GgmlVulkanInstance(const GemmProblem& p, const GemmData& in) : p_(p) {
        try {
            init(in);
        } catch (...) {
            release();
            throw;
        }
    }

    ~GgmlVulkanInstance() override { release(); }

    void run() override {
        if (ggml_backend_graph_compute_async(backend_, graph_) != GGML_STATUS_SUCCESS)
            throw std::runtime_error("ggml_backend_graph_compute_async failed");
    }

    void sync() override { ggml_backend_synchronize(backend_); }

    void readC(std::span<std::byte> out) override {
        sync();
        std::vector<float> c(static_cast<std::size_t>(ggml_nelements(c_)));
        ggml_backend_tensor_get(c_, c.data(), 0, ggml_nbytes(c_));
        if (p_.acc == DType::F16) {
            std::vector<Half> h(c.size());
            std::transform(c.begin(), c.end(), h.begin(), toHalf);
            std::memcpy(out.data(), h.data(), std::min(out.size(), h.size() * sizeof(Half)));
        } else {
            std::memcpy(out.data(), c.data(), std::min(out.size(), c.size() * sizeof(float)));
        }
    }

private:
    void init(const GemmData& in) {
        backend_ = ggml_backend_dev_init(pickDevice(), nullptr);
        if (!backend_) throw std::runtime_error("ggml_backend_dev_init failed");

        ggml_init_params ip{ggml_tensor_overhead() * 4 + ggml_graph_overhead(), nullptr, true};
        ctx_ = ggml_init(ip);
        if (!ctx_) throw std::runtime_error("ggml_init failed");
        const ggml_type type = p_.dtype == DType::F16 ? GGML_TYPE_F16 : GGML_TYPE_F32;
        ggml_tensor* w = ggml_new_tensor_3d(ctx_, type, p_.K, p_.N, p_.batchesB());
        ggml_tensor* x = ggml_new_tensor_3d(ctx_, type, p_.K, p_.M, p_.batch);
        c_ = ggml_mul_mat(ctx_, w, x);
        if (p_.acc == DType::F32 && !ggml_prec_set_acc(c_, GGML_PREC_F32))
            throw std::runtime_error("ggml_prec_set_acc failed");
        graph_ = ggml_new_graph(ctx_);
        ggml_build_forward_expand(graph_, c_);
        buffer_ = ggml_backend_alloc_ctx_tensors(ctx_, backend_);
        if (!buffer_) throw std::runtime_error("ggml_backend_alloc_ctx_tensors failed");

        ggml_backend_tensor_set(x, in.a.data(), 0, in.a.size());
        const std::vector<std::byte> t = transposedB(p_, in);
        ggml_backend_tensor_set(w, t.data(), 0, t.size());
    }

    void release() {
        ggml_backend_buffer_free(buffer_);
        ggml_free(ctx_);
        ggml_backend_free(backend_);
        buffer_ = nullptr;
        ctx_ = nullptr;
        backend_ = nullptr;
    }

    GemmProblem p_;
    ggml_backend_t backend_ = nullptr;
    ggml_context* ctx_ = nullptr;
    ggml_backend_buffer_t buffer_ = nullptr;
    ggml_cgraph* graph_ = nullptr;
    ggml_tensor* c_ = nullptr;
};

std::string matrixCores() {
    const ggml_backend_dev_t dev = pickDevice();
    if (!dev) return "none";
    const auto& cores = matrixCoresByDevice();
    const auto found = cores.find(ggml_backend_dev_name(dev));
    return found == cores.end() ? "unknown" : found->second;
}

class GgmlVulkanKernel final : public GemmKernel {
public:
    std::string name() const override {
        if (!f16Mode()) return "ggml_vulkan";
        const std::string cores = matrixCores();
        return cores == "KHR_coopmat" || cores.starts_with("NV_coopmat2") ? "ggml_vulkan_tc" : "ggml_vulkan_f16";
    }

    std::string backend() const override { return "vulkan"; }

    std::string device() const override {
        const ggml_backend_dev_t dev = pickDevice();
        if (!dev) return "no Vulkan device";
        return std::string(ggml_backend_dev_description(dev)) +
               (f16Mode() ? ", f16 mode, matrix cores: " + matrixCores() : ", f32 mode: fp16 and coopmat off");
    }

    bool supports(const GemmProblem& p) const override {
        return (f16Mode() || p.dtype == DType::F32) && pickDevice();
    }

    double inputRounding(const GemmProblem& p) const override {
        return f16Mode() && p.dtype == DType::F32 ? kF16Rounding : 0.0;
    }

    std::unique_ptr<GemmInstance> prepare(const GemmProblem& p, const GemmData& in) const override {
        return std::make_unique<GgmlVulkanInstance>(p, in);
    }
};

}

std::unique_ptr<GemmKernel> makeGgmlVulkanKernel() { return std::make_unique<GgmlVulkanKernel>(); }

}
