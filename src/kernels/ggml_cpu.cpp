#include "device.hpp"
#include "kernel.hpp"

#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-cpu.h>
#include <ggml.h>

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace gemm {
namespace {

class GgmlCpuInstance final : public GemmInstance {
public:
    GgmlCpuInstance(const GemmProblem& p, const GemmData& in) {
        try {
            init(p, cpuThreads(), in);
        } catch (...) {
            release();
            throw;
        }
    }

    ~GgmlCpuInstance() override { release(); }

    void run() override {
        if (ggml_backend_graph_compute(backend_, graph_) != GGML_STATUS_SUCCESS)
            throw std::runtime_error("ggml_backend_graph_compute failed");
    }

    void readC(std::span<std::byte> out) override {
        ggml_backend_tensor_get(c_, out.data(), 0, std::min(out.size(), ggml_nbytes(c_)));
    }

private:
    void init(const GemmProblem& p, int threads, const GemmData& in) {
        backend_ = ggml_backend_cpu_init();
        if (!backend_) throw std::runtime_error("ggml_backend_cpu_init failed");
        ggml_threadpool_params tp = ggml_threadpool_params_default(threads);
        threadpool_ = ggml_threadpool_new(&tp);
        if (!threadpool_) throw std::runtime_error("ggml_threadpool_new failed");
        ggml_backend_cpu_set_n_threads(backend_, threads);
        ggml_backend_cpu_set_threadpool(backend_, threadpool_);

        ggml_init_params ip{ggml_tensor_overhead() * 4 + ggml_graph_overhead(), nullptr, true};
        ctx_ = ggml_init(ip);
        if (!ctx_) throw std::runtime_error("ggml_init failed");
        ggml_tensor* w = ggml_new_tensor_3d(ctx_, GGML_TYPE_F32, p.K, p.N, p.batchesB());
        ggml_tensor* x = ggml_new_tensor_3d(ctx_, GGML_TYPE_F32, p.K, p.M, p.batch);
        c_ = ggml_mul_mat(ctx_, w, x);
        graph_ = ggml_new_graph(ctx_);
        ggml_build_forward_expand(graph_, c_);
        buffer_ = ggml_backend_alloc_ctx_tensors(ctx_, backend_);
        if (!buffer_) throw std::runtime_error("ggml_backend_alloc_ctx_tensors failed");

        ggml_backend_tensor_set(x, in.a.data(), 0, in.a.size());
        const std::vector<std::byte> t = transposedB(p, in);
        ggml_backend_tensor_set(w, t.data(), 0, t.size());
    }

    void release() {
        ggml_backend_buffer_free(buffer_);
        ggml_free(ctx_);
        ggml_backend_free(backend_);
        ggml_threadpool_free(threadpool_);
        buffer_ = nullptr;
        ctx_ = nullptr;
        backend_ = nullptr;
        threadpool_ = nullptr;
    }

    ggml_backend_t backend_ = nullptr;
    ggml_threadpool_t threadpool_ = nullptr;
    ggml_context* ctx_ = nullptr;
    ggml_backend_buffer_t buffer_ = nullptr;
    ggml_cgraph* graph_ = nullptr;
    ggml_tensor* c_ = nullptr;
};

class GgmlCpuKernel final : public GemmKernel {
public:
    std::string name() const override { return "ggml_cpu"; }
    std::string backend() const override { return "cpu"; }
    std::string device() const override { return cpuName(); }

    bool supports(const GemmProblem& p) const override { return p.dtype == DType::F32; }

    std::unique_ptr<GemmInstance> prepare(const GemmProblem& p, const GemmData& in) const override {
        return std::make_unique<GgmlCpuInstance>(p, in);
    }
};

}

std::unique_ptr<GemmKernel> makeGgmlCpuKernel() { return std::make_unique<GgmlCpuKernel>(); }

}
