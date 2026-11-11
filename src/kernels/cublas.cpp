#include "half.hpp"
#include "kernel.hpp"

#include <cublas_v2.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <climits>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

namespace gemm {
namespace {

void check(cudaError_t result, const char* what) {
    if (result != cudaSuccess) throw std::runtime_error(std::string(what) + " failed: " + cudaGetErrorString(result));
}

void check(cublasStatus_t result, const char* what) {
    if (result != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error(std::string(what) + " failed: " + cublasGetStatusString(result));
}

class Cuda {
public:
    static Cuda* get() {
        static const std::unique_ptr<Cuda> cuda = []() -> std::unique_ptr<Cuda> {
            try {
                std::unique_ptr<Cuda> created(new Cuda());
                std::cerr << "cuda: " << created->name_ << " (sm_" << created->arch_ << ")\n";
                return created;
            } catch (const std::exception& e) {
                std::cerr << "cuda: unavailable, cuBLAS kernel is skipped (" << e.what() << ")\n";
                return nullptr;
            }
        }();
        return cuda.get();
    }

    Cuda(const Cuda&) = delete;
    Cuda& operator=(const Cuda&) = delete;
    ~Cuda() { cublasDestroy(handle_); }

    cublasHandle_t handle() const { return handle_; }
    const std::string& deviceName() const { return name_; }
    int arch() const { return arch_; }

private:
    Cuda() {
        int count = 0;
        check(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
        if (count == 0) throw std::runtime_error("no CUDA devices");
        check(cudaSetDevice(0), "cudaSetDevice");
        cudaDeviceProp prop{};
        check(cudaGetDeviceProperties(&prop, 0), "cudaGetDeviceProperties");
        name_ = prop.name;
        arch_ = prop.major * 10 + prop.minor;
        check(cublasCreate(&handle_), "cublasCreate");
    }

    cublasHandle_t handle_ = nullptr;
    std::string name_;
    int arch_ = 0;
};

class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t bytes) { check(cudaMalloc(&ptr_, bytes), "cudaMalloc"); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    ~DeviceBuffer() { cudaFree(ptr_); }

    void* get() const { return ptr_; }

private:
    void* ptr_ = nullptr;
};

class Event {
public:
    Event() { check(cudaEventCreate(&event_), "cudaEventCreate"); }
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
    ~Event() { cudaEventDestroy(event_); }

    cudaEvent_t get() const { return event_; }

private:
    cudaEvent_t event_ = nullptr;
};

const float kOneF32 = 1.0f, kZeroF32 = 0.0f;
const Half kOneF16{0x3C00}, kZeroF16{0};

struct Mode {
    DType dtype;
    DType acc;
    int tc;
    cudaDataType_t in;
    cudaDataType_t out;
    cublasComputeType_t compute;
    const void* one;
    const void* zero;
    int minArch;
};

const Mode kModes[] = {
    {DType::F32, DType::F32, 0, CUDA_R_32F, CUDA_R_32F, CUBLAS_COMPUTE_32F_PEDANTIC, &kOneF32, &kZeroF32, 0},
    {DType::F16, DType::F32, 0, CUDA_R_16F, CUDA_R_32F, CUBLAS_COMPUTE_32F_PEDANTIC, &kOneF32, &kZeroF32, 0},
    {DType::F16, DType::F32, 1, CUDA_R_16F, CUDA_R_32F, CUBLAS_COMPUTE_32F, &kOneF32, &kZeroF32, 0},
    {DType::F16, DType::F16, 0, CUDA_R_16F, CUDA_R_16F, CUBLAS_COMPUTE_16F_PEDANTIC, &kOneF16, &kZeroF16, 53},
    {DType::F16, DType::F16, 1, CUDA_R_16F, CUDA_R_16F, CUBLAS_COMPUTE_16F, &kOneF16, &kZeroF16, 53},
};

const Mode* modeFor(const GemmProblem& p, int tc) {
    for (const Mode& m : kModes)
        if (m.dtype == p.dtype && m.acc == p.acc && m.tc == tc) return &m;
    return nullptr;
}

class CublasInstance final : public GemmInstance {
public:
    CublasInstance(const Cuda& cuda, const GemmProblem& p, const GemmData& in, const Mode& mode)
        : handle_(cuda.handle()), p_(p), mode_(mode), a_(in.a.size()), b_(in.b.size()), c_(p.bytesC()) {
        check(cudaMemcpy(a_.get(), in.a.data(), in.a.size(), cudaMemcpyHostToDevice), "cudaMemcpy A");
        check(cudaMemcpy(b_.get(), in.b.data(), in.b.size(), cudaMemcpyHostToDevice), "cudaMemcpy B");
    }

    void run() override {
        check(cublasGemmStridedBatchedEx(handle_, CUBLAS_OP_N, CUBLAS_OP_N, static_cast<int>(p_.N),
                                         static_cast<int>(p_.M), static_cast<int>(p_.K), mode_.one, b_.get(),
                                         mode_.in, static_cast<int>(p_.N), p_.broadcastB ? 0 : p_.elemsB(),
                                         a_.get(), mode_.in, static_cast<int>(p_.K), p_.elemsA(), mode_.zero,
                                         c_.get(), mode_.out, static_cast<int>(p_.N), p_.elemsC(),
                                         static_cast<int>(p_.batch), mode_.compute, CUBLAS_GEMM_DEFAULT),
              "cublasGemmStridedBatchedEx");
    }

    void sync() override { check(cudaDeviceSynchronize(), "cudaDeviceSynchronize"); }

    void beginGpuTiming() override { check(cudaEventRecord(start_.get()), "cudaEventRecord"); }

    double endGpuTimingMs() override {
        check(cudaEventRecord(stop_.get()), "cudaEventRecord");
        check(cudaEventSynchronize(stop_.get()), "cudaEventSynchronize");
        float ms = 0.0f;
        check(cudaEventElapsedTime(&ms, start_.get(), stop_.get()), "cudaEventElapsedTime");
        return static_cast<double>(ms);
    }

    void readC(std::span<std::byte> out) override {
        sync();
        check(cudaMemcpy(out.data(), c_.get(), std::min(out.size(), p_.bytesC()), cudaMemcpyDeviceToHost),
              "cudaMemcpy C");
    }

private:
    cublasHandle_t handle_;
    GemmProblem p_;
    const Mode& mode_;
    DeviceBuffer a_, b_, c_;
    Event start_, stop_;
};

class CublasKernel final : public GemmKernel {
public:
    explicit CublasKernel(int tc) : tc_(tc) {}

    std::string name() const override { return tc_ ? "cublas_tc" : "cublas"; }
    std::string backend() const override { return "cuda"; }
    std::string device() const override {
        const Cuda* cuda = Cuda::get();
        return cuda ? cuda->deviceName() : "no CUDA device";
    }

    bool supports(const GemmProblem& p) const override {
        const Cuda* cuda = Cuda::get();
        const Mode* mode = modeFor(p, tc_);
        if (!cuda || !mode || cuda->arch() < mode->minArch) return false;
        return p.M <= INT_MAX && p.N <= INT_MAX && p.K <= INT_MAX && p.batch <= INT_MAX;
    }

    std::unique_ptr<GemmInstance> prepare(const GemmProblem& p, const GemmData& in) const override {
        return std::make_unique<CublasInstance>(*Cuda::get(), p, in, *modeFor(p, tc_));
    }

private:
    int tc_;
};

}

std::unique_ptr<GemmKernel> makeCublasKernel(int tc) { return std::make_unique<CublasKernel>(tc); }

}
