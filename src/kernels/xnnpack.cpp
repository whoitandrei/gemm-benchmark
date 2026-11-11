#include "device.hpp"
#include "kernel.hpp"

#include <pthreadpool.h>
#include <xnnpack.h>

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace gemm {
namespace {

bool xnnpackReady() {
    static const bool ready = xnn_initialize(nullptr) == xnn_status_success;
    return ready;
}

void check(xnn_status status, const char* what) {
    if (status != xnn_status_success)
        throw std::runtime_error(std::string(what) + " failed: xnn_status " + std::to_string(status));
}

class XnnpackInstance final : public GemmInstance {
public:
    XnnpackInstance(const GemmProblem& p, const GemmData& in) : c_(static_cast<std::size_t>(p.batch * p.elemsC())) {
        try {
            init(p, cpuThreads(), in);
        } catch (...) {
            release();
            throw;
        }
    }

    ~XnnpackInstance() override { release(); }

    void run() override {
        for (xnn_operator_t op : ops_) check(xnn_run_operator(op, pool_), "xnn_run_operator");
    }

    void readC(std::span<std::byte> out) override {
        std::memcpy(out.data(), c_.data(), std::min(out.size(), c_.size() * sizeof(float)));
    }

private:
    void init(const GemmProblem& p, int threads, const GemmData& in) {
        if (threads > 1) {
            pool_ = pthreadpool_create(static_cast<size_t>(threads));
            if (!pool_) throw std::runtime_error("pthreadpool_create failed");
        }
        const bool oneB = p.broadcastB || p.batch == 1;
        const int64_t opCount = oneB ? 1 : p.batch;
        const int64_t rows = oneB ? p.batch * p.M : p.M;
        const auto* A = reinterpret_cast<const float*>(in.a.data());
        const auto* B = reinterpret_cast<const float*>(in.b.data());
        const float inf = std::numeric_limits<float>::infinity();

        for (int64_t i = 0; i < opCount; ++i) {
            xnn_operator_t op = nullptr;
            check(xnn_create_fully_connected_nc_f32(static_cast<size_t>(p.K), static_cast<size_t>(p.N),
                                                    static_cast<size_t>(p.K), static_cast<size_t>(p.N),
                                                    B + i * p.elemsB(), nullptr, -inf, inf, XNN_FLAG_TRANSPOSE_WEIGHTS, nullptr, &op),
                  "xnn_create_fully_connected_nc_f32");
            ops_.push_back(op);
            check(xnn_reshape_fully_connected_nc_f32(op, static_cast<size_t>(rows), pool_),
                  "xnn_reshape_fully_connected_nc_f32");
            check(xnn_setup_fully_connected_nc_f32(op, A + i * p.elemsA(), c_.data() + i * p.elemsC()),
                  "xnn_setup_fully_connected_nc_f32");
        }
    }

    void release() {
        for (xnn_operator_t op : ops_) xnn_delete_operator(op);
        ops_.clear();
        if (pool_) pthreadpool_destroy(pool_);
        pool_ = nullptr;
    }

    std::vector<float> c_;
    std::vector<xnn_operator_t> ops_;
    pthreadpool_t pool_ = nullptr;
};

class XnnpackKernel final : public GemmKernel {
public:
    std::string name() const override { return "xnnpack"; }
    std::string backend() const override { return "cpu"; }
    std::string device() const override { return cpuName(); }

    bool supports(const GemmProblem& p) const override {
        return p.dtype == DType::F32 && xnnpackReady();
    }

    std::unique_ptr<GemmInstance> prepare(const GemmProblem& p, const GemmData& in) const override {
        return std::make_unique<XnnpackInstance>(p, in);
    }
};

}

std::unique_ptr<GemmKernel> makeXnnpackKernel() { return std::make_unique<XnnpackKernel>(); }

}
