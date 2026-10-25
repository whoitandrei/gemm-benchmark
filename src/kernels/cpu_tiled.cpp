#include "kernel.hpp"

#include <algorithm>
#include <cstring>

namespace gemm {
namespace {

template <class TIn, class TOut>
void gemmTiled(const GemmProblem& p, const TIn* A, const TIn* B, TOut* C, int64_t BM, int64_t BN,
               int64_t BK) {
    const Strides sa = p.stridesA();
    const Strides sb = p.stridesB();
    for (int64_t b = 0; b < p.batch; ++b) {
        const TIn* a = A + b * p.elemsA();
        const TIn* bm = B + (p.broadcastB ? 0 : b) * p.elemsB();
        TOut* c = C + b * p.elemsC();
        std::fill(c, c + p.elemsC(), TOut{0});

        for (int64_t i0 = 0; i0 < p.M; i0 += BM) {
            const int64_t i1 = std::min(i0 + BM, p.M);
            for (int64_t j0 = 0; j0 < p.N; j0 += BN) {
                const int64_t j1 = std::min(j0 + BN, p.N);
                for (int64_t k0 = 0; k0 < p.K; k0 += BK) {
                    const int64_t k1 = std::min(k0 + BK, p.K);
                    for (int64_t i = i0; i < i1; ++i) {
                        TOut* crow = c + i * p.N;
                        for (int64_t k = k0; k < k1; ++k) {
                            const TOut av = static_cast<TOut>(a[i * sa.row + k * sa.col]);
                            const TIn* brow = bm + k * sb.row;
                            for (int64_t j = j0; j < j1; ++j)
                                crow[j] += av * static_cast<TOut>(brow[j * sb.col]);
                        }
                    }
                }
            }
        }
    }
}

class CpuTiledInstance final : public GemmInstance {
public:
    CpuTiledInstance(const GemmProblem& p, const KernelParams& params, const GemmData& in)
        : p_(p),
          in_(in),
          bm_(params.at("BM")),
          bn_(params.at("BN")),
          bk_(params.at("BK")),
          c_(static_cast<std::size_t>(p.batch * p.elemsC()) * outputBytes(p.dtype)) {}

    void run() override {
        switch (p_.dtype) {
        case DType::F32:
            gemmTiled(p_, reinterpret_cast<const float*>(in_.a.data()),
                      reinterpret_cast<const float*>(in_.b.data()),
                      reinterpret_cast<float*>(c_.data()), bm_, bn_, bk_);
            break;
        case DType::I8:
            gemmTiled(p_, reinterpret_cast<const int8_t*>(in_.a.data()),
                      reinterpret_cast<const int8_t*>(in_.b.data()),
                      reinterpret_cast<int32_t*>(c_.data()), bm_, bn_, bk_);
            break;
        }
    }

    void readC(std::span<std::byte> out) override {
        std::memcpy(out.data(), c_.data(), std::min(out.size(), c_.size()));
    }

private:
    GemmProblem p_;
    const GemmData& in_;
    int64_t bm_, bn_, bk_;
    std::vector<std::byte> c_;
};

class CpuTiledKernel final : public GemmKernel {
public:
    std::string name() const override { return "cpu_tiled"; }
    std::string backend() const override { return "cpu"; }

    ParamSpace paramSpace() const override {
        return {{"BM", {16, 64}}, {"BN", {64, 256}}, {"BK", {64}}};
    }

    bool supports(const GemmProblem& p, const KernelParams& params) const override {
        const bool dtypeOk = p.dtype == DType::F32 || p.dtype == DType::I8;
        const bool tilesOk = params.at("BM") > 0 && params.at("BN") > 0 && params.at("BK") > 0;
        return dtypeOk && tilesOk;
    }

    std::unique_ptr<GemmInstance> prepare(const GemmProblem& p, const KernelParams& params,
                                          const GemmData& in) const override {
        return std::make_unique<CpuTiledInstance>(p, params, in);
    }
};

}

std::unique_ptr<GemmKernel> makeCpuTiledKernel() { return std::make_unique<CpuTiledKernel>(); }

}
