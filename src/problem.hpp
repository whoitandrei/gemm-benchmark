#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace gemm {

enum class DType {
    F32,
    F16,
};

std::string toString(DType t);
std::size_t sizeOf(DType t);

struct GemmProblem {
    int64_t batch = 1;
    int64_t M = 0;
    int64_t K = 0;
    int64_t N = 0;
    DType dtype = DType::F32;
    DType acc = DType::F32;
    bool broadcastB = false;

    int64_t elemsA() const { return M * K; }
    int64_t elemsB() const { return K * N; }
    int64_t elemsC() const { return M * N; }
    int64_t batchesB() const { return broadcastB ? 1 : batch; }

    std::size_t bytesC() const { return static_cast<std::size_t>(batch * elemsC()) * sizeOf(acc); }

    std::string shape() const;
    std::string toString() const;
};

struct GemmData {
    std::vector<std::byte> a;
    std::vector<std::byte> b;
};

GemmData makeData(const GemmProblem& p, uint32_t seed = 42);

std::vector<std::byte> transposedB(const GemmProblem& p, const GemmData& in);

}
