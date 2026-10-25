#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace gemm {

enum class DType {
    F32,
    I8,
};

std::string toString(DType t);
DType parseDType(const std::string& s);
std::size_t inputBytes(DType t);
std::size_t outputBytes(DType t);

struct Strides {
    int64_t row;
    int64_t col;
};

struct GemmProblem {
    int64_t M = 0;
    int64_t N = 0;
    int64_t K = 0;
    int64_t batch = 1;
    DType dtype = DType::F32;
    bool transA = false;
    bool transB = false;
    bool broadcastB = false;

    int64_t elemsA() const { return M * K; }
    int64_t elemsB() const { return K * N; }
    int64_t elemsC() const { return M * N; }
    int64_t batchesB() const { return broadcastB ? 1 : batch; }

    Strides stridesA() const { return transA ? Strides{1, M} : Strides{K, 1}; }
    Strides stridesB() const { return transB ? Strides{1, K} : Strides{N, 1}; }

    double ops() const { return 2.0 * static_cast<double>(M * N * K * batch); }
    double bytes() const;
    std::string toString() const;
};

struct GemmData {
    std::vector<std::byte> a;
    std::vector<std::byte> b;
};

GemmData makeData(const GemmProblem& p, uint32_t seed = 42);

}
