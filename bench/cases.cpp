#include "cases.hpp"

namespace gemm::bench {
namespace {

GemmProblem problem(int64_t batch, int64_t m, int64_t k, int64_t n, bool broadcastB = false) {
    return {.batch = batch, .M = m, .K = k, .N = n, .broadcastB = broadcastB};
}

std::vector<Case> shapes() {
    std::vector<Case> cases;
    for (int64_t n : {256, 1024, 2048, 4096}) cases.push_back({"square_" + std::to_string(n), problem(1, n, n, n)});
    for (int64_t m : {1, 8, 32, 128})
        cases.push_back({"inference_m" + std::to_string(m), problem(1, m, 4096, 4096)});
    cases.push_back({"inference_k64", problem(1, 4096, 64, 4096)});
    for (int64_t n : {32, 64, 128})
        for (int64_t batch : {1, 16, 1024})
            cases.push_back({"batch_" + std::to_string(n) + "x" + std::to_string(batch), problem(batch, n, n, n)});
    cases.push_back({"batch_64x1024_shared_b", problem(1024, 64, 64, 64, true)});
    cases.push_back({"matmul_attention", problem(8, 100, 32, 100)});
    return cases;
}

}

std::string category(const Case& c) { return c.name.substr(0, c.name.find('_')); }

std::vector<Case> allCases() {
    const struct {
        DType dtype, acc;
        const char* suffix;
    } types[] = {
        {DType::F32, DType::F32, ""},
        {DType::F16, DType::F32, "_f16_f32"},
        {DType::F16, DType::F16, "_f16_f16"},
    };
    std::vector<Case> cases;
    for (const Case& base : shapes())
        for (const auto& t : types) {
            Case c = base;
            c.name += t.suffix;
            c.problem.dtype = t.dtype;
            c.problem.acc = t.acc;
            cases.push_back(c);
        }
    return cases;
}

}
