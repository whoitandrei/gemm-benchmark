#include "kernel.hpp"
#include "params.hpp"
#include "reference.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cstring>

using namespace gemm;

TEST_CASE("reference checks a hand-computed 2x2 product", "[reference]") {
    GemmProblem p;
    p.M = p.N = p.K = 2;
    GemmData d;
    const float a[] = {1, 2, 3, 4}, b[] = {5, 6, 7, 8};
    d.a.resize(sizeof a);
    d.b.resize(sizeof b);
    std::memcpy(d.a.data(), a, sizeof a);
    std::memcpy(d.b.data(), b, sizeof b);

    float c[] = {19, 22, 43, 50};
    std::span<const std::byte> view(reinterpret_cast<const std::byte*>(c), sizeof c);
    CHECK(verify(p, d, view).ok());
    CHECK(verify(p, d, view).checked == 4);

    c[3] = 51;
    CHECK_FALSE(verify(p, d, view).ok());

    p.transA = p.transB = true;
    const float ct[] = {23, 31, 34, 46};
    CHECK(verify(p, d, {reinterpret_cast<const std::byte*>(ct), sizeof ct}).ok());
}

TEST_CASE("parameter space expands to the cartesian product", "[params]") {
    const ParamSpace space = parseParamSpace("BM=16,32;BN=8");
    const auto sets = expand(space);
    REQUIRE(sets.size() == 2);
    CHECK(sets[0] == KernelParams{{"BM", 16}, {"BN", 8}});
    CHECK(sets[1] == KernelParams{{"BM", 32}, {"BN", 8}});

    const auto overridden = applyOverrides(space, parseParamSpace("BN=4,8;XX=1"));
    CHECK(overridden.at("BN") == std::vector<int>{4, 8});
    CHECK(overridden.count("XX") == 0);

    CHECK_THROWS(parseParamSpace("BM"));
    CHECK_THROWS(parseParamSpace("BM=1x"));
}

namespace {

std::vector<GemmProblem> testProblems() {
    std::vector<GemmProblem> problems;
    const struct { int64_t M, N, K, batch; } shapes[] = {
        {1, 1, 1, 1},
        {7, 13, 5, 1},
        {65, 129, 67, 1},
        {64, 64, 64, 3},
    };
    for (auto s : shapes)
        for (DType dtype : {DType::F32, DType::I8})
            for (int t = 0; t < 4; ++t)
                for (bool bcast : {false, true}) {
                    if (bcast && s.batch == 1) continue;
                    GemmProblem p;
                    p.M = s.M;
                    p.N = s.N;
                    p.K = s.K;
                    p.batch = s.batch;
                    p.dtype = dtype;
                    p.transA = (t & 1) != 0;
                    p.transB = (t & 2) != 0;
                    p.broadcastB = bcast;
                    problems.push_back(p);
                }
    return problems;
}

}

TEST_CASE("kernels match the reference", "[correctness]") {
    for (const auto& kernel : allKernels()) {
        for (const auto& params : expand(kernel->paramSpace())) {
            for (const auto& problem : testProblems()) {
                if (!kernel->supports(problem, params)) continue;
                INFO(kernel->name() << " [" << toString(params) << "] " << problem.toString());

                const GemmData data = makeData(problem);
                auto instance = kernel->prepare(problem, params, data);
                std::vector<std::byte> c(static_cast<std::size_t>(problem.batch * problem.elemsC()) *
                                         outputBytes(problem.dtype));
                instance->run();
                instance->readC(c);

                const VerifyResult r = verify(problem, data, c);
                CHECK(r.checked == problem.batch * problem.elemsC());
                CHECK(r.maxError <= r.tolerance);
            }
        }
    }
}
