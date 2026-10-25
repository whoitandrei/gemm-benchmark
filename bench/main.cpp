#include "config.hpp"
#include "kernel.hpp"

#include <catch2/catch_session.hpp>

#include <iostream>

namespace {

void listKernels() {
    for (const auto& k : gemm::allKernels()) {
        std::cout << k->name() << " (" << k->backend() << ")";
        for (const auto& [param, values] : k->paramSpace()) {
            std::cout << "  " << param << '=';
            for (std::size_t i = 0; i < values.size(); ++i) std::cout << (i ? "," : "") << values[i];
        }
        std::cout << '\n';
    }
}

}

int main(int argc, char* argv[]) {
    Catch::Session session;

    session.configData().benchmarkWarmupTime = 0;

    gemm::bench::BenchOptions o;
    bool list = false;
    using Catch::Clara::Opt;
    session.cli(session.cli()
        | Opt(o.kernels, "names")["--kernel"]("GEMM: kernels, comma-separated (default: all)")
        | Opt(o.shapes, "list")["--shapes"]("GEMM: sizes, N or MxNxK, comma-separated (default: 128,256,512)")
        | Opt(o.batches, "list")["--batch"]("GEMM: batch sizes, comma-separated (default: 1)")
        | Opt(o.dtypes, "list")["--dtype"]("GEMM: data types f32,i8 (default: f32,i8)")
        | Opt(o.trans, "list")["--trans"]("GEMM: layouts nn,nt,tn,tt (default: nn)")
        | Opt(o.broadcastB)["--broadcast-b"]("GEMM: one B matrix for the whole batch")
        | Opt(o.params, "spec")["--params"]("GEMM: override kernel parameters, e.g. \"BM=32,64;BK=128\"")
        | Opt(o.csv, "file")["--csv"]("GEMM: write results to a CSV file")
        | Opt(o.verifySamples, "n")["--verify-samples"]("GEMM: elements of C checked against the reference, 0 = all (default: 1024)")
        | Opt(o.warmupRuns, "n")["--warmup-runs"]("GEMM: minimum kernel runs before measuring (default: 3)")
        | Opt(o.warmupMs, "ms")["--warmup-ms"]("GEMM: minimum warm-up time with kernel runs, ms (default: 200)")
        | Opt(list)["--list-kernels"]("GEMM: list kernels and their parameters"));

    if (const int rc = session.applyCommandLine(argc, argv); rc != 0) return rc;
    if (list) {
        listKernels();
        return 0;
    }

    try {
        gemm::bench::benchConfig() = gemm::bench::parseConfig(o);
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 2;
    }

#ifndef NDEBUG
    std::cerr << "warning: built without NDEBUG (not Release), timings are not representative\n";
#endif

    return session.run();
}
