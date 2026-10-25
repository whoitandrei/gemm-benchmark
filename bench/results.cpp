#include "results.hpp"

#include "config.hpp"

#include <catch2/benchmark/detail/catch_benchmark_stats.hpp>
#include <catch2/reporters/catch_reporter_event_listener.hpp>
#include <catch2/reporters/catch_reporter_registrars.hpp>

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace gemm::bench {
namespace {

BenchCase currentCase;

struct Row {
    BenchCase c;
    double meanNs, meanLowNs, meanHighNs, stddevNs, medianNs, minNs;
    double outlierVariance;
    unsigned samples;
    int iterations;

    double gops() const { return c.problem.ops() / meanNs; }
    double gbps() const { return c.problem.bytes() / meanNs; }
};

std::string fmt(const char* format, double v) {
    char buf[64];
    std::snprintf(buf, sizeof buf, format, v);
    return buf;
}

std::string fmtTime(double ns) {
    if (ns < 1e3) return fmt("%.1f ns", ns);
    if (ns < 1e6) return fmt("%.2f us", ns / 1e3);
    if (ns < 1e9) return fmt("%.3f ms", ns / 1e6);
    return fmt("%.3f s", ns / 1e9);
}

void printTable(const std::vector<std::vector<std::string>>& rows) {
    std::vector<std::size_t> width;
    for (const auto& r : rows)
        for (std::size_t i = 0; i < r.size(); ++i) {
            if (width.size() <= i) width.push_back(0);
            width[i] = std::max(width[i], r[i].size());
        }
    for (const auto& r : rows) {
        for (std::size_t i = 0; i < r.size(); ++i)
            std::cout << r[i] << std::string(width[i] - r[i].size() + 2, ' ');
        std::cout << '\n';
    }
}

class ResultsListener : public Catch::EventListenerBase {
public:
    using EventListenerBase::EventListenerBase;

    void testRunStarting(const Catch::TestRunInfo&) override {
        const std::string& path = benchConfig().csvPath;
        if (path.empty()) return;
        csv_.open(path);
        if (!csv_) throw std::runtime_error("cannot open CSV file '" + path + "'");
        csv_ << "kernel,backend,params,dtype,M,N,K,batch,transA,transB,broadcastB,"
                "warmup_runs,samples,iterations,mean_ns,mean_low_ns,mean_high_ns,stddev_ns,median_ns,min_ns,"
                "outlier_variance,gops,gbps,max_error,tolerance,checked\n";
    }

    void benchmarkEnded(const Catch::BenchmarkStats<>& s) override {
        std::vector<double> sorted;
        for (const auto& d : s.samples) sorted.push_back(d.count());
        std::sort(sorted.begin(), sorted.end());

        Row r{currentCase,
              s.mean.point.count(),
              s.mean.lower_bound.count(),
              s.mean.upper_bound.count(),
              s.standardDeviation.point.count(),
              sorted.empty() ? 0.0 : sorted[sorted.size() / 2],
              sorted.empty() ? 0.0 : sorted.front(),
              s.outlierVariance,
              s.info.samples,
              s.info.iterations};
        rows_.push_back(r);
        if (csv_.is_open()) writeCsv(r);
    }

    void testRunEnded(const Catch::TestRunStats&) override {
        if (rows_.empty()) return;
        std::vector<std::vector<std::string>> table{
            {"kernel", "params", "problem", "runs", "mean", "stddev", "GOP/s", "GB/s", "max err"}};
        for (const auto& r : rows_) {
            const std::string runs = std::to_string(r.c.warmupRuns) + "+" + std::to_string(r.samples) +
                                     "x" + std::to_string(r.iterations);
            table.push_back({r.c.kernel, toString(r.c.params), r.c.problem.toString(), runs,
                             fmtTime(r.meanNs), fmtTime(r.stddevNs), fmt("%.2f", r.gops()),
                             fmt("%.2f", r.gbps()), fmt("%.1e", r.c.check.maxError)});
        }
        std::cout << "\nGEMM results (runs = warm-up + samples x iterations; "
                     "GOP/s and GB/s from the mean time):\n";
        printTable(table);
        if (csv_.is_open()) std::cout << "CSV: " << benchConfig().csvPath << '\n';
        std::cout << '\n';
    }

private:
    void writeCsv(const Row& r) {
        const GemmProblem& p = r.c.problem;
        csv_ << r.c.kernel << ',' << r.c.backend << ',' << toString(r.c.params) << ','
             << toString(p.dtype) << ',' << p.M << ',' << p.N << ',' << p.K << ',' << p.batch << ','
             << p.transA << ',' << p.transB << ',' << p.broadcastB << ',' << r.c.warmupRuns << ','
             << r.samples << ','
             << r.iterations << ',' << r.meanNs << ',' << r.meanLowNs << ',' << r.meanHighNs << ','
             << r.stddevNs << ',' << r.medianNs << ',' << r.minNs << ',' << r.outlierVariance << ','
             << r.gops() << ',' << r.gbps() << ',' << r.c.check.maxError << ','
             << r.c.check.tolerance << ',' << r.c.check.checked << '\n';
        csv_.flush();
    }

    std::ofstream csv_;
    std::vector<Row> rows_;
};

}

void setCurrentCase(const BenchCase& c) { currentCase = c; }

std::string caseName(const BenchCase& c) {
    return c.kernel + " [" + toString(c.params) + "] " + c.problem.toString();
}

}

CATCH_REGISTER_LISTENER(gemm::bench::ResultsListener)
