#include "cases.hpp"
#include "kernel.hpp"
#include "measure.hpp"
#include "reference.hpp"
#include "report.hpp"

#include <cstdlib>
#include <exception>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using namespace gemm;
using namespace gemm::bench;

namespace {

constexpr int64_t kVerifySamples = 1024;

struct Options {
    MeasureOptions timing;
    std::string device;
    std::set<std::string> kernels;
    std::string filter;
    std::string caseName;
    std::string report{"results/gemm_bench_latest.csv"};
    bool reportExplicit = false;
    bool list = false;

    bool filtered() const { return !device.empty() || !kernels.empty() || !filter.empty() || !caseName.empty(); }
};

[[noreturn]] void fail(const std::string& message) {
    std::cerr << message << std::endl;
    std::exit(1);
}

int toInt(const std::string& name, const std::string& value) {
    try {
        std::size_t used = 0;
        const int v = std::stoi(value, &used);
        if (used == value.size()) return v;
    } catch (const std::exception&) {
    }
    fail("invalid value for " + name + ": " + value);
}

Options parseOptions(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) fail("missing value for " + arg);
            return argv[++i];
        };
        if (arg == "--device") {
            options.device = next();
            if (options.device != "cpu" && options.device != "gpu") fail("--device must be cpu or gpu");
        } else if (arg == "--kernel") {
            std::stringstream names(next());
            for (std::string name; std::getline(names, name, ',');)
                if (!name.empty()) options.kernels.insert(name);
        } else if (arg == "--iters") {
            options.timing.iterations = toInt(arg, next());
        } else if (arg == "--warmup") {
            options.timing.warmup = toInt(arg, next());
        } else if (arg == "--rounds") {
            options.timing.rounds = toInt(arg, next());
        } else if (arg == "--case") {
            options.filter = next();
        } else if (arg == "--case-name") {
            options.caseName = next();
        } else if (arg == "--report") {
            options.report = next();
            options.reportExplicit = true;
        } else if (arg == "--list") {
            options.list = true;
        } else if (arg == "-h" || arg == "--help") {
            std::cout << "usage: gemm_bench [--device cpu|gpu] [--kernel a,b] [--iters 20] [--warmup 5] [--rounds 5]\n"
                         "                  [--case substring] [--case-name substring] [--report path] [--list]\n";
            std::exit(0);
        } else {
            fail("unknown argument " + arg);
        }
    }
    if (options.timing.warmup < 0 || options.timing.iterations <= 0 || options.timing.rounds <= 0)
        fail("--warmup must be >= 0, --iters and --rounds must be > 0");
    return options;
}

std::vector<std::unique_ptr<GemmKernel>> selectKernels(const Options& options) {
    std::vector<std::unique_ptr<GemmKernel>> selected;
    std::set<std::string> unknown = options.kernels;
    for (auto& kernel : allKernels()) {
        unknown.erase(kernel->name());
        if (!options.kernels.empty() && options.kernels.count(kernel->name()) == 0) continue;
        const bool cpu = kernel->backend() == "cpu";
        if (options.device == "cpu" && !cpu) continue;
        if (options.device == "gpu" && cpu) continue;
        selected.push_back(std::move(kernel));
    }
    for (const auto& name : unknown) fail("unknown kernel " + name + " (see --list)");
    return selected;
}

bool caseSelected(const Options& options, const Case& c) {
    return category(c).find(options.filter) != std::string::npos &&
           c.name.find(options.caseName) != std::string::npos;
}

void list() {
    std::cout << "kernels:\n";
    for (const auto& k : allKernels())
        std::cout << "  " << k->name() << " (" << k->backend() << ", " << k->device() << ")\n";
    std::cout << "cases:\n";
    for (const auto& c : allCases()) std::cout << "  " << c.name << "  " << c.problem.toString() << '\n';
}

}

int main(int argc, char** argv) {
    const Options options = parseOptions(argc, argv);
    if (options.list) {
        list();
        return 0;
    }

#ifndef NDEBUG
    std::cerr << "warning: built without NDEBUG (not Release), timings are not representative\n";
#endif

    const auto kernels = selectKernels(options);
    for (const auto& k : kernels) std::cout << "device: " << k->name() << " " << k->backend() << " " << k->device() << '\n';

    Report report;
    int failures = 0;
    for (const Case& c : allCases()) {
        if (!caseSelected(options, c)) continue;
        std::vector<const GemmKernel*> supported;
        for (const auto& k : kernels)
            if (k->supports(c.problem)) supported.push_back(k.get());
        if (supported.empty()) continue;

        const GemmData data = makeData(c.problem);
        std::vector<std::byte> out(c.problem.bytesC());
        for (const GemmKernel* k : supported) {
            try {
                auto instance = k->prepare(c.problem, data);
                instance->run();
                instance->sync();
                instance->readC(out);
                const VerifyResult check = verify(c.problem, data, out, kVerifySamples);
                if (!check.ok()) {
                    std::cerr << "FAILED case=" << c.name << " kernel=" << k->name() << " max_error=" << check.maxError
                              << " tolerance=" << check.tolerance << '\n';
                    ++failures;
                    continue;
                }
                const Timing timing = measure(*instance, options.timing);
                report.add({c.name, "matmul", c.problem.shape(), timing, k->name()});
            } catch (const std::exception& e) {
                std::cerr << "ERROR case=" << c.name << " kernel=" << k->name() << ": " << e.what() << '\n';
                ++failures;
            }
        }
    }

    if (report.results().empty()) {
        std::cerr << "no results: no selected kernel supports the selected cases\n";
        ++failures;
    }
    if (options.reportExplicit || !options.filtered()) {
        try {
            report.writeCsv(options.report);
        } catch (const std::exception& e) {
            fail(e.what());
        }
        std::cout << "report: " << options.report << std::endl;
    } else {
        std::cout << "report skipped for filtered run; pass --report to write it" << std::endl;
    }
    return failures == 0 ? 0 : 1;
}
