#include "config.hpp"

#include "kernel.hpp"

#include <algorithm>
#include <sstream>
#include <stdexcept>

namespace gemm::bench {
namespace {

std::vector<std::string> splitList(const std::string& s, char sep = ',') {
    std::vector<std::string> items;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, sep))
        if (!item.empty()) items.push_back(item);
    return items;
}

int64_t parsePositive(const std::string& s, const char* what) {
    std::size_t used = 0;
    long long v = 0;
    try {
        v = std::stoll(s, &used);
    } catch (const std::exception&) {
        used = 0;
    }
    if (used != s.size() || v <= 0)
        throw std::invalid_argument(std::string("bad ") + what + " '" + s + "'");
    return v;
}

struct Shape {
    int64_t M, N, K;
};

Shape parseShape(const std::string& s) {
    const auto dims = splitList(s, 'x');
    if (dims.size() == 1) {
        const int64_t n = parsePositive(dims[0], "shape");
        return {n, n, n};
    }
    if (dims.size() == 3)
        return {parsePositive(dims[0], "M"), parsePositive(dims[1], "N"), parsePositive(dims[2], "K")};
    throw std::invalid_argument("bad shape '" + s + "' (expected N or MxNxK)");
}

}

bool BenchConfig::kernelSelected(const std::string& name) const {
    return kernels.empty() || std::find(kernels.begin(), kernels.end(), name) != kernels.end();
}

BenchConfig parseConfig(const BenchOptions& o) {
    BenchConfig cfg;
    cfg.csvPath = o.csv;
    cfg.verifySamples = o.verifySamples;
    if (cfg.verifySamples < 0) throw std::invalid_argument("--verify-samples must be >= 0");
    cfg.warmupRuns = o.warmupRuns;
    cfg.warmupMs = o.warmupMs;
    if (cfg.warmupRuns < 0 || cfg.warmupMs < 0)
        throw std::invalid_argument("--warmup-runs and --warmup-ms must be >= 0");

    const auto kernels = allKernels();
    cfg.kernels = splitList(o.kernels);
    for (const auto& name : cfg.kernels) {
        const bool known = std::any_of(kernels.begin(), kernels.end(),
                                       [&](const auto& k) { return k->name() == name; });
        if (!known) throw std::invalid_argument("unknown kernel '" + name + "' (see --list-kernels)");
    }

    cfg.paramOverrides = parseParamSpace(o.params);
    for (const auto& [param, values] : cfg.paramOverrides) {
        const bool used = std::any_of(kernels.begin(), kernels.end(), [&](const auto& k) {
            return cfg.kernelSelected(k->name()) && k->paramSpace().count(param) > 0;
        });
        if (!used) throw std::invalid_argument("parameter '" + param + "' is not used by selected kernels");
    }

    for (const auto& shapeText : splitList(o.shapes)) {
        const Shape shape = parseShape(shapeText);
        for (const auto& batchText : splitList(o.batches)) {
            const int64_t batch = parsePositive(batchText, "batch");
            for (const auto& dtypeText : splitList(o.dtypes)) {
                for (const auto& t : splitList(o.trans)) {
                    if (t.size() != 2 || (t[0] != 'n' && t[0] != 't') || (t[1] != 'n' && t[1] != 't'))
                        throw std::invalid_argument("bad trans '" + t + "' (expected nn, nt, tn or tt)");
                    GemmProblem p;
                    p.M = shape.M;
                    p.N = shape.N;
                    p.K = shape.K;
                    p.batch = batch;
                    p.dtype = parseDType(dtypeText);
                    p.transA = t[0] == 't';
                    p.transB = t[1] == 't';
                    p.broadcastB = o.broadcastB;
                    cfg.problems.push_back(p);
                }
            }
        }
    }
    if (cfg.problems.empty()) throw std::invalid_argument("no problems to run (check --shapes, --batch, --dtype, --trans)");
    return cfg;
}

BenchConfig& benchConfig() {
    static BenchConfig cfg;
    return cfg;
}

}
