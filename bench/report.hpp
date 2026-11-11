#pragma once

#include "measure.hpp"

#include <string>
#include <vector>

namespace gemm::bench {

struct Result {
    std::string name;
    std::string phase;
    std::string shape;
    Timing timing;
    std::string kernel;
};

class Report {
public:
    void add(Result result);

    const std::vector<Result>& results() const { return results_; }

    void writeCsv(const std::string& path) const;

private:
    std::vector<Result> results_;
};

}
