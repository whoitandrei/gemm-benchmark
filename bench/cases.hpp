#pragma once

#include "problem.hpp"

#include <string>
#include <vector>

namespace gemm::bench {

struct Case {
    std::string name;
    GemmProblem problem;
};

std::string category(const Case& c);

std::vector<Case> loadCases(const std::string& path);

}
