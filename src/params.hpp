#pragma once

#include <map>
#include <string>
#include <vector>

namespace gemm {

using KernelParams = std::map<std::string, int>;

using ParamSpace = std::map<std::string, std::vector<int>>;

std::vector<KernelParams> expand(const ParamSpace& space);

ParamSpace parseParamSpace(const std::string& text);

ParamSpace applyOverrides(ParamSpace space, const ParamSpace& overrides);

std::string toString(const KernelParams& params);

}
