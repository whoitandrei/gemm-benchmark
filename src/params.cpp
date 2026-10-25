#include "params.hpp"

#include <sstream>
#include <stdexcept>

namespace gemm {

std::vector<KernelParams> expand(const ParamSpace& space) {
    std::vector<KernelParams> result{KernelParams{}};
    for (const auto& [name, values] : space) {
        std::vector<KernelParams> next;
        for (const auto& partial : result) {
            for (int v : values) {
                KernelParams p = partial;
                p[name] = v;
                next.push_back(std::move(p));
            }
        }
        result = std::move(next);
    }
    return result;
}

ParamSpace parseParamSpace(const std::string& text) {
    ParamSpace space;
    std::stringstream entries(text);
    std::string entry;
    while (std::getline(entries, entry, ';')) {
        if (entry.empty()) continue;
        const auto eq = entry.find('=');
        if (eq == std::string::npos || eq == 0 || eq + 1 == entry.size())
            throw std::invalid_argument("bad parameter '" + entry + "' (expected NAME=v1,v2)");
        std::vector<int> values;
        std::stringstream list(entry.substr(eq + 1));
        std::string item;
        while (std::getline(list, item, ',')) {
            std::size_t used = 0;
            const int v = std::stoi(item, &used);
            if (used != item.size()) throw std::invalid_argument("bad value '" + item + "'");
            values.push_back(v);
        }
        space[entry.substr(0, eq)] = std::move(values);
    }
    return space;
}

ParamSpace applyOverrides(ParamSpace space, const ParamSpace& overrides) {
    for (const auto& [name, values] : overrides) {
        if (auto it = space.find(name); it != space.end()) it->second = values;
    }
    return space;
}

std::string toString(const KernelParams& params) {
    std::string s;
    for (const auto& [name, value] : params) {
        if (!s.empty()) s += ' ';
        s += name + "=" + std::to_string(value);
    }
    return s;
}

}
