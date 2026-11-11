#include "cases.hpp"

#include <fstream>
#include <set>
#include <sstream>
#include <stdexcept>

namespace gemm::bench {
namespace {

constexpr const char* kHeader = "case,batch,m,k,n,dtype,acc,shared_b";

std::vector<std::string> split(const std::string& line) {
    std::vector<std::string> fields;
    std::stringstream in(line);
    for (std::string field; std::getline(in, field, ',');) fields.push_back(field);
    if (!line.empty() && line.back() == ',') fields.emplace_back();
    return fields;
}

int64_t positive(const std::string& text) {
    std::size_t used = 0;
    long long v = 0;
    try {
        v = std::stoll(text, &used);
    } catch (const std::logic_error&) {
        used = 0;
    }
    if (used == 0 || used != text.size() || v <= 0)
        throw std::invalid_argument("expected a positive integer, got '" + text + "'");
    return v;
}

DType dtype(const std::string& text) {
    if (text == "f32") return DType::F32;
    if (text == "f16") return DType::F16;
    throw std::invalid_argument("expected f32 or f16, got '" + text + "'");
}

Case parse(const std::string& line) {
    const std::vector<std::string> f = split(line);
    if (f.size() != 8) throw std::invalid_argument("expected 8 fields, got " + std::to_string(f.size()));
    if (f[0].empty()) throw std::invalid_argument("empty case name");
    if (f[7] != "0" && f[7] != "1") throw std::invalid_argument("shared_b must be 0 or 1, got '" + f[7] + "'");
    Case c{f[0],
           {.batch = positive(f[1]),
            .M = positive(f[2]),
            .K = positive(f[3]),
            .N = positive(f[4]),
            .dtype = dtype(f[5]),
            .acc = dtype(f[6]),
            .broadcastB = f[7] == "1"}};
    if (c.problem.dtype == DType::F32 && c.problem.acc == DType::F16)
        throw std::invalid_argument("f32 inputs need f32 accumulation");
    return c;
}

}

std::string category(const Case& c) { return c.name.substr(0, c.name.find('_')); }

std::vector<Case> loadCases(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open cases file " + path);
    std::string line;
    if (std::getline(in, line) && !line.empty() && line.back() == '\r') line.pop_back();
    if (line != kHeader)
        throw std::runtime_error(path + ":1: expected header " + kHeader);

    std::vector<Case> cases;
    std::set<std::string> names;
    for (int number = 2; std::getline(in, line); ++number) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        try {
            Case c = parse(line);
            if (!names.insert(c.name).second) throw std::invalid_argument("duplicate case " + c.name);
            cases.push_back(std::move(c));
        } catch (const std::exception& e) {
            throw std::runtime_error(path + ":" + std::to_string(number) + ": " + e.what());
        }
    }
    return cases;
}

}
