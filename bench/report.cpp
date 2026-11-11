#include "report.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace gemm::bench {
namespace {

std::string formatDouble(double value) {
    if (std::isnan(value)) return "nan";
    std::ostringstream out;
    out << std::setprecision(9) << value;
    return out.str();
}

std::string csvEscape(const std::string& value) {
    if (value.find_first_of(",\"\n") == std::string::npos) return value;
    std::string escaped = "\"";
    for (const char c : value) {
        if (c == '"') escaped.push_back('"');
        escaped.push_back(c);
    }
    return escaped + '"';
}

}

void Report::add(Result result) {
    std::cout << "RESULT case=" << result.name << " phase=" << result.phase << " shape=" << result.shape
              << " wall_ms=" << formatDouble(result.timing.wallMs) << " gpu_ms=" << formatDouble(result.timing.gpuMs)
              << " cpu_ms=" << formatDouble(result.timing.cpuMs);
    if (!result.kernel.empty()) std::cout << " kernel=" << result.kernel;
    std::cout << std::endl;
    results_.push_back(std::move(result));
}

void Report::writeCsv(const std::string& path) const {
    const std::filesystem::path destination(path);
    if (destination.has_parent_path()) std::filesystem::create_directories(destination.parent_path());

    std::ofstream out(path);
    if (!out) throw std::runtime_error("failed to open benchmark report: " + path);
    out << "case,phase,shape,wall_ms,gpu_ms,cpu_ms,kernel\n";
    for (const auto& r : results_)
        out << csvEscape(r.name) << ',' << csvEscape(r.phase) << ',' << csvEscape(r.shape) << ','
            << formatDouble(r.timing.wallMs) << ',' << formatDouble(r.timing.gpuMs) << ','
            << formatDouble(r.timing.cpuMs) << ',' << csvEscape(r.kernel) << '\n';
}

}
