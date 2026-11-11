#include "device.hpp"

#include <algorithm>
#include <cstddef>
#include <thread>

#if defined(__APPLE__)
#include <sys/sysctl.h>
#elif defined(__linux__)
#include <fstream>
#if defined(__ANDROID__)
#include <sys/system_properties.h>
#endif
#endif

namespace gemm {
namespace {

#if defined(__APPLE__)

std::string detect() {
    char buf[256] = {};
    std::size_t size = sizeof buf;
    if (sysctlbyname("machdep.cpu.brand_string", buf, &size, nullptr, 0) != 0) return {};
    return buf;
}

#elif defined(__linux__)

std::string trim(const std::string& s) {
    const auto first = s.find_first_not_of(" \t");
    if (first == std::string::npos) return {};
    return s.substr(first, s.find_last_not_of(" \t") - first + 1);
}

std::string cpuinfoField(const std::string& key) {
    std::ifstream in("/proc/cpuinfo");
    std::string line;
    while (std::getline(in, line)) {
        const auto colon = line.find(':');
        if (colon != std::string::npos && trim(line.substr(0, colon)) == key) return trim(line.substr(colon + 1));
    }
    return {};
}

std::string detect() {
#if defined(__ANDROID__)
    char soc[PROP_VALUE_MAX] = {};
    if (__system_property_get("ro.soc.model", soc) > 0) return soc;
#endif
    for (const char* key : {"model name", "Hardware"}) {
        std::string value = cpuinfoField(key);
        if (!value.empty()) return value;
    }
    return {};
}

#else

std::string detect() { return {}; }

#endif

}

const std::string& cpuName() {
    static const std::string name = [] {
        std::string detected = detect();
        return detected.empty() ? std::string("unknown CPU") : detected;
    }();
    return name;
}

int cpuThreads() { return static_cast<int>(std::max(1u, std::thread::hardware_concurrency())); }

}
