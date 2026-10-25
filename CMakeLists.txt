cmake_minimum_required(VERSION 3.20)
project(gemm_benchmark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(Catch2 3 QUIET)
if(NOT Catch2_FOUND)
  include(FetchContent)
  FetchContent_Declare(Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v3.16.1
    GIT_SHALLOW    TRUE)
  FetchContent_MakeAvailable(Catch2)
endif()

set(GEMM_WARNINGS
  $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic>
  $<$<CXX_COMPILER_ID:MSVC>:/W4>)

add_library(gemm_core STATIC
  src/problem.cpp
  src/params.cpp
  src/reference.cpp
  src/kernels/registry.cpp
  src/kernels/cpu_tiled.cpp)
target_include_directories(gemm_core PUBLIC src)
target_compile_options(gemm_core PRIVATE ${GEMM_WARNINGS})

add_executable(gemm_tests tests/test_gemm.cpp)
target_link_libraries(gemm_tests PRIVATE gemm_core Catch2::Catch2WithMain)
target_compile_options(gemm_tests PRIVATE ${GEMM_WARNINGS})

add_executable(gemm_bench
  bench/main.cpp
  bench/config.cpp
  bench/results.cpp
  bench/bench_gemm.cpp)
target_link_libraries(gemm_bench PRIVATE gemm_core Catch2::Catch2)
target_compile_options(gemm_bench PRIVATE ${GEMM_WARNINGS})

enable_testing()
add_test(NAME gemm_tests COMMAND gemm_tests)
