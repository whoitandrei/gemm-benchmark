cmake_minimum_required(VERSION 3.20)
project(gemm_benchmark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

include(FetchContent)
set(BUILD_SHARED_LIBS OFF)

set(GEMM_WARNINGS
  $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic>
  $<$<CXX_COMPILER_ID:MSVC>:/W4>)

add_library(gemm_core STATIC
  src/device.cpp
  src/problem.cpp
  src/measure.cpp
  src/reference.cpp
  src/kernels/registry.cpp)
target_include_directories(gemm_core PUBLIC src)
target_compile_options(gemm_core PRIVATE ${GEMM_WARNINGS})

option(GEMM_WITH_GGML "ggml CPU kernel (downloads ggml)" ON)
option(GEMM_WITH_GGML_VULKAN "ggml Vulkan kernel (downloads ggml, needs Vulkan SDK with glslc, long build)" OFF)
if(GEMM_WITH_GGML OR GEMM_WITH_GGML_VULKAN)
  set(GGML_METAL OFF)
  set(GGML_BLAS OFF)
  set(GGML_VULKAN ${GEMM_WITH_GGML_VULKAN})
  set(GGML_OPENMP OFF)
  set(GGML_LLAMAFILE ON)
  set(GGML_BUILD_TESTS OFF)
  set(GGML_BUILD_EXAMPLES OFF)
  FetchContent_Declare(ggml
    GIT_REPOSITORY https://github.com/ggml-org/ggml.git
    GIT_TAG        v0.26.0
    GIT_SHALLOW    TRUE)
  if(GEMM_WITH_GGML_VULKAN)
    find_package(Vulkan REQUIRED)
    get_filename_component(GEMM_VULKAN_LIB_DIR "${Vulkan_LIBRARY}" DIRECTORY)
    find_package(SPIRV-Headers CONFIG REQUIRED PATHS "${GEMM_VULKAN_LIB_DIR}/cmake/vulkan")
  endif()
  FetchContent_MakeAvailable(ggml)
  target_link_libraries(gemm_core PUBLIC ggml)
endif()
if(GEMM_WITH_GGML)
  target_sources(gemm_core PRIVATE src/kernels/ggml_cpu.cpp)
  target_compile_definitions(gemm_core PRIVATE GEMM_WITH_GGML)
endif()
if(GEMM_WITH_GGML_VULKAN)
  target_sources(gemm_core PRIVATE src/kernels/ggml_vulkan.cpp)
  target_compile_definitions(gemm_core PRIVATE GEMM_WITH_GGML_VULKAN)
endif()

option(GEMM_WITH_XNNPACK "XNNPACK kernel (downloads XNNPACK, long build)" OFF)
if(GEMM_WITH_XNNPACK)
  set(XNNPACK_BUILD_TESTS OFF)
  set(XNNPACK_BUILD_BENCHMARKS OFF)
  set(XNNPACK_BUILD_ALL_MICROKERNELS OFF)
  FetchContent_Declare(xnnpack
    URL      https://github.com/google/XNNPACK/archive/6269103e5416bb69f567e32b23eaaebccaa648f0.tar.gz
    URL_HASH SHA256=6162fd0f23ee210ed917d7e93f3415a2fdedabde4a19ddafa01f2a4b85857f23
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
  FetchContent_MakeAvailable(xnnpack)
  target_sources(gemm_core PRIVATE src/kernels/xnnpack.cpp)
  target_compile_definitions(gemm_core PRIVATE GEMM_WITH_XNNPACK)
  target_link_libraries(gemm_core PUBLIC XNNPACK pthreadpool)
endif()

find_package(CUDAToolkit 11.4 QUIET)
option(GEMM_WITH_CUBLAS "cuBLAS kernel (needs CUDA Toolkit 11.4+)" ${CUDAToolkit_FOUND})
if(GEMM_WITH_CUBLAS)
  if(NOT CUDAToolkit_FOUND)
    message(FATAL_ERROR "GEMM_WITH_CUBLAS needs CUDA Toolkit 11.4+ (set CUDAToolkit_ROOT if it is not found)")
  endif()
  target_sources(gemm_core PRIVATE src/kernels/cublas.cpp)
  target_compile_definitions(gemm_core PRIVATE GEMM_WITH_CUBLAS)
  target_link_libraries(gemm_core PUBLIC CUDA::cudart CUDA::cublas)
endif()

if(NOT (GEMM_WITH_GGML OR GEMM_WITH_XNNPACK OR GEMM_WITH_GGML_VULKAN OR GEMM_WITH_CUBLAS))
  message(FATAL_ERROR "no kernels: enable at least one of GEMM_WITH_GGML, GEMM_WITH_XNNPACK, GEMM_WITH_GGML_VULKAN, GEMM_WITH_CUBLAS")
endif()

add_executable(gemm_bench
  bench/main.cpp
  bench/cases.cpp
  bench/report.cpp)
target_link_libraries(gemm_bench PRIVATE gemm_core)
target_compile_options(gemm_bench PRIVATE ${GEMM_WARNINGS})
target_compile_definitions(gemm_bench PRIVATE GEMM_DEFAULT_CASES="${CMAKE_CURRENT_SOURCE_DIR}/datasets/matmul.csv")

find_package(Catch2 3 QUIET)
if(NOT Catch2_FOUND)
  FetchContent_Declare(Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v3.16.1
    GIT_SHALLOW    TRUE)
  FetchContent_MakeAvailable(Catch2)
endif()
add_executable(gemm_tests tests/test_gemm.cpp)
target_link_libraries(gemm_tests PRIVATE gemm_core Catch2::Catch2WithMain)
target_compile_options(gemm_tests PRIVATE ${GEMM_WARNINGS})

enable_testing()
add_test(NAME gemm_tests COMMAND gemm_tests)

add_test(NAME gemm_bench_smoke
  COMMAND gemm_bench --case-name matmul_attention --warmup 1 --iters 2 --rounds 3
    --report ${CMAKE_CURRENT_BINARY_DIR}/gemm_bench_smoke.csv)
