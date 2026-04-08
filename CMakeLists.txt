# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
#
# Open-source CMake build for hw_benchmarks on AMD (ROCm) machines.
#
# Prerequisites:
#   - ROCm SDK installed (provides hipcc, HIP runtime)
#   - PyTorch for ROCm (pip install torch --index-url https://download.pytorch.org/whl/rocm6.3)
#   - Triton (pip install triton) for gpu_hash_benchmark.py
#
# Usage:
#   mkdir build && cd build
#   cmake .. -DCMAKE_PREFIX_PATH=/opt/rocm
#   make -j$(nproc)
#
# This builds the HIP C++ binary: gpu_hash_benchmark_hip_main

cmake_minimum_required(VERSION 3.21)

# ROCm GPU architecture — must be set before project() enables HIP.
# Override with -DCMAKE_HIP_ARCHITECTURES=<arch> at cmake time.
# Common values: gfx942 (MI300X/MI325X), gfx950 (MI355X), gfx1100 (RX 7900).
if(NOT DEFINED CMAKE_HIP_ARCHITECTURES)
    set(CMAKE_HIP_ARCHITECTURES "gfx942" CACHE STRING "ROCm GPU architectures to compile for")
endif()

project(hw_benchmarks LANGUAGES CXX HIP)

# Find HIP runtime package (provides hip::device target for linking)
find_package(hip REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------- gpu_hash_benchmark HIP binary ----------
set(HIP_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/gpu_hash_benchmark")

# Build everything as a single HIP executable — hipcc handles both .hip and .cc
# files, so there is no flag leakage between compilers.
add_executable(gpu_hash_benchmark_hip_main
    "${HIP_SRC_DIR}/gpu_hash_benchmark_main.cc"
    "${HIP_SRC_DIR}/gpu_hash_benchmark.hip"
)

# Mark ALL sources as HIP so hipcc compiles them uniformly.
set_source_files_properties(
    "${HIP_SRC_DIR}/gpu_hash_benchmark_main.cc"
    "${HIP_SRC_DIR}/gpu_hash_benchmark.hip"
    PROPERTIES LANGUAGE HIP
)

target_include_directories(gpu_hash_benchmark_hip_main PRIVATE "${HIP_SRC_DIR}")
target_link_libraries(gpu_hash_benchmark_hip_main PRIVATE hip::device)

install(TARGETS gpu_hash_benchmark_hip_main DESTINATION bin)
