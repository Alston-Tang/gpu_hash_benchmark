// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <hip/hip_runtime.h>
#include <cstddef>
#include <cstdint>

#define HIP_CHECK(call)               \
  do {                                \
    hipError_t err = call;            \
    if (err != hipSuccess) {          \
      fprintf(                        \
          stderr,                     \
          "HIP error at %s:%d: %s\n", \
          __FILE__,                   \
          __LINE__,                   \
          hipGetErrorString(err));    \
      exit(EXIT_FAILURE);             \
    }                                 \
  } while (0)

struct BenchmarkConfig {
  int num_blocks = 1;
  int threads_per_block = 256;
  int iterations = 1000;
  int vector_size = 1024;
  int hash_rounds = 10;
  int runs = 10;
};

struct BenchmarkResults {
  double avg_time_ms;
  double min_time_ms;
  double max_time_ms;
  double std_time_ms;
  double elements_per_second;
  double hash_ops_per_second;
  size_t total_elements;
  int sm_count;
};

void run_hash_benchmark(
    const BenchmarkConfig& config,
    BenchmarkResults& results);

void get_device_info(int& sm_count, char* device_name, size_t name_len);
