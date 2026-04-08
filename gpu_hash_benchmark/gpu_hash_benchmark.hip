// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

/**
 * GPU Vector Hash Kernel Microbenchmark - HIP Implementation
 *
 * A simple microbenchmark program for measuring GPU SM utilization using HIP.
 * This is the AMD ROCm native implementation of the hash benchmark.
 *
 * This benchmark provides explicit control over thread blocks for precise
 * SM/CU utilization testing.
 *
 * The hash kernel uses a simplified mixing function:
 *   x ^= x >> 16
 *   x *= 0x45D9F3B
 *   x ^= x >> 13
 *   x *= 0x119DE1F3
 *   x ^= x >> 16
 */

#include "gpu_hash_benchmark.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

constexpr int BLOCK_SIZE = 1024;

/**
 * Hash kernel for compute-intensive workload.
 *
 * Each thread block processes BLOCK_SIZE elements.
 * The grid size determines how many blocks run in parallel.
 *
 * Uses simpler hash constants that fit in signed int32:
 *   - 0x45D9F3B (fits in int32)
 *   - 0x119DE1F3 (fits in int32)
 */
__global__ void hash_kernel(
    const int32_t* __restrict__ input,
    int32_t* __restrict__ output,
    size_t n_elements,
    int iterations,
    int hash_rounds) {
  size_t pid = blockIdx.x;
  size_t block_start = pid * BLOCK_SIZE;

  for (size_t i = threadIdx.x; i < BLOCK_SIZE; i += blockDim.x) {
    size_t offset = block_start + i;
    if (offset >= n_elements) {
      continue;
    }

    int32_t x = input[offset];

    for (int iter = 0; iter < iterations; ++iter) {
      for (int round = 0; round < hash_rounds; ++round) {
        x ^= (x >> 16);
        x *= 0x45D9F3B;
        x ^= (x >> 13);
        x *= 0x119DE1F3;
        x ^= (x >> 16);
      }
    }

    output[offset] = x;
  }
}

void get_device_info(int& sm_count, char* device_name, size_t name_len) {
  int device_id = 0;
  HIP_CHECK(hipGetDevice(&device_id));

  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop, device_id));

  sm_count = prop.multiProcessorCount;
  snprintf(device_name, name_len, "%s", prop.name);

  printf("Device: %s\n", prop.name);
  printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
  printf("  Total Memory: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
  printf("  Multiprocessors (CUs): %d\n", prop.multiProcessorCount);
  printf("  Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
  printf("  Warp Size: %d\n", prop.warpSize);
}

void run_hash_benchmark(
    const BenchmarkConfig& config,
    BenchmarkResults& results) {
  size_t total_elements =
      static_cast<size_t>(config.num_blocks) * BLOCK_SIZE;

  int sm_count = 0;
  {
    int device_id = 0;
    HIP_CHECK(hipGetDevice(&device_id));
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, device_id));
    sm_count = prop.multiProcessorCount;
  }

  printf("\nBenchmark Configuration:\n");
  printf("  Num Blocks: %d\n", config.num_blocks);
  printf("  Threads per Block: %d\n", BLOCK_SIZE);
  printf("  Total Elements: %zu\n", total_elements);
  printf("  Iterations per Kernel: %d\n", config.iterations);
  printf("  Hash Rounds per Element: %d\n", config.hash_rounds);
  printf("  Benchmark Runs: %d\n", config.runs);
  printf(
      "  Theoretical CU Usage: %d/%d\n",
      std::min(config.num_blocks, sm_count),
      sm_count);
  printf("  Backend: HIP/ROCm\n");

  std::vector<int32_t> host_data(total_elements);
  std::mt19937 rng(42);
  std::uniform_int_distribution<int32_t> dist(0, INT32_MAX);
  for (size_t i = 0; i < total_elements; ++i) {
    host_data[i] = dist(rng);
  }

  int32_t* d_input = nullptr;
  int32_t* d_output = nullptr;
  size_t data_size = total_elements * sizeof(int32_t);

  HIP_CHECK(hipMalloc(&d_input, data_size));
  HIP_CHECK(hipMalloc(&d_output, data_size));
  HIP_CHECK(
      hipMemcpy(d_input, host_data.data(), data_size, hipMemcpyHostToDevice));

  printf("\nRunning benchmark...\n");
  std::vector<double> times;
  times.reserve(config.runs);

  dim3 grid_dim(config.num_blocks);
  dim3 block_dim(BLOCK_SIZE);

  for (int run = 0; run < config.runs; ++run) {
    HIP_CHECK(hipDeviceSynchronize());

    auto start = std::chrono::high_resolution_clock::now();

    hipLaunchKernelGGL(
        hash_kernel,
        grid_dim,
        block_dim,
        0,
        0,
        d_input,
        d_output,
        total_elements,
        config.iterations,
        config.hash_rounds);

    HIP_CHECK(hipDeviceSynchronize());

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed_ms =
        std::chrono::duration<double, std::milli>(end - start).count();
    times.push_back(elapsed_ms);

    printf("  Run %d/%d: %.3f ms\n", run + 1, config.runs, elapsed_ms);
  }

  HIP_CHECK(hipFree(d_input));
  HIP_CHECK(hipFree(d_output));

  if (times.empty()) {
    fprintf(stderr, "Error: no benchmark runs completed (runs=%d)\n", config.runs);
    return;
  }

  double sum = 0.0;
  double min_time = times[0];
  double max_time = times[0];

  for (double t : times) {
    sum += t;
    min_time = std::min(min_time, t);
    max_time = std::max(max_time, t);
  }

  double avg_time = sum / times.size();

  double variance = 0.0;
  for (double t : times) {
    variance += (t - avg_time) * (t - avg_time);
  }
  double std_time = std::sqrt(variance / times.size());

  double elements_per_second =
      static_cast<double>(total_elements) *
      static_cast<double>(config.iterations) / (avg_time / 1000.0);
  double hash_ops_per_second =
      elements_per_second * static_cast<double>(config.hash_rounds);

  results.avg_time_ms = avg_time;
  results.min_time_ms = min_time;
  results.max_time_ms = max_time;
  results.std_time_ms = std_time;
  results.elements_per_second = elements_per_second;
  results.hash_ops_per_second = hash_ops_per_second;
  results.total_elements = total_elements;
  results.sm_count = sm_count;
}
