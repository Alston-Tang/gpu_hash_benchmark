// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

/**
 * GPU Vector Hash Kernel Microbenchmark - HIP/ROCm Implementation
 *
 * A simple microbenchmark program for measuring GPU SM/CU utilization.
 * This is the AMD ROCm native implementation using HIP runtime.
 *
 * This benchmark provides explicit control over thread blocks for precise
 * CU (Compute Unit) utilization testing.
 *
 * Usage:
 *     gpu_hash_benchmark_hip_main --num_blocks 1 --iterations 1000
 *
 * Parameters:
 *     --num_blocks: Number of thread blocks (controls CU utilization)
 *     --iterations: Number of iterations per kernel launch
 *     --runs: Number of benchmark runs
 *     --hash_rounds: Number of hash rounds per element (controls compute
 * intensity)
 */

#include <getopt.h>
#include <cstdio>
#include <cstdlib>

#include "gpu_hash_benchmark.h"

void print_usage(const char* program_name) {
  printf("GPU Vector Hash Kernel Microbenchmark (HIP/ROCm)\n\n");
  printf("Usage: %s [options]\n\n", program_name);
  printf("Options:\n");
  printf(
      "  --num_blocks N       Number of thread blocks (controls number of CUs used)\n");
  printf("                       Default: 1\n");
  printf("  --threads_per_block N Number of threads per block\n");
  printf("                       Default: 256\n");
  printf("  --iterations N       Number of iterations per kernel launch\n");
  printf("                       Default: 1000\n");
  printf("  --vector_size N      Size of vector per thread\n");
  printf("                       Default: 1024\n");
  printf("  --hash_rounds N      Number of hash rounds per element\n");
  printf("                       Default: 10\n");
  printf("  --runs N             Number of benchmark runs\n");
  printf("                       Default: 10\n");
  printf("  --help               Show this help message\n\n");
  printf("Examples:\n");
  printf("  # Run on single CU with minimal workload\n");
  printf("  %s --num_blocks 1 --iterations 100\n\n", program_name);
  printf("  # Run on all CUs with heavy workload\n");
  printf(
      "  %s --num_blocks 128 --iterations 10000 --hash_rounds 20\n",
      program_name);
}

void print_results(
    const BenchmarkConfig& config,
    const BenchmarkResults& results,
    const char* device_name) {
  printf("\n============================================================\n");
  printf("BENCHMARK RESULTS\n");
  printf("============================================================\n");
  printf("Device: %s\n", device_name);
  printf("Configuration:\n");
  printf(
      "  Blocks: %d (CUs available: %d)\n",
      config.num_blocks,
      results.sm_count);
  printf("  Threads/Block: %d\n", config.threads_per_block);
  printf("  Vector Size: %d\n", config.vector_size);
  printf("  Iterations: %d\n", config.iterations);
  printf("  Hash Rounds: %d\n", config.hash_rounds);
  printf("  Total Elements: %zu\n", results.total_elements);
  printf("\nTiming:\n");
  printf("  Average: %.3f ms\n", results.avg_time_ms);
  printf("  Min: %.3f ms\n", results.min_time_ms);
  printf("  Max: %.3f ms\n", results.max_time_ms);
  printf("  Std Dev: %.3f ms\n", results.std_time_ms);
  printf("\nThroughput:\n");
  printf("  Elements/sec: %.2e\n", results.elements_per_second);
  printf("  Hash ops/sec: %.2e\n", results.hash_ops_per_second);
  printf("============================================================\n");
}

int main(int argc, char* argv[]) {
  BenchmarkConfig config;

  static struct option long_options[] = {
      {"num_blocks", required_argument, nullptr, 'b'},
      {"threads_per_block", required_argument, nullptr, 't'},
      {"iterations", required_argument, nullptr, 'i'},
      {"vector_size", required_argument, nullptr, 'v'},
      {"hash_rounds", required_argument, nullptr, 'r'},
      {"runs", required_argument, nullptr, 'n'},
      {"help", no_argument, nullptr, 'h'},
      {nullptr, 0, nullptr, 0}};

  int opt;
  while ((opt = getopt_long(
              argc, argv, "b:t:i:v:r:n:h", long_options, nullptr)) != -1) {
    switch (opt) {
      case 'b':
        config.num_blocks = atoi(optarg);
        break;
      case 't':
        config.threads_per_block = atoi(optarg);
        break;
      case 'i':
        config.iterations = atoi(optarg);
        break;
      case 'v':
        config.vector_size = atoi(optarg);
        break;
      case 'r':
        config.hash_rounds = atoi(optarg);
        break;
      case 'n':
        config.runs = atoi(optarg);
        break;
      case 'h':
        print_usage(argv[0]);
        return 0;
      default:
        print_usage(argv[0]);
        return 1;
    }
  }

  printf("============================================================\n");
  printf("GPU VECTOR HASH KERNEL MICROBENCHMARK (HIP/ROCm)\n");
  printf("============================================================\n");

  int sm_count = 0;
  char device_name[256] = {0};
  get_device_info(sm_count, device_name, sizeof(device_name));

  BenchmarkResults results{};
  run_hash_benchmark(config, results);
  print_results(config, results, device_name);

  return 0;
}
