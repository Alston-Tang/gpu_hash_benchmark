# HW Benchmarks — AMD (ROCm) Open-Source Build

GPU hardware benchmarks for measuring compute performance, memory bandwidth,
PCIe transfer rates, and SM/CU utilization on AMD GPUs.

## Benchmarks

| Benchmark | Type | What it measures |
|-----------|------|------------------|
| `matrix_multiplication_benchmark` | Python (PyTorch) | GPU FLOPS via GEMM (FP64/FP32/FP16/BF16/FP8/INT8) |
| `stream_benchmark` | Python (PyTorch) | GPU memory (HBM) bandwidth (Copy/Scale/Add/Triad) |
| `device_host_transfer_benchmark` | Python (PyTorch) | PCIe bandwidth (host-to-device / device-to-host) |
| `gpu_hash_benchmark` | Python (PyTorch + Triton) | SM/CU utilization via hash kernel |
| `gpu_hash_benchmark_hip_main` | C++ (HIP/ROCm) | SM/CU utilization via native HIP hash kernel |

## Prerequisites

- **ROCm SDK** (6.0+) installed at `/opt/rocm` (provides `hipcc`, HIP runtime)
- **Python 3.10+**
- **CMake 3.21+** (for the HIP C++ benchmark)

## Setup

### Python dependencies

Install PyTorch for ROCm and Triton:

```bash
pip install torch --index-url https://download.pytorch.org/whl/rocm6.3
pip install -r requirements.txt
```

Or use the Makefile shortcut:

```bash
make install-deps
```

### Build the HIP C++ benchmark

Using Make (recommended):

```bash
# Default: builds for MI300X (gfx942)
make hip

# For MI355X:
GPU_TARGETS=gfx950 make hip
```

Or using CMake directly:

```bash
mkdir build && cd build
cmake .. -DCMAKE_PREFIX_PATH=/opt/rocm -DCMAKE_HIP_ARCHITECTURES=gfx942
make -j$(nproc)
```

Common GPU architecture values:

| GPU | `GPU_TARGETS` / `CMAKE_HIP_ARCHITECTURES` |
|-----|-------------------------------------------|
| MI300X / MI325X | `gfx942` (default) |
| MI355X | `gfx950` |
| RX 7900 | `gfx1100` |

## Running the benchmarks

### Matrix Multiplication (FLOPS)

```bash
# Default: FP32, 4096x4096 matrices (runs until Ctrl+C)
python3 matrix_multiplication_benchmark/matrix_multiplication_benchmark.py

# FP16, custom dimensions
python3 matrix_multiplication_benchmark/matrix_multiplication_benchmark.py \
    --dtype fp16 --m 8192 --n 8192 --k 4096
```

### STREAM (Memory Bandwidth)

```bash
# Triad kernel, unlimited bandwidth (runs until Ctrl+C)
python3 stream_benchmark/stream_benchmark.py --kernel triad --target_bandwidth_gbps 0

# Copy kernel with throttling
python3 stream_benchmark/stream_benchmark.py --kernel copy --target_bandwidth_gbps 500.0
```

### Device-Host Transfer (PCIe Bandwidth)

```bash
# Host-to-device, transfer 1 GB total
python3 device_host_transfer_benchmark/device_host_transfer_benchmark.py \
    --direction h2d --total_data_gb 1.0

# Device-to-host at target bandwidth (runs until Ctrl+C)
python3 device_host_transfer_benchmark/device_host_transfer_benchmark.py \
    --direction d2h --target_bandwidth_gbps 10.0

# Bidirectional, unlimited
python3 device_host_transfer_benchmark/device_host_transfer_benchmark.py \
    --direction bidirectional --target_bandwidth_gbps 0
```

### GPU Hash — PyTorch/Triton

```bash
python3 gpu_hash_benchmark/gpu_hash_benchmark.py \
    --num_blocks 128 --iterations 1000
```

### GPU Hash — HIP (C++)

```bash
# After building with make hip:
./build/gpu_hash_benchmark_hip_main --num_blocks 128 --iterations 1000

# See all options:
./build/gpu_hash_benchmark_hip_main --help
```

## Example Output

Below is an example run of the HIP C++ hash benchmark on an AMD Instinct MI300X:

```
============================================================
GPU VECTOR HASH KERNEL MICROBENCHMARK (HIP/ROCm)
============================================================
Device: AMD Instinct MI300X
  Compute Capability: 9.4
  Total Memory: 191.98 GB
  Multiprocessors (CUs): 304
  Max Threads per Block: 1024
  Warp Size: 64

Benchmark Configuration:
  Num Blocks: 1
  Threads per Block: 1024
  Total Elements: 1024
  Iterations per Kernel: 10000
  Hash Rounds per Element: 10000
  Benchmark Runs: 10
  Theoretical CU Usage: 1/304
  Backend: HIP/ROCm

Running benchmark...
  Run 1/10: 4343.667 ms
  Run 2/10: 4344.608 ms
  Run 3/10: 4348.495 ms
  Run 4/10: 4342.410 ms
  Run 5/10: 4344.577 ms
  Run 6/10: 4344.157 ms
  Run 7/10: 4342.589 ms
  Run 8/10: 4343.869 ms
  Run 9/10: 4343.765 ms
  Run 10/10: 4343.050 ms

============================================================
BENCHMARK RESULTS
============================================================
Device: AMD Instinct MI300X
Configuration:
  Blocks: 1 (CUs available: 304)
  Threads/Block: 256
  Vector Size: 1024
  Iterations: 10000
  Hash Rounds: 10000
  Total Elements: 1024

Timing:
  Average: 4344.119 ms
  Min: 4342.410 ms
  Max: 4348.495 ms
  Std Dev: 1.625 ms

Throughput:
  Elements/sec: 2.36e+06
  Hash ops/sec: 2.36e+10
============================================================
```

## Cleanup

```bash
make clean
```
