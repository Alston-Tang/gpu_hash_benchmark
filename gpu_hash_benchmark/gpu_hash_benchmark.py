#!/usr/bin/env python3
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""
GPU Vector Hash Kernel Microbenchmark

A simple microbenchmark program for measuring GPU SM utilization using Triton.
Supports both NVIDIA (CUDA) and AMD (ROCm) hardware.

This benchmark provides explicit control over thread blocks for precise SM utilization testing.

Usage:
    python gpu_hash_benchmark.py --num_blocks 1 --iterations 1000

Parameters:
    --num_blocks: Number of thread blocks (controls SM utilization)
    --iterations: Number of iterations per kernel launch
    --runs: Number of benchmark runs
    --hash_rounds: Number of hash rounds per element (controls compute intensity)
"""

import argparse
import sys
import time
from typing import Tuple

import torch

# Try to import Triton at module level for kernel definition
_triton_available = False
_triton_error = None

try:
    import triton
    import triton.language as tl

    # Import the fb config_backend to properly configure Triton for PAR files.
    # This backend auto-registers and handles libcuda/libdevice paths.
    # Optional: only available in Meta-internal (Buck/PAR) builds.
    try:
        from triton.backends import fb_config  # noqa: F401
    except ImportError:
        pass

    @triton.jit
    def _hash_kernel_triton(
        input_ptr,
        output_ptr,
        n_elements,
        iterations: tl.constexpr,
        hash_rounds: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        Triton kernel for hash computation.

        Each program (block) processes BLOCK_SIZE elements.
        The grid size determines how many blocks run in parallel.
        """
        pid = tl.program_id(0)
        # Each block processes BLOCK_SIZE elements starting at pid * BLOCK_SIZE
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        x = tl.load(input_ptr + offsets, mask=mask, other=0)

        # Use simpler hash constants that fit in signed int32
        # Based on a simplified mixing function
        for _ in range(iterations):
            for _ in range(hash_rounds):
                x = x ^ (x >> 16)
                x = x * 0x45D9F3B  # Fits in int32
                x = x ^ (x >> 13)
                x = x * 0x119DE1F3  # Fits in int32
                x = x ^ (x >> 16)

        tl.store(output_ptr + offsets, x, mask=mask)

    _triton_available = True
except ImportError as e:
    _triton_error = f"Triton import failed: {e}"
except Exception as e:
    _triton_error = f"Triton initialization failed: {e}"


def check_triton_available() -> None:
    """Check if Triton is available and raise error if not."""
    if not _triton_available:
        print(f"ERROR: {_triton_error}", file=sys.stderr)
        print(
            "This benchmark requires Triton for explicit thread block control.",
            file=sys.stderr,
        )
        sys.exit(1)


def get_device_info() -> Tuple[torch.device, str]:
    """Detect available GPU and return device info."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(0)
        device_props = torch.cuda.get_device_properties(0)
        print(f"Device: {device_name}")
        print(f"  Compute Capability: {device_props.major}.{device_props.minor}")
        print(f"  Total Memory: {device_props.total_memory / (1024**3):.2f} GB")
        print(f"  Multiprocessors (SMs): {device_props.multi_processor_count}")
        print(
            f"  Max Threads per Block: {getattr(device_props, 'max_threads_per_block', 'N/A')}"
        )
        return device, device_name
    else:
        raise RuntimeError("No CUDA/ROCm GPU available")


class GPUHashBenchmark:
    """GPU Hash Benchmark with configurable workload parameters using Triton."""

    def __init__(
        self,
        num_blocks: int = 1,
        threads_per_block: int = 256,
        iterations: int = 1000,
        vector_size: int = 1024,
        hash_rounds: int = 10,
        runs: int = 10,
    ):
        # Check Triton availability first
        check_triton_available()

        self.num_blocks = num_blocks
        self.threads_per_block = threads_per_block
        self.iterations = iterations
        self.vector_size = vector_size
        self.hash_rounds = hash_rounds
        self.runs = runs

        self.device, self.device_name = get_device_info()
        self.sm_count = torch.cuda.get_device_properties(0).multi_processor_count

        # Total elements = num_blocks * threads_per_block * vector_size
        self.total_elements = num_blocks * threads_per_block * vector_size

        print("\nBenchmark Configuration:")
        print(f"  Num Blocks: {num_blocks}")
        print(f"  Threads per Block: {threads_per_block}")
        print(f"  Vector Size per Thread: {vector_size}")
        print(f"  Total Elements: {self.total_elements:,}")
        print(f"  Iterations per Kernel: {iterations}")
        print(f"  Hash Rounds per Element: {hash_rounds}")
        print(f"  Benchmark Runs: {runs}")
        print(
            f"  Theoretical SM Usage: {min(num_blocks, self.sm_count)}/{self.sm_count}"
        )
        print("  Backend: Triton")

        # Initialize data
        self.data = torch.randint(
            0,
            2**31 - 1,
            (self.total_elements,),
            dtype=torch.int32,
            device=self.device,
        )

    def run_kernel(self) -> torch.Tensor:
        """Run the hash kernel once."""
        output = torch.empty_like(self.data)
        # Use a fixed BLOCK_SIZE that works with Triton (must be power of 2, <= 1024)
        BLOCK_SIZE = 1024
        # Each block processes BLOCK_SIZE elements
        grid = (self.num_blocks,)
        _hash_kernel_triton[grid](
            self.data,
            output,
            self.total_elements,
            # pyre-ignore[6]: Triton JIT handles constexpr types internally
            self.iterations,
            # pyre-ignore[6]: Triton JIT handles constexpr types internally
            self.hash_rounds,
            # pyre-ignore[6]: Triton JIT handles constexpr types internally
            BLOCK_SIZE,
        )
        return output

    def benchmark(self) -> dict:
        """Run the benchmark and return timing statistics."""
        print("\nRunning benchmark...")
        times = []

        for _i in range(self.runs):
            torch.cuda.synchronize()
            start = time.perf_counter()

            _ = self.run_kernel()

            torch.cuda.synchronize()
            end = time.perf_counter()

            elapsed_ms = (end - start) * 1000
            times.append(elapsed_ms)
            print(f"  Run {_i + 1}/{self.runs}: {elapsed_ms:.3f} ms")

        # Calculate statistics
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        std_time = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5

        # Calculate throughput
        elements_per_second = self.total_elements * self.iterations / (avg_time / 1000)
        hash_ops_per_second = elements_per_second * self.hash_rounds

        results = {
            "avg_time_ms": avg_time,
            "min_time_ms": min_time,
            "max_time_ms": max_time,
            "std_time_ms": std_time,
            "elements_per_second": elements_per_second,
            "hash_ops_per_second": hash_ops_per_second,
            "num_blocks": self.num_blocks,
            "threads_per_block": self.threads_per_block,
            "iterations": self.iterations,
            "vector_size": self.vector_size,
            "hash_rounds": self.hash_rounds,
            "total_elements": self.total_elements,
            "sm_count": self.sm_count,
        }

        return results

    def print_results(self, results: dict):
        """Print benchmark results."""
        print("\n" + "=" * 60)
        print("BENCHMARK RESULTS")
        print("=" * 60)
        print(f"Device: {self.device_name}")
        print("Configuration:")
        print(
            f"  Blocks: {results['num_blocks']} (SMs available: {results['sm_count']})"
        )
        print(f"  Threads/Block: {results['threads_per_block']}")
        print(f"  Vector Size: {results['vector_size']}")
        print(f"  Iterations: {results['iterations']}")
        print(f"  Hash Rounds: {results['hash_rounds']}")
        print(f"  Total Elements: {results['total_elements']:,}")
        print("\nTiming:")
        print(f"  Average: {results['avg_time_ms']:.3f} ms")
        print(f"  Min: {results['min_time_ms']:.3f} ms")
        print(f"  Max: {results['max_time_ms']:.3f} ms")
        print(f"  Std Dev: {results['std_time_ms']:.3f} ms")
        print("\nThroughput:")
        print(f"  Elements/sec: {results['elements_per_second']:.2e}")
        print(f"  Hash ops/sec: {results['hash_ops_per_second']:.2e}")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="GPU Vector Hash Kernel Microbenchmark (Triton-based)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run on single SM with minimal workload
  python gpu_hash_benchmark.py --num_blocks 1 --iterations 100

  # Run on all SMs with heavy workload
  python gpu_hash_benchmark.py --num_blocks 128 --iterations 10000 --hash_rounds 20
        """,
    )

    parser.add_argument(
        "--num_blocks",
        type=int,
        default=1,
        help="Number of thread blocks (controls number of SMs used)",
    )
    parser.add_argument(
        "--threads_per_block",
        type=int,
        default=256,
        help="Number of threads per block (default: 256)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1000,
        help="Number of iterations per kernel launch (default: 1000)",
    )
    parser.add_argument(
        "--vector_size",
        type=int,
        default=1024,
        help="Size of vector per thread (default: 1024)",
    )
    parser.add_argument(
        "--hash_rounds",
        type=int,
        default=10,
        help="Number of hash rounds per element (controls compute intensity, default: 10)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=10,
        help="Number of benchmark runs (default: 10)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("GPU VECTOR HASH KERNEL MICROBENCHMARK (Triton)")
    print("=" * 60)

    benchmark = GPUHashBenchmark(
        num_blocks=args.num_blocks,
        threads_per_block=args.threads_per_block,
        iterations=args.iterations,
        vector_size=args.vector_size,
        hash_rounds=args.hash_rounds,
        runs=args.runs,
    )
    results = benchmark.benchmark()
    benchmark.print_results(results)


if __name__ == "__main__":
    main()
