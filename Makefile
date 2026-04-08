# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
#
# Makefile for building hw_benchmarks on AMD (ROCm) machines outside of Buck.
#
# Prerequisites:
#   - ROCm SDK (provides hipcc) — typically at /opt/rocm
#   - Python 3.10+ with PyTorch for ROCm
#   - triton (for gpu_hash_benchmark.py)
#
# Targets:
#   make hip          — build the HIP C++ gpu_hash_benchmark binary
#   make install-deps — install Python dependencies (PyTorch ROCm + triton)
#   make all          — build everything
#   make clean        — remove build artifacts

ROCM_PATH   ?= /opt/rocm
GPU_TARGETS ?= gfx942
BUILD_DIR   ?= build

.PHONY: all hip install-deps clean

all: hip

# ---- HIP C++ benchmark ----
hip:
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && cmake .. \
		-DCMAKE_PREFIX_PATH=$(ROCM_PATH) \
		-DCMAKE_HIP_ARCHITECTURES=$(GPU_TARGETS) \
		-DCMAKE_BUILD_TYPE=Release
	cmake --build $(BUILD_DIR) -j$$(nproc)
	@echo ""
	@echo "HIP binary built: $(BUILD_DIR)/gpu_hash_benchmark_hip_main"

# ---- Python dependencies ----
install-deps:
	pip install torch --index-url https://download.pytorch.org/whl/rocm6.3
	pip install -r requirements.txt
	@echo ""
	@echo "Python dependencies installed."

clean:
	rm -rf $(BUILD_DIR)
