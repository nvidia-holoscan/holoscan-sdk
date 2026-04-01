# GPU Resident Example

This example demonstrates how to use GPU-resident operators in Holoscan applications.

## Overview

The example creates a simple pipeline in a fragment with GPU-resident operators that execute directly on the GPU. It also creates another Holoscan SDK fragment that includes non-GPU-resident operators. The two fragments are not connected to each other. It includes:

- A custom `CustomGpuOp` operator that inherits from `GPUResidentOperator`
- Device memory allocation for input and output ports

This is WiP skeleton code for now. We will add more features to this example in the future.

## Platform support and CUDA driver compatibility

GPU-resident execution relies on CUDA graph capture APIs (for example, `cudaStreamBeginCaptureToGraph`) that require CUDA Runtime 12.3+ support from the installed NVIDIA driver.

- Verify driver/runtime compatibility with `nvidia-smi` (the reported CUDA version must be at least 12.3).
- IGX Orin dGPU systems running IGX OS 1.x typically use R535 drivers (CUDA 12.2 compatibility), so this example is not supported on that default host configuration.
- On unsupported systems, startup can fail with:
  - `API call is not supported in the installed CUDA driver (36)`
- An optional workaround on IGX OS 1.x is to install [CUDA forward-compat components from the CUDA repository](https://docs.nvidia.com/deploy/cuda-compatibility/forward-compatibility.html). This can make the example run, but it diverges from the default IGX OS 1.x package set and is not fully validated for all deployments.
