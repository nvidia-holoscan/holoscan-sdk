# GPU Resident Multi-IO Example

This example shows a fully GPU-resident Holoscan pipeline with three operators and two streams of device-memory data.

## Pipeline

- **SourceMultiOutputGpuOp**: Emits two device outputs from a CUDA kernel. The source sequence increases from 0 to 100, then wraps and repeats.
- **AddSubGpuOp**: Takes two device inputs and launches a CUDA kernel that computes elementwise sum and difference into two outputs.
- **FinalAddGpuOp**: Takes the sum and difference outputs and launches a CUDA kernel to add them again in-place (no output port).

## GPU-Resident Execution

- No host-side per-iteration `cudaMemcpy` is used.
- A GPU-side data-ready handler marks each iteration ready on device.
- The application waits up to 10 seconds for graph launch, runs for 10 seconds, then tears down.

## Platform support and CUDA driver compatibility

GPU-resident execution relies on CUDA graph capture APIs (for example, `cudaStreamBeginCaptureToGraph`) that require CUDA Runtime 12.3+ support from the installed NVIDIA driver.

- Verify driver/runtime compatibility with `nvidia-smi` (the reported CUDA version must be at least 12.3).
- IGX Orin dGPU systems running IGX OS 1.x typically use R535 drivers (CUDA 12.2 compatibility), so this example is not supported on that default host configuration.
- On unsupported systems, startup can fail with:
  - `API call is not supported in the installed CUDA driver (36)`
- An optional workaround on IGX OS 1.x is to install [CUDA forward-compat components from the CUDA repository](https://docs.nvidia.com/deploy/cuda-compatibility/forward-compatibility.html). This can make the example run, but it diverges from the default IGX OS 1.x package set and is not fully validated for all deployments.
