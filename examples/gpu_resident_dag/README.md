# GPU Resident DAG Example

This example demonstrates a GPU-resident diamond-shaped DAG that uses the latest GPU-resident DAG support in Holoscan.

## Pipeline

- `SourceGpuOp`: exposes a single GPU-resident output buffer that the host fills before each iteration.
- `AddGpuOp`: receives the source buffer and adds a constant to every element.
- `SubtractGpuOp`: receives the same source buffer and subtracts a constant from every element.
- `MultiplySinkGpuOp`: converges the two branches and multiplies the add and subtract results elementwise.

The graph shape is:

`source -> add -> multiply_sink`

`source -> subtract -> multiply_sink`

## GPU-Resident Execution

- The source output fans out to both branch operators, exercising the new single-source DAG support for GPU-resident execution.
- The sink has two distinct device input ports.
- The host copies input data into the source output buffer, calls `data_ready()`, and then reads the sink result back for verification.
- `sync_with_host()` is enabled because the example performs host-side `cudaMemcpy` after each GPU-resident iteration.

## Platform support and CUDA driver compatibility

GPU-resident execution relies on CUDA graph capture APIs (for example, `cudaStreamBeginCaptureToGraph`) that require CUDA Runtime 12.3+ support from the installed NVIDIA driver.

- Verify driver/runtime compatibility with `nvidia-smi` (the reported CUDA version must be at least 12.3).
- IGX Orin dGPU systems running IGX OS 1.x typically use R535 drivers (CUDA 12.2 compatibility), so this example is not supported on that default host configuration.
- On unsupported systems, startup can fail with:
  - `API call is not supported in the installed CUDA driver (36)`
- An optional workaround on IGX OS 1.x is to install [CUDA forward-compat components from the CUDA repository](https://docs.nvidia.com/deploy/cuda-compatibility/forward-compatibility.html). This can make the example run, but it diverges from the default IGX OS 1.x package set and is not fully validated for all deployments.
