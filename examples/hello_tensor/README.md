# hello_tensor

> **User guide:** [Hello Tour](https://docs.nvidia.com/holoscan/sdk-user-guide/examples/hello-tour) —
> this install-smoke example isolates host Tensor construction and metadata.

Installation smoke test (not a tutorial) for `holoscan::Tensor`: it wraps a small host (CPU-only, no GPU needed) buffer in a DLPack `DLManagedTensorContext`, constructs a `Tensor`, and checks its `ndim()`/`shape()` metadata. If `find_package(holoscan)` cannot resolve the public `dlpack` dependency or the tensor headers are missing from the install tree, this example fails to configure or compile — exactly what it exists to catch.

## Build

From the `cpp/` folder:

```bash
cmake -B build -S . -Dholoscan_DIR=/opt/nvidia/holoscan/lib/cmake/holoscan
cmake --build build
```

## Run

```bash
./build/hello_tensor
```

## Expected output

```text
tensor ndim=2 shape=[2,3]
holoscan tensor sample complete
```

## What's next

For the guided introduction to Holoscan, continue at [../first_pipeline](../first_pipeline/README.md).
