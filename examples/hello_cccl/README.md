# hello_cccl

> **User guide:** [Hello Tour](https://docs.nvidia.com/holoscan/sdk-user-guide/examples/hello-tour) —
> this install-smoke example verifies CCCL interoperability.

Installation smoke test (not a tutorial) verifying that the installed SDK coexists with CCCL (CUB, Thrust, libcudacxx) and CUDA: it fills a `thrust::device_vector`, sums 1024 ones with a `cub::BlockReduce` kernel plus `cuda::atomic_ref`, and checks the result. Requires a CUDA-capable GPU; without one it prints a skip message and exits with code 77 (CTest skip).

## Build

From the `cpp/` folder:

```bash
cmake -B build -S .
cmake --build build
```

## Run

```bash
./build/hello_cccl
```

## Expected output

```
Hello Holoscan!
CCCL CUDA sum: 1024
CCCL CUDA sample complete
```

Or, with no CUDA device: `No CUDA-capable device is available; skipping hello_cccl.`

## What's next

For the guided Holoscan learning path, continue at [../first_pipeline](../first_pipeline/README.md).
