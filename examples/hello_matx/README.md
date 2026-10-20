# hello_matx

> **User guide:** [Hello Tour](https://docs.nvidia.com/holoscan/sdk-user-guide/examples/hello-tour) —
> this install-smoke example verifies MatX interoperability.

Installation smoke test (not a tutorial) verifying that the installed SDK coexists with MatX and CUDA: it runs a small MatX expression (`linspace + 1`) on a CUDA stream via `cudaExecutor` and checks the last element. Requires a CUDA-capable GPU; without one it prints a skip message and exits with code 77 (CTest skip).

## Build

From the `cpp/` folder:

```bash
cmake -B build -S .
cmake --build build
```

MatX is not included in the SDK Debian packages or in the development container image. An SDK built from source with examples enabled (the default) installs MatX into its install tree. The build uses an installed MatX when CMake finds one (pass `-DCMAKE_PREFIX_PATH=<sdk-install-dir>` or `-DCMAKE_PREFIX_PATH=<matx-prefix>` to select it) and otherwise downloads the revision the SDK is tested with, which needs network access.

## Run

```bash
./build/hello_matx
```

## Expected output

```
Hello Holoscan!
MatX CUDA last value: 2
MatX CUDA sample complete
```

Or, with no CUDA device: `No CUDA-capable device is available; skipping hello_matx.`

## What's next

For the guided Holoscan learning path, continue at [../first_pipeline](../first_pipeline/README.md).
