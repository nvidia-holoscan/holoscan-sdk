# hello_core

> **User guide:** [Hello Tour](https://docs.nvidia.com/holoscan/sdk-user-guide/examples/hello-tour) —
> this install-smoke example isolates the core errors-as-values surface.

Installation smoke test (not a tutorial) for the `holoscan::core` error-handling API: it round-trips a value and an error through `holoscan::expected<T, Error>` and names the error code via `error_code_name()`, so a broken install of the header-only `tl::expected` / `magic_enum` interface dependencies fails here at configure/compile time.

## Build

From the `cpp/` folder:

```bash
cmake -B build -S . -Dholoscan_DIR=/opt/nvidia/holoscan/lib/cmake/holoscan
cmake --build build
```

## Run

```bash
./build/hello_core
```

## Expected output

```text
value=42, error=kInvalidArgument: value was rejected
holoscan core sample complete
```

## What's next

This only proves the SDK is installed correctly. To start learning Holoscan, go to [../first_pipeline](../first_pipeline/README.md).
