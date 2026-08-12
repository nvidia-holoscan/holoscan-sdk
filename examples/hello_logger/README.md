# hello_logger

> **User guide:** [Hello Tour](https://docs.nvidia.com/holoscan/sdk-user-guide/examples/hello-tour) —
> this install-smoke example isolates logging configuration and macros.

Installation smoke test (not a tutorial) for `holoscan::logger`: it sets the log level and emits one fmt-formatted `HOLOSCAN_LOG_INFO` message, proving the installed logger target and its public header-only `fmt` dependency resolve through `find_package(holoscan)`. It also asserts the caller-owned compile-time log threshold policy is preserved by the installed headers.

## Build

From the `cpp/` folder:

```bash
cmake -B build -S . -Dholoscan_DIR=/opt/nvidia/holoscan/lib/cmake/holoscan
cmake --build build
```

## Run

```bash
./build/hello_logger
```

## Expected output

An INFO-level log line containing `hello from the holoscan logger: answer=42` (timestamp/severity prefix depends on logger formatting), then:

```text
holoscan logger sample complete
```

## What's next

To move from smoke tests to real learning, start at [../first_pipeline](../first_pipeline/README.md).
