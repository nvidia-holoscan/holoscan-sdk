# hello_world

> **User guide:** [Hello World](https://docs.nvidia.com/holoscan/sdk-user-guide/examples/hello-world) —
> this install-smoke example verifies the smallest supported application.

Smallest possible installation smoke test: verifies that an installed Holoscan SDK can be found by CMake, linked against `holoscan::core`, and run by calling `holoscan::hello()`. This is not a tutorial — it only confirms your install works.

## Build

From the `cpp/` folder:

```bash
cmake -B build -S .
cmake --build build
```

## Run

```bash
./build/hello_world
```

## Expected output

```text
Hello Holoscan!
```

## What's next

Ready to actually build something? Start the guided series at [../first_pipeline](../first_pipeline/README.md).
