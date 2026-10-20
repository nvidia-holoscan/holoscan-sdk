# hello_schema

> **User guide:** [Hello Tour](https://docs.nvidia.com/holoscan/sdk-user-guide/examples/hello-tour) —
> this install-smoke example isolates the generated schema helpers.

Installation smoke test (not a tutorial) for the `holoscan::schema` library: it builds image metadata for a small host tensor with `build_image_metadata()`, parses it back with `parse_image_metadata()`, and verifies the round trip. This exercises the install-tree story for the schema DSO and its FlatBuffers-generated headers; a broken schema install fails here at configure/compile time. Host-only — no CUDA device required.

## Build

From the `cpp/` folder:

```bash
cmake -B build -S .
cmake --build build
```

## Run

```bash
./build/hello_schema
```

## Expected output

```text
image metadata height=3 width=2 encoding=<MONO32F code>
holoscan schema sample complete
```

(The `encoding=` value is the numeric FlatBuffers code for `ImageEncoding_MONO32F`.)

## What's next

To start actually learning Holoscan, head to [../first_pipeline](../first_pipeline/README.md).
