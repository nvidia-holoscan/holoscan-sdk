# Buffer.idl — AccelBuffer and host Buffer types

## Overview

This example uses Fast DDS types from **`idl/Buffer.idl`** (multiple structs in one file), generated into `gen/`. The accelerated type **AccelBuffer** has:

- **`descriptor`** — a `PointerDescriptor` (from holoscan::ipc) with `key`, `handle_type`, `handle`, and `reply_to_topic_name`
- **`size`** — buffer size (`unsigned long`)
- **`index`** — 0-based publisher sample sequence (`unsigned long`), for detecting gaps on the subscriber

The device pointer is **not** transmitted. The publisher calls `Context::share_pointer(cuda_ptr, HandleType::CUDA_IPC)` to get a descriptor and publishes it inside `AccelBuffer`. The subscriber receives the sample, then calls `Context::acquire_pointer(descriptor)` (or `acquire_pointer_eager` with subscriber `--eager`) to obtain the device pointer (ACQUIRED/ACK/RELEASED protocol; eager opens before ACK—see SDK `docs/EAGER_ACQUIRE.md`). See [README.md](README.md) (latency measurement, `run_latency_benchmark.sh`, sweep) and the source (`PublisherApp.cpp`, `ListenerSubscriberApp.cpp`) for the full flow.

## Files (this example)

- **`idl/Buffer.idl`** — defines **`AccelBuffer`** (`PointerDescriptor` + `size` + `index`) and **`Buffer`** (`sequence<octet> data` + `index` for host payloads)
- **`gen/`** — IDL-generated C++ (`Buffer.hpp` with both structs, `BufferPubSubTypes.hpp`, etc.)
- **Publisher:** `PublisherApp.cpp`, `CudaDataWriterListener.cpp` — allocate CUDA memory, `share_pointer()`, write `AccelBuffer` samples
- **Subscriber:** `ListenerSubscriberApp.cpp` — receive samples, `acquire_pointer(descriptor).get()`, use pointer, then release
- **Application, CLI, main:** `Application.hpp`/`.cpp`, `CLIParser.hpp`, `main.cpp`

## What gets transmitted

**Over the wire (in `AccelBuffer.descriptor` and `AccelBuffer.size`):**

- **`key`** — opaque identifier for the pointer (sequence of bytes)
- **`handle_type`** — e.g. `HandleType::CUDA_IPC`
- **`handle`** — opaque handle bytes (e.g. CUDA IPC handle) for the subscriber to open via the backend
- **`reply_to_topic_name`** — control-channel topic so the publisher can receive ACQUIRED and send ACK/NACK
- **`size`** — buffer size (application payload)

**Not transmitted:** The device pointer itself. The subscriber obtains it by calling `acquire_pointer(descriptor)`; the IPC library runs the ACQUIRED/ACK protocol and opens the handle (e.g. `cudaIpcOpenMemHandle`) on the subscriber side.

## Building

This target is included when **IPC** and **Fast DDS** transport are enabled (`HOLOSCAN_BUILD_IPC`, `HOLOSCAN_IPC_TRANSPORT_FASTDDS`; IPC examples build with the IPC module—see `modules/holoipc/CMakeLists.txt` and `modules/holoipc/examples/CMakeLists.txt`).

From the Holoscan SDK build directory, enable IPC and Fast DDS transport (`HOLOSCAN_BUILD_IPC` defaults **OFF**; with IPC **ON**, `HOLOSCAN_IPC_TRANSPORT_FASTDDS` defaults **ON**):

```bash
cd build
cmake -DHOLOSCAN_BUILD_IPC=ON -DHOLOSCAN_IPC_TRANSPORT_FASTDDS=ON ..
cmake --build . --target fastdds_benchmark
```

## Running

Default **`-s` / `--samples`** is **0** (unlimited until **Ctrl+C**). For bounded runs, pass the same **`-s`**, **`-z`**, and **`-a`** (if used) on both sides. **`-z`** sets the subscriber’s receive buffer capacity and must be **≥** the publisher’s payload size or the subscriber may reject samples.

```bash
# Terminal 1 — subscriber first (recommended)
./fastdds_benchmark subscriber -s 10 -z 1024

# Terminal 2 — publisher
./fastdds_benchmark publisher -s 10 -z 1024
```

Accel path: add **`-a`** to both commands. See [README.md — Run the example](README.md#run-the-example) for more detail.
