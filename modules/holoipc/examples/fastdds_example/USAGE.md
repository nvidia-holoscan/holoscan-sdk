# CudaBuffer - DDS Type with Pointer Descriptor

## Overview

This example uses a Fast DDS type **CudaBuffer** defined in `idl/CudaBuffer.idl` and generated into `gen/`. The generated type has:

- **`descriptor`** — a `PointerDescriptor` (from holoscan::ipc) with `key`, `handle_type`, `handle`, and `reply_to_topic_name`
- **`size`** — buffer size (`unsigned long`)

The device pointer is **not** transmitted. The publisher calls `Context::share_pointer(cuda_ptr, HandleType::CUDA_IPC)` to get a descriptor and publishes it inside `CudaBuffer`. The subscriber receives the sample, then calls `Context::acquire_pointer(descriptor)` to get a `Future<std::shared_ptr<void>>` and uses `.get()` to obtain the device pointer (ACQUIRED/ACK/RELEASED protocol). See [README](README.md) and the source (`PublisherApp.cpp`, `ListenerSubscriberApp.cpp`, `WaitsetSubscriberApp.cpp`) for the actual flow.

## Files (this example)

- **`idl/CudaBuffer.idl`** — defines `CudaBuffer` (includes `pointer_descriptor.idl`, has `descriptor` + `size`)
- **`gen/`** — IDL-generated C++ types (e.g. `CudaBufferPubSubTypes.hpp`, `CudaBuffer.hpp`)
- **Publisher:** `PublisherApp.cpp`, `CudaDataWriterListener.cpp` — allocate CUDA memory, `share_pointer()`, write `CudaBuffer` samples
- **Subscriber:** `ListenerSubscriberApp.cpp` or `WaitsetSubscriberApp.cpp` — receive samples, `acquire_pointer(descriptor).get()`, use pointer, then release
- **Application, CLI, main:** `Application.hpp`/`.cpp`, `CLIParser.hpp`, `main.cpp`

## What gets transmitted

**Over the wire (in `CudaBuffer.descriptor` and `CudaBuffer.size`):**

- **`key`** — opaque identifier for the pointer (sequence of bytes)
- **`handle_type`** — e.g. `HandleType::CUDA_IPC`
- **`handle`** — opaque handle bytes (e.g. CUDA IPC handle) for the subscriber to open via the backend
- **`reply_to_topic_name`** — control-channel topic so the publisher can receive ACQUIRED and send ACK/NACK
- **`size`** — buffer size (application payload)

**Not transmitted:** The device pointer itself. The subscriber obtains it by calling `acquire_pointer(descriptor)`; the IPC library runs the ACQUIRED/ACK protocol and opens the handle (e.g. `cudaIpcOpenMemHandle`) on the subscriber side.

## Building

IPC examples are built when the IPC module and Fast DDS transport are enabled (`HOLOSCAN_BUILD_IPC`, `HOLOSCAN_IPC_TRANSPORT_FASTDDS`). `HOLOSCAN_BUILD_IPC` defaults **OFF**; when IPC is enabled, `HOLOSCAN_IPC_TRANSPORT_FASTDDS` defaults **ON**. From the Holoscan SDK build directory:

```bash
cd build
cmake -DHOLOSCAN_BUILD_IPC=ON -DHOLOSCAN_IPC_TRANSPORT_FASTDDS=ON ..
cmake --build . --target fastdds_example
```

## Running

```bash
# Terminal 1 - Publisher
./fastdds_example publisher

# Terminal 2 - Subscriber
./fastdds_example subscriber
```

For a wait-set based subscriber:

```bash
./fastdds_example subscriber --waitset
```
