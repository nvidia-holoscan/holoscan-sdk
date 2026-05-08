# Holoscan IPC

This component provides **zero-copy GPU (or other device) memory sharing** between processes. With the **Fast DDS** transport (optional at CMake configure; default **ON** in the public SDK), it follows the usual DDS publish–subscribe pattern: publishers send a **pointer descriptor** on the data topic; subscribers receive it and **acquire** a local reference to the same memory (e.g. via CUDA IPC) without copying the payload.

The public API is **`holoscan::ipc::Context<TransportType>`** plus **`make_context<TransportType>(...)`** (see [`context.hpp`](include/holoscan/ipc/context.hpp)). For the Fast DDS transport, use e.g. **`make_context<holoscan::ipc::transport::fastdds::FastDdsTransport>(domain_participant)`**. Then use **`share_pointer(ptr, handle_type)`** on the publisher side to obtain a descriptor to publish, and **`acquire_pointer(descriptor)`** on the subscriber side to get a `Future<std::shared_ptr<void>>` that completes with the device pointer after **ACK** (the open runs on the thread that waits on the future; if validation fails, the future completes immediately and `get()` returns an empty `shared_ptr` without blocking). For lower latency when your deployment can accept the documented risk window, **`acquire_pointer_eager`** opens the handle locally, sends **ACQUIRED**, and returns `std::shared_ptr<void>` without waiting for **ACK**; **NACK** after an eager open terminates the process (`std::terminate`). See [docs/EAGER_ACQUIRE.md](docs/EAGER_ACQUIRE.md). Lifecycle is coordinated over a **control channel** (ACQUIRED, RELEASED, ACK, NACK) so the publisher does not deallocate while any subscriber holds a reference.

## Documentation

Detailed documentation lives in the [docs](docs/) folder:

| Document | Description |
|----------|--------------|
| [docs/PROTOCOL_SPEC.md](docs/PROTOCOL_SPEC.md) | Wire format and semantics of the protocol (PointerDescriptor, ControlMessage, lifecycle). **Status: Draft.** |
| [docs/IDL_TYPES.md](docs/IDL_TYPES.md) | IDL types used by the IPC library, code generation from IDL, and where generated code lives. |
| [docs/IpcCore_internals.md](docs/IpcCore_internals.md) | Internal data structures in IpcCore: maps (e.g. `pointer_descriptors_`, `control_writers_`, `subscribers_`, `publishers_`) and when entries are added or removed. |
| [docs/EAGER_ACQUIRE.md](docs/EAGER_ACQUIRE.md) | Eager acquire API (`acquire_pointer_eager`): open before ACK, `std::terminate` on NACK. |

## Building and using

The **`holo_ipc`** target requires the **CUDA Toolkit** (CUDA IPC backend). The **Fast DDS** transport and the `examples/` apps are built when **`HOLOSCAN_IPC_TRANSPORT_FASTDDS`** is **ON** (default in [`public/CMakeLists.txt`](../../CMakeLists.txt)). See the top-level Holoscan SDK build instructions. For a publisher/subscriber sample with GPU buffers, see [fastdds_example](examples/fastdds_example/README.md). For latency measurements and optional eager-acquire comparison, see [fastdds_benchmark](examples/fastdds_benchmark/README.md).
