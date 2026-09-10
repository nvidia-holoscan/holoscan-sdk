# CUDA VMM stream memory operation safety

Documents the assumptions under which holomq may use CUDA stream memory operations on VMM-mapped memory.

**Related code:**

- `src/buffer_pool/sync/cuda_stream_generation.cpp` — `cuStreamWriteValue64_v2` / `cuStreamWaitValue64_v2`
- `src/buffer_pool/device/device_buffer_impl.cpp` — producer-side `record_ready`
- `src/buffer_pool/device/device_const_buffer_impl.cpp` — consumer-side `wait_ready`
- `src/buffer_pool/device/device_buffer_pool_impl.cpp` — VMM pool allocation/export

**See also:** `buffer-pool-slot-generation-sync.md`, `buffer-stream-associated-export.md`, `cuda-vmm-pooling-and-iceoryx2-memory-model.md`

---

## Summary

holomq VMM stream memops are safe for a **same-GPU** producer/consumer flow (for any device
ordinal N, not only GPU0):

```text
Producer process on GPU N creates VMM memory on GPU N
Consumer process imports/maps the same VMM allocation on GPU N
  (the source device ordinal travels in the BufferHandle, is checked against the imported
   allocation's properties, and the import grants access on that validated GPU N)
Producer enqueues WriteValue before publishing the handle
Consumer enqueues WaitValue on a GPU N stream only after receiving that handle
```

Under those constraints, the waited address is considered **local** to the waiting GPU, not a peer location. NVIDIA guidance clarified that a peer location is memory reached over NVLink or a peer PCIe path; a cross-process import of GPU N memory into another GPU N context is still local.

**Minimum NVIDIA driver:** **555** (r555) or newer. VMM-backed `WaitValue` / `WriteValue` on `cuMemMap` memory is not supported on r535 and earlier (`CUDA_ERROR_INVALID_VALUE`). holomq has been validated on driver `580.159.03`.

---

## Required constraints

### 1. Driver support

| Constraint | Requirement |
|------------|-------------|
| Minimum driver version | **555** (r555) or newer |
| Known unsupported | r535 and earlier for VMM + stream memops |
| Validated in holomq | `580.159.03` (RTX 6000 Ada) |

NVIDIA (CUDA user-mode driver team) indicated VMM pointer support for stream memops arrived around r555. On r535, `cuStreamWaitValue64_v2` / `cuStreamWriteValue64_v2` on VMM-mapped addresses return `CUDA_ERROR_INVALID_VALUE`, while the same operations succeed on `cuMemAlloc` memory.

Check the installed driver with:

```bash
nvidia-smi --query-gpu=driver_version --format=csv,noheader
```

The implementation also requires these CUDA driver APIs at build/link time:

- `cuStreamWriteValue64_v2`
- `cuStreamWaitValue64_v2`

### 2. Signal-before-wait submission

Do not use `WaitValue` as a pure "park this stream until a future producer eventually signals" primitive.

CUDA stream memory operations are not scheduler-visible dependencies in the same way as CUDA events. The safe ordering is:

```text
producer submits WriteValue
producer publishes handle containing the target generation
consumer receives handle
consumer submits WaitValue
```

This is the holomq data path: `Buffer::export_handle(stream)` enqueues `WriteValue` before returning the `BufferHandle`, and `Context::import_buffer(handle, stream)` enqueues `WaitValue` only after that handle is received.

### 3. Wait only on local memory

`WaitValue` is implemented by GPU ESCHED hardware. ESCHED cannot safely contain some post-translation failures on NVLink accesses. Therefore, do not wait on an address that may resolve to memory behind NVLink, peer PCIe, or other remote/fabric paths.

For cross-GPU or fabric topologies, use a different protocol, for example:

- producer signals a consumer-local notification word via `cudaMemcpyAsync` / batched copy, then the consumer waits locally;
- CUDA events or external synchronization with an explicit lifetime protocol;
- CPU-side robust synchronization before enqueueing GPU work.

---

## Supported topology

holomq stream memops on VMM-backed pools are intended for same-GPU pub/sub where:

- the host runs NVIDIA driver **555+** (see [Driver support](#1-driver-support));
- the VMM allocation is created with `CU_MEM_LOCATION_TYPE_DEVICE` for the same GPU used by subscribers;
- all subscribers validate the imported allocation location against the source ordinal;
- every readiness `WaitValue`, including the lazy host-readiness wait, uses a stream on that same
  GPU;
- the producer publishes a handle only after `WriteValue` has been submitted;
- the consumer waits only on the generation in the received handle.

It is **not** a general cross-GPU synchronization primitive.

---

## Remaining caveats

- Producer crash after publishing a handle but before the GPU reaches the submitted `WriteValue` can still leave consumers waiting indefinitely. Process liveness and cleanup remain a control-plane responsibility.
- Per-slot generations should be monotonic for the lifetime of the pool. Resetting the CPU-side generation while the GPU generation word still contains a larger old value can make `CU_STREAM_WAIT_VALUE_GEQ` complete on stale state.
- Runtime checks or configuration should prevent this path from being used when publisher and subscriber are on different GPUs or when the allocation may be physically remote from the waiting GPU.
