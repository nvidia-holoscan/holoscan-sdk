# HoloMQ buffer pools vs Holoscan `MemoryStorageType`

This note compares HoloMQ’s three `Context` buffer-pool factories to the four memory
storage kinds used by Holoscan Core allocators (`BlockMemoryPool`, `UnboundedAllocator`,
etc.). It also explains why there is no HoloMQ equivalent of Holoscan’s managed-memory
type for cross-process zero-copy IPC.

**Holoscan reference:** `holoscan::MemoryStorageType` in
[`include/holoscan/core/resource.hpp`](https://github.com/nvidia-holoscan/holoscan-sdk/blob/main/include/holoscan/core/resource.hpp)
(upstream main; not present in the HSDK 5 tree in this repository).

**HoloMQ reference:** `Context` in `context.hpp`, wire format in `detail/buffer_pool/host/handle_data.hpp`.

---

## HoloMQ: three buffer-pool factories

All three factories return an external `BufferPool` whose slots are exported as
`BufferHandle` descriptors in publish–subscribe payloads. Subscribers adopt the same
backing store via `Context::import_buffer` (zero-copy on one host; no bulk copy over
the messaging layer).

| Factory | Backing | CUDA Driver API (publisher) | Cross-process attach |
|---------|---------|----------------------------|----------------------|
| `create_device_buffer_pool` | CUDA VMM **pinned device** memory | `cuMemCreate` / `cuMemMap` / `cuMemExportToShareableHandle` | FD-pass → `cuMemImportFromShareableHandle` |
| `create_host_buffer_pool` | POSIX SHM payload (**pageable**) | None on payload (mmap only) | SHM name + split-gen VMM pool via FD-pass |
| `create_host_pinned_buffer_pool` | POSIX SHM payload (**pinned**) | `cuMemHostRegister(payload, cuda_host_register_flags)` | SHM name; optional gen-pool FD-pass (see flags) |

Common parameters: `buffer_bytes` (slot payload size), `buffer_count` (number of slots),
and an optional custom allocator (`DeviceBufferAllocator` or `HostBufferAllocator`; default
`{}` selects the built-in SHM / VMM allocators).

**Publisher setup:** device pools and host **split-gen VMM** pools (pageable host, or pinned host
without `DEVICEMAP`) require `ContextBuilder::with_vmm_fd_export(true)` so the internal
AF_UNIX FD-pass server can export VMM generation pools. Pinned host **devicemap interleaved**
(`CU_MEMHOSTREGISTER_DEVICEMAP`) does not require fd-pass for the payload itself.

---

## `cuda_host_register_flags` (pinned host only)

Host pools use two **generation-sync layouts**:

| Layout | Selected by | Generation words | Publisher fd-pass |
|--------|-------------|------------------|-------------------|
| **Devicemap interleaved** | `CU_MEMHOSTREGISTER_DEVICEMAP` on payload SHM | In the payload SHM mapping (GPU via `cuMemHostGetDevicePointer`) | Not required for payload |
| **Split-gen VMM** | `DEVICEMAP` clear (incl. pageable `create_host_buffer_pool`) | Separate publisher VMM pool, imported via fd-pass | Required for gen pool |

`cuda_host_register_flags` is the third parameter of
`create_host_pinned_buffer_pool`. It is passed verbatim to `cuMemHostRegister` on the
payload SHM in both publisher and subscriber processes. The value **`0`** is valid (CUDA
default register). It is **not** used by `create_host_buffer_pool` (pageable pools never
register payload SHM).

The flags are also stored on `HostBufferHandleData::host_register_flags` when
`payload_pinned != 0`. Subscribers use `payload_pinned` to decide whether to call
`cuMemHostRegister` on attach; for pageable handles, `host_register_flags` is ignored.

Common flags (CUDA Driver API):

| Flag | Effect |
|------|--------|
| `0` | Default pinned registration (split-gen VMM layout; fd-pass required) |
| `CU_MEMHOSTREGISTER_PORTABLE` | Registration valid across CUDA contexts / devices in the process |
| `CU_MEMHOSTREGISTER_DEVICEMAP` | Map host pages into GPU address space; generation words can live in the same SHM mapping (**devicemap interleaved**, interleaved layout) |
| `CU_MEMHOSTREGISTER_IOMEMORY` | I/O memory (platform-specific; not the usual HoloMQ path) |

**Layout split (pinned only):**

- **`DEVICEMAP` set (devicemap interleaved):** payload SHM holds interleaved GPU generation words + user
  bytes; GPU uses device pointers from the same mapping. No separate gen-pool fd-pass for
  generation sync.
- **`DEVICEMAP` clear (split-gen VMM):** payload SHM is payload-only; generation words live in a
  separate publisher VMM pool imported via fd-pass (same pattern as device pools). Requires
  `with_vmm_fd_export(true)` on the publisher.

Pageable `create_host_buffer_pool` always uses split-gen VMM (split generation + fd-pass); there
is no `cuMemHostRegister` and no devicemap interleaved variant for pageable SHM.

---

## `BufferMemoryType` on handles and views

`BufferMemoryType` (`include/holomq/buffer_memory_type.hpp`) is the public discriminant on
`BufferHandle`, `BufferPool`, `Buffer`, and `ConstBuffer`:

| Value | Factory / wire source |
|-------|------------------------|
| `Device` | `create_device_buffer_pool` |
| `HostPageable` | `create_host_buffer_pool` (`payload_pinned == 0` on wire) |
| `HostPinned` | `create_host_pinned_buffer_pool` (`payload_pinned != 0` on wire) |

`BufferHandle` hoists `memory_type_` separately from the wire union. Host wire still carries
`payload_pinned`; pinned handles encode `HostPinned` (3), not the older coarse `Host` (2).
Processes must run matching HoloMQ versions when exchanging whole `BufferHandle` structs
over IPC. Pageable handles remain value `2` (`HostPageable`).

`cuda_device_ordinal()` is populated for device pools and imported device views; host pools
and host import views return an empty **`std::optional<int32_t>`** (not an error).

---

## Comparison to Holoscan `MemoryStorageType`

Holoscan’s enum (values 0–2 map to GXF `nvidia::gxf::MemoryStorageType`):

```cpp
enum struct MemoryStorageType {
  kHost = 0,        // page-locked host (e.g. cudaMallocHost)
  kDevice = 1,      // GPU device memory (e.g. cudaMalloc)
  kSystem = 2,      // ordinary host heap (e.g. C++ new)
  kCudaManaged = 3  // unified / managed memory (cudaMallocManaged)
};
```

Holoscan allocators are **in-process**: one application owns the allocation for its
lifetime. HoloMQ pools are **cross-process**: backing is created once, identified by
handles, and attached in peer processes without copying the payload over iceoryx2.

| Holoscan type | Typical allocation | Closest HoloMQ factory | Important differences |
|---------------|-------------------|-------------------------|------------------------|
| **`kHost`** | `cudaMallocHost` / pinned host pool | **`create_host_pinned_buffer_pool`** | HoloMQ uses **POSIX SHM + `cuMemHostRegister`**, not a private pinned heap. Flags choose devicemap interleaved vs split-gen VMM and portability. IPC metadata (`BufferHandle`) is extra. |
| **`kDevice`** | `cudaMalloc` / device pool | **`create_device_buffer_pool`** | HoloMQ uses **CUDA VMM** (`cuMemCreate` pinned device type) and **fd-pass export**, not legacy `cuMemAlloc` + `cuIpcGetMemHandle`. Same role: GPU-resident bulk data. |
| **`kSystem`** | C++ `new` / pageable heap | **`create_host_buffer_pool`** | HoloMQ uses **pageable SHM** (shared mmap), not process-local heap. Conceptually “ordinary host memory,” but **explicitly shared** for zero-copy IPC. GPU access is not UVM; depends on pageable access or explicit copies. |
| **`kCudaManaged`** | `cudaMallocManaged` / UVM | **No equivalent** | See [Why managed memory is not supported](#why-managed-memory-is-not-supported-for-holomq). |

**Unified CPU+GPU access without UVM:** HoloMQ **devicemap interleaved** (`create_host_pinned_buffer_pool`
with `CU_MEMHOSTREGISTER_DEVICEMAP`) is the closest *behavior* to “one backing, CPU and GPU
both use it”—but it is **pinned SHM + explicit device mapping**, not the UVM driver managing
migration and residency.

---

## Why managed memory is not supported for HoloMQ

Holoscan **`kCudaManaged`** (`cudaMallocManaged` / `cuMemAllocManaged`) is designed for
**single-process unified virtual addressing**: the CUDA UVM driver tracks access, migrates
pages between host and device, may evict to host under pressure, and honors `cuMemAdvise` /
`cuMemPrefetchAsync`. That model assumes **one process’s UVM state** owns the allocation.

Cross-process zero-copy IPC requires a **stable, shareable backing** and **explicit import**
in each process. CUDA’s documented IPC paths do not include managed allocations.

### CUDA does not export managed memory for IPC

| Mechanism | Driver / runtime API | Managed (`cuMemAllocManaged`) |
|-----------|---------------------|-------------------------------|
| Legacy CUDA IPC | `cuIpcGetMemHandle` / `cuIpcOpenMemHandle` | **Not supported** (device `cuMemAlloc` only; Programming Guide excludes `cudaMallocManaged`) |
| VMM shareable handles | `cuMemExportToShareableHandle` / import | **Not supported** for managed type; HoloMQ uses `CU_MEM_ALLOCATION_TYPE_PINNED` device memory |
| VMM host allocations | `cuMemCreate` with `CU_MEM_LOCATION_TYPE_HOST` | **Not IPC-capable** per Driver API |
| Pointer query | `CU_POINTER_ATTRIBUTE_IS_LEGACY_CUDA_IPC_CAPABLE` | Managed pointers are not IPC-capable |

There is no supported way to pass a “managed memory handle” through iceoryx2 (or any OS IPC)
and have the subscriber receive the **same UVM allocation** with correct driver state.

### Why sharing managed memory between processes does not fit the model

1. **UVM state is per-process.** Migration, residency, faults, and advice are tracked in
   the importing process’s CUDA context. A second process cannot attach to another process’s
   managed allocation as a first-class UVM object; the driver has no IPC import path for it.

2. **Handles describe physical backing, not UVM policy.** HoloMQ shares **pages or VMM
   physical allocations** (SHM mmap, exported VMM fd). Managed memory’s semantics are
   “driver-owned placement and migration,” which conflicts with “both processes map the same
   fd/shm and agree on generation sync.”

3. **Documented cross-process “unified” pattern is different.** The CUDA Programming Guide
   describes **IPC-capable system-allocated memory** (`mmap MAP_SHARED`, POSIX SHM,
   `memfd_create`) that can be used with a unified-memory-*style* programming model after
   **OS-level sharing**—not export of `cuMemAllocManaged`. That path is what
   **`create_host_buffer_pool`** (and optionally pinned + `DEVICEMAP`) implements: shared
   host pages, not shared UVM heap objects.

4. **No stable cross-process zero-copy for “automatic” migration.** If each process used its
   own `cuMemAllocManaged` and synced via messages, you get **copies or duplicate
   allocations**, not the single backing store HoloMQ targets. Oversubscription and
   migration heuristics also assume one address space’s access pattern, not independent
   publisher/subscriber processes with different GPUs or contexts.

5. **Performance and correctness.** NVIDIA notes that IPC with unified-memory-style use of
   shared system memory can have **significant performance implications**. Adding true managed
   semantics on top of cross-process SHM would combine the worst of both: IPC page-sharing
   costs plus undefined UVM behavior without an official API.

### Practical takeaway

For HoloMQ’s goal—**zero-copy bulk data across processes on one machine**—the supported
design is:

- **Device bulk:** VMM pinned device memory + fd-pass (`create_device_buffer_pool`).
- **Host bulk (pageable):** POSIX SHM + mmap (`create_host_buffer_pool`).
- **Host bulk (pinned / GPU-mapped):** SHM + `cuMemHostRegister` flags
  (`create_host_pinned_buffer_pool`).

A **`create_managed_buffer_pool`** would not align with CUDA’s IPC capabilities and would
not provide the same guarantees as Holoscan’s in-process `kCudaManaged` allocator. Applications
that need UVM within one process should use Holoscan managed allocators locally; applications
that need cross-process zero-copy should use the three HoloMQ factories above and embed
`BufferHandle` in their payloads.
