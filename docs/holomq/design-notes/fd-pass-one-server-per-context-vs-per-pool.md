# FD-pass: VMM pool export over AF_UNIX

How holomq exports CUDA VMM pool file descriptors to remote processes so subscribers can
import GPU memory. The publisher runs **one FD-pass server per `Context`**; multiple device or
host generation pools share that endpoint and are dispatched by **`pool_cookie`**.

**Related code:**

- `src/fd_pass/` — `Server`, `Client`, `protocol.hpp`, `sockaddr.hpp`
- `src/context_builder.cpp`, `src/context_impl.hpp` — optional `fd_pass::Server` on `Context`
- `src/vmm_pool_export_registry.hpp`, `vmm_pool_export_registry.cpp` — `pool_cookie` → export dup callback
- `src/buffer_pool/device/device_buffer_pool_impl.*` — device VMM export + registry registration
- `src/buffer_pool/host/host_buffer_pool_impl.*` — split-gen host pools (generation VMM)
- `src/vmm_pool_import_cache.cpp` — subscriber import keyed by `(pool_cookie, fd_pass_socket_path)`
- `tests/unit/test_fd_pass*.cpp`, `test_cuda_vmm_import.cpp`

**See also:** [`multi-publisher-shm-and-gpu-pool-accounting.md`](multi-publisher-shm-and-gpu-pool-accounting.md),
[`vmm-pool-import-cache-entry-eviction.md`](vmm-pool-import-cache-entry-eviction.md),
[`host-buffer-pools.md`](host-buffer-pools.md)

---

## What FD-pass does

CUDA VMM allocations are represented as a POSIX file descriptor in the publisher process.
Remote processes cannot use that FD until it is duplicated into their address space.
holomq passes the FD over **AF_UNIX** with **SCM_RIGHTS** using a small request/response
protocol (`fd_pass/protocol.hpp`).

Subscribers do not connect to FD-pass directly from the public API. They read
`fd_pass_socket_path` and `pool_cookie` from a received **`BufferHandle`** (embedded in the
message payload). `VmmPoolImportCache::import_pool_base` opens the socket, sends
`fd_pass::Request{pool_cookie}`, receives the dup’d FD, and maps the pool once per distinct
`(pool_cookie, fd_pass_socket_path)` pair. Cookies alone are insufficient: they are opaque
non-secrets scoped to one publisher FD-pass server, and a hostile peer could otherwise race a
first import under a victim cookie via its own socket.

---

## Publisher layout (per-`Context` server)

```text
Context
├── configured_fd_pass_path     (Context::fd_pass_socket_path())
├── fd_pass::Server             (one worker thread, one listen socket)
├── VmmPoolExportRegistry       (pool_cookie → dup export FD)
└── BufferPool × N
    ├── DeviceBufferPoolImpl    (device VMM)
    └── HostBufferPoolImpl      (split-gen generation VMM when required)

Each pool at create:
  - exports VMM → registers dup callback in VmmPoolExportRegistry
  - copies Context fd_pass_socket_path into handle wire data
  - assigns unique pool_cookie in handle wire data
```

Enable on the publisher with `ContextBuilder::with_vmm_fd_export(true)`. Optional
`with_fd_pass_socket(path)` sets the listen path; empty path → default under `/tmp` at
`create()`. Subscriber contexts typically use `with_vmm_fd_export(false)`; they learn the path
from handles in received payloads.

When a client connects, the server handler calls
`VmmPoolExportRegistry::dup_exported_fd(pool_cookie)` and passes the result over SCM_RIGHTS.

`ContextImpl` destroys **`VmmPoolExportRegistry` before `fd_pass::Server`** so the worker stops
before registry entries are torn down. Pools unregister on destroy during normal teardown.

---

## End-to-end flow

```text
Publisher                              Subscriber
─────────                              ──────────
create_device_buffer_pool()
  → VmmExportedPool + register(pool_cookie)
  → fd_pass_socket_path in Context

export_handle / send payload
  → BufferHandle carries pool_cookie + fd_pass_socket_path

                                       import_buffer(handle)
                                         → ref-count SHM map
                                         → import_pool_base(descriptor)
                                            → fd_pass Client(pool_cookie)
                                            → VmmPoolImportCache entry
                                         → map slot, generation sync
```

Multiple pools in one publisher process share the **same** `fd_pass_socket_path` on the wire;
`pool_cookie` selects which VMM export FD the server duplicates.

---

## Wire fields

`detail::device::BufferHandleData` and split-gen host handles carry:

| Field | Role |
|-------|------|
| `fd_pass_socket_path` | AF_UNIX endpoint (max length in `fd_pass_socket_path_max`) |
| `pool_cookie` | Opaque id; FD-pass dispatch + half of the `VmmPoolImportCache` key (with socket path) |
| `slot_stride_bytes`, `buffer_bytes`, … | Import geometry (see device/host handle docs) |

`fd_pass::Request` sends `pool_cookie`. Server replies `Status::ok` + FD, or
`pool_unavailable` / `pool_mismatch` on failure.

`BufferHandle` also carries export fence `ready_generation` outside the wire POD union.

---

## Per-pool FD-pass server (comparison)

An alternative gives **each** `BufferPool` its own `fd_pass::Server`, listen path, and
worker thread. The handler would dup that pool's export FD directly — no
`VmmPoolExportRegistry`.

| | Per-`Context` | Per-pool server |
|--|-------------------------|-----------------|
| FD-pass threads | 1 | One per pool |
| Listen sockets | 1 | One per pool |
| Dispatch | `pool_cookie` → registry | Socket path identifies pool |
| `Context` setup | `with_vmm_fd_export`, shared path | Path generated per pool |

holomq uses **per-`Context`** so a process with many GPU pools pays one thread and one socket.
Pools stay focused on memory and export registration; IPC endpoint configuration stays on
`Context`. Handle wire format is the same either way (`fd_pass_socket_path` + `pool_cookie`).

A hybrid multiplexer (`poll` on several listen FDs in one thread) would reduce threads while
keeping per-pool paths — an alternative to the single-server design.

---

## When FD-pass is required

| Pool kind | Publisher `with_vmm_fd_export(true)` |
|-----------|--------------------------------------|
| Device VMM | **Required** |
| Host pageable (split-gen) | **Required** |
| Host pinned without `CU_MEMHOSTREGISTER_DEVICEMAP` | **Required** (generation VMM) |
| Host pinned with `DEVICEMAP` | **Not required**; when enabled, the pool uses the generation VMM, otherwise it falls back to interleaved generation words in payload SHM |

See [`host-buffer-pools.md`](host-buffer-pools.md) and [`PUBLIC_API.md`](../PUBLIC_API.md).

---

## Standalone and tests

`cuda_vmm_import_test` and some unit tests construct a standalone `fd_pass::Server` plus
`VmmPoolExportRegistry` without a full `Context`, using the same protocol and cookie lookup.

Socket mode (filesystem path vs Linux abstract name) is a **build-time** choice; see
`CONTRIBUTOR_READING_ORDER.md` and `fd_pass/sockaddr.hpp`.

---

## Subscriber import cache

`VmmPoolImportCache` retains one import entry per distinct
`(pool_cookie, fd_pass_socket_path)` for the lifetime of the `Context`. Destroying a
`ConstBuffer` does not evict the entry — repeated samples from the same publisher avoid
re-import cost. See
[`vmm-pool-import-cache-entry-eviction.md`](vmm-pool-import-cache-entry-eviction.md).
