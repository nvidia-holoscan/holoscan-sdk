# HoloMQ contributor reading order

Dependency-oriented map for **contributors and reviewers** — not the application-facing API
(see [`PUBLIC_API.md`](PUBLIC_API.md)). Read low-level types and OS primitives first, then
FD-pass, CUDA import/export, buffer pools and sync, transport, pub/sub, `Context`, and examples.

Paths are relative to `public/modules/holomq/`.

---

## Tier 1 — Foundations (almost no other holomq code)

| Order | File | Notes |
|------:|------|--------|
| 1 | `include/holomq/wait_strategy.hpp` | Wait modes and `WaitStrategy` used by subscriber/wait-set paths. |
| 2 | `include/holomq/limits.hpp` | Compile-time queue/history depths and `k_max_in_flight`. |
| 3 | `include/holomq/logger.hpp`, `src/logger.cpp` | HoloMQ logging; optional iceoryx2 bridge via `transport/iox2/log_bridge.*`. |
| 4 | `src/scoped_file_descriptor.hpp` | RAII for POSIX fds; FD-pass and `Context`. |
| 5 | `src/fd_pass/poll_set.hpp` | Thin `poll(2)` wrapper; FD-pass server. |
| 6 | `src/fd_pass/accept_cloexec.hpp` | `accept(2)` + `FD_CLOEXEC` helper; FD-pass server and tests. |
| 7 | `src/vmm_pool_export_registry.hpp`, `vmm_pool_export_registry.cpp` | Publisher-side fd-pass VMM export registry. |

---

## Tier 2 — Demo / domain payload (optional early read)

| Order | File | Notes |
|------:|------|--------|
| 8 | `examples/holomq_main/frame_payload.hpp` | Demo payload shape embedded in pub/sub messages. |

---

## Tier 3 — FD-pass wire format and addressing

| Order | File | Notes |
|------:|------|--------|
| 9 | `src/fd_pass/protocol.hpp` | Request/response layout and status codes. |
| 10 | `src/fd_pass/sockaddr.hpp` | Wire path field size, abstract vs path endpoint limits, and `fill_sockaddr_un`. |
| 11 | `src/fd_pass/sockaddr.cpp` | `fill_sockaddr_un` implementation. |

---

## Tier 4 — FD-pass transport (client then server)

| Order | File | Notes |
|------:|------|--------|
| 12 | `src/fd_pass/client.hpp` | Protocol + `ScopedFileDescriptor` + optional returns. |
| 13 | `src/fd_pass/client.cpp` | |
| 14 | `src/fd_pass/server.hpp` | Threaded server; handler type. |
| 15 | `src/fd_pass/server.cpp` | Uses `PollSet`, `accept_cloexec`, `sendmsg` / `recvmsg`, etc. |

---

## Tier 5 — CUDA driver helpers

| Order | File | Notes |
|------:|------|--------|
| 16 | `src/cuda/context.*` | CUDA driver context RAII. |
| 17 | `src/cuda/mem*.hpp`, `mem*.cpp`, `host_registration.*` | VMM reservation/mapping and host registration RAII. |
| 18 | `src/vmm_pool_import_cache.*` | Subscriber-side FD-pass import cache. |

---

## Tier 6 — Buffer pool (facade, shared layout, device, ref-count, sync, leases, host)

| Order | File | Notes |
|------:|------|--------|
| 19 | `include/holomq/buffer_pool.hpp`, `src/buffer_pool/buffer_pool.cpp` | `BufferPool` facade; pool metadata and `acquire`. |
| 20 | `src/buffer_pool/slot_region.hpp`, `pool_layout.hpp`, `payload_slot_array.hpp`, `generation_slot_array.*` | Slot stride/layout helpers and payload/generation slot views. |
| 21 | `include/holomq/buffer_memory_type.hpp`, `include/holomq/buffer_handle.hpp`, `src/buffer_pool/buffer_handle.cpp`, `tests/unit/test_buffer_handle.cpp` | `BufferMemoryType` enum; serializable IPC handle with hoisted `memory_type_`, `slot_index` / `ready_generation`, private wire union (`detail::device::BufferHandleData` / `detail::host::BufferHandleData`), `detail::BufferHandleWireAccess` for import paths, `make_*_buffer_handle` factories. |
| 22 | `include/holomq/detail/buffer_pool/device/handle_data.hpp`, `include/holomq/device_buffer_pool_memory.hpp` | Device wire layout (`detail::device`) and public VMM backing interface for custom allocators. |
| 23 | `include/holomq/detail/buffer_pool/host/handle_data.hpp`, `include/holomq/host_buffer_pool_memory.hpp` | Host wire layout (`detail::host`) and public SHM backing + attach metadata interface for custom allocators. |
| 24 | `include/holomq/device_buffer_allocator.hpp`, `src/buffer_pool/device/*` | `DeviceBufferPoolImpl`, `DeviceBufferImpl`, `DeviceConstBufferImpl`, `VmmExportedPool`, default VMM allocator. Read `vmm_exported_pool.*` before host split-gen VMM. |
| 25 | `src/buffer_pool/ref_count/*` | POSIX SHM ref-count table and segment cache (`ref_count_key` extracts identity from device/host wire). |
| 26 | `src/buffer_pool/buffer_pool_impl.hpp` | Shared acquire loop and `buffer_handle_released()` dispatch. |
| 27 | `src/buffer_pool/sync/*` | Stream generation and deferred ref release. |
| 28 | `include/holomq/buffer.hpp`, `const_buffer.hpp`, `src/buffer_pool/buffer.*`, `const_buffer.*`, `buffer_impl.*`, `const_buffer_impl.*`, `buffer_pool_impl.cpp` | Move-only owner/subscriber lease facades (`unique_ptr` pimpl) and shared `Impl` sync entry points. |
| 29 | `include/holomq/host_buffer_allocator.hpp`, `include/holomq/host_buffer_import.hpp`, `src/buffer_pool/host/*` | `HostBufferPoolImpl`, views, SHM backing cache, default host allocator, and the reusable exact-slot host import. Depends on tier 24 `vmm_exported_pool.*` for split-gen VMM. |

---

## Tier 7 — Transport abstraction (iceoryx2)

| Order | File | Notes |
|------:|------|--------|
| 30 | `include/holomq/detail/transport/backend.hpp`, `validate_backend.hpp`, `concepts.hpp`, `wait_set_backend.hpp` | iceoryx2 backend wiring; C++20 concepts + `consteval validate_backend()`. |
| 31 | `include/holomq/detail/transport/iox2/*` | Backend ports, runtime, loans, inbound messages. |
| 32 | `src/transport/iox2/*` | Backend `.cpp` glue (`runtime`, `wait_set_backend`, `log_bridge.*`). |

See `docs/design-notes/transport-abstraction.md`. Sync and pool depth: `docs/design-notes/README.md`.

---

## Tier 8 — Pub/sub ports and sample release

| Order | File | Notes |
|------:|------|--------|
| 33 | `include/holomq/outbound_message.hpp`, `inbound_message.hpp` | Application-facing loan/take wrappers. |
| 34 | `include/holomq/publisher.hpp`, `subscriber.hpp` | `loan`/`send` and `try_take`/`wait_and_take` ports. |
| 35 | `include/holomq/detail/publisher_sample_release_registry.hpp`, `src/publisher_sample_release_registry.cpp` | Sample-release callbacks keyed by payload address. |
| 36 | `include/holomq/wait_set_handle.hpp`, `wait_set.hpp`, `src/wait_set.cpp` | Multiplexed wait over subscriber ports. |

---

## Tier 9 — Context and builders

| Order | File | Notes |
|------:|------|--------|
| 37 | `include/holomq/context_builder.hpp`, `src/context_builder.cpp` | Fluent `Context` construction. |
| 38 | `include/holomq/context.hpp`, `src/context.cpp`, `src/context_impl.hpp`, `src/context_impl.cpp` | Transport runtime, FD-pass server, pool factories, import/backing caches. |
| 39 | `include/holomq/publisher_builder.hpp`, `subscriber_builder.hpp` | Topic + payload type factories. |

**Header note:** `context_builder.hpp` includes `context.hpp`, and `context.hpp` pulls in builders. Read tiers 8–9 as one block.

---

## Tier 10 — Integration adapter (Holoscan SDK only)

| Order | File | Notes |
|------:|------|--------|
| 40 | `include/holomq/adapters/holoscan_logger.hpp` | Optional redirect of HoloMQ logs to `holoscan::Logger`; requires Holoscan Core at compile time. |

---

## Tier 11 — Examples and unit tests

| Order | File | Notes |
|------:|------|--------|
| 41 | `examples/common/holoargs.hpp`, `holoargs.cpp` | Arg parsing (`argp`); shared by demos. |
| 42 | `examples/holomq_main/main.cpp` | End-to-end pub/sub + buffer pool demo. |
| 43 | `tests/unit/test_scoped_file_descriptor.cpp` | RAII fd helper. |
| 44 | `tests/unit/test_poll_set.cpp` | FD-pass poll wrapper. |
| 45 | `tests/unit/test_fd_pass_sockaddr.cpp` | Abstract vs path sockaddr fill. |
| 46 | `tests/unit/test_fd_pass.cpp` | FD-pass round trip over AF_UNIX. |
| 47 | `tests/unit/test_vmm_pool_export_registry.cpp` | Publisher export registry. |
| 48 | `tests/unit/test_slot_region.cpp` | Slot region / payload slot array layout. |
| 49 | `tests/unit/test_logger.cpp` | Logging handler and levels. |
| 50 | `tests/unit/test_cuda_pool.cpp` | Device buffer pool acquire/release and slot reuse. |
| 51 | `tests/unit/test_cuda_vmm_import.cpp` | Device buffer export, fd_pass, and import path. |
| 52 | `tests/unit/test_cuda_buffer_import_sync.cpp` | Cross-process device import + generation sync. |
| 53 | `tests/unit/test_cuda_host_pool.cpp`, `cuda_host_import_child.cpp` | Host pool export/import (multi-process). |
| 54 | `tests/unit/test_wait_set.cpp` | Wait-set multiplexing over pub/sub. |
| 55 | `tests/unit/test_publisher_sample_release_registry.cpp` | Sample-release register / dispatch / cancel. |

---

## CMake (build flags)

Read `CMakeLists.txt` and `examples/*/CMakeLists.txt` **after** FD-pass / sockaddr code so `HOLOMQ_TRANSPORT`, `HOLOMQ_FD_PASS_SOCKET`, and socket mode match the sources. Before adding or exporting public symbols, read [`abi_and_exports.md`](abi_and_exports.md).

In-tree examples link `holomq::holomq` and use the public `<holomq/...>` API only.

---

## Short path (time-boxed)

`wait_strategy.hpp` → `scoped_file_descriptor.hpp` → `fd_pass/protocol.hpp` → FD-pass client/server → `src/cuda/*` + `vmm_pool_import_cache.*` → `buffer_pool` facade → shared slot layout + `buffer_handle` → `device/*` → `ref_count/*` + `buffer_pool_impl.hpp` → `sync/*` → `buffer` / `const_buffer` impls → `host/*` → `detail/transport/backend.hpp` + iox2 runtime → `publisher` / `subscriber` / `wait_set` → builders → `context` (+ `context_builder`) → `examples/holomq_main/main.cpp`.
