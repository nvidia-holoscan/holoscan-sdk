# holomq public API

## Introduction

This document describes the **application-facing C++ API** of holomq as it exists today. It is intended as the basis for a design review: focus is on **names, types, ownership, and call patterns**, not implementation internals.

**Status:** draft — API shape under review; behavior may change.

---

## Overview

holomq is a C++ library for **zero-copy GPU IPC** over a publish–subscribe messaging layer. Applications:

1. Open a **`Context`** (transport runtime + optional CUDA VMM FD-export server).
2. Create external **`BufferPool`** instances for device or host memory (not owned by the transport).
3. Publish **trivially copyable payloads** that embed **`BufferHandle`** descriptors.
4. On the subscriber side, adopt GPU-accessible memory via **`Context::import_buffer`** → **`ConstBuffer`**.

The messaging backend is **iceoryx2**, fixed at compile time (`HOLOMQ_TRANSPORT=iox2`).
Application code should include only
top-level `<holomq/...>` headers — not `<holomq/detail/...>`.

**Namespace:** `holomq`
**C++ standard:** C++20
**Primary headers:** top-level files under `include/holomq/` (installed as `<holomq/...>`). Headers under `include/holomq/detail/` are implementation details and are not part of the supported application API; their types live in `holomq::detail` (for example `holomq::detail::transport` for messaging backends, `holomq::detail::device::BufferHandleData` for wire-layout structs included transitively by `BufferHandle`).

### Supported public headers (non-`detail`)

| Area | Headers |
|------|---------|
| Context | `context.hpp`, `context_builder.hpp` |
| Pub/sub | `publisher.hpp`, `publisher_builder.hpp`, `subscriber.hpp`, `subscriber_builder.hpp`, `pubsub_options.hpp`, `pubsub_capabilities.hpp`, `local_buffer_descriptor.hpp`, `outbound_message.hpp`, `inbound_message.hpp` |
| Wait | `wait_set.hpp`, `wait_set_handle.hpp`, `wait_strategy.hpp` |
| Buffers | `buffer_pool.hpp`, `buffer.hpp`, `const_buffer.hpp`, `host_buffer_import.hpp`, `buffer_handle.hpp`, `device_buffer_allocator.hpp`, `host_buffer_allocator.hpp` |
| Config / logging | `limits.hpp`, `logger.hpp` |
| Holoscan SDK bridge | `adapters/holoscan_logger.hpp` (requires Holoscan Core headers and `holoscan::Logger` at link time) |

---

## Build-time configuration

These affect the public API surface but are not runtime-selectable:

| CMake option | Effect on API |
|--------------|---------------|
| `HOLOMQ_TRANSPORT` (`iox2`, required) | iceoryx2 messaging backend. Compile definition: `HOLOMQ_TRANSPORT_IOX2=1`. |
| `HOLOMQ_DEFAULT_BACKEND` / `HOLOSCAN_MQ_BACKEND` (`iceoryx2`, required) | SDK/standalone cache names mapped to `HOLOMQ_TRANSPORT=iox2` when unset. |
| `HOLOMQ_FD_PASS_SOCKET` (`PATH`, `ABSTRACT`) | How FD-pass endpoints are encoded in `detail::device::BufferHandleData::fd_pass_socket_path` and `detail::host::BufferHandleData::fd_pass_socket_path`. Publisher and subscriber **must use the same build mode**. On Linux the default is `ABSTRACT`; elsewhere `PATH`. Exposed at compile time as `HOLOMQ_FD_PASS_USE_ABSTRACT`. |

In the Holoscan SDK tree, `HOLOMQ_TRANSPORT` defaults from `HOLOMQ_DEFAULT_BACKEND` /
`HOLOSCAN_MQ_BACKEND` when unset. Only `iceoryx2` → `iox2` is supported.

Default transport QoS values live in `limits.hpp`:

```cpp
#include <holomq/limits.hpp>

holomq::k_subscriber_queue_depth   // 8
holomq::k_subscriber_max_borrowed_samples  // 0, derive from queue depth
holomq::k_publisher_history_depth  // 0
holomq::k_lossless_flow            // true
holomq::k_max_in_flight            // default derived iox2 max_loaned_samples
```

Typed pub/sub builders separate a role-independent static service contract from
endpoint-local configuration. Pass the same
{cpp:class}`holomq::PubSubServiceOptions` to publisher and subscriber builders;
either role may create the service first. The service options own queue policy,
fanout capacity, per-subscriber queue and borrowed-sample maxima, publisher
history, and event-wake enablement. When an endpoint opens an existing service,
HoloMQ verifies queue policy, queue and borrow maxima, history, and each
explicit non-zero subscriber-capacity request, returning `kIncompatible` for a
mismatch. A zero subscriber capacity remains a backend-default wildcard.

Publishers configure the publisher loan limit, release/replacement callbacks,
and endpoint-local full-queue response with
{cpp:class}`holomq::PublisherPubSubOptions` or fluent convenience methods.
`PublisherPubSubOptions::expected_recipient_count` defaults to zero, which
disables attachment gating. A non-zero value makes both `send` overloads return
`kIncompleteFanout` before backend submission until the last refreshed subscriber count
reaches that value. It must not exceed an explicitly configured non-zero
`PubSubServiceOptions::max_subscribers`; the service maximum remains capacity
rather than an implicit required-peer count.
`PublisherPubSubOptions::enable_sample_release_callbacks` defaults to `true` for
compatibility. Set it to `false` for self-contained payloads to omit the release
registry and backend handler; the callback-taking `send` overload then returns
`kNotImplemented` while plain `send` remains available.
`PublisherPubSubOptions::pressure_policy` defaults to
{cpp:enumerator}`holomq::PublisherPressurePolicy::kFollowService`. The
{cpp:enumerator}`holomq::PublisherPressurePolicy::kRejectWhenFull` alternative is valid only for a
{cpp:enumerator}`holomq::QueuePolicy::kBlocking` service: it preserves already queued samples but
returns `kBackpressured` instead of waiting when a recipient queue is full. In a fanout, another
recipient may already have accepted that send, so an error does not prove zero delivery.
{cpp:enumerator}`holomq::PublisherPressurePolicy::kBoundedRetry` is also valid only for a blocking
service. Set `PublisherPubSubOptions::bounded_retry_budget_ns` to a non-zero endpoint-local budget;
the publisher retries a full recipient until that budget expires, then returns `kBackpressured`
without discarding the sample already queued at that recipient. Budget exhaustion can still follow
partial fanout delivery and does not authorize automatic retry of the logical publication. The
iceoryx2 backend records the full recipient's opaque 16-byte port identity in the failure-only
diagnostic as `HOLOMQ_BACKPRESSURED recipient=<32 lowercase hex digits>`; accepted sends perform no
attribution work unless they first encounter recipient pressure.
Subscribers request their endpoint-local queue depth and borrowed-sample allowance with
{cpp:class}`holomq::SubscriberPubSubOptions` or fluent convenience methods.
Subscriber queue depth must be greater than zero and no larger than the service
maximum. Non-zero subscriber borrowed-sample allowances must be no larger than
the service maximum; zero derives from the configured queue depth. Queue
safe-overflow behavior is a service-level transport property, not a
per-subscriber property; use separate service names when one logical output
needs mixed blocking and drop-oldest delivery classes. The backend does not
expose event-service intent or preserve exact queue-policy spelling through its
numeric sample-service static configuration. HoloMQ therefore records
`enable_event_wake` and `queue_policy` as required sample-service attributes.
It verifies the configured maximum subscriber queue depth directly against the
backend's numeric static configuration and rejects a peer with a different
value.

Typed pub/sub capabilities are setup-time metadata:

```cpp
auto context = holomq::ContextBuilder().create();
if (!context) { /* handle init failure */ }

holomq::PubSubCapabilities caps = context->pubsub_capabilities();

// For setup code that has not created a Context yet:
holomq::PubSubCapabilities default_caps = holomq::default_pubsub_capabilities();
```

The current iceoryx2-backed substrate supports typed POD payloads, fanout,
non-consuming readiness checks, wait sets, per-recipient delivery attribution,
and {cpp:enumerator}`holomq::QueuePolicy::kBlocking` /
{cpp:enumerator}`holomq::QueuePolicy::kDropOldest`. A payload can carry
either complete inline data, {cpp:class}`holomq::LocalBufferDescriptor` for a
same-process Holoscan Runtime sidecar store, or a HoloMQ
{cpp:class}`holomq::BufferHandle` for the cross-process buffer plane. HoloMQ
resolves {cpp:class}`holomq::BufferHandle`; it does **not** resolve
{cpp:class}`holomq::LocalBufferDescriptor` values into buffer lifetimes. That
local descriptor path
remains a Holoscan Runtime sidecar responsibility so same-process Tensor
channels can avoid CUDA VMM, fd-pass, and SHM ref-count import/export.

{cpp:class}`holomq::LocalBufferDescriptor` is a 40-byte standard-layout, trivially-copyable
control payload:

| Field | Type | Meaning |
|-------|------|---------|
| `abi_version` | `std::uint32_t` | Descriptor ABI version. Receivers reject unsupported values. |
| `flags` | `std::uint32_t` | Runtime-defined flags; HoloMQ does not interpret the bits. |
| `store_id` | `std::uint64_t` | Runtime-owned sidecar store identity. Zero means empty. |
| `record_index` | `std::uint64_t` | Runtime-owned record index within the sidecar store. |
| `generation` | `std::uint64_t` | ABA-protection generation for the record. |
| `token` | `std::uint64_t` | Runtime-defined token for validation or claim release. |

---

## Error reporting

Failure-returning HoloMQ APIs use **`holomq::expected<T, holomq::Error>`** (`#include <holomq/expected.hpp>`, `#include <holomq/errors.hpp>`). Check `has_value()` for success; on failure read `error().code` and optional `error().message`. Logged failures use `HOLOMQ_LOG_*` (see [Logging](#logging)); some error codes are normal control flow and are **not** logged.

Installed **`holomq::holomq`** links **`tl::expected`** publicly (header-only alias in `holomq/expected.hpp`). **`error_code_name()`** and **`error_message()`** are implemented in the library.

| `ErrorCode` | Value | Typical use |
|-------------|------:|-------------|
| `kFailure` | 1 | Generic / unclassified |
| `kReceiveError` | 2 | Unclassified subscriber transport receive failure |
| `kInvalidArgument` | 4 | Misconfiguration, bad handle, duplicate send callback, duplicate wait-set attach |
| `kNotFound` | 5 | Missing import / pool slot |
| `kNotImplemented` | 7 | Unsupported backend path |
| `kBackpressured` | 8 | Loan/pool exhaustion or publisher delivery backpressure |
| `kNotReady` | 9 | Empty subscriber queue or publisher connection not ready |
| `kStaleEndpoint` | 10 | Peer endpoint is corrupted or its data segment cannot be mapped |
| `kClosed` | 11 | Publisher endpoint is closed |
| `kTimeout` | 12 | `wait_and_take()` timed out (not logged) |
| `kResourceExhausted` | 13 | Endpoint, wait-set, service, or system capacity exhausted |
| `kPermissionDenied` | 14 | Insufficient permission to create, open, or wait on a resource |
| `kInterrupted` | 15 | Operation interrupted by a signal or cancellation request |
| `kProtocolError` | 16 | Peer, service, or notification contract violation |
| `kIncompatible` | 17 | Existing service or endpoint has incompatible type, policy, or capabilities |
| `kIncompleteFanout` | 18 | Fewer than the publisher's required recipients are attached |

The numeric values belong only to HoloMQ and are not a Holoscan Core compatibility contract.
Integrations map every named value explicitly and fail closed for unknown future values; they must
not cast an `ErrorCode`, assert numeric parity, or infer semantics from diagnostic text.

**`Subscriber::try_take()`:** `kNotReady` when no sample is queued or a publisher connection is not ready, `kBackpressured` when the subscriber has reached its borrowed-sample limit, and `kStaleEndpoint` when a publisher data segment cannot be mapped.

**`Subscriber::wait_and_take()`:** `kTimeout` on wait timeout (not logged; default **`TimedWait`** is ~1 ms after spin). `kInvalidArgument` when **`WaitMode::KernelWait`** is used without event wake. Other transport failures preserve their connection, backpressure, stale-endpoint, or internal error categories. Retries on spurious kernel wake when a sample is not yet queued.

**`Publisher::loan()` / `BufferPool::acquire()`:** `kBackpressured` when slots are exhausted (not logged).

**`Publisher::send`:** returns `expected<int, Error>` — success value is subscriber count. Transport
failures map to `kClosed`, `kStaleEndpoint`, `kNotReady`, `kBackpressured`, `kInvalidArgument`, or
`kFailure`. `kNotReady` and `kBackpressured` are ordinary flow-control outcomes and are emitted only
at trace severity, never as error diagnostics. A duplicate release callback returns
`kInvalidArgument`.

**Context, publisher, subscriber, and wait-set setup:** preserves invalid configuration,
incompatible service, resource exhaustion, permission, transient service state, stale service, and
internal failure categories. Setup errors include a diagnostic message with the affected service or
resource where available.

Semantic optionals remain **`std::optional`** where empty is not an error: `cuda_device_ordinal()`, `WaitStrategy::timeout`, and similar.

---

## Logging

```cpp
#include <holomq/logger.hpp>

holomq::set_log_level(holomq::LogLevel::Info);
holomq::set_log_handler(my_handler, user_data);  // optional override
HOLOMQ_LOG_WARN("publisher backlog: {}", depth);
```

| API | Description |
|-----|-------------|
| `HOLOMQ_LOG_TRACE` … `HOLOMQ_LOG_CRITICAL` | Level-aware macros (C++20 `std::format` syntax). Compile-time filtering via `HOLOMQ_LOG_ACTIVE_LEVEL` (default: all levels enabled). |
| `set_log_level` / `log_level` | Runtime filter. `HOLOMQ_LOG_LEVEL` env overrides an explicit `set_log_level` call (values: `TRACE`, `DEBUG`, `INFO`, `WARN`, `ERROR`, `CRITICAL`, `OFF`). |
| `set_log_handler` / `reset_log_handler` | Optional sink override. Handler signature: `(LogLevel, file, line, function, message, user_data)`. Handlers must not call back into HoloMQ logging APIs (deadlock). |
| `log_literal` | Pre-formatted message path for noexcept contexts (with or without source location). |
| `enable_transport_logging()` / `enable_transport_logging(LogLevel)` | Route iceoryx2 diagnostics through the HoloMQ logger. Honors `HOLOMQ_LOG_LEVEL` when set. |
| `disable_transport_logging()` | Best-effort disable of the IOX2 bridge (iceoryx2 cannot unset its logger once installed). |
| `ContextBuilder::with_transport_logging(bool)` | Installs the IOX2 bridge in `create()` before transport init. |

Default output (no handler): `[holomq][LEVEL] message` on stderr.

**Holoscan SDK apps** may redirect HoloMQ logs into `holoscan::Logger` (link `holoscan::Logger` / `libholoscan_logger`):

```cpp
#include <holomq/adapters/holoscan_logger.hpp>

// Before Context::create / enable_transport_logging / first HOLOMQ_LOG_*.
holomq::redirect_holomq_logs_to_holoscan();
holomq::set_log_level(holomq::LogLevel::Info);
holoscan::Logger::set_level(holoscan::LogLevel::INFO);  // align filters
```

Do not call HoloMQ logging APIs from custom `holoscan::Logger` sinks that might run during the bridge handler (deadlock).

**Transport logging (iox2):** call `redirect_holomq_logs_to_holoscan()` (or `set_log_handler` / `set_log_level`) first, then either `enable_transport_logging()` or `ContextBuilder().with_transport_logging(true)`. HoloMQ pre-installs the bridge before creating an iox2 runtime, so transport logging can be enabled after a context already exists; diagnostics emitted before logging is enabled are dropped. When the bridge is active, `HOLOMQ_LOG_LEVEL` controls both HoloMQ and iceoryx2 verbosity (`IOX2_LOG_LEVEL` is not read).

---

## Entry point: `Context` and `ContextBuilder`

### `ContextBuilder`

Fluent builder; call `create()` to open the node and optional VMM FD-pass server.

```cpp
#include <holomq/context_builder.hpp>

auto context = holomq::ContextBuilder()
    .with_vmm_fd_export(/* bool, default true */)
    .with_fd_pass_socket(/* std::string, optional */)
    .with_transport_logging(/* bool, default false */)
    .create();  // -> expected<Context, Error>
```

| Method | Description |
|--------|-------------|
| `with_vmm_fd_export(bool)` | When `true`, starts an internal AF_UNIX FD-pass server for cross-process VMM fd export. **Publishers** exporting `BufferHandle`s to remote processes should leave this enabled (see pool factories below). Default `true`. Subscribers that only import handles may pass `false`. |
| `with_fd_pass_socket(std::string)` | AF_UNIX listen path for the FD-pass server. Empty → default path under `/tmp` at `create()`. **Error at `create()`** if path is non-empty but export is disabled. |
| `with_transport_logging(bool)` | When `true`, route iceoryx2 transport diagnostics through the HoloMQ logger at `create()`. Default `false`. |
| `create()` | Returns an error on failure (transport init, FD-pass bind, etc.). |

**Typical roles:**

- **Publisher (cross-process buffer export):** default `with_vmm_fd_export(true)`; optionally set `with_fd_pass_socket(...)`. Required for device pools, pageable host pools, and pinned host pools without `CU_MEMHOSTREGISTER_DEVICEMAP`. **Exception:** pinned host with `CU_MEMHOSTREGISTER_DEVICEMAP` (devicemap interleaved) — pass `with_vmm_fd_export(false)`.
- **Subscriber:** `with_vmm_fd_export(false)`; FD-pass path comes from received `BufferHandle` descriptors when import needs it.

### `Context`

Owns transport runtime, subscriber ref-count SHM cache, GPU import cache, and (when configured) the FD-pass server.

```cpp
#include <holomq/context.hpp>
```

| Method | Returns | Description |
|--------|---------|-------------|
| `vmm_fd_export_enabled()` | `bool` | Whether the FD-pass server is running on this `Context`. |
| `fd_pass_socket_path()` | `const std::string&` | Resolved FD-pass listen path (always non-empty; embedded in buffer descriptors). A server listens on this path only when `vmm_fd_export_enabled()` is `true`. |
| `pubsub_capabilities()` | `PubSubCapabilities` | Setup-time typed pub/sub capabilities for this context's compiled backend. |
| `import_buffer(BufferHandle, CUstream)` | `expected<ConstBuffer, Error>` | **Recommended.** Map the handle and wait on `stream` until the publisher's export-time GPU work is visible. Error if `stream` is null while a wait is required. |
| `import_buffer_unsynchronized(BufferHandle)` | `expected<ConstBuffer, Error>` | **Advanced.** Map the handle only (no GPU wait). Undefined behavior to use `ptr()` on GPU before `ConstBuffer::enqueue_ready_wait(stream)` when the handle carries an export fence. |
| `import_host_buffer_unsynchronized(BufferHandle, HostBufferImport&)` | `expected<void, Error>` | **Advanced bounded-runtime path.** Import into an inactive reusable host-slot lease. The first call resolves mappings; later calls accept only the same exact pool slot, acquire a fresh SHM ref-count lease, and reuse the retained mappings. |
| `rebind_host_buffer_unsynchronized(BufferHandle, HostBufferImport&)` | `expected<void, Error>` | **Advanced bounded-borrow path.** Import into an inactive reusable lease; after first use, later handles may select another slot in the same fully validated host pool and reuse its retained mappings. |
| `import_host_buffer_unsynchronized(BufferHandle, span<HostBufferImport>)` | `expected<size_t, Error>` | **Advanced bounded-runtime path.** Route the opaque handle slot to a pool-sized reusable-import table in O(1), import that entry, and return its index. This avoids application wire-field access and a linear destination scan. |
| `create_device_buffer_pool(buffer_bytes, buffer_count[, allocator])` | `expected<BufferPool, Error>` | Device VMM pool; third argument optional (`DeviceBufferAllocator`). **Requires** `with_vmm_fd_export(true)` on the publisher `Context`. |
| `create_host_buffer_pool(buffer_bytes, buffer_count[, allocator])` | `expected<BufferPool, Error>` | Pageable host SHM pool (payload SHM + separate sync VMM pool). **Requires** `with_vmm_fd_export(true)` on the publisher `Context`. |
| `create_host_pinned_buffer_pool(buffer_bytes, buffer_count, cuda_host_register_flags[, allocator])` | `expected<BufferPool, Error>` | Host-pinned pool; `cuda_host_register_flags` passed to `cuMemHostRegister` (including `0`). **Requires** `with_vmm_fd_export(true)` when `CU_MEMHOSTREGISTER_DEVICEMAP` is clear (separate sync-pool layout). Devicemap interleaved (`DEVICEMAP` set) does not require publisher FD-pass. |
| `publisher_builder(std::string service_name)` | `PublisherServiceBuilder` | Start publisher setup for the given service name (see [Publish–subscribe](#publishsubscribe)). |
| `subscriber_builder(std::string service_name)` | `SubscriberServiceBuilder` | Start subscriber setup for the given service name (see [Publish–subscribe](#publishsubscribe)). |
| `create_waitset()` | `expected<WaitSet, Error>` | Create a multi-subscriber wait set. |

**Move-only:** `Context` is movable; copy is deleted. The public handle shares session state via an internal `std::shared_ptr`; **buffer pools keep the session alive** until they are destroyed, so destroying or reassigning a `Context` while pools remain is safe. Publishers, subscribers, and builders still require a live `Context` handle for their transport pointers.

#### `Context::create_device_buffer_pool` / `Context::create_host_buffer_pool` / `Context::create_host_pinned_buffer_pool`

```cpp
auto device_pool = context.create_device_buffer_pool(buffer_bytes, buffer_count);

auto pageable_host_pool = context.create_host_buffer_pool(buffer_bytes, buffer_count);

auto pinned_host_pool = context.create_host_pinned_buffer_pool(
    buffer_bytes,
    buffer_count,
    CU_MEMHOSTREGISTER_PORTABLE | CU_MEMHOSTREGISTER_DEVICEMAP);
```

| Parameter | `create_device_buffer_pool` | `create_host_buffer_pool` | `create_host_pinned_buffer_pool` |
|-----------|----------------------------|---------------------------|----------------------------------|
| `buffer_bytes` | `std::size_t` — byte size of each slot (non-zero) | same | same |
| `buffer_count` | `std::uint32_t` — number of slots (non-zero) | same | same |
| 3rd argument | Optional `DeviceBufferAllocator` (default `{}` → built-in CUDA VMM) | Optional `HostBufferAllocator` (default `{}` → built-in SHM allocator) | Required `unsigned int cuda_host_register_flags` — `CU_MEMHOSTREGISTER_*` for `cuMemHostRegister` (`0` = CUDA default register) |
| 4th argument | — | — | Optional `HostBufferAllocator` (default `{}` → built-in SHM allocator) |

See [DeviceBufferAllocator](#devicebufferallocator) and [HostBufferAllocator](#hostbufferallocator). See also [`host-buffer-pools.md`](design-notes/host-buffer-pools.md) and [`holomq-vs-holoscan-memory-types.md`](design-notes/holomq-vs-holoscan-memory-types.md).

**`BufferMemoryType` for pool creation** (`Buffer::MemoryType` is an alias):

| Value | Supported today | Description |
|-------|-----------------|-------------|
| `Device` | Yes | CUDA device memory (VMM pool). **Requires** publisher `with_vmm_fd_export(true)` (`create_device_buffer_pool`). |
| `HostPageable` | Yes | POSIX SHM pageable payload (`create_host_buffer_pool`) — publisher `with_vmm_fd_export(true)` required. |
| `HostPinned` | Yes | POSIX SHM payload registered with CUDA (`create_host_pinned_buffer_pool`). **Devicemap interleaved** (`CU_MEMHOSTREGISTER_DEVICEMAP`): publisher FD-pass not required. **Separate sync-pool layout** (no `DEVICEMAP`): publisher `with_vmm_fd_export(true)` required. |
| `None` | No | Used for empty handles, not as a pool kind. |

Pool factory failures (SHM, registration, invalid flags, missing CUDA context, fd-pass unavailable
when required) are logged and return **`expected<BufferPool, Error>`** (`kInvalidArgument` for
misconfiguration; `kFailure` for resource failures).

#### Buffer import

**Recommended subscriber path** — map the handle and, when it carries an export fence, wait on `stream` until the publisher's export-time GPU work is visible:

```cpp
auto buf_or = context.import_buffer(payload.buffer_handle, sub_stream);
if (!buf_or.has_value()) {
  /* kInvalidArgument (null stream while wait required), kNotFound, etc. */
}
holomq::ConstBuffer buf = std::move(buf_or).value();

// Safe to enqueue GPU reads on sub_stream using buf.ptr().
```

#### Advanced import (deferred GPU wait)

Use **`import_buffer_unsynchronized`** when mapping and the cross-process GPU fence must happen on separate steps (for example custom stream ordering before `enqueue_ready_wait`):

```cpp
auto buf_or = context.import_buffer_unsynchronized(payload.buffer_handle);
if (!buf_or.has_value()) { /* handle error */ }
holomq::ConstBuffer buf = std::move(buf_or).value();

// Application sync here must cover publisher export-time GPU work before GPU use.

if (!buf.enqueue_ready_wait(sub_stream)) { /* invalid stream or wait failure */ }

// Safe to enqueue GPU reads on sub_stream using buf.ptr().
```

`import_buffer(handle, stream)` is equivalent to `import_buffer_unsynchronized(handle)` followed by `enqueue_ready_wait(stream)` when the handle carries an export fence.

#### Reusable host import

Bounded runtimes that preallocate one receive control per host-pool slot can retain the resolved
mapping while releasing each message's SHM lease:

```cpp
#include <holomq/host_buffer_import.hpp>

holomq::HostBufferImport imported;

auto acquired = context.import_host_buffer_unsynchronized(payload.buffer_handle, imported);
if (!acquired) { /* wrong memory kind, active destination, or different pool slot */ }

// Enqueue imported.enqueue_ready_wait(stream) before GPU access when the handle has a fence.
consume(imported.ptr(), imported.size_bytes());
imported.release();
```

The first successful call may attach SHM/VMM mappings and allocate their cache entries. After
`release()`, reacquiring a later generation of the same exact pool slot does not repeat those
mapping-cache lookups or allocate another `ConstBuffer` implementation. Reacquisition still adds a
new remote ref-count lease; `release()` drops it immediately or defers it on the stream previously
passed to `enqueue_ready_wait()`.

`HostBufferImport` is move-only and serialized. Import, access, and release on one instance must not
run concurrently. It supports host handles only (`HostPageable` and `HostPinned`), rejects a
different pool or slot, and does not expose the opaque slot index. The object and its owning
`Context` must outlive each active access because the retained payload/VMM mappings belong to that
Context.

---

## Publish–subscribe

Messages carry a single application-defined **payload** type only. GPU memory is **not** managed by the transport; embed `BufferHandle` in the payload as needed.

Publisher and subscriber ports are joined by a **service name**: the string passed to `publisher_builder()` and `subscriber_builder()`. Publisher and subscriber **must use the same name** to connect. The backend maps this string to its own service/channel identifier (holomq does not define a separate topic type).

### Payload constraints

```cpp
static_assert(std::is_standard_layout_v<Payload>);
static_assert(std::is_trivially_copyable_v<Payload>);
```

Enforced at `PublisherBuilder::create()` and `SubscriberBuilder::create()`.

Both **`PublisherServiceBuilder`** / **`PublisherBuilder`** and
**`SubscriberServiceBuilder`** / **`SubscriberBuilder`** accept
**`service_options(PubSubServiceOptions)`** (before or after
`payload_type<T>()`). Their **`enable_event_wake(bool)`** convenience setter
updates that shared option set. Event wake defaults to `true`; symmetric
settings on publisher and subscriber are required for **`WaitMode::KernelWait`**
(kernel event listener).

Publisher and subscriber builders both support `service_options(...)` with
{cpp:class}`holomq::PubSubServiceOptions`. Publisher builders additionally
support `publisher_options(...)` with
{cpp:class}`holomq::PublisherPubSubOptions`, `queue_policy(...)` with
{cpp:enum}`holomq::QueuePolicy`, `max_subscriber_queue_depth(uint32_t)`,
`max_subscriber_borrowed_samples(uint32_t)`, and
`max_loaned_samples(uint32_t)`.
Subscriber builders additionally support `subscriber_options(...)` with
{cpp:class}`holomq::SubscriberPubSubOptions`, `queue_policy(...)`, and
`queue_depth(uint32_t)` and `max_borrowed_samples(uint32_t)`.
{cpp:enumerator}`holomq::QueuePolicy::kBlocking` is the default lossless/backpressured mode.
{cpp:enum}`holomq::PublisherPressurePolicy` is endpoint-local and does not change that service
contract. Its default `kFollowService` waits as before. `kRejectWhenFull` is intended for an
embedding scheduler that must regain execution control; it rejects a full recipient without
discarding the sample already queued there and returns `kBackpressured`.
`kBoundedRetry` retries a full recipient for the explicitly configured
`bounded_retry_budget_ns`, then applies the same preserve-queued-sample rejection. Its finite
budget bounds publisher delay but does not make multi-recipient publication atomic.
{cpp:enumerator}`holomq::QueuePolicy::kDropOldest` honors the configured depth
and replaces the oldest unread sample when the queue is full, retaining the
newest N samples. Use queue depth one for a freshest-only mailbox. Replacement
is attributed per subscriber when the backend supports per-recipient delivery
attribution.

For the iceoryx2 backend, these options map to the service and publisher
capacity controls that determine physical sample storage:

```text
number_of_samples =
    max_subscribers * (subscriber_max_buffer_size + subscriber_max_borrowed_samples)
  + publisher_history_size
  + max_loaned_samples
```

`subscriber_max_buffer_size` is the maximum in the shared service options;
each subscriber's effective
`queue_depth` must be within that maximum. `subscriber_max_borrowed_samples` is
the service maximum number of concurrently borrowed received samples per
subscriber; each subscriber's effective `max_borrowed_samples` must be within
that maximum. HoloMQ uses `0` as the default auto value for borrowed-sample
allowances: service setup derives from the configured maximum subscriber queue
depth, and subscriber-side endpoint setup derives from the configured
subscriber queue depth. This lets a subscriber receive and hold
queued samples unless the application explicitly tightens that limit. With
history disabled, Holoscan-style one-loan-at-a-time publishing can usually keep
`max_loaned_samples` small. The default `max_loaned_samples = 0` preserves the
previous HoloMQ behavior by deriving `max(max_subscriber_queue_depth,
publisher_history_depth)`.

When a subscriber is created before any publisher exists, HoloMQ creates the
underlying iceoryx2 service from the same
{cpp:class}`holomq::PubSubServiceOptions` accepted by the publisher path.
Consequently publisher history and the other static capacity settings no longer
depend on endpoint creation order.

{cpp:member}`holomq::PubSubCapabilities::delivery_attribution` describes whether the
substrate can identify individual recipients for delivered or superseded samples.
{cpp:func}`holomq::Subscriber::recipient_id` is meaningful only when this capability is
{cpp:enumerator}`holomq::DeliveryAttribution::kPerRecipient`; other backends return a default
id.
For descriptor-backed lossy channels, a lagging subscriber's unread descriptor
can be superseded; the runtime must release that subscriber's old sidecar claim
exactly once. The iceoryx2 replacement callback is installed when a publisher
uses {cpp:enumerator}`holomq::QueuePolicy::kDropOldest` and the effective
{cpp:class}`holomq::PublisherPubSubOptions` set `replacement_attribution_required`.
Calling {cpp:func}`holomq::PublisherBuilder::on_sample_replaced` requests that option on
the builder and provides the callback. HoloMQ does not gate this on current subscriber
count; Holoscan Runtime owns the static graph contract and can omit replacement
attribution when it can infer the affected subscriber locally.

For descriptor-backed payloads, prefer the publisher-level
{cpp:func}`holomq::PublisherBuilder::on_sample_replaced` callback when the
skipped descriptor can be read directly from the replaced payload. That callback is registered once at publisher setup
and avoids per-send callback registration. The payload reference and
{cpp:member}`holomq::SampleReplacedInfo::payload_ptr` are valid only during
the callback. The callback should not throw; HoloMQ contains and logs exceptions before returning
to the transport callback boundary. A replacement event is about one recipient's
queue entry being overwritten. It does not mean the physical transport sample
bucket has been released; a later sample-release event can still occur for the
same payload address if the caller registered {cpp:func}`holomq::Publisher::send`
with an `on_released` callback. This path depends on the HoloMQ-pinned iceoryx2 replacement-handler patch; if that
patch is removed before upstreaming an equivalent iceoryx2 API,
descriptor-backed drop-oldest replacement attribution must fall back to a
runtime-owned mirror or be rejected at graph finalization.

```cpp
// Publisher
holomq::PubSubServiceOptions service_options {};
service_options.queue_policy = holomq::QueuePolicy::kBlocking;

auto publisher = context.publisher_builder("my_service")
    .service_options(service_options)
    .payload_type<MyPayload>()
    .create();

// Subscriber (same service name as publisher)
auto subscriber = context.subscriber_builder("my_service")
    .service_options(service_options)
    .payload_type<MyPayload>()
    .create();
```

Freshest-only mailbox service:

```cpp
holomq::PubSubServiceOptions service_options {};
service_options.queue_policy = holomq::QueuePolicy::kDropOldest;
service_options.max_subscriber_queue_depth = 1;

holomq::PublisherPubSubOptions publisher_options {};

holomq::SubscriberPubSubOptions subscriber_options {};
subscriber_options.queue_depth = 1;

auto publisher = context.publisher_builder("control")
    .service_options(service_options)
    .publisher_options(publisher_options)
    .payload_type<ControlPayload>()
    .on_sample_replaced([](const ControlPayload& skipped,
                           const holomq::SampleReplacedInfo& info) {
      // Release runtime-owned sidecar state described by skipped.
      // info.receiver_id identifies the subscriber whose unread sample was replaced.
    })
    .create();

auto subscriber = context.subscriber_builder("control")
    .service_options(service_options)
    .subscriber_options(subscriber_options)
    .payload_type<ControlPayload>()
    .create();
```

Bounded drop-oldest service:

```cpp
holomq::PubSubServiceOptions service_options {};
service_options.queue_policy = holomq::QueuePolicy::kDropOldest;
service_options.max_subscriber_queue_depth = 4;

holomq::PublisherPubSubOptions publisher_options {};

holomq::SubscriberPubSubOptions subscriber_options {};
subscriber_options.queue_depth = 4;  // retains the newest four unread samples
```

### Builder flow

Publisher and subscriber ports are created with a fluent builder: set the service name and payload type, then call `create()`:

```cpp
constexpr const char* k_service = "my_service";

// Publisher
auto publisher = context.publisher_builder(k_service)
    .payload_type<MyPayload>()
    .create();  // -> expected<Publisher<MyPayload>, Error>

// Subscriber (same service name as publisher)
auto subscriber = context.subscriber_builder(k_service)
    .payload_type<MyPayload>()
    .create();  // -> expected<Subscriber<MyPayload>, Error>
```

The builder exposes the **service name**, **payload type**, shared
**`service_options`**, endpoint-local typed pub/sub options, and focused
convenience setters. Other backend-specific QoS settings remain internal.

### `Publisher<Payload>`

```cpp
#include <holomq/publisher.hpp>
```

| Method | Returns | Description |
|--------|---------|-------------|
| `update_connections()` | `expected<void, Error>` | Refresh connection state, preserving connection failures. |
| `has_subscribers()` | `bool` | Whether the last successfully refreshed state contains subscribers. |
| `number_of_subscribers()` | `uint64_t` | Exact subscriber count from the last successfully refreshed state. |
| `loan()` | `expected<OutboundMessage<Payload>, Error>` | Borrow an outbound message slot from the transport (`kBackpressured` when exhausted). |
| `send(OutboundMessage<Payload>&& message)` | `expected<int, Error>` | Send message. Success value: subscriber count; `kIncompleteFanout` before submission when the configured expected recipient count is not attached. For `BufferHandle` payloads, the caller must keep the owner `Buffer` alive separately; prefer the callback overload below. |
| `send(OutboundMessage<Payload>&&, Fn&& on_released)` | `expected<int, Error>` | Same as above. Move objects (e.g. `Buffer`) into **`on_released`** (typically by lambda capture) to extend their lifetime until subscribers have received and released the sample, or until the transport drops it; **`on_released` runs when the transport releases the sample slot**, including after a reported send error. `kNotImplemented` if release callbacks were disabled at publisher creation; `kInvalidArgument` if a callback is already pending for this payload address. |

### `Subscriber<Payload>`

```cpp
#include <holomq/subscriber.hpp>
```

| Method | Returns | Description |
|--------|---------|-------------|
| `try_take()` | `expected<InboundMessage<Payload>, Error>` | Non-blocking take. Preserves empty-queue, backpressure, connection, and stale-endpoint errors. |
| `set_wait_strategy(WaitStrategy)` | `void` | Default strategy for `wait_and_take()` when no per-call strategy is passed. |
| `wait_and_take()` | `expected<InboundMessage<Payload>, Error>` | Wait for a sample using the port’s default `WaitStrategy`. `kTimeout` on timeout (not logged; default **`TimedWait`** sleeps ~1 ms after spin). `kInvalidArgument` when **`WaitMode::KernelWait`** is used without event wake; other transport failures preserve their HoloMQ error category. Retries on spurious kernel wake when a sample is not yet available. |
| `wait_and_take(WaitStrategy)` | `expected<InboundMessage<Payload>, Error>` | Same using `strategy`. Use **`WaitMode::KernelWait`** with unset `timeout` to block until a sample arrives. **`WaitMode::KernelWait`** requires symmetric event wake (default enabled). |
| `has_samples()` | `expected<bool, Error>` | Whether samples are queued, preserving connection failures. |
| `has_publishers()` | `bool` | Advisory snapshot of whether the service reports a publisher. This is not receive readiness or a delivery guarantee; a moved-from subscriber returns `false`. |
| `recipient_id()` | `DeliveryRecipientId` | Backend-independent subscriber recipient id used for replacement attribution. |

### `WaitSet`

```cpp
#include <holomq/wait_set.hpp>
#include <holomq/wait_strategy.hpp>
```

Multiplexed blocking wait over **multiple** `Subscriber` ports. Attach subscribers, call `wait()` once, then inspect `ready_handles()` to see which attachment(s) may have samples and drain them with `try_take()`. Use a `WaitSet` when one thread serves several services; use `Subscriber::wait_and_take()` for a single port.

Blocking wait uses a parallel **event service** per subscriber when event wake is enabled (default); **`WaitMode::KernelWait`** requires at least one such attachment on a **`WaitSet`**, or event wake on the subscriber for **`wait_and_take`**. **`WaitMode::Return`** and **`WaitMode::TimedWait`** use userspace poll and sleep only. See [`wait-set-architecture.md`](design-notes/wait-set-architecture.md).

| Method | Returns | Description |
|--------|---------|-------------|
| `attach(Subscriber<Payload>&)` | `expected<WaitSetAttachment, Error>` | Register a subscriber for `wait()` / `ready_handles()`. At most one wait set per subscriber. Preserves invalid attachment, resource exhaustion, and internal errors. |
| `set_wait_strategy(WaitStrategy)` | `void` | Default strategy for `wait()` / `wait_for_ready()`. |
| `wait()` / `wait(WaitStrategy)` | `expected<WaitResult, Error>` | Block until work may be available. Global wait failures are returned through the `expected` error, preserving invalid configuration, permission, interruption, closed, and internal wait errors. On IOX2, `ready` can return with an empty `ready_handles()` after kernel spurious wake — loop and call `try_take()`. |
| `wait_for_ready()` / `wait_for_ready(WaitStrategy)` | `expected<WaitResult, Error>` | Convenience aliases of `wait()`; global wait failures are returned through the `expected` error. |
| `ready_handles()` | `span<const WaitSetHandle>` | Handles with pending samples after the last successful `wait()` (may be empty when `wait()` returned `ready` after kernel spurious wake). |
| `capacity()` | `size_t` | Maximum kernel-notification slots. |
| `attachment_count()` | `size_t` | Current attachments. |

### `WaitSetHandle`

```cpp
#include <holomq/wait_set_handle.hpp>
```

Opaque, comparable token that identifies **one subscriber’s attachment** on a `WaitSet`. Obtained from `WaitSetAttachment::handle()` at attach time; compare with `operator<=>` or `==` against entries in `WaitSet::ready_handles()` after `wait()` to dispatch `try_take()` to the right subscriber. Applications do not construct handles except the default (null) value.

| Method | Returns | Description |
|--------|---------|-------------|
| `id()` | `uint64_t` | Opaque numeric id (for logging or debugging; prefer `operator<=>` for dispatch). |

### `WaitSetAttachment`

```cpp
#include <holomq/wait_set.hpp>
```

RAII registration token returned by `WaitSet::attach`. Destroy (or move-from) the attachment to unregister the subscriber from the wait set. Move-only; not copyable.

| Method | Returns | Description |
|--------|---------|-------------|
| `handle()` | `WaitSetHandle` | Stable handle for this attachment; save and compare after `wait()`. |

### `WaitStrategy` / `WaitResult`

```cpp
#include <holomq/wait_strategy.hpp>
```

| `WaitMode` | Behavior |
|------------|----------|
| `Return` | Poll for `spin_duration`, then return. `timeout` is ignored (warning logged when set). |
| `TimedWait` | Poll, then `sleep(timeout)` in userspace regardless of event wake. Default sleep 1 ms when `timeout` is unset. |
| `KernelWait` | Poll, then kernel event wait. Requires event wake. Uses `timeout` when set; indefinite when unset. |

Default mode: `TimedWait` with 50 µs spin (`spin_duration = 50'000` ns).

| `WaitResult` | Meaning |
|--------------|---------|
| `ready` | Wait completed; inspect `ready_handles()` and `try_take()` (may be empty after IOX2 kernel spurious wake). |
| `timeout` | Timed wait expired with no attachment ready (`Return` after the spin phase, `timeout` on `TimedWait` sleep, `timeout` on `KernelWait` kernel wait, or IOX2 timed kernel/listener wait with no notification). |
| `error` | Reserved compatibility value. Global failures from `wait()`, `wait(WaitStrategy)`, `wait_for_ready()`, and `wait_for_ready(WaitStrategy)` are returned through `expected<WaitResult, Error>`; an individual subscriber error is reported by that subscriber's `try_take()`. |

### `OutboundMessage<Payload>`

```cpp
#include <holomq/outbound_message.hpp>
```

Move-only outbound message from `Publisher::loan()`. Fill via `payload()`, then pass to `Publisher::send`. Not default-constructible. Destroying without `send` returns the loan to the transport (sample not published).

| Method | Returns | Description |
|--------|---------|-------------|
| `payload()` | `Payload&` | Mutable access to embedded payload. |
| `payload() const` | `const Payload&` | Const access to embedded payload. |

### `InboundMessage<Payload>`

```cpp
#include <holomq/inbound_message.hpp>
```

Move-only inbound message from `Subscriber::try_take()` / `wait_and_take()`; backend hold released on destruction. Import any `BufferHandle` in the payload while the `InboundMessage` is still alive—copying the handle does not extend the transport sample hold. Not default-constructible. This is the application-facing type (`holomq::InboundMessage`), distinct from internal backend inbound types under `detail/transport/`.

| Method | Returns | Description |
|--------|---------|-------------|
| `payload() const` | `const Payload&` | Read received payload (contains handles to adopt). |

---

## Buffer pool and GPU memory

Device and host memory live in **external** `BufferPool` instances. The
`holomq::k_max_in_flight` constant is the default publisher-loan limit, not a
complete pool-capacity recommendation. A conservative pool also reserves
backing for publisher history and for each subscriber's effective unread queue
and borrowed-sample allowance:

```text
external_buffer_slots >=
    max_loaned_samples
  + publisher_history_depth
  + sum(effective_queue_depth[i] + max_borrowed_samples[i])
```

`QueuePolicy::kDropOldest` contributes the configured unread depth per
subscriber; depth one is the freshest-only mailbox idiom. A more precise bound
may account for one broadcast buffer being retained by multiple subscribers,
but must prove that overlap before reducing the conservative sum.

### `BufferPool`

```cpp
#include <holomq/buffer_pool.hpp>
```

There is no default-constructed empty pool. Creation failure returns an error from the Context factories below (`kInvalidArgument` for misconfiguration or invalid parameters; `kFailure` for resource/allocator failures; errors are logged by the implementation). **`BufferPool` is move-only** (copy is deleted).

| Method | Returns | Description |
|--------|---------|-------------|
| `memory_type()` | `BufferMemoryType` | Pool memory type (`Device`, `HostPageable`, or `HostPinned`). |
| `cuda_device_ordinal()` | `std::optional<int32_t>` | CUDA device ordinal when `memory_type()` is `Device`; empty for host pools. |
| `buffer_bytes()` | `std::size_t` | User payload bytes per slot. |
| `buffer_count()` | `std::uint32_t` | Number of slots. |
| `acquire()` | `expected<Buffer, Error>` | Take next free slot (CAS ref-count 0→1); `kBackpressured` if pool exhausted. |
| `explicit operator bool()` | `bool` | Whether the pool holds a valid implementation (always true for pools obtained from a successful factory call). |

Create pools via **`Context::create_device_buffer_pool`**, **`Context::create_host_buffer_pool`**, or **`Context::create_host_pinned_buffer_pool`** — see [Context pool factories](#contextcreate_device_buffer_pool--contextcreate_host_buffer_pool--contextcreate_host_pinned_buffer_pool). Failures are logged; do not call internal `BufferPool::create<ImplType>(...)` template factories directly unless extending holomq.

#### Ref-count and sample lifetime

- Each slot has a **POSIX SHM ref-count** shared across processes; `acquire()` CAS **0→1**; `Context::import_buffer` constructs a `ConstBuffer` → **fetch_add**.
- Embedding `BufferHandle` in a payload does **not** bump ref-count until import.
- Publisher **`send(..., on_released)`** extends owner `Buffer` lifetime until the transport releases the sample slot.
- `ref_count == 0` does **not** imply GPU idle — ref release may be deferred on the export/import stream.

See [`design-notes/ref-count-model.md`](design-notes/ref-count-model.md) and [`design-notes/publisher-sample-release-registry.md`](design-notes/publisher-sample-release-registry.md). Slide tour: [`presentations/buffer-sync-deep-dive.md`](presentations/buffer-sync-deep-dive.md).

### `Buffer`

```cpp
#include <holomq/buffer.hpp>
```

Mutable pool-owner lease of one fixed-size slot from `BufferPool::acquire()` (device or host pools).

| Method | Description |
|--------|-------------|
| `memory_type()` | `MemoryType` (`None`, `Device`, `HostPageable`, `HostPinned`). |
| `cuda_device_ordinal()` | `std::optional<int32_t>` — CUDA device ordinal when `memory_type()` is `Device`; empty otherwise. |
| `size_bytes()` | User payload byte length for this slot. |
| `explicit operator bool()` | Whether the lease is valid. |
| `ptr()` / `ptr() const` | Device-accessible address (`void*`; cast to `CUdeviceptr` as needed). |
| `export_handle(CUstream)` | After prior ops on `stream`: enqueue a ready-generation set, return `expected<BufferHandle, Error>`, and defer SHM ref-count release until `stream` completes. Does not block the caller. Equivalent to `enqueue_ready_set(stream)` then `export_handle_unsynchronized()`. |
| `export_handle_unsynchronized()` | Return `expected<BufferHandle, Error>` without enqueuing a ready-generation set or deferring ref-count release on a stream. When no prior `enqueue_ready_set`, the handle carries no cross-process GPU fence. **Advanced** — see GPU readiness. |
| `enqueue_ready_set(CUstream)` | Enqueue a ready-generation set on `stream` and defer SHM ref-count release until `stream` completes. Does not block the caller. Idempotent after the first success on this lease. Returns `false` on invalid `stream` or enqueue failure. Pair with `export_handle_unsynchronized()`. |

Move-only value type (copy deleted). Default-constructed and moved-from handles are empty (`operator bool()` is false; `memory_type()` is `MemoryType::None`).

### `ConstBuffer`

```cpp
#include <holomq/const_buffer.hpp>
```

Read-only subscriber import lease from `Context::import_buffer` or `Context::import_buffer_unsynchronized`. `CUdeviceptr` is not implicitly convertible to `void*`; import leases expose payload via `ptr()` as `const void*`.

| Method | Description |
|--------|-------------|
| `memory_type()` | `MemoryType` (`None`, `Device`, `HostPageable`, `HostPinned`). |
| `cuda_device_ordinal()` | `std::optional<int32_t>` — CUDA device ordinal when `memory_type()` is `Device`; empty otherwise. |
| `size_bytes()` | User payload byte length for this slot. |
| `explicit operator bool()` | Whether the lease is valid. |
| `ptr() const` | Device-accessible payload address. |
| `enqueue_ready_wait(CUstream)` | Enqueue a ready-generation wait on `stream` until the publisher's export-time GPU work is visible; arm deferred ref release on success. Does not block the caller. No-op success when the handle has no export fence. Idempotent after the first success. Returns `false` when `stream` is invalid and a wait is required. |
| `export_handle()` | `expected<BufferHandle, Error>` — Re-publish the imported handle without enqueuing a new ready-generation set (identity relay). Deferred ref release remains on the stream from `import_buffer` or `enqueue_ready_wait`. Returns `kInvalidArgument` when the lease is invalid; `kFailure` on export failure. |

Move-only value type (copy deleted). Default-constructed and moved-from handles are empty (`operator bool()` is false; `memory_type()` is `MemoryType::None`).

### `HostBufferImport`

```cpp
#include <holomq/host_buffer_import.hpp>
```

Reusable read-only host import for a bounded runtime that assigns persistent controls to one
validated pool. Exact-slot and same-pool-rebind modes are available. It is not a general
replacement for `ConstBuffer`: device handles and unrelated successive pools continue to use
`Context::import_buffer[_unsynchronized]`.

| Method | Description |
|--------|-------------|
| `explicit operator bool()` | Whether a per-message SHM lease is active. |
| `memory_type()` / `size_bytes()` / `ptr()` | Active host view; return `None`, zero, and null while inactive. |
| `cuda_device_ordinal()` | Empty for the supported host imports. |
| `enqueue_ready_wait(CUstream)` | Enqueue the active handle's ready-generation wait and arm deferred lease release. |
| `export_handle()` | Re-export the active imported handle without creating a new export fence. |
| `release()` | Drop the active remote lease but retain the exact slot mappings for reacquisition. Idempotent while inactive. |

The first import fixes the object's pool and slot identity. A later call to
`Context::import_host_buffer_unsynchronized` succeeds only while the object is inactive and the
handle identifies that same pool slot; only the ready-generation token may advance.

Bounded runtimes should use the `span<HostBufferImport>` overload when one reusable control is
preallocated per producer pool slot. HoloMQ uses the opaque handle's slot internally to select the
entry in O(1) and returns that index; callers neither inspect wire fields nor scan the table.

When the receiver instead knows only its maximum concurrent borrow count, it can preallocate that
many controls and use `rebind_host_buffer_unsynchronized`. An inactive control may then move among
slots in the same validated pool without another mapping-cache lookup or implementation allocation.

### Serializable handle (`BufferHandle`)

Standard-layout, trivially copyable type embedded in message payloads:

```cpp
#include <holomq/buffer_handle.hpp>
```

Applications treat it as an **opaque descriptor**: do not construct from wire-format fields, and do not read or mutate internal layout. The only supported interactions are export from a live resource, copy as part of the payload struct, and import on the subscriber:

| Direction | API |
|-----------|-----|
| Export (publisher, recommended) | `Buffer::export_handle(CUstream)` → `BufferHandle` |
| Export (publisher, advanced) | `Buffer::enqueue_ready_set(CUstream)` then `Buffer::export_handle_unsynchronized()` → `BufferHandle` |
| Import (subscriber, recommended) | `Context::import_buffer(BufferHandle, CUstream)` → `ConstBuffer` (waits for publisher GPU work when required) |
| Import (subscriber, advanced) | `Context::import_buffer_unsynchronized(BufferHandle)` → `ConstBuffer`, then `ConstBuffer::enqueue_ready_wait(CUstream)` |
| Import (bounded host runtime) | Exact-slot `Context::import_host_buffer_unsynchronized`, same-pool `rebind_host_buffer_unsynchronized`, or the pool-sized `span<HostBufferImport>` overload; then `HostBufferImport::release()` after use |
| Relay (subscriber → publisher) | `ConstBuffer::export_handle()` → `BufferHandle` (no new export fence) |

| Method | Description |
|--------|-------------|
| `explicit operator bool()` | Whether the handle is non-empty (`memory_type != None`). |
| `memory_type()` | `BufferMemoryType` (`None`, `Device`, `HostPageable`, `HostPinned`). |

Use `is_device_buffer_memory_type()` / `is_host_buffer_memory_type()` from `<holomq/buffer_memory_type.hpp>` for grouped checks.

Export fence state is embedded in the handle by `export_handle` (`ready_generation` on `BufferHandle`); applications do not read or set it directly.

`slot_index` and pool-kind wire fields live on the handle for IPC but are not part of the supported application API (library import uses them internally). Wire-layout structs are internal (`detail/buffer_pool/device/handle_data.hpp`, `detail/buffer_pool/host/handle_data.hpp`).

### `DeviceBufferAllocator`

```cpp
#include <holomq/device_buffer_allocator.hpp>
```

Custom device pool backing storage:

```cpp
#include <holomq/device_buffer_pool_memory.hpp>

using DeviceBufferAllocator = std::function<
    std::unique_ptr<DeviceBufferPoolMemory>(std::size_t number_of_bytes)>;

DeviceBufferAllocator default_device_buffer_allocator();  // built-in CUDA VMM path
```

| Helper | Description |
|--------|-------------|
| `recommended_device_vmm_granularity()` | Recommended VMM granularity for pinned device allocations with POSIX FD export. |
| `device_supports_vmm_posix_fd_export()` | Whether the current CUDA device supports exporting VMM allocations as POSIX file descriptors. |
| `device_vmm_allocation_is_exportable(CUmemGenericAllocationHandle)` | Whether a handle is a pinned device allocation exportable via POSIX FD. |

Pass a custom `DeviceBufferAllocator` as the optional third argument to `Context::create_device_buffer_pool` when needed. Omit the argument entirely (or pass `{}`) to use the default CUDA VMM allocator.

### `HostBufferAllocator`

```cpp
#include <holomq/host_buffer_allocator.hpp>
```

Custom host payload backing storage:

```cpp
#include <holomq/host_buffer_pool_memory.hpp>

using HostBufferAllocator = std::function<std::unique_ptr<HostBufferPoolMemory>(
    std::size_t payload_region_bytes, bool payload_pinned, unsigned int cuda_host_register_flags)>;

HostBufferAllocator default_host_buffer_allocator();
```

Pass a custom `HostBufferAllocator` as the optional third argument to `Context::create_host_buffer_pool` or the optional fourth argument to `Context::create_host_pinned_buffer_pool`. Omit it (or pass `{}`) to use the default SHM allocator (registers with CUDA when `payload_pinned` is true).

Custom allocators return `std::unique_ptr<DeviceBufferPoolMemory>` / `std::unique_ptr<HostBufferPoolMemory>` (public extension interfaces; included by the allocator headers above).

---

## GPU readiness (stream-associated export)

Cross-process GPU visibility is coordinated through `export_handle` / `import_buffer` using the export fence on **`BufferHandle`**.

| Responsibility | Owner |
|----------------|--------|
| Publisher → subscriber GPU visibility | **holomq** — `export_handle(stream)`, fence in handle, wait at `import_buffer(handle, stream)` |
| Stream A → stream B within one process | **Application** (CUDA events, ordering on streams) |
| Slot reuse vs in-flight GPU work | **holomq** — deferred SHM ref-count release on export/import streams |

**Contract:** Enqueue all GPU work that subscribers must see on **`stream`** before calling `export_handle(stream)` on that same stream. You do not need to wait for the GPU to finish first — stream ordering and the export fence handle completion at import time.

**Recommended:** `Buffer::export_handle(stream)` and `Context::import_buffer(handle, stream)`. **Advanced:** `enqueue_ready_set(stream)` then `export_handle_unsynchronized()` on the publisher; `import_buffer_unsynchronized` then `ConstBuffer::enqueue_ready_wait(stream)` on the subscriber.

Further reading: [`presentations/buffer-sync-deep-dive.md`](presentations/buffer-sync-deep-dive.md), [`design-notes/buffer-stream-associated-export.md`](design-notes/buffer-stream-associated-export.md), [`design-notes/vmm-stream-memops-safety.md`](design-notes/vmm-stream-memops-safety.md).

---

## Typical end-to-end flow

### Publisher

```cpp
auto ctx_or = ContextBuilder()
    .with_fd_pass_socket(path)  // or empty for auto /tmp path
    .create();
if (!ctx_or.has_value()) { /* handle error */ }
holomq::Context& ctx = ctx_or.value();

// One subscriber with current defaults: 8 producer loans + 8 unread + 8 borrowed.
constexpr std::uint32_t kExternalBufferSlots =
    holomq::k_max_in_flight + 2U * holomq::k_subscriber_queue_depth;
auto pool_or = ctx.create_device_buffer_pool(slot_bytes, kExternalBufferSlots);
if (!pool_or.has_value()) { /* handle error */ }
holomq::BufferPool& pool = pool_or.value();

auto publisher_or = ctx.publisher_builder("my_service")
    .payload_type<MyPayload>()
    .create();
if (!publisher_or.has_value()) { /* handle error */ }
holomq::Publisher<MyPayload>& publisher = publisher_or.value();

// Per frame:
auto msg_or = publisher.loan();
if (!msg_or.has_value()) { /* kBackpressured */ continue; }
auto msg = std::move(msg_or).value();

auto buf_or = pool.acquire();
if (!buf_or.has_value()) { /* kBackpressured */ continue; }
auto buf = std::move(buf_or).value();

// ... GPU work on buf.ptr() on stream ...

// Marks GPU work as ready for subscribers; pool slot reuse is deferred until stream finishes.
auto exported_or = buf.export_handle(stream);
if (!exported_or.has_value()) { /* handle error */ }
msg.payload().buffer_handle = exported_or.value();
publisher.send(std::move(msg), [buf = std::move(buf)]() {});
```

### Subscriber

```cpp
#include <holomq/wait_strategy.hpp>

auto ctx_or = ContextBuilder().with_vmm_fd_export(false).create();  // no local FD-pass server
if (!ctx_or.has_value()) { /* handle error */ }
holomq::Context& ctx = ctx_or.value();

auto subscriber_or = ctx.subscriber_builder("my_service")
    .payload_type<MyPayload>()
    .create();
if (!subscriber_or.has_value()) { /* handle error */ }
holomq::Subscriber<MyPayload>& subscriber = subscriber_or.value();

subscriber.set_wait_strategy(WaitStrategy{.mode = WaitMode::KernelWait});  // default TimedWait times out quickly
auto sample_or = subscriber.wait_and_take();
if (sample_or.has_value()) {
    const MyPayload& p = sample_or->payload();

    // Stream waits until the publisher's GPU work is done before the buffer can be used.
    auto buf_or = ctx.import_buffer(p.buffer_handle, stream);
    if (!buf_or.has_value()) { /* handle error */ }
    holomq::ConstBuffer buf = std::move(buf_or).value();

    // ... GPU work on buf.ptr() on stream ...
}
// Drop buf only after stream work completes; destroy all buffers before destroying the stream.
```

---

## Example payload shape

From the `holomq_main` demo (`examples/holomq_main/frame_payload.hpp`; application code in `holomq_main`, not the library namespace):

```cpp
namespace holomq_main {

struct FramePayload {
    std::uint64_t sequence{};
    std::uint32_t width{};
    std::uint32_t height{};
    holomq::BufferHandle buffer_handle{};  // opaque; filled by export_handle
};

} // namespace holomq_main
```

This illustrates the intended wire format: **metadata + handles**, not raw GPU pointers.

---
