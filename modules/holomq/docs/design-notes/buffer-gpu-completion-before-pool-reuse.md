# Buffer GPU completion and pool slot reuse

How holomq ties POSIX SHM ref-count release to CUDA stream completion so a pool slot is
not handed out again while GPU work queued before export (or on the import path) is still
in flight.

**Related code:**

- `src/buffer_pool/sync/deferred_ref_release.hpp`, `.cpp` — `defer_ref_release_on_stream`,
  `is_usable_stream`
- `src/buffer_pool/device/device_buffer_impl.cpp` — owner defer at `export_handle`, release in destructor
- `src/buffer_pool/device/device_const_buffer_impl.cpp` — subscriber defer at import / relay export
- `include/holomq/buffer.hpp`, `src/buffer_pool/buffer.cpp` — core `Buffer`
- `src/buffer_pool/buffer_pool_impl.hpp` — `acquire()` (SHM CAS `0→1`), `buffer_handle_released()` → `release_ref`
- `src/buffer_pool/ref_count/shm_ref_count.hpp` — cross-process slot accounting
- `include/holomq/publisher.hpp` — `send(OutboundMessage&&, on_released)` extends owner `Buffer` until iceoryx2 sample release

**See also:**

- [`ref-count-model.md`](ref-count-model.md) — SHM ref-count semantics and publisher release callback
- [`publisher-sample-release-registry.md`](publisher-sample-release-registry.md) — `on_released` callback registry
- [`buffer-pool-slot-generation-sync.md`](buffer-pool-slot-generation-sync.md) — per-slot generation sync
- [`buffer-stream-associated-export.md`](buffer-stream-associated-export.md) — stream-associated export contract

---

## SHM ref-count vs GPU work

A `Buffer` is a handle to one fixed-size slot in a `BufferPool`. Destroying the last handle
for a slot decrements the SHM ref-count; when the count reaches zero, `acquire()` may return
that slot index again.

Ref-count tracks **handle lifetime**, not CUDA stream or kernel completion. If application
code launches asynchronous GPU work on `buffer.ptr()` and drops the `Buffer` before that
work finishes, the pool can reuse the same device memory while a kernel is still running.

```text
  buf = pool.acquire()                    // ref_count: 0 → 1
  myKernel<<<..., stream>>>(buf.ptr(), …) // async — not waited on
  ~buf                                    // ref_count: 1 → 0  (slot “free” in SHM)
  buf2 = pool.acquire()                   // may return the same slot index
  otherKernel<<<..., stream>>>(buf2.ptr()) // races with myKernel on same VMM range
```

That is a **use-after-free on GPU memory**. CPU-side ref-counting does not detect it.

---

## Mechanisms that gate slot reuse

| Mechanism | What it gates | What it does not gate |
|-----------|---------------|------------------------|
| SHM `ref_count` | Slot not acquired while any owner `Buffer` or remote `ConstBuffer` exists | In-flight kernels after the last handle is dropped |
| `send(msg, on_released)` | Owner `Buffer` kept until iceoryx2 releases the sample bucket | GPU completion (orthogonal to transport lifetime) |
| `export_handle` + deferred ref release | SHM ref-count stays elevated until work queued on the bound stream before export completes | GPU work on other streams without explicit ordering |
| Generation sync (`wait_ready`) | Subscriber does not read device memory until producer work before export completes | Local slot reuse without export; work after export on other streams |

---

## Deferred ref-count release

On `export_handle(stream)` (owner) or import with `ready_generation > 0` (subscriber),
holomq records the stream. When the last local handle is destroyed (`~DeviceBufferImpl` or
`~DeviceConstBufferImpl`), holomq enqueues `cuLaunchHostFunc` on that stream. The host
callback calls `release_ref` with `{ ShmRefCountTable*, slot_index }`. Until the stream
reaches that point, the SHM ref-count stays elevated and `acquire()` cannot reuse the slot.

The `Buffer` / `ConstBuffer` object may be destroyed immediately. Only **ref-count release**
is deferred — not application use of `ptr()` on the destroyed handle.

### Publisher path

1. Enqueue subscriber-visible GPU work on `stream` before `export_handle(stream)` on that
   same stream.
2. `export_handle` records the generation fence and arms deferred ref release.
3. Use `Publisher::send(..., on_released)` so the owner `Buffer` outlives the transport
   sample independently of the ref-count gate.

### Subscriber path

1. Pass the `stream` used for subsequent GPU reads to `import_buffer`.
2. Destroying the view drops the C++ object immediately; ref-count falls only after the
   stream host function runs.
3. Destroy all buffer views before `cuStreamDestroy(stream)`.

### Errors and lifetime

Invalid streams are rejected at `export_handle` / `import_buffer`. If `cuLaunchHostFunc`
fails in a destructor on an otherwise valid stream, holomq synchronizes the bound stream then
calls `release_ref` (`defer_ref_release_on_stream` fallback; logged via the HoloMQ logger). If
`cuStreamSynchronize` also fails, holomq cannot establish GPU completion and keeps the ref-count
elevated to prevent unsafe slot reuse (logged at critical severity; the pinned slot can backpressure
the topic). Invalid mapping or stream state at release time also fails closed without releasing the
slot. Deferred callbacks pin the ref-count SHM mapping via
`ShmRefCountMappingPtr`; \ref BufferPool and \ref ConstBuffer may be destroyed after deferral is
enqueued.

---

## End-to-end slot lifecycle

Three independent gates must clear before a slot is reusable:

```text
  acquire()
      │
      ▼
  fill / kernels on stream
      │
      ├──► export_handle(stream)         ──► WriteValue(G) + defer ref release armed
      │
      └──► send(msg, [buf]{})            ──► iceoryx2 must not drop payload yet
              │
              ▼
         ~buf (may run immediately)
              │
              ▼
         cuLaunchHostFunc → release_ref when stream completes
              │
              ▼
         ref_count == 0 → acquire() eligible
```

| Gate | Clears when |
|------|-------------|
| IPC / message | iceoryx2 `SampleReleasedHandler` runs (`on_released`) |
| Cross-process GPU read | Subscriber `wait_ready` on import stream (generation sync) |
| Local pool reuse | Stream host function after `export_handle` / view drop |

---

## Subscriber views

Imported `ConstBuffer` views defer `release_ref` on the import stream when
`ready_generation > 0`. Bootstrap imports (`ready_generation == 0`) release immediately.
Relay `export_handle` on a view updates the stored release stream.

---

## Limitations

- **Mapping lifetime:** Deferred-release callbacks hold a `ShmRefCountMappingPtr` that keeps the
  ref-count SHM segment mapped until the host function runs. \ref BufferPool and imported
  \ref ConstBuffer views may be destroyed after deferral is enqueued; the mapping stays alive via
  the callback pin.
- **Stream lifetime:** Pending host functions must drain before \c cuStreamDestroy on the bound
  stream; destroying the stream while deferred releases are still queued is undefined. For
  application-owned streams, use \c holomq::cuda::Stream (internal helper, mirrors
  \c holomq::cuda::Context), which synchronizes before destroy.
- **Programming errors:** A null \c ShmRefCountMappingPtr passed to \c defer_ref_release_on_stream
  (or enqueued into a host function without a mapping pin) is treated as a fatal programming error
  (logged, then \c std::abort).
- **ABI:** \c defer_ref_release_on_stream is exported with
  \c std::shared_ptr<ShmRefCountMapping>; downstream binaries must relink when the signature changes.
- **Multi-stream:** Buffers touched from multiple `CUstream`s without `cuStreamWaitEvent`
  (or equivalent) are outside the single-stream export/import contract.
- **Effective capacity:** Deferred releases hold ref-counts until streams complete, so
  **`BufferPool::acquire()`** may return **`ErrorCode::kBackpressured`** while slots are waiting on
  GPU rather than actively leased.
- **Process death:** If a topic participant dies while a deferred ref-count release is still
  pending (for example with a `cuLaunchHostFunc` queued but never executed), the SHM ref-count
  for that slot can remain elevated and the publisher may block in `acquire()` waiting for a
  slot that will not return to zero in this process lifetime. EA1 does not implement crash
  recovery or orphaned ref-count repair. Unexpected process death on a topic is a **fatal topic
  error**: restart the topic (all participants) rather than expecting in-place recovery. A
  participant **disconnection** mechanism may be added in a future release to detect loss of
  peers and handle these cases more effectively; that is out of scope for EA1.
