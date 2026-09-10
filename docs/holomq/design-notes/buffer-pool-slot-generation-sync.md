# Buffer pool per-slot generation sync

**Driver requirement:** NVIDIA driver **555+** for VMM stream memops (see `vmm-stream-memops-safety.md`).

**Related code:**

- `src/buffer_pool/sync/*` — `cuStreamWriteValue64_v2` / `cuStreamWaitValue64_v2`
- `src/buffer_pool/generation_slot_array.*`, `slot_region.hpp` — generation slot views
- `src/buffer_pool/ref_count/shm_ref_count.*` — ref-count only (no generation field in SHM)
- `include/holomq/buffer.hpp` — `export_handle(stream)`
- `include/holomq/buffer_handle.hpp` — `ready_generation_`

**See also:** `device-slot-payload-generation-views.md`, `buffer-stream-associated-export.md`, `host-buffer-pools.md`

---

## Layout

Each device slot:

```text
[ uint64_t gpu_generation ][ user payload: buffer_bytes ][ stride padding ]
```

Slot stride is `round_up(8 + buffer_bytes, VMM granularity)`. `Buffer::ptr()` / `size_bytes()` cover **payload only** (slot base + 8).

POSIX SHM (ref-count only) has **no** generation field.

Device-pool creation initializes every GPU generation word to zero on a private CUDA stream and
waits for that stream before exposing the pool. Payload bytes are left untouched. This establishes
the same initial value as the pool's process-local generation counters without synchronizing
unrelated application streams.

Each slot keeps a **monotonic** per-slot export counter on the pool (`slot_generations_`). It is
**not** reset on acquire. The first export for a slot uses **G = 1**; each subsequent export on
that slot (after reuse) uses a larger **G**. `Buffer::Impl::lease_export_generation_` records the
value assigned on the current owner lease for `export_handle_unsynchronized`.

---

## API

| Call | Role |
|------|------|
| `Buffer::export_handle(stream)` | After prior ops on `stream`: increment lease export counter, stream memop write of **G**, embed **G** on `BufferHandle`. |
| `Context::import_buffer(h, stream)` | Import + stream memop wait when handle carries **G > 0**. |

Generation recording and waiting are internal to `Buffer` / `ConstBuffer` implementations;
applications use `export_handle` and `import_buffer`.

---

## Payload contract

Publish `BufferHandle` from `export_handle(stream)`. Subscribers call `Context::import_buffer(payload.buffer_handle, sub_stream)`.

Pinned **G** on the handle is the fence token for **this** export. Subscribers never read
generation from SHM or the pool.

---

## Host pools

Same **`export_handle` / `import_buffer`** contract. See [`host-buffer-pools.md`](host-buffer-pools.md) for devicemap interleaved vs split-gen VMM layouts.

---

## Hazards

- `ref_count == 0` does not imply GPU idle — deferred SHM ref release on export/import streams.
- Reused slots must use a **strictly larger** **G** than prior exports on that slot; subscribers
  wait with `GEQ` on the GPU generation word.
- `slot_generations_` is process-local; it shares the pool single-publisher-thread contract
  (see `buffer_pool_impl.hpp`). No lock is required when acquire, release, and export run on
  that thread.
- Recreate iceoryx2 pub/sub when `BufferHandle` or payload layout changes.

---

## Enqueue failure (fail closed)

`stream_write_generation` / `stream_wait_generation` return `false` when the stream or device
address is invalid, or when the driver rejects the memop enqueue.

| Failure site | Caller behavior |
|--------------|-----------------|
| `Buffer::enqueue_ready_set` / `export_handle` | `record_ready` returns `0` → `export_handle` returns **`ErrorCode::kFailure`**; no `BufferHandle` with a new fence is published. |
| `ConstBuffer::enqueue_ready_wait` / `Context::import_buffer` | Wait enqueue failure → `import_buffer` returns **`ErrorCode::kInvalidArgument`**; subscriber does not hold a view as if the wait succeeded. |

There is no silent fallback that continues without the export or import fence when **G > 0** is
required. Invalid streams are rejected before memop enqueue (`is_usable_stream`).

**Tests:** `CudaStreamGenerationTest.RejectsInvalidInputs`,
`CudaBufferImportSyncTest.UnsynchronizedImportThenDeferredWait`,
`CudaBufferImportSyncTest.UnsynchronizedExportHasNoExportFence`.
