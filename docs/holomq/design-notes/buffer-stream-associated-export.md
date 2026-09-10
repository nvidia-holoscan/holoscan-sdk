# Stream-associated buffer export and import

Cross-process GPU visibility contract: publisher `export_handle(stream)` and subscriber
`import_buffer(handle, stream)`.

**Related code:**

- `include/holomq/buffer.hpp` — `export_handle`
- `include/holomq/buffer_handle.hpp` — `ready_generation`
- `include/holomq/context.hpp` — `import_buffer`, `import_buffer_unsynchronized`
- `examples/holomq_main/` — external pool + `BufferHandle` in payload

**See also:** [`buffer-pool-slot-generation-sync.md`](buffer-pool-slot-generation-sync.md),
[`buffer-gpu-completion-before-pool-reuse.md`](buffer-gpu-completion-before-pool-reuse.md)

---

## Responsibility split

| Concern | Owner |
|---------|--------|
| Publisher → subscriber GPU visibility | holomq — `export_handle(stream)`, fence on `BufferHandle`, wait at `import_buffer` |
| Stream A → stream B within one process | Application (CUDA events, ordering on streams) |
| Slot reuse vs in-flight GPU work | holomq — deferred SHM ref-count release on export/import streams |

**Ordering:** Enqueue all subscriber-visible GPU work on `stream` before calling
`export_handle(stream)` on that same stream. Stream ordering and the export fence handle
completion at import time — you do not need to synchronize the GPU before calling
`export_handle`.

---

## API

| Call | Role |
|------|------|
| `BufferPool::acquire()` | Owner lease; no stream argument |
| `Buffer::export_handle(stream)` | Enqueue ready-generation set on `stream`; return `BufferHandle` (recommended) |
| `Buffer::enqueue_ready_set(stream)` | Enqueue ready-generation set on `stream` without returning a handle |
| `Buffer::export_handle_unsynchronized()` | Return `BufferHandle` using the current export fence state on this lease |
| `Context::import_buffer(h, stream)` | Import + wait export fence when `ready_generation > 0` |
| `Context::import_buffer_unsynchronized(h)` | Import only; call `ConstBuffer::enqueue_ready_wait` separately |
| `ConstBuffer::enqueue_ready_wait(stream)` | Enqueue ready-generation wait; arms deferred ref release |

`export_handle(stream)` composes `enqueue_ready_set(stream)` and `export_handle_unsynchronized()`.
`import_buffer` composes the unsynchronized import path and `enqueue_ready_wait` when the
handle carries an export fence.

---

## Export fence on `BufferHandle`

`ready_generation` is set by `export_handle` and honored at import. Value `0` means no
generation wait. Applications do not read or set it directly.

---

## Examples

**Publisher:**

```cpp
auto buf_or = pool.acquire();
if (!buf_or) { /* kBackpressured */ return; }
auto& buf = buf_or.value();
cuMemsetD8Async(buf.ptr(), …, pub_stream);
const auto exported_or = buf.export_handle(pub_stream);
if (!exported_or) { /* handle error */ return; }
payload.buffer_handle = exported_or.value();
```

**Subscriber:**

```cpp
auto buf_or = ctx.import_buffer(payload.buffer_handle, sub_stream);
if (!buf_or) { /* handle error */ return; }
holomq::ConstBuffer buf = std::move(buf_or).value();
```

**Subscriber (split import and wait):**

```cpp
auto buf_or = ctx.import_buffer_unsynchronized(payload.buffer_handle);
if (!buf_or) { /* handle error */ return; }
holomq::ConstBuffer buf = std::move(buf_or).value();
if (!buf.enqueue_ready_wait(sub_stream)) { /* handle error */ }
```

**Publisher (split enqueue and export):**

```cpp
auto buf_or = pool.acquire();
if (!buf_or) { /* kBackpressured */ return; }
auto& buf = buf_or.value();
cuMemsetD8Async(buf.ptr(), …, pub_stream);
if (!buf.enqueue_ready_set(pub_stream)) { /* handle error */ }
const auto exported_or = buf.export_handle_unsynchronized();
if (!exported_or) { /* handle error */ return; }
payload.buffer_handle = exported_or.value();
```

---

## Hazards

- GPU work on streams other than the export stream without explicit ordering.
- `export_handle_unsynchronized` without `enqueue_ready_set` when cross-process GPU visibility is required.
- `import_buffer_unsynchronized` without `enqueue_ready_wait` before GPU reads when the
  handle carries an export fence.
- `ref_count == 0` does not imply GPU idle — deferred ref release on export/import streams.
