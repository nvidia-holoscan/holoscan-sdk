# Ref-count and sample lifetime

**Related code:**

- `src/buffer_pool/buffer_pool_impl.hpp` — round-robin CAS acquire, `buffer_handle_released`
- `src/buffer_pool/ref_count/*` — POSIX SHM ref-count table
- `include/holomq/publisher.hpp` — `send(..., on_released)`
- `include/holomq/detail/publisher_sample_release_registry.hpp`

**See also:** [`publisher-sample-release-registry.md`](publisher-sample-release-registry.md),
[`buffer-pool-free-slot-accounting.md`](buffer-pool-free-slot-accounting.md),
[`buffer-gpu-completion-before-pool-reuse.md`](buffer-gpu-completion-before-pool-reuse.md),
[`PUBLIC_API.md`](../PUBLIC_API.md)

---

## What ref-count tracks

Each pool slot has a **POSIX SHM ref-count** (`ShmRefCountTable`). The count is the number
of live **`Buffer`** (owner) and **`ConstBuffer`** (import) handles for that slot **across
all processes**.

It does **not** count:

- Copies of `BufferHandle` embedded in message payloads
- iceoryx2 samples still in the subscriber queue

Ref-count answers: “how many participants still hold a GPU/SHM lease on this slot?”

---

## Owner acquire

Only the pool creator calls `BufferPool::acquire()`. Implementation scans slots round-robin
and CAS ref-count **0→1**. **`kBackpressured`** means every slot is in use (or deferred release
holds ref-count &gt; 0). There is **no** process-local free queue — see
[`buffer-pool-free-slot-accounting.md`](buffer-pool-free-slot-accounting.md).

---

## Remote import

1. Subscriber receives payload containing a `BufferHandle` (plain struct copy — no ref bump).
2. `Context::import_buffer` / `import_buffer_unsynchronized` constructs **`ConstBuffer`** →
   **`fetch_add(1)`** on the slot.
3. `~ConstBuffer` → deferred or immediate **`fetch_sub(1)`** (export/import stream may defer).

---

## Publisher sample lifetime

Owner `Buffer` must outlive the iceoryx2 loan until subscribers can take the sample. Pattern:

```cpp
publisher->send(std::move(msg), [buf = std::move(buf)]() { /* buf destroyed here */ });
```

The **`on_released`** callback runs when the transport releases the sample slot
(see [`publisher-sample-release-registry.md`](publisher-sample-release-registry.md)). That
decouples pool slot reuse from iceoryx2 queue depth without bumping ref-count on send.

---

## GPU completion vs ref-count

`ref_count == 0` does **not** mean GPU work finished. `Buffer::export_handle(stream)` and
subscriber view drop can **defer** SHM ref release until `stream` completes
([`buffer-gpu-completion-before-pool-reuse.md`](buffer-gpu-completion-before-pool-reuse.md)).

Generation sync (`ready_generation` on `BufferHandle`) protects **read-after-produce** across
processes; it is separate from ref-count lifetime.

---

## Process death

If a participant dies with a pending deferred release, ref-counts for affected slots can stay
non-zero permanently (see
[`buffer-gpu-completion-before-pool-reuse.md`](buffer-gpu-completion-before-pool-reuse.md)).
Do not expect automatic ref-count repair. Future releases may add a disconnection mechanism to detect
peer loss and recover or fail topics more effectively.

---

## Descriptor placement

`BufferHandle` descriptors live in application **payload** fields. Pools are external
`BufferPool` instances from `Context` factories — not in iceoryx2 user headers or publisher loans.
