# Buffer pool: owner acquire and free-slot accounting

How the pool owner finds a free slot on `acquire()`. POSIX SHM ref-count is the only
source of truth; there is no process-local free list.

**Related code:**

- `src/buffer_pool/buffer_pool_impl.hpp` — `acquire()`, `acquire_cursor_`, `buffer_handle_released()`
- `src/buffer_pool/buffer_pool_impl.cpp` — round-robin CAS acquire loop
- `src/buffer_pool/ref_count/shm_ref_count.hpp` — `try_acquire_slot`, `release_ref`, `add_ref`
- `src/buffer_pool/device/device_buffer_impl.cpp`, `host/host_buffer_impl.cpp` — owner release
- `src/buffer_pool/device/device_const_buffer_impl.cpp`, `host/host_const_buffer_impl.cpp` — remote release

**See also:** [`ref-count-model.md`](ref-count-model.md)

---

## Single source of truth

Each slot has a ref-count in POSIX SHM (`ShmRefCountTable`), updated by every process that
holds an owner `Buffer` or imported `ConstBuffer` for that slot.

| Ref-count | Meaning |
|-----------|---------|
| `0` | Slot has no live handles; owner may attempt `acquire()` (CAS `0→1`) |
| `≥ 1` | Slot is leased; `acquire()` skips it until count returns to zero |

There is **no** `free_buffer_indices_` queue or other publisher-local free list. A slot freed
by a subscriber’s last `fetch_sub` is visible to the publisher on the next `acquire()` scan.

---

## Owner `acquire()`

`BufferPool::Impl::acquire()` walks slot indices round-robin starting at `acquire_cursor_`.
For each candidate it calls `try_acquire_slot(index)` (atomic CAS ref-count `0→1`). On
success it advances `acquire_cursor_` past that index and returns a `Buffer`. If every slot
fails CAS, the implementation returns `std::nullopt`; the public **`BufferPool::acquire()`**
maps that to **`ErrorCode::kBackpressured`** (`expected<Buffer, Error>`).

```cpp
for (offset = 0; offset < buffer_count; ++offset) {
    index = (acquire_cursor_ + offset) % buffer_count_;
    if (try_acquire_slot(index)) { /* return buffer_at_index(index) */ }
}
return std::nullopt;  // impl only; public acquire() → kBackpressured
```

**Cost:** `O(buffer_count)` atomic attempts in the worst case. Typical pool sizes (4–32) keep
this cheap relative to transport and GPU work.

**Threading:** `acquire`, owner release, and `next_slot_generation` (owner export) are not
synchronized; the pool owner should call them from one thread (typical single publisher graph).

---

## Release paths

| Role | On last handle destroy | Effect on acquire |
|------|------------------------|-------------------|
| Owner `Buffer` | `buffer_handle_released` → `release_ref` (`fetch_sub`) | Slot eligible when count hits `0` |
| Remote `ConstBuffer` | `release_ref` on mapped SHM table | Same — no publisher-side bookkeeping |

Deferred ref-count release on an export/import stream delays `release_ref` until the bound
CUDA stream completes ([`buffer-gpu-completion-before-pool-reuse.md`](buffer-gpu-completion-before-pool-reuse.md)).
Until then the slot stays non-zero in SHM and `acquire()` will not return it.

`Publisher::send(..., on_released)` keeps an owner `Buffer` alive until iceoryx2 releases the
sample; that is transport lifetime, not an extra ref-count bump on send
([`publisher-sample-release-registry.md`](publisher-sample-release-registry.md)).

---

## Why SHM-only acquire

A process-local free queue of indices can disagree with SHM ref-count after cross-process
import/export: subscriber releases update SHM only, so the publisher queue could show a slot
as free while ref-count is still non-zero (or the reverse).

Round-robin scan + CAS uses SHM as the single source of truth: any process that drops the
last handle makes the slot visible on the publisher's next `acquire()` attempt.

For very large `buffer_count`, a cross-process structure in SHM (bitset, lock-free stack) could
reduce scan cost at the expense of extra SHM bookkeeping.

---

## Tests

`tests/unit/test_shm_ref_count.cpp` covers `try_acquire_slot` / `release_ref` CAS behavior.
CUDA pool integration tests exercise full acquire → export → import → release → re-acquire
loops across processes.
