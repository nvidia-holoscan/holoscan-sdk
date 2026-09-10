# Cross-process GPU writes and subscriber buffer access

How holomq exposes GPU memory across processes: read-only subscriber imports, sync
mechanisms, and the limits of writable shared slots.

**Related code:**

- `include/holomq/buffer.hpp` — owner `Buffer` (`void* ptr()`)
- `include/holomq/const_buffer.hpp` — subscriber `ConstBuffer` (`const void* ptr()`)
- `include/holomq/context.hpp` — `import_buffer`, pool factories
- `src/buffer_pool/ref_count/shm_ref_count.*` — POSIX SHM ref-count
- `src/buffer_pool/sync/*` — generation sync (`export_handle` / `enqueue_ready_wait`)

**See also:** [`ref-count-model.md`](ref-count-model.md), [`buffer-stream-associated-export.md`](buffer-stream-associated-export.md), [`buffer-pool-slot-generation-sync.md`](buffer-pool-slot-generation-sync.md)

---

## Read-only subscriber imports

| Role | Type | How obtained | Payload pointer |
|------|------|--------------|-----------------|
| Publisher (pool owner) | `Buffer` | `BufferPool::acquire()` | `void* ptr()` — mutable |
| Subscriber (remote) | `ConstBuffer` | `Context::import_buffer` / `import_buffer_unsynchronized` | `const void* ptr()` — read-only in the API |

Subscribers do **not** receive a mutable `Buffer` lease on another process's pool slot. The
pub/sub pattern is **one writer (publisher), many readers (subscribers)**.

```text
  Publisher                         Shared GPU slot              Subscribers
  Buffer::acquire()                 (same VMM mapping            ConstBuffer via
  write on stream                   per import)                  import_buffer
       │                                    │                         │
       └── export_handle ──► payload ───────┼────► read after wait ───┘
```

| Mechanism | Role |
|-----------|------|
| SHM ref-count | Slot lifetime across owner and imported views ([`ref-count-model.md`](ref-count-model.md)) |
| `export_handle` / `import_buffer` | Subscriber **reads** only after publisher GPU work before export is visible |
| Deferred ref-count release | Slot not reused locally until bound stream completes ([`buffer-gpu-completion-before-pool-reuse.md`](buffer-gpu-completion-before-pool-reuse.md)) |
| `Publisher::send(..., on_released)` | Owner `Buffer` outlives the iceoryx2 sample ([`publisher-sample-release-registry.md`](publisher-sample-release-registry.md)) |

**API vs hardware:** `ConstBuffer::ptr()` is `const` in C++; CUDA does not mark the imported
VMM mapping read-only in hardware. Applications must not launch write kernels against
imported broadcast memory. holomq does not enforce write exclusion on the GPU.

Subscribers that need writable GPU output should use a **local** `BufferPool` on their own
`Context`.

---

## Sync primitives and their limits

| Mechanism | What it provides | What it does not provide |
|-----------|------------------|---------------------------|
| Generation sync (`WriteValue` / `WaitValue`) | Read-after-produce across processes | Exclusive write access |
| SHM ref-count | Lease lifetime for owner and import views | Write mutual exclusion |
| CUDA IPC events | Cross-process stream ordering | Writer turn-taking |
| Per-process stream ordering | Order within one process | Cross-process write serialization |

Generation sync is a **read barrier after produce**. Ref-count tracks how many handles hold
a slot, not which writer may modify it.

---

## Writable shared slots

The public API does not support subscribers writing into a publisher's imported slot.
Concurrent GPU writes to the same physical memory from multiple processes would be a data
race unless an explicit coordination protocol is added.

Common coordination patterns (application- or library-level, outside holomq's subscriber API):

| Pattern | Idea | Notes |
|---------|------|-------|
| Read-only imports | Publisher writes; subscribers read | holomq default |
| Host SHM lock per slot | `atomic` lock word in ref-count SHM + stream sync before unlock | Fits existing SHM infrastructure |
| Device spinlock | `atomicCAS` in shared GPU memory | Kernel coordination; crash recovery needed |
| Partitioned regions | Each writer owns disjoint payload subranges | Layout contract in payload |

Host-side per-slot locking paired with **GPU completion before unlock** is a common
coordination pattern on POSIX SHM. The public API does not expose writable shared imports;
such use would need explicit acquire/release rules, generation-sync ordering, and crash/teardown
handling beyond `ConstBuffer`.

See [`PUBLIC_API.md`](../PUBLIC_API.md) for the supported subscriber path (`ConstBuffer`,
`import_buffer`).

---

## Summary

| Topic | Behavior |
|-------|----------|
| Subscriber import | Read-only `ConstBuffer` |
| Publisher write | Mutable `Buffer` on owner `BufferPool` |
| Multi-subscriber reads | After `import_buffer` / generation wait |
| Multi-subscriber writes to same slot | Not supported in the public API |
| Generation sync | Read-after-produce, not writer exclusion |
