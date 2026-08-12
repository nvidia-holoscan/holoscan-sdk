# Multi-publisher: shared SHM vs per-publisher GPU pools

How host payload SHM, device pools, and subscriber caches behave when several publishers
share one iceoryx2 service.

**Related code:**

- `include/holomq/buffer_pool.hpp`, `src/buffer_pool/device/device_buffer_pool_impl.*`
- `src/vmm_pool_import_cache.hpp` — `entries_` keyed by `pool_cookie`
- `examples/holomq_main/` — external device pool + `BufferHandle` in payload

**See also:** [`ref-count-model.md`](ref-count-model.md),
[`vmm-pool-import-cache-entry-eviction.md`](vmm-pool-import-cache-entry-eviction.md)

---

## Layer summary

On one iceoryx2 publish–subscribe **service** (same topic name):

| Layer | Shared across publishers? | Notes |
|-------|---------------------------|--------|
| Host payload (`Payload` in SHM) | **Yes** | One service data segment; publishers loan buckets from the same pool |
| Device `BufferPool` | **No** (typical) | One pool per publisher process; only the pool creator calls `acquire()` |
| GPU memory (VMM) | **No** | `buffer_count` slots per device pool instance |
| Subscriber import cache | **One per `Context`**, **one entry per `pool_cookie`** | fd-pass once per distinct GPU pool |
| Ref-count SHM cache | **One per subscriber**, **one entry per `(owner_pid, shm_ref_count_id)`** | `shm_open` + mmap once per ref-count segment |
| Cross-process slot leases | **POSIX SHM ref-count** per pool | Remotes use `Context::import_buffer`; no foreign `acquire()` |

iceoryx2 zero-copy applies to **CPU/SHM** payloads. GPU frames live in CUDA VMM and are
shared via fd-pass and `BufferHandle` in the payload.

---

## Multiple publishers on one service

A subscriber opens the **service**, not a single publisher port. Every publisher on that
service may send samples (up to `max_publishers`).

Each received sample is self-consistent: host payload and embedded `BufferHandle` arrive
together. The application calls `Context::import_buffer` when it needs GPU access.

The holomq public API does not expose per-sample publisher identity. Applications that
need to distinguish publishers can encode identity in the payload or use backend-specific
extensions.

---

## Publisher: external pool and payload descriptors

Typical flow (`holomq_main`):

1. Publisher process creates a **device `BufferPool`** via `Context::create_device_buffer_pool`.
2. App **`acquire()`s** a buffer, fills GPU data on a stream, and sets **`payload.buffer_handle`**
   from **`buf.export_handle(stream)`**.
3. **`send()`** publishes the payload; owner **`Buffer`** lifetime is tied to sample
   release via `on_released` when needed ([`publisher-sample-release-registry.md`](publisher-sample-release-registry.md)).

Slot choice is **`acquire()`** in the pool-creator process only.

---

## Subscriber caches

**GPU import (`VmmPoolImportCache`):**

```cpp
std::unordered_map<std::uint64_t, VmmPoolImportEntry> entries_;
```

- **Key:** `pool_cookie` from the handle wire data.
- **Value:** one CUDA import (fd-pass once per cookie) for all `slot_index` values on that pool.

**Ref-count SHM (`RefCountSegmentCache`)** is keyed by **`owner_pid` + `shm_ref_count_id`**.

| Scenario | Import cache entries |
|----------|---------------------|
| One publisher, one device pool | One `pool_cookie` → one entry |
| Two publisher processes, two pools | Two entries in the same `Context` cache |
| Publisher restart (new VMM reservation) | New `pool_cookie` → new entry; old entry retained until cache destroy ([`vmm-pool-import-cache-entry-eviction.md`](vmm-pool-import-cache-entry-eviction.md)) |

This is **one entry per distinct GPU pool instance**, not one entry per SHM sample.

---

## GPU memory scales with publisher count

Each publisher that owns a device **`BufferPool`** allocates **`buffer_count × buffer_bytes`**
(plus VMM alignment) for its own concurrency.

**Example:** two GPU publisher processes, each `buffer_count = 10`:

| Resource | Approximate total |
|----------|-------------------|
| GPU buffers | 20 (10 per pool × 2) |
| GPU bytes | `20 × buffer_bytes` (+ per-pool VMM overhead) |
| Host SHM | Shared segment; outstanding loans per iceoryx2 limits |

**Why:** fd-pass export and SHM ref-counts are tied to each pool creator. Each published
sample stays coherent — descriptor fields match that publisher's pool.

**Mitigations in application design:**

- One GPU publisher per topic; others host-only or relay descriptors.
- Smaller per-publisher `buffer_count` when N publishers are expected.
- Subscriber `Context` teardown to drop accumulated import-cache entries.

---

## Central topic pool (other layout)

Some deployments use a **dedicated process** that owns one CUDA VMM pool per topic, with
centrally leased `slot_index` values over a control plane. That caps GPU memory at topic
concurrency instead of publisher count, but requires coordinated leasing — independent
per-process `acquire()` on shared indices is unsafe. holomq uses **per-publisher pools**
with `BufferHandle` in payload and `VmmPoolImportCache` on subscribers.

---

## Normal operation

- Same SHM service, different GPU pools per publisher process — descriptors disambiguate pools.
- Host buckets reused over time — new samples carry current descriptor and content.
- One received sample — host payload and GPU view come from the **same** message's handle fields.
