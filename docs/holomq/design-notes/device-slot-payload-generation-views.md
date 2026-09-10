# Device slot payload and generation views

Internal layout model for addressing **user payload** and **GPU generation words** within a
pool slot. Export fence (`ready_generation`) lives on **`BufferHandle`**, not in wire POD or
ref-count SHM.

**Driver requirement:** NVIDIA driver **555+** for VMM stream memops on generation words
(see [`vmm-stream-memops-safety.md`](vmm-stream-memops-safety.md)).

**Related code:**

- `src/buffer_pool/slot_region.hpp` — `SlotRegion`, region factories
- `src/buffer_pool/payload_slot_array.hpp` — `PayloadSlotArray` / `PayloadSlotRef`
- `src/buffer_pool/generation_slot_array.hpp` — `GenerationSlotArray` / `GenerationSlotRef`
- `src/buffer_pool/generation_slot_array.cpp` — stream memop `write` / `wait`
- `src/buffer_pool/buffer_impl.hpp`, `const_buffer_impl.hpp` — shared owner/subscriber logic
- `src/buffer_pool/device/device_buffer_impl.*`, `device_const_buffer_impl.*`
- `src/buffer_pool/host/host_buffer_impl.*`, `host_const_buffer_impl.*`

**See also:** [`buffer-pool-slot-generation-sync.md`](buffer-pool-slot-generation-sync.md),
[`host-buffer-pools.md`](host-buffer-pools.md), [`buffer-stream-associated-export.md`](buffer-stream-associated-export.md)

---

## Role in the buffer pool

Each slot exposes two logical regions:

| Region | Size | Used for |
|--------|------|----------|
| Generation word | `uint64_t` (8 bytes) | `cuStreamWriteValue64` / `cuStreamWaitValue64` targets |
| User payload | `buffer_bytes` | `Buffer::ptr()` / `ConstBuffer::ptr()`, kernels, memcpy |

`Buffer::Impl` and `ConstBuffer::Impl` own a **`PayloadSlotArray`**, a **`GenerationSlotArray`**, and
a lease **`index_`**. Public `ptr()` is `payload_slots_[index_].ptr()`. Export and import paths
bind **slot regions** once per lease; pool-specific types (`DeviceBufferImpl`,
`HostConstBufferImpl`, etc.) supply regions and ref-count / export behavior.

All slot-array types are **internal** (`src/buffer_pool/`); they are not public API.

---

## Physical layouts

### Interleaved (device VMM, host devicemap interleaved)

One allocation per pool (or per host payload SHM segment). Each slot:

```text
slot[i] = [ uint64_t gpu_generation ][ user payload: buffer_bytes ][ stride padding ]
```

```text
slot_stride = round_up(8 + buffer_bytes, alignment)
gen_ptr(i)     = base + i * slot_stride
payload_ptr(i) = base + i * slot_stride + 8
```

Factory: `make_interleaved_slot_regions(base, slot_stride_bytes, buffer_bytes)`.

Host devicemap interleaved uses `make_host_devicemap_interleaved_slot_regions`: same geometry,
but the generation region’s `base` is the **device pointer** from `cuMemHostGetDevicePointer` on
the slot-0 generation word (`CU_MEMHOSTREGISTER_DEVICEMAP`).

### Split (host pageable / pinned without devicemap)

Payload and generation are **separate** allocations with independent bases and strides. Slot `i`
is paired by index only:

```text
payload_ptr(i) = payload_base + i * payload_stride
gen_ptr(i)     = generation_base + i * generation_stride
```

Factory: `make_split_slot_regions(payload_base, payload_stride, generation_base, generation_stride, buffer_bytes)`.

Generation pool is device-accessible VMM imported via fd-pass (see [`host-buffer-pools.md`](host-buffer-pools.md)).

---

## Internal types

### `SlotRegion`

Non-owning descriptor: `base`, `stride`, `element_size`, `slot_offset`. Address:

```text
element_ptr(i) = base + i * stride + slot_offset
```

### Slot arrays

| Type | `operator[]` returns | Operations |
|------|----------------------|------------|
| `PayloadSlotArray` | `PayloadSlotRef` | `ptr()`, `size_bytes()` |
| `GenerationSlotArray` | `GenerationSlotRef` | `write(stream, value)`, `wait(stream, generation)` |

`GenerationSlotRef` delegates to `cuda_stream_generation.*` (stream memory operations only).

### `SlotRegions`

```cpp
struct SlotRegions {
    SlotRegion payload;
    SlotRegion generation;
};
```

`Buffer::Impl` / `ConstBuffer::Impl` construct from `SlotRegions` + `index_`.

---

## Generation values (three roles)

Do not conflate these:

| Role | Where | Who uses it |
|------|-------|-------------|
| **Export fence G** | `BufferHandle` (`ready_generation`) | Subscriber `import_buffer` / `enqueue_ready_wait` |
| **GPU generation word** | Device-accessible slot address | `GenerationSlotRef::write` / `wait` on streams |
| **Per-slot monotonic counter** | `slot_generations_` on `BufferPool::Impl` | Owner `next_slot_generation` assigns G on each `record_ready` |
| **Per-lease assigned G** | `lease_export_generation_` on `Buffer::Impl` | Cached for `export_handle_unsynchronized` on this lease |

POSIX SHM ref-count has **no** generation field.

Subscribers use **`ready_generation` from the handle only** — not SHM, not `slot_generations_`.
Owner `record_ready` calls `next_slot_generation`, writes G to the GPU word, caches G on the
lease, and embeds G on the handle:

```cpp
const std::uint64_t generation = next_export_generation();  // pool ++slot_generations_[index]
generation_slots_[index_].write(stream, generation);
lease_export_generation_ = generation;
// G returned for BufferHandle
```

Subscriber `wait_ready` calls `generation_slots_[index_].wait(stream, pinned_generation)`.

---

## Owner vs subscriber binding

| Side | Typical `base` | `index_` | Example |
|------|----------------|----------|---------|
| Owner device | Pool VMM base | Leased slot index | `DeviceBufferImpl` |
| Subscriber device | Imported slot device pointer | Handle slot index | `DeviceConstBufferImpl` |
| Owner / subscriber host (interleaved) | Payload SHM pool base | Handle slot index | `HostBufferImpl` / devicemap import |
| Owner / subscriber host (split) | Separate payload + gen bases | Handle slot index | `make_split_slot_regions` |

Device owner: `make_interleaved_slot_regions(pool.base_ptr(), stride, buffer_bytes)` + slot
`index` from `acquire()`.

Device subscriber: VMM pool imported once per `(pool_cookie, fd_pass_socket_path)`; slot device pointer computed from
handle `slot_index`; regions bound with `make_interleaved_slot_regions(slot_base, stride, buffer_bytes)`.

Host layouts and fd-pass rules: [`host-buffer-pools.md`](host-buffer-pools.md).

---

## Layering

```text
Buffer::Impl / ConstBuffer::Impl
├── PayloadSlotArray
├── GenerationSlotArray
├── index_
├── lease_export_generation_   (owner only; G assigned on this lease)
├── ptr() → payload_slots_[index_]
├── record_ready / wait_ready  (generation_slots_[index_])

DeviceBufferImpl / HostBufferImpl
└── pool lease, export_handle, deferred ref release

DeviceConstBufferImpl / HostConstBufferImpl
└── SHM ref-count, import caches, enqueue_ready_wait, relay export_handle
```

`DeviceBufferImpl` does not reimplement `ptr()` or generation sync — it passes `SlotRegions`
into the shared `Buffer::Impl` base.

---

## Handle wire fields

`DeviceBufferHandleData` / `detail::host::BufferHandleData` carry layout metadata (`buffer_bytes`,
`slot_stride_bytes`, pool cookies, fd-pass paths). **`ready_generation`** is on the public
`BufferHandle` object, not inside the wire POD union.

Import validates `buffer_bytes` against stride (generation prefix + payload must fit in the slot).

---

## Tests

- `tests/unit/test_slot_region.cpp` — interleaved and split address math
- `tests/unit/test_cuda_pool.cpp`, `test_cuda_vmm_import.cpp` — device pool round-trip
- `tests/unit/test_cuda_buffer_import_sync.cpp` — generation wait at import
- `tests/unit/test_cuda_host_pool.cpp` — host layout variants
