# Host buffer pools

**Related code:**

- `include/holomq/detail/buffer_pool/host/handle_data.hpp` — wire layout
- `include/holomq/host_buffer_import.hpp` — reusable exact-slot import lease
- `src/buffer_pool/host/host_buffer_pool_impl.*`, `host_const_buffer_impl.*`
- `src/context.cpp` — factory dispatch
- `examples/holomq_main/main.cpp` — `--pool-type`, `--host-register-flags`

**See also:** [`PUBLIC_API.md`](../PUBLIC_API.md) (pool factories),
[`device-slot-payload-generation-views.md`](device-slot-payload-generation-views.md),
[`buffer-pool-slot-generation-sync.md`](buffer-pool-slot-generation-sync.md),
[`holomq-vs-holoscan-memory-types.md`](holomq-vs-holoscan-memory-types.md)

---

## Summary

Host pools use **POSIX SHM** for payload bytes and the same **generation sync protocol**
as device pools (`Buffer::export_handle(stream)`, `Context::import_buffer(handle, stream)`).
Ref-count SHM is separate from payload SHM (same model as device pools).

---

## Three layouts

| Factory | Payload | Generation word | Publisher fd-pass |
|---------|---------|-----------------|-------------------|
| `create_host_buffer_pool` | Pageable SHM (mmap) | Split-gen **VMM pool** | **Required** (`with_vmm_fd_export(true)`) |
| `create_host_pinned_buffer_pool` + `CU_MEMHOSTREGISTER_DEVICEMAP`, fd-pass export enabled | Pinned SHM, payload-only stride | Split-gen **VMM pool** | Enabled |
| `create_host_pinned_buffer_pool` + `CU_MEMHOSTREGISTER_DEVICEMAP`, fd-pass export disabled | Pinned SHM, devicemap interleaved | Interleaved in payload SHM | **Not required** |
| `create_host_pinned_buffer_pool` without `DEVICEMAP` | Pinned SHM, payload-only stride | Split-gen **VMM pool** | **Required** |

A `DEVICEMAP` pool prefers the split-gen VMM pool whenever fd-pass export is available. A consumer
waits for each ready generation on its CUDA stream (`cuStreamWaitValue64`). A wait on a word in host
memory polls system memory, and while it is parked it holds a GPU hardware work queue. Streams that
the driver maps to the same queue stall behind it, including the streams that carry buffer release
host functions. Which streams share a queue follows stream creation order, so a saturated
cross-process pinned-host Tensor channel ran anywhere from about 12k to 63k messages per second from
run to run, mostly at the low end. With device-resident generation words most runs reach about
60k. Pageable host pools always use split-gen, so their stream-ordered waits are on device-resident
generation words and never take the host-memory polling path.

Importers select the layout from the handle: devicemap interleaved means `DEVICEMAP` set *and* a
zero `gen_pool_cookie`.

Wire layout is in `detail::host::BufferHandleData`:

- **`payload_pinned`** — remotes call `cuMemHostRegister` when non-zero
- **`host_register_flags`** — passed to `cuMemHostRegister` (`0` = CUDA default register)
- Split-gen fields: **`gen_pool_cookie`**, **`fd_pass_socket_path`**, gen pool geometry

Generation words are **not** stored in ref-count SHM.

---

## Cross-process attach

- **Payload:** derived SHM name from `owner_pid` + `shm_pool_id` (`HostBackingSegmentCache`)
- **Split-gen VMM:** fd-pass + `VmmPoolImportCache::import_pool_base` (same path as device pools)
- **Devicemap interleaved** (`DEVICEMAP` and no split-gen pool): generation visible via
  `cuMemHostGetDevicePointer` on payload mapping

Validate stream memops on target platforms before claiming generation sync support
([`vmm-stream-memops-safety.md`](vmm-stream-memops-safety.md)).

## Reusable exact-slot imports

`Context::import_buffer_unsynchronized` creates an independent `ConstBuffer` implementation for
each imported message. That is the correct general path when successive handles may name arbitrary
pools or slots. A bounded runtime commonly knows the producer pool capacity and can retain one
subscriber control per slot instead.

`Context::import_host_buffer_unsynchronized(handle, HostBufferImport&)` supplies that narrower
path. The first import performs the normal handle, geometry, ref-count SHM, payload SHM, and split
generation VMM validation and retains the resolved mappings. `HostBufferImport::release()` drops
only the current remote slot lease. A later import must match the original memory kind, owner,
payload/ref-count/generation pool identities, complete geometry, fd-pass endpoint, and slot index;
it then updates the ready-generation token and adds a fresh remote ref count without consulting the
three mapping caches again.

This preserves the safety boundary:

- retained mappings are not a retained message lease;
- every active message interval executes one `add_ref` / `release_ref` pair;
- a different pool or slot fails closed rather than retargeting the cached view;
- a stream-associated release remains deferred through the existing callback path; and
- the owning `Context` must remain alive while a reusable import is active because its host and VMM
  caches own the retained mappings.

The reusable object is intentionally serialized and host-only. Device imports and callers without
an endpoint-sized slot table continue to use `ConstBuffer`.

For an endpoint-sized table, the `span<HostBufferImport>` overload reads the opaque handle's slot
inside HoloMQ, selects the corresponding control in O(1), and returns the selected index. This keeps
wire-format access out of the runtime layer and avoids a capacity-dependent scan on every receive.

Some receivers know a concurrent-borrow bound but not the producer's physical slot count. The
`rebind_host_buffer_unsynchronized` overload lets any inactive control retarget to another slot in
the same already validated pool. It updates the slot view and ready generation and acquires the new
slot lease without revisiting the mapping caches. Pool identity, memory kind, complete geometry,
FD-pass endpoint, and ref-count segment must remain identical.

---

## Demo matrix (`holomq_main`)

| `--pool-type` | Factory | Typical flags |
|---------------|---------|---------------|
| `device` (default) | `create_device_buffer_pool` | N/A |
| `host-pageable` | `create_host_buffer_pool` | N/A |
| `host-pinned` | `create_host_pinned_buffer_pool` | `--host-register-flags` (`portable`, `devicemap`, `portable-devicemap`, or pageable `0`) |

Publisher needs `ContextBuilder::with_vmm_fd_export(true)` when fd-pass is required (device,
pageable host, pinned split-gen). `DEVICEMAP` pools use split-gen whenever it is enabled, and fall
back to devicemap interleaved, without fd-pass, when it is disabled.

---

## Failure reporting

Pool factory methods return **`expected<BufferPool, Error>`** (`kInvalidArgument` for misconfiguration or invalid parameters; `kFailure` for SHM/VMM/CUDA resource failures). Errors are logged by the implementation.
