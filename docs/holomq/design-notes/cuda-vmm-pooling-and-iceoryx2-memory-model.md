# CUDA VMM pooling and the iceoryx2 memory model

Background on how holomq's CUDA VMM buffer pools relate to iceoryx2's shared-memory pool
model. holomq uses iceoryx2 for **host payload** pub/sub and implements **GPU pools**
separately with fd-pass import. This note explains that split and the architectural
parallels — it is not an upstream iceoryx2 specification.

## holomq's split

- **Host samples:** iceoryx2 SHM segment, loan/send, zero-copy offsets.
- **GPU bulk:** CUDA VMM pool per publisher (typical), `BufferHandle` in payload, fd-pass
  bootstrap, subscriber `VmmPoolImportCache`.
- **User API:** external `BufferPool` + `acquire()` / `export_handle` — not iceoryx2 bucket loans for device memory.

## How iceoryx2 models memory (relevant pieces)

### One backing region, pool of fixed buckets

The publisher side creates a **data segment** backed by shared memory. A **pool allocator** uses a single `chunk_layout` and `number_of_chunks`. Total size is essentially `chunks × bucket_size` plus alignment slack. That matches the mental model of “one slab, carved into equal cells.”

See `DataSegment::create_static_segment` in `iceoryx2/src/port/details/data_segment.rs`.

### Relative addresses across processes

Allocations yield **offsets** into the segment (for example `PointerOffset`), not absolute pointers that are valid in every process. Subscribers **translate** offsets to local addresses after mapping the same segment. Dynamic segments add registration when new mappings appear.

See `DataSegmentView::register_and_translate_offset` in the same file.

### Loan pattern and bounded concurrency

Publishers **loan** samples from the pool, fill them, and send. Limits such as `max_loaned_samples` bound how many outstanding allocations a port can hold. The same lifecycle fits “loan a CUDA sub-range, use it, return it,” with different failure modes and teardown rules.

## Mapping that model to CUDA VMM

### Where the ideas align

| iceoryx2 idea | CUDA / VMM analogue |
|---------------|---------------------|
| Single reservable slab | Reserve a VA range; create a pool of physical chunks; map/unmap subranges (`cuMemAddressReserve`, `cuMemCreate`, `cuMemMap`, etc., per your chosen VMM workflow). |
| Chunk identity | Prefer stable **logical ids** (segment id + offset, chunk index, generation) in messages, not raw GPU virtual addresses, if processes map the pool differently. |
| Loan / return | Explicit loan API + hard cap on concurrent loans for back-pressure (same spirit as `max_loaned_*`). |

### Where the platforms diverge

- **Sharing mechanism**: POSIX SHM is name + `mmap`. CUDA shareable memory uses **export/import handles** (for example POSIX **file descriptors** where supported), optional **fabric** for multi-node, and rules about **who may map what** and **ordering** (streams, semaphores, IPC signals). The pool layer owns that, not `shm_open`. See [Sharing mechanism](#sharing-mechanism-cross-process) below.
- **Who reads the payload**: iceoryx2 zero-copy assumes subscribers can read bytes from **mapped SHM** (with a coherent CPU ordering story). GPU memory may only be meaningful on the GPU unless you use unified/mapped memory or explicit copies. The middleware should state clearly **where** payload bytes are valid (CPU vs which GPU).
- **Process death**: If a process dies with loans outstanding, you need **reclaim** semantics (reset chunk state, close imports). Mirror the service/port failure modes you already expect from IPC middleware.

## Sharing mechanism (cross-process)

CUDA pool sharing needs a way to give the subscriber process a **kernel file description** for the exported pool (when using a POSIX-FD-style export path), not just an integer copied through pub/sub payloads.

### File descriptors are not portable as plain bytes

On Linux, an **fd number** is meaningful only inside one process. Putting `int fd = 42` inside an iceoryx2 (or any) **message body** does **not** give the peer a valid handle to the same object. The kernel must **duplicate** the file description into the receiver (for example via **`SCM_RIGHTS`**).

### `AF_UNIX` + `SCM_RIGHTS` (holomq bootstrap)

**`AF_UNIX`** stream sockets are **full-duplex**: either side may send and receive bytes. **`sendmsg` / `recvmsg`** with **`SCM_RIGHTS`** ancillary data is the standard way to **transfer an fd** between processes on the same machine. It avoids the TCP/IP stack and is appropriate for **rare control-plane** steps.

**holomq bootstrap flow:**

1. The **publisher** creates an **`AF_UNIX` listening socket** (filesystem path or abstract name, depending on platform and policy).
2. **CUDA / pool metadata** (carried with samples or a side descriptor) includes that **socket address** and a **pool id** (and ideally a **generation** / version if the pool can be recreated).
3. Each **subscriber**, on first use of that pool in its process (**cache miss**): **connects**, runs a small **handshake** (which pool, optional auth token), receives the fd via **`SCM_RIGHTS`**, then **`cuMemImportFromShareableHandle`** (and mapping) as required by the CUDA version and handle type; subsequent messages use **offsets** only and hit the **cache**.

**Fan-out:** the publisher **`accept`s** one connection per subscriber (or per bootstrap); many subscribers implies many connections over time, which is fine for a one-time cost.

### iceoryx2 request–response vs `AF_UNIX`

iceoryx2 request–response cannot replace **`SCM_RIGHTS`** — response payloads cannot safely
carry an fd as raw bytes. RR can still carry rendezvous metadata (socket path, nonce, errors).
holomq uses **`AF_UNIX` only** for fd-pass: bidirectional request/ack and the fd via ancillary
data, with the socket path embedded in `BufferHandle` metadata.

### Operational and safety notes

- **Permissions** on a filesystem-bound socket path (or unguessable abstract names) should match **who may subscribe** to the GPU data path; otherwise any local process that can connect might obtain an fd.
- **Stale socket files** after crashes: unique paths under `/run` or `$XDG_RUNTIME_DIR`, **`unlink`** on clean shutdown, and clear client behavior on **connect failure** (retry metadata, new generation).
- **Generation** in metadata avoids subscribers treating a **recreated** pool as the old mapping.

## Integration with iceoryx2

holomq uses the **on top of iceoryx2** model: host payloads stay on iceoryx2 SHM pub/sub;
GPU bulk uses external `BufferPool`, `BufferHandle` in payload, and fd-pass import.

iceoryx2 could theoretically host a CUDA-backed data segment behind offset-based APIs, but
that would require a second payload kind (CPU vs GPU) through the stack. holomq instead
implements VMM pools in a dedicated layer and reuses iceoryx2 for host sample signaling only.

**Bootstrap for the fd** uses a narrow `AF_UNIX` channel referenced from handle metadata; pub/sub
does not carry the fd itself.

## Performance

### Where cost actually lands

- **`AF_UNIX` + import + map** is a **one-time (or rare) cost per subscriber per pool generation** if you **cache** the imported mapping and **never** reconnect per message.
- **Steady-state hot path** is: **metadata + offsets + GPU execution + synchronization**—not the socket.

Replacing **`AF_UNIX`** for fd transfer rarely wins meaningful steady-state throughput; Linux already offers a **direct** fd-duplication primitive. If profiling shows the socket on the hot path, the usual fix is **too many reconnects or re-imports**, not a “faster fd primitive.”

### What usually improves end-to-end performance

1. **Fewer pools and fewer generations** — each new export/import/remap is large compared to tuning socket code.
2. **Device-resident payloads** — avoid extra **H2D/D2H** copies; use the shared pool + offsets on the GPU side where possible.
3. **Eager mapping** — perform connect + import when the subscriber **joins the topic** (or at discovery), not on the **first GPU sample**, to hide bootstrap latency off the critical frame path.
4. **Multi-GPU / cross-node** — when single-machine assumptions break, **NVLink**, **GPUDirect**, **NVSHMEM**, or **fabric** APIs are the next levers, not replacing **`SCM_RIGHTS`**.

## Operational considerations

- **Bootstrap vs steady state:** fd-pass is rare; steady-state cost is metadata, offsets, and GPU sync.
- **Lifecycle:** pool teardown, stale mappings after publisher restart, and fd-pass socket permissions matter for production use.
- **holomq choices:** fixed bucket size per pool, same-GPU stream memops for generation sync, lazy import on first `import_buffer` with warm `VmmPoolImportCache`, per-`Context` fd-pass server.

## Layout and topology decisions

| Decision | holomq approach |
|----------|------------------|
| Bucket size | Fixed `buffer_bytes` per pool |
| GPU topology | Same-GPU pub/sub for stream memops (see [`vmm-stream-memops-safety.md`](vmm-stream-memops-safety.md)) |
| Completion signaling | Per-slot generation words + `export_handle` / `import_buffer` |
| Bootstrap | Lazy fd-pass on cache miss; eager import is an application choice |

## Code references (iceoryx2)

- `iceoryx2/src/port/details/data_segment.rs` — static/dynamic segment creation, `allocate`, `register_and_translate_offset`, `unregister_offset`.
- `iceoryx2-cal/src/shm_allocator/pool_allocator.rs` — pool allocator behavior and offsets.
- `iceoryx2-cal/src/shm_allocator/pointer_offset.rs` — `PointerOffset` encoding (offset + segment id).

## Summary

A **VMM-backed pool with loaned subranges** matches iceoryx2’s structural pattern: reserved backing store, pool carve-up, **offset-style** chunk identity, and bounded loans. The **novel** work is CUDA **import/export**, **GPU-side coherence**, and explicit **CPU vs GPU** visibility—not the pool abstraction itself.

For **same-machine** sharing with a **POSIX fd** export path, **`AF_UNIX` listen + address in descriptor metadata + subscriber lazy (or eager) connect + `SCM_RIGHTS` + import + offset translation** is a **sound, standard** sharing layer; steady-state performance is dominated by **GPU data path and sync**, not by replacing that socket mechanism.

holomq follows the **pool + loans in a separate layer** plus **iceoryx2 for host payload
signaling** pattern described in [Integration with iceoryx2](#integration-with-iceoryx2).
