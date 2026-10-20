# HoloMQ design notes

Contributor-focused reference on buffer pools, sync, transport, and FD-pass. **Application
developers** should start with [`PUBLIC_API.md`](../PUBLIC_API.md) and the
[`presentations/`](../presentations/) decks; source code under `include/holomq/` and
`src/` is the single point of truth.

These notes describe **how holomq works** — mechanisms, contracts, layout, and
limitations. They are reference material, not change proposals or roadmaps. When a note
discusses other layouts, it explains trade-offs for readers navigating the code.

## Documentation map

| Layer | Document |
|-------|----------|
| API contract | [`PUBLIC_API.md`](../PUBLIC_API.md) |
| Slide tour | [`presentations/holomq-high-level-design.md`](../presentations/holomq-high-level-design.md) |
| Sync deep dive | [`presentations/buffer-sync-deep-dive.md`](../presentations/buffer-sync-deep-dive.md) |
| Contributor depth | This directory |
| Code navigation | [`CONTRIBUTOR_READING_ORDER.md`](../CONTRIBUTOR_READING_ORDER.md) |

## Index by topic

### Buffer pools and lifetime

| Note | Summary |
|------|---------|
| [`ref-count-model.md`](ref-count-model.md) | POSIX SHM ref-count semantics and sample lifetime |
| [`buffer-pool-free-slot-accounting.md`](buffer-pool-free-slot-accounting.md) | SHM-only owner acquire (round-robin CAS) |
| [`host-buffer-pools.md`](host-buffer-pools.md) | Host pool layouts, flags, and fd-pass requirements |
| [`device-slot-payload-generation-views.md`](device-slot-payload-generation-views.md) | Internal payload/generation slot regions and layouts |
| [`multi-publisher-shm-and-gpu-pool-accounting.md`](multi-publisher-shm-and-gpu-pool-accounting.md) | Shared payload SHM vs per-publisher GPU pools |
| [`holomq-vs-holoscan-memory-types.md`](holomq-vs-holoscan-memory-types.md) | HoloMQ factories vs Holoscan `MemoryStorageType` |

### GPU sync and export

| Note | Summary |
|------|---------|
| [`buffer-stream-associated-export.md`](buffer-stream-associated-export.md) | Stream-parameter export/import contract |
| [`buffer-pool-slot-generation-sync.md`](buffer-pool-slot-generation-sync.md) | Per-slot generation words and stream memops |
| [`buffer-gpu-completion-before-pool-reuse.md`](buffer-gpu-completion-before-pool-reuse.md) | Deferred ref-count release on export stream |
| [`vmm-stream-memops-safety.md`](vmm-stream-memops-safety.md) | VMM stream memops constraints (driver 555+) |
| [`cross-process-gpu-write-sync.md`](cross-process-gpu-write-sync.md) | Read-only subscriber imports; writable shared-slot coordination |

### IPC and transport

| Note | Summary |
|------|---------|
| [`transport-abstraction.md`](transport-abstraction.md) | iceoryx2 transport layer and port templates |
| [`wait-set-architecture.md`](wait-set-architecture.md) | IOX2 event wake, WaitSet loop, and cancellation triggers |
| [`publisher-sample-release-registry.md`](publisher-sample-release-registry.md) | `on_released` callback registry |
| [`fd-pass-one-server-per-context-vs-per-pool.md`](fd-pass-one-server-per-context-vs-per-pool.md) | AF_UNIX VMM fd-pass; one server per `Context` |
| [`vmm-pool-import-cache-entry-eviction.md`](vmm-pool-import-cache-entry-eviction.md) | Import cache retention and eviction trade-offs |
| [`cuda-vmm-pooling-and-iceoryx2-memory-model.md`](cuda-vmm-pooling-and-iceoryx2-memory-model.md) | iceoryx2 SHM model vs CUDA VMM pools |
