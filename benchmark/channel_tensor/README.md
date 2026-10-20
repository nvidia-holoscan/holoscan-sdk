# Channel Tensor Benchmark

This benchmark measures Holoscan Channel Tensor handle passing through the
same-process sidecar descriptor path. It complements the POD benchmark by
exercising the source-tree-only lower Tensor publisher seam and
`ChannelSubscriber<Tensor>` rather than inline typed payload buckets. The
publisher seam reuses the `ManualTensorPublisher` backend but is not an
installed public construction path.

The benchmark covers:

- one publisher to one subscriber with `ChannelQueuePolicy::kBlocking`;
- one publisher to one subscriber with depth-one `ChannelQueuePolicy::kDropOldest`;
- one publisher fanout to 2, 4, and 8 subscribers for both queue policies.
- lower Tensor publisher `emit()` with `ChannelQueuePolicy::kBlocking`
  after pre-filling 0, 1, 8, 64, 256, or 1024 unread sidecar descriptors.
- catalog Tensor `loan_tensor()` immediately after one publish/receive/retire
  cycle with fixed active work and configured slot capacities 1, 8, 64, or
  1024 (`BM_CatalogTensorLoanAfterPublicationByCapacity`);
- device-backed 1:1 and fanout-to-2 blocking variants
  (`BM_ChannelTensor_*_DeviceBlocking`) that carry real `kDevice` Tensors
  allocated from a `BlockMemoryPool`; these skip when no CUDA device is present;
- HoloMQ-pool-backed 1:1 and fanout-to-2 blocking variants
  (`BM_ChannelTensor_*_HoloMQPoolBlocking`) that carry host-pinned
  HoloMQ-buffer-backed Tensor handles on the same Tensor channel path.
- HoloMQ-device-pool-backed 1:1 blocking variants that compare sidecar and
  BufferHandle transport while holding the source Tensor allocation constant.
- a HoloMQ-pool-backed 1:1 blocking variant
  (`BM_ChannelTensor_OneToOne_HoloMQBufferHandleBlocking`) that uses direct
  Tensor `ChannelTensorTransferMode::kBufferHandle`, exports the HoloMQ buffer
  handle on emit, imports it on receive, and returns a read-only Tensor.
- isolated `TensorSidecarStore` insert/acquire and insert/force-release cycles
  at 1, 2, and 4 recipients, without HoloMQ transport or Channel endpoints.
- manually timed 1:1 emit-only and receive-only rows for native
  `BlockMemoryPool` device Tensors and HoloMQ-pool-backed Tensors.
- a matched HoloMQ-pool-backed Tensor emit row with event wake enabled, isolating
  the notification cost required by Runtime-managed process-local services.
- a bounded two-thread sidecar-store insert/acquire pipeline at depths 1, 8,
  and 64, paired with a descriptor-ring baseline.
- drop-oldest overwrite publisher latency at 1, 2, 4, and 8 subscribers, comparing
  the current transport replacement callback, a benchmark-only publisher-owned
  pending-descriptor mirror, and blocking delivery into an empty queue.
- matched direct-HoloMQ local sidecar round trips comparing the production
  unified Tensor transport payload with a benchmark-only slim local payload.

The benchmark preallocates a small ring of Tensor handles during setup. The
measured loop copies a Tensor handle into `emit()`, sends a descriptor through
HoloMQ, receives the Tensor handle from each subscriber, and drops the received
handles. Source Tensor allocation is intentionally excluded so this stays focused
on Channel sidecar transport and lifetime overhead.

The catalog-loan capacity sweep pauses timing for publication, receipt, and
retirement and times only the following `loan_tensor()` call. Exactly one loan
is active regardless of configured capacity. This makes the row sensitive to a
reintroduced full-pool free-slot scan without conflating capacity with useful
concurrency.

### Sidecar-store stage scope

The `BM_TensorSidecarStore_*` rows isolate the runtime-private sidecar record
state machine from endpoint and transport work. `InsertAcquire<N>` inserts one
Tensor, has all N recipients acquire it, and includes final record recycling.
`InsertForceRelease<N>` inserts one Tensor and exercises callback-style
`release_all()` cleanup without recipient acquisition. These rows use one
record repeatedly and report full-cycle time; they do not include HoloMQ loan,
send/take, sample-release callback registration, or public endpoint wrappers.

`BM_TensorSidecarStore_ConcurrentInsertAcquire` measures the same store with a
producer and consumer running concurrently through a bounded SPSC descriptor
ring. `BM_DescriptorSpscRing` measures the ring alone so synchronization and
record handoff costs can be distinguished. These rows are throughput-oriented;
they do not impose a barrier for every sample.

### Endpoint stage scope

The `*EmitBlocking` and `*ReceiveBlocking` rows use Google Benchmark manual
timing around only the named endpoint operation. The complementary operation
still runs every iteration to keep the channel in a bounded steady state, but
its time is excluded. Compare these rows within the same source-pool and
storage configuration; do not add independently measured stage medians and
treat the sum as a round-trip measurement.

`BM_ChannelTensor_OneToOne_HoloMQPoolEventWakeEmitBlocking` differs from the
matched HoloMQ-pool emit row only by enabling the HoloMQ event service. It does
not wait on a `WaitSet`; the row measures the synchronous publication-side
notification cost, not scheduler wake-to-run latency.

### HoloMQ-pool-backed variant scope

The `*_HoloMQPoolBlocking` variants allocate host-pinned HoloMQ buffers during
setup, wrap those buffers as `Tensor` handles, and then measure Channel
emit/receive of those handles through the same descriptor sidecar transport as
ordinary Tensor rows. They are a comparison point for schema rows that carry a
HoloMQ `BufferHandle` and materialize an imported Tensor on receive; buffer-pool
creation, acquisition, and Tensor wrapping are not included in the timed loop.

The `*_HoloMQBufferHandleBlocking` variant uses the same setup buffers, but
switches direct Tensor transport to `kBufferHandle`. The measured loop therefore
includes descriptor export on emit and HoloMQ buffer import plus read-only Tensor
wrapping on receive. It is the closest direct-Tensor comparison point for
interprocess-capable Tensor delivery, although this benchmark still runs both
endpoints in one process to avoid benchmark control-plane pipe/ack overhead.

The `*_HoloMQDevice*` variants provide the corresponding same-source comparison
for HoloMQ device-pool buffers. Sidecar rows reuse the source Tensor pointer
directly; BufferHandle rows exercise warmed same-process device-handle import.

### Device-backed variant scope

The `*_DeviceBlocking` variants measure descriptor/handle **transport** overhead
with real device-backed Tensors. They do **not** launch CUDA work, set
readiness/deallocation streams, or exercise DLPack lifetime extension, so their
numbers must not be read as full stream-safe production Tensor handoff overhead.
Because the channel transports only the fixed descriptor (not buffer data),
device-backed and host-backed transport costs are expected to be comparable; the
device variants exist to catch regressions on the real-device path, not to
measure GPU stream synchronization.

### Local payload size scope

The `BM_ChannelTensor_LocalPayload_*` rows isolate the transport cost of the
typed payload representation. `TransportOnly` measures blocking HoloMQ
loan/send/take and full payload assignment/copy. `CallbackOnly` adds the warmed
per-send release-callback path. The base `Unified` and `Slim` rows additionally
perform sidecar insert, Tensor acquire, and record release. Event wake is
disabled throughout.

`Unified` carries the production `TensorTransportPayload`, whose union also
supports an interprocess HoloMQ Tensor descriptor. `Slim` carries only
`SampleMetadata` and the 64-byte local descriptor. The `payload_bytes` counter
reports the transported type size. The slim type is benchmark-only and does
not imply a production transport split without a measured benefit.

The `BM_ChannelTensor_LocalTransport_*{Publish,Take}` rows further split raw
transport into batched publisher loan/assign/send and subscriber take/copy
stages at depths 1, 8, and 64. The complementary batch runs outside the manual
timing interval. The reported benchmark time is for a complete batch;
`items_per_second` is normalized by `batch_size` and is the preferred
per-sample comparison.

### Drop-oldest overwrite scope

The `BM_ChannelTensor_DropOldestOverwrite_*` rows time publisher loan, sidecar
insert, HoloMQ send, and cleanup bookkeeping. Depth-one drop-oldest queues begin each measured
iteration with one unread descriptor, so every send overwrites one descriptor
per subscriber. `TransportCallback` uses the production setup-time
`on_sample_replaced` callback. `PublisherMirror` disables transport replacement
attribution and adapts the older `holoscan::experimental` publisher-owned
per-recipient mirror to the current `TensorSidecarStore`. `BlockingBaseline`
sends into an empty depth-one blocking queue and drains subscribers outside the
manual timing interval.

All modes retain the normal per-send sample-release callback. The mirror row is
a benchmark model, not an alternate production implementation: it assumes one
publisher and full delivery to a static recipient set. The benchmark verifies
the number of callback or mirror replacements and drains the final sample to
validate sidecar lifetime cleanup. Event wake is disabled to match the default
Channel options used by the other polling benchmarks.

## Follow-Up Benchmark Work

- Add optional tail-latency sampling if mean timing does not capture jitter
  introduced by sidecar occupancy, fanout, or replacement attribution.
- Add a separate stream/event Tensor pipeline benchmark that launches CUDA work
  and exercises readiness/deallocation-stream synchronization and DLPack
  lifetime, to measure full stream-safe Tensor handoff overhead (distinct from
  the handle-transport variants here).
