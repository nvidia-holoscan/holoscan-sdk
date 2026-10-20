# Channel Schema Benchmark

This benchmark measures Holoscan Channel transport of prebuilt FlatBuffer
payloads carried inside fixed-capacity inline typed payload wrappers. It
exercises the native typed HoloMQ path through `ChannelPublisher<T>` /
`ChannelSubscriber<T>`. It also includes descriptor-backed tensor-schema rows
that use `ChannelTensorSchemaPublisher<T>` /
`ChannelTensorSchemaSubscriber<T>` and therefore mint a real same-process Tensor
sidecar descriptor on emit and resolve that descriptor back into a `Tensor` on
receive.

The benchmark covers:

- process-local `TimeT` transport framing with an actively draining receiver,
  measuring only 1, 2, 4, 8, and 64-publication Channel send batches for both
  polling and event-driven WaitSet receive modes;
- one publisher to one subscriber carrying an Imu FlatBuffer with
  `ChannelQueuePolicy::kBlocking`;
- one publisher to one subscriber carrying an Imu FlatBuffer with
  `ChannelQueuePolicy::kDropOldest` at depth one;
- one publisher fanout to 2, 4, and 8 subscribers for both queue policies;
- one publisher to one subscriber carrying Image metadata that references a
  Tensor schema object without materializing Tensor bytes;
- one publisher to one subscriber carrying Image metadata plus a fabricated
  `ChannelSampleDescriptor` that references external Tensor data, with
  `ChannelQueuePolicy::kBlocking` and depth-one `ChannelQueuePolicy::kDropOldest`. These
  rows exercise generic inline transport of descriptor metadata only; the
  fabricated descriptor is not resolved to a Tensor;
- one publisher to one subscriber carrying Image metadata plus a real
  sidecar-resolved Tensor handle through the materializing tensor-schema
  endpoint for both queue policies. These rows expose the materialized payload
  identity as `PayloadDescriptor`;
- one publisher fanout to 2, 4, and 8 subscribers carrying Image metadata plus
  a real sidecar-resolved Tensor handle for both queue policies;
- one publisher to one subscriber carrying Image metadata plus a HoloMQ
  `BufferHandle` payload reference through the materializing tensor-schema
  endpoint, importing the handle and returning a read-only `Tensor` on receive.

FlatBuffer construction happens during setup and is intentionally excluded from
the measured loop. The measured loop emits an already-built fixed-capacity
payload wrapper and receives it from each subscriber.

The `BM_ChannelSchema_TimeTransport_Concurrent*PublisherSend` rows are narrower:
they prebuild the exact canonical `TimeT` bytes and process-local schema
envelope used by Runtime, then time only
`InlinePayloadPublisher::emit_phase()`. Batch depth and subscriber queue depth
match at 1, 2, 4, 8, and 64 so the rows distinguish unamortized publication
from an already-running receiver. The receiver runs concurrently on another
thread, and receive completion, envelope validation, and checksum work are
outside the measured interval. Set `HOLOSCAN_BENCHMARK_PUBLISHER_CPU` and
`HOLOSCAN_BENCHMARK_SUBSCRIBER_CPU` to pin the two roles; `-1` leaves a role
unpinned.

The `BM_ChannelSchema_DescriptorBackedImageFlatBuffer_*` rows measure generic
transport of schema metadata plus a fabricated descriptor only. They do not touch
the Tensor sidecar store and are useful as a lower-level typed-payload reference
point.

The `BM_ChannelSchema_DescriptorBackedImageTensorSchema_*` rows use the
production-direction tensor-schema endpoints. Their measured loop emits verified
metadata plus a preallocated host Tensor handle, transports the descriptor-backed
metadata wrapper through HoloMQ, resolves the descriptor through the
same-process Tensor sidecar store in each subscriber, exposes the resolved
payload identity as `PayloadDescriptor`, and drops the resolved Tensor handle.

The `BM_ChannelSchema_DescriptorBackedImageHoloMQBuffer_OneToOne_Blocking` row
uses the same tensor-schema subscriber path, but the payload descriptor carries a
real HoloMQ buffer handle instead of a same-process sidecar reference. Its
measured loop emits prebuilt Image metadata plus that handle, imports the handle
on receive, materializes a read-only `Tensor`, and drops the imported Tensor
handle. FlatBuffer construction, HoloMQ pool creation, buffer acquisition, and
handle export are setup costs and are excluded from the timed loop.

This row exercises the interprocess-capable HoloMQ BufferHandle data path while
running both endpoints in one process. A true two-process benchmark should be
added separately once the benchmark harness can avoid adding pipe/ack overhead to
the measured loop.

## Follow-Up Benchmark Work

- Add percentile or per-message interval sampling if schema payload transport
  becomes a latency/jitter focus.
- Add separate GPU stream/lifetime schema benchmarks when operator/runtime APIs
  are available to exercise readiness and deallocation stream propagation. The
  materializing rows here measure handle transport and sidecar resolution, not
  CUDA work scheduling.
