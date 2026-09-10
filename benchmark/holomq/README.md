# HoloMQ benchmarks

HoloMQ currently provides same-process Google Benchmark microbenchmarks in
`holomq_pubsub_benchmarks`. These rows measure typed pub/sub endpoint and
release-callback costs without Holoscan Channel wrappers.

`holomq_cuda_sync_benchmarks` isolates the CUDA stream primitives used by
device BufferHandle readiness and deferred release. Its generation rows time
completed batches of `cuStreamWriteValue64` and already-satisfied
`cuStreamWaitValue64` operations independently. Its host-function row times
completed batches of `cuLaunchHostFunc` callbacks with a shared atomic counter
and no per-callback heap allocation. The latter is therefore a lower bound for
HoloMQ deferred release, which additionally allocates a payload, retains shared
mapping ownership, and updates the shared reference count.

All three rows include stream synchronization in the manually timed batch.
The wait row initializes its generation word before timing, so it measures the
driver cost of an immediately satisfied wait without introducing a
same-process cross-stream scheduling dependency. It is a primitive lower bound,
not a measurement of producer/consumer overlap.
Compare their normalized `items_per_second` values at batch depths 1, 8, and
64. The depth-one rows intentionally include the fixed synchronization cost;
deeper rows expose the completion-throughput ceiling relevant to a sustained
device BufferHandle stream.

`BM_DeviceBufferReadyReleaseCompletion` then measures the public HoloMQ device
Buffer lifecycle at the same batch depths: acquire pool slots, enqueue each
ready generation, destroy the owner buffers to schedule deferred SHM-reference
release, and synchronize the stream. It includes the real callback payload,
shared mapping ownership, and reference-count update but excludes pub/sub
transport and cross-process import. Comparing it with the primitive rows shows
how much batching survives the complete publisher BufferPool path.

`BM_CudaEventRecordQueryCompletion` measures the alternative used by the older
same-process experimental Tensor pool: preallocate one disabled-timing event
per slot, record release completion on the CUDA stream, and poll the events
before slot reuse. `BM_CudaGenerationEventRecordQueryCompletion` precedes every
event with the same generation write used by publisher readiness. Both include
active CPU polling and exclude SHM reference updates. Treat them as evidence
for or against a future Context-owned release reaper, not as a drop-in
replacement for cross-process ownership accounting.

The `bm_host_buffer_control_lifecycle_cold` and
`bm_host_buffer_control_lifecycle_reused` rows compare equivalent complete host
BufferHandle owner/export/import/release generations. The cold row constructs
new owner and import controls for every item. The reused row retains one owner
and one validated import mapping per slot, then times `prepare_for_reuse()`,
export, rebind, and release. The `reused_controls` counter identifies the
variant. These same-process rows isolate control and mapping reuse from pub/sub
transport and payload writes; compare normalized `items_per_second` at batch
depths 1, 8, and 64.

## Polling configuration

Ordinary pub/sub rows poll `Subscriber::try_take()` and explicitly disable
event wake on both endpoints. This avoids measuring notifier work that the
benchmark never consumes. The separately named
`BM_PubSubPOD_OneToOne_Blocking_LatchedWaitSet` enables event wake
symmetrically, publishes before waiting on the same thread, and consumes the
already-latched notification before taking the sample. It measures serialized
notification and latched-take cost, not a parked-consumer wake. The Runtime
Channel benchmark's separately named asynchronous-waiter rows provide broader
cross-thread parked-wake evidence.

Every concurrent row uses an exact one-recipient gate and a five-millisecond bounded-retry
publisher. Neither changes the healthy steady-state path; together they convert unexpected queue
pressure into bounded benchmark evidence instead of allowing a smoke test to park in `send()`.

`BM_PubSubPOD_ConcurrentWaitSet_PublisherSend` supplies the HoloMQ-side
publisher control for that case. A pinned subscriber thread actively waits and
drains while manual timing covers only the publisher's 1, 2, 4, 8, or
64-message loan/assign/send batch. Each row uses a matching queue depth. The
publisher additionally uses the event-wake configuration of a Runtime Graph service. Receive
completion and validation remain outside the timed interval. The matched
`BM_PubSubPOD_ConcurrentPolling_PublisherSend` row keeps the subscriber
actively polling and disables event wake, separating concurrent queue access
from notifier and wake-scheduling cost. Batch one exposes the unamortized
per-publication wake cost; deeper batches show how an already-running receiver
amortizes that cost.

All rows disable VMM FD export because these typed control-plane rows do not
create or transfer BufferHandles. Future BufferHandle benchmarks should enable
that facility in separately named rows.

The `event_wake` and `vmm_fd_export` counters record this configuration. Do not
compare rows with different settings unless that difference is intentional.

## Existing POD rows

The `BM_PubSubPOD_*` rows use the exact 64-byte `BenchmarkPOD64` comparison
payload and cover:

- serialized blocking and drop-oldest 1:1 round trips at representative depths;
- a serialized blocking 1:1 round trip with event wake and a latched `WaitSet`;
- a blocking 1:1 diagnostic with the otherwise-unneeded sample-release handler enabled;
- serialized blocking and depth-one drop-oldest fanout to 2, 4, and 8 subscribers;
- batched transport with and without a per-send release callback; and
- the release-callback registry without transport.

The ordinary self-contained POD rows disable sample-release callbacks. Compare
`BM_PubSubPOD_OneToOne_Blocking` with
`BM_PubSubPOD_OneToOne_Blocking_ReleaseHandlerEnabled` to isolate the transport
handler's empty-registry-dispatch cost without registering per-send callbacks.
The depth-one drop-oldest row is the freshest-only comparison. The depth-64
drop-oldest row keeps queue occupancy at one in the serialized loop and
provides a steady-state comparison at the default service depth. The HoloMQ
unit tests separately validate depth-N overflow and retention.

The registry controls separate the callback lifecycle from transport work.
`BM_PublisherSampleReleaseRegistry_RegisterDispatch` measures the complete
steady-state register-and-dispatch lifecycle. The `RegisterOnly` and
`DispatchOnly` variants use manual batch timers and move the complementary
stage outside the interval. Use their batch-64 normalized rates when comparing
these small stages; dispatch runs when a transport slot is released and is not
part of the original publisher `send()` interval.

## Fixed-payload stage rows

The `BM_PubSubFixed_*` rows use exact 64-, 128-, and 440-byte trivially
copyable payloads. The sizes were chosen because:

- 64 bytes provides a cache-line-sized control-descriptor baseline and matches
  the fixed descriptor size used by the SDK Channel layer;
- 128 bytes represents the benchmark-only slim local Tensor payload shape used
  in Channel transport investigations; and
- 440 bytes matches the current unified Channel direct-Tensor transport
  payload, including space for its interprocess-capable descriptor union.

These sizes let HoloMQ measurements be compared with the adjacent Channel
experiments without adding a dependency on Holoscan headers. Each benchmark
also reports `payload_bytes` so a future layout change cannot silently alter
the comparison.

Each payload size has:

- `PollingRoundTrip`: serialized loan, full payload assignment, send, take,
  full payload copy, and checksum;
- `PollingPublish`: manual timing around batched loan, assignment, and send;
  and
- `PollingTake`: manual timing around batched take, full payload copy, and
  checksum.

Publish and take rows run at batch depths 1, 8, and 64. The complementary batch
runs outside the manual timing interval. The time column is the complete batch
duration; use `items_per_second` for the normalized per-sample comparison.

Stage medians from deeper batches should not be added and treated as serialized
round-trip latency. Batching changes queue occupancy and cache reuse while
amortizing clock-read overhead.

## Publisher substage rows

The `BM_PubSubFixed_Publisher*` rows further decompose publisher-side work for
the same fixed payload sizes and batch depths:

- `LoanReturn` borrows each transport sample and returns it without publishing;
- `LoanAssignReturn` also assigns the complete payload before returning the
  unpublished loan; and
- `PreloanedSend` times only sending messages that were loaned and populated
  before the measurement interval.

The 64-byte `PreloanedSendExpectedRecipient` row repeats the preloaned send
with `expected_recipient_count=1`. Compare it with the ordinary 64-byte
`PreloanedSend` row to isolate the per-send topology preflight performed by the
public exact-recipient contract. The `expected_recipient_gate` counter records
which variant is active.

Four additional 64-byte rows model the remaining successful-send setup used
by Runtime Graph transport:

- `PreloanedSendBoundedRetry` installs the five-millisecond bounded-pressure
  handler without enabling event wake;
- `PreloanedSendEventWake` enables event wake and retains one attached wait-set
  listener without changing the publisher pressure policy; and
- `PreloanedSendRuntimeEventless` combines the exact-recipient gate and bounded
  pressure used by a current one-recipient eventless Runtime Graph service;
  and
- `PreloanedSendGraphEquivalent` additionally enables event wake and retains
  an attached listener as an eventful control.

The `bounded_retry`, `event_wake`, and `expected_recipient_gate` counters make
the active combination explicit. These rows time successful sends only; they
do not exercise a pressured queue or the bounded retry loop.

The difference between the first two rows estimates payload-write cost. The
preloaned row permits a full batch of outstanding publisher loans specifically
to expose the remaining send path. It is a diagnostic decomposition rather
than a recommended publishing pattern.

## Subscriber substage rows

The 64-byte `BM_PubSubFixed_Subscriber*` rows isolate the successful receive
sequence used by an eventless Runtime Graph service:

- `SubscriberEmptyProbe` and `SubscriberReadyProbe` time the same stable
  `has_samples()` topology refresh and queue observation with an empty or
  non-empty queue;
- `SubscriberEmptyTryTake` measures the alternative empty-queue receive
  attempt and typed `kNotReady` classification;
- `SubscriberTryTakeBorrowed` takes one transport sample and consumes its
  payload through the message loan without a payload copy;
- `SubscriberTryTakeCopyOut` adds the complete payload copy performed by the
  ordinary inline Runtime receive path; and
- `SubscriberRuntimeGraphEquivalent` performs the two ready probes currently
  made by the activation cut and inline endpoint, then takes and copies the
  sample.

The rows publish or drain outside the measured interval. They are diagnostic
and non-additive: compare `SubscriberRuntimeGraphEquivalent` directly with
`SubscriberTryTakeBorrowed` for the maximum combined ceiling of removing the
duplicate successful probes and copy-out. The `empty_probe`, `empty_take`,
`ready_probes_per_take`, `take`, `copy_out`, and `borrowed_payload` counters
make each interval explicit.

## Fanout backend stage rows

The `BM_PubSubFixed_Fanout*` rows keep the 64-byte fixed payload, eventless
blocking service, exact-recipient gate, five-millisecond bounded retry, and
`PubSubServiceOptions::max_subscribers` capacity used by Runtime Graph transport. Their
names identify the backend work they expose:

- `FanoutStableRefresh` times explicit `Publisher::update_connections()` calls
  after the full fanout is established and no endpoint membership changes;
- `FanoutEmptyReturnScan` times loan/return cycles when no subscriber has a
  returned sample pending, including the backend's empty per-connection scan;
- `FanoutReturnedSampleReclaim` publishes and drains outside the timer, then
  times the next loan that reclaims the returned offsets, decrements recipient
  references, releases the final transport sample, and allocates/returns one
  replacement loan;
- `FanoutReturnedSampleReclaimReleaseCallback` adds HoloMQ's real registered
  release-callback dispatch to that reclaim interval; and
- `FanoutPreloanedSend` times the successful-send path after loans and payload
  assignment, including the stable connection refresh, empty return scan,
  offset delivery, and per-recipient reference acquisition.

Fanout two has the detailed batch-depth sweep because it is the expected common
production shape. Fanout one, four, and eight are narrower scaling controls.
The stages are deliberately not additive: allocation, batching, queue state,
and cache state differ. Use them to locate a candidate and then require a
complete preloaned-send and Runtime Graph A/B result.

The time column is again a complete batch duration. Prefer normalized results
from the deeper batches when comparing small substages because depth-one rows
contain a larger proportion of clock-read overhead. These measurements are not
strictly additive: preloaning changes sample-slot occupancy and cache state.

## Running

Build and list the rows from an SDK development container:

```bash
cmake --build . --target holomq_pubsub_benchmarks holomq_cuda_sync_benchmarks \
  holomq_transport_benchmark
./benchmark/holomq/holomq_pubsub_benchmarks --benchmark_list_tests
./benchmark/holomq/holomq_cuda_sync_benchmarks --benchmark_list_tests
```

For production measurements, use a Release build, random interleaving, several
repetitions, and a nontrivial minimum duration. Keep raw JSON under the build
directory; do not commit benchmark output.

## Interprocess inline transport

`holomq_transport_benchmark` is a role-aware worker for the first topology
benchmark slice. `run_holomq_benchmark.py` launches a subscriber, waits for its
explicit readiness marker, launches the publisher, validates both role
results, and writes a combined result and host manifest.

The worker covers blocking, polling-without-event-wake interprocess transport
for exact 64-, 128-, and 440-byte inline payloads. Saturated throughput supports
1:N fanout. Paced latency is intentionally restricted to 1:1 with a queue depth
of one. The current smoke matrix covers 1, 2, 4, and 8 throughput subscribers.
It validates every sequence and payload checksum for every subscriber.

Subscriber processes start first. The publisher uses reserved probe sequences
until `queue_depth + 1` sends have reached every expected recipient, and
subscribers discard those probes before sequence validation. More successful
deliveries than one receiver queue can hold proves that every subscriber has
connected and drained at least one probe. This prevents partial startup fanout
from delivering duplicate measured sequences to subscribers that connected
early, including when a transport accepts samples into a not-yet-connected
receiver queue.

`completed_messages_per_second` uses the slowest valid subscriber completion
timestamp. `completed_deliveries_per_second` multiplies that logical message
rate by the subscriber count and is emitted only after every subscriber result
and timestamp passes validation. Throughput-mode latency percentiles come from
a saturated run and are diagnostic. Use `--workload latency` for latency
comparisons. Latency mode paces the publisher before taking the per-sample send
timestamp, so the intentional period is outside the one-way transport
measurement. It records `rate_limited=true`; its completed rate is an achieved
pacing check, not a transport-capacity result. For matched 1:1 transport
comparisons, this send timestamp through successful subscriber receive is the
common `publication_to_receive_completion` anchor.

`--payload-validation none` skips payload-content checks for production
transport measurements while retaining sequence, descriptor, protocol, and
completion-count validation. `final` retains and validates only the final
payload after the timed batch; for device storage it also preserves the
stream/lifetime synchronization needed to do so safely. `all` validates every
payload and is intended for smoke and correctness evidence because that work
can affect completion-rate timing. Native results record the selected mode and
whether validation ran inside, between, or after timed operations.

Run a smoke case from the SDK build directory:

```bash
python3 ../benchmark/holomq/run_holomq_benchmark.py \
  --binary ./benchmark/holomq/holomq_transport_benchmark \
  --output-dir ../build/holomq-transport-smoke \
  --payload-bytes 64 --samples 1000 --warmup 100 --queue-depth 8 \
  --subscriber-count 2 --subscriber-cpus 3,4 --publisher-cpu 2
```

The output directory contains `publisher.json`, indexed `subscriber_*.json`
results, their stderr logs, `combined.json`, and `manifest.json`. Keep these
generated files under an ignored build directory.

Use `--preset smoke` for 1,000 measured samples, 100 warmup samples, queue
depth 8, and one repetition. `--preset production` selects 100,000 measured
samples, 10,000 warmup samples, queue depth 64, and seven repetitions. Explicit
`--samples`, `--warmup`, `--queue-depth`, or `--repetitions` values override the
selected preset.

Multiple repetitions use `repetition_NNN` subdirectories with an independent
service name and complete role outputs for each run. The root `summary.json`
reports the median, minimum, and maximum of the primary rate and diagnostic
latency metrics. The root manifest records the requested preset and repetition
count. For pinned Linux runs, the runner resolves each CPU's package and
physical core through sysfs, records that mapping in `cpu_topology`, and
rejects role assignments that share a physical core. This prevents SMT
siblings from silently contending in production comparisons. Other platforms
record topology validation as unavailable and skip the Linux-specific check.
The worker applies affinity only after creating its HoloMQ context and
endpoint, so transport service threads retain the process's original affinity
instead of inheriting the measurement CPU.

Each combined run and repetition summary records attempted, accepted,
backpressured, indeterminate, and completed publication counts. Loan
backpressure is the retryable publication outcome; BufferPool acquire
backpressure remains a separate pre-publication resource diagnostic. For
BufferHandle rows, `pool_slots` and queue depth retain the explicit fixed
resource geometry used by the run.

Run a spin-paced 1:1 latency case with queue depth 1:

```bash
python3 ../benchmark/holomq/run_holomq_benchmark.py \
  --binary ./benchmark/holomq/holomq_transport_benchmark \
  --output-dir ../build/holomq-latency \
  --preset production --workload latency --publisher-period-us 10 \
  --publisher-pacing spin --publisher-cpu 2 --subscriber-cpu 3
```

Pacing uses an absolute deadline to avoid accumulating per-iteration drift.
`spin` gives the most faithful short period while consuming a dedicated CPU;
`sleep` is available when CPU consumption matters more than microsecond-scale
pacing precision. Latency mode defaults to a 10-us period and queue depth 1.

## Interprocess host BufferHandle transport

The bulk-buffer slice adds blocking 1:1 throughput for pageable and
CUDA-pinned shared host memory. Select it with `--payload buffer_handle`;
`--storage pageable_host` uses POSIX shared memory and
`--storage pinned_host` registers that shared memory with CUDA.
`--buffer-bytes` controls the external buffer size, defaults to 1 MiB for
maintained transport comparisons, and `--pool-slots` defaults to queue depth
plus two.
At queue depth 8, boundary runs found that 9 slots deadlocked while 10 slots
completed without acquire backpressure. The two-slot margin covers the current
outbound sample and release-callback handoff before an older owner lease becomes
reusable; configurations below that margin are rejected. The publisher creates
an FD-export-enabled `BufferPool`, loans the transport sample before acquiring
a pool slot, exports an unsynchronized host handle, and keeps the owner `Buffer`
alive in the sample-release callback. The subscriber imports the handle without
CUDA stream work and releases each import before taking the next sample.

The worker records `cold_first_import_ns` separately from measured
`buffer_import_*` percentiles. Warmup populates the subscriber's backing,
reference-count, and generation-pool caches before the measured imports. It
writes and validates 64-bit markers at the beginning and end of every buffer;
this checks the mapped slot and offset without making a full buffer copy part
of the transport benchmark. `buffer_acquire_*` and `buffer_import_*` retain the
isolated pool-draw and warmed-import intervals. With
`--payload-validation final`, `final_validation_ns` reports the post-batch
validation and required synchronization separately from transport timing.
Deferred release has no independent completion-observation seam yet and is not
approximated from later pool acquisition.

Run an explicitly small 4 KiB pageable-host smoke case:

```bash
python3 ../benchmark/holomq/run_holomq_benchmark.py \
  --binary ./benchmark/holomq/holomq_transport_benchmark \
  --output-dir ../build/holomq-host-buffer-smoke \
  --preset smoke --payload buffer_handle --buffer-bytes 4096 \
  --publisher-cpu 2 --subscriber-cpu 3
```

Both host rows intentionally exclude fanout and GPU readiness work. Pageable
and pinned backing remain separate rows so CUDA registration and import costs
stay visible.

Run the corresponding pinned-host case with:

```bash
python3 ../benchmark/holomq/run_holomq_benchmark.py \
  --binary ./benchmark/holomq/holomq_transport_benchmark \
  --output-dir ../build/holomq-pinned-buffer-smoke \
  --preset smoke --payload buffer_handle --storage pinned_host \
  --buffer-bytes 4096 --publisher-cpu 2 --subscriber-cpu 3
```

## Interprocess device BufferHandle transport

Select device VMM buffers with `--payload buffer_handle --storage device`.
Every pool slot is initialized to a fixed device pattern and synchronized once
before endpoint readiness and role pinning. The measured loop does not enqueue
bulk data work. After timing, the subscriber synchronizes as required and
copies boundary bytes from the final imported buffer to validate the device
mapping and contents.

Device mode defines a baseline and a full synchronization contract:

- `--device-sync none` uses unsynchronized export/import after the one-time
  pool initialization. This is the baseline for immutable, already-ready data
  and isolates handle transport plus warm VMM import.
- `--device-sync ready` sets a ready generation on the publisher stream,
  enqueues its wait on the subscriber stream, and defers both owner and import
  releases. Use this diagnostic for per-sample GPU producer/consumer ordering.

Two diagnostic modes split the full ready contract without changing production
APIs:

- `publisher_ready` performs the publisher generation write and deferred owner
  release, while the subscriber imports without waiting; and
- `subscriber_release` publishes without a fence, then arms deferred import
  release on the subscriber stream without a generation wait.

Both are valid here only because device slots are immutable after the one-time
initialization. They exist to attribute benchmark cost and must not be treated
as synchronization contracts for per-sample GPU writes.

These modes are not interchangeable. `none` is valid only because no producer
work modifies a slot after initialization. Any benchmark that enqueues
per-sample GPU writes must use `ready` or another explicit synchronization
contract.

Run the warmed no-sync device smoke case with an explicitly small 4 KiB
buffer:

```bash
python3 ../benchmark/holomq/run_holomq_benchmark.py \
  --binary ./benchmark/holomq/holomq_transport_benchmark \
  --output-dir ../build/holomq-device-buffer-smoke \
  --preset smoke --payload buffer_handle --storage device \
  --device-sync none --buffer-bytes 4096 \
  --publisher-cpu 2 --subscriber-cpu 3
```

The ready-generation mode intentionally exposes current CUDA callback and
deferred-release costs. Keep its acquire retry count adjacent to stage
percentiles; increasing pool capacity can hide backpressure temporarily while
building a longer stream backlog.

For an uncontended device-ready latency measurement, use the explicit latency
workload. Device BufferHandle latency defaults to a 500-us spin-paced publisher
period (2,000 samples/s), queue depth 1, and ten pool slots for device-ready
runs. The period leaves headroom below the currently observed deferred-release
throughput so the reported percentiles measure the ready/import path without
sustained pool backpressure. The additional slots absorb release-completion
jitter; they do not raise saturated steady-state throughput. The completed rate
is only a pacing check in this mode. Host BufferHandle latency defaults to 250
us because it does not use CUDA deferred release.

```bash
python3 ../benchmark/holomq/run_holomq_benchmark.py \
  --binary ./benchmark/holomq/holomq_transport_benchmark \
  --output-dir ../build/holomq-device-ready-latency \
  --preset smoke --workload latency --payload buffer_handle \
  --storage device --device-sync ready \
  --publisher-cpu 2 --subscriber-cpu 3
```
