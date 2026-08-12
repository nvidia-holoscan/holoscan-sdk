# Runtime Benchmarks

The source-tree-only Google Benchmark suites measure bounded EA1 Runtime paths.
For controlled latency measurements, apply and record the platform setup in
[Runtime Benchmark Hardware Tuning](HARDWARE_TUNING.md). Functional CTest
smokes do not require those settings.

`runtime_dispatcher_benchmark` covers:

- one preallocated Dispatcher freeze/select/complete cycle over 1, 8, or 64
  deterministic READY candidates;
- one preallocated candidate-coordinator ingest/freeze/retire batch over 1, 2,
  or 4 independent sources, reporting work per candidate; and
- one activation-selection begin/retire cycle over 0, 1, or 8 input-selection
  and temporal-sample records, isolating bounded metadata snapshot cost from
  scheduling and Channel ownership; and
- one physical readiness notification publish/drain/retire cycle through a
  capacity-one preallocated binding; and
- one transport-free composed controller activation transaction: bounded event
  ingest, candidate and Dispatcher freeze/select, activation snapshot, one
  busy-lane callback with Dispatcher completion, and source/activation
  retirement. It excludes Graph lookup, clocks, parameters, lifecycle,
  predicates, Channel storage, and transport; and
- one parked `RuntimeWakeState` signal-to-wait-return interval after a 5 ms idle period, excluding
  readiness scanning and execution-lane work, plus Linux-only benchmark-local futex and eventfd candidates
  with the same retained-token contract; and
- one empty `ExecutionLaneSet::dispatch()` round trip in sleeping-waitset and
  busy-poll modes, including controller-to-worker handoff and synchronous
  worker-to-controller completion, plus matched rows that dispatch only after
  5 ms of lane inactivity while excluding that idle interval from timing; and
- one complete blocking `run(plan)` attempt containing one explicitly injected,
  source-tree-only direct in-memory readiness witness, including session
  construction, lifecycle, one Operator callback, coordinated completion,
  teardown, and thread join; and
- the same complete one-activation run for an Operator with one
  `runtime_mutable` field and no submitted update, to quantify empty Parameter
  maintenance relative to the `NoParams` row; and
- paired `NoParams` and runtime-mutable/no-update sessions containing 65,536
  activations, reporting aggregate activation throughput so session startup and
  thread join are amortized across the steady callback path; and
- 8,192 real direct-mode host Tensor allocate/write/commit/cancel cycles while
  the cold pool capacity varies across 2, 8, 64, and 1,024 slots. This row
  detects accidental return of per-activation full-capacity scans; it includes
  one session materialization and teardown amortized across the callbacks.

`runtime_channel_benchmark` covers:

- one `RuntimeChannelSession` activation prepare/retire cycle while total
  graph-wide Channel owner count varies and the selected owner's local degree
  remains one; and
- one readiness poll over retained bursts that vary input count and queue
  depth, including a 1,024-sample case that distinguishes quadratic growth
  from the bounded owner-local candidate implementation; and
- the revision gate plus one synchronous busy-lane readiness-poll dispatch for
  a single process-local input. Injection and witness retirement are paused,
  so this row isolates the target-lane round trip without controller clock
  lookup, candidate ingestion/freezing, activation preparation, or compute. A
  matched direct-retained rows publish synchronously before timing and measure
  the exact compiler-authorized 1:1 poll both directly and through its target
  execution lane. An explicitly optimistic ceiling row then combines poll,
  typed receive, claim retirement, and the sink activation cut in one lane
  task; production global candidate ordering and activation selection are
  excluded rather than silently bypassed. Two diagnostic rows then compare
  the current two-task poll/consume handoff with a benchmark-local split-phase
  worker that reports poll completion, waits for controller authorization, and
  resumes consumption without publishing a second task. The split-phase row
  estimates a mechanism only; it does not implement or authorize a production
  Runtime continuation. A second benchmark-only ceiling compares the same
  direct-retained source and unique-successor callback bodies as ordinary
  three-task, receipt-elided two-task, and composite one-task transactions.
  Sleeping and busy-poll rows cover queue depths one and two plus controlled
  callback work of 0, 1, 5, 20, and 100 microseconds. Each row verifies serial
  callback execution and reports bounded source-to-successor and
  total-transaction percentile samples, lane-task counts, per-role CPU, and
  context switches. The composite arm is diagnostic only and supplies no
  production correctness evidence; and
- one witness retirement for an owner with one local input while total
  graph-wide Channel input count varies across the common 1-to-3 range, the
  previous 8/16 characterization points, and one 64-input scaling stress row.
  The stress row uses HoloMQ's explicit service-static subscriber capacity to
  provision the parallel event-wake service beyond the backend default. Each
  row retires the final subscriber in construction order so a session-wide
  linear lookup would traverse all preceding subscribers; and
- one complete HoloMQ-backed typed POD edge cycle: producer activation, emit,
  both post-callback process-local cuts, readiness publication, direct arrival
  selection, public `receive_data()`, and all witness, selector, and activation
  retirement. This component row excludes Dispatcher and execution-lane work.
  Two manual-time rows split producer publication plus its cut from downstream
  readiness/receive/retirement plus the sink cut. Their unmeasured companion
  half keeps the same bounded edge reusable, and an empty two-clock bracket
  quantifies the phase instrumentation floor. Two additional rows split the
  producer interval before and during the post-source cut while retaining the
  asynchronous pump race present in production. An emit-only row prepares the
  source activation before timing and retires it afterward, isolating the
  Runtime `Output<T>::emit()` path from activation bookkeeping. A no-op prebound
  output row bounds public sample construction and binding dispatch without
  Runtime admission, transport, or notification work. An adjacent private
  inline Channel row measures the exact four-byte event-enabled publication
  below `OutputState` and drains both the advisory event and data queue outside
  timing. Fanout 1, 2, and 4 component rows separately time one HoloMQ-backed
  `Output<T>::emit()`, producer publication through the post-source cut, and
  the complete sequential readiness/receive/retirement work for all sinks,
  with the unmeasured half keeping every bounded endpoint reusable. A matched
  two-dispatch row wraps every sink's poll and consume
  transaction in the production busy-lane handoffs while deliberately excluding
  global candidate selection. Matched direct-retained cycle and emit rows replace the HoloMQ-backed
  service with the compiler-authorized synchronous 1:1 retention path while
  preserving the same activation, readiness, receive, and retirement protocol.
  The complete typed-POD cycle reports `readiness_poll_retries_per_cycle` and
  `comparison_uncontaminated`; a zero comparison flag means asynchronous pump
  retries added readiness polls, so that run is not a clean direct-versus-HoloMQ
  component comparison.
  A fanout fused-poll/consume ceiling uses one busy-lane task per sink instead
  of the production-shaped two-dispatch row. It omits global candidate ordering
  and therefore measures continuation potential rather than an implemented
  Runtime path.

`runtime_payload_benchmark` covers one steady-state compiled-Graph catalog
publication and independent catalog receive for:

- a complete custom-POD contract with a fixed array field, forcing the
  fieldwise codec rather than the legacy inline-POD bit-cast path;
- fixed-size owning `schema::TimeT`;
- owning `schema::StringT` at 16, 256, and 384 bytes; and
- nested owning `schema::ImuT` with a header, timestamp, frame string, and
  vector fields.

`runtime_graph_benchmark` is a native production-Graph harness rather than a
Google Benchmark microbenchmark. Its initial paced-latency row compiles a blocking depth-one
`Output<BenchmarkPOD64>` to `Input<BenchmarkPOD64>` edge, starts a
persistent `RunSession` with `run_async()`, and measures real source-to-sink
`compute()` callbacks. Graph compilation, Runtime materialization, warmup, one
allocation-probe sample, stop, and join are excluded from its measurement
window. The allocation probe records callback-local source and sink allocation
counts without adding allocation-counter work to measured samples.

`runtime_graph_interprocess_benchmark` extends that production-Graph boundary
across same-host process boundaries. The Python coordinator starts receiving
processes first, waits on inherited readiness pipes after their selected
process groups reach lifecycle Start, and only then starts the producer. Each
process independently compiles the same Graph and calls the private deployment
seam for exactly one compiler-produced process group. The coordinator rejects
mismatched Graph hashes, plan digests, HoloMQ service identities, clock-domain
evidence, process IDs, process-group IDs, lifecycle starts, or final counts
before summarizing a run.

The distributed POD rows carry a 64-byte custom POD over blocking internal
Graph edges. The `1:1` topology supports paced `latency` and
notification-driven `saturated` workloads. The `1:2` topology launches two
independently selected consumer processes and supports the same two workloads
through one compiler-owned HoloMQ broadcast service. The `chain` topology runs
`source -> relay -> sink` in three separate processes across two
compiler-generated HoloMQ services and supports both workloads. It reports
first-hop, relay-residence, second-hop, and complete end-to-end distributions in
addition to producer, relay, and consumer callback costs. All topologies use one
same-host Linux `CLOCK_MONOTONIC` domain for cross-process subtraction. Process
creation, independent Graph compilation, endpoint discovery, readiness
coordination, warmup, stop, validation, and JSON aggregation remain outside the
measurement window. Each role also reports opt-in Runtime phase counts for
controller wait/wake attribution, Channel ingress, readiness polling, and
callback dispatch. Catalog import/export is intentionally not involved. Latency
rows require queue depth one or two; use depth one for a one-in-flight transport
comparison that does not include avoidable queue residence.

The `1:1` topology also carries a 1 MiB Tensor through compiler-owned HoloMQ
BufferHandle storage using `pageable_host`, `pinned_host`, or `device` backing.
The capture timestamp travels in `SampleMetadata`, so latency measurement does
not modify Tensor payload bytes. Two bulk-work profiles keep different costs
separate:

- `full_write` writes the complete Tensor before every publication. Saturated
  throughput includes that work; latency starts after commit but can include
  unfinished asynchronous device work and its readiness wait; and
- `descriptor_only` initializes each pool backing before it enters the measured
  window, then publishes immutable backing without per-sample payload writes.
  This is the Graph-level transport/control throughput row. The runner rejects
  it if a previously unseen backing requires initialization during measurement.

The matching `image` row uses the same final provider-backed Tensor loan and
BufferHandle transport while separately encoding generated `schema::ImageT`
metadata. The descriptor does not own a foreign Tensor: the producer writes
directly into the compiler-owned host, pinned-host, or device pool and
`emit_tensor()` publishes that backing without schema staging or a Tensor data
copy. This keeps Image metadata overhead distinguishable from the plain Tensor
row at the same 1 MiB payload size.

The two-field generated-schema rows compare the multi-Tensor publication APIs.
`multi_tensor_loan` acquires one transactional bundle of compiler-owned field
buffers using shape-only requests whose dtypes come from the admitted field
contracts, then writes the complete bundle with the typed `write_all()` batch API.
`multi_tensor_staging` emits generated objects containing foreign Tensors, so
the Runtime copies both fields into their compiler-owned staging pools before
publication. Both rows currently use equal-size pageable-host fields and the
paced `latency` workload:

```sh
./public/run launch --run-cmd \
  "python3 ../benchmark/runtime/run_runtime_graph_interprocess_benchmark.py \
     benchmark/runtime/runtime_graph_interprocess_benchmark \
     --payload multi_tensor_loan --tensor-bytes 1048576 \
     --tensor-storage pageable_host --bulk-work-profile descriptor_only \
     --workload latency --samples 1000 --warmup 100 \
     --queue-depth 1 --publisher-period-us 1000 \
     --output-dir /tmp/runtime-graph-interprocess-multi-tensor-loan"

./public/run launch --run-cmd \
  "python3 ../benchmark/runtime/run_runtime_graph_interprocess_benchmark.py \
     benchmark/runtime/runtime_graph_interprocess_benchmark \
     --payload multi_tensor_staging --tensor-bytes 1048576 \
     --tensor-storage pageable_host --bulk-work-profile descriptor_only \
     --workload latency --samples 1000 --warmup 100 \
     --queue-depth 1 --publisher-period-us 1000 \
     --output-dir /tmp/runtime-graph-interprocess-multi-tensor-staging"
```

The final warmup publication is an allocation probe excluded from timing for
every schema row. The multi-Tensor producer and consumer callbacks must each
report zero allocations when the consumer reads the Runtime-owned generated
object through `selected_batch()`. The fixed-field `schema::TimeT` owning row
must also report zero allocations. The singular `schema::ImageT` row instead
records the allocations performed by `Input<T>::receive()` without treating
them as transport regressions: that accessor intentionally makes an independent
generated C++ object and attaches a finite read claim so its selected Runtime
slot remains protected while the object escapes the callback. Tensor bytes are
still shared through their retained backing. Pointer-owned generated fields can
allocate while making that semantic copy; use `selected_batch()` when an
activation-scoped borrowed value is sufficient. Both multi-Tensor rows also
support the notification-driven `saturated` workload; their bounded smoke cases
prove that finite multi-field reservation exhaustion reaps completed transport
releases and resumes publication.

The matching `schema` row carries owning `schema::TimeT` values through the
generated FlatBuffer encode, verify, and materialize path. It supports both
latency and saturated workloads, reports the native application object size,
and leaves `inline_transport_bytes` unknown because the Runtime submits a
variable encoded prefix rather than exposing the exact per-message encoded
length.

```sh
./public/run launch --run-cmd \
  "python3 ../benchmark/runtime/run_runtime_graph_interprocess_benchmark.py \
     benchmark/runtime/runtime_graph_interprocess_benchmark \
     --workload latency --samples 1000 --warmup 100 \
     --queue-depth 1 --publisher-period-us 5000 \
     --producer-cpu 8 --consumer-cpu 10 \
     --producer-controller-cpu 12 \
     --consumer-controller-cpu 14 --consumer-pump-cpu 16 \
     --runtime-phase-trace \
     --output-dir /tmp/runtime-graph-interprocess-1to1"
```

The lane CPU options pin Operator callbacks but do not pin Runtime support
threads. The controller and pump options above assign those roles separately
and the runner rejects a result unless each requested affinity is read back
from the actual thread. Use distinct CPUs for all five roles to avoid turning
support-thread contention into apparent transport latency.
`--runtime-phase-trace` additionally allocates a bounded consumer-side trace
and reports the transport-wait, typed-ingress, pump-wake, controller-return,
readiness-poll, lane-dispatch, and callback-entry intervals. The result also
separates samples that woke a parked controller from samples observed while the
controller was already active. It is diagnostic instrumentation and must
remain disabled for ordinary timing baselines.
Add `--busy-poll` for a controlled low-latency ceiling in which both Operator
lanes remain active between samples. The default remains the production
sleeping-waitset placement; report busy-poll results separately because they
consume two dedicated CPU cores continuously.

For Tensor transport throughput without a per-sample 1 MiB producer write:

```sh
./public/run launch --run-cmd \
  "python3 ../benchmark/runtime/run_runtime_graph_interprocess_benchmark.py \
     benchmark/runtime/runtime_graph_interprocess_benchmark \
     --payload tensor --tensor-storage device --tensor-bytes 1048576 \
     --bulk-work-profile descriptor_only \
     --workload saturated --samples 10000 --warmup 1000 \
     --queue-depth 1 --publisher-period-us 0 \
     --output-dir /tmp/runtime-graph-interprocess-tensor-device-descriptor"
```

For the fixed-size generated-schema latency row:

```sh
./public/run launch --run-cmd \
  "python3 ../benchmark/runtime/run_runtime_graph_interprocess_benchmark.py \
     benchmark/runtime/runtime_graph_interprocess_benchmark \
     --payload schema --workload latency \
     --samples 1000 --warmup 100 --queue-depth 1 \
     --publisher-period-us 5000 \
     --output-dir /tmp/runtime-graph-interprocess-schema-time"
```

For `1:2`, provide one lane, controller, and pump CPU per consumer:

```sh
./public/run launch --run-cmd \
  "python3 ../benchmark/runtime/run_runtime_graph_interprocess_benchmark.py \
     benchmark/runtime/runtime_graph_interprocess_benchmark \
     --fanout 2 --workload latency --samples 1000 --warmup 100 \
     --queue-depth 2 --publisher-period-us 5000 \
     --producer-cpu 8 --consumer-cpus 10,11 \
     --producer-controller-cpu 12 \
     --consumer-controller-cpus 14,15 --consumer-pump-cpus 16,17 \
     --output-dir /tmp/runtime-graph-interprocess-1to2"
```

The source process has no receive endpoint, so it needs an Operator lane and a
controller but no Channel pump. Every consumer needs all three. The summary
keeps logical publications/s, aggregate deliveries/s, and the slowest sink's
completion rate and latency distinct.
The paced `1:2` row uses a two-slot recipient queue so one delayed sleeping
consumer cannot turn an otherwise idle multi-recipient send into an ambiguous
post-submit pressure outcome. The `1:1` latency row remains depth one.
Distributed role records also distinguish the notification-source capacity
from the edge queue depth, even though the current distributed harness binds
both to the requested `--queue-depth`. The runner verifies both values before
accepting a result.

Compatible fanout arms from one graph output are lowered to one compiler-owned
HoloMQ broadcast service. This removes the former sequential per-service
partial-publication window and preserves one logical publication across the
complete recipient set. Graph-owned blocking publication uses bounded retry for
transient pressure without retrying a partially observed logical publication.
The maintained saturated `1:2` row uses queue depth two and validates exact
ordered delivery at both sinks. Compare it with
`graph_interprocess_1to1_throughput_q2` for matched payload and queue geometry:

```sh
./public/run launch --run-cmd \
  "python3 ../benchmark/runtime/run_runtime_graph_interprocess_benchmark.py \
     benchmark/runtime/runtime_graph_interprocess_benchmark \
     --fanout 2 --workload saturated --samples 10000 --warmup 1000 \
     --queue-depth 2 --publisher-period-us 0 \
     --producer-cpu 8 --consumer-cpus 10,11 \
     --producer-controller-cpu 12 \
     --consumer-controller-cpus 14,15 --consumer-pump-cpus 16,17 \
     --output-dir /tmp/runtime-graph-interprocess-1to2-saturated"
```

For the three-process chain, the relay receives and forwards the exact sequence
before the sink consumes it:

```text
source process          relay process                    sink process
lane + controller       lane + controller + pump         lane + controller + pump
      Output<POD>  ==>  Input<POD> -> Output<POD>  ==>  Input<POD>
                    HoloMQ service 1                 HoloMQ service 2
```

```sh
./public/run launch --run-cmd \
  "python3 ../benchmark/runtime/run_runtime_graph_interprocess_benchmark.py \
     benchmark/runtime/runtime_graph_interprocess_benchmark \
     --topology chain --workload latency --samples 1000 --warmup 100 \
     --queue-depth 1 --publisher-period-us 1000 \
     --producer-cpu 8 --relay-cpu 9 --consumer-cpu 10 \
     --producer-controller-cpu 12 \
     --relay-controller-cpu 13 --relay-pump-cpu 15 \
     --consumer-controller-cpu 14 --consumer-pump-cpu 16 \
     --output-dir /tmp/runtime-graph-interprocess-chain"
```

Tensor fanout/chain, dynamic owning schemas, `1:3`, and lossless
fanout-throughput rows remain follow-ups.

These scenarios share that harness:

- `source_only` compiles one portless `OnNotified` Operator and measures notification-driven callback
  execution without a Channel endpoint/pump, downstream input poll, or sink callback. It supports
  `notification_latency` and `saturated`, and reports zero publications and deliveries; and
- `pod` carries the complete 64-byte `BenchmarkPOD64` over one process-local edge;
- `signal` carries one explicitly accounted byte solely to preserve causal source-to-sink activation. It is
  the no-application-payload floor, not a claim of a zero-byte transport operation; and
- `sample_forward` adds `--relay-count` intermediate Operators that call `receive()` and forward the owning
  envelope with `emit(Sample<T>)`, preserving the source `frame_id` through every process-local edge. The
  count defaults to one and is bounded to 64; all relays share one relay role in separate-lane mode so a
  depth sweep changes the Operator/edge count without also adding worker threads; and
- `schema` carries fixed-size owning `schema::TimeT` through the production generated-schema encode,
  verifier, materialization, metadata, and activation path; and
- `tensor` exercises the production `Output<Tensor>::allocate_tensor()` loan, write/commit, emit,
  readiness, `Input<Tensor>::receive()`, and retirement path for pageable host, pinned host, or CUDA-device
  storage. It verifies direct-pool pointer identity for every sample and validates final payload bytes after
  the measurement interval. `--payload-validation all` additionally validates each sample and intentionally
  adds validation work to the completion-rate path; and
- `image` carries bounded generated `schema::ImageT` metadata plus an embedded immutable host Tensor. It
  verifies exact backing-pointer retention and final payload bytes; use `--tensor-bytes` to set the backing
  size (`--tensor-storage pageable_host` is currently required).

The `pod`, `signal`, `sample_forward`, `schema`, `tensor`, and `image` scenarios use compiler-selected
direct-retained delivery for eligible process-local edges with total fanout one. Blocking and
`drop_oldest` are eligible at every accepted positive depth. They therefore construct no HoloMQ endpoint, WaitSet attachment,
Tensor sidecar record, or Channel pump. The benchmark's `--queue-policy <blocking|drop_oldest>` comparison exposes lossless and lossy
behavior for `pod`, `schema`, `tensor`, and `image`. The POD smoke matrix covers depth one separately
from depth two because the mailbox-sized unread ingress has distinct capacity geometry. Lossy rows
require fanout one. Pass
`--force-holomq-process-local-inline` to preserve the prior transport-backed path as an exact
same-binary control; the result name gains `_holomq_fallback`, the JSON records
`process_local_inline_delivery` and `queue_policy`, and the runner then admits `--channel-pump-cpu`. This is
a private measurement control, not application authoring policy.

The `pod` and `schema` scenarios accept `--fanout` from 1 through 8. Fanout greater than one creates one
source output connected to that many independently validated sinks and necessarily uses the process-local
HoloMQ path because direct retention is restricted to total fanout one. In separate-lane mode, the source
and every sink own distinct execution partitions, so a 1:N sweep measures both delivery fanout and the
corresponding N sink workers. A
logical sequence completes only after its slowest sink receives it; notification pressure is bounded by
those logical completions rather than aggregate deliveries. Results retain per-sink delivery/final-sequence
evidence, source-to-sink percentiles for every sink index, and logical publications/s separately from
aggregate deliveries/s. Use a forced-HoloMQ fanout-one control when attributing the fanout slope so the 1:1
direct-retained fast path is not conflated with fanout cost.

Use `--operator-lanes per_sink` to assign a distinct authored CPU to each POD or schema fanout subscriber.
The partition and worker topology is the same as separate-lane fanout, but optional
`--sink-cpus 8,10,12,14` pins those lanes in sink-index order and must contain exactly one CPU per sink. The
runner rejects `--sink-cpu` in this mode, checks that all requested benchmark partitions use distinct
logical and physical cores, and validates the compiled partition/lane inventory. This mode is a placement
and handoff comparison; it is not a recommendation to dedicate one production thread to every subscriber.
Separate-lane fanout records every sink worker as unpinned and rejects singular `--sink-cpu`, because one
ordinal cannot describe distinct placement for multiple workers. Use `per_sink` plus `--sink-cpus` when
authored placement is required.

Latency results also timestamp immediately after a successful source `emit()`. The resulting source-emit
and post-emit-to-each-sink distributions split publication from the source cut and downstream admission.
Source pre-return and sink callback-entry timestamps further isolate controller/lane transit from residual
source bookkeeping and typed receive. These diagnostic boundaries add two source `steady_clock` reads and
one read per sink for each measured publication; they are not application work.

The benchmark authors an exact partition cover: separate-lane runs place the source, relay chain, and sink
in distinct named partitions, while shared-lane runs place every Operator in `runtime-graph-shared`.
Because an explicit planner selection and a manual partition cover intentionally conflict, the Python
runner clears any inherited `HOLOSCAN_EXPERIMENTAL_PARTITION_ALGORITHM` value and requires the
`core_default` selection. Native JSON, `summary.json`, and `manifest.json` retain the accepted
`serial_baseline` algorithm, core-default provenance, and compiled partition inventory so relay
measurements cannot be mistaken for a different execution shape.

The edge-bearing scenarios support three distinct workloads. `latency` uses the admitted `OnClock` trigger and records
one source and sink timestamp per measured sample. `notification_latency` uses a `NotificationSource`, keeps
exactly one witness in flight, waits for its sink delivery, then idles for `--publisher-period-us` before the
next post so the controller must park. It reports notification-to-source-callback,
notification-to-sink, and source-emit-to-sink distributions. `saturated` uses the same public source but
drives it continuously from a foreign benchmark thread until the source publishes the fixed batch.
`--notification-capacity` defaults to one and allows controlled saturated source-pressure sweeps without
changing the production readiness path. For blocking, the saturated driver keeps at most that many accepted
notifications ahead of completed sink deliveries. For a lossy policy, it instead advances behind completed
source callbacks so the source can exercise replacement rather than being serialized by the sink. The
capacity must not exceed queue depth. The saturated row records no
per-sample latency timestamps: its completion window begins at the first measured source callback and ends
when the sink receives the final measured sample. Notification post attempts, accepted witnesses, not-ready
startup retries, source-capacity backpressure, emit backpressure, callbacks, publications, and deliveries
are reported separately. `consumer_visible_replacements` is accepted publications minus actual deliveries
for the admitted fanout-one lossy rows, and `replacement_rate` exposes that fraction.

`source_only` is the deliberate exception to the shared workload matrix. Under `notification_latency`, it
completes one activation before posting the next witness and uses notification-to-source callback time as
its primary `latency_ns`. Under `saturated`, the driver bounds accepted notifications by completed source
activations because no sink delivery exists to retire credits. Compare these rows with matching `pod` rows
to bound Channel materialization, downstream readiness polling, and the sink activation; do not report
source activations as publications.

For matched 1:1 transport comparisons, edge-bearing latency rows also report
`publication_to_receive_ns`. Its timestamp begins immediately before the
source's public `Output<T>::emit()` call and ends after the sink successfully
receives the sample. This is the common publication-to-receive anchor used by
the transport-suite normalizer. The broader `latency_ns` metric and the
notification, callback, scheduling, and retirement subintervals remain
available for Runtime attribution; they are not substituted for the common
anchor.

For a low-rate notification wake measurement:

```sh
./public/run launch --run-cmd \
  "./benchmark/runtime/runtime_graph_benchmark \
     --workload notification_latency --scenario pod --queue-depth 1 \
     --publisher-period-us 5000 --samples 1000 --warmup 100 \
     --output /tmp/runtime-graph-pod-notification-latency.json"
```

By default, each participating Operator belongs to a distinct execution partition
(`--operator-lanes separate`), so a
1:1 graph uses separate source and sink execution-lane workers. Use `--operator-lanes shared` to assign
every participating Operator to one exact partition and measure iterative inline serial execution on the
RunSession controller thread without an execution-lane worker or inter-Operator handoff. Shared mode is
available for paced `OnClock`, `notification_latency`, and `saturated` workloads. `--source-cpu` places the
inline partition owner while its callbacks execute; `--sink-cpu` and `--relay-cpu` are rejected because
those workers do not exist. The native result records the lane mode, accepted explicit Operator partition
count, compile-accepted CPU for every logical execution partition, and zero execution-lane worker samples
for the inline partition.

```sh
./public/run launch --run-cmd \
  "./benchmark/runtime/runtime_graph_benchmark \
     --workload notification_latency --scenario pod --operator-lanes shared \
     --source-cpu 2 --publisher-period-us 5000 \
     --samples 1000 --warmup 100 \
     --output /tmp/runtime-graph-pod-shared-lane.json"
```

For the matching source-only activation boundary:

```sh
./public/run launch --run-cmd \
  "./benchmark/runtime/runtime_graph_benchmark \
     --workload notification_latency --scenario source_only --queue-depth 1 \
     --publisher-period-us 5000 --samples 1000 --warmup 100 \
     --output /tmp/runtime-graph-source-only-notification-latency.json"
```

For the source-only saturated callback ceiling:

```sh
./public/run launch --run-cmd \
  "./benchmark/runtime/runtime_graph_benchmark \
     --workload saturated --scenario source_only --queue-depth 8 \
     --notification-capacity 8 --samples 10000 --warmup 1000 \
     --output /tmp/runtime-graph-source-only-saturated.json"
```

For a short saturated POD run:

```sh
./public/run launch --run-cmd \
  "./benchmark/runtime/runtime_graph_benchmark \
     --workload saturated --scenario pod --queue-depth 2 \
     --notification-capacity 2 \
     --samples 10000 --warmup 1000 \
     --output /tmp/runtime-graph-pod-saturated.json"
```

Saturated throughput is a scheduler-plus-graph measurement, not a Channel-only send-rate row. The source
notification capacity and queue depth are both part of its geometry, and the
completion boundary includes the slowest 1:1 sink. Before quoting a saturation
ceiling, sweep matched capacities and depths; capacity one is the serialized
post-to-complete control:

```sh
for capacity in 1 2 4 8; do
  ./public/run launch --run-cmd \
    "./benchmark/runtime/runtime_graph_benchmark \
       --workload saturated --scenario pod \
       --queue-depth ${capacity} --notification-capacity ${capacity} \
       --samples 10000 --warmup 1000 \
       --output /tmp/runtime-graph-pod-saturated-capacity-${capacity}.json"
done
```

Use the HoloMQ and Channel benchmarks to attribute transport mechanics below
this production Operator path.

Runtime results also report publication-backpressure capacity. Tensor and
Image rows add pool slots, release capacity, and maximum fanout release
streams from the accepted execution plan; non-bulk rows explicitly report no
pool. The Python runner checks this bounded-resource evidence after each run.
The transport-suite normalizer combines native publication and
`backpressured_emits` counters into attempted, accepted, backpressured,
indeterminate, and completed publication counts without placing any validation
inside the measured callbacks.

Every result records payload bytes, relay count, pipeline-edge count, queue policy, consumer-visible
replacements, and replacement rate. Saturated rows report logical publication rate from source samples plus
actual aggregate delivery and byte rates across every traversed edge, so a lossless chain with N relays
reports N+1 edge deliveries per logical publication while a lossy row cannot hide dropped samples behind
its accepted-publication rate. Paced latency
rows report those throughput-only fields as `null` and instead record
`configured_publication_rate_per_second`; that value is the pacing input, not
an achieved throughput result.

For matched direct-retained and forced-HoloMQ policy rows, keep every option fixed except the private
fallback flag. Re-run the blocking baseline with the same geometry:

```sh
for policy in blocking drop_oldest; do
  depth=1
  test "${policy}" = drop_oldest && depth=2
  python3 ../benchmark/runtime/run_runtime_graph_benchmark.py \
    ./benchmark/runtime/runtime_graph_benchmark \
    --workload saturated --scenario pod --queue-policy "${policy}" \
    --queue-depth "${depth}" --notification-capacity "${depth}" \
    --samples 100000 --warmup 1000 \
    --output-dir "runtime-pod-${policy}-direct"
  python3 ../benchmark/runtime/run_runtime_graph_benchmark.py \
    ./benchmark/runtime/runtime_graph_benchmark \
    --workload saturated --scenario pod --queue-policy "${policy}" \
    --queue-depth "${depth}" --notification-capacity "${depth}" \
    --force-holomq-process-local-inline \
    --samples 100000 --warmup 1000 \
    --output-dir "runtime-pod-${policy}-holomq"
done
```

Use the same drop-oldest comparison for generated schema, raw Tensor, and descriptor-backed Image:

```sh
for scenario in schema tensor image; do
  for delivery in direct holomq; do
    fallback=
    test "${delivery}" = holomq && fallback=--force-holomq-process-local-inline
    python3 ../benchmark/runtime/run_runtime_graph_benchmark.py \
      ./benchmark/runtime/runtime_graph_benchmark \
      --workload saturated --scenario "${scenario}" \
      --queue-policy drop_oldest --queue-depth 2 --notification-capacity 2 \
      --tensor-storage pageable_host --tensor-bytes 1048576 \
      ${fallback} --samples 100000 --warmup 1000 \
      --output-dir "runtime-${scenario}-drop-oldest-${delivery}"
  done
done
```

Repeat with `--workload latency --publisher-period-us 1000` to collect paced p50/p99. Use controlled
repetitions and the runner's quality gate before quoting performance; short smoke rows only validate the
measurement path.

For example, compare a four-relay direct-retained chain with its transport-backed control:

```sh
./public/run launch --run-cmd \
  "python3 ../benchmark/runtime/run_runtime_graph_benchmark.py \
     ./benchmark/runtime/runtime_graph_benchmark \
     --workload saturated --scenario sample_forward --relay-count 4 \
     --operator-lanes shared --busy-poll \
     --queue-depth 2 --notification-capacity 2 \
     --samples 100000 --warmup 1000 \
     --output-dir runtime-chain-direct-4"
```

Add `--force-holomq-process-local-inline` and select a different output directory for the matched HoloMQ
fallback. Use relay counts 1, 2, 4, and 8 for the planned scaling sweep.

For example, measure POD fanout four through the production transport-backed path:

```sh
./public/run launch --run-cmd \
  "python3 ../benchmark/runtime/run_runtime_graph_benchmark.py \
     ./benchmark/runtime/runtime_graph_benchmark \
     --workload saturated --scenario pod --fanout 4 \
     --operator-lanes shared --busy-poll \
     --queue-depth 2 --notification-capacity 2 \
     --samples 100000 --warmup 1000 \
     --output-dir runtime-pod-fanout-4"
```

Use `--fanout 1 --force-holomq-process-local-inline` for the mechanism-matched 1:1 control. The maintained
fanout sweep uses 1, 2, and 4; values through 8 are admitted for bounded stress and follow-up scaling work.

For example, run a 1 MiB CUDA-device Tensor row on device zero with:

```sh
./public/run launch --run-cmd \
  "./benchmark/runtime/runtime_graph_benchmark \
     --scenario tensor --tensor-storage device --tensor-bytes 1048576 \
     --tensor-device 0 --samples 10000 --warmup 1000 \
     --publisher-period-us 1000 --output /tmp/runtime-tensor-device.json"
```

Tensor results also report the admitted direct-allocation and device-resource record counts, plus whether the
compiler assigned the strict process-local 1:1 edge to a fused linear stream. These are immutable plan facts;
the timed row does not install the mutex-backed source-tree Runtime test counter.
For device storage, the default `final` validation makes `latency_ns` and
`source_to_sink_ns` descriptor/handle-propagation measurements: the producer's
`cudaMemsetAsync` is ordered but not synchronized before the sink timestamp.
Use `--payload-validation all` when the measured path must include the
consumer-stream wait, device-to-host copy, and synchronization that prove the
payload is data-ready. Use `--payload-validation none` for production transport
measurements that must exclude payload-content validation. This mode still
checks sequence, descriptor, metadata, delivery counts, and session
diagnostics. `final` validates retained content after the measured batch;
`all` validates every sample and can therefore affect completion-rate timing.

The native result uses the versioned
`holoscan.ea1.runtime-benchmark/v1` JSON schema. Each row records the precise
question it answers, clock source, workload geometry, compile-accepted lane
placement, completion-window throughput, callback/publication/delivery counts,
validation, and session diagnostics. Latency rows additionally report p50,
p90, p99, min, and max from preallocated per-sample tables; p99.9 is reported
only with at least 10,000 measured samples. For edge-bearing scenarios, `notification_latency` uses
post-to-sink as its primary `latency_ns` and retains the component `notification_to_source_ns`,
`source_callback_entry_to_pre_emit_ns`, `source_publication_call_ns`, and `source_to_sink_ns`
distributions. Callback-to-pre-emit covers source-side preparation before the public `emit()` call;
publication-call covers that call through its successful return. These components reuse timestamps already
needed by the other boundaries and add no timed-path clock reads. Tensor rows additionally report
`tensor_loan_acquire_ns`, `tensor_write_commit_ns`, and `tensor_receive_call_ns`. Those diagnostic
boundaries add two source-side and one sink-side clock read per measured Tensor sample; compare their
sub-microsecond values with the reported timestamp-read-pair calibration rather than treating them as an
unperturbed application path. A concurrently running sink can enter or complete before the producer records
its post-`emit()` or pre-return boundary. The corresponding
`source_post_emit_to_sink_overlaps` and per-sink `overlaps` fields classify those valid concurrent samples
instead of misreporting them as missing; every overlap-sensitive interval requires
`latency_ns.samples + overlaps` to equal the requested measured sample count. The
source-only row uses post-to-source for both its primary latency and `notification_to_source_ns`; the two
downstream component fields are absent. Repeated-run summaries report the
median plus minimum, maximum, and coefficient of variation for the primary
metric. Production rows use at least 1,000 measured samples for p99 evidence
and suppress p99.9 below 10,000 samples.

The graph compile, repeatable Operator declaration replay, endpoint attachment, and peer
discovery occur before timing. Each timed iteration includes Runtime encode,
catalog framing, HoloMQ submission/receive, catalog decode or generated-schema
verify/materialize, and public `Sample<T>` construction. The benchmark reports
wire size and thread-local allocations per sample in addition to elapsed time.

The READY candidates and accepted `ExecutionPlan` are constructed before the
timed loops. Dispatcher storage is reserved by `Dispatcher::create`; each timed
cycle reuses that storage. The injected direct run uses the same private `PortBinding`
factory seam and public free run path as the runtime conformance tests. It does
not expose a benchmark-only public API or add data queues to Operator ports.

Run the short benchmark smoke from the repository root:

```sh
./public/run bench --name runtime_dispatcher
./public/run bench --name runtime_channel
./public/run bench --name runtime_payload
```

Run the production Graph harness directly in the development container:

```sh
./public/run launch --run-cmd \
  "./benchmark/runtime/runtime_graph_benchmark \
     --workload latency --publisher-period-us 1000 \
     --samples 10000 --warmup 1000 --output /tmp/runtime-graph-latency.json"
```

Use the Python runner for repeated measurements. It preserves each native JSON and stdout/stderr log, then
writes `summary.json` and `manifest.json` with run-to-run median-latency dispersion, exact native commands,
Git/build/host/GPU metadata, requested Operator and Runtime support-thread placement, observed support-thread
affinity and native thread IDs, resolved CPU topology, and scaling governor. If
`--output-dir` is omitted, the runner prints the generated temporary artifact
directory to standard error. Repeated latency summaries retain p99 and also
p99.9 when the native 10,000-sample minimum is met:

```sh
./public/run launch --run-cmd \
  "python3 ../benchmark/runtime/run_runtime_graph_benchmark.py \
     ./benchmark/runtime/runtime_graph_benchmark \
     --scenario pod \
     --samples 10000 --warmup 1000 --repetitions 10 \
     --publisher-period-us 1000 --source-cpu 2 --sink-cpu 3 \
     --controller-cpu 4 \
     --git-revision $(git rev-parse HEAD) \
     --output-dir /tmp/runtime-graph-pod-latency"
```

When the runner is launched inside the SDK container, the outer shell expands the revision before
`public/run` starts the container. `HOLOSCAN_GIT_REVISION` provides the equivalent automation hook.

For notification-driven rows, also pass `--notification-driver-cpu` on a distinct physical core. The Operator CPU
options use the public compile-time lane-placement contract, whose startup barrier applies and verifies the
accepted placement. `--controller-cpu`, transport-backed `--channel-pump-cpu`, and
`--notification-driver-cpu` use a
source-tree-only measurement control: it applies Linux affinity once when each thread enters, reads the
affinity back, and records the native thread ID. This is deliberately not a public application placement API
or a serialized plan field, and it adds no per-activation work.

Use `--runtime-phase-counters` only for diagnostic runs. It installs a private, source-tree-only fixed counter
block and a capped, preallocated phase-trace buffer. The counters record controller readiness scans, waits and
wakes, candidate freeze/select transactions, input-poll dispatches, compute dispatches, completed activations,
and Runtime Channel prepare, emit, receive, readiness, read, witness-retirement, activation-retirement,
activation-cut, and backpressure boundaries. The trace timestamps readiness-poll return, candidate freeze,
compute-dispatch start/return, activation retirement, and activation-cut completion, then reports per-receiver
phase distributions.
It also reports source-activation runs, their maximum consecutive length, and all four
source/downstream transition counts so queue-depth diagnostics can distinguish batching from payload work.
For Tensor scenarios, the same bounded trace pairs output-port and sample-sequence identities and
reports total emit, validation, pre-transport, transport-publication, and post-transport
distributions. Direct-retained Tensor rows have no transport-publication interval; forced-HoloMQ
rows require all five Tensor checkpoints for every accepted publication.
Inline POD and generated-schema scenarios similarly report total emit, encoding, pre-transport,
transport-publication, and post-transport distributions. Direct-retained rows stop after encoding;
transport-backed rows require all five inline checkpoints for every accepted publication.
The runner rejects a diagnostic result unless every frozen candidate is selected, dispatched, and completed;
every wait returns by wake or timeout; and the Channel counts match the benchmark's exact callback,
publication, and delivery evidence. It also classifies execution-lane tasks by callback context and verifies
exact task publication, callback entry/exit, controller completion, and admitted busy-versus-sleeping handoff
counts.
`source_compute_entries` can exceed productive `source_callbacks` by one when the bounded notification driver
has already admitted the terminal source witness; that entry invokes `compute()` and returns without
publishing.

Counter increments use relaxed atomics in preallocated cache-line-aligned storage. Trace checkpoints also take
`steady_clock` timestamps, and the fixed buffer records explicit drops if its two-million-record cap is reached.
Both mechanisms perturb the hot path, so keep the flag disabled for timing baselines. The runner retains every
enabled snapshot under `runtime_phase_counter_runs` and `runtime_phase_trace_runs` so diagnostic cardinalities
and phase distributions are not aggregated into a misleading average.

The runner's quality gate is metadata, not a performance test threshold. It flags any unpinned participating
Operator partition or support-thread role, unavailable CPU topology, shared physical cores, a
non-`performance` governor, fewer than two repetitions, or excessive run-to-run CV of per-run p50 latency for
latency workloads
or achieved activation/publication rate for saturated workloads. It also rejects native results whose observed
support-thread affinity does not match the request. Valid measurements from ordinary developer and CTest
environments are still preserved without being presented as controlled baselines.

For repeated characterization and JSON output:

```sh
./public/run bench --stable-system --name runtime_dispatcher \
  --output-json runtime-dispatcher-benchmark.json \
  --benchmark-args \
  "--benchmark_min_time=0.01s --benchmark_repetitions=10 \
   --benchmark_report_aggregates_only=true"

./public/run bench --stable-system --name runtime_channel \
  --output-json runtime-channel-benchmark.json \
  --benchmark-args \
  "--benchmark_min_time=0.01s --benchmark_repetitions=10 \
   --benchmark_report_aggregates_only=true"

./public/run bench --stable-system --name runtime_payload \
  --output-json runtime-payload-benchmark.json \
  --benchmark-args \
  "--benchmark_min_time=0.01s --benchmark_repetitions=10 \
   --benchmark_report_aggregates_only=true"
```

CTest intentionally applies no latency or throughput threshold. Each target's
smoke runs only enough work to prove the benchmark remains executable. Results
are measurement evidence, not a release gate or a universal performance claim.
`BENCHMARK_RUNTIME_GRAPH` is likewise a threshold-free functional smoke. It
uses all-sample validation and a small bounded batch to prove that the real
Graph harness, result path, and cooperative stop remain executable.

The injected direct-mode row includes platform thread creation and join, so it
must not be compared with the Dispatcher-only row as if both measured the same
unit of work.

The two complete-run rows are directly comparable because they use the same
source, clock, lane, lifecycle, and callback shape. Their difference is a
coarse full-session bound on update-enabled generation-zero pinning and empty
publication/observation/ingress checks; thread creation and teardown dominate
both absolute values. Concurrent control activity is covered by a deterministic
held-mutex regression and requires separate repeated-activation latency
characterization before a quantitative contention claim.

The two 65,536-activation rows provide the corresponding steady-session
comparison. Each benchmark iteration still constructs and tears down one
`RunSession`, but `items_per_second` counts activations rather than sessions;
divide elapsed time by 65,536 for an amortized per-activation value. The source
count is preconfigured in the source-tree direct adapter, which keeps one
capacity-one witness resident and publishes the next only after retirement.
This isolates Runtime dispatch and callback maintenance rather than Channel
transport or producer-side readiness publication.

The direct Tensor pool rows use an unconnected output whose maximum per-compute
burst bound changes only the compiled physical slot count. Each callback exercises
the production host allocation state machine and immediate cancellation
retirement, with no Channel transport or CUDA event. Compare amortized
per-activation time across capacities; the benchmark has no pass/fail latency
threshold and does not replace the separate real-CUDA stream regressions.

The physical readiness row isolates the bounded source-to-Dispatcher handoff
whose atomic ordering protects terminal transitions and source-generation
advancement. It performs no backend I/O, allocation, or user callback in the
timed loop.

The composed controller-activation row keeps its coordinator, Dispatcher,
activation storage, binding, and busy execution lane alive across iterations.
It follows the production transaction order and completes Dispatcher selection
inside the lane callback. It is a component boundary for controller and lane
bookkeeping, not a `RunSession` replacement: the omitted plan lookup, lifecycle,
parameter, deadline, temporal, Channel, and diagnostic paths remain present in
the maintained steady activation row.

The Runtime-wake row keeps one real counting-semaphore wake state and one
driver thread alive for 1,024 iterations. The driver waits for an iteration
request, idles for 5 ms so the benchmark thread parks, records the signal
timestamp, and releases one durable token. Manual iteration time measures only
signal to `wait_for()` return; the idle interval, thread creation, and join are
excluded. The wake and post-idle lane rows retain their raw samples and report
nearest-rank p50, p90, and p99 counters instead of reducing their primary
high-variance result to a mean. This is an unpinned primitive diagnostic rather
than a production Graph or platform baseline. The Runtime
row includes the libstdc++ counting-semaphore abstraction and one relaxed
pending-count update; the Linux futex and eventfd candidates implement the same
durable-token contract directly. None of these rows calls
`RuntimeWakeState::drain()`, so their delta does not measure Runtime wake
coalescing.

The lane-dispatch rows keep one started lane alive across all iterations and
invoke an empty allocation-free task. They isolate the cross-thread handoff and
completion protocol used for both readiness polling and Operator callbacks;
they exclude Dispatcher selection, Channel operations, and session startup or
teardown. Use real time because the controller blocks while the worker runs.
The `enqueue_to_entry_ns` and `exit_to_return_ns` counters split the two
cross-thread halves with benchmark-local `steady_clock` reads; the reported
total includes that instrumentation and remains the primary regression value.

The Channel fixture creates the real private Runtime adapter and its bounded
typed stores before timing. It injects valid scalar metadata through a
source-tree-only retention seam while timing is paused, then measures the exact
production owner index, readiness-poll implementation, or witness-retirement
lookup. Transport refill and teardown are excluded. The readiness-burst rows
also exclude sample retirement, so they measure typed-store-to-readiness work.
The busy-lane readiness row additionally composes the revision gate and exact
`ExecutionLaneSet::dispatch()` handoff used for `PortBinding::poll()` in the
production controller. Its fixed `TimePoint` deliberately excludes the
controller's clock lookup. The two-dispatch and resumable rows keep publication
outside timing and execute the same direct-retained poll, drain, typed receive,
claim retirement, and sink cut. Both omit Dispatcher and activation-policy work;
their difference bounds only the task teardown/republication that a
semantic-preserving continuation protocol could avoid.
The retirement rows time only readiness-witness retirement; injection, polling,
event draining, temporal-record lookup, and final payload-claim release are
paused. Those isolated rows do not measure HoloMQ transport latency or
end-to-end publication; the typed POD cycle is the explicit exception and
includes one complete process-local transport and typed-receive lifecycle.

The payload suite measures one local-host HoloMQ catalog hop, but not
interprocess scheduling or network transport. These suites do not measure
catalog or cross-process Tensor transport, concurrent activations, lifecycle restart transactions, accepted
parameter publication, or distributed workers. Those paths belong to their owning components and require
separate benchmarks. The production Graph Tensor row does cover direct process-local Tensor/CUDA claims; the
older Google Benchmark direct-pool row remains an isolated allocation-state-machine measurement. The finite direct
binding is test/measurement infrastructure only; it is not a claim that
ordinary production inputs are implemented by this adapter.

The native Graph binary performs an excluded warm-up allocation probe around
each Operator callback class. Source and relay callbacks, and all sink
callbacks except `ImageT`, have a zero-allocation limit. The current owning
`ImageT` receive path has an explicit six-allocation sink limit for generated
schema materialization. A count above the scenario limit invalidates the
result, and the Python runner independently verifies both the declared limit
and observed count before aggregating artifacts. This is the intentional cost
of `Input<ImageT>::receive()` producing an independent generated C++ object;
the finite read claim keeps the selected Runtime slot protected, while the
Tensor bytes remain shared. Callback-lifetime code can use the borrowed
selection API when it requires an allocation-free path.
