# Runtime Benchmarks

## Interim channel-pump placement

For graphs that retain a channel pump, give the pump a CPU outside every FIFO
execution lane's affinity set. For example, after binding the graph's lanes:

```cpp
compile_options.deployment.bind_support_thread_affinity(
    holoscan::SupportThreadRole::kChannelPump, holoscan::CPUAffinity{7U});
```

Choose a CPU available to the process and not used by those lanes; `7` is only
an example. In the graph benchmark, use `--channel-pump-cpu` with a CPU distinct
from `--source-cpu` and `--sink-cpus`. The default benchmark checks reject overlap;
`--allow-pump-lane-cpu-sharing` exists for diagnostic comparisons.

Affinity does not grant the pump a FIFO policy. Explicit FIFO lanes can otherwise
share a CPU with an OTHER pump and delay its progress while the producer waits on
shared channel state. Check `placement.support_threads` and `placement.lane_startup`
in the native JSON for actual affinity, policy and priority. Pump isolation is an
interim mitigation for that co-location problem, not a tail-latency guarantee:
ordinary load on the pump's CPU can still delay progress.

## Owning versus borrowed Image input

Run `runtime_graph_benchmark --scenario image --operator-lanes shared --samples 10000 --warmup 100`
with and without `--borrow-image-input`. Owning `receive()` remains the default; the switch selects
`selected_sample()` only in the Image sink, with identical source, storage, and validation settings.
The JSON field `image_input_accessor` is `receive` or `selected_sample` for Image runs and `null`
for other scenarios. Other scenarios reject the switch. The native executable supports it; the
Python benchmark runner does not currently forward it.

Borrowed mode requires zero C++ allocations in the excluded sink-accessor warmup probe, not zero
whole-callback or transport allocations. Both accessors have maintained direct/HoloMQ CTest smokes;
the existing lane, fanout, storage, and workload options permit broader paired experiments without
a new benchmark target. This read-only sink comparison complements
`runtime_descriptor_receive_benchmark` and `runtime_descriptor_graph_benchmark`, which also
qualify scratch edits. Neither comparison measures Vision GPU kernels.

The switch is adapted from JC Fillion-Robin's `51b3e56cfd` on `jcfr/5x/selected-sample-view`, using
the retirement-aware scalar borrow on this branch. Record revision, host tuning, build profile,
run date, and both JSON outputs before making timing claims; CTest smokes are functional checks,
not controlled performance collection. Archive run artifacts outside the SDK source tree.

## Benchmark suites

The source-tree-only Google Benchmark suites measure bounded baseline Runtime paths.
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
- three matched 65,536-activation fixed-rate sessions that retain identical
  authored metadata and execute zero, one, or eight `ExecutionContext::rate()`
  reads per callback. These rows characterize the lazy first-query pin and
  repeated O(1) generation-table lookup. They do not observe stream rate and do
  not replace an exact-base feature-absent comparison; allocation freedom is a
  separate source-tree conformance assertion; and
- paired 65,536-call `BM_RunSessionSteadyOutputNoCompletion` and
  `BM_RunSessionSteadyComputeCallCompletion` sessions. They use the same Operator
  type, empty callback, unconnected output, direct readiness, asynchronous session
  route, and out-of-timing terminal validation. The completion row adds only one
  bounded rule, recurring admission, one output-close decision, and one terminal
  progress record, so this is the supported subtraction pair; and
- complete one- and 1,024-call `BM_RunSessionComputeCallCompletionAndDrain`
  sessions over one compiler-verified process-local Channel edge and two Runtime
  partitions. They include the limit-reaching callback, accepted data delivery,
  per-output EOS, downstream input retirement, natural completion, and one
  terminal progress record; and
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
  global candidate selection. Matched direct-retained cycle and emit rows
  replace the HoloMQ-backed service with the compiler-authorized synchronous
  1:1 retention path while preserving the same activation, readiness, receive,
  and retirement protocol. `BM_ChannelTypedPOD64DirectRetainedCycle` is the
  matched production-Graph floor: it uses the canonical 64-byte benchmark POD,
  blocking depth one, and final validation while still excluding Dispatcher
  selection and both Operator callback transactions.
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

`runtime_descriptor_receive_benchmark` isolates public scalar access from a
prepared selection. It compares `Input<Tensor>::receive()`, `Input<ImageT>::receive()`,
`receive_data()`, generated Image assignment into a previously prepared destination,
`selected_sample()`, and borrowed metadata edits through `ImageDescriptorScratch`, using
interleaved/three-plane metadata and 6/128-byte frame IDs. Every case has real 48-byte CPU Tensor
backing; validation checks independent nested metadata/wrapper objects and unchanged pixel pointers.
It uses a synthetic scalar binding with one read credit, not a compiled Graph: scheduling, transport,
CUDA execution, and production slot retirement are outside this benchmark.

The `first_read_allocations` and `allocations_per_read` counters measure C++ allocations on the
benchmark thread, after cold slot preparation/Tensor import. The latter counts a separate pass of
100 reads. Allocation instrumentation is disabled during the timed loop, which includes accessor
work, bounded validation, and destruction/replacement of the owning value or bounded scratch copy.
Borrowed cases assert zero first-read and repeated allocations. It does not cover arbitrary
native-library `malloc`. The benchmark intentionally reports rather than asserts exact owning-copy
counts because generated code and standard-library allocation details can vary.

Run the CTest smoke `BENCHMARK_RUNTIME_DESCRIPTOR_RECEIVE`, or collect JSON with:

```bash
./run bench --name BENCHMARK_RUNTIME_DESCRIPTOR_RECEIVE \
  --output-json descriptor-receive.json \
  --benchmark-args "--benchmark_min_time=0.1s --benchmark_repetitions=5"
```

`runtime_native_schema_benchmark` extends the component baseline to scalar-only `TimeT`, ordinary
`ImuT`, and an application-owned two-Tensor descriptor with strings, scalar/string vectors, optional
tables, nested tables, and optional structs. It separately measures Runtime codec materialization,
owning receive, scalar borrow, generated assignment plus a scalar edit, encoding, and by-value
`emit()` through an encode-only binding. The last case isolates argument-copy/encode work: it is
**not** production Runtime publication or transport. Accessor cases also use a synthetic selection;
they do not measure scheduling or prove selection retirement.

Each counted operation executes on the benchmark thread. The first operation, the maximum, and the
mean of the following 128 operations are reported separately; counter instrumentation is disabled
for the timed pass, which uses a fresh fixture. The descriptor materialization case performs the
production reset, wire validation, and assignment hooks with separately authoritative host Tensors.
Ordinary materialization constructs a fresh sample and destroys the previous one. Validation stays
inside the measured operation; source creation, wire preparation, and Tensor import stay outside.
Owning paths compare copied field values; borrow validates sample identity and read-credit state.
Their validation costs differ, so timing differences are not pure accessor/copy-cost estimates.
For the legacy materialization case, `cold_slot_allocations` counts the descriptor's cold preparation
hook only, not total fixture memory. Other legacy cases no longer prepare an unused descriptor slot.
These counters cover thread-local C++ `new`, not arbitrary native `malloc`, worker threads, or CUDA.

The `pattern` argument means:

| Pattern | Native-schema component fixture |
| --- | --- |
| 0 | Empty/absent metadata |
| 1 | Present metadata, short strings and small vectors |
| 2 | Present metadata, 128-byte strings and larger vectors |
| 3 | Alternate full and absent metadata |
| 4 | Short/full/empty/full capacity transitions |
| 5 | Cycle all five-bit optional-presence combinations |

IMU presence bits independently select Header, legacy Time, and the three covariances; legacy Time
is meaningful only when Header exists. Descriptor bits select the optional table, its struct and
child, a root struct, and root string-vector presence. A scalar-only Time control uses pattern 0.
Legacy borrowed access asserts zero allocations; other legacy phases report observed counts without
promising portability of exact STL allocation totals.

Prepared-schema modes use the same sources and field validation. The original `Materialize` case
retains the owning ordinary codec and the legacy descriptor callbacks explicitly; enabling bounds
on the fixture does not silently replace that baseline with prepared assignment.

| Additional mode | Measured boundary |
| --- | --- |
| `PreparedMaterialize` | Reset, verification, bounded in-place assignment, and value checks in a cold-prepared slot |
| `PreparedCopyEdit` | Generated scratch copy and scalar edit; descriptor scratch verifies both Tensor members remain null |
| `EmitRefEncodeBinding` | Ordinary-schema `emit_ref()` into the same synchronous encode-only binding; not production publication |
| `ColdPrepareSlot` | Destroy and cold-construct the complete prepared inventory in already allocated aligned slot memory |
| `ColdPrepareScratch` | Destroy and cold-create operator scratch, including its owning state allocation |

The first three modes assert zero first and maximum operation allocations across all patterns.
Reference publication is not available for descriptor schemas: they require separate Tensor-loan
authority. Cold modes run only pattern 0 because preparation is independent of message contents;
they intentionally allocate and report preparation/destruction cost, not whole-Graph startup or
the system allocator's first use. `cold_prepared_slot_bytes` reports generated budget accounting
for one prepared slot, including its native value, private inventory, owners, and container capacity.
`cold_scratch_bytes` and `cold_scratch_allocations` describe one operator scratch. These byte counts
exclude allocator bookkeeping, cold temporary peaks, unrelated transport state, and Tensor pixels.
The fixture's raw aligned slot allocation is excluded from `cold_slot_allocations`; the cold
scratch factory's state allocation is included in `cold_scratch_allocations`.

`runtime_payload_benchmark` supplies the complementary production compiled-catalog round trip for
ordinary schemas. It now counts activation, by-value publication, retirement, and external receive
on their actual executing thread in a separate pass of at least 65 messages (first plus at least
64 repeated messages), extended when needed to visit every value in the declared sequence.
`max_wire_bytes` reports the maximum encoded size across that complete sequence, collected in
the untimed pass; it is independent of the final timed sample and the timed iteration count.
`BM_CompiledCatalogImuTransitions` covers absent, full, alternating, growth, and all optional states
as patterns 0 through 4. These counts include the actual Runtime/Channel path, not just the isolated
codec. Timing instrumentation changed from earlier revisions, which counted allocations inside the
timed loop; do not compare those timings without accounting for that difference.

Run `BENCHMARK_RUNTIME_NATIVE_SCHEMA` and `BENCHMARK_RUNTIME_PAYLOAD` for functional smoke coverage.
For a controlled native-schema collection, after host tuning and all build/test work has finished,
run from the selected SDK container build directory:

```bash
taskset -c 2 benchmark/runtime/runtime_native_schema_benchmark \
  --benchmark_min_time=0.1s --benchmark_repetitions=5 \
  --benchmark_display_aggregates_only=true \
  --benchmark_out=native-schema-REVISION.json --benchmark_out_format=json
```

Choose an appropriate isolated CPU for the host, record the exact revision/build/tuning and command,
and save generated result files outside the SDK source tree. Smoke-test timing is not controlled
performance evidence, and neither harness measures Vision GPU kernels or guarantees whole-graph
allocation freedom.

`runtime_prepared_schema_graph_benchmark` compares ordinary `ImuT` transforms in owning receive,
callback-local borrow, and prepared copy/edit plus `emit_ref()` modes. Each mode runs over direct
retained and forced process-local HoloMQ delivery with tracing disabled and enabled. An untimed
pass reports C++ `new` counts for receive, prepared scratch work, publication, the first callback,
and the whole transform callback. Separate timed runs report mean transform callback cost and
source-to-sink p50/p95/p99 for 512 messages. Startup, native-library `malloc`, and background
transport threads remain outside those counters.

This harness sizes both data queues to the complete 512-message run and lets the source publish
without a retirement-ordered acknowledgment. That pacing keeps the comparison among its twelve
owning/borrowed/prepared cases consistent, but it differs from
`runtime_descriptor_graph_benchmark`, which uses depth-one data queues and grants source credit
only after sink retirement. Do not compare source-to-sink percentiles between the two benchmark
executables; use each only for comparisons that preserve its own pacing protocol.

Build `runtime_prepared_schema_graph_benchmark` and invoke the executable directly for a retained
controlled collection. Record the exact revision, build, affinity, host tuning, and command with
the JSON result, following the same evidence rules as the descriptor graph benchmark below.

`runtime_descriptor_graph_benchmark` complements that isolated accessor benchmark with real
source -> transform -> sink callbacks on three Runtime execution partitions. Modes 0/1/2 are owning
receive, borrowed unchanged metadata, and borrowed metadata edits. Transport 0 uses direct retained
delivery; 1 forces process-local HoloMQ. Every run sends 512 acknowledged messages using an eight-byte
host Tensor, a 128-byte frame ID, depth-one queues, and compiler-derived pools without extra slots.
The transform reads the selected pixels and writes a distinct output loan, preserving provenance.
Each sink publishes a scalar acknowledgment to a small control operator on that sink's existing
serial partition. That operator grants the next source credit only after the data sink's callback
and selected input have retired. An acknowledgment written inside the data callback would allow
the next transform to publish while a depth-one destination is still occupied. The source,
transform, and sink remain on distinct partitions; no data queue or Tensor pool is enlarged.
The control hop is outside the measured transform callback, and latency is recorded in the data
sink before acknowledgment. It still changes inter-message pacing, so compare timing only between
revisions with the same acknowledgment protocol. This harness protocol is not automatic Runtime
retry or a general lossless application backpressure mechanism.

An untimed run counts C++ `new` on the transform worker, separately for receive, metadata copy,
publication, and the whole callback. A second run disables allocation counting and reports mean
transform callback nanoseconds using Google Benchmark manual time. Source-to-sink p50/p95/p99
include scheduling and transport and are not GPU execution or throughput measurements. Both runs
validate pixels, metadata, sequence, provenance, and bounded progress. The prepared encode/decode
allocation test is `RUNTIME.ImageScratchCodec.SeparateTensorAuthorityAndPreparedDecodeAreAllocationFree`.
Native-library `malloc`, CUDA, and third-party wrapper allocations are outside this instrumentation.

Run `BENCHMARK_RUNTIME_DESCRIPTOR_GRAPH` for smoke coverage, or collect a retained comparison:

```bash
./run launch --standalone build-x86_64 --run-cmd \
  "taskset -c 2-4 benchmark/runtime/runtime_descriptor_graph_benchmark \\
   --benchmark_out=descriptor-graph-REVISION.json \\
   --benchmark_out_format=json"
```

Replace `REVISION` with the tested revision and record the exact command, CPU affinity, host tuning,
build options, compiler/library versions, and any dirty source hashes alongside the JSON. Collect
only after compilation and lint finish. This command writes to the selected build directory;
save generated JSON and interpretation outside the SDK source tree. Keep your owning baselines
without overwriting them. Zero borrowed-descriptor
allocations are regression assertions, not a claim that the complete
runtime or any arbitrary transport is allocation-free.

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
callback dispatch. A dispatch rejected by run closure does not enter the operator callback.
The diagnostic payload reports `compute_admissions_run_closing` and `stop_signals`;
validation requires dispatches to equal callbacks plus run-closing admissions, with
at most one such admission and a corresponding stop signal. Catalog import/export
is intentionally not involved. Latency
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
generated C++ object and consumes finite owning-read credit. The owning copy,
not the credit itself, retains its Tensor backing after callback retirement;
the credit does not pin the selected Runtime slot. Pointer-owned generated fields can
allocate while making that semantic copy. Use `--borrow-image-input` for scalar
`selected_sample()` access without that copy; `selected_batch()` serves batch-shaped inputs.
Both multi-Tensor rows also
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
- `image` carries bounded generated `schema::ImageT` metadata plus an embedded immutable Tensor. It
  verifies exact backing-pointer retention and final payload bytes; use `--tensor-bytes` to set the backing
  size and `--tensor-storage` to select pageable-host, pinned-host, or CUDA-device backing; and
- `multi_tensor` carries the generated two-field `runtime_benchmark::MultiTensorMessageT` schema and
  acquires one compiler-owned BufferPool backing independently for each field before publishing the
  composite value. Descriptor-only rows initialize each newly observed field backing during warmup; and
- `multi_tensor_loan` and `multi_tensor_staging` carry the generated two-field
  `runtime_benchmark::MultiTensorMessageT` descriptor schema. The loan row allocates and commits both
  compiler-owned fields as one bundle; the staging row emits a whole generated object containing two
  foreign Tensors in the selected storage domain. Both validate one logical sequence, both descriptors,
  both backing identities, and both payloads. Both are descriptor-only comparisons with Image: backing
  bytes are initialized before measurement and remain immutable while per-publication schema metadata and
  Tensor descriptors still flow through the complete API. The loan row borrows from the compiler-created
  BufferPool for each field. The benchmark derives the field-pool inventory from the accepted
  plan, circulates every compiled field-pool backing before measurement, and rejects a `--warmup` count
  smaller than that plan-derived minimum. Warmup retains received Tensors until both complete field-pool
  inventories have circulated, including slots that delayed transport retirement can make necessary.
  First-use backing initialization during measurement remains a hard failure.
  Staging uses a bounded preinitialized backing ring derived from queue depth so a
  later publication cannot reuse a backing that direct retention or the HoloMQ sidecar still holds. Before
  reuse, the source requires sole shared ownership and a sole live Tensor-handle claim for both slot
  Tensors; an occupied wrap reports ordinary source backpressure without modifying the backing. Each Tensor
  context also owns its ring storage through `TensorMemoryReference`, so retained final-payload validation
  remains valid even if Runtime completion tears down the source operator first. The borrowed
  `selected_batch()` sink and the loan source remain allocation-free at the excluded probe sample.
  The staging source admits at most one allocation for implementations that probe direct export before
  falling back to the bounded staging pool; implementations that bypass the known-inapplicable export
  probe report zero.
  `multi_tensor`, `multi_tensor_loan`, and `multi_tensor_staging` accept pageable-host, pinned-host, and
  CUDA-device storage. Pinned-host and device rows require CUDA device zero and return the standard skip
  code 77 when it is unavailable.

For example, exercise pinned-host loan and CUDA-device staging storage without changing descriptor-only
payload work:

```sh
python3 ../benchmark/runtime/run_runtime_graph_benchmark.py \
  ./benchmark/runtime/runtime_graph_benchmark \
  --workload latency --scenario multi_tensor_loan --operator-lanes shared \
  --tensor-storage pinned_host --tensor-bytes 4096 --bulk-work-profile descriptor_only \
  --samples 1000 --warmup 100 --queue-depth 1 --publisher-period-us 1000 \
  --output-dir runtime-multi-tensor-loan-pinned-host

python3 ../benchmark/runtime/run_runtime_graph_benchmark.py \
  ./benchmark/runtime/runtime_graph_benchmark \
  --workload latency --scenario multi_tensor_staging --operator-lanes shared \
  --tensor-storage device --tensor-bytes 4096 --bulk-work-profile descriptor_only \
  --samples 1000 --warmup 100 --queue-depth 1 --publisher-period-us 1000 \
  --output-dir runtime-multi-tensor-staging-device
```

The `pod`, `signal`, `sample_forward`, `schema`, `third_party_schema`, `tensor`, `image`, `multi_tensor`,
`multi_tensor_loan`, and `multi_tensor_staging` scenarios use compiler-selected direct-retained delivery
for eligible process-local edges with total fanout one. Blocking and `drop_oldest` are eligible at every
accepted positive depth.
They therefore construct no HoloMQ endpoint, WaitSet attachment, Tensor sidecar record, or Channel pump.
The benchmark's `--queue-policy <blocking|drop_oldest>` comparison exposes lossless and lossy behavior
for `pod`, `schema`, `third_party_schema`, `tensor`, `image`, and both descriptor-schema multi-Tensor
scenarios. The POD smoke matrix covers depth one separately from depth two because the mailbox-sized
unread ingress has distinct capacity geometry. Raw `multi_tensor` rejects lossy delivery because its
preallocated backing comparison requires contiguous publication sequences.
Ordinary completion-driven lossy rows require fanout one. The controlled fixed-rate POD fanout
exception is described below. Pass
`--force-holomq-process-local-inline` to preserve the prior transport-backed path as an exact
same-binary control; the result name gains `_holomq_fallback`, the JSON records
`process_local_inline_delivery` and `queue_policy`. A row with an active Channel pump accepts
`--channel-pump-cpu`, which authors public support-thread placement but never creates a pump. A
shared-lane eventless HoloMQ row is pump-free unless the exact pump control described below is
selected. The `--force-*` controls in this section are private measurement controls, not application
authoring policy.

For a shared-lane 1:1 POD, SDK-schema, third-party-schema, Tensor, Image, or multi-Tensor HoloMQ row, or a shared-lane
POD/SDK-schema/third-party-schema/raw-Tensor/Image/multi-Tensor fanout row at any positive depth
supported by the selected workload,
`--force-holomq-event-wake` selects a private compiler comparison seam that conservatively retains the
Channel event-wake service on every otherwise-eventless active edge. It does not add a graph vertex,
port, subscriber, or callback. The publication and subscription geometry is unchanged. Compare ordinary
dispatch with compiler-selected event-wake suppression against ordinary dispatch plus event wake. Tensor-backed and
HoloMQ paths deliberately remain ineligible for continuation receipts;
`--disable-direct-continuation-elision` therefore does not define another HoloMQ dispatch arm. The result
name gains `_event_wake_control`, and
`runtime_controls.holomq_event_wake_forced` records the provenance in native results, summaries, and
manifests.

An all-eventless process-local HoloMQ row starts no Channel pump: the serialized successful-compute
activation cut remains its transport-drain authority. `--force-process-local-channel-pump` retains the
former pump for an otherwise identical shared-lane eventless row without changing HoloMQ transport,
event wake, graph topology, payload work, queue policy, or queue depth. The result name gains
`_channel_pump_control`, `runtime_controls.process_local_channel_pump_forced` records the provenance,
and `--channel-pump-cpu` becomes available for the control role. The pump and forced-event-wake
controls are mutually exclusive so the pump-only comparison remains exact.

`compile()` delivers a split-lane (`separate` or `per_sink`) POD, signal, sample-forward, SDK-schema,
or third-party-schema HoloMQ row inside the producer's `emit()`. Such a row is eventless and starts
no Channel pump. Tensor-backed rows keep event wake and the pump.
`--force-process-local-pump-delivery` compiles an otherwise identical split-lane HoloMQ row with pump
delivery instead. Use it as the A/B control for delivery inside `emit()`. The result name gains
`_pump_delivery`, `runtime_controls.process_local_pump_delivery_forced` records the provenance, and
`--channel-pump-cpu` becomes available. The control is mutually exclusive with
`--force-holomq-event-wake` and the private fanout experiments.

The narrow `--multi-output` applicability row authors one shared-lane POD source with two outputs and
one subscriber per output. It requires `--fanout 2 --force-holomq-process-local-inline`, blocking
delivery, no offered-rate mode, and equal per-service queue depths when `--sink-queue-depths` is
provided. Each source callback publishes the same logical sequence once on
each output, so the result reports two logical deliveries per publication while phase evidence records
two physical emits. `geometry.source_outputs` and `geometry.subscribers_per_output` distinguish this
2x1 topology from ordinary 1x2 fanout. Pair the default compiler-selected eventless row with
`--force-holomq-event-wake`; every other graph, service, callback, and payload-work detail remains
identical.

The adjacent `--multi-input` row keeps the same two source outputs, two HoloMQ services, and two
physical emits, but connects them to two paired inputs on one sink Operator. It has one sink callback
per logical publication and two input reads in that callback. Both emits carry the same source
activation capture coordinate so the sink can select the first input as the exact peer of the
second-input trigger. The row has the same restrictions as `--multi-output` and reports
`geometry.sink_operators=1` plus `geometry.inputs_per_sink=2`; the multi-output row reports two sink
Operators with one input each. Pair it with
`--force-holomq-event-wake` to isolate the target-composition wake decision without changing the
graph.

The `pod`, `schema`, and `third_party_schema` scenarios accept `--fanout` from 1 through 32. Raw
`tensor`, `image`, `multi_tensor_loan`, and `multi_tensor_staging` accept `--fanout` from 1 through 8.
Fanout greater than one creates one source output connected to that many independently validated sinks.
It uses the process-local HoloMQ path, except that `compile()` delivers a `drop_oldest` POD,
SDK-schema, or third-party-schema fanout by direct retention into every sink, with no HoloMQ service.
`--force-holomq-fanout` keeps such a row on HoloMQ as a same-binary control; the result name gains
`_holomq_fanout`. The forced-event-wake, pump-delivery, and private fanout-experiment controls also
keep fanout on HoloMQ.
By default every subscriber uses `--queue-depth`. For a controlled mixed-capacity fanout, pass one
positive depth per sink in index order with `--sink-queue-depths 4,2`; the list length must equal
`--fanout`, and its maximum must equal `--queue-depth`, which remains the service-wide capacity and
publication-backpressure ceiling. Native JSON and the runner validate the ordered depths as
`geometry.subscriber_queue_depths`.
A same-policy POD, bounded SDK-schema/third-party-schema, raw-Tensor, Image, or multi-Tensor fanout
whose source and every sink share one execution partition suppresses event wakes service-wide; any
partial or split-partition match retains them. Tensor-backed fanout is currently restricted to
shared-lane measurement so the benchmark covers only its certified eventless gates. In separate-lane
mode for POD/schema,
the source and
every sink own distinct execution partitions, so a 1:N sweep measures both delivery fanout and the
corresponding N sink workers. A
logical sequence completes only after its slowest sink receives it; notification pressure is bounded by
those logical completions rather than aggregate deliveries. Results retain per-sink delivery/final-sequence
evidence, source-to-sink percentiles for every sink index, and logical publications/s separately from
aggregate deliveries/s. Use a forced-HoloMQ fanout-one control when attributing the fanout slope so the 1:1
direct-retained fast path is not conflated with fanout cost.

A shared-lane POD `drop_oldest` fanout can instead use
`--offered-rate-hz 40000 --offered-burst-size 4`. This benchmark-only mode replaces the
source-completion-driven notification loop with the production `OnClock` trigger. Each clock activation
emits the bounded burst back-to-back, while the trigger period preserves the configured average logical
publication rate. The burst must exceed `--queue-depth` and cannot exceed 64, so every leg experiences
real replacement rather than merely admitting the lossy policy. The measured sample count must contain
only complete bursts. Every clock tick that fires must publish one complete, accepted burst, and the
runner rejects a row unless every subscriber reports nonzero replacement and the final sequence. The
clock coalesces a tick that the source wakes too late for, so the accepted publication rate
(`accepted_publication_offer_rate_per_second`) falls below the clock rate by the platform's late wakes,
not by delivery. The result reports those coalesced ticks as `lost_offer_ticks`, and a row fails only
when more than a quarter of its ticks are lost, meaning the source cannot keep up with its clock. It
records per-leg delivery, replacement, and rate evidence; aggregate `consumer_visible_replacements` is
`publications * fanout - aggregate deliveries`. This mode is initially restricted to POD, saturated,
shared-lane fanout and does not create a notification-driver thread. `compile()` delivers this row by
direct fanout; `--force-holomq-fanout` selects the eventless HoloMQ arm and
`--force-holomq-event-wake` the retained-wake HoloMQ arm. The three fixed-rate
fanout CTest variants use `RUN_SERIAL` so other tests in the same CTest invocation do not add lost
ticks. Load from unrelated processes, or slow wakes from deep CPU idle states, can still lose ticks;
only a source that loses more than a quarter of them fails.

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

The default `--partition-cover legacy` preserves the benchmark's historical exact cover: separate-lane
runs place the source, relay chain, and sink in distinct named partitions, while shared-lane runs place every
Operator in `runtime-graph-shared`. Characterization rows may instead request `automatic`, `include_all`,
or `hybrid`. `automatic` leaves the complete cover to the compiler, `include_all` authors one named root
partition, and `hybrid` authors only the source partition before calling `auto_partition_remaining()`.
The runner clears the retired `HOLOSCAN_EXPERIMENTAL_PARTITION_ALGORITHM` environment variable and derives
the expected accepted algorithm from the cover authority. Native JSON and `summary.json` retain the
authoring choice, accepted mode and algorithm, authored/automatic counts, deterministic cover signature,
and compiled partition/lane owner inventory. Automatic generated names cannot be targeted by authored
placement, so automatic rows reject Operator CPU, busy-poll, and Green Context requests instead of silently
changing the comparison.

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
are reported separately. `consumer_visible_replacements` is accepted delivery opportunities minus actual
deliveries, and `replacement_rate` exposes that fraction. For ordinary fanout-one lossy rows the
opportunity count equals publications. The fixed-rate fanout mode uses `publications * fanout` and the
clock path above rather than a notification capacity.

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

Use the same drop-oldest comparison for SDK and third-party generated schema, raw Tensor, and
descriptor-backed Image:

```sh
for scenario in schema third_party_schema tensor image; do
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

CUDA Green Context phase-1 qualification uses the same production Tensor graph
with explicit matched bound/unbound layouts, exact plan artifacts, cold and
terminal-quiesce timing, and separate non-performance CUDA resource counters.
See [GREEN_CONTEXT_QUALIFICATION.md](GREEN_CONTEXT_QUALIFICATION.md) for the
matrix runner, Nsight attribution procedure, sanitizer lane, and `kRealTime`
disposition.

CUDA compute-stream policy qualification uses a two-sink device-Tensor fork
with the source and each sink in separate execution partitions. The native
binary accepts `--stream-policy shared|per_partition` and
`--cuda-work-profile none|low_occupancy|saturating`; non-empty profiles also
require `--cuda-branch-work-us B,C`. It reports exact plan stream/binding
counts, Runtime readiness/release counters, per-branch host and CUDA duration,
source-to-GPU-completion latency, and overlap derived from preallocated CUDA
events. The synthetic kernels only enqueue bounded work on each Operator's
compiled stream; the callback does not synchronize that work.

Use `run_cuda_stream_policy_qualification.py` for matched shared and
per-partition rows across no-device-work, low-occupancy, saturating, and
asymmetric regimes. The runner claims a fresh output directory, publishes an
atomic attempt status, hashes the binary and every result/plan artifact, and
reports repeated-run coefficient of variation. Its policy deltas are
characterization evidence only; the runner intentionally defines no universal
performance pass/fail threshold.

```sh
python3 ../benchmark/runtime/run_cuda_stream_policy_qualification.py \
  ./benchmark/runtime/runtime_graph_benchmark \
  --revision "$(git rev-parse HEAD)" \
  --output-dir runtime-cuda-stream-policy-qualification
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
`holoscan.test.runtime-benchmark/v1` JSON schema. Each row records the precise
question it answers, clock source, workload geometry, compile-accepted lane
placement, completion-window throughput, callback/publication/delivery counts,
validation, and session diagnostics. The v1 `placement.operator_lanes` field is
retained as a compatibility alias for the requested benchmark policy;
`placement.requested_operator_lanes` is the precise field required by current
consumers. Compiler-derived `operator_lane_count` and partition evidence remain
the authority for the accepted automatic layout. Latency rows additionally report p50,
p90, p99, min, and max from preallocated per-sample tables; p99.9 is reported
only with at least 10,000 measured samples. For edge-bearing scenarios, `notification_latency` uses
post-to-sink as its primary `latency_ns` and retains the component `notification_to_source_ns`,
`source_callback_entry_to_pre_emit_ns`, `source_publication_call_ns`, and `source_to_sink_ns`
distributions. Callback-to-pre-emit covers source-side preparation before the public `emit()` call;
publication-call covers that call through its successful return. These components reuse timestamps already
needed by the other boundaries and add no timed-path clock reads. Tensor rows additionally report
`tensor_loan_acquire_ns`, `tensor_write_commit_ns`, and `tensor_receive_call_ns`. Those diagnostic
boundaries add two source-side clock reads per measured publication and one sink-side clock read per
measured delivery; compare their sub-microsecond values with the reported timestamp-read-pair calibration
rather than treating them as an unperturbed application path. Loan-acquire and write-commit therefore each
contain `samples` observations, while receive-call aggregates `samples * fanout` independently paired
sink-call observations. A concurrently running sink can enter or complete before the producer records its
post-`emit()` or pre-return boundary. The corresponding
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
Git/build/host/GPU metadata, SHA-256 of the exact benchmark executable, requested Operator and Runtime
support-thread placement, observed support-thread affinity and native thread IDs, resolved CPU topology,
and scaling governor. `--source-state` and `--source-snapshot-sha256` record whether the measured source
tree was dirty and the deterministic snapshot fingerprint used to build the executable. If
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
     --source-state dirty-task040-worktree-snapshot \
     --source-snapshot-sha256 <sha256> \
     --output-dir /tmp/runtime-graph-pod-latency"
```

When the runner is launched inside the SDK container, the outer shell expands the revision before
`public/run` starts the container. `HOLOSCAN_GIT_REVISION` provides the equivalent automation hook.
For dirty-tree evidence, compute the source snapshot from the `git diff --no-ext-diff --binary
--full-index HEAD --` bytes followed by the lexicographically sorted untracked path and file-SHA-256
records from `git ls-files --others --exclude-standard`. Record the resulting SHA-256 and a precise
state label; `unknown` is retained only as a backward-compatible standalone default.

For automatic partition-cover rows, `requested_operator_lanes` is only a benchmark policy input. The
compiler owns the effective partition layout, which the native result records through the accepted
`operator_lane_count` and `operator_partition_uses` inventory. Automatic rows therefore require
`--operator-lanes separate`; shared and per-sink requested layouts would incorrectly imply control over
compiler-generated partition ownership. Source-only automatic rows retain one compiler-derived owner and
no sink inventory. To keep requested controls from changing an effective automatic layout, automatic
rows reject forced HoloMQ, and automatic rows without explicit fan-out reject Runtime phase counters.
Fan-out automatic phase-trace rows remain supported because their compiler-derived source and sink owners
are distinct.

Use the comparison runner to alternate baseline and candidate processes and retain every child artifact:

```sh
./public/run launch --run-cmd \
  "python3 ../benchmark/runtime/run_partition_cover_comparison.py \
     ./benchmark/runtime/runtime_graph_benchmark \
     --baseline-cover include_all --candidate-cover automatic \
     --samples 10000 --warmup 1000 --repetitions 10 \
     --git-revision $(git rev-parse HEAD) \
     --source-state dirty-task040-worktree-snapshot \
     --source-snapshot-sha256 <sha256> \
     --output-dir /tmp/runtime-graph-partition-cover"
```

`comparison.json` reports medians, p95/p99 across repetitions, exact interleave order, cover signatures,
partition-owner geometry, executable/source/build/revision checks, affinity/governor evidence, and the
relative regression. Its top-level `provenance` object retains each arm's binary SHA-256, source
state/snapshot SHA-256, revision, build, and host identity; it emits `shared` only when every child
manifest matches, rather than attributing baseline provenance to a drifted candidate.
Each child runner has a bounded wall time (`--arm-timeout-sec`, 1,800 seconds by default); a timeout
preserves the exact command and partial standard streams in that arm's artifact directory.
The 3% median parity threshold is enforced only when both arms have identical covers and owner geometry,
at least five repetitions, identical host/kernel/machine and per-role CPU topology, confirmed actual
support-thread affinity, one exact binary, stable explicit source/build/revision evidence, performance
governors, and each arm remains below
the configured median coefficient-of-variation limit. Execution-lane comparisons remain unqualified until
actual worker affinity is captured; requested CPU metadata alone is insufficient. Other runs remain useful
characterization evidence but are explicitly `unqualified`; they cannot pass or fail the parity gate.
The comparison command exits successfully only for a qualified passing parity result. Both `fail`
and `unqualified` return a nonzero status so automation cannot publish characterization-only data as
accepted parity evidence.

For a fan-out concurrency characterization, use `notification_latency`, `--runtime-phase-counters`, and
automatic versus `include_all`. The diagnostic native evidence records one stable TID per sink partition,
the source-owner TID, per-sink causal delivery counts, and source-return/sink-callback interval overlaps.
The comparison runner fails closed unless automatic sink owners are stable and distinct, the source has a
different owner, the `include_all` callbacks share the controller owner, every causal validation passes,
and at least one measured callback interval overlaps. This intentionally different-cover row is always
unqualified for the 3% parity gate.

For notification-driven rows, also pass `--notification-driver-cpu` on a distinct physical core. The
Operator CPU options use the public compile-time lane-placement contract, whose startup barrier applies
and verifies the accepted placement. `--controller-cpu` and pump-backed `--channel-pump-cpu` author the
public support-thread affinity contract; their accepted values are serialized in the execution plan and
applied before those Runtime threads perform work. The benchmark's observation hook independently reads
the affinity back and records the native thread ID without reapplying those public bindings.
`--notification-driver-cpu` remains a source-tree-only measurement control because the notification driver
is not a public support-thread role. None of these controls adds per-activation work.

Use `--runtime-phase-counters` only for diagnostic runs. It installs a private, source-tree-only fixed counter
block and a capped, preallocated phase-trace buffer. The counters record controller readiness scans, waits and
wakes, candidate freeze/select transactions, input-poll dispatches, compute dispatches, completed activations,
and Runtime Channel prepare, emit, receive, readiness, read, witness-retirement, activation-retirement,
activation-cut, and backpressure boundaries. The trace timestamps readiness-poll return; candidate freeze and
selection; the preceding activation cut; input-selection and activation preparation; compute dispatch,
callback entry, callback return, and
dispatch return; activation retirement; and activation-cut completion. Per-receiver distributions expose the
complete candidate-to-dispatch interval together with its selection, input/Channel preparation, and callback-
context suffix, plus callback and retirement subintervals. The finer preparation fields are present on ordinary
dispatch rows, including linear direct-retained and eventless-HoloMQ chains; specialized composite rows retain
the enclosing activation intervals without fabricating ordinary-dispatch phases.
It also reports source-activation runs, their maximum consecutive length, and all four
source/downstream transition counts so queue-depth diagnostics can distinguish batching from payload work.
For Tensor scenarios, the same bounded trace pairs output-port and sample-sequence identities and
reports total emit, validation, pre-transport, transport-publication, and post-transport
distributions. Direct-retained Tensor rows have no transport-publication interval; forced-HoloMQ
rows require all five Tensor checkpoints for every accepted publication.
Inline POD and generated-schema scenarios similarly report total emit, encoding, pre-transport,
transport-publication, and post-transport distributions. Direct-retained rows stop after encoding;
transport-backed rows require all five inline checkpoints for every accepted publication.
The runner rejects a diagnostic result unless every nonterminal frozen candidate is selected, dispatched,
and completed; every wait returns by wake or timeout; and the Channel counts match the benchmark's exact
callback, publication, and delivery evidence. A clean terminal stop may retain exactly one candidate that
froze but never entered dispatch. The bounded trace identifies that state explicitly and requires its count
to equal the frozen-versus-selected counter surplus. It also classifies execution-lane tasks by callback
context and verifies exact task publication, callback entry/exit, controller completion, and admitted
busy-versus-sleeping handoff counts.
`source_compute_entries` can exceed productive `source_callbacks` when scheduling has already admitted
a terminal source witness: by at most one on a clocked row, or by the exact accepted-notification surplus
within the configured capacity during notification-driven saturation. Those entries invoke `compute()` and
return without publishing; native and runner evidence account for the bounded surplus.

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

### Multi-partition fanout controls

For a matched POD comparison, use `--fanout N --operator-lanes per_sink` with
`--sink-cpus` listing one CPU per consumer. Add `--multi-output` to publish the
same logical sample through N independent outputs. Each output can then use
direct retention; `--force-holomq-process-local-inline` retains HoloMQ for a
transport control. Source emit timing includes all N emit calls. These rows
retain identical consumer partitions and validate delivery to every consumer.

`--source-work-us` and `--sink-work-us` independently override
`--callback-work-us` for ordinary POD rows (0 through 1,000,000 microseconds).
`--emit-then-work` moves source work after its emit loop. The source pre-return
marker follows that work; it is not the runtime's observation of callback
return. Receive/callback overlap with source emit or pre-return is counted
separately and excluded from nonnegative interval distributions.

`--burst-size N` emits up to N samples per clock callback on blocking POD rows.
It requires queue depth at least N and supports 1 through 64. Allocation-probe
and measurement boundaries can shorten a burst. An overloaded row must still
pass publication, delivery and callback accounting; a timeout is not a latency
result. The existing offered-rate drop-oldest controls remain separate.

`--lane-fifo-priority N` explicitly requests FIFO for separately authored lanes
(1 through 99; zero keeps OTHER). Startup readback is included in
`placement.lane_startup` without phase tracing. Policy values follow
`SchedulingPolicy` (OTHER=0, FIFO=1); support-thread policy values are native OS
values. An unavailable FIFO request produces exit code 77 and retained startup
evidence. Do not launch the entire process under inherited FIFO: default lane
readback rejects unexpected inherited scheduling.

`--allow-pump-lane-cpu-sharing` permits the pinned pump to share a lane's logical
CPU or physical core for scheduling diagnostics. It preserves the rejection of
other role collisions, including pump/controller sharing. Default runs require
separate cores. The flag does not change pump scheduling policy.

Percentile objects include `p9999` only as a numeric value when that distribution
has at least 100,000 samples; otherwise it is null. At that sample count, only
about ten observations populate the upper 0.01%. Use repeated runs and retain
sample counts, maxima, placement and instrumentation metadata when comparing tails.

`--experimental-fanout cut|emit` is a source-private prototype for blocking POD,
generated-schema (`--scenario schema` or `third_party_schema`), raw Tensor
(`--scenario tensor`), or descriptor-backed schema (`image`, `multi_tensor_loan`,
`multi_tensor_staging`) fanout with `per_sink` partitions. Multi-Tensor sinks share
one per-sample validation buffer, so their separate-lane rows use final-sample
validation. `cut` removes HoloMQ event attachment and the
pump, relying on existing activation cuts. `emit` additionally drains the
publication's immutable recipient list inside `emit()`, admitting it under the
recipients' transport mutexes instead of the channel scan lock (see below), and
retains the post-compute safety cut. Both retain existing owner notifications.
The private compiler seam also admits blocking scalar/POD, generated-schema, and
raw Tensor direct 1:1 services and complete same-partition eventless families alongside split
fanout. The inventory must remain pump-free: eventful services, external
channels, nonblocking policies and restart authority remain rejected. Raw Tensor,
Image, and multi-Tensor fanout may also run with `per_sink` or `separate` lanes
without the experiment, which gives the pump-path baseline for the same rows. A source that emits and
keeps computing can delay consumers in `cut`; compare `--emit-then-work` and
bursts explicitly. Neither mode changes ordinary graph compilation defaults.

`--fanout-attribution` records lightweight per-recipient checkpoints for blocking
POD fanout: admission completed, ready-hint call started, and that call returned.
The result reports emit-to-admission, admission-to-hint, hint-call duration, and
hint-to-sink-callback distributions. Callback overlap is counted separately.
Port/sequence-indexed storage is allocated before execution; no phase trace is
required. A hint checkpoint records the Runtime call, not proof of a kernel wake
or an isolated measurement of IPI/context-switch latency. Use separate runs for
attribution and ordinary latency comparisons. The checkpoints are HoloMQ
admission and ready-hint events, so a blocking family that `compile()` delivers
by direct retention needs `--force-holomq-fanout`; both parsers reject the row
otherwise.

`--measure-source-period` independently reports `timing.source_period_ns` for
clocked blocking POD rows. It measures successive measured source callback
starts, excluding warmup, with one observation per callback even for bursts and
for retries after backpressure, which repeat a publication sequence.
Consequently N measured callbacks provide N−1 periods, and a run that measures
more than eight callbacks per sample runs out of period storage and is invalid
(`timing.source_period_storage_exhausted`); at least 100,001 callbacks
are needed for a numeric period P99.99. It is opt-in because reading the clock at
callback entry adds instrumentation. Both controls appear in result names and
geometry so they cannot silently mix with uninstrumented comparison rows.

`--sink-fifo-priorities 1,2` gives separately authored sinks FIFO lane priorities;
startup readback validates each lane.

Whenever delivery inside `emit()` serves a service, Runtime admits concurrently: each
publication under its recipients' transport mutexes instead of the channel scan lock,
retrying only the subscribers whose capacity a retirement released and skipping heads
known to be blocked. No option selects this; the result's `geometry.concurrent_admission`
reports the mode the compiled plan selected, and the pump-delivery control reports `false`.

For partition-owned execution, `--busy-poll` honors the admitted lane policy
for Channel inputs even while no readiness revision changes. The owner rescans
with a processor pause instruction and preserves stop/deadline checks instead
of entering the sleeping wait. This explicitly trades a dedicated CPU per lane
for lower wake latency. Sleeping lanes retain their existing policy. This does
not remove transport or cache locks, or change whether the graph requires a pump.

### Concurrent small-fanout scaling

`runtime_fanout_scaling_benchmark` gives each producer its own Runtime partition.
The default layout also gives every recipient its own partition. With
`--shared-destinations`, sink index K from each producer connects to a distinct
input of one shared `OnAnyInput` sink operator/partition. This tests shared lane scheduling without
introducing a multi-input join or a shared subscriber. It does add selection
checks for each input, so compare delivery variants within a fixed layout.
Fanout 2–3 is the primary workload; 4 is a boundary control.

Build the target in the SDK development container. From its build directory:

```bash
python3 ../benchmark/runtime/run_runtime_fanout_scaling.py \
  benchmark/runtime/runtime_fanout_scaling_benchmark \
  --output-dir fanout-scaling/two-producers --repetitions 3 \
  --delivery emit --producers 2 --fanout 3 \
  --first-cpu 2 --controller-cpu 0 --samples 10000 --warmup 100
```

Select available CPUs on the workstation. Lane CPUs are contiguous: producers
first, then destination partitions. The binary rejects explicitly colocating
controller/pump CPUs with that range. Startup readback and callback thread IDs
are retained, and the run fails if the declared partition geometry is not realized.
`--delivery baseline` compiles with `compile()`, which delivers these inline payloads inside
`emit()`. Use `--delivery pump --pump-cpu 1` for the pump-delivery control or
`--delivery direct` for N separate outputs per producer. These controls preserve
the chosen partition layout. `--busy-poll` is an explicit dedicated-CPU policy.
Rows that deliver inside `emit()` admit each publication concurrently, under its
recipients' transport locks instead of the channel scan lock; the result's
`concurrent_admission` reports whether the compiled plan selected that mode.

The default source period is 1 ns, an always-due clock used to measure saturation;
it is not a notification-driver throughput measurement. `--period-us 1000`
provides a paced control. Producers rendezvous once after warmup, then publish
independently with no benchmark-global atomic update on each publication. The
reported aggregate rate covers the earliest measured publication through the
last measured recipient, including slow producers. Per-producer timing and
start/end coordinates allow overlap and fairness to be checked separately.

Every callback checks POD checksum, producer identity and contiguous sequence.
Every source completes after its finite publication count; success requires all
recipient tails and natural EOS completion without an explicit stop. A rejected
backpressured emit retries only that output, never an already accepted prefix of
separate outputs. Latency starts before the first attempt, so it includes retries.
This payload timestamp and validation envelope differs from the existing graph
benchmark; compare variants within this executable, not absolute numbers across
executables.

`--queue-depth 1 --sink-work-us 20` creates sustained recipient pressure.
`--stall-us 1000` suspends producer 0's first sink once halfway through measurement
while it retains its input witness. Results count other producers' deliveries
during that interval. This holds a callback/witness, **not a typed-cache mutex**;
the dedicated runtime cache-owner test covers the latter dependency.

`--counters` collects aggregate Runtime counters and whole-thread CPU usage without
a timestamped phase trace. Run it separately from latency/throughput comparisons:
shared counter writes can distort concurrent-producer scaling. Results include cut
skips, inventory scans, transport probes and receive backpressure. The runner
preserves commands, source revision, logs and failed results in a new output
directory, and rejects incomplete counts or unsupported P99.99 claims (at least
100,000 measured samples per producer are required).

If the SDK container mount excludes the checkout's Git metadata, capture source
state on the host and pass `--source-state <mounted-path.json>` to the runner.
From the SDK source root on the host, for example:

```bash
python3 - <<'PYTHON' > build-cu13-x86_64/fanout-source.json
import json
from pathlib import Path
import sys
sys.path.insert(0, "benchmark/runtime")
from run_runtime_fanout_scaling import git_state
print(json.dumps(git_state(Path.cwd()), indent=2))
PYTHON
```

Use `--source-state fanout-source.json` from the container's build directory.
Regenerate it after source changes; the runner preserves this supplied revision
and dirty-file status, but cannot independently verify an unmounted repository.

### Robot control loop

`runtime_control_loop_benchmark` reproduces the shape of a robot control benchmark:

- **Control partition:** a 1 kHz chain, `ReadJointState` (`OnClock`) → `MotionPlanner` (`OnEach`
  on joint state, with a `DrainAll` target side input) → `SafetyCheck` → `WriteMotorCommand`.
- **Target partition:** a 137 Hz `TargetProducer` (`OnClock`) that reads joint state as a
  `DrainAll` side input and publishes a target pose.
- **Edges:** every edge is drop-oldest with depth 1.
- **Payloads:** padding-free PODs of 80, 80, 176, and 192 bytes.
- **Work:** each stage runs a calibrated dependent floating-point loop, so stages of 0.2–2.4 µs
  cost no clock reads. `--work-profile zero` exposes framework cost alone. `--work-iterations`
  replays recorded iteration counts across builds.

Three modes share the work, payloads, timestamps, and statistics:

| Mode | What runs |
| --- | --- |
| `split` | The graph above, or a `--topology` variant |
| `fused` | One control operator doing read, plan, safety, and write in one callback, with an open-loop target |
| `no-framework` | Two plain threads with absolute `clock_nanosleep` deadlines and wait-free latest-value handoffs |

Topologies (`split` only, except `rcb` and `single`):

| Topology | Shape |
| --- | --- |
| `rcb` | One joint-state output with two subscribers, the second in the target partition |
| `split-outputs` | One joint-state output per consumer, so every edge has one subscriber |
| `single` | The target runs open-loop; joint state has one consumer |
| `same-partition-port` | As `single`, plus a second `MotionPlanner` input fed from the same joint-state output (fanout inside one partition) |
| `rcb-no-target-input` | As `rcb`, without `MotionPlanner`'s target input |

Compare `fused` with `split --topology single`, not with `rcb`: fusing the chain also removes the
target's joint-state input.

All timestamps are `CLOCK_MONOTONIC`. The scheduled activation of a clocked operator is converted
from the activation clock with an offset measured in the first callback (reported with its
uncertainty). The KPIs, in nanoseconds:

| KPI | Definition |
| --- | --- |
| `control_loop_latency` | Command completion in `WriteMotorCommand` minus the scheduled 1 kHz activation |
| `active_path` | Command completion minus `ReadJointState` callback entry |
| `entry_latency` | `ReadJointState` callback entry minus the scheduled activation |
| `control_period` | Successive `ReadJointState` callback entries |
| `target_pose_response_latency` | Completion of the first control tick that used a target version, minus that version's scheduled activation |
| `target_delivery` | First use of a target version by `MotionPlanner`, minus the start of the target's emit |
| `hop_read_to_motion` | `MotionPlanner` callback entry minus the joint state's emit return (split mode) |
| `hop_motion_to_safety` | `SafetyCheck` callback entry minus `MotionPlanner` callback entry |
| `hop_safety_to_write` | `WriteMotorCommand` callback entry minus `SafetyCheck` callback entry |

`worst_active_tick` gives the scheduled-relative stage times of the tick with the longest active
path, which locates a stall. `--observations PATH.csv` writes every tick's stage timestamps.

`side_input_staleness_ticks` counts control ticks between the newest joint state published before a
target activation and the one that activation consumed. The Tensor/descriptor CTest matrix
checks both `count.side_input_staleness_max<=4` and
`count.side_input_staleness_samples>0`; without the latter, an empty observation set would pass
because its maximum defaults to zero. Four ticks is an empirical smoke tolerance for unpinned
CI, not an SDK freshness guarantee; it still rejects the original 7–15-tick regression.
Controlled performance qualification can retain `--expect count.side_input_staleness_max<=2`.
JSON also reports `worst_side_input_staleness`, or null
when there are no observations. Its target version, selected/newest joint sequences and
publication timestamps identify the worst observation without adding instrumentation to the
callbacks. All timestamps in that record use the benchmark monotonic clock domain, including
the converted target scheduled time. They do not record the selector's freeze time or prove
that a newer sample was eligible at that point.

`missed_freshness` counts control ticks
that used an older target than one whose emit returned before the tick's scheduled activation; a
target still being emitted at that time does not count, because the tick could select only the
previous version. `late` counts ticks
completing after the next scheduled activation. `coalesced` counts skipped grid points.

Placement: `--control-cpu`, `--target-cpu`, `--controller-cpu`, `--pump-cpu`,
`--control-fifo-priority`, `--target-fifo-priority` (FIFO needs `CAP_SYS_NICE`; the SDK container
allows priorities up to 10), and `--busy-poll-control`. `--timer-slack-ns 1` sets the kernel timer slack of the
control and target threads, which otherwise inherit 50 µs for non-real-time threads and delay
every timed wake by up to that much; the result reports the slack in effect. The graph writes nothing during
measurement; `--mlock` locks memory. `--counters` adds session-wide phase counters, thread
usage, and per-partition owner accounting: readiness and no-progress scans; parks, and whether
each ended on a wake token, a clock deadline, or the 1 ms fallback cap; and owner-progress
signals received, the subsets sent by the partition's own thread and by threads that own no
partition, and those that published a wake permit. Partitions are labeled `control` or `target` by
their owner thread. Run `--counters` separately from latency comparisons. `--force-holomq` and
`--force-holomq-event-wake` force HoloMQ delivery for every process-local edge.
`compile()` delivers `rcb`'s drop-oldest joint-state fanout by direct retention into both
subscribers; `--force-holomq-fanout` keeps that fanout on HoloMQ and is the A/B control for it.
`--force-pump-delivery` compiles with pump delivery instead of `compile()`'s delivery inside the
producer's `emit()`; it is the A/B control for that delivery and combines with `--force-holomq`.
`--disable-side-input-continuation` keeps the `ReadJointState` → `MotionPlanner` hop on the
ordinary scan instead of continuing the causal chain into it.
`--disable-boundary-causal-chain` keeps the whole control partition on the ordinary scan; it is the
A/B control for the causal chain across the partition's boundary edges.

#### Parity matrix

These `split`-mode options vary one property of the `rcb` shape at a time, so a run shows whether
the direct, inside-`emit()`, and causal-chain fast paths apply beyond drop-oldest POD `DrainAll`:

| Option | Effect |
| --- | --- |
| `--joint-payload pod\|schema\|tensor-host\|tensor-pinned\|tensor-device\|descriptor\|descriptor-borrow` | The joint-state wire type. `schema` is an owning generated schema. `tensor-*` is a rank-1 byte Tensor from the runtime pool; the device arm exits 77 without CUDA. `descriptor` builds a fresh host Tensor and an owning `ImageT` each tick, which the runtime copies into its pool. `descriptor-borrow` publishes and receives the way the vision operators do: a Tensor loan for `ImageT.data`, a reused `ImageDescriptorScratch`, and borrowed selected-sample views |
| `--control-policy drop-oldest\|blocking` | The policy of the control chain's edges. Target edges stay drop-oldest, so `blocking` makes the joint-state fanout a mixed-policy family |
| `--export-joint-state` | Also exports the joint state as an external topic with an optional peer (POD and schema only) |
| `--side-selector drain-all\|latest-within\|reuse-last` | The selector of `MotionPlanner`'s target input and the target's joint-state input. `--side-selector-scope both\|motion\|target` limits it, and `--latest-within-us N` sets the maximum age. Producers stamp capture times only in these arms |

The result's `plan.joint_state` reports the joint-state output's process-local delivery, one
`policy:kind` per service, where the kind is `direct`, `emit` (delivered inside `emit()`), `pump`
(HoloMQ event wake), or `cut` (the post-compute activation cut). `--plan-only` compiles, prints it,
and exits.

`--expect PATH<op>VALUE` (repeatable; `==`, `!=`, `>=`, `<=`, `>`, `<`) checks the run and makes it
exit 1 when a check fails. `PATH` is one of:

- `phase.X` or `per_tick.X`: a phase counter, in total or per tick; needs `--counters`;
- `partition.control.X` or `partition.target.X`: a partition counter; needs `--counters`;
- `count.X`: a run count;
- `plan.joint_state`.

Unknown names are rejected at startup. The `BENCHMARK_CONTROL_LOOP_*` CTest rows use these checks
to assert each arm's delivery and counters without timing. For example,
`--expect plan.joint_state==drop_oldest:direct --expect per_tick.causal_frontier_accepted>=2.4`.

A measured run ends when `WriteMotorCommand` has recorded `--warmup` plus `--samples` ticks; it
then requests stop, so a valid graph run ends stopped with no diagnostics. Target records are sized
for the nominal run length: if control overruns keep the target lane publishing past it,
`counts.target_observation_overflows` counts the publications without a record and the run is
invalid. `--timeout-ms` bounds every mode, including `no-framework`. A run whose result cannot be
written exits with status 2. For example, from the build directory:

```bash
benchmark/runtime/runtime_control_loop_benchmark --mode split --topology rcb \
  --samples 100000 --warmup 1000 --control-cpu 4 --target-cpu 6 --controller-cpu 2 \
  --pump-cpu 8 --control-fifo-priority 10 --output rcb.json
```

`libcontrol_loop_yield_shim.so` (built beside the SDK libraries) counts `sched_yield()` calls per
thread when preloaded; the result reports the calls inside the measured window for the control,
target, and other threads. With `CONTROL_LOOP_YIELD_SHIM=noop` it returns without yielding, which
isolates the cost of yielding in timed waits. Compare `noop` with `pass`, never with no shim:

```bash
LD_PRELOAD=$PWD/lib/libcontrol_loop_yield_shim.so CONTROL_LOOP_YIELD_SHIM=noop \
  benchmark/runtime/runtime_control_loop_benchmark --mode split --require-yield-shim ...
```

`run_runtime_control_loop.py` runs named arms, each a binary with arguments and environment, for
`--rounds` rounds in a seeded random order per round, one process per run. It keeps each result,
log, and a manifest, and writes `summary.json` and `summary.md` with the median and range of
per-run percentiles per arm and deltas against the reference arm:

```bash
python3 ../benchmark/runtime/run_runtime_control_loop.py --arms arms.json \
  --output-dir control-loop/baseline --rounds 6
```

where `arms.json` is, for example,
`{"reference": "rcb", "arms": [{"name": "rcb", "binary": "benchmark/runtime/runtime_control_loop_benchmark", "args": ["--topology", "rcb", "--samples", "100000"]}]}`.

## Tensor stream registration cross-casts

`tensor_stream_registration_benchmark` isolates RTTI cross-casts from the base
Tensor memory reference to the private lifetime and callback-completion
extensions. It uses synthetic objects matching the native two-base and Python
three-base inheritance shapes. It excludes CUDA calls, pinning, locking,
registration, and shared-owner copying. A cached-pointer row is a synthetic
lower bound, not an implemented capability-cache optimization.

From the SDK source root:

```bash
./run launch --run-cmd \
  "cmake --build . --target tensor_stream_registration_benchmark -j 10 && \
   ./benchmark/runtime/tensor_stream_registration_benchmark \
   --benchmark_min_time=0.2s --benchmark_out=stream-registration.json \
   --benchmark_out_format=json"
```

This source-tree benchmark requires private Core types. Its CTest smoke case is
`BENCHMARK_TENSOR_STREAM_REGISTRATION`.
