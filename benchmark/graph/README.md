# Graph Authoring Benchmark

## Cold partition planning

`partition_planning_benchmark` first performs and validates an unmeasured zero-work compile to warm
process/compiler initialization, then measures one complete topology-specific `compile()` in the same
isolated process.
Its benchmark-local global allocation replacement snapshots C++ allocation count, cumulative
requested bytes, peak live requested bytes, and live requested bytes when `compile()` returns. The
probe remains active only until the returned plan is released so the result can also report retained
live bytes after plan destruction. It does not instrument Core or Runtime production paths.
Whole-process `ru_maxrss` is supporting evidence, not planner-only storage. The native probe
normalizes the platform-specific `getrusage` unit and always reports `process_max_rss_kib` in KiB.

The runner accepts configurable topology and size lists. Task-040 uses 16, 256, and 4,096 logical
Operators as practical characterization points; these are not the `uint32_t` representation limit:

```sh
python3 public/benchmark/graph/run_partition_planning_benchmark.py \
  build/benchmark/graph/partition_planning_benchmark \
  --topology linear --topology fanout --topology binary_fanout \
  --topology fanin --topology diamond \
  --topology multi_root --topology embedded --topology hybrid \
  --operators 16 --operators 256 --operators 4096 --repetitions 5 \
  --git-revision <exact-task-head> \
  --source-state <clean-or-dirty-snapshot-description> \
  --source-snapshot-sha256 <sha256> \
  --output-dir /tmp/partition-planning
```

The `linear` topology is also the structural-chain scaling probe for the source-chain planner and
validator. When changing candidate join or partition-cover validation, compare at least 1,024 and
2,048 Operators in one run so the retained summary makes superlinear growth visible without a
workstation-specific pass/fail threshold:

```sh
python3 public/benchmark/graph/run_partition_planning_benchmark.py \
  build/benchmark/graph/partition_planning_benchmark \
  --topology linear --operators 1024 --operators 2048 --repetitions 5 \
  --output-dir /tmp/partition-chain-scaling
```

Pass `--operators 0` in the same run to capture the same-binary instrumentation/process baseline.
Hybrid probes require either zero Operators (the baseline) or at least four Operators so the
benchmark can retain a nonempty exact anchor before requesting an automatic remainder; sizes 1–3
are rejected by both the native probe and the Python runner.
`fanout` is the single-root star stress case; `binary_fanout` repeats one-source/two-sink units so
branch behavior is not inferred from only one high-degree root. The summary records minimum, median,
p95, p99, maximum, deterministic plan digest and cover,
process affinity, platform identity, the exact benchmark-binary SHA-256, selected CMake/compiler
fingerprint, supplied exact revision, explicit source-snapshot state, and its deterministic snapshot
SHA-256. Every child process retains its command and standard streams even when its configured
process timeout expires, plus native JSON after a completed probe. If
`--output-dir` is omitted, the runner creates and prints a persistent temporary artifact directory.

This source-tree-only Google Benchmark suite measures the cold baseline Graph
authoring path. It records scaling for:

- Operator declarations without edges, which isolates append and side-index
  maintenance;
- linear chains and one-to-many fan-out DAGs, including cycle and fan-in
  validation; and
- immutable snapshot construction and canonical hashing;
- one embedded reusable Graph with increasing child size;
- increasing sibling count with ten Operators per child; and
- increasing nesting depth with a fixed total of 128 Operators, which isolates
  repeated hierarchy flattening and canonicalization from payload growth.

The direct-authoring and child-size series use 1, 10, 100, and 1,000 Operators.
The sibling series uses 1, 10, and 100 children, while the fixed-size nesting
series uses 1, 2, 4, and 8 hierarchy levels. The largest cases are regression
probes for accidental linear-to-quadratic behavior, not claims that every
deployment uses graphs of those sizes. CTest executes a short smoke run and
does not impose a workstation-specific latency threshold. Google Benchmark
reports complexity from each size series; historical results should be
compared when changing authoring indexes, validators, embedding, or snapshot
canonicalization.

Run the benchmark from the repository root:

```sh
./public/run bench --name graph_authoring
```

For a repeated characterization and JSON export:

```sh
./public/run bench --stable-system --name graph_authoring \
  --output-json graph-authoring-benchmark.json \
  --benchmark-args \
  "--benchmark_min_time=0.01s --benchmark_repetitions=5 \
   --benchmark_report_aggregates_only=true"
```

These are cold-path measurements. They do not model compilation, session
construction, scheduling, Channel attachment, or per-activation runtime work.
Those paths require separate benchmarks as their owning baseline components land.

## Cold rate planning and headroom characterization

`rate_planning_benchmark` is a separate EA2 source-tree characterization harness for Task-041.
It does not reuse the Task-040 partition-planning schema. Every child process still performs one
accepted zero-work warm-up compile before measuring one topology-specific compile, and the same
benchmark-local allocation probe records compile-local allocation count plus requested-byte
headroom. The native JSON therefore reports `live_requested_bytes_at_probe_end` together with
`plan_retained_at_probe_end`; it does not claim the plan has already been released when that probe
snapshot is taken. The compile-scaling mode exercises representative rate topologies instead of
partition authoring patterns:

- `linear_fixed`: one 1 kHz fixed-rate source through a relay chain;
- `fanout_fixed`: one 1 kHz fixed-rate source fanned out to many sinks;
- `dual_rate_fusion`: repeated 60 Hz camera plus 1 kHz IMU pairs into `OnEach(camera)` +
  `DrainAll(imu)` fusion sinks; and
- `multi_follows`: repeated relays that derive multiple output rates from one followed input;
- `embedded_follows`: fixed-rate sources passing through embedded child Graph relays; and
- `lifecycle_bridge`: repeated lifecycle-selected sources feeding `OnEach` sinks; and
- `lifecycle_cross_process_rejected`: one lifecycle-derived source split into a separate process
  from its dependent `OnEach` sink so compile reaches a real rejected plan with
  `RATE_LIFECYCLE_CROSS_PROCESS_UNSUPPORTED`.

Direct graph cycles remain graph-authoring rejections (`GRAPH_DIRECT_CYCLE`) before `compile()`.
They therefore stay unit-test evidence rather than executable benchmark rows; this harness does not
synthesize fake compile results for them.

The headroom mode uses one exact topology: a 1 kHz direct-Tensor source feeding a ~60 Hz
(`16,666,667 ns`) `DrainAll` sink with automatic queue sizing and runtime-pool Tensor storage.
The runner pairs `1/1` and `2/1` headroom probes for each requested payload (`small`, `large`) and
requested storage row (`host`, `pinned_host`, `cuda_device`) and queue policy (`blocking`,
`drop_oldest`). Each available row also runs finite real traffic through that same compiled graph:
after an unreported warm-up, the source allocates, initializes, and publishes the requested Tensor
payloads at a 1 ms clock while the sink consumes each `DrainAll` batch at its 16,666,667 ns clock.
The row reports the maximum selected batch size actually observed at those drain boundaries,
actual backpressured emit attempts, resource-exhausted/not-ready source attempts, and
consumer-visible replacement
count (`accepted - delivered`) together with finite-pipeline accepted-publication/delivery rates and
last-source-attempt-to-drain-boundary latency percentiles. Source retry wait is represented by the
attempt counters and is not included in those latency percentiles. The throughput denominator runs
from the last attempt of the first measured sample through the final measured sample's drain, so
these are finite-pipeline completion rates rather than isolated publisher-loop rates. The sampled
drain-boundary high-water mark is bounded diagnostic evidence; it is not a continuous occupancy
trace or a physical ring/reservation claim. A measured
zero counter remains zero, while an unavailable run retains `null` fields and a reason.

Available rows retain both immutable plan evidence and published Runtime generation evidence,
including the canonical
`render_rate_queue_inspection(..., kCanonicalJson)` join, `startup_to_published_generation_ns`
(broad startup timing, not allocator-exclusive), queue sizing, pool-capacity generation IDs,
provider decision and binding/allocation generation IDs, and optional stride or reserved-byte
evidence when Runtime publishes them. Full restart latency is not measured by this Task-041 probe,
so the schema records `restart_latency_ns: null` plus an explicit unavailable reason instead of
inventing a value. Unavailable rows stay explicitly unavailable with those runtime-only fields left
null rather than silently replacing missing runtime evidence with zeros.

Example:

```sh
python3 public/benchmark/graph/run_rate_planning_benchmark.py \
  build/benchmark/graph/rate_planning_benchmark \
  --mode compile_scaling \
  --topology linear_fixed --topology dual_rate_fusion \
  --topology lifecycle_cross_process_rejected \
  --operators 16 --operators 256 \
  --mode headroom \
  --payload small --payload large \
  --memory host --memory pinned_host --memory cuda_device \
  --queue-policy blocking --queue-policy drop_oldest \
  --headroom 1/1 --headroom 2/1 \
  --traffic-warmup-samples 34 --traffic-samples 64 \
  --repetitions 5 \
  --git-revision <exact-task-head> \
  --source-state <clean-or-dirty-snapshot-description> \
  --source-snapshot-sha256 <sha256> \
  --output-dir /tmp/rate-planning
```

The runner writes one raw directory per repetition plus one `summary.json` whose status is always
`characterization_only`. Compile-scaling summaries preserve deterministic plan identity and order
statistics. Headroom summaries preserve paired identity except for the requested headroom ratio and
report unique-pool aggregate slot/byte capacity rather than double-counting overlapping plan rows.
Before every child launch the runner removes any prior `result.json`, requires the child to create a
fresh regular result, and verifies the initially frozen benchmark-binary SHA-256 both before and
after each child plus immediately around summary publication. A relink or stale/missing result stops
the matrix instead of mixing rows or accepting old evidence.
Traffic summaries keep blocking and replacement policies as distinct paired workloads and aggregate
their real outcome counters, throughput, and latency across repetitions. Provider-reported stride
and total reserved bytes remain `null` when the provider does not publish them; the runner never
substitutes logical payload-capacity products for physical allocation evidence. Physical queue and
ring entries are likewise explicit `null` values with an unavailable reason because this Runtime
path does not publish that geometry; the logical 17/34-depth evidence must not be relabeled as a
provider-specific ring size.

## Instrumented rate-restart conformance probe

`run_rate_restart_conformance_probe.py` composes six exact focused `RunSessionTest` cases over the
same direct-host synthetic Tensor provider. Five rows cover the required outcomes: same-rate exact
reuse, smaller-rate compatible reuse, growth within a finite admitted maximum, admitted replacement
with bounded predecessor-candidate overlap, and a Discover value above the admitted maximum that
fails before restart `kAllocate`. The replacement row injects the private synthetic rate-reuse
compatibility fault; it is not evidence that an actual changed layout was observed. A sixth
supplemental row checks that synthetic incompatible reuse at `kArm` fails closed without allocating
a replacement. Every scenario in every repetition runs in its own process and retains its own raw
gtest JSON. The named tests are the conformance authority for the assertion-only rows. The
smaller-rate test additionally records instrumented timing, generation, callback, and resource
properties; other rows explicitly report timing as unavailable instead of applying one scenario's
measurements to another.

The instrumented smaller-rate test checks the restart barrier, stable run identity, new rate and
storage-binding generations, retained physical Tensor backing, the smaller active entry limit, and
the absence of a successor materialization. Its private test control records requested logical
payload-backing bytes; those bytes exclude provider metadata, alignment, and reservation overhead.
The provider does not publish reservation bytes on this path, so the summary reports them as
unavailable and `null` rather than inferring them from logical payload size.

The runner removes any prior raw result for a scenario repetition before launching its subprocess
and accepts only newly produced gtest JSON. It requires exactly the expected suite and test with
coherent zero-failure counters. For the instrumented smaller-rate case it also requires exactly two
Discover and two `kAllocate` callbacks and validates all timing, generation, and resource invariants.
The runner retains the raw gtest JSON, exact command, stdout, and stderr for every scenario
repetition. `summary.json` binds the complete matrix to one binary hash and optional source snapshot,
and reports minimum, median, p95, and maximum for the instrumented request-to-discover,
discover-to-user-`kAllocate`, `kAllocate`-to-successor-compute, and full
request-to-successor-compute intervals. These are non-product conformance-path characterizations,
not product latency thresholds or service-level objectives.

When `--hsb-validation-binary` is supplied, the same source-identity-bound summary also runs the
generated heterogeneous three-field `rate-fields-restart` and `rate-fields-restart-reject`
standalone scenarios. This is an HSB-shaped process-local schema boundary, not HSB transport or a
physical camera provider. The scenarios assert atomic compatible reuse and publication rollback.
They do not export timing, allocator-call, or registration-call properties, so those fields remain
`null` with explicit unavailable reasons. An actual HSB camera-provider matrix with those counters
remains a separate acceptance gap.

```sh
python3 public/benchmark/graph/run_rate_restart_conformance_probe.py \
  build/tests/runtime/runtime_test \
  --hsb-validation-binary build/tests/system/multi_tensor_schema_runtime_validation \
  --repetitions 5 \
  --git-revision <exact-task-head> \
  --source-state <clean-or-dirty-snapshot-description> \
  --source-snapshot-sha256 <sha256> \
  --output-dir /tmp/rate-restart-conformance
```
