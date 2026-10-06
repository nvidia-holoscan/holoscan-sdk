# Holoscan Benchmarks

This directory is the home for SDK performance coverage. When adding a new feature, add
representative benchmark coverage here alongside correctness tests.

The commands below assume they are run from the repository root that contains this
`benchmark` directory.

Benchmarks should:

- Use Google Benchmark targets from `benchmark::benchmark` or `benchmark::benchmark_main`.
- Follow the [Google Benchmark user guide](https://google.github.io/benchmark/user_guide.html)
  for benchmark registration, fixtures, arguments, counters, and timing controls.
- Register with CTest and set the `benchmark` label so they can run selectively with
  `ctest -L benchmark`.
- Keep default benchmark runtime short enough for frequent developer and CI smoke coverage.
- Install benchmark sources as reference material for contributors.

Benchmarks that only build with the local SDK source tree (the `runtime`,
`graph`, `time`, `temporal_contract`, `endpoint_catalog`, and
`runtime_aggregate` suites gated on `holoscan_SOURCE_DIR` in
`benchmark/CMakeLists.txt`) are white-box by design. They may include private
`src/` implementation headers with quoted paths, for example
`#include "channel/channel_payload_limits.hpp"`, construct `holoscan::detail`
plan and session records directly, and enable source-tree-only test-access
seams so they can time individual Runtime mechanisms that the public API does
not expose. The `<holoscan/...>` include convention applies to public headers
only. Benchmarks that must build from installed sources use public headers
exclusively.

Role-aware channel domain-comparison benchmarks are the main exception to the
Google Benchmark rule. They may use standalone publisher/subscriber roles plus
a runner script when the measured path must span separate processes or matched
concurrent threads and when role orchestration must stay outside the timed
loop.

## Add A Benchmark

1. Create a feature-specific subdirectory under `benchmark`, for example
   `benchmark/my_feature`.
2. Add a `CMakeLists.txt` in that subdirectory.
3. Add one or more benchmark source files.
4. Add the subdirectory from `benchmark/CMakeLists.txt`.
5. Add each benchmark executable to `holoscan-benchmark-targets` so `./run bench`
   refreshes it before execution.
6. Register each benchmark binary with CTest and set `LABELS benchmark`.

Pure Python benchmarks should be added under an `if(HOLOSCAN_BUILD_PYTHON)`
guard in the top-level `benchmark/CMakeLists.txt` and attached to
`holoscan-python-benchmark-targets`, not `holoscan-benchmark-targets`. This
keeps C++ benchmark builds available when Python bindings are disabled.

Use `core_hello` as the minimal pattern:

```cmake
add_executable(my_feature_benchmark my_feature_benchmark.cpp)
add_dependencies(holoscan-benchmark-targets my_feature_benchmark)
target_link_libraries(my_feature_benchmark
    PRIVATE
        benchmark::benchmark_main
        holoscan::core
)

add_test(
    NAME BENCHMARK_MY_FEATURE
    COMMAND my_feature_benchmark --benchmark_min_time=0.01s
)
set_tests_properties(BENCHMARK_MY_FEATURE
    PROPERTIES
        LABELS benchmark
)
```

## Build And Run With `./run`

For the full command reference and commented examples, run:

```sh
./run bench --help
```

List registered benchmarks:

```sh
./run bench --list
```

The list output shows both the CTest name and a short alias. The alias
`core_hello` and the CTest name `BENCHMARK_CORE_HELLO` are equivalent when
passed to `--name`.

Build missing benchmark targets automatically and run all benchmarks:

```sh
./run bench
```

`./run bench` executes inside the Holoscan build container and uses a Release
build by default. It runs benchmark binaries directly, so Google Benchmark output
keeps its normal formatting without CTest line prefixes. Before listing or
running benchmarks, it also builds the benchmark aggregate target so changed
CMake files, CTest metadata, and benchmark sources are refreshed. Use
`--no-build` only when you intentionally want to run an existing build tree
without checking for changes.

Run one benchmark by alias:

```sh
./run bench --name core_hello
```

The full CTest name works too:

```sh
./run bench --name BENCHMARK_CORE_HELLO
```

Use shell-style wildcards or repeat `--name` to select multiple benchmarks:

```sh
./run bench --name 'core_*'
./run bench --name core_hello --name another_benchmark
```

Export Google Benchmark JSON output:

```sh
./run bench --output-json benchmark.json
```

Pass Google Benchmark arguments to each benchmark executable:

```sh
./run bench --benchmark-args "--benchmark_min_time=0.1s --benchmark_repetitions=5"
```

Run through CTest when you want CI-style aggregation:

```sh
./run bench --ctest --name core_hello
```

For targeted local work, you can run custom build, test, or benchmark commands
inside the standard SDK container with `launch --run-cmd`. Run the commands
below from the SDK source root:

```sh
./run launch --run-cmd "cmake --build . --target core_hello_benchmark && \
  python3 /workspace/holoscan-sdk/benchmark/run_benchmark.py --run-ctest-benchmarks --regex core_hello"
```

## Benchmark Stability

Google Benchmark may warn that CPU scaling or ASLR is enabled. These are
host/kernel settings, not Docker-local settings, so an unprivileged container
cannot reliably fix them from inside the benchmark process.

Check the host settings before a benchmark run:

```sh
./run bench --check-system
```

For a lower-noise local run, use `--stable-system`:

```sh
./run bench --stable-system --name core_hello
```

This mode temporarily applies host-side settings while the benchmark command is
running:

- Saves current ASLR and CPU governor state.
- Sets `kernel.randomize_va_space=0`.
- Sets available CPU governors to `performance`.
- Runs the benchmark inside Docker with a less restrictive seccomp profile so
  the benchmark process can disable ASLR with `setarch -R`.
- Restores the saved host settings when the command exits.

For the transport and Runtime Graph latency suites, the separate
[`configure_cpu_performance.sh`](configure_cpu_performance.sh) helper also
disables exposed CPU idle states with a reported exit latency of at least 1 us.
It saves and restores the exact host settings and must run on the host through
`sudo`, outside the development container. See
[`runtime/HARDWARE_TUNING.md`](runtime/HARDWARE_TUNING.md) for matching dGPU,
Jetson, and IGX clock controls.

`--stable-system` is intended to reduce known benchmark noise sources and make
the run more reproducible. It does not guarantee lower latency or higher
throughput. In particular, setting the CPU governor to `performance` does not
pin the CPU to its maximum turbo frequency, and a normal run can still report a
faster result if the processor boosts higher for that run. Disabling ASLR can
also change code/data layout in a way that is neutral, better, or worse for a
small benchmark.

Use longer runtimes and repetitions to compare stability, mean, median, and
standard deviation. These options make the comparison more reliable, but they do
not make `--stable-system` inherently faster:

```sh
./run bench --name core_hello \
  --benchmark-args "--benchmark_min_time=1s --benchmark_repetitions=10 --benchmark_report_aggregates_only=true"
./run bench --stable-system --name core_hello \
  --benchmark-args "--benchmark_min_time=1s --benchmark_repetitions=10 --benchmark_report_aggregates_only=true"
```

`--stable-system` requires cached sudo credentials. Run `sudo -v` on the host
first if needed.

Use a non-release build only when debugging benchmark code:

```sh
./run bench --type debug --name core_hello
```

### Dispatcher Managed-signal Admission

The dispatcher's persistent and transient managed-signal rows require a
single-threaded process at domain admission. On Linux, a joined worker's entry
can briefly remain in `/proc/self/task`, including across benchmark cases.
The harness waits up to one second for those entries to retire before managed
admission. The census and wait are outside benchmark timing; persistent rows
still measure the complete create/run/close lifecycle. Caller-managed rows do
not perform this wait.

A census error or retirement timeout fails the benchmark with diagnostic
context. This setup wait neither retries failed domain creation nor relaxes
the Runtime's admission checks. Focused coverage is registered as
`BENCHMARK_RUNTIME_DISPATCHER_MANAGED_SIGNALS` and
`RUNTIME.BenchmarkSignalAdmission.*`; build `runtime_dispatcher_benchmark` and
`runtime_benchmark_signal_admission_test` before running those CTests.

## Transport Suite

`run_transport_benchmark_suite.py` composes the existing transport-oriented
HoloMQ, Channel, and Runtime Graph runners. It does not add benchmark kernels
or redefine their measurement windows. The current suite covers:

- HoloMQ intraprocess blocking 1:1/fanout throughput, fixed-inline polling
  round trips, and concurrent 1:1 polling-versus-WaitSet controls;
- HoloMQ interprocess inline and BufferHandle throughput/latency;
- the existing intraprocess, intraprocess-sidecar, and interprocess Channel
  POD/Tensor/Image matrix;
- intraprocess Graph 1:1, 1:2, and linear-chain throughput/latency; and
- interprocess Graph 1:1 POD/schema/Tensor/Image throughput and latency,
  three-process POD-chain throughput/latency, and POD 1:2 throughput/latency.

The interprocess Graph 1:2 saturated row uses the compiler-coalesced broadcast
service and graph-owned bounded pressure handling. Its queue-depth-two geometry
matches `graph_interprocess_1to1_throughput_q2`; summaries keep logical
publications, aggregate deliveries, and slowest-sink completion separate. The
1:2 latency row retains a 5 ms minimum pacing period so its tail measures the
sleeping-waitset path rather than sustained queue residence.
Interprocess Tensor and Image coverage remains 1:1; extending descriptor-backed
payloads to the 1:2 and three-process-chain fixtures is deferred until those
topologies have explicit provider/lifetime coverage in the native Graph runner.
The suite also does not treat decomposition microbenchmarks as topology
results.

The concurrent HoloMQ controls run publisher and subscriber work on distinct
threads. When `--cpus` is supplied, the master runner pins them to its first
two distinct physical-core selections. Polling rows repeatedly call the
nonblocking receive API; parked rows use event-enabled endpoints and a kernel
WaitSet. The serialized HoloMQ rows remain useful API-stage measurements, but
they are not substitutes for these concurrent controls.

Runtime Graph cases use the sleeping-waitset mode by default. Additional POD
1:1 latency rows use `--busy-poll` as explicit low-latency ceilings. They are
diagnostic controls, not recommended production defaults, because they keep
operator lanes active while waiting and therefore consume a CPU core.

Run the all-core host setup before entering the development container:

```sh
sudo ./benchmark/configure_cpu_performance.sh \
  apply /tmp/holoscan-cpu-performance.state
```

Ask the helper for up to eight online logical CPUs backed by distinct physical
cores on one socket:

```sh
CPUS=$(./benchmark/configure_cpu_performance.sh recommend 8)
```

When `/sys/devices/system/cpu/isolated` is non-empty, the helper selects only
from that isolated CPU set; otherwise, it considers every online CPU. When the
selected socket or isolated set exposes fewer physical cores than requested,
the helper returns all of them and reports the reduction on stderr. It does not
silently add housekeeping CPUs to a short isolated set. The recommendation is
deterministic and topology-based; it does not chase transient CPU utilization.
Run on an otherwise quiet machine and inspect a short utilization sample before
collecting a baseline:

```sh
mpstat -P "$CPUS" 1 5
```

Selected CPUs should remain close to 100% idle before the benchmark starts.
For stricter isolation, reserve a stable CPU set through the deployment's
cpuset or kernel-isolation policy. The helper automatically honors the kernel
isolated-CPU mask; pass an explicit list when using a cpuset that is not exposed
through that mask.

Then run the balanced suite from the SDK development container. This preset
uses three repetitions and p99-capable sample counts so routine full-suite
comparisons complete substantially faster than the extended production preset.
Balanced and production Tensor/Image latency rows use preinitialized immutable
backing so their publication-to-receive interval isolates transport and runtime
control. Their raw HoloMQ device-buffer control likewise uses no per-message
CUDA readiness or release fence because initialization completes before the
measured interval. Balanced and production also include a separate
stream-ordered HoloMQ device row with per-message producer readiness, consumer
wait, and deferred releases. The smoke preset keeps only that fully ordered
HoloMQ row and a full-write Graph latency row as functional stress checks.
The runner assigns the selected CPUs to publisher/operator lanes and Runtime
support threads as each topology needs. Run from the host checkout; `launch`
starts in the selected target's build directory, so `--build-dir .` refers to
that directory. Use the same `HOLOSCAN_BUILD_ARCH` setting as the build:

```bash
./run launch \
  --run-cmd "python3 ../benchmark/run_transport_benchmark_suite.py \
    --build-dir . \
    --output-dir transport-suite-\$(date +%Y%m%d-%H%M%S) \
    --preset balanced \
    --cpus $CPUS"
```

The docs preset accepts four or more selected physical cores. It records rows
whose operator and support-thread roles do not fit as topology-based skips:
a four-core run completes 17 rows, a six-core run completes 34, and an
eight-core run completes all 40. The generic suite summaries and
`core-runtime-tables.md` contain every completed row and state when the run is
partial. Updating the public documentation with `--update-doc` still requires
the complete eight-core, 40-row result.

To regenerate only the tables on the Core Runtime benchmark documentation
page, use the production-quality `docs-core-runtime` preset. It selects the
exact single-process and interprocess Graph rows used by that page, including
1:2 and 1:4 fanout, a 10-operator single-process chain, and the generated
multi-Tensor schema row. The single-process fanout rows place the source and
all subscribers either in one shared operator partition or in separate
partitions with one CPU per subscriber. The paired rows distinguish the
low-overhead serial HoloMQ path from cross-partition scheduling and parallel
subscriber execution. The docs preset uses descriptor-only raw Tensor and
multi-Tensor rows: compiler-owned pool backing is initialized during warmup,
and measured publications retain and transport descriptors without writing the
backing again. `runtime_benchmark::MultiTensorMessageT` contains a `uint64`
sequence plus two independent rank-1 `uint8` Tensor fields, `first` and
`second`; each docs field is 1 MiB and has its own compiler-owned pool:

Run this controlled recipe on a native Linux x86_64 host. The explicit target
matches `setarch x86_64`; `launch` resolves its build directory automatically.

```bash
./run launch --arch x86_64 \
  --security-opt=seccomp=unconfined \
  --run-cmd "setarch x86_64 -R \
    python3 ../benchmark/run_transport_benchmark_suite.py \
    --build-dir . \
    --output-dir transport-suite-docs-core-runtime-$(date +%Y%m%d-%H%M%S) \
    --preset docs-core-runtime \
    --cpus $CPUS"
```

The scoped seccomp override permits `setarch -R` to disable ASLR for the
benchmark process tree without changing the host-wide ASLR setting. Keep both
parts of the command together; the default development-container seccomp
profile rejects the required personality call. This command targets the
x86_64 documentation baseline; use the matching `setarch` architecture name
when collecting on another architecture.

The docs preset uses seven repetitions. Single-process latency rows retain the
production sample counts (at least 10,000 measured samples) and therefore
report p50, p99, and p99.9. Interprocess latency rows use 1,000 warmup plus
1,000 measured samples to keep the release workflow practical; they report
p50 and p99, but deliberately omit p99.9. Render the latency, logical-rate,
and aggregate-delivery tables with the build directory as the working
directory. The paths below are relative to that directory, matching the
`--build-dir .` suite command above:

```sh
python3 ../benchmark/summarize_runtime_benchmarks.py \
  transport-suite-docs-core-runtime-<timestamp> \
  --update-doc ../docs/benchmarks_core_runtime.mdx
```

Without `--update-doc`, the summarizer still writes
`core-runtime-tables.md` beside the suite artifacts.

Use `--preset production` for seven repetitions and at least 10,000 latency
samples where stable p999 evidence is required. Use `--preset smoke` for a
short functional run and `--groups` for a comma-separated subset. `--skip-gpu`
omits direct HoloMQ and Graph device rows; the existing Channel matrix
still includes its maintained device-storage cases. Use `--dry-run --no-build`
to generate `commands.sh` and `suite_manifest.json` without running anything.
After an interrupted run, pass `--resume --no-build` with the original
arguments and output directory to execute only failed and planned cases. An
unpinned run is accepted when `--cpus` is omitted, but it is not a controlled
performance baseline.

The master runner preserves every native artifact and invokes
`summarize_transport_benchmark_suite.py` to write `summary.json`,
`summary.csv`, and `summary.md`. The normalized table reports logical
messages/s, aggregate deliveries/s, and available p50/p99 latency. Its rows
also carry a versioned scenario identity with:

```text
scope / domain / topology / workload / payload / storage /
delivery / execution / synchronization / wake / geometry
```

Native `throughput` and `saturated` rows share the normalized
`saturated_throughput` workload class, while `latency` and
`notification_latency` share `rate_limited_latency`. The identity retains a
separate trigger mechanism so a serialized caller, publisher thread, Runtime
clock, and Runtime notification source do not become equivalent merely because
their workload class matches.

The same identity contract is embedded in `suite_manifest.json` and
`summary.json`. Native domain and payload spellings remain in the flat row.
Delivery distinguishes `direct_retained`, `holomq_inline`,
`holomq_sidecar`, and `holomq_buffer_handle`; execution distinguishes
serialized and concurrent endpoint work from shared, separate, and
per-subscriber Runtime lanes. Geometry includes fanout, relay count, queue and
pool bounds, process count, and the three byte-accounting dimensions. Graph
rows report notification-source capacity separately from edge queue depth and
freeze each subscriber's queue policy independently. The diagnostic
`--mixed-policy-fanout` row is limited to one shared-lane POD output feeding
one blocking and one drop-oldest subscriber so its eventless and forced-wake
variants retain identical authored graph topology.
Bulk rows also report pool slots, release capacity, publication-backpressure
capacity, and the maximum retained fanout release streams.
`native_benchmark` retains the exact producer-specific row name, while each
`native_runs` entry maps it to a manifest case, raw artifact, and replayable
command. This makes the normalized identity useful for matching without
discarding native provenance.

The suite sweeps the matched saturated 64-byte 1:1 POD case at queue depths
1, 2, 8, and 64 for sample-driven HoloMQ interprocess, Channel, Runtime Graph
intraprocess, and Runtime Graph interprocess rows. The serialized HoloMQ Google
Benchmark diagnostics do not expose their kernel-owned queue bound, and bulk
payload rows retain one representative depth to keep the full suite bounded;
these exclusions are recorded in `workload_geometry_policy` in both the
manifest and normalized summary.

Every normalized row records one common publication ledger: attempted,
accepted, retryable-backpressured, indeterminate, and completed counts. The
attempted count must equal accepted plus backpressured plus indeterminate, and
completed cannot exceed accepted. `publication_count_scope` identifies whether
warmup or setup publications are included. This ledger describes publication
outcomes; resource-acquisition retries that occur before publication remain
native diagnostics rather than being mislabeled as transport backpressure.

Matched sample-driven 1:1 rows use one normalized
`publication_to_receive_completion` anchor: the timestamp is taken immediately
before the public/layer publication and ends when receive succeeds. Runtime
Graph trigger, callback-entry, emit, scheduling, and retirement intervals
remain separate diagnostics and are not substituted for that common data-path
measurement. Existing
HoloMQ intraprocess Google Benchmark round-trip rows expose mean iteration
time rather than a per-message distribution, so that column is labeled
separately and p50/p99 remain empty.

The suite requests one latency period, 200 us by default. HoloMQ and Channel
sample-driven rows use absolute-deadline spin pacing at that period. Runtime
Graph rows retain their production trigger mechanisms: intraprocess rows use
the admitted `OnClock` trigger, while distributed rows use the notification
driver. The normalized result records the pacing mode, requested comparison
period, effective period, and any minimum-period override. Device-ready HoloMQ
BufferHandle rows retain at least 500 us for deferred-release headroom, and the
distributed Graph 1:2 row retains its current 5 ms flow-control minimum.
Serialized Google Benchmark diagnostics have no external pacer and remain
outside period-matched comparisons.

Every interprocess native result identifies its clock source, monotonicity, and
scope. Linux HoloMQ, Channel, and distributed Graph rows use
`clock_gettime(CLOCK_MONOTONIC)` so producer and consumer timestamps share one
same-host domain. Diagnostic Runtime Graph rows also perform a 4,096-sample
back-to-back clock-read calibration after the measured run. The normalized
`timestamp_read_pair_p50_ns` is therefore instrumentation context, not work
inside the message latency interval.

The production preset uses seven repetitions. Its inline policy uses 100,000
throughput or 10,000 latency samples, while host and device bulk rows use
smaller payload-class-specific counts; every class warms with 1,000 samples.
The suite manifest records those counts, minimum elapsed-duration targets, and
the 1,000/10,000-sample p99/p99.9 thresholds. Summaries retain raw repetitions
and report the primary metric's median, minimum, maximum, and coefficient of
variation. p99.9 is emitted only when at least 10,000 samples support it.

Production sample-driven rows pass `--payload-validation none`: sequence,
descriptor, protocol, and completion checks remain active, but full payload
content scans are excluded. Smoke and CTest rows use `all` where possible.
`final` validates one retained final payload after its receive timestamp and is
used for asynchronous CUDA cases that cannot perform a per-sample host check
without changing the workload. Native and normalized results record both the
mode and whether validation ran after or between timed operations.

Normalized rows use `inline`, `pageable_host`, `pinned_host`, and `device` for
storage and split byte accounting into:

- `application_payload_bytes`: the configured logical data body used for rate
  comparisons;
- `inline_transport_bytes`: the fixed inline value or descriptor bytes, not
  including transport implementation overhead; and
- `external_buffer_bytes`: the separately referenced BufferHandle or Tensor
  backing span.

For Tensor and Image rows, the configured backing is the application byte
count. Descriptor or fixed-capacity metadata bytes are reported separately as
inline transport bytes and are not added to that value. Direct-retained Graph
edges report zero inline transport bytes because they do not publish a HoloMQ
record. Transport-backed Graph Tensor and Image rows report the fixed private
Channel record sizes, guarded by Channel-side compile-time layout checks.
Transport-backed generated-schema rows report this field as `null` because
Runtime currently exposes only their maximum wire capacity, not each submitted
encoded length.

Maintained bulk comparisons use 1 MiB backing by default. HoloMQ BufferHandle
coverage reports pageable-host, CUDA-pinned-host, and device pools separately;
Channel and Graph Tensor/Image rows use the same backing size so descriptor and
ownership costs can be compared without silently changing payload geometry.

`payload_write_profile` distinguishes `full_payload_per_sample`,
`boundary_markers_per_sample`, `reused_preinitialized_contents`, and
`metadata_per_sample_reused_external_buffer`. This prevents descriptor-only or
immutable-backing rows from being interpreted as full-bandwidth writes.

`bulk_work_profile` provides the coarser comparison identity:

- `descriptor_only` uses immutable or preinitialized external backing and
  measures descriptor, ownership, synchronization, and materialization work;
- `full_write` writes the complete configured backing once per sample before
  publication. Saturated throughput includes that write; latency begins after
  commit but can include unfinished asynchronous device work and its readiness
  wait; and
- `not_applicable` identifies inline POD/schema rows.

The detailed and coarse fields are both retained because an ImageT row updates
its metadata per sample while reusing the same Tensor backing. ImageT remains a
separate payload scenario from plain Tensor even when both use 1 MiB backing.

Normalized bulk rows also expose stage diagnostics when the native benchmark
has an isolated timing seam:

- `pool_acquisition_p50_ns` is the HoloMQ BufferPool draw interval;
- `import_materialization_p50_ns` is HoloMQ BufferHandle import or the Channel
  Tensor/Image receive-and-materialize interval after readiness polling;
- `cuda_enqueue_p50_ns` is the Channel asynchronous CUDA-work enqueue interval;
- `deferred_release_p50_ns` is reserved for a future observable release
  completion seam; and
- `final_validation_ns` is one retained-payload validation after all timed
  transport operations.

A `null` stage is deliberate: the native layer does not expose that operation
as an independently measurable interval. In particular, reacquisition delay or
backpressure is not reported as deferred-release latency because it also
contains scheduling and competing-work effects. Final validation never extends
the publication-to-receive timing interval.

Every normalized row also reports three independent wake attributes:

- `event_wake`: publication can notify an event-backed waiter;
- `receive_polling`: the benchmark consumer repeatedly probes a nonblocking
  receive operation instead of parking; and
- `busy_poll`: Runtime operator lanes continuously rescan readiness rather than
  using their sleeping-waitset mode.

These attributes are intentionally not collapsed into a single "polling"
label. A Runtime row can use event wake for its Channel pump while its operator
lane independently uses either sleeping-waitset or busy-poll scheduling.
`runtime_support_threads` lists the Runtime controller, Channel pump, and
notification driver roles that actually entered; direct HoloMQ and Channel
rows report `none`.

Restore the exact host settings after the run:

```sh
sudo ./benchmark/configure_cpu_performance.sh \
  restore /tmp/holoscan-cpu-performance.state
```

## Direct CMake Fallback

Configure with benchmarks enabled:

```sh
cmake -S . -B build -G Ninja \
  -DHOLOSCAN_BUILD_BENCHMARKS=ON \
  -DHOLOSCAN_USE_SCCACHE=OFF
```

Build the sample benchmark target:

```sh
cmake --build build --target core_hello_benchmark
```

Run all benchmark-labeled CTest entries:

```sh
ctest --test-dir build -L benchmark --output-on-failure
```

## Build From Installed Sources

The installed benchmark folder is also buildable as a standalone CMake project:

```sh
cmake -S <install-prefix>/benchmark -B /tmp/holoscan-bench -G Ninja
cmake --build /tmp/holoscan-bench --target core_hello_benchmark
```

The installed SDK provides the `holoscan` CMake package, but it does not ship
Google Benchmark. Standalone benchmark builds first try `find_package(benchmark)`
and otherwise fetch Google Benchmark from GitHub at configure time. That fallback
lives in `benchmark/cmake/setup_dependencies.cmake`; full SDK builds use the
shared dependency module in `cmake/deps/benchmark.cmake` instead.

## Troubleshooting

If configure fails with `cannot find sccache executable`, keep
`-DHOLOSCAN_USE_SCCACHE=OFF` in the configure command or install `sccache` and
configure with `-DHOLOSCAN_USE_SCCACHE=ON`.

If the `build` directory was already configured with sccache enabled, rerun the
configure command with `-DHOLOSCAN_USE_SCCACHE=OFF` or start from a fresh build
directory.
