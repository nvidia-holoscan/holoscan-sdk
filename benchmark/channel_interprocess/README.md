# Channel Domain Comparison Benchmark

This directory contains matched intraprocess and interprocess Channel benchmark
coverage. It is separate from the Google Benchmark microbenchmarks because the
comparison needs concurrent role orchestration, role-local results, and
benchmark-local percentile aggregation. The existing Google Benchmark rows
remain single-thread round-trip microbenchmarks and should not be compared
directly with these concurrent rows.

Both domains use the same publisher and subscriber loops, payload construction,
warmup, retry behavior, validation, timing, and JSON schema. Interprocess rows
run one process per role. Intraprocess rows use one `ChannelRuntime`, one
publisher thread, and one thread per subscriber, with a start barrier before
traffic begins. `--domain interprocess` remains the default.

`--domain intraprocess_serial` is a Tensor-only 1:1 diagnostic baseline. It
creates both sidecar endpoints on one shared runtime and runs `emit_sample()`
followed immediately by `receive_sample()` on the same thread. It uses the same
native `BlockMemoryPool`, Tensor metadata, validation, and JSON fields as
`intraprocess_sidecar`, but intentionally removes pipeline overlap and
cross-thread record ownership. The serial thread uses `--publisher-cpu`;
`--subscriber-cpu` remains part of the matched runner configuration but does
not identify a second execution thread in this mode.

The first slice measures a small inline POD payload over the public
`ChannelPublisher<T>` / `ChannelSubscriber<T>` APIs. Tensor mode additionally
measures a HoloMQ Tensor transferred through
`ChannelTensorTransferMode::kBufferHandle`, including BufferHandle export,
transport, import, and Tensor materialization in the subscriber process. Image
mode measures Image FlatBuffer metadata plus a descriptor-backed Tensor
BufferHandle, with the subscriber materializing the imported Tensor through the
tensor-schema Channel path. Tensor and Image modes support three storage
classes: `pageable_host` uses a pageable-host HoloMQ pool and maps to DLPack CPU
memory, `pinned_host` uses a CUDA-pinned host HoloMQ pool and maps to DLPack
CUDA-host memory, and `device` uses the HoloMQ device VMM / BufferHandle
path. Tensor and Image modes default to `pinned_host`. The default rows do not launch
device work or measure CUDA stream readiness/deallocation. Direct Tensor device
rows can opt into that work with `--async-cuda-work`.

## Role-Aware Harness

`channel_interprocess_benchmark` is a role-aware binary:

```bash
channel_interprocess_benchmark --role subscriber --payload pod --channel-name <name>
channel_interprocess_benchmark --role publisher --payload pod --channel-name <name>

channel_interprocess_benchmark --role subscriber --payload tensor --channel-name <name>
channel_interprocess_benchmark --role publisher --payload tensor --channel-name <name>

channel_interprocess_benchmark --role subscriber --payload image --channel-name <name>
channel_interprocess_benchmark --role publisher --payload image --channel-name <name>

# The runner uses coordinator mode for matched intraprocess rows:
channel_interprocess_benchmark --role coordinator --domain intraprocess \
  --expected-subscribers 2 --subscriber-cpu 3 --subscriber-cpu 4 \
  --publisher-cpu 2 --payload pod --channel-name <name>

# Tensor-only serialized 1:1 sidecar baseline:
channel_interprocess_benchmark --role coordinator --domain intraprocess_serial \
  --expected-subscribers 1 --subscriber-cpu 3 --publisher-cpu 2 \
  --payload tensor --channel-name <name>

# Tensor/Image storage options:
#   --tensor-storage pageable_host  # pageable host, DLPack kDLCPU
#   --tensor-storage pinned_host    # CUDA pinned host, DLPack kDLCUDAHost
#   --tensor-storage device  # CUDA device/VMM, DLPack kDLCUDA
```

`run_channel_benchmark.py` selects the execution model with
`--domain`. In interprocess mode it launches subscribers first and then the
publisher. In intraprocess mode it launches the coordinator, which creates the
role threads around a shared runtime. The measured loop does not use pipes or
acknowledgements; each message carries benchmark-local timestamps so the
subscriber can report end-to-end latency.

For pinned runs, the Python runner records each role's Linux package/core
topology and rejects publisher/subscriber assignments that are SMT siblings of
the same physical core. Other platforms record topology validation as
unavailable and skip the Linux sysfs check. `intraprocess_serial` validates only
the publisher CPU because both endpoint calls execute on that one thread.
Role affinity is applied after Channel runtime, endpoint, and optional CUDA
setup but before the start barrier, discovery, and warmup. Runtime and transport
service threads therefore retain the process's original affinity instead of
inheriting a measurement CPU.

Fanout runs use `--expected-subscribers N`. Before timing begins, the publisher
sends sequence-zero probes until one probe reaches exactly N subscribers. Each
subscriber discards those probes, so discovery is verified through the Channel
transport without adding coordination to the measured loop. The runner then
requires every subscriber to receive the full warmup and measured sequence with
no gaps or validation errors.

Each role result records the bounded geometry used by that run. POD rows have
no external pool. Tensor and Image rows report the fixed pool-slot and release
capacities, the queue-bound publication-backpressure capacity, and the maximum
fanout release-stream count. The Python runner verifies these values before
accepting the result so changing queue depth or fanout cannot leave ambiguous
bulk-resource geometry in the artifact.

Tensor and Image subscribers validate descriptor, storage, size, read-only,
metadata, and sequence properties for every sample. The default
`--payload-validation final` mode retains the final imported Tensor and performs
one full payload-content check after the measured receive interval. This keeps
throughput focused on descriptor transport, BufferHandle import, and Tensor
materialization instead of making it scale with a per-sample host scan or
device-to-host validation copy. Use `--payload-validation all` for correctness
runs that fully validate every Tensor inside the receive loop; do not compare
throughput from that mode across payload sizes. Use `--payload-validation none`
for production measurements: structural and sequence checks remain enabled,
but no full payload-content scan is performed.

Measured intervals:

| Interval | Role | Start | Stop |
| --- | --- | --- | --- |
| `channel_emit` | publisher | before `ChannelPublisher<T>::emit_sample()` | after it returns |
| `publisher_full` | publisher | start of measured iteration | after `emit_sample()` returns |
| `channel_receive` | subscriber | before a successful `receive_sample()` call | after it returns |
| `end_to_end` | subscriber | publisher timestamp immediately before emit | subscriber receive completion |
| `full_end_to_end` | subscriber | publisher timestamp at iteration start | subscriber receive completion |

For matched 1:1 comparisons, `end_to_end` is normalized as
`publication_to_receive_completion`. Payload construction needed to carry the
timestamp occurs immediately before the public Channel publication.

Empty subscriber polls are not recorded as `channel_receive` samples. They are
reported separately as `not_ready_polls`.

For `intraprocess_serial`, publisher and subscriber intervals are measured
around their respective calls in the same loop. The completed-rate window spans
the first measured emit through the final matching receive. This is serialized
round-trip capacity, not concurrent pipeline throughput; queue depth cannot
create a sustained backlog because each emit is drained immediately.

For fanout, combined JSON reports publisher-local logical samples and delivery
capacity, the minimum and maximum subscriber throughput, and the worst
per-subscriber receive and end-to-end percentiles. It also reports a common
completion window from the first measured publisher sample through the final
measured receive by the slowest subscriber. Completed logical and delivery
rates use that window, so queued subscriber drain time is not omitted.

`end_to_end` is a one-way timestamp interval, but it is not automatically a
no-backlog handoff latency. In the default saturated mode
(`--publisher-period-us 0`, typically with queue depth greater than one), the
publisher runs as fast as possible and `end_to_end` includes any time the sample
resides in the subscriber queue behind older samples. This is useful for
understanding saturated FIFO behavior, but it can be much larger than the
individual `channel_emit` and `channel_receive` costs.

The default `--workload throughput` leaves the publisher unpaced and reports
saturated capacity. For paced one-way latency, use `--workload latency`, a
queue depth of one or two, and a non-zero `--publisher-period-us` longer than
the subscriber drain time. The runner rejects a latency workload without those
constraints. The publisher waits before taking the per-sample timestamps, so
the configured pacing interval is excluded from `end_to_end`.

Latency JSON retains completed-rate fields for completeness and validation,
but marks the row `rate_limited: true`. With the default sleep pacing,
`publisher_period_us` is a minimum relative delay; the observed completed rate
may be substantially lower because host sleep intervals can overshoot and
publisher work is performed between delays. That observed rate primarily
reflects the pacing mechanism and is not a transport-capacity result. The
summarizer therefore leaves throughput-ratio columns empty for latency
workloads; use the emit, successful receive, and end-to-end percentile columns
instead.

`--publisher-pacing sleep` retains this low-CPU relative-delay behavior and is
the default. `--publisher-pacing spin` uses absolute publish deadlines and
busy-waits on the publisher CPU, resynchronizing after a missed deadline rather
than emitting a catch-up burst. Spin pacing is appropriate for short periods
where scheduler wake-up latency would dominate, but it consumes a dedicated
CPU. The JSON and summary output identify the pacing mode used.

For same-process 1:1 runtime chains, the expected production direction is to
keep adjacent operators on the same CUDA stream where possible, so no extra
per-hop CUDA event/wait is needed. Interprocess rows cannot literally share a
CUDA stream across processes; the default device Tensor rows in both domains
measure HoloMQ BufferHandle export/import and Tensor materialization, not
downstream kernel launch or additional stream synchronization work.

`--async-cuda-work` is a separate direct Tensor device row. The publisher
acquires a fresh HoloMQ lease for every publication, enqueues a device fill on
S0, and records that lease's Tensor readiness before emit. Reacquisition is
required so every handle carries the ready generation for its own fill. The
subscriber enqueues the imported Tensor's S1 wait, a device-to-device copy, and
the S1 deallocation dependency without a per-sample host wait. One final stream
synchronization and validation occurs after the measured loop. JSON output
reports `async_cuda_work: true`, `async_lease_acquisitions`, and separate
`cuda_work_enqueue_*` percentiles; `channel_receive` continues to measure only
Channel receive/materialization.

## Example

From the SDK source root after building benchmarks.
`run launch` starts in the selected target's
build directory, so the commands below use paths relative to that directory.
Set `HOLOSCAN_BUILD_ARCH` consistently with the build if you selected a target
explicitly.

```bash
./run launch --run-cmd \
  "cmake --build . --target channel_interprocess_benchmark -j 8 && \
   python3 ../benchmark/channel_interprocess/run_channel_benchmark.py \
     ./benchmark/channel_interprocess/channel_interprocess_benchmark \
     --samples 100000 --warmup 1000 --publisher-cpu 2 --subscriber-cpu 3"
```

Run the same role loops with one shared runtime and concurrent threads:

```bash
./run launch --run-cmd \
  "cmake --build . --target channel_interprocess_benchmark -j 8 && \
   python3 ../benchmark/channel_interprocess/run_channel_benchmark.py \
     ./benchmark/channel_interprocess/channel_interprocess_benchmark \
     --domain intraprocess --expected-subscribers 2 \
     --samples 100000 --warmup 1000 --publisher-cpu 2 \
     --subscriber-cpus 3,4"
```

Run the Tensor sidecar path as a serialized 1:1 baseline:

```bash
./run launch --run-cmd \
  "cmake --build . --target channel_interprocess_benchmark -j 8 && \
   python3 ../benchmark/channel_interprocess/run_channel_benchmark.py \
     ./benchmark/channel_interprocess/channel_interprocess_benchmark \
     --domain intraprocess_serial --payload tensor --tensor-storage device \
     --tensor-bytes 1048576 --expected-subscribers 1 \
     --samples 100000 --warmup 10000 --publisher-cpu 2 --subscriber-cpu 3"
```

Run a low-queueing, paced Tensor latency row:

```bash
./run launch --run-cmd \
  "cmake --build . --target channel_interprocess_benchmark -j 8 && \
   python3 ../benchmark/channel_interprocess/run_channel_benchmark.py \
     ./benchmark/channel_interprocess/channel_interprocess_benchmark \
     --domain intraprocess_sidecar --payload tensor --tensor-storage device \
     --workload latency --publisher-period-us 10 --subscriber-queue-depth 1 \
     --publisher-pacing spin \
     --tensor-bytes 1048576 --expected-subscribers 1 \
     --samples 100000 --warmup 10000 --publisher-cpu 2 --subscriber-cpu 3"
```

Tensor BufferHandle mode with default pinned-host storage:

```bash
./run launch --run-cmd \
  "cmake --build . --target channel_interprocess_benchmark -j 8 && \
   python3 ../benchmark/channel_interprocess/run_channel_benchmark.py \
     ./benchmark/channel_interprocess/channel_interprocess_benchmark \
     --payload tensor --tensor-bytes 1048576 \
     --samples 10000 --warmup 1000 --publisher-cpu 2 --subscriber-cpu 3"
```

Tensor BufferHandle mode with pageable-host storage:

```bash
./run launch --run-cmd \
  "cmake --build . --target channel_interprocess_benchmark -j 8 && \
   python3 ../benchmark/channel_interprocess/run_channel_benchmark.py \
     ./benchmark/channel_interprocess/channel_interprocess_benchmark \
     --payload tensor --tensor-storage pageable_host --tensor-bytes 1048576 \
     --samples 10000 --warmup 1000 --publisher-cpu 2 --subscriber-cpu 3"
```

Tensor device BufferHandle mode:

```bash
./run launch --run-cmd \
  "cmake --build . --target channel_interprocess_benchmark -j 8 && \
   python3 ../benchmark/channel_interprocess/run_channel_benchmark.py \
     ./benchmark/channel_interprocess/channel_interprocess_benchmark \
     --payload tensor --tensor-storage device --tensor-bytes 1048576 \
     --samples 10000 --warmup 1000 --publisher-cpu 2 --subscriber-cpu 3"
```

Tensor device mode with asynchronous S0/S1 work:

```bash
./run launch --run-cmd \
  "cmake --build . --target channel_interprocess_benchmark -j 8 && \
   python3 ../benchmark/channel_interprocess/run_channel_benchmark.py \
     ./benchmark/channel_interprocess/channel_interprocess_benchmark \
     --payload tensor --tensor-storage device --async-cuda-work \
     --tensor-bytes 1048576 --samples 10000 --warmup 1000 \
     --publisher-cpu 2 --subscriber-cpu 3"
```

Image schema plus Tensor BufferHandle mode with default pinned-host storage:

```bash
./run launch --run-cmd \
  "cmake --build . --target channel_interprocess_benchmark -j 8 && \
   python3 ../benchmark/channel_interprocess/run_channel_benchmark.py \
     ./benchmark/channel_interprocess/channel_interprocess_benchmark \
     --payload image --tensor-bytes 1048576 \
     --samples 10000 --warmup 1000 --publisher-cpu 2 --subscriber-cpu 3"
```

Image schema plus device Tensor BufferHandle mode:

```bash
./run launch --run-cmd \
  "cmake --build . --target channel_interprocess_benchmark -j 8 && \
   python3 ../benchmark/channel_interprocess/run_channel_benchmark.py \
     ./benchmark/channel_interprocess/channel_interprocess_benchmark \
     --payload image --tensor-storage device --tensor-bytes 1048576 \
     --samples 10000 --warmup 1000 --publisher-cpu 2 --subscriber-cpu 3"
```

Paced one-way latency mode with minimal queue residency:

```bash
./run launch --run-cmd \
  "cmake --build . --target channel_interprocess_benchmark -j 8 && \
   python3 ../benchmark/channel_interprocess/run_channel_benchmark.py \
     ./benchmark/channel_interprocess/channel_interprocess_benchmark \
     --payload tensor --tensor-storage device --tensor-bytes 1048576 \
     --workload latency --subscriber-queue-depth 1 --publisher-period-us 1000 \
     --samples 10000 --warmup 1000 --publisher-cpu 2 --subscriber-cpu 3"
```

For device mode, `channel_receive` still measures only the Channel receive call.
Correctness validation copies the received device Tensor to a reusable host
scratch buffer after the receive interval; aggregate loop throughput therefore
includes that validation work.

For publishable results, run under the performance CPU governor, pin publisher
and subscriber processes to known cores, and use enough samples for stable tail
latency estimates.

## Recommended Execution

Interprocess latency measurements are sensitive to CPU frequency scaling,
process placement, background load, and first-use setup costs. Use the
following workflow for numbers intended to compare commits, machines, or
transport changes.

### 1. Capture Host Metadata

The host-prep helper is metadata-only by default:

```bash
./benchmark/channel_interprocess/prepare_benchmark_host.sh
```

It records the branch, commit, dirty status, CPU topology, CUDA tooling, GPU
state, and current CPU governors under:

```text
build/channel-interprocess-host-prep-<timestamp>/
```

Keep this directory with benchmark results so later comparisons can explain
differences in CPU governor, GPU, CUDA, or platform state.

### 2. Apply Performance Tuning

For stable publishable runs, use the reversible top-level helper. It switches
CPU-frequency policies to the performance governor and disables idle states
whose reported exit latency is at least 1 us:

```bash
sudo ./benchmark/configure_cpu_performance.sh \
  apply /tmp/holoscan-cpu-performance.state
```

On systems with `cpupower`, the equivalent direct command is:

```bash
sudo cpupower frequency-set -g performance
sudo cpupower idle-set -D 1
```

The Channel metadata helper retains an optional convenience mode for
`jetson_clocks` and an explicitly selected `nvpmodel`, but it does not preserve
enough state for a publishable restoration workflow:

```bash
APPLY=1 SET_NVP_MODEL=1 NVP_MODEL=0 \
  ./benchmark/channel_interprocess/prepare_benchmark_host.sh
```

Use the same `nvpmodel` mode across all compared runs.
For publishable dGPU or Jetson/IGX results, follow the apply, verification, and
restore commands in
[`../runtime/HARDWARE_TUNING.md`](../runtime/HARDWARE_TUNING.md).

### 3. Pin Benchmark Processes

Pin the publisher and subscriber to isolated or otherwise quiet cores when
possible. For 1:1 runs:

```bash
python3 ../benchmark/channel_interprocess/run_channel_benchmark.py \
  ./benchmark/channel_interprocess/channel_interprocess_benchmark \
  --samples 1000000 --warmup 10000 \
  --publisher-cpu 2 --subscriber-cpu 3
```

For fanout runs, give each subscriber an explicit CPU:

```bash
python3 ../benchmark/channel_interprocess/run_channel_benchmark.py \
  ./benchmark/channel_interprocess/channel_interprocess_benchmark \
  --samples 1000000 --warmup 10000 \
  --expected-subscribers 4 \
  --publisher-cpu 2 --subscriber-cpus 3,4,5,6
```

When CPU isolation is not configured, avoid CPU 0 for publishable results when
possible because it often handles more kernel and interrupt activity.

### 4. Run Smoke Before Long Collection

Run a short smoke first:

```bash
./run launch --run-cmd \
  "cmake --build . --target channel_interprocess_benchmark -j 8 && \
   python3 ../benchmark/channel_interprocess/run_channel_benchmark.py \
     ./benchmark/channel_interprocess/channel_interprocess_benchmark \
     --samples 1000 --warmup 100 --publisher-cpu 2 --subscriber-cpu 3"
```

Then increase to at least hundreds of thousands of measured samples, and use
around one million samples for more stable p99.9 estimates.

### 4a. Run The Standard Matrix

Use `run_channel_benchmark_matrix.py` for the standard matched row set: POD,
Tensor system/host/device, and Image system/host/device. It defaults to 1:1 and
the interprocess domain for compatibility. Pass
`--domains intraprocess,interprocess` to run each configuration through both
execution models. `--fanouts 1,2,4` adds 1:2 and 1:4 POD, direct Tensor
host/device, and Image system/host/device rows. Pageable-system Tensor remains
at 1:1. This bounded expansion measures control, BufferHandle, and
descriptor-backed schema fanout costs. The wrapper preserves each combined
result as JSON, writes role logs under each row directory, records the exact
commands in `commands.sh`, and writes a deduplicated `summary.md`.

```bash
./run launch --run-cmd \
  "cmake --build . --target channel_interprocess_benchmark -j 8 && \
   python3 ../benchmark/channel_interprocess/run_channel_benchmark_matrix.py \
     ./benchmark/channel_interprocess/channel_interprocess_benchmark \
     --domains intraprocess,interprocess \
     --output-dir ./channel-comparison-smoke-20260710 \
     --pod-samples 100000 --host-samples 20000 --device-samples 10000 \
     --warmup 1000 --tensor-bytes 1048576 \
     --publisher-cpu 2 --subscriber-cpu 3"
```

Run the bounded fanout matrix with distinct pinned subscriber CPUs:

```bash
./run launch --run-cmd \
  "cmake --build . --target channel_interprocess_benchmark -j 8 && \
   python3 ../benchmark/channel_interprocess/run_channel_benchmark_matrix.py \
     ./benchmark/channel_interprocess/channel_interprocess_benchmark \
     --output-dir ./channel-interprocess-fanout-20260708 \
     --fanouts 1,2,4 --subscriber-cpus 3,4,5,6 \
     --pod-samples 100000 --host-samples 20000 --device-samples 10000 \
     --warmup 1000 --tensor-bytes 1048576 --publisher-cpu 2"
```

Those parameters match the compact engineering smoke matrix used on
2026-06-30 after the HoloMQ memory-kind mapping update. The row-specific
commands generated by the wrapper are saved to:

```text
channel-interprocess-smoke-20260630/commands.sh
```

For publishable baselines, keep the same wrapper but increase sample counts,
run after host preparation, and preserve the host-prep output directory next to
the matrix output directory.

### 5. Preserve Raw Outputs

The runner writes a combined JSON result to stdout and leaves per-role files
under `--output-dir` or a temporary `hsdk-channel-ipc-bench-*` directory:

```text
publisher.json
publisher.log
subscriber_0.json
subscriber_0.log
...
```

Intraprocess runs additionally preserve `coordinator.jsonl` and
`coordinator.log`; the runner splits successful coordinator output into the
same `publisher.json` and `subscriber_*.json` files used by interprocess rows.

Preserve the full output directory together with the host-prep directory. The
role logs are needed to diagnose failed runs, timeouts, CUDA setup
failures, or unexpected not-ready polling behavior.

### 6. Summarize Results

Use the summarizer to convert one or more runner outputs into a comparison
table:

```bash
python3 ../benchmark/channel_interprocess/summarize_channel_benchmark_results.py \
  channel_interprocess_pod_run channel_interprocess_tensor_run
```

The input can be a combined runner JSON file, a single output directory
containing `publisher.json` and `subscriber_*.json`, or a parent directory with
multiple run outputs. When both domains have identical workload parameters,
host, and CPU placement, the summary reports the interprocess/intraprocess
completed-delivery-rate ratio; unmatched rows leave that field empty. Markdown
is the default output; use `--format csv` for spreadsheet ingestion.
