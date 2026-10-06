# CUDA Green Context qualification

This procedure qualifies the phase-1 manual Green Context implementation. It
uses the production `Graph`/`RunSession` Tensor path and compares each bound
layout with an unbound graph that has the same Operator-partition topology.
Throughput improvement is not an acceptance criterion.

## Static hot-path audit

The phase-1 implementation resolves CUDA placement before `kArm` and leaves the
callback path with cached handles:

- `RuntimeCudaExecutionResources::create()` materializes pools, contexts,
  streams, stream NVTX names, and stream-identity verification during Runtime
  materialization. `bind_partition_streams()` writes each verified native stream
  once into `RuntimeOperatorInstance::compute_stream`.
- `GraphExec::execute()` and the partition callback paths copy that cached field
  into `ExecutionContext`; `ExecutionContext::cuda_stream()` is a field read. No
  callback searches plan vectors or the process/device claim registry.
- Capability re-probing, `cuStreamGetGreenCtx()`, Green Context and stream
  creation, and NVTX naming occur only in
  `runtime_cuda_execution_resources.cpp`. Context push/pop is not used.
- Direct Tensor readiness, release, and retirement events are allocated as a
  fixed per-slot inventory in `RuntimeTensorCudaEvents::create()` before
  `kArm`. The callback path records, waits on, queries, and reuses those events;
  it does not create or destroy them. `RuntimeTensorOutputTestControl` can
  retain exact creation/destruction and record/wait/elision counts in a
  separately instrumented run.
- Tensor pools and their slot/control/readiness/release/retirement/quarantine
  records are sized from immutable plan capacities. Callback lookup uses fixed
  slot indices and bounded arrays; no per-stream associative container grows at
  runtime.
- NVTX stream and context names are the stable plan-local `stream_id` and
  `green_context_id`. Naming is cold-only. There is no production callback
  branch, formatting, or range construction for this instrumentation.

The benchmark retains an excluded warm callback allocation probe. Source and
sink compute callbacks must both report zero allocations in every matrix cell.
The instrumented row is kept separate because its source-tree counters use
locks and trace records and therefore are not suitable for performance
comparison.

## Matched matrix

Build `runtime_graph_benchmark`, then run from the source root in an
environment with the benchmark's runtime dependencies and a supported GPU.
Resolve the build directory using
the same `HOLOSCAN_BUILD_ARCH` setting as the build. Keep `BUILD_DIR` in the
same shell for the profiling and sanitizer commands below:

```bash
BUILD_DIR=$(./run get_build_dir)
python3 benchmark/runtime/run_green_context_qualification.py \
  "${BUILD_DIR}/benchmark/runtime/runtime_graph_benchmark" \
  --output-dir /tmp/green-context-qualification \
  --revision "$(git rev-parse HEAD)" \
  --samples 10000 --warmup 1000 --repetitions 5
```

`--output-dir` must be absent or empty and must not be a symbolic link. The
runner claims that directory for one attempt and refuses to overwrite retained
evidence; choose a new directory for every rerun. `attempt.json` is replaced
atomically as the attempt moves from `running` to `succeeded`, `skipped`, or
`failed`, so an interrupted status write leaves the last complete record.

The driver runs paced latency and saturated throughput for:

| Cell | CUDA placement | Matched baseline |
|---|---|---|
| `same_partition` | Source and sink share one dedicated context | `baseline_shared` |
| `dedicated_to_dedicated` | Source and sink use separate dedicated contexts | `baseline_separate` |
| `dedicated_to_residual` | Source is dedicated; sink and transfer work use the residual | `baseline_separate` |

Each performance repetition retains p50/p95/p99 end-to-end latency or saturated
throughput, cold compile/start/terminal-wait timing, planned device-pool bytes,
the exact canonical plan JSON, native command, and SHA-256 artifact identity.
The separate latency instrumented row retains callback/Runtime CPU evidence and
bounded phase timing. Latency and saturation instrumented rows both retain CUDA
event/wait/elision counts and the stream-quiesce portion of terminal cleanup;
phase tracing is omitted from saturation so its per-operation trace overhead
does not become artificial pool pressure. `summary.json` records run-to-run coefficient of variation
and a favorable-direction delta from the matched baseline; it deliberately
does not invent an unratified pass/fail performance threshold.

For a target-specific scaling control, pass `--green-context-sm-count <n>`
directly to `runtime_graph_benchmark`. Omitting it selects one structured
quantum as before. The explicit value must be a positive multiple of the
target quantum and must leave one quantum for the benchmark's compiler-owned
CUDA service. Use this control only in a separately reported cell; do not mix
it into the fixed five-cell matrix.

Use fixed application clocks, power limit, CPU affinity, and CPU performance
governor when publishing results. The summary records visible GPU UUID, PCI
identity, driver, persistence, power limit, current/maximum SM clocks, toolkit,
kernel, revision, raw paths, commands, and run count. Record unavailable clock
controls explicitly rather than treating them as fixed.

## Native attribution

Profile at least the three bound layouts. The benchmark already names every
materialized stream and Green Context with its stable plan-local identifier.
For example:

```bash
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --force-overwrite true \
  --output /tmp/green-context-dedicated-residual \
  "${BUILD_DIR}/benchmark/runtime/runtime_graph_benchmark" \
  --scenario tensor --tensor-storage device --tensor-bytes 1048576 \
  --operator-lanes separate \
  --green-context-layout dedicated_to_residual \
  --workload latency --samples 1000 --warmup 100 \
  --queue-depth 1 --publisher-period-us 1000 \
  --payload-validation final --timeout-ms 60000 \
  --output /tmp/green-context-dedicated-residual.json \
  --plan-output /tmp/green-context-dedicated-residual-plan.json
```

Export or inspect the report and join each CUDA operation's stream with the
plan's `cuda_compute_streams` record and NVTX name. The source memset must use
the source partition stream. Readiness waits and retirement must use the sink
or residual service stream selected by the layout. No HSDK transfer operation
may appear on an ordinary primary-context stream for an opted-in device.
Native `cuStreamGetGreenCtx()` verification before `kArm` is the executable
fail-closed guard; the Nsight report supplies independent profiler evidence.

## Stress and sanitizer

Use at least 10,000 samples for every matrix row and retain exact event
creation/destruction equality after terminal cleanup. Run the dedicated fanout,
same-partition, cross-context, dedicated/residual, cancellation, and restart
tests under Compute Sanitizer where the target supports the selected tool:

```bash
compute-sanitizer --tool racecheck --error-exitcode 99 \
  "${BUILD_DIR}/tests/runtime/runtime_test" \
  --gtest_filter='RunSessionTest.GreenContext*'
```

The CUDA race checker does not diagnose every host-side event-lifetime error;
the bounded metric and pointer-reuse assertions remain required. Record a tool
skip with its exact unsupported disposition instead of reporting it as a pass.

## Real-time profile scope

Terminal `RunSession::wait()` now includes dependency-safe synchronization of
every HSDK-owned stream before streams and contexts are destroyed. A
non-terminating kernel can therefore make the wait unbounded. Phase 1 admits
`kRealTime` Green Context placement under the qualified preallocated hot-path
contract: callback dispatch uses resources built before `kArm` and cached
stream bindings. This qualification does not promise bounded shutdown,
recovery, or restart after defective non-terminating CUDA work.
