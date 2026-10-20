# Tensor Python Protocol Benchmark

This pure-Python benchmark compares `holoscan.Tensor.as_tensor()` import
and `holoscan.Tensor` export across the supported zero-copy Python
protocols.

For import, the DLPack C exchange fast path is measured by passing a
`torch.Tensor` directly. The PyTorch Python DLPack fallback is measured with a
small proxy object that exposes only `__dlpack__` and `__dlpack_device__`,
hiding the type-level `__dlpack_c_exchange_api__` capsule. CuPy DLPack import
is measured separately because CuPy does not currently expose the C exchange
API and its Python DLPack path has different cost from PyTorch. Host
array-interface import is measured with a NumPy-backed proxy that exposes only
NumPy `__array_interface__`. Device array-interface import is measured with
separate PyTorch- and CuPy-backed proxies that expose only
`__cuda_array_interface__`. PyTorch CPU tensors do not expose
`__array_interface__`, so there is no PyTorch host array-interface row.

For export, the benchmark measures Holoscan Tensor conversion through the
DLPack C exchange test helper, which validates the C exchange table and
round-trips the exported tensor, Python `__dlpack__` consumed by PyTorch, host
`__array_interface__` consumed by NumPy, and device `__cuda_array_interface__`
consumed by CuPy.

The benchmark also includes stable-ABI evaluation rows: repeated
`Tensor.shape` and `Tensor.strides` property access, versioned `Tensor.__dlpack__`
capsule creation, and `Tensor.as_tensor()` round-trip from another Holoscan
Tensor. Use a multi-dimensional shape when comparing sequence builders so the
measurement includes more than one tuple entry.

Run directly from the SDK source root after building Python bindings.
Use the same `HOLOSCAN_BUILD_ARCH` setting as the build when selecting a target explicitly:

```bash
BUILD_DIR=$(./run get_build_dir)
PYTHONPATH="${BUILD_DIR}/python/lib" \
  python3 benchmark/tensor_python_as_tensor/tensor_as_tensor_fastpath_benchmark.py \
  --device auto --shape 2x2x2x2x2x2x2x2x2x2x2x2 \
  --iterations 50000 --warmup 5000 --repeats 11 \
  --configuration nanobind-3.0.1-abi3 \
  --benchmark_out=nanobind-3.0.1-abi3.json --benchmark_out_format=json
```

To run in the build container, invoke `launch` from the same host source root.
It selects the build directory and sets the Python module search path:

```bash
./run launch --run-cmd \
  "python3 ../benchmark/tensor_python_as_tensor/tensor_as_tensor_fastpath_benchmark.py"
```

For the measurement-only nanobind split-mode prototype, install the official
`nanobind-backend` package into the benchmark environment and configure with:

```sh
cmake -S . -B build-python-split \
  -DHOLOSCAN_PYTHON_NANOBIND_SPLIT_MODE_PROTOTYPE=ON
cmake --build build-python-split --target holoscan-python
```

The prototype option is off by default. Production builds continue to use the
linked `STABLE_ABI` module unless recorded performance evidence changes that
decision. This manual path was last built and imported on x86_64 during the
2026-09-01 S0 evaluation; CI does not exercise it.

## CuPy handoff and complete owner retirement

`tensor_stream_lifetime_benchmark.py` measures a Source → Forward → Sink CuPy
pipeline under either SDK stream policy, or an external import/export-and-retire
batch. It checks frame counts, session diagnostics, and release of every CuPy
owner after synchronization and worker drain. It exits with code 77 when CuPy
or a visible CUDA device is unavailable. This Linux benchmark uses `/proc` task
counters and the SDK's private release-drain test helper.

Run from the SDK source root in the development container:

```bash
./run launch --run-cmd \
  "python3 ../benchmark/tensor_python_as_tensor/tensor_stream_lifetime_benchmark.py \
   --policy per_partition --frames 1000 --warmup 100 \
   --tensor-bytes 1048576 --period-us 1000 --output cupy-pipeline.json"

./run launch --standalone --run-cmd \
  "python3 ../benchmark/tensor_python_as_tensor/tensor_stream_lifetime_benchmark.py \
   --mode import_retire --frames 3000 --tensor-bytes 1048576 \
   --output cupy-import-retire.json"
```

Pipeline delivery and post-stop session retirement have independent deadlines:
`--delivery-timeout-seconds` (default 60) and `--retirement-timeout-seconds`
(default 10). Both must be positive, finite and representable by Python's timed
waits. For longer workloads, select deadlines that cover nominal delivery and
queued GPU work. For example, a `--frames 100000 --period-us 1000` run can use
`--delivery-timeout-seconds 300` and `--retirement-timeout-seconds 60`. These
settings apply to pipeline mode; the CTest process timeout is a separate harness limit.
The delivery wait also ends promptly if the session fails or stops, preserving its diagnostics
rather than waiting for the frame deadline after delivery has become impossible.

Pipeline edges default to `--queue-depth 2`, as in the original measurements. A lossless queue
can still reject an emit during a burst or consumer scheduling stall; forwarding callbacks are
not automatically replayed, and the benchmark treats such a failure as an invalid run. The
finite 50-frame CTest pipeline smokes explicitly use `--queue-depth 50` to accommodate their
entire burst, with ten-second delivery and retirement deadlines inside the sixty-second CTest
limit. This tests complete delivery and owner retirement without relying on consumer scheduling.
Use the same explicit depth in both builds for any comparison; smoke results at depth fifty
are not qualification of the default depth-two workload.
They do not limit the final device synchronization or owner drain.

Pipeline handoff percentiles measure callback entry timestamps, not GPU work
completion. `--gpu-work-us 3000 --period-us 100` adds a one-thread GPU spin per
frame to exercise retirement with queued GPU work. The full-lifetime wall
window excludes source warmup but includes session destruction, synchronization,
collection, release-worker drain, and owner verification. Process CPU starts
before session launch and includes warmup and retirement, divided by all frames.
Thread snapshots omit threads that exit before the final read and have Linux
CPU-tick resolution; use process CPU for aggregate cost. The source continues
receiving clock callbacks after its last emission until delivery finishes.

The external mode keeps one caller-owned stream alive throughout allocation,
Tensor import, CuPy DLPack export, reference drop, and drain. Its wall and CPU
averages include the full batch. It does not submit a GPU work kernel.

Use identical optimized builds, fixed CPU placement, alternating version order,
and multiple fresh processes when comparing revisions. The pipeline binds its
three partitions and support threads to distinct CPUs when its allowed affinity
contains at least five CPUs. JSON records the affinity, arguments, Python/CuPy
versions, callback distributions, whole-process CPU, surviving-thread CPU, and
owner-release result. An import-only loop can hide asynchronous release cost;
compare `full_lifetime_ns_per_frame` as well as submission timing.

CTest has diagnostic and `_PLAIN` cases for each pipeline policy and external
import/retirement, plus `_PACED` pipeline cases for both policies.

Use `--release-timings` to record the first emit attempt, successful emit return
and weakref owner release for each source frame. These intervals exclude
allocation, producer submission and any pacing wait before emit. Owner release
includes downstream references, completion fencing, worker batching and GIL
scheduling; it is an upper bound on GPU completion, not an exact GPU timestamp.

`--pace-window 4` waits on a blocking producer event recorded four frames earlier
before allocating the next frame. Backpressure retries keep the pending array
and its event. This bounds queued producer work, rather than downstream work or
all retained references. A fixed-rate input also needs bounded or dropping
ingress so delaying frame creation does not move its backlog upstream.

Combining these options timestamps the return from each existing producer-event
wait and reports `producer_completion_observation`. Its observation-to-release
lag remains signed if the owner was released before the host returned from the
wait. Final-window frames are counted as unobserved; the probe adds no extra
waits. The observation can be later than actual device completion, so this lag
does not by itself separate stream-tail fencing, worker batching or GIL delay.

Use `--diagnostics` for attribution runs. It records cumulative process CPU and
thread snapshots before session start, after entering the session, at the first
measured frame, at delivery end before requesting stop, after session completion
and destruction, and after final owner drain. GPU-bound warmup also brackets its explicit
device fence. Callback thread IDs and CPU totals let attribution distinguish the
three Python roles from other session threads. `phase_deltas` reports differences
in chronological order. In short/warmup-free runs, asynchronous startup can overlap
these boundaries. Thread snapshots still have CPU-tick resolution and omit threads
that exit before a read; the delivery-end snapshot captures the active session
threads. Snapshot scans add overhead to the following phase.

Diagnostics also sample CuPy cached/used pool bytes, free blocks, and live source
owners every `--sample-every` frames (default 128) and at phase boundaries. These
are sampled high-water marks, not an exact count of SDK retirement records, and
weak owner counts do not include every imported view or internal allocation.
Use a longer GPU-bound run, for example `--frames 2560 --warmup 320`, to inspect
pool growth over time. A bounded result for one workload is not a general bound.

`--stage-timings` separately times pipeline allocation, receive and emit, or
external allocation, import, export and reference drop. It adds timestamp
calls and changes timing; keep it out of primary latency comparisons. External
mode diagnostics report submission/drain CPU separately and pool counts at
submission end. Both diagnostic modes should be applied identically to compared
builds. Preserve uninstrumented runs for acceptance and use syscall/profiling
traces only to explain their behavior.
