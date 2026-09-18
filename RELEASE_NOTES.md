# Holoscan SDK v5.0.0-rc5

Holoscan SDK 5.0.0-rc5 is a source-only release candidate intended for
integration and evaluation of the reviewed SDK, samples, and camera source
tuple.

> [!WARNING]
> Holoscan SDK 5.0.0-rc5 is a source-only release candidate provided for
> integration and evaluation. This candidate has not undergone the
> Holoscan SDK formal SQA qualification process and is not a GA release.

## <a id="release-artifacts"></a>Release Artifacts

- :octocat: **Source:** `v5.0.0-rc5` (publication pending)
- 🧪 **Early Access samples:** `holoscan-sdk-5.0.0-rc5/early-access-samples`
  (publication pending)
- 📷 **Camera module:** `holoscan-sdk-5.0.0-rc5/holoscan-camera-0.4.0`
  (publication pending)
- 📕 **Documentation:** Holoscan SDK 5.0.0-rc5 User Guide (publication pending)

> [!WARNING]
> The release candidate documentation is derived from the in-development
> Holoscan SDK 5.0 guide and may contain links to binary artifacts or public
> GitHub content that are not part of 5.0.0-rc5. Treat those links as
> unavailable for this release candidate; they are expected to become valid
> as part of the Holoscan SDK 5.0 GA publication.

> [!IMPORTANT]
> This release candidate includes source only. It does not include prebuilt
> containers, Python wheels, Debian packages, or other binary artifacts. Build
> the SDK from the published source tag when it becomes available.

## <a id="release-notes"></a>Release Notes

### New Features and Improvements

Holoscan SDK 5.0 is a ground-up rewrite of the Holoscan SDK runtime and API.
Items below identify EA2 additions where applicable; all are new relative to
the Holoscan SDK 4.x series. Unless a
platform restriction is noted, a feature runs on all supported platforms.

#### Breaking Changes

- **Unified startup parameter authoring (EA2)** - The detached `ParamOverrideSet` path was removed,
  including `decode_overrides()`, `to_overrides()`, `load_overrides()`,
  `make_param_rebind_target()`, `ParamAdmissionContext`, and
  `CompileOptions::parameter_overrides`. Decode ordinary startup configuration with
  `try_decode<T>()`, move the resulting `PreparedParams<T>` into
  `OperatorHandle::set_params()`, and call the normal `compile(graph, options)`. For an atomic
  multi-operator configuration or snapshot restore, call the graph-owned `params::set_params()` or
  `load_and_set_params()` form before compiling. No compatibility shim is provided for this pre-GA
  API. Applying semantic edits with `Document::patched()` may clear its exact current-source digest;
  guarded saves of an edited document should use `document.digest().base_source` as
  `SaveOptions::expected_source_digest`. The pre-GA `decode_params<T>()` and
  `decode_params_or_throw<T>()` names are now `try_decode<T>()` and `decode<T>()`, respectively.

- **Automatic partitioning is the default topology and artifact contract (EA2)** - Graphs with
  no partition declarations now use the production source-chain planner. Multiple resulting
  partition owners may execute callbacks concurrently and change observable ordering across
  partitions. Applications that depend on the previous one-owner topology should use
  `partition(name).include_all()` as the compatibility form or author exact anchors. The public
  partitioning-plan record layout requires downstream rebuilds. Embedded partition-inspection and
  Tensor-placement keys now use the same
  segment-encoded, no-leading-slash `OperatorPath` grammar as Parameter records; replace legacy
  spellings such as `/pipeline/source` with `pipeline/source`.

- **Private artifact schema v4 (EA2)** - The current private artifact schema is
  `holoscan.execution_plan.private.v4`. No forwards or backwards compatibility is provided: there
  is no older-schema reader or translation path. Regenerate artifacts with the current SDK because
  any schema-identifier mismatch fails before admission. The private schema is not an ABI promise.

#### Programming model

- **`holoscan::Graph` API** - Graph definition without an `Application`
  wrapper. `compile()` produces an immutable `ExecutionPlan`; `run_async()`
  returns a `RunSession` for deterministic asynchronous execution and clean
  shutdown (EA1-006, EA1-009, EA1-010).
- **Typed Operators and ports** - Operators declare strongly typed input and
  output ports; payloads flow as `Sample<T>` with causal metadata (EA1-003).
- **Reusable Graph embedding** - A compiled subgraph can be embedded into a
  parent graph and reused across multiple compositions (EA1-007).
- **Automatic graph partitioning and inspection (EA2)** - The production
  `holoscan.partition.source_chain/1` planner forms linear execution chains
  across the complete DAG, including fan-out, fan-in, diamond, and parallel
  edge topologies. Applications can preserve exact manual anchors and request
  an automatic remainder, use root `include_all()` for one complete composed
  partition, and inspect the accepted cover through `ExecutionPlan::partitions()`.
  Green Context bindings remain exact named-anchor requests; dedicated contexts
  are not assigned automatically.
- **CUDA Green Context manual placement (EA2, developer preview)** - Applications
  can query a qualified CUDA 13 target, bind an exact manually authored execution
  partition to a driver-structured set of streaming multiprocessors, and inspect
  the accepted placement and diagnostics in the compiled plan. The compiler and
  Runtime validate the target, materialize the Green Context streams before
  operator execution, and preserve Tensor ordering across ordinary and Green
  Context streams. Automatic graph partitioning does not automatically size or
  assign dedicated contexts. See the
  [Green Context example](https://github.com/nvidia-holoscan/holoscan-sdk/tree/main/examples/green_context)
  and [CUDA Green Contexts guide](https://nvidia-preview-v5-0-0-ea2.docs.buildwithfern.com/holoscan/sdk-user-guide/advanced/cuda-green-contexts).
- **Lifecycle stage graph** - Bounded operator lifecycle with deterministic
  startup, running, and shutdown stages (EA1-008).
- **Managed shutdown signals (EA2)** - On Linux, blocking `run(plan)`
  synchronously accepts SIGINT and SIGTERM through an HSDK signal domain by
  default; the first accepted occurrence requests cooperative stop.
  `RunSignalPolicy::kCallerManaged` provides a blocking opt-out, while early
  `ManagedShutdownSignals` plus `run_async(plan, signals)` provides managed
  asynchronous execution. Plain `run_async(plan)` remains caller-managed. A
  second accepted occurrence delegates the actual signal under the unchanged
  prior policy; first-signal shutdown remains cooperative, with no forced
  cancellation or whole-run completion bound. A clean first accepted SIGINT
  now returns control after cleanup with a nonzero blocking result (currently
  `1`) instead of taking the default SIGINT termination path (typically shell
  status `130`).

#### Temporal scheduling

- **Temporal contracts** - `OnEach`, `LatestWithin`, and `Skip` combinators
  control per-input selection policy; each operator declares a
  `TemporalContract` that is verified at compile time (EA1-005).
- **Deterministic time and clock identities** - Monotonic clock registry and
  execution identities for reproducible trace correlation (EA1-002).

#### Parameters

- **Typed parameter system** - Operators declare typed parameters with
  optional defaults; parameter values are snapshot-consistent across an
  activation (EA1-004).
- **Runtime parameter generations** - Parameter values can be updated at
  runtime between activations; a generation counter ensures consumers observe
  a consistent snapshot (EA1-011).
- **Parameter configuration and persistence** - Parameter sets are
  serializable to and from configuration files; an experimental tuner
  supports iterative parameter search (EA1-018).
- **Derived Tensor pool capacity (EA2)** - Tensor output byte-span capacity can
  be derived during compilation from Parameter bounds, input bounds,
  and lifecycle-selected limits. Nominal-rate evidence can independently derive
  the edge queue depth needed for that output. The compiled plan retains both
  derivations and reports invalid or incomplete formulas; explicit capacity and
  queue depth remain available when needed. See the
  [derived Tensor pool capacity example](https://github.com/nvidia-holoscan/holoscan-sdk/tree/main/examples/derived_tensor_pool_capacity)
  and [Tensor sizing guide](https://nvidia-preview-v5-0-0-ea2.docs.buildwithfern.com/holoscan/sdk-user-guide/data-and-payloads/tensors-in-depth#size-queue-entries-from-nominal-rates).

#### Channels and transport

- **In-process channel binding** - Zero-copy operator-to-operator
  communication within a process via a shared temporal cache; no serialization
  is required (EA1-012).
- **Bounded Tensor allocation with CUDA stream integration** - Host-pinned and
  device `Tensor` producers with explicit device placement, CUDA stream
  association, and loan/commit lifetime semantics (EA1-014). *Requires a CUDA
  device.*
- **Interprocess channel runtime** - HoloMQ/iceoryx2-backed bounded IPC
  transport with zero-copy GPU memory across process boundaries (EA1-015).
- **External channel clients** - Standalone non-Operator processes connect to
  a running graph using only `endpoint-catalog.json` and the stable endpoint
  name; client code does not require HoloMQ headers (EA1-016).
- **Explicit external subscriber close (EA2)** - Scalar, POD, generated-schema,
  raw-Tensor, and descriptor-Tensor Channel subscribers now provide a checked
  `close()` operation for deterministic local endpoint retirement. Subscriber
  close discards unread local work and makes later checked operations report
  `kClosed`; it does not publish producer EOS, close a graph port, or contribute
  natural-completion evidence. Publisher `close()` remains the ordered EOS
  operation.
- **External producer EOS propagation (EA2)** - Ordered publisher EOS now has
  explicit end-to-end coverage through finite input-driven graph paths. Earlier
  accepted data and retained work drain before eligible Operator outputs close;
  live external subscribers observe final data before `kClosed`, and a fully
  finite graph reaches natural `kCompleted` without `request_stop()`. Inputs on
  independent triggers remain side inputs, and admitted external-restart
  capacity keeps outputs open for the replacement generation. A required export
  subscriber must remain attached through remote EOS; disappearance after data
  acceptance fails terminal drain rather than satisfying EOS or retrying
  indefinitely.
- **Descriptor-Tensor schema external topics (EA2)** - Graph imports and exports
  now carry application-owned singular or multi-Tensor descriptor schemas
  through the endpoint catalog. Standalone non-Operator peers attach with the
  catalog-aware tensor-schema publisher and subscriber facades, preserving
  typed per-field contracts, metadata, capture coordinates, bounded backing,
  cross-process operation, and ordered health, data, and end-of-stream control.
  Tensor bytes remain in retained or imported backing rather than being
  serialized inline. See the
  [descriptor-schema external-topic example](https://github.com/nvidia-holoscan/holoscan-sdk/tree/main/examples/flatbuffer_schemas/descriptor_tensor_external_topic)
  and [External Topic Boundaries guide](https://nvidia-preview-v5-0-0-ea2.docs.buildwithfern.com/holoscan/sdk-user-guide/composing-applications/external-topics).
- **Larger catalog inline services (EA2)** - Ordinary catalog-attached scalar,
  POD, and generated-schema endpoints now support complete fixed-inline frames
  through 32 KiB instead of the previous 4-KiB catalog ceiling. After framing,
  the maximum application capacities are 32,744 bytes for scalar/POD and
  32,704 bytes for ordinary generated schemas. Raw Tensor and
  descriptor-Tensor backing remains out of band and does not count toward this
  inline limit. See the capacity details in
  [External Topic Boundaries](https://nvidia-preview-v5-0-0-ea2.docs.buildwithfern.com/holoscan/sdk-user-guide/composing-applications/external-topics)
  and [Standalone Channel Clients](https://nvidia-preview-v5-0-0-ea2.docs.buildwithfern.com/holoscan/sdk-user-guide/advanced/standalone-channel-clients).
- **Event-driven channel waits** - Typed subscribers block on their transport
  event lane with `wait_and_receive_sample` instead of polling
  `receive_sample`; a wait takes an absolute deadline, cancels through a
  `std::stop_token`, and reports timeout, cancellation, and end of stream as
  distinct outcomes (EA1-016).

#### Graph artifacts

- **Cold artifact generation** - A graph-definition executable can emit its
  deployment bundle (`endpoint-catalog.json`, FlatBuffers schema packages, and
  compiled plan) without starting operator lifecycle, CUDA, or transport. The
  `--emit-artifacts` CLI flag and `_graph_artifacts` CMake build target support
  this workflow. Rebuilding the target is a no-op when nothing has changed.
- **HoloMQ library (`holoscan::holomq`)** - Zero-copy GPU IPC publish/subscribe
  transport built on iceoryx2 v0.9.3 and shipped with the Holoscan SDK. The
  `holomq::holomq` target remains a compatibility alias. Features include publish/subscribe queue
  controls, drop-oldest queues, and reusable host-memory imports that can be
  rebound across connection refresh.

#### Python bindings

The `holoscan.core` bindings include:

- `holoscan.core.Tensor`, `DLDevice`, `DLDeviceType`, `DLDataType`, and
  `DLDataTypeCode`, with full DLPack support, including the DLPack 1.3 C
  exchange API, and array interface support through `__array_interface__` and
  `__cuda_array_interface__`.
- `holoscan.core.is_tensor_like()`.

*CUDA 13 platforms only.*

#### Example catalog

- **Install-tree smoke tests** - `hello_world`, `hello_core`, `hello_logger`,
  `hello_tensor`, `hello_schema`, `hello_matx`, and `hello_cccl` verify that
  SDK headers, CMake targets, and third-party dependencies (`tl::expected`,
  `magic_enum`, `fmt`, `dlpack`, `flatbuffers`, MatX, and CCCL) are correctly
  installed (EA1-019 Wave 1).
- **EA1 reference applications** - `ea1_slow_fast_fusion` demonstrates a
  two-rate graph with temporal scheduling; `ea1_camera_imu_control`
  demonstrates a fusion graph with external IMU import and control export;
  and `ea1_external_clients` provides `sensor_daemon` and `recorder` companion
  processes (EA1-019 Waves 2-3). *Requires a CUDA device.*

#### Performance

- Runtime dispatch suppresses redundant HoloMQ wake notifications and avoids
  polling cycles when no event-backed readiness source can make progress,
  reducing activation and fanout overhead. The expanded benchmark suites
  attribute Runtime and transport costs separately; their dated measurements
  are reference evidence rather than guarantees for other systems. See the
  [Core Runtime Benchmarks](https://nvidia-preview-v5-0-0-ea2.docs.buildwithfern.com/holoscan/sdk-user-guide/performance/core-runtime-benchmarks)
  and [Channel Benchmarks](https://nvidia-preview-v5-0-0-ea2.docs.buildwithfern.com/holoscan/sdk-user-guide/performance/channel-benchmarks).
- Tensor pool and catalog tensor-backing sharing reduce allocation pressure.
- Channel and HoloMQ benchmark suite (`comm_benchmark`).

### <a id="breaking-changes"></a>Breaking Changes

- **API breakage**: The Holoscan SDK v5.0 API is not backwards compatible with
  Holoscan SDK v4.x releases.
- **EA1 RC2 queue-policy simplification**: The internal source-only EA1 RC1 QA
  baseline exposed `QueuePolicy::kLatest`; RC2 removes it before the initial
  public release and assigns `kDropOldest` numeric value 1 in Core, Channel,
  and HoloMQ. RC1-compiled objects and libraries must not be mixed with RC2;
  perform a clean source rebuild. Migrate `kLatest` to `kDropOldest` with depth
  one to preserve its freshest-only mailbox behavior. Retaining an authored
  depth greater than one is an intentional semantic change: the queue now
  preserves the newest bounded N-sample backlog in FIFO order. See the
  [graph composition example](https://github.com/nvidia-holoscan/holoscan-sdk/tree/main/examples/graph_composition)
  and [Composing a Graph guide](https://nvidia-preview-v5-0-0-ea2.docs.buildwithfern.com/holoscan/sdk-user-guide/getting-started/composing-a-graph).
- **EA2 Tensor port-spec revision**: Tensor port authoring now uses
  `TensorRepresentation`, `TensorBounds`, `TensorInputSpec`,
  `TensorOutputSpec`, and `TensorBoundaryContract`. Replace
  `TensorPortLayout` with `TensorRepresentation`; replace separate
  `TensorAllocationBounds` and `tensor_allocation()` calls with
  `TensorOutputSpec{.bounds = tensor_bounds(...), .storage =
  TensorOutputStorage::kRuntimePool}`; replace `LogicalTensorRequirements` and
  `LogicalTensorBounds` with `TensorBoundaryContract`; and use the same input
  and output specs for descriptor-schema Tensor fields. Transfer policy
  belongs in `TensorOutputSpec::transfer`, while input memory conversion
  belongs in `TensorInputSpec::memory_conversion`. This is an intentional
  source and ABI break: perform a clean rebuild, and do not mix libraries,
  plugins, generated artifacts, or endpoint catalogs from before and after
  this change. See the
  [Tensors 101 example](https://github.com/nvidia-holoscan/holoscan-sdk/tree/main/examples/tensors_101),
  [Tensors 101 guide](https://nvidia-preview-v5-0-0-ea2.docs.buildwithfern.com/holoscan/sdk-user-guide/getting-started/tensors-101),
  and [Custom FlatBuffer Schemas guide](https://nvidia-preview-v5-0-0-ea2.docs.buildwithfern.com/holoscan/sdk-user-guide/data-and-payloads/custom-flatbuffer-schemas).
- **Sensor payload schemas moved to `holoscan::sensor_io`**: `image.fbs`,
  `imu.fbs`, and the Image metadata helpers moved out of the Core schema
  library. Sources include `<holoscan/sensor_io/image_metadata.hpp>` instead of
  `<holoscan/schema/image_metadata.hpp>` and link `holoscan::sensor_io`. The
  affected symbols keep their `holoscan::schema` names, but
  `libholoscan_sensor_io.so` defines them instead of `libholoscan_schema.so`, so
  objects and libraries built against `5.0.0.1` must not be mixed with this
  release; perform a clean source rebuild.
- **Platform availability**: Holoscan SDK v5.0 does not support IGX Orin
  platforms. See user documentation for additional platform support discussion.
- **Linux blocking-run signal policy (EA2)**: `run(plan)` now requires either
  an adoptable compatible managed domain or exactly one live task at both
  process-task census reads; otherwise it fails before session creation. With
  no adoptable domain, successful census reads that change or are not exactly
  one report `RUNTIME_MANAGED_SIGNAL_DOMAIN_REQUIRED`, while a census read error
  reports `RUNTIME_MANAGED_SIGNAL_INSTALL_FAILED`; an unavailable domain or
  binding, or a poisoned process registry, reports
  `RUNTIME_MANAGED_SIGNAL_OWNER_BUSY`. Adopting an explicit
  domain consumes its sole binding attempt, so close it after retirement and
  create a fresh domain before another managed run. Hosts that already own
  process signals should use
  `run(plan, RunSignalPolicy::kCallerManaged)` or plain `run_async(plan)`;
  hosts needing managed async behavior should establish
  `ManagedShutdownSignals` before creating participant threads. A child
  created while that domain is active inherits the blocked SIGINT/SIGTERM mask,
  and `exec()` preserves it: reset the child mask with `sigprocmask()` before
  `exec()`, or use `POSIX_SPAWN_SETSIGMASK` with an application-chosen mask.
  `system()` and `popen()` do not perform that reset automatically.

### <a id="bug-fixes"></a>Bug Fixes

| Issue | Description |
| ----- | ----------- |
| — | Fixed Runtime startup for process-local 1:1 and fanout graphs whose application-defined FlatBuffers schema contains one descriptor-backed CUDA-device Tensor field. The single-field schema now uses the same admitted binding path without requiring a dummy host Tensor field. See the [descriptor-Tensor example](https://github.com/nvidia-holoscan/holoscan-sdk/tree/main/examples/flatbuffer_schemas/descriptor_tensor_schema) and [Custom FlatBuffer Schemas guide](https://nvidia-preview-v5-0-0-ea2.docs.buildwithfern.com/holoscan/sdk-user-guide/data-and-payloads/custom-flatbuffer-schemas). |
| — | Fixed typed inline receive so temporary typed-slot contention cannot drop an already accepted sample while receive staging is blocked. |
| — | Fixed subscriber-owner wake routing for single-process graphs with multiple execution partitions. |
| — | Fixed bracketed temporal windows so their anchors stop after end of stream instead of repeatedly scheduling empty windows. See the [Temporal Contracts guide](https://nvidia-preview-v5-0-0-ea2.docs.buildwithfern.com/holoscan/sdk-user-guide/building-operators/temporal-contracts). |
| — | Fixed stale retained `LatestWithin` evidence retirement, including retained Tensor readiness and end-of-stream paths, so a stale sample cannot remain falsely reusable. See the [Temporal Contracts guide](https://nvidia-preview-v5-0-0-ea2.docs.buildwithfern.com/holoscan/sdk-user-guide/building-operators/temporal-contracts). |

### <a id="known-issues"></a>Known Issues

| Issue | Description |
| ----- | ----------- |
| 6525925 | In EA1, artifact emission and graph execution are intentionally separate: `ea1_camera_imu_control --emit-artifacts` writes the deployment bundle and exits without starting the graph or its Channel services. The `ea1_sensor_daemon` and `ea1_result_recorder` companions require a separately running deployed graph that owns their catalog endpoints, but EA1 does not provide a graph-only launcher for this manual three-process workflow. Use the no-argument `ea1_camera_imu_control` executable for the supported end-to-end smoke test. |
| 6526032 | On IGX Thor, the EA1 full test suite can report failures when PyTorch advertises CUDA availability but cannot initialize the exposed CUDA device. This affects the CUDA DLPack test and automatic Tensor benchmark device selection; CPU Tensor benchmark coverage remains available. |
| 6568005 | On GH200 and Jetson AGX Thor, the EA1 CTest `InterprocessPartitionBoundaryTest.CoalescedFanoutDeliversExactSequenceToEveryWorker` can time out during startup and report `RUNTIME_OPERATOR_CALLBACK_FAILED`. The underlying publication failure and its impact are under investigation. |
| 6571835 | On IGX Thor, the EA1 benchmark launcher can reject a valid CPU selection because the platform's exposed package and core identifiers do not distinguish the selected physical cores. As a result, the `docs-core-runtime` preset cannot be launched with the recommended eight-CPU selection on this platform. |
| 6572420 | The EA1 benchmark values published for x86_64 are reference measurements, not cross-platform pass thresholds. Individual results on aarch64 platforms such as GH200, particularly tail latencies, can differ by more than 50%. For EA1 compatibility testing, verify that the benchmark cases complete and that performance remains within the same order of magnitude. Treat median latency near or above 100 microseconds, or millisecond-scale results, as requiring investigation, including verification of the platform power-state setup. The complete performance-tuning procedure is needed only for reproducible measurements. |
| 6572980 | On DGX Spark, the EA1 `docs-core-runtime` benchmark suite can stop during the pinned-host Tensor 1:1 latency case with a protocol error. The failing Tensor invariant and runtime impact are under investigation. |
