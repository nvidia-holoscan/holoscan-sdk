# Holoscan SDK 5.0 EA3

Holoscan SDK 5.0 EA3 is a source-only Early Access release.

## Release Artifacts

- :octocat: **Source:** [`v5.0.0-ea3`](https://github.com/nvidia-holoscan/holoscan-sdk/tree/v5.0.0-ea3)
- 📕 **Documentation:** [Holoscan SDK 5.0 EA3 User Guide](https://nvidia-preview-v5-0-0-ea3.docs.buildwithfern.com/holoscan/sdk-user-guide/)

For supported platforms and system requirements, see
[Documentation → SDK Installation → Prerequisites](https://nvidia-preview-v5-0-0-ea3.docs.buildwithfern.com/holoscan/sdk-user-guide/setup/sdk-installation#prerequisites).

> [!WARNING]
> The Early Access documentation is derived from the in-development Holoscan SDK
> 5.0 guide and may contain links to binary artifacts or public GitHub content
> that are not part of EA3. Treat those links as unavailable for EA3; they are
> expected to become valid as part of the Holoscan SDK 5.0 GA publication.

> [!IMPORTANT]
> This release does not include prebuilt containers, Python wheels, Debian
> packages, or other binary artifacts. Build the SDK from the tagged source.

## <a id="release-notes"></a>Release Notes

### New Features and Improvements

Holoscan SDK 5.0 is a ground-up rewrite of the Holoscan SDK runtime and API.
EA1 and EA2 labels identify earlier additions; EA3 labels identify changes in
this release. See Known Limitations and Known Issues for platform and
dependency constraints.

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

- **Private artifact schema v7 (current source)** - The current private artifact schema is
  `holoscan.execution_plan.private.v7`. No forwards or backwards compatibility is provided: there
  is no older-schema reader or translation path. Regenerate artifacts with the current SDK because
  any schema-identifier mismatch fails before admission. The private schema is not an ABI promise.

#### Programming model

- **Callback-scoped scalar borrowing (C++)** - `Input<T>::selected_sample()`
  exposes immutable selected data and metadata without a payload copy, heap allocation,
  or owning read claim. `sensor_io::ImageDescriptorScratch` prepares bounded metadata
  storage during startup for Image transforms; output pixels still use a separate
  Tensor loan. Views and references must not escape the compute callback. Existing
  owning receives and public port types are unchanged.
- **`holoscan::Graph` API** - Graph definition without an `Application`
  wrapper. `compile()` produces an immutable `ExecutionPlan`; `run_async()`
  returns a `RunSession` for deterministic asynchronous execution and clean
  shutdown.
- **Typed Operators and ports** - Operators declare strongly typed input and
  output ports; payloads flow as `Sample<T>` with causal metadata.
- **Reusable Graph embedding** - A compiled subgraph can be embedded into a
  parent graph and reused across multiple compositions.
- **CUDA Graph regions** - Applications can explicitly fuse an eligible fixed-shape, linear GPU
  region with `RegionConstraint::cuda_graph()`. Runtime captures member CUDA work once after
  lifecycle startup and replays it on the source's planned stream while retaining ordinary Tensor
  readiness and publication behavior. The ordinary execution path remains available for comparison.
- **Automatic graph partitioning and inspection (EA2)** - The production
  `holoscan.partition.source_chain/1` planner forms linear execution chains
  across the complete DAG, including fan-out, fan-in, diamond, and parallel
  edge topologies. Applications can preserve exact manual anchors and request
  an automatic remainder, use root `include_all()` for one complete composed
  partition, and inspect the accepted cover through `ExecutionPlan::partitions()`.
  Green Context bindings remain exact named-anchor requests; dedicated contexts
  are not assigned automatically.
- **CUDA streams follow execution partitions (current source)** - By default, each active
  execution-partition/device pair receives one Runtime-owned CUDA compute stream,
  allowing independent branches to submit GPU work independently while Tensor
  readiness and release ordering remain automatic. Applications can select
  `CudaComputeStreamPolicy::kSharedProcessDevice` in `CompileOptions` when
  measurements favor one ordinary stream per process group and device. Separate
  streams permit but do not guarantee concurrent CUDA execution.
- **Support-thread placement (EA3, C++)** - Applications can request CPU affinity
  for the run-session controller and, when present, the process-local Channel
  pump. The compiled plan records the admitted placement. A pump binding does
  not create a pump when the selected transport does not need one.
- **CUDA Green Context manual placement (EA2, developer preview)** - Applications
  can query a qualified CUDA 13 target, bind an exact manually authored execution
  partition to a driver-structured set of streaming multiprocessors, and inspect
  the accepted placement and diagnostics in the compiled plan. The compiler and
  Runtime validate the target, materialize the Green Context streams before
  operator execution, and preserve Tensor ordering across ordinary and Green
  Context streams. Automatic graph partitioning does not automatically size or
  assign dedicated contexts. See the
  [CUDA Green Contexts guide](https://nvidia-preview-v5-0-0-ea3.docs.buildwithfern.com/holoscan/sdk-user-guide/advanced/cuda-green-contexts).
- **Lifecycle stage graph** - Bounded operator lifecycle with deterministic
  startup, running, and shutdown stages.
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
  `TemporalContract` that is verified at compile time.
- **Deterministic time and clock identities** - Monotonic clock registry and
  execution identities for reproducible trace correlation.

#### Parameters

- **Typed parameter system** - Operators declare typed parameters with
  optional defaults; parameter values are snapshot-consistent across an
  activation.
- **Runtime parameter generations** - Parameter values can be updated at
  runtime between activations; a generation counter ensures consumers observe
  a consistent snapshot.
- **Parameter configuration and persistence** - Parameter sets are
  serializable to and from configuration files; an experimental tuner
  supports iterative parameter search.
- **Derived Tensor pool capacity (EA2)** - Tensor output byte-span capacity can
  be derived during compilation from Parameter bounds, input bounds,
  and lifecycle-selected limits. Nominal-rate evidence can independently derive
  the edge queue depth needed for that output. The compiled plan retains both
  derivations and reports invalid or incomplete formulas; explicit capacity and
  queue depth remain available when needed. See the
  [Tensor sizing guide](https://nvidia-preview-v5-0-0-ea3.docs.buildwithfern.com/holoscan/sdk-user-guide/data-and-payloads/tensors-in-depth).
- **Parameter-derived Tensor storage (current source, C++)** - Tensor outputs can select
  host or device memory from compile-time parameter values. A named projector
  can derive a bounded output byte span when the built-in size expressions are
  insufficient; the compiler checks and freezes both choices before execution.
- **Bounded Vision outputs (current source, C++)** - Vision operators can derive fixed
  output-pool bounds from their typed parameters or an input Tensor's maximum
  size. Invalid or overflowing bounds fail during setup or compilation.

#### Channels and transport

- **In-process channel binding** - Zero-copy operator-to-operator
  communication within a process via a shared temporal cache; no serialization
  is required.
- **Bounded Tensor allocation with CUDA stream integration** - Host-pinned and
  device `Tensor` producers with explicit device placement, CUDA stream
  association, and loan/commit lifetime semantics. *Requires a CUDA
  device.*
- **Interprocess channel runtime** - HoloMQ/iceoryx2-backed bounded IPC
  transport with zero-copy GPU memory across process boundaries.
- **External channel clients** - Standalone non-Operator processes connect to
  a running graph using only `endpoint-catalog.json` and the stable endpoint
  name; client code does not require HoloMQ headers.
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
  [External Topic Boundaries guide](https://nvidia-preview-v5-0-0-ea3.docs.buildwithfern.com/holoscan/sdk-user-guide/composing-applications/external-topics).
- **Larger catalog inline services (EA2)** - Ordinary catalog-attached scalar,
  POD, and generated-schema endpoints now support complete fixed-inline frames
  through 32 KiB instead of the previous 4-KiB catalog ceiling. After framing,
  the maximum application capacities are 32,744 bytes for scalar/POD and
  32,704 bytes for ordinary generated schemas. Raw Tensor and
  descriptor-Tensor backing remains out of band and does not count toward this
  inline limit. See the capacity details in
  [External Topic Boundaries](https://nvidia-preview-v5-0-0-ea3.docs.buildwithfern.com/holoscan/sdk-user-guide/composing-applications/external-topics)
  and [Standalone Channel Clients](https://nvidia-preview-v5-0-0-ea3.docs.buildwithfern.com/holoscan/sdk-user-guide/advanced/standalone-channel-clients).
- **Event-driven channel waits** - Typed subscribers block on their transport
  event lane with `wait_and_receive_sample` instead of polling
  `receive_sample`; a wait takes an absolute deadline, cancels through a
  `std::stop_token`, and reports timeout, cancellation, and end of stream as
  distinct outcomes.
- **External subscriber event-loop integration (EA3, C++)** - Supported scalar,
  Tensor, and generated-schema subscribers expose a move-only readiness handle
  for Linux `poll`, `select`, or `epoll`. Applications acknowledge a wake and
  drain nonblocking receives without a separate polling helper thread.

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

#### Visualization

- **HolovizOp CUDA output (EA3)** - `HolovizOp` can render to a CUDA-backed
  `schema::ImageT` output for another visualization, encoder, or inference
  operator without an intermediate display. The application binds the output
  to the intended CUDA device; see the source-tree CUDA sink example.

#### Python bindings

The `holoscan` Core package exposes Python graph authoring, compilation,
execution sessions, typed ports and parameters, Tensor interchange, and
external-channel clients. Tensor values support DLPack and NumPy/CUDA array
interfaces. Python coverage is broad but not identical to the C++ API;
module-specific and newly introduced facilities can have narrower coverage.

Python bindings for Vision, Visualization, and Inference are built as separate
CUDA 13 component distributions, respectively `holoscan-vision-cu13`,
`holoscan-visualization-cu13`, and `holoscan-inference-cu13`, each depending on
the matching Core distribution. Their canonical packages are
`holoscan_vision`, `holoscan_visualization`, and `holoscan_inference`;
installed providers can also be discovered under the corresponding
`holoscan.<module>` alias. Vision exposes registered image-processing
operators, including Bayer demosaicing, through generated typed facades;
Visualization and Inference expose `HolovizOp` and `InferenceOp` graph
authoring through their module facades. The separate wheel build support is
present in this source tree; this note does not claim that any wheel has been
published, is downloadable, or has passed EA3 qualification.

#### Example catalog

- **Install-tree smoke tests** - `hello_world`, `hello_core`, `hello_logger`,
  `hello_tensor`, `hello_schema`, `hello_matx`, and `hello_cccl` check SDK
  headers, CMake targets, and selected third-party dependencies in an SDK
  installation.
- **SDK example applications** - `slow_fast_fusion` shows a two-rate graph
  with temporal scheduling in C++ and Python; `camera_imu_control` shows a
  camera/IMU fusion graph with external import and control export in both
  languages; and `external_clients` supplies `external_client_relay`,
  `sensor_daemon`, and `result_recorder` companion programs with Python client
  variants. See the [Slow/Fast Fusion](https://nvidia-preview-v5-0-0-ea3.docs.buildwithfern.com/holoscan/sdk-user-guide/examples/slow-fast-fusion),
  [Camera + IMU Control](https://nvidia-preview-v5-0-0-ea3.docs.buildwithfern.com/holoscan/sdk-user-guide/examples/camera-imu-control),
  and [External Clients](https://nvidia-preview-v5-0-0-ea3.docs.buildwithfern.com/holoscan/sdk-user-guide/examples/external-clients)
  guides. These are SDK source examples.

#### Performance

- Runtime dispatch suppresses redundant HoloMQ wake notifications and avoids
  polling cycles when no event-backed readiness source can make progress,
  reducing activation and fanout overhead. The expanded benchmark suites
  attribute Runtime and transport costs separately; their dated measurements
  are reference evidence rather than guarantees for other systems. See the
  [Core Runtime Benchmarks](https://nvidia-preview-v5-0-0-ea3.docs.buildwithfern.com/holoscan/sdk-user-guide/performance/core-runtime-benchmarks)
  and [Channel Benchmarks](https://nvidia-preview-v5-0-0-ea3.docs.buildwithfern.com/holoscan/sdk-user-guide/performance/channel-benchmarks).
- Tensor pool and catalog tensor-backing sharing reduce allocation pressure.
- Channel and HoloMQ benchmark suite (`comm_benchmark`).


### <a id="breaking-changes"></a>Breaking Changes

- **API breakage**: The Holoscan SDK v5.0 API is not backwards compatible with
  Holoscan SDK v4.x releases.
- **HolovizOp image layers carry `schema::ImageT` (EA3)**: Image-kind layers of
  `holoscan::visualization::HolovizOp` now bind `Input<holoscan::schema::ImageT>`
  ports and check each frame's `encoding`, `color_space`, and `plane_layouts`
  against the declared `ImageLayerDesc`; geometry, text, and LUT layers keep
  `Input<holoscan::Tensor>`. The uniform `layer_input(i)` accessor is replaced by
  `image_layer_input(i)` and `tensor_layer_input(i)`, both indexed by declared
  layer. Migrate `graph.add_flow(camera->frame, viz->layer_input(0))` to
  `graph.add_flow(camera->frame, viz->image_layer_input(0))`, where
  `camera->frame` is an `Output<schema::ImageT>` published with
  `emit_tensor(loan, descriptor, options)`, and overlay edges to
  `viz->tensor_layer_input(i)`. `ImageT::width`/`height` are the visible size;
  NV12 frames carry the schema's two `plane_layouts` records (Y, UV), so the
  backing Tensor may be a flat byte buffer. A wire `color_space` of
  `UNSPECIFIED` is accepted into any layer; `SRGB` requires
  `ColorSpace::kSRGBNonlinear` and `REC601`/`REC709`/`REC2020` on a YUV format
  require the matching `yuv_model`. `holoscan::sensor_io` is now a public link
  and runtime dependency of every HolovizOp consumer;
  `find_package(holoscan CONFIG COMPONENTS visualization)` loads it. The layer
  budget stays `kMaxLayerPorts` (8) in any kind mix. This is a source and ABI
  break: rebuild consumers.
- **EA1 RC2 queue-policy simplification**: The internal source-only EA1 RC1 QA
  baseline exposed `QueuePolicy::kLatest`; RC2 removes it before the initial
  public release and assigns `kDropOldest` numeric value 1 in Core, Channel,
  and HoloMQ. RC1-compiled objects and libraries must not be mixed with RC2;
  perform a clean source rebuild. Migrate `kLatest` to `kDropOldest` with depth
  one to preserve its freshest-only mailbox behavior. Retaining an authored
  depth greater than one is an intentional semantic change: the queue now
  preserves the newest bounded N-sample backlog in FIFO order. See the
  [Composing a Graph guide](https://nvidia-preview-v5-0-0-ea3.docs.buildwithfern.com/holoscan/sdk-user-guide/getting-started/composing-a-graph).
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
  [Tensors 101 guide](https://nvidia-preview-v5-0-0-ea3.docs.buildwithfern.com/holoscan/sdk-user-guide/getting-started/tensors-101),
  and [Custom FlatBuffer Schemas guide](https://nvidia-preview-v5-0-0-ea3.docs.buildwithfern.com/holoscan/sdk-user-guide/data-and-payloads/custom-flatbuffer-schemas).
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
| — | Fixed Runtime startup for process-local 1:1 and fanout graphs whose application-defined FlatBuffers schema contains one descriptor-backed CUDA-device Tensor field. The single-field schema now uses the same admitted binding path without requiring a dummy host Tensor field. See the [Custom FlatBuffer Schemas guide](https://nvidia-preview-v5-0-0-ea3.docs.buildwithfern.com/holoscan/sdk-user-guide/data-and-payloads/custom-flatbuffer-schemas). |
| — | Fixed typed inline receive so temporary typed-slot contention cannot drop an already accepted sample while receive staging is blocked. |
| — | Fixed subscriber-owner wake routing for single-process graphs with multiple execution partitions. |
| — | Fixed bracketed temporal windows so their anchors stop after end of stream instead of repeatedly scheduling empty windows. See the [Temporal Contracts guide](https://nvidia-preview-v5-0-0-ea3.docs.buildwithfern.com/holoscan/sdk-user-guide/building-operators/temporal-contracts). |
| — | Fixed stale retained `LatestWithin` evidence retirement, including retained Tensor readiness and end-of-stream paths, so a stale sample cannot remain falsely reusable. See the [Temporal Contracts guide](https://nvidia-preview-v5-0-0-ea3.docs.buildwithfern.com/holoscan/sdk-user-guide/building-operators/temporal-contracts). |
| — | Fixed pinned-host Tensor deallocation ordering so ordinary and CUDA Graph pipelines retain the consumer stream until asynchronous access is safe to release. |
| 6525925 | Clarified the external-client example procedure: artifact emission writes a deployment bundle, while the no-argument example starts the graph used for its end-to-end smoke test. |
| 6526032 | Updated Python Tensor test and benchmark device selection to handle deferred CUDA initialization on mixed-GPU Tegra systems. The affected runtime completion test now waits for its readiness boundary before replaying an event. |
| 6571835 | Corrected physical-core detection for benchmark CPU selection on platforms such as IGX Thor, allowing the documented Core Runtime benchmark preset to launch with the recommended CPU selection. |
| 6572980 | Stabilized the Core Runtime benchmark suite's Tensor and transport cases across the affected platforms. |
| 6568005 | Corrected Holoviz Vulkan initialization to select matching surface-maintenance and swapchain-maintenance extensions, and to reject an unsupported pairing early. |

### Known Limitations

- Python bindings do not yet cover every C++ capability. The separately built
  module bindings require their native dependencies; Inference also requires a
  compatible TensorRT 10 runtime in the deployment environment.
- Interprocess Channel transport and external subscriber readiness integration
  use Linux facilities. CUDA-backed examples and module operators require a
  supported CUDA 13 environment and suitable hardware.

### <a id="known-issues"></a>Known Issues

The following limitations remain in this release.

| Issue | Description |
| ----- | ----------- |
| — | On AGX Thor, display output at refresh rates above 144 Hz can crash the SoC display engine, including when a high-refresh-rate mode is selected through X11. Use a display refresh rate of 144 Hz or lower. If the crash occurs, physically disconnect and reconnect power to recover; a software reboot is insufficient. |
| 6572420 | GH200 Core Runtime benchmark results, including tail latencies, can differ by more than 50% from the published x86_64 reference measurements. Re-evaluation on EA2 and EA3 was blocked by 6717629; the reference values are not cross-platform pass thresholds. |
| 6717629 | On GH200, `run_transport_benchmark_suite.py` can fail in `graph_interprocess_docs_pod_1to2_latency` with `kPublicationIndeterminate` and `HOLOMQ_BACKPRESSURED` during publication. |
| 6835093 | Parallel source builds on GH200 and x86 can fail during Inference Python stub generation because the Image schema module is not yet available. |
| 6835133 | Five example/package tests fail on IGX Thor: `EXAMPLE_HOLOVIZ_IMAGE_TOPIC.starts_and_stops_cleanly`, `EXAMPLE_CPP_OPERATOR_TESTING_GREEN_CONTEXT_TEST`, `EXAMPLE_CPP_RUNNING_AND_STOPPING_TEST`, `OPERATOR_TESTING.installed_examples`, `LOGGING_POLICY.installed_examples_package`. Corrections for the running-and-stopping and Green Context tests are planned for the following release; the remaining example failures are under investigation. |
| 6837623 | The borrowed-descriptor CUDA tests `RUNTIME.BorrowedDescriptorCuda.DelayedKernelAndFanoutPreventPrematureReuse` and `RUNTIME.BorrowedDescriptorCuda.SubmittedReadsSurviveFailureCancellationStopRestartAndCompletion` can fail on Spark and Jetson. The diagnosed queue-ordering problem is in the test; a correction is planned for the following release. |
| 6843297, 6845982 | GH200 visualization and full-suite testing report failures in `EXAMPLE_HOLOVIZ_IMAGE_TOPIC.starts_and_stops_cleanly`, `holoviz_test_device_ownership`, `holoviz_test_swapchain_resize`, `holoviz_test_gate_staleness`, `holoviz_test_present_timing`, `holoviz_test_render_pass_load`, `holoviz_test_window_surface`, `holoviz_test_window_surface_fault_injection`, `holoviz_test_holoviz_op_image_admission`, `holoviz_test_holoviz_op_text_admission`. Full-suite testing also reports a separate failure in `BENCHMARK_RUNTIME_GRAPH_POD_FANOUT_DROP_OLDEST_EVENT_WAKE_CONTROL`. |
| 6843550 | Visualization applications on IGX Thor with an RTX PRO 5000 Blackwell dGPU can fail when the compiled CUDA architecture and runtime GPU selection do not match the display GPU. Build with `CMAKE_CUDA_ARCHITECTURES=120` and run with `CUDA_VISIBLE_DEVICES=0` when index 0 identifies that dGPU. This configuration allowed visualization tests to pass; application-launch failures remain under investigation. |
| 6857163 | Visualization applications fail to launch with `--display` on DGX Spark. |
