# Holoscan SDK 5.0 EA1

Holoscan SDK 5.0 EA1 is a source-only Early Access release intended for
invited collaborators.

## Release Artifacts

- :octocat: **Source:** [`v5.0.0-ea1.post1`](https://github.com/nvidia-holoscan/holoscan-sdk/releases/tag/v5.0.0-ea1.post1)
- 📕 **Documentation:** [Holoscan SDK 5.0 EA1 User Guide](https://nvidia-preview-v5-0-0-ea1.docs.buildwithfern.com/holoscan/sdk-user-guide)
- 🧪 **Early Access samples:** [`holoscan-sdk-5.0.0-ea1/early-access-samples`](https://github.com/nvidia-holoscan/holohub/releases/tag/holoscan-sdk-5.0.0-ea1/early-access-samples)

For supported platforms and system requirements, see **Documentation → SDK Installation → Prerequisites**.

> [!warning]
> The Early Access documentation is derived from the in-development Holoscan SDK 5.0 guide and may contain links to binary artifacts or public GitHub content that are not part of EA1. Treat those links as unavailable for EA1; they are expected to become valid as part of the Holoscan SDK 5.0 GA publication.

> [!important]
> This release does not include prebuilt containers, Python wheels, Debian
> packages, or other binary artifacts. Build the SDK from the tagged source.

## <a id="release-notes"></a>Release Notes

### New Features and Improvements

Holoscan SDK 5.0 EA1 is a ground-up rewrite of the Holoscan SDK runtime and
API. All items below are new relative to the Holoscan SDK 4.x series. Unless a
platform restriction is noted, a feature runs on all supported platforms.

#### Programming model

- **`holoscan::Graph` API** - Graph definition without an `Application`
  wrapper. `compile()` produces an immutable `ExecutionPlan`; `run_async()`
  returns a `RunSession` for deterministic asynchronous execution and clean
  shutdown (EA1-006, EA1-009, EA1-010).
- **Typed Operators and ports** - Operators declare strongly typed input and
  output ports; payloads flow as `Sample<T>` with causal metadata (EA1-003).
- **Reusable Graph embedding** - A compiled subgraph can be embedded into a
  parent graph and reused across multiple compositions (EA1-007).
- **Lifecycle stage graph** - Bounded operator lifecycle with deterministic
  startup, running, and shutdown stages (EA1-008).

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

#### Channels and transport

- **In-process channel binding** - Zero-copy operator-to-operator
  communication within a process via a shared temporal cache; no serialization
  is required (EA1-012).
- **Portable POD and schema channels** - Plain-old-data (POD) endpoint types
  and FlatBuffers-backed schema channels for typed inter-boundary
  communication (EA1-013).
- **Bounded Tensor allocation with CUDA stream integration** - Host-pinned and
  device `Tensor` producers with explicit device placement, CUDA stream
  association, and loan/commit lifetime semantics (EA1-014). *Requires a CUDA
  device.*
- **Interprocess channel runtime** - HoloMQ/iceoryx2-backed bounded IPC
  transport with zero-copy GPU memory across process boundaries (EA1-015).
- **External channel clients** - Standalone non-Operator processes connect to
  a running graph using only `endpoint-catalog.json` and the stable endpoint
  name; client code does not require HoloMQ headers (EA1-016).

#### Graph artifacts

- **Cold artifact generation** - A graph-definition executable can emit its
  deployment bundle (`endpoint-catalog.json`, FlatBuffers schema packages, and
  compiled plan) without starting operator lifecycle, CUDA, or transport. The
  `--emit-artifacts` CLI flag and `_graph_artifacts` CMake build target support
  this workflow. Rebuilding the target is a no-op when nothing has changed.
- **HoloMQ module (`holomq::holomq`)** - Zero-copy GPU IPC publish/subscribe
  transport built on iceoryx2. It ships as both an SDK-integrated dependency
  and a standalone CMake package. Features include publish/subscribe queue
  controls, drop-oldest queues, and reusable host-memory imports that can be
  rebound across connection refresh.

#### Python bindings

The `holoscan.core` bindings include:

- `holoscan.core.Tensor`, `DLDevice`, `DLDeviceType`, `DLDataType`, and
  `DLDataTypeCode`, with full DLPack support, including the DLPack 1.3 C
  exchange API, and array interface support through `__array_interface__` and
  `__cuda_array_interface__`.
- `holoscan.core.hello()` and `holoscan.core.is_tensor_like()`.

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

- Local graph execution optimization reduces activation overhead for
  in-process paths.
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
  preserves the newest bounded N-sample backlog in FIFO order.
- **Platform availability**: Holoscan SDK v5.0 does not support IGX Orin
  platforms. See user documentation for additional platform support discussion.

### <a id="bug-fixes"></a>Bug fixes

| Issue | Description |
| ----- | ----------- |

## <a id="known-issues"></a>Known Issues

| Issue | Description |
| ----- | ----------- |
| 6525925 | In EA1, artifact emission and graph execution are intentionally separate: `ea1_camera_imu_control --emit-artifacts` writes the deployment bundle and exits without starting the graph or its Channel services. The `ea1_sensor_daemon` and `ea1_result_recorder` companions require a separately running deployed graph that owns their catalog endpoints, but EA1 does not provide a graph-only launcher for this manual three-process workflow. Use the no-argument `ea1_camera_imu_control` executable for the supported end-to-end smoke test. |
| 6526032 | On IGX Thor, the EA1 full test suite can report failures when PyTorch advertises CUDA availability but cannot initialize the exposed CUDA device. This affects the CUDA DLPack test and automatic Tensor benchmark device selection; CPU Tensor benchmark coverage remains available. |
| 6568005 | On GH200 and Jetson AGX Thor, the EA1 CTest `InterprocessPartitionBoundaryTest.CoalescedFanoutDeliversExactSequenceToEveryWorker` can time out during startup and report `RUNTIME_OPERATOR_CALLBACK_FAILED`. The underlying publication failure and its impact are under investigation. |
| 6571835 | On IGX Thor, the EA1 benchmark launcher can reject a valid CPU selection because the platform's exposed package and core identifiers do not distinguish the selected physical cores. As a result, the `docs-core-runtime` preset cannot be launched with the recommended eight-CPU selection on this platform. |
| 6572420 | The EA1 benchmark values published for x86_64 are reference measurements, not cross-platform pass thresholds. Individual results on aarch64 platforms such as GH200, particularly tail latencies, can differ by more than 50%. For EA1 compatibility testing, verify that the benchmark cases complete and that performance remains within the same order of magnitude. Treat median latency near or above 100 microseconds, or millisecond-scale results, as requiring investigation, including verification of the platform power-state setup. The complete performance-tuning procedure is needed only for reproducible measurements. |
| 6572980 | On DGX Spark, the EA1 `docs-core-runtime` benchmark suite can stop during the pinned-host Tensor 1:1 latency case with a protocol error. The failing Tensor invariant and runtime impact are under investigation. |
| — | In EA1, Runtime startup can fail for process-local 1:1 and fanout graphs that use a user-defined FlatBuffers schema containing a single descriptor-backed CUDA-device Tensor field. The failure is reported as `RUNTIME_PORT_BINDING_UNAVAILABLE` or `PROFILE_MATERIALIZATION_FAILED`. Add a dummy host Tensor field to select the multi-Tensor schema path as a workaround. |
