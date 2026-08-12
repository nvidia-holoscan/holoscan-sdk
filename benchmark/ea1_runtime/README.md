# EA1 runtime aggregate benchmarks

`ea1_runtime_benchmark` is the cumulative, source-tree-only EA1 build target.
It depends on the existing Dispatcher, compiled Channel, and raw Channel POD
benchmark executables plus the dedicated `ea1_temporal_benchmark`; it does not
recompile their implementation or benchmark translation units. The
`BENCHMARK_EA1_RUNTIME` CTest runs the selected cases sequentially through the
common benchmark launcher and fails if any constituent executable fails.

The controlled scenarios are:

- S1 dispatch: `BM_DispatcherReadySetCycle/{1,8,64}`;
- S2 multi-rate: `BM_EA1_S2MultiRateFresh`,
  `BM_EA1_S2MultiRateStalledSkip`, and
  `BM_EA1_S2MultiRateStalledReuseLast`;
- raw Channel publish/receive: `BM_ChannelPOD_OneToOne_Blocking`; and
- compiled Core Channel activation bookkeeping:
  `BM_ChannelActivationCycleByGraphOwners/2`.

The raw Channel case measures one public-API publish and receive of a 64-byte
POD sample. The compiled Core case measures `prepare_activation()` plus
`retire_activation()` for a two-owner plan; it intentionally excludes payload
transfer, Operator callbacks, and Graph compilation. Reports must preserve
those operation boundaries. The two observations provide raw-versus-Graph
coverage, but their numerical difference is not a like-for-like Channel
overhead measurement and must not be presented as one.

The raw Channel case inherits the current HoloMQ/iceoryx2 benchmark platform
matrix. This aggregate target does not by itself establish native MSVC or
clang-cl evidence.

The S1 cases measure one bounded Dispatcher freeze/select/complete cycle for
each ready-set size. The S2 cases measure the temporal-selector primitive over
a preloaded 64-sample fast-input cache for a fresh sample, a stalled-input skip,
and bounded `ReuseLast`; they are not end-to-end Slow/Fast application latency.

Google Benchmark records distributions, iteration/item counts, and the
case-specific queue or ready-set geometry. The common launcher does not add a
build-provenance manifest, so controlled reports must archive compiler, build,
host, affinity, and governor metadata separately. The suite does not define a
product latency threshold, subtract unrelated measurements, or treat a raw
Channel or selector microbenchmark as Graph system acceptance. Controlled
baseline and regression thresholds require a separately reviewed release
baseline.
