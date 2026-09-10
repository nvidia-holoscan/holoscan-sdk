# Channel POD Benchmark

This benchmark measures the first Holoscan Channel wrapper over HoloMQ typed
pub/sub for small standard-layout, trivially-copyable payloads. It mirrors the
direct HoloMQ POD baseline in
`public/benchmark/holomq/typed_pubsub_benchmark.cpp` so the first
comparison isolates the overhead added by the Holoscan Channel API and erased
inline-payload bridge. Both layers use the exact 64-byte `BenchmarkPOD64`
comparison payload.

The benchmark covers:

- one publisher to one subscriber with `ChannelQueuePolicy::kBlocking`;
- one publisher to one subscriber with `ChannelQueuePolicy::kDropOldest` at
  depths 1 and 64;
- one publisher fanout to 2, 4, and 8 subscribers for blocking and depth-one
  drop-oldest.

The depth-one row is the freshest-only comparison. The depth-64 row keeps queue
occupancy at one in the serialized loop and provides a steady-state comparison
with the matching HoloMQ typed-POD row. Channel unit tests separately validate
depth-N overflow, FIFO retention, and Tensor sidecar reclamation.

## Follow-Up Benchmark Work

- Run formal comparisons with stable host settings:
  `./public/run bench --stable-system --name channel_pod`.
- Ensure the host is in performance-governor mode before collecting numbers.
- Add optional CPU affinity controls so publisher/subscriber polling executes on
  predictable cores, following the older experimental benchmark practice.
- Use longer runs and repeated measurements, for example
  `--benchmark_min_time=1s --benchmark_repetitions=10`.
- Export JSON for both this benchmark and the HoloMQ baseline so regression
  tracking can compare the same rows automatically.
- Add tail-latency collection only if the benchmark is extended beyond Google
  Benchmark mean/aggregate timing into per-message interval sampling.
- Keep this benchmark focused on small inline POD payloads. Tensor, schema, and
  buffer-lifetime benchmarks should remain separate because they exercise
  different hot-path mechanisms.
