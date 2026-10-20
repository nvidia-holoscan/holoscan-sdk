# Temporal Contract Benchmark

This source-tree-only Google Benchmark suite measures the payload-independent
baseline temporal cache and selection engines. It complements correctness and
allocation tests with scaling evidence for the private implementation used by
later compiler, Dispatcher, and Channel integration.

The benchmark emphasizes the common shallow retained depths of 1, 2, 4, and 8,
with sparse scaling probes at 16, 64, 256, and 1024, for:

- cache push, `Exact`, and `LatestWithin` selection;
- scalar cancellation and accepted front-sample commit;
- `DrainAll` snapshot, cancellation, full-batch commit, provisional arrival,
  and commit with a later arrival;
- aligned match/miss selection and accepted retirement; and
- bracketed-window close/materialization and completed-result pop/cache prune.

Fixture construction, queue population, transaction formation when measuring
only cancellation or commit, and state restoration are outside timed regions.
Operations that require fresh state use batches of independent prepared
fixtures. Timing is paused only between batches, so each reported iteration is
one operation while timer-control overhead is amortized across 32-128
operations. The batch size decreases with retained depth to keep fixture memory
bounded.
Scaling-sensitive rows call `SetComplexityN()` and register Google Benchmark
with the implementation's expected constant or linear complexity. CTest
executes the rows only as a short smoke test and does not impose a timing
acceptance threshold.

Run the benchmark from the repository root:

```sh
./public/run bench --name temporal_contract
```

For a repeated full-suite characterization and JSON export:

```sh
./public/run bench --stable-system --name temporal_contract \
  --output-json temporal-contract-benchmark.json \
  --benchmark-args \
  "--benchmark_min_time=0.001s --benchmark_repetitions=5 \
   --benchmark_report_aggregates_only=true"
```

For longer characterization, filter to the operation under study and increase
the minimum time. Destructive rows reconstruct their fixture outside the timed
region on every iteration, so increasing their minimum time also increases
untimed wall-clock work substantially:

```sh
./public/run bench --stable-system --name temporal_contract \
  --output-json temporal-contract-drain-commit.json \
  --benchmark-args \
  "--benchmark_filter=BM_DrainAllCommit.* --benchmark_min_time=0.01s \
   --benchmark_repetitions=10 --benchmark_report_aggregates_only=true"
```

The depths above 10 are regression probes rather than representative workload
weighting. Depth 1024 is a stress proxy, not a universal admitted maximum.
Deployment profiles with a different maximum should additionally measure that
exact bound. Google Benchmark repetitions characterize aggregate run-to-run
variation; they do not provide per-operation tail-latency distributions.

This suite intentionally does not simulate the synthesized `OnAnyInput`
ordinary-runtime path, production Channel-backed cache contention, a compiled
multi-rate Operator graph, or Tensor lease/stream retirement.
