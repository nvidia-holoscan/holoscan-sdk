# Graph Authoring Benchmark

This source-tree-only Google Benchmark suite measures the cold EA1 Graph
authoring path. It records scaling for:

- Operator declarations without edges, which isolates append and side-index
  maintenance;
- linear chains and one-to-many fan-out DAGs, including cycle and fan-in
  validation; and
- immutable snapshot construction and canonical hashing;
- one embedded reusable Graph with increasing child size;
- increasing sibling count with ten Operators per child; and
- increasing nesting depth with a fixed total of 128 Operators, which isolates
  repeated hierarchy flattening and canonicalization from payload growth.

The direct-authoring and child-size series use 1, 10, 100, and 1,000 Operators.
The sibling series uses 1, 10, and 100 children, while the fixed-size nesting
series uses 1, 2, 4, and 8 hierarchy levels. The largest cases are regression
probes for accidental linear-to-quadratic behavior, not claims that every
deployment uses graphs of those sizes. CTest executes a short smoke run and
does not impose a workstation-specific latency threshold. Google Benchmark
reports complexity from each size series; historical results should be
compared when changing authoring indexes, validators, embedding, or snapshot
canonicalization.

Run the benchmark from the repository root:

```sh
./public/run bench --name graph_authoring
```

For a repeated characterization and JSON export:

```sh
./public/run bench --stable-system --name graph_authoring \
  --output-json graph-authoring-benchmark.json \
  --benchmark-args \
  "--benchmark_min_time=0.01s --benchmark_repetitions=5 \
   --benchmark_report_aggregates_only=true"
```

These are cold-path measurements. They do not model compilation, session
construction, scheduling, Channel attachment, or per-activation runtime work.
Those paths require separate benchmarks as their owning EA1 components land.
