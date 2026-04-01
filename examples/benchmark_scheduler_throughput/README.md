# Scheduler Throughput Benchmark

This application benchmarks scheduler throughput by counting simple increment operations over a
period of time.

## Run the Benchmark

From the build directory:

```bash
./examples/benchmark_scheduler_throughput/cpp/benchmark_scheduler_throughput
```

## Metrics

For each trial, the benchmark results will display:

- **Threads**: The number of worker threads used by the scheduler.
- **Operators**: The number of (unconnected) operators in the graph.
- **Total Operations**: The total number of operations performed across all operators.
- **Throughput (ops / s)**: The number of operations performed per second across all operators.
