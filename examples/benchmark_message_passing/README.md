# Message Passing Benchmark

This application benchmarks message latency and throughput by passing messages
between source and sink operators.

## Run the Benchmark

From the build directory:

```bash
./examples/benchmark_message_passing/cpp/benchmark_message_passing
```

## Metrics

For each trial, the benchmark results will display:

- **Threads**: The number of worker threads used by the scheduler.
- **Flows**: The number of source -> sink operator flows.
- **Total Messages**: The total number of messages passed between sources and sinks across all flows.
- **Message Throughput (msgs / s)**: The number of messages passed per second across all flows.
- **Average Latency (us)**: The average latency from emit to receive for all messages across all flows.
- **Max Latency (us)**: The maximum latency from emit to receive for all messages across all flows.
