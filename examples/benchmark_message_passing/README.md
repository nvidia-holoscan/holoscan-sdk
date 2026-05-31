# Message Passing Benchmark

This application benchmarks message latency and throughput by passing messages
between source and sink operators.

## Run the Benchmark

From the build directory:

```bash
./examples/benchmark_message_passing/cpp/benchmark_message_passing
```

Optional benchmark flags:

```bash
./examples/benchmark_message_passing/cpp/benchmark_message_passing \
  --threads 0 \
  --add-timestamp \
  --add-cuda-stream \
  --add-metadata
```

- `--threads N`: Uses `EventBasedScheduler` with `N` worker threads when `N > 0`.
- `--threads 0`: Uses `GreedyScheduler`.
- `--add-timestamp`: Attaches an additional GXF `Timestamp` component to each emitted entity.
- `--add-cuda-stream`: Attaches an additional `CudaStreamId` component to each emitted entity.
- `--add-metadata`: Attaches a `MetadataDictionary` component to each emitted entity.

## Metrics

For each trial, the benchmark results will display:

- **Threads**: The number of worker threads used by the scheduler.
- **Flows**: The number of source -> sink operator flows.
- **Total Messages**: The total number of messages passed between sources and sinks across all flows.
- **Message Throughput (msgs / s)**: The number of messages passed per second across all flows.
- **Average Latency (us)**: The average latency from emit to receive for all messages across all flows.
- **Max Latency (us)**: The maximum latency from emit to receive for all messages across all flows.
