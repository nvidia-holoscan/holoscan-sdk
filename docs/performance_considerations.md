(performance_considerations)=
# Performance Considerations

This section discusses key performance considerations when designing and optimizing Holoscan applications.

## Operator Granularity and Scheduling Overhead

When designing Holoscan applications, it's important to understand the relationship between operator granularity and scheduling overhead. The current GXF-based scheduler/executor incurs overhead for:

- Selecting which operator to schedule next
- Invoking the operator's `compute()` method
- Passing messages (data/entities) between operators

For operators with trivial computations (e.g., basic arithmetic operations like addition, multiplication), this overhead can outweigh the actual computation time.

### Measured Overhead

The following measurements were obtained using the `scheduler_overhead_benchmark.py` script (installed to `/opt/nvidia/holoscan/bin/` or available at `scripts/` in the source tree). These values are illustrative and will vary depending on your hardware, OS, and Python environment—run the benchmark on your target system to obtain accurate figures:

#### Operator execution only (no messaging, two operators)

- Greedy Scheduler: ~10 µs/iteration
- Event-Based Scheduler: ~11–15 µs/iteration

#### With message passing (Tx → Rx)

- Greedy Scheduler: ~17 µs/iteration
- Event-Based Scheduler: ~25–62 µs/iteration

#### Test Configuration

The measurements above were obtained using the following configuration:

- **Script:** `scripts/scheduler_overhead_benchmark.py` (installed to `/opt/nvidia/holoscan/bin/scheduler_overhead_benchmark.py`)
- **Command-line flags:**
  ```bash
  --iterations 100000 --warmup-iterations 100 --workers 1
  ```

**Timing methodology:** The benchmark wraps `time.perf_counter()` around the entire `run_app()` call, which includes application initialization (constructor, `compose()`) and teardown—not just `compute()` execution. At high iteration counts (e.g., 100,000), this overhead is amortized, but for lower iteration counts the per-iteration time will appear inflated.

**Warmup behavior:** The warmup phase always runs with `num_workers=2` for the event-based scheduler, regardless of the `--workers` argument. This ensures consistent JIT/cache warm-up but means warmup conditions may differ from the actual benchmark when testing with `--workers 1` or `--workers 4`.

:::{note}
**Python-specific:** The event-based scheduler shows increased overhead with multiple worker threads in Python applications due to the Global Interpreter Lock (GIL) and thread synchronization. C++ pipelines do not incur GIL contention. We will update this guidance as more measurements become available.
:::

### Practical Guidance

- **Rule of thumb**: If your operator's computation takes less than ~20 µs, the scheduling and message-passing overhead may dominate the total execution time. Consider combining such operations into a single operator.
- **GPU-resident pipelines**: When using [GPU-resident operators](./gpu_resident.md) within a CUDA Graph, kernel transition latency is significantly lower (~0.5–2 µs), allowing for finer-grained operator decomposition.
- **Profiling is essential**: Use {ref}`NSight Systems traces <nsight-profiling>` and {ref}`data flow tracking <holoscan-flow-tracking>`
 to measure actual overhead in your specific application.
  Note: OS scheduling, CPU affinity (thread pinning), and Python version can materially impact these thresholds; validate on your target system.

### When to Split vs. Combine Operators

| Consider Splitting When... | Consider Combining When... |
|---------------------------|---------------------------|
| Each operator does substantial work (>100 µs) | Individual operations are trivial (<20 µs) |
| Operators can run in parallel | Operations must run sequentially |
| You need to reuse operators in different pipelines | Operations are always used together |
| You want clear separation of concerns | Messaging overhead is significant |

To reproduce the measurements on your system, run the benchmark script:

```bash
# From SDK installation
HOLOSCAN_LOG_LEVEL=ERROR python3 /opt/nvidia/holoscan/bin/scheduler_overhead_benchmark.py

# Or from source tree
HOLOSCAN_LOG_LEVEL=ERROR python3 scripts/scheduler_overhead_benchmark.py
```

Use `--help` for options such as iteration count and worker thread settings.

## See Also

- {ref}`holoscan-flow-tracking`
- {ref}`gxf-job-statistics`
- {ref}`nsight-profiling`
