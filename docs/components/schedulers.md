# Schedulers

The Scheduler component is a critical part of the system responsible for governing the execution of operators in a graph by enforcing conditions associated with each operator. Its primary responsibility includes orchestrating the execution of all operators defined in the graph while keeping track of their execution states.

The Holoscan SDK offers multiple schedulers that can cater to various use cases. These schedulers are:

1. [Greedy Scheduler](#greedy-scheduler): This basic single-threaded scheduler tests conditions in a greedy manner. It is suitable for simple use cases and provides predictable execution. However, it may not be ideal for large-scale applications as it may incur significant overhead in condition execution.
2. [MultiThread Scheduler](#multithread-scheduler): The multithread scheduler is designed to handle complex execution patterns in large-scale applications. This scheduler consists of a dispatcher thread that monitors the status of each operator and dispatches it to a thread pool of worker threads responsible for executing them. Once execution is complete, worker threads enqueue the operator back on the dispatch queue. The multithread scheduler offers superior performance and scalability over the greedy scheduler.
3. [Event-Based Scheduler](#event-based-scheduler): The event-based scheduler is also a multi-thread scheduler, but as the name indicates it is event-based rather than polling based. Instead of having a thread that constantly polls for the execution readiness of each operator, it instead waits for an event to be received which indicates that an operator is ready to execute. The event-based scheduler will have a lower latency than using the multi-thread scheduler with a long polling interval (`check_recession_period_ms`), but without the high CPU usage seen for a multi-thread scheduler with a very short polling interval. In general, this is an improvement over the older `MultiThreadScheduler` and provides additional features such as CPU thread pinning and options to enable Linux real-time scheduling.

It is essential to select the appropriate scheduler for the use case at hand to ensure optimal performance and efficient resource utilization. Since most parameters of the schedulers overlap, it is easy to switch between them to test which may be most performant for a given application. See {ref}`choosing-a-scheduler` for a decision summary, {ref}`scheduler-recipe-multi-branch-low-latency` for a worked configuration of a latency-sensitive multi-branch pipeline, and {ref}`scheduler-pitfalls` for common configuration mistakes.

Holoscan provides a base `holoscan::Scheduler` class ({cpp:class}`C++ <holoscan::Scheduler>`/{py:class}`Python <holoscan.core.Scheduler>`)) that each of these inherits from. This base class has a `clock` method ({cpp:func}`C++ <holoscan::Scheduler::clock>`/{py:func}`Python <holoscan.core.Scheduler.clock>`)) that can be used to retrieve the clock being used by the scheduler. This clock class provides a mechanism to get the time, a timestamp, sleep for some duration, or sleep until a target time. The specific behavior may depend on the concrete clock class being used. For all schedulers, the default clock is the `holoscan::RealtimeClock` class ({cpp:class}`C++ <holoscan::RealtimeClock>`/{py:class}`Python <holoscan.resources.RealtimeClock>`)). Examples of using the scheduler's clock from within the `compute` method of an operator are given in [examples/resources/clock](https://github.com/nvidia-holoscan/holoscan-sdk/tree/main/examples/resources/clock) and [examples/conditions/expiring_message](https://github.com/nvidia-holoscan/holoscan-sdk/tree/main/examples/conditions/expiring_message).

:::{note}
Detailed APIs can be found here: {ref}`C++ <api/holoscan_cpp_api:schedulers>`/{py:mod}`Python <holoscan.schedulers>`).
:::

## Greedy Scheduler

The greedy scheduler has a few parameters that the user can configure.

- The [clock](./resources.md#clock) used by the scheduler can be set to either a `realtime` or `manual` clock.
  - The realtime clock is what should be used for applications as it pauses execution as needed to respect user-specified conditions (e.g., operators with periodic conditions will wait the requested period before executing again).
  - The manual clock is of benefit mainly for testing purposes as it causes operators to run in a time-compressed fashion (e.g., periodic conditions are not respected and operators run in immediate succession).
- The user can specify a `max_duration_ms` that will cause execution of the application to terminate after a specified maximum duration. The default value of `-1` (or any other negative value) will result in no maximum duration being applied.
- This scheduler also has a Boolean parameter, `stop_on_deadlock` that controls whether the application will terminate if a deadlock occurs. A deadlock occurs when all operators are in a `WAIT` state, but there is no periodic condition pending to break out of this state. This parameter is `true` by default.
- When setting the `stop_on_deadlock_timeout` parameter, the scheduler will wait this amount of time (in ms) before determining that it is in deadlock and should stop. It will reset if a job comes in during the wait. A negative value means no stop on deadlock. This parameter only applies when  `stop_on_deadlock=true`.

## Multithread Scheduler

The multithread scheduler has several parameters that the user can configure. These are a superset of the parameters available for the `GreedyScheduler` (described in the section above). Only the parameters unique to the multithread scheduler are described here. The multi-thread scheduler uses a dedicated thread to poll the status of operators and schedule any that are ready to execute. This will lead to high CPU usage by this polling thread when `check_recession_period_ms` is close to 0.

- The number of worker threads used by the scheduler can be set via `worker_thread_number`, which defaults to `1`. This should be set based on a consideration of both the workflow and the available hardware. For example, the topology of the computation graph will determine how many operators it may be possible to run in parallel. Some operators may potentially launch multiple threads internally, so some amount of performance profiling may be required to determine optimal parameters for a given workflow.
- The value of `check_recession_period_ms` controls how long the scheduler will sleep before checking a given condition again. In other words, this is the polling interval for operators that are in a `WAIT` state. The default value for this parameter is `5` ms.
- The value of `strict_job_thread_pinning` controls then behavior when user-defined thread pools with thread pinning are used. If this value is `false` (the default), then whenever an operator pinned to a thread is not in a READY state, some other unpinned operator could make use of that thread. If `true` only the pinned operator can make use of the thread.

## Event-Based Scheduler

The event-based scheduler is also a multi-thread scheduler, but it is event-based rather than polling based. As such, there is no `check_recession_period_ms` parameter, and this scheduler will not have the high CPU usage that can occur when polling at a short interval. Instead, the scheduler only wakes up when an event is received indicating that an operator is ready to execute. The parameters of this scheduler are a superset of the parameters available for the `GreedyScheduler` (described above). Only the parameters unique to the event-based scheduler are described here.

- The number of worker threads used by the scheduler can be set via `worker_thread_number`, which defaults to `1`. This should be set based on a consideration of both the workflow and the available hardware. For example, the topology of the computation graph will determine how many operators it may be possible to run in parallel. Some operators may potentially launch multiple threads internally, so some amount of performance profiling may be required to determine optimal parameters for a given workflow. The `worker_thread_number` parameter creates a **default thread pool**. Any operators not explicitly assigned to a user-defined thread pool (via `make_thread_pool()`) will use this default pool.
- The worker threads in the default thread pool (created based on the `worker_thread_number` parameter) can be pinned to CPU cores via `pin_cores`. The parameter defaults to an empty list representing not to pin the worker threads to any CPU core. If a set of CPU core indices are given, all the worker threads in the default pool are pinned to the same set of specified CPU cores. Note that `pin_cores` only affects the default thread pool; to control CPU affinity for user-defined thread pools, use the `pin_cores` parameter in the `add()` or `add_realtime()` methods when assigning operators to those pools.
- The scheduler's separate dispatcher thread can also be pinned to one CPU core by setting `GXF_EBS_DISPATCHER_CPU_CORE=<core-id>` before launching the application. This is distinct from `pin_cores`, which only affects worker threads. The value must be an integer in the range `[0, 255]`.
- You can also configure the dispatcher thread to use `SCHED_FIFO` or `SCHED_RR` by setting `GXF_EBS_DISPATCHER_SCHED_POLICY` and `GXF_EBS_DISPATCHER_SCHED_PRIORITY`. When a dispatcher scheduling policy is specified, the priority variable must also be set to a valid priority for that policy.

For this scheduler, there is no `strict_job_thread_pinning` option (see description for the Multithread Scheduler above). The thread pinning is always strict.

### Advanced Performance-Tuning Parameters

The event-based scheduler exposes several advanced parameters that control internal scheduling strategies. These parameters are tuned to reasonable defaults and most users will not need to change them. They are primarily useful when profiling high-throughput, many-operator pipelines with large worker thread counts.

#### Work Stealing and Queue Assignment

Each worker thread in the default pool owns a private ready-queue. Operators are assigned to queues via a fixed, deterministic assignment computed once at graph launch. When a worker's own queue is empty, **work stealing** allows it to scan other workers' queues and take ("steal") a ready job instead of blocking. This reduces idle time when the workload is unevenly distributed across queues.

It is generally recommended to try enabling this for applications using the default worker pool, but the default has been kept as `false` in this release to avoid any unexpected change in default scheduling behavior.

- **`enable_queue_stealing`** (`bool`, default `false`) — Enables work stealing for default-pool workers.
- **`steal_scan_limit`** (`int`, default `0`) — Maximum number of other queues a worker scans per steal attempt. `0` means scan all queues.

#### Internal Event Sharding

The dispatcher receives notifications from workers and external events through internal queues. **Sharding** partitions these queues to reduce lock contention when many workers notify concurrently. There are two independent sets of sharded lists:

- The **internal notification shards** are the queues through which workers (and other internal paths) tell the dispatcher that an operator needs to be re-evaluated. These carry entity IDs for operators transitioning *into* any scheduling state (READY, WAIT, WAIT_TIME, WAIT_EVENT, or NEVER). The `internal_event_shard_count` and `dispatcher_internal_pop_batch_size` parameters control these queues.
- The **wait-state tracking shards** are lists that track which operators are currently in the WAIT_EVENT or WAIT scheduling states. These are the operators waiting on an asynchronous event (e.g., `AsynchronousCondition`) or a custom condition with no known ready time. The `wait_state_shard_count` parameter controls these lists. Note that WAIT_TIME operators (those with a known ready time, e.g., from `PeriodicCondition`) are tracked separately via a timed job list and are *not* affected by this parameter.

The sharding of notifications and batch pop are enabled by default for performance, but can optionally be disabled by setting `internal_event_shard_count=1`, `wait_state_shard_count=1` and `dispatcher_internal_pop_batch_size=1` to go back to the prior behavior in Holoscan<=v4.0.

Parameters:

- **`internal_event_shard_count`** (`int`, default `0`) — Number of shards for the dispatcher's internal notification queue. `0` selects one shard per worker thread automatically.
- **`dispatcher_internal_pop_batch_size`** (`int`, default `32`) — Maximum notifications the dispatcher drains from a single shard per pop step.
- **`wait_state_shard_count`** (`int`, default `1`) — Number of shards for the WAIT_EVENT and WAIT tracking lists.

#### Worker Post-Check Fast Path

After executing an operator, a worker can immediately re-check that operator's scheduling condition instead of sending it back through the dispatcher. This **post-check fast path** allows a worker to re-enqueue a still-READY operator directly into its own queue, avoiding a dispatcher round-trip and reducing latency.

When the post-check determines the operator is *not* ready, the worker falls back to notifying the dispatcher. A periodic fallback wake-up mechanism prevents rare edge cases where the dispatcher might miss a state change.

This optimization is currently disabled by default as it is still pending additional testing in real-world scenarios such as those involving thread priority (e.g. SCHED_FIFO for real-time threads).

- **`enable_worker_postcheck_fastpath`** (`bool`, default `false`) — Enables the worker-side post-check optimization.
- **`postcheck_fallback_notify_interval`** (`int`, default `256`) — Every N non-ready post-check results per worker, a periodic dispatcher wake-up is sent. `0` means only notify when no other workers are running.
- **`postcheck_fallback_notify_min_workers`** (`int`, default `8`) — The periodic fallback notification is only active when `worker_thread_number` is at least this value.
- **`postcheck_fallback_notify_min_period_ns`** (`int`, default `100000`) — Minimum wall-clock spacing (in nanoseconds) between periodic fallback dispatcher wake-ups.

#### Reverting to Holoscan 4.0 Scheduling Behavior

The optimizations above (work stealing, event sharding, and the post-check fast path) are new in Holoscan 4.1. As noted above, work-stealing and the post-check fast path are not currently enabled by default. A summary of the settings disabling all new scheduling feaetures is to set

`````{tab-set}
````{tab-item} C++
```cpp
  auto scheduler = make_resource<EventBasedScheduler>("scheduler",
      Arg("enable_queue_stealing", false),
      Arg("enable_worker_postcheck_fastpath", false),
      Arg("internal_event_shard_count", static_cast<int64_t>(1)),
      Arg("dispatcher_internal_pop_batch_size", static_cast<int64_t>(1)),
      Arg("wait_state_shard_count", static_cast<int64_t>(1)));
```
````
````{tab-item} Python
```python
scheduler = EventBasedScheduler(
    fragment,
    enable_queue_stealing=False,
    enable_worker_postcheck_fastpath=False,
    internal_event_shard_count=1,
    dispatcher_internal_pop_batch_size=1,
    wait_state_shard_count=1,
)
```
````
`````

Setting `internal_event_shard_count=1` and `wait_state_shard_count=1` disables sharding (all notifications go through a single queue). Setting `dispatcher_internal_pop_batch_size=1` disables batch popping so the dispatcher drains one notification at a time.

#### Diagnostics

- **`log_perf_stats`** (`bool`, default `false`) — When enabled, the scheduler logs internal instrumentation counters at shutdown. The report includes dispatcher loop and notification statistics, per-worker wait and execution times (with averages), work-steal attempt and success counts, and post-check fast-path/fallback hit rates. This is useful for diagnosing scheduling bottlenecks without requiring an external profiler.

(choosing-a-scheduler)=

## Choosing a Scheduler

The three schedulers differ in how they discover ready work and how much control they expose over CPU placement.

| Use case | Scheduler |
|---|---|
| Single-threaded debugging or fully serial pipelines | `GreedyScheduler` |
| Multi-threaded throughput workloads, no thread-affinity requirements | `MultiThreadScheduler` |
| Multi-threaded workloads requiring CPU pinning, real-time policies, or low-latency dispatch | `EventBasedScheduler` |

`EventBasedScheduler` is the recommended default for any pipeline that needs more than one thread. It avoids the polling-thread CPU cost of `MultiThreadScheduler`, supports per-branch CPU pinning, real-time scheduling policies (`SCHED_FIFO`, `SCHED_RR`, `SCHED_DEADLINE`) via user-defined thread pools created with `Fragment::make_thread_pool` ({cpp:func}`C++ <holoscan::Fragment::make_thread_pool>`/{py:func}`Python <holoscan.core.Fragment.make_thread_pool>`), and exposes the dispatcher itself for pinning via environment variables.

(scheduler-recipe-multi-branch-low-latency)=

## Recipe: Low-Latency Multi-Branch Pipeline

A common pattern is a single source feeding several parallel branches where one branch is latency-critical and the others must not steal cycles from it. The recipe below isolates each branch on its own pinned worker thread, gives the priority branch real-time scheduling, places the dispatcher on a dedicated RT core, and uses a separate CUDA stream pool per branch so GPU work follows the same priority story as CPU work.

For the pipeline shape itself (one source, multiple parallel branches with `add_flow`), see the [multi_branch_pipeline](https://github.com/nvidia-holoscan/holoscan-sdk/tree/main/examples/multi_branch_pipeline) example. The recipe below adds the scheduling configuration on top of that shape. For the underlying API signatures used here (`make_thread_pool`, `ThreadPool::add`, `ThreadPool::add_realtime`, scheduler `pin_cores`), see {ref}`configuring-app-thread-pools` and {ref}`configuring-app-thread-pools-realtime`.

`````{tab-set}
````{tab-item} C++
```cpp
// Pipeline shape (operators, add_flow): see public/examples/multi_branch_pipeline.
// The block below is the scheduling configuration added inside compose() on top
// of that shape. It assumes operators named `ae` (priority branch), `procA`,
// `procB` already exist.

// 1. Per-branch CUDA stream pools. Priority branch gets the highest
//    (most negative) priority; others use the default (0).
int low_pri = 0, high_pri = 0;
cudaDeviceGetStreamPriorityRange(&low_pri, &high_pri);

auto stream_priority  = make_resource<CudaStreamPool>(
    "stream_priority",
    Arg("stream_priority", high_pri),
    Arg("reserved_size", static_cast<uint32_t>(1)),
    Arg("max_size", static_cast<uint32_t>(4)));
auto stream_default_a = make_resource<CudaStreamPool>(
    "stream_default_a", Arg("stream_priority", 0));
auto stream_default_b = make_resource<CudaStreamPool>(
    "stream_default_b", Arg("stream_priority", 0));
// Pass each pool to its operator via Arg("cuda_stream_pool", ...) at construction.

// 2. One thread pool per branch, one thread each, pinned to distinct cores.
//    The priority branch uses real-time SCHED_FIFO at priority 80.
auto pool_priority = make_thread_pool("pool_priority", 1);
pool_priority->add_realtime(ae,
                            SchedulingPolicy::kFirstInFirstOut,
                            /*pin_operator=*/true,
                            /*pin_cores=*/{2},
                            /*sched_priority=*/80);

auto pool_a = make_thread_pool("pool_a", 1);
pool_a->add(procA, /*pin_operator=*/true, /*pin_cores=*/{3});

auto pool_b = make_thread_pool("pool_b", 1);
pool_b->add(procB, /*pin_operator=*/true, /*pin_cores=*/{4});

// 3. EventBasedScheduler with a small default pool as a safety net for any
//    operator not explicitly assigned (e.g. the source), pinned off the RT cores.
scheduler(make_scheduler<EventBasedScheduler>(
    "ebs",
    Arg("worker_thread_number", static_cast<int64_t>(2)),
    Arg("pin_cores", std::vector<uint32_t>{5, 6}),
    Arg("enable_queue_stealing", true),
    Arg("log_perf_stats", true)));
```
````
````{tab-item} Python
```python
# Pipeline shape (operators, add_flow): see public/examples/multi_branch_pipeline.
# The block below is the scheduling configuration added inside compose() on top
# of that shape. It assumes operators named `ae` (priority branch), `procA`,
# `procB` already exist.

# 1. Per-branch CUDA stream pools. Priority branch gets the highest
#    (most negative) priority; others use the default (0).
#    Query the device range with cudaDeviceGetStreamPriorityRange (via cuda /
#    cupy / ctypes); for example, high_pri = -5.
high_pri = -5
stream_priority  = CudaStreamPool(self, name="stream_priority",
                                  stream_priority=high_pri,
                                  reserved_size=1, max_size=4)
stream_default_a = CudaStreamPool(self, name="stream_default_a",
                                  stream_priority=0)
stream_default_b = CudaStreamPool(self, name="stream_default_b",
                                  stream_priority=0)
# Pass each pool to its operator via cuda_stream_pool=... at construction.

# 2. One thread pool per branch, one thread each, pinned to distinct cores.
#    The priority branch uses real-time SCHED_FIFO at priority 80.
pool_priority = self.make_thread_pool("pool_priority", initial_size=1)
pool_priority.add_realtime(ae, SchedulingPolicy.SCHED_FIFO,
                           pin_operator=True, pin_cores=[2],
                           sched_priority=80)

pool_a = self.make_thread_pool("pool_a", initial_size=1)
pool_a.add(procA, pin_operator=True, pin_cores=[3])

pool_b = self.make_thread_pool("pool_b", initial_size=1)
pool_b.add(procB, pin_operator=True, pin_cores=[4])

# 3. EventBasedScheduler with a small default pool as a safety net for any
#    operator not explicitly assigned (e.g. the source), pinned off the RT cores.
self.scheduler(EventBasedScheduler(
    self, name="ebs",
    worker_thread_number=2,
    pin_cores=[5, 6],
    enable_queue_stealing=True,
    log_perf_stats=True,
))
```
````
`````

Pin the dispatcher itself on a dedicated RT core by exporting environment variables before launching the application:

```bash
export GXF_EBS_DISPATCHER_CPU_CORE=1
export GXF_EBS_DISPATCHER_SCHED_POLICY=SCHED_FIFO
export GXF_EBS_DISPATCHER_SCHED_PRIORITY=99
```

The result: cores `1` (dispatcher), `2` (priority branch), `3`/`4` (other branches), `5`/`6` (default pool / unassigned operators). The priority branch's CPU thread runs `SCHED_FIFO` at priority 80, the dispatcher runs `SCHED_FIFO` at priority 99, and the priority branch's GPU work executes on a high-priority CUDA stream.

:::{note}
`cudaDeviceGetStreamPriorityRange()` is a CUDA runtime call. Holoscan does not wrap it; query the range from your application code and pass the resulting integer to `CudaStreamPool`'s `stream_priority` argument. Lower (more negative) values are higher priority.
:::

(scheduler-pitfalls)=

## Pitfalls

:::{warning}
**`strict_job_thread_pinning` does nothing on `EventBasedScheduler`.** It is a `MultiThreadScheduler` argument. `EventBasedScheduler` always pins strictly. Setting it on EBS is silently ignored — remove it from your `Arg` list to avoid confusion.
:::

:::{warning}
**`worker_thread_number` alone does not isolate branches.** It only sizes the default thread pool. Without `make_thread_pool` and `add(..., pin_operator=true, pin_cores={...})`, any worker can pick up any operator, so a heavy operator on one branch can stall the latency-critical operator on another. Branch isolation requires one user-defined thread pool per branch with explicit core pinning.
:::

:::{warning}
**`pin_cores` on the scheduler only affects the default pool.** It does not propagate to user-defined thread pools created with `make_thread_pool`. Set `pin_cores` per pool via the `add()` / `add_realtime()` arguments.
:::

:::{warning}
**Async buffer connectors require the consumer rate to be at least the producer rate.** Setting an input port to `IOSpec::ConnectorType::kAsyncBuffer` (C++) / `IOSpec.ConnectorType.ASYNC_BUFFER` (Python) gives a "latest frame wins" semantic — the consumer always reads the most recent value the producer published. If the consumer is *slower* than the producer, frames are dropped silently; if *faster*, the same frame is read multiple times. Use it only when stale-frame loss is acceptable and the consumer is guaranteed to be at least as fast as the producer. See [ping_simple_async_buffer](https://github.com/nvidia-holoscan/holoscan-sdk/tree/main/examples/ping_simple_async_buffer) and [ping_periodic_async_buffer](https://github.com/nvidia-holoscan/holoscan-sdk/tree/main/examples/ping_periodic_async_buffer) for reference shapes.
:::

:::{warning}
**Real-time policies need host and container privilege.** `add_realtime` and `GXF_EBS_DISPATCHER_SCHED_POLICY` set Linux RT scheduling policies (`SCHED_FIFO`, `SCHED_RR`, `SCHED_DEADLINE`). Without sufficient privilege, the policy is rejected at thread start and the thread silently falls back to `SCHED_OTHER` — defeating the purpose of the configuration. See {ref}`rt-scheduling-prerequisites` below.
:::

:::{tip}
**Recommended RT priorities: worker `80`, dispatcher `99`.** For the priority-branch worker thread set `sched_priority=80`; for the EBS dispatcher set `GXF_EBS_DISPATCHER_SCHED_PRIORITY=99`. **Why:** the dispatcher must always preempt workers so a newly-ready operator is never blocked behind a still-running compute, and leaving a 19-point gap keeps room above the worker for other system RT threads (e.g. kernel IRQ threads, `kthreadd` helpers) without inverting priority against the dispatcher. Adjust if your platform already uses RT priorities in the 80–99 band; the absolute numbers matter less than the ordering `dispatcher > worker > everything else`.
:::

:::{tip}
For sub-millisecond, fixed-cadence pipelines where the entire graph runs on the GPU, the EBS-on-RT-cores pattern can still be too coarse. Consider a GPU-resident graph instead, which lets the scheduler step the graph from device code and removes per-tick CPU dispatch overhead. See the [gpu_resident_dag](https://github.com/nvidia-holoscan/holoscan-sdk/tree/main/examples/gpu_resident_dag) and [gpu_resident_example](https://github.com/nvidia-holoscan/holoscan-sdk/tree/main/examples/gpu_resident_example) examples.
:::

(rt-scheduling-prerequisites)=

## Real-Time Scheduling Prerequisites

Real-time policies are set on the host kernel; the SDK only forwards the request. The host and the container must permit RT priorities for the call to take effect.

**Host kernel:** Linux throttles RT tasks by default. Disable the throttle (or raise it) to allow `SCHED_FIFO` / `SCHED_RR` threads to run without bound:

```bash
sudo sysctl -w kernel.sched_rt_runtime_us=-1
```

**Container:** Pass an RT priority limit when starting the container so the kernel will accept RT priority requests from inside it:

```bash
docker run --ulimit rtprio=99 ...
```

Without these, `add_realtime(...)` calls and `GXF_EBS_DISPATCHER_SCHED_POLICY=SCHED_FIFO` requests will be silently downgraded by the kernel.

:::{seealso}
RT priority alone does not prevent the kernel from scheduling unrelated work onto pinned cores. For full isolation — `isolcpus`, `nohz_full`, `rcu_nocbs`, and the `performance` CPU governor — see {ref}`host-cpu-isolation`.
:::

(operator-granularity-scheduling-overhead)=

## Operator Granularity and Scheduling Overhead

When designing Holoscan applications, it's important to understand the relationship between operator granularity and scheduling overhead. For operators with trivial computations, the scheduling and message-passing overhead can outweigh the actual computation time.

For detailed measurements, benchmarks, and guidance on when to split or combine operators, see {ref}`performance_considerations`.
