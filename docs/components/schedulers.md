# Schedulers

The Scheduler component is a critical part of the system responsible for governing the execution of operators in a graph by enforcing conditions associated with each operator. Its primary responsibility includes orchestrating the execution of all operators defined in the graph while keeping track of their execution states.

The Holoscan SDK offers multiple schedulers that can cater to various use cases. These schedulers are:

1. [Greedy Scheduler](#greedy-scheduler): This basic single-threaded scheduler tests conditions in a greedy manner. It is suitable for simple use cases and provides predictable execution. However, it may not be ideal for large-scale applications as it may incur significant overhead in condition execution.
2. [MultiThread Scheduler](#multithread-scheduler): The multithread scheduler is designed to handle complex execution patterns in large-scale applications. This scheduler consists of a dispatcher thread that monitors the status of each operator and dispatches it to a thread pool of worker threads responsible for executing them. Once execution is complete, worker threads enqueue the operator back on the dispatch queue. The multithread scheduler offers superior performance and scalability over the greedy scheduler.
3. [Event-Based Scheduler](#event-based-scheduler): The event-based scheduler is also a multi-thread scheduler, but as the name indicates it is event-based rather than polling based. Instead of having a thread that constantly polls for the execution readiness of each operator, it instead waits for an event to be received which indicates that an operator is ready to execute. The event-based scheduler will have a lower latency than using the multi-thread scheduler with a long polling interval (`check_recession_period_ms`), but without the high CPU usage seen for a multi-thread scheduler with a very short polling interval. In general, this is an improvement over the older `MultiThreadScheduler` and provides additional features such as CPU thread pinning and options to enable Linux real-time scheduling.

It is essential to select the appropriate scheduler for the use case at hand to ensure optimal performance and efficient resource utilization. Since most parameters of the schedulers overlap, it is easy to switch between them to test which may be most performant for a given application.

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

(operator-granularity-scheduling-overhead)=

## Operator Granularity and Scheduling Overhead

When designing Holoscan applications, it's important to understand the relationship between operator granularity and scheduling overhead. For operators with trivial computations, the scheduling and message-passing overhead can outweigh the actual computation time.

For detailed measurements, benchmarks, and guidance on when to split or combine operators, see {ref}`performance_considerations`.
