/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef PYHOLOSCAN_SCHEDULERS_EVENT_BASED_SCHEDULER_PYDOC_HPP
#define PYHOLOSCAN_SCHEDULERS_EVENT_BASED_SCHEDULER_PYDOC_HPP

#include <string>

#include "../macros.hpp"

namespace holoscan::doc {

namespace EventBasedScheduler {

PYDOC(EventBasedScheduler, R"doc(
Event-based multi-thread scheduler

Parameters
----------
fragment : Fragment
    The fragment the condition will be associated with
clock : holoscan.resources.Clock or None, optional
    The clock used by the scheduler to define the flow of time. If None, a default-constructed
    `holoscan.resources.RealtimeClock` will be used.
worker_thread_number : int
    The number of worker threads. This creates a default thread pool. Operators not explicitly
    assigned to a user-defined thread pool (via `make_thread_pool`) will use this default pool.
stop_on_deadlock : bool, optional
    If enabled the scheduler will stop when all entities are in a waiting state, but no periodic
    entity exists to break the dead end. Should be disabled when scheduling conditions can be
    changed by external actors, for example by clearing queues manually.
max_duration_ms : int, optional
    The maximum duration for which the scheduler will execute (in ms). If not specified (or if a
    negative value is provided), the scheduler will run until all work is done. If periodic terms
    are present, this means the application will run indefinitely.
stop_on_deadlock_timeout : int, optional
    The scheduler will wait this amount of time before determining that it is in deadlock
    and should stop. It will reset if a job comes in during the wait. A negative value means not
    stop on deadlock. This parameter only applies when `stop_on_deadlock=true`.
network_connection_timeout : int, optional
    During the initial phase when network connections are being established, this longer timeout
    (in ms) is used instead of stop_on_deadlock_timeout. This allows sufficient time for UCX
    connections to be established without triggering false deadlock detection. "This parameter has
    no effect on single fragment (non-distributed) applications. "Defaults to 5000 ms (5 seconds).
pin_cores : list of int, optional
    CPU core IDs to pin the default thread pool's worker threads to. If specified, all the worker
    threads in the default pool (created based on `worker_thread_number`) will be pinned to the
    same set of specified cores. Note: This only affects the default pool; to control CPU affinity
    for user-defined thread pools, use the pin_cores parameter in ThreadPool.add(). If not
    specified, the default pool's worker threads will not be pinned to any cores.
enable_queue_stealing : bool, optional
    If true, default worker threads attempt to steal ready jobs from other default worker queues
    before blocking on their own queue. Default is False.
steal_scan_limit : int, optional
    Maximum number of victim queues scanned per steal attempt (0 means scan all queues).
    Default is 0.
enable_worker_postcheck_fastpath : bool, optional
    If true, workers perform a fresh checkEntity() after executeEntity() and directly update
    READY/WAIT_TIME conditions without routing that entity through the dispatcher. Default is False.
postcheck_fallback_notify_interval : int, optional
    When worker postcheck returns a non-ready state, send a periodic dispatcher wake-up every N
    fallbacks per worker. Set to 0 to only notify when no workers are running. Default is 256.
postcheck_fallback_notify_min_workers : int, optional
    Periodic fallback notify is enabled only when worker_thread_number is at least this value.
    Default is 8.
postcheck_fallback_notify_min_period_ns : int, optional
    Minimum global time spacing (in nanoseconds) between periodic fallback dispatcher wake-ups.
    Default is 100000.
internal_event_shard_count : int, optional
    Number of internal notification shards used by the dispatcher (0 = auto =
    worker_thread_number). Default is 0.
dispatcher_internal_pop_batch_size : int, optional
    Maximum number of internal notifications drained from one shard per dispatcher pop step.
    Default is 32.
wait_state_shard_count : int, optional
    Number of shards used for WAIT_EVENT and WAIT tracking lists. Default is 1.
log_perf_stats : bool, optional
    If true, logs scheduler instrumentation counters during deinitialize(). Default is False.
name : str, optional
    The name of the scheduler.
)doc")

}  // namespace EventBasedScheduler
}  // namespace holoscan::doc

#endif /* PYHOLOSCAN_SCHEDULERS_EVENT_BASED_SCHEDULER_PYDOC_HPP */
