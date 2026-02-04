/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
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
name : str, optional
    The name of the scheduler.
)doc")

}  // namespace EventBasedScheduler
}  // namespace holoscan::doc

#endif /* PYHOLOSCAN_SCHEDULERS_EVENT_BASED_SCHEDULER_PYDOC_HPP */
