/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef PYHOLOSCAN_SCHEDULERS_GREEDY_SCHEDULER_PYDOC_HPP
#define PYHOLOSCAN_SCHEDULERS_GREEDY_SCHEDULER_PYDOC_HPP

#include <string>

#include "../macros.hpp"

namespace holoscan::doc {

namespace GreedyScheduler {

PYDOC(GreedyScheduler, R"doc(
Greedy scheduler

Parameters
----------
fragment : Fragment
    The fragment the condition will be associated with
clock : holoscan.resources.Clock or None, optional
    The clock used by the scheduler to define the flow of time. If None, a default-constructed
    `holoscan.resources.RealtimeClock` will be used.
stop_on_deadlock : bool, optional
    If enabled the scheduler will stop when all entities are in a waiting state, but no periodic
    entity exists to break the dead end. Should be disabled when scheduling conditions can be
    changed by external actors, for example by clearing queues manually.
max_duration_ms : int, optional
    The maximum duration for which the scheduler will execute (in ms). If not specified (or if a
    negative value is provided), the scheduler will run until all work is done. If periodic terms
    are present, this means the application will run indefinitely.
check_recession_period_ms : float, optional
    The maximum duration for which the scheduler would wait (in ms) when all operators are
    not ready to run in the current iteration.
stop_on_deadlock_timeout : int, optional
    The scheduler will wait this amount of time before determining that it is in deadlock
    and should stop. It will reset if a job comes in during the wait. A negative value means not
    stop on deadlock. This parameter only applies when `stop_on_deadlock=true`",
network_connection_timeout : int, optional
    During the initial phase when network connections are being established, this longer timeout
    (in ms) is used instead of stop_on_deadlock_timeout. This allows sufficient time for UCX
    connections to be established without triggering false deadlock detection. "This parameter has
    no effect on single fragment (non-distributed) applications. "Defaults to 5000 ms (5 seconds).
name : str, optional
    The name of the scheduler.
)doc")

}  // namespace GreedyScheduler

}  // namespace holoscan::doc

#endif  // PYHOLOSCAN_SCHEDULERS_GREEDY_SCHEDULER_PYDOC_HPP
