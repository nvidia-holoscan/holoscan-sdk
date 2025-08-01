/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef PYHOLOSCAN_RESOURCES_SYSTEM_RESOURCES_HPP
#define PYHOLOSCAN_RESOURCES_SYSTEM_RESOURCES_HPP

#include <string>

#include "../macros.hpp"

namespace holoscan::doc {

namespace ThreadPool {

PYDOC(ThreadPool_kwargs, R"doc(
ThreadPool for operators scheduled by EventBasedScheduler or MultiThreadScheduler.

Parameters
----------
initialize_size : int, optional
    The initial number of worker threads in the pool.
name : str, optional
    The name of the thread pool.
)doc")

PYDOC(add, R"doc(
Assign one or more operators to use the thread pool.

Parameters
----------
ops : Operator or list[Operator]
    The operator(s) to add to the thread pool.
pin_operator : bool, optional
    If True, the operator(s) will be pinned to a specific thread in the pool.
pin_cores : list[int], optional
    CPU core IDs to pin the worker thread to. Empty list means no core pinning. Default is empty list.
)doc")

PYDOC(add_realtime, R"doc(
Assign an operator to use the thread pool with real-time scheduling capabilities.

Parameters
----------
op : Operator
    The operator to add to the thread pool.
pin_operator : bool, optional
    If True, the operator will be pinned to a specific thread in the pool. Default is True.
pin_cores : list[int], optional
    CPU core IDs to pin the worker thread to. Empty list means no core pinning. Default is empty list.
sched_policy : SchedulingPolicy, optional
    Real-time scheduling policy. Default is SchedulingPolicy.UNSPECIFIED.
sched_priority : int, optional
    Thread priority for FirstInFirstOut and RoundRobin policies. Default is 0.
sched_runtime : int, optional
    Expected worst case execution time in nanoseconds for Deadline policy. Default is 0.
sched_deadline : int, optional
    Relative deadline in nanoseconds for Deadline policy. Default is 0.
sched_period : int, optional
    Period in nanoseconds for Deadline policy. Default is 0.
)doc")

PYDOC(operators, R"doc(
The operators associated with this thread pool.

Returns
----------
list[Operator]
    The list of operators that have been added to this thread pool.
)doc")

}  // namespace ThreadPool

}  // namespace holoscan::doc

#endif /* PYHOLOSCAN_RESOURCES_SYSTEM_RESOURCES_HPP */
