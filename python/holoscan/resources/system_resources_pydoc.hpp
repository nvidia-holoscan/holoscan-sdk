/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
