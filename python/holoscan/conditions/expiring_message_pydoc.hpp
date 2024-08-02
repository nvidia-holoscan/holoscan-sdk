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

#ifndef PYHOLOSCAN_CONDITIONS_EXPIRINGMESSAGE_AVAILABLE_PYDOC_HPP
#define PYHOLOSCAN_CONDITIONS_EXPIRINGMESSAGE_AVAILABLE_PYDOC_HPP

#include <string>

#include "../macros.hpp"

namespace holoscan::doc {

namespace ExpiringMessageAvailableCondition {

PYDOC(ExpiringMessageAvailableCondition, R"doc(
Condition that tries to wait for specified number of messages in receiver.
When the first message in the queue mature after specified delay since arrival it would fire
regardless.

Parameters
----------
fragment : holoscan.core.Fragment
    The fragment the condition will be associated with
max_batch_size : int
    The maximum number of messages to be batched together.
max_delay_ns: int or datetime.timedelta
    The maximum delay to wait from the time of the first message before submitting the workload
    anyway. If an int is provided, the value must be in nanoseconds. Any provided
    `datetime.timedelta` value will be converted internally to the corresponding number of
    nanoseconds to wait.
clock : holoscan.resources.Clock or None, optional
    The clock used by the scheduler to define the flow of time. If None, a default-constructed
    `holoscan.resources.RealtimeClock` will be used.
name : str, optional
    The name of the condition.
)doc")

PYDOC(gxf_typename, R"doc(
The GXF type name of the condition.

Returns
-------
str
    The GXF type name of the condition
)doc")

PYDOC(setup, R"doc(
Define the component specification.

Parameters
----------
spec : holoscan.core.ComponentSpec
    Component specification associated with the condition.
)doc")

PYDOC(receiver, R"doc(
The receiver associated with the condition.
)doc")

PYDOC(max_batch_size, R"doc(
The maximum number of messages to be batched together.
)doc")

PYDOC(max_delay, R"doc(
The maximum delay from first message to wait before submitting workload anyway.
)doc")

PYDOC(max_delay_ns, R"doc(
The maximum delay from first message to wait before submitting workload anyway.
)doc")

PYDOC(initialize, R"doc(
Initialize the condition

This method is called only once when the condition is created for the first
time, and uses a light-weight initialization.
)doc")

}  // namespace ExpiringMessageAvailableCondition

}  // namespace holoscan::doc

#endif /* PYHOLOSCAN_CONDITIONS_MESSAGE_AVAILABLE_PYDOC_HPP */
