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
fragment : holoscan.core.Fragment or holoscan.core.Subgraph
    The fragment (or subgraph) the condition will be associated with
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
receiver : str, optional
    The name of the operator's input port to which the condition would apply.
name : str, optional
    The name of the condition.

Notes
-----
This condition is typically set within the `Operator.setup` method using the `IOSpec.condition`
method with `ConditionType.EXPIRING_MESSAGE_AVAILABLE`. In that case, the receiver name is already
known from the port corresponding to the `IOSpec` object, so the "receiver" argument is unnecessary.

The `max_delay_ns` used by this condition type is relative to the timestamp of the oldest message
in the receiver queue. Use of this condition requires that the upstream operator emitted a
timestamp for at least one message in the queue. Holoscan Operators do not emit a timestamp
by default, but only when it is explicitly requested in the `Operator::emit` call. The built-in
operators of the SDK do not currently emit a timestamp, so this condition cannot be easily used
with the provided operators. As a potential alternative, please see
`MultiMessageAvailableTimeoutCondition` which can be configured to use a single port and a timeout
interval without needing a timestamp. A timestamp is not needed in the case of
`MultiMessageAvailableTimeoutCondition` because the interval measured is the time since the same
operator previously ticked.
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

}  // namespace ExpiringMessageAvailableCondition

}  // namespace holoscan::doc

#endif /* PYHOLOSCAN_CONDITIONS_MESSAGE_AVAILABLE_PYDOC_HPP */
