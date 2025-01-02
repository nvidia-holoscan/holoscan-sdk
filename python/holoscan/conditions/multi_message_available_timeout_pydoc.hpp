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

#ifndef PYHOLOSCAN_CONDITIONS_MULTI_MESSAGE_AVAILABLE_TIMEOUT_PYDOC_HPP
#define PYHOLOSCAN_CONDITIONS_MULTI_MESSAGE_AVAILABLE_TIMEOUT_PYDOC_HPP

#include <string>

#include "../macros.hpp"

namespace holoscan::doc {

namespace MultiMessageAvailableTimeoutCondition {

PYDOC(MultiMessageAvailableTimeoutCondition, R"doc(
Condition that checks the number of messages available across multiple inputs.

This condition is used to check if a sufficient number of messages are available across multiple
input ports. It can operator in one of two modes:

    1. ``SUM_OF_ALL``: The condition checks if the sum of messages available across all input ports
      is greater than or equal to a given threshold. For this mode, `min_sum` should be specified.
    2. ``PER_RECEIVER``: The condition checks if the number of messages available at each input
      port is greater than or equal to a given threshold. For this mode, `min_sizes` should be
      specified.

Parameters
----------
fragment : holoscan.core.Fragment
    The fragment the condition will be associated with
execution_frequency : std::string
    The 'execution frequency' indicates the amount of time after which the entity will be allowed
    to execute again, even if the specified number of messages have not yet been received. The
    period is specified as a string containing  of a number and an (optional) unit. If no unit is
    given the value is assumed to be in nanoseconds. Supported units are: Hz, s, ms.
    Examples: "10ms", "10000000", "0.2s", "50Hz".
sampling_mode : {"SumOfAll", "PerReceiver"} or MultiMessageAvailableTimeoutCondition.SamplingMode, optional
    The sampling method to use when checking for messages in receiver queues.
min_sum : int, optional
    The condition permits execution if the sum of message counts of all receivers has at
    least the given number of messages available. This option is only intended for use with
    "SumOfAll" `sampling_mode`.
min_sizes : list of int, optional
    The condition permits execution if all given receivers have at least the given number of
    messages available in this list. This option is only intended for use with
    "PerReceiver" `sampling_mode`. The length of `min_sizes` must match the
    number of receivers associated with the condition.
name : str, optional
    The name of the condition.

Notes
-----
This condition is typically set via the `Operator.multi_port_condition` method using
`ConditionType.MULTI_MESSAGE_AVAILABLE_TIMEOUT`. The "receivers" argument must be set
based on the input port names as described in the "Parameters" section.

This condition can also be used on a single port as a way to have a message-available condition
that also supports a timeout interval. For this single input port use case, the condition can be
added within `Operator.setup` using the `IOSpec.condition` method with condition type
`ConditionType.MULTI_MESSAGE_AVAILABLE_TIMEOUT`. In this case, the input port is already known
from the `IOSpec` object, so the "receivers" argument is unnecessary.

)doc")

PYDOC(receivers, R"doc(
The receivers associated with the condition.
)doc")

}  // namespace MultiMessageAvailableTimeoutCondition

}  // namespace holoscan::doc

#endif /* PYHOLOSCAN_CONDITIONS_MULTI_MESSAGE_AVAILABLE_TIMEOUT_PYDOC_HPP */
