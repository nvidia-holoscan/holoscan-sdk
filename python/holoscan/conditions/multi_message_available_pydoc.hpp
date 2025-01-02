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

#ifndef PYHOLOSCAN_CONDITIONS_MULTI_MESSAGE_AVAILABLE_PYDOC_HPP
#define PYHOLOSCAN_CONDITIONS_MULTI_MESSAGE_AVAILABLE_PYDOC_HPP

#include <string>

#include "../macros.hpp"

namespace holoscan::doc {

namespace MultiMessageAvailableCondition {

PYDOC(MultiMessageAvailableCondition, R"doc(
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
sampling_mode : {"SumOfAll", "PerReceiver"} or MultiMessageAvailableCondition.SamplingMode, optional
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
`ConditionType.MULTI_MESSAGE_AVAILABLE`. The "receivers" argument must be set based on the input
port names as described in the "Parameters" section.
)doc")

PYDOC(receivers, R"doc(
The receivers associated with the condition.
)doc")

PYDOC(sampling_mode, R"doc(
The sampling mode for the condition. This parameter determines how the minimum number of messages
is calculated. The two possible values are:
``MultiMessageAvailableCondition.SamplingMode.SUM_OF_ALL`` and
``MultiMessageAvailableCondition.SamplingMode.PER_RECEIVER.``)doc")

PYDOC(min_sizes, R"doc(
Get the minimum number of messages that permits the execution of the entity. There is one value per
receiver associated with this condition. This parameter is only used when `sampling_mode` is set
to ``MultiMessageAvailableCondition.SamplingMode.PER_RECEIVER``; otherwise, it is ignored.
)doc")

PYDOC(add_min_size, R"doc(
Append an integer value to the min_sizes vector.

Parameters
----------
value : int
    The value to append to the min_sizes vector.
)doc")

PYDOC(min_sum, R"doc(
The total number of messages that permits the execution of the entity. This total is over all
receivers associated with this condition. This parameter is only used when `sampling_mode` is set
to ``MultiMessageAvailableCondition.SamplingMode.SUM_OF_ALL``; otherwise, it is ignored.
)doc")

}  // namespace MultiMessageAvailableCondition

}  // namespace holoscan::doc

#endif /* PYHOLOSCAN_CONDITIONS_MULTI_MESSAGE_AVAILABLE_PYDOC_HPP */
