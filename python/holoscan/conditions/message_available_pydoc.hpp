/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef PYHOLOSCAN_CONDITIONS_MESSAGE_AVAILABLE_PYDOC_HPP
#define PYHOLOSCAN_CONDITIONS_MESSAGE_AVAILABLE_PYDOC_HPP

#include <string>

#include "../macros.hpp"

namespace holoscan::doc {

namespace MessageAvailableCondition {

PYDOC(MessageAvailableCondition, R"doc(
Condition that permits execution when an upstream message is available.

Satisfied when the associated receiver queue has at least a certain number of
elements. The receiver is specified using the receiver parameter of the
scheduling term. The minimum number of messages that permits the execution of
the entity is specified by `min_size`. An optional parameter for this
scheduling term is `front_stage_max_size`, the maximum front stage message
count. If this parameter is set, the scheduling term will only allow execution
if the number of messages in the queue does not exceed this count. It can be
used for operators which do not consume all messages from the queue.

Parameters
----------
fragment : holoscan.core.Fragment or holoscan.core.Subgraph
    The fragment (or subgraph) the condition will be associated with
min_size : int
    The total number of messages over a set of input channels needed
    to permit execution.
front_stage_max_size : int
    Threshold for the number of front stage messages. Execution is only
    allowed if the number of front stage messages does not exceed this
    count.
receiver : str, optional
    The name of the operator's input port to which the condition would apply.
name : str, optional
    The name of the condition.

Notes
-----
This condition is typically set within the `Operator.setup` method using the `IOSpec.condition`
method with `ConditionType.MESSAGE_AVAILABLE`. In that case, the receiver name is already known
from the port corresponding to the `IOSpec` object, so the "receiver" argument is unnecessary.
)doc")

PYDOC(receiver, R"doc(
The receiver associated with the condition.
)doc")

PYDOC(min_size, R"doc(
The total number of messages over a set of input channels needed
to permit execution.
)doc")

PYDOC(front_stage_max_size, R"doc(
Threshold for the number of front stage messages. Execution is only allowed
if the number of front stage messages does not exceed this count.
)doc")

}  // namespace MessageAvailableCondition

}  // namespace holoscan::doc

#endif /* PYHOLOSCAN_CONDITIONS_MESSAGE_AVAILABLE_PYDOC_HPP */
