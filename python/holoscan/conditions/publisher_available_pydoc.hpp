/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef PYHOLOSCAN_CONDITIONS_PUBLISHER_AVAILABLE_PYDOC_HPP
#define PYHOLOSCAN_CONDITIONS_PUBLISHER_AVAILABLE_PYDOC_HPP

#include <string>

#include "../macros.hpp"

namespace holoscan::doc {

namespace PublisherAvailableCondition {

PYDOC(PublisherAvailableCondition, R"doc(
Native condition that waits until a receiver has enough matched pub/sub publishers.

Parameters
----------
fragment : holoscan.core.Fragment or holoscan.core.Subgraph
    The fragment (or subgraph) the condition will be associated with.
min_publisher_count : int, optional
    Minimum number of matched publishers required before the condition becomes ready.
receiver : str, optional
    The name of the operator input port to monitor.
require_pubsub_connector : bool, optional
    If ``True``, the receiver must resolve to a pub/sub receiver.
poll_period_ms : int, optional
    Polling period in milliseconds while waiting for publishers to match.
latch_ready : bool, optional
    If ``True``, remain ready after the first successful match.
name : str, optional
    The name of the condition.
)doc")

}  // namespace PublisherAvailableCondition

}  // namespace holoscan::doc

#endif  // PYHOLOSCAN_CONDITIONS_PUBLISHER_AVAILABLE_PYDOC_HPP
