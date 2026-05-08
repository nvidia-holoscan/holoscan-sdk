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

#ifndef PYHOLOSCAN_CONDITIONS_SUBSCRIBER_AVAILABLE_PYDOC_HPP
#define PYHOLOSCAN_CONDITIONS_SUBSCRIBER_AVAILABLE_PYDOC_HPP

#include <string>

#include "../macros.hpp"

namespace holoscan::doc {

namespace SubscriberAvailableCondition {

PYDOC(SubscriberAvailableCondition, R"doc(
Native condition that waits until a transmitter has enough matched pub/sub subscribers.

Parameters
----------
fragment : holoscan.core.Fragment or holoscan.core.Subgraph
    The fragment (or subgraph) the condition will be associated with.
min_subscriber_count : int, optional
    Minimum number of matched subscribers required before the condition becomes ready.
transmitter : str, optional
    The name of the operator output port to monitor.
require_pubsub_connector : bool, optional
    If ``True``, the transmitter must resolve to a pub/sub transmitter.
poll_period_ms : int, optional
    Polling period in milliseconds while waiting for subscribers to match.
stabilization_ms : int, optional
    Additional stabilization delay in milliseconds after the match threshold is reached.
latch_ready : bool, optional
    If ``True``, remain ready after the first successful match.
ready_on_shutdown : bool, optional
    Reserved policy flag for future shutdown behavior.
name : str, optional
    The name of the condition.
)doc")

}  // namespace SubscriberAvailableCondition

}  // namespace holoscan::doc

#endif  // PYHOLOSCAN_CONDITIONS_SUBSCRIBER_AVAILABLE_PYDOC_HPP
