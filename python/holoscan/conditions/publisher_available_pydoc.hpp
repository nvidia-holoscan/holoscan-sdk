/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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
