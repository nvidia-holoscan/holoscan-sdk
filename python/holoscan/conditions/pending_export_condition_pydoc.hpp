/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef PYHOLOSCAN_CONDITIONS_PENDING_EXPORT_CONDITION_PYDOC_HPP
#define PYHOLOSCAN_CONDITIONS_PENDING_EXPORT_CONDITION_PYDOC_HPP

#include <string>

#include "../macros.hpp"

namespace holoscan::doc {

namespace PendingExportCondition {

PYDOC(PendingExportCondition, R"doc(
Native condition that blocks publisher execution while too many native-buffer
exports are in flight on a ``NativeBufferProtocolAdapter``.

This condition is intended for publisher-side gating with CUDA IPC / holoipc.
It queries the adapter's ``pending_export_count()`` and blocks when the count
reaches ``max_pending``.

Parameters
----------
fragment : holoscan.core.Fragment or holoscan.core.Subgraph
    The fragment (or subgraph) the condition will be associated with.
max_pending : int, optional
    Maximum number of pending exports allowed before blocking.  Must be >= 1.
network_context : holoscan.core.NetworkContext, optional
    A ``FastDdsPubSubNetworkContext`` whose native-buffer adapter will be used
    to resolve pending export counts.  If ``None``, the adapter must be set
    separately before the application runs.
name : str, optional
    The name of the condition.
)doc")

}  // namespace PendingExportCondition

}  // namespace holoscan::doc

#endif  // PYHOLOSCAN_CONDITIONS_PENDING_EXPORT_CONDITION_PYDOC_HPP
