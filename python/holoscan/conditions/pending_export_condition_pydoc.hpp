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
