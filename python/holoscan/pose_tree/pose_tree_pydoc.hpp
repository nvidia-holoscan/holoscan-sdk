/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef PYHOLOSCAN_POSE_TREE_POSE_TREE_PYDOC_HPP
#define PYHOLOSCAN_POSE_TREE_POSE_TREE_PYDOC_HPP

#include <string>

#include "../macros.hpp"

namespace holoscan::doc {

namespace PoseTreeManager {

PYDOC(PoseTreeManager_args_kwargs, R"doc(
Create a PoseTreeManager resource.

Parameters
----------
fragment : holoscan.core.Fragment
    The fragment that the resource belongs to.
port : int, optional
    Port for the UCX server to listen on in a distributed setup. Defaults to 13337.
number_frames : int, optional
    Maximum number of coordinate frames. Defaults to 1024.
number_edges : int, optional
    Maximum number of edges (direct transformations). Defaults to 16384.
history_length : int, optional
    Total capacity for historical pose data. Defaults to 1048576.
default_number_edges : int, optional
    Default edges allocated per new frame. Defaults to 16.
default_history_length : int, optional
    Default history capacity per new edge. Defaults to 1024.
edges_chunk_size : int, optional
    Allocation chunk size for a frame's edge list. Defaults to 4.
history_chunk_size : int, optional
    Allocation chunk size for an edge's history buffer. Defaults to 64.
request_timeout_ms : int, optional
    UCX client request timeout in milliseconds. Defaults to 5000.
request_poll_sleep_us : int, optional
    UCX client polling sleep interval in microseconds. Defaults to 10.
worker_progress_sleep_us : int, optional
    UCX progress loop sleep interval in microseconds. Defaults to 100.
server_shutdown_timeout_ms : int, optional
    UCX server shutdown timeout in milliseconds. Defaults to 1000.
server_shutdown_poll_sleep_ms : int, optional
    UCX server shutdown polling interval in milliseconds. Defaults to 10.
maximum_clients : int, optional
    Maximum number of UCX clients. Defaults to 1024.
name : str, optional
    The name of the resource.
)doc")

PYDOC(tree, R"doc(
Get the managed PoseTree instance.
)doc")

}  // namespace PoseTreeManager

}  // namespace holoscan::doc

#endif /* PYHOLOSCAN_POSE_TREE_POSE_TREE_PYDOC_HPP */
