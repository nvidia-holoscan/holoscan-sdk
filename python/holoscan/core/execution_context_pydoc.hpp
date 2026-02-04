/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef PYHOLOSCAN_CORE_EXECUTION_CONTEXT_PYDOC_HPP
#define PYHOLOSCAN_CORE_EXECUTION_CONTEXT_PYDOC_HPP

#include <string>

#include "../macros.hpp"

namespace holoscan::doc {

namespace ExecutionContext {

//  Constructor
PYDOC(ExecutionContext, R"doc(
Class representing an execution context.
)doc")

PYDOC(allocate_cuda_stream, R"doc(
Allocate a CUDA stream from the operator's CudaStreamPool.

Streams are cached by name — calling with the same name returns the same stream on subsequent
calls within the same operator. This is useful for root operators that need to allocate a
stream (rather than receiving one from upstream).

Streams allocated this way are **not** automatically emitted on output ports. Call
``OutputContext.set_cuda_stream()`` before ``emit()`` if you need to propagate the stream
to downstream operators.

Parameters
----------
name : str, optional
    A name for the stream. The same name returns the same stream on subsequent calls.
    If omitted, defaults to an empty string, so repeated calls without a name argument
    will return the same stream.

Returns
-------
stream_ptr : int or None
    The memory address of the allocated cudaStream_t. Returns ``None`` if no CudaStreamPool
    is available on the operator.
)doc")

PYDOC(synchronize_streams, R"doc(
Synchronize multiple CUDA streams to a target stream (non-blocking).

Uses ``cudaEventRecord`` and ``cudaStreamWaitEvent`` to create GPU-side dependencies without
blocking the CPU. This is the same mechanism used internally by ``receive_cuda_stream``.

When using ``receive_cuda_stream``, synchronization is handled automatically and this method
is not needed. It is provided for advanced manual stream handling use cases.

Parameters
----------
cuda_stream_ptrs : list[int or None]
    A list of memory addresses of the CUDA streams to synchronize. Any ``None`` elements are
    skipped.
target_stream_ptr : int
    The memory address of the target CUDA stream that will wait for all other streams.
)doc")

PYDOC(device_from_stream, R"doc(
Get the CUDA device ID for a given stream.

Only works with Holoscan-managed streams (those returned by ``receive_cuda_stream``,
``receive_cuda_streams``, or ``allocate_cuda_stream``).

Parameters
----------
cuda_stream_ptr : int
    The memory address of the CUDA stream to query.

Returns
-------
device_id : int or None
    The device ID if the stream is managed by Holoscan, otherwise ``None``.

)doc")

PYDOC(find_operator, R"doc(
Find an operator by name.

If the operator name is not provided, the current operator is returned.

Parameters
----------
op_name : str, optional
    The name of the operator to find. If not provided, returns the current operator.

Returns
-------
operator : Operator or None
    A shared pointer to the operator, or None if the operator is not found.
)doc")

PYDOC(get_operator_status, R"doc(
Get the status of the operator.

If the operator name is not provided, the status of the current operator is returned.

Parameters
----------
op_name : str, optional
    The name of the operator to check status for. If not provided, checks the current operator.

Returns
-------
status : OperatorStatus
    The status of the operator.

Raises
------
RuntimeError
    If the operator is not found or another error occurs.
)doc")
}  // namespace ExecutionContext

}  // namespace holoscan::doc

#endif  // PYHOLOSCAN_CORE_EXECUTION_CONTEXT_PYDOC_HPP
