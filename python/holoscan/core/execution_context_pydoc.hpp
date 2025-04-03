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
Allocate an internal CUDA stream using the operator's associated CUDA thread pool.

Parameters
----------
name : str, optional
    The name of the CUDA stream to allocate.

Returns
-------
stream_ptr : int
    The memory address corresponding to the cudaStream_t that was created.
)doc")

PYDOC(synchronize_streams, R"doc(
Allocate an internal CUDA stream using the operator's associated CUDA thread pool.

Parameters
----------
cuda_stream_ptrs: list[int or None], optional
    A list of memory addresses of the CUDA streams to synchronize. Any None elements will be
    ignored.
target_stream_ptr: int
    The memory address of the target CUDA stream to synchronize to.
)doc")

PYDOC(device_from_stream, R"doc(
Determine the device ID corresponding to a given CUDA stream.

Parameters
----------
cuda_stream_ptr: int
    The memory address of the CUDA stream. This must be a Holoscan-managed stream for the device
    query to work.

Returns
-------
device_id : int or None
    If the CUDA stream is managed by Holoscan, the device ID corresponding to the stream is
    returned. Otherwise the output will be None.

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
