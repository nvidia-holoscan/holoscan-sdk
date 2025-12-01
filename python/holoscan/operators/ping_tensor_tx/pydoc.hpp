/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef PYHOLOSCAN_OPERATORS_PING_TENSOR_TX_PYDOC_HPP
#define PYHOLOSCAN_OPERATORS_PING_TENSOR_TX_PYDOC_HPP

#include <string>

#include "../../macros.hpp"

namespace holoscan::doc::PingTensorTxOp {

// PingTensorTxOp Constructor
PYDOC(PingTensorTxOp, R"doc(
Tensor generation operator intended for use in tests and examples.

The contents of the generated tensor are uninitialized.

**==Named Outputs==**

    output : nvidia::gxf::Tensor
        A message containing a single tensor with the a specified shape, storage type,
        data type and name.

Parameters
----------
fragment : holoscan.core.Fragment or holoscan.core.Subgraph (constructor only)
    The fragment that the operator belongs to.
allocator : holoscan.resources.Allocator, optional
    The allocator used to allocate the tensor output. If unspecified a
    ``holoscan.resources.UnboundedAllocator`` is used.
storage_type : {"host", "device", "system"}, optional
    The memory storage type for the generated tensor. Here, `"system"` corresponds to CPU memory
    while `"host"` corresponds to pinned host memory allocated using CUDA's `cudaMallocHost`.
    Finally, `"device"` corresponds to GPU device memory allocated via `cudaMalloc`.
batch_size : int or None, optional
    Size of the batch dimension (default: 0). The tensor shape will be
    ([batch], rows, [columns], [channels]) where [] around a dimension indicates that it is only
    present if the corresponding parameter has a size > 0. If 0 or ``None``, no batch dimension
    will be present.
rows : int, optional
    The number of rows in the generated tensor.
cols : int, optional
    The number of columns in the generated tensor. If 0 or ``None``, no columns dimension will be
    present.
channels : int, optional
    The number of channels in the generated tensor. If 0 or ``None``, no channels dimension will be
    present.
data_type : str or numpy.dtype, optional
    The data type used by the tensor. Should be a string matching one of the following C++ types
    {"int8_t", "int16_t", "int32_t", "int64_t", "uint8_t", "uint16_t"," "uint32_t", "uint64_t",
     "float", "double", "complex<float>", "complex<double>"}. Alternatively, a ``numpy.dtype``
    object can be provided to indicate the desired data type.
tensor_name : str, optional
    The name of the output tensor.
cuda_stream_pool : holoscan.resources.CudaStreamPool, optional
    CUDA stream pool to use for memory allocation when `async_device_allocation` is ``True``.
async_device_allocation : bool, optional
    If True, allocate memory using cudaMallocAsync on an internally allocated CUDA stream.
    Otherwise, use cudaMalloc on the default stream.
name : str, optional (constructor only)
    The name of the operator. Default value is ``"ping_tensor_tx"``.

Notes
-----
When ``async_device_allocation`` is enabled, this operator allocates device memory asynchronously
on a CUDA stream. The ``compute`` method may return before all GPU work has completed. Downstream
operators that receive data from this operator should call
``op_input.receive_cuda_stream(<port_name>)`` to synchronize the CUDA stream with the downstream
operator's dedicated internal stream. This ensures proper synchronization before accessing the
data. For more details on CUDA stream handling in Holoscan, see:
https://docs.nvidia.com/holoscan/sdk-user-guide/holoscan_cuda_stream_handling.html
)doc")

}  // namespace holoscan::doc::PingTensorTxOp

#endif /* PYHOLOSCAN_OPERATORS_PING_TENSOR_TX_PYDOC_HPP */
