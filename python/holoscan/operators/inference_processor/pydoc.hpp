/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef PYHOLOSCAN_OPERATORS_INFERENCE_PROCESSOR_PYDOC_HPP
#define PYHOLOSCAN_OPERATORS_INFERENCE_PROCESSOR_PYDOC_HPP

#include <string>

#include "../../macros.hpp"

namespace holoscan::doc::InferenceProcessorOp {

// PyInferenceProcessorOp Constructor
PYDOC(InferenceProcessorOp, R"doc(
Holoinfer Processing operator.

**==Named Inputs==**

    receivers : multi-receiver accepting nvidia::gxf::Tensor(s)
        Any number of upstream ports may be connected to this ``receivers`` port. The operator will
        search across all messages for tensors matching those specified in ``in_tensor_names``.
        These are the set of input tensors used by the processing operations specified in
        ``process_map``.

**==Named Outputs==**

    transmitter : nvidia::gxf::Tensor(s)
        A message containing tensors corresponding to the processed results from operations will
        be emitted. The names of the tensors transmitted correspond to those in
        ``out_tensor_names``.

**==Device Memory Requirements==**

    When using this operator with a ``holoscan.resources.BlockMemoryPool``, ``num_blocks`` must be
    greater than or equal to the number of output tensors that will be produced. The ``block_size``
    in bytes must be greater than or equal to the largest output tensor (in bytes). If
    ``output_on_cuda`` is ``True``, the blocks should be in device memory (``storage_type=1``),
    otherwise they should be CUDA pinned host memory (``storage_type=0``).

Parameters
----------
fragment : holoscan.core.Fragment (constructor only)
    The fragment that the operator belongs to.
allocator : holoscan.resources.Allocator
    Memory allocator to use for the output.
process_operations : holoscan.operators.InferenceProcessorOp.DataVecMap
    Operations in sequence on tensors.
processed_map : holoscan.operators.InferenceProcessorOp::DataVecMap
    Input-output tensor mapping.
in_tensor_names : sequence of str, optional
    Names of input tensors in the order to be fed into the operator.
out_tensor_names : sequence of str, optional
    Names of output tensors in the order to be fed into the operator.
input_on_cuda : bool, optional
    Whether the input buffer is on the GPU. Default value is ``False``.
output_on_cuda : bool, optional
    Whether the output buffer is on the GPU. Default value is ``False``.
transmit_on_cuda : bool, optional
    Whether to transmit the message on the GPU. Default value is ``False``.
cuda_stream_pool : holoscan.resources.CudaStreamPool, optional
    ``holoscan.resources.CudaStreamPool`` instance to allocate CUDA streams.
    Default value is ``None``.
config_path : str, optional
    File path to the config file. Default value is ``""``.
disable_transmitter : bool, optional
    If ``True``, disable the transmitter output port of the operator.
    Default value is ``False``.
name : str, optional (constructor only)
    The name of the operator. Default value is ``"postprocessor"``.
)doc")

}  // namespace holoscan::doc::InferenceProcessorOp

#endif /* PYHOLOSCAN_OPERATORS_INFERENCE_PROCESSOR_PYDOC_HPP */
