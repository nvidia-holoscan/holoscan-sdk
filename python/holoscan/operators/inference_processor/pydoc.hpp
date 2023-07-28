/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef HOLOSCAN_OPERATORS_INFERENCE_PROCESSOR_PYDOC_HPP
#define HOLOSCAN_OPERATORS_INFERENCE_PROCESSOR_PYDOC_HPP

#include <string>

#include "../../macros.hpp"

namespace holoscan::doc::InferenceProcessorOp {

PYDOC(InferenceProcessorOp, R"doc(
Holoinfer Processing operator.
)doc")

// PyInferenceProcessorOp Constructor
PYDOC(InferenceProcessorOp_python, R"doc(
Holoinfer Processing operator.

Parameters
----------
fragment : holoscan.core.Fragment
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
    Whether the input buffer is on the GPU.
output_on_cuda : bool, optional
    Whether the output buffer is on the GPU.
transmit_on_cuda : bool, optional
    Whether to transmit the message on the GPU.
cuda_stream_pool : holoscan.resources.CudaStreamPool, optional
    CudaStreamPool instance to allocate CUDA streams.
config_path : str, optional
    File path to the config file.
disable_transmitter : bool, optional
    If ``True``, disable the transmitter output port of the operator.
name : str, optional
    The name of the operator.
)doc")

PYDOC(initialize, R"doc(
Initialize the operator.

This method is called only once when the operator is created for the first time,
and uses a light-weight initialization.
)doc")

PYDOC(setup, R"doc(
Define the operator specification.

Parameters
----------
spec : holoscan.core.OperatorSpec
    The operator specification.
)doc")

}  // namespace holoscan::doc::InferenceProcessorOp

#endif /* HOLOSCAN_OPERATORS_INFERENCE_PROCESSOR_PYDOC_HPP */
