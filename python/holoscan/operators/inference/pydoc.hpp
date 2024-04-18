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

#ifndef HOLOSCAN_OPERATORS_INFERENCE_PYDOC_HPP
#define HOLOSCAN_OPERATORS_INFERENCE_PYDOC_HPP

#include <string>

#include "../../macros.hpp"

namespace holoscan::doc::InferenceOp {

// PyInferenceOp Constructor
PYDOC(InferenceOp, R"doc(
Inference operator.

**==Named Inputs==**

    receivers : multi-receiver accepting nvidia::gxf::Tensor(s)
        Any number of upstream ports may be connected to this ``receivers`` port. The operator will
        search across all messages for tensors matching those specified in ``in_tensor_names``.
        These are the set of input tensors used by the models in ``inference_map``.

**==Named Outputs==**

    transmitter : nvidia::gxf::Tensor(s)
        A message containing tensors corresponding to the inference results from all models will
        be emitted. The names of the tensors transmitted correspond to those in
        ``out_tensor_names``.


For more details on ``InferenceOp`` parameters, see
[Customizing the Inference Operator](https://docs.nvidia.com/holoscan/sdk-user-guide/examples/byom.html#customizing-the-inference-operator)
or refer to [Inference](https://docs.nvidia.com/holoscan/sdk-user-guide/inference.html).

Parameters
----------
fragment : holoscan.core.Fragment (constructor only)
    The fragment that the operator belongs to.
backend : {"trt", "onnxrt", "torch"}
    Backend to use for inference. Set ``"trt"`` for TensorRT, ``"torch"`` for LibTorch and
    ``"onnxrt"`` for the ONNX runtime.
allocator : holoscan.resources.Allocator
    Memory allocator to use for the output.
inference_map : holoscan.operators.InferenceOp.DataVecMap
    Tensor to model map.
model_path_map : holoscan.operators.InferenceOp.DataMap
    Path to the ONNX model to be loaded.
pre_processor_map : holoscan.operators.InferenceOp::DataVecMap
    Pre processed data to model map.
device_map : holoscan.operators.InferenceOp.DataMap, optional
    Mapping of model to GPU ID for inference.
backend_map : holoscan.operators.InferenceOp.DataMap, optional
    Mapping of model to backend type for inference. Backend options: ``"trt"`` or ``"torch"``
in_tensor_names : sequence of str, optional
    Input tensors.
out_tensor_names : sequence of str, optional
    Output tensors.
infer_on_cpu : bool, optional
    Whether to run the computation on the CPU instead of GPU. Default value is ``False``.
parallel_inference : bool, optional
    Whether to enable parallel execution. Default value is ``True``.
input_on_cuda : bool, optional
    Whether the input buffer is on the GPU. Default value is ``True``.
output_on_cuda : bool, optional
    Whether the output buffer is on the GPU. Default value is ``True``.
transmit_on_cuda : bool, optional
    Whether to transmit the message on the GPU. Default value is ``True``.
enable_fp16 : bool, optional
    Use 16-bit floating point computations. Default value is ``False``.
is_engine_path : bool, optional
    Whether the input model path mapping is for trt engine files. Default value is ``False``.
cuda_stream_pool : holoscan.resources.CudaStreamPool, optional
    ``holoscan.resources.CudaStreamPool`` instance to allocate CUDA streams. Default value is
    ``None``.
name : str, optional (constructor only)
    The name of the operator. Default value is ``"inference"``.
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

}  // namespace holoscan::doc::InferenceOp

#endif /* HOLOSCAN_OPERATORS_INFERENCE_PYDOC_HPP */
