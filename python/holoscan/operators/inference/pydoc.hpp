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

#ifndef PYHOLOSCAN_OPERATORS_INFERENCE_PYDOC_HPP
#define PYHOLOSCAN_OPERATORS_INFERENCE_PYDOC_HPP

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

**==Device Memory Requirements==**

    When using this operator with a ``holoscan.resources.BlockMemoryPool``, ``num_blocks`` must be
    greater than or equal to the number of output tensors that will be produced. The ``block_size``
    in bytes must be greater than or equal to the largest output tensor (in bytes). If
    ``output_on_cuda`` is ``True``, the blocks should be in device memory (``storage_type=1``),
    otherwise they should be CUDA pinned host memory (``storage_type=0``).

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
inference_map : dict[str, List[str]]
    Tensor to model map.
model_path_map : dict[str, str]
    Path to the ONNX model to be loaded.
pre_processor_map : dict[str, List[str]]
    Pre processed data to model map.
device_map : dict[str, int], optional
    Mapping of model to GPU ID for inference.
temporal_map : dict[str, int], optional
    Mapping of model to frame delay for inference.
activation_map : dict[str, int], optional
    Mapping of model to activation state for inference.
backend_map : dict[str, str], optional
    Mapping of model to backend type for inference. Backend options: ``"trt"`` or ``"torch"``
in_tensor_names : sequence of str, optional
    Input tensors.
out_tensor_names : sequence of str, optional
    Output tensors.
trt_opt_profile : sequence of int, optional
    TensorRT optimization profile for models with dynamic inputs.
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
enable_cuda_graphs : bool, optional
    Use CUDA Graphs. Default value is ``True``.
is_engine_path : bool, optional
    Whether the input model path mapping is for trt engine files. Default value is ``False``.
cuda_stream_pool : holoscan.resources.CudaStreamPool, optional
    ``holoscan.resources.CudaStreamPool`` instance to allocate CUDA streams. Default value is
    ``None``.
name : str, optional (constructor only)
    The name of the operator. Default value is ``"inference"``.
)doc")

}  // namespace holoscan::doc::InferenceOp

#endif /* PYHOLOSCAN_OPERATORS_INFERENCE_PYDOC_HPP */
