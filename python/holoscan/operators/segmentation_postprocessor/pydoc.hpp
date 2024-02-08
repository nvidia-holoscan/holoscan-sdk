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

#ifndef HOLOSCAN_OPERATORS_SEGMENTATION_POSTPROCESSOR_PYDOC_HPP
#define HOLOSCAN_OPERATORS_SEGMENTATION_POSTPROCESSOR_PYDOC_HPP

#include <string>

#include "../../macros.hpp"

namespace holoscan::doc::SegmentationPostprocessorOp {

PYDOC(SegmentationPostprocessorOp, R"doc(
Operator carrying out post-processing operations on segmentation outputs.
)doc")

// PySegmentationPostprocessorOp Constructor
PYDOC(SegmentationPostprocessorOp_python, R"doc(
Operator carrying out post-processing operations on segmentation outputs.

Named inputs:
    in_tensor: nvidia::gxf::Tensor
        Expects a message containing a 32-bit floating point tensor with name `in_tensor_name`.
        The expected data layout of this tensor is HWC, NCHW or NHWC format as specified via
        `data_format`.

Named outputs:
    out_tensor: nvidia::gxf::Tensor
        Emits a message containing a tensor named "out_tensor" that contains the segmentation
        labels. This tensor will have unsigned 8-bit integer data type and shape (H, W, 1).

Parameters
----------
fragment : holoscan.core.Fragment
    The fragment that the operator belongs to.
allocator : holoscan.resources.Allocator
    Memory allocator to use for the output.
in_tensor_name : str, optional
    Name of the input tensor.
network_output_type : str, optional
    Network output type (e.g. 'softmax').
data_format : str, optional
    Data format of network output.
cuda_stream_pool : holoscan.resources.CudaStreamPool, optional
    `holoscan.resources.CudaStreamPool` instance to allocate CUDA streams.

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

}  // namespace holoscan::doc::SegmentationPostprocessorOp

#endif /* HOLOSCAN_OPERATORS_SEGMENTATION_POSTPROCESSOR_PYDOC_HPP */
