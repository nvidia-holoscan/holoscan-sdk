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

#ifndef PYHOLOSCAN_OPERATORS_SEGMENTATION_POSTPROCESSOR_PYDOC_HPP
#define PYHOLOSCAN_OPERATORS_SEGMENTATION_POSTPROCESSOR_PYDOC_HPP

#include <string>

#include "../../macros.hpp"

namespace holoscan::doc::SegmentationPostprocessorOp {

// PySegmentationPostprocessorOp Constructor
PYDOC(SegmentationPostprocessorOp, R"doc(
Operator carrying out post-processing operations on segmentation outputs.

**==Named Inputs==**

    in_tensor : nvidia::gxf::Tensor
        Expects a message containing a 32-bit floating point device tensor with name
        ``in_tensor_name``. The expected data layout of this tensor is HWC, NCHW or NHWC format as
        specified via ``data_format``.

**==Named Outputs==**

    out_tensor : nvidia::gxf::Tensor
        Emits a message containing a device tensor named "out_tensor" that contains the segmentation
        labels. This tensor will have unsigned 8-bit integer data type and shape (H, W, 1).

**==Device Memory Requirements==**

    When used with a ``holoscan.resources.BlockMemoryPool``, this operator requires only a single
    device memory block (``storage_type=1``) of size ``height * width`` bytes.

Parameters
----------
fragment : holoscan.core.Fragment (constructor only)
    The fragment that the operator belongs to.
allocator : holoscan.resources.Allocator
    Memory allocator to use for the output.
in_tensor_name : str, optional
    Name of the input tensor. Default value is ``""``.
network_output_type : str, optional
    Network output type (e.g. 'softmax'). Default value is ``"softmax"``.
data_format : str, optional
    Data format of network output. Default value is ``"hwc"``.
cuda_stream_pool : holoscan.resources.CudaStreamPool, optional
    ``holoscan.resources.CudaStreamPool`` instance to allocate CUDA streams.
    Default value is ``None``.
name : str, optional (constructor only)
    The name of the operator. Default value is ``"segmentation_postprocessor"``.
)doc")

}  // namespace holoscan::doc::SegmentationPostprocessorOp

#endif /* PYHOLOSCAN_OPERATORS_SEGMENTATION_POSTPROCESSOR_PYDOC_HPP */
