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

#ifndef HOLOSCAN_OPERATORS_BAYER_DEMOSAIC_PYDOC_HPP
#define HOLOSCAN_OPERATORS_BAYER_DEMOSAIC_PYDOC_HPP

#include <string>

#include "../../macros.hpp"

namespace holoscan::doc::BayerDemosaicOp {

// Constructor
PYDOC(BayerDemosaicOp, R"doc(
Bayer Demosaic operator.
)doc")

// PyBayerDemosaicOp Constructor
PYDOC(BayerDemosaicOp_python, R"doc(
Bayer Demosaic operator.

Named inputs:
    receiver: nvidia::gxf::Tensor or nvidia::gxf::VideoBuffer
        The input video frame to process. If the input is a VideoBuffer it must be an 8-bit
        unsigned grayscale video (nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_GRAY). The video
        buffer may be in either host or device memory (a host->device copy is performed if needed).
        If a video buffer is not found, the input port message is searched for a tensor with the
        name specified by `in_tensor_name`. This must be a device tensor in either 8-bit or 16-bit
        unsigned integer format.

Named outputs:
    transmitter: nvidia::gxf::Tensor
        The output video frame after demosaicing. This will be a 3-channel RGB image if
        `alpha_value` is ``True``, otherwise it will be a 4-channel RGBA image. The data type will
        be either 8-bit or 16-bit unsigned integer (matching the bit depth of the input). The
        name of the tensor that is output is controlled by `out_tensor_name`.

Parameters
----------
fragment : holoscan.core.Fragment
    The fragment that the operator belongs to.
pool : holoscan.resources.Allocator
    Memory pool allocator used by the operator.
cuda_stream_pool : holoscan.resources.CudaStreamPool, optional
    `holoscan.resources.CudaStreamPool` instance to allocate CUDA streams.
in_tensor_name : str, optional
    The name of the input tensor.
out_tensor_name : str, optional
    The name of the output tensor.
interpolation_mode : int, optional
    The interpolation model to be used for demosaicing. Values available at:
    https://docs.nvidia.com/cuda/npp/group__typedefs__npp.html#ga2b58ebd329141d560aa4367f1708f191
bayer_grid_pos : int, optional
    The Bayer grid position (default of 2 = GBRG). Values available at:
    https://docs.nvidia.com/cuda/npp/group__typedefs__npp.html#ga5597309d6766fb2dffe155990d915ecb
generate_alpha : bool, optional
    Generate alpha channel.
alpha_value : int, optional
    Alpha value to be generated if `generate_alpha` is set to ``True``.
name : str, optional
    The name of the operator.
)doc")

PYDOC(gxf_typename, R"doc(
The GXF type name of the resource.

Returns
-------
str
    The GXF type name of the resource
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

}  // namespace holoscan::doc::BayerDemosaicOp

#endif /* HOLOSCAN_OPERATORS_BAYER_DEMOSAIC_PYDOC_HPP */
