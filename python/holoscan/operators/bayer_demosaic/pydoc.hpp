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

#ifndef PYHOLOSCAN_OPERATORS_BAYER_DEMOSAIC_PYDOC_HPP
#define PYHOLOSCAN_OPERATORS_BAYER_DEMOSAIC_PYDOC_HPP

#include <string>

#include "../../macros.hpp"

namespace holoscan::doc::BayerDemosaicOp {

// PyBayerDemosaicOp Constructor
PYDOC(BayerDemosaicOp, R"doc(
Bayer Demosaic operator.

**==Named Inputs==**

    receiver : nvidia::gxf::Tensor or nvidia::gxf::VideoBuffer
        The input video frame to process. If the input is a VideoBuffer it must be an 8-bit
        unsigned grayscale video (`nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_GRAY`). If a video
        buffer is not found, the input port message is searched for a device
        tensor with the name specified by `in_tensor_name`. The tensor must have either 8-bit or
        16-bit unsigned integer format. The tensor or video buffer may be in either host or device
        memory (a host->device copy is performed if needed).

**==Named Outputs==**

    transmitter : nvidia::gxf::Tensor
        The output video frame after demosaicing. This will be a 3-channel RGB image if
        ``alpha_value`` is ``True``, otherwise it will be a 4-channel RGBA image. The data type will
        be either 8-bit or 16-bit unsigned integer (matching the bit depth of the input). The
        name of the tensor that is output is controlled by ``out_tensor_name``.

**==Device Memory Requirements==**

    When using this operator with a ``holoscan.resources.BlockMemoryPool``, the minimum
    ``block_size`` is ``(rows * columns * output_channels * element_size_bytes)`` where
    ``output_channels`` is 4 when ``generate_alpha`` is ``True`` and 3 otherwise. If the input
    tensor or video buffer is already on the device, only a single memory block is needed. However,
    if the input is on the host, a second memory block will also be needed in order to make an
    internal copy of the input to the device. The memory buffer must be on device
    (``storage_type=1``).

Parameters
----------
fragment : holoscan.core.Fragment (constructor only)
    The fragment that the operator belongs to.
pool : holoscan.resources.Allocator
    Memory pool allocator used by the operator.
cuda_stream_pool : holoscan.resources.CudaStreamPool, optional
    ``holoscan.resources.CudaStreamPool`` instance to allocate CUDA streams. Default value is ``None``.
in_tensor_name : str, optional
    The name of the input tensor. Default value is ``""`` (empty string).
out_tensor_name : str, optional
    The name of the output tensor. Default value is ``""`` (empty string).
interpolation_mode : int, optional
    The interpolation model to be used for demosaicing. Values available at:
    https://docs.nvidia.com/cuda/npp/nppdefs.html?highlight=Two%20parameter%20cubic%20filter#c.NppiInterpolationMode

    - NPPI_INTER_UNDEFINED (``0``): Undefined filtering interpolation mode.
    - NPPI_INTER_NN (``1``): Nearest neighbor filtering.
    - NPPI_INTER_LINEAR (``2``): Linear interpolation.
    - NPPI_INTER_CUBIC (``4``): Cubic interpolation.
    - NPPI_INTER_CUBIC2P_BSPLINE (``5``): Two-parameter cubic filter (B=1, C=0)
    - NPPI_INTER_CUBIC2P_CATMULLROM (``6``): Two-parameter cubic filter (B=0, C=1/2)
    - NPPI_INTER_CUBIC2P_B05C03 (``7``): Two-parameter cubic filter (B=1/2, C=3/10)
    - NPPI_INTER_SUPER (``8``): Super sampling.
    - NPPI_INTER_LANCZOS (``16``): Lanczos filtering.
    - NPPI_INTER_LANCZOS3_ADVANCED (``17``): Generic Lanczos filtering with order 3.
    - NPPI_SMOOTH_EDGE (``0x8000000``): Smooth edge filtering.

    Default value is ``0`` (NPPI_INTER_UNDEFINED).
bayer_grid_pos : int, optional
    The Bayer grid position. Values available at:
    https://docs.nvidia.com/cuda/npp/nppdefs.html?highlight=Two%20parameter%20cubic%20filter#c.NppiBayerGridPosition

    - NPPI_BAYER_BGGR (``0``): Default registration position BGGR.
    - NPPI_BAYER_RGGB (``1``): Registration position RGGB.
    - NPPI_BAYER_GBRG (``2``): Registration position GBRG.
    - NPPI_BAYER_GRBG (``3``): Registration position GRBG.

    Default value is ``2`` (NPPI_BAYER_GBRG).
generate_alpha : bool, optional
    Generate alpha channel. Default value is ``False``.
alpha_value : int, optional
    Alpha value to be generated if ``generate_alpha`` is set to ``True``. Default value is ``255``.
name : str, optional (constructor only)
    The name of the operator. Default value is ``"bayer_demosaic"``.
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

#endif /* PYHOLOSCAN_OPERATORS_BAYER_DEMOSAIC_PYDOC_HPP */
