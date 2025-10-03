/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef PYHOLOSCAN_OPERATORS_FORMAT_CONVERTER_PYDOC_HPP
#define PYHOLOSCAN_OPERATORS_FORMAT_CONVERTER_PYDOC_HPP

#include <string>

#include "../../macros.hpp"

namespace holoscan::doc::FormatConverterOp {

// PyFormatConverterOp Constructor
PYDOC(FormatConverterOp, R"doc(
Format conversion operator.

**==Named Inputs==**

    source_video : nvidia::gxf::Tensor or nvidia::gxf::VideoBuffer
        The input video frame to process. If the input is a VideoBuffer it must be in format
        GXF_VIDEO_FORMAT_RGBA, GXF_VIDEO_FORMAT_RGB or GXF_VIDEO_FORMAT_NV12,
        GXF_VIDEO_FORMAT_NV12_ER, GXF_VIDEO_FORMAT_NV12_709 or GXF_VIDEO_FORMAT_NV12_709_ER. If
        a video buffer is not found, the input port message is searched for a tensor with the name
        specified by `in_tensor_name`. This must be a tensor in one of several supported formats
        (unsigned 8-bit int or float32 graycale, unsigned 8-bit int RGB or RGBA YUV420,
        NV12BT601Full, NV12BT709CSC, NV12BT709HDTV, or YUYV). The tensor or video buffer may be in
        either host or device memory (a host->device copy is performed if needed).


**==Named Outputs==**

    tensor : nvidia::gxf::Tensor
        The output video frame after processing. The shape, data type and number of channels of this
        output tensor will depend on the specific parameters that were set for this operator. The
        name of the Tensor transmitted on this port is determined by ``out_tensor_name``.

**==Device Memory Requirements==**

    When using this operator with a ``holoscan.resources.BlockMemoryPool``, between 1 and 3 device
    memory blocks (``storage_type=1``) will be required based on the input tensors and parameters:

        - 1.) In all cases there is a memory block needed for the output tensor. The size of this
            block will be ``out_height * out_width * out_channels * out_element_size_bytes`` where
            ``(out_height, out_width)`` will either be ``(in_height, in_width)`` (or
            ``(resize_height, resize_width)`` a resize was specified). `out_element_size` is the
            element size in bytes (e.g. 1 for RGB888 or 4 for Float32).
        - 2.) If a resize is being done, another memory block is required for this. This block will
            have size ``resize_height * resize_width * in_channels * in_element_size_bytes``.
        - 3.) If the input tensor will be in host memory, a memory block is needed to copy the input
            to the device. This block will have size
            ``in_height * in_width * in_channels * in_element_size_bytes``.

    Thus when declaring the memory pool, `num_blocks` will be between 1 and 3 and `block_size`
    must be greater or equal to the largest of the individual blocks sizes described above.

Parameters
----------
fragment : holoscan.core.Fragment (constructor only)
    The fragment that the operator belongs to.
pool : holoscan.resources.Allocator
    Memory pool allocator used by the operator.
out_dtype : str
    Destination data type. The available options are:

    - ``"rgb888"``
    - ``"uint8"``
    - ``"float32"``
    - ``"rgba8888"``
    - ``"yuv420"``
in_dtype : str, optional
    Source data type. The available options are:

    - ``"rgb888"``
    - ``"uint8"``
    - ``"float32"``
    - ``"rgba8888"``
    - ``"yuv420"``
    - `"nv12"`            (alias for `"nv12_bt709_hdtv"`)
    - `"nv12_bt601_full"` (BT.601 full range as in NPP's NV12ToRGB functions and IPP's YUVToRGB)
    - `"nv12_bt709_hdtv"` (BT.709 recommendation for high-definition TV (HDTV))
    - `"nv12_bt709_csc"`  (BT.709 recommendation for computer systems consideration (CSC))
in_tensor_name : str, optional
    The name of the input tensor. Default value is ``""`` (empty string).
out_tensor_name : str, optional
    The name of the output tensor. Default value is ``""`` (empty string).
scale_min : float, optional
    Output will be clipped to this minimum value. Default value is ``0.0``.
scale_max : float, optional
    Output will be clipped to this maximum value. Default value is ``1.0``.
alpha_value : int, optional
    Unsigned integer in range [0, 255], indicating the alpha channel value to use
    when converting from RGB to RGBA. Default value is ``255``.
resize_height : int, optional
    Desired height for the (resized) output. Height will be unchanged if ``resize_height`` is ``0``.
    Default value is ``0``.
resize_width : int, optional
    Desired width for the (resized) output. Width will be unchanged if ``resize_width`` is ``0``.
    Default value is ``0``.
resize_mode : int, optional
    Resize mode enum value corresponding to NPP's NppiInterpolationMode.
    Values available at:
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

    Default value is ``0`` (NPPI_INTER_UNDEFINED) which would be equivalent to ``4``
    (NPPI_INTER_CUBIC).
out_channel_order : sequence of int
    Sequence of integers describing how channel values are permuted.
    Default value is ``[0, 1, 2]`` for 3-channel images and ``[0, 1, 2, 3]`` for 4-channel images.
cuda_stream_pool : holoscan.resources.CudaStreamPool, optional
    `holoscan.resources.CudaStreamPool` instance to allocate CUDA streams.
    Default value is ``None``.
name : str, optional (constructor only)
    The name of the operator. Default value is ``"format_converter"``.
)doc")

}  // namespace holoscan::doc::FormatConverterOp

#endif /* PYHOLOSCAN_OPERATORS_FORMAT_CONVERTER_PYDOC_HPP */
