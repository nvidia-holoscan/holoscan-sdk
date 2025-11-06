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

#ifndef HOLOSCAN_OPERATORS_FORMAT_CONVERTER_FORMAT_CONVERTER_HPP
#define HOLOSCAN_OPERATORS_FORMAT_CONVERTER_FORMAT_CONVERTER_HPP

#include <npp.h>

#include <memory>
#include <string>
#include <vector>

#include "holoscan/core/io_context.hpp"
#include "holoscan/core/io_spec.hpp"
#include "holoscan/core/operator.hpp"
#include "holoscan/core/operator_spec.hpp"
#include "holoscan/core/resources/gxf/allocator.hpp"
#include "holoscan/core/resources/gxf/cuda_stream_pool.hpp"

namespace holoscan::ops {

enum class FormatDType {
  kUnknown,
  kRGB888,
  kRGBA8888,
  kUnsigned8,
  kFloat32,
  kYUV420,
  kNV12BT709HDTV,
  kNV12BT709CSC,
  kNV12BT601Full,
  kYUYV,
  kRGB161616,
  kRGBA16161616,
  kUnsigned16
};

enum class FormatConversionType {
  kUnknown,
  kNone,
  kUnsigned8ToFloat32,
  kFloat32ToUnsigned8,
  kRGB888ToRGBA8888,
  kRGBA8888ToRGB888,
  kRGBA8888ToFloat32,
  kRGB888ToYUV420,
  kYUV420ToRGBA8888,
  kYUV420ToRGB888,
  kNV12BT709HDTVToRGB888,
  kNV12BT709CSCToRGB888,
  kNV12BT601FullToRGB888,
  kYUYVToRGB888,
  kRGB161616ToRGB888,
  kRGBA16161616ToRGB888,
};

/**
 * @brief Operator class to convert the data format of the input data.
 *
 * ==Named Inputs==
 *
 * - **source_video** : `nvidia::gxf::Tensor` or `nvidia::gxf::VideoBuffer`
 *   - The input video frame to process. If the input is a VideoBuffer it must be in format
 *     GXF_VIDEO_FORMAT_RGBA, GXF_VIDEO_FORMAT_RGB or GXF_VIDEO_FORMAT_NV12,
 *     GXF_VIDEO_FORMAT_NV12_ER, GXF_VIDEO_FORMAT_NV12_709 or GXF_VIDEO_FORMAT_NV12_709_ER. If
 *     a video buffer is not found, the input port message is searched for a tensor with the name
 *     specified by `in_tensor_name`. This must be a tensor in one of several supported formats
 *     (unsigned 8-bit int or float32 graycale, unsigned 8-bit int RGB or RGBA YUV420,
 *     NV12BT601Full, NV12BT709CSC, NV12BT709HDTV, or YUYV). The tensor or video buffer may be in
 *     either host or device memory (a host->device copy is performed if needed).
 *
 * ==Named Outputs==
 *
 * - **tensor** : `nvidia::gxf::Tensor`
 *   - The output video frame after processing. The shape, data type and number of channels
 *     of this output tensor will depend on the specific parameters that were set for this
 *     operator. The name of the Tensor transmitted on this port is determined by
 *     `out_tensor_name`.
 *
 * ==Parameters==
 *
 * - **pool**: Memory pool allocator (holoscan::Allocator) used by the operator.
 * - **out_dtype**: Destination data type. The available options are:
 *   - `"rgb888"`
 *   - `"uint8"`
 *   - `"float32"`
 *   - `"rgba8888"`
 *   - `"yuv420"`
 * - **in_dtype**: Source data type. The available options are:
 *   - `"rgb888"`
 *   - `"uint8"`
 *   - `"float32"`
 *   - `"rgba8888"`
 *   - `"yuv420"`
 *   - `"nv12"`            (alias for nv12_bt709_hdtv)
 *   - `"nv12_bt601_full"` (BT.601 full range as in NPP's NV12ToRGB functions and IPP's YUVToRGB)
 *   - `"nv12_bt709_hdtv"` (BT.709 recommendation for high-definition TV (HDTV))
 *   - `"nv12_bt709_csc"`  (BT.709 recommendation for computer systems consideration (CSC))
 *   - `"yuyv"`
 *   - `"rgb161616"`
 *   - `"rgba16161616"`
 *   Optional (default: `"rgb888"`).
 * - **in_tensor_name**: The name of the input tensor (or VideoBuffer). Optional (default: `""`).
 * - **out_tensor_name**: The name of the output tensor. Optional (default: `""`).
 * - **scale_min**: Output will be clipped to this minimum value. Optional (default: `0.0`).
 * - **scale_max**: Output will be clipped to this maximum value. Optional (default: `1.0`).
 * - **alpha_value**: Unsigned integer in range [0, 255], indicating the alpha channel value to use
 *   when converting from RGB to RGBA. Optional (default: `255`).
 * - **resize_height**: Desired height for the (resized) output. Height will be unchanged if
 *   `resize_height` is `0`. Optional (default: `0`).
 * - **resize_width**: Desired width for the (resized) output. Width will be unchanged if
 *   `resize_width` is `0`. Optional (default: `0`).
 * - **resize_mode**: Resize mode enum value corresponding to NPP's NppiInterpolationMode.
 *   Values available at:
 *   https://docs.nvidia.com/cuda/npp/nppdefs.html?highlight=Two%20parameter%20cubic%20filter#c.NppiInterpolationMode
 *   - NPPI_INTER_UNDEFINED (`0`): Undefined filtering interpolation mode.
 *   - NPPI_INTER_NN (`1`): Nearest neighbor filtering.
 *   - NPPI_INTER_LINEAR (`2`): Linear interpolation.
 *   - NPPI_INTER_CUBIC (`4`): Cubic interpolation.
 *   - NPPI_INTER_CUBIC2P_BSPLINE (`5`): Two-parameter cubic filter (B=1, C=0)
 *   - NPPI_INTER_CUBIC2P_CATMULLROM (`6`): Two-parameter cubic filter (B=0, C=1/2)
 *   - NPPI_INTER_CUBIC2P_B05C03 (`7`): Two-parameter cubic filter (B=1/2, C=3/10)
 *   - NPPI_INTER_SUPER (`8`): Super sampling.
 *   - NPPI_INTER_LANCZOS (`16`): Lanczos filtering.
 *   - NPPI_INTER_LANCZOS3_ADVANCED (`17`): Generic Lanczos filtering with order 3.
 *   - NPPI_SMOOTH_EDGE (`0x8000000`): Smooth edge filtering.
 *
 *   Optional (default: `0`). The default value `0` (NPPI_INTER_UNDEFINED) which would be
 *   equivalent to `4` (NPPI_INTER_CUBIC).
 * - **out_channel_order**: Sequence of integers describing how channel values are permuted.
 *   Optional (default: `[0, 1, 2]` for 3-channel images and `[0, 1, 2, 3]` for 4-channel images).
 * - **cuda_stream_pool**: `holoscan::CudaStreamPool` instance to allocate CUDA streams.
 *   Optional (default: `nullptr`).
 *
 * ==Device Memory Requirements==
 *
 * When using this operator with a `BlockMemoryPool`, between 1 and 3 device memory blocks
 * (`storage_type` = 1) will be required based on the input tensors and parameters:
 *   - 1.) In all cases there is a memory block needed for the output tensor. The size of this
 *     block will be `out_height * out_width * out_channels * out_element_size_bytes` where
 *     (out_height, out_width) will either be (in_height, in_width) (or
 *     (resize_height, resize_width) a resize was specified). `out_element_size` is the element
 *     size in bytes (e.g. 1 for RGB888 or 4 for Float32).
 *   - 2.) If a resize is being done, another memory block is required for this. This block will
 *     have size `resize_height * resize_width * in_channels * in_element_size_bytes`.
 *   - 3.) If the input tensor will be in host memory, a memory block is needed to copy the input
 *     to the device. This block will have size
 *     `in_height * in_width * in_channels * in_element_size_bytes`.
 *
 * Thus when declaring the memory pool, `num_blocks` should be between 1-3 and `block_size` should
 * be set to the maximum of the individual blocks sizes described above.
 */
class FormatConverterOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(FormatConverterOp)

  FormatConverterOp() = default;

  // TODO(gbae): use std::expected
  void setup(OperatorSpec& spec) override;
  void initialize() override;
  void start() override;
  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override;
  void stop() override;

  nvidia::gxf::Expected<void*> resizeImage(
      const void* in_tensor_data, const std::vector<nvidia::gxf::ColorPlane>& in_color_planes,
      const int32_t rows, const int32_t columns, const int16_t channels,
      const nvidia::gxf::PrimitiveType primitive_type, const int32_t resize_width,
      const int32_t resize_height);
  void convertTensorFormat(const void* in_tensor_data,
                           const std::vector<nvidia::gxf::ColorPlane>& in_color_planes,
                           void* out_tensor_data, const int32_t rows, const int32_t columns,
                           const int16_t out_channels);

 private:
  Parameter<holoscan::IOSpec*> in_;
  Parameter<holoscan::IOSpec*> out_;

  Parameter<std::string> in_tensor_name_;
  Parameter<std::string> out_tensor_name_;
  Parameter<float> scale_min_;
  Parameter<float> scale_max_;
  Parameter<uint8_t> alpha_value_;
  Parameter<int32_t> resize_width_;
  Parameter<int32_t> resize_height_;
  Parameter<int32_t> resize_mode_;
  Parameter<std::vector<int>> out_channel_order_;
  Parameter<std::shared_ptr<CudaStreamPool>> cuda_stream_pool_{};

  std::unique_ptr<nvidia::gxf::MemoryBuffer> resize_buffer_;
  std::unique_ptr<nvidia::gxf::MemoryBuffer> channel_buffer_;
  std::unique_ptr<nvidia::gxf::MemoryBuffer> device_scratch_buffer_;

  Parameter<std::shared_ptr<Allocator>> pool_;

  Parameter<std::string> in_dtype_str_;
  Parameter<std::string> out_dtype_str_;

  // internal state
  FormatDType in_dtype_ = FormatDType::kUnknown;
  FormatDType out_dtype_ = FormatDType::kUnknown;
  nvidia::gxf::PrimitiveType in_primitive_type_ = nvidia::gxf::PrimitiveType::kCustom;
  nvidia::gxf::PrimitiveType out_primitive_type_ = nvidia::gxf::PrimitiveType::kCustom;
  FormatConversionType format_conversion_type_ = FormatConversionType::kUnknown;

  NppStreamContext npp_stream_ctx_{};
};

}  // namespace holoscan::ops

#endif /* HOLOSCAN_OPERATORS_FORMAT_CONVERTER_FORMAT_CONVERTER_HPP */
