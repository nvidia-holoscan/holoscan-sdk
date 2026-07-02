/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef HOLOSCAN_OPERATORS_FORMAT_CONVERTER_GPU_RESIDENT_FORMAT_CONVERTER_GPU_RESIDENT_HPP
#define HOLOSCAN_OPERATORS_FORMAT_CONVERTER_GPU_RESIDENT_FORMAT_CONVERTER_GPU_RESIDENT_HPP

#include <npp.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "holoscan/core/gpu_resident_operator.hpp"
#include "holoscan/core/io_context.hpp"
#include "holoscan/core/io_spec.hpp"
#include "holoscan/core/operator_spec.hpp"
#include "holoscan/operators/format_converter/format_converter_common.hpp"

namespace holoscan::ops {

/**
 * @brief Row pitch and byte range for one color plane in a contiguous device buffer.
 *
 * Used only to pass NPP step sizes and plane base offsets/sizes (same information GXF's
 * `ColorPlane` carries for VideoBuffer layouts). This type avoids a GXF multimedia dependency in
 * the GPU-resident operator API.
 */
struct FormatConverterPlaneLayout {
  /// Bytes between consecutive rows (NPP `nSrcStep` / plane step).
  int32_t stride_bytes = 0;
  /// Byte offset of this plane from the start of the buffer.
  uint32_t byte_offset = 0;
  /// Size of this plane in bytes (for splitting Y/U/V or Y/UV in a single allocation).
  uint64_t plane_bytes = 0;
};

/**
 * @brief GPU-resident format conversion using NPP (fixed layout; device buffers only).
 *
 * This operator is designed for GPU-resident graph execution mode where:
 * - All setup and configuration are done once during initialization.
 * - The `compute` function is launched only once from the CPU.
 * - Subsequent executions are run entirely on the GPU via CUDA graph capture.
 *
 * It mirrors @ref FormatConverterOp conversions for tensor-packed / planar layouts that the
 * standard operator supports, with dimensions and dtypes fixed at initialization.
 *
 * ==Device inputs==
 * - **in** : Source frame in device memory (size depends on `width`, `height`, `in_dtype`).
 *
 * ==Device outputs==
 * - **out** : Converted frame in device memory (size depends on conversion and optional resize).
 *
 * ==Parameters==
 * - **width**, **height**: Source frame size in pixels. For YUV420 / NV12, both must be even.
 * - **in_dtype**, **out_dtype**: Same string values as `FormatConverterOp` (`rgb888`, `uint8`,
 *   `float32`, `rgba8888`, `yuv420`, `nv12_bt601_full`, `nv12_bt709_hdtv`, `nv12_bt709_csc`,
 *   `yuyv`, `uyvy`, `rgb161616`, `rgba16161616`, etc.).
 * - **resize_width**, **resize_height**: If both are positive, resize the packed RGB/RGBA/float
 *   input before conversion (same restriction as `FormatConverterOp`: not supported for planar
 *   YUV420 / NV12 inputs).
 * - **resize_mode**: NPP `NppiInterpolationMode` (0 selects cubic, as in `FormatConverterOp`).
 * - **in_plane_strides_bytes**, **in_plane_offsets_bytes**: Optional fixed input plane layout in
 *   bytes. If omitted, the operator uses its default packed/I420/NV12 layout assumptions. This is
 *   useful when matching a fixed GXF `VideoBuffer` layout without depending on GXF multimedia
 *   structs.
 * - **out_plane_strides_bytes**, **out_plane_offsets_bytes**: Optional fixed output plane layout
 *   in bytes. If omitted, the operator uses its default tightly-packed / I420 output layout.
 * - **scale_min**, **scale_max**, **alpha_value**, **out_channel_order**: Same semantics as
 *   `FormatConverterOp`.
 */
class FormatConverterGpuResidentOp : public holoscan::GPUResidentOperator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS_SUPER(FormatConverterGpuResidentOp, holoscan::GPUResidentOperator)

  FormatConverterGpuResidentOp() = default;
  ~FormatConverterGpuResidentOp() override;

  void setup(OperatorSpec& spec) override;
  void initialize() override;
  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override;

 private:
  void launchConvert(const void* in_tensor_data,
                     const std::vector<FormatConverterPlaneLayout>& input_plane_layout,
                     void* out_tensor_data,
                     const std::vector<FormatConverterPlaneLayout>& output_plane_layout);

  Parameter<int32_t> width_;
  Parameter<int32_t> height_;
  Parameter<std::string> in_dtype_str_;
  Parameter<std::string> out_dtype_str_;
  Parameter<float> scale_min_;
  Parameter<float> scale_max_;
  Parameter<uint8_t> alpha_value_;
  Parameter<int32_t> resize_width_;
  Parameter<int32_t> resize_height_;
  Parameter<int32_t> resize_mode_;
  Parameter<std::vector<int32_t>> in_plane_strides_bytes_;
  Parameter<std::vector<uint32_t>> in_plane_offsets_bytes_;
  Parameter<std::vector<int32_t>> out_plane_strides_bytes_;
  Parameter<std::vector<uint32_t>> out_plane_offsets_bytes_;
  Parameter<std::vector<int>> out_channel_order_;

  NppStreamContext npp_stream_ctx_{};
  NppiInterpolationMode npp_resize_mode_ = NPPI_INTER_CUBIC;

  FormatDType in_dtype_ = FormatDType::kUnknown;
  FormatDType out_dtype_ = FormatDType::kUnknown;
  uint32_t in_element_size_ = 0;
  uint32_t out_element_size_ = 0;
  FormatConversionType format_conversion_type_ = FormatConversionType::kUnknown;

  int32_t src_rows_ = 0;
  int32_t src_cols_ = 0;
  int32_t work_rows_ = 0;
  int32_t work_cols_ = 0;
  int16_t in_channels_ = 0;
  int16_t out_channels_ = 0;
  std::vector<FormatConverterPlaneLayout> input_plane_layout_;
  std::vector<FormatConverterPlaneLayout> output_plane_layout_;

  void* resize_buffer_dev_ = nullptr;
  size_t resize_buffer_bytes_ = 0;
  void* channel_buffer_dev_ = nullptr;
  size_t channel_buffer_bytes_ = 0;
};

}  // namespace holoscan::ops

#endif /* HOLOSCAN_OPERATORS_FORMAT_CONVERTER_GPU_RESIDENT_FORMAT_CONVERTER_GPU_RESIDENT_HPP */
