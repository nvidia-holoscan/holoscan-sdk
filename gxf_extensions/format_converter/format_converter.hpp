/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#ifndef NVIDIA_CLARA_HOLOSCAN_GXF_EXTENSIONS_FORMAT_CONVERTER_FORMAT_CONVERTER_HPP_
#define NVIDIA_CLARA_HOLOSCAN_GXF_EXTENSIONS_FORMAT_CONVERTER_FORMAT_CONVERTER_HPP_

#include <cinttypes>
#include <string>
#include <vector>

#include "gxf/std/allocator.hpp"
#include "gxf/std/codelet.hpp"
#include "gxf/std/receiver.hpp"
#include "gxf/std/tensor.hpp"
#include "gxf/std/transmitter.hpp"

namespace nvidia::holoscan::formatconverter {

enum class FormatDType { kUnknown, kRGB888, kRGBA8888, kUnsigned8, kFloat32 };

enum class FormatConversionType {
  kUnknown,
  kNone,
  kUnsigned8ToFloat32,
  kFloat32ToUnsigned8,
  kRGB888ToRGBA8888,
  kRGBA8888ToRGB888,
  kRGBA8888ToFloat32,
};

/// @brief Helper codelet for common tensor operations in inference pipelines.
///
/// Provides a codelet that provides common video or tensor operations in inference pipelines to
/// change datatypes, resize images, reorder channels, and normalize and scale values.
class FormatConverter : public gxf::Codelet {
 public:
  gxf_result_t start() override;
  gxf_result_t tick() override;
  gxf_result_t stop() override;
  gxf_result_t registerInterface(gxf::Registrar* registrar) override;

  gxf::Expected<void*> resizeImage(const void* in_tensor_data, const int32_t rows,
                                   const int32_t columns, const int16_t channels,
                                   const gxf::PrimitiveType primitive_type,
                                   const int32_t resize_width, const int32_t resize_height);
  gxf_result_t convertTensorFormat(const void* in_tensor_data, void* out_tensor_data,
                                   const int32_t rows, const int32_t columns,
                                   const int16_t in_channels, const int16_t out_channels);

 private:
  gxf::Parameter<gxf::Handle<gxf::Receiver>> in_;
  gxf::Parameter<std::string> in_tensor_name_;
  gxf::Parameter<std::string> in_dtype_str_;
  gxf::Parameter<gxf::Handle<gxf::Transmitter>> out_;
  gxf::Parameter<std::string> out_tensor_name_;
  gxf::Parameter<std::string> out_dtype_str_;
  gxf::Parameter<float> scale_min_;
  gxf::Parameter<float> scale_max_;
  gxf::Parameter<uint8_t> alpha_value_;
  gxf::Parameter<int32_t> resize_width_;
  gxf::Parameter<int32_t> resize_height_;
  gxf::Parameter<int32_t> resize_mode_;
  gxf::Parameter<std::vector<int>> out_channel_order_;
  gxf::Parameter<gxf::Handle<gxf::Allocator>> pool_;

  FormatDType in_dtype_ = FormatDType::kUnknown;
  FormatDType out_dtype_ = FormatDType::kUnknown;
  gxf::PrimitiveType in_primitive_type_ = gxf::PrimitiveType::kCustom;
  gxf::PrimitiveType out_primitive_type_ = gxf::PrimitiveType::kCustom;
  FormatConversionType format_conversion_type_ = FormatConversionType::kUnknown;

  gxf::MemoryBuffer resize_buffer_;
  gxf::MemoryBuffer channel_buffer_;
  gxf::MemoryBuffer device_scratch_buffer_;
};

}  // namespace nvidia::holoscan::formatconverter

#endif  // NVIDIA_CLARA_HOLOSCAN_GXF_EXTENSIONS_FORMAT_CONVERTER_FORMAT_CONVERTER_HPP_
