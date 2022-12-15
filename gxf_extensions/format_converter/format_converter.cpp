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
#include "format_converter.hpp"

#include <cuda_runtime.h>
#include <npp.h>

#include <string>
#include <utility>
#include <vector>

#include "gxf/core/entity.hpp"
#include "gxf/core/expected.hpp"
#include "gxf/multimedia/video.hpp"
#include "gxf/std/tensor.hpp"

#define CUDA_TRY(stmt)                                                                            \
  ({                                                                                              \
    cudaError_t _holoscan_cuda_err = stmt;                                                        \
    if (cudaSuccess != _holoscan_cuda_err) {                                                      \
      GXF_LOG_ERROR("CUDA Runtime call %s in line %d of file %s failed with '%s' (%d).\n", #stmt, \
                    __LINE__, __FILE__, cudaGetErrorString(_holoscan_cuda_err),                   \
                    _holoscan_cuda_err);                                                          \
    }                                                                                             \
    _holoscan_cuda_err;                                                                           \
  })

namespace nvidia::holoscan::formatconverter {

static FormatDType toFormatDType(const std::string& str) {
  if (str == "rgb888") {
    return FormatDType::kRGB888;
  } else if (str == "uint8") {
    return FormatDType::kUnsigned8;
  } else if (str == "float32") {
    return FormatDType::kFloat32;
  } else if (str == "rgba8888") {
    return FormatDType::kRGBA8888;
  } else {
    return FormatDType::kUnknown;
  }
}

static constexpr FormatConversionType getFormatConversionType(FormatDType from, FormatDType to) {
  if (from != FormatDType::kUnknown && to != FormatDType::kUnknown && from == to) {
    return FormatConversionType::kNone;
  } else if (from == FormatDType::kUnsigned8 && to == FormatDType::kFloat32) {
    return FormatConversionType::kUnsigned8ToFloat32;
  } else if (from == FormatDType::kFloat32 && to == FormatDType::kUnsigned8) {
    return FormatConversionType::kFloat32ToUnsigned8;
  } else if (from == FormatDType::kUnsigned8 && to == FormatDType::kRGBA8888) {
    return FormatConversionType::kRGB888ToRGBA8888;
  } else if (from == FormatDType::kRGBA8888 && to == FormatDType::kUnsigned8) {
    return FormatConversionType::kRGBA8888ToRGB888;
  } else if (from == FormatDType::kRGBA8888 && to == FormatDType::kFloat32) {
    return FormatConversionType::kRGBA8888ToFloat32;
  } else {
    return FormatConversionType::kUnknown;
  }
}

static constexpr FormatDType normalizeFormatDType(FormatDType dtype) {
  switch (dtype) {
    case FormatDType::kRGB888:
      return FormatDType::kUnsigned8;
    default:
      return dtype;
  }
}

static constexpr gxf::PrimitiveType primitiveTypeFromFormatDType(FormatDType dtype) {
  switch (dtype) {
    case FormatDType::kRGB888:
    case FormatDType::kRGBA8888:
    case FormatDType::kUnsigned8:
      return gxf::PrimitiveType::kUnsigned8;
    case FormatDType::kFloat32:
      return gxf::PrimitiveType::kFloat32;
    default:
      return gxf::PrimitiveType::kCustom;
  }
}

static constexpr FormatDType FormatDTypeFromPrimitiveType(gxf::PrimitiveType type) {
  switch (type) {
    case gxf::PrimitiveType::kUnsigned8:
      return FormatDType::kUnsigned8;
    case gxf::PrimitiveType::kFloat32:
      return FormatDType::kFloat32;
    default:
      return FormatDType::kUnknown;
  }
}

static gxf_result_t verifyFormatDTypeChannels(FormatDType dtype, int channel_count) {
  switch (dtype) {
    case FormatDType::kRGB888:
      if (channel_count != 3) {
        GXF_LOG_ERROR("Invalid channel count for RGB888 %d != 3\n", channel_count);
        return GXF_FAILURE;
      }
      break;
    case FormatDType::kRGBA8888:
      if (channel_count != 4) {
        GXF_LOG_ERROR("Invalid channel count for RGBA8888 %d != 4\n", channel_count);
        return GXF_FAILURE;
      }
      break;
    default:
      break;
  }
  return GXF_SUCCESS;
}

gxf_result_t FormatConverter::start() {
  out_dtype_ = toFormatDType(out_dtype_str_.get());
  if (out_dtype_ == FormatDType::kUnknown) {
    GXF_LOG_ERROR("Unsupported output format dtype: %s\n", out_dtype_str_.get().c_str());
    return GXF_FAILURE;
  }
  out_primitive_type_ = primitiveTypeFromFormatDType(out_dtype_);

  if (out_primitive_type_ == gxf::PrimitiveType::kCustom) {
    GXF_LOG_ERROR("Unsupported output format dtype: %s\n", out_dtype_str_.get().c_str());
    return GXF_FAILURE;
  }

  if (!in_dtype_str_.get().empty()) {
    in_dtype_ = toFormatDType(in_dtype_str_.get());
    if (in_dtype_ == FormatDType::kUnknown) {
      GXF_LOG_ERROR("Unsupported input format dtype: %s\n", in_dtype_str_.get().c_str());
      return GXF_FAILURE;
    }
    format_conversion_type_ =
        getFormatConversionType(normalizeFormatDType(in_dtype_), normalizeFormatDType(out_dtype_));
    in_primitive_type_ = primitiveTypeFromFormatDType(in_dtype_);
  }

  switch (resize_mode_) {
    case 0:
      resize_mode_.set(NPPI_INTER_CUBIC);
      break;
    case 1:                                // NPPI_INTER_NN
    case 2:                                // NPPI_INTER_LINEAR
    case 4:                                // NPPI_INTER_CUBIC
    case 5:                                // NPPI_INTER_CUBIC2P_BSPLINE
    case 6:                                // NPPI_INTER_CUBIC2P_CATMULLROM
    case 7:                                // NPPI_INTER_CUBIC2P_B05C03
    case 8:                                // NPPI_INTER_SUPER
    case 16:                               // NPPI_INTER_LANCZOS
    case 17:                               // NPPI_INTER_LANCZOS3_ADVANCED
    case static_cast<int32_t>(0x8000000):  // NPPI_SMOOTH_EDGE
      break;
    default:
      GXF_LOG_ERROR("Unsupported resize mode: %d\n", resize_mode_.get());
      return GXF_FAILURE;
  }

  return GXF_SUCCESS;
}

gxf_result_t FormatConverter::stop() {
  resize_buffer_.freeBuffer();
  channel_buffer_.freeBuffer();
  device_scratch_buffer_.freeBuffer();
  return GXF_SUCCESS;
}

gxf_result_t FormatConverter::tick() {
  // Process input message
  const auto in_message = in_->receive();
  if (!in_message || in_message.value().is_null()) { return GXF_CONTRACT_MESSAGE_NOT_AVAILABLE; }

  gxf::Shape out_shape{0, 0, 0};
  void* in_tensor_data = nullptr;
  gxf::PrimitiveType in_primitive_type = gxf::PrimitiveType::kCustom;
  gxf::MemoryStorageType in_memory_storage_type = gxf::MemoryStorageType::kHost;
  int32_t rows = 0;
  int32_t columns = 0;
  int16_t in_channels = 0;
  int16_t out_channels = 0;

  // Get tensor attached to the message
  auto maybe_video = in_message.value().get<gxf::VideoBuffer>();
  gxf::Handle<gxf::Tensor> in_tensor;
  if (maybe_video) {
    // Convert VideoBuffer to Tensor
    auto frame = maybe_video.value();

    // NOTE: VideoBuffer::moveToTensor() converts VideoBuffer instance to the Tensor instance
    // with an unexpected shape:  [width, height] or [width, height, num_planes].
    // And, if we use moveToTensor() to convert VideoBuffer to Tensor, we may lose the the original
    // video buffer when the VideoBuffer instance is used in other places. For that reason, we
    // directly access internal data of VideoBuffer instance to access Tensor data.
    const auto& buffer_info = frame->video_frame_info();
    switch (buffer_info.color_format) {
      case gxf::VideoFormat::GXF_VIDEO_FORMAT_RGBA:
        in_primitive_type = gxf::PrimitiveType::kUnsigned8;
        break;
      default:
        GXF_LOG_ERROR("Unsupported input format: %d\n", buffer_info.color_format);
        return GXF_FAILURE;
    }

    // Get needed information from the tensor
    in_memory_storage_type = frame->storage_type();
    out_shape = gxf::Shape{static_cast<int32_t>(buffer_info.height),
                           static_cast<int32_t>(buffer_info.width), 4};
    in_tensor_data = frame->pointer();
    rows = buffer_info.height;
    columns = buffer_info.width;
    in_channels = 4;  // RGBA
    out_channels = in_channels;

    // If the buffer is in host memory, copy it to a device (GPU) buffer
    // as needed for the NPP resize/convert operations.
    if (in_memory_storage_type == gxf::MemoryStorageType::kHost) {
      size_t buffer_size = rows * columns * in_channels;
      if (buffer_size > device_scratch_buffer_.size()) {
        device_scratch_buffer_.resize(pool_, buffer_size, gxf::MemoryStorageType::kDevice);
        if (!device_scratch_buffer_.pointer()) {
          GXF_LOG_ERROR("Failed to allocate device scratch buffer (%d bytes)", buffer_size);
          return GXF_FAILURE;
        }
      }
      CUDA_TRY(cudaMemcpy(device_scratch_buffer_.pointer(), frame->pointer(), buffer_size,
                          cudaMemcpyHostToDevice));
      in_tensor_data = device_scratch_buffer_.pointer();
      in_memory_storage_type = gxf::MemoryStorageType::kDevice;
    }
  } else {
    const auto maybe_tensor = in_message.value().get<gxf::Tensor>(in_tensor_name_.get().c_str());
    if (!maybe_tensor) {
      GXF_LOG_ERROR("Tensor '%s' not found in message.\n", in_tensor_name_.get().c_str());
      return GXF_FAILURE;
    }
    in_tensor = maybe_tensor.value();

    // Get needed information from the tensor
    out_shape = in_tensor->shape();
    in_tensor_data = in_tensor->pointer();
    in_primitive_type = in_tensor->element_type();
    in_memory_storage_type = in_tensor->storage_type();
    rows = in_tensor->shape().dimension(0);
    columns = in_tensor->shape().dimension(1);
    in_channels = in_tensor->shape().dimension(2);
    out_channels = in_channels;
  }

  if (in_memory_storage_type != gxf::MemoryStorageType::kDevice) {
    GXF_LOG_ERROR("Tensor('%s') or VideoBuffer is not allocated on device.\n",
                  in_tensor_name_.get().c_str());
    return GXF_MEMORY_INVALID_STORAGE_MODE;
  }

  if (in_dtype_ == FormatDType::kUnknown) {
    in_primitive_type_ = in_primitive_type;
    in_dtype_ = FormatDTypeFromPrimitiveType(in_primitive_type_);
    format_conversion_type_ =
        getFormatConversionType(normalizeFormatDType(in_dtype_), normalizeFormatDType(out_dtype_));
  }

  // Check if input tensor is consistent over all the frames
  if (in_primitive_type != in_primitive_type_) {
    GXF_LOG_ERROR("Input tensor element type is inconsistent over all the frames.\n");
    return GXF_FAILURE;
  }

  // Check if the input/output tensor is compatible with the format conversion
  if (format_conversion_type_ == FormatConversionType::kUnknown) {
    GXF_LOG_ERROR("Unsupported format conversion: %s (%" PRIu32 ") -> %s\n",
                  in_dtype_str_.get().c_str(), static_cast<uint32_t>(in_dtype_),
                  out_dtype_str_.get().c_str());
    return GXF_FAILURE;
  }

  // Check that if the format requires a specific number of channels they are consistent.
  // Some formats (e.g. float32) are agnostic to channel count, while others (e.g. RGB888) have a
  // specific channel count.
  if (GXF_SUCCESS != verifyFormatDTypeChannels(in_dtype_, in_channels)) {
    GXF_LOG_ERROR("Failed to verify the channels for the expected input dtype [%d]: %d.", in_dtype_,
                  in_channels);
    return GXF_FAILURE;
  }

  // Resize the input image before converting data type
  if (resize_width_ > 0 && resize_height_ > 0) {
    auto resize_result = resizeImage(in_tensor_data, rows, columns, in_channels, in_primitive_type,
                                     resize_width_, resize_height_);
    if (!resize_result) {
      GXF_LOG_ERROR("Failed to resize image.\n");
      return resize_result.error();
    }

    // Update the tensor pointer and shape
    out_shape = gxf::Shape{resize_height_, resize_width_, in_channels};
    in_tensor_data = resize_result.value();
    rows = resize_height_;
    columns = resize_width_;
  }

  // Create output message
  const uint32_t dst_typesize = gxf::PrimitiveTypeSize(out_primitive_type_);

  // Adjust output shape if the conversion involves the change in the channel dimension
  switch (format_conversion_type_) {
    case FormatConversionType::kRGB888ToRGBA8888: {
      out_channels = 4;
      out_shape = gxf::Shape{out_shape.dimension(0), out_shape.dimension(1), out_channels};
      break;
    }
    case FormatConversionType::kRGBA8888ToRGB888:
    case FormatConversionType::kRGBA8888ToFloat32: {
      out_channels = 3;
      out_shape = gxf::Shape{out_shape.dimension(0), out_shape.dimension(1), out_channels};
      break;
    }
    default:
      break;
  }

  // Check that if the format requires a specific number of channels they are consistent.
  // Some formats (e.g. float32) are agnostic to channel count, while others (e.g. RGB888) have a
  // specific channel count.
  if (GXF_SUCCESS != verifyFormatDTypeChannels(out_dtype_, out_channels)) {
    GXF_LOG_ERROR("Failed to verify the channels for the expected output dtype [%d]: %d.",
                  out_dtype_, out_channels);
    return GXF_FAILURE;
  }

  gxf::Expected<gxf::Entity> out_message = CreateTensorMap(
      context(), pool_,
      {{out_tensor_name_.get(), gxf::MemoryStorageType::kDevice, out_shape, out_primitive_type_, 0,
        gxf::ComputeTrivialStrides(out_shape, dst_typesize)}});

  if (!out_message) return out_message.error();
  const auto out_tensor = out_message.value().get<gxf::Tensor>();
  if (!out_tensor) { return out_tensor.error(); }

  // Set tensor to constant using NPP
  if (in_channels == 3 || in_channels == 4) {
    gxf_result_t convert_result = convertTensorFormat(in_tensor_data, out_tensor.value()->pointer(),
                                                      rows, columns, in_channels, out_channels);

    if (convert_result != GXF_SUCCESS) {
      GXF_LOG_ERROR("Failed to convert tensor format (conversion type:%d)",
                    static_cast<int>(format_conversion_type_));
      return GXF_FAILURE;
    }
  } else {
    GXF_LOG_ERROR("Only support 3 or 4 channel input tensor");
    return GXF_NOT_IMPLEMENTED;
  }

  const auto result = out_->publish(out_message.value());

  // Publish output message
  return gxf::ToResultCode(result);
}

gxf::Expected<void*> FormatConverter::resizeImage(const void* in_tensor_data, const int32_t rows,
                                                  const int32_t columns, const int16_t channels,
                                                  const gxf::PrimitiveType primitive_type,
                                                  const int32_t resize_width,
                                                  const int32_t resize_height) {
  if (resize_buffer_.size() == 0) {
    uint64_t buffer_size = resize_width * resize_height * channels;
    resize_buffer_.resize(pool_, buffer_size, gxf::MemoryStorageType::kDevice);
  }

  const auto converted_tensor_ptr = resize_buffer_.pointer();
  if (converted_tensor_ptr == nullptr) {
    GXF_LOG_ERROR("Failed to allocate memory for the resizing image");
    return gxf::ExpectedOrCode(GXF_FAILURE, nullptr);
  }

  // Resize image
  NppStatus status = NPP_ERROR;
  const NppiSize src_size = {static_cast<int>(columns), static_cast<int>(rows)};
  const NppiRect src_roi = {0, 0, static_cast<int>(columns), static_cast<int>(rows)};
  const NppiSize dst_size = {static_cast<int>(resize_width), static_cast<int>(resize_height)};
  const NppiRect dst_roi = {0, 0, static_cast<int>(resize_width), static_cast<int>(resize_height)};

  switch (channels) {
    case 3:
      switch (primitive_type) {
        case gxf::PrimitiveType::kUnsigned8:
          status = nppiResize_8u_C3R(static_cast<const Npp8u*>(in_tensor_data), columns * channels,
                                     src_size, src_roi, converted_tensor_ptr,
                                     resize_width * channels, dst_size, dst_roi, resize_mode_);
          break;
        default:
          GXF_LOG_ERROR("Unsupported input primitive type for resizing image");
          return gxf::ExpectedOrCode(GXF_FAILURE, nullptr);
      }
      break;
    case 4:
      switch (primitive_type) {
        case gxf::PrimitiveType::kUnsigned8:
          status = nppiResize_8u_C4R(static_cast<const Npp8u*>(in_tensor_data), columns * channels,
                                     src_size, src_roi, converted_tensor_ptr,
                                     resize_width * channels, dst_size, dst_roi, resize_mode_);
          break;
        default:
          GXF_LOG_ERROR("Unsupported input primitive type for resizing image");
          return gxf::ExpectedOrCode(GXF_FAILURE, nullptr);
      }
      break;
    default:
      GXF_LOG_ERROR("Unsupported input primitive type for resizing image (%d, %d)", channels,
                    static_cast<int32_t>(primitive_type));
      return gxf::ExpectedOrCode(GXF_FAILURE, nullptr);
      break;
  }

  if (status != NPP_SUCCESS) { return gxf::ExpectedOrCode(GXF_FAILURE, nullptr); }

  return gxf::ExpectedOrCode(GXF_SUCCESS, converted_tensor_ptr);
}

gxf_result_t FormatConverter::convertTensorFormat(const void* in_tensor_data, void* out_tensor_data,
                                                  const int32_t rows, const int32_t columns,
                                                  const int16_t in_channels,
                                                  const int16_t out_channels) {
  const uint32_t src_typesize = gxf::PrimitiveTypeSize(in_primitive_type_);
  const uint32_t dst_typesize = gxf::PrimitiveTypeSize(out_primitive_type_);

  const int32_t src_step = columns * in_channels * src_typesize;
  const int32_t dst_step = columns * out_channels * dst_typesize;

  const auto& out_channel_order = out_channel_order_.get();

  NppStatus status = NPP_ERROR;
  const NppiSize roi = {static_cast<int>(columns), static_cast<int>(rows)};

  switch (format_conversion_type_) {
    case FormatConversionType::kNone: {
      const auto in_tensor_ptr = static_cast<const uint8_t*>(in_tensor_data);
      auto out_tensor_ptr = static_cast<uint8_t*>(out_tensor_data);

      cudaError_t cuda_status = CUDA_TRY(
          cudaMemcpy(out_tensor_ptr, in_tensor_ptr, src_step * rows, cudaMemcpyDeviceToDevice));
      if (cuda_status) {
        GXF_LOG_ERROR("Failed to copy GPU data to GPU memory.");
        return GXF_FAILURE;
      }
      status = NPP_SUCCESS;
      break;
    }
    case FormatConversionType::kUnsigned8ToFloat32: {
      const auto in_tensor_ptr = static_cast<const uint8_t*>(in_tensor_data);
      const auto out_tensor_ptr = static_cast<float*>(out_tensor_data);
      status = nppiScale_8u32f_C3R(in_tensor_ptr, src_step, out_tensor_ptr, dst_step, roi,
                                   scale_min_, scale_max_);
      break;
    }
    case FormatConversionType::kFloat32ToUnsigned8: {
      const auto in_tensor_ptr = static_cast<const float*>(in_tensor_data);
      const auto out_tensor_ptr = static_cast<uint8_t*>(out_tensor_data);
      status = nppiScale_32f8u_C3R(in_tensor_ptr, src_step, out_tensor_ptr, dst_step, roi,
                                   scale_min_, scale_max_);
      break;
    }
    case FormatConversionType::kRGB888ToRGBA8888: {
      const auto in_tensor_ptr = static_cast<const uint8_t*>(in_tensor_data);
      const auto out_tensor_ptr = static_cast<uint8_t*>(out_tensor_data);
      // Convert RGB888 to RGBA8888 (3 channels -> 4 channels, uint8_t)
      int dst_order[4]{0, 1, 2, 3};
      if (!out_channel_order.empty()) {
        if (out_channel_order.size() != 4) {
          GXF_LOG_ERROR("Invalid channel order for RGBA8888");
          return GXF_FAILURE;
        }
        for (int i = 0; i < 4; i++) { dst_order[i] = out_channel_order[i]; }
      }
      status = nppiSwapChannels_8u_C3C4R(in_tensor_ptr, src_step, out_tensor_ptr,
                                         out_channels * dst_typesize * columns, roi, dst_order,
                                         alpha_value_.get());
      break;
    }
    case FormatConversionType::kRGBA8888ToRGB888: {
      const auto in_tensor_ptr = static_cast<const uint8_t*>(in_tensor_data);
      const auto out_tensor_ptr = static_cast<uint8_t*>(out_tensor_data);
      // Convert RGBA8888 to RGB888 (4 channels -> 3 channels, uint8_t)
      int dst_order[3]{0, 1, 2};
      if (!out_channel_order.empty()) {
        if (out_channel_order.size() != 3) {
          GXF_LOG_ERROR("Invalid channel order for RGB888");
          return GXF_FAILURE;
        }
        for (int i = 0; i < 3; i++) { dst_order[i] = out_channel_order[i]; }
      }
      status = nppiSwapChannels_8u_C4C3R(in_tensor_ptr, src_step, out_tensor_ptr,
                                         out_channels * dst_typesize * columns, roi, dst_order);
      break;
    }
    case FormatConversionType::kRGBA8888ToFloat32: {
      const auto in_tensor_ptr = static_cast<const uint8_t*>(in_tensor_data);
      const auto out_tensor_ptr = static_cast<float*>(out_tensor_data);

      if (channel_buffer_.size() == 0) {
        uint64_t buffer_size = rows * columns * 3;  // 4 channels -> 3 channels
        channel_buffer_.resize(pool_, buffer_size, gxf::MemoryStorageType::kDevice);
      }

      const auto converted_tensor_ptr = channel_buffer_.pointer();
      if (converted_tensor_ptr == nullptr) {
        GXF_LOG_ERROR("Failed to allocate memory for the channel conversion");
        return GXF_FAILURE;
      }

      int dst_order[3]{0, 1, 2};
      if (!out_channel_order.empty()) {
        if (out_channel_order.size() != 3) {
          GXF_LOG_ERROR("Invalid channel order for RGB888");
          return GXF_FAILURE;
        }
        for (int i = 0; i < 3; i++) { dst_order[i] = out_channel_order[i]; }
      }

      status = nppiSwapChannels_8u_C4C3R(in_tensor_ptr, src_step, converted_tensor_ptr,
                                         out_channels * src_typesize * columns, roi, dst_order);

      if (status == NPP_SUCCESS) {
        const int32_t new_src_step = columns * out_channels * src_typesize;
        status = nppiScale_8u32f_C3R(converted_tensor_ptr, new_src_step, out_tensor_ptr, dst_step,
                                     roi, scale_min_, scale_max_);
      } else {
        GXF_LOG_ERROR("Failed to convert channel order (NPP error code: %d)", status);
        return GXF_FAILURE;
      }
      break;
    }
    default:
      GXF_LOG_ERROR("Unsupported format conversion: %s (%" PRIu32 ") -> %s\n",
                    in_dtype_str_.get().c_str(), static_cast<uint32_t>(in_dtype_),
                    out_dtype_str_.get().c_str());
  }

  // Reorder channels in the output tensor (inplace) if needed.
  switch (format_conversion_type_) {
    case FormatConversionType::kNone:
    case FormatConversionType::kUnsigned8ToFloat32:
    case FormatConversionType::kFloat32ToUnsigned8: {
      if (!out_channel_order.empty()) {
        switch (out_channels) {
          case 3: {
            int dst_order[3]{0, 1, 2};
            if (out_channel_order.size() != 3) {
              GXF_LOG_ERROR("Invalid channel order for %s", out_dtype_str_.get().c_str());
              return GXF_FAILURE;
            }
            for (int i = 0; i < 3; i++) { dst_order[i] = out_channel_order[i]; }
            switch (out_primitive_type_) {
              case gxf::PrimitiveType::kUnsigned8: {
                auto out_tensor_ptr = static_cast<uint8_t*>(out_tensor_data);
                status = nppiSwapChannels_8u_C3IR(out_tensor_ptr, dst_step, roi, dst_order);
                break;
              }
              case gxf::PrimitiveType::kFloat32: {
                auto out_tensor_ptr = static_cast<float*>(out_tensor_data);
                status = nppiSwapChannels_32f_C3IR(out_tensor_ptr, dst_step, roi, dst_order);
                break;
              }
              default:
                GXF_LOG_ERROR("Unsupported output data type for reordering channels: %s",
                              out_dtype_str_.get().c_str());
            }
            break;
          }
          case 4: {
            int dst_order[4]{0, 1, 2, 3};
            if (out_channel_order.size() != 4) {
              GXF_LOG_ERROR("Invalid channel order for %s", out_dtype_str_.get().c_str());
              return GXF_FAILURE;
            }
            for (int i = 0; i < 4; i++) { dst_order[i] = out_channel_order[i]; }
            switch (out_primitive_type_) {
              case gxf::PrimitiveType::kUnsigned8: {
                auto out_tensor_ptr = static_cast<uint8_t*>(out_tensor_data);
                status = nppiSwapChannels_8u_C4IR(out_tensor_ptr, dst_step, roi, dst_order);
                break;
              }
              case gxf::PrimitiveType::kFloat32: {
                auto out_tensor_ptr = static_cast<float*>(out_tensor_data);
                status = nppiSwapChannels_32f_C4IR(out_tensor_ptr, dst_step, roi, dst_order);
                break;
              }
              default:
                GXF_LOG_ERROR("Unsupported output data type for reordering channels: %s\n",
                              out_dtype_str_.get().c_str());
            }
            break;
          }
        }
        if (status != NPP_SUCCESS) {
          GXF_LOG_ERROR("Failed to convert channel order");
          return GXF_FAILURE;
        }
      }
    }
    default:
      break;
  }

  if (status != NPP_SUCCESS) { return GXF_FAILURE; }

  return GXF_SUCCESS;
}

gxf_result_t FormatConverter::registerInterface(gxf::Registrar* registrar) {
  gxf::Expected<void> result;
  result &= registrar->parameter(in_, "in", "Input", "Input channel.");
  result &= registrar->parameter(in_tensor_name_, "in_tensor_name", "InputTensorName",
                                 "Name of the input tensor.", std::string(""));
  result &= registrar->parameter(in_dtype_str_, "in_dtype", "InputDataType", "Source data type.",
                                 std::string(""));
  result &= registrar->parameter(out_, "out", "Output", "Output channel.");
  result &= registrar->parameter(out_tensor_name_, "out_tensor_name", "OutputTensorName",
                                 "Name of the output tensor.", std::string(""));
  result &=
      registrar->parameter(out_dtype_str_, "out_dtype", "OutputDataType", "Destination data type.");
  result &= registrar->parameter(scale_min_, "scale_min", "Scale min",
                                 "Minimum value of the scale.", 0.f);
  result &= registrar->parameter(scale_max_, "scale_max", "Scale max",
                                 "Maximum value of the scale.", 1.f);
  result &= registrar->parameter(alpha_value_, "alpha_value", "Alpha value",
                                 "Alpha value that can be used to fill the alpha channel when "
                                 "converting RGB888 to RGBA8888.",
                                 static_cast<uint8_t>(255));
  result &= registrar->parameter(resize_width_, "resize_width", "Resize width",
                                 "Width for resize. No actions if this value is zero.", 0);
  result &= registrar->parameter(resize_height_, "resize_height", "Resize height",
                                 "Height for resize. No actions if this value is zero.", 0);
  result &= registrar->parameter(resize_mode_, "resize_mode", "Resize mode",
                                 "Mode for resize. 4 (NPPI_INTER_CUBIC) if this value is zero.", 0);
  result &= registrar->parameter(
      out_channel_order_, "out_channel_order", "Output channel order",
      "Host memory integer array describing how channel values are permutated.",
      std::vector<int>{});
  result &= registrar->parameter(pool_, "pool", "Pool", "Pool to allocate the output message.");
  return gxf::ToResultCode(result);
}

}  // namespace nvidia::holoscan::formatconverter
