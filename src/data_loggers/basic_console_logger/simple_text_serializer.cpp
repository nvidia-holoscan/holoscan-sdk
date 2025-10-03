/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "holoscan/data_loggers/basic_console_logger/simple_text_serializer.hpp"

#include <algorithm>
#include <any>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <magic_enum.hpp>
#include "gxf/core/entity.hpp"
#include "gxf/core/handle.hpp"
#include "gxf/multimedia/video.hpp"
#include "holoscan/core/domain/tensor.hpp"
#include "holoscan/core/domain/tensor_map.hpp"
#include "holoscan/core/gxf/entity.hpp"
#include "holoscan/core/metadata.hpp"
#include "holoscan/logger/logger.hpp"
#include "holoscan/utils/cuda_macros.hpp"

// #include "holoscan/core/component_spec.hpp"
// #include "holoscan/core/fragment.hpp"

namespace holoscan {
namespace data_loggers {

void SimpleTextSerializer::setup(ComponentSpec& spec) {
  spec.param(max_elements_,
             "max_elements",
             "Maximum Elements",
             "Maximum number of vector elements to display before truncation",
             static_cast<int64_t>(10));
  spec.param(max_metadata_items_,
             "max_metadata_items",
             "Maximum Metadata Items",
             "Maximum number of metadata dictionary items to display before truncation",
             static_cast<int64_t>(10));
  spec.param(log_video_buffer_content_,
             "log_video_buffer_content",
             "Log Video Buffer Content",
             "Whether to log the actual pixel data content of video buffers",
             false);

  // Initialize the default type encoders
  initialize_default_encoders();
}

void SimpleTextSerializer::initialize_default_encoders() {
  // String types
  encoders_[std::type_index(typeid(std::string))] = [](const std::any& value) {
    return std::any_cast<std::string>(value);
  };
  encoders_[std::type_index(typeid(const char*))] = [](const std::any& value) {
    return std::string(std::any_cast<const char*>(value));
  };
  encoders_[std::type_index(typeid(char*))] = [](const std::any& value) {
    return std::string(std::any_cast<char*>(value));
  };
  encoders_[std::type_index(typeid(std::shared_ptr<std::string>))] = [](const std::any& value) {
    return *std::any_cast<std::shared_ptr<std::string>>(value);
  };

  // Integer types
  encoders_[std::type_index(typeid(int))] = [](const std::any& value) {
    return std::to_string(std::any_cast<int>(value));
  };
  encoders_[std::type_index(typeid(int8_t))] = [](const std::any& value) {
    return std::to_string(std::any_cast<int8_t>(value));
  };
  encoders_[std::type_index(typeid(uint8_t))] = [](const std::any& value) {
    return std::to_string(std::any_cast<uint8_t>(value));
  };
  encoders_[std::type_index(typeid(int16_t))] = [](const std::any& value) {
    return std::to_string(std::any_cast<int16_t>(value));
  };
  encoders_[std::type_index(typeid(uint16_t))] = [](const std::any& value) {
    return std::to_string(std::any_cast<uint16_t>(value));
  };
  encoders_[std::type_index(typeid(int32_t))] = [](const std::any& value) {
    return std::to_string(std::any_cast<int32_t>(value));
  };
  encoders_[std::type_index(typeid(uint32_t))] = [](const std::any& value) {
    return std::to_string(std::any_cast<uint32_t>(value));
  };
  encoders_[std::type_index(typeid(int64_t))] = [](const std::any& value) {
    return std::to_string(std::any_cast<int64_t>(value));
  };
  encoders_[std::type_index(typeid(uint64_t))] = [](const std::any& value) {
    return std::to_string(std::any_cast<uint64_t>(value));
  };

  // Floating point types
  encoders_[std::type_index(typeid(float))] = [](const std::any& value) {
    return std::to_string(std::any_cast<float>(value));
  };
  encoders_[std::type_index(typeid(double))] = [](const std::any& value) {
    return std::to_string(std::any_cast<double>(value));
  };

  // Boolean type
  encoders_[std::type_index(typeid(bool))] = [](const std::any& value) {
    return std::any_cast<bool>(value) ? "true" : "false";
  };

  // MetadataDictionary type
  encoders_[std::type_index(typeid(MetadataDictionary))] = [this](const std::any& value) {
    return format_metadata_dict(std::any_cast<MetadataDictionary>(value));
  };

  // Vector types
  encoders_[std::type_index(typeid(std::vector<std::string>))] = [this](const std::any& value) {
    return format_vector(std::any_cast<std::vector<std::string>>(value));
  };
  encoders_[std::type_index(typeid(std::vector<int>))] = [this](const std::any& value) {
    return format_vector(std::any_cast<std::vector<int>>(value));
  };
  encoders_[std::type_index(typeid(std::vector<int8_t>))] = [this](const std::any& value) {
    return format_vector(std::any_cast<std::vector<int8_t>>(value));
  };
  encoders_[std::type_index(typeid(std::vector<uint8_t>))] = [this](const std::any& value) {
    return format_vector(std::any_cast<std::vector<uint8_t>>(value));
  };
  encoders_[std::type_index(typeid(std::vector<int16_t>))] = [this](const std::any& value) {
    return format_vector(std::any_cast<std::vector<int16_t>>(value));
  };
  encoders_[std::type_index(typeid(std::vector<uint16_t>))] = [this](const std::any& value) {
    return format_vector(std::any_cast<std::vector<uint16_t>>(value));
  };
  encoders_[std::type_index(typeid(std::vector<int32_t>))] = [this](const std::any& value) {
    return format_vector(std::any_cast<std::vector<int32_t>>(value));
  };
  encoders_[std::type_index(typeid(std::vector<uint32_t>))] = [this](const std::any& value) {
    return format_vector(std::any_cast<std::vector<uint32_t>>(value));
  };
  encoders_[std::type_index(typeid(std::vector<int64_t>))] = [this](const std::any& value) {
    return format_vector(std::any_cast<std::vector<int64_t>>(value));
  };
  encoders_[std::type_index(typeid(std::vector<uint64_t>))] = [this](const std::any& value) {
    return format_vector(std::any_cast<std::vector<uint64_t>>(value));
  };
  encoders_[std::type_index(typeid(std::vector<float>))] = [this](const std::any& value) {
    return format_vector(std::any_cast<std::vector<float>>(value));
  };
  encoders_[std::type_index(typeid(std::vector<double>))] = [this](const std::any& value) {
    return format_vector(std::any_cast<std::vector<double>>(value));
  };
  encoders_[std::type_index(typeid(std::vector<bool>))] = [this](const std::any& value) {
    return format_vector(std::any_cast<std::vector<bool>>(value));
  };

  // Handle<VideoBuffer> type
  encoders_[std::type_index(typeid(nvidia::gxf::Handle<nvidia::gxf::VideoBuffer>))] =
      [this](const std::any& value) {
        return serialize_video_buffer_to_string(
            std::any_cast<nvidia::gxf::Handle<nvidia::gxf::VideoBuffer>>(value));
      };
}

// Helper to convert std::any to string for common types
std::string SimpleTextSerializer::to_string(const std::any& data) const {
  if (!data.has_value())
    return "<empty>";

  // Look up the encoder for this type
  std::type_index type_idx(data.type());
  auto it = encoders_.find(type_idx);

  if (it != encoders_.end()) {
    return it->second(data);
  }

  // Fallback: print type name for unregistered types
  return std::string("<unhandled type: ") + data.type().name() + ">";
}

bool SimpleTextSerializer::can_handle_message(const std::type_info& /*message_type*/) const {
  // This serializer generates a message for any type so return true.
  // Types without an encoder will be reported as: "<unhandled type: ...>"
  return true;
}

// Helper template method to format vectors with size limit
template <typename T>
std::string SimpleTextSerializer::format_vector(const std::vector<T>& vec) const {
  std::ostringstream oss;
  oss << "[";

  size_t max_elements = static_cast<size_t>(max_elements_.get());
  size_t elements_to_show = std::min(vec.size(), max_elements);
  for (size_t i = 0; i < elements_to_show; ++i) {
    if (i > 0)
      oss << ", ";

    // Fast paths for common types to avoid std::any overhead
    if constexpr (std::is_same_v<T, std::string>) {
      oss << vec[i];
    } else if constexpr (std::is_same_v<T, float>) {
      oss << vec[i];
    } else if constexpr (std::is_same_v<T, double>) {
      oss << vec[i];
    } else if constexpr (std::is_same_v<T, bool>) {
      oss << (vec[i] ? "true" : "false");
    } else if constexpr (std::is_same_v<T, int>) {
      oss << vec[i];
    } else if constexpr (std::is_same_v<T, int8_t>) {
      // Special handling to avoid char interpretation
      oss << static_cast<int>(vec[i]);
    } else if constexpr (std::is_same_v<T, uint8_t>) {
      // Special handling to avoid char interpretation
      oss << static_cast<unsigned int>(vec[i]);
    } else if constexpr (std::is_same_v<T, int16_t>) {
      oss << vec[i];
    } else if constexpr (std::is_same_v<T, uint16_t>) {
      oss << vec[i];
    } else if constexpr (std::is_same_v<T, int32_t>) {
      oss << vec[i];
    } else if constexpr (std::is_same_v<T, uint32_t>) {
      oss << vec[i];
    } else if constexpr (std::is_same_v<T, int64_t>) {
      oss << vec[i];
    } else if constexpr (std::is_same_v<T, uint64_t>) {
      oss << vec[i];
    } else {
      // C++17 SFINAE-friendly approach to detect std::vector
      if constexpr (std::is_same_v<
                        T,
                        std::vector<typename T::value_type, typename T::allocator_type>>) {
        // T is a vector type - check if there's a registered encoder first
        std::type_index type_idx(typeid(T));
        auto it = encoders_.find(type_idx);
        if (it != encoders_.end()) {
          oss << it->second(vec[i]);
        } else {
          // No registered encoder - use recursive format_vector directly
          oss << format_vector(vec[i]);
        }
      } else {
        // General case: use the encoder registry via to_string
        std::any val = vec[i];
        oss << to_string(val);
      }
    }
  }

  if (vec.size() > max_elements) {
    oss << ", ... (" << (vec.size() - max_elements) << " more)";
  }

  oss << "]";
  return oss.str();
}

// Helper method to format MetadataDictionary
std::string SimpleTextSerializer::format_metadata_dict(const MetadataDictionary& dict) const {
  std::ostringstream oss;
  oss << "MetadataDictionary(size=" << dict.size() << ") {";

  if (dict.size() == 0) {
    oss << "}";
    return oss.str();
  }

  int64_t max_metadata_items = max_metadata_items_.get();
  size_t count = 0;
  bool first = true;

  for (const auto& [key, meta_obj_ptr] : dict) {
    if (max_metadata_items > 0 && count >= static_cast<size_t>(max_metadata_items)) {
      oss << ", ... (" << (dict.size() - max_metadata_items) << " more)";
      break;
    }

    if (!first)
      oss << ", ";
    first = false;

    oss << "'" << key << "': ";

    if (meta_obj_ptr) {
      // Get the std::any value from the MetadataObject and recursively serialize it
      auto value = meta_obj_ptr->value();
      oss << to_string(value);
    } else {
      oss << "null";
    }

    ++count;
  }

  oss << "}";
  return oss.str();
}

// Serialize the data as a UTF-8 string
std::string SimpleTextSerializer::serialize_to_string(std::any data) {
  return to_string(data);
}

// Serialize the data as a UTF-8 string
std::string SimpleTextSerializer::serialize_metadata_to_string(
    const std::shared_ptr<MetadataDictionary>& metadata) {
  return format_metadata_dict(*metadata);
}

std::string SimpleTextSerializer::serialize_tensor_to_string(const std::shared_ptr<Tensor>& tensor,
                                                             bool log_data_content) {
  std::ostringstream oss;

  // Add basic tensor information
  auto shape = tensor->shape();
  auto device = tensor->device();
  auto dtype = tensor->dtype();

  oss << "Tensor(";
  oss << "shape=[";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i > 0)
      oss << ", ";
    oss << shape[i];
  }
  std::string device_name{magic_enum::enum_name(static_cast<DLDeviceType>(device.device_type))};

  oss << "], ";
  oss << "dtype=" << magic_enum::enum_name(static_cast<DLDataTypeCode>(dtype.code)) << ":"
      << static_cast<int>(dtype.bits) << ":" << static_cast<int>(dtype.lanes) << ", ";
  oss << "device=" << device_name << ":" << device.device_id << ", ";
  oss << "size=" << tensor->size() << ", ";
  oss << "nbytes=" << tensor->nbytes() << ", ";
  oss << "C-contiguous=" << (tensor->is_contiguous() ? "true" : "false");

  if (log_data_content && tensor->size() > 0) {
    oss << ", data=";

    // Check that lanes == 1 (only support this case for now)
    if (dtype.lanes != 1) {
      oss << "<Unsupported dtype.lanes=" << static_cast<int>(dtype.lanes)
          << " (only lanes=1 supported)>";
    } else if (device.device_type != DLDeviceType::kDLCPU &&
               device.device_type != DLDeviceType::kDLCUDA &&
               device.device_type != DLDeviceType::kDLCUDAHost &&
               device.device_type != DLDeviceType::kDLCUDAManaged) {
      oss << "<Unsupported DLDeviceType: " << device_name << ">";
    } else {
      // Helper lambda to print data based on dtype
      auto print_data = [&](auto dummy_type) {
        using T = decltype(dummy_type);
        size_t num_elements = tensor->size() / sizeof(T);
        size_t max_elements = static_cast<size_t>(max_elements_.get());
        size_t elements_to_show = std::min(num_elements, max_elements);

        oss << "[";

        const T* data_ptr;
        std::vector<T> host_data;  // Only used for CUDA device or managed memory
        bool is_device_memory = device.device_type == DLDeviceType::kDLCUDA;
        bool is_cuda_managed = device.device_type == DLDeviceType::kDLCUDAManaged;
        if (is_device_memory || is_cuda_managed) {
          // For CUDA device memory, copy only the elements we need to show
          host_data.resize(elements_to_show);
          cudaError_t result = HOLOSCAN_CUDA_CALL(
              cudaMemcpy(host_data.data(),
                         tensor->data(),
                         elements_to_show * sizeof(T),
                         is_cuda_managed ? cudaMemcpyDefault : cudaMemcpyDeviceToHost));
          if (cudaSuccess != result) {
            HOLOSCAN_LOG_ERROR(
                "Copy of GPU data back to host failed... cannot log the data values");
            oss << "...cudaMemcpy error...]";
            return;
          }
          data_ptr = host_data.data();
        } else {
          // For host-accessible memory (CPU, CUDAHost, CUDAManaged), directly access the data
          // pointer
          data_ptr = static_cast<const T*>(tensor->data());
        }

        // print the values
        for (size_t i = 0; i < elements_to_show; ++i) {
          if (i > 0)
            oss << ", ";

          if constexpr (std::is_same_v<T, int8_t>) {
            oss << static_cast<int>(data_ptr[i]);
          } else if constexpr (std::is_same_v<T, uint8_t>) {
            oss << static_cast<unsigned int>(data_ptr[i]);
          } else if constexpr (std::is_floating_point_v<T>) {
            oss << std::fixed << std::setprecision(6) << std::showpoint << data_ptr[i];
          } else {
            oss << data_ptr[i];
          }
        }

        if (num_elements > max_elements) {
          oss << ", ... (" << (num_elements - max_elements) << " more)";
        }
        oss << "]";
      };

      // Support specified dtypes based on code and bits
      bool supported = true;
      switch (dtype.code) {
        case kDLInt:
          switch (dtype.bits) {
            case 8:
              print_data(int8_t{});
              break;
            case 16:
              print_data(int16_t{});
              break;
            case 32:
              print_data(int32_t{});
              break;
            case 64:
              print_data(int64_t{});
              break;
            default:
              supported = false;
              break;
          }
          break;
        case kDLUInt:
          switch (dtype.bits) {
            case 8:
              print_data(uint8_t{});
              break;
            case 16:
              print_data(uint16_t{});
              break;
            case 32:
              print_data(uint32_t{});
              break;
            case 64:
              print_data(uint64_t{});
              break;
            default:
              supported = false;
              break;
          }
          break;
        case kDLFloat:
          switch (dtype.bits) {
            case 32:
              print_data(float{});
              break;
            case 64:
              print_data(double{});
              break;
            default:
              supported = false;
              break;
          }
          break;
        default:
          supported = false;
          break;
      }

      if (!supported) {
        oss << "<Unsupported DLDataType: "
            << magic_enum::enum_name(static_cast<DLDataTypeCode>(dtype.code)) << ":"
            << static_cast<int>(dtype.bits) << ":" << static_cast<int>(dtype.lanes) << ">";
      }
    }
  }

  oss << ")";
  return oss.str();
}

std::string SimpleTextSerializer::serialize_tensormap_to_string(const TensorMap& tensor_map,
                                                                bool log_data_content) {
  std::ostringstream oss;

  oss << "TensorMap(size=" << tensor_map.size() << ") {";

  bool first = true;
  for (const auto& [key, tensor_ptr] : tensor_map) {
    if (!first)
      oss << ", ";
    first = false;

    oss << "'" << key << "': ";
    if (tensor_ptr) {
      // Recursively serialize the individual tensor
      oss << serialize_tensor_to_string(tensor_ptr, log_data_content);
    } else {
      oss << "null";
    }
  }

  oss << "}";
  return oss.str();
}

std::string SimpleTextSerializer::serialize_video_buffer_to_string(
    const nvidia::gxf::Handle<nvidia::gxf::VideoBuffer>& video_buffer) {
  std::ostringstream oss;

  auto maybe_frame_ptr = video_buffer.try_get();
  if (!maybe_frame_ptr) {
    oss << "VideoBuffer(null)";
    return oss.str();
  }

  auto frame = maybe_frame_ptr.value();
  const auto& buffer_info = frame->video_frame_info();

  // Basic video buffer information
  oss << "VideoBuffer(";
  oss << "width=" << buffer_info.width << ", ";
  oss << "height=" << buffer_info.height << ", ";
  oss << "color_format=" << magic_enum::enum_name(buffer_info.color_format) << ", ";
  oss << "storage_type=" << magic_enum::enum_name(frame->storage_type()) << ", ";
  oss << "surface_layout=" << magic_enum::enum_name(buffer_info.surface_layout) << ", ";
  oss << "num_planes=" << buffer_info.color_planes.size();

  // For single plane interleaved formats, show plane details
  if (buffer_info.color_planes.size() == 1) {
    const auto& plane = buffer_info.color_planes[0];
    oss << ", plane_info={";
    oss << "width=" << plane.width << ", ";
    oss << "height=" << plane.height << ", ";
    oss << "stride=" << plane.stride << ", ";
    oss << "size=" << plane.size << ", ";
    oss << "bytes_per_pixel=" << static_cast<int>(plane.bytes_per_pixel);
    oss << "}";

    // Only support single plane formats for data logging
    if (log_video_buffer_content_.get() && plane.size > 0) {
      oss << ", data=";

      // Get format information using extracted function
      const auto format_info = get_video_format_info(buffer_info.color_format);

      if (!format_info.supported) {
        oss << "<Unsupported color format for data display: "
            << magic_enum::enum_name(buffer_info.color_format) << ">";
      } else if (frame->storage_type() != nvidia::gxf::MemoryStorageType::kHost &&
                 frame->storage_type() != nvidia::gxf::MemoryStorageType::kSystem &&
                 frame->storage_type() != nvidia::gxf::MemoryStorageType::kCudaManaged &&
                 frame->storage_type() != nvidia::gxf::MemoryStorageType::kDevice) {
        oss << "<Unsupported memory storage type for data display: "
            << magic_enum::enum_name(frame->storage_type()) << ">";
      } else {
        // Show first few pixels
        size_t max_pixels = static_cast<size_t>(max_elements_.get());
        size_t total_pixels = buffer_info.width * buffer_info.height;
        size_t pixels_to_show = std::min(total_pixels, max_pixels);

        // Determine element size based on data type
        size_t element_size = 1;
        switch (format_info.element_type) {
          case VideoElementType::kUINT8:
            element_size = sizeof(uint8_t);
            break;
          case VideoElementType::kUINT16:
            element_size = sizeof(uint16_t);
            break;
          case VideoElementType::kUINT32:
            element_size = sizeof(uint32_t);
            break;
          case VideoElementType::kFLOAT32:
            element_size = sizeof(float);
            break;
          case VideoElementType::kFLOAT64:
            element_size = sizeof(double);
            break;
        }

        const void* data_ptr;
        std::vector<uint8_t> host_data;  // Only used for CUDA device memory
        bool is_device_memory = frame->storage_type() == nvidia::gxf::MemoryStorageType::kDevice;
        bool is_cuda_managed =
            frame->storage_type() == nvidia::gxf::MemoryStorageType::kCudaManaged;
        if (is_device_memory || is_cuda_managed) {
          // For CUDA device memory, copy only the data we need to show
          size_t bytes_to_copy = pixels_to_show * format_info.channels * element_size;
          host_data.resize(bytes_to_copy);
          cudaError_t result = HOLOSCAN_CUDA_CALL(
              cudaMemcpy(host_data.data(),
                         frame->pointer(),
                         bytes_to_copy,
                         is_cuda_managed ? cudaMemcpyDefault : cudaMemcpyDeviceToHost));
          if (cudaSuccess != result) {
            HOLOSCAN_LOG_ERROR(
                "Copy of GPU video buffer data back to host failed... cannot log the data values");
            oss << "...cudaMemcpy error...]";
            return oss.str();
          }
          data_ptr = host_data.data();
        } else {
          // For host-accessible memory, directly access the data pointer
          data_ptr = frame->pointer();
        }

        // serialize the pixel data
        oss << serialize_pixel_data(
            data_ptr, format_info.element_type, format_info.channels, pixels_to_show, total_pixels);
      }
    }
  } else {
    // Multi-plane formats not supported for data logging
    if (log_video_buffer_content_.get()) {
      oss << ", data=<Multi-plane formats not supported for data display>";
    }
  }

  oss << ")";
  return oss.str();
}

SimpleTextSerializer::VideoFormatInfo SimpleTextSerializer::get_video_format_info(
    nvidia::gxf::VideoFormat format) const {
  VideoFormatInfo info{0, VideoElementType::kUINT8, false};

  switch (format) {
    case nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_RGB:
    case nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_BGR:
      info.channels = 3;
      info.element_type = VideoElementType::kUINT8;
      info.supported = true;
      break;
    case nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_RGB16:
    case nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_BGR16:
      info.channels = 3;
      info.element_type = VideoElementType::kUINT16;
      info.supported = true;
      break;
    case nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_RGB32:
    case nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_BGR32:
      info.channels = 3;
      info.element_type = VideoElementType::kUINT32;
      info.supported = true;
      break;
    case nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_RGBA:
    case nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_BGRA:
    case nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_ARGB:
    case nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_ABGR:
    case nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_RGBX:
    case nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_BGRX:
    case nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_XRGB:
    case nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_XBGR:
    case nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_RGBD8:
      info.channels = 4;
      info.element_type = VideoElementType::kUINT8;
      info.supported = true;
      break;
    case nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_RGBD16:
    case nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_RAW16_RGGB:
    case nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_RAW16_BGGR:
    case nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_RAW16_GRBG:
    case nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_RAW16_GBRG:
      info.channels = 4;
      info.element_type = VideoElementType::kUINT16;
      info.supported = true;
      break;
    case nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_RGBD32:
      info.channels = 4;
      info.element_type = VideoElementType::kUINT32;
      info.supported = true;
      break;
    case nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_GRAY:
      info.channels = 1;
      info.element_type = VideoElementType::kUINT8;
      info.supported = true;
      break;
    case nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_GRAY16:
      info.channels = 1;
      info.element_type = VideoElementType::kUINT16;
      info.supported = true;
      break;
    case nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_GRAY32:
      info.channels = 1;
      info.element_type = VideoElementType::kUINT32;
      info.supported = true;
      break;
    case nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_GRAY32F:
    case nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_D32F:
      info.channels = 1;
      info.element_type = VideoElementType::kFLOAT32;
      info.supported = true;
      break;
    case nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_D64F:
      info.channels = 1;
      info.element_type = VideoElementType::kFLOAT64;
      info.supported = true;
      break;
    default:
      info.channels = 0;
      info.element_type = VideoElementType::kUINT8;
      info.supported = false;
      break;
  }

  return info;
}

std::string SimpleTextSerializer::serialize_pixel_data(const void* data_ptr,
                                                       VideoElementType element_type, int channels,
                                                       size_t pixels_to_show,
                                                       size_t total_pixels) const {
  std::ostringstream oss;
  oss << "[";

  // Templated helper function to print pixel values efficiently
  auto print_pixels_typed = [&](auto typed_data_ptr) {
    using T = std::remove_pointer_t<std::decay_t<decltype(typed_data_ptr)>>;
    for (size_t pixel = 0; pixel < pixels_to_show; ++pixel) {
      if (pixel > 0)
        oss << ", ";

      if (channels == 1) {
        // Grayscale
        if constexpr (std::is_same_v<T, uint8_t>) {
          oss << static_cast<unsigned int>(typed_data_ptr[pixel]);
        } else if constexpr (std::is_floating_point_v<T>) {
          oss << std::fixed << std::setprecision(6) << std::showpoint << typed_data_ptr[pixel];
        } else {
          oss << typed_data_ptr[pixel];
        }
      } else {
        // RGB or RGBA
        oss << "(";
        for (int c = 0; c < channels; ++c) {
          if (c > 0)
            oss << ",";
          if constexpr (std::is_same_v<T, uint8_t>) {
            oss << static_cast<unsigned int>(typed_data_ptr[pixel * channels + c]);
          } else if constexpr (std::is_floating_point_v<T>) {
            oss << std::fixed << std::setprecision(6) << std::showpoint
                << typed_data_ptr[pixel * channels + c];
          } else {
            oss << typed_data_ptr[pixel * channels + c];
          }
        }
        oss << ")";
      }
    }
  };  // NOLINT(readability/braces)

  // Dispatch to the appropriate templated function based on element type
  switch (element_type) {
    case VideoElementType::kUINT8:
      print_pixels_typed(static_cast<const uint8_t*>(data_ptr));
      break;
    case VideoElementType::kUINT16:
      print_pixels_typed(static_cast<const uint16_t*>(data_ptr));
      break;
    case VideoElementType::kUINT32:
      print_pixels_typed(static_cast<const uint32_t*>(data_ptr));
      break;
    case VideoElementType::kFLOAT32:
      print_pixels_typed(static_cast<const float*>(data_ptr));
      break;
    case VideoElementType::kFLOAT64:
      print_pixels_typed(static_cast<const double*>(data_ptr));
      break;
  }

  if (total_pixels > static_cast<size_t>(max_elements_.get())) {
    oss << ", ... (" << (total_pixels - static_cast<size_t>(max_elements_.get()))
        << " more pixels)";
  }
  oss << "]";

  return oss.str();
}

}  // namespace data_loggers
}  // namespace holoscan
