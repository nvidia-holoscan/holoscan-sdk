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

#ifndef HOLOSCAN_DATA_LOGGERS_BASIC_CONSOLE_LOGGER_SIMPLE_TEXT_SERIALIZER_HPP
#define HOLOSCAN_DATA_LOGGERS_BASIC_CONSOLE_LOGGER_SIMPLE_TEXT_SERIALIZER_HPP

#include <any>
#include <cstddef>
#include <functional>
#include <memory>  // For std::shared_ptr in parameters
#include <sstream>
#include <string>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>
#include <utility>
#include <vector>

#include "holoscan/core/component_spec.hpp"
#include "holoscan/core/resource.hpp"

// Forward declarations for GXF types
namespace nvidia {
namespace gxf {
template <typename T>
class Handle;
class VideoBuffer;
enum class VideoFormat : std::int64_t;
}  // namespace gxf
}  // namespace nvidia

namespace holoscan {

// Forward declarations
class Tensor;
class TensorMap;
class MetadataDictionary;

namespace data_loggers {

class SimpleTextSerializer : public Resource {
 public:
  HOLOSCAN_RESOURCE_FORWARD_ARGS_SUPER(SimpleTextSerializer, Resource)
  SimpleTextSerializer() = default;
  ~SimpleTextSerializer() override = default;

  void setup(ComponentSpec& spec) override;

  bool can_handle_message(const std::type_info& message_type) const;

  /**
   * @brief Serialize data to a simple text string
   */
  std::string serialize_to_string(std::any data);

  /**
   * @brief Serialize tensor data to a simple text string
   */
  std::string serialize_tensor_to_string(const std::shared_ptr<Tensor>& tensor,
                                         bool log_data_content = false);

  /**
   * @brief Serialize tensor map data to a simple text string
   */
  std::string serialize_tensormap_to_string(const TensorMap& tensor_map,
                                            bool log_data_content = false);

  /**
   * @brief Serialize metadata to a simple text string
   */
  std::string serialize_metadata_to_string(const std::shared_ptr<MetadataDictionary>& metadata);

  /**
   * @brief Serialize video buffer data to a simple text string
   */
  std::string serialize_video_buffer_to_string(
      const nvidia::gxf::Handle<nvidia::gxf::VideoBuffer>& video_buffer);

  /**
   * @brief Register a custom encoder for a specific type
   *
   * @tparam T The type to register an encoder for
   * @param encoder Function that takes const std::any& and returns std::string
   */
  template <typename T>
  void register_encoder(std::function<std::string(const std::any&)> encoder) {
    encoders_[std::type_index(typeid(T))] = std::move(encoder);
  }

 private:
  Parameter<int64_t> max_elements_;
  Parameter<int64_t> max_metadata_items_;
  Parameter<bool> log_video_buffer_content_;

  // Type encoder registry
  std::unordered_map<std::type_index, std::function<std::string(const std::any&)>> encoders_;

  // Initialize the default encoders
  void initialize_default_encoders();

  // Helper template method to format vectors with size limit
  template <typename T>
  std::string format_vector(const std::vector<T>& vec) const;

  // Helper method to format MetadataDictionary
  std::string format_metadata_dict(const MetadataDictionary& dict) const;

  // Helper to convert std::any to string for common types
  std::string to_string(const std::any& data) const;

  // Video buffer format detection structures and methods
  enum class VideoElementType { kUINT8, kUINT16, kUINT32, kFLOAT32, kFLOAT64 };

  struct VideoFormatInfo {
    int channels;
    VideoElementType element_type;
    bool supported;
  };

  /**
   * @brief Detect video format information from GXF VideoFormat
   * @param format The GXF video format to analyze
   * @return VideoFormatInfo containing channels, element type, and support status
   */
  VideoFormatInfo get_video_format_info(nvidia::gxf::VideoFormat format) const;

  /**
   * @brief Serialize pixel data to string representation
   * @param data_ptr Pointer to the pixel data
   * @param element_type Type of elements in the data
   * @param channels Number of channels per pixel
   * @param pixels_to_show Number of pixels to serialize
   * @param total_pixels Total number of pixels (for truncation message)
   * @return String representation of the pixel data
   */
  std::string serialize_pixel_data(const void* data_ptr, VideoElementType element_type,
                                   int channels, size_t pixels_to_show, size_t total_pixels) const;
};

}  // namespace data_loggers
}  // namespace holoscan

#endif /* HOLOSCAN_DATA_LOGGERS_BASIC_CONSOLE_LOGGER_SIMPLE_TEXT_SERIALIZER_HPP */
