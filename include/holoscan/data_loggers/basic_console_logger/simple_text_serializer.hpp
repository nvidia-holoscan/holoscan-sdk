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

#ifndef HOLOSCAN_DATA_LOGGERS_BASIC_DATA_LOGGER_SIMPLE_TEXT_SERIALIZER_HPP
#define HOLOSCAN_DATA_LOGGERS_BASIC_DATA_LOGGER_SIMPLE_TEXT_SERIALIZER_HPP

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
};

}  // namespace data_loggers
}  // namespace holoscan

#endif /* HOLOSCAN_DATA_LOGGERS_BASIC_DATA_LOGGER_SIMPLE_TEXT_SERIALIZER_HPP */
