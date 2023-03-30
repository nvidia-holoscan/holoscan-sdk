/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "holoscan/core/arg.hpp"

namespace holoscan {

std::string ArgType::to_string() const {
  std::string el_type_str = element_type_name_map_.at(element_type_);
  if (container_type_ == ArgContainerType::kNative) { return el_type_str; }
  std::string nested_str =
      container_type_ == ArgContainerType::kVector ? "std::vector<{}>" : "std::array<{},N>";
  std::string container_str = nested_str;
  for (int32_t i = 1; i < dimension_; ++i) { container_str = fmt::format(nested_str, nested_str); }
  return fmt::format(container_str, el_type_str);
}

template <typename T>
inline static YAML::Node scalar_as_node(const std::any& val) {
  return YAML::Node(std::any_cast<T>(val));
}

template <typename T>
inline static YAML::Node vector_as_node(const std::any& val) {
  try {
    return YAML::Node(std::any_cast<std::vector<T>>(val));
  } catch (const std::bad_cast& e) {  // 2d:  std::vector<std::vector<T>>
    try {
      return YAML::Node(std::any_cast<std::vector<std::vector<T>>>(val));
    } catch (const std::bad_cast& e) {
      return YAML::Node(YAML::NodeType::Undefined);
    }
  }
}

template <typename T>
inline static YAML::Node any_as_node(const std::any& val, ArgContainerType type) {
  switch (type) {
    case ArgContainerType::kNative:
      return scalar_as_node<T>(val);
    case ArgContainerType::kVector:
      return vector_as_node<T>(val);
    case ArgContainerType::kArray:
      // Don't know size of arrays, abort
      return YAML::Node(YAML::NodeType::Undefined);
  }
  return YAML::Node(YAML::NodeType::Undefined);
}

inline static YAML::Node any_as_node(const std::any& val, ArgType type) {
  ArgContainerType container_t = type.container_type();
  switch (type.element_type()) {
    case ArgElementType::kBoolean:
      return any_as_node<bool>(val, container_t);
    case ArgElementType::kInt8:
      return any_as_node<int8_t>(val, container_t);
    case ArgElementType::kUnsigned8:
      return any_as_node<uint8_t>(val, container_t);
    case ArgElementType::kInt16:
      return any_as_node<int16_t>(val, container_t);
    case ArgElementType::kUnsigned16:
      return any_as_node<uint16_t>(val, container_t);
    case ArgElementType::kInt32:
      return any_as_node<int32_t>(val, container_t);
    case ArgElementType::kUnsigned32:
      return any_as_node<uint32_t>(val, container_t);
    case ArgElementType::kInt64:
      return any_as_node<int64_t>(val, container_t);
    case ArgElementType::kUnsigned64:
      return any_as_node<uint64_t>(val, container_t);
    case ArgElementType::kFloat32:
      return any_as_node<float>(val, container_t);
    case ArgElementType::kFloat64:
      return any_as_node<double>(val, container_t);
    case ArgElementType::kString:
      return any_as_node<std::string>(val, container_t);
    case ArgElementType::kYAMLNode:
      return any_as_node<YAML::Node>(val, container_t);
    default:  // kCustom, kHandle, kIOSpec, kCondition, kResource
      return YAML::Node(YAML::NodeType::Undefined);
  }
  return YAML::Node(YAML::NodeType::Undefined);
}

YAML::Node Arg::to_yaml_node() const {
  YAML::Node node;
  node["name"] = name_;
  node["type"] = arg_type_.to_string();
  node["value"] = any_as_node(value_, arg_type_);
  return node;
}

std::string Arg::description() const {
  YAML::Emitter emitter;
  emitter << to_yaml_node();
  return emitter.c_str();
}

/**
 * @brief Get a YAML representation of the argument list.
 *
 * @return YAML node including the name, and arguments of the argument list.
 */
YAML::Node ArgList::to_yaml_node() const {
  YAML::Node node;
  node["name"] = name_;
  node["args"] = YAML::Node(YAML::NodeType::Sequence);
  for (const Arg& arg : args_) { node["args"].push_back(arg.to_yaml_node()); }
  return node;
}

/**
 * @brief Get a description of the argument list.
 *
 * @see to_yaml_node()
 * @return YAML string.
 */
std::string ArgList::description() const {
  YAML::Emitter emitter;
  emitter << to_yaml_node();
  return emitter.c_str();
}

}  // namespace holoscan
