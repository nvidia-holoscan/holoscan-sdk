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

#include "holoscan/core/arg.hpp"

#include <string>
#include <vector>

#include "holoscan/utils/yaml_parser.hpp"

namespace holoscan {

std::string ArgType::to_string() const {
  std::string el_type_str = element_type_name_map_.at(element_type_);
  if (container_type_ == ArgContainerType::kNative) {
    return el_type_str;
  }
  std::string nested_str =
      container_type_ == ArgContainerType::kVector ? "std::vector<{}>" : "std::array<{},N>";
  std::string container_str = nested_str;
  for (int32_t i = 1; i < dimension_; ++i) {
    container_str = fmt::vformat(nested_str, fmt::make_format_args(container_str));
  }
  return fmt::vformat(container_str, fmt::make_format_args(el_type_str));
}

template <typename T>
inline static bool try_any_cast(const std::any& a, YAML::Node& out) {
  if (a.type() == typeid(T)) {
    // Cast uint8_t and int8_t values to uint16_t and int16_t respectively to ensure they do not
    // appear as non-UTF-8 characters in python due to
    // https://pybind11.readthedocs.io/en/stable/advanced/cast/strings.html#returning-c-strings-to-python
    //   e.g. We want 255 as uint8_t to appear as 255 and not <unicode character> in
    //   Arg::description
    if constexpr (std::is_same_v<T, uint8_t>) {
      out = YAML::Node(static_cast<uint16_t>(std::any_cast<T>(a)));
    } else if constexpr (std::is_same_v<T, int8_t>) {
      out = YAML::Node(static_cast<int16_t>(std::any_cast<T>(a)));
    } else {
      out = YAML::Node(std::any_cast<T>(a));
    }
    return true;
  }
  return false;
}

template <typename... Ts>
inline static YAML::Node cast_any_node(const std::any& a) {
  YAML::Node result;
  bool matched = (try_any_cast<Ts>(a, result) || ...);
  if (!matched) {
    throw std::runtime_error(fmt::format("Unsupported type (in cast_any_node): {} -> {}",
                                         a.type().name(),
                                         typeid(YAML::Node).name()));
  }
  return result;
}

template <typename T>
inline static YAML::Node scalar_as_node(const std::any& val) {
  // Regaredless of the type, we want to cast it to a YAML::Node
  return cast_any_node<bool,
                       int8_t,
                       uint8_t,
                       int16_t,
                       uint16_t,
                       int32_t,
                       uint32_t,
                       int64_t,
                       uint64_t,
                       long,
                       unsigned long,
                       long long,
                       unsigned long long,
                       float,
                       double,
                       std::complex<float>,
                       std::complex<double>,
                       std::string,
                       YAML::Node>(val);
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
    case ArgElementType::kComplex64:
      return any_as_node<std::complex<float>>(val, container_t);
    case ArgElementType::kComplex128:
      return any_as_node<std::complex<double>>(val, container_t);
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

YAML::Node Arg::value_to_yaml_node() const {
  return any_as_node(value_, arg_type_);
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
  for (const Arg& arg : args_) {
    node["args"].push_back(arg.to_yaml_node());
  }
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
