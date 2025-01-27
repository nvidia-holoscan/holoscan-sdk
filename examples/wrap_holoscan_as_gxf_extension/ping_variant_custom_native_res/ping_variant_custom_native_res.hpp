/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef WRAP_HOLOSCAN_AS_GXF_EXTENSION_PING_VARIANT_CUSTOM_NATIVE_RES_PING_VARIANT_CUSTOM_NATIVE_RES_HPP  // NOLINT
#define WRAP_HOLOSCAN_AS_GXF_EXTENSION_PING_VARIANT_CUSTOM_NATIVE_RES_PING_VARIANT_CUSTOM_NATIVE_RES_HPP  // NOLINT

#include <yaml-cpp/yaml.h>

#include <sstream>
#include <string>

#include "holoscan/holoscan.hpp"

namespace myres {

class CustomIntType4Resource {
 public:
  CustomIntType4Resource() = default;
  explicit CustomIntType4Resource(int value) : value_(value) {}

  int value() const { return value_; }
  void value(int value) { value_ = value; }

 private:
  int value_ = 0;
};

class PingVarCustomNativeRes : public holoscan::Resource {
 public:
  HOLOSCAN_RESOURCE_FORWARD_ARGS(PingVarCustomNativeRes)

  PingVarCustomNativeRes() = default;

  void initialize() override;

  void setup(holoscan::ComponentSpec& spec) override;

  // Get the custom int value from `custom_int_type_`.
  int get_custom_int();

  // Get the float value from `float_type_`.
  float get_float();

 private:
  holoscan::Parameter<CustomIntType4Resource> custom_int_type_;
  holoscan::Parameter<float> float_type_;
};

}  // namespace myres

template <>
struct YAML::convert<myres::CustomIntType4Resource> {
  static Node encode(const myres::CustomIntType4Resource& rhs) {
    auto value = rhs.value();
    YAML::Node value_node;
    value_node = value;
    return value_node;
  }

  static bool decode(const Node& node, myres::CustomIntType4Resource& rhs) {
    if (!node.IsScalar()) return false;

    try {
      rhs.value(node.as<int>());
      return true;
    } catch (...) { return false; }
  }
};

#endif /* WRAP_HOLOSCAN_AS_GXF_EXTENSION_PING_VARIANT_CUSTOM_NATIVE_RES_PING_VARIANT_CUSTOM_NATIVE_RES_HPP */  // NOLINT
