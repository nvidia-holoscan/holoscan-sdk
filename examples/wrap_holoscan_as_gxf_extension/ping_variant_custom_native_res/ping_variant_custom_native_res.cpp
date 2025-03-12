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

#include "ping_variant_custom_native_res.hpp"

#include <string>
#include <vector>

using namespace holoscan;

// Register the custom type with Holoscan.
// NOLINTBEGIN(altera-struct-pack-align)
template <>
struct YAML::convert<void*> {
  static Node encode([[maybe_unused]] const void*& rhs) {
    throw std::runtime_error("void* is unsupported in YAML");
  }

  static bool decode([[maybe_unused]] const Node& node, [[maybe_unused]] void*& rhs) {
    throw std::runtime_error("void* is unsupported in YAML");
  }
};
// NOLINTEND(altera-struct-pack-align)

namespace myres {

void PingVarCustomNativeRes::initialize() {
  HOLOSCAN_LOG_INFO("PingVarCustomNativeRes::initialize() called.");
  // Register the custom type with Holoscan.
  register_converter<myres::CustomIntType4Resource>();
  // Call the base class initialize.
  holoscan::Resource::initialize();
  HOLOSCAN_LOG_INFO(
      "PingVarCustomNativeRes: custom_int_type={}, float_type={}, numeric={}, "
      "numeric_array={}, optional_numeric={}, optional_numeric_array={}, "
      "boolean={}, optional_void_ptr={}, string={}, optional_resource={}",
      custom_int_type_.get().value(),
      float_type_.get(),
      numeric_.get(),
      numeric_array_.get(),
      optional_numeric_.has_value() ? optional_numeric_.get() : -1,
      optional_numeric_array_.has_value() ? optional_numeric_array_.get() : std::vector<int>{},
      boolean_.get(),
      optional_void_ptr_.has_value() ? "non-null" : "null",
      string_.get(),
      optional_resource_.has_default_value() ? "non-null" : "null");
}

void PingVarCustomNativeRes::setup(ComponentSpec& spec) {
  register_converter<void*>();
  HOLOSCAN_LOG_INFO("PingVarCustomNativeRes::setup() called.");
  spec.param(custom_int_type_,
             "custom_int_type",
             "CustomIntType4Resource",
             "This is a sample parameter for a custom type.");
  spec.param(float_type_, "float_type", "float", "This is a sample parameter for a float type.");
  spec.param(numeric_, "numeric", "numeric", "numeric", 0);
  spec.param(numeric_array_,
             "numeric_array",
             "numeric array",
             "numeric array",
             std::vector<float>{0, 1.5, 2.5, 3.0, 4.0});
  spec.param(optional_numeric_,
             "optional_numeric",
             "optional numeric",
             "optional numeric",
             holoscan::ParameterFlag::kOptional);
  spec.param(optional_numeric_array_,
             "optional_numeric_array",
             "optional_numeric array",
             "optional_numeric array",
             holoscan::ParameterFlag::kOptional);
  spec.param(boolean_, "boolean", "boolean", "boolean");
  spec.param(optional_void_ptr_,
             "void_ptr",
             "optional void pointer",
             "optional void pointer",
             holoscan::ParameterFlag::kOptional);
  spec.param(string_, "string", "string", "string", std::string("test text"));
  spec.param(optional_resource_,
             "optional_resource",
             "optional resource",
             "optional resource",
             holoscan::ParameterFlag::kOptional);
}

int PingVarCustomNativeRes::get_custom_int() {
  return custom_int_type_.get().value();
}

float PingVarCustomNativeRes::get_float() {
  return float_type_.get();
}

}  // namespace myres
