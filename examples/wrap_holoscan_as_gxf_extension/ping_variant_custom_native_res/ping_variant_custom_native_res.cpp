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

using namespace holoscan;

namespace myres {

void PingVarCustomNativeRes::initialize() {
  HOLOSCAN_LOG_INFO("PingVarCustomNativeRes::initialize() called.");
  // Register the custom type with Holoscan.
  register_converter<myres::CustomIntType4Resource>();

  // Call the base class initialize.
  holoscan::Resource::initialize();
}

void PingVarCustomNativeRes::setup(ComponentSpec& spec) {
  HOLOSCAN_LOG_INFO("PingVarCustomNativeRes::setup() called.");
  spec.param(custom_int_type_,
             "custom_int_type",
             "CustomIntType4Resource",
             "This is a sample parameter for a custom type.");
  spec.param(float_type_, "float_type", "float", "This is a sample parameter for a float type.");
}

int PingVarCustomNativeRes::get_custom_int() {
  return custom_int_type_.get().value();
}

float PingVarCustomNativeRes::get_float() {
  return float_type_.get();
}

}  // namespace myres
