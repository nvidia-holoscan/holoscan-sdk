/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "holoscan/core/resource.hpp"

#include "holoscan/core/component_spec.hpp"

namespace holoscan {

void Resource::initialize() {
  if (is_initialized_) {
    HOLOSCAN_LOG_DEBUG("Resource '{}' is already initialized. Skipping...", name());
    return;
  }

  Component::initialize();

  if (!spec_) {
    HOLOSCAN_LOG_ERROR("No component spec for Resource '{}'", name());
    return;
  }

  set_parameters();

  is_initialized_ = true;
}

YAML::Node Resource::to_yaml_node() const {
  YAML::Node node = Component::to_yaml_node();
  node["type"] = (resource_type_ == ResourceType::kGXF) ? "GXF" : "native";
  if (spec_) {
    node["spec"] = spec_->to_yaml_node();
  } else {
    node["spec"] = YAML::Null;
  }
  return node;
}

void Resource::update_params_from_args() {
  update_params_from_args(spec_->params());
}

void Resource::set_parameters() {
  update_params_from_args();

  // Set default values for unspecified arguments if the resource is native
  if (resource_type_ == ResourceType::kNative) {
    // Set only default parameter values
    for (auto& [key, param_wrap] : spec_->params()) {
      // If no value is specified, the default value will be used by setting an empty argument.
      Arg empty_arg("");
      ArgumentSetter::set_param(param_wrap, empty_arg);
    }
  }
}

}  // namespace holoscan
