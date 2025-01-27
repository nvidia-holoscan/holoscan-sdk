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

#include "holoscan/core/condition.hpp"

#include <memory>
#include <string>
#include <utility>

#include "holoscan/core/component_spec.hpp"
#include "holoscan/core/operator.hpp"
#include "holoscan/core/resource.hpp"

namespace holoscan {

void Condition::initialize() {
  if (is_initialized_) {
    HOLOSCAN_LOG_DEBUG("Condition '{}' is already initialized. Skipping...", name());
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

YAML::Node Condition::to_yaml_node() const {
  YAML::Node node = Component::to_yaml_node();
  node["component_type"] = (condition_type_ == ConditionComponentType::kGXF) ? "GXF" : "native";
  if (spec_) {
    node["spec"] = spec_->to_yaml_node();
  } else {
    node["spec"] = YAML::Null;
  }
  return node;
}

void Condition::add_arg(const std::shared_ptr<Resource>& arg) {
  if (resources_.find(arg->name()) != resources_.end()) {
    HOLOSCAN_LOG_ERROR(
        "Resource '{}' already exists in the condition. Please specify a unique "
        "name when creating a Resource instance.",
        arg->name());
  } else {
    resources_[arg->name()] = arg;
  }
}

void Condition::add_arg(std::shared_ptr<Resource>&& arg) {
  if (resources_.find(arg->name()) != resources_.end()) {
    HOLOSCAN_LOG_ERROR(
        "Resource '{}' already exists in the condition. Please specify a unique "
        "name when creating a Resource instance.",
        arg->name());
  } else {
    resources_[arg->name()] = std::move(arg);
  }
}

void Condition::update_params_from_args() {
  update_params_from_args(spec_->params());
}

void Condition::set_parameters() {
  update_params_from_args();

  // Set default values for unspecified arguments if the condition is native
  if (condition_type_ == ConditionComponentType::kNative) {
    // Set only default parameter values
    for (auto& [key, param_wrap] : spec_->params()) {
      // If no value is specified, the default value will be used by setting an empty argument.
      Arg empty_arg("");
      ArgumentSetter::set_param(param_wrap, empty_arg);
    }
  }
}

std::optional<std::shared_ptr<Receiver>> Condition::receiver(const std::string& port_name) {
  if (op_ == nullptr) {
    HOLOSCAN_LOG_ERROR(
        "Could not retrieve Receiver because condition '{}' is not associated with an Operator",
        name_);
    return std::nullopt;
  }
  return op_->receiver(port_name);
}

std::optional<std::shared_ptr<Transmitter>> Condition::transmitter(const std::string& port_name) {
  if (op_ == nullptr) {
    HOLOSCAN_LOG_ERROR(
        "Could not retrieve Receiver because condition '{}' is not associated with an Operator",
        name_);
    return std::nullopt;
  }
  return op_->transmitter(port_name);
}

}  // namespace holoscan
