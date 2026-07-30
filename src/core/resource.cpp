/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/core/resource.hpp>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/component_spec.hpp>

namespace holoscan {

void Resource::initialize() {
  if (is_initialized_) {
    HOLOSCAN_LOG_DEBUG("Resource '{}' is already initialized. Skipping...", name());
    return;
  }

  HOLOSCAN_LOG_TRACE("Resource {}: calling Component::initialize()", name());
  Component::initialize();

  if (!spec_) {
    auto err_msg = fmt::format("No component spec for Resource '{}'", name());
    HOLOSCAN_LOG_ERROR(err_msg);
    throw std::runtime_error(err_msg);
  }

  HOLOSCAN_LOG_TRACE("Resource {}: calling set_parameters()", name());
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
  if (!spec_) {
    throw std::runtime_error(fmt::format("ComponentSpec of Resource '{}' has not been set", name_));
  }
  update_params_from_args(spec_->params());
}

void Resource::set_parameters() {
  HOLOSCAN_LOG_TRACE("Resource::set_parameters() for '{}': calling update_params_from_args()",
                     name());
  update_params_from_args();

  if (!spec_) {
    throw std::runtime_error(fmt::format("No component spec for Resource '{}'", name_));
  }

  // Set default values for unspecified arguments if the resource is native
  if (resource_type_ == ResourceType::kNative) {
    // Set only default parameter values
    std::vector<std::string> errors;
    for (auto& [key, param_wrap] : spec_->params()) {
      // If no value is specified, the default value will be used by setting an empty argument.
      Arg empty_arg("");
      try {
        ArgumentSetter::set_param(param_wrap, empty_arg);
      } catch (const std::exception& e) {
        std::string error_msg = fmt::format("Parameter '{}': {}", key, e.what());
        HOLOSCAN_LOG_ERROR("Resource '{}': failed to set default parameter - {}", name_, error_msg);
        errors.push_back(std::move(error_msg));
      }
    }

    if (!errors.empty()) {
      throw std::runtime_error(
          fmt::format("Resource '{}': failed to set {} default parameter(s):\n  - {}",
                      name_,
                      errors.size(),
                      fmt::join(errors, "\n  - ")));
    }
  }
}

}  // namespace holoscan
