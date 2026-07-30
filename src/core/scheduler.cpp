/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/core/scheduler.hpp>

#include <stdexcept>
#include <string>

#include <holoscan/core/component_spec.hpp>
#include <holoscan/core/fragment.hpp>

namespace holoscan {

void Scheduler::initialize() {
  // Initialize the operator through the executor
  auto fragment_ptr = fragment();
  if (fragment_ptr) {
    auto& executor = fragment_ptr->executor();
    if (!executor.initialize_scheduler(this)) {
      auto err_msg = std::string("Failed to initialize scheduler in executor");
      HOLOSCAN_LOG_ERROR(err_msg);
      throw std::runtime_error(err_msg);
    }
  } else {
    auto err_msg = std::string("Scheduler::initialize() - Fragment is not set");
    HOLOSCAN_LOG_ERROR(err_msg);
    throw std::runtime_error(err_msg);
  }
}

YAML::Node Scheduler::to_yaml_node() const {
  YAML::Node node = Component::to_yaml_node();
  if (spec_) {
    node["spec"] = spec_->to_yaml_node();
  } else {
    node["spec"] = YAML::Null;
  }
  node["resources"] = YAML::Node(YAML::NodeType::Sequence);
  for (const auto& r : resources_) {
    node["resources"].push_back(r.second->to_yaml_node());
  }
  return node;
}

void Scheduler::reset_backend_objects() {
  HOLOSCAN_LOG_TRACE("Scheduler '{}'::reset_backend_objects", name_);
  for (auto& [_, resource] : resources_) {
    if (resource) {
      resource->reset_backend_objects();
    }
  }
  // Clear the resources map to enable subsequent executions
  resources_.clear();
  Component::reset_backend_objects();
}

}  // namespace holoscan
