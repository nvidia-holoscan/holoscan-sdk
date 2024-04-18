/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "holoscan/core/scheduler.hpp"

#include "holoscan/core/component_spec.hpp"
#include "holoscan/core/fragment.hpp"

namespace holoscan {

void Scheduler::initialize() {
  // Initialize the operator through the executor
  auto fragment_ptr = fragment();
  if (fragment_ptr) {
    auto& executor = fragment_ptr->executor();
    executor.initialize_scheduler(this);
  } else {
    HOLOSCAN_LOG_WARN("Scheduler::initialize() - Fragment is not set");
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
  for (const auto& r : resources_) { node["resources"].push_back(r.second->to_yaml_node()); }
  return node;
}

void Scheduler::reset_graph_entities() {
  HOLOSCAN_LOG_TRACE("Scheduler '{}'::reset_graph_entities", name_);
  for (auto& [_, resource] : resources_) {
    if (resource) {
      auto gxf_resource = std::dynamic_pointer_cast<holoscan::gxf::GXFResource>(resource);
      if (gxf_resource) { gxf_resource->reset_gxf_graph_entity(); }
      resource->reset_graph_entities();
    }
  }
  Component::reset_graph_entities();
}

}  // namespace holoscan
