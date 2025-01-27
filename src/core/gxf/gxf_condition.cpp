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

#include "holoscan/core/gxf/gxf_condition.hpp"

#include <gxf/core/gxf.h>

#include <memory>
#include <string>
#include <vector>

#include "holoscan/core/component_spec.hpp"
#include "holoscan/core/executors/gxf/gxf_executor.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/gxf/gxf_utils.hpp"

namespace holoscan::gxf {

GXFCondition::GXFCondition(const std::string& name, nvidia::gxf::SchedulingTerm* term) {
  if (term == nullptr) {
    std::string err_msg =
        fmt::format("SchedulingTerm pointer is null. Cannot initialize GXFCondition '{}'", name);
    HOLOSCAN_LOG_ERROR(err_msg);
    throw std::runtime_error(err_msg);
  }
  id_ = term->cid();
  name_ = name;
  gxf_context_ = term->context();
  gxf_eid_ = term->eid();
  gxf_cid_ = term->cid();
  HOLOSCAN_GXF_CALL_FATAL(GxfComponentType(gxf_context_, gxf_cid_, &gxf_tid_));
  gxf_cname_ = name;
  gxf_cptr_ = term;
}

void GXFCondition::initialize() {
  if (is_initialized_) {
    HOLOSCAN_LOG_DEBUG("GXFCondition '{}' is already initialized. Skipping...", name());
    return;
  }

  // Set condition type before calling Condition::initialize()
  condition_type_ = holoscan::Condition::ConditionComponentType::kGXF;
  auto& executor = fragment()->executor();
  auto gxf_executor = dynamic_cast<GXFExecutor*>(&executor);
  if (gxf_executor == nullptr) {
    HOLOSCAN_LOG_ERROR("GXFCondition '{}' is not initialized with a GXFExecutor", name());
    return;
  }
  gxf_context_ = executor.context();

  // Set GXF component name
  std::string gxf_component_name = fmt::format("{}", name());
  gxf_cname(gxf_component_name);

  GXFComponent::gxf_initialize();

  // Set GXF component ID as the component ID
  id_ = gxf_cid_;

  if (!spec_) {
    HOLOSCAN_LOG_ERROR("No component spec for GXFCondition '{}'", name());
    return;
  }

  // Set arguments
  update_params_from_args();

  // Set Handler parameters
  for (auto& [key, param_wrap] : spec_->params()) { set_gxf_parameter(name_, key, param_wrap); }
  is_initialized_ = true;
  Condition::initialize();
}

void GXFCondition::add_to_graph_entity(Operator* op) {
  if (gxf_context_ == nullptr) {
    // cannot reassign to a different graph entity if the condition was already initialized with GXF
    if (gxf_graph_entity_ && is_initialized_) { return; }

    gxf_graph_entity_ = op->graph_entity();
    fragment_ = op->fragment();
    if (gxf_graph_entity_) {
      gxf_context_ = gxf_graph_entity_->context();
      gxf_eid_ = gxf_graph_entity_->eid();
    }
  }
  this->initialize();
}

YAML::Node GXFCondition::to_yaml_node() const {
  YAML::Node node = Condition::to_yaml_node();
  node["gxf_eid"] = YAML::Node(gxf_eid());
  node["gxf_cid"] = YAML::Node(gxf_cid());
  node["gxf_typename"] = YAML::Node(gxf_typename());
  return node;
}

}  // namespace holoscan::gxf
