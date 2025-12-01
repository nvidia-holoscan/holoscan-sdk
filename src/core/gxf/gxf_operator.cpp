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

#include <string>
#include <vector>

#include "holoscan/core/gxf/gxf_operator.hpp"

namespace holoscan::ops {

std::string GXFOperator::gxf_entity_group_name() const {
  const char* name;
  HOLOSCAN_GXF_CALL_FATAL(GxfEntityGroupName(gxf_context_, gxf_eid_, &name));
  return std::string{name};
}

void GXFOperator::initialize() {
  // Call base class initialize function.
  Operator::initialize();
}

gxf_uid_t GXFOperator::add_codelet_to_graph_entity() {
  HOLOSCAN_LOG_TRACE("calling graph_entity()->addCodelet for {}", name_);
  if (!graph_entity_) {
    throw std::runtime_error("graph entity is not initialized");
  }
  codelet_handle_ = graph_entity_->addCodelet(gxf_typename(), name_.c_str());
  if (!codelet_handle_) {
    throw std::runtime_error("Failed to add codelet of type " + std::string(gxf_typename()));
  }
  gxf_uid_t codelet_cid = codelet_handle_->cid();
  gxf_eid_ = graph_entity_->eid();
  gxf_cid_ = codelet_cid;
  gxf_context_ = graph_entity_->context();
  HOLOSCAN_LOG_TRACE("\tadded codelet with cid = {}", codelet_handle_->cid());
  return codelet_cid;
}

void GXFOperator::set_parameters() {
  update_params_from_args();

  if (!spec_) {
    throw std::runtime_error(fmt::format("No component spec for GXFOperator '{}'", name_));
  }

  // Set Handler parameters
  std::vector<std::string> errors;
  for (auto& [key, param_wrap] : spec_->params()) {
    HOLOSCAN_LOG_TRACE("GXFOperator '{}':: setting GXF parameter '{}'", name_, key);
    gxf_result_t result = ::holoscan::gxf::GXFParameterAdaptor::set_param(
        gxf_context_, gxf_cid_, key.c_str(), param_wrap);
    if (result != GXF_SUCCESS) {
      std::string error_msg = fmt::format("Parameter '{}': {} (error code: {})",
                                          key,
                                          GxfResultStr(result),
                                          static_cast<int>(result));
      HOLOSCAN_LOG_ERROR("GXFOperator '{}': failed to set GXF parameter - {}", name_, error_msg);
      errors.push_back(error_msg);
    }
  }

  if (!errors.empty()) {
    throw std::runtime_error(
        fmt::format("GXFOperator '{}' (type '{}'): failed to set {} GXF parameter(s):\n  - {}",
                    name_,
                    gxf_typename(),
                    errors.size(),
                    fmt::join(errors, "\n  - ")));
  }
}

YAML::Node GXFOperator::to_yaml_node() const {
  YAML::Node node = Operator::to_yaml_node();
  node["gxf_eid"] = YAML::Node(gxf_eid());
  node["gxf_cid"] = YAML::Node(gxf_cid());
  node["gxf_typename"] = YAML::Node(gxf_typename());
  node["gxf_entity_group_name"] = YAML::Node(gxf_entity_group_name());
  return node;
}

}  // namespace holoscan::ops
