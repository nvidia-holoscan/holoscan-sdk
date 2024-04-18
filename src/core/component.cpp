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
#include "holoscan/core/component.hpp"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "holoscan/core/fragment.hpp"

namespace holoscan {

YAML::Node ComponentBase::to_yaml_node() const {
  YAML::Node node;
  node["id"] = id_;
  node["name"] = name_;
  if (fragment_) {
    node["fragment"] = fragment_->name();
  } else {
    node["fragment"] = YAML::Null;
  }
  node["args"] = YAML::Node(YAML::NodeType::Sequence);
  for (const Arg& arg : args_) { node["args"].push_back(arg.to_yaml_node()); }
  return node;
}

std::string ComponentBase::description() const {
  YAML::Emitter emitter;
  emitter << to_yaml_node();
  return emitter.c_str();
}

void ComponentBase::update_params_from_args(
    std::unordered_map<std::string, ParameterWrapper>& params) {
  // Set arguments
  for (auto& arg : args_) {
    // Find if arg.name() is in spec_->params()
    if (params.find(arg.name()) == params.end()) {
      HOLOSCAN_LOG_WARN("Arg '{}' not found in spec_.params()", arg.name());
      continue;
    }

    // Set arg.value() to spec_->params()[arg.name()]
    auto& param_wrap = params[arg.name()];

    HOLOSCAN_LOG_TRACE("GXFOperator '{}':: setting argument '{}'", name_, arg.name());

    ArgumentSetter::set_param(param_wrap, arg);
  }
}

void Component::update_params_from_args() {
  update_params_from_args(spec_->params());
}

void ComponentBase::reset_graph_entities() {
  HOLOSCAN_LOG_TRACE("Component '{}'::reset_graph_entities", name_);
  // Note: Should NOT also be necessary to reset graph entities in spec_->params() as the
  // params are filled in via args.
  for (auto& arg : args_) {
    auto arg_type = arg.arg_type();
    auto element_type = arg_type.element_type();
    if ((element_type != ArgElementType::kResource) &&
        (element_type != ArgElementType::kCondition)) {
      continue;
    }
    auto container_type = arg_type.container_type();
    if ((container_type != ArgContainerType::kNative) &&
        (container_type != ArgContainerType::kVector)) {
      HOLOSCAN_LOG_ERROR(
          "Error setting GXF entity for argument '{}': Operator currently only supports scalar and "
          "vector containers for arguments of Condition or Resource type.",
          arg.name());
      continue;
    }
    if (element_type == ArgElementType::kCondition) {
      switch (container_type) {
        case ArgContainerType::kNative: {
          auto condition = std::any_cast<std::shared_ptr<Condition>>(arg.value());
          auto gxf_condition = std::dynamic_pointer_cast<gxf::GXFCondition>(condition);
          if (gxf_condition) { gxf_condition->reset_gxf_graph_entity(); }
        } break;
        case ArgContainerType::kVector: {
          auto conditions = std::any_cast<std::vector<std::shared_ptr<Condition>>>(arg.value());
          for (auto& condition : conditions) {
            auto gxf_condition = std::dynamic_pointer_cast<gxf::GXFCondition>(condition);
            if (gxf_condition) { gxf_condition->reset_gxf_graph_entity(); }
          }
        } break;
        default:
          break;
      }
    } else if (element_type == ArgElementType::kResource) {
      // Only GXF resources will use the GraphEntity
      switch (container_type) {
        case ArgContainerType::kNative: {
          auto resource = std::any_cast<std::shared_ptr<Resource>>(arg.value());
          auto gxf_resource = std::dynamic_pointer_cast<gxf::GXFResource>(resource);
          if (gxf_resource) {
            gxf_resource->reset_gxf_graph_entity();
            continue;
          }
        } break;
        case ArgContainerType::kVector: {
          auto resources = std::any_cast<std::vector<std::shared_ptr<Resource>>>(arg.value());
          for (auto& resource : resources) {
            auto gxf_resource = std::dynamic_pointer_cast<gxf::GXFResource>(resource);
            if (gxf_resource) {
              gxf_resource->reset_gxf_graph_entity();
              continue;
            }
          }
        } break;
        default:
          break;
      }
    }
  }
}

}  // namespace holoscan
