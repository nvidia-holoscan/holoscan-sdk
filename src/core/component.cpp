/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
  for (const Arg& arg : args_) {
    node["args"].push_back(arg.to_yaml_node());
  }
  return node;
}

std::string ComponentBase::description() const {
  YAML::Emitter emitter;
  emitter << to_yaml_node();
  return emitter.c_str();
}

void ComponentBase::update_params_from_args(
    std::unordered_map<std::string, ParameterWrapper>& params) {
  HOLOSCAN_LOG_TRACE("ComponentBase::update_params_from_args() for '{}':", name_);
  // Set arguments
  std::vector<std::string> errors;
  std::vector<std::string> unknown_args;
  for (auto& arg : args_) {
    // Find if arg.name() is in spec_->params()
    if (params.find(arg.name()) == params.end()) {
      auto msg_buf = fmt::memory_buffer();
      for (const auto& kv : params) {
        if (&kv == &(*params.begin())) {
          fmt::format_to(std::back_inserter(msg_buf), "{}", kv.first);
        } else {
          fmt::format_to(std::back_inserter(msg_buf), ", {}", kv.first);
        }
      }
      HOLOSCAN_LOG_WARN(
          "component '{}': Arg '{}' not found in spec_.params(). The defined parameters are "
          "({:.{}}).",
          name_,
          arg.name(),
          msg_buf.data(),
          msg_buf.size());
      unknown_args.push_back(arg.name());
      continue;
    }

    // Set arg.value() to spec_->params()[arg.name()]
    auto& param_wrap = params[arg.name()];

    HOLOSCAN_LOG_TRACE("\tComponent '{}':: setting argument '{}'", name_, arg.name());

    HOLOSCAN_LOG_TRACE("\tArgumentSetter::set_param() for argument  '{}':", arg.name());
    try {
      ArgumentSetter::set_param(param_wrap, arg);
    } catch (const std::exception& e) {
      std::string error_msg = fmt::format("Argument '{}': {}", arg.name(), e.what());
      HOLOSCAN_LOG_ERROR("Component '{}': failed to set argument - {}", name_, error_msg);
      errors.push_back(error_msg);
    }
  }

  if (!errors.empty()) {
    // Append unknown args to the aggregated error for completeness
    if (!unknown_args.empty()) {
      for (const auto& u : unknown_args) {
        errors.push_back(fmt::format("Unknown argument '{}': not found in component spec", u));
      }
    }
    throw std::runtime_error(fmt::format("Component '{}': failed to set {} argument(s):\n  - {}",
                                         name_,
                                         errors.size(),
                                         fmt::join(errors, "\n  - ")));
  }
}

void Component::update_params_from_args() {
  if (!spec_) {
    throw std::runtime_error(fmt::format("No component spec for GXFNetworkContext '{}'", name_));
  }
  update_params_from_args(spec_->params());
}

void ComponentBase::reset_backend_objects() {
  HOLOSCAN_LOG_TRACE("Component '{}'::reset_backend_objects", name_);
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
          if (condition) {
            condition->reset_backend_objects();
          }
        } break;
        case ArgContainerType::kVector: {
          auto conditions = std::any_cast<std::vector<std::shared_ptr<Condition>>>(arg.value());
          for (auto& condition : conditions) {
            if (condition) {
              condition->reset_backend_objects();
            }
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
          if (resource) {
            resource->reset_backend_objects();
          }
        } break;
        case ArgContainerType::kVector: {
          auto resources = std::any_cast<std::vector<std::shared_ptr<Resource>>>(arg.value());
          for (auto& resource : resources) {
            if (resource) {
              resource->reset_backend_objects();
            }
          }
        } break;
        default:
          break;
      }
    }
  }
}

void ComponentBase::fragment(Fragment* frag) {
  fragment_ = frag;
}

void ComponentBase::service_provider(FragmentServiceProvider* provider) {
  service_provider_ = provider;
}

}  // namespace holoscan
