/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "holoscan/core/component_spec.hpp"
#include "holoscan/core/fragment.hpp"

namespace holoscan {

YAML::Node ComponentSpec::to_yaml_node() const {
  YAML::Node node;
  if (fragment_) {
    node["fragment"] = fragment_->name();
  } else {
    node["fragment"] = YAML::Null;
  }
  node["params"] = YAML::Node(YAML::NodeType::Sequence);
  for (auto& [name, wrapper] : params_) {
    const std::string& type = wrapper.arg_type().to_string();
    auto param = static_cast<Parameter<void*>*>(wrapper.storage_ptr());
    YAML::Node param_node;
    param_node["name"] = name;
    param_node["type"] = type;
    param_node["description"] = param->description();
    node["params"].push_back(param_node);
  }
  return node;
}

std::string ComponentSpec::description() const {
  YAML::Emitter emitter;
  emitter << to_yaml_node();
  return emitter.c_str();
}

}  // namespace holoscan
