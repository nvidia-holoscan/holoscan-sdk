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
#include "holoscan/core/component.hpp"
#include "holoscan/core/fragment.hpp"

namespace holoscan {

YAML::Node Component::to_yaml_node() const {
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

std::string Component::description() const {
  YAML::Emitter emitter;
  emitter << to_yaml_node();
  return emitter.c_str();
}

}  // namespace holoscan
