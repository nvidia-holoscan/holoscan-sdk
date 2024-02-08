/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "holoscan/core/io_spec.hpp"

#include <string>
#include <unordered_map>

using std::string_literals::operator""s;

namespace holoscan {

YAML::Node IOSpec::to_yaml_node() const {
  YAML::Node node;

  std::unordered_map<IOType, std::string> iotype_namemap{
      {IOType::kInput, "kInput"s},
      {IOType::kOutput, "kOutput"s},
  };

  std::unordered_map<ConnectorType, std::string> connectortype_namemap{
      {ConnectorType::kDefault, "kDefault"s},
      {ConnectorType::kDoubleBuffer, "kDoubleBuffer"s},
      {ConnectorType::kUCX, "kUCX"s},
  };

  node["name"] = name();
  node["io_type"] = iotype_namemap[io_type()];
  node["typeinfo_name"] = std::string{typeinfo()->name()};
  node["connector_type"] = connectortype_namemap[connector_type()];
  auto conn = connector();
  if (conn) { node["connector"] = conn->to_yaml_node(); }
  node["conditions"] = YAML::Node(YAML::NodeType::Sequence);
  for (const auto& c : conditions_) {
    if (c.second) { node["conditions"].push_back(c.second->to_yaml_node()); }
  }
  return node;
}

std::string IOSpec::description() const {
  YAML::Emitter emitter;
  emitter << to_yaml_node();
  return emitter.c_str();
}

}  // namespace holoscan
