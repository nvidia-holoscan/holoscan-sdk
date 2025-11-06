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

#include <stdexcept>
#include <string>

#include "holoscan/core/fragment.hpp"
#include "holoscan/core/operator_spec.hpp"

namespace holoscan {

const std::string& OperatorSpec::input_output_unique_id(const std::string& name) const {
  const bool in_exists = inputs_.find(name) != inputs_.end();
  const bool out_exists = outputs_.find(name) != outputs_.end();
  if (in_exists && out_exists) {
    throw std::runtime_error(fmt::format("Port name '{}' exists in both inputs and outputs", name));
  }
  if (in_exists) {
    return inputs_.at(name)->unique_id();
  }
  if (out_exists) {
    return outputs_.at(name)->unique_id();
  }
  throw std::out_of_range(fmt::format("Port name '{}' not found in inputs or outputs", name));
}

YAML::Node OperatorSpec::to_yaml_node() const {
  YAML::Node node = ComponentSpec::to_yaml_node();
  node["inputs"] = YAML::Node(YAML::NodeType::Sequence);
  for (const auto& i : inputs_) {
    node["inputs"].push_back(i.second->to_yaml_node());
  }
  node["outputs"] = YAML::Node(YAML::NodeType::Sequence);
  for (const auto& o : outputs_) {
    node["outputs"].push_back(o.second->to_yaml_node());
  }
  return node;
}

}  // namespace holoscan
