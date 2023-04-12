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
#include "holoscan/core/operator_spec.hpp"
#include "holoscan/core/fragment.hpp"

namespace holoscan {

YAML::Node OperatorSpec::to_yaml_node() const {
  YAML::Node node = ComponentSpec::to_yaml_node();
  node["inputs"] = YAML::Node(YAML::NodeType::Sequence);
  for (const auto& i : inputs_) { node["inputs"].push_back(i.first); }
  node["outputs"] = YAML::Node(YAML::NodeType::Sequence);
  for (const auto& o : outputs_) { node["outputs"].push_back(o.first); }
  return node;
}

}  // namespace holoscan
