/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "holoscan/core/condition.hpp"

#include <memory>
#include <utility>
#include "holoscan/core/component_spec.hpp"
#include "holoscan/core/resource.hpp"

namespace holoscan {

YAML::Node Condition::to_yaml_node() const {
  YAML::Node node = Component::to_yaml_node();
  if (spec_) {
    node["spec"] = spec_->to_yaml_node();
  } else {
    node["spec"] = YAML::Null;
  }
  return node;
}

void Condition::add_arg(const std::shared_ptr<Resource>& arg) {
  if (resources_.find(arg->name()) != resources_.end()) {
    HOLOSCAN_LOG_ERROR(
        "Resource '{}' already exists in the condition. Please specify a unique "
        "name when creating a Resource instance.",
        arg->name());
  } else {
    resources_[arg->name()] = arg;
  }
}

void Condition::Condition::add_arg(std::shared_ptr<Resource>&& arg) {
  if (resources_.find(arg->name()) != resources_.end()) {
    HOLOSCAN_LOG_ERROR(
        "Resource '{}' already exists in the condition. Please specify a unique "
        "name when creating a Resource instance.",
        arg->name());
  } else {
    resources_[arg->name()] = std::move(arg);
  }
}

}  // namespace holoscan
