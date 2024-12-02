/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef HOLOSCAN_CORE_GXF_GXF_CONDITION_HPP
#define HOLOSCAN_CORE_GXF_GXF_CONDITION_HPP

#include <yaml-cpp/yaml.h>

#include <memory>
#include <string>

#include "../condition.hpp"
#include "./gxf_component.hpp"

#include "gxf/std/scheduling_terms.hpp"

namespace holoscan::gxf {

class GXFCondition : public holoscan::Condition, public gxf::GXFComponent {
 public:
  HOLOSCAN_CONDITION_FORWARD_ARGS_SUPER(GXFCondition, holoscan::Condition)
  GXFCondition() = default;
  GXFCondition(const std::string& name, nvidia::gxf::SchedulingTerm* term);

  void initialize() override;

  void add_to_graph_entity(Operator* op);
  void add_to_graph_entity(Fragment* fragment,
                           std::shared_ptr<nvidia::gxf::GraphEntity> graph_entity);

  /**
   * @brief Get a YAML representation of the condition.
   *
   * @return YAML node including type and spec of the condition in addition to the base component
   * properties.
   */
  YAML::Node to_yaml_node() const override;
};

}  // namespace holoscan::gxf

#endif /* HOLOSCAN_CORE_GXF_GXF_CONDITION_HPP */
