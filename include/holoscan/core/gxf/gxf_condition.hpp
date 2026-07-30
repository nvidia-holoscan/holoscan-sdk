/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOSCAN_CORE_GXF_GXF_CONDITION_HPP
#define HOLOSCAN_CORE_GXF_GXF_CONDITION_HPP

#include <yaml-cpp/yaml.h>

#include <memory>
#include <string>

#include "../condition.hpp"
#include "./gxf_component.hpp"

#include <gxf/std/scheduling_terms.hpp>

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

  /**
   * @brief Reset any backend-specific state
   */
  void reset_backend_objects() override;
};

}  // namespace holoscan::gxf

#endif /* HOLOSCAN_CORE_GXF_GXF_CONDITION_HPP */
