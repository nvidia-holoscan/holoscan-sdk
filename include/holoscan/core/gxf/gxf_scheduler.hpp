/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOSCAN_CORE_GXF_GXF_SCHEDULER_HPP
#define HOLOSCAN_CORE_GXF_GXF_SCHEDULER_HPP

#include <yaml-cpp/yaml.h>

#include <memory>
#include <string>
#include <utility>

#include <gxf/std/clock.hpp>
#include "../resources/gxf/clock.hpp"
#include "../scheduler.hpp"
#include "./gxf_component.hpp"

namespace holoscan::gxf {

// note: in GXF there is also a System class that inherits from Component
//       and is the parent of Scheduler
class GXFScheduler : public holoscan::Scheduler, public GXFComponent {
 public:
  HOLOSCAN_SCHEDULER_FORWARD_ARGS_SUPER(GXFScheduler, holoscan::Scheduler)
  GXFScheduler() = default;

  /**
   * @brief Get the type name of the GXF scheduler.
   *
   * The returned string is the type name of the GXF scheduler and is used to
   * create the GXF scheduler.
   *
   * Example: "nvidia::holoscan::GreedyScheduler"
   *
   * @return The type name of the GXF scheduler.
   */
  const char* gxf_typename() const override = 0;

  /**
   * @brief Get the GXF Clock pointer.
   *
   * @return The GXF clock pointer used by the scheduler.
   */
  virtual nvidia::gxf::Clock* gxf_clock();

  /**
   * @brief Get a YAML representation of the scheduler.
   *
   * @return YAML node including type, specs, and resources of the scheduler in addition
   * to the base component properties.
   */
  YAML::Node to_yaml_node() const override;

  /// Reset any backend-specific objects
  void reset_backend_objects() override;

  /// Set the parameters based on defaults (sets GXF parameters for GXF operators)
  void set_parameters() override;

 private:
  // raw pointer to the nvidia::gxf::Clock instance used by the scheduler
  virtual void* clock_gxf_cptr() const = 0;
};

}  // namespace holoscan::gxf

#endif /* HOLOSCAN_CORE_GXF_GXF_SCHEDULER_HPP */
