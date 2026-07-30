/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOSCAN_CORE_GXF_GXF_NETWORK_CONTEXT_HPP
#define HOLOSCAN_CORE_GXF_GXF_NETWORK_CONTEXT_HPP

#include <yaml-cpp/yaml.h>

#include <memory>
#include <string>
#include <utility>

#include "../network_context.hpp"
#include "./gxf_component.hpp"

namespace holoscan::gxf {

// note: in GXF there is also a System class that inherits from Component
//       and is the parent of NetworkContext
class GXFNetworkContext : public holoscan::NetworkContext, public GXFComponent {
 public:
  HOLOSCAN_NETWORK_CONTEXT_FORWARD_ARGS_SUPER(GXFNetworkContext, holoscan::NetworkContext)
  GXFNetworkContext() = default;

  /**
   * @brief Get the type name of the GXF network context.
   *
   * The returned string is the type name of the GXF network context and is used to
   * create the GXF network context.
   *
   * Example: "nvidia::holoscan::UcxContext"
   *
   * @return The type name of the GXF network context.
   */
  const char* gxf_typename() const override = 0;

  /**
   * @brief Get a YAML representation of the network context.
   *
   * @return YAML node including type, specs, resources of the network context in addition
   * to the base component properties.
   */
  YAML::Node to_yaml_node() const override;

  /// Reset any backend-specific objects
  void reset_backend_objects() override;

  /// Set the parameters based on defaults (sets GXF parameters for GXF operators)
  void set_parameters() override;
};

}  // namespace holoscan::gxf

#endif /* HOLOSCAN_CORE_GXF_GXF_NETWORK_CONTEXT_HPP */
