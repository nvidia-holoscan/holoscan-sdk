/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
  virtual const char* gxf_typename() const = 0;

  /**
   * @brief Get a YAML representation of the network context.
   *
   * @return YAML node including type, specs, resources of the network context in addition
   * to the base component properties.
   */
  YAML::Node to_yaml_node() const override;

 protected:
  // Make Fragment a friend class so it can call reset_graph_entities
  friend class holoscan::Fragment;

  /// Set the parameters based on defaults (sets GXF parameters for GXF operators)
  void set_parameters() override;

  /// Reset the GXF GraphEntity of all components associated with the network context
  void reset_graph_entities() override;
};

}  // namespace holoscan::gxf

#endif /* HOLOSCAN_CORE_GXF_GXF_NETWORK_CONTEXT_HPP */
