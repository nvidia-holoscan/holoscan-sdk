/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef HOLOSCAN_CORE_GXF_GXF_RESOURCE_HPP
#define HOLOSCAN_CORE_GXF_GXF_RESOURCE_HPP

#include <yaml-cpp/yaml.h>

#include <memory>
#include <string>

#include <gxf/core/component.hpp>
#include <gxf/std/resources.hpp>

#include "../resource.hpp"
#include "./gxf_component.hpp"
#include "./gxf_utils.hpp"

namespace holoscan::gxf {

class GXFResource : public holoscan::Resource, public gxf::GXFComponent {
 public:
  HOLOSCAN_RESOURCE_FORWARD_ARGS_SUPER(GXFResource, holoscan::Resource)
  GXFResource() = default;
  GXFResource(const std::string& name, nvidia::gxf::Component* component);

  void initialize() override;

 protected:
  // Make GXFExecutor a friend class so it can call protected initialization methods
  friend class holoscan::gxf::GXFExecutor;
  // Operator::initialize_resources() and Fragment::make_thread_pool call add_to_graph_entity()
  friend class holoscan::Operator;
  friend class holoscan::Fragment;

  virtual void add_to_graph_entity(Operator* op);
  void add_to_graph_entity(Fragment* fragment,
                           std::shared_ptr<nvidia::gxf::GraphEntity> graph_entity);

  /**
   * @brief Get a YAML representation of the resource.
   *
   * @return YAML node including type and specs of the resource in addition to the base
   * component properties.
   */
  YAML::Node to_yaml_node() const override;

  /**
   * This method is invoked by `GXFResource::initialize()`.
   * By overriding this method, we can modify how GXF Codelet's parameters are set from the
   * arguments.
   */
  void set_parameters() override;
  bool handle_dev_id(std::optional<int32_t>& dev_id_value);
  /// The GXF type name (used for GXFComponentResource)
  std::string gxf_typename_ = "unknown_gxf_typename";

  /**
   * @brief Reset any backend-specific state
   */
  void reset_backend_objects() override;
};

/**
 * @brief Base class to be used with Resource types that inherit from nvidia::gxf::ResourceBase
 *
 * Resource components that can be registered with GXF's Registrar::resource should inherit from
 * this class. Any other resource components should just use GXFResource directly.
 */
class GXFSystemResourceBase : public GXFResource {
 public:
  HOLOSCAN_RESOURCE_FORWARD_ARGS_SUPER(GXFSystemResourceBase, GXFResource)
  GXFSystemResourceBase() = default;
  GXFSystemResourceBase(const std::string& name, nvidia::gxf::ResourceBase* component);

  const char* gxf_typename() const override { return "nvidia::gxf::ResourceBase"; }
};

}  // namespace holoscan::gxf

#endif /* HOLOSCAN_CORE_GXF_GXF_RESOURCE_HPP */
