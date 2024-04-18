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

#include "holoscan/core/gxf/gxf_resource.hpp"

#include <any>
#include <functional>
#include <string>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>
#include <utility>

#include "holoscan/core/component_spec.hpp"
#include "holoscan/core/executors/gxf/gxf_executor.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/gxf/gxf_utils.hpp"

namespace holoscan::gxf {

GXFResource::GXFResource(const std::string& name, nvidia::gxf::Component* component) {
  id_ = component->cid();
  name_ = name;
  gxf_context_ = component->context();
  gxf_eid_ = component->eid();
  gxf_cid_ = component->cid();
  HOLOSCAN_GXF_CALL_FATAL(GxfComponentType(gxf_context_, gxf_cid_, &gxf_tid_));
  gxf_cname_ = name;
  gxf_cptr_ = component;
}

void GXFResource::initialize() {
  if (is_initialized_) {
    HOLOSCAN_LOG_DEBUG("GXFResource '{}' is already initialized. Skipping...", name());
    return;
  }

  // Set resource type before calling Resource::initialize()
  resource_type_ = holoscan::Resource::ResourceType::kGXF;

  Resource::initialize();
  auto& executor = fragment()->executor();
  auto gxf_executor = dynamic_cast<GXFExecutor*>(&executor);
  if (gxf_executor == nullptr) {
    HOLOSCAN_LOG_ERROR("GXFResource '{}' is not initialized with a GXFExecutor", name());
    return;
  }
  gxf_context_ = executor.context();

  // Create Entity for this Resource (without name so that new name is created) if gxf_eid_ is not
  // set.
  // Since now the resource is initialized lazily and bound to the entity of the first initialized
  // operator, the following code wouldn't be executed unless user explicitly calls
  // Resource::initialize() in Fragment::compose() method.
  if (gxf_eid_ == 0) {
    HOLOSCAN_LOG_WARN(
        "Resource '{}' of type '{}' initialized independent of a parent entity. This typically "
        "occurs if initialize() was called directly rather than allowing GXFExecutor to "
        "automatically initialize the resource.",
        gxf_cname_,
        gxf_typename());
    const GxfEntityCreateInfo entity_create_info = {nullptr, GXF_ENTITY_CREATE_PROGRAM_BIT};
    HOLOSCAN_GXF_CALL_FATAL(GxfCreateEntity(gxf_context_, &entity_create_info, &gxf_eid_));
  }

  // Set GXF component name
  std::string gxf_component_name = fmt::format("{}", name());
  gxf_cname(gxf_component_name);

  GXFComponent::gxf_initialize();

  // Set GXF component ID as the component ID
  id_ = gxf_cid_;

  if (!spec_) {
    HOLOSCAN_LOG_ERROR("No component spec for GXFResource '{}'", name());
    return;
  }

  update_params_from_args();

  static gxf_tid_t allocator_tid = GxfTidNull();  // issue 4336947

  // Set Handler parameters
  for (auto& [key, param_wrap] : spec_->params()) {
    // Issue 4336947: dev_id parameter for allocator needs to be handled manually
    bool dev_id_handled = false;
    if (key.compare(std::string("dev_id")) == 0) {
      if (!gxf_graph_entity_) {
        HOLOSCAN_LOG_ERROR(
            "`dev_id` parameter found, but gxf_graph_entity_ was not initialized so it could not "
            "be added to the entity group. This parameter will be ignored and default GPU device 0 "
            "will be used");
        continue;
      }
      gxf_tid_t derived_tid = GxfTidNull();
      bool is_derived = false;
      gxf_result_t tid_result;
      tid_result = GxfComponentTypeId(gxf_context_, gxf_typename(), &derived_tid);
      if (tid_result != GXF_SUCCESS) {
        HOLOSCAN_LOG_ERROR(
            "Unable to get component type id of '{}': {}", gxf_typename(), tid_result);
      }
      if (GxfTidIsNull(allocator_tid)) {
        tid_result = GxfComponentTypeId(gxf_context_, "nvidia::gxf::Allocator", &allocator_tid);
        if (tid_result != GXF_SUCCESS) {
          HOLOSCAN_LOG_ERROR("Unable to get component type id of 'nvidia::gxf::Allocator': {}",
                             tid_result);
        }
      }
      tid_result = GxfComponentIsBase(gxf_context_, derived_tid, allocator_tid, &is_derived);
      if (tid_result != GXF_SUCCESS) {
        HOLOSCAN_LOG_ERROR(
            "Unable to get determine if '{}' is derived from 'nvidia::gxf::Allocator': {}",
            gxf_typename(),
            tid_result);
      }
      if (is_derived) {
        HOLOSCAN_LOG_DEBUG(
            "The dev_id parameter is deprecated by GXF and will be removed from "
            "Holoscan SDK in the future.");

        auto dev_id_param = *std::any_cast<Parameter<int32_t>*>(param_wrap.value());
        if (dev_id_param.has_value()) {
          int32_t device_id = dev_id_param.get();

          auto devices = gxf_graph_entity_->findAll<nvidia::gxf::GPUDevice>();
          if (devices.size() > 0) {
            HOLOSCAN_LOG_WARN("Existing entity already has a GPUDevice resource");
          }

          // Create an EntityGroup to associate the GPUDevice with this resource
          std::string entity_group_name =
              fmt::format("{}_eid{}_dev_id{}_group", name(), gxf_eid_, device_id);
          auto entity_group_gid =
              ::holoscan::gxf::add_entity_group(gxf_context_, entity_group_name);

          // Add GPUDevice component to the same entity as this resource
          // TODO (GXF4): requested an addResource method to handle nvidia::gxf::ResourceBase types
          std::string device_component_name =
              fmt::format("{}_eid{}_gpu_device_id{}_component", name(), gxf_eid_, device_id);
          auto dev_handle =
              gxf_graph_entity_->addComponent("nvidia::gxf::GPUDevice",
                                              device_component_name.c_str(),
                                              {nvidia::gxf::Arg("dev_id", device_id)});
          if (dev_handle.is_null()) {
            HOLOSCAN_LOG_ERROR("Failed to create GPUDevice for resource '{}'", name_);
          } else {
            // TODO: warn and handle case if the resource was already in a different entity group

            // The GPUDevice and this resource have the same eid.
            // Make their eid is added to the newly created entity group.
            GXF_ASSERT_SUCCESS(GxfUpdateEntityGroup(gxf_context_, entity_group_gid, gxf_eid_));
          }
          dev_id_handled = true;
        }
      }
    }
    HOLOSCAN_LOG_TRACE(
        "GXF component '{}' of type '{}': setting GXF parameter '{}'", name_, gxf_typename(), key);
    if (!dev_id_handled) { set_gxf_parameter(name_, key, param_wrap); }
    // TODO: handle error
    HOLOSCAN_LOG_TRACE("GXFResource '{}':: setting GXF parameter '{}'", name(), key);
  }

  is_initialized_ = true;
}

void GXFResource::add_to_graph_entity(Operator* op) {
  if (gxf_context_ == nullptr) {
    // cannot reassign to a different graph entity if the resource was already initialized with GXF
    if (gxf_graph_entity_ && is_initialized_) { return; }

    gxf_graph_entity_ = op->graph_entity();
    fragment_ = op->fragment();
    if (gxf_graph_entity_) {
      gxf_context_ = gxf_graph_entity_->context();
      gxf_eid_ = gxf_graph_entity_->eid();
    }
  }
  this->initialize();
}

}  // namespace holoscan::gxf
