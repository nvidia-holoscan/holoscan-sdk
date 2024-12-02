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

#include "holoscan/core/gxf/gxf_component.hpp"

#include <gxf/core/gxf.h>

#include <memory>
#include <string>
#include <vector>

#include "gxf/app/arg.hpp"
#include "gxf/app/graph_entity.hpp"
#include "holoscan/core/component_spec.hpp"
#include "holoscan/core/executors/gxf/gxf_executor.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/gxf/gxf_utils.hpp"

namespace holoscan::gxf {

namespace {

nvidia::gxf::Handle<nvidia::gxf::Component> add_component_to_graph_entity(
    gxf_context_t context, std::shared_ptr<nvidia::gxf::GraphEntity> graph_entity,
    const char* type_name, const char* name, const std::vector<nvidia::gxf::Arg>& arg_list = {}) {
  auto null_component = nvidia::gxf::Handle<nvidia::gxf::Component>::Null();
  gxf_tid_t derived_tid = GxfTidNull();
  gxf_tid_t base_tid = GxfTidNull();
  bool is_derived = false;
  gxf_result_t result;
  result = GxfComponentTypeId(context, type_name, &derived_tid);
  if (result != GXF_SUCCESS) { return null_component; }
  result = GxfComponentTypeId(context, "nvidia::gxf::Codelet", &base_tid);
  if (result != GXF_SUCCESS) { return null_component; }
  result = GxfComponentIsBase(context, derived_tid, base_tid, &is_derived);
  if (result != GXF_SUCCESS) { return null_component; }
  if (is_derived) { return graph_entity->addCodelet(type_name, name); }
  result = GxfComponentTypeId(context, "nvidia::gxf::SchedulingTerm", &base_tid);
  if (result != GXF_SUCCESS) { return null_component; }
  result = GxfComponentIsBase(context, derived_tid, base_tid, &is_derived);
  if (result != GXF_SUCCESS) { return null_component; }
  if (is_derived) { return graph_entity->addSchedulingTerm(type_name, name, arg_list); }

  // Commented out use of addTransmitter or addReceiver as this have additional restrictions or
  // defaults in GXF that we don't currently want for Holoscan. Transmitters and receivers will
  // just be added via addComponent below instead.

  // result = GxfComponentTypeId(context, "nvidia::gxf::Transmitter", &base_tid);
  // if (result != GXF_SUCCESS) { return null_component; }
  // result = GxfComponentIsBase(context, derived_tid, base_tid, &is_derived);
  // if (result != GXF_SUCCESS) { return null_component; }
  // bool omit_term = true;  // do not automatically add a scheduling term for rx/tx
  // if (is_derived) { return graph_entity->addTransmitter(type_name, name, arg_list, omit_term); }
  // result = GxfComponentTypeId(context, "nvidia::gxf::Receiver", &base_tid);
  // if (result != GXF_SUCCESS) { return null_component; }
  // result = GxfComponentIsBase(context, derived_tid, base_tid, &is_derived);
  // if (result != GXF_SUCCESS) { return null_component; }
  // if (is_derived) { return graph_entity->addReceiver(type_name, name, arg_list, omit_term); }

  result = GxfComponentTypeId(context, "nvidia::gxf::Clock", &base_tid);
  if (result != GXF_SUCCESS) { return null_component; }
  result = GxfComponentIsBase(context, derived_tid, base_tid, &is_derived);
  if (result != GXF_SUCCESS) { return null_component; }
  if (is_derived) { return graph_entity->addClock(type_name, name, arg_list); }
  result = GxfComponentTypeId(context, "nvidia::gxf::Component", &base_tid);
  if (result != GXF_SUCCESS) { return null_component; }
  result = GxfComponentIsBase(context, derived_tid, base_tid, &is_derived);
  if (result != GXF_SUCCESS) { return null_component; }
  if (is_derived) { return graph_entity->addComponent(type_name, name, arg_list); }
  HOLOSCAN_LOG_ERROR("type_name {} is not of Component type", type_name);
  return nvidia::gxf::Handle<nvidia::gxf::Component>::Null();
}

}  // namespace

void GXFComponent::gxf_initialize() {
  if (gxf_context_ == nullptr) {
    HOLOSCAN_LOG_ERROR("Initializing with null GXF context");
    return;
  }
  if (gxf_eid_ == 0) {
    HOLOSCAN_LOG_ERROR("Initializing with null GXF Entity");
    return;
  }

  const char* type_name = gxf_typename();
  // set the type id
  HOLOSCAN_GXF_CALL(GxfComponentTypeId(gxf_context_, type_name, &gxf_tid_));

  if (gxf_graph_entity_) {
    HOLOSCAN_LOG_TRACE("Initializing component '{}' in entity '{}' via GraphEntity",
                       gxf_cname_,
                       gxf_graph_entity_->eid());
    const char* name = gxf_cname_.c_str();
    auto handle = add_component_to_graph_entity(gxf_context_, gxf_graph_entity_, type_name, name);
    if (handle.is_null()) {
      HOLOSCAN_LOG_ERROR("Failed to add component '{}' of type: '{}'", name, type_name);
      return;
    }
    gxf_component_ = handle;
    gxf_cid_ = handle->cid();
  } else {
    // TODO(unknown): make sure all components always get initialized via GraphEntity so we can
    //       remove this code path. Some cases such as passing Arg of type Condition or
    //       Resource to make_operator will currently still use this code path.
    HOLOSCAN_LOG_TRACE(
        "Initializing component '{}' in entity '{}' via GxfComponentAdd", gxf_cname_, gxf_eid_);
    HOLOSCAN_GXF_CALL(
        GxfComponentAdd(gxf_context_, gxf_eid_, gxf_tid_, gxf_cname().c_str(), &gxf_cid_));
  }

  // TODO(unknown): replace gxf_cptr_ with Handle<Component>?
  HOLOSCAN_GXF_CALL(
      GxfComponentPointer(gxf_context_, gxf_cid_, gxf_tid_, reinterpret_cast<void**>(&gxf_cptr_)));
}

void GXFComponent::set_gxf_parameter(const std::string& component_name, const std::string& key,
                                     ParameterWrapper& param_wrap) {
  HOLOSCAN_LOG_TRACE("GXF component '{}' of type '{}': setting GXF parameter '{}'",
                     component_name,
                     gxf_typename(),
                     key);
  HOLOSCAN_GXF_CALL_WARN_MSG(::holoscan::gxf::GXFParameterAdaptor::set_param(
                                 gxf_context_, gxf_cid_, key.c_str(), param_wrap),
                             "component '{}':: failed to set GXF parameter '{}'",
                             component_name,
                             key);
  // TODO(unknown): handle error
}

std::string GXFComponent::gxf_entity_group_name() {
  const char* name;
  HOLOSCAN_GXF_CALL_FATAL(GxfEntityGroupName(gxf_context_, gxf_eid_, &name));
  return std::string{name};
}

gxf_uid_t GXFComponent::gxf_entity_group_id() {
  gxf_uid_t gid;
  HOLOSCAN_GXF_CALL_FATAL(GxfEntityGroupId(gxf_context_, gxf_eid_, &gid));
  return gid;
}

}  // namespace holoscan::gxf
