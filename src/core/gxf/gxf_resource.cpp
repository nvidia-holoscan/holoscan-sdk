/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

  // Set arguments
  auto& params = spec_->params();

  for (auto& arg : args()) {
    // Find if arg.name() is in spec_->params()
    if (params.find(arg.name()) == params.end()) {
      HOLOSCAN_LOG_WARN("Argument '{}' is not defined in spec", arg.name());
      continue;
    }

    // Set arg.value() to spec_->params()[arg.name()]
    auto& param_wrap = params[arg.name()];

    HOLOSCAN_LOG_TRACE("GXFResource '{}':: setting argument '{}'", name(), arg.name());

    ArgumentSetter::set_param(param_wrap, arg);
  }

  // Set Handler parameters
  for (auto& [key, param_wrap] : params) {
    HOLOSCAN_GXF_CALL(::holoscan::gxf::GXFParameterAdaptor::set_param(
        gxf_context_, gxf_cid_, key.c_str(), param_wrap));
    // TODO: handle error
    HOLOSCAN_LOG_TRACE("GXFResource '{}':: setting GXF parameter '{}'", name(), key);
  }

  is_initialized_ = true;
}

}  // namespace holoscan::gxf
