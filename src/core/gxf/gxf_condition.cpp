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

#include "holoscan/core/gxf/gxf_condition.hpp"

#include <gxf/core/gxf.h>

#include "holoscan/core/component_spec.hpp"
#include "holoscan/core/executor.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/gxf/gxf_utils.hpp"

namespace holoscan::gxf {

GXFCondition::GXFCondition(const std::string& name, nvidia::gxf::SchedulingTerm* term) {
  id_ = term->cid();
  name_ = name;
  gxf_context_ = term->context();
  gxf_eid_ = term->eid();
  gxf_cid_ = term->cid();
  GxfComponentType(gxf_context_, gxf_cid_, &gxf_tid_);
  gxf_cname_ = name;
  gxf_cptr_ = term;
}

void GXFCondition::initialize() {
  Condition::initialize();
  gxf_context_ = fragment()->executor().context();

  // Set GXF component name
  gxf_cname(name());

  GXFComponent::gxf_initialize();

  // Set GXF component ID as the component ID
  id_ = gxf_cid_;

  if (!spec_) {
    HOLOSCAN_LOG_ERROR("No component spec for GXFCondition '{}'", name());
    return;
  }
  auto& spec = *spec_;

  gxf_result_t code;
  // Set arguments
  auto& params = spec.params();
  for (auto& arg : args_) {
    // Find if arg.name() is in spec.params()
    if (params.find(arg.name()) == params.end()) {
      HOLOSCAN_LOG_WARN("Argument '{}' not found in spec.params()", arg.name());
      continue;
    }

    // Set arg.value() to spec.params()[arg.name()]
    auto& param_wrap = params[arg.name()];

    HOLOSCAN_LOG_TRACE("GXFCondition '{}':: setting argument '{}'", name(), arg.name());

    ArgumentSetter::set_param(param_wrap, arg);
  }

  // Set Handler parameters
  for (auto& [key, param_wrap] : params) {
    code = ::holoscan::gxf::GXFParameterAdaptor::set_param(
        gxf_context_, gxf_cid_, key.c_str(), param_wrap);
    // TODO: handle error
    HOLOSCAN_LOG_TRACE("GXFCondition '{}':: setting GXF parameter '{}'", name(), key);
  }
  (void)code;
}

}  // namespace holoscan::gxf
