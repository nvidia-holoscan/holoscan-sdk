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

#include "holoscan/core/conditions/gxf/boolean.hpp"

#include <string>

#include <gxf/std/scheduling_terms.hpp>

#include "holoscan/core/component_spec.hpp"

namespace holoscan {

BooleanCondition::BooleanCondition(const std::string& name,
                                   nvidia::gxf::BooleanSchedulingTerm* term)
    : GXFCondition(name, term) {
  enable_tick_ = term->checkTickEnabled();
}

nvidia::gxf::BooleanSchedulingTerm* BooleanCondition::get() const {
  // Could use Component APIs, but keep gxf_cptr_ for now to handle any case
  // where gxf_graph_entity_ is null.

  // return gxf_component_.is_null()
  //            ? nullptr
  //            : dynamic_cast<nvidia::gxf::BooleanSchedulingTerm*>(gxf_component_.get());

  return static_cast<nvidia::gxf::BooleanSchedulingTerm*>(gxf_cptr_);
}

void BooleanCondition::setup(ComponentSpec& spec) {
  spec.param(enable_tick_,
             "enable_tick",
             "Enable Tick",
             "The default initial condition for enabling tick.",
             true);
}

void BooleanCondition::enable_tick() {
  auto boolean_condition = get();
  if (boolean_condition) { boolean_condition->enable_tick(); }
  enable_tick_ = true;
}

void BooleanCondition::disable_tick() {
  auto boolean_condition = get();
  if (boolean_condition) { boolean_condition->disable_tick(); }
  enable_tick_ = false;
}

bool BooleanCondition::check_tick_enabled() {
  auto boolean_condition = get();
  return boolean_condition ? boolean_condition->checkTickEnabled() : enable_tick_.get();
}

}  // namespace holoscan
