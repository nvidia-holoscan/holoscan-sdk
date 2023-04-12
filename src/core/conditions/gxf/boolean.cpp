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

#include <gxf/std/scheduling_terms.hpp>

#include "holoscan/core/conditions/gxf/boolean.hpp"

#include "holoscan/core/component_spec.hpp"

namespace holoscan {

BooleanCondition::BooleanCondition(const std::string& name,
                                   nvidia::gxf::BooleanSchedulingTerm* term)
    : GXFCondition(name, term) {
  enable_tick_ = term->checkTickEnabled();
}

void BooleanCondition::setup(ComponentSpec& spec) {
  spec.param(enable_tick_,
             "enable_tick",
             "Enable Tick",
             "The default initial condition for enabling tick.",
             true);
}

void BooleanCondition::enable_tick() {
  if (gxf_cptr_) {
    nvidia::gxf::BooleanSchedulingTerm* boolean_condition =
        static_cast<nvidia::gxf::BooleanSchedulingTerm*>(gxf_cptr_);
    boolean_condition->enable_tick();
  }
  enable_tick_ = true;
}

void BooleanCondition::disable_tick() {
  if (gxf_cptr_) {
    nvidia::gxf::BooleanSchedulingTerm* boolean_condition =
        static_cast<nvidia::gxf::BooleanSchedulingTerm*>(gxf_cptr_);
    boolean_condition->disable_tick();
  }
  enable_tick_ = false;
}

bool BooleanCondition::check_tick_enabled() {
  if (gxf_cptr_) {
    nvidia::gxf::BooleanSchedulingTerm* boolean_condition =
        static_cast<nvidia::gxf::BooleanSchedulingTerm*>(gxf_cptr_);
    enable_tick_ = boolean_condition->checkTickEnabled();
  }
  return enable_tick_.get();
}

}  // namespace holoscan
