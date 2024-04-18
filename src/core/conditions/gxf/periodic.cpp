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

#include "holoscan/core/conditions/gxf/periodic.hpp"

#include <string>

#include "holoscan/core/component_spec.hpp"
#include "holoscan/core/gxf/gxf_utils.hpp"

namespace holoscan {

nvidia::gxf::PeriodicSchedulingTerm* PeriodicCondition::get() const {
  return static_cast<nvidia::gxf::PeriodicSchedulingTerm*>(gxf_cptr_);
}

PeriodicCondition::PeriodicCondition(int64_t recess_period_ns) {
  recess_period_ = std::to_string(recess_period_ns);
  recess_period_ns_ = recess_period_ns;
}

PeriodicCondition::PeriodicCondition(const std::string& name,
                                     nvidia::gxf::PeriodicSchedulingTerm* term)
    : GXFCondition(name, term) {
  if (term) {
    recess_period_ns_ = term->recess_period_ns();
    recess_period_ = std::to_string(recess_period_ns_);
  } else {
    HOLOSCAN_LOG_ERROR("PeriodicCondition: term is null");
  }
}

void PeriodicCondition::setup(ComponentSpec& spec) {
  spec.param(recess_period_,
             "recess_period",
             "RecessPeriod",
             "The recess period indicates the minimum amount of time which has to pass before the "
             "entity is permitted to execute again. The period is specified as a string containing "
             "a number and an (optional) unit. If no unit is given the value is assumed to be "
             "in nanoseconds. Supported units are: Hz, s, ms. Example: 10ms, 10000000, 0.2s, 50Hz");
}

void PeriodicCondition::recess_period(int64_t recess_period_ns) {
  std::string recess_period = std::to_string(recess_period_ns);
  auto periodic = get();
  if (periodic) { periodic->setParameter<std::string>("recess_period", recess_period); }
  recess_period_ = recess_period;
  recess_period_ns_ = recess_period_ns;
}

int64_t PeriodicCondition::recess_period_ns() {
  auto periodic = get();
  if (periodic) {
    auto recess_period_ns = periodic->recess_period_ns();
    if (recess_period_ns != recess_period_ns_) {
      recess_period_ns_ = recess_period_ns;
      recess_period_ = std::to_string(recess_period_ns_);
    }
  }
  return recess_period_ns_;
}

int64_t PeriodicCondition::last_run_timestamp() {
  int64_t last_run_timestamp = 0;
  auto periodic = get();
  if (periodic) {
    auto result = periodic->last_run_timestamp();
    if (result) {
      last_run_timestamp = result.value();
    } else {
      HOLOSCAN_LOG_ERROR("PeriodicCondition: Unable to get the result of 'last_run_timestamp()'");
    }
  } else {
    HOLOSCAN_LOG_ERROR("PeriodicCondition: GXF component pointer is null");
  }
  return last_run_timestamp;
}

}  // namespace holoscan
