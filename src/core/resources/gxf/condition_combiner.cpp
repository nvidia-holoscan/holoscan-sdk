/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "holoscan/core/resources/gxf/condition_combiner.hpp"

#include <memory>
#include <string>
#include <vector>

#include "holoscan/core/component_spec.hpp"

namespace holoscan {

ConditionCombiner::ConditionCombiner(const std::string& name,
                                     nvidia::gxf::SchedulingTermCombiner* component)
    : gxf::GXFResource(name, component) {}

nvidia::gxf::SchedulingTermCombiner* ConditionCombiner::get() const {
  return static_cast<nvidia::gxf::SchedulingTermCombiner*>(gxf_cptr_);
}

OrConditionCombiner::OrConditionCombiner(const std::string& name,
                                         nvidia::gxf::OrSchedulingTermCombiner* component)
    : ConditionCombiner(name, component) {}

nvidia::gxf::OrSchedulingTermCombiner* OrConditionCombiner::get() const {
  return static_cast<nvidia::gxf::OrSchedulingTermCombiner*>(gxf_cptr_);
}

void OrConditionCombiner::setup(ComponentSpec& spec) {
  spec.param(terms_,
             "terms",
             "The conditions associated with this combiner.",
             "The conditions associated with this condition combiner. Any other conditions not "
             "associate with a combiner are AND combined with the result of this combiner.");
}

void OrConditionCombiner::initialize() {
  HOLOSCAN_LOG_TRACE("OrConditionCombiner::initialize");
  // Set up prerequisite parameters before calling GXFResource::initialize()

  // Find if there is an argument for 'component_serializers'
  auto has_terms = std::find_if(
      args().begin(), args().end(), [](const auto& arg) { return (arg.name() == "terms"); });
  // Create a terms if no component_serializers argument was provided
  if (has_terms != args().end()) {
    // Set the gxf_eid for each term to match that of this combiner (if it has not already been set)
    HOLOSCAN_LOG_TRACE("OrConditionCombiner::initialize for '{}': a terms argument was found",
                       name());
    HOLOSCAN_LOG_TRACE("\tgxf_eid_ = {}", gxf_eid_);

    auto terms_arg = *has_terms;
    auto terms = std::any_cast<std::vector<std::shared_ptr<Condition>>>(terms_arg.value());
    for (auto& term : terms) {
      // For any non-native conditions set the GXF entity ID to match this combiner
      auto gxf_term = std::dynamic_pointer_cast<gxf::GXFCondition>(term);
      if (gxf_term) {
        HOLOSCAN_LOG_TRACE(
            "\tterm: '{}' has gxf_eid() = {}", gxf_term->name(), gxf_term->gxf_eid());
        if (gxf_eid_ != 0 && gxf_term->gxf_eid() == 0) {
          HOLOSCAN_LOG_TRACE("\t\tsetting gxf_term->gxf_eid({})", gxf_eid_);
          gxf_term->gxf_eid(gxf_eid_);
        }
      } else {
        HOLOSCAN_LOG_TRACE("\tterm: '{}' is a native condition", term->name());
      }
    }
  }
  GXFResource::initialize();
}

}  // namespace holoscan
