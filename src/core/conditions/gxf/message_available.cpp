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

#include "holoscan/core/conditions/gxf/message_available.hpp"

#include "holoscan/core/component_spec.hpp"

namespace holoscan {

void MessageAvailableCondition::setup(ComponentSpec& spec) {
  spec.param(receiver_,
             "receiver",
             "Queue channel",
             "The scheduling term permits execution if this channel has at least a given number of "
             "messages available.");
  spec.param(min_size_,
             "min_size",
             "Minimum message count",
             "The scheduling term permits execution if the given receiver has at least the "
             "given number of messages available.",
             1UL);
  spec.param(
      front_stage_max_size_,
      "front_stage_max_size",
      "Maximum front stage message count",
      "If set the scheduling term will only allow execution if the number of messages in the front "
      "stage does not exceed this count. It can for example be used in combination with codelets "
      "which do not clear the front stage in every tick.",
      ParameterFlag::kOptional);
}

nvidia::gxf::MessageAvailableSchedulingTerm* MessageAvailableCondition::get() const {
  return static_cast<nvidia::gxf::MessageAvailableSchedulingTerm*>(gxf_cptr_);
}

void MessageAvailableCondition::min_size(uint64_t min_size) {
  auto cond = get();
  if (cond) {
    auto maybe_set = cond->setMinSize(min_size);
    if (!maybe_set) {
      throw std::runtime_error(
          fmt::format("Failed to set min_size: {}", GxfResultStr(maybe_set.error())));
    }
  }
  min_size_ = min_size;
}

void MessageAvailableCondition::front_stage_max_size(size_t front_stage_max_size) {
  auto cond = get();
  if (cond) {
    auto maybe_set = cond->setFrontStageMaxSize(front_stage_max_size);
    if (!maybe_set) {
      throw std::runtime_error(
          fmt::format("Failed to set front_stage_max_size: {}", GxfResultStr(maybe_set.error())));
    }
  }
  front_stage_max_size_ = front_stage_max_size;
}

}  // namespace holoscan
