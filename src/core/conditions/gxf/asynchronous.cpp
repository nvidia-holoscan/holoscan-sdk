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

#include "holoscan/core/conditions/gxf/asynchronous.hpp"

#include <string>

#include "holoscan/core/component_spec.hpp"

namespace holoscan {

AsynchronousCondition::AsynchronousCondition(const std::string& name,
                                             nvidia::gxf::AsynchronousSchedulingTerm* term)
    : GXFCondition(name, term) {}

nvidia::gxf::AsynchronousSchedulingTerm* AsynchronousCondition::get() const {
  return static_cast<nvidia::gxf::AsynchronousSchedulingTerm*>(gxf_cptr_);
}

void AsynchronousCondition::setup(ComponentSpec& spec) {
  (void)spec;  // no parameters to set
}

void AsynchronousCondition::event_state(AsynchronousEventState state) {
  auto asynchronous_scheduling_term = get();
  if (asynchronous_scheduling_term) { asynchronous_scheduling_term->setEventState(state); }
  event_state_ = state;
}

AsynchronousEventState AsynchronousCondition::event_state() const {
  auto asynchronous_scheduling_term = get();
  if (asynchronous_scheduling_term) { return asynchronous_scheduling_term->getEventState(); }
  return event_state_;
}

}  // namespace holoscan
