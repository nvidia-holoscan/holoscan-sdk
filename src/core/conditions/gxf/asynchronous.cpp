/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <magic_enum.hpp>
#include "holoscan/core/component_spec.hpp"

namespace holoscan {

AsynchronousCondition::AsynchronousCondition(const std::string& name,
                                             nvidia::gxf::AsynchronousSchedulingTerm* term)
    : GXFCondition(name, term) {}

nvidia::gxf::AsynchronousSchedulingTerm* AsynchronousCondition::get() const {
  return static_cast<nvidia::gxf::AsynchronousSchedulingTerm*>(gxf_cptr_);
}

namespace {
// use explicit conversion to avoid dependency on GXF headers in base condition.hpp
holoscan::AsynchronousEventState gxf_to_holoscan_event_state(
    nvidia::gxf::AsynchronousEventState state) {
  switch (state) {
    case nvidia::gxf::AsynchronousEventState::READY:
      return holoscan::AsynchronousEventState::kReady;
    case nvidia::gxf::AsynchronousEventState::WAIT:
      return holoscan::AsynchronousEventState::kWait;
    case nvidia::gxf::AsynchronousEventState::EVENT_WAITING:
      return holoscan::AsynchronousEventState::kEventWaiting;
    case nvidia::gxf::AsynchronousEventState::EVENT_DONE:
      return holoscan::AsynchronousEventState::kEventDone;
    case nvidia::gxf::AsynchronousEventState::EVENT_NEVER:
      return holoscan::AsynchronousEventState::kEventNever;
    default:
      std::string err_msg = fmt::format("Unknown nvidia::AsynchronousEventState event state: {})",
                                        magic_enum::enum_name(state));
      HOLOSCAN_LOG_ERROR(err_msg);
      throw std::runtime_error(err_msg);
  }
}

nvidia::gxf::AsynchronousEventState holoscan_to_gxf_event_state(
    holoscan::AsynchronousEventState state) {
  switch (state) {
    case holoscan::AsynchronousEventState::kReady:
      return nvidia::gxf::AsynchronousEventState::READY;
    case holoscan::AsynchronousEventState::kWait:
      return nvidia::gxf::AsynchronousEventState::WAIT;
    case holoscan::AsynchronousEventState::kEventWaiting:
      return nvidia::gxf::AsynchronousEventState::EVENT_WAITING;
    case holoscan::AsynchronousEventState::kEventDone:
      return nvidia::gxf::AsynchronousEventState::EVENT_DONE;
    case holoscan::AsynchronousEventState::kEventNever:
      return nvidia::gxf::AsynchronousEventState::EVENT_NEVER;
    default:
      std::string err_msg = fmt::format("Unknown holoscan::AsynchronousEventState event state: {})",
                                        magic_enum::enum_name(state));
      HOLOSCAN_LOG_ERROR(err_msg);
      throw std::runtime_error(err_msg);
  }
}
}  // namespace

void AsynchronousCondition::event_state(AsynchronousEventState state) {
  auto asynchronous_scheduling_term = get();
  auto gxf_event_state = holoscan_to_gxf_event_state(state);
  if (asynchronous_scheduling_term) {
    asynchronous_scheduling_term->setEventState(gxf_event_state);
  }
  event_state_ = state;
}

AsynchronousEventState AsynchronousCondition::event_state() const {
  auto asynchronous_scheduling_term = get();
  if (asynchronous_scheduling_term) {
    auto gxf_event_state = asynchronous_scheduling_term->getEventState();
    return gxf_to_holoscan_event_state(gxf_event_state);
  }
  return event_state_;
}

}  // namespace holoscan
