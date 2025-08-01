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

#include "holoscan/core/resources/gxf/manual_clock.hpp"

#include <string>

#include "holoscan/core/component_spec.hpp"
#include "holoscan/core/gxf/gxf_utils.hpp"

namespace holoscan {

ManualClock::ManualClock(const std::string& name, nvidia::gxf::ManualClock* component)
    : Clock(name, component) {
  auto maybe_initial_timestamp = component->getParameter<uint64_t>("initial_timestamp");
  if (!maybe_initial_timestamp) {
    throw std::runtime_error("Failed to get initial_timestamp");
  }
  initial_timestamp_ = maybe_initial_timestamp.value();
}

nvidia::gxf::ManualClock* ManualClock::get() const {
  return static_cast<nvidia::gxf::ManualClock*>(gxf_cptr_);
}

double ManualClock::time() const {
  auto clock = get();
  if (clock) {
    return clock->time();
  }
  return 0.0;
}

int64_t ManualClock::timestamp() const {
  auto clock = get();
  if (clock) {
    return clock->timestamp();
  }
  return 0;
}

void ManualClock::sleep_for(int64_t duration_ns) {
  auto clock = get();
  if (clock) {
    clock->sleepFor(duration_ns);
  } else {
    HOLOSCAN_LOG_ERROR("RealtimeClock component not yet registered with GXF");
  }
}

void ManualClock::sleep_until(int64_t target_time_ns) {
  auto clock = get();
  if (clock) {
    clock->sleepUntil(target_time_ns);
  } else {
    HOLOSCAN_LOG_ERROR("RealtimeClock component not yet registered with GXF");
  }
}

void ManualClock::setup(ComponentSpec& spec) {
  spec.param(initial_timestamp_,
             "initial_timestamp",
             "Initial timestamp",
             "The initial timestamp on the clock (in nanoseconds).",
             0L);
}

}  // namespace holoscan
