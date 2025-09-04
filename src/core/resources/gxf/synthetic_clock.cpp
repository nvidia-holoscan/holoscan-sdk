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

#include "holoscan/core/resources/gxf/synthetic_clock.hpp"

#include <string>

#include <gxf/std/synthetic_clock.hpp>
#include "holoscan/core/component_spec.hpp"
#include "holoscan/core/gxf/gxf_utils.hpp"

namespace holoscan {

SyntheticClock::SyntheticClock(const std::string& name, nvidia::gxf::SyntheticClock* component)
    : gxf::Clock(name, component) {
  if (!component) {
    throw std::invalid_argument("SyntheticClock component cannot be null");
  }
  auto maybe_initial_timestamp = component->getParameter<int64_t>("initial_timestamp");
  if (!maybe_initial_timestamp) {
    throw std::runtime_error("Failed to get initial_timestamp parameter from GXF SyntheticClock");
  }
  initial_timestamp_ = maybe_initial_timestamp.value();
}

nvidia::gxf::SyntheticClock* SyntheticClock::get() const {
  return static_cast<nvidia::gxf::SyntheticClock*>(gxf_cptr_);
}

double SyntheticClock::time() const {
  auto clock = get();
  if (clock) {
    return clock->time();
  }
  return 0.0;
}

int64_t SyntheticClock::timestamp() const {
  auto clock = get();
  if (clock) {
    return clock->timestamp();
  }
  return 0;
}

void SyntheticClock::sleep_for(int64_t duration_ns) {
  auto clock = get();
  if (clock) {
    clock->sleepFor(duration_ns);
  } else {
    HOLOSCAN_LOG_ERROR("SyntheticClock component not yet registered with GXF");
  }
}

void SyntheticClock::advance_to(int64_t new_time_ns) {
  auto clock = get();
  if (clock) {
    clock->advanceTo(new_time_ns);
  } else {
    HOLOSCAN_LOG_ERROR("SyntheticClock component not yet registered with GXF");
  }
}

void SyntheticClock::advance_by(int64_t time_delta_ns) {
  auto clock = get();
  if (clock) {
    clock->advanceBy(time_delta_ns);
  } else {
    HOLOSCAN_LOG_ERROR("SyntheticClock component not yet registered with GXF");
  }
}

void SyntheticClock::sleep_until(int64_t target_time_ns) {
  auto clock = get();
  if (clock) {
    clock->sleepUntil(target_time_ns);
  } else {
    HOLOSCAN_LOG_ERROR("SyntheticClock component not yet registered with GXF");
  }
}

void SyntheticClock::setup(ComponentSpec& spec) {
  spec.param(initial_timestamp_,
             "initial_timestamp",
             "Initial time Timestamp",
             "The initial timestamp on the clock (in nanoseconds).",
             static_cast<int64_t>(0));
}

}  // namespace holoscan
