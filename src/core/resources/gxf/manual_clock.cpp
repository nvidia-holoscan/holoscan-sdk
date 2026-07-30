/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/core/resources/gxf/manual_clock.hpp>

#include <string>

#include <gxf/std/clock.hpp>
#include <holoscan/core/component_spec.hpp>
#include <holoscan/core/gxf/gxf_utils.hpp>

namespace holoscan {

ManualClock::ManualClock(const std::string& name, nvidia::gxf::ManualClock* component)
    : gxf::Clock(name, component) {
  if (!component) {
    throw std::invalid_argument("ManualClock component cannot be null");
  }
  auto maybe_initial_timestamp = component->getParameter<uint64_t>("initial_timestamp");
  if (!maybe_initial_timestamp) {
    throw std::runtime_error("Failed to get initial_timestamp parameter from GXF ManualClock");
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
    HOLOSCAN_LOG_ERROR("ManualClock component not yet registered with GXF");
  }
}

void ManualClock::sleep_until(int64_t target_time_ns) {
  auto clock = get();
  if (clock) {
    clock->sleepUntil(target_time_ns);
  } else {
    HOLOSCAN_LOG_ERROR("ManualClock component not yet registered with GXF");
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
