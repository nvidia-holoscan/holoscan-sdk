/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/core/resources/gxf/realtime_clock.hpp>

#include <string>

#include <gxf/std/clock.hpp>
#include <holoscan/core/component_spec.hpp>
#include <holoscan/core/gxf/gxf_utils.hpp>

namespace holoscan {

RealtimeClock::RealtimeClock(const std::string& name, nvidia::gxf::RealtimeClock* component)
    : gxf::Clock(name, component) {
  if (!component) {
    throw std::invalid_argument("RealtimeClock component cannot be null");
  }
  auto maybe_offset = component->getParameter<double>("offset");
  if (!maybe_offset) {
    throw std::runtime_error("Failed to get initial_time_offset parameter from GXF RealtimeClock");
  }
  initial_time_offset_ = maybe_offset.value();

  auto maybe_scale = component->getParameter<double>("scale");
  if (!maybe_scale) {
    throw std::runtime_error("Failed to get initial_time_scale parameter from GXF RealtimeClock");
  }
  initial_time_scale_ = maybe_scale.value();

  auto maybe_use_epoch = component->getParameter<bool>("use_epoch");
  if (!maybe_use_epoch) {
    throw std::runtime_error("Failed to get use_time_since_epoch parameter from GXF RealtimeClock");
  }
  use_time_since_epoch_ = maybe_use_epoch.value();
}

nvidia::gxf::RealtimeClock* RealtimeClock::get() const {
  return static_cast<nvidia::gxf::RealtimeClock*>(gxf_cptr_);
}

double RealtimeClock::time() const {
  auto clock = get();
  if (clock) {
    return clock->time();
  }
  return 0.0;
}

int64_t RealtimeClock::timestamp() const {
  auto clock = get();
  if (clock) {
    return clock->timestamp();
  }
  return 0;
}

void RealtimeClock::sleep_for(int64_t duration_ns) {
  auto clock = get();
  if (clock) {
    clock->sleepFor(duration_ns);
  } else {
    HOLOSCAN_LOG_ERROR("RealtimeClock component not yet registered with GXF");
  }
}

void RealtimeClock::sleep_until(int64_t target_time_ns) {
  auto clock = get();
  if (clock) {
    clock->sleepUntil(target_time_ns);
  } else {
    HOLOSCAN_LOG_ERROR("RealtimeClock component not yet registered with GXF");
  }
}

void RealtimeClock::set_time_scale(double time_scale) {
  auto clock = get();
  if (clock) {
    clock->setTimeScale(time_scale);
  } else {
    HOLOSCAN_LOG_ERROR("RealtimeClock component not yet registered with GXF");
  }
}

void RealtimeClock::setup(ComponentSpec& spec) {
  spec.param(initial_time_offset_,
             "initial_time_offset",
             "Initial time offset",
             "The initial time offset used until time scale is changed manually.",
             0.0);
  spec.param(initial_time_scale_,
             "initial_time_scale",
             "Initial time scale",
             "The initial time scale used until time scale is changed manually.",
             1.0);
  spec.param(use_time_since_epoch_,
             "use_time_since_epoch",
             "Use time since epoch",
             "If true, clock time is time since epoch + initial_time_offset at initialize()."
             "Otherwise clock time is initial_time_offset at initialize().",
             false);
}

}  // namespace holoscan
