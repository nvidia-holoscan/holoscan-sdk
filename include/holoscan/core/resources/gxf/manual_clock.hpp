/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOSCAN_CORE_RESOURCES_GXF_MANUAL_CLOCK_HPP
#define HOLOSCAN_CORE_RESOURCES_GXF_MANUAL_CLOCK_HPP

#include <chrono>
#include <string>

#include "./clock.hpp"

namespace nvidia {
namespace gxf {
class ManualClock;
}
}  // namespace nvidia

namespace holoscan {

/**
 * @brief Manual clock class.
 *
 * The manual clock compresses time intervals, rather than waiting for specified durations
 * (e.g. via PeriodicCondition). It is used mainly for testing applications.
 *
 * ==Parameters==
 *
 * - **initial_timestamp** (int64_t): The initial time offset in nanoseconds.
 */
class ManualClock : public gxf::Clock {
 public:
  HOLOSCAN_RESOURCE_FORWARD_ARGS_SUPER(ManualClock, gxf::Clock)
  ManualClock() = default;
  ManualClock(const std::string& name, nvidia::gxf::ManualClock* component);

  /// @brief The underlying GXF component's name.
  const char* gxf_typename() const override { return "nvidia::gxf::ManualClock"; }

  void setup(ComponentSpec& spec);

  /// @brief The current time of the clock. Time is measured in seconds.
  double time() const override;

  /// @brief The current timestamp of the clock. Timestamps are measured in nanoseconds.
  int64_t timestamp() const override;

  /// @brief Waits until the given duration has elapsed on the clock
  void sleep_for(int64_t duration_ns) override;

  // Bring the templated sleep_for method from base class into scope
  using gxf::Clock::sleep_for;

  /// @brief Waits until the given target time
  void sleep_until(int64_t target_time_ns) override;

  nvidia::gxf::ManualClock* get() const;

 private:
  Parameter<int64_t> initial_timestamp_;
};
}  // namespace holoscan

#endif /* HOLOSCAN_CORE_RESOURCES_GXF_MANUAL_CLOCK_HPP */
