/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef HOLOSCAN_CORE_RESOURCES_GXF_REALTIME_CLOCK_HPP
#define HOLOSCAN_CORE_RESOURCES_GXF_REALTIME_CLOCK_HPP

#include <chrono>
#include <string>

#include "./clock.hpp"

namespace holoscan {

/**
 * @brief Real-time clock class.
 *
 * The RealtimeClock respects the true duration of conditions such as `PeriodicCondition`. It is
 * the default clock type used in Holoscan SDK.
 */
class RealtimeClock : public Clock {
 public:
  HOLOSCAN_RESOURCE_FORWARD_ARGS_SUPER(RealtimeClock, Clock)
  RealtimeClock() = default;
  RealtimeClock(const std::string& name, nvidia::gxf::RealtimeClock* component);

  /// @brief The underlying GXF component's name.
  const char* gxf_typename() const override { return "nvidia::gxf::RealtimeClock"; }

  void setup(ComponentSpec& spec);

  /// @brief The current time of the clock. Time is measured in seconds.
  double time() const override;

  /// @brief The current timestamp of the clock. Timestamps are measured in nanoseconds.
  int64_t timestamp() const override;

  /// @brief Waits until the given duration has elapsed on the clock
  void sleep_for(int64_t duration_ns) override;

  /// @brief Waits until the given target time
  void sleep_until(int64_t target_time_ns) override;

  /**
   * @brief Set the time scale of the clock. A value of 1.0 corresponds to realtime. Values larger
   * than 1.0 cause time to run faster, while values less than 1.0 cause time to run more slowly.
   */
  void set_time_scale(double time_scale);

 private:
  Parameter<double> initial_time_offset_;
  Parameter<double> initial_time_scale_;
  Parameter<bool> use_time_since_epoch_;
};
}  // namespace holoscan

#endif /* HOLOSCAN_CORE_RESOURCES_GXF_REALTIME_CLOCK_HPP */
