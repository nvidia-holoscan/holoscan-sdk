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

#ifndef HOLOSCAN_CORE_RESOURCES_GXF_SYNTHETIC_CLOCK_HPP
#define HOLOSCAN_CORE_RESOURCES_GXF_SYNTHETIC_CLOCK_HPP

#include <chrono>
#include <string>

#include <gxf/std/synthetic_clock.hpp>

#include "./clock.hpp"

namespace holoscan {

/**
 * @brief Synthetic clock class.
 *
 * A clock where time flow is synthesized, like from a recording or a simulation.
 *
 * ==Parameters==
 *
 * - **initial_timestamp** (int64_t): The initial timestamp on the clock (in nanoseconds)
 *   (default: 0.0).
 */
class SyntheticClock : public gxf::Clock {
 public:
  HOLOSCAN_RESOURCE_FORWARD_ARGS_SUPER(SyntheticClock, gxf::Clock)
  SyntheticClock() = default;
  SyntheticClock(const std::string& name, nvidia::gxf::SyntheticClock* component);

  /// @brief The underlying GXF component's name.
  const char* gxf_typename() const override { return "nvidia::gxf::SyntheticClock"; }

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

  /// @brief Manually advance the clock to a desired new target time.
  void advance_to(int64_t new_time_ns);

  /// @brief Manually advance the clock by a given delta.
  void advance_by(int64_t time_delta_ns);

  /**
   * @brief Set a duration to advance the clock by.
   *
   * @param duration The `std::chrono::duration` to advance the clock by.
   */
  template <typename Rep, typename Period>
  void advance_by(std::chrono::duration<Rep, Period> duration) {
    int64_t time_delta_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();
    advance_by(time_delta_ns);
  }

  nvidia::gxf::SyntheticClock* get() const;

 private:
  Parameter<int64_t> initial_timestamp_;
};
}  // namespace holoscan

#endif /* HOLOSCAN_CORE_RESOURCES_GXF_SYNTHETIC_CLOCK_HPP */
