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

#ifndef HOLOSCAN_CORE_RESOURCES_GXF_CLOCK_HPP
#define HOLOSCAN_CORE_RESOURCES_GXF_CLOCK_HPP

#include <string>

#include <gxf/std/clock.hpp>
#include "../../gxf/gxf_resource.hpp"

namespace holoscan {

/**
 * @brief Base clock class.
 *
 * Clock classes are used by a Scheduler to control the flow of time in an application.
 */
class Clock : public gxf::GXFResource {
 public:
  HOLOSCAN_RESOURCE_FORWARD_ARGS_SUPER(Clock, gxf::GXFResource)
  Clock() = default;
  Clock(const std::string& name, nvidia::gxf::Clock* component);

  /// @brief The underlying GXF component's name.
  const char* gxf_typename() const override { return "nvidia::gxf::Clock"; }

  /// @brief The current time of the clock. Time is measured in seconds.
  virtual double time() const = 0;

  /// @brief The current timestamp of the clock. Timestamps are measured in nanoseconds.
  virtual int64_t timestamp() const = 0;

  /// @brief Waits until the given duration has elapsed on the clock
  virtual void sleep_for(int64_t duration_ns) = 0;

  /**
   * @brief Set a duration to sleep.
   *
   * @param duration The sleep duration of type `std::chrono::duration`.
   */
  template <typename Rep, typename Period>
  void sleep_for(std::chrono::duration<Rep, Period> duration) {
    int64_t duration_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();
    sleep_for(duration_ns);
  }

  /// @brief Waits until the given target time
  virtual void sleep_until(int64_t target_time_ns) = 0;
};

}  // namespace holoscan

#endif /* HOLOSCAN_CORE_RESOURCES_GXF_CLOCK_HPP */
