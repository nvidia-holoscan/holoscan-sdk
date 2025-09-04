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

#ifndef HOLOSCAN_CORE_RESOURCES_GXF_CLOCK_HPP
#define HOLOSCAN_CORE_RESOURCES_GXF_CLOCK_HPP

#include <string>

#include "../../clock.hpp"
#include "../../gxf/gxf_resource.hpp"

namespace nvidia {
namespace gxf {
class Clock;
}
}  // namespace nvidia

namespace holoscan {

namespace gxf {

/**
 * @brief GXF-based clock implementation.
 *
 * This class wraps a GXF Clock component and implements the ClockInterface.
 */
class Clock : public ClockInterface, public GXFResource {
 public:
  HOLOSCAN_RESOURCE_FORWARD_ARGS_SUPER(Clock, GXFResource)
  Clock() = default;
  Clock(const std::string& name, nvidia::gxf::Clock* component);

  /// @brief The underlying GXF component's name.
  const char* gxf_typename() const override { return "nvidia::gxf::Clock"; }

  /// @brief The current time of the clock. Time is measured in seconds.
  double time() const override = 0;

  /// @brief The current timestamp of the clock. Timestamps are measured in nanoseconds.
  int64_t timestamp() const override = 0;

  /// @brief Waits until the given duration has elapsed on the clock
  void sleep_for(int64_t duration_ns) override = 0;

  // Bring the templated sleep_for method from ClockInterface into scope
  using ClockInterface::sleep_for;

  /// @brief Waits until the given target time
  void sleep_until(int64_t target_time_ns) override = 0;

  nvidia::gxf::Clock* get() const;
};

}  // namespace gxf

}  // namespace holoscan

#endif /* HOLOSCAN_CORE_RESOURCES_GXF_CLOCK_HPP */
