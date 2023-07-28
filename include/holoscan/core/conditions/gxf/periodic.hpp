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

#ifndef HOLOSCAN_CORE_CONDITIONS_GXF_PERIODIC_HPP
#define HOLOSCAN_CORE_CONDITIONS_GXF_PERIODIC_HPP

#include <string>
#include <chrono>

#include "../../gxf/gxf_condition.hpp"

namespace holoscan {

/**
 * @brief Condition class to support periodic execution of operators.
 *
 * The recess (pause) period indicates the minimum amount of
 * time that must elapse before the `compute()` method can be executed again.
 * The period is specified as a string containing a number and an (optional) unit.
 * If no unit is given the value is assumed to be in nanoseconds.
 * Supported units are: ms, s, hz (case insensitive)
 *
 * For example: "10000000", "10ms", "1s", "50Hz".
 *
 * Using `std::string` as the first parameter of `make_condition<T>` is only
 * available through `Arg`.
 *
 * For example: `Arg("recess_period") = "1s"` or `Arg("recess_period", "1s")`.
 *
 * The recess (pause) period can also be specified as an integer value (type `int64_t`) in
 * nanoseconds or as a value of type `std::chrono::duration<Rep, Period>`
 * (see https://en.cppreference.com/w/cpp/chrono/duration).
 *
 * Example:
 * - `1000` (1000 nanoseconds == 1 microsecond)
 * - `5ns`, `10us`, `1ms`, `0.5s`, `1min`, `0.5h`, etc.
 *   - requires `#include <chrono>` and `using namespace std::chrono_literals;`
 * - `std::chrono::milliseconds(10)`
 * - `std::chrono::duration<double, std::milli>(10)`
 * - `std::chrono::duration<double, std::ratio<1, 1000>>(10)`
 *
 * This class wraps GXF SchedulingTerm(`nvidia::gxf::PeriodicSchedulingTerm`).
 */
class PeriodicCondition : public gxf::GXFCondition {
 public:
  HOLOSCAN_CONDITION_FORWARD_ARGS_SUPER(PeriodicCondition, GXFCondition)

  PeriodicCondition() = default;

  explicit PeriodicCondition(int64_t recess_period_ns);

  template <typename Rep, typename Period>
  explicit PeriodicCondition(std::chrono::duration<Rep, Period> recess_period_duration) {
    recess_period_ns_ =
        std::chrono::duration_cast<std::chrono::nanoseconds>(recess_period_duration).count();
    recess_period_ = std::to_string(recess_period_ns_);
  }

  PeriodicCondition(const std::string& name, nvidia::gxf::PeriodicSchedulingTerm* term);

  const char* gxf_typename() const override { return "nvidia::gxf::PeriodicSchedulingTerm"; }

  void setup(ComponentSpec& spec) override;

  /**
   * @brief Set recess period.
   *
   * Note that calling this method doesn't affect the behavior of the condition once the condition
   * is initialized.
   *
   * @param recess_period_ns The integer representing recess period in nanoseconds.
   */
  void recess_period(int64_t recess_period_ns);

  /**
   * @brief Set recess period.
   *
   * Note that calling this method doesn't affect the behavior of the condition once the
   * condition is initialized.
   *
   * @param recess_period_duration The recess period of type `std::chrono::duration`.
   */
  template <typename Rep, typename Period>
  void recess_period(std::chrono::duration<Rep, Period> recess_period_duration) {
    int64_t recess_period_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(recess_period_duration).count();
    recess_period(recess_period_ns);
  }

  /**
   * @brief Get recess period in nano seconds.
   *
   * @return The minimum time which needs to elapse between two executions (in nano seconds)
   */
  int64_t recess_period_ns();

  /**
   * @brief Get the last run time stamp.
   *
   * @return The last run time stamp.
   */
  int64_t last_run_timestamp();

 private:
  Parameter<std::string> recess_period_;
  int64_t recess_period_ns_ = 0;
};

}  // namespace holoscan

#endif /* HOLOSCAN_CORE_CONDITIONS_GXF_PERIODIC_HPP */
