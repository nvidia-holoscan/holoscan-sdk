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
#ifndef HOLOSCAN_CORE_CLOCK_HPP
#define HOLOSCAN_CORE_CLOCK_HPP

#include <chrono>
#include <cstdint>
#include <memory>
#include <type_traits>

#include "./resource.hpp"

namespace holoscan {

/**
 * @brief Pure interface defining clock functionality.
 *
 * This interface defines the core clock operations that all clock implementations must provide.
 */
class ClockInterface {
 public:
  virtual ~ClockInterface() = default;

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
    int64_t duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();
    sleep_for(duration_ns);
  }

  /// @brief Waits until the given target time
  virtual void sleep_until(int64_t target_time_ns) = 0;
};

/**
 * @brief Clock resource class using Bridge pattern.
 *
 * Clock classes are used by a Scheduler to control the flow of time in an application.
 * This class acts as a Resource wrapper around a ClockInterface implementation.
 */
class Clock : public Resource {
 public:
  HOLOSCAN_RESOURCE_FORWARD_ARGS_SUPER(Clock, Resource)
  Clock() = default;

  /// @brief Constructor that takes a clock implementation
  explicit Clock(std::shared_ptr<ClockInterface> clock_impl) : clock_impl_(clock_impl) {}

  ~Clock() override = default;

  /// @brief Set the clock implementation
  void set_clock_impl(std::shared_ptr<ClockInterface> clock_impl) { clock_impl_ = clock_impl; }

  /// @brief Get the clock implementation
  std::shared_ptr<ClockInterface> clock_impl() const { return clock_impl_; }

  /// @brief Template method for safe casting to specific clock implementation types
  template <typename T>
  std::shared_ptr<T> cast_to() const {
    static_assert(std::is_base_of_v<ClockInterface, T>, "T must inherit from ClockInterface");
    return std::dynamic_pointer_cast<T>(clock_impl_);
  }

  /// @brief The current time of the clock. Time is measured in seconds.
  virtual double time() const {
    if (clock_impl_) {
      return clock_impl_->time();
    }
    return 0.0;
  }

  /// @brief The current timestamp of the clock. Timestamps are measured in nanoseconds.
  virtual int64_t timestamp() const {
    if (clock_impl_) {
      return clock_impl_->timestamp();
    }
    return 0;
  }

  /// @brief Waits until the given duration has elapsed on the clock
  virtual void sleep_for(int64_t duration_ns) {
    if (clock_impl_) {
      clock_impl_->sleep_for(duration_ns);
    }
  }

  /**
   * @brief Set a duration to sleep.
   *
   * @param duration The sleep duration of type `std::chrono::duration`.
   */
  template <typename Rep, typename Period>
  void sleep_for(std::chrono::duration<Rep, Period> duration) {
    int64_t duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();
    sleep_for(duration_ns);
  }

  /// @brief Waits until the given target time
  virtual void sleep_until(int64_t target_time_ns) {
    if (clock_impl_) {
      clock_impl_->sleep_until(target_time_ns);
    }
  }

 protected:
  std::shared_ptr<ClockInterface> clock_impl_;  ///< The clock implementation
};

}  // namespace holoscan

#endif /* HOLOSCAN_CORE_CLOCK_HPP */
