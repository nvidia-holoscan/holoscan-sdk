/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef HOLOSCAN_UTILS_TIMER_HPP
#define HOLOSCAN_UTILS_TIMER_HPP

#include <fmt/format.h>

#include <chrono>

namespace holoscan {

/**
 * @brief Utility class to measure time.
 *
 * This class is used to measure time. It can be used to measure the time between two points in the
 * code, or to measure the time of a function.
 *
 * The class can be used in two ways:
 *
 * - Using the start() and stop() methods to measure the time between two points in the code.
 * - Using the constructor with the auto_start parameter set to true to measure the time of a
 *    function.
 *
 * The class can also be used to print the time elapsed between two points in the code, or the time
 * elapsed to execute a function.
 *
 * The class can be used in two ways:
 *
 * - Using the print() method to print the time elapsed between two points in the code.
 * - Using the constructor with the auto_output parameter set to true to print the time elapsed to
 *   execute a function.
 *
 * Examples:
 *
 * ```cpp
 * #include "holoscan/utils/timer.hpp"
 *
 * void foo() {
 *   holoscan::Timer timer("foo() took {:.8f} seconds\n");
 *   // Do something
 * }
 * ...
 * ```
 *
 * ```cpp
 * void foo() {
 *   holoscan::Timer timer("bar() took {:.8f} seconds\n", true, false);
 *   bar();
 *   timer.stop();
 *   timer.print();
 *   return 0;
 * }
 * ```
 *
 * ```cpp
 * void foo() {
 *   holoscan::Timer timer("", true, false);
 *   bar();
 *   double elapsed_time = timer.stop();
 *   fmt::print(stderr, "bar() took {:.8f} seconds", elapsed_time);
 * ```
 */
class Timer {
 public:
  /**
   * @brief Construct a new Timer object.
   *
   * The `message` parameter is used to print the message when the print() method is called or when
   * the auto_output parameter is set to true and the destructor is called.
   * The first parameter `{}` in the message will be replaced by the time elapsed.
   *
   * `auto_start` is used to start the timer when the constructor is called.
   * `auto_output` is used to print the time elapsed when the destructor is called.
   * By default, `auto_start` and `auto_output` are set to `true`.
   *
   * ```cpp
   * Timer timer("Time elapsed for foo() method: {:.8f} seconds\n", true, false);
   * foo();
   * timer.stop();
   * timer.print();
   * ```
   *
   * @param message The message to print when the timer is stopped.
   * @param auto_start If true, the timer is started when the constructor is called.
   * @param auto_output If true, the time elapsed is printed when the destructor is called.
   */
  explicit Timer(const char* message, bool auto_start = true, bool auto_output = true) {
    message_ = message;
    is_auto_output_ = auto_output;
    if (auto_start) {
      elapsed_seconds_ = 0.0;
      start_ = std::chrono::high_resolution_clock::now();
    }
  }

  /**
   * @brief Destroy the Timer object.
   *
   * If the auto_output parameter is set to true, the time elapsed is printed when the destructor is
   * called.
   */
  ~Timer() {
    if (elapsed_seconds_ <= 0.0) {
      end_ = std::chrono::high_resolution_clock::now();
      elapsed_seconds_ =
          std::chrono::duration_cast<std::chrono::duration<double>>(end_ - start_).count();
    }
    if (is_auto_output_) { print(); }
  }

  /**
   * @brief Start the timer.
   */
  void start() {
    elapsed_seconds_ = 0.0;
    start_ = std::chrono::high_resolution_clock::now();
  }

  /**
   * @brief Stop the timer.
   *
   * @return The time elapsed in seconds.
   */
  double stop() {
    end_ = std::chrono::high_resolution_clock::now();
    elapsed_seconds_ =
        std::chrono::duration_cast<std::chrono::duration<double>>(end_ - start_).count();
    return elapsed_seconds_;
  }

  /**
   * @brief Return the time elapsed in seconds.
   *
   * @return The time elapsed in seconds.
   */
  double elapsed_time() { return elapsed_seconds_; }

  /**
   * @brief Print the time elapsed.
   *
   * The message passed to the constructor is printed with the time elapsed if
   * no `message` parameter is passed to the print() method.
   *
   * The first parameter `{}` in the message will be replaced by the time elapsed.
   *
   * @param message The message to print.
   */
  void print(const char* message = nullptr) {
    if (message) {
      fmt::print(stderr, message, elapsed_seconds_);
    } else {
      fmt::print(stderr, message_, elapsed_seconds_);
    }
  }

 private:
  const char* message_ = nullptr;
  bool is_auto_output_ = false;
  double elapsed_seconds_ = -1;
  std::chrono::time_point<std::chrono::system_clock> start_{};
  std::chrono::time_point<std::chrono::system_clock> end_{};
};

}  // namespace holoscan

#endif /* HOLOSCAN_UTILS_TIMER_HPP */
