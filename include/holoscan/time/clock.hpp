/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <limits>

#include <holoscan/holoscan_core_export.h>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>

namespace holoscan {

/// @brief Convert a time in seconds to a timestamp in nanoseconds.
///
/// Truncates toward zero and clamps to the representable range so that an out-of-range input cannot
/// trigger undefined behavior. NaN maps to zero. (Use `std::llround()` on the scaled value instead
/// if round-to-nearest sub-nanosecond behavior is ever needed; timekeeping here does not require
/// it, and truncation lowers to a single instruction whereas `llround` does not.)
[[nodiscard]] constexpr int64_t time_to_timestamp(double time_s) noexcept {
  if (time_s != time_s) {
    return 0;
  }
  const double ns = time_s * 1.0e9;
  // Converting an out-of-range double to int64_t is undefined behavior, so clamp first.
  if (ns >= static_cast<double>(std::numeric_limits<int64_t>::max())) {
    return std::numeric_limits<int64_t>::max();
  }
  if (ns <= static_cast<double>(std::numeric_limits<int64_t>::min())) {
    return std::numeric_limits<int64_t>::min();
  }
  return static_cast<int64_t>(ns);
}

/// @brief Convert a timestamp in nanoseconds to a time in seconds.
[[nodiscard]] constexpr double timestamp_to_time(int64_t timestamp_ns) noexcept {
  return static_cast<double>(timestamp_ns) * 1.0e-9;
}

/**
 * @brief Keeps track of time.
 *
 * Native Holoscan replacement for the GXF `nvidia::gxf::Clock` component. The interface
 * intentionally does not inherit from `Component` and does not rely on GXF parameters or a live
 * GXF context; concrete clocks are plain objects configured through their constructors.
 *
 * Time is reported both in seconds (`time()`) and in nanoseconds (`timestamp()`).
 *
 * Thread-safety is part of each concrete clock's contract; see the individual class docs. All of
 * `RealtimeClock`, `ManualClock`, and `SyntheticClock` allow `time()` / `timestamp()` to be called
 * concurrently from multiple threads.
 */
class HOLOSCAN_CORE_EXPORT Clock {
 public:
  virtual ~Clock() = default;

  /// @brief The current time of the clock, measured in seconds.
  [[nodiscard]] virtual double time() const = 0;

  /// @brief The current timestamp of the clock, measured in nanoseconds.
  [[nodiscard]] virtual int64_t timestamp() const = 0;

  /// @brief Wait until the given duration (in nanoseconds) has elapsed on the clock.
  [[nodiscard]] virtual expected<void, Error> sleep_for(int64_t duration_ns) = 0;

  /// @brief Wait until the clock reaches the given target time (in nanoseconds).
  [[nodiscard]] virtual expected<void, Error> sleep_until(int64_t target_time_ns) = 0;
};

}  // namespace holoscan
