/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/time/realtime_clock.hpp>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <thread>

#include <holoscan/logger/logger.hpp>

#include "clock_internal.hpp"
#include "core/diagnostic_reporting.hpp"

namespace holoscan {

namespace {

int64_t saturating_cast_to_int64_t(double value) {
  if (std::isnan(value)) {
    return 0;
  }
  if (value == std::numeric_limits<double>::infinity()) {
    return std::numeric_limits<int64_t>::max();
  }
  if (value == -std::numeric_limits<double>::infinity()) {
    return std::numeric_limits<int64_t>::min();
  }
  if (!std::isfinite(value)) {
    return 0;
  }
  if (value >= static_cast<double>(std::numeric_limits<int64_t>::max())) {
    return std::numeric_limits<int64_t>::max();
  }
  if (value <= static_cast<double>(std::numeric_limits<int64_t>::min())) {
    return std::numeric_limits<int64_t>::min();
  }
  return static_cast<int64_t>(value);
}

}  // namespace

RealtimeClock::RealtimeClock() : RealtimeClock(Config{}) {}

RealtimeClock::RealtimeClock(Config config) {
  reference_ = std::chrono::steady_clock::now();

  offset_ns_ = time_to_timestamp(config.initial_time_offset);
  if (config.use_time_since_epoch) {
    // Track the (epoch-scale) offset in integer nanoseconds rather than seconds-as-double. The time
    // scale is applied only to the elapsed time since the reference point, so the offset stays out
    // of the floating-point path and timestamp() keeps full nanosecond precision.
    const int64_t epoch_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                 std::chrono::system_clock::now().time_since_epoch())
                                 .count();
    offset_ns_ = detail::saturating_add(offset_ns_, epoch_ns);
    HOLOSCAN_LOG_INFO("RealtimeClock initial time {}", offset_ns_);
  }

  time_scale_ = config.initial_time_scale;
  if (!std::isfinite(time_scale_) || time_scale_ <= 0.0) {
    HOLOSCAN_LOG_AUTOMATIC_ERROR(
        "Initial time scale must be finite and > 0 (got {}); falling back to 1.0", time_scale_);
    time_scale_ = 1.0;
  }
}

int64_t RealtimeClock::timestamp() const {
  const auto now = std::chrono::steady_clock::now();
  const int64_t elapsed_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(now - reference_).count();
  // Scale only the small, monotonically growing elapsed time, then add the offset in integer
  // nanoseconds. An unscaled clock (the common case) stays bit-exact for arbitrarily long runs.
  const int64_t scaled_elapsed_ns =
      (time_scale_ == 1.0)
          ? elapsed_ns
          : saturating_cast_to_int64_t(static_cast<double>(elapsed_ns) * time_scale_);
  // This is the runtime read hot path, so use a plain add: the sum cannot overflow within the
  // representable int64 nanosecond range (~292 years) for any realistic offset and uptime.
  return offset_ns_ + scaled_elapsed_ns;
}

double RealtimeClock::time() const {
  return timestamp_to_time(timestamp());
}

expected<void, Error> RealtimeClock::sleep_for(int64_t duration_ns) {
  // The clock cannot go backwards.
  if (duration_ns < 0) {
    HOLOSCAN_LOG_AUTOMATIC_ERROR("Duration is negative: {}. Clock cannot go backwards.",
                                 duration_ns);
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument, "Sleep duration must be non-negative"});
  }

  const int64_t delta = saturating_cast_to_int64_t(static_cast<double>(duration_ns) / time_scale_);
  std::this_thread::sleep_for(std::chrono::duration<int64_t, std::nano>(delta));
  return {};
}

expected<void, Error> RealtimeClock::sleep_until(int64_t target_time_ns) {
  const int64_t now_ns = timestamp();
  // A target at or before "now" means the deadline has already passed (e.g. a fixed-rate loop that
  // overran). Proceed immediately rather than treating it as an error.
  if (target_time_ns <= now_ns) {
    return {};
  }
  return sleep_for(target_time_ns - now_ns);
}

}  // namespace holoscan
