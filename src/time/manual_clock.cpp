/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/time/manual_clock.hpp>

#include <holoscan/logger/logger.hpp>

#include "clock_internal.hpp"

namespace holoscan {

expected<void, Error> ManualClock::sleep_for(int64_t duration_ns) {
  return sleep_until(detail::saturating_add(current_time_ns_.load(), duration_ns));
}

expected<void, Error> ManualClock::sleep_until(int64_t target_time_ns) {
  // Sleeping advances the manual clock; it cannot go backwards. Use advance_to() for direct,
  // unrestricted control.
  const int64_t current_time_ns = current_time_ns_.load();
  if (target_time_ns < current_time_ns) {
    HOLOSCAN_LOG_ERROR("Target time {} is less than current time {}; clock cannot go backwards",
                       target_time_ns,
                       current_time_ns);
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument, "Sleep target is earlier than the current time"});
  }
  current_time_ns_.store(target_time_ns);
  return {};
}

}  // namespace holoscan
