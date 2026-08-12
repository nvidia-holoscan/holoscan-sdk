/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <holomq/logger.hpp>
#include <holomq/wait_strategy.hpp>

#include <chrono>

namespace holomq::detail {

/// Log when \ref WaitStrategy::timeout is set but \ref WaitMode::Return ignores it.
inline void warn_if_ignored_timeout_on_return(const WaitStrategy& strategy) noexcept {
  if (strategy.mode == WaitMode::Return && strategy.timeout.has_value()) {
    HOLOMQ_LOG_WARN("WaitStrategy::timeout is ignored when WaitMode::Return is selected");
  }
}

/// Default sleep duration for \ref WaitMode::TimedWait when \ref WaitStrategy::timeout is unset.
[[nodiscard]] inline std::chrono::nanoseconds timed_wait_sleep_duration(
    const WaitStrategy& strategy) noexcept {
  return strategy.timeout.has_value() ? *strategy.timeout : std::chrono::milliseconds(1);
}

}  // namespace holomq::detail
