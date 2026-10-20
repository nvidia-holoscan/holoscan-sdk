/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>

#include <holoscan/holoscan_core_export.h>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/time/clock.hpp>

namespace holoscan {

namespace detail {
class SyntheticClockTestAccess;
}  // namespace detail

/**
 * @brief A clock where time flow is synthesized, e.g. from a recording or a simulation.
 *
 * Unlike `ManualClock`, sleeping threads block until the clock is advanced past their target time
 * by another thread (via `advance_to()` / `advance_by()`), rather than the sleep itself advancing
 * the clock. This allows multiple workers to be released in lock-step as synthesized time moves
 * forward.
 *
 * Thread-safety: fully synchronized. All methods may be called concurrently. `advance_to()` /
 * `advance_by()` may run while other threads are blocked in `sleep_*()`. The destructor (via
 * `stop()`) wakes and then waits for every blocked sleeper to leave its wait before returning, so
 * the clock can be destroyed safely even with sleepers in flight (provided no new `sleep_*()` call
 * is started concurrently with destruction).
 */
class HOLOSCAN_CORE_EXPORT SyntheticClock final : public Clock {
 public:
  HOLOSCAN_CORE_EXPORT explicit SyntheticClock(int64_t initial_timestamp_ns = 0);
  ~SyntheticClock() override;

  [[nodiscard]] double time() const override;
  [[nodiscard]] int64_t timestamp() const override;
  [[nodiscard]] expected<void, Error> sleep_for(int64_t duration_ns) override;
  [[nodiscard]] expected<void, Error> sleep_until(int64_t target_time_ns) override;

  /// @brief Advance the clock to an absolute timestamp (nanoseconds), waking any sleepers.
  HOLOSCAN_CORE_EXPORT void advance_to(int64_t target_time_ns);

  /// @brief Advance the clock by a delta (nanoseconds), waking any sleepers.
  HOLOSCAN_CORE_EXPORT void advance_by(int64_t delta_ns);
  HOLOSCAN_CORE_EXPORT void advance_by(std::chrono::nanoseconds delta);

  /// @brief Stop the clock and release every blocked sleeper. Sleepers that have not yet reached
  /// their target time return an `ErrorCode::kClosed` error. This blocks until all woken sleepers
  /// have left their wait, so the clock's synchronization primitives can be torn down safely. The
  /// destructor calls this. Idempotent.
  HOLOSCAN_CORE_EXPORT void stop();

 private:
  mutable std::mutex mutex_;
  std::condition_variable condition_variable_;
  int64_t current_time_ns_ = 0;
  int active_sleepers_ = 0;
  bool stopped_ = false;

  friend class detail::SyntheticClockTestAccess;
};

}  // namespace holoscan
