/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>

#include <holoscan/holoscan_core_export.h>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/time/clock.hpp>

namespace holoscan {

/**
 * @brief A clock where time flow is controlled manually.
 *
 * Useful for deterministic tests. `sleep_for()` / `sleep_until()` model scheduler progression and
 * move the clock monotonically forward to the wake-up point; the clock cannot go backwards through
 * those calls. The `advance_to()` / `advance_by()` helpers provide direct manual control and may
 * move the clock to any timestamp.
 *
 * Thread-safety: the timestamp is held in an atomic, so each individual accessor (`time()`,
 * `timestamp()`) and mutator (`advance_to()`, `advance_by()`) is race-free — it is safe to read the
 * time from one thread while another advances it. Compound operations are not atomic as a whole: in
 * particular `sleep_until()`'s "reject going backwards" check-then-set and `sleep_for()`'s
 * read-then-sleep_until are not serialized against concurrent advances, so time should be driven
 * from a single controlling thread (the usual test/stepping pattern). The clock is non-copyable.
 */
class HOLOSCAN_CORE_EXPORT ManualClock final : public Clock {
 public:
  explicit ManualClock(int64_t initial_timestamp_ns = 0) noexcept
      : current_time_ns_(initial_timestamp_ns) {}

  [[nodiscard]] double time() const noexcept override {
    return timestamp_to_time(current_time_ns_.load());
  }
  [[nodiscard]] int64_t timestamp() const noexcept override { return current_time_ns_.load(); }
  [[nodiscard]] expected<void, Error> sleep_for(int64_t duration_ns) override;
  [[nodiscard]] expected<void, Error> sleep_until(int64_t target_time_ns) override;

  /// @brief Set the clock to an absolute timestamp (nanoseconds). May move backwards.
  void advance_to(int64_t target_time_ns) noexcept { current_time_ns_.store(target_time_ns); }

  /// @brief Advance the clock by a delta (nanoseconds). The delta may be negative.
  void advance_by(int64_t delta_ns) noexcept { current_time_ns_.fetch_add(delta_ns); }
  void advance_by(std::chrono::nanoseconds delta) noexcept { advance_by(delta.count()); }

 private:
  std::atomic<int64_t> current_time_ns_{0};
};

}  // namespace holoscan
