/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <chrono>
#include <cstdint>

#include <holoscan/holoscan_core_export.h>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/time/clock.hpp>

namespace holoscan {

/**
 * @brief A clock which runs based on a real-time (steady) clock.
 *
 * Time can be scaled to run slower or faster than wall-clock time. This is the clock intended to
 * drive production fixed-rate execution.
 *
 * Thread-safety: the clock is immutable after construction (offset, scale, and reference point are
 * fixed at construction time), so `time()`, `timestamp()`, `sleep_for()`, and `sleep_until()` may
 * all be called concurrently from any number of threads without external synchronization. Runtime
 * re-scaling is intentionally not supported; it can be reintroduced later with proper
 * synchronization if a use case requires it.
 *
 * Precision: `timestamp()` is computed in integer nanoseconds. The (potentially epoch-scale) offset
 * is kept out of the floating-point path and only the elapsed time since construction is scaled, so
 * nanosecond precision is preserved even when `use_time_since_epoch` is set. `time()` returns
 * seconds as a `double` and therefore carries the usual floating-point resolution limits at
 * epoch scale; prefer `timestamp()` when nanosecond precision matters.
 */
class HOLOSCAN_CORE_EXPORT RealtimeClock final : public Clock {
 public:
  struct Config {
    /// Initial time offset in seconds.
    double initial_time_offset = 0.0;
    /// Time scale applied to elapsed real time. Must be > 0; a non-positive value is clamped
    /// to 1.0.
    double initial_time_scale = 1.0;
    /// If true, the clock time starts at (time since epoch + initial_time_offset); otherwise it
    /// starts at initial_time_offset.
    bool use_time_since_epoch = false;
  };

  RealtimeClock();
  HOLOSCAN_CORE_EXPORT explicit RealtimeClock(Config config);

  [[nodiscard]] double time() const override;
  [[nodiscard]] int64_t timestamp() const override;
  [[nodiscard]] expected<void, Error> sleep_for(int64_t duration_ns) override;
  [[nodiscard]] expected<void, Error> sleep_until(int64_t target_time_ns) override;

 private:
  std::chrono::time_point<std::chrono::steady_clock> reference_;
  int64_t offset_ns_ = 0;
  double time_scale_ = 1.0;
};

}  // namespace holoscan
