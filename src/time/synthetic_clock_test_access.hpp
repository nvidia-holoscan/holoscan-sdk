/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <mutex>

#include <holoscan/time/synthetic_clock.hpp>

namespace holoscan::detail {

/// @brief Expose SyntheticClock sleeper registration to source-tree tests only.
///
/// This header is neither installed nor included by a public header. A test that destroys a clock
/// with a sleeper in flight must first see the sleeper registered inside its wait; a flag set
/// before the sleep call does not establish that.
class SyntheticClockTestAccess {
 public:
  /// @brief Return how many sleepers have registered and not yet left their wait.
  [[nodiscard]] static int active_sleepers(const SyntheticClock& clock) {
    const std::lock_guard<std::mutex> lock(clock.mutex_);
    return clock.active_sleepers_;
  }
};

}  // namespace holoscan::detail
