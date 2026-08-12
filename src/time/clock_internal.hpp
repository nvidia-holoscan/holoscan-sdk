/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Internal helpers shared by the clock implementation TUs (realtime_clock.cpp, manual_clock.cpp,
// synthetic_clock.cpp). This header is private to src/time and is not installed.

#pragma once

#include <cstdint>
#include <limits>

namespace holoscan::detail {

// Add two nanosecond counts, saturating at the int64 limits instead of overflowing (signed overflow
// is undefined behavior). Timestamps comfortably exceed any realistic uptime (~292 years), so
// saturation only guards against pathological offsets/durations. The comparisons are written so
// neither side can itself overflow. (C++26's std::add_sat will eventually replace this.)
constexpr int64_t saturating_add(int64_t a, int64_t b) noexcept {
  constexpr int64_t kMax = std::numeric_limits<int64_t>::max();
  constexpr int64_t kMin = std::numeric_limits<int64_t>::min();
  if (b > 0 && a > kMax - b) {
    return kMax;
  }
  if (b < 0 && a < kMin - b) {
    return kMin;
  }
  return a + b;
}

}  // namespace holoscan::detail
