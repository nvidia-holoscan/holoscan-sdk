/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <nanobind/nanobind.h>

#include <chrono>
#include <cmath>

namespace holoscan::python::detail {

inline std::chrono::nanoseconds seconds_to_timeout(double seconds) {
  constexpr double kMaxSeconds =
      static_cast<double>(std::chrono::nanoseconds::max().count()) / 1'000'000'000.0;
  if (!std::isfinite(seconds) || seconds < 0.0 || seconds > kMaxSeconds) {
    throw nanobind::value_error("timeout must be a finite nonnegative number of seconds");
  }
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::duration<double>{seconds});
}

}  // namespace holoscan::python::detail
