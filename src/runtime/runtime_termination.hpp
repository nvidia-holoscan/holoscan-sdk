/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <exception>
#include <utility>

namespace holoscan::detail {

/// Invoke diagnostic logging and then enforce a mandatory Runtime termination boundary.
template <typename LogCallable>
[[noreturn]] void runtime_log_and_terminate_noexcept(LogCallable&& log_callable) noexcept {
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  try {
#endif
    std::forward<LogCallable>(log_callable)();
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  } catch (...) {
    // Diagnostic formatting or sink failure cannot weaken a fatal Runtime invariant.
    static_cast<void>(0);
  }
#endif
  std::terminate();
}

}  // namespace holoscan::detail
