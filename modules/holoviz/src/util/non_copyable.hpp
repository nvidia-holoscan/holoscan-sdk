/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOVIZ_SRC_UTIL_NON_COPYABLE_HPP
#define HOLOVIZ_SRC_UTIL_NON_COPYABLE_HPP

namespace holoscan::viz {

/**
 * Ensure that classes derived from class NonCopyable cannot be copied.
 */
class NonCopyable {
 protected:
  constexpr NonCopyable() = default;
  virtual ~NonCopyable() = default;

  NonCopyable(const NonCopyable&) = delete;
  NonCopyable& operator=(const NonCopyable&) = delete;
};

}  // namespace holoscan::viz

#endif /* HOLOVIZ_SRC_UTIL_NON_COPYABLE_HPP */
