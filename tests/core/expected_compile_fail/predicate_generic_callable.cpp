/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/core/predicate.hpp>

// REQ: CONTRACT-010, COMPILE-016
auto invalid_predicate = holoscan::predicate("generic", [](const auto&) noexcept { return true; });

int main() {
  return 0;
}
