/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/core/predicate.hpp>

struct BoolProxy {
  explicit operator bool() const noexcept { return true; }
};

// REQ: CONTRACT-010, COMPILE-016
auto invalid_predicate = holoscan::predicate(
    "bool-proxy",
    [](const holoscan::PredicateEvaluationContext&) noexcept -> BoolProxy { return {}; });

int main() {
  return 0;
}
