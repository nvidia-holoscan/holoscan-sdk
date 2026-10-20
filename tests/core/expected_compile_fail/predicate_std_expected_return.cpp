/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <expected>

#include <holoscan/core/predicate.hpp>

// REQ: CONTRACT-010
// REQ: COMPILE-016
auto invalid_predicate =
    holoscan::predicate("std-expected",
                        [](const holoscan::PredicateEvaluationContext&) noexcept
                            -> std::expected<bool, holoscan::Error> { return true; });

int main() {
  return 0;
}
