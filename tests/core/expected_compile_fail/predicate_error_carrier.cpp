/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/core/predicate.hpp>

struct PredicateError {};

// REQ: CONTRACT-010, COMPILE-016
auto invalid_predicate =
    holoscan::predicate("predicate-error",
                        [](const holoscan::PredicateEvaluationContext&) noexcept
                            -> holoscan::expected<bool, PredicateError> { return true; });

int main() {
  return 0;
}
