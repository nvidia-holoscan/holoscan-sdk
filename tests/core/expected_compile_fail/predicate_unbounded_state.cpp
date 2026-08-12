/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <vector>

#include <holoscan/core/predicate.hpp>

using UnboundedState = std::vector<int>;

constexpr auto evaluator = [](const holoscan::PredicateEvaluationContext&,
                              UnboundedState&) noexcept { return true; };

// REQ: CONTRACT-010, COMPILE-016
constexpr bool invalid_state = holoscan::detail::validate_predicate_signature<
    UnboundedState, holoscan::detail::NoPredicateConfig, decltype(evaluator)>();

int main() {
  return invalid_state ? 0 : 1;
}
