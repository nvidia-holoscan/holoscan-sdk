/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <stdexcept>
#include <string_view>
#include <utility>

#include <holoscan/core/execution_plan.hpp>
#include <holoscan/core/parameter_control.hpp>

namespace holoscan::test::parameter_update {

template <typename Value>
[[nodiscard]] auto checked_value(Value value) {
  if (!value) {
    throw std::logic_error{"test attempted to access a disengaged optional or failed expected"};
  }
  // Test call sites use fatal assertions to document the expected state. Keep the single access
  // in this helper because clang-tidy cannot model GTest's fatal-assertion control flow.
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
  return std::move(value).value();
}

template <typename Value>
[[nodiscard]] decltype(auto) checked_ref(Value& value) {
  if (!value) {
    throw std::logic_error{"test attempted to access a disengaged optional or failed expected"};
  }
  // This overload cannot accept temporaries, so its reference remains tied to the local wrapper.
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
  return value.value();
}

[[nodiscard]] const ParameterFieldPlanRecord& parameter_field(const ExecutionPlan& plan,
                                                              std::string_view name);
[[nodiscard]] const ParameterFieldPlanRecord& runtime_value_field(const ExecutionPlan& plan);
[[nodiscard]] ParamAccessPolicy full_local_parameter_policy() noexcept;
[[nodiscard]] OperatorRef run_owner(const ParameterFieldPlanRecord& field,
                                    RunGenerationId run_generation = 101U) noexcept;

}  // namespace holoscan::test::parameter_update
