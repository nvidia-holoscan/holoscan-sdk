/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "parameter_update_test_support.hpp"

#include <algorithm>
#include <stdexcept>

namespace holoscan::test::parameter_update {

// Looks up a named ParameterFieldPlanRecord in the plan; throws if the field is absent.
// Used as the base helper by runtime_value_field and by tests needing a specific field.
[[nodiscard]] const ParameterFieldPlanRecord& parameter_field(const ExecutionPlan& plan,
                                                              std::string_view name) {
  const auto found = std::find_if(
      plan.parameter_fields().begin(),
      plan.parameter_fields().end(),
      [name](const ParameterFieldPlanRecord& field) { return field.canonical_name == name; });
  if (found == plan.parameter_fields().end()) {
    throw std::runtime_error{"runtime parameter test field is missing"};
  }
  return *found;
}

// Shorthand that returns the "value" field from standard runtime-param test plans.
[[nodiscard]] const ParameterFieldPlanRecord& runtime_value_field(const ExecutionPlan& plan) {
  return parameter_field(plan, "value");
}

// Builds a maximally permissive local-application access policy used as the
// baseline in tests that do not need restricted-scope verification.
[[nodiscard]] ParamAccessPolicy full_local_parameter_policy() noexcept {
  return ParamAccessPolicy{
      .origin = ParamControlOrigin::kLocalApplication,
      .permissions = ParamPermission::kRead | ParamPermission::kUpdate | ParamPermission::kAwait |
                     ParamPermission::kCapture | ParamPermission::kObserve,
      .max_event_cursors = 2U,
  };
}

// Constructs an OperatorRef from a field's plan-level owner stamped with a synthetic
// run_generation so the coordinator can be driven directly without a live RunSession.
[[nodiscard]] OperatorRef run_owner(const ParameterFieldPlanRecord& field,
                                    RunGenerationId run_generation) noexcept {
  OperatorRef owner = field.owner;
  owner.run_generation = run_generation;
  return owner;
}

}  // namespace holoscan::test::parameter_update
