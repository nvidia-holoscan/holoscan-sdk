/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/core/execution_context.hpp>

#include "parameter_dso_test_types.hpp"

namespace holoscan::test {

expected<void, Error> DSOParameterOperator::compute(ExecutionContext& context) {
  observed_params_ = &context.params<DSOParams>();
  return {};
}

DSOParameterOperator* create_dso_parameter_operator() {
  return new DSOParameterOperator{};
}

void destroy_dso_parameter_operator(DSOParameterOperator* op) noexcept {
  delete op;
}

const void* dso_aggregate_type_token() noexcept {
  return param_type_descriptor<DSOParams>().local_type_token;
}

const void* dso_field_type_token() noexcept {
  return param_schema_descriptor<DSOParams>()
      ->fields.front()
      .local_operations.local_field_type_token;
}

}  // namespace holoscan::test
