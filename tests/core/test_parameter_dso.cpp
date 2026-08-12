/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <memory>

#include <gtest/gtest.h>

#include <core/operator_parameter_binding.hpp>
#include <core/operator_test_binding.hpp>

#include "parameter_dso_test_types.hpp"

namespace holoscan::test {
namespace {

// Custom deleter so the DSO-allocated operator is freed via its own module's destructor,
// not the host module's delete — critical for correct cross-DSO memory ownership.
struct DSOOperatorDeleter {
  void operator()(DSOParameterOperator* op) const noexcept { destroy_dso_parameter_operator(op); }
};

// REQ: PARAM-013
// When an operator is defined in a DSO, OperatorParameterBinding must resolve the
// type and schema descriptors from the defining module, not the host's local registry.
// This guards against ODR confusion when two modules define the same struct name.
TEST(ParameterDSO, BindingUsesOperatorDefiningModuleIdentity) {
  // Key setup: operator allocated in the DSO; host has its own copy of DSOParams
  // with different local_type_token / local_field_type_token values.
  std::unique_ptr<DSOParameterOperator, DSOOperatorDeleter> op{create_dso_parameter_operator()};
  ASSERT_NE(op, nullptr);

  const auto& host_type = param_type_descriptor<DSOParams>();
  const auto* const host_schema = param_schema_descriptor<DSOParams>();
  ASSERT_NE(host_schema, nullptr);
  EXPECT_NE(host_type.local_type_token, dso_aggregate_type_token());
  EXPECT_NE(host_schema->fields.front().local_operations.local_field_type_token,
            dso_field_type_token());

  // Key assertion: defining_type/defining_schema on the DSO-allocated operator return
  // the DSO's local tokens, which differ from the host-side tokens observed above.
  const auto& bound_type = detail::OperatorParameterBinding::defining_type(*op);
  const auto* const bound_schema = detail::OperatorParameterBinding::defining_schema(*op);
  ASSERT_NE(bound_schema, nullptr);
  EXPECT_EQ(bound_type.local_type_token, dso_aggregate_type_token());
  EXPECT_EQ(bound_schema->fields.front().local_operations.local_field_type_token,
            dso_field_type_token());

  detail::SyntheticOperatorBinding binding;
  ASSERT_TRUE(binding.declare_ports(*op));
  ASSERT_TRUE(binding.invoke_once(*op));
  ASSERT_NE(op->observed_params(), nullptr);
  EXPECT_EQ(op->observed_params(), &op->params());
  EXPECT_EQ(op->observed_params()->value, DSOFieldValue::kDefault);
}

}  // namespace
}  // namespace holoscan::test
