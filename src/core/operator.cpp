/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/core/operator.hpp>

namespace holoscan {

OperatorBase::OperatorBase() = default;
OperatorBase::~OperatorBase() = default;

expected<void, Error> OperatorBase::invoke_compute(ExecutionContext& context) {
  return compute(context);
}

const ParamTypeDescriptor& OperatorBase::defining_param_type() const noexcept {
  return defining_param_type_impl();
}

const ParamSchemaDescriptor* OperatorBase::defining_param_schema() const noexcept {
  return defining_param_schema_impl();
}

}  // namespace holoscan
