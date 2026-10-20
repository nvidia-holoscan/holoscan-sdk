/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstddef>
#include <cstdint>

#include <holoscan/core/tensor_contract.hpp>

namespace holoscan::detail {
namespace {

[[nodiscard]] TensorByteSpanParameterReadStatus read_positive_byte_span(
    ParamScalarValue value, std::uint64_t& converted) noexcept {
  std::int64_t signed_value{};
  if (value.try_as(signed_value)) {
    if (signed_value <= 0) {
      return TensorByteSpanParameterReadStatus::kValueInvalid;
    }
    converted = static_cast<std::uint64_t>(signed_value);
    return TensorByteSpanParameterReadStatus::kSuccess;
  }
  std::uint64_t unsigned_value{};
  if (!value.try_as(unsigned_value)) {
    return TensorByteSpanParameterReadStatus::kTypeInvalid;
  }
  if (unsigned_value == 0U) {
    return TensorByteSpanParameterReadStatus::kValueInvalid;
  }
  converted = unsigned_value;
  return TensorByteSpanParameterReadStatus::kSuccess;
}

}  // namespace

TensorByteSpanParameterReadStatus TensorByteSpanParameterDependency::read(
    const void* aggregate, TensorByteSpanParameterValue& result) const noexcept {
  if (hosted_field == nullptr) {
    return read_initial != nullptr ? read_initial(aggregate, result)
                                   : TensorByteSpanParameterReadStatus::kTypeInvalid;
  }
  if (!shape) {
    if (hosted_field->type_shape.type != ParamLogicalType::kSignedInteger &&
        hosted_field->type_shape.type != ParamLogicalType::kUnsignedInteger) {
      return TensorByteSpanParameterReadStatus::kTypeInvalid;
    }
    ParamScalarValue value;
    if (!read_param_field_scalar(*hosted_field, aggregate, value)) {
      return TensorByteSpanParameterReadStatus::kValueInvalid;
    }
    const auto status = read_positive_byte_span(value, result.values[0]);
    result.count = status == TensorByteSpanParameterReadStatus::kSuccess ? 1U : 0U;
    return status;
  }

  if (hosted_field->type_shape.type != ParamLogicalType::kSequence ||
      (hosted_field->type_shape.element_or_rep_type != ParamLogicalType::kSignedInteger &&
       hosted_field->type_shape.element_or_rep_type != ParamLogicalType::kUnsignedInteger)) {
    return TensorByteSpanParameterReadStatus::kTypeInvalid;
  }
  const auto values = read_param_field_scalar_sequence(*hosted_field, aggregate);
  if (values.empty() || values.size() > result.values.size()) {
    return TensorByteSpanParameterReadStatus::kValueInvalid;
  }
  for (std::size_t index = 0U; index < values.size(); ++index) {
    const auto status = read_positive_byte_span(values[index], result.values[index]);
    if (status != TensorByteSpanParameterReadStatus::kSuccess) {
      result.count = 0U;
      return status;
    }
  }
  result.count = values.size();
  return TensorByteSpanParameterReadStatus::kSuccess;
}

}  // namespace holoscan::detail
