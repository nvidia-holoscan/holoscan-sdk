/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <optional>
#include <vector>

#include <holoscan/core/tensor_field.hpp>

#include "plan_record_contracts.hpp"

namespace holoscan::detail {

[[nodiscard]] inline bool optional_tensor_contract_equal(
    const std::optional<TensorContractRecord>& lhs,
    const std::optional<TensorContractRecord>& rhs) noexcept {
  if (lhs.has_value() != rhs.has_value()) {
    return false;
  }
  if (!lhs) {
    return true;
  }
  return tensor_contract_equal(*lhs, *rhs);
}

[[nodiscard]] inline bool tensor_field_payload_options_equal(
    const std::optional<TensorFieldTransferRecord>& lhs,
    const std::optional<TensorFieldTransferRecord>& rhs) noexcept {
  return lhs.has_value() == rhs.has_value() &&
         (!lhs || (lhs->max_inline_bytes == rhs->max_inline_bytes &&
                   lhs->max_tensor_byte_span == rhs->max_tensor_byte_span &&
                   lhs->interface_tensor_emit_policy == rhs->interface_tensor_emit_policy));
}

[[nodiscard]] inline bool tensor_field_configuration_equal(
    const std::vector<TensorFieldConfigurationRecord>& lhs,
    const std::vector<TensorFieldConfigurationRecord>& rhs) noexcept {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  for (std::size_t index = 0U; index < lhs.size(); ++index) {
    const auto& left = lhs[index];
    const auto& right = rhs[index];
    const bool allocation_equal =
        left.tensor_allocation.has_value() == right.tensor_allocation.has_value() &&
        (!left.tensor_allocation ||
         (left.tensor_allocation->max_rank == right.tensor_allocation->max_rank &&
          left.tensor_allocation->max_byte_span == right.tensor_allocation->max_byte_span));
    if (left.field_id != right.field_id || left.tensor_index != right.tensor_index ||
        !tensor_port_layout_exactly_equal(left.tensor_layout, right.tensor_layout) ||
        left.memory_conversion != right.memory_conversion || !allocation_equal ||
        !tensor_field_payload_options_equal(left.payload_options, right.payload_options) ||
        !optional_tensor_contract_equal(left.tensor_contract, right.tensor_contract) ||
        left.tensor_byte_span_expression != right.tensor_byte_span_expression ||
        left.tensor_memory_kind_expression != right.tensor_memory_kind_expression) {
      return false;
    }
  }
  return true;
}

}  // namespace holoscan::detail
