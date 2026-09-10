/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <span>

#include <holoscan/core/graph.hpp>

namespace holoscan::detail {

[[nodiscard]] inline bool tensor_boundary_contract_equal(
    const TensorBoundaryContractRecord& left, const TensorBoundaryContractRecord& right) noexcept {
  const bool dtype_equal = left.dtype.has_value() == right.dtype.has_value() &&
                           (!left.dtype || (left.dtype->code == right.dtype->code &&
                                            left.dtype->bits == right.dtype->bits &&
                                            left.dtype->lanes == right.dtype->lanes));
  if (left.memory_kind != right.memory_kind || !dtype_equal || left.rank != right.rank ||
      left.max_rank != right.max_rank || left.max_byte_span != right.max_byte_span ||
      left.strides != right.strides || left.foreign_emit_policy != right.foreign_emit_policy ||
      left.max_inline_bytes != right.max_inline_bytes ||
      left.max_staging_byte_span != right.max_staging_byte_span ||
      left.memory_conversion != right.memory_conversion || left.axes.size() != right.axes.size()) {
    return false;
  }
  return std::equal(
      left.axes.begin(), left.axes.end(), right.axes.begin(), [](const auto& lhs, const auto& rhs) {
        return lhs.max_extent == rhs.max_extent &&
               lhs.max_abs_stride_bytes == rhs.max_abs_stride_bytes;
      });
}

[[nodiscard]] inline bool descriptor_tensor_fields_equal(
    const std::span<const DescriptorTensorTopicFieldRecord> left,
    const std::span<const DescriptorTensorTopicFieldRecord> right) noexcept {
  if (left.size() != right.size()) {
    return false;
  }
  return std::equal(left.begin(), left.end(), right.begin(), [](const auto& lhs, const auto& rhs) {
    return lhs.field_id == rhs.field_id && lhs.tensor_index == rhs.tensor_index &&
           tensor_boundary_contract_equal(lhs.tensor, rhs.tensor);
  });
}

}  // namespace holoscan::detail
