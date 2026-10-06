/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <cstddef>
#include <optional>
#include <string_view>

#include <holoscan/core/execution_plan.hpp>

namespace holoscan::detail {

/// Return the stable enumerator name for a Tensor memory kind.
[[nodiscard]] inline constexpr std::string_view memory_kind_name(MemoryKind memory) noexcept {
  switch (memory) {
    case MemoryKind::kUnknown:
      return "kUnknown";
    case MemoryKind::kHost:
      return "kHost";
    case MemoryKind::kPinnedHost:
      return "kPinnedHost";
    case MemoryKind::kCudaDevice:
      return "kCudaDevice";
    case MemoryKind::kCudaManaged:
      return "kCudaManaged";
  }
  return "invalid";
}

/// Return whether two DLPack data types have identical code, width, and lane count.
[[nodiscard]] inline bool dl_data_type_exactly_equal(DLDataType left, DLDataType right) noexcept {
  return left.code == right.code && left.bits == right.bits && left.lanes == right.lanes;
}

/// Return whether two optional DLPack data types are both absent or exactly equal.
[[nodiscard]] inline bool dl_data_type_exactly_equal(
    const std::optional<DLDataType>& left, const std::optional<DLDataType>& right) noexcept {
  return left.has_value() == right.has_value() &&
         (!left || dl_data_type_exactly_equal(*left, *right));
}

/// Return whether two schema identities contain the same complete stable identity.
[[nodiscard]] inline bool schema_identity_exactly_equal(
    const SchemaIdentityRecord& left, const SchemaIdentityRecord& right) noexcept {
  return schema_identity_equal(left, right);
}

/// Return whether two optional schema identities are both absent or exactly equal.
[[nodiscard]] inline bool schema_identity_exactly_equal(
    const std::optional<SchemaIdentityRecord>& left,
    const std::optional<SchemaIdentityRecord>& right) noexcept {
  return left.has_value() == right.has_value() &&
         (!left || schema_identity_exactly_equal(*left, *right));
}

/// Return whether two POD identities contain the same complete ordered layout.
[[nodiscard]] inline bool pod_identity_exactly_equal(const PodIdentityRecord& left,
                                                     const PodIdentityRecord& right) noexcept {
  if (left.canonical_name != right.canonical_name || left.version != right.version ||
      left.object_size != right.object_size || left.object_alignment != right.object_alignment ||
      left.fields.size() != right.fields.size()) {
    return false;
  }
  return std::equal(left.fields.begin(),
                    left.fields.end(),
                    right.fields.begin(),
                    [](const auto& lhs, const auto& rhs) {
                      return lhs.name == rhs.name && lhs.byte_offset == rhs.byte_offset &&
                             lhs.scalar_kind == rhs.scalar_kind &&
                             lhs.element_count == rhs.element_count;
                    });
}

/// Return whether two payload identities contain the same complete tagged value.
[[nodiscard]] inline bool payload_identity_exactly_equal(
    const PayloadIdentityRecord& left, const PayloadIdentityRecord& right) noexcept {
  if (left.kind != right.kind || left.payload_kind != right.payload_kind ||
      left.transport_profile_tag != right.transport_profile_tag ||
      left.scalar.has_value() != right.scalar.has_value() ||
      left.schema.has_value() != right.schema.has_value() ||
      left.pod.has_value() != right.pod.has_value()) {
    return false;
  }
  return (!left.scalar || (left.scalar->kind == right.scalar->kind &&
                           left.scalar->native_object_size == right.scalar->native_object_size)) &&
         (!left.schema || schema_identity_exactly_equal(*left.schema, *right.schema)) &&
         (!left.pod || pod_identity_exactly_equal(*left.pod, *right.pod));
}

/// Return whether two optional Tensor port layouts are both absent or exactly equal.
[[nodiscard]] inline bool tensor_port_layout_exactly_equal(
    const std::optional<TensorRepresentation>& left,
    const std::optional<TensorRepresentation>& right) noexcept {
  return left.has_value() == right.has_value() &&
         (!left || (left->memory_kind == right->memory_kind && left->rank == right->rank &&
                    left->strides == right->strides &&
                    dl_data_type_exactly_equal(left->dtype, right->dtype)));
}

/// Return whether an allocation satisfies every Tensor property declared by an input port.
[[nodiscard]] inline bool tensor_allocation_satisfies_input(
    const TensorAllocationPlanRecord& allocation,
    const std::optional<TensorRepresentation>& input_layout) noexcept {
  if (!input_layout) {
    return true;
  }
  if (input_layout->memory_kind && *input_layout->memory_kind != allocation.memory_kind) {
    return false;
  }
  if (input_layout->dtype &&
      (!allocation.layout || !allocation.layout->dtype ||
       !dl_data_type_exactly_equal(*input_layout->dtype, *allocation.layout->dtype))) {
    return false;
  }
  if (input_layout->strides &&
      (!allocation.layout || allocation.layout->strides != input_layout->strides)) {
    return false;
  }
  return !input_layout->rank || (allocation.layout && allocation.layout->rank &&
                                 *input_layout->rank == *allocation.layout->rank);
}

/// Return whether two logical Tensor requirements contain the same complete ordered value.
[[nodiscard]] inline bool logical_tensor_requirements_exactly_equal(
    const TensorBoundaryContractRecord& left, const TensorBoundaryContractRecord& right) noexcept {
  if (left.memory_kind != right.memory_kind ||
      !dl_data_type_exactly_equal(left.dtype, right.dtype) || left.rank != right.rank ||
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

}  // namespace holoscan::detail
