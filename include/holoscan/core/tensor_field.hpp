/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <optional>
#include <utility>

#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/payload_options.hpp>

namespace holoscan {

/// @brief Identify one Tensor member of an owning descriptor-backed schema type.
///
/// Generated schema traits publish readable constants of this type. The FlatBuffers field id is
/// the canonical identity retained in Graph and plan records; \ref tensor_index is its resolved
/// position in ascending field-id order. Generated accessors, C++ member pointers, and field names
/// are not persisted as semantic identity.
///
/// @tparam SchemaT Owning generated schema type exposed on an Input or Output port.
template <typename SchemaT>
struct TensorFieldSelector {
  using schema_type = SchemaT;  ///< Owning generated schema type.

  std::uint16_t field_id{};      ///< Canonical FlatBuffers field id.
  std::uint16_t tensor_index{};  ///< Dense index in ascending field-id order.

  friend constexpr bool operator==(const TensorFieldSelector&,
                                   const TensorFieldSelector&) noexcept = default;
};

/// @brief Preserve normalized per-field Tensor transfer policy without schema identity.
///
/// Schema identity remains a property of the enclosing port. These values apply only to the
/// selected Tensor field and therefore may differ between fields of one generated root.
struct TensorFieldTransferRecord {
  std::optional<std::uint64_t> max_inline_bytes;      ///< Maximum inline-copy bytes, if enabled.
  std::optional<std::uint64_t> max_tensor_byte_span;  ///< Maximum descriptor-backed byte span.
  InterfaceTensorEmitPolicy interface_tensor_emit_policy{
      InterfaceTensorEmitPolicy::kRequireExportable};  ///< Boundary materialization policy.
};

/// @brief Create one typed input field entry from the common Tensor input spec.
template <typename SchemaT>
[[nodiscard]] constexpr std::pair<TensorFieldSelector<SchemaT>, TensorInputSpec> tensor_field(
    TensorFieldSelector<SchemaT> field, TensorInputSpec spec) noexcept {
  return {field, spec};
}

/// @brief Create one typed output field entry from the common Tensor output spec.
template <typename SchemaT>
[[nodiscard]] constexpr std::pair<TensorFieldSelector<SchemaT>, TensorOutputSpec> tensor_field(
    TensorFieldSelector<SchemaT> field, TensorOutputSpec spec) noexcept {
  return {field, spec};
}

/// @brief Preserve one field-scoped Tensor declaration in canonical pointer-free form.
struct TensorFieldConfigurationRecord {
  std::uint16_t field_id{};                           ///< Canonical FlatBuffers field id.
  std::uint16_t tensor_index{};                       ///< Dense index in ascending field-id order.
  std::optional<TensorRepresentation> tensor_layout;  ///< Field-specific logical Tensor facts.
  std::optional<MemoryConversionPolicy> memory_conversion;   ///< Field conversion request.
  std::optional<TensorPoolBounds> tensor_allocation;         ///< Field allocation bound.
  std::optional<TensorFieldTransferRecord> payload_options;  ///< Field materialization policy.
  std::optional<TensorContractRecord> tensor_contract;       ///< Owned normalized contract.
  /// Immutable compile-derived slot-capacity expression for this field, when authored.
  std::optional<TensorByteSpanExpressionRecord> tensor_byte_span_expression;
  /// Immutable compile-derived memory-kind expression for this field, when authored.
  std::optional<TensorMemoryKindExpressionRecord> tensor_memory_kind_expression;
};

}  // namespace holoscan
