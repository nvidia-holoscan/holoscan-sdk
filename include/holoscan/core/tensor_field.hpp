/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>

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

/// @brief Record per-field Tensor materialization policy without port-wide schema identity.
///
/// Schema identity remains a property of the enclosing port. These values apply only to the
/// selected Tensor field and therefore may differ between fields of one generated root.
struct TensorFieldPayloadOptions {
  std::optional<std::uint64_t> max_inline_bytes;      ///< Maximum inline-copy bytes, if enabled.
  std::optional<std::uint64_t> max_tensor_byte_span;  ///< Maximum descriptor-backed byte span.
  InterfaceTensorEmitPolicy interface_tensor_emit_policy{
      InterfaceTensorEmitPolicy::kRequireExportable};  ///< Boundary materialization policy.
};

/// @brief Declare the logical and conversion requirements for one input Tensor field.
///
/// Every member is consumer-side policy. An empty \ref memory_conversion leaves conversion at the
/// existing port and edge defaults.
struct TensorFieldInputContract {
  TensorPortLayout layout;  ///< Logical Tensor facts required from the producer.
  std::optional<MemoryConversionPolicy> memory_conversion;  ///< Optional conversion request.
};

/// @brief Declare the logical, allocation, and materialization facts for one output Tensor field.
///
/// \ref max_byte_span remains an explicit safety bound. When \ref max_rank is omitted, an exact
/// rank from \ref layout supplies the allocation rank bound, with logical rank zero requiring one
/// metadata slot. Omitting \ref payload_options preserves fail-closed export behavior. When an
/// explicit \c kCopyIfNeeded policy omits its staging span, \ref max_byte_span also bounds that
/// staging allocation.
struct TensorFieldOutputContract {
  TensorPortLayout layout;                ///< Logical Tensor facts produced by this field.
  std::size_t max_byte_span{};            ///< Positive maximum field allocation byte span.
  std::optional<std::uint32_t> max_rank;  ///< Explicit positive dynamic-rank bound, when needed.
  std::optional<TensorFieldPayloadOptions> payload_options;  ///< Optional copy/inline permission.
};

/// @brief Pair one generated Tensor field selector with its input contract.
/// @tparam SchemaT Owning generated descriptor-backed schema type.
template <typename SchemaT>
struct TensorFieldInputContractEntry {
  TensorFieldSelector<SchemaT> field;  ///< Generated schema-typed field selector.
  TensorFieldInputContract contract;   ///< Consumer contract for the selected field.
};

/// @brief Pair one generated Tensor field selector with its output contract.
/// @tparam SchemaT Owning generated descriptor-backed schema type.
template <typename SchemaT>
struct TensorFieldOutputContractEntry {
  TensorFieldSelector<SchemaT> field;  ///< Generated schema-typed field selector.
  TensorFieldOutputContract contract;  ///< Producer contract for the selected field.
};

/// @brief Create one typed input field-contract entry for aggregate declaration.
/// @tparam SchemaT Owning generated descriptor-backed schema type.
/// @param field Generated schema-typed field selector.
/// @param contract Consumer contract for the selected field.
/// @return Value suitable for an input \c tensor_fields aggregate.
template <typename SchemaT>
[[nodiscard]] constexpr TensorFieldInputContractEntry<SchemaT> tensor_field(
    TensorFieldSelector<SchemaT> field, TensorFieldInputContract contract) noexcept {
  return {.field = field, .contract = contract};
}

/// @brief Create one typed output field-contract entry for aggregate declaration.
/// @tparam SchemaT Owning generated descriptor-backed schema type.
/// @param field Generated schema-typed field selector.
/// @param contract Producer contract for the selected field.
/// @return Value suitable for a complete output \c tensor_fields aggregate.
template <typename SchemaT>
[[nodiscard]] constexpr TensorFieldOutputContractEntry<SchemaT> tensor_field(
    TensorFieldSelector<SchemaT> field, TensorFieldOutputContract contract) noexcept {
  return {.field = field, .contract = contract};
}

/// @brief Preserve one field-scoped Tensor declaration in canonical pointer-free form.
struct TensorFieldConfigurationRecord {
  std::uint16_t field_id{};                       ///< Canonical FlatBuffers field id.
  std::uint16_t tensor_index{};                   ///< Dense index in ascending field-id order.
  std::optional<TensorPortLayout> tensor_layout;  ///< Field-specific logical Tensor facts.
  std::optional<MemoryConversionPolicy> memory_conversion;   ///< Field conversion request.
  std::optional<TensorAllocationBounds> tensor_allocation;   ///< Field allocation bound.
  std::optional<TensorFieldPayloadOptions> payload_options;  ///< Field materialization policy.
};

}  // namespace holoscan
