/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <holoscan/holoscan_core_export.h>
#include <holoscan/core/parameter.hpp>
#include <holoscan/core/parameter_schema.hpp>

namespace holoscan {

namespace detail {
struct HostedParameterSnapshotState;
}  // namespace detail

/// @brief Supply one frozen language-neutral field to a hosted Parameter snapshot.
///
/// A field carries either one fixed scalar or a bounded sequence of fixed integral scalars. The
/// owning builder validates that the value, shape, constraints, and mutability agree before any
/// descriptor becomes visible to compilation.
struct HostedParameterField {
  std::string name;                  ///< Stable canonical top-level field name.
  std::string description;           ///< Optional descriptive UTF-8 metadata.
  ParamFieldTypeShape type_shape{};  ///< Exact portable scalar or sequence shape.
  ParamMutability mutability{ParamMutability::kCompileTime};  ///< Frozen update disposition.
  ParamConstraints constraints{};        ///< Portable range, allowed-value, and sequence bounds.
  std::vector<ParamScalarValue> values;  ///< One scalar or a bounded scalar sequence.
};

/// @brief Own an immutable parameter schema and generation-zero value without language callbacks.
///
/// Copies share immutable storage. Descriptor strings, field records, and scalar values therefore
/// remain stable for as long as a Graph factory, compiled Plan, or session-owned Operator retains a
/// snapshot. Reading the snapshot never invokes a foreign runtime or acquires a language lock.
struct HOLOSCAN_CORE_EXPORT HostedParameterSnapshot {
  std::shared_ptr<const detail::HostedParameterSnapshotState> state;  ///< Immutable owned storage.

  /// @brief Return whether this snapshot carries a declared parameter schema.
  [[nodiscard]] explicit operator bool() const noexcept;

  /// @brief Return the exact process-local aggregate descriptor.
  [[nodiscard]] const ParamTypeDescriptor& type() const noexcept;

  /// @brief Return the owned immutable schema, or null for an empty snapshot.
  [[nodiscard]] const ParamSchemaDescriptor* schema() const noexcept;

  /// @brief Read one scalar field by descriptor.
  /// @return `true` only for a present exact scalar value.
  [[nodiscard]] bool read_scalar(const ParamFieldDescriptor& field,
                                 ParamScalarValue& value) const noexcept;

  /// @brief Read one bounded scalar sequence by descriptor.
  /// @return Empty for a foreign or non-sequence field; an authored empty sequence is valid.
  [[nodiscard]] std::span<const ParamScalarValue> read_scalar_sequence(
      const ParamFieldDescriptor& field) const noexcept;
};

/// @brief Build one immutable runtime-declared Parameter schema and value snapshot.
class HOLOSCAN_CORE_EXPORT HostedParameterSnapshotBuilder {
 public:
  /// @brief Start a complete version-one schema.
  /// @param stable_id Stable non-empty schema identity copied by the builder.
  /// @param description Optional descriptive UTF-8 text copied by the builder.
  explicit HostedParameterSnapshotBuilder(std::string stable_id, std::string description = {});

  /// @brief Append one field in declaration order for validation by \ref build.
  HostedParameterSnapshotBuilder& field(HostedParameterField value);

  /// @brief Freeze the complete schema and generation-zero values.
  /// @throws RuntimeError when the schema identity or any field is invalid.
  [[nodiscard]] HostedParameterSnapshot build() const;

 private:
  std::string stable_id_;
  std::string description_;
  std::vector<HostedParameterField> fields_;
};

template <>
struct ParamStorageTraits<HostedParameterSnapshot> {
  [[nodiscard]] static consteval ParamStorageTraitsSummary summary() noexcept {
    return ParamStorageTraitsSummary{
        .owns_dynamic_storage = true,
        .fixed_object_extent = false,
        .has_portable_codec = false,
        .supports_bounded_materialization = true,
        .inline_bytes = 0,
    };
  }
};

}  // namespace holoscan
