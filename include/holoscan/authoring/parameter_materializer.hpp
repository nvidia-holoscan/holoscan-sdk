/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>

#include <holoscan/holoscan_core_export.h>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/parameter.hpp>
#include <holoscan/core/parameter_schema.hpp>

namespace holoscan::authoring {

/// @file
/// @brief Prototype schema-driven authoring parameter materialization.

struct ParameterAssignment;

/// @brief Classify one language-neutral authoring parameter value.
enum class ParameterValueKind : std::uint8_t {
  kUnset,        ///< Missing carrier initialization; never a valid assignment value.
  kNone,         ///< Explicit absence; valid only for an optional schema field.
  kScalar,       ///< Fixed scalar carried by `ParamScalarValue`.
  kString,       ///< Borrowed string bytes valid for the materialization call.
  kSequence,     ///< Borrowed sequence of scalar or string values.
  kAssignments,  ///< Borrowed assignments for one nested aggregate.
};

/// @brief Carry one language-neutral authoring value.
struct ParameterValue {
  ParameterValueKind kind{ParameterValueKind::kUnset};  ///< Active value member.
  ParamScalarValue scalar{};                            ///< Scalar carrier when selected.
  std::string_view string{};                            ///< Borrowed bytes when selected.
  const ParameterValue* sequence_data{};                ///< Borrowed sequence storage.
  std::size_t sequence_size{};                          ///< Borrowed sequence element count.
  const ParameterAssignment* assignment_data{};         ///< Borrowed nested assignment storage.
  std::size_t assignment_count{};                       ///< Borrowed nested assignment count.

  /// @brief Construct an explicit absent value.
  /// @return Value accepted only by an optional schema field.
  [[nodiscard]] static constexpr ParameterValue none() noexcept {
    return ParameterValue{.kind = ParameterValueKind::kNone};
  }

  /// @brief Construct a fixed scalar value.
  /// @tparam T Supported fixed parameter scalar.
  /// @param value Scalar to encode.
  /// @return Language-neutral scalar carrier.
  template <typename T>
    requires holoscan::detail::kFixedParameterScalar<std::remove_cv_t<T>>
  [[nodiscard]] static constexpr ParameterValue from(T value) noexcept {
    return ParameterValue{.kind = ParameterValueKind::kScalar,
                          .scalar = ParamScalarValue::from(value)};
  }

  /// @brief Construct a borrowed string value.
  /// @param value Bytes valid until materialization returns.
  /// @return Language-neutral string carrier.
  [[nodiscard]] static constexpr ParameterValue from(std::string_view value) noexcept {
    return ParameterValue{.kind = ParameterValueKind::kString, .string = value};
  }

  /// @brief Construct a borrowed scalar/string sequence or nested assignment group.
  /// @tparam Element `ParameterValue` or `ParameterAssignment`.
  /// @tparam Extent Static or dynamic span extent.
  /// @param values Elements valid until materialization returns.
  /// @return Language-neutral sequence or nested aggregate carrier.
  template <typename Element, std::size_t Extent>
    requires(std::same_as<std::remove_cv_t<Element>, ParameterValue> ||
             std::same_as<std::remove_cv_t<Element>, ParameterAssignment>)
  [[nodiscard]] static constexpr ParameterValue from(std::span<Element, Extent> values) noexcept {
    if constexpr (std::same_as<std::remove_cv_t<Element>, ParameterValue>) {
      return ParameterValue{.kind = ParameterValueKind::kSequence,
                            .sequence_data = values.data(),
                            .sequence_size = values.size()};
    } else {
      return ParameterValue{.kind = ParameterValueKind::kAssignments,
                            .assignment_data = values.data(),
                            .assignment_count = values.size()};
    }
  }

  /// @brief View borrowed sequence elements after both recursive public types are complete.
  [[nodiscard]] constexpr std::span<const ParameterValue> sequence_values() const noexcept;

  /// @brief View borrowed nested assignments after both recursive public types are complete.
  [[nodiscard]] constexpr std::span<const ParameterAssignment> nested_assignments() const noexcept;
};

/// @brief Associate one canonical schema field name or unique alias with an authoring value.
struct ParameterAssignment {
  std::string_view name;  ///< Exact canonical field name or unique declared alias.
  ParameterValue value;   ///< Borrowed or inline value.
};

constexpr std::span<const ParameterValue> ParameterValue::sequence_values() const noexcept {
  return std::span<const ParameterValue>{sequence_data, sequence_size};
}

constexpr std::span<const ParameterAssignment> ParameterValue::nested_assignments() const noexcept {
  return std::span<const ParameterAssignment>{assignment_data, assignment_count};
}

namespace detail {

/// @brief Result of one canonical-name or alias lookup.
struct ParameterFieldLookup {
  const ParamFieldDescriptor* field{};  ///< Unique matching field, when present.
  bool ambiguous{};                     ///< More than one field matched the supplied name.
};

/// @brief Find one unique schema field by canonical name or declared alias.
[[nodiscard]] inline ParameterFieldLookup lookup_parameter_field(
    const ParamSchemaDescriptor& schema, std::string_view name) noexcept {
  ParameterFieldLookup result;
  for (const ParamFieldDescriptor& field : schema.fields) {
    if (field.canonical_name == name || (!field.alias.empty() && field.alias == name)) {
      if (result.field != nullptr) {
        result.ambiguous = true;
        return result;
      }
      result.field = &field;
    }
  }
  return result;
}

/// @brief Find one unique schema field by canonical name or declared alias.
/// @return Matching field, or null when absent or ambiguous.
[[nodiscard]] inline const ParamFieldDescriptor* find_parameter_field(
    const ParamSchemaDescriptor& schema, std::string_view name) noexcept {
  const auto result = lookup_parameter_field(schema, name);
  return result.ambiguous ? nullptr : result.field;
}

/// @brief Join one canonical field name to an optional parent path.
[[nodiscard]] inline std::string parameter_field_path(std::string_view prefix,
                                                      std::string_view field) {
  if (prefix.empty()) {
    return std::string{field};
  }
  std::string path{prefix};
  path.push_back('.');
  path.append(field);
  return path;
}

}  // namespace detail

/// @brief Own one aligned, live parameter aggregate created from schema assignments.
class HOLOSCAN_CORE_EXPORT MaterializedParameters {
 public:
  /// @brief Create an empty value.
  MaterializedParameters() = default;
  /// @brief Destroy the live aggregate and release its aligned storage.
  ~MaterializedParameters();

  MaterializedParameters(const MaterializedParameters&) = delete;
  MaterializedParameters& operator=(const MaterializedParameters&) = delete;
  /// @brief Move one aggregate owner.
  MaterializedParameters(MaterializedParameters&& other) noexcept;
  /// @brief Replace this owner by moving another aggregate.
  MaterializedParameters& operator=(MaterializedParameters&& other) noexcept;

  /// @brief Return the live erased aggregate pointer.
  /// @return Pointer suitable for the matching registered Operator thunk, or null when empty.
  [[nodiscard]] const void* data() const noexcept { return value_; }

  /// @brief Return the exact process-local aggregate descriptor used for materialization.
  /// @return Descriptor for the live value, or null when empty.
  [[nodiscard]] const ParamTypeDescriptor* type() const noexcept { return type_; }

  /// @brief Return whether a live aggregate is owned.
  /// @return `true` after successful materialization.
  [[nodiscard]] explicit operator bool() const noexcept { return value_ != nullptr; }

 private:
  MaterializedParameters(const ParamTypeDescriptor& type, void* value) noexcept
      : type_(&type), value_(value) {}
  void reset() noexcept;

  const ParamTypeDescriptor* type_{};
  void* value_{};

  friend expected<MaterializedParameters, Error> materialize_parameters(
      const ParamTypeDescriptor&, const ParamSchemaDescriptor*,
      std::span<const ParameterAssignment>);
};

/// @brief Default-construct and populate one exact parameter aggregate from named values.
/// @param type Exact aggregate storage descriptor from a registered Operator row.
/// @param schema Declared field schema, or null only when `assignments` is empty.
/// @param assignments Named values to coerce and validate. Omitted fields retain their defaults;
/// explicit `ParameterValue::none()` clears only optional fields.
/// @return Owning erased aggregate, or an `Error` failure as data.
/// @note This function translates allocation and aggregate-construction exceptions into failure
/// data; no exception escapes the materialization boundary. Diagnostic allocation failure retains
/// the error code with an empty message.
[[nodiscard]] HOLOSCAN_CORE_EXPORT expected<MaterializedParameters, Error> materialize_parameters(
    const ParamTypeDescriptor& type, const ParamSchemaDescriptor* schema,
    std::span<const ParameterAssignment> assignments);

}  // namespace holoscan::authoring
