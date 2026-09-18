/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

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

/// @brief Classify one language-neutral authoring parameter value.
enum class ParameterValueKind : std::uint8_t {
  kScalar,  ///< Fixed scalar carried by `ParamScalarValue`.
  kString,  ///< Borrowed string bytes valid for the materialization call.
};

/// @brief Carry one language-neutral scalar or string authoring value.
struct ParameterValue {
  ParameterValueKind kind{ParameterValueKind::kScalar};  ///< Active value member.
  ParamScalarValue scalar{};                             ///< Scalar carrier when selected.
  std::string_view string{};                             ///< Borrowed bytes when selected.

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
};

/// @brief Associate one canonical schema field name or unique alias with an authoring value.
struct ParameterAssignment {
  std::string_view name;  ///< Exact canonical field name or unique declared alias.
  ParameterValue value;   ///< Borrowed or inline value.
};

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
/// @param assignments Named scalar/string values to coerce and validate. Assigned sequence values,
/// nested aggregates, and filesystem paths are rejected by this initial materializer with
/// `AUTHORING_PARAMETER_KIND_UNSUPPORTED`; its value carrier does not yet represent optional
/// clearing.
/// @return Owning erased aggregate, or an `Error` failure as data.
/// @note This function translates allocation and aggregate-construction exceptions into failure
/// data; no exception escapes the materialization boundary. Diagnostic allocation failure retains
/// the error code with an empty message.
[[nodiscard]] HOLOSCAN_CORE_EXPORT expected<MaterializedParameters, Error> materialize_parameters(
    const ParamTypeDescriptor& type, const ParamSchemaDescriptor* schema,
    std::span<const ParameterAssignment> assignments);

}  // namespace holoscan::authoring
