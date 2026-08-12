/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <optional>

#include <holoscan/core/operator_ref.hpp>
#include <holoscan/core/parameter_catalog.hpp>
#include <holoscan/core/parameter_override_set.hpp>
#include <holoscan/core/parameter_schema.hpp>

namespace holoscan::detail {

/// @brief Preserve exact privileged generation-zero value provenance.
enum class ParameterGenerationZeroOrigin : std::uint8_t {
  kDefault = 0,            ///< Value came from `ParamsT{}`.
  kTypedBaseline = 1,      ///< `OperatorHandle::set_params()` supplied the aggregate.
  kOverrideCaptured = 2,   ///< Canonical explicit field input supplied the value.
  kOverrideAlias = 3,      ///< A declared alias supplied the explicit field value.
  kOverrideMigration = 4,  ///< A qualified migration produced the explicit field value.
};

/// @brief Explain whether a protected canonical value digest exists.
enum class ParameterValueDigestDisposition : std::uint8_t {
  kCanonicalScalarFields = 0,  ///< Complete declared fixed scalars produced a semantic digest.
  kNoParams = 1,               ///< `NoParams` has no value-bearing fields.
  kOpaqueSchema = 2,           ///< No declared field inventory exists.
  kPartialSchema = 3,          ///< Declared fields do not cover the complete aggregate.
  kLocalTypedField = 4,        ///< At least one field lacks a canonical scalar read operation.
  kInvalidScalar = 5,          ///< A declared scalar could not produce a canonical carrier.
  kToolingCodec = 6,           ///< A bound whole-object Config codec produced the digest.
};

/// @brief Retain one protected semantic value digest without exposing it publicly.
struct ParameterProtectedValueDigest {
  ParameterValueDigestDisposition disposition{
      ParameterValueDigestDisposition::kOpaqueSchema};  ///< Total digest outcome.
  ParamSchemaHash digest{};  ///< Valid for canonical scalars or an admitted tooling codec.

  [[nodiscard]] bool available() const noexcept {
    return disposition == ParameterValueDigestDisposition::kCanonicalScalarFields ||
           disposition == ParameterValueDigestDisposition::kToolingCodec;
  }
};

/// @brief Retain privileged generation-zero facts for one Operator aggregate.
struct ParameterGenerationZeroOperatorRecord {
  OperatorRef owner{};  ///< Plan-bound Operator identity.
  ParameterGenerationZeroOrigin baseline_origin{
      ParameterGenerationZeroOrigin::kDefault};  ///< Defaults versus typed authoring baseline.
  ParameterProtectedValueDigest value_digest{};  ///< Final resolved semantic digest outcome.
};

/// @brief Retain privileged generation-zero facts for one declared field.
struct ParameterGenerationZeroFieldRecord {
  ParamInstanceId instance{};  ///< Plan-scoped field-instance projection.
  ParamFieldId field{};        ///< Schema-lineage-local field identity.
  ParamSchemaHash full_id{};   ///< Complete schema field identity.
  ParameterGenerationZeroOrigin origin{
      ParameterGenerationZeroOrigin::kDefault};  ///< Exact accepted contribution provenance.
  ParameterProtectedValueDigest value_digest{};  ///< Protected per-field semantic digest.
  std::optional<ParamConfigFieldProvenance>
      config_source;  ///< Exact document source for an explicit config contribution.
};

}  // namespace holoscan::detail
