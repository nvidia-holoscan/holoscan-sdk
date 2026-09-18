/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <holoscan/core/graph.hpp>
#include <holoscan/core/parameter.hpp>
#include <holoscan/core/parameter_provenance.hpp>
#include <holoscan/core/parameter_schema.hpp>

namespace holoscan::detail {

/// @brief Own graph-free evidence produced with one immutable typed parameter value.
struct ParameterInputEvidenceState {
  const void* local_type_token{};
  std::string schema_id;
  ParamSchemaVersion schema_version{};
  ParamSchemaHash structural_hash{};
  ParamSchemaHash contract_hash{};
  bool explicit_baseline{};
  std::vector<ParamConfigFieldProvenance> config_fields;
};

/// @brief Retain one sparse field contribution in a graph-owned transaction.
struct ParameterTransactionFieldState {
  ParamFieldId field{};
  ParamSchemaHash full_id{};
  ParamFieldRebindOrigin origin{ParamFieldRebindOrigin::kCaptured};
  std::optional<std::uint32_t> config_provenance_index;
  std::optional<std::uint32_t> rebind_provenance_index;
  std::optional<ParamScalarValue> scalar_value;
  const void* staged_type_token{};
  std::shared_ptr<const void> local_value;
  bool (*local_value_digest)(const void*, ParamSchemaHash&) noexcept {};
  ParamSchemaHash staged_value_digest{};
};

enum class ParameterContributionOrigin : std::uint8_t {
  kCaptured = 0,
  kAlias = 1,
  kMigration = 2,
};

struct ParameterContributionRecord {
  ParamFieldId field{};
  ParamSchemaHash full_id{};
  ParameterContributionOrigin origin{ParameterContributionOrigin::kCaptured};
  std::optional<std::uint32_t> config_provenance_index;
};

/// @brief Retain one Operator target and its sparse staged fields.
struct ParameterTransactionOperatorState {
  VertexId target{};
  const void* local_type_token{};
  std::string schema_id;
  ParamSchemaVersion schema_version{};
  ParamSchemaHash structural_hash{};
  ParamSchemaHash contract_hash{};
  std::optional<Profile> required_source_profile;
  std::vector<ParameterTransactionFieldState> fields;
  // Contiguous target-owned range, emitted in transaction operator order.
  std::uint32_t provenance_begin{};
  std::uint32_t provenance_count{};
};

struct ParameterTransactionScopeState {
  ParamRebindTargetScopeKind kind{ParamRebindTargetScopeKind::kWholeGraph};
  std::optional<GraphId> graph;
  std::vector<VertexId> operators;
};

enum class ParameterTransactionSource : std::uint8_t {
  kConfig = 0,
  kSnapshot = 1,
};

/// @brief Own one immutable all-or-nothing advanced parameter input.
struct ParameterTransactionState {
  ParameterTransactionSource source{ParameterTransactionSource::kConfig};
  std::vector<ParameterTransactionOperatorState> operators;
  ParameterTransactionScopeState scope;
  ParamRebindProvenance provenance;
};

[[nodiscard]] inline ParamSchemaHash parameter_transaction_scalar_digest(
    ParamScalarValue value) noexcept {
  StableHashBuilder hash;
  hash.add("holoscan.parameter.transaction_scalar.v1");
  hash.add(value.kind);
  hash.add(value.bit_width);
  hash.add(value.raw_bits);
  return hash.finish();
}

}  // namespace holoscan::detail
