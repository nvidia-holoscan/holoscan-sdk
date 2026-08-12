/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/graph.hpp>
#include <holoscan/core/parameter.hpp>
#include <holoscan/core/parameter_override_set.hpp>
#include <holoscan/core/parameter_schema.hpp>

namespace holoscan::detail {

/// @brief Classify one normalized explicit contribution retained by an override set.
enum class ParamOverrideContributionOrigin : std::uint8_t {
  kCaptured = 0,   ///< The canonical field spelling supplied the value.
  kAlias = 1,      ///< A declared alias supplied the value.
  kMigration = 2,  ///< A qualified migration produced the value.
};

/// @brief Retain one exact field contribution without storing a borrowed field value.
struct ParamOverrideFieldOriginRecord {
  ParamFieldId field{};       ///< Compact collision-checked field identity.
  ParamSchemaHash full_id{};  ///< Complete canonical field identity.
  ParamOverrideContributionOrigin origin{
      ParamOverrideContributionOrigin::kCaptured};  ///< Normalized contribution source.
  std::optional<std::uint32_t>
      config_provenance_index;  ///< Matching public config provenance record, when present.
};

/// @brief Own one fully resolved aggregate override for an exact Operator path.
///
/// EA1-018 constructs this private record after parsing, alias resolution, and
/// migration. EA1-009 revalidates every identity and the baseline digest before
/// cloning the resolved aggregate into the plan-owned executable factory. The
/// erased value owns an exact immutable C++ aggregate; no caller storage is
/// borrowed.
struct ParamOverrideOperatorState {
  std::string operator_path;                ///< Canonical target Operator path.
  std::string schema_id;                    ///< Exact target schema identity.
  ParamSchemaVersion schema_version{};      ///< Exact target schema version.
  ParamSchemaHash structural_hash{};        ///< Target decoding-shape identity.
  ParamSchemaHash contract_hash{};          ///< Target policy identity.
  const ParamTypeDescriptor* local_type{};  ///< Process-local exact aggregate descriptor.
  bool (*local_value_digest)(const void*,
                             ParamSchemaHash&) noexcept {};  ///< Optional codec digest.
  std::shared_ptr<const void> resolved_value;  ///< Owned immutable resolved aggregate.
  ParamSchemaHash baseline_value_digest{};     ///< Exact defaults-plus-typed-baseline identity.
  ParamSchemaHash resolved_value_digest{};     ///< Exact resolved semantic value identity.
  std::vector<ParamOverrideFieldOriginRecord>
      contributions;  ///< Explicit fields only; order is not trusted by compilation.
};

// The state remains private so contribution/provenance records can evolve without exposing parser,
// persistence, Graph, or runtime types in Core headers. Every variable-size field is state-owned.
struct ParamOverrideSetState {
  std::uint32_t representation_version{1};
  std::array<std::byte, 32> target_identity{};
  std::optional<GraphAuthoringIdentity> target_graph;  ///< Exact graph/version/hash binding.
  ParamSchemaHash admission_context_digest{};          ///< Producer-bound policy context identity.
  std::vector<ParamOverrideOperatorState> operators;   ///< Fully resolved Operator aggregates.
  ParamRebindProvenance provenance;   ///< Owning inspectable snapshot/config source provenance.
  ParamSchemaHash semantic_digest{};  ///< Canonical value-bearing identity for plan generation.
};

/// @brief Construct and inspect opaque override sets for compiler/tooling integration and tests.
class ParamOverrideSetAccess {
 public:
  [[nodiscard]] static ParamOverrideSet make(std::array<std::byte, 32> target_identity = {}) {
    auto state = std::make_shared<ParamOverrideSetState>();
    state->target_identity = target_identity;
    return ParamOverrideSet{std::move(state)};
  }

  /// @brief Construct a complete compiler-consumable override set.
  /// @param target Exact immutable Graph authoring identity.
  /// @param operators Fully resolved owning Operator overrides.
  /// @param admission_context_digest Producer-bound policy context identity.
  /// @param provenance Producer-validated, bounded, owning reconstruction evidence.
  /// @return Opaque immutable override set with representation version two.
  /// @pre Every variable-size provenance collection and text field satisfies the package-specific
  /// producer bounds. This private constructor normalizes identities but does not grant authority
  /// or silently truncate evidence.
  [[nodiscard]] static ParamOverrideSet make_complete(
      GraphAuthoringIdentity target, std::vector<ParamOverrideOperatorState> operators,
      ParamSchemaHash admission_context_digest = {}, ParamRebindProvenance provenance = {}) {
    for (auto& op : operators) {
      std::sort(op.contributions.begin(), op.contributions.end(), contribution_less);
    }
    std::sort(operators.begin(), operators.end(), [](const auto& lhs, const auto& rhs) {
      return lhs.operator_path < rhs.operator_path;
    });
    auto state = std::make_shared<ParamOverrideSetState>();
    state->representation_version = 2U;
    state->target_identity = target.hash.value;
    state->target_graph = target;
    state->admission_context_digest = admission_context_digest;
    state->operators = std::move(operators);
    provenance.target = target;
    provenance.admission_context_digest = admission_context_digest;
    state->provenance = std::move(provenance);
    state->semantic_digest = compute_semantic_digest(*state);
    return ParamOverrideSet{std::move(state)};
  }

  /// @brief Own one exact fully resolved typed aggregate for a complete set.
  /// @tparam ParamsT Exact Operator parameter aggregate.
  /// @param operator_path Canonical target Operator path.
  /// @param value Fully resolved aggregate after explicit contribution application.
  /// @param baseline_value_digest Digest of defaults plus typed authoring baseline.
  /// @param resolved_value_digest Digest of the fully resolved aggregate.
  /// @param contributions Explicit normalized field origins.
  /// @return Private owning record suitable for \ref make_complete.
  template <ParameterAggregate ParamsT>
  [[nodiscard]] static ParamOverrideOperatorState resolved_operator(
      std::string operator_path, ParamsT value, ParamSchemaHash baseline_value_digest,
      ParamSchemaHash resolved_value_digest,
      std::vector<ParamOverrideFieldOriginRecord> contributions) {
    const ParamSchemaDescriptor* const schema = param_schema_descriptor<ParamsT>();
    auto owned = std::make_shared<const ParamsT>(std::move(value));
    return ParamOverrideOperatorState{
        .operator_path = std::move(operator_path),
        .schema_id = schema == nullptr ? std::string{} : std::string{schema->stable_id},
        .schema_version = schema == nullptr ? ParamSchemaVersion{} : schema->version,
        .structural_hash = schema == nullptr ? ParamSchemaHash{} : schema->structural_hash,
        .contract_hash = schema == nullptr ? ParamSchemaHash{} : schema->contract_hash,
        .local_type = &param_type_descriptor<ParamsT>(),
        .local_value_digest = nullptr,
        .resolved_value = std::move(owned),
        .baseline_value_digest = baseline_value_digest,
        .resolved_value_digest = resolved_value_digest,
        .contributions = std::move(contributions),
    };
  }

  /// @brief Return the protected canonical semantic digest stored by a complete set.
  /// @param value Opaque immutable override set.
  /// @return Digest covering exact target, admission context, schemas, values, and origins.
  [[nodiscard]] static ParamSchemaHash semantic_digest(const ParamOverrideSet& value) noexcept {
    return state(value).semantic_digest;
  }

  /// @brief Recompute the protected canonical semantic digest from stored records.
  /// @param value Opaque immutable override set.
  /// @return Digest over canonical Operator/contribution order and accepted source-record order,
  /// independent of process-local pointers.
  [[nodiscard]] static ParamSchemaHash recompute_semantic_digest(
      const ParamOverrideSet& value) noexcept {
    return compute_semantic_digest(state(value));
  }

  /// @brief Compute the domain-separated identity of one canonical rebind target selection.
  /// @param kind Exact caller-declared destination scope category.
  /// @param scope_paths Canonical exact paths that define the scope.
  /// @param target_operators Canonical selected Operator/schema records.
  /// @return Stable digest independent of process-local factories and caller declaration order.
  [[nodiscard]] static ParamSchemaHash target_selection_digest(
      ParamRebindTargetScopeKind kind, std::span<const std::string> scope_paths,
      std::span<const ParamRebindTargetOperatorProvenance> target_operators) noexcept {
    StableHashBuilder hash;
    hash.add("holoscan.param.rebind_target_selection.v1");
    hash.add(kind);
    hash.add_canonical_size(scope_paths.size());
    for (const auto& path : scope_paths) {
      hash.add(path);
    }
    hash.add_canonical_size(target_operators.size());
    for (const auto& op : target_operators) {
      hash.add(op.operator_path);
      hash.add(op.schema_id);
      hash.add(op.schema_version);
      add_hash(hash, op.structural_hash);
      add_hash(hash, op.contract_hash);
    }
    return hash.finish();
  }

  [[nodiscard]] static const ParamOverrideSetState& state(const ParamOverrideSet& value) noexcept {
    return *value.state_;
  }

 private:
  static void add_hash(StableHashBuilder& hash, ParamSchemaHash value) noexcept {
    for (const std::byte byte : value.value) {
      hash.add_byte(std::to_integer<std::uint8_t>(byte));
    }
  }

  static void add_graph_identity(StableHashBuilder& hash,
                                 const GraphAuthoringIdentity& identity) noexcept {
    for (const std::byte byte : identity.graph.value) {
      hash.add_byte(std::to_integer<std::uint8_t>(byte));
    }
    hash.add(identity.version);
    for (const std::byte byte : identity.hash.value) {
      hash.add_byte(std::to_integer<std::uint8_t>(byte));
    }
  }

  static void add_text(StableHashBuilder& hash, const std::string& value) noexcept {
    hash.add(value);
  }

  static void add_config_document(StableHashBuilder& hash,
                                  const ParamConfigDocumentProvenance& document) noexcept {
    hash.add(document.semantic_version);
    for (const std::byte byte : document.semantic_digest) {
      hash.add_byte(std::to_integer<std::uint8_t>(byte));
    }
    hash.add(document.has_raw_source);
    hash.add(document.raw_algorithm_version);
    for (const std::byte byte : document.raw_digest) {
      hash.add_byte(std::to_integer<std::uint8_t>(byte));
    }
    hash.add(document.raw_byte_size);
  }

  static void add_provenance(StableHashBuilder& hash,
                             const ParamRebindProvenance& provenance) noexcept {
    hash.add(provenance.source_archive_semantic_version);
    for (const std::byte byte : provenance.source_archive_digest) {
      hash.add_byte(std::to_integer<std::uint8_t>(byte));
    }
    for (const std::byte byte : provenance.source_plan_digest.value) {
      hash.add_byte(std::to_integer<std::uint8_t>(byte));
    }
    hash.add(provenance.source_plan_generation);
    hash.add(provenance.source_run_generation);
    for (const std::byte byte : provenance.source_catalog_digest.value) {
      hash.add_byte(std::to_integer<std::uint8_t>(byte));
    }
    hash.add(provenance.source_scope_kind);
    add_text(hash, provenance.source_scope_path);
    hash.add(provenance.mode);
    add_graph_identity(hash, provenance.target);
    hash.add(provenance.target_profile);
    add_hash(hash, provenance.admission_context_digest);
    hash.add(provenance.target_scope_kind);
    hash.add_canonical_size(provenance.target_scope_paths.size());
    for (const auto& path : provenance.target_scope_paths) {
      add_text(hash, path);
    }
    hash.add_canonical_size(provenance.target_operators.size());
    for (const auto& op : provenance.target_operators) {
      add_text(hash, op.operator_path);
      add_text(hash, op.schema_id);
      hash.add(op.schema_version);
      add_hash(hash, op.structural_hash);
      add_hash(hash, op.contract_hash);
    }
    add_hash(hash, provenance.target_selection_digest);
    hash.add_canonical_size(provenance.migrations.size());
    for (const auto& migration : provenance.migrations) {
      add_hash(hash, migration.evidence_id);
      add_text(hash, migration.from_schema_id);
      hash.add(migration.from_schema_version);
      add_text(hash, migration.to_schema_id);
      hash.add(migration.to_schema_version);
    }
    hash.add_canonical_size(provenance.operator_aliases.size());
    for (const auto& alias : provenance.operator_aliases) {
      add_text(hash, alias.from_path);
      hash.add(alias.from_schema_version);
      add_text(hash, alias.to_path);
      hash.add(alias.to_schema_version);
    }
    hash.add_canonical_size(provenance.field_aliases.size());
    for (const auto& alias : provenance.field_aliases) {
      add_text(hash, alias.from_schema_id);
      hash.add(alias.from_schema_version);
      hash.add(alias.from_field_id);
      add_text(hash, alias.to_schema_id);
      hash.add(alias.to_schema_version);
      hash.add(alias.to_field_id);
    }
    hash.add_canonical_size(provenance.fields.size());
    for (const auto& field : provenance.fields) {
      add_text(hash, field.source_operator_path);
      add_text(hash, field.source_schema_id);
      hash.add(field.source_schema_version);
      hash.add(field.source_field_id);
      add_hash(hash, field.source_full_id);
      add_text(hash, field.target_operator_path);
      add_text(hash, field.target_schema_id);
      hash.add(field.target_schema_version);
      hash.add(field.target_field_id);
      add_hash(hash, field.target_full_id);
      hash.add(field.origin);
      hash.add(field.disposition);
    }
    hash.add_canonical_size(provenance.config_fields.size());
    for (const auto& field : provenance.config_fields) {
      add_config_document(hash, field.document);
      add_text(hash, field.source_path);
      hash.add(field.source_ordinal);
      add_text(hash, field.target_operator_path);
      add_text(hash, field.target_schema_id);
      hash.add(field.target_schema_version);
      hash.add(field.target_field_id);
      add_hash(hash, field.target_full_id);
      hash.add(field.origin);
    }
  }

  [[nodiscard]] static bool contribution_less(const ParamOverrideFieldOriginRecord& lhs,
                                              const ParamOverrideFieldOriginRecord& rhs) noexcept {
    if (lhs.field != rhs.field) {
      return lhs.field < rhs.field;
    }
    if (lhs.full_id.value != rhs.full_id.value) {
      return lhs.full_id.value < rhs.full_id.value;
    }
    return lhs.origin < rhs.origin;
  }

  [[nodiscard]] static ParamSchemaHash compute_semantic_digest(
      const ParamOverrideSetState& state) noexcept {
    StableHashBuilder hash;
    hash.add("holoscan.param.override_set.v1");
    hash.add(state.representation_version);
    for (const std::byte byte : state.target_identity) {
      hash.add_byte(std::to_integer<std::uint8_t>(byte));
    }
    hash.add(state.target_graph.has_value());
    if (state.target_graph) {
      add_graph_identity(hash, *state.target_graph);
    }
    add_hash(hash, state.admission_context_digest);
    add_provenance(hash, state.provenance);
    hash.add_canonical_size(state.operators.size());
    for (const auto& op : state.operators) {
      hash.add(op.operator_path);
      hash.add(op.schema_id);
      hash.add(op.schema_version);
      add_hash(hash, op.structural_hash);
      add_hash(hash, op.contract_hash);
      add_hash(hash, op.baseline_value_digest);
      add_hash(hash, op.resolved_value_digest);
      hash.add_canonical_size(op.contributions.size());
      for (const auto& contribution : op.contributions) {
        hash.add(contribution.field);
        add_hash(hash, contribution.full_id);
        hash.add(contribution.origin);
        hash.add(contribution.config_provenance_index.has_value());
        if (contribution.config_provenance_index) {
          hash.add(*contribution.config_provenance_index);
        }
      }
    }
    return hash.finish();
  }
};

}  // namespace holoscan::detail
