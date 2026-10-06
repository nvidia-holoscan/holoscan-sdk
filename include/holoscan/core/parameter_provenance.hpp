/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <holoscan/core/execution_plan.hpp>
#include <holoscan/core/parameter_catalog.hpp>
#include <holoscan/core/profile.hpp>

namespace holoscan {

enum class ParamSnapshotScopeKind : std::uint8_t;

/// @brief Select sparse contribution or explicit complete-object rebinding provenance.
enum class ParamRebindMode : std::uint8_t {
  kSparseContributions = 0,  ///< Preserve target authoring baselines for absent fields.
  kCompleteObject = 1,       ///< Begin with type defaults and require complete eligible input.
};

/// @brief Classify the exact destination authority captured for snapshot rebinding.
enum class ParamRebindTargetScopeKind : std::uint8_t {
  kWholeGraph = 0,   ///< Strictly address every Operator in the destination Graph.
  kGraph = 1,        ///< Address every Operator below one exact embedded Graph path.
  kOperator = 2,     ///< Address one exact Operator path.
  kOperatorSet = 3,  ///< Address one explicit canonical set of exact Operator paths.
};

/// @brief Classify the normalized origin of one rebound or configuration-sourced field.
enum class ParamFieldRebindOrigin : std::uint8_t {
  kCaptured = 0,   ///< Canonical source field supplied the value directly.
  kAlias = 1,      ///< One qualified directed alias supplied the value.
  kMigration = 2,  ///< One or more qualified migration steps produced the value.
  kBaseline = 3,   ///< Sparse mode retained the target authoring baseline.
  kDefault = 4,    ///< Complete mode retained the target type default.
};

/// @brief Classify how one source/target field participated in rebinding.
enum class ParamFieldRebindDisposition : std::uint8_t {
  kContributed = 0,       ///< An owning explicit contribution entered the parameter input.
  kRetainedBaseline = 1,  ///< Sparse mode retained the target authoring baseline.
  kAppliedDefault = 2,    ///< Complete mode retained the target type default.
};

/// @brief Own one qualified migration step retained as non-callable provenance.
struct ParamRebindMigrationProvenance {
  ParamSchemaHash evidence_id{};        ///< Stable package-qualified evidence identity.
  std::string from_schema_id;           ///< Exact source schema identity.
  std::uint32_t from_schema_version{};  ///< Exact source schema version.
  std::string to_schema_id;             ///< Exact target schema identity.
  std::uint32_t to_schema_version{};    ///< Exact target schema version.
};

/// @brief Own one exact directed Operator alias retained after rebinding.
struct ParamRebindOperatorAliasProvenance {
  std::string from_path;                ///< Canonical source Operator path.
  std::uint32_t from_schema_version{};  ///< Exact source schema version.
  std::string to_path;                  ///< Canonical target Operator path.
  std::uint32_t to_schema_version{};    ///< Exact target schema version.
};

/// @brief Own one exact directed field alias retained after rebinding.
struct ParamRebindFieldAliasProvenance {
  std::string from_schema_id;           ///< Exact source schema identity.
  std::uint32_t from_schema_version{};  ///< Exact source schema version.
  ParamFieldId from_field_id{};         ///< Exact source compact field identity.
  std::string to_schema_id;             ///< Exact target schema identity.
  std::uint32_t to_schema_version{};    ///< Exact target schema version.
  ParamFieldId to_field_id{};           ///< Exact target compact field identity.
};

/// @brief Own one archive-to-target field provenance record.
struct ParamRebindFieldProvenance {
  std::string source_operator_path;       ///< Canonical source Operator path.
  std::string source_schema_id;           ///< Exact source schema identity.
  std::uint32_t source_schema_version{};  ///< Exact source schema version.
  ParamFieldId source_field_id{};         ///< Source compact field identity.
  ParamSchemaHash source_full_id{};       ///< Complete source field identity.
  std::string target_operator_path;       ///< Canonical target Operator path.
  std::string target_schema_id;           ///< Exact target schema identity.
  std::uint32_t target_schema_version{};  ///< Exact target schema version.
  ParamFieldId target_field_id{};         ///< Target compact field identity.
  ParamSchemaHash target_full_id{};       ///< Complete target field identity.
  ParamFieldRebindOrigin origin{ParamFieldRebindOrigin::kCaptured};  ///< Normalized origin.
  ParamFieldRebindDisposition disposition{
      ParamFieldRebindDisposition::kContributed};  ///< Contribution/default/baseline result.
};

/// @brief Own one exact schema-bearing Operator admitted into a rebind target scope.
struct ParamRebindTargetOperatorProvenance {
  std::string operator_path;          ///< Canonical exact destination Operator path.
  std::string schema_id;              ///< Exact destination schema identity.
  std::uint32_t schema_version{};     ///< Exact nonzero destination schema version.
  ParamSchemaHash structural_hash{};  ///< Exact destination decoding-shape identity.
  ParamSchemaHash contract_hash{};    ///< Exact destination policy identity.
  friend bool operator==(const ParamRebindTargetOperatorProvenance&,
                         const ParamRebindTargetOperatorProvenance&) noexcept = default;
};

/// @brief Own immutable configuration identity used by explicit field bindings.
struct ParamConfigDocumentProvenance {
  std::uint32_t semantic_version{};             ///< Configuration semantic digest version.
  std::array<std::byte, 32> semantic_digest{};  ///< Exact semantic configuration digest.
  bool has_raw_source{};                        ///< Whether retained current bytes are available.
  std::uint32_t raw_algorithm_version{};        ///< Current-byte digest version when available.
  std::array<std::byte, 32> raw_digest{};       ///< Current retained-byte digest, when available.
  std::uint64_t raw_byte_size{};                ///< Current retained-byte count, when available.
  bool has_base_raw_source{};                   ///< Whether parsed-base evidence is available.
  std::uint32_t base_raw_algorithm_version{};   ///< Base digest version when available.
  std::array<std::byte, 32> base_raw_digest{};  ///< Parsed-base digest, when available.
  std::uint64_t base_raw_byte_size{};           ///< Parsed-base byte count, when available.
};

/// @brief Own one configuration field's exact source and resolved target identity.
///
/// Configuration conversion accepts at most 65,536 records in one transaction and bounds each
/// retained RFC 6901 source path to 512 bytes. These records are immutable evidence only. Neither
/// their path nor ordinal grants Graph, file-system, or Runtime authority.
struct ParamConfigFieldProvenance {
  ParamConfigDocumentProvenance document{};  ///< Immutable source configuration identity.
  std::string source_path;                   ///< Exact RFC 6901 source field path.
  std::uint32_t source_ordinal{};            ///< Zero-based explicit contribution ordinal.
  std::string target_operator_path;          ///< Canonical target Operator path.
  std::string target_schema_id;              ///< Exact target schema identity.
  std::uint32_t target_schema_version{};     ///< Exact target schema version.
  ParamFieldId target_field_id{};            ///< Target compact field identity.
  ParamSchemaHash target_full_id{};          ///< Complete target field identity.
  ParamFieldRebindOrigin origin{ParamFieldRebindOrigin::kCaptured};  ///< Canonical or alias.
};

/// @brief Inspect immutable owning snapshot or configuration conversion provenance.
///
/// Empty archive fields identify a live snapshot or configuration-produced input. Configuration
/// records use config_fields; snapshot rebinding uses the ordered migration, alias, and field
/// records. No member carries authority, a callable, a borrowed view, or a process-local address.
/// Configuration-produced inputs contain at most 1,024 source bindings and 65,536
/// \ref ParamConfigFieldProvenance records. Snapshot rebinding accepts at most 16 migrations,
/// 1,024 Operator aliases, 65,536 field aliases, 65,536 source Operators, 65,536 target Operators,
/// 65,536 source fields, and 65,536 target fields. Canonical paths are bounded to 512 bytes and
/// schema/type identifiers to 256 bytes. Compilation never accepts an unvalidated producer record
/// merely because it is present here.
struct ParamRebindProvenance {
  std::uint32_t source_archive_semantic_version{};    ///< Zero without an archive identity.
  std::array<std::byte, 32> source_archive_digest{};  ///< Archive semantic digest, if present.
  PlanDigest source_plan_digest{};                    ///< Source snapshot plan identity.
  PlanGenerationId source_plan_generation{};          ///< Source snapshot plan generation.
  RunGenerationId source_run_generation{};            ///< Source snapshot run generation.
  ParamCatalogDigest source_catalog_digest{};         ///< Source snapshot catalog identity.
  ParamSnapshotScopeKind source_scope_kind{};         ///< Source snapshot scope category.
  std::string source_scope_path;                      ///< Canonical source scope path, if any.
  ParamRebindMode mode{ParamRebindMode::kSparseContributions};  ///< Reconstruction mode.
  ParamRebindTargetScopeKind target_scope_kind{
      ParamRebindTargetScopeKind::kWholeGraph};  ///< Destination authority category.
  std::vector<std::string> target_scope_paths;   ///< Caller-supplied exact target paths.
  std::vector<ParamRebindTargetOperatorProvenance>
      target_operators;  ///< Schema-bearing Operators admitted by the target scope.
  std::vector<ParamRebindMigrationProvenance> migrations;  ///< Ordered qualified migrations.
  std::vector<ParamRebindOperatorAliasProvenance> operator_aliases;  ///< Applied aliases.
  std::vector<ParamRebindFieldAliasProvenance> field_aliases;        ///< Applied field aliases.
  std::vector<ParamRebindFieldProvenance> fields;         ///< Ordered archive field outcomes.
  std::vector<ParamConfigFieldProvenance> config_fields;  ///< Ordered configuration inputs.
};

}  // namespace holoscan
