/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/holoscan_params_export.h>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/parameter_provenance.hpp>
#include <holoscan/core/parameter_snapshot.hpp>
#include <holoscan/params/error.hpp>

namespace holoscan::params {

using ::holoscan::ParamFieldRebindDisposition;
using ::holoscan::ParamFieldRebindOrigin;
using ::holoscan::ParamRebindMode;
using ::holoscan::ParamRebindProvenance;
using ::holoscan::ParamRebindTargetScopeKind;

/// @brief Classify one exact rebinding rejection.
enum class ParamRebindErrorCode : std::uint8_t {
  kInvalidSnapshot = 0,         ///< Source snapshot structure or provenance is invalid.
  kTargetIdentityMismatch = 1,  ///< Source evidence does not identify the captured target.
  kScopeNotFound = 2,           ///< Requested snapshot scope is absent from the target.
  kProfileIncompatible = 3,     ///< Source and target profiles cannot be combined.
  kSchemaUnavailable = 4,       ///< Required source or target schema evidence is unavailable.
  kSchemaIncompatible = 5,      ///< Schema identities disagree without an admitted conversion.
  kMigrationRequired = 6,       ///< Conversion requires a migration the caller did not provide.
  kFieldIncompatible = 7,       ///< A field fails a non-type compatibility rule.
  kInvalidTarget = 8,           ///< Captured target structure is invalid or unsupported.
  kTargetChanged = 9,           ///< Graph authoring identity changed after target capture.
  kCoverageInsufficient = 10,   ///< Complete-object mode lacks required source coverage.
  kUnknownOperator = 11,        ///< A source Operator path has no exact target.
  kUnknownField = 12,           ///< A source field identity has no exact target.
  kAliasConflict = 13,          ///< Alias and direct resolution produce ambiguous targets.
  kMigrationChainInvalid = 14,  ///< Directed migration steps are malformed or exceed limits.
  kMigrationFailed = 15,        ///< An admitted migration implementation failed transactionally.
  kTypeMismatch = 16,           ///< Scalar logical or exact storage type is incompatible.
  kBoundsViolation = 17,        ///< Converted scalar violates target bounds.
  kProfileDenied = 18,          ///< Admission policy denies the requested target profile.
  kRequiredFieldMissing = 19,   ///< A required complete-object field is absent.
  kRestrictedField = 20,        ///< Source projection cannot authorize restoration of the field.
  kStaleOverrideSet = 21,       ///< Override provenance no longer matches compile input.
};

/// @brief Identify the exact cold-path stage that rejected rebinding.
enum class ParamRebindStage : std::uint8_t {
  kSourceValidation = 0,      ///< Validate source snapshot structure and frozen evidence.
  kTargetCapture = 1,         ///< Capture or validate immutable target evidence.
  kMigration = 2,             ///< Validate and execute the directed migration chain.
  kOperatorAlias = 3,         ///< Resolve exact Operator path aliases.
  kFieldAlias = 4,            ///< Resolve exact schema-field aliases.
  kFinalResolution = 5,       ///< Resolve the unique target Operator and field.
  kSparseConstruction = 6,    ///< Construct one sparse contribution aggregate.
  kCompleteConstruction = 7,  ///< Construct one complete target aggregate.
  kResultValidation = 8,      ///< Validate the immutable override result.
  kCompileRevalidation = 9,   ///< Revalidate target binding during Graph compilation.
};

/// @brief Own one typed snapshot-to-Graph conversion failure.
///
/// The value owns bounded path and schema context plus structured tooling detail.
/// Context that exceeds a declared capacity is truncated and reported by
/// context_truncated(). The object is safe to read concurrently while no thread
/// mutates or destroys it.
class HOLOSCAN_PARAMS_EXPORT ParamRebindError {
 public:
  static constexpr std::size_t kPathCapacity = 512U;      ///< Maximum bytes per retained path.
  static constexpr std::size_t kSchemaIdCapacity = 256U;  ///< Maximum bytes per schema ID.

  /// @brief Construct one bounded rebinding failure.
  /// @param code Stable failure category.
  /// @param stage Exact cold-path stage that rejected the operation.
  /// @param detail Owning tooling failure detail and optional nested cause.
  /// @param source_path Source Operator or field path, when known.
  /// @param target_path Target Operator or field path, when known.
  /// @param source_schema_id Source schema identity, when known.
  /// @param target_schema_id Target schema identity, when known.
  /// @param source_field Source field identity, or zero when unavailable.
  /// @param target_field Target field identity, or zero when unavailable.
  /// @param migration_evidence Qualified migration evidence ID, when applicable.
  /// @note Construction is non-throwing; oversized textual context is truncated.
  ParamRebindError(ParamRebindErrorCode code, ParamRebindStage stage, ParamToolError detail,
                   std::string_view source_path = {}, std::string_view target_path = {},
                   std::string_view source_schema_id = {}, std::string_view target_schema_id = {},
                   ParamFieldId source_field = {}, ParamFieldId target_field = {},
                   std::optional<ParamSchemaHash> migration_evidence = std::nullopt) noexcept;
  /// @brief Return the stable failure category.
  [[nodiscard]] ParamRebindErrorCode code() const noexcept { return code_; }
  /// @brief Return the exact stage that rejected the operation.
  [[nodiscard]] ParamRebindStage stage() const noexcept { return stage_; }
  /// @brief Return the retained source path.
  /// @note The view remains valid until this error is assigned or destroyed.
  [[nodiscard]] std::string_view source_path() const noexcept {
    return {source_path_.data(), source_path_size_};
  }
  /// @brief Return the retained target path.
  /// @note The view remains valid until this error is assigned or destroyed.
  [[nodiscard]] std::string_view target_path() const noexcept {
    return {target_path_.data(), target_path_size_};
  }
  /// @brief Return the retained source schema ID.
  /// @note The view remains valid until this error is assigned or destroyed.
  [[nodiscard]] std::string_view source_schema_id() const noexcept {
    return {source_schema_id_.data(), source_schema_id_size_};
  }
  /// @brief Return the retained target schema ID.
  /// @note The view remains valid until this error is assigned or destroyed.
  [[nodiscard]] std::string_view target_schema_id() const noexcept {
    return {target_schema_id_.data(), target_schema_id_size_};
  }
  /// @brief Return the source field ID, or zero when unavailable.
  [[nodiscard]] ParamFieldId source_field() const noexcept { return source_field_; }
  /// @brief Return the target field ID, or zero when unavailable.
  [[nodiscard]] ParamFieldId target_field() const noexcept { return target_field_; }
  /// @brief Return qualified migration evidence associated with the failure.
  /// @return Evidence hash, or empty when migration was not involved.
  [[nodiscard]] const std::optional<ParamSchemaHash>& migration_evidence() const noexcept {
    return migration_evidence_;
  }
  /// @brief Return whether any textual diagnostic context was truncated.
  [[nodiscard]] bool context_truncated() const noexcept { return context_truncated_; }
  /// @brief Return structured tooling detail retained by this error.
  /// @note The reference remains valid until this error is assigned or destroyed.
  [[nodiscard]] const ParamToolError& detail() const noexcept { return detail_; }

 private:
  ParamRebindErrorCode code_{ParamRebindErrorCode::kInvalidSnapshot};
  ParamRebindStage stage_{ParamRebindStage::kSourceValidation};
  std::array<char, kPathCapacity> source_path_{};
  std::array<char, kPathCapacity> target_path_{};
  std::array<char, kSchemaIdCapacity> source_schema_id_{};
  std::array<char, kSchemaIdCapacity> target_schema_id_{};
  std::uint16_t source_path_size_{};
  std::uint16_t target_path_size_{};
  std::uint16_t source_schema_id_size_{};
  std::uint16_t target_schema_id_size_{};
  ParamFieldId source_field_{};
  ParamFieldId target_field_{};
  std::optional<ParamSchemaHash> migration_evidence_;
  bool context_truncated_{};
  ParamToolError detail_;
};

/// @brief Result type used by snapshot rebinding operations.
/// @tparam T Successful value type.
template <typename T>
using ParamRebindResult = expected<T, ParamRebindError>;

/// @brief Carry one exact directed Operator path alias.
struct ParamOperatorAlias {
  OperatorPath from_path;               ///< Exact source Operator path.
  std::uint32_t from_schema_version{};  ///< Exact nonzero source schema version.
  OperatorPath to_path;                 ///< Exact target Operator path.
  std::uint32_t to_schema_version{};    ///< Exact nonzero target schema version.
};

/// @brief Carry one exact directed schema-field alias.
struct ParamFieldAlias {
  std::string from_schema_id;           ///< Owning exact source schema ID.
  std::uint32_t from_schema_version{};  ///< Exact nonzero source schema version.
  ParamFieldId from_field_id{};         ///< Exact nonzero source field ID.
  std::string to_schema_id;             ///< Owning exact target schema ID.
  std::uint32_t to_schema_version{};    ///< Exact nonzero target schema version.
  ParamFieldId to_field_id{};           ///< Exact nonzero target field ID.
};

/// @brief Identify one exact schema endpoint in a qualified migration chain.
struct ParamSchemaIdentity {
  std::string id;           ///< Owning stable schema ID.
  std::uint32_t version{};  ///< Exact nonzero schema version.
};

/// @brief Carry one qualified directed schema migration request.
/// @note The current implementation rejects nonempty migration chains because no migration
/// implementation catalog is installed yet.
struct ParamMigrationStep {
  ParamSchemaHash evidence_id{};             ///< Qualified immutable migration evidence ID.
  ParamSchemaIdentity from_schema_identity;  ///< Exact directed source endpoint.
  ParamSchemaIdentity to_schema_identity;    ///< Exact directed target endpoint.
};

/// @brief Own the exact destination Operator authority for one cold rebind target.
///
/// A default-constructed value is the strict whole-Graph scope. Graph and Operator
/// paths are decoded owning values; no member borrows a Graph, snapshot, or caller
/// span. Capture canonicalizes an Operator set by rendered exact path and rejects an
/// empty, duplicate, unknown, or over-limit set rather than inferring authority from
/// source archive contents.
class ParamRebindTargetScope {
 public:
  /// @brief Construct the strict whole-Graph destination scope.
  /// @note Construction does not allocate or bind to a live Graph.
  ParamRebindTargetScope() = default;

  /// @brief Return the strict whole-Graph destination scope.
  /// @return Scope requiring every destination Operator to expose an admitted schema.
  [[nodiscard]] static ParamRebindTargetScope whole_graph() noexcept { return {}; }

  /// @brief Return one exact embedded-Graph destination scope.
  /// @param path Canonical decoded embedded Graph instance path.
  /// @return Owning scope selecting every Operator below the supplied path.
  /// @throws std::bad_alloc if copying the decoded path fails.
  [[nodiscard]] static ParamRebindTargetScope graph(const GraphInstancePath& path) {
    std::vector<std::vector<std::string>> paths;
    paths.emplace_back(path.segments().begin(), path.segments().end());
    return ParamRebindTargetScope{ParamRebindTargetScopeKind::kGraph, std::move(paths)};
  }

  /// @brief Return one exact Operator destination scope.
  /// @param path Owning canonical decoded Operator path.
  /// @return Scope selecting exactly the supplied path.
  /// @note An empty path is retained as deferred-invalid input that target capture rejects.
  [[nodiscard]] static ParamRebindTargetScope operator_(const OperatorPath& path) {
    std::vector<std::vector<std::string>> paths;
    paths.emplace_back(path.segments().begin(), path.segments().end());
    return ParamRebindTargetScope{ParamRebindTargetScopeKind::kOperator, std::move(paths)};
  }

  /// @brief Return one explicit exact Operator-set destination scope.
  /// @param paths Owning decoded Operator paths; caller order has no semantic meaning.
  /// @return Scope whose paths are canonicalized and validated during target capture.
  /// @note Empty paths, an empty set, duplicates, and the finite capture limit fail closed.
  [[nodiscard]] static ParamRebindTargetScope operator_set(const std::vector<OperatorPath>& paths) {
    std::vector<std::vector<std::string>> decoded;
    decoded.reserve(paths.size());
    for (const auto& path : paths) {
      decoded.emplace_back(path.segments().begin(), path.segments().end());
    }
    return ParamRebindTargetScope{ParamRebindTargetScopeKind::kOperatorSet, std::move(decoded)};
  }

  /// @brief Return the exact destination scope category.
  [[nodiscard]] ParamRebindTargetScopeKind kind() const noexcept { return kind_; }

  /// @brief Return owning decoded paths supplied for this scope.
  /// @return Empty for whole-Graph scope, one path for Graph/Operator scope, or the caller set.
  /// @note The view remains valid until this scope is assigned or destroyed.
  [[nodiscard]] std::span<const std::vector<std::string>> paths() const noexcept { return paths_; }

 private:
  ParamRebindTargetScope(ParamRebindTargetScopeKind kind,
                         std::vector<std::vector<std::string>> paths)
      : kind_(kind), paths_(std::move(paths)) {}

  ParamRebindTargetScopeKind kind_{ParamRebindTargetScopeKind::kWholeGraph};
  std::vector<std::vector<std::string>> paths_;
};

/// @brief Configure one cold snapshot-to-Graph conversion.
struct ParamRebindOptions {
  /// Coverage construction mode.
  ParamRebindMode mode{ParamRebindMode::kSparseContributions};
  /// Borrowed directed Operator aliases consumed synchronously by set_params().
  std::span<const ParamOperatorAlias> operator_aliases{};
  /// Borrowed directed field aliases consumed synchronously by set_params().
  std::span<const ParamFieldAlias> field_aliases{};
  /// Borrowed ordered migration chain consumed synchronously by set_params().
  std::span<const ParamMigrationStep> migration_chain{};
};

/// @brief Stage one snapshot as an all-or-nothing graph-owned parameter input.
/// @param graph Destination Graph that owns the pending transaction.
/// @param snapshot Owning reusable source snapshot.
/// @param target_scope Exact destination authority, resolved without capturing the Graph.
/// @param options Coverage mode and borrowed directed conversion tables.
/// @return Owning conversion provenance or one typed attachment failure.
/// @note Compilation resolves final paths, profiles, schemas, and parameter baselines. This call
/// does not capture a Graph snapshot or retain the destination Graph in the parameter input.
/// @note This call mutates Graph authoring state and must not run concurrently with Graph mutation.
/// Configuration and snapshot attachment share one transaction slot, including across embedded
/// Graphs. A second transaction fails instead of replacing the first.
/// @note Attachment is not a non-mutating compatibility preflight. Compile a disposable Graph when
/// validation must not alter the intended destination.
[[nodiscard]] HOLOSCAN_PARAMS_EXPORT ParamRebindResult<ParamRebindProvenance> set_params(
    Graph& graph, const ParamSnapshot& snapshot,
    const ParamRebindTargetScope& target_scope = ParamRebindTargetScope::whole_graph(),
    ParamRebindOptions options = {});

}  // namespace holoscan::params
