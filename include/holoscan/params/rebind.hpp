/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/holoscan_params_export.h>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/parameter_override_set.hpp>
#include <holoscan/core/parameter_snapshot.hpp>
#include <holoscan/core/profile.hpp>
#include <holoscan/params/error.hpp>

namespace holoscan::params {

namespace detail {
struct ParamAdmissionContextState;
struct ParamRebindTargetState;
}  // namespace detail

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
/// @note baseline rejects a nonempty chain unless the installed admission context can resolve every
/// evidence ID to an immutable qualified migration implementation.
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
  /// Borrowed directed Operator aliases consumed synchronously by to_overrides().
  std::span<const ParamOperatorAlias> operator_aliases{};
  /// Borrowed directed field aliases consumed synchronously by to_overrides().
  std::span<const ParamFieldAlias> field_aliases{};
  /// Borrowed ordered migration chain consumed synchronously by to_overrides().
  std::span<const ParamMigrationStep> migration_chain{};
};

/// @brief Own immutable profile and package policy used while capturing a target.
///
/// The context is cheap to copy and owns its policy state. Concurrent const access
/// is safe. The baseline exposes only standard(); future named admission contexts require
/// separate qualified package evidence.
class HOLOSCAN_PARAMS_EXPORT ParamAdmissionContext {
 public:
  /// @brief Return the standard local authoring context.
  /// @return Owning standard-profile context, or a typed failure if unavailable.
  /// @note This cold operation may allocate and does not mutate a Graph.
  [[nodiscard]] static ParamRebindResult<ParamAdmissionContext> standard();

  /// @brief Return the requested profile.
  [[nodiscard]] Profile profile() const noexcept;

 private:
  explicit ParamAdmissionContext(std::shared_ptr<const detail::ParamAdmissionContextState> state)
      : state_(std::move(state)) {}
  std::shared_ptr<const detail::ParamAdmissionContextState> state_;
  friend ParamRebindResult<class ParamRebindTarget> make_param_rebind_target(
      Graph&, const ParamAdmissionContext&, const ParamRebindTargetScope&);
};

/// @brief Own one compose-once immutable target Graph snapshot and local factory inventory.
///
/// The target is process-local, cheap to copy, and retains the materialization
/// operations required to construct typed override aggregates. It owns no Graph
/// reference or runtime authority. Concurrent const use is safe after capture.
class HOLOSCAN_PARAMS_EXPORT ParamRebindTarget {
 public:
  /// @brief Copy one immutable shared target view.
  ParamRebindTarget(const ParamRebindTarget&) noexcept = default;
  /// @brief Move one immutable shared target view.
  ParamRebindTarget(ParamRebindTarget&&) noexcept = default;
  /// @brief Replace this view with a copy of another immutable target.
  ParamRebindTarget& operator=(const ParamRebindTarget&) noexcept = default;
  /// @brief Replace this view by moving another immutable target.
  ParamRebindTarget& operator=(ParamRebindTarget&&) noexcept = default;
  /// @brief Release this view and its process-local target state when last-owned.
  ~ParamRebindTarget();

  /// @brief Return the exact captured Graph authoring identity.
  /// @note Safe for concurrent reads.
  [[nodiscard]] GraphAuthoringIdentity identity() const noexcept;

  /// @brief Return the exact destination authority category captured by this target.
  [[nodiscard]] ParamRebindTargetScopeKind scope_kind() const noexcept;

  /// @brief Return canonical exact paths defining this target scope.
  /// @return Empty for whole-Graph scope, one path for Graph/Operator scope, or a sorted set.
  /// @note The view remains valid while this target owns its current state.
  [[nodiscard]] std::span<const std::string> scope_paths() const noexcept;

  /// @brief Return the domain-separated identity of the selected target schemas.
  /// @return Digest over scope kind, canonical scope paths, and exact selected schemas.
  [[nodiscard]] ParamSchemaHash selection_digest() const noexcept;

 private:
  explicit ParamRebindTarget(std::shared_ptr<const detail::ParamRebindTargetState> state)
      : state_(std::move(state)) {}
  std::shared_ptr<const detail::ParamRebindTargetState> state_;
  friend ParamRebindResult<ParamRebindTarget> make_param_rebind_target(
      Graph&, const ParamAdmissionContext&, const ParamRebindTargetScope&);
  friend ParamRebindResult<ParamOverrideSet> to_overrides(const ParamSnapshot&,
                                                          const ParamRebindTarget&,
                                                          ParamRebindOptions);
};

/// @brief Capture one immutable target Graph under an explicit admission context.
/// @param graph Draft Graph to inspect synchronously; no reference is retained.
/// @param context Immutable admission policy to apply.
/// @param scope Owning exact destination authority; defaults to strict whole-Graph capture.
/// @return Owning immutable target or a typed capture/admission failure.
/// @note This cold operation may allocate. Do not mutate graph concurrently.
[[nodiscard]] HOLOSCAN_PARAMS_EXPORT ParamRebindResult<ParamRebindTarget> make_param_rebind_target(
    Graph& graph, const ParamAdmissionContext& context,
    const ParamRebindTargetScope& scope = ParamRebindTargetScope::whole_graph());

/// @brief Convert one owning snapshot into target-bound scalar override contributions.
/// @param snapshot Owning source snapshot; values are copied into the result.
/// @param target Immutable target captured from the destination Graph.
/// @param options Coverage mode and borrowed directed conversion tables.
/// @return Complete immutable override set or one typed all-or-nothing failure.
/// @note This cold operation may allocate and performs no Graph mutation. The baseline executes
/// exact direct and alias resolution. A nonempty migration chain fails closed until qualified
/// package migration implementations are installed; no heuristic name conversion is performed.
[[nodiscard]] HOLOSCAN_PARAMS_EXPORT ParamRebindResult<ParamOverrideSet> to_overrides(
    const ParamSnapshot& snapshot, const ParamRebindTarget& target,
    ParamRebindOptions options = {});

/// @brief Convert one snapshot using the standard local target context.
/// @param snapshot Owning source snapshot; values are copied into the result.
/// @param graph Draft Graph captured synchronously under the standard context.
/// @param options Coverage mode and borrowed directed conversion tables.
/// @return Complete immutable override set or one typed capture/conversion failure.
/// @note This cold convenience may allocate and does not mutate graph. Do not mutate graph
/// concurrently.
[[nodiscard]] HOLOSCAN_PARAMS_EXPORT ParamRebindResult<ParamOverrideSet> to_overrides(
    const ParamSnapshot& snapshot, Graph& graph, ParamRebindOptions options = {});

/// @brief Convert one snapshot into an explicitly scoped standard local target.
/// @param snapshot Owning source snapshot; every source record must map into target_scope.
/// @param graph Draft Graph captured synchronously under the standard context.
/// @param target_scope Owning exact destination authority, independent of snapshot scope.
/// @param options Coverage mode and borrowed directed conversion tables.
/// @return Complete immutable override set or one typed capture/conversion failure.
/// @note This cold convenience may allocate and does not mutate graph. Sparse mode may omit
/// selected destination Operators, but neither sparse nor complete mode ignores an out-of-scope
/// source record. Do not mutate graph concurrently.
[[nodiscard]] HOLOSCAN_PARAMS_EXPORT ParamRebindResult<ParamOverrideSet> to_overrides(
    const ParamSnapshot& snapshot, Graph& graph, const ParamRebindTargetScope& target_scope,
    ParamRebindOptions options = {});

}  // namespace holoscan::params
