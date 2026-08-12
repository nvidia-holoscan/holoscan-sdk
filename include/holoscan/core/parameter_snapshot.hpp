/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <holoscan/holoscan_core_export.h>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_plan.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/parameter_catalog.hpp>
#include <holoscan/core/parameter_schema.hpp>
#include <holoscan/core/parameter_update.hpp>

namespace holoscan {

namespace detail {
class ParameterUpdateCoordinator;
class ParamSnapshotArchiveAccess;
}  // namespace detail

/// @brief Select the consistency boundary for one owning Parameter capture.
enum class ParamCaptureConsistency : std::uint8_t {
  kRequireStopped = 0,     ///< Require terminal callbacks and a global published frontier.
  kLivePerOperator = 1,    ///< Capture each Operator atomically without a graph-wide instant.
  kDispatcherBarrier = 2,  ///< Reserved for a later admitted bounded dispatcher barrier.
};

/// @brief Select the ordinary policy projection used by capture.
enum class ParamSnapshotFilter : std::uint8_t {
  kPolicyDefault = 0,  ///< Apply transient, sensitive, internal, profile, and codec policy.
};

/// @brief Select handling for declared fields without a supported value projection.
enum class ParamUnavailableValuePolicy : std::uint8_t {
  kReject = 0,             ///< Fail the complete capture before returning partial data.
  kRecordUnavailable = 1,  ///< Retain metadata with an explicit unavailable projection.
};

/// @brief Classify the value completeness of one captured Operator.
enum class ParamSnapshotValueCoverage : std::uint8_t {
  kUnavailable = 0,        ///< No declared value was projected.
  kPolicyFiltered = 1,     ///< At least one declared value was filtered or unavailable.
  kAllDeclaredFields = 2,  ///< Every declared field value was projected.
};

/// @brief Own one stable versioned field-value type identity.
struct ParamSnapshotTypeIdentity {
  std::string id;               ///< Stable nonempty type ID when the shape is declared.
  std::uint32_t version{1U};    ///< Positive identity-schema version.
  ParamFieldTypeShape shape{};  ///< Frozen logical family, widths, and duration facts.
};

/// @brief Own one admitted versioned codec identity.
struct ParamSnapshotCodecIdentity {
  std::string id;             ///< Stable nonempty codec ID.
  std::uint32_t version{1U};  ///< Positive codec-contract version.
};

/// @brief Configure one owning Parameter capture.
struct ParamCaptureOptions {
  ParamReadView view{ParamReadView::kPublished};  ///< Value view; EA1 admits published only.
  ParamCaptureConsistency consistency{
      ParamCaptureConsistency::kRequireStopped};  ///< Requested coherence boundary.
  std::optional<ParamUpdateSequence>
      after_update_sequence;  ///< Optional stopped-capture terminal frontier requirement.
  ParamSnapshotFilter filter{ParamSnapshotFilter::kPolicyDefault};  ///< Projection policy.
  ParamUnavailableValuePolicy unavailable_values{
      ParamUnavailableValuePolicy::kReject};  ///< Strict versus metadata-only unavailable values.
};

/// @brief Own one projected field and its explicit policy disposition.
struct ParamSnapshotFieldRecord {
  ParamAddress address{};     ///< Exact source-run identity retained as provenance only.
  ParamSchemaHash full_id{};  ///< Complete stable field identity.
  ParamPath path{};           ///< Canonical decoded field path.
  ParamLogicalType logical_type{ParamLogicalType::kObject};  ///< Frozen logical value family.
  ParamSnapshotTypeIdentity type{};  ///< Stable type identity copied from the frozen plan.
  std::optional<ParamSnapshotCodecIdentity>
      codec;  ///< Admitted codec, or empty with an unavailable projection.
  params::ParamProjectionStatus projection{
      params::ParamProjectionStatus::kExcludedUnsupportedCodec};  ///< First-match policy result.
  std::optional<ParamValue>
      value;  ///< Present only for \ref params::ParamProjectionStatus::kIncluded.
};

/// @brief Own one Operator's atomic capture record.
struct ParamSnapshotOperatorRecord {
  OperatorRef operator_instance{};  ///< Exact source-run identity retained as provenance only.
  OperatorPath path{};              ///< Canonical decoded Operator path.
  std::optional<Profile> effective_profile;  ///< Compile-resolved policy profile.
  ParamSchemaAvailability schema_availability{
      ParamSchemaAvailability::kOpaqueLocal};  ///< Frozen generic-schema availability.
  std::string schema_id;                       ///< Stable declared schema ID, or empty when opaque.
  ParamSchemaVersion schema_version{};         ///< Declared schema version, or zero when opaque.
  ParamSchemaHash structural_hash{};           ///< Frozen schema decoding-shape identity.
  ParamSchemaHash contract_hash{};             ///< Frozen schema policy identity.
  ParamSchemaHash metadata_hash{};             ///< Frozen descriptive schema identity.
  ParamSchemaCoverage schema_coverage{
      ParamSchemaCoverage::kDeclaredFieldsOnly};  ///< Declared aggregate coverage.
  ParamSchemaProof schema_proof{ParamSchemaProof::kAuthorAsserted};  ///< Coverage evidence.
  ParamGeneration generation{};                     ///< Atomically captured published generation.
  ParamUpdateSequence accepted_update_sequence{};   ///< Latest accepted intent for this Operator.
  ParamUpdateSequence published_update_sequence{};  ///< Sequence that produced \ref generation.
  ParamUpdateState state{ParamUpdateState::kPublished};  ///< Published or observed lineage state.
  ParamSnapshotFilter filter{ParamSnapshotFilter::kPolicyDefault};  ///< Applied projection filter.
  ParamSnapshotValueCoverage value_coverage{
      ParamSnapshotValueCoverage::kUnavailable};  ///< Aggregate projected-value completeness.
  std::vector<ParamSnapshotFieldRecord> fields;   ///< Fields ordered by stable field identity.
};

/// @brief Classify the owning scope of a Parameter snapshot.
enum class ParamSnapshotScopeKind : std::uint8_t {
  kRoot = 0,      ///< Complete run capture.
  kGraph = 1,     ///< One reusable Graph instance path.
  kOperator = 2,  ///< One exact Operator path.
};

/// @brief Own one structured Parameter snapshot record-container scope.
///
/// The scope describes the outer set of Operator records owned by a snapshot;
/// it is not a copy of the exact authority that produced the snapshot. A
/// field-restricted capture therefore owns one Operator-scoped envelope whose
/// field vector contains only the authorized field. Snapshots grant no control
/// authority, and slicing can only remove records or fields already present.
class ParamSnapshotScope {
 public:
  /// @brief Construct the complete-run root scope.
  /// @note Construction does not allocate or bind the scope to a live run.
  ParamSnapshotScope() = default;

  /// @brief Return complete-run root scope.
  /// @return Root scope that selects every policy-visible Operator in a run.
  /// @note The call does not allocate or block.
  [[nodiscard]] static ParamSnapshotScope root() noexcept { return {}; }

  /// @brief Return one embedded Graph-instance scope.
  /// @param path Canonical decoded embedded Graph path.
  /// @return Owning scope that selects every Operator below \p path.
  /// @throws std::bad_alloc if copying the path segments into the owning scope fails.
  [[nodiscard]] static ParamSnapshotScope graph(const GraphInstancePath& path) {
    return ParamSnapshotScope{
        ParamSnapshotScopeKind::kGraph,
        OperatorPath{std::vector<std::string>{path.segments().begin(), path.segments().end()}}};
  }

  /// @brief Return one Operator scope.
  /// @param path Canonical decoded Operator path to own in the returned scope.
  /// @return Owning scope that selects exactly \p path.
  /// @note A default-constructed empty path produces a deferred-invalid scope
  /// that \ref ParamSnapshot::slice rejects with \ref ErrorCode::kInvalidArgument.
  /// @note Moving an existing path into this call does not allocate. Passing an lvalue may allocate
  /// while copying the by-value argument before the function is entered.
  /// @throws std::bad_alloc if construction of the by-value \p path argument fails.
  [[nodiscard]] static ParamSnapshotScope operator_(OperatorPath path) {
    return ParamSnapshotScope{ParamSnapshotScopeKind::kOperator, std::move(path)};
  }

  /// @brief Return the scope kind.
  /// @return Exact tagged interpretation of \ref path.
  /// @note The call does not allocate or block.
  [[nodiscard]] ParamSnapshotScopeKind kind() const noexcept { return kind_; }

  /// @brief Return the decoded scope path for non-root scope.
  /// @return Borrowed path owned by this scope; empty for \ref ParamSnapshotScopeKind::kRoot.
  /// @note The reference remains valid until this scope is assigned or destroyed.
  [[nodiscard]] const OperatorPath& path() const noexcept { return path_; }

 private:
  ParamSnapshotScope(ParamSnapshotScopeKind kind, OperatorPath path)
      : kind_(kind), path_(std::move(path)) {}
  ParamSnapshotScopeKind kind_{ParamSnapshotScopeKind::kRoot};
  OperatorPath path_{};
  friend class detail::ParamSnapshotArchiveAccess;
};

/// @brief Own a consistency-labeled, value-safe Parameter capture envelope.
///
/// Snapshot storage is independent of live generation slots and remains valid
/// after the originating session retires. EA1 never copies opaque C++ object
/// representation into this envelope.
/// @note Distinct snapshot objects may be used concurrently. Concurrent const access to one
/// snapshot is safe, but assigning, moving, or destroying that object requires external exclusion
/// from every access to the same object.
class HOLOSCAN_CORE_EXPORT ParamSnapshot {
 public:
  /// @brief Construct an empty root-scoped snapshot.
  /// @note The snapshot has zero source identities and no Operator records until populated by an
  /// admitted capture operation.
  ParamSnapshot() = default;

  /// @brief Return the semantic source-plan digest.
  /// @return Exact digest of the source plan, or all zeros for an empty snapshot.
  [[nodiscard]] PlanDigest plan_digest() const noexcept { return plan_digest_; }

  /// @brief Return the source plan's process-local generation identity.
  /// @return Nonzero source generation for a captured snapshot, or zero when empty.
  [[nodiscard]] PlanGenerationId plan_generation() const noexcept { return plan_generation_; }

  /// @brief Return the source run generation.
  /// @return Nonzero source run generation for a captured snapshot, or zero when empty.
  [[nodiscard]] RunGenerationId run_generation() const noexcept { return run_generation_; }

  /// @brief Return the source catalog digest.
  /// @return Exact plan/run-bound digest of the source capture policy's visible catalog projection,
  /// or all zeros for an empty snapshot.
  /// @note A capture-only capability does not expose discovery metadata, so this digest
  /// intentionally differs from that handle's empty \ref ParamControlHandle::catalog projection
  /// while remaining equal to the same policy projection of a read-capable handle.
  /// @note Offline slicing preserves this source capability identity rather than claiming a new
  /// catalog projection for the subset of already-owned snapshot records.
  [[nodiscard]] ParamCatalogDigest catalog_digest() const noexcept { return catalog_digest_; }

  /// @brief Return the captured immutable value view.
  /// @return View selected by the capture operation.
  [[nodiscard]] ParamReadView view() const noexcept { return view_; }

  /// @brief Return the capture consistency mode.
  /// @return Consistency boundary established by the capture operation.
  [[nodiscard]] ParamCaptureConsistency consistency() const noexcept { return consistency_; }

  /// @brief Return the owning structured capture or slice scope.
  /// @return Borrowed scope retained by this snapshot.
  /// @note The reference remains valid until this snapshot is assigned or destroyed.
  [[nodiscard]] const ParamSnapshotScope& scope() const noexcept { return scope_; }

  /// @brief Return the global accepted frontier for a stopped capture.
  /// @return Accepted terminal sequence for a stopped capture, or empty for live-per-Operator
  /// capture.
  [[nodiscard]] std::optional<ParamUpdateSequence> accepted_through() const noexcept {
    return accepted_through_;
  }

  /// @brief Return the global published frontier for a stopped capture.
  /// @return Highest contiguous accepted sequence whose intent is reflected by the published view,
  /// either through publication or an equality-proven accepted-no-change result; empty for
  /// live-per-Operator capture.
  /// @note Superseded, canceled, rolled-back, and failed-stopped sequences create permanent holes.
  /// A lower value than \ref accepted_through records one of those terminal outcomes after this
  /// frontier; it does not imply pending publication. Accepted-no-change remains a distinct state
  /// and is never relabeled as published.
  [[nodiscard]] std::optional<ParamUpdateSequence> published_through() const noexcept {
    return published_through_;
  }

  /// @brief Return owning per-Operator records.
  /// @return Borrowed canonical records owned by this snapshot.
  /// @note The span remains valid until this snapshot is assigned or destroyed. Concurrent reads
  /// are safe while no thread mutates the same snapshot object.
  [[nodiscard]] std::span<const ParamSnapshotOperatorRecord> operators() const noexcept {
    return operators_;
  }

  /// @brief Return an owning snapshot restricted to one structured scope.
  /// @param scope Root, embedded-Graph, or Operator scope to select.
  /// @return Independent owning snapshot with preserved source and consistency metadata.
  /// @retval ErrorCode::kPermissionDenied The requested scope is wider than, or unrelated to, this
  /// snapshot's owning scope.
  /// @retval ErrorCode::kInvalidArgument A non-root scope has an empty path or the scope kind is
  /// invalid.
  /// @retval ErrorCode::kNotFound The requested non-root scope selects no captured Operator.
  /// @retval ErrorCode::kResourceExhausted Owning slice storage could not be allocated.
  /// @note Slicing reads only this snapshot, performs no live-runtime lookup, and grants no update
  /// authority. A slice can retain or narrow the source record-container scope and can never add
  /// an Operator or field omitted by the originating capture policy.
  [[nodiscard]] expected<ParamSnapshot, Error> slice(const ParamSnapshotScope& scope) const;

  /// @brief Return an owning snapshot restricted to one Operator path.
  /// @param path Exact decoded Operator path to select.
  /// @return Independent owning snapshot for \p path.
  /// @retval ErrorCode::kPermissionDenied The path is outside this snapshot's owning scope.
  /// @retval ErrorCode::kInvalidArgument The decoded path is empty.
  /// @retval ErrorCode::kNotFound No captured Operator has the exact path.
  /// @retval ErrorCode::kResourceExhausted Owning slice storage could not be allocated.
  [[nodiscard]] expected<ParamSnapshot, Error> slice(const OperatorPath& path) const;

  /// @brief Parse and immediately copy one Operator path before slicing.
  /// @param operator_path Canonical encoded Operator path.
  /// @return Independent owning snapshot for the parsed exact path.
  /// @retval ErrorCode::kInvalidArgument The path is not canonical or contains an invalid segment.
  /// @retval ErrorCode::kPermissionDenied The parsed path is outside this snapshot's owning scope.
  /// @retval ErrorCode::kNotFound No captured Operator has the parsed path.
  /// @retval ErrorCode::kResourceExhausted Path or slice storage could not be allocated.
  [[nodiscard]] expected<ParamSnapshot, Error> slice(std::string_view operator_path) const;

 private:
  PlanDigest plan_digest_{};
  PlanGenerationId plan_generation_{};
  RunGenerationId run_generation_{};
  ParamCatalogDigest catalog_digest_{};
  ParamReadView view_{ParamReadView::kPublished};
  ParamCaptureConsistency consistency_{ParamCaptureConsistency::kRequireStopped};
  ParamSnapshotScope scope_{};
  std::optional<ParamUpdateSequence> accepted_through_;
  std::optional<ParamUpdateSequence> published_through_;
  std::vector<ParamSnapshotOperatorRecord> operators_;
  bool has_archive_identity_{};
  std::uint32_t archive_semantic_version_{};
  std::array<std::byte, 32> archive_semantic_sha256_{};
  std::uint32_t archive_raw_source_version_{};
  std::array<std::byte, 32> archive_raw_source_sha256_{};
  std::uint64_t archive_raw_source_bytes_{};
  std::optional<Profile> archive_profile_;
  std::optional<std::string> archive_source_run_identity_;

  friend class detail::ParameterUpdateCoordinator;
  friend class detail::ParamSnapshotArchiveAccess;
};

}  // namespace holoscan
