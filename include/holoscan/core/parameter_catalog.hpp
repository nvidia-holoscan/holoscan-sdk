/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <holoscan/holoscan_core_export.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/operator_ref.hpp>
#include <holoscan/core/parameter_schema.hpp>
#include <holoscan/core/profile.hpp>

namespace holoscan {

class ParamSnapshotScope;

namespace detail {
class ParamControlAccess;
class ParameterCatalogAccess;
class ParameterUpdateCoordinator;
}  // namespace detail

/// @brief Identify one field on one Operator within a compiled plan.
///
/// The value is the collision-checked 64-bit projection of a versioned digest over the structured
/// Operator path and the field's complete schema identity. It is meaningful only with the
/// enclosing catalog compatibility digest and is never a dense table index.
using ParamInstanceId = std::uint64_t;

/// @brief Address one Parameter field with its full plan/run-bound Operator identity.
///
/// The value owns no storage and is meaningful only against a catalog whose plan and run
/// generations match \ref operator_instance. It is safe to copy between threads.
struct ParamAddress {
  OperatorRef operator_instance{};  ///< Exact plan/run-bound Operator identity.
  ParamFieldId field{};             ///< Schema-lineage-local field identity.
  /// @brief Compare complete plan/run-bound field addresses.
  /// @return \c true when both Operator identities and field identities match exactly.
  friend constexpr bool operator==(ParamAddress, ParamAddress) noexcept = default;
};

/// @brief Classify one field's compile-time active-update disposition.
///
/// The disposition is total even when generation-zero execution remains admitted. Only
/// \ref kAdmitted authorizes later Runtime materialization.
enum class ParameterFieldUpdateDisposition : std::uint8_t {
  kAdmitted = 0,                ///< Field passed every active-update admission rule.
  kPolicyDisabled = 1,          ///< The compile request disabled active updates.
  kCompileTime = 2,             ///< Field is immutable for the accepted plan.
  kLifecycleDeferred = 3,       ///< Lifecycle integration is not active for this compile.
  kRecompileRequired = 4,       ///< Updating the field requires a newly compiled plan.
  kUnsupportedLogicalType = 5,  ///< EA1 has no bounded scalar patch representation.
  kReplacementUnavailable = 6,  ///< No exact non-throwing scalar replacement route exists.
  kProfileDisallowed = 7,       ///< Effective profile rejects active mutation.
  kCapacityUnavailable = 8,     ///< Finite capacity validation rejected enablement.
};

/// @brief Record profile-aware admission of one aggregate schema.
enum class ParameterSchemaAdmissionDisposition : std::uint8_t {
  kOpaqueLocal = 0,             ///< No declared field schema; typed local execution only.
  kDeclaredPartial = 1,         ///< Declared fields cover only part of the aggregate.
  kCompleteAuthorAsserted = 2,  ///< Complete coverage is asserted but not independently proven.
  kCompleteProven = 3,          ///< Complete coverage has qualified proof evidence.
  kRejectedByProfile = 4,       ///< Effective profile required stronger schema evidence.
};

/// @brief Record the aggregate/field materialization strategy selected by compilation.
enum class ParameterMaterializationStrategy : std::uint8_t {
  kStaticNoParams = 0,  ///< One process-static \ref NoParams value; no generation storage.
  kFixedObject = 1,     ///< Complete fixed object storage with bounded scalar operations.
  kColdOwnedTyped = 2,  ///< Plan-owned cold typed value may use dynamic C++ storage.
  kRejected = 3,        ///< No strategy satisfies the effective profile and schema evidence.
};

/// @brief Select the only ordinary catalog projection admitted by EA1.
enum class ParamCatalogAccessPolicy : std::uint8_t {
  kPublic = 0,  ///< Apply transient/sensitive/internal/profile/codec projection policy.
};

/// @brief Own a decoded canonical path to one Operator instance.
///
/// Each segment is nonempty valid UTF-8 and the path contains no URI, Graph, or Operator borrow.
/// Copies own their segment storage. Distinct objects may be read concurrently; mutating or
/// assigning one object while another thread reads that same object requires external ordering.
class HOLOSCAN_CORE_EXPORT OperatorPath {
 public:
  /// @brief Create an empty path for result and record initialization.
  /// @note An empty value is not a valid lookup path; use \ref parse or \ref from_segments before
  /// lookup.
  OperatorPath() = default;

  /// @brief Parse slash-delimited percent-encoded path segments.
  /// @param path Nonempty canonical path with no leading, trailing, or repeated slash.
  /// @return Owning decoded path.
  /// @retval ErrorCode::kInvalidArgument The path grammar, escape, UTF-8, or segment policy is
  /// invalid.
  /// @retval ErrorCode::kResourceExhausted Owning decoded storage could not be allocated.
  [[nodiscard]] static expected<OperatorPath, Error> parse(std::string_view path);

  /// @brief Copy already-decoded path segments.
  /// @param segments Nonempty valid UTF-8 segments.
  /// @return Owning path.
  /// @retval ErrorCode::kInvalidArgument A segment is empty or violates the UTF-8/segment policy.
  /// @retval ErrorCode::kResourceExhausted Owning segment storage could not be allocated.
  [[nodiscard]] static expected<OperatorPath, Error> from_segments(
      std::span<const std::string> segments);

  /// @brief Return decoded root-to-leaf path segments.
  /// @return Borrowed segments valid for this value's lifetime.
  /// @note The call does not allocate or block. Assignment or destruction invalidates the returned
  /// span.
  [[nodiscard]] std::span<const std::string> segments() const noexcept { return segments_; }

  /// @brief Compare decoded path segments exactly.
  /// @return \c true when both paths contain the same ordered decoded segments.
  friend bool operator==(const OperatorPath&, const OperatorPath&) noexcept = default;

 private:
  explicit OperatorPath(std::vector<std::string> segments) : segments_(std::move(segments)) {}
  std::vector<std::string> segments_;
  friend class detail::ParameterUpdateCoordinator;
  friend class detail::ParameterCatalogAccess;
  friend class ParameterCatalogView;
  friend class ParamSnapshotScope;
};

/// @brief Own a decoded canonical path to one declared Parameter field.
///
/// The path owns its decoded segments and carries no member pointer, schema operation, aggregate
/// address, or Parameter value. The same concurrency rules as \ref OperatorPath apply.
class HOLOSCAN_CORE_EXPORT ParamPath {
 public:
  /// @brief Create an empty path for result and record initialization.
  /// @note An empty value is not a valid lookup path.
  ParamPath() = default;

  /// @brief Parse slash-delimited percent-encoded field-path segments.
  /// @param path Nonempty canonical path with no leading, trailing, or repeated slash.
  /// @return Owning decoded path.
  /// @retval ErrorCode::kInvalidArgument The path grammar, escape, UTF-8, or segment policy is
  /// invalid.
  /// @retval ErrorCode::kResourceExhausted Owning decoded storage could not be allocated.
  [[nodiscard]] static expected<ParamPath, Error> parse(std::string_view path);

  /// @brief Copy already-decoded field-path segments.
  /// @param segments Nonempty valid UTF-8 segments.
  /// @return Owning path.
  /// @retval ErrorCode::kInvalidArgument A segment is empty or violates the UTF-8/segment policy.
  /// @retval ErrorCode::kResourceExhausted Owning segment storage could not be allocated.
  [[nodiscard]] static expected<ParamPath, Error> from_segments(
      std::span<const std::string> segments);

  /// @brief Return decoded field-path segments.
  /// @return Borrowed segments valid for this value's lifetime.
  /// @note The call does not allocate or block. Assignment or destruction invalidates the returned
  /// span.
  [[nodiscard]] std::span<const std::string> segments() const noexcept { return segments_; }

  /// @brief Compare decoded field-path segments exactly.
  /// @return \c true when both paths contain the same ordered decoded segments.
  friend bool operator==(const ParamPath&, const ParamPath&) noexcept = default;

 private:
  explicit ParamPath(std::vector<std::string> segments) : segments_(std::move(segments)) {}
  std::vector<std::string> segments_;
  friend class detail::ParameterUpdateCoordinator;
  friend class detail::ParameterCatalogAccess;
  friend class ParameterCatalogView;
};

/// @brief Hold one deterministic compatibility digest for a Parameter catalog.
///
/// The digest binds one exact source plan/run and its ordered policy-visible identity and metadata
/// records. It is compatibility evidence only: it is not control authority, an authenticity proof,
/// a Parameter value digest, or a safe redaction of sensitive data.
struct ParamCatalogDigest {
  std::array<std::byte, 32> value{};  ///< Complete canonical digest bytes.
  /// @brief Compare complete canonical digest bytes.
  /// @return \c true when all bytes match exactly.
  friend constexpr bool operator==(ParamCatalogDigest, ParamCatalogDigest) noexcept = default;
};

/// @brief Own immutable catalog facts for one plan-bound Operator.
///
/// This record owns every variable-length field and contains no factory, aggregate, value, local
/// type token, operation pointer, or Runtime update authority.
struct OperatorParamRecord {
  OperatorRef operator_instance{};           ///< Exact plan-bound identity; run generation zero.
  OperatorPath path{};                       ///< Canonical structured Operator path.
  std::optional<Profile> effective_profile;  ///< Compile-resolved profile when admitted.
  ParamSchemaAvailability schema_availability{
      ParamSchemaAvailability::kOpaqueLocal};  ///< Frozen generic-schema availability.
  ParameterSchemaAdmissionDisposition schema_disposition{
      ParameterSchemaAdmissionDisposition::kOpaqueLocal};  ///< Total schema admission outcome.
  ParameterMaterializationStrategy materialization{
      ParameterMaterializationStrategy::kRejected};  ///< Accepted generation-zero strategy.
  std::string schema_id;                ///< Stable schema ID, or empty for opaque mode.
  ParamSchemaVersion schema_version{};  ///< Stable schema version, or zero.
  ParamSchemaHash structural_hash{};    ///< Portable decoding-shape identity.
  ParamSchemaHash contract_hash{};      ///< Portable policy identity.
  ParamSchemaHash metadata_hash{};      ///< Contract-plus-description identity.
  ParamSchemaCoverage schema_coverage{
      ParamSchemaCoverage::kDeclaredFieldsOnly};  ///< Declared aggregate coverage.
  ParamSchemaProof schema_proof{ParamSchemaProof::kAuthorAsserted};  ///< Coverage evidence.
};

/// @brief Own immutable policy-relevant facts for one plan-bound field.
///
/// The public record deliberately contains no startup value, protected value digest, value origin,
/// low-entropy secret hash, member pointer, or mutation authority. Sensitive values are represented
/// only by \ref params::ParamProjectionStatus::kRedactedSensitive.
struct ParamFieldCatalogRecord {
  ParamInstanceId instance{};  ///< Collision-checked plan-scoped compact field identity.
  ParamSchemaHash
      instance_identity_digest{};  ///< Complete Operator-path-plus-field identity digest.
  ParamAddress address{};          ///< Exact plan-bound field address.
  ParamSchemaHash full_id{};       ///< Complete schema-lineage-local field identity.
  ParamPath path{};                ///< Canonical structured field path.
  ParamLogicalType logical_type{ParamLogicalType::kObject};   ///< Frozen logical value family.
  ParamFieldTypeShape type_shape{};                           ///< Portable shape and scalar width.
  ParamMutability mutability{ParamMutability::kCompileTime};  ///< Declared update boundary.
  LifecycleStage earliest_invalidated_stage{
      LifecycleStage::kConfigure};  ///< Earliest lifecycle stage invalidated by an update.
  ParamPersistence persistence{ParamPersistence::kPersisted};  ///< Durable projection policy.
  ParamVisibility visibility{ParamVisibility::kPublic};        ///< Discovery visibility policy.
  ParamExternalAccess external_access{ParamExternalAccess::kNone};  ///< External eligibility.
  ParamSensitivity sensitivity{ParamSensitivity::kNormal};          ///< Value-redaction policy.
  bool required_on_load{};         ///< Whether complete import requires an archived value.
  std::string alias;               ///< Prior canonical input spelling; never output identity.
  std::string validator_id;        ///< Stable local-validator identity, or empty.
  ParamConstraints constraints{};  ///< Portable declared bounds and choices.
  ParamStringEncoding string_encoding{
      ParamStringEncoding::kArbitraryBytes};  ///< Direct string encoding policy.
  ParamStringEncoding element_string_encoding{
      ParamStringEncoding::kArbitraryBytes};  ///< Sequence-element string policy.
  ParamValueRouteDisposition value_route{
      ParamValueRouteDisposition::kUnsupported};  ///< Compile-frozen generic value route.
  ParamStorageTraitsSummary storage{};            ///< Intrinsic storage facts.
  ParameterMaterializationStrategy materialization{
      ParameterMaterializationStrategy::kRejected};  ///< Aggregate-backed field strategy.
  params::ParamProjectionStatus projection{
      params::ParamProjectionStatus::kExcludedUnsupportedCodec};  ///< Public projection outcome.
  ParameterFieldUpdateDisposition update_disposition{
      ParameterFieldUpdateDisposition::kPolicyDisabled};  ///< Total active-update disposition.
  bool adapter_eligible{};  ///< Whether stable identity/type/codec policy permits an adapter key.
};

/// @brief Borrow one immutable field record from a Parameter catalog.
///
/// The view never owns the record. Assignment, mutation, or destruction of the source
/// \ref ParameterCatalogView invalidates it. Concurrent reads are safe while the source catalog is
/// unchanged.
class ParamFieldView {
 public:
  /// @brief Create an empty field view.
  ParamFieldView() = default;

  /// @brief Return the borrowed field record pointer.
  /// @return Record pointer, or \c nullptr for an empty view.
  [[nodiscard]] const ParamFieldCatalogRecord* get() const noexcept { return record_; }

  /// @brief Return whether this view refers to a field record.
  /// @return \c true when \ref get is non-null.
  [[nodiscard]] explicit operator bool() const noexcept { return record_ != nullptr; }

  /// @brief Return the borrowed field record.
  /// @return Immutable record owned by the source catalog.
  /// @pre This view is nonempty and the source catalog remains alive and unchanged.
  [[nodiscard]] const ParamFieldCatalogRecord& operator*() const noexcept { return *record_; }

  /// @brief Return the borrowed field record pointer for member access.
  /// @return Non-null immutable record pointer.
  /// @pre This view is nonempty and the source catalog remains alive and unchanged.
  [[nodiscard]] const ParamFieldCatalogRecord* operator->() const noexcept { return record_; }

 private:
  explicit ParamFieldView(const ParamFieldCatalogRecord* record) noexcept : record_(record) {}
  const ParamFieldCatalogRecord* record_{};
  friend class ParameterCatalogView;
};

/// @brief Own an immutable policy-filtered plan-bound Parameter catalog.
///
/// Records own all names and paths and contain no aggregate address, member token, operation
/// pointer, typed value, value digest, or update authority. Copies remain valid independently of
/// the source ExecutionPlan. Task11 may rebind a copied filtered catalog to one run generation.
class HOLOSCAN_CORE_EXPORT ParameterCatalogView {
 public:
  /// @brief Create an empty catalog with an all-zero digest.
  ParameterCatalogView() = default;

  /// @brief Return the deterministic compatibility digest of this exact catalog view.
  /// @return Deterministic plan/run-bound policy-visible catalog digest. A default-constructed
  /// inert catalog returns all zeros; a valid admitted projection with no visible records retains
  /// a nonzero source plan/run-bound digest.
  /// @note The call does not allocate or expose a value digest.
  [[nodiscard]] ParamCatalogDigest compatibility_digest() const noexcept { return digest_; }

  /// @brief Return policy-visible Operator records.
  /// @return Borrowed canonical records valid until this catalog is assigned or destroyed.
  /// @note The call does not allocate or block.
  [[nodiscard]] std::span<const OperatorParamRecord> operators() const noexcept {
    return operators_;
  }

  /// @brief Return policy-visible field records in canonical order.
  /// @return Borrowed canonical records valid until this catalog is assigned or destroyed.
  /// @note Internal fields are absent; sensitive fields contain redaction metadata but no value or
  /// digest.
  [[nodiscard]] std::span<const ParamFieldCatalogRecord> fields() const noexcept { return fields_; }

  /// @brief Find one exact plan/run-bound field record.
  /// @param address Exact Operator generation and schema-local field identity.
  /// @return Borrowed immutable field view.
  /// @retval ErrorCode::kStaleOperatorRef The address generation differs from this nonempty
  /// catalog.
  /// @retval ErrorCode::kNotFound No policy-visible record has the exact address.
  /// @note The call performs a read-only linear lookup and does not allocate.
  [[nodiscard]] expected<ParamFieldView, Error> field(ParamAddress address) const;

  /// @brief Find one exact compact field-instance identity.
  /// @param instance Collision-checked identity from this catalog.
  /// @return Borrowed immutable field view.
  /// @retval ErrorCode::kNotFound No policy-visible record has the identity.
  /// @note Compact identities from another catalog must be qualified by
  /// \ref compatibility_digest.
  [[nodiscard]] expected<ParamFieldView, Error> field(ParamInstanceId instance) const;

  /// @brief Resolve one structured Operator and field path.
  /// @param operator_path Exact decoded Operator path.
  /// @param param_path Exact decoded field path.
  /// @return Plan/run-bound field address.
  /// @retval ErrorCode::kNotFound The Operator or policy-visible field is absent.
  /// @note Resolution does not allocate, normalize, apply aliases, or grant read/update authority.
  [[nodiscard]] expected<ParamAddress, Error> resolve(const OperatorPath& operator_path,
                                                      const ParamPath& param_path) const;

 private:
  ParamCatalogDigest digest_{};
  std::vector<OperatorParamRecord> operators_;
  std::vector<ParamFieldCatalogRecord> fields_;
  friend class detail::ParamControlAccess;
  friend class detail::ParameterCatalogAccess;
  friend class detail::ParameterUpdateCoordinator;
};

}  // namespace holoscan
