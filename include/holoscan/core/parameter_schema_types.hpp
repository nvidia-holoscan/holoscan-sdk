/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>
#include <type_traits>

#include <holoscan/core/parameter.hpp>

namespace holoscan {

/// @brief Carry a nonzero author-declared parameter schema version.
using ParamSchemaVersion = std::uint32_t;

/// @brief Hold one deterministic 256-bit parameter schema fingerprint.
struct ParamSchemaHash {
  std::array<std::byte, 32> value{};  ///< Complete fingerprint bytes.

  /// @brief Compare all canonical digest bytes.
  /// @return \c true when the digests are identical.
  friend constexpr bool operator==(ParamSchemaHash, ParamSchemaHash) noexcept = default;
};

/// @brief Identify whether a typed aggregate declares field metadata.
enum class ParamSchemaAvailability : std::uint8_t {
  kOpaqueLocal,  ///< No portable field schema is declared.
  kDeclared,     ///< A partial or complete static schema is declared.
};

/// @brief Identify whether a declaration covers selected fields or the complete aggregate.
enum class ParamSchemaCoverage : std::uint8_t {
  kDeclaredFieldsOnly,  ///< Only listed fields have generic capabilities.
  kComplete,            ///< The declaration classifies the complete aggregate.
};

/// @brief Identify the evidence strength behind a schema coverage claim.
enum class ParamSchemaProof : std::uint8_t {
  kAuthorAsserted,  ///< Coverage is asserted without a qualified inventory.
  kProven,          ///< A qualified inventory proved the coverage claim.
};

/// @brief Classify when a declared parameter field may change.
enum class ParamMutability : std::uint8_t {
  kCompileTime,  ///< Set before compile and frozen for one accepted plan.
  kRuntime,      ///< Eligible for a later admitted immutable generation update.
  kLifecycle,    ///< Requires a lifecycle transaction from the declared stage.
  kRecompile,    ///< Requires a newly compiled execution plan.
};

/// @brief Classify whether a field participates in policy-filtered durable projection.
enum class ParamPersistence : std::uint8_t {
  kPersisted,  ///< Eligible for policy-filtered durable projection.
  kTransient,  ///< Never included in durable projection.
};

/// @brief Classify the declared semantic encoding of one `std::string` field.
enum class ParamStringEncoding : std::uint8_t {
  kArbitraryBytes,  ///< Default byte string; no UTF-8 interpretation or validation.
  kUtf8,            ///< Require valid UTF-8 and permit `max_utf8_bytes`.
};

/// @brief Classify ordinary local field discovery.
enum class ParamVisibility : std::uint8_t {
  kPublic,    ///< Eligible for ordinary local discovery subject to policy.
  kInternal,  ///< Hidden from ordinary discovery; this does not imply sensitivity.
};

/// @brief Classify eligibility for separately authorized external access.
enum class ParamExternalAccess : std::uint8_t {
  kNone,      ///< No external read or update declaration.
  kReadOnly,  ///< Eligible for authorized projected reads.
  kTuning,    ///< Eligible for separately authorized runtime tuning.
};

/// @brief Classify whether ordinary projection must redact a field before value access.
enum class ParamSensitivity : std::uint8_t {
  kNormal,     ///< No sensitivity marker is declared.
  kSensitive,  ///< Ordinary projection must redact before value access.
};

/// @brief Record which qualification-sensitive field policy axes were authored explicitly.
///
/// The flags distinguish deliberate spelling from a semantically equal default. They are
/// qualification evidence rather than a different field policy: explicit and inferred forms with
/// the same policy values retain the same contract hash. A strict compile may require every flag
/// without changing ordinary low-ceremony schema authoring.
struct ParamFieldPolicyEvidence {
  bool persistence_explicit{};      ///< Persistence was selected by a builder call.
  bool external_access_explicit{};  ///< External-access disposition was selected explicitly.
  bool sensitivity_explicit{};      ///< Sensitivity disposition was selected explicitly.

  /// @brief Return whether all qualification-sensitive policy axes were authored explicitly.
  /// @return \c true only when persistence, external access, and sensitivity are all explicit.
  [[nodiscard]] constexpr bool complete() const noexcept {
    return persistence_explicit && external_access_explicit && sensitivity_explicit;
  }

  /// @brief Compare all explicit-policy evidence flags.
  /// @return \c true when both records preserve identical authored evidence.
  friend constexpr bool operator==(ParamFieldPolicyEvidence,
                                   ParamFieldPolicyEvidence) noexcept = default;
};

static_assert(std::is_standard_layout_v<ParamFieldPolicyEvidence>);
static_assert(std::is_trivially_copyable_v<ParamFieldPolicyEvidence>);

/// @brief Classify the portable logical value family of a declared field.
enum class ParamLogicalType : std::uint8_t {
  kBool,             ///< Boolean value.
  kSignedInteger,    ///< Signed integral or signed enum representation.
  kUnsignedInteger,  ///< Unsigned integral or enum representation.
  kFloatingPoint,    ///< Floating-point value no wider than double.
  kString,           ///< `std::string` bytes; explicit encoding policy may require UTF-8.
  kSequence,         ///< `std::vector` sequence value.
  kFilesystemPath,   ///< Cold `std::filesystem::path` value.
  kDuration,         ///< `std::chrono::duration` value.
  kObject,           ///< Nested or opaque object value.
};

/// @brief Classify the generic value route frozen for one declared field.
///
/// This disposition is independent of C++ storage boundedness and update
/// mutability. It records whether an admitted control plane may use the stable
/// fixed-scalar route without retaining a live schema descriptor.
enum class ParamValueRouteDisposition : std::uint8_t {
  kPortableScalar = 0,  ///< A stable fixed-scalar route is available.
  kLocalTypedOnly = 1,  ///< Only the owning typed Operator may interpret the field in baseline.
  kUnsupported = 2,     ///< No generic value route is available for the accepted plan.
};

/// @brief Describe the portable logical shape used for structural schema compatibility.
///
/// Widths are semantic bits derived from `std::numeric_limits`, never `sizeof(T)`. For a sequence,
/// `element_or_rep_type` and `element_or_rep_bit_width` describe one scalar element. For a direct
/// duration they describe the representation and the period fields are authoritative. A sequence
/// of durations is conservatively unstable because this V1 record has no second-level rep slot.
/// `has_stable_shape` is also false for a non-nested custom object, a native
/// `std::filesystem::path`, or another type lacking canonical shape identity; later compilation
/// must reject portable decoding until a qualified codec supplies that identity. Path sequences
/// inherit the same unstable disposition. A declared nested object becomes stable through its
/// nested schema ID/version and structural/contract hashes rather than through C++ object
/// representation.
struct ParamFieldTypeShape {
  ParamLogicalType type{ParamLogicalType::kObject};  ///< Top-level logical family.
  std::uint16_t bit_width{};                         ///< Scalar semantic width, or zero.
  ParamLogicalType element_or_rep_type{ParamLogicalType::kObject};  ///< Nested scalar family.
  std::uint16_t element_or_rep_bit_width{};     ///< Sequence-element or duration-rep width.
  std::int64_t duration_period_numerator{};     ///< Duration period numerator, or zero.
  std::int64_t duration_period_denominator{1};  ///< Duration period denominator, or one.
  bool optional{};          ///< Whether absence differs from a present underlying value.
  bool has_stable_shape{};  ///< Whether these fields fully identify the supported logical shape.
};

static_assert(std::is_standard_layout_v<ParamFieldTypeShape>);
static_assert(std::is_trivially_copyable_v<ParamFieldTypeShape>);

namespace params {

/// @brief Identify the single ordinary safe-projection outcome for a declared field.
///
/// Numeric values are stable inspection tokens only. Classification precedence is defined by
/// `classify_param_projection` and deliberately does not depend on this declaration order.
enum class ParamProjectionStatus : std::uint8_t {
  kIncluded = 0,                 ///< Policy permits an owning encoded value.
  kRedactedSensitive = 1,        ///< Sensitivity policy emits a no-value redaction marker.
  kExcludedInternal = 2,         ///< Internal visibility excludes ordinary projection.
  kExcludedTransient = 3,        ///< Persistence excludes the field before value access.
  kExcludedProfilePolicy = 4,    ///< Effective profile policy excludes projection.
  kExcludedUnsupportedCodec = 5  ///< No admitted portable codec can encode the field.
};

}  // namespace params

/// @brief Identify the exact scalar domain used by one portable field constraint value.
enum class ParamConstraintScalarKind : std::uint8_t {
  kNone,      ///< No scalar value is present.
  kSigned,    ///< `signed_value` is authoritative.
  kUnsigned,  ///< `unsigned_value` is authoritative.
  kFloating,  ///< `floating_value` is authoritative and finite.
};

/// @brief Store one portable scalar constraint without host-width or `long double` comparison.
///
/// The tagged member is compared only within its own domain. Schema lowering rejects non-finite
/// floating bounds and choices. Unused members are zero and have no semantic meaning.
struct ParamConstraintScalar {
  ParamConstraintScalarKind kind{ParamConstraintScalarKind::kNone};  ///< Active domain tag.
  std::int64_t signed_value{};     ///< Exact signed value when kind is `kSigned`.
  std::uint64_t unsigned_value{};  ///< Exact unsigned value when kind is `kUnsigned`.
  double floating_value{};         ///< Finite value when kind is `kFloating`.
};

static_assert(std::is_standard_layout_v<ParamConstraintScalar>);
static_assert(std::is_trivially_copyable_v<ParamConstraintScalar>);

/// @brief Maximum number of scalar choices stored by one static field declaration.
///
/// This fixed public capacity bounds schema storage and constant-evaluation work. Supplying a
/// seventeenth choice rejects the schema; the implementation never truncates or allocates.
inline constexpr std::size_t kParamAllowedValueCapacity = 16;

/// @brief Describe fixed-size portable validation constraints for one declared field.
struct ParamConstraints {
  bool has_range{};                   ///< Whether inclusive `range_min` and `range_max` apply.
  ParamConstraintScalar range_min{};  ///< Inclusive lower bound in the exact scalar domain.
  ParamConstraintScalar range_max{};  ///< Inclusive upper bound in the exact scalar domain.
  std::array<ParamConstraintScalar, kParamAllowedValueCapacity>
      allowed_values{};                  ///< Fixed-capacity allowed set.
  std::uint8_t allowed_value_count{};    ///< Active prefix length in `allowed_values`.
  std::size_t max_utf8_bytes{};          ///< Positive UTF-8 byte bound, or zero when absent.
  std::size_t max_element_utf8_bytes{};  ///< Positive per-string element bound, or zero.
  std::size_t max_elements{};            ///< Positive sequence element bound, or zero when absent.
  std::size_t max_encoded_bytes{};       ///< Positive codec byte bound, or zero when absent.
};

static_assert(std::is_standard_layout_v<ParamConstraints>);
static_assert(std::is_trivially_copyable_v<ParamConstraints>);

/// @brief Bind one stable authored enum label to its canonical scalar value.
struct ParamEnumChoice {
  std::string_view name{};   ///< Stable user-facing label with static lifetime.
  ParamScalarValue value{};  ///< Exact underlying enum value.
};

static_assert(std::is_standard_layout_v<ParamEnumChoice>);
static_assert(std::is_trivially_copyable_v<ParamEnumChoice>);

/// @brief Store a bounded authored enum domain in a static schema descriptor.
struct ParamEnumChoices {
  std::array<ParamEnumChoice, kParamAllowedValueCapacity> values{};  ///< Active-prefix storage.
  std::uint8_t count{};  ///< Number of active name/value pairs.
};

static_assert(std::is_standard_layout_v<ParamEnumChoices>);
static_assert(std::is_trivially_copyable_v<ParamEnumChoices>);

}  // namespace holoscan
