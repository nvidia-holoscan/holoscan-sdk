/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <forward_list>
#include <initializer_list>
#include <limits>
#include <list>
#include <map>
#include <memory>
#include <optional>
#include <queue>
#include <ratio>
#include <set>
#include <span>
#include <stack>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include <holoscan/core/lifecycle.hpp>
#include <holoscan/core/parameter.hpp>

namespace holoscan {

template <typename ParamsT, auto MemberPtr>
class ParamFieldBuilder;

struct ParamSchemaDescriptor;

template <typename ParamsT>
  requires ParameterAggregate<ParamsT>
[[nodiscard]] const ParamSchemaDescriptor* param_schema_descriptor() noexcept;

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

namespace detail {
class ParamSchemaTestAccess;

template <typename T>
struct IsChronoDuration : std::false_type {};

template <typename Rep, typename Period>
struct IsChronoDuration<std::chrono::duration<Rep, Period>> : std::true_type {};

template <typename T>
inline constexpr bool kIsChronoDuration = IsChronoDuration<std::remove_cv_t<T>>::value;

template <typename T>
struct IsStdVector : std::false_type {};

template <typename T, typename Allocator>
struct IsStdVector<std::vector<T, Allocator>> : std::true_type {};

template <typename T>
struct IsStringVector : std::false_type {};

template <typename Allocator>
struct IsStringVector<std::vector<std::string, Allocator>> : std::true_type {};

template <typename T>
struct StdVectorElement;

template <typename T, typename Allocator>
struct StdVectorElement<std::vector<T, Allocator>> {
  using Type = T;
};

template <typename T>
struct IsStdOptional : std::false_type {};

template <typename T>
struct IsStdOptional<std::optional<T>> : std::true_type {};

template <typename T>
struct UnwrapOptional {
  using Type = std::remove_cv_t<T>;
};

template <typename T>
struct UnwrapOptional<std::optional<T>> {
  using Type = T;
};

template <typename T>
using UnwrappedOptionalT = typename UnwrapOptional<std::remove_cv_t<T>>::Type;

template <typename T, bool = IsStdVector<UnwrappedOptionalT<T>>::value>
struct EnumDomain {
  using Type = UnwrappedOptionalT<T>;
};

template <typename T>
struct EnumDomain<T, true> {
  using Type = typename StdVectorElement<UnwrappedOptionalT<T>>::Type;
};

template <typename T>
using EnumDomainT = typename EnumDomain<T>::Type;

template <typename T>
inline constexpr bool kIsParamScalar =
    kIsChronoDuration<UnwrappedOptionalT<T>> || std::is_enum_v<UnwrappedOptionalT<T>> ||
    std::is_arithmetic_v<UnwrappedOptionalT<T>>;

template <typename T>
consteval ParamLogicalType logical_type() {
  using Value = std::remove_cv_t<T>;
  if constexpr (IsStdOptional<Value>::value) {
    static_assert(!IsStdOptional<UnwrappedOptionalT<Value>>::value,
                  "nested std::optional parameter fields are unsupported");
    return logical_type<UnwrappedOptionalT<Value>>();
  } else if constexpr (std::same_as<Value, bool>) {
    return ParamLogicalType::kBool;
  } else if constexpr (std::is_enum_v<Value>) {
    using Underlying = std::underlying_type_t<Value>;
    return std::is_signed_v<Underlying> ? ParamLogicalType::kSignedInteger
                                        : ParamLogicalType::kUnsignedInteger;
  } else if constexpr (std::is_unsigned_v<Value>) {
    return ParamLogicalType::kUnsignedInteger;
  } else if constexpr (std::is_integral_v<Value>) {
    return ParamLogicalType::kSignedInteger;
  } else if constexpr (std::is_floating_point_v<Value>) {
    return ParamLogicalType::kFloatingPoint;
  } else if constexpr (std::same_as<Value, std::string>) {
    return ParamLogicalType::kString;
  } else if constexpr (IsStdVector<Value>::value) {
    return ParamLogicalType::kSequence;
  } else if constexpr (std::same_as<Value, std::filesystem::path>) {
    return ParamLogicalType::kFilesystemPath;
  } else if constexpr (kIsChronoDuration<Value>) {
    return ParamLogicalType::kDuration;
  } else {
    return ParamLogicalType::kObject;
  }
}

template <typename Integer>
[[nodiscard]] consteval std::uint16_t integer_bit_width() {
  constexpr int digits =
      std::numeric_limits<Integer>::digits + (std::numeric_limits<Integer>::is_signed ? 1 : 0);
  static_assert(digits > 0 && digits <= std::numeric_limits<std::uint16_t>::max(),
                "parameter integer width must fit the portable shape record");
  return static_cast<std::uint16_t>(digits);
}

template <typename Floating>
[[nodiscard]] consteval std::uint16_t portable_floating_bit_width() {
  if constexpr (std::same_as<Floating, float> && std::numeric_limits<Floating>::is_iec559 &&
                std::numeric_limits<Floating>::digits == 24 &&
                std::numeric_limits<Floating>::max_exponent == 128) {
    return 32;
  }
  if constexpr (std::same_as<Floating, double> && std::numeric_limits<Floating>::is_iec559 &&
                std::numeric_limits<Floating>::digits == 53 &&
                std::numeric_limits<Floating>::max_exponent == 1024) {
    return 64;
  }
  return 0;
}

[[nodiscard]] constexpr bool intmax_fits_int64(std::intmax_t value) noexcept {
#if INTMAX_MAX <= INT64_MAX
  (void)value;
  return true;
#else
  return value >= static_cast<std::intmax_t>(INT64_MIN) &&
         value <= static_cast<std::intmax_t>(INT64_MAX);
#endif
}

template <typename T, bool AllowOptional = true>
[[nodiscard]] consteval ParamFieldTypeShape field_type_shape() {
  using Value = std::remove_cv_t<T>;
  if constexpr (IsStdOptional<Value>::value) {
    static_assert(AllowOptional, "only top-level std::optional parameter fields are supported");
    static_assert(!IsStdOptional<UnwrappedOptionalT<Value>>::value,
                  "nested std::optional parameter fields are unsupported");
    ParamFieldTypeShape result = field_type_shape<UnwrappedOptionalT<Value>, false>();
    result.optional = true;
    return result;
  } else if constexpr (std::same_as<Value, bool>) {
    return ParamFieldTypeShape{
        .type = ParamLogicalType::kBool, .bit_width = 1, .has_stable_shape = true};
  } else if constexpr (std::is_enum_v<Value>) {
    using Underlying = std::underlying_type_t<Value>;
    return ParamFieldTypeShape{
        .type = std::is_signed_v<Underlying> ? ParamLogicalType::kSignedInteger
                                             : ParamLogicalType::kUnsignedInteger,
        .bit_width = integer_bit_width<Underlying>(),
        .has_stable_shape = true,
    };
  } else if constexpr (std::is_integral_v<Value>) {
    return ParamFieldTypeShape{
        .type = std::is_signed_v<Value> ? ParamLogicalType::kSignedInteger
                                        : ParamLogicalType::kUnsignedInteger,
        .bit_width = integer_bit_width<Value>(),
        .has_stable_shape = true,
    };
  } else if constexpr (std::is_floating_point_v<Value>) {
    constexpr std::uint16_t width = portable_floating_bit_width<Value>();
    return ParamFieldTypeShape{
        .type = ParamLogicalType::kFloatingPoint,
        .bit_width = width,
        .has_stable_shape = width != 0,
    };
  } else if constexpr (std::same_as<Value, std::string>) {
    return ParamFieldTypeShape{.type = ParamLogicalType::kString, .has_stable_shape = true};
  } else if constexpr (IsStdVector<Value>::value) {
    using Element = typename StdVectorElement<Value>::Type;
    constexpr ParamFieldTypeShape element = field_type_shape<Element, false>();
    return ParamFieldTypeShape{
        .type = ParamLogicalType::kSequence,
        .element_or_rep_type = element.type,
        .element_or_rep_bit_width = element.bit_width,
        .duration_period_numerator = element.duration_period_numerator,
        .duration_period_denominator = element.duration_period_denominator,
        .has_stable_shape = element.has_stable_shape &&
                            element.type != ParamLogicalType::kSequence &&
                            element.type != ParamLogicalType::kDuration &&
                            element.type != ParamLogicalType::kObject,
    };
  } else if constexpr (std::same_as<Value, std::filesystem::path>) {
    // A native path remains a legal cold typed value, but its character representation and
    // normalization are platform-specific.  Keep the shape unstable until a qualified codec
    // supplies the explicitly declared logical UTF-8 path identity owned by HSDK5-018.
    return ParamFieldTypeShape{.type = ParamLogicalType::kFilesystemPath};
  } else if constexpr (kIsChronoDuration<Value>) {
    using Rep = typename Value::rep;
    using Period = typename Value::period;
    constexpr ParamFieldTypeShape rep = field_type_shape<Rep, false>();
    constexpr bool period_fits = intmax_fits_int64(Period::num) && intmax_fits_int64(Period::den);
    return ParamFieldTypeShape{
        .type = ParamLogicalType::kDuration,
        .element_or_rep_type = rep.type,
        .element_or_rep_bit_width = rep.bit_width,
        .duration_period_numerator =
            period_fits ? static_cast<std::int64_t>(Period::num) : std::int64_t{},
        .duration_period_denominator =
            period_fits ? static_cast<std::int64_t>(Period::den) : std::int64_t{1},
        .has_stable_shape = rep.has_stable_shape && period_fits,
    };
  } else {
    return ParamFieldTypeShape{.type = ParamLogicalType::kObject};
  }
}

template <typename T>
constexpr ParamConstraintScalar to_scalar(T value) {
  using Value = std::remove_cv_t<T>;
  if constexpr (kIsChronoDuration<Value>) {
    return to_scalar(value.count());
  } else if constexpr (std::is_enum_v<Value>) {
    return to_scalar(static_cast<std::underlying_type_t<Value>>(value));
  } else if constexpr (std::is_floating_point_v<Value>) {
    static_assert(sizeof(Value) <= sizeof(double),
                  "parameter constraints support floating types no wider than double");
    const double converted = static_cast<double>(value);
    return ParamConstraintScalar{.kind = ParamConstraintScalarKind::kFloating,
                                 .floating_value = converted == 0.0 ? 0.0 : converted};
  } else if constexpr (std::is_signed_v<Value>) {
    return ParamConstraintScalar{.kind = ParamConstraintScalarKind::kSigned,
                                 .signed_value = static_cast<std::int64_t>(value)};
  } else if constexpr (std::is_integral_v<Value>) {
    return ParamConstraintScalar{.kind = ParamConstraintScalarKind::kUnsigned,
                                 .unsigned_value = static_cast<std::uint64_t>(value)};
  } else {
    static_assert(std::is_arithmetic_v<Value>, "parameter constraint requires a scalar value");
  }
}

struct DurationScalarConversion {
  ParamConstraintScalar scalar{};
  bool valid{};
};

[[nodiscard]] constexpr bool scalar_finite(ParamConstraintScalar value) noexcept;

template <typename Integer>
[[nodiscard]] constexpr std::pair<bool, std::uint64_t> integer_magnitude(Integer value) noexcept {
  static_assert(std::is_integral_v<Integer> && sizeof(Integer) <= sizeof(std::uint64_t));
  if constexpr (std::same_as<Integer, bool>) {
    return {false, static_cast<std::uint64_t>(value)};
  } else if constexpr (std::is_signed_v<Integer>) {
    using Unsigned = std::make_unsigned_t<Integer>;
    if (value < 0) {
      return {true,
              static_cast<std::uint64_t>(static_cast<Unsigned>(0) - static_cast<Unsigned>(value))};
    }
    return {false, static_cast<std::uint64_t>(value)};
  } else {
    return {false, static_cast<std::uint64_t>(value)};
  }
}

template <typename TargetRep>
[[nodiscard]] constexpr DurationScalarConversion integral_duration_scalar(
    bool negative, std::uint64_t magnitude) noexcept {
  static_assert(std::is_integral_v<TargetRep> && sizeof(TargetRep) <= sizeof(std::uint64_t));
  if constexpr (std::is_signed_v<TargetRep>) {
    constexpr std::uint64_t positive_limit =
        static_cast<std::uint64_t>(std::numeric_limits<TargetRep>::max());
    constexpr std::uint64_t negative_limit = positive_limit + 1U;
    if ((!negative && magnitude > positive_limit) || (negative && magnitude > negative_limit)) {
      return {};
    }
    const std::int64_t encoded =
        negative ? (magnitude == negative_limit
                        ? static_cast<std::int64_t>(std::numeric_limits<TargetRep>::min())
                        : -static_cast<std::int64_t>(magnitude))
                 : static_cast<std::int64_t>(magnitude);
    return {
        ParamConstraintScalar{.kind = ParamConstraintScalarKind::kSigned, .signed_value = encoded},
        true};
  } else {
    if (negative || magnitude > static_cast<std::uint64_t>(std::numeric_limits<TargetRep>::max())) {
      return {};
    }
    return {ParamConstraintScalar{.kind = ParamConstraintScalarKind::kUnsigned,
                                  .unsigned_value = magnitude},
            true};
  }
}

template <typename TargetDuration, typename SourceDuration>
[[nodiscard]] constexpr DurationScalarConversion integral_duration_scalar_in_field_domain(
    SourceDuration value) noexcept {
  using TargetRep = typename TargetDuration::rep;
  auto [negative, magnitude] = integer_magnitude(value.count());
  constexpr std::uint64_t numerator_a = static_cast<std::uint64_t>(SourceDuration::period::num);
  constexpr std::uint64_t numerator_b = static_cast<std::uint64_t>(TargetDuration::period::den);
  constexpr std::uint64_t denominator_a = static_cast<std::uint64_t>(SourceDuration::period::den);
  constexpr std::uint64_t denominator_b = static_cast<std::uint64_t>(TargetDuration::period::num);
  constexpr std::uint64_t positive_limit =
      static_cast<std::uint64_t>(std::numeric_limits<TargetRep>::max());
  constexpr std::uint64_t target_limit =
      std::is_signed_v<TargetRep> ? positive_limit + 1U : positive_limit;
  std::uint64_t scaled{};
  if (!scale_magnitude_exact(magnitude,
                             numerator_a,
                             numerator_b,
                             denominator_a,
                             denominator_b,
                             target_limit,
                             scaled)) {
    return {};
  }
  return integral_duration_scalar<TargetRep>(negative, scaled);
}

template <typename TargetDuration, typename SourceDuration>
[[nodiscard]] constexpr DurationScalarConversion arithmetic_duration_scalar_in_field_domain(
    SourceDuration value) noexcept {
  using TargetRep = typename TargetDuration::rep;
  std::uint64_t numerator_a = static_cast<std::uint64_t>(SourceDuration::period::num);
  std::uint64_t numerator_b = static_cast<std::uint64_t>(TargetDuration::period::den);
  std::uint64_t denominator_a = static_cast<std::uint64_t>(SourceDuration::period::den);
  std::uint64_t denominator_b = static_cast<std::uint64_t>(TargetDuration::period::num);
  cancel_scale_factor(numerator_a, denominator_a);
  cancel_scale_factor(numerator_a, denominator_b);
  cancel_scale_factor(numerator_b, denominator_a);
  cancel_scale_factor(numerator_b, denominator_b);
  const long double scaled =
      static_cast<long double>(value.count()) * static_cast<long double>(numerator_a) *
      static_cast<long double>(numerator_b) / static_cast<long double>(denominator_a) /
      static_cast<long double>(denominator_b);
  if (scaled != scaled) {
    return {};
  }
  if constexpr (std::is_integral_v<TargetRep>) {
    if (scaled < static_cast<long double>(std::numeric_limits<TargetRep>::lowest()) ||
        scaled > static_cast<long double>(std::numeric_limits<TargetRep>::max())) {
      return {};
    }
    const TargetRep converted = static_cast<TargetRep>(scaled);
    if (static_cast<long double>(converted) != scaled) {
      return {};
    }
    return {to_scalar(converted), true};
  } else {
    if (scaled < static_cast<long double>(std::numeric_limits<TargetRep>::lowest()) ||
        scaled > static_cast<long double>(std::numeric_limits<TargetRep>::max())) {
      return {};
    }
    const ParamConstraintScalar scalar = to_scalar(static_cast<TargetRep>(scaled));
    return {scalar, scalar_finite(scalar)};
  }
}

template <typename TargetDuration, typename SourceDuration>
[[nodiscard]] constexpr DurationScalarConversion duration_scalar_in_field_domain(
    SourceDuration value) noexcept {
  static_assert(kIsChronoDuration<TargetDuration> && kIsChronoDuration<SourceDuration>);
  using TargetRep = typename TargetDuration::rep;
  using SourceRep = typename SourceDuration::rep;
  static_assert(std::is_arithmetic_v<TargetRep> && std::is_arithmetic_v<SourceRep>,
                "parameter duration constraints require arithmetic representation types");

  if constexpr (std::is_integral_v<TargetRep> && std::is_integral_v<SourceRep>) {
    return integral_duration_scalar_in_field_domain<TargetDuration>(value);
  } else {
    return arithmetic_duration_scalar_in_field_domain<TargetDuration>(value);
  }
}

[[nodiscard]] constexpr bool valid_utf8(std::string_view text) noexcept {
  std::size_t index = 0;
  while (index < text.size()) {
    const auto lead = static_cast<std::uint8_t>(text[index]);
    std::size_t continuation_count = 0;
    std::uint32_t code_point = 0;
    if (lead <= 0x7FU) {
      ++index;
      continue;
    }
    if ((lead & 0xE0U) == 0xC0U) {
      continuation_count = 1;
      code_point = lead & 0x1FU;
    } else if ((lead & 0xF0U) == 0xE0U) {
      continuation_count = 2;
      code_point = lead & 0x0FU;
    } else if ((lead & 0xF8U) == 0xF0U) {
      continuation_count = 3;
      code_point = lead & 0x07U;
    } else {
      return false;
    }
    if (index + continuation_count >= text.size()) {
      return false;
    }
    for (std::size_t offset = 1; offset <= continuation_count; ++offset) {
      const auto continuation = static_cast<std::uint8_t>(text[index + offset]);
      if ((continuation & 0xC0U) != 0x80U) {
        return false;
      }
      code_point = (code_point << 6U) | (continuation & 0x3FU);
    }
    const bool overlong = (continuation_count == 1U && code_point < 0x80U) ||
                          (continuation_count == 2U && code_point < 0x800U) ||
                          (continuation_count == 3U && code_point < 0x10000U);
    if (overlong || code_point > 0x10FFFFU || (code_point >= 0xD800U && code_point <= 0xDFFFU)) {
      return false;
    }
    index += continuation_count + 1U;
  }
  return true;
}

[[nodiscard]] constexpr bool valid_schema_text(std::string_view text) noexcept {
  if (text.empty() || !valid_utf8(text)) {
    return false;
  }
  for (const char character : text) {
    const auto byte = static_cast<std::uint8_t>(character);
    if (byte == 0U || byte < 0x20U || byte == 0x7FU) {
      return false;
    }
  }
  return text.front() != ' ' && text.back() != ' ';
}

[[nodiscard]] constexpr bool fits_canonical_size(std::size_t value) noexcept {
#if SIZE_MAX <= UINT64_MAX
  (void)value;
  return true;
#else
  return value <= static_cast<std::size_t>(UINT64_MAX);
#endif
}

template <typename T>
concept HasParamSchema = requires { holoscan_param_schema(type_tag<T>{}); };

[[nodiscard]] constexpr bool scalar_finite(ParamConstraintScalar value) noexcept {
  if (value.kind != ParamConstraintScalarKind::kFloating) {
    return value.kind != ParamConstraintScalarKind::kNone;
  }
  return value.floating_value == value.floating_value &&
         value.floating_value != std::numeric_limits<double>::infinity() &&
         value.floating_value != -std::numeric_limits<double>::infinity();
}

[[nodiscard]] constexpr bool scalar_less(ParamConstraintScalar lhs,
                                         ParamConstraintScalar rhs) noexcept {
  if (lhs.kind != rhs.kind) {
    return false;
  }
  switch (lhs.kind) {
    case ParamConstraintScalarKind::kSigned:
      return lhs.signed_value < rhs.signed_value;
    case ParamConstraintScalarKind::kUnsigned:
      return lhs.unsigned_value < rhs.unsigned_value;
    case ParamConstraintScalarKind::kFloating:
      return lhs.floating_value < rhs.floating_value;
    case ParamConstraintScalarKind::kNone:
      return false;
  }
  return false;
}

[[nodiscard]] constexpr bool scalar_equal(ParamConstraintScalar lhs,
                                          ParamConstraintScalar rhs) noexcept {
  if (lhs.kind != rhs.kind) {
    return false;
  }
  switch (lhs.kind) {
    case ParamConstraintScalarKind::kSigned:
      return lhs.signed_value == rhs.signed_value;
    case ParamConstraintScalarKind::kUnsigned:
      return lhs.unsigned_value == rhs.unsigned_value;
    case ParamConstraintScalarKind::kFloating:
      return lhs.floating_value == rhs.floating_value;
    case ParamConstraintScalarKind::kNone:
      return true;
  }
  return false;
}

// This dependency-free constexpr SHA-256 implementation hashes only explicitly encoded logical
// schema facts. It never hashes C++ object representation, RTTI, addresses, source locations, or
// callable bytes. Every caller supplies a versioned domain tag before its canonical tuple.
// Boolean values use one byte; every other integral or enum value uses its exact declared width
// in little-endian order; finite floating constraint values use IEEE binary64 bits; host sizes are
// first checked and widened to uint64_t. The golden schema triples freeze this V1 numeric codec.
class StableHashBuilder {
 public:
  constexpr StableHashBuilder() = default;

  constexpr void add_byte(std::uint8_t byte) noexcept {
    buffer_[buffer_size_++] = byte;
    ++total_bytes_;
    if (buffer_size_ == buffer_.size()) {
      transform();
      buffer_size_ = 0;
    }
  }

  constexpr void add(std::string_view text) noexcept {
    for (const char character : text) {
      add_byte(static_cast<std::uint8_t>(character));
    }
    add_byte(0xFFU);
  }

  template <typename T>
    requires std::is_integral_v<T> || std::is_enum_v<T>
  constexpr void add(T value) noexcept {
    if constexpr (std::is_enum_v<T>) {
      add(static_cast<std::underlying_type_t<T>>(value));
    } else if constexpr (std::same_as<std::remove_cv_t<T>, bool>) {
      add_byte(value ? 1U : 0U);
    } else {
      using Unsigned = std::make_unsigned_t<T>;
      Unsigned encoded = static_cast<Unsigned>(value);
      for (std::size_t index = 0; index < sizeof(Unsigned); ++index) {
        add_byte(static_cast<std::uint8_t>(encoded & static_cast<Unsigned>(0xFFU)));
        if (index + 1U < sizeof(Unsigned)) {
          encoded = static_cast<Unsigned>(encoded >> 8U);
        }
      }
    }
  }

  /// Encode one host size as a canonical unsigned 64-bit little-endian value.
  ///
  /// Schema builders reject values that are not representable before this method is reached.
  constexpr void add_canonical_size(std::size_t value) noexcept {
    add(static_cast<std::uint64_t>(value));
  }

  constexpr void add(ParamConstraintScalar value) noexcept {
    add(value.kind);
    switch (value.kind) {
      case ParamConstraintScalarKind::kSigned:
        add(value.signed_value);
        break;
      case ParamConstraintScalarKind::kUnsigned:
        add(value.unsigned_value);
        break;
      case ParamConstraintScalarKind::kFloating:
        add(std::bit_cast<std::uint64_t>(value.floating_value));
        break;
      case ParamConstraintScalarKind::kNone:
        break;
    }
  }

  [[nodiscard]] constexpr ParamSchemaHash finish() const noexcept {
    StableHashBuilder copy = *this;
    const std::uint64_t bit_length = copy.total_bytes_ * 8U;
    copy.buffer_[copy.buffer_size_++] = 0x80U;
    if (copy.buffer_size_ > 56U) {
      while (copy.buffer_size_ < copy.buffer_.size()) {
        copy.buffer_[copy.buffer_size_++] = 0U;
      }
      copy.transform();
      copy.buffer_size_ = 0;
    }
    while (copy.buffer_size_ < 56U) {
      copy.buffer_[copy.buffer_size_++] = 0U;
    }
    for (std::size_t index = 0; index < sizeof(bit_length); ++index) {
      copy.buffer_[63U - index] =
          static_cast<std::uint8_t>(bit_length >> static_cast<unsigned>(index * 8U));
    }
    copy.transform();

    ParamSchemaHash result;
    for (std::size_t word = 0; word < copy.state_.size(); ++word) {
      for (std::size_t byte = 0; byte < sizeof(std::uint32_t); ++byte) {
        result.value[word * sizeof(std::uint32_t) + byte] =
            static_cast<std::byte>(copy.state_[word] >> static_cast<unsigned>((3U - byte) * 8U));
      }
    }
    return result;
  }

 private:
  [[nodiscard]] static constexpr std::uint32_t rotate_right(std::uint32_t value,
                                                            unsigned count) noexcept {
    return (value >> count) | (value << (32U - count));
  }

  constexpr void transform() noexcept {
    std::array<std::uint32_t, 64> words{};
    for (std::size_t index = 0; index < 16U; ++index) {
      const std::size_t offset = index * 4U;
      words[index] = (static_cast<std::uint32_t>(buffer_[offset]) << 24U) |
                     (static_cast<std::uint32_t>(buffer_[offset + 1U]) << 16U) |
                     (static_cast<std::uint32_t>(buffer_[offset + 2U]) << 8U) |
                     static_cast<std::uint32_t>(buffer_[offset + 3U]);
    }
    for (std::size_t index = 16U; index < words.size(); ++index) {
      const std::uint32_t s0 = rotate_right(words[index - 15U], 7U) ^
                               rotate_right(words[index - 15U], 18U) ^ (words[index - 15U] >> 3U);
      const std::uint32_t s1 = rotate_right(words[index - 2U], 17U) ^
                               rotate_right(words[index - 2U], 19U) ^ (words[index - 2U] >> 10U);
      words[index] = words[index - 16U] + s0 + words[index - 7U] + s1;
    }

    std::uint32_t a = state_[0];
    std::uint32_t b = state_[1];
    std::uint32_t c = state_[2];
    std::uint32_t d = state_[3];
    std::uint32_t e = state_[4];
    std::uint32_t f = state_[5];
    std::uint32_t g = state_[6];
    std::uint32_t h = state_[7];
    for (std::size_t index = 0; index < words.size(); ++index) {
      const std::uint32_t sum1 = rotate_right(e, 6U) ^ rotate_right(e, 11U) ^ rotate_right(e, 25U);
      const std::uint32_t choose = (e & f) ^ ((~e) & g);
      const std::uint32_t temporary1 = h + sum1 + choose + constants_[index] + words[index];
      const std::uint32_t sum0 = rotate_right(a, 2U) ^ rotate_right(a, 13U) ^ rotate_right(a, 22U);
      const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
      const std::uint32_t temporary2 = sum0 + majority;
      h = g;
      g = f;
      f = e;
      e = d + temporary1;
      d = c;
      c = b;
      b = a;
      a = temporary1 + temporary2;
    }
    state_[0] += a;
    state_[1] += b;
    state_[2] += c;
    state_[3] += d;
    state_[4] += e;
    state_[5] += f;
    state_[6] += g;
    state_[7] += h;
  }

  std::array<std::uint32_t, 8> state_{
      0x6A09E667U,
      0xBB67AE85U,
      0x3C6EF372U,
      0xA54FF53AU,
      0x510E527FU,
      0x9B05688CU,
      0x1F83D9ABU,
      0x5BE0CD19U,
  };
  std::array<std::uint8_t, 64> buffer_{};
  std::size_t buffer_size_{};
  std::uint64_t total_bytes_{};
  static constexpr std::array<std::uint32_t, 64> constants_{
      0x428A2F98U, 0x71374491U, 0xB5C0FBCFU, 0xE9B5DBA5U, 0x3956C25BU, 0x59F111F1U, 0x923F82A4U,
      0xAB1C5ED5U, 0xD807AA98U, 0x12835B01U, 0x243185BEU, 0x550C7DC3U, 0x72BE5D74U, 0x80DEB1FEU,
      0x9BDC06A7U, 0xC19BF174U, 0xE49B69C1U, 0xEFBE4786U, 0x0FC19DC6U, 0x240CA1CCU, 0x2DE92C6FU,
      0x4A7484AAU, 0x5CB0A9DCU, 0x76F988DAU, 0x983E5152U, 0xA831C66DU, 0xB00327C8U, 0xBF597FC7U,
      0xC6E00BF3U, 0xD5A79147U, 0x06CA6351U, 0x14292967U, 0x27B70A85U, 0x2E1B2138U, 0x4D2C6DFCU,
      0x53380D13U, 0x650A7354U, 0x766A0ABBU, 0x81C2C92EU, 0x92722C85U, 0xA2BFE8A1U, 0xA81A664BU,
      0xC24B8B70U, 0xC76C51A3U, 0xD192E819U, 0xD6990624U, 0xF40E3585U, 0x106AA070U, 0x19A4C116U,
      0x1E376C08U, 0x2748774CU, 0x34B0BCB5U, 0x391C0CB3U, 0x4ED8AA4AU, 0x5B9CCA4FU, 0x682E6FF3U,
      0x748F82EEU, 0x78A5636FU, 0x84C87814U, 0x8CC70208U, 0x90BEFFFAU, 0xA4506CEBU, 0xBEF9A3F7U,
      0xC67178F2U,
  };
};

template <auto MemberPtr>
inline constexpr std::byte kMemberToken{};

template <typename T>
struct MemberPointerTraits;

template <typename FieldT, typename OwnerT>
struct MemberPointerTraits<FieldT OwnerT::*> {
  using Field = FieldT;
  using Owner = OwnerT;
};

}  // namespace detail

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

/// @brief Carry one authored enum label and typed value into schema lowering.
/// @tparam EnumT Exact enum type.
template <typename EnumT>
struct NamedParamEnumValue {
  std::string_view name;  ///< Stable user-facing label.
  EnumT value;            ///< Exact enum value.
};

/// @brief Create one typed enum label/value declaration for `enum_values()`.
/// @tparam EnumT Exact enum type inferred from p value.
/// @param name Stable user-facing label with static lifetime.
/// @param value Exact enum value.
/// @return Typed declaration consumed during constant evaluation.
template <typename EnumT>
  requires std::is_enum_v<EnumT>
[[nodiscard]] constexpr NamedParamEnumValue<EnumT> param_enum_value(std::string_view name,
                                                                    EnumT value) noexcept {
  return NamedParamEnumValue<EnumT>{.name = name, .value = value};
}

/// @brief Carry process-local erased staging operations for one exact declared field type.
///
/// These operations support a later compiler/runtime patch engine without placing a C++ type,
/// member pointer, callable address, or ABI layout into a portable artifact. Callers must verify
/// `local_field_type_token`, allocate correctly aligned raw staging storage, and destroy every
/// successfully constructed staging value exactly once. None of these operations is copied into
/// `ParamFieldDocument` or any schema hash.
struct ParamFieldLocalOperations {
  const void* local_field_type_token{};  ///< Exact process-local field token; never portable.
  std::size_t object_size{};             ///< Required raw staging bytes in this process.
  std::size_t object_alignment{};        ///< Required staging alignment in this process.

  /// Cold placement-copy from the member of an immutable aggregate into uninitialized storage.
  /// This operation may propagate the exact field copy constructor's exception. Such failure
  /// leaves both the aggregate and destination lifetime unchanged.
  void (*copy_construct_from_member)(void* destination, const void* aggregate);

  /// Cold placement-copy from one live exact staged field into uninitialized storage.
  /// This operation may propagate the exact field copy constructor's exception.
  void (*copy_construct_from_staged)(void* destination, const void* staged);

  /// Destroy one successfully constructed exact field value in staging storage.
  ///
  /// Schema admission requires the exact declared field type to be nothrow-destructible; an
  /// enclosing aggregate's destructor trait cannot satisfy that requirement on the field's behalf.
  void (*destroy_staged)(void* value) noexcept;

  /// Swap one live exact staging value with the aggregate member without throwing.
  ///
  /// This pointer is null when the exact field is not nothrow-swappable. On success, staging
  /// remains live and owns the prior member value, which the caller must later destroy.
  void (*replace_member_from_staged)(void* aggregate, void* staged) noexcept {};

  /// Compare the aggregate member with one live exact staging value without throwing.
  ///
  /// This pointer is null unless exact `operator==` is available and both that expression and its
  /// conversion to `bool` are `noexcept`. Equality is an optimization fact, not an update or access
  /// grant.
  bool (*semantic_equal_to_staged)(const void* aggregate, const void* staged) noexcept {};

  /// Compare the exact field members of two aggregates.
  ///
  /// This pointer is null unless exact `operator==` is available. Unlike
  /// `semantic_equal_to_staged`, the operation may propagate from the field comparison. Runtime
  /// callers must contain exceptions and preserve canonical scalar comparison where applicable.
  bool (*semantic_equal_members)(const void* left, const void* right) = nullptr;

  /// Replace one supported fixed scalar member from its DSO-safe canonical carrier.
  ///
  /// This pointer is null for const, dynamic, floating-representation duration, sequence, string,
  /// path, and object fields. Integral-representation durations use a checked canonical nanosecond
  /// carrier. A kind/width/conversion mismatch returns false without changing the aggregate.
  bool (*replace_member_from_scalar)(void* aggregate, ParamScalarValue value) noexcept {};

  /// Replace one mutable string member from borrowed bytes.
  ///
  /// Optional string fields are engaged by the same operation. This pointer is null for every
  /// other field type. Assignment may propagate an allocation exception.
  void (*replace_member_from_string)(void* aggregate, std::string_view value) = nullptr;

  /// Replace one mutable scalar sequence from canonical element carriers.
  ///
  /// This pointer is available for direct `std::vector<T>` fields when `T` is a supported fixed
  /// scalar. It is also available for optional vectors when their allocator is default
  /// constructible, so an absent value can be engaged without inventing allocator state. Staging
  /// preserves a present vector's allocator. Conversion failure leaves the member unchanged.
  /// Allocation may propagate an exception.
  bool (*replace_member_from_scalar_sequence)(void* aggregate,
                                              std::span<const ParamScalarValue> values) = nullptr;

  /// Replace one mutable string sequence from borrowed element views.
  ///
  /// This pointer is available for direct `std::vector<std::string>` fields and for optional
  /// vectors whose allocator is default constructible. Staging preserves a present vector's
  /// allocator. The operation copies every element before replacing the member and may propagate
  /// an exception.
  void (*replace_member_from_string_sequence)(void* aggregate,
                                              std::span<const std::string_view> values) = nullptr;

  /// Clear one mutable optional member without changing a non-optional field.
  ///
  /// This pointer is null unless the exact field type is `std::optional<T>`.
  void (*clear_optional_member)(void* aggregate) noexcept = nullptr;

  /// Read one supported fixed scalar member into its DSO-safe canonical carrier.
  ///
  /// This pointer is null outside the fixed scalar and integral-duration subset. An
  /// unrepresentable duration returns false; no process-local token or address escapes.
  bool (*read_member_as_scalar)(const void* aggregate, ParamScalarValue& value) noexcept {};

  /// Read one hosted fixed scalar through a runtime-assigned field index.
  ///
  /// This additive seam is null for ordinary member-pointer schemas. It lets a language adapter
  /// expose frozen values without synthesizing C++ member pointers or invoking foreign code.
  bool (*read_hosted_member_as_scalar)(const void* aggregate, std::size_t field_index,
                                       ParamScalarValue& value) noexcept {};

  /// Read one hosted bounded scalar sequence through a runtime-assigned field index.
  ///
  /// The returned view borrows immutable snapshot storage. It is null for ordinary schemas and
  /// remains process-local like every other operation in this table.
  std::span<const ParamScalarValue> (*read_hosted_member_as_scalar_sequence)(
      const void* aggregate, std::size_t field_index) noexcept {};

  /// Runtime-owned field index consumed only by the hosted read operations above.
  std::size_t hosted_field_index{};

  /// Return the nested member, engaging an optional nested value when necessary.
  ///
  /// This pointer is null unless the field is mutable and explicitly declares a nested schema.
  /// Default construction of an absent optional nested value may propagate an exception. The
  /// returned address is valid while the aggregate and its field value remain unchanged.
  void* (*mutable_nested_member)(void* aggregate) = nullptr;

  /// Return the present nested member without changing the aggregate.
  ///
  /// This pointer is null unless the field explicitly declares a nested schema. It returns null
  /// for an absent optional nested value.
  const void* (*nested_member)(const void* aggregate) noexcept {};

  /// Return the process-local descriptor for the exact nested member type.
  ///
  /// This pointer is null for fields that do not explicitly declare nesting. The descriptor and
  /// all field thunks have process lifetime and never enter a portable artifact.
  const ParamSchemaDescriptor* (*nested_schema)() noexcept {};
};

static_assert(std::is_standard_layout_v<ParamFieldLocalOperations>);
static_assert(std::is_trivially_copyable_v<ParamFieldLocalOperations>);

/// @brief Describe one declared parameter field for local cold-path Core operations.
struct ParamFieldDescriptor {
  ParamFieldId id{};                  ///< Big-endian 64-bit projection of `full_id`.
  ParamSchemaHash full_id{};          ///< Complete canonical field SHA-256 identity.
  std::string_view canonical_name{};  ///< Static-lifetime canonical structured path.
  ParamLogicalType logical_type{ParamLogicalType::kObject};  ///< Portable value family.
  ParamFieldTypeShape type_shape{};  ///< Canonical structural type and decoding shape.
  ParamMutability mutability{ParamMutability::kCompileTime};              ///< Application boundary.
  LifecycleStage earliest_invalidated_stage{LifecycleStage::kConfigure};  ///< Restart stage.
  ParamPersistence persistence{ParamPersistence::kPersisted};  ///< Durable-projection policy.
  ParamStringEncoding string_encoding{ParamStringEncoding::kArbitraryBytes};  ///< String semantics.
  ParamStringEncoding element_string_encoding{
      ParamStringEncoding::kArbitraryBytes};             ///< `vector<string>` element semantics.
  ParamVisibility visibility{ParamVisibility::kPublic};  ///< Ordinary discovery policy.
  ParamExternalAccess external_access{ParamExternalAccess::kNone};  ///< External eligibility.
  ParamSensitivity sensitivity{ParamSensitivity::kNormal};          ///< Redaction classification.
  ParamFieldPolicyEvidence policy_evidence{};  ///< Explicit-vs-inferred qualification facts.
  bool required_on_load{};  ///< Whether complete import must supply an archived value.
  bool nested{};            ///< Whether this field recursively references a declared nested schema.
  std::string_view nested_schema_id{};         ///< Nested stable ID, or empty when not nested.
  ParamSchemaVersion nested_schema_version{};  ///< Nested version, or zero when not nested.
  ParamSchemaHash nested_structural_hash{};    ///< Nested decoding-shape identity.
  ParamSchemaHash nested_contract_hash{};      ///< Nested policy identity.
  std::string_view alias{};             ///< Prior canonical input name, never output identity.
  std::string_view description{};       ///< Static descriptive UTF-8 metadata.
  std::string_view unit{};              ///< Static descriptive unit metadata.
  std::string_view validator_id{};      ///< Stable local-validator identity, or empty.
  ParamConstraints constraints{};       ///< Portable bounds and choices.
  ParamEnumChoices enum_choices{};      ///< Stable authored enum label/value domain.
  ParamStorageTraitsSummary storage{};  ///< Intrinsic field storage evidence.
  const void* local_member_token{};     ///< Process-local exact member token; never portable.
  ParamFieldLocalOperations local_operations{};  ///< Process-local staging operations.
  ParamValidationResult (*validate)(
      const void*, const ParamFieldDescriptor&) noexcept {};  ///< Local validation thunk.
  bool (*custom_validator)(const void*) noexcept {};  ///< Optional local Boolean validator thunk.
};

static_assert(std::is_standard_layout_v<ParamFieldDescriptor>);
static_assert(std::is_trivially_copyable_v<ParamFieldDescriptor>);

/// @brief Read one fixed scalar through either a static-member or hosted-field schema operation.
/// @return `true` only when the field has an exact present scalar value.
[[nodiscard]] inline bool read_param_field_scalar(const ParamFieldDescriptor& field,
                                                  const void* aggregate,
                                                  ParamScalarValue& value) noexcept {
  if (field.local_operations.read_member_as_scalar != nullptr) {
    return field.local_operations.read_member_as_scalar(aggregate, value);
  }
  return field.local_operations.read_hosted_member_as_scalar != nullptr &&
         field.local_operations.read_hosted_member_as_scalar(
             aggregate, field.local_operations.hosted_field_index, value);
}

/// @brief Read one bounded hosted scalar sequence without invoking foreign code.
/// @return Borrowed immutable values, or an empty view when the route is unavailable.
[[nodiscard]] inline std::span<const ParamScalarValue> read_param_field_scalar_sequence(
    const ParamFieldDescriptor& field, const void* aggregate) noexcept {
  if (field.local_operations.read_hosted_member_as_scalar_sequence == nullptr) {
    return {};
  }
  return field.local_operations.read_hosted_member_as_scalar_sequence(
      aggregate, field.local_operations.hosted_field_index);
}

namespace detail {

template <typename T>
concept NothrowSemanticEquality = requires(const T& left, const T& right) {
  { left == right } noexcept -> std::convertible_to<bool>;
  requires std::is_nothrow_convertible_v<decltype(left == right), bool>;
};

template <typename T>
inline constexpr bool kSupportsDirectSemanticEquality = requires(const T& left, const T& right) {
  { left == right } -> std::convertible_to<bool>;
};

template <typename T>
struct SupportsSemanticEquality : std::bool_constant<kSupportsDirectSemanticEquality<T>> {};

template <typename T>
inline constexpr bool kSupportsSemanticEquality =
    SupportsSemanticEquality<std::remove_cvref_t<T>>::value;

// Standard container equality operators are not consistently constrained by their element
// equality requirements. Inspect the supported container elements explicitly so a declaration-only
// operator== does not install a thunk whose body cannot be instantiated.
template <typename T, typename Allocator>
struct SupportsSemanticEquality<std::vector<T, Allocator>>
    : std::bool_constant<kSupportsSemanticEquality<T>> {};

template <typename T, typename Allocator>
struct SupportsSemanticEquality<std::deque<T, Allocator>>
    : std::bool_constant<kSupportsSemanticEquality<T>> {};

template <typename T, typename Allocator>
struct SupportsSemanticEquality<std::list<T, Allocator>>
    : std::bool_constant<kSupportsSemanticEquality<T>> {};

template <typename T, typename Allocator>
struct SupportsSemanticEquality<std::forward_list<T, Allocator>>
    : std::bool_constant<kSupportsSemanticEquality<T>> {};

template <typename T, std::size_t Size>
struct SupportsSemanticEquality<std::array<T, Size>>
    : std::bool_constant<kSupportsSemanticEquality<T>> {};

template <typename T>
struct SupportsSemanticEquality<std::optional<T>>
    : std::bool_constant<kSupportsSemanticEquality<T>> {};

template <typename Key, typename Value, typename Compare, typename Allocator>
struct SupportsSemanticEquality<std::map<Key, Value, Compare, Allocator>>
    : std::bool_constant<kSupportsSemanticEquality<Key> && kSupportsSemanticEquality<Value>> {};

template <typename Key, typename Value, typename Compare, typename Allocator>
struct SupportsSemanticEquality<std::multimap<Key, Value, Compare, Allocator>>
    : std::bool_constant<kSupportsSemanticEquality<Key> && kSupportsSemanticEquality<Value>> {};

template <typename Key, typename Value, typename Hash, typename KeyEqual, typename Allocator>
struct SupportsSemanticEquality<std::unordered_map<Key, Value, Hash, KeyEqual, Allocator>>
    : std::bool_constant<kSupportsSemanticEquality<Key> && kSupportsSemanticEquality<Value>> {};

template <typename Key, typename Value, typename Hash, typename KeyEqual, typename Allocator>
struct SupportsSemanticEquality<std::unordered_multimap<Key, Value, Hash, KeyEqual, Allocator>>
    : std::bool_constant<kSupportsSemanticEquality<Key> && kSupportsSemanticEquality<Value>> {};

template <typename Key, typename Compare, typename Allocator>
struct SupportsSemanticEquality<std::set<Key, Compare, Allocator>>
    : std::bool_constant<kSupportsSemanticEquality<Key>> {};

template <typename Key, typename Compare, typename Allocator>
struct SupportsSemanticEquality<std::multiset<Key, Compare, Allocator>>
    : std::bool_constant<kSupportsSemanticEquality<Key>> {};

template <typename Key, typename Hash, typename KeyEqual, typename Allocator>
struct SupportsSemanticEquality<std::unordered_set<Key, Hash, KeyEqual, Allocator>>
    : std::bool_constant<kSupportsSemanticEquality<Key>> {};

template <typename Key, typename Hash, typename KeyEqual, typename Allocator>
struct SupportsSemanticEquality<std::unordered_multiset<Key, Hash, KeyEqual, Allocator>>
    : std::bool_constant<kSupportsSemanticEquality<Key>> {};

template <typename T, typename Container>
struct SupportsSemanticEquality<std::queue<T, Container>>
    : std::bool_constant<kSupportsSemanticEquality<Container>> {};

template <typename T, typename Container>
struct SupportsSemanticEquality<std::stack<T, Container>>
    : std::bool_constant<kSupportsSemanticEquality<Container>> {};

template <typename First, typename Second>
struct SupportsSemanticEquality<std::pair<First, Second>>
    : std::bool_constant<kSupportsSemanticEquality<First> && kSupportsSemanticEquality<Second>> {};

template <typename... Elements>
struct SupportsSemanticEquality<std::tuple<Elements...>>
    : std::bool_constant<(kSupportsSemanticEquality<Elements> && ...)> {};

template <typename... Alternatives>
struct SupportsSemanticEquality<std::variant<Alternatives...>>
    : std::bool_constant<(kSupportsSemanticEquality<Alternatives> && ...)> {};

template <typename T>
concept SemanticEquality = kSupportsSemanticEquality<T>;

template <typename ParamsT, auto MemberPtr>
void copy_construct_field_from_member(void* destination, const void* aggregate) {
  using FieldT = typename MemberPointerTraits<decltype(MemberPtr)>::Field;
  ::new (destination) FieldT(static_cast<const ParamsT*>(aggregate)->*MemberPtr);
}

template <typename ParamsT, auto MemberPtr>
void copy_construct_staged_field(void* destination, const void* staged) {
  using FieldT = typename MemberPointerTraits<decltype(MemberPtr)>::Field;
  ::new (destination) FieldT(*static_cast<const FieldT*>(staged));
}

template <typename ParamsT, auto MemberPtr>
void destroy_staged_field(void* value) noexcept {
  using FieldT = typename MemberPointerTraits<decltype(MemberPtr)>::Field;
  static_assert(std::is_nothrow_destructible_v<FieldT>);
  static_cast<FieldT*>(value)->~FieldT();
}

template <typename ParamsT, auto MemberPtr>
void replace_field_from_staged(void* aggregate, void* staged) noexcept {
  using FieldT = typename MemberPointerTraits<decltype(MemberPtr)>::Field;
  static_assert(!std::is_const_v<FieldT> && std::is_nothrow_swappable_v<FieldT>);
  using std::swap;
  swap(static_cast<ParamsT*>(aggregate)->*MemberPtr, *static_cast<FieldT*>(staged));
}

template <typename ParamsT, auto MemberPtr>
bool field_semantically_equals_staged(const void* aggregate, const void* staged) noexcept {
  using FieldT = typename MemberPointerTraits<decltype(MemberPtr)>::Field;
  static_assert(NothrowSemanticEquality<FieldT>);
  return static_cast<const ParamsT*>(aggregate)->*MemberPtr == *static_cast<const FieldT*>(staged);
}

template <typename ParamsT, auto MemberPtr>
bool field_members_semantically_equal(const void* left, const void* right) {
  using FieldT = typename MemberPointerTraits<decltype(MemberPtr)>::Field;
  static_assert(SemanticEquality<FieldT>);
  return static_cast<const ParamsT*>(left)->*MemberPtr ==
         static_cast<const ParamsT*>(right)->*MemberPtr;
}

template <typename ParamsT, auto MemberPtr>
bool replace_field_from_scalar(void* aggregate, ParamScalarValue value) noexcept {
  using FieldT = typename MemberPointerTraits<decltype(MemberPtr)>::Field;
  using ValueT = std::remove_cv_t<FieldT>;
  using TargetT = UnwrappedOptionalT<ValueT>;
  static_assert(!std::is_const_v<FieldT> && kFixedParameterScalar<TargetT>);
  TargetT decoded{};
  if (!value.try_as(decoded)) {
    return false;
  }
  static_cast<ParamsT*>(aggregate)->*MemberPtr = decoded;
  return true;
}

template <typename ParamsT, auto MemberPtr>
bool read_field_as_scalar(const void* aggregate, ParamScalarValue& value) noexcept {
  using FieldT = typename MemberPointerTraits<decltype(MemberPtr)>::Field;
  using ValueT = std::remove_cv_t<FieldT>;
  using TargetT = UnwrappedOptionalT<ValueT>;
  static_assert(kFixedParameterScalar<TargetT>);
  const auto& field = static_cast<const ParamsT*>(aggregate)->*MemberPtr;
  if constexpr (IsStdOptional<ValueT>::value) {
    if (!field.has_value()) {
      return false;
    }
    value = ParamScalarValue::from(*field);
  } else {
    value = ParamScalarValue::from(field);
  }
  return value.kind != ParamScalarKind::kNone;
}

template <typename ParamsT, auto MemberPtr>
void replace_field_from_string(void* aggregate, std::string_view value) {
  using FieldT = typename MemberPointerTraits<decltype(MemberPtr)>::Field;
  using ValueT = std::remove_cv_t<FieldT>;
  using AssignmentT = UnwrappedOptionalT<ValueT>;
  static_assert((std::same_as<AssignmentT, std::string> ||
                 std::same_as<AssignmentT, std::filesystem::path>) &&
                !std::is_const_v<FieldT>);
  if constexpr (std::same_as<AssignmentT, std::string>) {
    static_cast<ParamsT*>(aggregate)->*MemberPtr = value;
  } else {
    static_cast<ParamsT*>(aggregate)->*MemberPtr = std::filesystem::path{value};
  }
}

template <typename SequenceT, typename MemberT>
SequenceT make_allocator_preserving_sequence(MemberT& member) {
  static_assert(IsStdVector<SequenceT>::value);
  using UnqualifiedMemberT = std::remove_cvref_t<MemberT>;
  if constexpr (IsStdOptional<UnqualifiedMemberT>::value) {
    using AllocatorT = typename SequenceT::allocator_type;
    static_assert(std::default_initializable<AllocatorT>);
    if (member.has_value()) {
      return SequenceT(member->get_allocator());
    }
    return SequenceT(AllocatorT{});
  } else {
    return SequenceT(member.get_allocator());
  }
}

template <typename ParamsT, auto MemberPtr>
bool replace_field_from_scalar_sequence(void* aggregate, std::span<const ParamScalarValue> values) {
  using FieldT = typename MemberPointerTraits<decltype(MemberPtr)>::Field;
  using ValueT = std::remove_cv_t<FieldT>;
  using SequenceT = UnwrappedOptionalT<ValueT>;
  using ElementT = typename StdVectorElement<SequenceT>::Type;
  static_assert(IsStdVector<SequenceT>::value && kFixedParameterScalar<ElementT> &&
                !std::is_const_v<FieldT>);
  auto& member = static_cast<ParamsT*>(aggregate)->*MemberPtr;
  SequenceT decoded = make_allocator_preserving_sequence<SequenceT>(member);
  decoded.reserve(values.size());
  for (const ParamScalarValue value : values) {
    ElementT element{};
    if (!value.try_as(element)) {
      return false;
    }
    decoded.push_back(element);
  }
  member = std::move(decoded);
  return true;
}

template <typename ParamsT, auto MemberPtr>
void replace_field_from_string_sequence(void* aggregate, std::span<const std::string_view> values) {
  using FieldT = typename MemberPointerTraits<decltype(MemberPtr)>::Field;
  using ValueT = std::remove_cv_t<FieldT>;
  using SequenceT = UnwrappedOptionalT<ValueT>;
  static_assert(IsStringVector<SequenceT>::value && !std::is_const_v<FieldT>);
  auto& member = static_cast<ParamsT*>(aggregate)->*MemberPtr;
  SequenceT decoded = make_allocator_preserving_sequence<SequenceT>(member);
  decoded.reserve(values.size());
  for (const std::string_view value : values) {
    decoded.emplace_back(value);
  }
  member = std::move(decoded);
}

template <typename ParamsT, auto MemberPtr>
void clear_optional_field(void* aggregate) noexcept {
  using FieldT = typename MemberPointerTraits<decltype(MemberPtr)>::Field;
  using ValueT = std::remove_cv_t<FieldT>;
  static_assert(IsStdOptional<ValueT>::value && !std::is_const_v<FieldT>);
  (static_cast<ParamsT*>(aggregate)->*MemberPtr).reset();
}

template <typename ParamsT, auto MemberPtr>
void* mutable_nested_field(void* aggregate) {
  using FieldT = typename MemberPointerTraits<decltype(MemberPtr)>::Field;
  using ValueT = std::remove_cv_t<FieldT>;
  auto& field = static_cast<ParamsT*>(aggregate)->*MemberPtr;
  if constexpr (IsStdOptional<ValueT>::value) {
    if (!field.has_value()) {
      field.emplace();
    }
    return &*field;
  } else {
    return &field;
  }
}

template <typename ParamsT, auto MemberPtr>
const void* nested_field(const void* aggregate) noexcept {
  using FieldT = typename MemberPointerTraits<decltype(MemberPtr)>::Field;
  using ValueT = std::remove_cv_t<FieldT>;
  const auto& field = static_cast<const ParamsT*>(aggregate)->*MemberPtr;
  if constexpr (IsStdOptional<ValueT>::value) {
    return field.has_value() ? &*field : nullptr;
  } else {
    return &field;
  }
}

template <typename ParamsT, auto MemberPtr>
const ParamSchemaDescriptor* nested_field_schema() noexcept {
  using FieldT = typename MemberPointerTraits<decltype(MemberPtr)>::Field;
  return param_schema_descriptor<UnwrappedOptionalT<FieldT>>();
}

template <typename ParamsT, auto MemberPtr>
// This constexpr table builder keeps independent capabilities visible together; splitting its
// compile-time branches would obscure which operations are installed without reducing runtime work.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
constexpr ParamFieldLocalOperations make_field_local_operations(bool nested) noexcept {
  using FieldT = typename MemberPointerTraits<decltype(MemberPtr)>::Field;
  using UnqualifiedFieldT = std::remove_cv_t<FieldT>;
  static_assert(std::is_copy_constructible_v<FieldT>,
                "declared parameter fields must be copy-constructible for staged updates");
  static_assert(std::is_nothrow_destructible_v<FieldT>,
                "declared parameter fields must be nothrow-destructible for staged updates");

  ParamFieldLocalOperations operations{
      .local_field_type_token = &kParameterTypeToken<UnqualifiedFieldT>,
      .object_size = sizeof(FieldT),
      .object_alignment = alignof(FieldT),
      .copy_construct_from_member = &copy_construct_field_from_member<ParamsT, MemberPtr>,
      .copy_construct_from_staged = &copy_construct_staged_field<ParamsT, MemberPtr>,
      .destroy_staged = &destroy_staged_field<ParamsT, MemberPtr>,
  };
  if constexpr (!std::is_const_v<FieldT> && std::is_nothrow_swappable_v<FieldT>) {
    operations.replace_member_from_staged = &replace_field_from_staged<ParamsT, MemberPtr>;
  }
  if constexpr (NothrowSemanticEquality<FieldT>) {
    operations.semantic_equal_to_staged = &field_semantically_equals_staged<ParamsT, MemberPtr>;
  }
  if constexpr (SemanticEquality<FieldT>) {
    operations.semantic_equal_members = &field_members_semantically_equal<ParamsT, MemberPtr>;
  }
  using AssignmentT = UnwrappedOptionalT<UnqualifiedFieldT>;
  if constexpr (kFixedParameterScalar<AssignmentT>) {
    operations.read_member_as_scalar = &read_field_as_scalar<ParamsT, MemberPtr>;
    if constexpr (!std::is_const_v<FieldT>) {
      operations.replace_member_from_scalar = &replace_field_from_scalar<ParamsT, MemberPtr>;
    }
  }
  if constexpr ((std::same_as<AssignmentT, std::string> ||
                 std::same_as<AssignmentT, std::filesystem::path>) &&
                !std::is_const_v<FieldT>) {
    operations.replace_member_from_string = &replace_field_from_string<ParamsT, MemberPtr>;
  }
  if constexpr (IsStdVector<AssignmentT>::value && !std::is_const_v<FieldT>) {
    using ElementT = typename StdVectorElement<AssignmentT>::Type;
    using AllocatorT = typename AssignmentT::allocator_type;
    constexpr bool kCanEngageOptional =
        !IsStdOptional<UnqualifiedFieldT>::value || std::default_initializable<AllocatorT>;
    if constexpr (kCanEngageOptional) {
      if constexpr (kFixedParameterScalar<ElementT>) {
        operations.replace_member_from_scalar_sequence =
            &replace_field_from_scalar_sequence<ParamsT, MemberPtr>;
      } else if constexpr (std::same_as<ElementT, std::string>) {
        operations.replace_member_from_string_sequence =
            &replace_field_from_string_sequence<ParamsT, MemberPtr>;
      }
    }
  }
  if constexpr (IsStdOptional<UnqualifiedFieldT>::value && !std::is_const_v<FieldT>) {
    operations.clear_optional_member = &clear_optional_field<ParamsT, MemberPtr>;
  }
  using NestedT = UnwrappedOptionalT<UnqualifiedFieldT>;
  if constexpr (HasParamSchema<NestedT>) {
    if (nested) {
      operations.nested_member = &nested_field<ParamsT, MemberPtr>;
      operations.nested_schema = &nested_field_schema<ParamsT, MemberPtr>;
      if constexpr (!std::is_const_v<FieldT>) {
        operations.mutable_nested_member = &mutable_nested_field<ParamsT, MemberPtr>;
      }
    }
  }
  return operations;
}

template <typename ParamsT, auto MemberPtr>
constexpr ParamFieldDescriptor make_field_descriptor(
    std::string_view schema_id, const ParamFieldBuilder<ParamsT, MemberPtr>& builder);

}  // namespace detail

/// @brief Describe one field without carrying process-local member or function pointers.
struct ParamFieldDocument {
  ParamFieldId id{};                  ///< Big-endian 64-bit projection of `full_id`.
  ParamSchemaHash full_id{};          ///< Complete canonical field SHA-256 identity.
  std::string_view canonical_name{};  ///< Static canonical structured path.
  ParamLogicalType logical_type{ParamLogicalType::kObject};  ///< Portable value family.
  ParamFieldTypeShape type_shape{};  ///< Canonical structural type and decoding shape.
  ParamMutability mutability{ParamMutability::kCompileTime};              ///< Application boundary.
  LifecycleStage earliest_invalidated_stage{LifecycleStage::kConfigure};  ///< Restart stage.
  ParamPersistence persistence{ParamPersistence::kPersisted};  ///< Durable-projection policy.
  ParamStringEncoding string_encoding{ParamStringEncoding::kArbitraryBytes};  ///< String semantics.
  ParamStringEncoding element_string_encoding{
      ParamStringEncoding::kArbitraryBytes};             ///< `vector<string>` element semantics.
  ParamVisibility visibility{ParamVisibility::kPublic};  ///< Ordinary discovery policy.
  ParamExternalAccess external_access{ParamExternalAccess::kNone};  ///< External eligibility.
  ParamSensitivity sensitivity{ParamSensitivity::kNormal};          ///< Redaction classification.
  ParamFieldPolicyEvidence policy_evidence{};  ///< Explicit-vs-inferred qualification facts.
  bool required_on_load{};              ///< Whether complete import requires an archived value.
  bool nested{};                        ///< Whether nested schema identity is present.
  std::string_view nested_schema_id{};  ///< Nested stable ID, or empty.
  ParamSchemaVersion nested_schema_version{};  ///< Nested version, or zero.
  ParamSchemaHash nested_structural_hash{};    ///< Nested decoding-shape identity.
  ParamSchemaHash nested_contract_hash{};      ///< Nested policy identity.
  std::string_view alias{};                    ///< Prior canonical migration input name.
  std::string_view description{};              ///< Descriptive UTF-8 metadata.
  std::string_view unit{};                     ///< Descriptive unit metadata.
  std::string_view validator_id{};             ///< Stable validator identity without a callable.
  ParamConstraints constraints{};              ///< Portable bounds and choices.
  ParamEnumChoices enum_choices{};             ///< Stable authored enum label/value domain.
  ParamStorageTraitsSummary storage{};         ///< Intrinsic field storage evidence.
};

/// @brief Supply compile-selected facts needed by ordinary field projection classification.
///
/// These facts are already-authorized policy and codec dispositions. The classifier neither
/// discovers a codec nor decides profile admission, and it never receives a parameter value.
struct ParamProjectionContext {
  bool profile_allows{};   ///< Whether the effective profile permits this field.
  bool codec_available{};  ///< Whether an admitted portable codec is available.
};

/// @brief Classify one field before any parameter value is read or encoded.
/// @param field Operation-free logical field metadata.
/// @param context Compile-selected profile and codec dispositions.
/// @return Exactly one outcome using transient, sensitive, internal, profile, then codec
/// precedence.
/// @note This pure metadata operation does not inspect, borrow, log, allocate, encode, or mutate
/// a parameter value. A sensitive field is redacted before internal/profile/codec evaluation so
/// tooling can report the redaction without probing the value.
[[nodiscard]] constexpr params::ParamProjectionStatus classify_param_projection(
    const ParamFieldDocument& field, ParamProjectionContext context) noexcept {
  if (field.persistence == ParamPersistence::kTransient) {
    return params::ParamProjectionStatus::kExcludedTransient;
  }
  if (field.sensitivity == ParamSensitivity::kSensitive) {
    return params::ParamProjectionStatus::kRedactedSensitive;
  }
  if (field.visibility == ParamVisibility::kInternal) {
    return params::ParamProjectionStatus::kExcludedInternal;
  }
  if (!context.profile_allows) {
    return params::ParamProjectionStatus::kExcludedProfilePolicy;
  }
  if (!context.codec_available) {
    return params::ParamProjectionStatus::kExcludedUnsupportedCodec;
  }
  return params::ParamProjectionStatus::kIncluded;
}

/// @brief Describe one immutable process-local erased parameter schema.
struct ParamSchemaDescriptor {
  std::string_view stable_id{};       ///< Static stable schema identity.
  ParamSchemaVersion version{};       ///< Nonzero author-declared schema version.
  std::string_view description{};     ///< Static descriptive UTF-8 metadata.
  ParamSchemaHash structural_hash{};  ///< Decoding-shape compatibility fingerprint.
  ParamSchemaHash contract_hash{};    ///< Enforced-policy compatibility fingerprint.
  ParamSchemaHash metadata_hash{};    ///< Contract-plus-description fingerprint.
  ParamSchemaCoverage coverage{ParamSchemaCoverage::kComplete};  ///< Declared coverage.
  ParamSchemaProof proof{ParamSchemaProof::kAuthorAsserted};     ///< Separate proof evidence.
  const ParamTypeDescriptor* local_type{};         ///< Process-local aggregate operations.
  std::span<const ParamFieldDescriptor> fields{};  ///< Static declaration-order fields.
};

template <typename ParamsT>
  requires ParameterAggregate<ParamsT>
[[nodiscard]] const ParamSchemaDescriptor* param_schema_descriptor() noexcept;

template <typename ParamsT>
  requires ParameterAggregate<ParamsT>
[[nodiscard]] ParamValidationResult validate_params(const ParamsT& params) noexcept;

/// @brief Provide a dependency-free portable projection boundary for optional tooling.
///
/// This static view contains no C++ member pointer, operation thunk, object address, parser node,
/// file path, backend handle, or mutable runtime authority. HSDK5-018 owns owning archive and
/// persistence representations.
struct ParamSchemaDocument {
  std::string_view stable_id{};       ///< Static stable schema identity.
  ParamSchemaVersion version{};       ///< Nonzero schema version.
  std::string_view description{};     ///< Static descriptive UTF-8 metadata.
  ParamSchemaHash structural_hash{};  ///< Decoding-shape fingerprint.
  ParamSchemaHash contract_hash{};    ///< Enforced-policy fingerprint.
  ParamSchemaHash metadata_hash{};    ///< Contract-plus-description fingerprint.
  ParamSchemaCoverage coverage{ParamSchemaCoverage::kComplete};  ///< Declared coverage.
  ParamSchemaProof proof{ParamSchemaProof::kAuthorAsserted};     ///< Separate proof evidence.
  std::span<const ParamFieldDocument> fields{};  ///< Static logical field projections.
};

/// @brief Build the required leading schema clause for `HOLOSCAN_PARAM_SCHEMA`.
class ParamSchemaHeader {
 public:
  /// @brief Create a complete author-asserted schema header.
  /// @param stable_id Nonempty valid UTF-8 identity with static lifetime.
  /// @param version Nonzero schema version.
  constexpr ParamSchemaHeader(std::string_view stable_id, ParamSchemaVersion version) noexcept
      : stable_id_(stable_id), version_(version) {}

  /// @brief Select declared-fields-only coverage.
  /// @return This header for fluent declaration.
  constexpr ParamSchemaHeader& partial() noexcept {
    coverage_ = ParamSchemaCoverage::kDeclaredFieldsOnly;
    return *this;
  }

  /// @brief Attach descriptive schema metadata.
  /// @param text Valid UTF-8 text with static lifetime.
  /// @return This header for fluent declaration.
  constexpr ParamSchemaHeader& description(std::string_view text) noexcept {
    description_ = text;
    return *this;
  }

  /// @brief Return the declared stable schema identity.
  /// @return Static-lifetime UTF-8 identity.
  [[nodiscard]] constexpr std::string_view stable_id() const noexcept { return stable_id_; }
  /// @brief Return the nonzero declared schema version.
  /// @return Schema version supplied to `param_schema()`.
  [[nodiscard]] constexpr ParamSchemaVersion version() const noexcept { return version_; }
  /// @brief Return partial or complete declared coverage.
  /// @return Current coverage selection.
  [[nodiscard]] constexpr ParamSchemaCoverage coverage() const noexcept { return coverage_; }
  /// @brief Return separate qualified-inventory proof evidence.
  /// @return Author-asserted for ordinary public declarations.
  [[nodiscard]] constexpr ParamSchemaProof proof() const noexcept { return proof_; }
  /// @brief Return descriptive schema metadata.
  /// @return Static-lifetime UTF-8 text, or empty.
  [[nodiscard]] constexpr std::string_view description_text() const noexcept {
    return description_;
  }

 private:
  std::string_view stable_id_{};
  ParamSchemaVersion version_{};
  ParamSchemaCoverage coverage_{ParamSchemaCoverage::kComplete};
  ParamSchemaProof proof_{ParamSchemaProof::kAuthorAsserted};
  std::string_view description_{};

  friend class detail::ParamSchemaTestAccess;
};

/// @brief Create a complete author-asserted schema header.
/// @param stable_id Nonempty valid UTF-8 identity with static lifetime.
/// @param version Nonzero schema version.
/// @return Fluent header consumed as the first `HOLOSCAN_PARAM_SCHEMA` clause.
[[nodiscard]] constexpr ParamSchemaHeader param_schema(std::string_view stable_id,
                                                       ParamSchemaVersion version) noexcept {
  return ParamSchemaHeader{stable_id, version};
}

template <typename ParamsT, auto MemberPtr>
/// @brief Build one compile-time field declaration for an exact aggregate member.
/// @tparam ParamsT Exact aggregate owning `MemberPtr`.
/// @tparam MemberPtr Exact non-static data-member pointer.
///
/// The builder is a transient constant-evaluation value. It registers nothing, stores no object
/// address, and is not thread-shared at runtime. Invalid combinations make schema lowering fail.
class ParamFieldBuilder {
 public:
  /// @brief Expose member-pointer traits for compile-time schema lowering.
  using Traits = detail::MemberPointerTraits<decltype(MemberPtr)>;
  /// @brief Name the exact declared field type.
  using Field = typename Traits::Field;
  /// @brief Name the exact owning aggregate type.
  using Owner = typename Traits::Owner;

  /// @brief Create a field declaration with default compile-time/public/persisted policies.
  /// @param canonical_name Nonempty valid UTF-8 structured path with static lifetime.
  constexpr explicit ParamFieldBuilder(std::string_view canonical_name) noexcept
      : canonical_name_(canonical_name) {}

  /// @brief Attach descriptive field metadata.
  /// @param text Valid UTF-8 text with static lifetime.
  /// @return This builder.
  constexpr ParamFieldBuilder& description(std::string_view text) noexcept {
    description_ = text;
    return *this;
  }
  /// @brief Attach a descriptive unit name.
  /// @param text Valid UTF-8 unit text with static lifetime.
  /// @return This builder.
  constexpr ParamFieldBuilder& unit(std::string_view text) noexcept {
    unit_ = text;
    return *this;
  }
  /// @brief Require a newly compiled plan for changes after compile.
  /// @return This builder.
  constexpr ParamFieldBuilder& compile_time() noexcept {
    mutability_ = ParamMutability::kCompileTime;
    reset_lifecycle_state();
    return *this;
  }
  /// @brief Declare eligibility for separately admitted immutable generation updates.
  /// @return This builder; the marker grants no update authority by itself.
  constexpr ParamFieldBuilder& runtime_mutable() noexcept {
    mutability_ = ParamMutability::kRuntime;
    reset_lifecycle_state();
    return *this;
  }
  /// @brief Declare a lifecycle transaction from the earliest invalidated forward stage.
  /// @param stage Forward stage from `kConfigure` through `kStart`.
  /// @return This builder; invalid teardown or unknown stages fail schema lowering.
  constexpr ParamFieldBuilder& lifecycle_mutable(
      LifecycleStage stage = LifecycleStage::kConfigure) noexcept {
    mutability_ = ParamMutability::kLifecycle;
    earliest_invalidated_stage_ = stage;
    invalid_lifecycle_stage_ = !is_lifecycle_forward_stage(stage);
    return *this;
  }
  /// @brief Require a newly compiled plan for every value change.
  /// @return This builder.
  constexpr ParamFieldBuilder& recompile_required() noexcept {
    mutability_ = ParamMutability::kRecompile;
    reset_lifecycle_state();
    return *this;
  }

  template <typename Minimum, typename Maximum>
  /// @brief Add an inclusive finite range in the field's exact scalar domain.
  /// @tparam Minimum Exact field type, or a duration convertible to the field's duration domain.
  /// @tparam Maximum Exact field type, or a duration convertible to the field's duration domain.
  /// @param minimum Inclusive finite lower bound; duration conversion must be representable, with
  /// integral fields requiring an exact tick.
  /// @param maximum Inclusive finite upper bound; duration conversion must be representable, with
  /// integral fields requiring an exact tick.
  /// @return This builder; reversed, non-finite, or mismatched bounds fail lowering.
  constexpr ParamFieldBuilder& range(Minimum minimum, Maximum maximum) {
    static_assert(detail::kIsParamScalar<Field>,
                  "parameter range requires an arithmetic, enum, or duration field");
    constraints_.has_range = true;
    using FieldValue = detail::UnwrappedOptionalT<Field>;
    using MinimumValue = std::remove_cvref_t<Minimum>;
    using MaximumValue = std::remove_cvref_t<Maximum>;
    if constexpr (detail::kIsChronoDuration<FieldValue>) {
      static_assert(
          detail::kIsChronoDuration<MinimumValue> && detail::kIsChronoDuration<MaximumValue>,
          "parameter duration range bounds must be durations");
      if constexpr (detail::kIsChronoDuration<MinimumValue> &&
                    detail::kIsChronoDuration<MaximumValue>) {
        const detail::DurationScalarConversion converted_minimum =
            detail::duration_scalar_in_field_domain<FieldValue>(minimum);
        const detail::DurationScalarConversion converted_maximum =
            detail::duration_scalar_in_field_domain<FieldValue>(maximum);
        constraints_.range_min = converted_minimum.scalar;
        constraints_.range_max = converted_maximum.scalar;
        invalid_range_ = !converted_minimum.valid || !converted_maximum.valid;
      }
    } else {
      static_assert(
          std::same_as<MinimumValue, FieldValue> && std::same_as<MaximumValue, FieldValue>,
          "parameter range type must exactly match the declared field type");
      if constexpr (std::same_as<MinimumValue, FieldValue> &&
                    std::same_as<MaximumValue, FieldValue>) {
        constraints_.range_min = detail::to_scalar(minimum);
        constraints_.range_max = detail::to_scalar(maximum);
      }
    }
    invalid_range_ = !detail::scalar_finite(constraints_.range_min) || invalid_range_ ||
                     !detail::scalar_finite(constraints_.range_max) ||
                     detail::scalar_less(constraints_.range_max, constraints_.range_min);
    return *this;
  }

  template <typename T>
  /// @brief Add a fixed-capacity allowed-value set in the exact scalar domain.
  /// @tparam T Exact field type.
  /// @param values Up to 16 finite values copied during constant evaluation.
  /// @return This builder; overflow, non-finite, or mismatched values fail lowering.
  constexpr ParamFieldBuilder& allowed_values(std::initializer_list<T> values) {
    static_assert(std::same_as<std::remove_cvref_t<T>, detail::UnwrappedOptionalT<Field>>,
                  "parameter allowed-value type must exactly match the declared field type");
    static_assert(detail::kIsParamScalar<Field>,
                  "parameter allowed values require an arithmetic, enum, or duration field");
    const std::size_t active_count = constraints_.allowed_value_count;
    if (values.size() > constraints_.allowed_values.size() - active_count) {
      too_many_allowed_values_ = true;
      return *this;
    }
    for (T value : values) {
      const ParamConstraintScalar scalar = detail::to_scalar(value);
      invalid_allowed_value_ = invalid_allowed_value_ || !detail::scalar_finite(scalar);
      constraints_.allowed_values[constraints_.allowed_value_count++] = scalar;
    }
    return *this;
  }

  template <typename EnumT>
  /// @brief Declare the bounded stable label/value domain of an enum field.
  /// @tparam EnumT Exact direct, optional, or sequence-element enum type.
  /// @param values Up to 16 unique labels and values copied during constant evaluation.
  /// @return This builder; the declared domain is also the allowed-value constraint.
  constexpr ParamFieldBuilder& enum_values(
      std::initializer_list<NamedParamEnumValue<EnumT>> values) {
    static_assert(std::is_enum_v<EnumT>, "enum values require an enum type");
    static_assert(std::same_as<EnumT, detail::EnumDomainT<Field>>,
                  "enum value type must exactly match the field enum domain");
    if (values.size() == 0U) {
      invalid_enum_choice_ = true;
      return *this;
    }
    if (values.size() > enum_choices_.values.size() - enum_choices_.count ||
        values.size() > constraints_.allowed_values.size() - constraints_.allowed_value_count) {
      too_many_allowed_values_ = true;
      return *this;
    }
    for (const NamedParamEnumValue<EnumT>& named : values) {
      const ParamScalarValue scalar = ParamScalarValue::from(named.value);
      bool duplicate = false;
      for (std::size_t index = 0; index < enum_choices_.count; ++index) {
        duplicate = duplicate || enum_choices_.values[index].name == named.name ||
                    enum_choices_.values[index].value.raw_bits == scalar.raw_bits;
      }
      invalid_enum_choice_ = invalid_enum_choice_ || !detail::valid_schema_text(named.name) ||
                             scalar.kind == ParamScalarKind::kNone || duplicate;
      enum_choices_.values[enum_choices_.count++] =
          ParamEnumChoice{.name = named.name, .value = scalar};
      constraints_.allowed_values[constraints_.allowed_value_count++] =
          detail::to_scalar(named.value);
    }
    return *this;
  }

  template <auto Validator>
  /// @brief Attach a named process-local side-effect-free Boolean validator.
  /// @tparam Validator `noexcept bool(const Field&)` callable known at compile time.
  /// @param stable_validator_id Nonempty stable UTF-8 identity exported without the callable.
  /// @return This builder.
  constexpr ParamFieldBuilder& validator(std::string_view stable_validator_id) noexcept {
    using Result = decltype(Validator(std::declval<const Field&>()));
    static_assert(std::same_as<Result, bool>, "parameter validator must return bool");
    static_assert(noexcept(Validator(std::declval<const Field&>())),
                  "parameter validator must be noexcept");
    validator_id_ = stable_validator_id;
    custom_validator_ = &run_custom_validator<Validator>;
    return *this;
  }

  /// @brief Select the durable-projection policy explicitly.
  /// @param policy Persisted or transient disposition to record as authored evidence.
  /// @return This builder; conflicting repeated selections fail lowering.
  constexpr ParamFieldBuilder& persistence(ParamPersistence policy) noexcept {
    persistence_conflict_ =
        persistence_conflict_ || (persistence_explicit_ && persistence_ != policy);
    persistence_explicit_ = true;
    persistence_ = policy;
    return *this;
  }

  /// @brief Mark the field eligible for policy-filtered durable projection.
  /// @return This builder; conflicting with `transient()` fails lowering.
  constexpr ParamFieldBuilder& persisted() noexcept {
    return persistence(ParamPersistence::kPersisted);
  }

  /// @brief Exclude the field from every ordinary durable projection.
  /// @return This builder; conflicting with `persisted()` fails lowering.
  constexpr ParamFieldBuilder& transient() noexcept {
    return persistence(ParamPersistence::kTransient);
  }
  /// @brief Hide the field from ordinary local discovery.
  /// @return This builder; internality does not imply sensitivity.
  constexpr ParamFieldBuilder& internal() noexcept {
    visibility_ = ParamVisibility::kInternal;
    return *this;
  }
  /// @brief Select the external-access policy explicitly.
  /// @param policy None, read-only, or tuning eligibility; this grants no authority or codec.
  /// @return This builder; conflicting repeated selections fail lowering.
  constexpr ParamFieldBuilder& external_access(ParamExternalAccess policy) noexcept {
    external_access_conflict_ =
        external_access_conflict_ || (external_access_explicit_ && external_access_ != policy);
    external_access_ = policy;
    external_access_explicit_ = true;
    return *this;
  }

  /// @brief Declare eligibility for separately authorized external reads.
  /// @return This builder; no authority or codec is granted.
  constexpr ParamFieldBuilder& external_read_only() noexcept {
    return external_access(ParamExternalAccess::kReadOnly);
  }

  /// @brief Declare eligibility for separately authorized external tuning.
  /// @return This builder; runtime mutability and non-sensitive public visibility are required.
  constexpr ParamFieldBuilder& external_tuning() noexcept {
    return external_access(ParamExternalAccess::kTuning);
  }

  /// @brief Select the sensitivity/redaction policy explicitly.
  /// @param policy Normal or sensitive disposition to record as authored evidence.
  /// @return This builder; conflicting repeated selections fail lowering.
  constexpr ParamFieldBuilder& sensitivity(ParamSensitivity policy) noexcept {
    sensitivity_conflict_ =
        sensitivity_conflict_ || (sensitivity_explicit_ && sensitivity_ != policy);
    sensitivity_ = policy;
    sensitivity_explicit_ = true;
    return *this;
  }

  /// @brief Require ordinary projection to redact before value access.
  /// @return This builder.
  constexpr ParamFieldBuilder& sensitive() noexcept {
    return sensitivity(ParamSensitivity::kSensitive);
  }
  /// @brief Require an archived contribution during future complete import.
  /// @return This builder; no loader is implemented by this declaration.
  constexpr ParamFieldBuilder& required_on_load() noexcept {
    required_on_load_ = true;
    return *this;
  }
  /// @brief Add one prior canonical migration input name.
  /// @param prior_name Nonempty valid UTF-8 prior identity with static lifetime.
  /// @return This builder; aliases never become output identity.
  constexpr ParamFieldBuilder& alias(std::string_view prior_name) noexcept {
    alias_ = prior_name;
    return *this;
  }
  /// @brief Declare semantic encoding for a `std::string` field.
  /// @param encoding `kArbitraryBytes` or `kUtf8`.
  /// @return This builder; UTF-8 bounds require `kUtf8` explicitly.
  constexpr ParamFieldBuilder& encoding(ParamStringEncoding encoding) noexcept {
    static_assert(std::same_as<detail::UnwrappedOptionalT<Field>, std::string>,
                  "string encoding policy requires a std::string field");
    string_encoding_ = encoding;
    return *this;
  }
  /// @brief Declare semantic encoding for each `std::string` sequence element.
  /// @param encoding `kArbitraryBytes` or `kUtf8`.
  /// @return This builder; per-element UTF-8 bounds require `kUtf8` explicitly.
  constexpr ParamFieldBuilder& element_encoding(ParamStringEncoding encoding) noexcept {
    static_assert(detail::IsStringVector<detail::UnwrappedOptionalT<Field>>::value,
                  "element encoding policy requires a std::vector<std::string> field");
    element_string_encoding_ = encoding;
    return *this;
  }
  /// @brief Bound valid UTF-8 storage for a `std::string` field.
  /// @param maximum Positive byte count.
  /// @return This builder; zero or a non-string field fails lowering.
  constexpr ParamFieldBuilder& max_utf8_bytes(std::size_t maximum) noexcept {
    static_assert(std::same_as<detail::UnwrappedOptionalT<Field>, std::string>,
                  "max_utf8_bytes requires a std::string field");
    invalid_constraint_ =
        invalid_constraint_ || maximum == 0U || !detail::fits_canonical_size(maximum);
    constraints_.max_utf8_bytes = maximum;
    return *this;
  }
  /// @brief Bound valid UTF-8 bytes for each `std::string` sequence element.
  /// @param maximum Positive byte count per element.
  /// @return This builder; zero, another field type, or omitted UTF-8 element policy rejects.
  constexpr ParamFieldBuilder& max_element_utf8_bytes(std::size_t maximum) noexcept {
    static_assert(detail::IsStringVector<detail::UnwrappedOptionalT<Field>>::value,
                  "max_element_utf8_bytes requires a std::vector<std::string> field");
    invalid_constraint_ =
        invalid_constraint_ || maximum == 0U || !detail::fits_canonical_size(maximum);
    constraints_.max_element_utf8_bytes = maximum;
    return *this;
  }
  /// @brief Bound the element count of a `std::vector` field.
  /// @param maximum Positive element count.
  /// @return This builder; zero or a non-vector field fails lowering.
  constexpr ParamFieldBuilder& max_elements(std::size_t maximum) noexcept {
    static_assert(detail::IsStdVector<detail::UnwrappedOptionalT<Field>>::value,
                  "max_elements requires a std::vector field");
    invalid_constraint_ =
        invalid_constraint_ || maximum == 0U || !detail::fits_canonical_size(maximum);
    constraints_.max_elements = maximum;
    return *this;
  }
  /// @brief Bound encoded bytes for a string, vector, path, or codec-qualified field.
  /// @param maximum Positive encoded byte count.
  /// @return This builder; zero or an inapplicable field fails lowering.
  constexpr ParamFieldBuilder& max_encoded_bytes(std::size_t maximum) noexcept {
    static_assert(
        std::same_as<detail::UnwrappedOptionalT<Field>, std::string> ||
            detail::IsStdVector<detail::UnwrappedOptionalT<Field>>::value ||
            std::same_as<detail::UnwrappedOptionalT<Field>, std::filesystem::path> ||
            ParamStorageTraits<detail::UnwrappedOptionalT<Field>>::summary().has_portable_codec,
        "max_encoded_bytes requires a string, vector, path, or codec-qualified field");
    invalid_constraint_ =
        invalid_constraint_ || maximum == 0U || !detail::fits_canonical_size(maximum);
    constraints_.max_encoded_bytes = maximum;
    return *this;
  }

  template <typename NestedParamsT>
  /// @brief Bind this exact object field to its already declared nested schema.
  /// @tparam NestedParamsT Exact field type with a declared schema.
  /// @return This builder; nested identity participates in parent hashes.
  constexpr ParamFieldBuilder& nested() noexcept {
    static_assert(std::same_as<detail::UnwrappedOptionalT<Field>, NestedParamsT>,
                  "nested parameter schema type must match the field type");
    static_assert(detail::HasParamSchema<NestedParamsT>,
                  "nested parameter field requires a declared nested schema");
    nested_ = true;
    return *this;
  }

 private:
  constexpr void reset_lifecycle_state() noexcept {
    earliest_invalidated_stage_ = LifecycleStage::kConfigure;
    invalid_lifecycle_stage_ = false;
  }

  template <auto Validator>
  static bool run_custom_validator(const void* aggregate) noexcept {
    const auto& value = static_cast<const ParamsT*>(aggregate)->*MemberPtr;
    return Validator(value);
  }

  template <typename ValueT>
  static ParamValidationResult validate_scalar_value(
      const ValueT& value, const ParamFieldDescriptor& descriptor) noexcept {
    if constexpr (detail::kIsParamScalar<ValueT>) {
      if (descriptor.constraints.has_range) {
        const ParamConstraintScalar observed = detail::to_scalar(value);
        if (!detail::scalar_finite(observed) ||
            detail::scalar_less(observed, descriptor.constraints.range_min) ||
            detail::scalar_less(descriptor.constraints.range_max, observed)) {
          return ParamValidationResult{
              ParamValidationCode::kOutOfRange, descriptor.id, descriptor.canonical_name};
        }
      }
      if (descriptor.constraints.allowed_value_count != 0U) {
        const ParamConstraintScalar observed = detail::to_scalar(value);
        bool found = false;
        for (std::size_t index = 0; index < descriptor.constraints.allowed_value_count; ++index) {
          found =
              found || detail::scalar_equal(observed, descriptor.constraints.allowed_values[index]);
        }
        if (!found) {
          return ParamValidationResult{
              ParamValidationCode::kNotAllowed, descriptor.id, descriptor.canonical_name};
        }
      }
    }
    return {};
  }

  template <typename ValueT>
  static ParamValidationResult validate_scalar_or_enum_sequence_value(
      const ValueT& value, const ParamFieldDescriptor& descriptor) noexcept {
    if constexpr (detail::IsStdVector<std::remove_cv_t<ValueT>>::value &&
                  std::is_enum_v<detail::EnumDomainT<ValueT>>) {
      for (const auto element : value) {
        const ParamValidationResult result = validate_scalar_value(element, descriptor);
        if (!result.ok()) {
          return result;
        }
      }
      return {};
    } else {
      return validate_scalar_value(value, descriptor);
    }
  }

  static ParamValidationResult validate_scalar_constraints(
      const Field& value, const ParamFieldDescriptor& descriptor) noexcept {
    if constexpr (detail::IsStdOptional<std::remove_cv_t<Field>>::value) {
      return value.has_value() ? validate_scalar_or_enum_sequence_value(*value, descriptor)
                               : ParamValidationResult{};
    } else {
      return validate_scalar_or_enum_sequence_value(value, descriptor);
    }
  }

  template <typename ValueT>
  static ParamValidationResult validate_encoding_value(
      const ValueT& value, const ParamFieldDescriptor& descriptor) noexcept {
    if constexpr (std::same_as<std::remove_cv_t<ValueT>, std::string>) {
      if (descriptor.string_encoding == ParamStringEncoding::kUtf8 && !detail::valid_utf8(value)) {
        return ParamValidationResult{
            ParamValidationCode::kInvalidEncoding, descriptor.id, descriptor.canonical_name};
      }
    } else if constexpr (detail::IsStringVector<std::remove_cv_t<ValueT>>::value) {
      if (descriptor.element_string_encoding == ParamStringEncoding::kUtf8) {
        for (const std::string& element : value) {
          if (!detail::valid_utf8(element)) {
            return ParamValidationResult{
                ParamValidationCode::kInvalidEncoding, descriptor.id, descriptor.canonical_name};
          }
        }
      }
    }
    return {};
  }

  static ParamValidationResult validate_encoding(const Field& value,
                                                 const ParamFieldDescriptor& descriptor) noexcept {
    if constexpr (detail::IsStdOptional<std::remove_cv_t<Field>>::value) {
      return value.has_value() ? validate_encoding_value(*value, descriptor)
                               : ParamValidationResult{};
    } else {
      return validate_encoding_value(value, descriptor);
    }
  }

  template <typename ValueT>
  static ParamValidationResult validate_cardinality_value(
      const ValueT& value, const ParamFieldDescriptor& descriptor) noexcept {
    if constexpr (std::same_as<std::remove_cv_t<ValueT>, std::string>) {
      if (descriptor.constraints.max_utf8_bytes != 0U &&
          value.size() > descriptor.constraints.max_utf8_bytes) {
        return ParamValidationResult{
            ParamValidationCode::kCardinalityExceeded, descriptor.id, descriptor.canonical_name};
      }
    } else if constexpr (detail::IsStdVector<std::remove_cv_t<ValueT>>::value) {
      if (descriptor.constraints.max_elements != 0U &&
          value.size() > descriptor.constraints.max_elements) {
        return ParamValidationResult{
            ParamValidationCode::kCardinalityExceeded, descriptor.id, descriptor.canonical_name};
      }
      if constexpr (detail::IsStringVector<std::remove_cv_t<ValueT>>::value) {
        if (descriptor.constraints.max_element_utf8_bytes != 0U) {
          for (const std::string& element : value) {
            if (element.size() > descriptor.constraints.max_element_utf8_bytes) {
              return ParamValidationResult{ParamValidationCode::kCardinalityExceeded,
                                           descriptor.id,
                                           descriptor.canonical_name};
            }
          }
        }
      }
    }
    return {};
  }

  static ParamValidationResult validate_cardinality(
      const Field& value, const ParamFieldDescriptor& descriptor) noexcept {
    if constexpr (detail::IsStdOptional<std::remove_cv_t<Field>>::value) {
      return value.has_value() ? validate_cardinality_value(*value, descriptor)
                               : ParamValidationResult{};
    } else {
      return validate_cardinality_value(value, descriptor);
    }
  }

  static ParamValidationResult validate_nested(const Field& value,
                                               const ParamFieldDescriptor& descriptor) noexcept {
    using ValueT = std::remove_cv_t<Field>;
    using NestedT = detail::UnwrappedOptionalT<ValueT>;
    if constexpr (detail::HasParamSchema<NestedT>) {
      if (descriptor.nested) {
        if constexpr (detail::IsStdOptional<ValueT>::value) {
          return value.has_value() ? validate_params(*value) : ParamValidationResult{};
        } else {
          return validate_params(value);
        }
      }
    }
    return {};
  }

  static ParamValidationResult validate_field(const void* aggregate,
                                              const ParamFieldDescriptor& descriptor) noexcept {
    const auto& value = static_cast<const ParamsT*>(aggregate)->*MemberPtr;
    const ParamValidationResult scalar_result = validate_scalar_constraints(value, descriptor);
    if (!scalar_result.ok()) {
      return scalar_result;
    }
    // Reject cheap byte/count bounds before scanning UTF-8.  Besides bounding cold-path work,
    // this freezes cardinality as the deterministic first failure when both facts are invalid.
    const ParamValidationResult cardinality_result = validate_cardinality(value, descriptor);
    if (!cardinality_result.ok()) {
      return cardinality_result;
    }
    const ParamValidationResult encoding_result = validate_encoding(value, descriptor);
    if (!encoding_result.ok()) {
      return encoding_result;
    }
    if (descriptor.custom_validator != nullptr && !descriptor.custom_validator(aggregate)) {
      return ParamValidationResult{ParamValidationCode::kValidatorRejected,
                                   descriptor.id,
                                   descriptor.canonical_name,
                                   descriptor.validator_id};
    }
    return validate_nested(value, descriptor);
  }

 public:
  /// @brief Return whether the accumulated declaration violates a static schema rule.
  /// @return \c true for an invalid declaration; intended for consteval schema lowering.
  [[nodiscard]] constexpr bool invalid() const noexcept {
    return !detail::valid_schema_text(canonical_name_) ||
           (!description_.empty() && !detail::valid_utf8(description_)) ||
           (!unit_.empty() && !detail::valid_utf8(unit_)) ||
           (custom_validator_ != nullptr && !detail::valid_schema_text(validator_id_)) ||
           (!alias_.empty() && !detail::valid_schema_text(alias_)) || invalid_lifecycle_stage_ ||
           invalid_range_ || invalid_allowed_value_ || invalid_constraint_ ||
           invalid_enum_choice_ || too_many_allowed_values_ ||
           (kEnumField && enum_choices_.count != 0U &&
            enum_choices_.count != constraints_.allowed_value_count) ||
           persistence_conflict_ ||
           static_cast<std::uint8_t>(string_encoding_) >
               static_cast<std::uint8_t>(ParamStringEncoding::kUtf8) ||
           static_cast<std::uint8_t>(element_string_encoding_) >
               static_cast<std::uint8_t>(ParamStringEncoding::kUtf8) ||
           (constraints_.max_utf8_bytes != 0U && string_encoding_ != ParamStringEncoding::kUtf8) ||
           (constraints_.max_element_utf8_bytes != 0U &&
            element_string_encoding_ != ParamStringEncoding::kUtf8) ||
           static_cast<std::uint8_t>(persistence_) >
               static_cast<std::uint8_t>(ParamPersistence::kTransient) ||
           static_cast<std::uint8_t>(external_access_) >
               static_cast<std::uint8_t>(ParamExternalAccess::kTuning) ||
           static_cast<std::uint8_t>(sensitivity_) >
               static_cast<std::uint8_t>(ParamSensitivity::kSensitive) ||
           external_access_conflict_ || sensitivity_conflict_ ||
           (external_access_ == ParamExternalAccess::kTuning &&
            (mutability_ != ParamMutability::kRuntime ||
             visibility_ == ParamVisibility::kInternal ||
             sensitivity_ == ParamSensitivity::kSensitive));
  }

 private:
  std::string_view canonical_name_{};
  std::string_view description_{};
  std::string_view unit_{};
  std::string_view validator_id_{};
  std::string_view alias_{};
  ParamMutability mutability_{ParamMutability::kCompileTime};
  LifecycleStage earliest_invalidated_stage_{LifecycleStage::kConfigure};
  ParamPersistence persistence_{ParamPersistence::kPersisted};
  ParamStringEncoding string_encoding_{ParamStringEncoding::kArbitraryBytes};
  ParamStringEncoding element_string_encoding_{ParamStringEncoding::kArbitraryBytes};
  ParamVisibility visibility_{ParamVisibility::kPublic};
  ParamExternalAccess external_access_{ParamExternalAccess::kNone};
  ParamSensitivity sensitivity_{ParamSensitivity::kNormal};
  ParamConstraints constraints_{};
  ParamEnumChoices enum_choices_{};
  bool persistence_explicit_{};
  bool external_access_explicit_{};
  bool sensitivity_explicit_{};
  bool persistence_conflict_{};
  bool external_access_conflict_{};
  bool sensitivity_conflict_{};
  bool required_on_load_{};
  bool nested_{};
  bool invalid_lifecycle_stage_{};
  bool invalid_range_{};
  bool invalid_allowed_value_{};
  bool invalid_constraint_{};
  bool invalid_enum_choice_{};
  bool too_many_allowed_values_{};
  bool (*custom_validator_)(const void*) noexcept {};

  static constexpr bool kEnumField = std::is_enum_v<detail::EnumDomainT<Field>>;

  friend constexpr ParamFieldDescriptor detail::make_field_descriptor<ParamsT, MemberPtr>(
      std::string_view schema_id, const ParamFieldBuilder& builder);
};

template <auto MemberPtr>
/// @brief Create a compile-time declaration builder for one exact aggregate member.
/// @tparam MemberPtr Non-static data-member pointer.
/// @param canonical_name Nonempty valid UTF-8 structured path with static lifetime.
/// @return Fluent constant-evaluation builder.
[[nodiscard]] constexpr auto field(std::string_view canonical_name) noexcept {
  using Traits = detail::MemberPointerTraits<decltype(MemberPtr)>;
  return ParamFieldBuilder<typename Traits::Owner, MemberPtr>{canonical_name};
}

namespace detail {

template <auto First, auto... Rest>
struct ParamPathValidator;

template <typename ExpectedOwner, auto Next, auto... Rest>
[[nodiscard]] consteval bool valid_nested_param_path() {
  using NextTraits = MemberPointerTraits<decltype(Next)>;
  if constexpr (!std::same_as<UnwrappedOptionalT<std::remove_cv_t<ExpectedOwner>>,
                              std::remove_cv_t<typename NextTraits::Owner>>) {
    return false;
  } else {
    return ParamPathValidator<Next, Rest...>::valid();
  }
}

template <auto First, auto... Rest>
struct ParamPathValidator {
  [[nodiscard]] static consteval bool valid() {
    using Traits = MemberPointerTraits<decltype(First)>;
    using Owner = typename Traits::Owner;
    using Field = typename Traits::Field;
    if constexpr (!HasParamSchema<Owner>) {
      return false;
    } else {
      constexpr auto static_schema = holoscan_param_schema(type_tag<Owner>{});
      bool declared = false;
      bool nested = false;
      for (const ParamFieldDescriptor& descriptor : static_schema.fields) {
        if (descriptor.local_member_token == &kMemberToken<First>) {
          declared = true;
          nested = descriptor.nested;
        }
      }
      if constexpr (sizeof...(Rest) == 0U) {
        return declared;
      } else {
        return declared && nested && valid_nested_param_path<Field, Rest...>();
      }
    }
  }
};

template <auto... MemberPath>
[[nodiscard]] consteval bool valid_param_path() {
  if constexpr (sizeof...(MemberPath) == 0U) {
    return false;
  } else {
    return ParamPathValidator<MemberPath...>::valid();
  }
}

}  // namespace detail

template <auto... MemberPath>
  requires(detail::valid_param_path<MemberPath...>())
/// @brief Carry one compile-time dependency path through declared schema fields.
/// @tparam MemberPath One declared field pointer followed only by declared nested field pointers.
///
/// The empty value grants no read or update authority. Opaque and unlisted fields cannot
/// instantiate this type. Compilation later lowers it to pointer-free plan identity.
struct ParamRef {};

template <auto... MemberPath>
  requires(detail::valid_param_path<MemberPath...>())
/// @brief Create a dependency token for one declared schema path.
/// @tparam MemberPath Valid declared field path.
/// @return Empty compile-time token carrying only type-level member evidence.
[[nodiscard]] constexpr ParamRef<MemberPath...> param_ref() noexcept {
  return {};
}

template <auto... MemberPath>
  requires(!detail::valid_param_path<MemberPath...>())
/// @brief Reject a dependency token whose path is absent from the declared schema.
/// @tparam MemberPath Undeclared or otherwise invalid parameter field path.
/// @note This overload exists only to provide one implementation-owned portable diagnostic.
consteval void param_ref() noexcept {
  static_assert(detail::valid_param_path<MemberPath...>(),
                "param_ref requires a declared parameter schema path");
}

namespace detail {

/// Carry process-local authoring operations for one duration-valued Parameter dependency.
///
/// These pointers are consumed only while replaying an Operator factory. They are never copied
/// into an ExecutionPlan, schema document, diagnostic, digest, trace, or transport record.
struct ParamDurationDependency {
  const void* root_type_token{};
  const ParamSchemaDescriptor* (*root_schema)() noexcept {};
  const ParamFieldDescriptor* (*final_field)() noexcept {};
  std::string (*canonical_path)() = nullptr;
  bool (*read_initial_ns)(const void*, std::int64_t&) noexcept {};
  bool (*constraint_to_ns)(ParamConstraintScalar, std::int64_t&) noexcept {};

  [[nodiscard]] constexpr explicit operator bool() const noexcept {
    return root_type_token != nullptr && root_schema != nullptr && final_field != nullptr &&
           canonical_path != nullptr && read_initial_ns != nullptr && constraint_to_ns != nullptr;
  }
};

template <auto First, auto... Rest>
struct ParamPathValue;

template <auto First>
struct ParamPathValue<First> {
  using Type = typename MemberPointerTraits<decltype(First)>::Field;
};

template <auto First, auto Second, auto... Rest>
struct ParamPathValue<First, Second, Rest...> {
  using Current = typename MemberPointerTraits<decltype(First)>::Field;
  using NextOwner = typename MemberPointerTraits<decltype(Second)>::Owner;
  static_assert(
      std::same_as<UnwrappedOptionalT<std::remove_cv_t<Current>>, std::remove_cv_t<NextOwner>>);
  using Type = typename ParamPathValue<Second, Rest...>::Type;
};

template <auto First, auto... Rest, typename Object>
[[nodiscard]] constexpr auto param_path_value(const Object& object) noexcept {
  const auto& field = object.*First;
  if constexpr (sizeof...(Rest) == 0U) {
    return std::addressof(field);
  } else if constexpr (IsStdOptional<std::remove_cvref_t<decltype(field)>>::value) {
    using Result = decltype(param_path_value<Rest...>(*field));
    if (!field.has_value()) {
      return Result{};
    }
    return param_path_value<Rest...>(*field);
  } else {
    return param_path_value<Rest...>(field);
  }
}

template <auto Member>
[[nodiscard]] const ParamFieldDescriptor* declared_member_descriptor() noexcept {
  using Owner = typename MemberPointerTraits<decltype(Member)>::Owner;
  const ParamSchemaDescriptor* schema = param_schema_descriptor<std::remove_cv_t<Owner>>();
  if (schema == nullptr) {
    return nullptr;
  }
  const auto found =
      std::find_if(schema->fields.begin(), schema->fields.end(), [](const auto& field) {
        return field.local_member_token == &kMemberToken<Member>;
      });
  return found == schema->fields.end() ? nullptr : &*found;
}

template <auto First, auto... Rest>
[[nodiscard]] const ParamFieldDescriptor* final_param_field() noexcept {
  if constexpr (sizeof...(Rest) == 0U) {
    return declared_member_descriptor<First>();
  } else {
    return final_param_field<Rest...>();
  }
}

template <auto Member>
[[nodiscard]] bool declared_member_is_sensitive() noexcept {
  const ParamFieldDescriptor* field = declared_member_descriptor<Member>();
  return field == nullptr || field->sensitivity == ParamSensitivity::kSensitive;
}

template <auto... MemberPath>
[[nodiscard]] bool param_path_has_sensitive_member() noexcept {
  return (declared_member_is_sensitive<MemberPath>() || ...);
}

template <auto First, auto... Rest>
[[nodiscard]] std::string canonical_param_path() {
  std::string result;
  const ParamFieldDescriptor* first = declared_member_descriptor<First>();
  if (first == nullptr) {
    return result;
  }
  result = first->canonical_name;
  auto append = [&]<auto Member>() {
    const ParamFieldDescriptor* field = declared_member_descriptor<Member>();
    if (field == nullptr) {
      result.clear();
      return;
    }
    result.push_back('.');
    result += field->canonical_name;
  };  // NOLINT(readability/braces)
  (append.template operator()<Rest>(), ...);
  return result;
}

template <typename Duration>
[[nodiscard]] bool duration_to_canonical_ns(Duration value, std::int64_t& result) noexcept {
  using Rep = typename Duration::rep;
  if constexpr (!std::is_integral_v<Rep>) {
    return false;
  } else {
    const DurationScalarConversion converted =
        integral_duration_scalar_in_field_domain<std::chrono::nanoseconds>(value);
    if (!converted.valid) {
      return false;
    }
    if (converted.scalar.kind == ParamConstraintScalarKind::kSigned) {
      result = converted.scalar.signed_value;
      return true;
    }
    if (converted.scalar.kind == ParamConstraintScalarKind::kUnsigned &&
        converted.scalar.unsigned_value <=
            static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
      result = static_cast<std::int64_t>(converted.scalar.unsigned_value);
      return true;
    }
    return false;
  }
}

template <typename Duration>
[[nodiscard]] bool duration_constraint_to_canonical_ns(ParamConstraintScalar scalar,
                                                       std::int64_t& result) noexcept {
  using Rep = typename Duration::rep;
  if constexpr (!std::is_integral_v<Rep>) {
    return false;
  } else if constexpr (std::is_signed_v<Rep>) {
    if (scalar.kind != ParamConstraintScalarKind::kSigned ||
        scalar.signed_value < static_cast<std::int64_t>(std::numeric_limits<Rep>::min()) ||
        scalar.signed_value > static_cast<std::int64_t>(std::numeric_limits<Rep>::max())) {
      return false;
    }
    return duration_to_canonical_ns(Duration{static_cast<Rep>(scalar.signed_value)}, result);
  } else {
    if (scalar.kind != ParamConstraintScalarKind::kUnsigned ||
        scalar.unsigned_value > static_cast<std::uint64_t>(std::numeric_limits<Rep>::max())) {
      return false;
    }
    return duration_to_canonical_ns(Duration{static_cast<Rep>(scalar.unsigned_value)}, result);
  }
}

template <auto First, auto... Rest>
struct ParamDurationDependencyModel {
  using Root = std::remove_cv_t<typename MemberPointerTraits<decltype(First)>::Owner>;
  using Duration = std::remove_cv_t<typename ParamPathValue<First, Rest...>::Type>;
  static_assert(kIsChronoDuration<Duration>,
                "temporal param_ref dependencies require a chrono duration field");
  static_assert(std::is_integral_v<typename Duration::rep>,
                "temporal param_ref dependencies require an integral duration representation");
  static_assert(sizeof(typename Duration::rep) <= sizeof(std::uint64_t),
                "temporal param_ref dependencies require at most a 64-bit representation");

  [[nodiscard]] static const ParamSchemaDescriptor* root_schema() noexcept {
    return param_schema_descriptor<Root>();
  }

  [[nodiscard]] static bool read_initial_ns(const void* aggregate, std::int64_t& result) noexcept {
    if (aggregate == nullptr) {
      return false;
    }
    const Duration* value = param_path_value<First, Rest...>(*static_cast<const Root*>(aggregate));
    return value != nullptr && duration_to_canonical_ns(*value, result);
  }

  [[nodiscard]] static ParamDurationDependency make() noexcept {
    return ParamDurationDependency{
        .root_type_token = &kParameterTypeToken<Root>,
        .root_schema = &root_schema,
        .final_field = &final_param_field<First, Rest...>,
        .canonical_path = &canonical_param_path<First, Rest...>,
        .read_initial_ns = &read_initial_ns,
        .constraint_to_ns = &duration_constraint_to_canonical_ns<Duration>,
    };
  }
};

template <auto... MemberPath>
[[nodiscard]] ParamDurationDependency make_param_duration_dependency(
    ParamRef<MemberPath...>) noexcept {
  return ParamDurationDependencyModel<MemberPath...>::make();
}

}  // namespace detail

template <typename ParamsT, std::size_t FieldCount>
/// @brief Store one side-effect-free static typed parameter schema.
/// @tparam ParamsT Exact aggregate type.
/// @tparam FieldCount Number of declared fields.
///
/// Every member has static lifetime when returned by `param_schema<ParamsT>()`. The local field
/// array contains operation thunks; `document_fields` contains the operation-free projection.
struct StaticParamSchema {
  ParamSchemaHeader header;  ///< Identity, version, coverage, proof, and description.
  std::array<ParamFieldDescriptor, FieldCount> fields;         ///< Typed local descriptors.
  std::array<ParamFieldDocument, FieldCount> document_fields;  ///< Logical projections.
  ParamSchemaHash structural_hash;  ///< Decoding-shape compatibility fingerprint.
  ParamSchemaHash contract_hash;    ///< Enforced-policy compatibility fingerprint.
  ParamSchemaHash metadata_hash;    ///< Contract-plus-description fingerprint.
};

namespace detail {

template <typename ParamsT, typename... Fields>
/// @brief Retain declared field types without adding them to the erased schema descriptor.
/// @tparam ParamsT Exact aggregate type.
/// @tparam Fields Exact field builder types in declaration order.
struct TypedParamSchema : StaticParamSchema<ParamsT, sizeof...(Fields)> {
  using FieldBuilders = std::tuple<Fields...>;  ///< Typed declarations for optional tooling.
};

template <typename ParamsT, auto MemberPtr>
constexpr ParamFieldDescriptor make_field_descriptor(
    std::string_view schema_id, const ParamFieldBuilder<ParamsT, MemberPtr>& builder) {
  StableHashBuilder id_hash;
  id_hash.add("holoscan.param.field.v1");
  id_hash.add(schema_id);
  id_hash.add(builder.canonical_name_);
  const ParamSchemaHash full_id = id_hash.finish();
  std::uint64_t compact_id{};
  for (std::size_t index = 0; index < sizeof(compact_id); ++index) {
    compact_id = (compact_id << 8U) | std::to_integer<std::uint8_t>(full_id.value[index]);
  }
  using FieldT = typename ParamFieldBuilder<ParamsT, MemberPtr>::Field;
  ParamFieldDescriptor result{
      .id = compact_id,
      .full_id = full_id,
      .canonical_name = builder.canonical_name_,
      .logical_type = logical_type<FieldT>(),
      .type_shape = field_type_shape<FieldT>(),
      .mutability = builder.mutability_,
      .earliest_invalidated_stage = builder.earliest_invalidated_stage_,
      .persistence = builder.persistence_,
      .string_encoding = builder.string_encoding_,
      .element_string_encoding = builder.element_string_encoding_,
      .visibility = builder.visibility_,
      .external_access = builder.external_access_,
      .sensitivity = builder.sensitivity_,
      .policy_evidence =
          ParamFieldPolicyEvidence{
              .persistence_explicit = builder.persistence_explicit_,
              .external_access_explicit = builder.external_access_explicit_,
              .sensitivity_explicit = builder.sensitivity_explicit_,
          },
      .required_on_load = builder.required_on_load_,
      .nested = builder.nested_,
      .alias = builder.alias_,
      .description = builder.description_,
      .unit = builder.unit_,
      .validator_id = builder.validator_id_,
      .constraints = builder.constraints_,
      .enum_choices = builder.enum_choices_,
      .storage = ParamStorageTraits<std::remove_cv_t<FieldT>>::summary(),
      .local_member_token = &kMemberToken<MemberPtr>,
      .local_operations = make_field_local_operations<ParamsT, MemberPtr>(builder.nested_),
      .validate = &ParamFieldBuilder<ParamsT, MemberPtr>::validate_field,
      .custom_validator = builder.custom_validator_,
  };
  using NestedT = UnwrappedOptionalT<FieldT>;
  if constexpr (HasParamSchema<NestedT>) {
    if (builder.nested_) {
      constexpr auto nested_schema = holoscan_param_schema(type_tag<NestedT>{});
      result.nested_schema_id = nested_schema.header.stable_id();
      result.nested_schema_version = nested_schema.header.version();
      result.nested_structural_hash = nested_schema.structural_hash;
      result.nested_contract_hash = nested_schema.contract_hash;
      bool nested_shape_stable = nested_schema.header.coverage() == ParamSchemaCoverage::kComplete;
      for (const ParamFieldDescriptor& nested_field : nested_schema.fields) {
        nested_shape_stable = nested_shape_stable && nested_field.type_shape.has_stable_shape;
      }
      result.type_shape.has_stable_shape = nested_shape_stable;
    }
  }
  return result;
}

constexpr ParamFieldDocument make_document_field(const ParamFieldDescriptor& field) noexcept {
  return ParamFieldDocument{
      .id = field.id,
      .full_id = field.full_id,
      .canonical_name = field.canonical_name,
      .logical_type = field.logical_type,
      .type_shape = field.type_shape,
      .mutability = field.mutability,
      .earliest_invalidated_stage = field.earliest_invalidated_stage,
      .persistence = field.persistence,
      .string_encoding = field.string_encoding,
      .element_string_encoding = field.element_string_encoding,
      .visibility = field.visibility,
      .external_access = field.external_access,
      .sensitivity = field.sensitivity,
      .policy_evidence = field.policy_evidence,
      .required_on_load = field.required_on_load,
      .nested = field.nested,
      .nested_schema_id = field.nested_schema_id,
      .nested_schema_version = field.nested_schema_version,
      .nested_structural_hash = field.nested_structural_hash,
      .nested_contract_hash = field.nested_contract_hash,
      .alias = field.alias,
      .description = field.description,
      .unit = field.unit,
      .validator_id = field.validator_id,
      .constraints = field.constraints,
      .enum_choices = field.enum_choices,
      .storage = field.storage,
  };
}

constexpr void add_structural_field(StableHashBuilder& hash,
                                    const ParamFieldDescriptor& field) noexcept {
  hash.add(field.canonical_name);
  hash.add(field.type_shape.type);
  hash.add(field.type_shape.bit_width);
  hash.add(field.type_shape.element_or_rep_type);
  hash.add(field.type_shape.element_or_rep_bit_width);
  hash.add(field.type_shape.duration_period_numerator);
  hash.add(field.type_shape.duration_period_denominator);
  if (field.type_shape.optional) {
    hash.add("holoscan.param.optional.v1");
  }
  hash.add(field.type_shape.has_stable_shape);
  if (field.enum_choices.count != 0U) {
    hash.add("holoscan.param.enum_choices.v1");
    hash.add(field.enum_choices.count);
    for (std::size_t index = 0; index < field.enum_choices.count; ++index) {
      const ParamEnumChoice& choice = field.enum_choices.values[index];
      hash.add(choice.name);
      hash.add(choice.value.kind);
      hash.add(choice.value.bit_width);
      hash.add(choice.value.raw_bits);
    }
  }
  hash.add(field.string_encoding);
  hash.add(field.element_string_encoding);
  hash.add(field.nested);
  hash.add(field.nested_schema_id);
  hash.add(field.nested_schema_version);
  for (const std::byte byte : field.nested_structural_hash.value) {
    hash.add_byte(std::to_integer<std::uint8_t>(byte));
  }
  hash.add_canonical_size(field.constraints.max_utf8_bytes);
  hash.add_canonical_size(field.constraints.max_element_utf8_bytes);
  hash.add_canonical_size(field.constraints.max_elements);
  hash.add_canonical_size(field.constraints.max_encoded_bytes);
}

constexpr void add_contract_field(StableHashBuilder& hash,
                                  const ParamFieldDescriptor& field) noexcept {
  add_structural_field(hash, field);
  hash.add(field.mutability);
  hash.add(field.earliest_invalidated_stage);
  hash.add(field.persistence);
  hash.add(field.visibility);
  hash.add(field.external_access);
  hash.add(field.sensitivity);
  hash.add(field.required_on_load);
  hash.add(field.constraints.has_range);
  hash.add(field.constraints.range_min);
  hash.add(field.constraints.range_max);
  hash.add(field.constraints.allowed_value_count);
  for (std::size_t index = 0; index < field.constraints.allowed_value_count; ++index) {
    hash.add(field.constraints.allowed_values[index]);
  }
  hash.add(field.alias);
  hash.add(field.validator_id);
  for (const std::byte byte : field.nested_contract_hash.value) {
    hash.add_byte(std::to_integer<std::uint8_t>(byte));
  }
}

}  // namespace detail

namespace detail {

template <typename ParamsT, typename... Fields>
consteval auto make_param_schema(ParamSchemaHeader header, Fields... builders) {
  static_assert(ParameterAggregate<ParamsT>,
                "parameter schema requires an exact default-initializable copyable aggregate");
  static_assert((std::same_as<typename Fields::Owner, ParamsT> && ...),
                "parameter field pointer must belong to the declared aggregate");
  if (!detail::valid_schema_text(header.stable_id()) || header.version() == 0 ||
      (!header.description_text().empty() && !detail::valid_utf8(header.description_text()))) {
    throw "parameter schema requires a valid UTF-8 stable id and nonzero version";
  }
  if ((builders.invalid() || ...)) {
    throw "invalid parameter field policy or constraint";
  }

  TypedParamSchema<ParamsT, Fields...> result{
      StaticParamSchema<ParamsT, sizeof...(Fields)>{
          .header = header,
          .fields = {detail::make_field_descriptor(header.stable_id(), builders)...},
      },
  };
  for (std::size_t index = 0; index < result.fields.size(); ++index) {
    result.document_fields[index] = detail::make_document_field(result.fields[index]);
    for (std::size_t prior = 0; prior < index; ++prior) {
      if (result.fields[index].canonical_name == result.fields[prior].canonical_name ||
          result.fields[index].id == result.fields[prior].id ||
          result.fields[index].local_member_token == result.fields[prior].local_member_token) {
        throw "duplicate parameter field member, key, or compact id";
      }
    }
  }

  detail::StableHashBuilder structural;
  detail::StableHashBuilder contract;
  detail::StableHashBuilder metadata;
  structural.add("holoscan.param.structural.v1");
  contract.add("holoscan.param.contract.v1");
  metadata.add("holoscan.param.metadata.v1");
  structural.add(header.stable_id());
  structural.add(header.version());
  structural.add(header.coverage());
  contract.add(header.stable_id());
  contract.add(header.version());
  contract.add(header.coverage());
  metadata.add(header.stable_id());
  metadata.add(header.version());
  metadata.add(header.coverage());
  metadata.add(header.description_text());
  for (const ParamFieldDescriptor& field_descriptor : result.fields) {
    detail::add_structural_field(structural, field_descriptor);
    detail::add_contract_field(contract, field_descriptor);
    detail::add_contract_field(metadata, field_descriptor);
    metadata.add(field_descriptor.policy_evidence.persistence_explicit);
    metadata.add(field_descriptor.policy_evidence.external_access_explicit);
    metadata.add(field_descriptor.policy_evidence.sensitivity_explicit);
    metadata.add(field_descriptor.description);
    metadata.add(field_descriptor.unit);
  }
  result.structural_hash = structural.finish();
  result.contract_hash = contract.finish();
  result.metadata_hash = metadata.finish();
  return result;
}

template <typename ParamsT>
  requires ParameterAggregate<ParamsT> && HasParamSchema<ParamsT>
inline constexpr auto kParamSchema = holoscan_param_schema(type_tag<ParamsT>{});

template <typename ParamsT, std::size_t N>
[[nodiscard]] const StaticParamSchema<ParamsT, N>& public_param_schema_ref(
    const StaticParamSchema<ParamsT, N>& schema) noexcept {
  return schema;
}

}  // namespace detail

template <typename ParamsT>
  requires ParameterAggregate<ParamsT> && detail::HasParamSchema<ParamsT>
/// @brief Return the static typed schema declared for an exact aggregate.
/// @tparam ParamsT Exact aggregate with `HOLOSCAN_PARAM_SCHEMA`.
/// @return Process-lifetime immutable `StaticParamSchema` reference.
/// @note This operation performs no registration, allocation, lookup, or synchronization.
[[nodiscard]] const auto& param_schema() noexcept {
  return detail::public_param_schema_ref(detail::kParamSchema<ParamsT>);
}

template <typename ParamsT>
  requires ParameterAggregate<ParamsT>
/// @brief Return whether an exact aggregate declares field metadata.
/// @tparam ParamsT Exact aggregate type.
/// @return `kDeclared` when ADL finds a static schema, otherwise `kOpaqueLocal`.
[[nodiscard]] consteval ParamSchemaAvailability param_schema_availability() noexcept {
  return detail::HasParamSchema<ParamsT> ? ParamSchemaAvailability::kDeclared
                                         : ParamSchemaAvailability::kOpaqueLocal;
}

template <typename ParamsT>
  requires ParameterAggregate<ParamsT>
/// @brief Return the immutable process-local erased schema view.
/// @tparam ParamsT Exact aggregate type.
/// @return Process-lifetime descriptor, or `nullptr` for opaque-local mode.
/// @note The returned local operation pointers must never enter a plan or portable artifact.
[[nodiscard]] const ParamSchemaDescriptor* param_schema_descriptor() noexcept {
  if constexpr (!detail::HasParamSchema<ParamsT>) {
    return nullptr;
  } else {
    const auto& static_schema = param_schema<ParamsT>();
    static const ParamSchemaDescriptor descriptor{
        .stable_id = static_schema.header.stable_id(),
        .version = static_schema.header.version(),
        .description = static_schema.header.description_text(),
        .structural_hash = static_schema.structural_hash,
        .contract_hash = static_schema.contract_hash,
        .metadata_hash = static_schema.metadata_hash,
        .coverage = static_schema.header.coverage(),
        .proof = static_schema.header.proof(),
        .local_type = &param_type_descriptor<ParamsT>(),
        .fields = static_schema.fields,
    };
    return &descriptor;
  }
}

template <typename ParamsT>
  requires ParameterAggregate<ParamsT>
/// @brief Return the immutable operation-free logical schema projection.
/// @tparam ParamsT Exact aggregate type.
/// @return Process-lifetime document view, or `nullptr` for opaque-local mode.
/// @note String and field views borrow static schema storage; HSDK5-018 owns an owning archive
/// form.
[[nodiscard]] const ParamSchemaDocument* param_schema_document() noexcept {
  if constexpr (!detail::HasParamSchema<ParamsT>) {
    return nullptr;
  } else {
    const auto& static_schema = param_schema<ParamsT>();
    static const ParamSchemaDocument document{
        .stable_id = static_schema.header.stable_id(),
        .version = static_schema.header.version(),
        .description = static_schema.header.description_text(),
        .structural_hash = static_schema.structural_hash,
        .contract_hash = static_schema.contract_hash,
        .metadata_hash = static_schema.metadata_hash,
        .coverage = static_schema.header.coverage(),
        .proof = static_schema.header.proof(),
        .fields = static_schema.document_fields,
    };
    return &document;
  }
}

template <typename ParamsT>
  requires ParameterAggregate<ParamsT>
/// @brief Validate every declared field of one complete aggregate in declaration order.
/// @tparam ParamsT Exact aggregate type.
/// @param params Immutable aggregate to validate; it is never modified.
/// @return Success for opaque mode or accepted fields, otherwise the first bounded failure.
/// @note This is a cold-path operation and may execute a declared local `noexcept` validator.
[[nodiscard]] ParamValidationResult validate_params(const ParamsT& params) noexcept {
  const ParamSchemaDescriptor* descriptor = param_schema_descriptor<ParamsT>();
  if (descriptor == nullptr) {
    return {};
  }
  for (const ParamFieldDescriptor& field_descriptor : descriptor->fields) {
    const ParamValidationResult result = field_descriptor.validate(&params, field_descriptor);
    if (!result.ok()) {
      return result;
    }
  }
  return {};
}

/// @brief Declare one side-effect-free static parameter schema through ADL.
/// @param PARAMS_TYPE Exact aggregate receiving the schema.
/// @param ... One leading `param_schema(id, version)` clause followed by typed `field` clauses.
/// @note Invoke this macro in `PARAMS_TYPE`'s associated namespace so ADL can find both emitted
/// overloads.
#define HOLOSCAN_PARAM_SCHEMA(PARAMS_TYPE, ...)                                              \
  consteval auto holoscan_param_schema(::holoscan::detail::type_tag<PARAMS_TYPE>) noexcept { \
    return ::holoscan::detail::make_param_schema<PARAMS_TYPE>(__VA_ARGS__);                  \
  }                                                                                          \
  inline ::holoscan::ParamValidationResult holoscan_param_validate(                          \
      ::holoscan::detail::type_tag<PARAMS_TYPE>, const void* aggregate) noexcept {           \
    return ::holoscan::validate_params(*static_cast<const PARAMS_TYPE*>(aggregate));         \
  }

}  // namespace holoscan
