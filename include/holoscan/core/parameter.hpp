/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <bit>
#include <chrono>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <string_view>
#include <type_traits>

namespace holoscan {

/// @brief Carry the collision-checked 64-bit projection of a canonical parameter field digest.
///
/// V1 hashes `"holoscan.param.field.v1" 0xff schema-id 0xff canonical-path 0xff`, excludes
/// schema version, and interprets digest bytes 0 through 7 in big-endian order. Static schema
/// descriptors retain the complete digest so a projection collision can reject deterministically.
using ParamFieldId = std::uint64_t;

/// @brief Classify the exact fixed scalar representation carried by a parameter operation.
///
/// The numeric values and accompanying bit width describe semantic scalar bits, not a native C++
/// object layout or wire encoding. Enum values use their underlying scalar representation and do
/// not retain a process-local C++ enum identity.
enum class ParamScalarKind : std::uint8_t {
  kNone = 0,                 ///< No scalar value is present.
  kBool = 1,                 ///< One Boolean bit is present.
  kSignedInteger = 2,        ///< A two's-complement signed integer bit pattern is present.
  kUnsignedInteger = 3,      ///< An unsigned integer bit pattern is present.
  kFloatingPoint = 4,        ///< An IEC 60559 binary32 or binary64 bit pattern is present.
  kDurationNanoseconds = 5,  ///< A checked signed 64-bit nanosecond count is present.
};

/// @brief Identify one bounded cold-path parameter validation result category.
enum class ParamValidationCode : std::uint8_t {
  kOk,                   ///< Every declared constraint accepted.
  kOutOfRange,           ///< A finite scalar value violated its inclusive range.
  kNotAllowed,           ///< A scalar value was absent from the allowed set.
  kCardinalityExceeded,  ///< A declared finite size bound was exceeded.
  kInvalidEncoding,      ///< A string was not valid UTF-8.
  kValidatorRejected,    ///< A named local Boolean validator returned false.
};

/// @brief Report one bounded cold-path aggregate validation outcome.
struct ParamValidationResult {
  ParamValidationCode code{ParamValidationCode::kOk};  ///< Primary bounded result.
  ParamFieldId field_id{};          ///< Failing field projection, or zero on success.
  std::string_view field_name{};    ///< Static canonical field name, or empty on success.
  std::string_view validator_id{};  ///< Stable validator ID only for validator rejection.

  /// @brief Return whether validation accepted the complete aggregate.
  /// @return \c true when no declared constraint rejected the aggregate.
  [[nodiscard]] constexpr bool ok() const noexcept { return code == ParamValidationCode::kOk; }
};

static_assert(std::is_standard_layout_v<ParamValidationResult>);
static_assert(std::is_trivially_copyable_v<ParamValidationResult>);

namespace detail {
template <typename T>
struct type_tag {};

template <typename T>
[[nodiscard]] constexpr ParamValidationResult holoscan_param_validate(type_tag<T>,
                                                                      const void* value) noexcept {
  (void)value;
  return {};
}
}  // namespace detail

/// @brief Represent an Operator with no configurable parameter aggregate.
///
/// Runtime binds one process-static immutable value at generation zero. The type carries no
/// hidden state and requires no per-Operator generation, candidate, update, or reclamation slot.
struct NoParams {};

/// @brief Require the exact plain aggregate shape accepted as Operator parameter storage.
/// @tparam T Candidate exact unqualified aggregate type.
///
/// The `default_initializable` requirement transitively requires `T` to satisfy
/// `std::destructible`, including non-throwing destruction. Admission as callback storage does not
/// imply that every private cold-path materialization strategy can replace a live aggregate;
/// callers must separately qualify the strategy they use.
template <typename T>
concept ParameterAggregate =
    std::same_as<T, std::remove_cvref_t<T>> && std::default_initializable<T> &&
    std::copy_constructible<T> && std::is_aggregate_v<T>;

/// @brief Summarize intrinsic C++ storage facts without selecting a Runtime profile strategy.
///
/// These facts do not claim that a value is portable, serialized, allocation-free to clone, or
/// admitted by a profile. In particular, a declared length bound is not bounded-materialization
/// proof for an ordinary standard-library container.
struct ParamStorageTraitsSummary {
  bool owns_dynamic_storage{};              ///< The value may own separately allocated storage.
  bool fixed_object_extent{};               ///< `sizeof(T)` contains the complete live value.
  bool has_portable_codec{};                ///< A separately qualified portable codec is known.
  bool supports_bounded_materialization{};  ///< Complete finite materialization is proven.
  std::size_t inline_bytes{};               ///< Proven inline semantic bytes, excluding padding.
};

static_assert(std::is_standard_layout_v<ParamStorageTraitsSummary>);
static_assert(std::is_trivially_copyable_v<ParamStorageTraitsSummary>);

/// @brief Customize intrinsic storage evidence for one exact parameter or field type.
/// @tparam T Exact unqualified C++ type.
///
/// The conservative default treats non-trivially-copyable values as potentially dynamic and does
/// not infer a portable codec. Applications and qualified packages may explicitly specialize this
/// trait; graph compilation remains responsible for accepting or rejecting a concrete strategy.
template <typename T>
struct ParamStorageTraits {
  /// @brief Return conservative intrinsic storage facts for \p T.
  /// @return Facts independent of a selected Runtime profile.
  [[nodiscard]] static consteval ParamStorageTraitsSummary summary() noexcept {
    constexpr bool intrinsically_bounded = std::is_arithmetic_v<T> || std::is_enum_v<T>;
    return ParamStorageTraitsSummary{
        .owns_dynamic_storage = !intrinsically_bounded,
        .fixed_object_extent = intrinsically_bounded,
        .has_portable_codec = false,
        .supports_bounded_materialization = intrinsically_bounded,
        .inline_bytes = intrinsically_bounded ? sizeof(T) : 0,
    };
  }
};

/// @brief Describe fixed integral `std::chrono::duration` parameter storage.
/// @tparam Rep Exact integral duration representation type.
/// @tparam Period Exact compile-time duration period.
///
/// The C++ object representation is process-local and is never a portable artifact codec.
/// Compilation converts admitted values and finite constraints exactly into canonical signed
/// nanoseconds before publishing plan records.
template <typename Rep, typename Period>
  requires(std::integral<Rep> && sizeof(Rep) <= sizeof(std::uint64_t))
struct ParamStorageTraits<std::chrono::duration<Rep, Period>> {
  /// @brief Return fixed, bounded intrinsic duration storage facts.
  /// @return Allocation-free facts for the exact duration object.
  [[nodiscard]] static consteval ParamStorageTraitsSummary summary() noexcept {
    return ParamStorageTraitsSummary{
        .owns_dynamic_storage = false,
        .fixed_object_extent = true,
        .has_portable_codec = false,
        .supports_bounded_materialization = true,
        .inline_bytes = sizeof(std::chrono::duration<Rep, Period>),
    };
  }
};

template <>
struct ParamStorageTraits<NoParams> {
  /// @brief Return the no-storage `NoParams` disposition.
  /// @return Fixed, bounded facts with zero semantic bytes.
  [[nodiscard]] static consteval ParamStorageTraitsSummary summary() noexcept {
    return ParamStorageTraitsSummary{
        .owns_dynamic_storage = false,
        .fixed_object_extent = true,
        .has_portable_codec = false,
        .supports_bounded_materialization = true,
        .inline_bytes = 0,
    };
  }
};

/// @brief Describe process-local whole-aggregate operations for one exact parameter type.
///
/// The descriptor has static lifetime. Its token and operation pointers are process-local and must
/// never enter a plan, schema document, diagnostic identity, snapshot, or transport record.
/// Pointer-free plan identity and profile-selected storage strategy are compiled by later owners.
struct ParamTypeDescriptor {
  const void* local_type_token{};       ///< Exact process-local token; never portable identity.
  std::size_t object_size{};            ///< Complete C++ object size in this process.
  std::size_t object_alignment{};       ///< Required C++ object alignment in this process.
  ParamStorageTraitsSummary storage{};  ///< Intrinsic, profile-independent storage evidence.
  bool copy_construct_is_nothrow{};     ///< Whether exact aggregate copy construction cannot throw.
  bool trivially_copyable{};            ///< Whether bytewise object copying is language-defined.
  bool destroy_is_nothrow{};  ///< Whether ending the exact aggregate lifetime cannot throw.

  /// Placement-default-construct one exact aggregate on a cold path.
  ///
  /// `destination` must identify uninitialized storage of at least `object_size` bytes aligned to
  /// `object_alignment`. The operation may propagate the exact aggregate constructor exception;
  /// on success, one live exact object exists and must later be passed once to `destroy`.
  void (*default_construct)(void* destination) = nullptr;

  /// Placement-copy-construct one exact aggregate on a cold path.
  ///
  /// `destination` has the same uninitialized-size/alignment preconditions as
  /// `default_construct`; `source` must point to a live exact aggregate and may alias no part of
  /// destination storage. The operation may propagate the exact copy-constructor exception; on
  /// success, one independent live object exists and must later be passed once to `destroy`.
  void (*copy_construct)(void* destination, const void* source) = nullptr;

  /// Destroy one live exact aggregate constructed through this descriptor.
  ///
  /// `value` must be non-null, correctly aligned, and identify a live exact aggregate not already
  /// destroyed. The operation ends that lifetime without releasing the caller-owned raw storage.
  void (*destroy)(void* value) noexcept {};

  /// Validate one exact aggregate through its optional static schema.
  ///
  /// The erased input must point to a live object matching `local_type_token`. Opaque aggregates
  /// return success. The thunk is process-local and must not enter portable identity or artifacts.
  ParamValidationResult (*validate_aggregate)(const void* value) noexcept {};
};

static_assert(std::is_standard_layout_v<ParamTypeDescriptor>);
static_assert(std::is_trivially_copyable_v<ParamTypeDescriptor>);

namespace detail {

/// @brief Cancel the greatest common factor shared by two positive scale terms.
/// @param numerator Mutable numerator factor.
/// @param denominator Mutable denominator factor.
constexpr void cancel_scale_factor(std::uint64_t& numerator, std::uint64_t& denominator) noexcept {
  std::uint64_t lhs = numerator;
  std::uint64_t rhs = denominator;
  while (rhs != 0U) {
    const std::uint64_t remainder = lhs % rhs;
    lhs = rhs;
    rhs = remainder;
  }
  if (lhs != 0U) {
    numerator /= lhs;
    denominator /= lhs;
  }
}

/// @brief Scale one unsigned magnitude exactly without overflowing intermediate products.
/// @param magnitude Input magnitude.
/// @param numerator_a First numerator factor.
/// @param numerator_b Second numerator factor.
/// @param denominator_a First denominator factor.
/// @param denominator_b Second denominator factor.
/// @param limit Inclusive output limit.
/// @param output Receives the scaled magnitude on success.
/// @return \c true only when the rational result is integral and fits \p limit.
[[nodiscard]] constexpr bool scale_magnitude_exact(std::uint64_t magnitude,
                                                   std::uint64_t numerator_a,
                                                   std::uint64_t numerator_b,
                                                   std::uint64_t denominator_a,
                                                   std::uint64_t denominator_b, std::uint64_t limit,
                                                   std::uint64_t& output) noexcept {
  if (denominator_a == 0U || denominator_b == 0U) {
    return false;
  }

  cancel_scale_factor(numerator_a, denominator_a);
  cancel_scale_factor(numerator_a, denominator_b);
  cancel_scale_factor(numerator_b, denominator_a);
  cancel_scale_factor(numerator_b, denominator_b);
  cancel_scale_factor(magnitude, denominator_a);
  cancel_scale_factor(magnitude, denominator_b);
  if (denominator_a != 1U || denominator_b != 1U) {
    return false;
  }
  if (numerator_a != 0U && magnitude > limit / numerator_a) {
    return false;
  }
  magnitude *= numerator_a;
  if (numerator_b != 0U && magnitude > limit / numerator_b) {
    return false;
  }
  output = magnitude * numerator_b;
  return true;
}

/// @brief Scale one signed canonical value exactly without a wider integer extension.
/// @param value Input signed value.
/// @param numerator_a First numerator factor.
/// @param numerator_b Second numerator factor.
/// @param denominator_a First denominator factor.
/// @param denominator_b Second denominator factor.
/// @param output Receives the scaled signed value on success.
/// @return \c true only when the rational result is integral and fits an \c int64_t.
[[nodiscard]] constexpr bool scale_signed_exact(std::int64_t value, std::uint64_t numerator_a,
                                                std::uint64_t numerator_b,
                                                std::uint64_t denominator_a,
                                                std::uint64_t denominator_b,
                                                std::int64_t& output) noexcept {
  const bool negative = value < 0;
  const std::uint64_t magnitude = negative ? std::uint64_t{} - static_cast<std::uint64_t>(value)
                                           : static_cast<std::uint64_t>(value);
  const std::uint64_t positive_limit =
      static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());
  const std::uint64_t limit = negative ? positive_limit + 1U : positive_limit;
  std::uint64_t scaled{};
  if (!scale_magnitude_exact(
          magnitude, numerator_a, numerator_b, denominator_a, denominator_b, limit, scaled)) {
    return false;
  }
  if (negative && scaled == positive_limit + 1U) {
    output = std::numeric_limits<std::int64_t>::min();
  } else {
    output = negative ? -static_cast<std::int64_t>(scaled) : static_cast<std::int64_t>(scaled);
  }
  return true;
}

template <typename T, bool = std::is_enum_v<T>>
struct ParamScalarBase {
  using type = T;
};

template <typename T>
struct ParamScalarBase<T, true> {
  using type = std::underlying_type_t<T>;
};

template <typename T>
using ParamScalarBaseT = typename ParamScalarBase<T>::type;

template <typename T>
inline constexpr bool kParameterCharacterType =
    std::same_as<T, char> || std::same_as<T, wchar_t> || std::same_as<T, char8_t> ||
    std::same_as<T, char16_t> || std::same_as<T, char32_t>;

template <typename T>
inline constexpr bool kFixedParameterInteger =
    std::is_integral_v<T> && !std::same_as<T, bool> && !kParameterCharacterType<T> &&
    sizeof(T) <= sizeof(std::uint64_t) &&
    ((std::is_signed_v<T> &&
      std::numeric_limits<T>::digits + 1 == static_cast<int>(sizeof(T) * 8U)) ||
     (std::is_unsigned_v<T> && std::numeric_limits<T>::digits == static_cast<int>(sizeof(T) * 8U)));

template <typename T>
inline constexpr bool kFixedParameterFloat =
    (std::same_as<T, float> && sizeof(T) == sizeof(std::uint32_t) &&
     std::numeric_limits<T>::is_iec559) ||
    (std::same_as<T, double> && sizeof(T) == sizeof(std::uint64_t) &&
     std::numeric_limits<T>::is_iec559);

template <typename T>
struct FixedParameterDuration : std::false_type {};

template <typename Rep, typename Period>
struct FixedParameterDuration<std::chrono::duration<Rep, Period>>
    : std::bool_constant<kFixedParameterInteger<Rep> && (Period::num > 0) && (Period::den > 0)> {};

template <typename T>
inline constexpr bool kFixedParameterDuration = FixedParameterDuration<std::remove_cv_t<T>>::value;

template <typename Integer>
[[nodiscard]] constexpr bool integer_to_i64(Integer value, std::int64_t& output) noexcept {
  static_assert(kFixedParameterInteger<Integer>);
  if constexpr (std::is_signed_v<Integer>) {
    if constexpr (sizeof(Integer) > sizeof(std::int64_t)) {
      if (value < std::numeric_limits<std::int64_t>::min() ||
          value > std::numeric_limits<std::int64_t>::max()) {
        return false;
      }
    }
  } else if (static_cast<std::uint64_t>(value) >
             static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
    return false;
  }
  output = static_cast<std::int64_t>(value);
  return true;
}

template <typename Integer>
[[nodiscard]] constexpr bool i64_to_integer(std::int64_t value, Integer& output) noexcept {
  static_assert(kFixedParameterInteger<Integer>);
  if constexpr (std::is_signed_v<Integer>) {
    if (value < static_cast<std::int64_t>(std::numeric_limits<Integer>::min()) ||
        value > static_cast<std::int64_t>(std::numeric_limits<Integer>::max())) {
      return false;
    }
  } else if (value < 0 || static_cast<std::uint64_t>(value) >
                              static_cast<std::uint64_t>(std::numeric_limits<Integer>::max())) {
    return false;
  }
  output = static_cast<Integer>(value);
  return true;
}

template <typename Duration>
[[nodiscard]] constexpr bool duration_to_nanoseconds(Duration value,
                                                     std::int64_t& output) noexcept {
  using Rep = typename Duration::rep;
  using Period = typename Duration::period;
  std::int64_t count{};
  return integer_to_i64<Rep>(value.count(), count) &&
         scale_signed_exact(count,
                            static_cast<std::uint64_t>(Period::num),
                            1'000'000'000U,
                            static_cast<std::uint64_t>(Period::den),
                            1U,
                            output);
}

template <typename Duration>
[[nodiscard]] constexpr bool nanoseconds_to_duration(std::int64_t value,
                                                     Duration& output) noexcept {
  using Rep = typename Duration::rep;
  using Period = typename Duration::period;
  std::int64_t count{};
  Rep converted{};
  if (!scale_signed_exact(value,
                          static_cast<std::uint64_t>(Period::den),
                          1U,
                          static_cast<std::uint64_t>(Period::num),
                          1'000'000'000U,
                          count) ||
      !i64_to_integer<Rep>(count, converted)) {
    return false;
  }
  output = Duration{converted};
  return true;
}

template <typename T>
inline constexpr bool kFixedParameterScalar = [] {
  using Value = std::remove_cv_t<T>;
  using Base = ParamScalarBaseT<Value>;
  return std::same_as<Base, bool> || kFixedParameterInteger<Base> || kFixedParameterFloat<Base> ||
         kFixedParameterDuration<Value>;
}();

}  // namespace detail

/// @brief Store one DSO-safe fixed scalar as a stable kind, semantic width, and raw bits.
///
/// This carrier is the low-level bridge used by schema-generated field read/write thunks. It owns
/// no dynamic storage and contains no process-local type token. It is not a transport record: a
/// portable artifact or wire protocol must define its own byte order and enclosing schema.
struct ParamScalarValue {
  ParamScalarKind kind{ParamScalarKind::kNone};  ///< Semantic scalar family.
  std::uint8_t bit_width{};                      ///< Authoritative semantic width in bits.
  std::uint64_t raw_bits{};                      ///< Right-aligned exact scalar bit pattern.

  /// @brief Encode one supported fixed scalar without allocation.
  /// @tparam T Boolean, fixed-width signed/unsigned integer, enum with a supported underlying
  /// type, IEC 60559 `float`/`double`, or an integral-representation duration exactly
  /// representable as signed 64-bit nanoseconds.
  /// @param value Scalar value to encode; enum identity is reduced to its underlying scalar.
  /// @return Stable scalar carrier containing the exact value bits, or the empty carrier when a
  /// duration cannot be represented exactly in signed 64-bit nanoseconds.
  template <typename T>
    requires detail::kFixedParameterScalar<std::remove_cv_t<T>>
  [[nodiscard]] static constexpr ParamScalarValue from(T value) noexcept {
    using Value = std::remove_cv_t<T>;
    using Base = detail::ParamScalarBaseT<Value>;
    if constexpr (detail::kFixedParameterDuration<Value>) {
      std::int64_t nanoseconds{};
      if (!detail::duration_to_nanoseconds(value, nanoseconds)) {
        return {};
      }
      return ParamScalarValue{
          .kind = ParamScalarKind::kDurationNanoseconds,
          .bit_width = 64,
          .raw_bits = std::bit_cast<std::uint64_t>(nanoseconds),
      };
    } else if constexpr (std::same_as<Base, bool>) {
      return ParamScalarValue{
          .kind = ParamScalarKind::kBool,
          .bit_width = 1,
          .raw_bits = static_cast<Base>(value) ? std::uint64_t{1} : std::uint64_t{0},
      };
    } else if constexpr (std::is_integral_v<Base>) {
      const Base base = static_cast<Base>(value);
      using Unsigned = std::make_unsigned_t<Base>;
      return ParamScalarValue{
          .kind = std::is_signed_v<Base> ? ParamScalarKind::kSignedInteger
                                         : ParamScalarKind::kUnsignedInteger,
          .bit_width = static_cast<std::uint8_t>(sizeof(Base) * 8U),
          .raw_bits = static_cast<std::uint64_t>(std::bit_cast<Unsigned>(base)),
      };
    } else if constexpr (std::same_as<Base, float>) {
      return ParamScalarValue{
          .kind = ParamScalarKind::kFloatingPoint,
          .bit_width = 32,
          .raw_bits = std::bit_cast<std::uint32_t>(static_cast<Base>(value)),
      };
    } else {
      return ParamScalarValue{
          .kind = ParamScalarKind::kFloatingPoint,
          .bit_width = 64,
          .raw_bits = std::bit_cast<std::uint64_t>(static_cast<Base>(value)),
      };
    }
  }

  /// @brief Decode this carrier into one compatible fixed scalar.
  /// @tparam T Exact supported fixed scalar type.
  /// @param output Destination written only when kind and bit width match exactly.
  /// @return `true` after an exact-kind/width decode, otherwise `false` unchanged.
  template <typename T>
    requires std::same_as<T, std::remove_cv_t<T>> && detail::kFixedParameterScalar<T>
  // Keep the type-specific decode branches together so each accepted carrier is auditable.
  // NOLINTNEXTLINE(readability-function-cognitive-complexity)
  [[nodiscard]] constexpr bool try_as(T& output) const noexcept {
    using Value = std::remove_cv_t<T>;
    using Base = detail::ParamScalarBaseT<Value>;
    if constexpr (detail::kFixedParameterDuration<Value>) {
      if (kind != ParamScalarKind::kDurationNanoseconds || bit_width != 64) {
        return false;
      }
      const std::int64_t nanoseconds = std::bit_cast<std::int64_t>(raw_bits);
      return detail::nanoseconds_to_duration(nanoseconds, output);
    } else {
      Base decoded{};
      if constexpr (std::same_as<Base, bool>) {
        if (kind != ParamScalarKind::kBool || bit_width != 1 || raw_bits > 1) {
          return false;
        }
        decoded = raw_bits != 0;
      } else if constexpr (std::is_integral_v<Base>) {
        const ParamScalarKind expected = std::is_signed_v<Base> ? ParamScalarKind::kSignedInteger
                                                                : ParamScalarKind::kUnsignedInteger;
        using Unsigned = std::make_unsigned_t<Base>;
        if (kind != expected || bit_width != sizeof(Base) * 8U ||
            (sizeof(Unsigned) < sizeof(raw_bits) &&
             raw_bits > static_cast<std::uint64_t>(std::numeric_limits<Unsigned>::max()))) {
          return false;
        }
        decoded = std::bit_cast<Base>(static_cast<Unsigned>(raw_bits));
      } else if constexpr (std::same_as<Base, float>) {
        if (kind != ParamScalarKind::kFloatingPoint || bit_width != 32 ||
            raw_bits > std::numeric_limits<std::uint32_t>::max()) {
          return false;
        }
        decoded = std::bit_cast<float>(static_cast<std::uint32_t>(raw_bits));
      } else {
        if (kind != ParamScalarKind::kFloatingPoint || bit_width != 64) {
          return false;
        }
        decoded = std::bit_cast<double>(raw_bits);
      }

      if constexpr (std::is_enum_v<Value>) {
        output = static_cast<Value>(decoded);
      } else {
        output = decoded;
      }
      return true;
    }
  }

  friend constexpr bool operator==(ParamScalarValue lhs, ParamScalarValue rhs) noexcept = default;
};

static_assert(std::is_standard_layout_v<ParamScalarValue>);
static_assert(std::is_trivially_copyable_v<ParamScalarValue>);

namespace detail {

template <typename T>
inline constexpr std::byte kParameterTypeToken{};

template <typename T>
void default_construct_param(void* destination) {
  ::new (destination) T{};
}

template <typename T>
void copy_construct_param(void* destination, const void* source) {
  ::new (destination) T(*static_cast<const T*>(source));
}

template <typename T>
void destroy_param(void* value) noexcept {
  static_cast<T*>(value)->~T();
}

template <typename T>
[[nodiscard]] ParamValidationResult validate_param_aggregate(const void* value) noexcept {
  // The generic overload below keeps opaque aggregates valid. ADL selects the exact non-template
  // overload emitted beside a declared schema without creating a registry or serializable pointer.
  return holoscan_param_validate(type_tag<T>{}, value);
}

inline constexpr NoParams kNoParams{};

}  // namespace detail

/// @brief Return the static process-local descriptor for one parameter aggregate.
/// @tparam T Exact unqualified, default-initializable, copy-constructible aggregate type.
/// @return Process-lifetime descriptor containing no portable type identity.
template <typename T>
  requires ParameterAggregate<T>
[[nodiscard]] const ParamTypeDescriptor& param_type_descriptor() noexcept {
  static constexpr ParamTypeDescriptor descriptor{
      .local_type_token = &detail::kParameterTypeToken<T>,
      .object_size = sizeof(T),
      .object_alignment = alignof(T),
      .storage = ParamStorageTraits<T>::summary(),
      .copy_construct_is_nothrow = std::is_nothrow_copy_constructible_v<T>,
      .trivially_copyable = std::is_trivially_copyable_v<T>,
      .destroy_is_nothrow = std::is_nothrow_destructible_v<T>,
      .default_construct = &detail::default_construct_param<T>,
      .copy_construct = &detail::copy_construct_param<T>,
      .destroy = &detail::destroy_param<T>,
      .validate_aggregate = &detail::validate_param_aggregate<T>,
  };
  return descriptor;
}

/// @brief Return the immutable initial generation number for one parameter type.
/// @tparam T Exact parameter aggregate.
/// @return Generation zero for the initial plan-frozen value.
template <typename T>
  requires ParameterAggregate<T>
[[nodiscard]] consteval std::uint64_t initial_param_generation() noexcept {
  return 0;
}

}  // namespace holoscan
