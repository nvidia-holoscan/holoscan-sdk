/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <array>
#include <bit>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string_view>
#include <tuple>
#include <type_traits>

namespace holoscan {

/// @brief Supply the exact name/version identity for an admitted raw POD payload.
/// @tparam T Raw POD payload type.
///
/// The primary template is intentionally undefined. Applications explicitly
/// specialize it; Core provides no RTTI, compiler-name, size-only, or mutable-
/// registry fallback.
///
/// @warning Raw POD payloads are an experimental, expert-level escape hatch. Prefer a generated
/// FlatBuffers schema for ordinary application-defined messages. Applications using this API own
/// coordinated producer/consumer deployment, identity versioning, and binary compatibility.
template <class T>
struct ChannelTypeIdentity;

/// @brief Supply the complete ordered field layout for an admitted raw POD payload.
/// @tparam T Raw POD payload type.
///
/// The primary template is intentionally undefined. Graph authoring requires
/// the ordered fields to cover every object byte exactly once, with no gap,
/// overlap, or tail padding. Distinct C++ types may interoperate when their
/// canonical name, version, object size/alignment, and complete field layout
/// are identical. HSDK5-013 adds the canonical layout digest and fieldwise codec.
///
/// @warning Raw POD payloads provide no schema evolution or compatibility tooling. A layout or
/// semantic change requires a new \ref ChannelTypeIdentity version and coordinated rollout.
template <class T>
struct ChannelPodLayout;

/// @brief Identify one closed fixed-width POD scalar representation.
enum class PodScalarKind : std::uint8_t {
  kUInt8,    ///< Unsigned 8-bit integer field element.
  kInt8,     ///< Signed 8-bit integer field element.
  kUInt16,   ///< Unsigned 16-bit integer field element.
  kInt16,    ///< Signed 16-bit integer field element.
  kUInt32,   ///< Unsigned 32-bit integer field element.
  kInt32,    ///< Signed 32-bit integer field element.
  kUInt64,   ///< Unsigned 64-bit integer field element.
  kInt64,    ///< Signed 64-bit integer field element.
  kFloat32,  ///< IEEE-compatible 32-bit floating-point field element.
  kFloat64,  ///< IEEE-compatible 64-bit floating-point field element.
};

namespace detail {

[[nodiscard]] constexpr std::uint32_t pod_scalar_size(PodScalarKind kind) noexcept {
  switch (kind) {
    case PodScalarKind::kUInt8:
    case PodScalarKind::kInt8:
      return 1U;
    case PodScalarKind::kUInt16:
    case PodScalarKind::kInt16:
      return 2U;
    case PodScalarKind::kUInt32:
    case PodScalarKind::kInt32:
    case PodScalarKind::kFloat32:
      return 4U;
    case PodScalarKind::kUInt64:
    case PodScalarKind::kInt64:
    case PodScalarKind::kFloat64:
      return 8U;
  }
  return 0U;
}

[[nodiscard]] constexpr bool pod_identity_name_valid(std::string_view name) noexcept {
  if (name.empty() || name.size() > 255U) {
    return false;
  }
  std::size_t offset = 0U;
  while (offset < name.size()) {
    const auto lead = static_cast<std::uint8_t>(name[offset]);
    std::size_t count{};
    std::uint32_t code_point{};
    if (lead <= 0x7FU) {
      count = 1U;
      code_point = lead;
    } else if (lead >= 0xC2U && lead <= 0xDFU) {
      count = 2U;
      code_point = lead & 0x1FU;
    } else if (lead >= 0xE0U && lead <= 0xEFU) {
      count = 3U;
      code_point = lead & 0x0FU;
    } else if (lead >= 0xF0U && lead <= 0xF4U) {
      count = 4U;
      code_point = lead & 0x07U;
    } else {
      return false;
    }
    if (count > name.size() - offset) {
      return false;
    }
    for (std::size_t index = 1U; index < count; ++index) {
      const auto continuation = static_cast<std::uint8_t>(name[offset + index]);
      if ((continuation & 0xC0U) != 0x80U) {
        return false;
      }
      code_point = (code_point << 6U) | (continuation & 0x3FU);
    }
    if ((count == 3U && code_point < 0x800U) || (count == 4U && code_point < 0x10000U) ||
        code_point > 0x10FFFFU || (code_point >= 0xD800U && code_point <= 0xDFFFU) ||
        code_point <= 0x1FU || code_point == 0x7FU) {
      return false;
    }
    offset += count;
  }
  return true;
}

[[nodiscard]] constexpr bool pod_type_name_valid(std::string_view name) noexcept {
  return pod_identity_name_valid(name) && name.front() != '.' && name.back() != '.' &&
         name.find('.') != std::string_view::npos;
}

template <typename T>
concept HasChannelPodContract =
    std::is_standard_layout_v<T> && std::is_trivially_copyable_v<T> && requires {
      { ChannelTypeIdentity<T>::name } -> std::convertible_to<std::string_view>;
      { ChannelTypeIdentity<T>::version } -> std::convertible_to<std::uint32_t>;
      ChannelPodLayout<T>::fields;
    };

/// @brief Identify the first invalid property of an experimental raw-POD contract.
enum class ChannelPodContractIssue {
  kNone,                  ///< Identity and layout form one complete portable contract.
  kIdentityMemberType,    ///< Identity members do not use the required public types.
  kObjectGeometry,        ///< The object is empty or has unsupported alignment.
  kIdentityName,          ///< The canonical identity name is malformed.
  kIdentityVersion,       ///< The identity version is zero or has the wrong representation.
  kEmptyLayout,           ///< No fields were declared.
  kFieldOwner,            ///< A field belongs to a different enclosing C++ type.
  kFieldName,             ///< A field name is empty, malformed, or too long.
  kDuplicateFieldName,    ///< Two fields have the same name.
  kUnsupportedFieldType,  ///< A field type has no admitted portable scalar representation.
  kFieldOrder,            ///< Field offsets are unordered, overlapping, or contain a gap.
  kFieldMetadata,         ///< Declared scalar metadata disagrees with the C++ member type.
  kFieldExtent,           ///< A field extends beyond the enclosing object.
  kFieldMemberOffset,     ///< A pointer-to-member disagrees with the declared byte offset.
  kIncompleteCoverage,    ///< Declared fields do not cover every object byte.
};

template <typename T>
struct PodElementTraits {
  using Raw = std::remove_cv_t<T>;
  using Element = std::conditional_t<std::is_bounded_array_v<Raw>,
                                     std::remove_cv_t<std::remove_extent_t<Raw>>, Raw>;
  static constexpr bool is_array = std::is_bounded_array_v<Raw>;
  static constexpr std::uint32_t count =
      std::is_bounded_array_v<Raw> ? static_cast<std::uint32_t>(std::extent_v<Raw>) : 1U;
};

template <typename T, std::size_t N>
struct PodElementTraits<std::array<T, N>> {
  using Element = std::remove_cv_t<T>;
  static constexpr bool is_array = true;
  static constexpr std::uint32_t count = static_cast<std::uint32_t>(N);
};

template <typename T>
inline constexpr bool kSupportedPodScalar =
    std::same_as<T, std::uint8_t> || std::same_as<T, std::int8_t> ||
    std::same_as<T, std::uint16_t> || std::same_as<T, std::int16_t> ||
    std::same_as<T, std::uint32_t> || std::same_as<T, std::int32_t> ||
    std::same_as<T, std::uint64_t> || std::same_as<T, std::int64_t> ||
    (std::same_as<T, float> && std::numeric_limits<float>::is_iec559 && sizeof(float) == 4) ||
    (std::same_as<T, double> && std::numeric_limits<double>::is_iec559 && sizeof(double) == 8);

/// Verify that a pointer-to-member selects the bytes declared by its companion offset.
///
/// C++ has no portable operation that converts a pointer-to-member into an integer offset. Build a
/// valid marker object representation instead: only the claimed field bytes contain the value one,
/// then access the member and require every selected element to observe that marker. This remains
/// a cold constant-expression check and works for admitted scalar, C-array, and `std::array`
/// members without default-constructing the enclosing payload.
template <typename Class, typename Field>
[[nodiscard]] constexpr bool pod_member_matches_offset(Field Class::* member,
                                                       std::size_t byte_offset) {
  if (member == nullptr) {
    return false;
  }
  using Traits = PodElementTraits<Field>;
  using Element = typename Traits::Element;
  const Element marker = static_cast<Element>(1);
  const auto marker_bytes = std::bit_cast<std::array<std::byte, sizeof(Element)>>(marker);
  std::array<std::byte, sizeof(Class)> object_bytes{};
  for (std::size_t element = 0U; element < Traits::count; ++element) {
    for (std::size_t index = 0U; index < marker_bytes.size(); ++index) {
      object_bytes[byte_offset + element * marker_bytes.size() + index] = marker_bytes[index];
    }
  }
  const Class object = std::bit_cast<Class>(object_bytes);
  if constexpr (Traits::is_array) {
    for (std::size_t index = 0U; index < Traits::count; ++index) {
      if ((object.*member)[index] != marker) {
        return false;
      }
    }
    return true;
  } else {
    return object.*member == marker;
  }
}

template <typename T>
[[nodiscard]] consteval PodScalarKind pod_scalar_kind() {
  static_assert(kSupportedPodScalar<T>,
                "POD fields require fixed-width integer, float, double, or fixed arrays");
  if constexpr (std::same_as<T, std::uint8_t>)
    return PodScalarKind::kUInt8;
  if constexpr (std::same_as<T, std::int8_t>)
    return PodScalarKind::kInt8;
  if constexpr (std::same_as<T, std::uint16_t>)
    return PodScalarKind::kUInt16;
  if constexpr (std::same_as<T, std::int16_t>)
    return PodScalarKind::kInt16;
  if constexpr (std::same_as<T, std::uint32_t>)
    return PodScalarKind::kUInt32;
  if constexpr (std::same_as<T, std::int32_t>)
    return PodScalarKind::kInt32;
  if constexpr (std::same_as<T, std::uint64_t>)
    return PodScalarKind::kUInt64;
  if constexpr (std::same_as<T, std::int64_t>)
    return PodScalarKind::kInt64;
  if constexpr (std::same_as<T, float>)
    return PodScalarKind::kFloat32;
  return PodScalarKind::kFloat64;
}

}  // namespace detail

/// @brief Describe one field in a specialization-owned raw POD layout.
/// @tparam Class Complete standard-layout payload type.
/// @tparam Field Exact member type, optionally a fixed array.
template <class Class, class Field>
struct PodFieldDescriptor {
  using class_type = Class;  ///< Complete object type that owns the represented member.
  using field_type = Field;  ///< Exact scalar or fixed-array member type.

  std::string_view name;        ///< Exact UTF-8 field name with static storage.
  Field Class::* member;        ///< Compile-time field accessor; never serialized as identity.
  std::uint32_t byte_offset;    ///< Actual object byte offset supplied with `offsetof`.
  PodScalarKind scalar_kind;    ///< Closed fixed-width scalar representation.
  std::uint32_t element_count;  ///< One for scalars or the exact fixed-array length.
};

/// @brief Create one constexpr POD field descriptor.
/// @tparam Class Complete payload type.
/// @tparam Field Exact field type.
/// @param name Exact field name retained by the specialization.
/// @param member Pointer to the represented member.
/// @param byte_offset Actual `offsetof(Class, field)` value.
/// @return Complete descriptor used by Graph capture and exact coverage validation.
/// @warning This helper belongs to the experimental raw-POD escape hatch. Prefer generated
/// FlatBuffers schemas for ordinary application messages.
template <class Class, class Field>
[[nodiscard]] consteval auto pod_field(std::string_view name, Field Class::* member,
                                       std::size_t byte_offset) {
  using Traits = detail::PodElementTraits<Field>;
  using Element = typename Traits::Element;
  static_assert(
      std::is_standard_layout_v<Class>,
      "experimental raw POD payload types must be standard-layout; use a FlatBuffers schema "
      "for classes with constructors, inheritance, or other nonstandard object layout");
  static_assert(
      std::is_trivially_copyable_v<Class>,
      "experimental raw POD payload types must be trivially copyable; use a FlatBuffers schema "
      "for owning or dynamically sized fields");
  static_assert(std::copy_constructible<Class>,
                "experimental raw POD payload types must be copy constructible");
  static_assert(
      detail::kSupportedPodScalar<Element>,
      "experimental raw POD fields must be fixed-width integers, IEEE float/double, or fixed "
      "arrays of those types; bool, enums, pointers, nested structs, and strings are unsupported");
  if (byte_offset > UINT32_MAX) {
    throw "POD field byte offset exceeds the baseline uint32 representation";
  }
  return PodFieldDescriptor<Class, Field>{name,
                                          member,
                                          static_cast<std::uint32_t>(byte_offset),
                                          detail::pod_scalar_kind<Element>(),
                                          Traits::count};
}

/// @brief Collect ordered constexpr POD field descriptors.
/// @tparam Fields Exact descriptor types produced by \ref pod_field.
/// @param fields Descriptors in increasing byte-offset order.
/// @return Tuple retained by `ChannelPodLayout<T>::fields`.
/// @warning This helper belongs to the experimental raw-POD escape hatch. The application owns
/// identity versioning and coordinated producer/consumer deployment.
template <class... Fields>
[[nodiscard]] consteval auto pod_fields(Fields... fields) {
  return std::tuple<Fields...>{fields...};
}

namespace detail {

template <typename T, std::size_t FieldCount, typename Descriptor>
[[nodiscard]] constexpr ChannelPodContractIssue append_channel_pod_field(
    const Descriptor& field, std::array<std::string_view, FieldCount>& field_names,
    std::size_t& field_index, std::size_t& next_offset) {
  using Field = typename Descriptor::field_type;
  using Traits = PodElementTraits<Field>;
  using Element = typename Traits::Element;
  if constexpr (!kSupportedPodScalar<Element>) {
    return ChannelPodContractIssue::kUnsupportedFieldType;
  } else if constexpr (!std::same_as<typename Descriptor::class_type, T>) {
    return ChannelPodContractIssue::kFieldOwner;
  } else {
    if (field.name.empty() || field.name.size() > 255U || !pod_identity_name_valid(field.name)) {
      return ChannelPodContractIssue::kFieldName;
    }
    for (std::size_t prior = 0U; prior < field_index; ++prior) {
      if (field_names[prior] == field.name) {
        return ChannelPodContractIssue::kDuplicateFieldName;
      }
    }
    if (field.byte_offset != next_offset) {
      return ChannelPodContractIssue::kFieldOrder;
    }
    if (field.scalar_kind != pod_scalar_kind<Element>() || field.element_count != Traits::count ||
        field.element_count == 0U || pod_scalar_size(field.scalar_kind) != sizeof(Element)) {
      return ChannelPodContractIssue::kFieldMetadata;
    }
    const std::size_t byte_count = sizeof(Element) * static_cast<std::size_t>(Traits::count);
    if (next_offset > sizeof(T) || byte_count > sizeof(T) - next_offset) {
      return ChannelPodContractIssue::kFieldExtent;
    }
    if (!pod_member_matches_offset<T>(field.member, field.byte_offset)) {
      return ChannelPodContractIssue::kFieldMemberOffset;
    }
    field_names[field_index++] = field.name;
    next_offset += byte_count;
    return ChannelPodContractIssue::kNone;
  }
}

/// @brief Diagnose one compile-time raw-POD contract before a codec is instantiated.
/// @tparam T Exact application payload type.
/// @return The first identity or layout problem, or \ref ChannelPodContractIssue::kNone.
///
/// This cold compile-time check rejects gaps, overlap, tail padding, altered descriptor metadata,
/// unsupported scalar representations, and object geometry outside the version-one catalog
/// representation. Runtime catalog parsing repeats the equivalent checks for untrusted artifacts.
template <typename T>
  requires HasChannelPodContract<T>
[[nodiscard]] consteval ChannelPodContractIssue channel_pod_contract_issue() {
  if constexpr (!std::same_as<std::remove_cv_t<decltype(ChannelTypeIdentity<T>::name)>,
                              std::string_view> ||
                !std::same_as<std::remove_cv_t<decltype(ChannelTypeIdentity<T>::version)>,
                              std::uint32_t>) {
    return ChannelPodContractIssue::kIdentityMemberType;
  }
  if constexpr (sizeof(T) > std::numeric_limits<std::uint32_t>::max() ||
                alignof(T) > std::numeric_limits<std::uint32_t>::max()) {
    return ChannelPodContractIssue::kObjectGeometry;
  }
  if (!pod_type_name_valid(std::string_view{ChannelTypeIdentity<T>::name})) {
    return ChannelPodContractIssue::kIdentityName;
  }
  if (ChannelTypeIdentity<T>::version == 0U) {
    return ChannelPodContractIssue::kIdentityVersion;
  }

  using Fields = std::remove_cv_t<decltype(ChannelPodLayout<T>::fields)>;
  if constexpr (std::tuple_size_v<Fields> == 0U) {
    return ChannelPodContractIssue::kEmptyLayout;
  }

  constexpr std::size_t kFieldCount = std::tuple_size_v<Fields>;
  std::array<std::string_view, kFieldCount> field_names{};
  std::size_t field_index{};
  std::size_t next_offset{};
  ChannelPodContractIssue issue = ChannelPodContractIssue::kNone;
  std::apply(
      [&field_names, &field_index, &next_offset, &issue](const auto&... field) {
        const auto append_field = [&field_names, &field_index, &next_offset, &issue](
                                      const auto& descriptor) {
          if (issue != ChannelPodContractIssue::kNone) {
            return;
          }
          issue = append_channel_pod_field<T>(descriptor, field_names, field_index, next_offset);
        };
        (append_field(field), ...);
      },
      ChannelPodLayout<T>::fields);
  if (issue != ChannelPodContractIssue::kNone) {
    return issue;
  }
  return next_offset == sizeof(T) ? ChannelPodContractIssue::kNone
                                  : ChannelPodContractIssue::kIncompleteCoverage;
}

/// @brief Verify one complete experimental raw-POD contract.
/// @tparam T Exact application payload type.
/// @return \c true when the contract is complete; otherwise compilation fails with a focused
/// diagnostic.
template <typename T>
  requires HasChannelPodContract<T>
[[nodiscard]] consteval bool assert_complete_channel_pod_contract() {
  constexpr ChannelPodContractIssue kIssue = channel_pod_contract_issue<T>();
  static_assert(
      kIssue != ChannelPodContractIssue::kIdentityMemberType,
      "experimental raw POD identity requires name to be exactly std::string_view and version "
      "to be exactly std::uint32_t");
  static_assert(kIssue != ChannelPodContractIssue::kObjectGeometry,
                "experimental raw POD object size or alignment exceeds the wire contract");
  static_assert(
      kIssue != ChannelPodContractIssue::kIdentityName,
      "experimental raw POD identity name must be dotted, nonempty valid UTF-8 with 255 bytes "
      "or fewer, and must not begin or end with a dot");
  static_assert(kIssue != ChannelPodContractIssue::kIdentityVersion,
                "experimental raw POD identity version must be a nonzero std::uint32_t");
  static_assert(kIssue != ChannelPodContractIssue::kEmptyLayout,
                "experimental raw POD layout must declare at least one field");
  static_assert(kIssue != ChannelPodContractIssue::kFieldOwner,
                "experimental raw POD layout contains a field owned by another C++ type");
  static_assert(
      kIssue != ChannelPodContractIssue::kFieldName,
      "experimental raw POD field names must be nonempty valid UTF-8 with 255 bytes or fewer");
  static_assert(kIssue != ChannelPodContractIssue::kDuplicateFieldName,
                "experimental raw POD field names must be unique");
  static_assert(
      kIssue != ChannelPodContractIssue::kUnsupportedFieldType,
      "experimental raw POD fields must be fixed-width integers, IEEE float/double, or fixed "
      "arrays of those types; bool, enums, pointers, nested structs, and strings are unsupported");
  static_assert(
      kIssue != ChannelPodContractIssue::kFieldOrder,
      "experimental raw POD fields must appear in exact byte-offset order without gaps or overlap");
  static_assert(kIssue != ChannelPodContractIssue::kFieldMetadata,
                "experimental raw POD field metadata does not match its declared C++ member type");
  static_assert(kIssue != ChannelPodContractIssue::kFieldExtent,
                "experimental raw POD field extent exceeds the enclosing object");
  static_assert(kIssue != ChannelPodContractIssue::kFieldMemberOffset,
                "experimental raw POD field member does not match its declared offsetof value");
  static_assert(
      kIssue != ChannelPodContractIssue::kIncompleteCoverage,
      "experimental raw POD fields must cover every object byte; implicit and tail padding are "
      "unsupported");
  return true;
}

template <typename T>
[[nodiscard]] consteval bool assert_channel_pod_contract_if_declared() {
  if constexpr (HasChannelPodContract<T>) {
    return assert_complete_channel_pod_contract<T>();
  }
  return true;
}

/// @brief Verify one compile-time raw-POD contract before a codec is instantiated.
/// @tparam T Exact application payload type.
/// @return Whether identity and ordered fields describe every object byte exactly once.
template <typename T>
  requires HasChannelPodContract<T>
[[nodiscard]] consteval bool complete_channel_pod_contract() {
  return channel_pod_contract_issue<T>() == ChannelPodContractIssue::kNone;
}

}  // namespace detail

}  // namespace holoscan
