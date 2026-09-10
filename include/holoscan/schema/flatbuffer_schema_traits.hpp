/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

#include "holoscan/core/schema_identity.hpp"

namespace holoscan::schema {

// Image and Imu are sensor payloads owned by holoscan::sensor_io, and their identities live in
// holoscan/sensor_io/sensor_schema_traits.hpp. They keep this namespace because a FlatBuffers
// namespace is part of the wire identity, so the types are still holoscan::schema::Image and
// holoscan::schema::Imu; only the component that declares them changed.
struct String;
struct Tensor;
struct TensorPacket;
struct Time;

namespace detail {

[[nodiscard]] consteval std::uint8_t schema_hex_nibble(char value) {
  if (value >= '0' && value <= '9') {
    return static_cast<std::uint8_t>(value - '0');
  }
  if (value >= 'a' && value <= 'f') {
    return static_cast<std::uint8_t>(value - 'a' + 10);
  }
  return static_cast<std::uint8_t>(value - 'A' + 10);
}

template <std::size_t Size>
[[nodiscard]] consteval std::array<std::byte, Size> schema_hex_bytes(
    // The array reference intentionally constrains callers to a compile-time string literal whose
    // bound exactly matches the requested digest width.
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays)
    const char (&text)[Size * 2U + 1U]) {
  std::array<std::byte, Size> result{};
  for (std::size_t index = 0U; index < Size; ++index) {
    result[index] = static_cast<std::byte>((schema_hex_nibble(text[index * 2U]) << 4U) |
                                           schema_hex_nibble(text[index * 2U + 1U]));
  }
  return result;
}

}  // namespace detail

/// @brief Map a generated FlatBuffer root type to complete schema metadata.
/// @tparam Root Public generated semantic root/native type.
///
/// Generated packages specialize this trait for semantic root types only. A
/// specialization provides an exact canonical identity and fixed-capacity
/// authoring defaults; verifier/parser functions and generator evidence remain
/// in `SchemaPackageDescriptor<Root>`. Buffer wrappers, transport envelopes,
/// parser types, and structurally similar application types must not specialize
/// this trait. Graph admission still validates the complete identity and final
/// capacity instead of treating these convenience defaults as negotiation.
template <typename Root>
struct FlatBufferSchemaTraits;

/// @brief Describe the SDK String root's complete identity and authoring defaults.
template <>
struct FlatBufferSchemaTraits<String> {
  using root_type = String;  ///< Public generated semantic root type.
  /// @brief Carry the exact name, version, artifact digest, and derived compact identity.
  static constexpr SchemaIdentity identity{
      .canonical_name = "holoscan.schema.String",
      .version = 1U,
      .schema_sha256 = detail::schema_hex_bytes<32>(
          "341429563d8efa47bcd77eac32f5a47c7e3bfb644f558165ae139bdfbeeb9b2f"),
      .id = SchemaId{.value = detail::schema_hex_bytes<16>("d0a5cc971394c0e0cf0c846cf1839203")},
  };
  static constexpr std::uint32_t schema_version = 1U;  ///< Exact semantic schema version.
  static constexpr std::string_view root_name =
      identity.canonical_name;  ///< Canonical globally scoped UTF-8 name.
  static constexpr std::size_t default_inline_capacity = 512U;  ///< Inline authoring default.
  static constexpr std::size_t default_descriptor_metadata_capacity =
      512U;  ///< Descriptor metadata authoring default.
};

/// @brief Describe the SDK Tensor root's complete identity and authoring defaults.
template <>
struct FlatBufferSchemaTraits<Tensor> {
  using root_type = Tensor;  ///< Public generated semantic root type.
  /// @brief Carry the exact name, version, artifact digest, and derived compact identity.
  static constexpr SchemaIdentity identity{
      .canonical_name = "holoscan.schema.Tensor",
      .version = 1U,
      .schema_sha256 = detail::schema_hex_bytes<32>(
          "5b09fe576e352937a1509a94a62c426ca23ffa5e7a4380d4a74f075980215a38"),
      .id = SchemaId{.value = detail::schema_hex_bytes<16>("9181ca1476f162c804a3d9ce39aafa0b")},
  };
  static constexpr std::uint32_t schema_version = 1U;  ///< Exact semantic schema version.
  static constexpr std::string_view root_name =
      identity.canonical_name;  ///< Canonical globally scoped UTF-8 name.
  static constexpr std::size_t default_inline_capacity = 1024U;  ///< Inline authoring default.
  static constexpr std::size_t default_descriptor_metadata_capacity =
      1024U;  ///< Descriptor metadata authoring default.
};

/// @brief Describe the SDK TensorPacket root's complete identity and authoring defaults.
template <>
struct FlatBufferSchemaTraits<TensorPacket> {
  using root_type = TensorPacket;  ///< Public generated semantic root type.
  /// @brief Carry the exact name, version, artifact digest, and derived compact identity.
  static constexpr SchemaIdentity identity{
      .canonical_name = "holoscan.schema.TensorPacket",
      .version = 1U,
      .schema_sha256 = detail::schema_hex_bytes<32>(
          "5e08faedc5f1502d71ab88d0cf582b12c77833bd40daaaf2bd3baadd3fa64d50"),
      .id = SchemaId{.value = detail::schema_hex_bytes<16>("4ed72eab76b103bb7db73b49dcf3b0f4")},
  };
  static constexpr std::uint32_t schema_version = 1U;  ///< Exact semantic schema version.
  static constexpr std::string_view root_name =
      identity.canonical_name;  ///< Canonical globally scoped UTF-8 name.
  static constexpr std::size_t default_inline_capacity = 1024U;  ///< Inline authoring default.
  static constexpr std::size_t default_descriptor_metadata_capacity =
      512U;  ///< Descriptor metadata authoring default.
};

/// @brief Describe the SDK Time root's complete identity and authoring defaults.
template <>
struct FlatBufferSchemaTraits<Time> {
  using root_type = Time;  ///< Public generated semantic root type.
  /// @brief Carry the exact name, version, artifact digest, and derived compact identity.
  static constexpr SchemaIdentity identity{
      .canonical_name = "holoscan.schema.Time",
      .version = 1U,
      .schema_sha256 = detail::schema_hex_bytes<32>(
          "c928d005a7d0ea8fed27fa58fa815e59dce599f5413a0eb8305e6f4cb156084a"),
      .id = SchemaId{.value = detail::schema_hex_bytes<16>("f741bdf4a75605048694790fe6bd885c")},
  };
  static constexpr std::uint32_t schema_version = 1U;  ///< Exact semantic schema version.
  static constexpr std::string_view root_name =
      identity.canonical_name;  ///< Canonical globally scoped UTF-8 name.
  static constexpr std::size_t default_inline_capacity = 512U;  ///< Inline authoring default.
  static constexpr std::size_t default_descriptor_metadata_capacity =
      512U;  ///< Descriptor metadata authoring default.
};

/// @brief Complete compact identity for a generated FlatBuffer semantic root.
template <typename Root>
inline constexpr SchemaId flatbuffer_schema_id_v = FlatBufferSchemaTraits<Root>::identity.id;

/// @brief Complete semantic identity for a generated FlatBuffer semantic root.
template <typename Root>
inline constexpr SchemaIdentity flatbuffer_schema_identity_v =
    FlatBufferSchemaTraits<Root>::identity;

/// @brief Exact semantic schema version for a generated FlatBuffer root.
template <typename Root>
inline constexpr std::uint32_t flatbuffer_schema_version_v =
    FlatBufferSchemaTraits<Root>::schema_version;

/// @brief Canonical globally scoped UTF-8 name for a generated FlatBuffer root.
template <typename Root>
inline constexpr std::string_view flatbuffer_schema_name_v =
    FlatBufferSchemaTraits<Root>::root_name;

/// @brief Fixed-capacity inline authoring default for a generated FlatBuffer root.
template <typename Root>
inline constexpr std::size_t flatbuffer_default_inline_capacity_v =
    FlatBufferSchemaTraits<Root>::default_inline_capacity;

/// @brief Fixed-capacity descriptor metadata default for a generated FlatBuffer root.
template <typename Root>
inline constexpr std::size_t flatbuffer_default_descriptor_metadata_capacity_v =
    FlatBufferSchemaTraits<Root>::default_descriptor_metadata_capacity;

}  // namespace holoscan::schema

namespace holoscan {

/// @brief Return the generated SDK String schema identity.
template <>
consteval SchemaIdentity schema_identity<schema::String>() {
  return schema::FlatBufferSchemaTraits<schema::String>::identity;
}

/// @brief Return the generated SDK Tensor schema identity.
template <>
consteval SchemaIdentity schema_identity<schema::Tensor>() {
  return schema::FlatBufferSchemaTraits<schema::Tensor>::identity;
}

/// @brief Return the generated SDK TensorPacket schema identity.
template <>
consteval SchemaIdentity schema_identity<schema::TensorPacket>() {
  return schema::FlatBufferSchemaTraits<schema::TensorPacket>::identity;
}

/// @brief Return the generated SDK Time schema identity.
template <>
consteval SchemaIdentity schema_identity<schema::Time>() {
  return schema::FlatBufferSchemaTraits<schema::Time>::identity;
}

}  // namespace holoscan
