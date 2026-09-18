/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <limits>
#include <span>
#include <string_view>
#include <type_traits>

namespace holoscan {

namespace schema {
template <typename Root>
struct FlatBufferSchemaBinding;
}  // namespace schema

/// @brief Carry the compact semantic identity of one generated schema.
///
/// Every 128-bit pattern, including all zero bits, is a valid identity. Absence
/// is represented by absence of the enclosing value, never by a sentinel in
/// this type. The bytes have no numeric interpretation and are independent of
/// C++ RTTI, object addresses, and executable parser or verifier functions.
struct SchemaId {
  std::array<std::byte, 16> value{};  ///< Complete portable 128-bit identity bytes.

  /// @brief Compare schema identities by all 16 bytes.
  /// @param lhs Left identity.
  /// @param rhs Right identity.
  /// @return \c true when every identity byte is equal.
  friend constexpr bool operator==(SchemaId lhs, SchemaId rhs) noexcept = default;
};

/// @brief Carry the exact four-byte FlatBuffers root file identifier.
///
/// The identifier is an independent in-buffer type check. It is deliberately
/// excluded from schema-family ID derivation and is not authentication.
struct FlatBufferFileIdentifier {
  std::array<char, 4> value{};  ///< Exact four identifier bytes.

  /// @brief Compare all four identifier bytes.
  friend constexpr bool operator==(FlatBufferFileIdentifier,
                                   FlatBufferFileIdentifier) noexcept = default;
};

/// @brief Identify one author-governed FlatBuffers compatibility family.
///
/// The canonical name and compatibility epoch define the default compact ID.
/// Compatible schema revisions retain both values. An intentionally
/// incompatible revision changes the epoch or family name. Generated
/// specializations provide storage with static lifetime for the borrowed name.
struct SchemaFamilyIdentity {
  SchemaId id{};                         ///< Stable compact family identity.
  std::string_view canonical_name{};     ///< Stable fully qualified FlatBuffers root name.
  std::uint32_t compatibility_epoch{1};  ///< Positive intentional wire-break epoch.

  /// @brief Compare all family identity facts.
  friend constexpr bool operator==(SchemaFamilyIdentity, SchemaFamilyIdentity) noexcept = default;
};

/// @brief Describe one stable FlatBuffers family and its independent buffer identifier.
///
/// The canonical name storage supplied by a specialization must have static
/// lifetime because this value borrows it as a \c std::string_view. Exact BFBS
/// artifact fingerprints do not participate in this runtime identity.
struct SchemaIdentity {
  std::string_view canonical_name{};           ///< Validated globally scoped UTF-8 family name.
  std::uint32_t compatibility_epoch{1U};       ///< Positive intentional wire-break epoch.
  SchemaId family_id{};                        ///< Stable compact compatibility-family identity.
  FlatBufferFileIdentifier file_identifier{};  ///< Exact independent in-buffer type check.

  /// @brief Compare every stable identity and buffer-type fact.
  friend constexpr bool operator==(SchemaIdentity, SchemaIdentity) noexcept = default;
};

/// @brief Return the generated semantic identity for \p T.
///
/// Generated roots obtain this through their single schema binding; generated owning types
/// delegate through their FlatBuffers `TableType` association. The unspecialized template is
/// deleted so arbitrary classes and transport wrappers do not acquire schema identity.
///
/// @tparam T Unqualified semantic generated root type.
/// @return Complete compile-time semantic schema identity.
template <typename T>
consteval SchemaIdentity schema_identity() = delete;

/// @brief Return the generated compatibility-family identity for \p T.
///
/// Generated roots obtain this through their single schema binding, while owning native types
/// delegate through their generated root association. Artifact fingerprints and codec operations
/// are deliberately separate.
///
/// @tparam T Unqualified generated root or owning native type.
/// @return Compile-time schema family identity.
template <typename T>
consteval SchemaFamilyIdentity schema_family_identity() = delete;

namespace detail {

// This dependency-free SHA-256 implementation is shared by generated-schema
// package validation and family identity derivation. Callers add only the
// byte-exact inputs defined by the schema contract; C++ object representation,
// addresses, RTTI, and host byte order never enter the digest.
class SchemaSha256Builder {
 public:
  constexpr void add_byte(std::uint8_t byte) noexcept {
    buffer_[buffer_size_++] = byte;
    ++total_bytes_;
    if (buffer_size_ == buffer_.size()) {
      transform();
      buffer_size_ = 0U;
    }
  }

  constexpr void add(std::span<const std::byte> bytes) noexcept {
    for (const std::byte byte : bytes) {
      add_byte(std::to_integer<std::uint8_t>(byte));
    }
  }

  constexpr void add(std::string_view bytes) noexcept {
    for (const char byte : bytes) {
      add_byte(static_cast<std::uint8_t>(byte));
    }
  }

  constexpr void add_u32_be(std::uint32_t value) noexcept {
    for (std::size_t index = 0U; index < 4U; ++index) {
      const auto shift = static_cast<unsigned>((3U - index) * 8U);
      add_byte(static_cast<std::uint8_t>(value >> shift));
    }
  }

  [[nodiscard]] constexpr std::array<std::byte, 32> finish() const noexcept {
    SchemaSha256Builder copy = *this;
    const std::uint64_t bit_length = copy.total_bytes_ * 8U;
    copy.buffer_[copy.buffer_size_++] = 0x80U;
    if (copy.buffer_size_ > 56U) {
      while (copy.buffer_size_ < copy.buffer_.size()) {
        copy.buffer_[copy.buffer_size_++] = 0U;
      }
      copy.transform();
      copy.buffer_size_ = 0U;
    }
    while (copy.buffer_size_ < 56U) {
      copy.buffer_[copy.buffer_size_++] = 0U;
    }
    for (std::size_t index = 0U; index < sizeof(bit_length); ++index) {
      copy.buffer_[63U - index] =
          static_cast<std::uint8_t>(bit_length >> static_cast<unsigned>(index * 8U));
    }
    copy.transform();

    std::array<std::byte, 32> result{};
    for (std::size_t word = 0U; word < copy.state_.size(); ++word) {
      for (std::size_t byte = 0U; byte < sizeof(std::uint32_t); ++byte) {
        result[word * sizeof(std::uint32_t) + byte] =
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
    for (std::size_t index = 0U; index < 16U; ++index) {
      const std::size_t offset = index * 4U;
      words[index] = (static_cast<std::uint32_t>(buffer_[offset]) << 24U) |
                     (static_cast<std::uint32_t>(buffer_[offset + 1U]) << 16U) |
                     (static_cast<std::uint32_t>(buffer_[offset + 2U]) << 8U) |
                     static_cast<std::uint32_t>(buffer_[offset + 3U]);
    }
    for (std::size_t index = 16U; index < words.size(); ++index) {
      const std::uint32_t sigma0 = rotate_right(words[index - 15U], 7U) ^
                                   rotate_right(words[index - 15U], 18U) ^
                                   (words[index - 15U] >> 3U);
      const std::uint32_t sigma1 = rotate_right(words[index - 2U], 17U) ^
                                   rotate_right(words[index - 2U], 19U) ^
                                   (words[index - 2U] >> 10U);
      words[index] = words[index - 16U] + sigma0 + words[index - 7U] + sigma1;
    }

    std::uint32_t a = state_[0];
    std::uint32_t b = state_[1];
    std::uint32_t c = state_[2];
    std::uint32_t d = state_[3];
    std::uint32_t e = state_[4];
    std::uint32_t f = state_[5];
    std::uint32_t g = state_[6];
    std::uint32_t h = state_[7];
    for (std::size_t index = 0U; index < words.size(); ++index) {
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

template <typename T>
concept HasFlatBufferSchemaRootBinding = requires {
  typename schema::FlatBufferSchemaBinding<T>::root_type;
  typename schema::FlatBufferSchemaBinding<T>::native_type;
  { schema::FlatBufferSchemaBinding<T>::identity } -> std::same_as<const SchemaIdentity&>;
  {
    schema::FlatBufferSchemaBinding<T>::family_identity
  } -> std::same_as<const SchemaFamilyIdentity&>;
} && std::same_as<typename schema::FlatBufferSchemaBinding<T>::root_type, T>;

template <typename T>
using FlatBufferSchemaNativeType =
    typename schema::FlatBufferSchemaBinding<typename T::TableType>::native_type;

template <typename T>
concept HasFlatBufferSchemaNativeBinding = requires {
  typename T::TableType;
  requires HasFlatBufferSchemaRootBinding<typename T::TableType>;
  requires std::same_as<FlatBufferSchemaNativeType<T>, T>;
};

}  // namespace detail

/// @brief Return identity from the one generated binding for a FlatBuffers root.
template <typename Root>
  requires detail::HasFlatBufferSchemaRootBinding<Root>
[[nodiscard]] consteval SchemaIdentity schema_identity() {
  return schema::FlatBufferSchemaBinding<Root>::identity;
}

/// @brief Return a generated owning type's identity through its root binding.
template <typename Native>
  requires detail::HasFlatBufferSchemaNativeBinding<Native>
[[nodiscard]] consteval SchemaIdentity schema_identity() {
  return schema::FlatBufferSchemaBinding<typename Native::TableType>::identity;
}

/// @brief Return family identity from the one generated binding for a FlatBuffers root.
template <typename Root>
  requires detail::HasFlatBufferSchemaRootBinding<Root>
[[nodiscard]] consteval SchemaFamilyIdentity schema_family_identity() {
  return schema::FlatBufferSchemaBinding<Root>::family_identity;
}

/// @brief Return a generated owning type's family identity through its root binding.
template <typename Native>
  requires detail::HasFlatBufferSchemaNativeBinding<Native>
[[nodiscard]] consteval SchemaFamilyIdentity schema_family_identity() {
  return schema::FlatBufferSchemaBinding<typename Native::TableType>::family_identity;
}

namespace detail {

template <typename T>
concept HasSchemaIdentity = requires {
  { schema_identity<T>() } -> std::same_as<SchemaIdentity>;
};

template <typename T>
concept HasSchemaFamilyIdentity = requires {
  { schema_family_identity<T>() } -> std::same_as<SchemaFamilyIdentity>;
};

template <typename T>
concept HasGeneratedSchemaName = requires {
  { T::GetFullyQualifiedName() } -> std::convertible_to<const char*>;
};

}  // namespace detail

/// @brief Derive the compact v2 schema-family identifier.
/// @param canonical_name Stable fully qualified FlatBuffers root name.
/// @param compatibility_epoch Positive intentional wire-break epoch.
/// @return First 16 bytes of the domain-separated SHA-256 derivation.
///
/// The preimage is exactly `"holoscan.schema-family/v2\\0"`, the unmodified
/// name bytes, and the epoch as unsigned 32-bit big-endian. Artifact digests,
/// generated C++ spellings, and per-revision counters never enter this value.
[[nodiscard]] constexpr SchemaId derive_schema_family_id(
    std::string_view canonical_name, std::uint32_t compatibility_epoch) noexcept {
  detail::SchemaSha256Builder hash;
  hash.add("holoscan.schema-family/v2");
  hash.add_byte(0U);
  hash.add(canonical_name);
  hash.add_u32_be(compatibility_epoch);
  const auto digest = hash.finish();

  SchemaId result;
  for (std::size_t index = 0U; index < result.value.size(); ++index) {
    result.value[index] = digest[index];
  }
  return result;
}

/// @brief Build the default derived identity for a stable family name.
/// @param canonical_name Stable fully qualified FlatBuffers root name.
/// @param compatibility_epoch Positive intentional wire-break epoch.
/// @return Family identity whose ID is derived from name and epoch.
[[nodiscard]] consteval SchemaFamilyIdentity make_schema_family_identity(
    std::string_view canonical_name, std::uint32_t compatibility_epoch = 1U) {
  if (canonical_name.empty() || compatibility_epoch == 0U) {
    std::abort();
  }
  return SchemaFamilyIdentity{
      .id = derive_schema_family_id(canonical_name, compatibility_epoch),
      .canonical_name = canonical_name,
      .compatibility_epoch = compatibility_epoch,
  };
}

/// @brief Build an exact four-byte FlatBuffers root file identifier.
/// @tparam Size Character-array width, including the trailing null byte.
/// @param identifier Four identifier bytes followed by a null terminator.
/// @return Exact identifier bytes without the terminator.
template <std::size_t Size>
[[nodiscard]] consteval FlatBufferFileIdentifier make_flatbuffer_file_identifier(
    const char (&identifier)[Size]) {
  static_assert(Size == 5U, "FlatBuffers file_identifier must contain exactly four bytes");
  if (identifier[4U] != '\0') {
    std::abort();
  }
  return FlatBufferFileIdentifier{
      .value = {identifier[0U], identifier[1U], identifier[2U], identifier[3U]},
  };
}

/// @brief Return a null-terminated FlatBuffers identifier for generated APIs.
/// @param identifier Exact four-byte identifier.
/// @return Five-byte array containing the identifier and a trailing null.
[[nodiscard]] constexpr std::array<char, 5> flatbuffer_file_identifier_c_string(
    FlatBufferFileIdentifier identifier) noexcept {
  return {
      identifier.value[0U], identifier.value[1U], identifier.value[2U], identifier.value[3U], '\0'};
}

/// @brief Return whether an identifier contains four printable nonzero ASCII bytes.
/// @param identifier Exact four-byte identifier.
/// @return c true when every byte is suitable for a FlatBuffers schema declaration.
[[nodiscard]] constexpr bool valid_flatbuffer_file_identifier(
    FlatBufferFileIdentifier identifier) noexcept {
  for (const char byte : identifier.value) {
    const auto value = static_cast<unsigned char>(byte);
    if (value < 0x20U || value > 0x7eU) {
      return false;
    }
  }
  return true;
}

/// @brief Compare every stable schema identity and buffer-type fact.
[[nodiscard]] constexpr bool schema_identity_equal(const SchemaIdentity& left,
                                                   const SchemaIdentity& right) noexcept {
  return left == right;
}

/// @brief Project the compatibility-family view from a complete schema identity.
[[nodiscard]] constexpr SchemaFamilyIdentity schema_family_identity_of(
    const SchemaIdentity& identity) noexcept {
  return SchemaFamilyIdentity{
      .id = identity.family_id,
      .canonical_name = identity.canonical_name,
      .compatibility_epoch = identity.compatibility_epoch,
  };
}

namespace detail {

// Keep the UTF-8 decoder in one bounded pass so schema names cannot enter diagnostics with
// overlong, surrogate, or out-of-range encodings.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] constexpr bool valid_schema_utf8(std::string_view value,
                                               bool reject_ascii_controls) noexcept {
  std::size_t index = 0U;
  while (index < value.size()) {
    const auto first = static_cast<unsigned char>(value[index++]);
    if (first <= 0x7FU) {
      if (reject_ascii_controls && (first < 0x20U || first == 0x7FU)) {
        return false;
      }
      continue;
    }
    std::uint32_t code_point{};
    std::size_t continuation_count{};
    if (first >= 0xC2U && first <= 0xDFU) {
      code_point = first & 0x1FU;
      continuation_count = 1U;
    } else if (first >= 0xE0U && first <= 0xEFU) {
      code_point = first & 0x0FU;
      continuation_count = 2U;
    } else if (first >= 0xF0U && first <= 0xF4U) {
      code_point = first & 0x07U;
      continuation_count = 3U;
    } else {
      return false;
    }
    if (index + continuation_count > value.size()) {
      return false;
    }
    for (std::size_t count = 0U; count < continuation_count; ++count) {
      const auto next = static_cast<unsigned char>(value[index++]);
      if ((next & 0xC0U) != 0x80U) {
        return false;
      }
      code_point = (code_point << 6U) | (next & 0x3FU);
    }
    if ((continuation_count == 2U && code_point < 0x800U) ||
        (continuation_count == 3U && code_point < 0x10000U) ||
        (code_point >= 0xD800U && code_point <= 0xDFFFU) || code_point > 0x10FFFFU) {
      return false;
    }
  }
  return true;
}

}  // namespace detail

/// @brief Return whether a canonical schema name is globally scoped valid UTF-8.
[[nodiscard]] constexpr bool valid_schema_name(std::string_view value) noexcept {
  return !value.empty() && value.size() <= std::numeric_limits<std::uint32_t>::max() &&
         value.front() != '.' && value.back() != '.' && value.find('.') != std::string_view::npos &&
         detail::valid_schema_utf8(value, true);
}

/// @brief Validate one complete borrowed schema identity.
[[nodiscard]] constexpr bool valid_schema_identity(const SchemaIdentity& identity) noexcept {
  return valid_schema_name(identity.canonical_name) && identity.compatibility_epoch != 0U &&
         derive_schema_family_id(identity.canonical_name, identity.compatibility_epoch) ==
             identity.family_id &&
         valid_flatbuffer_file_identifier(identity.file_identifier);
}

/// @brief Combine stable family identity with its independent FlatBuffers identifier.
/// @param family_identity Stable name, compatibility epoch, and derived family ID.
/// @param file_identifier Exact four-byte identifier stamped in every buffer.
/// @return Complete stable schema identity used by records and attachment.
[[nodiscard]] consteval SchemaIdentity make_schema_identity(
    SchemaFamilyIdentity family_identity, FlatBufferFileIdentifier file_identifier) {
  if (!valid_flatbuffer_file_identifier(file_identifier)) {
    std::abort();
  }
  return SchemaIdentity{
      .canonical_name = family_identity.canonical_name,
      .compatibility_epoch = family_identity.compatibility_epoch,
      .family_id = family_identity.id,
      .file_identifier = file_identifier,
  };
}

/// @brief Build the default identity from a generated FlatBuffers root name.
/// @tparam Root Generated root exposing `GetFullyQualifiedName()`.
/// @param compatibility_epoch Positive intentional wire-break epoch.
/// @return Family identity derived directly from generated root facts.
template <typename Root>
  requires detail::HasGeneratedSchemaName<Root>
[[nodiscard]] consteval SchemaFamilyIdentity make_schema_family_identity(
    std::uint32_t compatibility_epoch = 1U) {
  return make_schema_family_identity(std::string_view(Root::GetFullyQualifiedName()),
                                     compatibility_epoch);
}

static_assert(sizeof(SchemaId) == 16);
static_assert(sizeof(FlatBufferFileIdentifier) == 4);
static_assert(alignof(SchemaId) == alignof(std::byte));
static_assert(std::is_standard_layout_v<SchemaId>);
static_assert(std::is_trivially_copyable_v<SchemaId>);
static_assert(std::is_standard_layout_v<FlatBufferFileIdentifier>);
static_assert(std::is_trivially_copyable_v<FlatBufferFileIdentifier>);
static_assert(std::is_standard_layout_v<SchemaFamilyIdentity>);
static_assert(std::is_trivially_copyable_v<SchemaFamilyIdentity>);
}  // namespace holoscan

namespace std {

/// @brief Hash a \ref holoscan::SchemaId by all of its portable identity bytes.
template <>
struct hash<holoscan::SchemaId> {
  /// @brief Return a hash suitable for equality-based containers.
  /// @param id Schema identity to hash.
  /// @return Hash derived from all 16 identity bytes.
  [[nodiscard]] std::size_t operator()(holoscan::SchemaId id) const noexcept {
    // FNV-1a keeps this implementation independent of object padding and host
    // byte order. Hash values are process-container helpers, not durable IDs.
    std::size_t result = sizeof(std::size_t) == 8U
                             ? static_cast<std::size_t>(14695981039346656037ULL)
                             : static_cast<std::size_t>(2166136261U);
    const std::size_t prime = sizeof(std::size_t) == 8U ? static_cast<std::size_t>(1099511628211ULL)
                                                        : static_cast<std::size_t>(16777619U);
    for (const std::byte byte : id.value) {
      result ^= static_cast<std::size_t>(std::to_integer<std::uint8_t>(byte));
      result *= prime;
    }
    return result;
  }
};

}  // namespace std
