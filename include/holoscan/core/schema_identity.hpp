/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <string_view>
#include <type_traits>

namespace holoscan {

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

/// @brief Describe the complete compile-time semantic identity of a schema root.
///
/// Generated schema packages supply this value for the semantic root/native
/// object exposed to application code. The identity does not carry verifier,
/// parser, artifact-lifetime, or transport-package authority. The canonical
/// name and digest storage supplied by a specialization must have static
/// lifetime because this value borrows the name as a \c std::string_view.
struct SchemaIdentity {
  std::string_view canonical_name{};          ///< Validated globally scoped UTF-8 name.
  std::uint32_t version{};                    ///< Explicit nonzero semantic schema version.
  std::array<std::byte, 32> schema_sha256{};  ///< Raw SHA-256 bytes of the canonical artifact.
  SchemaId id{};                              ///< Compact identity derived by the schema package.
};

/// @brief Return the generated semantic identity for \p T.
///
/// Generated packages explicitly specialize this function for their public
/// semantic root/native object type. The unspecialized template is deleted so
/// arbitrary copyable classes, transport wrappers, and parser artifacts do not
/// acquire schema identity by structural inference.
///
/// @tparam T Unqualified semantic generated root type.
/// @return Complete compile-time semantic schema identity.
template <typename T>
consteval SchemaIdentity schema_identity() = delete;

namespace detail {

// This dependency-free SHA-256 implementation is shared by generated-schema
// package validation and compact identity derivation. Callers add only the
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
concept HasSchemaIdentity = requires {
  { schema_identity<T>() } -> std::same_as<SchemaIdentity>;
};

}  // namespace detail

/// @brief Compute SHA-256 over exact generated binary-schema artifact bytes.
/// @param bytes Canonical package-pinned `.bfbs` byte sequence.
/// @return Raw 32-byte SHA-256 digest.
///
/// This helper performs no schema parsing, text normalization, path rewriting,
/// or object-representation hashing. Generated-package validation compares its
/// result with the complete digest in \ref SchemaIdentity.
[[nodiscard]] constexpr std::array<std::byte, 32> schema_artifact_sha256(
    std::span<const std::byte> bytes) noexcept {
  detail::SchemaSha256Builder hash;
  hash.add(bytes);
  return hash.finish();
}

/// @brief Derive the compact v1 schema identifier from complete semantic identity.
/// @param canonical_name Exact globally scoped UTF-8 schema name.
/// @param version Explicit nonzero semantic schema version.
/// @param schema_sha256 Full digest of the canonical package-pinned `.bfbs` bytes.
/// @return First 16 bytes of the domain-separated SHA-256 derivation.
///
/// The encoded preimage is exactly `"holoscan.schema-id/v1\\0"`, the name
/// byte count as unsigned 32-bit big-endian, the unmodified name bytes, the
/// version as unsigned 32-bit big-endian, and the 32 raw artifact-digest bytes.
/// Callers validate the name length and UTF-8 contract before invoking this
/// bounded derivation; Core never performs Unicode normalization.
[[nodiscard]] constexpr SchemaId derive_schema_id(
    std::string_view canonical_name, std::uint32_t version,
    std::span<const std::byte, 32> schema_sha256) noexcept {
  detail::SchemaSha256Builder hash;
  hash.add(std::string_view{"holoscan.schema-id/v1\0", 22U});
  hash.add_u32_be(static_cast<std::uint32_t>(canonical_name.size()));
  hash.add(canonical_name);
  hash.add_u32_be(version);
  hash.add(schema_sha256);
  const auto digest = hash.finish();

  SchemaId result;
  for (std::size_t index = 0U; index < result.value.size(); ++index) {
    result.value[index] = digest[index];
  }
  return result;
}

/// @brief Recompute the compact identifier carried by a complete identity.
/// @param identity Complete semantic schema identity.
/// @return Derived compact identifier using the v1 byte-exact algorithm.
[[nodiscard]] constexpr SchemaId derive_schema_id(const SchemaIdentity& identity) noexcept {
  return derive_schema_id(identity.canonical_name, identity.version, identity.schema_sha256);
}

static_assert(sizeof(SchemaId) == 16);
static_assert(alignof(SchemaId) == alignof(std::byte));
static_assert(std::is_standard_layout_v<SchemaId>);
static_assert(std::is_trivially_copyable_v<SchemaId>);

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
