/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "pod_codec.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <set>
#include <span>
#include <string_view>

#include <holoscan/core/parameter_schema.hpp>

namespace holoscan::channel_internal {
namespace {

void add_hash_literal(detail::StableHashBuilder& hash, std::string_view literal) noexcept {
  for (const char byte : literal) {
    hash.add_byte(static_cast<std::uint8_t>(byte));
  }
}

void add_hash_u32_be(detail::StableHashBuilder& hash, std::uint32_t value) noexcept {
  for (std::size_t shift = 4U; shift != 0U; --shift) {
    hash.add_byte(static_cast<std::uint8_t>(value >> ((shift - 1U) * 8U)));
  }
}

void add_hash_bytes(detail::StableHashBuilder& hash, std::string_view bytes) noexcept {
  for (const char byte : bytes) {
    hash.add_byte(static_cast<std::uint8_t>(byte));
  }
}

[[nodiscard]] std::string_view pod_scalar_token(PodScalarKind kind) noexcept {
  switch (kind) {
    case PodScalarKind::kUInt8:
      return "u8";
    case PodScalarKind::kInt8:
      return "i8";
    case PodScalarKind::kUInt16:
      return "u16";
    case PodScalarKind::kInt16:
      return "i16";
    case PodScalarKind::kUInt32:
      return "u32";
    case PodScalarKind::kInt32:
      return "i32";
    case PodScalarKind::kUInt64:
      return "u64";
    case PodScalarKind::kInt64:
      return "i64";
    case PodScalarKind::kFloat32:
      return "f32";
    case PodScalarKind::kFloat64:
      return "f64";
  }
  return {};
}

[[nodiscard]] bool codec_geometry_valid(const PodIdentityRecord& identity) noexcept {
  if (identity.object_size == 0U || identity.fields.empty()) {
    return false;
  }
  std::uint64_t next_offset{};
  for (const auto& field : identity.fields) {
    const std::size_t scalar_size = detail::pod_scalar_size(field.scalar_kind);
    const ScalarIdentityKind scalar = pod_codec_detail::scalar_identity(field.scalar_kind);
    if (scalar_size == 0U || field.element_count == 0U || field.byte_offset != next_offset ||
        CatalogScalarCodec::wire_size(scalar) != scalar_size ||
        CatalogScalarCodec::native_size(scalar) != scalar_size) {
      return false;
    }
    const std::uint64_t byte_count = static_cast<std::uint64_t>(scalar_size) * field.element_count;
    if (byte_count / scalar_size != field.element_count ||
        next_offset > std::numeric_limits<std::uint64_t>::max() - byte_count) {
      return false;
    }
    next_offset += byte_count;
    if (next_offset > identity.object_size) {
      return false;
    }
  }
  return next_offset == identity.object_size;
}

[[nodiscard]] bool pod_identity_valid(const PodIdentityRecord& identity) {
  if (!detail::pod_type_name_valid(identity.canonical_name) || identity.version == 0U ||
      identity.object_size == 0U || identity.object_alignment == 0U ||
      (identity.object_alignment & (identity.object_alignment - 1U)) != 0U ||
      identity.object_size % identity.object_alignment != 0U || identity.fields.empty()) {
    return false;
  }
  std::uint64_t next_offset{};
  std::set<std::string_view> field_names;
  for (const auto& field : identity.fields) {
    const std::uint32_t scalar_size = detail::pod_scalar_size(field.scalar_kind);
    if (!detail::pod_identity_name_valid(field.name) || field.element_count == 0U ||
        scalar_size == 0U || field.byte_offset != next_offset ||
        !field_names.insert(field.name).second) {
      return false;
    }
    const std::uint64_t byte_count = static_cast<std::uint64_t>(scalar_size) * field.element_count;
    if (byte_count / scalar_size != field.element_count ||
        next_offset > std::numeric_limits<std::uint64_t>::max() - byte_count) {
      return false;
    }
    next_offset += byte_count;
    if (next_offset > identity.object_size) {
      return false;
    }
  }
  return next_offset == identity.object_size;
}

[[nodiscard]] std::array<std::uint8_t, 32> finish_digest(detail::StableHashBuilder& hash) noexcept {
  const auto digest = hash.finish();
  std::array<std::uint8_t, 32> result{};
  std::transform(digest.value.begin(), digest.value.end(), result.begin(), [](std::byte byte) {
    return std::to_integer<std::uint8_t>(byte);
  });
  return result;
}

[[nodiscard]] std::array<std::uint8_t, 32> identity_digest(
    const PodIdentityRecord& identity) noexcept {
  constexpr std::string_view kDomain{"holoscan.channel.pod-type\0v1\0",
                                     sizeof("holoscan.channel.pod-type\0v1\0") - 1U};
  detail::StableHashBuilder hash;
  add_hash_literal(hash, kDomain);
  add_hash_u32_be(hash, static_cast<std::uint32_t>(identity.canonical_name.size()));
  add_hash_bytes(hash, identity.canonical_name);
  add_hash_u32_be(hash, identity.version);
  return finish_digest(hash);
}

[[nodiscard]] std::array<std::uint8_t, 32> layout_digest(
    const PodIdentityRecord& identity) noexcept {
  constexpr std::string_view kDomain{"holoscan.channel.pod-layout\0v1\0",
                                     sizeof("holoscan.channel.pod-layout\0v1\0") - 1U};
  detail::StableHashBuilder hash;
  add_hash_literal(hash, kDomain);
  add_hash_u32_be(hash, identity.object_size);
  add_hash_u32_be(hash, identity.object_alignment);
  add_hash_u32_be(hash, static_cast<std::uint32_t>(identity.fields.size()));
  for (const auto& field : identity.fields) {
    add_hash_u32_be(hash, static_cast<std::uint32_t>(field.name.size()));
    add_hash_bytes(hash, field.name);
    add_hash_u32_be(hash, field.byte_offset);
    add_hash_u32_be(hash, field.element_count);
    const std::string_view scalar = pod_scalar_token(field.scalar_kind);
    add_hash_u32_be(hash, static_cast<std::uint32_t>(scalar.size()));
    add_hash_bytes(hash, scalar);
  }
  return finish_digest(hash);
}

[[nodiscard]] bool identity_equal(const PodIdentityRecord& left,
                                  const PodIdentityRecord& right) noexcept {
  if (left.canonical_name != right.canonical_name || left.version != right.version ||
      left.object_size != right.object_size || left.object_alignment != right.object_alignment ||
      left.fields.size() != right.fields.size()) {
    return false;
  }
  for (std::size_t index = 0U; index < left.fields.size(); ++index) {
    const auto& lhs = left.fields[index];
    const auto& rhs = right.fields[index];
    if (lhs.name != rhs.name || lhs.byte_offset != rhs.byte_offset ||
        lhs.scalar_kind != rhs.scalar_kind || lhs.element_count != rhs.element_count) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool key_matches_digest(const PodContractEvidence& evidence) noexcept {
  return std::equal(
      evidence.key.bytes.begin(), evidence.key.bytes.end(), evidence.identity_digest.begin());
}

}  // namespace

expected<PodContractEvidence, Error> derive_pod_contract_evidence(
    const PodIdentityRecord& identity) {
  if (!pod_identity_valid(identity) ||
      identity.fields.size() > std::numeric_limits<std::uint32_t>::max()) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument, "ENDPOINT_PAYLOAD_IDENTITY_INVALID"});
  }
  PodContractEvidence result{
      .identity_digest = identity_digest(identity),
      .layout_digest = layout_digest(identity),
  };
  std::copy_n(result.identity_digest.begin(), result.key.bytes.size(), result.key.bytes.begin());
  return result;
}

expected<void, Error> validate_pod_attachment_contract(const PodIdentityRecord& expected_identity,
                                                       const PodContractEvidence& expected_evidence,
                                                       const PodIdentityRecord& actual_identity,
                                                       const PodContractEvidence& actual_evidence) {
  if (!key_matches_digest(expected_evidence) || !key_matches_digest(actual_evidence)) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument, "ENDPOINT_POD_KEY_INCONSISTENT"});
  }

  const bool same_identity = identity_equal(expected_identity, actual_identity);
  const bool same_evidence = expected_evidence.identity_digest == actual_evidence.identity_digest &&
                             expected_evidence.key == actual_evidence.key &&
                             expected_evidence.layout_digest == actual_evidence.layout_digest;
  if (same_identity && same_evidence) {
    return {};
  }

  const bool same_authority = expected_identity.canonical_name == actual_identity.canonical_name &&
                              expected_identity.version == actual_identity.version;
  if (same_authority || expected_evidence.key == actual_evidence.key) {
    return make_unexpected(Error{ErrorCode::kIncompatible, "ENDPOINT_POD_IDENTITY_COLLISION"});
  }
  return make_unexpected(Error{ErrorCode::kIncompatible, "ENDPOINT_TYPE_MISMATCH"});
}

expected<void, Error> PodFieldwiseCodec::encode(const PodIdentityRecord& identity,
                                                std::span<const std::byte> native,
                                                std::span<std::byte> wire) {
  if (native.size() != identity.object_size || wire.size() != identity.object_size) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument, "ENDPOINT_POD_CODEC_SIZE_INVALID"});
  }
  if (!codec_geometry_valid(identity)) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument, "ENDPOINT_PAYLOAD_IDENTITY_INVALID"});
  }
  return encode_prevalidated(identity, native, wire);
}

expected<void, Error> PodFieldwiseCodec::validate(const PodIdentityRecord& identity) {
  if (!codec_geometry_valid(identity)) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument, "ENDPOINT_PAYLOAD_IDENTITY_INVALID"});
  }
  return {};
}

expected<void, Error> PodFieldwiseCodec::encode_prevalidated(const PodIdentityRecord& identity,
                                                             std::span<const std::byte> native,
                                                             std::span<std::byte> wire) {
  return encode_prevalidated_for_native_endian<std::endian::native>(identity, native, wire);
}

expected<void, Error> PodFieldwiseCodec::decode(const PodIdentityRecord& identity,
                                                std::span<const std::byte> wire,
                                                std::span<std::byte> native) {
  if (native.size() != identity.object_size) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument, "ENDPOINT_POD_CODEC_SIZE_INVALID"});
  }
  if (wire.size() != identity.object_size) {
    return make_unexpected(Error{ErrorCode::kProtocolError, "ENDPOINT_POD_CODEC_SIZE_INVALID"});
  }
  if (!codec_geometry_valid(identity)) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument, "ENDPOINT_PAYLOAD_IDENTITY_INVALID"});
  }
  return decode_prevalidated(identity, wire, native);
}

expected<void, Error> PodFieldwiseCodec::decode_prevalidated(const PodIdentityRecord& identity,
                                                             std::span<const std::byte> wire,
                                                             std::span<std::byte> native) {
  return decode_prevalidated_for_native_endian<std::endian::native>(identity, wire, native);
}

}  // namespace holoscan::channel_internal
