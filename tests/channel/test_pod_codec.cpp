/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

#include <holoscan/core/operator_spec.hpp>

#include "channel/pod_codec.hpp"

namespace holoscan {

namespace channel_internal {

// White-box accessor exposing the endian-specific codec branches that the production
// encode/decode API never reaches on a little-endian host. Exercises the big-endian
// path directly. Used by BigEndianArrayBranchPreservesCanonicalElementOrder.
class PodFieldwiseCodecTestAccess {
 public:
  [[nodiscard]] static expected<void, Error> encode_big_endian(const PodIdentityRecord& identity,
                                                               std::span<const std::byte> native,
                                                               std::span<std::byte> wire) {
    auto valid = PodFieldwiseCodec::validate(identity);
    if (!valid) {
      return valid;
    }
    return PodFieldwiseCodec::encode_prevalidated_for_native_endian<std::endian::big>(
        identity, native, wire);
  }

  [[nodiscard]] static expected<void, Error> decode_big_endian(const PodIdentityRecord& identity,
                                                               std::span<const std::byte> wire,
                                                               std::span<std::byte> native) {
    auto valid = PodFieldwiseCodec::validate(identity);
    if (!valid) {
      return valid;
    }
    return PodFieldwiseCodec::decode_prevalidated_for_native_endian<std::endian::big>(
        identity, wire, native);
  }
};

}  // namespace channel_internal

// Representative POD covering every supported fixed-width scalar (u64, f64, u32,
// i32, f32, u16, i16, u8, i8) and a multi-byte array tail. Used by
// EncodesEveryFixedWidthFieldInCanonicalOrder and PrevalidatedArrayFields*.
struct CodecWirePod {
  std::uint64_t u64;
  double f64;
  std::uint32_t u32;
  std::int32_t i32;
  float f32;
  std::uint16_t u16;
  std::int16_t i16;
  std::uint8_t u8;
  std::int8_t i8;
  std::array<std::uint8_t, 6> tail;
};

// Two-element uint32 array. Tests that per-element byte-reversal on a big-endian
// host preserves element order (reversing the whole field would also swap elements).
// Used by BigEndianArrayBranchPreservesCanonicalElementOrder.
struct CodecArrayPod {
  std::array<std::uint32_t, 2> values;
};

template <>
struct ChannelTypeIdentity<CodecWirePod> {
  static constexpr std::string_view name = "nvidia.holoscan.test.CodecWirePod";
  static constexpr std::uint32_t version = 7U;
};

template <>
struct ChannelPodLayout<CodecWirePod> {
  static constexpr auto fields =
      pod_fields(pod_field("u64", &CodecWirePod::u64, offsetof(CodecWirePod, u64)),
                 pod_field("f64", &CodecWirePod::f64, offsetof(CodecWirePod, f64)),
                 pod_field("u32", &CodecWirePod::u32, offsetof(CodecWirePod, u32)),
                 pod_field("i32", &CodecWirePod::i32, offsetof(CodecWirePod, i32)),
                 pod_field("f32", &CodecWirePod::f32, offsetof(CodecWirePod, f32)),
                 pod_field("u16", &CodecWirePod::u16, offsetof(CodecWirePod, u16)),
                 pod_field("i16", &CodecWirePod::i16, offsetof(CodecWirePod, i16)),
                 pod_field("u8", &CodecWirePod::u8, offsetof(CodecWirePod, u8)),
                 pod_field("i8", &CodecWirePod::i8, offsetof(CodecWirePod, i8)),
                 pod_field("tail", &CodecWirePod::tail, offsetof(CodecWirePod, tail)));
};

template <>
struct ChannelTypeIdentity<CodecArrayPod> {
  static constexpr std::string_view name = "nvidia.holoscan.test.CodecArrayPod";
  static constexpr std::uint32_t version = 1U;
};

template <>
struct ChannelPodLayout<CodecArrayPod> {
  static constexpr auto fields =
      pod_fields(pod_field("values", &CodecArrayPod::values, offsetof(CodecArrayPod, values)));
};

// POD whose default constructor is explicitly deleted. Proves the decoder writes
// directly into caller-supplied raw storage without needing to default-construct
// the target type. Used by DecodesWithoutDefaultConstructingThePayload.
struct CodecNonDefaultPod {
  explicit CodecNonDefaultPod(std::uint32_t initial) : value(initial) {}
  CodecNonDefaultPod() = delete;

  std::uint32_t value;
};

template <>
struct ChannelTypeIdentity<CodecNonDefaultPod> {
  static constexpr std::string_view name = "nvidia.holoscan.test.CodecNonDefaultPod";
  static constexpr std::uint32_t version = 1U;
};

template <>
struct ChannelPodLayout<CodecNonDefaultPod> {
  static constexpr auto fields = pod_fields(
      pod_field("value", &CodecNonDefaultPod::value, offsetof(CodecNonDefaultPod, value)));
};

// Two distinct C++ types that share one canonical identity name/version. Used by
// AcceptsDistinctCppTypesOnlyWhenTheirCompleteContractMatches to verify that
// structurally equivalent aliases are accepted at runtime endpoint attachment.
struct CodecAliasA {
  std::uint32_t value;
};

// Shares the canonical identity of CodecAliasA via ChannelTypeIdentity inheritance.
struct CodecAliasB {
  std::uint32_t value;
};

// No ChannelPodLayout specialization. Confirms HasChannelPodContract is false and
// that the static contract checker rejects types without a complete pod contract.
struct CodecMissingContractPod {
  std::uint32_t value;
};

// uint8 + uint32 pair with a natural alignment gap between fields. The contract
// must reject it because field byte offsets are not contiguous — there is no
// pad-byte entry in the field list to account for the gap.
struct CodecGapPod {
  std::uint8_t first;
  std::uint32_t second;
};

// Both ChannelPodLayout fields map to the name "value". The contract must reject
// duplicate field names to prevent ambiguous wire encoding.
struct CodecDuplicateFieldPod {
  std::uint16_t first;
  std::uint16_t second;
};

// Identity name contains a DEL (0x7F) control character. The contract must reject
// non-printable ASCII in canonical type names.
struct CodecInvalidIdentityNamePod {
  std::uint32_t value;
};

// Field descriptors where the member pointer and byte offset intentionally refer to
// different members (cross-wired). The contract must reject this because any encode
// would silently read from the wrong struct member.
struct CodecMismatchedMemberPod {
  std::uint32_t first;
  std::uint32_t second;
};

template <>
struct ChannelTypeIdentity<CodecMismatchedMemberPod> {
  static constexpr std::string_view name = "nvidia.holoscan.test.CodecMismatchedMemberPod";
  static constexpr std::uint32_t version = 1U;
};

template <>
struct ChannelPodLayout<CodecMismatchedMemberPod> {
  static constexpr auto fields = pod_fields(pod_field("first", &CodecMismatchedMemberPod::second,
                                                      offsetof(CodecMismatchedMemberPod, first)),
                                            pod_field("second", &CodecMismatchedMemberPod::first,
                                                      offsetof(CodecMismatchedMemberPod, second)));
};

template <>
struct ChannelTypeIdentity<CodecAliasA> {
  static constexpr std::string_view name = "nvidia.holoscan.test.CodecAlias";
  static constexpr std::uint32_t version = 1U;
};

template <>
struct ChannelTypeIdentity<CodecAliasB> : ChannelTypeIdentity<CodecAliasA> {};

template <>
struct ChannelPodLayout<CodecAliasA> {
  static constexpr auto fields =
      pod_fields(pod_field("value", &CodecAliasA::value, offsetof(CodecAliasA, value)));
};

template <>
struct ChannelPodLayout<CodecAliasB> {
  static constexpr auto fields =
      pod_fields(pod_field("value", &CodecAliasB::value, offsetof(CodecAliasB, value)));
};

template <>
struct ChannelTypeIdentity<CodecGapPod> {
  static constexpr std::string_view name = "nvidia.holoscan.test.CodecGapPod";
  static constexpr std::uint32_t version = 1U;
};

template <>
struct ChannelPodLayout<CodecGapPod> {
  static constexpr auto fields =
      pod_fields(pod_field("first", &CodecGapPod::first, offsetof(CodecGapPod, first)),
                 pod_field("second", &CodecGapPod::second, offsetof(CodecGapPod, second)));
};

template <>
struct ChannelTypeIdentity<CodecDuplicateFieldPod> {
  static constexpr std::string_view name = "nvidia.holoscan.test.CodecDuplicateFieldPod";
  static constexpr std::uint32_t version = 1U;
};

template <>
struct ChannelPodLayout<CodecDuplicateFieldPod> {
  static constexpr auto fields = pod_fields(
      pod_field("value", &CodecDuplicateFieldPod::first, offsetof(CodecDuplicateFieldPod, first)),
      pod_field("value", &CodecDuplicateFieldPod::second,
                offsetof(CodecDuplicateFieldPod, second)));
};

template <>
struct ChannelTypeIdentity<CodecInvalidIdentityNamePod> {
  static constexpr std::string_view name = "invalid\x7fidentity";
  static constexpr std::uint32_t version = 1U;
};

template <>
struct ChannelPodLayout<CodecInvalidIdentityNamePod> {
  static constexpr auto fields = pod_fields(pod_field(
      "value", &CodecInvalidIdentityNamePod::value, offsetof(CodecInvalidIdentityNamePod, value)));
};

namespace {

using channel_internal::derive_pod_contract_evidence;
using channel_internal::PodContractEvidence;
using channel_internal::PodFieldwiseCodec;
using channel_internal::PodFieldwiseCodecTestAccess;
using channel_internal::validate_pod_attachment_contract;

static_assert(detail::complete_channel_pod_contract<CodecWirePod>());
static_assert(detail::complete_channel_pod_contract<CodecArrayPod>());
static_assert(detail::complete_channel_pod_contract<CodecNonDefaultPod>());
static_assert(!detail::HasChannelPodContract<CodecMissingContractPod>);
static_assert(!detail::complete_channel_pod_contract<CodecGapPod>());
static_assert(!detail::complete_channel_pod_contract<CodecDuplicateFieldPod>());
static_assert(!detail::complete_channel_pod_contract<CodecInvalidIdentityNamePod>());
static_assert(!detail::complete_channel_pod_contract<CodecMismatchedMemberPod>());
static_assert(!CatalogChannelPod<CodecMismatchedMemberPod>);
static_assert(std::is_standard_layout_v<CodecNonDefaultPod>);
static_assert(std::is_trivially_copyable_v<CodecNonDefaultPod>);
static_assert(!std::is_default_constructible_v<CodecNonDefaultPod>);

// REQ: CHAN-074
TEST(PodCodec, RejectsMemberAccessorAndOffsetDisagreement) {
  EXPECT_FALSE(detail::complete_channel_pod_contract<CodecMismatchedMemberPod>());
}

// Extracts a PodIdentityRecord from the operator_spec payload identity for type T.
// Avoids manually constructing identity records in every codec test.
template <typename T>
[[nodiscard]] PodIdentityRecord pod_identity() {
  auto payload = detail::operator_spec_payload_identity<T>();
  EXPECT_TRUE(payload.pod.has_value());
  if (!payload.pod) {
    return {};
  }
  return *std::move(payload.pod);
}

// REQ: CHAN-074
TEST(PodCodec, EncodesEveryFixedWidthFieldInCanonicalOrder) {
  // Key setup: expected wire bytes are pre-computed little-endian canonical values,
  //   covering negative two's-complement integers and IEEE 754 float encodings.
  const auto identity = pod_identity<CodecWirePod>();
  const CodecWirePod source{
      .u64 = 0x0102030405060708ULL,
      .f64 = 1.5,
      .u32 = 0x0A0B0C0DU,
      .i32 = -2,
      .f32 = -2.5F,
      .u16 = 0x1122U,
      .i16 = -3,
      .u8 = 0xAAU,
      .i8 = -4,
      .tail = {1U, 2U, 3U, 4U, 5U, 6U},
  };
  constexpr std::array<std::uint8_t, 40> kExpected{
      0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01,  // u64
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xF8, 0x3F,  // f64
      0x0D, 0x0C, 0x0B, 0x0A,                          // u32
      0xFE, 0xFF, 0xFF, 0xFF,                          // i32
      0x00, 0x00, 0x20, 0xC0,                          // f32
      0x22, 0x11,                                      // u16
      0xFD, 0xFF,                                      // i16
      0xAA,                                            // u8
      0xFC,                                            // i8
      0x01, 0x02, 0x03, 0x04, 0x05, 0x06,              // tail
  };

  std::array<std::byte, sizeof(CodecWirePod)> wire{};
  ASSERT_TRUE(PodFieldwiseCodec::encode(
      identity, std::as_bytes(std::span<const CodecWirePod>{&source, 1U}), wire));
  // Key assertion: every wire byte matches the expected canonical layout; decode
  //   round-trip reproduces each source field value exactly.
  EXPECT_TRUE(std::equal(wire.begin(), wire.end(), std::as_bytes(std::span{kExpected}).begin()));

  std::array<std::byte, sizeof(CodecWirePod)> decoded_bytes{};
  auto decoded_result = PodFieldwiseCodec::decode(identity, wire, decoded_bytes);
  ASSERT_TRUE(decoded_result) << decoded_result.error().message;
  const CodecWirePod decoded = std::bit_cast<CodecWirePod>(decoded_bytes);
  EXPECT_EQ(decoded.u64, source.u64);
  EXPECT_EQ(decoded.f64, source.f64);
  EXPECT_EQ(decoded.u32, source.u32);
  EXPECT_EQ(decoded.i32, source.i32);
  EXPECT_EQ(decoded.f32, source.f32);
  EXPECT_EQ(decoded.u16, source.u16);
  EXPECT_EQ(decoded.i16, source.i16);
  EXPECT_EQ(decoded.u8, source.u8);
  EXPECT_EQ(decoded.i8, source.i8);
  EXPECT_EQ(decoded.tail, source.tail);
}

// Endpoint-bound codecs validate immutable geometry once, then retain the same declared-field
// canonical representation without repeating the geometry walk for every sample.
// REQ: CHAN-074
TEST(PodCodec, PrevalidatedArrayFieldsMatchCheckedCanonicalPath) {
  // Key setup: validate() called once before encode_prevalidated, skipping the
  //   per-call geometry walk that encode() performs internally.
  const auto identity = pod_identity<CodecWirePod>();
  ASSERT_TRUE(PodFieldwiseCodec::validate(identity));
  const CodecWirePod source{
      .u64 = 0x0102030405060708ULL,
      .f64 = 1.5,
      .u32 = 0x0A0B0C0DU,
      .i32 = -2,
      .f32 = -2.5F,
      .u16 = 0x1122U,
      .i16 = -3,
      .u8 = 0xAAU,
      .i8 = -4,
      .tail = {1U, 2U, 3U, 4U, 5U, 6U},
  };
  const auto native = std::as_bytes(std::span<const CodecWirePod>{&source, 1U});
  std::array<std::byte, sizeof(source)> checked{};
  std::array<std::byte, sizeof(source)> prevalidated{};
  ASSERT_TRUE(PodFieldwiseCodec::encode(identity, native, checked));
  ASSERT_TRUE(PodFieldwiseCodec::encode_prevalidated(identity, native, prevalidated));
  // Key assertion: prevalidated output bytes equal the checked-path output bytes.
  EXPECT_EQ(prevalidated, checked);

  std::array<std::byte, sizeof(source)> decoded_bytes{};
  ASSERT_TRUE(PodFieldwiseCodec::decode_prevalidated(identity, prevalidated, decoded_bytes));
  const CodecWirePod decoded = std::bit_cast<CodecWirePod>(decoded_bytes);
  EXPECT_EQ(decoded.u64, source.u64);
  EXPECT_EQ(decoded.f64, source.f64);
  EXPECT_EQ(decoded.tail, source.tail);
}

// Force the byte-order branch unavailable on x86. Each array element must be reversed
// independently; reversing the whole field would also reverse element order.
// REQ: CHAN-074
TEST(PodCodec, BigEndianArrayBranchPreservesCanonicalElementOrder) {
  // Key setup: synthetic big-endian native bytes injected via PodFieldwiseCodecTestAccess
  //   to exercise the dead branch that a little-endian host never reaches in production.
  const auto identity = pod_identity<CodecArrayPod>();
  constexpr std::array<std::byte, sizeof(CodecArrayPod)> kBigEndianNative{
      std::byte{0x01},
      std::byte{0x02},
      std::byte{0x03},
      std::byte{0x04},
      std::byte{0xA1},
      std::byte{0xB2},
      std::byte{0xC3},
      std::byte{0xD4},
  };
  constexpr std::array<std::byte, sizeof(CodecArrayPod)> kCanonicalLittleEndian{
      std::byte{0x04},
      std::byte{0x03},
      std::byte{0x02},
      std::byte{0x01},
      std::byte{0xD4},
      std::byte{0xC3},
      std::byte{0xB2},
      std::byte{0xA1},
  };

  std::array<std::byte, sizeof(CodecArrayPod)> wire{};
  ASSERT_TRUE(PodFieldwiseCodecTestAccess::encode_big_endian(identity, kBigEndianNative, wire));
  // Key assertion: each 4-byte element is individually reversed to little-endian,
  //   preserving element order rather than reversing the entire field buffer.
  EXPECT_EQ(wire, kCanonicalLittleEndian);

  std::array<std::byte, sizeof(CodecArrayPod)> decoded{};
  ASSERT_TRUE(PodFieldwiseCodecTestAccess::decode_big_endian(identity, wire, decoded));
  EXPECT_EQ(decoded, kBigEndianNative);
}

// REQ: CHAN-074
TEST(PodCodec, DecodesWithoutDefaultConstructingThePayload) {
  // Key setup: CodecNonDefaultPod has no default constructor; bit_cast is the only
  //   way to recover a typed value from the raw decoded bytes.
  const auto identity = pod_identity<CodecNonDefaultPod>();
  const CodecNonDefaultPod source{0x01020304U};
  std::array<std::byte, sizeof(source)> wire{};
  ASSERT_TRUE(PodFieldwiseCodec::encode(
      identity, std::as_bytes(std::span<const CodecNonDefaultPod>{&source, 1U}), wire));
  EXPECT_EQ(wire, (std::array{std::byte{0x04}, std::byte{0x03}, std::byte{0x02}, std::byte{0x01}}));

  std::array<std::byte, sizeof(CodecNonDefaultPod)> decoded_bytes{};
  auto decoded_result = PodFieldwiseCodec::decode(identity, wire, decoded_bytes);
  ASSERT_TRUE(decoded_result) << decoded_result.error().message;
  const CodecNonDefaultPod decoded = std::bit_cast<CodecNonDefaultPod>(decoded_bytes);
  // Key assertion: decoded value matches source after encode/decode round-trip.
  EXPECT_EQ(decoded.value, source.value);
}

// REQ: CHAN-074
TEST(PodCodec, RejectsIncorrectWireGeometryWithoutPartialSuccess) {
  // Key setup: wire buffer is 3 bytes — one byte short of sizeof(CodecNonDefaultPod).
  const auto identity = pod_identity<CodecNonDefaultPod>();
  CodecNonDefaultPod source{42U};
  std::array<std::byte, 3> short_wire{};
  auto encoded = PodFieldwiseCodec::encode(
      identity, std::as_bytes(std::span<const CodecNonDefaultPod>{&source, 1U}), short_wire);
  // Key assertion: both encode and decode fail with ENDPOINT_POD_CODEC_SIZE_INVALID.
  ASSERT_FALSE(encoded);
  EXPECT_EQ(encoded.error().code, ErrorCode::kInvalidArgument);
  EXPECT_EQ(encoded.error().message, "ENDPOINT_POD_CODEC_SIZE_INVALID");

  std::array<std::byte, sizeof(CodecNonDefaultPod)> decoded_bytes{};
  auto decoded = PodFieldwiseCodec::decode(identity, short_wire, decoded_bytes);
  ASSERT_FALSE(decoded);
  EXPECT_EQ(decoded.error().code, ErrorCode::kProtocolError);
  EXPECT_EQ(decoded.error().message, "ENDPOINT_POD_CODEC_SIZE_INVALID");
}

// REQ: CHAN-074
TEST(PodCodec, DerivesFrozenTypeAndLayoutDigestEvidence) {
  // Key setup: evidence derived from CodecAliasA whose canonical name is the shared
  //   "nvidia.holoscan.test.CodecAlias" identity (also used by CodecAliasB).
  const auto identity = pod_identity<CodecAliasA>();
  auto evidence = derive_pod_contract_evidence(identity);
  ASSERT_TRUE(evidence) << evidence.error().message;

  constexpr std::array<std::uint8_t, 32> kIdentityDigest{
      0xB4, 0xE8, 0x8F, 0x3E, 0x5B, 0x08, 0xB8, 0xFD, 0x07, 0x3E, 0x10,
      0x5B, 0xF4, 0x66, 0xC7, 0xEC, 0xFF, 0xBF, 0xBD, 0xA5, 0xBD, 0x48,
      0x4E, 0x03, 0x75, 0xD6, 0x9F, 0xE2, 0x7E, 0x60, 0x3B, 0x87,
  };
  constexpr std::array<std::uint8_t, 32> kLayoutDigest{
      0xC9, 0xE2, 0x25, 0x5C, 0x0D, 0xC2, 0xC6, 0xE3, 0x06, 0x0A, 0x37,
      0x6A, 0x96, 0x07, 0x6A, 0x53, 0x4B, 0xA5, 0x9E, 0x7B, 0xE4, 0x99,
      0xCA, 0xB7, 0xE8, 0x73, 0xA4, 0x11, 0x7C, 0x2C, 0x24, 0x86,
  };
  EXPECT_EQ(evidence->identity_digest, kIdentityDigest);
  EXPECT_EQ(evidence->layout_digest, kLayoutDigest);
  // Key assertion: compact key bytes equal the identity digest prefix.
  EXPECT_TRUE(std::equal(
      evidence->key.bytes.begin(), evidence->key.bytes.end(), evidence->identity_digest.begin()));
}

// REQ: CHAN-074
TEST(PodCodec, AcceptsDistinctCppTypesOnlyWhenTheirCompleteContractMatches) {
  // Key setup: CodecAliasA and CodecAliasB share canonical name/version/layout;
  //   wrong_version increments the version to force a mismatch.
  const auto identity_a = pod_identity<CodecAliasA>();
  const auto identity_b = pod_identity<CodecAliasB>();
  auto evidence_a = derive_pod_contract_evidence(identity_a);
  auto evidence_b = derive_pod_contract_evidence(identity_b);
  ASSERT_TRUE(evidence_a);
  ASSERT_TRUE(evidence_b);
  // Key assertion: matching aliases accepted; version bump produces ENDPOINT_TYPE_MISMATCH.
  EXPECT_TRUE(validate_pod_attachment_contract(identity_a, *evidence_a, identity_b, *evidence_b));

  auto wrong_version = identity_b;
  wrong_version.version += 1U;
  auto wrong_evidence = derive_pod_contract_evidence(wrong_version);
  ASSERT_TRUE(wrong_evidence);
  auto mismatch =
      validate_pod_attachment_contract(identity_a, *evidence_a, wrong_version, *wrong_evidence);
  ASSERT_FALSE(mismatch);
  EXPECT_EQ(mismatch.error().code, ErrorCode::kIncompatible);
  EXPECT_EQ(mismatch.error().message, "ENDPOINT_TYPE_MISMATCH");
}

// REQ: CHAN-074
TEST(PodCodec, RejectsSameAuthorityLayoutChangesAndInjectedCompactKeyCollisions) {
  // Key setup: three collision scenarios — same-authority layout change (renamed field),
  //   injected foreign type with a cloned compact key, and an inconsistent compact key.
  const auto identity = pod_identity<CodecAliasA>();
  auto evidence = derive_pod_contract_evidence(identity);
  ASSERT_TRUE(evidence);

  auto conflicting_identity = identity;
  conflicting_identity.fields.front().name = "other";
  auto conflicting_evidence = derive_pod_contract_evidence(conflicting_identity);
  ASSERT_TRUE(conflicting_evidence);
  auto authority_collision = validate_pod_attachment_contract(
      identity, *evidence, conflicting_identity, *conflicting_evidence);
  // Key assertion: each collision scenario returns a distinct error message.
  ASSERT_FALSE(authority_collision);
  EXPECT_EQ(authority_collision.error().message, "ENDPOINT_POD_IDENTITY_COLLISION");

  auto foreign_identity = identity;
  foreign_identity.canonical_name = "nvidia.holoscan.test.ForeignCodecAlias";
  auto injected = derive_pod_contract_evidence(foreign_identity);
  ASSERT_TRUE(injected);
  std::copy(
      evidence->key.bytes.begin(), evidence->key.bytes.end(), injected->identity_digest.begin());
  injected->key = evidence->key;
  auto key_collision =
      validate_pod_attachment_contract(identity, *evidence, foreign_identity, *injected);
  ASSERT_FALSE(key_collision);
  EXPECT_EQ(key_collision.error().code, ErrorCode::kIncompatible);
  EXPECT_EQ(key_collision.error().message, "ENDPOINT_POD_IDENTITY_COLLISION");

  injected->key.bytes.front() ^= 0x80U;
  auto inconsistent =
      validate_pod_attachment_contract(identity, *evidence, foreign_identity, *injected);
  ASSERT_FALSE(inconsistent);
  EXPECT_EQ(inconsistent.error().code, ErrorCode::kInvalidArgument);
  EXPECT_EQ(inconsistent.error().message, "ENDPOINT_POD_KEY_INCONSISTENT");
}

// REQ: CHAN-074
TEST(PodCodec, RejectsIncompleteLayoutAndInvalidUtf8IdentityArtifacts) {
  // Key setup: three mutated identity records — gap offset (byte_offset=1), duplicate
  //   field name, and an overlong UTF-8 sequence (C0 80) in the canonical name.
  // Key assertion: derive_pod_contract_evidence returns an error for each mutation.
  auto gap = pod_identity<CodecAliasA>();
  gap.fields.front().byte_offset = 1U;
  auto gap_result = derive_pod_contract_evidence(gap);
  ASSERT_FALSE(gap_result);
  EXPECT_EQ(gap_result.error().message, "ENDPOINT_PAYLOAD_IDENTITY_INVALID");
  EXPECT_FALSE(PodFieldwiseCodec::validate(gap));

  auto duplicate = pod_identity<CodecWirePod>();
  duplicate.fields[1].name = duplicate.fields[0].name;
  EXPECT_FALSE(derive_pod_contract_evidence(duplicate));

  auto invalid_utf8 = pod_identity<CodecAliasA>();
  invalid_utf8.canonical_name = std::string{"invalid\xC0\x80", 9U};
  EXPECT_FALSE(derive_pod_contract_evidence(invalid_utf8));
}

}  // namespace
}  // namespace holoscan
