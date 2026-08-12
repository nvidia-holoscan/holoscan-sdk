/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string_view>
#include <type_traits>

#include <holoscan/core/schema_identity.hpp>

namespace holoscan {

// TestGeneratedRoot is a stand-in for any FlatBuffers-generated schema root type.
// Its schema_identity specialisation manually sets all four fields (name, version,
// sha256 digest, compact id) so tests can verify each field individually.
struct TestGeneratedRoot {};

template <>
consteval SchemaIdentity schema_identity<TestGeneratedRoot>() {
  std::array<std::byte, 32> digest{};
  digest[0] = std::byte{0xA5};
  std::array<std::byte, 16> compact{};
  compact[0] = std::byte{0x5A};
  return SchemaIdentity{
      .canonical_name = "nvidia.holoscan.test.TestGeneratedRoot",
      .version = 3,
      .schema_sha256 = digest,
      .id = SchemaId{.value = compact},
  };
}

namespace {

static_assert(sizeof(SchemaId) == 16);
static_assert(alignof(SchemaId) == alignof(std::byte));
static_assert(std::is_standard_layout_v<SchemaId>);
static_assert(std::is_trivially_copyable_v<SchemaId>);
static_assert(!std::is_convertible_v<SchemaId, bool>);
static_assert(std::equality_comparable<SchemaId>);
static_assert(std::is_invocable_r_v<std::size_t, std::hash<SchemaId>, SchemaId>);
static_assert(std::is_same_v<decltype(SchemaId::value), std::array<std::byte, 16>>);
static_assert(std::is_same_v<decltype(SchemaIdentity::canonical_name), std::string_view>);
static_assert(std::is_same_v<decltype(SchemaIdentity::version), std::uint32_t>);
static_assert(std::is_same_v<decltype(SchemaIdentity::schema_sha256), std::array<std::byte, 32>>);
static_assert(std::is_same_v<decltype(SchemaIdentity::id), SchemaId>);

// REQ: OPERATOR-007
// SchemaId uses all-zero as a valid (not sentinel-null) compact ID. This matters because
// generated schemas may produce a 16-byte compact hash that happens to be all zeros.
TEST(SchemaIdentity, EveryCompactBitPatternIncludingZeroIsAValue) {
  constexpr SchemaId all_zero{};
  constexpr SchemaId another_all_zero{};
  constexpr SchemaId nonzero{.value = {std::byte{1}}};

  static_assert(all_zero == another_all_zero);
  static_assert(all_zero != nonzero);
  EXPECT_EQ(std::hash<SchemaId>{}(all_zero), std::hash<SchemaId>{}(another_all_zero));
}

// REQ: OPERATOR-007
// schema_identity<T>() is consteval, so the full identity (name, version, digests)
// must be accessible at compile time without any runtime initialisation.
TEST(SchemaIdentity, GeneratedSemanticRootProjectsCompleteIdentityAtCompileTime) {
  constexpr SchemaIdentity identity = schema_identity<TestGeneratedRoot>();

  static_assert(identity.canonical_name == "nvidia.holoscan.test.TestGeneratedRoot");
  static_assert(identity.version == 3);
  static_assert(identity.schema_sha256[0] == std::byte{0xA5});
  static_assert(identity.id.value[0] == std::byte{0x5A});
  EXPECT_EQ(identity.version, 3);
}

}  // namespace
}  // namespace holoscan
