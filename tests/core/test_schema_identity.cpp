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

#include "../common/schema_id_test_utils.hpp"

namespace holoscan {

// TestGeneratedRoot is a stand-in for any FlatBuffers-generated schema root type.
// Its schema_identity specialization supplies the complete stable family and
// independent buffer identifier.
struct TestGeneratedRoot {};

struct TestFamilyRevisionOne {
  static constexpr const char* GetFullyQualifiedName() {
    return "nvidia.holoscan.test.EvolvingRoot";
  }
};

struct TestFamilyRevisionTwo {
  static constexpr const char* GetFullyQualifiedName() {
    return "nvidia.holoscan.test.EvolvingRoot";
  }
};

template <>
consteval SchemaIdentity schema_identity<TestGeneratedRoot>() {
  return make_schema_identity(
      make_schema_family_identity("nvidia.holoscan.test.TestGeneratedRoot", 3U),
      make_flatbuffer_file_identifier("TGRT"));
}

template <>
consteval SchemaFamilyIdentity schema_family_identity<TestFamilyRevisionOne>() {
  return make_schema_family_identity<TestFamilyRevisionOne>();
}

template <>
consteval SchemaFamilyIdentity schema_family_identity<TestFamilyRevisionTwo>() {
  return make_schema_family_identity<TestFamilyRevisionTwo>();
}

namespace {

constexpr SchemaId kTestGeneratedRootFamilyId =
    test::schema_id_from_hex("e1a2cb4c4bf02f2a047b813726208a44");
constexpr SchemaId kEvolvingRootFamilyId =
    test::schema_id_from_hex("cda23484b72792d2f70cb2ee0393fd9d");

static_assert(sizeof(SchemaId) == 16);
static_assert(alignof(SchemaId) == alignof(std::byte));
static_assert(std::is_standard_layout_v<SchemaId>);
static_assert(std::is_trivially_copyable_v<SchemaId>);
static_assert(!std::is_convertible_v<SchemaId, bool>);
static_assert(std::equality_comparable<SchemaId>);
static_assert(std::is_invocable_r_v<std::size_t, std::hash<SchemaId>, SchemaId>);
static_assert(std::is_same_v<decltype(SchemaId::value), std::array<std::byte, 16>>);
static_assert(std::is_same_v<decltype(SchemaIdentity::canonical_name), std::string_view>);
static_assert(std::is_same_v<decltype(SchemaIdentity::compatibility_epoch), std::uint32_t>);
static_assert(std::is_same_v<decltype(SchemaIdentity::family_id), SchemaId>);
static_assert(std::is_same_v<decltype(SchemaIdentity::file_identifier), FlatBufferFileIdentifier>);
static_assert(std::is_standard_layout_v<SchemaFamilyIdentity>);
static_assert(std::is_trivially_copyable_v<SchemaFamilyIdentity>);
static_assert(detail::HasSchemaFamilyIdentity<TestFamilyRevisionOne>);

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
// schema_identity<T>() is consteval, so the full stable identity
// must be accessible at compile time without any runtime initialisation.
TEST(SchemaIdentity, GeneratedSemanticRootProjectsCompleteIdentityAtCompileTime) {
  constexpr SchemaIdentity identity = schema_identity<TestGeneratedRoot>();

  static_assert(identity.canonical_name == "nvidia.holoscan.test.TestGeneratedRoot");
  static_assert(identity.compatibility_epoch == 3U);
  static_assert(identity.family_id == kTestGeneratedRootFamilyId);
  static_assert(identity.file_identifier == make_flatbuffer_file_identifier("TGRT"));
  EXPECT_EQ(identity.compatibility_epoch, 3U);
}

// REQ: SCHEMA-001
TEST(SchemaIdentity, FamilyIdentityIsStableForCompatibleRevisions) {
  constexpr auto first_family = schema_family_identity<TestFamilyRevisionOne>();
  constexpr auto second_family = schema_family_identity<TestFamilyRevisionTwo>();

  static_assert(first_family == second_family);
  static_assert(first_family.compatibility_epoch == 1U);
  static_assert(first_family.id == kEvolvingRootFamilyId);
  EXPECT_EQ(first_family, second_family);
}

// REQ: SCHEMA-001
TEST(SchemaIdentity, FamilyNameAndEpochOwnTheCompatibilityBoundary) {
  constexpr auto default_identity = make_schema_family_identity<TestFamilyRevisionOne>();
  constexpr auto epoch_identity = make_schema_family_identity<TestFamilyRevisionOne>(2U);
  constexpr auto renamed_identity = make_schema_family_identity("nvidia.holoscan.test.RenamedRoot");

  static_assert(default_identity.id != epoch_identity.id);
  static_assert(default_identity.id != renamed_identity.id);
  EXPECT_NE(default_identity.id, epoch_identity.id);
  EXPECT_NE(default_identity.id, renamed_identity.id);
}

}  // namespace
}  // namespace holoscan
