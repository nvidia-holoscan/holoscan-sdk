/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <ratio>

#include <holoscan/core/parameter_schema.hpp>

namespace holoscan {
namespace {

consteval ParamSchemaHash empty_sha256() {
  const detail::StableHashBuilder builder;
  return builder.finish();
}

constexpr std::array<std::byte, 32> kEmptySha256{
    std::byte{0xE3}, std::byte{0xB0}, std::byte{0xC4}, std::byte{0x42}, std::byte{0x98},
    std::byte{0xFC}, std::byte{0x1C}, std::byte{0x14}, std::byte{0x9A}, std::byte{0xFB},
    std::byte{0xF4}, std::byte{0xC8}, std::byte{0x99}, std::byte{0x6F}, std::byte{0xB9},
    std::byte{0x24}, std::byte{0x27}, std::byte{0xAE}, std::byte{0x41}, std::byte{0xE4},
    std::byte{0x64}, std::byte{0x9B}, std::byte{0x93}, std::byte{0x4C}, std::byte{0xA4},
    std::byte{0x95}, std::byte{0x99}, std::byte{0x1B}, std::byte{0x78}, std::byte{0x52},
    std::byte{0xB8}, std::byte{0x55},
};
static_assert(empty_sha256().value == kEmptySha256);

consteval ParamSchemaHash canonical_size_sha256() {
  detail::StableHashBuilder builder;
  builder.add_canonical_size(16U);
  return builder.finish();
}

constexpr std::array<std::byte, 32> kCanonicalSizeSha256{
    std::byte{0x5E}, std::byte{0xB6}, std::byte{0xDA}, std::byte{0x0E}, std::byte{0x0E},
    std::byte{0x52}, std::byte{0x21}, std::byte{0x04}, std::byte{0xC6}, std::byte{0xD5},
    std::byte{0x0B}, std::byte{0x07}, std::byte{0x48}, std::byte{0xE6}, std::byte{0x89},
    std::byte{0x37}, std::byte{0x62}, std::byte{0xF7}, std::byte{0xF2}, std::byte{0xC0},
    std::byte{0x0A}, std::byte{0x71}, std::byte{0x63}, std::byte{0xA4}, std::byte{0x6C},
    std::byte{0xCB}, std::byte{0xB5}, std::byte{0x35}, std::byte{0xBF}, std::byte{0xD6},
    std::byte{0x1A}, std::byte{0x0D},
};
static_assert(canonical_size_sha256().value == kCanonicalSizeSha256);

// HashParamsA and HashParamsB share the same name, version, field name, type, and
// constraints but differ only in their human-readable description. The test relies on
// this pair to verify that description changes affect only the metadata hash.
struct HashParamsA {
  float gain{1.0F};
};

HOLOSCAN_PARAM_SCHEMA(HashParamsA, holoscan::param_schema("nvidia.holoscan.test.HashParams", 1),
                      holoscan::field<&HashParamsA::gain>("gain")
                          .runtime_mutable()
                          .range(0.0F, 2.0F)
                          .description("First description."));

constexpr auto kHashParamsASchema = holoscan_param_schema(detail::type_tag<HashParamsA>{});
constexpr std::array<std::byte, 32> kHashParamsAStructuralHash{
    std::byte{0x60}, std::byte{0xD7}, std::byte{0xD2}, std::byte{0xBC}, std::byte{0xE8},
    std::byte{0x4B}, std::byte{0x35}, std::byte{0x10}, std::byte{0x77}, std::byte{0x92},
    std::byte{0xC9}, std::byte{0x01}, std::byte{0x1A}, std::byte{0x84}, std::byte{0x52},
    std::byte{0x64}, std::byte{0xA5}, std::byte{0x94}, std::byte{0xA0}, std::byte{0xC3},
    std::byte{0x98}, std::byte{0xBB}, std::byte{0x3D}, std::byte{0x08}, std::byte{0xFB},
    std::byte{0x1E}, std::byte{0x42}, std::byte{0xCB}, std::byte{0x6E}, std::byte{0x65},
    std::byte{0xAC}, std::byte{0x60},
};
constexpr std::array<std::byte, 32> kHashParamsAContractHash{
    std::byte{0x08}, std::byte{0x73}, std::byte{0xE9}, std::byte{0x65}, std::byte{0x42},
    std::byte{0xD3}, std::byte{0xFC}, std::byte{0xB7}, std::byte{0x00}, std::byte{0x95},
    std::byte{0x66}, std::byte{0x76}, std::byte{0x8D}, std::byte{0x3C}, std::byte{0x7C},
    std::byte{0x4D}, std::byte{0xA1}, std::byte{0x36}, std::byte{0xA4}, std::byte{0x80},
    std::byte{0x73}, std::byte{0x70}, std::byte{0xC9}, std::byte{0xFF}, std::byte{0x44},
    std::byte{0xD5}, std::byte{0xD2}, std::byte{0xF7}, std::byte{0x25}, std::byte{0x97},
    std::byte{0x70}, std::byte{0xA4},
};
constexpr std::array<std::byte, 32> kHashParamsAMetadataHash{
    std::byte{0x6F}, std::byte{0xC6}, std::byte{0xE6}, std::byte{0x96}, std::byte{0x1C},
    std::byte{0x0C}, std::byte{0x3F}, std::byte{0xEA}, std::byte{0x38}, std::byte{0x0B},
    std::byte{0xBF}, std::byte{0xF2}, std::byte{0x1F}, std::byte{0x46}, std::byte{0x4D},
    std::byte{0xCE}, std::byte{0x35}, std::byte{0xF3}, std::byte{0x21}, std::byte{0xE1},
    std::byte{0x9E}, std::byte{0xB9}, std::byte{0x1A}, std::byte{0xF2}, std::byte{0xC4},
    std::byte{0xB2}, std::byte{0x2B}, std::byte{0x66}, std::byte{0x12}, std::byte{0xB7},
    std::byte{0x63}, std::byte{0x4B},
};

// REQ: PARAM-004
static_assert(kHashParamsASchema.structural_hash.value == kHashParamsAStructuralHash);
static_assert(kHashParamsASchema.contract_hash.value == kHashParamsAContractHash);
static_assert(kHashParamsASchema.metadata_hash.value == kHashParamsAMetadataHash);

struct HashParamsB {
  float gain{1.0F};
};

HOLOSCAN_PARAM_SCHEMA(HashParamsB, holoscan::param_schema("nvidia.holoscan.test.HashParams", 1),
                      holoscan::field<&HashParamsB::gain>("gain")
                          .runtime_mutable()
                          .range(0.0F, 2.0F)
                          .description("Changed description."));

// HashParamsV2 bumps the schema version from 1 to 2; the structural hash must differ
// from HashParamsA because the schema version is part of the structural identity.
struct HashParamsV2 {
  float gain{1.0F};
};

HOLOSCAN_PARAM_SCHEMA(HashParamsV2, holoscan::param_schema("nvidia.holoscan.test.HashParams", 2),
                      holoscan::field<&HashParamsV2::gain>("gain")
                          .runtime_mutable()
                          .range(0.0F, 2.0F)
                          .description("Second schema version."));

using FloatMilliseconds = std::chrono::duration<double, std::milli>;

// PositiveZeroParams and NegativeZeroParams use +0.0 and −0.0 respectively in every
// constraint field. The hash must be identical because IEEE 754 +0 and −0 are equal
// in value; the implementation normalises −0 to +0 before hashing (REQ: PARAM-016).
struct PositiveZeroParams {
  float range_value{};
  float allowed_value{};
  FloatMilliseconds duration{};
};

HOLOSCAN_PARAM_SCHEMA(
    PositiveZeroParams, holoscan::param_schema("nvidia.holoscan.test.ZeroParams", 1),
    holoscan::field<&PositiveZeroParams::range_value>("range_value").range(0.0F, 2.0F),
    holoscan::field<&PositiveZeroParams::allowed_value>("allowed_value")
        .allowed_values({0.0F, 1.0F}),
    holoscan::field<&PositiveZeroParams::duration>("duration")
        .range(FloatMilliseconds{0.0}, FloatMilliseconds{2.0}));

struct NegativeZeroParams {
  float range_value{};
  float allowed_value{};
  FloatMilliseconds duration{};
};

HOLOSCAN_PARAM_SCHEMA(
    NegativeZeroParams, holoscan::param_schema("nvidia.holoscan.test.ZeroParams", 1),
    holoscan::field<&NegativeZeroParams::range_value>("range_value").range(-0.0F, 2.0F),
    holoscan::field<&NegativeZeroParams::allowed_value>("allowed_value")
        .allowed_values({-0.0F, 1.0F}),
    holoscan::field<&NegativeZeroParams::duration>("duration")
        .range(FloatMilliseconds{-0.0}, FloatMilliseconds{2.0}));

// REQ: PARAM-004
// Structural = name+version+field types+constraints; contract = structural without
// descriptions; metadata = all three tiers combined. Only the description differs
// between A and B, so structural and contract match but metadata diverges.
TEST(ParameterSchemaHash, StructuralContractAndMetadataHashesRemainSeparate) {
  // Key setup: three descriptors — (A,B) differ only in description; (A,V2) differ
  // in schema version so structural hashes must diverge.
  const ParamSchemaDescriptor* first = param_schema_descriptor<HashParamsA>();
  const ParamSchemaDescriptor* second = param_schema_descriptor<HashParamsB>();
  const ParamSchemaDescriptor* version_two = param_schema_descriptor<HashParamsV2>();
  ASSERT_NE(first, nullptr);
  ASSERT_NE(second, nullptr);
  ASSERT_NE(version_two, nullptr);
  EXPECT_EQ(first->structural_hash, second->structural_hash);
  EXPECT_EQ(first->contract_hash, second->contract_hash);
  EXPECT_NE(first->metadata_hash, second->metadata_hash);
  EXPECT_EQ(first->fields[0].id, second->fields[0].id);
  EXPECT_EQ(first->fields[0].id, version_two->fields[0].id);
  EXPECT_EQ(first->fields[0].full_id, version_two->fields[0].full_id);
  EXPECT_NE(first->structural_hash, version_two->structural_hash);
  EXPECT_EQ(first->fields[0].id, 0xC334569928EE74F7ULL);
  constexpr std::array<std::byte, 32> expected_field_digest{
      std::byte{0xC3}, std::byte{0x34}, std::byte{0x56}, std::byte{0x99}, std::byte{0x28},
      std::byte{0xEE}, std::byte{0x74}, std::byte{0xF7}, std::byte{0x69}, std::byte{0xE4},
      std::byte{0x62}, std::byte{0xBF}, std::byte{0x40}, std::byte{0x5D}, std::byte{0x93},
      std::byte{0xBE}, std::byte{0xDF}, std::byte{0x0B}, std::byte{0x7F}, std::byte{0x23},
      std::byte{0xF6}, std::byte{0x04}, std::byte{0x3A}, std::byte{0x8C}, std::byte{0x27},
      std::byte{0x90}, std::byte{0x6F}, std::byte{0x0D}, std::byte{0x9C}, std::byte{0x8C},
      std::byte{0x8F}, std::byte{0x2F},
  };
  EXPECT_EQ(first->fields[0].full_id.value, expected_field_digest);
}

// REQ: PARAM-016
// Confirms that the negative-zero normalisation applies to all three constraint kinds:
// range bounds, allowed-values list, and duration range (FloatMilliseconds).
TEST(ParameterSchemaHash, FloatingZeroConstraintsUseOneCanonicalHash) {
  // Key setup: NegativeZeroParams uses -0.0 everywhere; bit_cast on the stored
  // descriptor values should show +0 bits, proving normalisation happened at schema time.
  const ParamSchemaDescriptor* positive = param_schema_descriptor<PositiveZeroParams>();
  const ParamSchemaDescriptor* negative = param_schema_descriptor<NegativeZeroParams>();
  ASSERT_NE(positive, nullptr);
  ASSERT_NE(negative, nullptr);
  ASSERT_EQ(positive->fields.size(), 3U);
  ASSERT_EQ(negative->fields.size(), positive->fields.size());

  constexpr std::uint64_t kPositiveZeroBits = 0;
  EXPECT_EQ(std::bit_cast<std::uint64_t>(negative->fields[0].constraints.range_min.floating_value),
            kPositiveZeroBits);
  EXPECT_EQ(std::bit_cast<std::uint64_t>(
                negative->fields[1].constraints.allowed_values[0].floating_value),
            kPositiveZeroBits);
  EXPECT_EQ(std::bit_cast<std::uint64_t>(negative->fields[2].constraints.range_min.floating_value),
            kPositiveZeroBits);

  EXPECT_EQ(positive->structural_hash, negative->structural_hash);
  EXPECT_EQ(positive->contract_hash, negative->contract_hash);
  EXPECT_EQ(positive->metadata_hash, negative->metadata_hash);
}

}  // namespace
}  // namespace holoscan
