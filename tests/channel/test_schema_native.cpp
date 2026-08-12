/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <array>
#include <cstddef>
#include <span>
#include <string>

#include <gtest/gtest.h>

#include <holoscan/channel/runtime.hpp>
#include <holoscan/core/domain/tensor.hpp>
// Provides generated owning types and schema-trait specializations.
#include <holoscan/schema/flatbuffer_schema_package.hpp>  // NOLINT(misc-include-cleaner)
#include <holoscan/schema/inline_flatbuffer_payload.hpp>

#include "../common/allocation_counter.hpp"

namespace {

static_assert(holoscan::PublishedPayload<holoscan::schema::ImuT>);
static_assert(holoscan::PublishedPayload<holoscan::schema::StringT>);
static_assert(holoscan::PublishedPayload<holoscan::schema::TimeT>);
static_assert(!holoscan::ChannelInlinePayload<holoscan::schema::TimeT>);
static_assert(holoscan::CatalogChannelSchema<holoscan::schema::ImuT>);
static_assert(holoscan::CatalogChannelSchema<holoscan::schema::StringT>);
static_assert(holoscan::CatalogChannelSchema<holoscan::schema::TimeT>);
static_assert(!holoscan::CatalogChannelSchema<holoscan::schema::ImageT>);
static_assert(!holoscan::CatalogChannelSchema<holoscan::schema::Image>);
static_assert(!holoscan::CatalogChannelSchema<holoscan::schema::Tensor>);
static_assert(!holoscan::CatalogChannelSchema<holoscan::schema::String>);
static_assert(!holoscan::CatalogChannelSchema<holoscan::Tensor>);
static_assert(!holoscan::CatalogChannelSchema<
              holoscan::schema::InlineFlatBufferPayload<holoscan::schema::String, 512U>>);

// Verifies that generated native type Native inherits its canonical name, version,
// schema digest, and schema ID from its FlatBuffers root type. Also confirms the
// singleton schema_package<Native> is valid and reports the correct max serialized
// size. Used by ProjectsRootIdentityOntoOwningGeneratedType.
template <typename Native>
void expect_native_package_projects_root_identity() {
  using Root = typename holoscan::GeneratedSchemaNativeTraits<Native>::root_type;
  constexpr holoscan::SchemaIdentity kRootIdentity = holoscan::schema_identity<Root>();
  constexpr holoscan::SchemaIdentity kNativeIdentity = holoscan::schema_identity<Native>();
  EXPECT_EQ(kNativeIdentity.canonical_name, kRootIdentity.canonical_name);
  EXPECT_EQ(kNativeIdentity.version, kRootIdentity.version);
  EXPECT_EQ(kNativeIdentity.schema_sha256, kRootIdentity.schema_sha256);
  EXPECT_EQ(kNativeIdentity.id, kRootIdentity.id);

  const auto& package = holoscan::schema_package<Native>();
  EXPECT_EQ(package.identity().id, kNativeIdentity.id);
  EXPECT_EQ(package.max_serialized_size(),
            holoscan::generated_schema_max_serialized_size_v<Native>);
  EXPECT_TRUE(package.has_encoder());
  EXPECT_TRUE(package.has_materializer());
  EXPECT_TRUE(holoscan::validate_schema_package(package).has_value());
}

// Compile-time-only check that ChannelRuntime::create_publisher accepts a native
// schema type. [[maybe_unused]] because this function is never called at runtime.
[[maybe_unused]] holoscan::expected<holoscan::ChannelPublisher<holoscan::schema::StringT>,
                                    holoscan::Error>
attach_native_schema_publisher(holoscan::ChannelRuntime& runtime,
                               holoscan::ChannelEndpointCatalog::EndpointView endpoint) {
  return runtime.create_publisher<holoscan::schema::StringT>(endpoint);
}

// Compile-time-only check that ChannelRuntime::create_subscriber accepts a native
// schema type. [[maybe_unused]] because this function is never called at runtime.
[[maybe_unused]] holoscan::expected<holoscan::ChannelSubscriber<holoscan::schema::StringT>,
                                    holoscan::Error>
attach_native_schema_subscriber(holoscan::ChannelRuntime& runtime,
                                holoscan::ChannelEndpointCatalog::EndpointView endpoint) {
  return runtime.create_subscriber<holoscan::schema::StringT>(endpoint);
}

// REQ: CHAN-075
// REQ: CHAN-120
TEST(ChannelNativeSchemaPackage, ProjectsRootIdentityOntoOwningGeneratedType) {
  // Key assertion: identity fields of each native type equal those of its root,
  //   and the singleton schema package passes validate_schema_package.
  expect_native_package_projects_root_identity<holoscan::schema::ImuT>();
  expect_native_package_projects_root_identity<holoscan::schema::StringT>();
  expect_native_package_projects_root_identity<holoscan::schema::TimeT>();
}

// REQ: CHAN-075
// REQ: CHAN-120
TEST(ChannelNativeSchemaPackage, EncodesVerifiesAndMaterializesOwningValue) {
  // Key setup: StringT owns a heap-allocated std::string, exercising the full
  //   encode/verify/materialize path for a variable-size native type.
  using Native = holoscan::schema::StringT;
  alignas(std::max_align_t)
      std::array<std::byte, holoscan::generated_schema_max_serialized_size_v<Native>>
          storage{};
  const auto& package = holoscan::schema_package<Native>();
  Native source;
  source.data = "verified native value";

  auto encoded = package.encode(source, storage);
  ASSERT_TRUE(encoded.has_value());
  ASSERT_GT(*encoded, 0U);
  const auto bytes = std::span<const std::byte>(storage).first(*encoded);
  EXPECT_TRUE(package.verify(bytes));

  auto materialized = package.materialize(bytes);
  ASSERT_TRUE(materialized.has_value());
  // Key assertion: materialized data field equals the original source string.
  EXPECT_EQ(materialized->data, source.data);
}

// Generated UnPackTo materializes into caller-owned storage. A fixed-size native root therefore
// needs no top-level heap object; variable-size member allocations remain owned by their fields.
// REQ: CHAN-075
// REQ: CHAN-120
TEST(ChannelNativeSchemaPackage, FixedSizeMaterializationAvoidsTopLevelAllocation) {
  // Key setup: 32 back-to-back materializations of a fixed-size TimeT are bracketed
  //   by ScopedAllocationCounter after a warmup call to exclude first-run overhead.
  using Native = holoscan::schema::TimeT;
  alignas(std::max_align_t)
      std::array<std::byte, holoscan::generated_schema_max_serialized_size_v<Native>>
          storage{};
  const auto& package = holoscan::schema_package<Native>();
  Native source;
  source.time_ns = 1234;
  source.acqtime_ns = 5678;
  auto encoded = package.encode(source, storage);
  ASSERT_TRUE(encoded) << holoscan::runtime_error_message(encoded.error());
  const auto bytes = std::span<const std::byte>{storage}.first(*encoded);
  auto warmup = package.materialize(bytes);
  ASSERT_TRUE(warmup) << holoscan::runtime_error_message(warmup.error());

  bool values_match = true;
  std::size_t allocation_count = 0U;
  {
    const holoscan::test::ScopedAllocationCounter allocations;
    for (std::size_t iteration = 0U; iteration < 32U; ++iteration) {
      auto value = package.materialize(bytes);
      values_match = values_match && value.has_value() && value->time_ns == source.time_ns &&
                     value->acqtime_ns == source.acqtime_ns;
    }
    allocation_count = allocations.count();
  }
  EXPECT_TRUE(values_match);
  // Key assertion: zero heap allocations for 32 materializations of a fixed-size type.
  EXPECT_EQ(allocation_count, 0U);
}

// REQ: CHAN-075
// REQ: CHAN-120
TEST(ChannelNativeSchemaPackage, RejectsCapacityAndMalformedPayloadWithoutTypedExposure) {
  // Key setup: oversized string fills storage.size()*2 bytes (exceeds max); 8-byte
  //   all-zero span is structurally invalid FlatBuffer data.
  using Native = holoscan::schema::StringT;
  alignas(std::max_align_t)
      std::array<std::byte, holoscan::generated_schema_max_serialized_size_v<Native>>
          storage{};
  const auto& package = holoscan::schema_package<Native>();
  Native oversized;
  oversized.data = std::string(storage.size() * 2U, 'x');

  auto encoded = package.encode(oversized, storage);
  ASSERT_FALSE(encoded.has_value());
  // Key assertion: encode returns kResourceExhausted; materialize returns kCodecError.
  EXPECT_EQ(encoded.error().code, holoscan::ErrorCode::kResourceExhausted);

  const std::array<std::byte, 8> malformed{};
  auto materialized = package.materialize(malformed);
  ASSERT_FALSE(materialized.has_value());
  EXPECT_EQ(materialized.error().code, holoscan::ErrorCode::kCodecError);
}

}  // namespace
