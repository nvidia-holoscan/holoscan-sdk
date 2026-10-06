/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <span>

#include <holoscan/schema/schema_package.hpp>

namespace holoscan::schema_test {

struct PackageValidationRoot {};

}  // namespace holoscan::schema_test

namespace holoscan {

template <>
consteval SchemaIdentity schema_identity<schema_test::PackageValidationRoot>() {
  return make_schema_identity(
      make_schema_family_identity("nvidia.holoscan.test.PackageValidationRoot"),
      make_flatbuffer_file_identifier("PKGV"));
}

template <>
consteval SchemaFamilyIdentity schema_family_identity<schema_test::PackageValidationRoot>() {
  return make_schema_family_identity("nvidia.holoscan.test.PackageValidationRoot");
}

}  // namespace holoscan

namespace {

using Root = holoscan::schema_test::PackageValidationRoot;

[[nodiscard]] bool verify_payload(std::span<const std::byte> bytes) noexcept {
  return !bytes.empty();
}

[[nodiscard]] holoscan::expected<std::size_t, holoscan::Error> encode_payload(
    const Root&, std::span<std::byte> destination) noexcept {
  if (destination.empty()) {
    return holoscan::make_unexpected(holoscan::Error{holoscan::ErrorCode::kResourceExhausted});
  }
  destination.front() = std::byte{0x01};
  return 1U;
}

[[nodiscard]] holoscan::expected<Root, holoscan::Error> materialize_payload(
    std::span<const std::byte>) noexcept {
  return Root{};
}

// A valid owning package deliberately has no borrowed parser. The borrowed-view API must report
// that missing capability instead of calling a null parser and misclassifying it as bad wire data.
// REQ: CHAN-075
TEST(SchemaPackageValidation, BorrowedParseRejectsOwningOnlyPackageAsUnsupported) {
  const holoscan::SchemaPackageDescriptor<Root> owning_only{
      holoscan::schema_identity<Root>(),
      holoscan::SchemaCodecDescriptor<Root>{
          .verify = &verify_payload,
          .encode = &encode_payload,
          .materialize = &materialize_payload,
      },
      holoscan::SchemaCodecCapacity{.max_serialized_size = 4U},
      holoscan::SchemaTransportProfile{},
  };
  const holoscan::SchemaEnvelopeHeader header{
      .schema_family_id = holoscan::schema_identity<Root>().family_id,
      .payload_size = 1U,
      .transport_profile_tag = 0U,
      .endpoint_generation = 1U,
  };
  const std::array payload{std::byte{0x01}};

  auto parsed = holoscan::verify_and_parse_received_schema(
      owning_only, header, holoscan::schema_identity<Root>(), 1U, payload);
  ASSERT_FALSE(parsed);
  EXPECT_EQ(parsed.error().code, holoscan::ErrorCode::kNotSupported);
}

}  // namespace
