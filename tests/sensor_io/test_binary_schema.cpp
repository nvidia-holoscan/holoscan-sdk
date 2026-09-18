/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/// @file
/// @brief Verify that embedded BFBS reflection bytes stay bound to their own generated schema.

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>

#include <gtest/gtest.h>

#include <flatbuffers/reflection.h>
#include <flatbuffers/verifier.h>

#include <holoscan/core/schema_identity.hpp>
#include <holoscan/schema/binary_schema.hpp>
#include <holoscan/schema/schema_package.hpp>
#include <holoscan/schema/sdk_schemas.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>

namespace holoscan {
namespace {

// Reading the root table name back out of the bytes is what actually proves the binding: a
// descriptor could hand out well-formed reflection data that describes some other schema, and only
// the name recorded inside the buffer distinguishes that from the correct one.
// Each failure returns rather than reporting and falling through: a non-fatal assertion would
// otherwise leave the malformed buffer to be dereferenced anyway, turning a legible schema failure
// into a crash that takes the rest of the suite with it.
[[nodiscard]] std::string reflected_root_name(const std::span<const std::byte> bytes) {
  flatbuffers::Verifier verifier(reinterpret_cast<const std::uint8_t*>(bytes.data()), bytes.size());
  if (!reflection::VerifySchemaBuffer(verifier)) {
    ADD_FAILURE() << "embedded reflection bytes are not a valid FlatBuffers schema buffer";
    return {};
  }
  const reflection::Schema* schema = reflection::GetSchema(bytes.data());
  if (schema == nullptr || schema->root_table() == nullptr) {
    ADD_FAILURE() << "embedded reflection bytes declare no root table";
    return {};
  }
  return schema->root_table()->name()->str();
}

// REQ: SENSOR-IO-009
TEST(SensorIoBinarySchema, EmbeddedBytesDescribeTheirOwnGeneratedRoot) {
  const auto image_bytes = binary_schema<schema::Image>();
  ASSERT_FALSE(image_bytes.empty());
  EXPECT_EQ(reflected_root_name(image_bytes), schema_identity<schema::Image>().canonical_name);

  const auto imu_bytes = binary_schema<schema::Imu>();
  ASSERT_FALSE(imu_bytes.empty());
  EXPECT_EQ(reflected_root_name(imu_bytes), schema_identity<schema::Imu>().canonical_name);

  EXPECT_NE(image_bytes.data(), imu_bytes.data());
}

// REQ: SENSOR-IO-009
TEST(SensorIoBinarySchema, OwningNativeTypesShareTheirGeneratedRootBytes) {
  EXPECT_EQ(binary_schema<schema::ImageT>().data(), binary_schema<schema::Image>().data());
  EXPECT_EQ(binary_schema<schema::ImageT>().size(), binary_schema<schema::Image>().size());
  EXPECT_EQ(binary_schema<schema::ImuT>().data(), binary_schema<schema::Imu>().data());
  EXPECT_EQ(binary_schema<schema::ImuT>().size(), binary_schema<schema::Imu>().size());
}

// REQ: SENSOR-IO-009
TEST(SensorIoBinarySchema, EverySdkSchemaWithAGeneratedBindingCarriesReflection) {
  static_assert(HasBinarySchema<schema::Image>);
  static_assert(HasBinarySchema<schema::ImageT>);
  static_assert(HasBinarySchema<schema::Imu>);
  static_assert(HasBinarySchema<schema::ImuT>);
  static_assert(HasBinarySchema<schema::String>);
  static_assert(HasBinarySchema<schema::StringT>);
  static_assert(HasBinarySchema<schema::Time>);
  static_assert(HasBinarySchema<schema::TimeT>);
  static_assert(HasBinarySchema<schema::Header>);
  static_assert(HasBinarySchema<schema::TensorPacket>);

  EXPECT_EQ(reflected_root_name(binary_schema<schema::String>()),
            schema_identity<schema::String>().canonical_name);
  EXPECT_EQ(reflected_root_name(binary_schema<schema::Time>()),
            schema_identity<schema::Time>().canonical_name);
}

// REQ: SENSOR-IO-009
TEST(SensorIoBinarySchema, ReflectionBytesRemainOptInAndOutsideIdentity) {
  // Tensor's binding is written by hand rather than generated, so it names no reflection bytes.
  // It is the sharper negative than a type with no binding at all: the binding resolves, and the
  // accessor is still unavailable, which is what shows embedding is genuinely opt-in rather than
  // falling back to some other schema's bytes.
  static_assert(detail::HasFlatBufferSchemaRootBinding<schema::Tensor>);
  static_assert(!HasBinarySchema<schema::Tensor>);

  // Embedding is build provenance, never a term in identity or the package contract.
  EXPECT_TRUE(validate_schema_package(schema_package<schema::Image>()));
  EXPECT_EQ(schema_package<schema::Image>().identity(), schema_identity<schema::Image>());
}

}  // namespace
}  // namespace holoscan
