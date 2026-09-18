/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/// @file
/// @brief Verify generated sensor-schema family identity and package validation.

#include <gtest/gtest.h>

#include <holoscan/core/schema_identity.hpp>
#include <holoscan/schema/schema_package.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>

namespace holoscan {
namespace {

// REQ: SENSOR-IO-009
TEST(SensorIoSchemaIdentity, RootPackagesPublishGeneratedFamilyIdentity) {
  const auto& image_package = schema_package<schema::Image>();
  ASSERT_TRUE(validate_schema_package(image_package));
  EXPECT_EQ(image_package.identity(), schema_identity<schema::Image>());
  EXPECT_EQ(image_package.family_identity(), schema_family_identity<schema::Image>());
  EXPECT_EQ(image_package.identity().canonical_name, "holoscan.schema.Image");
  EXPECT_EQ(image_package.identity().compatibility_epoch, 1U);
  EXPECT_EQ(image_package.identity().file_identifier, make_flatbuffer_file_identifier("HSIG"));

  const auto& imu_package = schema_package<schema::Imu>();
  ASSERT_TRUE(validate_schema_package(imu_package));
  EXPECT_EQ(imu_package.identity(), schema_identity<schema::Imu>());
  EXPECT_EQ(imu_package.family_identity(), schema_family_identity<schema::Imu>());
  EXPECT_EQ(imu_package.identity().canonical_name, "holoscan.schema.Imu");
  EXPECT_EQ(imu_package.identity().compatibility_epoch, 1U);
  EXPECT_EQ(imu_package.identity().file_identifier, make_flatbuffer_file_identifier("HSIM"));
}

// REQ: SENSOR-IO-009
TEST(SensorIoSchemaIdentity, OwningNativeTypesShareTheirGeneratedRootIdentity) {
  EXPECT_EQ(schema_identity<schema::ImageT>(), schema_identity<schema::Image>());
  EXPECT_EQ(schema_identity<schema::ImuT>(), schema_identity<schema::Imu>());

  const auto& image_native_package = schema_package<schema::ImageT>();
  ASSERT_TRUE(validate_schema_package(image_native_package));
  EXPECT_EQ(image_native_package.identity(), schema_identity<schema::Image>());
  EXPECT_EQ(image_native_package.family_identity(), schema_family_identity<schema::Image>());

  const auto& imu_native_package = schema_package<schema::ImuT>();
  ASSERT_TRUE(validate_schema_package(imu_native_package));
  EXPECT_EQ(imu_native_package.identity(), schema_identity<schema::Imu>());
  EXPECT_EQ(imu_native_package.family_identity(), schema_family_identity<schema::Imu>());
}

}  // namespace
}  // namespace holoscan
