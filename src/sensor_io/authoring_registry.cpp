/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/sensor_io/sensor_schema_package.hpp>

#include <holoscan/authoring/register_payload.hpp>

namespace holoscan::sensor_io {

bool register_sensor_schema_payloads() noexcept {
  static const bool registered = []() noexcept {
    constexpr SchemaIdentity image_identity = schema_identity<schema::ImageT>();
    constexpr SchemaIdentity imu_identity = schema_identity<schema::ImuT>();
    const bool image_registered = authoring::detail::register_payload_automatically<schema::ImageT>(
        image_identity.canonical_name);
    const bool imu_registered = authoring::detail::register_payload_automatically<schema::ImuT>(
        imu_identity.canonical_name);
    return image_registered && imu_registered;
  }();
  return registered;
}

namespace {

[[maybe_unused]] const bool kSensorSchemaPayloadsRegistered = register_sensor_schema_payloads();

}  // namespace
}  // namespace holoscan::sensor_io
