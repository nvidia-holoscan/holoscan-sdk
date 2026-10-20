/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <cstddef>
#include <cstdint>
#include <string_view>
#include <typeinfo>

#include <test_operator_provider_export.h>
#include <holoscan/authoring/registry.hpp>
#include <holoscan/schema/sdk_schemas.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>
#include <holoscan/testing/operator_test_harness.hpp>

// Registry rows and owning test adapters cross the boundary, never an Operator definition.
[[nodiscard]] TEST_OPERATOR_PROVIDER_EXPORT const holoscan::authoring::RegisteredOperatorType&
test_operator_provider();
[[nodiscard]] TEST_OPERATOR_PROVIDER_EXPORT const holoscan::authoring::RegisteredOperatorType&
test_tensor_operator_provider();
[[nodiscard]] TEST_OPERATOR_PROVIDER_EXPORT const holoscan::authoring::RegisteredOperatorType&
test_clock_operator_provider();

[[nodiscard]] TEST_OPERATOR_PROVIDER_EXPORT holoscan::testing::StagedInput test_tensor_input(
    holoscan::Tensor value);
[[nodiscard]] TEST_OPERATOR_PROVIDER_EXPORT holoscan::testing::CaptureRequest test_tensor_capture();

// One common native definition is compiled independently into provider and consumer.
struct InstalledTestPod {
  std::uint32_t value{};
};
struct InstalledOtherPod {
  std::uint32_t value{};
};
namespace holoscan {
template <>
struct ChannelTypeIdentity<InstalledTestPod> {
  static constexpr std::string_view name = "holoscan.test.InstalledPod";
  static constexpr std::uint32_t version = 1U;
};
template <>
struct ChannelTypeIdentity<InstalledOtherPod> : ChannelTypeIdentity<InstalledTestPod> {};
template <>
struct ChannelPodLayout<InstalledTestPod> {
  static constexpr auto fields =
      pod_fields(pod_field("value", &InstalledTestPod::value, offsetof(InstalledTestPod, value)));
};
template <>
struct ChannelPodLayout<InstalledOtherPod> {
  static constexpr auto fields =
      pod_fields(pod_field("value", &InstalledOtherPod::value, offsetof(InstalledOtherPod, value)));
};
}  // namespace holoscan

[[nodiscard]] TEST_OPERATOR_PROVIDER_EXPORT const holoscan::authoring::RegisteredOperatorType&
test_pod_operator_provider();
[[nodiscard]] TEST_OPERATOR_PROVIDER_EXPORT const holoscan::authoring::RegisteredOperatorType&
test_schema_operator_provider();
[[nodiscard]] TEST_OPERATOR_PROVIDER_EXPORT const holoscan::authoring::RegisteredOperatorType&
test_packet_operator_provider();
[[nodiscard]] TEST_OPERATOR_PROVIDER_EXPORT const holoscan::authoring::RegisteredOperatorType&
test_image_operator_provider();
[[nodiscard]] TEST_OPERATOR_PROVIDER_EXPORT holoscan::testing::StagedInput test_pod_input(
    InstalledTestPod value);
[[nodiscard]] TEST_OPERATOR_PROVIDER_EXPORT holoscan::testing::CaptureRequest test_pod_capture();
[[nodiscard]] TEST_OPERATOR_PROVIDER_EXPORT holoscan::testing::StagedInput test_schema_input(
    holoscan::schema::StringT value);
[[nodiscard]] TEST_OPERATOR_PROVIDER_EXPORT holoscan::testing::CaptureRequest test_schema_capture();
[[nodiscard]] TEST_OPERATOR_PROVIDER_EXPORT holoscan::testing::StagedInput test_packet_input(
    holoscan::schema::TensorPacketT value);
[[nodiscard]] TEST_OPERATOR_PROVIDER_EXPORT holoscan::testing::CaptureRequest test_packet_capture();
[[nodiscard]] TEST_OPERATOR_PROVIDER_EXPORT holoscan::testing::StagedInput test_image_input(
    holoscan::schema::ImageT value);
[[nodiscard]] TEST_OPERATOR_PROVIDER_EXPORT holoscan::testing::CaptureRequest test_image_capture();
// Diagnostic witnesses verify that the hidden shared-provider case has independent RTTI objects.
[[nodiscard]] TEST_OPERATOR_PROVIDER_EXPORT const std::type_info& test_pod_native_type() noexcept;
[[nodiscard]] TEST_OPERATOR_PROVIDER_EXPORT const std::type_info&
test_schema_native_type() noexcept;
[[nodiscard]] TEST_OPERATOR_PROVIDER_EXPORT const std::type_info&
test_packet_native_type() noexcept;
[[nodiscard]] TEST_OPERATOR_PROVIDER_EXPORT const std::type_info& test_image_native_type() noexcept;
/// Configure native restart authority inside the provider that owns the operator definition.
TEST_OPERATOR_PROVIDER_EXPORT void test_allow_restart(holoscan::testing::NativeTestGraphView& view);
