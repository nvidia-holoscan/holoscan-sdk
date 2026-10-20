// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Cross-device submit rejection: Device::submit() must reject command lists and
// swapchain gates owned by another device before collecting any native handle
// (vkQueueSubmit valid usage requires common device ownership). The rejection is
// phase-1 validation, so nothing is consumed and both devices stay fully usable.

#include <chrono>
#include <string>
#include <utility>
#include <vector>

#include <holoviz/holoviz.hpp>

#include <gtest/gtest.h>

#include "holoviz_test_skip.hpp"

namespace holoviz {
namespace {

class DeviceOwnershipTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto window = create_window({.title = "device ownership test", .size = {320, 240}});
    if (!window) {
      HOLOVIZ_ENV_SKIP() << "no window system available: " << window.error().message;
    }
    window_ = std::move(*window);

    auto surface = window_.create_surface();
    ASSERT_TRUE(surface.has_value()) << surface.error().message;
    surface_ = std::move(*surface);

    auto device = create_device_for_surface({}, surface_);
    if (!device) {
      HOLOVIZ_ENV_SKIP() << "no presentable adapter: " << device.error().message;
    }
    device_ = std::move(*device);

    auto swapchain = device_.create_swapchain(
        {.surface = surface_, .present_mode = PresentMode::kFIFO, .image_count = 3});
    ASSERT_TRUE(swapchain.has_value()) << swapchain.error().message;
    swapchain_ = std::move(*swapchain);

    auto adapters = enumerate_adapters();
    ASSERT_TRUE(adapters.has_value()) << adapters.error().message;
    ASSERT_FALSE(adapters->empty());
    auto other_device = (*adapters)[0].create_device({});
    if (!other_device) {
      HOLOVIZ_ENV_SKIP() << "second device creation failed: " << other_device.error().message;
    }
    other_device_ = std::move(*other_device);
  }

  void TearDown() override {
    if (device_) {
      EXPECT_TRUE(device_.wait_idle().has_value());
    }
    if (swapchain_) {
      EXPECT_TRUE(swapchain_.destroy(std::chrono::seconds(5)).has_value());
    }
  }

  Window window_;
  Surface surface_;
  Device device_;
  Swapchain swapchain_;
  Device other_device_;
};

TEST_F(DeviceOwnershipTest, CommandListFromAnotherDeviceIsRejected) {
  auto command_list = device_.create_command_list({.type = CommandListType::kTransient});
  ASSERT_TRUE(command_list.has_value()) << command_list.error().message;
  ASSERT_TRUE(command_list->begin().has_value());
  ASSERT_TRUE(command_list->end().has_value());

  const SubmitDesc batch{.command_lists = {&*command_list, 1}};
  const auto submitted = other_device_.submit(batch);
  ASSERT_FALSE(submitted.has_value());
  EXPECT_EQ(submitted.error().code, ErrorCode::kInvalidArgument);
  EXPECT_NE(submitted.error().message.find("another device"), std::string::npos)
      << submitted.error().message;

  // The rejected submit mutated nothing: the owning device still accepts the list.
  const auto owner_submit = device_.submit(batch);
  ASSERT_TRUE(owner_submit.has_value()) << owner_submit.error().message;
  EXPECT_TRUE(device_.wait_on_submit(*owner_submit, std::chrono::seconds(5)).has_value());
}

TEST_F(DeviceOwnershipTest, AcquireGateFromAnotherDeviceIsRejected) {
  auto token = swapchain_.acquire(std::chrono::seconds(5));
  ASSERT_TRUE(token.has_value()) << token.error().message;

  SubmitDesc batch;
  batch.acquire_gates = {&token->acquire_gate, 1};
  const auto submitted = other_device_.submit(batch);
  ASSERT_FALSE(submitted.has_value());
  EXPECT_EQ(submitted.error().code, ErrorCode::kInvalidArgument);
  EXPECT_NE(submitted.error().message.find("another device"), std::string::npos)
      << submitted.error().message;

  // Phase-1 rejection leaves the gate unconsumed; hand the image back cleanly.
  EXPECT_FALSE(token->acquire_gate.consumed().value());
  EXPECT_TRUE(swapchain_.present(std::move(*token), SubmitSyncToken{}).has_value());
}

TEST_F(DeviceOwnershipTest, PresentGateFromAnotherDeviceIsRejected) {
  auto token = swapchain_.acquire(std::chrono::seconds(5));
  ASSERT_TRUE(token.has_value()) << token.error().message;

  SubmitDesc batch;
  batch.present_gates = {&token->present_gate, 1};
  const auto submitted = other_device_.submit(batch);
  ASSERT_FALSE(submitted.has_value());
  EXPECT_EQ(submitted.error().code, ErrorCode::kInvalidArgument);
  EXPECT_NE(submitted.error().message.find("another device"), std::string::npos)
      << submitted.error().message;

  EXPECT_FALSE(token->present_gate.consumed().value());
  EXPECT_TRUE(swapchain_.present(std::move(*token), SubmitSyncToken{}).has_value());
}

}  // namespace
}  // namespace holoviz
