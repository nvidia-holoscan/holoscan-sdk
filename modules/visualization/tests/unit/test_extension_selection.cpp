/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <initializer_list>
#include <string_view>

#include <gtest/gtest.h>

#include "vulkan/vk_extension_util.hpp"

namespace holoviz::vulkan {
namespace {

const char* select_extension(std::string_view instance_extension,
                             std::initializer_list<std::string_view> device_extensions) {
  return select_swapchain_maintenance1_extension(instance_extension,
                                                 [device_extensions](std::string_view name) {
                                                   for (const auto extension : device_extensions) {
                                                     if (extension == name) {
                                                       return true;
                                                     }
                                                   }
                                                   return false;
                                                 });
}

TEST(ExtensionSelectionTest, ExtInstanceUsesExtWhenDeviceOffersBothSpellings) {
  // GH200: the loader/layer exposes EXT at instance level while the device also offers KHR.
  EXPECT_STREQ(select_extension("VK_EXT_surface_maintenance1",
                                {"VK_KHR_swapchain_maintenance1", "VK_EXT_swapchain_maintenance1"}),
               "VK_EXT_swapchain_maintenance1");
}

TEST(ExtensionSelectionTest, KhrInstanceUsesKhrWhenDeviceOffersBothSpellings) {
  EXPECT_STREQ(select_extension("VK_KHR_surface_maintenance1",
                                {"VK_EXT_swapchain_maintenance1", "VK_KHR_swapchain_maintenance1"}),
               "VK_KHR_swapchain_maintenance1");
}

TEST(ExtensionSelectionTest, AcceptsMatchingSingleExtension) {
  EXPECT_STREQ(select_extension("VK_EXT_surface_maintenance1", {"VK_EXT_swapchain_maintenance1"}),
               "VK_EXT_swapchain_maintenance1");
  EXPECT_STREQ(select_extension("VK_KHR_surface_maintenance1", {"VK_KHR_swapchain_maintenance1"}),
               "VK_KHR_swapchain_maintenance1");
}

TEST(ExtensionSelectionTest, RejectsDeviceExtensionWithoutItsInstanceDependency) {
  EXPECT_EQ(select_extension("VK_EXT_surface_maintenance1", {"VK_KHR_swapchain_maintenance1"}),
            nullptr);
  EXPECT_EQ(select_extension("VK_KHR_surface_maintenance1", {"VK_EXT_swapchain_maintenance1"}),
            nullptr);
}

TEST(ExtensionSelectionTest, RejectsMissingMaintenanceExtensions) {
  EXPECT_EQ(select_extension("VK_EXT_surface_maintenance1", {}), nullptr);
  EXPECT_EQ(select_extension("VK_KHR_surface_maintenance1", {}), nullptr);
  EXPECT_EQ(
      select_extension("", {"VK_EXT_swapchain_maintenance1", "VK_KHR_swapchain_maintenance1"}),
      nullptr);
}

}  // namespace
}  // namespace holoviz::vulkan
