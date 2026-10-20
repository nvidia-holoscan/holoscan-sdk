/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <fcntl.h>
#include <limits.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <unistd.h>
#include <array>

#include <cerrno>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>

#include "display_systems/display_ownership.hpp"
#include "display_systems/display_ownership_policy.hpp"
#include "display_systems/drm_ioctl.hpp"

namespace {

using holoviz::DisplayAcquisitionMode;

const auto kPhysicalDevice = reinterpret_cast<VkPhysicalDevice>(uintptr_t{0x10});
const auto kDisplay = reinterpret_cast<VkDisplayKHR>(uintptr_t{0x20});

struct FakePlatform {
  std::unique_ptr<holoviz::vulkan::PhysicalDeviceInfo> device_info;
  int pipe_fd = -1;
  int stat_calls = 0;
  int realpath_calls = 0;
  int drm_calls = 0;
  int acquire_calls = 0;
  int translate_calls = 0;
  bool master = true;
  bool allow_set_master = true;
  bool kms = true;
  bool deny_adapter_open = false;
  VkResult acquire_result = VK_SUCCESS;
  VkResult translate_result = VK_SUCCESS;
  std::vector<std::string> opened_paths;
  std::vector<int> opened_fds;
};

FakePlatform g_fake;

bool fd_is_open(int fd) {
  return fcntl(fd, F_GETFD) != -1 || errno != EBADF;
}

VKAPI_ATTR VkResult VKAPI_CALL fake_acquire(VkPhysicalDevice device, int fd, VkDisplayKHR display) {
  ++g_fake.acquire_calls;
  EXPECT_EQ(device, kPhysicalDevice);
  EXPECT_EQ(display, kDisplay);
  EXPECT_TRUE(fd_is_open(fd));
  return g_fake.acquire_result;
}

VKAPI_ATTR VkResult VKAPI_CALL fake_translate(VkPhysicalDevice device, int, uint32_t,
                                              VkDisplayKHR* display) {
  ++g_fake.translate_calls;
  EXPECT_EQ(device, kPhysicalDevice);
  *display = g_fake.translate_result == VK_SUCCESS ? kDisplay : VK_NULL_HANDLE;
  return g_fake.translate_result;
}

}  // namespace

PFN_vkAcquireDrmDisplayEXT vkAcquireDrmDisplayEXT = fake_acquire;
PFN_vkGetDrmDisplayEXT vkGetDrmDisplayEXT = fake_translate;

// Wrap discovery calls from the production TU, leaving unrelated runtime calls alone.
// card0 is the adapter's primary node; card1 models a separate connected platform
// device. Finding that output must never make it an acquisition candidate for this GPU.
// GNU ld --wrap requires this reserved symbol spelling.
// NOLINTNEXTLINE(bugprone-reserved-identifier)
extern "C" int __real_stat(const char*, struct stat*);
// GNU ld --wrap requires this reserved symbol spelling.
// NOLINTNEXTLINE(bugprone-reserved-identifier)
extern "C" int __wrap_stat(const char* path, struct stat* info) {
  if (!std::string_view(path).starts_with("/dev/dri/")) {
    return __real_stat(path, info);
  }
  ++g_fake.stat_calls;
  if (std::strcmp(path, "/dev/dri/card0") != 0 && std::strcmp(path, "/dev/dri/card1") != 0) {
    errno = ENOENT;
    return -1;
  }
  *info = {};
  info->st_mode = S_IFCHR | 0600;
  info->st_rdev = makedev(226, std::strcmp(path, "/dev/dri/card0") == 0 ? 2 : 3);
  return 0;
}

// GNU ld --wrap requires this reserved symbol spelling.
// NOLINTNEXTLINE(bugprone-reserved-identifier)
extern "C" int __real_open(const char*, int, ...);
// GNU ld --wrap requires this reserved symbol spelling.
// NOLINTNEXTLINE(bugprone-reserved-identifier)
extern "C" int __wrap_open(const char* path, int flags, ...) {
  if (std::string_view(path).starts_with("/dev/dri/")) {
    g_fake.opened_paths.emplace_back(path);
    if (g_fake.deny_adapter_open && std::strcmp(path, "/dev/dri/card0") == 0) {
      errno = EACCES;
      return -1;
    }
    const int fd = dup(g_fake.pipe_fd);
    g_fake.opened_fds.push_back(fd);
    return fd;
  }
  if ((flags & O_CREAT) != 0) {
    va_list args;
    va_start(args, flags);
    const auto mode = va_arg(args, unsigned int);
    va_end(args);
    return __real_open(path, flags, mode);
  }
  return __real_open(path, flags);
}

// GNU ld --wrap requires this reserved symbol spelling.
// NOLINTNEXTLINE(bugprone-reserved-identifier)
extern "C" char* __real_realpath(const char*, char*);
// GNU ld --wrap requires this reserved symbol spelling.
// NOLINTNEXTLINE(bugprone-reserved-identifier)
extern "C" char* __wrap_realpath(const char* path, char* resolved) {
  if (!std::string_view(path).starts_with("/sys/class/drm/")) {
    return __real_realpath(path, resolved);
  }
  ++g_fake.realpath_calls;
  // The old unrelated-platform scan would resolve this and open card1.
  std::snprintf(resolved, PATH_MAX, "/sys/bus/platform");
  return resolved;
}

namespace holoviz::vulkan {

expected<const PhysicalDeviceInfo*> find_physical_device_info(VkPhysicalDevice device) {
  EXPECT_EQ(device, kPhysicalDevice);
  return g_fake.device_info.get();
}

std::string vk_error_string(VkResult result, const char*, int32_t) {
  return "VkResult " + std::to_string(static_cast<int>(result));
}

}  // namespace holoviz::vulkan

namespace holoviz::display_systems::drm {

// Stub the existing DRM client seam, not the ownership policy. This counts every
// possible native query/master change without touching a real kernel device.
int default_ioctl(int, unsigned long, void*) {
  ++g_fake.drm_calls;
  ADD_FAILURE() << "the ownership seam should use the fake DRM client";
  return -1;
}

bool is_master(const Node&) {
  ++g_fake.drm_calls;
  return g_fake.master;
}

expected<void> set_master(const Node&) {
  ++g_fake.drm_calls;
  if (!g_fake.allow_set_master) {
    return make_unexpected(Error{.code = ErrorCode::kFailure, .message = "DRM master denied"});
  }
  return {};
}

expected<void> drop_master(const Node&) {
  ++g_fake.drm_calls;
  return {};
}

expected<Resources> get_resources(const Node&) {
  ++g_fake.drm_calls;
  if (!g_fake.kms) {
    return make_unexpected(Error{.code = ErrorCode::kFailure, .message = "KMS unavailable"});
  }
  return Resources{.connectors = {7}};
}

expected<Connector> get_connector(const Node&, uint32_t id) {
  ++g_fake.drm_calls;
  return Connector{.connector_id = id,
                   .connector_type = DRM_MODE_CONNECTOR_DisplayPort,
                   .connector_type_id = 1,
                   .connection = kConnectionConnected};
}

expected<Encoder> get_encoder(const Node&, uint32_t) {
  ++g_fake.drm_calls;
  return Encoder{};
}

expected<Crtc> get_crtc(const Node&, uint32_t) {
  ++g_fake.drm_calls;
  return Crtc{};
}

expected<ObjectProperties> get_object_properties(const Node&, uint32_t, uint32_t) {
  ++g_fake.drm_calls;
  return ObjectProperties{};
}

expected<PropertyName> get_property_name(const Node&, uint32_t) {
  ++g_fake.drm_calls;
  return PropertyName{};
}

expected<std::vector<uint8_t>> get_property_blob(const Node&, uint32_t) {
  ++g_fake.drm_calls;
  return std::vector<uint8_t>{};
}

}  // namespace holoviz::display_systems::drm

namespace {

class DisplayAcquisitionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    g_fake = FakePlatform{};
    g_fake.device_info = std::make_unique<holoviz::vulkan::PhysicalDeviceInfo>();
    auto& info = *g_fake.device_info;
    info.physical_device_properties_.properties = {
        .vendorID = 0x10de,
        .deviceID = 0x2b00,
        .deviceType = VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU};
    info.driver_properties_.driverID = VK_DRIVER_ID_NVIDIA_PROPRIETARY;
    info.has_drm_properties_ = true;
    info.drm_properties_ = {.hasPrimary = VK_TRUE, .primaryMajor = 226, .primaryMinor = 2};
    std::array<int, 2> fds{};
    ASSERT_EQ(pipe(fds.data()), 0);
    g_fake.pipe_fd = fds[0];
    other_end_ = fds[1];
    vkAcquireDrmDisplayEXT = fake_acquire;
    vkGetDrmDisplayEXT = fake_translate;
    instance_.acquire_drm_supported_ = true;
  }

  void TearDown() override {
    for (const int fd : g_fake.opened_fds) {
      EXPECT_FALSE(fd_is_open(fd)) << "ownership path leaked fd " << fd;
      if (fd_is_open(fd)) {
        close(fd);
      }
    }
    close(g_fake.pipe_fd);
    close(other_end_);
  }

  void select_discrete_gpu() {
    g_fake.device_info->physical_device_properties_.properties.deviceType =
        VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
  }

  auto acquire() {
    return holoviz::display_systems::acquire_display(
        instance_, kPhysicalDevice, reinterpret_cast<uint64_t>(kDisplay));
  }

  auto classify() {
    return holoviz::display_systems::classify_displays(
        instance_, kPhysicalDevice, *g_fake.device_info);
  }

  void expect_no_native_access() {
    EXPECT_EQ(g_fake.stat_calls, 0);
    EXPECT_EQ(g_fake.realpath_calls, 0);
    EXPECT_TRUE(g_fake.opened_paths.empty());
    EXPECT_EQ(g_fake.drm_calls, 0);
    EXPECT_EQ(g_fake.acquire_calls, 0);
    EXPECT_EQ(g_fake.translate_calls, 0);
  }

  holoviz::vulkan::VulkanInstance instance_{};
  int other_end_ = -1;
};

TEST_F(DisplayAcquisitionTest, ThorBypassesDRMWithAndWithoutAcquisitionExtension) {
  for (const bool available : {true, false}) {
    instance_.acquire_drm_supported_ = available;
    vkAcquireDrmDisplayEXT = available ? fake_acquire : nullptr;
    vkGetDrmDisplayEXT = available ? fake_translate : nullptr;
    // No native identity is required on Thor, including in a container without sysfs.
    g_fake.device_info->has_drm_properties_ = false;
    const auto classified = classify();
    ASSERT_TRUE(classified.has_value());
    EXPECT_TRUE(classified->empty());
    const auto acquired = acquire();
    ASSERT_TRUE(acquired.has_value());
    EXPECT_EQ(acquired->mode, DisplayAcquisitionMode::kDriverManaged);
    EXPECT_EQ(acquired->drm_fd, -1);
    expect_no_native_access();
  }
}

TEST_F(DisplayAcquisitionTest, ModeCapAppliesToTegraIgpusOnly) {
  auto& properties = g_fake.device_info->physical_device_properties_.properties;
  EXPECT_TRUE(
      holoviz::display_systems::igpu_mode_cap_applies(properties, VK_DRIVER_ID_NVIDIA_PROPRIETARY));
  const auto thor = properties;
  properties.deviceType = VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
  EXPECT_FALSE(
      holoviz::display_systems::igpu_mode_cap_applies(properties, VK_DRIVER_ID_NVIDIA_PROPRIETARY));
  properties = thor;
}

TEST_F(DisplayAcquisitionTest, EveryIdentityFieldIsRequired) {
  auto& properties = g_fake.device_info->physical_device_properties_.properties;
  const auto matches = [&](VkDriverId driver = VK_DRIVER_ID_NVIDIA_PROPRIETARY) {
    return holoviz::display_systems::uses_driver_managed_display(properties, driver);
  };
  EXPECT_TRUE(matches());
  const auto thor = properties;
  properties.vendorID = 0x1002;
  EXPECT_FALSE(matches());
  properties = thor;
  // Every Tegra iGPU takes the same path: Thor (0x2b00) and Orin (0x97ba03d7), and the
  // device ID is not part of the match.
  for (const uint32_t device_id : {0x97ba03d7u, 0x2b3fu, 0u}) {
    properties.deviceID = device_id;
    EXPECT_TRUE(matches()) << device_id;
  }
  properties = thor;
  for (const auto type : {VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU,
                          VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU,
                          VK_PHYSICAL_DEVICE_TYPE_CPU,
                          VK_PHYSICAL_DEVICE_TYPE_OTHER}) {
    properties.deviceType = type;
    EXPECT_FALSE(matches()) << type;
  }
  properties = thor;
  EXPECT_FALSE(matches(VK_DRIVER_ID_MESA_NVK));
  EXPECT_FALSE(matches(VK_DRIVER_ID_MESA_LLVMPIPE));
  std::snprintf(properties.deviceName, sizeof(properties.deviceName), "arbitrary name");
  properties.driverVersion = 0;
  EXPECT_TRUE(matches());
}

TEST_F(DisplayAcquisitionTest, NonThorExplicitAcquisitionRetainsOnlyAdapterFD) {
  select_discrete_gpu();
  auto acquired = acquire();
  ASSERT_TRUE(acquired.has_value()) << acquired.error().message;
  EXPECT_EQ(acquired->mode, DisplayAcquisitionMode::kExplicit);
  EXPECT_TRUE(fd_is_open(acquired->drm_fd));
  EXPECT_EQ(g_fake.acquire_calls, 1);
  ASSERT_EQ(g_fake.opened_paths, (std::vector<std::string>{"/dev/dri/card0"}));
  EXPECT_EQ(g_fake.realpath_calls, 0);
  holoviz::display_systems::close_acquisition_resources(*acquired);
  EXPECT_EQ(acquired->drm_fd, -1);
}

TEST_F(DisplayAcquisitionTest, InitializationFailureIsNotMasked) {
  select_discrete_gpu();
  g_fake.acquire_result = VK_ERROR_INITIALIZATION_FAILED;
  const auto acquired = acquire();
  ASSERT_FALSE(acquired.has_value());
  EXPECT_EQ(acquired.error().code, holoviz::ErrorCode::kVulkanError);
  EXPECT_EQ(g_fake.acquire_calls, 1);
  EXPECT_EQ(g_fake.opened_paths, (std::vector<std::string>{"/dev/dri/card0"}));
}

TEST_F(DisplayAcquisitionTest, InaccessibleAdapterNeverFallsBackToForeignPlatformNode) {
  select_discrete_gpu();
  g_fake.deny_adapter_open = true;
  const auto acquired = acquire();
  ASSERT_FALSE(acquired.has_value());
  EXPECT_EQ(acquired.error().code, holoviz::ErrorCode::kFailure);
  EXPECT_EQ(g_fake.acquire_calls, 0);
  EXPECT_EQ(g_fake.opened_paths, (std::vector<std::string>{"/dev/dri/card0"}));
  EXPECT_EQ(g_fake.realpath_calls, 0);
}

TEST_F(DisplayAcquisitionTest, UnknownFailureOnKMSIsNotMasked) {
  select_discrete_gpu();
  g_fake.acquire_result = VK_ERROR_UNKNOWN;
  const auto acquired = acquire();
  ASSERT_FALSE(acquired.has_value());
  EXPECT_EQ(acquired.error().code, holoviz::ErrorCode::kVulkanError);
  EXPECT_NE(acquired.error().message.find("exposes KMS"), std::string::npos);
}

TEST_F(DisplayAcquisitionTest, MissingKMSKeepsExistingImplicitReleasePolicy) {
  select_discrete_gpu();
  g_fake.acquire_result = VK_ERROR_UNKNOWN;
  g_fake.kms = false;
  const auto acquired = acquire();
  ASSERT_TRUE(acquired.has_value());
  EXPECT_EQ(acquired->mode, DisplayAcquisitionMode::kImplicit);
  EXPECT_EQ(acquired->drm_fd, -1);
  EXPECT_EQ(g_fake.opened_paths, (std::vector<std::string>{"/dev/dri/card0"}));
  EXPECT_EQ(g_fake.realpath_calls, 0);
}

TEST_F(DisplayAcquisitionTest, MissingExtensionKeepsExistingImplicitReleasePolicy) {
  select_discrete_gpu();
  instance_.acquire_drm_supported_ = false;
  const auto acquired = acquire();
  ASSERT_TRUE(acquired.has_value());
  EXPECT_EQ(acquired->mode, DisplayAcquisitionMode::kImplicit);
  expect_no_native_access();
}

TEST_F(DisplayAcquisitionTest, MasterDenialClosesFDAndNeverAcquires) {
  select_discrete_gpu();
  g_fake.master = false;
  g_fake.allow_set_master = false;
  const auto acquired = acquire();
  ASSERT_FALSE(acquired.has_value());
  EXPECT_EQ(acquired.error().code, holoviz::ErrorCode::kFailure);
  EXPECT_EQ(g_fake.acquire_calls, 0);
}

TEST_F(DisplayAcquisitionTest, UnresolvedConnectorOnlyDisablesNativeClassification) {
  select_discrete_gpu();
  g_fake.translate_result = VK_ERROR_UNKNOWN;
  const auto classified = classify();
  ASSERT_TRUE(classified.has_value());
  EXPECT_TRUE(classified->empty());
  EXPECT_EQ(g_fake.translate_calls, 1);
  // Presentation still follows the independent explicit acquisition path.
  auto acquired = acquire();
  ASSERT_TRUE(acquired.has_value());
  EXPECT_EQ(acquired->mode, DisplayAcquisitionMode::kExplicit);
  holoviz::display_systems::close_acquisition_resources(*acquired);
}

}  // namespace
