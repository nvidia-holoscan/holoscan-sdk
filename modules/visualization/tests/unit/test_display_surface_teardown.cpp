/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Pins the direct-display surface's ownership/teardown policy through the public Surface
// wrapper, GPU-free: vulkan/vk_display_surface.cpp is compiled into this test with fake volk
// entry points (the DSO's copies are hidden, so the definitions below are the ones that TU
// binds to) and a real pipe fd standing in for the DRM fd. The policy under test: a release
// the driver rejects is reported, but the display handle and the fd are still relinquished
// exactly once, and the public handle empties — no owner-less fd, no double release.

#include <gtest/gtest.h>
#include <array>

#include <fcntl.h>
#include <unistd.h>

#include <cerrno>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <holoviz/logger.hpp>
#include <holoviz/surface.hpp>

#include "display_systems/display_ownership.hpp"
#include "vulkan/vk_display_surface.hpp"

namespace {

struct FakeVulkan {
  int release_calls = 0;
  int destroy_surface_calls = 0;
  VkResult release_result = VK_SUCCESS;
  int fd_at_release = -1;           // the fd handed to the surface, sampled when released
  bool fd_open_at_release = false;  // release-before-close ordering witness
  int close_calls = 0;
  std::vector<std::string> error_logs;
};

FakeVulkan g_fake;

bool fd_is_open(int fd) {
  return fcntl(fd, F_GETFD) != -1 || errno != EBADF;
}

VKAPI_ATTR VkResult VKAPI_CALL fake_release_display(VkPhysicalDevice, VkDisplayKHR) {
  ++g_fake.release_calls;
  g_fake.fd_open_at_release = g_fake.fd_at_release >= 0 && fd_is_open(g_fake.fd_at_release);
  return g_fake.release_result;
}

VKAPI_ATTR void VKAPI_CALL fake_destroy_surface(VkInstance, VkSurfaceKHR,
                                                const VkAllocationCallbacks*) {
  ++g_fake.destroy_surface_calls;
}

void capture_log(holoviz::LogLevel level, const char*, int, const char*, const char* message,
                 void*) noexcept {
  if (level == holoviz::LogLevel::kError) {
    g_fake.error_logs.emplace_back(message);
  }
}

}  // namespace

// volk declares these as writable globals; the surface TU calls through them.
PFN_vkReleaseDisplayEXT vkReleaseDisplayEXT = fake_release_display;
PFN_vkDestroySurfaceKHR vkDestroySurfaceKHR = fake_destroy_surface;

namespace holoviz::display_systems {

// Stands in for display_ownership_drm.cpp: same contract (close and reset the fd), counted.
void close_acquisition_resources(DisplayAcquisition& acquisition) {
  ++g_fake.close_calls;
  if (acquisition.drm_fd >= 0) {
    close(acquisition.drm_fd);
    acquisition.drm_fd = -1;
  }
}

}  // namespace holoviz::display_systems

namespace holoviz::vulkan {

// Stands in for vk_errors.cpp (which needs the Vulkan utility headers).
std::string vk_error_string(VkResult result, const char* file, int32_t line) {
  return "VkResult " + std::to_string(static_cast<int>(result)) + " at " + file + ":" +
         std::to_string(line);
}

}  // namespace holoviz::vulkan

namespace holoviz::vulkan {
namespace {

const auto kFakeInstance = reinterpret_cast<VkInstance>(uintptr_t{0x10});
const auto kFakeSurface = reinterpret_cast<VkSurfaceKHR>(uintptr_t{0x20});
const auto kFakePhysicalDevice = reinterpret_cast<VkPhysicalDevice>(uintptr_t{0x30});
const auto kFakeDisplay = reinterpret_cast<VkDisplayKHR>(uintptr_t{0x40});

class DisplaySurfaceTeardown : public ::testing::Test {
 protected:
  void SetUp() override {
    g_fake = FakeVulkan{};
    set_log_handler(&capture_log, nullptr);
  }
  void TearDown() override {
    reset_log_handler();
    if (fd_ >= 0 && fd_is_open(fd_)) {
      close(fd_);
    }
    if (other_end_ >= 0) {
      close(other_end_);
    }
  }

  /// A fresh pipe's read end (fd_) stands in for the DRM fd; its liveness is observable via
  /// fcntl.
  void make_fd() {
    std::array<int, 2> fds{};
    ASSERT_EQ(pipe(fds.data()), 0);
    fd_ = fds[0];
    other_end_ = fds[1];
    g_fake.fd_at_release = fd_;
  }

  /// The public handle over a native display surface with the given acquisition state.
  static Surface make_surface(const display_systems::DisplayAcquisition& acquisition) {
    auto impl = std::make_unique<VulkanDisplaySurface>();
    EXPECT_TRUE(impl->init(kFakeInstance,
                           nullptr,
                           kFakeSurface,
                           Extent2D{1920, 1080},
                           kFakePhysicalDevice,
                           kFakeDisplay,
                           acquisition)
                    .has_value());
    return Surface{std::move(impl)};
  }

  int fd_ = -1;
  int other_end_ = -1;
};

TEST_F(DisplaySurfaceTeardown, SuccessfulDestroyReleasesThenClosesExactlyOnce) {
  make_fd();
  const int fd = fd_;
  Surface surface = make_surface({.mode = DisplayAcquisitionMode::kExplicit, .drm_fd = fd});
  ASSERT_TRUE(surface);

  ASSERT_TRUE(surface.destroy().has_value());
  EXPECT_FALSE(surface);
  EXPECT_EQ(g_fake.destroy_surface_calls, 1);
  EXPECT_EQ(g_fake.release_calls, 1);
  EXPECT_EQ(g_fake.close_calls, 1);
  EXPECT_TRUE(g_fake.fd_open_at_release) << "the fd must outlive the Vulkan release";
  EXPECT_FALSE(fd_is_open(fd));

  // Repeated destroy on the emptied handle: no-op success, nothing touched again.
  ASSERT_TRUE(surface.destroy().has_value());
  EXPECT_EQ(g_fake.release_calls, 1);
  EXPECT_EQ(g_fake.close_calls, 1);
  EXPECT_TRUE(g_fake.error_logs.empty());
}

TEST_F(DisplaySurfaceTeardown, RejectedReleaseIsTerminalThroughExplicitDestroy) {
  make_fd();
  const int fd = fd_;
  g_fake.release_result = VK_ERROR_UNKNOWN;
  Surface surface = make_surface({.mode = DisplayAcquisitionMode::kExplicit, .drm_fd = fd});

  const auto destroyed = surface.destroy();
  ASSERT_FALSE(destroyed.has_value());
  EXPECT_EQ(destroyed.error().code, ErrorCode::kVulkanError);
  EXPECT_NE(destroyed.error().message.find("vkReleaseDisplayEXT"), std::string::npos);

  // The handle empties (Surface::destroy contract) and ownership was fully relinquished:
  // exactly one release attempt, exactly one close, fd gone, ordering preserved.
  EXPECT_FALSE(surface);
  EXPECT_EQ(g_fake.release_calls, 1);
  EXPECT_EQ(g_fake.close_calls, 1);
  EXPECT_TRUE(g_fake.fd_open_at_release);
  EXPECT_FALSE(fd_is_open(fd));

  // The destructor already ran inside destroy(); it must not have retried the release.
  g_fake.release_result = VK_SUCCESS;
  ASSERT_TRUE(surface.destroy().has_value());
  EXPECT_EQ(g_fake.release_calls, 1);
  EXPECT_EQ(g_fake.close_calls, 1);
}

TEST_F(DisplaySurfaceTeardown, RejectedReleaseInDestructorStillCleansUpAndLogs) {
  make_fd();
  const int fd = fd_;
  g_fake.release_result = VK_ERROR_UNKNOWN;
  {
    const Surface surface = make_surface({.mode = DisplayAcquisitionMode::kExplicit, .drm_fd = fd});
    ASSERT_TRUE(surface);
  }
  EXPECT_EQ(g_fake.destroy_surface_calls, 1);
  EXPECT_EQ(g_fake.release_calls, 1);
  EXPECT_EQ(g_fake.close_calls, 1);
  EXPECT_FALSE(fd_is_open(fd));
  bool destructor_logged = false;
  for (const std::string& line : g_fake.error_logs) {
    destructor_logged |= line.find("destroy failed in destructor") != std::string::npos;
  }
  EXPECT_TRUE(destructor_logged);
}

// Keep the existing non-Thor fallback's explicit-release behavior when the extension
// is unavailable or the adapter has no KMS, even though no DRM fd was acquired.
TEST_F(DisplaySurfaceTeardown, ImplicitDriverHoldIsReleasedWithoutOSResources) {
  Surface surface = make_surface({.mode = DisplayAcquisitionMode::kImplicit, .drm_fd = -1});
  ASSERT_TRUE(surface.destroy().has_value());
  EXPECT_FALSE(surface);
  EXPECT_EQ(g_fake.release_calls, 1);
  EXPECT_EQ(g_fake.close_calls, 1);  // called, closes nothing
}

// An implicit hold on the adapter's own node keeps reporting a rejected release: the driver may
// still hold the display, and nothing about that path makes the rejection expected.
TEST_F(DisplaySurfaceTeardown, RejectedImplicitReleaseIsReported) {
  g_fake.release_result = VK_ERROR_UNKNOWN;
  Surface surface = make_surface({.mode = DisplayAcquisitionMode::kImplicit, .drm_fd = -1});

  const auto destroyed = surface.destroy();
  ASSERT_FALSE(destroyed.has_value());
  EXPECT_EQ(destroyed.error().code, ErrorCode::kVulkanError);
  EXPECT_FALSE(surface);
  EXPECT_EQ(g_fake.release_calls, 1);
}

// Thor ownership belongs to swapchains, not surfaces. Even a driver configured to reject
// vkReleaseDisplayEXT must see no release call on surface teardown or creation unwind.
TEST_F(DisplaySurfaceTeardown, DriverManagedSurfaceNeverCallsExplicitRelease) {
  g_fake.release_result = VK_ERROR_UNKNOWN;
  Surface surface = make_surface({.mode = DisplayAcquisitionMode::kDriverManaged});

  ASSERT_TRUE(surface.destroy().has_value());
  ASSERT_TRUE(surface.destroy().has_value());
  EXPECT_FALSE(surface);
  EXPECT_EQ(g_fake.destroy_surface_calls, 1);
  EXPECT_EQ(g_fake.release_calls, 0);
  EXPECT_EQ(g_fake.close_calls, 1);
  EXPECT_TRUE(g_fake.error_logs.empty());
}

TEST_F(DisplaySurfaceTeardown, DriverManagedUnwindNeverCallsExplicitRelease) {
  g_fake.release_result = VK_ERROR_UNKNOWN;
  display_systems::DisplayAcquisition acquisition{.mode = DisplayAcquisitionMode::kDriverManaged};
  EXPECT_EQ(release_display_then_close(kFakePhysicalDevice, kFakeDisplay, acquisition), VK_SUCCESS);
  EXPECT_EQ(g_fake.release_calls, 0);
  EXPECT_EQ(acquisition.drm_fd, -1);
}

TEST_F(DisplaySurfaceTeardown, ExplicitAcquisitionUnwindReleasesBeforeClosing) {
  make_fd();
  const int fd = fd_;
  display_systems::DisplayAcquisition acquisition{.mode = DisplayAcquisitionMode::kExplicit,
                                                  .drm_fd = fd};
  g_fake.release_result = VK_ERROR_UNKNOWN;
  EXPECT_EQ(release_display_then_close(kFakePhysicalDevice, kFakeDisplay, acquisition),
            VK_ERROR_UNKNOWN);
  EXPECT_EQ(g_fake.release_calls, 1);
  EXPECT_TRUE(g_fake.fd_open_at_release);
  EXPECT_FALSE(fd_is_open(fd));
  EXPECT_EQ(acquisition.drm_fd, -1);
}

TEST_F(DisplaySurfaceTeardown, NativeObjectsAnswerSurfaceAndDisplay) {
  Surface surface = make_surface({.mode = DisplayAcquisitionMode::kImplicit, .drm_fd = -1});
  EXPECT_EQ(surface.native_object(NativeObjectType::kVkSurfaceKHR)->value,
            reinterpret_cast<uint64_t>(kFakeSurface));
  EXPECT_EQ(surface.native_object(NativeObjectType::kVkDisplayKHR)->value,
            reinterpret_cast<uint64_t>(kFakeDisplay));
  EXPECT_EQ(surface.extent()->width, 1920U);
  EXPECT_FALSE(surface.resizable().value());
  ASSERT_TRUE(surface.destroy().has_value());
}

TEST_F(DisplaySurfaceTeardown, ReportsItsAcquisitionMode) {
  for (const DisplayAcquisitionMode mode : {DisplayAcquisitionMode::kImplicit,
                                            DisplayAcquisitionMode::kExplicit,
                                            DisplayAcquisitionMode::kDriverManaged}) {
    Surface surface = make_surface({.mode = mode});
    const auto reported = surface.display_acquisition_mode();
    ASSERT_TRUE(reported.has_value()) << reported.error().message;
    EXPECT_EQ(*reported, mode);
    ASSERT_TRUE(surface.destroy().has_value());
  }
  const Surface empty;
  const auto reported = empty.display_acquisition_mode();
  ASSERT_FALSE(reported.has_value());
  EXPECT_EQ(reported.error().code, ErrorCode::kInvalidState);
}

}  // namespace
}  // namespace holoviz::vulkan
