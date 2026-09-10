/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Linux display ownership + acquisition: DRM (VK_EXT_acquire_drm_display) as
// the base path, with an X11/RandR overlay (display_ownership_x11.cpp,
// VK_EXT_acquire_xlib_display) when the session is X11. The split matters
// because under X the DRM view is doubly wrong: the X server holds DRM master
// device-wide (so vkAcquireDrmDisplayEXT always fails, even for detached
// outputs) and the NVIDIA X driver modesets outside KMS (so every connector
// reads inactive). RandR classification also names outputs the way users see
// them in xrandr instead of by kernel connector. Wayland DRM-lease is deferred;
// Wayland sessions (including XWayland) keep the plain DRM classification.

#include "display_ownership.hpp"

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <unistd.h>
#include <xf86drm.h>
#include <xf86drmMode.h>

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <span>
#include <string>
#include <vector>

#include <holoviz/logger.hpp>

#include "rhi/vulkan/vk_display_util.hpp"
#include "rhi/vulkan/vk_errors.hpp"

#ifdef HOLOVIZ_HAS_XLIB_XRANDR
#include "display_ownership_x11.hpp"
#endif

namespace holoviz::rhi::display_systems {

namespace {

#ifdef HOLOVIZ_HAS_XLIB_XRANDR
bool x11_path_preferred() {
  return vulkan::prefer_x11_display_path(
      std::getenv("DISPLAY"), std::getenv("WAYLAND_DISPLAY"), std::getenv("XDG_SESSION_TYPE"));
}
#endif

/// Open the primary DRM node backing `physical_device`, joined through
/// VK_EXT_physical_device_drm's primary major/minor numbers.
expected<int> open_drm_node(VkPhysicalDevice physical_device) {
  uint32_t extension_count = 0;
  VIZ_VK_FORWARD_ERROR(
      vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &extension_count, nullptr));
  std::vector<VkExtensionProperties> extensions(extension_count);
  VIZ_VK_FORWARD_ERROR(vkEnumerateDeviceExtensionProperties(
      physical_device, nullptr, &extension_count, extensions.data()));

  const bool has_drm_properties =
      std::any_of(extensions.begin(), extensions.end(), [](const VkExtensionProperties& ext) {
        return std::strcmp(ext.extensionName, VK_EXT_PHYSICAL_DEVICE_DRM_EXTENSION_NAME) == 0;
      });
  if (!has_drm_properties) {
    return make_unexpected(
        Error{.code = ErrorCode::kNotImplemented,
              .message = "[holoviz] Cannot locate the adapter's DRM node: the driver does not "
                         "support " VK_EXT_PHYSICAL_DEVICE_DRM_EXTENSION_NAME "."});
  }

  VkPhysicalDeviceDrmPropertiesEXT drm_properties{
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRM_PROPERTIES_EXT,
  };
  VkPhysicalDeviceProperties2 properties2{
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
      .pNext = &drm_properties,
  };
  vkGetPhysicalDeviceProperties2(physical_device, &properties2);

  if (!drm_properties.hasPrimary) {
    return make_unexpected(Error{.code = ErrorCode::kNotFound,
                                 .message = "[holoviz] The adapter reports no primary DRM node."});
  }

  for (int i = 0; i < 64; ++i) {
    char path[32];
    std::snprintf(path, sizeof(path), "/dev/dri/card%d", i);
    struct stat node_stat{};
    if (stat(path, &node_stat) != 0 || !S_ISCHR(node_stat.st_mode)) {
      continue;
    }
    if (major(node_stat.st_rdev) != static_cast<unsigned int>(drm_properties.primaryMajor) ||
        minor(node_stat.st_rdev) != static_cast<unsigned int>(drm_properties.primaryMinor)) {
      continue;
    }
    const int fd = open(path, O_RDWR | O_CLOEXEC);
    if (fd < 0) {
      return make_unexpected(Error{.code = ErrorCode::kFailure,
                                   .message = std::string("[holoviz] Failed to open DRM node ") +
                                              path + ": " + std::strerror(errno)});
    }
    return fd;
  }

  return make_unexpected(
      Error{.code = ErrorCode::kNotFound,
            .message = "[holoviz] No /dev/dri/card* node matches the adapter's DRM major/minor."});
}

}  // namespace

expected<std::vector<NativeDisplayInfo>> classify_displays(
    const vulkan::VulkanInstance& instance, VkPhysicalDevice physical_device,
    const vulkan::PhysicalDeviceInfo& /*device_info*/) {
  if (!instance.acquire_drm_supported_ || vkGetDrmDisplayEXT == nullptr) {
    return make_unexpected(Error{.code = ErrorCode::kNotImplemented,
                                 .message =
                                     "[holoviz] " VK_EXT_ACQUIRE_DRM_DISPLAY_EXTENSION_NAME
                                     " is unavailable; DRM display classification disabled."});
  }

  VIZ_UNWRAP(int, fd, open_drm_node(physical_device));

  // Another DRM master running means a display server composites this GPU.
  // (Opening the node may have made this fd the implicit master on a bare
  // console; that reads as "no other master", which is correct there.)
  bool other_master_present = false;
  if (drmIsMaster(fd) == 0) {
    if (drmSetMaster(fd) == 0) {
      drmDropMaster(fd);
    } else {
      other_master_present = true;
    }
  }

  drmModeRes* resources = drmModeGetResources(fd);
  if (resources == nullptr) {
    close(fd);
    return make_unexpected(
        Error{.code = ErrorCode::kFailure,
              .message = "[holoviz] drmModeGetResources failed; DRM display classification "
                         "disabled."});
  }

  std::vector<NativeDisplayInfo> displays;
  for (int i = 0; i < resources->count_connectors; ++i) {
    const uint32_t connector_id = resources->connectors[i];
    drmModeConnector* connector = drmModeGetConnector(fd, connector_id);
    if (connector == nullptr) {
      continue;
    }
    if (connector->connection != DRM_MODE_CONNECTED) {
      drmModeFreeConnector(connector);
      continue;
    }

    // The VkDisplayKHR join: the driver resolves connector ids on its own node.
    VkDisplayKHR vk_display = VK_NULL_HANDLE;
    if (vkGetDrmDisplayEXT(physical_device, fd, connector_id, &vk_display) != VK_SUCCESS ||
        vk_display == VK_NULL_HANDLE) {
      drmModeFreeConnector(connector);
      continue;
    }

    bool has_active_crtc = false;
    DisplayMode current_mode{};
    if (connector->encoder_id != 0) {
      if (drmModeEncoder* encoder = drmModeGetEncoder(fd, connector->encoder_id);
          encoder != nullptr) {
        if (encoder->crtc_id != 0) {
          if (drmModeCrtc* crtc = drmModeGetCrtc(fd, encoder->crtc_id); crtc != nullptr) {
            if (crtc->mode_valid != 0) {
              has_active_crtc = true;
              current_mode = DisplayMode{
                  .resolution = {crtc->mode.hdisplay, crtc->mode.vdisplay},
                  .refresh_rate_mhz = crtc->mode.vrefresh * 1000,
              };
            }
            drmModeFreeCrtc(crtc);
          }
        }
        drmModeFreeEncoder(encoder);
      }
    }

    NativeDisplayInfo info{};
    info.vk_display = reinterpret_cast<uint64_t>(vk_display);
    info.ownership = vulkan::classify_drm_connector(true, has_active_crtc, other_master_present);
    info.current_mode = current_mode;
    const char* type_name = drmModeGetConnectorTypeName(connector->connector_type);
    std::snprintf(info.name.data(),
                  info.name.size(),
                  "%s-%u",
                  type_name != nullptr ? type_name : "connector",
                  connector->connector_type_id);
    displays.push_back(info);

    drmModeFreeConnector(connector);
  }

  drmModeFreeResources(resources);
  close(fd);

#ifdef HOLOVIZ_HAS_XLIB_XRANDR
  // X11 overlay: where the session is X11, RandR is authoritative — it sees
  // scanout the NVIDIA X driver performs outside KMS and names outputs the way
  // xrandr does. Outputs RandR cannot resolve keep their DRM classification.
  if (instance.acquire_xlib_supported_ && vkGetRandROutputDisplayEXT != nullptr &&
      x11_path_preferred()) {
    auto x11_displays = x11::classify_outputs(physical_device);
    if (x11_displays) {
      vulkan::merge_native_display_info(displays,
                                        std::span<const NativeDisplayInfo>(x11_displays.value()));
    } else {
      HOLOVIZ_LOG_WARN(
          "[holoviz] X11 display classification unavailable, keeping the DRM "
          "view: {}",
          x11_displays.error().message);
    }
  }
#endif

  return displays;
}

expected<DisplayAcquisition> acquire_display(const vulkan::VulkanInstance& instance,
                                             VkPhysicalDevice physical_device,
                                             uint64_t vk_display) {
#ifdef HOLOVIZ_HAS_XLIB_XRANDR
  // Under X11 the X server holds DRM master device-wide, so the DRM acquire
  // below cannot succeed — go through the X server instead. A failed X11
  // acquire downgrades to "proceed unacquired" rather than an error: NVIDIA
  // only implements vkAcquireXlibDisplayEXT for displays on its HMD allowlist
  // and rejects ordinary monitors with VK_ERROR_UNKNOWN, yet still permits
  // direct scanout to a display the X server is not driving — the driver has
  // the final word at surface/swapchain creation (the legacy exclusive-display
  // flow, and the same contract as the missing-extension case below).
  if (instance.acquire_xlib_supported_ && vkAcquireXlibDisplayEXT != nullptr &&
      x11_path_preferred()) {
    auto x11_acquisition = x11::acquire(physical_device, vk_display);
    if (x11_acquisition) {
      return DisplayAcquisition{.acquired = true,
                                .x11_connection = x11_acquisition->x11_connection};
    }
    HOLOVIZ_LOG_WARN(
        "[holoviz] Could not acquire the display from the X server, continuing "
        "without acquiring it (the driver decides at surface creation whether "
        "the display is free): {}",
        x11_acquisition.error().message);
    return DisplayAcquisition{};
  }
#endif

  if (!instance.acquire_drm_supported_ || vkAcquireDrmDisplayEXT == nullptr) {
    HOLOVIZ_LOG_WARN(
        "[holoviz] " VK_EXT_ACQUIRE_DRM_DISPLAY_EXTENSION_NAME
        " is unavailable; continuing without acquiring the display (the driver decides at "
        "surface creation whether the display is free).");
    return DisplayAcquisition{};
  }

  VIZ_UNWRAP(int, fd, open_drm_node(physical_device));

  const VkResult result =
      vkAcquireDrmDisplayEXT(physical_device, fd, reinterpret_cast<VkDisplayKHR>(vk_display));
  if (result != VK_SUCCESS) {
    close(fd);
    return make_unexpected(Error{
        .code = ErrorCode::kVulkanError,
        .message = std::string("[holoviz] vkAcquireDrmDisplayEXT failed (") +
                   vulkan::vk_error_string(result, __FILE__, __LINE__) +
                   "). Free the display from the desktop session first (stop the compositor or "
                   "detach the output), then retry."});
  }

  return DisplayAcquisition{.acquired = true, .drm_fd = fd};
}

void close_acquisition_resources(DisplayAcquisition& acquisition) {
  if (acquisition.drm_fd >= 0) {
    close(acquisition.drm_fd);
    acquisition.drm_fd = -1;
  }
#ifdef HOLOVIZ_HAS_XLIB_XRANDR
  x11::close_connection(acquisition.x11_connection);
#endif
  acquisition.x11_connection = nullptr;
}

}  // namespace holoviz::rhi::display_systems
