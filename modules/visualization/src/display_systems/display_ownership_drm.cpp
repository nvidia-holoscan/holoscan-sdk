/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Linux display ownership + acquisition through the kernel DRM UAPI (raw KMS
// ioctls via drm_ioctl.hpp, no libdrm link) and VK_EXT_acquire_drm_display.
// vkAcquireDrmDisplayEXT needs DRM master on the adapter's primary node, and a
// running display server (X11 or Wayland) holds it device-wide, so a display is
// acquirable only when no desktop session drives the GPU; under a session every
// connected connector classifies as kCompositor. X11 (VK_EXT_acquire_xlib_display)
// and Wayland DRM-lease acquisition are deferred.
// The qualified Thor configuration instead uses Vulkan enumeration and driver-managed
// swapchain ownership; its GPU DRM node does not control the separate display device.

#include "display_ownership_drm.hpp"
#include "display_ownership.hpp"
#include "display_ownership_policy.hpp"
#include "drm_ioctl.hpp"

#include <fcntl.h>
#include <limits.h>
#include <link.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>

#include <holoviz/logger.hpp>

#include "vulkan/vk_adapter.hpp"
#include "vulkan/vk_display_util.hpp"
#include "vulkan/vk_errors.hpp"

namespace holoviz::display_systems {

namespace {

/// An open DRM primary node; the path is kept for diagnostics.
struct OpenedNode {
  int fd = -1;
  std::string path;
};

expected<OpenedNode> open_drm_path(const char* path) {
  const int fd = open(path, O_RDWR | O_CLOEXEC);
  if (fd < 0) {
    return make_unexpected(Error{.code = ErrorCode::kFailure,
                                 .message = std::string("[holoviz] Failed to open DRM node ") +
                                            path + ": " + std::strerror(errno)});
  }
  return OpenedNode{.fd = fd, .path = path};
}

bool card_matches_pci_device(int card_index,
                             const VkPhysicalDevicePCIBusInfoPropertiesEXT& pci_properties) {
  std::array<char, 64> device_link{};
  std::snprintf(
      device_link.data(), sizeof(device_link), "/sys/class/drm/card%d/device", card_index);
  std::array<char, PATH_MAX> resolved_device{};
  if (realpath(device_link.data(), resolved_device.data()) == nullptr) {
    return false;
  }

  std::array<char, 32> pci_slot{};
  std::snprintf(pci_slot.data(),
                sizeof(pci_slot),
                "%04x:%02x:%02x.%u",
                pci_properties.pciDomain,
                pci_properties.pciBus,
                pci_properties.pciDevice,
                pci_properties.pciFunction);
  const char* path_leaf = std::strrchr(resolved_device.data(), '/');
  return path_leaf != nullptr && std::strcmp(path_leaf + 1, pci_slot.data()) == 0;
}

/// The card index the adapter claims as its own, or -1 when none does. Prefer
/// VK_EXT_physical_device_drm's major/minor join; some NVIDIA ARM drivers
/// report those fields as absent, so fall back to the adapter's
/// VK_EXT_pci_bus_info identity and the kernel's DRM sysfs links.
int find_adapter_card(const VkPhysicalDeviceDrmPropertiesEXT& drm_properties,
                      const VkPhysicalDevicePCIBusInfoPropertiesEXT& pci_properties,
                      bool has_pci_bus_info, bool* matched_by_pci) {
  for (int i = 0; i < 64; ++i) {
    std::array<char, 32> path{};
    std::snprintf(path.data(), sizeof(path), "/dev/dri/card%d", i);
    struct stat node_stat{};
    if (stat(path.data(), &node_stat) != 0 || !S_ISCHR(node_stat.st_mode)) {
      continue;
    }
    if (drm_properties.hasPrimary != 0 &&
        major(node_stat.st_rdev) == static_cast<unsigned int>(drm_properties.primaryMajor) &&
        minor(node_stat.st_rdev) == static_cast<unsigned int>(drm_properties.primaryMinor)) {
      *matched_by_pci = false;
      return i;
    }
    if (has_pci_bus_info && card_matches_pci_device(i, pci_properties)) {
      *matched_by_pci = true;
      return i;
    }
  }
  return -1;
}

/// Open only the adapter's own primary node, joined by Vulkan DRM/PCI identity.
/// Connected outputs on an unrelated platform node do not establish an association
/// with this GPU. Thor's separate display device uses the driver-managed policy.
expected<OpenedNode> open_drm_node(VkPhysicalDevice physical_device) {
  VIZ_UNWRAP(const vulkan::PhysicalDeviceInfo*,
             device_info,
             vulkan::find_physical_device_info(physical_device));

  const bool has_pci_bus_info = device_info->has_pci_bus_info_properties_;
  if (!device_info->has_drm_properties_ && !has_pci_bus_info) {
    return make_unexpected(
        Error{.code = ErrorCode::kNotImplemented,
              .message = "[holoviz] Cannot locate the adapter's DRM node: the driver does not "
                         "support " VK_EXT_PHYSICAL_DEVICE_DRM_EXTENSION_NAME
                         " or " VK_EXT_PCI_BUS_INFO_EXTENSION_NAME "."});
  }

  // Zeroed when the extension is unsupported, so hasPrimary stays false below.
  const VkPhysicalDeviceDrmPropertiesEXT& drm_properties = device_info->drm_properties_;
  const VkPhysicalDevicePCIBusInfoPropertiesEXT& pci_properties =
      device_info->pci_bus_info_properties_;

  bool matched_by_pci = false;
  const int adapter_card =
      find_adapter_card(drm_properties, pci_properties, has_pci_bus_info, &matched_by_pci);
  std::array<char, 32> adapter_path = {};
  const auto warn_if_matched_by_pci = [&]() {
    if (!matched_by_pci) {
      return;
    }
    if (drm_properties.hasPrimary == 0) {
      HOLOVIZ_LOG_WARN(
          "[holoviz] The adapter reports no primary DRM node; using PCI identity to select {}.",
          adapter_path.data());
    } else {
      HOLOVIZ_LOG_WARN(
          "[holoviz] No DRM node matches the adapter's reported major/minor; using PCI identity "
          "to select {}.",
          adapter_path.data());
    }
  };
  if (adapter_card < 0) {
    return make_unexpected(Error{
        .code = ErrorCode::kNotFound,
        .message = "[holoviz] No /dev/dri/card* node matches the adapter's DRM or PCI identity."});
  }
  std::snprintf(adapter_path.data(), sizeof(adapter_path), "/dev/dri/card%d", adapter_card);
  auto adapter_node = open_drm_path(adapter_path.data());
  if (adapter_node) {
    warn_if_matched_by_pci();
  }
  return adapter_node;
}

/// Whether the node exposes KMS. nvidia-drm registers KMS only with modeset=1; without
/// it DRM_IOCTL_MODE_GETRESOURCES fails (EOPNOTSUPP). Probing the node needs no access to
/// /sys/module/nvidia_drm/parameters/modeset, which is root-only.
bool node_exposes_kms(const drm::Node& node) {
  return drm::get_resources(node).has_value();
}

/// Log once when the libdrm.so.2 loaded into this process is NVIDIA's Tegra shim. HoloViz
/// itself no longer touches libdrm (drm_ioctl.hpp), but the Vulkan ICD or the application
/// may, and on a discrete GPU the shim cannot serve their KMS calls: the DRM display
/// acquisition may then fail for reasons outside HoloViz.
void warn_if_libdrm_shim_loaded() {
  static bool checked = false;
  if (checked) {
    return;
  }
  checked = true;
  dl_iterate_phdr(
      [](dl_phdr_info* info, size_t /*size*/, void* /*data*/) -> int {
        if (info->dlpi_name == nullptr) {
          return 0;
        }
        const std::string_view path(info->dlpi_name);
        const size_t slash = path.rfind('/');
        const std::string_view base =
            slash == std::string_view::npos ? path : path.substr(slash + 1);
        if (!base.starts_with("libdrm.so")) {
          return 0;
        }
        if (path.find("/nvidia/") != std::string_view::npos ||
            path.find("/tegra/") != std::string_view::npos) {
          HOLOVIZ_LOG_ERROR(
              "[holoviz] The libdrm loaded in this process is NVIDIA's Tegra shim ({}), which "
              "cannot drive a discrete GPU's KMS node. HoloViz does not use it, but the Vulkan "
              "driver may; if the DRM display acquisition fails, prefer the Mesa libdrm2 "
              "(e.g. LD_PRELOAD=/usr/lib/<triplet>/libdrm.so.2).",
              path);
          return 1;
        }
        return 0;
      },
      nullptr);
}

/// The connector's VRR facts: the kernel's vrr_capable property (G-Sync/Adaptive-Sync
/// handshake possible) and the VRR range floor from the EDID blob. Best-effort; a
/// connector without the properties simply reports {false, 0}.
struct ConnectorVrrInfo {
  bool vrr_capable = false;
  uint32_t vrr_min_refresh_mhz = 0;
};

ConnectorVrrInfo read_connector_vrr_info(const drm::Node& node, uint32_t connector_id) {
  ConnectorVrrInfo vrr{};
  const auto properties = drm::get_object_properties(node, connector_id, DRM_MODE_OBJECT_CONNECTOR);
  if (!properties) {
    return vrr;
  }
  uint32_t edid_min_mhz = 0;
  for (size_t i = 0; i < properties->props.size(); ++i) {
    const auto name = drm::get_property_name(node, properties->props[i]);
    if (!name) {
      continue;
    }
    const uint64_t value = properties->prop_values[i];
    if (std::strcmp(name->data(), "vrr_capable") == 0) {
      vrr.vrr_capable = value != 0;
    } else if (std::strcmp(name->data(), "EDID") == 0 && value != 0) {
      if (const auto blob = drm::get_property_blob(node, static_cast<uint32_t>(value)); blob) {
        const auto min_hz = vulkan::edid_min_vertical_rate_hz(*blob);
        if (min_hz.has_value()) {
          edid_min_mhz = *min_hz * 1000;
        }
      }
    }
  }
  // The EDID range floor is only meaningful as a VRR floor when VRR is possible at all.
  if (vrr.vrr_capable) {
    vrr.vrr_min_refresh_mhz = edid_min_mhz;
  }
  return vrr;
}

}  // namespace

// Keep adapter identity, DRM ownership, and connector matching together so fallbacks preserve
// attribution.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
expected<std::vector<NativeDisplayInfo>> classify_displays(
    const vulkan::VulkanInstance& instance, VkPhysicalDevice physical_device,
    const vulkan::PhysicalDeviceInfo& device_info) {
  if (uses_driver_managed_display(device_info.physical_device_properties_.properties,
                                  device_info.driver_properties_.driverID)) {
    // GPU DRM ownership cannot classify the separate Tegra display device. Preserve
    // Vulkan's enumeration with kUnknown ownership; the swapchain checks availability.
    return std::vector<NativeDisplayInfo>{};
  }
  if (!instance.acquire_drm_supported_ || vkGetDrmDisplayEXT == nullptr) {
    return make_unexpected(Error{.code = ErrorCode::kNotImplemented,
                                 .message =
                                     "[holoviz] " VK_EXT_ACQUIRE_DRM_DISPLAY_EXTENSION_NAME
                                     " is unavailable; DRM display classification disabled."});
  }

  warn_if_libdrm_shim_loaded();
  VIZ_UNWRAP(OpenedNode, opened, open_drm_node(physical_device));
  const drm::Node node{.fd = opened.fd};

  // Another DRM master running means a display server composites this GPU.
  // (Opening the node may have made this fd the implicit master on a bare
  // console; that reads as "no other master", which is correct there.)
  bool other_master_present = false;
  if (!drm::is_master(node)) {
    if (drm::set_master(node)) {
      (void)drm::drop_master(node);
    } else {
      other_master_present = true;
    }
  }

  const auto resources = drm::get_resources(node);
  if (!resources) {
    close(node.fd);
    return make_unexpected(Error{.code = ErrorCode::kFailure,
                                 .message = resources.error().message + " on " + opened.path +
                                            "; DRM display classification disabled."});
  }

  std::vector<NativeDisplayInfo> displays;
  for (const uint32_t connector_id : resources->connectors) {
    const auto connector = drm::get_connector(node, connector_id);
    if (!connector || connector->connection != drm::kConnectionConnected) {
      continue;
    }

    // The VkDisplayKHR join: the driver resolves connector ids on its own node.
    VkDisplayKHR vk_display = VK_NULL_HANDLE;
    if (vkGetDrmDisplayEXT(physical_device, node.fd, connector_id, &vk_display) != VK_SUCCESS ||
        vk_display == VK_NULL_HANDLE) {
      // Dropping this silently leaves the caller reporting "no displays connected"
      // for a panel that is lit, so name what was found and where.
      const char* unresolved_type_name = drm::connector_type_name(connector->connector_type);
      HOLOVIZ_LOG_WARN(
          "[holoviz] {}-{} is connected on {}, but this adapter cannot resolve it; "
          "DRM classification is unavailable for that output; Vulkan enumeration is retained.",
          unresolved_type_name != nullptr ? unresolved_type_name : "connector",
          connector->connector_type_id,
          opened.path);
      continue;
    }

    bool has_active_crtc = false;
    DisplayMode current_mode{};
    if (connector->encoder_id != 0) {
      if (const auto encoder = drm::get_encoder(node, connector->encoder_id);
          encoder && encoder->crtc_id != 0) {
        if (const auto crtc = drm::get_crtc(node, encoder->crtc_id); crtc && crtc->mode_valid) {
          has_active_crtc = true;
          current_mode = DisplayMode{
              .resolution = {crtc->mode.hdisplay, crtc->mode.vdisplay},
              .refresh_rate_mhz = crtc->mode.vrefresh * 1000,
          };
        }
      }
    }

    NativeDisplayInfo info{};
    info.vk_display = reinterpret_cast<uint64_t>(vk_display);
    info.ownership = vulkan::classify_drm_connector(true, has_active_crtc, other_master_present);
    info.current_mode = current_mode;
    const ConnectorVrrInfo vrr = read_connector_vrr_info(node, connector_id);
    info.vrr_capable = vrr.vrr_capable;
    info.vrr_min_refresh_mhz = vrr.vrr_min_refresh_mhz;
    const char* type_name = drm::connector_type_name(connector->connector_type);
    std::snprintf(info.name.data(),
                  info.name.size(),
                  "%s-%u",
                  type_name != nullptr ? type_name : "connector",
                  connector->connector_type_id);
    displays.push_back(info);
  }

  close(node.fd);

  return displays;
}

expected<DisplayAcquisition> acquire_display(const vulkan::VulkanInstance& instance,
                                             VkPhysicalDevice physical_device,
                                             uint64_t vk_display) {
  VIZ_UNWRAP(const vulkan::PhysicalDeviceInfo*,
             device_info,
             vulkan::find_physical_device_info(physical_device));
  if (uses_driver_managed_display(device_info->physical_device_properties_.properties,
                                  device_info->driver_properties_.driverID)) {
    HOLOVIZ_LOG_DEBUG(
        "[holoviz] Using driver-managed display ownership on this Tegra iGPU; "
        "availability is checked at swapchain creation.");
    return DisplayAcquisition{.mode = DisplayAcquisitionMode::kDriverManaged};
  }

  if (!instance.acquire_drm_supported_ || vkAcquireDrmDisplayEXT == nullptr) {
    HOLOVIZ_LOG_WARN(
        "[holoviz] " VK_EXT_ACQUIRE_DRM_DISPLAY_EXTENSION_NAME
        " is unavailable; continuing without acquiring the display (the driver decides at "
        "swapchain creation whether the display is free).");
    return DisplayAcquisition{};
  }

  warn_if_libdrm_shim_loaded();
  VIZ_UNWRAP(OpenedNode, opened, open_drm_node(physical_device));
  const int fd = opened.fd;
  const drm::Node node{.fd = fd};

  // vkAcquireDrmDisplayEXT requires a primary-node fd with DRM master
  // permissions. Opening the first unused primary node usually grants master
  // implicitly, but that is not guaranteed across DRM drivers and container
  // runtimes.
  if (!drm::is_master(node)) {
    if (const auto set = drm::set_master(node); !set) {
      close(fd);
      return make_unexpected(
          Error{.code = ErrorCode::kFailure,
                .message = set.error().message + " on " + opened.path +
                           ": cannot become DRM master. Stop the display server or other DRM "
                           "master, then retry."});
    }
  }

  const VkResult result =
      vkAcquireDrmDisplayEXT(physical_device, fd, reinterpret_cast<VkDisplayKHR>(vk_display));
  if (result != VK_SUCCESS) {
    // Preserve the adapter-without-KMS fallback. An arbitrary Vulkan failure on
    // a KMS node must not select driver-managed ownership or hide the error.
    const bool kms_active = result == VK_ERROR_UNKNOWN && node_exposes_kms(node);
    close(fd);
    if (result == VK_ERROR_UNKNOWN) {
      // With KMS active (nvidia-drm modeset=1) the driver only hands a head over
      // through this call; proceeding unacquired then always dies later at
      // vkCreateSwapchainKHR with VK_ERROR_INITIALIZATION_FAILED, far from the
      // cause. Fail here instead. Without KMS the driver takes the display at
      // swapchain creation, so the adapter-without-KMS fallback stays.
      if (kms_active) {
        return make_unexpected(
            Error{.code = ErrorCode::kVulkanError,
                  .message = "[holoviz] vkAcquireDrmDisplayEXT returned VK_ERROR_UNKNOWN on " +
                             opened.path +
                             " while the node exposes KMS (nvidia-drm modeset=1): the display "
                             "cannot be taken without the DRM handoff. Check that this process "
                             "holds DRM master on that node (no display server or other DRM "
                             "master on the GPU) and has access to it (video group, container "
                             "device binding), then retry."});
      }
      HOLOVIZ_LOG_WARN(
          "[holoviz] vkAcquireDrmDisplayEXT returned VK_ERROR_UNKNOWN on a node without KMS; "
          "continuing without explicit acquisition (the driver decides at swapchain creation "
          "whether the display is free).");
      return DisplayAcquisition{};
    }
    return make_unexpected(Error{
        .code = ErrorCode::kVulkanError,
        .message = std::string("[holoviz] vkAcquireDrmDisplayEXT failed (") +
                   vulkan::vk_error_string(result, __FILE__, __LINE__) +
                   "). Stop the display server or compositor holding DRM master on this GPU (run "
                   "from a text console), then retry."});
  }

  return DisplayAcquisition{.mode = DisplayAcquisitionMode::kExplicit, .drm_fd = fd};
}

void close_acquisition_resources(DisplayAcquisition& acquisition) {
  if (acquisition.drm_fd >= 0) {
    close(acquisition.drm_fd);
    acquisition.drm_fd = -1;
  }
}

}  // namespace holoviz::display_systems
