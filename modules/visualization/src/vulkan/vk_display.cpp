/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "vk_display.hpp"
#include "../detail/destroy_timeout.hpp"

#include <algorithm>
#include <cstdio>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <holoviz/adapter.hpp>
#include <holoviz/logger.hpp>

#include "../display_systems/display_ownership.hpp"
#include "vk_adapter.hpp"
#include "vk_display_util.hpp"
#include "vk_errors.hpp"

namespace holoviz::vulkan {

// vk_display_util.hpp mirrors these bits so the pure helpers stay free of
// Vulkan headers; keep the mirror honest.
static_assert(kDisplayPlaneAlphaOpaque == VK_DISPLAY_PLANE_ALPHA_OPAQUE_BIT_KHR);
static_assert(kDisplayPlaneAlphaGlobal == VK_DISPLAY_PLANE_ALPHA_GLOBAL_BIT_KHR);
static_assert(kDisplayPlaneAlphaPerPixel == VK_DISPLAY_PLANE_ALPHA_PER_PIXEL_BIT_KHR);
static_assert(kDisplayPlaneAlphaPerPixelPremultiplied ==
              VK_DISPLAY_PLANE_ALPHA_PER_PIXEL_PREMULTIPLIED_BIT_KHR);

namespace {

expected<std::vector<VkDisplayPropertiesKHR>> get_display_properties(
    VkPhysicalDevice physical_device) {
  uint32_t display_count = 0;
  VIZ_VK_FORWARD_ERROR(
      vkGetPhysicalDeviceDisplayPropertiesKHR(physical_device, &display_count, nullptr));
  std::vector<VkDisplayPropertiesKHR> display_properties(display_count);
  VIZ_VK_FORWARD_ERROR(vkGetPhysicalDeviceDisplayPropertiesKHR(
      physical_device, &display_count, display_properties.data()));
  display_properties.resize(display_count);
  return display_properties;
}

expected<std::vector<VkDisplayModePropertiesKHR>> get_display_mode_properties(
    VkPhysicalDevice physical_device, VkDisplayKHR display) {
  uint32_t mode_count = 0;
  VIZ_VK_FORWARD_ERROR(
      vkGetDisplayModePropertiesKHR(physical_device, display, &mode_count, nullptr));
  std::vector<VkDisplayModePropertiesKHR> mode_properties(mode_count);
  VIZ_VK_FORWARD_ERROR(
      vkGetDisplayModePropertiesKHR(physical_device, display, &mode_count, mode_properties.data()));
  mode_properties.resize(mode_count);
  return mode_properties;
}

DisplayMode to_display_mode(const VkDisplayModeParametersKHR& parameters) {
  return DisplayMode{
      .resolution = {parameters.visibleRegion.width, parameters.visibleRegion.height},
      .refresh_rate_mhz = parameters.refreshRate,
  };
}

expected<const PhysicalDeviceInfo*> resolve_display_device(const Display& display) {
  VIZ_UNWRAP(VulkanInstance, instance, GetVulkanInstance());
  if (!instance.display_supported_) {
    return make_unexpected(
        Error{.code = ErrorCode::kNotImplemented,
              .message = "[holoviz] Direct display is unavailable: the Vulkan instance does not "
                         "support " VK_KHR_DISPLAY_EXTENSION_NAME "."});
  }
  if (display.display_id == 0) {
    return make_unexpected(Error{.code = ErrorCode::kInvalidState,
                                 .message = "[holoviz] Display: empty display handle."});
  }
  return find_physical_device_info(display.adapter.uuid);
}

/// Mode/plane/alpha selection plus surface creation — everything after the
/// acquisition, split out so the caller can release the display when any step
/// fails without threading cleanup through each early return.
struct DisplaySurfaceCreation {
  VkSurfaceKHR surface = VK_NULL_HANDLE;
  Extent2D extent{};
};

expected<DisplaySurfaceCreation> create_plane_surface(const VulkanInstance& instance,
                                                      VkPhysicalDevice physical_device,
                                                      VkDisplayKHR display,
                                                      const DisplaySurfaceDesc& desc) {
  // Pick the closest supported mode (0 = don't care, prefer largest/fastest).
  VIZ_UNWRAP(std::vector<VkDisplayModePropertiesKHR>,
             mode_properties,
             get_display_mode_properties(physical_device, display));
  if (mode_properties.empty()) {
    return make_unexpected(Error{.code = ErrorCode::kNotFound,
                                 .message = "[holoviz] Display reports no display modes."});
  }

  std::vector<DisplayModeInfo> modes;
  modes.reserve(mode_properties.size());
  for (const auto& properties : mode_properties) {
    modes.push_back(DisplayModeInfo{
        .handle = reinterpret_cast<uint64_t>(properties.displayMode),
        .mode = to_display_mode(properties.parameters),
    });
  }

  const size_t best_index = select_display_mode(modes, desc.mode).value();
  const VkDisplayModePropertiesKHR& selected = mode_properties[best_index];

  if (!mode_matches_desired(modes[best_index].mode, desc.mode)) {
    HOLOVIZ_LOG_WARN(
        "[holoviz] Did not find a display mode with the desired properties {}x{} {:.3f} Hz, "
        "using closest match. Available display modes:",
        desc.mode.resolution.width,
        desc.mode.resolution.height,
        static_cast<float>(desc.mode.refresh_rate_mhz) / 1000.F);
    for (const auto& m : modes) {
      HOLOVIZ_LOG_WARN("[holoviz]  {}x{} {:.3f} Hz",
                       m.mode.resolution.width,
                       m.mode.resolution.height,
                       static_cast<float>(m.mode.refresh_rate_mhz) / 1000.F);
    }
  }
  HOLOVIZ_LOG_INFO("[holoviz] Using display mode {}x{} {:.3f} Hz",
                   selected.parameters.visibleRegion.width,
                   selected.parameters.visibleRegion.height,
                   static_cast<float>(selected.parameters.refreshRate) / 1000.F);

  // Pick the first compatible plane: not bound to another display, and the
  // target display in its supported set.
  uint32_t plane_count = 0;
  VIZ_VK_FORWARD_ERROR(
      vkGetPhysicalDeviceDisplayPlanePropertiesKHR(physical_device, &plane_count, nullptr));
  std::vector<VkDisplayPlanePropertiesKHR> plane_properties(plane_count);
  VIZ_VK_FORWARD_ERROR(vkGetPhysicalDeviceDisplayPlanePropertiesKHR(
      physical_device, &plane_count, plane_properties.data()));
  plane_properties.resize(plane_count);

  std::vector<DisplayPlaneInfo> plane_infos;
  plane_infos.reserve(plane_properties.size());
  for (uint32_t i = 0; i < plane_properties.size(); ++i) {
    uint32_t supported_count = 0;
    VIZ_VK_FORWARD_ERROR(
        vkGetDisplayPlaneSupportedDisplaysKHR(physical_device, i, &supported_count, nullptr));
    std::vector<VkDisplayKHR> supported_displays(supported_count);
    VIZ_VK_FORWARD_ERROR(vkGetDisplayPlaneSupportedDisplaysKHR(
        physical_device, i, &supported_count, supported_displays.data()));
    supported_displays.resize(supported_count);

    const bool supports_target =
        std::find(supported_displays.begin(), supported_displays.end(), display) !=
        supported_displays.end();
    plane_infos.push_back(DisplayPlaneInfo{
        .current_display = reinterpret_cast<uint64_t>(plane_properties[i].currentDisplay),
        .current_stack_index = plane_properties[i].currentStackIndex,
        .supports_target = supports_target,
    });
  }

  const std::optional<uint32_t> plane_index =
      select_display_plane(plane_infos, reinterpret_cast<uint64_t>(display));
  if (!plane_index.has_value()) {
    return make_unexpected(Error{.code = ErrorCode::kNotFound,
                                 .message = "[holoviz] Could not find a compatible display plane "
                                            "for the display."});
  }

  VkDisplayPlaneCapabilitiesKHR plane_capabilities{};
  VIZ_VK_FORWARD_ERROR(vkGetDisplayPlaneCapabilitiesKHR(
      physical_device, selected.displayMode, *plane_index, &plane_capabilities));
  const auto alpha_mode = static_cast<VkDisplayPlaneAlphaFlagBitsKHR>(
      select_alpha_mode(plane_capabilities.supportedAlpha));

  const VkDisplaySurfaceCreateInfoKHR surface_create_info{
      .sType = VK_STRUCTURE_TYPE_DISPLAY_SURFACE_CREATE_INFO_KHR,
      .displayMode = selected.displayMode,
      .planeIndex = *plane_index,
      .planeStackIndex = plane_properties[*plane_index].currentStackIndex,
      .transform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR,
      .globalAlpha = 1.0F,
      .alphaMode = alpha_mode,
      .imageExtent = selected.parameters.visibleRegion,
  };

  DisplaySurfaceCreation creation{};
  VIZ_VK_FORWARD_ERROR(vkCreateDisplayPlaneSurfaceKHR(
      instance.vk_instance_, &surface_create_info, instance.vk_allocator_, &creation.surface));
  creation.extent = {selected.parameters.visibleRegion.width,
                     selected.parameters.visibleRegion.height};
  return creation;
}

}  // namespace

expected<std::vector<Display>> enumerate_displays(const Adapter& adapter) {
  VIZ_UNWRAP(VulkanInstance, instance, GetVulkanInstance());
  if (!instance.display_supported_) {
    return make_unexpected(
        Error{.code = ErrorCode::kNotImplemented,
              .message = "[holoviz] Display enumeration is unavailable: the Vulkan instance does "
                         "not support " VK_KHR_DISPLAY_EXTENSION_NAME "."});
  }
  VIZ_UNWRAP(const PhysicalDeviceInfo*, device_info, find_physical_device_info(adapter.uuid));

  VIZ_UNWRAP(std::vector<VkDisplayPropertiesKHR>,
             display_properties,
             get_display_properties(device_info->physical_device_));

  std::vector<Display> displays;
  displays.reserve(display_properties.size());
  for (size_t i = 0; i < display_properties.size(); ++i) {
    const VkDisplayPropertiesKHR& properties = display_properties[i];
    Display display{};
    if (properties.displayName != nullptr && properties.displayName[0] != '\0') {
      std::snprintf(display.name.data(), display.name.size(), "%s", properties.displayName);
    } else {
      std::snprintf(display.name.data(), display.name.size(), "display %zu", i);
    }
    display.adapter = adapter;
    display.backend = Backend::kVulkan;
    display.display_id = reinterpret_cast<uint64_t>(properties.display);
    display.physical_size_mm = {properties.physicalDimensions.width,
                                properties.physicalDimensions.height};
    displays.push_back(display);
  }

  // Fold in the platform's view: ownership, better names, current modes, and
  // targets the plain Vulkan enumeration missed. Classification failure
  // downgrades to kUnknown, never to an enumeration error; platforms without a
  // direct-display path return an empty view and every display stays kUnknown.
  auto classified =
      display_systems::classify_displays(instance, device_info->physical_device_, *device_info);
  if (!classified) {
    HOLOVIZ_LOG_WARN("[holoviz] Display ownership classification unavailable: {}",
                     classified.error().message);
    return displays;
  }

  for (const display_systems::NativeDisplayInfo& native : classified.value()) {
    Display* matched = nullptr;
    if (native.vk_display != 0) {
      for (Display& display : displays) {
        if (display.display_id == native.vk_display) {
          matched = &display;
          break;
        }
      }
    }
    if (matched == nullptr) {
      // A display the platform resolved that the plain Vulkan enumeration missed;
      // with a handle it is fully usable, without one it is enumerate-only.
      Display display{};
      display.adapter = adapter;
      display.backend = Backend::kVulkan;
      display.display_id = native.vk_display;
      displays.push_back(display);
      matched = &displays.back();
    }
    matched->ownership = native.ownership;
    if (native.name[0] != '\0') {
      matched->name = native.name;
    }
    if (native.current_mode.resolution.width > 0) {
      matched->current_mode = native.current_mode;
    }
    if (native.vrr_capable) {
      matched->vrr_capable = true;
    }
    if (native.vrr_min_refresh_mhz > 0) {
      matched->vrr_min_refresh_mhz = native.vrr_min_refresh_mhz;
    }
  }

  return displays;
}

expected<std::vector<DisplayMode>> display_modes(const Display& display) {
  VIZ_UNWRAP(const PhysicalDeviceInfo*, device_info, resolve_display_device(display));

  VIZ_UNWRAP(std::vector<VkDisplayModePropertiesKHR>,
             mode_properties,
             get_display_mode_properties(device_info->physical_device_,
                                         reinterpret_cast<VkDisplayKHR>(display.display_id)));

  std::vector<DisplayMode> modes;
  modes.reserve(mode_properties.size());
  for (const auto& properties : mode_properties) {
    modes.push_back(to_display_mode(properties.parameters));
  }
  return modes;
}

expected<NativeObject> display_native_object(const Display& display, NativeObjectType type) {
  if (type != NativeObjectType::kVkDisplayKHR) {
    return make_unexpected(Error{.code = ErrorCode::kInvalidArgument,
                                 .message = "[holoviz] Display native object type not supported"});
  }
  return NativeObject{type, display.display_id};
}

expected<std::unique_ptr<detail::SurfaceImpl>> display_create_surface(
    const Display& display, const DisplaySurfaceDesc& desc) {
  // Ownership first: a compositor-owned display gets the remediation message
  // before any backend call.
  if (display.ownership == DisplayOwnership::kCompositor) {
    return make_unexpected(Error{
        .code = ErrorCode::kInvalidState,
        .message = std::string("[holoviz] Display \"") + display.name.data() +
                   "\" is owned by the compositor and cannot be acquired for direct scanout. "
                   "Stop the desktop session driving this GPU (no display server may hold DRM "
                   "master, e.g. run from a text console with the display manager stopped), "
                   "then re-enumerate and retry."});
  }

  VIZ_UNWRAP(VulkanInstance, instance, GetVulkanInstance());
  VIZ_UNWRAP(const PhysicalDeviceInfo*, device_info, resolve_display_device(display));
  VkPhysicalDevice physical_device = device_info->physical_device_;
  auto vk_display = reinterpret_cast<VkDisplayKHR>(display.display_id);

  // Guard against a stale handle from a previous enumeration (hotplug).
  VIZ_UNWRAP(std::vector<VkDisplayPropertiesKHR>,
             display_properties,
             get_display_properties(physical_device));
  const bool still_present = std::any_of(
      display_properties.begin(),
      display_properties.end(),
      [vk_display](const VkDisplayPropertiesKHR& p) { return p.display == vk_display; });
  if (!still_present) {
    return make_unexpected(
        Error{.code = ErrorCode::kNotFound,
              .message = "[holoviz] Display is no longer present; re-enumerate displays."});
  }

  VIZ_UNWRAP(display_systems::DisplayAcquisition,
             acquisition,
             display_systems::acquire_display(instance, physical_device, display.display_id));

  // `surface_created` widens the release beyond the platform acquire: once a VkSurfaceKHR
  // exists on the display the driver holds it implicitly (the "proceed unacquired" paths of
  // acquire_display), and that hold must be released explicitly too. The release-then-close
  // order and the close-regardless policy live in release_display_then_close (shared with
  // VulkanDisplaySurface::destroy so the two teardown paths cannot drift). A rejected release is
  // logged rather than returned: the creation failure being propagated is the primary error.
  auto release_acquisition = [&](bool surface_created) {
    if (!acquisition.acquired && !surface_created) {
      display_systems::close_acquisition_resources(acquisition);
      return;
    }
    const VkResult result = release_display_then_close(physical_device, vk_display, acquisition);
    if (result != VK_SUCCESS) {
      HOLOVIZ_LOG_ERROR(
          "[holoviz] vkReleaseDisplayEXT failed while unwinding surface "
          "creation; the display may stay held until process exit: {}",
          vk_error_string(result, __FILE__, __LINE__));
    }
  };

  auto creation = create_plane_surface(instance, physical_device, vk_display, desc);
  if (!creation) {
    release_acquisition(/*surface_created=*/false);
    return forward_error(creation);
  }

  auto surface = std::make_unique<VulkanDisplaySurface>();
  if (expected<void> inited = surface->init(instance.vk_instance_,
                                            instance.vk_allocator_,
                                            creation->surface,
                                            creation->extent,
                                            physical_device,
                                            vk_display,
                                            acquisition);
      !inited) {
    vkDestroySurfaceKHR(instance.vk_instance_, creation->surface, instance.vk_allocator_);
    release_acquisition(/*surface_created=*/true);
    return forward_error(inited);
  }

  return surface;
}

}  // namespace holoviz::vulkan
