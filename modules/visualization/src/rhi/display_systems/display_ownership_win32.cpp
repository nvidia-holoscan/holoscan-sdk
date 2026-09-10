/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Windows display ownership + acquisition through WinRT
// (Windows.Devices.Display.Core) and VK_NV_acquire_winrt_display, compiled on
// Windows only. Every join is an exact key — no EDID or name string matching:
// the adapter joins by LUID (VkPhysicalDeviceIDProperties.deviceLUID ==
// DisplayAdapter.Id), and each DisplayTarget joins to a VkDisplayKHR through
// vkGetWinrtDisplayNV(target.AdapterRelativeId()), the same key the
// acquisition uses. Only targets configured as "Specialized display"
// (Settings > System > Display) are acquirable; desktop targets classify as
// kCompositor. All WinRT usage stays inside this TU (exceptions are caught at
// the boundary and converted to expected<> errors).

#include "display_ownership.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <winrt/Windows.Devices.Display.Core.h>
#include <winrt/Windows.Devices.Display.h>
#include <winrt/Windows.Foundation.Collections.h>
#include <winrt/Windows.Graphics.h>
#include <winrt/base.h>

#include <holoviz/logger.hpp>

#include "rhi/vulkan/vk_display_util.hpp"
#include "rhi/vulkan/vk_errors.hpp"

namespace holoviz::rhi::display_systems {

namespace {

namespace wdd = winrt::Windows::Devices::Display;
namespace wddc = winrt::Windows::Devices::Display::Core;

/// Idempotent COM init for this process: multithreaded preferred, an apartment
/// someone else already initialized is fine as-is.
void ensure_winrt_apartment() {
  try {
    winrt::init_apartment(winrt::apartment_type::multi_threaded);
  } catch (const winrt::hresult_error&) {
    // RPC_E_CHANGED_MODE: the thread already has an apartment; use it.
  }
}

bool adapter_luid_matches(const vulkan::PhysicalDeviceInfo& device_info,
                          const wddc::DisplayAdapter& adapter) {
  if (device_info.physical_device_id_properties_.deviceLUIDValid == VK_FALSE) {
    return false;
  }
  const winrt::Windows::Graphics::DisplayAdapterId adapter_id = adapter.Id();
  uint8_t luid_bytes[VK_LUID_SIZE]{};
  std::memcpy(luid_bytes, &adapter_id.LowPart, sizeof(adapter_id.LowPart));
  std::memcpy(
      luid_bytes + sizeof(adapter_id.LowPart), &adapter_id.HighPart, sizeof(adapter_id.HighPart));
  return std::memcmp(
             luid_bytes, device_info.physical_device_id_properties_.deviceLUID, VK_LUID_SIZE) == 0;
}

expected<bool> physical_device_has_acquire_winrt(VkPhysicalDevice physical_device) {
  uint32_t extension_count = 0;
  VIZ_VK_FORWARD_ERROR(
      vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &extension_count, nullptr));
  std::vector<VkExtensionProperties> extensions(extension_count);
  VIZ_VK_FORWARD_ERROR(vkEnumerateDeviceExtensionProperties(
      physical_device, nullptr, &extension_count, extensions.data()));
  return std::any_of(extensions.begin(), extensions.end(), [](const VkExtensionProperties& ext) {
    return std::strcmp(ext.extensionName, VK_NV_ACQUIRE_WINRT_DISPLAY_EXTENSION_NAME) == 0;
  });
}

}  // namespace

expected<std::vector<NativeDisplayInfo>> classify_displays(
    const vulkan::VulkanInstance& instance, VkPhysicalDevice physical_device,
    const vulkan::PhysicalDeviceInfo& device_info) {
  if (!instance.display_supported_ || vkGetWinrtDisplayNV == nullptr) {
    return make_unexpected(Error{.code = ErrorCode::kNotImplemented,
                                 .message =
                                     "[holoviz] " VK_NV_ACQUIRE_WINRT_DISPLAY_EXTENSION_NAME
                                     " is unavailable; Windows display classification disabled."});
  }
  if (device_info.physical_device_id_properties_.deviceLUIDValid == VK_FALSE) {
    return make_unexpected(Error{.code = ErrorCode::kNotImplemented,
                                 .message =
                                     "[holoviz] The adapter reports no valid LUID; Windows display "
                                     "classification disabled."});
  }

  ensure_winrt_apartment();

  try {
    std::vector<NativeDisplayInfo> displays;

    wddc::DisplayManager manager = wddc::DisplayManager::Create(wddc::DisplayManagerOptions::None);

    // Snapshot of the currently configured paths: what each target actually
    // scans out (resolution + presentation rate). This is the mode that also
    // governs a direct-display swapchain — the driver presents into the
    // configured path rather than mode-setting to the requested VkDisplayModeKHR.
    wddc::DisplayState display_state = nullptr;
    if (const auto state_result = manager.TryReadCurrentStateForAllTargets();
        state_result.ErrorCode() == wddc::DisplayManagerResult::Success) {
      display_state = state_result.State();
    }

    for (const wddc::DisplayTarget& target : manager.GetCurrentTargets()) {
      if (!target.IsConnected()) {
        continue;
      }
      const wddc::DisplayAdapter target_adapter = target.Adapter();
      if (target_adapter == nullptr || !adapter_luid_matches(device_info, target_adapter)) {
        continue;
      }

      // The Vulkan join. The NVIDIA driver only resolves targets that are
      // acquirable (specialized displays); a desktop target that fails to
      // resolve is still reported — handle-less — so enumeration lists every
      // monitor with its ownership, which is the whole point of the API.
      VkDisplayKHR vk_display = VK_NULL_HANDLE;
      if (vkGetWinrtDisplayNV(physical_device,
                              static_cast<uint32_t>(target.AdapterRelativeId()),
                              &vk_display) != VK_SUCCESS) {
        vk_display = VK_NULL_HANDLE;
      }

      NativeDisplayInfo info{};
      info.vk_display = reinterpret_cast<uint64_t>(vk_display);
      // SpecialPurpose == the "Specialized display" toggle in Windows Settings.
      info.ownership = vulkan::classify_winrt_target(target.UsageKind() ==
                                                     wdd::DisplayMonitorUsageKind::SpecialPurpose);

      if (const wdd::DisplayMonitor monitor = target.TryGetMonitor(); monitor != nullptr) {
        const std::string display_name = winrt::to_string(monitor.DisplayName());
        if (!display_name.empty()) {
          std::snprintf(info.name.data(), info.name.size(), "%s", display_name.c_str());
        }
      }

      if (display_state != nullptr) {
        if (const wddc::DisplayPath path = display_state.GetPathForTarget(target);
            path != nullptr) {
          const auto resolution = path.TargetResolution();
          const auto rate = path.PresentationRate();
          if (resolution != nullptr && rate != nullptr &&
              rate.Value().VerticalSyncRate.Denominator != 0) {
            info.current_mode = DisplayMode{
                .resolution = {static_cast<uint32_t>(resolution.Value().Width),
                               static_cast<uint32_t>(resolution.Value().Height)},
                .refresh_rate_mhz = static_cast<uint32_t>(
                    static_cast<uint64_t>(rate.Value().VerticalSyncRate.Numerator) * 1000ULL /
                    rate.Value().VerticalSyncRate.Denominator),
            };
          }
        }
      }

      displays.push_back(info);
    }
    manager.Close();

    return displays;
  } catch (const winrt::hresult_error& error) {
    return make_unexpected(
        Error{.code = ErrorCode::kFailure,
              .message = std::string("[holoviz] WinRT display enumeration failed: ") +
                         winrt::to_string(error.message())});
  }
}

expected<DisplayAcquisition> acquire_display(const vulkan::VulkanInstance& instance,
                                             VkPhysicalDevice physical_device,
                                             uint64_t vk_display) {
  (void)instance;
  VIZ_UNWRAP(bool, has_acquire, physical_device_has_acquire_winrt(physical_device));
  if (!has_acquire || vkAcquireWinrtDisplayNV == nullptr) {
    HOLOVIZ_LOG_WARN(
        "[holoviz] " VK_NV_ACQUIRE_WINRT_DISPLAY_EXTENSION_NAME
        " is unavailable; continuing without acquiring the display (the driver decides at "
        "surface creation whether the display is free).");
    return DisplayAcquisition{};
  }

  const VkResult result =
      vkAcquireWinrtDisplayNV(physical_device, reinterpret_cast<VkDisplayKHR>(vk_display));
  if (result != VK_SUCCESS) {
    return make_unexpected(Error{
        .code = ErrorCode::kVulkanError,
        .message = std::string("[holoviz] vkAcquireWinrtDisplayNV failed (") +
                   vulkan::vk_error_string(result, __FILE__, __LINE__) +
                   "). Configure the monitor as a specialized display first: Windows Settings > "
                   "System > Display > select the monitor > Specialized display."});
  }

  return DisplayAcquisition{.acquired = true};
}

void close_acquisition_resources(DisplayAcquisition& /*acquisition*/) {
  // A WinRT acquisition holds no OS resources beyond the Vulkan-side acquire.
}

}  // namespace holoviz::rhi::display_systems
