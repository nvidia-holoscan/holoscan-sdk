/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda.h>
#include <array>
#include <memory>
#include <span>
#include <vector>

#include <holoviz/errors.hpp>
#include <holoviz/expected.hpp>

#include <holoviz/adapter.hpp>

#include "vk_impls.hpp"  // IWYU pragma: keep

namespace holoviz::vulkan {

/// HOLOVIZ_VALIDATION=1|true|on enables the Khronos validation layer (when the
/// loader finds it), the debug-utils messenger, and the module's extra debug checks
/// (e.g. the device live-child count); unset/0/false/off disables them all. Default is
/// off in every build type. Read once — the value is latched before the lazily created
/// instance exists (the first read happens inside the once-guarded instance init), so
/// changing the variable later in the process has no effect.
bool validation_enabled();

/// Everything queried from a physical device at instance creation: supported extensions,
/// the full properties chain, and the full features chain — each filled by exactly one
/// vkGetPhysicalDeviceProperties2 / vkGetPhysicalDeviceFeatures2 call in init(). Every
/// later consumer reads from this cache instead of re-querying the device.
struct PhysicalDeviceInfo {
  PhysicalDeviceInfo() = default;
  // The pNext chains point at sibling members, so records must never be copied or moved;
  // they are default-constructed in place and then chained by init().
  PhysicalDeviceInfo(const PhysicalDeviceInfo&) = delete;
  PhysicalDeviceInfo& operator=(const PhysicalDeviceInfo&) = delete;

  /// Enumerates the device's extensions, builds the properties/features pNext chains in
  /// place (extension structs are chained only when their extension is supported — an
  /// unsupported struct in the chain trips the validation layers), and performs the
  /// single Properties2/Features2 query pair.
  expected<void> init(VkPhysicalDevice physical_device);

  /// True when this physical device supports the device extension \p name.
  [[nodiscard]] bool extension_supported(const char* name) const;

  VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;

  /// Device extensions, enumerated once by init().
  std::vector<VkExtensionProperties> supported_extensions_;

  /// Chosen extension spelling, KHR preferred over EXT; nullptr when unsupported.
  const char* swapchain_maintenance1_ext_name_ = nullptr;
  const char* fifo_latest_ready_ext_name_ = nullptr;
  const char* calibrated_timestamps_ext_name_ = nullptr;

  /// Time domains vkGetCalibratedTimestamps* can sample together, queried once when
  /// calibrated_timestamps_ext_name_ is set (empty otherwise).
  std::vector<VkTimeDomainKHR> calibrateable_time_domains_;

  /// True when the device and the host monotonic clock (CLOCK_MONOTONIC on Linux, QPC on
  /// Windows) can be sampled together: the pair Device::calibrated_timestamps() returns.
  [[nodiscard]] bool calibrated_host_domain_supported() const {
    bool device_domain = false;
    bool host_domain = false;
    for (const VkTimeDomainKHR domain : calibrateable_time_domains_) {
      device_domain |= domain == VK_TIME_DOMAIN_DEVICE_KHR;
#ifdef _WIN32
      host_domain |= domain == VK_TIME_DOMAIN_QUERY_PERFORMANCE_COUNTER_KHR;
#else
      host_domain |= domain == VK_TIME_DOMAIN_CLOCK_MONOTONIC_KHR;
#endif
    }
    return device_domain && host_domain;
  }

  // Properties chain: Properties2 -> DriverProperties -> IDProperties (core in 1.2)
  // -> the conditional structs
  // below, each valid only when its has_* flag is set (zeroed otherwise).
  VkPhysicalDeviceProperties2 physical_device_properties_{};
  VkPhysicalDeviceIDProperties physical_device_id_properties_{};
  VkPhysicalDeviceDriverProperties driver_properties_{};
  bool has_external_compute_queue_properties_ = false;  ///< VK_NV_external_compute_queue
  VkPhysicalDeviceExternalComputeQueuePropertiesNV external_compute_queue_properties_{};
  bool has_drm_properties_ = false;  ///< VK_EXT_physical_device_drm
  VkPhysicalDeviceDrmPropertiesEXT drm_properties_{};
  bool has_pci_bus_info_properties_ = false;  ///< VK_EXT_pci_bus_info
  VkPhysicalDevicePCIBusInfoPropertiesEXT pci_bus_info_properties_{};

  // Features chain: Features2 -> TimelineSemaphore -> ShaderDrawParameters (core under
  // apiVersion 1.2, always chained) -> the conditional extension structs below. A struct
  // whose extension is unsupported stays zeroed, so its feature bits read VK_FALSE.
  VkPhysicalDeviceFeatures2 physical_device_features_{};
  VkPhysicalDeviceTimelineSemaphoreFeatures timeline_semaphore_features_{};
  VkPhysicalDeviceShaderDrawParametersFeatures shader_draw_parameters_features_{};
#if defined(VK_EXT_present_mode_fifo_latest_ready)
  VkPhysicalDevicePresentModeFifoLatestReadyFeaturesEXT fifo_latest_ready_features_{};
#endif
  VkPhysicalDeviceSwapchainMaintenance1FeaturesEXT swapchain_maintenance1_features_{};
#if defined(VK_KHR_present_id) && defined(VK_KHR_present_wait)
  VkPhysicalDevicePresentIdFeaturesKHR present_id_features_{};
  VkPhysicalDevicePresentWaitFeaturesKHR present_wait_features_{};
#endif
#if defined(VK_EXT_present_timing)
  VkPhysicalDevicePresentTimingFeaturesEXT present_timing_features_{};
#endif
#if defined(VK_KHR_present_id2)
  VkPhysicalDevicePresentId2FeaturesKHR
      present_id2_features_{};  ///< VK_EXT_present_timing dependency
#endif

  std::vector<VkQueueFamilyProperties2> queue_family_properties_;

  int cuda_device_index_;
  CUdevice cuda_device_;
  CUuuid cuda_device_uuid_;
};

struct VulkanInstance {
  VkInstance vk_instance_;
  VkAllocationCallbacks* vk_allocator_;
  VkDebugUtilsMessengerEXT vk_debug_messenger_;
  std::span<PhysicalDeviceInfo> physical_device_infos_;

  /// The window system was unavailable at instance creation (headless console);
  /// window creation fails, only direct-display surfaces can present.
  bool headless_ = false;
  /// Optional direct-display instance extensions, enabled when the driver has them.
  bool display_supported_ = false;              ///< VK_KHR_display
  bool direct_mode_display_supported_ = false;  ///< VK_EXT_direct_mode_display
  bool acquire_drm_supported_ = false;          ///< VK_EXT_acquire_drm_display (Linux)
  /// VK_EXT_display_surface_counter: the instance dependency of VK_EXT_display_control
  /// (display events, swapchain vblank counters).
  bool display_surface_counter_supported_ = false;
  /// specVersion of VK_LAYER_KHRONOS_validation when it was enabled (HOLOVIZ_VALIDATION);
  /// 0 otherwise. A layer older than an extension the module enables logs unknown-structure
  /// errors and can crash, so device creation gates such extensions on this.
  uint32_t validation_layer_spec_version_ = 0;
  /// The surface_maintenance1 instance extension spelling that was enabled (KHR or EXT).
  /// Device creation must enable the matching swapchain_maintenance1 spelling: the KHR
  /// device extension formally depends on the KHR instance extension, and EXT on EXT.
  const char* surface_maintenance1_ext_name_ = nullptr;
};

[[nodiscard]] expected<VulkanInstance> GetVulkanInstance();

/// Resolve an adapter UUID to the instance's physical-device record.
[[nodiscard]] expected<const PhysicalDeviceInfo*> find_physical_device_info(
    const std::array<uint8_t, 16>& uuid);

/// Resolve a VkPhysicalDevice handle to the instance's physical-device record.
[[nodiscard]] expected<const PhysicalDeviceInfo*> find_physical_device_info(
    VkPhysicalDevice physical_device);

expected<std::vector<Adapter>> enumerate_adapters();

/// Return the first queue family with graphics support that, when \p surface is not
/// VK_NULL_HANDLE, can also present to it. HoloViz creates a single queue used for both
/// rendering and presentation, so surface support anywhere else on the device does not help.
[[nodiscard]] expected<uint32_t> find_graphics_queue_family(VkPhysicalDevice physical_device,
                                                            VkSurfaceKHR surface);

expected<bool> adapter_supports_surface(const Adapter& adapter, const Surface& surface);
expected<NativeObject> adapter_native_object(const Adapter& adapter, NativeObjectType type);

/// When \p surface is non-null the device's queue family is chosen to both render and
/// present to it; creation fails if no such family exists.
expected<std::unique_ptr<detail::DeviceImpl>> adapter_create_device(
    const Adapter& adapter, const DeviceDesc& desc, const Surface* surface = nullptr);

}  // namespace holoviz::vulkan
