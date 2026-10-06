/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdio>
#include <cstring>

#include "vk_adapter.hpp"
#include "vk_device.hpp"
#include "vk_errors.hpp"
#include "vk_extension_util.hpp"
#include "vulkan/vk_impls.hpp"

#include "../window_systems/glfw_window.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <holoviz/logger.hpp>

#include "../cuda/cu_device.hpp"

namespace holoviz::vulkan {

static VulkanInstance g_vulkan_instance;

bool PhysicalDeviceInfo::extension_supported(const char* name) const {
  return std::any_of(supported_extensions_.begin(),
                     supported_extensions_.end(),
                     [name](const VkExtensionProperties& ext) {
                       return std::strcmp(ext.extensionName, name) == 0;
                     });
}

expected<void> PhysicalDeviceInfo::init(VkPhysicalDevice physical_device) {
  physical_device_ = physical_device;

  uint32_t supported_extension_count = 0;
  VIZ_VK_FORWARD_ERROR(vkEnumerateDeviceExtensionProperties(
      physical_device, nullptr, &supported_extension_count, nullptr));
  supported_extensions_.resize(supported_extension_count);
  VIZ_VK_FORWARD_ERROR(vkEnumerateDeviceExtensionProperties(
      physical_device, nullptr, &supported_extension_count, supported_extensions_.data()));

  // Extensions promoted from EXT to KHR share their structs and commands, but each device
  // extension formally depends on the matching instance spelling
  // (VUID-vkCreateDevice-ppEnabledExtensionNames-01387), so pick the swapchain maintenance
  // spelling that matches the enabled surface maintenance instance extension.
#if defined(VK_EXT_swapchain_maintenance1)
  swapchain_maintenance1_ext_name_ = select_swapchain_maintenance1_extension(
      g_vulkan_instance.surface_maintenance1_ext_name_ != nullptr
          ? g_vulkan_instance.surface_maintenance1_ext_name_
          : "",
      [this](const char* name) { return extension_supported(name); });
#endif
#if defined(VK_KHR_present_mode_fifo_latest_ready)
  if (extension_supported(VK_KHR_PRESENT_MODE_FIFO_LATEST_READY_EXTENSION_NAME)) {
    fifo_latest_ready_ext_name_ = VK_KHR_PRESENT_MODE_FIFO_LATEST_READY_EXTENSION_NAME;
  }
#endif
#if defined(VK_EXT_present_mode_fifo_latest_ready)
  if (fifo_latest_ready_ext_name_ == nullptr &&
      extension_supported(VK_EXT_PRESENT_MODE_FIFO_LATEST_READY_EXTENSION_NAME)) {
    fifo_latest_ready_ext_name_ = VK_EXT_PRESENT_MODE_FIFO_LATEST_READY_EXTENSION_NAME;
  }
#endif
#if defined(VK_KHR_calibrated_timestamps)
  if (extension_supported(VK_KHR_CALIBRATED_TIMESTAMPS_EXTENSION_NAME)) {
    calibrated_timestamps_ext_name_ = VK_KHR_CALIBRATED_TIMESTAMPS_EXTENSION_NAME;
  }
#endif
#if defined(VK_EXT_calibrated_timestamps)
  if (calibrated_timestamps_ext_name_ == nullptr &&
      extension_supported(VK_EXT_CALIBRATED_TIMESTAMPS_EXTENSION_NAME)) {
    calibrated_timestamps_ext_name_ = VK_EXT_CALIBRATED_TIMESTAMPS_EXTENSION_NAME;
  }
#endif
  if (calibrated_timestamps_ext_name_ != nullptr) {
    // The KHR and EXT entry points are aliases; volk resolves whichever the instance offers.
    const auto calibrateable_domains = vkGetPhysicalDeviceCalibrateableTimeDomainsKHR != nullptr
                                           ? vkGetPhysicalDeviceCalibrateableTimeDomainsKHR
                                           : vkGetPhysicalDeviceCalibrateableTimeDomainsEXT;
    uint32_t domain_count = 0;
    if (calibrateable_domains != nullptr &&
        calibrateable_domains(physical_device, &domain_count, nullptr) == VK_SUCCESS) {
      calibrateable_time_domains_.resize(domain_count);
      if (calibrateable_domains(
              physical_device, &domain_count, calibrateable_time_domains_.data()) != VK_SUCCESS) {
        calibrateable_time_domains_.clear();
      }
    }
  }

  uint32_t queue_family_count = 0;
  vkGetPhysicalDeviceQueueFamilyProperties2(physical_device, &queue_family_count, nullptr);
  queue_family_properties_.resize(queue_family_count,
                                  {.sType = VK_STRUCTURE_TYPE_QUEUE_FAMILY_PROPERTIES_2});
  vkGetPhysicalDeviceQueueFamilyProperties2(
      physical_device, &queue_family_count, queue_family_properties_.data());

  // Properties chain. Extension structs join the chain only when their extension is
  // supported: an unsupported struct in the chain trips the validation layers, and the
  // zeroed struct is the correct "absent" value for consumers.
  physical_device_id_properties_ = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES,
  };
  physical_device_properties_ = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
      .pNext = &physical_device_id_properties_,
  };
  const auto chain_properties = [this](void* struct_base, VkStructureType sType) {
    auto* header = static_cast<VkBaseOutStructure*>(struct_base);
    header->sType = sType;
    header->pNext = static_cast<VkBaseOutStructure*>(physical_device_properties_.pNext);
    physical_device_properties_.pNext = header;
  };
  if (extension_supported(VK_NV_EXTERNAL_COMPUTE_QUEUE_EXTENSION_NAME)) {
    has_external_compute_queue_properties_ = true;
    chain_properties(&external_compute_queue_properties_,
                     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_COMPUTE_QUEUE_PROPERTIES_NV);
  }
  if (extension_supported(VK_EXT_PHYSICAL_DEVICE_DRM_EXTENSION_NAME)) {
    has_drm_properties_ = true;
    chain_properties(&drm_properties_, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRM_PROPERTIES_EXT);
  }
  if (extension_supported(VK_EXT_PCI_BUS_INFO_EXTENSION_NAME)) {
    has_pci_bus_info_properties_ = true;
    chain_properties(&pci_bus_info_properties_,
                     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PCI_BUS_INFO_PROPERTIES_EXT);
  }
  vkGetPhysicalDeviceProperties2(physical_device, &physical_device_properties_);

  // Features chain, same conditional-chaining rule. Timeline semaphores and shader draw
  // parameters are core under apiVersion 1.2 and always chained.
  timeline_semaphore_features_ = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES,
  };
  shader_draw_parameters_features_ = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_DRAW_PARAMETERS_FEATURES,
      .pNext = &timeline_semaphore_features_,
  };
  physical_device_features_ = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
      .pNext = &shader_draw_parameters_features_,
  };
  const auto chain_features = [this](void* struct_base, VkStructureType sType) {
    auto* header = static_cast<VkBaseOutStructure*>(struct_base);
    header->sType = sType;
    header->pNext = static_cast<VkBaseOutStructure*>(physical_device_features_.pNext);
    physical_device_features_.pNext = header;
  };
#if defined(VK_EXT_present_mode_fifo_latest_ready)
  if (fifo_latest_ready_ext_name_ != nullptr) {
    chain_features(&fifo_latest_ready_features_,
                   VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_MODE_FIFO_LATEST_READY_FEATURES_EXT);
  }
#endif
  if (swapchain_maintenance1_ext_name_ != nullptr) {
    chain_features(&swapchain_maintenance1_features_,
                   VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SWAPCHAIN_MAINTENANCE_1_FEATURES_EXT);
  }
#if defined(VK_KHR_present_id) && defined(VK_KHR_present_wait)
  if (extension_supported(VK_KHR_PRESENT_ID_EXTENSION_NAME)) {
    chain_features(&present_id_features_,
                   VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_ID_FEATURES_KHR);
  }
  if (extension_supported(VK_KHR_PRESENT_WAIT_EXTENSION_NAME)) {
    chain_features(&present_wait_features_,
                   VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_WAIT_FEATURES_KHR);
  }
#endif
#if defined(VK_EXT_present_timing)
  if (extension_supported(VK_EXT_PRESENT_TIMING_EXTENSION_NAME)) {
    chain_features(&present_timing_features_,
                   VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_TIMING_FEATURES_EXT);
  }
#endif
#if defined(VK_KHR_present_id2)
  if (extension_supported(VK_KHR_PRESENT_ID_2_EXTENSION_NAME)) {
    chain_features(&present_id2_features_,
                   VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_ID_2_FEATURES_KHR);
  }
#endif
  vkGetPhysicalDeviceFeatures2(physical_device, &physical_device_features_);

  return {};
}

// Declared in vk_adapter.hpp; latched-static behavior documented there.
bool validation_enabled() {
  static const bool enabled = [] {
    const char* env = std::getenv("HOLOVIZ_VALIDATION");
    if (env == nullptr || env[0] == '\0') {
      return false;
    }
    const auto iequals = [env](const char* other) {
      const char* a = env;
      for (; *a != '\0' && *other != '\0'; ++a, ++other) {
        if (std::tolower(static_cast<unsigned char>(*a)) !=
            std::tolower(static_cast<unsigned char>(*other))) {
          return false;
        }
      }
      return *a == '\0' && *other == '\0';
    };
    return !(std::strcmp(env, "0") == 0 || iequals("false") || iequals("off"));
  }();
  return enabled;
}

static VkBool32 VKAPI_PTR vulkan_custom_debug_callback(
    VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
    VkDebugUtilsMessageTypeFlagsEXT messageTypes,
    const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData) {
  LogLevel level = LogLevel::kDebug;
  if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
    level = LogLevel::kError;
  } else if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
    level = LogLevel::kWarn;
  } else if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT) {
    level = LogLevel::kInfo;
  }
  log_literal(level, pCallbackData->pMessage != nullptr ? pCallbackData->pMessage : "");
  return VK_FALSE;
}

/// Builds g_vulkan_instance. Runs exactly once, inside GetVulkanInstance()'s
/// call_once; must not be called from anywhere else. A midway failure (e.g.
/// after vkCreateInstance) leaves the half-built instance in place without
/// tearing it down — pre-existing behavior; the stored error keeps every
/// later caller from touching it.
static expected<void> init_vulkan_instance() {
  VIZ_VK_FORWARD_ERROR(volkInitialize());

  // The window system is optional: on a headless console (the direct-display
  // deployment) GLFW cannot initialize, but adapter/display enumeration and
  // display surfaces only need VK_KHR_surface.
  std::vector<const char*> extensions;
  expected<void> glfw_inited = glfw::faux_window_create();
  if (glfw_inited) {
    uint32_t glfw_extension_count = 0;
    const char** glfw_extensions = glfwGetRequiredInstanceExtensions(&glfw_extension_count);
    if (glfw_extensions != nullptr && glfw_extension_count > 0) {
      extensions.assign(glfw_extensions, glfw_extensions + glfw_extension_count);
    } else {
      // GLFW came up but found no Vulkan-capable window system; drop the reference.
      glfw::faux_window_destroy();
      glfw_inited = make_unexpected(
          Error{ErrorCode::kFailure, "GLFW reports no Vulkan-capable window system"});
    }
  }
  if (extensions.empty()) {
    // kNotFound is the expected headless case (no route to any window system in the
    // environment); anything else means a route was advertised (e.g. XDG_RUNTIME_DIR is set but
    // no compositor listens) and could not be used.
    if (glfw_inited.error().code == ErrorCode::kNotFound) {
      HOLOVIZ_LOG_INFO(
          "[holoviz] No window system route (DISPLAY, WAYLAND_DISPLAY, WAYLAND_SOCKET and "
          "XDG_RUNTIME_DIR unset); windows are unavailable, direct-display surfaces remain "
          "available.");
    } else {
      HOLOVIZ_LOG_WARN(
          "[holoviz] Window system unavailable ({}); continuing headless (window creation will "
          "fail, direct-display surfaces remain available).",
          glfw_inited.error().message);
    }
    g_vulkan_instance.headless_ = true;
    extensions.push_back(VK_KHR_SURFACE_EXTENSION_NAME);
  }

  std::vector<const char*> layers;  // Add validation layers if needed.

  extensions.push_back(VK_KHR_GET_SURFACE_CAPABILITIES_2_EXTENSION_NAME);
  extensions.push_back(VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME);
  extensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME);

  if (validation_enabled()) {
    HOLOVIZ_LOG_INFO("[holoviz] Vulkan validation enabled via HOLOVIZ_VALIDATION");
    extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

    // Validation layers are a debugging aid, not a requirement: enable them when the
    // loader can find them, otherwise continue without and just warn.
    uint32_t supported_layer_count = 0;
    VIZ_VK_FORWARD_ERROR(vkEnumerateInstanceLayerProperties(&supported_layer_count, nullptr));
    std::vector<VkLayerProperties> supported_layers(supported_layer_count);
    VIZ_VK_FORWARD_ERROR(
        vkEnumerateInstanceLayerProperties(&supported_layer_count, supported_layers.data()));

    const char* validation_layer_name = "VK_LAYER_KHRONOS_validation";
    const auto validation_layer =
        std::find_if(supported_layers.begin(),
                     supported_layers.end(),
                     [validation_layer_name](const VkLayerProperties& layer) {
                       return std::strcmp(layer.layerName, validation_layer_name) == 0;
                     });
    if (validation_layer != supported_layers.end()) {
      layers.push_back(validation_layer_name);
      g_vulkan_instance.validation_layer_spec_version_ = validation_layer->specVersion;
    } else {
      HOLOVIZ_LOG_WARN(
          "[holoviz] Vulkan validation layer VK_LAYER_KHRONOS_validation not found; continuing "
          "without validation.");
    }
  }

  uint32_t supported_extension_count = 0;
  VIZ_VK_FORWARD_ERROR(
      vkEnumerateInstanceExtensionProperties(nullptr, &supported_extension_count, nullptr));
  std::vector<VkExtensionProperties> supported_extensions(supported_extension_count);
  VIZ_VK_FORWARD_ERROR(vkEnumerateInstanceExtensionProperties(
      nullptr, &supported_extension_count, supported_extensions.data()));

  // Enabled layers may provide instance extensions the loader alone does not report. A layer
  // that fails to enumerate here is simply skipped; a missing layer surfaces at vkCreateInstance.
  for (const char* layer : layers) {
    uint32_t layer_extension_count = 0;
    if (vkEnumerateInstanceExtensionProperties(layer, &layer_extension_count, nullptr) !=
        VK_SUCCESS) {
      continue;
    }
    std::vector<VkExtensionProperties> layer_extensions(layer_extension_count);
    if (vkEnumerateInstanceExtensionProperties(
            layer, &layer_extension_count, layer_extensions.data()) != VK_SUCCESS) {
      continue;
    }
    supported_extensions.insert(
        supported_extensions.end(), layer_extensions.begin(), layer_extensions.end());
  }

  const auto instance_extension_supported = [&supported_extensions](const char* extension) {
    return std::any_of(supported_extensions.begin(),
                       supported_extensions.end(),
                       [extension](const VkExtensionProperties& ext) {
                         return std::strcmp(ext.extensionName, extension) == 0;
                       });
  };

  // Required: surface maintenance (per-present-mode capabilities, scaling capabilities).
  // The device side requires VK_EXT_swapchain_maintenance1, which depends on this instance
  // extension. Prefer the promoted KHR extension, fall back to EXT.
  const char* surface_maintenance1_ext_name = nullptr;
#if defined(VK_KHR_surface_maintenance1)
  if (instance_extension_supported(VK_KHR_SURFACE_MAINTENANCE_1_EXTENSION_NAME)) {
    surface_maintenance1_ext_name = VK_KHR_SURFACE_MAINTENANCE_1_EXTENSION_NAME;
  }
#endif
#if defined(VK_EXT_surface_maintenance1)
  if (surface_maintenance1_ext_name == nullptr &&
      instance_extension_supported(VK_EXT_SURFACE_MAINTENANCE_1_EXTENSION_NAME)) {
    surface_maintenance1_ext_name = VK_EXT_SURFACE_MAINTENANCE_1_EXTENSION_NAME;
  }
#endif
  g_vulkan_instance.surface_maintenance1_ext_name_ = surface_maintenance1_ext_name;
  if (surface_maintenance1_ext_name == nullptr) {
    return make_unexpected(
        Error{.code = ErrorCode::kFailure,
              .message = "[holoviz] Vulkan instance does not support the required extension "
                         "VK_EXT_surface_maintenance1 (or its KHR promotion)."});
  }
  extensions.push_back(surface_maintenance1_ext_name);
  g_vulkan_instance.surface_maintenance1_ext_name_ = surface_maintenance1_ext_name;

  // Optional: direct-to-display. VK_KHR_display drives display enumeration
  // (enumerate_displays); VK_EXT_direct_mode_display plus the Linux acquire
  // extension VK_EXT_acquire_drm_display lets a display no compositor owns be
  // latched onto directly. Exclusive display is Linux-only; on other platforms the
  // display extensions are enabled for enumeration but CapabilityReport::
  // exclusive_display stays false. Absence only disables that path — reported
  // through CapabilityReport::exclusive_display and per-display DisplayOwnership.
  if (instance_extension_supported(VK_KHR_DISPLAY_EXTENSION_NAME)) {
    extensions.push_back(VK_KHR_DISPLAY_EXTENSION_NAME);
    g_vulkan_instance.display_supported_ = true;

    // Optional: surface counters (vblank), the instance dependency of the
    // VK_EXT_display_control device extension (display events, swapchain counters).
    if (instance_extension_supported(VK_EXT_DISPLAY_SURFACE_COUNTER_EXTENSION_NAME)) {
      extensions.push_back(VK_EXT_DISPLAY_SURFACE_COUNTER_EXTENSION_NAME);
      g_vulkan_instance.display_surface_counter_supported_ = true;
    }

    if (instance_extension_supported(VK_EXT_DIRECT_MODE_DISPLAY_EXTENSION_NAME)) {
      extensions.push_back(VK_EXT_DIRECT_MODE_DISPLAY_EXTENSION_NAME);
      g_vulkan_instance.direct_mode_display_supported_ = true;

#ifdef __linux__
      if (instance_extension_supported(VK_EXT_ACQUIRE_DRM_DISPLAY_EXTENSION_NAME)) {
        extensions.push_back(VK_EXT_ACQUIRE_DRM_DISPLAY_EXTENSION_NAME);
        g_vulkan_instance.acquire_drm_supported_ = true;
      }
#endif
    }
  }

  for (const char* extension : extensions) {
    if (!instance_extension_supported(extension)) {
      return make_unexpected(
          Error{.code = ErrorCode::kFailure,
                .message = std::string("[holoviz] Vulkan instance does not support the required "
                                       "extension: ") +
                           extension});
    }
  }

  VkApplicationInfo app_info{
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = "holoviz",
      .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
      .pEngineName = "holoviz",
      .engineVersion = VK_MAKE_VERSION(1, 0, 0),
      .apiVersion = VK_API_VERSION_1_2,
  };

  VkInstanceCreateInfo instance_create_info{
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .pApplicationInfo = &app_info,
      .enabledLayerCount = static_cast<uint32_t>(layers.size()),
      .ppEnabledLayerNames = layers.data(),
      .enabledExtensionCount = static_cast<uint32_t>(extensions.size()),
      .ppEnabledExtensionNames = extensions.data()};

  VIZ_VK_FORWARD_ERROR(vkCreateInstance(
      &instance_create_info, g_vulkan_instance.vk_allocator_, &g_vulkan_instance.vk_instance_));

  volkLoadInstance(g_vulkan_instance.vk_instance_);

  if (validation_enabled()) {
    VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo = {};
    debugCreateInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    debugCreateInfo.pNext = NULL;
    debugCreateInfo.flags = 0;
    debugCreateInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
                                      VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT |
                                      VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                                      VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    debugCreateInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                                  VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                                  VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    debugCreateInfo.pfnUserCallback = vulkan_custom_debug_callback;
    debugCreateInfo.pUserData = NULL;

    VIZ_VK_FORWARD_ERROR(vkCreateDebugUtilsMessengerEXT(g_vulkan_instance.vk_instance_,
                                                        &debugCreateInfo,
                                                        g_vulkan_instance.vk_allocator_,
                                                        &g_vulkan_instance.vk_debug_messenger_));
  }

  uint32_t physical_device_count = 0;
  VIZ_VK_FORWARD_ERROR(
      vkEnumeratePhysicalDevices(g_vulkan_instance.vk_instance_, &physical_device_count, nullptr));

  std::vector<VkPhysicalDevice> physical_devices(physical_device_count);
  VIZ_VK_FORWARD_ERROR(vkEnumeratePhysicalDevices(
      g_vulkan_instance.vk_instance_, &physical_device_count, physical_devices.data()));

  g_vulkan_instance.physical_device_infos_ = std::span<PhysicalDeviceInfo>(
      new PhysicalDeviceInfo[physical_device_count], physical_device_count);

  // Displays are deliberately not cached here: hotplug and desktop-topology
  // changes would invalidate them. enumerate_displays() queries on demand.
  for (int i = 0; i < physical_device_count; ++i) {
    VIZ_FORWARD_ERROR(g_vulkan_instance.physical_device_infos_[i].init(physical_devices[i]));
  }

  return {};
}

expected<VulkanInstance> GetVulkanInstance() {
  // Thread-safe one-time initialization with a sticky result: instance creation
  // failures don't self-heal, so every later caller re-sees the stored error.
  // The initializer cannot throw (no-exceptions codebase), so call_once always
  // completes on the first call.
  static std::once_flag once;
  static expected<void> status;  // written only inside call_once
  std::call_once(once, [] { status = init_vulkan_instance(); });
  if (!status) {
    return make_unexpected(status.error());
  }
  return g_vulkan_instance;
}

expected<std::vector<Adapter>> enumerate_adapters() {
  VIZ_UNWRAP(VulkanInstance, instance, GetVulkanInstance());

  std::vector<Adapter> adapters;
  for (const auto& device_info : instance.physical_device_infos_) {
    Adapter adapter;
    adapter.name = std::array<char, 256>{};
    std::snprintf(adapter.name.data(),
                  adapter.name.size(),
                  "%s",
                  device_info.physical_device_properties_.properties.deviceName);
    adapter.uuid = std::array<uint8_t, 16>{};
    std::copy(std::begin(device_info.physical_device_id_properties_.deviceUUID),
              std::end(device_info.physical_device_id_properties_.deviceUUID),
              std::begin(adapter.uuid));

    auto cuda_ordinal_result = cuda::get_cuda_ordinal(adapter.uuid);
    if (!cuda_ordinal_result) {
      HOLOVIZ_LOG_WARN("[holoviz] No matching CUDA device found for Vulkan adapter: {}",
                       adapter.name.data());
      continue;  // Skip this adapter if we can't get a CUDA ordinal
    }
    adapter.cuda_device_index = cuda_ordinal_result.value();
    adapter.backend = Backend::kVulkan;
    adapters.push_back(adapter);
  }

  if (adapters.empty()) {
    return make_unexpected(Error{.code = ErrorCode::kNotFound,
                                 .message = "[holoviz] No suitable Vulkan adapters found."});
  }

  return adapters;
}

expected<uint32_t> find_graphics_queue_family(VkPhysicalDevice physical_device,
                                              VkSurfaceKHR surface) {
  uint32_t queue_family_count = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count, nullptr);
  std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
  vkGetPhysicalDeviceQueueFamilyProperties(
      physical_device, &queue_family_count, queue_families.data());

  for (uint32_t i = 0; i < queue_family_count; ++i) {
    if ((queue_families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) == 0) {
      continue;
    }
    if (surface != VK_NULL_HANDLE) {
      VkBool32 present_supported = VK_FALSE;
      VIZ_VK_FORWARD_ERROR(
          vkGetPhysicalDeviceSurfaceSupportKHR(physical_device, i, surface, &present_supported));
      if (present_supported != VK_TRUE) {
        continue;
      }
    }
    return i;
  }

  return make_unexpected(
      Error{.code = ErrorCode::kNotFound,
            .message = surface != VK_NULL_HANDLE
                           ? "[holoviz] No queue family supports both graphics and presentation to "
                             "the surface."
                           : "[holoviz] No suitable graphics queue found for Vulkan device."});
}

expected<bool> adapter_supports_surface(const Adapter& adapter, const Surface& surface) {
  VIZ_UNWRAP_NATIVE(VkPhysicalDevice, physical_device, adapter);
  VIZ_UNWRAP_NATIVE(VkSurfaceKHR, vk_surface, surface);

  // Supported means usable by a device this adapter would create: the device owns a single
  // queue that must both render and present, so a present-capable family without graphics
  // does not count.
  auto family = find_graphics_queue_family(physical_device, vk_surface);
  if (!family) {
    if (family.error().code == ErrorCode::kNotFound) {
      return false;
    }
    return make_unexpected(family.error());
  }
  return true;
}

expected<const PhysicalDeviceInfo*> find_physical_device_info(const std::array<uint8_t, 16>& uuid) {
  VIZ_UNWRAP(VulkanInstance, instance, GetVulkanInstance());

  for (const auto& device_info : instance.physical_device_infos_) {
    std::array<uint8_t, 16> device_uuid{};
    std::copy(std::begin(device_info.physical_device_id_properties_.deviceUUID),
              std::end(device_info.physical_device_id_properties_.deviceUUID),
              std::begin(device_uuid));

    if (device_uuid == uuid) {
      return &device_info;
    }
  }

  return make_unexpected(
      Error{.code = ErrorCode::kNotFound, .message = "[holoviz] VkPhysicalDevice not found"});
}

expected<const PhysicalDeviceInfo*> find_physical_device_info(VkPhysicalDevice physical_device) {
  VIZ_UNWRAP(VulkanInstance, instance, GetVulkanInstance());

  for (const auto& device_info : instance.physical_device_infos_) {
    if (device_info.physical_device_ == physical_device) {
      return &device_info;
    }
  }

  return make_unexpected(
      Error{.code = ErrorCode::kNotFound, .message = "[holoviz] VkPhysicalDevice not found"});
}

expected<NativeObject> adapter_native_object(const Adapter& adapter, NativeObjectType type) {
  if (type != NativeObjectType::kVkPhysicalDevice) {
    return make_unexpected(Error{.code = ErrorCode::kInvalidArgument,
                                 .message = "[holoviz] Adapter native object type not supported"});
  }

  VIZ_UNWRAP(const PhysicalDeviceInfo*, device_info, find_physical_device_info(adapter.uuid));
  return NativeObject{type, reinterpret_cast<uint64_t>(device_info->physical_device_)};
}

expected<std::unique_ptr<detail::DeviceImpl>> adapter_create_device(const Adapter& adapter,
                                                                    const DeviceDesc& desc,
                                                                    const Surface* surface) {
  VIZ_UNWRAP(VulkanInstance, instance, GetVulkanInstance());
  VIZ_UNWRAP_NATIVE(VkPhysicalDevice, physical_device, adapter);
  VkSurfaceKHR present_surface = VK_NULL_HANDLE;
  if (surface != nullptr) {
    VIZ_UNWRAP_NATIVE(VkSurfaceKHR, vk_surface, *surface);
    present_surface = vk_surface;
  }
  auto device = std::make_unique<VulkanDevice>();
  VIZ_FORWARD_ERROR(device->init(instance, physical_device, desc, adapter, present_surface));
  return std::move(device);
}

}  // namespace holoviz::vulkan
