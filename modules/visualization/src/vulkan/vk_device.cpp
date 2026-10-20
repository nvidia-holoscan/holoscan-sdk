/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <array>
#include <cstring>
#include "../detail/destroy_timeout.hpp"

#include <cuda.h>

#include <holoviz/display.hpp>
#include <holoviz/logger.hpp>

#include "vk_adapter.hpp"
#include "vk_buffer.hpp"
#include "vk_command_list.hpp"
#include "vk_device.hpp"
#include "vk_errors.hpp"
#include "vk_extension_util.hpp"
#include "vk_external_semaphore.hpp"
#include "vk_image.hpp"
#include "vk_pipeline.hpp"
#include "vk_query.hpp"
#include "vk_render_pass.hpp"
#include "vk_swapchain.hpp"

#include "vk_enum_conversions.hpp"

#include "../cuda/cu_errors.hpp"
#include "../cuda/cu_scoped_context.hpp"

#include "detail/surface_impl.hpp"

#include <algorithm>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace holoviz::vulkan {

namespace {

/// Queries what the adapter supports for one format, mirroring how VulkanImage::init
/// actually creates images: 2D, optimal tiling, and (for the CUDA export path) the
/// opaque OS external-memory handle type.
FormatCapability query_format_capabilities(VkPhysicalDevice physical_device, VkFormat vk_format) {
  VkFormatProperties format_properties{};
  vkGetPhysicalDeviceFormatProperties(physical_device, vk_format, &format_properties);
  const VkFormatFeatureFlags features = format_properties.optimalTilingFeatures;

  FormatCapability caps = FormatCapability::kNone;
  if ((features & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT) != 0) {
    caps = caps | FormatCapability::kSampled;
  }
  if ((features & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT) != 0) {
    caps = caps | FormatCapability::kRenderTarget;
  }
  if ((features & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT) != 0) {
    caps = caps | FormatCapability::kStorage;
  }
  if ((features & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BLEND_BIT) != 0) {
    caps = caps | FormatCapability::kBlendable;
  }

  // kCudaImportableImage: an image of this format must be creatable with the external-memory
  // handle type the export path uses, and the driver must mark that combination exportable
  // (CUDA imports what Vulkan exports).
  VkPhysicalDeviceExternalImageFormatInfo external_image_format_info{
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_IMAGE_FORMAT_INFO,
#if defined(_WIN32)
      .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT,
#else
      .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT,
#endif
  };
  const VkPhysicalDeviceImageFormatInfo2 image_format_info{
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2,
      .pNext = &external_image_format_info,
      .format = vk_format,
      .type = VK_IMAGE_TYPE_2D,
      .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
               VK_IMAGE_USAGE_SAMPLED_BIT,
  };
  VkExternalImageFormatProperties external_image_format_properties{
      .sType = VK_STRUCTURE_TYPE_EXTERNAL_IMAGE_FORMAT_PROPERTIES,
  };
  VkImageFormatProperties2 image_format_properties{
      .sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2,
      .pNext = &external_image_format_properties,
  };
  if (vkGetPhysicalDeviceImageFormatProperties2(
          physical_device, &image_format_info, &image_format_properties) == VK_SUCCESS &&
      (external_image_format_properties.externalMemoryProperties.externalMemoryFeatures &
       VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT) != 0) {
    caps = caps | FormatCapability::kCudaImportableImage;
  }

  return caps;
}

}  // namespace

// Keep context ownership, capability admission, and native resource creation in dependency order.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
expected<void> VulkanDevice::init(const VulkanInstance& instance, VkPhysicalDevice physical_device,
                                  const DeviceDesc& desc, const Adapter& adapter,
                                  VkSurfaceKHR present_surface) {
  if (desc.policy != DevicePolicy::kDynamic) {
    return make_unexpected(Error{.code = ErrorCode::kFailure,
                                 .message = "[holoviz] Unsupported device policy requested."});
  }

  if (desc.backend != Backend::kAuto && desc.backend != Backend::kVulkan) {
    return make_unexpected(
        Error{.code = ErrorCode::kFailure, .message = "[holoviz] Unsupported backend requested."});
  }

  if (desc.cuda_context_mode != CudaContextMode::kExternal && desc.cuda_context != nullptr) {
    return make_unexpected(
        Error{.code = ErrorCode::kFailure,
              .message = "[holoviz] Cannot supply a CUDA context when requesting CiG."});
  }

  if (desc.cuda_device >= 0 && desc.cuda_device != adapter.cuda_device_index) {
    return make_unexpected(Error{.code = ErrorCode::kFailure,
                                 .message = "[holoviz] CUDA device index does not match adapter."});
  }

  VIZ_CU_FORWARD_ERROR(cuInit(0));
  VIZ_CU_FORWARD_ERROR(cuDeviceGet(&cuda_device_, adapter.cuda_device_index));

  // All supported-extension checks and feature/property reads below come from the
  // instance's per-physical-device cache, filled by a single Features2/Properties2
  // query pair at instance creation.
  {
    auto device_info = find_physical_device_info(physical_device);
    if (!device_info) {
      return make_unexpected(
          Error{.code = ErrorCode::kFailure,
                .message = "[holoviz] Vulkan physical device is not known to the instance."});
    }
    physical_device_info_ = *device_info;
  }
  const PhysicalDeviceInfo& info = physical_device_info();

  // CiG (CUDA in Graphics) contexts share the GPU with the graphics queue through
  // VK_NV_external_compute_queue. Support is probed here because the extension and
  // the reserved queue count must be requested at device creation, but the context
  // itself can only be created after the device exists (it wraps the Vulkan queue).
  bool use_cig = false;
  uint32_t cig_shared_data_size = 0;
  if (desc.cuda_context_mode != CudaContextMode::kExternal) {
    if (info.has_external_compute_queue_properties_) {
      const VkPhysicalDeviceExternalComputeQueuePropertiesNV& cig_properties =
          info.external_compute_queue_properties_;
      use_cig = cig_properties.externalDataSize > 0 &&
                cig_properties.maxExternalQueues >= desc.reserved_external_queues;
      cig_shared_data_size = cig_properties.externalDataSize;
    }

    if (!use_cig) {
      if (desc.cuda_context_mode == CudaContextMode::kCigRequired) {
        return make_unexpected(Error{
            .code = ErrorCode::kFailure,
            .message =
                "[holoviz] CiG context required but " VK_NV_EXTERNAL_COMPUTE_QUEUE_EXTENSION_NAME
                " is not supported by this device/driver."});
      }
      HOLOVIZ_LOG_WARN(
          "[holoviz] CiG could not be enabled ({} is not supported by this device/driver), "
          "falling back to a regular CUDA context.",
          VK_NV_EXTERNAL_COMPUTE_QUEUE_EXTENSION_NAME);
    }
  }

  const auto retain_primary_context = [this]() -> expected<void> {
    VIZ_CU_FORWARD_ERROR(cuDevicePrimaryCtxRetain(&cuda_context_, cuda_device_));
    // Set immediately after a successful retain so the destructor releases the
    // primary context even if a later init step fails.
    owns_cuda_context_ = true;

    VIZ_CU_FORWARD_ERROR(cuCtxSetCurrent(cuda_context_));
    return {};
  };

  if (use_cig) {
    // Context creation is deferred until the Vulkan device and queue exist below.
  } else if (desc.cuda_context == nullptr) {
    if (auto retained = retain_primary_context(); !retained) {
      return retained;
    }
  } else {
    cuda_context_ = desc.cuda_context;
    owns_cuda_context_ = false;

    // Set (not push) like the HoloViz-owned paths: no context-stack entry leaks on the
    // caller's thread. See DeviceDesc::cuda_context for the currency contract.
    VIZ_CU_FORWARD_ERROR(cuCtxSetCurrent(cuda_context_));
  }

  instance_ = instance.vk_instance_;
  allocator_ = instance.vk_allocator_;
  physical_device_ = physical_device;
  adapter_ = adapter;

  // Snapshot the device limits and per-format capabilities callers can branch on up front.
  // The present feature flags stay at their defaults until the extension probes below.
  const VkPhysicalDeviceProperties& physical_device_properties =
      info.physical_device_properties_.properties;
  capabilities_.backend = Backend::kVulkan;
  capabilities_.adapter = adapter;
  capabilities_.max_image_extent = physical_device_properties.limits.maxImageDimension2D;
  capabilities_.max_push_data_bytes = physical_device_properties.limits.maxPushConstantsSize;
  capabilities_.max_bind_groups = physical_device_properties.limits.maxBoundDescriptorSets;
  capabilities_.storage_buffer_alignment =
      physical_device_properties.limits.minStorageBufferOffsetAlignment;

  // A format with no Vulkan spelling (kUnknown) keeps FormatCapability::kNone.
  for (uint32_t i = 0; i <= static_cast<uint32_t>(kLastFormat); ++i) {
    const auto vk_format = from_holoviz_format(static_cast<Format>(i));
    if (!vk_format) {
      continue;
    }
    capabilities_.format_caps[i] = query_format_capabilities(physical_device, *vk_format);
  }

  std::vector<const char*> desiredExtensions{VK_KHR_SWAPCHAIN_EXTENSION_NAME,
#ifdef _WIN32
                                             VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME,
                                             VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME
#else
                                             VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
                                             VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME
#endif
  };

  if (use_cig) {
    desiredExtensions.push_back(VK_NV_EXTERNAL_COMPUTE_QUEUE_EXTENSION_NAME);
  }

  // Optional: fill-rectangle rasterization (PolygonMode::kFillRectangle). Enabled whenever
  // the driver offers it; pipelines opt in per PipelineDesc and creation validates against
  // the capability flag.
  if (info.extension_supported(VK_NV_FILL_RECTANGLE_EXTENSION_NAME)) {
    desiredExtensions.push_back(VK_NV_FILL_RECTANGLE_EXTENSION_NAME);
    capabilities_.fill_rectangle = true;
  }

  // Direct-display capability: "the compositor-bypass path is wired on this device/driver",
  // not "a display is currently free" — that is per-display (DisplayOwnership,
  // holoviz/display.hpp).
  // Exclusive display is Linux-only (DRM); the acquire extension is instance-level, so nothing
  // here needs enabling on the device.
#ifdef __linux__
  capabilities_.exclusive_display =
      instance.display_supported_ && instance.direct_mode_display_supported_;
#else
  capabilities_.exclusive_display = false;
#endif

  // Optional: display control (VK_EXT_display_control — display events, swapchain vblank
  // counters). Requires the VK_EXT_display_surface_counter instance extension; reported as
  // CapabilityReport::display_events.
  if (instance.display_surface_counter_supported_ &&
      info.extension_supported(VK_EXT_DISPLAY_CONTROL_EXTENSION_NAME)) {
    desiredExtensions.push_back(VK_EXT_DISPLAY_CONTROL_EXTENSION_NAME);
    display_control_enabled_ = true;
    capabilities_.display_events = true;
  }

  // Optional: shared presentable images (VK_KHR_shared_presentable_image — the
  // kSharedDemandRefresh / kSharedContinuousRefresh present modes). Extension-only, no
  // feature struct; its VK_KHR_get_surface_capabilities_2 instance dependency is always
  // enabled. surface_capabilities() masks the shared modes against this flag.
  if (info.extension_supported(VK_KHR_SHARED_PRESENTABLE_IMAGE_EXTENSION_NAME)) {
    desiredExtensions.push_back(VK_KHR_SHARED_PRESENTABLE_IMAGE_EXTENSION_NAME);
    shared_present_enabled_ = true;
    capabilities_.shared_present = true;
  }

  // Optional: FIFO_LATEST_READY presentation (PresentMode::kFifoLatestReady). The present
  // mode is only valid when the device extension *and* its feature are enabled — a surface
  // may report the mode regardless, which is why surface_capabilities() masks it against
  // fifo_latest_ready_enabled_. Prefer the promoted KHR extension, fall back to EXT.
#if defined(VK_EXT_present_mode_fifo_latest_ready)
  if (info.fifo_latest_ready_ext_name_ != nullptr &&
      info.fifo_latest_ready_features_.presentModeFifoLatestReady == VK_TRUE) {
    desiredExtensions.push_back(info.fifo_latest_ready_ext_name_);
    fifo_latest_ready_enabled_ = true;
  }
#endif

  // Optional: native sRGB attachment views even when scanout advertises only UNORM.
  // Vulkan 1.2 already provides this extension's image-format-list dependency.
  if (info.extension_supported(VK_KHR_SWAPCHAIN_MUTABLE_FORMAT_EXTENSION_NAME)) {
    desiredExtensions.push_back(VK_KHR_SWAPCHAIN_MUTABLE_FORMAT_EXTENSION_NAME);
    capabilities_.mutable_swapchain_format = true;
  }

  // Required: swapchain maintenance (VK_EXT_swapchain_maintenance1 or its KHR promotion).
  // The module's swapchain depends on it: present fences make retired-swapchain and
  // semaphore teardown deterministic, vkReleaseSwapchainImages returns acquired-but-
  // unpresented images during resize, and per-present mode switching avoids recreation.
  // Each device extension requires its corresponding instance extension
  // (VUID-vkCreateDevice-ppEnabledExtensionNames-01387); PhysicalDeviceInfo::init picks the
  // spelling that matches the enabled surface maintenance extension.
#if defined(VK_EXT_swapchain_maintenance1)
  if (info.swapchain_maintenance1_ext_name_ == nullptr ||
      info.swapchain_maintenance1_features_.swapchainMaintenance1 != VK_TRUE) {
    return make_unexpected(
        Error{.code = ErrorCode::kFailure,
              .message = "[holoviz] Vulkan device requires swapchainMaintenance1 and a swapchain "
                         "maintenance extension matching the enabled surface maintenance "
                         "instance extension."});
  }
  desiredExtensions.push_back(info.swapchain_maintenance1_ext_name_);
#else
#error "Vulkan headers too old: VK_EXT_swapchain_maintenance1 is required."
#endif

  // Optional: present feedback (VK_KHR_present_id + VK_KHR_present_wait). Enabled together
  // or not at all — an id you cannot wait on serves nothing this module needs — and
  // reported as CapabilityReport::present_completion. Swapchain::present() then tags frames
  // with the caller's present_id and Swapchain::wait_for_present() blocks until a tagged
  // frame is displayed.
#if defined(VK_KHR_present_id) && defined(VK_KHR_present_wait)
  // The cached structs stay zeroed (VK_FALSE) unless their extension is supported, so
  // testing the features alone also covers extension support.
  if (info.present_id_features_.presentId == VK_TRUE &&
      info.present_wait_features_.presentWait == VK_TRUE) {
    desiredExtensions.push_back(VK_KHR_PRESENT_ID_EXTENSION_NAME);
    desiredExtensions.push_back(VK_KHR_PRESENT_WAIT_EXTENSION_NAME);
    capabilities_.present_completion = true;
  }
#endif

  // Optional: measured present timing (VK_EXT_present_timing). Samples are correlated by
  // present id, so it rides on present_completion; reported as
  // CapabilityReport::present_timing and opted into per swapchain
  // (SwapchainDesc::enable_present_timing).
#if defined(VK_EXT_present_timing)
  // A validation layer older than the extension (published in Vulkan 1.4.335) rejects its
  // structures as unknown and has been observed to crash on them; leave the capability off
  // under such a layer rather than let a debugging aid take the process down.
  constexpr uint32_t kPresentTimingMinValidationLayerVersion = VK_MAKE_API_VERSION(0, 1, 4, 335);
  const bool validation_layer_too_old =
      instance.validation_layer_spec_version_ != 0 &&
      instance.validation_layer_spec_version_ < kPresentTimingMinValidationLayerVersion;
  // VK_EXT_present_timing formally depends on VK_KHR_present_id2 (enabled alongside; the
  // module keeps tagging presents through VkPresentIdKHR, which the timing samples report).
  bool present_id2_available = false;
#if defined(VK_KHR_present_id2)
  present_id2_available = info.extension_supported(VK_KHR_PRESENT_ID_2_EXTENSION_NAME) &&
                          info.present_id2_features_.presentId2 == VK_TRUE;
#endif
  if (capabilities_.present_completion && present_id2_available &&
      info.present_timing_features_.presentTiming == VK_TRUE) {
    if (validation_layer_too_old) {
      HOLOVIZ_LOG_WARN(
          "[holoviz] VK_LAYER_KHRONOS_validation {}.{}.{} predates VK_EXT_present_timing "
          "(1.4.335); present timing stays disabled while validation is on.",
          VK_API_VERSION_MAJOR(instance.validation_layer_spec_version_),
          VK_API_VERSION_MINOR(instance.validation_layer_spec_version_),
          VK_API_VERSION_PATCH(instance.validation_layer_spec_version_));
    } else {
#if defined(VK_KHR_present_id2)
      desiredExtensions.push_back(VK_KHR_PRESENT_ID_2_EXTENSION_NAME);
#endif
      desiredExtensions.push_back(VK_EXT_PRESENT_TIMING_EXTENSION_NAME);
      present_timing_enabled_ = true;
      capabilities_.present_timing = true;
    }
  }
#endif

  // Optional: calibrated timestamps (VK_KHR_calibrated_timestamps or its EXT spelling),
  // only worth enabling when the device and the host monotonic clock can be sampled
  // together; reported as CapabilityReport::calibrated_timestamps.
  if (info.calibrated_timestamps_ext_name_ != nullptr && info.calibrated_host_domain_supported()) {
    desiredExtensions.push_back(info.calibrated_timestamps_ext_name_);
    calibrated_timestamps_ext_name_ = info.calibrated_timestamps_ext_name_;
    capabilities_.calibrated_timestamps = true;
  }

  for (const char* extension : desiredExtensions) {
    if (!info.extension_supported(extension)) {
      return make_unexpected(
          Error{.code = ErrorCode::kFailure,
                .message = std::string("[holoviz] Vulkan device does not support the required "
                                       "extension: ") +
                           extension});
    }
  }

  // The same joint graphics(+present) selection adapter_supports_surface() answers for, so a
  // surface-aware device never ends up with a queue that cannot present to its surface.
  VIZ_UNWRAP(
      uint32_t, queue_family_index, find_graphics_queue_family(physical_device, present_surface));

  // Timeline semaphores are an opt-in feature, they must be enabled at device creation
  // before they can be used.
  if (info.timeline_semaphore_features_.timelineSemaphore == 0) {
    return make_unexpected(
        Error{.code = ErrorCode::kFailure,
              .message = "[holoviz] Vulkan device does not support timeline semaphores."});
  }

  const float queue_priority = 1.0f;

  const VkDeviceQueueCreateInfo queue_create_info{
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = queue_family_index,
      .queueCount = 1,
      .pQueuePriorities = &queue_priority,
  };

  VkExternalComputeQueueDeviceCreateInfoNV external_queue_reserve_info{
      .sType = VK_STRUCTURE_TYPE_EXTERNAL_COMPUTE_QUEUE_DEVICE_CREATE_INFO_NV,
      .reservedExternalQueues = desc.reserved_external_queues,
  };

  // Shaders reading SV_InstanceID/SV_VertexID (e.g. the vertex-buffer-less monospace text
  // renderer) compile to SPIR-V declaring the DrawParameters capability, which requires the
  // Vulkan 1.1 core shaderDrawParameters feature. Enable it when the device supports it;
  // chaining it disabled is harmless.
  VkPhysicalDeviceShaderDrawParametersFeatures draw_parameters_features{
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_DRAW_PARAMETERS_FEATURES,
      .pNext = use_cig ? &external_queue_reserve_info : nullptr,
      .shaderDrawParameters = info.shader_draw_parameters_features_.shaderDrawParameters,
  };

#if defined(VK_EXT_present_mode_fifo_latest_ready)
  VkPhysicalDevicePresentModeFifoLatestReadyFeaturesEXT fifo_latest_ready_features{
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_MODE_FIFO_LATEST_READY_FEATURES_EXT,
      .pNext = draw_parameters_features.pNext,
      .presentModeFifoLatestReady = VK_TRUE,
  };
  if (fifo_latest_ready_enabled_) {
    draw_parameters_features.pNext = &fifo_latest_ready_features;
  }
#endif

  VkPhysicalDeviceSwapchainMaintenance1FeaturesEXT swapchain_maintenance1_features{
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SWAPCHAIN_MAINTENANCE_1_FEATURES_EXT,
      .pNext = draw_parameters_features.pNext,
      .swapchainMaintenance1 = VK_TRUE,
  };
  draw_parameters_features.pNext = &swapchain_maintenance1_features;

#if defined(VK_KHR_present_id) && defined(VK_KHR_present_wait)
  VkPhysicalDevicePresentWaitFeaturesKHR present_wait_features{
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_WAIT_FEATURES_KHR,
      .pNext = draw_parameters_features.pNext,
      .presentWait = VK_TRUE,
  };
  VkPhysicalDevicePresentIdFeaturesKHR present_id_features{
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_ID_FEATURES_KHR,
      .pNext = &present_wait_features,
      .presentId = VK_TRUE,
  };
  if (capabilities_.present_completion) {
    draw_parameters_features.pNext = &present_id_features;
  }
#endif

#if defined(VK_EXT_present_timing) && defined(VK_KHR_present_id2)
  VkPhysicalDevicePresentId2FeaturesKHR present_id2_features{
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_ID_2_FEATURES_KHR,
      .pNext = draw_parameters_features.pNext,
      .presentId2 = VK_TRUE,
  };
  VkPhysicalDevicePresentTimingFeaturesEXT present_timing_features{
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_TIMING_FEATURES_EXT,
      .pNext = &present_id2_features,
      .presentTiming = VK_TRUE,
  };
  if (present_timing_enabled_) {
    draw_parameters_features.pNext = &present_timing_features;
  }
#endif

  VkPhysicalDeviceTimelineSemaphoreFeatures timeline_semaphore_features{
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES,
      .pNext = &draw_parameters_features,
      .timelineSemaphore = VK_TRUE,
  };

  // Features enabled through this chain, `pEnabledFeatures` must stay null.
  VkPhysicalDeviceFeatures2 enabled_features{
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
      .pNext = &timeline_semaphore_features,
  };

  const VkDeviceCreateInfo device_create_info{
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .pNext = &enabled_features,
      .queueCreateInfoCount = 1,
      .pQueueCreateInfos = &queue_create_info,
      .enabledExtensionCount = static_cast<uint32_t>(desiredExtensions.size()),
      .ppEnabledExtensionNames = desiredExtensions.data(),
  };

  VIZ_VK_FORWARD_ERROR(
      vkCreateDevice(physical_device, &device_create_info, instance.vk_allocator_, &device_));

  volkLoadDeviceTable(&fn_, device_);

  VIZ_VK_DEVICE_FUNC_NO_ERROR(vkGetDeviceQueue, this, queue_family_index, 0, &queue_);
  queue_family_index_ = static_cast<int32_t>(queue_family_index);

  // Timestamp parameters of the graphics queue: the one tick-to-ns conversion every raw GPU
  // timestamp (query result, calibrated sample) goes through (device_ticks_to_ns).
  // timestampValidBits == 0 means the queue
  // cannot timestamp at all, which query-pool creation reports.
  timestamp_period_ns_ = info.physical_device_properties_.properties.limits.timestampPeriod;
  if (queue_family_index < info.queue_family_properties_.size()) {
    const uint32_t valid_bits =
        info.queue_family_properties_[queue_family_index].queueFamilyProperties.timestampValidBits;
    timestamp_valid_bits_mask_ =
        valid_bits == 0 ? 0 : (valid_bits >= 64 ? ~0ULL : (1ULL << valid_bits) - 1);
  }

  if (use_cig) {
    auto cig_context = [&]() -> expected<void> {
      const VkExternalComputeQueueCreateInfoNV external_queue_info{
          .sType = VK_STRUCTURE_TYPE_EXTERNAL_COMPUTE_QUEUE_CREATE_INFO_NV,
          .preferredQueue = queue_,
      };
      VIZ_VK_DEVICE_FUNC(vkCreateExternalComputeQueueNV,
                         this,
                         &external_queue_info,
                         allocator_,
                         &external_compute_queue_);

      VkExternalComputeQueueDataParamsNV data_params{
          .sType = VK_STRUCTURE_TYPE_EXTERNAL_COMPUTE_QUEUE_DATA_PARAMS_NV,
      };
      std::vector<uint8_t> shared_data(cig_shared_data_size);
      fn_.vkGetExternalComputeQueueDataNV(
          external_compute_queue_, &data_params, shared_data.data());

      CUctxCigParam cig_param{
          .sharedDataType = CIG_DATA_TYPE_NV_BLOB,
          .sharedData = shared_data.data(),
      };
      CUctxCreateParams context_params{
          .cigParams = &cig_param,
      };
      VIZ_CU_FORWARD_ERROR(cuCtxCreate_v4(&cuda_context_, &context_params, 0, cuda_device_));
      // cuCtxCreate makes the new context current on this thread, no push/set needed.
      owns_cuda_context_ = true;
      cuda_context_is_cig_ = true;
      return {};
    }();

    if (!cig_context) {
      if (desc.cuda_context_mode == CudaContextMode::kCigRequired) {
        return make_unexpected(cig_context.error());
      }

      HOLOVIZ_LOG_WARN(
          "[holoviz] CiG could not be enabled ({}), falling back to a regular CUDA context.",
          cig_context.error().message);
      if (external_compute_queue_ != VK_NULL_HANDLE) {
        VIZ_VK_DEVICE_FUNC_NO_ERROR(
            vkDestroyExternalComputeQueueNV, this, external_compute_queue_, allocator_);
        external_compute_queue_ = VK_NULL_HANDLE;
      }
      if (auto retained = retain_primary_context(); !retained) {
        return retained;
      }
    }
  }

  const VmaVulkanFunctions vmaVulkanFunctions = {
      .vkGetInstanceProcAddr = vkGetInstanceProcAddr,
      .vkGetDeviceProcAddr = vkGetDeviceProcAddr,
      .vkAllocateMemory = fn_.vkAllocateMemory,
      .vkFreeMemory = fn_.vkFreeMemory,
      .vkMapMemory = fn_.vkMapMemory,
      .vkUnmapMemory = fn_.vkUnmapMemory,
  };

  const VmaAllocatorCreateInfo allocatorCreateInfo = {
      .physicalDevice = physical_device_,
      .device = device_,
      .pVulkanFunctions = &vmaVulkanFunctions,
      .instance = instance_,
      .vulkanApiVersion = VK_API_VERSION_1_2,
  };

  VIZ_VK_FORWARD_ERROR(vmaCreateAllocator(&allocatorCreateInfo, &vma_allocator_));

  const VkFenceCreateInfo fence_info{
      .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
  };

  VIZ_VK_DEVICE_FUNC(vkCreateFence, this, &fence_info, allocator_, &cmd_fence_);

  auto semaphore_pair = make_exported_semaphore_pair(this);
  if (!semaphore_pair) {
    return make_unexpected(semaphore_pair.error());
  }

  timeline_semaphore_ = semaphore_pair->first;
  timeline_external_semaphore_ = semaphore_pair->second;

  const VkCommandPoolCreateInfo command_pool_info{
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .queueFamilyIndex = queue_family_index,
  };

  VIZ_VK_DEVICE_FUNC(vkCreateCommandPool, this, &command_pool_info, allocator_, &command_pool_);

  const VkCommandBufferAllocateInfo alloc_info{
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = command_pool_,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1,
  };

  VIZ_VK_DEVICE_FUNC(vkAllocateCommandBuffers, this, &alloc_info, &command_buffer_);

  vkGetPhysicalDeviceMemoryProperties(physical_device_, &memory_properties_);

  return {};
}

expected<void> VulkanDevice::seal() {
  return make_unexpected(Error{.code = ErrorCode::kFailure,
                               .message = "[holoviz] VulkanDevice::seal() is not implemented."});
}

expected<void> VulkanDevice::wait_idle() {
  VIZ_VK_DEVICE_FUNC(vkDeviceWaitIdle, this);
  return {};
}

expected<void> VulkanDevice::wait_on_submit(SubmitSyncToken sync_token,
                                            std::chrono::nanoseconds timeout) {
  uint64_t last_completed = 0;
  VIZ_VK_DEVICE_FUNC(vkGetSemaphoreCounterValue, this, timeline_semaphore_, &last_completed);
  if (last_completed >= sync_token.timeline_value) {
    return {};
  }

  const VkSemaphoreWaitInfo wait_info{
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
      .semaphoreCount = 1,
      .pSemaphores = &timeline_semaphore_,
      .pValues = &sync_token.timeline_value,
  };
  const VkResult result = fn_.vkWaitSemaphores(device_, &wait_info, timeout.count());

  if (result == VK_TIMEOUT) {
    return make_unexpected(
        Error{.code = ErrorCode::kBackpressured,
              .message = "[holoviz] Timeout while waiting for Vulkan timeline semaphore."});
  }

  VIZ_VK_FORWARD_ERROR(result);

  return {};
}

// Keep gate validation, synchronization arrays, and their single queue-submit commit together.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
expected<SubmitSyncToken> VulkanDevice::submit(const SubmitDesc& batch) {
  std::vector<VkSemaphore> wait_semaphores;
  std::vector<VkPipelineStageFlags> wait_stages;
  std::vector<uint64_t> wait_values;

  std::vector<VkSemaphore> signal_semaphores;
  std::vector<uint64_t> signal_values;

  uint64_t wait_timeline_value = 0;
  for (const auto& submit_token : batch.wait_submit_sync_tokens) {
    if (submit_token.timeline_value > timeline_value_) {
      return make_unexpected(
          Error{.code = ErrorCode::kFailure,
                .message = "[holoviz] VulkanDevice::submit() received a SubmitSyncToken "
                           "with a timeline value greater than the current timeline."});
    }

    if (submit_token.timeline_value > wait_timeline_value) {
      wait_timeline_value = submit_token.timeline_value;
    }
  }

  if (wait_timeline_value > 0) {
    wait_semaphores.push_back(timeline_semaphore_);
    wait_stages.push_back(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
    wait_values.push_back(wait_timeline_value);
  }

  std::vector<VulkanExternalSemaphore*> vulkan_external_semaphores;
  for (const auto& external_sync_token : batch.wait_external_sync_tokens) {
    if (external_sync_token.impl == nullptr && external_sync_token.backend == Backend::kAuto) {
      // Default-constructed no-op token (e.g. from a producer with nothing to synchronize,
      // like the monospace text renderer): nothing to wait on. Mirrors the timeline_value == 0
      // skip for wait_submit_sync_tokens above.
      continue;
    }

    if (external_sync_token.backend != Backend::kVulkan) {
      return make_unexpected(Error{
          .code = ErrorCode::kFailure,
          .message =
              "[holoviz] VulkanDevice::submit() only supports wait on Vulkan ExternalSyncToken."});
    }

    if (external_sync_token.impl == nullptr) {
      return make_unexpected(Error{
          .code = ErrorCode::kFailure,
          .message =
              "[holoviz] VulkanDevice::submit() received an ExternalSyncToken with null impl."});
    }

    VulkanExternalSemaphore* external_semaphore =
        reinterpret_cast<VulkanExternalSemaphore*>(external_sync_token.impl);
    if (external_semaphore->device() != this) {
      return make_unexpected(Error{.code = ErrorCode::kFailure,
                                   .message = "[holoviz] VulkanDevice::submit() received an "
                                              "ExternalSyncToken from a different device."});
    }

    wait_semaphores.push_back(external_semaphore->timeline_semaphore());
    wait_stages.push_back(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
    wait_values.push_back(external_sync_token.timeline_value);
    // Stamped with this submit's timeline value in phase 3, so the semaphore's
    // destroy() can drain the submits that wait on it.
    vulkan_external_semaphores.push_back(external_semaphore);
  }

  signal_semaphores.push_back(timeline_semaphore_);
  signal_values.push_back(timeline_value_ + 1);

  // Phase 1 of 3 — validate and collect, with zero state mutation. Gates are
  // consumed only after vkQueueSubmit succeeds (phase 3): a failed submit must
  // leave every gate unconsumed so the swapchain's unpresented-acquire drain
  // path (release_unpresented_acquire) sees the truth and the caller's frame
  // loop can drain/retry or recreate cleanly.
  //
  // The same gate passed twice in one batch used to be caught by the first
  // consume() tripping the second consumed() check; with consumption deferred,
  // reject duplicates explicitly by semaphore identity.
  std::vector<VkSemaphore> gate_semaphores;

  // Swapchain-minted gates get this submit's timeline value recorded on their owning frame
  // slot in phase 3, so the swapchain's resize()/destroy() drains cover the submit even if
  // the frame is never presented. A gate with a foreign impl has no slot to record on and
  // is skipped (the recording is backend bookkeeping, not part of the submit contract).
  std::vector<VulkanAcquireSignalGate*> vulkan_acquire_gates;
  std::vector<VulkanPresentWaitGate*> vulkan_present_gates;

  const auto is_duplicate_gate = [&gate_semaphores](VkSemaphore semaphore) {
    return std::find(gate_semaphores.begin(), gate_semaphores.end(), semaphore) !=
           gate_semaphores.end();
  };

  // TODO: This for loop and the one below should be combined somehow (they are so similar)
  for (auto& acquire_gate : batch.acquire_gates) {
    if (!acquire_gate) {
      return make_unexpected(
          Error{.code = ErrorCode::kFailure,
                .message = "[holoviz] Acquire gate operation requires a valid gate."});
    }

    VIZ_UNWRAP(bool, acquire_gate_consumed, acquire_gate.consumed());

    if (acquire_gate_consumed) {
      return make_unexpected(Error{.code = ErrorCode::kFailure,
                                   .message = "[holoviz] Acquire gate has already been consumed."});
    }

    VIZ_UNWRAP_NATIVE(VkDevice, acquire_gate_device, acquire_gate);
    if (acquire_gate_device != device_) {
      return make_unexpected(Error{.code = ErrorCode::kInvalidArgument,
                                   .message = "[holoviz] Acquire gate belongs to another device."});
    }

    VIZ_UNWRAP_NATIVE(VkSemaphore, acquire_semaphore, acquire_gate);
    if (is_duplicate_gate(acquire_semaphore)) {
      return make_unexpected(
          Error{.code = ErrorCode::kFailure,
                .message = "[holoviz] The same gate was passed more than once in one batch."});
    }
    gate_semaphores.push_back(acquire_semaphore);
    if (auto* vulkan_gate = dynamic_cast<VulkanAcquireSignalGate*>(&acquire_gate.impl())) {
      vulkan_acquire_gates.push_back(vulkan_gate);
    }
    VIZ_UNWRAP(GateWaitStage, wait_stage, acquire_gate.wait_stage());
    wait_semaphores.push_back(acquire_semaphore);
    wait_stages.push_back(from_holoviz_gate_wait_stage(wait_stage));
    // Binary semaphore: VkTimelineSemaphoreSubmitInfo ignores the value at this index.
    wait_values.push_back(0);
  }

  for (auto& present_gate : batch.present_gates) {
    if (!present_gate) {
      return make_unexpected(
          Error{.code = ErrorCode::kFailure,
                .message = "[holoviz] Present gate operation requires a valid gate."});
    }
    VIZ_UNWRAP(bool, present_gate_consumed, present_gate.consumed());

    if (present_gate_consumed) {
      return make_unexpected(Error{.code = ErrorCode::kFailure,
                                   .message = "[holoviz] Present gate has already been consumed."});
    }

    VIZ_UNWRAP_NATIVE(VkDevice, present_gate_device, present_gate);
    if (present_gate_device != device_) {
      return make_unexpected(Error{.code = ErrorCode::kInvalidArgument,
                                   .message = "[holoviz] Present gate belongs to another device."});
    }

    VIZ_UNWRAP_NATIVE(VkSemaphore, present_semaphore, present_gate);
    if (is_duplicate_gate(present_semaphore)) {
      return make_unexpected(
          Error{.code = ErrorCode::kFailure,
                .message = "[holoviz] The same gate was passed more than once in one batch."});
    }
    gate_semaphores.push_back(present_semaphore);
    if (auto* vulkan_gate = dynamic_cast<VulkanPresentWaitGate*>(&present_gate.impl())) {
      vulkan_present_gates.push_back(vulkan_gate);
    }
    signal_semaphores.push_back(present_semaphore);
    // Binary semaphore: VkTimelineSemaphoreSubmitInfo ignores the value at this index.
    signal_values.push_back(0);
  }

  VkTimelineSemaphoreSubmitInfo timeline_info{
      .sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
      .waitSemaphoreValueCount = static_cast<uint32_t>(wait_values.size()),
      .pWaitSemaphoreValues = wait_values.data(),
      .signalSemaphoreValueCount = static_cast<uint32_t>(signal_values.size()),
      .pSignalSemaphoreValues = signal_values.data(),
  };

  std::vector<VkCommandBuffer> command_buffers;
  std::vector<VulkanCommandList*> vulkan_cmd_lists;
  for (auto& command_list : batch.command_lists) {
    VIZ_UNWRAP_NATIVE(VkDevice, command_device, command_list);
    if (command_device != device_) {
      return make_unexpected(Error{.code = ErrorCode::kInvalidArgument,
                                   .message = "[holoviz] Command list belongs to another device."});
    }

    VIZ_UNWRAP_NATIVE(VkCommandBuffer, vk_command_buffer, command_list);
    command_buffers.push_back(vk_command_buffer);

    auto* vulkan_cmd_list = dynamic_cast<VulkanCommandList*>(&command_list.impl());
    if (vulkan_cmd_list == nullptr) {
      return make_unexpected(Error{
          .code = ErrorCode::kFailure,
          .message = "[holoviz] Command list operation requires a valid Vulkan command list."});
    }
    vulkan_cmd_lists.push_back(vulkan_cmd_list);
  }

  const VkSubmitInfo submit_info{
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .pNext = &timeline_info,
      .waitSemaphoreCount = static_cast<uint32_t>(wait_semaphores.size()),
      .pWaitSemaphores = wait_semaphores.data(),
      .pWaitDstStageMask = wait_stages.data(),
      .commandBufferCount = static_cast<uint32_t>(command_buffers.size()),
      .pCommandBuffers = command_buffers.data(),
      .signalSemaphoreCount = static_cast<uint32_t>(signal_semaphores.size()),
      .pSignalSemaphores = signal_semaphores.data(),
  };

  // Phase 2 of 3 — the submit itself. On failure the function returns with no
  // state mutated: gates unconsumed, timeline_value_ unchanged, no submit
  // timeline recorded on any command list.
  VIZ_VK_FORWARD_ERROR(fn_.vkQueueSubmit(queue_, 1, &submit_info, VK_NULL_HANDLE));

  // Phase 3 of 3 — commit. Only bookkeeping past this point; the Vulkan gate
  // consume() is an infallible flag write (see vk_swapchain.cpp), ordered after
  // the successful submit so it records the truth.
  timeline_value_++;

  for (VulkanCommandList* vulkan_cmd_list : vulkan_cmd_lists) {
    vulkan_cmd_list->set_submit_timeline(timeline_value_);
  }

  // Record on the owning swapchain frame slots before anything can observe the consumed
  // gates: an infallible pointer write, keeping this commit phase no-fail.
  for (VulkanAcquireSignalGate* vulkan_gate : vulkan_acquire_gates) {
    vulkan_gate->record_submit_timeline(timeline_value_);
  }
  for (VulkanPresentWaitGate* vulkan_gate : vulkan_present_gates) {
    vulkan_gate->record_submit_timeline(timeline_value_);
  }

  // Same for the external semaphores this batch waited on: their destroy() drains
  // the device timeline up to the latest waiting submit before releasing the
  // semaphore Vulkan is still referencing.
  for (VulkanExternalSemaphore* external_semaphore : vulkan_external_semaphores) {
    external_semaphore->record_submit_timeline(timeline_value_);
  }

  for (auto& acquire_gate : batch.acquire_gates) {
    VIZ_FORWARD_ERROR(acquire_gate.consume());
  }
  for (auto& present_gate : batch.present_gates) {
    VIZ_FORWARD_ERROR(present_gate.consume());
  }

  return SubmitSyncToken{.timeline_value = timeline_value_};
}

void VulkanDevice::register_render_pass(VulkanRenderPass* pass) const {
  const std::lock_guard<std::mutex> lock(render_pass_registry_mutex_);
  render_pass_registry_.push_back(pass);
}

void VulkanDevice::unregister_render_pass(VulkanRenderPass* pass) const {
  const std::lock_guard<std::mutex> lock(render_pass_registry_mutex_);
  render_pass_registry_.erase(
      std::remove(render_pass_registry_.begin(), render_pass_registry_.end(), pass),
      render_pass_registry_.end());
}

void VulkanDevice::notify_image_view_destroyed(VkImageView view) const {
  const std::lock_guard<std::mutex> lock(render_pass_registry_mutex_);
  // invalidate_framebuffers never re-enters the registry, so holding the lock
  // across the calls is fine.
  for (VulkanRenderPass* pass : render_pass_registry_) {
    pass->invalidate_framebuffers(view);
  }
}

expected<CalibratedTimestamps> VulkanDevice::calibrated_timestamps() const {
  if (calibrated_timestamps_ext_name_ == nullptr) {
    return make_unexpected(
        Error{.code = ErrorCode::kNotImplemented,
              .message = "[holoviz] calibrated_timestamps() needs VK_KHR_calibrated_timestamps "
                         "with the device and host monotonic domains; branch on "
                         "CapabilityReport::calibrated_timestamps."});
  }
#ifdef _WIN32
  constexpr VkTimeDomainKHR kHostDomain = VK_TIME_DOMAIN_QUERY_PERFORMANCE_COUNTER_KHR;
#else
  constexpr VkTimeDomainKHR kHostDomain = VK_TIME_DOMAIN_CLOCK_MONOTONIC_KHR;
#endif
  const std::array<VkCalibratedTimestampInfoKHR, 2> infos{{
      {.sType = VK_STRUCTURE_TYPE_CALIBRATED_TIMESTAMP_INFO_KHR,
       .timeDomain = VK_TIME_DOMAIN_DEVICE_KHR},
      {.sType = VK_STRUCTURE_TYPE_CALIBRATED_TIMESTAMP_INFO_KHR, .timeDomain = kHostDomain},
  }};
  std::array<uint64_t, 2> timestamps{};
  uint64_t max_deviation = 0;
  // The KHR and EXT entry points are aliases; the device table holds whichever was enabled.
  const auto get_calibrated = fn_.vkGetCalibratedTimestampsKHR != nullptr
                                  ? fn_.vkGetCalibratedTimestampsKHR
                                  : fn_.vkGetCalibratedTimestampsEXT;
  if (get_calibrated == nullptr) {
    return make_unexpected(Error{.code = ErrorCode::kNotImplemented,
                                 .message = "[holoviz] vkGetCalibratedTimestamps entry point "
                                            "missing despite the extension being enabled."});
  }
  VIZ_VK_FORWARD_ERROR(get_calibrated(device_,
                                      static_cast<uint32_t>(infos.size()),
                                      infos.data(),
                                      timestamps.data(),
                                      &max_deviation));
  CalibratedTimestamps calibrated{
      .device_ns = device_ticks_to_ns(timestamps[0]),
      .host_ns = timestamps[1],
      .max_deviation_ns = max_deviation,
  };
#ifdef _WIN32
  // QueryPerformanceCounter ticks -> nanoseconds.
  static const double kQpcTicksPerSecond = [] {
    LARGE_INTEGER frequency{};
    QueryPerformanceFrequency(&frequency);
    return static_cast<double>(frequency.QuadPart);
  }();
  calibrated.host_ns =
      static_cast<uint64_t>(static_cast<double>(timestamps[1]) * 1e9 / kQpcTicksPerSecond);
#endif
  return calibrated;
}

// Keep present-mode, scaling, and timing queries together for one consistent surface capability
// result.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
expected<SurfaceCapabilities> VulkanDevice::surface_capabilities(
    const detail::SurfaceImpl& surface, std::optional<PresentMode> present_mode) const {
  VIZ_UNWRAP_NATIVE(VkSurfaceKHR, vk_surface, surface);

  // Per-present-mode query (VK_EXT_surface_maintenance1): with the mode chained, the
  // image-count/extent bounds are those the swapchain will be validated against for that
  // mode; without it the driver reports mode-agnostic bounds, which can be looser.
  VkSurfacePresentModeEXT surface_present_mode{
      .sType = VK_STRUCTURE_TYPE_SURFACE_PRESENT_MODE_EXT,
  };
  if (present_mode.has_value()) {
    if (*present_mode == PresentMode::kAuto) {
      return make_unexpected(
          Error{.code = ErrorCode::kInvalidArgument,
                .message = "[holoviz] surface_capabilities() needs a concrete present mode, "
                           "not kAuto; resolve the mode first."});
    }
    VIZ_UNWRAP(VkPresentModeKHR, vk_mode, from_holoviz_present_mode(*present_mode));
    surface_present_mode.presentMode = vk_mode;
  }

  // Surface capabilities through the "2" query so extension capability structs can be
  // chained: present timing (stage mask) when the device enabled it.
#if defined(VK_EXT_present_timing)
  VkPresentTimingSurfaceCapabilitiesEXT timing_caps{
      .sType = VK_STRUCTURE_TYPE_PRESENT_TIMING_SURFACE_CAPABILITIES_EXT,
  };
#endif
  VkSurfaceCapabilities2KHR surface_caps2{
      .sType = VK_STRUCTURE_TYPE_SURFACE_CAPABILITIES_2_KHR,
  };
#if defined(VK_EXT_present_timing)
  if (present_timing_enabled_) {
    surface_caps2.pNext = &timing_caps;
  }
#endif
  const VkPhysicalDeviceSurfaceInfo2KHR surface_info{
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SURFACE_INFO_2_KHR,
      .pNext = present_mode.has_value() ? &surface_present_mode : nullptr,
      .surface = vk_surface,
  };
  VIZ_VK_FORWARD_ERROR(
      vkGetPhysicalDeviceSurfaceCapabilities2KHR(physical_device_, &surface_info, &surface_caps2));
  const VkSurfaceCapabilitiesKHR& vk_caps = surface_caps2.surfaceCapabilities;

  uint32_t mode_count = 0;
  VIZ_VK_FORWARD_ERROR(vkGetPhysicalDeviceSurfacePresentModesKHR(
      physical_device_, vk_surface, &mode_count, nullptr));
  std::vector<VkPresentModeKHR> vk_modes(mode_count);
  VIZ_VK_FORWARD_ERROR(vkGetPhysicalDeviceSurfacePresentModesKHR(
      physical_device_, vk_surface, &mode_count, vk_modes.data()));

  // Vulkan's "no fixed extent" sentinel is {0xFFFFFFFF, 0xFFFFFFFF}; the module maps it to
  // {0,0} so the sentinel value never leaks past the backend.
  constexpr uint32_t kNoFixedExtent = 0xFFFFFFFFU;

  SurfaceCapabilities caps{};
  uint32_t format_count = 0;
  VIZ_VK_FORWARD_ERROR(
      vkGetPhysicalDeviceSurfaceFormatsKHR(physical_device_, vk_surface, &format_count, nullptr));
  std::vector<VkSurfaceFormatKHR> vk_formats(format_count);
  VIZ_VK_FORWARD_ERROR(vkGetPhysicalDeviceSurfaceFormatsKHR(
      physical_device_, vk_surface, &format_count, vk_formats.data()));
  vk_formats.resize(format_count);
  // The spec requires at least one pair and forbids a VK_FORMAT_UNDEFINED entry (the early-1.0
  // "any format" sentinel no longer exists), so dropping unmodelled pairs cannot empty a valid
  // list by itself.
  for (const VkSurfaceFormatKHR& vk_format : vk_formats) {
    const auto format = to_holoviz_format(vk_format.format);
    const auto color_space = to_holoviz_color_space(vk_format.colorSpace);
    if (format && color_space) {
      caps.formats.push_back(SurfaceFormat{*format, *color_space});
    }
  }
  caps.min_image_count = vk_caps.minImageCount;
  caps.max_image_count = vk_caps.maxImageCount;  // Vulkan uses the same 0 = unlimited convention.
  if (vk_caps.currentExtent.width != kNoFixedExtent) {
    caps.current_extent = {vk_caps.currentExtent.width, vk_caps.currentExtent.height};
  }
  caps.min_image_extent = {vk_caps.minImageExtent.width, vk_caps.minImageExtent.height};
  caps.max_image_extent = {vk_caps.maxImageExtent.width, vk_caps.maxImageExtent.height};

  for (const VkPresentModeKHR vk_mode : vk_modes) {
    const expected<PresentMode> mode = to_holoviz_present_mode(vk_mode);
    if (!mode) {
      continue;  // a mode this module does not model (e.g. shared-demand-refresh)
    }
    // The surface may report FIFO_LATEST_READY even when the device extension/feature could
    // not be enabled; using it would then be invalid, so it is masked out here.
    if (*mode == PresentMode::kFIFOLatestReady && !fifo_latest_ready_enabled_) {
      continue;
    }
    // Same masking for the shared modes against VK_KHR_shared_presentable_image.
    if ((*mode == PresentMode::kSharedDemandRefresh ||
         *mode == PresentMode::kSharedContinuousRefresh) &&
        !shared_present_enabled_) {
      continue;
    }
    caps.present_modes.add(*mode);
  }

#if defined(VK_EXT_present_timing)
  if (present_timing_enabled_ && timing_caps.presentTimingSupported == VK_TRUE) {
    caps.present_stages = to_present_stage_set(timing_caps.presentStageQueries);
    caps.present_timing = !caps.present_stages.empty();
  }
#endif

  return caps;
}

expected<std::vector<SurfaceFormat>> VulkanDevice::surface_formats(
    const detail::SurfaceImpl& surface) const {
  VIZ_UNWRAP_NATIVE(VkSurfaceKHR, vk_surface, surface);

  uint32_t format_count = 0;
  VIZ_VK_FORWARD_ERROR(
      vkGetPhysicalDeviceSurfaceFormatsKHR(physical_device_, vk_surface, &format_count, nullptr));
  std::vector<VkSurfaceFormatKHR> vk_formats(format_count);
  VIZ_VK_FORWARD_ERROR(vkGetPhysicalDeviceSurfaceFormatsKHR(
      physical_device_, vk_surface, &format_count, vk_formats.data()));

  std::vector<SurfaceFormat> formats;
  formats.reserve(format_count);
  for (const VkSurfaceFormatKHR& vk_format : vk_formats) {
    const expected<Format> format = to_holoviz_format(vk_format.format);
    const expected<ColorSpace> color_space = to_holoviz_color_space(vk_format.colorSpace);
    if (!format || !color_space) {
      continue;  // a pairing this module does not model
    }
    formats.push_back(SurfaceFormat{.format = *format, .color_space = *color_space});
  }

  return formats;
}

expected<void> VulkanDevice::wait_for_display_event(const Display& display, DisplayEventType event,
                                                    std::chrono::nanoseconds timeout) {
  if (!display_control_enabled_) {
    return make_unexpected(
        Error{.code = ErrorCode::kNotImplemented,
              .message = "[holoviz] Display events need VK_EXT_display_control (and the "
                         "VK_EXT_display_surface_counter instance extension), which this "
                         "device/driver does not provide; fall back to a time-based wait."});
  }
  if (display.adapter.uuid != adapter_.uuid) {
    return make_unexpected(
        Error{.code = ErrorCode::kInvalidArgument,
              .message = "[holoviz] The display belongs to a different adapter than this "
                         "device; enumerate displays on the device's adapter."});
  }
  VIZ_UNWRAP(NativeObject, native, display.native_object(NativeObjectType::kVkDisplayKHR));
  auto* vk_display = reinterpret_cast<VkDisplayKHR>(native.value);

  VkDisplayEventInfoEXT event_info{.sType = VK_STRUCTURE_TYPE_DISPLAY_EVENT_INFO_EXT};
  switch (event) {
    case DisplayEventType::kFirstPixelOut:
      event_info.displayEvent = VK_DISPLAY_EVENT_TYPE_FIRST_PIXEL_OUT_EXT;
      break;
    default:
      return make_unexpected(
          Error{ErrorCode::kInvalidArgument, "[holoviz] Unknown display event type."});
  }

  VkFence fence = VK_NULL_HANDLE;
  const VkResult registered =
      fn_.vkRegisterDisplayEventEXT(device_, vk_display, &event_info, allocator_, &fence);
  if (registered != VK_SUCCESS) {
    // Some drivers advertise VK_EXT_display_control but reject the registration at runtime
    // (VK_ERROR_UNKNOWN observed on AGX Thor/Jedha, bug 6194895). Report the same
    // "unsupported here" contract as the missing extension so callers degrade uniformly.
    return make_unexpected(
        Error{.code = ErrorCode::kNotImplemented,
              .message = std::string("[holoviz] vkRegisterDisplayEventEXT failed (") +
                         std::to_string(static_cast<int32_t>(registered)) +
                         "); the driver does not deliver display events for this display — "
                         "fall back to a time-based wait."});
  }

  const uint64_t timeout_ns = timeout == std::chrono::nanoseconds::max()
                                  ? UINT64_MAX
                                  : static_cast<uint64_t>(std::max<int64_t>(timeout.count(), 0));
  const VkResult waited = fn_.vkWaitForFences(device_, 1, &fence, VK_TRUE, timeout_ns);
  // A display-event fence is not tied to a queue submission, so it is destroyable after
  // the wait regardless of outcome (unlike submit fences).
  fn_.vkDestroyFence(device_, fence, allocator_);
  if (waited == VK_TIMEOUT) {
    return make_unexpected(Error{.code = ErrorCode::kBackpressured,
                                 .message = "[holoviz] Timed out waiting for the display event."});
  }
  VIZ_VK_FORWARD_ERROR(waited);
  return {};
}

expected<CUdevice> VulkanDevice::cuda_device() const {
  return adapter_.cuda_device_index;
}

expected<CUcontext> VulkanDevice::cuda_context() const {
  return cuda_context_;
}

expected<std::unique_ptr<detail::BufferImpl>> VulkanDevice::create_buffer(
    const BufferDesc& desc) const {
  auto buffer = std::make_unique<VulkanBuffer>();

  VIZ_FORWARD_ERROR(buffer->init(*this, desc));

  return std::move(buffer);
}

expected<std::unique_ptr<detail::ImageImpl>> VulkanDevice::create_image(
    const ImageDesc& desc) const {
  auto image = std::make_unique<VulkanImage>();

  VIZ_FORWARD_ERROR(image->init(*this, desc));

  return std::move(image);
}

expected<std::unique_ptr<detail::RenderPassImpl>> VulkanDevice::create_render_pass(
    const RenderPassDesc& desc) const {
  auto render_pass = std::make_unique<VulkanRenderPass>();

  VIZ_FORWARD_ERROR(render_pass->init(*this, desc));

  return std::move(render_pass);
}

expected<std::unique_ptr<detail::PipelineImpl>> VulkanDevice::create_pipeline(
    const PipelineDesc& desc) const {
  auto pipeline = std::make_unique<VulkanPipeline>();

  VIZ_FORWARD_ERROR(pipeline->init(*this, desc));

  return std::move(pipeline);
}

expected<std::unique_ptr<detail::CommandListImpl>> VulkanDevice::create_command_list(
    const CommandListDesc& desc) const {
  auto command_list = std::make_unique<VulkanCommandList>();

  VIZ_FORWARD_ERROR(command_list->init(*this, desc));

  return std::move(command_list);
}

expected<std::unique_ptr<detail::SwapchainImpl>> VulkanDevice::create_swapchain(
    const SwapchainDesc& desc) const {
  auto swapchain = std::make_unique<VulkanSwapchain>();

  VIZ_FORWARD_ERROR(swapchain->init(*this, desc));

  return std::move(swapchain);
}

expected<std::unique_ptr<detail::QueryPoolImpl>> VulkanDevice::create_query_pool(
    const QueryPoolDesc& desc) const {
  auto query_pool = std::make_unique<VulkanQueryPool>();

  VIZ_FORWARD_ERROR(query_pool->init(*this, desc));

  return std::move(query_pool);
}

expected<std::unique_ptr<detail::ExternalSemaphoreImpl>> VulkanDevice::create_external_semaphore(
    CUstream stream) const {
  auto sync = std::make_unique<VulkanExternalSemaphore>();

  VIZ_FORWARD_ERROR(sync->init(*this, stream));

  return std::move(sync);
}

expected<void> VulkanDevice::upload_to_buffer(const BufferSlice& dst,
                                              std::span<const std::byte> bytes) {
  if (bytes.size() > dst.size_bytes) {
    return make_unexpected(
        Error{.code = ErrorCode::kFailure,
              .message = "[holoviz] Upload size exceeds destination buffer size."});
  }

  if (bytes.size() > staging_buffer_size_) {
    VIZ_FORWARD_ERROR(staging_buffer_min_size(bytes.size()));
  }

  void* mapped_data = nullptr;
  VIZ_VK_FORWARD_ERROR(vmaMapMemory(vma_allocator_, staging_allocation_, &mapped_data));

  std::memcpy(mapped_data, bytes.data(), bytes.size());

  // Flush while the memory is still mapped: vkFlushMappedMemoryRanges requires it.
  VIZ_VK_FORWARD_ERROR(vmaFlushAllocation(vma_allocator_, staging_allocation_, 0, bytes.size()));

  vmaUnmapMemory(vma_allocator_, staging_allocation_);

  VIZ_UNWRAP(VkCommandBuffer, cmd_buffer, start_cmd_buffer_record());

  VIZ_UNWRAP_NATIVE(VkBuffer, dst_buffer, *dst.buffer);

  const VkBufferCopy copy_region{
      .srcOffset = 0,
      .dstOffset = dst.offset_bytes,
      .size = bytes.size(),
  };
  fn_.vkCmdCopyBuffer(cmd_buffer, staging_buffer_, dst_buffer, 1, &copy_region);

  VIZ_FORWARD_ERROR(submit_cmd_buffer(cmd_buffer));

  return {};
}

expected<void> VulkanDevice::download_from_buffer(const BufferSlice& src,
                                                  std::span<std::byte> bytes) {
  if (bytes.size() < src.size_bytes) {
    return make_unexpected(
        Error{.code = ErrorCode::kFailure,
              .message = "[holoviz] Download span capacity is smaller than source buffer size."});
  }

  if (src.size_bytes > staging_buffer_size_) {
    VIZ_FORWARD_ERROR(staging_buffer_min_size(src.size_bytes));
  }

  VIZ_UNWRAP(VkCommandBuffer, cmd_buffer, start_cmd_buffer_record());

  VIZ_UNWRAP_NATIVE(VkBuffer, src_buffer, *src.buffer);

  const VkBufferCopy copy_region{
      .srcOffset = src.offset_bytes,
      .dstOffset = 0,
      .size = src.size_bytes,
  };
  fn_.vkCmdCopyBuffer(cmd_buffer, src_buffer, staging_buffer_, 1, &copy_region);

  VIZ_FORWARD_ERROR(submit_cmd_buffer(cmd_buffer));

  void* mapped_data = nullptr;
  VIZ_VK_FORWARD_ERROR(vmaMapMemory(vma_allocator_, staging_allocation_, &mapped_data));

  // Invalidate while the memory is mapped: vkInvalidateMappedMemoryRanges requires it.
  VIZ_VK_FORWARD_ERROR(
      vmaInvalidateAllocation(vma_allocator_, staging_allocation_, 0, src.size_bytes));

  std::memcpy(bytes.data(), mapped_data, src.size_bytes);

  vmaUnmapMemory(vma_allocator_, staging_allocation_);

  return {};
}

namespace {

// Per-mip copy geometry shared by upload_to_image/download_from_image: the mip level's
// extent, the caller's row pitch converted to VkBufferImageCopy::bufferRowLength (texels;
// 0 = tightly packed), and the byte count the host buffer must provide/receive
// ((height - 1) * pitch + width * texel, per Vulkan's buffer-image addressing).
struct ImageCopyGeometry {
  VkExtent3D extent;
  uint32_t buffer_row_length;
  size_t required_bytes;
};

expected<ImageCopyGeometry> image_copy_geometry(const char* method, const VulkanImage& image,
                                                uint32_t mip, size_t row_pitch_bytes,
                                                size_t bytes_available) {
  const Extent2D full_extent = image.extent();
  const VkExtent3D mip_extent{
      .width = std::max(1U, full_extent.width >> mip),
      .height = std::max(1U, full_extent.height >> mip),
      .depth = 1,
  };
  VIZ_UNWRAP(uint32_t, texel_bytes, vk_format_texel_size_bytes(image.vk_format()));
  const size_t tight_row_bytes = static_cast<size_t>(mip_extent.width) * texel_bytes;
  uint32_t buffer_row_length = 0;
  size_t row_bytes = tight_row_bytes;
  if (row_pitch_bytes != 0) {
    if (row_pitch_bytes % texel_bytes != 0 || row_pitch_bytes < tight_row_bytes) {
      return make_unexpected(
          Error{.code = ErrorCode::kInvalidArgument,
                .message = std::string("[holoviz] ") + method +
                           ": row_pitch_bytes must be a multiple of the texel size and at least "
                           "one tightly packed row"});
    }
    buffer_row_length = static_cast<uint32_t>(row_pitch_bytes / texel_bytes);
    row_bytes = row_pitch_bytes;
  }
  const size_t required_bytes = (mip_extent.height - 1) * row_bytes + tight_row_bytes;
  if (bytes_available < required_bytes) {
    return make_unexpected(Error{.code = ErrorCode::kInvalidArgument,
                                 .message = std::string("[holoviz] ") + method +
                                            ": byte span holds " + std::to_string(bytes_available) +
                                            " bytes but the mip level needs " +
                                            std::to_string(required_bytes)});
  }
  return ImageCopyGeometry{mip_extent, buffer_row_length, required_bytes};
}

}  // namespace

expected<void> VulkanDevice::upload_to_image(const detail::ImageImpl& dst, uint32_t mip,
                                             std::span<const std::byte> bytes,
                                             size_t row_pitch_bytes) {
  if (mip != 0) {
    return make_unexpected(
        Error{.code = ErrorCode::kInvalidArgument,
              .message = std::string("[holoviz] upload_to_image: only mip level 0 is supported")});
  }

  VIZ_UNWRAP_NATIVE(VkImage, dst_image, dst);
  const auto& vulkan_image = dynamic_cast<const VulkanImage&>(dst);
  if (!vulkan_image.owns_image()) {
    return make_unexpected(
        Error{.code = ErrorCode::kInvalidArgument,
              .message = std::string("[holoviz] upload_to_image: destination image must be owned "
                                     "by the device. This error is normally encountered when "
                                     "trying to upload data to an image owned by the swapchain.")});
  }

  VIZ_UNWRAP(
      ImageCopyGeometry,
      geometry,
      image_copy_geometry("upload_to_image", vulkan_image, mip, row_pitch_bytes, bytes.size()));

  if (bytes.size() > staging_buffer_size_) {
    VIZ_FORWARD_ERROR(staging_buffer_min_size(bytes.size()));
  }

  void* mapped_data = nullptr;
  VIZ_VK_FORWARD_ERROR(vmaMapMemory(vma_allocator_, staging_allocation_, &mapped_data));
  std::memcpy(mapped_data, bytes.data(), bytes.size());
  // Flush while the memory is still mapped: vkFlushMappedMemoryRanges requires it.
  VIZ_VK_FORWARD_ERROR(vmaFlushAllocation(vma_allocator_, staging_allocation_, 0, bytes.size()));
  vmaUnmapMemory(vma_allocator_, staging_allocation_);

  VIZ_UNWRAP(VkCommandBuffer, cmd_buffer, start_cmd_buffer_record());

  const VkImageMemoryBarrier barrier{
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
      .srcAccessMask = vulkan_image.pinned_access_flags(),
      .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
      .oldLayout = vulkan_image.pinned_layout(),
      .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image = dst_image,
      .subresourceRange =
          // NOLINTNEXTLINE(whitespace/braces)
      {
          .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
          .baseMipLevel = 0,
          .levelCount = 1,
          .baseArrayLayer = 0,
          .layerCount = 1,
      },
  };

  fn_.vkCmdPipelineBarrier(cmd_buffer,
                           VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                           VK_PIPELINE_STAGE_TRANSFER_BIT,
                           0,
                           0,
                           nullptr,
                           0,
                           nullptr,
                           1,
                           &barrier);

  const VkBufferImageCopy copy_region{
      .bufferOffset = 0,
      .bufferRowLength = geometry.buffer_row_length,
      .bufferImageHeight = 0,
      .imageSubresource =
          // NOLINTNEXTLINE(whitespace/braces)
      {
          .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
          .mipLevel = mip,
          .baseArrayLayer = 0,
          .layerCount = 1,
      },
      .imageOffset = {0, 0, 0},
      .imageExtent = geometry.extent,
  };

  fn_.vkCmdCopyBufferToImage(cmd_buffer,
                             staging_buffer_,
                             dst_image,
                             VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                             1,
                             &copy_region);

  const VkImageMemoryBarrier barrier_2{
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
      .dstAccessMask = vulkan_image.pinned_access_flags(),
      .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
      .newLayout = vulkan_image.pinned_layout(),
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image = dst_image,
      .subresourceRange =
          // NOLINTNEXTLINE(whitespace/braces)
      {
          .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
          .baseMipLevel = 0,
          .levelCount = 1,
          .baseArrayLayer = 0,
          .layerCount = 1,
      },
  };

  fn_.vkCmdPipelineBarrier(cmd_buffer,
                           VK_PIPELINE_STAGE_TRANSFER_BIT,
                           VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                           0,
                           0,
                           nullptr,
                           0,
                           nullptr,
                           1,
                           &barrier_2);
  VIZ_FORWARD_ERROR(submit_cmd_buffer(cmd_buffer));
  return {};
}

expected<void> VulkanDevice::download_from_image(const detail::ImageImpl& src, uint32_t mip,
                                                 std::span<std::byte> bytes,
                                                 size_t row_pitch_bytes) {
  // Mirror of upload_to_image: stage through the internal host-visible buffer and block.

  if (mip != 0) {
    return make_unexpected(Error{
        .code = ErrorCode::kInvalidArgument,
        .message = std::string("[holoviz] download_from_image: only mip level 0 is supported")});
  }

  VIZ_UNWRAP_NATIVE(VkImage, src_image, src);
  const auto& vulkan_image = dynamic_cast<const VulkanImage&>(src);
  if (!vulkan_image.owns_image()) {
    return make_unexpected(
        Error{.code = ErrorCode::kInvalidArgument,
              .message = std::string("[holoviz] download_from_image: source image must be owned "
                                     "by the device. This error is normally encountered when "
                                     "trying to download data from an image owned by the "
                                     "swapchain.")});
  }

  VIZ_UNWRAP(
      ImageCopyGeometry,
      geometry,
      image_copy_geometry("download_from_image", vulkan_image, mip, row_pitch_bytes, bytes.size()));

  if (bytes.size() > staging_buffer_size_) {
    VIZ_FORWARD_ERROR(staging_buffer_min_size(bytes.size()));
  }

  VIZ_UNWRAP(VkCommandBuffer, cmd_buffer, start_cmd_buffer_record());

  const VkImageMemoryBarrier barrier{
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
      .srcAccessMask = vulkan_image.pinned_access_flags(),
      .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
      .oldLayout = vulkan_image.pinned_layout(),
      .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image = src_image,
      .subresourceRange =
          // NOLINTNEXTLINE(whitespace/braces)
      {
          .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
          .baseMipLevel = 0,
          .levelCount = 1,
          .baseArrayLayer = 0,
          .layerCount = 1,
      },
  };

  fn_.vkCmdPipelineBarrier(cmd_buffer,
                           VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                           VK_PIPELINE_STAGE_TRANSFER_BIT,
                           0,
                           0,
                           nullptr,
                           0,
                           nullptr,
                           1,
                           &barrier);

  const VkBufferImageCopy copy_region{
      .bufferOffset = 0,
      .bufferRowLength = geometry.buffer_row_length,
      .bufferImageHeight = 0,
      .imageSubresource =
          // NOLINTNEXTLINE(whitespace/braces)
      {
          .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
          .mipLevel = mip,
          .baseArrayLayer = 0,
          .layerCount = 1,
      },
      .imageOffset = {0, 0, 0},
      .imageExtent = geometry.extent,
  };

  fn_.vkCmdCopyImageToBuffer(cmd_buffer,
                             src_image,
                             VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                             staging_buffer_,
                             1,
                             &copy_region);

  const VkImageMemoryBarrier barrier_2{
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
      .dstAccessMask = vulkan_image.pinned_access_flags(),
      .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
      .newLayout = vulkan_image.pinned_layout(),
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image = src_image,
      .subresourceRange =
          // NOLINTNEXTLINE(whitespace/braces)
      {
          .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
          .baseMipLevel = 0,
          .levelCount = 1,
          .baseArrayLayer = 0,
          .layerCount = 1,
      },
  };

  fn_.vkCmdPipelineBarrier(cmd_buffer,
                           VK_PIPELINE_STAGE_TRANSFER_BIT,
                           VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                           0,
                           0,
                           nullptr,
                           0,
                           nullptr,
                           1,
                           &barrier_2);
  VIZ_FORWARD_ERROR(submit_cmd_buffer(cmd_buffer));

  void* mapped_data = nullptr;
  VIZ_VK_FORWARD_ERROR(vmaMapMemory(vma_allocator_, staging_allocation_, &mapped_data));
  // Invalidate while the memory is still mapped so host reads see the device writes.
  // Only the copy's addressed range is meaningful; bytes beyond it are left untouched.
  VIZ_VK_FORWARD_ERROR(
      vmaInvalidateAllocation(vma_allocator_, staging_allocation_, 0, geometry.required_bytes));
  std::memcpy(bytes.data(), mapped_data, geometry.required_bytes);
  vmaUnmapMemory(vma_allocator_, staging_allocation_);
  return {};
}

expected<NativeObject> VulkanDevice::native_object(NativeObjectType type) const {
  switch (type) {
    case NativeObjectType::kVkInstance:
      return NativeObject{.type = NativeObjectType::kVkInstance,
                          .value = reinterpret_cast<uint64_t>(instance_)};
    case NativeObjectType::kVkPhysicalDevice:
      return NativeObject{.type = NativeObjectType::kVkPhysicalDevice,
                          .value = reinterpret_cast<uint64_t>(physical_device_)};
    case NativeObjectType::kVkDevice:
      return NativeObject{.type = NativeObjectType::kVkDevice,
                          .value = reinterpret_cast<uint64_t>(device_)};
    case NativeObjectType::kVkQueue:
      return NativeObject{.type = NativeObjectType::kVkQueue,
                          .value = reinterpret_cast<uint64_t>(queue_)};
    default:
      return unexpected<Error>(
          Error{.code = ErrorCode::kFailure,
                .message = "[holoviz] Unsupported native object type requested."});
  }
}

expected<VkSampler> VulkanDevice::get_sampler(const SamplerDesc& desc) {
  uint32_t sampler_found = static_cast<uint32_t>(-1);

  for (uint32_t i = 0; i < sampler_descs_.size(); ++i) {
    if (sampler_descs_[i].is_equal(desc)) {
      sampler_found = i;
      break;
    }
  }

  if (sampler_found != static_cast<uint32_t>(-1)) {
    return samplers_[sampler_found];
  }

  VIZ_UNWRAP(VkFilter, mag_filter, from_holoviz_filter(desc.mag_filter));
  VIZ_UNWRAP(VkFilter, min_filter, from_holoviz_filter(desc.min_filter));
  VIZ_UNWRAP(VkSamplerAddressMode, address_mode_u, from_holoviz_address_mode(desc.address_u));
  VIZ_UNWRAP(VkSamplerAddressMode, address_mode_v, from_holoviz_address_mode(desc.address_v));

  const VkSamplerCreateInfo sampler_info{
      .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
      .magFilter = mag_filter,
      .minFilter = min_filter,
      .addressModeU = address_mode_u,
      .addressModeV = address_mode_v,
      .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,  // No 3D images yet anyways
      .mipLodBias = 0.0f,
      .anisotropyEnable = VK_FALSE,
      .maxAnisotropy = 1.0f,
      .compareEnable = VK_FALSE,
      .compareOp = VK_COMPARE_OP_ALWAYS,
      .minLod = 0.0f,
      .maxLod = 0.0f,
      .borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK,
      .unnormalizedCoordinates = VK_FALSE};

  VkSampler sampler{};
  VIZ_VK_DEVICE_FUNC(vkCreateSampler, this, &sampler_info, allocator_, &sampler);

  sampler_descs_.push_back(desc);
  samplers_.push_back(sampler);

  return sampler;
}

expected<void> VulkanDevice::staging_buffer_min_size(size_t size_bytes) {
  if (size_bytes <= staging_buffer_size_) {
    return {};
  }

  if (staging_buffer_ != VK_NULL_HANDLE) {
    vmaDestroyBuffer(vma_allocator_, staging_buffer_, staging_allocation_);
    staging_buffer_ = VK_NULL_HANDLE;
    staging_allocation_ = VK_NULL_HANDLE;
  }
  staging_buffer_size_ = size_bytes;

  const VkBufferCreateInfo buffer_info{
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = staging_buffer_size_,
      .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
  };

  const VmaAllocationCreateInfo allocation_create_info{
      .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT,
      .usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
  };

  VIZ_VK_FORWARD_ERROR(vmaCreateBuffer(vma_allocator_,
                                       &buffer_info,
                                       &allocation_create_info,
                                       &staging_buffer_,
                                       &staging_allocation_,
                                       &staging_allocation_info_));

  return {};
}

expected<VkCommandBuffer> VulkanDevice::start_cmd_buffer_record() const {
  const VkCommandBufferBeginInfo begin_info{
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
  };

  VIZ_VK_FORWARD_ERROR(fn_.vkBeginCommandBuffer(command_buffer_, &begin_info));

  return command_buffer_;
}

expected<void> VulkanDevice::submit_cmd_buffer(VkCommandBuffer cmd_buffer) const {
  VIZ_VK_FORWARD_ERROR(fn_.vkEndCommandBuffer(cmd_buffer));

  const VkSubmitInfo submit_info{
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .commandBufferCount = 1,
      .pCommandBuffers = &cmd_buffer,
  };

  VIZ_VK_FORWARD_ERROR(fn_.vkQueueSubmit(queue_, 1, &submit_info, cmd_fence_));
  VIZ_VK_DEVICE_FUNC(vkWaitForFences, this, 1, &cmd_fence_, VK_TRUE, UINT64_MAX);
  VIZ_VK_DEVICE_FUNC(vkResetFences, this, 1, &cmd_fence_);
  VIZ_VK_DEVICE_FUNC(vkResetCommandPool, this, command_pool_, 0);
  return {};
}

expected<uint64_t> VulkanDevice::submit_semaphore_signal(VkSemaphore semaphore) const {
  const std::array<VkSemaphore, 2> signal_semaphores = {{timeline_semaphore_, semaphore}};
  // Binary semaphore: VkTimelineSemaphoreSubmitInfo ignores the value at that index.
  const std::array<uint64_t, 2> signal_values = {{timeline_value_ + 1, 0}};

  const VkTimelineSemaphoreSubmitInfo timeline_info{
      .sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
      .signalSemaphoreValueCount = 2,
      .pSignalSemaphoreValues = signal_values.data(),
  };

  const VkSubmitInfo submit_info{
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .pNext = &timeline_info,
      .signalSemaphoreCount = 2,
      .pSignalSemaphores = signal_semaphores.data(),
  };

  // Same commit discipline as submit(): timeline_value_ is bumped only after the
  // submit succeeds, so a failure leaves the timeline bookkeeping untouched.
  VIZ_VK_FORWARD_ERROR(fn_.vkQueueSubmit(queue_, 1, &submit_info, VK_NULL_HANDLE));
  timeline_value_++;

  return timeline_value_;
}

expected<uint64_t> VulkanDevice::submit_cmd_buffer_gated(VkCommandBuffer cmd_buffer,
                                                         VkSemaphore wait_semaphore,
                                                         VkPipelineStageFlags wait_stage,
                                                         VkSemaphore resignal_semaphore) const {
  const std::array<VkSemaphore, 2> signal_semaphores = {{timeline_semaphore_, resignal_semaphore}};
  // Binary semaphore: VkTimelineSemaphoreSubmitInfo ignores the value at that index.
  const std::array<uint64_t, 2> signal_values = {{timeline_value_ + 1, 0}};
  const std::array<uint64_t, 1> wait_values = {{0}};

  const VkTimelineSemaphoreSubmitInfo timeline_info{
      .sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
      .waitSemaphoreValueCount = 1,
      .pWaitSemaphoreValues = wait_values.data(),
      .signalSemaphoreValueCount = 2,
      .pSignalSemaphoreValues = signal_values.data(),
  };

  const VkSubmitInfo submit_info{
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .pNext = &timeline_info,
      .waitSemaphoreCount = 1,
      .pWaitSemaphores = &wait_semaphore,
      .pWaitDstStageMask = &wait_stage,
      .commandBufferCount = 1,
      .pCommandBuffers = &cmd_buffer,
      .signalSemaphoreCount = 2,
      .pSignalSemaphores = signal_semaphores.data(),
  };

  // Same commit discipline as submit(): timeline_value_ is bumped only after the
  // submit succeeds, so a failure leaves the timeline bookkeeping untouched.
  VIZ_VK_FORWARD_ERROR(fn_.vkQueueSubmit(queue_, 1, &submit_info, VK_NULL_HANDLE));
  timeline_value_++;

  return timeline_value_;
}

expected<VkDeviceMemory> VulkanDevice::allocate_device_memory(const VkMemoryRequirements& mem_reqs,
                                                              VkMemoryPropertyFlags required_flags,
                                                              void* pNext) const {
  VkMemoryAllocateInfo memoryAllocateInfo;
  memoryAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  memoryAllocateInfo.pNext = pNext;
  memoryAllocateInfo.allocationSize = mem_reqs.size;

  constexpr uint32_t kInvalidMemoryTypeIndex = std::numeric_limits<uint32_t>::max();
  memoryAllocateInfo.memoryTypeIndex = kInvalidMemoryTypeIndex;

  uint32_t memoryTypeBits = mem_reqs.memoryTypeBits;
  for (uint32_t i = 0; i < 32; ++i) {
    const VkMemoryType memoryType = memory_properties_.memoryTypes[i];
    if ((memoryTypeBits & 1) != 0) {
      if ((memoryType.propertyFlags & required_flags) == required_flags) {
        memoryAllocateInfo.memoryTypeIndex = i;
        break;
      }
    }
    memoryTypeBits = memoryTypeBits >> 1;
  }

  if (memoryAllocateInfo.memoryTypeIndex == kInvalidMemoryTypeIndex) {
    return make_unexpected(
        Error{.code = ErrorCode::kFailure,
              .message = "[holoviz] Failed to find suitable Vulkan memory type."});
  }

  VkDeviceMemory memory{};
  VIZ_VK_DEVICE_FUNC(vkAllocateMemory, this, &memoryAllocateInfo, allocator_, &memory);
  return memory;
}

VulkanDevice::~VulkanDevice() {
  if (expected<void> result = destroy(detail::kDestructorDestroyTimeout); !result) {
    try {
      HOLOVIZ_LOG_ERROR("[holoviz] VulkanDevice: destroy failed in destructor: {}",
                        result.error().message);
      // Logging can throw during teardown; cleanup must continue without escaping.
      // NOLINTNEXTLINE(bugprone-empty-catch)
    } catch (...) {
    }
  }
}

// Keep the outstanding-child proof before native device and context release.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
expected<void> VulkanDevice::destroy(std::chrono::nanoseconds timeout) {
  expected<void> result;

  // Bounded drain before any teardown. timeline_value_ is bumped by every successful
  // submit() (command lists AND gate-only submits) and by submit_semaphore_signal()
  // (present-wait-gate signals), and signaled on completion, so waiting it out retires
  // every device-issued queue submit. Swapchain-internal drain submits are fence-bounded
  // and drained by VulkanSwapchain::destroy(), which the children-before-device contract
  // orders ahead of this call. On timeout return kBackpressured with NOTHING destroyed:
  // the handle layer keeps the impl alive on kBackpressured, so the caller can
  // wait_idle() and retry. The drain must precede even the CUDA external-semaphore
  // teardown below — that semaphore is imported from timeline_semaphore_ and must not
  // be released while the timeline is live.
  if (device_ != VK_NULL_HANDLE && timeline_semaphore_ != VK_NULL_HANDLE) {
    if (timeline_value_ > 0) {
      expected<void> drained =
          wait_on_submit(SubmitSyncToken{.timeline_value = timeline_value_}, timeout);
      if (!drained) {
        if (drained.error().code == ErrorCode::kBackpressured) {
          return drained;  // retryable: no state was torn down
        }
        // Device-lost class of failure: record it and fall through to teardown —
        // refusing here would leak everything.
        result = std::move(drained);
      }
    }
    // Debug-only misuse check: children must be destroyed before the device. Log and
    // proceed — refusing would turn a diagnostic into a behavior change (and leak the
    // whole VkDevice from destructor-path teardown). Using those children afterwards
    // remains undefined behavior.
    if (validation_enabled()) {
      if (const int live = live_child_count(); live != 0) {
        HOLOVIZ_LOG_ERROR(
            "[holoviz] VulkanDevice::destroy(): {} child object(s) (buffers, images, "
            "swapchains, ...) are still alive; destroy all children before the device. "
            "Proceeding — any later use of those children is undefined.",
            live);
      }
    }
  }

  // Terminal poison (report_leaked_resources()): a child's teardown gave up on wedged
  // pending work — a swapchain destructor on a wedged presentation engine, or a render
  // backend destructor whose bounded drain timed out — and leaked/quarantined objects
  // that still reference this VkDevice, with work possibly still pending on the queue or
  // a producer stream — work the timeline drain above deliberately does not cover.
  // Destroying the device (or the CiG context wrapping its queue) under live children
  // would be undefined behavior, and picking apart which sibling objects are still safe
  // under a wedged driver buys nothing in terminal teardown: leak all remaining native
  // state instead. Handles are nulled so a repeated destroy()/destructor call is a no-op.
  if (resources_leaked() && device_ != VK_NULL_HANDLE) {
    try {
      HOLOVIZ_LOG_ERROR(
          "[holoviz] VulkanDevice::destroy(): leaked resources (WSI objects from a wedged "
          "presentation engine, or quarantined render-backend state from a timed-out "
          "teardown) still reference this device; leaking the native device and its "
          "resources instead of destroying them.");
      // Logging can throw during teardown; cleanup must continue without escaping.
      // NOLINTNEXTLINE(bugprone-empty-catch)
    } catch (...) {
    }
    timeline_external_semaphore_ = nullptr;
    staging_buffer_ = VK_NULL_HANDLE;
    staging_allocation_ = VK_NULL_HANDLE;
    vma_allocator_ = VK_NULL_HANDLE;
    cmd_fence_ = VK_NULL_HANDLE;
    command_pool_ = VK_NULL_HANDLE;
    timeline_semaphore_ = VK_NULL_HANDLE;
    for (auto& sampler : samplers_) {
      sampler = VK_NULL_HANDLE;
    }
    cuda_context_ = nullptr;
    owns_cuda_context_ = false;
    cuda_context_is_cig_ = false;
    external_compute_queue_ = VK_NULL_HANDLE;
    device_ = VK_NULL_HANDLE;
    return result;
  }

  if (timeline_external_semaphore_ != nullptr) {
    // Destroy the import under the device's own context — teardown may run with no
    // context current. Scoped so the pop happens before cuCtxDestroy below (a CiG
    // context must not be destroyed while pushed).
    const ScopedCudaContext cuda_scope(cuda_context_);
    VIZ_CU_ACCUMULATE_ERROR(result, cuDestroyExternalSemaphore(timeline_external_semaphore_));
    timeline_external_semaphore_ = nullptr;
  }

  // fn_ is only populated once vkCreateDevice has succeeded; a device that
  // failed init partway must not call through the empty function table.
  if (device_ != VK_NULL_HANDLE) {
    if (staging_buffer_ != VK_NULL_HANDLE) {
      vmaDestroyBuffer(vma_allocator_, staging_buffer_, staging_allocation_);
      staging_buffer_ = VK_NULL_HANDLE;
      staging_allocation_ = VK_NULL_HANDLE;
    }

    if (vma_allocator_ != VK_NULL_HANDLE) {
      vmaDestroyAllocator(vma_allocator_);
      vma_allocator_ = VK_NULL_HANDLE;
    }

    if (cmd_fence_ != VK_NULL_HANDLE) {
      VIZ_VK_DEVICE_FUNC_NO_ERROR(vkDestroyFence, this, cmd_fence_, allocator_);
      cmd_fence_ = VK_NULL_HANDLE;
    }

    if (command_pool_ != VK_NULL_HANDLE) {
      VIZ_VK_DEVICE_FUNC_NO_ERROR(vkDestroyCommandPool, this, command_pool_, allocator_);
      command_pool_ = VK_NULL_HANDLE;
    }

    if (timeline_semaphore_ != VK_NULL_HANDLE) {
      VIZ_VK_DEVICE_FUNC_NO_ERROR(vkDestroySemaphore, this, timeline_semaphore_, allocator_);
      timeline_semaphore_ = VK_NULL_HANDLE;
    }

    for (auto& sampler : samplers_) {
      if (sampler != VK_NULL_HANDLE) {
        VIZ_VK_DEVICE_FUNC_NO_ERROR(vkDestroySampler, this, sampler, allocator_);
        sampler = VK_NULL_HANDLE;
      }
    }

    // A CiG context wraps the graphics queue, so unlike the primary context below
    // it must be torn down before the external compute queue and device it came from.
    if (cuda_context_is_cig_) {
      VIZ_CU_ACCUMULATE_ERROR(result, cuCtxDestroy(cuda_context_));
      cuda_context_ = nullptr;
      owns_cuda_context_ = false;
      cuda_context_is_cig_ = false;
    }

    if (external_compute_queue_ != VK_NULL_HANDLE) {
      VIZ_VK_DEVICE_FUNC_NO_ERROR(
          vkDestroyExternalComputeQueueNV, this, external_compute_queue_, allocator_);
      external_compute_queue_ = VK_NULL_HANDLE;
    }

    VIZ_VK_DEVICE_FUNC_NO_ERROR(vkDestroyDevice, this, allocator_);
    device_ = VK_NULL_HANDLE;
  }

  // Released last so CUDA objects above are destroyed while the context is alive.
  // The context came from cuDevicePrimaryCtxRetain, so release it — never destroy
  // the primary context out from under the rest of the process.
  if (owns_cuda_context_) {
    VIZ_CU_ACCUMULATE_ERROR(result, cuDevicePrimaryCtxRelease(cuda_device_));
    owns_cuda_context_ = false;
    cuda_context_ = nullptr;
  }

  return result;
}

}  // namespace holoviz::vulkan
