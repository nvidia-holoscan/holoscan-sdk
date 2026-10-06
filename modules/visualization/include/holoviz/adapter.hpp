/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <holoviz/holoviz_export.h>

#include <array>
#include <vector>

#include "common.hpp"
#include "device.hpp"
#include "holoviz/expected.hpp"

namespace holoviz {

class Surface;

enum class CudaContextMode {
  kExternal,      // Default mode: app can supply context, HoloViz will create one if needed
  kCigRequired,   // HoloViz creates a CiG context; fail if unsupported
  kCigPreferred,  // try CiG, fall back to kExternal
};

struct DeviceDesc {
  Backend backend = Backend::kAuto;
  DevicePolicy policy = DevicePolicy::kDynamic;

  CudaContextMode cuda_context_mode = CudaContextMode::kExternal;
  CUdevice cuda_device = -1;  // -1 means "auto pick"
  /// kExternal: explicit handle; nullptr = retain the device's primary context.
  /// Currency contract: device creation makes this context current on the calling
  /// thread (cuCtxSetCurrent, no stack push) and expects it current on any thread
  /// that creates HoloViz resources; teardown paths push/pop it internally.
  CUcontext cuda_context = nullptr;
  uint32_t reserved_external_queues = 1;  // CiG modes only
};

/// A physical GPU, described as a plain value type so adapter lists can be
/// copied and filtered freely. The UUID is the cross-API join key: it matches
/// what CUDA reports via cuDeviceGetUuid, which is how a CUDA ordinal and an
/// HoloViz adapter are identified as the same silicon (UUID-first selection).
struct Adapter {
  std::array<char, 256> name{};  ///< NUL-terminated driver device name.
  std::array<uint8_t, 16> uuid{};
  Backend backend = Backend::kAuto;
  size_t device_memory_bytes = 0;
  int cuda_device_index = -1;  ///< -1 if no CUDA device matches this adapter.

  /// Return whether this adapter can present to the given surface — specifically, whether a
  /// queue family supports both graphics and presentation to it, since a Device created here
  /// renders and presents on one queue. Call during setup, not per frame.
  [[nodiscard]] HOLOVIZ_EXPORT expected<bool> supports_surface(const Surface& surface) const;
  [[nodiscard]] HOLOVIZ_EXPORT expected<NativeObject> native_object(NativeObjectType type) const;

  /// Create a logical Device on this adapter. The Device owns the single
  /// graphics queue; there is no separate queue object (see device.hpp).
  [[nodiscard]] HOLOVIZ_EXPORT expected<Device> create_device(const DeviceDesc& desc) const;
};

/// Enumerate the adapters visible to a backend. First call initializes the
/// backend instance (and the windowing layer it needs for surface support),
/// which then lives for the remainder of the process. First-call backend
/// initialization is thread-safe — concurrent first calls serialize, and an
/// initialization failure is sticky for the process. (Per-Device threading
/// contracts are unchanged; see device.hpp.)
[[nodiscard]] HOLOVIZ_EXPORT expected<std::vector<Adapter>> enumerate_adapters(
    Backend backend = Backend::kAuto);

/// Convenience: create a Device on the first enumerated adapter that can
/// present to `surface`; the Device's queue is guaranteed to both render and
/// present to it. Prefer explicit UUID-based selection when interop with a
/// specific CUDA device matters.
[[nodiscard]] HOLOVIZ_EXPORT expected<Device> create_device_for_surface(const DeviceDesc& desc,
                                                                        const Surface& surface);

/// Per-format usability flags. The Format enum says what the API can spell;
/// this report says what the adapter actually supports for each use.
enum class FormatCapability : uint32_t {
  kNone = 0,
  kSampled = 1u << 0,
  kRenderTarget = 1u << 1,
  kStorage = 1u << 2,
  kBlendable = 1u << 3,
  kCudaImportableImage = 1u << 4,
};
constexpr FormatCapability operator|(FormatCapability a, FormatCapability b) {
  return static_cast<FormatCapability>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}
constexpr bool has_flag(FormatCapability v, FormatCapability f) {
  return (static_cast<uint32_t>(v) & static_cast<uint32_t>(f)) != 0;
}

/// Snapshot of adapter limits and optional features, queried once so callers
/// can branch on capabilities up front instead of handling per-call failures.
/// (Format is defined in common.hpp; this report is the runtime truth.)
struct CapabilityReport {
  Backend backend{};
  Adapter adapter{};

  uint32_t max_image_extent = 0;
  uint32_t max_push_data_bytes = 0;  ///< >= 128.
  uint32_t max_bind_groups = 0;      ///< >= 4.
  size_t storage_buffer_alignment = 0;

  bool present_completion = false;     ///< Present ids + present-wait (VK_KHR_present_id /
                                       ///< VK_KHR_present_wait): Swapchain::wait_for_present().
  bool present_timing = false;         ///< Measured present-stage timing (VK_EXT_present_timing):
                                       ///< SwapchainDesc::enable_present_timing. Implies
                                       ///< present_completion; per surface see
                                       ///< SurfaceCapabilities::present_timing.
  bool calibrated_timestamps = false;  ///< Device::calibrated_timestamps() available
                                       ///< (VK_KHR_calibrated_timestamps or its EXT spelling,
                                       ///< with the device and host monotonic domains).

  bool exclusive_display = false;  ///< Direct-to-display (compositor bypass) path available on
                                   ///< this device/driver; see holoviz/display.hpp. Whether a given
                                   ///< display is currently free is per-display (DisplayOwnership).
  bool display_events = false;     ///< Display-event waits (Device::wait_for_display_event()) and
                                   ///< swapchain vblank counters available
                                   ///< (VK_EXT_display_control + VK_EXT_display_surface_counter).
  bool shared_present = false;     ///< Shared presentable images available
                                ///< (VK_KHR_shared_presentable_image): the kSharedDemandRefresh /
                                ///< kSharedContinuousRefresh present modes.
  bool fill_rectangle = false;  ///< PolygonMode::kFillRectangle available (VK_NV_fill_rectangle).

  /// Indexed by static_cast<uint32_t>(Format); sized with headroom for appends.
  static constexpr size_t kFormatCapacity = 32;
  static_assert(static_cast<size_t>(kLastFormat) < kFormatCapacity,
                "Format outgrew CapabilityReport::format_caps");
  std::array<FormatCapability, kFormatCapacity> format_caps{};
  /// Compatible attachment views of swapchain images (e.g. sRGB views of UNORM scanout).
  bool mutable_swapchain_format = false;

  FormatCapability format_capabilities(Format format) const {
    const auto index = static_cast<uint32_t>(format);
    return index < format_caps.size() ? format_caps[index] : FormatCapability::kNone;
  }
};

}  // namespace holoviz
