/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Surface: a presentable target backed by a graphics API surface object (e.g.
// VkSurfaceKHR). Surfaces are created from a Window (Window::create_surface())
// or from a Display output (Display::create_surface(), holoviz/display.hpp), and
// passed to SwapchainDesc. The surface must outlive any swapchain created
// from it.

#pragma once

#include <chrono>
#include <memory>
#include <vector>

#include <holoviz/holoviz_export.h>

#include "common.hpp"

namespace holoviz {

namespace detail {
class SurfaceImpl;
}

/// Set of PresentMode values a surface supports (bit = 1u << static_cast<uint8_t>(mode)).
/// kAuto is a request, not a mode, and is never set.
struct PresentModeSet {
  uint32_t mask = 0;

  [[nodiscard]] bool contains(PresentMode mode) const {
    return (mask & (1U << static_cast<uint8_t>(mode))) != 0;
  }
  void add(PresentMode mode) { mask |= (1U << static_cast<uint8_t>(mode)); }
};

/// Points in a present's life the presentation engine can timestamp (VK_EXT_present_timing
/// present stages). Which ones a surface reports is per (device, surface):
/// SurfaceCapabilities::present_stages.
enum class PresentStage : uint8_t {
  kQueueOperationsEnd,  ///< The present's queue work (wait semaphores) completed.
  kRequestDequeued,     ///< The presentation engine took the request off its queue.
  kFirstPixelOut,       ///< The first pixel of the image left the display engine (scanout).
  kFirstPixelVisible,   ///< The first pixel became visible on the display panel.
};

/// Set of PresentStage values (bit = 1u << static_cast<uint8_t>(stage)).
struct PresentStageSet {
  uint32_t mask = 0;

  [[nodiscard]] bool contains(PresentStage stage) const {
    return (mask & (1U << static_cast<uint8_t>(stage))) != 0;
  }
  void add(PresentStage stage) { mask |= (1U << static_cast<uint8_t>(stage)); }
  [[nodiscard]] bool empty() const { return mask == 0; }
};

/// An image format and presentation color space supported together by a surface.
struct SurfaceFormat {
  Format format = Format::kUnknown;
  ColorSpace color_space = ColorSpace::kSrgbNonlinear;
};

/// Snapshot of what a (device, surface) pair supports, from
/// Device::surface_capabilities(). Swapchain creation validates against these; a
/// PresentBackend queries them up front to resolve PresentMode::kAuto and clamp its
/// requested image count.
struct SurfaceCapabilities {
  uint32_t min_image_count = 0;
  uint32_t max_image_count = 0;  ///< 0 = no upper limit
  /// The surface's current size; {0,0} when the surface defers to the swapchain's choice
  /// (some window systems report no fixed extent).
  Extent2D current_extent{};
  Extent2D min_image_extent{};
  Extent2D max_image_extent{};
  /// Modes reported by the surface, masked to what the device actually enabled (a mode
  /// whose device extension is unavailable is not listed even if the surface offers it).
  PresentModeSet present_modes;
  /// Supported pairs representable by this API; unrecognized native formats/color spaces
  /// are omitted. An empty list does not imply that the surface supports no native formats.
  std::vector<SurfaceFormat> formats;
  /// SwapchainDesc::enable_present_timing is accepted on this surface. Requires
  /// CapabilityReport::present_timing as well; false when the device lacks it.
  bool present_timing = false;
  /// Stages the presentation engine timestamps on this surface; empty unless
  /// present_timing.
  PresentStageSet present_stages;
};

/// A graphics API presentable surface. Move-only; not thread-safe.
class Surface {
 public:
  HOLOVIZ_EXPORT explicit Surface(std::unique_ptr<detail::SurfaceImpl> impl);
  HOLOVIZ_EXPORT Surface();
  HOLOVIZ_EXPORT Surface(Surface&&) noexcept;
  HOLOVIZ_EXPORT Surface& operator=(Surface&&) noexcept;
  Surface(const Surface&) = delete;
  Surface& operator=(const Surface&) = delete;
  HOLOVIZ_EXPORT ~Surface();

  explicit operator bool() const { return impl_ != nullptr; }

  /// Release the underlying backend resources now instead of at destruction.
  /// \p timeout bounds how long to wait for the object to stop being in use:
  /// zero (the default) fails immediately with ErrorCode::kBackpressured if
  /// the object is in use; a non-zero timeout waits up to that duration before
  /// returning ErrorCode::kBackpressured. Only CommandList currently tracks
  /// in-use state; other objects destroy immediately. On success (and on any
  /// error other than kBackpressured) the handle becomes empty. Safe to call
  /// on an empty handle (no-op success). The destructor calls this
  /// automatically and logs any error.
  [[nodiscard]] HOLOVIZ_EXPORT expected<void> destroy(
      std::chrono::nanoseconds timeout = std::chrono::nanoseconds::zero());

  /// Return the presentable area in pixels (what swapchains are sized to).
  [[nodiscard]] HOLOVIZ_EXPORT expected<Extent2D> extent() const;

  /// Whether the presentable area can change at runtime (the surface came from a resizable
  /// window). Swapchains created on a non-resizable surface skip all per-frame resize
  /// detection — a fixed-size window or a direct-to-display output pays nothing for the
  /// automatic-resize machinery.
  [[nodiscard]] HOLOVIZ_EXPORT expected<bool> resizable() const;

  /// Vulkan: answers kVkSurfaceKHR.
  [[nodiscard]] HOLOVIZ_EXPORT expected<NativeObject> native_object(NativeObjectType type) const;

  /// Backend-internal access; not part of the stable API surface.
  /// Precondition: the handle is non-empty.
  detail::SurfaceImpl& impl() { return *impl_; }
  const detail::SurfaceImpl& impl() const { return *impl_; }

 private:
  std::unique_ptr<detail::SurfaceImpl> impl_;
};

}  // namespace holoviz
