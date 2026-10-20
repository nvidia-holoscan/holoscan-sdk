/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <chrono>

#include <holoviz/surface.hpp>

namespace holoviz::detail {

/// Backend interface for Surface; see the Surface handle for API documentation.
class SurfaceImpl {
 public:
  SurfaceImpl(const SurfaceImpl&) = delete;
  SurfaceImpl& operator=(const SurfaceImpl&) = delete;
  SurfaceImpl(SurfaceImpl&&) = delete;
  SurfaceImpl& operator=(SurfaceImpl&&) = delete;
  virtual ~SurfaceImpl() = default;

  /// Release all owned backend resources now. \p timeout bounds how long to
  /// wait for the object to stop being in use: zero (the default) fails
  /// immediately with ErrorCode::kBackpressured if the object is in use, and a
  /// non-zero timeout waits up to that duration before returning
  /// ErrorCode::kBackpressured. Backends without in-use tracking ignore the
  /// timeout and destroy immediately. Idempotent: repeated calls are no-ops
  /// returning success. Called by the destructor if not already destroyed.
  [[nodiscard]] virtual expected<void> destroy(std::chrono::nanoseconds timeout) = 0;

  virtual Extent2D extent() const = 0;

  /// Whether this surface's presentable area can change at runtime (a resizable window, or a
  /// fullscreen window whose video mode can change). True opts the swapchain into automatic
  /// recreation on resize/out-of-date (one surface-capabilities query per frame). Swapchains
  /// skip all of that for non-resizable surfaces — a fixed-size window or a direct-to-display
  /// output pays nothing for the machinery. Defaults to false; window backends report it.
  virtual bool resizable() const { return false; }

  /// How the surface's display is acquired for exclusive scanout. Defaults to kNone; the
  /// direct-display backend reports its acquisition policy.
  virtual DisplayAcquisitionMode display_acquisition_mode() const {
    return DisplayAcquisitionMode::kNone;
  }

  [[nodiscard]] virtual expected<NativeObject> native_object(NativeObjectType type) const = 0;

 protected:
  SurfaceImpl() = default;
};

}  // namespace holoviz::detail
