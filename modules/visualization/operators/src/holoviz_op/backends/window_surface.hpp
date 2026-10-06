// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// WindowSurface: the window-surface implementation of the PresentBackend seam, built on
// holoviz::Swapchain and its acquire/present gates. DisplaySurface (exclusive display) is a
// sibling implementation of the same seam, and a cuDisp-backed CUDA output would be another,
// which is why nothing here names a graphics API.

#pragma once

#include <cstdint>
#include <memory>

#include <holoviz/device.hpp>
#include <holoviz/expected.hpp>
#include <holoviz/window.hpp>

#include "holoviz_op/backends/present_backend.hpp"
#include "holoviz_op/backends/swapchain_present_backend.hpp"
#include "holoviz_op/types.hpp"

namespace holoscan::visualization::backends {

/// Wrap a window's presentable surface into the type-erased SurfaceHandle the seam carries
/// (SwapchainDesc::surface). The window must outlive every copy of the returned handle, and
/// the handle must outlive any swapchain created from it.
[[nodiscard]] expected<SurfaceHandle> make_window_surface_handle(const holoviz::Window& window);

/// Create the window-surface present backend on `device`. The device is borrowed and must outlive
/// the returned backend; destroy the backend (which tears down its swapchain) before the device and
/// before the window whose surface it presents to.
[[nodiscard]] expected<std::unique_ptr<PresentBackend>> create_window_surface(
    holoviz::Device& device);

/// The window-surface implementation of the PresentBackend seam: the swapchain-driven base
/// with windowed policy — a three-image default and no opt-in swapchain features (the window
/// system governs the surface extent; see SwapchainPresentBackend for the protocol).
class WindowSurface final : public SwapchainPresentBackend {
 public:
  explicit WindowSurface(holoviz::Device& device) : SwapchainPresentBackend(device) {}

 protected:
  [[nodiscard]] uint32_t default_image_count() const override { return 3; }
  [[nodiscard]] const char* backend_name() const override { return "window_surface"; }
};

}  // namespace holoscan::visualization::backends
