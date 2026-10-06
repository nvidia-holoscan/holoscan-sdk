// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// DisplaySurface: the direct-to-display (compositor-bypass) implementation of the
// PresentBackend seam, built on a holoviz::Display surface and holoviz::Swapchain. Same protocol as
// WindowSurface; different surface policy (fixed-size surface, two-image default).

#pragma once

#include <cstdint>
#include <memory>

#include <holoviz/device.hpp>
#include <holoviz/display.hpp>
#include <holoviz/expected.hpp>

#include "holoviz_op/backends/present_backend.hpp"
#include "holoviz_op/backends/swapchain_present_backend.hpp"
#include "holoviz_op/types.hpp"

namespace holoscan::visualization::backends {

/// Acquire \p display for exclusive scanout and wrap its presentable surface into the
/// type-erased SurfaceHandle the seam carries (SwapchainDesc::surface). Destroying the last
/// copy of the handle releases the display. Fails with \c kInvalidState when the display is
/// owned by the desktop / window system (\c holoviz::DisplayOwnership::kCompositor).
[[nodiscard]] expected<SurfaceHandle> make_display_surface_handle(
    const holoviz::Display& display, const holoviz::DisplaySurfaceDesc& desc = {});

/// Create the exclusive-display present backend on \p device (borrowed; must outlive the
/// returned backend). Destroy the backend before the device and before the surface handle it
/// presents to.
[[nodiscard]] expected<std::unique_ptr<PresentBackend>> create_display_surface(
    holoviz::Device& device);

/// The exclusive-display implementation of the PresentBackend seam: the swapchain-driven
/// base with direct-display policy. The surface is fixed-size (the chosen display mode), so
/// resize() is rejected and SwapchainDesc::preferred_extent is ignored; the image count
/// defaults to 2 for latency.
class DisplaySurface final : public SwapchainPresentBackend {
 public:
  explicit DisplaySurface(holoviz::Device& device) : SwapchainPresentBackend(device) {}

  [[nodiscard]] expected<void> resize(Extent2D new_extent) override;

 protected:
  [[nodiscard]] uint32_t default_image_count() const override { return 2; }
  [[nodiscard]] const char* backend_name() const override { return "display_surface"; }
};

}  // namespace holoscan::visualization::backends
