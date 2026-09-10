// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Factory for the WSI (window-system-integration) PresentBackend: the graphics-substrate
// implementation of the holoviz::PresentBackend seam, built on rhi::Swapchain and its
// acquire/present gates. A cuDisp-backed CUDA output would be a sibling implementation of
// the same seam, which is why nothing here names a graphics API.

#pragma once

#include <memory>

#include <holoviz/holoviz_export.h>

#include <holoviz/expected.hpp>
#include <holoviz/present_backend.hpp>
#include <holoviz/rhi/device.hpp>
#include <holoviz/rhi/window.hpp>

namespace holoviz::backends {

/// Wrap a window's presentable surface into the type-erased SurfaceHandle the seam carries
/// (SwapchainDesc::surface). The window must outlive every copy of the returned handle, and
/// the handle must outlive any swapchain created from it.
[[nodiscard]] HOLOVIZ_EXPORT expected<SurfaceHandle> make_surface_handle(const rhi::Window& window);

/// Create the WSI present backend on `device`. The device is borrowed and must outlive the
/// returned backend; destroy the backend (which tears down its swapchain) before the device
/// and before the window whose surface it presents to.
[[nodiscard]] HOLOVIZ_EXPORT expected<std::unique_ptr<PresentBackend>> create_wsi_present_backend(
    rhi::Device& device);

}  // namespace holoviz::backends
