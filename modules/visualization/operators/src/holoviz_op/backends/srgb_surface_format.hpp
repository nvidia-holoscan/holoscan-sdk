// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <span>

#include <holoviz/expected.hpp>
#include <holoviz/surface.hpp>

namespace holoscan::visualization::backends {

/// Prefer native sRGB scanout; otherwise allow UNORM storage only when a mutable swapchain
/// can provide an sRGB attachment view. Never substitute a different presentation color space.
inline holoviz::expected<holoviz::Format> resolve_srgb_surface_format(
    std::span<const holoviz::SurfaceFormat> formats, bool mutable_format = false) {
  for (const auto candidate : {holoviz::Format::kB8G8R8A8Srgb,
                               holoviz::Format::kR8G8B8A8Srgb,
                               holoviz::Format::kB8G8R8A8Unorm,
                               holoviz::Format::kR8G8B8A8Unorm}) {
    if (!mutable_format && (candidate == holoviz::Format::kB8G8R8A8Unorm ||
                            candidate == holoviz::Format::kR8G8B8A8Unorm)) {
      continue;
    }
    if (std::any_of(formats.begin(), formats.end(), [&](const auto& pair) {
          return pair.format == candidate &&
                 pair.color_space == holoviz::ColorSpace::kSrgbNonlinear;
        })) {
      return candidate;
    }
  }
  return holoviz::make_unexpected(
      holoviz::Error{.code = holoviz::ErrorCode::kNotImplemented,
                     .message = "[holoviz] framebuffer_srgb requires RGBA8/BGRA8 with sRGB "
                                "presentation and either native sRGB scanout or mutable "
                                "swapchain support for an sRGB attachment view."});
}

/// Pick the UNORM scanout format for the default (non-sRGB) framebuffer: BGRA8, then RGBA8,
/// both with sRGB-nonlinear presentation. Never substitute a different presentation color space.
inline holoviz::expected<holoviz::Format> resolve_unorm_surface_format(
    std::span<const holoviz::SurfaceFormat> formats) {
  for (const auto candidate : {holoviz::Format::kB8G8R8A8Unorm, holoviz::Format::kR8G8B8A8Unorm}) {
    if (std::any_of(formats.begin(), formats.end(), [&](const auto& pair) {
          return pair.format == candidate &&
                 pair.color_space == holoviz::ColorSpace::kSrgbNonlinear;
        })) {
      return candidate;
    }
  }
  return holoviz::make_unexpected(
      holoviz::Error{.code = holoviz::ErrorCode::kNotImplemented,
                     .message = "[holoviz] The surface offers neither BGRA8 nor RGBA8 UNORM with "
                                "sRGB presentation; no other swapchain format is supported."});
}

}  // namespace holoscan::visualization::backends
