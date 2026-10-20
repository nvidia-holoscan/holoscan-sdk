// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// The pixel-format vocabulary of the visualization operators: HolovizOp's image layers and sink
// descriptors (<holoscan/visualization/holoviz_op.hpp>) declare their formats with it.

#pragma once

#include <cstdint>

namespace holoscan::visualization {

/// Pixel formats accepted by image layers and named by sink descriptors. Values are append-only.
enum class ImageFormat : uint32_t {
  kUnknown,
  kR8Unorm,            ///< single-channel 8-bit (e.g. segmentation mask for a LUT layer)
  kR8G8B8Unorm,        ///< 24-bit RGB
  kR8G8B8A8Unorm,      ///< 32-bit RGBA
  kNV12,               ///< YUV 4:2:0, 2-plane
  kYUYV422,            ///< YUV 4:2:2, packed
  kB8G8R8A8Unorm,      ///< 32-bit BGRA (default window swapchain format)
  kR16Unorm,           ///< single-channel 16-bit grayscale, 0..65535 maps to black..white
  kR16G16B16A16Unorm,  ///< 64-bit RGBA, 16 bits per channel
  kR8G8B8A8Srgb,       ///< sRGB RGBA output; image sources use UNORM + ColorSpace instead
  kB8G8R8A8Srgb,       ///< sRGB BGRA output; image sources use UNORM + ColorSpace instead
};

}  // namespace holoscan::visualization
