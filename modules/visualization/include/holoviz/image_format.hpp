// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace holoviz {

/// Pixel formats retained for the visualization module.
///
/// This is an intentionally small starting set covering the formats a survey of real
/// applications showed are actually used (color RGB/RGBA, YUV 4:2:0/4:2:2 camera input, and
/// a single-channel format for LUT/segmentation-mask layers). Add formats here only once the
/// full trimmed format list is agreed, not speculatively.

enum class ImageFormat : uint32_t {
  kUnknown,        ///< uninitialized; never used in a swapchain
  kR8Unorm,        ///< single-channel 8-bit, e.g. segmentation mask input to a LUT layer
  kR8G8B8Unorm,    ///< 24-bit RGB
  kR8G8B8A8Unorm,  ///< 32-bit RGBA
  kNV12,           ///< YUV 4:2:0, 2-plane (Y plane + interleaved UV plane)
  kYUYV422,        ///< YUV 4:2:2, packed (endoscopy/surgical camera input)
  kB8G8R8A8Unorm,  ///< 32-bit BGRA. Swapchain/render-target format (the default WSI
                   ///< swapchain format), reported via SwapchainInfo::format and accepted
                   ///< by RenderBackend::build(); not a source-image upload format.
};

/// YUV → RGB color model conversion; applies to YUV \p ImageFormat values only.
enum class YUVModelConversion : uint8_t {
  kBT601,
  kBT709,
  kBT2020,
};

/// YUV sample range; applies to YUV \p ImageFormat values only.
enum class YUVRange : uint8_t {
  kLimited,
  kFull,
};

}  // namespace holoviz
