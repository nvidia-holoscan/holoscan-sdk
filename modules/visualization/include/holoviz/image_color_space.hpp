// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace holoviz {

/// How source layer image data is interpreted when composited/presented on screen.
/// Distinct from rhi::ColorSpace (<holoviz/rhi/common.hpp>), which is the
/// swapchain surface color space.
enum class ImageColorSpace : uint8_t {
  kPassThrough,    ///< color components are used "as is"
  kSrgbNonlinear,  ///< sRGB color space
  kBt709Linear,    ///< BT709 color space, displayed using a linear EOTF
};

}  // namespace holoviz
