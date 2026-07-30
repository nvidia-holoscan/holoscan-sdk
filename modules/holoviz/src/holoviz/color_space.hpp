/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef MODULES_HOLOVIZ_SRC_HOLOVIZ_COLOR_SPACE_HPP
#define MODULES_HOLOVIZ_SRC_HOLOVIZ_COLOR_SPACE_HPP

#include <cstdint>

#include "holoviz/image_format.hpp"

namespace holoscan::viz {

/**
 * The color space specifies how the surface data is interpreted when presented on screen.
 */
enum class ColorSpace {
  SRGB_NONLINEAR,        ///< sRGB color space
  EXTENDED_SRGB_LINEAR,  ///< extended sRGB color space to be displayed using a linear EOTF
  BT2020_LINEAR,         ///< BT2020 color space to be displayed using a linear EOTF
  HDR10_ST2084,  ///< HDR10 (BT2020 color) space to be displayed using the SMPTE ST2084 Perceptual
                 ///< Quantizer (PQ) EOTF
  PASS_THROUGH,  ///< color components are used “as is”
  BT709_LINEAR,  ///< BT709 color space to be displayed using a linear EOTF
};

}  // namespace holoscan::viz

#endif /* MODULES_HOLOVIZ_SRC_HOLOVIZ_COLOR_SPACE_HPP */
