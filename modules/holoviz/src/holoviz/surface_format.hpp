/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef MODULES_HOLOVIZ_SRC_HOLOVIZ_SURFACE_FORMAT_HPP
#define MODULES_HOLOVIZ_SRC_HOLOVIZ_SURFACE_FORMAT_HPP

#include <cstdint>

#include "holoviz/color_space.hpp"
#include "holoviz/image_format.hpp"

namespace holoscan::viz {

/**
 * Describes image format-color space pair
 */
struct SurfaceFormat {
  ImageFormat image_format_;  ///< image format
  ColorSpace color_space_;    ///< color space
};

}  // namespace holoscan::viz

#endif /* MODULES_HOLOVIZ_SRC_HOLOVIZ_SURFACE_FORMAT_HPP */
