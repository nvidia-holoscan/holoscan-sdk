/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOSCAN_VIZ_HOLOVIZ_DEPTH_MAP_RENDER_MODER_HPP
#define HOLOSCAN_VIZ_HOLOVIZ_DEPTH_MAP_RENDER_MODER_HPP

#include <cstdint>

namespace holoscan::viz {

/**
 * Depth map render mode.
 */
enum class DepthMapRenderMode {
  POINTS,    ///< render points
  LINES,     ///< render lines
  TRIANGLES  ///< render triangles
};

}  // namespace holoscan::viz

#endif /* HOLOSCAN_VIZ_HOLOVIZ_DEPTH_MAP_RENDER_MODER_HPP */
