/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOSCAN_VIZ_HOLOVIZ_RENDER_FLAGS_HPP
#define HOLOSCAN_VIZ_HOLOVIZ_RENDER_FLAGS_HPP

#include <cstdint>
#include <type_traits>

namespace holoscan::viz {

namespace render_flags {

/// Flags passed controlling the render pass
typedef enum {
  NONE = 0x00000000,               ///< none
  DONT_CLEAR_COLOR = 0x00000001,   ///< don't clear the color buffer
  DONT_CLEAR_DEPTH = 0x00000002,   ///< don't clear the depth buffer
  DONT_SWAP_BUFFERS = 0x00000004,  ///< don't swap the buffers
} RenderFlags;

}  // namespace render_flags

using RenderFlags = render_flags::RenderFlags;

}  // namespace holoscan::viz

// NOLINTBEGIN(clang-analyzer-optin.core.EnumCastOutOfRange) bitmask enum combination is valid

/**
 * Bitwise OR operator for RenderFlags
 */
constexpr holoscan::viz::RenderFlags operator|(holoscan::viz::RenderFlags a,
                                               holoscan::viz::RenderFlags b) {
  return static_cast<holoscan::viz::RenderFlags>(
      static_cast<std::underlying_type<holoscan::viz::RenderFlags>::type>(a) |
      static_cast<std::underlying_type<holoscan::viz::RenderFlags>::type>(b));
}

/**
 * Bitwise AND operator for RenderFlags
 */
constexpr holoscan::viz::RenderFlags operator&(holoscan::viz::RenderFlags a,
                                               holoscan::viz::RenderFlags b) {
  return static_cast<holoscan::viz::RenderFlags>(
      static_cast<std::underlying_type<holoscan::viz::RenderFlags>::type>(a) &
      static_cast<std::underlying_type<holoscan::viz::RenderFlags>::type>(b));
}

// NOLINTEND(clang-analyzer-optin.core.EnumCastOutOfRange)

#endif /* HOLOSCAN_VIZ_HOLOVIZ_RENDER_FLAGS_HPP */
