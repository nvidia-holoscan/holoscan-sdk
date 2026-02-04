/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
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

#endif /* HOLOSCAN_VIZ_HOLOVIZ_RENDER_FLAGS_HPP */
