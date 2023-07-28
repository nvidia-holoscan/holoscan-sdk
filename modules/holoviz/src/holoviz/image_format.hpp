/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef HOLOSCAN_VIZ_HOLOVIZ_IMAGE_FORMAT_HPP
#define HOLOSCAN_VIZ_HOLOVIZ_IMAGE_FORMAT_HPP

#include <cstdint>

namespace holoscan::viz {

/**
 * Image formats.
 *
 * {component format}_{numeric format}
 *
 * - component format
 *   - indicates the size in bits of the R, G, B and A components if present
 * - numeric format
 *   - UNORM - unsigned normalize values, range [0, 1]
 *   - SNORM - signed normalized values, range [-1,1]
 *   - UINT - unsigned integer values, range [0,2n-1]
 *   - SINT - signed integer values, range [-2n-1,2n-1-1]
 *   - SFLOAT - signed floating-point numbers
 *   - SRGB - the R, G, and B components are unsigned normalized values that
 *            represent values using sRGB nonlinear encoding, while the A
 *            component (if one exists) is a regular unsigned normalized value
 */
enum class ImageFormat {
  R8_UINT,     ///< specifies a one-component,
               ///  8-bit unsigned integer format that has a single 8-bit R component
  R16_UINT,    ///< specifies a one-component,
               ///  16-bit unsigned integer format that has a single 16-bit R component
  R16_SFLOAT,  ///< specifies a one-component,
               ///  16-bit signed floating-point format that has a single 16-bit R component
  R32_UINT,    ///< specifies a one-component,
               ///  16-bit unsigned integer format that has a single 16-bit R component
  R32_SFLOAT,  ///< specifies a one-component,
               ///  32-bit signed floating-point format that has a single 32-bit R component

  R8G8B8_UNORM,  ///< specifies a three-component,
                 ///  24-bit unsigned normalized format that has an 8-bit R component in byte 0,
                 ///  an 8-bit G component in byte 1, and an 8-bit B component in byte 2
  B8G8R8_UNORM,  ///< specifies a three-component,
                 ///  24-bit unsigned normalized format that has an 8-bit B component in byte 0,
                 ///  an 8-bit G component in byte 1, and an 8-bit R component in byte 2

  R8G8B8A8_UNORM,  ///< specifies a four-component,
                   ///  32-bit unsigned normalized format that has an 8-bit R component in byte 0,
                   ///  an 8-bit G component in byte 1, an 8-bit B component in byte 2,
                   ///  and an 8-bit A component in byte 3
  B8G8R8A8_UNORM,  ///< specifies a four-component,
                   ///  32-bit unsigned normalized format that has an 8-bit B component in byte 0,
                   ///  an 8-bit G component in byte 1, an 8-bit R component in byte 2,
                   ///  and an 8-bit A component in byte 3
  R16G16B16A16_UNORM,   ///< specifies a four-component,
                        ///  64-bit unsigned normalized format that has
                        ///  a 16-bit R component in bytes 0..1,
                        ///  a 16-bit G component in bytes 2..3,
                        ///  a 16-bit B component in bytes 4..5,
                        ///  and a 16-bit A component in bytes 6..7
  R16G16B16A16_SFLOAT,  ///< specifies a four-component,
                        ///  64-bit signed floating-point format that has
                        ///  a 16-bit R component in bytes 0..1,
                        ///  a 16-bit G component in bytes 2..3,
                        ///  a 16-bit B component in bytes 4..5,
                        ///  and a 16-bit A component in bytes 6..7
  R32G32B32A32_SFLOAT,  ///< specifies a four-component,
                        ///  128-bit signed floating-point format that has
                        ///  a 32-bit R component in bytes 0..3,
                        ///  a 32-bit G component in bytes 4..7,
                        ///  a 32-bit B component in bytes 8..11,
                        ///  and a 32-bit A component in bytes 12..15
  R8_UNORM,             ///< specifies a one-component,
                        ///  8-bit unsigned normalized format that has a single 8-bit R component.
  R16_UNORM,            ///< specifies a one-component,
                        ///  16-bit unsigned normalized format that has a single 16-bit R component.
  D32_SFLOAT,           ///< specifies a one-component, 32-bit signed floating-point format that has
                        ///  32 bits in the depth component.
};

}  // namespace holoscan::viz

#endif /* HOLOSCAN_VIZ_HOLOVIZ_IMAGE_FORMAT_HPP */
