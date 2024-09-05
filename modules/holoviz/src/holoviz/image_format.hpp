/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef MODULES_HOLOVIZ_SRC_HOLOVIZ_IMAGE_FORMAT_HPP
#define MODULES_HOLOVIZ_SRC_HOLOVIZ_IMAGE_FORMAT_HPP

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
 * - multi-planar formats
 *   - 2PLANE - data is stored in two separate memory planes
 *   - 3PLANE - data is stored in three separate memory planes
 * - YUV formats
 *   - 420 - the horizontal and vertical resolution of the chroma (UV) planes is halved
 *   - 422 - the horizontal of the chroma (UV) planes is halved
 */
enum class ImageFormat {
  R8_UINT,   ///< specifies a one-component, 8-bit unsigned integer format that has
             ///  a single 8-bit R component
  R8_SINT,   ///< specifies a one-component, 8-bit signed integer format that has
             ///  a single 8-bit R component
  R8_UNORM,  ///< specifies a one-component, 8-bit unsigned normalized format that has
             ///  a single 8-bit R component
  R8_SNORM,  ///< specifies a one-component, 8-bit signed normalized format that has
             ///  a single 8-bit R component
  R8_SRGB,   ///< specifies a one-component, 8-bit unsigned normalized format that has
             ///  a single 8-bit R component stored with sRGB nonlinear encoding

  R16_UINT,    ///< specifies a one-component, 16-bit unsigned integer format that has
               ///  a single 16-bit R component
  R16_SINT,    ///< specifies a one-component, 16-bit signed integer format that has
               ///  a single 16-bit R component
  R16_UNORM,   ///< specifies a one-component, 16-bit unsigned normalized format that has
               ///  a single 16-bit R component
  R16_SNORM,   ///< specifies a one-component, 16-bit signed normalized format that has
               ///  a single 16-bit R component
  R16_SFLOAT,  ///< specifies a one-component, 16-bit signed floating-point format that has
               ///  a single 16-bit R component

  R32_UINT,    ///< specifies a one-component, 16-bit unsigned integer format that has
               ///  a single 16-bit R component
  R32_SINT,    ///< specifies a one-component, 16-bit signed integer format that has
               ///  a single 16-bit R component
  R32_SFLOAT,  ///< specifies a one-component, 32-bit signed floating-point format that has
               ///  a single 32-bit R component

  R8G8B8_UNORM,  ///< specifies a three-component, 24-bit unsigned normalized format that has
                 ///  a 8-bit R component in byte 0,
                 ///  a 8-bit G component in byte 1,
                 ///  and a 8-bit B component in byte 2
  R8G8B8_SNORM,  ///< specifies a three-component, 24-bit signed normalized format that has
                 ///  a 8-bit R component in byte 0,
                 ///  a 8-bit G component in byte 1,
                 ///  and a 8-bit B component in byte 2
  R8G8B8_SRGB,   ///< specifies a three-component, 24-bit unsigned normalized format that has
                 ///  a 8-bit R component stored with sRGB nonlinear encoding in byte 0,
                 ///  a 8-bit G component stored with sRGB nonlinear encoding in byte 1,
                 ///  and a 8-bit B component stored with sRGB nonlinear encoding in byte 2

  R8G8B8A8_UNORM,  ///< specifies a four-component, 32-bit unsigned normalized format that has
                   ///  a 8-bit R component in byte 0,
                   ///  a 8-bit G component in byte 1,
                   ///  a 8-bit B component in byte 2,
                   ///  and a 8-bit A component in byte 3
  R8G8B8A8_SNORM,  ///< specifies a four-component, 32-bit signed normalized format that has
                   ///  a 8-bit R component in byte 0,
                   ///  a 8-bit G component in byte 1,
                   ///  a 8-bit B component in byte 2,
                   ///  and a 8-bit A component in byte 3
  R8G8B8A8_SRGB,   ///< specifies a four-component, 32-bit unsigned normalized format that has
                   ///  a 8-bit R component stored with sRGB nonlinear encoding in byte 0,
                   ///  a 8-bit G component stored with sRGB nonlinear encoding in byte 1,
                   ///  a 8-bit B component stored with sRGB nonlinear encoding in byte 2,
                   ///  and a 8-bit A component in byte 3

  R16G16B16A16_UNORM,   ///< specifies a four-component,
                        ///  64-bit unsigned normalized format that has
                        ///  a 16-bit R component in bytes 0..1,
                        ///  a 16-bit G component in bytes 2..3,
                        ///  a 16-bit B component in bytes 4..5,
                        ///  and a 16-bit A component in bytes 6..7
  R16G16B16A16_SNORM,   ///< specifies a four-component,
                        ///  64-bit signed normalized format that has
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

  D16_UNORM,     ///< specifies a one-component, 16-bit unsigned normalized format that has
                 ///  a single 16-bit depth component
  X8_D24_UNORM,  ///< specifies a two-component, 32-bit format that has
                 ///  24 unsigned normalized bits in the depth component,
                 ///  and, optionally, 8 bits that are unused
  D32_SFLOAT,    ///< specifies a one-component, 32-bit signed floating-point format that has
                 ///  32 bits in the depth component

  A2B10G10R10_UNORM_PACK32,  ///< specifies a four-component, 32-bit packed unsigned normalized
                             ///  format that has
                             ///  a 2-bit A component in bits 30..31,
                             ///  a 10-bit B component in bits 20..29,
                             ///  a 10-bit G component in bits 10..19,
                             ///  and a 10-bit R component in bits 0..9.

  A2R10G10B10_UNORM_PACK32,  ///< specifies a four-component, 32-bit packed unsigned normalized
                             ///  format that has
                             ///  a 2-bit A component in bits 30..31,
                             ///  a 10-bit R component in bits 20..29,
                             ///  a 10-bit G component in bits 10..19,
                             ///  and a 10-bit B component in bits 0..9.

  B8G8R8A8_UNORM,  ///< specifies a four-component, 32-bit unsigned normalized format that has
                   ///  a 8-bit B component in byte 0,
                   ///  a 8-bit G component in byte 1,
                   ///  a 8-bit R component in byte 2,
                   ///  and a 8-bit A component in byte 3
  B8G8R8A8_SRGB,   ///< specifies a four-component, 32-bit unsigned normalized format that has
                   ///  a 8-bit B component stored with sRGB nonlinear encoding in byte 0,
                   ///  a 8-bit G component stored with sRGB nonlinear encoding in byte 1,
                   ///  a 8-bit R component stored with sRGB nonlinear encoding in byte 2,
                   ///  and a 8-bit A component in byte 3

  A8B8G8R8_UNORM_PACK32,  ///< specifies a four-component, 32-bit packed unsigned normalized format
                          ///  that has
                          ///  an 8-bit A component in bits 24..31,
                          ///  an 8-bit B component in bits 16..23,
                          ///  an 8-bit G component in bits 8..15,
                          ///  and an 8-bit R component in bits 0..7.
  A8B8G8R8_SRGB_PACK32,   ///< specifies a four-component, 32-bit packed unsigned normalized format
                          ///  that has
                          ///  an 8-bit A component in bits 24..31,
                          ///  an 8-bit B component stored with sRGB nonlinear encoding in
                          ///  bits 16..23,
                          ///  an 8-bit G component stored with sRGB nonlinear encoding
                          ///  in bits 8..15,
                          ///  and an 8-bit R component stored with sRGB nonlinear
                          ///  encoding in bits 0..7.

  Y8U8Y8V8_422_UNORM,  ///< specifies a four-component, 32-bit format containing a pair of Y
                       ///  components, a V component, and a U component, collectively encoding a
                       ///  2×1 rectangle of unsigned normalized RGB texel data. One Y value is
                       ///  present at each i coordinate, with the U and V values shared across both
                       ///  Y values and thus recorded at half the horizontal resolution of the
                       ///  image. This format has an 8-bit Y component for the even i coordinate in
                       ///  byte 0, an 8-bit U component in byte 1, an 8-bit Y component for the odd
                       ///  i coordinate in byte 2, and an 8-bit V component in byte 3. This format
                       ///  only supports images with a width that is a multiple of two.
  U8Y8V8Y8_422_UNORM,  ///< specifies a four-component, 32-bit format containing a pair of Y
                       ///  components, a V component, and a U component, collectively encoding a
                       ///  2×1 rectangle of unsigned normalized RGB texel data. One Y value is
                       ///  present at each i coordinate, with the U and V values shared across both
                       ///  Y values and thus recorded at half the horizontal resolution of the
                       ///  image. This format has an 8-bit U component in byte 0, an 8-bit Y
                       ///  component for the even i coordinate in byte 1, an 8-bit V component in
                       ///  byte 2, and an 8-bit Y component for the odd i coordinate in byte 3.
                       ///  This format only supports images with a width that is a multiple of two.
  Y8_U8V8_2PLANE_420_UNORM,  ///< specifies an unsigned normalized multi-planar format that has an
                             ///  8-bit Y component in plane 0, and a two-component, 16-bit UV plane
                             ///  1 consisting of an 8-bit U component in byte 0 and an 8-bit V
                             ///  component in byte 1. The horizontal and vertical dimensions of the
                             ///  UV plane are halved relative to the image dimensions. This format
                             ///  only supports images with a width and height that are a multiple
                             ///  of two.
  Y8_U8V8_2PLANE_422_UNORM,  ///< specifies an unsigned normalized multi-planar format that has an
                             ///  8-bit Y component in plane 0, and a two-component, 16-bit UV plane
                             ///  1 consisting of an 8-bit U component in byte 0 and an 8-bit V
                             ///  component in byte 1. The horizontal dimension of the UV plane is
                             ///  halved relative to the image dimensions. This format only supports
                             ///  images with a width that is a multiple of two.
  Y8_U8_V8_3PLANE_420_UNORM,  ///< specifies an unsigned normalized multi-planar format that has an
                              ///  8-bit Y component in plane 0, an 8-bit U component in plane 1,
                              ///  and an 8-bit V component in plane 2. The horizontal and vertical
                              ///  dimensions of the V and U planes are halved relative to the image
                              ///  dimensions. This format only supports images with a width and
                              ///  height that are a multiple of two.
  Y8_U8_V8_3PLANE_422_UNORM,  ///< specifies an unsigned normalized multi-planar format that has an
                              ///  8-bit Y component in plane 0, an 8-bit U component in plane 1,
                              ///  and an 8-bit V component in plane 2. The horizontal dimension of
                              ///  the V and U plane is halved relative to the image dimensions.
                              ///  This format only supports images with a width that is a multiple
                              ///  of two.
  Y16_U16V16_2PLANE_420_UNORM,  ///< specifies an unsigned normalized multi-planar format that has a
                                ///  16-bit Y component in each 16-bit word of plane 0, and a
                                ///  two-component, 32-bit UV plane 1 consisting of a 16-bit U
                                ///  component in the word in bytes 0..1, and a 16-bit V component
                                ///  in the word in bytes 2..3. The horizontal and vertical
                                ///  dimensions of the UV plane are halved relative to the image
                                ///  dimensions. This format only supports images with a width and
                                ///  height that are a multiple of two.
  Y16_U16V16_2PLANE_422_UNORM,  ///< specifies an unsigned normalized multi-planar format that has a
                                ///  16-bit Y component in each 16-bit word of plane 0, and a
                                ///  two-component, 32-bit UV plane 1 consisting of a 16-bit U
                                ///  component in the word in bytes 0..1, and a 16-bit V component
                                ///  in the word in bytes 2..3. The horizontal dimension of the UV
                                ///  plane is halved relative to the image dimensions. This format
                                ///  only supports images with a width that is a multiple of two.
  Y16_U16_V16_3PLANE_420_UNORM,  ///< specifies an unsigned normalized multi-planar format that has
                                 ///  a 16-bit Y component in each 16-bit word of plane 0, a 16-bit
                                 ///  U component in each 16-bit word of plane 1, and a 16-bit V
                                 ///  component in each 16-bit word of plane 2. The horizontal and
                                 ///  vertical dimensions of the V and U planes are halved relative
                                 ///  to the image dimensions. This format only supports images with
                                 ///  a width and height that are a multiple of two.
  Y16_U16_V16_3PLANE_422_UNORM,  ///< specifies an unsigned normalized multi-planar format that has
                                 ///  a 16-bit Y component in each 16-bit word of plane 0, a 16-bit
                                 ///  U component in each 16-bit word of plane 1, and a 16-bit V
                                 ///  component in each 16-bit word of plane 2. The horizontal
                                 ///  dimension of the V and U plane is halved relative to the image
                                 ///  dimensions. This format only supports images with a width that
                                 ///  is a multiple of two.
};

/**
 * Component swizzle.
 *
 * Specifies the component value placed in each component of the output vector.
 *
 * For example, to render a BGRA 8-bit image set the ImageFormat to R8G8B8A8_??? and the component
 * mapping to
 * @code{.cpp}
 *  { ComponentSwizzle::B, ComponentSwizzle::G, ComponentSwizzle::R, ComponentSwizzle::A }
 * @endcode
 */
enum class ComponentSwizzle {
  IDENTITY,  ///< specifies that the component is set to the identity swizzle
  ZERO,      ///< specifies that the component is set to zero
  ONE,  ///< specifies that the component is set to either 1 or 1.0, depending on whether the type
        ///  of the image view format is integer or floating-point respectively
  R,    ///< specifies that the component is set to the value of the R component of the image
  G,    ///< specifies that the component is set to the value of the G component of the image
  B,    ///< specifies that the component is set to the value of the B component of the image
  A     ///< specifies that the component is set to the value of the A component of the image
};

/**
 * Defines the conversion from the source color model to the shader color model.
 */
enum class YuvModelConversion {
  YUV_601,   ///< specifies the color model conversion from YUV to RGB defined in BT.601
  YUV_709,   ///< specifies the color model conversion from YUV to RGB defined in BT.709
  YUV_2020,  ///< specifies the color model conversion from YUV to RGB defined in BT.2020
};

/**
 * Specifies the YUV range
 */
enum class YuvRange {
  ITU_FULL,    ///< specifies that the full range of the encoded values are valid and
               ///< interpreted according to the ITU “full range” quantization rules
  ITU_NARROW,  ///< specifies that headroom and foot room are reserved in the numerical range
               ///< of encoded values, and the remaining values are expanded according to the
               ///< ITU “narrow range” quantization rules
};

/**
 * Defines the location of downsampled chroma component samples relative to the luma samples
 */
enum class ChromaLocation {
  COSITED_EVEN,  ///< specifies that downsampled chroma samples are aligned with luma samples with
                 ///< even coordinates
  MIDPOINT,  ///< specifies that downsampled chroma samples are located half way between each even
             ///< luma sample and the nearest higher odd luma sample.
};

}  // namespace holoscan::viz

#endif /* MODULES_HOLOVIZ_SRC_HOLOVIZ_IMAGE_FORMAT_HPP */
