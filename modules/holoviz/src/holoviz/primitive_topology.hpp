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

#ifndef HOLOSCAN_VIZ_HOLOVIZ_PRIMITIVE_TOPOLOGY_HPP
#define HOLOSCAN_VIZ_HOLOVIZ_PRIMITIVE_TOPOLOGY_HPP

namespace holoscan::viz {

/**
 * Primitive topology
 */
enum class PrimitiveTopology {
  POINT_LIST,        ///< point primitives, one coordinate (x, y) per primitive
  LINE_LIST,         ///< line primitives, two coordinates (x0, y0) and (x1, y1) per primitive
  LINE_STRIP,        ///< line strip primitive, a line primitive i is defined by
                     ///  each coordinate (xi, yi) and the following (xi+1, yi+1)
  TRIANGLE_LIST,     ///< triangle primitive, three coordinates
                     ///  (x0, y0), (x1, y1) and (x2, y2) per primitive
  CROSS_LIST,        ///< cross primitive, a cross is defined by the center coordinate
                     ///  and the size (xi, yi, si)
  RECTANGLE_LIST,    ///< axis aligned rectangle primitive,
                     ///  each rectangle is defined by two coordinates (xi, yi) and (xi+1, yi+1)
  OVAL_LIST,         ///< oval primitive, an oval primitive is defined by the center coordinate
                     ///  and the axis sizes (xi, yi, sxi, syi)
  POINT_LIST_3D,     ///< 3D point primitives, one coordinate (x, y, z) per primitive
  LINE_LIST_3D,      ///< 3D line primitives, two coordinates (x0, y0, z0) and (x1, y1, z1) per
                     ///  primitive
  LINE_STRIP_3D,     ///< 3D line strip primitive, a line primitive i is defined by
                     ///  each coordinate (xi, yi, zi) and (xi+1, yi+1, zi+1) per
                     ///  primitive
  TRIANGLE_LIST_3D,  ///< 3D triangle primitive, three coordinates
                     ///  (x0, y0, z0), (x1, y1, z1) and (x2, y2, z2) per primitive
};

}  // namespace holoscan::viz

#endif /* HOLOSCAN_VIZ_HOLOVIZ_PRIMITIVE_TOPOLOGY_HPP */
