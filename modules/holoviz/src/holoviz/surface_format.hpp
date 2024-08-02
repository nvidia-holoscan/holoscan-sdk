/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
