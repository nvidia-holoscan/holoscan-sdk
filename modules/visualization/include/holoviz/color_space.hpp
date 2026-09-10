// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace holoviz {

/// How surface data is interpreted when presented on screen.
enum class ColorSpace {
  kPassThrough,    ///< color components are used "as is"
  kSRGBNonlinear,  ///< sRGB color space
  kBT709Linear,    ///< BT.709 color space, displayed using a linear EOTF
};

}  // namespace holoviz
