/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <holoviz/pipeline.hpp>
#include "holoviz_op/types.hpp"

namespace holoscan::visualization::backends {

/// The layer shaders emit straight (non-premultiplied) alpha — layer sources
/// (caller RGBA textures, LUT palettes, geometry colors) are straight alpha by
/// nature. The HoloViz BlendState defaults compose premultiplied (src_color = kOne),
/// so every layer pipeline sets these factors explicitly; this is the same
/// convention and state as the text renderers (see monospace_text_renderer.cpp).
/// The alpha channel accumulates coverage: a = a_s + a_d * (1 - a_s).
inline constexpr holoviz::BlendState kStraightAlphaLayerBlend{
    .enable = true,
    .src_color = holoviz::BlendFactor::kSrcAlpha,
    .dst_color = holoviz::BlendFactor::kOneMinusSrcAlpha,
    .src_alpha = holoviz::BlendFactor::kOne,
    .dst_alpha = holoviz::BlendFactor::kOneMinusSrcAlpha,
};

}  // namespace holoscan::visualization::backends
