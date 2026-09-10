// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace holoviz {

/// Overlay primitive topologies retained by the visualization module (2D only; a survey of
/// real applications showed near-zero use of 3D/triangle primitives, so those are deferred
/// to a future submodule).
///
/// These are semantic overlay shapes, not GPU primitives: a RenderBackend implementation
/// expands each value into rhi::PrimitiveTopology primitives at record time
/// (kLineList/kLineStrip pass through; kCrossList expands to line lists; kRectangleList and
/// kOvalList expand to triangle lists). There is deliberately no public mapping function.
enum class GeometryTopology : uint8_t {
  kLineList,
  kLineStrip,
  kCrossList,      ///< a cross is defined by a center coordinate and a size
  kRectangleList,  ///< axis-aligned rectangle, defined by two corner coordinates
  kOvalList,       ///< defined by a center coordinate and two axis sizes
};

}  // namespace holoviz
