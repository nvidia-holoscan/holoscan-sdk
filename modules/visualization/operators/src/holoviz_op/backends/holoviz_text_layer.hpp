// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// The text layer of HolovizRenderBackend, on the existing monospace TextRenderer
// (<holoviz/text/monospace_text_renderer.hpp>): zero per-frame GPU allocation and zero
// uploads — glyph runs travel as push constants, so end_frame()'s token is a no-op the
// device skips. That makes it the simplest layer kind and the one that proves the seam's
// frame path end to end first.

#pragma once

#include <cstddef>
#include <vector>

#include <holoscan/visualization/holoviz_op.hpp>
#include <holoviz/command_list.hpp>
#include <holoviz/common.hpp>
#include <holoviz/device.hpp>
#include <holoviz/expected.hpp>
#include <holoviz/text/text_renderer.hpp>
#include "holoviz_op/backends/frame_data.hpp"
#include "holoviz_op/types.hpp"

namespace holoscan::visualization::backends {

struct HolovizTextLayerState {
  TextLayerDesc desc;
  text::TextRenderer renderer;
  size_t glyph_high_water = 0;
};

/// Create the layer's renderer (monospace, built-in font). All GPU state the renderer will
/// ever touch — atlas, glyph table, pipeline object — is created here, per the allocate()
/// contract.
[[nodiscard]] expected<HolovizTextLayerState> allocate_holoviz_text_layer(
    holoviz::Device& device, const TextLayerDesc& desc);

/// Record one frame's runs into the caller's open render scope. `slot` is the backend's
/// frame index, forwarded with `slot_last_use` to the renderer's per-slot reuse gate.
/// Enforces the declared max_glyphs bound (exceeding it is a mis-declaration:
/// kInvalidArgument). Clobbers viewport/scissor/pipeline state — the caller re-sets its
/// own afterwards. Appends the upload token to `waits` (a no-op token for the monospace
/// path; kept uniform so swapping in an uploading text renderer changes nothing here).
[[nodiscard]] expected<void> record_holoviz_text_layer(
    HolovizTextLayerState& state, holoviz::CommandList& cmd, holoviz::Extent2D extent,
    uint32_t slot, holoviz::SubmitSyncToken slot_last_use, const TextFrameData& frame,
    std::vector<holoviz::ExternalSyncToken>& waits);

}  // namespace holoscan::visualization::backends
