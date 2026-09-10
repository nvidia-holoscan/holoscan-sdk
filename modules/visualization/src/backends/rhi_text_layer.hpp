// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// The text layer of RhiRenderBackend, on the existing monospace TextRenderer
// (<holoviz/text/monospace_text_renderer.hpp>): zero per-frame GPU allocation and zero
// uploads — glyph runs travel as push constants, so end_frame()'s token is a no-op the
// device skips. That makes it the simplest layer kind and the one that proves the seam's
// frame path end to end first.

#pragma once

#include <cstddef>
#include <vector>

#include <holoviz/expected.hpp>
#include <holoviz/frame_data.hpp>
#include <holoviz/layer_desc.hpp>
#include <holoviz/rhi/command_list.hpp>
#include <holoviz/rhi/common.hpp>
#include <holoviz/rhi/device.hpp>
#include <holoviz/text/text_renderer.hpp>

namespace holoviz::backends {

struct RhiTextLayerState {
  TextLayerDesc desc;
  text::TextRenderer renderer;
  size_t glyph_high_water = 0;
};

/// Create the layer's renderer (monospace, built-in font). All GPU state the renderer will
/// ever touch — atlas, glyph table, pipeline object — is created here, per the allocate()
/// contract.
[[nodiscard]] expected<RhiTextLayerState> allocate_rhi_text_layer(rhi::Device& device,
                                                                  const TextLayerDesc& desc);

/// Record one frame's runs into the caller's open render scope. `slot` is the backend's
/// frame index, forwarded with `slot_last_use` to the renderer's per-slot reuse gate.
/// Enforces the declared max_glyphs bound (exceeding it is a mis-declaration:
/// kInvalidArgument). Clobbers viewport/scissor/pipeline state — the caller re-sets its
/// own afterwards. Appends the upload token to `waits` (a no-op token for the monospace
/// path; kept uniform so swapping in an uploading text renderer changes nothing here).
[[nodiscard]] expected<void> record_rhi_text_layer(RhiTextLayerState& state, rhi::CommandList& cmd,
                                                   rhi::Extent2D extent, uint32_t slot,
                                                   rhi::SubmitSyncToken slot_last_use,
                                                   const TextFrameData& frame,
                                                   std::vector<rhi::ExternalSyncToken>& waits);

}  // namespace holoviz::backends
