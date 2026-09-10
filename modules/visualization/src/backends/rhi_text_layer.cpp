// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "backends/rhi_text_layer.hpp"

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

#include <holoviz/text/monospace_text_renderer.hpp>

namespace holoviz::backends {

namespace {

/// Close the renderer's frame if record_rhi_text_layer errors out between begin_frame and
/// end_frame; otherwise the layer would refuse every later frame ("previous frame missing
/// end_frame"). cancel_frame is idempotent and host-side only.
struct FrameCancelGuard {
  text::TextRenderer* renderer = nullptr;  // nullptr once disarmed
  FrameCancelGuard(const FrameCancelGuard&) = delete;
  FrameCancelGuard& operator=(const FrameCancelGuard&) = delete;
  FrameCancelGuard(FrameCancelGuard&&) = delete;
  FrameCancelGuard& operator=(FrameCancelGuard&&) = delete;
  explicit FrameCancelGuard(text::TextRenderer& text_renderer) : renderer(&text_renderer) {}
  ~FrameCancelGuard() {
    if (renderer != nullptr) {
      (void)renderer->cancel_frame();
    }
  }
};

}  // namespace

expected<RhiTextLayerState> allocate_rhi_text_layer(rhi::Device& device,
                                                    const TextLayerDesc& desc) {
  if (desc.max_glyphs == 0) {
    return make_unexpected(
        Error{.code = ErrorCode::kInvalidArgument,
              .message = "[holoviz] TextLayerDesc::max_glyphs must be nonzero."});
  }
  VIZ_UNWRAP(
      text::TextRenderer,
      renderer,
      text::create_monospace_text_renderer(device,
                                           {.debug_name = "rhi render backend text layer",
                                            // Sizes the renderer's per-frame draw list for the
                                            // declared budget, so record() stays allocation-free.
                                            .max_glyphs = static_cast<uint32_t>(desc.max_glyphs)}));
  return RhiTextLayerState{desc, std::move(renderer)};
}

expected<void> record_rhi_text_layer(RhiTextLayerState& state, rhi::CommandList& cmd,
                                     rhi::Extent2D extent, uint32_t slot,
                                     rhi::SubmitSyncToken slot_last_use, const TextFrameData& frame,
                                     std::vector<rhi::ExternalSyncToken>& waits) {
  // Validate every run before opening the renderer's frame: a rejection past begin_frame
  // would leave the frame open, and the mirror check inside add_text would then be sticky.
  size_t glyph_count = 0;
  for (const TextRun& run : frame.runs) {
    glyph_count += run.utf8_text.size();
    if (!(run.size_px > 0.0F) || !std::isfinite(run.size_px)) {
      return make_unexpected(
          Error{.code = ErrorCode::kInvalidArgument,
                .message = "[holoviz] TextRun::size_px must be positive and finite."});
    }
  }
  if (glyph_count > state.desc.max_glyphs) {
    return make_unexpected(
        Error{.code = ErrorCode::kInvalidArgument,
              .message = "[holoviz] Frame text exceeds the layer's declared max_glyphs — a "
                         "mis-declaration, not an expected code path."});
  }
  state.glyph_high_water = std::max(state.glyph_high_water, glyph_count);

  VIZ_FORWARD_ERROR(state.renderer.begin_frame(extent, slot, slot_last_use));
  FrameCancelGuard cancel_guard(state.renderer);
  for (const TextRun& run : frame.runs) {
    // TextRun positions are normalized [0,1] framebuffer coordinates; the renderer speaks
    // pixels (top-left anchored, matching TextRun's contract).
    VIZ_FORWARD_ERROR(state.renderer.add_text(
        text::TextDesc{
            .position = {run.x * static_cast<float>(extent.width),
                         run.y * static_cast<float>(extent.height)},
            .size_px = run.size_px,
            .color = run.color,
        },
        run.utf8_text));
  }
  VIZ_UNWRAP(rhi::ExternalSyncToken, upload_token, state.renderer.end_frame());
  cancel_guard.renderer = nullptr;  // frame closed; nothing left to cancel
  waits.push_back(upload_token);

  return state.renderer.record(cmd, extent);
}

}  // namespace holoviz::backends
