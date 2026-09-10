/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Push-constant-only monospace text ("push-text"): zero per-frame buffer
// writes — no CUDA stream, no frames-in-flight ring. Each string run is one
// instanced draw — draw(6, char_count), or draw(3, char_count) in the
// fill-rectangle mode (see MonospaceFillMode) — whose per-draw data is push
// constants only. The block uses std140 layout (uint4 char words):
//
//   struct PushConstants {          // 256 bytes total
//     float2 origin_px;             // top-left of the run, framebuffer pixels
//     float2 clip_scale;            // 2/fb_width, 2/fb_height
//     float  glyph_height_px;       // rendered height; advance scales with it
//     uint32_t rgba;                // packed color, red in the lowest byte
//     /* 8 pad bytes */
//     uint4 packed_chars[14];       // 224 ASCII bytes, 4 per word
//   };
//
// 256 bytes is NVIDIA's push-constant budget (above the 128-byte Vulkan
// minimum); creation validates the device limit and the Slang-reflected
// block size and fails cleanly on anything smaller. The vertex shader derives
// each quad from SV_VertexID/SV_InstanceID — no vertex buffer exists — and
// looks up UVs, quad corners, and the monospace advance in a static glyph
// table (StructuredBuffer, 97 entries: metrics header + ASCII 32..127)
// uploaded once at creation next to the baked atlas image (kSampled, bound in
// a Descriptor). ASCII only: other bytes render as '?'. Longer strings split
// into multiple draws with advanced origins. Because nothing is uploaded per
// frame, end_frame() returns a default (no-op) ExternalSyncToken, which
// Device::submit() skips.

#pragma once

#include <cstdint>

#include <holoviz/holoviz_export.h>

#include <holoviz/expected.hpp>
#include <holoviz/text/text_renderer.hpp>

namespace holoviz::text {

/// Meaningful push-header bytes; the GPU block pads to 32 before the char words.
inline constexpr uint32_t kMonospacePushHeaderBytes = 24;
/// Longest run one instanced draw covers; longer strings split into multiple
/// draws with advanced origins.
inline constexpr uint32_t kMonospaceMaxCharsPerDraw = 224;

/// How glyph quads are rasterized. kFillRectangle draws one triangle per glyph
/// and lets VK_NV_fill_rectangle rasterize its axis-aligned bounding rectangle
/// — half the vertex work, no diagonal seam. It requires
/// rhi::CapabilityReport::fill_rectangle: creation fails without it, so gate
/// the selection on rhi::Device::capabilities().
enum class MonospaceFillMode : uint8_t {
  kTriangles,      ///< Two triangles per glyph (works everywhere).
  kFillRectangle,  ///< One triangle per glyph via VK_NV_fill_rectangle.
};

struct MonospaceTextRendererDesc {
  /// Glyph rasterization mode; kFillRectangle requires
  /// rhi::CapabilityReport::fill_rectangle (see MonospaceFillMode).
  MonospaceFillMode fill_mode = MonospaceFillMode::kTriangles;
  /// Monospace TTF to bake the ASCII atlas from; nullptr selects the built-in
  /// font (ImGui's ProggyClean, which bakes at 13 px — atlas_glyph_height_px
  /// is ignored in that case, and larger draw sizes scale up softly).
  const char* ttf_path = nullptr;
  /// Glyph height the atlas is baked at; draws scale from it.
  float atlas_glyph_height_px = 32.0F;
  /// Optional label used for the renderer's pipeline and error messages;
  /// nullptr selects a default.
  const char* debug_name = nullptr;
  /// Upper bound on glyphs queued per frame, used to preallocate the per-frame draw list
  /// (worst case one draw per glyph: single-character lines). 0 keeps a grow-on-demand
  /// draw list, which may allocate on the frame path.
  uint32_t max_glyphs = 0;
};

/// Create the push-constant monospace renderer behind the same TextRenderer class as
/// create_text_renderer(), so the two are interchangeable at the call site. Requires a
/// device with a 256-byte push-constant budget (NVIDIA hardware); creation fails with a
/// clear error otherwise.
[[nodiscard]] HOLOVIZ_EXPORT expected<TextRenderer> create_monospace_text_renderer(
    rhi::Device& device, const MonospaceTextRendererDesc& desc);

}  // namespace holoviz::text
