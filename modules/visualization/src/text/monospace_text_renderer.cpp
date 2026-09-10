/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "holoviz/text/monospace_text_renderer.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <holoviz/imgui/imgui.h>

#include <holoviz/logger.hpp>
#include <holoviz/rhi/adapter.hpp>
#include <holoviz/rhi/buffer.hpp>
#include <holoviz/rhi/descriptor.hpp>
#include <holoviz/rhi/device.hpp>
#include <holoviz/rhi/image.hpp>
#include <holoviz/rhi/pipeline.hpp>
#include <holoviz/rhi/slang_compiler.hpp>

#include "text/detail/text_renderer_impl.hpp"
#include "text/monospace_shader.hpp"

namespace holoviz::text {

namespace detail {

namespace {

/// Host mirror of the shader's std140 push-constant block: 24-byte header,
/// 8 pad bytes before the uint4[14] char words, 256 bytes total (the NVIDIA
/// push-constant budget this renderer assumes; validated at creation).
struct MonospacePushData {
  float origin_px[2];
  float clip_scale[2];  // patched at record() time from the real framebuffer extent
  float glyph_height_px = 0.0F;
  uint32_t rgba = 0;
  uint32_t reserved[2];       // std140 pad: uint4 array aligns to 16
  uint32_t packed_chars[56];  // uint4[14] = 56 contiguous words = 224 chars
};
static_assert(sizeof(MonospacePushData) == 256, "push block must match the shader's std140 layout");
static_assert(kMonospaceMaxCharsPerDraw == sizeof(MonospacePushData::packed_chars),
              "public constant out of sync with the push block");

/// One shader glyph-table entry; must mirror the Slang GlyphInfo struct
/// (StructuredBuffer natural layout: two float4 = 32 bytes).
struct GlyphInfo {
  float uv[4];    // u0, v0, u1, v1
  float quad[4];  // x0, y0, x1, y1 (bake px, relative to the top-left pen)
};
static_assert(sizeof(GlyphInfo) == 32, "glyph table must match the shader layout");

constexpr uint32_t kFirstGlyphCode = 32;
constexpr uint32_t kLastGlyphCode = 127;
constexpr uint32_t kGlyphTableEntries = kLastGlyphCode - kFirstGlyphCode + 2;  // [0] = metrics

/// Anchor -> fraction of the text bounding box subtracted from the position.
/// (Local copy of the helper in text_renderer.cpp, which is file-local there.)
struct AnchorFactors {
  float x;
  float y;
};

AnchorFactors anchor_factors(Anchor anchor) {
  switch (anchor) {
    case Anchor::kTopLeft:
      return {0.0F, 0.0F};
    case Anchor::kTopCenter:
      return {0.5F, 0.0F};
    case Anchor::kTopRight:
      return {1.0F, 0.0F};
    case Anchor::kMiddleLeft:
      return {0.0F, 0.5F};
    case Anchor::kMiddleCenter:
      return {0.5F, 0.5F};
    case Anchor::kMiddleRight:
      return {1.0F, 0.5F};
    case Anchor::kBottomLeft:
      return {0.0F, 1.0F};
    case Anchor::kBottomCenter:
      return {0.5F, 1.0F};
    case Anchor::kBottomRight:
      return {1.0F, 1.0F};
  }
  return {0.0F, 0.0F};
}

/// Map a text byte to a drawable ASCII code: tabs become spaces, everything
/// outside the printable range (including UTF-8 continuation bytes) becomes
/// '?'. The shader clamps again as a safety net.
uint8_t to_drawable_ascii(char byte) {
  const auto code = static_cast<uint8_t>(byte);
  if (code == '\t') {
    return ' ';
  }
  if (code < kFirstGlyphCode || code > kLastGlyphCode) {
    return '?';
  }
  return code;
}

/// TextRenderer implementation whose only per-frame GPU data is push
/// constants: no buffer uploads, no CUDA stream, no frames-in-flight ring.
class MonospaceTextRendererImpl final : public TextRendererImpl {
 public:
  MonospaceTextRendererImpl(rhi::Device& device, rhi::Image atlas_image, rhi::Buffer glyph_table,
                            rhi::Pipeline pipeline, rhi::Descriptor descriptor,
                            float bake_font_size_px, float advance_bake_px,
                            uint32_t vertices_per_glyph, uint32_t max_glyphs)
      : device_(&device),
        atlas_image_(std::move(atlas_image)),
        glyph_table_(std::move(glyph_table)),
        pipeline_(std::move(pipeline)),
        descriptor_(std::move(descriptor)),
        bake_font_size_px_(bake_font_size_px),
        advance_bake_px_(advance_bake_px),
        vertices_per_glyph_(vertices_per_glyph) {
    // Worst case is one draw per glyph (single-character lines); reserving that here keeps
    // add_text's push_back off the allocator for any frame within the declared budget.
    draws_.reserve(max_glyphs);
  }

  expected<void> destroy(std::chrono::nanoseconds timeout) override {
    frame_open_ = false;
    expected<void> result{};
    auto accumulate = [&result](expected<void> step) {
      if (!step && result) {
        result = std::move(step);
      }
    };
    accumulate(descriptor_.destroy(timeout));
    accumulate(pipeline_.destroy(timeout));
    accumulate(glyph_table_.destroy(timeout));
    accumulate(atlas_image_.destroy(timeout));
    device_ = nullptr;
    return result;
  }

  expected<void> begin_frame([[maybe_unused]] rhi::Extent2D framebuffer_extent,
                             [[maybe_unused]] uint32_t renderer_slot,
                             [[maybe_unused]] rhi::SubmitSyncToken slot_last_use) override {
    // Nothing per frame touches GPU memory, so the slot and reuse token are deliberately
    // unused.
    if (frame_open_) {
      return make_unexpected(Error{
          .code = ErrorCode::kInvalidState,
          .message = "[holoviz] TextRenderer::begin_frame: previous frame missing end_frame",
      });
    }
    draws_.clear();
    frame_open_ = true;
    return {};
  }

  expected<void> add_text(const TextDesc& desc, std::string_view utf8_text) override {
    if (!frame_open_) {
      return make_unexpected(Error{
          .code = ErrorCode::kInvalidState,
          .message = "[holoviz] TextRenderer::add_text called outside begin_frame/end_frame",
      });
    }
    if (utf8_text.empty()) {
      return {};  // nothing to draw
    }
    if (!(desc.size_px > 0.0F) || !std::isfinite(desc.size_px)) {
      return make_unexpected(Error{
          .code = ErrorCode::kInvalidArgument,
          .message = "[holoviz] TextRenderer::add_text: size_px must be positive and finite",
      });
    }

    const float scale = desc.size_px / bake_font_size_px_;
    const float advance_px = advance_bake_px_ * scale;

    // Two allocation-free passes over the text ('\r' dropped for CRLF friendliness).
    // Pass 1 measures the block for anchoring; line count matches ImGui's
    // CalcTextSizeA rules (a trailing '\n' adds a line to the height).
    size_t line_count = 1;
    size_t max_line_length = 0;
    size_t current_line_length = 0;
    for (const char byte : utf8_text) {
      if (byte == '\n') {
        ++line_count;
        current_line_length = 0;
      } else if (byte != '\r') {
        ++current_line_length;
        max_line_length = std::max(max_line_length, current_line_length);
      }
    }
    if (max_line_length == 0) {
      return {};  // only newlines: no visible glyphs
    }

    const AnchorFactors factors = anchor_factors(desc.anchor);
    const float left =
        desc.position.x - static_cast<float>(max_line_length) * advance_px * factors.x;
    const float top = desc.position.y - static_cast<float>(line_count) * desc.size_px * factors.y;

    // Pass 2 emits draws straight from the byte range: one draw per line, chunked at
    // kMonospaceMaxCharsPerDraw characters. `draw` is only pushed once it holds glyphs.
    size_t line_index = 0;
    size_t column = 0;      // glyphs emitted on the current line so far
    size_t chunk_fill = 0;  // glyphs packed into `draw` so far
    PendingDraw draw{};
    const auto start_chunk = [&]() {
      draw = PendingDraw{};
      draw.push.origin_px[0] = left + static_cast<float>(column) * advance_px;
      draw.push.origin_px[1] = top + static_cast<float>(line_index) * desc.size_px;
      draw.push.glyph_height_px = desc.size_px;
      draw.push.rgba = desc.color;
      chunk_fill = 0;
    };
    const auto flush_chunk = [&]() {
      if (chunk_fill > 0) {
        draw.char_count = static_cast<uint32_t>(chunk_fill);
        draws_.push_back(draw);
        chunk_fill = 0;
      }
    };
    for (const char byte : utf8_text) {
      if (byte == '\n') {
        flush_chunk();
        ++line_index;
        column = 0;
        continue;
      }
      if (byte == '\r') {
        continue;
      }
      if (chunk_fill == 0) {
        start_chunk();
      }
      const auto code = static_cast<uint32_t>(to_drawable_ascii(byte));
      draw.push.packed_chars[chunk_fill / 4] |= code << ((chunk_fill % 4) * 8);
      ++chunk_fill;
      ++column;
      if (chunk_fill == kMonospaceMaxCharsPerDraw) {
        flush_chunk();
      }
    }
    flush_chunk();
    return {};
  }

  expected<rhi::ExternalSyncToken> end_frame() override {
    if (!frame_open_) {
      return make_unexpected(Error{
          .code = ErrorCode::kInvalidState,
          .message = "[holoviz] TextRenderer::end_frame called without begin_frame",
      });
    }
    frame_open_ = false;
    // Nothing was uploaded: return the no-op token, which Device::submit() skips.
    return rhi::ExternalSyncToken{};
  }

  expected<void> cancel_frame() override {
    // Everything queued so far is host-side; dropping it closes the frame completely.
    frame_open_ = false;
    draws_.clear();
    return {};
  }

  expected<void> record(rhi::CommandList& command_list, rhi::Extent2D framebuffer_extent) override {
    if (draws_.empty()) {
      return {};  // nothing was drawn this frame
    }
    if (framebuffer_extent.width == 0 || framebuffer_extent.height == 0) {
      return make_unexpected(Error{
          .code = ErrorCode::kInvalidArgument,
          .message = "[holoviz] TextRenderer::record: framebuffer extent is empty",
      });
    }

    const rhi::Rect2D full_extent{0, 0, framebuffer_extent.width, framebuffer_extent.height};
    VIZ_FORWARD_ERROR(command_list.set_viewport(full_extent));
    VIZ_FORWARD_ERROR(command_list.set_scissor(full_extent));
    VIZ_FORWARD_ERROR(command_list.bind_pipeline(pipeline_));
    VIZ_FORWARD_ERROR(command_list.bind_descriptor(descriptor_));

    const float clip_scale_x = 2.0F / static_cast<float>(framebuffer_extent.width);
    const float clip_scale_y = 2.0F / static_cast<float>(framebuffer_extent.height);
    for (PendingDraw& draw : draws_) {
      draw.push.clip_scale[0] = clip_scale_x;
      draw.push.clip_scale[1] = clip_scale_y;
      VIZ_FORWARD_ERROR(command_list.set_push_data(
          pipeline_, std::span{reinterpret_cast<const std::byte*>(&draw.push), sizeof(draw.push)}));
      VIZ_FORWARD_ERROR(command_list.draw(vertices_per_glyph_, draw.char_count));
    }
    return {};
  }

 private:
  struct PendingDraw {
    MonospacePushData push{};
    uint32_t char_count = 0;
  };

  rhi::Device* device_ = nullptr;
  rhi::Image atlas_image_;
  rhi::Buffer glyph_table_;
  rhi::Pipeline pipeline_;
  rhi::Descriptor descriptor_;
  float bake_font_size_px_ = 0.0F;
  float advance_bake_px_ = 0.0F;
  // 6 for the two-triangle path; 3 in fill-rectangle mode, where the first
  // corner-table triangle's bounding rect covers the whole glyph quad.
  uint32_t vertices_per_glyph_ = 6;
  std::vector<PendingDraw> draws_;
  bool frame_open_ = false;
};

}  // namespace

}  // namespace detail

expected<TextRenderer> create_monospace_text_renderer(rhi::Device& device,
                                                      const MonospaceTextRendererDesc& desc) {
  if (!(desc.atlas_glyph_height_px > 0.0F) || !std::isfinite(desc.atlas_glyph_height_px)) {
    return make_unexpected(Error{
        .code = ErrorCode::kInvalidArgument,
        .message = "[holoviz] create_monospace_text_renderer: atlas_glyph_height_px must be "
                   "positive and finite",
    });
  }

  // This renderer packs its whole per-draw payload into a 256-byte push-constant block —
  // NVIDIA's budget, above the 128-byte Vulkan minimum — so validate the assumption up front.
  VIZ_UNWRAP(rhi::CapabilityReport, capabilities, device.capabilities());
  if (capabilities.max_push_data_bytes < sizeof(detail::MonospacePushData)) {
    return make_unexpected(Error{
        .code = ErrorCode::kPushConstantOverflow,
        .message = "[holoviz] create_monospace_text_renderer: this renderer assumes a 256-byte "
                   "push-constant budget (NVIDIA hardware); the device reports " +
                   std::to_string(capabilities.max_push_data_bytes) + " bytes",
    });
  }

  const bool fill_rectangle = desc.fill_mode == MonospaceFillMode::kFillRectangle;
  if (fill_rectangle && !capabilities.fill_rectangle) {
    return make_unexpected(Error{
        .code = ErrorCode::kFailure,
        .message = "[holoviz] create_monospace_text_renderer: MonospaceFillMode::kFillRectangle "
                   "requires VK_NV_fill_rectangle, which this device does not support; check "
                   "Device::capabilities().fill_rectangle before selecting it",
    });
  }

  // Bake the ASCII atlas with a standalone (context-free) ImGui font atlas.
  ImFontAtlas atlas;
  static const ImWchar kAsciiRange[] = {0x0020, 0x007E, 0};
  ImFont* font = nullptr;
  if (desc.ttf_path != nullptr) {
    font =
        atlas.AddFontFromFileTTF(desc.ttf_path, desc.atlas_glyph_height_px, nullptr, kAsciiRange);
    if (font == nullptr) {
      return make_unexpected(Error{
          .code = ErrorCode::kNotFound,
          .message = std::string("[holoviz] create_monospace_text_renderer: failed to load "
                                 "font '") +
                     desc.ttf_path + "'",
      });
    }
  } else {
    // Built-in ProggyClean, a monospace bitmap font; it bakes at 13 px regardless
    // of atlas_glyph_height_px.
    font = atlas.AddFontDefault();
    if (font == nullptr) {
      return make_unexpected(Error{
          .code = ErrorCode::kFailure,
          .message = "[holoviz] create_monospace_text_renderer: failed to load the built-in font",
      });
    }
  }

  unsigned char* atlas_pixels = nullptr;
  int atlas_width = 0;
  int atlas_height = 0;
  atlas.GetTexDataAsRGBA32(&atlas_pixels, &atlas_width, &atlas_height);
  if (atlas_pixels == nullptr || atlas_width <= 0 || atlas_height <= 0) {
    return make_unexpected(Error{
        .code = ErrorCode::kFailure,
        .message = "[holoviz] create_monospace_text_renderer: font atlas rasterization failed",
    });
  }

  // Build the static glyph table: [0] = metrics, [1..96] = ASCII 32..127.
  const ImFontGlyph* fallback = font->FindGlyph(static_cast<ImWchar>('?'));
  std::array<detail::GlyphInfo, detail::kGlyphTableEntries> glyph_table_data{};
  float advance = 0.0F;
  bool proportional = false;
  for (uint32_t code = detail::kFirstGlyphCode; code <= detail::kLastGlyphCode; ++code) {
    const ImFontGlyph* glyph = font->FindGlyph(static_cast<ImWchar>(code));
    if (glyph == nullptr) {
      glyph = fallback;
    }
    if (glyph == nullptr) {
      continue;  // leave the zeroed (invisible) entry
    }
    if (advance != 0.0F && std::fabs(glyph->AdvanceX - advance) > 0.01F) {
      proportional = true;
    }
    advance = std::max(advance, glyph->AdvanceX);
    glyph_table_data[code - detail::kFirstGlyphCode + 1] = {
        {glyph->U0, glyph->V0, glyph->U1, glyph->V1}, {glyph->X0, glyph->Y0, glyph->X1, glyph->Y1}};
  }
  if (!(advance > 0.0F)) {
    return make_unexpected(Error{
        .code = ErrorCode::kFailure,
        .message = "[holoviz] create_monospace_text_renderer: font has no usable glyph advance",
    });
  }
  if (proportional) {
    HOLOVIZ_LOG_WARN(
        "create_monospace_text_renderer: font is not monospace; forcing the widest advance");
  }
  glyph_table_data[0] = {{advance, font->FontSize, font->Ascent, 0.0F}, {0, 0, 0, 0}};

  const rhi::ImageDesc atlas_image_desc{
      .extent = {static_cast<uint32_t>(atlas_width), static_cast<uint32_t>(atlas_height)},
      .format = rhi::Format::kR8G8B8A8Unorm,
      .usage = rhi::ImageUsage::kSampled,
      .debug_name = "monospace font atlas",
  };
  auto atlas_image = device.create_image(atlas_image_desc);
  if (!atlas_image) {
    return forward_error(atlas_image);
  }
  const size_t atlas_row_pitch = static_cast<size_t>(atlas_width) * 4;
  const std::span<const std::byte> atlas_bytes{reinterpret_cast<const std::byte*>(atlas_pixels),
                                               atlas_row_pitch * static_cast<size_t>(atlas_height)};
  VIZ_FORWARD_ERROR(device.upload_to_image(*atlas_image, 0, atlas_bytes, atlas_row_pitch));

  auto glyph_table = device.create_buffer({
      .size_bytes = sizeof(glyph_table_data),
      .share = rhi::ShareMode::kPrivate,
      .debug_name = "monospace glyph table",
  });
  if (!glyph_table) {
    return forward_error(glyph_table);
  }
  const auto glyph_table_slice = glyph_table->whole();
  if (!glyph_table_slice) {
    return forward_error(glyph_table_slice);
  }
  VIZ_FORWARD_ERROR(
      device.upload_to_buffer(*glyph_table_slice,
                              std::span{reinterpret_cast<const std::byte*>(glyph_table_data.data()),
                                        sizeof(glyph_table_data)}));

  VIZ_UNWRAP(rhi::Backend, backend, device.backend());
  VIZ_UNWRAP(rhi::ShaderBlob,
             shader_blob,
             rhi::compile_slang_shader(backend,
                                       detail::kMonospaceVertexEntryPoint,
                                       detail::kMonospaceFragmentEntryPoint,
                                       detail::kMonospaceSlangSource));
  // Guard the two layout invariants the design leans on: the std140 push block
  // size and the zero-attribute (vertex-buffer-less) input.
  if (shader_blob.push_constant_size != sizeof(detail::MonospacePushData)) {
    return make_unexpected(Error{
        .code = ErrorCode::kFailure,
        .message = "[holoviz] create_monospace_text_renderer: Slang push-constant layout "
                   "changed: reflected " +
                   std::to_string(shader_blob.push_constant_size) + " bytes, expected " +
                   std::to_string(sizeof(detail::MonospacePushData)),
    });
  }
  if (!shader_blob.attributes.empty()) {
    return make_unexpected(Error{
        .code = ErrorCode::kFailure,
        .message = "[holoviz] create_monospace_text_renderer: shader unexpectedly reflected "
                   "vertex attributes; the pipeline must be vertex-buffer-less",
    });
  }

  // Text outputs straight (non-premultiplied) alpha; the BlendState defaults compose
  // premultiplied, so set the factors explicitly (same state as the ImGui pipeline).
  const rhi::BlendState straight_alpha_blend{
      .enable = true,
      .src_color = rhi::BlendFactor::kSrcAlpha,
      .dst_color = rhi::BlendFactor::kOneMinusSrcAlpha,
      .src_alpha = rhi::BlendFactor::kOne,
      .dst_alpha = rhi::BlendFactor::kOneMinusSrcAlpha,
  };
  rhi::PipelineDesc pipeline_desc{
      .shader_blob = std::move(shader_blob),
      .topology = rhi::PrimitiveTopology::kTriangleList,
      .raster = {.cull = rhi::CullMode::kNone,
                 .polygon_mode =
                     fill_rectangle ? rhi::PolygonMode::kFillRectangle : rhi::PolygonMode::kFill},
      .depth = {.test = false, .write = false},
      .blend = straight_alpha_blend,
      .debug_name = desc.debug_name != nullptr ? desc.debug_name : "monospace text renderer",
  };
  auto pipeline = device.create_pipeline(pipeline_desc);
  if (!pipeline) {
    return forward_error(pipeline);
  }
  auto descriptor = pipeline->create_descriptor();
  if (!descriptor) {
    return forward_error(descriptor);
  }
  VIZ_FORWARD_ERROR(descriptor->bind_image(*atlas_image, "fontTexture"));
  VIZ_FORWARD_ERROR(descriptor->bind_sampler(device, {}, "fontSampler"));
  VIZ_FORWARD_ERROR(descriptor->bind_buffer(*glyph_table_slice, "glyphTable"));

  // In fill-rectangle mode one triangle per glyph suffices: the first corner-table
  // triangle's axis-aligned bounding rect is the whole glyph quad.
  const uint32_t vertices_per_glyph = fill_rectangle ? 3 : 6;
  return TextRenderer{std::make_unique<detail::MonospaceTextRendererImpl>(device,
                                                                          std::move(*atlas_image),
                                                                          std::move(*glyph_table),
                                                                          std::move(*pipeline),
                                                                          std::move(*descriptor),
                                                                          font->FontSize,
                                                                          advance,
                                                                          vertices_per_glyph,
                                                                          desc.max_glyphs)};
}

}  // namespace holoviz::text
