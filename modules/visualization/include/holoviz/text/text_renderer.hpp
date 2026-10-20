/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// TextRenderer: convenience per-frame text drawing on top of HoloViz. The 5.0
// implementation drives a private Dear ImGui context through imgui::Renderer
// (create_text_renderer); the class itself is a handle over a pure-virtual
// impl seam, so the push-constant monospace renderer (monospace_text_renderer.hpp,
// create_monospace_text_renderer) slots in behind the same public API.

#pragma once

#include <chrono>
#include <cstdint>
#include <memory>
#include <string_view>

#include <cuda.h>

#include <holoviz/holoviz_export.h>

#include <holoviz/command_list.hpp>
#include <holoviz/common.hpp>
#include <holoviz/expected.hpp>

namespace holoviz {
class Device;
}  // namespace holoviz

namespace holoviz::text {

/// Which point of the text's bounding box TextDesc::position refers to.
enum class Anchor : uint8_t {
  kTopLeft,
  kTopCenter,
  kTopRight,
  kMiddleLeft,
  kMiddleCenter,
  kMiddleRight,
  kBottomLeft,
  kBottomCenter,
  kBottomRight,
};

struct TextDesc {
  /// Anchor point in framebuffer pixels (origin top-left; HiDPI callers scale by
  /// Window::content_scale() themselves).
  float2 position;
  /// Glyph height in pixels. Sizes above the rasterized atlas size scale bitmaps up and soften;
  /// see TextRendererDesc::base_font_size_px.
  float size_px = 16.0F;
  /// Packed 8-bit RGBA with red in the lowest byte (0xAABBGGRR, ImGui's IM_COL32 order).
  uint32_t color = 0xFFFFFFFFu;
  /// Which point of the text's bounding box \p position places (see Anchor).
  Anchor anchor = Anchor::kTopLeft;
};

struct TextRendererDesc {
  /// Stream all per-frame host-to-device copies and semaphore operations run on. Same contract
  /// as imgui::RendererDesc::stream: must be non-null (creation fails with kInvalidArgument
  /// otherwise) and must belong to the device's CUDA context.
  CUstream stream = nullptr;
  /// Buffer ring depth; swapchain image count + 1 is a safe choice. Callers keep one
  /// SubmitSyncToken per slot for begin_frame (see imgui::Renderer::begin_frame).
  uint32_t frames_in_flight = 4;
  /// Optional TTF font; nullptr keeps the ImGui default font (ProggyClean, rasterized at 13 px).
  const char* ttf_path = nullptr;
  /// Atlas rasterization size for \p ttf_path; add_text sizes scale from it.
  float base_font_size_px = 32.0F;
  /// Optional label used for the renderer's pipeline and error messages; nullptr selects a
  /// default.
  const char* debug_name = nullptr;
};

namespace detail {
class TextRendererImpl;
}  // namespace detail

/// Per-frame text drawing. Frame flow:
///
///   text.begin_frame(extent, frame, slot_tokens[frame % frames_in_flight]);
///   text.add_text({...}, "hello");                 // any number of times, host-side only
///   auto token = text.end_frame();                 // bakes geometry + async CUDA upload
///   ... caller: begin()/begin_render() ...
///   text.record(cl, framebuffer_extent);           // inside the caller's render scope
///   ... caller: submit with the token in wait_external_sync_tokens ...
///   slot_tokens[frame % frames_in_flight] = submit_token;  // the slot's next reuse gate
///
/// The Device must outlive the renderer and stay at a stable address; destroy the renderer
/// before the device.
class TextRenderer {
 public:
  HOLOVIZ_EXPORT explicit TextRenderer(std::unique_ptr<detail::TextRendererImpl> impl);
  HOLOVIZ_EXPORT TextRenderer();
  HOLOVIZ_EXPORT TextRenderer(TextRenderer&&) noexcept;
  HOLOVIZ_EXPORT TextRenderer& operator=(TextRenderer&&) noexcept;
  TextRenderer(const TextRenderer&) = delete;
  TextRenderer& operator=(const TextRenderer&) = delete;
  HOLOVIZ_EXPORT ~TextRenderer();

  explicit operator bool() const { return impl_ != nullptr; }

  /// Release the underlying resources now instead of at destruction. The GPU must be done with
  /// the renderer's draws (e.g. Device::wait_idle()) before calling this. On success the handle
  /// becomes empty. Safe to call on an empty handle (no-op success). The destructor calls this
  /// automatically and logs any error.
  [[nodiscard]] HOLOVIZ_EXPORT expected<void> destroy(
      std::chrono::nanoseconds timeout = std::chrono::nanoseconds::zero());

  /// Start a frame in buffer slot `renderer_slot % frames_in_flight`. \p slot_last_use is the
  /// token returned by the submit that last consumed this renderer's draws for the same slot
  /// (a default token on the first lap) — keyed by the frame counter, NOT by swapchain image;
  /// see imgui::Renderer::begin_frame for the exact contract.
  [[nodiscard]] HOLOVIZ_EXPORT expected<void> begin_frame(Extent2D framebuffer_extent,
                                                          uint32_t renderer_slot,
                                                          SubmitSyncToken slot_last_use);

  /// Queue one run of UTF-8 text for this frame. Host-side only; nothing touches the GPU until
  /// end_frame().
  [[nodiscard]] HOLOVIZ_EXPORT expected<void> add_text(const TextDesc& desc,
                                                       std::string_view utf8_text);

  /// Finish the frame's text and start the asynchronous GPU upload. Add the returned token to
  /// the consuming SubmitDesc::wait_external_sync_tokens.
  [[nodiscard]] HOLOVIZ_EXPORT expected<ExternalSyncToken> end_frame();

  /// Abandon an open frame: discard the draws queued since begin_frame() and close the frame
  /// without touching the GPU, so a later begin_frame() starts clean after a mid-frame error
  /// (e.g. a rejected add_text). Idempotent — success when no frame is open.
  [[nodiscard]] HOLOVIZ_EXPORT expected<void> cancel_frame();

  /// Record this frame's draws. Must be called inside the caller's begin_render/end_render
  /// scope, after end_frame(). Sets viewport, scissor, pipeline, bindings, and push data —
  /// re-set any caller state afterwards.
  [[nodiscard]] HOLOVIZ_EXPORT expected<void> record(CommandList& command_list,
                                                     Extent2D framebuffer_extent);

  /// Backend-internal access; not part of the stable API surface.
  /// Precondition: the handle is non-empty.
  detail::TextRendererImpl& impl() { return *impl_; }
  const detail::TextRendererImpl& impl() const { return *impl_; }

 private:
  std::unique_ptr<detail::TextRendererImpl> impl_;
};

/// Create a text renderer that drives a private Dear ImGui context through an internal
/// imgui::Renderer. The device must outlive the renderer and stay at a stable address.
[[nodiscard]] HOLOVIZ_EXPORT expected<TextRenderer> create_text_renderer(
    Device& device, const TextRendererDesc& desc);

}  // namespace holoviz::text
