/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "holoviz/text/text_renderer.hpp"

#include <cfloat>
#include <chrono>
#include <memory>
#include <string_view>
#include <utility>

#include "detail/handle_errors.hpp"
#include "imgui/detail/context_guard.hpp"
#include "imgui/renderer_impl.hpp"
#include "text/detail/text_renderer_impl.hpp"

namespace holoviz::text {

namespace detail {

namespace {

/// Anchor -> fraction of the text bounding box subtracted from the position.
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

/// TextRenderer implementation that drives a private ImGui context's background draw list
/// through an owned imgui::Renderer.
class ImGuiTextRendererImpl final : public TextRendererImpl {
 public:
  explicit ImGuiTextRendererImpl(imgui::Renderer backend) : backend_(std::move(backend)) {}

  expected<void> destroy(std::chrono::nanoseconds timeout) override {
    frame_open_ = false;
    return backend_.destroy(timeout);
  }

  expected<void> begin_frame(Extent2D framebuffer_extent, uint32_t renderer_slot,
                             SubmitSyncToken slot_last_use) override {
    if (frame_open_) {
      return make_unexpected(Error{
          .code = ErrorCode::kInvalidState,
          .message = "[holoviz] TextRenderer::begin_frame: previous frame missing end_frame",
      });
    }
    VIZ_FORWARD_ERROR(backend_.begin_frame(renderer_slot, slot_last_use));
    // new_frame() switches the process-global current ImGui context; restore the
    // caller's on every exit path.
    const imgui::detail::ContextGuard context_guard;
    if (expected<void> begun = backend_.impl().new_frame(framebuffer_extent); !begun) {
      // The backend frame opened but the ImGui frame did not; roll the backend back so
      // this failure is not sticky for the next begin_frame.
      (void)backend_.cancel_frame();
      return begun;
    }
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
    VIZ_UNWRAP(ImGuiContext*, context, backend_.context());
    const imgui::detail::ContextGuard context_guard;
    ImGui::SetCurrentContext(context);
    ImFont* font = ImGui::GetFont();
    const char* text_begin = utf8_text.data();
    const char* text_end = text_begin + utf8_text.size();

    ImVec2 position(desc.position.x, desc.position.y);
    const AnchorFactors factors = anchor_factors(desc.anchor);
    if (factors.x != 0.0F || factors.y != 0.0F) {
      const ImVec2 text_size =
          font->CalcTextSizeA(desc.size_px, FLT_MAX, 0.0F, text_begin, text_end);
      position.x -= text_size.x * factors.x;
      position.y -= text_size.y * factors.y;
    }

    ImGui::GetBackgroundDrawList()->AddText(
        font, desc.size_px, position, desc.color, text_begin, text_end);
    return {};
  }

  expected<ExternalSyncToken> end_frame() override {
    if (!frame_open_) {
      return make_unexpected(Error{
          .code = ErrorCode::kInvalidState,
          .message = "[holoviz] TextRenderer::end_frame called without begin_frame",
      });
    }
    frame_open_ = false;
    VIZ_UNWRAP(ImGuiContext*, context, backend_.context());
    const imgui::detail::ContextGuard context_guard;
    ImGui::SetCurrentContext(context);
    ImGui::Render();
    VIZ_UNWRAP(imgui::FrameSync, sync, backend_.upload_draw_data(ImGui::GetDrawData()));
    return sync.upload_done;
  }

  expected<void> record(CommandList& command_list, Extent2D framebuffer_extent) override {
    return backend_.record_draws(command_list, framebuffer_extent);
  }

  expected<void> cancel_frame() override {
    if (!frame_open_) {
      return {};  // idempotent: nothing open
    }
    frame_open_ = false;
    VIZ_UNWRAP(ImGuiContext*, context, backend_.context());
    const imgui::detail::ContextGuard context_guard;
    ImGui::SetCurrentContext(context);
    // Close the NewFrame() begin_frame ran, without Render()/upload. The stream-side
    // semaphore wait begin_frame enqueued targets an already-passed timeline value and
    // is harmless to leave behind.
    ImGui::EndFrame();
    return backend_.cancel_frame();
  }

 private:
  imgui::Renderer backend_;
  bool frame_open_ = false;
};

}  // namespace

}  // namespace detail

TextRenderer::TextRenderer(std::unique_ptr<detail::TextRendererImpl> impl)
    : impl_(std::move(impl)) {}
TextRenderer::TextRenderer() = default;
TextRenderer::TextRenderer(TextRenderer&&) noexcept = default;
TextRenderer& TextRenderer::operator=(TextRenderer&&) noexcept = default;
TextRenderer::~TextRenderer() = default;

expected<void> TextRenderer::destroy(std::chrono::nanoseconds timeout) {
  if (!impl_) {
    return {};  // empty handle: nothing to free
  }
  expected<void> result = impl_->destroy(timeout);
  if (result || result.error().code != ErrorCode::kBackpressured) {
    impl_.reset();
  }
  return result;
}

expected<void> TextRenderer::begin_frame(Extent2D framebuffer_extent, uint32_t renderer_slot,
                                         SubmitSyncToken slot_last_use) {
  if (!impl_) {
    return holoviz::detail::null_impl_error("TextRenderer::begin_frame");
  }
  return impl_->begin_frame(framebuffer_extent, renderer_slot, slot_last_use);
}

expected<void> TextRenderer::add_text(const TextDesc& desc, std::string_view utf8_text) {
  if (!impl_) {
    return holoviz::detail::null_impl_error("TextRenderer::add_text");
  }
  return impl_->add_text(desc, utf8_text);
}

expected<ExternalSyncToken> TextRenderer::end_frame() {
  if (!impl_) {
    return holoviz::detail::null_impl_error("TextRenderer::end_frame");
  }
  return impl_->end_frame();
}

expected<void> TextRenderer::cancel_frame() {
  if (!impl_) {
    return holoviz::detail::null_impl_error("TextRenderer::cancel_frame");
  }
  return impl_->cancel_frame();
}

expected<void> TextRenderer::record(CommandList& command_list, Extent2D framebuffer_extent) {
  if (!impl_) {
    return holoviz::detail::null_impl_error("TextRenderer::record");
  }
  return impl_->record(command_list, framebuffer_extent);
}

expected<TextRenderer> create_text_renderer(Device& device, const TextRendererDesc& desc) {
  const imgui::RendererDesc backend_desc{
      .stream = desc.stream,
      .frames_in_flight = desc.frames_in_flight,
      .ttf_path = desc.ttf_path,
      .ttf_size_px = desc.base_font_size_px,
      .debug_name = desc.debug_name != nullptr ? desc.debug_name : "text renderer",
  };
  VIZ_UNWRAP(imgui::Renderer, backend, imgui::create_renderer(device, backend_desc));
  return TextRenderer{std::make_unique<detail::ImGuiTextRendererImpl>(std::move(backend))};
}

}  // namespace holoviz::text
