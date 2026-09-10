/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Pure-virtual seam behind the TextRenderer handle. Both implementations plug in
// here without any public API change: the ImGui-based one (text_renderer.cpp) and
// the push-constant monospace one (monospace_text_renderer.cpp).

#pragma once

#include <chrono>
#include <cstdint>
#include <string_view>

#include <holoviz/expected.hpp>
#include <holoviz/rhi/command_list.hpp>
#include <holoviz/rhi/common.hpp>
#include <holoviz/text/text_renderer.hpp>

namespace holoviz::text::detail {

class TextRendererImpl {
 public:
  TextRendererImpl() = default;
  TextRendererImpl(const TextRendererImpl&) = delete;
  TextRendererImpl& operator=(const TextRendererImpl&) = delete;
  TextRendererImpl(TextRendererImpl&&) = delete;
  TextRendererImpl& operator=(TextRendererImpl&&) = delete;
  virtual ~TextRendererImpl() = default;

  [[nodiscard]] virtual expected<void> destroy(std::chrono::nanoseconds timeout) = 0;
  [[nodiscard]] virtual expected<void> begin_frame(rhi::Extent2D framebuffer_extent,
                                                   uint32_t renderer_slot,
                                                   rhi::SubmitSyncToken slot_last_use) = 0;
  [[nodiscard]] virtual expected<void> add_text(const TextDesc& desc,
                                                std::string_view utf8_text) = 0;
  [[nodiscard]] virtual expected<rhi::ExternalSyncToken> end_frame() = 0;
  /// Discard the open frame's queued draws and close it without touching the GPU.
  /// Idempotent: success when no frame is open.
  [[nodiscard]] virtual expected<void> cancel_frame() = 0;
  [[nodiscard]] virtual expected<void> record(rhi::CommandList& command_list,
                                              rhi::Extent2D framebuffer_extent) = 0;
};

}  // namespace holoviz::text::detail
