/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <chrono>

#include <holoviz/render_pass.hpp>

namespace holoviz::detail {

class ImageImpl;

/// Backend interface for RenderPass; see the RenderPass handle for API docs.
class RenderPassImpl {
 public:
  RenderPassImpl(const RenderPassImpl&) = delete;
  RenderPassImpl& operator=(const RenderPassImpl&) = delete;
  RenderPassImpl(RenderPassImpl&&) = delete;
  RenderPassImpl& operator=(RenderPassImpl&&) = delete;
  virtual ~RenderPassImpl() = default;

  /// Release all owned backend resources now. \p timeout bounds how long to
  /// wait for the object to stop being in use: zero (the default) fails
  /// immediately with ErrorCode::kBackpressured if the object is in use, and a
  /// non-zero timeout waits up to that duration before returning
  /// ErrorCode::kBackpressured. Backends without in-use tracking ignore the
  /// timeout and destroy immediately. Idempotent: repeated calls are no-ops
  /// returning success. Called by the destructor if not already destroyed.
  [[nodiscard]] virtual expected<void> destroy(std::chrono::nanoseconds timeout) = 0;

  [[nodiscard]] virtual expected<void> set_render_image(const ImageImpl& image,
                                                        uint32_t attachment_index,
                                                        Color clear_value) = 0;
  [[nodiscard]] virtual expected<void> set_depth_image(const ImageImpl& image,
                                                       float clear_value) = 0;
  /// See RenderPass::prepare_render_target. Default: no-op success for backends without
  /// per-target caching.
  [[nodiscard]] virtual expected<void> prepare_render_target(const ImageImpl& /*image*/) {
    return {};
  }
  [[nodiscard]] virtual expected<NativeObject> native_object(NativeObjectType type) const = 0;

  /// Cumulative native per-target creations (Vulkan: vkCreateFramebuffer calls) —
  /// instrumentation for the allocation-free-frame tests; backends increment
  /// render_targets_created_ on every creation, cached hits do not count.
  [[nodiscard]] uint64_t render_targets_created() const { return render_targets_created_; }

 protected:
  RenderPassImpl() = default;

  uint64_t render_targets_created_ = 0;
};

}  // namespace holoviz::detail
