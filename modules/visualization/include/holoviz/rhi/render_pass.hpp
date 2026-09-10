/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// RenderPass: a format-signature object plus a mutable attachment table.
// Creation fixes only the attachment *formats*; the actual images are pointed
// in later via set_render_image()/set_depth_image() and take effect at the
// next CommandList::begin_render(). Framebuffers are cached internally per
// attachment set, so re-pointing at the same images (e.g. cycling swapchain
// images) is free after the first frame. This is the classic pass-object
// lowering.

#pragma once

#include <chrono>
#include <memory>
#include <vector>

#include <holoviz/holoviz_export.h>

#include "common.hpp"
#include "image.hpp"

namespace holoviz::rhi {

// LoadOp/StoreOp are not yet wired up: 5.0 always clears color targets on
// load and stores on end (see the commented-out desc fields below).
enum class LoadOp : uint8_t { kLoad, kClear, kDontCare };

enum class StoreOp : uint8_t { kStore, kDontCare };

struct RenderTargetDesc {
  // uint32_t mip = 0;
  // uint32_t layer = 0;
  // LoadOp load = LoadOp::kClear;
  // StoreOp store = StoreOp::kStore;
  Format format = Format::kUnknown;
  // Color clear_value = {0.0F, 0.0F, 0.0F, 1.0F};
};

struct DepthTargetDesc {
  Format format = Format::kUnknown;
  // uint32_t mip = 0;
  // uint32_t layer = 0;
  // LoadOp load = LoadOp::kClear;
  // StoreOp store = StoreOp::kStore;
  // float clear_depth = 1.0F;
};

struct RenderPassDesc {
  std::vector<RenderTargetDesc> render_targets;
  DepthTargetDesc depth_target;
};

namespace detail {
class RenderPassImpl;
}

/// A render pass with pluggable attachments. 5.0 color attachments end the
/// pass in present-ready state — passes are geared toward rendering into
/// swapchain images (use the swapchain's format() when creating the pass).
class RenderPass {
 public:
  HOLOVIZ_EXPORT explicit RenderPass(std::unique_ptr<detail::RenderPassImpl> impl);
  HOLOVIZ_EXPORT RenderPass();
  HOLOVIZ_EXPORT RenderPass(RenderPass&&) noexcept;
  HOLOVIZ_EXPORT RenderPass& operator=(RenderPass&&) noexcept;
  RenderPass(const RenderPass&) = delete;
  RenderPass& operator=(const RenderPass&) = delete;
  HOLOVIZ_EXPORT ~RenderPass();

  explicit operator bool() const { return impl_ != nullptr; }

  /// Release the underlying backend resources now instead of at destruction.
  /// \p timeout bounds how long to wait for the object to stop being in use:
  /// zero (the default) fails immediately with ErrorCode::kBackpressured if
  /// the object is in use; a non-zero timeout waits up to that duration before
  /// returning ErrorCode::kBackpressured. Only CommandList currently tracks
  /// in-use state; other objects destroy immediately. On success (and on any
  /// error other than kBackpressured) the handle becomes empty. Safe to call
  /// on an empty handle (no-op success). The destructor calls this
  /// automatically and logs any error.
  [[nodiscard]] HOLOVIZ_EXPORT expected<void> destroy(
      std::chrono::nanoseconds timeout = std::chrono::nanoseconds::zero());

  /// Point an attachment slot at an image and set its clear color. Borrowed:
  /// the image must stay alive while it is set. Takes effect at the next
  /// begin_render on this pass.
  [[nodiscard]] HOLOVIZ_EXPORT expected<void> set_render_image(const Image& image,
                                                               uint32_t attachment_index,
                                                               Color clear_value);

  /// Create and cache this pass's per-target state (Vulkan: the framebuffer) for
  /// rendering into \p image as attachment 0 at the image's full extent, now, on a setup
  /// path — so the next begin_render on this target performs no creation and the frame
  /// loop stays allocation-free. Idempotent per live target; cached state is evicted
  /// automatically when the image's view is destroyed. Backends without per-target
  /// caching treat this as a no-op success.
  [[nodiscard]] HOLOVIZ_EXPORT expected<void> prepare_render_target(const Image& image);
  [[nodiscard]] HOLOVIZ_EXPORT expected<void> set_depth_image(const Image& image,
                                                              float clear_value);

  /// Vulkan: answers kVkRenderPass.
  [[nodiscard]] HOLOVIZ_EXPORT expected<NativeObject> native_object(NativeObjectType type) const;

  /// Backend-internal access; not part of the stable API surface.
  /// Precondition: the handle is non-empty.
  detail::RenderPassImpl& impl() { return *impl_; }
  const detail::RenderPassImpl& impl() const { return *impl_; }

 private:
  std::unique_ptr<detail::RenderPassImpl> impl_;
};

}  // namespace holoviz::rhi
