/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "holoviz/render_pass.hpp"

#include <memory>
#include <utility>

#include "detail/handle_errors.hpp"
#include "detail/render_pass_impl.hpp"

namespace holoviz {

RenderPass::RenderPass(std::unique_ptr<detail::RenderPassImpl> impl) : impl_(std::move(impl)) {}
RenderPass::RenderPass() = default;
RenderPass::RenderPass(RenderPass&&) noexcept = default;
RenderPass& RenderPass::operator=(RenderPass&&) noexcept = default;
RenderPass::~RenderPass() = default;

expected<void> RenderPass::destroy(std::chrono::nanoseconds timeout) {
  if (!impl_) {
    return {};  // empty handle: nothing to free
  }
  expected<void> result = impl_->destroy(timeout);
  if (result || result.error().code != ErrorCode::kBackpressured) {
    impl_.reset();
  }
  return result;
}

expected<void> RenderPass::set_render_image(const Image& image, uint32_t attachment_index,
                                            Color clear_value) {
  if (!impl_) {
    return detail::null_impl_error("RenderPass::set_render_image");
  }
  if (!image) {
    return detail::null_arg_error("RenderPass::set_render_image", "image");
  }
  return impl_->set_render_image(image.impl(), attachment_index, clear_value);
}

expected<void> RenderPass::prepare_render_target(const Image& image) {
  if (!impl_) {
    return detail::null_impl_error("RenderPass::prepare_render_target");
  }
  if (!image) {
    return detail::null_arg_error("RenderPass::prepare_render_target", "image");
  }
  return impl_->prepare_render_target(image.impl());
}

expected<void> RenderPass::set_depth_image(const Image& image, float clear_value) {
  if (!impl_) {
    return detail::null_impl_error("RenderPass::set_depth_image");
  }
  if (!image) {
    return detail::null_arg_error("RenderPass::set_depth_image", "image");
  }
  return impl_->set_depth_image(image.impl(), clear_value);
}

expected<NativeObject> RenderPass::native_object(NativeObjectType type) const {
  if (!impl_) {
    return detail::null_impl_error("RenderPass::native_object");
  }
  return impl_->native_object(type);
}

}  // namespace holoviz
