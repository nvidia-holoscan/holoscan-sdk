/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "holoviz/rhi/image.hpp"

#include <memory>
#include <utility>

#include "detail/handle_errors.hpp"
#include "detail/image_impl.hpp"

namespace holoviz::rhi {

Image::Image(std::unique_ptr<detail::ImageImpl> impl) : impl_(std::move(impl)) {}
Image::Image() = default;
Image::Image(Image&&) noexcept = default;
Image& Image::operator=(Image&&) noexcept = default;
Image::~Image() = default;

expected<void> Image::destroy(std::chrono::nanoseconds timeout) {
  if (!impl_) {
    return {};  // empty handle: nothing to free
  }
  expected<void> result = impl_->destroy(timeout);
  if (result || result.error().code != ErrorCode::kBackpressured) {
    impl_.reset();
  }
  return result;
}

expected<CUmipmappedArray> Image::cuda_mipmapped_array() const {
  if (!impl_) {
    return detail::null_impl_error("Image::cuda_mipmapped_array");
  }
  return impl_->cuda_mipmapped_array();
}

expected<CUarray> Image::cuda_mipmapped_array_get_level(uint32_t level) const {
  if (!impl_) {
    return detail::null_impl_error("Image::cuda_mipmapped_array_get_level");
  }
  return impl_->cuda_mipmapped_array_get_level(level);
}

expected<Extent2D> Image::extent() const {
  if (!impl_) {
    return detail::null_impl_error("Image::extent");
  }
  return impl_->extent();
}

expected<Format> Image::format() const {
  if (!impl_) {
    return detail::null_impl_error("Image::format");
  }
  return impl_->format();
}

expected<NativeObject> Image::native_object(NativeObjectType type) const {
  if (!impl_) {
    return detail::null_impl_error("Image::native_object");
  }
  return impl_->native_object(type);
}

}  // namespace holoviz::rhi
