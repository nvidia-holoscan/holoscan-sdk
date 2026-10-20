/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "holoviz/surface.hpp"

#include <memory>
#include <utility>

#include "detail/handle_errors.hpp"
#include "detail/surface_impl.hpp"

namespace holoviz {

Surface::Surface(std::unique_ptr<detail::SurfaceImpl> impl) : impl_(std::move(impl)) {}
Surface::Surface() = default;
Surface::Surface(Surface&&) noexcept = default;
Surface& Surface::operator=(Surface&&) noexcept = default;
Surface::~Surface() = default;

expected<void> Surface::destroy(std::chrono::nanoseconds timeout) {
  if (!impl_) {
    return {};  // empty handle: nothing to free
  }
  expected<void> result = impl_->destroy(timeout);
  if (result || result.error().code != ErrorCode::kBackpressured) {
    impl_.reset();
  }
  return result;
}

expected<Extent2D> Surface::extent() const {
  if (!impl_) {
    return detail::null_impl_error("Surface::extent");
  }
  return impl_->extent();
}

expected<bool> Surface::resizable() const {
  if (!impl_) {
    return detail::null_impl_error("Surface::resizable");
  }
  return impl_->resizable();
}

expected<DisplayAcquisitionMode> Surface::display_acquisition_mode() const {
  if (!impl_) {
    return detail::null_impl_error("Surface::display_acquisition_mode");
  }
  return impl_->display_acquisition_mode();
}

expected<NativeObject> Surface::native_object(NativeObjectType type) const {
  if (!impl_) {
    return detail::null_impl_error("Surface::native_object");
  }
  return impl_->native_object(type);
}

}  // namespace holoviz
