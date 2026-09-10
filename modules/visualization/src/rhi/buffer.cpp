/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "holoviz/rhi/buffer.hpp"

#include <memory>
#include <utility>

#include "detail/buffer_impl.hpp"
#include "detail/handle_errors.hpp"

namespace holoviz::rhi {

Buffer::Buffer(std::unique_ptr<detail::BufferImpl> impl) : impl_(std::move(impl)) {}
Buffer::Buffer() = default;
Buffer::Buffer(Buffer&&) noexcept = default;
Buffer& Buffer::operator=(Buffer&&) noexcept = default;
Buffer::~Buffer() = default;

expected<void> Buffer::destroy(std::chrono::nanoseconds timeout) {
  if (!impl_) {
    return {};  // empty handle: nothing to free
  }
  expected<void> result = impl_->destroy(timeout);
  if (result || result.error().code != ErrorCode::kBackpressured) {
    impl_.reset();
  }
  return result;
}

expected<BufferSlice> Buffer::whole() const {
  if (!impl_) {
    return detail::null_impl_error("Buffer::whole");
  }
  return impl_->whole();
}

expected<BufferSlice> Buffer::slice(size_t offset_bytes, size_t size_bytes) const {
  if (!impl_) {
    return detail::null_impl_error("Buffer::slice");
  }
  return impl_->slice(offset_bytes, size_bytes);
}

expected<CUdeviceptr> Buffer::cuda_device_ptr() const {
  if (!impl_) {
    return detail::null_impl_error("Buffer::cuda_device_ptr");
  }
  return impl_->cuda_device_ptr();
}

expected<NativeObject> Buffer::native_object(NativeObjectType type) const {
  if (!impl_) {
    return detail::null_impl_error("Buffer::native_object");
  }
  return impl_->native_object(type);
}

}  // namespace holoviz::rhi
