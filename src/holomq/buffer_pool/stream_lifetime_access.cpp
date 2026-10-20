/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <memory>

#include "buffer_pool/stream_lifetime_access.hpp"

#include "buffer_pool/buffer_impl.hpp"
#include "buffer_pool/const_buffer_impl.hpp"

namespace holomq {

bool BufferStreamAccess::register_consumer(
    Buffer& buffer, CUstream_st* stream,
    const std::shared_ptr<const cuda::StreamLifetime>& lifetime) noexcept {
  return buffer.impl_ != nullptr && buffer.impl_->register_stream(stream, lifetime);
}

bool BufferStreamAccess::register_consumer(
    ConstBuffer& buffer, CUstream_st* stream,
    const std::shared_ptr<const cuda::StreamLifetime>& lifetime) noexcept {
  return buffer.impl_ != nullptr && buffer.impl_->active() &&
         buffer.impl_->register_stream(stream, lifetime);
}

bool BufferStreamAccess::wait(
    ConstBuffer& buffer, CUstream_st* stream,
    const std::shared_ptr<const cuda::StreamLifetime>& lifetime) noexcept {
  if (buffer.impl_ == nullptr || !buffer.impl_->active() ||
      !buffer.impl_->register_stream(stream, lifetime)) {
    return false;
  }
  auto admission = lifetime != nullptr ? lifetime->try_pin() : cuda::StreamLifetime::PinResult{};
  // Cache lookup inspects capture status, so protect it with the same pin as a new wait.
  return (lifetime == nullptr || admission.pin) &&
         (buffer.impl_->ready_on(stream) || buffer.enqueue_ready_wait(stream));
}

expected<BufferHandle, Error> BufferStreamAccess::export_handle(
    Buffer& buffer, CUstream_st* stream,
    const std::shared_ptr<const cuda::StreamLifetime>& lifetime) noexcept {
  if (lifetime != nullptr && lifetime->state() == cuda::StreamLifetime::State::kRetired) {
    return buffer.export_handle_unsynchronized();
  }
  if (!register_consumer(buffer, stream, lifetime)) {
    return make_unexpected(Error{ErrorCode::kFailure});
  }
  auto admission = lifetime != nullptr ? lifetime->try_pin() : cuda::StreamLifetime::PinResult{};
  if (lifetime != nullptr && !admission.pin) {
    return make_unexpected(Error{ErrorCode::kFailure});
  }
  return buffer.export_handle(stream);
}

}  // namespace holomq
