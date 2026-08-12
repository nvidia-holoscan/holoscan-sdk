/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holomq/buffer.hpp>

#include <holomq/errors.hpp>
#include <holomq/expected.hpp>
#include "buffer_pool/buffer_impl.hpp"

#include <utility>

namespace holomq {

Buffer::Buffer() noexcept = default;

Buffer::Buffer(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}

Buffer::Buffer(Buffer&&) noexcept = default;
Buffer& Buffer::operator=(Buffer&&) noexcept = default;
Buffer::~Buffer() = default;

BufferMemoryType Buffer::memory_type() const noexcept {
  return impl_ ? impl_->memory_type() : BufferMemoryType::None;
}

std::optional<int32_t> Buffer::cuda_device_ordinal() const noexcept {
  return impl_ ? impl_->cuda_device_ordinal() : std::nullopt;
}

std::size_t Buffer::size_bytes() const noexcept {
  return impl_ ? impl_->size_bytes() : 0U;
}

Buffer::operator bool() const noexcept {
  return static_cast<bool>(impl_) && static_cast<bool>(*impl_);
}

void* Buffer::ptr() noexcept {
  return impl_ ? impl_->ptr() : nullptr;
}

const void* Buffer::ptr() const noexcept {
  return impl_ ? impl_->ptr() : nullptr;
}

expected<BufferHandle, Error> Buffer::export_handle(CUstream_st* const stream) noexcept {
  if (!impl_) {
    return make_unexpected_nothrow<BufferHandle>(ErrorCode::kInvalidArgument,
                                                 "buffer lease is empty");
  }
  if (auto handle_or = impl_->export_handle(stream)) {
    return std::move(handle_or).value();
  }
  return make_unexpected_nothrow<BufferHandle>(ErrorCode::kFailure,
                                               "failed to export buffer handle");
}

expected<BufferHandle, Error> Buffer::export_handle_unsynchronized() noexcept {
  if (!impl_) {
    return make_unexpected_nothrow<BufferHandle>(ErrorCode::kInvalidArgument,
                                                 "buffer lease is empty");
  }
  if (auto handle_or = impl_->export_handle_unsynchronized()) {
    return std::move(handle_or).value();
  }
  return make_unexpected_nothrow<BufferHandle>(ErrorCode::kFailure,
                                               "failed to export buffer handle without a stream");
}

bool Buffer::enqueue_ready_set(CUstream_st* const stream) noexcept {
  return impl_ != nullptr && impl_->enqueue_ready_set(stream);
}

bool Buffer::defer_release_on_stream(CUstream_st* const stream) noexcept {
  return impl_ != nullptr && impl_->defer_release_on_stream(stream);
}

bool Buffer::prepare_for_reuse() noexcept {
  return impl_ != nullptr && impl_->prepare_for_reuse();
}

}  // namespace holomq
