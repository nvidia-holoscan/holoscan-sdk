/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holomq/buffer_pool.hpp>

#include <holomq/errors.hpp>
#include <holomq/expected.hpp>
#include "buffer_pool/buffer_pool_impl.hpp"

#include <cassert>
#include <optional>
#include <utility>

namespace holomq {

BufferPool::BufferPool(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {
  assert(impl_ != nullptr);
  assert(static_cast<bool>(*impl_));
}

BufferPool::BufferPool(BufferPool&&) noexcept = default;

BufferPool& BufferPool::operator=(BufferPool&&) noexcept = default;

BufferPool::~BufferPool() = default;

BufferMemoryType BufferPool::memory_type() const noexcept {
  assert(impl_ != nullptr);
  return impl_->memory_type();
}

std::optional<int32_t> BufferPool::cuda_device_ordinal() const noexcept {
  assert(impl_ != nullptr);
  return impl_->cuda_device_ordinal();
}

std::size_t BufferPool::buffer_bytes() const noexcept {
  assert(impl_ != nullptr);
  return impl_->buffer_bytes();
}

std::uint32_t BufferPool::buffer_count() const noexcept {
  assert(impl_ != nullptr);
  return impl_->buffer_count();
}

BufferPool::operator bool() const noexcept {
  return impl_ != nullptr && static_cast<bool>(*impl_);
}

auto BufferPool::acquire() noexcept -> expected<Buffer, Error> {
  assert(impl_ != nullptr);
  if (auto buffer_or = impl_->acquire()) {
    return std::move(buffer_or).value();
  }
  return make_unexpected(Error{ErrorCode::kBackpressured});
}

}  // namespace holomq
