/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holomq/const_buffer.hpp>

#include <holomq/errors.hpp>
#include <holomq/expected.hpp>
#include <holomq/host_buffer_import.hpp>
#include "buffer_pool/const_buffer_impl.hpp"
#include "buffer_pool/host/host_const_buffer_impl.hpp"

#include <utility>

namespace holomq {

ConstBuffer::ConstBuffer() noexcept = default;

ConstBuffer::ConstBuffer(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}

ConstBuffer::ConstBuffer(ConstBuffer&&) noexcept = default;
ConstBuffer& ConstBuffer::operator=(ConstBuffer&&) noexcept = default;
ConstBuffer::~ConstBuffer() = default;

BufferMemoryType ConstBuffer::memory_type() const noexcept {
  return impl_ ? impl_->memory_type() : BufferMemoryType::None;
}

std::optional<int32_t> ConstBuffer::cuda_device_ordinal() const noexcept {
  return impl_ ? impl_->cuda_device_ordinal() : std::nullopt;
}

std::size_t ConstBuffer::size_bytes() const noexcept {
  return impl_ && impl_->active() ? impl_->size_bytes() : 0U;
}

ConstBuffer::operator bool() const noexcept {
  return static_cast<bool>(impl_) && impl_->active() && static_cast<bool>(*impl_);
}

const void* ConstBuffer::ptr() const noexcept {
  return impl_ && impl_->active() ? impl_->ptr() : nullptr;
}

bool ConstBuffer::has_producer_readiness_fence() const noexcept {
  return impl_ && impl_->active() && impl_->has_producer_readiness_fence();
}

bool ConstBuffer::enqueue_ready_wait(CUstream_st* const stream) noexcept {
  return impl_ ? impl_->enqueue_ready_wait(stream) : false;
}

expected<BufferHandle, Error> ConstBuffer::export_handle() noexcept {
  if (!impl_) {
    return make_unexpected_nothrow<BufferHandle>(ErrorCode::kInvalidArgument,
                                                 "imported buffer lease is empty");
  }
  if (auto handle_or = impl_->export_handle()) {
    return std::move(handle_or).value();
  }
  return make_unexpected_nothrow<BufferHandle>(ErrorCode::kFailure,
                                               "failed to re-export imported buffer handle");
}

void ConstBuffer::release() noexcept {
  if (impl_) {
    impl_->release_for_reuse();
  }
}

HostBufferImport::HostBufferImport() noexcept = default;

HostBufferImport::HostBufferImport(HostBufferImport&& other) noexcept
    : buffer_(std::move(other.buffer_)), active_(std::exchange(other.active_, false)) {}

HostBufferImport& HostBufferImport::operator=(HostBufferImport&& other) noexcept {
  if (this != &other) {
    release();
    buffer_ = std::move(other.buffer_);
    active_ = std::exchange(other.active_, false);
  }
  return *this;
}

HostBufferImport::~HostBufferImport() {
  release();
}

HostBufferImport::operator bool() const noexcept {
  return active_ && static_cast<bool>(buffer_);
}

BufferMemoryType HostBufferImport::memory_type() const noexcept {
  return active_ ? buffer_.memory_type() : BufferMemoryType::None;
}

std::size_t HostBufferImport::size_bytes() const noexcept {
  return active_ ? buffer_.size_bytes() : 0U;
}

const void* HostBufferImport::ptr() const noexcept {
  return active_ ? buffer_.ptr() : nullptr;
}

std::optional<std::int32_t> HostBufferImport::cuda_device_ordinal() const noexcept {
  return active_ ? buffer_.cuda_device_ordinal() : std::nullopt;
}

bool HostBufferImport::enqueue_ready_wait(CUstream_st* const stream) noexcept {
  return active_ && buffer_.enqueue_ready_wait(stream);
}

expected<BufferHandle, Error> HostBufferImport::export_handle() noexcept {
  if (!active_) {
    return make_unexpected_nothrow<BufferHandle>(ErrorCode::kInvalidArgument,
                                                 "reusable host import is inactive");
  }
  return buffer_.export_handle();
}

void HostBufferImport::release() noexcept {
  if (!active_ || !buffer_.impl_) {
    active_ = false;
    return;
  }
  static_cast<HostConstBufferImpl*>(buffer_.impl_.get())->release_view_lease();
  active_ = false;
}

}  // namespace holomq
