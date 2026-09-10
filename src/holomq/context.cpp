/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holomq/context.hpp>

#include <holomq/device_buffer_allocator.hpp>
#include <holomq/errors.hpp>
#include <holomq/expected.hpp>
#include <holomq/host_buffer_import.hpp>
#include "buffer_pool/device/device_buffer_pool_impl.hpp"
#include "buffer_pool/device/device_const_buffer_impl.hpp"
#include "buffer_pool/host/host_buffer_pool_impl.hpp"
#include "buffer_pool/host/host_const_buffer_impl.hpp"
#include "context_impl.hpp"

#include <holomq/logger.hpp>

#include <cuda.h>

#include <cstddef>
#include <span>
#include <string>
#include <utility>

namespace holomq {

namespace {

[[nodiscard]] const std::string& empty_fd_pass_socket_path() noexcept {
  static const std::string k_empty;
  return k_empty;
}

}  // namespace

Context::Context(std::shared_ptr<ContextImpl> impl) noexcept : impl_(std::move(impl)) {}

bool Context::vmm_fd_export_enabled() const noexcept {
  return impl_ != nullptr && impl_->vmm_fd_export_enabled();
}

const std::string& Context::fd_pass_socket_path() const noexcept {
  if (impl_ == nullptr) {
    return empty_fd_pass_socket_path();
  }
  return impl_->fd_pass_socket_path();
}

PubSubCapabilities Context::pubsub_capabilities() const noexcept {
  return default_pubsub_capabilities();
}

auto Context::import_buffer_unsynchronized(const BufferHandle& handle) noexcept
    -> expected<ConstBuffer, Error> {
  if (impl_ == nullptr) {
    return make_unexpected_nothrow<ConstBuffer>(ErrorCode::kInvalidArgument,
                                                "HoloMQ context is empty");
  }
  switch (handle.memory_type()) {
    case BufferMemoryType::Device:
      if (auto buffer_or = DeviceConstBufferImpl::import(
              handle, impl_->vmm_pool_import_cache(), impl_->ref_count_segment_cache())) {
        return std::move(buffer_or).value();
      }
      return make_unexpected_nothrow<ConstBuffer>(ErrorCode::kNotFound,
                                                  "failed to import device buffer handle");
    case BufferMemoryType::HostPageable:
    case BufferMemoryType::HostPinned:
      if (auto buffer_or = HostConstBufferImpl::import(handle,
                                                       impl_->host_backing_segment_cache(),
                                                       impl_->ref_count_segment_cache(),
                                                       impl_->vmm_pool_import_cache())) {
        return std::move(buffer_or).value();
      }
      return make_unexpected_nothrow<ConstBuffer>(ErrorCode::kNotFound,
                                                  "failed to import host buffer handle");
    case BufferMemoryType::None:
    default:
      return make_unexpected_nothrow<ConstBuffer>(ErrorCode::kInvalidArgument,
                                                  "buffer handle has no memory type");
  }
}

auto Context::import_buffer(const BufferHandle& handle, CUstream_st* const stream) noexcept
    -> expected<ConstBuffer, Error> {
  auto buffer_or = import_buffer_unsynchronized(handle);
  if (!buffer_or.has_value()) {
    return std::move(buffer_or);
  }

  ConstBuffer buffer = std::move(buffer_or).value();
  if (!buffer.enqueue_ready_wait(stream)) {
    return make_unexpected_nothrow<ConstBuffer>(ErrorCode::kInvalidArgument,
                                                "failed to enqueue imported buffer readiness wait");
  }
  return buffer;
}

auto Context::rebind_buffer_unsynchronized(const BufferHandle& handle,
                                           ConstBuffer& destination) noexcept
    -> expected<void, Error> {
  if (impl_ == nullptr) {
    return make_unexpected_nothrow<void>(ErrorCode::kInvalidArgument, "HoloMQ context is empty");
  }
  if (destination) {
    return make_unexpected_nothrow<void>(ErrorCode::kInvalidArgument,
                                         "reusable buffer import is already active");
  }
  if (!destination.impl_) {
    auto imported = import_buffer_unsynchronized(handle);
    if (!imported) {
      return make_unexpected(std::move(imported).error());
    }
    destination = std::move(*imported);
    return {};
  }
  if (!destination.impl_->reacquire(handle)) {
    return make_unexpected_nothrow<void>(ErrorCode::kInvalidArgument,
                                         "buffer handle does not match reusable import pool");
  }
  return {};
}

auto Context::import_host_buffer_unsynchronized(const BufferHandle& handle,
                                                HostBufferImport& destination) noexcept
    -> expected<void, Error> {
  if (impl_ == nullptr) {
    return make_unexpected_nothrow<void>(ErrorCode::kInvalidArgument, "HoloMQ context is empty");
  }
  if (destination.active_) {
    return make_unexpected_nothrow<void>(ErrorCode::kInvalidArgument,
                                         "reusable host import is already active");
  }
  if (handle.memory_type() != BufferMemoryType::HostPageable &&
      handle.memory_type() != BufferMemoryType::HostPinned) {
    return make_unexpected_nothrow<void>(ErrorCode::kInvalidArgument,
                                         "buffer handle is not host memory");
  }

  if (!destination.buffer_.impl_) {
    auto imported = HostConstBufferImpl::import(handle,
                                                impl_->host_backing_segment_cache(),
                                                impl_->ref_count_segment_cache(),
                                                impl_->vmm_pool_import_cache());
    if (!imported) {
      return make_unexpected_nothrow<void>(ErrorCode::kNotFound,
                                           "failed to import reusable host buffer handle");
    }
    destination.buffer_ = std::move(imported).value();
  } else if (!static_cast<HostConstBufferImpl*>(destination.buffer_.impl_.get())
                  ->reacquire(handle, true)) {
    return make_unexpected_nothrow<void>(ErrorCode::kInvalidArgument,
                                         "host buffer handle does not match reusable pool slot");
  }
  destination.active_ = true;
  return {};
}

auto Context::rebind_host_buffer_unsynchronized(const BufferHandle& handle,
                                                HostBufferImport& destination) noexcept
    -> expected<void, Error> {
  if (impl_ == nullptr) {
    return make_unexpected_nothrow<void>(ErrorCode::kInvalidArgument, "HoloMQ context is empty");
  }
  if (destination.active_) {
    return make_unexpected_nothrow<void>(ErrorCode::kInvalidArgument,
                                         "reusable host import is already active");
  }
  if (handle.memory_type() != BufferMemoryType::HostPageable &&
      handle.memory_type() != BufferMemoryType::HostPinned) {
    return make_unexpected_nothrow<void>(ErrorCode::kInvalidArgument,
                                         "buffer handle is not host memory");
  }

  if (!destination.buffer_.impl_) {
    return import_host_buffer_unsynchronized(handle, destination);
  }
  if (!static_cast<HostConstBufferImpl*>(destination.buffer_.impl_.get())
           ->reacquire(handle, false)) {
    return make_unexpected_nothrow<void>(ErrorCode::kInvalidArgument,
                                         "host buffer handle does not match reusable pool");
  }
  destination.active_ = true;
  return {};
}

auto Context::import_host_buffer_unsynchronized(const BufferHandle& handle,
                                                std::span<HostBufferImport> destinations) noexcept
    -> expected<std::size_t, Error> {
  if (handle.memory_type() != BufferMemoryType::HostPageable &&
      handle.memory_type() != BufferMemoryType::HostPinned) {
    return make_unexpected_nothrow<std::size_t>(ErrorCode::kInvalidArgument,
                                                "buffer handle is not host memory");
  }

  const std::size_t destination_index = detail::BufferHandleWireAccess::slot_index(handle);
  if (destination_index >= destinations.size()) {
    return make_unexpected_nothrow<std::size_t>(ErrorCode::kResourceExhausted,
                                                "buffer handle slot exceeds reusable import table");
  }

  auto imported = import_host_buffer_unsynchronized(handle, destinations[destination_index]);
  if (!imported.has_value()) {
    return forward_error(std::move(imported));
  }
  return destination_index;
}

auto Context::create_device_buffer_pool(std::size_t buffer_bytes, std::uint32_t buffer_count,
                                        DeviceBufferAllocator allocator) const noexcept
    -> expected<BufferPool, Error> {
  if (impl_ == nullptr) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  if (!vmm_fd_export_enabled()) {
    HOLOMQ_LOG_ERROR("Context create_device_buffer_pool failed: vmm fd-pass export required");
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }

  return BufferPool::create<DeviceBufferPoolImpl>(
      buffer_bytes, buffer_count, std::move(allocator), impl_->fd_pass_socket_path(), impl_);
}

auto Context::create_host_buffer_pool(std::size_t buffer_bytes, std::uint32_t buffer_count,
                                      HostBufferAllocator allocator) const noexcept
    -> expected<BufferPool, Error> {
  if (impl_ == nullptr) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  if (!vmm_fd_export_enabled()) {
    HOLOMQ_LOG_ERROR("Context create_host_buffer_pool failed: vmm fd-pass export required");
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }

  return BufferPool::create<HostBufferPoolImpl>(buffer_bytes,
                                                buffer_count,
                                                false,
                                                0U,
                                                std::move(allocator),
                                                impl_->fd_pass_socket_path(),
                                                impl_);
}

auto Context::create_host_pinned_buffer_pool(std::size_t buffer_bytes, std::uint32_t buffer_count,
                                             const unsigned int cuda_host_register_flags,
                                             HostBufferAllocator allocator) const noexcept
    -> expected<BufferPool, Error> {
  if (impl_ == nullptr) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }

  if ((cuda_host_register_flags & CU_MEMHOSTREGISTER_DEVICEMAP) == 0U && !vmm_fd_export_enabled()) {
    HOLOMQ_LOG_ERROR(
        "Context create_host_pinned_buffer_pool failed: split-gen VMM layout requires vmm fd-pass "
        "export");
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }

  return BufferPool::create<HostBufferPoolImpl>(buffer_bytes,
                                                buffer_count,
                                                true,
                                                cuda_host_register_flags,
                                                std::move(allocator),
                                                impl_->fd_pass_socket_path(),
                                                impl_);
}

PublisherServiceBuilder Context::publisher_builder(std::string service_name) {
  return PublisherServiceBuilder(impl_ != nullptr ? &impl_->transport() : nullptr,
                                 std::move(service_name));
}

SubscriberServiceBuilder Context::subscriber_builder(std::string service_name) {
  return SubscriberServiceBuilder(impl_ != nullptr ? &impl_->transport() : nullptr,
                                  std::move(service_name));
}

}  // namespace holomq
