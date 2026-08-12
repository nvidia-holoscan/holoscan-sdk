/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "buffer_pool/device/device_buffer_impl.hpp"

#include <holomq/buffer_handle.hpp>
#include "buffer_pool/device/device_buffer_pool_impl.hpp"
#include "buffer_pool/slot_region.hpp"
#include "buffer_pool/sync/deferred_ref_release.hpp"

#include <optional>
#include <utility>

namespace holomq {

std::uint64_t DeviceBufferImpl::next_export_generation() noexcept {
  return pool_ != nullptr ? pool_->next_slot_generation(index()) : 0U;
}

void DeviceBufferImpl::release_slot_lease() noexcept {
  if (pool_ == nullptr) {
    return;
  }
  if (defer_release_on_export_ && release_stream_ != nullptr) {
    defer_ref_release_on_stream(pool_->ref_count_mapping_owner(), index(), release_stream_);
  } else {
    pool_->buffer_handle_released(index());
  }
  pool_ = nullptr;
}

DeviceBufferImpl::DeviceBufferImpl(DeviceBufferPoolImpl& pool, const std::uint32_t index) noexcept
    : Impl(make_interleaved_slot_regions(pool.base_ptr(), pool.slot_stride_bytes(),
                                         pool.buffer_bytes()),
           index),
      pool_(&pool) {}

DeviceBufferImpl::~DeviceBufferImpl() {
  release_slot_lease();
}

DeviceBufferImpl::DeviceBufferImpl(DeviceBufferImpl&& other) noexcept : Impl() {
  swap(*this, other);
}

void swap(DeviceBufferImpl& lhs, DeviceBufferImpl& rhs) noexcept {
  using std::swap;
  swap(static_cast<Buffer::Impl&>(lhs), static_cast<Buffer::Impl&>(rhs));
  swap(lhs.pool_, rhs.pool_);
  swap(lhs.release_stream_, rhs.release_stream_);
  swap(lhs.defer_release_on_export_, rhs.defer_release_on_export_);
}

DeviceBufferImpl& DeviceBufferImpl::operator=(DeviceBufferImpl&& other) noexcept {
  DeviceBufferImpl tmp(std::move(other));
  swap(*this, tmp);
  return *this;
}

BufferMemoryType DeviceBufferImpl::memory_type() const noexcept {
  return BufferMemoryType::Device;
}

std::optional<int32_t> DeviceBufferImpl::cuda_device_ordinal() const noexcept {
  return pool_ != nullptr ? pool_->cuda_device_ordinal() : std::nullopt;
}

std::optional<BufferHandle> DeviceBufferImpl::export_handle_unsynchronized() noexcept {
  if (pool_ == nullptr) {
    return std::nullopt;
  }

  const BufferHandle identity = pool_->handle(index());
  if (!identity) {
    return std::nullopt;
  }

  return detail::make_device_buffer_handle(detail::BufferHandleWireAccess::device_wire(identity),
                                           detail::BufferHandleWireAccess::slot_index(identity),
                                           lease_export_generation_);
}

bool DeviceBufferImpl::enqueue_ready_set(CUstream_st* const stream) noexcept {
  if (pool_ == nullptr || !is_usable_stream(stream)) {
    return false;
  }
  if (defer_release_on_export_) {
    if (lease_export_generation_ != 0U) {
      return true;
    }
    if (release_stream_ != stream) {
      return false;
    }
  }

  const std::uint64_t generation = record_ready(stream);
  if (generation == 0U) {
    return false;
  }

  release_stream_ = stream;
  defer_release_on_export_ = true;
  return true;
}

bool DeviceBufferImpl::defer_release_on_stream(CUstream_st* const stream) noexcept {
  if (pool_ == nullptr || !is_usable_stream(stream)) {
    return false;
  }
  if (defer_release_on_export_) {
    return release_stream_ == stream;
  }
  release_stream_ = stream;
  defer_release_on_export_ = true;
  return true;
}

bool DeviceBufferImpl::prepare_for_reuse() noexcept {
  if (pool_ == nullptr || pool_->ref_count_table().ref_count(index()) != 1U) {
    return false;
  }
  lease_export_generation_ = 0U;
  release_stream_ = nullptr;
  defer_release_on_export_ = false;
  return true;
}

std::optional<BufferHandle> DeviceBufferImpl::export_handle(CUstream_st* const stream) noexcept {
  if (!enqueue_ready_set(stream)) {
    return std::nullopt;
  }
  return export_handle_unsynchronized();
}

}  // namespace holomq
