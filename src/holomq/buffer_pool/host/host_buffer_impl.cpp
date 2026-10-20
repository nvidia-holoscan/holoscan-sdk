/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "buffer_pool/host/host_buffer_impl.hpp"

#include <holomq/buffer_handle.hpp>
#include "buffer_pool/host/host_buffer_pool_impl.hpp"
#include "buffer_pool/slot_region.hpp"
#include "buffer_pool/sync/deferred_ref_release.hpp"

#include <cassert>
#include <optional>
#include <utility>

namespace holomq {

namespace {

[[nodiscard]] SlotRegions make_owner_slot_regions(HostBufferPoolImpl& pool) noexcept {
  if (pool.uses_split_generation_vmm()) {
    return make_split_slot_regions(pool.base_ptr(),
                                   pool.slot_stride_bytes(),
                                   pool.generation_base_ptr(),
                                   pool.generation_slot_stride_bytes(),
                                   pool.buffer_bytes());
  }

  const auto regions_or = make_host_devicemap_interleaved_slot_regions(
      pool.base_ptr(), pool.slot_stride_bytes(), pool.buffer_bytes());
  assert(regions_or.has_value());
  if (!regions_or.has_value()) {
    return {};
  }
  return regions_or.value();
}

}  // namespace

ShmRefCountMappingPtr HostBufferImpl::mapping_owner() const noexcept {
  return pool_ != nullptr ? pool_->ref_count_mapping_owner() : nullptr;
}

std::uint64_t HostBufferImpl::next_export_generation() noexcept {
  return pool_ != nullptr ? pool_->next_slot_generation(index()) : 0U;
}

void HostBufferImpl::release_slot_lease() noexcept {
  if (pool_ == nullptr) {
    return;
  }
  if (!release_streams_.empty()) {
    release_streams_.release();
  } else {
    pool_->buffer_handle_released(index());
  }
  pool_ = nullptr;
}

HostBufferImpl::HostBufferImpl(HostBufferPoolImpl& pool, const std::uint32_t index) noexcept
    : Impl(make_owner_slot_regions(pool), index), pool_(&pool) {
  retirement_owners_ = pool.retirement_owners();
}

HostBufferImpl::~HostBufferImpl() {
  release_slot_lease();
}

HostBufferImpl::HostBufferImpl(HostBufferImpl&& other) noexcept : Impl() {
  swap(*this, other);
}

void swap(HostBufferImpl& lhs, HostBufferImpl& rhs) noexcept {
  using std::swap;
  swap(static_cast<Buffer::Impl&>(lhs), static_cast<Buffer::Impl&>(rhs));
  swap(lhs.pool_, rhs.pool_);
  swap(lhs.release_stream_, rhs.release_stream_);
  swap(lhs.defer_release_on_export_, rhs.defer_release_on_export_);
}

HostBufferImpl& HostBufferImpl::operator=(HostBufferImpl&& other) noexcept {
  HostBufferImpl tmp(std::move(other));
  swap(*this, tmp);
  return *this;
}

BufferMemoryType HostBufferImpl::memory_type() const noexcept {
  if (pool_ == nullptr) {
    return BufferMemoryType::None;
  }
  return pool_->payload_pinned() ? BufferMemoryType::HostPinned : BufferMemoryType::HostPageable;
}

std::optional<int32_t> HostBufferImpl::cuda_device_ordinal() const noexcept {
  return std::nullopt;
}

std::optional<BufferHandle> HostBufferImpl::export_handle_unsynchronized() noexcept {
  if (pool_ == nullptr) {
    return std::nullopt;
  }

  const BufferHandle identity = pool_->handle(index());
  if (!identity) {
    return std::nullopt;
  }

  return detail::make_host_buffer_handle(detail::BufferHandleWireAccess::host_wire(identity),
                                         detail::BufferHandleWireAccess::slot_index(identity),
                                         lease_export_generation_);
}

bool HostBufferImpl::enqueue_ready_set(CUstream_st* const stream) noexcept {
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

  if (!register_stream(stream, {})) {
    return false;
  }
  const std::uint64_t generation = record_ready(stream);
  if (generation == 0U) {
    return false;
  }

  release_stream_ = stream;
  defer_release_on_export_ = true;
  return true;
}

bool HostBufferImpl::defer_release_on_stream(CUstream_st* const stream) noexcept {
  if (pool_ == nullptr || !is_usable_stream(stream)) {
    return false;
  }
  if (defer_release_on_export_) {
    return release_stream_ == stream;
  }
  if (!register_stream(stream, {})) {
    return false;
  }
  release_stream_ = stream;
  defer_release_on_export_ = true;
  return true;
}

bool HostBufferImpl::prepare_for_reuse() noexcept {
  if (pool_ == nullptr || pool_->ref_count_table().ref_count(index()) != 1U) {
    return false;
  }
  release_streams_.prepare_for_reuse();
  lease_export_generation_ = 0U;
  release_stream_ = nullptr;
  defer_release_on_export_ = false;
  return true;
}

std::optional<BufferHandle> HostBufferImpl::export_handle(CUstream_st* const stream) noexcept {
  if (!enqueue_ready_set(stream)) {
    return std::nullopt;
  }
  return export_handle_unsynchronized();
}

}  // namespace holomq
