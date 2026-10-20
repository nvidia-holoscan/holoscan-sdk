/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "buffer_pool/host/host_const_buffer_impl.hpp"

#include <holomq/buffer_handle.hpp>
#include "buffer_pool/ref_count/shm_ref_count.hpp"
#include "buffer_pool/slot_region.hpp"
#include "buffer_pool/sync/deferred_ref_release.hpp"
#include "vmm_pool_import_cache.hpp"

#include <cuda.h>

#include <cstring>
#include <memory>
#include <new>
#include <optional>
#include <utility>

namespace holomq {

namespace {

[[nodiscard]] bool is_adoptable_handle_data(
    const detail::host::BufferHandleData& handle_data) noexcept {
  return detail::is_valid_host_buffer_wire(handle_data);
}

[[nodiscard]] void* generation_pool_base_ptr(const CUdeviceptr base) noexcept {
  return reinterpret_cast<void*>(static_cast<std::uintptr_t>(base));
}

[[nodiscard]] bool same_host_pool(const detail::host::BufferHandleData& lhs,
                                  const detail::host::BufferHandleData& rhs) noexcept {
  return lhs.owner_pid == rhs.owner_pid && lhs.shm_pool_id == rhs.shm_pool_id &&
         lhs.shm_ref_count_id == rhs.shm_ref_count_id &&
         lhs.slot_stride_bytes == rhs.slot_stride_bytes && lhs.buffer_bytes == rhs.buffer_bytes &&
         lhs.pool_total_bytes == rhs.pool_total_bytes && lhs.payload_pinned == rhs.payload_pinned &&
         lhs.host_register_flags == rhs.host_register_flags &&
         lhs.gen_pool_cookie == rhs.gen_pool_cookie &&
         lhs.gen_slot_stride_bytes == rhs.gen_slot_stride_bytes &&
         lhs.gen_pool_total_bytes == rhs.gen_pool_total_bytes &&
         std::memcmp(lhs.fd_pass_socket_path,
                     rhs.fd_pass_socket_path,
                     sizeof(lhs.fd_pass_socket_path)) == 0;
}

}  // namespace

auto HostConstBufferImpl::import(const BufferHandle& handle, HostBackingSegmentCache& backing_cache,
                                 RefCountSegmentCache& ref_count_cache,
                                 VmmPoolImportCache& import_cache) noexcept
    -> std::optional<ConstBuffer> {
  if (!handle) {
    return std::nullopt;
  }
  switch (handle.memory_type()) {
    case BufferMemoryType::HostPageable:
    case BufferMemoryType::HostPinned:
      break;
    default:
      return std::nullopt;
  }

  const auto& handle_data = detail::BufferHandleWireAccess::host_wire(handle);
  if (!is_adoptable_handle_data(handle_data)) {
    return std::nullopt;
  }

  const std::uint32_t slot_index = detail::BufferHandleWireAccess::slot_index(handle);
  if (!detail::is_valid_host_buffer_slot_geometry(handle_data, slot_index)) {
    return std::nullopt;
  }
  const std::uint64_t ready_generation = detail::BufferHandleWireAccess::ready_generation(handle);

  auto ref_count_mapping_or = ref_count_cache.map(handle);
  if (!ref_count_mapping_or.has_value()) {
    return std::nullopt;
  }
  ShmRefCountMappingPtr ref_count_mapping = std::move(ref_count_mapping_or).value();

  auto segment_or = backing_cache.attach(static_cast<std::uint64_t>(handle_data.owner_pid),
                                         handle_data.shm_pool_id,
                                         handle_data.pool_total_bytes,
                                         handle.memory_type() == BufferMemoryType::HostPinned,
                                         handle_data.host_register_flags);
  if (!segment_or.has_value()) {
    return std::nullopt;
  }
  const auto& segment = segment_or.value();

  SlotRegions regions{};
  std::shared_ptr<void> generation_owner;

  if (detail::host_handle_uses_devicemap_interleaved(handle_data)) {
    const auto regions_or = make_host_devicemap_interleaved_slot_regions(
        segment.base, handle_data.slot_stride_bytes, handle_data.buffer_bytes);
    if (!regions_or.has_value()) {
      return std::nullopt;
    }
    regions = regions_or.value();
  } else {
    VmmPoolImportDescriptor import_descriptor{};
    import_descriptor.pool_cookie = handle_data.gen_pool_cookie;
    import_descriptor.slot_stride_bytes = handle_data.gen_slot_stride_bytes;
    import_descriptor.pool_total_bytes = handle_data.gen_pool_total_bytes;
    import_descriptor.fd_pass_socket_path = handle_data.fd_pass_socket_path;

    auto pool_base_or = import_cache.import_pool_base(import_descriptor);
    if (!pool_base_or.has_value()) {
      return std::nullopt;
    }

    generation_owner = pool_base_or->mapping_owner;
    regions = make_split_slot_regions(segment.base,
                                      handle_data.slot_stride_bytes,
                                      generation_pool_base_ptr(pool_base_or->base),
                                      handle_data.gen_slot_stride_bytes,
                                      handle_data.buffer_bytes);
  }

  auto impl = std::unique_ptr<HostConstBufferImpl>(
      new (std::nothrow) HostConstBufferImpl(std::move(ref_count_mapping),
                                             handle.memory_type(),
                                             handle_data,
                                             slot_index,
                                             ready_generation,
                                             regions));
  if (!impl) {
    return std::nullopt;
  }
  impl->retirement_owners_.payload = segment.mapping_owner;
  impl->retirement_owners_.generation = std::move(generation_owner);
  return ConstBuffer(std::move(impl));
}

auto HostConstBufferImpl::create(const BufferHandle& handle, CUstream_st* const stream,
                                 HostBackingSegmentCache& backing_cache,
                                 RefCountSegmentCache& ref_count_cache,
                                 VmmPoolImportCache& import_cache) noexcept
    -> std::optional<ConstBuffer> {
  auto buffer_or = import(handle, backing_cache, ref_count_cache, import_cache);
  if (!buffer_or.has_value()) {
    return std::nullopt;
  }

  ConstBuffer buffer = std::move(buffer_or).value();
  if (!buffer.enqueue_ready_wait(stream)) {
    return std::nullopt;
  }
  return buffer;
}

HostConstBufferImpl::HostConstBufferImpl(ShmRefCountMappingPtr ref_count_mapping,
                                         const BufferMemoryType memory_type,
                                         const detail::host::BufferHandleData handle_data,
                                         const std::uint32_t slot_index,
                                         const std::uint64_t ready_generation,
                                         SlotRegions regions) noexcept
    : Impl(regions, slot_index),
      ref_count_mapping_(std::move(ref_count_mapping)),
      memory_type_(memory_type),
      handle_data_(handle_data),
      slot_index_(slot_index),
      ready_generation_(ready_generation) {
  ref_count_mapping_->table().add_ref(slot_index_);
}

void HostConstBufferImpl::release_view_lease() noexcept {
  if (!ref_count_mapping_ || !lease_active_) {
    return;
  }
  const std::uint32_t slot_index = slot_index_;
  if (!release_streams_.empty()) {
    release_streams_.release();
  } else {
    ref_count_mapping_->table().release_ref(slot_index);
  }
  lease_active_ = false;
}

bool HostConstBufferImpl::reacquire(const BufferHandle& handle,
                                    const bool require_same_slot) noexcept {
  if (lease_active_ || !ref_count_mapping_ || !handle || handle.memory_type() != memory_type_) {
    return false;
  }
  const auto& handle_data = detail::BufferHandleWireAccess::host_wire(handle);
  const std::uint32_t slot_index = detail::BufferHandleWireAccess::slot_index(handle);
  if (!is_adoptable_handle_data(handle_data) ||
      !detail::is_valid_host_buffer_slot_geometry(handle_data, slot_index) ||
      (require_same_slot && slot_index != slot_index_) ||
      !same_host_pool(handle_data_, handle_data) ||
      !ref_count_mapping_->table().matches_segment(handle_data.owner_pid,
                                                   handle_data.shm_ref_count_id) ||
      slot_index >= ref_count_mapping_->table().buffer_count()) {
    return false;
  }
  index_ = slot_index;
  slot_index_ = slot_index;
  ready_generation_ = detail::BufferHandleWireAccess::ready_generation(handle);
  ref_count_mapping_->table().add_ref(slot_index_);
  lease_active_ = true;
  return true;
}

bool HostConstBufferImpl::reacquire(const BufferHandle& handle) noexcept {
  return reacquire(handle, false);
}

bool HostConstBufferImpl::active() const noexcept {
  return lease_active_;
}

void HostConstBufferImpl::release_for_reuse() noexcept {
  release_view_lease();
}

HostConstBufferImpl::~HostConstBufferImpl() {
  release_view_lease();
}

HostConstBufferImpl::HostConstBufferImpl(HostConstBufferImpl&& other) noexcept : Impl() {
  swap(*this, other);
}

void swap(HostConstBufferImpl& lhs, HostConstBufferImpl& rhs) noexcept {
  using std::swap;
  swap(static_cast<ConstBuffer::Impl&>(lhs), static_cast<ConstBuffer::Impl&>(rhs));
  swap(lhs.ref_count_mapping_, rhs.ref_count_mapping_);
  swap(lhs.memory_type_, rhs.memory_type_);
  swap(lhs.handle_data_, rhs.handle_data_);
  swap(lhs.slot_index_, rhs.slot_index_);
  swap(lhs.ready_generation_, rhs.ready_generation_);
  swap(lhs.lease_active_, rhs.lease_active_);
}

HostConstBufferImpl& HostConstBufferImpl::operator=(HostConstBufferImpl&& other) noexcept {
  HostConstBufferImpl tmp(std::move(other));
  swap(*this, tmp);
  return *this;
}

BufferMemoryType HostConstBufferImpl::memory_type() const noexcept {
  return lease_active_ ? memory_type_ : BufferMemoryType::None;
}

std::optional<int32_t> HostConstBufferImpl::cuda_device_ordinal() const noexcept {
  return std::nullopt;
}

bool HostConstBufferImpl::has_producer_readiness_fence() const noexcept {
  return lease_active_ && ready_generation_ != 0U;
}

bool HostConstBufferImpl::enqueue_ready_wait(CUstream_st* const stream) noexcept {
  if (!ref_count_mapping_ || !lease_active_) {
    return false;
  }
  if (release_streams_.contains(stream)) {
    return true;
  }

  const std::uint64_t ready_generation = ready_generation_;
  if (ready_generation == 0U) {
    if (is_usable_stream(stream)) {
      if (!register_stream(stream, {})) {
        return false;
      }
      release_streams_.mark_ready(stream);
    }
    return true;
  }
  if (!is_usable_stream(stream)) {
    return false;
  }
  if (!register_stream(stream, {}) || !wait_ready(stream, ready_generation)) {
    return false;
  }
  release_streams_.mark_ready(stream);
  return true;
}

std::optional<BufferHandle> HostConstBufferImpl::export_handle() noexcept {
  if (!ref_count_mapping_ || !lease_active_) {
    return std::nullopt;
  }

  return detail::make_host_buffer_handle(handle_data_, slot_index_, ready_generation_);
}

}  // namespace holomq
