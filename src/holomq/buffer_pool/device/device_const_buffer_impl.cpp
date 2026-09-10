/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "buffer_pool/device/device_const_buffer_impl.hpp"

#include <holomq/buffer_handle.hpp>
#include "buffer_pool/ref_count/shm_ref_count.hpp"
#include "buffer_pool/slot_region.hpp"
#include "buffer_pool/sync/deferred_ref_release.hpp"

#include <cuda.h>

#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <optional>
#include <utility>

namespace holomq {

namespace {

[[nodiscard]] VmmPoolImportDescriptor vmm_pool_import_descriptor(
    const detail::device::BufferHandleData& handle_data) noexcept {
  VmmPoolImportDescriptor descriptor{};
  descriptor.pool_cookie = handle_data.pool_cookie;
  descriptor.slot_stride_bytes = handle_data.slot_stride_bytes;
  descriptor.pool_total_bytes = handle_data.pool_total_bytes;
  descriptor.fd_pass_socket_path = handle_data.fd_pass_socket_path;
  descriptor.cuda_device_ordinal = handle_data.cuda_device_ordinal;
  return descriptor;
}

[[nodiscard]] void* device_ptr_to_host_ptr(const CUdeviceptr device_ptr) noexcept {
  return reinterpret_cast<void*>(static_cast<std::uintptr_t>(device_ptr));
}

[[nodiscard]] bool same_device_pool(const detail::device::BufferHandleData& lhs,
                                    const detail::device::BufferHandleData& rhs) noexcept {
  return lhs.pool_cookie == rhs.pool_cookie && lhs.slot_stride_bytes == rhs.slot_stride_bytes &&
         lhs.buffer_bytes == rhs.buffer_bytes && lhs.pool_total_bytes == rhs.pool_total_bytes &&
         lhs.cuda_device_ordinal == rhs.cuda_device_ordinal && lhs.owner_pid == rhs.owner_pid &&
         lhs.shm_ref_count_id == rhs.shm_ref_count_id &&
         std::memcmp(lhs.fd_pass_socket_path,
                     rhs.fd_pass_socket_path,
                     sizeof(lhs.fd_pass_socket_path)) == 0;
}

[[nodiscard]] std::optional<CUdeviceptr> imported_pool_slot_device_ptr(
    const VmmPoolImportBase& pool, const std::uint32_t slot_index) noexcept {
  if (pool.slot_stride_bytes == 0U || pool.pool_total_bytes == 0U) {
    return std::nullopt;
  }
  if (slot_index != 0U && pool.slot_stride_bytes > (std::numeric_limits<std::size_t>::max() /
                                                    static_cast<std::size_t>(slot_index))) {
    return std::nullopt;
  }
  const std::size_t slot_offset = static_cast<std::size_t>(slot_index) * pool.slot_stride_bytes;
  // Subtract rather than add: slot_offset + stride can wrap (e.g. stride=2^63, index=1) and
  // spuriously pass a size_t upper-bound check, yielding a wild device pointer.
  if (slot_offset > pool.pool_total_bytes ||
      pool.slot_stride_bytes > pool.pool_total_bytes - slot_offset) {
    return std::nullopt;
  }
  return pool.base + slot_offset;
}

}  // namespace

auto DeviceConstBufferImpl::import(const BufferHandle& handle, VmmPoolImportCache& import_cache,
                                   RefCountSegmentCache& ref_count_cache) noexcept
    -> std::optional<ConstBuffer> {
  if (!handle || handle.memory_type() != BufferMemoryType::Device) {
    return std::nullopt;
  }

  const auto& handle_data = detail::BufferHandleWireAccess::device_wire(handle);
  if (!detail::is_valid_device_buffer_wire(handle_data)) {
    return std::nullopt;
  }

  const std::uint32_t slot_index = detail::BufferHandleWireAccess::slot_index(handle);
  // Raw iceoryx2 BufferHandle bytes bypass make_device_buffer_handle; reject out-of-range or
  // wrapping slot geometry before mapping or forming device pointers.
  if (!detail::is_valid_device_buffer_slot_geometry(handle_data, slot_index)) {
    return std::nullopt;
  }
  const std::uint64_t ready_generation = detail::BufferHandleWireAccess::ready_generation(handle);

  auto ref_count_mapping_or = ref_count_cache.map(handle);
  if (!ref_count_mapping_or.has_value()) {
    return std::nullopt;
  }
  ShmRefCountMappingPtr ref_count_mapping = std::move(ref_count_mapping_or).value();
  if (slot_index >= ref_count_mapping->table().buffer_count()) {
    return std::nullopt;
  }

  auto pool_or = import_cache.import_pool_base(vmm_pool_import_descriptor(handle_data));
  if (!pool_or.has_value()) {
    return std::nullopt;
  }

  const VmmPoolImportBase pool_base = pool_or.value();
  auto slot_ptr_or = imported_pool_slot_device_ptr(pool_base, slot_index);
  if (!slot_ptr_or.has_value()) {
    return std::nullopt;
  }

  // Report the imported allocation's validated device rather than trusting the wire a second time.
  auto impl = std::unique_ptr<DeviceConstBufferImpl>(
      new (std::nothrow) DeviceConstBufferImpl(std::move(ref_count_mapping),
                                               handle_data,
                                               slot_index,
                                               ready_generation,
                                               device_ptr_to_host_ptr(pool_base.base),
                                               pool_base.cuda_device_ordinal));
  if (!impl) {
    return std::nullopt;
  }
  return ConstBuffer(std::move(impl));
}

auto DeviceConstBufferImpl::create(const BufferHandle& handle, CUstream_st* const stream,
                                   VmmPoolImportCache& import_cache,
                                   RefCountSegmentCache& ref_count_cache) noexcept
    -> std::optional<ConstBuffer> {
  auto buffer_or = import(handle, import_cache, ref_count_cache);
  if (!buffer_or.has_value()) {
    return std::nullopt;
  }

  ConstBuffer buffer = std::move(buffer_or).value();
  if (!buffer.enqueue_ready_wait(stream)) {
    return std::nullopt;
  }
  return buffer;
}

DeviceConstBufferImpl::DeviceConstBufferImpl(ShmRefCountMappingPtr ref_count_mapping,
                                             const detail::device::BufferHandleData& handle_data,
                                             const std::uint32_t slot_index,
                                             const std::uint64_t ready_generation,
                                             void* const pool_base,
                                             const int32_t cuda_device_ordinal) noexcept
    : Impl(make_interleaved_slot_regions(pool_base, handle_data.slot_stride_bytes,
                                         handle_data.buffer_bytes),
           slot_index),
      ref_count_mapping_(std::move(ref_count_mapping)),
      handle_data_(handle_data),
      slot_index_(slot_index),
      ready_generation_(ready_generation),
      cuda_device_ordinal_(cuda_device_ordinal) {
  ref_count_mapping_->table().add_ref(slot_index_);
}

void DeviceConstBufferImpl::release_view_lease() noexcept {
  if (!ref_count_mapping_ || !lease_active_) {
    return;
  }
  const std::uint32_t slot_index = slot_index_;
  if (release_stream_ != nullptr) {
    defer_ref_release_on_stream(ref_count_mapping_, slot_index, release_stream_);
  } else {
    ref_count_mapping_->table().release_ref(slot_index);
  }
  release_stream_ = nullptr;
  lease_active_ = false;
}

bool DeviceConstBufferImpl::reacquire(const BufferHandle& handle) noexcept {
  if (lease_active_ || !ref_count_mapping_ || !handle ||
      handle.memory_type() != BufferMemoryType::Device) {
    return false;
  }
  const auto& handle_data = detail::BufferHandleWireAccess::device_wire(handle);
  const std::uint32_t slot_index = detail::BufferHandleWireAccess::slot_index(handle);
  if (!detail::is_valid_device_buffer_wire(handle_data) ||
      !detail::is_valid_device_buffer_slot_geometry(handle_data, slot_index) ||
      !same_device_pool(handle_data_, handle_data) ||
      !ref_count_mapping_->table().matches_segment(handle_data.owner_pid,
                                                   handle_data.shm_ref_count_id) ||
      slot_index >= ref_count_mapping_->table().buffer_count()) {
    return false;
  }
  index_ = slot_index;
  slot_index_ = slot_index;
  ready_generation_ = detail::BufferHandleWireAccess::ready_generation(handle);
  release_stream_ = nullptr;
  ref_count_mapping_->table().add_ref(slot_index_);
  lease_active_ = true;
  return true;
}

bool DeviceConstBufferImpl::active() const noexcept {
  return lease_active_;
}

void DeviceConstBufferImpl::release_for_reuse() noexcept {
  release_view_lease();
}

DeviceConstBufferImpl::~DeviceConstBufferImpl() {
  release_view_lease();
}

DeviceConstBufferImpl::DeviceConstBufferImpl(DeviceConstBufferImpl&& other) noexcept : Impl() {
  swap(*this, other);
}

void swap(DeviceConstBufferImpl& lhs, DeviceConstBufferImpl& rhs) noexcept {
  using std::swap;
  swap(static_cast<ConstBuffer::Impl&>(lhs), static_cast<ConstBuffer::Impl&>(rhs));
  swap(lhs.ref_count_mapping_, rhs.ref_count_mapping_);
  swap(lhs.handle_data_, rhs.handle_data_);
  swap(lhs.slot_index_, rhs.slot_index_);
  swap(lhs.ready_generation_, rhs.ready_generation_);
  swap(lhs.cuda_device_ordinal_, rhs.cuda_device_ordinal_);
  swap(lhs.release_stream_, rhs.release_stream_);
  swap(lhs.lease_active_, rhs.lease_active_);
}

DeviceConstBufferImpl& DeviceConstBufferImpl::operator=(DeviceConstBufferImpl&& other) noexcept {
  DeviceConstBufferImpl tmp(std::move(other));
  swap(*this, tmp);
  return *this;
}

BufferMemoryType DeviceConstBufferImpl::memory_type() const noexcept {
  return lease_active_ ? BufferMemoryType::Device : BufferMemoryType::None;
}

std::optional<int32_t> DeviceConstBufferImpl::cuda_device_ordinal() const noexcept {
  return lease_active_ ? std::optional<int32_t>{cuda_device_ordinal_} : std::nullopt;
}

bool DeviceConstBufferImpl::has_producer_readiness_fence() const noexcept {
  return lease_active_ && ready_generation_ != 0U;
}

bool DeviceConstBufferImpl::enqueue_ready_wait(CUstream_st* const stream) noexcept {
  if (!ref_count_mapping_ || !lease_active_) {
    return false;
  }
  if (release_stream_ != nullptr) {
    return true;
  }

  const std::uint64_t ready_generation = ready_generation_;
  if (ready_generation == 0U) {
    if (is_usable_stream(stream)) {
      release_stream_ = stream;
    }
    return true;
  }
  if (!is_usable_stream(stream)) {
    return false;
  }
  if (!wait_ready(stream, ready_generation)) {
    return false;
  }
  release_stream_ = stream;
  return true;
}

std::optional<BufferHandle> DeviceConstBufferImpl::export_handle() noexcept {
  if (!ref_count_mapping_ || !lease_active_ || handle_data_.pool_cookie == 0U) {
    return std::nullopt;
  }

  return detail::make_device_buffer_handle(handle_data_, slot_index_, ready_generation_);
}

}  // namespace holomq
