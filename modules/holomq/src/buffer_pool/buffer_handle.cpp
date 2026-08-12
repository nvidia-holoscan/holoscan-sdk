/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holomq/buffer_handle.hpp>

#include "buffer_pool/slot_region.hpp"

#include <cuda.h>

#include <limits>

namespace holomq {

BufferHandle::BufferHandle(const BufferMemoryType memory_type, HandleData data,
                           const std::uint32_t slot_index,
                           const std::uint64_t ready_generation) noexcept
    : memory_type_(memory_type),
      slot_index_(slot_index),
      ready_generation_(ready_generation),
      data_(data) {}

BufferHandle::operator bool() const noexcept {
  return memory_type_ != BufferMemoryType::None;
}

BufferMemoryType BufferHandle::memory_type() const noexcept {
  return memory_type_;
}

namespace detail {

const detail::device::BufferHandleData& BufferHandleWireAccess::device_wire(
    const BufferHandle& handle) noexcept {
  return handle.data_.device;
}

const detail::host::BufferHandleData& BufferHandleWireAccess::host_wire(
    const BufferHandle& handle) noexcept {
  return handle.data_.host;
}

std::uint32_t BufferHandleWireAccess::slot_index(const BufferHandle& handle) noexcept {
  return handle.slot_index_;
}

std::uint64_t BufferHandleWireAccess::ready_generation(const BufferHandle& handle) noexcept {
  return handle.ready_generation_;
}

namespace {

[[nodiscard]] BufferMemoryType host_wire_memory_type(
    const host::BufferHandleData& handle_data) noexcept {
  return handle_data.payload_pinned != 0U ? BufferMemoryType::HostPinned
                                          : BufferMemoryType::HostPageable;
}

[[nodiscard]] bool host_handle_uses_devicemap_interleaved(
    const host::BufferHandleData& handle_data) noexcept {
  return host_wire_memory_type(handle_data) == BufferMemoryType::HostPinned &&
         (handle_data.host_register_flags & CU_MEMHOSTREGISTER_DEVICEMAP) != 0U;
}

[[nodiscard]] bool is_valid_device_buffer_wire_impl(
    const device::BufferHandleData& handle_data) noexcept {
  if (handle_data.owner_pid == 0U || handle_data.shm_ref_count_id == 0U ||
      handle_data.slot_stride_bytes == 0U || handle_data.buffer_bytes == 0U ||
      handle_data.pool_total_bytes == 0U || handle_data.pool_cookie == 0U) {
    return false;
  }
  if (handle_data.buffer_bytes + k_slot_gpu_generation_bytes > handle_data.slot_stride_bytes) {
    return false;
  }
  if (handle_data.cuda_device_ordinal < 0) {
    return false;
  }
  return handle_data.fd_pass_socket_path[0] != '\0';
}

[[nodiscard]] bool slot_index_fits_in_pool(const std::size_t slot_stride_bytes,
                                           const std::size_t pool_total_bytes,
                                           const std::uint32_t slot_index) noexcept {
  if (slot_stride_bytes == 0U || pool_total_bytes == 0U) {
    return false;
  }
  if (slot_index != 0U && slot_stride_bytes > (std::numeric_limits<std::size_t>::max() /
                                               static_cast<std::size_t>(slot_index))) {
    return false;
  }
  const std::size_t slot_offset = static_cast<std::size_t>(slot_index) * slot_stride_bytes;
  return slot_offset <= pool_total_bytes && slot_stride_bytes <= pool_total_bytes - slot_offset;
}

[[nodiscard]] bool is_valid_host_buffer_wire_impl(
    const host::BufferHandleData& handle_data) noexcept {
  if (handle_data.owner_pid == 0U || handle_data.shm_pool_id == 0U ||
      handle_data.shm_ref_count_id == 0U || handle_data.slot_stride_bytes == 0U ||
      handle_data.buffer_bytes == 0U || handle_data.pool_total_bytes == 0U) {
    return false;
  }

  if (host_wire_memory_type(handle_data) == BufferMemoryType::HostPageable &&
      handle_data.host_register_flags != 0U) {
    return false;
  }

  if (host_handle_uses_devicemap_interleaved(handle_data)) {
    return handle_data.buffer_bytes + k_slot_gpu_generation_bytes <= handle_data.slot_stride_bytes;
  }

  if (handle_data.slot_stride_bytes != handle_data.buffer_bytes) {
    return false;
  }
  if (handle_data.gen_pool_cookie == 0U || handle_data.gen_pool_total_bytes == 0U ||
      handle_data.gen_slot_stride_bytes < k_slot_gpu_generation_bytes ||
      handle_data.fd_pass_socket_path[0] == '\0') {
    return false;
  }
  return true;
}

}  // namespace

bool is_valid_device_buffer_wire(const device::BufferHandleData& handle_data) noexcept {
  return is_valid_device_buffer_wire_impl(handle_data);
}

bool is_valid_host_buffer_wire(const host::BufferHandleData& handle_data) noexcept {
  return is_valid_host_buffer_wire_impl(handle_data);
}

bool is_valid_host_buffer_slot_geometry(const host::BufferHandleData& handle_data,
                                        const std::uint32_t slot_index) noexcept {
  if (!slot_index_fits_in_pool(
          handle_data.slot_stride_bytes, handle_data.pool_total_bytes, slot_index)) {
    return false;
  }
  return host_handle_uses_devicemap_interleaved(handle_data) ||
         slot_index_fits_in_pool(
             handle_data.gen_slot_stride_bytes, handle_data.gen_pool_total_bytes, slot_index);
}

BufferHandle make_device_buffer_handle(const device::BufferHandleData& handle_data,
                                       const std::uint32_t slot_index,
                                       const std::uint64_t ready_generation) noexcept {
  if (!is_valid_device_buffer_wire_impl(handle_data)) {
    return BufferHandle{};
  }
  if (!slot_index_fits_in_pool(
          handle_data.slot_stride_bytes, handle_data.pool_total_bytes, slot_index)) {
    return BufferHandle{};
  }

  BufferHandle::HandleData data{};
  data.device = handle_data;
  return BufferHandle(BufferMemoryType::Device, data, slot_index, ready_generation);
}

BufferHandle make_host_buffer_handle(const host::BufferHandleData& handle_data,
                                     const std::uint32_t slot_index,
                                     const std::uint64_t ready_generation) noexcept {
  if (!is_valid_host_buffer_wire_impl(handle_data)) {
    return BufferHandle{};
  }
  if (!is_valid_host_buffer_slot_geometry(handle_data, slot_index)) {
    return BufferHandle{};
  }

  BufferHandle::HandleData data{};
  data.host = handle_data;
  return BufferHandle(host_wire_memory_type(handle_data), data, slot_index, ready_generation);
}

}  // namespace detail

}  // namespace holomq
