/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "buffer_pool/ref_count/ref_count_key.hpp"

namespace holomq {

namespace {

[[nodiscard]] std::optional<RefCountKey> ref_count_key_from_ids(
    const std::uint32_t owner_pid, const std::uint32_t shm_ref_count_id) noexcept {
  if (owner_pid == 0U || shm_ref_count_id == 0U) {
    return std::nullopt;
  }
  return RefCountKey{static_cast<std::uint64_t>(owner_pid), shm_ref_count_id};
}

}  // namespace

std::optional<RefCountKey> ref_count_key(const BufferHandle& handle) noexcept {
  if (!handle) {
    return std::nullopt;
  }

  // TODO: revisit this implementation to make it more efficient.
  // Option 1: Hoist ref-count identity onto BufferHandle
  // Option 2: Shared embedded sub-struct in each *HandleData
  // Option 3: A base class and inheritance pattern
  switch (handle.memory_type()) {
    case BufferMemoryType::Device: {
      const auto& device = detail::BufferHandleWireAccess::device_wire(handle);
      return ref_count_key_from_ids(device.owner_pid, device.shm_ref_count_id);
    }
    case BufferMemoryType::HostPageable:
    case BufferMemoryType::HostPinned: {
      const auto& host = detail::BufferHandleWireAccess::host_wire(handle);
      return ref_count_key_from_ids(host.owner_pid, host.shm_ref_count_id);
    }
    case BufferMemoryType::None:
    default:
      return std::nullopt;
  }
}

}  // namespace holomq
