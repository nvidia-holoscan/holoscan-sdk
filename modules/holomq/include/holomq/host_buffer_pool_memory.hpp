/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>

namespace holomq {

/// Contiguous host SHM payload backing for a host \ref BufferPool (pageable or CUDA-registered
/// pinned).
///
/// Implement and return from a custom \ref HostBufferAllocator. Exposes SHM attach metadata used to
/// fill wire handles and key \c HostBackingSegmentCache.
class HostBufferPoolMemory {
 public:
  virtual ~HostBufferPoolMemory() = default;

  [[nodiscard]] virtual void* ptr() noexcept = 0;
  [[nodiscard]] virtual const void* ptr() const noexcept = 0;
  [[nodiscard]] virtual std::size_t size() const noexcept = 0;
  [[nodiscard]] virtual std::uint32_t shm_pool_id() const noexcept = 0;
  [[nodiscard]] virtual bool payload_pinned() const noexcept = 0;
  [[nodiscard]] virtual unsigned int host_register_flags() const noexcept = 0;
};

}  // namespace holomq
