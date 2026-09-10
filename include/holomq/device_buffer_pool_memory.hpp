/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>

namespace holomq {

/// Contiguous CUDA VMM backing for a device \ref BufferPool.
///
/// Implement and return from a custom \ref DeviceBufferAllocator. The mapping at \ref ptr must be a
/// pinned device VMM allocation exportable as a POSIX file descriptor (\ref
/// device_vmm_allocation_is_exportable); otherwise
/// \ref VmmExportedPool::create fails. VMM pool cookie and fd-pass export live in \c
/// VmmExportedPool; this interface exposes only the mapped region.
class DeviceBufferPoolMemory {
 public:
  virtual ~DeviceBufferPoolMemory() = default;

  [[nodiscard]] virtual void* ptr() noexcept = 0;
  [[nodiscard]] virtual const void* ptr() const noexcept = 0;

  /// Byte length of the mapped region starting at \ref ptr.
  [[nodiscard]] virtual std::size_t size() const noexcept = 0;
};

}  // namespace holomq
