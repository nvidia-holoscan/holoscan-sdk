/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <holomq/device_buffer_pool_memory.hpp>

#include <cuda.h>

#include <cstddef>
#include <functional>
#include <memory>

namespace holomq {

/// One-shot device pool backing allocator: \p number_of_bytes is the full region (e.g. aligned slot
/// size × slot count). Returns \c nullptr on failure; the returned \ref DeviceBufferPoolMemory owns
/// teardown of the mapping.
using DeviceBufferAllocator =
    std::function<std::unique_ptr<DeviceBufferPoolMemory>(std::size_t number_of_bytes)>;

/// Recommended VMM granularity for pinned device allocations with POSIX FD export.
[[nodiscard]] std::size_t recommended_device_vmm_granularity() noexcept;

/// Returns \c true when the current CUDA device supports exporting VMM allocations as POSIX file
/// descriptors.
[[nodiscard]] bool device_supports_vmm_posix_fd_export() noexcept;

/// Returns \c true when \p handle is a pinned device allocation exportable via POSIX FD.
[[nodiscard]] bool device_vmm_allocation_is_exportable(
    CUmemGenericAllocationHandle handle) noexcept;

/// Built-in CUDA VMM reservation + \c cuMemCreate + \c cuMemMap + device access.
[[nodiscard]] DeviceBufferAllocator default_device_buffer_allocator();

}  // namespace holomq
