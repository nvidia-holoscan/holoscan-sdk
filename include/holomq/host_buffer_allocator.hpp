/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <holomq/host_buffer_pool_memory.hpp>

#include <cstddef>
#include <functional>
#include <memory>

namespace holomq {

/// One-shot host pool payload allocator: \p payload_region_bytes is the full SHM-backed region.
/// When \p payload_pinned is \c true, \p cuda_host_register_flags is passed to \c cuMemHostRegister
/// (including \c 0 for CUDA default register). Returns \c nullptr on failure.
using HostBufferAllocator = std::function<std::unique_ptr<HostBufferPoolMemory>(
    std::size_t payload_region_bytes, bool payload_pinned, unsigned int cuda_host_register_flags)>;

/// Built-in POSIX SHM create + map; registers with CUDA when \p payload_pinned is \c true.
[[nodiscard]] HostBufferAllocator default_host_buffer_allocator();

}  // namespace holomq
