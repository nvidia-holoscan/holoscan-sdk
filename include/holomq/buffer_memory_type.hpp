/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>

namespace holomq {

/// Physical storage type for buffer handles, pool slots, and import views.
///
/// @note Holoscan runtime mapping for Tensor/DLPack:
/// - \c Device -> \c holoscan::MemoryStorageType::kDevice / DLPack \c kDLCUDA (with CUDA device
///   ordinal from \ref BufferPool::cuda_device_ordinal or buffer views)
/// - \c HostPageable -> \c holoscan::MemoryStorageType::kSystem / DLPack \c kDLCPU
/// - \c HostPinned -> \c holoscan::MemoryStorageType::kHost / DLPack \c kDLCUDAHost
enum class BufferMemoryType : std::uint8_t {
  None = 0,
  Device = 1,
  HostPageable = 2,
  HostPinned = 3,
};

[[nodiscard]] constexpr bool is_device_buffer_memory_type(const BufferMemoryType type) noexcept {
  return type == BufferMemoryType::Device;
}

[[nodiscard]] constexpr bool is_host_buffer_memory_type(const BufferMemoryType type) noexcept {
  return type == BufferMemoryType::HostPageable || type == BufferMemoryType::HostPinned;
}

}  // namespace holomq
