/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>

#include <holoscan/holoscan_core_export.h>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/memory/memory_storage_type.hpp>

namespace holoscan {

/**
 * @brief Candidate native allocator facade for Holoscan 5 buffer ownership.
 *
 * This interface intentionally does not inherit from Component and does not use
 * GXF-style ABI hooks. It is a small production-direction surface for code that
 * wants expected-returning allocation, explicit storage type, and no dependency
 * on GXF lifecycle machinery. CUDA stream-ordered behavior is modeled by the
 * CUDA-specific derived interface in cuda_stream_allocator.hpp.
 */
class HOLOSCAN_CORE_EXPORT Allocator {
 public:
  virtual ~Allocator() = default;

  [[nodiscard]] virtual expected<void, Error> is_available(uint64_t size,
                                                           MemoryStorageType storage_type) = 0;
  [[nodiscard]] virtual expected<std::byte*, Error> allocate(uint64_t size,
                                                             MemoryStorageType storage_type) = 0;
  [[nodiscard]] virtual expected<void, Error> free(std::byte* pointer) = 0;

  [[nodiscard]] virtual uint64_t block_size([[maybe_unused]] MemoryStorageType storage_type) const {
    return 1;
  }

  // Sentinel returned by device_id() when the allocator does not bind allocations to a fixed CUDA
  // device (e.g. system memory, or an allocator that uses the current/implicit device). Callers
  // should derive the device ordinal from the backing pointer in that case.
  static constexpr int32_t kUnknownDeviceId = -1;

  // CUDA device ordinal that allocations of the given storage type are placed on, or
  // kUnknownDeviceId when the allocator does not bind allocations to a fixed device. This lets
  // MemoryBuffer record the owning device at allocation time so tensor construction can set DLPack
  // device metadata without a per-allocation pointer query.
  [[nodiscard]] virtual int32_t device_id([[maybe_unused]] MemoryStorageType storage_type) const {
    return kUnknownDeviceId;
  }
};

}  // namespace holoscan
