/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <set>

#include <holoscan/holoscan_core_export.h>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/memory/allocator.hpp>

namespace holoscan {

/**
 * @brief Native unbounded allocator for bring-up and unconstrained profiles.
 *
 * This allocator intentionally uses dynamic host/CUDA allocation and bookkeeping, so it is useful
 * for tests and early integration but is not evidence for bounded deterministic profiles.
 */
class HOLOSCAN_CORE_EXPORT UnboundedAllocator final : public Allocator {
 public:
  UnboundedAllocator() = default;
  ~UnboundedAllocator() override = default;

  [[nodiscard]] expected<void, Error> is_available(uint64_t size,
                                                   MemoryStorageType storage_type) override;
  [[nodiscard]] expected<std::byte*, Error> allocate(uint64_t size,
                                                     MemoryStorageType storage_type) override;
  [[nodiscard]] expected<void, Error> free(std::byte* pointer) override;

 private:
  std::mutex mutex_;
  std::set<std::byte*> system_blocks_;
  std::set<std::byte*> cuda_blocks_;
  std::set<std::byte*> cuda_host_blocks_;
};

}  // namespace holoscan
