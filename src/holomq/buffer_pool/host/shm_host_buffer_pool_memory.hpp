/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <holomq/host_buffer_pool_memory.hpp>
#include "buffer_pool/ref_count/shared_memory.hpp"
#include "cuda/host_registration.hpp"

#include <cstdint>
#include <optional>
#include <string>

namespace holomq {

class ShmHostBufferPoolMemory final : public HostBufferPoolMemory {
 public:
  ShmHostBufferPoolMemory(SharedMemory shm, std::size_t mapped_size, std::uint32_t shm_pool_id,
                          bool payload_pinned, unsigned int cuda_host_register_flags) noexcept;

  ShmHostBufferPoolMemory(const ShmHostBufferPoolMemory&) = delete;
  ShmHostBufferPoolMemory& operator=(const ShmHostBufferPoolMemory&) = delete;
  ShmHostBufferPoolMemory(ShmHostBufferPoolMemory&&) noexcept = default;
  ShmHostBufferPoolMemory& operator=(ShmHostBufferPoolMemory&&) noexcept = default;
  ~ShmHostBufferPoolMemory() override = default;

  [[nodiscard]] void* ptr() noexcept override;
  [[nodiscard]] const void* ptr() const noexcept override;
  [[nodiscard]] std::size_t size() const noexcept override;
  [[nodiscard]] std::uint32_t shm_pool_id() const noexcept override;
  [[nodiscard]] bool payload_pinned() const noexcept override;
  [[nodiscard]] unsigned int host_register_flags() const noexcept override;

  [[nodiscard]] bool host_registered() const noexcept;
  /**
   * @brief Return the CUDA driver result associated with host-memory registration.
   * @return The registration result when registration is active, or \c CUDA_SUCCESS when host
   * registration is not active.
   */
  [[nodiscard]] CUresult host_register_result() const noexcept;

  [[nodiscard]] static std::string shm_name(std::uint64_t owner_pid,
                                            std::uint32_t shm_pool_id) noexcept;

 private:
  SharedMemory shm_;
  std::size_t mapped_size_{0U};
  std::uint32_t shm_pool_id_{0U};
  bool payload_pinned_{false};
  unsigned int cuda_host_register_flags_{0U};
  std::optional<cuda::RegisteredHostMemory> host_memory_registration_;
};

}  // namespace holomq
