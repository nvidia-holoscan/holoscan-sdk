/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "buffer_pool/host/shm_host_buffer_pool_memory.hpp"

#include <sstream>
#include <string>
#include <utility>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace holomq {

std::string ShmHostBufferPoolMemory::shm_name(const std::uint64_t owner_pid,
                                              const std::uint32_t shm_pool_id) noexcept {
  std::ostringstream name;
  name << "/holomq_host_" << std::hex << owner_pid << "_" << shm_pool_id;
  return name.str();
}

ShmHostBufferPoolMemory::ShmHostBufferPoolMemory(
    SharedMemory shm, const std::size_t mapped_size, const std::uint32_t shm_pool_id,
    const bool payload_pinned, const unsigned int cuda_host_register_flags) noexcept
    : shm_(std::move(shm)),
      mapped_size_(mapped_size),
      shm_pool_id_(shm_pool_id),
      payload_pinned_(payload_pinned),
      cuda_host_register_flags_(cuda_host_register_flags) {
  if (payload_pinned_) {
    host_memory_registration_.emplace(shm_.mapped_base(), mapped_size_, cuda_host_register_flags_);
  }
}

bool ShmHostBufferPoolMemory::host_registered() const noexcept {
  return !payload_pinned_ || static_cast<bool>(host_memory_registration_);
}

CUresult ShmHostBufferPoolMemory::host_register_result() const noexcept {
  if (!payload_pinned_ || !host_memory_registration_.has_value()) {
    return CUDA_SUCCESS;
  }
  return host_memory_registration_->register_result();
}

void* ShmHostBufferPoolMemory::ptr() noexcept {
  return shm_.mapped_base();
}

const void* ShmHostBufferPoolMemory::ptr() const noexcept {
  return shm_.mapped_base();
}

std::size_t ShmHostBufferPoolMemory::size() const noexcept {
  return mapped_size_;
}

std::uint32_t ShmHostBufferPoolMemory::shm_pool_id() const noexcept {
  return shm_pool_id_;
}

bool ShmHostBufferPoolMemory::payload_pinned() const noexcept {
  return payload_pinned_;
}

unsigned int ShmHostBufferPoolMemory::host_register_flags() const noexcept {
  return cuda_host_register_flags_;
}

}  // namespace holomq
