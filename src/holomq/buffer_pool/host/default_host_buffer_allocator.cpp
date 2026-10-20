/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holomq/host_buffer_allocator.hpp>

#include <holomq/logger.hpp>
#include "buffer_pool/host/shm_host_buffer_pool_memory.hpp"
#include "buffer_pool/ref_count/shared_memory.hpp"

#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <memory>
#include <utility>

#include <sys/stat.h>
#include <unistd.h>

namespace holomq {

namespace {

/// Return the next per-process host payload SHM segment id.
///
/// Ids are drawn from a pseudo-random SplitMix64 sequence instead of a sequential 1, 2, 3, ...
/// counter. Repeated tests run inside containers with `--ipc=host`; after an abort the kernel can
/// reuse the same PID while stale `/holomq_host_<pid>_<id>` objects still exist. Non-sequential ids
/// make those stale objects vanishingly unlikely to collide with the next process.
[[nodiscard]] std::uint32_t allocate_shm_pool_id() noexcept {
  static std::atomic<std::uint64_t> state{[]() noexcept -> std::uint64_t {
    const std::uint64_t seed = (static_cast<std::uint64_t>(::getpid()) << 32U) ^
                               static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(&state));
    return seed != 0U ? seed : 0x9E3779B97F4A7C15ULL;
  }()};

  std::uint64_t x = state.fetch_add(0x9E3779B97F4A7C15ULL, std::memory_order_relaxed);
  x ^= x >> 30U;
  x *= 0xBF58476D1CE4E5B9ULL;
  x ^= x >> 27U;
  x *= 0x94D049BB133111EBULL;
  x ^= x >> 31U;
  const std::uint32_t id = static_cast<std::uint32_t>(x);
  return id == 0U ? 1U : id;
}

constexpr int kShmPayloadCreateAttempts = 16;

}  // namespace

HostBufferAllocator default_host_buffer_allocator() {
  return [](const std::size_t payload_region_bytes,
            const bool payload_pinned,
            const unsigned int cuda_host_register_flags) -> std::unique_ptr<HostBufferPoolMemory> {
    if (payload_region_bytes == 0U) {
      return nullptr;
    }

    const std::uint64_t owner_pid = static_cast<std::uint64_t>(::getpid());
    SharedMemory shm;
    std::uint32_t shm_pool_id = 0U;
    for (int attempt = 0; attempt < kShmPayloadCreateAttempts; ++attempt) {
      shm_pool_id = allocate_shm_pool_id();
      shm = SharedMemory(ShmHostBufferPoolMemory::shm_name(owner_pid, shm_pool_id), true);
      if (shm) {
        break;
      }
      if (errno != EEXIST) {
        HOLOMQ_LOG_ERROR("host payload backing create failed: shm_open failed (errno={}, {})",
                         errno,
                         std::strerror(errno));
        return nullptr;
      }
    }
    if (!shm) {
      HOLOMQ_LOG_ERROR(
          "host payload backing create failed: shm_open failed after retries (errno={}, {})",
          errno,
          std::strerror(errno));
      return nullptr;
    }
    if (::ftruncate(shm.fd_get(), static_cast<off_t>(payload_region_bytes)) != 0) {
      HOLOMQ_LOG_ERROR("host payload backing create failed: ftruncate failed (errno={}, {})",
                       errno,
                       std::strerror(errno));
      return nullptr;
    }
    if (!shm.map(payload_region_bytes)) {
      HOLOMQ_LOG_ERROR("host payload backing create failed: mmap failed (errno={}, {})",
                       errno,
                       std::strerror(errno));
      return nullptr;
    }

    auto backing = std::make_unique<ShmHostBufferPoolMemory>(std::move(shm),
                                                             payload_region_bytes,
                                                             shm_pool_id,
                                                             payload_pinned,
                                                             cuda_host_register_flags);
    if (backing->ptr() == nullptr) {
      HOLOMQ_LOG_ERROR("host payload backing create failed: mapped base is null");
      return nullptr;
    }
    if (!backing->host_registered()) {
      HOLOMQ_LOG_ERROR("host payload backing create failed: cuMemHostRegister failed (CUresult={})",
                       static_cast<int>(backing->host_register_result()));
      return nullptr;
    }
    return backing;
  };
}

}  // namespace holomq
