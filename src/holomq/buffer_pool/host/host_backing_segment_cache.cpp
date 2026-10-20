/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "buffer_pool/host/host_backing_segment_cache.hpp"

#include "buffer_pool/host/shm_host_buffer_pool_memory.hpp"
#include "buffer_pool/ref_count/shm_ref_count.hpp"
#include "buffer_pool/sync/stream_release.hpp"

#include <holomq/logger.hpp>

#include <memory>
#include <utility>

#include <sys/stat.h>

namespace holomq {

namespace {

[[nodiscard]] auto validate_cached_backing(const std::shared_ptr<HostBufferPoolMemory>& owner,
                                           const std::size_t expected_size_bytes,
                                           const bool payload_pinned,
                                           const unsigned int cuda_host_register_flags) noexcept
    -> std::optional<HostBackingSegmentCache::AttachedSegment> {
  auto& backing = *owner;
  if (backing.size() < expected_size_bytes || backing.payload_pinned() != payload_pinned ||
      backing.host_register_flags() != cuda_host_register_flags) {
    HOLOMQ_LOG_ERROR(
        "host backing segment attach failed: metadata mismatch "
        "(expected_size_bytes={}, actual_size_bytes={}, expected_payload_pinned={}, "
        "actual_payload_pinned={}, expected_cuda_host_register_flags={}, "
        "actual_cuda_host_register_flags={})",
        expected_size_bytes,
        backing.size(),
        payload_pinned,
        backing.payload_pinned(),
        cuda_host_register_flags,
        backing.host_register_flags());
    return std::nullopt;
  }
  return HostBackingSegmentCache::AttachedSegment{
      owner,
      backing.ptr(),
      backing.size(),
      backing.payload_pinned(),
      backing.host_register_flags(),
  };
}

}  // namespace

auto HostBackingSegmentCache::attach(const std::uint64_t owner_pid, const std::uint32_t shm_pool_id,
                                     const std::size_t expected_size_bytes,
                                     const bool payload_pinned,
                                     const unsigned int cuda_host_register_flags) noexcept
    -> std::optional<AttachedSegment> {
  if (owner_pid == 0U || shm_pool_id == 0U || expected_size_bytes == 0U) {
    HOLOMQ_LOG_ERROR("host backing segment attach failed: invalid parameters");
    return std::nullopt;
  }

  try {
    const Key key{owner_pid, shm_pool_id};

    {
      const std::lock_guard<std::mutex> lock(mutex_);
      const auto found = entries_.find(key);
      if (found != entries_.end()) {
        return validate_cached_backing(
            found->second.backing, expected_size_bytes, payload_pinned, cuda_host_register_flags);
      }
    }

    SharedMemory shm(ShmHostBufferPoolMemory::shm_name(owner_pid, shm_pool_id), false);
    if (!shm) {
      HOLOMQ_LOG_ERROR("host backing segment attach failed: shm_open failed");
      return std::nullopt;
    }

    struct ::stat st{};
    if (::fstat(shm.fd_get(), &st) != 0 || st.st_size <= 0) {
      HOLOMQ_LOG_ERROR("host backing segment attach failed: fstat failed");
      return std::nullopt;
    }

    const std::size_t mapped_size = static_cast<std::size_t>(st.st_size);
    if (mapped_size < expected_size_bytes) {
      HOLOMQ_LOG_ERROR("host backing segment attach failed: segment smaller than expected size");
      return std::nullopt;
    }

    if (!shm.map(mapped_size)) {
      HOLOMQ_LOG_ERROR("host backing segment attach failed: mmap failed");
      return std::nullopt;
    }

    auto backing = std::make_shared<ShmHostBufferPoolMemory>(
        std::move(shm), mapped_size, shm_pool_id, payload_pinned, cuda_host_register_flags);
    if (backing->ptr() == nullptr || !backing->host_registered()) {
      HOLOMQ_LOG_ERROR("host backing segment attach failed: host registration failed");
      return std::nullopt;
    }
    if (payload_pinned) {
      CUcontext current{};
      CUdevice device{};
      BufferStreamRelease::Owners deferred{backing, {}};
      if (cuCtxGetCurrent(&current) != CUDA_SUCCESS ||
          (current != nullptr ? cuCtxGetDevice(&device) : cuDeviceGet(&device, 0)) !=
              CUDA_SUCCESS ||
          !BufferStreamRelease::defer_cuda_owners(deferred, device)) {
        return std::nullopt;
      }
      backing =
          std::shared_ptr<ShmHostBufferPoolMemory>(std::move(deferred.payload), backing.get());
    }

    AttachedSegment segment{
        backing,
        backing->ptr(),
        backing->size(),
        backing->payload_pinned(),
        backing->host_register_flags(),
    };

    const std::lock_guard<std::mutex> lock(mutex_);
    auto [iter, inserted] = entries_.emplace(key, Entry{std::move(backing)});
    if (!inserted) {
      return validate_cached_backing(
          iter->second.backing, expected_size_bytes, payload_pinned, cuda_host_register_flags);
    }
    return segment;
  } catch (...) {
    return std::nullopt;
  }
}

}  // namespace holomq
