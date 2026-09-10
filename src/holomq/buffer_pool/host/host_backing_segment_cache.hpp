/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <holomq/host_buffer_pool_memory.hpp>
#include "buffer_pool/ref_count/shm_ref_count.hpp"

#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <unordered_map>

namespace holomq {

/// Process-local cache of attached host payload SHM segments keyed by \c (owner_pid, shm_pool_id).
///
/// \ref attach is thread-safe; concurrent attaches for the same key coalesce to one cached mapping.
class HostBackingSegmentCache {
 public:
  struct AttachedSegment {
    void* base{nullptr};
    std::size_t size_bytes{0U};
    bool payload_pinned{false};
    unsigned int host_register_flags{0U};
  };

  /// Attaches (or reuses cached) payload SHM for \p owner_pid / \p shm_pool_id. On first attach,
  /// validates the object size with \c fstat before \c mmap and rejects segments smaller than \p
  /// expected_size_bytes. Cache hits also validate \p expected_size_bytes and registration
  /// metadata.
  [[nodiscard]] auto attach(std::uint64_t owner_pid, std::uint32_t shm_pool_id,
                            std::size_t expected_size_bytes, bool payload_pinned,
                            unsigned int cuda_host_register_flags) noexcept
      -> std::optional<AttachedSegment>;

 private:
  struct Key {
    std::uint64_t owner_pid{0U};
    std::uint32_t shm_pool_id{0U};

    [[nodiscard]] friend bool operator==(const Key& lhs, const Key& rhs) noexcept {
      return lhs.owner_pid == rhs.owner_pid && lhs.shm_pool_id == rhs.shm_pool_id;
    }
  };

  struct KeyHash {
    [[nodiscard]] std::size_t operator()(const Key& key) const noexcept {
      return std::hash<std::uint64_t>{}(key.owner_pid) ^
             (std::hash<std::uint32_t>{}(key.shm_pool_id) << 1U);
    }
  };

  struct Entry {
    std::unique_ptr<HostBufferPoolMemory> backing;
  };

  std::mutex mutex_;
  std::unordered_map<Key, Entry, KeyHash> entries_;
};

}  // namespace holomq
