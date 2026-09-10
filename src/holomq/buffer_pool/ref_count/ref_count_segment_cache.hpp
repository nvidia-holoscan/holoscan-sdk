/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <holomq/buffer_handle.hpp>
#include "buffer_pool/ref_count/ref_count_key.hpp"
#include "buffer_pool/ref_count/shm_ref_count.hpp"

#include <cstdint>
#include <functional>
#include <mutex>
#include <optional>
#include <unordered_map>

namespace holomq {

/// Process-local cache of mapped ref-count SHM segments keyed by \c owner_pid + \c
/// shm_ref_count_id.
class RefCountSegmentCache {
 public:
  /// Returns shared ownership of the ref-count SHM mapping for the publisher segment named in \p
  /// handle.
  ///
  /// On first use for a given \c (owner_pid, shm_ref_count_id), attaches the POSIX SHM object
  /// under the cache mutex and stores the mapping. Later calls reuse the cached
  /// \ref ShmRefCountMapping. Validates segment identity fields and that \c slot_index is within
  /// \ref ShmRefCountTable::buffer_count.
  ///
  /// The returned \ref ShmRefCountMappingPtr pins the mapping until the last holder or deferred
  /// release callback drops it, even if this cache is destroyed.
  [[nodiscard]] auto map(const BufferHandle& handle) noexcept
      -> std::optional<ShmRefCountMappingPtr>;

 private:
  struct Key {
    std::uint64_t owner_pid{0};
    std::uint32_t shm_ref_count_id{0};

    [[nodiscard]] explicit Key(const RefCountKey& key) noexcept
        : owner_pid(key.owner_pid), shm_ref_count_id(key.shm_ref_count_id) {}

    [[nodiscard]] friend bool operator==(const Key& lhs, const Key& rhs) noexcept {
      return lhs.owner_pid == rhs.owner_pid && lhs.shm_ref_count_id == rhs.shm_ref_count_id;
    }
  };

  struct KeyHash {
    [[nodiscard]] std::size_t operator()(const Key& key) const noexcept {
      return std::hash<std::uint64_t>{}(key.owner_pid) ^
             (std::hash<std::uint32_t>{}(key.shm_ref_count_id) << 1U);
    }
  };

  std::mutex mutex_;
  std::unordered_map<Key, ShmRefCountMappingPtr, KeyHash> entries_;
};

}  // namespace holomq
