/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "buffer_pool/ref_count/ref_count_segment_cache.hpp"

#include <holomq/logger.hpp>

#include <utility>

namespace holomq {

auto RefCountSegmentCache::map(const BufferHandle& handle) noexcept
    -> std::optional<ShmRefCountMappingPtr> {
  const auto key_or = ref_count_key(handle);
  if (!key_or.has_value()) {
    HOLOMQ_LOG_ERROR("ref-count segment cache: invalid buffer handle");
    return std::nullopt;
  }
  const RefCountKey& key = key_or.value();

  const std::uint32_t slot_index = detail::BufferHandleWireAccess::slot_index(handle);

  const std::lock_guard<std::mutex> lock(mutex_);

  const auto found = entries_.find(Key{key});
  if (found != entries_.end()) {
    const ShmRefCountTable& table = found->second->table();
    if (!table.matches_segment(key.owner_pid, key.shm_ref_count_id)) {
      HOLOMQ_LOG_ERROR("ref-count segment cache: header mismatch");
      return std::nullopt;
    }
    if (slot_index >= table.buffer_count()) {
      HOLOMQ_LOG_ERROR("ref-count segment cache: slot index out of range");
      return std::nullopt;
    }
    return found->second;
  }

  auto owner_or = ShmRefCountMapping::attach(key.owner_pid, key.shm_ref_count_id);
  if (!owner_or.has_value()) {
    return std::nullopt;
  }
  ShmRefCountMappingPtr owner = std::move(owner_or).value();

  const ShmRefCountTable& table = owner->table();
  if (!table.matches_segment(key.owner_pid, key.shm_ref_count_id)) {
    HOLOMQ_LOG_ERROR("ref-count segment cache: header mismatch");
    return std::nullopt;
  }
  if (slot_index >= table.buffer_count()) {
    HOLOMQ_LOG_ERROR("ref-count segment cache: slot index out of range");
    return std::nullopt;
  }

  auto [iter, inserted] = entries_.emplace(Key{key}, std::move(owner));
  static_cast<void>(inserted);
  return iter->second;
}

}  // namespace holomq
