/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <holomq/buffer_pool.hpp>
#include "buffer_pool/ref_count/shm_ref_count.hpp"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace holomq {

/// Shared \ref BufferPool implementation: ref-count SHM, round-robin \ref acquire, and owner
/// release dispatch.
///
/// \ref acquire, \ref buffer_handle_released, and \ref next_slot_generation are not
/// synchronized; the pool owner must call them from a single publisher thread (typical graph /
/// port usage). \c slot_generations_ is process-local and is only touched on the owner export
/// path (\ref Buffer::export_handle / \ref Buffer::enqueue_ready_set); subscribers never read or
/// increment it, so no lock is required under that contract.
struct BufferPool::Impl {
  virtual ~Impl() = default;

  [[nodiscard]] virtual BufferMemoryType memory_type() const noexcept = 0;
  [[nodiscard]] virtual void* base_ptr() const noexcept = 0;
  [[nodiscard]] virtual BufferHandle handle(std::uint32_t index) const noexcept = 0;

  [[nodiscard]] virtual std::optional<int32_t> cuda_device_ordinal() const noexcept = 0;

  [[nodiscard]] std::size_t buffer_bytes() const noexcept { return buffer_bytes_; }

  [[nodiscard]] std::size_t slot_stride_bytes() const noexcept { return slot_stride_bytes_; }

  [[nodiscard]] ShmRefCountTable& ref_count_table() noexcept { return ref_count_mapping_->table(); }

  [[nodiscard]] const ShmRefCountTable& ref_count_table() const noexcept {
    return ref_count_mapping_->table();
  }

  [[nodiscard]] const ShmRefCountMappingPtr& ref_count_mapping_owner() const noexcept {
    return ref_count_mapping_;
  }

  [[nodiscard]] std::uint32_t buffer_count() const noexcept { return buffer_count_; }

  /// Returns the next available pool slot as a \ref Buffer.
  ///
  /// Scans indices round-robin starting at \c acquire_cursor_. For each candidate, attempts an
  /// owner acquire (\c try_acquire_slot: CAS ref-count 0→1). On success, advances \c
  /// acquire_cursor_ past that index and wraps the slot in \ref buffer_at_index. If every slot is
  /// in use, returns empty.
  [[nodiscard]] auto acquire() noexcept -> std::optional<Buffer>;

  [[nodiscard]] explicit operator bool() const noexcept {
    return base_ptr() != nullptr && buffer_bytes_ != 0U && slot_stride_bytes_ != 0U &&
           buffer_count_ != 0U;
  }

  /// Drops the owner lease on \p index (\ref ShmRefCountTable::release_ref). Not thread-safe; see
  /// class comment.
  void buffer_handle_released(std::uint32_t index) noexcept {
    ref_count_mapping_->table().release_ref(index);
  }

  /// Returns the next monotonic export-generation value for \p index (never reset on acquire).
  /// Not thread-safe; call only from the same publisher thread as \ref acquire.
  [[nodiscard]] std::uint64_t next_slot_generation(std::uint32_t index) noexcept;

 protected:
  explicit Impl(ShmRefCountMappingPtr ref_count_mapping, std::size_t buffer_bytes,
                std::size_t slot_stride_bytes, std::uint32_t buffer_count) noexcept
      : ref_count_mapping_(std::move(ref_count_mapping)),
        buffer_bytes_(buffer_bytes),
        slot_stride_bytes_(slot_stride_bytes),
        buffer_count_(buffer_count),
        slot_generations_(buffer_count, 0U) {
    assert(ref_count_mapping_ != nullptr);
  }

  [[nodiscard]] virtual Buffer buffer_at_index(std::uint32_t index) noexcept = 0;

  /// POSIX SHM segment for cross-process ref-count (shared so deferred callbacks pin it).
  ShmRefCountMappingPtr ref_count_mapping_;
  /// User-visible bytes per slot (excludes per-slot generation word and stride padding).
  std::size_t buffer_bytes_{0};
  /// Byte offset between consecutive slot origins (device interleaved layout or host payload
  /// stride).
  std::size_t slot_stride_bytes_{0};
  /// Number of slots in the pool and in \c ref_count_mapping_.
  std::uint32_t buffer_count_{0};
  /// Round-robin start index for the next \ref acquire scan.
  std::uint32_t acquire_cursor_{0};
  /// Per-slot monotonic export generation (GPU stream-memop fence token source).
  std::vector<std::uint64_t> slot_generations_;
};

}  // namespace holomq
