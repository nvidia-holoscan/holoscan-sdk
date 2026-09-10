/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "buffer_pool/ref_count/shared_memory.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>

namespace holomq {

class ShmRefCountMapping;

/// View over a POSIX SHM mapping: header and per-slot ref-count state.
class ShmRefCountTable {
 public:
  /// Per-buffer slot in the ref-count SHM segment (standard-layout for mapped memory).
  struct Slot {
    std::atomic<std::uint32_t> ref_count{0};
  };

  static_assert(sizeof(Slot) == 4U);
  static_assert(alignof(Slot) == 4U);

  /// Header at the start of the ref-count SHM mapping (standard-layout for mapped memory).
  struct Header {
    std::uint64_t owner_pid{0};
    std::uint32_t shm_ref_count_id{0};
    std::uint32_t buffer_count{0};

    Header() noexcept = default;

    Header(std::uint64_t owner_pid, std::uint32_t shm_ref_count_id,
           std::uint32_t buffer_count) noexcept
        : owner_pid(owner_pid), shm_ref_count_id(shm_ref_count_id), buffer_count(buffer_count) {}
  };

  static_assert(sizeof(Header) == 16U);

  ShmRefCountTable() noexcept = default;

  [[nodiscard]] std::uint64_t owner_pid() const noexcept { return owner_pid_; }

  [[nodiscard]] std::uint32_t shm_ref_count_id() const noexcept { return shm_ref_count_id_; }

  [[nodiscard]] std::uint32_t buffer_count() const noexcept { return buffer_count_; }

  [[nodiscard]] bool is_mapped() const noexcept { return header_ != nullptr; }

  /// Returns true if the mapped header matches expected segment identity.
  [[nodiscard]] bool matches_segment(std::uint64_t owner_pid,
                                     std::uint32_t shm_ref_count_id) const noexcept;

  /// Owner acquire: CAS slot from 0 to 1. Returns false if the slot is in use.
  [[nodiscard]] bool try_acquire_slot(std::uint32_t index) noexcept;

  /// Remote view lease: increment shared ref count; no-op when saturated at \c UINT32_MAX.
  void add_ref(std::uint32_t index) noexcept;

  /// Drop a ref; no-op if the slot is already zero.
  void release_ref(std::uint32_t index) noexcept;

  [[nodiscard]] std::uint32_t ref_count(std::uint32_t index) const noexcept;

 private:
  friend class ShmRefCountMapping;

  [[nodiscard]] static std::size_t mapped_size_bytes(std::uint32_t buffer_count) noexcept;

  explicit ShmRefCountTable(void* mapped_base) noexcept;

  ShmRefCountTable(void* mapped_base, Header header) noexcept;

  [[nodiscard]] Slot& slot_at(std::uint32_t index) noexcept;
  [[nodiscard]] const Slot& slot_at(std::uint32_t index) const noexcept;

  /// Captures immutable header fields into process-local storage. Peers map the segment
  /// \c O_RDWR / \c PROT_WRITE, so live \c header_->buffer_count must not be re-read for bounds.
  void capture_header_snapshot(const Header& header) noexcept {
    owner_pid_ = header.owner_pid;
    shm_ref_count_id_ = header.shm_ref_count_id;
    buffer_count_ = header.buffer_count;
  }

  const Header* header_{nullptr};
  Slot* slots_{nullptr};
  std::uint64_t owner_pid_{0U};
  std::uint32_t shm_ref_count_id_{0U};
  std::uint32_t buffer_count_{0U};
};

/// Shared ownership of a ref-count SHM mapping (keeps the segment mapped).
using ShmRefCountMappingPtr = std::shared_ptr<ShmRefCountMapping>;

/// Owning mapping of a ref-count SHM object (creator or remote).
class ShmRefCountMapping {
 public:
  ShmRefCountMapping() noexcept = default;
  ShmRefCountMapping(ShmRefCountMapping&&) noexcept;
  ShmRefCountMapping& operator=(ShmRefCountMapping&&) noexcept;
  ShmRefCountMapping(const ShmRefCountMapping&) = delete;
  ShmRefCountMapping& operator=(const ShmRefCountMapping&) = delete;
  ~ShmRefCountMapping() = default;

  [[nodiscard]] explicit operator bool() const noexcept { return table_.is_mapped(); }

  [[nodiscard]] ShmRefCountTable& table() noexcept { return table_; }

  [[nodiscard]] const ShmRefCountTable& table() const noexcept { return table_; }

  [[nodiscard]] const std::string& shm_name() const noexcept { return shm_.name(); }

  [[nodiscard]] std::uint64_t owner_pid() const noexcept { return table_.owner_pid(); }

  [[nodiscard]] std::uint32_t shm_ref_count_id() const noexcept {
    return table_.shm_ref_count_id();
  }

  [[nodiscard]] static auto create(std::uint32_t buffer_count) noexcept
      -> std::optional<ShmRefCountMappingPtr>;

  [[nodiscard]] static auto attach(std::uint64_t owner_pid, std::uint32_t shm_ref_count_id) noexcept
      -> std::optional<ShmRefCountMappingPtr>;

  friend void swap(ShmRefCountMapping& lhs, ShmRefCountMapping& rhs) noexcept {
    using std::swap;
    swap(lhs.table_, rhs.table_);
    swap(lhs.shm_, rhs.shm_);
  }

 private:
  ShmRefCountMapping(ShmRefCountTable table, SharedMemory shm) noexcept;

  SharedMemory shm_;
  ShmRefCountTable table_{};
};

}  // namespace holomq
