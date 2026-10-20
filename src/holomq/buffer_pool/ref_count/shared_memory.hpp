/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../../scoped_file_descriptor.hpp"

#include <cstddef>
#include <optional>
#include <string>
#include <utility>

namespace holomq {

/// Owns a POSIX shared-memory object name and fd; optionally unlinks the name on teardown.
class SharedMemory {
 public:
  SharedMemory() noexcept = default;

  /// Opens or creates the named object. When \c create is \c true, uses \c O_CREAT | \c O_RDWR | \c
  /// O_EXCL only (no fallback if the name already exists). Unlinks the name on destruction when
  /// create succeeds. When \c create is \c false, opens an existing object with \c O_RDWR. Leaves
  /// the fd empty if
  /// \c shm_open fails.
  SharedMemory(std::string name, bool create) noexcept;

  SharedMemory(SharedMemory&&) noexcept;
  SharedMemory& operator=(SharedMemory&&) noexcept;
  SharedMemory(const SharedMemory&) = delete;
  SharedMemory& operator=(const SharedMemory&) = delete;

  ~SharedMemory() = default;

  [[nodiscard]] explicit operator bool() const noexcept { return static_cast<bool>(fd_); }

  [[nodiscard]] const std::string& name() const noexcept { return name_; }

  [[nodiscard]] const ScopedFileDescriptor& fd() const noexcept { return fd_; }

  [[nodiscard]] int fd_get() const noexcept { return fd_.get(); }

  [[nodiscard]] bool has_unlinker() const noexcept { return unlinker_.has_value(); }

  [[nodiscard]] bool is_mapped() const noexcept { return mapping_.has_value(); }

  [[nodiscard]] void* mapped_base() const noexcept {
    return mapping_.has_value() ? mapping_->base() : nullptr;
  }

  [[nodiscard]] std::size_t mapped_size() const noexcept {
    return mapping_.has_value() ? mapping_->size() : 0U;
  }

  /// Maps \c size bytes from the open fd. Returns \c false if already mapped or \c mmap fails.
  [[nodiscard]] bool map(std::size_t size) noexcept;

  friend void swap(SharedMemory& lhs, SharedMemory& rhs) noexcept {
    using std::swap;
    swap(lhs.name_, rhs.name_);
    swap(lhs.fd_, rhs.fd_);
    swap(lhs.unlinker_, rhs.unlinker_);
    swap(lhs.mapping_, rhs.mapping_);
  }

 private:
  class Mapping {
   public:
    Mapping() noexcept = default;

    Mapping(void* base, std::size_t size) noexcept;

    Mapping(Mapping&&) noexcept;
    Mapping& operator=(Mapping&&) noexcept;
    Mapping(const Mapping&) = delete;
    Mapping& operator=(const Mapping&) = delete;

    ~Mapping();

    [[nodiscard]] void* base() const noexcept { return base_; }

    [[nodiscard]] std::size_t size() const noexcept { return size_; }

   private:
    void* base_{nullptr};
    std::size_t size_{0};
  };

  class Unlinker {
   public:
    Unlinker() noexcept = default;

    explicit Unlinker(std::string shm_name) noexcept;

    Unlinker(Unlinker&&) noexcept;
    Unlinker& operator=(Unlinker&&) noexcept;
    Unlinker(const Unlinker&) = delete;
    Unlinker& operator=(const Unlinker&) = delete;

    ~Unlinker();

    void reset() noexcept;

   private:
    std::string shm_name_;
  };

  std::string name_;
  ScopedFileDescriptor fd_;
  std::optional<Unlinker> unlinker_;
  std::optional<Mapping> mapping_;
};

}  // namespace holomq
