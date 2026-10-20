/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "buffer_pool/ref_count/shared_memory.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <string>
#include <utility>

namespace holomq {

SharedMemory::Mapping::Mapping(void* base, std::size_t size) noexcept : base_(base), size_(size) {}

SharedMemory::Mapping::Mapping(Mapping&& other) noexcept : Mapping() {
  std::swap(base_, other.base_);
  std::swap(size_, other.size_);
}

SharedMemory::Mapping& SharedMemory::Mapping::operator=(Mapping&& other) noexcept {
  std::swap(base_, other.base_);
  std::swap(size_, other.size_);
  return *this;
}

SharedMemory::Mapping::~Mapping() {
  if (base_ != nullptr && base_ != MAP_FAILED) {
    (void)::munmap(base_, size_);
  }
}

SharedMemory::Unlinker::Unlinker(std::string shm_name) noexcept : shm_name_(std::move(shm_name)) {}

SharedMemory::Unlinker::Unlinker(Unlinker&& other) noexcept : Unlinker() {
  std::swap(shm_name_, other.shm_name_);
}

SharedMemory::Unlinker& SharedMemory::Unlinker::operator=(Unlinker&& other) noexcept {
  std::swap(shm_name_, other.shm_name_);
  return *this;
}

void SharedMemory::Unlinker::reset() noexcept {
  shm_name_.clear();
}

SharedMemory::Unlinker::~Unlinker() {
  if (!shm_name_.empty()) {
    (void)::shm_unlink(shm_name_.c_str());
  }
}

bool SharedMemory::map(std::size_t size) noexcept {
  if (mapping_.has_value() || !fd_ || size == 0U) {
    return false;
  }

  void* const base = ::mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd_.get(), 0);
  if (base == MAP_FAILED) {
    return false;
  }

  mapping_.emplace(base, size);
  return true;
}

SharedMemory::SharedMemory(std::string name, bool create) noexcept : name_(std::move(name)) {
  if (create) {
    fd_ = ScopedFileDescriptor(
        ::shm_open(name_.c_str(), O_CREAT | O_RDWR | O_EXCL, static_cast<mode_t>(0600)));
    if (fd_) {
      unlinker_.emplace(name_);
    }
  } else {
    fd_ = ScopedFileDescriptor(::shm_open(name_.c_str(), O_RDWR, 0));
  }
}

SharedMemory::SharedMemory(SharedMemory&& other) noexcept : SharedMemory() {
  swap(*this, other);
}

SharedMemory& SharedMemory::operator=(SharedMemory&& other) noexcept {
  swap(*this, other);
  return *this;
}

}  // namespace holomq
