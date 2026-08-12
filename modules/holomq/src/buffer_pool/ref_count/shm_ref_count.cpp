/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "buffer_pool/ref_count/shm_ref_count.hpp"

#include <holomq/logger.hpp>
#include "buffer_pool/pool_layout.hpp"
#include "buffer_pool/splitmix64_id.hpp"

#include <atomic>
#include <cassert>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>
#include <sstream>
#include <string>
#include <utility>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace holomq {

namespace {

[[nodiscard]] bool check_lock_free() noexcept {
  return std::atomic<std::uint32_t>::is_always_lock_free;
}

[[nodiscard]] std::string generate_shm_name(std::uint64_t owner_pid,
                                            std::uint32_t shm_ref_count_id) noexcept {
  std::ostringstream name;
  name << "/holomq_ref_" << std::hex << owner_pid << "_" << shm_ref_count_id;
  return name.str();
}

[[nodiscard]] std::byte* slots_base(void* mapped_base) noexcept {
  return static_cast<std::byte*>(mapped_base) + sizeof(ShmRefCountTable::Header);
}

/// Return the next per-process ref-count SHM segment id (\c buffer_pool/splitmix64_id.hpp).
[[nodiscard]] std::uint32_t allocate_shm_ref_count_id() noexcept {
  const std::uint32_t id = static_cast<std::uint32_t>(detail::next_splitmix64_nonzero_id());
  return id == 0U ? 1U : id;
}

/// Unlink a stale ref-count SHM object left behind after a prior crash in this process.
///
/// Opens \p shm_name read-only and unlinks it only when the mapped header's \c owner_pid matches
/// \p owner_pid and every slot ref-count is zero. Returns \c false when the object is missing, in
/// use, owned by another process, or cannot be mapped.
[[nodiscard]] bool try_unlink_stale_ref_count_segment(const std::string& shm_name,
                                                      const std::uint64_t owner_pid) noexcept {
  const int fd = ::shm_open(shm_name.c_str(), O_RDWR, 0);
  if (fd < 0) {
    return false;
  }

  struct ::stat st{};
  if (::fstat(fd, &st) != 0 || st.st_size < static_cast<off_t>(sizeof(ShmRefCountTable::Header))) {
    ::close(fd);
    return false;
  }

  void* const mapped =
      ::mmap(nullptr, static_cast<std::size_t>(st.st_size), PROT_READ, MAP_SHARED, fd, 0);
  ::close(fd);
  if (mapped == MAP_FAILED) {
    return false;
  }

  const auto* const header = static_cast<const ShmRefCountTable::Header*>(mapped);
  const bool owner_matches = header->owner_pid == owner_pid;
  const std::uint32_t buffer_count = header->buffer_count;
  bool all_slots_zero = true;
  if (owner_matches && buffer_count > 0U) {
    const auto* const slots = reinterpret_cast<const ShmRefCountTable::Slot*>(
        static_cast<const std::byte*>(mapped) + sizeof(ShmRefCountTable::Header));
    for (std::uint32_t index = 0U; index < buffer_count; ++index) {
      if (slots[index].ref_count.load(std::memory_order_relaxed) != 0U) {
        all_slots_zero = false;
        break;
      }
    }
  } else {
    all_slots_zero = false;
  }

  (void)::munmap(mapped, static_cast<std::size_t>(st.st_size));

  if (!owner_matches || !all_slots_zero) {
    return false;
  }

  return ::shm_unlink(shm_name.c_str()) == 0;
}

/// Create the ref-count SHM object named \p shm_name with \c O_CREAT | \c O_EXCL.
///
/// On \c EEXIST, call try_unlink_stale_ref_count_segment and retry once. Returns empty when
/// creation still fails.
[[nodiscard]] std::optional<SharedMemory> open_ref_count_shm_for_create(
    const std::string& shm_name, const std::uint64_t owner_pid) noexcept {
  SharedMemory shm(shm_name, true);
  if (static_cast<bool>(shm)) {
    return shm;
  }

  if (errno == EEXIST && try_unlink_stale_ref_count_segment(shm_name, owner_pid)) {
    shm = SharedMemory(shm_name, true);
    if (static_cast<bool>(shm)) {
      return shm;
    }
  }

  return std::nullopt;
}

/// Maximum \c shm_open create attempts in \ref ShmRefCountMapping::create before failing.
constexpr int k_shm_ref_count_create_attempts = 16;

}  // namespace

std::size_t ShmRefCountTable::mapped_size_bytes(const std::uint32_t buffer_count) noexcept {
  const auto slots_bytes_or = checked_product(static_cast<std::size_t>(buffer_count), sizeof(Slot));
  if (!slots_bytes_or.has_value()) {
    return 0U;
  }
  const std::size_t slots_bytes = slots_bytes_or.value();
  if (slots_bytes > (std::numeric_limits<std::size_t>::max() - sizeof(Header))) {
    return 0U;
  }
  return sizeof(Header) + slots_bytes;
}

ShmRefCountTable::ShmRefCountTable(void* mapped_base) noexcept
    : header_(static_cast<const Header*>(mapped_base)),
      slots_(reinterpret_cast<Slot*>(slots_base(mapped_base))) {}

ShmRefCountTable::ShmRefCountTable(void* mapped_base, Header header) noexcept {
  auto* const mapped_header = static_cast<Header*>(mapped_base);
  new (mapped_header) Header(header);
  header_ = mapped_header;
  slots_ = reinterpret_cast<Slot*>(slots_base(mapped_base));
  for (std::uint32_t index = 0U; index < header_->buffer_count; ++index) {
    slot_at(index).ref_count.store(0U, std::memory_order_relaxed);
  }
}

ShmRefCountTable::Slot& ShmRefCountTable::slot_at(const std::uint32_t index) noexcept {
  assert(header_ != nullptr);
  assert(index < header_->buffer_count);
  assert(slots_ != nullptr);
  return slots_[index];
}

const ShmRefCountTable::Slot& ShmRefCountTable::slot_at(const std::uint32_t index) const noexcept {
  assert(header_ != nullptr);
  assert(index < header_->buffer_count);
  assert(slots_ != nullptr);
  return slots_[index];
}

bool ShmRefCountTable::matches_segment(const std::uint64_t owner_pid,
                                       const std::uint32_t shm_ref_count_id) const noexcept {
  return header_ != nullptr && header_->owner_pid == owner_pid &&
         header_->shm_ref_count_id == shm_ref_count_id;
}

bool ShmRefCountTable::try_acquire_slot(const std::uint32_t index) noexcept {
  if (header_ == nullptr || slots_ == nullptr || index >= header_->buffer_count) {
    return false;
  }
  std::uint32_t expected = 0U;
  return slot_at(index).ref_count.compare_exchange_strong(
      expected, 1U, std::memory_order_acq_rel, std::memory_order_relaxed);
}

void ShmRefCountTable::add_ref(const std::uint32_t index) noexcept {
  if (header_ == nullptr || slots_ == nullptr || index >= header_->buffer_count) {
    return;
  }
  auto& ref_count = slot_at(index).ref_count;
  std::uint32_t expected = ref_count.load(std::memory_order_relaxed);
  while (expected != std::numeric_limits<std::uint32_t>::max()) {
    if (ref_count.compare_exchange_weak(
            expected, expected + 1U, std::memory_order_relaxed, std::memory_order_relaxed)) {
      return;
    }
  }
}

void ShmRefCountTable::release_ref(const std::uint32_t index) noexcept {
  if (header_ == nullptr || slots_ == nullptr || index >= header_->buffer_count) {
    return;
  }
  auto& ref_count = slot_at(index).ref_count;
  std::uint32_t expected = ref_count.load(std::memory_order_acquire);
  while (expected != 0U) {
    if (ref_count.compare_exchange_weak(
            expected, expected - 1U, std::memory_order_acq_rel, std::memory_order_relaxed)) {
      return;
    }
  }
}

std::uint32_t ShmRefCountTable::ref_count(const std::uint32_t index) const noexcept {
  if (header_ == nullptr || slots_ == nullptr || index >= header_->buffer_count) {
    return 0U;
  }
  return slot_at(index).ref_count.load(std::memory_order_acquire);
}

ShmRefCountMapping::ShmRefCountMapping(ShmRefCountTable table, SharedMemory shm) noexcept
    : table_(table), shm_(std::move(shm)) {}

ShmRefCountMapping::ShmRefCountMapping(ShmRefCountMapping&& other) noexcept : ShmRefCountMapping() {
  swap(*this, other);
}

ShmRefCountMapping& ShmRefCountMapping::operator=(ShmRefCountMapping&& other) noexcept {
  swap(*this, other);
  return *this;
}

auto ShmRefCountMapping::create(const std::uint32_t buffer_count) noexcept
    -> std::optional<ShmRefCountMappingPtr> {
  if (!check_lock_free()) {
    HOLOMQ_LOG_ERROR("SHM ref-count mapping failed: atomic not lock free");
    return std::nullopt;
  }
  if (buffer_count == 0U) {
    HOLOMQ_LOG_ERROR("SHM ref-count mapping failed: buffer count zero");
    return std::nullopt;
  }

  const std::uint64_t owner_pid = static_cast<std::uint64_t>(::getpid());

  // Retry with a fresh shm_ref_count_id when O_EXCL fails (e.g. orphaned segment after a prior
  // crash with the same owner PID). Fail closed after a bounded number of attempts.
  for (int attempt = 0; attempt < k_shm_ref_count_create_attempts; ++attempt) {
    const std::uint32_t shm_ref_count_id = allocate_shm_ref_count_id();
    const std::string shm_name = generate_shm_name(owner_pid, shm_ref_count_id);

    auto shm_or = open_ref_count_shm_for_create(shm_name, owner_pid);
    if (!shm_or.has_value()) {
      continue;
    }
    SharedMemory shm = std::move(shm_or).value();

    const std::size_t mapped_size = ShmRefCountTable::mapped_size_bytes(buffer_count);
    if (mapped_size == 0U) {
      HOLOMQ_LOG_ERROR("SHM ref-count mapping failed: mapped size overflow");
      return std::nullopt;
    }
    if (::ftruncate(shm.fd_get(), static_cast<off_t>(mapped_size)) != 0) {
      HOLOMQ_LOG_ERROR("SHM ref-count mapping failed: ftruncate failed");
      return std::nullopt;
    }

    if (!shm.map(mapped_size)) {
      HOLOMQ_LOG_ERROR("SHM ref-count mapping failed: mmap failed");
      return std::nullopt;
    }

    const ShmRefCountTable table(
        shm.mapped_base(), ShmRefCountTable::Header(owner_pid, shm_ref_count_id, buffer_count));

    ShmRefCountMappingPtr mapping{new (std::nothrow) ShmRefCountMapping(table, std::move(shm))};
    if (!mapping) {
      HOLOMQ_LOG_ERROR("SHM ref-count mapping failed: allocation failed");
      return std::nullopt;
    }
    return mapping;
  }

  HOLOMQ_LOG_ERROR("SHM ref-count mapping failed: shm_open failed after retries (%s)",
                   std::strerror(errno));
  return std::nullopt;
}

auto ShmRefCountMapping::attach(const std::uint64_t owner_pid,
                                const std::uint32_t shm_ref_count_id) noexcept
    -> std::optional<ShmRefCountMappingPtr> {
  if (!check_lock_free()) {
    HOLOMQ_LOG_ERROR("SHM ref-count mapping failed: atomic not lock free");
    return std::nullopt;
  }

  // Attach to the shared memory object
  SharedMemory shm(generate_shm_name(owner_pid, shm_ref_count_id), false);
  if (!shm) {
    HOLOMQ_LOG_ERROR("SHM ref-count mapping failed: shm_open failed");
    return std::nullopt;
  }

  // Validate the shared memory object
  struct ::stat st{};
  if (::fstat(shm.fd_get(), &st) != 0 || st.st_size <= 0) {
    HOLOMQ_LOG_ERROR("SHM ref-count mapping failed: fstat failed");
    return std::nullopt;
  }

  const std::size_t mapped_size = static_cast<std::size_t>(st.st_size);
  if (mapped_size < sizeof(ShmRefCountTable::Header)) {
    HOLOMQ_LOG_ERROR("SHM ref-count mapping failed: segment too small");
    return std::nullopt;
  }

  // Map the shared memory
  if (!shm.map(mapped_size)) {
    HOLOMQ_LOG_ERROR("SHM ref-count mapping failed: mmap failed");
    return std::nullopt;
  }

  // Create the ref-count table
  const ShmRefCountTable table(shm.mapped_base());
  if (table.buffer_count() == 0U) {
    HOLOMQ_LOG_ERROR("SHM ref-count mapping failed: buffer count zero");
    return std::nullopt;
  }

  const std::size_t expected_size = ShmRefCountTable::mapped_size_bytes(table.buffer_count());
  if (expected_size == 0U) {
    HOLOMQ_LOG_ERROR("SHM ref-count mapping failed: invalid buffer count");
    return std::nullopt;
  }
  if (mapped_size < expected_size) {
    HOLOMQ_LOG_ERROR("SHM ref-count mapping failed: segment smaller than header layout");
    return std::nullopt;
  }

  // Validate the ref-count table
  if (!table.matches_segment(owner_pid, shm_ref_count_id)) {
    HOLOMQ_LOG_ERROR("SHM ref-count mapping failed: header mismatch");
    return std::nullopt;
  }

  ShmRefCountMappingPtr mapping{new (std::nothrow) ShmRefCountMapping(table, std::move(shm))};
  if (!mapping) {
    HOLOMQ_LOG_ERROR("SHM ref-count mapping failed: allocation failed");
    return std::nullopt;
  }
  return mapping;
}

}  // namespace holomq
