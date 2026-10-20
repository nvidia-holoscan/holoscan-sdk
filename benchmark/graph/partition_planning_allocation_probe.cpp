/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "partition_planning_allocation_probe.hpp"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <new>

namespace {

constexpr std::uint64_t kAllocationMagic = 0x484F4C4F504C414EULL;

struct alignas(std::max_align_t) AllocationHeader {
  void* raw{};
  std::size_t requested{};
  std::uint64_t generation{};
  std::uint64_t magic{};
};

struct ProbeState {
  std::uint64_t generation{};
  std::size_t allocation_count{};
  std::size_t cumulative_requested_bytes{};
  std::size_t peak_live_requested_bytes{};
  std::size_t live_requested_bytes{};
};

constinit thread_local ProbeState g_probe;
constinit thread_local int g_thread_token;
constinit std::atomic<const void*> g_owner_token{nullptr};
constinit std::atomic<std::uint64_t> g_active_generation{0U};
constinit std::atomic_bool g_cross_thread_activity{false};
constinit std::atomic_bool g_cross_generation_activity{false};
constinit std::atomic<std::uint64_t> g_next_generation{1U};

[[nodiscard]] void* allocate(const std::size_t requested, const std::size_t alignment) {
  const std::size_t effective_size = requested == 0U ? 1U : requested;
  const std::size_t effective_alignment = std::max(alignment, alignof(AllocationHeader));
  if (effective_size > std::numeric_limits<std::size_t>::max() - sizeof(AllocationHeader) ||
      effective_size + sizeof(AllocationHeader) >
          std::numeric_limits<std::size_t>::max() - (effective_alignment - 1U)) {
    throw std::bad_alloc();
  }
  const std::size_t bytes = effective_size + sizeof(AllocationHeader) + effective_alignment - 1U;
  void* const raw = std::malloc(bytes);
  if (raw == nullptr) {
    throw std::bad_alloc();
  }
  const auto begin = reinterpret_cast<std::uintptr_t>(raw) + sizeof(AllocationHeader);
  const auto aligned = (begin + effective_alignment - 1U) & ~(effective_alignment - 1U);
  auto* const header = reinterpret_cast<AllocationHeader*>(aligned) - 1;
  const std::uint64_t generation = g_active_generation.load(std::memory_order_relaxed);
  header->raw = raw;
  header->requested = requested;
  header->generation = 0U;
  header->magic = kAllocationMagic;
  if (generation != 0U) {
    if (g_owner_token.load(std::memory_order_relaxed) != &g_thread_token) {
      g_cross_thread_activity.store(true, std::memory_order_relaxed);
    } else if (g_probe.generation == generation) {
      header->generation = generation;
      ++g_probe.allocation_count;
      g_probe.cumulative_requested_bytes += requested;
      g_probe.live_requested_bytes += requested;
      g_probe.peak_live_requested_bytes =
          std::max(g_probe.peak_live_requested_bytes, g_probe.live_requested_bytes);
    }
  }
  return reinterpret_cast<void*>(aligned);
}

void deallocate(void* const pointer) noexcept {
  if (pointer == nullptr) {
    return;
  }
  auto* const header = reinterpret_cast<AllocationHeader*>(pointer) - 1;
  if (header->magic != kAllocationMagic) {
    std::abort();
  }
  const std::uint64_t generation = g_active_generation.load(std::memory_order_relaxed);
  if (header->generation != 0U) {
    if (header->generation != generation) {
      g_cross_generation_activity.store(true, std::memory_order_relaxed);
    } else if (g_owner_token.load(std::memory_order_relaxed) != &g_thread_token ||
               g_probe.generation != generation ||
               header->requested > g_probe.live_requested_bytes) {
      g_cross_thread_activity.store(true, std::memory_order_relaxed);
    } else {
      g_probe.live_requested_bytes -= header->requested;
    }
  }
  header->magic = 0U;
  std::free(header->raw);
}

}  // namespace

namespace holoscan::benchmarking {

PlanningAllocationProbe::PlanningAllocationProbe() noexcept {
  g_probe = {};
  g_probe.generation = g_next_generation.fetch_add(1U, std::memory_order_relaxed);
  g_cross_thread_activity.store(false, std::memory_order_relaxed);
  g_cross_generation_activity.store(false, std::memory_order_relaxed);
  g_owner_token.store(&g_thread_token, std::memory_order_relaxed);
  g_active_generation.store(g_probe.generation, std::memory_order_release);
}

PlanningAllocationProbe::~PlanningAllocationProbe() noexcept {
  if (!finished_) {
    static_cast<void>(finish());
  }
}

PlanningAllocationStats PlanningAllocationProbe::finish() noexcept {
  if (finished_) {
    return {};
  }
  g_active_generation.store(0U, std::memory_order_release);
  g_owner_token.store(nullptr, std::memory_order_relaxed);
  finished_ = true;
  return snapshot();
}

PlanningAllocationStats PlanningAllocationProbe::snapshot() const noexcept {
  return PlanningAllocationStats{
      .allocation_count = g_probe.allocation_count,
      .cumulative_requested_bytes = g_probe.cumulative_requested_bytes,
      .peak_live_requested_bytes = g_probe.peak_live_requested_bytes,
      .live_requested_bytes = g_probe.live_requested_bytes,
      .cross_thread_activity = g_cross_thread_activity.load(std::memory_order_relaxed),
      .cross_generation_activity = g_cross_generation_activity.load(std::memory_order_relaxed),
  };
}

}  // namespace holoscan::benchmarking

void* operator new(const std::size_t size) {
  return allocate(size, alignof(std::max_align_t));
}
void* operator new[](const std::size_t size) {
  return allocate(size, alignof(std::max_align_t));
}
void* operator new(const std::size_t size, const std::align_val_t alignment) {
  return allocate(size, static_cast<std::size_t>(alignment));
}
void* operator new[](const std::size_t size, const std::align_val_t alignment) {
  return allocate(size, static_cast<std::size_t>(alignment));
}
void* operator new(const std::size_t size, const std::nothrow_t&) noexcept {
  try {
    return allocate(size, alignof(std::max_align_t));
  } catch (...) {
    return nullptr;
  }
}
void* operator new[](const std::size_t size, const std::nothrow_t&) noexcept {
  try {
    return allocate(size, alignof(std::max_align_t));
  } catch (...) {
    return nullptr;
  }
}
void* operator new(const std::size_t size, const std::align_val_t alignment,
                   const std::nothrow_t&) noexcept {
  try {
    return allocate(size, static_cast<std::size_t>(alignment));
  } catch (...) {
    return nullptr;
  }
}
void* operator new[](const std::size_t size, const std::align_val_t alignment,
                     const std::nothrow_t&) noexcept {
  try {
    return allocate(size, static_cast<std::size_t>(alignment));
  } catch (...) {
    return nullptr;
  }
}
void operator delete(void* pointer) noexcept {
  deallocate(pointer);
}
void operator delete[](void* pointer) noexcept {
  deallocate(pointer);
}
void operator delete(void* pointer, std::size_t) noexcept {
  deallocate(pointer);
}
void operator delete[](void* pointer, std::size_t) noexcept {
  deallocate(pointer);
}
void operator delete(void* pointer, std::align_val_t) noexcept {
  deallocate(pointer);
}
void operator delete[](void* pointer, std::align_val_t) noexcept {
  deallocate(pointer);
}
void operator delete(void* pointer, std::size_t, std::align_val_t) noexcept {
  deallocate(pointer);
}
void operator delete[](void* pointer, std::size_t, std::align_val_t) noexcept {
  deallocate(pointer);
}
void operator delete(void* pointer, const std::nothrow_t&) noexcept {
  deallocate(pointer);
}
void operator delete[](void* pointer, const std::nothrow_t&) noexcept {
  deallocate(pointer);
}
void operator delete(void* pointer, std::align_val_t, const std::nothrow_t&) noexcept {
  deallocate(pointer);
}
void operator delete[](void* pointer, std::align_val_t, const std::nothrow_t&) noexcept {
  deallocate(pointer);
}
