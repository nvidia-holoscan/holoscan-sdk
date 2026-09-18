/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <new>

#include "allocation_counter.hpp"

namespace {

constinit thread_local bool g_count_test_allocations = false;
constinit thread_local std::size_t g_test_allocation_count = 0;
constinit thread_local bool g_fail_test_allocation = false;
constinit thread_local std::size_t g_successful_allocations_before_failure = 0;
constinit thread_local bool g_exhaust_test_allocations = false;
constinit thread_local std::size_t g_successful_allocations_before_exhaustion = 0;
constinit thread_local bool g_fail_next_test_aligned_allocation = false;

void fail_test_allocation_if_requested() {
  if (g_fail_test_allocation && g_successful_allocations_before_failure == 0U) {
    g_fail_test_allocation = false;
    throw std::bad_alloc();
  }
  if (g_exhaust_test_allocations) {
    if (g_successful_allocations_before_exhaustion == 0U) {
      throw std::bad_alloc();
    }
  }
  if (g_fail_test_allocation) {
    --g_successful_allocations_before_failure;
  }
  if (g_exhaust_test_allocations) {
    --g_successful_allocations_before_exhaustion;
  }
}

void fail_test_aligned_allocation_if_requested() {
  if (!g_fail_next_test_aligned_allocation) {
    return;
  }
  g_fail_next_test_aligned_allocation = false;
  throw std::bad_alloc();
}

void* counted_malloc(std::size_t size) {
  fail_test_allocation_if_requested();
  if (g_count_test_allocations) {
    ++g_test_allocation_count;
  }
  if (void* ptr = std::malloc(size == 0 ? 1 : size)) {
    return ptr;
  }
  throw std::bad_alloc();
}

void* counted_aligned_malloc(std::size_t size, std::align_val_t alignment) {
  fail_test_aligned_allocation_if_requested();
  fail_test_allocation_if_requested();
  if (g_count_test_allocations) {
    ++g_test_allocation_count;
  }
  void* ptr = nullptr;
  const auto align = std::max(static_cast<std::size_t>(alignment), sizeof(void*));
  if (posix_memalign(&ptr, align, size == 0 ? 1 : size) == 0) {
    return ptr;
  }
  throw std::bad_alloc();
}

void* counted_nothrow_malloc(const std::size_t size) noexcept {
  try {
    return counted_malloc(size);
  } catch (...) {
    return nullptr;
  }
}

void* counted_nothrow_aligned_malloc(const std::size_t size,
                                     const std::align_val_t alignment) noexcept {
  try {
    return counted_aligned_malloc(size, alignment);
  } catch (...) {
    return nullptr;
  }
}

}  // namespace

namespace holoscan::test {

ScopedAllocationCounter::ScopedAllocationCounter() noexcept
    : previous_enabled_(g_count_test_allocations), previous_count_(g_test_allocation_count) {
  g_test_allocation_count = 0;
  g_count_test_allocations = true;
}

ScopedAllocationCounter::~ScopedAllocationCounter() noexcept {
  g_count_test_allocations = previous_enabled_;
  if (previous_enabled_) {
    g_test_allocation_count += previous_count_;
  }
}

std::size_t ScopedAllocationCounter::count() const noexcept {
  return g_test_allocation_count;
}

ScopedAllocationFailure::ScopedAllocationFailure(
    const std::size_t successful_allocations_before_failure) noexcept
    : previous_enabled_(g_fail_test_allocation),
      previous_remaining_(g_successful_allocations_before_failure) {
  g_fail_test_allocation = true;
  g_successful_allocations_before_failure = successful_allocations_before_failure;
}

ScopedAllocationFailure::~ScopedAllocationFailure() noexcept {
  g_fail_test_allocation = previous_enabled_;
  g_successful_allocations_before_failure = previous_remaining_;
}

ScopedAllocationExhaustion::ScopedAllocationExhaustion(
    const std::size_t successful_allocations_before_failure) noexcept
    : previous_enabled_(g_exhaust_test_allocations),
      previous_remaining_(g_successful_allocations_before_exhaustion) {
  g_exhaust_test_allocations = true;
  g_successful_allocations_before_exhaustion = successful_allocations_before_failure;
}

ScopedAllocationExhaustion::~ScopedAllocationExhaustion() noexcept {
  g_exhaust_test_allocations = previous_enabled_;
  g_successful_allocations_before_exhaustion = previous_remaining_;
}

ScopedAlignedAllocationFailure::ScopedAlignedAllocationFailure() noexcept
    : previous_enabled_(g_fail_next_test_aligned_allocation) {
  g_fail_next_test_aligned_allocation = true;
}

ScopedAlignedAllocationFailure::~ScopedAlignedAllocationFailure() noexcept {
  g_fail_next_test_aligned_allocation = previous_enabled_;
}

}  // namespace holoscan::test

void* operator new(std::size_t size) {
  return counted_malloc(size);
}

void* operator new[](std::size_t size) {
  return counted_malloc(size);
}

void* operator new(std::size_t size, std::align_val_t alignment) {
  return counted_aligned_malloc(size, alignment);
}

void* operator new[](std::size_t size, std::align_val_t alignment) {
  return counted_aligned_malloc(size, alignment);
}

void* operator new(std::size_t size, const std::nothrow_t&) noexcept {
  return counted_nothrow_malloc(size);
}

void* operator new[](std::size_t size, const std::nothrow_t&) noexcept {
  return counted_nothrow_malloc(size);
}

void* operator new(std::size_t size, std::align_val_t alignment, const std::nothrow_t&) noexcept {
  return counted_nothrow_aligned_malloc(size, alignment);
}

void* operator new[](std::size_t size, std::align_val_t alignment, const std::nothrow_t&) noexcept {
  return counted_nothrow_aligned_malloc(size, alignment);
}

void operator delete(void* ptr) noexcept {
  std::free(ptr);
}

void operator delete[](void* ptr) noexcept {
  std::free(ptr);
}

void operator delete(void* ptr, std::size_t) noexcept {
  std::free(ptr);
}

void operator delete[](void* ptr, std::size_t) noexcept {
  std::free(ptr);
}

void operator delete(void* ptr, std::align_val_t) noexcept {
  std::free(ptr);
}

void operator delete[](void* ptr, std::align_val_t) noexcept {
  std::free(ptr);
}

void operator delete(void* ptr, std::size_t, std::align_val_t) noexcept {
  std::free(ptr);
}

void operator delete[](void* ptr, std::size_t, std::align_val_t) noexcept {
  std::free(ptr);
}

void operator delete(void* ptr, const std::nothrow_t&) noexcept {
  std::free(ptr);
}

void operator delete[](void* ptr, const std::nothrow_t&) noexcept {
  std::free(ptr);
}

void operator delete(void* ptr, std::align_val_t, const std::nothrow_t&) noexcept {
  std::free(ptr);
}

void operator delete[](void* ptr, std::align_val_t, const std::nothrow_t&) noexcept {
  std::free(ptr);
}
