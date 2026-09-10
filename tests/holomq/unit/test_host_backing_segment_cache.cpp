/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "buffer_pool/host/host_backing_segment_cache.hpp"
#include "buffer_pool/host/shm_host_buffer_pool_memory.hpp"
#include "buffer_pool/ref_count/shared_memory.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <thread>
#include <vector>

#include <sys/stat.h>
#include <unistd.h>

namespace {

constexpr std::size_t kSegmentBytes = 4096U;
constexpr std::size_t kOversizedExpectedBytes = 8192U;

[[nodiscard]] std::uint64_t owner_pid() noexcept {
  return static_cast<std::uint64_t>(::getpid());
}

}  // namespace

TEST(HostBackingSegmentCacheTest, AttachRejectsSegmentSmallerThanExpectedSize) {
  constexpr std::uint32_t shm_pool_id = 9001U;
  holomq::SharedMemory shm(holomq::ShmHostBufferPoolMemory::shm_name(owner_pid(), shm_pool_id),
                           true);
  ASSERT_TRUE(shm);
  ASSERT_EQ(::ftruncate(shm.fd_get(), static_cast<off_t>(kSegmentBytes)), 0);

  holomq::HostBackingSegmentCache cache;
  EXPECT_FALSE(cache.attach(owner_pid(), shm_pool_id, kOversizedExpectedBytes, false, 0U));
}

TEST(HostBackingSegmentCacheTest, AttachMapsSegmentWhenSizeMatchesExpected) {
  constexpr std::uint32_t shm_pool_id = 9002U;
  holomq::SharedMemory shm(holomq::ShmHostBufferPoolMemory::shm_name(owner_pid(), shm_pool_id),
                           true);
  ASSERT_TRUE(shm);
  ASSERT_EQ(::ftruncate(shm.fd_get(), static_cast<off_t>(kSegmentBytes)), 0);

  holomq::HostBackingSegmentCache cache;
  const auto segment_or = cache.attach(owner_pid(), shm_pool_id, kSegmentBytes, false, 0U);
  ASSERT_TRUE(segment_or.has_value());
  EXPECT_NE(segment_or->base, nullptr);
  EXPECT_EQ(segment_or->size_bytes, kSegmentBytes);
  EXPECT_FALSE(segment_or->payload_pinned);
}

TEST(HostBackingSegmentCacheTest, ConcurrentMismatchedMetadataIsRejected) {
  constexpr std::uint32_t kShmPoolId = 9003U;
  holomq::SharedMemory shm(holomq::ShmHostBufferPoolMemory::shm_name(owner_pid(), kShmPoolId),
                           true);
  ASSERT_TRUE(shm);
  ASSERT_EQ(::ftruncate(shm.fd_get(), static_cast<off_t>(kSegmentBytes)), 0);

  constexpr std::size_t kThreadCount = 8U;
  holomq::HostBackingSegmentCache cache;
  std::vector<unsigned int> requested_flags(kThreadCount);
  std::vector<std::optional<holomq::HostBackingSegmentCache::AttachedSegment>> results(
      kThreadCount);
  std::atomic<std::size_t> ready_count{0U};
  std::atomic<bool> start{false};
  std::vector<std::thread> threads;
  threads.reserve(kThreadCount);

  for (std::size_t i = 0U; i < kThreadCount; ++i) {
    requested_flags[i] = static_cast<unsigned int>(i % 2U);
    threads.emplace_back([&, i] {
      ready_count.fetch_add(1U, std::memory_order_release);
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      results[i] = cache.attach(owner_pid(), kShmPoolId, kSegmentBytes, false, requested_flags[i]);
    });
  }
  while (ready_count.load(std::memory_order_acquire) != kThreadCount) {
    std::this_thread::yield();
  }
  start.store(true, std::memory_order_release);
  for (auto& thread : threads) {
    thread.join();
  }

  std::size_t success_count = 0U;
  for (std::size_t i = 0U; i < kThreadCount; ++i) {
    if (!results[i].has_value()) {
      continue;
    }
    ++success_count;
    EXPECT_EQ(results[i]->host_register_flags, requested_flags[i])
        << "a racing caller must not receive a cache entry with mismatched metadata";
  }
  EXPECT_EQ(success_count, kThreadCount / 2U)
      << "only callers whose metadata matches the cached winner may succeed";
}
