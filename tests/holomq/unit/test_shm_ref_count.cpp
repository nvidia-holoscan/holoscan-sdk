/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "buffer_pool/ref_count/ref_count_key.hpp"
#include "buffer_pool/ref_count/ref_count_segment_cache.hpp"
#include "buffer_pool/ref_count/shared_memory.hpp"
#include "buffer_pool/ref_count/shm_ref_count.hpp"
#include "buffer_pool/slot_region.hpp"

#include <holomq/buffer_handle.hpp>
#include <holomq/detail/buffer_pool/device/handle_data.hpp>
#include <holomq/detail/buffer_pool/host/handle_data.hpp>

#include <gtest/gtest.h>

#include <atomic>
#include <cstddef>
#include <limits>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <sys/mman.h>

namespace {

constexpr std::uint32_t kBufferCount = 4U;

[[nodiscard]] std::string unique_shm_name() {
  static std::atomic<unsigned int> counter{0U};
  return "/holomq_ref_count_test_" +
         std::to_string(counter.fetch_add(1U, std::memory_order_relaxed));
}

void unlink_if_present(const std::string& name) {
  (void)::shm_unlink(name.c_str());
}

[[nodiscard]] std::string ref_count_shm_name(const std::uint64_t owner_pid,
                                             const std::uint32_t shm_ref_count_id) {
  std::ostringstream name;
  name << "/holomq_ref_" << std::hex << owner_pid << "_" << shm_ref_count_id;
  return name.str();
}

[[nodiscard]] holomq::BufferHandle make_device_ref_count_handle(
    const std::uint64_t owner_pid, const std::uint32_t shm_ref_count_id,
    const std::uint32_t slot_index, const std::uint32_t wire_slot_count = kBufferCount) {
  holomq::detail::device::BufferHandleData handle_data{};
  handle_data.pool_cookie = 0x1000U;
  handle_data.owner_pid = static_cast<std::uint32_t>(owner_pid);
  handle_data.shm_ref_count_id = shm_ref_count_id;
  handle_data.slot_stride_bytes = 264U;
  handle_data.buffer_bytes = 256U;
  handle_data.pool_total_bytes = 264U * wire_slot_count;
  handle_data.cuda_device_ordinal = 0;
  handle_data.fd_pass_socket_path[0] = '/';
  return holomq::detail::make_device_buffer_handle(handle_data, slot_index);
}

[[nodiscard]] std::size_t ref_count_mapped_size_bytes(const std::uint32_t buffer_count) noexcept {
  return sizeof(holomq::ShmRefCountTable::Header) +
         static_cast<std::size_t>(buffer_count) * sizeof(holomq::ShmRefCountTable::Slot);
}

void set_slot_ref_count(void* mapped_base, const std::uint32_t index,
                        const std::uint32_t value) noexcept {
  auto* const slots = reinterpret_cast<holomq::ShmRefCountTable::Slot*>(
      static_cast<std::byte*>(mapped_base) + sizeof(holomq::ShmRefCountTable::Header));
  slots[index].ref_count.store(value, std::memory_order_relaxed);
}

}  // namespace

TEST(ShmRefCountTest, SharedMemoryCreateFailsWhenNameExists) {
  const std::string name = unique_shm_name();
  unlink_if_present(name);

  holomq::SharedMemory first(name, true);
  ASSERT_TRUE(first);

  holomq::SharedMemory second(name, true);
  EXPECT_FALSE(second);

  unlink_if_present(name);
}

TEST(ShmRefCountTest, CreateInitializesZeroRefsAndAttachRoundTrip) {
  auto mapping_or = holomq::ShmRefCountMapping::create(kBufferCount);
  ASSERT_TRUE(mapping_or.has_value()) << "ShmRefCountMapping::create failed";

  const holomq::ShmRefCountMapping& created = *mapping_or.value();
  const holomq::ShmRefCountTable& table = created.table();
  EXPECT_EQ(table.buffer_count(), kBufferCount);
  for (std::uint32_t index = 0U; index < kBufferCount; ++index) {
    EXPECT_EQ(table.ref_count(index), 0U);
  }

  auto attached_or =
      holomq::ShmRefCountMapping::attach(created.owner_pid(), created.shm_ref_count_id());
  ASSERT_TRUE(attached_or.has_value()) << "ShmRefCountMapping::attach failed";
  EXPECT_TRUE(attached_or.value()->table().matches_segment(created.owner_pid(),
                                                           created.shm_ref_count_id()));
}

TEST(ShmRefCountTest, TryAcquireSlotOnlyFromZero) {
  auto mapping_or = holomq::ShmRefCountMapping::create(kBufferCount);
  ASSERT_TRUE(mapping_or.has_value());

  holomq::ShmRefCountTable& table = mapping_or.value()->table();
  EXPECT_TRUE(table.try_acquire_slot(0U));
  EXPECT_EQ(table.ref_count(0U), 1U);
  EXPECT_FALSE(table.try_acquire_slot(0U));
  table.release_ref(0U);
  EXPECT_EQ(table.ref_count(0U), 0U);
  EXPECT_TRUE(table.try_acquire_slot(0U));
}

TEST(ShmRefCountTest, AddRefAndReleaseRefRoundTrip) {
  auto mapping_or = holomq::ShmRefCountMapping::create(kBufferCount);
  ASSERT_TRUE(mapping_or.has_value());

  holomq::ShmRefCountTable& table = mapping_or.value()->table();
  table.add_ref(1U);
  table.add_ref(1U);
  EXPECT_EQ(table.ref_count(1U), 2U);
  table.release_ref(1U);
  EXPECT_EQ(table.ref_count(1U), 1U);
  table.release_ref(1U);
  EXPECT_EQ(table.ref_count(1U), 0U);
}

TEST(ShmRefCountTest, DoubleReleaseDoesNotUnderflow) {
  auto mapping_or = holomq::ShmRefCountMapping::create(kBufferCount);
  ASSERT_TRUE(mapping_or.has_value());

  holomq::ShmRefCountTable& table = mapping_or.value()->table();
  table.add_ref(2U);
  table.release_ref(2U);
  EXPECT_EQ(table.ref_count(2U), 0U);
  table.release_ref(2U);
  EXPECT_EQ(table.ref_count(2U), 0U);
}

TEST(ShmRefCountTest, AddRefSaturatesAtMax) {
  auto mapping_or = holomq::ShmRefCountMapping::create(kBufferCount);
  ASSERT_TRUE(mapping_or.has_value());

  holomq::SharedMemory side_channel(mapping_or.value()->shm_name(), false);
  ASSERT_TRUE(side_channel);
  ASSERT_TRUE(side_channel.map(ref_count_mapped_size_bytes(kBufferCount)));

  set_slot_ref_count(side_channel.mapped_base(), 3U, std::numeric_limits<std::uint32_t>::max());

  holomq::ShmRefCountTable& table = mapping_or.value()->table();
  table.add_ref(3U);
  EXPECT_EQ(table.ref_count(3U), std::numeric_limits<std::uint32_t>::max());
}

// Subscribers map the ref-count segment O_RDWR. A hostile peer can enlarge header.buffer_count
// after attach; bounds checks must use the process-local snapshot from map time, not a live
// re-fetch that would steer slot atomics past the mapped slot array.
TEST(ShmRefCountTest, BoundsIgnoreHostileBufferCountMutation) {
  auto mapping_or = holomq::ShmRefCountMapping::create(kBufferCount);
  ASSERT_TRUE(mapping_or.has_value());

  holomq::SharedMemory side_channel(mapping_or.value()->shm_name(), false);
  ASSERT_TRUE(side_channel);
  ASSERT_TRUE(side_channel.map(ref_count_mapped_size_bytes(kBufferCount)));

  auto* const header = static_cast<holomq::ShmRefCountTable::Header*>(side_channel.mapped_base());
  ASSERT_NE(header, nullptr);
  header->buffer_count = std::numeric_limits<std::uint32_t>::max();

  holomq::ShmRefCountTable& table = mapping_or.value()->table();
  EXPECT_EQ(table.buffer_count(), kBufferCount);
  EXPECT_FALSE(table.try_acquire_slot(kBufferCount));
  EXPECT_FALSE(table.try_acquire_slot(kBufferCount + 1U));
  table.add_ref(kBufferCount);  // must be a no-op, not an OOB atomic
  EXPECT_EQ(table.ref_count(kBufferCount), 0U);
  EXPECT_TRUE(table.try_acquire_slot(0U));
  EXPECT_EQ(table.ref_count(0U), 1U);
  table.release_ref(0U);
}

TEST(ShmRefCountTest, CreateRetriesWhenOrphanedNameExists) {
  auto first_or = holomq::ShmRefCountMapping::create(kBufferCount);
  ASSERT_TRUE(first_or.has_value());

  const std::uint64_t owner_pid = first_or.value()->owner_pid();
  const std::uint32_t blocked_id = first_or.value()->shm_ref_count_id() + 1U;
  first_or.reset();

  holomq::SharedMemory orphan(ref_count_shm_name(owner_pid, blocked_id), true);
  ASSERT_TRUE(orphan);

  auto second_or = holomq::ShmRefCountMapping::create(kBufferCount);
  ASSERT_TRUE(second_or.has_value());
  EXPECT_NE(second_or.value()->shm_ref_count_id(), blocked_id);
}

TEST(ShmRefCountTest, RefCountKeyFromHostAndDeviceHandles) {
  holomq::detail::host::BufferHandleData host_data{};
  host_data.owner_pid = 11U;
  host_data.shm_pool_id = 22U;
  host_data.shm_ref_count_id = 33U;
  host_data.slot_stride_bytes = 256U;
  host_data.buffer_bytes = 256U;
  host_data.pool_total_bytes = 256U * 4U;
  host_data.payload_pinned = 1U;
  host_data.host_register_flags = 0U;
  host_data.gen_pool_cookie = 0xABCDU;
  host_data.gen_pool_total_bytes = holomq::k_slot_gpu_generation_bytes * 4U;
  host_data.gen_slot_stride_bytes = holomq::k_slot_gpu_generation_bytes;
  host_data.fd_pass_socket_path[0] = '/';

  const holomq::BufferHandle host_handle = holomq::detail::make_host_buffer_handle(host_data, 0U);
  ASSERT_TRUE(host_handle);

  const auto host_key_or = holomq::ref_count_key(host_handle);
  ASSERT_TRUE(host_key_or.has_value());
  EXPECT_EQ(host_key_or->owner_pid, 11U);
  EXPECT_EQ(host_key_or->shm_ref_count_id, 33U);

  const holomq::BufferHandle device_handle = make_device_ref_count_handle(44U, 55U, 0U);
  ASSERT_TRUE(device_handle);

  const auto device_key_or = holomq::ref_count_key(device_handle);
  ASSERT_TRUE(device_key_or.has_value());
  EXPECT_EQ(device_key_or->owner_pid, 44U);
  EXPECT_EQ(device_key_or->shm_ref_count_id, 55U);
}

TEST(ShmRefCountTest, RefCountKeyRejectsEmptyHandleAndZeroIds) {
  EXPECT_FALSE(holomq::ref_count_key(holomq::BufferHandle{}));

  holomq::detail::device::BufferHandleData device_data{};
  device_data.pool_cookie = 0x1000U;
  device_data.owner_pid = 1U;
  device_data.shm_ref_count_id = 0U;
  device_data.slot_stride_bytes = 264U;
  device_data.buffer_bytes = 256U;
  device_data.pool_total_bytes = 264U * kBufferCount;
  device_data.fd_pass_socket_path[0] = '/';
  EXPECT_FALSE(holomq::ref_count_key(holomq::detail::make_device_buffer_handle(device_data, 0U)));
}

TEST(ShmRefCountTest, RefCountSegmentCacheMapsAndRejectsOutOfRangeSlot) {
  auto mapping_or = holomq::ShmRefCountMapping::create(kBufferCount);
  ASSERT_TRUE(mapping_or.has_value());

  const holomq::ShmRefCountMapping& created = *mapping_or.value();
  holomq::RefCountSegmentCache cache;

  const holomq::BufferHandle valid_handle =
      make_device_ref_count_handle(created.owner_pid(), created.shm_ref_count_id(), 0U);
  ASSERT_TRUE(valid_handle);

  const auto ref_count_mapping_or = cache.map(valid_handle);
  ASSERT_TRUE(ref_count_mapping_or.has_value());
  const holomq::ShmRefCountMappingPtr& ref_count_mapping = ref_count_mapping_or.value();
  EXPECT_EQ(ref_count_mapping->table().buffer_count(), kBufferCount);

  const auto cached_or = cache.map(valid_handle);
  ASSERT_TRUE(cached_or.has_value());
  EXPECT_EQ(cached_or.value(), ref_count_mapping);

  const holomq::BufferHandle out_of_range_handle = make_device_ref_count_handle(
      created.owner_pid(), created.shm_ref_count_id(), kBufferCount, kBufferCount + 4U);
  ASSERT_TRUE(out_of_range_handle);
  EXPECT_FALSE(cache.map(out_of_range_handle));
}

TEST(ShmRefCountTest, RefCountSegmentCacheConcurrentMapReturnsSameOwner) {
  auto mapping_or = holomq::ShmRefCountMapping::create(kBufferCount);
  ASSERT_TRUE(mapping_or.has_value());

  const holomq::ShmRefCountMapping& created = *mapping_or.value();
  holomq::RefCountSegmentCache cache;

  const holomq::BufferHandle valid_handle =
      make_device_ref_count_handle(created.owner_pid(), created.shm_ref_count_id(), 0U);
  ASSERT_TRUE(valid_handle);

  constexpr int kThreadCount = 8;
  std::vector<std::thread> threads;
  std::vector<holomq::ShmRefCountMappingPtr> results(static_cast<std::size_t>(kThreadCount));
  std::atomic<int> ready_count{0};

  for (int thread_index = 0; thread_index < kThreadCount; ++thread_index) {
    threads.emplace_back([&, thread_index]() {
      ready_count.fetch_add(1, std::memory_order_release);
      while (ready_count.load(std::memory_order_acquire) < kThreadCount) {
      }
      const auto mapped_or = cache.map(valid_handle);
      if (mapped_or.has_value()) {
        results[static_cast<std::size_t>(thread_index)] = mapped_or.value();
      }
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }

  ASSERT_TRUE(results[0]);
  for (int thread_index = 1; thread_index < kThreadCount; ++thread_index) {
    EXPECT_EQ(results[static_cast<std::size_t>(thread_index)], results[0]);
  }
}
