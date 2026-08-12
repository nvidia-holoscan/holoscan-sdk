/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holomq/buffer_handle.hpp>

#include "buffer_pool/slot_region.hpp"

#include <gtest/gtest.h>

namespace {

constexpr std::uint32_t kSlotIndex = 2U;
constexpr std::uint64_t kReadyGeneration = 42U;
constexpr std::size_t kBufferBytes = 256U;
constexpr std::size_t kInterleavedStride = kBufferBytes + holomq::k_slot_gpu_generation_bytes;
constexpr std::int32_t kCUDADeviceOrdinal = 2;  // nonzero: exercises multi-GPU wire

[[nodiscard]] holomq::detail::host::BufferHandleData make_valid_devicemap_host_handle() noexcept {
  holomq::detail::host::BufferHandleData handle_data{};
  handle_data.owner_pid = 1U;
  handle_data.shm_pool_id = 2U;
  handle_data.shm_ref_count_id = 3U;
  handle_data.slot_stride_bytes = kInterleavedStride;
  handle_data.buffer_bytes = kBufferBytes;
  handle_data.pool_total_bytes = kInterleavedStride * 4U;
  handle_data.payload_pinned = 1U;
  handle_data.host_register_flags = CU_MEMHOSTREGISTER_DEVICEMAP;
  return handle_data;
}

[[nodiscard]] holomq::detail::host::BufferHandleData make_valid_split_gen_host_handle() noexcept {
  holomq::detail::host::BufferHandleData handle_data{};
  handle_data.owner_pid = 1U;
  handle_data.shm_pool_id = 2U;
  handle_data.shm_ref_count_id = 3U;
  handle_data.slot_stride_bytes = kBufferBytes;
  handle_data.buffer_bytes = kBufferBytes;
  handle_data.pool_total_bytes = kBufferBytes * 4U;
  handle_data.payload_pinned = 1U;
  handle_data.host_register_flags = 0U;
  handle_data.gen_pool_cookie = 0xABCDU;
  handle_data.gen_pool_total_bytes = holomq::k_slot_gpu_generation_bytes * 4U;
  handle_data.gen_slot_stride_bytes = holomq::k_slot_gpu_generation_bytes;
  handle_data.fd_pass_socket_path[0] = '/';
  return handle_data;
}

[[nodiscard]] holomq::detail::host::BufferHandleData
make_valid_pageable_split_gen_host_handle() noexcept {
  holomq::detail::host::BufferHandleData handle_data = make_valid_split_gen_host_handle();
  handle_data.payload_pinned = 0U;
  handle_data.host_register_flags = 0U;
  return handle_data;
}

[[nodiscard]] holomq::detail::device::BufferHandleData make_valid_device_handle() noexcept {
  holomq::detail::device::BufferHandleData handle_data{};
  handle_data.pool_cookie = 0x1000U;
  handle_data.owner_pid = 1U;
  handle_data.shm_ref_count_id = 3U;
  handle_data.slot_stride_bytes = kInterleavedStride;
  handle_data.buffer_bytes = kBufferBytes;
  handle_data.pool_total_bytes = kInterleavedStride * 4U;
  handle_data.cuda_device_ordinal = kCUDADeviceOrdinal;
  handle_data.fd_pass_socket_path[0] = '/';
  return handle_data;
}

}  // namespace

TEST(BufferHandleTest, DefaultHandleIsEmpty) {
  const holomq::BufferHandle handle;
  EXPECT_FALSE(handle);
  EXPECT_EQ(handle.memory_type(), holomq::BufferMemoryType::None);
  EXPECT_EQ(holomq::detail::BufferHandleWireAccess::ready_generation(handle), 0U);
}

TEST(BufferHandleTest, DeviceHandleValidatesGeometry) {
  EXPECT_TRUE(holomq::detail::make_device_buffer_handle(make_valid_device_handle(), kSlotIndex));

  auto handle_data = make_valid_device_handle();
  handle_data.pool_cookie = 0U;
  EXPECT_FALSE(holomq::detail::make_device_buffer_handle(handle_data, kSlotIndex));

  handle_data = make_valid_device_handle();
  handle_data.fd_pass_socket_path[0] = '\0';
  EXPECT_FALSE(holomq::detail::make_device_buffer_handle(handle_data, kSlotIndex));

  handle_data = make_valid_device_handle();
  handle_data.slot_stride_bytes = kBufferBytes;
  EXPECT_FALSE(holomq::detail::make_device_buffer_handle(handle_data, kSlotIndex));

  handle_data = make_valid_device_handle();
  handle_data.cuda_device_ordinal = -1;  // unset source device ordinal is rejected
  EXPECT_FALSE(holomq::detail::make_device_buffer_handle(handle_data, kSlotIndex));

  handle_data = make_valid_device_handle();
  EXPECT_FALSE(holomq::detail::make_device_buffer_handle(handle_data, 4U));

  const holomq::BufferHandle handle = holomq::detail::make_device_buffer_handle(
      make_valid_device_handle(), kSlotIndex, kReadyGeneration);
  EXPECT_TRUE(handle);
  EXPECT_EQ(handle.memory_type(), holomq::BufferMemoryType::Device);
  EXPECT_EQ(holomq::detail::BufferHandleWireAccess::slot_index(handle), kSlotIndex);
  EXPECT_EQ(holomq::detail::BufferHandleWireAccess::ready_generation(handle), kReadyGeneration);
  // The nonzero source device ordinal survives the wire round-trip so subscribers on a multi-GPU
  // host import on the exporter's device rather than their current context's device.
  EXPECT_EQ(holomq::detail::BufferHandleWireAccess::device_wire(handle).cuda_device_ordinal,
            kCUDADeviceOrdinal);
}

TEST(BufferHandleTest, HostDevicemapHandleValidatesGeometry) {
  auto handle_data = make_valid_devicemap_host_handle();
  EXPECT_TRUE(holomq::detail::make_host_buffer_handle(handle_data, kSlotIndex));

  handle_data.owner_pid = 0U;
  EXPECT_FALSE(holomq::detail::make_host_buffer_handle(handle_data, kSlotIndex));

  handle_data = make_valid_devicemap_host_handle();
  handle_data.slot_stride_bytes = kBufferBytes;
  EXPECT_FALSE(holomq::detail::make_host_buffer_handle(handle_data, kSlotIndex));

  handle_data = make_valid_devicemap_host_handle();
  EXPECT_FALSE(holomq::detail::make_host_buffer_handle(handle_data, 4U));
}

TEST(BufferHandleTest, HostSplitGenHandleValidatesGeometry) {
  auto handle_data = make_valid_split_gen_host_handle();
  EXPECT_TRUE(holomq::detail::make_host_buffer_handle(handle_data, kSlotIndex));

  handle_data.fd_pass_socket_path[0] = '\0';
  EXPECT_FALSE(holomq::detail::make_host_buffer_handle(handle_data, kSlotIndex));

  handle_data = make_valid_split_gen_host_handle();
  handle_data.slot_stride_bytes = kBufferBytes + 1U;
  EXPECT_FALSE(holomq::detail::make_host_buffer_handle(handle_data, kSlotIndex));

  handle_data = make_valid_split_gen_host_handle();
  EXPECT_FALSE(holomq::detail::make_host_buffer_handle(handle_data, 4U));

  handle_data = make_valid_split_gen_host_handle();
  handle_data.gen_pool_total_bytes =
      handle_data.gen_slot_stride_bytes * static_cast<std::size_t>(kSlotIndex);
  EXPECT_FALSE(holomq::detail::make_host_buffer_handle(handle_data, kSlotIndex));
}

TEST(BufferHandleTest, PageableSplitGenHandleRequiresZeroRegisterFlags) {
  EXPECT_TRUE(holomq::detail::make_host_buffer_handle(make_valid_pageable_split_gen_host_handle(),
                                                      kSlotIndex));

  auto handle_data = make_valid_pageable_split_gen_host_handle();
  handle_data.host_register_flags = CU_MEMHOSTREGISTER_PORTABLE;
  EXPECT_FALSE(holomq::detail::make_host_buffer_handle(handle_data, kSlotIndex));
}

TEST(BufferHandleTest, TypedAccessorsMatchMemoryType) {
  const holomq::BufferHandle host_handle = holomq::detail::make_host_buffer_handle(
      make_valid_devicemap_host_handle(), kSlotIndex, kReadyGeneration);
  ASSERT_TRUE(host_handle);
  EXPECT_EQ(host_handle.memory_type(), holomq::BufferMemoryType::HostPinned);
  EXPECT_EQ(holomq::detail::BufferHandleWireAccess::slot_index(host_handle), kSlotIndex);
  EXPECT_EQ(holomq::detail::BufferHandleWireAccess::ready_generation(host_handle),
            kReadyGeneration);
}

TEST(BufferHandleTest, MemoryTypeMapsWireStorageClass) {
  const holomq::BufferHandle device_handle = holomq::detail::make_device_buffer_handle(
      make_valid_device_handle(), kSlotIndex, kReadyGeneration);
  ASSERT_TRUE(device_handle);
  EXPECT_EQ(device_handle.memory_type(), holomq::BufferMemoryType::Device);

  const holomq::BufferHandle pinned_handle = holomq::detail::make_host_buffer_handle(
      make_valid_devicemap_host_handle(), kSlotIndex, kReadyGeneration);
  ASSERT_TRUE(pinned_handle);
  EXPECT_EQ(pinned_handle.memory_type(), holomq::BufferMemoryType::HostPinned);

  const holomq::BufferHandle split_gen_handle = holomq::detail::make_host_buffer_handle(
      make_valid_split_gen_host_handle(), kSlotIndex, kReadyGeneration);
  ASSERT_TRUE(split_gen_handle);
  EXPECT_EQ(split_gen_handle.memory_type(), holomq::BufferMemoryType::HostPinned);

  const holomq::BufferHandle pageable_handle = holomq::detail::make_host_buffer_handle(
      make_valid_pageable_split_gen_host_handle(), kSlotIndex, kReadyGeneration);
  ASSERT_TRUE(pageable_handle);
  EXPECT_EQ(pageable_handle.memory_type(), holomq::BufferMemoryType::HostPageable);

  const holomq::BufferHandle empty_handle;
  EXPECT_EQ(empty_handle.memory_type(), holomq::BufferMemoryType::None);
}
