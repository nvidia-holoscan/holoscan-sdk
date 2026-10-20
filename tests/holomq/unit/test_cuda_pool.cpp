/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holomq/buffer.hpp>
#include <holomq/buffer_pool.hpp>
#include <holomq/detail/transport/backend.hpp>
#include <holomq/device_buffer_allocator.hpp>
#include "buffer_pool/device/device_buffer_pool_impl.hpp"
#include "buffer_pool/slot_region.hpp"
#include "context_impl.hpp"
#include "cuda/context.hpp"

#include <cuda.h>

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

constexpr std::uint32_t kDefaultBufferCount = 4U;
constexpr std::size_t kBufferBytes = 256U;
constexpr std::uint8_t kDirtyByte = 0xA5U;

[[nodiscard]] std::optional<std::string> cuda_error_message(const CUresult result) {
  if (result == CUDA_SUCCESS) {
    return std::nullopt;
  }
  const char* name = nullptr;
  const char* msg = nullptr;
  static_cast<void>(cuGetErrorName(result, &name));
  static_cast<void>(cuGetErrorString(result, &msg));
  return std::string(name != nullptr ? name : "?") + " (" + (msg != nullptr ? msg : "?") + ")";
}

[[nodiscard]] CUdeviceptr buffer_device_ptr(const holomq::Buffer& buffer) noexcept {
  return static_cast<CUdeviceptr>(reinterpret_cast<std::uintptr_t>(buffer.ptr()));
}

[[nodiscard]] bool write_pattern(const CUdeviceptr device_ptr, const std::size_t size_bytes,
                                 const std::uint32_t seed, std::string* error_out) {
  const std::size_t word_count = size_bytes / sizeof(std::uint32_t);
  std::vector<std::uint32_t> host(word_count);
  for (std::size_t i = 0; i < word_count; ++i) {
    host[i] = seed + static_cast<std::uint32_t>(i);
  }
  const CUresult result = cuMemcpyHtoD(device_ptr, host.data(), size_bytes);
  if (result != CUDA_SUCCESS && error_out != nullptr) {
    *error_out = "cuMemcpyHtoD failed: " + cuda_error_message(result).value_or("unknown");
  }
  return result == CUDA_SUCCESS;
}

[[nodiscard]] bool read_and_verify(const CUdeviceptr device_ptr, const std::size_t size_bytes,
                                   const std::uint32_t seed, std::string* error_out) {
  const std::size_t word_count = size_bytes / sizeof(std::uint32_t);
  std::vector<std::uint32_t> host(word_count);
  CUresult result = cuMemcpyDtoH(host.data(), device_ptr, size_bytes);
  if (result != CUDA_SUCCESS) {
    if (error_out != nullptr) {
      *error_out = "cuMemcpyDtoH failed: " + cuda_error_message(result).value_or("unknown");
    }
    return false;
  }
  for (std::size_t i = 0; i < word_count; ++i) {
    const std::uint32_t expected = seed + static_cast<std::uint32_t>(i);
    if (host[i] != expected) {
      if (error_out != nullptr) {
        *error_out = "verify mismatch at word " + std::to_string(i) + " (got " +
                     std::to_string(host[i]) + ", expected " + std::to_string(expected) + ")";
      }
      return false;
    }
  }
  return true;
}

struct SlotRecord {
  std::uint32_t seed{0};
};

void skip_unless_cuda_available() {
  const CUresult init_result = cuInit(0);
  if (init_result != CUDA_SUCCESS) {
    GTEST_SKIP() << "CUDA driver not available: "
                 << cuda_error_message(init_result).value_or("unknown");
  }

  CUdevice device{};
  const CUresult device_result = cuDeviceGet(&device, 0);
  if (device_result != CUDA_SUCCESS) {
    GTEST_SKIP() << "No CUDA device available: "
                 << cuda_error_message(device_result).value_or("unknown");
  }
}

[[nodiscard]] holomq::cuda::Context make_cuda_context() {
  CUdevice device{};
  if (cuDeviceGet(&device, 0) != CUDA_SUCCESS) {
    return {};
  }
  return holomq::cuda::Context{device};
}

[[nodiscard]] holomq::DeviceBufferAllocator dirty_device_buffer_allocator() {
  return [allocator = holomq::default_device_buffer_allocator()](
             const std::size_t number_of_bytes) -> std::unique_ptr<holomq::DeviceBufferPoolMemory> {
    auto memory = allocator(number_of_bytes);
    if (!memory) {
      return nullptr;
    }
    const CUdeviceptr base =
        static_cast<CUdeviceptr>(reinterpret_cast<std::uintptr_t>(memory->ptr()));
    if (cuMemsetD8(base, kDirtyByte, number_of_bytes) != CUDA_SUCCESS ||
        cuCtxSynchronize() != CUDA_SUCCESS) {
      return nullptr;
    }
    return memory;
  };
}

}  // namespace

TEST(CudaPoolTest, InitializesGenerationWordsWithoutClearingPayloads) {
  skip_unless_cuda_available();

  holomq::cuda::Context cuda_ctx = make_cuda_context();
  ASSERT_TRUE(static_cast<bool>(cuda_ctx)) << "cuCtxCreate failed";
  ASSERT_EQ(cuCtxSetCurrent(cuda_ctx.get()), CUDA_SUCCESS) << "cuCtxSetCurrent failed";

  auto transport_or = holomq::detail::transport::create_runtime();
  ASSERT_TRUE(transport_or.has_value()) << "create_runtime failed";
  const auto session = std::make_shared<holomq::ContextImpl>(
      std::move(transport_or).value(), "/tmp/holomq_cuda_pool_generation_test.sock", std::nullopt);
  auto pool_or = holomq::BufferPool::create<holomq::DeviceBufferPoolImpl>(
      kBufferBytes, kDefaultBufferCount, dirty_device_buffer_allocator(), std::string{}, session);
  ASSERT_TRUE(pool_or.has_value()) << "BufferPool::create failed";

  holomq::BufferPool pool = std::move(pool_or).value();
  // hold onto buffers so we check a unique slot each time
  std::vector<holomq::Buffer> held_buffers;
  held_buffers.reserve(kDefaultBufferCount);
  for (std::uint32_t index = 0U; index < kDefaultBufferCount; ++index) {
    auto buffer_or = pool.acquire();
    ASSERT_TRUE(buffer_or.has_value()) << "acquire failed at slot " << index;
    holomq::Buffer buffer = std::move(buffer_or).value();

    const CUdeviceptr payload = buffer_device_ptr(buffer);
    ASSERT_NE(payload, 0U);
    const CUdeviceptr generation = payload - holomq::k_slot_gpu_generation_bytes;

    std::uint64_t observed_generation = 1U;
    std::uint8_t observed_payload = 0U;
    ASSERT_EQ(cuMemcpyDtoH(&observed_generation, generation, sizeof(observed_generation)),
              CUDA_SUCCESS);
    ASSERT_EQ(cuMemcpyDtoH(&observed_payload, payload, sizeof(observed_payload)), CUDA_SUCCESS);
    EXPECT_EQ(observed_generation, 0U);
    EXPECT_EQ(observed_payload, kDirtyByte);

    held_buffers.push_back(std::move(buffer));
  }
}

TEST(CudaPoolTest, AcquireReleaseAndSlotReuse) {
  skip_unless_cuda_available();

  holomq::cuda::Context cuda_ctx = make_cuda_context();
  ASSERT_TRUE(static_cast<bool>(cuda_ctx)) << "cuCtxCreate failed";
  ASSERT_EQ(cuCtxSetCurrent(cuda_ctx.get()), CUDA_SUCCESS) << "cuCtxSetCurrent failed";

  auto transport_or = holomq::detail::transport::create_runtime();
  ASSERT_TRUE(transport_or.has_value()) << "create_runtime failed";
  const auto session = std::make_shared<holomq::ContextImpl>(
      std::move(transport_or).value(), "/tmp/holomq_cuda_pool_test.sock", std::nullopt);
  auto pool_or = holomq::BufferPool::create<holomq::DeviceBufferPoolImpl>(
      kBufferBytes, kDefaultBufferCount, std::string{}, session);
  ASSERT_TRUE(pool_or.has_value()) << "BufferPool::create failed";

  holomq::BufferPool pool = std::move(pool_or).value();
  const std::size_t buffer_bytes = pool.buffer_bytes();
  ASSERT_EQ(pool.buffer_count(), kDefaultBufferCount);

  CUdevice expected_device{};
  ASSERT_EQ(cuDeviceGet(&expected_device, 0), CUDA_SUCCESS);
  const auto pool_ordinal = pool.cuda_device_ordinal();
  ASSERT_TRUE(pool_ordinal.has_value());
  EXPECT_EQ(pool_ordinal.value(), static_cast<int32_t>(expected_device));
  EXPECT_EQ(pool.memory_type(), holomq::BufferMemoryType::Device);

  std::unordered_map<CUdeviceptr, SlotRecord> slot_by_ptr;
  slot_by_ptr.reserve(kDefaultBufferCount);

  std::vector<holomq::Buffer> held_buffers;
  held_buffers.reserve(kDefaultBufferCount);

  for (std::uint32_t iteration = 0U; iteration < kDefaultBufferCount; ++iteration) {
    auto buffer_or = pool.acquire();
    ASSERT_TRUE(buffer_or.has_value())
        << "acquire failed before pool exhausted (iteration " << iteration << ")";

    holomq::Buffer buffer = std::move(buffer_or).value();
    const CUdeviceptr device_ptr = buffer_device_ptr(buffer);
    ASSERT_NE(device_ptr, 0U) << "acquired buffer has null device pointer";
    EXPECT_EQ(buffer.memory_type(), holomq::BufferMemoryType::Device);
    ASSERT_TRUE(buffer.cuda_device_ordinal().has_value());
    EXPECT_EQ(buffer.cuda_device_ordinal().value(), pool_ordinal.value());

    const std::uint32_t seed = 0xA0000000U + iteration;
    std::string error;
    ASSERT_TRUE(write_pattern(device_ptr, buffer_bytes, seed, &error)) << error;

    slot_by_ptr[device_ptr] = SlotRecord{seed};
    held_buffers.push_back(std::move(buffer));
  }

  EXPECT_FALSE(pool.acquire().has_value()) << "expected acquire to fail after exhausting pool";

  held_buffers.clear();

  std::uint32_t reacquired = 0U;
  while (reacquired < kDefaultBufferCount) {
    auto buffer_or = pool.acquire();
    ASSERT_TRUE(buffer_or.has_value()) << "could not re-acquire all buffers after release";

    holomq::Buffer buffer = std::move(buffer_or).value();
    const CUdeviceptr device_ptr = buffer_device_ptr(buffer);
    const auto record_it = slot_by_ptr.find(device_ptr);
    ASSERT_NE(record_it, slot_by_ptr.end()) << "re-acquired unexpected device address";

    std::string error;
    ASSERT_TRUE(read_and_verify(device_ptr, buffer_bytes, record_it->second.seed, &error)) << error;

    held_buffers.push_back(std::move(buffer));
    ++reacquired;
  }

  EXPECT_EQ(reacquired, kDefaultBufferCount);
  EXPECT_FALSE(pool.acquire().has_value()) << "expected acquire to fail after second exhaustion";
}
