/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// GPU-gated Tensor lifetime / CUDA-stream-ordered reuse tests. These exercise real device
// allocation through BlockMemoryPool and CudaMemoryResourceAllocator plus the Tensor readiness /
// deallocation stream API. They live in their own target so the host-only tensor_test stays
// GPU-free; tests skip when no CUDA device is present.

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <thread>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "../common/allocation_counter.hpp"
#include "../common/cuda_test_utils.hpp"
#include "../common/memory_allocator_test_utils.hpp"

#include <holoscan/core/domain/tensor.hpp>

#include <holoscan/cuda_execution/cuda_stream_pool.hpp>
#include <holoscan/memory/block_memory_pool.hpp>
#include <holoscan/memory/cuda_memory_resource_allocator.hpp>
#include <holoscan/memory/tensor_storage.hpp>
#include "core/domain/tensor_access.hpp"

namespace holoscan {

namespace {

using ::holoscan::test::blocking_host_callback;
using ::holoscan::test::BlockingHostState;
using ::holoscan::test::create_non_blocking_cuda_stream;
using ::holoscan::test::CudaPinnedHostDeleter;
using ::holoscan::test::release_blocking_host;
using ::holoscan::test::ScopedAllocationCounter;
using ::holoscan::test::skip_without_cuda_device;
using ::holoscan::test::StreamRecordingAllocator;
using ::holoscan::test::UniqueCudaStream;
using ::holoscan::test::wait_until_started;

// DLDataType shorthands for element types used across reshape and allocation tests.
DLDataType float32_dtype() {
  return DLDataType{kDLFloat, 32, 1};
}

DLDataType uint8_dtype() {
  return DLDataType{kDLUInt, 8, 1};
}

// Shared body for DLPackExportSynchronizesConsumerStreamWithProducerStream and
// VersionedDLPackExportSynchronizesConsumerStreamWithProducerStream. Allocates a
// device Tensor on producer_stream, blocks it behind a CUDA host callback, then
// exports via DLPack and confirms the consumer_stream event-waits on the producer.
//
//   producer_stream: [memset 0xA5] --[BlockingHostCallback]-- (blocked) --> done
//   consumer_stream:                [event-wait on producer] ----------> [D2H copy]
void run_dlpack_cross_stream_readiness_test(bool versioned_export) {
  skip_without_cuda_device();

  UniqueCudaStream producer_stream;
  ASSERT_EQ(create_non_blocking_cuda_stream(producer_stream), cudaSuccess);

  UniqueCudaStream consumer_stream;
  ASSERT_EQ(create_non_blocking_cuda_stream(consumer_stream), cudaSuccess);

  auto allocator = std::make_shared<CudaMemoryResourceAllocator>(
      CudaMemoryResourceAllocator::Config{"1MB", "2MB", "1MB", "2MB", 0});
  ASSERT_TRUE(allocator->initialize_pool());

  constexpr std::size_t kSize = 64;
  constexpr uint8_t kExpected = 0xA5;
  const std::array<int64_t, 1> shape{static_cast<int64_t>(kSize)};

  void* raw_host = nullptr;
  ASSERT_EQ(cudaMallocHost(&raw_host, kSize), cudaSuccess);
  const std::unique_ptr<uint8_t, CudaPinnedHostDeleter> host(static_cast<uint8_t*>(raw_host));

  Tensor tensor;
  const auto result = reshape_custom(tensor,
                                     shape,
                                     uint8_dtype(),
                                     std::nullopt,
                                     MemoryStorageType::kDevice,
                                     allocator,
                                     producer_stream.get());
  ASSERT_TRUE(result) << runtime_error_message(result.error());
  ASSERT_NE(tensor.data(), nullptr);

  auto writable_data = tensor.mutable_data();
  ASSERT_TRUE(writable_data) << runtime_error_message(writable_data.error());
  ASSERT_EQ(cudaMemsetAsync(writable_data.value(), kExpected, kSize, producer_stream.get()),
            cudaSuccess);

  // Key setup: a blocking host callback stalls producer_stream so the consumer's event-wait
  // can be observed with cudaStreamQuery before the block is released.
  BlockingHostState state;
  ASSERT_EQ(cudaLaunchHostFunc(producer_stream.get(), blocking_host_callback, &state), cudaSuccess);
  if (!wait_until_started(state)) {
    release_blocking_host(state);
    EXPECT_EQ(cudaStreamSynchronize(producer_stream.get()), cudaSuccess);
    FAIL() << "Timed out waiting for CUDA host function to start";
  }

  EXPECT_TRUE(tensor.set_readiness_stream(producer_stream.get()));
  auto sync_result = tensor.synchronize_for_consumer_stream(consumer_stream.get());
  EXPECT_TRUE(sync_result) << runtime_error_message(sync_result.error());

  DLManagedTensor* exported = nullptr;
  DLManagedTensorVersioned* versioned = nullptr;
  DLTensor* dl_tensor = nullptr;
  if (versioned_export) {
    auto exported_result = tensor.to_dlpack_versioned();
    ASSERT_TRUE(exported_result) << runtime_error_message(exported_result.error());
    versioned = exported_result.value();
    ASSERT_NE(versioned, nullptr);
    ASSERT_NE(versioned->deleter, nullptr);
    dl_tensor = &versioned->dl_tensor;
  } else {
    auto exported_result = tensor.to_dlpack();
    ASSERT_TRUE(exported_result) << runtime_error_message(exported_result.error());
    exported = exported_result.value();
    ASSERT_NE(exported, nullptr);
    ASSERT_NE(exported->deleter, nullptr);
    dl_tensor = &exported->dl_tensor;
  }
  ASSERT_NE(dl_tensor, nullptr);

  const cudaError_t copy_status = cudaMemcpyAsync(
      host.get(), dl_tensor->data, kSize, cudaMemcpyDeviceToHost, consumer_stream.get());
  EXPECT_EQ(copy_status, cudaSuccess);

  // Key assertion: consumer_stream is not yet ready — it is waiting on the event recorded
  // after the blocked producer callback, proving readiness ordering is enforced.
  // The consumer stream should be waiting on the event recorded after the producer's blocked host
  // function. Without the DLPack readiness wait, this copy could complete before release below.
  if (copy_status == cudaSuccess) {
    EXPECT_EQ(cudaStreamQuery(consumer_stream.get()), cudaErrorNotReady);
  }

  release_blocking_host(state);
  ASSERT_EQ(cudaStreamSynchronize(consumer_stream.get()), cudaSuccess);
  for (std::size_t i = 0; i < kSize; ++i) {
    EXPECT_EQ(host.get()[i], kExpected);
  }

  EXPECT_TRUE(tensor.set_deallocation_stream(consumer_stream.get()));
  if (versioned_export) {
    versioned->deleter(versioned);
  } else {
    exported->deleter(exported);
  }
  tensor = Tensor();

  ASSERT_EQ(cudaStreamSynchronize(consumer_stream.get()), cudaSuccess);
  EXPECT_TRUE(allocator->deinitialize_pool());
}

}  // namespace

// Verifies reshape_custom() attaches a native TensorMemoryReference so
// Tensor::set_deallocation_stream() can update the stream passed to allocator free.
TEST(TensorDevice, ReshapeCustomUsesNativeMemoryReferenceForStreamAwareRelease) {
  auto allocator = std::make_shared<StreamRecordingAllocator>();
  Tensor tensor;
  auto allocation_stream = reinterpret_cast<cudaStream_t>(uintptr_t{0x1234});
  const std::array<int64_t, 2> shape{2, 3};

  const auto result = reshape_custom(tensor,
                                     shape,
                                     float32_dtype(),
                                     std::nullopt,
                                     MemoryStorageType::kSystem,
                                     allocator,
                                     allocation_stream);
  ASSERT_TRUE(result) << runtime_error_message(result.error());
  ASSERT_NE(detail::TensorAccess::context(tensor), nullptr);
  ASSERT_NE(detail::TensorAccess::context(tensor)->memory_reference, nullptr);
  EXPECT_EQ(allocator->allocated_stream, reinterpret_cast<void*>(allocation_stream));

  auto stream = reinterpret_cast<cudaStream_t>(uintptr_t{0x4321});
  EXPECT_TRUE(tensor.set_deallocation_stream(stream));

  tensor = Tensor{};
  EXPECT_EQ(allocator->allocate_count, 1);
  EXPECT_EQ(allocator->free_count, 1);
  EXPECT_EQ(allocator->last_storage_type, MemoryStorageType::kSystem);
  EXPECT_EQ(allocator->released_stream, reinterpret_cast<void*>(stream));
}

// Verifies reshape_custom() can allocate real CUDA device storage through the CCCL-backed memory
// resource allocator and release it on the recorded CUDA stream.
TEST(TensorDevice, ReshapeCustomUsesCudaMemoryResourceAllocatorForDeviceStorage) {
  skip_without_cuda_device();

  UniqueCudaStream stream;
  ASSERT_EQ(create_non_blocking_cuda_stream(stream), cudaSuccess);

  auto allocator = std::make_shared<CudaMemoryResourceAllocator>(
      CudaMemoryResourceAllocator::Config{"1MB", "2MB", "1MB", "2MB", 0});
  ASSERT_TRUE(allocator->initialize_pool());

  Tensor tensor;
  const std::array<int64_t, 2> shape{2, 3};
  const auto result = reshape_custom(tensor,
                                     shape,
                                     float32_dtype(),
                                     std::nullopt,
                                     MemoryStorageType::kDevice,
                                     allocator,
                                     stream.get());
  ASSERT_TRUE(result) << runtime_error_message(result.error());

  ASSERT_NE(tensor.data(), nullptr);
  EXPECT_EQ(tensor.device().device_type, kDLCUDA);
  EXPECT_EQ(tensor.shape(), (std::vector<int64_t>{2, 3}));
  EXPECT_EQ(tensor.nbytes(), 24);

  auto writable_data = tensor.mutable_data();
  ASSERT_TRUE(writable_data) << runtime_error_message(writable_data.error());
  ASSERT_EQ(cudaMemsetAsync(writable_data.value(), 0x2A, tensor.nbytes(), stream.get()),
            cudaSuccess);
  EXPECT_TRUE(tensor.set_deallocation_stream(stream.get()));

  tensor = Tensor{};
  ASSERT_EQ(cudaStreamSynchronize(stream.get()), cudaSuccess);
  EXPECT_TRUE(allocator->deinitialize_pool());
}

// Verifies reshape_custom() can allocate CUDA-visible pinned host Tensor storage through the
// CCCL-backed memory resource allocator and release it through Tensor ownership.
TEST(TensorDevice, ReshapeCustomUsesCudaMemoryResourceAllocatorForPinnedHostStorage) {
  skip_without_cuda_device();

  UniqueCudaStream stream;
  ASSERT_EQ(create_non_blocking_cuda_stream(stream), cudaSuccess);

  auto allocator = std::make_shared<CudaMemoryResourceAllocator>(
      CudaMemoryResourceAllocator::Config{"1MB", "2MB", "1MB", "2MB", 0});
  ASSERT_TRUE(allocator->initialize_pool());

  Tensor tensor;
  const std::array<int64_t, 2> shape{2, 3};
  const auto result = reshape_custom(tensor,
                                     shape,
                                     float32_dtype(),
                                     std::nullopt,
                                     MemoryStorageType::kHost,
                                     allocator,
                                     stream.get());
  ASSERT_TRUE(result) << runtime_error_message(result.error());

  ASSERT_NE(tensor.data(), nullptr);
  EXPECT_EQ(tensor.device().device_type, kDLCUDAHost);
  EXPECT_EQ(tensor.shape(), (std::vector<int64_t>{2, 3}));
  EXPECT_EQ(tensor.nbytes(), 24);

  cudaPointerAttributes attributes{};
  ASSERT_EQ(cudaPointerGetAttributes(&attributes, tensor.data()), cudaSuccess);
  EXPECT_EQ(attributes.type, cudaMemoryTypeHost);

  auto writable_data = tensor.mutable_data_as<float>();
  ASSERT_TRUE(writable_data) << runtime_error_message(writable_data.error());
  auto* data = writable_data.value();
  data[0] = 1.0F;
  data[5] = 6.0F;

  auto maybe_used = allocator->pool_size(MemoryStorageType::kHost);
  ASSERT_TRUE(maybe_used) << runtime_error_message(maybe_used.error());
  EXPECT_EQ(maybe_used.value(), static_cast<std::size_t>(tensor.nbytes()));

  EXPECT_TRUE(tensor.set_deallocation_stream(stream.get()));
  tensor = Tensor{};

  maybe_used = allocator->pool_size(MemoryStorageType::kHost);
  ASSERT_TRUE(maybe_used) << runtime_error_message(maybe_used.error());
  EXPECT_EQ(maybe_used.value(), 0U);

  EXPECT_TRUE(allocator->deinitialize_pool());
}

// Verifies unversioned DLPack export can be paired with Tensor's readiness hook so a consumer
// stream waits for pending producer-stream CUDA work before reading the exported data.
TEST(TensorDevice, DLPackExportSynchronizesConsumerStreamWithProducerStream) {
  run_dlpack_cross_stream_readiness_test(/*versioned_export=*/false);
}

// Verifies the same producer-to-consumer stream readiness contract for versioned DLPack export.
TEST(TensorDevice, VersionedDLPackExportSynchronizesConsumerStreamWithProducerStream) {
  run_dlpack_cross_stream_readiness_test(/*versioned_export=*/true);
}

// Verifies a CPU-side consumer can explicitly materialize device Tensor data into caller-owned
// host memory; Tensor handles readiness ordering but does not allocate the destination.
TEST(TensorDevice, CopyToHostWaitsForProducerStream) {
  skip_without_cuda_device();

  UniqueCudaStream stream;
  ASSERT_EQ(create_non_blocking_cuda_stream(stream), cudaSuccess);

  auto allocator = std::make_shared<CudaMemoryResourceAllocator>(
      CudaMemoryResourceAllocator::Config{"1MB", "2MB", "1MB", "2MB", 0});
  ASSERT_TRUE(allocator->initialize_pool());

  Tensor tensor;
  const std::array<int64_t, 1> shape{64};
  const auto result = reshape_custom(tensor,
                                     shape,
                                     uint8_dtype(),
                                     std::nullopt,
                                     MemoryStorageType::kDevice,
                                     allocator,
                                     stream.get());
  ASSERT_TRUE(result) << runtime_error_message(result.error());

  const auto byte_count = static_cast<std::size_t>(tensor.nbytes());
  void* raw_host = nullptr;
  ASSERT_EQ(cudaMallocHost(&raw_host, byte_count), cudaSuccess);
  const std::unique_ptr<uint8_t, CudaPinnedHostDeleter> host(static_cast<uint8_t*>(raw_host));
  std::fill(host.get(), host.get() + byte_count, uint8_t{0});

  auto writable_data = tensor.mutable_data();
  ASSERT_TRUE(writable_data) << runtime_error_message(writable_data.error());
  ASSERT_EQ(cudaMemsetAsync(writable_data.value(), 0x5A, byte_count, stream.get()), cudaSuccess);
  EXPECT_TRUE(tensor.set_readiness_stream(stream.get()));
  auto copy_result = tensor.copy_to_host(host.get(), byte_count, stream.get());
  ASSERT_TRUE(copy_result) << runtime_error_message(copy_result.error());
  for (std::size_t i = 0; i < byte_count; ++i) {
    EXPECT_EQ(static_cast<int>(host.get()[i]), 0x5A);
  }

  tensor = Tensor();
  EXPECT_TRUE(allocator->deinitialize_pool());
}

// Verifies Tensor's CUDA-facing methods accept borrowed CCCL stream references from the native
// stream pool instead of forcing operator/runtime code to unwrap raw cudaStream_t handles.
TEST(TensorDevice, CudaStreamRefOverloadsUsePoolLeasedStream) {
  skip_without_cuda_device();

  CudaStreamPool stream_pool(/*device_id=*/0,
                             cudaStreamNonBlocking,
                             ::cuda::stream::default_priority,
                             /*reserved_size=*/1,
                             /*max_size=*/1);
  ASSERT_NO_THROW(stream_pool.initialize());
  auto maybe_lease = stream_pool.acquire();
  ASSERT_TRUE(maybe_lease) << runtime_error_message(maybe_lease.error());
  auto lease = std::move(maybe_lease.value());
  auto stream = lease.stream();

  auto allocator = std::make_shared<CudaMemoryResourceAllocator>(
      CudaMemoryResourceAllocator::Config{"1MB", "2MB", "1MB", "2MB", 0});
  ASSERT_TRUE(allocator->initialize_pool());

  Tensor tensor;
  const std::array<int64_t, 1> shape{64};
  const auto result = reshape_custom(tensor,
                                     shape,
                                     uint8_dtype(),
                                     std::nullopt,
                                     MemoryStorageType::kDevice,
                                     allocator,
                                     stream.get());
  ASSERT_TRUE(result) << runtime_error_message(result.error());

  const auto byte_count = static_cast<std::size_t>(tensor.nbytes());
  void* raw_host = nullptr;
  ASSERT_EQ(cudaMallocHost(&raw_host, byte_count), cudaSuccess);
  const std::unique_ptr<uint8_t, CudaPinnedHostDeleter> host(static_cast<uint8_t*>(raw_host));
  std::fill(host.get(), host.get() + byte_count, uint8_t{0});

  auto writable_data = tensor.mutable_data();
  ASSERT_TRUE(writable_data) << runtime_error_message(writable_data.error());
  ASSERT_EQ(cudaMemsetAsync(writable_data.value(), 0xA5, byte_count, stream.get()), cudaSuccess);
  EXPECT_TRUE(tensor.set_readiness_stream(stream));
  auto sync_result = tensor.synchronize_for_consumer_stream(stream);
  ASSERT_TRUE(sync_result) << runtime_error_message(sync_result.error());
  auto copy_result = tensor.copy_to_host(host.get(), byte_count, stream);
  ASSERT_TRUE(copy_result) << runtime_error_message(copy_result.error());
  for (std::size_t i = 0; i < byte_count; ++i) {
    EXPECT_EQ(static_cast<int>(host.get()[i]), 0xA5);
  }

  EXPECT_TRUE(tensor.set_deallocation_stream(stream));
  tensor = Tensor();
  ASSERT_EQ(cudaStreamSynchronize(stream.get()), cudaSuccess);
  EXPECT_TRUE(allocator->deinitialize_pool());
  EXPECT_TRUE(lease.release());
}

// Verifies the value-returning Tensor construction helper accepts CCCL stream references, matching
// the stream vocabulary used by the runtime stream pool and Tensor readiness/release APIs.
TEST(TensorDevice, MakeTensorAcceptsCudaStreamRef) {
  skip_without_cuda_device();

  CudaStreamPool stream_pool(/*device_id=*/0,
                             cudaStreamNonBlocking,
                             ::cuda::stream::default_priority,
                             /*reserved_size=*/1,
                             /*max_size=*/1);
  ASSERT_NO_THROW(stream_pool.initialize());
  auto maybe_lease = stream_pool.acquire();
  ASSERT_TRUE(maybe_lease) << runtime_error_message(maybe_lease.error());
  auto lease = std::move(maybe_lease.value());
  auto stream = lease.stream();

  auto allocator = std::make_shared<CudaMemoryResourceAllocator>(
      CudaMemoryResourceAllocator::Config{"1MB", "2MB", "1MB", "2MB", 0});
  ASSERT_TRUE(allocator->initialize_pool());

  const std::array<int64_t, 1> shape{64};
  auto maybe_tensor = make_tensor(
      shape, uint8_dtype(), std::nullopt, MemoryStorageType::kDevice, allocator, stream);
  ASSERT_TRUE(maybe_tensor) << runtime_error_message(maybe_tensor.error());
  auto tensor = std::move(maybe_tensor.value());

  ASSERT_NE(tensor.data(), nullptr);
  EXPECT_EQ(tensor.device().device_type, kDLCUDA);
  EXPECT_EQ(tensor.nbytes(), 64);

  EXPECT_TRUE(tensor.set_deallocation_stream(stream));
  tensor = Tensor();
  ASSERT_EQ(cudaStreamSynchronize(stream.get()), cudaSuccess);
  EXPECT_TRUE(allocator->deinitialize_pool());
  EXPECT_TRUE(lease.release());
}

// Verifies Tensor-owned BlockMemoryPool storage is not returned to the pool until the CUDA event
// recorded on the Tensor's deallocation stream has completed.
TEST(TensorDevice, ReshapeCustomBlockMemoryPoolDefersDeviceBlockReuseUntilStreamEvent) {
  skip_without_cuda_device();

  UniqueCudaStream stream;
  ASSERT_EQ(create_non_blocking_cuda_stream(stream), cudaSuccess);

  auto pool = std::make_shared<BlockMemoryPool>(MemoryStorageType::kDevice, 64, 1);
  ASSERT_TRUE(pool->initialize_pool());

  void* expected_data = nullptr;
  {
    Tensor tensor;
    const std::array<int64_t, 2> shape{2, 3};
    const auto result = reshape_custom(tensor,
                                       shape,
                                       float32_dtype(),
                                       std::nullopt,
                                       MemoryStorageType::kDevice,
                                       pool,
                                       stream.get());
    ASSERT_TRUE(result) << runtime_error_message(result.error());

    auto writable_data = tensor.mutable_data();
    ASSERT_TRUE(writable_data) << runtime_error_message(writable_data.error());
    expected_data = writable_data.value();
    ASSERT_EQ(cudaMemsetAsync(expected_data, 0x5A, tensor.nbytes(), stream.get()), cudaSuccess);

    // Key setup: block the deallocation stream so the Tensor can be destroyed while CUDA
    // work is still pending, proving the pool defers reuse until the stream event fires.
    BlockingHostState state;
    ASSERT_EQ(cudaLaunchHostFunc(stream.get(), blocking_host_callback, &state), cudaSuccess);
    if (!wait_until_started(state)) {
      release_blocking_host(state);
      EXPECT_EQ(cudaStreamSynchronize(stream.get()), cudaSuccess);
      tensor = Tensor{};
      EXPECT_TRUE(pool->deinitialize_pool());
      FAIL() << "Timed out waiting for CUDA host function to start";
    }

    EXPECT_TRUE(tensor.set_deallocation_stream(stream.get()));
    tensor = Tensor{};

    // Key assertion: the C++ Tensor handle was dropped but the stream event has not fired;
    // the one-block pool must remain exhausted until the host callback is released.
    // Tensor memory was released above, but stream work is still pending, so the block is not yet
    // available for reuse.
    auto available = pool->allocate(1, MemoryStorageType::kDevice);
    EXPECT_FALSE(available);

    release_blocking_host(state);
    ASSERT_EQ(cudaStreamSynchronize(stream.get()), cudaSuccess);
  }

  auto second = pool->allocate(1, MemoryStorageType::kDevice);
  ASSERT_TRUE(second) << runtime_error_message(second.error());
  EXPECT_EQ(second.value(), expected_data);

  EXPECT_TRUE(pool->free(second.value()));
  EXPECT_TRUE(pool->deinitialize_pool());
}

// Verifies a pool block remains unavailable while the Tensor still owns it, even if all queued
// CUDA work has completed, and becomes reusable only after the Tensor releases its context.
TEST(TensorDevice, ReshapeCustomBlockMemoryPoolKeepsDeviceBlockUnavailableWhileTensorOwnsIt) {
  skip_without_cuda_device();

  UniqueCudaStream stream;
  ASSERT_EQ(create_non_blocking_cuda_stream(stream), cudaSuccess);

  auto pool = std::make_shared<BlockMemoryPool>(MemoryStorageType::kDevice, 64, 1);
  ASSERT_TRUE(pool->initialize_pool());

  Tensor tensor;
  const std::array<int64_t, 2> shape{2, 3};
  const auto result = reshape_custom(
      tensor, shape, float32_dtype(), std::nullopt, MemoryStorageType::kDevice, pool, stream.get());
  ASSERT_TRUE(result) << runtime_error_message(result.error());

  auto writable_data = tensor.mutable_data();
  ASSERT_TRUE(writable_data) << runtime_error_message(writable_data.error());
  void* expected_data = writable_data.value();
  ASSERT_EQ(cudaMemsetAsync(expected_data, 0x5A, tensor.nbytes(), stream.get()), cudaSuccess);
  ASSERT_EQ(cudaStreamSynchronize(stream.get()), cudaSuccess);

  // Stream work has completed, but the Tensor still owns the block, so the pool must not reuse it.
  auto available_while_owned = pool->allocate(1, MemoryStorageType::kDevice);
  EXPECT_FALSE(available_while_owned);

  EXPECT_TRUE(tensor.set_deallocation_stream(stream.get()));
  tensor = Tensor{};
  ASSERT_EQ(cudaStreamSynchronize(stream.get()), cudaSuccess);

  auto available_after_release = pool->allocate(1, MemoryStorageType::kDevice);
  ASSERT_TRUE(available_after_release) << runtime_error_message(available_after_release.error());
  EXPECT_EQ(available_after_release.value(), expected_data);

  EXPECT_TRUE(pool->free(available_after_release.value()));
  EXPECT_TRUE(pool->deinitialize_pool());
}

// Verifies copied Tensor handles behave like same-process fanout access guards for ownership:
// the device block is not returned to a one-block pool until every consumer copy releases it.
TEST(TensorDevice, BlockMemoryPoolDeviceTensorCopiesRetainBlockUntilAllConsumersRelease) {
  skip_without_cuda_device();

  auto pool = std::make_shared<BlockMemoryPool>(MemoryStorageType::kDevice, 64, 1);
  ASSERT_TRUE(pool->initialize_pool());

  Tensor producer;
  const std::array<int64_t, 2> shape{2, 3};
  const auto result = reshape_custom(
      producer, shape, float32_dtype(), std::nullopt, MemoryStorageType::kDevice, pool);
  ASSERT_TRUE(result) << runtime_error_message(result.error());
  auto writable_data = producer.mutable_data();
  ASSERT_TRUE(writable_data) << runtime_error_message(writable_data.error());
  void* expected_data = writable_data.value();

  Tensor consumer_a = producer;
  Tensor consumer_b = producer;
  producer = Tensor();

  auto available_while_two_consumers_hold = pool->allocate(1, MemoryStorageType::kDevice);
  EXPECT_FALSE(available_while_two_consumers_hold);

  consumer_a = Tensor();
  auto available_while_one_consumer_holds = pool->allocate(1, MemoryStorageType::kDevice);
  EXPECT_FALSE(available_while_one_consumer_holds);

  consumer_b = Tensor();
  auto available_after_all_consumers_release = pool->allocate(1, MemoryStorageType::kDevice);
  ASSERT_TRUE(available_after_all_consumers_release)
      << runtime_error_message(available_after_all_consumers_release.error());
  EXPECT_EQ(available_after_all_consumers_release.value(), expected_data);

  EXPECT_TRUE(pool->free(available_after_all_consumers_release.value()));
  EXPECT_TRUE(pool->deinitialize_pool());
}

// Verifies copied Tensor consumers can each record a CUDA release stream. The fast consumer records
// last, but the pool must still wait for the slower consumer's blocked stream before reusing the
// device block.
TEST(TensorDevice, BlockMemoryPoolDeviceTensorFanoutWaitsForAllConsumerReleaseStreams) {
  skip_without_cuda_device();

  UniqueCudaStream slow_stream;
  ASSERT_EQ(create_non_blocking_cuda_stream(slow_stream), cudaSuccess);

  UniqueCudaStream fast_stream;
  ASSERT_EQ(create_non_blocking_cuda_stream(fast_stream), cudaSuccess);

  auto pool = std::make_shared<BlockMemoryPool>(MemoryStorageType::kDevice, 64, 1);
  ASSERT_TRUE(pool->initialize_pool());

  Tensor producer;
  const std::array<int64_t, 2> shape{2, 3};
  const auto result = reshape_custom(
      producer, shape, float32_dtype(), std::nullopt, MemoryStorageType::kDevice, pool);
  ASSERT_TRUE(result) << runtime_error_message(result.error());
  auto writable_data = producer.mutable_data();
  ASSERT_TRUE(writable_data) << runtime_error_message(writable_data.error());
  void* expected_data = writable_data.value();

  Tensor slow_consumer = producer;
  Tensor fast_consumer = producer;
  producer = Tensor();

  BlockingHostState slow_state;
  ASSERT_EQ(cudaLaunchHostFunc(slow_stream.get(), blocking_host_callback, &slow_state),
            cudaSuccess);
  if (!wait_until_started(slow_state)) {
    release_blocking_host(slow_state);
    EXPECT_EQ(cudaStreamSynchronize(slow_stream.get()), cudaSuccess);
    FAIL() << "Timed out waiting for CUDA host function to start";
  }

  ASSERT_EQ(cudaMemsetAsync(expected_data, 0x2A, 24, fast_stream.get()), cudaSuccess);
  ASSERT_EQ(cudaStreamSynchronize(fast_stream.get()), cudaSuccess);

  // Key setup: both consumers record independent release streams; fast_consumer's stream has
  // already completed, but slow_consumer's stream is still blocked by the host callback.
  EXPECT_TRUE(slow_consumer.set_deallocation_stream(slow_stream.get()));
  EXPECT_TRUE(fast_consumer.set_deallocation_stream(fast_stream.get()));
  fast_consumer = Tensor();
  slow_consumer = Tensor();

  // Key assertion: both Tensor handles dropped but slow_consumer's stream is still blocked;
  // the pool must wait for all fanout release streams, not just the last one recorded.
  // Both Tensor handles have released, but the slow consumer's stream is still blocked. Reusing
  // the one-block pool here would mean fanout release used only the most recent stream.
  auto available_while_slow_stream_blocked = pool->allocate(1, MemoryStorageType::kDevice);
  EXPECT_FALSE(available_while_slow_stream_blocked);

  release_blocking_host(slow_state);
  ASSERT_EQ(cudaStreamSynchronize(slow_stream.get()), cudaSuccess);

  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (std::chrono::steady_clock::now() < deadline) {
    if (pool->stats().available == 1) {
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  auto available_after_all_release_streams = pool->allocate(1, MemoryStorageType::kDevice);
  ASSERT_TRUE(available_after_all_release_streams)
      << runtime_error_message(available_after_all_release_streams.error());
  EXPECT_EQ(available_after_all_release_streams.value(), expected_data);

  EXPECT_TRUE(pool->free(available_after_all_release_streams.value()));
  EXPECT_TRUE(pool->deinitialize_pool());
}

// Verifies Tensor fanout release is allocation-stable after setup: copied Tensor handles record
// independent release streams, final release joins those streams using preallocated MemoryBuffer
// CUDA sync resources, and the BlockMemoryPool records deferred reuse without growing host-side
// containers.
TEST(TensorDevice, BlockMemoryPoolDeviceTensorFanoutReleaseHotPathDoesNotAllocate) {
  skip_without_cuda_device();

  UniqueCudaStream stream_a;
  ASSERT_EQ(create_non_blocking_cuda_stream(stream_a), cudaSuccess);

  UniqueCudaStream stream_b;
  ASSERT_EQ(create_non_blocking_cuda_stream(stream_b), cudaSuccess);

  auto pool = std::make_shared<BlockMemoryPool>(MemoryStorageType::kDevice, 64, 1);
  ASSERT_TRUE(pool->initialize_pool());

  Tensor producer;
  const std::array<int64_t, 2> shape{2, 3};
  const auto result = reshape_custom(
      producer, shape, float32_dtype(), std::nullopt, MemoryStorageType::kDevice, pool);
  ASSERT_TRUE(result) << runtime_error_message(result.error());
  auto writable_data = producer.mutable_data();
  ASSERT_TRUE(writable_data) << runtime_error_message(writable_data.error());
  void* original_data_pointer = writable_data.value();

  Tensor consumer_a = producer;
  Tensor consumer_b = producer;
  // Clear producer handle.
  producer = Tensor();

  // Launch work. Scheduling order does not matter because output is not verified.
  ASSERT_EQ(cudaMemsetAsync(original_data_pointer, 0x1A, 24, stream_a.get()), cudaSuccess);
  ASSERT_EQ(cudaMemsetAsync(original_data_pointer, 0x2B, 24, stream_b.get()), cudaSuccess);
  ASSERT_EQ(cudaStreamSynchronize(stream_a.get()), cudaSuccess);
  ASSERT_EQ(cudaStreamSynchronize(stream_b.get()), cudaSuccess);

  EXPECT_TRUE(consumer_a.set_deallocation_stream(stream_a.get()));
  EXPECT_TRUE(consumer_b.set_deallocation_stream(stream_b.get()));

  // Key setup: both streams have completed, so the release hot path exercises only the
  // preallocated CUDA sync resources inside the MemoryBuffer, with no dynamic allocation.
  // The sole block remains checked out while fanout consumers still hold Tensor handles.
  auto available_before_release = pool->allocate(1, MemoryStorageType::kDevice);
  EXPECT_FALSE(available_before_release);

  std::size_t allocation_count = 0;
  {
    const ScopedAllocationCounter allocations;
    // Clear consumer handles to trigger buffer release.
    consumer_a = Tensor();
    consumer_b = Tensor();
    allocation_count = allocations.count();
  }
  // Key assertion: releasing two consumer Tensor handles with independent deallocation
  // streams must not trigger any heap allocation in the fanout release hot path.
  EXPECT_EQ(allocation_count, 0U);

  auto available_after_release = pool->allocate(1, MemoryStorageType::kDevice);
  ASSERT_TRUE(available_after_release) << runtime_error_message(available_after_release.error());
  EXPECT_EQ(available_after_release.value(), original_data_pointer);

  EXPECT_TRUE(pool->free(available_after_release.value()));
  EXPECT_TRUE(pool->deinitialize_pool());
}

// Verifies a DLPack export keeps BlockMemoryPool-backed Tensor storage checked out after the C++
// Tensor handle drops, and returns the block only when the DLPack deleter releases the context.
TEST(TensorDevice, DLPackExportRetainsBlockMemoryPoolDeviceBlockUntilDeleter) {
  skip_without_cuda_device();

  UniqueCudaStream stream;
  ASSERT_EQ(create_non_blocking_cuda_stream(stream), cudaSuccess);

  auto pool = std::make_shared<BlockMemoryPool>(MemoryStorageType::kDevice, 64, 1);
  ASSERT_TRUE(pool->initialize_pool());

  DLManagedTensor* exported = nullptr;
  void* expected_data = nullptr;
  {
    Tensor tensor;
    const std::array<int64_t, 2> shape{2, 3};
    const auto result = reshape_custom(tensor,
                                       shape,
                                       float32_dtype(),
                                       std::nullopt,
                                       MemoryStorageType::kDevice,
                                       pool,
                                       stream.get());
    ASSERT_TRUE(result) << runtime_error_message(result.error());

    auto writable_data = tensor.mutable_data();
    ASSERT_TRUE(writable_data) << runtime_error_message(writable_data.error());
    expected_data = writable_data.value();
    ASSERT_EQ(cudaMemsetAsync(expected_data, 0x3C, tensor.nbytes(), stream.get()), cudaSuccess);
    EXPECT_TRUE(tensor.set_deallocation_stream(stream.get()));

    auto exported_result = tensor.to_dlpack();
    ASSERT_TRUE(exported_result) << runtime_error_message(exported_result.error());
    exported = exported_result.value();
    ASSERT_NE(exported, nullptr);
    ASSERT_NE(exported->deleter, nullptr);
    EXPECT_EQ(exported->dl_tensor.data, expected_data);
  }

  // The C++ Tensor has dropped, but the exported DLPack handle still owns the context, so the
  // one-block pool must remain exhausted.
  auto available_while_exported = pool->allocate(1, MemoryStorageType::kDevice);
  EXPECT_FALSE(available_while_exported);

  exported->deleter(exported);
  ASSERT_EQ(cudaStreamSynchronize(stream.get()), cudaSuccess);

  auto available_after_deleter = pool->allocate(1, MemoryStorageType::kDevice);
  ASSERT_TRUE(available_after_deleter) << runtime_error_message(available_after_deleter.error());
  EXPECT_EQ(available_after_deleter.value(), expected_data);

  EXPECT_TRUE(pool->free(available_after_deleter.value()));
  EXPECT_TRUE(pool->deinitialize_pool());
}

// Verifies DLPack export keeps allocator-backed Tensor storage and metadata alive after the
// originating Tensor goes out of scope, then releases the storage through the DLPack deleter.
TEST(TensorDevice, DLPackExportRetainsAllocatorBackedStorageUntilDeleter) {
  auto allocator = std::make_shared<StreamRecordingAllocator>();
  DLManagedTensor* exported = nullptr;
  void* expected_data = nullptr;
  auto allocation_stream = reinterpret_cast<cudaStream_t>(uintptr_t{0x1234});
  auto stream = reinterpret_cast<cudaStream_t>(uintptr_t{0x5678});

  {
    Tensor tensor;
    const std::array<int64_t, 2> shape{2, 3};
    const auto result = reshape_custom(tensor,
                                       shape,
                                       float32_dtype(),
                                       std::nullopt,
                                       MemoryStorageType::kSystem,
                                       allocator,
                                       allocation_stream);
    ASSERT_TRUE(result) << runtime_error_message(result.error());
    EXPECT_EQ(allocator->allocated_stream, reinterpret_cast<void*>(allocation_stream));
    auto writable_data = tensor.mutable_data();
    ASSERT_TRUE(writable_data) << runtime_error_message(writable_data.error());
    expected_data = writable_data.value();
    EXPECT_TRUE(tensor.set_deallocation_stream(stream));

    auto exported_result = tensor.to_dlpack();
    ASSERT_TRUE(exported_result) << runtime_error_message(exported_result.error());
    exported = exported_result.value();
    ASSERT_NE(exported, nullptr);
    ASSERT_NE(exported->deleter, nullptr);
    EXPECT_EQ(exported->dl_tensor.data, expected_data);
  }

  EXPECT_EQ(allocator->free_count, 0);
  ASSERT_NE(exported, nullptr);
  ASSERT_NE(exported->dl_tensor.shape, nullptr);
  EXPECT_EQ(exported->dl_tensor.shape[0], 2);
  EXPECT_EQ(exported->dl_tensor.shape[1], 3);
  EXPECT_EQ(exported->dl_tensor.data, expected_data);

  exported->deleter(exported);
  EXPECT_EQ(allocator->allocate_count, 1);
  EXPECT_EQ(allocator->free_count, 1);
  EXPECT_EQ(allocator->released_stream, reinterpret_cast<void*>(stream));
}

// Verifies versioned DLPack export has the same allocator-backed storage lifetime contract as the
// unversioned path, including release through the stream recorded before export.
TEST(TensorDevice, VersionedDLPackExportRetainsAllocatorBackedStorageUntilDeleter) {
  auto allocator = std::make_shared<StreamRecordingAllocator>();
  DLManagedTensorVersioned* exported = nullptr;
  void* expected_data = nullptr;
  auto allocation_stream = reinterpret_cast<cudaStream_t>(uintptr_t{0x1234});
  auto stream = reinterpret_cast<cudaStream_t>(uintptr_t{0x6789});

  {
    Tensor tensor;
    const std::array<int64_t, 2> shape{2, 3};
    const auto result = reshape_custom(tensor,
                                       shape,
                                       float32_dtype(),
                                       std::nullopt,
                                       MemoryStorageType::kSystem,
                                       allocator,
                                       allocation_stream);
    ASSERT_TRUE(result) << runtime_error_message(result.error());
    EXPECT_EQ(allocator->allocated_stream, reinterpret_cast<void*>(allocation_stream));
    auto writable_data = tensor.mutable_data();
    ASSERT_TRUE(writable_data) << runtime_error_message(writable_data.error());
    expected_data = writable_data.value();
    EXPECT_TRUE(tensor.set_deallocation_stream(stream));

    auto exported_result = tensor.to_dlpack_versioned();
    ASSERT_TRUE(exported_result) << runtime_error_message(exported_result.error());
    exported = exported_result.value();
    ASSERT_NE(exported, nullptr);
    ASSERT_NE(exported->deleter, nullptr);
    EXPECT_EQ(exported->dl_tensor.data, expected_data);
  }

  EXPECT_EQ(allocator->free_count, 0);
  ASSERT_NE(exported, nullptr);
  EXPECT_EQ(exported->version.major, HOLOSCAN_DLPACK_IMPL_VERSION_MAJOR);
  EXPECT_EQ(exported->version.minor, HOLOSCAN_DLPACK_IMPL_VERSION_MINOR);
  EXPECT_EQ(exported->flags, 0U);
  ASSERT_NE(exported->dl_tensor.shape, nullptr);
  EXPECT_EQ(exported->dl_tensor.shape[0], 2);
  EXPECT_EQ(exported->dl_tensor.shape[1], 3);
  EXPECT_EQ(exported->dl_tensor.data, expected_data);

  exported->deleter(exported);
  EXPECT_EQ(allocator->allocate_count, 1);
  EXPECT_EQ(allocator->free_count, 1);
  EXPECT_EQ(allocator->released_stream, reinterpret_cast<void*>(stream));
}

}  // namespace holoscan
