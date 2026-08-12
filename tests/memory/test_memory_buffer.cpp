/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <type_traits>
#include <utility>

#include "../common/allocation_counter.hpp"
#include "../common/cuda_test_utils.hpp"
#include "../common/memory_allocator_test_utils.hpp"

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/cuda_execution/cuda_stream_pool.hpp>
#include <holoscan/memory/block_memory_pool.hpp>
#include <holoscan/memory/memory_buffer.hpp>
#include <holoscan/memory/tensor_storage.hpp>
#include <holoscan/memory/unbounded_allocator.hpp>
#include "core/domain/tensor_access.hpp"

namespace holoscan {
using ::holoscan::test::ScopedAllocationCounter;
using ::holoscan::test::skip_without_cuda_device;
using ::holoscan::test::StreamRecordingAllocator;

static_assert(!std::is_copy_constructible_v<BufferAllocation>);
static_assert(!std::is_copy_assignable_v<BufferAllocation>);
static_assert(std::is_move_constructible_v<BufferAllocation>);
static_assert(std::is_move_assignable_v<BufferAllocation>);

namespace {

// RAII guard that saves the current CUDA device on construction and restores it on destruction.
// Used by multi-device tests that switch the global device context mid-test to avoid leaving
// the wrong device active for subsequent tests in the same process.
class ScopedCudaDeviceRestore {
 public:
  ScopedCudaDeviceRestore() {
    if (cudaGetDevice(&device_) != cudaSuccess) {
      device_ = -1;
    }
  }

  ScopedCudaDeviceRestore(const ScopedCudaDeviceRestore&) = delete;
  ScopedCudaDeviceRestore& operator=(const ScopedCudaDeviceRestore&) = delete;

  ~ScopedCudaDeviceRestore() {
    if (device_ >= 0) {
      (void)cudaSetDevice(device_);
    }
  }

 private:
  int device_ = -1;
};

}  // namespace

// Verifies MemoryBuffer owns allocator-backed storage, exposes the allocation metadata, and
// releases the storage through the shared allocator.
TEST(MemoryBuffer, ResizeAndFreeWithSharedAllocator) {
  auto allocator = std::make_shared<UnboundedAllocator>();
  MemoryBuffer buffer;

  auto resize_result = buffer.resize(allocator, 256, MemoryStorageType::kSystem);
  ASSERT_TRUE(resize_result) << runtime_error_message(resize_result.error());
  EXPECT_EQ(buffer.size(), 256);
  EXPECT_EQ(buffer.storage_type(), MemoryStorageType::kSystem);
  ASSERT_NE(buffer.pointer(), nullptr);

  buffer.pointer()[42] = std::byte{0x5};

  auto free_result = buffer.free_buffer();
  EXPECT_TRUE(free_result) << runtime_error_message(free_result.error());
  EXPECT_EQ(buffer.size(), 0);
  EXPECT_EQ(buffer.pointer(), nullptr);
}

// Verifies MemoryBuffer can wrap externally-owned memory and invoke the supplied release callback
// when the buffer is freed.
TEST(MemoryBuffer, WrapMemoryUsesReleaseFunction) {
  bool released = false;
  std::array<std::byte, 16> data{};
  MemoryBuffer buffer;

  auto result =
      buffer.wrap_memory(data.data(), data.size(), MemoryStorageType::kSystem, [&](void*) {
        released = true;
        return expected<void, Error>{};
      });
  ASSERT_TRUE(result) << runtime_error_message(result.error());
  EXPECT_EQ(buffer.pointer(), data.data());

  auto free_result = buffer.free_buffer();
  EXPECT_TRUE(free_result) << runtime_error_message(free_result.error());
  EXPECT_TRUE(released);
}

// Verifies Tensor copy and move operations preserve shared ownership of wrapped external memory,
// keeping the release callback deferred until every Tensor handle has been destroyed.
TEST(MemoryBuffer, WrappedTensorCopiesRetainExternalStorageUntilLastHandle) {
  // Key setup: external storage is wrapped with a release callback; the original Tensor and a
  //   copy-constructed handle both reference the same wrapped memory.
  // Key assertion: release callback fires exactly once, only after the last handle drops.
  std::uint32_t release_count = 0U;
  auto* storage = new std::uint64_t{42U};
  auto tensor = make_tensor(TensorShape{1},
                            DLDataType{kDLUInt, 64U, 1U},
                            std::nullopt,
                            MemoryStorageType::kSystem,
                            storage,
                            [&](void* pointer) -> expected<void, Error> {
                              delete static_cast<std::uint64_t*>(pointer);
                              ++release_count;
                              return {};
                            });
  ASSERT_TRUE(tensor) << runtime_error_message(tensor.error());
  EXPECT_EQ(release_count, 0U);

  Tensor retained = *tensor;
  *tensor = Tensor{};
  EXPECT_EQ(release_count, 0U);
  ASSERT_NE(retained.data_as<std::uint64_t>(), nullptr);
  EXPECT_EQ(*retained.data_as<std::uint64_t>(), 42U);

  retained = Tensor{};
  EXPECT_EQ(release_count, 1U);
}

// Verifies MemoryBuffer forwards stream context to stream-aware allocators on allocation and uses
// the currently recorded stream when releasing storage.
TEST(MemoryBuffer, ResizeAndFreePropagateStreamContext) {
  auto allocator = std::make_shared<StreamRecordingAllocator>();
  MemoryBuffer buffer;
  int allocation_stream_marker = 0;
  int release_stream_marker = 0;

  auto resize_result =
      buffer.resize(allocator, 256, MemoryStorageType::kSystem, &allocation_stream_marker);
  ASSERT_TRUE(resize_result) << runtime_error_message(resize_result.error());
  EXPECT_EQ(allocator->allocate_count, 1);
  EXPECT_EQ(allocator->allocated_stream, &allocation_stream_marker);
  EXPECT_EQ(buffer.stream(), &allocation_stream_marker);

  EXPECT_TRUE(buffer.set_stream(&release_stream_marker));
  EXPECT_EQ(buffer.stream(), &release_stream_marker);

  auto free_result = buffer.free_buffer();
  EXPECT_TRUE(free_result) << runtime_error_message(free_result.error());
  EXPECT_EQ(allocator->free_count, 1);
  EXPECT_EQ(allocator->released_stream, &release_stream_marker);
}

// Verifies BufferAllocation is the move-only RAII handle over MemoryBuffer: moving transfers the
// visible token, and explicit release returns the allocation through the underlying allocator.
TEST(MemoryBuffer, BufferAllocationMoveTransfersOwnershipAndRelease) {
  auto allocator = std::make_shared<StreamRecordingAllocator>();
  int allocation_stream_marker = 0;
  int release_stream_marker = 0;

  auto maybe_allocation = BufferAllocation::allocate(
      allocator, 128, MemoryStorageType::kSystem, &allocation_stream_marker);
  ASSERT_TRUE(maybe_allocation) << runtime_error_message(maybe_allocation.error());
  auto allocation = std::move(maybe_allocation.value());
  ASSERT_TRUE(allocation);
  EXPECT_EQ(allocator->allocate_count, 1);
  EXPECT_EQ(allocator->allocated_stream, &allocation_stream_marker);

  auto* expected_pointer = allocation.data();
  BufferAllocation moved = std::move(allocation);
  // NOLINTNEXTLINE(bugprone-use-after-move)
  EXPECT_FALSE(allocation);
  EXPECT_TRUE(moved);
  EXPECT_EQ(moved.data(), expected_pointer);
  EXPECT_TRUE(moved.record_release_stream(&release_stream_marker));

  auto release_result = moved.release();
  EXPECT_TRUE(release_result) << runtime_error_message(release_result.error());
  EXPECT_FALSE(moved);
  EXPECT_EQ(allocator->free_count, 1);
  EXPECT_EQ(allocator->released_stream, &release_stream_marker);
}

// Verifies BufferAllocation can sit over a fixed BlockMemoryPool: pool exhaustion is surfaced as
// an expected error, and releasing the token returns the fixed block for reuse.
TEST(MemoryBuffer, BufferAllocationBlockMemoryPoolExhaustionAndReuse) {
  auto pool = std::make_shared<BlockMemoryPool>(MemoryStorageType::kSystem, 64, 1);
  ASSERT_TRUE(pool->initialize_pool());

  auto first = BufferAllocation::allocate(pool, 64, MemoryStorageType::kSystem);
  ASSERT_TRUE(first) << runtime_error_message(first.error());
  auto allocation = std::move(first.value());
  auto* expected_pointer = allocation.data();

  auto unavailable = BufferAllocation::allocate(pool, 1, MemoryStorageType::kSystem);
  EXPECT_FALSE(unavailable);

  auto release = allocation.release();
  ASSERT_TRUE(release) << runtime_error_message(release.error());

  auto second = BufferAllocation::allocate(pool, 1, MemoryStorageType::kSystem);
  ASSERT_TRUE(second) << runtime_error_message(second.error());
  auto reused = std::move(second.value());
  EXPECT_EQ(reused.data(), expected_pointer);

  EXPECT_TRUE(reused.release());
  EXPECT_TRUE(pool->deinitialize_pool());
}

// Verifies Tensor construction can consume the move-only BufferAllocation token while the Tensor
// retains the hidden buffer control block until the last Tensor handle drops.
TEST(MemoryBuffer, TensorAttachmentRetainsBufferAfterAllocationTokenDrops) {
  auto allocator = std::make_shared<StreamRecordingAllocator>();
  Tensor tensor;
  std::byte* expected_pointer = nullptr;

  {
    auto maybe_allocation = BufferAllocation::allocate(allocator, 64, MemoryStorageType::kSystem);
    ASSERT_TRUE(maybe_allocation) << runtime_error_message(maybe_allocation.error());
    auto allocation = std::move(maybe_allocation.value());
    expected_pointer = allocation.data();

    auto attached = attach_allocation(
        tensor, TensorShape{64}, DLDataType{kDLUInt, 8, 1}, std::nullopt, std::move(allocation));
    ASSERT_TRUE(attached) << runtime_error_message(attached.error());
    // NOLINTNEXTLINE(bugprone-use-after-move)
    EXPECT_FALSE(allocation);
  }

  // Key setup: allocation token consumed by attach_allocation; source token is empty after scope.
  // Key assertion: free_count stays 0 while the Tensor is live; becomes 1 when it is reset.
  EXPECT_EQ(allocator->free_count, 0);
  ASSERT_NE(detail::TensorAccess::context(tensor), nullptr);
  EXPECT_EQ(tensor.data(), expected_pointer);
  EXPECT_EQ(tensor.nbytes(), 64);

  tensor = Tensor();
  EXPECT_EQ(allocator->free_count, 1);
}

// Verifies a Tensor constructed from a fixed-pool BufferAllocation keeps the block checked out
// after the allocation token is consumed, then returns the block when the Tensor handle drops.
TEST(MemoryBuffer, TensorAttachmentKeepsBlockMemoryPoolAllocationCheckedOut) {
  auto pool = std::make_shared<BlockMemoryPool>(MemoryStorageType::kSystem, 64, 1);
  ASSERT_TRUE(pool->initialize_pool());

  auto maybe_allocation = BufferAllocation::allocate(pool, 64, MemoryStorageType::kSystem);
  ASSERT_TRUE(maybe_allocation) << runtime_error_message(maybe_allocation.error());
  auto allocation = std::move(maybe_allocation.value());
  auto* expected_pointer = allocation.data();

  Tensor tensor;
  auto attached = attach_allocation(
      tensor, TensorShape{64}, DLDataType{kDLUInt, 8, 1}, std::nullopt, std::move(allocation));
  ASSERT_TRUE(attached) << runtime_error_message(attached.error());

  // Key setup: pool holds only 1 block, now kept by the Tensor's hidden control block.
  // Key assertion: reallocation fails while Tensor is live; succeeds after Tensor is reset.
  auto unavailable = BufferAllocation::allocate(pool, 1, MemoryStorageType::kSystem);
  EXPECT_FALSE(unavailable);

  tensor = Tensor();

  auto available = BufferAllocation::allocate(pool, 1, MemoryStorageType::kSystem);
  ASSERT_TRUE(available) << runtime_error_message(available.error());
  auto reused = std::move(available.value());
  EXPECT_EQ(reused.data(), expected_pointer);

  EXPECT_TRUE(reused.release());
  EXPECT_TRUE(pool->deinitialize_pool());
}

// Verifies Tensor copy and move operations retain shared storage ownership until every Tensor
// handle referencing the BufferAllocation-backed context has been released.
TEST(MemoryBuffer, TensorCopyAndMoveRetainBufferAllocation) {
  auto allocator = std::make_shared<StreamRecordingAllocator>();
  Tensor tensor;

  auto maybe_allocation = BufferAllocation::allocate(allocator, 48, MemoryStorageType::kSystem);
  ASSERT_TRUE(maybe_allocation) << runtime_error_message(maybe_allocation.error());
  auto allocation = std::move(maybe_allocation.value());
  auto* expected_pointer = allocation.data();

  auto attached = attach_allocation(
      tensor, TensorShape{48}, DLDataType{kDLUInt, 8, 1}, std::nullopt, std::move(allocation));
  ASSERT_TRUE(attached) << runtime_error_message(attached.error());

  // Key setup: Tensor is copy-constructed, then the copy is move-constructed into a third handle.
  // Key assertion: free_count stays 0 while any handle is live; becomes 1 only when all drop.
  Tensor copied = tensor;
  Tensor moved = std::move(tensor);
  EXPECT_EQ(allocator->free_count, 0);
  EXPECT_EQ(copied.data(), expected_pointer);
  EXPECT_EQ(moved.data(), expected_pointer);

  copied = Tensor();
  EXPECT_EQ(allocator->free_count, 0);

  moved = Tensor();
  EXPECT_EQ(allocator->free_count, 1);
}

// Verifies DLPack export extends the lifetime of a BufferAllocation-backed Tensor beyond the C++
// Tensor handle, then releases storage when the external DLPack deleter runs.
TEST(MemoryBuffer, DLPackExportRetainsBufferAllocationBackedTensorUntilDeleter) {
  auto allocator = std::make_shared<StreamRecordingAllocator>();
  DLManagedTensor* exported = nullptr;
  std::byte* expected_pointer = nullptr;

  {
    auto maybe_allocation = BufferAllocation::allocate(allocator, 32, MemoryStorageType::kSystem);
    ASSERT_TRUE(maybe_allocation) << runtime_error_message(maybe_allocation.error());
    auto allocation = std::move(maybe_allocation.value());
    expected_pointer = allocation.data();

    Tensor tensor;
    auto attached = attach_allocation(
        tensor, TensorShape{32}, DLDataType{kDLUInt, 8, 1}, std::nullopt, std::move(allocation));
    ASSERT_TRUE(attached) << runtime_error_message(attached.error());
    auto export_result = tensor.to_dlpack();
    ASSERT_TRUE(export_result) << runtime_error_message(export_result.error());
    exported = export_result.value();
    ASSERT_NE(exported, nullptr);
    ASSERT_NE(exported->deleter, nullptr);
  }

  // Key setup: C++ Tensor drops at inner-scope exit while the DLManagedTensor export persists.
  // Key assertion: free is deferred until the DLPack deleter is explicitly invoked by the caller.
  EXPECT_EQ(allocator->free_count, 0);
  EXPECT_EQ(exported->dl_tensor.data, expected_pointer);

  exported->deleter(exported);
  EXPECT_EQ(allocator->free_count, 1);
}

// Verifies DLPack metadata reports the CUDA device that owns the allocation rather than always
// using device 0.
TEST(MemoryBuffer, TensorAttachmentReportsAllocationCudaDeviceInDLPack) {
  int device_count = 0;
  const cudaError_t device_count_error = cudaGetDeviceCount(&device_count);
  if (device_count_error != cudaSuccess || device_count < 2) {
    GTEST_SKIP() << "At least two CUDA devices are required";
  }
  const ScopedCudaDeviceRestore restore_device;

  // Key setup: pool bound to device 1; BufferAllocation-backed Tensor constructed on that device.
  // Key assertion: DLTensor.device.device_type == kDLCUDA with device_id == target_device (1).
  constexpr int32_t target_device = 1;
  auto pool = std::make_shared<BlockMemoryPool>(MemoryStorageType::kDevice, 64, 1, target_device);
  ASSERT_TRUE(pool->initialize_pool());

  auto maybe_allocation = BufferAllocation::allocate(pool, 64, MemoryStorageType::kDevice);
  ASSERT_TRUE(maybe_allocation) << runtime_error_message(maybe_allocation.error());
  auto allocation = std::move(maybe_allocation.value());

  Tensor tensor;
  auto attached = attach_allocation(
      tensor, TensorShape{64}, DLDataType{kDLUInt, 8, 1}, std::nullopt, std::move(allocation));
  ASSERT_TRUE(attached) << runtime_error_message(attached.error());
  ASSERT_NE(detail::TensorAccess::context(tensor), nullptr);

  const DLTensor& dl_tensor = detail::TensorAccess::context(tensor)->tensor.dl_tensor;
  EXPECT_EQ(dl_tensor.device.device_type, kDLCUDA);
  EXPECT_EQ(dl_tensor.device.device_id, target_device);

  tensor = Tensor();
  EXPECT_TRUE(pool->deinitialize_pool());
}

// Verifies the allocator-reported CUDA device ordinal is recorded on the MemoryBuffer so tensor
// construction can set DLPack device metadata without a per-attach pointer query.
TEST(MemoryBuffer, AllocationRecordsAllocatorDeviceId) {
  skip_without_cuda_device();
  const ScopedCudaDeviceRestore restore_device;

  auto pool = std::make_shared<BlockMemoryPool>(MemoryStorageType::kDevice, 64, 1, /*device_id=*/0);
  EXPECT_EQ(pool->device_id(MemoryStorageType::kDevice), 0);
  EXPECT_EQ(pool->device_id(MemoryStorageType::kSystem), Allocator::kUnknownDeviceId);
  // Pinned host memory is kDLCUDAHost (device_id 0 by DLPack convention), so it carries no ordinal.
  EXPECT_EQ(pool->device_id(MemoryStorageType::kHost), Allocator::kUnknownDeviceId);
  // Managed memory is kDLCUDAManaged (device_id 0 by DLPack convention), so it carries no ordinal.
  EXPECT_EQ(pool->device_id(MemoryStorageType::kCudaManaged), Allocator::kUnknownDeviceId);
  ASSERT_TRUE(pool->initialize_pool());

  auto maybe_allocation = BufferAllocation::allocate(pool, 64, MemoryStorageType::kDevice);
  ASSERT_TRUE(maybe_allocation) << runtime_error_message(maybe_allocation.error());
  auto allocation = std::move(maybe_allocation.value());
  ASSERT_NE(allocation.memory_buffer(), nullptr);
  EXPECT_EQ(allocation.memory_buffer()->device_id(), 0);

  allocation = BufferAllocation();
  EXPECT_TRUE(pool->deinitialize_pool());
}

// Wrapped external memory has no allocator-reported device, so device_id() stays unknown and tensor
// construction falls back to deriving the ordinal from the backing pointer.
TEST(MemoryBuffer, WrapMemoryLeavesDeviceIdUnknown) {
  std::array<std::byte, 64> storage{};
  MemoryBuffer buffer;
  ASSERT_TRUE(
      buffer.wrap_memory(storage.data(), storage.size(), MemoryStorageType::kSystem, nullptr));
  EXPECT_EQ(buffer.device_id(), Allocator::kUnknownDeviceId);
}

// Wrapped external memory without a device hint uses the pointer-query fallback. Reject mismatches
// rather than emitting kDLCUDA metadata for host or otherwise non-device pointers.
TEST(MemoryBuffer, WrapMemoryRejectsStorageAndPointerDeviceTypeMismatch) {
  skip_without_cuda_device();

  std::array<std::byte, 64> storage{};
  Tensor tensor;
  auto wrapped = wrap_memory(tensor,
                             TensorShape{64},
                             DLDataType{kDLUInt, 8, 1},
                             std::nullopt,
                             MemoryStorageType::kDevice,
                             storage.data(),
                             nullptr);

  ASSERT_FALSE(wrapped);
  EXPECT_EQ(wrapped.error().code, ErrorCode::kInvalidArgument);
  EXPECT_EQ(wrapped.error().message,
            "Tensor storage type does not match the CUDA pointer memory type");
}

// wrap_memory records a caller-supplied device ordinal verbatim (DLPack storage-type conventions
// are applied later at tensor construction), so callers with known provenance can avoid the
// pointer-attribute query.
TEST(MemoryBuffer, WrapMemoryRecordsProvidedDeviceId) {
  std::array<std::byte, 64> storage{};
  MemoryBuffer buffer;
  ASSERT_TRUE(buffer.wrap_memory(
      storage.data(), storage.size(), MemoryStorageType::kSystem, nullptr, /*device_id=*/3));
  EXPECT_EQ(buffer.device_id(), 3);
}

// CUDA pinned host memory maps to kDLCUDAHost with device_id 0 by DLPack convention, regardless of
// the device the allocator was bound to.
TEST(MemoryBuffer, HostAllocationReportsHostDLPackDeviceIdZero) {
  skip_without_cuda_device();
  const ScopedCudaDeviceRestore restore_device;

  auto pool = std::make_shared<BlockMemoryPool>(MemoryStorageType::kHost, 64, 1, /*device_id=*/0);
  ASSERT_TRUE(pool->initialize_pool());

  auto maybe_allocation = BufferAllocation::allocate(pool, 64, MemoryStorageType::kHost);
  ASSERT_TRUE(maybe_allocation) << runtime_error_message(maybe_allocation.error());
  auto allocation = std::move(maybe_allocation.value());

  Tensor tensor;
  auto attached = attach_allocation(
      tensor, TensorShape{64}, DLDataType{kDLUInt, 8, 1}, std::nullopt, std::move(allocation));
  ASSERT_TRUE(attached) << runtime_error_message(attached.error());
  ASSERT_NE(detail::TensorAccess::context(tensor), nullptr);

  const DLTensor& dl_tensor = detail::TensorAccess::context(tensor)->tensor.dl_tensor;
  EXPECT_EQ(dl_tensor.device.device_type, kDLCUDAHost);
  EXPECT_EQ(dl_tensor.device.device_id, 0);

  tensor = Tensor();
  EXPECT_TRUE(pool->deinitialize_pool());
}

// CUDA managed memory maps to kDLCUDAManaged with device_id 0 by DLPack convention, even when the
// allocator uses a nonzero CUDA device for allocation.
TEST(MemoryBuffer, ManagedAllocationReportsDLPackDeviceIdZero) {
  skip_without_cuda_device();
  int device_count = 0;
  const cudaError_t device_count_error = cudaGetDeviceCount(&device_count);
  ASSERT_EQ(device_count_error, cudaSuccess);
  const ScopedCudaDeviceRestore restore_device;

  const int32_t target_device = device_count > 1 ? 1 : 0;
  auto pool =
      std::make_shared<BlockMemoryPool>(MemoryStorageType::kCudaManaged, 64, 1, target_device);
  ASSERT_TRUE(pool->initialize_pool());

  auto maybe_allocation = BufferAllocation::allocate(pool, 64, MemoryStorageType::kCudaManaged);
  ASSERT_TRUE(maybe_allocation) << runtime_error_message(maybe_allocation.error());
  auto allocation = std::move(maybe_allocation.value());
  ASSERT_NE(allocation.memory_buffer(), nullptr);
  EXPECT_EQ(allocation.memory_buffer()->device_id(), Allocator::kUnknownDeviceId);

  Tensor tensor;
  auto attached = attach_allocation(
      tensor, TensorShape{64}, DLDataType{kDLUInt, 8, 1}, std::nullopt, std::move(allocation));
  ASSERT_TRUE(attached) << runtime_error_message(attached.error());
  ASSERT_NE(detail::TensorAccess::context(tensor), nullptr);

  const DLTensor& dl_tensor = detail::TensorAccess::context(tensor)->tensor.dl_tensor;
  EXPECT_EQ(dl_tensor.device.device_type, kDLCUDAManaged);
  EXPECT_EQ(dl_tensor.device.device_id, 0);

  tensor = Tensor();
  EXPECT_TRUE(pool->deinitialize_pool());
}

// Re-wrapping the same pointer a MemoryBuffer already holds must not retain a stale allocator
// device hint from a prior resize(); free_buffer() (which clears the hint) is skipped on the
// same-pointer path, so wrap_memory() must clear it directly.
TEST(MemoryBuffer, WrapSamePointerClearsAllocatorDeviceId) {
  skip_without_cuda_device();
  const ScopedCudaDeviceRestore restore_device;

  auto pool = std::make_shared<BlockMemoryPool>(MemoryStorageType::kDevice, 64, 1, /*device_id=*/0);
  ASSERT_TRUE(pool->initialize_pool());

  MemoryBuffer buffer;
  ASSERT_TRUE(buffer.resize(pool, 64, MemoryStorageType::kDevice));
  EXPECT_EQ(buffer.device_id(), 0);  // allocator-reported ordinal

  // Re-wrap the same pointer, returning it to the pool on release. The device hint must clear even
  // though free_buffer() is skipped because the pointer is unchanged.
  std::byte* pointer = buffer.pointer();
  ASSERT_TRUE(buffer.wrap_memory(
      pointer, buffer.size(), MemoryStorageType::kDevice, [pool](void* p) -> expected<void, Error> {
        return pool->free(static_cast<std::byte*>(p));
      }));
  EXPECT_EQ(buffer.device_id(), Allocator::kUnknownDeviceId);

  buffer = MemoryBuffer();  // releases via the wrap release_func
  EXPECT_TRUE(pool->deinitialize_pool());
}

// Verifies MemoryBuffer accepts borrowed CCCL stream references from the native stream pool at the
// allocation and release call sites while preserving the existing allocator-facing stream handle.
TEST(MemoryBuffer, ResizeAndFreePropagateStreamRefContext) {
  skip_without_cuda_device();

  CudaStreamPool stream_pool(/*device_id=*/0,
                             cudaStreamNonBlocking,
                             ::cuda::stream::default_priority,
                             /*reserved_size=*/2,
                             /*max_size=*/2);
  ASSERT_NO_THROW(stream_pool.initialize());
  auto maybe_allocation_lease = stream_pool.acquire();
  ASSERT_TRUE(maybe_allocation_lease) << runtime_error_message(maybe_allocation_lease.error());
  auto allocation_lease = std::move(maybe_allocation_lease.value());
  auto maybe_release_lease = stream_pool.acquire();
  ASSERT_TRUE(maybe_release_lease) << runtime_error_message(maybe_release_lease.error());
  auto release_lease = std::move(maybe_release_lease.value());

  auto allocator = std::make_shared<StreamRecordingAllocator>();
  MemoryBuffer buffer;

  auto resize_result =
      buffer.resize(allocator, 256, MemoryStorageType::kSystem, allocation_lease.stream());
  ASSERT_TRUE(resize_result) << runtime_error_message(resize_result.error());
  EXPECT_EQ(allocator->allocate_count, 1);
  EXPECT_EQ(allocator->allocated_stream, static_cast<void*>(allocation_lease.native_handle()));
  EXPECT_EQ(buffer.stream(), static_cast<void*>(allocation_lease.native_handle()));

  EXPECT_TRUE(buffer.set_stream(release_lease.stream()));
  EXPECT_EQ(buffer.stream(), static_cast<void*>(release_lease.native_handle()));

  auto free_result = buffer.free_buffer();
  EXPECT_TRUE(free_result) << runtime_error_message(free_result.error());
  EXPECT_EQ(allocator->free_count, 1);
  EXPECT_EQ(allocator->released_stream, static_cast<void*>(release_lease.native_handle()));

  EXPECT_TRUE(release_lease.release());
  EXPECT_TRUE(allocation_lease.release());
}

// Verifies moving a MemoryBuffer transfers the allocator-backed allocation and preserves the stream
// context that will be used for release.
TEST(MemoryBuffer, MovePreservesStreamAwareReleaseState) {
  auto allocator = std::make_shared<StreamRecordingAllocator>();
  int allocation_stream_marker = 0;
  int release_stream_marker = 0;

  MemoryBuffer source;
  auto resize_result =
      source.resize(allocator, 64, MemoryStorageType::kSystem, &allocation_stream_marker);
  ASSERT_TRUE(resize_result) << runtime_error_message(resize_result.error());

  MemoryBuffer moved = std::move(source);
  // NOLINTNEXTLINE(bugprone-use-after-move,clang-analyzer-cplusplus.Move)
  EXPECT_EQ(source.pointer(), nullptr);
  EXPECT_EQ(moved.stream(), &allocation_stream_marker);

  EXPECT_TRUE(moved.set_stream(&release_stream_marker));
  auto free_result = moved.free_buffer();
  EXPECT_TRUE(free_result) << runtime_error_message(free_result.error());
  EXPECT_EQ(allocator->free_count, 1);
  EXPECT_EQ(allocator->released_stream, &release_stream_marker);
}

// Verifies the current implementation uses bounded release-stream slots instead of dynamic stream
// storage. The fixed limit is a transitional default until graph setup passes the exact fanout
// count.
TEST(MemoryBuffer, ReleaseStreamRecordingUsesFixedFanoutCapacity) {
  auto allocator = std::make_shared<StreamRecordingAllocator>();
  MemoryBuffer buffer;
  EXPECT_EQ(buffer.max_fanout_release_streams(), MemoryBuffer::kDefaultMaxFanoutReleaseStreams);
  ASSERT_TRUE(buffer.resize(allocator, 64, MemoryStorageType::kSystem));

  std::array<int, MemoryBuffer::kDefaultMaxFanoutReleaseStreams + 1> stream_markers{};
  for (std::size_t i = 0; i < MemoryBuffer::kDefaultMaxFanoutReleaseStreams; ++i) {
    EXPECT_TRUE(buffer.set_stream(&stream_markers[i])) << "release stream slot " << i;
    EXPECT_EQ(buffer.stream(), &stream_markers[i]);
  }

  EXPECT_TRUE(buffer.set_stream(&stream_markers[0]));
  EXPECT_FALSE(buffer.set_stream(&stream_markers.back()));
  EXPECT_EQ(buffer.stream(), &stream_markers[MemoryBuffer::kDefaultMaxFanoutReleaseStreams - 1]);
}

// Verifies setup can narrow the fixed release-stream capacity, which lets graph-derived fanout
// limits reject unexpected extra consumers without growing hot-path storage.
TEST(MemoryBuffer, ReleaseStreamRecordingCanUseConfiguredFanoutCapacity) {
  auto allocator = std::make_shared<StreamRecordingAllocator>();
  MemoryBuffer::Options options;
  options.max_fanout_release_streams = 2;
  MemoryBuffer buffer(options);
  EXPECT_EQ(buffer.max_fanout_release_streams(), 2);
  ASSERT_TRUE(buffer.resize(allocator, 64, MemoryStorageType::kSystem));
  ASSERT_NE(buffer.stream_state(), nullptr);
  EXPECT_EQ(buffer.stream_state()->max_release_streams, 2);

  std::array<int, 3> stream_markers{};
  EXPECT_TRUE(buffer.set_stream(&stream_markers[0]));
  EXPECT_TRUE(buffer.set_stream(&stream_markers[1]));
  EXPECT_TRUE(buffer.set_stream(&stream_markers[0]));
  EXPECT_FALSE(buffer.set_stream(&stream_markers[2]));
  EXPECT_EQ(buffer.stream(), &stream_markers[1]);
}

// Verifies invalid fanout capacities fail before attaching storage, so production setup can catch
// impossible topology-derived limits early.
TEST(MemoryBuffer, ReleaseStreamCapacityRejectsInvalidConfiguration) {
  auto allocator = std::make_shared<StreamRecordingAllocator>();

  MemoryBuffer zero_capacity;
  auto zero_configured = zero_capacity.set_max_fanout_release_streams(0);
  EXPECT_FALSE(zero_configured);

  MemoryBuffer::Options too_large_options;
  too_large_options.max_fanout_release_streams = MemoryBuffer::kDefaultMaxFanoutReleaseStreams + 1;
  MemoryBuffer too_large(too_large_options);
  auto resize_result = too_large.resize(allocator, 64, MemoryStorageType::kSystem);
  EXPECT_FALSE(resize_result);
  EXPECT_EQ(allocator->allocate_count, 0);
  EXPECT_EQ(too_large.pointer(), nullptr);
}

// Verifies BufferAllocation can carry the same setup-time fanout capacity into the hidden
// MemoryBuffer control block used by Tensor storage.
TEST(MemoryBuffer, BufferAllocationPropagatesConfiguredFanoutCapacity) {
  auto allocator = std::make_shared<StreamRecordingAllocator>();
  MemoryBuffer::Options options;
  options.max_fanout_release_streams = 2;

  auto maybe_allocation =
      BufferAllocation::allocate_with_options(allocator, 64, MemoryStorageType::kSystem, options);
  ASSERT_TRUE(maybe_allocation) << runtime_error_message(maybe_allocation.error());
  auto allocation = std::move(maybe_allocation.value());
  ASSERT_TRUE(allocation);
  ASSERT_NE(allocation.memory_buffer(), nullptr);
  EXPECT_EQ(allocation.memory_buffer()->max_fanout_release_streams(), 2);

  std::array<int, 3> stream_markers{};
  EXPECT_TRUE(allocation.record_release_stream(&stream_markers[0]));
  EXPECT_TRUE(allocation.record_release_stream(&stream_markers[1]));
  EXPECT_FALSE(allocation.record_release_stream(&stream_markers[2]));

  auto release = allocation.release();
  EXPECT_TRUE(release) << runtime_error_message(release.error());
}

// Verifies repeated release-stream recording uses the fixed StreamState slots without host heap
// allocation once the MemoryBuffer has been initialized.
TEST(MemoryBuffer, ReleaseStreamRecordingHotPathDoesNotAllocate) {
  auto allocator = std::make_shared<StreamRecordingAllocator>();
  MemoryBuffer buffer;
  ASSERT_TRUE(buffer.resize(allocator, 64, MemoryStorageType::kSystem));

  std::array<int, MemoryBuffer::kDefaultMaxFanoutReleaseStreams> stream_markers{};

  // Key setup: 16 iterations each filling all kDefaultMaxFanoutReleaseStreams stream slots.
  // Key assertion: ScopedAllocationCounter reports zero allocations during the hot-path loop.
  bool ok = true;
  std::size_t allocation_count = 0;
  {
    const ScopedAllocationCounter allocations;
    for (int iteration = 0; iteration < 16; ++iteration) {
      for (std::size_t i = 0; i < stream_markers.size(); ++i) {
        if (!buffer.set_stream(&stream_markers[i])) {
          ok = false;
          break;
        }
      }
      if (!ok) {
        break;
      }
    }
    allocation_count = allocations.count();
  }

  EXPECT_TRUE(ok);
  EXPECT_EQ(allocation_count, 0);
  EXPECT_EQ(buffer.stream(), &stream_markers.back());
}

// Verifies multi-consumer device-buffer release uses CUDA synchronization resources allocated at
// buffer setup time, so final release does not create host objects or CUDA events/streams.
//
// Fanout release topology:
//   MemoryBuffer (device, 64 B)
//     ├── set_stream(first_stream)
//     └── set_stream(second_stream)
//   free_buffer() merges via pre-allocated fanout_join_stream -> allocator free
TEST(MemoryBuffer, FanoutDeviceReleaseUsesPreallocatedSyncResources) {
  skip_without_cuda_device();

  CudaStreamPool stream_pool(/*device_id=*/0,
                             cudaStreamNonBlocking,
                             ::cuda::stream::default_priority,
                             /*reserved_size=*/2,
                             /*max_size=*/2);
  ASSERT_TRUE(stream_pool.initialize_pool());

  auto first_stream = stream_pool.acquire();
  ASSERT_TRUE(first_stream) << runtime_error_message(first_stream.error());
  auto first_lease = std::move(first_stream.value());

  auto second_stream = stream_pool.acquire();
  ASSERT_TRUE(second_stream) << runtime_error_message(second_stream.error());
  auto second_lease = std::move(second_stream.value());

  auto allocator = std::make_shared<StreamRecordingAllocator>();
  MemoryBuffer buffer;
  // Key setup: two stream-pool leases registered as release consumers on a single device buffer.
  // Key assertion: free_buffer() succeeds with zero heap allocations; released_stream equals the
  //   pre-allocated fanout_join_stream, not either consumer stream.
  ASSERT_TRUE(buffer.resize(allocator, 64, MemoryStorageType::kDevice));
  ASSERT_NE(buffer.pointer(), nullptr);
  ASSERT_NE(buffer.stream_state(), nullptr);
  ASSERT_TRUE(buffer.stream_state()->fanout_sync_resources_ready());
  cudaStream_t join_stream = buffer.stream_state()->fanout_join_stream();
  ASSERT_NE(join_stream, nullptr);
  EXPECT_NE(join_stream, first_lease.native_handle());
  EXPECT_NE(join_stream, second_lease.native_handle());

  EXPECT_TRUE(buffer.set_stream(first_lease.stream()));
  EXPECT_TRUE(buffer.set_stream(second_lease.stream()));

  bool free_ok = false;
  std::size_t allocation_count = 0;
  {
    const ScopedAllocationCounter allocations;
    auto free_result = buffer.free_buffer();
    free_ok = static_cast<bool>(free_result);
    allocation_count = allocations.count();
  }

  EXPECT_TRUE(free_ok);
  EXPECT_EQ(allocation_count, 0);
  EXPECT_EQ(allocator->free_count, 1);
  EXPECT_EQ(allocator->released_stream, static_cast<void*>(join_stream));

  EXPECT_TRUE(second_lease.release());
  EXPECT_TRUE(first_lease.release());
}

}  // namespace holoscan
