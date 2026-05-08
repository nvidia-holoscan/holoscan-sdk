/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <matx.h>

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/holoscan.hpp>
#include <holoscan/utils/matx_allocator.hpp>

namespace holoscan {

// ============================================================================
// Standalone tests (no GXF context needed)
// ============================================================================

namespace {

TEST(MatXAllocator, NullAllocatorThrows) {
  EXPECT_THROW(MatXAllocator(nullptr), std::invalid_argument);
}

TEST(MatXAllocator, NullAllocatorWithRealStreamThrows) {
  // Verify the null-allocator check fires even when a valid (non-null) stream is provided.
  cudaStream_t stream;
  auto err = cudaStreamCreate(&stream);
  ASSERT_EQ(err, cudaSuccess);
  EXPECT_THROW(MatXAllocator(nullptr, stream), std::invalid_argument);
  cudaStreamDestroy(stream);
}

TEST(MatXAllocator, NullAllocatorFullCtorThrows) {
  EXPECT_THROW(MatXAllocator(nullptr, MemoryStorageType::kDevice, nullptr), std::invalid_argument);
}

TEST(MatXAllocator, NullSharedPtrThrows) {
  // Constructing from a null shared_ptr should throw std::invalid_argument
  // because the delegating Allocator* constructor checks for null.
  std::shared_ptr<Allocator> null_alloc;
  EXPECT_THROW(MatXAllocator alloc_a(null_alloc), std::invalid_argument);
  EXPECT_THROW(MatXAllocator alloc_b(null_alloc, MemoryStorageType::kDevice),
               std::invalid_argument);
}

}  // namespace

// ============================================================================
// Operator-based tests (allocator needs GXF lifecycle)
// ============================================================================

namespace ops {

/// @brief Test operator that exercises MatXAllocator with various test scenarios.
/// The test mode determines which specific test is run.
class MatXAllocatorTestOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(MatXAllocatorTestOp)

  MatXAllocatorTestOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.param(allocator_, "allocator", "Allocator", "Memory allocator.");
    spec.param(test_mode_, "test_mode", "Test Mode", "Which test to run.", std::string("basic"));
  }

  void compute([[maybe_unused]] InputContext& op_input, [[maybe_unused]] OutputContext& op_output,
               ExecutionContext& context) override {
    auto mode = test_mode_.get();

    if (mode == "basic_alloc") {
      test_basic_alloc();
    } else if (mode == "make_tensor") {
      test_make_tensor();
    } else if (mode == "stream_aware") {
      test_stream_aware(context);
    } else if (mode == "multiple_tensors") {
      test_multiple_tensors();
    } else if (mode == "matx_operations") {
      test_matx_operations();
    } else if (mode == "accessors") {
      test_accessors();
    } else if (mode == "copy_move") {
      test_copy_move();
    } else if (mode == "dealloc_null") {
      test_dealloc_null();
    } else if (mode == "allocate_zero") {
      test_allocate_zero();
    } else if (mode == "storage_type_validation") {
      test_storage_type_validation();
    } else if (mode == "allocation_failure") {
      test_allocation_failure();
    } else if (mode == "host_memory") {
      test_host_memory();
    } else if (mode == "shared_ptr_basic") {
      test_shared_ptr_basic();
    } else if (mode == "shared_ptr_stream") {
      test_shared_ptr_stream();
    } else if (mode == "shared_ptr_host") {
      test_shared_ptr_host();
    } else if (mode == "with_stream") {
      test_with_stream();
    } else if (mode == "with_stream_validation") {
      test_with_stream_validation();
    } else {
      HOLOSCAN_LOG_ERROR("Unknown test mode: {}", mode);
    }
  }

 private:
  void test_basic_alloc() {
    auto* alloc = allocator_.get().get();
    MatXAllocator matx_alloc(alloc);

    // Allocate 1024 bytes
    void* ptr = matx_alloc.allocate(1024);
    if (!ptr) {
      throw std::runtime_error("basic_alloc: allocation returned null");
    }

    // Verify we can write to the device memory
    float value = 42.0f;
    auto err = cudaMemcpy(ptr, &value, sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
      throw std::runtime_error("basic_alloc: cudaMemcpy H2D failed");
    }

    // Verify we can read it back
    float result = 0.0f;
    err = cudaMemcpy(&result, ptr, sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
      throw std::runtime_error("basic_alloc: cudaMemcpy D2H failed");
    }
    if (result != 42.0f) {
      throw std::runtime_error("basic_alloc: data mismatch");
    }

    // Deallocate
    matx_alloc.deallocate(ptr, 1024);

    HOLOSCAN_LOG_INFO("PASS: basic_alloc");
  }

  void test_make_tensor() {
    auto* alloc = allocator_.get().get();
    HOLOSCAN_LOG_INFO("make_tensor: creating MatXAllocator wrapper");
    MatXAllocator matx_alloc(alloc);

    HOLOSCAN_LOG_INFO("make_tensor: calling matx::make_tensor<float>({{10}}, matx_alloc)");
    auto tensor = matx::make_tensor<float>({10}, matx_alloc);
    HOLOSCAN_LOG_INFO("make_tensor: tensor created, Data()={}", static_cast<void*>(tensor.Data()));

    // Write values directly via cudaMemcpy instead of SetVals
    std::vector<float> src = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    auto err = cudaMemcpy(tensor.Data(), src.data(), 10 * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
      throw std::runtime_error("make_tensor: cudaMemcpy H2D failed");
    }
    HOLOSCAN_LOG_INFO("make_tensor: values written via cudaMemcpy");

    std::vector<float> host(10);
    err = cudaMemcpy(host.data(), tensor.Data(), 10 * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
      throw std::runtime_error("make_tensor: cudaMemcpy D2H failed");
    }

    for (int i = 0; i < 10; ++i) {
      if (host[i] != static_cast<float>(i + 1)) {
        throw std::runtime_error("make_tensor: data mismatch at index " + std::to_string(i));
      }
    }

    HOLOSCAN_LOG_INFO("PASS: make_tensor");
  }

  void test_stream_aware([[maybe_unused]] ExecutionContext& context) {
    auto* alloc = allocator_.get().get();

    cudaStream_t stream;
    auto cuda_err = cudaStreamCreate(&stream);
    if (cuda_err != cudaSuccess) {
      throw std::runtime_error("stream_aware: failed to create CUDA stream");
    }

    // Use a stream-bound allocator to exercise allocate_async/free_async paths.
    MatXAllocator stream_alloc(alloc, stream);
    if (stream_alloc.stream() != stream) {
      cudaStreamDestroy(stream);
      throw std::runtime_error("stream_aware: stream accessor mismatch");
    }

    constexpr size_t kNumElems = 10;
    constexpr size_t kBytes = kNumElems * sizeof(float);
    void* raw_ptr = stream_alloc.allocate(kBytes);
    if (!raw_ptr) {
      cudaStreamDestroy(stream);
      throw std::runtime_error("stream_aware: stream allocation returned null");
    }

    // Wrap the allocated memory in a MatX tensor view (no ownership transfer).
    auto tensor = matx::make_tensor<float>(static_cast<float*>(raw_ptr), {kNumElems});

    // Note: cudaMemcpyAsync with pageable host memory (std::vector) is correct but
    // may not be fully asynchronous — CUDA may internally stage the copy when the
    // source is pageable. This is acceptable for a unit test verifying allocator
    // behavior; use pinned memory (cudaMallocHost) in production for true async.
    std::vector<float> src = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    auto err = cudaMemcpyAsync(tensor.Data(), src.data(), kBytes, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
      stream_alloc.deallocate(raw_ptr, kBytes);
      cudaStreamDestroy(stream);
      throw std::runtime_error("stream_aware: cudaMemcpy H2D failed");
    }

    // Perform a MatX operation on a specific stream
    (tensor = tensor * 2.f).run(stream);

    std::vector<float> host(kNumElems);
    err = cudaMemcpyAsync(host.data(), tensor.Data(), kBytes, cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) {
      stream_alloc.deallocate(raw_ptr, kBytes);
      cudaStreamDestroy(stream);
      throw std::runtime_error("stream_aware: cudaMemcpy D2H failed");
    }

    // Synchronize and verify
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
      stream_alloc.deallocate(raw_ptr, kBytes);
      cudaStreamDestroy(stream);
      throw std::runtime_error("stream_aware: stream sync failed");
    }

    for (size_t i = 0; i < kNumElems; ++i) {
      float expected = static_cast<float>((i + 1) * 2);
      if (host[i] != expected) {
        stream_alloc.deallocate(raw_ptr, kBytes);
        cudaStreamDestroy(stream);
        throw std::runtime_error("stream_aware: data mismatch at index " + std::to_string(i));
      }
    }

    // Explicit deallocation on the same stream exercises free_async path.
    stream_alloc.deallocate(raw_ptr, kBytes);
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
      cudaStreamDestroy(stream);
      throw std::runtime_error("stream_aware: post-free stream sync failed");
    }

    cudaStreamDestroy(stream);
    HOLOSCAN_LOG_INFO("PASS: stream_aware");
  }

  void test_multiple_tensors() {
    auto* alloc = allocator_.get().get();
    MatXAllocator matx_alloc(alloc);

    // Create multiple tensors from the same allocator
    auto tensor1 = matx::make_tensor<float>({10}, matx_alloc);
    auto tensor2 = matx::make_tensor<float>({20}, matx_alloc);
    auto tensor3 = matx::make_tensor<float>({5}, matx_alloc);

    // Write values via cudaMemcpy (avoid SetVals which may dereference device memory)
    std::vector<float> src = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    auto err = cudaMemcpy(tensor1.Data(), src.data(), 10 * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
      throw std::runtime_error("multiple_tensors: cudaMemcpy H2D failed");
    }

    // Verify they point to different memory
    if (static_cast<void*>(tensor1.Data()) == static_cast<void*>(tensor2.Data())) {
      throw std::runtime_error("multiple_tensors: tensor1 and tensor2 share memory");
    }
    if (static_cast<void*>(tensor1.Data()) == static_cast<void*>(tensor3.Data())) {
      throw std::runtime_error("multiple_tensors: tensor1 and tensor3 share memory");
    }

    // Verify tensor1 data integrity
    std::vector<float> host(10);
    err = cudaMemcpy(host.data(), tensor1.Data(), 10 * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
      throw std::runtime_error("multiple_tensors: cudaMemcpy D2H failed");
    }

    for (int i = 0; i < 10; ++i) {
      if (host[i] != static_cast<float>(i + 1)) {
        throw std::runtime_error("multiple_tensors: data mismatch at index " + std::to_string(i));
      }
    }

    HOLOSCAN_LOG_INFO("PASS: multiple_tensors");
  }

  void test_matx_operations() {
    auto* alloc = allocator_.get().get();
    MatXAllocator matx_alloc(alloc);

    // Create a tensor and perform MatX operations
    auto tensor = matx::make_tensor<float>({10}, matx_alloc);

    // Write values via cudaMemcpy (avoid SetVals which may dereference device memory)
    std::vector<float> src = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    auto err = cudaMemcpy(tensor.Data(), src.data(), 10 * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
      throw std::runtime_error("matx_operations: cudaMemcpy H2D failed");
    }

    // Run: tensor = tensor * 2 + 1
    (tensor = tensor * 2.f + matx::ones<float>({10})).run();

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
      throw std::runtime_error("matx_operations: sync failed");
    }

    std::vector<float> host(10);
    err = cudaMemcpy(host.data(), tensor.Data(), 10 * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
      throw std::runtime_error("matx_operations: cudaMemcpy D2H failed");
    }

    for (int i = 0; i < 10; ++i) {
      float expected = static_cast<float>((i + 1) * 2 + 1);
      if (host[i] != expected) {
        throw std::runtime_error("matx_operations: data mismatch at index " + std::to_string(i) +
                                 " expected=" + std::to_string(expected) +
                                 " got=" + std::to_string(host[i]));
      }
    }

    HOLOSCAN_LOG_INFO("PASS: matx_operations");
  }

  void test_accessors() {
    auto* alloc = allocator_.get().get();

    // Default construction
    MatXAllocator matx_alloc_default(alloc);
    if (matx_alloc_default.allocator() != alloc) {
      throw std::runtime_error("accessors: allocator() mismatch");
    }
    if (matx_alloc_default.storage_type() != MemoryStorageType::kDevice) {
      throw std::runtime_error("accessors: default storage_type() mismatch");
    }
    if (matx_alloc_default.stream() != nullptr) {
      throw std::runtime_error("accessors: default stream() should be nullptr");
    }

    // With host memory type
    MatXAllocator matx_alloc_host(alloc, MemoryStorageType::kHost);
    if (matx_alloc_host.storage_type() != MemoryStorageType::kHost) {
      throw std::runtime_error("accessors: host storage_type() mismatch");
    }

    HOLOSCAN_LOG_INFO("PASS: accessors");
  }

  void test_copy_move() {
    auto* alloc = allocator_.get().get();

    MatXAllocator original(alloc, MemoryStorageType::kDevice, nullptr);
    MatXAllocator copy(original);

    if (copy.allocator() != original.allocator()) {
      throw std::runtime_error("copy_move: copy allocator mismatch");
    }
    if (copy.storage_type() != original.storage_type()) {
      throw std::runtime_error("copy_move: copy storage_type mismatch");
    }
    if (copy.stream() != original.stream()) {
      throw std::runtime_error("copy_move: copy stream mismatch");
    }

    // Verify the copied allocator can allocate and deallocate.
    void* ptr = copy.allocate(256);
    if (!ptr) {
      throw std::runtime_error("copy_move: copy allocation returned null");
    }
    copy.deallocate(ptr, 256);

    MatXAllocator moved(std::move(copy));
    if (moved.allocator() != alloc) {
      throw std::runtime_error("copy_move: moved allocator mismatch");
    }

    // Verify the moved allocator can allocate and deallocate.
    ptr = moved.allocate(256);
    if (!ptr) {
      throw std::runtime_error("copy_move: moved allocation returned null");
    }
    moved.deallocate(ptr, 256);

    HOLOSCAN_LOG_INFO("PASS: copy_move");
  }

  void test_dealloc_null() {
    auto* alloc = allocator_.get().get();
    MatXAllocator matx_alloc(alloc);

    // Should not crash or throw
    matx_alloc.deallocate(nullptr, 0);
    matx_alloc.deallocate(nullptr, 1024);

    HOLOSCAN_LOG_INFO("PASS: dealloc_null");
  }

  void test_allocate_zero() {
    auto* alloc = allocator_.get().get();
    MatXAllocator matx_alloc(alloc);

    // allocate(0) should return nullptr without throwing
    void* ptr = matx_alloc.allocate(0);
    if (ptr != nullptr) {
      throw std::runtime_error("allocate_zero: expected nullptr for size=0");
    }
    // deallocate(nullptr) should be a no-op
    matx_alloc.deallocate(ptr, 0);

    HOLOSCAN_LOG_INFO("PASS: allocate_zero");
  }

  void test_storage_type_validation() {
    auto* alloc = allocator_.get().get();

    // If this allocator is a CudaAllocator, constructing with kHost + stream
    // should throw std::invalid_argument.
    auto* cuda_alloc = dynamic_cast<CudaAllocator*>(alloc);
    if (cuda_alloc) {
      cudaStream_t stream;
      auto err = cudaStreamCreate(&stream);
      if (err != cudaSuccess) {
        throw std::runtime_error("storage_type_validation: failed to create stream");
      }
      bool threw = false;
      try {
        MatXAllocator bad_alloc(alloc, MemoryStorageType::kHost, stream);
      } catch (const std::invalid_argument&) {
        threw = true;
      }
      cudaStreamDestroy(stream);
      if (!threw) {
        throw std::runtime_error(
            "storage_type_validation: expected std::invalid_argument for kHost + stream");
      }
    }

    HOLOSCAN_LOG_INFO("PASS: storage_type_validation");
  }

  void test_allocation_failure() {
    auto* alloc = allocator_.get().get();
    MatXAllocator matx_alloc(alloc);

    // Attempt to allocate far more than the pool can provide.
    // BlockMemoryPool with 1 MB blocks cannot satisfy a single 8 MB allocation.
    // RMMAllocator with 32 MB max will fail on a 1 GB allocation.
    constexpr size_t kHugeSize = 1024ULL * 1024ULL * 1024ULL;  // 1 GB
    bool threw_bad_alloc = false;
    try {
      void* ptr = matx_alloc.allocate(kHugeSize);
      // If allocation somehow succeeded, free it to avoid leaking.
      if (ptr) {
        matx_alloc.deallocate(ptr, kHugeSize);
      }
    } catch (const std::bad_alloc&) {
      threw_bad_alloc = true;
    }

    if (!threw_bad_alloc) {
      throw std::runtime_error(
          "allocation_failure: expected std::bad_alloc for oversized allocation");
    }

    HOLOSCAN_LOG_INFO("PASS: allocation_failure");
  }

  void test_host_memory() {
    auto* alloc = allocator_.get().get();

    // Allocate host (pinned) memory via the synchronous path with kHost.
    MatXAllocator host_alloc(alloc, MemoryStorageType::kHost);
    if (host_alloc.storage_type() != MemoryStorageType::kHost) {
      throw std::runtime_error("host_memory: storage_type() mismatch");
    }

    constexpr size_t kNumElems = 16;
    constexpr size_t kBytes = kNumElems * sizeof(float);
    void* ptr = host_alloc.allocate(kBytes);
    if (!ptr) {
      throw std::runtime_error("host_memory: allocation returned null");
    }

    // Host (pinned) memory should be directly accessible from the CPU.
    auto* fptr = static_cast<float*>(ptr);
    for (size_t i = 0; i < kNumElems; ++i) {
      fptr[i] = static_cast<float>(i + 1);
    }

    // Verify values
    for (size_t i = 0; i < kNumElems; ++i) {
      float expected = static_cast<float>(i + 1);
      if (fptr[i] != expected) {
        host_alloc.deallocate(ptr, kBytes);
        throw std::runtime_error("host_memory: data mismatch at index " + std::to_string(i));
      }
    }

    host_alloc.deallocate(ptr, kBytes);
    HOLOSCAN_LOG_INFO("PASS: host_memory");
  }

  void test_shared_ptr_basic() {
    // Construct MatXAllocator from shared_ptr<Allocator> (no stream).
    // allocator_.get() returns std::shared_ptr<Allocator>& — should match
    // the new shared_ptr constructor overload directly.
    MatXAllocator matx_alloc(allocator_.get());

    void* ptr = matx_alloc.allocate(1024);
    if (!ptr) {
      throw std::runtime_error("shared_ptr_basic: allocation returned null");
    }

    // Verify write/read on device memory
    float value = 99.0f;
    auto err = cudaMemcpy(ptr, &value, sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
      throw std::runtime_error("shared_ptr_basic: cudaMemcpy H2D failed");
    }
    float result = 0.0f;
    err = cudaMemcpy(&result, ptr, sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
      throw std::runtime_error("shared_ptr_basic: cudaMemcpy D2H failed");
    }
    if (result != 99.0f) {
      throw std::runtime_error("shared_ptr_basic: data mismatch");
    }

    matx_alloc.deallocate(ptr, 1024);

    // Also verify make_tensor works through the shared_ptr-constructed allocator.
    auto tensor = matx::make_tensor<float>({8}, matx_alloc);
    std::vector<float> src = {1, 2, 3, 4, 5, 6, 7, 8};
    err = cudaMemcpy(tensor.Data(), src.data(), 8 * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
      throw std::runtime_error("shared_ptr_basic: tensor cudaMemcpy H2D failed");
    }
    std::vector<float> host(8);
    err = cudaMemcpy(host.data(), tensor.Data(), 8 * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
      throw std::runtime_error("shared_ptr_basic: tensor cudaMemcpy D2H failed");
    }
    for (int i = 0; i < 8; ++i) {
      if (host[i] != static_cast<float>(i + 1)) {
        throw std::runtime_error("shared_ptr_basic: tensor data mismatch at " + std::to_string(i));
      }
    }

    HOLOSCAN_LOG_INFO("PASS: shared_ptr_basic");
  }

  void test_shared_ptr_stream() {
    // Construct MatXAllocator from shared_ptr<Allocator> + stream.
    cudaStream_t stream;
    auto cuda_err = cudaStreamCreate(&stream);
    if (cuda_err != cudaSuccess) {
      throw std::runtime_error("shared_ptr_stream: failed to create stream");
    }

    MatXAllocator stream_alloc(allocator_.get(), stream);
    if (stream_alloc.stream() != stream) {
      cudaStreamDestroy(stream);
      throw std::runtime_error("shared_ptr_stream: stream accessor mismatch");
    }
    if (stream_alloc.storage_type() != MemoryStorageType::kDevice) {
      cudaStreamDestroy(stream);
      throw std::runtime_error("shared_ptr_stream: storage_type mismatch");
    }

    // Allocate, write, verify, deallocate
    constexpr size_t kBytes = 10 * sizeof(float);
    void* raw = stream_alloc.allocate(kBytes);
    if (!raw) {
      cudaStreamDestroy(stream);
      throw std::runtime_error("shared_ptr_stream: allocation returned null");
    }

    std::vector<float> src = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100};
    auto err = cudaMemcpyAsync(raw, src.data(), kBytes, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
      stream_alloc.deallocate(raw, kBytes);
      cudaStreamDestroy(stream);
      throw std::runtime_error("shared_ptr_stream: H2D failed");
    }

    std::vector<float> host(10);
    err = cudaMemcpyAsync(host.data(), raw, kBytes, cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) {
      stream_alloc.deallocate(raw, kBytes);
      cudaStreamDestroy(stream);
      throw std::runtime_error("shared_ptr_stream: D2H failed");
    }
    cudaStreamSynchronize(stream);

    for (int i = 0; i < 10; ++i) {
      if (host[i] != src[i]) {
        stream_alloc.deallocate(raw, kBytes);
        cudaStreamDestroy(stream);
        throw std::runtime_error("shared_ptr_stream: data mismatch at " + std::to_string(i));
      }
    }

    stream_alloc.deallocate(raw, kBytes);
    cudaStreamDestroy(stream);
    HOLOSCAN_LOG_INFO("PASS: shared_ptr_stream");
  }

  void test_shared_ptr_host() {
    // Construct MatXAllocator from shared_ptr with kHost storage type.
    MatXAllocator host_alloc(allocator_.get(), MemoryStorageType::kHost);
    if (host_alloc.storage_type() != MemoryStorageType::kHost) {
      throw std::runtime_error("shared_ptr_host: storage_type mismatch");
    }

    constexpr size_t kBytes = 16 * sizeof(float);
    void* ptr = host_alloc.allocate(kBytes);
    if (!ptr) {
      throw std::runtime_error("shared_ptr_host: allocation returned null");
    }

    // Host memory should be directly accessible from CPU.
    auto* fptr = static_cast<float*>(ptr);
    for (int i = 0; i < 16; ++i) {
      fptr[i] = static_cast<float>(i * 10);
    }
    for (int i = 0; i < 16; ++i) {
      if (fptr[i] != static_cast<float>(i * 10)) {
        host_alloc.deallocate(ptr, kBytes);
        throw std::runtime_error("shared_ptr_host: data mismatch at " + std::to_string(i));
      }
    }

    host_alloc.deallocate(ptr, kBytes);
    HOLOSCAN_LOG_INFO("PASS: shared_ptr_host");
  }

  void test_with_stream() {
    // Verify with_stream() creates a new allocator bound to a different stream.
    MatXAllocator base_alloc(allocator_.get());
    if (base_alloc.stream() != nullptr) {
      throw std::runtime_error("with_stream: base should have null stream");
    }

    cudaStream_t stream1;
    cudaStream_t stream2;
    auto err = cudaStreamCreate(&stream1);
    if (err != cudaSuccess) {
      throw std::runtime_error("with_stream: failed to create stream1");
    }
    err = cudaStreamCreate(&stream2);
    if (err != cudaSuccess) {
      cudaStreamDestroy(stream1);
      throw std::runtime_error("with_stream: failed to create stream2");
    }

    auto alloc1 = base_alloc.with_stream(stream1);
    auto alloc2 = base_alloc.with_stream(stream2);

    // All three should share the same underlying allocator.
    if (alloc1.allocator() != base_alloc.allocator()) {
      cudaStreamDestroy(stream1);
      cudaStreamDestroy(stream2);
      throw std::runtime_error("with_stream: alloc1 allocator mismatch");
    }
    if (alloc2.allocator() != base_alloc.allocator()) {
      cudaStreamDestroy(stream1);
      cudaStreamDestroy(stream2);
      throw std::runtime_error("with_stream: alloc2 allocator mismatch");
    }

    // Each should have the correct stream.
    if (alloc1.stream() != stream1) {
      cudaStreamDestroy(stream1);
      cudaStreamDestroy(stream2);
      throw std::runtime_error("with_stream: alloc1 stream mismatch");
    }
    if (alloc2.stream() != stream2) {
      cudaStreamDestroy(stream1);
      cudaStreamDestroy(stream2);
      throw std::runtime_error("with_stream: alloc2 stream mismatch");
    }

    // Storage type should be preserved.
    if (alloc1.storage_type() != base_alloc.storage_type()) {
      cudaStreamDestroy(stream1);
      cudaStreamDestroy(stream2);
      throw std::runtime_error("with_stream: storage_type not preserved");
    }

    // Verify allocation through stream-bound allocator works.
    void* ptr = alloc1.allocate(512);
    if (!ptr) {
      cudaStreamDestroy(stream1);
      cudaStreamDestroy(stream2);
      throw std::runtime_error("with_stream: allocation returned null");
    }
    alloc1.deallocate(ptr, 512);
    cudaStreamSynchronize(stream1);

    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    HOLOSCAN_LOG_INFO("PASS: with_stream");
  }

  void test_with_stream_validation() {
    auto* alloc = allocator_.get().get();

    // If the allocator is a CudaAllocator, constructing a kHost allocator
    // and then calling with_stream(non-null) should throw because the
    // primary constructor rejects CudaAllocator + stream + non-kDevice.
    auto* cuda_alloc = dynamic_cast<CudaAllocator*>(alloc);
    if (cuda_alloc) {
      // First create a kHost allocator (no stream — valid).
      MatXAllocator host_alloc(alloc, MemoryStorageType::kHost);

      cudaStream_t stream;
      auto err = cudaStreamCreate(&stream);
      if (err != cudaSuccess) {
        throw std::runtime_error("with_stream_validation: failed to create stream");
      }

      bool threw = false;
      try {
        // with_stream() delegates to MatXAllocator(allocator_, kHost, stream)
        // which should throw std::invalid_argument.
        auto bad = host_alloc.with_stream(stream);
        (void)bad;
      } catch (const std::invalid_argument&) {
        threw = true;
      }
      cudaStreamDestroy(stream);
      if (!threw) {
        throw std::runtime_error("with_stream_validation: expected std::invalid_argument");
      }
    }

    HOLOSCAN_LOG_INFO("PASS: with_stream_validation");
  }

  Parameter<std::shared_ptr<Allocator>> allocator_;
  Parameter<std::string> test_mode_;
};

}  // namespace ops

// ============================================================================
// Test applications — each configures a different allocator type.
// ============================================================================

/// @brief Test app with RMMAllocator (CudaAllocator subclass).
class MatXAllocatorTestApp : public holoscan::Application {
 public:
  void set_test_mode(const std::string& mode) { test_mode_ = mode; }

  void compose() override {
    auto rmm = make_resource<RMMAllocator>("rmm-allocator",
                                           Arg("device_memory_initial_size", std::string("16MB")),
                                           Arg("device_memory_max_size", std::string("32MB")),
                                           Arg("host_memory_initial_size", std::string("16MB")),
                                           Arg("host_memory_max_size", std::string("32MB")));

    auto test_op = make_operator<ops::MatXAllocatorTestOp>("matx_allocator_test_op",
                                                           make_condition<CountCondition>(1),
                                                           Arg("allocator", rmm),
                                                           Arg("test_mode", test_mode_));

    add_operator(test_op);
  }

 private:
  std::string test_mode_ = "basic_alloc";
};

/// @brief Test app with BlockMemoryPool (non-CudaAllocator, exercises Path 2).
class MatXAllocatorBlockPoolApp : public holoscan::Application {
 public:
  void set_test_mode(const std::string& mode) { test_mode_ = mode; }

  void compose() override {
    auto pool = make_resource<BlockMemoryPool>(
        "block-pool",
        Arg("storage_type", static_cast<int32_t>(1)),           // kDevice
        Arg("block_size", static_cast<uint64_t>(1024 * 1024)),  // 1 MB blocks
        Arg("num_blocks", static_cast<uint64_t>(4)));

    auto test_op = make_operator<ops::MatXAllocatorTestOp>("matx_allocator_test_op",
                                                           make_condition<CountCondition>(1),
                                                           Arg("allocator", pool),
                                                           Arg("test_mode", test_mode_));

    add_operator(test_op);
  }

 private:
  std::string test_mode_ = "basic_alloc";
};

/// @brief Test app with StreamOrderedAllocator (CudaAllocator subclass, Path 1 variant).
class MatXAllocatorStreamOrderedApp : public holoscan::Application {
 public:
  void set_test_mode(const std::string& mode) { test_mode_ = mode; }

  void compose() override {
    auto alloc =
        make_resource<StreamOrderedAllocator>("stream-ordered",
                                              Arg("device_memory_initial_size", std::string("1MB")),
                                              Arg("device_memory_max_size", std::string("32MB")),
                                              Arg("release_threshold", std::string("4MB")),
                                              Arg("dev_id", static_cast<int32_t>(0)));

    auto test_op = make_operator<ops::MatXAllocatorTestOp>("matx_allocator_test_op",
                                                           make_condition<CountCondition>(1),
                                                           Arg("allocator", alloc),
                                                           Arg("test_mode", test_mode_));

    add_operator(test_op);
  }

 private:
  std::string test_mode_ = "basic_alloc";
};

/// @brief Test app with UnboundedAllocator (non-CudaAllocator, exercises Path 3/4).
class MatXAllocatorUnboundedApp : public holoscan::Application {
 public:
  void set_test_mode(const std::string& mode) { test_mode_ = mode; }

  void compose() override {
    auto alloc = make_resource<UnboundedAllocator>("unbounded");

    auto test_op = make_operator<ops::MatXAllocatorTestOp>("matx_allocator_test_op",
                                                           make_condition<CountCondition>(1),
                                                           Arg("allocator", alloc),
                                                           Arg("test_mode", test_mode_));

    add_operator(test_op);
  }

 private:
  std::string test_mode_ = "basic_alloc";
};

// ============================================================================
// GTest test cases — each launches an app with the appropriate test mode.
// ============================================================================

namespace {

// --- RMMAllocator tests (CudaAllocator, Path 1 + Path 4) ---

TEST(MatXAllocator, BasicAllocDealloc) {
  auto app = make_application<MatXAllocatorTestApp>();
  app->set_test_mode("basic_alloc");

  testing::internal::CaptureStderr();
  app->run();
  std::string log = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log.find("PASS: basic_alloc") != std::string::npos) << "=== LOG ===\n"
                                                                  << log << "\n===========\n";
}

TEST(MatXAllocator, MakeTensor) {
  auto app = make_application<MatXAllocatorTestApp>();
  app->set_test_mode("make_tensor");

  testing::internal::CaptureStderr();
  app->run();
  std::string log = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log.find("PASS: make_tensor") != std::string::npos) << "=== LOG ===\n"
                                                                  << log << "\n===========\n";
}

TEST(MatXAllocator, StreamAwareAllocation) {
  auto app = make_application<MatXAllocatorTestApp>();
  app->set_test_mode("stream_aware");

  testing::internal::CaptureStderr();
  app->run();
  std::string log = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log.find("PASS: stream_aware") != std::string::npos) << "=== LOG ===\n"
                                                                   << log << "\n===========\n";
}

TEST(MatXAllocator, MultipleTensors) {
  auto app = make_application<MatXAllocatorTestApp>();
  app->set_test_mode("multiple_tensors");

  testing::internal::CaptureStderr();
  app->run();
  std::string log = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log.find("PASS: multiple_tensors") != std::string::npos) << "=== LOG ===\n"
                                                                       << log << "\n===========\n";
}

TEST(MatXAllocator, MatXOperations) {
  auto app = make_application<MatXAllocatorTestApp>();
  app->set_test_mode("matx_operations");

  testing::internal::CaptureStderr();
  app->run();
  std::string log = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log.find("PASS: matx_operations") != std::string::npos) << "=== LOG ===\n"
                                                                      << log << "\n===========\n";
}

TEST(MatXAllocator, Accessors) {
  auto app = make_application<MatXAllocatorTestApp>();
  app->set_test_mode("accessors");

  testing::internal::CaptureStderr();
  app->run();
  std::string log = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log.find("PASS: accessors") != std::string::npos) << "=== LOG ===\n"
                                                                << log << "\n===========\n";
}

TEST(MatXAllocator, CopyMoveSemantics) {
  auto app = make_application<MatXAllocatorTestApp>();
  app->set_test_mode("copy_move");

  testing::internal::CaptureStderr();
  app->run();
  std::string log = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log.find("PASS: copy_move") != std::string::npos) << "=== LOG ===\n"
                                                                << log << "\n===========\n";
}

TEST(MatXAllocator, DeallocateNullIsNoOp) {
  auto app = make_application<MatXAllocatorTestApp>();
  app->set_test_mode("dealloc_null");

  testing::internal::CaptureStderr();
  app->run();
  std::string log = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log.find("PASS: dealloc_null") != std::string::npos) << "=== LOG ===\n"
                                                                   << log << "\n===========\n";
}

TEST(MatXAllocator, AllocateZeroReturnsNull) {
  auto app = make_application<MatXAllocatorTestApp>();
  app->set_test_mode("allocate_zero");

  testing::internal::CaptureStderr();
  app->run();
  std::string log = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log.find("PASS: allocate_zero") != std::string::npos) << "=== LOG ===\n"
                                                                    << log << "\n===========\n";
}

TEST(MatXAllocator, StorageTypeValidation) {
  auto app = make_application<MatXAllocatorTestApp>();
  app->set_test_mode("storage_type_validation");

  testing::internal::CaptureStderr();
  app->run();
  std::string log = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log.find("PASS: storage_type_validation") != std::string::npos)
      << "=== LOG ===\n"
      << log << "\n===========\n";
}

// --- BlockMemoryPool tests (non-CudaAllocator, Path 2 + Path 4) ---

TEST(MatXAllocator, BlockPoolBasicAlloc) {
  auto app = make_application<MatXAllocatorBlockPoolApp>();
  app->set_test_mode("basic_alloc");

  testing::internal::CaptureStderr();
  app->run();
  std::string log = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log.find("PASS: basic_alloc") != std::string::npos) << "=== LOG ===\n"
                                                                  << log << "\n===========\n";
}

TEST(MatXAllocator, BlockPoolStreamAware) {
  auto app = make_application<MatXAllocatorBlockPoolApp>();
  app->set_test_mode("stream_aware");

  testing::internal::CaptureStderr();
  app->run();
  std::string log = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log.find("PASS: stream_aware") != std::string::npos) << "=== LOG ===\n"
                                                                   << log << "\n===========\n";
}

// --- StreamOrderedAllocator tests (CudaAllocator, Path 1 variant) ---

TEST(MatXAllocator, StreamOrderedStreamAware) {
  auto app = make_application<MatXAllocatorStreamOrderedApp>();
  app->set_test_mode("stream_aware");

  testing::internal::CaptureStderr();
  app->run();
  std::string log = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log.find("PASS: stream_aware") != std::string::npos) << "=== LOG ===\n"
                                                                   << log << "\n===========\n";
}

// --- UnboundedAllocator tests (non-CudaAllocator, Path 4) ---

TEST(MatXAllocator, UnboundedBasicAlloc) {
  auto app = make_application<MatXAllocatorUnboundedApp>();
  app->set_test_mode("basic_alloc");

  testing::internal::CaptureStderr();
  app->run();
  std::string log = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log.find("PASS: basic_alloc") != std::string::npos) << "=== LOG ===\n"
                                                                  << log << "\n===========\n";
}

// --- Allocation failure tests ---

TEST(MatXAllocator, AllocationFailureThrows) {
  // Use BlockMemoryPool with limited capacity — a 1 GB allocation must fail.
  auto app = make_application<MatXAllocatorBlockPoolApp>();
  app->set_test_mode("allocation_failure");

  testing::internal::CaptureStderr();
  app->run();
  std::string log = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log.find("PASS: allocation_failure") != std::string::npos)
      << "=== LOG ===\n"
      << log << "\n===========\n";
}

// --- Host memory tests ---

TEST(MatXAllocator, HostMemoryAllocation) {
  // RMMAllocator supports host (pinned) memory via the kHost storage type.
  auto app = make_application<MatXAllocatorTestApp>();
  app->set_test_mode("host_memory");

  testing::internal::CaptureStderr();
  app->run();
  std::string log = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log.find("PASS: host_memory") != std::string::npos) << "=== LOG ===\n"
                                                                  << log << "\n===========\n";
}

// --- shared_ptr constructor tests ---

TEST(MatXAllocator, SharedPtrBasicAlloc) {
  auto app = make_application<MatXAllocatorTestApp>();
  app->set_test_mode("shared_ptr_basic");

  testing::internal::CaptureStderr();
  app->run();
  std::string log = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log.find("PASS: shared_ptr_basic") != std::string::npos) << "=== LOG ===\n"
                                                                       << log << "\n===========\n";
}

TEST(MatXAllocator, SharedPtrStreamAware) {
  auto app = make_application<MatXAllocatorTestApp>();
  app->set_test_mode("shared_ptr_stream");

  testing::internal::CaptureStderr();
  app->run();
  std::string log = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log.find("PASS: shared_ptr_stream") != std::string::npos) << "=== LOG ===\n"
                                                                        << log << "\n===========\n";
}

TEST(MatXAllocator, SharedPtrHostMemory) {
  auto app = make_application<MatXAllocatorTestApp>();
  app->set_test_mode("shared_ptr_host");

  testing::internal::CaptureStderr();
  app->run();
  std::string log = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log.find("PASS: shared_ptr_host") != std::string::npos) << "=== LOG ===\n"
                                                                      << log << "\n===========\n";
}

// --- with_stream() tests ---

TEST(MatXAllocator, WithStream) {
  auto app = make_application<MatXAllocatorTestApp>();
  app->set_test_mode("with_stream");

  testing::internal::CaptureStderr();
  app->run();
  std::string log = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log.find("PASS: with_stream") != std::string::npos) << "=== LOG ===\n"
                                                                  << log << "\n===========\n";
}

TEST(MatXAllocator, WithStreamValidation) {
  auto app = make_application<MatXAllocatorTestApp>();
  app->set_test_mode("with_stream_validation");

  testing::internal::CaptureStderr();
  app->run();
  std::string log = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log.find("PASS: with_stream_validation") != std::string::npos)
      << "=== LOG ===\n"
      << log << "\n===========\n";
}

}  // namespace
}  // namespace holoscan
