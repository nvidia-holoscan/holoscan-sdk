/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "../config.hpp"
#include "holoscan/holoscan.hpp"
#include "holoscan/operators/format_converter/format_converter.hpp"
#include "holoscan/operators/ping_tensor_rx/ping_tensor_rx.hpp"
#include "holoscan/operators/ping_tensor_tx/ping_tensor_tx.hpp"

namespace holoscan {

namespace ops {

class CudaAllocatorOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(CudaAllocatorOp)

  CudaAllocatorOp() = default;

  void setup(OperatorSpec& spec) {
    spec.param(
        cuda_allocator_, "cuda_allocator", "CUDA Allocator", "CUDA memory allocator support .");
    spec.param(cuda_stream_pool_,
               "cuda_stream_pool",
               "CUDA Stream Pool",
               "Instance of gxf::CudaStreamPool.",
               ParameterFlag::kOptional);
  }

  void compute([[maybe_unused]] InputContext& op_input, [[maybe_unused]] OutputContext& op_output,
               ExecutionContext& context) {
    auto cuda_allocator = cuda_allocator_.get();
    if (!cuda_allocator) {
      HOLOSCAN_LOG_ERROR("Failed to get CUDA allocator");
      return;
    }

    if (!cuda_stream_pool_.has_value()) {
      HOLOSCAN_LOG_INFO("Unable to allocate/free asynchronously without a stream pool");
      auto byte_ptr = cuda_allocator->allocate(4096, MemoryStorageType::kDevice);
      cuda_allocator->free(byte_ptr);
    } else {
      auto stream_pool = cuda_stream_pool_.get();
      auto maybe_stream = context.allocate_cuda_stream("internal2");
      if (!maybe_stream) {
        throw std::runtime_error(
            fmt::format("Failed to get CUDA stream: {}", maybe_stream.error().what()));
      }
      auto cuda_stream = maybe_stream.value();
      auto byte_ptr = cuda_allocator->allocate_async(4096, cuda_stream);
      auto mem_size = cuda_allocator->pool_size(MemoryStorageType::kDevice);
      HOLOSCAN_LOG_INFO("CudaAllocator->pool_size(device) = {}", mem_size);
      cuda_allocator->free_async(byte_ptr, cuda_stream);
    }
  }

 private:
  Parameter<std::shared_ptr<CudaAllocator>> cuda_allocator_;
  Parameter<std::shared_ptr<CudaStreamPool>> cuda_stream_pool_;
  size_t count_ = 1;
};

}  // namespace ops

/**
 * @brief Application testing various aspects of CudaStreamHandling
 *
 * PingTensorTxOp tests
 *    ExecutionContext::allocate_cuda_stream
 *    CudaAllocator::allocate_async
 *    CudaAllocator::free_async
 *
 */
class CudaAllocatorApp : public holoscan::Application {
 public:
  enum class PoolType { kStreamOrdered, kRMM };

  void compose() override {
    const int32_t width = 320;
    const int32_t height = 240;

    // Configure a CudaAllocator
    std::shared_ptr<CudaAllocator> cuda_allocator;
    if (stream_pool_type_ == PoolType::kStreamOrdered) {
      cuda_allocator = make_resource<StreamOrderedAllocator>(
          "stream_ordered_allocator",
          Arg{"device_memory_initial_size", std::string{"1KB"}},
          Arg{"device_memory_max_size", std::string{"32MB"}},
          Arg{"release_threshold", std::string{"4MB"}},
          Arg{"dev_id", static_cast<int32_t>(0)});
    } else if (stream_pool_type_ == PoolType::kRMM) {
      // TODO(grelee): Why does allocation fail unless `device_memory_initial_size` is larger than
      // the amount of memory requested via allocate_async?

      // Note: cannot set 0B for any Host memory sizes
      cuda_allocator =
          make_resource<RMMAllocator>("rmm-allocator",
                                      Arg{"device_memory_initial_size", std::string{"16KB"}},
                                      Arg{"device_memory_max_size", std::string{"32MB"}},
                                      Arg{"host_memory_initial_size", std::string{"1KB"}},
                                      Arg{"host_memory_max_size", std::string{"1KB"}},
                                      Arg{"dev_id", static_cast<int32_t>(0)});
    } else {
      throw std::runtime_error(
          fmt::format("Invalid pool type: {}", static_cast<int>(stream_pool_type_)));
    }
    ArgList extra_args{
        Arg("cuda_allocator", cuda_allocator),
    };

    // optionally add a CudaStreamPool
    if (use_stream_pool_) {
      auto cuda_stream_pool = make_resource<CudaStreamPool>("cuda_stream", 0, 0, 0, 1, 10);
      extra_args.add(Arg("cuda_stream_pool", cuda_stream_pool));
    }

    auto alloc_op = make_operator<ops::CudaAllocatorOp>(
        "stream_ordered_allocation_op", make_condition<CountCondition>(10), extra_args);

    add_operator(alloc_op);
  }

  void cuda_allocator_type(PoolType pool_type) { stream_pool_type_ = pool_type; }
  void use_stream_pool(bool value) { use_stream_pool_ = value; }

 private:
  bool use_stream_pool_ = false;
  PoolType stream_pool_type_ = PoolType::kStreamOrdered;
};

TEST(CudaAllocatorApps, TestStreamOrderedAllocatorAppWithStreamPool) {
  // Test fix for issue 4313690 (failure to initialize graph when using BayerDemosaicOp)
  using namespace holoscan;

  auto app = make_application<CudaAllocatorApp>();
  app->use_stream_pool(true);
  app->cuda_allocator_type(CudaAllocatorApp::PoolType::kStreamOrdered);

  // capture output to check that the expected messages were logged
  testing::internal::CaptureStderr();

  app->run();

  std::string log_output = testing::internal::GetCapturedStderr();

  // Check that expected number of bytes were allocated
  std::string stream_msg = "CudaAllocator->pool_size(device) = 4096";
  EXPECT_TRUE(log_output.find(stream_msg) != std::string::npos) << "=== LOG ===\n"
                                                                << log_output << "\n===========\n";
}

TEST(CudaAllocatorApps, TestRMMAllocatorAppWithStreamPool) {
  // Test fix for issue 4313690 (failure to initialize graph when using BayerDemosaicOp)
  using namespace holoscan;

  auto app = make_application<CudaAllocatorApp>();
  app->use_stream_pool(true);
  app->cuda_allocator_type(CudaAllocatorApp::PoolType::kRMM);

  // capture output to check that the expected messages were logged
  testing::internal::CaptureStderr();

  app->run();

  std::string log_output = testing::internal::GetCapturedStderr();

  // RMMAllocator seems to report the initial pool size here not the allocated size
  std::string stream_msg = "CudaAllocator->pool_size(device) = 16384";
  EXPECT_TRUE(log_output.find(stream_msg) != std::string::npos) << "=== LOG ===\n"
                                                                << log_output << "\n===========\n";
}

}  // namespace holoscan

TEST(CudaAllocatorApps, TestStreamOrderedAllocatorAppNoStreamPool) {
  // Test fix for issue 4313690 (failure to initialize graph when using BayerDemosaicOp)
  using namespace holoscan;

  auto app = make_application<CudaAllocatorApp>();
  app->use_stream_pool(false);
  app->cuda_allocator_type(CudaAllocatorApp::PoolType::kStreamOrdered);

  // capture output to check that the expected messages were logged
  testing::internal::CaptureStderr();

  app->run();

  std::string log_output = testing::internal::GetCapturedStderr();

  // Check that expected number of bytes were allocated
  std::string stream_msg = "Unable to allocate/free asynchronously without a stream pool";
  EXPECT_TRUE(log_output.find(stream_msg) != std::string::npos) << "=== LOG ===\n"
                                                                << log_output << "\n===========\n";
}
