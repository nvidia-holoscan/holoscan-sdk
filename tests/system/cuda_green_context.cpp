/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <vector>

#include "../config.hpp"
#include "holoscan/holoscan.hpp"
#include "holoscan/core/resources/gxf/cuda_green_context.hpp"
#include "holoscan/core/resources/gxf/cuda_green_context_pool.hpp"
#include "holoscan/operators/ping_tensor_rx/ping_tensor_rx.hpp"
#include "holoscan/operators/ping_tensor_tx/ping_tensor_tx.hpp"

namespace holoscan {

namespace ops {

/**
 * @brief Test operator that uses a CUDA green context pool and green context.
 *
 * This operator demonstrates how to use the green context APIs and provides
 * a way to test their functionality.
 */
class CudaGreenContextTestOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(CudaGreenContextTestOp)

  CudaGreenContextTestOp() = default;

  void setup(OperatorSpec& spec) {
    spec.input<TensorMap>("in");
    spec.output<TensorMap>("out");
    spec.param(allocator_, "allocator", "Allocator", "Memory allocator for tensors");
  }

  void compute(InputContext& op_input, OutputContext& op_output, ExecutionContext& context) {
    auto input = op_input.receive<TensorMap>("in");
    if (!input) {
      return;
    }

    // Simple pass-through operation to test green context functionality
    auto output = input.value();
    op_output.emit(output, "out");

    // Mark that compute was called successfully
    compute_called_ = true;
  }

  bool compute_was_called() const { return compute_called_; }

 private:
  Parameter<std::shared_ptr<Allocator>> allocator_;
  bool compute_called_ = false;
};

}  // namespace ops

/**
 * @brief Test fixture for CUDA Green Context tests.
 */
class CudaGreenContextTest : public testing::Test {
 public:
  void SetUp() override {
    using namespace holoscan;
    app_ = std::make_unique<Application>();
  }

  void TearDown() override { app_.reset(); }

  std::unique_ptr<Application> app_;
};

// ================================================================================================
// CudaGreenContextPool Tests
// ================================================================================================
TEST_F(CudaGreenContextTest, CudaGreenContextPoolCreation) {
  // Test default construction
  auto pool1 = app_->make_resource<CudaGreenContextPool>("test_pool_1");
  EXPECT_NE(pool1, nullptr);
  EXPECT_EQ(pool1->name(), "test_pool_1");

  // Test construction with parameters
  std::vector<uint32_t> partitions = {2, 2, 2, 2};
  auto pool2 = app_->make_resource<CudaGreenContextPool>("test_pool_2",
                                                         0,                  // dev_id
                                                         0,                  // flags
                                                         partitions.size(),  // num_partitions
                                                         partitions);

  EXPECT_NE(pool2, nullptr);
  EXPECT_EQ(pool2->name(), "test_pool_2");
}

TEST_F(CudaGreenContextTest, CudaGreenContextPoolInitialization) {
  std::vector<uint32_t> partitions = {2, 2};
  auto pool = app_->make_resource<CudaGreenContextPool>("init_test_pool",
                                                        0,                  // dev_id
                                                        0,                  // flags
                                                        partitions.size(),  // num_partitions
                                                        partitions);

  EXPECT_NE(pool, nullptr);

  // Initialize the pool
  EXPECT_NO_THROW(pool->initialize());

  // Check that the underlying GXF component is available
  auto gxf_pool = pool->get();
  EXPECT_NE(gxf_pool, nullptr);
}

TEST_F(CudaGreenContextTest, CudaGreenContextPoolMultiplePartitions) {
  // Test with multiple partitions
  std::vector<uint32_t> partitions = {2, 2, 4};
  auto pool = app_->make_resource<CudaGreenContextPool>("multi_partition_pool",
                                                        0,                  // dev_id
                                                        0,                  // flags
                                                        partitions.size(),  // num_partitions
                                                        partitions);

  EXPECT_NE(pool, nullptr);
  EXPECT_NO_THROW(pool->initialize());
}

// ================================================================================================
// CudaGreenContext Tests
// ================================================================================================

TEST_F(CudaGreenContextTest, CudaGreenContextCreation) {
  // First create a pool
  std::vector<uint32_t> partitions = {4, 4};
  auto pool = app_->make_resource<CudaGreenContextPool>("context_pool",
                                                        0,                  // dev_id
                                                        0,                  // flags
                                                        partitions.size(),  // num_partitions
                                                        partitions);

  EXPECT_NE(pool, nullptr);
  pool->initialize();

  // Create green contexts from the pool
  auto context1 = app_->make_resource<CudaGreenContext>("green_context_1",
                                                        pool,  // green_context_pool
                                                        0);    // index
  EXPECT_NE(context1, nullptr);
  EXPECT_EQ(context1->name(), "green_context_1");

  auto context2 = app_->make_resource<CudaGreenContext>("green_context_2", pool, 1);
  EXPECT_NE(context2, nullptr);
  EXPECT_EQ(context2->name(), "green_context_2");
}

TEST_F(CudaGreenContextTest, CudaGreenContextInitialization) {
  // Create pool and initialize
  std::vector<uint32_t> partitions = {4};
  auto pool = app_->make_resource<CudaGreenContextPool>("context_init_pool", 0, 0, 1, partitions);
  pool->initialize();

  // Create and initialize green context
  auto context = app_->make_resource<CudaGreenContext>("init_context",
                                                       pool,  // green_context_pool
                                                       0);    // index

  EXPECT_NO_THROW(context->initialize());

  // Check that the underlying GXF component is available
  auto green_context = context->get();
  EXPECT_NE(green_context, nullptr);
}

TEST_F(CudaGreenContextTest, CudaDefaultGreenContextInitializationWithIndex) {
  // Create pool and initialize
  std::vector<uint32_t> partitions = {4, 4};
  // Test default green context index=0
  auto pool = app_->make_resource<CudaGreenContextPool>(
      "context_init_pool", 0, 0, partitions.size(), partitions, 1);
  EXPECT_NO_THROW(pool->initialize());
  EXPECT_NE(pool->get(), nullptr);

  // Create and initialize green context
  auto context = app_->make_resource<CudaGreenContext>("init_context", pool, -1);

  EXPECT_NO_THROW(context->initialize());

  // Check that the underlying GXF component is available
  auto green_context = context->get();
  EXPECT_NE(green_context, nullptr);
}

TEST_F(CudaGreenContextTest, CudaDefaultGreenContextInitialization) {
  // Create pool and initialize
  std::vector<uint32_t> partitions = {4, 4};
  // Test default green context index=-1(last partition)
  auto pool = app_->make_resource<CudaGreenContextPool>(
      "context_init_pool", 0, 0, partitions.size(), partitions);
  EXPECT_NO_THROW(pool->initialize());
  EXPECT_NE(pool->get(), nullptr);

  // Create and initialize green context
  auto context = app_->make_resource<CudaGreenContext>("init_context", pool);
  EXPECT_NO_THROW(context->initialize());

  // Check that the underlying GXF component is available
  EXPECT_NE(context->get(), nullptr);
}

// ================================================================================================
// Resource Management Tests
// ================================================================================================

TEST_F(CudaGreenContextTest, ResourceLifetime) {
  std::vector<uint32_t> partitions = {4};
  auto pool = app_->make_resource<CudaGreenContextPool>("lifetime_pool", 0, 0, 1, partitions);

  auto context = app_->make_resource<CudaGreenContext>("lifetime_context", pool, 0);

  // Initialize both
  pool->initialize();
  context->initialize();

  // Reset the context first (should be safe)
  context.reset();

  // Pool should still be valid
  EXPECT_NE(pool->get(), nullptr);

  // Reset the pool
  pool.reset();
}

TEST_F(CudaGreenContextTest, MultipleContextsFromSamePool) {
  // Create pool with multiple partitions
  std::vector<uint32_t> partitions = {4, 4};
  auto pool = app_->make_resource<CudaGreenContextPool>(
      "multi_context_pool", 0, 0, partitions.size(), partitions);
  pool->initialize();

  // Create multiple contexts from the same pool
  std::vector<std::shared_ptr<CudaGreenContext>> contexts;
  for (int i = 0; i < partitions.size(); ++i) {
    auto context = app_->make_resource<CudaGreenContext>("context_" + std::to_string(i), pool, i);
    context->initialize();
    contexts.push_back(context);
  }

  // All contexts should be valid
  for (const auto& context : contexts) {
    EXPECT_NE(context->get(), nullptr);
  }
}

}  // namespace holoscan
