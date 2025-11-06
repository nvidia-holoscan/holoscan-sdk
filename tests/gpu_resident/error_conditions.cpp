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

#include <memory>
#include <string>

#include <holoscan/core/executors/gpu_resident/gpu_resident_executor.hpp>
#include <holoscan/core/gpu_resident_operator.hpp>
#include <holoscan/holoscan.hpp>

#include "../system/env_wrapper.hpp"
#include "test_operators.hpp"

/// This file tests basic error conditions for GPU-resident fragment and executor APIs.

namespace holoscan {

// Test fixture
class GPUResidentErrorConditionsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Check if CUDA is available
    int device_count = 0;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    if (error != cudaSuccess || device_count == 0) {
      GTEST_SKIP() << "No CUDA devices available, skipping GPU-resident tests";
    }
  }
};

TEST_F(GPUResidentErrorConditionsTest, TestEmptyFragment) {
  Fragment fragment;

  // Empty fragment should not be GPU-resident (no GPU-resident operators added)
  EXPECT_FALSE(fragment.is_gpu_resident());
}

TEST_F(GPUResidentErrorConditionsTest, TestNormalOperatorGPUResidentOperator) {
  Fragment fragment;
  auto normal_op = fragment.make_operator<Operator>("normal_op");
  fragment.add_operator(normal_op);

  EXPECT_FALSE(fragment.is_gpu_resident());

  auto gpu_resident_op = fragment.make_operator<TestSourceGpuOp>("gpu_resident_op");
  EXPECT_THROW(fragment.add_operator(gpu_resident_op), holoscan::RuntimeError);
}

TEST_F(GPUResidentErrorConditionsTest, TestNormalOperatorPlusGPUResidentAddFlow) {
  Fragment fragment;
  auto normal_op = fragment.make_operator<Operator>("normal_op");
  fragment.add_operator(normal_op);

  EXPECT_FALSE(fragment.is_gpu_resident());

  auto source_op = fragment.make_operator<TestSourceGpuOp>("gpu_resident_op");
  auto sink_op = fragment.make_operator<TestSinkGpuOp>("sink_op");

  EXPECT_THROW(fragment.add_flow(source_op, sink_op), holoscan::RuntimeError);
}

// This test a single operator fragment with no CUDA code
TEST_F(GPUResidentErrorConditionsTest, TestSingleOperatorFragment) {
  Fragment fragment;
  auto source = fragment.make_operator<TestSourceGpuOp>("source");
  fragment.add_operator(source);

  // Fragment should be automatically marked as GPU-resident
  EXPECT_TRUE(fragment.is_gpu_resident());

  // Get the automatically created executor
  auto executor = std::dynamic_pointer_cast<GPUResidentExecutor>(fragment.executor_shared());
  ASSERT_NE(executor, nullptr);

  // Set log level to WARN to ensure warning messages are captured
  EnvVarWrapper wrapper("HOLOSCAN_LOG_LEVEL", "WARN");

  // Capture stderr to check for warning message
  testing::internal::CaptureStderr();

  // run_async on fragment with operator that has no CUDA code should show HOLOSCAN_LOG_WARN
  auto future = fragment.run_async();

  // wait for five seconds to see if fragment.gpu_resident().is_launched() is true. if
  // not, then future.get() and declare the test failed
  auto start_time = std::chrono::steady_clock::now();
  while (!fragment.gpu_resident().is_launched()) {
    if (std::chrono::steady_clock::now() - start_time >= std::chrono::seconds(5)) {
      FAIL() << "Fragment did not launch within 5 seconds";
      future.get();
    }
  }

  std::string log_output = testing::internal::GetCapturedStderr();

  // Verify that the warning about empty workload graph is present
  EXPECT_TRUE(log_output.find("Workload graph of GPU-resident execution is empty.") !=
              std::string::npos)
      << "Expected warning not found in log output:\n"
      << log_output;

  EXPECT_NO_THROW(fragment.gpu_resident().tear_down());

  future.get();
}

TEST_F(GPUResidentErrorConditionsTest, TestDisconnectedOperators) {
  Fragment fragment;
  auto source1 = fragment.make_operator<TestSourceGpuOp>("source1");
  auto source2 = fragment.make_operator<TestSourceGpuOp>("source2");
  fragment.add_operator(source1);
  fragment.add_operator(source2);

  // Get the automatically created executor
  auto executor = std::dynamic_pointer_cast<GPUResidentExecutor>(fragment.executor_shared());
  ASSERT_NE(executor, nullptr);

  // Multiple disconnected operators should fail (multiple root nodes)
  EXPECT_FALSE(executor->verify_graph_topology(fragment.graph()));
}

TEST_F(GPUResidentErrorConditionsTest, TestTwoSourceOneSink) {
  Fragment fragment;
  auto source1 = fragment.make_operator<TestSourceGpuOp>("source1");
  auto source2 = fragment.make_operator<TestSourceGpuOp>("source2");
  auto sink = fragment.make_operator<TestSinkGpuOp>("sink");
  fragment.add_flow(source1, sink);
  fragment.add_flow(source2, sink);

  // Get the automatically created executor
  auto executor = std::dynamic_pointer_cast<GPUResidentExecutor>(fragment.executor_shared());
  ASSERT_NE(executor, nullptr);

  auto& graph = fragment.graph();

  // Y-shaped merge should fail (multiple root nodes)
  EXPECT_FALSE(executor->verify_graph_topology(fragment.graph()));
}

TEST_F(GPUResidentErrorConditionsTest, TestOneSourceTwoSink) {
  Fragment fragment;
  auto source = fragment.make_operator<TestSourceGpuOp>("source");
  auto sink1 = fragment.make_operator<TestSinkGpuOp>("sink1");
  auto sink2 = fragment.make_operator<TestSinkGpuOp>("sink2");
  fragment.add_flow(source, sink1);

  // The following call should throw runtime error
  EXPECT_THROW(fragment.add_flow(source, sink2), holoscan::RuntimeError);
}

TEST_F(GPUResidentErrorConditionsTest, TestZeroSizeMemoryBlock) {
  Fragment fragment;

  // Creating an operator with zero-size memory block should throw std::invalid_argument
  // during setup() when spec.device_output() is called
  EXPECT_THROW(fragment.make_operator<ZeroSizeMemoryOp>("source"), std::invalid_argument);
}

TEST_F(GPUResidentErrorConditionsTest, TestInvalidPortName) {
  Fragment fragment;

  // Creating an operator with port name containing '.' should throw std::invalid_argument
  // during setup() when spec.device_output() is called
  EXPECT_THROW(fragment.make_operator<InvalidPortNameOp>("source"), std::invalid_argument);
}

// Test unsupported executor methods
TEST_F(GPUResidentErrorConditionsTest, TestUnsupportedExecutorMethods) {
  Fragment fragment;
  auto source = fragment.make_operator<TestSourceGpuOp>("source");
  fragment.add_operator(source);

  // Get the automatically created executor
  auto executor = std::dynamic_pointer_cast<GPUResidentExecutor>(fragment.executor_shared());
  ASSERT_NE(executor, nullptr);

  // These methods should throw as they're not supported by GPU-resident executor
  EXPECT_THROW(executor->context(nullptr), std::runtime_error);
  EXPECT_THROW(executor->initialize_scheduler(nullptr), std::runtime_error);
  EXPECT_THROW(executor->initialize_network_context(nullptr), std::runtime_error);
  EXPECT_THROW(executor->initialize_fragment_services(), std::runtime_error);
}

// The following test checks whether GPU-resident status check APIs throw errors
// when the fragment is not even run
TEST_F(GPUResidentErrorConditionsTest, TestStatusCheckBeforeRunAsync) {
  Fragment fragment;
  auto source = fragment.make_operator<TestSourceGpuOp>("source");
  fragment.add_operator(source);

  auto executor = std::dynamic_pointer_cast<GPUResidentExecutor>(fragment.executor_shared());
  ASSERT_NE(executor, nullptr);

  // Capture the output and look for HOLOSCAN_LOG_ERROR messages
  testing::internal::CaptureStderr();

  EXPECT_FALSE(executor->is_launched());

  executor->data_ready();
  executor->result_ready();
  executor->tear_down();

  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("GPU-resident CUDA workload is not yet launched. "
                              "data_ready trigger cannot be performed.") != std::string::npos)
      << "Expected error not found in log output:\n"
      << log_output;

  EXPECT_TRUE(log_output.find("GPU-resident CUDA workload is not yet launched. "
                              "result_ready trigger cannot be performed.") != std::string::npos)
      << "Expected error not found in log output:\n"
      << log_output;

  EXPECT_TRUE(log_output.find("GPU-resident CUDA workload is not yet launched. "
                              "tear_down trigger cannot be performed.") != std::string::npos)
      << "Expected error not found in log output:\n"
      << log_output;
}

}  // namespace holoscan
