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
#include <set>
#include <string>
#include <utility>  // for std::pair

#include <holoscan/core/executors/gpu_resident/gpu_resident_executor.hpp>
#include <holoscan/core/gpu_resident_operator.hpp>
#include <holoscan/holoscan.hpp>
#include <holoscan/utils/cuda_macros.hpp>

#include "../system/env_wrapper.hpp"
#include "test_operators.hpp"

/// This file tests high-level GPU-resident fragment and executor APIs.
/// This does not test any functionality supposed to happen after the graph is
/// launched.
/// Those functionalities are tested in the example applications.

namespace holoscan {

// Test fragment for GPU-resident API tests
class TestGPUResidentFragment : public Fragment {
 public:
  void compose() override {
    auto source = make_operator<TestSourceGpuOp>("source");
    auto sink = make_operator<TestSinkGpuOp>("sink");
    add_flow(source, sink);
  }
};

// Test fixture for GPU-resident operator and fragment tests
class GPUResidentFragmentTest : public ::testing::Test {
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

// ================================================================================================
// Basic Operator API Tests
// ================================================================================================

// Test basic operator construction
TEST_F(GPUResidentFragmentTest, TestOperatorConstruction) {
  auto op = std::make_shared<TestSourceGpuOp>();
  ASSERT_NE(op, nullptr);
  EXPECT_EQ(op->name(), "");
}

// Test operator with name
TEST_F(GPUResidentFragmentTest, TestOperatorWithName) {
  Fragment fragment;
  auto op = fragment.make_operator<TestSourceGpuOp>("test_source");
  ASSERT_NE(op, nullptr);
  EXPECT_EQ(op->name(), "test_source");
}

// Test operator setup with device_output
TEST_F(GPUResidentFragmentTest, TestOperatorSetupDeviceOutput) {
  Fragment fragment;
  auto op = fragment.make_operator<TestSourceGpuOp>("test_source");

  // Check that the operator spec has the output port
  auto spec = op->spec();
  ASSERT_NE(spec, nullptr);

  auto& outputs = spec->outputs();
  EXPECT_EQ(outputs.size(), 1);
  EXPECT_NE(outputs.find("out"), outputs.end());
  EXPECT_EQ(outputs["out"]->memory_block_size(), sizeof(int) * 128);
}

// Test operator setup with device_input and device_output
TEST_F(GPUResidentFragmentTest, TestOperatorSetupDeviceInputOutput) {
  Fragment fragment;
  auto op = fragment.make_operator<TestComputeGpuOp>("test_compute");

  auto spec = op->spec();
  ASSERT_NE(spec, nullptr);

  auto& inputs = spec->inputs();
  auto& outputs = spec->outputs();

  EXPECT_EQ(inputs.size(), 1);
  EXPECT_EQ(outputs.size(), 1);

  EXPECT_NE(inputs.find("in"), inputs.end());
  EXPECT_NE(outputs.find("out"), outputs.end());

  EXPECT_EQ(inputs["in"]->memory_block_size(), sizeof(int) * 128);
  EXPECT_EQ(outputs["out"]->memory_block_size(), sizeof(int) * 128);
}

// Test operator setup with device_input only
TEST_F(GPUResidentFragmentTest, TestOperatorSetupDeviceInput) {
  Fragment fragment;
  auto op = fragment.make_operator<TestSinkGpuOp>("test_sink");

  auto spec = op->spec();
  ASSERT_NE(spec, nullptr);

  auto& inputs = spec->inputs();
  EXPECT_EQ(inputs.size(), 1);
  EXPECT_NE(inputs.find("in"), inputs.end());
  EXPECT_EQ(inputs["in"]->memory_block_size(), sizeof(int) * 128);
}

// Test multiple device inputs
TEST_F(GPUResidentFragmentTest, TestMultipleDeviceInputs) {
  Fragment fragment;
  auto op = fragment.make_operator<TestMultiInputGpuOp>("test_multi");

  auto spec = op->spec();
  auto& inputs = spec->inputs();
  auto& outputs = spec->outputs();

  EXPECT_EQ(inputs.size(), 2);
  EXPECT_EQ(outputs.size(), 1);

  EXPECT_NE(inputs.find("in1"), inputs.end());
  EXPECT_NE(inputs.find("in2"), inputs.end());
  EXPECT_NE(outputs.find("out"), outputs.end());

  EXPECT_EQ(inputs["in1"]->memory_block_size(), sizeof(int) * 64);
  EXPECT_EQ(inputs["in2"]->memory_block_size(), sizeof(float) * 32);
  EXPECT_EQ(outputs["out"]->memory_block_size(), sizeof(double) * 128);
}

// Test operator type is set correctly
TEST_F(GPUResidentFragmentTest, TestOperatorType) {
  Fragment fragment;
  auto op = fragment.make_operator<TestSourceGpuOp>("test_source");

  // GPU-resident operators should have operator_type set to kUnknown
  EXPECT_EQ(op->operator_type(), Operator::OperatorType::kUnknown);
}

// ================================================================================================
// Automatic Executor Detection Tests
// ================================================================================================

// Test that GPU-resident operator automatically gets GPU-resident executor
TEST_F(GPUResidentFragmentTest, TestAutomaticGPUResidentExecutor) {
  Fragment fragment;
  auto op = fragment.make_operator<TestSourceGpuOp>("test_source");
  fragment.add_operator(op);

  // After adding GPU-resident operator, fragment should automatically use GPU-resident executor
  EXPECT_TRUE(fragment.is_gpu_resident());

  // Should be able to get GPU-resident executor without manually setting it
  EXPECT_NO_THROW(auto exec = op->gpu_resident_executor());
  auto exec = op->gpu_resident_executor();
  EXPECT_NE(exec, nullptr);
}

// Test is_gpu_resident() API
TEST_F(GPUResidentFragmentTest, TestIsGPUResident) {
  auto fragment = std::make_shared<TestGPUResidentFragment>();

  // Before composing, should not be GPU-resident
  EXPECT_FALSE(fragment->is_gpu_resident());

  // Compose the fragment (adds GPU-resident operators)
  fragment->compose();

  // After composing with GPU-resident operators, should be GPU-resident
  EXPECT_TRUE(fragment->is_gpu_resident());
}

// ================================================================================================
// Device Memory and CUDA Stream Tests
// ================================================================================================

// Tests whether an uninitialized GPU-resident fragment returns null device memory
TEST_F(GPUResidentFragmentTest, TestFragmentUninitializedDeviceMemory) {
  Fragment fragment;
  auto op = fragment.make_operator<TestSourceGpuOp>("test_source");
  fragment.add_operator(op);

  auto addr = op->device_memory("out");
  EXPECT_EQ(addr, nullptr);
}

// Tests whether an uninitialized GPU-resident fragment returns a valid CUDA stream
TEST_F(GPUResidentFragmentTest, TestFragmentUninitializedCudaStream) {
  Fragment fragment;
  auto op = fragment.make_operator<TestSourceGpuOp>("test_source");
  fragment.add_operator(op);

  auto stream = op->cuda_stream();
  ASSERT_NE(stream, nullptr);
  EXPECT_NE(*stream, nullptr);
}

// Test device_memory API
TEST_F(GPUResidentFragmentTest, TestDeviceMemory) {
  Fragment fragment;
  auto op = fragment.make_operator<TestSourceGpuOp>("test_source");
  auto sink = fragment.make_operator<TestSinkGpuOp>("test_sink");
  fragment.add_flow(op, sink);

  // Before fragment initialization, device_memory should return nullptr
  auto addr = op->device_memory("out");
  EXPECT_EQ(addr, nullptr);

  addr = sink->device_memory("in");
  EXPECT_EQ(addr, nullptr);

  // After fragment initialization, device_memory should be allocated
  auto executor = std::dynamic_pointer_cast<GPUResidentExecutor>(fragment.executor_shared());
  ASSERT_NE(executor, nullptr);
  EXPECT_TRUE(executor->initialize_fragment());

  // Now device_memory should return valid addresses
  auto in_addr = sink->device_memory("in");
  auto out_addr = op->device_memory("out");
  EXPECT_NE(in_addr, nullptr);
  EXPECT_NE(out_addr, nullptr);

  // The memory addresses should be the same
  EXPECT_EQ(in_addr, out_addr);
}

// ================================================================================================
// Fragment API Tests
// ================================================================================================

// Test timeout_ms() API
TEST_F(GPUResidentFragmentTest, TestSetGPUResidentTimeout) {
  auto fragment = std::make_shared<TestGPUResidentFragment>();
  fragment->compose();

  // Should not throw when setting timeout
  EXPECT_NO_THROW(fragment->gpu_resident().timeout_ms(1000));
  EXPECT_NO_THROW(fragment->gpu_resident().timeout_ms(0));
}

// Test that Fragment API throws when not using GPU-resident executor
TEST_F(GPUResidentFragmentTest, TestAPIThrowsWithoutGPUResidentExecutor) {
  auto fragment = std::make_shared<TestGPUResidentFragment>();

  // Don't compose (no GPU-resident operators added)

  // The accessor itself should throw when fragment is not GPU-resident
  EXPECT_THROW(fragment->gpu_resident(), holoscan::RuntimeError);

  // All accessor method calls should also throw (each creates a new accessor that throws)
  EXPECT_THROW(fragment->gpu_resident().timeout_ms(1000), holoscan::RuntimeError);
  EXPECT_THROW(fragment->gpu_resident().tear_down(), holoscan::RuntimeError);
  EXPECT_THROW(fragment->gpu_resident().result_ready(), holoscan::RuntimeError);
  EXPECT_THROW(fragment->gpu_resident().data_ready(), holoscan::RuntimeError);
  EXPECT_THROW(fragment->gpu_resident().is_launched(), holoscan::RuntimeError);
}

// ================================================================================================
// Executor API Tests
// ================================================================================================

// Test device memory API with invalid port name
TEST_F(GPUResidentFragmentTest, TestDeviceMemoryInvalidPort) {
  auto fragment = std::make_shared<TestGPUResidentFragment>();
  fragment->compose();

  // Get the automatically created executor
  auto executor = std::dynamic_pointer_cast<GPUResidentExecutor>(fragment->executor_shared());
  ASSERT_NE(executor, nullptr);

  EXPECT_TRUE(executor->initialize_fragment());

  auto graph = fragment->graph_shared();
  auto source_node = graph->find_node("source");

  // Query with invalid port name should throw runtime error as the port is not found
  EXPECT_THROW(executor->device_memory(source_node, "invalid_port"), std::out_of_range);
}

// Test CUDA initialization
TEST_F(GPUResidentFragmentTest, TestCudaInitialization) {
  auto fragment = std::make_shared<TestGPUResidentFragment>();
  fragment->compose();

  // Get the automatically created executor
  auto executor = std::dynamic_pointer_cast<GPUResidentExecutor>(fragment->executor_shared());
  ASSERT_NE(executor, nullptr);

  // Should not throw when initializing CUDA
  EXPECT_NO_THROW(executor->initialize_cuda());

  // Verify that device 0 is set
  int current_device;
  HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaGetDevice(&current_device),
                                 "Failed to get current CUDA device");
  EXPECT_EQ(current_device, 0);
}

TEST_F(GPUResidentFragmentTest, TestLongChain) {
  Fragment fragment;
  auto source = fragment.make_operator<TestSourceGpuOp>("source");
  std::shared_ptr<Operator> prev = source;
  for (int i = 0; i < 100; ++i) {
    auto compute = fragment.make_operator<TestComputeGpuOp>("compute_" + std::to_string(i));
    fragment.add_flow(prev, compute);
    prev = compute;
  }
  auto sink = fragment.make_operator<TestSinkGpuOp>("sink");
  fragment.add_flow(prev, sink);

  // Get the automatically created executor
  auto executor = std::dynamic_pointer_cast<GPUResidentExecutor>(fragment.executor_shared());
  ASSERT_NE(executor, nullptr);

  // Should be able to initialize
  EXPECT_TRUE(executor->initialize_fragment());
}

// ================================================================================================
// Device Pointer Access Tests
// ================================================================================================

// Test that device pointers can be retrieved
TEST_F(GPUResidentFragmentTest, TestDevicePointerAccess) {
  auto fragment = std::make_shared<TestGPUResidentFragment>();
  fragment->compose();

  // Get the automatically created executor
  auto executor = std::dynamic_pointer_cast<GPUResidentExecutor>(fragment->executor_shared());
  ASSERT_NE(executor, nullptr);

  void* data_ready_ptr = fragment->gpu_resident().data_ready_device_address();
  void* result_ready_ptr = fragment->gpu_resident().result_ready_device_address();
  void* tear_down_ptr = fragment->gpu_resident().tear_down_device_address();

  // All pointers should be non-null
  EXPECT_NE(data_ready_ptr, nullptr);
  EXPECT_NE(result_ready_ptr, nullptr);
  EXPECT_NE(tear_down_ptr, nullptr);

  // All pointers should be unique
  EXPECT_NE(data_ready_ptr, result_ready_ptr);
  EXPECT_NE(data_ready_ptr, tear_down_ptr);
  EXPECT_NE(result_ready_ptr, tear_down_ptr);
}

// Test that executor device pointers match fragment accessor device pointers
TEST_F(GPUResidentFragmentTest, TestExecutorDevicePointerAccess) {
  auto fragment = std::make_shared<TestGPUResidentFragment>();
  fragment->compose();

  // Get the automatically created executor
  auto executor = std::dynamic_pointer_cast<GPUResidentExecutor>(fragment->executor_shared());
  ASSERT_NE(executor, nullptr);

  // Get device pointers from both fragment accessor and executor
  void* fragment_data_ready = fragment->gpu_resident().data_ready_device_address();
  void* fragment_result_ready = fragment->gpu_resident().result_ready_device_address();
  void* fragment_tear_down = fragment->gpu_resident().tear_down_device_address();

  void* executor_data_ready = executor->data_ready_device_address();
  void* executor_result_ready = executor->result_ready_device_address();
  void* executor_tear_down = executor->tear_down_device_address();

  // Fragment accessor and executor should return the same pointers
  EXPECT_EQ(fragment_data_ready, executor_data_ready);
  EXPECT_EQ(fragment_result_ready, executor_result_ready);
  EXPECT_EQ(fragment_tear_down, executor_tear_down);
}

// ================================================================================================
// GPU-resident Status Check Tests with an empty workload graph
// ================================================================================================

TEST_F(GPUResidentFragmentTest, TestTimeoutEmptyWorkloadGraph) {
  Fragment fragment;
  auto source = fragment.make_operator<TestSourceGpuOp>("source");
  fragment.add_operator(source);

  unsigned long long timeouts[3] = {100, 500, 1000};
  unsigned long long overhead_ms =
      20;  // less than 20ms could have been enough, but giving a bit extra buffer time to be safe

  for (auto timeout : timeouts) {
    EXPECT_NO_THROW(fragment.gpu_resident().timeout_ms(timeout));  // set timeout to timeout ms

    auto future = fragment.run_async();

    auto start_time = std::chrono::steady_clock::now();
    while (!fragment.gpu_resident().is_launched()) {
      if (std::chrono::steady_clock::now() - start_time >= std::chrono::seconds(5)) {
        future.get();
        FAIL() << "Fragment did not launch within 5 seconds";
      }
    }
    // wait for timeout ms
    std::this_thread::sleep_for(std::chrono::milliseconds(timeout + overhead_ms));
    // see if the graph has been torn down
    EXPECT_FALSE(fragment.gpu_resident().is_launched());
    future.get();
  }
}

TEST_F(GPUResidentFragmentTest, TestTimeoutInRunningGraph) {
  Fragment fragment;
  auto source = fragment.make_operator<TestSourceGpuOp>("source");
  fragment.add_operator(source);

  EXPECT_NO_THROW(fragment.gpu_resident().timeout_ms(1000));

  // capture the output and look for HOLOSCAN_LOG_ERROR messages
  testing::internal::CaptureStderr();
  auto future = fragment.run_async();

  // wait for the graph to be launched (max 5 seconds)
  auto start_time = std::chrono::steady_clock::now();
  while (!fragment.gpu_resident().is_launched()) {
    if (std::chrono::steady_clock::now() - start_time >= std::chrono::seconds(5)) {
      future.get();
      FAIL() << "Fragment did not launch within 5 seconds";
    }
  }

  // try to set the timeout to 100ms in a running graph
  fragment.gpu_resident().timeout_ms(100);

  std::string log_output = testing::internal::GetCapturedStderr();

  EXPECT_TRUE(log_output.find("timeout_ms cannot be set.") != std::string::npos)
      << "Expected error not found in log output:\n"
      << log_output;

  // wait for the workload to be torn down
  std::this_thread::sleep_for(std::chrono::milliseconds(1020));  // 20 ms buffer time to be safe
  EXPECT_FALSE(fragment.gpu_resident().is_launched());
  future.get();
}
}  // namespace holoscan

// ================================================================================================
// GPU-resident indegree constraint tests
// ================================================================================================
namespace holoscan {

// Prior indegree > 0 across two add_flow calls should be rejected in GPU-resident mode
TEST_F(GPUResidentFragmentTest, DownstreamIndegreeNonZeroAcrossCalls) {
  Fragment fragment;

  auto tx1 = fragment.make_operator<TestSourceGpuOp>("tx1");
  auto tx2 = fragment.make_operator<TestSourceGpuOp>("tx2");
  auto rx = fragment.make_operator<TestSinkGpuOp>("rx");

  // First connection establishes indegree(rx.in) = 1
  fragment.add_flow(tx1, rx, {{"out", "in"}});

  // Second connection to the same downstream input must throw in GPU-resident path
  EXPECT_THROW(fragment.add_flow(tx2, rx, {{"out", "in"}}), holoscan::RuntimeError);
}

// Two outputs to one input in a single call should be rejected in GPU-resident mode
TEST_F(GPUResidentFragmentTest, TestTwoOutputsToOneInput_InSingleCall) {
  Fragment fragment;

  auto tx = fragment.make_operator<TestTwoOutGpuOp>("tx");
  auto rx = fragment.make_operator<TestSinkGpuOp>("rx");

  const std::pair<std::string, std::string> p1{"out0", "in"};
  const std::pair<std::string, std::string> p2{"out1", "in"};
  std::set<std::pair<std::string, std::string>> port_pairs{p1, p2};

  EXPECT_THROW(fragment.add_flow(tx, rx, port_pairs), holoscan::RuntimeError);
}

}  // namespace holoscan
