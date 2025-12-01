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
#include <holoscan/utils/cuda_macros.hpp>

#include "../system/env_wrapper.hpp"
#include "test_operators.hpp"

/// This file tests GPU-resident data ready handler functionality.
/// It verifies that data ready handler fragments can be registered,
/// initialized, and properly integrated with the main workload.

namespace holoscan {

// Test operator that uses data ready handler functionality
class TestDataReadyHandlerOp : public GPUResidentOperator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS_SUPER(TestDataReadyHandlerOp, GPUResidentOperator)
  TestDataReadyHandlerOp() = default;

  void setup([[maybe_unused]] OperatorSpec& spec) override {}

  void compute([[maybe_unused]] InputContext& op_input, [[maybe_unused]] OutputContext& op_output,
               [[maybe_unused]] ExecutionContext& context) override {
    // Access data ready device address
    data_ready_addr_ = data_ready_device_address();

    // Access data ready handler CUDA stream
    drh_stream_ = data_ready_handler_cuda_stream();

    compute_called_ = true;
  }

  void* data_ready_addr_ = nullptr;
  std::shared_ptr<cudaStream_t> drh_stream_ = nullptr;
  bool compute_called_ = false;
};

// Test fragment for data ready handler
class TestDataReadyHandlerFragment : public Fragment {
 public:
  void compose() override {
    auto drh_op = make_operator<TestDataReadyHandlerOp>("drh_op");
    add_operator(drh_op);
  }
};

// Test fragment for main workload
class TestMainWorkloadFragment : public Application {
 public:
  void compose() override {
    auto source = make_operator<TestSourceGpuOp>("source");
    auto sink = make_operator<TestSinkGpuOp>("sink");
    add_flow(source, sink);
  }
};

// Test fixture for data ready handler tests
class GPUResidentDataReadyHandlerTest : public ::testing::Test {
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
// Data Ready Handler Registration Tests
// ================================================================================================

// Test registering a data ready handler fragment
TEST_F(GPUResidentDataReadyHandlerTest, TestRegisterDataReadyHandler) {
  auto main_fragment = std::make_shared<TestMainWorkloadFragment>();
  auto drh_fragment = std::make_shared<TestDataReadyHandlerFragment>();

  main_fragment->compose();

  // Should not throw when registering data ready handler
  EXPECT_NO_THROW(main_fragment->gpu_resident().register_data_ready_handler(drh_fragment));
}

// Test registering nullptr as data ready handler (should throw)
TEST_F(GPUResidentDataReadyHandlerTest, TestRegisterNullDataReadyHandler) {
  auto main_fragment = std::make_shared<TestMainWorkloadFragment>();
  main_fragment->compose();

  // Registering nullptr should throw std::runtime_error
  EXPECT_THROW(main_fragment->gpu_resident().register_data_ready_handler(nullptr),
               std::runtime_error);
}

// Test registering a non-GPU-resident fragment as data ready handler (should throw)
TEST_F(GPUResidentDataReadyHandlerTest, TestRegisterNonGPUResidentDataReadyHandler) {
  auto main_fragment = std::make_shared<TestMainWorkloadFragment>();
  main_fragment->compose();

  // Create a non-GPU-resident fragment (empty fragment with no operators)
  auto non_gpu_fragment = std::make_shared<Fragment>();

  // Registering non-GPU-resident fragment should throw std::runtime_error
  EXPECT_THROW(main_fragment->gpu_resident().register_data_ready_handler(non_gpu_fragment),
               std::runtime_error);
}

// Test getting data ready handler fragment
TEST_F(GPUResidentDataReadyHandlerTest, TestGetDataReadyHandlerFragment) {
  auto main_fragment = std::make_shared<TestMainWorkloadFragment>();
  auto drh_fragment = std::make_shared<TestDataReadyHandlerFragment>();

  main_fragment->compose();

  // Initially should be nullptr (before registration)
  EXPECT_EQ(main_fragment->gpu_resident().data_ready_handler_fragment(), nullptr);

  // Register data ready handler
  main_fragment->gpu_resident().register_data_ready_handler(drh_fragment);

  // Now should return the registered fragment
  EXPECT_EQ(main_fragment->gpu_resident().data_ready_handler_fragment(), drh_fragment);
}

// Test data ready handler gets the same executor as main fragment
TEST_F(GPUResidentDataReadyHandlerTest, TestDataReadyHandlerSharesExecutor) {
  auto main_fragment = std::make_shared<TestMainWorkloadFragment>();
  auto drh_fragment = std::make_shared<TestDataReadyHandlerFragment>();

  main_fragment->compose();

  // Register data ready handler
  main_fragment->gpu_resident().register_data_ready_handler(drh_fragment);

  // Data ready handler fragment should have the same executor as main fragment
  EXPECT_EQ(drh_fragment->executor_shared(), main_fragment->executor_shared());
  EXPECT_NE(drh_fragment->executor_shared(), nullptr);
}

// ================================================================================================
// Data Ready Handler Capture Stream Tests
// ================================================================================================

// Test that data ready handler capture stream is created
TEST_F(GPUResidentDataReadyHandlerTest, TestDataReadyHandlerCaptureStream) {
  auto main_fragment = std::make_shared<TestMainWorkloadFragment>();
  auto drh_fragment = std::make_shared<TestDataReadyHandlerFragment>();

  main_fragment->compose();

  // Register data ready handler
  main_fragment->gpu_resident().register_data_ready_handler(drh_fragment);

  // Get the operator from data ready handler fragment
  drh_fragment->compose_graph();
  auto graph = drh_fragment->graph_shared();
  auto op_node = graph->find_node("drh_op");
  ASSERT_NE(op_node, nullptr);

  auto drh_op = std::dynamic_pointer_cast<TestDataReadyHandlerOp>(op_node);
  ASSERT_NE(drh_op, nullptr);

  // Get data ready handler capture stream through operator API
  auto stream = drh_op->data_ready_handler_cuda_stream();
  ASSERT_NE(stream, nullptr);

  // Verify the stream is valid
  EXPECT_NE(*stream, nullptr);
}

// Test that data ready handler capture stream is different from main capture stream
TEST_F(GPUResidentDataReadyHandlerTest, TestSeparateCaptureStreams) {
  auto main_fragment = std::make_shared<TestMainWorkloadFragment>();
  auto drh_fragment = std::make_shared<TestDataReadyHandlerFragment>();

  main_fragment->compose();

  // Register data ready handler
  main_fragment->gpu_resident().register_data_ready_handler(drh_fragment);

  // Get operator from main fragment
  auto main_graph = main_fragment->graph_shared();
  auto main_op_node = main_graph->find_node("source");
  ASSERT_NE(main_op_node, nullptr);
  auto main_op = std::dynamic_pointer_cast<TestSourceGpuOp>(main_op_node);
  ASSERT_NE(main_op, nullptr);

  // Get operator from data ready handler fragment
  drh_fragment->compose_graph();
  auto drh_graph = drh_fragment->graph_shared();
  auto drh_op_node = drh_graph->find_node("drh_op");
  ASSERT_NE(drh_op_node, nullptr);
  auto drh_op = std::dynamic_pointer_cast<TestDataReadyHandlerOp>(drh_op_node);
  ASSERT_NE(drh_op, nullptr);

  // Get streams through operator APIs
  auto main_stream = main_op->cuda_stream();
  auto drh_stream = drh_op->data_ready_handler_cuda_stream();

  ASSERT_NE(main_stream, nullptr);
  ASSERT_NE(drh_stream, nullptr);

  // The two streams should be different
  EXPECT_NE(*main_stream, *drh_stream);
}

// ================================================================================================
// Data Ready Device Address Tests
// ================================================================================================

// Test that data ready device address is accessible
TEST_F(GPUResidentDataReadyHandlerTest, TestDataReadyDeviceAddress) {
  auto main_fragment = std::make_shared<TestMainWorkloadFragment>();
  main_fragment->compose();

  auto addr = main_fragment->gpu_resident().data_ready_device_address();

  // Should return a valid device address
  EXPECT_NE(addr, nullptr);
}

// Test that result ready device address is accessible
TEST_F(GPUResidentDataReadyHandlerTest, TestResultReadyDeviceAddress) {
  auto main_fragment = std::make_shared<TestMainWorkloadFragment>();
  main_fragment->compose();

  auto addr = main_fragment->gpu_resident().result_ready_device_address();

  // Should return a valid device address
  EXPECT_NE(addr, nullptr);
}

// Test that tear down device address is accessible
TEST_F(GPUResidentDataReadyHandlerTest, TestTearDownDeviceAddress) {
  auto main_fragment = std::make_shared<TestMainWorkloadFragment>();
  main_fragment->compose();

  auto addr = main_fragment->gpu_resident().tear_down_device_address();

  // Should return a valid device address
  EXPECT_NE(addr, nullptr);
}

// Test that all control signal device addresses are different
TEST_F(GPUResidentDataReadyHandlerTest, TestControlSignalAddressesAreDifferent) {
  auto main_fragment = std::make_shared<TestMainWorkloadFragment>();
  main_fragment->compose();

  auto data_ready_addr = main_fragment->gpu_resident().data_ready_device_address();
  auto result_ready_addr = main_fragment->gpu_resident().result_ready_device_address();
  auto tear_down_addr = main_fragment->gpu_resident().tear_down_device_address();

  // All addresses should be different
  EXPECT_NE(data_ready_addr, result_ready_addr);
  EXPECT_NE(data_ready_addr, tear_down_addr);
  EXPECT_NE(result_ready_addr, tear_down_addr);
}

// ================================================================================================
// Data Ready Handler Initialization Tests
// ================================================================================================

// Test that data ready handler fragment is composed
TEST_F(GPUResidentDataReadyHandlerTest, TestDataReadyHandlerGraphCompose) {
  auto main_fragment = std::make_shared<TestMainWorkloadFragment>();
  auto drh_fragment = std::make_shared<TestDataReadyHandlerFragment>();

  main_fragment->compose();

  // Register data ready handler before initialization
  main_fragment->gpu_resident().register_data_ready_handler(drh_fragment);

  // Verify data ready handler fragment was composed
  EXPECT_NE(drh_fragment->graph_shared(), nullptr);
}

// Test operator access to data ready device address during compute
TEST_F(GPUResidentDataReadyHandlerTest, TestOperatorDataReadyDeviceAddress) {
  // Create a fragment with TestDataReadyHandlerOp
  class TestFragment : public Fragment {
   public:
    void compose() override {
      auto drh_op = make_operator<TestDataReadyHandlerOp>("drh_op");
      add_operator(drh_op);
    }
  };

  auto fragment = std::make_shared<TestFragment>();
  fragment->compose();

  // Get the operator and verify it can access data ready address
  auto graph = fragment->graph_shared();
  auto op_node = graph->find_node("drh_op");
  ASSERT_NE(op_node, nullptr);

  auto drh_op = std::dynamic_pointer_cast<TestDataReadyHandlerOp>(op_node);
  ASSERT_NE(drh_op, nullptr);

  // The operator should have access to data ready device address
  EXPECT_NO_THROW(drh_op->data_ready_device_address());
  auto addr = drh_op->data_ready_device_address();
  EXPECT_NE(addr, nullptr);
}

// ================================================================================================
// Data Ready Handler Graph Topology Tests
// ================================================================================================

// Test that data ready handler with invalid topology throws during initialization
TEST_F(GPUResidentDataReadyHandlerTest, TestInvalidDataReadyHandlerTopology) {
  // Create a main workload fragment
  auto main_fragment = std::make_shared<TestMainWorkloadFragment>();

  // Create a data ready handler fragment with branching (invalid for GPU-resident)
  class InvalidDRHFragment : public Fragment {
   public:
    void compose() override {
      auto source = make_operator<TestSourceGpuOp>("drh_source");
      auto sink1 = make_operator<TestSinkGpuOp>("drh_sink1");
      auto sink2 = make_operator<TestSinkGpuOp>("drh_sink2");
      add_flow(source, sink1);
      add_flow(source, sink2);  // Branching - not allowed
    }
  };

  auto drh_fragment = std::make_shared<InvalidDRHFragment>();

  // Registration should throw since drh_fragment has invalid topology
  EXPECT_THROW(main_fragment->gpu_resident().register_data_ready_handler(drh_fragment),
               holoscan::RuntimeError);
}

// Test that data ready handler with single operator is valid
TEST_F(GPUResidentDataReadyHandlerTest, TestSingleOperatorDataReadyHandler) {
  auto main_fragment = std::make_shared<TestMainWorkloadFragment>();
  auto drh_fragment = std::make_shared<TestDataReadyHandlerFragment>();

  main_fragment->compose();
  main_fragment->gpu_resident().register_data_ready_handler(drh_fragment);

  auto executor = std::dynamic_pointer_cast<GPUResidentExecutor>(main_fragment->executor_shared());
  ASSERT_NE(executor, nullptr);

  // Initialize should succeed with single operator data ready handler
  EXPECT_TRUE(executor->initialize_fragment());
}

// Test that data ready handler with linear chain is valid
TEST_F(GPUResidentDataReadyHandlerTest, TestLinearChainDataReadyHandler) {
  auto main_fragment = std::make_shared<TestMainWorkloadFragment>();

  // Create a data ready handler with linear chain
  class LinearDRHFragment : public Fragment {
   public:
    void compose() override {
      auto source = make_operator<TestSourceGpuOp>("handler_source");
      auto compute = make_operator<TestComputeGpuOp>("handler_compute");
      auto sink = make_operator<TestSinkGpuOp>("handler_sink");
      add_flow(source, compute);
      add_flow(compute, sink);
    }
  };

  auto drh_fragment = std::make_shared<LinearDRHFragment>();

  main_fragment->compose();
  main_fragment->gpu_resident().register_data_ready_handler(drh_fragment);

  auto executor = std::dynamic_pointer_cast<GPUResidentExecutor>(main_fragment->executor_shared());
  ASSERT_NE(executor, nullptr);

  // Initialize should succeed with linear chain data ready handler
  EXPECT_TRUE(executor->initialize_fragment());
}

// ================================================================================================
// Multiple Registration Tests
// ================================================================================================

// Test registering multiple data ready handlers (last one should be used)
TEST_F(GPUResidentDataReadyHandlerTest, TestMultipleDataReadyHandlerRegistrations) {
  auto main_fragment = std::make_shared<TestMainWorkloadFragment>();
  auto drh_fragment1 = std::make_shared<TestDataReadyHandlerFragment>();
  auto drh_fragment2 = std::make_shared<TestDataReadyHandlerFragment>();

  main_fragment->compose();

  // Register first handler
  main_fragment->gpu_resident().register_data_ready_handler(drh_fragment1);

  EXPECT_EQ(main_fragment->gpu_resident().data_ready_handler_fragment(), drh_fragment1);

  // Register second handler (should replace first)
  main_fragment->gpu_resident().register_data_ready_handler(drh_fragment2);

  EXPECT_EQ(main_fragment->gpu_resident().data_ready_handler_fragment(), drh_fragment2);
}

// ================================================================================================
// Duplicate Operator Name Tests
// ================================================================================================

// Test that duplicate operator names between fragments throw during initialization
TEST_F(GPUResidentDataReadyHandlerTest, TestDuplicateOpNamesBetweenMainAndHandlerFragments) {
  auto main_fragment = std::make_shared<TestMainWorkloadFragment>();

  // Create a data ready handler fragment with one duplicate operator name
  class DuplicateNameDRHFragment : public Fragment {
   public:
    void compose() override {
      // "source" is the same as in TestMainWorkloadFragment - duplicate!
      auto source = make_operator<TestSourceGpuOp>("source");
      // "drh_compute" is different - OK
      auto compute = make_operator<TestComputeGpuOp>("drh_compute");
      add_flow(source, compute);
    }
  };

  auto drh_fragment = std::make_shared<DuplicateNameDRHFragment>();

  main_fragment->compose();
  main_fragment->gpu_resident().register_data_ready_handler(drh_fragment);

  auto executor = std::dynamic_pointer_cast<GPUResidentExecutor>(main_fragment->executor_shared());
  ASSERT_NE(executor, nullptr);

  // Initialize should throw due to duplicate operator name "source"
  EXPECT_THROW(executor->initialize_fragment(), std::runtime_error);
}

// ================================================================================================
// Error Condition Tests
// ================================================================================================

// Test that accessing data ready handler API on non-GPU-resident fragment throws
TEST_F(GPUResidentDataReadyHandlerTest, TestAccessDataReadyHandlerAPIOnNonGPUResidentFragment) {
  auto non_gpu_fragment = std::make_shared<Fragment>();
  auto drh_fragment = std::make_shared<TestDataReadyHandlerFragment>();

  // Trying to register data ready handler on non-GPU-resident fragment should throw
  EXPECT_THROW(non_gpu_fragment->gpu_resident().register_data_ready_handler(drh_fragment),
               holoscan::RuntimeError);
}

// Test registration before main fragment is composed
TEST_F(GPUResidentDataReadyHandlerTest, TestRegisterBeforeCompose) {
  auto main_fragment = std::make_shared<TestMainWorkloadFragment>();
  auto drh_fragment = std::make_shared<TestDataReadyHandlerFragment>();

  // Main fragment is not yet composed, so it's not GPU-resident yet
  // This should throw when trying to access gpu_resident()
  EXPECT_THROW(main_fragment->gpu_resident().register_data_ready_handler(drh_fragment),
               holoscan::RuntimeError);
}

// Test that data ready handler is properly initialized with main workload
TEST_F(GPUResidentDataReadyHandlerTest, TestDataReadyHandlerProperInitialization) {
  auto main_fragment = std::make_shared<TestMainWorkloadFragment>();
  auto drh_fragment = std::make_shared<TestDataReadyHandlerFragment>();

  main_fragment->compose();
  main_fragment->gpu_resident().register_data_ready_handler(drh_fragment);

  // Before initialization, data ready handler fragment should be set
  EXPECT_EQ(main_fragment->gpu_resident().data_ready_handler_fragment(), drh_fragment);

  auto executor = std::dynamic_pointer_cast<GPUResidentExecutor>(main_fragment->executor_shared());
  ASSERT_NE(executor, nullptr);

  // Initialize the fragment
  EXPECT_TRUE(executor->initialize_fragment());

  // After initialization, data ready handler should still be set
  EXPECT_EQ(main_fragment->gpu_resident().data_ready_handler_fragment(), drh_fragment);

  // Verify data ready handler fragment's graph is composed
  EXPECT_NE(drh_fragment->graph_shared(), nullptr);
}

// Test that all control signal addresses remain consistent
TEST_F(GPUResidentDataReadyHandlerTest, TestControlSignalAddressConsistency) {
  auto main_fragment = std::make_shared<TestMainWorkloadFragment>();
  main_fragment->compose();

  // Get addresses multiple times through fragment API
  // No initialization needed - allocated when executor is created
  auto data_ready_addr1 = main_fragment->gpu_resident().data_ready_device_address();
  auto data_ready_addr2 = main_fragment->gpu_resident().data_ready_device_address();
  auto result_ready_addr1 = main_fragment->gpu_resident().result_ready_device_address();
  auto result_ready_addr2 = main_fragment->gpu_resident().result_ready_device_address();
  auto tear_down_addr1 = main_fragment->gpu_resident().tear_down_device_address();
  auto tear_down_addr2 = main_fragment->gpu_resident().tear_down_device_address();

  // Addresses should be consistent across multiple calls
  EXPECT_EQ(data_ready_addr1, data_ready_addr2);
  EXPECT_EQ(result_ready_addr1, result_ready_addr2);
  EXPECT_EQ(tear_down_addr1, tear_down_addr2);
}

// Test multiple initialization calls and verify debug messages
TEST_F(GPUResidentDataReadyHandlerTest, TestMultipleInitializationCalls) {
  EnvVarWrapper wrapper("HOLOSCAN_LOG_LEVEL", "DEBUG");

  auto main_fragment = holoscan::make_application<TestMainWorkloadFragment>();
  auto drh_fragment = std::make_shared<TestDataReadyHandlerFragment>();

  main_fragment->compose();
  main_fragment->gpu_resident().register_data_ready_handler(drh_fragment);

  auto executor = std::dynamic_pointer_cast<GPUResidentExecutor>(main_fragment->executor_shared());
  ASSERT_NE(executor, nullptr);

  // Capture log output
  testing::internal::CaptureStderr();

  // First initialization - initializes both main and data ready handler fragments
  EXPECT_TRUE(executor->initialize_fragment());

  // Second initialization - should return early with "already initialized" debug message
  EXPECT_TRUE(executor->initialize_fragment());

  std::string output = testing::internal::GetCapturedStderr();

  // Verify the "already initialized" debug messages are present
  EXPECT_TRUE(output.find("Main workload fragment") != std::string::npos &&
              output.find("has already been initialized") != std::string::npos)
      << "Expected debug message about main fragment already initialized not found in output:\n"
      << output;

  EXPECT_TRUE(output.find("Data ready handler fragment") != std::string::npos &&
              output.find("has already been initialized") != std::string::npos)
      << "Expected debug message about data ready handler fragment already initialized not found "
         "in output:\n"
      << output;
}

}  // namespace holoscan
