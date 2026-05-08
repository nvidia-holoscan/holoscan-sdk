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

#include <gtest/gtest.h>

#include <memory>
#include <string>

#include <holoscan/core/executors/gpu_resident/gpu_resident_executor.hpp>
#include <holoscan/core/gpu_resident_operator.hpp>
#include <holoscan/holoscan.hpp>

#include "../system/env_wrapper.hpp"
#include "test_operators.hpp"

/// This file tests the prepare_data_flow algorithm that decides between
/// memory-block allocation and device-pointer connection for inter-operator
/// data flow in GPU-resident graph execution.
///
/// The algorithm (implemented in GPUResidentExecutor::prepare_data_flow):
///   Case 1: Both sides have memory_block_size > 0  -->  allocate shared buffer
///   Case 2: One side has device_ptr, other has nothing or memory_block_size
///           -->  use device_ptr (warn if other has memory_block_size)
///   Case 3: Both sides have device_ptr  -->  LOG ERROR, use source device_ptr
///   Case 4: Only one side has memory_block_size > 0  -->  allocate, warn
///   Fallback: Neither has anything  -->  throw

namespace holoscan {

// ============================================================================
// Test fixture
// ============================================================================

class DevicePtrConnectionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    int device_count = 0;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    if (error != cudaSuccess || device_count == 0) {
      GTEST_SKIP() << "No CUDA devices available, skipping GPU-resident tests";
    }
  }
};

// ============================================================================
// Case 1: Both sides have memory_block_size
// ============================================================================

// Both source and destination have matching memory_block_size --> allocate
TEST_F(DevicePtrConnectionTest, BothMemoryBlockSize_Matching) {
  Fragment fragment;
  auto source = fragment.make_operator<TestSourceGpuOp>("source");
  auto sink = fragment.make_operator<TestSinkGpuOp>("sink");
  fragment.add_flow(source, sink);

  auto executor = std::dynamic_pointer_cast<GPUResidentExecutor>(fragment.executor_shared());
  ASSERT_NE(executor, nullptr);
  EXPECT_TRUE(executor->initialize_fragment());

  // Both ports should resolve to the same executor-allocated buffer
  auto out_addr = source->device_memory("out");
  auto in_addr = sink->device_memory("in");
  EXPECT_NE(out_addr, nullptr);
  EXPECT_NE(in_addr, nullptr);
  EXPECT_EQ(out_addr, in_addr);
}

// Both have memory_block_size but sizes differ --> throw
TEST_F(DevicePtrConnectionTest, BothMemoryBlockSize_Mismatched) {
  Fragment fragment;
  auto source = fragment.make_operator<MismatchedSizeSourceOp>("source");  // 256 ints
  auto sink = fragment.make_operator<TestSinkGpuOp>("sink");               // 128 ints
  fragment.add_flow(source, sink);

  auto executor = std::dynamic_pointer_cast<GPUResidentExecutor>(fragment.executor_shared());
  ASSERT_NE(executor, nullptr);
  EXPECT_THROW(executor->initialize_fragment(), std::runtime_error);
}

// ============================================================================
// Case 2: One side has device_ptr
// ============================================================================

// Source has device_ptr, sink has nothing --> use source device_ptr
TEST_F(DevicePtrConnectionTest, SourceDevicePtr_SinkNothing) {
  Fragment fragment;
  auto source = fragment.make_operator<DevicePtrSourceOp>("source");
  auto sink = fragment.make_operator<ZeroSizeInputMemoryOp>("sink");
  fragment.add_flow(source, sink);

  auto executor = std::dynamic_pointer_cast<GPUResidentExecutor>(fragment.executor_shared());
  ASSERT_NE(executor, nullptr);
  EXPECT_TRUE(executor->initialize_fragment());

  // Both ports should resolve to the source's device pointer
  auto out_addr = source->device_memory("out");
  auto in_addr = sink->device_memory("in");
  EXPECT_NE(out_addr, nullptr);
  EXPECT_EQ(out_addr, source->dev_ptr());
  EXPECT_EQ(out_addr, in_addr);
}

// Sink has device_ptr, source has nothing --> use sink device_ptr
TEST_F(DevicePtrConnectionTest, SinkDevicePtr_SourceNothing) {
  Fragment fragment;
  auto source = fragment.make_operator<ZeroSizeOutputMemoryOp>("source");
  auto sink = fragment.make_operator<DevicePtrSinkOp>("sink");
  fragment.add_flow(source, sink);

  auto executor = std::dynamic_pointer_cast<GPUResidentExecutor>(fragment.executor_shared());
  ASSERT_NE(executor, nullptr);
  EXPECT_TRUE(executor->initialize_fragment());

  auto out_addr = source->device_memory("out");
  auto in_addr = sink->device_memory("in");
  EXPECT_NE(in_addr, nullptr);
  EXPECT_EQ(in_addr, sink->dev_ptr());
  EXPECT_EQ(out_addr, in_addr);
}

// Source has device_ptr, sink has memory_block_size --> use device_ptr, warn
TEST_F(DevicePtrConnectionTest, SourceDevicePtr_SinkMemoryBlockSize_Warns) {
  EnvVarWrapper wrapper("HOLOSCAN_LOG_LEVEL", "WARN");
  Fragment fragment;
  auto source = fragment.make_operator<DevicePtrSourceOp>("source");
  auto sink = fragment.make_operator<TestSinkGpuOp>("sink");
  fragment.add_flow(source, sink);

  auto executor = std::dynamic_pointer_cast<GPUResidentExecutor>(fragment.executor_shared());
  ASSERT_NE(executor, nullptr);

  testing::internal::CaptureStderr();
  EXPECT_TRUE(executor->initialize_fragment());
  std::string log_output = testing::internal::GetCapturedStderr();

  // Should warn about ignoring memory block size
  EXPECT_TRUE(log_output.find("ignoring the memory block size") != std::string::npos)
      << "Expected warning about ignoring memory block size not found in:\n"
      << log_output;

  // Both ports should resolve to the source's device pointer
  auto out_addr = source->device_memory("out");
  auto in_addr = sink->device_memory("in");
  EXPECT_EQ(out_addr, source->dev_ptr());
  EXPECT_EQ(out_addr, in_addr);
}

// Sink has device_ptr, source has memory_block_size --> use device_ptr, warn
TEST_F(DevicePtrConnectionTest, SinkDevicePtr_SourceMemoryBlockSize_Warns) {
  EnvVarWrapper wrapper("HOLOSCAN_LOG_LEVEL", "WARN");
  Fragment fragment;
  auto source = fragment.make_operator<TestSourceGpuOp>("source");
  auto sink = fragment.make_operator<DevicePtrSinkOp>("sink");
  fragment.add_flow(source, sink);

  auto executor = std::dynamic_pointer_cast<GPUResidentExecutor>(fragment.executor_shared());
  ASSERT_NE(executor, nullptr);

  testing::internal::CaptureStderr();
  EXPECT_TRUE(executor->initialize_fragment());
  std::string log_output = testing::internal::GetCapturedStderr();

  EXPECT_TRUE(log_output.find("ignoring the memory block size") != std::string::npos)
      << "Expected warning about ignoring memory block size not found in:\n"
      << log_output;

  auto out_addr = source->device_memory("out");
  auto in_addr = sink->device_memory("in");
  EXPECT_EQ(in_addr, sink->dev_ptr());
  EXPECT_EQ(out_addr, in_addr);
}

// ============================================================================
// Case 3: Both sides have device_ptr
// ============================================================================

// Both have device_ptr --> LOG ERROR, use source device_ptr
TEST_F(DevicePtrConnectionTest, BothDevicePtr_UsesSource) {
  EnvVarWrapper wrapper("HOLOSCAN_LOG_LEVEL", "ERROR");
  Fragment fragment;
  auto source = fragment.make_operator<DevicePtrSourceOp>("source");
  auto sink = fragment.make_operator<DevicePtrSinkOp>("sink");
  fragment.add_flow(source, sink);

  auto executor = std::dynamic_pointer_cast<GPUResidentExecutor>(fragment.executor_shared());
  ASSERT_NE(executor, nullptr);

  testing::internal::CaptureStderr();
  EXPECT_TRUE(executor->initialize_fragment());
  std::string log_output = testing::internal::GetCapturedStderr();

  // Should log an error about both sides having device pointers
  EXPECT_TRUE(log_output.find("Both source") != std::string::npos &&
              log_output.find("have device pointers") != std::string::npos)
      << "Expected error about both sides having device pointers not found in:\n"
      << log_output;

  // Both ports should resolve to the *source's* device pointer
  auto out_addr = source->device_memory("out");
  auto in_addr = sink->device_memory("in");
  EXPECT_EQ(out_addr, source->dev_ptr());
  EXPECT_EQ(out_addr, in_addr);

  // The sink's own device pointer is NOT used
  EXPECT_NE(in_addr, sink->dev_ptr());
}

// ============================================================================
// Case 4: Only one side has memory_block_size, the other has nothing
// ============================================================================

// Source has memory_block_size, sink has nothing --> allocate, warn
TEST_F(DevicePtrConnectionTest, SourceMemBlockOnly_SinkNothing_Warns) {
  EnvVarWrapper wrapper("HOLOSCAN_LOG_LEVEL", "WARN");
  Fragment fragment;
  auto source = fragment.make_operator<TestSourceGpuOp>("source");
  auto sink = fragment.make_operator<ZeroSizeInputMemoryOp>("sink");
  fragment.add_flow(source, sink);

  auto executor = std::dynamic_pointer_cast<GPUResidentExecutor>(fragment.executor_shared());
  ASSERT_NE(executor, nullptr);

  testing::internal::CaptureStderr();
  EXPECT_TRUE(executor->initialize_fragment());
  std::string log_output = testing::internal::GetCapturedStderr();

  // Should warn that the other operator has neither memory block size nor device pointer
  EXPECT_TRUE(log_output.find("has neither a valid memory block size nor a valid device pointer") !=
              std::string::npos)
      << "Expected warning not found in:\n"
      << log_output;

  // Both ports should still get a valid (executor-allocated) address
  auto out_addr = source->device_memory("out");
  auto in_addr = sink->device_memory("in");
  EXPECT_NE(out_addr, nullptr);
  EXPECT_NE(in_addr, nullptr);
  EXPECT_EQ(out_addr, in_addr);
}

// Sink has memory_block_size, source has nothing --> allocate, warn
TEST_F(DevicePtrConnectionTest, SinkMemBlockOnly_SourceNothing_Warns) {
  EnvVarWrapper wrapper("HOLOSCAN_LOG_LEVEL", "WARN");
  Fragment fragment;
  auto source = fragment.make_operator<ZeroSizeOutputMemoryOp>("source");
  auto sink = fragment.make_operator<TestSinkGpuOp>("sink");
  fragment.add_flow(source, sink);

  auto executor = std::dynamic_pointer_cast<GPUResidentExecutor>(fragment.executor_shared());
  ASSERT_NE(executor, nullptr);

  testing::internal::CaptureStderr();
  EXPECT_TRUE(executor->initialize_fragment());
  std::string log_output = testing::internal::GetCapturedStderr();

  EXPECT_TRUE(log_output.find("has neither a valid memory block size nor a valid device pointer") !=
              std::string::npos)
      << "Expected warning not found in:\n"
      << log_output;

  auto out_addr = source->device_memory("out");
  auto in_addr = sink->device_memory("in");
  EXPECT_NE(out_addr, nullptr);
  EXPECT_NE(in_addr, nullptr);
  EXPECT_EQ(out_addr, in_addr);
}

// ============================================================================
// Invalid device pointer types (not cudaMemoryTypeDevice)
// ============================================================================

// cudaHostAlloc (pinned host memory) is not valid as a device pointer --> throw
TEST_F(DevicePtrConnectionTest, InvalidDevicePtr_HostAlloc_Throws) {
  Fragment fragment;
  auto source = fragment.make_operator<HostAllocSourceOp>("source");
  auto sink = fragment.make_operator<TestSinkGpuOp>("sink");
  fragment.add_flow(source, sink);

  auto executor = std::dynamic_pointer_cast<GPUResidentExecutor>(fragment.executor_shared());
  ASSERT_NE(executor, nullptr);

  std::string msg;
  try {
    executor->initialize_fragment();
  } catch (const std::runtime_error& e) {
    msg = e.what();
  }
  EXPECT_FALSE(msg.empty()) << "Expected std::runtime_error to be thrown";
  EXPECT_TRUE(msg.find("Not a valid device memory pointer") != std::string::npos)
      << "Expected message about invalid device pointer, got: " << msg;
}

// cudaMallocManaged (unified/managed memory) is not valid as a device pointer --> throw
TEST_F(DevicePtrConnectionTest, InvalidDevicePtr_ManagedMemory_Throws) {
  Fragment fragment;
  auto source = fragment.make_operator<ManagedAllocSourceOp>("source");
  auto sink = fragment.make_operator<TestSinkGpuOp>("sink");
  fragment.add_flow(source, sink);

  auto executor = std::dynamic_pointer_cast<GPUResidentExecutor>(fragment.executor_shared());
  ASSERT_NE(executor, nullptr);

  std::string msg;
  try {
    executor->initialize_fragment();
  } catch (const std::runtime_error& e) {
    msg = e.what();
  }
  EXPECT_FALSE(msg.empty()) << "Expected std::runtime_error to be thrown";
  EXPECT_TRUE(msg.find("Not a valid device memory pointer") != std::string::npos)
      << "Expected message about invalid device pointer, got: " << msg;
}

// Sink with cudaHostAlloc (input port) is not valid as device pointer --> throw
TEST_F(DevicePtrConnectionTest, InvalidDevicePtr_SinkHostAlloc_Throws) {
  Fragment fragment;
  auto source = fragment.make_operator<TestSourceGpuOp>("source");
  auto sink = fragment.make_operator<HostAllocSinkOp>("sink");
  fragment.add_flow(source, sink);

  auto executor = std::dynamic_pointer_cast<GPUResidentExecutor>(fragment.executor_shared());
  ASSERT_NE(executor, nullptr);

  std::string msg;
  try {
    executor->initialize_fragment();
  } catch (const std::runtime_error& e) {
    msg = e.what();
  }
  EXPECT_FALSE(msg.empty()) << "Expected std::runtime_error to be thrown";
  EXPECT_TRUE(msg.find("Not a valid device memory pointer") != std::string::npos)
      << "Expected message about invalid device pointer, got: " << msg;
}

// ============================================================================
// Fallback: Neither side has memory_block_size or device_ptr
// ============================================================================

// Both have zero memory_block_size and null device_ptr --> throw
TEST_F(DevicePtrConnectionTest, NeitherSideHasAnything_Throws) {
  Fragment fragment;
  auto source = fragment.make_operator<ZeroSizeOutputMemoryOp>("source");
  auto sink = fragment.make_operator<ZeroSizeInputMemoryOp>("sink");
  fragment.add_flow(source, sink);

  auto executor = std::dynamic_pointer_cast<GPUResidentExecutor>(fragment.executor_shared());
  ASSERT_NE(executor, nullptr);

  // initialize_fragment should throw because neither side can provide
  // a memory block size or device pointer
  EXPECT_THROW(executor->initialize_fragment(), std::runtime_error);
}

// ============================================================================
// Multi-hop chain with mixed connection types
// ============================================================================

// Chain: MemBlock source --> DevicePtr compute --> MemBlock sink
// Tests that each connection independently selects the correct strategy
TEST_F(DevicePtrConnectionTest, MixedChain_MemBlock_DevicePtr_MemBlock) {
  EnvVarWrapper wrapper("HOLOSCAN_LOG_LEVEL", "WARN");
  Fragment fragment;
  auto source = fragment.make_operator<TestSourceGpuOp>("source");
  auto compute = fragment.make_operator<DevicePtrComputeOp>("compute");
  auto sink = fragment.make_operator<TestSinkGpuOp>("sink");
  fragment.add_flow(source, compute);
  fragment.add_flow(compute, sink);

  auto executor = std::dynamic_pointer_cast<GPUResidentExecutor>(fragment.executor_shared());
  ASSERT_NE(executor, nullptr);

  testing::internal::CaptureStderr();
  EXPECT_TRUE(executor->initialize_fragment());
  std::string log_output = testing::internal::GetCapturedStderr();

  // Connection source->compute: source has mem_block, compute input has device_ptr
  // --> Case 2: use device_ptr, warn about ignoring mem_block
  // Connection compute->sink: compute output has device_ptr, sink has mem_block
  // --> Case 2: use device_ptr, warn about ignoring mem_block
  EXPECT_TRUE(log_output.find("ignoring the memory block size") != std::string::npos)
      << "Expected warning about ignoring memory block size not found in:\n"
      << log_output;

  // source->compute connection uses compute's input device_ptr
  auto source_out = source->device_memory("out");
  auto compute_in = compute->device_memory("in");
  EXPECT_NE(source_out, nullptr);
  EXPECT_EQ(source_out, compute_in);
  EXPECT_EQ(compute_in, compute->in_dev_ptr());

  // compute->sink connection uses compute's output device_ptr
  auto compute_out = compute->device_memory("out");
  auto sink_in = sink->device_memory("in");
  EXPECT_NE(compute_out, nullptr);
  EXPECT_EQ(compute_out, sink_in);
  EXPECT_EQ(compute_out, compute->out_dev_ptr());

  // The two connections should use different memory
  EXPECT_NE(source_out, compute_out);
}

// ============================================================================
// Fan-out to multiple inputs on the same downstream operator
// ============================================================================

// Source has device_ptr and fans out in a single add_flow call to one mem input and one device_ptr
// input on the same sink operator. The source device_ptr must win for both connections.
TEST_F(DevicePtrConnectionTest, SourceDevicePtr_FanOutSingleCall_SameSinkMixedInputs) {
  EnvVarWrapper wrapper("HOLOSCAN_LOG_LEVEL", "WARN");
  Fragment fragment;
  auto source = fragment.make_operator<DevicePtrSourceOp>("source");
  auto sink = fragment.make_operator<MultiPortMixedSinkOp>("sink");
  fragment.add_flow(source, sink, {{"out", "in0"}, {"out", "in2"}});

  auto executor = std::dynamic_pointer_cast<GPUResidentExecutor>(fragment.executor_shared());
  ASSERT_NE(executor, nullptr);

  testing::internal::CaptureStderr();
  EXPECT_TRUE(executor->initialize_fragment());
  std::string log_output = testing::internal::GetCapturedStderr();

  EXPECT_TRUE(log_output.find("ignoring the memory block size") != std::string::npos)
      << "Expected warning about ignoring memory block size not found in:\n"
      << log_output;
  EXPECT_TRUE(log_output.find("Both source") != std::string::npos &&
              log_output.find("have device pointers") != std::string::npos)
      << "Expected message about both sides having device pointers not found in:\n"
      << log_output;

  auto source_out = source->device_memory("out");
  auto sink_in0 = sink->device_memory("in0");
  auto sink_in2 = sink->device_memory("in2");

  EXPECT_EQ(source_out, source->dev_ptr());
  EXPECT_EQ(source_out, sink_in0);
  EXPECT_EQ(source_out, sink_in2);
  EXPECT_NE(sink_in2, sink->dev_ptr_2());
}

// Same topology as above, but the two fan-out connections are added across separate add_flow calls.
TEST_F(DevicePtrConnectionTest, SourceDevicePtr_FanOutSeparateCalls_SameSinkMixedInputs) {
  EnvVarWrapper wrapper("HOLOSCAN_LOG_LEVEL", "WARN");
  Fragment fragment;
  auto source = fragment.make_operator<DevicePtrSourceOp>("source");
  auto sink = fragment.make_operator<MultiPortMixedSinkOp>("sink");
  fragment.add_flow(source, sink, {{"out", "in0"}});
  fragment.add_flow(source, sink, {{"out", "in2"}});

  auto executor = std::dynamic_pointer_cast<GPUResidentExecutor>(fragment.executor_shared());
  ASSERT_NE(executor, nullptr);

  testing::internal::CaptureStderr();
  EXPECT_TRUE(executor->initialize_fragment());
  std::string log_output = testing::internal::GetCapturedStderr();

  EXPECT_TRUE(log_output.find("ignoring the memory block size") != std::string::npos)
      << "Expected warning about ignoring memory block size not found in:\n"
      << log_output;
  EXPECT_TRUE(log_output.find("Both source") != std::string::npos &&
              log_output.find("have device pointers") != std::string::npos)
      << "Expected message about both sides having device pointers not found in:\n"
      << log_output;

  auto source_out = source->device_memory("out");
  auto sink_in0 = sink->device_memory("in0");
  auto sink_in2 = sink->device_memory("in2");

  EXPECT_EQ(source_out, source->dev_ptr());
  EXPECT_EQ(source_out, sink_in0);
  EXPECT_EQ(source_out, sink_in2);
  EXPECT_NE(sink_in2, sink->dev_ptr_2());
}

// ============================================================================
// Multi-port mixed connection types (3 ports)
// ============================================================================

// Source: out0 = mem_block, out1 = device_ptr, out2 = mem_block
// Sink:  in0  = mem_block, in1  = mem_block,   in2  = device_ptr
//
// Per-port expected behaviour:
//   out0 (mem) -> in0 (mem)   : Case 1 — allocate shared buffer
//   out1 (ptr) -> in1 (mem)   : Case 2 — use source device_ptr, warn
//   out2 (mem) -> in2 (ptr)   : Case 2 — use sink device_ptr, warn
TEST_F(DevicePtrConnectionTest, MultiPort_MixedConnectionTypes) {
  EnvVarWrapper wrapper("HOLOSCAN_LOG_LEVEL", "WARN");
  Fragment fragment;
  auto source = fragment.make_operator<MultiPortMixedSourceOp>("source");
  auto sink = fragment.make_operator<MultiPortMixedSinkOp>("sink");
  fragment.add_flow(source, sink, {{"out0", "in0"}, {"out1", "in1"}, {"out2", "in2"}});

  auto executor = std::dynamic_pointer_cast<GPUResidentExecutor>(fragment.executor_shared());
  ASSERT_NE(executor, nullptr);

  testing::internal::CaptureStderr();
  EXPECT_TRUE(executor->initialize_fragment());
  std::string log_output = testing::internal::GetCapturedStderr();

  // Ports out1->in1 and out2->in2 should each produce a warning about ignoring mem_block
  EXPECT_TRUE(log_output.find("ignoring the memory block size") != std::string::npos)
      << "Expected warning about ignoring memory block size not found in:\n"
      << log_output;

  // --- Port 0: both mem_block → executor-allocated shared buffer ---
  auto out0 = source->device_memory("out0");
  auto in0 = sink->device_memory("in0");
  EXPECT_NE(out0, nullptr);
  EXPECT_NE(in0, nullptr);
  EXPECT_EQ(out0, in0);

  // --- Port 1: source device_ptr wins ---
  auto out1 = source->device_memory("out1");
  auto in1 = sink->device_memory("in1");
  EXPECT_NE(out1, nullptr);
  EXPECT_EQ(out1, source->dev_ptr_1());
  EXPECT_EQ(out1, in1);

  // --- Port 2: sink device_ptr wins ---
  auto out2 = source->device_memory("out2");
  auto in2 = sink->device_memory("in2");
  EXPECT_NE(in2, nullptr);
  EXPECT_EQ(in2, sink->dev_ptr_2());
  EXPECT_EQ(out2, in2);

  // All three connections must use distinct memory
  EXPECT_NE(out0, out1);
  EXPECT_NE(out0, out2);
  EXPECT_NE(out1, out2);
}

}  // namespace holoscan
