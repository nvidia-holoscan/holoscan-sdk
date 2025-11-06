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

#ifndef HOLOSCAN_TESTS_GPU_RESIDENT_TEST_OPERATORS_HPP
#define HOLOSCAN_TESTS_GPU_RESIDENT_TEST_OPERATORS_HPP

#include <holoscan/holoscan.hpp>
#include <holoscan/core/gpu_resident_operator.hpp>

namespace holoscan {

// Test GPU-resident operator that only has output port (source)
class TestSourceGpuOp : public GPUResidentOperator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS_SUPER(TestSourceGpuOp, GPUResidentOperator)
  TestSourceGpuOp() = default;

  void setup(OperatorSpec& spec) override { spec.device_output("out", sizeof(int) * 128); }

  void compute([[maybe_unused]] InputContext& op_input, [[maybe_unused]] OutputContext& op_output,
               [[maybe_unused]] ExecutionContext& context) override {}
};

// Test GPU-resident operator with both input and output ports (compute)
class TestComputeGpuOp : public GPUResidentOperator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS_SUPER(TestComputeGpuOp, GPUResidentOperator)
  TestComputeGpuOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.device_input("in", sizeof(int) * 128);
    spec.device_output("out", sizeof(int) * 128);
  }

  void compute([[maybe_unused]] InputContext& op_input, [[maybe_unused]] OutputContext& op_output,
               [[maybe_unused]] ExecutionContext& context) override {
    // Test device_memory API
    auto in_addr = device_memory("in");
    auto out_addr = device_memory("out");

    in_device_address_ = in_addr;
    out_device_address_ = out_addr;
  }

  void* in_device_address_ = nullptr;
  void* out_device_address_ = nullptr;
};

// Test GPU-resident operator that only has input port (sink)
class TestSinkGpuOp : public GPUResidentOperator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS_SUPER(TestSinkGpuOp, GPUResidentOperator)
  TestSinkGpuOp() = default;

  void setup(OperatorSpec& spec) override { spec.device_input("in", sizeof(int) * 128); }

  void compute([[maybe_unused]] InputContext& op_input, [[maybe_unused]] OutputContext& op_output,
               [[maybe_unused]] ExecutionContext& context) override {}
};

// Test operator with multiple device inputs
class TestMultiInputGpuOp : public GPUResidentOperator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS_SUPER(TestMultiInputGpuOp, GPUResidentOperator)
  TestMultiInputGpuOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.device_input("in1", sizeof(int) * 64);
    spec.device_input("in2", sizeof(float) * 32);
    spec.device_output("out", sizeof(double) * 128);
  }

  void compute([[maybe_unused]] InputContext& op_input, [[maybe_unused]] OutputContext& op_output,
               [[maybe_unused]] ExecutionContext& context) override {}
};

// Test operator with zero-size memory block (for error testing)
class ZeroSizeMemoryOp : public GPUResidentOperator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS_SUPER(ZeroSizeMemoryOp, GPUResidentOperator)
  ZeroSizeMemoryOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.device_output("out", 0);  // Zero-size memory block - should throw
  }

  void compute([[maybe_unused]] InputContext& op_input, [[maybe_unused]] OutputContext& op_output,
               [[maybe_unused]] ExecutionContext& context) override {}
};

// Test operator with invalid port name (containing dot)
class InvalidPortNameOp : public GPUResidentOperator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS_SUPER(InvalidPortNameOp, GPUResidentOperator)
  InvalidPortNameOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.device_output("out.port", sizeof(int) * 32);  // Port name with dot - should throw
  }

  void compute([[maybe_unused]] InputContext& op_input, [[maybe_unused]] OutputContext& op_output,
               [[maybe_unused]] ExecutionContext& context) override {}
};

}  // namespace holoscan

#endif  // HOLOSCAN_TESTS_GPU_RESIDENT_TEST_OPERATORS_HPP
