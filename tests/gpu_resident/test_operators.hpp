/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cuda_runtime.h>

#include <holoscan/core/gpu_resident_operator.hpp>
#include <holoscan/holoscan.hpp>
#include <holoscan/utils/cuda_macros.hpp>

#include "test_kernels.cuh"

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

// Test GPU-resident operator that performs actual CUDA compute work
class TestCudaWorkGpuOp : public GPUResidentOperator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS_SUPER(TestCudaWorkGpuOp, GPUResidentOperator)
  TestCudaWorkGpuOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.device_input("in", sizeof(int) * 128);
    spec.device_output("out", sizeof(int) * 128);
  }

  void compute([[maybe_unused]] InputContext& op_input, [[maybe_unused]] OutputContext& op_output,
               [[maybe_unused]] ExecutionContext& context) override {
    auto* out_addr = device_memory("out");

    // Perform actual CUDA compute work - launch kernel that adds a value to each element
    if (out_addr != nullptr) {
      auto stream_ptr = cuda_stream();
      cudaStream_t stream = *stream_ptr;
      launch_add_value_kernel(static_cast<int*>(out_addr), 1, 128, stream);
    }
  }
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

// Test GPU-resident operator with two output ports
class TestTwoOutGpuOp : public GPUResidentOperator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS_SUPER(TestTwoOutGpuOp, GPUResidentOperator)
  TestTwoOutGpuOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.device_output("out0", sizeof(int) * 64);
    spec.device_output("out1", sizeof(int) * 64);
  }

  void compute([[maybe_unused]] InputContext& op_input, [[maybe_unused]] OutputContext& op_output,
               [[maybe_unused]] ExecutionContext& context) override {}
};

// GPU-resident operator with a device-pointer output (externally allocated)
class DevicePtrSourceOp : public GPUResidentOperator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS_SUPER(DevicePtrSourceOp, GPUResidentOperator)
  DevicePtrSourceOp() = default;

  ~DevicePtrSourceOp() override {
    if (dev_ptr_) {
      HOLOSCAN_CUDA_CALL_WARN(cudaFree(dev_ptr_));
    }
  }

  void setup(OperatorSpec& spec) override {
    HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaMalloc(&dev_ptr_, alloc_size_),
                                   "Failed to allocate device memory");
    spec.device_output("out", reinterpret_cast<CUdeviceptr>(dev_ptr_));
  }

  void compute([[maybe_unused]] InputContext& op_input, [[maybe_unused]] OutputContext& op_output,
               [[maybe_unused]] ExecutionContext& context) override {}

  void* dev_ptr() const { return dev_ptr_; }

 private:
  void* dev_ptr_ = nullptr;
  size_t alloc_size_ = sizeof(int) * 128;
};

// GPU-resident operator with a device-pointer input (externally allocated)
class DevicePtrSinkOp : public GPUResidentOperator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS_SUPER(DevicePtrSinkOp, GPUResidentOperator)
  DevicePtrSinkOp() = default;

  ~DevicePtrSinkOp() override {
    if (dev_ptr_) {
      HOLOSCAN_CUDA_CALL_WARN(cudaFree(dev_ptr_));
    }
  }

  void setup(OperatorSpec& spec) override {
    HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaMalloc(&dev_ptr_, alloc_size_),
                                   "Failed to allocate device memory");
    spec.device_input("in", reinterpret_cast<CUdeviceptr>(dev_ptr_));
  }

  void compute([[maybe_unused]] InputContext& op_input, [[maybe_unused]] OutputContext& op_output,
               [[maybe_unused]] ExecutionContext& context) override {}

  void* dev_ptr() const { return dev_ptr_; }

 private:
  void* dev_ptr_ = nullptr;
  size_t alloc_size_ = sizeof(int) * 128;
};

// GPU-resident operator with device-pointer input and device-pointer output
class DevicePtrComputeOp : public GPUResidentOperator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS_SUPER(DevicePtrComputeOp, GPUResidentOperator)
  DevicePtrComputeOp() = default;

  ~DevicePtrComputeOp() override {
    if (in_dev_ptr_) {
      HOLOSCAN_CUDA_CALL_WARN(cudaFree(in_dev_ptr_));
    }
    if (out_dev_ptr_) {
      HOLOSCAN_CUDA_CALL_WARN(cudaFree(out_dev_ptr_));
    }
  }

  void setup(OperatorSpec& spec) override {
    HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaMalloc(&in_dev_ptr_, alloc_size_),
                                   "Failed to allocate input device memory");
    HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaMalloc(&out_dev_ptr_, alloc_size_),
                                   "Failed to allocate output device memory");
    spec.device_input("in", reinterpret_cast<CUdeviceptr>(in_dev_ptr_));
    spec.device_output("out", reinterpret_cast<CUdeviceptr>(out_dev_ptr_));
  }

  void compute([[maybe_unused]] InputContext& op_input, [[maybe_unused]] OutputContext& op_output,
               [[maybe_unused]] ExecutionContext& context) override {}

  void* in_dev_ptr() const { return in_dev_ptr_; }
  void* out_dev_ptr() const { return out_dev_ptr_; }

 private:
  void* in_dev_ptr_ = nullptr;
  void* out_dev_ptr_ = nullptr;
  size_t alloc_size_ = sizeof(int) * 128;
};

// GPU-resident source that uses cudaHostAlloc (pinned host memory) - invalid as device pointer
class HostAllocSourceOp : public GPUResidentOperator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS_SUPER(HostAllocSourceOp, GPUResidentOperator)
  HostAllocSourceOp() = default;

  ~HostAllocSourceOp() override {
    if (host_ptr_) {
      HOLOSCAN_CUDA_CALL_WARN(cudaFreeHost(host_ptr_));
    }
  }

  void setup(OperatorSpec& spec) override {
    HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaHostAlloc(&host_ptr_, alloc_size_, cudaHostAllocDefault),
                                   "Failed to allocate pinned host memory");
    spec.device_output("out", reinterpret_cast<CUdeviceptr>(host_ptr_));
  }

  void compute([[maybe_unused]] InputContext& op_input, [[maybe_unused]] OutputContext& op_output,
               [[maybe_unused]] ExecutionContext& context) override {}

 private:
  void* host_ptr_ = nullptr;
  size_t alloc_size_ = sizeof(int) * 128;
};

// GPU-resident source that uses cudaMallocManaged (unified memory) - invalid as device pointer
class ManagedAllocSourceOp : public GPUResidentOperator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS_SUPER(ManagedAllocSourceOp, GPUResidentOperator)
  ManagedAllocSourceOp() = default;

  ~ManagedAllocSourceOp() override {
    if (managed_ptr_) {
      HOLOSCAN_CUDA_CALL_WARN(cudaFree(managed_ptr_));
    }
  }

  void setup(OperatorSpec& spec) override {
    HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaMallocManaged(&managed_ptr_, alloc_size_),
                                   "Failed to allocate managed memory");
    spec.device_output("out", reinterpret_cast<CUdeviceptr>(managed_ptr_));
  }

  void compute([[maybe_unused]] InputContext& op_input, [[maybe_unused]] OutputContext& op_output,
               [[maybe_unused]] ExecutionContext& context) override {}

 private:
  void* managed_ptr_ = nullptr;
  size_t alloc_size_ = sizeof(int) * 128;
};

// GPU-resident sink that uses cudaHostAlloc for input port - invalid as device pointer
class HostAllocSinkOp : public GPUResidentOperator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS_SUPER(HostAllocSinkOp, GPUResidentOperator)
  HostAllocSinkOp() = default;

  ~HostAllocSinkOp() override {
    if (host_ptr_) {
      HOLOSCAN_CUDA_CALL_WARN(cudaFreeHost(host_ptr_));
    }
  }

  void setup(OperatorSpec& spec) override {
    HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaHostAlloc(&host_ptr_, alloc_size_, cudaHostAllocDefault),
                                   "Failed to allocate pinned host memory");
    spec.device_input("in", reinterpret_cast<CUdeviceptr>(host_ptr_));
  }

  void compute([[maybe_unused]] InputContext& op_input, [[maybe_unused]] OutputContext& op_output,
               [[maybe_unused]] ExecutionContext& context) override {}

 private:
  void* host_ptr_ = nullptr;
  size_t alloc_size_ = sizeof(int) * 128;
};

// GPU-resident source with a mismatched memory block size (for error testing)
class MismatchedSizeSourceOp : public GPUResidentOperator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS_SUPER(MismatchedSizeSourceOp, GPUResidentOperator)
  MismatchedSizeSourceOp() = default;

  void setup(OperatorSpec& spec) override { spec.device_output("out", sizeof(int) * 256); }

  void compute([[maybe_unused]] InputContext& op_input, [[maybe_unused]] OutputContext& op_output,
               [[maybe_unused]] ExecutionContext& context) override {}
};

// Test operator with zero-size memory block (for error testing)
class ZeroSizeOutputMemoryOp : public GPUResidentOperator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS_SUPER(ZeroSizeOutputMemoryOp, GPUResidentOperator)
  ZeroSizeOutputMemoryOp() = default;

  void setup(OperatorSpec& spec) override { spec.device_output("out", 0); }

  void compute([[maybe_unused]] InputContext& op_input, [[maybe_unused]] OutputContext& op_output,
               [[maybe_unused]] ExecutionContext& context) override {}
};

class ZeroSizeInputMemoryOp : public GPUResidentOperator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS_SUPER(ZeroSizeInputMemoryOp, GPUResidentOperator)
  ZeroSizeInputMemoryOp() = default;

  void setup(OperatorSpec& spec) override { spec.device_input("in", 0); }

  void compute([[maybe_unused]] InputContext& op_input, [[maybe_unused]] OutputContext& op_output,
               [[maybe_unused]] ExecutionContext& context) override {}
};

// 3-output source: port 0 = memory block, port 1 = device_ptr, port 2 = memory block
class MultiPortMixedSourceOp : public GPUResidentOperator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS_SUPER(MultiPortMixedSourceOp, GPUResidentOperator)
  MultiPortMixedSourceOp() = default;

  ~MultiPortMixedSourceOp() override {
    if (dev_ptr_1_) {
      HOLOSCAN_CUDA_CALL_WARN(cudaFree(dev_ptr_1_));
    }
  }

  void setup(OperatorSpec& spec) override {
    spec.device_output("out0", kPortSize);
    HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaMalloc(&dev_ptr_1_, kPortSize),
                                   "Failed to allocate device memory");
    spec.device_output("out1", reinterpret_cast<CUdeviceptr>(dev_ptr_1_));
    spec.device_output("out2", kPortSize);
  }

  void compute([[maybe_unused]] InputContext& op_input, [[maybe_unused]] OutputContext& op_output,
               [[maybe_unused]] ExecutionContext& context) override {}

  void* dev_ptr_1() const { return dev_ptr_1_; }

 private:
  void* dev_ptr_1_ = nullptr;
  static constexpr size_t kPortSize = sizeof(int) * 128;
};

// 3-input sink: port 0 = memory block, port 1 = memory block, port 2 = device_ptr
class MultiPortMixedSinkOp : public GPUResidentOperator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS_SUPER(MultiPortMixedSinkOp, GPUResidentOperator)
  MultiPortMixedSinkOp() = default;

  ~MultiPortMixedSinkOp() override {
    if (dev_ptr_2_) {
      HOLOSCAN_CUDA_CALL_WARN(cudaFree(dev_ptr_2_));
    }
  }

  void setup(OperatorSpec& spec) override {
    spec.device_input("in0", kPortSize);
    spec.device_input("in1", kPortSize);
    HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaMalloc(&dev_ptr_2_, kPortSize),
                                   "Failed to allocate device memory");
    spec.device_input("in2", reinterpret_cast<CUdeviceptr>(dev_ptr_2_));
  }

  void compute([[maybe_unused]] InputContext& op_input, [[maybe_unused]] OutputContext& op_output,
               [[maybe_unused]] ExecutionContext& context) override {}

  void* dev_ptr_2() const { return dev_ptr_2_; }

 private:
  void* dev_ptr_2_ = nullptr;
  static constexpr size_t kPortSize = sizeof(int) * 128;
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
