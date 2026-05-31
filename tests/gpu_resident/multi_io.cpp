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

#include <chrono>
#include <future>
#include <memory>
#include <set>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <holoscan/core/executors/gpu_resident/gpu_resident_executor.hpp>
#include <holoscan/core/gpu_resident_operator.hpp>
#include <holoscan/holoscan.hpp>
#include <holoscan/utils/cuda_macros.hpp>

#include "multi_io_test_kernels.cuh"
#include "test_operators.hpp"

namespace holoscan {

namespace {
constexpr int kElems = 32;
}  // namespace

// ================================================================================================
// Template GPU-resident operators with configurable port count
// ================================================================================================

/// Source operator: N output ports, each of sizeof(int)*kElems bytes.
/// Kernel writes a deterministic pattern: out_j[i] = j*1000 + i.
template <int N>
class MultiOutputSourceGpuOp : public GPUResidentOperator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_TEMPLATE()
  explicit MultiOutputSourceGpuOp(ArgT&& arg, ArgsT&&... args)
      : GPUResidentOperator(std::forward<ArgT>(arg), std::forward<ArgsT>(args)...) {}
  MultiOutputSourceGpuOp() = default;

  void setup(OperatorSpec& spec) override {
    for (int i = 0; i < N; ++i) {
      spec.device_output("out" + std::to_string(i), sizeof(int) * kElems);
    }
  }

  void compute([[maybe_unused]] InputContext& op_input, [[maybe_unused]] OutputContext& op_output,
               [[maybe_unused]] ExecutionContext& context) override {
    cudaStream_t stream = *cuda_stream();
    for (int i = 0; i < N; ++i) {
      auto* ptr = static_cast<int*>(device_memory("out" + std::to_string(i)));
      if (ptr) {
        launch_init_pattern_kernel(ptr, i, kElems, stream);
      }
    }
  }
};

/// Compute operator: N input ports and N output ports.
/// Kernel copies each input to the corresponding output, adding 1.
template <int N>
class MultiIOComputeGpuOp : public GPUResidentOperator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_TEMPLATE()
  explicit MultiIOComputeGpuOp(ArgT&& arg, ArgsT&&... args)
      : GPUResidentOperator(std::forward<ArgT>(arg), std::forward<ArgsT>(args)...) {}
  MultiIOComputeGpuOp() = default;

  void setup(OperatorSpec& spec) override {
    for (int i = 0; i < N; ++i) {
      spec.device_input("in" + std::to_string(i), sizeof(int) * kElems);
      spec.device_output("out" + std::to_string(i), sizeof(int) * kElems);
    }
  }

  void compute([[maybe_unused]] InputContext& op_input, [[maybe_unused]] OutputContext& op_output,
               [[maybe_unused]] ExecutionContext& context) override {
    cudaStream_t stream = *cuda_stream();
    for (int i = 0; i < N; ++i) {
      auto* in_ptr = static_cast<const int*>(device_memory("in" + std::to_string(i)));
      auto* out_ptr = static_cast<int*>(device_memory("out" + std::to_string(i)));
      if (in_ptr && out_ptr) {
        launch_copy_add_kernel(out_ptr, in_ptr, 1, kElems, stream);
      }
    }
  }
};

/// Source with 3 outputs where port 1 has a deliberately larger buffer size (for error testing).
class MismatchedMultiPortSourceOp : public GPUResidentOperator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_TEMPLATE()
  explicit MismatchedMultiPortSourceOp(ArgT&& arg, ArgsT&&... args)
      : GPUResidentOperator(std::forward<ArgT>(arg), std::forward<ArgsT>(args)...) {}
  MismatchedMultiPortSourceOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.device_output("out0", sizeof(int) * kElems);
    spec.device_output("out1", sizeof(int) * kElems * 2);
    spec.device_output("out2", sizeof(int) * kElems);
  }

  void compute([[maybe_unused]] InputContext& op_input, [[maybe_unused]] OutputContext& op_output,
               [[maybe_unused]] ExecutionContext& context) override {}
};

/// Two-output source with per-port configurable memory-backed vs externally managed outputs.
template <bool Output0UsesDevicePtr, bool Output1UsesDevicePtr>
class MixedTwoOutputSourceGpuOp : public GPUResidentOperator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_TEMPLATE()
  explicit MixedTwoOutputSourceGpuOp(ArgT&& arg, ArgsT&&... args)
      : GPUResidentOperator(std::forward<ArgT>(arg), std::forward<ArgsT>(args)...) {}
  MixedTwoOutputSourceGpuOp() = default;

  ~MixedTwoOutputSourceGpuOp() override {
    if (dev_ptr_0_) {
      HOLOSCAN_CUDA_CALL_WARN(cudaFree(dev_ptr_0_));
    }
    if (dev_ptr_1_) {
      HOLOSCAN_CUDA_CALL_WARN(cudaFree(dev_ptr_1_));
    }
  }

  void setup(OperatorSpec& spec) override {
    if constexpr (Output0UsesDevicePtr) {
      HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaMalloc(&dev_ptr_0_, sizeof(int) * kElems),
                                     "Failed to allocate device memory");
      spec.device_output("out0", reinterpret_cast<CUdeviceptr>(dev_ptr_0_));
    } else {
      spec.device_output("out0", sizeof(int) * kElems);
    }

    if constexpr (Output1UsesDevicePtr) {
      HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaMalloc(&dev_ptr_1_, sizeof(int) * kElems),
                                     "Failed to allocate device memory");
      spec.device_output("out1", reinterpret_cast<CUdeviceptr>(dev_ptr_1_));
    } else {
      spec.device_output("out1", sizeof(int) * kElems);
    }
  }

  void compute([[maybe_unused]] InputContext& op_input, [[maybe_unused]] OutputContext& op_output,
               [[maybe_unused]] ExecutionContext& context) override {
    cudaStream_t stream = *cuda_stream();
    auto* out0 = static_cast<int*>(device_memory("out0"));
    auto* out1 = static_cast<int*>(device_memory("out1"));
    if (out0) {
      launch_init_pattern_kernel(out0, 0, kElems, stream);
    }
    if (out1) {
      launch_init_pattern_kernel(out1, 1, kElems, stream);
    }
  }

 private:
  void* dev_ptr_0_ = nullptr;
  void* dev_ptr_1_ = nullptr;
};

/// Single-output source with configurable memory-backed vs externally managed output.
template <bool OutputUsesDevicePtr>
class MixedSingleOutputSourceGpuOp : public GPUResidentOperator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_TEMPLATE()
  explicit MixedSingleOutputSourceGpuOp(ArgT&& arg, ArgsT&&... args)
      : GPUResidentOperator(std::forward<ArgT>(arg), std::forward<ArgsT>(args)...) {}
  MixedSingleOutputSourceGpuOp() = default;

  ~MixedSingleOutputSourceGpuOp() override {
    if (dev_ptr_) {
      HOLOSCAN_CUDA_CALL_WARN(cudaFree(dev_ptr_));
    }
  }

  void setup(OperatorSpec& spec) override {
    if constexpr (OutputUsesDevicePtr) {
      HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaMalloc(&dev_ptr_, sizeof(int) * kElems),
                                     "Failed to allocate device memory");
      spec.device_output("out0", reinterpret_cast<CUdeviceptr>(dev_ptr_));
    } else {
      spec.device_output("out0", sizeof(int) * kElems);
    }
  }

  void compute([[maybe_unused]] InputContext& op_input, [[maybe_unused]] OutputContext& op_output,
               [[maybe_unused]] ExecutionContext& context) override {
    cudaStream_t stream = *cuda_stream();
    auto* out0 = static_cast<int*>(device_memory("out0"));
    if (out0) {
      launch_init_pattern_kernel(out0, 0, kElems, stream);
    }
  }

 private:
  void* dev_ptr_ = nullptr;
};

/// Single-port compute operator with independently configurable input/output storage type.
template <bool InputUsesDevicePtr, bool OutputUsesDevicePtr>
class MixedSingleIOComputeGpuOp : public GPUResidentOperator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_TEMPLATE()
  explicit MixedSingleIOComputeGpuOp(ArgT&& arg, ArgsT&&... args)
      : GPUResidentOperator(std::forward<ArgT>(arg), std::forward<ArgsT>(args)...) {}
  MixedSingleIOComputeGpuOp() = default;

  ~MixedSingleIOComputeGpuOp() override {
    if (in_dev_ptr_) {
      HOLOSCAN_CUDA_CALL_WARN(cudaFree(in_dev_ptr_));
    }
    if (out_dev_ptr_) {
      HOLOSCAN_CUDA_CALL_WARN(cudaFree(out_dev_ptr_));
    }
  }

  void setup(OperatorSpec& spec) override {
    if constexpr (InputUsesDevicePtr) {
      HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaMalloc(&in_dev_ptr_, sizeof(int) * kElems),
                                     "Failed to allocate input device memory");
      spec.device_input("in0", reinterpret_cast<CUdeviceptr>(in_dev_ptr_));
    } else {
      spec.device_input("in0", sizeof(int) * kElems);
    }

    if constexpr (OutputUsesDevicePtr) {
      HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaMalloc(&out_dev_ptr_, sizeof(int) * kElems),
                                     "Failed to allocate output device memory");
      spec.device_output("out0", reinterpret_cast<CUdeviceptr>(out_dev_ptr_));
    } else {
      spec.device_output("out0", sizeof(int) * kElems);
    }
  }

  void compute([[maybe_unused]] InputContext& op_input, [[maybe_unused]] OutputContext& op_output,
               [[maybe_unused]] ExecutionContext& context) override {
    cudaStream_t stream = *cuda_stream();
    auto* in_ptr = static_cast<const int*>(device_memory("in0"));
    auto* out_ptr = static_cast<int*>(device_memory("out0"));
    if (in_ptr && out_ptr) {
      launch_copy_add_kernel(out_ptr, in_ptr, 1, kElems, stream);
    }
  }

 private:
  void* in_dev_ptr_ = nullptr;
  void* out_dev_ptr_ = nullptr;
};

/// Two-input sink with per-port configurable memory-backed vs externally managed inputs.
template <bool Input0UsesDevicePtr, bool Input1UsesDevicePtr>
class MixedTwoInputSinkGpuOp : public GPUResidentOperator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_TEMPLATE()
  explicit MixedTwoInputSinkGpuOp(ArgT&& arg, ArgsT&&... args)
      : GPUResidentOperator(std::forward<ArgT>(arg), std::forward<ArgsT>(args)...) {}
  MixedTwoInputSinkGpuOp() = default;

  ~MixedTwoInputSinkGpuOp() override {
    if (dev_ptr_0_) {
      HOLOSCAN_CUDA_CALL_WARN(cudaFree(dev_ptr_0_));
    }
    if (dev_ptr_1_) {
      HOLOSCAN_CUDA_CALL_WARN(cudaFree(dev_ptr_1_));
    }
  }

  void setup(OperatorSpec& spec) override {
    if constexpr (Input0UsesDevicePtr) {
      HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaMalloc(&dev_ptr_0_, sizeof(int) * kElems),
                                     "Failed to allocate input device memory");
      spec.device_input("in0", reinterpret_cast<CUdeviceptr>(dev_ptr_0_));
    } else {
      spec.device_input("in0", sizeof(int) * kElems);
    }

    if constexpr (Input1UsesDevicePtr) {
      HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaMalloc(&dev_ptr_1_, sizeof(int) * kElems),
                                     "Failed to allocate input device memory");
      spec.device_input("in1", reinterpret_cast<CUdeviceptr>(dev_ptr_1_));
    } else {
      spec.device_input("in1", sizeof(int) * kElems);
    }
  }

  void compute([[maybe_unused]] InputContext& op_input, [[maybe_unused]] OutputContext& op_output,
               [[maybe_unused]] ExecutionContext& context) override {}

 private:
  void* dev_ptr_0_ = nullptr;
  void* dev_ptr_1_ = nullptr;
};

/// Sink operator: N input ports, no-op compute.
template <int N>
class MultiInputSinkGpuOp : public GPUResidentOperator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_TEMPLATE()
  explicit MultiInputSinkGpuOp(ArgT&& arg, ArgsT&&... args)
      : GPUResidentOperator(std::forward<ArgT>(arg), std::forward<ArgsT>(args)...) {}
  MultiInputSinkGpuOp() = default;

  void setup(OperatorSpec& spec) override {
    for (int i = 0; i < N; ++i) {
      spec.device_input("in" + std::to_string(i), sizeof(int) * kElems);
    }
  }

  void compute([[maybe_unused]] InputContext& op_input, [[maybe_unused]] OutputContext& op_output,
               [[maybe_unused]] ExecutionContext& context) override {}
};

// ================================================================================================
// Helpers
// ================================================================================================

template <int N>
std::set<std::pair<std::string, std::string>> make_port_map() {
  std::set<std::pair<std::string, std::string>> pm;
  for (int i = 0; i < N; ++i) {
    pm.insert({"out" + std::to_string(i), "in" + std::to_string(i)});
  }
  return pm;
}

template <int N>
struct MultiIOGraph {
  std::shared_ptr<MultiOutputSourceGpuOp<N>> source;
  std::vector<std::shared_ptr<MultiIOComputeGpuOp<N>>> computes;
  std::shared_ptr<MultiInputSinkGpuOp<N>> sink;
};

struct DiamondGraph {
  std::shared_ptr<MultiOutputSourceGpuOp<2>> source;
  std::shared_ptr<MultiIOComputeGpuOp<1>> left;
  std::shared_ptr<MultiIOComputeGpuOp<1>> right;
  std::shared_ptr<MultiInputSinkGpuOp<2>> sink;
};

struct FanOutGraph {
  std::shared_ptr<MultiOutputSourceGpuOp<1>> source;
  std::shared_ptr<MultiIOComputeGpuOp<1>> left;
  std::shared_ptr<MultiIOComputeGpuOp<1>> right;
  std::shared_ptr<MultiInputSinkGpuOp<2>> sink;
};

// chain_length = 0: source -> sink
// chain_length = 1: source -> compute_0 -> sink
// ... and so on
template <int N>
MultiIOGraph<N> build_graph(Fragment& fragment, int chain_length) {
  MultiIOGraph<N> g;
  g.source = fragment.make_operator<MultiOutputSourceGpuOp<N>>("source");

  std::shared_ptr<Operator> prev = g.source;
  auto pm = make_port_map<N>();

  for (int i = 0; i < chain_length; ++i) {
    auto compute = fragment.make_operator<MultiIOComputeGpuOp<N>>("compute_" + std::to_string(i));
    fragment.add_flow(prev, compute, pm);
    g.computes.push_back(compute);
    prev = compute;
  }

  g.sink = fragment.make_operator<MultiInputSinkGpuOp<N>>("sink");
  fragment.add_flow(prev, g.sink, pm);
  return g;
}

DiamondGraph build_diamond_graph(Fragment& fragment) {
  DiamondGraph g;
  g.source = fragment.make_operator<MultiOutputSourceGpuOp<2>>("source");
  g.left = fragment.make_operator<MultiIOComputeGpuOp<1>>("left");
  g.right = fragment.make_operator<MultiIOComputeGpuOp<1>>("right");
  g.sink = fragment.make_operator<MultiInputSinkGpuOp<2>>("sink");

  fragment.add_flow(g.source, g.left, {{"out0", "in0"}});
  fragment.add_flow(g.source, g.right, {{"out1", "in0"}});
  fragment.add_flow(g.left, g.sink, {{"out0", "in0"}});
  fragment.add_flow(g.right, g.sink, {{"out0", "in1"}});

  return g;
}

FanOutGraph build_fan_out_graph(Fragment& fragment) {
  FanOutGraph g;
  g.source = fragment.make_operator<MultiOutputSourceGpuOp<1>>("source");
  g.left = fragment.make_operator<MultiIOComputeGpuOp<1>>("left");
  g.right = fragment.make_operator<MultiIOComputeGpuOp<1>>("right");
  g.sink = fragment.make_operator<MultiInputSinkGpuOp<2>>("sink");

  fragment.add_flow(g.source, g.left, {{"out0", "in0"}});
  fragment.add_flow(g.source, g.right, {{"out0", "in0"}});
  fragment.add_flow(g.left, g.sink, {{"out0", "in0"}});
  fragment.add_flow(g.right, g.sink, {{"out0", "in1"}});

  return g;
}

template <bool SourceOut0UsesDevicePtr, bool SourceOut1UsesDevicePtr, bool LeftInputUsesDevicePtr,
          bool LeftOutputUsesDevicePtr, bool RightInputUsesDevicePtr, bool RightOutputUsesDevicePtr,
          bool SinkIn0UsesDevicePtr, bool SinkIn1UsesDevicePtr>
std::shared_ptr<GPUResidentOperator> build_mixed_diamond_graph(Fragment& fragment) {
  auto source = fragment.make_operator<
      MixedTwoOutputSourceGpuOp<SourceOut0UsesDevicePtr, SourceOut1UsesDevicePtr>>("source");
  auto left = fragment.make_operator<
      MixedSingleIOComputeGpuOp<LeftInputUsesDevicePtr, LeftOutputUsesDevicePtr>>("left");
  auto right = fragment.make_operator<
      MixedSingleIOComputeGpuOp<RightInputUsesDevicePtr, RightOutputUsesDevicePtr>>("right");
  auto sink =
      fragment.make_operator<MixedTwoInputSinkGpuOp<SinkIn0UsesDevicePtr, SinkIn1UsesDevicePtr>>(
          "sink");

  fragment.add_flow(source, left, {{"out0", "in0"}});
  fragment.add_flow(source, right, {{"out1", "in0"}});
  fragment.add_flow(left, sink, {{"out0", "in0"}});
  fragment.add_flow(right, sink, {{"out0", "in1"}});

  return sink;
}

template <bool SourceOutputUsesDevicePtr, bool LeftInputUsesDevicePtr, bool LeftOutputUsesDevicePtr,
          bool RightInputUsesDevicePtr, bool RightOutputUsesDevicePtr, bool SinkIn0UsesDevicePtr,
          bool SinkIn1UsesDevicePtr>
std::shared_ptr<GPUResidentOperator> build_mixed_fan_out_graph(Fragment& fragment) {
  auto source =
      fragment.make_operator<MixedSingleOutputSourceGpuOp<SourceOutputUsesDevicePtr>>("source");
  auto left = fragment.make_operator<
      MixedSingleIOComputeGpuOp<LeftInputUsesDevicePtr, LeftOutputUsesDevicePtr>>("left");
  auto right = fragment.make_operator<
      MixedSingleIOComputeGpuOp<RightInputUsesDevicePtr, RightOutputUsesDevicePtr>>("right");
  auto sink =
      fragment.make_operator<MixedTwoInputSinkGpuOp<SinkIn0UsesDevicePtr, SinkIn1UsesDevicePtr>>(
          "sink");

  fragment.add_flow(source, left, {{"out0", "in0"}});
  fragment.add_flow(source, right, {{"out0", "in0"}});
  fragment.add_flow(left, sink, {{"out0", "in0"}});
  fragment.add_flow(right, sink, {{"out0", "in1"}});

  return sink;
}

std::shared_ptr<GPUResidentOperator> build_asymmetric_fan_out_graph(Fragment& fragment) {
  auto source = fragment.make_operator<MixedSingleOutputSourceGpuOp<true>>("source");
  auto compute = fragment.make_operator<MixedSingleIOComputeGpuOp<false, false>>("compute");
  auto sink = fragment.make_operator<MixedTwoInputSinkGpuOp<true, false>>("sink");

  fragment.add_flow(source, sink, {{"out0", "in0"}});
  fragment.add_flow(source, compute, {{"out0", "in0"}});
  fragment.add_flow(compute, sink, {{"out0", "in1"}});

  return sink;
}

bool wait_for_launch(Fragment& fragment, int timeout_sec = 10) {
  auto start = std::chrono::steady_clock::now();
  while (!fragment.gpu_resident().is_launched()) {
    if (std::chrono::steady_clock::now() - start >= std::chrono::seconds(timeout_sec)) {
      return false;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
  return true;
}

bool wait_for_result(Fragment& fragment, int timeout_sec = 10) {
  auto start = std::chrono::steady_clock::now();
  while (!fragment.gpu_resident().result_ready()) {
    if (std::chrono::steady_clock::now() - start >= std::chrono::seconds(timeout_sec)) {
      return false;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  return true;
}

void teardown_fragment(Fragment& fragment, std::future<void>& future, int timeout_sec = 10) {
  if (fragment.gpu_resident().is_launched()) {
    fragment.gpu_resident().tear_down();
  }
  auto start = std::chrono::steady_clock::now();
  while (fragment.gpu_resident().is_launched()) {
    if (std::chrono::steady_clock::now() - start >= std::chrono::seconds(timeout_sec)) {
      ADD_FAILURE() << "teardown_fragment: fragment still launched after " << timeout_sec << "s";
      return;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  if (future.valid()) {
    future.get();
  }
}

template <int N>
void verify_sink_data(const std::shared_ptr<MultiInputSinkGpuOp<N>>& sink, int chain_length) {
  std::vector<int> host_data(kElems);
  for (int port = 0; port < N; ++port) {
    std::string port_name = "in" + std::to_string(port);
    void* dev_ptr = sink->device_memory(port_name);
    ASSERT_NE(dev_ptr, nullptr) << "sink." << port_name << " device_memory is null";

    HOLOSCAN_CUDA_CALL_THROW_ERROR(
        cudaMemcpy(host_data.data(), dev_ptr, sizeof(int) * kElems, cudaMemcpyDeviceToHost),
        "cudaMemcpy failed");

    for (int i = 0; i < kElems; ++i) {
      int expected = port * 1000 + i + chain_length;
      EXPECT_EQ(host_data[i], expected)
          << "sink." << port_name << "[" << i << "]: expected " << expected << " got "
          << host_data[i] << " (chain_length=" << chain_length << ")";
    }
  }
}

void verify_diamond_sink_data(const std::shared_ptr<GPUResidentOperator>& sink) {
  std::vector<int> host_data(kElems);

  for (int port = 0; port < 2; ++port) {
    std::string port_name = "in" + std::to_string(port);
    void* dev_ptr = sink->device_memory(port_name);
    ASSERT_NE(dev_ptr, nullptr) << "sink." << port_name << " device_memory is null";

    HOLOSCAN_CUDA_CALL_THROW_ERROR(
        cudaMemcpy(host_data.data(), dev_ptr, sizeof(int) * kElems, cudaMemcpyDeviceToHost),
        "cudaMemcpy failed");

    for (int i = 0; i < kElems; ++i) {
      int expected = port * 1000 + i + 1;
      EXPECT_EQ(host_data[i], expected) << "sink." << port_name << "[" << i << "]: expected "
                                        << expected << " got " << host_data[i];
    }
  }
}

void verify_fan_out_sink_data(const std::shared_ptr<GPUResidentOperator>& sink) {
  std::vector<int> host_data(kElems);

  for (int port = 0; port < 2; ++port) {
    std::string port_name = "in" + std::to_string(port);
    void* dev_ptr = sink->device_memory(port_name);
    ASSERT_NE(dev_ptr, nullptr) << "sink." << port_name << " device_memory is null";

    HOLOSCAN_CUDA_CALL_THROW_ERROR(
        cudaMemcpy(host_data.data(), dev_ptr, sizeof(int) * kElems, cudaMemcpyDeviceToHost),
        "cudaMemcpy failed");

    for (int i = 0; i < kElems; ++i) {
      int expected = i + 1;
      EXPECT_EQ(host_data[i], expected) << "sink." << port_name << "[" << i << "]: expected "
                                        << expected << " got " << host_data[i];
    }
  }
}

void verify_asymmetric_fan_out_sink_data(const std::shared_ptr<GPUResidentOperator>& sink) {
  std::vector<int> host_data(kElems);

  for (int port = 0; port < 2; ++port) {
    std::string port_name = "in" + std::to_string(port);
    void* dev_ptr = sink->device_memory(port_name);
    ASSERT_NE(dev_ptr, nullptr) << "sink." << port_name << " device_memory is null";

    HOLOSCAN_CUDA_CALL_THROW_ERROR(
        cudaMemcpy(host_data.data(), dev_ptr, sizeof(int) * kElems, cudaMemcpyDeviceToHost),
        "cudaMemcpy failed");

    for (int i = 0; i < kElems; ++i) {
      int expected = (port == 0) ? i : i + 1;
      EXPECT_EQ(host_data[i], expected) << "sink." << port_name << "[" << i << "]: expected "
                                        << expected << " got " << host_data[i];
    }
  }
}

// ================================================================================================
// Test Fixture
// ================================================================================================

class GPUResidentMultiIOTest : public ::testing::Test {
 protected:
  void SetUp() override {
    int device_count = 0;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    if (error != cudaSuccess || device_count == 0) {
      GTEST_SKIP() << "No CUDA devices available, skipping GPU-resident multi-IO tests";
    }
  }
};

enum class MixedDagCorrectnessVariant {
  Diamond_SourceMemPtr_ComputeInputsPtrMem_OutputsMemMem_SinkMemMem,
  Diamond_SourcePtrMem_ComputeInputsMemPtr_OutputsMemPtr_SinkPtrMem,
  Diamond_SourcePtrPtr_ComputeInputsPtrPtr_OutputsPtrPtr_SinkPtrPtr,
  FanOut_SourcePtr_ComputeInputsPtrMem_OutputsMemMem_SinkMemMem,
  FanOut_SourcePtr_ComputeInputsMemPtr_OutputsMemMem_SinkMemMem,
  FanOut_SourcePtr_ComputeInputsMemMem_OutputsMemPtr_SinkPtrMem,
  FanOut_SourceMem_ComputeInputsMemMem_OutputsMemPtr_SinkPtrMem,
  FanOut_SourcePtr_ComputeInputsPtrPtr_OutputsPtrPtr_SinkPtrPtr,
};

std::string mixed_dag_correctness_variant_name(MixedDagCorrectnessVariant variant) {
  switch (variant) {
    case MixedDagCorrectnessVariant::
        Diamond_SourceMemPtr_ComputeInputsPtrMem_OutputsMemMem_SinkMemMem:
      return "Diamond_SourceMemPtr_ComputeInputsPtrMem_OutputsMemMem_SinkMemMem";
    case MixedDagCorrectnessVariant::
        Diamond_SourcePtrMem_ComputeInputsMemPtr_OutputsMemPtr_SinkPtrMem:
      return "Diamond_SourcePtrMem_ComputeInputsMemPtr_OutputsMemPtr_SinkPtrMem";
    case MixedDagCorrectnessVariant::
        Diamond_SourcePtrPtr_ComputeInputsPtrPtr_OutputsPtrPtr_SinkPtrPtr:
      return "Diamond_SourcePtrPtr_ComputeInputsPtrPtr_OutputsPtrPtr_SinkPtrPtr";
    case MixedDagCorrectnessVariant::FanOut_SourcePtr_ComputeInputsPtrMem_OutputsMemMem_SinkMemMem:
      return "FanOut_SourcePtr_ComputeInputsPtrMem_OutputsMemMem_SinkMemMem";
    case MixedDagCorrectnessVariant::FanOut_SourcePtr_ComputeInputsMemPtr_OutputsMemMem_SinkMemMem:
      return "FanOut_SourcePtr_ComputeInputsMemPtr_OutputsMemMem_SinkMemMem";
    case MixedDagCorrectnessVariant::FanOut_SourcePtr_ComputeInputsMemMem_OutputsMemPtr_SinkPtrMem:
      return "FanOut_SourcePtr_ComputeInputsMemMem_OutputsMemPtr_SinkPtrMem";
    case MixedDagCorrectnessVariant::FanOut_SourceMem_ComputeInputsMemMem_OutputsMemPtr_SinkPtrMem:
      return "FanOut_SourceMem_ComputeInputsMemMem_OutputsMemPtr_SinkPtrMem";
    case MixedDagCorrectnessVariant::FanOut_SourcePtr_ComputeInputsPtrPtr_OutputsPtrPtr_SinkPtrPtr:
      return "FanOut_SourcePtr_ComputeInputsPtrPtr_OutputsPtrPtr_SinkPtrPtr";
  }

  return "Unknown";
}

class GPUResidentMixedDagCorrectnessTest
    : public GPUResidentMultiIOTest,
      public ::testing::WithParamInterface<MixedDagCorrectnessVariant> {};

std::shared_ptr<GPUResidentOperator> build_mixed_correctness_graph(
    Fragment& fragment, MixedDagCorrectnessVariant variant) {
  switch (variant) {
    case MixedDagCorrectnessVariant::
        Diamond_SourceMemPtr_ComputeInputsPtrMem_OutputsMemMem_SinkMemMem:
      return build_mixed_diamond_graph<false, true, true, false, false, false, false, false>(
          fragment);
    case MixedDagCorrectnessVariant::
        Diamond_SourcePtrMem_ComputeInputsMemPtr_OutputsMemPtr_SinkPtrMem:
      return build_mixed_diamond_graph<true, false, false, false, true, true, true, false>(
          fragment);
    case MixedDagCorrectnessVariant::
        Diamond_SourcePtrPtr_ComputeInputsPtrPtr_OutputsPtrPtr_SinkPtrPtr:
      return build_mixed_diamond_graph<true, true, true, true, true, true, true, true>(fragment);
    case MixedDagCorrectnessVariant::FanOut_SourcePtr_ComputeInputsPtrMem_OutputsMemMem_SinkMemMem:
      return build_mixed_fan_out_graph<true, true, false, false, false, false, false>(fragment);
    case MixedDagCorrectnessVariant::FanOut_SourcePtr_ComputeInputsMemPtr_OutputsMemMem_SinkMemMem:
      return build_mixed_fan_out_graph<true, false, false, true, false, false, false>(fragment);
    case MixedDagCorrectnessVariant::FanOut_SourcePtr_ComputeInputsMemMem_OutputsMemPtr_SinkPtrMem:
      return build_mixed_fan_out_graph<true, false, false, false, true, true, false>(fragment);
    case MixedDagCorrectnessVariant::FanOut_SourceMem_ComputeInputsMemMem_OutputsMemPtr_SinkPtrMem:
      return build_mixed_fan_out_graph<false, false, false, false, true, true, false>(fragment);
    case MixedDagCorrectnessVariant::FanOut_SourcePtr_ComputeInputsPtrPtr_OutputsPtrPtr_SinkPtrPtr:
      return build_mixed_fan_out_graph<true, true, true, true, true, true, true>(fragment);
  }

  throw std::runtime_error("Unhandled mixed DAG correctness variant");
}

bool is_diamond_variant(MixedDagCorrectnessVariant variant) {
  switch (variant) {
    case MixedDagCorrectnessVariant::
        Diamond_SourceMemPtr_ComputeInputsPtrMem_OutputsMemMem_SinkMemMem:
    case MixedDagCorrectnessVariant::
        Diamond_SourcePtrMem_ComputeInputsMemPtr_OutputsMemPtr_SinkPtrMem:
    case MixedDagCorrectnessVariant::
        Diamond_SourcePtrPtr_ComputeInputsPtrPtr_OutputsPtrPtr_SinkPtrPtr:
      return true;
    case MixedDagCorrectnessVariant::FanOut_SourcePtr_ComputeInputsPtrMem_OutputsMemMem_SinkMemMem:
    case MixedDagCorrectnessVariant::FanOut_SourcePtr_ComputeInputsMemPtr_OutputsMemMem_SinkMemMem:
    case MixedDagCorrectnessVariant::FanOut_SourcePtr_ComputeInputsMemMem_OutputsMemPtr_SinkPtrMem:
    case MixedDagCorrectnessVariant::FanOut_SourceMem_ComputeInputsMemMem_OutputsMemPtr_SinkPtrMem:
    case MixedDagCorrectnessVariant::FanOut_SourcePtr_ComputeInputsPtrPtr_OutputsPtrPtr_SinkPtrPtr:
      return false;
  }

  return false;
}

// ================================================================================================
// Section 1: Port Setup Verification
//
// Verifies that MultiOutput/MultiIOCompute/MultiInput operators declare the correct
// number and size of device ports for N = 1, 3, 5, 10, 20.
// ================================================================================================

template <int N>
void test_port_setup() {
  Fragment fragment;
  auto source = fragment.make_operator<MultiOutputSourceGpuOp<N>>("source");
  auto compute = fragment.make_operator<MultiIOComputeGpuOp<N>>("compute");
  auto sink = fragment.make_operator<MultiInputSinkGpuOp<N>>("sink");

  EXPECT_EQ(source->spec()->outputs().size(), static_cast<size_t>(N));
  EXPECT_EQ(source->spec()->inputs().size(), 0u);

  EXPECT_EQ(compute->spec()->inputs().size(), static_cast<size_t>(N));
  EXPECT_EQ(compute->spec()->outputs().size(), static_cast<size_t>(N));

  EXPECT_EQ(sink->spec()->inputs().size(), static_cast<size_t>(N));
  EXPECT_EQ(sink->spec()->outputs().size(), 0u);

  for (int i = 0; i < N; ++i) {
    std::string out_name = "out" + std::to_string(i);
    std::string in_name = "in" + std::to_string(i);

    auto& src_out = source->spec()->outputs();
    EXPECT_NE(src_out.find(out_name), src_out.end());
    EXPECT_EQ(src_out[out_name]->memory_block_size(), sizeof(int) * kElems);

    auto& comp_in = compute->spec()->inputs();
    auto& comp_out = compute->spec()->outputs();
    ASSERT_NE(comp_in.find(in_name), comp_in.end());
    ASSERT_NE(comp_out.find(out_name), comp_out.end());
    EXPECT_EQ(comp_in[in_name]->memory_block_size(), sizeof(int) * kElems);
    EXPECT_EQ(comp_out[out_name]->memory_block_size(), sizeof(int) * kElems);

    auto& sink_in = sink->spec()->inputs();
    ASSERT_NE(sink_in.find(in_name), sink_in.end());
    EXPECT_EQ(sink_in[in_name]->memory_block_size(), sizeof(int) * kElems);
  }
}

TEST_F(GPUResidentMultiIOTest, PortSetup_1Port) {
  test_port_setup<1>();
}
TEST_F(GPUResidentMultiIOTest, PortSetup_3Ports) {
  test_port_setup<3>();
}
TEST_F(GPUResidentMultiIOTest, PortSetup_5Ports) {
  test_port_setup<5>();
}
TEST_F(GPUResidentMultiIOTest, PortSetup_10Ports) {
  test_port_setup<10>();
}
TEST_F(GPUResidentMultiIOTest, PortSetup_20Ports) {
  test_port_setup<20>();
}

// ================================================================================================
// Section 2: Fragment Initialization
//
// Builds multi-IO graphs, calls initialize_fragment(), and checks that every
// port across every operator has a non-null device_memory() address.
// ================================================================================================

template <int N>
void test_fragment_init(int chain_length) {
  Fragment fragment;
  auto g = build_graph<N>(fragment, chain_length);

  auto executor = std::dynamic_pointer_cast<GPUResidentExecutor>(fragment.executor_shared());
  ASSERT_NE(executor, nullptr);
  EXPECT_TRUE(executor->initialize_fragment());

  for (int i = 0; i < N; ++i) {
    std::string out_name = "out" + std::to_string(i);
    std::string in_name = "in" + std::to_string(i);

    EXPECT_NE(g.source->device_memory(out_name), nullptr) << "source." << out_name << " is null";
    EXPECT_NE(g.sink->device_memory(in_name), nullptr) << "sink." << in_name << " is null";

    for (int c = 0; c < chain_length; ++c) {
      EXPECT_NE(g.computes[c]->device_memory(in_name), nullptr)
          << "compute_" << c << "." << in_name << " is null";
      EXPECT_NE(g.computes[c]->device_memory(out_name), nullptr)
          << "compute_" << c << "." << out_name << " is null";
    }
  }
}

// Varying port counts, chain_length = 0 (source -> sink)
TEST_F(GPUResidentMultiIOTest, FragmentInit_1Port_Chain0) {
  test_fragment_init<1>(0);
}
TEST_F(GPUResidentMultiIOTest, FragmentInit_3Ports_Chain0) {
  test_fragment_init<3>(0);
}
TEST_F(GPUResidentMultiIOTest, FragmentInit_5Ports_Chain0) {
  test_fragment_init<5>(0);
}
TEST_F(GPUResidentMultiIOTest, FragmentInit_10Ports_Chain0) {
  test_fragment_init<10>(0);
}
TEST_F(GPUResidentMultiIOTest, FragmentInit_20Ports_Chain0) {
  test_fragment_init<20>(0);
}

TEST_F(GPUResidentMultiIOTest, FragmentInit_DiamondDag) {
  Fragment fragment;
  auto g = build_diamond_graph(fragment);

  auto executor = std::dynamic_pointer_cast<GPUResidentExecutor>(fragment.executor_shared());
  ASSERT_NE(executor, nullptr);
  EXPECT_TRUE(executor->initialize_fragment());

  EXPECT_NE(g.source->device_memory("out0"), nullptr);
  EXPECT_NE(g.source->device_memory("out1"), nullptr);
  EXPECT_NE(g.left->device_memory("in0"), nullptr);
  EXPECT_NE(g.left->device_memory("out0"), nullptr);
  EXPECT_NE(g.right->device_memory("in0"), nullptr);
  EXPECT_NE(g.right->device_memory("out0"), nullptr);
  EXPECT_NE(g.sink->device_memory("in0"), nullptr);
  EXPECT_NE(g.sink->device_memory("in1"), nullptr);

  EXPECT_EQ(g.source->device_memory("out0"), g.left->device_memory("in0"));
  EXPECT_EQ(g.source->device_memory("out1"), g.right->device_memory("in0"));
  EXPECT_EQ(g.left->device_memory("out0"), g.sink->device_memory("in0"));
  EXPECT_EQ(g.right->device_memory("out0"), g.sink->device_memory("in1"));

  EXPECT_NE(g.source->device_memory("out0"), g.source->device_memory("out1"));
  EXPECT_NE(g.sink->device_memory("in0"), g.sink->device_memory("in1"));
}

TEST_F(GPUResidentMultiIOTest, FragmentInit_FanOutAcrossOperators) {
  Fragment fragment;
  auto g = build_fan_out_graph(fragment);

  auto executor = std::dynamic_pointer_cast<GPUResidentExecutor>(fragment.executor_shared());
  ASSERT_NE(executor, nullptr);
  EXPECT_TRUE(executor->initialize_fragment());

  EXPECT_NE(g.source->device_memory("out0"), nullptr);
  EXPECT_NE(g.left->device_memory("in0"), nullptr);
  EXPECT_NE(g.right->device_memory("in0"), nullptr);
  EXPECT_NE(g.left->device_memory("out0"), nullptr);
  EXPECT_NE(g.right->device_memory("out0"), nullptr);
  EXPECT_NE(g.sink->device_memory("in0"), nullptr);
  EXPECT_NE(g.sink->device_memory("in1"), nullptr);

  EXPECT_EQ(g.source->device_memory("out0"), g.left->device_memory("in0"));
  EXPECT_EQ(g.source->device_memory("out0"), g.right->device_memory("in0"));
  EXPECT_EQ(g.left->device_memory("out0"), g.sink->device_memory("in0"));
  EXPECT_EQ(g.right->device_memory("out0"), g.sink->device_memory("in1"));
}

TEST_F(GPUResidentMultiIOTest, FragmentInit_FanOutSingleCall) {
  Fragment fragment;
  auto source = fragment.make_operator<MultiOutputSourceGpuOp<1>>("source");
  auto sink = fragment.make_operator<MultiInputSinkGpuOp<2>>("sink");

  fragment.add_flow(source, sink, {{"out0", "in0"}, {"out0", "in1"}});

  auto executor = std::dynamic_pointer_cast<GPUResidentExecutor>(fragment.executor_shared());
  ASSERT_NE(executor, nullptr);
  EXPECT_TRUE(executor->initialize_fragment());

  EXPECT_EQ(source->device_memory("out0"), sink->device_memory("in0"));
  EXPECT_EQ(source->device_memory("out0"), sink->device_memory("in1"));
}

// Varying chain lengths, 3 ports
TEST_F(GPUResidentMultiIOTest, FragmentInit_3Ports_Chain1) {
  test_fragment_init<3>(1);
}
TEST_F(GPUResidentMultiIOTest, FragmentInit_3Ports_Chain3) {
  test_fragment_init<3>(3);
}
TEST_F(GPUResidentMultiIOTest, FragmentInit_3Ports_Chain5) {
  test_fragment_init<3>(5);
}
TEST_F(GPUResidentMultiIOTest, FragmentInit_3Ports_Chain10) {
  test_fragment_init<3>(10);
}

// ================================================================================================
// Section 3: Device Memory Sharing
//
// Verifies that connected ports share the same device address and that distinct
// ports have different addresses.
// ================================================================================================

template <int N>
void test_memory_sharing(int chain_length) {
  Fragment fragment;
  auto g = build_graph<N>(fragment, chain_length);

  auto executor = std::dynamic_pointer_cast<GPUResidentExecutor>(fragment.executor_shared());
  ASSERT_NE(executor, nullptr);
  EXPECT_TRUE(executor->initialize_fragment());

  for (int i = 0; i < N; ++i) {
    std::string out_name = "out" + std::to_string(i);
    std::string in_name = "in" + std::to_string(i);

    if (chain_length == 0) {
      EXPECT_EQ(g.source->device_memory(out_name), g.sink->device_memory(in_name))
          << "Port " << i << ": source.out and sink.in must share memory (chain=0)";
    } else {
      EXPECT_EQ(g.source->device_memory(out_name), g.computes[0]->device_memory(in_name))
          << "Port " << i << ": source.out and compute_0.in must share memory";

      for (int c = 0; c + 1 < chain_length; ++c) {
        EXPECT_EQ(g.computes[c]->device_memory(out_name), g.computes[c + 1]->device_memory(in_name))
            << "Port " << i << ": compute_" << c << ".out and compute_" << c + 1
            << ".in must share memory";
      }

      EXPECT_EQ(g.computes.back()->device_memory(out_name), g.sink->device_memory(in_name))
          << "Port " << i << ": last_compute.out and sink.in must share memory";
    }
  }

  if constexpr (N > 1) {
    // Distinct port indices must have distinct addresses on the source
    for (int i = 0; i < N; ++i) {
      for (int j = i + 1; j < N; ++j) {
        EXPECT_NE(g.source->device_memory("out" + std::to_string(i)),
                  g.source->device_memory("out" + std::to_string(j)))
            << "source: out" << i << " and out" << j << " must not share memory";
      }
    }
  }

  // Each compute operator's input and output for the same port index must be distinct,
  // and different port indices must also be distinct from each other.
  for (int c = 0; c < chain_length; ++c) {
    for (int i = 0; i < N; ++i) {
      void* in_i = g.computes[c]->device_memory("in" + std::to_string(i));
      void* out_i = g.computes[c]->device_memory("out" + std::to_string(i));
      EXPECT_NE(in_i, out_i) << "compute_" << c << ": in" << i << " and out" << i
                             << " must not share memory";

      if constexpr (N > 1) {
        for (int j = i + 1; j < N; ++j) {
          void* in_j = g.computes[c]->device_memory("in" + std::to_string(j));
          void* out_j = g.computes[c]->device_memory("out" + std::to_string(j));
          EXPECT_NE(in_i, in_j) << "compute_" << c << ": in" << i << " and in" << j
                                << " must not share memory";
          EXPECT_NE(out_i, out_j) << "compute_" << c << ": out" << i << " and out" << j
                                  << " must not share memory";
        }
      }
    }
  }

  if constexpr (N > 1) {
    // Distinct port indices on the sink must have distinct addresses
    for (int i = 0; i < N; ++i) {
      for (int j = i + 1; j < N; ++j) {
        EXPECT_NE(g.sink->device_memory("in" + std::to_string(i)),
                  g.sink->device_memory("in" + std::to_string(j)))
            << "sink: in" << i << " and in" << j << " must not share memory";
      }
    }
  }
}

TEST_F(GPUResidentMultiIOTest, MemorySharing_1Port_Chain0) {
  test_memory_sharing<1>(0);
}
TEST_F(GPUResidentMultiIOTest, MemorySharing_3Ports_Chain0) {
  test_memory_sharing<3>(0);
}
TEST_F(GPUResidentMultiIOTest, MemorySharing_3Ports_Chain1) {
  test_memory_sharing<3>(1);
}
TEST_F(GPUResidentMultiIOTest, MemorySharing_3Ports_Chain3) {
  test_memory_sharing<3>(3);
}
TEST_F(GPUResidentMultiIOTest, MemorySharing_5Ports_Chain1) {
  test_memory_sharing<5>(1);
}
TEST_F(GPUResidentMultiIOTest, MemorySharing_10Ports_Chain1) {
  test_memory_sharing<10>(1);
}
TEST_F(GPUResidentMultiIOTest, MemorySharing_20Ports_Chain0) {
  test_memory_sharing<20>(0);
}
TEST_F(GPUResidentMultiIOTest, MemorySharing_20Ports_Chain5) {
  test_memory_sharing<20>(5);
}

// ================================================================================================
// Section 4: End-to-End CUDA Kernel Correctness
//
// Runs the fragment, triggers one data_ready iteration, then copies device memory
// back to the host and verifies the values.
//
// Expected value at sink port j, element i after passing through `chain_length`
// compute operators:   j*1000 + i + chain_length
//   (source writes j*1000+i; each compute adds 1)
// ================================================================================================

template <int N>
void test_correctness(int chain_length) {
  Fragment fragment;
  auto g = build_graph<N>(fragment, chain_length);

  fragment.gpu_resident().sync_with_host();

  auto future = fragment.run_async();
  ASSERT_TRUE(wait_for_launch(fragment)) << "Fragment did not launch within timeout";

  fragment.gpu_resident().data_ready();
  ASSERT_TRUE(wait_for_result(fragment)) << "Result not ready within timeout";

  verify_sink_data<N>(g.sink, chain_length);
  teardown_fragment(fragment, future);
}

// 1:1 mapping baseline (single port)
TEST_F(GPUResidentMultiIOTest, Correctness_1Port_Chain0) {
  test_correctness<1>(0);
}
TEST_F(GPUResidentMultiIOTest, Correctness_1Port_Chain5) {
  test_correctness<1>(5);
}

// 3 ports
TEST_F(GPUResidentMultiIOTest, Correctness_3Ports_Chain0) {
  test_correctness<3>(0);
}
TEST_F(GPUResidentMultiIOTest, Correctness_3Ports_Chain1) {
  test_correctness<3>(1);
}
TEST_F(GPUResidentMultiIOTest, Correctness_3Ports_Chain5) {
  test_correctness<3>(5);
}
TEST_F(GPUResidentMultiIOTest, Correctness_3Ports_Chain10) {
  test_correctness<3>(10);
}

// 5 ports
TEST_F(GPUResidentMultiIOTest, Correctness_5Ports_Chain0) {
  test_correctness<5>(0);
}
TEST_F(GPUResidentMultiIOTest, Correctness_5Ports_Chain1) {
  test_correctness<5>(1);
}
TEST_F(GPUResidentMultiIOTest, Correctness_5Ports_Chain3) {
  test_correctness<5>(3);
}

// 10 ports
TEST_F(GPUResidentMultiIOTest, Correctness_10Ports_Chain0) {
  test_correctness<10>(0);
}
TEST_F(GPUResidentMultiIOTest, Correctness_10Ports_Chain1) {
  test_correctness<10>(1);
}
TEST_F(GPUResidentMultiIOTest, Correctness_10Ports_Chain3) {
  test_correctness<10>(3);
}

// 20 ports (maximum)
TEST_F(GPUResidentMultiIOTest, Correctness_20Ports_Chain0) {
  test_correctness<20>(0);
}
TEST_F(GPUResidentMultiIOTest, Correctness_20Ports_Chain1) {
  test_correctness<20>(1);
}

TEST_F(GPUResidentMultiIOTest, Correctness_DiamondDag) {
  Fragment fragment;
  auto g = build_diamond_graph(fragment);

  fragment.gpu_resident().sync_with_host();

  auto future = fragment.run_async();
  ASSERT_TRUE(wait_for_launch(fragment)) << "Fragment did not launch within timeout";

  fragment.gpu_resident().data_ready();
  ASSERT_TRUE(wait_for_result(fragment)) << "Result not ready within timeout";

  verify_diamond_sink_data(g.sink);
  teardown_fragment(fragment, future);
}

TEST_F(GPUResidentMultiIOTest, Correctness_FanOutAcrossOperators) {
  Fragment fragment;
  auto g = build_fan_out_graph(fragment);

  fragment.gpu_resident().sync_with_host();

  auto future = fragment.run_async();
  ASSERT_TRUE(wait_for_launch(fragment)) << "Fragment did not launch within timeout";

  fragment.gpu_resident().data_ready();
  ASSERT_TRUE(wait_for_result(fragment)) << "Result not ready within timeout";

  verify_fan_out_sink_data(g.sink);
  teardown_fragment(fragment, future);
}

TEST_F(GPUResidentMultiIOTest, Correctness_FanOutAsymmetricDepth_SourcePtr) {
  Fragment fragment;
  auto sink = build_asymmetric_fan_out_graph(fragment);

  fragment.gpu_resident().sync_with_host();

  auto future = fragment.run_async();
  ASSERT_TRUE(wait_for_launch(fragment)) << "Fragment did not launch within timeout";

  fragment.gpu_resident().data_ready();
  ASSERT_TRUE(wait_for_result(fragment)) << "Result not ready within timeout";

  verify_asymmetric_fan_out_sink_data(sink);
  teardown_fragment(fragment, future);
}

TEST_P(GPUResidentMixedDagCorrectnessTest, Correctness_MixedDagAndFanOut) {
  Fragment fragment;
  auto variant = GetParam();
  auto sink = build_mixed_correctness_graph(fragment, variant);

  fragment.gpu_resident().sync_with_host();

  auto future = fragment.run_async();
  ASSERT_TRUE(wait_for_launch(fragment)) << "Fragment did not launch within timeout";

  fragment.gpu_resident().data_ready();
  ASSERT_TRUE(wait_for_result(fragment)) << "Result not ready within timeout";

  if (is_diamond_variant(variant)) {
    verify_diamond_sink_data(sink);
  } else {
    verify_fan_out_sink_data(sink);
  }
  teardown_fragment(fragment, future);
}

INSTANTIATE_TEST_SUITE_P(
    MixedDagCorrectnessCoverage, GPUResidentMixedDagCorrectnessTest,
    testing::Values(
        MixedDagCorrectnessVariant::
            Diamond_SourceMemPtr_ComputeInputsPtrMem_OutputsMemMem_SinkMemMem,
        MixedDagCorrectnessVariant::
            Diamond_SourcePtrMem_ComputeInputsMemPtr_OutputsMemPtr_SinkPtrMem,
        MixedDagCorrectnessVariant::
            Diamond_SourcePtrPtr_ComputeInputsPtrPtr_OutputsPtrPtr_SinkPtrPtr,
        MixedDagCorrectnessVariant::FanOut_SourcePtr_ComputeInputsPtrMem_OutputsMemMem_SinkMemMem,
        MixedDagCorrectnessVariant::FanOut_SourcePtr_ComputeInputsMemPtr_OutputsMemMem_SinkMemMem,
        MixedDagCorrectnessVariant::FanOut_SourcePtr_ComputeInputsMemMem_OutputsMemPtr_SinkPtrMem,
        MixedDagCorrectnessVariant::FanOut_SourceMem_ComputeInputsMemMem_OutputsMemPtr_SinkPtrMem,
        MixedDagCorrectnessVariant::FanOut_SourcePtr_ComputeInputsPtrPtr_OutputsPtrPtr_SinkPtrPtr),
    [](const testing::TestParamInfo<MixedDagCorrectnessVariant>& info) {
      return mixed_dag_correctness_variant_name(info.param);
    });

// ================================================================================================
// Section 5: Multiple Iterations
//
// Verifies determinism across repeated graph replays. Because the source kernel
// always re-initialises the pattern, final values must be identical after every
// iteration.
// ================================================================================================

template <int N>
void test_multiple_iterations(int chain_length, int num_iterations) {
  Fragment fragment;
  auto g = build_graph<N>(fragment, chain_length);

  fragment.gpu_resident().sync_with_host();

  auto future = fragment.run_async();
  ASSERT_TRUE(wait_for_launch(fragment)) << "Fragment did not launch within timeout";

  for (int iter = 0; iter < num_iterations; ++iter) {
    fragment.gpu_resident().data_ready();
    ASSERT_TRUE(wait_for_result(fragment)) << "Result not ready at iteration " << iter;
  }

  verify_sink_data<N>(g.sink, chain_length);
  teardown_fragment(fragment, future);
}

TEST_F(GPUResidentMultiIOTest, MultipleIterations_1Port_Chain1_10Iters) {
  test_multiple_iterations<1>(1, 10);
}

TEST_F(GPUResidentMultiIOTest, MultipleIterations_3Ports_Chain1_10Iters) {
  test_multiple_iterations<3>(1, 10);
}

TEST_F(GPUResidentMultiIOTest, MultipleIterations_5Ports_Chain3_5Iters) {
  test_multiple_iterations<5>(3, 5);
}

TEST_F(GPUResidentMultiIOTest, MultipleIterations_10Ports_Chain1_3Iters) {
  test_multiple_iterations<10>(1, 3);
}

TEST_F(GPUResidentMultiIOTest, MultipleIterations_20Ports_Chain0_5Iters) {
  test_multiple_iterations<20>(0, 5);
}

// ================================================================================================
// Section 6: Fan-out with mixed connection types
// ================================================================================================

TEST_F(GPUResidentMultiIOTest, FanOut_DevicePtrSource_TwoMemorySinks) {
  Fragment fragment;
  auto source = fragment.make_operator<DevicePtrSourceOp>("source");
  auto sink1 = fragment.make_operator<TestSinkGpuOp>("sink1");
  auto sink2 = fragment.make_operator<TestSinkGpuOp>("sink2");

  fragment.add_flow(source, sink1);
  fragment.add_flow(source, sink2);

  auto executor = std::dynamic_pointer_cast<GPUResidentExecutor>(fragment.executor_shared());
  ASSERT_NE(executor, nullptr);
  EXPECT_TRUE(executor->initialize_fragment());

  auto source_out = source->device_memory("out");
  auto sink1_in = sink1->device_memory("in");
  auto sink2_in = sink2->device_memory("in");

  EXPECT_EQ(source_out, source->dev_ptr());
  EXPECT_EQ(source_out, sink1_in);
  EXPECT_EQ(source_out, sink2_in);
}

TEST_F(GPUResidentMultiIOTest, FanOut_DevicePtrSource_MemoryAndDevicePtrSinks) {
  Fragment fragment;
  auto source = fragment.make_operator<DevicePtrSourceOp>("source");
  auto sink_mem = fragment.make_operator<TestSinkGpuOp>("sink_mem");
  auto sink_ptr = fragment.make_operator<DevicePtrSinkOp>("sink_ptr");

  fragment.add_flow(source, sink_mem);
  fragment.add_flow(source, sink_ptr);

  auto executor = std::dynamic_pointer_cast<GPUResidentExecutor>(fragment.executor_shared());
  ASSERT_NE(executor, nullptr);

  testing::internal::CaptureStderr();
  EXPECT_TRUE(executor->initialize_fragment());
  std::string log_output = testing::internal::GetCapturedStderr();

  EXPECT_TRUE(log_output.find("Both source") != std::string::npos &&
              log_output.find("have device pointers") != std::string::npos)
      << "Expected error about both sides having device pointers not found in:\n"
      << log_output;

  auto source_out = source->device_memory("out");
  auto sink_mem_in = sink_mem->device_memory("in");
  auto sink_ptr_in = sink_ptr->device_memory("in");

  EXPECT_EQ(source_out, source->dev_ptr());
  EXPECT_EQ(source_out, sink_mem_in);
  EXPECT_EQ(source_out, sink_ptr_in);
  EXPECT_NE(sink_ptr_in, sink_ptr->dev_ptr());
}

TEST_F(GPUResidentMultiIOTest, FanOut_DevicePtrSource_TwoDevicePtrSinks) {
  Fragment fragment;
  auto source = fragment.make_operator<DevicePtrSourceOp>("source");
  auto sink_ptr1 = fragment.make_operator<DevicePtrSinkOp>("sink_ptr1");
  auto sink_ptr2 = fragment.make_operator<DevicePtrSinkOp>("sink_ptr2");

  fragment.add_flow(source, sink_ptr1);
  fragment.add_flow(source, sink_ptr2);

  auto executor = std::dynamic_pointer_cast<GPUResidentExecutor>(fragment.executor_shared());
  ASSERT_NE(executor, nullptr);

  testing::internal::CaptureStderr();
  EXPECT_TRUE(executor->initialize_fragment());
  std::string log_output = testing::internal::GetCapturedStderr();

  size_t first_match = log_output.find("Both source");
  ASSERT_NE(first_match, std::string::npos)
      << "Expected first error about both sides having device pointers not found in:\n"
      << log_output;
  EXPECT_NE(log_output.find("Both source", first_match + 1), std::string::npos)
      << "Expected second error about both sides having device pointers not found in:\n"
      << log_output;
  EXPECT_TRUE(log_output.find("have device pointers") != std::string::npos)
      << "Expected error about both sides having device pointers not found in:\n"
      << log_output;

  auto source_out = source->device_memory("out");
  auto sink_ptr1_in = sink_ptr1->device_memory("in");
  auto sink_ptr2_in = sink_ptr2->device_memory("in");

  EXPECT_EQ(source_out, source->dev_ptr());
  EXPECT_EQ(source_out, sink_ptr1_in);
  EXPECT_EQ(source_out, sink_ptr2_in);
  EXPECT_NE(sink_ptr1_in, sink_ptr1->dev_ptr());
  EXPECT_NE(sink_ptr2_in, sink_ptr2->dev_ptr());
}

TEST_F(GPUResidentMultiIOTest, FanOut_NoneSource_TwoMemorySinks) {
  Fragment fragment;
  auto source = fragment.make_operator<ZeroSizeOutputMemoryOp>("source");
  auto sink1 = fragment.make_operator<TestSinkGpuOp>("sink1");
  auto sink2 = fragment.make_operator<TestSinkGpuOp>("sink2");

  fragment.add_flow(source, sink1);
  fragment.add_flow(source, sink2);

  auto executor = std::dynamic_pointer_cast<GPUResidentExecutor>(fragment.executor_shared());
  ASSERT_NE(executor, nullptr);
  EXPECT_TRUE(executor->initialize_fragment());

  auto source_out = source->device_memory("out");
  auto sink1_in = sink1->device_memory("in");
  auto sink2_in = sink2->device_memory("in");

  EXPECT_NE(source_out, nullptr);
  EXPECT_EQ(source_out, sink1_in);
  EXPECT_EQ(source_out, sink2_in);
}

TEST_F(GPUResidentMultiIOTest, ErrorFanOut_NoneSource_DevicePtrThenMemorySink) {
  Fragment fragment;
  auto source = fragment.make_operator<ZeroSizeOutputMemoryOp>("source");
  auto sink_ptr = fragment.make_operator<DevicePtrSinkOp>("sink_ptr");
  auto sink_mem = fragment.make_operator<TestSinkGpuOp>("sink_mem");

  fragment.add_flow(source, sink_ptr);
  fragment.add_flow(source, sink_mem);

  auto executor = std::dynamic_pointer_cast<GPUResidentExecutor>(fragment.executor_shared());
  ASSERT_NE(executor, nullptr);

  std::string msg;
  try {
    executor->initialize_fragment();
  } catch (const std::runtime_error& e) {
    msg = e.what();
  }

  EXPECT_FALSE(msg.empty()) << "Expected std::runtime_error to be thrown";
  EXPECT_TRUE(msg.find("already connected through externally managed device pointers") !=
              std::string::npos)
      << "Expected message about source port already bound to externally managed device pointers, "
         "got: "
      << msg;
}

TEST_F(GPUResidentMultiIOTest, ErrorFanOut_NoneSource_MemoryThenDevicePtrSink) {
  Fragment fragment;
  auto source = fragment.make_operator<ZeroSizeOutputMemoryOp>("source");
  auto sink_mem = fragment.make_operator<TestSinkGpuOp>("sink_mem");
  auto sink_ptr = fragment.make_operator<DevicePtrSinkOp>("sink_ptr");

  fragment.add_flow(source, sink_mem);
  fragment.add_flow(source, sink_ptr);

  auto executor = std::dynamic_pointer_cast<GPUResidentExecutor>(fragment.executor_shared());
  ASSERT_NE(executor, nullptr);

  std::string msg;
  try {
    executor->initialize_fragment();
  } catch (const std::runtime_error& e) {
    msg = e.what();
  }

  EXPECT_FALSE(msg.empty()) << "Expected std::runtime_error to be thrown";
  EXPECT_TRUE(msg.find("already connected through executor-allocated device buffers") !=
              std::string::npos)
      << "Expected message about source port already bound to executor-allocated device buffers, "
         "got: "
      << msg;
}

TEST_F(GPUResidentMultiIOTest, ErrorFanOut_NoneSource_TwoDevicePtrSinks) {
  Fragment fragment;
  auto source = fragment.make_operator<ZeroSizeOutputMemoryOp>("source");
  auto sink_ptr1 = fragment.make_operator<DevicePtrSinkOp>("sink_ptr1");
  auto sink_ptr2 = fragment.make_operator<DevicePtrSinkOp>("sink_ptr2");

  fragment.add_flow(source, sink_ptr1);
  fragment.add_flow(source, sink_ptr2);

  auto executor = std::dynamic_pointer_cast<GPUResidentExecutor>(fragment.executor_shared());
  ASSERT_NE(executor, nullptr);

  std::string msg;
  try {
    executor->initialize_fragment();
  } catch (const std::runtime_error& e) {
    msg = e.what();
  }

  EXPECT_FALSE(msg.empty()) << "Expected std::runtime_error to be thrown";
  EXPECT_TRUE(msg.find("already connected to a different externally managed device pointer") !=
              std::string::npos)
      << "Expected message about conflicting externally managed device pointers, got: " << msg;
}

TEST_F(GPUResidentMultiIOTest, ErrorFanOut_MemorySource_MemoryThenDevicePtrSink) {
  Fragment fragment;
  auto source = fragment.make_operator<TestSourceGpuOp>("source");
  auto sink_mem = fragment.make_operator<TestSinkGpuOp>("sink_mem");
  auto sink_ptr = fragment.make_operator<DevicePtrSinkOp>("sink_ptr");

  fragment.add_flow(source, sink_mem);
  fragment.add_flow(source, sink_ptr);

  auto executor = std::dynamic_pointer_cast<GPUResidentExecutor>(fragment.executor_shared());
  ASSERT_NE(executor, nullptr);

  std::string msg;
  try {
    executor->initialize_fragment();
  } catch (const std::runtime_error& e) {
    msg = e.what();
  }

  EXPECT_FALSE(msg.empty()) << "Expected std::runtime_error to be thrown";
  EXPECT_TRUE(msg.find("already connected through executor-allocated device buffers") !=
              std::string::npos)
      << "Expected message about conflicting buffer/device-pointer connection, got: " << msg;
}

TEST_F(GPUResidentMultiIOTest, ErrorFanOut_MemorySource_TwoDevicePtrSinks) {
  Fragment fragment;
  auto source = fragment.make_operator<TestSourceGpuOp>("source");
  auto sink_ptr1 = fragment.make_operator<DevicePtrSinkOp>("sink_ptr1");
  auto sink_ptr2 = fragment.make_operator<DevicePtrSinkOp>("sink_ptr2");

  fragment.add_flow(source, sink_ptr1);
  fragment.add_flow(source, sink_ptr2);

  auto executor = std::dynamic_pointer_cast<GPUResidentExecutor>(fragment.executor_shared());
  ASSERT_NE(executor, nullptr);

  std::string msg;
  try {
    executor->initialize_fragment();
  } catch (const std::runtime_error& e) {
    msg = e.what();
  }

  EXPECT_FALSE(msg.empty()) << "Expected std::runtime_error to be thrown";
  EXPECT_TRUE(msg.find("already connected to a different externally managed device pointer") !=
              std::string::npos)
      << "Expected message about conflicting externally managed device pointers, got: " << msg;
}

// ================================================================================================
// Section 7: Error Conditions Specific to Multi-IO
// ================================================================================================

// Mismatched memory block size on one port out of three.
// Source: out0=128B, out1=256B, out2=128B.  Sink: in0=128B, in1=128B, in2=128B.
// Port 1 sizes differ (256 vs 128) -> connect_ports must throw for that pair.
TEST_F(GPUResidentMultiIOTest, ErrorMismatchedSizeOnOnePort) {
  Fragment fragment;
  auto source = fragment.make_operator<MismatchedMultiPortSourceOp>("source");
  auto sink = fragment.make_operator<MultiInputSinkGpuOp<3>>("sink");
  fragment.add_flow(source, sink, {{"out0", "in0"}, {"out1", "in1"}, {"out2", "in2"}});

  auto executor = std::dynamic_pointer_cast<GPUResidentExecutor>(fragment.executor_shared());
  ASSERT_NE(executor, nullptr);
  EXPECT_THROW(executor->initialize_fragment(), std::runtime_error);
}

// Indegree > 1 with multi-IO: two sources connecting to the same sink input
// across separate add_flow calls must be rejected.
TEST_F(GPUResidentMultiIOTest, ErrorIndegreeViolationMultiIO) {
  Fragment fragment;
  auto source1 = fragment.make_operator<MultiOutputSourceGpuOp<2>>("source1");
  auto source2 = fragment.make_operator<MultiOutputSourceGpuOp<2>>("source2");
  auto sink = fragment.make_operator<MultiInputSinkGpuOp<2>>("sink");

  fragment.add_flow(source1, sink, {{"out0", "in0"}, {"out1", "in1"}});

  EXPECT_THROW(fragment.add_flow(source2, sink, {{"out0", "in0"}}), holoscan::RuntimeError);
}

}  // namespace holoscan
