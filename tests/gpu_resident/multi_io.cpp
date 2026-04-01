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

  // Distinct port indices must have distinct addresses on the source
  for (int i = 0; i < N; ++i) {
    for (int j = i + 1; j < N; ++j) {
      EXPECT_NE(g.source->device_memory("out" + std::to_string(i)),
                g.source->device_memory("out" + std::to_string(j)))
          << "source: out" << i << " and out" << j << " must not share memory";
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

  // Distinct port indices on the sink must have distinct addresses
  for (int i = 0; i < N; ++i) {
    for (int j = i + 1; j < N; ++j) {
      EXPECT_NE(g.sink->device_memory("in" + std::to_string(i)),
                g.sink->device_memory("in" + std::to_string(j)))
          << "sink: in" << i << " and in" << j << " must not share memory";
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
// Section 6: Error Conditions Specific to Multi-IO
// ================================================================================================

// Fan-out: one source port mapped to two destination ports must be rejected.
// The add_flow call may succeed (it checks indegree per dest port, not source
// fan-out), but prepare_data_flow / initialize_fragment must throw because
// GPU-resident requires exactly 1 destination per source port.
TEST_F(GPUResidentMultiIOTest, ErrorFanOutSingleCall) {
  Fragment fragment;
  auto source = fragment.make_operator<MultiOutputSourceGpuOp<1>>("source");
  auto sink = fragment.make_operator<MultiInputSinkGpuOp<2>>("sink");

  std::set<std::pair<std::string, std::string>> pm{{"out0", "in0"}, {"out0", "in1"}};

  bool threw_at_add_flow = false;
  try {
    fragment.add_flow(source, sink, pm);
  } catch (const std::exception&) {
    threw_at_add_flow = true;
  }

  if (!threw_at_add_flow) {
    auto executor = std::dynamic_pointer_cast<GPUResidentExecutor>(fragment.executor_shared());
    ASSERT_NE(executor, nullptr);
    EXPECT_THROW(executor->initialize_fragment(), std::runtime_error);
  }
}

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
