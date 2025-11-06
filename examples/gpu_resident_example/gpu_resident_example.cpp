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

#include <array>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <memory>
#include <thread>

#include <holoscan/core/gpu_resident_operator.hpp>
#include <holoscan/holoscan.hpp>
#include <holoscan/operators/ping_rx/ping_rx.hpp>
#include <holoscan/operators/ping_tx/ping_tx.hpp>
#include <holoscan/utils/cuda_macros.hpp>
#include "add_kernel.cu.hpp"

namespace holoscan::ops {

// SourceGpuOp: Only has output port, does not do anything in compute
class SourceGpuOp : public holoscan::GPUResidentOperator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS_SUPER(SourceGpuOp, holoscan::GPUResidentOperator)

  SourceGpuOp() = default;

  void setup(OperatorSpec& spec) override { spec.device_output("out", sizeof(int) * 512); }

  void compute([[maybe_unused]] holoscan::InputContext& op_input,
               [[maybe_unused]] holoscan::OutputContext& op_output,
               [[maybe_unused]] holoscan::ExecutionContext& context) override {
    HOLOSCAN_LOG_INFO("SourceGpuOp::compute() -- {} -- No computation performed", name());
  }
};

// ComputeGpuOp: Has both input and output ports, launches add_five_kernel in compute
class ComputeGpuOp : public holoscan::GPUResidentOperator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS_SUPER(ComputeGpuOp, holoscan::GPUResidentOperator)

  ComputeGpuOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.device_input("in", sizeof(int) * 512);
    spec.device_output("out", sizeof(int) * 512);
  }

  void compute([[maybe_unused]] holoscan::InputContext& op_input,
               [[maybe_unused]] holoscan::OutputContext& op_output,
               [[maybe_unused]] holoscan::ExecutionContext& context) override {
    auto* in_device_address = device_memory("in");
    auto* out_device_address = device_memory("out");

    HOLOSCAN_LOG_INFO("ComputeGpuOp::compute() -- {} -- in: {} out: {}",
                      name(),
                      in_device_address,
                      out_device_address);

    // Launch CUDA kernel if both input and output addresses are valid
    if (in_device_address != nullptr && out_device_address != nullptr) {
      // Get the CUDA stream for this operator
      auto stream_ptr = cuda_stream();
      cudaStream_t stream = *stream_ptr;

      // Launch the add five kernel
      launch_add_five_kernel(
          static_cast<int*>(in_device_address), static_cast<int*>(out_device_address), 512, stream);

      HOLOSCAN_LOG_INFO("ComputeGpuOp::compute() -- {} -- Launched add_five_kernel", name());
    }
  }
};

// SinkGpuOp: Only has input port, consumes the input
class SinkGpuOp : public holoscan::GPUResidentOperator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS_SUPER(SinkGpuOp, holoscan::GPUResidentOperator)

  SinkGpuOp() = default;

  void setup(OperatorSpec& spec) override { spec.device_input("in", sizeof(int) * 512); }

  void compute([[maybe_unused]] holoscan::InputContext& op_input,
               [[maybe_unused]] holoscan::OutputContext& op_output,
               [[maybe_unused]] holoscan::ExecutionContext& context) override {
    auto* in_device_address = device_memory("in");

    HOLOSCAN_LOG_INFO(
        "SinkGpuOp::compute() -- {} -- Consuming input at: {}", name(), in_device_address);
  }
};

}  // namespace holoscan::ops

class GpuResidentFragment : public holoscan::Fragment {
 public:
  void compose() override {
    using namespace holoscan;

    auto source_op = make_operator<ops::SourceGpuOp>("source_op");
    auto compute_op1 = make_operator<ops::ComputeGpuOp>("compute_op1");
    auto compute_op2 = make_operator<ops::ComputeGpuOp>("compute_op2");
    auto compute_op3 = make_operator<ops::ComputeGpuOp>("compute_op3");
    auto sink_op = make_operator<ops::SinkGpuOp>("sink_op");

    add_flow(source_op, compute_op1);
    add_flow(compute_op1, compute_op2);
    add_flow(compute_op2, compute_op3);
    add_flow(compute_op3, sink_op);
  }
};

class NormalFragment : public holoscan::Fragment {
 public:
  void compose() override {
    using namespace holoscan;

    // add tx and rx
    auto tx = make_operator<ops::PingTxOp>("tx", make_condition<CountCondition>(10));
    auto rx = make_operator<ops::PingRxOp>("rx");
    add_flow(tx, rx);
  }
};

class GRApp : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;

    auto gr_fragment = make_fragment<GpuResidentFragment>("gr_fragment");
    auto normal_fragment = make_fragment<NormalFragment>("normal_fragment");

    add_fragment(gr_fragment);
    add_fragment(normal_fragment);
  }
};

// Cleanup function that sends tear down command and calls future.get()
int cleanup_and_exit(const std::shared_ptr<holoscan::Fragment>& gr_fragment,
                     std::future<void>& future) {
  if (gr_fragment) {
    gr_fragment->gpu_resident().tear_down();
  }
  future.get();
  return 1;
}

// Helper function to wait for the GPU-resident CUDA graph to be launched
bool wait_for_graph_launch(const std::shared_ptr<holoscan::Fragment>& gr_fragment) {
  HOLOSCAN_LOG_INFO("Waiting for GPU-resident CUDA graph to be launched...");
  auto start_time = std::chrono::steady_clock::now();

  while (true) {
    if (gr_fragment->gpu_resident().is_launched()) {
      HOLOSCAN_LOG_INFO("GPU-resident CUDA graph has been launched!");
      return true;
    }

    auto elapsed = std::chrono::steady_clock::now() - start_time;
    if (elapsed >= std::chrono::seconds(5)) {
      HOLOSCAN_LOG_ERROR("Timeout: GPU-resident CUDA graph was not launched within 5 seconds");
      return false;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
}

// Helper function to get GPU resident operators from the fragment
bool get_gpu_resident_operators(const std::shared_ptr<holoscan::Fragment>& gr_fragment,
                                holoscan::GPUResidentOperator*& source_op,
                                holoscan::GPUResidentOperator*& sink_op) {
  auto graph = gr_fragment->graph_shared();
  auto source_node = graph->find_node("source_op");
  auto sink_node = graph->find_node("sink_op");

  source_op = nullptr;
  sink_op = nullptr;

  if (source_node) {
    source_op = dynamic_cast<holoscan::GPUResidentOperator*>(source_node.get());
  }
  if (sink_node) {
    sink_op = dynamic_cast<holoscan::GPUResidentOperator*>(sink_node.get());
  }

  if (source_op == nullptr || sink_op == nullptr) {
    HOLOSCAN_LOG_ERROR("Could not find source_op or sink_op operators");
    return false;
  }

  return true;
}

// Helper function to wait for result to be ready
bool wait_for_result(const std::shared_ptr<holoscan::Fragment>& gr_fragment, int iteration) {
  int check_result_count = 0;
  constexpr int max_checks = 5;

  while (check_result_count < max_checks) {
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    if (gr_fragment->gpu_resident().result_ready()) {
      HOLOSCAN_LOG_INFO(
          "Iteration {} - Result is ready after {} checks", iteration + 1, check_result_count + 1);
      return true;
    }
    check_result_count++;
    HOLOSCAN_LOG_INFO(
        "Iteration {} - Result not ready, check count {}", iteration + 1, check_result_count);
  }

  HOLOSCAN_LOG_ERROR(
      "Iteration {} - Result is not ready after {} checks", iteration + 1, max_checks);
  return false;
}

// Helper function to verify computation results
bool verify_results(const std::array<int, 512>& host_input, const std::array<int, 512>& host_result,
                    int iteration) {
  for (size_t i = 0; i < 512; ++i) {
    int expected = host_input.at(i) + 15;  // Original value + 15 from 3 add_five_kernels
    if (host_result.at(i) != expected) {
      HOLOSCAN_LOG_ERROR("Iteration {} - Incorrect result at index {}: expected {}, got {}",
                         iteration + 1,
                         i,
                         expected,
                         host_result.at(i));
      return false;
    }
  }

  HOLOSCAN_LOG_INFO("Iteration {} - All results are correct! Data processed successfully.",
                    iteration + 1);
  return true;
}

// Helper function to run a single iteration of the GPU resident execution
bool run_iteration(holoscan::GPUResidentOperator* source_op, holoscan::GPUResidentOperator* sink_op,
                   const std::shared_ptr<holoscan::Fragment>& gr_fragment, unsigned int& seed,
                   int iteration) {
  HOLOSCAN_LOG_INFO("Iteration {} - Preparing random data", iteration + 1);

  // Generate random integers for this iteration
  std::array<int, 512> host_input{};
  for (int& value : host_input) {
    value = rand_r(&seed) % 1000 + 1;  // Random number between 1 and 1000
  }

  // Get device memory address from the source operator's output
  void* source_output_device_addr = source_op->device_memory("out");
  if (source_output_device_addr == nullptr) {
    HOLOSCAN_LOG_ERROR("Could not find source operator output device memory");
    return false;
  }

  // Copy random data from host to device
  HOLOSCAN_CUDA_CALL_THROW_ERROR(
      cudaMemcpy(
          source_output_device_addr, host_input.data(), sizeof(int) * 512, cudaMemcpyHostToDevice),
      "Failed to copy random numbers from host to source operator output device memory");
  HOLOSCAN_LOG_INFO("Iteration {} - Copied random data to source operator output", iteration + 1);

  // Make data ready for GPU resident execution
  gr_fragment->gpu_resident().data_ready();

  // Wait for result to be ready
  if (!wait_for_result(gr_fragment, iteration)) {
    return false;
  }

  // Get the input from the sink operator (which should contain the processed data)
  void* sink_input_device_addr = sink_op->device_memory("in");
  if (sink_input_device_addr == nullptr) {
    HOLOSCAN_LOG_ERROR("Could not find sink operator input device memory");
    return false;
  }

  // Copy result back to host for verification
  std::array<int, 512> host_result{};
  HOLOSCAN_CUDA_CALL_THROW_ERROR(
      cudaMemcpy(
          host_result.data(), sink_input_device_addr, sizeof(int) * 512, cudaMemcpyDeviceToHost),
      "Failed to copy result from sink operator input device memory to host");

  // Verify the results
  return verify_results(host_input, host_result, iteration);
}

int main() {
  auto app = holoscan::make_application<GRApp>();
  auto future = app->run_async();

  // Get the GPU resident fragment from the application
  auto& fragment_graph = app->fragment_graph();
  auto gr_fragment = fragment_graph.find_node("gr_fragment");

  if (!gr_fragment) {
    HOLOSCAN_LOG_ERROR("Could not find gr_fragment");
    return cleanup_and_exit(nullptr, future);
  }

  // Wait for the GPU-resident CUDA graph to be launched
  if (!wait_for_graph_launch(gr_fragment)) {
    return cleanup_and_exit(gr_fragment, future);
  }

  // Get the source and sink operators
  holoscan::GPUResidentOperator* source_op = nullptr;
  holoscan::GPUResidentOperator* sink_op = nullptr;
  if (!get_gpu_resident_operators(gr_fragment, source_op, sink_op)) {
    return cleanup_and_exit(gr_fragment, future);
  }

  // Run 10 iterations to test the GPU resident execution
  unsigned int seed = time(nullptr);
  for (int iteration = 0; iteration < 10; ++iteration) {
    if (!run_iteration(source_op, sink_op, gr_fragment, seed, iteration)) {
      break;
    }
  }

  HOLOSCAN_LOG_INFO("Tearing down the GPU-resident fragment");
  cleanup_and_exit(gr_fragment, future);

  return 0;
}
