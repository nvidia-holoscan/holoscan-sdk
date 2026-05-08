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

#include <array>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <future>
#include <memory>
#include <thread>

#include <holoscan/core/gpu_resident_operator.hpp>
#include <holoscan/holoscan.hpp>
#include <holoscan/utils/cuda_macros.hpp>

#include "diamond_kernels.cu.hpp"

namespace {

constexpr size_t kElementCount = 512;
constexpr int kAddConstant = 5;
constexpr int kSubtractConstant = 3;
constexpr int kIterations = 10;

}  // namespace

namespace holoscan::ops {

class SourceGpuOp : public holoscan::GPUResidentOperator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS_SUPER(SourceGpuOp, holoscan::GPUResidentOperator)

  SourceGpuOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.device_output("out", sizeof(int) * kElementCount);
  }

  void compute([[maybe_unused]] holoscan::InputContext& op_input,
               [[maybe_unused]] holoscan::OutputContext& op_output,
               [[maybe_unused]] holoscan::ExecutionContext& context) override {
    HOLOSCAN_LOG_INFO("SourceGpuOp::compute() -- {} -- waiting for host input", name());
  }
};

class AddGpuOp : public holoscan::GPUResidentOperator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS_SUPER(AddGpuOp, holoscan::GPUResidentOperator)

  AddGpuOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.device_input("in", sizeof(int) * kElementCount);
    spec.device_output("out", sizeof(int) * kElementCount);
  }

  void compute([[maybe_unused]] holoscan::InputContext& op_input,
               [[maybe_unused]] holoscan::OutputContext& op_output,
               [[maybe_unused]] holoscan::ExecutionContext& context) override {
    auto* input = static_cast<int*>(device_memory("in"));
    auto* output = static_cast<int*>(device_memory("out"));
    if (input == nullptr || output == nullptr) {
      HOLOSCAN_LOG_ERROR("AddGpuOp::compute() -- {} -- invalid input/output pointers", name());
      return;
    }

    launch_add_constant_kernel(input, output, kAddConstant, kElementCount, *cuda_stream());
  }
};

class SubtractGpuOp : public holoscan::GPUResidentOperator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS_SUPER(SubtractGpuOp, holoscan::GPUResidentOperator)

  SubtractGpuOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.device_input("in", sizeof(int) * kElementCount);
    spec.device_output("out", sizeof(int) * kElementCount);
  }

  void compute([[maybe_unused]] holoscan::InputContext& op_input,
               [[maybe_unused]] holoscan::OutputContext& op_output,
               [[maybe_unused]] holoscan::ExecutionContext& context) override {
    auto* input = static_cast<int*>(device_memory("in"));
    auto* output = static_cast<int*>(device_memory("out"));
    if (input == nullptr || output == nullptr) {
      HOLOSCAN_LOG_ERROR("SubtractGpuOp::compute() -- {} -- invalid input/output pointers", name());
      return;
    }

    launch_subtract_constant_kernel(
        input, output, kSubtractConstant, kElementCount, *cuda_stream());
  }
};

class MultiplySinkGpuOp : public holoscan::GPUResidentOperator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS_SUPER(MultiplySinkGpuOp, holoscan::GPUResidentOperator)

  MultiplySinkGpuOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.device_input("add_in", sizeof(int) * kElementCount);
    spec.device_input("subtract_in", sizeof(int) * kElementCount);
  }

  void compute([[maybe_unused]] holoscan::InputContext& op_input,
               [[maybe_unused]] holoscan::OutputContext& op_output,
               [[maybe_unused]] holoscan::ExecutionContext& context) override {
    auto* add_input = static_cast<int*>(device_memory("add_in"));
    auto* subtract_input = static_cast<int*>(device_memory("subtract_in"));
    if (add_input == nullptr || subtract_input == nullptr) {
      HOLOSCAN_LOG_ERROR("MultiplySinkGpuOp::compute() -- {} -- invalid input pointers", name());
      return;
    }

    // The sink reuses add_in as the destination buffer so the host can read the final result back
    // from one of the sink inputs after each iteration completes.
    launch_multiply_kernel(add_input, subtract_input, add_input, kElementCount, *cuda_stream());
  }
};

}  // namespace holoscan::ops

class DiamondGpuResidentFragment : public holoscan::Fragment {
 public:
  void compose() override {
    using namespace holoscan;

    auto source_op = make_operator<ops::SourceGpuOp>("source_op");
    auto add_op = make_operator<ops::AddGpuOp>("add_op");
    auto subtract_op = make_operator<ops::SubtractGpuOp>("subtract_op");
    auto multiply_sink_op = make_operator<ops::MultiplySinkGpuOp>("multiply_sink_op");

    // Exercise DAG support with source fan-out and sink fan-in.
    add_flow(source_op, add_op);
    add_flow(source_op, subtract_op);
    add_flow(add_op, multiply_sink_op, {{"out", "add_in"}});
    add_flow(subtract_op, multiply_sink_op, {{"out", "subtract_in"}});
  }
};

class DiamondGpuResidentApp : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;

    auto diamond_fragment = make_fragment<DiamondGpuResidentFragment>("diamond_fragment");
    add_fragment(diamond_fragment);
  }
};

namespace {

void cleanup_and_wait(const std::shared_ptr<holoscan::Fragment>& gr_fragment,
                      std::future<void>& future) {
  if (gr_fragment && gr_fragment->gpu_resident().is_launched()) {
    gr_fragment->gpu_resident().tear_down();
  }
  if (future.valid()) {
    future.get();
  }
}

bool wait_for_graph_launch(const std::shared_ptr<holoscan::Fragment>& gr_fragment) {
  HOLOSCAN_LOG_INFO("Waiting for GPU-resident DAG to launch...");
  auto start_time = std::chrono::steady_clock::now();

  while (!gr_fragment->gpu_resident().is_launched()) {
    if (std::chrono::steady_clock::now() - start_time >= std::chrono::seconds(5)) {
      HOLOSCAN_LOG_ERROR("Timeout: GPU-resident DAG did not launch within 5 seconds");
      return false;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  HOLOSCAN_LOG_INFO("GPU-resident DAG launched");
  return true;
}

bool get_gpu_resident_operators(const std::shared_ptr<holoscan::Fragment>& gr_fragment,
                                holoscan::GPUResidentOperator*& source_op,
                                holoscan::GPUResidentOperator*& sink_op) {
  auto graph = gr_fragment->graph_shared();
  auto source_node = graph->find_node("source_op");
  auto sink_node = graph->find_node("multiply_sink_op");

  source_op =
      source_node ? dynamic_cast<holoscan::GPUResidentOperator*>(source_node.get()) : nullptr;
  sink_op = sink_node ? dynamic_cast<holoscan::GPUResidentOperator*>(sink_node.get()) : nullptr;

  if (source_op == nullptr || sink_op == nullptr) {
    HOLOSCAN_LOG_ERROR("Could not find source_op or multiply_sink_op");
    return false;
  }

  return true;
}

bool wait_for_result(const std::shared_ptr<holoscan::Fragment>& gr_fragment, int iteration) {
  constexpr int kMaxChecks = 25;

  for (int check = 0; check < kMaxChecks; ++check) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    if (gr_fragment->gpu_resident().result_ready()) {
      HOLOSCAN_LOG_INFO("Iteration {} - result is ready after {} checks", iteration + 1, check + 1);
      return true;
    }
  }

  HOLOSCAN_LOG_ERROR("Iteration {} - result did not become ready", iteration + 1);
  return false;
}

bool verify_results(const std::array<int, kElementCount>& host_input,
                    const std::array<int, kElementCount>& host_result, int iteration) {
  for (size_t i = 0; i < kElementCount; ++i) {
    int expected = (host_input.at(i) + kAddConstant) * (host_input.at(i) - kSubtractConstant);
    if (host_result.at(i) != expected) {
      HOLOSCAN_LOG_ERROR("Iteration {} - Incorrect result at index {}: expected {}, got {}",
                         iteration + 1,
                         i,
                         expected,
                         host_result.at(i));
      return false;
    }
  }

  HOLOSCAN_LOG_INFO("Iteration {} - All diamond DAG results are correct", iteration + 1);
  return true;
}

bool run_iteration(holoscan::GPUResidentOperator* source_op, holoscan::GPUResidentOperator* sink_op,
                   const std::shared_ptr<holoscan::Fragment>& gr_fragment, unsigned int& seed,
                   int iteration) {
  std::array<int, kElementCount> host_input{};
  for (int& value : host_input) {
    value = rand_r(&seed) % 1000 + 10;
  }

  void* source_output = source_op->device_memory("out");
  if (source_output == nullptr) {
    HOLOSCAN_LOG_ERROR("Could not find source operator output device memory");
    return false;
  }

  HOLOSCAN_CUDA_CALL_THROW_ERROR(
      cudaMemcpy(
          source_output, host_input.data(), sizeof(int) * kElementCount, cudaMemcpyHostToDevice),
      "Failed to copy host input to source operator output");
  HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaStreamSynchronize(0), "Failed to synchronize default stream");

  gr_fragment->gpu_resident().data_ready();

  if (!wait_for_result(gr_fragment, iteration)) {
    return false;
  }

  void* sink_result = sink_op->device_memory("add_in");
  if (sink_result == nullptr) {
    HOLOSCAN_LOG_ERROR("Could not find sink operator result device memory");
    return false;
  }

  std::array<int, kElementCount> host_result{};
  HOLOSCAN_CUDA_CALL_THROW_ERROR(
      cudaMemcpy(
          host_result.data(), sink_result, sizeof(int) * kElementCount, cudaMemcpyDeviceToHost),
      "Failed to copy sink result back to host");
  HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaStreamSynchronize(0), "Failed to synchronize default stream");

  return verify_results(host_input, host_result, iteration);
}

}  // namespace

int main() {
  auto app = holoscan::make_application<DiamondGpuResidentApp>();
  app->compose_graph();

  auto& fragment_graph = app->fragment_graph();
  auto gr_fragment = fragment_graph.find_node("diamond_fragment");
  if (!gr_fragment) {
    HOLOSCAN_LOG_ERROR("Could not find diamond_fragment");
    return 1;
  }

  gr_fragment->compose_graph();

  // This sample reads the sink result back to host memory after each iteration.
  gr_fragment->gpu_resident().sync_with_host();

  auto future = app->run_async();
  if (!wait_for_graph_launch(gr_fragment)) {
    cleanup_and_wait(gr_fragment, future);
    return 1;
  }

  holoscan::GPUResidentOperator* source_op = nullptr;
  holoscan::GPUResidentOperator* sink_op = nullptr;
  if (!get_gpu_resident_operators(gr_fragment, source_op, sink_op)) {
    cleanup_and_wait(gr_fragment, future);
    return 1;
  }

  unsigned int seed = static_cast<unsigned int>(std::time(nullptr));
  bool success = true;
  for (int iteration = 0; iteration < kIterations; ++iteration) {
    if (!run_iteration(source_op, sink_op, gr_fragment, seed, iteration)) {
      success = false;
      break;
    }
  }

  cleanup_and_wait(gr_fragment, future);

  if (!success) {
    return 1;
  }

  HOLOSCAN_LOG_INFO("Diamond GPU-resident DAG example completed");
  return 0;
}
