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

#include <chrono>
#include <memory>
#include <thread>

#include <holoscan/core/gpu_resident_operator.hpp>
#include <holoscan/holoscan.hpp>
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
        "SinkGpuOp::compute() -- {} -- Verifying results at: {}", name(), in_device_address);

    // Launch verification kernel if input address is valid
    if (in_device_address != nullptr) {
      // Get the CUDA stream for this operator
      auto stream_ptr = cuda_stream();
      cudaStream_t stream = *stream_ptr;

      // Launch the verification kernel
      // Expected: input was (idx + 1), after 3 add_five kernels: (idx + 1) + 15 = idx + 16
      launch_verify_results_kernel(static_cast<int*>(in_device_address), 512, stream);

      HOLOSCAN_LOG_INFO("SinkGpuOp::compute() -- {} -- Launched verification kernel", name());
    }
  }
};

class InputGpuOp : public holoscan::GPUResidentOperator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS_SUPER(InputGpuOp, holoscan::GPUResidentOperator)

  InputGpuOp() = default;

  void setup([[maybe_unused]] OperatorSpec& spec) override {}

  void set_source_operator(std::shared_ptr<holoscan::GPUResidentOperator> source_op) {
    source_op_ = source_op;
  }

  void compute([[maybe_unused]] holoscan::InputContext& op_input,
               [[maybe_unused]] holoscan::OutputContext& op_output,
               [[maybe_unused]] holoscan::ExecutionContext& context) override {
    HOLOSCAN_LOG_INFO("InputGpuOp::compute() -- {}", name());

    // Get the data ready device address
    auto* data_ready_addr = static_cast<unsigned int*>(data_ready_device_address());

    if (data_ready_addr == nullptr) {
      HOLOSCAN_LOG_ERROR("InputGpuOp::compute() -- {} -- data_ready_device_address is null",
                         name());
      return;
    }

    if (source_op_ == nullptr) {
      HOLOSCAN_LOG_ERROR("InputGpuOp::compute() -- {} -- source_op is null", name());
      return;
    }

    // Get the device memory from the source operator's output port directly
    auto* source_output_device_addr = source_op_->device_memory("out");

    if (source_output_device_addr == nullptr) {
      HOLOSCAN_LOG_ERROR("InputGpuOp::compute() -- {} -- source_output_device_addr is null",
                         name());
      return;
    }

    HOLOSCAN_LOG_INFO("InputGpuOp::compute() -- {} -- data_ready_address: {}, source_output: {}",
                      name(),
                      static_cast<void*>(data_ready_addr),
                      source_output_device_addr);

    auto stream_ptr = data_ready_handler_cuda_stream();
    cudaStream_t stream = *stream_ptr;

    // Launch the data ready handler kernel to generate random data and mark data as ready
    launch_data_ready_handler_kernel(
        data_ready_addr, static_cast<int*>(source_output_device_addr), 512, stream);

    HOLOSCAN_LOG_INFO("InputGpuOp::compute() -- {} -- Launched data_ready_handler_kernel", name());
  }

 private:
  std::shared_ptr<holoscan::GPUResidentOperator> source_op_;
};

}  // namespace holoscan::ops

class DataReadyGpuResidentFragment : public holoscan::Fragment {
 public:
  void compose() override {
    using namespace holoscan;

    auto input_op = make_operator<ops::InputGpuOp>("input_op");

    add_operator(input_op);
  }
};

class GpuResidentApplication : public holoscan::Application {
 public:
  void setup_data_ready_handler(std::shared_ptr<holoscan::GPUResidentOperator> source_op) {
    // make the data ready fragment
    auto data_ready_fragment = make_fragment<DataReadyGpuResidentFragment>("data_ready_fragment");

    data_ready_fragment->compose_graph();

    auto data_ready_graph = data_ready_fragment->graph_shared();
    auto input_node = data_ready_graph->find_node("input_op");
    if (!input_node) {
      HOLOSCAN_LOG_ERROR("Could not find input_op in data ready handler fragment");
      return;
    }

    auto* input_op = dynamic_cast<holoscan::ops::InputGpuOp*>(input_node.get());
    if (!input_op) {
      HOLOSCAN_LOG_ERROR("Could not cast input_op to InputGpuOp");
      return;
    }

    // Configure the InputGpuOp with the source operator
    input_op->set_source_operator(source_op);
    HOLOSCAN_LOG_INFO("Configured InputGpuOp with source operator: {}", source_op->name());

    // register the data ready fragment
    gpu_resident().register_data_ready_handler(data_ready_fragment);
  }

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

    setup_data_ready_handler(source_op);
  }
};

// Cleanup function that sends tear down command and calls future.get()
int cleanup_and_exit(const std::shared_ptr<holoscan::Application>& app, std::future<void>& future) {
  if (app) {
    app->gpu_resident().tear_down();
  }
  future.get();
  return 1;
}

// Helper function to wait for the GPU-resident CUDA graph to be launched
bool wait_for_graph_launch(std::shared_ptr<holoscan::Application> app) {
  HOLOSCAN_LOG_INFO("Waiting for GPU-resident CUDA graph to be launched...");
  auto start_time = std::chrono::steady_clock::now();

  while (true) {
    if (app->gpu_resident().is_launched()) {
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

int main() {
  auto app = holoscan::make_application<GpuResidentApplication>();

  auto future = app->run_async();

  // Wait for the GPU-resident CUDA graph to be launched
  if (!wait_for_graph_launch(app)) {
    return cleanup_and_exit(app, future);
  }

  HOLOSCAN_LOG_INFO("Running the GPU resident execution for 1 second");
  std::this_thread::sleep_for(std::chrono::seconds(1));

  HOLOSCAN_LOG_INFO("Tearing down the GPU-resident fragment");
  cleanup_and_exit(app, future);
  // check if it is torn down
  while (app->gpu_resident().is_launched()) {
    HOLOSCAN_LOG_INFO("Waiting for GPU-resident fragment to be torn down");
    // publish status messages every 500ms
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
  }
  HOLOSCAN_LOG_INFO("GPU-resident fragment is torn down");

  return 0;
}
