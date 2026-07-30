/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <chrono>
#include <future>
#include <memory>
#include <thread>
#include <utility>

#include <holoscan/core/gpu_resident_operator.hpp>
#include <holoscan/holoscan.hpp>

#include "multi_io_kernels.cu.hpp"

namespace {
constexpr int kElementCount = 512;
}  // namespace

namespace holoscan::ops {

class SourceMultiOutputGpuOp : public holoscan::GPUResidentOperator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS_SUPER(SourceMultiOutputGpuOp, holoscan::GPUResidentOperator)

  SourceMultiOutputGpuOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.device_output("out0", sizeof(int) * kElementCount);
    spec.device_output("out1", sizeof(int) * kElementCount);
  }

  void compute([[maybe_unused]] holoscan::InputContext& op_input,
               [[maybe_unused]] holoscan::OutputContext& op_output,
               [[maybe_unused]] holoscan::ExecutionContext& context) override {
    auto* out0_device_address = static_cast<int*>(device_memory("out0"));
    auto* out1_device_address = static_cast<int*>(device_memory("out1"));

    if (out0_device_address == nullptr || out1_device_address == nullptr) {
      HOLOSCAN_LOG_ERROR("SourceMultiOutputGpuOp::compute() -- {} -- invalid output pointers",
                         name());
      return;
    }

    cudaStream_t stream = *cuda_stream();
    launch_source_emit_kernel(out0_device_address, out1_device_address, kElementCount, stream);
  }
};

class AddSubGpuOp : public holoscan::GPUResidentOperator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS_SUPER(AddSubGpuOp, holoscan::GPUResidentOperator)

  AddSubGpuOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.device_input("in0", sizeof(int) * kElementCount);
    spec.device_input("in1", sizeof(int) * kElementCount);
    spec.device_output("sum", sizeof(int) * kElementCount);
    spec.device_output("diff", sizeof(int) * kElementCount);
  }

  void compute([[maybe_unused]] holoscan::InputContext& op_input,
               [[maybe_unused]] holoscan::OutputContext& op_output,
               [[maybe_unused]] holoscan::ExecutionContext& context) override {
    auto* in0_device_address = static_cast<int*>(device_memory("in0"));
    auto* in1_device_address = static_cast<int*>(device_memory("in1"));
    auto* sum_device_address = static_cast<int*>(device_memory("sum"));
    auto* diff_device_address = static_cast<int*>(device_memory("diff"));

    if (in0_device_address == nullptr || in1_device_address == nullptr ||
        sum_device_address == nullptr || diff_device_address == nullptr) {
      HOLOSCAN_LOG_ERROR("AddSubGpuOp::compute() -- {} -- invalid input/output pointers", name());
      return;
    }

    cudaStream_t stream = *cuda_stream();
    launch_add_sub_kernel(in0_device_address,
                          in1_device_address,
                          sum_device_address,
                          diff_device_address,
                          kElementCount,
                          stream);
  }
};

class FinalAddGpuOp : public holoscan::GPUResidentOperator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS_SUPER(FinalAddGpuOp, holoscan::GPUResidentOperator)

  FinalAddGpuOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.device_input("in_sum", sizeof(int) * kElementCount);
    spec.device_input("in_diff", sizeof(int) * kElementCount);
  }

  void compute([[maybe_unused]] holoscan::InputContext& op_input,
               [[maybe_unused]] holoscan::OutputContext& op_output,
               [[maybe_unused]] holoscan::ExecutionContext& context) override {
    auto* in_sum_device_address = static_cast<int*>(device_memory("in_sum"));
    auto* in_diff_device_address = static_cast<int*>(device_memory("in_diff"));

    if (in_sum_device_address == nullptr || in_diff_device_address == nullptr) {
      HOLOSCAN_LOG_ERROR("FinalAddGpuOp::compute() -- {} -- invalid input pointers", name());
      return;
    }

    cudaStream_t stream = *cuda_stream();
    // Final stage has no output port; reuse in_sum as the destination buffer.
    ::launch_final_add_kernel(in_sum_device_address,
                              in_diff_device_address,
                              in_sum_device_address,
                              kElementCount,
                              stream);
  }
};

class DataReadyGpuOp : public holoscan::GPUResidentOperator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS_SUPER(DataReadyGpuOp, holoscan::GPUResidentOperator)

  DataReadyGpuOp() = default;

  void setup([[maybe_unused]] OperatorSpec& spec) override {}

  void compute([[maybe_unused]] holoscan::InputContext& op_input,
               [[maybe_unused]] holoscan::OutputContext& op_output,
               [[maybe_unused]] holoscan::ExecutionContext& context) override {
    auto* data_ready_addr = static_cast<unsigned int*>(data_ready_device_address());
    if (data_ready_addr == nullptr) {
      HOLOSCAN_LOG_ERROR("DataReadyGpuOp::compute() -- {} -- data_ready_address is null", name());
      return;
    }

    cudaStream_t stream = *data_ready_handler_cuda_stream();
    launch_mark_data_ready_kernel(data_ready_addr, stream);
  }
};

}  // namespace holoscan::ops

class DataReadyFragment : public holoscan::Fragment {
 public:
  void compose() override {
    auto data_ready_op = make_operator<holoscan::ops::DataReadyGpuOp>("data_ready_op");
    add_operator(data_ready_op);
  }
};

class MultiIoGpuResidentApp : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;

    auto source_op = make_operator<ops::SourceMultiOutputGpuOp>("source_op");
    auto add_sub_op = make_operator<ops::AddSubGpuOp>("add_sub_op");
    auto final_add_op = make_operator<ops::FinalAddGpuOp>("final_add_op");

    add_flow(source_op, add_sub_op, {{"out0", "in0"}, {"out1", "in1"}});
    add_flow(add_sub_op, final_add_op, {{"sum", "in_sum"}, {"diff", "in_diff"}});

    auto data_ready_fragment = make_fragment<DataReadyFragment>("data_ready_fragment");
    data_ready_fragment->compose_graph();
    gpu_resident().register_data_ready_handler(std::move(data_ready_fragment));
  }
};

void cleanup_and_wait(const std::shared_ptr<holoscan::Application>& app,
                      std::future<void>& future) {
  if (app && app->gpu_resident().is_launched()) {
    app->gpu_resident().tear_down();
  }
  if (future.valid()) {
    future.get();
  }
}

bool wait_for_graph_launch(const std::shared_ptr<holoscan::Application>& app,
                           std::chrono::seconds timeout) {
  auto start_time = std::chrono::steady_clock::now();
  while (!app->gpu_resident().is_launched()) {
    if (std::chrono::steady_clock::now() - start_time >= timeout) {
      HOLOSCAN_LOG_ERROR("Timeout: GPU-resident CUDA graph was not launched in {} seconds",
                         timeout.count());
      return false;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
  HOLOSCAN_LOG_INFO("GPU-resident CUDA graph launched");
  return true;
}

int main() {
  auto app = holoscan::make_application<MultiIoGpuResidentApp>();
  app->compose_graph();

  auto future = app->run_async();

  if (!wait_for_graph_launch(app, std::chrono::seconds(10))) {
    cleanup_and_wait(app, future);
    return 1;
  }

  HOLOSCAN_LOG_INFO("Running fully GPU-resident multi-IO graph for 10 seconds");
  std::this_thread::sleep_for(std::chrono::seconds(10));

  cleanup_and_wait(app, future);
  HOLOSCAN_LOG_INFO("GPU-resident multi-IO example completed");
  return 0;
}
