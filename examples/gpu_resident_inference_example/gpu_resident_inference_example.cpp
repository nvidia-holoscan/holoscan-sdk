/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <future>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <holoscan/core/gpu_resident_operator.hpp>
#include <holoscan/holoscan.hpp>
#include <holoscan/operators/gpu_resident_inference/gpu_resident_inference.hpp>
#include <holoscan/utils/cuda_macros.hpp>

namespace holoscan::ops {

// SourceGpuOp: Only has output port, does not do anything in compute
class SourceGpuOp : public holoscan::GPUResidentOperator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS_SUPER(SourceGpuOp, holoscan::GPUResidentOperator)

  SourceGpuOp() = default;

  void setup(OperatorSpec& spec) override {
    size_t width = 512;
    size_t height = 512;
    size_t channels = 3;
    size_t in_buffer_size = width * height * channels * 4;
    spec.device_output("out", in_buffer_size);
  }

  void compute([[maybe_unused]] holoscan::InputContext& op_input,
               [[maybe_unused]] holoscan::OutputContext& op_output,
               [[maybe_unused]] holoscan::ExecutionContext& context) override {
    auto* out_device_address = device_memory("out");
    HOLOSCAN_LOG_INFO("SourceGpuOp::compute() -- {} -- No computation performed", name());
    HOLOSCAN_LOG_INFO("SourceGpuOp::compute() -- {} -- output at: {}", name(), out_device_address);
  }
};

// DestinationGpuOp: Only has input port, consumes the input
class DestinationGpuOp : public holoscan::GPUResidentOperator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS_SUPER(DestinationGpuOp, holoscan::GPUResidentOperator)

  DestinationGpuOp() = default;

  void setup(OperatorSpec& spec) override {
    size_t width = 512;
    size_t height = 512;
    size_t channels = 3;
    size_t out_buffer_size = width * height * channels * 4;
    spec.device_input("in", out_buffer_size);
  }

  void compute([[maybe_unused]] holoscan::InputContext& op_input,
               [[maybe_unused]] holoscan::OutputContext& op_output,
               [[maybe_unused]] holoscan::ExecutionContext& context) override {
    auto* in_device_address = device_memory("in");

    HOLOSCAN_LOG_INFO(
        "DestinationGpuOp::compute() -- {} -- Consuming input at: {}", name(), in_device_address);
  }
};

}  // namespace holoscan::ops

class GpuResidentFragment : public holoscan::Fragment {
 public:
  explicit GpuResidentFragment(const std::string& config_path) : config_path_(config_path) {}

  void compose() override {
    using namespace holoscan;

    auto source_op = make_operator<ops::SourceGpuOp>("source_op");
    auto infer_op = make_operator<ops::GPUResidentInferenceOp>("infer_op", config_path_);

    auto destination_op = make_operator<ops::DestinationGpuOp>("destination_op");

    add_flow(source_op, infer_op);
    add_flow(infer_op, destination_op);
  }

 private:
  std::string config_path_;
};

// Application class that creates the fragment and adds it to the application
class GPUResidentApp : public holoscan::Application {
 public:
  explicit GPUResidentApp(const std::string& config_path) : config_path_(config_path) {}

  void compose() override {
    using namespace holoscan;

    auto gr_fragment = make_fragment<GpuResidentFragment>("gr_fragment", config_path_);

    add_fragment(gr_fragment);
    HOLOSCAN_LOG_INFO("fragments created");
  }

 private:
  std::string config_path_;
};

// Cleanup function that sends tear down command and calls future.get()
void cleanup_and_wait(const std::shared_ptr<holoscan::Fragment>& gr_fragment,
                      std::future<void>& future) {
  if (gr_fragment) {
    gr_fragment->gpu_resident().tear_down();
  }
  future.get();
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
                                holoscan::GPUResidentOperator*& destination_op) {
  auto graph = gr_fragment->graph_shared();
  auto source_node = graph->find_node("source_op");
  auto sink_node = graph->find_node("destination_op");

  source_op = nullptr;
  destination_op = nullptr;

  if (source_node) {
    source_op = dynamic_cast<holoscan::GPUResidentOperator*>(source_node.get());
  }
  if (sink_node) {
    destination_op = dynamic_cast<holoscan::GPUResidentOperator*>(sink_node.get());
  }

  if (source_op == nullptr || destination_op == nullptr) {
    HOLOSCAN_LOG_ERROR("Could not find source_op or destination_op operators");
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

// Helper function to run a single iteration of the GPU resident graph execution
bool run_iteration(holoscan::GPUResidentOperator* source_op,
                   holoscan::GPUResidentOperator* destination_op,
                   const std::shared_ptr<holoscan::Fragment>& gr_fragment, int iteration) {
  HOLOSCAN_LOG_INFO("Iteration {} - Preparing data", iteration + 1);

  size_t width = 512;
  size_t height = 512;
  size_t channels = 3;
  size_t output_channels = 1;
  float value = 6.0 + static_cast<float>(iteration);

  size_t buffer_size = width * height * channels * 4;
  std::vector<float> input_data(width * height * channels, value);

  // Get device memory address from the source operator's output
  void* source_output_device_addr = source_op->device_memory("out");
  if (source_output_device_addr == nullptr) {
    HOLOSCAN_LOG_ERROR("Could not find source operator output device memory");
    return false;
  }

  // Copy input data from host to device
  HOLOSCAN_CUDA_CALL_THROW_ERROR(
      cudaMemcpy(static_cast<void*>(source_output_device_addr),
                 static_cast<const void*>(input_data.data()),
                 buffer_size,
                 cudaMemcpyHostToDevice),
      "Failed to copy input data from host to source operator output device memory");
  HOLOSCAN_LOG_INFO("Iteration {} - Copied input data to source operator output", iteration + 1);

  // synchronize the default stream
  HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaStreamSynchronize(0), "Failed to synchronize default stream");
  // Make data ready for GPU resident graph execution
  gr_fragment->gpu_resident().data_ready();

  // Wait for result to be ready
  if (!wait_for_result(gr_fragment, iteration)) {
    return false;
  }

  // Get the input from the destination operator (which should contain the processed data)
  void* destination_input_device_addr = destination_op->device_memory("in");
  if (destination_input_device_addr == nullptr) {
    HOLOSCAN_LOG_ERROR("Could not find destination operator input device memory");
    return false;
  }

  // Copy inference output from device to host
  std::vector<float> output_data_float(width * height * 3);
  size_t output_buffer_size = width * height * 3 * 4;
  HOLOSCAN_CUDA_CALL_THROW_ERROR(
      cudaMemcpy(static_cast<void*>(output_data_float.data()),
                 destination_input_device_addr,
                 output_buffer_size,
                 cudaMemcpyDeviceToHost),
      "Failed to copy result from destination operator input device memory to host");

  // synchronize the default stream
  HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaStreamSynchronize(0), "Failed to synchronize default stream");
  // std::vector<unsigned char> output_data_temp(output_data_float.begin(),
  // output_data_float.end());

  float fmax = 0, fmin = 255;

  for (auto index = 0; index < output_data_float.size(); index++) {
    auto v = output_data_float.data()[index];
    if (fmax < v) {
      fmax = v;
    }
    if (fmin > v) {
      fmin = v;
    }
  }
  HOLOSCAN_LOG_INFO("Max value: {}, Min value: {}", fmax, fmin);
  HOLOSCAN_LOG_INFO("Original Value: {}", value);

  if (std::abs(value - fmax) < 1e-6 && std::abs(value - fmin) < 1e-6) {
    HOLOSCAN_LOG_INFO("All results are correct!");
  } else {
    HOLOSCAN_LOG_ERROR("Results are incorrect!");
    return false;
  }
  return true;
}

int main(int argc, char** argv) {
  try {
    std::string config_path;
    try {
      if (argc < 1) {
        HOLOSCAN_LOG_ERROR("Error: Invalid number of arguments in GPU resident inference example");
        return 1;
      }
      config_path = std::filesystem::canonical(argv[0]).parent_path() / "app_config.yaml";
    } catch (const std::filesystem::filesystem_error& e) {
      HOLOSCAN_LOG_ERROR("Error: {}", e.what());
      return 1;
    }
    if (!std::filesystem::exists(config_path)) {
      HOLOSCAN_LOG_ERROR("Error: App config path does not exist: {}", config_path);
      return 1;
    }
    auto app = holoscan::make_application<GPUResidentApp>(config_path);

    // Compose the application graph to create the fragment objects.
    app->compose_graph();

    // Get the GPU resident fragment from the application
    auto& fragment_graph = app->fragment_graph();
    auto gr_fragment = fragment_graph.find_node("gr_fragment");

    if (!gr_fragment) {
      HOLOSCAN_LOG_ERROR("Could not find gr_fragment");
      return 1;
    }

    // Compose the fragment's operator graph so GPU-resident functions become available.
    gr_fragment->compose_graph();

    // This example reads back results to the host via cudaMemcpy between iterations.
    // Enable sync_with_host so that a system-wide fence is issued at the end of each
    // iteration, guaranteeing that all device memory writes are visible to the host
    // before result_ready() returns true.
    // Note: sync_with_host is not required when the pipeline is driven entirely by
    // GPU-side data ready handlers with no host-side readback between iterations.
    gr_fragment->gpu_resident().sync_with_host();

    auto future = app->run_async();

    // Wait for the GPU-resident CUDA graph to be launched
    if (!wait_for_graph_launch(gr_fragment)) {
      cleanup_and_wait(gr_fragment, future);
      return 1;
    }

    // Get the source and sink operators
    holoscan::GPUResidentOperator* source_op = nullptr;
    holoscan::GPUResidentOperator* destination_op = nullptr;
    if (!get_gpu_resident_operators(gr_fragment, source_op, destination_op)) {
      cleanup_and_wait(gr_fragment, future);
      return 1;
    }

    bool all_iterations_succeeded = true;
    // Run 10 iterations to test the GPU resident graph execution
    for (int iteration = 0; iteration < 10; ++iteration) {
      if (!run_iteration(source_op, destination_op, gr_fragment, iteration)) {
        all_iterations_succeeded = false;
        break;
      }
    }

    HOLOSCAN_LOG_INFO("Tearing down the GPU-resident fragment");
    cleanup_and_wait(gr_fragment, future);
    return all_iterations_succeeded ? 0 : 1;
  } catch (const std::exception& e) {
    HOLOSCAN_LOG_ERROR("{}", e.what());
  }

  return 0;
}
