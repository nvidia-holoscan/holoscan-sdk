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

#include <cuda.h>
#include <cuda_runtime.h>
#include <sys/utsname.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/holoscan.hpp>
#include <holoscan/utils/cuda_macros.hpp>
#include "test_kernel.cu.hpp"

constexpr int kMinCudaDriverVersion = 12040;
constexpr const char* kHsdkFaqUrl =
    "https://docs.nvidia.com/holoscan/sdk-user-guide/hsdk_faq.html";

static std::optional<std::vector<uint32_t>> green_context_partitions_for_current_arch() {
  struct utsname os_info {};
  if (uname(&os_info) != 0) {
    return std::nullopt;
  }
  std::string arch(static_cast<const char*>(os_info.machine));
  if (arch == "x86_64" || arch == "amd64") {
    return std::vector<uint32_t>{8, 8};
  }
  if (arch == "aarch64" || arch == "arm64") {
    // For reference, Jetson Orin AGX has 16 SMs,
    //                Jetson Orin Nano has 8 SMs
    //                Jetson Thor has 22 SMs
    return std::vector<uint32_t>{4, 4};
  }
  return std::nullopt;
}

// Example requires a certain minimum number of GPU Streaming Multiprocessors (SMs) to run
static int required_sm_count_for_green_context(const std::vector<uint32_t>& partitions) {
  int total = 0;
  for (const auto partition_sm_count : partitions) {
    total += static_cast<int>(partition_sm_count);
  }
  return total;
}

// Gets the current CUDA Driver API, such as "12040" for 12.4
static std::optional<int> detect_cuda_driver_version() {
  int version = 0;
  if (cudaDriverGetVersion(&version) != cudaSuccess) {
    return std::nullopt;
  }
  return version;
}

// Green Context APIs are introduced in CUDA Driver 12.4
static bool green_context_supported_by_cuda_driver() {
  auto version = detect_cuda_driver_version();
  return version.has_value() && version.value() >= kMinCudaDriverVersion;
}

// Detects the number of GPU Streaming Multiprocessors (SMs) available on the device
static std::optional<int> detect_device_multiprocessor_count() {
  int sm_count = 0;
  if (cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0) != cudaSuccess) {
    return std::nullopt;
  }
  return sm_count;
}

namespace holoscan::ops {
class PingTxOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(PingTxOp)

  PingTxOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.output<int>("out");
    spec.param(cuda_stream_pool_,
               "cuda_stream_pool",
               "CUDA Stream Pool",
               "CUDA Stream Pool",
               std::shared_ptr<CudaStreamPool>(nullptr));
  }

  // NOLINTBEGIN(readability-function-cognitive-complexity)
  void start() override {
    size_t tensor_bytes = sizeof(float) * tensor_size_ * tensor_size_;
    // Allocate CUDA memory
    HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaMalloc(&input1_, tensor_bytes), "cudaMalloc failed!");
    HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaMalloc(&input2_, tensor_bytes), "cudaMalloc failed!");
    HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaMalloc(&output_, tensor_bytes), "cudaMalloc failed!");
    // Create CUDA events for timing
    HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaEventCreate(&start_event_), "cudaEventCreate failed!");
    HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaEventCreate(&stop_event_), "cudaEventCreate failed!");
  }

  void stop() override {
    HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaEventDestroy(start_event_), "cudaEventDestroy failed!");
    HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaEventDestroy(stop_event_), "cudaEventDestroy failed!");
    HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaFree(input1_), "cudaFree failed!");
    HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaFree(input2_), "cudaFree failed!");
    HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaFree(output_), "cudaFree failed!");
  }
  // NOLINTEND(readability-function-cognitive-complexity)

  void compute([[maybe_unused]] InputContext& op_input, OutputContext& op_output,
               [[maybe_unused]] ExecutionContext& context) override {
    auto value = index_++;
    if (stream_ == nullptr) {
      auto maybe_new_stream = context.allocate_cuda_stream("tx_stream");
      if (maybe_new_stream) {
        stream_ = maybe_new_stream.value();
      }
    }
    cudaStream_t stream = stream_;

    HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaEventRecord(start_event_, stream),
                                   "cudaEventRecord failed!");
    asyncLaunchMatrixMultiplyKernel(input1_, input2_, output_, tensor_size_, stream);
    HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaEventRecord(stop_event_, stream), "cudaEventRecord failed!");

    op_output.emit(value, "out");
  };

 private:
  int index_ = 1;
  int tensor_size_ = 256;
  cudaEvent_t start_event_ = nullptr;
  cudaEvent_t stop_event_ = nullptr;
  cudaStream_t stream_ = nullptr;
  float* input1_ = nullptr;
  float* input2_ = nullptr;
  float* output_ = nullptr;

  Parameter<std::shared_ptr<CudaGreenContext>> cuda_green_context_;
  Parameter<std::shared_ptr<CudaGreenContextPool>> cuda_green_context_pool_;
  Parameter<std::shared_ptr<CudaStreamPool>> cuda_stream_pool_;
};
class PingRxOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(PingRxOp)

  PingRxOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.input<std::vector<int>>("receivers", IOSpec::kAnySize);
  }

  void compute(InputContext& op_input, [[maybe_unused]] OutputContext& op_output,
               [[maybe_unused]] ExecutionContext& context) override {
    auto value_vector = op_input.receive<std::vector<int>>("receivers").value();

    HOLOSCAN_LOG_INFO("Rx message received (count: {}, size: {})", count_++, value_vector.size());

    if (!value_vector.empty()) {
      HOLOSCAN_LOG_INFO("Rx message value: {}", value_vector[0]);
    }
  };

 private:
  int count_ = 1;
};

}  // namespace holoscan::ops

class SampleCudaStreamPoolApp : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;

    // Create a cuda green context
    auto partitions = green_context_partitions_for_current_arch();
    if (!partitions.has_value()) {
      throw std::runtime_error("Unsupported platform architecture for Green Context sample");
    }

    // Create a green context pool which will be used as the default green context pool for the
    // current fragment
    const auto cuda_green_context_pool =
        add_default_green_context_pool(0, std::move(partitions.value()));

    // Use green context 0 from the provided green context pool
    const auto cuda_green_context1 =
        make_resource<CudaGreenContext>("cuda_green_context", cuda_green_context_pool, 0, "tx1");

    const auto cuda_stream_pool1 = make_resource<CudaStreamPool>(
        "cuda_stream_pool", 0, 0, 0, 1, 5, cuda_green_context1, "tx1");
    // Define the tx and rx operators, allowing the tx operator to execute 10 times
    auto tx1 =
        make_operator<ops::PingTxOp>("tx1", make_condition<CountCondition>(10), cuda_stream_pool1);

    auto rx1 = make_operator<ops::PingRxOp>("rx1");

    // Create a thread pool with two threads
    auto pool1 = make_thread_pool("pool1", 2);
    // can assign operators individually to this thread pool (setting pinning to true)
    pool1->add(tx1, true);
    pool1->add(rx1, true);
    add_flow(tx1, rx1);

    // Use a specific green context identified by "index"
    auto cuda_green_context2 =
        make_resource<CudaGreenContext>("cuda_green_context", cuda_green_context_pool, 1, "tx2");
    auto cuda_stream_pool2 = make_resource<CudaStreamPool>(
        "cuda_stream_pool2", 0, 0, 0, 1, 5, cuda_green_context2, "tx2");
    auto tx2 =
        make_operator<ops::PingTxOp>("tx2", make_condition<CountCondition>(15), cuda_stream_pool2);
    auto rx2 = make_operator<ops::PingRxOp>("rx2");

    // Create a thread pool with two threads
    auto pool2 = make_thread_pool("pool2", 2);
    pool2->add(tx2, true);
    pool2->add(rx2, true);
    add_flow(tx2, rx2);

    // Use the default green context from the provided green context pool
    auto tx3 = make_operator<ops::PingTxOp>(
        "tx3", make_condition<CountCondition>(20), cuda_green_context_pool);
    auto rx3 = make_operator<ops::PingRxOp>("rx3");

    // Create a thread pool with two threads
    auto pool3 = make_thread_pool("pool3", 2);
    pool3->add(tx3, true);
    pool3->add(rx3, true);
    add_flow(tx3, rx3);

    // Use the fragment default green context pool without providing a green context pool
    auto tx4 = make_operator<ops::PingTxOp>("tx4", make_condition<CountCondition>(25));
    auto rx4 = make_operator<ops::PingRxOp>("rx4");

    // Create a thread pool with two threads
    auto pool4 = make_thread_pool("pool4", 2);
    pool4->add(tx4, true);
    pool4->add(rx4, true);
    add_flow(tx4, rx4);
  }
};

// CTest skip return code when Green Context is not available.
constexpr int kSkipReturnCode = 77;

int main() {
  if (!green_context_supported_by_cuda_driver()) {
    auto version = detect_cuda_driver_version();
    std::cerr << "Green Context requires CUDA Driver API >= 12.4 (cudaDriverGetVersion >= "
              << kMinCudaDriverVersion << ", detected: "
              << (version.has_value() ? std::to_string(version.value()) : "unknown")
              << "). See " << kHsdkFaqUrl << std::endl;
    return kSkipReturnCode;
  }

  auto partitions = green_context_partitions_for_current_arch();
  if (!partitions.has_value()) {
    std::cerr << "Green Context sample is not configured for this architecture."
              << " See " << kHsdkFaqUrl << std::endl;
    return kSkipReturnCode;
  }

  const int required_sm_count = required_sm_count_for_green_context(partitions.value());
  auto sm_count = detect_device_multiprocessor_count();
  if (!sm_count.has_value() || sm_count.value() < required_sm_count) {
    std::cerr << "Green Context requires at least " << required_sm_count
              << " SMs for this sample's partitioning (detected: "
              << (sm_count.has_value() ? std::to_string(sm_count.value()) : "unknown")
              << "). See " << kHsdkFaqUrl << std::endl;
    return kSkipReturnCode;
  }

  auto app = holoscan::make_application<SampleCudaStreamPoolApp>();

  app->scheduler(app->make_scheduler<holoscan::EventBasedScheduler>(
      "event-based", holoscan::Arg("worker_thread_number", static_cast<int64_t>(4))));

  app->run();

  return 0;
}
