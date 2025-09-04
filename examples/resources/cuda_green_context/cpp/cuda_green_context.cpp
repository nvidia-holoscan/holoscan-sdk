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

#include <cuda.h>
#include <cuda_runtime.h>
#include <sys/utsname.h>

#include <memory>
#include <string>
#include <vector>

#include <holoscan/holoscan.hpp>
#include <holoscan/utils/cuda_macros.hpp>
#include "test_kernel.cu.hpp"

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
  cudaEvent_t start_event_;
  cudaEvent_t stop_event_;
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
    std::vector<uint32_t> partitions;
    struct utsname osInfo{};
    uname(&osInfo);
    std::string arch(osInfo.machine);
    if (arch == "x86_64" || arch == "amd64") {
      partitions = std::vector<uint32_t>{8, 8};
    } else if (arch == "aarch64" || arch == "arm64") {
      // For reference, Jetson Orin AGX has 16 SMs,
      //                Jetson Orin Nano has 8 SMs
      //                Jetson Thor has 22 SMs
      partitions = std::vector<uint32_t>{4, 4};
    } else {
      throw std::runtime_error(
          fmt::format("Unsupported platform architecture: {}", arch));
    }
    const auto cuda_green_context_pool =
        make_resource<CudaGreenContextPool>("cuda_green_context_pool",
                                            Arg("dev_id", 0),
                                            Arg("num_partitions", (uint32_t)partitions.size()),
                                            Arg("sms_per_partition", partitions));

    const auto cuda_green_context1 = make_resource<CudaGreenContext>(
        "cuda_green_context", cuda_green_context_pool);
    // Create a stream pool with a 5 streams capacity (5 operators could share the same pool)
    const auto cuda_stream_pool1 =
        make_resource<CudaStreamPool>("cuda_stream_pool", 0, 0, 0, 1, 5, cuda_green_context1);
    // Define the tx and rx operators, allowing the tx operator to execute 10 times
    auto tx1 = make_operator<ops::PingTxOp>("tx1",
                                            make_condition<CountCondition>(10),
                                            cuda_stream_pool1);

    auto rx1 = make_operator<ops::PingRxOp>("rx1");

    // Create a thread pool with two threads
    auto pool1 = make_thread_pool("pool1", 2);
    // can assign operators individually to this thread pool (setting pinning to true)
    pool1->add(tx1, true);
    pool1->add(rx1, true);
    add_flow(tx1, rx1);

    // Add another flow to use a different green context and stream pool
    auto cuda_green_context2 =
        make_resource<CudaGreenContext>("cuda_green_context",
                                        cuda_green_context_pool, 1);
    auto cuda_stream_pool2 = make_resource<CudaStreamPool>(0, 0, 0, 1, 5, cuda_green_context2);
    auto tx2 = make_operator<ops::PingTxOp>("tx2",
                                            make_condition<CountCondition>(15),
                                            cuda_stream_pool2);
    auto rx2 = make_operator<ops::PingRxOp>("rx2");

    // Create a thread pool with two threads
    auto pool2 = make_thread_pool("pool2", 2);
    pool2->add(tx2, true);
    pool2->add(rx2, true);
    add_flow(tx2, rx2);

    // Create a flow to use default green context and default stream pool
    // cuda_green_context_pool is mondatory to use green context with the default green context pool
    auto tx3 =
        make_operator<ops::PingTxOp>("tx3",
                                     make_condition<CountCondition>(20),
                                     cuda_green_context_pool);
    auto rx3 = make_operator<ops::PingRxOp>("rx3");

    // Create a thread pool with two threads
    auto pool3 = make_thread_pool("pool3", 2);
    pool3->add(tx3, true);
    pool3->add(rx3, true);
    add_flow(tx3, rx3);
  }
};

int main() {
  auto app = holoscan::make_application<SampleCudaStreamPoolApp>();

  app->scheduler(app->make_scheduler<holoscan::EventBasedScheduler>(
      "event-based", holoscan::Arg("worker_thread_number", static_cast<int64_t>(4))));
  app->run();

  return 0;
}
