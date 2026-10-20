/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// green_context — manually give one CUDA execution partition a dedicated structured SM group.
//
//   producer --(CUDA Tensor)--> transform --(CUDA Tensor)--> sink
//   "critical"                  "background"                 "background"
//   explicitly bound            automatically assigned to the structured residual
//
// The example queries the target, requests one driver-qualified SM quantum for the critical
// partition, prints the admitted resource layout, and runs both kernels on Runtime-owned streams.

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <string_view>
#include <thread>
#include <utility>

#include <cuda_runtime_api.h>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/domain/dlpack.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/execution_plan.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/time/realtime_clock.hpp>

namespace {

using namespace std::chrono_literals;

constexpr int kNoGreenContextReturnCode = 77;
constexpr std::size_t kFrameBytes = 256U;
constexpr int kThreadsPerBlock = 128;
constexpr DLDataType kByteDtype{kDLUInt, 8U, 1U};
constexpr std::array<std::int64_t, 1U> kFrameShape{static_cast<std::int64_t>(kFrameBytes)};
constexpr holoscan::TensorInputSpec kDeviceByteFrameInput{
    .representation = {.memory_kind = holoscan::MemoryKind::kCudaDevice,
                       .dtype = kByteDtype,
                       .rank = 1U},
};
constexpr holoscan::TensorOutputSpec kDeviceByteFrameOutput{
    .representation = {.memory_kind = holoscan::MemoryKind::kCudaDevice,
                       .dtype = kByteDtype,
                       .rank = 1U},
    .bounds = holoscan::tensor_bounds(kFrameBytes),
    .storage = holoscan::TensorOutputStorage::kRuntimePool,
};

__global__ void make_pattern(std::uint8_t* output, std::size_t size, std::uint8_t value) {
  const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    output[index] = value;
  }
}

__global__ void transform_pattern(const std::uint8_t* input, std::uint8_t* output,
                                  std::size_t size) {
  const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    output[index] = static_cast<std::uint8_t>(input[index] + 1U);
  }
}

[[nodiscard]] holoscan::expected<void, holoscan::Error> cuda_launch_result(
    cudaError_t result, std::string_view operation) {
  if (result == cudaSuccess) {
    return {};
  }
  std::string message{operation};
  message.append(": ");
  message.append(cudaGetErrorString(result));
  return holoscan::make_unexpected(
      holoscan::Error{holoscan::ErrorCode::kFailure, std::move(message)});
}

class CUDAProducer final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.output(frame, "frame").produces_tensor(kDeviceByteFrameOutput);
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = 50ms});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    // Green Context placement is realized as this borrowed stream. Application CUDA work must use
    // it; creating or using an unrelated stream would put that work outside the HSDK guarantee.
    cudaStream_t stream = context.cuda_stream().get();
    if (stream == nullptr) {
      return holoscan::make_unexpected(
          holoscan::Error{holoscan::ErrorCode::kNotSupported,
                          "CUDAProducer.frame: missing Runtime CUDA compute stream resource"});
    }

    auto loan = frame.allocate_tensor(
        holoscan::TensorLoanRequest{.shape = kFrameShape, .dtype = kByteDtype});
    if (!loan) {
      return holoscan::make_unexpected(std::move(loan).error());
    }
    auto writer = loan->write(context.cuda_stream());
    if (!writer) {
      return holoscan::make_unexpected(std::move(writer).error());
    }

    const int blocks = static_cast<int>((kFrameBytes + kThreadsPerBlock - 1U) / kThreadsPerBlock);
    make_pattern<<<blocks, kThreadsPerBlock, 0, stream>>>(
        static_cast<std::uint8_t*>(writer->data()), kFrameBytes, ++value_);
    if (auto launched = cuda_launch_result(cudaGetLastError(), "make_pattern launch"); !launched) {
      return launched;
    }

    auto committed = std::move(*writer).commit();
    if (!committed) {
      return holoscan::make_unexpected(std::move(committed).error());
    }
    return frame.emit(std::move(*loan),
                      holoscan::EmitOptions{.capture_time = context.activation_time()});
  }

  holoscan::Output<holoscan::Tensor> frame;

 private:
  std::uint8_t value_{};
};

class CUDATransform final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(input, "input").queue_depth(2U).expects_tensor(kDeviceByteFrameInput);
    spec.output(output, "output").produces_tensor(kDeviceByteFrameOutput);
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{input});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    // Normal receive() already makes this residual stream wait for the producer's dedicated
    // stream. No explicit stream receive or host synchronization is needed.
    auto sample = input.receive();
    if (!sample) {
      return holoscan::make_unexpected(std::move(sample).error());
    }
    const holoscan::Tensor& source = sample->data;
    cudaStream_t stream = context.cuda_stream().get();
    if (stream == nullptr || source.data() == nullptr ||
        source.nbytes() != static_cast<std::int64_t>(kFrameBytes)) {
      return holoscan::make_unexpected(
          holoscan::Error{holoscan::ErrorCode::kInvalidArgument,
                          "CUDATransform.input: invalid device Tensor or CUDA compute stream "
                          "resource"});
    }

    auto loan = output.allocate_tensor(
        holoscan::TensorLoanRequest{.shape = kFrameShape, .dtype = kByteDtype});
    if (!loan) {
      return holoscan::make_unexpected(std::move(loan).error());
    }
    auto writer = loan->write(context.cuda_stream());
    if (!writer) {
      return holoscan::make_unexpected(std::move(writer).error());
    }

    const int blocks = static_cast<int>((kFrameBytes + kThreadsPerBlock - 1U) / kThreadsPerBlock);
    transform_pattern<<<blocks, kThreadsPerBlock, 0, stream>>>(
        static_cast<const std::uint8_t*>(source.data()),
        static_cast<std::uint8_t*>(writer->data()),
        kFrameBytes);
    if (auto launched = cuda_launch_result(cudaGetLastError(), "transform_pattern launch");
        !launched) {
      return launched;
    }

    auto committed = std::move(*writer).commit();
    if (!committed) {
      return holoscan::make_unexpected(std::move(committed).error());
    }
    return output.emit(std::move(*loan),
                       holoscan::EmitOptions{.capture_time = context.activation_time()});
  }

  holoscan::Input<holoscan::Tensor> input;
  holoscan::Output<holoscan::Tensor> output;
};

class TensorSink final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(input, "input").queue_depth(2U).expects_tensor(kDeviceByteFrameInput);
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{input});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    auto sample = input.receive();
    if (!sample) {
      return holoscan::make_unexpected(std::move(sample).error());
    }
    return {};
  }

  holoscan::Input<holoscan::Tensor> input;
};

std::string_view resource_role_name(holoscan::CudaGreenContextResourceRole role) noexcept {
  switch (role) {
    case holoscan::CudaGreenContextResourceRole::kDedicatedPartition:
      return "dedicated";
    case holoscan::CudaGreenContextResourceRole::kSharedServiceResidual:
      return "shared residual";
    case holoscan::CudaGreenContextResourceRole::kReservedResidual:
      return "reserved residual";
  }
  return "unknown";
}

std::string_view partition_name(const holoscan::ExecutionPlan& plan,
                                holoscan::ExecutionPartitionId id) noexcept {
  for (const auto& partition : plan.execution_partitions()) {
    if (partition.id == id) {
      return partition.authored_name ? std::string_view{*partition.authored_name} : "implicit";
    }
  }
  return "unknown";
}

void print_plan(const holoscan::ExecutionPlan& plan) {
  for (const auto& pool : plan.cuda_green_context_pools()) {
    const std::uint64_t structured_capacity =
        static_cast<std::uint64_t>(pool.max_structured_context_count) * pool.structured_sm_quantum;
    std::cout << "structured capacity: " << structured_capacity
              << " SMs; unprovisioned topology tail: " << pool.unprovisioned_sm_count << " SMs\n";
  }
  for (const auto& resource : plan.cuda_green_context_resources()) {
    std::cout << "resource: " << resource_role_name(resource.role) << ", "
              << resource.accepted_sm_count << " SMs\n";
  }
  for (const auto& stream : plan.cuda_compute_streams()) {
    if (stream.partition && stream.green_context_id) {
      std::cout << "partition " << partition_name(plan, *stream.partition) << " -> "
                << *stream.green_context_id << '\n';
    }
  }
}

void print_diagnostics(const holoscan::ExecutionPlan& plan) {
  for (const auto& diagnostic : plan.diagnostics()) {
    std::cerr << "  " << diagnostic.token;
    if (!diagnostic.provided.empty()) {
      std::cerr << ": " << diagnostic.provided;
    }
    std::cerr << '\n';
  }
}

}  // namespace

int main() {
  // Green Context sizes are target evidence, not portable percentages. Query the exact device and
  // use its driver-qualified quantum rather than dividing total_sm_count ourselves.
  const holoscan::DeviceId device{0};
  const auto capability = holoscan::query_cuda_green_context_capability(device);
  if (!capability) {
    std::cout << "green_context: skipped: " << holoscan::runtime_error_message(capability.error())
              << '\n';
    return kNoGreenContextReturnCode;
  }
  if (capability->structured_sm_quantum == 0U || capability->max_structured_context_count < 2U) {
    std::cout << "green_context: skipped: target cannot provide a dedicated group plus residual\n";
    return kNoGreenContextReturnCode;
  }

  std::cout << "device: " << capability->deployment_device_id
            << "; total SMs: " << capability->total_sm_count
            << "; structured quantum: " << capability->structured_sm_quantum << '\n';

  holoscan::Graph graph{"green-context"};
  const auto producer = graph.op<CUDAProducer>("producer");
  const auto transform = graph.op<CUDATransform>("transform");
  const auto sink = graph.op<TensorSink>("sink");
  graph.add_flow(producer->frame, transform->input, holoscan::ConnectionOptions{.queue_depth = 2U});
  graph.add_flow(transform->output, sink->input, holoscan::ConnectionOptions{.queue_depth = 2U});
  graph.partition("critical").add(producer);
  graph.partition("background").add(transform).add(sink);
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

  holoscan::CompileOptions options;
  // Tensor device placement and SM placement are independent declarations. Both Tensor outputs
  // name device 0 explicitly; HSDK never infers an omitted device as ordinal zero.
  options.deployment
      .bind_tensor_output_device({
          .operator_path = "producer",
          .output_port = "frame",
          .device = device,
      })
      .bind_tensor_output_device({
          .operator_path = "transform",
          .output_port = "output",
          .device = device,
      });

  // Bind only the critical partition. This opts the complete process/device into Green Context
  // execution, so the unbound background partition is assigned to the structured residual.
  options.deployment.bind_cuda_green_context("critical",
                                             holoscan::CudaGreenContextPlacement{
                                                 .device = device,
                                                 .sm_count = capability->structured_sm_quantum,
                                             });

  const holoscan::ExecutionPlan plan = holoscan::compile(graph, std::move(options));
  if (!plan.ok()) {
    std::cerr << "green_context: compilation rejected\n";
    print_diagnostics(plan);
    return 1;
  }
  print_plan(plan);

  holoscan::RunSession session = holoscan::run_async(plan);
  std::this_thread::sleep_for(500ms);
  session.request_stop();
  session.wait();
  if (session.termination() == holoscan::RunTermination::kFailed) {
    std::cerr << "green_context: runtime failed\n";
    for (const auto& diagnostic : session.diagnostics()) {
      std::cerr << "  " << diagnostic.token << '\n';
    }
    return 2;
  }

  std::cout << "green_context example complete\n";
  return 0;
}
