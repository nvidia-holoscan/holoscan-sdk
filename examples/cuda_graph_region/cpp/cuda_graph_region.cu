/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuda_runtime_api.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <string>
#include <string_view>
#include <utility>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/domain/dlpack.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/time/realtime_clock.hpp>

namespace {

using namespace std::chrono_literals;

constexpr std::uint32_t kSampleCount = 4U;
constexpr int kNoCUDADevice = 77;
constexpr DLDataType kUint32{.code = kDLUInt, .bits = 32U, .lanes = 1U};
constexpr std::array<std::int64_t, 1U> kShape{1};

constexpr holoscan::TensorRepresentation kDeviceScalar{
    .memory_kind = holoscan::MemoryKind::kCudaDevice,
    .dtype = kUint32,
    .rank = 1U,
    .strides = holoscan::TensorStridePolicy::kContiguousOnly,
};

[[nodiscard]] holoscan::TensorOutputSpec output_spec() {
  return holoscan::TensorOutputSpec{
      .representation = kDeviceScalar,
      .bounds = holoscan::tensor_bounds(
          kShape, holoscan::TensorBoundsOptions{.max_byte_span = sizeof(std::uint32_t)}),
      .storage = holoscan::TensorOutputStorage::kRuntimePool,
  };
}

[[nodiscard]] holoscan::TensorInputSpec input_spec() {
  return holoscan::TensorInputSpec{.representation = kDeviceScalar};
}

[[nodiscard]] holoscan::expected<void, holoscan::Error> cuda_failure() {
  return holoscan::make_unexpected(holoscan::Error{holoscan::ErrorCode::kFailure});
}

struct Observation {
  static void record(std::uint32_t observed, holoscan::ExecutionContext& context) {
    const std::uint32_t index = count.fetch_add(1U, std::memory_order_relaxed) + 1U;
    std::cout << "result " << index << ": " << observed << '\n';
    if (observed != 2U * index + 3U) {
      valid.store(false, std::memory_order_relaxed);
    }
    if (index == kSampleCount) {
      context.request_completion();
    }
  }

  [[nodiscard]] static bool passed() noexcept {
    return count.load(std::memory_order_relaxed) == kSampleCount &&
           valid.load(std::memory_order_relaxed);
  }

  static inline std::atomic<std::uint32_t> count{};
  static inline std::atomic_bool valid{true};
};

__global__ void write_value(std::uint32_t* output, std::uint32_t value) {
  *output = value;
}

__global__ void transform_value(const std::uint32_t* input, std::uint32_t* output,
                                std::uint32_t multiplier, std::uint32_t increment) {
  *output = *input * multiplier + increment;
}

class DeviceSource final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.output(output, "output").max_emits_per_compute(1U).produces_tensor(output_spec());
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = 2ms});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    if (sequence_ == kSampleCount) {
      return {};
    }
    auto loan = output.allocate_tensor(kShape);
    if (!loan) {
      // Retry on the next clock activation while the one-slot region ingress is still owned.
      if (loan.error().code == holoscan::ErrorCode::kBackpressured ||
          loan.error().code == holoscan::ErrorCode::kResourceExhausted) {
        return {};
      }
      return holoscan::make_unexpected(std::move(loan.error()));
    }
    auto writer = loan->write(context.cuda_stream());
    if (!writer) {
      return holoscan::make_unexpected(std::move(writer.error()));
    }
    write_value<<<1, 1, 0, context.cuda_stream().get()>>>(
        static_cast<std::uint32_t*>(writer->data()), sequence_ + 1U);
    if (cudaGetLastError() != cudaSuccess) {
      return cuda_failure();
    }
    auto committed = std::move(*writer).commit();
    if (!committed) {
      return committed;
    }
    auto emitted = output.emit(std::move(*loan));
    if (!emitted) {
      if (emitted.error().code == holoscan::ErrorCode::kBackpressured) {
        return {};
      }
      return emitted;
    }
    ++sequence_;
    return {};
  }

  holoscan::Output<holoscan::Tensor> output;

 private:
  std::uint32_t sequence_{};
};

class Transform final : public holoscan::Operator<> {
 public:
  Transform(std::uint32_t multiplier, std::uint32_t increment)
      : multiplier_(multiplier), increment_(increment) {}

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(input, "input").queue_depth(1U).expects_tensor(input_spec());
    spec.output(output, "output").max_emits_per_compute(1U).produces_tensor(output_spec());
    spec.cuda_graph_capture_eligible();
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{input});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    auto sample = input.receive();
    if (!sample) {
      return holoscan::make_unexpected(std::move(sample.error()));
    }
    auto loan = output.allocate_tensor(kShape);
    if (!loan) {
      return holoscan::make_unexpected(std::move(loan.error()));
    }
    auto writer = loan->write(context.cuda_stream());
    if (!writer) {
      return holoscan::make_unexpected(std::move(writer.error()));
    }
    transform_value<<<1, 1, 0, context.cuda_stream().get()>>>(
        static_cast<const std::uint32_t*>(sample->data.data()),
        static_cast<std::uint32_t*>(writer->data()),
        multiplier_,
        increment_);
    if (cudaGetLastError() != cudaSuccess) {
      return cuda_failure();
    }
    auto committed = std::move(*writer).commit();
    if (!committed) {
      return committed;
    }
    return output.emit(std::move(*loan));
  }

  holoscan::Input<holoscan::Tensor> input;
  holoscan::Output<holoscan::Tensor> output;

 private:
  std::uint32_t multiplier_{};
  std::uint32_t increment_{};
};

class ResultSink final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(input, "input").queue_depth(1U).expects_tensor(input_spec());
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{input});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    auto sample = input.receive();
    if (!sample) {
      return holoscan::make_unexpected(std::move(sample.error()));
    }
    std::uint32_t observed{};
    // The runtime orders each input onto the operator's execution stream before compute().
    // A normal-mode allocation may therefore elide its separate readiness event, so copy on the
    // already-ordered stream instead of requiring synchronize_for_host().
    if (cudaMemcpyAsync(&observed,
                        sample->data.data(),
                        sizeof(observed),
                        cudaMemcpyDeviceToHost,
                        context.cuda_stream().get()) != cudaSuccess ||
        cudaStreamSynchronize(context.cuda_stream().get()) != cudaSuccess) {
      return cuda_failure();
    }
    Observation::record(observed, context);
    return {};
  }

  holoscan::Input<holoscan::Tensor> input;
};

void bind_output(holoscan::CompileOptions& options, std::string_view operator_path) {
  options.deployment.bind_tensor_output_device({
      .operator_path = operator_path,
      .output_port = "output",
      .device = holoscan::DeviceId{0},
  });
}

}  // namespace

// Compare ordinary dispatch and CUDA Graph replay over the same Tensor pipeline.
int main(int argc, char** argv) {
  const std::string mode = argc > 1 ? argv[1] : "--cuda-graph";
  if (argc > 2 || (mode != "--cuda-graph" && mode != "--normal")) {
    std::cerr << "usage: cuda_graph_region [--cuda-graph|--normal]\n";
    return 2;
  }
  const bool use_cuda_graph = mode == "--cuda-graph";

  int device_count{};
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    std::cout << "cuda_graph_region skipped: no CUDA device\n";
    return kNoCUDADevice;
  }

  holoscan::Graph graph{"cuda-graph-region"};
  const auto source = graph.op<DeviceSource>("source");
  const auto multiply = graph.op<Transform>("multiply", 2U, 0U);
  const auto add = graph.op<Transform>("add", 1U, 3U);
  const auto sink = graph.op<ResultSink>("sink");
  graph.add_flow(source->output, multiply->input, holoscan::ConnectionOptions{.queue_depth = 1U});
  graph.add_flow(multiply->output, add->input, holoscan::ConnectionOptions{.queue_depth = 1U});
  graph.add_flow(add->output, sink->input, holoscan::ConnectionOptions{.queue_depth = 1U});
  if (use_cuda_graph) {
    graph.region("transforms")
        .add(multiply)
        .add(add)
        .constrain(holoscan::RegionConstraint::cuda_graph());
  }
  graph.partition("pipeline").include_all();
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

  holoscan::CompileOptions options;
  bind_output(options, "source");
  bind_output(options, "multiply");
  bind_output(options, "add");
  const holoscan::ExecutionPlan plan = holoscan::compile(graph, std::move(options));
  if (!plan.ok()) {
    std::cerr << plan.json() << '\n';
    return 1;
  }

  holoscan::RunSession session = holoscan::run_async(plan);
  session.wait();
  if (session.termination() != holoscan::RunTermination::kRequestedCompleted ||
      !Observation::passed()) {
    std::cerr << "cuda_graph_region failed: termination=" << static_cast<int>(session.termination())
              << " results_valid=" << Observation::passed() << '\n';
    for (const auto& diagnostic : session.diagnostics()) {
      std::cerr << "  " << diagnostic.token << ": " << diagnostic.provided << '\n';
    }
    return 2;
  }
  std::cout << "cuda_graph_region " << (use_cuda_graph ? "CUDA_GRAPH" : "NORMAL") << " PASS\n";
  return 0;
}
