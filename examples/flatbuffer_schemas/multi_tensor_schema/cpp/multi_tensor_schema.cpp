/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <iostream>
#include <memory>
#include <mutex>
#include <ranges>
#include <span>
#include <utility>

#include <cuda_runtime_api.h>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "stereo_frame_traits.hpp"

namespace {

using namespace std::chrono_literals;

using Frame = stereo::StereoFrameT;
using Fields = stereo::StereoFrameTensorFields;

constexpr holoscan::DeviceId kDevice{0};
constexpr DLDataType kImageDtype{kDLUInt, 8U, 1U};
constexpr DLDataType kCalibrationDtype{kDLFloat, 32U, 1U};
constexpr std::array<std::int64_t, 2U> kImageShape{2, 2};
constexpr std::array<std::int64_t, 2U> kCalibrationShape{2, 3};
constexpr std::array<std::uint8_t, 4U> kLeftPixels{1U, 2U, 3U, 4U};
constexpr std::array<std::uint8_t, 4U> kRightPixels{5U, 6U, 7U, 8U};
constexpr std::array<float, 6U> kCalibration{1.0F, 0.0F, 0.5F, 0.0F, 1.0F, -0.5F};
constexpr holoscan::TensorRepresentation kImageLayout{
    .memory_kind = holoscan::MemoryKind::kCudaDevice,
    .dtype = kImageDtype,
    .rank = 2U,
};
constexpr holoscan::TensorRepresentation kCalibrationLayout{
    .memory_kind = holoscan::MemoryKind::kHost,
    .dtype = kCalibrationDtype,
    .rank = 2U,
};
constexpr holoscan::TensorOutputSpec kImageSpec{
    .representation = kImageLayout,
    .bounds = holoscan::tensor_bounds(kLeftPixels.size()),
    .storage = holoscan::TensorOutputStorage::kRuntimePool,
};
constexpr holoscan::TensorOutputSpec kCalibrationSpec{
    .representation = kCalibrationLayout,
    .bounds = holoscan::tensor_bounds(kCalibration.size() * sizeof(kCalibration.front())),
    .storage = holoscan::TensorOutputStorage::kRuntimePool,
};

struct Completion {
  static void reset() {
    const std::lock_guard lock(mutex);
    done = false;
  }

  static void signal() {
    {
      const std::lock_guard lock(mutex);
      done = true;
    }
    ready.notify_all();
  }

  [[nodiscard]] static bool wait() {
    std::unique_lock lock(mutex);
    return ready.wait_for(lock, 10s, [] { return done; });
  }

  static inline std::mutex mutex;
  static inline std::condition_variable ready;
  static inline bool done{};
};

class Source final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    auto& frames = spec.output(output, "frames").max_emits_per_compute(1U);
    frames.tensor_fields(std::array{Fields::left, Fields::right}, kImageSpec);
    frames.tensor_field(Fields::calibration, kCalibrationSpec);
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = 1ms});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    if (emitted_) {
      return {};
    }

    const std::array requests{
        holoscan::tensor_shape(Fields::left, kImageShape),
        holoscan::tensor_shape(Fields::right, kImageShape),
        holoscan::tensor_shape(Fields::calibration, kCalibrationShape),
    };
    auto bundle = output.allocate_tensors(requests);
    if (!bundle) {
      Completion::signal();
      return holoscan::make_unexpected(std::move(bundle).error());
    }
    const auto device_copy = [&](std::span<const std::uint8_t> values) {
      return [&, values](const holoscan::TensorOutputWriteGuard& writer)
                 -> holoscan::expected<void, holoscan::Error> {
        if (cudaMemcpyAsync(writer.data(),
                            values.data(),
                            values.size(),
                            cudaMemcpyHostToDevice,
                            context.cuda_stream().get()) != cudaSuccess) {
          return holoscan::make_unexpected(holoscan::Error{holoscan::ErrorCode::kFailure});
        }
        return {};
      };
    };
    auto written = bundle->write_all(
        holoscan::device_field(Fields::left, context.cuda_stream(), device_copy(kLeftPixels)),
        holoscan::device_field(Fields::right, context.cuda_stream(), device_copy(kRightPixels)),
        holoscan::host_field(Fields::calibration,
                             [](const holoscan::TensorOutputWriteGuard& writer) {
                               std::ranges::copy(kCalibration, writer.data_as<float>());
                             }));
    if (!written) {
      Completion::signal();
      return holoscan::make_unexpected(std::move(written).error());
    }

    Frame frame;
    frame.header = std::make_unique<stereo::FrameHeaderT>();
    frame.header->sequence = 1U;
    frame.header->source = "stereo-camera";
    frame.exposure_us = 2500U;
    auto emitted = output.emit_tensors(std::move(*bundle), frame);
    if (!emitted && (emitted.error().code == holoscan::ErrorCode::kNotReady ||
                     emitted.error().code == holoscan::ErrorCode::kBackpressured)) {
      return {};
    }
    if (!emitted) {
      Completion::signal();
    }
    if (emitted) {
      emitted_ = true;
    }
    return emitted;
  }

  holoscan::Output<Frame> output;

 private:
  bool emitted_{};
};

class Sink final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    auto& builder = spec.input(input, "frames").queue_depth(1U);
    // Purely optional: constrain the logical Tensor contract expected for each schema field.
    builder.tensor_field(Fields::left)
        .expects_tensor(holoscan::TensorInputSpec{.representation = kImageLayout});
    builder.tensor_field(Fields::right)
        .expects_tensor(holoscan::TensorInputSpec{.representation = kImageLayout});
    builder.tensor_field(Fields::calibration)
        .expects_tensor(holoscan::TensorInputSpec{.representation = kCalibrationLayout});
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
      Completion::signal();
      return holoscan::make_unexpected(std::move(sample).error());
    }
    const Frame& frame = sample->data;
    std::array<std::uint8_t, kLeftPixels.size()> left{};
    std::array<std::uint8_t, kRightPixels.size()> right{};
    const float* const calibration =
        frame.calibration != nullptr ? frame.calibration->data_as<float>() : nullptr;
    if (frame.header == nullptr || frame.left == nullptr || frame.right == nullptr ||
        frame.calibration == nullptr || frame.header->sequence != 1U ||
        frame.header->source != "stereo-camera" || frame.exposure_us != 2500U ||
        context.cuda_stream().get() == nullptr ||
        cudaMemcpyAsync(left.data(),
                        frame.left->data(),
                        left.size(),
                        cudaMemcpyDeviceToHost,
                        context.cuda_stream().get()) != cudaSuccess ||
        cudaMemcpyAsync(right.data(),
                        frame.right->data(),
                        right.size(),
                        cudaMemcpyDeviceToHost,
                        context.cuda_stream().get()) != cudaSuccess ||
        cudaStreamSynchronize(context.cuda_stream().get()) != cudaSuccess || left != kLeftPixels ||
        right != kRightPixels ||
        frame.calibration->nbytes() !=
            static_cast<std::int64_t>(kCalibration.size() * sizeof(kCalibration.front())) ||
        calibration == nullptr ||
        !std::equal(calibration, calibration + kCalibration.size(), kCalibration.begin())) {
      Completion::signal();
      return holoscan::make_unexpected(
          holoscan::Error{holoscan::ErrorCode::kFailure, "StereoFrame data mismatch"});
    }
    Completion::signal();
    return {};
  }

  holoscan::Input<Frame> input;
};

}  // namespace

int main() {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= kDevice.value) {
    std::cout << "SKIP: multi_tensor_schema requires CUDA device 0\n";
    return 77;
  }

  Completion::reset();
  holoscan::Graph graph{"multi-tensor-schema"};
  const auto source = graph.op<Source>("source");
  const auto sink = graph.op<Sink>("sink");
  graph.add_flow(source->output, sink->input, holoscan::ConnectionOptions{.queue_depth = 1U});
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("runtime-clock"));

  holoscan::CompileOptions options;
  options.deployment.bind_tensor_output_device({
      .operator_path = "source",
      .output_port = "frames",
      .device = kDevice,
  });
  const holoscan::ExecutionPlan plan = holoscan::compile(graph, std::move(options));
  if (!plan.ok()) {
    std::cerr << plan.json() << '\n';
    return 1;
  }

  holoscan::RunSession session = holoscan::run_async(plan);
  const bool complete = Completion::wait();
  session.request_stop();
  session.wait();
  if (!complete || session.termination() != holoscan::RunTermination::kStopped ||
      !session.diagnostics().empty()) {
    return 2;
  }

  std::cout << "multi_tensor_schema: received one StereoFrameT sample\n";
  return 0;
}
