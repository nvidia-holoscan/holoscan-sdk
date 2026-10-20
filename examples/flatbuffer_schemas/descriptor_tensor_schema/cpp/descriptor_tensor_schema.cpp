/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <array>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <mutex>
#include <ranges>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/channel/tensor_schema_endpoints.hpp>
#include <holoscan/core/compile.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/schema/descriptor_flatbuffer_payload.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "descriptor_frame_schema_traits.hpp"

namespace {

using namespace std::chrono_literals;

using InstalledDescriptorPayload =
    holoscan::schema::DescriptorFlatBufferPayloadFor<app3p::DescriptorFrame>;

[[maybe_unused]] constexpr auto kInstalledDescriptorSubscriberClose =
    &holoscan::ChannelTensorSchemaSubscriber<InstalledDescriptorPayload>::close;

constexpr std::array<std::int64_t, 1U> kShape{8};
constexpr std::array<std::uint8_t, 8U> kBytes{1U, 3U, 5U, 7U, 9U, 11U, 13U, 15U};
constexpr DLDataType kTensorDtype{kDLUInt, 8U, 1U};
constexpr holoscan::TensorRepresentation kTensorLayout{
    .memory_kind = holoscan::MemoryKind::kHost,
    .dtype = kTensorDtype,
    .rank = 1U,
};

// This finite example stops after the sink receives its frame.
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
    spec.output(output, "output")
        .max_emits_per_compute(1U)
        .produces_tensor(holoscan::TensorOutputSpec{
            .representation = kTensorLayout,
            .bounds = holoscan::tensor_bounds(kBytes.size()),
            .storage = holoscan::TensorOutputStorage::kRuntimePool,
        });
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = 1ms});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    if (emitted_) {
      return {};
    }
    emitted_ = true;

    auto loan = output.allocate_tensor(holoscan::TensorLoanRequest{
        .shape = kShape,
        .dtype = kTensorDtype,
    });
    if (!loan) {
      Completion::signal();
      return holoscan::make_unexpected(std::move(loan).error());
    }

    auto written = loan->write_host_and_commit([](const holoscan::TensorOutputWriteGuard& writer) {
      std::ranges::copy(kBytes, writer.data_as<std::uint8_t>());
    });
    if (!written) {
      Completion::signal();
      return holoscan::make_unexpected(std::move(written).error());
    }

    app3p::DescriptorFrameT frame;
    frame.sequence = 17U;
    frame.label = "descriptor frame";
    frame.gain = 0.25F;
    frame.samples = std::vector<std::int16_t>{2, 4, 8, 16};
    frame.aliases = std::vector<std::string>{"alpha", "beta"};
    frame.details = std::make_shared<app3p::FrameDetailsT>();
    frame.details->code = 29U;
    frame.details->note = "generated nested";
    frame.point = std::make_shared<app3p::FramePoint>(1.5F, -2.0F);

    auto result =
        output.emit_tensor(std::move(*loan), frame, holoscan::EmitOptions{.frame_id = 23U});
    if (!result) {
      Completion::signal();
    }
    return result;
  }

  holoscan::Output<app3p::DescriptorFrameT> output;

 private:
  bool emitted_{};
};

class Sink final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    // Optional consumer compatibility constraint for the schema's Tensor field.
    spec.input(input, "input")
        .queue_depth(1U)
        .expects_tensor(holoscan::TensorInputSpec{.representation = kTensorLayout});
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
      Completion::signal();
      return holoscan::make_unexpected(std::move(sample).error());
    }

    const holoscan::Tensor* tensor = sample->data.tensor.get();
    if (tensor == nullptr || tensor->nbytes() != static_cast<std::int64_t>(kBytes.size())) {
      std::cerr << "descriptor_tensor_schema: received an invalid Tensor\n";
      return holoscan::make_unexpected(
          holoscan::Error{holoscan::ErrorCode::kProtocolError, "received Tensor is invalid"});
    }
    std::cout << "frame: sequence=" << sample->data.sequence << " label=" << sample->data.label
              << " tensor_bytes=" << tensor->nbytes() << '\n';
    Completion::signal();
    return {};
  }

  holoscan::Input<app3p::DescriptorFrameT> input;
};

void print_diagnostics(const holoscan::RunSession& session) {
  for (const auto& diagnostic : session.diagnostics()) {
    std::cerr << diagnostic.token << " required=" << diagnostic.required
              << " provided=" << diagnostic.provided << '\n';
  }
}

}  // namespace

int main() {
  Completion::reset();

  holoscan::Graph graph{"descriptor-tensor-schema"};
  const auto source = graph.op<Source>("source");
  const auto sink = graph.op<Sink>("sink");
  graph.add_flow(source->output, sink->input, holoscan::ConnectionOptions{.queue_depth = 1U});
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("runtime-clock"));

  const holoscan::ExecutionPlan plan = holoscan::compile(graph);
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
    print_diagnostics(session);
    return 2;
  }

  std::cout << "descriptor_tensor_schema: received one DescriptorFrameT sample\n";
  return 0;
}
