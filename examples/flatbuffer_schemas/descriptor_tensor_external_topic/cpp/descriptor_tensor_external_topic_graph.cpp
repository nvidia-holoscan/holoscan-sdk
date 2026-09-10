/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <array>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <string>
#include <string_view>
#include <utility>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "external_topic_common.hpp"

namespace {

using namespace std::chrono_literals;
using descriptor_tensor_external_topic::Fields;
using descriptor_tensor_external_topic::Frame;

struct Completion {
  static void signal(const bool completed_successfully) {
    {
      const std::lock_guard lock{mutex};
      done = true;
      success = completed_successfully;
    }
    ready.notify_all();
  }

  [[nodiscard]] static bool wait() {
    std::unique_lock lock{mutex};
    return ready.wait_for(lock, 20s, [] { return done; }) && success;
  }

  static inline std::mutex mutex;
  static inline std::condition_variable ready;
  static inline bool done{};
  static inline bool success{};
};

class FrameRelay final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    auto& input_builder = spec.input(input, "input").queue_depth(2U);
    input_builder.tensor_field(Fields::pixels)
        .expects_tensor(holoscan::TensorInputSpec{
            .representation = descriptor_tensor_external_topic::kPixelsRepresentation});
    input_builder.tensor_field(Fields::calibration)
        .expects_tensor(holoscan::TensorInputSpec{
            .representation = descriptor_tensor_external_topic::kCalibrationRepresentation});

    auto& output_builder = spec.output(output, "output").max_emits_per_compute(1U);
    output_builder.tensor_field(
        Fields::pixels,
        holoscan::TensorOutputSpec{
            .representation = descriptor_tensor_external_topic::kPixelsRepresentation,
            .bounds = holoscan::tensor_bounds(descriptor_tensor_external_topic::kPixelsShape),
            .storage = holoscan::TensorOutputStorage::kRuntimePool,
            .transfer =
                holoscan::TensorTransferPolicy{
                    .foreign_emit_policy = holoscan::InterfaceTensorEmitPolicy::kCopyIfNeeded},
        });
    output_builder.tensor_field(
        Fields::calibration,
        holoscan::TensorOutputSpec{
            .representation = descriptor_tensor_external_topic::kCalibrationRepresentation,
            .bounds = holoscan::tensor_bounds(descriptor_tensor_external_topic::kCalibrationShape),
            .storage = holoscan::TensorOutputStorage::kRuntimePool,
            .transfer =
                holoscan::TensorTransferPolicy{
                    .foreign_emit_policy = holoscan::InterfaceTensorEmitPolicy::kCopyIfNeeded},
        });
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
      return fail("input receive", std::move(sample).error());
    }
    if (sample->data.sequence == descriptor_tensor_external_topic::kAckSequence) {
      Completion::signal(true);
      return {};
    }
    if (sample->data.sequence != descriptor_tensor_external_topic::kDataSequence ||
        sample->data.source != descriptor_tensor_external_topic::kClientSource ||
        sample->data.pixels == nullptr || sample->data.calibration == nullptr) {
      return fail("imported descriptor metadata is invalid");
    }
    const std::array tensors{*sample->data.pixels, *sample->data.calibration};
    if (!descriptor_tensor_external_topic::tensors_match(tensors)) {
      return fail("imported Tensor fields are invalid");
    }

    sample->data.source = descriptor_tensor_external_topic::kGraphSource;
    auto emitted = output.emit(std::move(sample->data));
    if (!emitted) {
      return fail("output emit", std::move(emitted).error());
    }
    return {};
  }

  holoscan::Input<Frame> input;
  holoscan::Output<Frame> output;

 private:
  [[nodiscard]] static holoscan::expected<void, holoscan::Error> fail(
      const std::string_view message) {
    Completion::signal(false);
    std::cerr << message << '\n';
    return holoscan::make_unexpected(
        holoscan::Error{holoscan::ErrorCode::kProtocolError, std::string{message}});
  }

  [[nodiscard]] static holoscan::expected<void, holoscan::Error> fail(
      const std::string_view operation, holoscan::Error error) {
    Completion::signal(false);
    std::cerr << operation << " failed: " << holoscan::runtime_error_message(error) << '\n';
    return holoscan::make_unexpected(std::move(error));
  }
};

struct Arguments {
  std::filesystem::path catalog;
  std::string instance;
};

[[nodiscard]] bool parse_arguments(const int argc, char** argv, Arguments& result) {
  for (int index = 1; index + 1 < argc; index += 2) {
    const std::string_view option{argv[index]};
    if (option == "--catalog") {
      result.catalog = argv[index + 1];
    } else if (option == "--instance") {
      result.instance = argv[index + 1];
    } else {
      return false;
    }
  }
  return argc == 5 && !result.catalog.empty() && !result.instance.empty();
}

void print_diagnostics(const holoscan::RunSession& session) {
  for (const auto& diagnostic : session.diagnostics()) {
    std::cerr << diagnostic.token << " required=" << diagnostic.required
              << " provided=" << diagnostic.provided << '\n';
  }
}

}  // namespace

int main(int argc, char** argv) {
  Arguments arguments;
  if (!parse_arguments(argc, argv, arguments)) {
    std::cerr << "usage: descriptor_tensor_external_topic_graph --catalog PATH --instance NAME\n";
    return 2;
  }

  const std::string import_name =
      descriptor_tensor_external_topic::import_endpoint_name(arguments.instance);
  const std::string export_name =
      descriptor_tensor_external_topic::export_endpoint_name(arguments.instance);
  holoscan::Graph graph{"descriptor-tensor-external-topic"};
  auto imported =
      graph.import_topic<Frame>("holomq://" + import_name,
                                descriptor_tensor_external_topic::import_topic_options(import_name),
                                descriptor_tensor_external_topic::topic_contract(
                                    holoscan::InterfaceTensorEmitPolicy::kCopyIfNeeded));
  auto exported =
      graph.export_topic<Frame>("holomq://" + export_name,
                                descriptor_tensor_external_topic::export_topic_options(export_name),
                                descriptor_tensor_external_topic::topic_contract(
                                    holoscan::InterfaceTensorEmitPolicy::kCopyIfNeeded));
  const auto relay = graph.op<FrameRelay>("relay");
  graph.add_flow(imported.output(), relay->input, holoscan::ConnectionOptions{.queue_depth = 2U});
  graph.add_flow(relay->output, exported.input(), holoscan::ConnectionOptions{.queue_depth = 1U});
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("runtime-clock"));

  const holoscan::ExecutionPlan plan = holoscan::compile(graph);
  if (!plan.ok()) {
    std::cerr << plan.json() << '\n';
    return 3;
  }
  if (auto saved = plan.endpoint_catalog().save_json(arguments.catalog); !saved) {
    std::cerr << "failed to save endpoint catalog: " << saved.error().message << '\n';
    return 4;
  }
  std::cout << "catalog ready: " << arguments.catalog << '\n' << std::flush;

  holoscan::RunSession session = holoscan::run_async(plan);
  const bool completed = Completion::wait();
  session.request_stop();
  session.wait();
  if (!completed || session.termination() != holoscan::RunTermination::kStopped ||
      !session.diagnostics().empty()) {
    print_diagnostics(session);
    return 5;
  }
  std::cout << "descriptor Tensor external-topic graph complete\n";
  return 0;
}
