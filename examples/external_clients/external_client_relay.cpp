/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <string>
#include <string_view>
#include <utility>

#include <holoscan/holoscan.hpp>

namespace {

inline constexpr std::string_view kInputEndpoint = "example.external_clients.input";
inline constexpr std::string_view kOutputEndpoint = "example.external_clients.output";

class Relay final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(input, "input").queue_depth(8U);
    spec.output(output, "output").max_emits_per_compute(1U);
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
    return output.emit(sample->data);
  }

  holoscan::Input<std::uint32_t> input;
  holoscan::Output<std::uint32_t> output;
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

}  // namespace

int main(int argc, char** argv) {
  Arguments arguments;
  if (!parse_arguments(argc, argv, arguments)) {
    std::cerr << "usage: external_client_relay --catalog PATH --instance NAME\n";
    return 2;
  }
  std::string input_endpoint{kInputEndpoint};
  input_endpoint += "." + arguments.instance;
  std::string output_endpoint{kOutputEndpoint};
  output_endpoint += "." + arguments.instance;

  holoscan::Graph graph{"external-client-relay"};
  const auto clock = graph.add_clock<holoscan::RealtimeClock>(
      "runtime-clock", holoscan::RealtimeClock::Config{.use_time_since_epoch = true});
  graph.set_default_clock(clock);

  holoscan::TopicOptions input_options;
  input_options.endpoint_name = input_endpoint;
  input_options.capture_time = holoscan::RequireCaptureTime{.clock = clock};
  input_options.attach = holoscan::TopicAttachPolicy{
      .peer = holoscan::PeerRequirement::kRequired,
      .max_wait = std::chrono::seconds{10},
  };
  input_options.bounds.publisher = holoscan::PublisherEndpointBounds{.max_loaned_samples = 1U};

  holoscan::TopicOptions output_options;
  output_options.endpoint_name = output_endpoint;
  output_options.attach = holoscan::TopicAttachPolicy{
      .peer = holoscan::PeerRequirement::kRequired,
      .max_wait = std::chrono::seconds{10},
  };
  output_options.bounds.subscriber =
      holoscan::SubscriberEndpointBounds{.queue_depth = 8U, .max_borrowed_samples = 1U};

  auto imported =
      graph.import_topic<std::uint32_t>("holomq://" + input_endpoint, std::move(input_options));
  auto exported =
      graph.export_topic<std::uint32_t>("holomq://" + output_endpoint, std::move(output_options));
  const auto relay = graph.op<Relay>("relay");
  graph.add_flow(imported.output(), relay->input, holoscan::ConnectionOptions{.queue_depth = 8U});
  graph.add_flow(relay->output, exported.input(), holoscan::ConnectionOptions{.queue_depth = 8U});

  const auto plan = holoscan::compile(graph);
  if (!plan.ok()) {
    std::cerr << plan.json() << '\n';
    return 3;
  }
  if (auto saved = plan.endpoint_catalog().save_json(arguments.catalog); !saved) {
    std::cerr << "failed to save endpoint catalog: " << saved.error().message << '\n';
    return 4;
  }
  std::cout << "catalog ready: " << arguments.catalog << '\n' << std::flush;

  auto session = holoscan::run_async(plan);
  session.wait();
  if (session.termination() != holoscan::RunTermination::kCompleted ||
      !session.diagnostics().empty()) {
    if (!session.diagnostics().empty()) {
      std::cerr << holoscan::format_diagnostics(session.diagnostics(), plan.graph()) << '\n';
    }
    return 5;
  }
  std::cout << "external client relay complete\n";
  return 0;
}
