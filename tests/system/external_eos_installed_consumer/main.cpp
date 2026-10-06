/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <utility>

#include <holoscan/channel/channel.hpp>
#include <holoscan/holoscan.hpp>

namespace {

using namespace std::chrono_literals;

class InstalledExternalEosForwarder final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(input, "input").queue_depth(2U);
    spec.output(output, "output").max_emits_per_compute(1U);
  }

  holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{input});
    return result;
  }

  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override {
    auto sample = input.receive();
    if (!sample) {
      return holoscan::make_unexpected(std::move(sample).error());
    }
    compute_calls.fetch_add(1U, std::memory_order_relaxed);
    return output.emit(std::move(*sample));
  }

  static inline std::atomic<std::uint32_t> compute_calls{};
  holoscan::Input<std::uint32_t> input;
  holoscan::Output<std::uint32_t> output;
};

struct InstalledEosPlan {
  holoscan::ExecutionPlan plan;
  std::string import_name;
  std::string export_name;
};

struct ClientState {
  std::mutex mutex;
  std::condition_variable ready;
  bool session_returned{};
  bool client_finished{};
  bool succeeded{};
};

[[nodiscard]] std::string unique_endpoint_name(std::string_view stem) {
  static std::atomic<std::uint64_t> ordinal{};
  return "installed_external_eos_" + std::string(stem) + "_" +
         std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + "_" +
         std::to_string(ordinal.fetch_add(1U, std::memory_order_relaxed));
}

[[nodiscard]] InstalledEosPlan make_plan() {
  InstalledEosPlan result{
      .import_name = unique_endpoint_name("import"),
      .export_name = unique_endpoint_name("export"),
  };
  holoscan::Graph graph{"installed-external-eos"};
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("runtime-clock"));
  holoscan::TopicOptions import_options;
  import_options.endpoint_name = result.import_name;
  import_options.bounds.publisher = holoscan::PublisherEndpointBounds{.max_loaned_samples = 1U};
  holoscan::TopicOptions export_options;
  export_options.endpoint_name = result.export_name;
  export_options.bounds.subscriber = holoscan::SubscriberEndpointBounds{
      .queue_depth = 2U,
      .max_borrowed_samples = 1U,
  };
  const auto imported =
      graph.import_topic<std::uint32_t>("holomq://" + result.import_name, import_options);
  const auto exported =
      graph.export_topic<std::uint32_t>("holomq://" + result.export_name, export_options);
  const auto forwarder = graph.op<InstalledExternalEosForwarder>("forwarder");
  graph.add_flow(
      imported.output(), forwarder->input, holoscan::ConnectionOptions{.queue_depth = 2U});
  graph.add_flow(
      forwarder->output, exported.input(), holoscan::ConnectionOptions{.queue_depth = 2U});
  result.plan = holoscan::compile(graph);
  return result;
}

template <typename Publisher, typename Subscriber>
[[nodiscard]] bool wait_for_connection(Publisher& publisher, Subscriber& subscriber) {
  const auto deadline = std::chrono::steady_clock::now() + 10s;
  while (std::chrono::steady_clock::now() < deadline) {
    auto publisher_updated = publisher.update_connections();
    auto subscriber_updated = subscriber.has_samples();
    if (!publisher_updated || !subscriber_updated) {
      return false;
    }
    if (publisher.has_subscribers() && subscriber.has_publishers()) {
      return true;
    }
    std::this_thread::sleep_for(1ms);
  }
  return false;
}

template <typename Subscriber>
[[nodiscard]] auto receive_until(Subscriber& subscriber) {
  using Result = decltype(subscriber.receive_sample());
  const auto deadline = std::chrono::steady_clock::now() + 10s;
  while (std::chrono::steady_clock::now() < deadline) {
    Result received = subscriber.receive_sample();
    if (received || received.error().code != holoscan::ErrorCode::kNotReady) {
      return received;
    }
    std::this_thread::sleep_for(1ms);
  }
  return Result{holoscan::make_unexpected(holoscan::Error{holoscan::ErrorCode::kTimeout})};
}

[[nodiscard]] bool run_once(const InstalledEosPlan& installed, std::uint32_t value) {
  auto runtime = holoscan::ChannelRuntime::create();
  if (!runtime) {
    return false;
  }
  auto publisher = runtime->create_publisher<std::uint32_t>(installed.plan.endpoint_catalog(),
                                                            installed.import_name);
  auto subscriber = runtime->create_subscriber<std::uint32_t>(installed.plan.endpoint_catalog(),
                                                              installed.export_name);
  if (!publisher || !subscriber) {
    return false;
  }

  ClientState state;
  std::thread client{[publisher = std::move(*publisher),
                      subscriber = std::move(*subscriber),
                      value,
                      &state]() mutable {
    bool succeeded = wait_for_connection(publisher, subscriber);
    if (succeeded) {
      auto emitted = publisher.emit(value);
      auto producer_closed = publisher.close();
      succeeded = emitted && producer_closed;
    }
    if (succeeded) {
      auto received = receive_until(subscriber);
      succeeded = received && received->data == value;
    }
    if (succeeded) {
      auto remote_end_of_stream = receive_until(subscriber);
      succeeded = !remote_end_of_stream &&
                  remote_end_of_stream.error().code == holoscan::ErrorCode::kClosed;
    }
    if (succeeded) {
      std::unique_lock lock{state.mutex};
      succeeded = state.ready.wait_for(lock, 10s, [&state] { return state.session_returned; });
    }
    if (succeeded) {
      succeeded = static_cast<bool>(subscriber.close());
    }
    {
      const std::lock_guard lock{state.mutex};
      state.succeeded = succeeded;
      state.client_finished = true;
    }
    state.ready.notify_all();
  }};

  holoscan::RunSession session = holoscan::run_async(installed.plan);
  const auto deadline = std::chrono::steady_clock::now() + 10s;
  bool client_failed = false;
  while (session.is_running() && !client_failed && std::chrono::steady_clock::now() < deadline) {
    std::unique_lock lock{state.mutex};
    state.ready.wait_for(lock, 1ms, [&state] { return state.client_finished; });
    client_failed = state.client_finished && !state.succeeded;
  }
  const bool completed_before_deadline = !session.is_running();
  if (client_failed || !completed_before_deadline) {
    session.request_stop();
  }
  session.wait();
  const holoscan::RunTermination termination = session.termination();
  {
    const std::lock_guard lock{state.mutex};
    state.session_returned = true;
  }
  state.ready.notify_all();
  client.join();
  bool client_succeeded = false;
  {
    const std::lock_guard lock{state.mutex};
    client_succeeded = state.client_finished && state.succeeded;
  }
  return completed_before_deadline && termination == holoscan::RunTermination::kCompleted &&
         session.diagnostics().empty() && client_succeeded;
}

}  // namespace

// REQ: CHAN-132
// REQ: CHAN-136
// REQ: RUNTIME-017
int main() {
  const InstalledEosPlan installed = make_plan();
  if (!installed.plan.ok()) {
    return 1;
  }
  InstalledExternalEosForwarder::compute_calls.store(0U, std::memory_order_relaxed);
  if (!run_once(installed, 17U) || !run_once(installed, 29U)) {
    return 2;
  }
  return InstalledExternalEosForwarder::compute_calls.load(std::memory_order_relaxed) == 2U ? 0 : 3;
}
