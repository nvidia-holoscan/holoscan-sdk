/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <array>
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
#include <holoscan/core/compile.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/run.hpp>
// Provides generated owning types and schema-trait specializations.
#include <holoscan/schema/sdk_schemas.hpp>  // NOLINT(misc-include-cleaner)
#include <holoscan/time/realtime_clock.hpp>

#include "channel/channel_catalog_binding.hpp"
#include "channel/channel_inline_payload_test_access.hpp"
#include "channel/process_identity.hpp"

namespace holoscan {
namespace {

using namespace std::chrono_literals;

// Thread-safe static probe capturing count and payload of received StringT samples.
// Used by RuntimeExternalSchemaSink to enable cross-thread delivery verification
// after the session terminates (or is stopped).
struct RuntimeExternalSchemaProbe {
  static void reset() {
    const std::lock_guard lock(mutex);
    count.store(0U, std::memory_order_relaxed);
    observed_value.clear();
  }

  static void record(const Sample<schema::StringT>& sample) {
    {
      const std::lock_guard lock(mutex);
      observed_value = sample.data.data;
      count.fetch_add(1U, std::memory_order_release);
    }
    ready.notify_all();
  }

  [[nodiscard]] static bool wait_for_sample(std::chrono::milliseconds timeout = 5s) {
    std::unique_lock lock(mutex);
    return ready.wait_for(
        lock, timeout, [] { return count.load(std::memory_order_acquire) != 0U; });
  }

  [[nodiscard]] static std::string value_snapshot() {
    const std::lock_guard lock(mutex);
    return observed_value;
  }

  static inline std::atomic<std::uint32_t> count{};
  static inline std::mutex mutex;
  static inline std::condition_variable ready;
  static inline std::string observed_value;
};

// Sink operator that receives schema::StringT on a depth-1 input and forwards each sample to
// RuntimeExternalSchemaProbe. Used by both external-schema tests to verify typed delivery.
class RuntimeExternalSchemaSink final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override { spec.input(input, "input").queue_depth(1U); }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    auto sample = input.receive();
    if (!sample) {
      return make_unexpected(std::move(sample.error()));
    }
    RuntimeExternalSchemaProbe::record(*sample);
    return {};
  }

  Input<schema::StringT> input;
};

// Generates a process-scoped HoloMQ endpoint name to prevent subscriber cross-contamination
// when independently launched test processes share the same HoloMQ daemon.
[[nodiscard]] std::string unique_schema_endpoint_name() {
  static std::atomic<std::uint64_t> ordinal{};
  return "runtime_external_schema_" + std::to_string(channel_internal::current_process_identity()) +
         "_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + "_" +
         std::to_string(ordinal.fetch_add(1U, std::memory_order_relaxed));
}

// Polls a publisher until at least one subscriber connects or the timeout elapses.
// Subscriber discovery is asynchronous over the HoloMQ transport; polling avoids
// a fixed sleep while still bounding the wait to the test timeout.
[[nodiscard]] bool wait_for_subscriber(ChannelPublisher<schema::StringT>& publisher,
                                       std::chrono::milliseconds timeout = 5s) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    auto refreshed = publisher.update_connections();
    if (refreshed && publisher.has_subscribers()) {
      return true;
    }
    std::this_thread::sleep_for(1ms);
  }
  return publisher.has_subscribers();
}

// Polls a RunSession until it exits the running state or the timeout elapses.
[[nodiscard]] bool wait_for_terminal(RunSession& session, std::chrono::milliseconds timeout = 5s) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (session.is_running() && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(1ms);
  }
  return !session.is_running();
}

// [external: holomq://endpoint] --[queue=1]--> RuntimeExternalSchemaSink("sink")
// Six catalog-frame corruption variants are each run in a separate graph instance;
// all must fail the session before the typed sample reaches the sink.
//
// Malformed catalog bytes must cross the actual external transport and Runtime drain path, but
// must fail before generated-schema materialization can publish readiness or invoke the sink.
// REQ: CHAN-075
// REQ: RUNTIME-019
// REQ: SCHEMA-003
TEST(RuntimeExternalSchemaTest, CorruptCatalogFramesFailBeforeTypedExposure) {
  using channel_internal::CatalogChannelBindingTestAccess;
  using channel_internal::CatalogSchemaFrameCorruption;
  using channel_internal::InlinePayloadTestAccess;

  constexpr std::array kCorruptions{
      CatalogSchemaFrameCorruption::kMalformedEnvelope,
      CatalogSchemaFrameCorruption::kSchemaId,
      CatalogSchemaFrameCorruption::kPayloadSize,
      CatalogSchemaFrameCorruption::kEndpointGeneration,
      CatalogSchemaFrameCorruption::kPayloadVerifier,
      CatalogSchemaFrameCorruption::kEnvelopeAbi,
  };

  for (const auto corruption : kCorruptions) {
    SCOPED_TRACE(static_cast<std::uint32_t>(corruption));
    RuntimeExternalSchemaProbe::reset();
    const std::string endpoint_name = unique_schema_endpoint_name();

    Graph graph{"runtime-external-schema-corruption"};
    const ClockKey runtime_clock = graph.add_clock<RealtimeClock>("runtime-clock");
    graph.set_default_clock(runtime_clock);
    TopicOptions options;
    options.endpoint_name = endpoint_name;
    options.bounds.publisher = PublisherEndpointBounds{.max_loaned_samples = 1U};
    const auto imported =
        graph.import_topic<schema::StringT>("holomq://" + endpoint_name, std::move(options));
    const auto sink = graph.op<RuntimeExternalSchemaSink>("sink");
    graph.add_flow(imported.output(), sink->input, ConnectionOptions{.queue_depth = 1U});

    const ExecutionPlan plan = compile(graph);
    ASSERT_TRUE(plan.ok()) << plan.json();
    auto external_runtime = ChannelRuntime::create();
    ASSERT_TRUE(external_runtime) << runtime_error_message(external_runtime.error());
    auto publisher =
        external_runtime->create_publisher<schema::StringT>(plan.endpoint_catalog(), endpoint_name);
    ASSERT_TRUE(publisher) << runtime_error_message(publisher.error());

    RunSession session = run_async(plan);
    if (!wait_for_subscriber(*publisher)) {
      session.request_stop();
      session.wait();
      FAIL() << "generated-schema Graph subscriber did not connect; termination="
             << static_cast<int>(session.termination()) << "; diagnostics="
             << (session.diagnostics().empty() ? "<none>" : session.diagnostics().front().token)
             << "; plan=" << plan.json();
    }

    // Key setup: corrupt_next_schema_publication injects the fault into the next catalog frame
    //            so the typed payload arrives without a valid schema negotiation header.
    auto* const raw_publisher = InlinePayloadTestAccess::publisher_endpoint(*publisher);
    ASSERT_NE(raw_publisher, nullptr);
    ASSERT_TRUE(CatalogChannelBindingTestAccess::corrupt_next_schema_publication(*raw_publisher,
                                                                                 corruption));
    schema::StringT value;
    value.data = "must not reach the Runtime Operator";
    auto emitted = publisher->emit(std::move(value));
    ASSERT_TRUE(emitted) << runtime_error_message(emitted.error());

    const bool terminal = wait_for_terminal(session);
    if (!terminal) {
      session.request_stop();
    }
    session.wait();

    EXPECT_TRUE(terminal) << "corrupt schema frame did not fail the session";
    // Key assertion: count == 0 and termination == kFailed — the sink never receives the sample.
    EXPECT_EQ(session.termination(), RunTermination::kFailed);
    EXPECT_EQ(RuntimeExternalSchemaProbe::count.load(std::memory_order_relaxed), 0U);
    EXPECT_FALSE(session.diagnostics().empty());
  }
}

// [external: holomq://endpoint] --[queue=1]--> RuntimeExternalSchemaSink("sink")
//
// A clean catalog frame must traverse the compiled external import, generated verifier and owning
// materializer before the selected Sample reaches the real Operator callback.
// REQ: CHAN-075
// REQ: RUNTIME-019
TEST(RuntimeExternalSchemaTest, ValidCatalogFrameReachesCompiledGraphOperator) {
  RuntimeExternalSchemaProbe::reset();
  const std::string endpoint_name = unique_schema_endpoint_name();

  Graph graph{"runtime-external-schema-valid"};
  const ClockKey runtime_clock = graph.add_clock<RealtimeClock>("runtime-clock");
  graph.set_default_clock(runtime_clock);
  TopicOptions options;
  options.endpoint_name = endpoint_name;
  options.bounds.publisher = PublisherEndpointBounds{.max_loaned_samples = 1U};
  const auto imported =
      graph.import_topic<schema::StringT>("holomq://" + endpoint_name, std::move(options));
  const auto sink = graph.op<RuntimeExternalSchemaSink>("sink");
  graph.add_flow(imported.output(), sink->input, ConnectionOptions{.queue_depth = 1U});

  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  auto external_runtime = ChannelRuntime::create();
  ASSERT_TRUE(external_runtime) << runtime_error_message(external_runtime.error());
  auto publisher =
      external_runtime->create_publisher<schema::StringT>(plan.endpoint_catalog(), endpoint_name);
  ASSERT_TRUE(publisher) << runtime_error_message(publisher.error());

  RunSession session = run_async(plan);
  if (!wait_for_subscriber(*publisher)) {
    session.request_stop();
    session.wait();
    FAIL() << "generated-schema Graph subscriber did not connect; termination="
           << static_cast<int>(session.termination()) << "; diagnostics="
           << (session.diagnostics().empty() ? "<none>" : session.diagnostics().front().token)
           << "; plan=" << plan.json();
  }

  // Key setup: a well-formed publisher emits one StringT sample after subscriber discovery.
  constexpr std::string_view kExpected = "valid external generated-schema payload";
  schema::StringT value;
  value.data = kExpected;
  auto emitted = publisher->emit(std::move(value));
  ASSERT_TRUE(emitted) << runtime_error_message(emitted.error());

  const bool observed = RuntimeExternalSchemaProbe::wait_for_sample();
  session.request_stop();
  session.wait();

  // Key assertion: count == 1, value matches kExpected, and termination == kStopped.
  ASSERT_TRUE(observed);
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
  EXPECT_EQ(RuntimeExternalSchemaProbe::count.load(std::memory_order_acquire), 1U);
  EXPECT_EQ(RuntimeExternalSchemaProbe::value_snapshot(), kExpected);
}

}  // namespace
}  // namespace holoscan
