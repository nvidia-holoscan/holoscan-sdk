/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
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
#include <holoscan/time/realtime_clock.hpp>

#include "../support/optional_assertions.hpp"
#include "channel/process_identity.hpp"
#include "compiler/compile_internal.hpp"
#include "compiler/execution_plan_state.hpp"
#include "runtime/runtime_phase_counters.hpp"
#include "runtime/runtime_test_access.hpp"

namespace holoscan {

// Simple two-field POD used as the cross-runtime wire type. Registered via ChannelTypeIdentity
// and ChannelPodLayout so the Channel codec can serialize/deserialize without coupling the
// external publisher and graph-internal operators through shared headers.
// Representative test: IndependentPublisherReachesCompiledGraphImportWithMetadata.
struct RuntimeExternalPod {
  std::uint32_t frame{};
  std::int32_t value{};

  friend bool operator==(const RuntimeExternalPod&, const RuntimeExternalPod&) noexcept = default;
};

// Registers the stable catalog identity across independent ChannelRuntime instances.
// The name+version pair is the only interoperability contract between the external process
// and the graph Runtime; no shared headers or FlatBuffers schemas are required.
template <>
struct ChannelTypeIdentity<RuntimeExternalPod> {
  static constexpr std::string_view name = "holoscan.test.RuntimeExternalPod";
  static constexpr std::uint32_t version = 1U;
};

// Declares the fieldwise byte layout so the Channel codec can encode/decode the struct
// without generated FlatBuffers or shared headers between producer and consumer runtimes.
template <>
struct ChannelPodLayout<RuntimeExternalPod> {
  static constexpr auto fields = pod_fields(
      pod_field("frame", &RuntimeExternalPod::frame, offsetof(RuntimeExternalPod, frame)),
      pod_field("value", &RuntimeExternalPod::value, offsetof(RuntimeExternalPod, value)));
};

namespace {

using namespace std::chrono_literals;

// Thread-safe observation latch used by RuntimeExternalPodSink to transfer a received sample
// to the test thread. Exposes reset/record/wait_for_sample to avoid shared-state races between
// the runtime dispatch thread and test-main assertions.
// Representative test: IndependentPublisherReachesCompiledGraphImportWithMetadata.
struct RuntimeExternalPodProbe {
  static void reset() noexcept {
    const std::lock_guard lock{mutex};
    observed.reset();
  }

  static void record(const Sample<RuntimeExternalPod>& sample) {
    {
      const std::lock_guard lock{mutex};
      // Keep the payload and metadata without the sample's read claim. A held claim keeps the
      // sink's input from retiring its end of stream, so the session could not complete naturally.
      observed.emplace(sample.data, sample.metadata);
    }
    ready.notify_all();
  }

  [[nodiscard]] static std::optional<Sample<RuntimeExternalPod>> wait_for_sample(
      std::chrono::milliseconds timeout = 5s) {
    std::unique_lock lock{mutex};
    if (!ready.wait_for(lock, timeout, [] { return observed.has_value(); })) {
      return std::nullopt;
    }
    return observed;
  }

  static inline std::mutex mutex;
  static inline std::condition_variable ready;
  static inline std::optional<Sample<RuntimeExternalPod>> observed;
};

// Single-input sink operator that records each received Sample<RuntimeExternalPod> into the
// RuntimeExternalPodProbe latch so the test thread can observe it. Queue depth is 1.
// Representative test: IndependentPublisherReachesCompiledGraphImportWithMetadata.
class RuntimeExternalPodSink final : public Operator<> {
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
    RuntimeExternalPodProbe::record(*sample);
    return {};
  }

  Input<RuntimeExternalPod> input;
};

// Input-driven pass-through used to characterize external producer EOS through an ordinary
// Operator and back to a live external subscriber. The callback counter distinguishes zero-data
// terminal propagation from an accidentally fabricated payload activation.
class RuntimeExternalPodForwarder final : public Operator<> {
 public:
  static void reset() noexcept { compute_count.store(0U, std::memory_order_relaxed); }

  void setup(OperatorSpec& spec) override {
    spec.input(input, "input").queue_depth(4U);
    spec.output(output, "output").max_emits_per_compute(1U);
  }

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
    compute_count.fetch_add(1U, std::memory_order_relaxed);
    return output.emit(std::move(*sample));
  }

  Input<RuntimeExternalPod> input;
  Output<RuntimeExternalPod> output;
  static inline std::atomic<std::uint32_t> compute_count{};
};

// A pass-through with the ordinary any-input trigger. This separately proves that external EOS
// propagation follows input-driven authority rather than relying on OnEach-specific behavior.
class RuntimeExternalPodOnAnyForwarder final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(input, "input").queue_depth(4U);
    spec.output(output, "output").max_emits_per_compute(1U);
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnAnyInput{});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    auto sample = input.receive();
    if (!sample) {
      return make_unexpected(std::move(sample.error()));
    }
    return output.emit(std::move(*sample));
  }

  Input<RuntimeExternalPod> input;
  Output<RuntimeExternalPod> output;
};

// Records one activation of a two-input aligned join. This keeps the fan-in EOS test independent
// of output publication: natural completion must wait for both imported producer generations.
struct RuntimeExternalPodFanInProbe {
  static void reset() noexcept {
    const std::lock_guard lock{mutex};
    observed.reset();
  }

  static void record(const RuntimeExternalPod& first, const RuntimeExternalPod& second) {
    {
      const std::lock_guard lock{mutex};
      observed = std::array{first, second};
    }
    ready.notify_all();
  }

  [[nodiscard]] static std::optional<std::array<RuntimeExternalPod, 2U>> wait_for_values(
      std::chrono::milliseconds timeout = 5s) {
    std::unique_lock lock{mutex};
    if (!ready.wait_for(lock, timeout, [] { return observed.has_value(); })) {
      return std::nullopt;
    }
    return observed;
  }

  static inline std::mutex mutex;
  static inline std::condition_variable ready;
  static inline std::optional<std::array<RuntimeExternalPod, 2U>> observed;
};

class RuntimeExternalPodAlignedJoin final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(first, "first").queue_depth(2U);
    spec.input(second, "second").queue_depth(2U);
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result
        .trigger(OnAligned{
            .ports = ports(first, second),
            .window = 1ms,
        })
        .input(first, Exact{})
        .input(second, Exact{});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    auto first_sample = first.receive();
    if (!first_sample) {
      return make_unexpected(std::move(first_sample).error());
    }
    auto second_sample = second.receive();
    if (!second_sample) {
      return make_unexpected(std::move(second_sample).error());
    }
    RuntimeExternalPodFanInProbe::record(first_sample->data, second_sample->data);
    return {};
  }

  Input<RuntimeExternalPod> first;
  Input<RuntimeExternalPod> second;
};

// Source operator with a 30-second clock period that gates emission behind an external
// allow_emit() signal, enabling the test thread to sequence publisher readiness before
// the sample is sent. Exposes coordination via static mutex+condvar+atomic flags.
// Representative test: CompiledGraphExportReachesIndependentSubscriberWithMetadata.
class RuntimeExternalPodSource final : public Operator<> {
 public:
  static void reset() noexcept {
    const std::lock_guard lock{coordination_mutex};
    compute_started = false;
    emit_allowed = false;
    compute_count.store(0U, std::memory_order_relaxed);
    emit_error.store(ErrorCode::kSuccess, std::memory_order_relaxed);
  }

  [[nodiscard]] static bool wait_for_compute_started(std::chrono::milliseconds timeout = 5s) {
    std::unique_lock lock{coordination_mutex};
    return coordination_ready.wait_for(lock, timeout, [] { return compute_started; });
  }

  static void allow_emit() noexcept {
    {
      const std::lock_guard lock{coordination_mutex};
      emit_allowed = true;
    }
    coordination_ready.notify_all();
  }

  void setup(OperatorSpec& spec) override {
    spec.output(output, "output").max_emits_per_compute(1U);
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 30s});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    compute_count.fetch_add(1U, std::memory_order_relaxed);
    {
      std::unique_lock lock{coordination_mutex};
      compute_started = true;
      coordination_ready.notify_all();
      if (!coordination_ready.wait_for(lock, 5s, [] { return emit_allowed; })) {
        emit_error.store(ErrorCode::kTimeout, std::memory_order_relaxed);
        return make_unexpected(Error{ErrorCode::kTimeout});
      }
    }

    auto emitted = output.emit(Sample<RuntimeExternalPod>{
        RuntimeExternalPod{.frame = 41U, .value = -73},
        SampleMetadata{
            .trace_id = 0x1357U,
            .frame_id = 0x2468U,
            .flags = SampleFlags::kDegraded,
        },
    });
    if (!emitted) {
      emit_error.store(emitted.error().code, std::memory_order_relaxed);
    }
    return emitted;
  }

  Output<RuntimeExternalPod> output;
  static inline std::atomic<std::uint32_t> compute_count{};
  static inline std::atomic<ErrorCode> emit_error{ErrorCode::kSuccess};
  static inline std::mutex coordination_mutex;
  static inline std::condition_variable coordination_ready;
  static inline bool compute_started{};
  static inline bool emit_allowed{};
};

// Generates a unique HoloMQ endpoint name per test invocation using the process identity,
// a monotonic counter, and a steady-clock timestamp to prevent cross-process collisions.
[[nodiscard]] std::string unique_pod_endpoint_name() {
  static std::atomic<std::uint64_t> ordinal{};
  return "runtime_external_pod_" + std::to_string(channel_internal::current_process_identity()) +
         "_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + "_" +
         std::to_string(ordinal.fetch_add(1U, std::memory_order_relaxed));
}

// Polls publisher.update_connections() until at least one subscriber is discovered or the
// timeout elapses. Returns kTimeout on failure; other errors are returned immediately.
template <typename T>
[[nodiscard]] expected<void, Error> wait_for_subscriber(ChannelPublisher<T>& publisher,
                                                        std::chrono::milliseconds timeout = 5s) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    auto refreshed = publisher.update_connections();
    if (!refreshed) {
      if (refreshed.error().code != ErrorCode::kNotReady) {
        return make_unexpected(std::move(refreshed).error());
      }
    } else if (publisher.has_subscribers()) {
      return {};
    }
    std::this_thread::sleep_for(1ms);
  }
  return publisher.has_subscribers() ? expected<void, Error>{}
                                     : make_unexpected(Error{ErrorCode::kTimeout});
}

// Polls subscriber.has_samples() until two consecutive iterations observe a publisher,
// guarding against transient discovery races. Returns kTimeout on failure.
template <typename T>
[[nodiscard]] expected<void, Error> wait_for_publisher(ChannelSubscriber<T>& subscriber,
                                                       std::chrono::milliseconds timeout = 5s) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  std::uint32_t consecutive_observations = 0U;
  while (std::chrono::steady_clock::now() < deadline) {
    auto refreshed = subscriber.has_samples();
    if (!refreshed) {
      return make_unexpected(std::move(refreshed.error()));
    }
    if (subscriber.has_publishers()) {
      ++consecutive_observations;
      if (consecutive_observations == 2U) {
        return {};
      }
    } else {
      consecutive_observations = 0U;
    }
    std::this_thread::sleep_for(1ms);
  }
  return make_unexpected(Error{ErrorCode::kTimeout});
}

// Polls subscriber.receive_sample() until a sample is available or the timeout elapses.
// Only kNotReady is retried; any other error is returned immediately.
template <typename T>
[[nodiscard]] expected<Sample<T>, Error> wait_for_sample(ChannelSubscriber<T>& subscriber,
                                                         std::chrono::milliseconds timeout = 5s) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    auto received = subscriber.receive_sample();
    if (received || received.error().code != ErrorCode::kNotReady) {
      return received;
    }
    std::this_thread::sleep_for(1ms);
  }
  return make_unexpected(Error{ErrorCode::kTimeout});
}

// Bounds lifecycle characterization without converting a timeout into a successful stop result.
// The caller may request stop only after this returns false so a failed test still retires safely.
[[nodiscard]] bool wait_for_terminal(RunSession& session, std::chrono::milliseconds timeout = 5s) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (session.is_running() && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(1ms);
  }
  return !session.is_running();
}

enum class PodEosPartitioning : std::uint8_t { kAutomatic, kIncludeAll };
enum class PodEosTrigger : std::uint8_t { kOnEach, kOnAnyInput };

struct PodEosTopology {
  ExecutionPlan plan;
  std::string import_name;
  std::string export_name;
};

[[nodiscard]] PodEosTopology make_pod_eos_topology(PodEosPartitioning partitioning,
                                                   PodEosTrigger trigger = PodEosTrigger::kOnEach,
                                                   bool require_export_peer = false) {
  const std::string import_name = unique_pod_endpoint_name();
  const std::string export_name = unique_pod_endpoint_name();
  Graph graph{partitioning == PodEosPartitioning::kAutomatic
                  ? "runtime-external-pod-eos-automatic"
                  : "runtime-external-pod-eos-include-all"};
  const ClockKey runtime_clock = graph.add_clock<RealtimeClock>("runtime-clock");
  graph.set_default_clock(runtime_clock);
  TopicOptions import_options;
  import_options.endpoint_name = import_name;
  import_options.bounds.publisher = PublisherEndpointBounds{.max_loaned_samples = 1U};
  TopicOptions export_options;
  export_options.endpoint_name = export_name;
  if (require_export_peer) {
    export_options.attach.peer = PeerRequirement::kRequired;
  }
  export_options.bounds.subscriber = SubscriberEndpointBounds{
      .queue_depth = 4U,
      .max_borrowed_samples = 1U,
  };
  const auto imported =
      graph.import_topic<RuntimeExternalPod>("holomq://" + import_name, std::move(import_options));
  const auto exported =
      graph.export_topic<RuntimeExternalPod>("holomq://" + export_name, std::move(export_options));
  if (trigger == PodEosTrigger::kOnEach) {
    const auto forwarder = graph.op<RuntimeExternalPodForwarder>("forwarder");
    graph.add_flow(imported.output(), forwarder->input, ConnectionOptions{.queue_depth = 4U});
    graph.add_flow(forwarder->output, exported.input(), ConnectionOptions{.queue_depth = 4U});
  } else {
    const auto forwarder = graph.op<RuntimeExternalPodOnAnyForwarder>("forwarder");
    graph.add_flow(imported.output(), forwarder->input, ConnectionOptions{.queue_depth = 4U});
    graph.add_flow(forwarder->output, exported.input(), ConnectionOptions{.queue_depth = 4U});
  }
  if (partitioning == PodEosPartitioning::kIncludeAll) {
    graph.partition("application").include_all();
  }
  return PodEosTopology{
      .plan = compile(graph),
      .import_name = import_name,
      .export_name = export_name,
  };
}

[[nodiscard]] testing::AssertionResult run_pod_eos_once(
    const PodEosTopology& topology, std::span<const Sample<RuntimeExternalPod>> expected_samples) {
  if (!topology.plan.ok()) {
    return testing::AssertionFailure() << "external EOS plan rejected: " << topology.plan.json();
  }
  auto external_runtime = ChannelRuntime::create();
  if (!external_runtime) {
    return testing::AssertionFailure() << "external Channel Runtime creation failed: "
                                       << runtime_error_message(external_runtime.error());
  }
  auto publisher = external_runtime->create_publisher<RuntimeExternalPod>(
      topology.plan.endpoint_catalog(), topology.import_name);
  auto subscriber = external_runtime->create_subscriber<RuntimeExternalPod>(
      topology.plan.endpoint_catalog(), topology.export_name);
  if (!publisher || !subscriber) {
    return testing::AssertionFailure()
           << "external endpoint creation failed: publisher="
           << (publisher ? "ok" : runtime_error_message(publisher.error()))
           << ", subscriber=" << (subscriber ? "ok" : runtime_error_message(subscriber.error()));
  }

  RunSession session = run_async(topology.plan);
  const auto fail_session = [&](std::string_view message) {
    if (session.is_running()) {
      session.request_stop();
    }
    session.wait();
    static_cast<void>(subscriber->close());
    return testing::AssertionFailure()
           << message << "; termination=" << static_cast<int>(session.termination())
           << "; diagnostic="
           << (session.diagnostics().empty() ? "<none>" : session.diagnostics().front().token);
  };

  auto import_connected = wait_for_subscriber(*publisher);
  auto export_connected = wait_for_publisher(*subscriber);
  if (!import_connected || !export_connected) {
    return fail_session("external EOS endpoints did not attach");
  }

  for (const auto& expected : expected_samples) {
    auto emitted = publisher->emit_sample(expected);
    if (!emitted) {
      return fail_session("external producer did not accept a data sample");
    }
  }
  auto producer_closed = publisher->close();
  if (!producer_closed) {
    return fail_session("external producer did not accept ordered close");
  }

  for (const auto& expected : expected_samples) {
    auto received = wait_for_sample(*subscriber);
    if (!received) {
      return fail_session("external subscriber did not receive every accepted sample");
    }
    if (received->data != expected.data ||
        received->metadata.trace_id != expected.metadata.trace_id ||
        received->metadata.frame_id != expected.metadata.frame_id ||
        received->metadata.flags != expected.metadata.flags) {
      return fail_session("external subscriber observed reordered or altered data before EOS");
    }
  }

  auto remote_end_of_stream = wait_for_sample(*subscriber);
  if (remote_end_of_stream || remote_end_of_stream.error().code != ErrorCode::kClosed) {
    return fail_session("live external subscriber did not observe remote EOS");
  }

  const bool completed = wait_for_terminal(session);
  if (!completed) {
    return fail_session("external EOS did not complete the session");
  }
  session.wait();
  if (session.termination() != RunTermination::kCompleted || !session.diagnostics().empty()) {
    return fail_session("external EOS froze an unexpected terminal result");
  }

  auto locally_closed = subscriber->close();
  if (!locally_closed) {
    return testing::AssertionFailure()
           << "local subscriber cleanup failed after remote EOS and session completion: "
           << runtime_error_message(locally_closed.error());
  }
  auto after_local_close = subscriber->receive_sample();
  if (after_local_close || after_local_close.error().code != ErrorCode::kClosed) {
    return testing::AssertionFailure()
           << "locally closed subscriber did not retain its closed facade state";
  }
  return testing::AssertionSuccess();
}

// Graph topology:
//   external_publisher --[holomq://endpoint]--> imported --[queue_depth=1]--> sink
// The external publisher uses an independent ChannelRuntime; the graph Runtime owns the
// import topic and sink.
// The external publisher and compiled Graph Runtime own independent ChannelRuntime objects. The
// catalog identity and fieldwise codec are therefore the only supported POD interoperability seam.
// REQ: CHAN-074
// REQ: EXTERNAL-002
// REQ: RUNTIME-015
// REQ: RUNTIME-017
TEST(RuntimeExternalPodTest, IndependentPublisherReachesCompiledGraphImportWithMetadata) {
  // Key setup: external ChannelRuntime creates the publisher against plan.endpoint_catalog();
  // the graph Runtime materializes the import topic after compile().
  RuntimeExternalPodProbe::reset();
  const std::string endpoint_name = unique_pod_endpoint_name();

  Graph graph{"runtime-external-pod-import"};
  const ClockKey runtime_clock = graph.add_clock<RealtimeClock>("runtime-clock");
  graph.set_default_clock(runtime_clock);
  TopicOptions options;
  options.endpoint_name = endpoint_name;
  options.bounds.publisher = PublisherEndpointBounds{.max_loaned_samples = 1U};
  const auto imported =
      graph.import_topic<RuntimeExternalPod>("holomq://" + endpoint_name, std::move(options));
  const auto sink = graph.op<RuntimeExternalPodSink>("sink");
  graph.add_flow(imported.output(), sink->input, ConnectionOptions{.queue_depth = 1U});

  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  auto external_runtime = ChannelRuntime::create();
  ASSERT_TRUE(external_runtime) << runtime_error_message(external_runtime.error());
  auto publisher = external_runtime->create_publisher<RuntimeExternalPod>(plan.endpoint_catalog(),
                                                                          endpoint_name);
  ASSERT_TRUE(publisher) << runtime_error_message(publisher.error());

  RunSession session = run_async(plan);
  auto connected = wait_for_subscriber(*publisher);
  if (!connected) {
    session.request_stop();
    session.wait();
    FAIL() << "custom-POD Graph subscriber did not connect: "
           << runtime_error_message(connected.error())
           << "; termination=" << static_cast<int>(session.termination()) << "; diagnostics="
           << (session.diagnostics().empty() ? "<none>" : session.diagnostics().front().token)
           << "; plan=" << plan.json();
  }
  auto emitted = publisher->emit_sample(Sample<RuntimeExternalPod>{
      RuntimeExternalPod{.frame = 17U, .value = -42},
      SampleMetadata{
          .trace_id = 0x1020U,
          .frame_id = 0x3040U,
          .flags = SampleFlags::kDegraded,
      },
  });
  ASSERT_TRUE(emitted) << runtime_error_message(emitted.error());
  const auto received = RuntimeExternalPodProbe::wait_for_sample();

  session.request_stop();
  session.wait();
  if (!received.has_value()) {
    ADD_FAILURE() << "runtime did not deliver the external POD sample";
    return;
  }
  // Key assertion: sample data, trace_id, frame_id, flags, byte_size, and payload_kind all
  // match the emission, confirming end-to-end fieldwise codec fidelity including metadata.
  const auto& received_sample = *received;
  EXPECT_EQ(received_sample.data, (RuntimeExternalPod{.frame = 17U, .value = -42}));
  EXPECT_GT(received_sample.metadata.sequence, 0U);
  EXPECT_EQ(received_sample.metadata.trace_id, 0x1020U);
  EXPECT_EQ(received_sample.metadata.frame_id, 0x3040U);
  EXPECT_EQ(received_sample.metadata.flags, SampleFlags::kDegraded);
  EXPECT_EQ(received_sample.metadata.byte_size, sizeof(RuntimeExternalPod));
  EXPECT_EQ(received_sample.metadata.payload_kind, PayloadKind::kInlineTyped);
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
}

// An external producer generation closes through one ordinary input-driven Operator. The live
// output subscriber must observe exact data and remote EOS before the session freezes as natural
// completion; local subscriber close is cleanup only and occurs after both semantic witnesses.
// REQ: CHAN-132
// REQ: CHAN-136
// REQ: RUNTIME-017
TEST(RuntimeExternalPodTest, ProducerEndOfStreamPropagatesThroughForwarderToNaturalCompletion) {
  RuntimeExternalPodForwarder::reset();
  const std::array expected_samples{Sample<RuntimeExternalPod>{
      RuntimeExternalPod{.frame = 73U, .value = -91},
      SampleMetadata{
          .trace_id = 0x5152U,
          .frame_id = 0x6162U,
          .flags = SampleFlags::kDegraded,
      },
  }};
  const PodEosTopology topology = make_pod_eos_topology(PodEosPartitioning::kAutomatic);
  EXPECT_TRUE(run_pod_eos_once(topology, expected_samples));
  EXPECT_EQ(RuntimeExternalPodForwarder::compute_count.load(std::memory_order_relaxed), 1U);
}

// OnAnyInput is input-driven even though it does not name one trigger port. Its accepted input
// data must retire before EOS closes the forwarder's output and completes the graph.
// REQ: CHAN-132
// REQ: CHAN-136
// REQ: RUNTIME-017
TEST(RuntimeExternalPodTest, OnAnyInputPropagatesExternalEndOfStreamAfterAcceptedData) {
  const PodEosTopology topology =
      make_pod_eos_topology(PodEosPartitioning::kAutomatic, PodEosTrigger::kOnAnyInput);
  const std::array samples{Sample<RuntimeExternalPod>{
      RuntimeExternalPod{.frame = 909U, .value = -909},
      SampleMetadata{.trace_id = 9009U, .frame_id = 9909U},
  }};
  EXPECT_TRUE(run_pod_eos_once(topology, samples));
}

// REQ: CHAN-132
// REQ: CHAN-136
// REQ: RUNTIME-017
TEST(RuntimeExternalPodTest, ZeroDataProducerClosePropagatesWithoutFabricatingAComputeCall) {
  RuntimeExternalPodForwarder::reset();
  const PodEosTopology topology = make_pod_eos_topology(PodEosPartitioning::kIncludeAll);
  EXPECT_TRUE(run_pod_eos_once(topology, {}));
  EXPECT_EQ(RuntimeExternalPodForwarder::compute_count.load(std::memory_order_relaxed), 0U);
}

// REQ: CHAN-132
// REQ: CHAN-136
// REQ: RUNTIME-017
TEST(RuntimeExternalPodTest, BoundedBurstDrainsInOrderBeforeRemoteEndOfStream) {
  RuntimeExternalPodForwarder::reset();
  const std::array expected_samples{
      Sample<RuntimeExternalPod>{RuntimeExternalPod{.frame = 1U, .value = -1},
                                 SampleMetadata{.trace_id = 11U, .frame_id = 101U}},
      Sample<RuntimeExternalPod>{RuntimeExternalPod{.frame = 2U, .value = -2},
                                 SampleMetadata{.trace_id = 12U, .frame_id = 102U}},
      Sample<RuntimeExternalPod>{RuntimeExternalPod{.frame = 3U, .value = -3},
                                 SampleMetadata{.trace_id = 13U, .frame_id = 103U}},
      Sample<RuntimeExternalPod>{RuntimeExternalPod{.frame = 4U, .value = -4},
                                 SampleMetadata{.trace_id = 14U, .frame_id = 104U}},
  };
  const PodEosTopology topology = make_pod_eos_topology(PodEosPartitioning::kAutomatic);
  EXPECT_TRUE(run_pod_eos_once(topology, expected_samples));
  EXPECT_EQ(RuntimeExternalPodForwarder::compute_count.load(std::memory_order_relaxed),
            expected_samples.size());
}

// REQ: CHAN-132
// REQ: CHAN-136
// REQ: RUNTIME-017
TEST(RuntimeExternalPodTest, SamePlanRunsTwiceWithFreshExternalClientGenerations) {
  RuntimeExternalPodForwarder::reset();
  const PodEosTopology topology = make_pod_eos_topology(PodEosPartitioning::kAutomatic);
  const std::array first_samples{Sample<RuntimeExternalPod>{
      RuntimeExternalPod{.frame = 101U, .value = -101},
      SampleMetadata{.trace_id = 1011U, .frame_id = 1101U},
  }};
  const std::array second_samples{Sample<RuntimeExternalPod>{
      RuntimeExternalPod{.frame = 202U, .value = -202},
      SampleMetadata{.trace_id = 2022U, .frame_id = 2202U},
  }};
  ASSERT_TRUE(run_pod_eos_once(topology, first_samples));
  EXPECT_TRUE(run_pod_eos_once(topology, second_samples));
  EXPECT_EQ(RuntimeExternalPodForwarder::compute_count.load(std::memory_order_relaxed), 2U);
}

// One logical output fans out to two required external exports. Each arm must receive the final
// accepted value followed by its own EOS before the session can complete naturally.
// REQ: CHAN-132
// REQ: CHAN-136
// REQ: RUNTIME-017
TEST(RuntimeExternalPodTest, FanoutPublishesFinalDataAndEndOfStreamToEveryExternalArm) {
  RuntimeExternalPodForwarder::reset();
  const std::string import_name = unique_pod_endpoint_name();
  const std::array export_names{unique_pod_endpoint_name(), unique_pod_endpoint_name()};

  Graph graph{"runtime-external-pod-eos-fanout"};
  graph.set_default_clock(graph.add_clock<RealtimeClock>("runtime-clock"));
  TopicOptions import_options;
  import_options.endpoint_name = import_name;
  import_options.bounds.publisher = PublisherEndpointBounds{.max_loaned_samples = 1U};
  const auto imported =
      graph.import_topic<RuntimeExternalPod>("holomq://" + import_name, std::move(import_options));
  const auto forwarder = graph.op<RuntimeExternalPodForwarder>("forwarder");
  graph.add_flow(imported.output(), forwarder->input, ConnectionOptions{.queue_depth = 2U});
  for (const std::string& export_name : export_names) {
    TopicOptions export_options;
    export_options.endpoint_name = export_name;
    export_options.attach = TopicAttachPolicy{
        .peer = PeerRequirement::kRequired,
        .max_wait = 10s,
    };
    export_options.bounds.subscriber = SubscriberEndpointBounds{
        .queue_depth = 2U,
        .max_borrowed_samples = 1U,
    };
    const auto exported = graph.export_topic<RuntimeExternalPod>("holomq://" + export_name,
                                                                 std::move(export_options));
    graph.add_flow(forwarder->output, exported.input(), ConnectionOptions{.queue_depth = 2U});
  }

  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  auto external_runtime = ChannelRuntime::create();
  ASSERT_TRUE(external_runtime) << runtime_error_message(external_runtime.error());
  auto publisher =
      external_runtime->create_publisher<RuntimeExternalPod>(plan.endpoint_catalog(), import_name);
  ASSERT_TRUE(publisher) << runtime_error_message(publisher.error());

  RunSession session = run_async(plan);
  auto first_subscriber = external_runtime->create_subscriber<RuntimeExternalPod>(
      plan.endpoint_catalog(), export_names[0]);
  auto second_subscriber = external_runtime->create_subscriber<RuntimeExternalPod>(
      plan.endpoint_catalog(), export_names[1]);
  ASSERT_TRUE(first_subscriber) << runtime_error_message(first_subscriber.error());
  ASSERT_TRUE(second_subscriber) << runtime_error_message(second_subscriber.error());
  std::array subscribers{std::move(*first_subscriber), std::move(*second_subscriber)};

  ASSERT_TRUE(wait_for_subscriber(*publisher));
  for (auto& subscriber : subscribers) {
    ASSERT_TRUE(wait_for_publisher(subscriber));
  }
  const RuntimeExternalPod expected{.frame = 303U, .value = -303};
  ASSERT_TRUE(publisher->emit(expected));
  ASSERT_TRUE(publisher->close());

  for (auto& subscriber : subscribers) {
    auto received = wait_for_sample(subscriber);
    ASSERT_TRUE(received) << runtime_error_message(received.error());
    EXPECT_EQ(received->data, expected);
    auto terminal = wait_for_sample(subscriber);
    ASSERT_FALSE(terminal);
    EXPECT_EQ(terminal.error().code, ErrorCode::kClosed);
  }
  ASSERT_TRUE(wait_for_terminal(session));
  session.wait();
  EXPECT_EQ(session.termination(), RunTermination::kCompleted);
  EXPECT_TRUE(session.diagnostics().empty());
  EXPECT_EQ(RuntimeExternalPodForwarder::compute_count.load(std::memory_order_relaxed), 1U);
  for (auto& subscriber : subscribers) {
    EXPECT_TRUE(subscriber.close());
  }
}

// A drop-oldest family with one exported member and one Graph-internal member keeps the internal
// member direct, as the fanout seam's HoloMQ family does not. One publication reaches both arms,
// and each receives the forwarder's end of stream before the session completes naturally.
//
//   import --> forwarder.output --[drop-oldest]--> export     (transported leg)
//                              `--[drop-oldest]--> local.in   (direct recipient)
//
// REQ: CHAN-165
// REQ: CHAN-136
TEST(RuntimeExternalPodTest, MixedLocalityFanoutDeliversToTheExportAndTheDirectRecipient) {
  for (const bool force_holomq_fanout : {false, true}) {
    SCOPED_TRACE(force_holomq_fanout ? "holomq family" : "mixed-locality family");
    RuntimeExternalPodForwarder::reset();
    RuntimeExternalPodProbe::reset();
    const std::string import_name = unique_pod_endpoint_name();
    const std::string export_name = unique_pod_endpoint_name();

    Graph graph{"runtime-external-pod-mixed-locality-fanout"};
    graph.set_default_clock(graph.add_clock<RealtimeClock>("runtime-clock"));
    TopicOptions import_options;
    import_options.endpoint_name = import_name;
    import_options.bounds.publisher = PublisherEndpointBounds{.max_loaned_samples = 1U};
    const auto imported = graph.import_topic<RuntimeExternalPod>("holomq://" + import_name,
                                                                 std::move(import_options));
    const auto forwarder = graph.op<RuntimeExternalPodForwarder>("forwarder");
    const auto local = graph.op<RuntimeExternalPodSink>("local");
    graph.add_flow(imported.output(), forwarder->input, ConnectionOptions{.queue_depth = 2U});
    TopicOptions export_options;
    export_options.endpoint_name = export_name;
    export_options.attach = TopicAttachPolicy{
        .peer = PeerRequirement::kRequired,
        .max_wait = 10s,
    };
    export_options.bounds.subscriber = SubscriberEndpointBounds{
        .queue_depth = 2U,
        .max_borrowed_samples = 1U,
    };
    const auto exported = graph.export_topic<RuntimeExternalPod>("holomq://" + export_name,
                                                                 std::move(export_options));
    graph.add_flow(forwarder->output,
                   exported.input(),
                   ConnectionOptions{.queue_policy = QueuePolicy::kDropOldest, .queue_depth = 2U});
    graph.add_flow(forwarder->output,
                   local->input,
                   ConnectionOptions{.queue_policy = QueuePolicy::kDropOldest, .queue_depth = 1U});

    const ExecutionPlan plan =
        force_holomq_fanout
            ? detail::compile_with_process_local_holomq_fanout(graph, CompileOptions{})
            : compile(graph);
    ASSERT_TRUE(plan.ok()) << plan.json();
    const auto state = detail::ExecutionPlanAccess::retain(plan);
    ASSERT_NE(state, nullptr);
    ASSERT_EQ(state->process_local_channel_services.size(), 1U);
    EXPECT_EQ(state->process_local_channel_services.front().delivery_kind,
              force_holomq_fanout ? detail::ProcessLocalChannelDeliveryKind::kHoloMQ
                                  : detail::ProcessLocalChannelDeliveryKind::kDirectRetained);
    EXPECT_EQ(state->process_local_channel_services.front().max_subscribers, 1U);

    auto external_runtime = ChannelRuntime::create();
    ASSERT_TRUE(external_runtime) << runtime_error_message(external_runtime.error());
    auto publisher = external_runtime->create_publisher<RuntimeExternalPod>(plan.endpoint_catalog(),
                                                                            import_name);
    ASSERT_TRUE(publisher) << runtime_error_message(publisher.error());
    RunSession session = run_async(plan);
    auto subscriber = external_runtime->create_subscriber<RuntimeExternalPod>(
        plan.endpoint_catalog(), export_name);
    ASSERT_TRUE(subscriber) << runtime_error_message(subscriber.error());
    ASSERT_TRUE(wait_for_subscriber(*publisher));
    ASSERT_TRUE(wait_for_publisher(*subscriber));
    const RuntimeExternalPod expected{.frame = 505U, .value = -505};
    ASSERT_TRUE(publisher->emit(expected));
    ASSERT_TRUE(publisher->close());

    // Key assertion: the export and the direct recipient each receive the one publication, the
    // export then reads end of stream, and natural completion proves the local recipient did too.
    auto received = wait_for_sample(*subscriber);
    ASSERT_TRUE(received) << runtime_error_message(received.error());
    EXPECT_EQ(received->data, expected);
    auto terminal = wait_for_sample(*subscriber);
    ASSERT_FALSE(terminal);
    EXPECT_EQ(terminal.error().code, ErrorCode::kClosed);
    const auto local_sample = RuntimeExternalPodProbe::wait_for_sample();
    HOLOSCAN_ASSERT_HAS_VALUE(local_sample) << "Expected local sample";
    EXPECT_EQ(local_sample->data, expected);
    EXPECT_EQ(RuntimeExternalPodForwarder::compute_count.load(std::memory_order_relaxed), 1U);
    ASSERT_TRUE(wait_for_terminal(session));
    session.wait();
    EXPECT_EQ(session.termination(), RunTermination::kCompleted);
    EXPECT_TRUE(session.diagnostics().empty())
        << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
    EXPECT_TRUE(subscriber->close());
  }
}

// A required export that accepted data remains terminal-obligated even if its external consumer
// retires before producer EOS. Consumer abandonment must fail the run; it cannot be reclassified
// as successful EOS delivery or natural completion.
// REQ: CHAN-136
// REQ: CHAN-144
// REQ: RUNTIME-017
TEST(RuntimeExternalPodTest, RequiredExportSubscriberCloseFailsInsteadOfSatisfyingEndOfStream) {
  RuntimeExternalPodForwarder::reset();
  const PodEosTopology topology =
      make_pod_eos_topology(PodEosPartitioning::kAutomatic, PodEosTrigger::kOnEach, true);
  ASSERT_TRUE(topology.plan.ok()) << topology.plan.json();

  auto external_runtime = ChannelRuntime::create();
  ASSERT_TRUE(external_runtime) << runtime_error_message(external_runtime.error());
  auto publisher = external_runtime->create_publisher<RuntimeExternalPod>(
      topology.plan.endpoint_catalog(), topology.import_name);
  ASSERT_TRUE(publisher) << runtime_error_message(publisher.error());

  RunSession session = run_async(topology.plan);
  auto subscriber = external_runtime->create_subscriber<RuntimeExternalPod>(
      topology.plan.endpoint_catalog(), topology.export_name);
  ASSERT_TRUE(subscriber) << runtime_error_message(subscriber.error());
  ASSERT_TRUE(wait_for_subscriber(*publisher));
  ASSERT_TRUE(wait_for_publisher(*subscriber));
  const RuntimeExternalPod expected{.frame = 404U, .value = -404};
  ASSERT_TRUE(publisher->emit(expected));
  auto received = wait_for_sample(*subscriber);
  ASSERT_TRUE(received) << runtime_error_message(received.error());
  EXPECT_EQ(received->data, expected);

  ASSERT_TRUE(subscriber->close());
  ASSERT_TRUE(publisher->close());
  const bool failed_without_timeout = wait_for_terminal(session);
  if (!failed_without_timeout) {
    session.request_stop();
  }
  session.wait();

  ASSERT_TRUE(failed_without_timeout);
  EXPECT_EQ(session.termination(), RunTermination::kFailed);
  ASSERT_FALSE(session.diagnostics().empty());
  EXPECT_EQ(session.diagnostics().front().token, "RUNTIME_READINESS_BINDING_UNAVAILABLE");
  EXPECT_EQ(session.diagnostics().front().service, "terminal_input_end_of_stream");
  EXPECT_EQ(RuntimeExternalPodForwarder::compute_count.load(std::memory_order_relaxed), 1U);
}

// A replacement subscriber attached before terminal drain becomes the current required recipient.
// It receives EOS even though the prior subscriber received the generation's data and retired.
// REQ: CHAN-136
// REQ: CHAN-144
// REQ: RUNTIME-017
TEST(RuntimeExternalPodTest, RequiredExportReplacementPresentAtTerminalDrainReceivesEndOfStream) {
  RuntimeExternalPodForwarder::reset();
  const PodEosTopology topology =
      make_pod_eos_topology(PodEosPartitioning::kAutomatic, PodEosTrigger::kOnEach, true);
  ASSERT_TRUE(topology.plan.ok()) << topology.plan.json();

  auto external_runtime = ChannelRuntime::create();
  ASSERT_TRUE(external_runtime) << runtime_error_message(external_runtime.error());
  auto publisher = external_runtime->create_publisher<RuntimeExternalPod>(
      topology.plan.endpoint_catalog(), topology.import_name);
  ASSERT_TRUE(publisher) << runtime_error_message(publisher.error());

  RunSession session = run_async(topology.plan);
  auto initial_subscriber = external_runtime->create_subscriber<RuntimeExternalPod>(
      topology.plan.endpoint_catalog(), topology.export_name);
  ASSERT_TRUE(initial_subscriber) << runtime_error_message(initial_subscriber.error());
  ASSERT_TRUE(wait_for_subscriber(*publisher));
  ASSERT_TRUE(wait_for_publisher(*initial_subscriber));

  const RuntimeExternalPod expected{.frame = 505U, .value = -505};
  ASSERT_TRUE(publisher->emit(expected));
  auto received = wait_for_sample(*initial_subscriber);
  ASSERT_TRUE(received) << runtime_error_message(received.error());
  EXPECT_EQ(received->data, expected);
  ASSERT_TRUE(initial_subscriber->close());

  auto replacement = external_runtime->create_subscriber<RuntimeExternalPod>(
      topology.plan.endpoint_catalog(), topology.export_name);
  ASSERT_TRUE(replacement) << runtime_error_message(replacement.error());
  ASSERT_TRUE(wait_for_publisher(*replacement));
  ASSERT_TRUE(publisher->close());

  auto terminal = wait_for_sample(*replacement);
  ASSERT_FALSE(terminal);
  EXPECT_EQ(terminal.error().code, ErrorCode::kClosed);
  const bool completed_without_timeout = wait_for_terminal(session);
  if (!completed_without_timeout) {
    session.request_stop();
  }
  session.wait();

  ASSERT_TRUE(completed_without_timeout);
  EXPECT_EQ(session.termination(), RunTermination::kCompleted);
  EXPECT_TRUE(session.diagnostics().empty());
  EXPECT_EQ(RuntimeExternalPodForwarder::compute_count.load(std::memory_order_relaxed), 1U);
  EXPECT_TRUE(replacement->close());
}

// A two-input aligned join cannot treat the first producer EOS as whole-operator completion.
// The run stays live until the second input's data and EOS retire, then completes naturally.
// REQ: CHAN-132
// REQ: CHAN-136
// REQ: RUNTIME-017
TEST(RuntimeExternalPodTest, AlignedFanInWaitsForEveryExternalProducerEndOfStream) {
  RuntimeExternalPodFanInProbe::reset();
  const std::array import_names{unique_pod_endpoint_name(), unique_pod_endpoint_name()};

  Graph graph{"runtime-external-pod-eos-aligned-fan-in"};
  const ClockKey runtime_clock = graph.add_clock<RealtimeClock>("runtime-clock");
  graph.set_default_clock(runtime_clock);
  const auto import = [&](const std::string& endpoint_name) {
    TopicOptions options;
    options.endpoint_name = endpoint_name;
    options.bounds.publisher = PublisherEndpointBounds{.max_loaned_samples = 1U};
    options.capture_time = RequireCaptureTime{.clock = runtime_clock};
    return graph.import_topic<RuntimeExternalPod>("holomq://" + endpoint_name, std::move(options));
  };
  const auto first_import = import(import_names[0]);
  const auto second_import = import(import_names[1]);
  const auto join = graph.op<RuntimeExternalPodAlignedJoin>("join");
  graph.add_flow(first_import.output(), join->first, ConnectionOptions{.queue_depth = 2U});
  graph.add_flow(second_import.output(), join->second, ConnectionOptions{.queue_depth = 2U});

  const ExecutionPlan compiled = compile(graph);
  auto phase_counters = std::make_shared<detail::RuntimePhaseCounters>();
  const ExecutionPlan plan =
      detail::RuntimeTestPlanAccess::with_phase_counters(compiled, phase_counters);
  ASSERT_TRUE(plan.ok()) << plan.json();
  auto external_runtime = ChannelRuntime::create();
  ASSERT_TRUE(external_runtime) << runtime_error_message(external_runtime.error());
  auto first = external_runtime->create_publisher<RuntimeExternalPod>(plan.endpoint_catalog(),
                                                                      import_names[0]);
  auto second = external_runtime->create_publisher<RuntimeExternalPod>(plan.endpoint_catalog(),
                                                                       import_names[1]);
  ASSERT_TRUE(first) << runtime_error_message(first.error());
  ASSERT_TRUE(second) << runtime_error_message(second.error());

  RunSession session = run_async(plan);
  ASSERT_TRUE(wait_for_subscriber(*first));
  ASSERT_TRUE(wait_for_subscriber(*second));
  const std::array expected{RuntimeExternalPod{.frame = 1U, .value = -1},
                            RuntimeExternalPod{.frame = 2U, .value = -2}};
  ASSERT_TRUE(first->emit_captured(expected[0], 100));
  ASSERT_TRUE(second->emit_captured(expected[1], 100));
  const auto observed = RuntimeExternalPodFanInProbe::wait_for_values();
  EXPECT_EQ(observed, std::optional{expected});

  const auto phase_count = [&](const detail::RuntimePhaseCounter phase) {
    return phase_counters->snapshot()[detail::RuntimePhaseCounters::index(phase)];
  };
  const std::uint64_t readiness_baseline =
      phase_count(detail::RuntimePhaseCounter::kChannelReadinessPublished);
  ASSERT_TRUE(first->close());
  const auto first_eos_deadline = std::chrono::steady_clock::now() + 5s;
  while (phase_count(detail::RuntimePhaseCounter::kChannelReadinessPublished) <
             readiness_baseline + 1U &&
         session.is_running() && std::chrono::steady_clock::now() < first_eos_deadline) {
    std::this_thread::yield();
  }
  EXPECT_EQ(phase_count(detail::RuntimePhaseCounter::kChannelReadinessPublished),
            readiness_baseline + 1U);
  EXPECT_TRUE(session.is_running());
  EXPECT_EQ(session.termination(), RunTermination::kRunning);
  ASSERT_TRUE(second->close());
  ASSERT_TRUE(wait_for_terminal(session));
  session.wait();
  EXPECT_EQ(session.termination(), RunTermination::kCompleted);
  EXPECT_TRUE(session.diagnostics().empty());
}

// Graph topology:
//   source --[queue_depth=1]--> exported --[holomq://endpoint]--> external_subscriber
// The external subscriber attaches before the graph Runtime materializes the export topic
// publisher, verifying discovery ordering does not affect sample delivery.
// The external subscriber attaches before Runtime materializes the Graph-owned publisher. The
// emitted POD still crosses only the immutable catalog contract and retains caller metadata.
// REQ: CHAN-074
// REQ: RUNTIME-015
// REQ: RUNTIME-017
TEST(RuntimeExternalPodTest, CompiledGraphExportReachesIndependentSubscriberWithMetadata) {
  // Key setup: external ChannelRuntime creates the subscriber; RuntimeExternalPodSource gates
  // emission behind allow_emit() so the subscriber can discover the publisher before the
  // sample is sent.
  RuntimeExternalPodSource::reset();
  const std::string endpoint_name = unique_pod_endpoint_name();

  Graph graph{"runtime-external-pod-export"};
  const ClockKey runtime_clock = graph.add_clock<RealtimeClock>("runtime-clock");
  graph.set_default_clock(runtime_clock);
  TopicOptions options;
  options.endpoint_name = endpoint_name;
  options.bounds.subscriber = SubscriberEndpointBounds{
      .queue_depth = 1U,
      .max_borrowed_samples = 1U,
  };
  const auto exported =
      graph.export_topic<RuntimeExternalPod>("holomq://" + endpoint_name, std::move(options));
  const auto source = graph.op<RuntimeExternalPodSource>("source");
  graph.add_flow(source->output, exported.input(), ConnectionOptions{.queue_depth = 1U});

  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  auto external_runtime = ChannelRuntime::create();
  ASSERT_TRUE(external_runtime) << runtime_error_message(external_runtime.error());
  auto subscriber = external_runtime->create_subscriber<RuntimeExternalPod>(plan.endpoint_catalog(),
                                                                            endpoint_name);
  ASSERT_TRUE(subscriber) << runtime_error_message(subscriber.error());

  RunSession session = run_async(plan);
  if (!RuntimeExternalPodSource::wait_for_compute_started()) {
    RuntimeExternalPodSource::allow_emit();
    session.request_stop();
    session.wait();
    FAIL() << "custom-POD Graph source did not enter compute; termination="
           << static_cast<int>(session.termination()) << "; diagnostics="
           << (session.diagnostics().empty() ? "<none>" : session.diagnostics().front().token)
           << "; plan=" << plan.json();
  }
  auto subscriber_ready = wait_for_publisher(*subscriber);
  RuntimeExternalPodSource::allow_emit();
  ASSERT_TRUE(subscriber_ready) << runtime_error_message(subscriber_ready.error());
  auto received = wait_for_sample(*subscriber);

  session.request_stop();
  session.wait();
  // Key assertion: received sample data, trace_id, frame_id, flags, byte_size, and
  // payload_kind all match the emission from RuntimeExternalPodSource.
  ASSERT_TRUE(received) << runtime_error_message(received.error()) << "; source computes="
                        << RuntimeExternalPodSource::compute_count.load(std::memory_order_relaxed)
                        << "; emit error="
                        << static_cast<int>(RuntimeExternalPodSource::emit_error.load(
                               std::memory_order_relaxed));
  EXPECT_EQ(received->data, (RuntimeExternalPod{.frame = 41U, .value = -73}));
  EXPECT_GT(received->metadata.sequence, 0U);
  EXPECT_EQ(received->metadata.trace_id, 0x1357U);
  EXPECT_EQ(received->metadata.frame_id, 0x2468U);
  EXPECT_EQ(received->metadata.flags, SampleFlags::kDegraded);
  EXPECT_EQ(received->metadata.byte_size, sizeof(RuntimeExternalPod));
  EXPECT_EQ(received->metadata.payload_kind, PayloadKind::kInlineTyped);
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
}

}  // namespace
}  // namespace holoscan
