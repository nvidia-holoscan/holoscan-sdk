/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
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

#include "channel/process_identity.hpp"

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
  static constexpr std::string_view name = "nvidia.holoscan.test.RuntimeExternalPod";
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
      observed = sample;
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
