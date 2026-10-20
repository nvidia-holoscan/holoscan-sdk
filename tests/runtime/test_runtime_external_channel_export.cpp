/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <holomq/context_builder.hpp>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <utility>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/memory/tensor_storage.hpp>
#include <holoscan/schema/descriptor_flatbuffer_payload.hpp>
#include <holoscan/schema/sdk_schemas.hpp>
#include <holoscan/sensor_io/image_metadata.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>
#include <holoscan/time/manual_clock.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "../common/cuda_test_utils.hpp"
#include "image_generated.h"

#include "channel/channel_common.hpp"
#include "channel/channel_payload_frame.hpp"
#include "channel/channel_tensor_test_access.hpp"
#include "channel/endpoint_catalog_state.hpp"
#include "compiler/compile_internal.hpp"
#include "core/graph_compiler_access.hpp"
#include "runtime/channel_runtime_binding_access.hpp"
#include "runtime/endpoint_clock_health_runtime.hpp"
#include "runtime/graph_exec.hpp"
#include "runtime/runtime_channel_binding_test_access.hpp"
#include "runtime/runtime_phase_counters.hpp"
#include "runtime/runtime_support_thread_control.hpp"
#include "runtime/runtime_test_access.hpp"
#include "runtime_external_channel_test_support.hpp"
#include "runtime_test_graph.hpp"

namespace holoscan {
namespace {

using namespace std::chrono_literals;

using namespace test::external_channel;

[[nodiscard]] TopicOptions external_image_export_options(std::string_view endpoint_name) {
  TopicOptions options;
  options.endpoint_name = endpoint_name;
  options.bounds.subscriber =
      SubscriberEndpointBounds{.queue_depth = 1U, .max_borrowed_samples = 1U};
  return options;
}

// Forwards imported raw Tensor samples to an external export while preserving the
// Tensor contract and ownership across the graph boundary.
// Representative test: RequiredRawTensorExportSubscriberCloseFailsWithoutTerminalRetry.
class ExternalTensorForwarder final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(input, "input")
        .queue_depth(1U)
        .expects_tensor(TensorInputSpec{
            .representation = {.memory_kind = MemoryKind::kHost,
                               .dtype = kExternalTensorDtype,
                               .rank = 1U,
                               .strides = TensorStridePolicy::kContiguousOnly},
            .bounds = tensor_bounds(kExternalTensorShape),
        });
    spec.output(output, "output")
        .max_emits_per_compute(1U)
        .produces_tensor(TensorOutputSpec{
            .representation = {.memory_kind = MemoryKind::kHost,
                               .dtype = kExternalTensorDtype,
                               .rank = 1U,
                               .strides = TensorStridePolicy::kContiguousOnly},
            .bounds = tensor_bounds(kExternalTensorShape),
            .transfer = TensorTransferPolicy{.foreign_emit_policy =
                                                 InterfaceTensorEmitPolicy::kCopyIfNeeded},
        });
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    auto sample = input.receive();
    if (!sample) {
      return make_unexpected(std::move(sample).error());
    }
    return output.emit(std::move(*sample));
  }

  Input<Tensor> input;
  Output<Tensor> output;
};

class ExternalImageExportSource final : public Operator<> {
 public:
  static void reset(const std::optional<std::uint64_t> completion_limit = std::nullopt) noexcept {
    const std::lock_guard lock{coordination_mutex};
    compute_started = false;
    emit_allowed = false;
    emit_completed = false;
    completes_naturally.store(completion_limit.has_value(), std::memory_order_relaxed);
    compute_limit.store(completion_limit.value_or(0U), std::memory_order_relaxed);
    emit_error.store(ErrorCode::kSuccess, std::memory_order_relaxed);
    capture_time.reset();
  }

  [[nodiscard]] static bool wait_for_compute_started(std::chrono::milliseconds timeout = 5s) {
    std::unique_lock lock{coordination_mutex};
    return coordination_ready.wait_for(lock, timeout, [] { return compute_started; });
  }

  [[nodiscard]] static bool wait_for_emit_completed(std::chrono::milliseconds timeout = 5s) {
    std::unique_lock lock{coordination_mutex};
    return coordination_ready.wait_for(lock, timeout, [] { return emit_completed; });
  }

  static void allow_emit() noexcept {
    {
      const std::lock_guard lock{coordination_mutex};
      emit_allowed = true;
    }
    coordination_ready.notify_all();
  }

  static void set_capture_time(TimePoint capture) noexcept {
    const std::lock_guard lock{coordination_mutex};
    capture_time = capture;
  }

  void setup(OperatorSpec& spec) override {
    spec.output(output, "output")
        .max_emits_per_compute(1U)
        .tensor_field(kExternalImageData)
        .produces_tensor(TensorOutputSpec{
            .representation = {.memory_kind = MemoryKind::kHost,
                               .dtype = kExternalTensorDtype,
                               .rank = 2U,
                               .strides = TensorStridePolicy::kContiguousOnly},
            .bounds = tensor_bounds(kExternalImageShape),
            .transfer = TensorTransferPolicy{.foreign_emit_policy =
                                                 InterfaceTensorEmitPolicy::kCopyIfNeeded},
        });
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 50ms});
    if (completes_naturally.load(std::memory_order_relaxed)) {
      result.complete_after(ComputeCalls{.limit = compute_limit.load(std::memory_order_relaxed)});
    }
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    if (emitted_) {
      return {};
    }
    emitted_ = true;
    std::optional<TimePoint> selected_capture;
    {
      std::unique_lock lock{coordination_mutex};
      compute_started = true;
      coordination_ready.notify_all();
      if (!coordination_ready.wait_for(lock, 5s, [] { return emit_allowed; })) {
        return finish(make_unexpected(Error{ErrorCode::kTimeout}));
      }
      selected_capture = capture_time;
    }
    auto tensor = make_external_image_tensor(0x5aU);
    if (!tensor) {
      return finish(make_unexpected(std::move(tensor).error()));
    }
    schema::ImageT image{};
    image.height = 1U;
    image.width = 8U;
    image.encoding = schema::ImageEncoding_MONO8;
    image.data = std::make_shared<Tensor>(std::move(*tensor));
    SampleMetadata metadata{.trace_id = 31U, .frame_id = 41U, .flags = SampleFlags::kDegraded};
    if (selected_capture) {
      metadata.capture_timestamp_ns = selected_capture->timestamp_ns;
      metadata.source_clock_id = selected_capture->clock_id;
    }
    return finish(output.emit(Sample<schema::ImageT>{std::move(image), metadata}));
  }

  Output<schema::ImageT> output;
  static inline std::atomic<ErrorCode> emit_error{ErrorCode::kSuccess};

 private:
  static expected<void, Error> finish(expected<void, Error> result) noexcept {
    if (!result) {
      HOLOSCAN_LOG_ERROR("External Image export failed with {}",
                         runtime_error_message(result.error()));
    }
    emit_error.store(result ? ErrorCode::kSuccess : result.error().code, std::memory_order_relaxed);
    {
      const std::lock_guard lock{coordination_mutex};
      emit_completed = true;
    }
    coordination_ready.notify_all();
    return result;
  }

  bool emitted_{};
  static inline std::mutex coordination_mutex;
  static inline std::condition_variable coordination_ready;
  static inline bool compute_started{};
  static inline bool emit_allowed{};
  static inline bool emit_completed{};
  static inline std::atomic_bool completes_naturally{};
  static inline std::atomic<std::uint64_t> compute_limit{};
  static inline std::optional<TimePoint> capture_time;
};

class LateAttachExternalImageExportSource final : public Operator<> {
 public:
  static void reset(const bool complete_after_second_emit = false) noexcept {
    first_emit_completed.store(false, std::memory_order_release);
    second_emit_allowed.store(false, std::memory_order_release);
    second_emit_completed.store(false, std::memory_order_release);
    completes_naturally.store(complete_after_second_emit, std::memory_order_release);
    emit_error.store(ErrorCode::kSuccess, std::memory_order_relaxed);
  }

  [[nodiscard]] static bool wait_for_first_emit(std::chrono::milliseconds timeout = 5s) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (!first_emit_completed.load(std::memory_order_acquire) &&
           std::chrono::steady_clock::now() < deadline) {
      std::this_thread::yield();
    }
    return first_emit_completed.load(std::memory_order_acquire);
  }

  [[nodiscard]] static bool wait_for_second_emit(std::chrono::milliseconds timeout = 5s) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (!second_emit_completed.load(std::memory_order_acquire) &&
           std::chrono::steady_clock::now() < deadline) {
      std::this_thread::yield();
    }
    return second_emit_completed.load(std::memory_order_acquire);
  }

  static void allow_second_emit() noexcept {
    second_emit_allowed.store(true, std::memory_order_release);
  }

  void setup(OperatorSpec& spec) override {
    spec.output(output, "output")
        .max_emits_per_compute(1U)
        .tensor_field(kExternalImageData)
        .produces_tensor(TensorOutputSpec{
            .representation = {.memory_kind = MemoryKind::kHost,
                               .dtype = kExternalTensorDtype,
                               .rank = 2U,
                               .strides = TensorStridePolicy::kContiguousOnly},
            .bounds = tensor_bounds(kExternalImageShape),
            .transfer = TensorTransferPolicy{.foreign_emit_policy =
                                                 InterfaceTensorEmitPolicy::kCopyIfNeeded},
        });
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 10ms});
    if (completes_naturally.load(std::memory_order_acquire)) {
      result.complete_after(ComputeCalls{.limit = 2U});
    }
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    const bool first = !first_emit_completed.load(std::memory_order_acquire);
    if (!first && completes_naturally.load(std::memory_order_acquire) &&
        !second_emit_completed.load(std::memory_order_acquire)) {
      const auto deadline = std::chrono::steady_clock::now() + 5s;
      while (!second_emit_allowed.load(std::memory_order_acquire) &&
             std::chrono::steady_clock::now() < deadline) {
        std::this_thread::yield();
      }
      if (!second_emit_allowed.load(std::memory_order_acquire)) {
        return finish(false, make_unexpected(Error{ErrorCode::kTimeout}));
      }
    }
    if (!first && (!second_emit_allowed.load(std::memory_order_acquire) ||
                   second_emit_completed.load(std::memory_order_acquire))) {
      return {};
    }
    auto tensor = make_external_image_tensor(first ? 0x21U : 0x42U);
    if (!tensor) {
      return finish(first, make_unexpected(std::move(tensor).error()));
    }
    schema::ImageT image{};
    image.height = 1U;
    image.width = 8U;
    image.encoding = schema::ImageEncoding_MONO8;
    image.data = std::make_shared<Tensor>(std::move(*tensor));
    return finish(first, output.emit(std::move(image)));
  }

  Output<schema::ImageT> output;
  static inline std::atomic<ErrorCode> emit_error{ErrorCode::kSuccess};

 private:
  static expected<void, Error> finish(const bool first, expected<void, Error> result) noexcept {
    if (!result) {
      emit_error.store(result.error().code, std::memory_order_relaxed);
    }
    (first ? first_emit_completed : second_emit_completed).store(true, std::memory_order_release);
    return result;
  }

  static inline std::atomic<bool> first_emit_completed{};
  static inline std::atomic<bool> second_emit_allowed{};
  static inline std::atomic<bool> second_emit_completed{};
  static inline std::atomic_bool completes_naturally{};
};

// Source that allocates and emits a single host Tensor, gating the write on allow_emit().
// Emits at most once per run to keep assertion state deterministic.
// Representative test: OptionalTensorGraphExportAcceptsZeroConnectedRecipients.
class ExternalTensorExportSource final : public Operator<> {
 public:
  static void reset() noexcept {
    const std::lock_guard lock{coordination_mutex};
    compute_started = false;
    emit_allowed = false;
    emit_completed = false;
    emit_error.store(ErrorCode::kSuccess, std::memory_order_relaxed);
  }

  [[nodiscard]] static bool wait_for_compute_started(std::chrono::milliseconds timeout = 5s) {
    std::unique_lock lock{coordination_mutex};
    return coordination_ready.wait_for(lock, timeout, [] { return compute_started; });
  }

  [[nodiscard]] static bool wait_for_emit_completed(std::chrono::milliseconds timeout = 5s) {
    std::unique_lock lock{coordination_mutex};
    return coordination_ready.wait_for(lock, timeout, [] { return emit_completed; });
  }

  [[nodiscard]] static ErrorCode last_emit_error() noexcept {
    return emit_error.load(std::memory_order_relaxed);
  }

  static void allow_emit() noexcept {
    {
      const std::lock_guard lock{coordination_mutex};
      emit_allowed = true;
    }
    coordination_ready.notify_all();
  }

  void setup(OperatorSpec& spec) override {
    spec.output(output, "output")
        .max_emits_per_compute(1U)
        .produces_tensor(TensorOutputSpec{
            .representation = {.memory_kind = MemoryKind::kHost,
                               .dtype = kExternalTensorDtype,
                               .rank = 1U},
            .bounds = tensor_bounds(8U),
            .storage = TensorOutputStorage::kRuntimePool,
            .transfer = TensorTransferPolicy{.foreign_emit_policy =
                                                 InterfaceTensorEmitPolicy::kCopyIfNeeded},
        });
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 50ms});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    if (emitted_) {
      return {};
    }
    emitted_ = true;
    {
      std::unique_lock lock{coordination_mutex};
      compute_started = true;
      coordination_ready.notify_all();
      if (!coordination_ready.wait_for(lock, 5s, [] { return emit_allowed; })) {
        return finish(make_unexpected(Error{ErrorCode::kTimeout}));
      }
    }
    auto loan = output.allocate_tensor(TensorLoanRequest{
        .shape = kExternalTensorShape,
        .dtype = kExternalTensorDtype,
    });
    if (!loan) {
      return finish(make_unexpected(std::move(loan).error()));
    }
    auto writer = loan->write_host();
    if (!writer) {
      return finish(make_unexpected(std::move(writer).error()));
    }
    std::fill_n(static_cast<std::byte*>(writer->data()), writer->byte_size(), std::byte{0x2a});
    auto committed = std::move(*writer).commit();
    if (!committed) {
      return finish(make_unexpected(std::move(committed).error()));
    }
    return finish(output.emit(std::move(*loan)));
  }

  Output<Tensor> output;

 private:
  static expected<void, Error> finish(expected<void, Error> result) noexcept {
    emit_error.store(result ? ErrorCode::kSuccess : result.error().code, std::memory_order_relaxed);
    {
      const std::lock_guard lock{coordination_mutex};
      emit_completed = true;
    }
    coordination_ready.notify_all();
    return result;
  }

  bool emitted_{};
  static inline std::atomic<ErrorCode> emit_error{ErrorCode::kSuccess};
  static inline std::mutex coordination_mutex;
  static inline std::condition_variable coordination_ready;
  static inline bool compute_started{};
  static inline bool emit_allowed{};
  static inline bool emit_completed{};
};

template <ChannelDescriptorFlatBufferPayload Payload>
[[nodiscard]] expected<ChannelResolvedTensorSample<Payload>, Error> wait_for_schema_sample(
    ChannelTensorSchemaSubscriber<Payload>& subscriber, std::chrono::milliseconds timeout = 5s) {
  static_assert(Payload::max_tensors == 1U);
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    auto ready = subscriber.has_samples();
    if (!ready) {
      return make_unexpected(std::move(ready).error());
    }
    if (*ready) {
      auto sample = subscriber.receive();
      if (sample || sample.error().code != ErrorCode::kNotReady) {
        return sample;
      }
    }
    std::this_thread::sleep_for(1ms);
  }
  return make_unexpected(Error{ErrorCode::kTimeout});
}

template <ChannelDescriptorFlatBufferPayload Payload>
[[nodiscard]] expected<void, Error> wait_for_schema_publisher(
    ChannelTensorSchemaSubscriber<Payload>& subscriber, std::chrono::milliseconds timeout = 5s) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  std::uint32_t consecutive_observations = 0U;
  while (std::chrono::steady_clock::now() < deadline) {
    auto refreshed = subscriber.has_samples();
    if (!refreshed) {
      return make_unexpected(std::move(refreshed).error());
    }
    if (subscriber.has_publishers()) {
      if (++consecutive_observations == 2U) {
        return {};
      }
    } else {
      consecutive_observations = 0U;
    }
    std::this_thread::sleep_for(1ms);
  }
  return make_unexpected(Error{ErrorCode::kTimeout});
}

void expect_completion_progress(const RunSession& session, const std::uint64_t limit) {
  ASSERT_EQ(session.operator_completion_progress().size(), 1U);
  const auto& progress = session.operator_completion_progress().front();
  EXPECT_EQ(progress.configured_limit, limit);
  EXPECT_EQ(progress.successful_compute_calls, limit);
  EXPECT_TRUE(progress.limit_reached);
}

[[nodiscard]] expected<Sample<Tensor>, Error> wait_for_tensor_sample(
    ChannelSubscriber<Tensor>& subscriber, std::chrono::milliseconds timeout = 5s) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    auto ready = subscriber.has_samples();
    if (!ready) {
      return make_unexpected(std::move(ready).error());
    }
    if (*ready) {
      auto sample = subscriber.receive_sample();
      if (sample || sample.error().code != ErrorCode::kNotReady) {
        return sample;
      }
    }
    std::this_thread::sleep_for(1ms);
  }
  return make_unexpected(Error{ErrorCode::kTimeout});
}

// REQ: CHAN-068
// REQ: CHAN-101
// REQ: RUNTIME-015
TEST(RuntimeExternalChannelTest, OptionalGraphExportAcceptsZeroConnectedRecipients) {
  ExternalScalarExportSource::reset();
  const std::string export_name = unique_endpoint_name();

  //   source --[drop-oldest, depth=4]--> holomq://endpoint  (optional export, no subscribers)
  Graph graph{"runtime-optional-external-export"};
  const ClockKey runtime_clock = graph.add_clock<RealtimeClock>("runtime-clock");
  graph.set_default_clock(runtime_clock);

  TopicOptions export_options;
  export_options.endpoint_name = export_name;
  const auto exported =
      graph.export_topic<std::uint32_t>("holomq://" + export_name, std::move(export_options));
  const auto source = graph.op<ExternalScalarExportSource>("source");
  graph.add_flow(source->output,
                 exported.input(),
                 ConnectionOptions{.queue_policy = QueuePolicy::kDropOldest, .queue_depth = 4U});

  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();

  // Key setup: this output has only the external export arm, so its first emit exercises the
  // export-phase OutputState and its retained owning Runtime session.
  RunSession session = run_async(plan);
  const bool compute_started = ExternalScalarExportSource::wait_for_compute_started();
  ExternalScalarExportSource::allow_emit();
  const bool emit_completed = ExternalScalarExportSource::wait_for_emit_completed();
  session.request_stop();
  session.wait();

  ASSERT_TRUE(compute_started);
  ASSERT_TRUE(emit_completed);
  // Key assertion: the external-only first emit safely reaches the optional-drop result without
  // dereferencing an absent Runtime owner.
  EXPECT_EQ(ExternalScalarExportSource::emit_error.load(std::memory_order_relaxed),
            ErrorCode::kSuccess);
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
}

// Retiring an attached external subscriber is local client teardown, not graph terminal evidence.
// The unbounded source remains running, its optional export follows the ordinary no-observer path,
// and only the explicit session stop produces a terminal outcome.
// REQ: CHAN-143
// REQ: RUNTIME-017
TEST(RuntimeExternalChannelTest, LocalSubscriberCloseDoesNotCompleteGraphExport) {
  ExternalScalarExportSource::reset();
  const std::string export_name = unique_endpoint_name();

  Graph graph{"runtime-local-subscriber-close-isolation"};
  const ClockKey runtime_clock = graph.add_clock<RealtimeClock>("runtime-clock");
  graph.set_default_clock(runtime_clock);
  TopicOptions export_options;
  export_options.endpoint_name = export_name;
  const auto exported =
      graph.export_topic<std::uint32_t>("holomq://" + export_name, std::move(export_options));
  const auto source = graph.op<ExternalScalarExportSource>("source");
  graph.add_flow(source->output,
                 exported.input(),
                 ConnectionOptions{.queue_policy = QueuePolicy::kDropOldest, .queue_depth = 4U});

  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  auto external_runtime = ChannelRuntime::create();
  ASSERT_TRUE(external_runtime) << runtime_error_message(external_runtime.error());
  auto subscriber =
      external_runtime->create_subscriber<std::uint32_t>(plan.endpoint_catalog(), export_name);
  ASSERT_TRUE(subscriber) << runtime_error_message(subscriber.error());

  RunSession session = run_async(plan);
  ASSERT_TRUE(ExternalScalarExportSource::wait_for_compute_started());
  auto publisher_ready = wait_for_publisher(*subscriber);
  ASSERT_TRUE(publisher_ready) << runtime_error_message(publisher_ready.error());
  ASSERT_TRUE(subscriber->close());
  EXPECT_TRUE(session.is_running());
  EXPECT_EQ(session.termination(), RunTermination::kRunning);
  EXPECT_TRUE(session.diagnostics().empty());

  ExternalScalarExportSource::allow_emit();
  ASSERT_TRUE(ExternalScalarExportSource::wait_for_emit_completed());
  EXPECT_EQ(ExternalScalarExportSource::emit_error.load(std::memory_order_relaxed),
            ErrorCode::kSuccess);
  EXPECT_TRUE(session.is_running());

  session.request_stop();
  session.wait();
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
}

// A raw-Tensor export that has already delivered data remains obligated to its required external
// consumer. Retiring that consumer before imported EOS must fail the run without delivering EOS.
// REQ: CHAN-136
// REQ: CHAN-144
// REQ: RUNTIME-017
TEST(RuntimeExternalChannelTest, RequiredRawTensorExportSubscriberCloseFailsWithoutTerminalRetry) {
  const std::string import_name = unique_endpoint_name();
  const std::string export_name = unique_endpoint_name();

  Graph graph{"runtime-required-raw-tensor-export-abandonment"};
  graph.set_default_clock(graph.add_clock<RealtimeClock>("runtime-clock"));
  TopicOptions import_options = optional_tensor_export_options(import_name);
  import_options.bounds.subscriber.reset();
  import_options.bounds.publisher = PublisherEndpointBounds{.max_loaned_samples = 1U};
  const auto imported =
      graph.import_topic<Tensor>("holomq://" + import_name, std::move(import_options));
  TopicOptions export_options = optional_tensor_export_options(export_name);
  export_options.attach = TopicAttachPolicy{
      .peer = PeerRequirement::kRequired,
      .max_wait = 10s,
  };
  const auto exported =
      graph.export_topic<Tensor>("holomq://" + export_name, std::move(export_options));
  const auto forwarder = graph.op<ExternalTensorForwarder>("forwarder");
  graph.add_flow(imported.output(), forwarder->input, ConnectionOptions{.queue_depth = 1U});
  graph.add_flow(forwarder->output, exported.input(), ConnectionOptions{.queue_depth = 1U});

  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  auto external_runtime = ChannelRuntime::create();
  ASSERT_TRUE(external_runtime) << runtime_error_message(external_runtime.error());
  auto publisher = external_runtime->create_publisher<Tensor>(plan.endpoint_catalog(), import_name);
  ASSERT_TRUE(publisher) << runtime_error_message(publisher.error());

  RunSession session = run_async(plan);
  auto subscriber =
      external_runtime->create_subscriber<Tensor>(plan.endpoint_catalog(), export_name);
  ASSERT_TRUE(subscriber) << runtime_error_message(subscriber.error());

  ASSERT_TRUE(wait_for_subscriber(*publisher));
  ASSERT_TRUE(wait_for_publisher(*subscriber));
  ASSERT_TRUE(publish_external_host_tensor(*publisher, std::nullopt));
  {
    auto received = wait_for_tensor_sample(*subscriber);
    ASSERT_TRUE(received) << runtime_error_message(received.error());
    ASSERT_NE(received->data.data_as<std::uint8_t>(), nullptr);
    EXPECT_EQ(received->data.data_as<std::uint8_t>()[0], 0x2aU);
  }
  EXPECT_EQ(
      detail::RuntimeChannelBindingAccess::accepted_tensor_end_of_stream_sequence(*subscriber), 0U);

  ASSERT_TRUE(subscriber->close());
  ASSERT_TRUE(publisher->close());
  const bool failed_without_timeout = wait_for_session_terminal(session);
  if (!failed_without_timeout) {
    session.request_stop();
  }
  session.wait();

  ASSERT_TRUE(failed_without_timeout);
  EXPECT_EQ(session.termination(), RunTermination::kFailed);
  ASSERT_EQ(session.diagnostics().size(), 1U);
  EXPECT_EQ(session.diagnostics().front().token, "RUNTIME_READINESS_BINDING_UNAVAILABLE");
  EXPECT_EQ(session.diagnostics().front().service, "terminal_input_end_of_stream");
  EXPECT_EQ(
      detail::RuntimeChannelBindingAccess::accepted_tensor_end_of_stream_sequence(*subscriber), 0U);
}

// A replacement raw-Tensor subscriber attached before terminal drain becomes the current required
// recipient and receives EOS after the initial subscriber's accepted data.
// REQ: CHAN-136
// REQ: CHAN-144
// REQ: RUNTIME-017
TEST(RuntimeExternalChannelTest,
     RequiredRawTensorExportReplacementPresentAtTerminalDrainReceivesEndOfStream) {
  const std::string import_name = unique_endpoint_name();
  const std::string export_name = unique_endpoint_name();

  Graph graph{"runtime-required-raw-tensor-export-replacement-terminal"};
  graph.set_default_clock(graph.add_clock<RealtimeClock>("runtime-clock"));
  TopicOptions import_options = optional_tensor_export_options(import_name);
  import_options.bounds.subscriber.reset();
  import_options.bounds.publisher = PublisherEndpointBounds{.max_loaned_samples = 1U};
  const auto imported =
      graph.import_topic<Tensor>("holomq://" + import_name, std::move(import_options));
  TopicOptions export_options = optional_tensor_export_options(export_name);
  export_options.attach = TopicAttachPolicy{
      .peer = PeerRequirement::kRequired,
      .max_wait = 10s,
  };
  const auto exported =
      graph.export_topic<Tensor>("holomq://" + export_name, std::move(export_options));
  const auto forwarder = graph.op<ExternalTensorForwarder>("forwarder");
  graph.add_flow(imported.output(), forwarder->input, ConnectionOptions{.queue_depth = 1U});
  graph.add_flow(forwarder->output, exported.input(), ConnectionOptions{.queue_depth = 1U});

  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  auto external_runtime = ChannelRuntime::create();
  ASSERT_TRUE(external_runtime) << runtime_error_message(external_runtime.error());
  auto publisher = external_runtime->create_publisher<Tensor>(plan.endpoint_catalog(), import_name);
  ASSERT_TRUE(publisher) << runtime_error_message(publisher.error());

  RunSession session = run_async(plan);
  auto initial_subscriber =
      external_runtime->create_subscriber<Tensor>(plan.endpoint_catalog(), export_name);
  ASSERT_TRUE(initial_subscriber) << runtime_error_message(initial_subscriber.error());
  ASSERT_TRUE(wait_for_subscriber(*publisher));
  ASSERT_TRUE(wait_for_publisher(*initial_subscriber));
  ASSERT_TRUE(publish_external_host_tensor(*publisher, std::nullopt));
  {
    auto received = wait_for_tensor_sample(*initial_subscriber);
    ASSERT_TRUE(received) << runtime_error_message(received.error());
    ASSERT_NE(received->data.data_as<std::uint8_t>(), nullptr);
    EXPECT_EQ(received->data.data_as<std::uint8_t>()[0], 0x2aU);
  }
  ASSERT_TRUE(initial_subscriber->close());

  auto replacement =
      external_runtime->create_subscriber<Tensor>(plan.endpoint_catalog(), export_name);
  ASSERT_TRUE(replacement) << runtime_error_message(replacement.error());
  ASSERT_TRUE(wait_for_publisher(*replacement));
  ASSERT_TRUE(publisher->close());

  auto terminal = wait_for_tensor_sample(*replacement);
  ASSERT_FALSE(terminal);
  EXPECT_EQ(terminal.error().code, ErrorCode::kClosed);
  const bool completed_without_timeout = wait_for_session_terminal(session);
  if (!completed_without_timeout) {
    session.request_stop();
  }
  session.wait();

  ASSERT_TRUE(completed_without_timeout);
  EXPECT_EQ(session.termination(), RunTermination::kCompleted);
  EXPECT_TRUE(session.diagnostics().empty());
  EXPECT_TRUE(replacement->close());
}

// Descriptor-Tensor output closure has a distinct result shape from raw Tensor closure. Exercise
// that branch after one accepted value so subscriber abandonment is terminal and non-retryable.
// REQ: CHAN-136
// REQ: CHAN-144
// REQ: RUNTIME-017
TEST(RuntimeExternalChannelTest,
     RequiredDescriptorTensorExportSubscriberCloseFailsWithoutTerminalRetry) {
  const std::string import_name = unique_endpoint_name();
  const std::string export_name = unique_endpoint_name();

  Graph graph{"runtime-required-descriptor-tensor-export-abandonment"};
  graph.set_default_clock(graph.add_clock<RealtimeClock>("runtime-clock"));
  const auto imported =
      graph.import_topic<schema::ImageT>("holomq://" + import_name,
                                         external_image_import_options(import_name),
                                         external_image_contract());
  TopicOptions export_options = external_image_export_options(export_name);
  export_options.attach = TopicAttachPolicy{
      .peer = PeerRequirement::kRequired,
      .max_wait = 10s,
  };
  const auto exported = graph.export_topic<schema::ImageT>(
      "holomq://" + export_name, std::move(export_options), external_image_contract());
  const auto forwarder = graph.op<ExternalImageForwarder>("forwarder");
  graph.add_flow(imported.output(), forwarder->input, ConnectionOptions{.queue_depth = 1U});
  graph.add_flow(forwarder->output, exported.input(), ConnectionOptions{.queue_depth = 1U});

  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  auto external_runtime = ChannelRuntime::create();
  ASSERT_TRUE(external_runtime) << runtime_error_message(external_runtime.error());
  auto publisher = external_runtime->create_tensor_schema_publisher<ExternalImagePayload>(
      plan.endpoint_catalog(), import_name);
  ASSERT_TRUE(publisher) << runtime_error_message(publisher.error());

  RunSession session = run_async(plan);
  auto subscriber = external_runtime->create_tensor_schema_subscriber<ExternalImagePayload>(
      plan.endpoint_catalog(), export_name);
  ASSERT_TRUE(subscriber) << runtime_error_message(subscriber.error());

  ASSERT_TRUE(wait_for_schema_subscriber(*publisher));
  ASSERT_TRUE(wait_for_schema_publisher(*subscriber));
  auto tensor = make_external_image_tensor(0x3cU);
  ASSERT_TRUE(tensor) << runtime_error_message(tensor.error());
  auto metadata = schema::build_image_metadata(
      *tensor,
      schema::ImageMetadataOptions{
          .height = 1U, .width = 8U, .encoding = schema::ImageEncoding_MONO8});
  ASSERT_TRUE(metadata) << runtime_error_message(metadata.error());
  ASSERT_TRUE(publisher->emit(*metadata, std::move(*tensor)));
  {
    auto received = wait_for_schema_sample(*subscriber);
    ASSERT_TRUE(received) << runtime_error_message(received.error());
    ASSERT_NE(received->tensor.data_as<std::uint8_t>(), nullptr);
    EXPECT_EQ(received->tensor.data_as<std::uint8_t>()[0], 0x3cU);
  }

  ASSERT_TRUE(subscriber->close());
  ASSERT_TRUE(publisher->close());
  const bool failed_without_timeout = wait_for_session_terminal(session);
  if (!failed_without_timeout) {
    session.request_stop();
  }
  session.wait();

  ASSERT_TRUE(failed_without_timeout);
  EXPECT_EQ(session.termination(), RunTermination::kFailed);
  ASSERT_EQ(session.diagnostics().size(), 1U);
  EXPECT_EQ(session.diagnostics().front().token, "RUNTIME_READINESS_BINDING_UNAVAILABLE");
  EXPECT_EQ(session.diagnostics().front().service, "terminal_input_end_of_stream");
}

// A replacement descriptor-Tensor subscriber attached before terminal drain becomes the current
// required recipient and receives EOS after the initial subscriber's accepted data.
// REQ: CHAN-136
// REQ: CHAN-144
// REQ: RUNTIME-017
TEST(RuntimeExternalChannelTest,
     RequiredDescriptorTensorExportReplacementPresentAtTerminalDrainReceivesEndOfStream) {
  const std::string import_name = unique_endpoint_name();
  const std::string export_name = unique_endpoint_name();

  Graph graph{"runtime-required-descriptor-tensor-export-replacement-terminal"};
  graph.set_default_clock(graph.add_clock<RealtimeClock>("runtime-clock"));
  const auto imported =
      graph.import_topic<schema::ImageT>("holomq://" + import_name,
                                         external_image_import_options(import_name),
                                         external_image_contract());
  TopicOptions export_options = external_image_export_options(export_name);
  export_options.attach = TopicAttachPolicy{
      .peer = PeerRequirement::kRequired,
      .max_wait = 10s,
  };
  const auto exported = graph.export_topic<schema::ImageT>(
      "holomq://" + export_name, std::move(export_options), external_image_contract());
  const auto forwarder = graph.op<ExternalImageForwarder>("forwarder");
  graph.add_flow(imported.output(), forwarder->input, ConnectionOptions{.queue_depth = 1U});
  graph.add_flow(forwarder->output, exported.input(), ConnectionOptions{.queue_depth = 1U});

  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  auto external_runtime = ChannelRuntime::create();
  ASSERT_TRUE(external_runtime) << runtime_error_message(external_runtime.error());
  auto publisher = external_runtime->create_tensor_schema_publisher<ExternalImagePayload>(
      plan.endpoint_catalog(), import_name);
  ASSERT_TRUE(publisher) << runtime_error_message(publisher.error());

  RunSession session = run_async(plan);
  auto initial_subscriber = external_runtime->create_tensor_schema_subscriber<ExternalImagePayload>(
      plan.endpoint_catalog(), export_name);
  ASSERT_TRUE(initial_subscriber) << runtime_error_message(initial_subscriber.error());
  ASSERT_TRUE(wait_for_schema_subscriber(*publisher));
  ASSERT_TRUE(wait_for_schema_publisher(*initial_subscriber));
  auto tensor = make_external_image_tensor(0x3cU);
  ASSERT_TRUE(tensor) << runtime_error_message(tensor.error());
  auto metadata = schema::build_image_metadata(
      *tensor,
      schema::ImageMetadataOptions{
          .height = 1U, .width = 8U, .encoding = schema::ImageEncoding_MONO8});
  ASSERT_TRUE(metadata) << runtime_error_message(metadata.error());
  ASSERT_TRUE(publisher->emit(*metadata, std::move(*tensor)));
  {
    auto received = wait_for_schema_sample(*initial_subscriber);
    ASSERT_TRUE(received) << runtime_error_message(received.error());
    ASSERT_NE(received->tensor.data_as<std::uint8_t>(), nullptr);
    EXPECT_EQ(received->tensor.data_as<std::uint8_t>()[0], 0x3cU);
  }
  ASSERT_TRUE(initial_subscriber->close());

  auto replacement = external_runtime->create_tensor_schema_subscriber<ExternalImagePayload>(
      plan.endpoint_catalog(), export_name);
  ASSERT_TRUE(replacement) << runtime_error_message(replacement.error());
  ASSERT_TRUE(wait_for_schema_publisher(*replacement));
  ASSERT_TRUE(publisher->close());

  auto terminal = wait_for_schema_sample(*replacement);
  ASSERT_FALSE(terminal);
  EXPECT_EQ(terminal.error().code, ErrorCode::kClosed);
  const bool completed_without_timeout = wait_for_session_terminal(session);
  if (!completed_without_timeout) {
    session.request_stop();
  }
  session.wait();

  ASSERT_TRUE(completed_without_timeout);
  EXPECT_EQ(session.termination(), RunTermination::kCompleted);
  EXPECT_TRUE(session.diagnostics().empty());
  EXPECT_TRUE(replacement->close());
}

// REQ: CHAN-090
// REQ: CHAN-101
// REQ: RUNTIME-019
// REQ: SYSTEM-003
// REQ: SYSTEM-006
TEST(RuntimeExternalChannelTest, OptionalTensorGraphExportAcceptsZeroConnectedRecipients) {
  ExternalTensorExportSource::reset();
  const std::string export_name = unique_endpoint_name();

  //   source --[drop-oldest, depth=1]--> holomq://endpoint  (optional Tensor export, no
  //   subscribers)
  Graph graph{"runtime-optional-external-tensor-export"};
  const ClockKey runtime_clock = graph.add_clock<RealtimeClock>("runtime-clock");
  graph.set_default_clock(runtime_clock);
  auto exported = graph.export_topic<Tensor>("holomq://" + export_name,
                                             optional_tensor_export_options(export_name));
  const auto source = graph.op<ExternalTensorExportSource>("source");
  graph.add_flow(source->output,
                 exported.input(),
                 ConnectionOptions{.queue_policy = QueuePolicy::kDropOldest, .queue_depth = 1U});

  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();

  RunSession session = run_async(plan);
  const bool compute_started = ExternalTensorExportSource::wait_for_compute_started();
  ExternalTensorExportSource::allow_emit();
  const bool emit_completed = ExternalTensorExportSource::wait_for_emit_completed();
  session.request_stop();
  session.wait();

  ASSERT_TRUE(compute_started);
  ASSERT_TRUE(emit_completed);
  // Key assertion: Tensor emit returns kSuccess even with no subscribers (optional drop).
  EXPECT_EQ(ExternalTensorExportSource::last_emit_error(), ErrorCode::kSuccess);
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
}

// REQ: CHAN-122
// REQ: EXTERNAL-010
TEST(RuntimeExternalChannelTest, OptionalDescriptorTensorExportRecoversAfterSubscriberAttaches) {
  LateAttachExternalImageExportSource::reset();
  const std::string endpoint_name = unique_endpoint_name();

  Graph graph{"runtime-late-attach-image-export"};
  const ClockKey runtime_clock = graph.add_clock<RealtimeClock>("runtime-clock");
  graph.set_default_clock(runtime_clock);
  auto exported = graph.export_topic<schema::ImageT>("holomq://" + endpoint_name,
                                                     external_image_export_options(endpoint_name),
                                                     external_image_contract());
  const auto source = graph.op<LateAttachExternalImageExportSource>("source");
  graph.add_flow(source->output, exported.input(), ConnectionOptions{.queue_depth = 1U});

  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  auto external_runtime = ChannelRuntime::create();
  ASSERT_TRUE(external_runtime) << runtime_error_message(external_runtime.error());

  // Key setup: let this external-only descriptor output consume its first graph-local emit before
  // any subscriber exists, then attach the subscriber while the same owning RunSession remains
  // live.
  RunSession session = run_async(plan);
  ASSERT_TRUE(LateAttachExternalImageExportSource::wait_for_first_emit());
  ASSERT_EQ(LateAttachExternalImageExportSource::emit_error.load(std::memory_order_relaxed),
            ErrorCode::kSuccess);

  auto subscriber = external_runtime->create_tensor_schema_subscriber<ExternalImagePayload>(
      plan.endpoint_catalog(), endpoint_name);
  ASSERT_TRUE(subscriber) << runtime_error_message(subscriber.error());
  ASSERT_TRUE(wait_for_schema_publisher(*subscriber));
  LateAttachExternalImageExportSource::allow_second_emit();
  ASSERT_TRUE(LateAttachExternalImageExportSource::wait_for_second_emit());
  auto received = wait_for_schema_sample(*subscriber);

  session.request_stop();
  session.wait();
  ASSERT_TRUE(received) << runtime_error_message(received.error());
  ASSERT_NE(received->tensor.data_as<std::uint8_t>(), nullptr);
  // Key assertion: the export-phase OutputState safely retains its Runtime owner, and the
  // publisher's private operation sequence remains contiguous despite the earlier graph-local
  // skip, so the first post-attach Tensor reaches the external subscriber.
  EXPECT_EQ(received->tensor.data_as<std::uint8_t>()[0], 0x42U);
  EXPECT_EQ(LateAttachExternalImageExportSource::emit_error.load(std::memory_order_relaxed),
            ErrorCode::kSuccess);
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
}

// An optional descriptor-schema endpoint that no peer ever attaches to has no terminal observer.
// ComputeCalls{0} therefore completes without executing the source or retrying close forever.
// REQ: CHAN-136
// REQ: RUNTIME-050
TEST(RuntimeExternalChannelTest,
     OptionalDescriptorTensorExportCompletesWithoutPeerAtZeroComputeCalls) {
  ExternalImageExportSource::reset(0U);
  const std::string endpoint_name = unique_endpoint_name();

  Graph graph{"runtime-optional-image-export-zero-compute"};
  const ClockKey runtime_clock = graph.add_clock<RealtimeClock>("runtime-clock");
  graph.set_default_clock(runtime_clock);
  auto exported = graph.export_topic<schema::ImageT>("holomq://" + endpoint_name,
                                                     external_image_export_options(endpoint_name),
                                                     external_image_contract());
  const auto source = graph.op<ExternalImageExportSource>("source");
  graph.add_flow(source->output, exported.input(), ConnectionOptions{.queue_depth = 1U});

  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();

  RunSession session = run_async(plan);
  const bool completed = wait_for_session_terminal(session);
  if (!completed) {
    session.request_stop();
  }
  session.wait();

  ASSERT_TRUE(completed) << "zero-limit optional descriptor export did not complete within 5s";
  EXPECT_FALSE(ExternalImageExportSource::wait_for_compute_started(100ms));
  expect_completion_progress(session, 0U);
  EXPECT_EQ(session.termination(), RunTermination::kCompleted);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
}

// A positive completion limit may skip descriptor-schema data before any optional peer attaches.
// That graph-local skip must not strand the terminal close on a missing subscriber.
// REQ: CHAN-136
// REQ: RUNTIME-050
TEST(RuntimeExternalChannelTest,
     OptionalDescriptorTensorExportCompletesWithoutPeerAfterOneComputeCall) {
  ExternalImageExportSource::reset(1U);
  const std::string endpoint_name = unique_endpoint_name();

  Graph graph{"runtime-optional-image-export-one-compute"};
  const ClockKey runtime_clock = graph.add_clock<RealtimeClock>("runtime-clock");
  graph.set_default_clock(runtime_clock);
  auto exported = graph.export_topic<schema::ImageT>("holomq://" + endpoint_name,
                                                     external_image_export_options(endpoint_name),
                                                     external_image_contract());
  const auto source = graph.op<ExternalImageExportSource>("source");
  graph.add_flow(source->output, exported.input(), ConnectionOptions{.queue_depth = 1U});

  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();

  RunSession session = run_async(plan);
  ASSERT_TRUE(ExternalImageExportSource::wait_for_compute_started());
  ExternalImageExportSource::allow_emit();
  ASSERT_TRUE(ExternalImageExportSource::wait_for_emit_completed());
  const bool completed = wait_for_session_terminal(session);
  if (!completed) {
    session.request_stop();
  }
  session.wait();

  ASSERT_TRUE(completed) << "one-call optional descriptor export did not complete within 5s";
  EXPECT_EQ(ExternalImageExportSource::emit_error.load(std::memory_order_relaxed),
            ErrorCode::kSuccess);
  expect_completion_progress(session, 1U);
  EXPECT_EQ(session.termination(), RunTermination::kCompleted);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
}

// A late descriptor-schema peer observes its own contiguous data/EOS sequence after the endpoint
// skipped an earlier optional publication. The close must publish exactly one endpoint-local EOS.
// REQ: CHAN-136
// REQ: EXTERNAL-010
// REQ: RUNTIME-050
TEST(RuntimeExternalChannelTest,
     OptionalDescriptorTensorExportNaturalCompletionUsesEndpointLocalEndOfStreamSequence) {
  LateAttachExternalImageExportSource::reset(true);
  const std::string endpoint_name = unique_endpoint_name();

  Graph graph{"runtime-optional-image-export-natural-late-attach"};
  const ClockKey runtime_clock = graph.add_clock<RealtimeClock>("runtime-clock");
  graph.set_default_clock(runtime_clock);
  auto exported = graph.export_topic<schema::ImageT>("holomq://" + endpoint_name,
                                                     external_image_export_options(endpoint_name),
                                                     external_image_contract());
  const auto source = graph.op<LateAttachExternalImageExportSource>("source");
  graph.add_flow(source->output, exported.input(), ConnectionOptions{.queue_depth = 1U});

  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  auto external_runtime = ChannelRuntime::create();
  ASSERT_TRUE(external_runtime) << runtime_error_message(external_runtime.error());

  RunSession session = run_async(plan);
  ASSERT_TRUE(LateAttachExternalImageExportSource::wait_for_first_emit());
  ASSERT_EQ(LateAttachExternalImageExportSource::emit_error.load(std::memory_order_relaxed),
            ErrorCode::kSuccess);

  auto subscriber = external_runtime->create_tensor_schema_subscriber<ExternalImagePayload>(
      plan.endpoint_catalog(), endpoint_name);
  ASSERT_TRUE(subscriber) << runtime_error_message(subscriber.error());
  ASSERT_TRUE(wait_for_schema_publisher(*subscriber));
  LateAttachExternalImageExportSource::allow_second_emit();
  ASSERT_TRUE(LateAttachExternalImageExportSource::wait_for_second_emit());
  ASSERT_EQ(LateAttachExternalImageExportSource::emit_error.load(std::memory_order_relaxed),
            ErrorCode::kSuccess);

  auto data = wait_for_catalog_tensor_schema_operation(*subscriber);
  ASSERT_TRUE(data) << runtime_error_message(data.error());
  ASSERT_EQ(data->kind, channel_internal::CatalogReceivedOperationKind::kData);
  EXPECT_EQ(data->metadata.sequence, 1U);
  detail::RuntimeChannelBindingAccess::release_catalog_tensor_schema_payload_loan(*subscriber);
  auto terminal = wait_for_catalog_tensor_schema_operation(*subscriber);
  ASSERT_TRUE(terminal) << runtime_error_message(terminal.error());
  EXPECT_EQ(terminal->kind, channel_internal::CatalogReceivedOperationKind::kEndOfStream);
  EXPECT_EQ(terminal->metadata.sequence, 2U);

  const bool completed = wait_for_session_terminal(session);
  if (!completed) {
    session.request_stop();
  }
  session.wait();
  ASSERT_TRUE(completed) << "late-attached optional descriptor export did not complete within 5s";
  expect_completion_progress(session, 2U);
  EXPECT_EQ(session.termination(), RunTermination::kCompleted);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);

  auto after_terminal =
      detail::RuntimeChannelBindingAccess::receive_catalog_tensor_schema_operation(*subscriber,
                                                                                   true);
  ASSERT_TRUE(after_terminal) << runtime_error_message(after_terminal.error());
  EXPECT_EQ(after_terminal->kind, channel_internal::CatalogReceivedOperationKind::kNone);
}

// REQ: CHAN-090
// REQ: RUNTIME-019
// REQ: SYSTEM-003
// REQ: SYSTEM-006
TEST(RuntimeExternalChannelTest, OptionalTensorDropDoesNotPoisonAcceptedInternalFanout) {
  ExternalTensorExportSource::reset();
  ExternalTensorExportSink::reset();
  const std::string export_name = unique_endpoint_name();

  //   source --[depth=1]--> sink
  //   source --[depth=1]--> holomq://endpoint  (optional, no subscribers -- may drop)
  // Key setup: fanout to an internal sink and an optional external export with no subscribers.
  Graph graph{"runtime-mixed-optional-external-tensor-export"};
  const ClockKey runtime_clock = graph.add_clock<RealtimeClock>("runtime-clock");
  graph.set_default_clock(runtime_clock);
  auto exported = graph.export_topic<Tensor>("holomq://" + export_name,
                                             optional_tensor_export_options(export_name));
  const auto source = graph.op<ExternalTensorExportSource>("source");
  const auto sink = graph.op<ExternalTensorExportSink>("sink");
  graph.add_flow(source->output, sink->input, ConnectionOptions{.queue_depth = 1U});
  graph.add_flow(source->output, exported.input(), ConnectionOptions{.queue_depth = 1U});

  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();

  RunSession session = run_async(plan);
  const bool compute_started = ExternalTensorExportSource::wait_for_compute_started();
  ExternalTensorExportSource::allow_emit();
  const bool emit_completed = ExternalTensorExportSource::wait_for_emit_completed();
  const bool internal_received = ExternalTensorExportSink::wait_for_sample();
  session.request_stop();
  session.wait();

  ASSERT_TRUE(compute_started);
  ASSERT_TRUE(emit_completed);
  ASSERT_TRUE(internal_received);
  EXPECT_EQ(ExternalTensorExportSource::last_emit_error(), ErrorCode::kSuccess);
  // Key assertion: internal sink receives exactly 1 sample despite the external export drop.
  EXPECT_EQ(ExternalTensorExportSink::received.load(std::memory_order_relaxed), 1U);
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
}

// REQ: CHAN-122
// REQ: EXTERNAL-010
TEST(RuntimeExternalChannelTest,
     DescriptorTensorSchemaExportStagesGraphBackingForCatalogSubscriber) {
  ExternalImageExportSource::reset();
  const std::string endpoint_name = unique_endpoint_name();

  Graph graph{"runtime-external-image-export"};
  (void)graph.add_clock<ManualClock>("a-unrelated-clock", std::int64_t{0});
  const ClockKey capture_clock = graph.add_clock<ManualClock>("z-capture-clock", std::int64_t{0});
  const ClockKey runtime_clock = graph.add_clock<RealtimeClock>("m-runtime-clock");
  graph.set_default_clock(runtime_clock);
  graph.add_clock_conversion(capture_clock, runtime_clock, AffineClockConversion{});
  TopicOptions export_options = external_image_export_options(endpoint_name);
  export_options.capture_time = RequireCaptureTime{.clock = capture_clock};
  auto exported = graph.export_topic<schema::ImageT>(
      "holomq://" + endpoint_name, std::move(export_options), external_image_contract());
  const auto source = graph.op<ExternalImageExportSource>("source");
  graph.add_flow(source->output, exported.input(), ConnectionOptions{.queue_depth = 1U});

  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  const ClockCatalogView clock_catalog = plan.clock_catalog();
  const auto plan_capture_clock = std::find_if(
      clock_catalog.clocks.begin(),
      clock_catalog.clocks.end(),
      [&](const ClockPlanRecord& candidate) { return candidate.key == capture_clock; });
  ASSERT_NE(plan_capture_clock, clock_catalog.clocks.end());
  const auto endpoint = plan.endpoint_catalog().endpoint(endpoint_name);
  ASSERT_TRUE(endpoint);
  const ClockId catalog_capture_clock =
      detail::ChannelEndpointCatalogAccess::capture_clock_id(*endpoint);
  ASSERT_TRUE(catalog_capture_clock.valid());
  ASSERT_NE(catalog_capture_clock, plan_capture_clock->id);
  ExternalImageExportSource::set_capture_time(
      TimePoint{.timestamp_ns = 654321, .clock_id = plan_capture_clock->id});
  auto external_runtime = ChannelRuntime::create();
  ASSERT_TRUE(external_runtime) << runtime_error_message(external_runtime.error());

  const auto* const endpoint_record = detail::ChannelEndpointCatalogAccess::record(*endpoint);
  const auto* const type_ops = detail::runtime_port_type_ops<schema::ImageT>();
  ASSERT_NE(endpoint_record, nullptr);
  ASSERT_NE(type_ops, nullptr);
  ASSERT_TRUE(endpoint_record->payload_identity.schema);
  ASSERT_EQ(endpoint_record->service_static.tensor_fields.size(), 1U);
  const auto& tensor_contract = endpoint_record->service_static.tensor_fields.front().tensor;
  ASSERT_TRUE(tensor_contract.max_byte_span);
  ASSERT_TRUE(tensor_contract.memory_kind);
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access): guarded by ASSERT_TRUE above.
  const std::uint64_t max_byte_span = *tensor_contract.max_byte_span;
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access): guarded by ASSERT_TRUE above.
  const MemoryKind memory_kind = *tensor_contract.memory_kind;
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access): guarded by ASSERT_TRUE above.
  const auto& payload_schema = *endpoint_record->payload_identity.schema;
  // Key setup: first prove a staging override cannot widen the immutable field contract, then run
  // the valid graph export into a catalog descriptor-schema subscriber.
  std::array descriptor_offsets{type_ops->descriptor_offset};
  std::array staging_override{channel_internal::TensorSchemaStagingOptions{
      .max_byte_span = max_byte_span + 1U,
      .slot_count = 1U,
      .dtype = tensor_contract.dtype,
      .rank = tensor_contract.rank,
      .axes = tensor_contract.axes,
      .strides = tensor_contract.strides,
      .memory_kind = memory_kind,
      .device = {},
      .foreign_emit_policy = tensor_contract.foreign_emit_policy,
  }};
  auto malformed_staging =
      detail::RuntimeChannelBindingAccess::create_catalog_tensor_schema_publisher(
          *external_runtime,
          *endpoint,
          payload_schema,
          type_ops->wire_capacity,
          descriptor_offsets,
          staging_override,
          nullptr);
  ASSERT_FALSE(malformed_staging);
  EXPECT_EQ(malformed_staging.error().code, ErrorCode::kInvalidArgument);
  EXPECT_EQ(malformed_staging.error().message,
            "descriptor Tensor schema staging override changes an immutable contract field");

  RunSession session = run_async(plan);
  ASSERT_TRUE(ExternalImageExportSource::wait_for_compute_started());
  auto subscriber = external_runtime->create_tensor_schema_subscriber<ExternalImagePayload>(
      plan.endpoint_catalog(), endpoint_name);
  ASSERT_TRUE(subscriber) << runtime_error_message(subscriber.error()) << '\n'
                          << plan.endpoint_catalog().json();
  ASSERT_TRUE(wait_for_schema_publisher(*subscriber));
  ExternalImageExportSource::allow_emit();
  ASSERT_TRUE(ExternalImageExportSource::wait_for_emit_completed());
  ASSERT_EQ(ExternalImageExportSource::emit_error.load(std::memory_order_relaxed),
            ErrorCode::kSuccess);
  auto received = wait_for_schema_sample(*subscriber);

  session.request_stop();
  session.wait();
  ASSERT_TRUE(received) << runtime_error_message(received.error());
  // Key assertion: the subscriber receives BufferHandle-backed Tensor bytes plus every authored
  // SampleMetadata field, including the canonical catalog capture-clock identity.
  ASSERT_NE(received->root(), nullptr);
  EXPECT_EQ(received->root()->height(), 1U);
  EXPECT_EQ(received->root()->width(), 8U);
  EXPECT_EQ(received->payload.payload_kind, ChannelPayloadKind::kBufferHandle);
  EXPECT_EQ(received->tensor.nbytes(), 8U);
  ASSERT_NE(received->tensor.data_as<std::uint8_t>(), nullptr);
  EXPECT_EQ(received->tensor.data_as<std::uint8_t>()[0], 0x5aU);
  EXPECT_EQ(received->sample_metadata.capture_timestamp_ns, 654321);
  EXPECT_EQ(received->sample_metadata.source_clock_id, catalog_capture_clock);
  EXPECT_EQ(received->sample_metadata.trace_id, 31U);
  EXPECT_EQ(received->sample_metadata.frame_id, 41U);
  EXPECT_EQ(received->sample_metadata.flags, SampleFlags::kDegraded);
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
}

// REQ: CHAN-068
// REQ: RUNTIME-015
// REQ: RUNTIME-017
TEST(RuntimeExternalChannelTest, CanonicalScalarGraphExportReachesExternalSubscriberWithMetadata) {
  ExternalScalarExportSource::reset();
  const std::string export_name = unique_endpoint_name();

  //   source --[depth=4]--> holomq://endpoint --> external_subscriber
  // Key setup: subscriber created before run; Runtime publisher connects lazily during emit.
  Graph graph{"runtime-external-export"};
  const ClockKey runtime_clock = graph.add_clock<RealtimeClock>("runtime-clock");
  graph.set_default_clock(runtime_clock);

  TopicOptions export_options;
  export_options.endpoint_name = export_name;
  const auto exported =
      graph.export_topic<std::uint32_t>("holomq://" + export_name, std::move(export_options));
  const auto source = graph.op<ExternalScalarExportSource>("source");
  graph.add_flow(source->output, exported.input(), ConnectionOptions{.queue_depth = 4U});

  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  auto external_runtime = ChannelRuntime::create();
  ASSERT_TRUE(external_runtime) << runtime_error_message(external_runtime.error());
  auto subscriber =
      external_runtime->create_subscriber<std::uint32_t>(plan.endpoint_catalog(), export_name);
  ASSERT_TRUE(subscriber) << runtime_error_message(subscriber.error());

  RunSession session = run_async(plan);
  if (!ExternalScalarExportSource::wait_for_compute_started()) {
    ExternalScalarExportSource::allow_emit();
    session.request_stop();
    session.wait();
    FAIL() << "Graph export source did not enter compute; termination="
           << static_cast<int>(session.termination()) << "; diagnostics="
           << (session.diagnostics().empty() ? "<none>" : session.diagnostics().front().token);
  }
  // The endpoint is optional, so Graph start must not wait. Observe the Runtime-owned publisher
  // only after compute has entered, then release the test source. The subscriber remains alive and
  // Runtime performs the matching publisher refresh immediately before the send boundary.
  auto subscriber_ready = wait_for_publisher(*subscriber);
  ExternalScalarExportSource::allow_emit();
  ASSERT_TRUE(subscriber_ready) << runtime_error_message(subscriber_ready.error());
  auto received = wait_for_sample(*subscriber);

  session.request_stop();
  session.wait();
  ASSERT_TRUE(received)
      << runtime_error_message(received.error()) << "; source computes="
      << ExternalScalarExportSource::compute_count.load(std::memory_order_relaxed)
      << "; emit error="
      << static_cast<int>(ExternalScalarExportSource::emit_error.load(std::memory_order_relaxed))
      << "; termination=" << static_cast<int>(session.termination()) << "; diagnostics="
      << (session.diagnostics().empty() ? "<none>" : session.diagnostics().front().token);
  // Key assertion: sample carries correct trace_id/frame_id and kInlineTyped payload kind.
  EXPECT_EQ(received->data, 0xdecafbadU);
  EXPECT_NE(received->metadata.sequence, 0U);
  EXPECT_EQ(received->metadata.trace_id, 17U);
  EXPECT_EQ(received->metadata.frame_id, 23U);
  EXPECT_EQ(received->metadata.byte_size, sizeof(std::uint32_t));
  EXPECT_EQ(received->metadata.payload_kind, PayloadKind::kInlineTyped);
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
}

// REQ: CHAN-068
// REQ: RUNTIME-017
TEST(RuntimeExternalChannelTest,
     CapturedGraphExportTranslatesPlanClockIdentityIntoCanonicalCatalogIdentity) {
  ExternalScalarExportSource::reset();
  const std::string export_name = unique_endpoint_name();

  //   source --[depth=4]--> holomq://endpoint --> external_subscriber
  //   Three clocks: a-unrelated (plan-local only), z-capture, m-health-age (catalog).
  //   Unrelated clock shifts plan-local IDs so capture clock ID != catalog clock ID.
  Graph graph{"runtime-external-export-clock-translation"};
  // The unrelated domain remains plan-local, while the portable endpoint catalog projects only
  // capture and its conversion target. The resulting dense identity spaces intentionally differ.
  (void)graph.add_clock<ManualClock>("a-unrelated-clock", std::int64_t{0});
  const ClockKey capture_key = graph.add_clock<ManualClock>("z-capture-clock", std::int64_t{0});
  const ClockKey age_key = graph.add_clock<RealtimeClock>("m-health-age-clock");
  graph.set_default_clock(age_key);
  graph.add_clock_conversion(capture_key, age_key, AffineClockConversion{});

  TopicOptions export_options;
  export_options.endpoint_name = export_name;
  export_options.capture_time = RequireCaptureTime{.clock = capture_key};
  const auto exported =
      graph.export_topic<std::uint32_t>("holomq://" + export_name, std::move(export_options));
  const auto source = graph.op<ExternalScalarExportSource>("source");
  graph.add_flow(source->output, exported.input(), ConnectionOptions{.queue_depth = 4U});

  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  const ClockCatalogView clock_catalog = plan.clock_catalog();
  const auto plan_clock =
      std::find_if(clock_catalog.clocks.begin(),
                   clock_catalog.clocks.end(),
                   [&](const ClockPlanRecord& candidate) { return candidate.key == capture_key; });
  ASSERT_NE(plan_clock, clock_catalog.clocks.end());
  const auto endpoint = plan.endpoint_catalog().endpoint(export_name);
  ASSERT_TRUE(endpoint);
  const ClockId catalog_capture_clock =
      detail::ChannelEndpointCatalogAccess::capture_clock_id(*endpoint);
  ASSERT_TRUE(catalog_capture_clock.valid());
  // Key setup: catalog_capture_clock != plan_clock->id; Runtime must translate IDs on send.
  ASSERT_NE(catalog_capture_clock, plan_clock->id);
  ExternalScalarExportSource::set_capture_time(
      TimePoint{.timestamp_ns = 4242, .clock_id = plan_clock->id});

  auto external_runtime = ChannelRuntime::create();
  ASSERT_TRUE(external_runtime) << runtime_error_message(external_runtime.error());
  auto subscriber =
      external_runtime->create_subscriber<std::uint32_t>(plan.endpoint_catalog(), export_name);
  ASSERT_TRUE(subscriber) << runtime_error_message(subscriber.error());

  RunSession session = run_async(plan);
  if (!ExternalScalarExportSource::wait_for_compute_started()) {
    ExternalScalarExportSource::allow_emit();
    session.request_stop();
    session.wait();
    FAIL() << "captured Graph export source did not enter compute; termination="
           << static_cast<int>(session.termination());
  }
  auto subscriber_ready = wait_for_publisher(*subscriber);
  ExternalScalarExportSource::allow_emit();
  ASSERT_TRUE(subscriber_ready) << runtime_error_message(subscriber_ready.error());
  auto received = wait_for_sample(*subscriber);

  session.request_stop();
  session.wait();
  ASSERT_TRUE(received)
      << runtime_error_message(received.error()) << "; emit error="
      << static_cast<int>(ExternalScalarExportSource::emit_error.load(std::memory_order_relaxed))
      << "; termination=" << static_cast<int>(session.termination()) << "; diagnostics="
      << (session.diagnostics().empty() ? "<none>" : session.diagnostics().front().token);
  EXPECT_EQ(received->data, 0xdecafbadU);
  EXPECT_EQ(received->metadata.capture_timestamp_ns, 4242);
  // Key assertion: on-wire clock ID is the catalog identity, not the plan-local ID.
  EXPECT_EQ(received->metadata.source_clock_id, catalog_capture_clock);
  EXPECT_EQ(received->metadata.trace_id, 17U);
  EXPECT_EQ(received->metadata.frame_id, 23U);
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
}

// Forwards one imported ImageT to a drop-oldest export, and to a drop-oldest Graph-internal sink
// when `local_member` is set, then closes the import. Both arms must receive the publication and
// end of stream, and the session must complete naturally.
//
//   import --> forwarder.output --[drop-oldest]--> export
//                              `--[drop-oldest]--> local.in   (when local_member)
void expect_drop_oldest_image_export(const bool local_member, const bool force_holomq_fanout) {
  ExternalImageImportProbe::reset();
  const std::string import_name = unique_endpoint_name();
  const std::string export_name = unique_endpoint_name();

  Graph graph{"runtime-external-image-drop-oldest-export"};
  graph.set_default_clock(graph.add_clock<RealtimeClock>("runtime-clock"));
  const auto imported =
      graph.import_topic<schema::ImageT>("holomq://" + import_name,
                                         external_image_import_options(import_name),
                                         external_image_contract());
  const auto forwarder = graph.op<ExternalImageForwarder>("forwarder");
  TopicOptions export_options = external_image_export_options(export_name);
  export_options.attach = TopicAttachPolicy{
      .peer = PeerRequirement::kRequired,
      .max_wait = 10s,
  };
  const auto exported = graph.export_topic<schema::ImageT>(
      "holomq://" + export_name, std::move(export_options), external_image_contract());
  const ConnectionOptions drop_oldest{.queue_policy = QueuePolicy::kDropOldest, .queue_depth = 1U};
  graph.add_flow(imported.output(), forwarder->input, ConnectionOptions{.queue_depth = 1U});
  graph.add_flow(forwarder->output, exported.input(), drop_oldest);
  if (local_member) {
    const auto local = graph.op<ExternalImageImportSink>("local");
    graph.add_flow(forwarder->output, local->input, drop_oldest);
  }

  const ExecutionPlan plan = force_holomq_fanout ? detail::compile_with_process_local_holomq_fanout(
                                                       graph, CompileOptions{})
                                                 : compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto state = detail::ExecutionPlanAccess::retain(plan);
  ASSERT_NE(state, nullptr);
  if (local_member) {
    ASSERT_EQ(state->process_local_channel_services.size(), 1U);
    EXPECT_EQ(state->process_local_channel_services.front().delivery_kind,
              force_holomq_fanout ? detail::ProcessLocalChannelDeliveryKind::kHoloMQ
                                  : detail::ProcessLocalChannelDeliveryKind::kDirectRetained);
  }

  auto external_runtime = ChannelRuntime::create();
  ASSERT_TRUE(external_runtime) << runtime_error_message(external_runtime.error());
  auto publisher = external_runtime->create_tensor_schema_publisher<ExternalImagePayload>(
      plan.endpoint_catalog(), import_name);
  ASSERT_TRUE(publisher) << runtime_error_message(publisher.error());
  RunSession session = run_async(plan);
  auto subscriber = external_runtime->create_tensor_schema_subscriber<ExternalImagePayload>(
      plan.endpoint_catalog(), export_name);
  if (!subscriber) {
    session.request_stop();
    session.wait();
  }
  ASSERT_TRUE(subscriber) << runtime_error_message(subscriber.error()) << "; "
                          << (session.diagnostics().empty() ? ""
                                                            : session.diagnostics().front().token);
  ASSERT_TRUE(wait_for_schema_subscriber(*publisher));
  ASSERT_TRUE(wait_for_schema_publisher(*subscriber));
  auto tensor = make_external_image_tensor(0x6dU);
  ASSERT_TRUE(tensor) << runtime_error_message(tensor.error());
  auto metadata = schema::build_image_metadata(
      *tensor,
      schema::ImageMetadataOptions{
          .height = 1U, .width = 8U, .encoding = schema::ImageEncoding_MONO8});
  ASSERT_TRUE(metadata) << runtime_error_message(metadata.error());
  ASSERT_TRUE(publisher->emit(*metadata, std::move(*tensor)));

  // Key assertion: the export (and the local sink) receive the one publication, the export then
  // reads end of stream, and natural completion proves the local sink did too.
  {
    auto received = wait_for_schema_sample(*subscriber);
    ASSERT_TRUE(received) << runtime_error_message(received.error());
    ASSERT_NE(received->tensor.data_as<std::uint8_t>(), nullptr);
    EXPECT_EQ(received->tensor.data_as<std::uint8_t>()[0], 0x6dU);
  }
  if (local_member) {
    ASSERT_TRUE(ExternalImageImportProbe::wait_for_sample());
    EXPECT_EQ(ExternalImageImportProbe::first_byte.load(std::memory_order_relaxed), 0x6dU);
    EXPECT_EQ(ExternalImageImportProbe::width.load(std::memory_order_relaxed), 8U);
  }
  ASSERT_TRUE(publisher->close());
  auto terminal = wait_for_schema_sample(*subscriber);
  ASSERT_FALSE(terminal);
  EXPECT_EQ(terminal.error().code, ErrorCode::kClosed);
  const bool completed = wait_for_session_terminal(session);
  if (!completed) {
    session.request_stop();
  }
  session.wait();
  EXPECT_TRUE(completed);
  EXPECT_EQ(session.termination(), RunTermination::kCompleted);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
  EXPECT_EQ(ExternalImageImportProbe::count.load(std::memory_order_acquire),
            local_member ? 1U : 0U);
  EXPECT_TRUE(subscriber->close());
}

// A drop-oldest descriptor-Tensor export gets a replacing publisher: the graph's bounded retry
// applies only to blocking queues, as for inline and raw Tensor exports.
// REQ: CHAN-128
TEST(RuntimeExternalChannelTest, DropOldestDescriptorTensorExportDeliversAndCloses) {
  expect_drop_oldest_image_export(false, false);
}

// A drop-oldest ImageT family with an exported member and a Graph-internal member keeps the
// internal member direct, as the HoloMQ fanout control does not (CHAN-169). The export's leg is
// reserved before the local store, copies the forwarded foreign Tensor into its staging pool, and
// re-encodes the payload with its catalog sequence after the store.
// REQ: CHAN-169
TEST(RuntimeExternalChannelTest, MixedLocalityImageFanoutDeliversToTheExportAndTheDirectRecipient) {
  for (const bool force_holomq_fanout : {false, true}) {
    SCOPED_TRACE(force_holomq_fanout ? "holomq family" : "mixed-locality family");
    expect_drop_oldest_image_export(true, force_holomq_fanout);
  }
}

}  // namespace
}  // namespace holoscan
