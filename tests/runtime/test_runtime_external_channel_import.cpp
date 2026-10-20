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
#include <holoscan/memory/block_memory_pool.hpp>
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
#include "core/diagnostic_registry.hpp"
#include "core/graph_compiler_access.hpp"
#include "runtime/channel_runtime_binding_access.hpp"
#include "runtime/endpoint_clock_health_runtime.hpp"
#include "runtime/graph_exec.hpp"
#include "runtime/runtime_channel_binding_test_access.hpp"
#include "runtime/runtime_phase_counters.hpp"
#include "runtime/runtime_test_access.hpp"
#include "runtime_external_channel_test_support.hpp"
#include "runtime_test_graph.hpp"

namespace holoscan {
namespace {

using namespace std::chrono_literals;

using namespace test::external_channel;

class RetryableRetainedTensorReplacementBindControl final
    : public detail::RuntimeChannelSessionBindTestControl {
 public:
  RetryableRetainedTensorReplacementBindControl(
      const PortId input, const VertexId owner, const PortId contention_input = {},
      const std::uint32_t paused_activation_retirement_ordinal = 0U,
      const bool reject_reuse_replacement = true) noexcept
      : input_(input),
        contention_input_(contention_input),
        owner_(owner),
        paused_activation_retirement_ordinal_(paused_activation_retirement_ordinal),
        reject_reuse_replacement_(reject_reuse_replacement) {}

  ~RetryableRetainedTensorReplacementBindControl() override { release_activation_retirement(); }

  [[nodiscard]] expected<void, Error> bind(
      detail::RuntimeChannelSession& session) noexcept override {
    if (contention_input_) {
      auto observed = detail::RuntimeChannelSessionTestAccess::set_input_poll_contention_observer(
          session, contention_input_, this, &observe_input_poll_contention);
      if (!observed) {
        return observed;
      }
    }
    if (paused_activation_retirement_ordinal_ != 0U) {
      auto observed = detail::RuntimeChannelSessionTestAccess::set_activation_retirement_observer(
          session, this, &observe_activation_retirement);
      if (!observed) {
        return observed;
      }
    }
    if (paused_activation_retirement_ordinal_ == 0U && reject_reuse_replacement_) {
      auto armed = detail::RuntimeChannelSessionTestAccess::reject_next_reuse_replacement_receive(
          session, input_, ErrorCode::kBackpressured);
      if (!armed) {
        return armed;
      }
      armed_.store(true, std::memory_order_release);
    }
    session_.store(&session, std::memory_order_release);
    return {};
  }

  [[nodiscard]] bool armed() const noexcept { return armed_.load(std::memory_order_acquire); }

  void notification_posted() noexcept {
    notification_posted_.store(true, std::memory_order_release);
  }

  [[nodiscard]] std::uint32_t poll_contention_count() const noexcept {
    return poll_contention_count_.load(std::memory_order_acquire);
  }

  [[nodiscard]] bool poll_contention_identity_matches() const noexcept {
    if (!poll_contention_identity_recorded_.load(std::memory_order_acquire)) {
      return false;
    }
    return observed_owner_ == owner_ && observed_input_ == contention_input_ && observed_source_;
  }

  [[nodiscard]] bool wait_until_activation_retirement_paused() {
    std::unique_lock lock{retirement_mutex_};
    return retirement_cv_.wait_for(lock, 5s, [this] { return retirement_paused_; });
  }

  void release_activation_retirement() noexcept {
    const std::lock_guard lock{retirement_mutex_};
    retirement_released_ = true;
    retirement_cv_.notify_all();
  }

  [[nodiscard]] bool activation_retirement_identity_matches() const noexcept {
    const std::lock_guard lock{retirement_mutex_};
    return retirement_paused_ && retirement_owner_ == owner_ && retirement_generation_ != 0U &&
           retirement_accepted_;
  }

  [[nodiscard]] expected<std::optional<std::uint64_t>, Error> staged_data_sequence()
      const noexcept {
    detail::RuntimeChannelSession* const session = session_.load(std::memory_order_acquire);
    if (session == nullptr) {
      return make_unexpected(Error{ErrorCode::kNotReady});
    }
    return detail::RuntimeChannelSessionTestAccess::staged_data_sequence(*session, input_);
  }

  [[nodiscard]] expected<bool, Error> input_terminal_observed() const noexcept {
    detail::RuntimeChannelSession* const session = session_.load(std::memory_order_acquire);
    if (session == nullptr) {
      return make_unexpected(Error{ErrorCode::kNotReady});
    }
    return session->input_terminal_observed(input_);
  }

  [[nodiscard]] expected<void, Error> with_contention_input_lock(
      void* const state, const detail::RuntimeChannelInputLockObserver observer) const noexcept {
    detail::RuntimeChannelSession* const session = session_.load(std::memory_order_acquire);
    if (session == nullptr) {
      return make_unexpected(Error{ErrorCode::kNotReady});
    }
    return detail::RuntimeChannelSessionTestAccess::with_input_lock(
        *session, contention_input_, state, observer);
  }

 private:
  static expected<void, Error> observe_activation_retirement(void* const opaque,
                                                             const VertexId owner,
                                                             const std::uint64_t generation,
                                                             const bool accepted) noexcept {
    auto& self = *static_cast<RetryableRetainedTensorReplacementBindControl*>(opaque);
    std::unique_lock lock{self.retirement_mutex_};
    ++self.retirement_observation_count_;
    if (self.retirement_observation_count_ != self.paused_activation_retirement_ordinal_) {
      return {};
    }
    self.retirement_owner_ = owner;
    self.retirement_generation_ = generation;
    self.retirement_accepted_ = accepted;
    self.retirement_paused_ = true;
    self.retirement_cv_.notify_all();
    self.retirement_cv_.wait(lock, [&self] { return self.retirement_released_; });
    return {};
  }

  static void observe_input_poll_contention(void* const opaque, const VertexId owner,
                                            const PortId input, const SourceId source) noexcept {
    auto& self = *static_cast<RetryableRetainedTensorReplacementBindControl*>(opaque);
    if (!self.notification_posted_.load(std::memory_order_acquire)) {
      return;
    }
    bool expected = false;
    if (self.poll_contention_claimed_.compare_exchange_strong(
            expected, true, std::memory_order_acq_rel, std::memory_order_acquire)) {
      self.observed_owner_ = owner;
      self.observed_input_ = input;
      self.observed_source_ = source;
      self.poll_contention_identity_recorded_.store(true, std::memory_order_release);
    }
    self.poll_contention_count_.fetch_add(1U, std::memory_order_release);
  }

  PortId input_{};
  PortId contention_input_{};
  VertexId owner_{};
  std::atomic<detail::RuntimeChannelSession*> session_{};
  std::atomic<bool> armed_{};
  std::atomic<bool> notification_posted_{};
  std::atomic<bool> poll_contention_claimed_{};
  std::atomic<bool> poll_contention_identity_recorded_{};
  std::atomic<std::uint32_t> poll_contention_count_{};
  VertexId observed_owner_{};
  PortId observed_input_{};
  SourceId observed_source_{};
  std::uint32_t paused_activation_retirement_ordinal_{};
  bool reject_reuse_replacement_{};
  mutable std::mutex retirement_mutex_;
  std::condition_variable retirement_cv_;
  std::uint32_t retirement_observation_count_{};
  VertexId retirement_owner_{};
  std::uint64_t retirement_generation_{};
  bool retirement_accepted_{};
  bool retirement_paused_{};
  bool retirement_released_{};
};

enum class IndependentSideInputTriggerRoute : std::uint8_t {
  kClock,
  kNotified,
};

// Coordinates one independently triggered Operator with a catalog Channel side input. The test
// publishes data and EOS before the next independent witness, then verifies that only the clock or
// notification source can authorize compute.
struct IndependentSideInputProbe {
  static void reset() noexcept {
    const std::lock_guard lock{mutex};
    sender.reset();
    compute_count.store(0U, std::memory_order_relaxed);
    value.store(0U, std::memory_order_relaxed);
    sequence.store(0U, std::memory_order_relaxed);
  }

  static void publish_sender(NotificationSender notification_sender) {
    {
      const std::lock_guard lock{mutex};
      sender = std::move(notification_sender);
    }
    ready.notify_all();
  }

  static void record(const Sample<std::uint32_t>& sample) noexcept {
    value.store(sample.data, std::memory_order_relaxed);
    sequence.store(sample.metadata.sequence, std::memory_order_relaxed);
    compute_count.fetch_add(1U, std::memory_order_release);
    ready.notify_all();
  }

  [[nodiscard]] static bool wait_for_sender() {
    std::unique_lock lock{mutex};
    return ready.wait_for(lock, 5s, [] { return sender.has_value(); });
  }

  [[nodiscard]] static NotificationSender sender_copy() {
    const std::lock_guard lock{mutex};
    return sender.value_or(NotificationSender{});
  }

  [[nodiscard]] static bool wait_for_compute() {
    std::unique_lock lock{mutex};
    return ready.wait_for(
        lock, 5s, [] { return compute_count.load(std::memory_order_acquire) != 0U; });
  }

  static inline std::mutex mutex;
  static inline std::condition_variable ready;
  static inline std::optional<NotificationSender> sender;
  static inline std::atomic<std::size_t> compute_count{};
  static inline std::atomic<std::uint32_t> value{};
  static inline std::atomic<std::uint64_t> sequence{};
};

class IndependentSideInputSink final : public Operator<> {
 public:
  explicit IndependentSideInputSink(IndependentSideInputTriggerRoute route) : route_(route) {}

  void setup(OperatorSpec& spec) override {
    spec.input(input, "input").queue_depth(2U);
    if (route_ == IndependentSideInputTriggerRoute::kNotified) {
      spec.notification_source(notification, "notification")
          .capacity(1U)
          .sender_reference_capacity(1U);
      spec.lifecycle().stage(LifecycleStage::kArm, &IndependentSideInputSink::arm);
    }
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    if (route_ == IndependentSideInputTriggerRoute::kClock) {
      result.trigger(OnClock{.period = 2s});
    } else {
      result.trigger(OnNotified{.event = notification});
    }
    result.input(input, LatestWithin{.max_age = 30s}).on_missing(input, Skip{});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    auto sample = input.receive();
    if (!sample) {
      return make_unexpected(std::move(sample.error()));
    }
    IndependentSideInputProbe::record(*sample);
    return {};
  }

  LifecycleStatus arm(LifecycleContext& context) noexcept {
    auto sender = context.notification_sender(notification);
    if (!sender) {
      return LifecycleStatus::kFatalFailure;
    }
    IndependentSideInputProbe::publish_sender(std::move(*sender));
    return LifecycleStatus::kOk;
  }

  Input<std::uint32_t> input;
  NotificationSource notification;

 private:
  IndependentSideInputTriggerRoute route_;
};

// Coordinates an independently posted anchor with an externally closed bracketed input. The
// source count lets the test distinguish accepted notification callbacks from downstream window
// activations.
struct PostEndOfStreamWindowProbe {
  static void reset() noexcept {
    const std::lock_guard lock{mutex};
    sender.reset();
    anchor_count.store(0U, std::memory_order_relaxed);
    window_count.store(0U, std::memory_order_relaxed);
  }

  static void publish_sender(NotificationSender notification_sender) {
    {
      const std::lock_guard lock{mutex};
      sender = std::move(notification_sender);
    }
    ready.notify_all();
  }

  [[nodiscard]] static NotificationSender sender_copy() {
    const std::lock_guard lock{mutex};
    return sender.value_or(NotificationSender{});
  }

  static void record_anchor() noexcept {
    anchor_count.fetch_add(1U, std::memory_order_release);
    ready.notify_all();
  }

  static void record_window() noexcept {
    window_count.fetch_add(1U, std::memory_order_release);
    ready.notify_all();
  }

  [[nodiscard]] static bool wait_for_sender() {
    std::unique_lock lock{mutex};
    return ready.wait_for(lock, 5s, [] { return sender.has_value(); });
  }

  [[nodiscard]] static bool wait_for_anchor_count(std::size_t expected) {
    std::unique_lock lock{mutex};
    return ready.wait_for(
        lock, 5s, [expected] { return anchor_count.load(std::memory_order_acquire) >= expected; });
  }

  [[nodiscard]] static bool wait_for_window_count(std::size_t expected,
                                                  std::chrono::milliseconds timeout) {
    std::unique_lock lock{mutex};
    return ready.wait_for(lock, timeout, [expected] {
      return window_count.load(std::memory_order_acquire) >= expected;
    });
  }

  static inline std::mutex mutex;
  static inline std::condition_variable ready;
  static inline std::optional<NotificationSender> sender;
  static inline std::atomic<std::size_t> anchor_count{};
  static inline std::atomic<std::size_t> window_count{};
};

class PostEndOfStreamAnchorSource final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.output(anchor, "anchor");
    spec.notification_source(notification, "notification")
        .capacity(1U)
        .sender_reference_capacity(1U);
    spec.lifecycle().stage(LifecycleStage::kArm, &PostEndOfStreamAnchorSource::arm);
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnNotified{.event = notification});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    const TimePoint activation_time = context.activation_time();
    auto emitted =
        anchor.emit(next_anchor_,
                    EmitOptions{.capture_time = TimePoint{
                                    .timestamp_ns = activation_time.timestamp_ns + next_anchor_,
                                    .clock_id = activation_time.clock_id,
                                }});
    if (!emitted) {
      return emitted;
    }
    ++next_anchor_;
    PostEndOfStreamWindowProbe::record_anchor();
    return {};
  }

  LifecycleStatus arm(LifecycleContext& context) noexcept {
    auto sender = context.notification_sender(notification);
    if (!sender) {
      return LifecycleStatus::kFatalFailure;
    }
    PostEndOfStreamWindowProbe::publish_sender(std::move(*sender));
    return LifecycleStatus::kOk;
  }

  Output<std::uint32_t> anchor;
  NotificationSource notification;

 private:
  std::uint32_t next_anchor_{1U};
};

class PostEndOfStreamWindowSink final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(anchor, "anchor").queue_depth(2U);
    spec.input(windowed, "windowed").queue_depth(8U);
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnWindow{.anchor = anchor, .max_wait = 20ms, .max_pending_windows = 1U})
        .input(anchor, Exact{})
        .input(windowed, BracketedWindow{.lookbehind = 1U, .lookahead = 1U, .max_samples = 3U})
        .on_missing(windowed, SafeDefault{});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    auto received = anchor.receive();
    if (!received) {
      return make_unexpected(std::move(received.error()));
    }
    PostEndOfStreamWindowProbe::record_window();
    return {};
  }

  Input<std::uint32_t> anchor;
  Input<std::uint32_t> windowed;
};

class EndOfStreamWindowDeadlineSink final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(anchor, "anchor").queue_depth(2U);
    spec.input(windowed, "windowed").queue_depth(8U);
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnWindow{.anchor = anchor, .max_wait = 5s, .max_pending_windows = 1U})
        .input(anchor, Exact{})
        .input(windowed, BracketedWindow{.lookbehind = 1U, .lookahead = 1U, .max_samples = 3U})
        .on_missing(windowed, SafeDefault{})
        .max_start_delay(1ns)
        .on_violation(OnViolation::kFailRun);
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Input<std::uint32_t> anchor;
  Input<std::uint32_t> windowed;
};

/// @brief Retain one independently sourced aligned activation without changing sample metadata.
struct IndependentAlignedImportObservation {
  SampleMetadata first_metadata;
  SampleMetadata second_metadata;
  std::uint32_t first_value{};
  std::uint32_t second_value{};
  TagTime tag_time{};
  TimePoint activation_time{};
};

/// @brief Share the bounded result of one independently sourced aligned activation.
struct IndependentAlignedImportProbe {
  static void reset() noexcept {
    const std::lock_guard lock{mutex};
    observation.reset();
  }

  static void record(const Sample<std::uint32_t>& first, const Sample<std::uint32_t>& second,
                     TagTime tag_time, TimePoint activation_time) noexcept {
    {
      const std::lock_guard lock{mutex};
      observation = IndependentAlignedImportObservation{
          .first_metadata = first.metadata,
          .second_metadata = second.metadata,
          .first_value = first.data,
          .second_value = second.data,
          .tag_time = tag_time,
          .activation_time = activation_time,
      };
    }
    ready.notify_all();
  }

  [[nodiscard]] static std::optional<IndependentAlignedImportObservation> wait_for_observation(
      std::chrono::milliseconds timeout = 5s) {
    std::unique_lock lock{mutex};
    if (!ready.wait_for(lock, timeout, [] { return observation.has_value(); })) {
      return std::nullopt;
    }
    return observation;
  }

  [[nodiscard]] static bool observed() noexcept {
    const std::lock_guard lock{mutex};
    return observation.has_value();
  }

  static inline std::mutex mutex;
  static inline std::condition_variable ready;
  static inline std::optional<IndependentAlignedImportObservation> observation;
};

/// @brief Join two imported streams whose capture clocks normalize to one comparison clock.
class IndependentAlignedImportSink final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(first, "first").queue_depth(4U);
    spec.input(second, "second").queue_depth(4U);
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result
        .trigger(OnAligned{
            .ports = ports(first, second),
            .window = 100ns,
        })
        .input(first, Exact{})
        .input(second, Exact{});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    auto first_sample = first.receive();
    if (!first_sample) {
      return make_unexpected(std::move(first_sample).error());
    }
    auto second_sample = second.receive();
    if (!second_sample) {
      return make_unexpected(std::move(second_sample).error());
    }
    IndependentAlignedImportProbe::record(
        *first_sample, *second_sample, context.tag_time(), context.activation_time());
    return {};
  }

  Input<std::uint32_t> first;
  Input<std::uint32_t> second;
};

// Coordinates two independently monitored imports whose sinks execute on distinct partition
// owners. Each callback waits at the same bounded barrier so the test proves real overlap rather
// than merely observing two different owner thread identities at different times.
struct ConcurrentMonitoredImportProbe {
  struct Snapshot {
    std::array<std::uint32_t, 2U> values{};
    std::array<std::uint64_t, 2U> sequences{};
    std::array<std::thread::id, 2U> owner_threads{};
    std::size_t entered{};
    std::size_t completed{};
    bool barrier_timed_out{};
  };

  static void reset() noexcept {
    const std::lock_guard lock{mutex};
    values.fill(0U);
    sequences.fill(0U);
    owner_threads.fill(std::thread::id{});
    entered = 0U;
    completed = 0U;
    barrier_timed_out = false;
  }

  [[nodiscard]] static bool record(std::size_t index,
                                   const Sample<std::uint32_t>& sample) noexcept {
    std::unique_lock lock{mutex};
    values[index] = sample.data;
    sequences[index] = sample.metadata.sequence;
    owner_threads[index] = std::this_thread::get_id();
    ++entered;
    ready.notify_all();
    if (!ready.wait_for(lock, 5s, [] { return entered == 2U; })) {
      barrier_timed_out = true;
      ready.notify_all();
      return false;
    }
    ++completed;
    ready.notify_all();
    return true;
  }

  [[nodiscard]] static bool wait_for_completion(std::chrono::milliseconds timeout = 5s) {
    std::unique_lock lock{mutex};
    return ready.wait_for(lock, timeout, [] { return completed == 2U || barrier_timed_out; });
  }

  [[nodiscard]] static Snapshot snapshot() noexcept {
    const std::lock_guard lock{mutex};
    return Snapshot{
        .values = values,
        .sequences = sequences,
        .owner_threads = owner_threads,
        .entered = entered,
        .completed = completed,
        .barrier_timed_out = barrier_timed_out,
    };
  }

  static inline std::mutex mutex;
  static inline std::condition_variable ready;
  static inline std::array<std::uint32_t, 2U> values{};
  static inline std::array<std::uint64_t, 2U> sequences{};
  static inline std::array<std::thread::id, 2U> owner_threads{};
  static inline std::size_t entered{};
  static inline std::size_t completed{};
  static inline bool barrier_timed_out{};
};

class ConcurrentMonitoredImportSink final : public Operator<> {
 public:
  explicit ConcurrentMonitoredImportSink(std::size_t index) : index_(index) {}

  void setup(OperatorSpec& spec) override { spec.input(input, "input").queue_depth(2U); }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnAnyInput{})
        .input(input, LatestWithin{.max_age = 5s})
        .on_missing(input, Skip{});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    auto sample = input.receive();
    if (!sample) {
      return make_unexpected(std::move(sample.error()));
    }
    if (!ConcurrentMonitoredImportProbe::record(index_, *sample)) {
      return make_unexpected(Error{ErrorCode::kTimeout});
    }
    return {};
  }

  Input<std::uint32_t> input;

 private:
  std::size_t index_{};
};

using WrongExternalImagePayload = schema::DescriptorFlatBufferPayloadFor<schema::TensorPacket>;

struct ExternalImageDeadlineProbe {
  static void reset() noexcept {
    const std::lock_guard lock{mutex};
    count.store(0U, std::memory_order_release);
    activation_reasons.fill(ActivationReason::kInvariantViolated);
    input_states.fill(InputState::kAbsent);
    sequences.fill(0U);
    values.fill(0U);
  }

  static void record(const ActivationStatus activation, const InputState input_state,
                     const Sample<schema::ImageT>& sample) noexcept {
    const std::lock_guard lock{mutex};
    const std::size_t index = count.load(std::memory_order_relaxed);
    if (index < activation_reasons.size()) {
      activation_reasons[index] = activation.reason;
      input_states[index] = input_state;
      sequences[index] = sample.metadata.sequence;
      const auto* bytes = sample.data.data ? sample.data.data->data_as<std::uint8_t>() : nullptr;
      values[index] = bytes == nullptr ? 0U : bytes[0];
    }
    count.store(index + 1U, std::memory_order_release);
    ready.notify_all();
  }

  [[nodiscard]] static bool wait_for_count(const std::size_t expected,
                                           std::chrono::milliseconds timeout = 5s) {
    std::unique_lock lock{mutex};
    return ready.wait_for(
        lock, timeout, [expected] { return count.load(std::memory_order_acquire) >= expected; });
  }

  static inline std::mutex mutex;
  static inline std::condition_variable ready;
  static inline std::atomic<std::size_t> count{};
  static inline std::array<ActivationReason, 2U> activation_reasons{};
  static inline std::array<InputState, 2U> input_states{};
  static inline std::array<std::uint64_t, 2U> sequences{};
  static inline std::array<std::uint8_t, 2U> values{};
};

class ExternalImageDeadlineSink final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(input, "input")
        .queue_depth(1U)
        .tensor_field(kExternalImageData)
        .expects_tensor(
            TensorInputSpec{.representation = {.memory_kind = MemoryKind::kHost,
                                               .dtype = kExternalTensorDtype,
                                               .rank = 2U,
                                               .strides = TensorStridePolicy::kContiguousOnly}});
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnAnyInput{})
        .input(input, LatestWithin{.max_age = 2h})
        .on_missing(input, ReuseLast{.max_age = 2h})
        .max_start_delay(30s)
        .on_violation(OnViolation::kEmitWitness);
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    const InputState input_state = input.selection().state;
    if (input_state != InputState::kSelected && input_state != InputState::kReused) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    auto sample = input.receive();
    if (!sample) {
      return make_unexpected(std::move(sample.error()));
    }
    ExternalImageDeadlineProbe::record(context.activation(), input_state, *sample);
    return {};
  }

  Input<schema::ImageT> input;
};

struct ControllerCudaDeviceObserver {
  static void observe(void* const state) noexcept {
    auto& self = *static_cast<ControllerCudaDeviceObserver*>(state);
    self.status = cudaGetDevice(&self.device);
    self.called = true;
  }

  int device{-1};
  cudaError_t status{cudaErrorUnknown};
  bool called{};
};

// Retains one imported Tensor through ReuseLast while a notification source remains the sole
// activation authority. Scenario-specific synchronization separately covers a retryable handoff,
// an incomplete freshness poll, an unpublished post-handoff prerequisite, and notification credit
// held through complete activation retirement.
struct RetainedTensorEndOfStreamProbe {
  static void reset() noexcept {
    const std::lock_guard lock{mutex};
    sender.reset();
    count.store(0U, std::memory_order_relaxed);
    data.fill(nullptr);
    sequences.fill(0U);
    values.fill(0U);
  }

  static void publish_sender(NotificationSender notification_sender) {
    {
      const std::lock_guard lock{mutex};
      sender = std::move(notification_sender);
    }
    ready.notify_all();
  }

  static void record(const Sample<Tensor>& sample) noexcept {
    const std::lock_guard lock{mutex};
    const std::size_t index = count.load(std::memory_order_relaxed);
    if (index < data.size()) {
      data[index] = sample.data.data();
      sequences[index] = sample.metadata.sequence;
      const auto* const bytes = sample.data.data_as<std::uint8_t>();
      values[index] = bytes == nullptr ? 0U : bytes[0];
    }
    count.store(index + 1U, std::memory_order_release);
    ready.notify_all();
  }

  [[nodiscard]] static bool wait_for_sender() {
    std::unique_lock lock{mutex};
    return ready.wait_for(lock, 5s, [] { return sender.has_value(); });
  }

  [[nodiscard]] static NotificationSender sender_copy() {
    const std::lock_guard lock{mutex};
    return sender.value_or(NotificationSender{});
  }

  [[nodiscard]] static bool wait_for_count(const std::size_t expected) {
    std::unique_lock lock{mutex};
    return ready.wait_for(
        lock, 5s, [expected] { return count.load(std::memory_order_acquire) >= expected; });
  }

  static inline std::mutex mutex;
  static inline std::condition_variable ready;
  static inline std::optional<NotificationSender> sender;
  static inline std::atomic<std::size_t> count{};
  static inline std::array<const void*, 3U> data{};
  static inline std::array<std::uint64_t, 3U> sequences{};
  static inline std::array<std::uint8_t, 3U> values{};
};

struct LockedSiblingInputNotificationPollProbe {
  static void observe(void* const opaque) noexcept {
    auto& self = *static_cast<LockedSiblingInputNotificationPollProbe*>(opaque);
    auto closed = self.publisher->close();
    self.publisher_closed = static_cast<bool>(closed);
    if (!closed) {
      self.close_error = closed.error().code;
      return;
    }
    const auto post_deadline = std::chrono::steady_clock::now() + 5s;
    while (std::chrono::steady_clock::now() < post_deadline) {
      auto posted = self.sender.post();
      if (posted) {
        self.notification_posted = true;
        break;
      }
      if (posted.error().code != ErrorCode::kBackpressured) {
        return;
      }
      std::this_thread::yield();
    }
    if (!self.notification_posted) {
      return;
    }
    self.control->notification_posted();
    const auto contention_deadline = std::chrono::steady_clock::now() + 5s;
    while (self.notification_posted && self.session->is_running() &&
           self.control->poll_contention_count() < 1U &&
           std::chrono::steady_clock::now() < contention_deadline) {
      std::this_thread::yield();
    }
    self.poll_contention_count = self.control->poll_contention_count();
    self.delivery_count_while_contended =
        RetainedTensorEndOfStreamProbe::count.load(std::memory_order_acquire);
  }

  NotificationSender sender;
  ChannelPublisher<Tensor>* publisher{};
  RunSession* session{};
  RetryableRetainedTensorReplacementBindControl* control{};
  ErrorCode close_error{ErrorCode::kSuccess};
  std::uint32_t delivery_count_while_contended{};
  bool publisher_closed{};
  bool notification_posted{};
  std::uint32_t poll_contention_count{};
};

class DormantContentionSource final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.output(output, "output").max_emits_per_compute(1U);
    spec.notification_source(notification, "notification").capacity(1U);
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnNotified{.event = notification});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Output<std::uint32_t> output;
  NotificationSource notification;
};

class RetainedTensorEndOfStreamSink final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(input, "input")
        .queue_depth(1U)
        .expects_tensor(TensorInputSpec{
            .representation = {.memory_kind = MemoryKind::kHost,
                               .dtype = kExternalTensorDtype,
                               .rank = 1U},
        });
    spec.input(contention, "contention").queue_depth(1U);
    spec.notification_source(notification, "notification")
        .capacity(1U)
        .sender_reference_capacity(1U);
    spec.lifecycle().stage(LifecycleStage::kArm, &RetainedTensorEndOfStreamSink::arm);
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnNotified{.event = notification})
        .input(input, LatestWithin{.max_age = 30s})
        .on_missing(input, ReuseLast{.max_age = 30s});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    auto sample = input.receive();
    if (!sample) {
      return make_unexpected(std::move(sample).error());
    }
    RetainedTensorEndOfStreamProbe::record(*sample);
    return {};
  }

  LifecycleStatus arm(LifecycleContext& context) noexcept {
    auto sender = context.notification_sender(notification);
    if (!sender) {
      return LifecycleStatus::kFatalFailure;
    }
    RetainedTensorEndOfStreamProbe::publish_sender(std::move(*sender));
    return LifecycleStatus::kOk;
  }

  Input<Tensor> input;
  Input<std::uint32_t> contention;
  NotificationSource notification;
};

[[nodiscard]] expected<void, Error> publish_external_device_tensor(
    ChannelPublisher<Tensor>& publisher, const ::cuda::stream_ref producer_stream,
    std::optional<TimePoint> capture_time) {
  auto loan = publisher.loan_tensor(
      TensorLoanRequest{.shape = kExternalTensorShape, .dtype = kExternalTensorDtype});
  if (!loan) {
    return make_unexpected(std::move(loan).error());
  }
  auto writer = loan->write(producer_stream);
  if (!writer) {
    return make_unexpected(std::move(writer).error());
  }
  if (cudaMemsetAsync(writer->data(), 0x5a, writer->byte_size(), producer_stream.get()) !=
      cudaSuccess) {
    return make_unexpected(Error{ErrorCode::kFailure});
  }
  auto committed = std::move(*writer).commit();
  if (!committed) {
    return make_unexpected(std::move(committed).error());
  }
  EmitOptions options;
  options.capture_time = capture_time;
  auto emitted = publisher.emit(std::move(*loan), options);
  return emitted ? expected<void, Error>{}
                 : expected<void, Error>{make_unexpected(std::move(emitted).error())};
}

// REQ: CHAN-122
// REQ: EXTERNAL-010
TEST(RuntimeExternalChannelTest,
     DescriptorTensorSchemaImportMaterializesCatalogPublisherBackingBeforeTypedExposure) {
  ExternalImageImportProbe::reset();
  const std::string endpoint_name = unique_endpoint_name();

  Graph graph{"runtime-external-image-import"};
  (void)graph.add_clock<ManualClock>("a-unrelated-clock", std::int64_t{0});
  const ClockKey capture_clock = graph.add_clock<ManualClock>("z-capture-clock", std::int64_t{0});
  const ClockKey runtime_clock = graph.add_clock<RealtimeClock>("m-runtime-clock");
  graph.set_default_clock(runtime_clock);
  constexpr std::int64_t kCaptureTimestampNs = 123456;
  const std::int64_t capture_offset = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                          std::chrono::steady_clock::now().time_since_epoch())
                                          .count() -
                                      kCaptureTimestampNs;
  graph.add_clock_conversion(
      capture_clock, runtime_clock, AffineClockConversion{.offset_ns = capture_offset});
  TopicOptions import_options = external_image_import_options(endpoint_name);
  import_options.capture_time = RequireCaptureTime{.clock = capture_clock};
  auto imported = graph.import_topic<schema::ImageT>(
      "holomq://" + endpoint_name, std::move(import_options), external_image_contract());
  const auto sink = graph.op<ExternalImageImportSink>("sink");
  graph.add_flow(imported.output(), sink->input, ConnectionOptions{.queue_depth = 1U});

  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  // OnEach is arrival-backed and therefore has no temporal input-selection record. Its valid
  // capture coordinate is still converted into the Runtime reference clock for tag ordering,
  // while the typed Sample metadata retains the original capture coordinate.
  EXPECT_TRUE(plan.input_selections().empty());
  ASSERT_EQ(plan.clock_conversions().size(), 1U);
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
  auto external_runtime = ChannelRuntime::create();
  ASSERT_TRUE(external_runtime) << runtime_error_message(external_runtime.error());
  auto wrong_role = external_runtime->create_tensor_schema_subscriber<ExternalImagePayload>(
      plan.endpoint_catalog(), endpoint_name);
  ASSERT_FALSE(wrong_role);
  EXPECT_EQ(wrong_role.error().code, ErrorCode::kInvalidArgument);
  EXPECT_EQ(wrong_role.error().message, "ENDPOINT_CLIENT_ROLE_MISMATCH");
  auto wrong_schema = external_runtime->create_tensor_schema_publisher<WrongExternalImagePayload>(
      plan.endpoint_catalog(), endpoint_name);
  ASSERT_FALSE(wrong_schema);
  EXPECT_EQ(wrong_schema.error().code, ErrorCode::kIncompatible);
  EXPECT_EQ(wrong_schema.error().message, "ENDPOINT_TYPE_MISMATCH");
  auto publisher = external_runtime->create_tensor_schema_publisher<ExternalImagePayload>(
      plan.endpoint_catalog(), endpoint_name);
  ASSERT_TRUE(publisher) << runtime_error_message(publisher.error()) << '\n'
                         << plan.endpoint_catalog().json();

  // Key setup: submit a verifier-valid prepared payload whose descriptor byte span disagrees
  // with both the Image Tensor view and the catalog's immutable eight-byte field contract.
  auto prepared_tensor = make_external_image_tensor(0x2bU);
  ASSERT_TRUE(prepared_tensor) << runtime_error_message(prepared_tensor.error());
  auto prepared_metadata = schema::build_image_metadata(
      *prepared_tensor,
      schema::ImageMetadataOptions{
          .height = 1U, .width = 8U, .encoding = schema::ImageEncoding_MONO8});
  ASSERT_TRUE(prepared_metadata) << runtime_error_message(prepared_metadata.error());
  const PayloadDescriptor oversized_descriptor{
      .payload_kind = ChannelPayloadKind::kLocalBufferDescriptor,
      .schema_id = schema_identity<schema::Tensor>().family_id,
      .sequence = 1U,
      .byte_size = 9U,
      .local =
          LocalPayloadReference{
              .process_id = 1U, .token = 1U, .generation = 1U, .expected_recipient_count = 1U},
  };
  auto prepared_payload = ExternalImagePayload::copy_from(*prepared_metadata, oversized_descriptor);
  ASSERT_TRUE(prepared_payload) << runtime_error_message(prepared_payload.error());
  auto prepared_capture = publisher->make_capture_time(kCaptureTimestampNs);
  ASSERT_TRUE(prepared_capture) << runtime_error_message(prepared_capture.error());
  auto rejected_prepared =
      publisher->emit_payload(*prepared_payload,
                              SampleMetadata{.capture_timestamp_ns = prepared_capture->timestamp_ns,
                                             .source_clock_id = prepared_capture->clock_id});

  // Key assertion: the immutable field contract rejects the prepared payload before finite
  // reservation or sequence authority; the ordinary valid sample below remains sequence one.
  ASSERT_FALSE(rejected_prepared);
  EXPECT_EQ(rejected_prepared.error().code, ErrorCode::kInvalidArgument);

  RunSession session = run_async(plan);
  ASSERT_TRUE(wait_for_schema_subscriber(*publisher));
  auto tensor = make_external_image_tensor(0x3cU);
  ASSERT_TRUE(tensor) << runtime_error_message(tensor.error());
  auto metadata = schema::build_image_metadata(
      *tensor,
      schema::ImageMetadataOptions{.height = 1U,
                                   .width = 8U,
                                   .encoding = schema::ImageEncoding_MONO8,
                                   .capture_timestamp_ns = 1200,
                                   .frame_id = "external-camera"});
  ASSERT_TRUE(metadata) << runtime_error_message(metadata.error());
  auto emitted = publisher->emit_captured(
      *metadata,
      std::move(*tensor),
      kCaptureTimestampNs,
      EmitOptions{
          .frame_id = 77U, .trace = TraceContext::root(88U), .flags = SampleFlags::kDegraded});
  ASSERT_TRUE(emitted) << runtime_error_message(emitted.error());
  EXPECT_EQ(*emitted, 1U);
  EXPECT_TRUE(ExternalImageImportProbe::wait_for_sample());

  session.request_stop();
  session.wait();
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
  // Key assertion: typed exposure preserves the materialized bytes and complete SampleMetadata
  // while translating only the capture clock into the plan-local identity.
  EXPECT_EQ(ExternalImageImportProbe::count.load(std::memory_order_acquire), 1U);
  EXPECT_EQ(ExternalImageImportProbe::height.load(std::memory_order_relaxed), 1U);
  EXPECT_EQ(ExternalImageImportProbe::width.load(std::memory_order_relaxed), 8U);
  EXPECT_EQ(ExternalImageImportProbe::first_byte.load(std::memory_order_relaxed), 0x3cU);
  EXPECT_EQ(ExternalImageImportProbe::capture_timestamp_ns.load(std::memory_order_relaxed),
            kCaptureTimestampNs);
  EXPECT_EQ(ExternalImageImportProbe::source_clock_id.load(std::memory_order_relaxed),
            plan_capture_clock->id.value);
  EXPECT_EQ(ExternalImageImportProbe::trace_id.load(std::memory_order_relaxed), 88U);
  EXPECT_EQ(ExternalImageImportProbe::frame_id.load(std::memory_order_relaxed), 77U);
  EXPECT_EQ(ExternalImageImportProbe::flags.load(std::memory_order_relaxed), 2U);
  EXPECT_EQ(ExternalImageImportProbe::sequence.load(std::memory_order_relaxed), 1U);
}

// Sample-driven deadlines include transport delay by starting from the imported capture instant.
// Ordered EOS remains a control transition without a producer capture coordinate and must not
// turn a retained ReuseLast sample into an extra callback.
// REQ: RUNTIME-007
// REQ: RUNTIME-034
// REQ: EXTERNAL-010
// REQ: CHAN-136
TEST(RuntimeExternalChannelTest,
     DescriptorTensorSchemaImportDeadlineUsesCaptureTimeAndTreatsEndOfStreamAsControl) {
  ExternalImageDeadlineProbe::reset();
  const std::string endpoint_name = unique_endpoint_name();

  Graph graph{"runtime-external-image-capture-deadline"};
  const ClockKey capture_clock = graph.add_clock<ManualClock>("capture-clock", std::int64_t{0});
  const ClockKey runtime_clock = graph.add_clock<RealtimeClock>(
      "runtime-clock", RealtimeClock::Config{.initial_time_offset = 7200.0});
  graph.set_default_clock(runtime_clock);
  constexpr std::int64_t kCaptureTimestampNs = 123456;
  constexpr std::int64_t kOneHourNs =
      std::chrono::duration_cast<std::chrono::nanoseconds>(1h).count();
  constexpr std::int64_t kCaptureOffset = kOneHourNs - kCaptureTimestampNs;
  graph.add_clock_conversion(
      capture_clock, runtime_clock, AffineClockConversion{.offset_ns = kCaptureOffset});
  TopicOptions options = external_image_import_options(endpoint_name);
  options.capture_time = RequireCaptureTime{.clock = capture_clock};
  options.clock_health = ClockHealthOptions{
      .capability = ClockHealthCapability::kStatusReports,
      .requirement = ClockHealthRequirement::kFreshHealthyReport,
      .max_report_age = 5s,
  };
  const auto imported = graph.import_topic<schema::ImageT>(
      "holomq://" + endpoint_name, std::move(options), external_image_contract());
  const auto sink = graph.op<ExternalImageDeadlineSink>("sink");
  graph.add_flow(imported.output(), sink->input, ConnectionOptions{.queue_depth = 1U});

  const ExecutionPlan compiled = compile(graph);
  ASSERT_TRUE(compiled.ok()) << compiled.json();
  ASSERT_EQ(compiled.max_start_delays().size(), 1U);
  EXPECT_EQ(compiled.max_start_delays().front().source_coordinate,
            MaxStartDelaySourceCoordinateKind::kTriggerSample);
  auto phase_counters = std::make_shared<detail::RuntimePhaseCounters>();
  const ExecutionPlan plan =
      detail::RuntimeTestPlanAccess::with_phase_counters(compiled, phase_counters);
  auto external_runtime = ChannelRuntime::create();
  ASSERT_TRUE(external_runtime) << runtime_error_message(external_runtime.error());
  auto publisher = external_runtime->create_tensor_schema_publisher<ExternalImagePayload>(
      plan.endpoint_catalog(), endpoint_name);
  ASSERT_TRUE(publisher) << runtime_error_message(publisher.error());

  RunSession session = run_async(plan);
  ASSERT_TRUE(wait_for_schema_subscriber(*publisher));
  const auto phase_count = [&](const detail::RuntimePhaseCounter phase) {
    return phase_counters->snapshot()[detail::RuntimePhaseCounters::index(phase)];
  };
  const auto wait_for_phase = [&](const detail::RuntimePhaseCounter phase,
                                  const std::uint64_t expected) {
    const auto timeout = std::chrono::steady_clock::now() + 5s;
    while (phase_count(phase) < expected && session.is_running() &&
           std::chrono::steady_clock::now() < timeout) {
      std::this_thread::yield();
    }
    return phase_count(phase) >= expected;
  };
  const std::uint64_t readiness =
      phase_count(detail::RuntimePhaseCounter::kChannelReadinessPublished);

  // Key setup: the affine conversion places both samples one hour before observed arrival. Wait
  // until N has populated the real ReuseLast cache, then publish final N+1 and ordered EOS. The
  // staged-candidate gate must compare in the plan reference namespace, not the catalog clock.
  ASSERT_TRUE(publisher->report_clock_health(ClockHealthReport::healthy()));
  auto tensor = make_external_image_tensor(0x3cU);
  ASSERT_TRUE(tensor) << runtime_error_message(tensor.error());
  auto metadata = schema::build_image_metadata(
      *tensor,
      schema::ImageMetadataOptions{
          .height = 1U, .width = 8U, .encoding = schema::ImageEncoding_MONO8});
  ASSERT_TRUE(metadata) << runtime_error_message(metadata.error());
  auto emitted = publisher->emit_captured(*metadata, std::move(*tensor), kCaptureTimestampNs);
  ASSERT_TRUE(emitted) << runtime_error_message(emitted.error());
  ASSERT_TRUE(ExternalImageDeadlineProbe::wait_for_count(1U));

  auto final_tensor = make_external_image_tensor(0x5dU);
  ASSERT_TRUE(final_tensor) << runtime_error_message(final_tensor.error());
  auto final_metadata = schema::build_image_metadata(
      *final_tensor,
      schema::ImageMetadataOptions{
          .height = 1U, .width = 8U, .encoding = schema::ImageEncoding_MONO8});
  ASSERT_TRUE(final_metadata) << runtime_error_message(final_metadata.error());
  auto final_emitted =
      publisher->emit_captured(*final_metadata, std::move(*final_tensor), kCaptureTimestampNs + 1);
  ASSERT_TRUE(final_emitted) << runtime_error_message(final_emitted.error());
  ASSERT_TRUE(publisher->close());
  ASSERT_TRUE(ExternalImageDeadlineProbe::wait_for_count(2U))
      << "evictions=" << phase_count(detail::RuntimePhaseCounter::kTemporalRetentionEvictions)
      << " readiness=" << phase_count(detail::RuntimePhaseCounter::kChannelReadinessPublished)
      << " running=" << session.is_running() << " diagnostics=" << session.diagnostics().size();
  ASSERT_TRUE(
      wait_for_phase(detail::RuntimePhaseCounter::kChannelReadinessPublished, readiness + 3U));
  session.wait();
  EXPECT_EQ(session.termination(), RunTermination::kCompleted);
  // Key assertion: the sample misses relative to capture, while EOS remains a control transition
  // that closes the input without dispatching an extra ReuseLast callback or deadline check.
  EXPECT_EQ(ExternalImageDeadlineProbe::count.load(std::memory_order_acquire), 2U);
  EXPECT_EQ(ExternalImageDeadlineProbe::activation_reasons,
            (std::array<ActivationReason, 2U>{ActivationReason::kMaxStartDelayExceeded,
                                              ActivationReason::kMaxStartDelayExceeded}));
  EXPECT_EQ(ExternalImageDeadlineProbe::input_states,
            (std::array<InputState, 2U>{InputState::kSelected, InputState::kSelected}));
  // The preceding ordered health report is operation one, so data N/N+1 are sequences two/three
  // and the protected terminal accepted by natural drain is sequence four.
  EXPECT_EQ(ExternalImageDeadlineProbe::sequences, (std::array<std::uint64_t, 2U>{2U, 3U}));
  EXPECT_EQ(ExternalImageDeadlineProbe::values, (std::array<std::uint8_t, 2U>{0x3cU, 0x5dU}));
  ASSERT_EQ(session.diagnostics().size(), 2U);
  EXPECT_EQ(session.diagnostics().front().token, "MAX_START_DELAY_MISS");
}

// Descriptor-backed catalog imports must preserve the exact capture-contract terminal diagnostic
// used by raw Tensor imports when transport metadata omits a promised capture coordinate.
// REQ: RUNTIME-034
TEST(RuntimeExternalChannelTest,
     MissingDescriptorTensorCaptureReportsExactTerminalDiagnosticBeforeTypedExposure) {
  ExternalImageImportProbe::reset();
  const std::string endpoint_name = unique_endpoint_name();

  Graph graph{"runtime-external-image-capture-contract"};
  const ClockKey capture_clock = graph.add_clock<ManualClock>("capture-clock", std::int64_t{0});
  graph.set_default_clock(capture_clock);
  TopicOptions options = external_image_import_options(endpoint_name);
  options.capture_time = RequireCaptureTime{.clock = capture_clock};
  const auto imported = graph.import_topic<schema::ImageT>(
      "holomq://" + endpoint_name, std::move(options), external_image_contract());
  const auto sink = graph.op<ExternalImageImportSink>("sink");
  graph.add_flow(imported.output(), sink->input, ConnectionOptions{.queue_depth = 1U});

  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto sink_vertex = std::find_if(plan.graph().vertices().begin(),
                                        plan.graph().vertices().end(),
                                        [](const auto& vertex) { return vertex.name == "sink"; });
  ASSERT_NE(sink_vertex, plan.graph().vertices().end());
  const auto endpoint = plan.endpoint_catalog().endpoint(endpoint_name);
  ASSERT_TRUE(endpoint);
  const auto* endpoint_record = detail::ChannelEndpointCatalogAccess::record(*endpoint);
  ASSERT_NE(endpoint_record, nullptr);
  // Key setup: create the external publisher from a deliberately weakened catalog copy that
  // omits the capture requirement while the immutable runtime plan still requires it.
  auto malformed_endpoint_record = *endpoint_record;
  malformed_endpoint_record.capture_time_clock.reset();
  malformed_endpoint_record.publisher_request.reset();
  auto malformed_catalog = detail::ChannelEndpointCatalogAccess::normalize(
      "runtime-external-image-missing-capture", {std::move(malformed_endpoint_record)});
  ASSERT_TRUE(malformed_catalog) << runtime_error_message(malformed_catalog.error());

  auto external_runtime = ChannelRuntime::create();
  ASSERT_TRUE(external_runtime) << runtime_error_message(external_runtime.error());
  auto publisher = external_runtime->create_tensor_schema_publisher<ExternalImagePayload>(
      *malformed_catalog, endpoint_name);
  ASSERT_TRUE(publisher) << runtime_error_message(publisher.error());
  RunSession session = run_async(plan);
  ASSERT_TRUE(wait_for_schema_subscriber(*publisher));

  auto tensor = make_external_image_tensor(0x3cU);
  ASSERT_TRUE(tensor) << runtime_error_message(tensor.error());
  auto metadata = schema::build_image_metadata(
      *tensor,
      schema::ImageMetadataOptions{
          .height = 1U, .width = 8U, .encoding = schema::ImageEncoding_MONO8});
  ASSERT_TRUE(metadata) << runtime_error_message(metadata.error());
  auto emitted = publisher->emit(*metadata, std::move(*tensor));
  ASSERT_TRUE(emitted) << runtime_error_message(emitted.error());

  const auto deadline = std::chrono::steady_clock::now() + 5s;
  while (session.is_running() && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(1ms);
  }
  if (session.is_running()) {
    session.request_stop();
    session.wait();
    FAIL() << "missing descriptor capture metadata did not fail the run within 5 seconds";
  }
  session.wait();

  // Key assertion: Runtime fails before typed exposure with the registered capture diagnostic
  // attributed to the consuming sink from the original plan.
  ASSERT_EQ(session.termination(), RunTermination::kFailed);
  ASSERT_FALSE(session.diagnostics().empty());
  const Diagnostic& diagnostic = session.diagnostics().front();
  EXPECT_EQ(diagnostic.token, "TEMPORAL_CAPTURE_TIME_REQUIRED");
  EXPECT_EQ(diagnostic.id, detail::diagnostic_id("TEMPORAL_CAPTURE_TIME_REQUIRED"));
  EXPECT_EQ(diagnostic.source.vertex, sink_vertex->id);
  EXPECT_EQ(ExternalImageImportProbe::count.load(std::memory_order_acquire), 0U);
}

// Monitored raw Tensor imports run on the controller rather than the background pump. Device
// BufferHandle materialization must therefore initialize the controller's CUDA import context and
// select the compiler-planned device before the catalog subscriber resolves the allocation.
// REQ: CHAN-103
// REQ: RUNTIME-017
TEST(RuntimeExternalChannelTest, MonitoredRawTensorImportSelectsPlannedCudaDevice) {
  test::skip_without_cuda_device();
  int device_ordinal = -1;
  ASSERT_EQ(cudaGetDevice(&device_ordinal), cudaSuccess);
  const std::string endpoint_name = unique_endpoint_name();

  Graph graph{"runtime-external-device-tensor-health"};
  const ClockKey capture_clock = graph.add_clock<ManualClock>("capture-clock", std::int64_t{0});
  const ClockKey runtime_clock = graph.add_clock<RealtimeClock>("runtime-clock");
  graph.set_default_clock(runtime_clock);
  const std::int64_t capture_offset = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                          std::chrono::steady_clock::now().time_since_epoch())
                                          .count();
  graph.add_clock_conversion(
      capture_clock, runtime_clock, AffineClockConversion{.offset_ns = capture_offset});
  TopicOptions options = optional_tensor_export_options(endpoint_name);
  options.bounds.subscriber.reset();
  options.bounds.publisher = PublisherEndpointBounds{.max_loaned_samples = 1U};
  options.capture_time = RequireCaptureTime{.clock = capture_clock};
  options.clock_health = ClockHealthOptions{
      .capability = ClockHealthCapability::kStatusReports,
      .requirement = ClockHealthRequirement::kFreshHealthyReport,
      .max_report_age = 5s,
  };
  ASSERT_TRUE(options.tensor);
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access): guarded by ASSERT_TRUE above.
  options.tensor->representation.memory_kind = MemoryKind::kCudaDevice;
  const auto imported = graph.import_topic<Tensor>("holomq://" + endpoint_name, std::move(options));
  const auto sink = graph.op<ExternalTensorExportSink>("sink");
  graph.add_flow(imported.output(), sink->input, ConnectionOptions{.queue_depth = 1U});

  // Key setup: compile a monitored raw Tensor import onto the current CUDA device through the
  // public endpoint binding. This import is controller-drained and intentionally absent from the
  // background-pump subscriber inventory.
  CompileOptions compile_options;
  compile_options.deployment.bind_imported_topic_device(ImportedTopicDevicePlacement{
      .endpoint_name = endpoint_name, .device = DeviceId{device_ordinal}});
  const ExecutionPlan plan = compile(graph, std::move(compile_options));
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto state = detail::ExecutionPlanAccess::retain(plan);
  ASSERT_NE(state, nullptr);
  const auto planned_input =
      std::find_if(state->ports.begin(), state->ports.end(), [](const PortPlanRecord& port) {
        return port.owner_kind == PortOwnerKind::kOperator &&
               port.direction == PortDirection::kInput && port.local_name == "input";
      });
  ASSERT_NE(planned_input, state->ports.end());
  ExternalTensorExportSink runtime_sink;
  OperatorSpec runtime_sink_spec;
  runtime_sink.setup(runtime_sink_spec);
  const std::array replay_ports{detail::RuntimeChannelPortReplayRecord{
      .port = planned_input->id,
      .owner = planned_input->owner,
      .direction = PortDirection::kInput,
      .declaration = static_cast<void*>(&runtime_sink.input),
      .type_ops = detail::runtime_port_type_ops<Tensor>(),
  }};
  auto runtime_exec = detail::GraphExec::reconstruct(state, RunGenerationId{1U});
  ASSERT_TRUE(runtime_exec) << runtime_error_message(runtime_exec.error());
  auto session = detail::RuntimeChannelSession::create(*state, replay_ports);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  auto health =
      detail::EndpointClockHealthRuntimeCoordinator::create(state->endpoint_clock_health,
                                                            state->clocks,
                                                            state->plan_generation,
                                                            RunGenerationId{1U},
                                                            (*runtime_exec)->clock_registry());
  ASSERT_TRUE(health) << (health ? std::string_view{} : health.error().token);
  std::mutex clock_health_serialization;
  ASSERT_TRUE((*session)->bind_clock_health(**health, clock_health_serialization));
  ControllerCudaDeviceObserver selected_device;
  ASSERT_TRUE(detail::RuntimeChannelSessionTestAccess::set_typed_ingress_observer(
      **session, planned_input->id, &selected_device, &ControllerCudaDeviceObserver::observe));

  auto external_runtime = ChannelRuntime::create();
  ASSERT_TRUE(external_runtime) << runtime_error_message(external_runtime.error());
  auto publisher =
      external_runtime->create_publisher<Tensor>(plan.endpoint_catalog(), endpoint_name);
  ASSERT_TRUE(publisher) << runtime_error_message(publisher.error());
  test::UniqueCudaStream producer_stream;
  ASSERT_EQ(test::create_non_blocking_cuda_stream(producer_stream), cudaSuccess);

  const auto connection_deadline = std::chrono::steady_clock::now() + 5s;
  std::optional<std::chrono::steady_clock::time_point> connected_since;
  while (std::chrono::steady_clock::now() < connection_deadline) {
    auto drained = detail::RuntimeChannelSessionTestAccess::drain_external_inputs(
        **session, planned_input->owner);
    ASSERT_TRUE(drained) << runtime_error_message(drained.error());
    auto refreshed = publisher->update_connections();
    ASSERT_TRUE(refreshed) << runtime_error_message(refreshed.error());
    if (publisher->has_subscribers()) {
      const auto now = std::chrono::steady_clock::now();
      connected_since = connected_since.value_or(now);
      if (now - connected_since.value_or(now) >= 20ms) {
        break;
      }
    } else {
      connected_since.reset();
    }
    std::this_thread::sleep_for(1ms);
  }
  ASSERT_TRUE(connected_since.has_value());
  auto reported = publisher->report_clock_health(ClockHealthReport::healthy());
  ASSERT_TRUE(reported) << runtime_error_message(reported.error());
  auto capture = publisher->make_capture_time(1);
  ASSERT_TRUE(capture) << runtime_error_message(capture.error());
  ASSERT_TRUE(publish_external_device_tensor(
      *publisher, ::cuda::stream_ref{producer_stream.get()}, *capture));

  const auto receive_deadline = std::chrono::steady_clock::now() + 5s;
  while (std::chrono::steady_clock::now() < receive_deadline) {
    ASSERT_TRUE(detail::RuntimeChannelSessionTestAccess::drain_external_inputs(
        **session, planned_input->owner));
    auto observed =
        detail::RuntimeChannelSessionTestAccess::input_reset_state(**session, planned_input->id);
    ASSERT_TRUE(observed) << runtime_error_message(observed.error());
    if (observed->occupied_samples == 1U) {
      break;
    }
    std::this_thread::sleep_for(1ms);
  }

  // Key assertion: the controller selected the planned device during BufferHandle import and
  // retained exactly one typed Tensor after the preceding health operation.
  EXPECT_TRUE(selected_device.called);
  EXPECT_EQ(selected_device.status, cudaSuccess);
  EXPECT_EQ(selected_device.device, device_ordinal);
  auto input_state =
      detail::RuntimeChannelSessionTestAccess::input_reset_state(**session, planned_input->id);
  ASSERT_TRUE(input_state) << runtime_error_message(input_state.error());
  EXPECT_EQ(input_state->occupied_samples, 1U);
  auto initialization_count =
      detail::RuntimeChannelSessionTestAccess::controller_descriptor_import_initialization_count(
          **session);
  ASSERT_TRUE(initialization_count) << runtime_error_message(initialization_count.error());
  EXPECT_EQ(*initialization_count, 1U);
}

// Counts imported descriptor-schema samples whose Image data field is device storage.
class ExternalImageDeviceSink final : public Operator<> {
 public:
  static void reset() noexcept { received.store(0U, std::memory_order_relaxed); }

  [[nodiscard]] static bool wait_for_sample(std::chrono::milliseconds timeout = 5s) {
    std::unique_lock lock{mutex};
    return ready.wait_for(
        lock, timeout, [] { return received.load(std::memory_order_acquire) != 0U; });
  }

  void setup(OperatorSpec& spec) override {
    spec.input(input, "input")
        .queue_depth(1U)
        .tensor_field(kExternalImageData)
        .expects_tensor(
            TensorInputSpec{.representation = {.memory_kind = MemoryKind::kCudaDevice,
                                               .dtype = kExternalTensorDtype,
                                               .rank = 2U,
                                               .strides = TensorStridePolicy::kContiguousOnly}});
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
    received.fetch_add(1U, std::memory_order_release);
    ready.notify_all();
    return {};
  }

  Input<schema::ImageT> input;
  static inline std::atomic<std::uint32_t> received{};
  static inline std::mutex mutex;
  static inline std::condition_variable ready;
};

// Counts imported raw Tensor samples that the consumer declares as CUDA device storage.
class ExternalCudaTensorSink final : public Operator<> {
 public:
  static void reset() noexcept { received.store(0U, std::memory_order_relaxed); }

  [[nodiscard]] static bool wait_for_sample(std::chrono::milliseconds timeout = 5s) {
    std::unique_lock lock{mutex};
    return ready.wait_for(
        lock, timeout, [] { return received.load(std::memory_order_acquire) != 0U; });
  }

  void setup(OperatorSpec& spec) override {
    spec.input(input, "input")
        .queue_depth(1U)
        .expects_tensor(
            TensorInputSpec{.representation = {.memory_kind = MemoryKind::kCudaDevice,
                                               .dtype = kExternalTensorDtype,
                                               .rank = 1U,
                                               .strides = TensorStridePolicy::kContiguousOnly}});
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
    received.fetch_add(1U, std::memory_order_release);
    ready.notify_all();
    return {};
  }

  Input<Tensor> input;
  static inline std::atomic<std::uint32_t> received{};
  static inline std::mutex mutex;
  static inline std::condition_variable ready;
};

// An imported CUDA Tensor topic compiled with only the public endpoint binding is delivered to a
// sink through a full run, with no internal compile evidence.
//
//   external publisher --CUDA Tensor--> [holomq://endpoint] --> sink
// REQ: EXTERNAL-010
TEST(RuntimeExternalChannelTest, CudaRawTensorImportBoundByEndpointDeliversToSink) {
  test::skip_without_cuda_device();
  int device_ordinal = -1;
  ASSERT_EQ(cudaGetDevice(&device_ordinal), cudaSuccess);
  ExternalCudaTensorSink::reset();
  const std::string endpoint_name = unique_endpoint_name();

  Graph graph{"runtime-external-cuda-tensor-delivery"};
  const ClockKey capture_clock = graph.add_clock<ManualClock>("capture-clock", std::int64_t{0});
  graph.set_default_clock(capture_clock);
  TopicOptions options = optional_tensor_export_options(endpoint_name);
  options.bounds.subscriber.reset();
  options.bounds.publisher = PublisherEndpointBounds{.max_loaned_samples = 1U};
  options.capture_time = RequireCaptureTime{.clock = capture_clock};
  ASSERT_TRUE(options.tensor);
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access): guarded by ASSERT_TRUE above.
  options.tensor->representation.memory_kind = MemoryKind::kCudaDevice;
  const auto imported = graph.import_topic<Tensor>("holomq://" + endpoint_name, std::move(options));
  const auto sink = graph.op<ExternalCudaTensorSink>("sink");
  graph.add_flow(imported.output(), sink->input, ConnectionOptions{.queue_depth = 1U});

  CompileOptions compile_options;
  compile_options.deployment.bind_imported_topic_device(ImportedTopicDevicePlacement{
      .endpoint_name = endpoint_name, .device = DeviceId{device_ordinal}});
  const ExecutionPlan plan = compile(graph, std::move(compile_options));
  ASSERT_TRUE(plan.ok()) << plan.json();
  auto external_runtime = ChannelRuntime::create();
  ASSERT_TRUE(external_runtime) << runtime_error_message(external_runtime.error());
  auto publisher =
      external_runtime->create_publisher<Tensor>(plan.endpoint_catalog(), endpoint_name);
  ASSERT_TRUE(publisher) << runtime_error_message(publisher.error());
  test::UniqueCudaStream producer_stream;
  ASSERT_EQ(test::create_non_blocking_cuda_stream(producer_stream), cudaSuccess);

  // Key setup: run the bound plan and publish one device Tensor from a producer stream.
  RunSession session = run_async(plan);
  ASSERT_TRUE(wait_for_subscriber(*publisher));
  auto capture = publisher->make_capture_time(1);
  ASSERT_TRUE(capture) << runtime_error_message(capture.error());
  ASSERT_TRUE(publish_external_device_tensor(
      *publisher, ::cuda::stream_ref{producer_stream.get()}, *capture));
  const bool delivered = ExternalCudaTensorSink::wait_for_sample();

  session.request_stop();
  session.wait();
  // Key assertion: the sink observed the sample and the session stopped cleanly.
  EXPECT_TRUE(delivered);
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
  EXPECT_EQ(ExternalCudaTensorSink::received.load(std::memory_order_acquire), 1U);
}

// An imported descriptor-schema topic whose Image data field is CUDA device storage, compiled
// with only the public endpoint binding, delivers to a sink through a full run.
//
//   external publisher --ImageT(CUDA data)--> [holomq://endpoint] --> sink
// REQ: EXTERNAL-010
TEST(RuntimeExternalChannelTest, CudaDescriptorSchemaImportBoundByEndpointDeliversToSink) {
  test::skip_without_cuda_device();
  int device_ordinal = -1;
  ASSERT_EQ(cudaGetDevice(&device_ordinal), cudaSuccess);
  ExternalImageDeviceSink::reset();
  const std::string endpoint_name = unique_endpoint_name();

  Graph graph{"runtime-external-cuda-image-delivery"};
  const ClockKey capture_clock = graph.add_clock<ManualClock>("capture-clock", std::int64_t{0});
  graph.set_default_clock(capture_clock);
  DescriptorTensorTopicContract<schema::ImageT> contract;
  contract.field(kExternalImageData,
                 TensorBoundaryContract{
                     .representation = {.memory_kind = MemoryKind::kCudaDevice,
                                        .dtype = kExternalTensorDtype,
                                        .rank = 2U,
                                        .strides = TensorStridePolicy::kContiguousOnly},
                     .bounds = tensor_bounds(kExternalImageShape),
                     .transfer = TensorTransferPolicy{.foreign_emit_policy =
                                                          InterfaceTensorEmitPolicy::kCopyIfNeeded},
                 });
  TopicOptions import_options = external_image_import_options(endpoint_name);
  import_options.capture_time = RequireCaptureTime{.clock = capture_clock};
  const auto imported = graph.import_topic<schema::ImageT>(
      "holomq://" + endpoint_name, std::move(import_options), std::move(contract));
  const auto sink = graph.op<ExternalImageDeviceSink>("sink");
  graph.add_flow(imported.output(), sink->input, ConnectionOptions{.queue_depth = 1U});

  CompileOptions compile_options;
  compile_options.deployment.bind_imported_topic_device(ImportedTopicDevicePlacement{
      .endpoint_name = endpoint_name, .device = DeviceId{device_ordinal}});
  const ExecutionPlan plan = compile(graph, std::move(compile_options));
  ASSERT_TRUE(plan.ok()) << plan.json();
  // The producer stream and source pool are declared before the publisher so they outlive the
  // Tensor it retains until the transport releases the sample.
  test::UniqueCudaStream producer_stream;
  ASSERT_EQ(test::create_non_blocking_cuda_stream(producer_stream), cudaSuccess);
  auto source_pool =
      std::make_shared<BlockMemoryPool>(MemoryStorageType::kDevice, 8U, 1U, device_ordinal);
  ASSERT_TRUE(source_pool->initialize_pool());
  auto external_runtime = ChannelRuntime::create();
  ASSERT_TRUE(external_runtime) << runtime_error_message(external_runtime.error());
  auto publisher = external_runtime->create_tensor_schema_publisher<ExternalImagePayload>(
      plan.endpoint_catalog(), endpoint_name);
  ASSERT_TRUE(publisher) << runtime_error_message(publisher.error());

  // Key setup: author a device-resident Image data Tensor ordered after its producer stream.
  auto tensor = make_tensor(kExternalImageShape,
                            kExternalTensorDtype,
                            std::nullopt,
                            MemoryStorageType::kDevice,
                            source_pool,
                            producer_stream.get());
  ASSERT_TRUE(tensor) << runtime_error_message(tensor.error());
  auto data = tensor->mutable_data();
  ASSERT_TRUE(data);
  ASSERT_EQ(cudaMemsetAsync(*data, 0x3c, 8U, producer_stream.get()), cudaSuccess);
  ASSERT_TRUE(tensor->set_readiness_stream(producer_stream.get()));
  auto metadata = schema::build_image_metadata(
      *tensor,
      schema::ImageMetadataOptions{
          .height = 1U, .width = 8U, .encoding = schema::ImageEncoding_MONO8});
  ASSERT_TRUE(metadata) << runtime_error_message(metadata.error());

  RunSession session = run_async(plan);
  ASSERT_TRUE(wait_for_schema_subscriber(*publisher));
  auto emitted = publisher->emit_captured(*metadata, std::move(*tensor), 1);
  ASSERT_TRUE(emitted) << runtime_error_message(emitted.error());
  const bool delivered = ExternalImageDeviceSink::wait_for_sample();

  session.request_stop();
  session.wait();
  // Key assertion: the schema field's device binding reaches a typed sink without diagnostics.
  EXPECT_TRUE(delivered);
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
  EXPECT_EQ(ExternalImageDeviceSink::received.load(std::memory_order_acquire), 1U);
}

// A CUDA descriptor-schema publisher stages foreign device Tensors into a bounded pool whose
// slots return only through transport returned-sample callbacks. When a consumer returns every
// staged sample after the last send reclaimed returns, the next emit finds the pool full; it must
// still reclaim those returns instead of staying backpressured forever. A full-size frame flood
// makes that interleaving routine without the fix.
// REQ: TENSOR-SCHEMA-005
TEST(RuntimeExternalChannelTest, CudaDescriptorSchemaImportKeepsDeliveringUnderEmitFlood) {
  test::skip_without_cuda_device();
  int device_ordinal = -1;
  ASSERT_EQ(cudaGetDevice(&device_ordinal), cudaSuccess);
  ExternalImageDeviceSink::reset();
  const std::string endpoint_name = unique_endpoint_name();
  constexpr std::array<std::int64_t, 2U> kFrameShape{480, 1920};
  constexpr std::size_t kFrameBytes = 480U * 1920U;
  constexpr std::uint32_t kRequiredFrames = 100U;

  Graph graph{"runtime-external-cuda-image-flood"};
  const ClockKey capture_clock = graph.add_clock<ManualClock>("capture-clock", std::int64_t{0});
  graph.set_default_clock(capture_clock);
  DescriptorTensorTopicContract<schema::ImageT> contract;
  contract.field(kExternalImageData,
                 TensorBoundaryContract{
                     .representation = {.memory_kind = MemoryKind::kCudaDevice,
                                        .dtype = kExternalTensorDtype,
                                        .rank = 2U,
                                        .strides = TensorStridePolicy::kContiguousOnly},
                     .bounds = tensor_bounds(kFrameShape),
                     .transfer = TensorTransferPolicy{.foreign_emit_policy =
                                                          InterfaceTensorEmitPolicy::kCopyIfNeeded},
                 });
  TopicOptions import_options = external_image_import_options(endpoint_name);
  import_options.capture_time = RequireCaptureTime{.clock = capture_clock};
  const auto imported = graph.import_topic<schema::ImageT>(
      "holomq://" + endpoint_name, std::move(import_options), std::move(contract));
  const auto sink = graph.op<ExternalImageDeviceSink>("sink");
  graph.add_flow(imported.output(), sink->input, ConnectionOptions{.queue_depth = 1U});

  CompileOptions compile_options;
  compile_options.deployment.bind_imported_topic_device(ImportedTopicDevicePlacement{
      .endpoint_name = endpoint_name, .device = DeviceId{device_ordinal}});
  const ExecutionPlan plan = compile(graph, std::move(compile_options));
  ASSERT_TRUE(plan.ok()) << plan.json();
  test::UniqueCudaStream producer_stream;
  ASSERT_EQ(test::create_non_blocking_cuda_stream(producer_stream), cudaSuccess);
  // Refused and retired Tensors return their blocks, so this bounds only samples in flight.
  auto source_pool = std::make_shared<BlockMemoryPool>(
      MemoryStorageType::kDevice, kFrameBytes, 16U, device_ordinal);
  ASSERT_TRUE(source_pool->initialize_pool());
  auto external_runtime = ChannelRuntime::create();
  ASSERT_TRUE(external_runtime) << runtime_error_message(external_runtime.error());
  auto publisher = external_runtime->create_tensor_schema_publisher<ExternalImagePayload>(
      plan.endpoint_catalog(), endpoint_name);
  ASSERT_TRUE(publisher) << runtime_error_message(publisher.error());

  std::int64_t capture_timestamp_ns = 0;
  const auto emit = [&]() -> expected<std::uint32_t, Error> {
    auto tensor = make_tensor(kFrameShape,
                              kExternalTensorDtype,
                              std::nullopt,
                              MemoryStorageType::kDevice,
                              source_pool,
                              producer_stream.get());
    if (!tensor) {
      return make_unexpected(std::move(tensor).error());
    }
    auto data = tensor->mutable_data();
    if (!data || cudaMemsetAsync(*data, 0x5a, kFrameBytes, producer_stream.get()) != cudaSuccess ||
        !tensor->set_readiness_stream(producer_stream.get())) {
      return make_unexpected(Error{ErrorCode::kFailure});
    }
    auto metadata = schema::build_image_metadata(
        *tensor,
        schema::ImageMetadataOptions{
            .height = 480U, .width = 1920U, .encoding = schema::ImageEncoding_MONO8});
    if (!metadata) {
      return make_unexpected(std::move(metadata).error());
    }
    return publisher->emit_captured(*metadata, std::move(*tensor), ++capture_timestamp_ns);
  };

  RunSession session = run_async(plan);
  ASSERT_TRUE(wait_for_schema_subscriber(*publisher));

  // Key setup: emit as fast as staging accepts. Count delivery progress from the start, or from
  // the first refused emit once one occurs: without the fix, that refusal is where delivery stops
  // for good.
  std::uint32_t last_seen = 0U;
  auto last_progress = std::chrono::steady_clock::now();
  bool stalled = false;
  std::optional<std::uint32_t> delivered_at_first_refusal;
  const auto delivered_progress = [&] {
    return delivered_at_first_refusal
               ? ExternalImageDeviceSink::received.load(std::memory_order_acquire) -
                     *delivered_at_first_refusal
               : ExternalImageDeviceSink::received.load(std::memory_order_acquire);
  };
  const auto deadline = std::chrono::steady_clock::now() + 30s;
  while (delivered_progress() < kRequiredFrames && std::chrono::steady_clock::now() < deadline) {
    const auto now = std::chrono::steady_clock::now();
    if (const auto seen = ExternalImageDeviceSink::received.load(std::memory_order_acquire);
        seen != last_seen) {
      last_seen = seen;
      last_progress = now;
    } else if (now - last_progress > 5s) {
      stalled = true;
      break;
    }
    auto emitted = emit();
    if (!emitted) {
      ASSERT_EQ(emitted.error().code, ErrorCode::kBackpressured)
          << runtime_error_message(emitted.error());
      if (!delivered_at_first_refusal) {
        delivered_at_first_refusal = ExternalImageDeviceSink::received.load();
      }
      std::this_thread::sleep_for(1ms);
    }
  }
  const std::uint32_t delivered = ExternalImageDeviceSink::received.load();
  const std::uint32_t recovered = delivered_progress();

  session.request_stop();
  session.wait();
  // Key assertion: delivery keeps progressing, including after any refused emit; before the fix
  // it stopped at the staging-slot count while every later emit was refused.
  EXPECT_FALSE(stalled) << "sink stopped at " << delivered << " frames";
  EXPECT_GE(recovered, kRequiredFrames);
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
}

// A catalog Tensor import is drained by the Channel pump, unlike value-typed catalog imports.
// Missing promised capture metadata must retain the exact terminal reason across that support
// thread, fail the owning sink before callback, and leave typed storage untouched. Reusing the
// immutable plan in a fresh RunSession then proves valid captured ingress is unaffected.
// REQ: RUNTIME-034
TEST(RuntimeExternalChannelTest,
     MissingTensorCaptureFailsOwningRunBeforeRetentionAndFreshRunAcceptsCapturedSample) {
  ExternalTensorExportSink::reset();
  const std::string endpoint_name = unique_endpoint_name();

  Graph graph{"runtime-external-tensor-capture-contract"};
  const ClockKey capture_clock = graph.add_clock<ManualClock>("capture-clock", std::int64_t{0});
  graph.set_default_clock(capture_clock);
  TopicOptions options = optional_tensor_export_options(endpoint_name);
  options.bounds.subscriber.reset();
  options.bounds.publisher = PublisherEndpointBounds{.max_loaned_samples = 1U};
  options.capture_time = RequireCaptureTime{.clock = capture_clock};
  const auto imported = graph.import_topic<Tensor>("holomq://" + endpoint_name, std::move(options));
  const auto sink = graph.op<ExternalTensorExportSink>("sink");
  graph.add_flow(imported.output(), sink->input, ConnectionOptions{.queue_depth = 1U});

  const ExecutionPlan compiled = compile(graph);
  ASSERT_TRUE(compiled.ok()) << compiled.json();
  const auto sink_vertex = std::find_if(compiled.graph().vertices().begin(),
                                        compiled.graph().vertices().end(),
                                        [](const auto& vertex) { return vertex.name == "sink"; });
  ASSERT_NE(sink_vertex, compiled.graph().vertices().end());
  const auto endpoint = compiled.endpoint_catalog().endpoint(endpoint_name);
  ASSERT_TRUE(endpoint);
  const ClockId catalog_capture_clock =
      detail::ChannelEndpointCatalogAccess::capture_clock_id(*endpoint);
  ASSERT_TRUE(catalog_capture_clock.valid());
  const auto* endpoint_record = detail::ChannelEndpointCatalogAccess::record(*endpoint);
  ASSERT_NE(endpoint_record, nullptr);
  auto malformed_endpoint_record = *endpoint_record;
  malformed_endpoint_record.capture_time_clock.reset();
  malformed_endpoint_record.publisher_request.reset();
  auto malformed_catalog = detail::ChannelEndpointCatalogAccess::normalize(
      "runtime-external-tensor-missing-capture", {std::move(malformed_endpoint_record)});
  ASSERT_TRUE(malformed_catalog) << runtime_error_message(malformed_catalog.error());

  auto malformed_counters = std::make_shared<detail::RuntimePhaseCounters>();
  const ExecutionPlan malformed_plan =
      detail::RuntimeTestPlanAccess::with_phase_counters(compiled, malformed_counters);
  {
    auto external_runtime = ChannelRuntime::create();
    ASSERT_TRUE(external_runtime) << runtime_error_message(external_runtime.error());
    auto publisher = external_runtime->create_publisher<Tensor>(*malformed_catalog, endpoint_name);
    ASSERT_TRUE(publisher) << runtime_error_message(publisher.error());
    RunSession malformed = run_async(malformed_plan);
    ASSERT_TRUE(wait_for_subscriber(*publisher));
    const auto malformed_emitted = publish_external_host_tensor(*publisher, std::nullopt);
    ASSERT_TRUE(malformed_emitted)
        << (malformed_emitted ? "" : runtime_error_message(malformed_emitted.error()));
    const auto deadline = std::chrono::steady_clock::now() + 5s;
    while (malformed.is_running() && std::chrono::steady_clock::now() < deadline) {
      std::this_thread::sleep_for(1ms);
    }
    if (malformed.is_running()) {
      malformed.request_stop();
      malformed.wait();
      FAIL() << "malformed capture-time metadata did not fail the run within 5 seconds";
    }
    malformed.wait();

    ASSERT_EQ(malformed.termination(), RunTermination::kFailed);
    ASSERT_FALSE(malformed.diagnostics().empty());
    const Diagnostic& diagnostic = malformed.diagnostics().front();
    EXPECT_EQ(diagnostic.token, "TEMPORAL_CAPTURE_TIME_REQUIRED");
    EXPECT_EQ(diagnostic.id, detail::diagnostic_id("TEMPORAL_CAPTURE_TIME_REQUIRED"));
    EXPECT_EQ(diagnostic.source.vertex, sink_vertex->id);
    EXPECT_EQ(ExternalTensorExportSink::received.load(std::memory_order_acquire), 0U);
    const auto malformed_snapshot = malformed_counters->snapshot();
    EXPECT_EQ(malformed_snapshot[detail::RuntimePhaseCounters::index(
                  detail::RuntimePhaseCounter::kChannelSamplesReceived)],
              0U);
  }

  ExternalTensorExportSink::reset();
  {
    auto external_runtime = ChannelRuntime::create();
    ASSERT_TRUE(external_runtime) << runtime_error_message(external_runtime.error());
    auto publisher =
        external_runtime->create_publisher<Tensor>(compiled.endpoint_catalog(), endpoint_name);
    ASSERT_TRUE(publisher) << runtime_error_message(publisher.error());
    RunSession fresh = run_async(compiled);
    ASSERT_TRUE(wait_for_subscriber(*publisher));
    ASSERT_TRUE(publish_external_host_tensor(
        *publisher, TimePoint{.timestamp_ns = 42, .clock_id = catalog_capture_clock}));
    EXPECT_TRUE(ExternalTensorExportSink::wait_for_sample());
    fresh.request_stop();
    fresh.wait();
    EXPECT_EQ(fresh.termination(), RunTermination::kStopped);
    EXPECT_TRUE(fresh.diagnostics().empty())
        << (fresh.diagnostics().empty() ? "" : fresh.diagnostics().front().token);
    EXPECT_EQ(ExternalTensorExportSink::received.load(std::memory_order_acquire), 1U);
  }
}

// A catalog Tensor close is a validated transport control operation, not an endpoint failure.
// Runtime must preserve its exact sequence, publish one EOS readiness witness, and never expose
// the control record as another Tensor sample.
// REQ: CHAN-078
// REQ: EXTERNAL-003
// REQ: SYSTEM-006
TEST(RuntimeExternalChannelTest, RawTensorImportConsumesOrderedEndOfStreamWithoutFailingRun) {
  ExternalTensorExportSink::reset();
  const std::string endpoint_name = unique_endpoint_name();

  Graph graph{"runtime-external-tensor-end-of-stream"};
  const ClockKey capture_clock = graph.add_clock<ManualClock>("capture-clock", std::int64_t{0});
  graph.set_default_clock(capture_clock);
  TopicOptions options = optional_tensor_export_options(endpoint_name);
  options.bounds.subscriber.reset();
  options.bounds.publisher = PublisherEndpointBounds{.max_loaned_samples = 1U};
  options.capture_time = RequireCaptureTime{.clock = capture_clock};
  const auto imported = graph.import_topic<Tensor>("holomq://" + endpoint_name, std::move(options));
  const auto sink = graph.op<ExternalTensorExportSink>("sink");
  graph.add_flow(imported.output(), sink->input, ConnectionOptions{.queue_depth = 1U});

  const ExecutionPlan compiled = compile(graph);
  // Retain only EOS sequence evidence: unrelated idle wake/park checkpoints have no fixed
  // bound under CI load. A duplicate EOS must still fail the counter and overflow assertions.
  std::array<detail::RuntimePhaseTraceRecord, 1U> phase_trace{};
  auto phase_counters = std::make_shared<detail::RuntimePhaseCounters>(
      phase_trace, detail::RuntimePhaseTracePoint::kChannelEndOfStreamRetained);
  const ExecutionPlan plan =
      detail::RuntimeTestPlanAccess::with_phase_counters(compiled, phase_counters);
  ASSERT_TRUE(plan.ok()) << plan.json();

  auto external_runtime = ChannelRuntime::create();
  ASSERT_TRUE(external_runtime) << runtime_error_message(external_runtime.error());
  auto publisher =
      external_runtime->create_publisher<Tensor>(plan.endpoint_catalog(), endpoint_name);
  ASSERT_TRUE(publisher) << runtime_error_message(publisher.error());

  RunSession session = run_async(plan);
  ASSERT_TRUE(wait_for_subscriber(*publisher));
  const auto phase_count = [&](const detail::RuntimePhaseCounter phase) {
    return phase_counters->snapshot()[detail::RuntimePhaseCounters::index(phase)];
  };
  const std::uint64_t readiness_baseline =
      phase_count(detail::RuntimePhaseCounter::kChannelReadinessPublished);

  auto capture_time = publisher->make_capture_time(42);
  ASSERT_TRUE(capture_time) << runtime_error_message(capture_time.error());
  ASSERT_TRUE(publish_external_host_tensor(*publisher, *capture_time));
  ASSERT_TRUE(publisher->close());
  auto repeated_close = publisher->close();
  ASSERT_FALSE(repeated_close);
  EXPECT_EQ(repeated_close.error().code, ErrorCode::kClosed);

  ASSERT_TRUE(ExternalTensorExportSink::wait_for_sample());
  const auto eos_deadline = std::chrono::steady_clock::now() + 5s;
  while (phase_count(detail::RuntimePhaseCounter::kChannelReadinessPublished) <
             readiness_baseline + 2U &&
         session.is_running() && std::chrono::steady_clock::now() < eos_deadline) {
    std::this_thread::yield();
  }
  EXPECT_EQ(phase_count(detail::RuntimePhaseCounter::kChannelReadinessPublished),
            readiness_baseline + 2U);
  EXPECT_TRUE(session.is_running());
  EXPECT_EQ(ExternalTensorExportSink::received.load(std::memory_order_acquire), 1U);

  session.request_stop();
  session.wait();
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
  std::size_t end_of_stream_records = 0U;
  for (const detail::RuntimePhaseTraceRecord& record : phase_counters->trace_records()) {
    if (record.point != detail::RuntimePhaseTracePoint::kChannelEndOfStreamRetained) {
      continue;
    }
    ++end_of_stream_records;
    EXPECT_EQ(record.source_sequence, 2U);
    EXPECT_GT(record.source_sequence,
              ExternalTensorExportSink::last_sequence.load(std::memory_order_acquire));
  }
  EXPECT_EQ(end_of_stream_records, 1U);
  EXPECT_EQ(phase_count(detail::RuntimePhaseCounter::kChannelEndOfStreamRetained), 1U);
  EXPECT_EQ(phase_counters->trace_dropped(), 0U);
}

// EOS can close a pending window that already owns its anchor sample. The control record itself
// has no capture coordinate, but the admitted activation remains sample-driven and must retain the
// window's compiled deadline.
// REQ: RUNTIME-007
// REQ: RUNTIME-034
TEST(RuntimeExternalChannelTest, EndOfStreamClosedWindowRetainsCompiledDeadline) {
  const std::string anchor_endpoint = unique_endpoint_name();
  const std::string window_endpoint = unique_endpoint_name();

  Graph graph{"runtime-window-end-of-stream-deadline"};
  const ClockKey runtime_clock = graph.add_clock<RealtimeClock>("runtime-clock");
  graph.set_default_clock(runtime_clock);
  TopicOptions anchor_options;
  anchor_options.endpoint_name = anchor_endpoint;
  anchor_options.capture_time = RequireCaptureTime{.clock = runtime_clock};
  const auto imported_anchor =
      graph.import_topic<std::uint32_t>("holomq://" + anchor_endpoint, std::move(anchor_options));
  TopicOptions window_options;
  window_options.endpoint_name = window_endpoint;
  window_options.capture_time = RequireCaptureTime{.clock = runtime_clock};
  const auto imported_window =
      graph.import_topic<std::uint32_t>("holomq://" + window_endpoint, std::move(window_options));
  const auto sink = graph.op<EndOfStreamWindowDeadlineSink>("window-sink");
  graph.add_flow(imported_anchor.output(), sink->anchor, ConnectionOptions{.queue_depth = 2U});
  graph.add_flow(imported_window.output(), sink->windowed, ConnectionOptions{.queue_depth = 8U});

  const ExecutionPlan compiled = compile(graph);
  auto phase_counters = std::make_shared<detail::RuntimePhaseCounters>();
  const ExecutionPlan plan =
      detail::RuntimeTestPlanAccess::with_phase_counters(compiled, phase_counters);
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.max_start_delays().size(), 1U);
  auto external_runtime = ChannelRuntime::create();
  ASSERT_TRUE(external_runtime) << runtime_error_message(external_runtime.error());
  auto anchor_publisher =
      external_runtime->create_publisher<std::uint32_t>(plan.endpoint_catalog(), anchor_endpoint);
  ASSERT_TRUE(anchor_publisher) << runtime_error_message(anchor_publisher.error());
  auto window_publisher =
      external_runtime->create_publisher<std::uint32_t>(plan.endpoint_catalog(), window_endpoint);
  ASSERT_TRUE(window_publisher) << runtime_error_message(window_publisher.error());

  RunSession session = run_async(plan);
  ASSERT_TRUE(wait_for_subscriber(*anchor_publisher));
  ASSERT_TRUE(wait_for_subscriber(*window_publisher));
  const auto phase_count = [&](detail::RuntimePhaseCounter phase) {
    return phase_counters->snapshot()[detail::RuntimePhaseCounters::index(phase)];
  };
  const auto wait_for_phase = [&](detail::RuntimePhaseCounter phase, std::uint64_t expected) {
    const auto timeout = std::chrono::steady_clock::now() + 5s;
    while (phase_count(phase) < expected && session.is_running() &&
           std::chrono::steady_clock::now() < timeout) {
      std::this_thread::yield();
    }
    return phase_count(phase) >= expected;
  };
  // Key setup: two stale-capture anchors create one pending window. Window EOS closes the
  // bracket, then anchor EOS supplies the all-input terminal proof that admits the activation.
  std::uint64_t retired = phase_count(detail::RuntimePhaseCounter::kChannelWitnessRetired);
  ASSERT_TRUE(anchor_publisher->emit_captured(1U, 0));
  ASSERT_TRUE(wait_for_phase(detail::RuntimePhaseCounter::kChannelWitnessRetired, retired + 1U));
  retired = phase_count(detail::RuntimePhaseCounter::kChannelWitnessRetired);
  ASSERT_TRUE(anchor_publisher->emit_captured(2U, 1));
  ASSERT_TRUE(wait_for_phase(detail::RuntimePhaseCounter::kChannelWitnessRetired, retired + 1U));

  std::uint64_t readiness = phase_count(detail::RuntimePhaseCounter::kChannelReadinessPublished);
  ASSERT_TRUE(window_publisher->close());
  ASSERT_TRUE(
      wait_for_phase(detail::RuntimePhaseCounter::kChannelReadinessPublished, readiness + 1U));
  const std::uint64_t no_progress = phase_count(detail::RuntimePhaseCounter::kNoProgressScans);
  ASSERT_TRUE(wait_for_phase(detail::RuntimePhaseCounter::kNoProgressScans, no_progress + 1U));
  readiness = phase_count(detail::RuntimePhaseCounter::kChannelReadinessPublished);
  ASSERT_TRUE(anchor_publisher->close());
  ASSERT_TRUE(
      wait_for_phase(detail::RuntimePhaseCounter::kChannelReadinessPublished, readiness + 1U));
  session.wait();

  // Key assertion: EOS does not bypass the sample-relative kFailRun deadline of the selected
  // anchor that the window activation still carries.
  EXPECT_EQ(session.termination(), RunTermination::kFailed);
  ASSERT_EQ(session.diagnostics().size(), 2U);
  EXPECT_EQ(session.diagnostics()[0].token, "MAX_START_DELAY_MISS");
  EXPECT_EQ(session.diagnostics()[1].token, "RUNTIME_MAX_START_DELAY_FAIL_RUN");
}

// A bracketed input's EOS terminates its selector generation even when no window is pending. An
// independent anchor can continue producing afterwards, but RunSession must not forward it into
// the terminal bracketed selector and create a window whose EOS has already been consumed.
// REQ: CHAN-078
// REQ: RUNTIME-007
// REQ: CONTRACT-019
TEST(RuntimeExternalChannelTest, BracketedInputIgnoresAnchorsAfterEmptyEndOfStreamClosure) {
  PostEndOfStreamWindowProbe::reset();
  const std::string endpoint_name = unique_endpoint_name();

  Graph graph{"runtime-bracketed-input-post-end-of-stream-anchor"};
  const ClockKey runtime_clock = graph.add_clock<RealtimeClock>("runtime-clock");
  graph.set_default_clock(runtime_clock);
  TopicOptions options;
  options.endpoint_name = endpoint_name;
  options.capture_time = RequireCaptureTime{.clock = runtime_clock};
  const auto imported =
      graph.import_topic<std::uint32_t>("holomq://" + endpoint_name, std::move(options));
  const auto source = graph.op<PostEndOfStreamAnchorSource>("anchor-source");
  const auto sink = graph.op<PostEndOfStreamWindowSink>("window-sink");
  graph.add_flow(source->anchor, sink->anchor, ConnectionOptions{.queue_depth = 2U});
  graph.add_flow(imported.output(), sink->windowed, ConnectionOptions{.queue_depth = 8U});

  const ExecutionPlan compiled = compile(graph);
  auto phase_counters = std::make_shared<detail::RuntimePhaseCounters>();
  const ExecutionPlan plan =
      detail::RuntimeTestPlanAccess::with_phase_counters(compiled, phase_counters);
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.partitioning_plan().mode, PartitioningMode::kAutomatic);
  ASSERT_EQ(plan.partitioning_plan().automatic_partition_count, 3U);
  ASSERT_EQ(plan.execution_partitions().size(), 3U);
  ASSERT_EQ(plan.schedule_edges().size(), 2U);

  auto external_runtime = ChannelRuntime::create();
  ASSERT_TRUE(external_runtime) << runtime_error_message(external_runtime.error());
  auto publisher =
      external_runtime->create_publisher<std::uint32_t>(plan.endpoint_catalog(), endpoint_name);
  ASSERT_TRUE(publisher) << runtime_error_message(publisher.error());

  RunSession session = run_async(plan);
  ASSERT_TRUE(PostEndOfStreamWindowProbe::wait_for_sender());
  ASSERT_TRUE(wait_for_subscriber(*publisher));
  const auto phase_count = [&](detail::RuntimePhaseCounter phase) {
    return phase_counters->snapshot()[detail::RuntimePhaseCounters::index(phase)];
  };
  const auto wait_for_phase = [&](detail::RuntimePhaseCounter phase, std::uint64_t expected) {
    const auto deadline = std::chrono::steady_clock::now() + 5s;
    while (phase_count(phase) < expected && session.is_running() &&
           std::chrono::steady_clock::now() < deadline) {
      std::this_thread::yield();
    }
    return phase_count(phase) >= expected;
  };

  std::uint64_t retired = phase_count(detail::RuntimePhaseCounter::kChannelWitnessRetired);
  const NotificationSender sender = PostEndOfStreamWindowProbe::sender_copy();
  auto first_post = sender.post();
  const auto start_deadline = std::chrono::steady_clock::now() + 5s;
  // Arm publishes the sender before the global kStart barrier activates it. Automatic execution
  // keeps that contract across the three independent partition owners, so retry only kNotReady.
  while (!first_post && first_post.error().code == ErrorCode::kNotReady &&
         std::chrono::steady_clock::now() < start_deadline) {
    std::this_thread::yield();
    first_post = sender.post();
  }
  ASSERT_TRUE(first_post) << runtime_error_message(first_post.error());
  ASSERT_TRUE(PostEndOfStreamWindowProbe::wait_for_anchor_count(1U));
  ASSERT_TRUE(wait_for_phase(detail::RuntimePhaseCounter::kChannelWitnessRetired, retired + 1U));

  retired = phase_count(detail::RuntimePhaseCounter::kChannelWitnessRetired);
  const auto capture_time = std::chrono::system_clock::now().time_since_epoch();
  auto emitted = publisher->emit_captured(
      7U, std::chrono::duration_cast<std::chrono::nanoseconds>(capture_time).count());
  ASSERT_TRUE(emitted) << runtime_error_message(emitted.error());
  ASSERT_GT(*emitted, 0U);
  ASSERT_TRUE(wait_for_phase(detail::RuntimePhaseCounter::kChannelWitnessRetired, retired + 1U));

  const std::uint64_t readiness =
      phase_count(detail::RuntimePhaseCounter::kChannelReadinessPublished);
  ASSERT_TRUE(publisher->close());
  ASSERT_TRUE(
      wait_for_phase(detail::RuntimePhaseCounter::kChannelReadinessPublished, readiness + 1U));
  // EOS has no sample-slot witness to count as kChannelWitnessRetired. A subsequent no-progress
  // scan proves RunSession ingested and retired the ineligible EOS candidate before the test posts
  // the successor anchor.
  const std::uint64_t no_progress = phase_count(detail::RuntimePhaseCounter::kNoProgressScans);
  ASSERT_TRUE(wait_for_phase(detail::RuntimePhaseCounter::kNoProgressScans, no_progress + 1U));

  retired = phase_count(detail::RuntimePhaseCounter::kChannelWitnessRetired);
  ASSERT_TRUE(sender.post());
  ASSERT_TRUE(PostEndOfStreamWindowProbe::wait_for_anchor_count(2U));
  ASSERT_TRUE(wait_for_phase(detail::RuntimePhaseCounter::kChannelWitnessRetired, retired + 1U));
  // Without the RunSession routing guard, this successor creates a pending window and its 20 ms
  // timeout activates the sink. The terminal selector must remain quiescent instead.
  EXPECT_FALSE(PostEndOfStreamWindowProbe::wait_for_window_count(1U, 100ms));
  EXPECT_EQ(PostEndOfStreamWindowProbe::window_count.load(std::memory_order_acquire), 0U);
  EXPECT_TRUE(session.is_running());

  session.request_stop();
  session.wait();
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
}

enum class RawTensorFullReuseLastScenario : std::uint8_t {
  kPostRetirementIdle,
  kIncompleteFreshnessPoll,
  kDeferredReuseReplacementPrerequisite,
  kLateIndependentWitnessAfterFreshnessPass,
  kNotificationCreditDuringRetirement,
};

void run_raw_tensor_full_reuse_last_scenario(const RawTensorFullReuseLastScenario scenario) {
  RetainedTensorEndOfStreamProbe::reset();
  const std::string endpoint_name = unique_endpoint_name();
  const bool contend_freshness =
      scenario == RawTensorFullReuseLastScenario::kIncompleteFreshnessPoll;
  const bool defer_replacement_prerequisite =
      scenario == RawTensorFullReuseLastScenario::kDeferredReuseReplacementPrerequisite;
  const bool admit_late_witness =
      scenario == RawTensorFullReuseLastScenario::kLateIndependentWitnessAfterFreshnessPass;
  const bool hold_retirement =
      scenario == RawTensorFullReuseLastScenario::kNotificationCreditDuringRetirement;

  Graph graph{"runtime-external-tensor-full-cache-end-of-stream"};
  const ClockKey capture_clock = graph.add_clock<RealtimeClock>("runtime-clock");
  graph.set_default_clock(capture_clock);
  TopicOptions options = optional_tensor_export_options(endpoint_name);
  options.bounds.subscriber.reset();
  options.bounds.publisher = PublisherEndpointBounds{.max_loaned_samples = 1U};
  options.capture_time = RequireCaptureTime{.clock = capture_clock};
  const auto imported = graph.import_topic<Tensor>("holomq://" + endpoint_name, std::move(options));
  const auto sink = graph.op<RetainedTensorEndOfStreamSink>("sink");
  graph.add_flow(imported.output(), sink->input, ConnectionOptions{.queue_depth = 1U});
  std::optional<OperatorHandle<DormantContentionSource>> contention_source;
  if (contend_freshness) {
    contention_source.emplace(graph.op<DormantContentionSource>("contention-source"));
    graph.add_flow(
        (*contention_source)->output, sink->contention, ConnectionOptions{.queue_depth = 1U});
    graph.partition("contention-source-partition").add(*contention_source);
    graph.partition("sink-partition").add(sink);
  }

  const ExecutionPlan compiled = compile(graph);
  ASSERT_TRUE(compiled.ok()) << compiled.json();
  const auto state = detail::ExecutionPlanAccess::retain(compiled);
  ASSERT_TRUE(state);
  const auto data_input =
      std::find_if(state->ports.begin(), state->ports.end(), [](const PortPlanRecord& port) {
        return port.direction == PortDirection::kInput && port.local_name == "input";
      });
  ASSERT_NE(data_input, state->ports.end());
  PortId contention_input_id{};
  if (contend_freshness) {
    const auto contention_input =
        std::find_if(state->ports.begin(), state->ports.end(), [](const PortPlanRecord& port) {
          return port.direction == PortDirection::kInput && port.local_name == "contention";
        });
    ASSERT_NE(contention_input, state->ports.end());
    contention_input_id = contention_input->id;
  } else {
    ASSERT_EQ(state->partition_runners.size(), 1U) << compiled.json();
  }
  auto phase_counters = std::make_shared<detail::RuntimePhaseCounters>();
  const ExecutionPlan counted =
      detail::RuntimeTestPlanAccess::with_phase_counters(compiled, phase_counters);
  // The contended-poll scenario also forces the owner's contention retries incomplete while the
  // sibling lock is held, so that round still ends with an incomplete ordinary scan while the
  // sole witness is pending. It is armed only at that point so ordinary pump contention earlier
  // in the run cannot consume the budget.
  const auto incomplete_poll_control =
      defer_replacement_prerequisite
          ? std::make_shared<detail::RuntimeChannelIncompletePollTestControl>(1U)
      : contend_freshness ? std::make_shared<detail::RuntimeChannelIncompletePollTestControl>(
                                0U, detail::RuntimeChannelIncompletePollKind::kContentionRetry)
                          : nullptr;
  const ExecutionPlan poll_controlled =
      incomplete_poll_control
          ? detail::RuntimeTestPlanAccess::with_channel_incomplete_poll_test_control(
                counted, incomplete_poll_control)
          : counted;
  const auto late_drain_control =
      admit_late_witness ? std::make_shared<detail::RuntimeLateReadinessDrainTestControl>()
                         : nullptr;
  const ExecutionPlan late_drain_controlled =
      late_drain_control ? detail::RuntimeTestPlanAccess::with_late_readiness_drain_test_control(
                               poll_controlled, late_drain_control)
                         : poll_controlled;
  const auto replacement_control = std::make_shared<RetryableRetainedTensorReplacementBindControl>(
      data_input->id,
      data_input->owner,
      contention_input_id,
      hold_retirement ? 2U : 0U,
      !defer_replacement_prerequisite && !admit_late_witness);
  const ExecutionPlan plan = detail::RuntimeTestPlanAccess::with_channel_session_bind_test_control(
      late_drain_controlled, replacement_control);
  ASSERT_TRUE(plan.ok()) << plan.json();

  auto external_runtime = ChannelRuntime::create();
  ASSERT_TRUE(external_runtime) << runtime_error_message(external_runtime.error());
  auto publisher =
      external_runtime->create_publisher<Tensor>(plan.endpoint_catalog(), endpoint_name);
  ASSERT_TRUE(publisher) << runtime_error_message(publisher.error());

  RunSession session = run_async(plan);
  ASSERT_TRUE(RetainedTensorEndOfStreamProbe::wait_for_sender());
  ASSERT_TRUE(wait_for_subscriber(*publisher));
  const auto phase_count = [&](const detail::RuntimePhaseCounter phase) {
    return phase_counters->snapshot()[detail::RuntimePhaseCounters::index(phase)];
  };
  const std::uint64_t readiness_baseline =
      phase_count(detail::RuntimePhaseCounter::kChannelReadinessPublished);
  const std::uint64_t activation_baseline =
      phase_count(detail::RuntimePhaseCounter::kActivationsCompleted);
  const std::uint64_t replacement_baseline =
      phase_count(detail::RuntimePhaseCounter::kTemporalRetentionEvictions);
  const std::uint64_t refusal_baseline =
      phase_count(detail::RuntimePhaseCounter::kTemporalRetentionDeclinedEvictionRefused);
  const auto wait_for_channel_readiness = [&](const std::uint64_t expected) {
    const auto deadline = std::chrono::steady_clock::now() + 5s;
    while (phase_count(detail::RuntimePhaseCounter::kChannelReadinessPublished) < expected &&
           session.is_running() && std::chrono::steady_clock::now() < deadline) {
      std::this_thread::yield();
    }
    return phase_count(detail::RuntimePhaseCounter::kChannelReadinessPublished) >= expected;
  };
  const auto wait_for_phase = [&](const detail::RuntimePhaseCounter phase,
                                  const std::uint64_t expected) {
    const auto deadline = std::chrono::steady_clock::now() + 5s;
    while (phase_count(phase) < expected && session.is_running() &&
           std::chrono::steady_clock::now() < deadline) {
      std::this_thread::yield();
    }
    return phase_count(phase) >= expected;
  };
  const auto post_notification = [&] {
    const auto deadline = std::chrono::steady_clock::now() + 5s;
    while (std::chrono::steady_clock::now() < deadline) {
      auto posted = RetainedTensorEndOfStreamProbe::sender_copy().post();
      if (posted) {
        return true;
      }
      if (posted.error().code != ErrorCode::kBackpressured) {
        return false;
      }
      std::this_thread::yield();
    }
    return false;
  };

  auto capture_time = publisher->make_capture_time(42);
  ASSERT_TRUE(capture_time) << runtime_error_message(capture_time.error());
  ASSERT_TRUE(publish_external_host_tensor(*publisher, *capture_time));
  ASSERT_TRUE(wait_for_channel_readiness(readiness_baseline + 1U));
  ASSERT_TRUE(post_notification());
  ASSERT_TRUE(RetainedTensorEndOfStreamProbe::wait_for_count(1U));

  if (hold_retirement) {
    // W2 has no staged successor yet, so it deterministically reuses A. Its callback records A
    // before the observer pauses the second Channel retirement transaction.
    ASSERT_TRUE(post_notification());
    ASSERT_TRUE(RetainedTensorEndOfStreamProbe::wait_for_count(2U));
    EXPECT_EQ(RetainedTensorEndOfStreamProbe::sequences[1], 1U);
    EXPECT_EQ(RetainedTensorEndOfStreamProbe::values[1], 0x2aU);
  }

  if (hold_retirement) {
    // Pause W2 after Channel retirement validation and before any retirement mutation. Stage B
    // and EOS while W2 still owns the capacity-one notification cell, then prove W3 cannot be
    // accepted until the complete retirement transaction releases that physical credit.
    const bool retirement_paused = replacement_control->wait_until_activation_retirement_paused();
    bool capture_created = false;
    bool final_data_published = false;
    bool final_data_staged = false;
    bool publisher_closed = false;
    bool post_attempted = false;
    bool post_accepted = false;
    ErrorCode capture_error = ErrorCode::kSuccess;
    ErrorCode publish_error = ErrorCode::kSuccess;
    ErrorCode staged_error = ErrorCode::kSuccess;
    ErrorCode close_error = ErrorCode::kSuccess;
    ErrorCode post_error = ErrorCode::kSuccess;
    if (retirement_paused) {
      auto final_capture_time = publisher->make_capture_time(84);
      capture_created = static_cast<bool>(final_capture_time);
      if (!final_capture_time) {
        capture_error = final_capture_time.error().code;
      } else {
        auto published =
            publish_external_host_tensor(*publisher, *final_capture_time, std::byte{0x5a});
        final_data_published = static_cast<bool>(published);
        if (!published) {
          publish_error = published.error().code;
        }
      }
      if (final_data_published) {
        const auto staged_data_deadline = std::chrono::steady_clock::now() + 5s;
        while (session.is_running() && std::chrono::steady_clock::now() < staged_data_deadline) {
          const auto staged_sequence = replacement_control->staged_data_sequence();
          if (!staged_sequence) {
            staged_error = staged_sequence.error().code;
            break;
          }
          if (*staged_sequence == std::optional<std::uint64_t>{2U}) {
            final_data_staged = true;
            break;
          }
          std::this_thread::yield();
        }
      }
      if (final_data_staged) {
        auto closed = publisher->close();
        publisher_closed = static_cast<bool>(closed);
        if (!closed) {
          close_error = closed.error().code;
        }
      }
      if (publisher_closed) {
        auto posted = RetainedTensorEndOfStreamProbe::sender_copy().post();
        post_attempted = true;
        post_accepted = static_cast<bool>(posted);
        if (!posted) {
          post_error = posted.error().code;
        }
      }
    }
    const bool credit_retained_while_paused =
        post_attempted && !post_accepted && post_error == ErrorCode::kBackpressured;
    replacement_control->release_activation_retirement();

    ASSERT_TRUE(retirement_paused);
    ASSERT_TRUE(replacement_control->activation_retirement_identity_matches());
    ASSERT_TRUE(capture_created) << "capture-time creation failed with "
                                 << static_cast<std::uint32_t>(capture_error);
    ASSERT_TRUE(final_data_published)
        << "final-data publication failed with " << static_cast<std::uint32_t>(publish_error);
    ASSERT_TRUE(final_data_staged)
        << "final-data staging failed with " << static_cast<std::uint32_t>(staged_error);
    ASSERT_TRUE(publisher_closed) << "publisher close failed with "
                                  << static_cast<std::uint32_t>(close_error);
    ASSERT_TRUE(credit_retained_while_paused)
        << "post accepted=" << post_accepted << " error=" << static_cast<std::uint32_t>(post_error);

    ASSERT_TRUE(post_notification());
    ASSERT_TRUE(wait_for_channel_readiness(readiness_baseline + 2U));
    ASSERT_TRUE(RetainedTensorEndOfStreamProbe::wait_for_count(3U));
    ASSERT_TRUE(wait_for_phase(detail::RuntimePhaseCounter::kActivationsCompleted,
                               activation_baseline + 3U));
    ASSERT_GT(phase_count(detail::RuntimePhaseCounter::kTemporalRetentionEvictions),
              replacement_baseline);
    EXPECT_EQ(phase_count(detail::RuntimePhaseCounter::kTemporalRetentionDeclinedEvictionRefused),
              refusal_baseline);
    EXPECT_EQ(RetainedTensorEndOfStreamProbe::count.load(std::memory_order_acquire), 3U);
    EXPECT_NE(RetainedTensorEndOfStreamProbe::data[0], nullptr);
    EXPECT_NE(RetainedTensorEndOfStreamProbe::data[1], nullptr);
    EXPECT_NE(RetainedTensorEndOfStreamProbe::data[2], nullptr);
    EXPECT_EQ(RetainedTensorEndOfStreamProbe::data[0], RetainedTensorEndOfStreamProbe::data[1]);
    EXPECT_EQ(RetainedTensorEndOfStreamProbe::sequences[0], 1U);
    EXPECT_EQ(RetainedTensorEndOfStreamProbe::sequences[1], 1U);
    EXPECT_EQ(RetainedTensorEndOfStreamProbe::sequences[2], 2U);
    EXPECT_EQ(RetainedTensorEndOfStreamProbe::values[0], 0x2aU);
    EXPECT_EQ(RetainedTensorEndOfStreamProbe::values[1], 0x2aU);
    EXPECT_EQ(RetainedTensorEndOfStreamProbe::values[2], 0x5aU);

    const auto terminal_deadline = std::chrono::steady_clock::now() + 5s;
    bool terminal_observed = false;
    while (session.is_running() && std::chrono::steady_clock::now() < terminal_deadline) {
      const auto observed = replacement_control->input_terminal_observed();
      ASSERT_TRUE(observed) << runtime_error_message(observed.error());
      if (*observed) {
        terminal_observed = true;
        break;
      }
      std::this_thread::yield();
    }
    ASSERT_TRUE(terminal_observed);
    session.request_stop();
    session.wait();
    EXPECT_EQ(session.termination(), RunTermination::kStopped);
    EXPECT_TRUE(session.diagnostics().empty())
        << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
    return;
  }

  const auto first_retirement_deadline = std::chrono::steady_clock::now() + 5s;
  while (phase_count(detail::RuntimePhaseCounter::kActivationsCompleted) <
             activation_baseline + 1U &&
         session.is_running() && std::chrono::steady_clock::now() < first_retirement_deadline) {
    std::this_thread::yield();
  }
  ASSERT_GE(phase_count(detail::RuntimePhaseCounter::kActivationsCompleted),
            activation_baseline + 1U);

  auto final_capture_time = publisher->make_capture_time(84);
  ASSERT_TRUE(final_capture_time) << runtime_error_message(final_capture_time.error());
  ASSERT_TRUE(publish_external_host_tensor(*publisher, *final_capture_time, std::byte{0x5a}));
  if (scenario == RawTensorFullReuseLastScenario::kPostRetirementIdle) {
    ASSERT_TRUE(publisher->close());
  }
  ASSERT_TRUE(session.is_running());
  const auto staged_data_deadline = std::chrono::steady_clock::now() + 5s;
  bool final_data_staged = false;
  while (session.is_running() && std::chrono::steady_clock::now() < staged_data_deadline) {
    const auto staged_sequence = replacement_control->staged_data_sequence();
    ASSERT_TRUE(staged_sequence) << runtime_error_message(staged_sequence.error());
    if (*staged_sequence == std::optional<std::uint64_t>{2U}) {
      final_data_staged = true;
      break;
    }
    std::this_thread::yield();
  }
  ASSERT_TRUE(session.is_running());
  ASSERT_TRUE(final_data_staged);

  if (admit_late_witness) {
    // B is already staged behind full ReuseLast state. Wait for one complete no-witness pass, then
    // pause the next owner round after its ordinary poll and freshness reconcile but before its
    // fallback binding drain. W2 is accepted only while paused, making the late admission exact.
    ASSERT_NE(late_drain_control, nullptr);
    const std::uint64_t idle_baseline = phase_count(detail::RuntimePhaseCounter::kNoProgressScans);
    ASSERT_TRUE(wait_for_phase(detail::RuntimePhaseCounter::kNoProgressScans, idle_baseline + 1U));
    late_drain_control->arm();
    const bool late_drain_paused = late_drain_control->wait_until_paused();
    bool notification_posted = false;
    ErrorCode post_error = ErrorCode::kSuccess;
    if (late_drain_paused) {
      auto posted = RetainedTensorEndOfStreamProbe::sender_copy().post();
      notification_posted = static_cast<bool>(posted);
      if (!posted) {
        post_error = posted.error().code;
      }
    }
    const std::size_t delivery_count_while_paused =
        RetainedTensorEndOfStreamProbe::count.load(std::memory_order_acquire);
    late_drain_control->release();

    ASSERT_TRUE(late_drain_paused);
    ASSERT_TRUE(notification_posted)
        << "notification post failed with " << static_cast<std::uint32_t>(post_error);
    ASSERT_EQ(delivery_count_while_paused, 1U);
    ASSERT_TRUE(RetainedTensorEndOfStreamProbe::wait_for_count(2U));
    ASSERT_TRUE(wait_for_phase(detail::RuntimePhaseCounter::kActivationsCompleted,
                               activation_baseline + 2U));
    ASSERT_GT(phase_count(detail::RuntimePhaseCounter::kTemporalRetentionEvictions),
              replacement_baseline);
    EXPECT_EQ(phase_count(detail::RuntimePhaseCounter::kTemporalRetentionDeclinedEvictionRefused),
              refusal_baseline);
    EXPECT_EQ(RetainedTensorEndOfStreamProbe::count.load(std::memory_order_acquire), 2U);
    EXPECT_EQ(RetainedTensorEndOfStreamProbe::sequences[0], 1U);
    EXPECT_EQ(RetainedTensorEndOfStreamProbe::sequences[1], 2U);
    EXPECT_EQ(RetainedTensorEndOfStreamProbe::values[0], 0x2aU);
    EXPECT_EQ(RetainedTensorEndOfStreamProbe::values[1], 0x5aU);

    ASSERT_TRUE(publisher->close());
    const auto terminal_deadline = std::chrono::steady_clock::now() + 5s;
    bool terminal_observed = false;
    while (session.is_running() && std::chrono::steady_clock::now() < terminal_deadline) {
      const auto observed = replacement_control->input_terminal_observed();
      ASSERT_TRUE(observed) << runtime_error_message(observed.error());
      if (*observed) {
        terminal_observed = true;
        break;
      }
      std::this_thread::yield();
    }
    ASSERT_TRUE(terminal_observed);
    session.request_stop();
    session.wait();
    EXPECT_EQ(session.termination(), RunTermination::kStopped);
    EXPECT_TRUE(session.diagnostics().empty())
        << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
    return;
  }

  if (defer_replacement_prerequisite) {
    // No synthetic replacement refusal participates in this scenario. After the first activation
    // is completely idle, W2 alone authorizes the staged A-to-B handoff. The forced incomplete
    // blocking recovery poll leaves B's exact input-arrival prerequisite unpublished, so W2 must
    // remain coordinator-owned until a later bounded round ingests that prerequisite.
    ASSERT_FALSE(replacement_control->armed());
    ASSERT_NE(incomplete_poll_control, nullptr);
    const std::uint64_t idle_baseline = phase_count(detail::RuntimePhaseCounter::kNoProgressScans);
    ASSERT_TRUE(wait_for_phase(detail::RuntimePhaseCounter::kNoProgressScans, idle_baseline + 1U));
    ASSERT_EQ(incomplete_poll_control->forced(), 0U);
    const std::uint64_t incomplete_poll_baseline =
        phase_count(detail::RuntimePhaseCounter::kChannelPollIncomplete);
    ASSERT_TRUE(post_notification());
    const auto forced_poll_deadline = std::chrono::steady_clock::now() + 5s;
    while (incomplete_poll_control->forced() == 0U && session.is_running() &&
           std::chrono::steady_clock::now() < forced_poll_deadline) {
      std::this_thread::yield();
    }
    ASSERT_EQ(incomplete_poll_control->forced(), 1U);
    ASSERT_TRUE(wait_for_phase(detail::RuntimePhaseCounter::kChannelPollIncomplete,
                               incomplete_poll_baseline + 1U));
    ASSERT_TRUE(wait_for_phase(detail::RuntimePhaseCounter::kTemporalRetentionEvictions,
                               replacement_baseline + 1U));
    ASSERT_TRUE(wait_for_channel_readiness(readiness_baseline + 2U));
    ASSERT_TRUE(RetainedTensorEndOfStreamProbe::wait_for_count(2U));
    ASSERT_TRUE(wait_for_phase(detail::RuntimePhaseCounter::kActivationsCompleted,
                               activation_baseline + 2U));
    EXPECT_EQ(phase_count(detail::RuntimePhaseCounter::kTemporalRetentionDeclinedEvictionRefused),
              refusal_baseline);
    EXPECT_EQ(RetainedTensorEndOfStreamProbe::count.load(std::memory_order_acquire), 2U);
    EXPECT_NE(RetainedTensorEndOfStreamProbe::data[0], nullptr);
    EXPECT_NE(RetainedTensorEndOfStreamProbe::data[1], nullptr);
    EXPECT_EQ(RetainedTensorEndOfStreamProbe::sequences[0], 1U);
    EXPECT_EQ(RetainedTensorEndOfStreamProbe::sequences[1], 2U);
    EXPECT_EQ(RetainedTensorEndOfStreamProbe::values[0], 0x2aU);
    EXPECT_EQ(RetainedTensorEndOfStreamProbe::values[1], 0x5aU);

    ASSERT_TRUE(publisher->close());
    const auto terminal_deadline = std::chrono::steady_clock::now() + 5s;
    bool terminal_observed = false;
    while (session.is_running() && std::chrono::steady_clock::now() < terminal_deadline) {
      const auto observed = replacement_control->input_terminal_observed();
      ASSERT_TRUE(observed) << runtime_error_message(observed.error());
      if (*observed) {
        terminal_observed = true;
        break;
      }
      std::this_thread::yield();
    }
    ASSERT_TRUE(terminal_observed);
    session.request_stop();
    session.wait();
    EXPECT_EQ(session.termination(), RunTermination::kStopped);
    EXPECT_TRUE(session.diagnostics().empty())
        << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
    return;
  }

  // The notification is the immutable activation witness for this side input. Publish it before
  // waiting for handoff so selector eligibility is proven against a real admitted witness; the
  // no-witness path intentionally preserves the existing ReuseLast record.
  ASSERT_TRUE(post_notification());
  const auto refusal_deadline = std::chrono::steady_clock::now() + 5s;
  while (phase_count(detail::RuntimePhaseCounter::kTemporalRetentionDeclinedEvictionRefused) <=
             refusal_baseline &&
         session.is_running() && std::chrono::steady_clock::now() < refusal_deadline) {
    std::this_thread::yield();
  }
  if (phase_count(detail::RuntimePhaseCounter::kTemporalRetentionDeclinedEvictionRefused) !=
      refusal_baseline + 1U) {
    session.request_stop();
    session.wait();
    FAIL() << "replacement-refusal timeout: scenario=" << static_cast<std::uint32_t>(scenario)
           << " deliveries="
           << RetainedTensorEndOfStreamProbe::count.load(std::memory_order_acquire)
           << " activations=" << phase_count(detail::RuntimePhaseCounter::kActivationsCompleted)
           << " evictions=" << phase_count(detail::RuntimePhaseCounter::kTemporalRetentionEvictions)
           << " refusals="
           << phase_count(detail::RuntimePhaseCounter::kTemporalRetentionDeclinedEvictionRefused)
           << " candidates="
           << phase_count(detail::RuntimePhaseCounter::kTemporalRetentionCandidates)
           << " above_floor="
           << phase_count(detail::RuntimePhaseCounter::kTemporalRetentionDeclinedAboveFloor)
           << " not_superseded="
           << phase_count(detail::RuntimePhaseCounter::kTemporalRetentionDeclinedNotSuperseded)
           << " incomplete=" << phase_count(detail::RuntimePhaseCounter::kChannelPollIncomplete)
           << " coordinator_skips="
           << phase_count(detail::RuntimePhaseCounter::kReadinessPollSkippedByCoordinatorBound)
           << " source_skips="
           << phase_count(detail::RuntimePhaseCounter::kReadinessPollSkippedBySourceCredit)
           << " frozen=" << phase_count(detail::RuntimePhaseCounter::kCandidatesFrozen)
           << " selected=" << phase_count(detail::RuntimePhaseCounter::kCandidatesSelected)
           << " readiness=" << phase_count(detail::RuntimePhaseCounter::kChannelReadinessPublished)
           << " no_target="
           << phase_count(detail::RuntimePhaseCounter::kReadinessPollSkippedByNoTarget)
           << " no_progress=" << phase_count(detail::RuntimePhaseCounter::kNoProgressScans)
           << " diagnostics=" << session.diagnostics().size();
  }
  ASSERT_TRUE(replacement_control->armed());
  ASSERT_TRUE(RetainedTensorEndOfStreamProbe::wait_for_count(2U));
  EXPECT_EQ(RetainedTensorEndOfStreamProbe::sequences[1], 1U);
  EXPECT_EQ(RetainedTensorEndOfStreamProbe::values[1], 0x2aU);
  const auto refusal_retirement_deadline = std::chrono::steady_clock::now() + 5s;
  while (phase_count(detail::RuntimePhaseCounter::kActivationsCompleted) <
             activation_baseline + 2U &&
         session.is_running() && std::chrono::steady_clock::now() < refusal_retirement_deadline) {
    std::this_thread::yield();
  }
  ASSERT_GE(phase_count(detail::RuntimePhaseCounter::kActivationsCompleted),
            activation_baseline + 2U);

  if (scenario == RawTensorFullReuseLastScenario::kPostRetirementIdle) {
    // kActivationsCompleted is inside the owner transaction. A later no-progress scan on this
    // sole partition proves caller cleanup and a complete idle readiness pass before W3.
    const std::uint64_t idle_baseline = phase_count(detail::RuntimePhaseCounter::kNoProgressScans);
    ASSERT_TRUE(wait_for_phase(detail::RuntimePhaseCounter::kNoProgressScans, idle_baseline + 1U));
    ASSERT_TRUE(post_notification());
  } else {
    // Hold an inert sibling input's cache while publishing ordered EOS and the later notification.
    // The exact failed try-lock observer acknowledges only after that notification post completes,
    // proving the owner-local Channel poll was incomplete while the sole witness was pending. The
    // observer fires only from a real lost try_lock, and this probe releases the lock as soon as it
    // has seen one, so the owner's following contention retries are forced incomplete: the round
    // that lost the lock still ends unpolled (the owner may then take the bounded fallback park,
    // exactly as before the retry existed) instead of converting the revision through a retry once
    // the lock is gone. One post-notification decline is sufficient because RUNTIME-051 requires
    // one actual incomplete ordinary scan for the exact binding. Four forced retries cover an
    // EOS-triggered pre-post round plus a post-notification round with margin; leftover budget is
    // harmless because a forced retry only follows a real decline and then costs that round nothing
    // more than the pre-existing bounded park.
    ASSERT_NE(incomplete_poll_control, nullptr);
    incomplete_poll_control->arm(4U);
    LockedSiblingInputNotificationPollProbe contention{
        .sender = RetainedTensorEndOfStreamProbe::sender_copy(),
        .publisher = &*publisher,
        .session = &session,
        .control = replacement_control.get(),
    };
    auto contended = replacement_control->with_contention_input_lock(
        &contention, &LockedSiblingInputNotificationPollProbe::observe);
    ASSERT_TRUE(contended) << runtime_error_message(contended.error());
    ASSERT_TRUE(contention.publisher_closed)
        << "publisher close failed with " << static_cast<std::uint32_t>(contention.close_error);
    ASSERT_TRUE(contention.notification_posted);
    ASSERT_GE(contention.poll_contention_count, 1U);
    // The probe returns the moment the observer counts the lost try_lock, while the owner is still
    // inside that poll; the forced contention retry runs afterwards on the owner thread, so wait
    // for it instead of racing it.
    const auto forced_retry_deadline = std::chrono::steady_clock::now() + 5s;
    while (incomplete_poll_control->forced() < 1U && session.is_running() &&
           std::chrono::steady_clock::now() < forced_retry_deadline) {
      std::this_thread::yield();
    }
    EXPECT_GE(incomplete_poll_control->forced(), 1U);
    ASSERT_TRUE(replacement_control->poll_contention_identity_matches());
    ASSERT_EQ(contention.delivery_count_while_contended, 2U);
  }
  ASSERT_TRUE(session.is_running());
  const auto replacement_deadline = std::chrono::steady_clock::now() + 5s;
  while (phase_count(detail::RuntimePhaseCounter::kTemporalRetentionEvictions) <=
             replacement_baseline &&
         session.is_running() && std::chrono::steady_clock::now() < replacement_deadline) {
    std::this_thread::yield();
  }
  ASSERT_GT(phase_count(detail::RuntimePhaseCounter::kTemporalRetentionEvictions),
            replacement_baseline);
  EXPECT_EQ(phase_count(detail::RuntimePhaseCounter::kTemporalRetentionDeclinedEvictionRefused),
            refusal_baseline + 1U);
  ASSERT_TRUE(wait_for_channel_readiness(readiness_baseline + 2U));
  ASSERT_TRUE(session.is_running());
  if (!RetainedTensorEndOfStreamProbe::wait_for_count(3U)) {
    session.request_stop();
    session.wait();
    FAIL() << "third delivery timeout: deliveries="
           << RetainedTensorEndOfStreamProbe::count.load(std::memory_order_acquire)
           << " activations=" << phase_count(detail::RuntimePhaseCounter::kActivationsCompleted)
           << " frozen=" << phase_count(detail::RuntimePhaseCounter::kCandidatesFrozen)
           << " selected=" << phase_count(detail::RuntimePhaseCounter::kCandidatesSelected)
           << " readiness=" << phase_count(detail::RuntimePhaseCounter::kChannelReadinessPublished)
           << " retired=" << phase_count(detail::RuntimePhaseCounter::kChannelWitnessRetired)
           << " incomplete=" << phase_count(detail::RuntimePhaseCounter::kChannelPollIncomplete)
           << " coordinator_skips="
           << phase_count(detail::RuntimePhaseCounter::kReadinessPollSkippedByCoordinatorBound)
           << " source_skips="
           << phase_count(detail::RuntimePhaseCounter::kReadinessPollSkippedBySourceCredit)
           << " no_target="
           << phase_count(detail::RuntimePhaseCounter::kReadinessPollSkippedByNoTarget)
           << " refill_published="
           << phase_count(detail::RuntimePhaseCounter::kChannelReleasedClaimRefillPublished)
           << " missed_after_eviction="
           << phase_count(detail::RuntimePhaseCounter::kSelectionMissedAfterFreshnessEviction)
           << " missed_after_ungated_eviction="
           << phase_count(detail::RuntimePhaseCounter::kSelectionMissedAfterUngatedEviction)
           << " missed_after_unpolled_eviction="
           << phase_count(detail::RuntimePhaseCounter::kSelectionMissedAfterUnpolledEviction)
           << " missed_after_unpublished_refill="
           << phase_count(detail::RuntimePhaseCounter::kSelectionMissedAfterUnrefilledEviction)
           << " missed_without_eviction="
           << phase_count(detail::RuntimePhaseCounter::kSelectionMissedWithoutFreshnessEviction)
           << " no_progress=" << phase_count(detail::RuntimePhaseCounter::kNoProgressScans)
           << " timeouts=" << phase_count(detail::RuntimePhaseCounter::kTimeoutReturns)
           << " wakes=" << phase_count(detail::RuntimePhaseCounter::kWakeReturns)
           << " termination=" << static_cast<std::uint32_t>(session.termination())
           << " diagnostics=" << session.diagnostics().size();
  }
  const auto second_retirement_deadline = std::chrono::steady_clock::now() + 5s;
  while (phase_count(detail::RuntimePhaseCounter::kActivationsCompleted) <
             activation_baseline + 3U &&
         session.is_running() && std::chrono::steady_clock::now() < second_retirement_deadline) {
    std::this_thread::yield();
  }
  ASSERT_GE(phase_count(detail::RuntimePhaseCounter::kActivationsCompleted),
            activation_baseline + 3U);

  EXPECT_EQ(RetainedTensorEndOfStreamProbe::count.load(std::memory_order_acquire), 3U);
  EXPECT_NE(RetainedTensorEndOfStreamProbe::data[0], nullptr);
  EXPECT_NE(RetainedTensorEndOfStreamProbe::data[1], nullptr);
  EXPECT_NE(RetainedTensorEndOfStreamProbe::data[2], nullptr);
  EXPECT_EQ(RetainedTensorEndOfStreamProbe::data[0], RetainedTensorEndOfStreamProbe::data[1]);
  EXPECT_EQ(RetainedTensorEndOfStreamProbe::sequences[0], 1U);
  EXPECT_EQ(RetainedTensorEndOfStreamProbe::sequences[1], 1U);
  EXPECT_EQ(RetainedTensorEndOfStreamProbe::sequences[2], 2U);
  EXPECT_EQ(RetainedTensorEndOfStreamProbe::values[0], 0x2aU);
  EXPECT_EQ(RetainedTensorEndOfStreamProbe::values[1], 0x2aU);
  EXPECT_EQ(RetainedTensorEndOfStreamProbe::values[2], 0x5aU);

  // Both terminal-order scenarios publish EOS ahead of W3. EOS remains non-activating and becomes
  // observable only after the accepted witness admits the staged final data.
  const auto terminal_deadline = std::chrono::steady_clock::now() + 5s;
  bool terminal_observed = false;
  while (session.is_running() && std::chrono::steady_clock::now() < terminal_deadline) {
    const auto observed = replacement_control->input_terminal_observed();
    ASSERT_TRUE(observed) << runtime_error_message(observed.error());
    if (*observed) {
      terminal_observed = true;
      break;
    }
    std::this_thread::yield();
  }
  ASSERT_TRUE(terminal_observed);

  session.request_stop();
  session.wait();
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
}

// A capacity-one ReuseLast cache has no writable data slot after its first activation. The
// original CI topology waits for a post-retirement idle scan before W3, then proves final data is
// delivered before the already-published ordered EOS.
// REQ: CHAN-078
// REQ: RUNTIME-025
// REQ: RUNTIME-017
// REQ: SYSTEM-006
// REQ: CHAN-136
TEST(RuntimeExternalChannelTest, RawTensorEndOfStreamCrossesFullReuseLastCache) {
  ASSERT_NO_FATAL_FAILURE(
      run_raw_tensor_full_reuse_last_scenario(RawTensorFullReuseLastScenario::kPostRetirementIdle));
}

// An actual incomplete ordinary poll must preserve the sole independent W3 identity rather than
// freezing it against stale ReuseLast state while a sibling cache mutex hides staged data.
// REQ: RUNTIME-051
TEST(RuntimeExternalChannelTest,
     RawTensorIndependentWitnessSurvivesIncompleteReuseLastFreshnessPoll) {
  ASSERT_NO_FATAL_FAILURE(run_raw_tensor_full_reuse_last_scenario(
      RawTensorFullReuseLastScenario::kIncompleteFreshnessPoll));
}

// A staged ReuseLast replacement is not selector-visible until its exact Channel prerequisite is
// published and ingested. One forced incomplete recovery poll pins that cold publication gap; the
// sole successor notification must remain pending even though OnNotified has no witness port.
// REQ: RUNTIME-051
TEST(RuntimeExternalChannelTest, RawTensorIndependentWitnessWaitsForReuseReplacementPrerequisite) {
  ASSERT_NO_FATAL_FAILURE(run_raw_tensor_full_reuse_last_scenario(
      RawTensorFullReuseLastScenario::kDeferredReuseReplacementPrerequisite));
}

// A witness first admitted by the late fallback drain arrived after the ordinary freshness pass.
// Preserve it for a complete bounded rescan instead of freezing it immediately against stale A.
// REQ: RUNTIME-051
TEST(RuntimeExternalChannelTest,
     RawTensorLateIndependentWitnessRestartsFreshnessBeforeFallbackFreeze) {
  ASSERT_NO_FATAL_FAILURE(run_raw_tensor_full_reuse_last_scenario(
      RawTensorFullReuseLastScenario::kLateIndependentWitnessAfterFreshnessPass));
}

// Notification capacity remains charged until selector, Channel, and activation retirement all
// finish. W3 may retry while retirement is open, but it cannot be accepted until W2 releases its
// physical credit and must then observe staged B before EOS.
// REQ: RUNTIME-025
TEST(RuntimeExternalChannelTest, RawTensorNotificationCreditSpansCompleteActivationRetirement) {
  ASSERT_NO_FATAL_FAILURE(run_raw_tensor_full_reuse_last_scenario(
      RawTensorFullReuseLastScenario::kNotificationCreditDuringRetirement));
}

// Data and EOS from an auxiliary Channel source update bounded selector state but cannot replace
// the activation authority of an OnClock or OnNotified trigger. The later independent witness must
// observe the retained sample exactly once.
// REQ: CHAN-102
// REQ: RUNTIME-025
TEST(RuntimeExternalChannelTest, AuxiliaryEndOfStreamWaitsForIndependentActivationAuthority) {
  for (const IndependentSideInputTriggerRoute route :
       {IndependentSideInputTriggerRoute::kClock, IndependentSideInputTriggerRoute::kNotified}) {
    SCOPED_TRACE(route == IndependentSideInputTriggerRoute::kClock ? "OnClock" : "OnNotified");
    IndependentSideInputProbe::reset();
    const std::string endpoint_name = unique_endpoint_name();

    Graph graph{"runtime-independent-side-input-eos"};
    const ClockKey runtime_clock = graph.add_clock<RealtimeClock>("runtime-clock");
    graph.set_default_clock(runtime_clock);
    TopicOptions options;
    options.endpoint_name = endpoint_name;
    options.capture_time = RequireCaptureTime{.clock = runtime_clock};
    const auto imported =
        graph.import_topic<std::uint32_t>("holomq://" + endpoint_name, std::move(options));
    const auto sink = graph.op<IndependentSideInputSink>("sink", route);
    graph.add_flow(imported.output(), sink->input, ConnectionOptions{.queue_depth = 2U});

    const ExecutionPlan compiled = compile(graph);
    auto phase_counters = std::make_shared<detail::RuntimePhaseCounters>();
    const ExecutionPlan plan =
        detail::RuntimeTestPlanAccess::with_phase_counters(compiled, phase_counters);
    ASSERT_TRUE(plan.ok()) << plan.json();
    auto external_runtime = ChannelRuntime::create();
    ASSERT_TRUE(external_runtime) << runtime_error_message(external_runtime.error());
    auto publisher =
        external_runtime->create_publisher<std::uint32_t>(plan.endpoint_catalog(), endpoint_name);
    ASSERT_TRUE(publisher) << runtime_error_message(publisher.error());

    RunSession session = run_async(plan);
    if (auto connected = wait_for_subscriber(*publisher); !connected) {
      session.request_stop();
      session.wait();
      FAIL() << "independent side-input subscriber did not connect: "
             << runtime_error_message(connected.error());
    }
    if (route == IndependentSideInputTriggerRoute::kNotified &&
        !IndependentSideInputProbe::wait_for_sender()) {
      session.request_stop();
      session.wait();
      FAIL() << "independent notification sender was not published";
    }

    const auto phase_count = [&](detail::RuntimePhaseCounter phase) {
      return phase_counters->snapshot()[detail::RuntimePhaseCounters::index(phase)];
    };
    if (route == IndependentSideInputTriggerRoute::kClock) {
      const auto initial_clock_deadline = std::chrono::steady_clock::now() + 5s;
      while (phase_count(detail::RuntimePhaseCounter::kCandidatesFrozen) == 0U &&
             session.is_running() && std::chrono::steady_clock::now() < initial_clock_deadline) {
        std::this_thread::yield();
      }
      ASSERT_NE(phase_count(detail::RuntimePhaseCounter::kCandidatesFrozen), 0U);
      EXPECT_EQ(IndependentSideInputProbe::compute_count.load(std::memory_order_acquire), 0U);
    }

    const std::uint64_t readiness_baseline =
        phase_count(detail::RuntimePhaseCounter::kChannelReadinessPublished);
    ASSERT_TRUE(publisher->emit_captured(42U, 0));
    ASSERT_TRUE(publisher->close());
    const auto eos_deadline = std::chrono::steady_clock::now() + 5s;
    while (phase_count(detail::RuntimePhaseCounter::kChannelReadinessPublished) <
               readiness_baseline + 2U &&
           session.is_running() && std::chrono::steady_clock::now() < eos_deadline) {
      std::this_thread::yield();
    }
    ASSERT_EQ(phase_count(detail::RuntimePhaseCounter::kChannelReadinessPublished),
              readiness_baseline + 2U);
    // Both ordered operations are now visible to Runtime. Allow the already-signaled dispatcher
    // turn to consume and retire them before checking that neither supplied activation authority.
    std::this_thread::sleep_for(50ms);
    ASSERT_TRUE(session.is_running());
    EXPECT_EQ(IndependentSideInputProbe::compute_count.load(std::memory_order_acquire), 0U);

    if (route == IndependentSideInputTriggerRoute::kNotified) {
      ASSERT_TRUE(IndependentSideInputProbe::sender_copy().post());
    }
    const bool computed = IndependentSideInputProbe::wait_for_compute();
    session.request_stop();
    session.wait();

    EXPECT_TRUE(computed);
    EXPECT_EQ(session.termination(), RunTermination::kStopped);
    EXPECT_TRUE(session.diagnostics().empty())
        << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
    EXPECT_EQ(IndependentSideInputProbe::compute_count.load(std::memory_order_acquire), 1U);
    EXPECT_EQ(IndependentSideInputProbe::value.load(std::memory_order_relaxed), 42U);
    EXPECT_NE(IndependentSideInputProbe::sequence.load(std::memory_order_relaxed), 0U);
  }
}

// Two independently monitored imports share the run-wide clock-health substrate while their
// callback-bearing partitions retain distinct serial owners. A bounded rendezvous inside both
// callbacks proves the health snapshot is not protected by a lock held across user code.
// REQ: CHAN-069
// REQ: RUNTIME-015
// REQ: RUNTIME-041
TEST(RuntimeExternalChannelTest,
     IndependentMonitoredImportsOverlapOnDistinctPartitionOwnersWithoutLegacyHandoffs) {
  ConcurrentMonitoredImportProbe::reset();
  const std::array endpoints{unique_endpoint_name(), unique_endpoint_name()};
  const std::int64_t capture_offset = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                          std::chrono::steady_clock::now().time_since_epoch())
                                          .count();

  Graph graph{"runtime-independent-monitored-import-owners"};
  const ClockKey capture_clock = graph.add_clock<ManualClock>("capture-clock", std::int64_t{0});
  const ClockKey runtime_clock = graph.add_clock<RealtimeClock>("runtime-clock");
  graph.set_default_clock(runtime_clock);
  graph.add_clock_conversion(
      capture_clock, runtime_clock, AffineClockConversion{.offset_ns = capture_offset});
  const auto import = [&](std::size_t index) {
    TopicOptions options;
    options.endpoint_name = endpoints[index];
    options.capture_time = RequireCaptureTime{.clock = capture_clock};
    options.clock_health = ClockHealthOptions{
        .capability = ClockHealthCapability::kStatusReports,
        .requirement = ClockHealthRequirement::kFreshHealthyReport,
        .max_report_age = 5s,
    };
    return graph.import_topic<std::uint32_t>("holomq://" + endpoints[index], std::move(options));
  };
  const auto first_import = import(0U);
  const auto second_import = import(1U);
  const auto first_sink = graph.op<ConcurrentMonitoredImportSink>("first-sink", 0U);
  const auto second_sink = graph.op<ConcurrentMonitoredImportSink>("second-sink", 1U);
  graph.add_flow(first_import.output(), first_sink->input, ConnectionOptions{.queue_depth = 2U});
  graph.add_flow(second_import.output(), second_sink->input, ConnectionOptions{.queue_depth = 2U});
  graph.partition("first-owner").add(first_sink);
  graph.partition("second-owner").add(second_sink);

  const ExecutionPlan compiled = compile(graph);
  ASSERT_TRUE(compiled.ok()) << compiled.json();
  EXPECT_EQ(compiled.scheduler_plan().worker_count, 2U);
  EXPECT_EQ(compiled.scheduler_plan().max_concurrent_activations, 2U);
  EXPECT_TRUE(compiled.scheduler_plan().activation_overlap);
  auto phase_counters = std::make_shared<detail::RuntimePhaseCounters>();
  const ExecutionPlan plan =
      detail::RuntimeTestPlanAccess::with_phase_counters(compiled, phase_counters);
  ASSERT_TRUE(plan.ok()) << plan.json();

  auto external_runtime = ChannelRuntime::create();
  ASSERT_TRUE(external_runtime) << runtime_error_message(external_runtime.error());
  auto first_publisher =
      external_runtime->create_publisher<std::uint32_t>(plan.endpoint_catalog(), endpoints[0]);
  auto second_publisher =
      external_runtime->create_publisher<std::uint32_t>(plan.endpoint_catalog(), endpoints[1]);
  ASSERT_TRUE(first_publisher) << runtime_error_message(first_publisher.error());
  ASSERT_TRUE(second_publisher) << runtime_error_message(second_publisher.error());

  RunSession session = run_async(plan);
  const auto first_connected = wait_for_subscriber(*first_publisher);
  const auto second_connected = wait_for_subscriber(*second_publisher);
  if (!first_connected || !second_connected) {
    session.request_stop();
    session.wait();
    FAIL() << "independent monitored publishers did not connect: first="
           << (first_connected ? "connected" : runtime_error_message(first_connected.error()))
           << ", second="
           << (second_connected ? "connected" : runtime_error_message(second_connected.error()));
  }
  auto reported = first_publisher->report_clock_health(ClockHealthReport::healthy());
  if (!reported) {
    session.request_stop();
    session.wait();
    FAIL() << "first independent health report failed: " << runtime_error_message(reported.error());
  }
  reported = second_publisher->report_clock_health(ClockHealthReport::healthy());
  if (!reported) {
    session.request_stop();
    session.wait();
    FAIL() << "second independent health report failed: "
           << runtime_error_message(reported.error());
  }

  constexpr std::array kValues{0x01020304U, 0xa1b2c3d4U};
  auto emitted = first_publisher->emit_captured(kValues[0], 0);
  if (!emitted) {
    session.request_stop();
    session.wait();
    FAIL() << "first independent sample failed: " << runtime_error_message(emitted.error());
  }
  emitted = second_publisher->emit_captured(kValues[1], 1);
  if (!emitted) {
    session.request_stop();
    session.wait();
    FAIL() << "second independent sample failed: " << runtime_error_message(emitted.error());
  }
  const bool completed = ConcurrentMonitoredImportProbe::wait_for_completion();
  session.request_stop();
  session.wait();

  const auto observation = ConcurrentMonitoredImportProbe::snapshot();
  ASSERT_TRUE(completed) << "entered=" << observation.entered
                         << ", completed=" << observation.completed
                         << ", timed_out=" << observation.barrier_timed_out;
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
  EXPECT_FALSE(observation.barrier_timed_out);
  EXPECT_EQ(observation.entered, 2U);
  EXPECT_EQ(observation.completed, 2U);
  EXPECT_EQ(observation.values, kValues);
  EXPECT_EQ(observation.sequences, (std::array<std::uint64_t, 2U>{2U, 2U}));
  EXPECT_NE(observation.owner_threads[0], std::thread::id{});
  EXPECT_NE(observation.owner_threads[1], std::thread::id{});
  EXPECT_NE(observation.owner_threads[0], observation.owner_threads[1]);

  const auto phase_snapshot = phase_counters->snapshot();
  const auto count = [&](detail::RuntimePhaseCounter counter) {
    return phase_snapshot[detail::RuntimePhaseCounters::index(counter)];
  };
  EXPECT_EQ(count(detail::RuntimePhaseCounter::kLaneComputeDispatches), 0U);
  EXPECT_GE(count(detail::RuntimePhaseCounter::kPartitionRunnerSweeps), 2U);
}

// Two independent publishers intentionally expose the newer normalized capture coordinate first.
// The later-visible older witness must retain its source timestamp and clock while advancing the
// run-wide TagTime cursor by one microstep. Reusing the immutable plan must replay the same result.
// REQ: CONTRACT-004
// REQ: CONTRACT-006
// REQ: RUNTIME-007
// REQ: RUNTIME-018
// REQ: RUNTIME-034
TEST(RuntimeExternalChannelTest,
     IndependentAlignedSourcesPreserveCaptureCoordinatesAndReplayLateVisibility) {
  const std::string first_endpoint = unique_endpoint_name();
  const std::string second_endpoint = unique_endpoint_name();

  // first: 2 * 40 + 20 = 100 in the comparison clock.
  // second: 190 + 10 = 200 in the comparison clock. Publish second first so the eventual first
  // witness is valid late visibility rather than a timestamp regression or capacity failure.
  Graph graph{"runtime-independent-aligned-imports"};
  const ClockKey first_clock = graph.add_clock<ManualClock>("first-capture", std::int64_t{0});
  const ClockKey second_clock = graph.add_clock<ManualClock>("second-capture", std::int64_t{0});
  const ClockKey comparison_clock = graph.add_clock<RealtimeClock>("comparison-clock");
  graph.set_default_clock(comparison_clock);
  graph.add_clock_conversion(first_clock,
                             comparison_clock,
                             AffineClockConversion{
                                 .scale = ClockRateRatio{.numerator = 2, .denominator = 1U},
                                 .offset_ns = 20,
                             });
  graph.add_clock_conversion(
      second_clock, comparison_clock, AffineClockConversion{.offset_ns = 10});

  const auto import = [&](std::string_view endpoint_name, ClockKey clock) {
    TopicOptions options;
    options.endpoint_name = endpoint_name;
    options.capture_time = RequireCaptureTime{.clock = std::move(clock)};
    options.clock_health = ClockHealthOptions{
        .capability = ClockHealthCapability::kStatusReports,
        .requirement = ClockHealthRequirement::kFreshHealthyReport,
        .max_report_age = 5s,
    };
    return graph.import_topic<std::uint32_t>("holomq://" + std::string{endpoint_name},
                                             std::move(options));
  };
  const auto first = import(first_endpoint, first_clock);
  const auto second = import(second_endpoint, second_clock);
  const auto sink = graph.op<IndependentAlignedImportSink>("sink");
  graph.add_flow(first.output(), sink->first, ConnectionOptions{.queue_depth = 4U});
  graph.add_flow(second.output(), sink->second, ConnectionOptions{.queue_depth = 4U});

  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto phase_counters = std::make_shared<detail::RuntimePhaseCounters>();
  const ExecutionPlan instrumented_plan =
      detail::RuntimeTestPlanAccess::with_phase_counters(plan, phase_counters);
  ASSERT_TRUE(instrumented_plan.ok()) << instrumented_plan.json();
  const ClockCatalogView clocks = plan.clock_catalog();
  const auto planned_clock = [&](const ClockKey& key) {
    return std::find_if(clocks.clocks.begin(), clocks.clocks.end(), [&](const auto& candidate) {
      return candidate.key == key;
    });
  };
  const auto planned_first = planned_clock(first_clock);
  const auto planned_second = planned_clock(second_clock);
  const auto planned_comparison = planned_clock(comparison_clock);
  ASSERT_NE(planned_first, clocks.clocks.end());
  ASSERT_NE(planned_second, clocks.clocks.end());
  ASSERT_NE(planned_comparison, clocks.clocks.end());
  ASSERT_EQ(plan.clock_conversions().size(), 2U);
  ASSERT_EQ(plan.aligned_triggers().size(), 1U);
  EXPECT_EQ(plan.aligned_triggers().front().comparison_clock, planned_comparison->id);

  const auto run_once = [&]() -> std::optional<IndependentAlignedImportObservation> {
    IndependentAlignedImportProbe::reset();
    auto external_runtime = ChannelRuntime::create();
    if (!external_runtime) {
      ADD_FAILURE() << runtime_error_message(external_runtime.error());
      return std::nullopt;
    }
    auto first_publisher =
        external_runtime->create_publisher<std::uint32_t>(plan.endpoint_catalog(), first_endpoint);
    auto second_publisher =
        external_runtime->create_publisher<std::uint32_t>(plan.endpoint_catalog(), second_endpoint);
    if (!first_publisher || !second_publisher) {
      ADD_FAILURE() << "independent aligned publisher construction failed: first="
                    << (first_publisher ? "created"
                                        : runtime_error_message(first_publisher.error()))
                    << ", second="
                    << (second_publisher ? "created"
                                         : runtime_error_message(second_publisher.error()));
      return std::nullopt;
    }
    RunSession session = run_async(instrumented_plan);
    const auto first_connected = wait_for_subscriber(*first_publisher);
    const auto second_connected = wait_for_subscriber(*second_publisher);
    if (!first_connected || !second_connected) {
      session.request_stop();
      session.wait();
      ADD_FAILURE() << "independent aligned publishers did not connect: first="
                    << (first_connected ? "connected"
                                        : runtime_error_message(first_connected.error()))
                    << ", second="
                    << (second_connected ? "connected"
                                         : runtime_error_message(second_connected.error()));
      return std::nullopt;
    }

    auto reported = first_publisher->report_clock_health(ClockHealthReport::healthy());
    if (!reported) {
      session.request_stop();
      session.wait();
      ADD_FAILURE() << "first independent clock health failed publication: "
                    << runtime_error_message(reported.error());
      return std::nullopt;
    }
    reported = second_publisher->report_clock_health(ClockHealthReport::healthy());
    if (!reported) {
      session.request_stop();
      session.wait();
      ADD_FAILURE() << "second independent clock health failed publication: "
                    << runtime_error_message(reported.error());
      return std::nullopt;
    }

    const auto witness_retirement_count = [&]() {
      return phase_counters->snapshot()[detail::RuntimePhaseCounters::index(
          detail::RuntimePhaseCounter::kChannelWitnessRetired)];
    };
    const std::uint64_t retirement_baseline = witness_retirement_count();
    auto emitted = second_publisher->emit_captured(2U, 190);
    if (!emitted) {
      session.request_stop();
      session.wait();
      ADD_FAILURE() << "newer independent witness failed publication: "
                    << runtime_error_message(emitted.error());
      return std::nullopt;
    }
    const auto retirement_deadline = std::chrono::steady_clock::now() + 5s;
    while (witness_retirement_count() == retirement_baseline && session.is_running() &&
           std::chrono::steady_clock::now() < retirement_deadline) {
      std::this_thread::yield();
    }
    if (witness_retirement_count() == retirement_baseline) {
      session.request_stop();
      session.wait();
      ADD_FAILURE() << "newer independent witness was not retired before late visibility";
      return std::nullopt;
    }
    EXPECT_TRUE(session.is_running());
    EXPECT_FALSE(IndependentAlignedImportProbe::observed());

    emitted = first_publisher->emit_captured(1U, 40);
    if (!emitted) {
      session.request_stop();
      session.wait();
      ADD_FAILURE() << "late independent witness failed publication: "
                    << runtime_error_message(emitted.error());
      return std::nullopt;
    }
    auto observation = IndependentAlignedImportProbe::wait_for_observation();
    session.request_stop();
    session.wait();
    EXPECT_EQ(session.termination(), RunTermination::kStopped);
    EXPECT_TRUE(session.diagnostics().empty())
        << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
    return observation;
  };

  const auto first_run = run_once();
  const auto second_run = run_once();
  if (!first_run.has_value() || !second_run.has_value()) {
    ADD_FAILURE() << "independent aligned import omitted deterministic replay evidence";
    return;
  }
  const auto& first_observation = *first_run;
  const auto& second_observation = *second_run;

  const auto expect_exact_observation = [&](const auto& observation) {
    EXPECT_EQ(observation.first_value, 1U);
    EXPECT_EQ(observation.second_value, 2U);
    EXPECT_EQ(observation.first_metadata.capture_timestamp_ns, 40);
    EXPECT_EQ(observation.first_metadata.source_clock_id, planned_first->id);
    EXPECT_EQ(observation.second_metadata.capture_timestamp_ns, 190);
    EXPECT_EQ(observation.second_metadata.source_clock_id, planned_second->id);
    EXPECT_EQ(observation.tag_time, (TagTime{200, 1U, 0U}));
    EXPECT_EQ(observation.activation_time, (TimePoint{100, planned_comparison->id}));
  };
  expect_exact_observation(first_observation);
  expect_exact_observation(second_observation);
  EXPECT_EQ(first_observation.tag_time, second_observation.tag_time);
  EXPECT_EQ(first_observation.first_metadata.capture_timestamp_ns,
            second_observation.first_metadata.capture_timestamp_ns);
  EXPECT_EQ(first_observation.second_metadata.capture_timestamp_ns,
            second_observation.second_metadata.capture_timestamp_ns);
}

}  // namespace
}  // namespace holoscan
