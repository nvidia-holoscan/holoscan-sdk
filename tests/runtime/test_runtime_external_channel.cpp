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
#include <future>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include <holoscan/channel/channel.hpp>
#include <holoscan/core/compile.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/schema/descriptor_flatbuffer_payload.hpp>
#include <holoscan/schema/sdk_schemas.hpp>
#include <holoscan/sensor_io/image_metadata.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>
#include <holoscan/time/manual_clock.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "../common/cuda_test_utils.hpp"
#include "image_generated.h"

#include "channel/channel_common.hpp"
#include "channel/channel_tensor_test_access.hpp"
#include "channel/endpoint_catalog_state.hpp"
#include "channel/process_identity.hpp"
#include "compiler/compile_internal.hpp"
#include "compiler/process_placement_internal.hpp"
#include "compiler/runtime_package_capabilities.hpp"
#include "core/diagnostic_registry.hpp"
#include "core/graph_compiler_access.hpp"
#include "runtime/channel_runtime_binding_access.hpp"
#include "runtime/endpoint_clock_health_runtime.hpp"
#include "runtime/graph_exec.hpp"
#include "runtime/runtime_channel_binding_test_access.hpp"
#include "runtime/runtime_phase_counters.hpp"
#include "runtime/runtime_support_thread_control.hpp"
#include "runtime/runtime_test_access.hpp"
#include "runtime/runtime_work_inventory.hpp"
#include "runtime/runtime_worker.hpp"
#include "runtime_test_graph.hpp"

namespace holoscan {
namespace {

using namespace std::chrono_literals;

// Records cold Runtime support-thread entry so external-import tests can distinguish the
// controller-owned catalog drain from the process-local HoloMQ pump.
class ExternalImportSupportThreadProbe final : public detail::RuntimeSupportThreadControl {
 public:
  [[nodiscard]] expected<void, Error> enter(const detail::RuntimeSupportThreadRole role) override {
    entered_[static_cast<std::size_t>(role)].store(true, std::memory_order_release);
    return {};
  }

  [[nodiscard]] bool entered(const detail::RuntimeSupportThreadRole role) const noexcept {
    return entered_[static_cast<std::size_t>(role)].load(std::memory_order_acquire);
  }

 private:
  std::array<std::atomic_bool, 2U> entered_{};
};

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

// Collects up to kCapacity samples received by ExternalImportSink for post-run assertion.
// Captures data values and sequence numbers to verify ordering and metadata preservation.
// Representative test: UnmonitoredScalarImportDrainsEverySampleInPublisherOrder.
struct ExternalImportProbe {
  static constexpr std::size_t kCapacity = 4U;

  static void reset() noexcept {
    const std::lock_guard lock{mutex};
    values.fill(0U);
    sequences.fill(0U);
    count.store(0U, std::memory_order_release);
  }

  static void record(const Sample<std::uint32_t>& sample) noexcept {
    const std::lock_guard lock{mutex};
    const std::size_t index = count.load(std::memory_order_relaxed);
    if (index < kCapacity) {
      values[index] = sample.data;
      sequences[index] = sample.metadata.sequence;
      count.store(index + 1U, std::memory_order_release);
    }
    ready.notify_all();
  }

  [[nodiscard]] static bool wait_for_count(std::size_t expected,
                                           std::chrono::milliseconds timeout = 5s) {
    std::unique_lock lock{mutex};
    return ready.wait_for(
        lock, timeout, [expected] { return count.load(std::memory_order_acquire) >= expected; });
  }

  static inline std::mutex mutex;
  static inline std::condition_variable ready;
  static inline std::atomic<std::size_t> count{};
  static inline std::array<std::uint32_t, kCapacity> values{};
  static inline std::array<std::uint64_t, kCapacity> sequences{};
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
        .deadline(1ns)
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

// Records whether compute was ever called on a graph where an external topic exists only
// in the catalog with no wired edge to any operator.
// Representative test: DisconnectedImportRemainsCatalogOnlyAtRuntime.
struct CatalogOnlyRuntimeProbe {
  static void reset() noexcept { compute_count.store(0U, std::memory_order_relaxed); }

  [[nodiscard]] static bool wait_for_compute(std::chrono::milliseconds timeout = 5s) {
    std::unique_lock lock{mutex};
    return ready.wait_for(
        lock, timeout, [] { return compute_count.load(std::memory_order_acquire) != 0U; });
  }

  static inline std::mutex mutex;
  static inline std::condition_variable ready;
  static inline std::atomic<std::uint32_t> compute_count{};
};

// Clock-triggered operator with no inputs or outputs. Proves that a catalog-registered
// external topic does not block execution when it has no wired edges.
// Used by: DisconnectedImportRemainsCatalogOnlyAtRuntime,
//          DisconnectedExportRemainsCatalogOnlyAtRuntime.
class CatalogOnlyClockOperator final : public Operator<> {
 public:
  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 30s});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    CatalogOnlyRuntimeProbe::compute_count.fetch_add(1U, std::memory_order_release);
    CatalogOnlyRuntimeProbe::ready.notify_all();
    return {};
  }
};

// Receives uint32_t samples from an imported external topic and records each one via
// ExternalImportProbe for value and ordering assertions after the session ends.
// Representative test: UnmonitoredScalarImportDrainsEverySampleInPublisherOrder.
class ExternalImportSink final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override { spec.input(input, "input").queue_depth(8U); }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnAnyInput{})
        .input(input, LatestWithin{.max_age = 1s})
        .on_missing(input, Skip{});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    auto sample = input.receive();
    if (!sample) {
      return make_unexpected(std::move(sample.error()));
    }
    ExternalImportProbe::record(*sample);
    return {};
  }

  Input<std::uint32_t> input;
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

// Source that gates emission on a test-controlled rendezvous: blocks in compute until
// allow_emit() is called, then sends a fixed scalar payload with optional capture-time
// metadata. Representative test: CanonicalScalarGraphExportReachesExternalSubscriber.
class ExternalScalarExportSource final : public Operator<> {
 public:
  static void reset() noexcept {
    const std::lock_guard lock{coordination_mutex};
    compute_started = false;
    emit_allowed = false;
    emit_completed = false;
    compute_count.store(0U, std::memory_order_relaxed);
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
    spec.output(output, "output").max_emits_per_compute(1U);
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 50ms});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    compute_count.fetch_add(1U, std::memory_order_relaxed);
    std::optional<TimePoint> selected_capture;
    {
      std::unique_lock lock{coordination_mutex};
      compute_started = true;
      coordination_ready.notify_all();
      if (!coordination_ready.wait_for(lock, 5s, [] { return emit_allowed; })) {
        emit_error.store(ErrorCode::kTimeout, std::memory_order_relaxed);
        return make_unexpected(Error{ErrorCode::kTimeout});
      }
      selected_capture = capture_time;
    }
    SampleMetadata metadata{.trace_id = 17U, .frame_id = 23U};
    if (selected_capture) {
      metadata.capture_timestamp_ns = selected_capture->timestamp_ns;
      metadata.source_clock_id = selected_capture->clock_id;
    }
    auto emitted = output.emit(Sample<std::uint32_t>{
        0xdecafbadU,
        metadata,
    });
    if (!emitted) {
      emit_error.store(emitted.error().code, std::memory_order_relaxed);
    }
    {
      const std::lock_guard lock{coordination_mutex};
      emit_completed = true;
    }
    coordination_ready.notify_all();
    return emitted;
  }

  Output<std::uint32_t> output;
  static inline std::atomic<std::uint32_t> compute_count{};
  static inline std::atomic<ErrorCode> emit_error{ErrorCode::kSuccess};
  static inline std::mutex coordination_mutex;
  static inline std::condition_variable coordination_ready;
  static inline bool compute_started{};
  static inline bool emit_allowed{};
  static inline bool emit_completed{};
  static inline std::optional<TimePoint> capture_time;
};

constexpr DLDataType kExternalTensorDtype{kDLUInt, 8U, 1U};
constexpr std::array<std::int64_t, 1U> kExternalTensorShape{8};
constexpr std::array<TensorAxisBounds, 1U> kExternalTensorAxes{
    TensorAxisBounds{.max_extent = 8U, .max_abs_stride_bytes = 1U}};

using ExternalImagePayload = schema::DescriptorFlatBufferPayloadFor<schema::Image>;
using WrongExternalImagePayload = schema::DescriptorFlatBufferPayloadFor<schema::TensorPacket>;
constexpr TensorFieldSelector<schema::ImageT> kExternalImageData{4U, 0U};
constexpr std::array<std::int64_t, 2U> kExternalImageShape{1, 8};

[[nodiscard]] expected<Tensor, Error> make_external_image_tensor(const std::uint8_t value) {
  auto* storage = new (std::nothrow) std::uint8_t[8U];
  if (storage == nullptr) {
    return make_unexpected(Error{ErrorCode::kResourceExhausted});
  }
  std::fill_n(storage, 8U, value);
  auto tensor = make_tensor(kExternalImageShape,
                            kExternalTensorDtype,
                            std::nullopt,
                            MemoryStorageType::kSystem,
                            storage,
                            [](void* pointer) -> expected<void, Error> {
                              delete[] static_cast<std::uint8_t*>(pointer);
                              return {};
                            });
  if (!tensor) {
    delete[] storage;
    return make_unexpected(std::move(tensor).error());
  }
  return tensor;
}

[[nodiscard]] DescriptorTensorTopicContract<schema::ImageT> external_image_contract() {
  DescriptorTensorTopicContract<schema::ImageT> result;
  result.field(kExternalImageData,
               TensorBoundaryContract{
                   .representation = {.memory_kind = MemoryKind::kHost,
                                      .dtype = kExternalTensorDtype,
                                      .rank = 2U,
                                      .strides = TensorStridePolicy::kContiguousOnly},
                   .bounds = tensor_bounds(kExternalImageShape),
                   .transfer = TensorTransferPolicy{.foreign_emit_policy =
                                                        InterfaceTensorEmitPolicy::kCopyIfNeeded},
               });
  return result;
}

[[nodiscard]] TopicOptions external_image_import_options(std::string_view endpoint_name) {
  TopicOptions options;
  options.endpoint_name = endpoint_name;
  options.bounds.publisher = PublisherEndpointBounds{.max_loaned_samples = 1U};
  return options;
}

[[nodiscard]] TopicOptions external_image_export_options(std::string_view endpoint_name) {
  TopicOptions options;
  options.endpoint_name = endpoint_name;
  options.bounds.subscriber =
      SubscriberEndpointBounds{.queue_depth = 1U, .max_borrowed_samples = 1U};
  return options;
}

struct ExternalImageImportProbe {
  static void reset() noexcept {
    count.store(0U, std::memory_order_release);
    height.store(0U, std::memory_order_relaxed);
    width.store(0U, std::memory_order_relaxed);
    first_byte.store(0U, std::memory_order_relaxed);
    capture_timestamp_ns.store(0, std::memory_order_relaxed);
    source_clock_id.store(0U, std::memory_order_relaxed);
    trace_id.store(0U, std::memory_order_relaxed);
    frame_id.store(0U, std::memory_order_relaxed);
    flags.store(0U, std::memory_order_relaxed);
    sequence.store(0U, std::memory_order_relaxed);
  }

  static void record(const Sample<schema::ImageT>& sample) noexcept {
    height.store(sample.data.height, std::memory_order_relaxed);
    width.store(sample.data.width, std::memory_order_relaxed);
    const auto* bytes = sample.data.data ? sample.data.data->data_as<std::uint8_t>() : nullptr;
    first_byte.store(bytes == nullptr ? 0U : bytes[0], std::memory_order_relaxed);
    capture_timestamp_ns.store(sample.metadata.capture_timestamp_ns, std::memory_order_relaxed);
    source_clock_id.store(sample.metadata.source_clock_id.value, std::memory_order_relaxed);
    trace_id.store(sample.metadata.trace_id, std::memory_order_relaxed);
    frame_id.store(sample.metadata.frame_id, std::memory_order_relaxed);
    flags.store(static_cast<std::uint32_t>(sample.metadata.flags), std::memory_order_relaxed);
    sequence.store(sample.metadata.sequence, std::memory_order_relaxed);
    count.fetch_add(1U, std::memory_order_release);
    ready.notify_all();
  }

  [[nodiscard]] static bool wait_for_count(std::uint32_t expected,
                                           std::chrono::milliseconds timeout = 5s) {
    std::unique_lock lock{mutex};
    return ready.wait_for(
        lock, timeout, [expected] { return count.load(std::memory_order_acquire) >= expected; });
  }

  [[nodiscard]] static bool wait_for_sample(std::chrono::milliseconds timeout = 5s) {
    return wait_for_count(1U, timeout);
  }

  static inline std::mutex mutex;
  static inline std::condition_variable ready;
  static inline std::atomic<std::uint32_t> count{};
  static inline std::atomic<std::uint32_t> height{};
  static inline std::atomic<std::uint32_t> width{};
  static inline std::atomic<std::uint8_t> first_byte{};
  static inline std::atomic<std::int64_t> capture_timestamp_ns{};
  static inline std::atomic<std::uint64_t> source_clock_id{};
  static inline std::atomic<std::uint64_t> trace_id{};
  static inline std::atomic<std::uint64_t> frame_id{};
  static inline std::atomic<std::uint32_t> flags{};
  static inline std::atomic<std::uint64_t> sequence{};
};

class ExternalImageImportSink final : public Operator<> {
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
    result.trigger(OnEach{input});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    auto sample = input.receive();
    if (!sample) {
      return make_unexpected(std::move(sample).error());
    }
    ExternalImageImportProbe::record(*sample);
    return {};
  }

  Input<schema::ImageT> input;
};

// Keeps an output edge live while an externally imported descriptor-schema input is awaiting a
// lifecycle restart.  This is intentionally a forwarding operator rather than a terminal sink:
// closing its output during the first generation's terminal drain would reject the restart.
class ExternalImageForwarder final : public Operator<> {
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

  Input<schema::ImageT> input;
  Output<schema::ImageT> output;
};

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
        .deadline(30s)
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

struct CatalogScalarReuseProbe {
  static void reset() noexcept {
    const std::lock_guard lock{mutex};
    count.store(0U, std::memory_order_release);
    values.fill(0U);
    sequences.fill(0U);
    states.fill(InputState::kAbsent);
  }

  static void record(const Sample<std::uint32_t>& sample, const InputState state) noexcept {
    const std::lock_guard lock{mutex};
    const std::size_t index = count.load(std::memory_order_relaxed);
    if (index < values.size()) {
      values[index] = sample.data;
      sequences[index] = sample.metadata.sequence;
      states[index] = state;
    }
    count.store(index + 1U, std::memory_order_release);
    ready.notify_all();
  }

  [[nodiscard]] static bool wait_for_count(const std::size_t expected) {
    std::unique_lock lock{mutex};
    return ready.wait_for(
        lock, 5s, [expected] { return count.load(std::memory_order_acquire) >= expected; });
  }

  static inline std::mutex mutex;
  static inline std::condition_variable ready;
  static inline std::atomic<std::size_t> count{};
  static inline std::array<std::uint32_t, 2U> values{};
  static inline std::array<std::uint64_t, 2U> sequences{};
  static inline std::array<InputState, 2U> states{};
};

class CatalogScalarReuseSink final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override { spec.input(input, "input").queue_depth(1U); }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnAnyInput{})
        .input(input, LatestWithin{.max_age = 30s})
        .on_missing(input, ReuseLast{.max_age = 30s});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    auto sample = input.receive();
    if (!sample) {
      return make_unexpected(std::move(sample).error());
    }
    CatalogScalarReuseProbe::record(*sample, input.selection().state);
    return {};
  }

  Input<std::uint32_t> input;
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

// Internal sink that receives Tensor samples from a fanout edge. Verifies that a dropped
// optional external export does not suppress delivery on the internal path.
// Representative test: OptionalTensorDropDoesNotPoisonAcceptedInternalFanout.
class ExternalTensorExportSink final : public Operator<> {
 public:
  static void reset() noexcept {
    received.store(0U, std::memory_order_relaxed);
    last_sequence.store(0U, std::memory_order_relaxed);
  }

  [[nodiscard]] static bool wait_for_sample(std::chrono::milliseconds timeout = 5s) {
    std::unique_lock lock{mutex};
    return ready.wait_for(
        lock, timeout, [] { return received.load(std::memory_order_acquire) != 0U; });
  }

  [[nodiscard]] static bool wait_for_count(const std::uint32_t expected,
                                           std::chrono::milliseconds timeout = 5s) {
    std::unique_lock lock{mutex};
    return ready.wait_for(
        lock, timeout, [expected] { return received.load(std::memory_order_acquire) >= expected; });
  }

  void setup(OperatorSpec& spec) override { spec.input(input, "input").queue_depth(1U); }

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
    last_sequence.store(sample->metadata.sequence, std::memory_order_relaxed);
    received.fetch_add(1U, std::memory_order_release);
    ready.notify_all();
    return {};
  }

  Input<Tensor> input;
  static inline std::atomic<std::uint32_t> received{};
  static inline std::atomic<std::uint64_t> last_sequence{};
  static inline std::mutex mutex;
  static inline std::condition_variable ready;
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

// Builds TopicOptions for an optional host Tensor export with fixed 8-byte single-axis
// bounds. Shared across optional-tensor tests to avoid repeating boilerplate.
[[nodiscard]] TopicOptions optional_tensor_export_options(std::string_view endpoint_name) {
  TopicOptions options;
  options.endpoint_name = endpoint_name;
  options.bounds.subscriber =
      SubscriberEndpointBounds{.queue_depth = 1U, .max_borrowed_samples = 1U};
  options.tensor = TensorBoundaryContract{
      .representation = {.memory_kind = MemoryKind::kHost,
                         .dtype = kExternalTensorDtype,
                         .rank = 1U,
                         .strides = TensorStridePolicy::kContiguousOnly},
      .bounds =
          TensorBounds{
              .max_rank = 1U,
              .max_byte_span = 8U,
              .axes = kExternalTensorAxes,
          },
      .transfer =
          TensorTransferPolicy{.foreign_emit_policy = InterfaceTensorEmitPolicy::kCopyIfNeeded},
  };
  return options;
}

// Returns a process-, time-, and counter-stamped endpoint name to prevent resource collisions
// between independently launched test processes sharing the same HoloMQ daemon.
[[nodiscard]] std::string unique_endpoint_name() {
  static std::atomic<std::uint64_t> ordinal{};
  return "runtime_external_import_" + std::to_string(channel_internal::current_process_identity()) +
         "_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + "_" +
         std::to_string(ordinal.fetch_add(1U, std::memory_order_relaxed));
}

// Polls update_connections() until the publisher observes at least one subscriber or timeout.
// Called after run_async() to synchronize before the external test publisher emits.
template <PublishedPayload T>
[[nodiscard]] expected<void, Error> wait_for_subscriber(
    ChannelPublisher<T>& publisher, std::chrono::milliseconds timeout = 5s,
    std::chrono::milliseconds stable_for = 0ms) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  std::optional<std::chrono::steady_clock::time_point> connected_since;
  while (std::chrono::steady_clock::now() < deadline) {
    auto refreshed = publisher.update_connections();
    if (!refreshed) {
      return make_unexpected(std::move(refreshed).error());
    }
    if (publisher.has_subscribers()) {
      const auto now = std::chrono::steady_clock::now();
      connected_since = connected_since.value_or(now);
      if (now - *connected_since >= stable_for) {
        return {};
      }
    } else {
      connected_since.reset();
    }
    std::this_thread::sleep_for(1ms);
  }
  return make_unexpected(Error{ErrorCode::kTimeout});
}

[[nodiscard]] expected<void, Error> wait_for_catalog_tensor_subscriber(
    ChannelPublisher<Tensor>& publisher, ChannelSubscriber<Tensor>& subscriber,
    std::chrono::milliseconds timeout = 5s) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  std::uint32_t consecutive_observations = 0U;
  while (std::chrono::steady_clock::now() < deadline) {
    auto receiver_refreshed = subscriber.has_samples();
    if (!receiver_refreshed) {
      return make_unexpected(std::move(receiver_refreshed).error());
    }
    auto publisher_refreshed = publisher.update_connections();
    if (!publisher_refreshed) {
      return make_unexpected(std::move(publisher_refreshed).error());
    }
    if (subscriber.has_publishers() && publisher.has_subscribers()) {
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

[[nodiscard]] expected<void, Error> publish_external_host_tensor(
    ChannelPublisher<Tensor>& publisher, std::optional<TimePoint> capture_time,
    const std::byte fill = std::byte{0x2a}) {
  auto loan = publisher.loan_tensor(
      TensorLoanRequest{.shape = kExternalTensorShape, .dtype = kExternalTensorDtype});
  if (!loan) {
    return make_unexpected(std::move(loan).error());
  }
  auto writer = loan->write_host();
  if (!writer) {
    return make_unexpected(std::move(writer).error());
  }
  std::fill_n(static_cast<std::byte*>(writer->data()), writer->byte_size(), fill);
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

template <ChannelDescriptorFlatBufferPayload Payload>
[[nodiscard]] expected<void, Error> wait_for_schema_subscriber(
    ChannelTensorSchemaPublisher<Payload>& publisher, std::chrono::milliseconds timeout = 5s,
    std::chrono::milliseconds stable_for = 0ms) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  std::optional<std::chrono::steady_clock::time_point> connected_since;
  while (std::chrono::steady_clock::now() < deadline) {
    auto refreshed = publisher.update_connections();
    if (!refreshed) {
      return make_unexpected(std::move(refreshed).error());
    }
    if (publisher.has_subscribers()) {
      const auto now = std::chrono::steady_clock::now();
      connected_since = connected_since.value_or(now);
      if (now - *connected_since >= stable_for) {
        return {};
      }
    } else {
      connected_since.reset();
    }
    std::this_thread::sleep_for(1ms);
  }
  return make_unexpected(Error{ErrorCode::kTimeout});
}

template <ChannelDescriptorFlatBufferPayload Payload>
[[nodiscard]] expected<void, Error> wait_for_catalog_schema_subscriber(
    ChannelTensorSchemaPublisher<Payload>& publisher,
    channel_internal::TensorSchemaPayloadSubscriber& subscriber,
    std::chrono::milliseconds timeout = 5s) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  std::uint32_t consecutive_observations = 0U;
  while (std::chrono::steady_clock::now() < deadline) {
    auto receiver_refreshed = subscriber.has_samples();
    if (!receiver_refreshed) {
      return make_unexpected(std::move(receiver_refreshed).error());
    }
    auto publisher_refreshed = publisher.update_connections();
    if (!publisher_refreshed) {
      return make_unexpected(std::move(publisher_refreshed).error());
    }
    if (subscriber.has_publishers() && publisher.has_subscribers()) {
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

template <typename Publisher>
[[nodiscard]] expected<void, Error> wait_for_no_subscribers(
    Publisher& publisher, std::chrono::milliseconds timeout = 5s) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    auto refreshed = publisher.update_connections();
    if (!refreshed) {
      return make_unexpected(std::move(refreshed).error());
    }
    if (!publisher.has_subscribers()) {
      return {};
    }
    std::this_thread::sleep_for(1ms);
  }
  return make_unexpected(Error{ErrorCode::kTimeout});
}

[[nodiscard]] expected<channel_internal::CatalogReceivedOperation, Error>
wait_for_catalog_tensor_operation(ChannelSubscriber<Tensor>& subscriber,
                                  std::chrono::milliseconds timeout = 5s) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    auto operation =
        detail::RuntimeChannelBindingAccess::receive_catalog_tensor_operation(subscriber, true);
    if (!operation || operation->kind != channel_internal::CatalogReceivedOperationKind::kNone) {
      return operation;
    }
    std::this_thread::sleep_for(1ms);
  }
  return make_unexpected(Error{ErrorCode::kTimeout});
}

[[nodiscard]] expected<channel_internal::CatalogReceivedOperation, Error>
wait_for_catalog_tensor_schema_operation(
    channel_internal::TensorSchemaPayloadSubscriber& subscriber,
    std::chrono::milliseconds timeout = 5s) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    auto operation = detail::RuntimeChannelBindingAccess::receive_catalog_tensor_schema_operation(
        subscriber, true);
    if (!operation || operation->kind != channel_internal::CatalogReceivedOperationKind::kNone) {
      return operation;
    }
    std::this_thread::sleep_for(1ms);
  }
  return make_unexpected(Error{ErrorCode::kTimeout});
}

template <ChannelDescriptorFlatBufferPayload Payload>
[[nodiscard]] expected<channel_internal::CatalogReceivedOperation, Error>
wait_for_catalog_tensor_schema_operation(ChannelTensorSchemaSubscriber<Payload>& subscriber,
                                         std::chrono::milliseconds timeout = 5s) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    auto operation = detail::RuntimeChannelBindingAccess::receive_catalog_tensor_schema_operation(
        subscriber, true);
    if (!operation || operation->kind != channel_internal::CatalogReceivedOperationKind::kNone) {
      return operation;
    }
    std::this_thread::sleep_for(1ms);
  }
  return make_unexpected(Error{ErrorCode::kTimeout});
}

[[nodiscard]] bool wait_for_session_terminal(const RunSession& session,
                                             std::chrono::milliseconds timeout = 5s) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (session.is_running() && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(1ms);
  }
  return !session.is_running();
}

[[nodiscard]] detail::ProcessPlacementBinding external_endpoint_process_placement(
    std::string_view endpoint_uri) {
  const std::array selectors{detail::GraphObjectSelector{.external_topic = endpoint_uri}};
  detail::ProcessPlacementBinding placement;
  placement.bind_process_role("external-endpoint",
                              detail::ProcessPlacement{
                                  .isolation = detail::ProcessIsolation::kExternalProcess,
                                  .members = selectors,
                                  .launch_group = "external-endpoint",
                              });
  return placement;
}

void expect_completion_progress(const RunSession& session, const std::uint64_t limit) {
  ASSERT_EQ(session.operator_completion_progress().size(), 1U);
  const auto& progress = session.operator_completion_progress().front();
  EXPECT_EQ(progress.configured_limit, limit);
  EXPECT_EQ(progress.successful_compute_calls, limit);
  EXPECT_TRUE(progress.limit_reached);
}

// Polls has_samples() until a publisher is seen on two consecutive checks, guarding against
// a transient first observation before the Runtime-owned publisher is fully live.
template <typename T>
[[nodiscard]] expected<void, Error> wait_for_publisher(ChannelSubscriber<T>& subscriber,
                                                       std::chrono::milliseconds timeout = 5s) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  std::uint32_t consecutive_observations = 0U;
  while (std::chrono::steady_clock::now() < deadline) {
    // Refresh receiver-side mappings without consuming data, then require two observations while
    // the subscriber remains alive. The advisory Boolean coordinates this bounded test handoff;
    // it is not treated as proof of the later publication result.
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

// Polls has_samples() and drains one sample when ready, retrying until the deadline.
// Returns kTimeout if no sample arrives within the window.
[[nodiscard]] expected<Sample<std::uint32_t>, Error> wait_for_sample(
    ChannelSubscriber<std::uint32_t>& subscriber, std::chrono::milliseconds timeout = 5s) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    auto ready = subscriber.has_samples();
    if (!ready) {
      return make_unexpected(std::move(ready.error()));
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

// REQ: CHAN-073
// REQ: RUNTIME-015
TEST(RuntimeExternalChannelTest, DisconnectedImportRemainsCatalogOnlyAtRuntime) {
  CatalogOnlyRuntimeProbe::reset();
  //   holomq://endpoint  (catalog-only, no edge to any operator)
  //   independent        (OnClock period=30s, no inputs/outputs)
  Graph graph{"runtime-disconnected-external-import"};
  (void)graph.import_topic<std::uint32_t>("holomq://" + unique_endpoint_name());
  (void)graph.op<CatalogOnlyClockOperator>("independent");
  const ClockKey default_clock = graph.add_clock<RealtimeClock>("runtime-clock");
  graph.set_default_clock(default_clock);
  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  // Key setup: external topic registered in catalog but not connected to any operator.
  ASSERT_EQ(plan.graph().external_topics().size(), 1U);
  ASSERT_TRUE(plan.payload_edges().empty());

  RunSession session = run_async(plan);
  const bool computed = CatalogOnlyRuntimeProbe::wait_for_compute();
  session.request_stop();
  session.wait();

  // Key assertion: graph runs normally despite the catalog-only import topic.
  EXPECT_TRUE(computed);
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
}

// REQ: CHAN-073
// REQ: RUNTIME-015
TEST(RuntimeExternalChannelTest, RejectsExternalTopicWithoutEndpointCatalogAuthority) {
  // Graph topology:
  //   import topic    independent clock-triggered operator  (no connecting edge)
  Graph graph{"runtime-external-topic-without-catalog"};
  (void)graph.import_topic<std::uint32_t>("holomq://" + unique_endpoint_name());
  (void)graph.op<CatalogOnlyClockOperator>("independent");
  const ClockKey default_clock = graph.add_clock<RealtimeClock>("runtime-clock");
  graph.set_default_clock(default_clock);
  const ExecutionPlan compiled = compile(graph);
  ASSERT_TRUE(compiled.ok()) << compiled.json();
  ASSERT_EQ(compiled.graph().external_topics().size(), 1U);

  // Key setup: retain the authored topic while removing its immutable catalog authority.
  const ExecutionPlan malformed = detail::RuntimeTestPlanAccess::remove_endpoint_catalog(compiled);
  ASSERT_TRUE(malformed.ok()) << malformed.json();
  const auto state = detail::ExecutionPlanAccess::retain(malformed);
  ASSERT_NE(state, nullptr);

  const auto reconstructed = detail::GraphExec::reconstruct(state, RunGenerationId{1U});

  // Key assertion: replay fails closed before consulting Runtime external bindings.
  ASSERT_FALSE(reconstructed);
  EXPECT_EQ(reconstructed.error().code, ErrorCode::kNotSupported);
  EXPECT_EQ(reconstructed.error().message, "RUNTIME_EXTERNAL_BINDING_UNAVAILABLE");
}

// REQ: CHAN-073
// REQ: RUNTIME-015
TEST(RuntimeExternalChannelTest, DisconnectedExportRemainsCatalogOnlyAtRuntime) {
  CatalogOnlyRuntimeProbe::reset();
  //   holomq://endpoint  (catalog-only, no edge from any operator)
  //   independent        (OnClock period=30s, no inputs/outputs)
  Graph graph{"runtime-disconnected-external-export"};
  (void)graph.export_topic<std::uint32_t>("holomq://" + unique_endpoint_name());
  (void)graph.op<CatalogOnlyClockOperator>("independent");
  const ClockKey default_clock = graph.add_clock<RealtimeClock>("runtime-clock");
  graph.set_default_clock(default_clock);
  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  // Key setup: external topic registered in catalog but not connected to any operator.
  ASSERT_EQ(plan.graph().external_topics().size(), 1U);
  ASSERT_TRUE(plan.payload_edges().empty());

  RunSession session = run_async(plan);
  const bool computed = CatalogOnlyRuntimeProbe::wait_for_compute();
  session.request_stop();
  session.wait();

  // Key assertion: graph runs normally despite the catalog-only export topic.
  EXPECT_TRUE(computed);
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
}

// An endpoint-only include_all() plan retains one topology lane without inventing a callback
// owner. Runtime must accept that lane and complete the catalog-only application as a no-op.
// REQ: CHAN-073
// REQ: COMPILE-068
// REQ: GRAPH-032
// REQ: RUNTIME-015
// REQ: RUNTIME-041
TEST(RuntimeExternalChannelTest,
     IncludeAllStandaloneImportRetainsTopologyLaneWithoutCallbackOwner) {
  // Graph topology:
  //   standalone import topic  (included in the application partition)
  Graph graph{"runtime-include-all-standalone-import"};
  (void)graph.import_topic<std::uint32_t>("holomq://" + unique_endpoint_name());
  graph.partition("application").include_all();

  // Key setup: include the endpoint-only topology in one authored partition.
  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.execution_partitions().size(), 1U);
  EXPECT_EQ(plan.execution_partitions().front().authored_name,
            std::optional<std::string>{"application"});
  ASSERT_EQ(plan.execution_partitions().front().members.size(), 1U);
  EXPECT_TRUE(plan.execution_partitions().front().operator_paths.empty());
  ASSERT_EQ(plan.execution_lanes().size(), 1U);
  EXPECT_EQ(plan.execution_lanes().front().partition, plan.execution_partitions().front().id);
  EXPECT_EQ(plan.scheduler_plan().execution_lane_count, 1U);
  EXPECT_EQ(plan.scheduler_plan().worker_count, 0U);
  EXPECT_EQ(plan.scheduler_plan().max_concurrent_activations, 0U);
  EXPECT_TRUE(plan.payload_edges().empty());
  EXPECT_TRUE(plan.schedule_edges().empty());
  const auto accepted = detail::ExecutionPlanAccess::retain(plan);
  ASSERT_NE(accepted, nullptr);
  EXPECT_TRUE(detail::has_structural_runtime_topology(*accepted));
  EXPECT_FALSE(detail::has_plan_callback_owners(*accepted));
  auto reconstructed = detail::GraphExec::reconstruct(accepted, RunGenerationId{17U});
  ASSERT_TRUE(reconstructed) << reconstructed.error().message;
  EXPECT_FALSE((*reconstructed)->has_process_local_callback_owners());

  RunSession session = run_async(plan);
  const bool completed = wait_for_session_terminal(session);
  if (!completed) {
    session.request_stop();
  }
  session.wait();

  // Key assertion: the structural lane has no callback owner and completes as a no-op.
  EXPECT_TRUE(completed);
  EXPECT_EQ(session.termination(), RunTermination::kCompleted);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
}

// REQ: CHAN-073
// REQ: COMPILE-068
// REQ: GRAPH-032
// REQ: RUNTIME-015
// REQ: RUNTIME-041
TEST(RuntimeExternalChannelTest,
     IncludeAllStandaloneExportRetainsTopologyLaneWithoutCallbackOwner) {
  // Graph topology:
  //   standalone export topic  (included in the application partition)
  Graph graph{"runtime-include-all-standalone-export"};
  (void)graph.export_topic<std::uint32_t>("holomq://" + unique_endpoint_name());
  graph.partition("application").include_all();

  // Key setup: include the endpoint-only topology in one authored partition.
  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.execution_partitions().size(), 1U);
  EXPECT_EQ(plan.execution_partitions().front().authored_name,
            std::optional<std::string>{"application"});
  ASSERT_EQ(plan.execution_partitions().front().members.size(), 1U);
  EXPECT_TRUE(plan.execution_partitions().front().operator_paths.empty());
  ASSERT_EQ(plan.execution_lanes().size(), 1U);
  EXPECT_EQ(plan.execution_lanes().front().partition, plan.execution_partitions().front().id);
  EXPECT_EQ(plan.scheduler_plan().execution_lane_count, 1U);
  EXPECT_EQ(plan.scheduler_plan().worker_count, 0U);
  EXPECT_EQ(plan.scheduler_plan().max_concurrent_activations, 0U);
  EXPECT_TRUE(plan.payload_edges().empty());
  EXPECT_TRUE(plan.schedule_edges().empty());
  const auto accepted = detail::ExecutionPlanAccess::retain(plan);
  ASSERT_NE(accepted, nullptr);
  EXPECT_TRUE(detail::has_structural_runtime_topology(*accepted));
  EXPECT_FALSE(detail::has_plan_callback_owners(*accepted));
  auto reconstructed = detail::GraphExec::reconstruct(accepted, RunGenerationId{17U});
  ASSERT_TRUE(reconstructed) << reconstructed.error().message;
  EXPECT_FALSE((*reconstructed)->has_process_local_callback_owners());

  RunSession session = run_async(plan);
  const bool completed = wait_for_session_terminal(session);
  if (!completed) {
    session.request_stop();
  }
  session.wait();

  // Key assertion: the structural lane has no callback owner and completes as a no-op.
  EXPECT_TRUE(completed);
  EXPECT_EQ(session.termination(), RunTermination::kCompleted);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
}

// Interface-only topology retains authored clocks as authenticated plan evidence, but no callback
// owner can observe a run-owned clock instance. Cover both external-topic directions.
//
// Graph topology (one case per direction):
//   imported-topic   or   exported-topic   (no Operator callback owner)
// REQ: RUNTIME-010
// REQ: RUNTIME-015
TEST(RuntimeExternalChannelTest, StandaloneExternalTopicsDoNotConstructUnobservableClocks) {
  for (const bool imported : {true, false}) {
    SCOPED_TRACE(imported ? "import" : "export");
    Graph graph{imported ? "runtime-standalone-import-clock" : "runtime-standalone-export-clock"};
    if (imported) {
      (void)graph.import_topic<std::uint32_t>("holomq://" + unique_endpoint_name());
    } else {
      (void)graph.export_topic<std::uint32_t>("holomq://" + unique_endpoint_name());
    }
    graph.partition("application").include_all();
    const ClockKey default_clock = graph.add_clock<test::RuntimeProbeClock>("runtime-probe-clock");
    graph.set_default_clock(default_clock);
    const ExecutionPlan plan = compile(graph);
    ASSERT_TRUE(plan.ok()) << plan.json();
    ASSERT_FALSE(plan.graph().vertices().empty());
    ASSERT_TRUE(std::ranges::none_of(plan.graph().vertices(), [](const auto& vertex) {
      return vertex.kind == VertexKind::kOperator;
    }));

    // Key setup: exclude compile-time factory activity from the Runtime observation.
    test::RuntimeProbeState::reset();
    RunSession session = run_async(plan);
    const bool completed = wait_for_session_terminal(session);
    if (!completed) {
      session.request_stop();
    }
    session.wait();

    // Key assertion: the interface-only run completes without constructing an inert clock.
    EXPECT_TRUE(completed);
    EXPECT_EQ(session.termination(), RunTermination::kCompleted);
    EXPECT_TRUE(session.diagnostics().empty())
        << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
    EXPECT_EQ(test::RuntimeProbeState::clocks_constructed.load(std::memory_order_relaxed), 0U);
  }
}

// Disconnected catalog endpoints remain part of the exact include_all() executable cover while
// only the Operator-bearing lane contributes a Runtime callback owner.
// REQ: CHAN-073
// REQ: COMPILE-068
// REQ: GRAPH-032
// REQ: RUNTIME-015
// REQ: RUNTIME-041
TEST(RuntimeExternalChannelTest, IncludeAllRunsMixedOperatorAndDisconnectedEndpointsUnderOneOwner) {
  CatalogOnlyRuntimeProbe::reset();
  Graph graph{"runtime-include-all-disconnected-mixed"};
  (void)graph.import_topic<std::uint32_t>("holomq://" + unique_endpoint_name());
  (void)graph.export_topic<std::uint32_t>("holomq://" + unique_endpoint_name());
  (void)graph.op<CatalogOnlyClockOperator>("independent");
  const ClockKey default_clock = graph.add_clock<RealtimeClock>("runtime-clock");
  graph.set_default_clock(default_clock);
  graph.partition("application").include_all();

  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.execution_partitions().size(), 1U);
  EXPECT_EQ(plan.execution_partitions().front().authored_name,
            std::optional<std::string>{"application"});
  EXPECT_EQ(plan.execution_partitions().front().members.size(), 3U);
  EXPECT_EQ(plan.execution_partitions().front().operator_paths,
            std::vector<std::string>{"independent"});
  ASSERT_EQ(plan.execution_lanes().size(), 1U);
  EXPECT_EQ(plan.scheduler_plan().execution_lane_count, 1U);
  EXPECT_EQ(plan.scheduler_plan().worker_count, 1U);
  EXPECT_EQ(plan.scheduler_plan().max_concurrent_activations, 1U);
  EXPECT_FALSE(plan.scheduler_plan().activation_overlap);
  EXPECT_TRUE(plan.payload_edges().empty());
  EXPECT_TRUE(plan.schedule_edges().empty());

  RunSession session = run_async(plan);
  const bool computed = CatalogOnlyRuntimeProbe::wait_for_compute();
  session.request_stop();
  session.wait();

  EXPECT_TRUE(computed);
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
}

// A constrained automatic cover attaches a sole adjacent imported endpoint to the authored
// Operator owner when both vertices share the default process group. The merged owner retains the
// external Channel receive boundary and executes the transferred sample.
// REQ: COMPILE-068
// REQ: RUNTIME-015
// REQ: RUNTIME-041
TEST(RuntimeExternalChannelTest, SameProcessImportEndpointAttachesToAuthoredOwnerAndReceives) {
  ExternalImportProbe::reset();
  const std::string endpoint_name = unique_endpoint_name();
  const std::string endpoint_uri = "holomq://" + endpoint_name;

  Graph graph{"runtime-constrained-same-process-import"};
  const ClockKey capture_clock = graph.add_clock<ManualClock>("capture-clock", std::int64_t{1000});
  graph.set_default_clock(capture_clock);
  TopicOptions options;
  options.endpoint_name = endpoint_name;
  options.capture_time = RequireCaptureTime{.clock = capture_clock};
  const auto imported = graph.import_topic<std::uint32_t>(endpoint_uri, std::move(options));
  const auto sink = graph.op<ExternalImportSink>("sink");
  graph.add_flow(imported.output(), sink->input, ConnectionOptions{.queue_depth = 8U});
  graph.partition("anchor").add(sink);
  graph.auto_partition_remaining();

  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.execution_partitions().size(), 1U);
  ASSERT_EQ(plan.execution_lanes().size(), 1U);
  EXPECT_EQ(plan.scheduler_plan().execution_lane_count, 1U);
  EXPECT_EQ(plan.scheduler_plan().worker_count, 1U);
  EXPECT_TRUE(plan.schedule_edges().empty());
  const auto& anchor = plan.execution_partitions().front();
  EXPECT_EQ(anchor.authored_name, std::optional<std::string>{"anchor"});
  EXPECT_EQ(anchor.members.size(), 2U);
  EXPECT_EQ(anchor.operator_paths, std::vector<std::string>{"sink"});
  const auto endpoint_vertex = std::ranges::find_if(
      plan.graph().vertices(),
      [](const auto& vertex) { return vertex.kind == VertexKind::kImportedTopic; });
  const auto sink_vertex = std::ranges::find_if(plan.graph().vertices(), [](const auto& vertex) {
    return vertex.kind == VertexKind::kOperator && vertex.name == "sink";
  });
  ASSERT_NE(endpoint_vertex, plan.graph().vertices().end());
  ASSERT_NE(sink_vertex, plan.graph().vertices().end());
  EXPECT_NE(std::ranges::find(anchor.members, endpoint_vertex->id), anchor.members.end());
  EXPECT_NE(std::ranges::find(anchor.members, sink_vertex->id), anchor.members.end());
  const auto placements = detail::ProcessPlacementPlanAccess::vertex_process_placements(plan);
  const auto endpoint_placement = std::ranges::find(
      placements, endpoint_vertex->id, &detail::VertexProcessPlacementPlanRecord::vertex);
  const auto sink_placement = std::ranges::find(
      placements, sink_vertex->id, &detail::VertexProcessPlacementPlanRecord::vertex);
  ASSERT_NE(endpoint_placement, placements.end());
  ASSERT_NE(sink_placement, placements.end());
  EXPECT_EQ(endpoint_placement->process_group, sink_placement->process_group);

  auto external_runtime = ChannelRuntime::create();
  ASSERT_TRUE(external_runtime) << runtime_error_message(external_runtime.error());
  auto publisher =
      external_runtime->create_publisher<std::uint32_t>(plan.endpoint_catalog(), endpoint_name);
  ASSERT_TRUE(publisher) << runtime_error_message(publisher.error());

  RunSession session = run_async(plan);
  if (auto connected = wait_for_subscriber(*publisher); !connected) {
    session.request_stop();
    session.wait();
    FAIL() << "same-process import subscriber did not connect: "
           << runtime_error_message(connected.error());
  }
  ASSERT_TRUE(publisher->emit_captured(0x12345678U, 100));
  const bool received = ExternalImportProbe::wait_for_count(1U);
  session.request_stop();
  session.wait();

  ASSERT_TRUE(received);
  EXPECT_EQ(ExternalImportProbe::values[0], 0x12345678U);
  EXPECT_EQ(ExternalImportProbe::sequences[0], 1U);
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
}

// The export-side same-process case is symmetric: the exported endpoint joins the authored source
// owner, which retains and drives the external publisher without a crossing schedule edge.
// REQ: COMPILE-068
// REQ: RUNTIME-015
// REQ: RUNTIME-041
TEST(RuntimeExternalChannelTest, SameProcessExportEndpointAttachesToAuthoredOwnerAndPublishes) {
  ExternalScalarExportSource::reset();
  const std::string endpoint_name = unique_endpoint_name();
  const std::string endpoint_uri = "holomq://" + endpoint_name;

  Graph graph{"runtime-constrained-same-process-export"};
  const ClockKey runtime_clock = graph.add_clock<RealtimeClock>("runtime-clock");
  graph.set_default_clock(runtime_clock);
  TopicOptions options;
  options.endpoint_name = endpoint_name;
  const auto exported = graph.export_topic<std::uint32_t>(endpoint_uri, std::move(options));
  const auto source = graph.op<ExternalScalarExportSource>("source");
  graph.add_flow(source->output, exported.input(), ConnectionOptions{.queue_depth = 4U});
  graph.partition("anchor").add(source);
  graph.auto_partition_remaining();

  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.execution_partitions().size(), 1U);
  ASSERT_EQ(plan.execution_lanes().size(), 1U);
  EXPECT_EQ(plan.scheduler_plan().execution_lane_count, 1U);
  EXPECT_EQ(plan.scheduler_plan().worker_count, 1U);
  EXPECT_TRUE(plan.schedule_edges().empty());
  const auto& anchor = plan.execution_partitions().front();
  EXPECT_EQ(anchor.authored_name, std::optional<std::string>{"anchor"});
  EXPECT_EQ(anchor.members.size(), 2U);
  EXPECT_EQ(anchor.operator_paths, std::vector<std::string>{"source"});
  const auto endpoint_vertex = std::ranges::find_if(
      plan.graph().vertices(),
      [](const auto& vertex) { return vertex.kind == VertexKind::kExportedTopic; });
  const auto source_vertex = std::ranges::find_if(plan.graph().vertices(), [](const auto& vertex) {
    return vertex.kind == VertexKind::kOperator && vertex.name == "source";
  });
  ASSERT_NE(endpoint_vertex, plan.graph().vertices().end());
  ASSERT_NE(source_vertex, plan.graph().vertices().end());
  EXPECT_NE(std::ranges::find(anchor.members, endpoint_vertex->id), anchor.members.end());
  EXPECT_NE(std::ranges::find(anchor.members, source_vertex->id), anchor.members.end());
  const auto placements = detail::ProcessPlacementPlanAccess::vertex_process_placements(plan);
  const auto endpoint_placement = std::ranges::find(
      placements, endpoint_vertex->id, &detail::VertexProcessPlacementPlanRecord::vertex);
  const auto source_placement = std::ranges::find(
      placements, source_vertex->id, &detail::VertexProcessPlacementPlanRecord::vertex);
  ASSERT_NE(endpoint_placement, placements.end());
  ASSERT_NE(source_placement, placements.end());
  EXPECT_EQ(endpoint_placement->process_group, source_placement->process_group);

  auto external_runtime = ChannelRuntime::create();
  ASSERT_TRUE(external_runtime) << runtime_error_message(external_runtime.error());
  auto subscriber =
      external_runtime->create_subscriber<std::uint32_t>(plan.endpoint_catalog(), endpoint_name);
  ASSERT_TRUE(subscriber) << runtime_error_message(subscriber.error());

  RunSession session = run_async(plan);
  if (!ExternalScalarExportSource::wait_for_compute_started()) {
    ExternalScalarExportSource::allow_emit();
    session.request_stop();
    session.wait();
    FAIL() << "same-process export source did not enter compute; termination="
           << static_cast<int>(session.termination());
  }
  auto publisher_ready = wait_for_publisher(*subscriber);
  ExternalScalarExportSource::allow_emit();
  ASSERT_TRUE(publisher_ready) << runtime_error_message(publisher_ready.error());
  auto received = wait_for_sample(*subscriber);

  session.request_stop();
  session.wait();
  ASSERT_TRUE(received) << runtime_error_message(received.error());
  EXPECT_EQ(received->data, 0xdecafbadU);
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
}

// A constrained automatic cover may attach an external import to its sole authored neighbor only
// when both vertices share one process. A separately placed endpoint retains its topology lane,
// and the Operator worker still owns the external-channel receive boundary.
// REQ: COMPILE-068
// REQ: RUNTIME-015
// REQ: RUNTIME-041
TEST(RuntimeExternalChannelTest,
     CrossProcessImportEndpointRemainsSeparateAndOperatorWorkerReceives) {
  ExternalImportProbe::reset();
  const std::string endpoint_name = unique_endpoint_name();
  const std::string endpoint_uri = "holomq://" + endpoint_name;

  Graph graph{"runtime-constrained-cross-process-import"};
  const ClockKey capture_clock = graph.add_clock<ManualClock>("capture-clock", std::int64_t{1000});
  graph.set_default_clock(capture_clock);
  TopicOptions options;
  options.endpoint_name = endpoint_name;
  options.capture_time = RequireCaptureTime{.clock = capture_clock};
  const auto imported = graph.import_topic<std::uint32_t>(endpoint_uri, std::move(options));
  const auto sink = graph.op<ExternalImportSink>("sink");
  graph.add_flow(imported.output(), sink->input, ConnectionOptions{.queue_depth = 8U});
  graph.partition("anchor").add(sink);
  graph.auto_partition_remaining();

  const ExecutionPlan plan = detail::compile_with_process_placement(
      graph, CompileOptions{}, external_endpoint_process_placement(endpoint_uri));
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.execution_partitions().size(), 2U);
  ASSERT_EQ(plan.execution_lanes().size(), 2U);
  EXPECT_EQ(plan.scheduler_plan().execution_lane_count, 2U);
  EXPECT_EQ(plan.scheduler_plan().worker_count, 1U);
  ASSERT_EQ(plan.schedule_edges().size(), 1U);

  const auto endpoint_vertex = std::ranges::find_if(
      plan.graph().vertices(),
      [](const auto& vertex) { return vertex.kind == VertexKind::kImportedTopic; });
  const auto sink_vertex = std::ranges::find_if(plan.graph().vertices(), [](const auto& vertex) {
    return vertex.kind == VertexKind::kOperator && vertex.name == "sink";
  });
  ASSERT_NE(endpoint_vertex, plan.graph().vertices().end());
  ASSERT_NE(sink_vertex, plan.graph().vertices().end());
  const auto endpoint_partition =
      std::ranges::find_if(plan.execution_partitions(), [&](const auto& partition) {
        return partition.members == std::vector<VertexId>{endpoint_vertex->id};
      });
  const auto anchor_partition = std::ranges::find_if(
      plan.execution_partitions(),
      [](const auto& partition) { return partition.authored_name == "anchor"; });
  ASSERT_NE(endpoint_partition, plan.execution_partitions().end());
  ASSERT_NE(anchor_partition, plan.execution_partitions().end());
  EXPECT_NE(endpoint_partition->id, anchor_partition->id);
  EXPECT_EQ(plan.schedule_edges().front().source, endpoint_partition->id);
  EXPECT_EQ(plan.schedule_edges().front().destination, anchor_partition->id);

  const auto placements = detail::ProcessPlacementPlanAccess::vertex_process_placements(plan);
  const auto endpoint_placement = std::ranges::find(
      placements, endpoint_vertex->id, &detail::VertexProcessPlacementPlanRecord::vertex);
  const auto sink_placement = std::ranges::find(
      placements, sink_vertex->id, &detail::VertexProcessPlacementPlanRecord::vertex);
  ASSERT_NE(endpoint_placement, placements.end());
  ASSERT_NE(sink_placement, placements.end());
  ASSERT_NE(endpoint_placement->process_group, sink_placement->process_group);

  auto external_runtime = ChannelRuntime::create();
  ASSERT_TRUE(external_runtime) << runtime_error_message(external_runtime.error());
  auto publisher =
      external_runtime->create_publisher<std::uint32_t>(plan.endpoint_catalog(), endpoint_name);
  ASSERT_TRUE(publisher) << runtime_error_message(publisher.error());

  RunSession session = detail::run_async_worker(plan, sink_placement->process_group);
  if (auto connected = wait_for_subscriber(*publisher); !connected) {
    session.request_stop();
    session.wait();
    FAIL() << "cross-process import subscriber did not connect: "
           << runtime_error_message(connected.error());
  }
  ASSERT_TRUE(publisher->emit_captured(0x12345678U, 100));
  const bool received = ExternalImportProbe::wait_for_count(1U);
  session.request_stop();
  session.wait();

  ASSERT_TRUE(received);
  EXPECT_EQ(ExternalImportProbe::values[0], 0x12345678U);
  EXPECT_EQ(ExternalImportProbe::sequences[0], 1U);
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
}

// The export-side process-boundary case is symmetric: the endpoint keeps its synthetic lane while
// the authored source worker materializes and drives the external publisher.
// REQ: COMPILE-068
// REQ: RUNTIME-015
// REQ: RUNTIME-041
TEST(RuntimeExternalChannelTest,
     CrossProcessExportEndpointRemainsSeparateAndOperatorWorkerPublishes) {
  ExternalScalarExportSource::reset();
  const std::string endpoint_name = unique_endpoint_name();
  const std::string endpoint_uri = "holomq://" + endpoint_name;

  Graph graph{"runtime-constrained-cross-process-export"};
  const ClockKey runtime_clock = graph.add_clock<RealtimeClock>("runtime-clock");
  graph.set_default_clock(runtime_clock);
  TopicOptions options;
  options.endpoint_name = endpoint_name;
  const auto exported = graph.export_topic<std::uint32_t>(endpoint_uri, std::move(options));
  const auto source = graph.op<ExternalScalarExportSource>("source");
  graph.add_flow(source->output, exported.input(), ConnectionOptions{.queue_depth = 4U});
  graph.partition("anchor").add(source);
  graph.auto_partition_remaining();

  const ExecutionPlan plan = detail::compile_with_process_placement(
      graph, CompileOptions{}, external_endpoint_process_placement(endpoint_uri));
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.execution_partitions().size(), 2U);
  ASSERT_EQ(plan.execution_lanes().size(), 2U);
  EXPECT_EQ(plan.scheduler_plan().execution_lane_count, 2U);
  EXPECT_EQ(plan.scheduler_plan().worker_count, 1U);
  ASSERT_EQ(plan.schedule_edges().size(), 1U);

  const auto endpoint_vertex = std::ranges::find_if(
      plan.graph().vertices(),
      [](const auto& vertex) { return vertex.kind == VertexKind::kExportedTopic; });
  const auto source_vertex = std::ranges::find_if(plan.graph().vertices(), [](const auto& vertex) {
    return vertex.kind == VertexKind::kOperator && vertex.name == "source";
  });
  ASSERT_NE(endpoint_vertex, plan.graph().vertices().end());
  ASSERT_NE(source_vertex, plan.graph().vertices().end());
  const auto endpoint_partition =
      std::ranges::find_if(plan.execution_partitions(), [&](const auto& partition) {
        return partition.members == std::vector<VertexId>{endpoint_vertex->id};
      });
  const auto anchor_partition = std::ranges::find_if(
      plan.execution_partitions(),
      [](const auto& partition) { return partition.authored_name == "anchor"; });
  ASSERT_NE(endpoint_partition, plan.execution_partitions().end());
  ASSERT_NE(anchor_partition, plan.execution_partitions().end());
  EXPECT_NE(endpoint_partition->id, anchor_partition->id);
  EXPECT_EQ(plan.schedule_edges().front().source, anchor_partition->id);
  EXPECT_EQ(plan.schedule_edges().front().destination, endpoint_partition->id);

  const auto placements = detail::ProcessPlacementPlanAccess::vertex_process_placements(plan);
  const auto endpoint_placement = std::ranges::find(
      placements, endpoint_vertex->id, &detail::VertexProcessPlacementPlanRecord::vertex);
  const auto source_placement = std::ranges::find(
      placements, source_vertex->id, &detail::VertexProcessPlacementPlanRecord::vertex);
  ASSERT_NE(endpoint_placement, placements.end());
  ASSERT_NE(source_placement, placements.end());
  ASSERT_NE(endpoint_placement->process_group, source_placement->process_group);

  auto external_runtime = ChannelRuntime::create();
  ASSERT_TRUE(external_runtime) << runtime_error_message(external_runtime.error());
  auto subscriber =
      external_runtime->create_subscriber<std::uint32_t>(plan.endpoint_catalog(), endpoint_name);
  ASSERT_TRUE(subscriber) << runtime_error_message(subscriber.error());

  RunSession session = detail::run_async_worker(plan, source_placement->process_group);
  if (!ExternalScalarExportSource::wait_for_compute_started()) {
    ExternalScalarExportSource::allow_emit();
    session.request_stop();
    session.wait();
    FAIL() << "cross-process export source did not enter compute; termination="
           << static_cast<int>(session.termination());
  }
  auto publisher_ready = wait_for_publisher(*subscriber);
  ExternalScalarExportSource::allow_emit();
  ASSERT_TRUE(publisher_ready) << runtime_error_message(publisher_ready.error());
  auto received = wait_for_sample(*subscriber);

  session.request_stop();
  session.wait();
  ASSERT_TRUE(received) << runtime_error_message(received.error());
  EXPECT_EQ(received->data, 0xdecafbadU);
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
}

// REQ: CHAN-069
// REQ: RUNTIME-011
// REQ: RUNTIME-016
// REQ: RUNTIME-017
TEST(RuntimeExternalChannelTest, UnmonitoredScalarImportDrainsEverySampleInPublisherOrder) {
  ExternalImportProbe::reset();
  const std::string endpoint_name = unique_endpoint_name();

  //   external_publisher --(holomq://endpoint, unmonitored)--> imported_topic
  //       --[depth=8]--> sink
  Graph graph{"runtime-unmonitored-external-import"};
  const ClockKey capture_clock = graph.add_clock<ManualClock>("capture-clock", std::int64_t{1000});
  graph.set_default_clock(capture_clock);
  TopicOptions options;
  options.endpoint_name = endpoint_name;
  options.capture_time = RequireCaptureTime{.clock = capture_clock};
  const auto imported =
      graph.import_topic<std::uint32_t>("holomq://" + endpoint_name, std::move(options));
  const auto sink = graph.op<ExternalImportSink>("sink");
  graph.add_flow(imported.output(), sink->input, ConnectionOptions{.queue_depth = 8U});

  const ExecutionPlan compiled = compile(graph);
  auto support_threads = std::make_shared<ExternalImportSupportThreadProbe>();
  const ExecutionPlan plan =
      detail::RuntimeTestPlanAccess::with_support_thread_control(compiled, support_threads);
  ASSERT_TRUE(plan.ok()) << plan.json();
  // Key setup: import has RequireCaptureTime but no ClockHealthOptions. Runtime must drain this
  // externally driven catalog input on the controller path rather than starting a Channel pump.
  ASSERT_EQ(plan.endpoint_clock_health().size(), 1U);
  EXPECT_EQ(plan.endpoint_clock_health().front().capability, ClockHealthCapability::kUnavailable);
  auto external_runtime = ChannelRuntime::create();
  ASSERT_TRUE(external_runtime) << runtime_error_message(external_runtime.error());
  auto publisher =
      external_runtime->create_publisher<std::uint32_t>(plan.endpoint_catalog(), endpoint_name);
  ASSERT_TRUE(publisher) << runtime_error_message(publisher.error());

  RunSession session = run_async(plan);
  if (auto connected = wait_for_subscriber(*publisher); !connected) {
    session.request_stop();
    session.wait();
    FAIL() << "unmonitored external subscriber did not connect: "
           << runtime_error_message(connected.error());
  }

  constexpr std::array kValues{0x01020304U, 0x11223344U};
  for (std::size_t index = 0U; index < kValues.size(); ++index) {
    ASSERT_TRUE(publisher->emit_captured(kValues[index], 100 + static_cast<std::int64_t>(index)));
  }
  EXPECT_TRUE(ExternalImportProbe::wait_for_count(kValues.size()));

  session.request_stop();
  session.wait();
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
  // Key assertion: both values arrive in emission order with monotonically increasing sequences.
  EXPECT_EQ(ExternalImportProbe::count.load(std::memory_order_acquire), kValues.size());
  EXPECT_EQ(ExternalImportProbe::values[0], kValues[0]);
  EXPECT_EQ(ExternalImportProbe::values[1], kValues[1]);
  EXPECT_EQ(ExternalImportProbe::sequences[0], 1U);
  EXPECT_EQ(ExternalImportProbe::sequences[1], 2U);
  EXPECT_TRUE(support_threads->entered(detail::RuntimeSupportThreadRole::kController));
  EXPECT_FALSE(support_threads->entered(detail::RuntimeSupportThreadRole::kChannelPump));
}

// The catalog inline wrapper may retain N+1 in its own bounded staging buffer while the typed
// depth-one slot still owns ReuseLast N. That wrapper-owned head must participate in the same
// controller handoff as raw Tensor and descriptor-schema imports before ordered EOS is consumed.
// REQ: CHAN-136
// REQ: RUNTIME-050
TEST(RuntimeExternalChannelTest, CatalogInlineReuseLastPublishesFinalDataBeforeOrderedTerminal) {
  CatalogScalarReuseProbe::reset();
  const std::string endpoint_name = unique_endpoint_name();

  Graph graph{"runtime-catalog-inline-reuse-terminal"};
  const ClockKey runtime_clock = graph.add_clock<RealtimeClock>("runtime-clock");
  graph.set_default_clock(runtime_clock);
  TopicOptions options;
  options.endpoint_name = endpoint_name;
  options.capture_time = RequireCaptureTime{.clock = runtime_clock};
  const auto imported =
      graph.import_topic<std::uint32_t>("holomq://" + endpoint_name, std::move(options));
  const auto sink = graph.op<CatalogScalarReuseSink>("sink");
  graph.add_flow(imported.output(),
                 sink->input,
                 ConnectionOptions{.queue_policy = QueuePolicy::kDropOldest, .queue_depth = 1U});

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
    FAIL() << "catalog inline subscriber did not connect: "
           << runtime_error_message(connected.error());
  }
  const std::int64_t capture = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                   std::chrono::steady_clock::now().time_since_epoch())
                                   .count();
  ASSERT_TRUE(publisher->emit_captured(1U, capture));
  ASSERT_TRUE(CatalogScalarReuseProbe::wait_for_count(1U));
  ASSERT_TRUE(publisher->emit_captured(2U, capture + 1));
  ASSERT_TRUE(publisher->close());
  ASSERT_TRUE(CatalogScalarReuseProbe::wait_for_count(2U))
      << "evictions="
      << phase_counters->snapshot()[detail::RuntimePhaseCounters::index(
             detail::RuntimePhaseCounter::kTemporalRetentionEvictions)]
      << " readiness="
      << phase_counters->snapshot()[detail::RuntimePhaseCounters::index(
             detail::RuntimePhaseCounter::kChannelReadinessPublished)]
      << " backpressured="
      << phase_counters->snapshot()[detail::RuntimePhaseCounters::index(
             detail::RuntimePhaseCounter::kChannelReceiveBackpressured)]
      << " candidates="
      << phase_counters->snapshot()[detail::RuntimePhaseCounters::index(
             detail::RuntimePhaseCounter::kTemporalRetentionCandidates)]
      << " declined="
      << phase_counters->snapshot()[detail::RuntimePhaseCounters::index(
             detail::RuntimePhaseCounter::kTemporalRetentionDeclinedAboveFloor)]
      << " refused="
      << phase_counters->snapshot()[detail::RuntimePhaseCounters::index(
             detail::RuntimePhaseCounter::kTemporalRetentionDeclinedEvictionRefused)]
      << " not_superseded="
      << phase_counters->snapshot()[detail::RuntimePhaseCounters::index(
             detail::RuntimePhaseCounter::kTemporalRetentionDeclinedNotSuperseded)]
      << " running=" << session.is_running() << " diagnostics=" << session.diagnostics().size();
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kCompleted);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
  EXPECT_EQ(CatalogScalarReuseProbe::count.load(std::memory_order_acquire), 2U);
  EXPECT_EQ(CatalogScalarReuseProbe::values, (std::array<std::uint32_t, 2U>{1U, 2U}));
  EXPECT_EQ(CatalogScalarReuseProbe::sequences, (std::array<std::uint64_t, 2U>{1U, 2U}));
  EXPECT_EQ(CatalogScalarReuseProbe::states,
            (std::array<InputState, 2U>{InputState::kSelected, InputState::kSelected}));
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
  ASSERT_EQ(compiled.deadlines().size(), 1U);
  EXPECT_EQ(compiled.deadlines().front().source_coordinate,
            DeadlineSourceCoordinateKind::kTriggerSample);
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
            (std::array<ActivationReason, 2U>{ActivationReason::kDeadlineExceeded,
                                              ActivationReason::kDeadlineExceeded}));
  EXPECT_EQ(ExternalImageDeadlineProbe::input_states,
            (std::array<InputState, 2U>{InputState::kSelected, InputState::kSelected}));
  // The preceding ordered health report is operation one, so data N/N+1 are sequences two/three
  // and the protected terminal accepted by natural drain is sequence four.
  EXPECT_EQ(ExternalImageDeadlineProbe::sequences, (std::array<std::uint64_t, 2U>{2U, 3U}));
  EXPECT_EQ(ExternalImageDeadlineProbe::values, (std::array<std::uint8_t, 2U>{0x3cU, 0x5dU}));
  ASSERT_EQ(session.diagnostics().size(), 2U);
  EXPECT_EQ(session.diagnostics().front().token, "DEADLINE_MISS");
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

// Descriptor-backed schema imports share the catalog's ordered health stream with scalar and raw
// Tensor imports. A preceding unhealthy report suppresses the following sample without exposing
// its schema value to compute or failing the run.
// REQ: CHAN-069
// REQ: EXTERNAL-010
// REQ: RUNTIME-016
// REQ: RUNTIME-017
TEST(RuntimeExternalChannelTest,
     DescriptorTensorSchemaImportAppliesPrecedingClockHealthBeforeCompute) {
  ExternalImageImportProbe::reset();
  const std::string endpoint_name = unique_endpoint_name();

  Graph graph{"runtime-external-image-health"};
  const ClockKey capture_clock = graph.add_clock<ManualClock>("capture-clock", std::int64_t{0});
  const ClockKey runtime_clock = graph.add_clock<RealtimeClock>("runtime-clock");
  graph.set_default_clock(runtime_clock);
  const std::int64_t capture_offset = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                          std::chrono::steady_clock::now().time_since_epoch())
                                          .count();
  graph.add_clock_conversion(
      capture_clock, runtime_clock, AffineClockConversion{.offset_ns = capture_offset});
  TopicOptions options = external_image_import_options(endpoint_name);
  options.capture_time = RequireCaptureTime{.clock = capture_clock};
  options.clock_health = ClockHealthOptions{
      .capability = ClockHealthCapability::kStatusReports,
      .requirement = ClockHealthRequirement::kFreshHealthyReport,
      .max_report_age = 5s,
  };
  const auto imported = graph.import_topic<schema::ImageT>(
      "holomq://" + endpoint_name, std::move(options), external_image_contract());
  const auto sink = graph.op<ExternalImageImportSink>("sink");
  graph.add_flow(imported.output(), sink->input, ConnectionOptions{.queue_depth = 1U});

  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  auto external_runtime = ChannelRuntime::create();
  ASSERT_TRUE(external_runtime) << runtime_error_message(external_runtime.error());
  auto publisher = external_runtime->create_tensor_schema_publisher<ExternalImagePayload>(
      plan.endpoint_catalog(), endpoint_name);
  ASSERT_TRUE(publisher) << runtime_error_message(publisher.error());

  // Key setup: place a healthy report before the first descriptor Tensor and an unhealthy report
  // before the second within the catalog's shared ordered operation stream.
  RunSession session = run_async(plan);
  ASSERT_TRUE(wait_for_schema_subscriber(*publisher));
  ASSERT_TRUE(publisher->report_clock_health(ClockHealthReport::healthy()));
  auto first_tensor = make_external_image_tensor(0x31U);
  ASSERT_TRUE(first_tensor) << runtime_error_message(first_tensor.error());
  auto first_metadata = schema::build_image_metadata(
      *first_tensor,
      schema::ImageMetadataOptions{
          .height = 1U, .width = 8U, .encoding = schema::ImageEncoding_MONO8});
  ASSERT_TRUE(first_metadata) << runtime_error_message(first_metadata.error());
  ASSERT_TRUE(publisher->emit_captured(*first_metadata, std::move(*first_tensor), 1));
  ASSERT_TRUE(ExternalImageImportProbe::wait_for_count(1U));

  ASSERT_TRUE(publisher->report_clock_health(
      ClockHealthReport::unhealthy(ClockHealthReason::kSynchronizationLockLost)));
  auto suppressed_tensor = make_external_image_tensor(0x62U);
  ASSERT_TRUE(suppressed_tensor) << runtime_error_message(suppressed_tensor.error());
  auto suppressed_metadata = schema::build_image_metadata(
      *suppressed_tensor,
      schema::ImageMetadataOptions{
          .height = 1U, .width = 8U, .encoding = schema::ImageEncoding_MONO8});
  ASSERT_TRUE(suppressed_metadata) << runtime_error_message(suppressed_metadata.error());
  ASSERT_TRUE(publisher->emit_captured(*suppressed_metadata, std::move(*suppressed_tensor), 2));
  std::this_thread::sleep_for(100ms);

  session.request_stop();
  session.wait();
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
  // Key assertion: only the Tensor preceded by healthy state reaches typed compute; suppression
  // of the second sample is nonterminal.
  EXPECT_EQ(ExternalImageImportProbe::count.load(std::memory_order_acquire), 1U);
  EXPECT_EQ(ExternalImageImportProbe::first_byte.load(std::memory_order_relaxed), 0x31U);
}

// The descriptor-schema readiness facade must expose control-only clock-health traffic so a
// standalone consumer can decide to drain the next ordered catalog operation without waiting for
// unrelated Tensor data.
// REQ: CHAN-069
// REQ: RUNTIME-017
TEST(RuntimeExternalChannelTest, DescriptorTensorSchemaHasSamplesIncludesClockHealthLane) {
  const std::string endpoint_name = unique_endpoint_name();

  Graph graph{"runtime-external-image-health-readiness"};
  const ClockKey capture_clock = graph.add_clock<ManualClock>("capture-clock", std::int64_t{0});
  const ClockKey runtime_clock = graph.add_clock<RealtimeClock>("runtime-clock");
  graph.set_default_clock(runtime_clock);
  graph.add_clock_conversion(capture_clock, runtime_clock, AffineClockConversion{});
  TopicOptions options = external_image_import_options(endpoint_name);
  options.capture_time = RequireCaptureTime{.clock = capture_clock};
  options.clock_health = ClockHealthOptions{
      .capability = ClockHealthCapability::kStatusReports,
      .requirement = ClockHealthRequirement::kFreshHealthyReport,
      .max_report_age = 5s,
  };
  const auto imported = graph.import_topic<schema::ImageT>(
      "holomq://" + endpoint_name, std::move(options), external_image_contract());
  const auto sink = graph.op<ExternalImageImportSink>("sink");
  graph.add_flow(imported.output(), sink->input, ConnectionOptions{.queue_depth = 1U});

  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << (plan.diagnostics().empty() ? "<no diagnostic>"
                                                        : plan.diagnostics().front().token);
  const auto endpoint = plan.endpoint_catalog().endpoint(endpoint_name);
  ASSERT_TRUE(endpoint) << runtime_error_message(endpoint.error());
  const auto* const endpoint_record = detail::ChannelEndpointCatalogAccess::record(*endpoint);
  const auto* const type_ops = detail::runtime_port_type_ops<schema::ImageT>();
  ASSERT_NE(endpoint_record, nullptr);
  ASSERT_NE(type_ops, nullptr);
  ASSERT_TRUE(endpoint_record->payload_identity.schema);
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access): guarded by ASSERT_TRUE above.
  const auto& payload_schema = *endpoint_record->payload_identity.schema;

  auto graph_runtime = ChannelRuntime::create();
  auto external_runtime = ChannelRuntime::create();
  ASSERT_TRUE(graph_runtime) << runtime_error_message(graph_runtime.error());
  ASSERT_TRUE(external_runtime) << runtime_error_message(external_runtime.error());
  auto subscriber = detail::RuntimeChannelBindingAccess::create_catalog_tensor_schema_subscriber(
      *graph_runtime,
      *endpoint,
      payload_schema,
      type_ops->wire_capacity,
      type_ops->descriptor_tensor_static_count);
  ASSERT_TRUE(subscriber) << runtime_error_message(subscriber.error());

  // Key setup: probe the normalized field contract retained by the actual catalog subscriber
  // with otherwise structural metadata that independently violates each constrained dimension.
  const PayloadDescriptor valid_descriptor{
      .payload_kind = ChannelPayloadKind::kLocalBufferDescriptor,
      .schema_id = schema_identity<schema::Tensor>().family_id,
      .sequence = 1U,
      .byte_size = 8U,
      .local =
          LocalPayloadReference{
              .process_id = 1U, .token = 1U, .generation = 1U, .expected_recipient_count = 1U},
  };
  const TensorViewMetadata valid_view{
      .rank = 2U,
      .dtype = kExternalTensorDtype,
      .device = DLDevice{.device_type = kDLCPU, .device_id = 0},
      .byte_size = 8U,
      .shape = {1, 8},
  };
  const auto expect_field_rejection = [&](TensorViewMetadata candidate,
                                          PayloadDescriptor descriptor) {
    auto validated = (*subscriber)->validate_field_descriptor(0U, candidate, descriptor);
    ASSERT_FALSE(validated);
    EXPECT_EQ(validated.error().code, ErrorCode::kInvalidArgument);
  };
  auto wrong_dtype = valid_view;
  wrong_dtype.dtype.bits = 16U;
  expect_field_rejection(wrong_dtype, valid_descriptor);
  auto wrong_rank = valid_view;
  wrong_rank.rank = 1U;
  expect_field_rejection(wrong_rank, valid_descriptor);
  auto wrong_memory = valid_view;
  wrong_memory.device = DLDevice{.device_type = kDLCUDA, .device_id = 0};
  expect_field_rejection(wrong_memory, valid_descriptor);
  auto wrong_strides = valid_view;
  wrong_strides.has_explicit_byte_strides = true;
  wrong_strides.byte_strides = {16, 1};
  expect_field_rejection(wrong_strides, valid_descriptor);
  auto wrong_span = valid_view;
  wrong_span.byte_size = 9U;
  auto oversized_descriptor = valid_descriptor;
  oversized_descriptor.byte_size = 9U;
  expect_field_rejection(wrong_span, oversized_descriptor);
  auto mismatched_descriptor = valid_descriptor;
  mismatched_descriptor.byte_size = 7U;
  expect_field_rejection(valid_view, mismatched_descriptor);

  // Key assertion: all constrained dimensions reject before import, while the exact field view
  // remains admissible through the same contract hook used by standalone and Runtime facades.
  auto valid_field = (*subscriber)->validate_field_descriptor(0U, valid_view, valid_descriptor);
  ASSERT_TRUE(valid_field) << runtime_error_message(valid_field.error());
  auto publisher = external_runtime->create_tensor_schema_publisher<ExternalImagePayload>(
      plan.endpoint_catalog(), endpoint_name);
  ASSERT_TRUE(publisher) << runtime_error_message(publisher.error());
  ASSERT_TRUE(wait_for_catalog_schema_subscriber(*publisher, **subscriber));

  // Key setup: publish only a healthy clock-control operation; no descriptor Tensor data is
  // available to make the data or terminal lanes ready.
  auto reported = publisher->report_clock_health(ClockHealthReport::healthy());
  ASSERT_TRUE(reported) << runtime_error_message(reported.error());
  const auto readiness_deadline = std::chrono::steady_clock::now() + 5s;
  expected<bool, Error> ready{false};
  while (std::chrono::steady_clock::now() < readiness_deadline) {
    ready = (*subscriber)->has_samples();
    ASSERT_TRUE(ready) << runtime_error_message(ready.error());
    if (*ready) {
      break;
    }
    std::this_thread::sleep_for(1ms);
  }

  // Key assertion: has_samples reports the queued health lane and the same operation drains as
  // the first ordered clock-health record.
  ASSERT_TRUE(*ready);
  auto operation = wait_for_catalog_tensor_schema_operation(**subscriber);
  ASSERT_TRUE(operation) << runtime_error_message(operation.error());
  ASSERT_EQ(operation->kind, channel_internal::CatalogReceivedOperationKind::kClockHealth);
  EXPECT_EQ(operation->clock_health.publisher_operation_sequence, 1U);
  EXPECT_EQ(operation->clock_health.report_sequence, 1U);
}

// Descriptor-schema EOS is authoritative lifecycle control only after the complete terminal
// record, including its protected lane, has passed canonical validation.
// REQ: EXTERNAL-009
// REQ: SYSTEM-024
TEST(RuntimeExternalChannelTest, MalformedDescriptorSchemaEndOfStreamCannotAuthorizeRebind) {
  enum class Malformation : std::uint8_t {
    kMetadata,
    kPayload,
    kLane,
  };
  constexpr std::array kMalformations{
      Malformation::kMetadata, Malformation::kPayload, Malformation::kLane};
  using Envelope = channel_internal::TensorSchemaRuntimeEnvelope<ExternalImagePayload>;
  constexpr std::size_t kBucketSize =
      channel_internal::inline_payload_bucket_size(sizeof(Envelope)).value_or(0U);
  static_assert(kBucketSize != 0U);
  using Bucket = channel_internal::TensorSchemaPayloadBucket<kBucketSize>;

  for (const Malformation malformation : kMalformations) {
    SCOPED_TRACE(static_cast<std::uint32_t>(malformation));
    const std::string endpoint_name = unique_endpoint_name();
    Graph graph{"runtime-external-malformed-image-eos"};
    const ClockKey runtime_clock = graph.add_clock<RealtimeClock>("runtime-clock");
    graph.set_default_clock(runtime_clock);
    const auto imported =
        graph.import_topic<schema::ImageT>("holomq://" + endpoint_name,
                                           external_image_import_options(endpoint_name),
                                           external_image_contract());
    const auto sink = graph.op<ExternalImageImportSink>("sink");
    graph.add_flow(imported.output(), sink->input, ConnectionOptions{.queue_depth = 1U});

    const ExecutionPlan plan = compile(graph);
    ASSERT_TRUE(plan.ok()) << plan.json();
    const auto compiled_endpoint = plan.endpoint_catalog().endpoint(endpoint_name);
    ASSERT_TRUE(compiled_endpoint) << runtime_error_message(compiled_endpoint.error());
    const auto* const compiled_record =
        detail::ChannelEndpointCatalogAccess::record(*compiled_endpoint);
    ASSERT_NE(compiled_record, nullptr);
    auto lossy_record = *compiled_record;
    lossy_record.queue_policy = QueuePolicy::kDropOldest;
    lossy_record.service_static.queue_policy = QueuePolicy::kDropOldest;
    auto lossy_catalog = detail::ChannelEndpointCatalogAccess::normalize(
        "runtime-external-malformed-image-eos", {std::move(lossy_record)});
    ASSERT_TRUE(lossy_catalog) << runtime_error_message(lossy_catalog.error());
    const auto endpoint = lossy_catalog->endpoint(endpoint_name);
    ASSERT_TRUE(endpoint) << runtime_error_message(endpoint.error());
    const auto* const endpoint_record = detail::ChannelEndpointCatalogAccess::record(*endpoint);
    const auto* const type_ops = detail::runtime_port_type_ops<schema::ImageT>();
    ASSERT_NE(endpoint_record, nullptr);
    ASSERT_NE(type_ops, nullptr);
    ASSERT_TRUE(endpoint_record->payload_identity.schema);
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access): guarded by ASSERT_TRUE above.
    const auto& payload_schema = *endpoint_record->payload_identity.schema;
    ASSERT_EQ(endpoint_record->service_static.queue_policy, QueuePolicy::kDropOldest);

    auto subscriber_runtime = ChannelRuntime::create();
    ASSERT_TRUE(subscriber_runtime) << runtime_error_message(subscriber_runtime.error());
    auto subscriber = detail::RuntimeChannelBindingAccess::create_catalog_tensor_schema_subscriber(
        *subscriber_runtime,
        *endpoint,
        payload_schema,
        type_ops->wire_capacity,
        type_ops->descriptor_tensor_static_count);
    ASSERT_TRUE(subscriber) << runtime_error_message(subscriber.error());
    auto publisher_context = holomq::ContextBuilder().with_vmm_fd_export(true).create();
    ASSERT_TRUE(publisher_context) << publisher_context.error().message;

    const std::string service =
        detail::ChannelEndpointCatalogAccess::backend_service_binding(*endpoint);
    const holomq::PubSubServiceOptions data_service_options{
        .queue_policy = holomq::QueuePolicy::kDropOldest,
        .max_subscribers = 1U,
        .max_subscriber_queue_depth = endpoint_record->service_static.max_subscriber_depth,
        .max_subscriber_borrowed_samples = endpoint_record->service_static.max_subscriber_borrowed,
        .publisher_history_depth = endpoint_record->service_static.publisher_history_depth,
        .enable_event_wake = endpoint_record->service_static.event_wake,
    };
    auto data_publisher = publisher_context->publisher_builder(service)
                              .service_options(data_service_options)
                              .publisher_options(holomq::PublisherPubSubOptions{
                                  .max_loaned_samples = 1U,
                                  .expected_recipient_count = 1U,
                              })
                              .payload_type<Bucket>()
                              .create();
    ASSERT_TRUE(data_publisher) << data_publisher.error().message;
    const holomq::PubSubServiceOptions terminal_service_options{
        .queue_policy = holomq::QueuePolicy::kBlocking,
        .max_subscribers = 1U,
        .max_subscriber_queue_depth = 1U,
        .max_subscriber_borrowed_samples = 1U,
        .publisher_history_depth = 0U,
        .enable_event_wake = endpoint_record->service_static.event_wake,
    };
    auto terminal_publisher = publisher_context->publisher_builder(service + ".hsdk-terminal-v1")
                                  .service_options(terminal_service_options)
                                  .publisher_options(holomq::PublisherPubSubOptions{
                                      .max_loaned_samples = 1U,
                                      .expected_recipient_count = 1U,
                                  })
                                  .payload_type<Bucket>()
                                  .create();
    ASSERT_TRUE(terminal_publisher) << terminal_publisher.error().message;

    const auto connection_deadline = std::chrono::steady_clock::now() + 2s;
    bool connected = false;
    while (!connected && std::chrono::steady_clock::now() < connection_deadline) {
      ASSERT_TRUE((*subscriber)->has_samples());
      ASSERT_TRUE(data_publisher->update_connections());
      ASSERT_TRUE(terminal_publisher->update_connections());
      connected = data_publisher->has_subscribers() && terminal_publisher->has_subscribers();
      if (!connected) {
        std::this_thread::sleep_for(1ms);
      }
    }
    ASSERT_TRUE(connected);

    // Key setup: forge one EOS with either noncanonical metadata, nonempty payload bytes, or the
    // lossy data lane instead of the protected terminal lane.
    auto* selected_publisher =
        malformation == Malformation::kLane ? &*data_publisher : &*terminal_publisher;
    auto terminal = selected_publisher->loan();
    ASSERT_TRUE(terminal) << terminal.error().message;
    terminal->payload() = Bucket{
        .operation = channel_internal::InlinePayloadOperation::kEndOfStream,
        .metadata =
            SampleMetadata{
                .sequence = 1U,
                .trace_id = malformation == Malformation::kMetadata ? 7U : 0U,
                .payload_kind = PayloadKind::kDescriptorBacked,
            },
        .producer_generation = 1U,
        .payload_size = malformation == Malformation::kPayload ? 1U : 0U,
    };
    auto sent = selected_publisher->send(std::move(*terminal));
    ASSERT_TRUE(sent) << sent.error().message;
    auto received = wait_for_catalog_tensor_schema_operation(**subscriber);

    // Key assertion: validation rejects before generation/EOS authority is latched, so lifecycle
    // replacement cannot mistake the malformed terminal record for a completed stream.
    ASSERT_FALSE(received);
    EXPECT_EQ(received.error().code, ErrorCode::kProtocolError);
    auto generation =
        detail::RuntimeChannelBindingAccess::latched_catalog_tensor_schema_producer_generation(
            **subscriber);
    ASSERT_TRUE(generation) << runtime_error_message(generation.error());
    EXPECT_EQ(*generation, 0U);
    auto quiescence =
        detail::RuntimeChannelBindingAccess::catalog_tensor_schema_import_quiescence(**subscriber);
    ASSERT_TRUE(quiescence) << runtime_error_message(quiescence.error());
    EXPECT_EQ(quiescence->latched_producer_generation, 0U);
  }
}

// Raw Tensor imports use the same ordered catalog health protocol as descriptor-backed schemas.
// A healthy report authorizes the following sample, while a preceding unhealthy report suppresses
// data before the Tensor reaches compute.
// REQ: CHAN-069
// REQ: EXTERNAL-010
// REQ: RUNTIME-016
// REQ: RUNTIME-017
TEST(RuntimeExternalChannelTest, RawTensorImportAppliesPrecedingClockHealthBeforeCompute) {
  ExternalTensorExportSink::reset();
  const std::string endpoint_name = unique_endpoint_name();

  Graph graph{"runtime-external-tensor-health"};
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
  const auto imported = graph.import_topic<Tensor>("holomq://" + endpoint_name, std::move(options));
  const auto sink = graph.op<ExternalTensorExportSink>("sink");
  graph.add_flow(imported.output(), sink->input, ConnectionOptions{.queue_depth = 1U});

  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  auto external_runtime = ChannelRuntime::create();
  ASSERT_TRUE(external_runtime) << runtime_error_message(external_runtime.error());
  auto publisher =
      external_runtime->create_publisher<Tensor>(plan.endpoint_catalog(), endpoint_name);
  ASSERT_TRUE(publisher) << runtime_error_message(publisher.error());

  // Key setup: place healthy and unhealthy health operations immediately before two raw Tensor
  // publications on the same catalog sequence.
  RunSession session = run_async(plan);
  ASSERT_TRUE(wait_for_subscriber(*publisher));
  ASSERT_TRUE(publisher->report_clock_health(ClockHealthReport::healthy()));
  auto first_capture = publisher->make_capture_time(1);
  ASSERT_TRUE(first_capture) << runtime_error_message(first_capture.error());
  ASSERT_TRUE(publish_external_host_tensor(*publisher, *first_capture));
  ASSERT_TRUE(ExternalTensorExportSink::wait_for_sample());

  ASSERT_TRUE(publisher->report_clock_health(
      ClockHealthReport::unhealthy(ClockHealthReason::kSynchronizationLockLost)));
  auto suppressed_capture = publisher->make_capture_time(2);
  ASSERT_TRUE(suppressed_capture) << runtime_error_message(suppressed_capture.error());
  ASSERT_TRUE(publish_external_host_tensor(*publisher, *suppressed_capture));
  std::this_thread::sleep_for(100ms);

  session.request_stop();
  session.wait();
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
  // Key assertion: raw Tensor ingress exposes only the healthy-authorized sample without failing
  // the session on the deliberately suppressed publication.
  EXPECT_EQ(ExternalTensorExportSink::received.load(std::memory_order_acquire), 1U);
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

  const auto snapshot = detail::GraphCompilerAccess::canonical_snapshot(graph);
  std::vector<detail::TensorDeviceBindingRecord> device_bindings;
  device_bindings.reserve(snapshot->ports.size());
  for (const auto& port : snapshot->ports) {
    if (port.direction == PortDirection::kOutput &&
        port.payload_identity.kind == PayloadIdentityKind::kTensor) {
      device_bindings.push_back(detail::TensorDeviceBindingRecord{
          .port = PortId{port.id.value}, .device = DeviceId{device_ordinal}});
    }
  }

  // Key setup: compile a monitored raw Tensor import onto the current CUDA device. This import is
  // controller-drained and intentionally absent from the background-pump subscriber inventory.
  const ExecutionPlan plan = detail::compile_with_runtime_evidence(
      graph,
      {},
      detail::RuntimeCompileEvidenceView{
          .package_capabilities = detail::runtime_package_capability_view(),
          .tensor_device_bindings = device_bindings,
      });
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

// A replacement descriptor-schema attachment starts after an earlier subscriber consumed data
// and health operations. Its first visible data and first visible health operation establish the
// new attachment's sequence baseline without weakening strict ordering after either baseline.
// REQ: CHAN-069
// REQ: RUNTIME-008
TEST(RuntimeExternalChannelTest,
     DescriptorTensorSchemaReplacementAttachmentBaselinesDataAndClockHealth) {
  const std::string endpoint_name = unique_endpoint_name();

  Graph graph{"runtime-external-image-attachment-baseline"};
  const ClockKey capture_clock = graph.add_clock<ManualClock>("capture-clock", std::int64_t{0});
  const ClockKey runtime_clock = graph.add_clock<RealtimeClock>("runtime-clock");
  graph.set_default_clock(runtime_clock);
  graph.add_clock_conversion(capture_clock, runtime_clock, AffineClockConversion{});
  TopicOptions options = external_image_import_options(endpoint_name);
  options.capture_time = RequireCaptureTime{.clock = capture_clock};
  options.clock_health = ClockHealthOptions{
      .capability = ClockHealthCapability::kStatusReports,
      .requirement = ClockHealthRequirement::kFreshHealthyReport,
      .max_report_age = 5s,
  };
  const auto imported = graph.import_topic<schema::ImageT>(
      "holomq://" + endpoint_name, std::move(options), external_image_contract());
  const auto sink = graph.op<ExternalImageImportSink>("sink");
  graph.add_flow(imported.output(), sink->input, ConnectionOptions{.queue_depth = 1U});

  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto endpoint = plan.endpoint_catalog().endpoint(endpoint_name);
  ASSERT_TRUE(endpoint);
  const auto* const endpoint_record = detail::ChannelEndpointCatalogAccess::record(*endpoint);
  const auto* const type_ops = detail::runtime_port_type_ops<schema::ImageT>();
  ASSERT_NE(endpoint_record, nullptr);
  ASSERT_NE(type_ops, nullptr);
  ASSERT_TRUE(endpoint_record->payload_identity.schema);
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access): guarded by ASSERT_TRUE above.
  const auto& payload_schema = *endpoint_record->payload_identity.schema;

  auto graph_runtime = ChannelRuntime::create();
  auto external_runtime = ChannelRuntime::create();
  ASSERT_TRUE(graph_runtime) << runtime_error_message(graph_runtime.error());
  ASSERT_TRUE(external_runtime) << runtime_error_message(external_runtime.error());
  auto publisher = external_runtime->create_tensor_schema_publisher<ExternalImagePayload>(
      plan.endpoint_catalog(), endpoint_name);
  ASSERT_TRUE(publisher) << runtime_error_message(publisher.error());
  const auto create_subscriber = [&]() {
    return detail::RuntimeChannelBindingAccess::create_catalog_tensor_schema_subscriber(
        *graph_runtime,
        *endpoint,
        payload_schema,
        type_ops->wire_capacity,
        type_ops->descriptor_tensor_static_count);
  };
  const auto emit = [&](const std::uint8_t value,
                        const std::int64_t capture_timestamp_ns) -> expected<void, Error> {
    auto tensor = make_external_image_tensor(value);
    if (!tensor) {
      return make_unexpected(std::move(tensor).error());
    }
    auto metadata = schema::build_image_metadata(
        *tensor,
        schema::ImageMetadataOptions{
            .height = 1U, .width = 8U, .encoding = schema::ImageEncoding_MONO8});
    if (!metadata) {
      return make_unexpected(std::move(metadata).error());
    }
    auto emitted = publisher->emit_captured(*metadata, std::move(*tensor), capture_timestamp_ns);
    return emitted ? expected<void, Error>{}
                   : expected<void, Error>{make_unexpected(std::move(emitted).error())};
  };

  auto first = create_subscriber();
  ASSERT_TRUE(first) << runtime_error_message(first.error());
  ASSERT_TRUE(wait_for_schema_subscriber(*publisher));
  ASSERT_TRUE(emit(0x11U, 1));
  auto first_data = wait_for_catalog_tensor_schema_operation(**first);
  ASSERT_TRUE(first_data) << runtime_error_message(first_data.error());
  ASSERT_EQ(first_data->kind, channel_internal::CatalogReceivedOperationKind::kData);
  EXPECT_EQ(first_data->metadata.sequence, 1U);
  (*first)->release_payload_loan();
  ASSERT_TRUE(publisher->report_clock_health(ClockHealthReport::healthy()));
  auto first_health = wait_for_catalog_tensor_schema_operation(**first);
  ASSERT_TRUE(first_health) << runtime_error_message(first_health.error());
  ASSERT_EQ(first_health->kind, channel_internal::CatalogReceivedOperationKind::kClockHealth);
  EXPECT_EQ(first_health->clock_health.publisher_operation_sequence, 2U);
  EXPECT_EQ(first_health->clock_health.report_sequence, 1U);
  first->reset();
  ASSERT_TRUE(wait_for_no_subscribers(*publisher));

  // Key setup: replace the receive attachment after the publisher has already consumed one data
  // and one health operation, so neither sequence domain starts at one for this subscriber.
  auto data_baseline = create_subscriber();
  ASSERT_TRUE(data_baseline) << runtime_error_message(data_baseline.error());
  ASSERT_TRUE(wait_for_schema_subscriber(*publisher));
  ASSERT_TRUE(emit(0x22U, 2));
  auto later_data = wait_for_catalog_tensor_schema_operation(**data_baseline);
  ASSERT_TRUE(later_data) << runtime_error_message(later_data.error());
  // Key assertion: the first visible data operation establishes both the operation-sequence and
  // carried health-report baselines; the immediately following health report remains contiguous.
  ASSERT_EQ(later_data->kind, channel_internal::CatalogReceivedOperationKind::kData);
  EXPECT_EQ(later_data->metadata.sequence, 3U);
  (*data_baseline)->release_payload_loan();
  ASSERT_TRUE(publisher->report_clock_health(ClockHealthReport::healthy()));
  auto later_health = wait_for_catalog_tensor_schema_operation(**data_baseline);
  ASSERT_TRUE(later_health) << runtime_error_message(later_health.error());
  ASSERT_EQ(later_health->kind, channel_internal::CatalogReceivedOperationKind::kClockHealth);
  EXPECT_EQ(later_health->clock_health.publisher_operation_sequence, 4U);
  EXPECT_EQ(later_health->clock_health.report_sequence, 2U);
  data_baseline->reset();
  ASSERT_TRUE(wait_for_no_subscribers(*publisher));

  auto health_baseline = create_subscriber();
  ASSERT_TRUE(health_baseline) << runtime_error_message(health_baseline.error());
  ASSERT_TRUE(wait_for_schema_subscriber(*publisher));
  ASSERT_TRUE(publisher->report_clock_health(ClockHealthReport::healthy()));
  auto first_visible_health = wait_for_catalog_tensor_schema_operation(**health_baseline);
  ASSERT_TRUE(first_visible_health) << runtime_error_message(first_visible_health.error());
  // Key assertion: a health-first replacement independently adopts both nonzero sequence values.
  ASSERT_EQ(first_visible_health->kind,
            channel_internal::CatalogReceivedOperationKind::kClockHealth);
  EXPECT_EQ(first_visible_health->clock_health.publisher_operation_sequence, 5U);
  EXPECT_EQ(first_visible_health->clock_health.report_sequence, 3U);
}

// Catalog descriptor-schema sequencing must arbitrate a final data record ahead of protected EOS
// when the terminal becomes visible after the receiver's first empty data probe.
// REQ: CHAN-136
TEST(RuntimeExternalChannelTest,
     DescriptorTensorSchemaCatalogProtectedTerminalRechecksFinalDataAfterInitialProbe) {
  const std::string endpoint_name = unique_endpoint_name();
  Graph graph{"runtime-external-image-terminal-recheck"};
  const ClockKey runtime_clock = graph.add_clock<RealtimeClock>("runtime-clock");
  graph.set_default_clock(runtime_clock);
  const auto imported =
      graph.import_topic<schema::ImageT>("holomq://" + endpoint_name,
                                         external_image_import_options(endpoint_name),
                                         external_image_contract());
  const auto sink = graph.op<ExternalImageImportSink>("sink");
  graph.add_flow(imported.output(),
                 sink->input,
                 ConnectionOptions{.queue_policy = QueuePolicy::kDropOldest, .queue_depth = 1U});

  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto endpoint = plan.endpoint_catalog().endpoint(endpoint_name);
  ASSERT_TRUE(endpoint) << runtime_error_message(endpoint.error());
  const auto* const endpoint_record = detail::ChannelEndpointCatalogAccess::record(*endpoint);
  const auto* const type_ops = detail::runtime_port_type_ops<schema::ImageT>();
  ASSERT_NE(endpoint_record, nullptr);
  ASSERT_NE(type_ops, nullptr);
  ASSERT_TRUE(endpoint_record->payload_identity.schema);
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access): guarded by ASSERT_TRUE above.
  const auto& payload_schema = *endpoint_record->payload_identity.schema;

  auto subscriber_runtime = ChannelRuntime::create();
  auto publisher_runtime = ChannelRuntime::create();
  ASSERT_TRUE(subscriber_runtime) << runtime_error_message(subscriber_runtime.error());
  ASSERT_TRUE(publisher_runtime) << runtime_error_message(publisher_runtime.error());
  channel_internal::TensorPublisherTestAccess::ReceiveProbeBarrier barrier;
  channel_internal::TensorPublisherTestAccess::
      attach_receive_probe_barrier_to_next_subscriber_for_test(barrier);
  auto subscriber = detail::RuntimeChannelBindingAccess::create_catalog_tensor_schema_subscriber(
      *subscriber_runtime,
      *endpoint,
      payload_schema,
      type_ops->wire_capacity,
      type_ops->descriptor_tensor_static_count);
  auto publisher = publisher_runtime->create_tensor_schema_publisher<ExternalImagePayload>(
      plan.endpoint_catalog(), endpoint_name);
  ASSERT_TRUE(subscriber) << runtime_error_message(subscriber.error());
  ASSERT_TRUE(publisher) << runtime_error_message(publisher.error());
  ASSERT_TRUE(wait_for_catalog_schema_subscriber(*publisher, **subscriber));

  auto receive_future = std::async(std::launch::async, [&subscriber] {
    return detail::RuntimeChannelBindingAccess::receive_catalog_tensor_schema_operation(
        **subscriber, true);
  });
  const bool first_data_probe_completed = barrier.reached.try_acquire_for(2s);
  std::optional<expected<std::uint32_t, Error>> emitted;
  std::optional<expected<void, Error>> closed;
  if (first_data_probe_completed) {
    auto tensor = make_external_image_tensor(0x39U);
    if (tensor) {
      auto metadata = schema::build_image_metadata(
          *tensor,
          schema::ImageMetadataOptions{
              .height = 1U, .width = 8U, .encoding = schema::ImageEncoding_MONO8});
      if (metadata) {
        emitted.emplace(publisher->emit(*metadata, std::move(*tensor)));
      }
    }
    closed.emplace(publisher->close());
  }
  barrier.resume.release();

  auto data = receive_future.get();
  ASSERT_TRUE(first_data_probe_completed);
  ASSERT_TRUE(emitted.has_value());
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access): guarded by ASSERT_TRUE above.
  ASSERT_TRUE(*emitted) << runtime_error_message(emitted->error());
  ASSERT_TRUE(closed.has_value());
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access): guarded by ASSERT_TRUE above.
  ASSERT_TRUE(*closed) << runtime_error_message(closed->error());
  ASSERT_TRUE(data) << runtime_error_message(data.error());
  ASSERT_EQ(data->kind, channel_internal::CatalogReceivedOperationKind::kData);
  EXPECT_EQ(data->metadata.sequence, 1U);
  (*subscriber)->release_payload_loan();

  auto terminal = detail::RuntimeChannelBindingAccess::receive_catalog_tensor_schema_operation(
      **subscriber, true);
  ASSERT_TRUE(terminal) << runtime_error_message(terminal.error());
  EXPECT_EQ(terminal->kind, channel_internal::CatalogReceivedOperationKind::kEndOfStream);
  EXPECT_EQ(terminal->metadata.sequence, 2U);
}

// Raw Tensor attachments use the same sequence-baseline rule as descriptor-schema attachments.
// Exercise both a first-visible data sample carrying an established health watermark and a
// first-visible health report after replacing an earlier subscriber.
// REQ: CHAN-069
// REQ: RUNTIME-008
TEST(RuntimeExternalChannelTest, RawTensorReplacementAttachmentBaselinesDataAndClockHealth) {
  const std::string endpoint_name = unique_endpoint_name();

  Graph graph{"runtime-external-tensor-attachment-baseline"};
  const ClockKey capture_clock = graph.add_clock<ManualClock>("capture-clock", std::int64_t{0});
  const ClockKey runtime_clock = graph.add_clock<RealtimeClock>("runtime-clock");
  graph.set_default_clock(runtime_clock);
  graph.add_clock_conversion(capture_clock, runtime_clock, AffineClockConversion{});
  TopicOptions options = optional_tensor_export_options(endpoint_name);
  options.bounds.subscriber.reset();
  options.bounds.publisher = PublisherEndpointBounds{.max_loaned_samples = 1U};
  options.capture_time = RequireCaptureTime{.clock = capture_clock};
  options.clock_health = ClockHealthOptions{
      .capability = ClockHealthCapability::kStatusReports,
      .requirement = ClockHealthRequirement::kFreshHealthyReport,
      .max_report_age = 5s,
  };
  const auto imported = graph.import_topic<Tensor>("holomq://" + endpoint_name, std::move(options));
  const auto sink = graph.op<ExternalTensorExportSink>("sink");
  graph.add_flow(imported.output(), sink->input, ConnectionOptions{.queue_depth = 1U});

  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto endpoint = plan.endpoint_catalog().endpoint(endpoint_name);
  ASSERT_TRUE(endpoint);
  auto graph_runtime = ChannelRuntime::create();
  auto external_runtime = ChannelRuntime::create();
  ASSERT_TRUE(graph_runtime) << runtime_error_message(graph_runtime.error());
  ASSERT_TRUE(external_runtime) << runtime_error_message(external_runtime.error());
  auto publisher =
      external_runtime->create_publisher<Tensor>(plan.endpoint_catalog(), endpoint_name);
  ASSERT_TRUE(publisher) << runtime_error_message(publisher.error());
  const auto create_subscriber = [&]() {
    return detail::RuntimeChannelBindingAccess::create_catalog_tensor_subscriber(
        *graph_runtime, *endpoint, 0U);
  };
  const auto publish = [&](const std::int64_t capture_timestamp_ns) -> expected<void, Error> {
    auto capture_time = publisher->make_capture_time(capture_timestamp_ns);
    if (!capture_time) {
      return make_unexpected(std::move(capture_time).error());
    }
    return publish_external_host_tensor(*publisher, *capture_time);
  };

  auto created_first = create_subscriber();
  ASSERT_TRUE(created_first) << runtime_error_message(created_first.error());
  std::optional<ChannelSubscriber<Tensor>> first{std::move(*created_first)};
  ASSERT_TRUE(wait_for_catalog_tensor_subscriber(*publisher, *first));
  ASSERT_TRUE(publish(1));
  auto first_data = wait_for_catalog_tensor_operation(*first);
  ASSERT_TRUE(first_data) << runtime_error_message(first_data.error());
  ASSERT_EQ(first_data->kind, channel_internal::CatalogReceivedOperationKind::kData);
  EXPECT_EQ(first_data->metadata.sequence, 1U);
  first_data->tensor_sample.reset();
  ASSERT_TRUE(publisher->report_clock_health(ClockHealthReport::healthy()));
  auto first_health = wait_for_catalog_tensor_operation(*first);
  ASSERT_TRUE(first_health) << runtime_error_message(first_health.error());
  ASSERT_EQ(first_health->kind, channel_internal::CatalogReceivedOperationKind::kClockHealth);
  EXPECT_EQ(first_health->clock_health.publisher_operation_sequence, 2U);
  EXPECT_EQ(first_health->clock_health.report_sequence, 1U);
  first.reset();
  ASSERT_TRUE(wait_for_no_subscribers(*publisher));

  // Key setup: replace the receive attachment after one raw Tensor and one health report have
  // advanced the publisher's independent operation and report sequence domains.
  auto created_data_baseline = create_subscriber();
  ASSERT_TRUE(created_data_baseline) << runtime_error_message(created_data_baseline.error());
  std::optional<ChannelSubscriber<Tensor>> data_baseline{std::move(*created_data_baseline)};
  ASSERT_TRUE(wait_for_catalog_tensor_subscriber(*publisher, *data_baseline));
  ASSERT_TRUE(publish(2));
  auto later_data = wait_for_catalog_tensor_operation(*data_baseline);
  ASSERT_TRUE(later_data) << runtime_error_message(later_data.error());
  // Key assertion: first-visible raw data adopts the carried health watermark and later health
  // still has to be exactly contiguous in both sequence domains.
  ASSERT_EQ(later_data->kind, channel_internal::CatalogReceivedOperationKind::kData);
  EXPECT_EQ(later_data->metadata.sequence, 3U);
  later_data->tensor_sample.reset();
  ASSERT_TRUE(publisher->report_clock_health(ClockHealthReport::healthy()));
  auto later_health = wait_for_catalog_tensor_operation(*data_baseline);
  ASSERT_TRUE(later_health) << runtime_error_message(later_health.error());
  ASSERT_EQ(later_health->kind, channel_internal::CatalogReceivedOperationKind::kClockHealth);
  EXPECT_EQ(later_health->clock_health.publisher_operation_sequence, 4U);
  EXPECT_EQ(later_health->clock_health.report_sequence, 2U);
  data_baseline.reset();
  ASSERT_TRUE(wait_for_no_subscribers(*publisher));

  auto created_health_baseline = create_subscriber();
  ASSERT_TRUE(created_health_baseline) << runtime_error_message(created_health_baseline.error());
  std::optional<ChannelSubscriber<Tensor>> health_baseline{std::move(*created_health_baseline)};
  ASSERT_TRUE(wait_for_catalog_tensor_subscriber(*publisher, *health_baseline));
  ASSERT_TRUE(publisher->report_clock_health(ClockHealthReport::healthy()));
  auto first_visible_health = wait_for_catalog_tensor_operation(*health_baseline);
  ASSERT_TRUE(first_visible_health) << runtime_error_message(first_visible_health.error());
  // Key assertion: a health-first raw attachment accepts the established nonzero baseline.
  ASSERT_EQ(first_visible_health->kind,
            channel_internal::CatalogReceivedOperationKind::kClockHealth);
  EXPECT_EQ(first_visible_health->clock_health.publisher_operation_sequence, 5U);
  EXPECT_EQ(first_visible_health->clock_health.report_sequence, 3U);
}

// A replacement subscriber may briefly observe operations from the explicitly superseded and
// accepted producer generations on different catalog lanes. The known-retired generation must be
// removed before equal-sequence ambiguity is evaluated.
// REQ: CHAN-069
// REQ: RUNTIME-008
TEST(RuntimeExternalChannelTest,
     DescriptorTensorSchemaSupersededHealthDoesNotConflictWithReplacementData) {
  const std::string endpoint_name = unique_endpoint_name();

  Graph graph{"runtime-external-image-superseded-tie"};
  const ClockKey capture_clock = graph.add_clock<ManualClock>("capture-clock", std::int64_t{0});
  const ClockKey runtime_clock = graph.add_clock<RealtimeClock>("runtime-clock");
  graph.set_default_clock(runtime_clock);
  graph.add_clock_conversion(capture_clock, runtime_clock, AffineClockConversion{});
  TopicOptions options = external_image_import_options(endpoint_name);
  options.capture_time = RequireCaptureTime{.clock = capture_clock};
  options.clock_health = ClockHealthOptions{
      .capability = ClockHealthCapability::kStatusReports,
      .requirement = ClockHealthRequirement::kFreshHealthyReport,
      .max_report_age = 5s,
  };
  const auto imported = graph.import_topic<schema::ImageT>(
      "holomq://" + endpoint_name, std::move(options), external_image_contract());
  const auto sink = graph.op<ExternalImageImportSink>("sink");
  graph.add_flow(imported.output(), sink->input, ConnectionOptions{.queue_depth = 1U});

  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto endpoint = plan.endpoint_catalog().endpoint(endpoint_name);
  ASSERT_TRUE(endpoint) << runtime_error_message(endpoint.error());
  const auto* const endpoint_record = detail::ChannelEndpointCatalogAccess::record(*endpoint);
  const auto* const type_ops = detail::runtime_port_type_ops<schema::ImageT>();
  ASSERT_NE(endpoint_record, nullptr);
  ASSERT_NE(type_ops, nullptr);
  ASSERT_TRUE(endpoint_record->payload_identity.schema);

  auto graph_runtime = ChannelRuntime::create();
  auto external_runtime = ChannelRuntime::create();
  ASSERT_TRUE(graph_runtime) << runtime_error_message(graph_runtime.error());
  ASSERT_TRUE(external_runtime) << runtime_error_message(external_runtime.error());
  const auto create_subscriber = [&]() {
    return detail::RuntimeChannelBindingAccess::create_catalog_tensor_schema_subscriber(
        *graph_runtime,
        *endpoint,
        *endpoint_record->payload_identity.schema,
        type_ops->wire_capacity,
        type_ops->descriptor_tensor_static_count);
  };
  const auto emit = [&](ChannelTensorSchemaPublisher<ExternalImagePayload>& publisher,
                        const std::uint8_t value,
                        const std::int64_t capture_timestamp_ns) -> expected<void, Error> {
    auto tensor = make_external_image_tensor(value);
    if (!tensor) {
      return make_unexpected(std::move(tensor).error());
    }
    auto metadata = schema::build_image_metadata(
        *tensor,
        schema::ImageMetadataOptions{
            .height = 1U, .width = 8U, .encoding = schema::ImageEncoding_MONO8});
    if (!metadata) {
      return make_unexpected(std::move(metadata).error());
    }
    auto emitted = publisher.emit_captured(*metadata, std::move(*tensor), capture_timestamp_ns);
    return emitted ? expected<void, Error>{}
                   : expected<void, Error>{make_unexpected(std::move(emitted).error())};
  };

  auto first = create_subscriber();
  ASSERT_TRUE(first) << runtime_error_message(first.error());
  auto created_retired = external_runtime->create_tensor_schema_publisher<ExternalImagePayload>(
      plan.endpoint_catalog(), endpoint_name);
  ASSERT_TRUE(created_retired) << runtime_error_message(created_retired.error());
  std::optional<ChannelTensorSchemaPublisher<ExternalImagePayload>> retired{
      std::move(*created_retired)};
  ASSERT_TRUE(wait_for_catalog_schema_subscriber(*retired, **first));
  ASSERT_TRUE(emit(*retired, 0x11U, 1));
  auto retired_data = wait_for_catalog_tensor_schema_operation(**first);
  ASSERT_TRUE(retired_data) << runtime_error_message(retired_data.error());
  ASSERT_EQ(retired_data->kind, channel_internal::CatalogReceivedOperationKind::kData);
  (*first)->release_payload_loan();
  auto retired_generation =
      detail::RuntimeChannelBindingAccess::latched_catalog_tensor_schema_producer_generation(
          **first);
  ASSERT_TRUE(retired_generation) << runtime_error_message(retired_generation.error());
  ASSERT_NE(*retired_generation, 0U);

  auto replacement = external_runtime->create_tensor_schema_publisher<ExternalImagePayload>(
      plan.endpoint_catalog(), endpoint_name);
  ASSERT_TRUE(replacement) << runtime_error_message(replacement.error());
  ASSERT_TRUE(wait_for_catalog_schema_subscriber(*replacement, **first));
  ASSERT_TRUE(replacement->report_clock_health(ClockHealthReport::healthy()));
  auto replacement_warmup = wait_for_catalog_tensor_schema_operation(**first);
  ASSERT_TRUE(replacement_warmup) << runtime_error_message(replacement_warmup.error());
  ASSERT_EQ(replacement_warmup->kind,
            channel_internal::CatalogReceivedOperationKind::kPeerIncarnationChanged);
  first->reset();
  ASSERT_TRUE(wait_for_no_subscribers(*retired));
  ASSERT_TRUE(wait_for_no_subscribers(*replacement));

  auto current = create_subscriber();
  ASSERT_TRUE(current) << runtime_error_message(current.error());
  ASSERT_TRUE(
      detail::RuntimeChannelBindingAccess::supersede_catalog_tensor_schema_producer_generation(
          **current, *retired_generation));
  ASSERT_TRUE(wait_for_catalog_schema_subscriber(*retired, **current));
  ASSERT_TRUE(wait_for_catalog_schema_subscriber(*replacement, **current));

  // Key setup: queue equal operation sequence two on separate lanes: retired clock health and
  // replacement descriptor-Tensor data.
  ASSERT_TRUE(retired->report_clock_health(ClockHealthReport::healthy()));
  ASSERT_TRUE(emit(*replacement, 0x42U, 2));

  auto pending =
      detail::RuntimeChannelBindingAccess::catalog_tensor_schema_import_quiescence(**current);
  ASSERT_TRUE(pending) << runtime_error_message(pending.error());
  EXPECT_TRUE(pending->pending_current);
  EXPECT_TRUE(pending->pending_foreign);

  // Key assertion: the stale lane reports a generation cut instead of a protocol conflict, then
  // the replacement data remains queued and establishes the accepted producer generation.
  auto fenced = wait_for_catalog_tensor_schema_operation(**current);
  ASSERT_TRUE(fenced) << runtime_error_message(fenced.error());
  ASSERT_EQ(fenced->kind, channel_internal::CatalogReceivedOperationKind::kPeerIncarnationChanged);
  auto accepted = wait_for_catalog_tensor_schema_operation(**current);
  ASSERT_TRUE(accepted) << runtime_error_message(accepted.error());
  ASSERT_EQ(accepted->kind, channel_internal::CatalogReceivedOperationKind::kData);
  EXPECT_EQ(accepted->metadata.sequence, 2U);
  (*current)->release_payload_loan();
  auto accepted_generation =
      detail::RuntimeChannelBindingAccess::latched_catalog_tensor_schema_producer_generation(
          **current);
  ASSERT_TRUE(accepted_generation) << runtime_error_message(accepted_generation.error());
  EXPECT_NE(*accepted_generation, *retired_generation);

  // The endpoint admits two publishers; the retired lane is fully fenced, so release that
  // publisher before reusing its slot for an unknown third incarnation.
  retired.reset();
  auto foreign = external_runtime->create_tensor_schema_publisher<ExternalImagePayload>(
      plan.endpoint_catalog(), endpoint_name);
  ASSERT_TRUE(foreign) << runtime_error_message(foreign.error());
  ASSERT_TRUE(wait_for_catalog_schema_subscriber(*foreign, **current));
  ASSERT_TRUE(emit(*foreign, 0x11U, 1));
  auto foreign_first = wait_for_catalog_tensor_schema_operation(**current);
  ASSERT_TRUE(foreign_first) << runtime_error_message(foreign_first.error());
  EXPECT_EQ(foreign_first->kind,
            channel_internal::CatalogReceivedOperationKind::kPeerIncarnationChanged);
  ASSERT_TRUE(foreign->report_clock_health(ClockHealthReport::healthy()));
  auto foreign_second = wait_for_catalog_tensor_schema_operation(**current);
  ASSERT_TRUE(foreign_second) << runtime_error_message(foreign_second.error());
  EXPECT_EQ(foreign_second->kind,
            channel_internal::CatalogReceivedOperationKind::kPeerIncarnationChanged);

  // Key setup: after latching the replacement, queue its health and unknown-generation data at
  // sequence three on distinct catalog lanes.
  ASSERT_TRUE(emit(*foreign, 0x13U, 3));
  ASSERT_TRUE(replacement->report_clock_health(ClockHealthReport::healthy()));
  auto latched_fence = wait_for_catalog_tensor_schema_operation(**current);
  ASSERT_TRUE(latched_fence) << runtime_error_message(latched_fence.error());
  EXPECT_EQ(latched_fence->kind,
            channel_internal::CatalogReceivedOperationKind::kPeerIncarnationChanged);
  auto accepted_health = wait_for_catalog_tensor_schema_operation(**current);
  ASSERT_TRUE(accepted_health) << runtime_error_message(accepted_health.error());
  EXPECT_EQ(accepted_health->kind, channel_internal::CatalogReceivedOperationKind::kClockHealth);
  EXPECT_EQ(accepted_health->clock_health.publisher_operation_sequence, 3U);
}

// Raw Tensor catalog subscribers enforce the same superseded-generation-before-ambiguity rule as
// descriptor-backed schemas.
// REQ: CHAN-069
// REQ: RUNTIME-008
TEST(RuntimeExternalChannelTest, RawTensorSupersededHealthDoesNotConflictWithReplacementData) {
  const std::string endpoint_name = unique_endpoint_name();

  Graph graph{"runtime-external-tensor-superseded-tie"};
  const ClockKey capture_clock = graph.add_clock<ManualClock>("capture-clock", std::int64_t{0});
  const ClockKey runtime_clock = graph.add_clock<RealtimeClock>("runtime-clock");
  graph.set_default_clock(runtime_clock);
  graph.add_clock_conversion(capture_clock, runtime_clock, AffineClockConversion{});
  TopicOptions options = optional_tensor_export_options(endpoint_name);
  options.bounds.subscriber.reset();
  options.bounds.publisher = PublisherEndpointBounds{.max_loaned_samples = 1U};
  options.capture_time = RequireCaptureTime{.clock = capture_clock};
  options.clock_health = ClockHealthOptions{
      .capability = ClockHealthCapability::kStatusReports,
      .requirement = ClockHealthRequirement::kFreshHealthyReport,
      .max_report_age = 5s,
  };
  const auto imported = graph.import_topic<Tensor>("holomq://" + endpoint_name, std::move(options));
  const auto sink = graph.op<ExternalTensorExportSink>("sink");
  graph.add_flow(imported.output(), sink->input, ConnectionOptions{.queue_depth = 1U});

  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto endpoint = plan.endpoint_catalog().endpoint(endpoint_name);
  ASSERT_TRUE(endpoint) << runtime_error_message(endpoint.error());
  auto graph_runtime = ChannelRuntime::create();
  auto external_runtime = ChannelRuntime::create();
  ASSERT_TRUE(graph_runtime) << runtime_error_message(graph_runtime.error());
  ASSERT_TRUE(external_runtime) << runtime_error_message(external_runtime.error());
  const auto create_subscriber = [&]() {
    return detail::RuntimeChannelBindingAccess::create_catalog_tensor_subscriber(
        *graph_runtime, *endpoint, 2U);
  };
  const auto publish = [&](ChannelPublisher<Tensor>& publisher,
                           const std::int64_t capture_timestamp_ns) -> expected<void, Error> {
    auto capture_time = publisher.make_capture_time(capture_timestamp_ns);
    if (!capture_time) {
      return make_unexpected(std::move(capture_time).error());
    }
    return publish_external_host_tensor(publisher, *capture_time);
  };

  auto created_first = create_subscriber();
  ASSERT_TRUE(created_first) << runtime_error_message(created_first.error());
  std::optional<ChannelSubscriber<Tensor>> first{std::move(*created_first)};
  auto created_retired =
      external_runtime->create_publisher<Tensor>(plan.endpoint_catalog(), endpoint_name);
  ASSERT_TRUE(created_retired) << runtime_error_message(created_retired.error());
  std::optional<ChannelPublisher<Tensor>> retired{std::move(*created_retired)};
  ASSERT_TRUE(wait_for_catalog_tensor_subscriber(*retired, *first));
  ASSERT_TRUE(publish(*retired, 1));
  auto retired_data = wait_for_catalog_tensor_operation(*first);
  ASSERT_TRUE(retired_data) << runtime_error_message(retired_data.error());
  ASSERT_EQ(retired_data->kind, channel_internal::CatalogReceivedOperationKind::kData);
  retired_data->tensor_sample.reset();
  const std::uint64_t retired_generation =
      detail::RuntimeChannelBindingAccess::latched_catalog_tensor_producer_generation(*first);
  ASSERT_NE(retired_generation, 0U);

  auto replacement =
      external_runtime->create_publisher<Tensor>(plan.endpoint_catalog(), endpoint_name);
  ASSERT_TRUE(replacement) << runtime_error_message(replacement.error());
  ASSERT_TRUE(wait_for_catalog_tensor_subscriber(*replacement, *first));
  ASSERT_TRUE(replacement->report_clock_health(ClockHealthReport::healthy()));
  auto replacement_warmup = wait_for_catalog_tensor_operation(*first);
  ASSERT_TRUE(replacement_warmup) << runtime_error_message(replacement_warmup.error());
  ASSERT_EQ(replacement_warmup->kind,
            channel_internal::CatalogReceivedOperationKind::kPeerIncarnationChanged);
  first.reset();
  ASSERT_TRUE(wait_for_no_subscribers(*retired));
  ASSERT_TRUE(wait_for_no_subscribers(*replacement));

  auto created_current = create_subscriber();
  ASSERT_TRUE(created_current) << runtime_error_message(created_current.error());
  std::optional<ChannelSubscriber<Tensor>> current{std::move(*created_current)};
  ASSERT_TRUE(detail::RuntimeChannelBindingAccess::supersede_catalog_tensor_producer_generation(
      *current, retired_generation));
  ASSERT_TRUE(wait_for_catalog_tensor_subscriber(*retired, *current));
  ASSERT_TRUE(wait_for_catalog_tensor_subscriber(*replacement, *current));

  // Key setup: queue the retired publisher's health operation and replacement publisher's data
  // operation at the same sequence number on independent catalog lanes.
  ASSERT_TRUE(retired->report_clock_health(ClockHealthReport::healthy()));
  ASSERT_TRUE(publish(*replacement, 2));

  auto pending = detail::RuntimeChannelBindingAccess::catalog_tensor_import_quiescence(*current);
  ASSERT_TRUE(pending) << runtime_error_message(pending.error());
  EXPECT_TRUE(pending->pending_current);
  EXPECT_TRUE(pending->pending_foreign);

  // Key assertion: the raw subscriber fences the retired lane first, then accepts sequence two
  // from the replacement generation without reporting an ambiguous-incarnation protocol error.
  auto fenced = wait_for_catalog_tensor_operation(*current);
  ASSERT_TRUE(fenced) << runtime_error_message(fenced.error());
  ASSERT_EQ(fenced->kind, channel_internal::CatalogReceivedOperationKind::kPeerIncarnationChanged);
  auto accepted = wait_for_catalog_tensor_operation(*current);
  ASSERT_TRUE(accepted) << runtime_error_message(accepted.error());
  ASSERT_EQ(accepted->kind, channel_internal::CatalogReceivedOperationKind::kData);
  EXPECT_EQ(accepted->metadata.sequence, 2U);
  accepted->tensor_sample.reset();
  EXPECT_NE(
      detail::RuntimeChannelBindingAccess::latched_catalog_tensor_producer_generation(*current),
      retired_generation);

  // The endpoint admits two publishers; the retired lane is fully fenced, so release that
  // publisher before reusing its slot for an unknown third incarnation.
  retired.reset();
  auto foreign = external_runtime->create_publisher<Tensor>(plan.endpoint_catalog(), endpoint_name);
  ASSERT_TRUE(foreign) << runtime_error_message(foreign.error());
  ASSERT_TRUE(wait_for_catalog_tensor_subscriber(*foreign, *current));
  ASSERT_TRUE(publish(*foreign, 1));
  auto foreign_first = wait_for_catalog_tensor_operation(*current);
  ASSERT_TRUE(foreign_first) << runtime_error_message(foreign_first.error());
  EXPECT_EQ(foreign_first->kind,
            channel_internal::CatalogReceivedOperationKind::kPeerIncarnationChanged);
  ASSERT_TRUE(foreign->report_clock_health(ClockHealthReport::healthy()));
  auto foreign_second = wait_for_catalog_tensor_operation(*current);
  ASSERT_TRUE(foreign_second) << runtime_error_message(foreign_second.error());
  EXPECT_EQ(foreign_second->kind,
            channel_internal::CatalogReceivedOperationKind::kPeerIncarnationChanged);

  // Key setup: after latching the replacement, queue unknown-generation data and current health
  // at sequence three so equal per-generation counters meet on distinct lanes.
  ASSERT_TRUE(publish(*foreign, 3));
  ASSERT_TRUE(replacement->report_clock_health(ClockHealthReport::healthy()));
  auto latched_fence = wait_for_catalog_tensor_operation(*current);
  ASSERT_TRUE(latched_fence) << runtime_error_message(latched_fence.error());
  EXPECT_EQ(latched_fence->kind,
            channel_internal::CatalogReceivedOperationKind::kPeerIncarnationChanged);
  auto accepted_health = wait_for_catalog_tensor_operation(*current);
  ASSERT_TRUE(accepted_health) << runtime_error_message(accepted_health.error());
  EXPECT_EQ(accepted_health->kind, channel_internal::CatalogReceivedOperationKind::kClockHealth);
  EXPECT_EQ(accepted_health->clock_health.publisher_operation_sequence, 3U);
}

// Closing one descriptor-schema producer incarnation establishes the generation fence needed for
// a lifecycle restart. The replacement attachment rejects the retired generation and accepts a
// fresh publisher without rebuilding the immutable ExecutionPlan.
// REQ: EXTERNAL-010
// REQ: RUNTIME-008
// REQ: SYSTEM-024
TEST(RuntimeExternalChannelTest, DescriptorTensorSchemaImportRebindsAcrossLifecycleRestart) {
  ExternalImageImportProbe::reset();
  const std::string endpoint_name = unique_endpoint_name();

  Graph graph{"runtime-external-image-restart"};
  const ClockKey capture_clock = graph.add_clock<ManualClock>("capture-clock", std::int64_t{0});
  graph.set_default_clock(capture_clock);
  TopicOptions options = external_image_import_options(endpoint_name);
  options.capture_time = RequireCaptureTime{.clock = capture_clock};
  const auto imported = graph.import_topic<schema::ImageT>(
      "holomq://" + endpoint_name, std::move(options), external_image_contract());
  const auto sink = graph.op<ExternalImageImportSink>("sink");
  graph.add_flow(imported.output(), sink->input, ConnectionOptions{.queue_depth = 1U});
  graph.lifecycle().allow_external_restart(sink);

  CompileOptions compile_options;
  compile_options.lifecycle_transactions = LifecycleTransactionOptions{
      .max_pending_requests = 4U,
      .max_pending_transactions = 4U,
      .max_receipts = 8U,
      .max_fenced_operators = 4U,
      .max_candidate_generations = 4U,
      .max_events = 64U,
      .max_audit_records = 8U,
      .max_quiesce_records = 16U,
      .max_rollback_records = 16U,
  };
  const ExecutionPlan compiled = compile(graph, compile_options);
  ASSERT_TRUE(compiled.ok()) << compiled.json();
  ASSERT_EQ(compiled.lifecycle_restart_targets().size(), 1U);
  ASSERT_EQ(compiled.imported_generation_capacities().size(), 1U);
  EXPECT_TRUE(compiled.imported_generation_capacities().front().restart_capable_minimum_two);
  auto phase_counters = std::make_shared<detail::RuntimePhaseCounters>();
  const ExecutionPlan plan =
      detail::RuntimeTestPlanAccess::with_phase_counters(compiled, phase_counters);

  auto external_runtime = ChannelRuntime::create();
  ASSERT_TRUE(external_runtime) << runtime_error_message(external_runtime.error());
  RunSession session = run_async(plan);
  const auto phase_count = [&](const detail::RuntimePhaseCounter phase) {
    return phase_counters->snapshot()[detail::RuntimePhaseCounters::index(phase)];
  };
  const std::uint64_t readiness_baseline =
      phase_count(detail::RuntimePhaseCounter::kChannelReadinessPublished);
  const auto publish = [&](ChannelTensorSchemaPublisher<ExternalImagePayload>& publisher,
                           const std::uint8_t value,
                           const std::int64_t capture_timestamp_ns) -> expected<void, Error> {
    auto tensor = make_external_image_tensor(value);
    if (!tensor) {
      return make_unexpected(std::move(tensor).error());
    }
    auto metadata = schema::build_image_metadata(
        *tensor,
        schema::ImageMetadataOptions{
            .height = 1U, .width = 8U, .encoding = schema::ImageEncoding_MONO8});
    if (!metadata) {
      return make_unexpected(std::move(metadata).error());
    }
    auto emitted = publisher.emit_captured(*metadata, std::move(*tensor), capture_timestamp_ns);
    if (!emitted) {
      return make_unexpected(std::move(emitted).error());
    }
    return expected<void, Error>{};
  };

  // Key setup: close the first descriptor-schema publisher, publish its ordered EOS, restart the
  // owning sink, and attach a replacement publisher whose operation sequence restarts at one.
  {
    auto first = external_runtime->create_tensor_schema_publisher<ExternalImagePayload>(
        plan.endpoint_catalog(), endpoint_name);
    ASSERT_TRUE(first) << runtime_error_message(first.error());
    ASSERT_TRUE(wait_for_schema_subscriber(*first, 5s, 20ms));
    const auto first_publish = publish(*first, 0x21U, 1);
    ASSERT_TRUE(first_publish) << runtime_error_message(first_publish.error());
    ASSERT_TRUE(ExternalImageImportProbe::wait_for_count(1U));
    ASSERT_TRUE(first->close());
    const auto eos_deadline = std::chrono::steady_clock::now() + 5s;
    while (phase_count(detail::RuntimePhaseCounter::kChannelReadinessPublished) <
               readiness_baseline + 2U &&
           session.is_running() && std::chrono::steady_clock::now() < eos_deadline) {
      std::this_thread::yield();
    }
    ASSERT_EQ(phase_count(detail::RuntimePhaseCounter::kChannelReadinessPublished),
              readiness_baseline + 2U);
  }

  // Prove the fully drained import remains in its bounded lifecycle-restart wait state instead of
  // racing natural completion before the host can submit the replacement transaction.
  const std::uint64_t idle_baseline = phase_count(detail::RuntimePhaseCounter::kNoProgressScans);
  const auto idle_deadline = std::chrono::steady_clock::now() + 5s;
  while (phase_count(detail::RuntimePhaseCounter::kNoProgressScans) < idle_baseline + 2U &&
         session.is_running() && std::chrono::steady_clock::now() < idle_deadline) {
    std::this_thread::yield();
  }
  ASSERT_GE(phase_count(detail::RuntimePhaseCounter::kNoProgressScans), idle_baseline + 2U);
  ASSERT_TRUE(session.is_running());

  const auto run_target = session.operator_ref(compiled.lifecycle_restart_targets().front().target);
  ASSERT_TRUE(run_target) << runtime_error_message(run_target.error());
  const std::array targets{*run_target};
  constexpr std::array stages{LifecycleStage::kConfigure};
  auto control =
      session.lifecycle_control({.allowed_targets = targets, .allowed_from_stages = stages});
  ASSERT_TRUE(control) << runtime_error_message(control.error());
  auto receipt = control->request_restart(*run_target);
  ASSERT_TRUE(receipt) << runtime_error_message(receipt.error());
  auto completed = control->await(*receipt, 5s);
  ASSERT_TRUE(completed) << runtime_error_message(completed.error());
  ASSERT_EQ(completed->state, LifecycleTransactionState::kPublished);

  auto second = external_runtime->create_tensor_schema_publisher<ExternalImagePayload>(
      plan.endpoint_catalog(), endpoint_name);
  ASSERT_TRUE(second) << runtime_error_message(second.error());
  // A replacement HoloMQ subscriber becomes visible before every publisher-side data segment is
  // necessarily mapped. Require a short stable-presence window so the first measured payload is
  // the post-restart sample rather than a transport-connectivity probe.
  ASSERT_TRUE(wait_for_schema_subscriber(*second, 5s, 20ms));
  const auto second_publish = publish(*second, 0x42U, 2);
  ASSERT_TRUE(second_publish) << runtime_error_message(second_publish.error());
  ASSERT_TRUE(ExternalImageImportProbe::wait_for_count(2U));
  ASSERT_TRUE(second->close());

  session.wait();
  // Key assertion: the replacement generation delivers its Tensor through the unchanged plan and
  // supersedes the first generation's observed byte value. Exhausting the two-generation budget
  // restores ordinary natural completion after the replacement EOS.
  EXPECT_EQ(session.termination(), RunTermination::kCompleted);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
  EXPECT_EQ(ExternalImageImportProbe::count.load(std::memory_order_acquire), 2U);
  EXPECT_EQ(ExternalImageImportProbe::first_byte.load(std::memory_order_relaxed), 0x42U);
}

// An imported generation's EOS must retire only the input side of a restart-capable forwarding
// operator.  Its output remains live until the replacement generation has published its payload;
// the final EOS then closes the forwarding edge and permits ordinary completion.
// REQ: EXTERNAL-010
// REQ: RUNTIME-008
// REQ: SYSTEM-024
TEST(RuntimeExternalChannelTest,
     DescriptorTensorSchemaImportForwarderKeepsOutputOpenAcrossRestart) {
  ExternalImageImportProbe::reset();
  const std::string endpoint_name = unique_endpoint_name();

  Graph graph{"runtime-external-image-forwarder-restart"};
  const ClockKey capture_clock = graph.add_clock<ManualClock>("capture-clock", std::int64_t{0});
  graph.set_default_clock(capture_clock);
  TopicOptions options = external_image_import_options(endpoint_name);
  options.capture_time = RequireCaptureTime{.clock = capture_clock};
  const auto imported = graph.import_topic<schema::ImageT>(
      "holomq://" + endpoint_name, std::move(options), external_image_contract());
  const auto forwarder = graph.op<ExternalImageForwarder>("forwarder");
  const auto sink = graph.op<ExternalImageImportSink>("sink");
  graph.add_flow(imported.output(), forwarder->input, ConnectionOptions{.queue_depth = 1U});
  graph.add_flow(forwarder->output, sink->input, ConnectionOptions{.queue_depth = 1U});
  graph.lifecycle().allow_external_restart(forwarder);

  CompileOptions compile_options;
  compile_options.lifecycle_transactions = LifecycleTransactionOptions{
      .max_pending_requests = 4U,
      .max_pending_transactions = 4U,
      .max_receipts = 8U,
      .max_fenced_operators = 4U,
      .max_candidate_generations = 4U,
      .max_events = 64U,
      .max_audit_records = 8U,
      .max_quiesce_records = 16U,
      .max_rollback_records = 16U,
  };
  const ExecutionPlan compiled = compile(graph, compile_options);
  ASSERT_TRUE(compiled.ok()) << compiled.json();
  const auto restart_target =
      std::find_if(compiled.lifecycle_restart_targets().begin(),
                   compiled.lifecycle_restart_targets().end(),
                   [](const auto& target) { return target.external_control_allowed; });
  ASSERT_NE(restart_target, compiled.lifecycle_restart_targets().end()) << compiled.json();
  ASSERT_EQ(std::count_if(compiled.lifecycle_restart_targets().begin(),
                          compiled.lifecycle_restart_targets().end(),
                          [](const auto& target) { return target.external_control_allowed; }),
            1U)
      << compiled.json();
  ASSERT_EQ(compiled.imported_generation_capacities().size(), 1U);
  EXPECT_TRUE(compiled.imported_generation_capacities().front().restart_capable_minimum_two);
  auto phase_counters = std::make_shared<detail::RuntimePhaseCounters>();
  const ExecutionPlan plan =
      detail::RuntimeTestPlanAccess::with_phase_counters(compiled, phase_counters);

  auto external_runtime = ChannelRuntime::create();
  ASSERT_TRUE(external_runtime) << runtime_error_message(external_runtime.error());
  RunSession session = run_async(plan);
  const auto phase_count = [&](const detail::RuntimePhaseCounter phase) {
    return phase_counters->snapshot()[detail::RuntimePhaseCounters::index(phase)];
  };
  const std::uint64_t readiness_baseline =
      phase_count(detail::RuntimePhaseCounter::kChannelReadinessPublished);
  const auto publish = [&](ChannelTensorSchemaPublisher<ExternalImagePayload>& publisher,
                           const std::uint8_t value,
                           const std::int64_t capture_timestamp_ns) -> expected<void, Error> {
    auto tensor = make_external_image_tensor(value);
    if (!tensor) {
      return make_unexpected(std::move(tensor).error());
    }
    auto metadata = schema::build_image_metadata(
        *tensor,
        schema::ImageMetadataOptions{
            .height = 1U, .width = 8U, .encoding = schema::ImageEncoding_MONO8});
    if (!metadata) {
      return make_unexpected(std::move(metadata).error());
    }
    auto emitted = publisher.emit_captured(*metadata, std::move(*tensor), capture_timestamp_ns);
    if (!emitted) {
      return make_unexpected(std::move(emitted).error());
    }
    return expected<void, Error>{};
  };

  {
    auto first = external_runtime->create_tensor_schema_publisher<ExternalImagePayload>(
        plan.endpoint_catalog(), endpoint_name);
    ASSERT_TRUE(first) << runtime_error_message(first.error());
    ASSERT_TRUE(wait_for_schema_subscriber(*first, 5s, 20ms));
    ASSERT_TRUE(publish(*first, 0x21U, 1));
    ASSERT_TRUE(ExternalImageImportProbe::wait_for_count(1U));
    ASSERT_TRUE(first->close());
    const auto eos_deadline = std::chrono::steady_clock::now() + 5s;
    while (phase_count(detail::RuntimePhaseCounter::kChannelReadinessPublished) <
               readiness_baseline + 2U &&
           session.is_running() && std::chrono::steady_clock::now() < eos_deadline) {
      std::this_thread::yield();
    }
    ASSERT_EQ(phase_count(detail::RuntimePhaseCounter::kChannelReadinessPublished),
              readiness_baseline + 2U);
  }

  // Wait until the first generation's terminal drain and the downstream callback have both
  // crossed the controller cut. This proves the run remains live specifically because a bounded
  // import replacement is admissible, and prevents the restart preflight from racing a retained
  // downstream Tensor claim.
  const std::uint64_t idle_baseline = phase_count(detail::RuntimePhaseCounter::kNoProgressScans);
  const auto idle_deadline = std::chrono::steady_clock::now() + 5s;
  while (phase_count(detail::RuntimePhaseCounter::kNoProgressScans) < idle_baseline + 2U &&
         session.is_running() && std::chrono::steady_clock::now() < idle_deadline) {
    std::this_thread::yield();
  }
  ASSERT_GE(phase_count(detail::RuntimePhaseCounter::kNoProgressScans), idle_baseline + 2U);
  ASSERT_TRUE(session.is_running());

  const auto run_target = session.operator_ref(restart_target->target);
  ASSERT_TRUE(run_target) << runtime_error_message(run_target.error());
  const std::array targets{*run_target};
  constexpr std::array stages{LifecycleStage::kConfigure};
  auto control =
      session.lifecycle_control({.allowed_targets = targets, .allowed_from_stages = stages});
  ASSERT_TRUE(control) << runtime_error_message(control.error());
  auto receipt = control->request_restart(*run_target);
  ASSERT_TRUE(receipt) << runtime_error_message(receipt.error());
  auto completed = control->await(*receipt, 5s);
  ASSERT_TRUE(completed) << runtime_error_message(completed.error());
  ASSERT_EQ(completed->state, LifecycleTransactionState::kPublished);

  auto second = external_runtime->create_tensor_schema_publisher<ExternalImagePayload>(
      plan.endpoint_catalog(), endpoint_name);
  ASSERT_TRUE(second) << runtime_error_message(second.error());
  ASSERT_TRUE(wait_for_schema_subscriber(*second, 5s, 20ms));
  ASSERT_TRUE(publish(*second, 0x42U, 2));
  ASSERT_TRUE(ExternalImageImportProbe::wait_for_count(2U));
  ASSERT_TRUE(second->close());

  session.wait();
  EXPECT_EQ(session.termination(), RunTermination::kCompleted);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
  EXPECT_EQ(ExternalImageImportProbe::count.load(std::memory_order_acquire), 2U);
  EXPECT_EQ(ExternalImageImportProbe::first_byte.load(std::memory_order_relaxed), 0x42U);
}

// Raw Tensor imports use the same catalog generation fence as descriptor-backed schemas. The
// Runtime must replace the typed endpoint after restart and accept the replacement producer's
// reset publication sequence without rebuilding the immutable ExecutionPlan.
// REQ: EXTERNAL-010
// REQ: RUNTIME-008
// REQ: SYSTEM-024
TEST(RuntimeExternalChannelTest, RawTensorImportRebindsAcrossLifecycleRestart) {
  ExternalTensorExportSink::reset();
  const std::string endpoint_name = unique_endpoint_name();

  Graph graph{"runtime-external-tensor-restart"};
  const ClockKey capture_clock = graph.add_clock<ManualClock>("capture-clock", std::int64_t{0});
  graph.set_default_clock(capture_clock);
  TopicOptions options = optional_tensor_export_options(endpoint_name);
  options.bounds.subscriber.reset();
  options.bounds.publisher = PublisherEndpointBounds{.max_loaned_samples = 1U};
  options.capture_time = RequireCaptureTime{.clock = capture_clock};
  const auto imported = graph.import_topic<Tensor>("holomq://" + endpoint_name, std::move(options));
  const auto sink = graph.op<ExternalTensorExportSink>("sink");
  graph.add_flow(imported.output(), sink->input, ConnectionOptions{.queue_depth = 1U});
  graph.lifecycle().allow_external_restart(sink);

  CompileOptions compile_options;
  compile_options.lifecycle_transactions = LifecycleTransactionOptions{
      .max_pending_requests = 4U,
      .max_pending_transactions = 4U,
      .max_receipts = 8U,
      .max_fenced_operators = 4U,
      .max_candidate_generations = 4U,
      .max_events = 64U,
      .max_audit_records = 8U,
      .max_quiesce_records = 16U,
      .max_rollback_records = 16U,
  };
  const ExecutionPlan compiled = compile(graph, compile_options);
  ASSERT_TRUE(compiled.ok()) << compiled.json();
  ASSERT_EQ(compiled.lifecycle_restart_targets().size(), 1U);
  ASSERT_EQ(compiled.imported_generation_capacities().size(), 1U);
  EXPECT_TRUE(compiled.imported_generation_capacities().front().restart_capable_minimum_two);
  auto phase_counters = std::make_shared<detail::RuntimePhaseCounters>();
  const ExecutionPlan plan =
      detail::RuntimeTestPlanAccess::with_phase_counters(compiled, phase_counters);

  auto external_runtime = ChannelRuntime::create();
  ASSERT_TRUE(external_runtime) << runtime_error_message(external_runtime.error());
  RunSession session = run_async(plan);
  const auto phase_count = [&](const detail::RuntimePhaseCounter phase) {
    return phase_counters->snapshot()[detail::RuntimePhaseCounters::index(phase)];
  };
  const std::uint64_t readiness_baseline =
      phase_count(detail::RuntimePhaseCounter::kChannelReadinessPublished);

  // Key setup: close the first raw-Tensor publisher, wait for ordered EOS readiness, restart the
  // sink, then attach a replacement publisher with a reset sequence.
  {
    auto first = external_runtime->create_publisher<Tensor>(plan.endpoint_catalog(), endpoint_name);
    ASSERT_TRUE(first) << runtime_error_message(first.error());
    ASSERT_TRUE(wait_for_subscriber(*first, 5s, 20ms));
    auto first_capture = first->make_capture_time(1);
    ASSERT_TRUE(first_capture) << runtime_error_message(first_capture.error());
    const auto first_publish = publish_external_host_tensor(*first, *first_capture);
    ASSERT_TRUE(first_publish) << runtime_error_message(first_publish.error());
    ASSERT_TRUE(ExternalTensorExportSink::wait_for_count(1U));
    ASSERT_TRUE(first->close());
    const auto eos_deadline = std::chrono::steady_clock::now() + 5s;
    while (phase_count(detail::RuntimePhaseCounter::kChannelReadinessPublished) <
               readiness_baseline + 2U &&
           session.is_running() && std::chrono::steady_clock::now() < eos_deadline) {
      std::this_thread::yield();
    }
    ASSERT_EQ(phase_count(detail::RuntimePhaseCounter::kChannelReadinessPublished),
              readiness_baseline + 2U);
  }

  // Prove the fully drained import remains in its bounded lifecycle-restart wait state instead of
  // racing natural completion before the host can submit the replacement transaction.
  const std::uint64_t idle_baseline = phase_count(detail::RuntimePhaseCounter::kNoProgressScans);
  const auto idle_deadline = std::chrono::steady_clock::now() + 5s;
  while (phase_count(detail::RuntimePhaseCounter::kNoProgressScans) < idle_baseline + 2U &&
         session.is_running() && std::chrono::steady_clock::now() < idle_deadline) {
    std::this_thread::yield();
  }
  ASSERT_GE(phase_count(detail::RuntimePhaseCounter::kNoProgressScans), idle_baseline + 2U);
  ASSERT_TRUE(session.is_running());

  const auto run_target = session.operator_ref(compiled.lifecycle_restart_targets().front().target);
  ASSERT_TRUE(run_target) << runtime_error_message(run_target.error());
  const std::array targets{*run_target};
  constexpr std::array stages{LifecycleStage::kConfigure};
  auto control =
      session.lifecycle_control({.allowed_targets = targets, .allowed_from_stages = stages});
  ASSERT_TRUE(control) << runtime_error_message(control.error());
  auto receipt = control->request_restart(*run_target);
  ASSERT_TRUE(receipt) << runtime_error_message(receipt.error());
  auto completed = control->await(*receipt, 5s);
  ASSERT_TRUE(completed) << runtime_error_message(completed.error());
  ASSERT_EQ(completed->state, LifecycleTransactionState::kPublished);

  auto second = external_runtime->create_publisher<Tensor>(plan.endpoint_catalog(), endpoint_name);
  ASSERT_TRUE(second) << runtime_error_message(second.error());
  ASSERT_TRUE(wait_for_subscriber(*second, 5s, 20ms));
  auto second_capture = second->make_capture_time(2);
  ASSERT_TRUE(second_capture) << runtime_error_message(second_capture.error());
  const auto second_publish = publish_external_host_tensor(*second, *second_capture);
  ASSERT_TRUE(second_publish) << runtime_error_message(second_publish.error());
  ASSERT_TRUE(ExternalTensorExportSink::wait_for_count(2U));
  ASSERT_TRUE(second->close());

  session.wait();
  // Key assertion: both producer generations deliver exactly one Tensor through one immutable
  // execution plan. The replacement EOS exhausts the admitted two-generation budget, so the run
  // returns to ordinary natural completion without diagnostics.
  EXPECT_EQ(session.termination(), RunTermination::kCompleted);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
  EXPECT_EQ(ExternalTensorExportSink::received.load(std::memory_order_acquire), 2U);
}

// A raw Tensor subscriber that accepted EOS is generation-fenced until Runtime explicitly replaces
// it. Traffic from a newly created publisher must not keep the retired pump endpoint ready.
// REQ: RUNTIME-008
// REQ: SYSTEM-024
TEST(RuntimeExternalChannelTest, ClosedRawTensorCatalogSubscriberIgnoresReplacementTraffic) {
  const std::string endpoint_name = unique_endpoint_name();

  Graph graph{"runtime-external-closed-tensor-readiness"};
  const ClockKey runtime_clock = graph.add_clock<RealtimeClock>("runtime-clock");
  graph.set_default_clock(runtime_clock);
  TopicOptions options = optional_tensor_export_options(endpoint_name);
  options.bounds.subscriber.reset();
  options.bounds.publisher = PublisherEndpointBounds{.max_loaned_samples = 1U};
  const auto imported = graph.import_topic<Tensor>("holomq://" + endpoint_name, std::move(options));
  const auto sink = graph.op<ExternalTensorExportSink>("sink");
  graph.add_flow(imported.output(), sink->input, ConnectionOptions{.queue_depth = 1U});

  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << (plan.diagnostics().empty() ? "<no diagnostic>"
                                                        : plan.diagnostics().front().token);
  const auto endpoint = plan.endpoint_catalog().endpoint(endpoint_name);
  ASSERT_TRUE(endpoint) << runtime_error_message(endpoint.error());
  auto graph_runtime = ChannelRuntime::create();
  auto external_runtime = ChannelRuntime::create();
  ASSERT_TRUE(graph_runtime) << runtime_error_message(graph_runtime.error());
  ASSERT_TRUE(external_runtime) << runtime_error_message(external_runtime.error());
  auto subscriber = detail::RuntimeChannelBindingAccess::create_catalog_tensor_subscriber(
      *graph_runtime, *endpoint, 2U);
  ASSERT_TRUE(subscriber) << runtime_error_message(subscriber.error());

  // Key setup: attach a replacement publisher before accepting the first publisher's ordered EOS,
  // preserving a live transport mapping that can receive traffic after the generation fence.
  auto first = external_runtime->create_publisher<Tensor>(plan.endpoint_catalog(), endpoint_name);
  ASSERT_TRUE(first) << runtime_error_message(first.error());
  ASSERT_TRUE(wait_for_catalog_tensor_subscriber(*first, *subscriber));
  auto replacement =
      external_runtime->create_publisher<Tensor>(plan.endpoint_catalog(), endpoint_name);
  ASSERT_TRUE(replacement) << runtime_error_message(replacement.error());
  ASSERT_TRUE(wait_for_catalog_tensor_subscriber(*replacement, *subscriber));
  ASSERT_TRUE(first->close());
  auto terminal = wait_for_catalog_tensor_operation(*subscriber);
  ASSERT_TRUE(terminal) << runtime_error_message(terminal.error());
  ASSERT_EQ(terminal->kind, channel_internal::CatalogReceivedOperationKind::kEndOfStream);

  auto loan = replacement->loan_tensor(
      TensorLoanRequest{.shape = kExternalTensorShape, .dtype = kExternalTensorDtype});
  ASSERT_TRUE(loan) << runtime_error_message(loan.error());
  auto writer = loan->write_host();
  ASSERT_TRUE(writer) << runtime_error_message(writer.error());
  std::fill_n(static_cast<std::byte*>(writer->data()), writer->byte_size(), std::byte{0x5a});
  ASSERT_TRUE(std::move(*writer).commit());
  auto emitted = replacement->emit(std::move(*loan));
  ASSERT_TRUE(emitted) << runtime_error_message(emitted.error());
  ASSERT_EQ(*emitted, 1U);

  // Key assertion: EOS keeps the retired catalog subscriber non-ready even when replacement data
  // has reached its still-live HoloMQ recipient; explicit rebind owns the generation cutover.
  auto pending =
      detail::RuntimeChannelBindingAccess::has_pending_catalog_tensor_operation(*subscriber);
  ASSERT_TRUE(pending) << runtime_error_message(pending.error());
  EXPECT_FALSE(*pending);
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
  // Retain the data activation plus ordered data/EOS ingress and controller wake/park evidence.
  // Callback and timing-dependent partition-controller records require a conservative fixed
  // test-only bound rather than assuming one controller cycle per ingress.
  std::array<detail::RuntimePhaseTraceRecord, 256U> phase_trace{};
  auto phase_counters = std::make_shared<detail::RuntimePhaseCounters>(phase_trace);
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
  ASSERT_EQ(plan.deadlines().size(), 1U);
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
  EXPECT_EQ(session.diagnostics()[0].token, "DEADLINE_MISS");
  EXPECT_EQ(session.diagnostics()[1].token, "RUNTIME_DEADLINE_FAIL_RUN");
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

// REQ: CHAN-069
// REQ: RUNTIME-016
// REQ: RUNTIME-017
TEST(RuntimeExternalChannelTest,
     CanonicalScalarImportAppliesEveryPrecedingHealthOperationBeforeCompute) {
  ExternalImportProbe::reset();
  const std::string endpoint_name = unique_endpoint_name();

  //   external_publisher --(holomq://endpoint, clock-health=kFreshHealthyReport)-->
  //       imported_topic --[depth=8]--> sink
  //   Health sequence (publisher total order): H1, S1, S2, H2(unhealthy), S3
  Graph graph{"runtime-external-import"};
  const ClockKey capture_clock = graph.add_clock<ManualClock>("capture-clock", std::int64_t{0});
  const ClockKey age_clock = graph.add_clock<RealtimeClock>("health-age-clock");
  graph.set_default_clock(age_clock);
  graph.add_clock_conversion(capture_clock, age_clock, AffineClockConversion{});

  TopicOptions options;
  options.endpoint_name = endpoint_name;
  options.capture_time = RequireCaptureTime{.clock = capture_clock};
  options.clock_health = ClockHealthOptions{
      .capability = ClockHealthCapability::kStatusReports,
      .requirement = ClockHealthRequirement::kFreshHealthyReport,
      .max_report_age = 5s,
  };
  const auto imported =
      graph.import_topic<std::uint32_t>("holomq://" + endpoint_name, std::move(options));
  const auto sink = graph.op<ExternalImportSink>("sink");
  graph.add_flow(imported.output(), sink->input, ConnectionOptions{.queue_depth = 8U});

  const ExecutionPlan plan = compile(graph);
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
    FAIL() << "external subscriber did not connect: " << runtime_error_message(connected.error())
           << "; termination=" << static_cast<int>(session.termination()) << ", diagnostics="
           << (session.diagnostics().empty() ? "<none>" : session.diagnostics().front().token)
           << ", plan=" << plan.json();
  }

  // Key setup: RequireCaptureTime + ClockHealthOptions{kFreshHealthyReport}; health ops
  // precede each sample in the publisher's total order and filter via Standard-profile policy.
  // H1 has no readiness authority. S1 and S2 carry the healthy state that precedes them in the
  // publisher's total operation order. H2 is then applied before S3, so Standard-profile skip
  // policy suppresses S3 without losing either earlier sample or fabricating a health callback.
  ASSERT_TRUE(publisher->report_clock_health(ClockHealthReport::healthy()));  // H1
  std::this_thread::sleep_for(20ms);
  ASSERT_TRUE(session.is_running())
      << "health-only operation failed the run; diagnostics="
      << (session.diagnostics().empty() ? "<none>" : session.diagnostics().front().token);
  EXPECT_EQ(ExternalImportProbe::count.load(std::memory_order_acquire), 0U);

  constexpr std::uint32_t kFirst = 0x01020304U;
  ASSERT_TRUE(publisher->emit_captured(kFirst, 101));  // S1
  std::this_thread::sleep_for(100ms);
  ASSERT_TRUE(session.is_running())
      << "first data operation failed the run; diagnostics="
      << (session.diagnostics().empty() ? "<none>" : session.diagnostics().front().token);
  if (!ExternalImportProbe::wait_for_count(1U)) {
    session.request_stop();
    session.wait();
    FAIL() << "first healthy sample did not reach compute; termination="
           << static_cast<int>(session.termination()) << ", diagnostics="
           << (session.diagnostics().empty() ? "<none>" : session.diagnostics().front().token)
           << ", vertex="
           << (session.diagnostics().empty() ? 0U
                                             : session.diagnostics().front().source.vertex.value);
  }
  ASSERT_TRUE(publisher->emit_captured(0x11223344U, 102));  // S2
  ASSERT_TRUE(ExternalImportProbe::wait_for_count(2U));

  ASSERT_TRUE(publisher->report_clock_health(
      ClockHealthReport::unhealthy(ClockHealthReason::kSynchronizationLockLost)));  // H2
  std::this_thread::sleep_for(20ms);
  EXPECT_EQ(ExternalImportProbe::count.load(std::memory_order_acquire), 2U);
  ASSERT_TRUE(publisher->emit_captured(0xaabbccddU, 103));  // S3
  std::this_thread::sleep_for(100ms);

  session.request_stop();
  session.wait();
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
  // Key assertion: exactly 2 samples arrive; S3 is suppressed by H2 (unhealthy) without error.
  EXPECT_EQ(ExternalImportProbe::count.load(std::memory_order_acquire), 2U);
  EXPECT_EQ(ExternalImportProbe::values[0], kFirst);
  EXPECT_EQ(ExternalImportProbe::values[1], 0x11223344U);
  EXPECT_EQ(ExternalImportProbe::sequences[0], 2U);
  EXPECT_EQ(ExternalImportProbe::sequences[1], 3U);
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

}  // namespace
}  // namespace holoscan
