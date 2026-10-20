/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

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
#include <thread>
#include <utility>
#include <vector>

#include "compiler/compile_internal.hpp"
#include "partition_parity_test_support.hpp"
#include "runtime/runtime_test_access.hpp"
#include "runtime_test_graph.hpp"

namespace holoscan {
namespace {

using namespace std::chrono_literals;

// Shared static-storage aggregator for channel lifecycle restart tests. Tracks operator
// start/compute counts, sink values, and tag sequences; provides CV-guarded waits so
// TEST bodies can block until the expected number of sink observations arrive.
// Representative test: PublishedRestartDiscardsOldIngressAndRearmsAfterStart.
struct ChannelRestartProbeState {
  static void reset(bool fail_forward_start, bool require_feeder_prefix = false) {
    const std::lock_guard lock{mutex};
    fail_restart_start.store(fail_forward_start, std::memory_order_relaxed);
    feeder_prefix_required.store(require_feeder_prefix, std::memory_order_relaxed);
    source_compute_count.store(0U, std::memory_order_relaxed);
    source_forward_start_count.store(0U, std::memory_order_relaxed);
    source_restore_start_count.store(0U, std::memory_order_relaxed);
    sink_start_count.store(0U, std::memory_order_relaxed);
    member_start_count.store(0U, std::memory_order_relaxed);
    member_forward_start_count.store(0U, std::memory_order_relaxed);
    feeder_compute_count.store(0U, std::memory_order_relaxed);
    feeder_prefix_published.store(false, std::memory_order_relaxed);
    feeder_post_data_published.store(false, std::memory_order_relaxed);
    feeder_post_published.store(false, std::memory_order_relaxed);
    unrelated_start_count.store(0U, std::memory_order_relaxed);
    unrelated_forward_start_count.store(0U, std::memory_order_relaxed);
    restart_accepted.store(false, std::memory_order_relaxed);
    restart_error.store(ErrorCode::kSuccess, std::memory_order_relaxed);
    sink_values.clear();
    sink_sequences.clear();
    source_tags.clear();
    sink_tags.clear();
    sink_activation_reasons.clear();
    sink_activation_policies.clear();
    member_values.clear();
  }

  static void record_source_tag(Tag tag) {
    const std::lock_guard lock{mutex};
    source_tags.push_back(tag);
  }

  static void record_sink(const Sample<std::uint32_t>& sample, Tag tag,
                          ActivationStatus activation) {
    const std::lock_guard lock{mutex};
    sink_values.push_back(sample.data);
    sink_sequences.push_back(sample.metadata.sequence);
    sink_tags.push_back(tag);
    sink_activation_reasons.push_back(activation.reason);
    sink_activation_policies.push_back(activation.policy);
    cv.notify_all();
  }

  [[nodiscard]] static bool wait_for_sink(std::size_t count) {
    std::unique_lock lock{mutex};
    return cv.wait_for(lock, 5s, [count] { return sink_values.size() >= count; });
  }

  static void record_member_batch(const SelectedBatchView<std::uint32_t>& selected) {
    const std::lock_guard lock{mutex};
    member_values.clear();
    member_values.reserve(selected.size());
    for (const Sample<std::uint32_t>& sample : selected) {
      member_values.push_back(sample.data);
    }
    cv.notify_all();
  }

  [[nodiscard]] static bool wait_for_member_batch() {
    std::unique_lock lock{mutex};
    return cv.wait_for(lock, 5s, [] { return !member_values.empty(); });
  }

  static void publish_feeder_prefix() noexcept {
    feeder_prefix_published.store(true, std::memory_order_release);
    cv.notify_all();
  }

  [[nodiscard]] static std::vector<std::uint32_t> observed_member_values() {
    const std::lock_guard lock{mutex};
    return member_values;
  }

  [[nodiscard]] static std::vector<std::uint32_t> values() {
    const std::lock_guard lock{mutex};
    return sink_values;
  }

  [[nodiscard]] static std::vector<std::uint64_t> sequences() {
    const std::lock_guard lock{mutex};
    return sink_sequences;
  }

  [[nodiscard]] static std::vector<Tag> observed_source_tags() {
    const std::lock_guard lock{mutex};
    return source_tags;
  }

  [[nodiscard]] static std::vector<Tag> observed_sink_tags() {
    const std::lock_guard lock{mutex};
    return sink_tags;
  }

  static inline std::mutex mutex;
  static inline std::condition_variable cv;
  static inline std::atomic_bool fail_restart_start{};
  static inline std::atomic_bool feeder_prefix_required{};
  static inline std::atomic<std::size_t> source_compute_count{};
  static inline std::atomic<std::size_t> source_forward_start_count{};
  static inline std::atomic<std::size_t> source_restore_start_count{};
  static inline std::atomic<std::size_t> sink_start_count{};
  static inline std::atomic<std::size_t> member_start_count{};
  static inline std::atomic<std::size_t> member_forward_start_count{};
  static inline std::atomic<std::size_t> feeder_compute_count{};
  static inline std::atomic_bool feeder_prefix_published{};
  static inline std::atomic_bool feeder_post_data_published{};
  static inline std::atomic_bool feeder_post_published{};
  static inline std::atomic<std::size_t> unrelated_start_count{};
  static inline std::atomic<std::size_t> unrelated_forward_start_count{};
  static inline std::atomic_bool restart_accepted{};
  static inline std::atomic<ErrorCode> restart_error{ErrorCode::kSuccess};
  static inline std::vector<std::uint32_t> sink_values;
  static inline std::vector<std::uint64_t> sink_sequences;
  static inline std::vector<Tag> source_tags;
  static inline std::vector<Tag> sink_tags;
  static inline std::vector<ActivationReason> sink_activation_reasons;
  static inline std::vector<OnViolation> sink_activation_policies;
  static inline std::vector<std::uint32_t> member_values;
};

// Periodic source (50 ms) that emits 4 samples on activation 0 then calls request_restart().
// Activation 1 emits one post-restart sample. The lifecycle stage handler counts
// kRestartForward and kRestoreOld transitions to distinguish success from rollback.
// Used by PublishedRestartDiscardsOldIngressAndRearmsAfterStart.
class ChannelRestartSourceOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.output(output, "output").max_emits_per_compute(4U);
    spec.lifecycle().stage(LifecycleStage::kStart, &ChannelRestartSourceOperator::start_stage);
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    // Leave enough time for the joined Channel pump to re-arm and expose preserved ingress after
    // rollback before the next clock witness advances global candidate time.
    result.trigger(OnClock{.period = 50ms});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    // Parallel partition owners may observe their initial clock witnesses in either order. The
    // multi-member closure test defers this source's scripted first activation until the feeder
    // has proved that its prefix belongs to the generation the restart must discard.
    if (ChannelRestartProbeState::feeder_prefix_required.load(std::memory_order_acquire) &&
        !ChannelRestartProbeState::feeder_prefix_published.load(std::memory_order_acquire)) {
      return {};
    }
    const std::size_t index =
        ChannelRestartProbeState::source_compute_count.fetch_add(1U, std::memory_order_relaxed);
    if (index >= 2U) {
      return {};
    }
    ChannelRestartProbeState::record_source_tag(context.tag());
    const TimePoint activation = context.activation_time();
    if (!activation.valid()) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    const std::uint32_t first_value = index == 0U ? 1U : 5U;
    const std::uint32_t emit_count = index == 0U ? 4U : 1U;
    for (std::uint32_t offset = 0U; offset < emit_count; ++offset) {
      auto emitted = output.emit(Sample<std::uint32_t>{
          first_value + offset,
          SampleMetadata{.capture_timestamp_ns = activation.timestamp_ns,
                         .source_clock_id = activation.clock_id},
      });
      if (!emitted) {
        return emitted;
      }
    }
    if (index == 0U) {
      auto receipt = context.request_restart();
      if (!receipt) {
        ChannelRestartProbeState::restart_error.store(receipt.error().code,
                                                      std::memory_order_relaxed);
        return make_unexpected(std::move(receipt.error()));
      }
      ChannelRestartProbeState::restart_accepted.store(true, std::memory_order_release);
    }
    return {};
  }

  LifecycleStatus start_stage(LifecycleContext& context) noexcept {
    if (context.transition_phase() == LifecycleTransitionPhase::kRestartForward) {
      ChannelRestartProbeState::source_forward_start_count.fetch_add(1U, std::memory_order_relaxed);
      if (ChannelRestartProbeState::fail_restart_start.load(std::memory_order_relaxed)) {
        return LifecycleStatus::kFatalFailure;
      }
    } else if (context.transition_phase() == LifecycleTransitionPhase::kRestoreOld) {
      ChannelRestartProbeState::source_restore_start_count.fetch_add(1U, std::memory_order_relaxed);
    }
    return LifecycleStatus::kOk;
  }

  Output<std::uint32_t> output;
};

// Sink with queue_depth=4; triggered by OnAnyInput with LatestWithin(1 s) and Skip on missing.
// Records each activation's data value, sequence, tag, activation reason, and policy so
// restart tests can verify which generation's samples are observed.
// Used by PublishedRestartDiscardsOldIngressAndRearmsAfterStart.
class ChannelRestartSinkOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(input, "input").queue_depth(4U);
    spec.lifecycle().stage(LifecycleStage::kStart, &ChannelRestartSinkOperator::start_stage);
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnAnyInput{})
        .input(input, LatestWithin{.max_age = 1s})
        .on_missing(input, Skip{})
        .max_start_delay(1s)
        .on_violation(OnViolation::kEmitWitness);
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    auto sample = input.receive();
    if (!sample) {
      return make_unexpected(std::move(sample.error()));
    }
    ChannelRestartProbeState::record_sink(*sample, context.tag(), context.activation());
    return {};
  }

  LifecycleStatus start_stage(LifecycleContext&) noexcept {
    ChannelRestartProbeState::sink_start_count.fetch_add(1U, std::memory_order_relaxed);
    return LifecycleStatus::kOk;
  }

  Input<std::uint32_t> input;
};

// Bystander operator with a 30 s clock period — never triggered during the test window.
// Counts both total and kRestartForward start events to verify that restart does not
// propagate forward-start to operators outside the lifecycle closure.
// Used by PublishedRestartDiscardsOldIngressAndRearmsAfterStart.
class ChannelRestartUnrelatedOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.lifecycle().stage(LifecycleStage::kStart, &ChannelRestartUnrelatedOperator::start_stage);
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 30s});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override { return {}; }

  LifecycleStatus start_stage(LifecycleContext& context) noexcept {
    ChannelRestartProbeState::unrelated_start_count.fetch_add(1U, std::memory_order_relaxed);
    if (context.transition_phase() == LifecycleTransitionPhase::kRestartForward) {
      ChannelRestartProbeState::unrelated_forward_start_count.fetch_add(1U,
                                                                        std::memory_order_relaxed);
    }
    return LifecycleStatus::kOk;
  }
};

// Retains bounded observations for a fast kDropOldest side input sampled by an independent clock.
// The source deliberately outruns both the queue depth and the sink cadence. Arrival witnesses
// must release readiness credit without releasing the temporal sample retained for LatestWithin
// or ReuseLast.
struct ChannelCreditRetirementProbeState {
  static constexpr std::size_t kRestartAfter = 96U;
  static constexpr std::size_t kRequiredPublications = 256U;
  static constexpr std::size_t kRequiredSelections = 4U;

  static void reset() {
    const std::lock_guard lock{mutex};
    source_compute_count.store(0U, std::memory_order_relaxed);
    sink_compute_count.store(0U, std::memory_order_relaxed);
    restart_accepted.store(false, std::memory_order_relaxed);
    restart_error.store(ErrorCode::kSuccess, std::memory_order_relaxed);
    selected_values.clear();
    selected_sequences.clear();
  }

  static void record_source() {
    source_compute_count.fetch_add(1U, std::memory_order_release);
    cv.notify_all();
  }

  static void record_sink(const Sample<std::uint32_t>& sample) {
    const std::lock_guard lock{mutex};
    selected_values.push_back(sample.data);
    selected_sequences.push_back(sample.metadata.sequence);
    sink_compute_count.store(selected_values.size(), std::memory_order_release);
    cv.notify_all();
  }

  [[nodiscard]] static bool wait_for_progress() {
    std::unique_lock lock{mutex};
    return cv.wait_for(lock, 5s, [] {
      return source_compute_count.load(std::memory_order_acquire) >= kRequiredPublications &&
             sink_compute_count.load(std::memory_order_acquire) >= kRequiredSelections &&
             restart_accepted.load(std::memory_order_acquire);
    });
  }

  [[nodiscard]] static std::vector<std::uint32_t> values() {
    const std::lock_guard lock{mutex};
    return selected_values;
  }

  [[nodiscard]] static std::vector<std::uint64_t> sequences() {
    const std::lock_guard lock{mutex};
    return selected_sequences;
  }

  static inline std::mutex mutex;
  static inline std::condition_variable cv;
  static inline std::atomic<std::size_t> source_compute_count{};
  static inline std::atomic<std::size_t> sink_compute_count{};
  static inline std::atomic_bool restart_accepted{};
  static inline std::atomic<ErrorCode> restart_error{ErrorCode::kSuccess};
  static inline std::vector<std::uint32_t> selected_values;
  static inline std::vector<std::uint64_t> selected_sequences;
};

class ChannelCreditRetirementSourceOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.output(output, "output");
    spec.output(holo_fallback, "holo-fallback");
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 100us});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    const std::size_t index =
        ChannelCreditRetirementProbeState::source_compute_count.load(std::memory_order_relaxed);
    const TimePoint activation = context.activation_time();
    if (!activation.valid()) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    auto emitted = output.emit(Sample<std::uint32_t>{
        static_cast<std::uint32_t>(index + 1U),
        SampleMetadata{.capture_timestamp_ns = activation.timestamp_ns,
                       .source_clock_id = activation.clock_id},
    });
    if (!emitted) {
      return emitted;
    }
    ChannelCreditRetirementProbeState::record_source();
    if (index + 1U == ChannelCreditRetirementProbeState::kRestartAfter) {
      auto receipt = context.request_restart();
      if (!receipt) {
        ChannelCreditRetirementProbeState::restart_error.store(receipt.error().code,
                                                               std::memory_order_relaxed);
        return make_unexpected(std::move(receipt.error()));
      }
      ChannelCreditRetirementProbeState::restart_accepted.store(true, std::memory_order_release);
      ChannelCreditRetirementProbeState::cv.notify_all();
    }
    return {};
  }

  Output<std::uint32_t> output;
  Output<std::uint32_t> holo_fallback;
};

class ChannelCreditRetirementSinkOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(input, "input").queue_depth(1U);
    spec.input(holo_fallback, "holo-fallback").queue_depth(2U);
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 3ms})
        .input(input, LatestWithin{.max_age = 1s})
        .on_missing(input, ReuseLast{.max_age = 1s});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    auto sample = input.receive();
    if (!sample) {
      return make_unexpected(std::move(sample.error()));
    }
    ChannelCreditRetirementProbeState::record_sink(*sample);
    return {};
  }

  Input<std::uint32_t> input;
  Input<std::uint32_t> holo_fallback;
};

// Coordinates an independently notified Operator with one retained Channel side input. The
// source publishes exactly once; two later notifications prove that arrival alone never activates
// the sink and that the committed sample remains available through ReuseLast.
struct NotifiedSideInputProbeState {
  static void reset() {
    const std::lock_guard lock{mutex};
    sender.reset();
    source_emitted.store(false, std::memory_order_relaxed);
    sink_compute_count.store(0U, std::memory_order_relaxed);
    values.fill(0U);
    sequences.fill(0U);
  }

  static void publish_sender(NotificationSender value) {
    {
      const std::lock_guard lock{mutex};
      sender = std::move(value);
    }
    cv.notify_all();
  }

  static void publish_source_complete() noexcept {
    source_emitted.store(true, std::memory_order_release);
    cv.notify_all();
  }

  static void record_sink(const Sample<std::uint32_t>& sample) {
    const std::lock_guard lock{mutex};
    const std::size_t index = sink_compute_count.load(std::memory_order_relaxed);
    if (index < values.size()) {
      values[index] = sample.data;
      sequences[index] = sample.metadata.sequence;
    }
    sink_compute_count.store(index + 1U, std::memory_order_release);
    cv.notify_all();
  }

  [[nodiscard]] static bool wait_until_sender() {
    std::unique_lock lock{mutex};
    return cv.wait_for(lock, 5s, [] { return sender.has_value(); });
  }

  [[nodiscard]] static bool wait_until_source() {
    std::unique_lock lock{mutex};
    return cv.wait_for(lock, 5s, [] { return source_emitted.load(std::memory_order_acquire); });
  }

  [[nodiscard]] static bool wait_until_sink_count(std::size_t expected) {
    std::unique_lock lock{mutex};
    return cv.wait_for(lock, 5s, [expected] {
      return sink_compute_count.load(std::memory_order_acquire) >= expected;
    });
  }

  [[nodiscard]] static NotificationSender sender_copy() {
    const std::lock_guard lock{mutex};
    return sender.value_or(NotificationSender{});
  }

  static inline std::mutex mutex;
  static inline std::condition_variable cv;
  static inline std::optional<NotificationSender> sender;
  static inline std::atomic_bool source_emitted{};
  static inline std::atomic<std::size_t> sink_compute_count{};
  static inline std::array<std::uint32_t, 2U> values{};
  static inline std::array<std::uint64_t, 2U> sequences{};
};

class NotifiedSideInputSourceOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override { spec.output(output, "output"); }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 1ms});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    if (NotifiedSideInputProbeState::source_emitted.load(std::memory_order_acquire)) {
      return {};
    }
    const TimePoint activation = context.activation_time();
    auto emitted = output.emit(Sample<std::uint32_t>{
        42U,
        SampleMetadata{.capture_timestamp_ns = activation.timestamp_ns,
                       .source_clock_id = activation.clock_id},
    });
    if (!emitted) {
      return emitted;
    }
    NotifiedSideInputProbeState::publish_source_complete();
    return {};
  }

  Output<std::uint32_t> output;
};

class NotifiedSideInputSinkOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(input, "input").queue_depth(1U);
    spec.notification_source(notification, "notification")
        .capacity(1U)
        .sender_reference_capacity(1U);
    spec.lifecycle().stage(LifecycleStage::kArm, &NotifiedSideInputSinkOperator::arm);
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
      return make_unexpected(std::move(sample.error()));
    }
    NotifiedSideInputProbeState::record_sink(*sample);
    return {};
  }

  LifecycleStatus arm(LifecycleContext& context) noexcept {
    auto sender = context.notification_sender(notification);
    if (!sender) {
      return LifecycleStatus::kFatalFailure;
    }
    NotifiedSideInputProbeState::publish_sender(std::move(*sender));
    return LifecycleStatus::kOk;
  }

  Input<std::uint32_t> input;
  NotificationSource notification;
};

struct OnAnyInputParityProbeState {
  static void reset() noexcept {
    source_emitted.store(false, std::memory_order_relaxed);
    sink_value.store(0U, std::memory_order_relaxed);
    sink_compute_count.store(0U, std::memory_order_relaxed);
  }

  [[nodiscard]] static bool wait_for_sink() {
    std::unique_lock lock{mutex};
    return cv.wait_for(
        lock, 5s, [] { return sink_compute_count.load(std::memory_order_acquire) == 1U; });
  }

  static inline std::mutex mutex;
  static inline std::condition_variable cv;
  static inline std::atomic_bool source_emitted{};
  static inline std::atomic<std::uint32_t> sink_value{};
  static inline std::atomic<std::size_t> sink_compute_count{};
};

class OnAnyInputParitySourceOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override { spec.output(output, "output"); }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 30s});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    if (OnAnyInputParityProbeState::source_emitted.exchange(true, std::memory_order_relaxed)) {
      return {};
    }
    const TimePoint activation = context.activation_time();
    return output.emit(Sample<std::uint32_t>{
        42U,
        SampleMetadata{.capture_timestamp_ns = activation.timestamp_ns,
                       .source_clock_id = activation.clock_id},
    });
  }

  Output<std::uint32_t> output;
};

class OnAnyInputParitySinkOperator final : public Operator<> {
 public:
  explicit OnAnyInputParitySinkOperator(bool explicit_contract)
      : explicit_contract_(explicit_contract) {}

  void setup(OperatorSpec& spec) override { spec.input(input, "input").queue_depth(2U); }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    if (explicit_contract_) {
      result.trigger(OnAnyInput{});
    }
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    auto sample = input.receive();
    if (!sample) {
      return make_unexpected(std::move(sample.error()));
    }
    OnAnyInputParityProbeState::sink_value.store(sample->data, std::memory_order_relaxed);
    OnAnyInputParityProbeState::sink_compute_count.fetch_add(1U, std::memory_order_release);
    OnAnyInputParityProbeState::cv.notify_all();
    return {};
  }

  Input<std::uint32_t> input;

 private:
  bool explicit_contract_{};
};

enum class ClockInputTriggerRoute : std::uint8_t {
  kOnClock,
  kOnPredicate,
};

struct ClockInputPlanOptions {
  bool block_source_after_first_emit{};
  bool restart_on_first_sink_compute{true};
  std::chrono::nanoseconds deadline{50ms};
  OnViolation violation{OnViolation::kFailRun};
};

// Proves that ordinary and predicate clock triggers can retain a typed side input, execute,
// restart, and execute again from the same immutable plan. Reconstruction-only coverage cannot
// expose the historical startup rejection that incorrectly treated every clock trigger as
// input-free.
struct ClockPredicateInputProbeState {
  static void reset() noexcept {
    source_compute_count.store(0U, std::memory_order_relaxed);
    predicate_evaluation_count.store(0U, std::memory_order_relaxed);
    sink_compute_count.store(0U, std::memory_order_relaxed);
    sink_forward_start_count.store(0U, std::memory_order_relaxed);
    restart_accepted.store(false, std::memory_order_relaxed);
    restart_error.store(ErrorCode::kSuccess, std::memory_order_relaxed);
    last_value.store(0U, std::memory_order_relaxed);
    source_block_entered.store(false, std::memory_order_relaxed);
    release_source.store(false, std::memory_order_relaxed);
  }

  [[nodiscard]] static bool wait_for_restart_execution() {
    std::unique_lock lock{mutex};
    return cv.wait_for(lock, 5s, [] {
      return sink_compute_count.load(std::memory_order_acquire) >= 2U &&
             sink_forward_start_count.load(std::memory_order_acquire) == 1U &&
             restart_accepted.load(std::memory_order_acquire);
    });
  }

  [[nodiscard]] static bool wait_for_source_block(
      std::chrono::steady_clock::duration timeout = 5s) {
    std::unique_lock lock{mutex};
    return cv.wait_for(
        lock, timeout, [] { return source_block_entered.load(std::memory_order_acquire); });
  }

  static void block_source_until_released() {
    std::unique_lock lock{mutex};
    source_block_entered.store(true, std::memory_order_release);
    cv.notify_all();
    cv.wait(lock, [] { return release_source.load(std::memory_order_acquire); });
  }

  static void release_source_block() noexcept {
    {
      const std::lock_guard lock{mutex};
      release_source.store(true, std::memory_order_release);
    }
    cv.notify_all();
  }

  static inline std::mutex mutex;
  static inline std::condition_variable cv;
  static inline std::atomic<std::size_t> source_compute_count{};
  static inline std::atomic<std::size_t> predicate_evaluation_count{};
  static inline std::atomic<std::size_t> sink_compute_count{};
  static inline std::atomic<std::size_t> sink_forward_start_count{};
  static inline std::atomic_bool restart_accepted{};
  static inline std::atomic<ErrorCode> restart_error{ErrorCode::kSuccess};
  static inline std::atomic<std::uint32_t> last_value{};
  static inline std::atomic_bool source_block_entered{};
  static inline std::atomic_bool release_source{};
};

class ClockInputSourceReleaseGuard {
 public:
  ClockInputSourceReleaseGuard() = default;
  ClockInputSourceReleaseGuard(const ClockInputSourceReleaseGuard&) = delete;
  ClockInputSourceReleaseGuard& operator=(const ClockInputSourceReleaseGuard&) = delete;
  ~ClockInputSourceReleaseGuard() { ClockPredicateInputProbeState::release_source_block(); }
};

class ClockInputPreDispatchReleaseGuard {
 public:
  explicit ClockInputPreDispatchReleaseGuard(
      std::shared_ptr<detail::RuntimePreDispatchTestControl> control)
      : control_(std::move(control)) {}
  ClockInputPreDispatchReleaseGuard(const ClockInputPreDispatchReleaseGuard&) = delete;
  ClockInputPreDispatchReleaseGuard& operator=(const ClockInputPreDispatchReleaseGuard&) = delete;
  ~ClockInputPreDispatchReleaseGuard() {
    if (control_ != nullptr) {
      control_->release();
    }
  }

 private:
  std::shared_ptr<detail::RuntimePreDispatchTestControl> control_;
};

class ClockPredicateInputSourceOperator final : public Operator<> {
 public:
  explicit ClockPredicateInputSourceOperator(bool block_after_first_emit)
      : block_after_first_emit_(block_after_first_emit) {}

  void setup(OperatorSpec& spec) override { spec.output(output, "output"); }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    // Keep publication slower than the predicate clock so arrival ingestion cannot dominate the
    // independent clock candidate under test. A fresh sample still arrives after restart.
    result.trigger(OnClock{.period = 5ms});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    const std::size_t index = ClockPredicateInputProbeState::source_compute_count.fetch_add(
        1U, std::memory_order_relaxed);
    const TimePoint activation = context.activation_time();
    auto emitted = output.emit(Sample<std::uint32_t>{
        static_cast<std::uint32_t>(index + 1U),
        SampleMetadata{.capture_timestamp_ns = activation.timestamp_ns,
                       .source_clock_id = activation.clock_id},
    });
    if (!emitted) {
      return make_unexpected(std::move(emitted.error()));
    }
    if (block_after_first_emit_ && index == 0U) {
      ClockPredicateInputProbeState::block_source_until_released();
    }
    return {};
  }

  Output<std::uint32_t> output;

 private:
  bool block_after_first_emit_{};
};

class ClockPredicateInputSinkOperator final : public Operator<> {
 public:
  ClockPredicateInputSinkOperator(ClockInputTriggerRoute route, bool restart_on_first_compute,
                                  std::chrono::nanoseconds deadline, OnViolation violation)
      : route_(route),
        restart_on_first_compute_(restart_on_first_compute),
        max_start_delay_(deadline),
        violation_(violation) {}

  void setup(OperatorSpec& spec) override {
    spec.input(input, "input").queue_depth(1U);
    spec.lifecycle().stage(LifecycleStage::kStart, &ClockPredicateInputSinkOperator::start_stage);
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    if (route_ == ClockInputTriggerRoute::kOnPredicate) {
      auto evaluator =
          predicate("clock-input-present",
                    predicate_inputs(input),
                    [](const PredicateEvaluationContext&,
                       const PredicateInputView<std::uint32_t>& selected) noexcept {
                      ClockPredicateInputProbeState::predicate_evaluation_count.fetch_add(
                          1U, std::memory_order_relaxed);
                      return selected.sample().data != 0U;
                    });
      result.trigger(OnPredicate{
          .evaluate_on = OnClock{.period = 2ms},
          .predicate = std::move(evaluator),
          .firing = PredicateFiring::kOnEveryTrueEvaluation,
      });
    } else {
      result.trigger(OnClock{.period = 2ms});
    }
    result.input(input, LatestWithin{.max_age = 1s})
        .on_missing(input, Skip{})
        .max_start_delay(max_start_delay_)
        .on_violation(violation_);
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    auto sample = input.receive();
    if (!sample) {
      return make_unexpected(std::move(sample.error()));
    }
    ClockPredicateInputProbeState::last_value.store(sample->data, std::memory_order_relaxed);
    const std::size_t count =
        ClockPredicateInputProbeState::sink_compute_count.fetch_add(1U, std::memory_order_release) +
        1U;
    if (restart_on_first_compute_ && count == 1U) {
      auto receipt = context.request_restart();
      if (!receipt) {
        ClockPredicateInputProbeState::restart_error.store(receipt.error().code,
                                                           std::memory_order_relaxed);
        return make_unexpected(std::move(receipt.error()));
      }
      ClockPredicateInputProbeState::restart_accepted.store(true, std::memory_order_release);
    }
    ClockPredicateInputProbeState::cv.notify_all();
    return {};
  }

  LifecycleStatus start_stage(LifecycleContext& context) noexcept {
    if (context.transition_phase() == LifecycleTransitionPhase::kRestartForward) {
      ClockPredicateInputProbeState::sink_forward_start_count.fetch_add(1U,
                                                                        std::memory_order_release);
      ClockPredicateInputProbeState::cv.notify_all();
    }
    return LifecycleStatus::kOk;
  }

  Input<std::uint32_t> input;

 private:
  ClockInputTriggerRoute route_{ClockInputTriggerRoute::kOnClock};
  bool restart_on_first_compute_{true};
  std::chrono::nanoseconds max_start_delay_{50ms};
  OnViolation violation_{OnViolation::kFailRun};
};

// Publish one full old-generation prefix, then one sample after the dependency member restarts.
// Drives the member via a separate 'driver' output to decouple trigger timing from data timing.
// Used by MultiMemberClosureFencesNonTargetChannelInput.
class ChannelRestartMemberFeederOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.output(output, "output").max_emits_per_compute(4U);
    spec.output(driver, "driver").max_emits_per_compute(1U);
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 10ms});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    const std::size_t index =
        ChannelRestartProbeState::feeder_compute_count.fetch_add(1U, std::memory_order_relaxed);
    const bool member_restarted =
        ChannelRestartProbeState::member_forward_start_count.load(std::memory_order_acquire) != 0U;
    if (index != 0U && !member_restarted) {
      return {};
    }
    const TimePoint activation = context.activation_time();
    if (!activation.valid()) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    if (index != 0U && ChannelRestartProbeState::feeder_post_data_published.exchange(
                           true, std::memory_order_acq_rel)) {
      if (ChannelRestartProbeState::feeder_post_published.exchange(true,
                                                                   std::memory_order_acq_rel)) {
        return {};
      }
      return driver.emit(Sample<std::uint32_t>{
          1U,
          SampleMetadata{.capture_timestamp_ns = activation.timestamp_ns,
                         .source_clock_id = activation.clock_id},
      });
    }
    const std::uint32_t first = index == 0U ? 1U : 5U;
    const std::uint32_t count = index == 0U ? 4U : 1U;
    for (std::uint32_t offset = 0U; offset < count; ++offset) {
      auto emitted = output.emit(Sample<std::uint32_t>{
          first + offset,
          SampleMetadata{.capture_timestamp_ns = activation.timestamp_ns,
                         .source_clock_id = activation.clock_id},
      });
      if (!emitted) {
        return emitted;
      }
    }
    if (index == 0U) {
      ChannelRestartProbeState::publish_feeder_prefix();
    }
    return {};
  }

  Output<std::uint32_t> output;
  Output<std::uint32_t> driver;
};

// Observe only the post-restart prefix retained for a non-target lifecycle-closure member.
// Triggered by OnEach(driver) with DrainAll on the data input; forward-start count signals
// the feeder that the member has been restarted and post-restart data may be published.
// Used by MultiMemberClosureFencesNonTargetChannelInput.
class ChannelRestartMemberOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(input, "input").queue_depth(4U);
    spec.input(driver, "driver").queue_depth(1U);
    spec.lifecycle().stage(LifecycleStage::kStart, &ChannelRestartMemberOperator::start_stage);
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnEach{driver}).input(input, DrainAll{});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    auto selected = input.selected_batch();
    if (!selected) {
      return make_unexpected(std::move(selected.error()));
    }
    auto trigger = driver.receive();
    if (!trigger) {
      return make_unexpected(std::move(trigger.error()));
    }
    ChannelRestartProbeState::record_member_batch(*selected);
    return {};
  }

  LifecycleStatus start_stage(LifecycleContext& context) noexcept {
    ChannelRestartProbeState::member_start_count.fetch_add(1U, std::memory_order_relaxed);
    if (context.transition_phase() == LifecycleTransitionPhase::kRestartForward) {
      ChannelRestartProbeState::member_forward_start_count.fetch_add(1U, std::memory_order_release);
    }
    return LifecycleStatus::kOk;
  }

  Input<std::uint32_t> input;
  Input<std::uint32_t> driver;
};

// Shared static-storage aggregator for window-selector restart tests. Records the anchor
// value and the full bracketed batch observed after the first window close post-restart.
// Representative test: PendingWindowReprimesAcrossPublishedRestart.
struct ChannelWindowRestartProbeState {
  static void reset() {
    const std::lock_guard lock{mutex};
    source_compute_count.store(0U, std::memory_order_relaxed);
    source_forward_start_count.store(0U, std::memory_order_relaxed);
    restart_accepted.store(false, std::memory_order_relaxed);
    anchor_value = 0U;
    batch_values.clear();
    batch_sequences.clear();
  }

  static void record(const Sample<std::uint32_t>& anchor,
                     const SelectedBatchView<std::uint32_t>& selected) {
    const std::lock_guard lock{mutex};
    anchor_value = anchor.data;
    batch_values.clear();
    batch_sequences.clear();
    batch_values.reserve(selected.size());
    batch_sequences.reserve(selected.size());
    for (const Sample<std::uint32_t>& sample : selected) {
      batch_values.push_back(sample.data);
      batch_sequences.push_back(sample.metadata.sequence);
    }
    cv.notify_all();
  }

  [[nodiscard]] static bool wait_for_batch() {
    std::unique_lock lock{mutex};
    return cv.wait_for(lock, 5s, [] { return !batch_values.empty(); });
  }

  [[nodiscard]] static std::uint32_t observed_anchor() {
    const std::lock_guard lock{mutex};
    return anchor_value;
  }

  [[nodiscard]] static std::vector<std::uint32_t> observed_values() {
    const std::lock_guard lock{mutex};
    return batch_values;
  }

  [[nodiscard]] static std::vector<std::uint64_t> observed_sequences() {
    const std::lock_guard lock{mutex};
    return batch_sequences;
  }

  static inline std::mutex mutex;
  static inline std::condition_variable cv;
  static inline std::atomic<std::size_t> source_compute_count{};
  static inline std::atomic<std::size_t> source_forward_start_count{};
  static inline std::atomic_bool restart_accepted{};
  static inline std::uint32_t anchor_value{};
  static inline std::vector<std::uint32_t> batch_values;
  static inline std::vector<std::uint64_t> batch_sequences;
};

// Prime one incomplete window, restart, and then publish one complete replacement window.
// Activation 0: deposits 2 windowed samples and 1 anchor, then calls request_restart().
// Activation 1: publishes 5 windowed samples with anchors at values 110 and 120.
// Used by PendingWindowReprimesAcrossPublishedRestart.
class ChannelWindowRestartSourceOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.output(anchor, "anchor").max_emits_per_compute(2U);
    spec.output(windowed, "windowed").max_emits_per_compute(5U);
    spec.lifecycle().stage(LifecycleStage::kStart,
                           &ChannelWindowRestartSourceOperator::start_stage);
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 50ms});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    const std::size_t index = ChannelWindowRestartProbeState::source_compute_count.fetch_add(
        1U, std::memory_order_relaxed);
    if (index >= 2U) {
      return {};
    }
    const TimePoint activation = context.activation_time();
    if (!activation.valid()) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    const auto emit_captured =
        [&](Output<std::uint32_t>& output, std::uint32_t value, std::int64_t offset) {
          return output.emit(Sample<std::uint32_t>{
              value,
              SampleMetadata{.capture_timestamp_ns = activation.timestamp_ns + offset,
                             .source_clock_id = activation.clock_id},
          });
        };
    if (index == 0U) {
      auto emitted = emit_captured(windowed, 5U, 5);
      if (!emitted) {
        return emitted;
      }
      emitted = emit_captured(anchor, 10U, 10);
      if (!emitted) {
        return emitted;
      }
      emitted = emit_captured(windowed, 10U, 10);
      if (!emitted) {
        return emitted;
      }
      auto receipt = context.request_restart();
      if (!receipt) {
        return make_unexpected(std::move(receipt.error()));
      }
      ChannelWindowRestartProbeState::restart_accepted.store(true, std::memory_order_release);
      return {};
    }

    for (const auto [value, offset] : std::array{
             std::pair{105U, 5}, std::pair{110U, 10}, std::pair{115U, 15}, std::pair{120U, 20}}) {
      auto emitted = emit_captured(windowed, value, offset);
      if (!emitted) {
        return emitted;
      }
      if (value == 110U || value == 120U) {
        emitted = emit_captured(anchor, value, offset);
        if (!emitted) {
          return emitted;
        }
      }
    }
    return emit_captured(windowed, 125U, 25);
  }

  LifecycleStatus start_stage(LifecycleContext& context) noexcept {
    if (context.transition_phase() == LifecycleTransitionPhase::kRestartForward) {
      ChannelWindowRestartProbeState::source_forward_start_count.fetch_add(
          1U, std::memory_order_relaxed);
    }
    return LifecycleStatus::kOk;
  }

  Output<std::uint32_t> anchor;
  Output<std::uint32_t> windowed;
};

// Record the first closed window after the producer restart. Uses BracketedWindow with
// lookbehind=1, lookahead=1, max_samples=5 on 'windowed'; anchored by OnWindow trigger.
// Used by PendingWindowReprimesAcrossPublishedRestart.
class ChannelWindowRestartSinkOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(anchor, "anchor").queue_depth(8U);
    spec.input(windowed, "windowed").queue_depth(16U);
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnWindow{.anchor = anchor, .max_wait = 30s, .max_pending_windows = 2U})
        .input(anchor, Exact{})
        .input(windowed, BracketedWindow{.lookbehind = 1U, .lookahead = 1U, .max_samples = 5U})
        .on_missing(windowed, Skip{});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    auto anchor_sample = anchor.receive();
    if (!anchor_sample) {
      return make_unexpected(std::move(anchor_sample.error()));
    }
    auto selected = windowed.selected_batch();
    if (!selected) {
      return make_unexpected(std::move(selected.error()));
    }
    ChannelWindowRestartProbeState::record(*anchor_sample, *selected);
    return {};
  }

  Input<std::uint32_t> anchor;
  Input<std::uint32_t> windowed;
};

// Runtime-mutable parameter holding the uint32 payload emitted by
// ChannelNextComputeSourceOperator. Initial value is 10; tests patch it to 20 via
// NextCompute to verify Channel epoch continuity across a param update.
struct ChannelNextComputeParams {
  std::int32_t value{10};
};

HOLOSCAN_PARAM_SCHEMA(ChannelNextComputeParams,
                      param_schema("holoscan.test.ChannelNextComputeParams", 1),
                      field<&ChannelNextComputeParams::value>("value").runtime_mutable());

// Shared static-storage aggregator for NextCompute parameter-update tests. Provides a
// mutex-guarded barrier (enter/release first compute) that lets the test submit a param
// patch while the source is still inside its first activation callback.
// Representative test: NextComputePreservesChannelEpochAndPublishedSequence.
struct ChannelNextComputeProbeState {
  static void reset() {
    const std::lock_guard lock{mutex};
    first_compute_entered = false;
    release_first_compute = false;
    source_compute_count.store(0U, std::memory_order_relaxed);
    sink_values.clear();
    sink_sequences.clear();
  }

  static void enter_first_compute() {
    std::unique_lock lock{mutex};
    first_compute_entered = true;
    cv.notify_all();
    cv.wait_for(lock, 5s, [] { return release_first_compute; });
  }

  [[nodiscard]] static bool wait_for_first_compute() {
    std::unique_lock lock{mutex};
    return cv.wait_for(lock, 5s, [] { return first_compute_entered; });
  }

  static void release_first() {
    const std::lock_guard lock{mutex};
    release_first_compute = true;
    cv.notify_all();
  }

  static void record_sink(const Sample<std::uint32_t>& sample) {
    const std::lock_guard lock{mutex};
    sink_values.push_back(sample.data);
    sink_sequences.push_back(sample.metadata.sequence);
    cv.notify_all();
  }

  [[nodiscard]] static bool wait_for_sink(std::size_t count) {
    std::unique_lock lock{mutex};
    return cv.wait_for(lock, 5s, [count] { return sink_values.size() >= count; });
  }

  static inline std::mutex mutex;
  static inline std::condition_variable cv;
  static inline bool first_compute_entered{};
  static inline bool release_first_compute{};
  static inline std::atomic<std::size_t> source_compute_count{};
  static inline std::vector<std::uint32_t> sink_values;
  static inline std::vector<std::uint64_t> sink_sequences;
};

// Clock-triggered source (1 ms period) that emits params().value cast to uint32 each tick.
// Blocks inside its first compute via enter_first_compute() to give the test time to
// submit a NextCompute param patch before the emission completes.
// Used by NextComputePreservesChannelEpochAndPublishedSequence.
class ChannelNextComputeSourceOperator final : public Operator<ChannelNextComputeParams> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.output(output, "output").max_emits_per_compute(1U);
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 1ms});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    const std::size_t index =
        ChannelNextComputeProbeState::source_compute_count.fetch_add(1U, std::memory_order_relaxed);
    if (index >= 2U) {
      return {};
    }
    const TimePoint activation = context.activation_time();
    if (!activation.valid()) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    auto emitted = output.emit(Sample<std::uint32_t>{
        static_cast<std::uint32_t>(context.params<ChannelNextComputeParams>().value),
        SampleMetadata{.capture_timestamp_ns = activation.timestamp_ns,
                       .source_clock_id = activation.clock_id},
    });
    if (!emitted) {
      return emitted;
    }
    if (index == 0U) {
      ChannelNextComputeProbeState::enter_first_compute();
    }
    return {};
  }

  Output<std::uint32_t> output;
};

// Passive sink that collects each received uint32 sample and its sequence number.
// Used by NextComputePreservesChannelEpochAndPublishedSequence.
class ChannelNextComputeSinkOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override { spec.input(input, "input").queue_depth(4U); }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnAnyInput{}).input(input, Exact{}).on_missing(input, Skip{});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    auto sample = input.receive();
    if (!sample) {
      return make_unexpected(std::move(sample.error()));
    }
    ChannelNextComputeProbeState::record_sink(*sample);
    return {};
  }

  Input<std::uint32_t> input;
};

//   source --[depth=4]--> sink
//   Clock: RealtimeClock (1 ms period); runtime param updates enabled (kEnabledIfAdmitted).
[[nodiscard]] ExecutionPlan make_channel_next_compute_plan() {
  Graph graph{"channel-next-compute"};
  const auto source = graph.op<ChannelNextComputeSourceOperator>("source");
  const auto sink = graph.op<ChannelNextComputeSinkOperator>("sink");
  graph.add_flow(source->output, sink->input, ConnectionOptions{.queue_depth = 4U});
  const ClockKey clock = graph.add_clock<RealtimeClock>("runtime-test-clock");
  graph.set_default_clock(clock);

  CompileOptions options;
  options.profile = Profile::kStandard;
  options.parameters.runtime_updates = ParamRuntimeUpdatePolicy::kEnabledIfAdmitted;
  return compile(graph, std::move(options));
}

// RAII guard that releases the ChannelNextComputeProbeState first-compute barrier on
// destruction. Ensures the source is never left blocked if a test ASSERT fails early.
class FirstComputeRelease final {
 public:
  ~FirstComputeRelease() {
    if (!released_) {
      ChannelNextComputeProbeState::release_first();
    }
  }

  void release() {
    ChannelNextComputeProbeState::release_first();
    released_ = true;
  }

 private:
  bool released_{};
};

// Selects which trigger path the ChannelSelectorCoherenceSinkOperator uses:
// OnEach(driver) or OnPredicate evaluated against predicate_stale.
// Used to parameterize NextComputeLatestWithinUsesOnePinnedGenerationAcrossSelectorRoutes.
enum class ChannelSelectorCoherenceRoute : std::uint8_t {
  kOnEach,
  kPredicate,
};

// Runtime-mutable max_age (range 1-20 ms) controlling both LatestWithin selector windows.
// Tests patch it from 5 ms to 20 ms via NextCompute to widen stale-sample acceptance.
struct ChannelSelectorCoherenceParams {
  std::chrono::milliseconds max_age{5};
};

HOLOSCAN_PARAM_SCHEMA(ChannelSelectorCoherenceParams,
                      param_schema("holoscan.test.ChannelSelectorCoherenceParams", 1),
                      field<&ChannelSelectorCoherenceParams::max_age>("max_age")
                          .runtime_mutable()
                          .range(std::chrono::milliseconds{1}, std::chrono::milliseconds{20}));

// Shared static-storage aggregator for selector generation-coherence tests. Records the
// values selected via predicate and compute routes plus the observed param generation to
// verify that both routes see the same coherent parameter snapshot in a single activation.
// Representative test: NextComputeLatestWithinUsesOnePinnedGenerationAcrossSelectorRoutes.
struct ChannelSelectorCoherenceProbeState {
  static void reset() {
    const std::lock_guard lock{mutex};
    source_compute_entered = false;
    release_source_compute = false;
    sink_compute_count = 0U;
    predicate_selected_value = 0U;
    compute_selected_value = 0U;
    observed_max_age = 0ms;
    observed_generation = 0U;
  }

  static void enter_source_compute() {
    std::unique_lock lock{mutex};
    source_compute_entered = true;
    cv.notify_all();
    cv.wait_for(lock, 5s, [] { return release_source_compute; });
  }

  [[nodiscard]] static bool wait_for_source_compute() {
    std::unique_lock lock{mutex};
    return cv.wait_for(lock, 5s, [] { return source_compute_entered; });
  }

  static void release_source() {
    const std::lock_guard lock{mutex};
    release_source_compute = true;
    cv.notify_all();
  }

  static void record_sink(std::uint32_t predicate_value, std::uint32_t compute_value,
                          std::chrono::milliseconds max_age, ParamGeneration generation) {
    const std::lock_guard lock{mutex};
    predicate_selected_value = predicate_value;
    compute_selected_value = compute_value;
    observed_max_age = max_age;
    observed_generation = generation;
    ++sink_compute_count;
    cv.notify_all();
  }

  [[nodiscard]] static bool wait_for_sink() {
    std::unique_lock lock{mutex};
    return cv.wait_for(lock, 5s, [] { return sink_compute_count != 0U; });
  }

  static inline std::mutex mutex;
  static inline std::condition_variable cv;
  static inline bool source_compute_entered{};
  static inline bool release_source_compute{};
  static inline std::uint32_t sink_compute_count{};
  static inline std::uint32_t predicate_selected_value{};
  static inline std::uint32_t compute_selected_value{};
  static inline std::chrono::milliseconds observed_max_age{};
  static inline ParamGeneration observed_generation{};
};

// Tick-0: emits stale samples on predicate_stale (value=98) and compute_stale (value=99),
// then blocks in compute so the test can submit a max_age patch before the cut is flushed.
// Tick-1: emits the driver that triggers the sink. The 10 ms phase separation lies outside
// the initial 5 ms max_age but inside the updated 20 ms max_age.
// Used by NextComputeLatestWithinUsesOnePinnedGenerationAcrossSelectorRoutes.
class ChannelSelectorCoherenceSourceOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.output(predicate_stale, "predicate_stale").max_emits_per_compute(1U);
    spec.output(compute_stale, "compute_stale").max_emits_per_compute(1U);
    spec.output(driver, "driver").max_emits_per_compute(1U);
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    // The first tick publishes the retained samples and the second tick publishes the driver.
    // Their exact 10 ms phase separation lies outside the initial 5 ms selector age and inside
    // the runtime-updated 20 ms age without introducing a retrograde readiness coordinate.
    result.trigger(OnClock{.period = 10ms});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    const std::size_t activation_index = activation_count_++;
    if (activation_index >= 2U) {
      return {};
    }
    const TimePoint activation = context.activation_time();
    if (!activation.valid()) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    if (activation_index == 1U) {
      return driver.emit(Sample<std::uint32_t>{
          1U,
          SampleMetadata{.capture_timestamp_ns = activation.timestamp_ns,
                         .source_clock_id = activation.clock_id},
      });
    }

    const SampleMetadata stale_metadata{
        .capture_timestamp_ns = activation.timestamp_ns,
        .source_clock_id = activation.clock_id,
    };
    auto predicate_result = predicate_stale.emit(Sample<std::uint32_t>{
        98U,
        stale_metadata,
    });
    if (!predicate_result) {
      return predicate_result;
    }
    auto compute_result = compute_stale.emit(Sample<std::uint32_t>{
        99U,
        stale_metadata,
    });
    if (!compute_result) {
      return compute_result;
    }
    // The process-local activation cut is flushed only after compute returns. Holding the first
    // callback lets the test submit NextCompute before either selector route can begin.
    ChannelSelectorCoherenceProbeState::enter_source_compute();
    return {};
  }

  Output<std::uint32_t> predicate_stale;
  Output<std::uint32_t> compute_stale;
  Output<std::uint32_t> driver;

 private:
  std::size_t activation_count_{};
};

// Receives via OnEach(driver) or OnPredicate; applies LatestWithin(max_age) from the
// runtime-mutable param to both stale inputs. Records selected values and the active param
// generation to verify a coherent snapshot across both selector routes per activation.
// Used by NextComputeLatestWithinUsesOnePinnedGenerationAcrossSelectorRoutes.
class ChannelSelectorCoherenceSinkOperator final : public Operator<ChannelSelectorCoherenceParams> {
 public:
  explicit ChannelSelectorCoherenceSinkOperator(ChannelSelectorCoherenceRoute route)
      : route_(route) {}

  void setup(OperatorSpec& spec) override {
    spec.input(predicate_stale, "predicate_stale").queue_depth(2U);
    spec.input(compute_stale, "compute_stale").queue_depth(2U);
    spec.input(driver, "driver").queue_depth(2U);
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    if (route_ == ChannelSelectorCoherenceRoute::kOnEach) {
      result.trigger(OnEach{driver});
    } else {
      auto evaluator = predicate(
          "channel-selector-generation-coherence",
          predicate_inputs(predicate_stale),
          [](const PredicateEvaluationContext&,
             const PredicateInputView<std::uint32_t>& selected_driver,
             const PredicateInputView<std::uint32_t>& selected_stale) noexcept {
            return selected_driver.sample().data == 1U && selected_stale.sample().data == 98U;
          });
      result.trigger(OnPredicate{
          .evaluate_on = OnInputUpdate{driver},
          .predicate = std::move(evaluator),
          .firing = PredicateFiring::kOnEveryTrueEvaluation,
      });
      result.input(driver, Exact{});
    }
    result
        .input(predicate_stale, LatestWithin{param_ref<&ChannelSelectorCoherenceParams::max_age>()})
        .on_missing(predicate_stale, Skip{})
        .input(compute_stale, LatestWithin{param_ref<&ChannelSelectorCoherenceParams::max_age>()})
        .on_missing(compute_stale, Skip{});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    auto selected_predicate = predicate_stale.receive();
    if (!selected_predicate) {
      return make_unexpected(std::move(selected_predicate.error()));
    }
    auto selected_compute = compute_stale.receive();
    if (!selected_compute) {
      return make_unexpected(std::move(selected_compute.error()));
    }
    auto selected_driver = driver.receive();
    if (!selected_driver) {
      return make_unexpected(std::move(selected_driver.error()));
    }
    if (selected_driver->data != 1U) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    ChannelSelectorCoherenceProbeState::record_sink(
        selected_predicate->data,
        selected_compute->data,
        context.params<ChannelSelectorCoherenceParams>().max_age,
        context.params_generation());
    return {};
  }

  Input<std::uint32_t> predicate_stale;
  Input<std::uint32_t> compute_stale;
  Input<std::uint32_t> driver;

 private:
  ChannelSelectorCoherenceRoute route_{ChannelSelectorCoherenceRoute::kOnEach};
};

// RAII guard that releases the ChannelSelectorCoherenceProbeState source-compute barrier
// on destruction. Ensures the source is never left blocked if a test ASSERT fails early.
class SelectorSourceRelease final {
 public:
  ~SelectorSourceRelease() {
    if (!released_) {
      ChannelSelectorCoherenceProbeState::release_source();
    }
  }

  void release() {
    ChannelSelectorCoherenceProbeState::release_source();
    released_ = true;
  }

 private:
  bool released_{};
};

//   source(predicate_stale) --[depth=2]--> sink
//   source(compute_stale)   --[depth=2]--> sink
//   source(driver)          --[depth=2]--> sink
//   Route selects OnEach(driver) or OnPredicate; max_age param controls both stale windows.
[[nodiscard]] ExecutionPlan make_channel_selector_coherence_plan(
    ChannelSelectorCoherenceRoute route) {
  Graph graph{"channel-selector-generation-coherence"};
  const auto source = graph.op<ChannelSelectorCoherenceSourceOperator>("source");
  const auto sink = graph.op<ChannelSelectorCoherenceSinkOperator>("sink", route);
  graph.add_flow(
      source->predicate_stale, sink->predicate_stale, ConnectionOptions{.queue_depth = 2U});
  graph.add_flow(source->compute_stale, sink->compute_stale, ConnectionOptions{.queue_depth = 2U});
  graph.add_flow(source->driver, sink->driver, ConnectionOptions{.queue_depth = 2U});
  const ClockKey clock = graph.add_clock<RealtimeClock>("runtime-test-clock");
  graph.set_default_clock(clock);

  CompileOptions options;
  options.profile = Profile::kStandard;
  options.parameters.runtime_updates = ParamRuntimeUpdatePolicy::kEnabledIfAdmitted;
  return compile(graph, std::move(options));
}

//   source --[depth=4]--> sink
//   unrelated (30 s period, no connections -- proves restart does not propagate forward-start)
//   allow_self_restart(source); lifecycle closure covers source alone (closure size=1).
[[nodiscard]] ExecutionPlan make_channel_restart_plan() {
  Graph graph{"channel-lifecycle-restart"};
  const auto source = graph.op<ChannelRestartSourceOperator>("source");
  const auto sink = graph.op<ChannelRestartSinkOperator>("sink");
  const auto unrelated = graph.op<ChannelRestartUnrelatedOperator>("unrelated");
  graph.add_flow(source->output, sink->input, ConnectionOptions{.queue_depth = 4U});
  graph.partition("source-sink").add(source).add(sink);
  graph.partition("unrelated").add(unrelated);
  graph.lifecycle().allow_self_restart(source);
  const ClockKey clock = graph.add_clock<RealtimeClock>("runtime-test-clock");
  graph.set_default_clock(clock);

  CompileOptions options;
  options.lifecycle_transactions = LifecycleTransactionOptions{
      .max_pending_requests = 4U,
      .max_pending_transactions = 4U,
      .max_receipts = 8U,
      .max_fenced_operators = 8U,
      .max_candidate_generations = 8U,
      .max_events = 64U,
      .max_audit_records = 16U,
      .max_quiesce_records = 16U,
      .max_rollback_records = 32U,
  };
  return compile(graph, std::move(options));
}

//   fast-source.output        --[kDropOldest, depth=1]--> slow-sampler.input
//              .holo-fallback --[kBlocking, depth=2]  -->             .holo-fallback
// The source self-restarts once. Running the immutable plan twice proves that Channel generations,
// temporal retention, and readiness credits are rebuilt rather than inherited by a fresh session.
// The unpublished mixed-policy sibling preserves both service records while the test compiler
// explicitly selects the HoloMQ fallback path.
[[nodiscard]] ExecutionPlan make_channel_credit_retirement_plan() {
  Graph graph{"channel-credit-retirement"};
  const auto source = graph.op<ChannelCreditRetirementSourceOperator>("fast-source");
  const auto sink = graph.op<ChannelCreditRetirementSinkOperator>("slow-sampler");
  graph.partition("credit-retirement-source").add(source);
  graph.partition("credit-retirement-sink").add(sink);
  graph.add_flow(source->output,
                 sink->input,
                 ConnectionOptions{.queue_policy = QueuePolicy::kDropOldest, .queue_depth = 1U});
  graph.add_flow(source->holo_fallback,
                 sink->holo_fallback,
                 ConnectionOptions{.queue_policy = QueuePolicy::kBlocking, .queue_depth = 2U});
  graph.lifecycle().allow_self_restart(source);
  const ClockKey clock = graph.add_clock<RealtimeClock>("runtime-test-clock");
  graph.set_default_clock(clock);

  CompileOptions options;
  options.lifecycle_transactions = LifecycleTransactionOptions{
      .max_pending_requests = 4U,
      .max_pending_transactions = 4U,
      .max_receipts = 8U,
      .max_fenced_operators = 8U,
      .max_candidate_generations = 8U,
      .max_events = 64U,
      .max_audit_records = 16U,
      .max_quiesce_records = 16U,
      .max_rollback_records = 32U,
  };
  return detail::compile_with_process_local_holomq(graph, std::move(options));
}

//   one-shot source --[kDropOldest, depth=1]--> independently notified sink
// Arrival advances LatestWithin retention but cannot activate the sink. Two explicit notification
// witnesses select the retained sample first as fresh and then through ReuseLast.
[[nodiscard]] ExecutionPlan make_notified_side_input_plan() {
  Graph graph{"notified-channel-side-input"};
  const auto source = graph.op<NotifiedSideInputSourceOperator>("source");
  const auto sink = graph.op<NotifiedSideInputSinkOperator>("notified-sink");
  graph.partition("notified-side-input-source").add(source);
  graph.partition("notified-side-input-worker").add(sink);
  graph.add_flow(source->output,
                 sink->input,
                 ConnectionOptions{.queue_policy = QueuePolicy::kDropOldest, .queue_depth = 1U});
  const ClockKey clock = graph.add_clock<RealtimeClock>("runtime-test-clock");
  graph.set_default_clock(clock);
  CompileOptions options;
  options.deployment.bind_thread_placement(
      "notified-side-input-worker", ThreadPlacement{.wake_mode = WakeMode::kSleepingWaitset});
  return compile(graph, std::move(options));
}

[[nodiscard]] ExecutionPlan make_on_any_input_parity_plan(bool explicit_contract) {
  Graph graph{explicit_contract ? "explicit-on-any-input" : "implicit-on-any-input"};
  const auto source = graph.op<OnAnyInputParitySourceOperator>("source");
  const auto sink = graph.op<OnAnyInputParitySinkOperator>("sink", explicit_contract);
  graph.add_flow(source->output, sink->input, ConnectionOptions{.queue_depth = 2U});
  const ClockKey clock = graph.add_clock<RealtimeClock>("runtime-test-clock");
  graph.set_default_clock(clock);
  return compile(graph);
}

[[nodiscard]] ExecutionPlan make_clock_input_plan(Profile profile, ClockInputTriggerRoute route,
                                                  ClockInputPlanOptions requested = {}) {
  Graph graph{route == ClockInputTriggerRoute::kOnPredicate ? "clock-predicate-input"
                                                            : "clock-input"};
  const auto source = graph.op<ClockPredicateInputSourceOperator>(
      "source", requested.block_source_after_first_emit);
  const auto sink =
      graph.op<ClockPredicateInputSinkOperator>("sink",
                                                route,
                                                requested.restart_on_first_sink_compute,
                                                requested.deadline,
                                                requested.violation);
  graph.partition("clock-input-source").add(source);
  graph.partition("clock-input-sink").add(sink);
  graph.add_flow(source->output,
                 sink->input,
                 ConnectionOptions{.queue_policy = QueuePolicy::kDropOldest, .queue_depth = 1U});
  if (requested.restart_on_first_sink_compute) {
    graph.lifecycle().allow_self_restart(sink);
  }
  const ClockKey clock = graph.add_clock<RealtimeClock>("runtime-test-clock");
  graph.set_default_clock(clock);
  CompileOptions options;
  options.profile = profile;
  options.lifecycle_transactions = LifecycleTransactionOptions{
      .max_pending_requests = 4U,
      .max_pending_transactions = 4U,
      .max_receipts = 8U,
      .max_fenced_operators = 8U,
      .max_candidate_generations = 8U,
      .max_events = 64U,
      .max_audit_records = 16U,
      .max_quiesce_records = 16U,
      .max_rollback_records = 32U,
  };
  return compile(graph, std::move(options));
}

void expect_clock_input_plan_replays(const ExecutionPlan& plan, bool expect_predicate) {
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.input_selections().size(), 1U);
  ASSERT_EQ(plan.max_start_delays().size(), 1U);
  ASSERT_EQ(plan.clock_triggers().size(), 2U);
  EXPECT_EQ(plan.predicate_triggers().size(), expect_predicate ? 1U : 0U);

  for (std::size_t replay = 0U; replay < 2U; ++replay) {
    SCOPED_TRACE(replay);
    ClockPredicateInputProbeState::reset();
    RunSession session = run_async(plan);
    const bool observed = ClockPredicateInputProbeState::wait_for_restart_execution();
    session.request_stop();
    session.wait();

    EXPECT_TRUE(observed)
        << "source="
        << ClockPredicateInputProbeState::source_compute_count.load(std::memory_order_acquire)
        << " sink="
        << ClockPredicateInputProbeState::sink_compute_count.load(std::memory_order_acquire);
    EXPECT_EQ(session.termination(), RunTermination::kStopped);
    EXPECT_TRUE(session.diagnostics().empty())
        << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
    EXPECT_TRUE(ClockPredicateInputProbeState::restart_accepted.load(std::memory_order_acquire));
    EXPECT_EQ(ClockPredicateInputProbeState::restart_error.load(std::memory_order_relaxed),
              ErrorCode::kSuccess);
    EXPECT_EQ(
        ClockPredicateInputProbeState::sink_forward_start_count.load(std::memory_order_acquire),
        1U);
    EXPECT_EQ(ClockPredicateInputProbeState::predicate_evaluation_count.load(
                  std::memory_order_acquire) != 0U,
              expect_predicate);
    EXPECT_GE(ClockPredicateInputProbeState::sink_compute_count.load(std::memory_order_acquire),
              2U);
    EXPECT_NE(ClockPredicateInputProbeState::last_value.load(std::memory_order_relaxed), 0U);
  }
}

//   source --[depth=4]--> sink
//   feeder --[depth=4]--> member (data input)
//   feeder --[depth=1]--> member (driver, OnEach trigger)
//   Lifecycle: before(source, member, kConfigure); closure covers source + member (size=2).
[[nodiscard]] ExecutionPlan make_multi_member_channel_restart_plan() {
  Graph graph{"multi-member-channel-lifecycle-restart"};
  const auto source = graph.op<ChannelRestartSourceOperator>("source");
  const auto sink = graph.op<ChannelRestartSinkOperator>("sink");
  const auto feeder = graph.op<ChannelRestartMemberFeederOperator>("member-feeder");
  const auto member = graph.op<ChannelRestartMemberOperator>("dependency-member");
  const auto unrelated = graph.op<ChannelRestartUnrelatedOperator>("unrelated");
  graph.add_flow(source->output, sink->input, ConnectionOptions{.queue_depth = 4U});
  graph.add_flow(feeder->output, member->input, ConnectionOptions{.queue_depth = 4U});
  graph.add_flow(feeder->driver, member->driver, ConnectionOptions{.queue_depth = 1U});
  graph.partition("source-sink").add(source).add(sink);
  graph.partition("member-flow").add(feeder).add(member);
  graph.partition("unrelated").add(unrelated);
  graph.lifecycle().before(source, member, LifecycleStage::kConfigure).allow_self_restart(source);
  const ClockKey clock = graph.add_clock<RealtimeClock>("runtime-test-clock");
  graph.set_default_clock(clock);

  CompileOptions options;
  options.lifecycle_transactions = LifecycleTransactionOptions{
      .max_pending_requests = 4U,
      .max_pending_transactions = 4U,
      .max_receipts = 8U,
      .max_fenced_operators = 8U,
      .max_candidate_generations = 8U,
      .max_events = 64U,
      .max_audit_records = 16U,
      .max_quiesce_records = 16U,
      .max_rollback_records = 32U,
  };
  return compile(graph, std::move(options));
}

//   source(anchor)   --[depth=8]  --> sink(anchor)   [Exact; OnWindow trigger]
//   source(windowed) --[depth=16] --> sink(windowed)  [BracketedWindow: lb=1, la=1, max=5]
//   max_pending_windows=2; max_wait=30 s.
[[nodiscard]] ExecutionPlan make_channel_window_restart_plan() {
  Graph graph{"channel-window-lifecycle-restart"};
  const auto source = graph.op<ChannelWindowRestartSourceOperator>("source");
  const auto sink = graph.op<ChannelWindowRestartSinkOperator>("sink");
  graph.add_flow(source->anchor, sink->anchor, ConnectionOptions{.queue_depth = 8U});
  graph.add_flow(source->windowed, sink->windowed, ConnectionOptions{.queue_depth = 16U});
  graph.lifecycle().allow_self_restart(source);
  const ClockKey clock = graph.add_clock<RealtimeClock>("runtime-test-clock");
  graph.set_default_clock(clock);

  CompileOptions options;
  options.lifecycle_transactions = LifecycleTransactionOptions{
      .max_pending_requests = 4U,
      .max_pending_transactions = 4U,
      .max_receipts = 8U,
      .max_fenced_operators = 8U,
      .max_candidate_generations = 8U,
      .max_events = 64U,
      .max_audit_records = 16U,
      .max_quiesce_records = 16U,
      .max_rollback_records = 32U,
  };
  return compile(graph, std::move(options));
}

// REQ: RUNTIME-008
// REQ: RUNTIME-016
// REQ: RUNTIME-018
// REQ: RUNTIME-034
// REQ: RUNTIME-020
// REQ: RUNTIME-024
// REQ: RUNTIME-052
TEST(ChannelLifecycleRestart, PublishedRestartDiscardsOldIngressAndRearmsAfterStart) {
  ChannelRestartProbeState::reset(false);
  const ExecutionPlan plan = make_channel_restart_plan();
  test::partition_parity::expect_accepted_cover(plan);
  ASSERT_TRUE(plan.parameter_derived_terms().empty());
  const auto latest = std::find_if(
      plan.input_selections().begin(), plan.input_selections().end(), [](const auto& selection) {
        return selection.selector == InputSelectorKind::kLatestWithin;
      });
  ASSERT_NE(latest, plan.input_selections().end());
  EXPECT_EQ(latest->max_age_ns, std::chrono::duration_cast<std::chrono::nanoseconds>(1s).count());
  const auto source_closure =
      std::find_if(plan.lifecycle_restart_closures().begin(),
                   plan.lifecycle_restart_closures().end(),
                   [](const auto& closure) { return closure.target.id == 1U; });
  ASSERT_NE(source_closure, plan.lifecycle_restart_closures().end());
  ASSERT_EQ(source_closure->closure.size(), 1U);
  ChannelRestartProbeState::reset(false);

  const test::partition_parity::ScopedEnvironmentMutation changed_after_compile{
      "definitely-not-an-algorithm"};
  // Key setup: source fills queue depth=4 on activation 0 then requests restart; restart succeeds.
  RunSession session = run_async(plan);
  const bool observed = ChannelRestartProbeState::wait_for_sink(1U);
  session.request_stop();
  session.wait();

  EXPECT_TRUE(observed);
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token) << " vertex="
      << (session.diagnostics().empty() ? 0U : session.diagnostics().front().source.vertex.value);
  EXPECT_TRUE(ChannelRestartProbeState::restart_accepted.load(std::memory_order_acquire));
  EXPECT_EQ(ChannelRestartProbeState::restart_error.load(std::memory_order_relaxed),
            ErrorCode::kSuccess);
  // The periodic source may be scheduled again while the stop request races with sink
  // observation. Only the first two activations publish; lifecycle-stage counts below prove
  // that this scheduler progress is not a redundant restart.
  EXPECT_GE(ChannelRestartProbeState::source_compute_count.load(), 2U);
  EXPECT_EQ(ChannelRestartProbeState::source_forward_start_count.load(), 1U);
  EXPECT_EQ(ChannelRestartProbeState::source_restore_start_count.load(), 0U);
  EXPECT_EQ(ChannelRestartProbeState::sink_start_count.load(), 1U);
  EXPECT_EQ(ChannelRestartProbeState::unrelated_start_count.load(), 1U);
  EXPECT_EQ(ChannelRestartProbeState::unrelated_forward_start_count.load(), 0U);
  // The first activation filled the complete compiler-frozen queue depth. Published restart
  // discards that bounded old-generation prefix; only the post-restart value is observable.
  // Key assertion: old-gen prefix [1-4] purged by restart; only post-restart value 5 arrives.
  EXPECT_EQ(ChannelRestartProbeState::values(), (std::vector<std::uint32_t>{5U}));
  EXPECT_EQ(ChannelRestartProbeState::sequences(), (std::vector<std::uint64_t>{5U}));
  EXPECT_EQ(
      ChannelRestartProbeState::observed_source_tags(),
      (std::vector<Tag>{{.sequence = 1U, .generation = 1U}, {.sequence = 1U, .generation = 2U}}));
  EXPECT_EQ(ChannelRestartProbeState::observed_sink_tags(),
            (std::vector<Tag>{{.sequence = 1U, .generation = 2U}}));
  EXPECT_EQ(ChannelRestartProbeState::sink_activation_reasons,
            (std::vector<ActivationReason>{ActivationReason::kNominal}));
  EXPECT_EQ(ChannelRestartProbeState::sink_activation_policies,
            (std::vector<OnViolation>{OnViolation::kEmitWitness}));
}

// REQ: CHAN-102
// REQ: RUNTIME-018
// REQ: RUNTIME-024
// REQ: CHAN-131
TEST(ChannelLifecycleRestart,
     FastDropOldestSideInputRetiresReadinessCreditAcrossRestartAndFreshRuns) {
  ChannelCreditRetirementProbeState::reset();
  const ExecutionPlan plan = make_channel_credit_retirement_plan();
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.channel_locality().size(), 2U);
  for (const auto& locality : plan.channel_locality()) {
    EXPECT_EQ(locality.accepted, ChannelLocalityKind::kProcessLocal);
  }
  const auto state = detail::ExecutionPlanAccess::retain(plan);
  ASSERT_NE(state, nullptr);
  ASSERT_EQ(state->process_local_channel_services.size(), 2U);
  for (const auto& service : state->process_local_channel_services) {
    EXPECT_EQ(service.delivery_kind, detail::ProcessLocalChannelDeliveryKind::kHoloMQ);
  }

  for (std::size_t run_index = 0U; run_index < 2U; ++run_index) {
    SCOPED_TRACE(run_index);
    ChannelCreditRetirementProbeState::reset();
    RunSession session = run_async(plan);
    const bool progressed = ChannelCreditRetirementProbeState::wait_for_progress();
    session.request_stop();
    session.wait();

    EXPECT_TRUE(progressed)
        << "source="
        << ChannelCreditRetirementProbeState::source_compute_count.load(std::memory_order_acquire)
        << " sink="
        << ChannelCreditRetirementProbeState::sink_compute_count.load(std::memory_order_acquire)
        << " restart="
        << ChannelCreditRetirementProbeState::restart_accepted.load(std::memory_order_acquire);
    EXPECT_EQ(session.termination(), RunTermination::kStopped);
    EXPECT_TRUE(session.diagnostics().empty());
    EXPECT_TRUE(
        ChannelCreditRetirementProbeState::restart_accepted.load(std::memory_order_acquire));
    EXPECT_EQ(ChannelCreditRetirementProbeState::restart_error.load(std::memory_order_relaxed),
              ErrorCode::kSuccess);
    EXPECT_GE(
        ChannelCreditRetirementProbeState::source_compute_count.load(std::memory_order_acquire),
        ChannelCreditRetirementProbeState::kRequiredPublications);

    const auto values = ChannelCreditRetirementProbeState::values();
    const auto sequences = ChannelCreditRetirementProbeState::sequences();
    ASSERT_GE(values.size(), ChannelCreditRetirementProbeState::kRequiredSelections);
    ASSERT_EQ(sequences.size(), values.size());
    EXPECT_TRUE(std::is_sorted(values.begin(), values.end()));
    EXPECT_GT(values.back(), ChannelCreditRetirementProbeState::kRestartAfter);
    EXPECT_TRUE(std::all_of(
        sequences.begin(), sequences.end(), [](std::uint64_t sequence) { return sequence != 0U; }));
  }
}

// REQ: CHAN-102
// REQ: RUNTIME-025
// A connected arrival populates temporal retention for an OnNotified Operator but never owns
// activation authority. The first notification selects LatestWithin; the second observes the same
// committed sample through ReuseLast after the arrival credit has already retired.
TEST(ChannelLifecycleRestart, NotifiedSideInputWaitsForIndependentReadinessBeforeActivation) {
  NotifiedSideInputProbeState::reset();
  const ExecutionPlan plan = make_notified_side_input_plan();
  ASSERT_TRUE(plan.ok()) << plan.json();

  RunSession session = run_async(plan);
  const bool sender_ready = NotifiedSideInputProbeState::wait_until_sender();
  if (!sender_ready) {
    session.request_stop();
    session.wait();
  }
  ASSERT_TRUE(sender_ready)
      << "termination=" << static_cast<int>(session.termination()) << " diagnostic="
      << (session.diagnostics().empty() ? "<none>" : session.diagnostics().front().token)
      << " vertex="
      << (session.diagnostics().empty() ? 0U : session.diagnostics().front().source.vertex.value)
      << " service="
      << (session.diagnostics().empty() ? "<none>" : session.diagnostics().front().service);
  ASSERT_TRUE(NotifiedSideInputProbeState::wait_until_source());
  // Give the Channel pump ample opportunity to ingest the one-shot publication. Arrival is a
  // side-input witness and therefore must not activate the independently notified Operator.
  std::this_thread::sleep_for(50ms);
  EXPECT_EQ(NotifiedSideInputProbeState::sink_compute_count.load(std::memory_order_acquire), 0U);

  const NotificationSender sender = NotifiedSideInputProbeState::sender_copy();
  auto first_post = sender.post();
  const auto start_deadline = std::chrono::steady_clock::now() + 5s;
  while (!first_post && first_post.error().code == ErrorCode::kNotReady &&
         std::chrono::steady_clock::now() < start_deadline) {
    std::this_thread::yield();
    first_post = sender.post();
  }
  ASSERT_TRUE(first_post);
  ASSERT_TRUE(NotifiedSideInputProbeState::wait_until_sink_count(1U));

  auto second_post = sender.post();
  const auto retirement_deadline = std::chrono::steady_clock::now() + 5s;
  while (!second_post && second_post.error().code == ErrorCode::kBackpressured &&
         std::chrono::steady_clock::now() < retirement_deadline) {
    std::this_thread::yield();
    second_post = sender.post();
  }
  ASSERT_TRUE(second_post);
  ASSERT_TRUE(NotifiedSideInputProbeState::wait_until_sink_count(2U));

  session.request_stop();
  session.wait();
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
  EXPECT_EQ(NotifiedSideInputProbeState::values, (std::array<std::uint32_t, 2U>{42U, 42U}));
  EXPECT_NE(NotifiedSideInputProbeState::sequences[0], 0U);
  EXPECT_EQ(NotifiedSideInputProbeState::sequences[1], NotifiedSideInputProbeState::sequences[0]);
}

// REQ: CONTRACT-001
// REQ: RUNTIME-018
TEST(ChannelLifecycleRestart, ImplicitAndExplicitOnAnyInputPinArrivalWitnessIdentically) {
  for (const bool explicit_contract : {false, true}) {
    SCOPED_TRACE(explicit_contract ? "explicit" : "implicit");
    OnAnyInputParityProbeState::reset();
    const ExecutionPlan plan = make_on_any_input_parity_plan(explicit_contract);
    ASSERT_TRUE(plan.ok()) << plan.json();
    EXPECT_TRUE(plan.input_selections().empty()) << plan.json();

    RunSession session = run_async(plan);
    const bool observed = OnAnyInputParityProbeState::wait_for_sink();
    session.request_stop();
    session.wait();

    EXPECT_TRUE(observed);
    EXPECT_EQ(session.termination(), RunTermination::kStopped);
    EXPECT_TRUE(session.diagnostics().empty())
        << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
    EXPECT_EQ(OnAnyInputParityProbeState::sink_compute_count.load(std::memory_order_acquire), 1U);
    EXPECT_EQ(OnAnyInputParityProbeState::sink_value.load(std::memory_order_relaxed), 42U);
  }
}

// Clock-triggered input selection must have the same accepted execution surface as an
// input-triggered Operator. Run both admitted local profiles and execute each immutable plan in
// two fresh sessions. Each session crosses one successful restart boundary so source and selector
// reconstruction, deadline binding, and cooperative stop are exercised together.
// REQ: RUNTIME-004
// REQ: RUNTIME-015
// REQ: RUNTIME-016
// REQ: RUNTIME-018
TEST(ChannelLifecycleRestart, OnClockWithInputExecutesAndRestartsAcrossLocalProfiles) {
  for (const Profile profile : {Profile::kStandard, Profile::kResearch}) {
    SCOPED_TRACE(static_cast<unsigned>(profile));
    const ExecutionPlan plan = make_clock_input_plan(profile, ClockInputTriggerRoute::kOnClock);
    expect_clock_input_plan_replays(plan, false);
  }
}

// REQ: RUNTIME-004
// REQ: RUNTIME-015
// REQ: RUNTIME-016
// REQ: RUNTIME-018
TEST(ChannelLifecycleRestart, ClockPredicateWithInputExecutesAndRestartsAcrossLocalProfiles) {
  for (const Profile profile : {Profile::kStandard, Profile::kResearch}) {
    SCOPED_TRACE(static_cast<unsigned>(profile));
    const ExecutionPlan plan = make_clock_input_plan(profile, ClockInputTriggerRoute::kOnPredicate);
    expect_clock_input_plan_replays(plan, true);
  }
}

// Stop is requested while the source callback still owns a successfully emitted sample. The
// post-compute activation cut makes that sample visible after the callback returns, but the stop
// fence must cancel the already-frozen clock candidate before input selection, predicate
// evaluation, deadline evidence, or sink callback entry.
// REQ: RUNTIME-004
// REQ: RUNTIME-015
// REQ: RUNTIME-016
// REQ: RUNTIME-028
// REQ: RUNTIME-081
TEST(ChannelLifecycleRestart, ClockInputStopFenceSuppressesSelectionPredicateAndMaxStartDelayGate) {
  // This bounds test-only worker coordination under a fully loaded parallel CI executor.
  constexpr auto kSynchronizationTimeout = 30s;
  for (const ClockInputTriggerRoute route :
       {ClockInputTriggerRoute::kOnClock, ClockInputTriggerRoute::kOnPredicate}) {
    SCOPED_TRACE(static_cast<unsigned>(route));
    ClockPredicateInputProbeState::reset();
    const ExecutionPlan base = make_clock_input_plan(Profile::kStandard,
                                                     route,
                                                     {
                                                         .block_source_after_first_emit = true,
                                                         .restart_on_first_sink_compute = false,
                                                         .deadline = 1ns,
                                                         .violation = OnViolation::kEmitWitness,
                                                     });
    ASSERT_TRUE(base.ok()) << base.json();
    ASSERT_EQ(base.max_start_delays().size(), 1U);
    auto invocation_control = std::make_shared<detail::RuntimePreDispatchTestControl>(
        base.max_start_delays().front().owner.id);
    const auto phase_counters = std::make_shared<detail::RuntimePhaseCounters>();
    const ExecutionPlan counted =
        detail::RuntimeTestPlanAccess::with_phase_counters(base, phase_counters);
    const ExecutionPlan plan = detail::RuntimeTestPlanAccess::enable_pre_dispatch_interleaving(
        counted, invocation_control);
    ASSERT_TRUE(plan.ok()) << plan.json();

    RunSession session = run_async(plan);
    // On assertion unwind, unblock both workers before RunSession joins them.
    const ClockInputPreDispatchReleaseGuard invocation_release{invocation_control};
    const ClockInputSourceReleaseGuard release_guard;
    ASSERT_TRUE(ClockPredicateInputProbeState::wait_for_source_block(kSynchronizationTimeout));
    ASSERT_TRUE(invocation_control->wait_until_paused(kSynchronizationTimeout));
    const auto phase_count = [&](detail::RuntimePhaseCounter phase) {
      return phase_counters->snapshot()[detail::RuntimePhaseCounters::index(phase)];
    };
    const std::uint64_t selected_before_stop =
        phase_count(detail::RuntimePhaseCounter::kCandidatesSelected);
    session.request_stop();
    ClockPredicateInputProbeState::release_source_block();
    // Keep the sink frozen until source retirement flushes the emitted sample's Channel cut.
    const auto activation_cut_deadline = std::chrono::steady_clock::now() + kSynchronizationTimeout;
    const auto completed_activation_count = [&] {
      return phase_count(detail::RuntimePhaseCounter::kActivationsCompleted);
    };
    while (completed_activation_count() == 0U &&
           std::chrono::steady_clock::now() < activation_cut_deadline) {
      std::this_thread::sleep_for(1ms);
    }
    ASSERT_GE(completed_activation_count(), 1U);
    invocation_control->release();
    session.wait();

    EXPECT_EQ(session.termination(), RunTermination::kStopped);
    EXPECT_GE(ClockPredicateInputProbeState::source_compute_count.load(std::memory_order_acquire),
              1U);
    EXPECT_EQ(ClockPredicateInputProbeState::sink_compute_count.load(std::memory_order_acquire),
              0U);
    EXPECT_EQ(phase_count(detail::RuntimePhaseCounter::kCandidatesSelected), selected_before_stop);
    EXPECT_EQ(
        ClockPredicateInputProbeState::predicate_evaluation_count.load(std::memory_order_acquire),
        0U);
    EXPECT_TRUE(std::none_of(
        session.diagnostics().begin(),
        session.diagnostics().end(),
        [](const Diagnostic& diagnostic) { return diagnostic.token == "MAX_START_DELAY_MISS"; }));
  }
}

// REQ: RUNTIME-008
// REQ: RUNTIME-016
// REQ: RUNTIME-020
// REQ: RUNTIME-024
TEST(ChannelLifecycleRestart, RolledBackRestartPreservesOldIngressAndPublishedGeneration) {
  ChannelRestartProbeState::reset(true);
  const ExecutionPlan plan = make_channel_restart_plan();
  test::partition_parity::expect_accepted_cover(plan);
  ChannelRestartProbeState::reset(true);

  const test::partition_parity::ScopedEnvironmentMutation changed_after_compile{
      "definitely-not-an-algorithm"};
  // Key setup: fail_restart_start=true causes forward-start kFatalFailure; rollback triggered.
  RunSession session = run_async(plan);
  const bool observed = ChannelRestartProbeState::wait_for_sink(5U);
  session.request_stop();
  session.wait();

  EXPECT_TRUE(observed);
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  ASSERT_EQ(session.diagnostics().size(), 1U);
  EXPECT_EQ(session.diagnostics().front().severity, DiagnosticSeverity::kWarning);
  const auto& failure_detail = session.diagnostics().front().detail;
  if (!failure_detail.has_value()) {
    FAIL() << "Expected diagnostic detail";
  }
  EXPECT_EQ(failure_detail->stage, LifecycleStage::kStart);
  EXPECT_EQ(failure_detail->phase, LifecycleTransitionPhase::kRestartForward);
  EXPECT_NE(failure_detail->transaction_id, 0U);
  EXPECT_TRUE(ChannelRestartProbeState::restart_accepted.load(std::memory_order_acquire));
  EXPECT_EQ(ChannelRestartProbeState::restart_error.load(std::memory_order_relaxed),
            ErrorCode::kSuccess);
  // The periodic source may be scheduled again while the stop request races with sink
  // observation. Only the first two activations publish; lifecycle-stage counts below prove
  // that this scheduler progress is not a redundant restart.
  EXPECT_GE(ChannelRestartProbeState::source_compute_count.load(), 2U);
  EXPECT_EQ(ChannelRestartProbeState::source_forward_start_count.load(), 1U);
  EXPECT_EQ(ChannelRestartProbeState::source_restore_start_count.load(), 1U);
  EXPECT_EQ(ChannelRestartProbeState::sink_start_count.load(), 1U);
  EXPECT_EQ(ChannelRestartProbeState::unrelated_start_count.load(), 1U);
  // Key assertion: rollback restores the original queue; all 5 samples [1-5] reach the sink.
  EXPECT_EQ(ChannelRestartProbeState::values(), (std::vector<std::uint32_t>{1U, 2U, 3U, 4U, 5U}));
  EXPECT_EQ(ChannelRestartProbeState::sequences(),
            (std::vector<std::uint64_t>{1U, 2U, 3U, 4U, 5U}));
  EXPECT_EQ(ChannelRestartProbeState::sink_activation_reasons,
            (std::vector<ActivationReason>{ActivationReason::kNominal,
                                           ActivationReason::kNominal,
                                           ActivationReason::kNominal,
                                           ActivationReason::kNominal,
                                           ActivationReason::kNominal}));
  EXPECT_EQ(ChannelRestartProbeState::sink_activation_policies,
            (std::vector<OnViolation>{OnViolation::kEmitWitness,
                                      OnViolation::kEmitWitness,
                                      OnViolation::kEmitWitness,
                                      OnViolation::kEmitWitness,
                                      OnViolation::kEmitWitness}));
}

// A non-target closure member owns an input from an independent producer. Its old prefix can be
// purged only if RunSession carries the real compiler closure into the Channel restart scope.
// REQ: RUNTIME-008
// REQ: RUNTIME-020
// REQ: RUNTIME-024
TEST(ChannelLifecycleRestart, MultiMemberClosureFencesNonTargetChannelInput) {
  ChannelRestartProbeState::reset(false, true);
  const ExecutionPlan plan = make_multi_member_channel_restart_plan();
  test::partition_parity::expect_accepted_cover(plan);
  const auto source_closure =
      std::find_if(plan.lifecycle_restart_closures().begin(),
                   plan.lifecycle_restart_closures().end(),
                   [](const auto& closure) { return closure.closure.size() == 2U; });
  ASSERT_NE(source_closure, plan.lifecycle_restart_closures().end());
  EXPECT_EQ(std::count_if(plan.lifecycle_restart_closures().begin(),
                          plan.lifecycle_restart_closures().end(),
                          [](const auto& closure) { return closure.closure.size() == 2U; }),
            1U);
  EXPECT_TRUE(
      std::any_of(source_closure->closure.begin(),
                  source_closure->closure.end(),
                  [&](const OperatorRef member) { return member == source_closure->target; }));
  EXPECT_TRUE(
      std::any_of(source_closure->closure.begin(),
                  source_closure->closure.end(),
                  [&](const OperatorRef member) { return member != source_closure->target; }));
  ChannelRestartProbeState::reset(false, true);

  // Key setup: two-member closure (source + member) fences the feeder->member input on restart.
  RunSession session = run_async(plan);
  const bool observed = ChannelRestartProbeState::wait_for_member_batch();
  session.request_stop();
  session.wait();

  EXPECT_TRUE(observed);
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
  EXPECT_TRUE(ChannelRestartProbeState::restart_accepted.load(std::memory_order_acquire));
  EXPECT_EQ(ChannelRestartProbeState::source_forward_start_count.load(), 1U);
  EXPECT_EQ(ChannelRestartProbeState::member_forward_start_count.load(), 1U);
  EXPECT_EQ(ChannelRestartProbeState::member_start_count.load(), 2U);
  EXPECT_TRUE(ChannelRestartProbeState::feeder_post_published.load(std::memory_order_acquire));
  // Key assertion: member sees only post-restart value 5; old prefix [1-4] was discarded.
  EXPECT_EQ(ChannelRestartProbeState::observed_member_values(), (std::vector<std::uint32_t>{5U}));
}

// The first source activation leaves an incomplete window in the downstream selector. Restart must
// discard that state so the first closed result contains only the replacement generation.
// REQ: RUNTIME-008
// REQ: RUNTIME-020
// REQ: RUNTIME-024
// REQ: RUNTIME-052
TEST(ChannelLifecycleRestart, PendingWindowReprimesAcrossPublishedRestart) {
  ChannelWindowRestartProbeState::reset();
  const ExecutionPlan plan = make_channel_window_restart_plan();
  test::partition_parity::expect_accepted_cover(plan);
  ChannelWindowRestartProbeState::reset();

  // Key setup: activation 0 leaves an incomplete window in the selector; restart must purge it.
  RunSession session = run_async(plan);
  const bool observed = ChannelWindowRestartProbeState::wait_for_batch();
  session.request_stop();
  session.wait();

  EXPECT_TRUE(observed);
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
  EXPECT_TRUE(ChannelWindowRestartProbeState::restart_accepted.load(std::memory_order_acquire));
  EXPECT_EQ(ChannelWindowRestartProbeState::source_forward_start_count.load(), 1U);
  // Key assertion: first closed result is entirely from the replacement generation (anchor=120).
  EXPECT_EQ(ChannelWindowRestartProbeState::observed_anchor(), 120U);
  EXPECT_EQ(ChannelWindowRestartProbeState::observed_values(),
            (std::vector<std::uint32_t>{105U, 110U, 115U, 120U, 125U}));
  EXPECT_EQ(ChannelWindowRestartProbeState::observed_sequences(),
            (std::vector<std::uint64_t>{3U, 4U, 5U, 6U, 7U}));
}
// REQ: RUNTIME-020
// REQ: RUNTIME-024
TEST(ChannelLifecycleRestart, NextComputePreservesChannelEpochAndPublishedSequence) {
  ChannelNextComputeProbeState::reset();
  const ExecutionPlan plan = make_channel_next_compute_plan();
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.parameter_fields().size(), 1U);
  const ParameterFieldPlanRecord field = plan.parameter_fields().front();

  // Key setup: source blocks inside activation 0; patch submitted at value=20 via NextCompute.
  RunSession session = run_async(plan);
  FirstComputeRelease release;
  ASSERT_TRUE(ChannelNextComputeProbeState::wait_for_first_compute());

  const auto run_owner = session.operator_ref(field.owner);
  ASSERT_TRUE(run_owner);
  const ParamAccessPolicy policy{
      .origin = ParamControlOrigin::kLocalApplication,
      .permissions = ParamPermission::kRead | ParamPermission::kUpdate | ParamPermission::kAwait,
  };
  auto handle_result = session.parameter_control(policy);
  ASSERT_TRUE(handle_result);
  ParamControlHandle handle = std::move(*handle_result);
  const std::array patch{
      ParamPatch{.target = ParamAddress{.operator_instance = *run_owner, .field = field.field},
                 .value = ParamValue::from(std::int32_t{20})},
  };
  const ParamBatchUpdateResult accepted =
      handle.submit(patch, ParamUpdateOptions{.effective = NextCompute{}});
  ASSERT_TRUE(accepted.committed);
  EXPECT_EQ(accepted.status, ParamUpdateStatus::kAccepted);

  release.release();
  const bool observed = ChannelNextComputeProbeState::wait_for_sink(2U);
  session.request_stop();
  session.wait();

  EXPECT_TRUE(observed);
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
  EXPECT_GE(ChannelNextComputeProbeState::source_compute_count.load(), 2U);
  // Key assertion: param takes effect on the next compute; sequences are contiguous [1, 2].
  EXPECT_EQ(ChannelNextComputeProbeState::sink_values, (std::vector<std::uint32_t>{10U, 20U}));
  EXPECT_EQ(ChannelNextComputeProbeState::sink_sequences, (std::vector<std::uint64_t>{1U, 2U}));
  const auto observed_update = handle.await(accepted.update_sequence, 0ns);
  ASSERT_TRUE(observed_update);
  EXPECT_EQ(observed_update->state, ParamUpdateState::kObserved);
  EXPECT_EQ(observed_update->target_generation, 1U);
}

// REQ: RUNTIME-007
// REQ: PARAM-017
// REQ: PARAM-018
// REQ: RUNTIME-007
// REQ: RUNTIME-018
TEST(ChannelLifecycleRestart, NextComputeLatestWithinUsesOnePinnedGenerationAcrossSelectorRoutes) {
  constexpr std::array kRoutes{ChannelSelectorCoherenceRoute::kOnEach,
                               ChannelSelectorCoherenceRoute::kPredicate};
  for (const ChannelSelectorCoherenceRoute route : kRoutes) {
    SCOPED_TRACE(route == ChannelSelectorCoherenceRoute::kOnEach ? "OnEach" : "OnPredicate");
    ChannelSelectorCoherenceProbeState::reset();
    const ExecutionPlan plan = make_channel_selector_coherence_plan(route);
    test::partition_parity::expect_accepted_cover(plan);
    ASSERT_EQ(plan.parameter_fields().size(), 1U);
    const ParameterFieldPlanRecord field = plan.parameter_fields().front();
    ASSERT_EQ(plan.parameter_derived_terms().size(), 2U);
    EXPECT_NE(plan.parameter_derived_terms()[0].input, plan.parameter_derived_terms()[1].input);
    for (const ParameterDerivedTermPlanRecord& term : plan.parameter_derived_terms()) {
      EXPECT_EQ(term.kind, ParameterDerivedTermKind::kLatestWithinMaxAge);
      EXPECT_EQ(term.owner, field.owner);
      EXPECT_TRUE(term.input);
      EXPECT_EQ(term.update_disposition, ParameterDerivedTermUpdateDisposition::kRuntimeAtomic);
      EXPECT_EQ(term.initial_value_ns,
                std::chrono::duration_cast<std::chrono::nanoseconds>(5ms).count());
    }

    const test::partition_parity::ScopedEnvironmentMutation changed_after_compile{
        "definitely-not-an-algorithm"};
    RunSession session = run_async(plan);
    SelectorSourceRelease release;
    // Key setup: max_age patched 5->20 ms while source is blocked; stale samples are ~10 ms old.
    const bool source_entered = ChannelSelectorCoherenceProbeState::wait_for_source_compute();
    if (!source_entered) {
      session.request_stop();
      session.wait();
    }
    ASSERT_TRUE(source_entered)
        << "termination=" << static_cast<std::uint32_t>(session.termination()) << " diagnostic="
        << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);

    const auto run_owner = session.operator_ref(field.owner);
    ASSERT_TRUE(run_owner);
    const ParamAccessPolicy policy{
        .origin = ParamControlOrigin::kLocalApplication,
        .permissions = ParamPermission::kRead | ParamPermission::kUpdate | ParamPermission::kAwait,
    };
    auto handle_result = session.parameter_control(policy);
    ASSERT_TRUE(handle_result);
    ParamControlHandle handle = std::move(*handle_result);
    const std::array patch{
        ParamPatch{.target = ParamAddress{.operator_instance = *run_owner, .field = field.field},
                   .value = ParamValue::from(20ms)},
    };
    const ParamBatchUpdateResult accepted =
        handle.submit(patch, ParamUpdateOptions{.effective = NextCompute{}});
    ASSERT_TRUE(accepted.committed);
    EXPECT_EQ(accepted.status, ParamUpdateStatus::kAccepted);

    release.release();
    const bool observed = ChannelSelectorCoherenceProbeState::wait_for_sink();
    session.request_stop();
    session.wait();

    EXPECT_TRUE(observed);
    EXPECT_EQ(session.termination(), RunTermination::kStopped);
    EXPECT_TRUE(session.diagnostics().empty())
        << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
    EXPECT_EQ(ChannelSelectorCoherenceProbeState::sink_compute_count, 1U);
    // Key assertion: both selector routes accept the stale samples at the updated max_age.
    EXPECT_EQ(ChannelSelectorCoherenceProbeState::predicate_selected_value, 98U);
    EXPECT_EQ(ChannelSelectorCoherenceProbeState::compute_selected_value, 99U);
    EXPECT_EQ(ChannelSelectorCoherenceProbeState::observed_max_age, 20ms);
    EXPECT_EQ(ChannelSelectorCoherenceProbeState::observed_generation, accepted.target_generation);
    const auto observed_update = handle.await(accepted.update_sequence, 0ns);
    ASSERT_TRUE(observed_update);
    EXPECT_EQ(observed_update->state, ParamUpdateState::kObserved);
    EXPECT_EQ(observed_update->target_generation, accepted.target_generation);
  }
}

// REQ: PARAM-017
// REQ: RUNTIME-007
// REQ: RUNTIME-018
TEST(ChannelLifecycleRestart, OrphanRuntimeLatestWithinTermFailsBeforeDispatch) {
  ChannelSelectorCoherenceProbeState::reset();
  const ExecutionPlan baseline =
      make_channel_selector_coherence_plan(ChannelSelectorCoherenceRoute::kPredicate);
  ASSERT_TRUE(baseline.ok()) << baseline.json();
  const auto exact = std::find_if(
      baseline.input_selections().begin(),
      baseline.input_selections().end(),
      [](const auto& selection) { return selection.selector == InputSelectorKind::kExact; });
  ASSERT_NE(exact, baseline.input_selections().end());

  const ExecutionPlan malformed =
      detail::RuntimeTestPlanAccess::append_orphan_latest_within_term_for_test(baseline,
                                                                               exact->input);
  ASSERT_TRUE(malformed.ok()) << malformed.json();
  // Key setup: LatestWithin term injected into a plan slot with no corresponding param binding.
  RunSession session = run_async(malformed);
  session.wait();

  // Key assertion: dispatcher fails before any compute; RUNTIME_PARAMETER_BINDING_UNAVAILABLE.
  EXPECT_EQ(session.termination(), RunTermination::kFailed);
  ASSERT_EQ(session.diagnostics().size(), 1U);
  EXPECT_EQ(session.diagnostics().front().token, "RUNTIME_PARAMETER_BINDING_UNAVAILABLE");
  EXPECT_EQ(ChannelSelectorCoherenceProbeState::sink_compute_count, 0U);
}

}  // namespace
}  // namespace holoscan
