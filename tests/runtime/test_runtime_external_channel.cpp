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
#include <string>
#include <thread>
#include <utility>

#include <holoscan/channel/channel.hpp>
#include <holoscan/core/compile.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/time/manual_clock.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "channel/endpoint_catalog_state.hpp"
#include "channel/process_identity.hpp"
#include "core/diagnostic_registry.hpp"
#include "runtime/runtime_phase_counters.hpp"
#include "runtime/runtime_support_thread_control.hpp"
#include "runtime/runtime_test_access.hpp"

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
        .produces_tensor(TensorPortLayout{
            .memory_kind = MemoryKind::kHost,
            .dtype = kExternalTensorDtype,
            .rank = 1U,
        })
        .tensor_allocation(TensorAllocationBounds{.max_rank = 1U, .max_byte_span = 8U})
        .payload_options(PayloadOptions{
            .interface_tensor_emit_policy = InterfaceTensorEmitPolicy::kCopyIfNeeded,
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

// Retains one imported Tensor through ReuseLast while a notification source remains the sole
// activation authority. The two callback observations must share the same Runtime-retained
// backing even after the publisher's ordered EOS has crossed a capacity-one cache.
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
  static inline std::array<const void*, 2U> data{};
  static inline std::array<std::uint64_t, 2U> sequences{};
  static inline std::array<std::uint8_t, 2U> values{};
};

class RetainedTensorEndOfStreamSink final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(input, "input")
        .queue_depth(1U)
        .expects_tensor(TensorPortLayout{
            .memory_kind = MemoryKind::kHost,
            .dtype = kExternalTensorDtype,
            .rank = 1U,
        });
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
  NotificationSource notification;
};

// Builds TopicOptions for an optional host Tensor export with fixed 8-byte single-axis
// bounds. Shared across optional-tensor tests to avoid repeating boilerplate.
[[nodiscard]] TopicOptions optional_tensor_export_options(std::string_view endpoint_name) {
  TopicOptions options;
  options.endpoint_name = endpoint_name;
  options.bounds.subscriber =
      SubscriberEndpointBounds{.queue_depth = 1U, .max_borrowed_samples = 1U};
  options.tensor = LogicalTensorRequirements{
      .memory_kind = MemoryKind::kHost,
      .dtype = kExternalTensorDtype,
      .rank = 1U,
      .logical_bounds =
          LogicalTensorBounds{
              .max_rank = 1U,
              .max_byte_span = 8U,
              .axes = kExternalTensorAxes,
          },
      .strides = TensorStridePolicy::kContiguousOnly,
      .foreign_emit_policy = InterfaceTensorEmitPolicy::kCopyIfNeeded,
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
[[nodiscard]] expected<void, Error> wait_for_subscriber(ChannelPublisher<T>& publisher,
                                                        std::chrono::milliseconds timeout = 5s) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    auto refreshed = publisher.update_connections();
    if (!refreshed) {
      return make_unexpected(std::move(refreshed).error());
    }
    if (publisher.has_subscribers()) {
      return {};
    }
    std::this_thread::sleep_for(1ms);
  }
  return make_unexpected(Error{ErrorCode::kNotReady});
}

[[nodiscard]] expected<void, Error> publish_external_host_tensor(
    ChannelPublisher<Tensor>& publisher, std::optional<TimePoint> capture_time) {
  auto loan = publisher.loan_tensor(
      TensorLoanRequest{.shape = kExternalTensorShape, .dtype = kExternalTensorDtype});
  if (!loan) {
    return make_unexpected(std::move(loan).error());
  }
  auto writer = loan->write_host();
  if (!writer) {
    return make_unexpected(std::move(writer).error());
  }
  std::fill_n(static_cast<std::byte*>(writer->data()), writer->byte_size(), std::byte{0x2a});
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

// Polls has_samples() until a publisher is seen on two consecutive checks, guarding against
// a transient first observation before the Runtime-owned publisher is fully live.
[[nodiscard]] expected<void, Error> wait_for_publisher(ChannelSubscriber<std::uint32_t>& subscriber,
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
  ASSERT_TRUE(PostEndOfStreamWindowProbe::sender_copy().post());
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
  ASSERT_TRUE(PostEndOfStreamWindowProbe::sender_copy().post());
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

// A capacity-one ReuseLast cache has no writable data slot after its first activation. Runtime
// must nevertheless stage and accept the next ordered EOS control record without consuming or
// dropping a data head, then preserve the retained Tensor for the next independent activation.
// REQ: CHAN-078
// REQ: RUNTIME-025
// REQ: SYSTEM-006
TEST(RuntimeExternalChannelTest, RawTensorEndOfStreamCrossesFullReuseLastCache) {
  RetainedTensorEndOfStreamProbe::reset();
  const std::string endpoint_name = unique_endpoint_name();

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

  const ExecutionPlan compiled = compile(graph);
  auto phase_counters = std::make_shared<detail::RuntimePhaseCounters>();
  const ExecutionPlan plan =
      detail::RuntimeTestPlanAccess::with_phase_counters(compiled, phase_counters);
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
  const auto wait_for_channel_readiness = [&](const std::uint64_t expected) {
    const auto deadline = std::chrono::steady_clock::now() + 5s;
    while (phase_count(detail::RuntimePhaseCounter::kChannelReadinessPublished) < expected &&
           session.is_running() && std::chrono::steady_clock::now() < deadline) {
      std::this_thread::yield();
    }
    return phase_count(detail::RuntimePhaseCounter::kChannelReadinessPublished) >= expected;
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
  const auto first_retirement_deadline = std::chrono::steady_clock::now() + 5s;
  while (phase_count(detail::RuntimePhaseCounter::kActivationsCompleted) <
             activation_baseline + 1U &&
         session.is_running() && std::chrono::steady_clock::now() < first_retirement_deadline) {
    std::this_thread::yield();
  }
  ASSERT_GE(phase_count(detail::RuntimePhaseCounter::kActivationsCompleted),
            activation_baseline + 1U);

  ASSERT_TRUE(publisher->close());
  ASSERT_TRUE(wait_for_channel_readiness(readiness_baseline + 2U));
  ASSERT_TRUE(session.is_running());
  ASSERT_TRUE(post_notification());
  ASSERT_TRUE(RetainedTensorEndOfStreamProbe::wait_for_count(2U));
  const auto second_retirement_deadline = std::chrono::steady_clock::now() + 5s;
  while (phase_count(detail::RuntimePhaseCounter::kActivationsCompleted) <
             activation_baseline + 2U &&
         session.is_running() && std::chrono::steady_clock::now() < second_retirement_deadline) {
    std::this_thread::yield();
  }
  ASSERT_GE(phase_count(detail::RuntimePhaseCounter::kActivationsCompleted),
            activation_baseline + 2U);

  EXPECT_EQ(RetainedTensorEndOfStreamProbe::count.load(std::memory_order_acquire), 2U);
  EXPECT_NE(RetainedTensorEndOfStreamProbe::data[0], nullptr);
  EXPECT_EQ(RetainedTensorEndOfStreamProbe::data[0], RetainedTensorEndOfStreamProbe::data[1]);
  EXPECT_EQ(RetainedTensorEndOfStreamProbe::sequences[0], 1U);
  EXPECT_EQ(RetainedTensorEndOfStreamProbe::sequences[1], 1U);
  EXPECT_EQ(RetainedTensorEndOfStreamProbe::values[0], 0x2aU);
  EXPECT_EQ(RetainedTensorEndOfStreamProbe::values[1], 0x2aU);

  session.request_stop();
  session.wait();
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
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

  RunSession session = run_async(plan);
  const bool compute_started = ExternalScalarExportSource::wait_for_compute_started();
  ExternalScalarExportSource::allow_emit();
  const bool emit_completed = ExternalScalarExportSource::wait_for_emit_completed();
  session.request_stop();
  session.wait();

  ASSERT_TRUE(compute_started);
  ASSERT_TRUE(emit_completed);
  // Key assertion: emit succeeds with kSuccess even when no external subscriber is connected.
  EXPECT_EQ(ExternalScalarExportSource::emit_error.load(std::memory_order_relaxed),
            ErrorCode::kSuccess);
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
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
