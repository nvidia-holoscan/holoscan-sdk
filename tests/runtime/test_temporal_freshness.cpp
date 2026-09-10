/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Freshness of a kDropOldest scalar selection feeding a slower independent trigger.
//
// Active cases cover direct scalar depths one through three, internal transported scalar delivery,
// ReuseLast supersession, finite terminal drain, the original high-rate selection-loss regression,
// and mixed-source recovery. External raw Tensor and descriptor-schema coverage remains in the
// dedicated external-Channel suite.

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include <fmt/format.h>
#include <gtest/gtest.h>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/run_session.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "compiler/compile_internal.hpp"
#include "runtime/port_binding.hpp"
#include "runtime/runtime_channel_binding.hpp"
#include "runtime/runtime_channel_binding_test_access.hpp"
#include "runtime/runtime_phase_counters.hpp"
#include "runtime/runtime_test_access.hpp"
#include "support/host_unique_test_name.hpp"

namespace holoscan {
namespace {

using namespace std::chrono_literals;

constexpr std::chrono::nanoseconds kSourcePeriod = 20ms;
constexpr std::chrono::nanoseconds kConsumerPeriod = 250ms;
constexpr std::chrono::nanoseconds kMaxSampleAge = 1s;  // deliberately above kConsumerPeriod
constexpr std::chrono::nanoseconds kReuseMaxAge = 30s;  // deliberately longer than any run here
constexpr std::uint32_t kQueueDepth = 1U;               // the depth-one case under test
constexpr std::size_t kObservedActivations = 6U;

// Freshness is asserted in samples, not wall time. A correct selection is one or two samples
// behind whatever the source has published; the pinned behavior is a whole consumer period behind,
// which at these rates is about twelve. Five separates them unambiguously.
//
// Counting samples rather than nanoseconds is what keeps this honest on a loaded runner. The
// source is a 20 ms OnClock operator in its own partition, so a stall of that partition inflates a
// wall-age bound while the defect is absent -- but it stalls publication and selection together,
// leaving the sample lag unchanged. Age is still reported for diagnosis.
constexpr std::uint64_t kMaximumFreshSampleLag = 5U;

// Every case below needs at most 3 s of run time, so this is a generous ceiling for a passing run
// on a loaded host. It exists to bound the cost of a failing one: a selector that stops admitting
// samples entirely produces no further activations, and waiting out a long budget for that answer
// would dominate the suite.
constexpr std::chrono::seconds kActivationBudget{10};

[[nodiscard]] bool wait_for_terminal(const RunSession& session) {
  const auto deadline = std::chrono::steady_clock::now() + kActivationBudget;
  while (session.is_running() && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(1ms);
  }
  return !session.is_running();
}

// How a consumer answers a missing record. kReuseLast is exercised because CHAN-131 lists it in
// the same retained-ownership sentence as LatestWithin; kSafeDefault is the only spelling that
// still enters compute on a missing record, which is what lets a test count lost selections
// directly instead of inferring them from activation timing.
enum class MissingPolicy : std::uint8_t { kSkip, kReuseLast, kSafeDefault };

// The reconcile outcomes each activation is reported against, in report order.
//
// A run total cannot answer the question these cases ask. "The reconcile evicted 900 times and
// three activations selected nothing" is consistent both with the eviction causing those three and
// with it causing none of them, and two candidate fixes were already accepted and then rejected on
// exactly that ambiguity. Deltas across the interval preceding one activation separate the two:
// a loss caused by the eviction is preceded by one, and a loss with no eviction behind it has a
// different cause and needs a different fix.
//
// The two poll-skip rows are here because they are the only way the eviction's ordering argument
// can hold and the activation still be lost -- the successor it promoted is never polled, so no
// ordering of the release could have saved it.
//
// The two MISS rows lead because they are the answer and the rest is the explanation. A lost
// selection is attributed at the moment it is formed, so it does not have to be inferred from
// interval totals: the reconcile fires tens of times per activation and only the round that froze
// the witness could have emptied the cache it read.
constexpr std::array<std::pair<detail::RuntimePhaseCounter, std::string_view>, 21U>
    kReportedCounters{{
        {detail::RuntimePhaseCounter::kSelectionMissedAfterFreshnessEviction, "MISS-evicted"},
        {detail::RuntimePhaseCounter::kSelectionMissedAfterUngatedEviction, "MISS-ungated"},
        {detail::RuntimePhaseCounter::kSelectionMissedAfterUnpolledEviction, "MISS-unpolled"},
        {detail::RuntimePhaseCounter::kSelectionMissedAfterUnrefilledEviction, "MISS-unrefilled"},
        {detail::RuntimePhaseCounter::kTemporalRetentionFloorAbsent, "nofloor"},
        {detail::RuntimePhaseCounter::kSelectionMissedWithoutFreshnessEviction, "MISS-other"},
        {detail::RuntimePhaseCounter::kTemporalRetentionEvictionPromotedNothing, "norefill"},
        {detail::RuntimePhaseCounter::kChannelRetirementPromotedNothing, "stuck"},
        {detail::RuntimePhaseCounter::kTemporalRetentionEvictions, "evict"},
        {detail::RuntimePhaseCounter::kTemporalRetentionDeclinedNotSuperseded, "same"},
        {detail::RuntimePhaseCounter::kTemporalRetentionDeclinedAboveFloor, "above"},
        {detail::RuntimePhaseCounter::kTemporalRetentionDeclinedUncomparableCoordinate, "uncmp"},
        {detail::RuntimePhaseCounter::kTemporalRetentionDeclinedUnboundedRound, "noclk"},
        {detail::RuntimePhaseCounter::kTemporalRetentionDeclinedEvictionRefused, "refuse"},
        {detail::RuntimePhaseCounter::kReadinessPollSkippedByCoordinatorBound, "bound"},
        {detail::RuntimePhaseCounter::kReadinessPollSkippedBySourceCredit, "credit"},
        {detail::RuntimePhaseCounter::kChannelPollIncomplete, "incompl"},
        {detail::RuntimePhaseCounter::kChannelPreWitnessReplacements, "pre-witness-replaced"},
        {detail::RuntimePhaseCounter::kChannelFreshnessReservationReleaseDeferred,
         "release-deferred"},
        {detail::RuntimePhaseCounter::kChannelReleasedClaimRefillReplacementDeclined,
         "refill-protected"},
        {detail::RuntimePhaseCounter::kChannelReleasedClaimRefillPublished, "refill-ok"},
    }};

using ReportedCounterValues = std::array<std::uint64_t, kReportedCounters.size()>;

// One row per consumer activation, in activation order. Keeping missing activations in the same
// sequence as selected ones is what lets an assertion skip the pre-steady-state prefix uniformly:
// before the first activation the source may legitimately have published nothing at all, and a
// separate missing counter cannot tell that startup case apart from a mid-run loss.
struct ObservedActivation {
  bool selected{};
  InputState selection_state{InputState::kAbsent};
  std::int64_t age_ns{};
  std::uint64_t value{};
  // What the source had published when this activation ran, so freshness can be judged against
  // the producer's own progress rather than against the wall clock.
  std::uint64_t published{};
  // The trigger coordinate this activation ran at. Freshness alone cannot see a *lost* activation:
  // the run simply waits one more period and the next one looks healthy, so a dropped clock tick is
  // invisible to an age assertion. The gap between consecutive coordinates is what exposes it.
  std::int64_t activation_time_ns{};
  // Run totals as of this activation's compute, not deltas. Storing totals keeps record() a plain
  // read with no dependence on the previous row, so a dropped or reordered activation cannot
  // corrupt the interval arithmetic that describe() does afterwards.
  ReportedCounterValues counters{};
};

struct FreshnessProbe {
  static void reset(std::shared_ptr<detail::RuntimePhaseCounters> attached = nullptr) {
    const std::lock_guard lock(mutex);
    activations.clear();
    counters = std::move(attached);
    published.store(0U, std::memory_order_relaxed);
  }

  // Published count, owned here rather than by the source so the consumer can read it without the
  // two operators sharing anything else.
  static inline std::atomic<std::uint64_t> published{};

  static void record(ObservedActivation activation) {
    {
      const std::lock_guard lock(mutex);
      if (counters != nullptr) {
        const detail::RuntimePhaseCounters::Snapshot snapshot = counters->snapshot();
        for (std::size_t index = 0U; index < kReportedCounters.size(); ++index) {
          activation.counters[index] =
              snapshot[detail::RuntimePhaseCounters::index(kReportedCounters[index].first)];
        }
      }
      activations.push_back(activation);
    }
    cv.notify_all();
  }

  [[nodiscard]] static bool wait_for_activations(std::size_t count) {
    std::unique_lock lock(mutex);
    return cv.wait_for(lock, kActivationBudget, [count] { return activations.size() >= count; });
  }

  // Every failure in this file is about a sequence, not a single value: one activation's age says
  // nothing about whether the selector is stale, pinned, or starving. Attach the whole observed
  // sequence to each assertion so a failure is diagnosable without a rerun.
  [[nodiscard]] static std::string describe() {
    const std::lock_guard lock(mutex);
    fmt::memory_buffer buffer;
    fmt::format_to(std::back_inserter(buffer), "observed [age(sample)]:");
    for (const ObservedActivation& activation : activations) {
      if (activation.selected) {
        fmt::format_to(std::back_inserter(buffer),
                       " {}{}ms(v{})",
                       activation.selection_state == InputState::kReused ? "reuse:" : "",
                       activation.age_ns / 1000000,
                       activation.value);
      } else {
        fmt::format_to(std::back_inserter(buffer), " none");
      }
    }
    if (activations.empty()) {
      fmt::format_to(std::back_inserter(buffer), " <no activation>");
    }
    if (counters != nullptr) {
      // The comparison that permitted each lost selection, in microseconds relative to the witness
      // that froze. A successor at or below the witness reference should have sorted ahead of it,
      // so a non-negative "ahead" on a lost selection means availability failed; a negative one
      // means the ordering did.
      for (const detail::TemporalRetentionMissRecord& row : counters->retention_misses()) {
        fmt::format_to(std::back_inserter(buffer),
                       " <MISS ahead={}us",
                       (row.witness_reference_ns - row.successor_capture_ns) / 1000);
        if (!row.floor_present) {
          fmt::format_to(std::back_inserter(buffer), " floor=none");
        } else {
          fmt::format_to(std::back_inserter(buffer),
                         " floorslack={}us",
                         (row.floor_ns - row.successor_capture_ns) / 1000);
        }
        fmt::format_to(std::back_inserter(buffer),
                       " state={} reason={} ingested={} polled={} pending={} pendingdelta={}us "
                       "refill={} promoted={}us>",
                       row.selection_state,
                       row.selection_reason,
                       row.ingested_this_round ? 1 : 0,
                       row.round_ingested_count,
                       row.pending_at_selection,
                       row.pending_oldest_reference_ns == 0
                           ? 0
                           : (row.pending_oldest_reference_ns - row.witness_reference_ns) / 1000,
                       row.refill_outcome == 1U   ? "published"
                       : row.refill_outcome == 2U ? "protected"
                                                  : "none",
                       row.promoted_capture_ns == 0
                           ? 0
                           : (row.promoted_capture_ns - row.witness_reference_ns) / 1000);
      }
      // Zero rows are dropped rather than printed as "x0". In steady state most of these branches
      // never fire, and printing every name for every activation would bury the one that did.
      fmt::format_to(std::back_inserter(buffer), "; per-activation reconcile deltas:");
      ReportedCounterValues previous{};
      for (std::size_t row = 0U; row < activations.size(); ++row) {
        fmt::format_to(
            std::back_inserter(buffer), " [{}{}", row, activations[row].selected ? "" : "!");
        for (std::size_t index = 0U; index < kReportedCounters.size(); ++index) {
          const std::uint64_t total = activations[row].counters[index];
          const std::uint64_t delta = total >= previous[index] ? total - previous[index] : 0U;
          if (delta != 0U) {
            fmt::format_to(
                std::back_inserter(buffer), " {}={}", kReportedCounters[index].second, delta);
          }
        }
        fmt::format_to(std::back_inserter(buffer), "]");
        previous = activations[row].counters;
      }
    }
    return fmt::to_string(buffer);
  }

  static inline std::mutex mutex;
  static inline std::condition_variable cv;
  static inline std::vector<ObservedActivation> activations;
  // Test-owned and retained by the plan, so it outlives the session that writes it.
  static inline std::shared_ptr<detail::RuntimePhaseCounters> counters;
};

struct CrossSourceRecoveryProbe {
  static void reset() {
    activation_count.store(0U, std::memory_order_relaxed);
    blocking_poll_calls.store(0U, std::memory_order_relaxed);
    incomplete_status_calls.store(0U, std::memory_order_relaxed);
  }

  static inline std::atomic<std::size_t> activation_count{};
  static inline std::atomic<std::size_t> blocking_poll_calls{};
  static inline std::atomic<std::size_t> incomplete_status_calls{};
};

// A periodic spin witness whose ordinary poll deliberately reports incomplete. It represents an
// unrelated physical source: a freshness eviction on the Channel side input must not borrow this
// binding's stale/incomplete state and run a blocking recovery against it.
class CrossSourceIncompleteBinding final : public detail::PortBinding {
 public:
  CrossSourceIncompleteBinding(const detail::RuntimePortBindingRequest& request,
                               TimePoint start_coordinate)
      : request_(request),
        next_due_(start_coordinate.timestamp_ns + kPeriod.count(), start_coordinate.clock_id),
        next_sequence_(request.initial_source_sequence + 1U) {}

  [[nodiscard]] expected<void, Error> poll(TimePoint observed_at) noexcept override {
    if (!observed_at.valid() || observed_at.clock_id != request_.reference_clock) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    if (!pending_ && !outstanding_ && observed_at.timestamp_ns >= next_due_.timestamp_ns) {
      event_ = detail::RuntimeEvent{
          .kind = request_.kind,
          .source = request_.source,
          .port = request_.port,
          .source_decision_time = observed_at,
          .reference_time = observed_at,
          .source_generation = request_.source_generation,
          .source_sequence = next_sequence_,
      };
      pending_ = true;
    }
    return {};
  }

  [[nodiscard]] expected<void, Error> poll_blocking(TimePoint) noexcept override {
    CrossSourceRecoveryProbe::blocking_poll_calls.fetch_add(1U, std::memory_order_relaxed);
    return make_unexpected(Error{ErrorCode::kFailure, "INJECTED_UNRELATED_RECOVERY_POLL"});
  }

  [[nodiscard]] std::optional<detail::PortBindingPollTarget> poll_target() const noexcept override {
    return detail::PortBindingPollTarget{
        .execution_lane = request_.execution_lane,
        .reference_clock = request_.reference_clock,
    };
  }

  [[nodiscard]] bool last_poll_incomplete() const noexcept override {
    CrossSourceRecoveryProbe::incomplete_status_calls.fetch_add(1U, std::memory_order_relaxed);
    return true;
  }

  [[nodiscard]] SourceId source() const noexcept override { return request_.source; }

  [[nodiscard]] expected<std::size_t, Error> drain(
      std::span<detail::RuntimeEvent> destination) noexcept override {
    if (!pending_ || destination.empty()) {
      return 0U;
    }
    destination.front() = event_;
    pending_ = false;
    outstanding_ = true;
    return 1U;
  }

  [[nodiscard]] expected<void, Error> retire(std::uint64_t source_sequence) noexcept override {
    if (!outstanding_ || source_sequence != next_sequence_) {
      return make_unexpected(Error{ErrorCode::kStaleEndpoint});
    }
    outstanding_ = false;
    ++next_sequence_;
    next_due_.timestamp_ns += kPeriod.count();
    return {};
  }

  void cancel() noexcept override { closed_ = true; }

  [[nodiscard]] std::size_t capacity() const noexcept override { return request_.capacity; }
  [[nodiscard]] bool closed() const noexcept override { return closed_; }
  [[nodiscard]] std::optional<TimePoint> next_wake() const noexcept override { return next_due_; }

 private:
  static constexpr std::chrono::nanoseconds kPeriod{20ms};

  detail::RuntimePortBindingRequest request_;
  TimePoint next_due_;
  detail::RuntimeEvent event_;
  std::uint64_t next_sequence_{};
  bool pending_{};
  bool outstanding_{};
  bool closed_{};
};

class CrossSourceIncompleteFactory final : public detail::RuntimePortBindingFactory {
 public:
  [[nodiscard]] expected<std::unique_ptr<detail::PortBinding>, Error> bind(
      const detail::RuntimePortBindingRequest& request,
      TimePoint start_coordinate) noexcept override {
    if (!request.source || request.capacity == 0U || !request.reference_clock.valid() ||
        request.initial_source_sequence == std::numeric_limits<std::uint64_t>::max() ||
        !start_coordinate.valid() || start_coordinate.clock_id != request.reference_clock) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    return std::unique_ptr<detail::PortBinding>{
        std::make_unique<CrossSourceIncompleteBinding>(request, start_coordinate)};
  }
};

// Separates physical spin activation from Channel-backed freshness input so tests can verify that
// recovery stays source-local; used by FreshnessRecoveryDoesNotPollAnUnrelatedIncompleteSource.
class CrossSourceFreshnessConsumer final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.spin_source(activation_, "activation").capacity(1U);
    spec.input(data_, "data").queue_depth(1U);
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnSpin{.event = activation_})
        .input(data_, LatestWithin{.max_age = 1s})
        .on_missing(data_, SafeDefault{});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    const std::size_t count =
        CrossSourceRecoveryProbe::activation_count.fetch_add(1U, std::memory_order_relaxed) + 1U;
    if (count == kActivationLimit) {
      context.request_completion();
    }
    return {};
  }

  static constexpr std::size_t kActivationLimit{8U};

  // Public only for graph authoring in the test below.
  SpinSource activation_;
  Input<std::uint64_t> data_;
};

// Publishes its own activation count, stamped with the activation coordinate so the consumer can
// age the selected sample without a second clock read.
template <std::int64_t PeriodUs = 20000>
class FreshnessSource final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override { spec.output(output, "output"); }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = std::chrono::microseconds{PeriodUs}});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    const TimePoint activation = context.activation_time();
    ++published_;
    FreshnessProbe::published.store(published_, std::memory_order_relaxed);
    return output.emit(Sample<std::uint64_t>{
        published_,
        SampleMetadata{
            .capture_timestamp_ns = activation.timestamp_ns,
            .source_clock_id = activation.clock_id,
        },
    });
  }

  Output<std::uint64_t> output;

 private:
  std::uint64_t published_{};
};

// Depth is a template parameter rather than a constructor argument because queue_depth is authored
// in setup(), before any instance state is available.
template <std::uint32_t Depth, MissingPolicy Missing = MissingPolicy::kSkip,
          std::int64_t PeriodUs = 250000>
class FreshnessConsumer final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override { spec.input(data, "data").queue_depth(Depth); }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = std::chrono::microseconds{PeriodUs}})
        .input(data, LatestWithin{.max_age = kMaxSampleAge});
    if constexpr (Missing == MissingPolicy::kReuseLast) {
      // Longer than the whole run, so reuse never expires and the reuse claim is continuously
      // held. The question under test is whether a newer sample can be admitted while it is held,
      // not what happens when the reuse window closes -- an expiring window would end the run in a
      // wait timeout and report a cadence failure instead of the staleness that causes it.
      result.on_missing(data, ReuseLast{.max_age = kReuseMaxAge});
    } else if constexpr (Missing == MissingPolicy::kSafeDefault) {
      result.on_missing(data, SafeDefault{});
    } else {
      result.on_missing(data, Skip{});
    }
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    if (data.selection().state != InputState::kSelected &&
        data.selection().state != InputState::kReused) {
      FreshnessProbe::record(
          {.selected = false,
           .selection_state = data.selection().state,
           .published = FreshnessProbe::published.load(std::memory_order_relaxed),
           .activation_time_ns = context.activation_time().timestamp_ns});
      return {};
    }
    auto sample = data.receive();
    if (!sample) {
      return make_unexpected(std::move(sample).error());
    }
    const std::int64_t age_ns = context.activation_time().timestamp_ns -
                                static_cast<std::int64_t>(sample->metadata.capture_timestamp_ns);
    FreshnessProbe::record({.selected = true,
                            .selection_state = data.selection().state,
                            .age_ns = age_ns,
                            .value = sample->data,
                            .published = FreshnessProbe::published.load(std::memory_order_relaxed),
                            .activation_time_ns = context.activation_time().timestamp_ns});
    return {};
  }

  Input<std::uint64_t> data;
};

enum class Realization : std::uint8_t { kDirect, kForcedHoloMq };

// Records the driver coordinates and side-input outcomes for a deliberately decreasing OnEach
// reference sequence. Used by
// TemporalFreshnessTest.LateVisibleLowerReferenceSelectsEvidencePreservedAfterStaleMiss.
struct LateVisibleReferenceProbe {
  static void reset() {
    const std::lock_guard lock{mutex};
    driver_capture_times.clear();
    selection_states.clear();
    selected_values.clear();
  }

  static void record(const std::uint64_t driver_capture_time, const InputState selection_state,
                     const std::optional<std::uint64_t> selected_value) {
    const std::lock_guard lock{mutex};
    driver_capture_times.push_back(driver_capture_time);
    selection_states.push_back(selection_state);
    selected_values.push_back(selected_value);
  }

  static inline std::mutex mutex;
  static inline std::vector<std::uint64_t> driver_capture_times;
  static inline std::vector<InputState> selection_states;
  static inline std::vector<std::optional<std::uint64_t>> selected_values;
};

// Publishes one t=90 side-input sample, then OnEach driver samples at t=200 and t=100. Used by
// TemporalFreshnessTest.LateVisibleLowerReferenceSelectsEvidencePreservedAfterStaleMiss.
class LateVisibleReferenceSource final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.output(data, "data").max_emits_per_compute(1U);
    spec.output(driver, "driver").max_emits_per_compute(1U);
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 20ms}).complete_after(ComputeCalls{.limit = 6U});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    ++calls_;
    const ClockId clock = context.activation_time().clock_id;
    if (calls_ == 1U) {
      return data.emit(Sample<std::uint64_t>{
          90U, SampleMetadata{.capture_timestamp_ns = 90U, .source_clock_id = clock}});
    }
    if (calls_ == 2U) {
      return driver.emit(Sample<std::uint64_t>{
          1U, SampleMetadata{.capture_timestamp_ns = 200U, .source_clock_id = clock}});
    }
    if (calls_ == 3U) {
      return driver.emit(Sample<std::uint64_t>{
          2U, SampleMetadata{.capture_timestamp_ns = 100U, .source_clock_id = clock}});
    }
    return {};
  }

  Output<std::uint64_t> data;
  Output<std::uint64_t> driver;

 private:
  std::uint64_t calls_{};
};

// Applies LatestWithin + SafeDefault to the side input while OnEach supplies decreasing semantic
// references. Used by
// TemporalFreshnessTest.LateVisibleLowerReferenceSelectsEvidencePreservedAfterStaleMiss.
class LateVisibleReferenceConsumer final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(data, "data").queue_depth(1U);
    spec.input(driver, "driver").queue_depth(1U);
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnEach{driver})
        .input(data, LatestWithin{.max_age = 50ns})
        .on_missing(data, SafeDefault{});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    auto driver_sample = driver.receive();
    if (!driver_sample) {
      return make_unexpected(std::move(driver_sample).error());
    }
    std::optional<std::uint64_t> selected_value;
    if (data.selection().state == InputState::kSelected) {
      auto data_sample = data.receive();
      if (!data_sample) {
        return make_unexpected(std::move(data_sample).error());
      }
      selected_value = data_sample->data;
    }
    LateVisibleReferenceProbe::record(
        driver_sample->metadata.capture_timestamp_ns, data.selection().state, selected_value);
    return {};
  }

  Input<std::uint64_t> data;
  Input<std::uint64_t> driver;
};

template <Realization Delivery>
void run_late_visible_reference_graph() {
  LateVisibleReferenceProbe::reset();
  Graph graph{"late-visible-lower-reference"};
  auto source = graph.op<LateVisibleReferenceSource>("source");
  auto consumer = graph.op<LateVisibleReferenceConsumer>("consumer");
  graph.add_flow(source->driver, consumer->driver);
  graph.add_flow(source->data,
                 consumer->data,
                 ConnectionOptions{.queue_policy = QueuePolicy::kBlocking, .queue_depth = 1U});
  graph.partition("source").add(source);
  graph.partition("consumer").add(consumer);
  graph.set_default_clock(graph.add_clock<RealtimeClock>("runtime_clock"));
  const ExecutionPlan plan =
      Delivery == Realization::kForcedHoloMq
          ? detail::compile_with_process_local_holomq(graph, CompileOptions{})
          : compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  RunSession session = run_async(plan);
  const bool terminal = wait_for_terminal(session);
  if (!terminal) {
    session.request_stop();
  }
  session.wait();
  ASSERT_TRUE(terminal);
  EXPECT_EQ(session.termination(), RunTermination::kCompleted);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
  ASSERT_EQ(LateVisibleReferenceProbe::driver_capture_times,
            (std::vector<std::uint64_t>{200U, 100U}));
  ASSERT_EQ(LateVisibleReferenceProbe::selection_states,
            (std::vector<InputState>{InputState::kSafeDefault, InputState::kSelected}));
  ASSERT_EQ(LateVisibleReferenceProbe::selected_values.size(), 2U);
  EXPECT_FALSE(LateVisibleReferenceProbe::selected_values[0].has_value());
  EXPECT_EQ(LateVisibleReferenceProbe::selected_values[1], 90U);
}

enum class FiniteReuseSelector : std::uint8_t { kExact, kLatestWithin };
enum class FiniteReuseWitness : std::uint8_t {
  kMatching,
  kIneligibleOnly,
  kZeroCoordinate,
};

struct FiniteReuseProbe {
  static void reset() {
    const std::lock_guard lock{mutex};
    values.clear();
    sequences.clear();
    witnesses.clear();
    states.clear();
    reasons.clear();
    retention_evictions = 0U;
    emission_error.store(ErrorCode::kSuccess, std::memory_order_relaxed);
  }
  static void record(const Sample<std::uint64_t>& sample, const std::uint64_t witness,
                     const InputState state,
                     const SelectionReason reason = SelectionReason::kNone) {
    const std::lock_guard lock{mutex};
    values.push_back(sample.data);
    sequences.push_back(sample.metadata.sequence);
    witnesses.push_back(witness);
    states.push_back(state);
    reasons.push_back(reason);
  }
  static inline std::mutex mutex;
  static inline std::vector<std::uint64_t> values;
  static inline std::vector<std::uint64_t> sequences;
  static inline std::vector<std::uint64_t> witnesses;
  static inline std::vector<InputState> states;
  static inline std::vector<SelectionReason> reasons;
  static inline std::uint64_t retention_evictions{};
  static inline std::atomic<ErrorCode> emission_error{ErrorCode::kSuccess};
};

[[nodiscard]] std::uint64_t phase_value(const detail::RuntimePhaseCounters::Snapshot& snapshot,
                                        const detail::RuntimePhaseCounter counter) noexcept {
  return snapshot[detail::RuntimePhaseCounters::index(counter)];
}

void expect_retention_counter_identity(const detail::RuntimePhaseCounters::Snapshot& snapshot) {
  const std::uint64_t accounted =
      phase_value(snapshot, detail::RuntimePhaseCounter::kTemporalRetentionEvictions) +
      phase_value(snapshot, detail::RuntimePhaseCounter::kTemporalRetentionDeclinedNotSuperseded) +
      phase_value(snapshot, detail::RuntimePhaseCounter::kTemporalRetentionDeclinedUnboundedRound) +
      phase_value(snapshot,
                  detail::RuntimePhaseCounter::kTemporalRetentionDeclinedUncomparableCoordinate) +
      phase_value(snapshot, detail::RuntimePhaseCounter::kTemporalRetentionDeclinedAboveFloor) +
      phase_value(snapshot, detail::RuntimePhaseCounter::kTemporalRetentionDeclinedEvictionRefused);
  EXPECT_EQ(phase_value(snapshot, detail::RuntimePhaseCounter::kTemporalRetentionCandidates),
            accounted);
}

class FreshnessHandoffRetryBindControl final : public detail::RuntimeChannelSessionBindTestControl {
 public:
  [[nodiscard]] expected<void, Error> bind(
      detail::RuntimeChannelSession& session) noexcept override {
    auto armed =
        detail::RuntimeChannelSessionTestAccess::fail_next_freshness_handoff_retry(session);
    if (armed) {
      armed_.store(true, std::memory_order_release);
    }
    return armed;
  }

  [[nodiscard]] bool armed() const noexcept { return armed_.load(std::memory_order_acquire); }

 private:
  std::atomic<bool> armed_{};
};

class CaptureAbsentReuseSource final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.output(output, "output").max_emits_per_compute(1U);
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    // Leave bounded controller time for the capture-absent arrival to dispatch before closure;
    // no later payload is emitted, so this changes only the close tick, not selection evidence.
    result.trigger(OnClock{.period = 20ms}).complete_after(ComputeCalls{.limit = 6U});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    ++calls_;
    if (calls_ == 1U) {
      return output.emit(Sample<std::uint64_t>{
          1U,
          SampleMetadata{.capture_timestamp_ns = context.activation_time().timestamp_ns,
                         .source_clock_id = context.activation_time().clock_id}});
    }
    if (calls_ == 2U) {
      auto emitted = output.emit(Sample<std::uint64_t>{2U, SampleMetadata{}});
      if (!emitted) {
        FiniteReuseProbe::emission_error.store(emitted.error().code, std::memory_order_relaxed);
      }
      return emitted;
    }
    return {};
  }

  Output<std::uint64_t> output;

 private:
  std::uint64_t calls_{};
};

class CaptureAbsentReuseConsumer final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override { spec.input(input, "input").queue_depth(1U); }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnAnyInput{});
    result.input(input, LatestWithin{.max_age = 1s});
    result.on_missing(input, ReuseLast{.max_age = 30s});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    auto sample = input.receive();
    if (!sample) {
      return make_unexpected(std::move(sample).error());
    }
    FiniteReuseProbe::record(*sample, 0U, input.selection().state, input.selection().reason);
    return {};
  }

  Input<std::uint64_t> input;
};

template <Realization Delivery>
void run_capture_absent_reuse_graph() {
  FiniteReuseProbe::reset();
  Graph graph{"capture-absent-reuse-terminal-drain"};
  auto source = graph.op<CaptureAbsentReuseSource>("source");
  auto consumer = graph.op<CaptureAbsentReuseConsumer>("consumer");
  graph.add_flow(source->output,
                 consumer->input,
                 ConnectionOptions{.queue_policy = QueuePolicy::kDropOldest, .queue_depth = 1U});
  graph.partition("source").add(source);
  graph.partition("consumer").add(consumer);
  graph.set_default_clock(graph.add_clock<RealtimeClock>("runtime_clock"));
  const ExecutionPlan plan =
      Delivery == Realization::kForcedHoloMq
          ? detail::compile_with_process_local_holomq(graph, CompileOptions{})
          : compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  RunSession session = run_async(plan);
  const bool terminal = wait_for_terminal(session);
  if (!terminal) {
    session.request_stop();
  }
  session.wait();
  ASSERT_TRUE(terminal);
  EXPECT_EQ(session.termination(), RunTermination::kCompleted);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
  EXPECT_EQ(FiniteReuseProbe::emission_error.load(std::memory_order_relaxed), ErrorCode::kSuccess);
}

template <FiniteReuseSelector Selector, FiniteReuseWitness Witness = FiniteReuseWitness::kMatching>
class FiniteReuseSource final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.output(data, "data").max_emits_per_compute(1U);
    spec.output(driver, "driver").max_emits_per_compute(1U);
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    // Complete one tick after the final driver so producer EOS cannot race that activation. The
    // test then observes Runtime's terminal retirement, not a scheduler tie between data and EOS.
    constexpr std::uint64_t kCalls = (Selector == FiniteReuseSelector::kExact ? 4U : 5U) +
                                     (Witness == FiniteReuseWitness::kIneligibleOnly ? 25U : 0U);
    result.trigger(OnClock{.period = 20ms}).complete_after(ComputeCalls{.limit = kCalls});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    ++calls_;
    const SampleMetadata metadata{
        .capture_timestamp_ns = Witness == FiniteReuseWitness::kZeroCoordinate
                                    ? 0U
                                    : context.activation_time().timestamp_ns,
        .source_clock_id = context.activation_time().clock_id};
    constexpr std::uint64_t kFinalDataCall = Selector == FiniteReuseSelector::kExact ? 2U : 3U;
    constexpr std::uint64_t kFinalDriverCall = kFinalDataCall + 1U;
    if (calls_ <= kFinalDataCall) {
      auto emitted = data.emit(Sample<std::uint64_t>{calls_, metadata});
      if (!emitted) {
        return emitted;
      }
      if (calls_ == kFinalDataCall) {
        final_data_metadata_ = metadata;
      }
    }
    if (calls_ == 1U) {
      first_data_metadata_ = metadata;
      return driver.emit(Sample<std::uint64_t>{calls_, metadata});
    }
    if (calls_ == kFinalDriverCall) {
      if constexpr (Witness == FiniteReuseWitness::kIneligibleOnly) {
        SampleMetadata ineligible = first_data_metadata_;
        ineligible.capture_timestamp_ns += (final_data_metadata_.capture_timestamp_ns -
                                            first_data_metadata_.capture_timestamp_ns) /
                                           2;
        return driver.emit(Sample<std::uint64_t>{calls_, ineligible});
      }
      return driver.emit(Sample<std::uint64_t>{calls_, final_data_metadata_});
    }
    return {};
  }

  Output<std::uint64_t> data;
  Output<std::uint64_t> driver;

 private:
  std::uint64_t calls_{};
  SampleMetadata final_data_metadata_{};
  SampleMetadata first_data_metadata_{};
};

template <FiniteReuseSelector Selector, std::uint32_t Depth,
          FiniteReuseWitness Witness = FiniteReuseWitness::kMatching>
class FiniteReuseConsumer final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(data, "data").queue_depth(Depth);
    spec.input(driver, "driver").queue_depth(1U);
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnEach{driver});
    if constexpr (Selector == FiniteReuseSelector::kExact) {
      result.input(data, Exact{});
    } else {
      result.input(data, LatestWithin{.max_age = 1s});
    }
    result.on_missing(data, ReuseLast{.max_age = 30s});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    auto witness = driver.receive();
    if (!witness) {
      return make_unexpected(std::move(witness).error());
    }
    auto sample = data.receive();
    if (!sample) {
      return make_unexpected(std::move(sample).error());
    }
    FiniteReuseProbe::record(*sample, witness->data, data.selection().state);
    return {};
  }

  Input<std::uint64_t> data;
  Input<std::uint64_t> driver;
};

template <FiniteReuseSelector Selector, std::uint32_t Depth,
          FiniteReuseWitness Witness = FiniteReuseWitness::kMatching>
void run_finite_reuse_graph(
    detail::RuntimePhaseCounters::Snapshot* const final_snapshot = nullptr) {
  FiniteReuseProbe::reset();
  Graph graph{test::make_host_unique_test_name("temporal-finite-reuse-terminal-drain")};
  auto source = graph.op<FiniteReuseSource<Selector, Witness>>("source");
  auto consumer = graph.op<FiniteReuseConsumer<Selector, Depth, Witness>>("consumer");
  // Register the trigger edge first so equal-coordinate trigger and side-input revisions sort with
  // the trigger first. The zero-coordinate case therefore depends on prerequisite ingestion, not
  // incidental source registration order.
  graph.add_flow(source->driver, consumer->driver);
  graph.add_flow(source->data,
                 consumer->data,
                 ConnectionOptions{.queue_policy = QueuePolicy::kDropOldest, .queue_depth = Depth});
  graph.partition("source").add(source);
  graph.partition("consumer").add(consumer);
  graph.set_default_clock(graph.add_clock<RealtimeClock>("runtime_clock"));
  const ExecutionPlan compiled = detail::compile_with_process_local_holomq(graph, CompileOptions{});
  ASSERT_TRUE(compiled.ok()) << compiled.json();
  auto counters = std::make_shared<detail::RuntimePhaseCounters>();
  const ExecutionPlan plan = detail::RuntimeTestPlanAccess::with_phase_counters(compiled, counters);
  ASSERT_TRUE(plan.ok()) << plan.json();
  RunSession session = run_async(plan);
  const bool terminal = wait_for_terminal(session);
  if (!terminal) {
    session.request_stop();
  }
  session.wait();
  const auto snapshot = counters->snapshot();
  if (final_snapshot != nullptr) {
    *final_snapshot = snapshot;
  }
  FiniteReuseProbe::retention_evictions = snapshot[detail::RuntimePhaseCounters::index(
      detail::RuntimePhaseCounter::kTemporalRetentionEvictions)];
  ASSERT_TRUE(terminal)
      << "values=" << FiniteReuseProbe::values.size() << " candidates="
      << snapshot[detail::RuntimePhaseCounters::index(
             detail::RuntimePhaseCounter::kTemporalRetentionCandidates)]
      << " evictions="
      << snapshot[detail::RuntimePhaseCounters::index(
             detail::RuntimePhaseCounter::kTemporalRetentionEvictions)]
      << " refused="
      << snapshot[detail::RuntimePhaseCounters::index(
             detail::RuntimePhaseCounter::kTemporalRetentionDeclinedEvictionRefused)]
      << " same="
      << snapshot[detail::RuntimePhaseCounters::index(
             detail::RuntimePhaseCounter::kTemporalRetentionDeclinedNotSuperseded)]
      << " above="
      << snapshot[detail::RuntimePhaseCounters::index(
             detail::RuntimePhaseCounter::kTemporalRetentionDeclinedAboveFloor)];
  EXPECT_EQ(session.termination(), RunTermination::kCompleted);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
  expect_retention_counter_identity(snapshot);
}

// Whether the compiler is allowed to pick direct process-local retention, or is forced onto the
// source-private HoloMQ realization. CHAN-128 requires both to preserve the same temporal
// selection, so every freshness property below has to hold for either.
template <std::uint32_t Depth, MissingPolicy Missing = MissingPolicy::kSkip,
          Realization Delivery = Realization::kDirect, std::int64_t SourceUs = 20000,
          std::int64_t ConsumerUs = 250000>
void run_freshness_graph(std::size_t wanted_activations, bool& observed) {
  Graph graph{"temporal-freshness-drop-oldest"};
  auto source = graph.op<FreshnessSource<SourceUs>>("source");
  auto consumer = graph.op<FreshnessConsumer<Depth, Missing, ConsumerUs>>("consumer");
  graph.add_flow(source->output,
                 consumer->data,
                 ConnectionOptions{.queue_policy = QueuePolicy::kDropOldest, .queue_depth = Depth});

  // Independent execution roots, so the slow consumer never paces the fast source. Sharing one
  // root would let the consumer's period throttle production and hide the defect.
  graph.partition("source").add(source);
  graph.partition("consumer").add(consumer);
  graph.set_default_clock(graph.add_clock<RealtimeClock>("runtime_clock"));

  const ExecutionPlan compiled =
      Delivery == Realization::kForcedHoloMq
          ? detail::compile_with_process_local_holomq(graph, CompileOptions{})
          : compile(graph);
  ASSERT_TRUE(compiled.ok()) << compiled.json();

  // Attached for every case, not only the failing one. These counters are what distinguishes a
  // reconcile that declined from one that never had a candidate, and a case that regresses later
  // is diagnosable from its own failure output instead of needing this wiring added back.
  auto counters = std::make_shared<detail::RuntimePhaseCounters>();
  const ExecutionPlan plan = detail::RuntimeTestPlanAccess::with_phase_counters(compiled, counters);
  ASSERT_TRUE(plan.ok()) << plan.json();

  FreshnessProbe::reset(counters);
  RunSession session = run_async(plan);
  observed = FreshnessProbe::wait_for_activations(wanted_activations);
  session.request_stop();
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty());
}

// Every steady-state activation must select a sample no older than a few source periods. The first
// is excluded because before any steady state exists the source may legitimately have published
// only its very first value.
void expect_steady_state_freshness(bool observed, std::size_t wanted,
                                   std::int64_t consumer_period_ns = kConsumerPeriod.count()) {
  ASSERT_TRUE(observed) << "only " << FreshnessProbe::activations.size() << " of " << wanted
                        << " consumer activations ran; " << FreshnessProbe::describe();
  ASSERT_GE(FreshnessProbe::activations.size(), wanted) << FreshnessProbe::describe();
  for (std::size_t index = 1U; index < wanted; ++index) {
    const ObservedActivation& activation = FreshnessProbe::activations[index];
    EXPECT_TRUE(activation.selected)
        << "activation " << index << " selected no sample; " << FreshnessProbe::describe();
    if (activation.selected) {
      const std::uint64_t lag =
          activation.published > activation.value ? activation.published - activation.value : 0U;
      EXPECT_LE(lag, kMaximumFreshSampleLag)
          << "activation " << index << " selected sample " << activation.value << " while the "
          << "source had published " << activation.published << ", " << lag << " samples behind ("
          << (activation.age_ns / 1000000) << " ms); " << FreshnessProbe::describe();
    }
    // Tick continuity. Under Skip an activation the reconcile costs does not appear as a stale
    // sample, it appears as no activation at all, and waiting one more period hides that from the
    // lag assertion above. Two periods is the smallest gap meaning a tick was lost rather than
    // merely late.
    //
    // This is a guard, not a reproduction, and it is deliberately confined to the slow cases. A
    // timing gap cannot distinguish a lost activation from a coalesced one under
    // kCoalesceLateTicks, so the same assertion at the 50 us rate flakes on a loaded runner --
    // measured, at exactly the two-period boundary. At a 250 ms trigger the margin is far above
    // load jitter. Catching the high-rate case is what the SafeDefault variant is for: it stays in
    // compute on a missing record and counts the loss directly, with no timing inference at all.
    const std::int64_t gap =
        activation.activation_time_ns - FreshnessProbe::activations[index - 1U].activation_time_ns;
    EXPECT_LT(gap, 2 * consumer_period_ns)
        << "activation " << index << " ran " << (gap / 1000000) << " ms after activation "
        << (index - 1U) << ", at least one trigger period missing; " << FreshnessProbe::describe();
  }
}

// REQ: RUNTIME-017
// REQ: CHAN-131
// A retained scalar record marks its transport slot temporal_retained, which excludes the slot
// from drop-oldest replacement. At depth one that pinned the sole typed slot for the whole
// inter-activation interval, so each activation selected the sample promoted just after the
// previous one -- an age of about one consumer period rather than one source period, with every
// sample produced in between staged and then dropped.
//
// The assertion is on freshness, not progress. The pinned behavior still advanced the selected
// value on every activation, so asserting only that selected values increase would pass against
// it. Raising queue_depth does not help either: typed_slot_capacity is
// max(requested_queue_depth, cache_capacity) and cache_capacity is the same resolved retention
// depth, so the selector cache can always pin every typed slot.
TEST(TemporalFreshnessTest, DepthOneDropOldestLatestWithinSelectsTheNewestCausalSample) {
  bool observed = false;
  ASSERT_NO_FATAL_FAILURE((run_freshness_graph<kQueueDepth>(kObservedActivations, observed)));
  expect_steady_state_freshness(observed, kObservedActivations);
}

// REQ: CONTRACT-004
// REQ: CONTRACT-021
// REQ: RUNTIME-017
// REQ: RUNTIME-043
TEST(TemporalFreshnessTest, LateVisibleLowerReferenceSelectsEvidencePreservedAfterStaleMiss) {
  // Graph:
  //
  //   [source: data(t=90), driver(t=200 -> t=100)]
  //                | data + driver
  //                v
  //   [consumer: OnEach(driver), LatestWithin(data, 50 ns) + SafeDefault]
  //
  // Key setup: two process-local partitions run the graph once through direct retention and once
  // through forced HoloMQ; the first driver makes data t=90 too stale, while the independently
  // accepted later-visible driver at t=100 makes that same sample eligible.
  // Key assertion: both realizations preserve t=90 across the committed SafeDefault activation at
  // t=200 and select its payload when the lower semantic reference becomes visible.
  ASSERT_NO_FATAL_FAILURE((run_late_visible_reference_graph<Realization::kDirect>()));
  ASSERT_NO_FATAL_FAILURE((run_late_visible_reference_graph<Realization::kForcedHoloMq>()));
}

// REQ: CHAN-136
// REQ: RUNTIME-050
// A production RunSession selector creates the real temporal_retained + reuse_retained state. A
// forced transported successor must advance that state without TestAccess clearing either claim;
// transport-specific final-data cases below cover payload, sequence, EOS, and idempotence.
TEST(TemporalFreshnessTest, DepthOneReuseLastAdvancesThroughRealTransportedSelectorHandoff) {
  bool observed = false;
  constexpr std::size_t kAdvancementActivations = 3U;
  ASSERT_NO_FATAL_FAILURE(
      (run_freshness_graph<kQueueDepth, MissingPolicy::kReuseLast, Realization::kForcedHoloMq>(
          kAdvancementActivations, observed)));
  ASSERT_TRUE(observed) << FreshnessProbe::describe();
  ASSERT_GE(FreshnessProbe::activations.size(), kAdvancementActivations)
      << FreshnessProbe::describe();
  EXPECT_GT(FreshnessProbe::activations[kAdvancementActivations - 1U].value,
            FreshnessProbe::activations.front().value)
      << FreshnessProbe::describe();
}

// A real forced-HoloMQ ordinary LatestWithin handoff must invoke the auxiliary retry only after
// selector and Channel ownership commit. The one-shot failure is the branch witness: before the
// fix the handoff records a successful promotion while the run stays live; after the fix that same
// handoff consumes the injection and fails through the established binding diagnostic.
// REQ: CHAN-136
// REQ: RUNTIME-017
TEST(TemporalFreshnessTest, ForcedHoloMqOrdinaryLatestWithinHandoffConsumesThePostCommitRetry) {
  Graph graph{"ordinary-latest-within-post-commit-retry"};
  auto source = graph.op<FreshnessSource<20000>>("source");
  auto consumer = graph.op<FreshnessConsumer<1U, MissingPolicy::kSkip, 250000>>("consumer");
  graph.add_flow(source->output,
                 consumer->data,
                 ConnectionOptions{.queue_policy = QueuePolicy::kDropOldest, .queue_depth = 1U});
  graph.partition("source").add(source);
  graph.partition("consumer").add(consumer);
  graph.set_default_clock(graph.add_clock<RealtimeClock>("runtime_clock"));

  const ExecutionPlan compiled = detail::compile_with_process_local_holomq(graph, CompileOptions{});
  ASSERT_TRUE(compiled.ok()) << compiled.json();
  const auto counters = std::make_shared<detail::RuntimePhaseCounters>();
  const ExecutionPlan counted =
      detail::RuntimeTestPlanAccess::with_phase_counters(compiled, counters);
  const auto control = std::make_shared<FreshnessHandoffRetryBindControl>();
  const ExecutionPlan plan =
      detail::RuntimeTestPlanAccess::with_channel_session_bind_test_control(counted, control);
  ASSERT_TRUE(plan.ok()) << plan.json();

  FreshnessProbe::reset(counters);
  RunSession session = run_async(plan);
  const auto deadline = std::chrono::steady_clock::now() + 2s;
  detail::RuntimePhaseCounters::Snapshot snapshot{};
  while (session.is_running() && std::chrono::steady_clock::now() < deadline) {
    if (!control->armed()) {
      std::this_thread::yield();
      continue;
    }
    snapshot = counters->snapshot();
    if (phase_value(snapshot, detail::RuntimePhaseCounter::kTemporalRetentionEvictionPromoted) !=
        0U) {
      break;
    }
    std::this_thread::yield();
  }
  const bool terminal = !session.is_running();
  if (!terminal) {
    session.request_stop();
  }
  session.wait();
  snapshot = counters->snapshot();

  ASSERT_TRUE(control->armed());
  EXPECT_GT(phase_value(snapshot, detail::RuntimePhaseCounter::kTemporalRetentionEvictions), 0U)
      << "the injected retry must be reached through a committed ordinary handoff";
  EXPECT_TRUE(terminal) << "the injected post-commit retry failure never reached the run";
  EXPECT_EQ(session.termination(), RunTermination::kFailed);
  ASSERT_FALSE(session.diagnostics().empty());
  EXPECT_EQ(session.diagnostics().front().token, "RUNTIME_PORT_BINDING_UNAVAILABLE");
}

// Reuse-only recovery passes have a complete counter identity even when no replacement is staged.
// The finite graph reaches those ordinary minimum checks while keeping its retained identity
// intact, so every candidate must resolve to one eviction or one explicit decline.
// REQ: CHAN-136
// REQ: RUNTIME-017
TEST(TemporalFreshnessTest, ReuseOnlyPassesAccountForEveryCandidate) {
  detail::RuntimePhaseCounters::Snapshot snapshot{};
  ASSERT_NO_FATAL_FAILURE(
      (run_finite_reuse_graph<FiniteReuseSelector::kLatestWithin, 1U>(&snapshot)));
  EXPECT_EQ(FiniteReuseProbe::values, (std::vector<std::uint64_t>{1U, 3U}));
  EXPECT_EQ(FiniteReuseProbe::sequences, (std::vector<std::uint64_t>{1U, 3U}));
  EXPECT_GT(
      phase_value(snapshot, detail::RuntimePhaseCounter::kTemporalRetentionDeclinedNotSuperseded),
      0U);
}

// REQ: CHAN-136
// REQ: RUNTIME-050
TEST(TemporalFreshnessTest, DepthOneExactReuseLastPublishesFinalDataBeforeSingleTerminalDrain) {
  ASSERT_NO_FATAL_FAILURE((run_finite_reuse_graph<FiniteReuseSelector::kExact, 1U>()));
  ASSERT_EQ(FiniteReuseProbe::values.size(), 2U);
  EXPECT_EQ(FiniteReuseProbe::values, (std::vector<std::uint64_t>{1U, 2U}));
  EXPECT_EQ(FiniteReuseProbe::sequences, (std::vector<std::uint64_t>{1U, 2U}));
  EXPECT_EQ(FiniteReuseProbe::states,
            (std::vector<InputState>{InputState::kSelected, InputState::kSelected}));
}

// REQ: CHAN-136
// REQ: RUNTIME-050
// REQ: RUNTIME-051
TEST(TemporalFreshnessTest, ExactReuseLastIngestsZeroCoordinateSuccessorBeforeMatchingTrigger) {
  ASSERT_NO_FATAL_FAILURE((run_finite_reuse_graph<FiniteReuseSelector::kExact,
                                                  1U,
                                                  FiniteReuseWitness::kZeroCoordinate>()));
  EXPECT_EQ(FiniteReuseProbe::values, (std::vector<std::uint64_t>{1U, 2U}));
  EXPECT_EQ(FiniteReuseProbe::sequences, (std::vector<std::uint64_t>{1U, 2U}));
  EXPECT_EQ(FiniteReuseProbe::witnesses, (std::vector<std::uint64_t>{1U, 3U}));
  EXPECT_EQ(FiniteReuseProbe::states,
            (std::vector<InputState>{InputState::kSelected, InputState::kSelected}));
}

// REQ: CHAN-136
// REQ: RUNTIME-050
TEST(TemporalFreshnessTest, ExactReuseLastKeepsRetainedSampleForMismatchedWitness) {
  ASSERT_NO_FATAL_FAILURE((run_finite_reuse_graph<FiniteReuseSelector::kExact,
                                                  1U,
                                                  FiniteReuseWitness::kIneligibleOnly>()));
  EXPECT_EQ(FiniteReuseProbe::values, (std::vector<std::uint64_t>{1U, 1U}));
  EXPECT_EQ(FiniteReuseProbe::sequences, (std::vector<std::uint64_t>{1U, 1U}));
  EXPECT_EQ(FiniteReuseProbe::witnesses, (std::vector<std::uint64_t>{1U, 3U}));
  EXPECT_EQ(FiniteReuseProbe::states,
            (std::vector<InputState>{InputState::kSelected, InputState::kReused}));
  EXPECT_EQ(FiniteReuseProbe::retention_evictions, 1U);
}

// REQ: RUNTIME-043
// REQ: RUNTIME-050
TEST(TemporalFreshnessTest, DirectCaptureAbsentArrivalReusesPriorPayloadThenDrains) {
  ASSERT_NO_FATAL_FAILURE((run_capture_absent_reuse_graph<Realization::kDirect>()));
  EXPECT_EQ(FiniteReuseProbe::values, (std::vector<std::uint64_t>{1U, 1U}));
  EXPECT_EQ(FiniteReuseProbe::sequences, (std::vector<std::uint64_t>{1U, 1U}));
  EXPECT_EQ(FiniteReuseProbe::states,
            (std::vector<InputState>{InputState::kSelected, InputState::kReused}));
  EXPECT_EQ(
      FiniteReuseProbe::reasons,
      (std::vector<SelectionReason>{SelectionReason::kNone, SelectionReason::kCaptureTimeAbsent}));
}

// REQ: RUNTIME-043
// REQ: RUNTIME-050
TEST(TemporalFreshnessTest, TransportedCaptureAbsentArrivalReusesPriorPayloadThenDrains) {
  ASSERT_NO_FATAL_FAILURE((run_capture_absent_reuse_graph<Realization::kForcedHoloMq>()));
  EXPECT_EQ(FiniteReuseProbe::values, (std::vector<std::uint64_t>{1U, 1U}));
  EXPECT_EQ(FiniteReuseProbe::sequences, (std::vector<std::uint64_t>{1U, 1U}));
  EXPECT_EQ(FiniteReuseProbe::states,
            (std::vector<InputState>{InputState::kSelected, InputState::kReused}));
  EXPECT_EQ(
      FiniteReuseProbe::reasons,
      (std::vector<SelectionReason>{SelectionReason::kNone, SelectionReason::kCaptureTimeAbsent}));
}

// REQ: CHAN-136
// REQ: RUNTIME-050
TEST(TemporalFreshnessTest,
     DepthTwoLatestWithinReuseLastReplacesExactReuseSlotThenDrainsSingleTerminal) {
  ASSERT_NO_FATAL_FAILURE((run_finite_reuse_graph<FiniteReuseSelector::kLatestWithin, 2U>()));
  ASSERT_EQ(FiniteReuseProbe::values.size(), 2U);
  EXPECT_EQ(FiniteReuseProbe::values, (std::vector<std::uint64_t>{1U, 3U}));
  EXPECT_EQ(FiniteReuseProbe::sequences, (std::vector<std::uint64_t>{1U, 3U}));
  EXPECT_EQ(FiniteReuseProbe::states,
            (std::vector<InputState>{InputState::kSelected, InputState::kSelected}));
}

// REQ: CHAN-136
// REQ: RUNTIME-050
TEST(TemporalFreshnessTest, LatestWithinReuseLastKeepsRetainedSampleWhenStagedDataIsFuture) {
  ASSERT_NO_FATAL_FAILURE((run_finite_reuse_graph<FiniteReuseSelector::kLatestWithin,
                                                  2U,
                                                  FiniteReuseWitness::kIneligibleOnly>()));
  EXPECT_EQ(FiniteReuseProbe::values, (std::vector<std::uint64_t>{1U, 2U}));
  EXPECT_EQ(FiniteReuseProbe::sequences, (std::vector<std::uint64_t>{1U, 2U}));
  EXPECT_EQ(FiniteReuseProbe::witnesses, (std::vector<std::uint64_t>{1U, 4U}));
  EXPECT_EQ(FiniteReuseProbe::states,
            (std::vector<InputState>{InputState::kSelected, InputState::kSelected}));
}

// REQ: CHAN-131
// ReuseLast is named in the same retained-ownership sentence as LatestWithin, and the same
// depth-one side input must still "expose the newest eligible retained sample to the later
// witness". This regression starts from the real selector-owned reuse claim and proves the
// controller transaction advances it instead of pinning the sole typed slot behind the staged
// successor.
//
// A staged sample captured just after a clock witness is ineligible for that LatestWithin
// selection, so one callback may legitimately reuse the previously selected record. Selected
// records must remain fresh; reused records must preserve the exact retained value, stay inside
// the ReuseLast bound, and yield to a later fresh selection instead of pinning the sole slot.
// REQ: CHAN-136
TEST(TemporalFreshnessTest, DepthOneReuseLastAdmitsNewerSamplesWhileTheReuseClaimIsHeld) {
  bool observed = false;
  ASSERT_NO_FATAL_FAILURE((
      run_freshness_graph<kQueueDepth, MissingPolicy::kReuseLast>(kObservedActivations, observed)));
  ASSERT_TRUE(observed) << "only " << FreshnessProbe::activations.size() << " of "
                        << kObservedActivations << " consumer activations ran; "
                        << FreshnessProbe::describe();
  ASSERT_GE(FreshnessProbe::activations.size(), kObservedActivations) << FreshnessProbe::describe();
  std::optional<std::uint64_t> retained;
  std::uint64_t first_selected{};
  bool advanced{};
  std::size_t consecutive_reuse{};
  for (std::size_t index = 0U; index < kObservedActivations; ++index) {
    const ObservedActivation& activation = FreshnessProbe::activations[index];
    ASSERT_TRUE(activation.selected)
        << "activation " << index << " selected no sample; " << FreshnessProbe::describe();
    if (index != 0U) {
      EXPECT_GE(activation.value, FreshnessProbe::activations[index - 1U].value)
          << FreshnessProbe::describe();
    }
    if (activation.selection_state == InputState::kSelected) {
      const std::uint64_t lag =
          activation.published > activation.value ? activation.published - activation.value : 0U;
      if (index != 0U) {
        EXPECT_LE(lag, kMaximumFreshSampleLag) << FreshnessProbe::describe();
      }
      if (!retained) {
        first_selected = activation.value;
      } else {
        advanced = advanced || activation.value > first_selected;
      }
      retained = activation.value;
      consecutive_reuse = 0U;
      continue;
    }
    ASSERT_EQ(activation.selection_state, InputState::kReused) << FreshnessProbe::describe();
    ASSERT_TRUE(retained.has_value()) << FreshnessProbe::describe();
    EXPECT_EQ(activation.value, *retained) << FreshnessProbe::describe();
    EXPECT_LE(activation.age_ns, kReuseMaxAge.count()) << FreshnessProbe::describe();
    ++consecutive_reuse;
    constexpr std::size_t kMaximumConsecutiveReuse =
        static_cast<std::size_t>(kReuseMaxAge / kConsumerPeriod);
    EXPECT_LE(consecutive_reuse, kMaximumConsecutiveReuse) << FreshnessProbe::describe();
  }
  EXPECT_TRUE(advanced) << "the sole ReuseLast slot never advanced; " << FreshnessProbe::describe();
}

// REQ: CHAN-128
// "Larger depths retain the newest N unread survivors in arrival order." A depth-two selector
// commits and erases exactly one record per activation, leaving one claim; retiring that claim
// frees one typed slot, one staged survivor is promoted, and the cache is full again long before
// the next activation. The newest survivor available at the consumer's own causal cut is therefore
// never the one it selects.
TEST(TemporalFreshnessTest, DepthTwoDropOldestLatestWithinSelectsTheNewestCausalSample) {
  bool observed = false;
  ASSERT_NO_FATAL_FAILURE((run_freshness_graph<2U>(kObservedActivations, observed)));
  expect_steady_state_freshness(observed, kObservedActivations);
}

// REQ: CHAN-128
// Depth three is the same contract with more survivors pinned, and is kept as a separate case
// because a fix scoped to "the selector holds exactly one claim" repairs neither depth but a fix
// scoped to "exactly two" would repair only one of them.
TEST(TemporalFreshnessTest, DepthThreeDropOldestLatestWithinSelectsTheNewestCausalSample) {
  bool observed = false;
  ASSERT_NO_FATAL_FAILURE((run_freshness_graph<3U>(kObservedActivations, observed)));
  expect_steady_state_freshness(observed, kObservedActivations);
}

// REQ: CHAN-128
// "Fanout, external, cross-process, unsupported-payload, and source-private forced-HoloMQ cases
// shall preserve the same policy, depth, survivors, sequence gaps, metadata, validation, readiness,
// temporal selection, claim, retirement, reset, and backing lifetime." Forced HoloMQ is the
// cheapest internal transported scalar realization to reach from a runtime test. Its pumped
// non-blocking subscriber now drains a sample into bounded staging while the typed slot is pinned,
// which lets the same freshness reservation and handoff operate on direct and transported scalar
// ingress. This case does not stand in for external catalog imports, multi-subscriber fanout,
// cross-process construction, or Tensor-bearing transport; those retain separate parity scope.
TEST(TemporalFreshnessTest, ForcedHoloMqDepthOneDropOldestSelectsTheNewestCausalSample) {
  bool observed = false;
  ASSERT_NO_FATAL_FAILURE(
      (run_freshness_graph<kQueueDepth, MissingPolicy::kSkip, Realization::kForcedHoloMq>(
          kObservedActivations, observed)));
  expect_steady_state_freshness(observed, kObservedActivations);
}

// REQ: RUNTIME-017
// REQ: CHAN-131
// Refreshing a superseded candidate must not cost the consumer activations that would otherwise
// have selected something. This case was a regression -- it passes on main-5x and failed once the
// freshness reconcile was enabled -- and it is the end-to-end guard for three separate ways the
// release can fail to be a trade. Each is pinned deterministically elsewhere, because this case
// can only report a reintroduced defect as a lower pass rate:
//
//   the promoted refill is replaced before its readiness publishes
//       -> APromotedRefillIsNotReplacedBeforeItsReadinessPublishes
//   the refill is held but its poll abandons a contended cache lock
//       -> PollBlockingCompletesAPassAContendedPollAbandons
//   the record admitted is not the record the eviction qualified
//       -> EvictionRefillsTheCacheItEmptiedWithinTheSamePass
//
// The window each opens is about one dispatcher wake, so all three are invisible at a 20 ms
// producer and dominant once the producer period falls below that latency, which is why this case
// runs at 50 us.
//
// SafeDefault rather than Skip: it is the only missing policy that still enters compute, so a lost
// selection is counted directly instead of inferred from a gap in activation timestamps. Under
// kCoalesceLateTicks a timing gap cannot distinguish "the activation selected nothing" from "the
// tick coalesced", and this test must not fail for the second reason.
TEST(TemporalFreshnessTest, HighRateSourceDoesNotCostTheConsumerItsSelections) {
  constexpr std::size_t kHighRateActivations = 40U;
  constexpr std::int64_t kHighRateSourceUs = 50;       // far below one dispatcher wake
  constexpr std::int64_t kHighRateConsumerUs = 50000;  // 2 s of run time for 40 activations
  bool observed = false;
  ASSERT_NO_FATAL_FAILURE(
      (run_freshness_graph<kQueueDepth,
                           MissingPolicy::kSafeDefault,
                           Realization::kDirect,
                           kHighRateSourceUs,
                           kHighRateConsumerUs>(kHighRateActivations, observed)));
  ASSERT_TRUE(observed) << "only " << FreshnessProbe::activations.size() << " of "
                        << kHighRateActivations << " consumer activations ran; "
                        << FreshnessProbe::describe();
  ASSERT_GE(FreshnessProbe::activations.size(), kHighRateActivations) << FreshnessProbe::describe();

  // Key assertion: a producer this fast always has a causal sample available, so no steady-state
  // activation may report a missing record. Age is not asserted here -- the failure mode under
  // test is losing the selection outright, and the depth-one cases above already cover staleness.
  // The first activation is excluded on the same grounds as everywhere else in this file: the
  // consumer's opening tick can legitimately precede the source's first publication.
  std::size_t missing = 0U;
  for (std::size_t index = 1U; index < kHighRateActivations; ++index) {
    missing += static_cast<std::size_t>(!FreshnessProbe::activations[index].selected);
  }
  EXPECT_EQ(missing, 0U) << missing << " of " << (kHighRateActivations - 1U)
                         << " steady-state activations selected no sample from a "
                         << kHighRateSourceUs << " us producer; " << FreshnessProbe::describe();
}

// A Channel freshness eviction may recover only the binding that owns the promoted successor.
// Borrowing last_poll_incomplete() from an unrelated physical trigger makes stale status from a
// prior poll initiate a blocking call on the wrong execution source, and the current recovery path
// also discards the injected error from that call.
// REQ: RUNTIME-017
TEST(TemporalFreshnessTest, FreshnessRecoveryDoesNotPollAnUnrelatedIncompleteSource) {
  Graph graph{"temporal-freshness-source-local-recovery"};
  auto source = graph.op<FreshnessSource<50>>("source");
  auto consumer = graph.op<CrossSourceFreshnessConsumer>("consumer");
  graph.add_flow(source->output,
                 consumer->data_,
                 ConnectionOptions{.queue_policy = QueuePolicy::kDropOldest, .queue_depth = 1U});
  graph.partition("source").add(source);
  graph.partition("consumer").add(consumer);
  graph.set_default_clock(graph.add_clock<RealtimeClock>("runtime_clock"));

  CompileOptions options;
  options.deployment.bind_thread_placement("consumer",
                                           ThreadPlacement{.wake_mode = WakeMode::kBusyPoll});
  const ExecutionPlan compiled = compile(graph, std::move(options));
  ASSERT_TRUE(compiled.ok()) << compiled.json();
  const auto counters = std::make_shared<detail::RuntimePhaseCounters>();
  const ExecutionPlan with_counters =
      detail::RuntimeTestPlanAccess::with_phase_counters(compiled, counters);
  const ExecutionPlan plan = detail::RuntimeTestPlanAccess::with_port_binding_factory(
      with_counters, std::make_shared<CrossSourceIncompleteFactory>());
  ASSERT_TRUE(plan.ok()) << plan.json();

  CrossSourceRecoveryProbe::reset();
  RunSession session = run_async(plan);
  const bool completed = wait_for_terminal(session);
  if (!completed) {
    session.request_stop();
  }
  session.wait();

  ASSERT_TRUE(completed)
      << "session did not complete within " << kActivationBudget.count() << " seconds; activations="
      << CrossSourceRecoveryProbe::activation_count.load(std::memory_order_relaxed) << "/"
      << CrossSourceFreshnessConsumer::kActivationLimit << ", incomplete_status_calls="
      << CrossSourceRecoveryProbe::incomplete_status_calls.load(std::memory_order_relaxed)
      << ", blocking_poll_calls="
      << CrossSourceRecoveryProbe::blocking_poll_calls.load(std::memory_order_relaxed)
      << ", diagnostics=" << session.diagnostics().size();
  ASSERT_EQ(session.termination(), RunTermination::kRequestedCompleted);
  ASSERT_TRUE(session.diagnostics().empty());
  EXPECT_EQ(CrossSourceRecoveryProbe::activation_count.load(std::memory_order_relaxed),
            CrossSourceFreshnessConsumer::kActivationLimit);
  const auto snapshot = counters->snapshot();
  EXPECT_GT(snapshot[detail::RuntimePhaseCounters::index(
                detail::RuntimePhaseCounter::kTemporalRetentionEvictions)],
            0U);
  EXPECT_EQ(CrossSourceRecoveryProbe::incomplete_status_calls.load(std::memory_order_relaxed), 0U);
  EXPECT_EQ(CrossSourceRecoveryProbe::blocking_poll_calls.load(std::memory_order_relaxed), 0U);
}

// A blocking Channel poll can complete its call while declining the pass, for example when the
// same activation owner also has an external import whose transport mutex is busy. The promoted
// successor remains published for the next round, so that bounded miss must not terminate the run.
// REQ: RUNTIME-017
// REQ: RUNTIME-050
TEST(TemporalFreshnessTest, IncompleteFreshnessRecoveryDefersTheSuccessorWithoutFailingTheRun) {
  Graph graph{"temporal-freshness-incomplete-recovery"};
  auto source = graph.op<FreshnessSource<50>>("source");
  auto consumer = graph.op<CrossSourceFreshnessConsumer>("consumer");
  graph.add_flow(source->output,
                 consumer->data_,
                 ConnectionOptions{.queue_policy = QueuePolicy::kDropOldest, .queue_depth = 1U});
  graph.partition("source").add(source);
  graph.partition("consumer").add(consumer);
  graph.set_default_clock(graph.add_clock<RealtimeClock>("runtime_clock"));

  CompileOptions options;
  options.deployment.bind_thread_placement("consumer",
                                           ThreadPlacement{.wake_mode = WakeMode::kBusyPoll});
  const ExecutionPlan compiled = compile(graph, std::move(options));
  ASSERT_TRUE(compiled.ok()) << compiled.json();
  const auto counters = std::make_shared<detail::RuntimePhaseCounters>();
  const ExecutionPlan counted =
      detail::RuntimeTestPlanAccess::with_phase_counters(compiled, counters);
  const auto incomplete = std::make_shared<detail::RuntimeChannelIncompletePollTestControl>(1U);
  const ExecutionPlan controlled =
      detail::RuntimeTestPlanAccess::with_channel_incomplete_poll_test_control(counted, incomplete);
  const ExecutionPlan plan = detail::RuntimeTestPlanAccess::with_port_binding_factory(
      controlled, std::make_shared<CrossSourceIncompleteFactory>());
  ASSERT_TRUE(plan.ok()) << plan.json();

  CrossSourceRecoveryProbe::reset();
  RunSession session = run_async(plan);
  const bool completed = wait_for_terminal(session);
  if (!completed) {
    session.request_stop();
  }
  session.wait();

  ASSERT_TRUE(completed) << "session did not complete within " << kActivationBudget.count()
                         << " seconds; activations="
                         << CrossSourceRecoveryProbe::activation_count.load(
                                std::memory_order_relaxed)
                         << "/" << CrossSourceFreshnessConsumer::kActivationLimit
                         << ", forced_incomplete_polls=" << incomplete->forced()
                         << ", diagnostics=" << session.diagnostics().size();
  ASSERT_EQ(session.termination(), RunTermination::kRequestedCompleted);
  ASSERT_TRUE(session.diagnostics().empty());
  EXPECT_EQ(incomplete->forced(), 1U);
  EXPECT_EQ(CrossSourceRecoveryProbe::activation_count.load(std::memory_order_relaxed),
            CrossSourceFreshnessConsumer::kActivationLimit);
  const auto snapshot = counters->snapshot();
  EXPECT_GT(snapshot[detail::RuntimePhaseCounters::index(
                detail::RuntimePhaseCounter::kTemporalRetentionEvictions)],
            0U);
}

}  // namespace
}  // namespace holoscan
