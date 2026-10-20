/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Freshness of a kDropOldest scalar selection feeding a slower independent trigger.
//
// Active cases cover direct scalar depths one through three, internal transported scalar delivery,
// ReuseLast supersession, finite terminal drain, the original high-rate selection-loss regression,
// mixed-source recovery, and one CUDA Tensor parity case. External raw Tensor and descriptor-
// schema coverage remains in the dedicated external-Channel suite.

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include <cuda_runtime_api.h>

#include <fmt/format.h>
#include <fmt/ranges.h>
#include <gtest/gtest.h>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/run_session.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "compiler/compile_internal.hpp"
#include "compiler/execution_plan_state.hpp"
#include "core/source_id_internal.hpp"
#include "runtime/port_binding.hpp"
#include "runtime/run_session_internal.hpp"
#include "runtime/runtime_channel_binding.hpp"
#include "runtime/runtime_channel_binding_test_access.hpp"
#include "runtime/runtime_diagnostic_sink.hpp"
#include "runtime/runtime_phase_counters.hpp"
#include "runtime/runtime_test_access.hpp"
#include "support/host_unique_test_name.hpp"
#include "support/optional_assertions.hpp"

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

class ScopedCudaDeviceRestore final {
 public:
  explicit ScopedCudaDeviceRestore(const int device) noexcept : device_(device) {}

  ScopedCudaDeviceRestore(const ScopedCudaDeviceRestore&) = delete;
  ScopedCudaDeviceRestore& operator=(const ScopedCudaDeviceRestore&) = delete;

  ~ScopedCudaDeviceRestore() {
    if (device_ >= 0) {
      static_cast<void>(cudaSetDevice(device_));
    }
  }

 private:
  int device_{-1};
};

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
constexpr std::array<std::pair<detail::RuntimePhaseCounter, std::string_view>, 22U>
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
        {detail::RuntimePhaseCounter::kChannelFreshnessDeclinedStagedAboveFloor, "staged-above"},
    }};

using ReportedCounterValues = std::array<std::uint64_t, kReportedCounters.size()>;

// One row per consumer activation, in activation order. Keeping missing activations in the same
// sequence as selected ones is what lets an assertion skip the pre-steady-state prefix uniformly:
// before the first activation the source may legitimately have published nothing at all, and a
// separate missing counter cannot tell that startup case apart from a mid-run loss.
struct ObservedActivation {
  bool selected{};
  InputState selection_state{InputState::kAbsent};
  SelectionReason selection_reason{SelectionReason::kNone};
  std::int64_t age_ns{};
  std::uint64_t value{};
  // What the source had published when this activation ran, so freshness can be judged against
  // the producer's own progress rather than against the wall clock.
  std::uint64_t published{};
  // The trigger coordinate this activation ran at. Freshness alone cannot see a *lost* activation:
  // the run simply waits one more period and the next one looks healthy, so a dropped clock tick is
  // invisible to an age assertion. The gap between consecutive coordinates is what exposes it.
  std::int64_t activation_time_ns{};
  // DrainAll consumers only: the batch size, and whether its values rose strictly, starting above
  // every value an earlier batch delivered.
  std::size_t batch_size{};
  bool batch_ordered{true};
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
    expired_publications.store(0U, std::memory_order_relaxed);
    for (auto& completed : publication_completed_ns) {
      completed.store(0, std::memory_order_relaxed);
    }
  }

  // Runtime-clock time at which the capture-less source's emit of each value returned, indexed by
  // value. A drop-oldest DrainAll tick drains what was observed by its tick, so freshness is judged
  // against the publications complete by then; `published` read in compute() also counts the ones
  // that land between the tick's freeze and a delayed compute() or a source catching up after a
  // stall.
  static inline std::array<std::atomic<std::int64_t>, 4096U> publication_completed_ns{};

  static void record_publication(const std::uint64_t value, const std::int64_t completed_ns) {
    if (value < publication_completed_ns.size()) {
      publication_completed_ns[value].store(completed_ns, std::memory_order_relaxed);
    }
  }

  // The newest value whose emit had returned by `time_ns`; values complete in order. Call after
  // the run stops.
  [[nodiscard]] static std::uint64_t published_by(const std::int64_t time_ns) {
    std::uint64_t newest = 0U;
    for (std::uint64_t value = 1U; value < publication_completed_ns.size(); ++value) {
      const std::int64_t completed =
          publication_completed_ns[value].load(std::memory_order_relaxed);
      if (completed == 0 || completed > time_ns) {
        break;
      }
      newest = value;
    }
    return newest;
  }

  // Published count, owned here rather than by the source so the consumer can read it without the
  // two operators sharing anything else.
  static inline std::atomic<std::uint64_t> published{};
  // Blocking publications a TolerateBlockingExpiry source dropped before the stop request.
  static inline std::atomic<std::uint64_t> expired_publications{};

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
    if (const auto expired = expired_publications.load(std::memory_order_relaxed); expired != 0U) {
      fmt::format_to(std::back_inserter(buffer), "; expired publications: {}", expired);
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

// Stamp scheduled coordinates by default. Actual capture time models a camera callback that can
// publish just after a coincident consumer tick.
// TolerateBlockingExpiry treats a publication failure after a stop request as cooperative
// shutdown: a blocking recipient that has already stopped can no longer drain, so the bounded send
// expires. Before the stop request it drops and counts a publication whose blocking budget
// expired before submission: a host that deschedules a depth-one recipient for longer than that
// budget delays the recipient without stranding it. Partial or indeterminate outcomes still fail.
template <std::int64_t PeriodUs = 20000, bool CaptureNow = false, typename Payload = std::uint64_t,
          bool TolerateBlockingExpiry = false>
class FreshnessSource final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    auto& port = spec.output(output, "output");
    if constexpr (std::is_same_v<Payload, Tensor>) {
      port.produces_tensor(
          TensorOutputSpec{.representation = {.memory_kind = MemoryKind::kHost,
                                              .dtype = DLDataType{kDLUInt, 64U, 1U},
                                              .rank = 1U},
                           .bounds = tensor_bounds(sizeof(std::uint64_t)),
                           .storage = TensorOutputStorage::kRuntimePool});
    }
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = std::chrono::microseconds{PeriodUs}});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    // CaptureNow instantiations replace the activation coordinate with a fresh reading.
    // NOLINTNEXTLINE(misc-const-correctness)
    TimePoint activation = context.activation_time();
    if constexpr (CaptureNow) {
      auto captured = context.clock().now();
      if (!captured) {
        return make_unexpected(std::move(captured).error());
      }
      activation = *captured;
    }
    ++published_;
    FreshnessProbe::published.store(published_, std::memory_order_relaxed);
    if constexpr (std::is_same_v<Payload, Tensor>) {
      constexpr std::array<std::int64_t, 1U> kShape{1};
      auto loan = output.allocate_tensor(
          TensorLoanRequest{.shape = kShape, .dtype = DLDataType{kDLUInt, 64U, 1U}});
      if (!loan) {
        return make_unexpected(std::move(loan).error());
      }
      auto committed = loan->write_host_and_commit(
          [this](const TensorOutputWriteGuard& writer) -> expected<void, Error> {
            std::memcpy(writer.data(), &published_, sizeof(published_));
            return {};
          });
      if (!committed) {
        return make_unexpected(std::move(committed).error());
      }
      return output.emit(std::move(*loan), EmitOptions{.capture_time = activation});
    } else {
      auto emitted = output.emit(Sample<std::uint64_t>{
          published_,
          SampleMetadata{
              .capture_timestamp_ns = activation.timestamp_ns,
              .source_clock_id = activation.clock_id,
          },
      });
      if constexpr (TolerateBlockingExpiry) {
        if (!emitted && context.cancellation().stop_requested()) {
          return {};
        }
        if (!emitted && emitted.error().code == ErrorCode::kBackpressured) {
          FreshnessProbe::expired_publications.fetch_add(1U, std::memory_order_relaxed);
          return {};
        }
      }
      return emitted;
    }
  }

  Output<Payload> output;

 private:
  std::uint64_t published_{};
};

// Depth is a template parameter rather than a constructor argument because queue_depth is authored
// in setup(), before any instance state is available.
template <std::uint32_t Depth, MissingPolicy Missing = MissingPolicy::kSkip,
          std::int64_t PeriodUs = 250000, typename Payload = std::uint64_t,
          std::int64_t MaxAgeNs = kMaxSampleAge.count()>
class FreshnessConsumer final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    auto& port = spec.input(data, "data").queue_depth(Depth);
    if constexpr (std::is_same_v<Payload, Tensor>) {
      port.expects_tensor(TensorInputSpec{.representation = {.memory_kind = MemoryKind::kHost,
                                                             .dtype = DLDataType{kDLUInt, 64U, 1U},
                                                             .rank = 1U}});
    }
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = std::chrono::microseconds{PeriodUs}})
        .input(data, LatestWithin{.max_age = std::chrono::nanoseconds{MaxAgeNs}});
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
           .selection_reason = data.selection().reason,
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
    const std::uint64_t value = [&] {
      if constexpr (std::is_same_v<Payload, Tensor>) {
        std::uint64_t result{};
        std::memcpy(&result, sample->data.data(), sizeof(result));
        return result;
      } else {
        return sample->data;
      }
    }();
    FreshnessProbe::record({.selected = true,
                            .selection_state = data.selection().state,
                            .age_ns = age_ns,
                            .value = value,
                            .published = FreshnessProbe::published.load(std::memory_order_relaxed),
                            .activation_time_ns = context.activation_time().timestamp_ns});
    return {};
  }

  Input<Payload> data;
};

// Publish without a capture time, like a robot's joint-state reader. A side-only DrainAll input
// orders these samples by publication, independently from when its consumer polls them.
template <std::int64_t PeriodUs = 20000, typename Payload = std::uint64_t>
class CapturelessFreshnessSource final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    auto& port = spec.output(output, "output");
    if constexpr (std::is_same_v<Payload, Tensor>) {
      port.produces_tensor(
          TensorOutputSpec{.representation = {.memory_kind = MemoryKind::kHost,
                                              .dtype = DLDataType{kDLUInt, 64U, 1U},
                                              .rank = 1U},
                           .bounds = tensor_bounds(sizeof(std::uint64_t)),
                           .storage = TensorOutputStorage::kRuntimePool});
    }
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = std::chrono::microseconds{PeriodUs}});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    ++published_;
    FreshnessProbe::published.store(published_, std::memory_order_relaxed);
    const auto emitted = [&]() -> expected<void, Error> {
      if constexpr (std::is_same_v<Payload, Tensor>) {
        constexpr std::array<std::int64_t, 1U> kShape{1};
        auto loan = output.allocate_tensor(
            TensorLoanRequest{.shape = kShape, .dtype = DLDataType{kDLUInt, 64U, 1U}});
        if (!loan) {
          return make_unexpected(std::move(loan).error());
        }
        auto committed = loan->write_host_and_commit(
            [this](const TensorOutputWriteGuard& writer) -> expected<void, Error> {
              std::memcpy(writer.data(), &published_, sizeof(published_));
              return {};
            });
        if (!committed) {
          return make_unexpected(std::move(committed).error());
        }
        return output.emit(std::move(*loan));
      } else {
        return output.emit(published_);
      }
    }();
    if (auto now = context.clock().now(); emitted && now) {
      FreshnessProbe::record_publication(published_, now->timestamp_ns);
    }
    return emitted;
  }

  Output<Payload> output;

 private:
  std::uint64_t published_{};
};

// Drain a drop-oldest side input on an independent clock and record the newest value of each batch.
template <std::uint32_t Depth, std::int64_t PeriodUs = 250000, typename Payload = std::uint64_t>
class DrainAllFreshnessConsumer final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    auto& port = spec.input(data, "data").queue_depth(Depth);
    if constexpr (std::is_same_v<Payload, Tensor>) {
      port.expects_tensor(TensorInputSpec{.representation = {.memory_kind = MemoryKind::kHost,
                                                             .dtype = DLDataType{kDLUInt, 64U, 1U},
                                                             .rank = 1U}});
    }
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = std::chrono::microseconds{PeriodUs}}).input(data, DrainAll{});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    auto batch = data.selected_batch();
    if (!batch) {
      return make_unexpected(std::move(batch).error());
    }
    ObservedActivation activation{
        .selected = !batch->empty(),
        .selection_state = data.selection().state,
        .published = FreshnessProbe::published.load(std::memory_order_relaxed),
        .activation_time_ns = context.activation_time().timestamp_ns,
        .batch_size = batch->size()};
    for (const Sample<Payload>& sample : *batch) {
      const auto value = [&] {
        if constexpr (std::is_same_v<Payload, Tensor>) {
          std::uint64_t result{};
          std::memcpy(&result, sample.data.data(), sizeof(result));
          return result;
        } else {
          return sample.data;
        }
      }();
      activation.batch_ordered = activation.batch_ordered && value > delivered_;
      delivered_ = std::max(delivered_, value);
      activation.value = value;
    }
    FreshnessProbe::record(activation);
    return {};
  }

  Input<Payload> data;

 private:
  std::uint64_t delivered_{};
};

class StallingClockSource final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.output(output, "output").max_emits_per_compute(1U);
  }
  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 10ms});
    return result;
  }
  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    auto now = context.clock().now();
    if (!now) {
      return make_unexpected(std::move(now).error());
    }
    if (!start_) {
      start_ = *now;
    }
    const std::int64_t elapsed_ms = (now->timestamp_ns - start_->timestamp_ns) / 1000000;
    const SampleMetadata stamped{.capture_timestamp_ns = now->timestamp_ns,
                                 .source_clock_id = now->clock_id};
    if (!first_sent_ && elapsed_ms >= 220) {
      first_sent_ = true;
      return output.emit(Sample<std::uint64_t>{1U, stamped});
    }
    if (first_sent_ && !second_sent_ && elapsed_ms >= 480) {
      second_sent_ = true;
      return output.emit(Sample<std::uint64_t>{2U, stamped});
    }
    if (second_sent_ && !stalled_ && elapsed_ms >= 520) {
      // The controller has had ~40 ms of rounds to reconcile sample 2. Now stall it for more
      // than two consumer periods (300 ms each) so the next drained tick coalesces.
      stalled_ = true;
      std::this_thread::sleep_for(700ms);
    }
    return {};
  }
  Output<std::uint64_t> output;

 private:
  std::optional<TimePoint> start_;
  bool first_sent_{};
  bool second_sent_{};
  bool stalled_{};
};

// Publish a counter on its own clock for an OnEach trigger input.
template <std::int64_t PeriodUs>
class EachTriggerSource final : public Operator<> {
 public:
  static inline std::atomic<std::uint64_t> published{};

  void setup(OperatorSpec& spec) override { spec.output(output, "output"); }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = std::chrono::microseconds{PeriodUs}});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    return output.emit(published.fetch_add(1U, std::memory_order_relaxed) + 1U);
  }

  Output<std::uint64_t> output;
};

// Run on each trigger sample and drain a drop-oldest side input, like the robot's planner.
class EachTriggerSideDrainConsumer final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(trigger, "trigger");
    spec.input(side, "side").queue_depth(1U);
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnEach{trigger}).input(side, DrainAll{});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    auto batch = side.selected_batch();
    if (!batch) {
      return make_unexpected(std::move(batch).error());
    }
    ObservedActivation activation{
        .selected = !batch->empty(),
        .selection_state = side.selection().state,
        .published = FreshnessProbe::published.load(std::memory_order_relaxed),
        .activation_time_ns = context.activation_time().timestamp_ns,
        .batch_size = batch->size()};
    for (const Sample<std::uint64_t>& sample : *batch) {
      activation.value = sample.data;
    }
    FreshnessProbe::record(activation);
    return {};
  }

  Input<std::uint64_t> trigger;
  Input<std::uint64_t> side;
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
  kCaptureAbsentThird,
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
      phase_value(snapshot,
                  detail::RuntimePhaseCounter::kTemporalRetentionDeclinedEvictionRefused) +
      phase_value(snapshot, detail::RuntimePhaseCounter::kTemporalRetentionDeferredForCredit);
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

class FreshnessDiagnosticSink final : public detail::RuntimeStructuralDiagnosticSink {
 public:
  void publish(std::span<const Diagnostic> diagnostics) override {
    const std::lock_guard lock{mutex_};
    diagnostics_.assign(diagnostics.begin(), diagnostics.end());
  }

  void publish_runtime_details(std::span<const detail::RuntimeDiagnosticDetail> details) override {
    const std::lock_guard lock{mutex_};
    details_.assign(details.begin(), details.end());
  }

  [[nodiscard]] std::vector<Diagnostic> diagnostics() const {
    const std::lock_guard lock{mutex_};
    return diagnostics_;
  }

  [[nodiscard]] std::vector<detail::RuntimeDiagnosticDetail> details() const {
    const std::lock_guard lock{mutex_};
    return details_;
  }

 private:
  mutable std::mutex mutex_;
  std::vector<Diagnostic> diagnostics_;
  std::vector<detail::RuntimeDiagnosticDetail> details_;
};

class DecreasingArrivalBindControl final : public detail::RuntimeChannelSessionBindTestControl {
 public:
  DecreasingArrivalBindControl(const PortId input, const ClockId clock)
      : input_(input), clock_(clock) {}

  [[nodiscard]] expected<void, Error> bind(
      detail::RuntimeChannelSession& session) noexcept override {
    constexpr std::uint64_t kFirstValue = 1U;
    constexpr std::uint64_t kSecondValue = 2U;
    auto first_counter =
        detail::RuntimeChannelSessionTestAccess::set_arrival_counters(session, input_, 24U, 1U);
    if (!first_counter) {
      return first_counter;
    }
    auto first = detail::RuntimeChannelSessionTestAccess::inject_direct_inline_payload(
        session,
        input_,
        std::as_bytes(std::span{&kFirstValue, 1U}),
        SampleMetadata{.sequence = 1U,
                       .capture_timestamp_ns = 100,
                       .source_clock_id = clock_,
                       .publish_timestamp_ns = 100,
                       .publish_clock_id = clock_,
                       .byte_size = sizeof(kFirstValue),
                       .payload_kind = PayloadKind::kInlineTyped});
    if (!first) {
      return first;
    }
    auto second_counter =
        detail::RuntimeChannelSessionTestAccess::set_arrival_counters(session, input_, 23U, 2U);
    if (!second_counter) {
      return second_counter;
    }
    auto second = detail::RuntimeChannelSessionTestAccess::inject_direct_inline_payload(
        session,
        input_,
        std::as_bytes(std::span{&kSecondValue, 1U}),
        SampleMetadata{.sequence = 2U,
                       .capture_timestamp_ns = 200,
                       .source_clock_id = clock_,
                       .publish_timestamp_ns = 200,
                       .publish_clock_id = clock_,
                       .byte_size = sizeof(kSecondValue),
                       .payload_kind = PayloadKind::kInlineTyped});
    if (second) {
      configured_.store(true, std::memory_order_release);
    }
    return second;
  }

  [[nodiscard]] bool configured() const noexcept {
    return configured_.load(std::memory_order_acquire);
  }

 private:
  PortId input_{};
  ClockId clock_{};
  std::atomic<bool> configured_{};
};

class DecreasingArrivalSource final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override { spec.output(output, "output"); }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 60s});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Output<std::uint64_t> output;
};

class DecreasingArrivalConsumer final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override { spec.input(data, "data").queue_depth(4U); }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 60s})
        .input(data, LatestWithin{.max_age = 1s})
        .on_missing(data, ReuseLast{.max_age = 30s});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Input<std::uint64_t> data;
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
      if constexpr (Witness == FiniteReuseWitness::kCaptureAbsentThird) {
        if (calls_ == 2U) {
          final_data_metadata_ = metadata;
        } else if (calls_ == 3U) {
          return data.emit(Sample<std::uint64_t>{calls_, SampleMetadata{}});
        }
      }
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

// Observe retained freshness while optionally checking an unwired side for
// TemporalFreshnessTest.OptionalAbsenceCoexistsWithDepthOneReuseLastFreshness.
template <FiniteReuseSelector Selector, std::uint32_t Depth,
          FiniteReuseWitness Witness = FiniteReuseWitness::kMatching, bool WithOptional = false>
class FiniteReuseConsumer final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(data, "data").queue_depth(Depth);
    spec.input(driver, "driver").queue_depth(1U);
    if constexpr (WithOptional) {
      spec.input(optional, "optional").optional();
    }
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
    if constexpr (WithOptional) {
      result.input(optional, Exact{}).on_missing(optional, SafeDefault{});
    }
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    if constexpr (WithOptional) {
      if (optional.selection().state != InputState::kSafeDefault ||
          optional.selection().reason != SelectionReason::kNotConnected || optional.receive() ||
          optional.selected_batch()) {
        return make_unexpected(Error{ErrorCode::kProtocolError});
      }
    }
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
  Input<std::uint64_t> optional;
};

// Run finite retained delivery and collect terminal evidence for
// TemporalFreshnessTest.OptionalAbsenceCoexistsWithDepthOneReuseLastFreshness.
// Graph: source.{driver,data} --> consumer.{driver,data}; consumer.optional is unwired if enabled.
template <FiniteReuseSelector Selector, std::uint32_t Depth,
          FiniteReuseWitness Witness = FiniteReuseWitness::kMatching,
          Realization Delivery = Realization::kForcedHoloMq, bool WithOptional = false>
void run_finite_reuse_graph(
    detail::RuntimePhaseCounters::Snapshot* const final_snapshot = nullptr) {
  FiniteReuseProbe::reset();
  Graph graph{test::make_host_unique_test_name("temporal-finite-reuse-terminal-drain")};
  auto source = graph.op<FiniteReuseSource<Selector, Witness>>("source");
  auto consumer = graph.op<FiniteReuseConsumer<Selector, Depth, Witness, WithOptional>>("consumer");
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
  const ExecutionPlan compiled =
      Delivery == Realization::kForcedHoloMq
          ? detail::compile_with_process_local_holomq(graph, CompileOptions{})
          : compile(graph);
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

struct ExactDepthManyProbe {
  static void reset() {
    const std::lock_guard lock{mutex};
    values.clear();
    sequences.clear();
    states.clear();
    timed_out = false;
  }

  [[nodiscard]] static bool wait_for_activations(const std::size_t count) {
    std::unique_lock lock{mutex};
    const bool observed = cv.wait_for(
        lock, kActivationBudget, [count] { return values.size() >= count || timed_out; });
    timed_out = timed_out || !observed;
    return observed && values.size() >= count;
  }

  static void record(const Sample<std::uint64_t>& sample, const InputState state) {
    const std::lock_guard lock{mutex};
    values.push_back(sample.data);
    sequences.push_back(sample.metadata.sequence);
    states.push_back(state);
    cv.notify_all();
  }

  [[nodiscard]] static std::string describe() {
    const std::lock_guard lock{mutex};
    return fmt::format("values={} sequences={} states={} timeout={}",
                       fmt::join(values, ","),
                       fmt::join(sequences, ","),
                       states.size(),
                       timed_out);
  }

  [[nodiscard]] static bool did_time_out() {
    const std::lock_guard lock{mutex};
    return timed_out;
  }

  [[nodiscard]] static std::size_t activation_count() {
    const std::lock_guard lock{mutex};
    return values.size();
  }

  static inline std::mutex mutex;
  static inline std::condition_variable cv;
  static inline std::vector<std::uint64_t> values;
  static inline std::vector<std::uint64_t> sequences;
  static inline std::vector<InputState> states;
  static inline bool timed_out{};
};

class ExactDepthManySource final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.output(data, "data").max_emits_per_compute(2U);
    spec.output(driver, "driver").max_emits_per_compute(1U);
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 1ms});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    ++calls_;
    const ClockId clock = context.activation_time().clock_id;
    if (calls_ == 1U) {
      auto emitted = emit(data, 1U, 10, clock);
      return emitted ? emit(driver, 1U, 10, clock) : emitted;
    }
    // Let the first activation retire and establish ReuseLast before the depth-many burst.
    if (calls_ < 21U) {
      return {};
    }
    if (calls_ == 21U) {
      return emit(data, 2U, 20, clock);
    }
    // Let sample 2 occupy the second typed slot, then publish both staged survivors without a
    // controller interleave between them.
    if (calls_ < 41U) {
      return {};
    }
    if (calls_ == 41U) {
      for (std::uint64_t value = 3U; value <= 4U; ++value) {
        auto emitted = emit(data, value, static_cast<std::int64_t>(value * 10U), clock);
        if (!emitted) {
          return emitted;
        }
      }
      return {};
    }
    if (calls_ == 42U) {
      emitted_first_staged_driver_ = true;
      return emit(driver, 4U, 40, clock);
    }
    if (emitted_first_staged_driver_ && !emitted_predecessor_driver_ &&
        ExactDepthManyProbe::activation_count() >= 2U) {
      emitted_predecessor_driver_ = true;
      return emit(driver, 3U, 30, clock);
    }
    if (emitted_predecessor_driver_ && !emitted_target_driver_ &&
        ExactDepthManyProbe::activation_count() >= 3U) {
      emitted_target_driver_ = true;
      return emit(driver, 4U, 40, clock);
    }
    return {};
  }

  Output<std::uint64_t> data;
  Output<std::uint64_t> driver;

 private:
  [[nodiscard]] static expected<void, Error> emit(Output<std::uint64_t>& output,
                                                  const std::uint64_t value,
                                                  const std::int64_t capture_ns,
                                                  const ClockId clock) {
    return output.emit(Sample<std::uint64_t>{
        value,
        SampleMetadata{.capture_timestamp_ns = capture_ns, .source_clock_id = clock},
    });
  }

  std::uint64_t calls_{};
  bool emitted_first_staged_driver_{};
  bool emitted_predecessor_driver_{};
  bool emitted_target_driver_{};
};

class ExactDepthManyConsumer final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(data, "data").queue_depth(2U);
    spec.input(driver, "driver").queue_depth(1U);
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnEach{driver}).input(data, Exact{}).on_missing(data, ReuseLast{.max_age = 30s});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    auto witness = driver.receive();
    auto sample = data.receive();
    if (!witness) {
      return make_unexpected(std::move(witness).error());
    }
    if (!sample) {
      return make_unexpected(std::move(sample).error());
    }
    ExactDepthManyProbe::record(*sample, data.selection().state);
    return ExactDepthManyProbe::did_time_out()
               ? expected<void, Error>{make_unexpected(Error{ErrorCode::kTimeout})}
               : expected<void, Error>{};
  }

  Input<std::uint64_t> data;
  Input<std::uint64_t> driver;
};

template <Realization Delivery>
void run_exact_depth_many_reuse_graph() {
  ExactDepthManyProbe::reset();
  Graph graph{test::make_host_unique_test_name("temporal-exact-depth-many-reuse-order")};
  auto source = graph.op<ExactDepthManySource>("source");
  auto consumer = graph.op<ExactDepthManyConsumer>("consumer");
  graph.add_flow(source->driver,
                 consumer->driver,
                 ConnectionOptions{.queue_policy = QueuePolicy::kDropOldest, .queue_depth = 1U});
  graph.add_flow(source->data,
                 consumer->data,
                 ConnectionOptions{.queue_policy = QueuePolicy::kDropOldest, .queue_depth = 2U});
  graph.partition("source").add(source);
  graph.partition("consumer").add(consumer);
  graph.set_default_clock(graph.add_clock<RealtimeClock>("runtime_clock"));

  const ExecutionPlan compiled =
      Delivery == Realization::kForcedHoloMq
          ? detail::compile_with_process_local_holomq(graph, CompileOptions{})
          : compile(graph);
  ASSERT_TRUE(compiled.ok()) << compiled.json();
  auto counters = std::make_shared<detail::RuntimePhaseCounters>();
  const ExecutionPlan plan = detail::RuntimeTestPlanAccess::with_phase_counters(compiled, counters);
  ASSERT_TRUE(plan.ok()) << plan.json();

  RunSession session = run_async(plan);
  const bool observed = ExactDepthManyProbe::wait_for_activations(4U);
  // Leave room for ordinary promotion to expose any predecessor that was incorrectly admitted
  // after the matching target.
  const auto settle_deadline = std::chrono::steady_clock::now() + 100ms;
  while (session.is_running() && std::chrono::steady_clock::now() < settle_deadline) {
    std::this_thread::sleep_for(1ms);
  }
  if (session.is_running()) {
    session.request_stop();
  }
  session.wait();
  const auto snapshot = counters->snapshot();

  EXPECT_TRUE(observed) << ExactDepthManyProbe::describe();
  EXPECT_EQ(session.termination(), RunTermination::kStopped) << ExactDepthManyProbe::describe();
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token) << "; "
      << ExactDepthManyProbe::describe();
  EXPECT_EQ(ExactDepthManyProbe::values, (std::vector<std::uint64_t>{1U, 1U, 3U, 4U}));
  EXPECT_EQ(ExactDepthManyProbe::sequences, (std::vector<std::uint64_t>{1U, 1U, 3U, 4U}));
  EXPECT_EQ(ExactDepthManyProbe::states,
            (std::vector<InputState>{InputState::kSelected,
                                     InputState::kReused,
                                     InputState::kSelected,
                                     InputState::kSelected}));
  EXPECT_GT(phase_value(snapshot, detail::RuntimePhaseCounter::kTemporalRetentionEvictions), 0U);
}

struct MultiInputReuseOrderProbe {
  static void reset() {
    const std::lock_guard lock{mutex};
    fast_committed = 0U;
    driver_committed = 0U;
    consumer_entered = 0U;
    slow_initial_committed = false;
    release_final_consumer = false;
    barrier_timeout = false;
    driver_values.clear();
    fast_values.clear();
    slow_values.clear();
    fast_states.clear();
    slow_states.clear();
    tensor_payload_values.clear();
    tensor_payloads_valid = true;
  }

  static void publish_fast_commit(const std::uint32_t value) {
    const std::lock_guard lock{mutex};
    fast_committed = std::max(fast_committed, value);
    cv.notify_all();
  }

  static void publish_slow_initial_commit() { publish(slow_initial_committed); }

  static void publish_driver_commit(const std::uint32_t value) {
    const std::lock_guard lock{mutex};
    driver_committed = std::max(driver_committed, value);
    cv.notify_all();
  }

  [[nodiscard]] static bool wait_for_initial_commits() {
    return wait_for([] { return fast_committed >= 1U && slow_initial_committed; });
  }

  [[nodiscard]] static bool wait_for_consumer(const std::uint32_t value) {
    return wait_for([value] { return consumer_entered >= value; });
  }

  [[nodiscard]] static bool wait_for_fast_commit(const std::uint32_t value) {
    return wait_for([value] { return fast_committed >= value; });
  }

  [[nodiscard]] static bool wait_for_driver_commit(const std::uint32_t value) {
    return wait_for([value] { return driver_committed >= value; });
  }

  static void record(const Sample<std::uint64_t>& driver, const Sample<std::uint64_t>& fast,
                     const InputState fast_state, const Sample<std::uint64_t>& slow,
                     const InputState slow_state) {
    record_values(driver.data, fast.data, fast_state, slow.data, slow_state);
  }

  static void record_values(const std::uint64_t driver_value, const std::uint64_t fast_value,
                            const InputState fast_state, const std::uint64_t slow_value,
                            const InputState slow_state) {
    std::unique_lock lock{mutex};
    driver_values.push_back(driver_value);
    fast_values.push_back(fast_value);
    slow_values.push_back(slow_value);
    fast_states.push_back(fast_state);
    slow_states.push_back(slow_state);
    consumer_entered = std::max(consumer_entered, static_cast<std::uint32_t>(driver_value));
    cv.notify_all();
    if (driver_value < 5U) {
      if (!cv.wait_for(lock, kActivationBudget, [driver_value] {
            return driver_committed >= static_cast<std::uint32_t>(driver_value + 1U);
          })) {
        barrier_timeout = true;
      }
    } else if (!cv.wait_for(lock, kActivationBudget, [] { return release_final_consumer; })) {
      barrier_timeout = true;
    }
  }

  static void record_tensor_payload(const std::uint8_t value, const bool valid) {
    const std::lock_guard lock{mutex};
    tensor_payload_values.push_back(value);
    tensor_payloads_valid = tensor_payloads_valid && valid;
  }

  [[nodiscard]] static bool wait_for_activations(const std::size_t count) {
    std::unique_lock lock{mutex};
    return cv.wait_for(lock, kActivationBudget, [count] {
      return driver_values.size() >= count || barrier_timeout;
    }) && driver_values.size() >= count;
  }

  static void release_all() {
    const std::lock_guard lock{mutex};
    fast_committed = 5U;
    driver_committed = 5U;
    consumer_entered = 5U;
    slow_initial_committed = true;
    release_final_consumer = true;
    cv.notify_all();
  }

  [[nodiscard]] static std::string describe() {
    const std::lock_guard lock{mutex};
    return fmt::format("driver={} fast={} slow={} fast_states={} slow_states={} timeout={}",
                       fmt::join(driver_values, ","),
                       fmt::join(fast_values, ","),
                       fmt::join(slow_values, ","),
                       fast_states.size(),
                       slow_states.size(),
                       barrier_timeout);
  }

  [[nodiscard]] static bool timed_out() {
    const std::lock_guard lock{mutex};
    return barrier_timeout;
  }

  static inline std::mutex mutex;
  static inline std::condition_variable cv;
  static inline std::uint32_t fast_committed{};
  static inline std::uint32_t driver_committed{};
  static inline std::uint32_t consumer_entered{};
  static inline bool slow_initial_committed{};
  static inline bool release_final_consumer{};
  static inline bool barrier_timeout{};
  static inline std::vector<std::uint64_t> driver_values;
  static inline std::vector<std::uint64_t> fast_values;
  static inline std::vector<std::uint64_t> slow_values;
  static inline std::vector<std::uint64_t> tensor_payload_values;
  static inline std::vector<InputState> fast_states;
  static inline std::vector<InputState> slow_states;
  static inline bool tensor_payloads_valid{true};

 private:
  static void publish(bool& event) {
    const std::lock_guard lock{mutex};
    event = true;
    cv.notify_all();
  }

  template <typename Predicate>
  [[nodiscard]] static bool wait_for(Predicate&& predicate) {
    std::unique_lock lock{mutex};
    const bool ready = cv.wait_for(lock, kActivationBudget, std::forward<Predicate>(predicate));
    barrier_timeout = barrier_timeout || !ready;
    return ready;
  }
};

enum class MultiInputFastPayload : std::uint8_t {
  kScalar,
  kCudaTensor,
};

class MultiInputFastSource final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override { spec.output(output, "output"); }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 1ms});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    ++calls_;
    if (calls_ > 5U) {
      return {};
    }
    if (calls_ > 1U && !MultiInputReuseOrderProbe::wait_for_consumer(calls_ - 1U)) {
      return make_unexpected(Error{ErrorCode::kTimeout});
    }
    const std::int64_t capture_ns = calls_ == 1U ? 10 : 80 + calls_ * 10;
    auto emitted = output.emit(Sample<std::uint64_t>{
        calls_,
        SampleMetadata{.capture_timestamp_ns = capture_ns,
                       .source_clock_id = context.activation_time().clock_id}});
    if (emitted) {
      MultiInputReuseOrderProbe::publish_fast_commit(calls_);
    }
    return emitted;
  }

  Output<std::uint64_t> output;

 private:
  std::uint32_t calls_{};
};

[[nodiscard]] expected<void, Error> publish_multi_input_cuda_byte(Output<Tensor>& output,
                                                                  ExecutionContext& context,
                                                                  const std::uint8_t value,
                                                                  const TimePoint capture_time) {
  constexpr std::array<std::int64_t, 1U> kShape{1};
  auto loan = output.allocate_tensor(TensorLoanRequest{
      .shape = kShape,
      .dtype = DLDataType{kDLUInt, 8U, 1U},
  });
  if (!loan) {
    return make_unexpected(std::move(loan).error());
  }
  auto* const stream = context.cuda_stream().get();
  if (stream == nullptr) {
    return make_unexpected(Error{ErrorCode::kNotSupported});
  }
  auto committed = loan->write_and_commit(
      context.cuda_stream(),
      [stream, value](const TensorOutputWriteGuard& writer) -> expected<void, Error> {
        if (cudaMemsetAsync(writer.data(), value, 1U, stream) != cudaSuccess) {
          return make_unexpected(Error{ErrorCode::kFailure});
        }
        return {};
      });
  if (!committed) {
    return make_unexpected(std::move(committed).error());
  }
  return output.emit(std::move(*loan), EmitOptions{.capture_time = capture_time});
}

class MultiInputCudaFastSource final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.output(output, "output")
        .max_emits_per_compute(1U)
        .produces_tensor(TensorOutputSpec{
            .representation = {.memory_kind = MemoryKind::kCudaDevice,
                               .dtype = DLDataType{kDLUInt, 8U, 1U},
                               .rank = 1U},
            .bounds = tensor_bounds(1U),
            .storage = TensorOutputStorage::kRuntimePool,
        });
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 1ms});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    ++calls_;
    if (calls_ > 5U) {
      return {};
    }
    if (calls_ > 1U && !MultiInputReuseOrderProbe::wait_for_consumer(calls_ - 1U)) {
      return make_unexpected(Error{ErrorCode::kTimeout});
    }
    const std::int64_t capture_ns = calls_ == 1U ? 10 : 80 + calls_ * 10;
    auto emitted = publish_multi_input_cuda_byte(
        output,
        context,
        static_cast<std::uint8_t>(calls_),
        TimePoint{.timestamp_ns = capture_ns, .clock_id = context.activation_time().clock_id});
    if (emitted) {
      MultiInputReuseOrderProbe::publish_fast_commit(calls_);
    }
    return emitted;
  }

  Output<Tensor> output;

 private:
  std::uint32_t calls_{};
};

class MultiInputSlowSource final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override { spec.output(output, "output"); }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 1ms});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    ++calls_;
    if (calls_ != 1U) {
      return {};
    }
    auto emitted = output.emit(Sample<std::uint64_t>{
        1U,
        SampleMetadata{.capture_timestamp_ns = 10,
                       .source_clock_id = context.activation_time().clock_id}});
    if (emitted) {
      MultiInputReuseOrderProbe::publish_slow_initial_commit();
    }
    return emitted;
  }

  Output<std::uint64_t> output;

 private:
  std::uint64_t calls_{};
};

class MultiInputDriverSource final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override { spec.output(output, "output"); }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 1ms});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    ++calls_;
    if (calls_ > 5U) {
      return {};
    }
    if (calls_ == 1U ? !MultiInputReuseOrderProbe::wait_for_initial_commits()
                     : !MultiInputReuseOrderProbe::wait_for_fast_commit(calls_)) {
      return make_unexpected(Error{ErrorCode::kTimeout});
    }
    const std::int64_t capture_ns = calls_ == 1U ? 15 : calls_ == 5U ? 200 : calls_ * 10;
    auto emitted = emit(calls_, capture_ns, context.activation_time().clock_id);
    if (emitted) {
      MultiInputReuseOrderProbe::publish_driver_commit(calls_);
    }
    return emitted;
  }

  Output<std::uint64_t> output;

 private:
  [[nodiscard]] expected<void, Error> emit(const std::uint64_t value, const std::int64_t capture_ns,
                                           const ClockId clock) {
    return output.emit(Sample<std::uint64_t>{
        value, SampleMetadata{.capture_timestamp_ns = capture_ns, .source_clock_id = clock}});
  }

  std::uint32_t calls_{};
};

template <std::uint32_t Depth>
class MultiInputReuseOrderConsumer final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(fast, "fast").queue_depth(Depth);
    spec.input(slow, "slow").queue_depth(Depth);
    spec.input(driver, "driver").queue_depth(1U);
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnEach{driver})
        .input(fast, LatestWithin{.max_age = 1s})
        .on_missing(fast, ReuseLast{.max_age = 30s})
        .input(slow, LatestWithin{.max_age = 1s})
        .on_missing(slow, ReuseLast{.max_age = 30s});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    auto driver_sample = driver.receive();
    auto fast_sample = fast.receive();
    auto slow_sample = slow.receive();
    if (!driver_sample) {
      return make_unexpected(std::move(driver_sample).error());
    }
    if (!fast_sample) {
      return make_unexpected(std::move(fast_sample).error());
    }
    if (!slow_sample) {
      return make_unexpected(std::move(slow_sample).error());
    }
    MultiInputReuseOrderProbe::record(
        *driver_sample, *fast_sample, fast.selection().state, *slow_sample, slow.selection().state);
    return MultiInputReuseOrderProbe::timed_out()
               ? expected<void, Error>{make_unexpected(Error{ErrorCode::kTimeout})}
               : expected<void, Error>{};
  }

  Input<std::uint64_t> fast;
  Input<std::uint64_t> slow;
  Input<std::uint64_t> driver;
};

template <std::uint32_t Depth>
class MultiInputCudaReuseOrderConsumer final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(fast, "fast")
        .queue_depth(Depth)
        .expects_tensor(TensorInputSpec{
            .representation = {.memory_kind = MemoryKind::kCudaDevice,
                               .dtype = DLDataType{kDLUInt, 8U, 1U},
                               .rank = 1U},
        });
    spec.input(slow, "slow").queue_depth(Depth);
    spec.input(driver, "driver").queue_depth(1U);
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnEach{driver})
        .input(fast, LatestWithin{.max_age = 1s})
        .on_missing(fast, ReuseLast{.max_age = 30s})
        .input(slow, LatestWithin{.max_age = 1s})
        .on_missing(slow, ReuseLast{.max_age = 30s});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    auto driver_sample = driver.receive();
    auto fast_sample = fast.receive();
    auto slow_sample = slow.receive();
    if (!driver_sample) {
      return make_unexpected(std::move(driver_sample).error());
    }
    if (!fast_sample) {
      return make_unexpected(std::move(fast_sample).error());
    }
    if (!slow_sample) {
      return make_unexpected(std::move(slow_sample).error());
    }

    std::uint8_t copied_value{};
    const bool backing_valid =
        fast_sample->data.data() != nullptr && fast_sample->data.nbytes() == 1U;
    bool copied = false;
    if (backing_valid) {
      auto copied_result = fast_sample->data.copy_to_host(
          &copied_value, sizeof(copied_value), context.cuda_stream());
      copied = copied_result.has_value();
    }
    MultiInputReuseOrderProbe::record_tensor_payload(
        copied_value, backing_valid && copied && fast_sample->metadata.sequence == copied_value);
    MultiInputReuseOrderProbe::record_values(driver_sample->data,
                                             fast_sample->metadata.sequence,
                                             fast.selection().state,
                                             slow_sample->data,
                                             slow.selection().state);
    return MultiInputReuseOrderProbe::timed_out()
               ? expected<void, Error>{make_unexpected(Error{ErrorCode::kTimeout})}
               : expected<void, Error>{};
  }

  Input<Tensor> fast;
  Input<std::uint64_t> slow;
  Input<std::uint64_t> driver;
};

template <std::uint32_t Depth, Realization Delivery = Realization::kDirect,
          MultiInputFastPayload FastPayload = MultiInputFastPayload::kScalar>
void run_multi_input_reuse_order_graph() {
  MultiInputReuseOrderProbe::reset();
  Graph graph{test::make_host_unique_test_name("temporal-multi-input-reuse-order")};
  const ClockKey runtime_clock = graph.add_clock<RealtimeClock>("runtime_clock");
  using FastSource = std::conditional_t<FastPayload == MultiInputFastPayload::kCudaTensor,
                                        MultiInputCudaFastSource,
                                        MultiInputFastSource>;
  using Consumer = std::conditional_t<FastPayload == MultiInputFastPayload::kCudaTensor,
                                      MultiInputCudaReuseOrderConsumer<Depth>,
                                      MultiInputReuseOrderConsumer<Depth>>;
  auto fast = graph.op<FastSource>("fast");
  auto slow = graph.op<MultiInputSlowSource>("slow");
  auto driver = graph.op<MultiInputDriverSource>("driver");
  auto consumer = graph.op<Consumer>("consumer");

  const ConnectionOptions side_options{.queue_policy = QueuePolicy::kDropOldest,
                                       .queue_depth = Depth};
  graph.add_flow(fast->output, consumer->fast, side_options);
  graph.add_flow(slow->output, consumer->slow, side_options);
  graph.add_flow(driver->output,
                 consumer->driver,
                 ConnectionOptions{.queue_policy = QueuePolicy::kDropOldest, .queue_depth = 1U});
  graph.partition("fast").add(fast);
  graph.partition("slow").add(slow);
  graph.partition("driver").add(driver);
  graph.partition("consumer").add(consumer);
  graph.set_default_clock(runtime_clock);

  const ExecutionPlan compiled = [&] {
    if constexpr (FastPayload == MultiInputFastPayload::kCudaTensor) {
      static_assert(Delivery == Realization::kDirect);
      CompileOptions options;
      options.deployment.bind_tensor_output_device(TensorOutputDevicePlacement{
          .operator_path = "fast",
          .output_port = "output",
          .device = DeviceId{0},
      });
      return compile(graph, std::move(options));
    } else if constexpr (Delivery == Realization::kForcedHoloMq) {
      return detail::compile_with_process_local_holomq(graph, CompileOptions{});
    } else {
      return compile(graph);
    }
  }();
  ASSERT_TRUE(compiled.ok()) << compiled.json();
  if constexpr (FastPayload == MultiInputFastPayload::kCudaTensor) {
    const auto plan_state = detail::ExecutionPlanAccess::retain(compiled);
    ASSERT_TRUE(plan_state);
    const auto tensor_service =
        std::find_if(plan_state->process_local_channel_services.begin(),
                     plan_state->process_local_channel_services.end(),
                     [](const auto& service) { return service.tensor_payload; });
    ASSERT_NE(tensor_service, plan_state->process_local_channel_services.end());
    EXPECT_EQ(std::count_if(plan_state->process_local_channel_services.begin(),
                            plan_state->process_local_channel_services.end(),
                            [](const auto& service) { return service.tensor_payload; }),
              1U);
    EXPECT_EQ(tensor_service->delivery_kind,
              detail::ProcessLocalChannelDeliveryKind::kDirectRetained);
    EXPECT_EQ(tensor_service->tensor_memory_kind, MemoryKind::kCudaDevice);
    EXPECT_EQ(tensor_service->tensor_device, DeviceId{0});
  } else if constexpr (Delivery == Realization::kForcedHoloMq) {
    const auto plan_state = detail::ExecutionPlanAccess::retain(compiled);
    ASSERT_TRUE(plan_state);
    ASSERT_EQ(plan_state->process_local_channel_services.size(), 3U);
    EXPECT_TRUE(std::all_of(plan_state->process_local_channel_services.begin(),
                            plan_state->process_local_channel_services.end(),
                            [](const auto& service) {
                              return service.delivery_kind ==
                                     detail::ProcessLocalChannelDeliveryKind::kHoloMQ;
                            }));
  }
  auto counters = std::make_shared<detail::RuntimePhaseCounters>();
  const ExecutionPlan plan = detail::RuntimeTestPlanAccess::with_phase_counters(compiled, counters);
  ASSERT_TRUE(plan.ok()) << plan.json();

  RunSession session = run_async(plan);
  const bool observed = MultiInputReuseOrderProbe::wait_for_activations(5U);
  if (session.is_running()) {
    session.request_stop();
  }
  MultiInputReuseOrderProbe::release_all();
  session.wait();
  const detail::RuntimePhaseCounters::Snapshot snapshot = counters->snapshot();

  EXPECT_TRUE(observed) << MultiInputReuseOrderProbe::describe();
  EXPECT_EQ(session.termination(), RunTermination::kStopped)
      << MultiInputReuseOrderProbe::describe();
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token) << "; "
      << MultiInputReuseOrderProbe::describe();
  EXPECT_FALSE(MultiInputReuseOrderProbe::timed_out()) << MultiInputReuseOrderProbe::describe();
  EXPECT_EQ(MultiInputReuseOrderProbe::driver_values,
            (std::vector<std::uint64_t>{1U, 2U, 3U, 4U, 5U}));
  EXPECT_EQ(MultiInputReuseOrderProbe::fast_values,
            (std::vector<std::uint64_t>{1U, 1U, 1U, 1U, 5U}));
  EXPECT_EQ(MultiInputReuseOrderProbe::slow_values,
            (std::vector<std::uint64_t>{1U, 1U, 1U, 1U, 1U}));
  EXPECT_EQ(MultiInputReuseOrderProbe::fast_states,
            (std::vector<InputState>{InputState::kSelected,
                                     InputState::kReused,
                                     InputState::kReused,
                                     InputState::kReused,
                                     InputState::kSelected}));
  EXPECT_EQ(MultiInputReuseOrderProbe::slow_states,
            (std::vector<InputState>{InputState::kSelected,
                                     InputState::kReused,
                                     InputState::kReused,
                                     InputState::kReused,
                                     InputState::kReused}));
  if constexpr (FastPayload == MultiInputFastPayload::kCudaTensor) {
    EXPECT_TRUE(MultiInputReuseOrderProbe::tensor_payloads_valid)
        << MultiInputReuseOrderProbe::describe();
    EXPECT_EQ(MultiInputReuseOrderProbe::tensor_payload_values,
              (std::vector<std::uint64_t>{1U, 1U, 1U, 1U, 5U}));
  }
  if constexpr (Depth > 1U) {
    EXPECT_GE(phase_value(snapshot, detail::RuntimePhaseCounter::kTemporalRetentionEvictions), 1U);
    if constexpr (Delivery == Realization::kDirect) {
      EXPECT_GE(
          phase_value(snapshot, detail::RuntimePhaseCounter::kTemporalRetentionEvictionPromoted),
          1U);
      EXPECT_GE(phase_value(snapshot, detail::RuntimePhaseCounter::kChannelDirectIngressPromotions),
                1U);
    } else {
      EXPECT_GE(
          phase_value(snapshot, detail::RuntimePhaseCounter::kChannelReleasedClaimRefillPublished),
          1U);
    }
  }
}

struct AllLatestClockObservation {
  std::array<std::uint64_t, 3U> values{};
  std::array<InputState, 3U> states{};
};

struct AllLatestClockProbe {
  static void reset() {
    const std::lock_guard lock{mutex};
    observations.clear();
  }

  static void record(AllLatestClockObservation observation) {
    const std::lock_guard lock{mutex};
    observations.push_back(std::move(observation));
    cv.notify_all();
  }

  [[nodiscard]] static bool wait_for(const std::size_t count) {
    std::unique_lock lock{mutex};
    return cv.wait_for(lock, kActivationBudget, [count] { return observations.size() >= count; });
  }

  [[nodiscard]] static std::vector<AllLatestClockObservation> snapshot() {
    const std::lock_guard lock{mutex};
    return observations;
  }

  static inline std::mutex mutex;
  static inline std::condition_variable cv;
  static inline std::vector<AllLatestClockObservation> observations;
};

class AllLatestClockSource final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.output(layer0, "layer0").max_emits_per_compute(1U);
    spec.output(fast, "fast").max_emits_per_compute(1U);
    spec.output(slow, "slow").max_emits_per_compute(1U);
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 10ms});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    ++calls_;
    const TimePoint activation = context.activation_time();
    auto emitted = emit(fast, calls_, activation);
    if (!emitted) {
      return emitted;
    }
    if ((calls_ - 1U) % 3U == 0U) {
      emitted = emit(layer0, calls_, activation);
      if (!emitted) {
        return emitted;
      }
    }
    if ((calls_ - 1U) % 5U == 0U) {
      emitted = emit(slow, calls_, activation);
    }
    return emitted;
  }

  Output<std::uint64_t> layer0;
  Output<std::uint64_t> fast;
  Output<std::uint64_t> slow;

 private:
  [[nodiscard]] static expected<void, Error> emit(Output<std::uint64_t>& output,
                                                  const std::uint64_t value,
                                                  const TimePoint capture) {
    return output.emit(
        Sample<std::uint64_t>{value,
                              SampleMetadata{.capture_timestamp_ns = capture.timestamp_ns,
                                             .source_clock_id = capture.clock_id}});
  }

  std::uint64_t calls_{};
};

class AllLatestClockConsumer final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(layer0, "layer0").queue_depth(4U);
    spec.input(fast, "fast").queue_depth(4U);
    spec.input(slow, "slow").queue_depth(4U);
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 33ms})
        .input(layer0, LatestWithin{.max_age = 25ms})
        .on_missing(layer0, ReuseLast{.max_age = 500ms})
        .input(fast, LatestWithin{.max_age = 25ms})
        .on_missing(fast, ReuseLast{.max_age = 500ms})
        .input(slow, LatestWithin{.max_age = 25ms})
        .on_missing(slow, ReuseLast{.max_age = 500ms});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    auto layer0_sample = layer0.receive();
    auto fast_sample = fast.receive();
    auto slow_sample = slow.receive();
    if (!layer0_sample) {
      return make_unexpected(std::move(layer0_sample).error());
    }
    if (!fast_sample) {
      return make_unexpected(std::move(fast_sample).error());
    }
    if (!slow_sample) {
      return make_unexpected(std::move(slow_sample).error());
    }
    AllLatestClockProbe::record(AllLatestClockObservation{
        .values = {layer0_sample->data, fast_sample->data, slow_sample->data},
        .states = {layer0.selection().state, fast.selection().state, slow.selection().state},
    });
    return {};
  }

  Input<std::uint64_t> layer0;
  Input<std::uint64_t> fast;
  Input<std::uint64_t> slow;
};

void run_all_latest_clock_graph() {
  constexpr std::size_t kWantedActivations = 6U;
  AllLatestClockProbe::reset();
  Graph graph{test::make_host_unique_test_name("all-latest-clock-refill-order")};
  auto source = graph.op<AllLatestClockSource>("source");
  auto consumer = graph.op<AllLatestClockConsumer>("consumer");
  const ConnectionOptions options{.queue_policy = QueuePolicy::kDropOldest, .queue_depth = 4U};
  graph.add_flow(source->layer0, consumer->layer0, options);
  graph.add_flow(source->fast, consumer->fast, options);
  graph.add_flow(source->slow, consumer->slow, options);
  graph.partition("source").add(source);
  graph.partition("consumer").add(consumer);
  graph.set_default_clock(graph.add_clock<RealtimeClock>("runtime_clock"));

  const ExecutionPlan compiled = compile(graph);
  ASSERT_TRUE(compiled.ok()) << compiled.json();
  auto counters = std::make_shared<detail::RuntimePhaseCounters>();
  const ExecutionPlan plan = detail::RuntimeTestPlanAccess::with_phase_counters(compiled, counters);
  ASSERT_TRUE(plan.ok()) << plan.json();

  RunSession session = run_async(plan);
  const bool observed = AllLatestClockProbe::wait_for(kWantedActivations);
  if (session.is_running()) {
    session.request_stop();
  }
  session.wait();
  const auto observations = AllLatestClockProbe::snapshot();
  const auto snapshot = counters->snapshot();

  ASSERT_TRUE(observed) << "observed " << observations.size() << " clock activations";
  ASSERT_GE(observations.size(), kWantedActivations);
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
  bool slow_reused = false;
  for (std::size_t index = 0U; index < kWantedActivations; ++index) {
    for (std::size_t input = 0U; input < observations[index].states.size(); ++input) {
      EXPECT_TRUE(observations[index].states[input] == InputState::kSelected ||
                  observations[index].states[input] == InputState::kReused);
      if (index != 0U) {
        EXPECT_GE(observations[index].values[input], observations[index - 1U].values[input]);
        if (observations[index].states[input] == InputState::kReused) {
          EXPECT_EQ(observations[index].values[input], observations[index - 1U].values[input]);
        }
      }
      slow_reused =
          slow_reused || (input == 2U && observations[index].states[input] == InputState::kReused);
    }
  }
  EXPECT_GT(observations[kWantedActivations - 1U].values[1], observations[0].values[1]);
  EXPECT_TRUE(slow_reused);
  EXPECT_GT(phase_value(snapshot, detail::RuntimePhaseCounter::kTemporalRetentionEvictions), 0U);
}

// Whether the compiler is allowed to pick direct process-local retention, or is forced onto the
// source-private HoloMQ realization. CHAN-128 requires both to preserve the same temporal
// selection, so every freshness property below has to hold for either.
template <typename SourceOperator, typename ConsumerOperator, std::uint32_t Depth,
          Realization Delivery, QueuePolicy Policy = QueuePolicy::kDropOldest>
void run_source_consumer_graph(std::size_t wanted_activations, bool& observed) {
  Graph graph{"temporal-freshness-drop-oldest"};
  auto source = graph.op<SourceOperator>("source");
  auto consumer = graph.op<ConsumerOperator>("consumer");
  graph.add_flow(source->output,
                 consumer->data,
                 ConnectionOptions{.queue_policy = Policy, .queue_depth = Depth});

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

template <std::uint32_t Depth, MissingPolicy Missing = MissingPolicy::kSkip,
          Realization Delivery = Realization::kDirect, std::int64_t SourceUs = 20000,
          std::int64_t ConsumerUs = 250000, bool CaptureNow = false,
          typename Payload = std::uint64_t>
void run_freshness_graph(std::size_t wanted_activations, bool& observed) {
  run_source_consumer_graph<FreshnessSource<SourceUs, CaptureNow, Payload>,
                            FreshnessConsumer<Depth, Missing, ConsumerUs, Payload>,
                            Depth,
                            Delivery>(wanted_activations, observed);
}

// Every steady-state activation must select a sample no older than a few source periods. The first
// is excluded because before any steady state exists the source may legitimately have published
// only its very first value.
void expect_steady_state_freshness(bool observed, std::size_t wanted,
                                   std::int64_t consumer_period_ns = kConsumerPeriod.count(),
                                   std::uint64_t maximum_sample_lag = kMaximumFreshSampleLag) {
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
      EXPECT_LE(lag, maximum_sample_lag)
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

// Keeps up with every sample so the fanout family's other recipient never paces the source.
class FreshnessPeerSink final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override { spec.input(data, "data"); }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnEach{data});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    auto sample = data.receive();
    if (!sample) {
      return make_unexpected(std::move(sample).error());
    }
    return {};
  }

  Input<std::uint64_t> data;
};

// A 1 ms source fans out on blocking edges to a 2 ms LatestWithin consumer on a depth-one edge and
// a fast peer. The fanout family lowers to eventless transport, so the private emit-time experiment
// admits it. With concurrent admission, admission and post-compute cuts skip the consumer's head
// while it is known to be blocked; only the freshness reconcile can hand the newer staged sample to
// the selector. Its handoff releases capacity on unknown subscribers and takes the complete cut.
// Abandoned-reservation recovery retries the input's own subscribers. The consumer period stays
// well inside the 5 ms graph blocking budget: a slower clock consumer would fail the source with
// kPublicationIndeterminate in either admission mode.
//
// The peer only makes the family a fanout. Its deep edge keeps a descheduled peer from pacing the
// source or adding its own known-blocked skips. A loaded host can still deschedule the consumer
// past the budget, so the source drops and counts such pre-submit expiries instead of failing the
// run. A stranded consumer head would show up as stale or missing selections, which the freshness
// assertions reject.
constexpr std::int64_t kBlockingFreshnessSourceUs = 1000;
constexpr std::int64_t kBlockingFreshnessConsumerUs = 2000;
constexpr std::uint32_t kBlockingFreshnessPeerDepth = 64U;

template <MissingPolicy Missing>
void run_blocking_fanout_freshness_graph(const bool concurrent_admission,
                                         const std::size_t wanted_activations, bool& observed,
                                         detail::RuntimePhaseCounters::Snapshot& snapshot) {
  Graph graph{test::make_host_unique_test_name("temporal-blocking-fanout-freshness")};
  auto source =
      graph.op<FreshnessSource<kBlockingFreshnessSourceUs, false, std::uint64_t, true>>("source");
  auto consumer =
      graph.op<FreshnessConsumer<kQueueDepth, Missing, kBlockingFreshnessConsumerUs>>("consumer");
  auto peer = graph.op<FreshnessPeerSink>("peer");
  graph.add_flow(
      source->output,
      consumer->data,
      ConnectionOptions{.queue_policy = QueuePolicy::kBlocking, .queue_depth = kQueueDepth});
  graph.add_flow(source->output,
                 peer->data,
                 ConnectionOptions{.queue_policy = QueuePolicy::kBlocking,
                                   .queue_depth = kBlockingFreshnessPeerDepth});
  graph.partition("source").add(source);
  graph.partition("consumer").add(consumer);
  graph.partition("peer").add(peer);
  graph.set_default_clock(graph.add_clock<RealtimeClock>("runtime_clock"));

  ExecutionPlan plan = detail::compile_with_process_local_fanout_experiment(
      graph, CompileOptions{}, detail::ProcessLocalFanoutExperiment::kEmitRecipients);
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto state = detail::ExecutionPlanAccess::retain(plan);
  ASSERT_TRUE(state);
  ASSERT_EQ(state->process_local_channel_services.size(), 1U);
  EXPECT_EQ(state->process_local_channel_services.front().delivery_kind,
            detail::ProcessLocalChannelDeliveryKind::kHoloMQ);
  if (concurrent_admission) {
    auto control = std::make_shared<detail::RuntimeFanoutTestControl>();
    control->concurrent_admission = true;
    plan = detail::RuntimeTestPlanAccess::with_fanout_test_control(plan, std::move(control));
  }
  auto counters = std::make_shared<detail::RuntimePhaseCounters>();
  plan = detail::RuntimeTestPlanAccess::with_phase_counters(plan, counters);
  ASSERT_TRUE(plan.ok()) << plan.json();

  FreshnessProbe::reset(counters);
  RunSession session = run_async(plan);
  observed = FreshnessProbe::wait_for_activations(wanted_activations);
  session.request_stop();
  session.wait();
  snapshot = counters->snapshot();

  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
}

constexpr std::size_t kBlockingFreshnessActivations = 40U;

// Selections advance and stay within a few source periods. Tick gaps are not asserted: at a 2 ms
// period they depend on load, as expect_steady_state_freshness() explains.
void expect_blocking_fanout_freshness(const bool observed) {
  ASSERT_TRUE(observed) << "only " << FreshnessProbe::activations.size() << " of "
                        << kBlockingFreshnessActivations << " consumer activations ran; "
                        << FreshnessProbe::describe();
  std::uint64_t previous = 0U;
  for (std::size_t index = 1U; index < kBlockingFreshnessActivations; ++index) {
    const ObservedActivation& activation = FreshnessProbe::activations[index];
    EXPECT_TRUE(activation.selected)
        << "activation " << index << " selected no sample; " << FreshnessProbe::describe();
    if (!activation.selected) {
      continue;
    }
    EXPECT_GE(activation.value, previous) << FreshnessProbe::describe();
    previous = activation.value;
    const std::uint64_t lag =
        activation.published > activation.value ? activation.published - activation.value : 0U;
    EXPECT_LE(lag, kMaximumFreshSampleLag)
        << "activation " << index << " selected sample " << activation.value << " while the "
        << "source had published " << activation.published << "; " << FreshnessProbe::describe();
  }
}

// Concurrent admission must not strand a LatestWithin consumer behind a head it skips as known
// blocked. Each activation still selects a sample within a few source periods, exactly as with
// the emit-time experiment under the channel scan lock, and the run stops cleanly.
// REQ: RUNTIME-017
// REQ: CHAN-136
TEST(TemporalFreshnessTest, ConcurrentAdmissionKeepsBlockingFanoutLatestWithinFresh) {
  for (const bool concurrent : {false, true}) {
    SCOPED_TRACE(::testing::Message() << "concurrent_admission=" << concurrent);
    bool observed = false;
    detail::RuntimePhaseCounters::Snapshot snapshot{};
    ASSERT_NO_FATAL_FAILURE((run_blocking_fanout_freshness_graph<MissingPolicy::kSkip>(
        concurrent, kBlockingFreshnessActivations, observed, snapshot)));
    expect_blocking_fanout_freshness(observed);
    if (concurrent) {
      // Key assertion: heads really were skipped as known blocked, yet freshness held.
      EXPECT_GE(phase_value(snapshot, detail::RuntimePhaseCounter::kChannelKnownBlockedSkips), 1U);
    }
  }
}

// The same guarantee for a ReuseLast input, whose staged successor is replacement-capable and so
// is never skipped as known blocked.
// REQ: RUNTIME-017
// REQ: CHAN-136
TEST(TemporalFreshnessTest, ConcurrentAdmissionKeepsBlockingFanoutReuseLastFresh) {
  for (const bool concurrent : {false, true}) {
    SCOPED_TRACE(::testing::Message() << "concurrent_admission=" << concurrent);
    bool observed = false;
    detail::RuntimePhaseCounters::Snapshot snapshot{};
    ASSERT_NO_FATAL_FAILURE((run_blocking_fanout_freshness_graph<MissingPolicy::kReuseLast>(
        concurrent, kBlockingFreshnessActivations, observed, snapshot)));
    expect_blocking_fanout_freshness(observed);
    // Key assertion: the reconcile handed the retained record off to newer samples, the path whose
    // retry releases capacity on unknown subscribers.
    EXPECT_GE(phase_value(snapshot, detail::RuntimePhaseCounter::kTemporalRetentionEvictions), 1U)
        << FreshnessProbe::describe();
    if (concurrent) {
      // Key assertion: the replacement-capable ReuseLast subscriber is never skipped.
      EXPECT_EQ(phase_value(snapshot, detail::RuntimePhaseCounter::kChannelKnownBlockedSkips), 0U);
    }
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

// REQ: CHAN-146
// Actual capture can put the newest publication just after the current tick. The authored next
// tick may advance a separate causal candidate, while preserving the exact ReuseLast fallback.
TEST(TemporalFreshnessTest, DepthOneClockReuseLastSelectsRecentActualCapture) {
  bool observed = false;
  ASSERT_NO_FATAL_FAILURE((
      run_freshness_graph<1U, MissingPolicy::kReuseLast, Realization::kDirect, 33333, 200000, true>(
          12U, observed)));
  expect_steady_state_freshness(observed, 12U, 200000000, 3U);
}

// REQ: CHAN-146
TEST(TemporalFreshnessTest, TransportedTensorClockReuseLastSelectsRecentActualCapture) {
  bool observed = false;
  ASSERT_NO_FATAL_FAILURE((run_freshness_graph<1U,
                                               MissingPolicy::kReuseLast,
                                               Realization::kForcedHoloMq,
                                               33333,
                                               200000,
                                               true,
                                               Tensor>(12U, observed)));
  expect_steady_state_freshness(observed, 12U, 200000000, 6U);
}

// REQ: CHAN-146
// Once the source stops publishing, a coalesced tick can make the waiting candidate too old.
// The last selected payload must remain independently reusable.
TEST(TemporalFreshnessTest, CoalescedClockTickPreservesTheLastSelectedFallback) {
  FreshnessProbe::reset();
  Graph graph{test::make_host_unique_test_name("clock-overrun-reuse-fallback")};
  auto source = graph.op<StallingClockSource>("source");
  auto consumer =
      graph.op<FreshnessConsumer<1U, MissingPolicy::kReuseLast, 300000, std::uint64_t, 150000000>>(
          "consumer");
  graph.add_flow(source->output,
                 consumer->data,
                 ConnectionOptions{.queue_policy = QueuePolicy::kDropOldest, .queue_depth = 1U});
  // Both callbacks share the controller, so the source's stall coalesces the consumer clock.
  graph.partition("inline").include_all();
  graph.set_default_clock(graph.add_clock<RealtimeClock>("runtime_clock"));
  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  RunSession session = run_async(plan);
  const bool observed = FreshnessProbe::wait_for_activations(3U);
  session.request_stop();
  session.wait();
  ASSERT_TRUE(observed) << FreshnessProbe::describe();
  ASSERT_TRUE(session.diagnostics().empty());
  ASSERT_GE(FreshnessProbe::activations.size(), 3U);
  const auto& before = FreshnessProbe::activations[0];
  const auto& after = FreshnessProbe::activations[1];
  EXPECT_EQ(before.selection_state, InputState::kSelected);
  EXPECT_EQ(before.value, 1U);
  EXPECT_GE(after.activation_time_ns - before.activation_time_ns, 600000000);
  EXPECT_EQ(after.selection_state, InputState::kReused) << FreshnessProbe::describe();
  EXPECT_EQ(after.value, 1U) << FreshnessProbe::describe();
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

// REQ: CHAN-128
// REQ: RUNTIME-017
// REQ: RUNTIME-043
// Exact cannot skip an older staged producer record to admit a later matching record. The first
// witness reuses prior data; later matching witnesses select the staged records in producer order.
TEST(TemporalFreshnessTest, DirectDepthTwoExactReuseLastDoesNotOvertakeStagedPredecessor) {
  ASSERT_NO_FATAL_FAILURE((run_exact_depth_many_reuse_graph<Realization::kDirect>()));
}

// REQ: CHAN-128
// REQ: RUNTIME-017
// REQ: RUNTIME-043
TEST(TemporalFreshnessTest, ForcedHoloMqDepthTwoExactReuseLastMatchesDirectHeadOrdering) {
  ASSERT_NO_FATAL_FAILURE((run_exact_depth_many_reuse_graph<Realization::kForcedHoloMq>()));
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
// REQ: CHAN-128
// REQ: CHAN-146
// Select 1; publish 2 then 3; trigger at 2's capture time. The private reuse claim keeps 1 while
// the authored typed position admits 2, so future sample 3 cannot replace the causal predecessor.
TEST(TemporalFreshnessTest, DepthOneReuseLastPreservesCausalPredecessorBehindFutureSample) {
  ASSERT_NO_FATAL_FAILURE((run_finite_reuse_graph<FiniteReuseSelector::kLatestWithin,
                                                  1U,
                                                  FiniteReuseWitness::kIneligibleOnly,
                                                  Realization::kDirect>()));
  EXPECT_EQ(FiniteReuseProbe::values, (std::vector<std::uint64_t>{1U, 2U}));
  ASSERT_NO_FATAL_FAILURE((run_finite_reuse_graph<FiniteReuseSelector::kLatestWithin,
                                                  1U,
                                                  FiniteReuseWitness::kIneligibleOnly>()));
  EXPECT_EQ(FiniteReuseProbe::values, (std::vector<std::uint64_t>{1U, 2U}));
}

// REQ: CHAN-131
// REQ: CHAN-146
TEST(TemporalFreshnessTest, DepthOneReuseLastAdvancesWhenNewestSampleIsCausal) {
  ASSERT_NO_FATAL_FAILURE((run_finite_reuse_graph<FiniteReuseSelector::kLatestWithin,
                                                  1U,
                                                  FiniteReuseWitness::kMatching,
                                                  Realization::kDirect>()));
  EXPECT_EQ(FiniteReuseProbe::values, (std::vector<std::uint64_t>{1U, 3U}));
}

// REQ: CHAN-146
// REQ: RUNTIME-055
TEST(TemporalFreshnessTest, OptionalAbsenceCoexistsWithDepthOneReuseLastFreshness) {
  // Key setup: add an unwired optional side to a depth-one ReuseLast flow on both delivery paths.
  // Key assertion: absence preserves fresh values {1, 3} for direct and forced HoloMQ delivery.
  // Graph:
  //   source.driver --> consumer.driver
  //   source.data   --> consumer.data [depth 1, DropOldest, ReuseLast]
  //   [unwired]         consumer.optional
  ASSERT_NO_FATAL_FAILURE((run_finite_reuse_graph<FiniteReuseSelector::kLatestWithin,
                                                  1U,
                                                  FiniteReuseWitness::kMatching,
                                                  Realization::kDirect,
                                                  true>()));
  EXPECT_EQ(FiniteReuseProbe::values, (std::vector<std::uint64_t>{1U, 3U}));
  ASSERT_NO_FATAL_FAILURE((run_finite_reuse_graph<FiniteReuseSelector::kLatestWithin,
                                                  1U,
                                                  FiniteReuseWitness::kMatching,
                                                  Realization::kForcedHoloMq,
                                                  true>()));
  EXPECT_EQ(FiniteReuseProbe::values, (std::vector<std::uint64_t>{1U, 3U}));
}

// REQ: CHAN-146
TEST(TemporalFreshnessTest, CaptureAbsentArrivalPreservesTheCausalCacheCandidate) {
  ASSERT_NO_FATAL_FAILURE((run_finite_reuse_graph<FiniteReuseSelector::kLatestWithin,
                                                  1U,
                                                  FiniteReuseWitness::kCaptureAbsentThird,
                                                  Realization::kDirect>()));
  EXPECT_EQ(FiniteReuseProbe::values, (std::vector<std::uint64_t>{1U, 2U}));
  ASSERT_NO_FATAL_FAILURE((run_finite_reuse_graph<FiniteReuseSelector::kLatestWithin,
                                                  1U,
                                                  FiniteReuseWitness::kCaptureAbsentThird>()));
  EXPECT_EQ(FiniteReuseProbe::values, (std::vector<std::uint64_t>{1U, 2U}));
}

// REQ: RUNTIME-017
// The temporal cache remains fail-closed for a genuinely decreasing same-epoch arrival. Preserve
// the registered public token, but retain the exact typed error and witness identity internally so
// the diagnostic cannot misidentify this injected protocol violation as a missing port binding.
TEST(TemporalFreshnessTest, DecreasingArrivalRetainsTypedDiagnosticAtRunSessionBoundary) {
  Graph graph{"decreasing-arrival-diagnostic"};
  auto source = graph.op<DecreasingArrivalSource>("source");
  auto consumer = graph.op<DecreasingArrivalConsumer>("consumer");
  graph.add_flow(source->output,
                 consumer->data,
                 ConnectionOptions{.queue_policy = QueuePolicy::kDropOldest, .queue_depth = 4U});
  graph.partition("source").add(source);
  graph.partition("consumer").add(consumer);
  graph.set_default_clock(graph.add_clock<RealtimeClock>("runtime_clock"));

  const ExecutionPlan compiled = compile(graph);
  ASSERT_TRUE(compiled.ok()) << compiled.json();
  const auto input = std::ranges::find_if(compiled.ports(), [](const auto& port) {
    return port.local_name == "data" && port.direction == PortDirection::kInput;
  });
  ASSERT_NE(input, compiled.ports().end());
  const auto selection =
      std::ranges::find(compiled.input_selections(), input->id, &InputSelectionPlanRecord::input);
  ASSERT_NE(selection, compiled.input_selections().end());
  const auto plan_state = detail::ExecutionPlanAccess::retain(compiled);
  ASSERT_TRUE(plan_state);
  ASSERT_EQ(plan_state->process_local_channel_services.size(), 1U);
  ASSERT_EQ(plan_state->process_local_channel_services.front().delivery_kind,
            detail::ProcessLocalChannelDeliveryKind::kDirectRetained);

  auto sink = std::make_shared<FreshnessDiagnosticSink>();
  auto control =
      std::make_shared<DecreasingArrivalBindControl>(input->id, selection->reference_clock);
  const ExecutionPlan observed =
      detail::RuntimeTestPlanAccess::with_diagnostic_sink(compiled, sink);
  const ExecutionPlan plan =
      detail::RuntimeTestPlanAccess::with_channel_session_bind_test_control(observed, control);
  ASSERT_TRUE(plan.ok()) << plan.json();

  RunSession session = run_async(plan);
  const bool terminal = wait_for_terminal(session);
  if (!terminal) {
    session.request_stop();
  }
  session.wait();
  ASSERT_TRUE(terminal);
  ASSERT_TRUE(control->configured());
  EXPECT_EQ(session.termination(), RunTermination::kFailed);

  const auto diagnostics = sink->diagnostics();
  const auto details = sink->details();
  ASSERT_EQ(diagnostics.size(), details.size());
  const auto primary = std::ranges::find(
      diagnostics, std::string_view{"RUNTIME_PORT_BINDING_UNAVAILABLE"}, &Diagnostic::token);
  ASSERT_NE(primary, diagnostics.end());
  EXPECT_EQ(
      std::ranges::count(
          diagnostics, std::string_view{"RUNTIME_PORT_BINDING_UNAVAILABLE"}, &Diagnostic::token),
      1U);
  EXPECT_EQ(std::ranges::count(diagnostics,
                               std::string_view{"RUNTIME_READINESS_BINDING_UNAVAILABLE"},
                               &Diagnostic::token),
            0U);
  EXPECT_EQ(primary->source.vertex, input->owner);
  const std::size_t primary_index =
      static_cast<std::size_t>(std::distance(diagnostics.begin(), primary));
  ASSERT_LT(primary_index, details.size());
  EXPECT_TRUE(details[primary_index].present);
  EXPECT_EQ(details[primary_index].error.code, ErrorCode::kInvalidArgument);
  EXPECT_TRUE(details[primary_index].source);
  EXPECT_EQ(details[primary_index].source_sequence, 2U);
  EXPECT_EQ(details[primary_index].reference_time, (TimePoint{200, selection->reference_clock}));
}

// REQ: CHAN-128
// REQ: RUNTIME-043
// REQ: RUNTIME-051
// A depth-many drop-oldest side input can have an older readiness candidate already pending while
// freshness refill promotes a newer staged successor for the same ReuseLast claim. The exact
// prerequisite path must not admit that successor ahead of the older same-epoch prefix. The
// barriers hold real operator activations and prove the replacement, below-floor decline, and
// later handoff without injecting Channel or selector state.
TEST(TemporalFreshnessTest, DepthFourMultiInputReuseLastRefillDoesNotOvertakeOlderPendingArrival) {
  ASSERT_NO_FATAL_FAILURE((run_multi_input_reuse_order_graph<4U>()));
}

// REQ: CHAN-128
// Depth-two and depth-three queues have the same survivor-order contract as depth four; exercise
// both explicitly rather than inferring immunity from a finite timing run.
TEST(TemporalFreshnessTest, DepthTwoMultiInputReuseLastRefillPreservesArrivalOrder) {
  ASSERT_NO_FATAL_FAILURE((run_multi_input_reuse_order_graph<2U>()));
}

// REQ: CHAN-128
TEST(TemporalFreshnessTest, DepthThreeMultiInputReuseLastRefillPreservesArrivalOrder) {
  ASSERT_NO_FATAL_FAILURE((run_multi_input_reuse_order_graph<3U>()));
}

TEST(TemporalFreshnessTest, FreshnessPrerequisitePolicyDistinguishesWindowSideInputs) {
  constexpr PortId kAnchor{1U};
  constexpr PortId kSideInput{2U};
  constexpr SourceId kSharedSource = detail::make_source_id(1U);
  constexpr SourceId kIndependentSource = detail::make_source_id(2U);
  constexpr InputSelectionPlanRecord kAnchorSelection{
      .input = kAnchor,
      .selector = InputSelectorKind::kExact,
  };
  constexpr InputSelectionPlanRecord kBracketSelection{
      .input = kSideInput,
      .selector = InputSelectorKind::kBracketedWindow,
      .missing = TemporalRecordMissingKind::kSafeDefault,
  };
  constexpr InputSelectionPlanRecord kFreshnessSelection{
      .input = kSideInput,
      .selector = InputSelectorKind::kLatestWithin,
      .missing = TemporalRecordMissingKind::kReuseLast,
  };

  EXPECT_FALSE(detail::freshness_input_is_nontrigger_prerequisite(
      kAnchorSelection, kAnchor, kSharedSource, kSharedSource));
  EXPECT_FALSE(detail::freshness_input_is_nontrigger_prerequisite(
      kBracketSelection, kAnchor, kSharedSource, kSharedSource));
  EXPECT_TRUE(detail::freshness_input_is_nontrigger_prerequisite(
      kFreshnessSelection, kAnchor, kSharedSource, kSharedSource));
  EXPECT_TRUE(detail::freshness_input_is_nontrigger_prerequisite(
      kFreshnessSelection, PortId{}, kIndependentSource, kSharedSource));
  EXPECT_FALSE(detail::freshness_input_is_nontrigger_prerequisite(
      kFreshnessSelection, PortId{}, kSharedSource, kSharedSource));
}

TEST(TemporalFreshnessTest, UpcomingClockFreshnessPolicyIsNarrow) {
  constexpr InputSelectionPlanRecord kDepthOneLatestReuse{
      .selector = InputSelectorKind::kLatestWithin,
      .cache_capacity = 1U,
      .missing = TemporalRecordMissingKind::kReuseLast,
  };
  constexpr InputSelectionPlanRecord kDepthOneExactReuse{
      .selector = InputSelectorKind::kExact,
      .cache_capacity = 1U,
      .missing = TemporalRecordMissingKind::kReuseLast,
  };
  constexpr InputSelectionPlanRecord kDepthTwoLatestReuse{
      .selector = InputSelectorKind::kLatestWithin,
      .cache_capacity = 2U,
      .missing = TemporalRecordMissingKind::kReuseLast,
  };

  EXPECT_TRUE(detail::qualifies_upcoming_clock_freshness(kDepthOneLatestReuse, true, true));
  EXPECT_FALSE(detail::qualifies_upcoming_clock_freshness(kDepthOneLatestReuse, false, true));
  EXPECT_FALSE(detail::qualifies_upcoming_clock_freshness(kDepthOneExactReuse, true, true));
  EXPECT_FALSE(detail::qualifies_upcoming_clock_freshness(kDepthTwoLatestReuse, true, true));
  EXPECT_FALSE(detail::qualifies_upcoming_clock_freshness(kDepthOneLatestReuse, true, false));
}

// REQ: CHAN-128
// Source-private HoloMQ must preserve the same multi-input refill and selector ordering as direct
// retained delivery, including the identity returned by transported replacement.
TEST(TemporalFreshnessTest, ForcedHoloMqMultiInputReuseLastRefillPreservesArrivalOrder) {
  ASSERT_NO_FATAL_FAILURE((run_multi_input_reuse_order_graph<4U, Realization::kForcedHoloMq>()));
}

// REQ: CHAN-128
// REQ: RUNTIME-043
// Match the reported Holoviz-style contract: an independent clock drives the consumer and every
// layer is LatestWithin + ReuseLast. The scenario covers the faster, near-rate, and
// slower inputs together while preserving nondecreasing selected payload identities.
TEST(TemporalFreshnessTest, ClockDrivenAllLatestWithinReuseLastInputsPreserveArrivalOrder) {
  ASSERT_NO_FATAL_FAILURE(run_all_latest_clock_graph());
}

// REQ: CHAN-128
// REQ: RUNTIME-043
// Preserve the same ordering and ReuseLast guarantees when the fast side input owns a
// deployment-bound CUDA Tensor. The payload copy also proves that the selected and reused
// sequence identities retain their live device backing through the handoff.
TEST(TemporalFreshnessTest,
     DepthFourMultiInputCudaTensorReuseLastRefillDoesNotOvertakeOlderPendingArrival) {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    GTEST_SKIP() << "CUDA device unavailable";
  }
  int original_device = -1;
  ASSERT_EQ(cudaGetDevice(&original_device), cudaSuccess);
  const ScopedCudaDeviceRestore restore_device{original_device};
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  ASSERT_NO_FATAL_FAILURE(
      (run_multi_input_reuse_order_graph<4U,
                                         Realization::kDirect,
                                         MultiInputFastPayload::kCudaTensor>()));
}

// REQ: CHAN-128
// The identical synchronized graph at depth one has no older survivor for a refill prerequisite
// to overtake. It is the negative control for the production-path interleaving above, not a
// workaround encoded into the expected result.
TEST(TemporalFreshnessTest, DepthOneMultiInputReuseLastRefillHasNoOlderPendingArrival) {
  ASSERT_NO_FATAL_FAILURE((run_multi_input_reuse_order_graph<1U>()));
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
// REQ: CHAN-146
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

// A scripted physical trigger holds its owner after A/B have been admitted and retired as
// non-trigger arrivals. While that lane is held, the independent producer stages C/D. Publishing
// T in this same poll exposes the single-handoff bug without relying on dispatcher timing.
struct DrainCutProbe {
  static inline std::atomic<detail::RuntimeChannelSession*> channels{};
  static inline PortId input;
  static inline bool late_poll{};
  static inline std::atomic<std::int64_t> tail_published_at{};
  static inline bool capture_preserved{true};
  static inline std::atomic<bool> stage_tail{};
  static inline std::atomic<bool> timed_out{};
  static inline std::atomic<std::uint32_t> zero_credit_passes{};
  static inline std::mutex mutex;
  static inline std::condition_variable cv;
  static inline std::vector<std::vector<std::uint64_t>> batches;
};

class DrainCutBindControl final : public detail::RuntimeChannelSessionBindTestControl {
 public:
  explicit DrainCutBindControl(bool exhaust_credit) : exhausted_(!exhaust_credit) {}
  expected<void, Error> bind(detail::RuntimeChannelSession& session) noexcept override {
    DrainCutProbe::channels.store(&session, std::memory_order_release);
    return {};
  }
  std::size_t freshness_refill_credit(std::size_t credit) noexcept override {
    if (DrainCutProbe::stage_tail.load(std::memory_order_acquire) && !exhausted_) {
      exhausted_ = true;
      DrainCutProbe::zero_credit_passes.fetch_add(1U, std::memory_order_relaxed);
      return 0U;
    }
    return credit;
  }

 private:
  bool exhausted_{};
};

template <bool Captured>
class DrainCutSource final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override { spec.output(output, "output"); }
  Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 1ms});
    return result;
  }
  expected<void, Error> compute(ExecutionContext& context) override {
    if (published_ == 4U) {
      return {};
    }
    if (published_ == 2U && !DrainCutProbe::stage_tail.load(std::memory_order_acquire)) {
      return {};
    }
    ++published_;
    if constexpr (Captured) {
      return output.emit(Sample<std::uint64_t>{
          published_,
          SampleMetadata{.capture_timestamp_ns = static_cast<std::int64_t>(published_),
                         .source_clock_id = context.activation_time().clock_id}});
    } else {
      auto emitted = output.emit(published_);
      if (!emitted) {
        return emitted;
      }
      if (published_ == 4U) {
        auto now = context.clock().now();
        if (!now) {
          return make_unexpected(std::move(now).error());
        }
        DrainCutProbe::tail_published_at.store(now->timestamp_ns, std::memory_order_release);
      }
      return {};
    }
  }
  Output<std::uint64_t> output;

 private:
  std::uint64_t published_{};
};

class DrainCutConsumer final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.spin_source(tick, "tick").capacity(1U);
    spec.input(data, "data").queue_depth(2U);
  }
  Contract contract() const override {
    Contract result;
    result.trigger(OnSpin{.event = tick}).input(data, DrainAll{});
    return result;
  }
  expected<void, Error> compute(ExecutionContext&) override {
    auto selected = data.selected_batch();
    if (!selected) {
      return make_unexpected(std::move(selected).error());
    }
    const std::lock_guard lock{DrainCutProbe::mutex};
    std::vector<std::uint64_t> values;
    for (const auto& sample : *selected) {
      values.push_back(sample.data);
      if (DrainCutProbe::late_poll) {
        DrainCutProbe::capture_preserved =
            DrainCutProbe::capture_preserved && !sample.metadata.capture_time().valid();
      }
    }
    DrainCutProbe::batches.push_back(std::move(values));
    DrainCutProbe::cv.notify_all();
    return {};
  }
  SpinSource tick;
  Input<std::uint64_t> data;
};

class DrainCutBinding final : public detail::PortBinding {
 public:
  explicit DrainCutBinding(detail::RuntimePortBindingRequest request) : request_(request) {}
  expected<void, Error> poll(TimePoint observed_at) noexcept override {
    if (pending_ || outstanding_ || sequence_ > 2U) {
      return {};
    }
    auto* channels = DrainCutProbe::channels.load(std::memory_order_acquire);
    if (channels == nullptr) {
      return {};
    }
    if (sequence_ == 1U) {
      auto inventory = detail::RuntimeChannelSessionTestAccess::input_reset_state(
          *channels, DrainCutProbe::input);
      if (!inventory || inventory->retired_temporal_samples != 2U) {
        return {};
      }
      DrainCutProbe::stage_tail.store(true, std::memory_order_release);
      const auto deadline = std::chrono::steady_clock::now() + kActivationBudget;
      const auto staged = [&]() {
        return inventory && inventory->pending_direct_ingress_samples == 2U &&
               (!DrainCutProbe::late_poll ||
                DrainCutProbe::tail_published_at.load(std::memory_order_acquire) != 0);
      };
      do {
        inventory = detail::RuntimeChannelSessionTestAccess::input_reset_state(
            *channels, DrainCutProbe::input);
        if (staged()) {
          break;
        }
        std::this_thread::yield();
      } while (std::chrono::steady_clock::now() < deadline);
      if (!staged()) {
        DrainCutProbe::timed_out.store(true, std::memory_order_release);
        return make_unexpected(Error{ErrorCode::kFailure});
      }
      // A synthetic witness bounds the cut without sleeping until a real tick. By default it is
      // far in the future, beyond any capture-less reservation's mutex-serialized clock reading.
      // A late poll instead places it just after D's publication, so it precedes this poll and
      // any later clock reading.
      trigger_ =
          TimePoint{DrainCutProbe::late_poll
                        ? DrainCutProbe::tail_published_at.load(std::memory_order_acquire) + 1
                        : observed_at.timestamp_ns + 1000000000,
                    observed_at.clock_id};
    }
    event_ = detail::RuntimeEvent{.kind = request_.kind,
                                  .source = request_.source,
                                  .port = request_.port,
                                  .source_decision_time = trigger_,
                                  .reference_time = trigger_,
                                  .source_generation = request_.source_generation,
                                  .source_sequence = sequence_};
    pending_ = true;
    return {};
  }
  std::optional<detail::PortBindingPollTarget> poll_target() const noexcept override {
    return detail::PortBindingPollTarget{.execution_lane = request_.execution_lane,
                                         .reference_clock = request_.reference_clock};
  }
  SourceId source() const noexcept override { return request_.source; }
  expected<std::size_t, Error> drain(
      std::span<detail::RuntimeEvent> destination) noexcept override {
    if (!pending_ || destination.empty()) {
      return 0U;
    }
    destination.front() = event_;
    pending_ = false;
    outstanding_ = true;
    return 1U;
  }
  expected<void, Error> retire(std::uint64_t sequence) noexcept override {
    if (!outstanding_ || sequence != sequence_) {
      return make_unexpected(Error{ErrorCode::kStaleEndpoint});
    }
    outstanding_ = false;
    ++sequence_;
    ++trigger_.timestamp_ns;
    return {};
  }
  void cancel() noexcept override { closed_ = true; }
  std::size_t capacity() const noexcept override { return request_.capacity; }
  bool closed() const noexcept override { return closed_; }

 private:
  detail::RuntimePortBindingRequest request_;
  detail::RuntimeEvent event_;
  TimePoint trigger_;
  std::uint64_t sequence_{1U};
  bool pending_{};
  bool outstanding_{};
  bool closed_{};
};

class DrainCutFactory final : public detail::RuntimePortBindingFactory {
 public:
  expected<std::unique_ptr<detail::PortBinding>, Error> bind(
      const detail::RuntimePortBindingRequest& request, TimePoint) noexcept override {
    return std::unique_ptr<detail::PortBinding>{std::make_unique<DrainCutBinding>(request)};
  }
};

template <bool Captured>
void run_drain_cut_graph(bool forced_holomq, bool defer_recovery, bool late_poll = false) {
  DrainCutProbe::channels.store(nullptr, std::memory_order_release);
  DrainCutProbe::late_poll = late_poll;
  DrainCutProbe::tail_published_at.store(0, std::memory_order_release);
  DrainCutProbe::capture_preserved = true;
  DrainCutProbe::stage_tail.store(false, std::memory_order_release);
  DrainCutProbe::timed_out.store(false, std::memory_order_release);
  DrainCutProbe::batches.clear();
  DrainCutProbe::zero_credit_passes.store(0U, std::memory_order_relaxed);
  Graph graph{test::make_host_unique_test_name("drain-finite-survivor-cut")};
  auto source = graph.op<DrainCutSource<Captured>>("source");
  auto consumer = graph.op<DrainCutConsumer>("consumer");
  graph.add_flow(source->output,
                 consumer->data,
                 ConnectionOptions{.queue_policy = QueuePolicy::kDropOldest, .queue_depth = 2U});
  graph.partition("source").add(source);
  graph.partition("consumer").add(consumer);
  graph.set_default_clock(graph.add_clock<RealtimeClock>("clock"));
  CompileOptions options;
  options.deployment.bind_thread_placement("consumer",
                                           ThreadPlacement{.wake_mode = WakeMode::kBusyPoll});
  ExecutionPlan plan = forced_holomq
                           ? detail::compile_with_process_local_holomq(graph, std::move(options))
                           : compile(graph, std::move(options));
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto input = std::ranges::find_if(plan.ports(), [](const auto& port) {
    return port.local_name == "data" && port.direction == PortDirection::kInput;
  });
  ASSERT_NE(input, plan.ports().end());
  DrainCutProbe::input = input->id;
  plan = detail::RuntimeTestPlanAccess::with_port_binding_factory(
      plan, std::make_shared<DrainCutFactory>());
  plan = detail::RuntimeTestPlanAccess::with_channel_session_bind_test_control(
      plan, std::make_shared<DrainCutBindControl>(defer_recovery));
  const auto incomplete =
      std::make_shared<detail::RuntimeChannelIncompletePollTestControl>(defer_recovery ? 1U : 0U);
  plan = detail::RuntimeTestPlanAccess::with_channel_incomplete_poll_test_control(plan, incomplete);
  const auto counters = std::make_shared<detail::RuntimePhaseCounters>();
  plan = detail::RuntimeTestPlanAccess::with_phase_counters(plan, counters);
  RunSession session = run_async(plan);
  bool observed = false;
  {
    std::unique_lock lock{DrainCutProbe::mutex};
    observed = DrainCutProbe::cv.wait_for(
        lock, kActivationBudget, [] { return DrainCutProbe::batches.size() >= 2U; });
  }
  session.request_stop();
  session.wait();
  ASSERT_TRUE(observed);
  EXPECT_FALSE(DrainCutProbe::timed_out.load(std::memory_order_acquire));
  EXPECT_TRUE(session.diagnostics().empty());
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  ASSERT_GE(DrainCutProbe::batches.size(), 2U);
  EXPECT_EQ(DrainCutProbe::batches[0], (std::vector<std::uint64_t>{3U, 4U}));
  EXPECT_TRUE(DrainCutProbe::batches[1].empty());
  EXPECT_TRUE(DrainCutProbe::capture_preserved);
  EXPECT_EQ(incomplete->forced(), defer_recovery ? 1U : 0U);
  EXPECT_EQ(DrainCutProbe::zero_credit_passes.load(std::memory_order_relaxed),
            defer_recovery ? 1U : 0U);
  // The zero-credit pass kept the cut reserved instead of letting the trigger through.
  const auto snapshot = counters->snapshot();
  EXPECT_EQ(phase_value(snapshot, detail::RuntimePhaseCounter::kTemporalRetentionDeferredForCredit),
            defer_recovery ? 1U : 0U);
  expect_retention_counter_identity(snapshot);
}

// REQ: CHAN-128
TEST(TemporalFreshnessTest, DepthTwoDrainAllFinishesTheFiniteCapturedCutBeforeItsTrigger) {
  for (const bool forced : {false, true}) {
    for (const bool deferred : {false, true}) {
      SCOPED_TRACE(fmt::format("forced={} deferred={}", forced, deferred));
      ASSERT_NO_FATAL_FAILURE(run_drain_cut_graph<true>(forced, deferred));
    }
  }
}

// The same finite cut with capture-less samples and a trigger far after them. Each member
// qualifies by its own coordinate: its publication coordinate where startup orders the input by
// publication, otherwise the reading that qualified the cut's newest member.
// REQ: CHAN-128
TEST(TemporalFreshnessTest, DepthTwoDrainAllFinishesTheFiniteCaptureAbsentCutBeforeItsTrigger) {
  for (const bool forced : {false, true}) {
    for (const bool deferred : {false, true}) {
      SCOPED_TRACE(fmt::format("forced={} deferred={}", forced, deferred));
      ASSERT_NO_FATAL_FAILURE(run_drain_cut_graph<false>(forced, deferred, false));
    }
  }
}

// REQ: RUNTIME-065
// REQ: CHAN-131
// Startup must opt the compiled capture-less DrainAll input into publication ordering. T is
// strictly after C/D publication, but before the subsequent poll and refill-clock reading.
// Ordering at either later observation would leave the first callback draining stale A/B.
TEST(TemporalFreshnessTest, PublicationOrderedDrainAllSelectsTheCaptureAbsentCutOnALatePoll) {
  for (const bool forced : {false, true}) {
    for (const bool deferred : {false, true}) {
      SCOPED_TRACE(fmt::format("forced={} deferred={}", forced, deferred));
      ASSERT_NO_FATAL_FAILURE(run_drain_cut_graph<false>(forced, deferred, true));
    }
  }
}

// Every steady-state DrainAll batch holds at most the authored depth, rises strictly, and continues
// above everything delivered before it, and the run refreshed its retained record at least once:
// by a freshness trade, or, for an owner that parks until its tick, by replacing an unwitnessed
// record before the tick.
void expect_ordered_drain_batches(std::size_t wanted, std::size_t depth) {
  ASSERT_GE(FreshnessProbe::activations.size(), wanted) << FreshnessProbe::describe();
  for (std::size_t index = 1U; index < wanted; ++index) {
    const ObservedActivation& activation = FreshnessProbe::activations[index];
    EXPECT_LE(activation.batch_size, depth)
        << "activation " << index << "; " << FreshnessProbe::describe();
    EXPECT_TRUE(activation.batch_ordered)
        << "activation " << index << " repeated or reordered a sample; "
        << FreshnessProbe::describe();
  }
  constexpr auto kEvictions =
      std::ranges::find(kReportedCounters,
                        detail::RuntimePhaseCounter::kTemporalRetentionEvictions,
                        &std::pair<detail::RuntimePhaseCounter, std::string_view>::first);
  constexpr auto kPreWitness =
      std::ranges::find(kReportedCounters,
                        detail::RuntimePhaseCounter::kChannelPreWitnessReplacements,
                        &std::pair<detail::RuntimePhaseCounter, std::string_view>::first);
  const auto evictions_index =
      static_cast<std::size_t>(std::distance(kReportedCounters.begin(), kEvictions));
  const auto pre_witness_index =
      static_cast<std::size_t>(std::distance(kReportedCounters.begin(), kPreWitness));
  const auto& last = FreshnessProbe::activations[wanted - 1U].counters;
  EXPECT_GT(last[evictions_index] + last[pre_witness_index], 0U) << FreshnessProbe::describe();
}

// REQ: CHAN-131
// DrainAll over a drop-oldest edge is the robot pattern: a capture-less joint-state stream drained
// by a slower, independently clocked planner. The owner witnesses each arrival as soon as it is
// delivered, and the witness pinned the sole typed slot until the next tick, so newer samples
// replaced each other in staging and every tick drained the record admitted just after the
// previous one, about one consumer period old (twelve samples here). The reconcile now trades the
// retained record for the oldest newer staged sample while the tick is still ahead.
TEST(TemporalFreshnessTest, DepthOneDropOldestDrainAllSelectsTheNewestSample) {
  bool observed = false;
  ASSERT_NO_FATAL_FAILURE(
      (run_source_consumer_graph<CapturelessFreshnessSource<>,
                                 DrainAllFreshnessConsumer<1U>,
                                 1U,
                                 Realization::kDirect>(kObservedActivations, observed)));
  expect_steady_state_freshness(observed, kObservedActivations);
  expect_ordered_drain_batches(kObservedActivations, 1U);
}

// REQ: CHAN-128
// REQ: CHAN-131
// The pumped HoloMQ subscriber stages like the direct one, so the same trade must apply to it.
TEST(TemporalFreshnessTest, ForcedHoloMqDepthOneDropOldestDrainAllSelectsTheNewestSample) {
  bool observed = false;
  ASSERT_NO_FATAL_FAILURE(
      (run_source_consumer_graph<CapturelessFreshnessSource<>,
                                 DrainAllFreshnessConsumer<1U>,
                                 1U,
                                 Realization::kForcedHoloMq>(kObservedActivations, observed)));
  expect_steady_state_freshness(observed, kObservedActivations);
  expect_ordered_drain_batches(kObservedActivations, 1U);
}

// REQ: CHAN-131
// A captured sample's coordinate is its capture time, which the trade must also keep ahead of
// the tick.
TEST(TemporalFreshnessTest, DepthOneDropOldestDrainAllSelectsTheNewestCapturedSample) {
  bool observed = false;
  ASSERT_NO_FATAL_FAILURE(
      (run_source_consumer_graph<FreshnessSource<>,
                                 DrainAllFreshnessConsumer<1U>,
                                 1U,
                                 Realization::kDirect>(kObservedActivations, observed)));
  expect_steady_state_freshness(observed, kObservedActivations);
  expect_ordered_drain_batches(kObservedActivations, 1U);
}

// REQ: CHAN-131
// LatestWithin cannot rank a sample without a capture time: the selector reports
// kCaptureTimeAbsent, and SafeDefault keeps the consumer running without data. The freshness
// reconcile behind the input sees the same capture-less samples staged behind a retained one, and
// must decline them rather than fail the run with RUNTIME_PORT_BINDING_UNAVAILABLE.
TEST(TemporalFreshnessTest, CapturelessDropOldestLatestWithinRunsWithoutData) {
  bool observed = false;
  ASSERT_NO_FATAL_FAILURE(
      (run_source_consumer_graph<CapturelessFreshnessSource<2000>,
                                 FreshnessConsumer<1U, MissingPolicy::kSafeDefault, 20000>,
                                 1U,
                                 Realization::kDirect>(6U, observed)));
  ASSERT_TRUE(observed) << FreshnessProbe::describe();
  for (const ObservedActivation& activation : FreshnessProbe::activations) {
    EXPECT_FALSE(activation.selected) << FreshnessProbe::describe();
  }
  // The source publishes every 2 ms, so every activation after the first has samples to reject.
  for (std::size_t index = 1U; index < FreshnessProbe::activations.size(); ++index) {
    EXPECT_EQ(FreshnessProbe::activations[index].selection_reason,
              SelectionReason::kCaptureTimeAbsent)
        << "activation " << index << "; " << FreshnessProbe::describe();
  }
}

// REQ: CHAN-128
// At depth two the batch is the newest two unread survivors in producer order. Trading the oldest
// retained record for the oldest newer staged one keeps that order; promoting the newest first
// would strand older survivors behind it.
TEST(TemporalFreshnessTest, DepthTwoDropOldestDrainAllKeepsTheNewestTwoInOrder) {
  bool observed = false;
  ASSERT_NO_FATAL_FAILURE(
      (run_source_consumer_graph<CapturelessFreshnessSource<>,
                                 DrainAllFreshnessConsumer<2U>,
                                 2U,
                                 Realization::kDirect>(kObservedActivations, observed)));
  expect_steady_state_freshness(observed, kObservedActivations);
  expect_ordered_drain_batches(kObservedActivations, 2U);
}

// One counter of the consumer's partition owner, by default the signals it received. Accounting
// indices follow the runners' plan ordinals, and the consumer's runner is the only one with an
// incoming boundary edge.
[[nodiscard]] std::optional<std::uint64_t> consumer_partition_signals(
    const ExecutionPlan& plan, const detail::RuntimePhaseCounters& counters,
    const detail::RuntimePartitionCounter counter = detail::RuntimePartitionCounter::kSignals) {
  const auto state = detail::ExecutionPlanAccess::retain(plan);
  if (state == nullptr) {
    return std::nullopt;
  }
  std::vector<const detail::PartitionRunnerPlanRecord*> runners;
  for (const auto& runner : state->partition_runners) {
    runners.push_back(&runner);
  }
  std::ranges::sort(runners, {}, &detail::PartitionRunnerPlanRecord::ordinal);
  const auto consumer = std::ranges::find_if(
      runners, [](const auto* runner) { return !runner->incoming_boundary_edges.empty(); });
  if (consumer == runners.end()) {
    return std::nullopt;
  }
  return counters.partition_snapshot()[static_cast<std::size_t>(consumer - runners.begin())]
                                      [detail::RuntimePhaseCounters::index(counter)];
}

// Retention protection does not grant Tensor-backed inputs the scalar path's long-park policy.
// Exercise an actual compiled pool-backed graph, not just Channel's private admission call.
// REQ: RUNTIME-089
// REQ: CHAN-131
// REQ: TENSOR-STORAGE-016
TEST(TemporalFreshnessTest, DirectTensorDrainAllKeepsTheOneMillisecondFallback) {
  constexpr std::size_t kWanted = 12U;
  Graph graph{"temporal-freshness-tensor-tick-wait-policy"};
  auto source = graph.op<CapturelessFreshnessSource<2000, Tensor>>("source");
  auto consumer = graph.op<DrainAllFreshnessConsumer<1U, 20000, Tensor>>("consumer");
  graph.add_flow(source->output,
                 consumer->data,
                 ConnectionOptions{.queue_policy = QueuePolicy::kDropOldest, .queue_depth = 1U});
  graph.partition("source").add(source);
  graph.partition("consumer").add(consumer);
  graph.set_default_clock(graph.add_clock<RealtimeClock>("runtime-clock"));
  const ExecutionPlan compiled = compile(graph);
  ASSERT_TRUE(compiled.ok()) << compiled.json();
  auto counters = std::make_shared<detail::RuntimePhaseCounters>();
  const ExecutionPlan plan = detail::RuntimeTestPlanAccess::with_phase_counters(compiled, counters);
  FreshnessProbe::reset(counters);
  RunSession session = run_async(plan);
  const bool observed = FreshnessProbe::wait_for_activations(kWanted);
  session.request_stop();
  session.wait();
  ASSERT_TRUE(observed) << FreshnessProbe::describe();
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty());
  for (std::size_t index = 1U; index < kWanted; ++index) {
    const auto& activation = FreshnessProbe::activations[index];
    EXPECT_TRUE(activation.selected) << index;
    EXPECT_TRUE(activation.batch_ordered) << index;
    EXPECT_LE(activation.batch_size, 1U) << index;
  }
  const auto long_parks =
      consumer_partition_signals(plan, *counters, detail::RuntimePartitionCounter::kLongParks);
  const auto cap_returns =
      consumer_partition_signals(plan, *counters, detail::RuntimePartitionCounter::kCapReturns);
  ASSERT_TRUE(long_parks);
  ASSERT_TRUE(cap_returns);
  EXPECT_EQ(*long_parks, 0U);
  EXPECT_GT(*cap_returns, 0U);
}

// REQ: RUNTIME-068
namespace {

// How a side-only edge reaches its consumer in the owner-hint tests.
enum class SideHintPath : std::uint8_t { kDirect, kEmit, kPump };

[[nodiscard]] const char* side_hint_path_name(const SideHintPath path) {
  switch (path) {
    case SideHintPath::kEmit:
      return "holomq-emit";
    case SideHintPath::kPump:
      return "holomq-pump";
    case SideHintPath::kDirect:
      break;
  }
  return "direct";
}

// Direct retention, HoloMQ delivered inside emit() (the default for an eligible service), or
// HoloMQ served by the Channel pump.
[[nodiscard]] ExecutionPlan compile_side_hint_graph(Graph& graph, const SideHintPath path) {
  switch (path) {
    case SideHintPath::kEmit:
      return detail::compile_with_process_local_holomq(graph, CompileOptions{});
    case SideHintPath::kPump:
      return detail::compile_with_process_local_pump_delivery(graph, CompileOptions{}, true);
    case SideHintPath::kDirect:
      break;
  }
  return compile(graph);
}

}  // namespace

// Run a 2 ms capture-less producer into a depth-one drop-oldest DrainAll consumer, whose partition
// parks until its next tick, and require each tick to drain a sample at most one publication old.
// With a 20 ms consumer every tick coincides with a publication; 20.3 ms drifts off phase.
template <std::int64_t ConsumerPeriodUs>
void expect_long_parked_drain_all_freshness(const Realization delivery) {
  constexpr std::size_t kWanted = 12U;
  Graph graph{"temporal-freshness-long-park"};
  auto source = graph.op<CapturelessFreshnessSource<2000>>("source");
  auto consumer = graph.op<DrainAllFreshnessConsumer<1U, ConsumerPeriodUs>>("consumer");
  graph.add_flow(source->output,
                 consumer->data,
                 ConnectionOptions{.queue_policy = QueuePolicy::kDropOldest, .queue_depth = 1U});
  graph.partition("source").add(source);
  graph.partition("consumer").add(consumer);
  graph.set_default_clock(graph.add_clock<RealtimeClock>("runtime_clock"));
  const ExecutionPlan compiled =
      delivery == Realization::kForcedHoloMq
          ? detail::compile_with_process_local_holomq(graph, CompileOptions{})
          : compile(graph);
  ASSERT_TRUE(compiled.ok()) << compiled.json();
  auto counters = std::make_shared<detail::RuntimePhaseCounters>();
  ExecutionPlan plan = detail::RuntimeTestPlanAccess::with_phase_counters(compiled, counters);
  plan = detail::RuntimeTestPlanAccess::with_owner_park_cap(plan, 10s);
  ASSERT_TRUE(plan.ok()) << plan.json();

  FreshnessProbe::reset(counters);
  RunSession session = run_async(plan);
  const bool observed = FreshnessProbe::wait_for_activations(kWanted);
  session.request_stop();
  session.wait();
  ASSERT_TRUE(observed) << FreshnessProbe::describe();
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty());
  for (std::size_t index = 1U; index < kWanted; ++index) {
    const ObservedActivation& activation = FreshnessProbe::activations[index];
    ASSERT_TRUE(activation.selected)
        << "activation " << index << "; " << FreshnessProbe::describe();
    // A publication racing the tick may land on either side of it, hence one sample of slack.
    const std::uint64_t by_tick = FreshnessProbe::published_by(activation.activation_time_ns);
    EXPECT_LE(by_tick - std::min(by_tick, activation.value), 1U)
        << "activation " << index << " drained " << activation.value << "; " << by_tick
        << " publications were complete by its tick and " << activation.published
        << " by its compute; " << FreshnessProbe::describe();
  }
  // Only direct retention is tick-reconciled; a HoloMQ-delivered input keeps the 1 ms bound.
  const auto long_parks =
      consumer_partition_signals(plan, *counters, detail::RuntimePartitionCounter::kLongParks);
  HOLOSCAN_ASSERT_HAS_VALUE(long_parks) << "consumer partition counter unavailable";
  if (delivery == Realization::kDirect) {
    EXPECT_GT(*long_parks, 0U);
  } else {
    EXPECT_EQ(*long_parks, 0U);
  }
}

// A long-parked DrainAll side input keeps the newest survivor at or before its tick when a
// publication races the tick scan: staging holds one survivor beyond the authored depth. The
// 20.3 ms consumer is the phase-offset control.
// REQ: RUNTIME-088
// REQ: RUNTIME-089
// REQ: CHAN-131
TEST(TemporalFreshnessTest, LongParkedDrainAllSideInputSurvivesAPublicationRacingTheTick) {
  for (const Realization delivery : {Realization::kDirect, Realization::kForcedHoloMq}) {
    SCOPED_TRACE(testing::Message() << "holomq=" << (delivery == Realization::kForcedHoloMq));
    {
      SCOPED_TRACE("phase-aligned 20 ms consumer");
      expect_long_parked_drain_all_freshness<20000>(delivery);
    }
    {
      SCOPED_TRACE("phase-offset 20.3 ms consumer");
      expect_long_parked_drain_all_freshness<20300>(delivery);
    }
  }
}

// REQ: RUNTIME-087
// REQ: RUNTIME-088
// REQ: RUNTIME-089
// The drained input cannot activate its clocked consumer, so its arrivals publish their readiness
// without waking the consumer's partition, and each tick's scan ingests them. Freshness is
// unchanged. Under pump-served HoloMQ the control keeps a wake per received sample. A direct
// drop-oldest edge, like HoloMQ delivered inside emit(), publishes readiness only while a typed
// slot is free and stages the rest, so its control wakes about once per tick; the elision still
// removes those.
TEST(TemporalFreshnessTest, SideOnlyDrainAllArrivalsDoNotWakeTheClockedConsumer) {
  constexpr std::size_t kWanted = 10U;
  for (const SideHintPath path :
       {SideHintPath::kDirect, SideHintPath::kEmit, SideHintPath::kPump}) {
    for (const bool keep_hints : {false, true}) {
      SCOPED_TRACE(testing::Message()
                   << "path=" << side_hint_path_name(path) << " keep_hints=" << keep_hints);
      Graph graph{"temporal-freshness-side-only-hints"};
      auto source = graph.op<CapturelessFreshnessSource<2000>>("source");
      auto consumer = graph.op<DrainAllFreshnessConsumer<1U, 20000>>("consumer");
      graph.add_flow(
          source->output,
          consumer->data,
          ConnectionOptions{.queue_policy = QueuePolicy::kDropOldest, .queue_depth = 1U});
      graph.partition("source").add(source);
      graph.partition("consumer").add(consumer);
      graph.set_default_clock(graph.add_clock<RealtimeClock>("runtime_clock"));
      const ExecutionPlan compiled = compile_side_hint_graph(graph, path);
      ASSERT_TRUE(compiled.ok()) << compiled.json();
      auto counters = std::make_shared<detail::RuntimePhaseCounters>();
      ExecutionPlan plan = detail::RuntimeTestPlanAccess::with_phase_counters(compiled, counters);
      if (keep_hints) {
        plan = detail::RuntimeTestPlanAccess::keep_side_only_data_hints_for_test(plan);
      }
      ASSERT_TRUE(plan.ok()) << plan.json();

      FreshnessProbe::reset(counters);
      RunSession session = run_async(plan);
      const bool observed = FreshnessProbe::wait_for_activations(kWanted);
      session.request_stop();
      session.wait();
      ASSERT_TRUE(observed) << FreshnessProbe::describe();
      EXPECT_EQ(session.termination(), RunTermination::kStopped);
      EXPECT_TRUE(session.diagnostics().empty());
      for (std::size_t index = 1U; index < kWanted; ++index) {
        const ObservedActivation& activation = FreshnessProbe::activations[index];
        ASSERT_TRUE(activation.selected)
            << "activation " << index << "; " << FreshnessProbe::describe();
        EXPECT_LE(activation.published - std::min(activation.published, activation.value),
                  kMaximumFreshSampleLag)
            << "activation " << index << "; " << FreshnessProbe::describe();
      }

      const auto consumer_signals = consumer_partition_signals(plan, *counters);
      HOLOSCAN_ASSERT_HAS_VALUE(consumer_signals) << "consumer partition counter unavailable";
      const std::uint64_t signals = *consumer_signals;
      // Its tick scan alone reconciles a direct input (RUNTIME-088), so that owner parks until
      // its next deadline. A HoloMQ-delivered input keeps the 1 ms bound.
      const auto long_parks =
          consumer_partition_signals(plan, *counters, detail::RuntimePartitionCounter::kLongParks);
      HOLOSCAN_ASSERT_HAS_VALUE(long_parks) << "consumer partition counter unavailable";
      if (path == SideHintPath::kDirect) {
        EXPECT_GT(*long_parks, 0U);
      } else {
        EXPECT_EQ(*long_parks, 0U);
      }
      const std::uint64_t elided = counters->snapshot()[detail::RuntimePhaseCounters::index(
          detail::RuntimePhaseCounter::kChannelSideOnlyDataHintsElided)];
      const std::uint64_t published = FreshnessProbe::published.load(std::memory_order_relaxed);
      ASSERT_GE(published, 4U * kWanted);
      const std::uint64_t data_hints =
          path == SideHintPath::kPump ? published / 2U : std::uint64_t{1U};
      if (keep_hints) {
        EXPECT_EQ(elided, 0U);
        EXPECT_GE(signals, data_hints);
      } else {
        // Key assertion: the source's arrivals no longer wake the consumer's partition.
        EXPECT_GE(elided, data_hints);
        EXPECT_LE(signals, published / 10U);
      }
    }
  }
}

// REQ: RUNTIME-068
// An OnEach owner polls its trigger and side arrivals through one binding in arrival order, so
// only trigger samples need to wake it: each trigger's scan also ingests every earlier side sample.
// The side input is ten times faster here, so under pump-served HoloMQ its wakes would otherwise
// dominate. A direct drop-oldest side edge, like HoloMQ delivered inside emit(), stages while its
// typed slot is pinned, so it wakes rarely anyway.
TEST(TemporalFreshnessTest, EachOwnerWakesOnlyForItsTriggerInput) {
  constexpr std::size_t kWanted = 20U;
  for (const SideHintPath path :
       {SideHintPath::kDirect, SideHintPath::kEmit, SideHintPath::kPump}) {
    for (const bool keep_hints : {false, true}) {
      SCOPED_TRACE(testing::Message()
                   << "path=" << side_hint_path_name(path) << " keep_hints=" << keep_hints);
      Graph graph{"temporal-freshness-each-side-hints"};
      auto trigger = graph.op<EachTriggerSource<10000>>("trigger");
      auto side = graph.op<CapturelessFreshnessSource<1000>>("side");
      auto consumer = graph.op<EachTriggerSideDrainConsumer>("consumer");
      graph.add_flow(trigger->output, consumer->trigger);
      graph.add_flow(
          side->output,
          consumer->side,
          ConnectionOptions{.queue_policy = QueuePolicy::kDropOldest, .queue_depth = 1U});
      graph.partition("trigger").add(trigger);
      graph.partition("side").add(side);
      graph.partition("consumer").add(consumer);
      graph.set_default_clock(graph.add_clock<RealtimeClock>("runtime_clock"));
      const ExecutionPlan compiled = compile_side_hint_graph(graph, path);
      ASSERT_TRUE(compiled.ok()) << compiled.json();
      auto counters = std::make_shared<detail::RuntimePhaseCounters>();
      ExecutionPlan plan = detail::RuntimeTestPlanAccess::with_phase_counters(compiled, counters);
      if (keep_hints) {
        plan = detail::RuntimeTestPlanAccess::keep_side_only_data_hints_for_test(plan);
      }
      ASSERT_TRUE(plan.ok()) << plan.json();

      EachTriggerSource<10000>::published.store(0U, std::memory_order_relaxed);
      FreshnessProbe::reset(counters);
      RunSession session = run_async(plan);
      const bool observed = FreshnessProbe::wait_for_activations(kWanted);
      session.request_stop();
      session.wait();
      ASSERT_TRUE(observed) << FreshnessProbe::describe();
      EXPECT_EQ(session.termination(), RunTermination::kStopped);
      EXPECT_TRUE(session.diagnostics().empty());
      std::uint64_t latest{};
      for (std::size_t index = 0U; index < kWanted; ++index) {
        const ObservedActivation& activation = FreshnessProbe::activations[index];
        if (activation.selected) {
          EXPECT_GT(activation.value, latest) << FreshnessProbe::describe();
          latest = activation.value;
        }
      }
      EXPECT_GT(latest, 0U) << FreshnessProbe::describe();

      const auto consumer_signals = consumer_partition_signals(plan, *counters);
      HOLOSCAN_ASSERT_HAS_VALUE(consumer_signals) << "consumer partition counter unavailable";
      const std::uint64_t signals = *consumer_signals;
      const std::uint64_t elided = counters->snapshot()[detail::RuntimePhaseCounters::index(
          detail::RuntimePhaseCounter::kChannelSideOnlyDataHintsElided)];
      const std::uint64_t triggers = EachTriggerSource<10000>::published.load();
      const std::uint64_t side_published =
          FreshnessProbe::published.load(std::memory_order_relaxed);
      ASSERT_GE(side_published, 4U * triggers);
      const std::uint64_t side_hints =
          path == SideHintPath::kPump ? side_published / 2U : std::uint64_t{1U};
      if (keep_hints) {
        EXPECT_EQ(elided, 0U);
        EXPECT_GE(signals, side_hints);
      } else {
        // Key assertion: the consumer's partition is signaled for triggers, not for side samples.
        EXPECT_GE(elided, side_hints);
        EXPECT_LE(signals, triggers + side_published / 10U);
      }
    }
  }
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
