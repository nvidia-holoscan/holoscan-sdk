/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>

#if defined(__linux__)
#include <sys/resource.h>
#endif

#include <holoscan/core/source_id.hpp>

namespace holoscan::detail {

/// Identify one source-tree-only Runtime execution phase.
enum class RuntimePhaseCounter : std::uint8_t {
  kReadinessScans,
  kNoProgressScans,
  kWaitCalls,
  kWakeSignals,
  kWakeReturns,
  kTimeoutReturns,
  kStopSignals,
  kCandidatesFrozen,
  kCandidatesSelected,
  kReadinessPollDispatches,
  kReadinessPollSkippedByCoordinatorBound,
  kReadinessPollSkippedBySourceCredit,
  kReadinessPollSkippedByNoTarget,
  kDirectContinuationReceiptsAccepted,
  kDirectContinuationPollElisions,
  kDirectCompositeAttempts,
  kDirectCompositeCompletions,
  kDirectCompositeFallbacks,
  kDirectCompositeFailures,
  kDirectCompositeCallbacksAttempted,
  kDirectCompositeCallbacksCompleted,
  kComputeDispatches,
  kActivationsCompleted,
  kChannelPrepareCalls,
  kChannelEmitAttempts,
  kChannelEmitAccepted,
  kChannelEmitBackpressured,
  kChannelSamplesReceived,
  kChannelReadinessPublished,
  kChannelInputReads,
  kChannelWitnessRetired,
  kChannelRetireCalls,
  kChannelActivationCuts,
  kChannelActivationCutDrains,
  kChannelActivationCutSkips,
  kChannelActivationCutInventoryScans,
  kChannelActivationCutSubscriberProbes,
  kChannelActivationCutTransportProbes,
  kChannelActivationCutMaterializations,
  kChannelReceiveBackpressured,
  kChannelRetirementPromotedNothing,
  kChannelPollIncomplete,
  kChannelDirectIngressPromotionPasses,
  kChannelDirectIngressPromotions,
  kChannelStagedEvidenceSweepVisits,
  kChannelProtectedTerminalProbesWithPendingData,
  kChannelPreWitnessReplacements,
  kChannelFreshnessReservationReleaseDeferred,
  kChannelReleasedClaimRefillReplacementDeclined,
  kChannelReleasedClaimRefillPublished,
  // One group per outcome of the superseded-retention reconcile. Every decline is counted on its
  // own branch: the reconcile trades a bounded staleness for a possible lost activation, so which
  // branch declined is the whole diagnostic question, and an aggregate "declined" would not
  // separate a gate that is working from a gate that never fires.
  kTemporalRetentionCandidates,
  kTemporalRetentionEvictions,
  kTemporalRetentionDeclinedNotSuperseded,
  kTemporalRetentionDeclinedUnboundedRound,
  kTemporalRetentionDeclinedUncomparableCoordinate,
  kTemporalRetentionDeclinedAboveFloor,
  kTemporalRetentionDeclinedEvictionRefused,
  kTemporalRetentionFloorResolved,
  kTemporalRetentionFloorAbsent,
  kTemporalRetentionEvictionPromoted,
  kTemporalRetentionEvictionPromotedNothing,
  kTemporalReuseLastRetainedRecordQueries,
  kTemporalReuseLastComparatorCalls,
  // A lost ordinary selection, split by whether the reconcile evicted from that selector in the
  // same freeze round. Together they account for every ordinary selection that found no record.
  kSelectionMissedAfterFreshnessEviction,
  kSelectionMissedAfterUngatedEviction,
  kSelectionMissedAfterUnpolledEviction,
  kSelectionMissedAfterUnrefilledEviction,
  kSelectionMissedWithoutFreshnessEviction,
  kCandidateSourceLookupProbes,
  kCandidateOrderComparisons,
  kCandidateOrderMoves,
  kCandidateTagTimeReservations,
  kLaneDispatches,
  kLaneComputeDispatches,
  kLaneDispatcherDispatches,
  kLaneLifecycleDispatches,
  kLaneUnknownDispatches,
  kLaneBusyTaskPublications,
  kLaneSleepingWakeSignals,
  kLaneCallbacksEntered,
  kLaneCallbacksExited,
  kLaneCompletionsObserved,
  kLaneSerializationContentions,
  kLaneInlineOwnerInvocations,
  kPartitionRunnerSweeps,
  kPartitionRunnerWakeSignals,
  kPartitionTailDispatchAttempts,
  kPartitionTailDispatchAccepted,
  kPartitionTailDispatchFallbacks,
  kPartitionCausalFrontierAttempts,
  kPartitionCausalFrontierAccepted,
  kPartitionCausalFrontierFallbacks,
  kPartitionControlBarriers,
  kCount,
};

inline constexpr std::size_t kRuntimePhaseCounterCount =
    static_cast<std::size_t>(RuntimePhaseCounter::kCount);

/// Identify one source-tree-only Runtime timing checkpoint.
enum class RuntimePhaseTracePoint : std::uint8_t {
  kChannelTransportWaitReturned,
  kChannelIngressRetained,
  kChannelEndOfStreamRetained,
  kChannelReadinessWakeSignaled,
  kControllerWakeSignaled,
  kControllerParkStarted,
  kControllerWakeReturned,
  kControllerTimeoutReturned,
  kReadinessPollReturned,
  kCandidateFrozen,
  kCandidateSelected,
  kInputSelectionPrepared,
  kActivationPrepared,
  kComputeDispatchStarted,
  kComputeCallbackEntered,
  kComputeCallbackReturned,
  kDirectCompositeStarted,
  kDirectCompositeSuccessorEntered,
  kDirectCompositeReturned,
  kComputeDispatchReturned,
  kActivationRetirementCompleted,
  kActivationCutCompleted,
  kTensorEmitStarted,
  kTensorEmitValidated,
  kTensorTransportPublicationStarted,
  kTensorTransportPublicationReturned,
  kTensorEmitCompleted,
  kInlinePayloadEmitStarted,
  kInlinePayloadEncoded,
  kInlinePayloadTransportPublicationStarted,
  kInlinePayloadTransportPublicationReturned,
  kInlinePayloadEmitCompleted,
};

inline constexpr std::size_t kNoRuntimePhaseReceiver = std::numeric_limits<std::size_t>::max();

/// Retain one bounded Runtime timing checkpoint without owning payload state.
struct RuntimePhaseTraceRecord {
  RuntimePhaseTracePoint point{RuntimePhaseTracePoint::kCandidateFrozen};
  std::size_t receiver_index{kNoRuntimePhaseReceiver};
  SourceId source{};
  std::uint64_t source_sequence{};
  std::uint64_t output_port_id{};
  std::int64_t timestamp_ns{};
};

/// Identify one source-tree-only Runtime thread population.
enum class RuntimeThreadUsageRole : std::uint8_t {
  kController,
  kExecutionLane,
  kChannelPump,
  kNotificationDriver,
  kCount,
};

inline constexpr std::size_t kRuntimeThreadUsageRoleCount =
    static_cast<std::size_t>(RuntimeThreadUsageRole::kCount);

/// Retain one role's aggregated per-thread resource-use evidence.
struct RuntimeThreadUsageRecord {
  RuntimeThreadUsageRole role{RuntimeThreadUsageRole::kController};
  bool supported{};
  std::uint64_t samples{};
  std::uint64_t user_cpu_ns{};
  std::uint64_t system_cpu_ns{};
  std::uint64_t voluntary_context_switches{};
  std::uint64_t involuntary_context_switches{};
  std::uint64_t minor_page_faults{};
  std::uint64_t major_page_faults{};
};

using RuntimeThreadUsageSnapshot =
    std::array<RuntimeThreadUsageRecord, kRuntimeThreadUsageRoleCount>;

namespace runtime_thread_usage_detail {

#if defined(__linux__) && defined(RUSAGE_THREAD)
inline constexpr bool kSupported = true;
#else
inline constexpr bool kSupported = false;
#endif

struct Sample {
  std::uint64_t user_cpu_ns{};
  std::uint64_t system_cpu_ns{};
  std::uint64_t voluntary_context_switches{};
  std::uint64_t involuntary_context_switches{};
  std::uint64_t minor_page_faults{};
  std::uint64_t major_page_faults{};
};

#if defined(__linux__) && defined(RUSAGE_THREAD)
inline bool timeval_to_ns(const timeval& value, std::uint64_t& result) noexcept {
  if (value.tv_sec < 0 || value.tv_usec < 0 || value.tv_usec >= 1'000'000) {
    return false;
  }
  constexpr std::uint64_t kNanosecondsPerSecond = 1'000'000'000U;
  constexpr std::uint64_t kNanosecondsPerMicrosecond = 1'000U;
  const auto seconds = static_cast<std::uint64_t>(value.tv_sec);
  const auto microseconds = static_cast<std::uint64_t>(value.tv_usec);
  if (seconds > std::numeric_limits<std::uint64_t>::max() / kNanosecondsPerSecond) {
    return false;
  }
  result = seconds * kNanosecondsPerSecond + microseconds * kNanosecondsPerMicrosecond;
  return true;
}

inline bool nonnegative_to_u64(long value, std::uint64_t& result) noexcept {
  if (value < 0) {
    return false;
  }
  result = static_cast<std::uint64_t>(value);
  return true;
}
#endif

inline bool sample_current_thread(Sample& result) noexcept {
#if defined(__linux__) && defined(RUSAGE_THREAD)
  rusage usage{};
  if (::getrusage(RUSAGE_THREAD, &usage) != 0) {
    return false;
  }
  return timeval_to_ns(usage.ru_utime, result.user_cpu_ns) &&
         timeval_to_ns(usage.ru_stime, result.system_cpu_ns) &&
         nonnegative_to_u64(usage.ru_nvcsw, result.voluntary_context_switches) &&
         nonnegative_to_u64(usage.ru_nivcsw, result.involuntary_context_switches) &&
         nonnegative_to_u64(usage.ru_minflt, result.minor_page_faults) &&
         nonnegative_to_u64(usage.ru_majflt, result.major_page_faults);
#else
  static_cast<void>(result);
  return false;
#endif
}

inline bool subtract(const Sample& start, const Sample& finish, Sample& result) noexcept {
  if (finish.user_cpu_ns < start.user_cpu_ns || finish.system_cpu_ns < start.system_cpu_ns ||
      finish.voluntary_context_switches < start.voluntary_context_switches ||
      finish.involuntary_context_switches < start.involuntary_context_switches ||
      finish.minor_page_faults < start.minor_page_faults ||
      finish.major_page_faults < start.major_page_faults) {
    return false;
  }
  result = Sample{
      .user_cpu_ns = finish.user_cpu_ns - start.user_cpu_ns,
      .system_cpu_ns = finish.system_cpu_ns - start.system_cpu_ns,
      .voluntary_context_switches =
          finish.voluntary_context_switches - start.voluntary_context_switches,
      .involuntary_context_switches =
          finish.involuntary_context_switches - start.involuntary_context_switches,
      .minor_page_faults = finish.minor_page_faults - start.minor_page_faults,
      .major_page_faults = finish.major_page_faults - start.major_page_faults,
  };
  return true;
}

}  // namespace runtime_thread_usage_detail

/// Retain the coordinates one lost selection was decided on.
///
/// Kept apart from \ref RuntimePhaseTraceRecord deliberately. The general ring is written by
/// high-frequency checkpoints and would both overflow before a rare lost selection occurred and
/// add a write to the dispatch path this record exists to measure. This sink is written only when
/// a selection is lost after an eviction, which is a few times per run.
struct TemporalRetentionMissRecord {
  std::int64_t witness_reference_ns{};
  std::int64_t successor_capture_ns{};
  std::int64_t floor_ns{};
  // The selector's own account of why it selected nothing, which separates "the cache was empty"
  // from "the cache held a record this witness could not use" -- opposite causes with opposite
  // fixes, and indistinguishable from the fact of the loss alone.
  std::uint8_t selection_state{};
  std::uint8_t selection_reason{};
  bool floor_present{};
  /// Whether any record reached the selector's cache in the round that lost the selection.
  bool ingested_this_round{};
  /// Fate of the record the eviction was traded for: 0 none/pending, 1 published, 2 protected.
  std::uint8_t refill_outcome{};
  /// Capture of the record actually promoted, against successor_capture_ns which the gate approved.
  std::int64_t promoted_capture_ns{};
  /// Readiness events the poll loop ingested into the coordinator in that same round.
  std::uint32_t round_ingested_count{};
  /// Candidates still pending in the coordinator when the losing selection was formed. Non-zero
  /// means the round ingested an event it never froze; zero means everything ingested was frozen
  /// and none of it belonged to this selector's port.
  std::uint32_t pending_at_selection{};
  /// Reference coordinate of the oldest candidate still pending, or 0 when none is.
  std::int64_t pending_oldest_reference_ns{};
};

inline constexpr std::size_t kTemporalRetentionMissCapacity = 256U;

/// Own opt-in relaxed counters used by source-tree benchmarks and tests.
class RuntimePhaseCounters final {
 public:
  using Snapshot = std::array<std::uint64_t, kRuntimePhaseCounterCount>;

  RuntimePhaseCounters() = default;

  /// Borrow preallocated trace storage for source-tree diagnostics.
  ///
  /// The storage must outlive this counter object and every RunSession that retains it.
  explicit RuntimePhaseCounters(std::span<RuntimePhaseTraceRecord> trace_storage) noexcept
      : trace_storage_(trace_storage) {}

  void increment(RuntimePhaseCounter counter) noexcept { add(counter, 1U); }

  void add(RuntimePhaseCounter counter, std::uint64_t delta) noexcept {
    counters_[index(counter)].value.fetch_add(delta, std::memory_order_relaxed);
  }

  [[nodiscard]] Snapshot snapshot() const noexcept {
    Snapshot result{};
    for (std::size_t index = 0U; index < result.size(); ++index) {
      result[index] = counters_[index].value.load(std::memory_order_relaxed);
    }
    return result;
  }

  [[nodiscard]] static constexpr std::size_t index(RuntimePhaseCounter counter) noexcept {
    return static_cast<std::size_t>(counter);
  }

  [[nodiscard]] static constexpr std::size_t thread_usage_index(
      RuntimeThreadUsageRole role) noexcept {
    return static_cast<std::size_t>(role);
  }

  /// Return role-aggregated thread evidence after the observed threads have stopped.
  [[nodiscard]] RuntimeThreadUsageSnapshot thread_usage_snapshot() const noexcept {
    RuntimeThreadUsageSnapshot result{};
    for (std::size_t index = 0U; index < result.size(); ++index) {
      const ThreadUsageSlot& slot = thread_usage_[index];
      result[index] = RuntimeThreadUsageRecord{
          .role = static_cast<RuntimeThreadUsageRole>(index),
          .supported = runtime_thread_usage_detail::kSupported,
          .samples = slot.samples.load(std::memory_order_relaxed),
          .user_cpu_ns = slot.user_cpu_ns.load(std::memory_order_relaxed),
          .system_cpu_ns = slot.system_cpu_ns.load(std::memory_order_relaxed),
          .voluntary_context_switches =
              slot.voluntary_context_switches.load(std::memory_order_relaxed),
          .involuntary_context_switches =
              slot.involuntary_context_switches.load(std::memory_order_relaxed),
          .minor_page_faults = slot.minor_page_faults.load(std::memory_order_relaxed),
          .major_page_faults = slot.major_page_faults.load(std::memory_order_relaxed),
      };
    }
    return result;
  }

  /// Accumulate one completed Runtime thread lifetime without allocating.
  void record_thread_usage(RuntimeThreadUsageRole role,
                           const runtime_thread_usage_detail::Sample& delta) noexcept {
    ThreadUsageSlot& slot = thread_usage_[thread_usage_index(role)];
    saturating_add(slot.user_cpu_ns, delta.user_cpu_ns);
    saturating_add(slot.system_cpu_ns, delta.system_cpu_ns);
    saturating_add(slot.voluntary_context_switches, delta.voluntary_context_switches);
    saturating_add(slot.involuntary_context_switches, delta.involuntary_context_switches);
    saturating_add(slot.minor_page_faults, delta.minor_page_faults);
    saturating_add(slot.major_page_faults, delta.major_page_faults);
    saturating_add(slot.samples, 1U);
  }

  /// Append one timestamped checkpoint or count a bounded-storage overflow.
  void trace(RuntimePhaseTracePoint point, std::size_t receiver_index, SourceId source,
             std::uint64_t source_sequence, std::uint64_t output_port_id = 0U) noexcept {
    if (trace_storage_.empty()) {
      return;
    }
    const std::size_t index = trace_write_index_.fetch_add(1U, std::memory_order_relaxed);
    if (index >= trace_storage_.size()) {
      trace_dropped_.fetch_add(1U, std::memory_order_relaxed);
      return;
    }
    trace_storage_[index] = RuntimePhaseTraceRecord{
        .point = point,
        .receiver_index = receiver_index,
        .source = source,
        .source_sequence = source_sequence,
        .output_port_id = output_port_id,
        .timestamp_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now().time_since_epoch())
                            .count(),
    };
  }

  /// Return the initialized prefix after the observed RunSession has stopped.
  [[nodiscard]] std::span<const RuntimePhaseTraceRecord> trace_records() const noexcept {
    return trace_storage_.first(
        std::min(trace_write_index_.load(std::memory_order_relaxed), trace_storage_.size()));
  }

  /// Return the borrowed checkpoint capacity.
  [[nodiscard]] std::size_t trace_capacity() const noexcept { return trace_storage_.size(); }

  /// Return every attempted checkpoint, including records rejected after storage exhaustion.
  [[nodiscard]] std::size_t trace_attempted() const noexcept {
    return trace_write_index_.load(std::memory_order_relaxed);
  }

  /// Record one lost selection's deciding coordinates, or drop it once the sink is full.
  void record_retention_miss(const TemporalRetentionMissRecord& record) noexcept {
    const std::size_t index = retention_miss_index_.fetch_add(1U, std::memory_order_relaxed);
    if (index >= retention_misses_.size()) {
      return;
    }
    retention_misses_[index] = record;
  }

  /// Return the initialized prefix after the observed RunSession has stopped.
  [[nodiscard]] std::span<const TemporalRetentionMissRecord> retention_misses() const noexcept {
    return std::span{retention_misses_}.first(
        std::min(retention_miss_index_.load(std::memory_order_relaxed), retention_misses_.size()));
  }

  /// Return checkpoints dropped after exhausting borrowed storage.
  [[nodiscard]] std::uint64_t trace_dropped() const noexcept {
    return trace_dropped_.load(std::memory_order_relaxed);
  }

  /// Return whether enabled checkpoint recording exhausted its bounded borrowed storage.
  [[nodiscard]] bool trace_capacity_exhausted() const noexcept { return trace_dropped() != 0U; }

 private:
  // Enabled diagnostics can update controller and producer-owned counters concurrently. Keep
  // independently written counters off the same cache line; production plans allocate none.
  struct alignas(64) CounterSlot {
    std::atomic<std::uint64_t> value{};
  };

  struct alignas(64) ThreadUsageSlot {
    std::atomic<std::uint64_t> samples{};
    std::atomic<std::uint64_t> user_cpu_ns{};
    std::atomic<std::uint64_t> system_cpu_ns{};
    std::atomic<std::uint64_t> voluntary_context_switches{};
    std::atomic<std::uint64_t> involuntary_context_switches{};
    std::atomic<std::uint64_t> minor_page_faults{};
    std::atomic<std::uint64_t> major_page_faults{};
  };

  static void saturating_add(std::atomic<std::uint64_t>& value, std::uint64_t delta) noexcept {
    std::uint64_t current = value.load(std::memory_order_relaxed);
    for (;;) {
      const std::uint64_t next = delta > std::numeric_limits<std::uint64_t>::max() - current
                                     ? std::numeric_limits<std::uint64_t>::max()
                                     : current + delta;
      if (value.compare_exchange_weak(
              current, next, std::memory_order_relaxed, std::memory_order_relaxed)) {
        return;
      }
    }
  }

  std::array<CounterSlot, kRuntimePhaseCounterCount> counters_{};
  std::array<ThreadUsageSlot, kRuntimeThreadUsageRoleCount> thread_usage_{};
  std::span<RuntimePhaseTraceRecord> trace_storage_;
  std::atomic<std::size_t> trace_write_index_{};
  std::atomic<std::uint64_t> trace_dropped_{};
  std::array<TemporalRetentionMissRecord, kTemporalRetentionMissCapacity> retention_misses_{};
  std::atomic<std::size_t> retention_miss_index_{};
};

/// Aggregate one opt-in Runtime thread lifetime into preallocated role counters.
class RuntimeThreadUsageScope final {
 public:
  RuntimeThreadUsageScope(RuntimePhaseCounters* counters, RuntimeThreadUsageRole role) noexcept
      : counters_(counters), role_(role) {
    if (counters_ != nullptr) {
      sampled_ = runtime_thread_usage_detail::sample_current_thread(start_);
    }
  }

  RuntimeThreadUsageScope(const RuntimeThreadUsageScope&) = delete;
  RuntimeThreadUsageScope& operator=(const RuntimeThreadUsageScope&) = delete;
  RuntimeThreadUsageScope(RuntimeThreadUsageScope&&) = delete;
  RuntimeThreadUsageScope& operator=(RuntimeThreadUsageScope&&) = delete;

  ~RuntimeThreadUsageScope() {
    if (!sampled_) {
      return;
    }
    runtime_thread_usage_detail::Sample finish{};
    runtime_thread_usage_detail::Sample delta{};
    if (runtime_thread_usage_detail::sample_current_thread(finish) &&
        runtime_thread_usage_detail::subtract(start_, finish, delta)) {
      counters_->record_thread_usage(role_, delta);
    }
  }

 private:
  RuntimePhaseCounters* counters_{};
  RuntimeThreadUsageRole role_{RuntimeThreadUsageRole::kController};
  runtime_thread_usage_detail::Sample start_{};
  bool sampled_{};
};

}  // namespace holoscan::detail
