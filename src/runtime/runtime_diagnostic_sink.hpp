/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>

#include <holoscan/core/diagnostic.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/lifecycle.hpp>
#include <holoscan/core/run_session.hpp>
#include <holoscan/core/source_id.hpp>
#include <holoscan/core/tag.hpp>
#include <holoscan/core/temporal_plan.hpp>
#include <holoscan/time/time_point.hpp>

#include "../core/lifecycle_kernel.hpp"
#include "execution_lane.hpp"
#include "runtime_cuda_execution_diagnostic.hpp"
#include "runtime_replay_mismatch.hpp"

namespace holoscan::detail {

/// Maximum number of structural diagnostics retained and published by one run.
inline constexpr std::size_t kRuntimeStructuralDiagnosticCapacity = 16U;

/// @brief Identify the terminal outcome represented by one supplemental Runtime summary.
enum class RuntimeTerminalCategory : std::uint8_t {
  kFailed,
};

/// @brief Retain bounded context for one supplemental terminal-session summary.
///
/// Structured diagnostics and the terminal RunSession state remain authoritative. This private
/// projection exists only to make the once-per-failed-session summary replaceable in source-tree
/// tests. \c first_token is borrowed only for the duration of the sink call.
struct RuntimeTerminalSummary {
  RunGenerationId run_generation{};
  RuntimeTerminalCategory category{RuntimeTerminalCategory::kFailed};
  std::size_t diagnostic_count{};
  DiagnosticId first_diagnostic{};
  DiagnosticSource first_source{};
  std::string_view first_token;
};

/// @brief Receive one supplemental summary after a failed RunSession commits its terminal state.
///
/// Implementations must copy borrowed text before returning if they need longer retention.
/// Runtime contains sink exceptions so observability cannot change cleanup or termination.
class RuntimeTerminalSummarySink {
 public:
  RuntimeTerminalSummarySink() = default;
  RuntimeTerminalSummarySink(const RuntimeTerminalSummarySink&) = delete;
  RuntimeTerminalSummarySink& operator=(const RuntimeTerminalSummarySink&) = delete;
  virtual ~RuntimeTerminalSummarySink() = default;

  /// @brief Publish one bounded failed-session summary.
  /// @param summary Borrowed summary valid only for the duration of this call.
  virtual void publish(const RuntimeTerminalSummary& summary) = 0;
};

/// Identify the admitted ingress route for one lifecycle-control receipt.
enum class RuntimeLifecycleRequestOrigin : std::uint8_t {
  kParameter,
  kExternal,
  kComputeSelf,
  kComputeCrossTarget,
};

/// Identify the terminal publication or recovery result for one accepted receipt.
enum class RuntimeLifecycleRecoveryDisposition : std::uint8_t {
  kPending,
  kPublished,
  kRestoredOldEpoch,
  kFailedStopped,
  kCanceled,
};

/// Retain bounded request, fence, transaction, and recovery lineage for one accepted receipt.
///
/// This private record is deliberately pointer-free and contains no log prose. The lifecycle
/// control state preallocates exactly the plan's receipt bound, updates a record in place as its
/// transaction advances, and publishes the terminal array only after ingress closes. Epoch one is
/// the initially published lifecycle generation; a successful restart advances it, rollback and
/// cancellation retain it, and failed-stop leaves no effective epoch. The four failure slots keep
/// candidate, candidate-rollback, restoration, and restoration-rollback evidence independent so
/// later cleanup failures never overwrite the failure that initiated recovery.
struct RuntimeLifecycleReceiptDiagnostic {
  RuntimeLifecycleRequestOrigin origin{RuntimeLifecycleRequestOrigin::kExternal};
  OperatorRef requester{};
  LifecycleRestartReceipt receipt{};
  LifecycleTransactionState state{LifecycleTransactionState::kRejected};
  LifecycleTransactionId transaction_id{};
  LifecycleTransactionId coalesced_into_transaction{};
  LifecycleTransactionId related_transaction_id{};
  LifecycleFenceId fence_id{};
  LifecycleStage effective_stage{LifecycleStage::kConfigure};
  ParamGeneration candidate_generation{};
  LifecycleEpoch previous_epoch{};
  LifecycleEpoch candidate_epoch{};
  LifecycleEpoch effective_epoch{};
  LifecycleRestartReason reason{};
  RuntimeLifecycleRecoveryDisposition recovery{RuntimeLifecycleRecoveryDisposition::kPending};
  LifecycleExecutionFailure primary_failure;
  LifecycleExecutionFailure rollback_failure;
  LifecycleExecutionFailure restoration_failure;
  LifecycleExecutionFailure restoration_rollback_failure;
};

static_assert(std::is_standard_layout_v<RuntimeLifecycleReceiptDiagnostic>);
static_assert(std::is_trivially_copyable_v<RuntimeLifecycleReceiptDiagnostic>);

/// Retain one latest-start violation without widening the public callback surface.
///
/// The source and reference coordinates remain distinct because the budget is added in the
/// source domain and only the computed latest start is converted into the reference domain.
/// `dispatch_start` is the final callback-entry observation, not callback completion. A checked
/// lateness of zero is possible only when an exactly late fractional conversion rounds to the
/// same integral reference-clock nanosecond; exact comparison remains the classification source.
struct RuntimeDeadlineMissDiagnostic {
  bool present{};
  DeadlineSourceCoordinateKind source_coordinate{DeadlineSourceCoordinateKind::kTriggerSample};
  TimePoint source_decision_time{};
  TimePoint reference_time{};
  std::int64_t budget_ns{};
  TimePoint latest_start{};
  TimePoint dispatch_start{};
  std::int64_t checked_lateness_ns{};
  OnViolation action{OnViolation::kEmitWitness};
  std::uint32_t callback_count{};
};

static_assert(std::is_standard_layout_v<RuntimeDeadlineMissDiagnostic>);
static_assert(std::is_trivially_copyable_v<RuntimeDeadlineMissDiagnostic>);

/// Bounded managed-shutdown-signal phase carried by private Runtime diagnostics.
enum class RuntimeManagedSignalPhase : std::uint8_t {
  kNone = 0,
  kAcquireDomain,
  kAcquireBinding,
  kSnapshotMask,
  kBlockMask,
  kThreadCensus,
  kQueryDisposition,
  kCreateSignalFd,
  kCreateNotification,
  kNotifyController,
  kDrainNotification,
  kStartBroker,
  kAcceptSignal,
  kPoll,
  kRead,
  kFinalDrain,
  kBindingCleanup,
  kRestoreMask,
  kCloseDomain,
  kDelegateSignal,
};

enum class RuntimeManagedSignal : std::uint8_t {
  kNone = 0,
  kSigInt,
  kSigTerm,
};

enum class RuntimeManagedSignalOccurrence : std::uint8_t {
  kNone = 0,
  kFirst,
  kSecond,
};

enum class RuntimeManagedSignalDisposition : std::uint8_t {
  kUnknown = 0,
  kDefault,
  kIgnored,
  kCustom,
};

enum class RuntimeManagedSignalCensusReason : std::uint8_t {
  kNone = 0,
  kMultipleThreads,
  kCensusUnavailable,
  kCensusChanged,
  kIncompatibleDomain,
  kWrongThread,
  kForkedProcess,
};

/// Pointer-free managed-signal evidence retained beside one structural diagnostic.
struct RuntimeManagedSignalDiagnostic {
  bool present{};
  RuntimeManagedSignalPhase phase{RuntimeManagedSignalPhase::kNone};
  RuntimeManagedSignal signal{RuntimeManagedSignal::kNone};
  RuntimeManagedSignalOccurrence occurrence{RuntimeManagedSignalOccurrence::kNone};
  RuntimeManagedSignalDisposition prior_disposition{RuntimeManagedSignalDisposition::kUnknown};
  RuntimeManagedSignalCensusReason census_reason{RuntimeManagedSignalCensusReason::kNone};
  bool originally_blocked{};
  std::uint64_t domain_generation{};
  std::uint64_t binding_generation{};
  std::uint32_t observed_thread_count{};
  std::int32_t os_error{};
};

static_assert(std::is_standard_layout_v<RuntimeManagedSignalDiagnostic>);
static_assert(std::is_trivially_copyable_v<RuntimeManagedSignalDiagnostic>);

/// Retain exact callback failure evidence beside one public structural diagnostic.
///
/// Public `Diagnostic` remains a stable structural projection and does not overload endpoint or
/// JSON-location fields with callback identity. This session-owned record preserves the original
/// Core `Error` and frozen predicate candidate context for the private diagnostic sink after the
/// allocation-free evaluator path has returned. A default record means that the parallel public
/// diagnostic has no callback-specific detail.
struct RuntimeDiagnosticDetail {
  bool present{};
  Error error{ErrorCode::kSuccess};
  std::string predicate_id;
  OperatorInstanceId operator_id{};
  SourceId source{};
  std::uint64_t source_sequence{};
  Tag tag{};
  TagTime tag_time{};
  TimePoint reference_time{};
  std::uint64_t state_generation{};
  /// Exact latest-start evidence; absent for every non-deadline diagnostic.
  RuntimeDeadlineMissDiagnostic deadline_miss;
  /// Exact cold Operator replay context; absent for every non-reconstruction diagnostic.
  RuntimeReplayMismatchDetail replay_mismatch;
  /// Pointer-free CUDA execution materialization or teardown evidence.
  RuntimeCudaExecutionDiagnostic cuda_execution;
  /// Pointer-free managed SIGINT/SIGTERM domain, acceptance, or cleanup evidence.
  RuntimeManagedSignalDiagnostic managed_signal;
};

/// Own one run's fixed-capacity structural diagnostics and parallel private details.
///
/// The final public slot is initialized during cold construction and reserved for
/// `RUNTIME_DIAGNOSTIC_CAPACITY_EXHAUSTED`. The first append that cannot retain an ordinary
/// record publishes that marker without allocation or replacement of an earlier record. Later
/// appends are ignored because the explicit marker already makes loss observable. Lifecycle
/// receipt lineage uses its separately admitted plan capacity and never shares or competes for
/// these slots.
class RuntimeDiagnosticBuffer {
 public:
  RuntimeDiagnosticBuffer(DiagnosticId overflow_id, DiagnosticSource overflow_source)
      : diagnostics_{} {
    diagnostics_.back() = Diagnostic{
        .id = overflow_id,
        .severity = ::holoscan::DiagnosticSeverity::kError,
        .token = "RUNTIME_DIAGNOSTIC_CAPACITY_EXHAUSTED",
        .source = overflow_source,
    };
  }

  RuntimeDiagnosticBuffer(const RuntimeDiagnosticBuffer&) = delete;
  RuntimeDiagnosticBuffer& operator=(const RuntimeDiagnosticBuffer&) = delete;

  /// Retain one ordinary diagnostic or publish the preconstructed exhaustion marker.
  void append(
      DiagnosticId id, std::string_view token, DiagnosticSource source,
      std::string_view service = {}, const RuntimeDiagnosticDetail* detail = nullptr,
      std::string_view required = {}, std::string_view provided = {},
      ::holoscan::DiagnosticSeverity severity = ::holoscan::DiagnosticSeverity::kError) noexcept {
    const std::size_t index = count_.load(std::memory_order_relaxed);
    if (index >= diagnostics_.size()) {
      return;
    }
    if (index + 1U == diagnostics_.size()) {
      details_[index] = RuntimeDiagnosticDetail{};
      count_.store(index + 1U, std::memory_order_release);
      return;
    }
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
    try {
#endif
      diagnostics_[index] = Diagnostic{
          .id = id,
          .severity = severity,
          .token = std::string{token},
          .source = source,
          .service = std::string{service},
          .required = std::string{required},
          .provided = std::string{provided},
      };
      details_[index] = detail != nullptr ? *detail : RuntimeDiagnosticDetail{};
      count_.store(index + 1U, std::memory_order_release);
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
      // Diagnostic retention is best-effort after terminal state has already become authoritative.
      // NOLINTNEXTLINE(bugprone-empty-catch)
    } catch (...) {
      // The terminal session state remains authoritative if cold diagnostic
      // string/detail storage itself cannot be copied. Count remains unchanged,
      // so the same bounded slot is deterministically reusable by a later append.
    }
#endif
  }

  /// Return the retained public sequence, including an exhaustion marker when present.
  [[nodiscard]] std::span<const Diagnostic> diagnostics() const noexcept {
    return {diagnostics_.data(), count_.load(std::memory_order_acquire)};
  }

  /// Return private details exactly parallel to `diagnostics()`.
  [[nodiscard]] std::span<const RuntimeDiagnosticDetail> details() const noexcept {
    return {details_.data(), count_.load(std::memory_order_acquire)};
  }

 private:
  std::array<Diagnostic, kRuntimeStructuralDiagnosticCapacity> diagnostics_{};
  std::array<RuntimeDiagnosticDetail, kRuntimeStructuralDiagnosticCapacity> details_{};
  std::atomic<std::size_t> count_{};
};

/// Receive terminal, bounded structural-diagnostic sequences from a run.
///
/// This private interface is retained by immutable plan state. Implementations
/// must accept concurrent publication when callers execute the same plan more
/// than once. The borrowed records remain valid only for the duration of the
/// call, so a consumer that needs longer retention must copy at most
/// `kRuntimeStructuralDiagnosticCapacity` total records: at most one fewer ordinary records plus
/// the reserved exhaustion marker. Lifecycle-control receipt records use the separately admitted
/// plan receipt bound. Runtime contains exceptions from either
/// diagnostic-only hook; a sink failure never changes run cleanup or the terminal outcome.
class RuntimeStructuralDiagnosticSink {
 public:
  RuntimeStructuralDiagnosticSink() = default;
  RuntimeStructuralDiagnosticSink(const RuntimeStructuralDiagnosticSink&) = delete;
  RuntimeStructuralDiagnosticSink& operator=(const RuntimeStructuralDiagnosticSink&) = delete;
  virtual ~RuntimeStructuralDiagnosticSink() = default;

  /// Publish the exact terminal sequence retained by the originating session.
  virtual void publish(std::span<const Diagnostic> diagnostics) = 0;

  /// Publish the callback-detail array parallel to the ordinary diagnostic sequence.
  ///
  /// The default preserves compatibility for sinks interested only in public structural
  /// diagnostics. The span has the same length and ordering as the immediately preceding
  /// `publish()` call from the originating session.
  virtual void publish_runtime_details(std::span<const RuntimeDiagnosticDetail> details) {
    static_cast<void>(details);
  }

  /// Publish one managed-signal occurrence before possible prior-policy redelivery.
  ///
  /// This ordinary-context event hook is separate from the final session sequence so a default
  /// second-signal disposition cannot make accepted RECEIVED/ESCALATED evidence unobservable.
  /// Implementations must copy any record they retain beyond the call.
  virtual void publish_managed_signal_event(const Diagnostic& diagnostic,
                                            const RuntimeManagedSignalDiagnostic& detail) {
    static_cast<void>(diagnostic);
    static_cast<void>(detail);
  }

  /// Publish terminal lifecycle receipt lineage retained by the originating session.
  ///
  /// The default preserves compatibility for sinks interested only in ordinary diagnostics.
  /// Records are ordered by acceptance sequence and borrowed only for this call.
  virtual void publish_lifecycle_receipts(
      std::span<const RuntimeLifecycleReceiptDiagnostic> diagnostics) {
    static_cast<void>(diagnostics);
  }

  /// Publish the terminal session's bounded lane-startup evidence.
  ///
  /// The default preserves compatibility for sinks interested only in
  /// structural diagnostics. Records are in dense execution-lane order and
  /// borrowed only for this call; every record has a non-pending disposition.
  virtual void publish_lane_startup_evidence(
      std::span<const ExecutionLaneStartupEvidence> evidence) {
    static_cast<void>(evidence);
  }

  /// Publish `RUNTIME_OPERATOR_COMPUTE_CALL_PROGRESS_TERMINAL` records.
  ///
  /// The default preserves compatibility for diagnostic sinks that do not
  /// consume Operator completion progress. Records are the selected session's
  /// canonical process-group-local plan subsequence and are borrowed only for
  /// this call.
  virtual void publish_operator_completion_progress(
      std::span<const OperatorCompletionProgressRecord> progress) {
    static_cast<void>(progress);
  }
};

}  // namespace holoscan::detail
