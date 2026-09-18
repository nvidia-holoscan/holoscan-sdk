/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string_view>
#include <thread>
#include <vector>

#include <holoscan/core/run.hpp>

#include "../compiler/execution_plan_state.hpp"
#include "../core/execution_context_internal.hpp"
#include "../core/operator_runtime_access.hpp"
#include "../core/source_id_internal.hpp"
#include "../core/temporal_contract_internal.hpp"
#include "dispatcher.hpp"
#include "endpoint_clock_health_runtime.hpp"
#include "execution_lane.hpp"
#include "graph_exec.hpp"
#include "lifecycle_control_state.hpp"
#include "managed_shutdown_signals.hpp"
#include "parameter_control_state.hpp"
#include "predicate_evaluator.hpp"
#include "run_session_port_bindings.hpp"
#include "runtime_callback_context.hpp"
#include "runtime_channel_binding.hpp"
#include "runtime_cuda_execution_resources.hpp"
#include "runtime_diagnostic_sink.hpp"
#include "runtime_direct_composite.hpp"
#include "runtime_notification_source.hpp"
#include "runtime_phase_counters.hpp"
#include "runtime_rate_state.hpp"
#include "runtime_support_thread_control.hpp"
#include "runtime_tensor_resources.hpp"
#include "runtime_termination.hpp"
#include "runtime_worker.hpp"
#include "source_wait.hpp"
#include "temporal_depth_publication.hpp"

namespace holoscan::detail {

class ActivationParameterBinding;
class RunSessionState;

[[nodiscard]] RunSession make_run_session(RunSessionState* state) noexcept;

[[nodiscard]] RunSession start_run_session_retained(
    const std::shared_ptr<const ExecutionPlanState>& retained,
    std::optional<ProcessGroupId> process_group,
    const std::shared_ptr<RuntimeTerminalParticipant>& terminal_participant);

[[nodiscard]] const void* current_run_owned_session() noexcept;

void set_run_owned_session(const void* owner) noexcept;

[[nodiscard]] std::string_view registered_token_or(std::string_view candidate,
                                                   std::string_view fallback) noexcept;

void publish_runtime_terminal_summary(RunGenerationId run_generation,
                                      std::span<const Diagnostic> diagnostics,
                                      RuntimeTerminalSummarySink* sink,
                                      DiagnosticSource fallback_source = {}) noexcept;

void quarantine_run_execution_owners(std::unique_ptr<GraphExec> graph,
                                     std::unique_ptr<RuntimeTensorResources> tensors,
                                     std::shared_ptr<RuntimeChannelSession> channels,
                                     std::unique_ptr<RuntimeCudaExecutionResources> cuda) noexcept;

/// Return whether freshness admission must precede a distinct activation witness.
[[nodiscard]] constexpr bool freshness_input_is_nontrigger_prerequisite(
    const InputSelectionPlanRecord& selection, const PortId witness_port,
    const SourceId readiness_source, const SourceId witness_source) noexcept {
  const bool freshness_selector = selection.selector == InputSelectorKind::kLatestWithin ||
                                  selection.missing == TemporalRecordMissingKind::kReuseLast;
  return freshness_selector && selection.input != witness_port &&
         (witness_port || readiness_source != witness_source);
}

/// Return whether a staged successor may be qualified against the next periodic clock tick.
[[nodiscard]] constexpr bool qualifies_upcoming_clock_freshness(
    const InputSelectionPlanRecord& selection, const bool drop_oldest_refill_supported,
    const bool clock_driven) noexcept {
  return drop_oldest_refill_supported && clock_driven &&
         selection.selector == InputSelectorKind::kLatestWithin && selection.cache_capacity == 1U &&
         selection.missing == TemporalRecordMissingKind::kReuseLast;
}

/// Own one run generation and every runtime resource whose lifetime it bounds.
///
/// Cold startup publishes immutable and preallocated state to the worker; the worker alone
/// performs steady-state dispatch. Public control methods synchronize through the explicit
/// mutex, atomic, and wake members below. Member order is a teardown dependency.
class RunSessionState final {
  static constexpr std::uint64_t kCallbackClaimCountMask = (std::uint64_t{1} << 60U) - 1U;
  static constexpr std::uint64_t kRequestedStop = std::uint64_t{1} << 60U;
  static constexpr std::uint64_t kRequestedCompletion = std::uint64_t{1} << 61U;
  static constexpr std::uint64_t kTerminationFrozen = std::uint64_t{1} << 62U;
  static constexpr std::uint64_t kCallbackAdmissionClosed = std::uint64_t{1} << 63U;

  static_assert(std::atomic<std::uint64_t>::is_always_lock_free,
                "callback admission must use one lock-free publication point");
  static_assert(std::atomic<ParameterUpdateCoordinator*>::is_always_lock_free,
                "callback shutdown must publish parameter ingress without a mutex");

  enum class OperatorCompletionPhase : std::uint8_t {
    kOpen = 0,
    kClosing,
    kClosed,
  };

  enum class InputTerminalDrainPhase : std::uint8_t {
    kOpen = 0,
    kDraining,
    kClosed,
  };

  void initialize_local_operator_completion_progress();

  class CallbackAdmissionClaim final {
   public:
    CallbackAdmissionClaim() = default;
    CallbackAdmissionClaim(const CallbackAdmissionClaim&) = delete;
    CallbackAdmissionClaim& operator=(const CallbackAdmissionClaim&) = delete;
    ~CallbackAdmissionClaim() { release(); }

    [[nodiscard]] bool acquire(std::atomic<std::uint64_t>& state) noexcept {
      std::uint64_t observed = state.load(std::memory_order_acquire);
      for (;;) {
        if ((observed & kCallbackAdmissionClosed) != 0U ||
            (observed & kCallbackClaimCountMask) == kCallbackClaimCountMask) {
          return false;
        }
        if (state.compare_exchange_weak(
                observed, observed + 1U, std::memory_order_acq_rel, std::memory_order_acquire)) {
          state_ = &state;
          return true;
        }
      }
    }

    void release() noexcept {
      if (state_ != nullptr) {
        static_cast<void>(state_->fetch_sub(1U, std::memory_order_acq_rel));
        state_ = nullptr;
      }
    }

   private:
    std::atomic<std::uint64_t>* state_{};
  };

  enum class ComputeAdmissionDisposition : std::uint8_t {
    kPending = 0,
    kAdmitted = 1,
    kDeadlineReadFailed = 2,
    kDeadlineEvaluationFailed = 3,
    kDeadlineFailRun = 4,
    kParameterObservationFailed = 5,
    kCudaDeviceUnavailable = 6,
    kRunClosing = 7,
    kOperatorCompleted = 8,
  };

  static constexpr std::size_t kNoCompletionIndex = std::numeric_limits<std::size_t>::max();
  using ComputeLaneInvoke = expected<void, Error> (*)(void*);

  struct ComputeLaneInvocation {
    RunSessionState* session{};
    CallbackAdmissionClaim callback_claim;
    RuntimeOperatorInstance* instance{};
    ExecutionContextState* context{};
    Dispatcher* dispatcher{};
    const DispatcherSelection* selection{};
    const RuntimeDeadlineBindingRecord* deadline{};
    ClockRef deadline_clock{};
    TimePoint source_decision_time{};
    TimePoint reference_time{};
    std::int64_t deadline_budget_ns{};
    ParameterUpdateCoordinator* parameter_updates{};
    OperatorRef self{};
    const ParameterGenerationPin* parameter_generation{};
    std::uint64_t source_sequence{};
    DeviceId cuda_device{};
    RuntimePhaseCounters* phase_counters{};
    RuntimeChannelSession* channels{};
    std::size_t completion_index{kNoCompletionIndex};
    VertexId channel_owner{};
    std::uint32_t execution_lane{};
    ClockRef direct_continuation_clock{};
    std::optional<DeadlineGateResult> deadline_gate;
    std::optional<RuntimeChannelDirectContinuationReceipt> direct_continuation_receipt;
    Error admission_error{ErrorCode::kSuccess};
    Error direct_continuation_error{ErrorCode::kSuccess};
    ComputeAdmissionDisposition admission{ComputeAdmissionDisposition::kPending};
    bool callback_failed{};
    bool cuda_device_restore_failed{};
    bool completed_selection{};
    bool direct_continuation_enabled{};
    bool direct_continuation_clock_failed{};
  };

  // Lane invocation keeps CUDA selection, deadline sampling, callback admission, Parameter
  // observation, user compute, device restoration, and direct continuation in order. Completion
  // admission is a source-private wrapper so a plan without completion rules executes this plain
  // lane body without a per-callback completion branch.
  template <bool TraceCallback>
  // NOLINTNEXTLINE(readability-function-cognitive-complexity)
  [[nodiscard]] static expected<void, Error> invoke_compute_lane(void* state) noexcept;

  // Completion is the first lane-side admission fence. A stale selection must not mutate any
  // per-callback admission state, including CUDA device selection, deadline evidence, or
  // Parameter observation. The wrapper tail-calls the single plain lane body after admission.
  template <bool TraceCallback, bool ObserveFinalEntry>
  [[nodiscard]] static expected<void, Error> invoke_compute_lane_with_completion(
      void* state) noexcept;

  [[nodiscard]] static ComputeLaneInvoke select_compute_lane_with_completion_invoke(
      bool trace_callback, bool observe_final_entry) noexcept;

 public:
  struct SharedStartupFailureTag {};
  struct ProcessGroupSelectionFailureTag {};
  struct ManagedSignalAdmissionFailureTag {};

  RunSessionState(std::shared_ptr<const ExecutionPlanState> plan, RunGenerationId run_generation,
                  ProcessGroupId process_group);

  /// Complete the fallible startup work after session-local progress cells exist.
  void initialize_after_progress_materialization();

  void fail_startup() noexcept;

  RunSessionState(std::shared_ptr<const ExecutionPlanState> plan, RunGenerationId run_generation,
                  ProcessGroupSelectionFailureTag);

  RunSessionState(std::shared_ptr<const ExecutionPlanState> plan,
                  const ManagedSignalAdmissionFailure& failure, ManagedSignalAdmissionFailureTag);

  explicit RunSessionState(SharedStartupFailureTag);

  RunSessionState(const RunSessionState&) = delete;
  RunSessionState& operator=(const RunSessionState&) = delete;
  ~RunSessionState();

  [[nodiscard]] static RunSessionState* shared_startup_failure() noexcept;

  void attach_terminal_participant(
      std::shared_ptr<RuntimeTerminalParticipant> participant) noexcept;

  void release_owner() noexcept;

  void publish_startup_failure(const std::shared_ptr<const ExecutionPlanState>& attempted_plan,
                               RunGenerationId attempted_run_generation = {}) const noexcept;

  void start() noexcept;

  [[nodiscard]] bool is_running() const noexcept;

  [[nodiscard]] RunTermination termination() const noexcept;

  void request_stop() noexcept;

  static void request_termination(void* opaque, detail::TerminationRequest request) noexcept;

  [[nodiscard]] bool publish_termination_request(detail::TerminationRequest request) noexcept;

  void request_callback_termination(detail::TerminationRequest request) noexcept;

  void request_shutdown() noexcept;

  void wait();

  [[nodiscard]] std::span<const Diagnostic> diagnostics() const noexcept;

  [[nodiscard]] std::span<const OperatorCompletionProgressRecord> operator_completion_progress()
      const noexcept;

  void count_phase(RuntimePhaseCounter counter) noexcept;

  /// Invoke one steady-state task directly on the already verified inline owner.
  ///
  /// The controller and sole partition runner are the same thread in this
  /// specialization. Re-entering ExecutionLaneSet would only reacquire the
  /// lane's stop/serialization locks around a callback that cannot migrate.
  [[nodiscard]] expected<void, Error> invoke_inline_owner_task(
      const ExecutionLaneTask& task) noexcept;

  void signal_controller_wake(RuntimePhaseCounter counter) noexcept;

  void signal_channel_owner_progress(VertexId owner) noexcept;

  void trace_phase(RuntimePhaseTracePoint point, std::size_t receiver_index, SourceId source,
                   std::uint64_t source_sequence) noexcept;

  [[nodiscard]] expected<OperatorRef, Error> operator_ref(OperatorRef plan_ref) const noexcept;

  [[nodiscard]] expected<LifecycleControlHandle, Error> lifecycle_control(
      const LifecycleAccessPolicy& policy);

  [[nodiscard]] expected<ParamControlHandle, Error> parameter_control(
      const ParamAccessPolicy& policy);

  [[nodiscard]] expected<ParamStopFenceResult, Error> close_parameter_update_ingress();

  [[nodiscard]] expected<ParamSnapshot, Error> capture_params(ParamCaptureOptions options);

  [[nodiscard]] expected<RateGenerationSnapshot, Error> capture_rate_generations();

 private:
  void fail_worker_start() noexcept;

  [[nodiscard]] expected<void, Error> attach_parameter_control(ParameterUpdateCoordinator& updates);

  void attach_rate_snapshot_state(RuntimeRateState& rates) noexcept;

  void terminalize_parameter_updates_at_run_end() noexcept;

  void retire_parameter_control() noexcept;

  [[nodiscard]] expected<void, Error> release_execution_resources(
      RuntimeCudaExecutionDiagnostic* cuda_diagnostic);

  void append_callback_diagnostic(
      std::string_view token, VertexId vertex, std::string_view service, const Error& error,
      const RuntimeCallbackIdentity& identity, std::string_view predicate_id = {},
      const RuntimeDeadlineMissDiagnostic* deadline = nullptr,
      ::holoscan::DiagnosticSeverity severity = ::holoscan::DiagnosticSeverity::kError) noexcept;

  void append_compute_diagnostics(const RuntimeOperatorInstance& instance,
                                  const ComputeLaneInvocation& invocation, const Error& error,
                                  const RuntimeCallbackIdentity& identity) noexcept;

  void append_lifecycle_diagnostics(
      const GraphExec& exec,
      ::holoscan::DiagnosticSeverity severity = ::holoscan::DiagnosticSeverity::kError) noexcept;

  void append_diagnostic(
      std::string_view token, VertexId vertex = {}, std::string_view service = {},
      const RuntimeDiagnosticDetail* detail = nullptr, std::string_view required = {},
      std::string_view provided = {},
      ::holoscan::DiagnosticSeverity severity = ::holoscan::DiagnosticSeverity::kError) noexcept;

  void append_managed_signal_diagnostic(std::string_view token,
                                        const RuntimeManagedSignalDiagnostic& managed_detail,
                                        bool publish_event) noexcept;

  void worker_main() noexcept;

  void worker_main_with_thread_usage() noexcept;

  /// Own run-wide startup resources borrowed by every partition-local readiness state.
  struct ReadySharedResources {
    std::shared_ptr<EndpointClockHealthRuntimeCoordinator> clock_health;
    std::shared_ptr<ExecutionLaneSet> lanes;
    std::shared_ptr<RuntimeChannelSession> channels;
    std::shared_ptr<RuntimeNotificationSourceRegistry> notifications;
    std::vector<RuntimeChannelPortReplayRecord> channel_replay_ports;
    std::vector<std::uint8_t> channel_activation_owners;
    std::vector<std::uint8_t> channel_input_owners;
  };

  /// Own one partition runner's frozen bindings, bounded scratch, and dispatch state.
  struct ReadyRunState {
    // Set when this round's poll loop abandoned a pass on a contended cache lock, so a readiness
    // change published earlier in the same round has not become a candidate yet.
    bool poll_incomplete_this_round{};
    // How many readiness events the poll loop ingested into the coordinator this round. Zero on a
    // losing round means the poll produced nothing to freeze; non-zero means it produced something
    // that was not this selector's successor.
    std::uint32_t round_ingested_count{};
    // Set when the reconcile released a record this round. Paired with an incomplete poll it is
    // what makes the blocking retry below both necessary and rare.
    bool freshness_evicted_this_round{};
    struct CompletionReceiverState {
      std::size_t receiver{};
      std::size_t completion_index{};
    };

    struct InputTerminalDrainState {
      std::size_t receiver{};
      InputTerminalDrainPhase phase{InputTerminalDrainPhase::kOpen};
    };

    struct InputSelectorSlot {
      VertexId owner{};
      PortId input{};
      std::size_t receiver_index{std::numeric_limits<std::size_t>::max()};
      PortId witness_port{};
      SourceId witness_source{};
      SourceId readiness_source{};
      // True only when this input's arrivals populate selector state but cannot activate the
      // owning Operator under its trigger contract.
      bool nontrigger_prerequisite{};
      bool freshness_reconcile{};
      bool connected{true};
      bool upcoming_clock_freshness{};
      RuntimeInputSelectionState selector;
      std::optional<std::uint32_t> derived_term_index;
      std::uint64_t arrival_watermark{};
      std::uint64_t producer_epoch{1U};
      std::string_view predicate_id;
      std::uint32_t predicate_ordinal{};
      bool predicate_input{};
      bool aligned_member{};
      bool end_of_stream_received{};
      // Set when the superseded-retention reconcile released this slot's record in the current
      // freeze round, and rewritten at the top of every round. Diagnostics only: a run total
      // cannot say whether an eviction cost a particular activation its selection, because in
      // steady state the reconcile fires tens of times per activation and only the one in the
      // round that froze the witness could have emptied the cache it read.
      bool freshness_evicted_this_round{};
      // Whether that eviction's claim release actually admitted a staged successor. False means
      // the release freed nothing and the cache is empty with nothing in flight.
      bool freshness_promoted_this_round{};
      // Whether the causality gate had no floor to compare against, and so did not run.
      bool freshness_gate_inert_this_round{};
      // Coordinates the eviction was decided on, and the witness reference the selection that
      // follows is formed at. Retained so a lost selection can report the comparison that let it
      // happen rather than only that it happened.
      std::int64_t freshness_successor_capture_ns{};
      std::int64_t freshness_floor_ns{};
      // Capture of the record actually promoted, which need not be the one the gate approved:
      // the approved successor can be dropped from staging before the promotion reaches it.
      std::int64_t freshness_promoted_capture_ns{};
      // Durable marker for a committed ReuseLast handoff until that exact non-trigger input and
      // every earlier arrival on the same port are ingested, or its selector/producer generation
      // resets. The complete temporal identity disambiguates equal capture coordinates and lets
      // the prerequisite path prove that it has reached, rather than passed, the successor. Store
      // the capture coordinate normalized into the selector's reference clock, matching the exact
      // record ReadinessBinding publishes for temporal_sample().
      // Exact promoted arrival that must be ingested before its authorizing non-trigger witness.
      std::optional<TemporalSampleRecord> freshness_refill_prerequisite;
      // Newest witness-qualified arrival fenced while surviving predecessors are admitted first.
      std::optional<TemporalSampleRecord> freshness_refill_target;
      std::int64_t selection_reference_ns{};
      // Whether any record reached this selector's cache during the current freeze round.
      bool freshness_ingested_this_round{};
      bool end_of_stream_decision_pending{};
      std::vector<TemporalSampleRecord> channel_payload_claims;
      std::vector<TemporalSampleRecord> channel_payload_claim_scratch;
      std::vector<TemporalSampleRecord> channel_payload_retire_scratch;
    };

    struct OperatorParameterPolicy {
      std::vector<std::uint32_t> selector_indices;
      std::optional<std::uint32_t> deadline_term_index;
    };

    struct AlignedSelectorSlot {
      VertexId owner{};
      RuntimeAlignedSelectionState selector;
    };

    struct ClockBindingRequest {
      RuntimePortBindingRequest source;
      ClockTriggerPlanRecord schedule;
      std::size_t receiver_index{};
    };

    struct FreshnessBindingSlot {
      PortBinding* binding{};
      bool poll_complete_this_round{};
      // Distinguish an actual nonblocking poll that declined on Channel serialization from a
      // binding that was not polled because an ordinary coordinator bound stopped the pass.
      bool ordinary_poll_incomplete_this_round{};
    };

    /// Reused hot-path selection storage sized during cold readiness materialization.
    struct SelectionScratch {
      std::vector<DispatcherCandidate> candidates;
      std::vector<FrozenRuntimeCandidate> frozen_candidates;
      std::vector<RuntimeInputSelectionState::Transaction> selector_transactions;
      std::vector<RuntimeAlignedSelectionState::Transaction> aligned_transactions;
      std::vector<InputSelection> input_selections;
      std::vector<TemporalSampleRecord> selected_samples;
      std::size_t maximum_selected_samples{};
      // Input-arrival triggers can name ports without selectors. Keep the trigger pin outside
      // selected_samples so no Channel borrow points into a vector that later appends can move.
      std::optional<TemporalSampleRecord> arrival_trigger_sample;
      std::vector<RuntimeChannelActivationSelection> channel_selections;
      std::vector<PredicateFrozenInput> predicate_inputs;
    };

    struct RestartScratch {
      std::vector<std::size_t> receivers;
      std::vector<VertexId> member_vertices;
      std::vector<VertexId> vertices;
      std::vector<VertexId> aggregate_member_vertices;
      std::vector<VertexId> aggregate_vertices;
      std::vector<RuntimeEvent> events;
      std::vector<std::optional<TimePoint>> phases;
    };

    struct DirectChainState {
      std::optional<RuntimeChannelDirectContinuationReceipt> continuation_receipt;
      std::vector<RuntimeDirectCompositeBinding> composites_by_receiver;
      std::vector<std::size_t> composite_owner_by_receiver;
      std::vector<std::size_t> successor_by_receiver;
      std::vector<PortBinding*> binding_by_producer;
      SourceId root_source{};
      ClockId root_clock{};
      ClockTriggerPortBinding* root_binding{};
      std::size_t root_receiver{std::numeric_limits<std::size_t>::max()};
      std::optional<TimePoint> next_root_wake;
    };

    SchedulerPlanRecord dispatcher_scheduler;
    std::unique_ptr<Dispatcher> dispatcher;
    std::unique_ptr<RuntimeCandidateCoordinator> coordinator;
    // These views borrow ReadySharedResources. Refresh them after any backing-owner or vector
    // mutation, and destroy every ReadyRunState before the shared owner.
    EndpointClockHealthRuntimeCoordinator* clock_health{};
    ExecutionLaneSet* lanes{};
    RuntimeChannelSession* channels{};
    RuntimeNotificationSourceRegistry* notifications{};
    // Selected once after the partition-local completion inventory is materialized. A partition
    // with no completion rules calls the plain lane directly and performs no completion lookup or
    // admission branch on recurring activations.
    ComputeLaneInvoke compute_lane_invoke{};
    std::span<const std::uint8_t> channel_activation_owners;
    std::span<const std::uint8_t> channel_input_owners;
    std::vector<RuntimePortBindingRequest> binding_requests;
    std::vector<std::pair<VertexId, RuntimePortBindingRequest>> channel_binding_requests;
    std::vector<ClockBindingRequest> clock_binding_requests;
    std::size_t expected_binding_count{};
    std::vector<std::unique_ptr<PortBinding>> notification_bindings;
    std::vector<std::unique_ptr<PortBinding>> bindings;
    std::vector<FreshnessBindingSlot> freshness_bindings;
    // Present only when at least one binding owns a freshness selector. Each entry maps the
    // same-position binding to its cold recovery slot, avoiding a per-round linear lookup while
    // leaving ordinary graphs allocation-free.
    std::vector<std::size_t> freshness_binding_index_by_binding;
    // One flag per same-position binding: set for Channel readiness bindings, whose ordinary poll
    // can decline on a contended input cache lock and may retry once before the owner parks.
    // Physical bindings never decline, so this keeps the incomplete-status query off their path.
    std::vector<std::uint8_t> contention_retry_eligible_by_binding;
    std::vector<ClockTriggerPortBinding*> clock_bindings;
    std::vector<InputSelectorSlot> input_selectors;
    // Reused only while a non-trigger freshness prefix is live. Each fence exposes that input's
    // oldest arrival while keeping the owner's activation witnesses pending; unrelated owners
    // retain the coordinator's ordinary global ordering.
    std::vector<RuntimeCandidateInputFence> freshness_input_fences;
    std::vector<OperatorParameterPolicy> parameter_policies;
    std::vector<AlignedSelectorSlot> aligned_selectors;
    std::vector<ActivationSelectionState> activation_selections;
    // Sorted, partition-local ownership records keep per-dispatch and restart lookup bounded
    // without allocating global-operator-sized tables in every partition.
    std::vector<CompletionReceiverState> completion_receivers;
    std::vector<InputTerminalDrainState> input_terminal_receivers;
    // Runtime sweeps visit only admitted terminal work. A receiver occurs at most once in each
    // active list because insertion happens only on the corresponding open-to-active transition.
    std::vector<CompletionReceiverState> active_completion_receivers;
    std::vector<std::size_t> active_input_terminal_receivers;
    std::vector<std::size_t> closed_input_terminal_receivers;
    std::vector<std::optional<RuntimePredicateEvaluatorEngine>> predicates;
    std::vector<PredicateReceiver> predicate_receivers;
    std::vector<std::optional<RuntimeDeadlineBindingRecord>> deadline_bindings;
    SelectionScratch selection;
    RestartScratch restart;
    DirectChainState direct_chain;
    std::vector<std::uint8_t> owned_receivers;
    RuntimeWakeState* wake{};
    const PartitionRunnerPlanRecord* runner{};
    std::optional<std::uint32_t> inline_execution_lane;
    bool inline_partition_execution{};
    bool partition_owner_execution{};

    [[nodiscard]] bool uses_inline_invoke() const noexcept;
  };

  [[nodiscard]] static const ReadyRunState::CompletionReceiverState* completion_receiver_state(
      const ReadyRunState& state, std::size_t receiver) noexcept;

  [[nodiscard]] static ReadyRunState::InputTerminalDrainState* input_terminal_receiver_state(
      ReadyRunState& state, std::size_t receiver) noexcept;

  [[nodiscard]] static const ReadyRunState::InputTerminalDrainState* input_terminal_receiver_state(
      const ReadyRunState& state, std::size_t receiver) noexcept;

  static void bind_shared_resources(ReadyRunState& state, ReadySharedResources& shared) noexcept;

  class StartupRuntimeTeardown final {
   public:
    StartupRuntimeTeardown(RunSessionState& session, ReadySharedResources& shared) noexcept;
    StartupRuntimeTeardown(const StartupRuntimeTeardown&) = delete;
    StartupRuntimeTeardown& operator=(const StartupRuntimeTeardown&) = delete;
    ~StartupRuntimeTeardown() noexcept;

    void dismiss() noexcept;

   private:
    RunSessionState* session_{};
    ReadySharedResources* shared_{};
    bool active_{true};
  };

  [[nodiscard]] static bool owns_receiver(const ReadyRunState& state,
                                          std::size_t receiver) noexcept;

  [[nodiscard]] static bool owns_vertex(const ReadyRunState& state, const GraphExec& exec,
                                        VertexId vertex) noexcept;

  enum class ClosedIslandActivationDisposition : std::uint8_t {
    kCompleted = 0,
    kNoFire,
    kStopped,
    kRestarted,
    kFailed,
  };

  struct ClosedIslandActivationResult {
    RuntimeCallbackIdentity callback_identity{};
    std::optional<RuntimeChannelDirectContinuationReceipt> direct_continuation_receipt;
    Error error{ErrorCode::kSuccess};
    ClosedIslandActivationDisposition disposition{ClosedIslandActivationDisposition::kFailed};
    bool callback_attempted{};
    bool callback_completed{};
    bool activation_retired{};
    bool activation_cut_completed{};
    bool direct_continuation_clock_failed{};
  };

  /// Borrow the worker-owned startup graph and resources through lifecycle materialization.
  struct RateResourceMaterializationState {
    RunSessionState* session{};
    GraphExec* exec{};
    ReadySharedResources* shared{};
    std::span<ReadyRunState> ready_states;
    RuntimeCudaExecutionResources* cuda{};
    bool defer_startup{};
    bool resources_materialized{};
  };

  [[nodiscard]] bool has_owned_unwired_inputs(const GraphExec& exec,
                                              const ReadyRunState& state) const;

  [[nodiscard]] bool initialize_channel_session(GraphExec& exec, ReadyRunState& state,
                                                ReadySharedResources& shared,
                                                const RuntimeCudaExecutionResources& cuda_resources,
                                                RuntimeChannelSizingOverride sizing_override = {});

  [[nodiscard]] expected<RuntimeQueueCapacity, Error> selector_rate_capacity(
      PortId input, std::uint32_t planned_capacity,
      const RuntimeRateMaterializationView* materialization) const noexcept;

  // Materialize each selector together with its parameter-derived term and retained-claim bounds;
  // partial initialization cannot be exposed to dispatch.
  [[nodiscard]] bool initialize_input_selectors(
      GraphExec& exec, ReadyRunState& state,
      const RuntimeRateMaterializationView* materialization = nullptr);

  [[nodiscard]] bool initialize_aligned_selectors(
      GraphExec& exec, ReadyRunState& state,
      const RuntimeRateMaterializationView* materialization = nullptr);

  // Runner materialization validates and commits the complete per-partition runtime state.
  [[nodiscard]] bool materialize_partition_runner_states(const GraphExec& exec,
                                                         std::vector<ReadyRunState>& states);

  [[nodiscard]] bool initialize_lanes(const GraphExec& exec, ReadyRunState& state,
                                      ReadySharedResources& shared);

  [[nodiscard]] bool stop_lanes(ExecutionLaneSet* lanes) noexcept;

  void teardown_startup_runtime(ReadySharedResources& shared) noexcept;

  [[nodiscard]] expected<std::optional<RuntimePredicateEvaluatorEngine>, Error>
  initialize_predicate(const RuntimeOperatorInstance& instance,
                       const TemporalTriggerPlanRecord& trigger, PredicateReceiver& receiver);

  // Freeze one coherent readiness inventory before any source can publish into the Dispatcher.
  [[nodiscard]] bool initialize_ready_state(
      GraphExec& exec, ReadyRunState& state, ReadySharedResources& shared,
      const RuntimeRateMaterializationView* materialization = nullptr);

  [[nodiscard]] std::optional<RuntimeNotificationSourceDescriptor> resolve_notification_descriptor(
      GraphExec& exec, const RuntimePortBindingRequest& request) noexcept;

  // Bind production notifications before Arm so lifecycle callbacks can acquire sender authority.
  // One registry owns every participating partition; posts remain not-ready until the complete
  // Start barrier succeeds. Single-state insertion order and shared-state canonical source order
  // remain the same as the pre-consolidation paths.
  [[nodiscard]] bool bind_shared_production_notifications(GraphExec& exec,
                                                          std::span<ReadyRunState> states,
                                                          ReadySharedResources& shared);

  // Bind every remaining physical, Channel, and clock source before opening lifecycle ingress;
  // partial binding is a startup failure and must not publish a runnable session.
  [[nodiscard]] bool bind_ready_sources(GraphExec& exec, ReadyRunState& state,
                                        bool manage_channel_session = true);

  [[nodiscard]] bool cancel_completion_receiver(ReadyRunState& state,
                                                std::size_t receiver) noexcept;

  [[nodiscard]] bool begin_zero_completion_closures(ReadyRunState& state) noexcept;

  [[nodiscard]] bool refresh_and_arm_channels(RuntimeChannelSession* channels);

  [[nodiscard]] PortBinding* physical_binding_for(ReadyRunState& state, SourceId source) noexcept;

  // Direct-chain admission derives one all-or-nothing topology and binding contract.
  [[nodiscard]] bool initialize_direct_chain(GraphExec& exec, ReadyRunState& state) noexcept;

  // Composite initialization preserves conservative fallback across every candidate island.
  [[nodiscard]] bool initialize_direct_composites(GraphExec& exec, ReadyRunState& state) noexcept;

  template <bool HandlesTerminal>
  [[nodiscard]] expected<void, Error> retire_frozen_source_impl(
      ReadyRunState& state, FrozenRuntimeCandidate& candidate) noexcept;

  [[nodiscard]] expected<void, Error> retire_frozen_source(
      ReadyRunState& state, FrozenRuntimeCandidate& candidate) noexcept;

  [[nodiscard]] expected<void, Error> retire_nonterminal_frozen_source(
      ReadyRunState& state, FrozenRuntimeCandidate& candidate) noexcept;

  [[nodiscard]] bool cancel_ready_sources(ReadyRunState& state) noexcept;

  [[nodiscard]] bool stop_channels(RuntimeChannelSession* channels) noexcept;

  // Validate every member's pending-window state before publishing a timeout witness for the
  // operator, preserving one coherent closure and clock coordinate.
  [[nodiscard]] bool enqueue_due_window_timeouts(GraphExec& exec, ReadyRunState& state) noexcept;

  [[nodiscard]] bool arrival_trigger_pin_matches(
      const RuntimeOperatorInstance& instance,
      const FrozenRuntimeCandidate& candidate) const noexcept;

  [[nodiscard]] bool arrival_trigger_pin_matches(
      const GraphExec& exec, const FrozenRuntimeCandidate& candidate) const noexcept;

  [[nodiscard]] expected<void, Error> synchronize_temporal_payload_claims(
      ReadyRunState& state, ReadyRunState::InputSelectorSlot& slot,
      std::uint32_t* promoted_out = nullptr,
      const RuntimeChannelFreshnessReservation* freshness_reservation = nullptr,
      std::int64_t* promoted_capture_out = nullptr) noexcept;

  [[nodiscard]] bool synchronize_temporal_payload_claims(ReadyRunState& state,
                                                         VertexId owner) noexcept;

  [[nodiscard]] bool synchronize_temporal_payload_claims(ReadyRunState& state) noexcept;

  /// Release a superseded drop-oldest candidate so the newest staged sample can be admitted.
  ///
  /// A scalar selector that retains a record marks its transport slot \c temporal_retained, and
  /// \c writable_slot() excludes retained slots from drop-oldest replacement. At depth one the
  /// sole typed slot therefore stays pinned for the whole inter-activation interval: every sample
  /// produced meanwhile is staged and then dropped by the staging pool's own drop-oldest, so the
  /// next activation selects the record admitted one period ago rather than the newest causal one.
  ///
  /// Releasing the claim here rather than at selection time is what lets the readiness pipeline
  /// finish the job. The ordinary source poll and ingest first establish a complete existing
  /// prefix. Retiring the claim then promotes the reserved staged sample and publishes its
  /// readiness change; the source-local blocking poll turns that revision into an event, and
  /// ingest transfers the payload claim to the selector before this dispatch iteration selects.
  /// The readiness pipeline therefore keeps sole ownership of admission and nothing is injected
  /// into the cache by hand.
  ///
  /// Only a strictly newer staged sample justifies giving up the current one: releasing with
  /// nothing behind it would turn a stale selection into a missing one, which is worse. Every
  /// refusal before the eviction leaves the existing candidate untouched.
  /// Bound the reference coordinate of every witness this freeze round can produce.
  ///
  /// freeze_next() orders candidates by reference time first, so a staged sample whose capture
  /// coordinate is at or below this floor sorts ahead of every witness the round can freeze, and
  /// is therefore ingested before selection reads the cache. A sample above the floor may not be,
  /// which is exactly the window in which releasing the retained record leaves the cache empty for
  /// a witness the record would have satisfied.
  ///
  /// Both contributors are needed. Pending candidates cover witnesses the ordinary poll already
  /// made visible; clock bindings conservatively include their next admitted tick so an idle
  /// coordinator cannot justify a successor beyond the next activation horizon.
  ///
  /// @param state Run state for this freeze round.
  /// @param floor Receives the bound, or \c std::nullopt when no witness is possible this round.
  /// @return False when a clock could not be read, leaving \p floor meaningless. No floor means no
  ///         eviction can be justified, so the caller must decline rather than proceed unbounded.
  [[nodiscard]] static bool witness_reference_floor(ReadyRunState& state,
                                                    std::optional<TimePoint>& floor) noexcept;

  /// Resolve the next witness owned by one selector's readiness source.
  ///
  /// A global causal floor cannot prove that staged data is selectable by this selector: Exact
  /// requires coordinate equality and LatestWithin requires membership in its own age window.
  [[nodiscard]] static std::optional<TimePoint> selector_witness_reference(
      ReadyRunState& state, VertexId owner, SourceId witness_source, PortId witness_port) noexcept;

  [[nodiscard]] static bool input_arrival_can_activate(const TemporalTriggerPlanRecord& trigger,
                                                       const PredicateTriggerPlanRecord* predicate,
                                                       PortId input) noexcept;

  [[nodiscard]] bool reconcile_superseded_retention(ReadyRunState& state, SourceId readiness_source,
                                                    std::size_t& refill_credit,
                                                    std::size_t& recovered_count) noexcept;

  // Channel ingestion atomically reconciles producer epochs, temporal selectors, bracket anchors,
  // predicate state, and payload claims for one frozen witness.
  /// Freshness prerequisite draining may supply the immutable temporal record it already resolved
  /// while proving prefix order. Ordinary ingestion leaves it null and resolves the record once.
  [[nodiscard]] bool ingest_channel_candidate(
      GraphExec& exec, ReadyRunState& state, const FrozenRuntimeCandidate& candidate,
      const TemporalSampleRecord* pre_resolved_sample = nullptr);

  [[nodiscard]] bool channel_candidate_is_eligible(
      const GraphExec& exec, const ReadyRunState& state,
      const FrozenRuntimeCandidate& candidate) const noexcept;

  // Form the bounded activation snapshot, health disposition, and retirement obligations in one
  // pass so dispatch never observes a partially frozen candidate set.
  [[nodiscard]] bool freeze_ready_candidates(
      GraphExec& exec, ReadyRunState& state,
      std::optional<std::uint32_t> current_execution_lane = std::nullopt);

  enum class CausalFrontierDisposition : std::uint8_t {
    kAccepted,
    kHandled,
    kFallback,
    kFailed,
  };

  // Causal continuation validation and fallback form one ordered frontier transition.
  [[nodiscard]] CausalFrontierDisposition freeze_causal_successor(GraphExec& exec,
                                                                  ReadyRunState& state,
                                                                  std::size_t producer_receiver,
                                                                  TagTime parent_tag_time) noexcept;

  // Freeze the exact lifecycle-member and Channel-dependent receiver closure, including saved
  // clock phases, before any restart fence mutates dispatcher or transport state.
  [[nodiscard]] bool prepare_runtime_restart_scope(
      GraphExec& exec, ReadyRunState& state, std::span<const OperatorInstanceId> members,
      std::span<const VertexId> member_vertices) noexcept;

  [[nodiscard]] bool prepare_runtime_restart_aggregate(
      GraphExec& exec, std::span<ReadyRunState> states,
      std::span<const OperatorInstanceId> members) noexcept;

  [[nodiscard]] expected<bool, Error> restart_scope_uses_channels(
      const ReadyRunState& aggregate) const noexcept;

  [[nodiscard]] bool fence_runtime_restart_state(ReadyRunState& state) noexcept;

  [[nodiscard]] bool fence_runtime_restart(GraphExec& exec, std::span<ReadyRunState> states,
                                           std::span<const OperatorInstanceId> members,
                                           LifecycleStage from_stage,
                                           bool& channels_quiesced) noexcept;

  // Retire the coordinator and binding prefixes for the complete prepared restart scope before any
  // source generation is advanced.
  [[nodiscard]] bool retire_restart_witnesses(ReadyRunState& state) noexcept;

  // Publish Channel, selector, readiness-source, clock, and predicate generations as one ordered
  // restart transaction after the exact affected scope has been fenced.
  [[nodiscard]] bool publish_runtime_restart_state(GraphExec& exec, ReadyRunState& state) noexcept;

  [[nodiscard]] bool publish_runtime_restart(GraphExec& exec, std::span<ReadyRunState> states,
                                             OperatorRef target, bool channels_quiesced) noexcept;

  [[nodiscard]] bool rollback_runtime_restart(std::span<ReadyRunState> states,
                                              bool channels_quiesced) noexcept;

  void discard_rate_candidate_temporal_depths(std::span<ReadyRunState> states) noexcept;

  [[nodiscard]] expected<void, Error> prepare_rate_candidate_temporal_depths(
      GraphExec& exec, std::span<ReadyRunState> states,
      const RuntimeRateMaterializationView& materialization) noexcept;

  [[nodiscard]] static expected<void, Error> ensure_rate_candidate_resources(
      void* state, const RuntimeRateMaterializationView& materialization, LifecycleStage stage,
      LifecycleTransitionPhase phase) noexcept;

  // Restart processing is one rollback-capable transaction across all runtime subsystems.
  [[nodiscard]] bool process_pending_restarts(GraphExec& exec,
                                              std::span<ReadyRunState> states) noexcept;

  [[nodiscard]] bool process_pending_restarts(GraphExec& exec, ReadyRunState& state) noexcept;

  enum class SelectionPreparation : std::uint8_t {
    kReady,
    kNoFire,
    kFailed,
  };

  struct PredicateComputeSelectionContext {
    RunSessionState* session{};
    ReadyRunState* ready{};
    const RuntimeOperatorInstance* instance{};
    const PredicateTriggerPlanRecord* predicate{};
    const FrozenRuntimeCandidate* candidate{};
    TimePoint* semantic_reference{};
  };

  void clear_input_selection_scratch(ReadyRunState& state, const FrozenRuntimeCandidate& candidate,
                                     TimePoint& semantic_reference) noexcept;

  [[nodiscard]] bool append_selected_samples(
      ReadyRunState& state, std::span<const TemporalSampleRecord> samples) noexcept;

  [[nodiscard]] bool append_frozen_input_selection(
      ReadyRunState& state, ReadyRunState::InputSelectorSlot& slot,
      RuntimeInputSelectionState::Transaction& transaction) noexcept;

  [[nodiscard]] expected<void, Error> append_input_selection(
      ReadyRunState& state, ReadyRunState::InputSelectorSlot& slot, TimePoint reference_time,
      TimePoint& semantic_reference,
      std::optional<TimePoint> future_reference_lower_bound = std::nullopt) noexcept;

  // Form aligned, bracketed-window, and ordinary selections in authoring order while retaining one
  // semantic reference and one transaction set for the activation.
  [[nodiscard]] SelectionPreparation prepare_nonpredicate_input_selections(
      ReadyRunState& state, const RuntimeOperatorInstance& instance,
      const FrozenRuntimeCandidate& candidate, TimePoint& semantic_reference) noexcept;

  [[nodiscard]] expected<void, Error> prepare_predicate_input_selections(
      ReadyRunState& state, const RuntimeOperatorInstance& instance,
      const PredicateTriggerPlanRecord& predicate, const FrozenRuntimeCandidate& candidate,
      TimePoint& semantic_reference) noexcept;

  [[nodiscard]] expected<std::span<const InputSelection>, Error>
  prepare_predicate_compute_selections(PredicateComputeSelectionContext& context,
                                       TimePoint reference_time) noexcept;

  [[nodiscard]] static expected<std::span<const InputSelection>, Error>
  select_predicate_compute_inputs(void* opaque, TimePoint reference_time) noexcept;

  [[nodiscard]] bool finish_input_selection_prefix(ReadyRunState& state,
                                                   std::size_t accepted_selector_count,
                                                   bool aligned_accepted) noexcept;

  [[nodiscard]] bool finish_input_selections(ReadyRunState& state, bool accepted) noexcept;

  [[nodiscard]] bool terminal_successor_ready(const ReadyRunState& state, VertexId owner,
                                              bool include_aligned) const noexcept;

  // One physical EOS can expose several already-bounded terminal selector decisions. Preserve
  // its sole coordinator/binding credit until the last decision, but give every local successor
  // a fresh receiver Tag and global TagTime microstep after the prior Dispatcher, selector,
  // Channel activation, and activation identity have retired.
  [[nodiscard]] expected<void, Error> retire_or_rearm_terminal_source(
      ReadyRunState& state, FrozenRuntimeCandidate& frozen, VertexId owner,
      bool include_aligned) noexcept;

  // The ordinary no-completion, non-terminal retirement path deliberately keeps the pre-terminal
  // transaction shape. Completion progress and EOS successor rearming are selected by the caller
  // and never add control work to this dominant path.
  // NOLINTNEXTLINE(readability-function-cognitive-complexity)
  [[nodiscard]] bool retire_ordinary_activation_transaction(
      GraphExec& exec, ReadyRunState& state, RuntimeOperatorInstance& instance,
      const DispatcherSelection& selection, FrozenRuntimeCandidate& frozen,
      std::uint64_t activation_generation, bool channel_activation_prepared,
      ActivationParameterBinding& parameter_binding, bool callback_succeeded,
      bool completed_selection) noexcept;

  // Completion, notification, and terminal activation retirement keeps source, selector, Channel,
  // parameter, and completion progress in one ordered owner-thread transaction; any partial
  // retirement fails closed. Notification credit remains charged until Channel retirement.
  // NOLINTNEXTLINE(readability-function-cognitive-complexity)
  [[nodiscard]] bool retire_extended_activation_transaction(
      GraphExec& exec, ReadyRunState& state, RuntimeOperatorInstance& instance,
      const DispatcherSelection& selection, FrozenRuntimeCandidate& frozen,
      std::uint64_t activation_generation, bool channel_activation_prepared,
      ActivationParameterBinding& parameter_binding, bool callback_succeeded,
      bool completed_selection, std::size_t completion_index) noexcept;

  [[nodiscard]] expected<bool, Error> advance_owner_output_drain(
      ReadyRunState& state, const RuntimeOperatorInstance& instance) noexcept;

  [[nodiscard]] bool advance_completion_closures(GraphExec& exec, ReadyRunState& state) noexcept;

  [[nodiscard]] expected<void, Error> release_terminal_selector_history(ReadyRunState& state,
                                                                        VertexId owner) noexcept;

  [[nodiscard]] bool advance_input_terminal_drains(GraphExec& exec, ReadyRunState& state) noexcept;

  // Execute the exact callback transaction admitted by ClosedIslandActivationPlanRecord.
  //
  // This is deliberately narrower than dispatch_ready(): it accepts no predicate, deadline,
  // mutable Parameter, lifecycle, or Tensor state. Both the ordinary one-dispatch-per-callback
  // control and the two-callback composite call this function, so changing lane-task geometry
  // cannot silently change selection, Channel preparation, callback context, or retirement.
  void run_closed_island_activation_transaction(GraphExec& exec, ReadyRunState& state,
                                                const RuntimeDirectCompositeBinding& binding,
                                                const DispatcherSelection& selection,
                                                FrozenRuntimeCandidate& frozen,
                                                bool arm_direct_continuation,
                                                bool successor_boundary,
                                                ClosedIslandActivationResult& result) noexcept;

  // Direct-composite execution preserves activation, CUDA, and fallback state in one loop.
  void run_direct_composite_lane(GraphExec& exec, ReadyRunState& state,
                                 const RuntimeDirectCompositeBinding& binding,
                                 const DispatcherSelection& source_selection,
                                 FrozenRuntimeCandidate& source_candidate,
                                 RuntimeDirectCompositeCompletionReport& report) noexcept;

  // Keep selection, policy gating, callback invocation, and exact credit retirement adjacent so a
  // terminal path cannot skip one part of the activation transaction.
  [[nodiscard]] RunTermination dispatch_ready(GraphExec& exec, ReadyRunState& state) noexcept;

  template <bool TracksTerminalWork>
  [[nodiscard]] bool all_sources_closed(GraphExec& exec, const ReadyRunState& state) const noexcept;

  [[nodiscard]] bool wait_for_source_progress(GraphExec& exec, const ReadyRunState& state) noexcept;

  // Terminal sweep capability is immutable after cold state materialization. Compile the two
  // empty active-work checks and the extended closure proof out of a partition that owns neither
  // completion rules nor input-terminal drains.
  template <bool TracksTerminalWork>
  [[nodiscard]] RunTermination run_shared_owner(GraphExec& exec, ReadyRunState& ready) noexcept;

  struct PartitionOwnerGroup {
    std::atomic_bool failed{false};
  };

  struct PartitionOwnerContext {
    RunSessionState* session{};
    GraphExec* exec{};
    ReadyRunState* ready{};
    PartitionOwnerGroup* group{};
    RunTermination outcome{RunTermination::kCompleted};
  };

  // The owner loop serializes lifecycle, readiness, activation, and termination transitions.
  template <bool TracksTerminalWork>
  // NOLINTNEXTLINE(readability-function-cognitive-complexity)
  [[nodiscard]] RunTermination run_partition_owner(PartitionOwnerContext& context) noexcept;

  template <bool TracksTerminalWork>
  [[nodiscard]] static expected<void, Error> invoke_partition_owner(void* opaque) noexcept;

  struct PendingStartupRateCandidate {
    RuntimeRateState* state{};
    bool terminal{};
    Error failure{ErrorCode::kFailure, "RATE_GENERATION_REJECTED"};

    ~PendingStartupRateCandidate();

    void set_failure(Error error);
    void publish() noexcept;
    void cancel() noexcept;
  };

  /// Materialize every shared and partition-borrowed startup resource before lifecycle callbacks.
  [[nodiscard]] bool materialize_startup_resources(
      GraphExec& exec, std::vector<ReadyRunState>& ready_states, ReadySharedResources& shared,
      RuntimeCudaExecutionResources& cuda, RuntimeRateState& rates,
      PendingStartupRateCandidate& pending_rate_candidate);

  /// Bind notifications and lifecycle, then publish every source at the global Arm boundary.
  [[nodiscard]] RunTermination arm_startup_sources(
      GraphExec& exec, std::vector<ReadyRunState>& ready_states, ReadySharedResources& shared,
      RateResourceMaterializationState& rate_materialization,
      PendingStartupRateCandidate& pending_rate_candidate);

  // Keep startup, dispatch, and terminal cleanup in one transaction so ReadyRunState ownership and
  // shutdown ordering remain visible at the call site.
  [[nodiscard]] RunTermination worker_main_impl();

  void finish(RunTermination outcome) noexcept;

  void finish_staged_start_failure() noexcept;

  void finish_after_terminal_participant(RunTermination outcome) noexcept;

  std::shared_ptr<const ExecutionPlanState> plan_;
  RunGenerationId run_generation_{};
  ProcessGroupId process_group_{};
  std::shared_ptr<LifecycleControlState> lifecycle_control_;
  std::unique_ptr<GraphExec> graph_exec_;
  std::unique_ptr<RuntimeRateState> rate_state_;
  std::unique_ptr<ParameterUpdateCoordinator> retained_parameter_updates_;
  std::unique_ptr<RuntimeCudaExecutionResources> cuda_execution_resources_;
  std::unique_ptr<RuntimeTensorResources> tensor_resources_;
  std::shared_ptr<RuntimeChannelSession> channel_session_;
  std::shared_ptr<RuntimeTerminalParticipant> terminal_participant_;
  std::vector<OperatorCompletionProgressRecord> operator_completion_progress_;
  std::vector<OperatorCompletionPhase> operator_completion_phases_;
  std::atomic_bool stop_requested_{false};
  std::atomic<std::uint64_t> termination_state_{};
  std::atomic<RunTermination> termination_{RunTermination::kRunning};
  std::thread worker_;
  RuntimeWakeState readiness_wake_;
  std::vector<std::unique_ptr<RuntimeWakeState>> partition_wakes_;
  // Cold-built dense routing avoids waking unrelated partition owners for one direct-retained
  // edge. Index zero remains invalid because VertexId is one-based.
  RuntimePartitionWakeRouter partition_wake_by_vertex_;
  std::size_t local_partition_runner_count_{};
  RuntimePhaseCounters* phase_counters_{};
  ComputeLaneInvoke compute_lane_invoke_ = &invoke_compute_lane<false>;
  ComputeLaneInvoke compute_lane_with_completion_invoke_ =
      &invoke_compute_lane_with_completion<false, false>;
  mutable std::mutex diagnostic_mutex_;
  mutable std::mutex clock_health_mutex_;
  mutable std::mutex join_mutex_;
  std::condition_variable join_cv_;
  bool joining_{};
  bool joined_{};
  bool shared_terminal_{};
  RuntimeDiagnosticBuffer diagnostic_buffer_;
  std::vector<ExecutionLaneStartupEvidence> lane_startup_evidence_;
  mutable std::mutex parameter_mutex_;
  std::condition_variable parameter_cv_;
  ParamControlLink* parameter_control_link_{};
  ParameterUpdateCoordinator* parameter_updates_{};
  std::atomic_bool parameter_ingress_close_requested_{};
  std::atomic<ParameterUpdateCoordinator*> parameter_ingress_close_target_{};
  mutable std::mutex rate_snapshot_mutex_;
  std::condition_variable rate_snapshot_cv_;
  RuntimeRateState* rate_snapshot_state_{};
};

}  // namespace holoscan::detail
