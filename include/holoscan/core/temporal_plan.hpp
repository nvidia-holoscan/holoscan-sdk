/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <holoscan/core/activation.hpp>
#include <holoscan/core/operator_ref.hpp>
#include <holoscan/core/payload_identity.hpp>
#include <holoscan/core/port_types.hpp>
#include <holoscan/core/profile.hpp>
#include <holoscan/core/source_id.hpp>
#include <holoscan/core/temporal_contract_records.hpp>
#include <holoscan/core/temporal_contract_types.hpp>
#include <holoscan/time/clock_registry.hpp>
#include <holoscan/time/time_point.hpp>

namespace holoscan {

/// @brief Identify one selector in an immutable execution plan.
enum class InputSelectorKind : std::uint8_t {
  kExact = 0,        ///< Select the sample at the frozen reference coordinate.
  kLatestWithin,     ///< Select the latest causal sample within the admitted age.
  kDrainAll,         ///< Select every retained FIFO survivor as one batch.
  kBracketedWindow,  ///< Select one complete bounded bracketed window.
};

/// @brief Identify the accessor cardinality admitted for one selected input.
enum class InputSelectionCardinality : std::uint8_t {
  kSingleSample = 0,  ///< Scalar access is valid and batch access is rejected.
  kSampleBatch,       ///< Batch access is valid and scalar access is rejected.
};

/// @brief Explain how the compiler derived a selected-sample maximum.
enum class SelectedSampleBoundOrigin : std::uint8_t {
  kSingleSample = 0,          ///< The scalar selector admits at most one sample.
  kResolvedRetainedCapacity,  ///< Queue/cache retention produced the finite maximum.
  kSelectorMaxSamples,        ///< The authored selector maximum produced the bound.
};

/// @brief Describe the state effect of accepting one selection.
enum class BatchSelectionMode : std::uint8_t {
  kNotApplicable = 0,       ///< The selector is scalar.
  kConsumeSelected,         ///< Acceptance advances the complete FIFO selection atomically.
  kNonDestructiveSnapshot,  ///< Overlapping activations may share retained entries.
};

/// @brief Identify one compile-admitted Operator completion rule.
enum class OperatorCompletionRuleKind : std::uint8_t {
  kComputeCalls = 0,  ///< Complete after the configured number of successful compute calls.
};

/// @brief Record one immutable Operator-owned completion rule.
///
/// The record contains only plan-local identity and value semantics. It owns no
/// Operator address, Runtime progress, lifecycle generation, or scheduler state.
struct OperatorCompletionPlanRecord {
  OperatorRef owner{};  ///< Plan-bound logical Operator that owns the rule.
  OperatorCompletionRuleKind kind{
      OperatorCompletionRuleKind::kComputeCalls};  ///< Closed completion-rule discriminator.
  std::uint64_t limit{};  ///< Complete unsigned compute-call limit; no value is a sentinel.

  friend constexpr bool operator==(const OperatorCompletionPlanRecord&,
                                   const OperatorCompletionPlanRecord&) noexcept = default;
};

/// @brief Record one fully resolved input selector.
///
/// Every identity is plan-local and must be interpreted with the enclosing
/// execution plan's generation. The record owns no queue, cache, sample, or
/// authoring address. Runtime may allocate the declared finite capacity but
/// must not derive a different selector or bound.
struct InputSelectionPlanRecord {
  PortId input{};                                         ///< Dense local input identity.
  InputSelectorKind selector{InputSelectorKind::kExact};  ///< Closed selector family.
  InputSelectionCardinality cardinality{
      InputSelectionCardinality::kSingleSample};  ///< Admitted accessor shape.
  std::uint32_t max_selected_samples{};           ///< Positive finite result bound.
  SelectedSampleBoundOrigin max_selected_samples_origin{
      SelectedSampleBoundOrigin::kSingleSample};  ///< Bound derivation.
  BatchSelectionMode batch_selection_mode{
      BatchSelectionMode::kNotApplicable};  ///< Acceptance state effect.
  std::uint32_t cache_capacity{};           ///< Positive finite retained-entry capacity.
  ClockId reference_clock{};                ///< Frozen selector comparison domain.
  std::uint64_t conversion_id{};            ///< Direct conversion identity, or zero for same-clock.
  TemporalRecordMissingKind missing{TemporalRecordMissingKind::kUnspecified};
  std::int64_t max_age_ns{};        ///< Positive only for \c kLatestWithin.
  std::int64_t reuse_max_age_ns{};  ///< Positive only for \c kReuseLast.
  std::uint32_t lookbehind{};       ///< Positive only for \c kBracketedWindow.
  std::uint32_t lookahead{};        ///< Positive only for \c kBracketedWindow.
  /// Compiler-validated intentional absence; no endpoint, payload capacity, or claim is required.
  bool unwired_safe_default{};
};

/// @brief Record one normalized trigger and its static presence guarantee.
///
/// Specialized window and predicate details live in their dedicated record
/// families. This common record is nevertheless total for every admitted baseline
/// trigger and provides the scheduler with one closed, pointer-free trigger
/// inventory.
struct TemporalTriggerPlanRecord {
  OperatorRef owner{};  ///< Plan-bound owning Operator.
  TemporalRecordTriggerKind kind{TemporalRecordTriggerKind::kAnyInput};
  TemporalRecordOrigin origin{TemporalRecordOrigin::kAuthored};
  PortId trigger_input{};                          ///< Each/window/predicate driver, or invalid.
  std::vector<PortId> declared_inputs;             ///< Canonical contract input order.
  std::vector<PortId> presence_guaranteed_inputs;  ///< Exact static P(trigger) set.
  SourceId readiness_source{};     ///< Logical candidate source for readiness or clock forms.
  ClockId trigger_clock{};         ///< Schedule/wait clock, or invalid.
  ClockId reference_clock{};       ///< Frozen comparison domain, or invalid.
  std::uint64_t conversion_id{};   ///< Start-delay source to reference-clock conversion; zero when
                                   ///< no start-delay conversion is required.
  std::int64_t period_ns{};        ///< Positive only for clock forms.
  std::optional<TimePoint> phase;  ///< Explicit phase when authored.
  std::uint32_t bounded_state_capacity{};          ///< Positive retained witness/state bound.
  std::string witness_identity;                    ///< Stable schema-qualified witness token.
  std::string replay_identity;                     ///< Stable schema-qualified replay token.
  std::optional<std::int64_t> max_start_delay_ns;  ///< Positive latest-start budget when present.
  std::optional<OnViolation> violation;  ///< Present exactly with \ref max_start_delay_ns.
};

/// @brief Identify one finite fact that can close an incomplete aligned tuple.
enum class AlignedClosureProofKind : std::uint8_t {
  kStreamAdvancement = 0,  ///< Admitted advancement proves the absent head cannot arrive.
  kLossWitness,            ///< A bounded loss record closes the affected source interval.
  kPrivateWatermark,       ///< Private admitted watermark evidence closes the interval.
  kEndOfStream,            ///< End-of-stream proves no later member head can arrive.
  kProducerEpochClosed,    ///< The selected producer epoch has closed.
  kTemporalClosure,        ///< Another accepted temporal closure record supplies proof.
};

/// @brief Preserve one ordered member of an oldest-head aligned join.
struct AlignedMemberPlanRecord {
  PortId input{};                    ///< Dense member input identity.
  std::uint32_t member_ordinal{};    ///< Stable zero-based presentation order.
  ClockId event_clock{};             ///< Genuine capture-time domain before conversion.
  std::uint64_t conversion_id{};     ///< Direct conversion to the comparison clock, or zero.
  std::uint32_t queue_capacity{};    ///< Positive admitted FIFO queue capacity.
  std::uint32_t cache_capacity{};    ///< Positive retained-entry capacity.
  bool capture_time_required{true};  ///< Canonical absent time is never eligible.
  bool nondecreasing_normalized_time{true};  ///< Producer-order admission requirement.
};

/// @brief Preserve the complete deterministic policy for one \c OnAligned trigger.
///
/// Runtime inspects only the oldest eligible head of every member, converts it
/// directly to \ref comparison_clock, and accepts the exact heads only when the
/// checked inclusive spread is within \ref inclusive_window_ns. The record owns
/// no queue, sample, watermark object, or backend handle.
struct AlignedTriggerPlanRecord {
  OperatorRef owner{};                           ///< Plan-bound owning Operator.
  std::vector<AlignedMemberPlanRecord> members;  ///< Stable member-port order.
  std::int64_t inclusive_window_ns{};            ///< Nonnegative inclusive spread bound.
  ClockId comparison_clock{};                    ///< Frozen normalized comparison domain.
  std::string algorithm{"holoscan.temporal.aligned-oldest-head/v1"};
  bool fifo_heads_only{true};             ///< Later queue entries are never scanned.
  bool no_scan_ahead{true};               ///< Matching cannot skip an eligible head.
  bool checked_direct_conversion{true};   ///< Every cross-domain member converts directly.
  bool checked_spread_subtraction{true};  ///< Overflow is formation failure, not a miss.
  bool exact_window_matches{true};        ///< A spread equal to the window is accepted.
  bool reference_is_maximum_head{true};   ///< Complete tuples freeze the normalized high head.
  bool accepted_once_consumption{true};   ///< Every accepted head advances exactly once.
  bool retire_all_minimum_heads{true};    ///< A miss retires every head at the minimum time.
  std::string retirement_order{"port_id_then_source_sequence"};
  std::string miss_witness_identity{"holoscan.temporal.alignment-window-miss.v1"};
  std::string formation_failure_identity{"holoscan.temporal.alignment-formation-failure.v1"};
  std::vector<AlignedClosureProofKind> closure_proofs;  ///< Finite accepted proof vocabulary.
  bool silence_is_not_closure{true};         ///< Silence alone cannot complete missing policy.
  bool missing_requires_finite_proof{true};  ///< Missing aggregation follows proof closure.
  bool public_watermark_protocol{false};     ///< baseline exposes no public watermark/reorder API.
};

/// @brief Identify how one fixed clock phase is established.
enum class ClockPhaseOrigin : std::uint8_t {
  kCapturedAtSuccessfulStart = 0,  ///< Capture phase after the Operator crosses \c kStart.
  kExplicitAbsolute,               ///< Preserve the authored absolute phase.
};

/// @brief Identify the terminal response to an unexpected clock epoch change.
enum class ClockEpochDiscontinuityDisposition : std::uint8_t {
  kTerminalRestartRequired = 0,  ///< Stop candidates until an accepted lifecycle restart.
};

/// @brief Preserve one fixed-phase, constant-work \c OnClock schedule.
struct ClockTriggerPlanRecord {
  OperatorRef owner{};       ///< Plan-bound owning Operator and schedule generation.
  ClockId clock{};           ///< Frozen readable schedule/reference clock.
  std::int64_t period_ns{};  ///< Positive representable period.
  ClockPhaseOrigin phase_origin{ClockPhaseOrigin::kCapturedAtSuccessfulStart};
  std::optional<TimePoint> explicit_phase;  ///< Absolute phase exactly when explicitly authored.
  std::uint64_t
      phase_conversion_id{};  ///< Direct conversion to \ref clock, or zero for same-clock.
  ClockTickPolicy tick_policy{ClockTickPolicy::kCoalesceLateTicks};  ///< Closed baseline policy.
  std::uint64_t initial_tick_index{};                  ///< First schedule index, always zero.
  std::optional<std::uint64_t> initial_handled_index;  ///< Empty before the first frozen tick.
  std::uint32_t coalesced_range_capacity{1U};     ///< One fixed first/last/count evidence record.
  bool global_start_barrier{true};                ///< Compute waits for every successful start.
  bool omitted_phase_index_zero_immediate{true};  ///< Captured index zero is immediately eligible.
  bool exact_now_eligible{true};                  ///< \c tick(n) == now is due.
  bool greatest_due_constant_time{true};          ///< Selection performs no per-period loop.
  bool one_candidate_per_observation{true};       ///< Late work creates at most one candidate.
  bool skipped_indices_have_no_candidate{
      true};                               ///< Skipped ticks create no candidate or start gate.
  bool scheduled_tick_is_reference{true};  ///< Delay never changes the source coordinate.
  bool checked_terminal_no_wrap{true};     ///< Arithmetic/index exhaustion is terminal.
  bool omitted_phase_recaptured_on_restart{true};  ///< Successful restart captures a new phase.
  bool explicit_phase_preserved_on_restart{true};  ///< Absolute phase survives restart.
  bool failed_rollback_restores_watermark{true};   ///< Failed restart preserves old schedule state.
  bool stop_blocks_future_freeze{true};            ///< A stop fence prevents new candidates.
  ClockEpochDiscontinuityDisposition epoch_discontinuity{
      ClockEpochDiscontinuityDisposition::kTerminalRestartRequired};
  std::string coalesced_range_identity{"holoscan.temporal.clock-coalesced-range.v1"};
  std::string arithmetic_failure_identity{"holoscan.temporal.clock-arithmetic-terminal.v1"};
  std::string stale_anchor_identity{"holoscan.temporal.clock-stale-anchor.v1"};
};

/// @brief Identify the frozen source coordinate used by a maximum compute-start delay.
enum class MaxStartDelaySourceCoordinateKind : std::uint8_t {
  kTriggerSample = 0,          ///< Accepted AnyInput/Each sample capture time.
  kPredicateDriver,            ///< Accepted input-driven predicate sample.
  kScheduledTick,              ///< Selected OnClock or predicate-clock tick.
  kAlignedHighOrClosureProof,  ///< Complete tuple high head or comparable closure proof.
  kWindowClose,                ///< Terminal bounded-window close decision.
  kReadinessWitness,           ///< Accepted normalized readiness witness.
};

/// @brief Preserve the final invocation-gate contract for one Operator's maximum start delay.
struct MaxStartDelayPlanRecord {
  OperatorRef owner{};                            ///< Plan-bound owning Operator.
  std::int64_t budget_ns{};                       ///< Positive finite latest-start budget.
  OnViolation action{OnViolation::kEmitWitness};  ///< Explicit late-start action.
  MaxStartDelaySourceCoordinateKind source_coordinate{
      MaxStartDelaySourceCoordinateKind::kTriggerSample};  ///< Frozen source-decision coordinate.
  ClockId source_clock{};                                  ///< Plan-local comparison domain.
  std::uint64_t conversion_id{};                           ///< Direct source conversion, or zero.
  std::uint32_t callback_count_if_late{1U};  ///< One except for \c kFailRun, which is zero.
  bool checked_addition{true};               ///< Overflow fails formation instead of wrapping.
  bool exact_boundary_is_on_time{true};      ///< Dispatch exactly at the latest start may enter.
  bool final_invocation_gate{true};          ///< Selection/predicate/missing precede this gate.
  bool consumes_candidate_tag{true};         ///< Suppression and failure retain the reserved Tag.
  bool automatic_output{false};              ///< Runtime never manufactures fallback output.
  bool callback_state_rollback{false};       ///< Latest-start policy never rolls back a callback.
  bool duration_measurement_separate{true};  ///< Callback duration is a distinct observation.
  std::string violation_witness_identity{"holoscan.temporal.max-start-delay-violation.v1"};
  std::string source_coordinate_identity{"holoscan.temporal.max-start-delay-source-coordinate.v1"};
};

/// @brief Identify the response to a fatal post-start readable-source error.
enum class ReadableSourceErrorDisposition : std::uint8_t {
  kFailRun = 0,  ///< Close the generation and coordinate ordinary run teardown.
};

/// @brief Preserve the bounded level-readable activation contract.
struct ReadableTriggerPlanRecord {
  OperatorRef owner{};                           ///< Plan-bound owning Operator.
  SourceId source{};                             ///< Stable logical readable-source identity.
  PlanGenerationId plan_generation{};            ///< Provenance for source and clock identities.
  ClockId witness_clock{};                       ///< Normalized accepted-witness coordinate domain.
  std::uint64_t initial_source_generation{1U};   ///< First nonzero source generation.
  std::uint64_t initial_source_sequence{};       ///< Zero before the first witness.
  std::uint32_t effective_capacity{1U};          ///< Exactly one outstanding witness/candidate.
  std::uint32_t max_outstanding_candidates{1U};  ///< Level readiness never creates a queue.
  bool level_readable{true};                     ///< Rearm observes current level state.
  bool bind_at_arm{true};                        ///< Concrete adapter binds during \c kArm.
  bool enable_after_global_start{true};          ///< Acceptance begins only after start barrier.
  bool disable_before_stop{true};                ///< Stop fencing precedes backend disablement.
  bool unbind_at_release{true};                  ///< Concrete adapter releases during \c kRelease.
  bool restart_advances_generation{true};        ///< Successful restart creates a fresh generation.
  bool stale_generation_rejected{true};          ///< Old callbacks cannot create candidates.
  bool rearm_only_if_still_readable{true};       ///< Retirement rechecks the same enabled source.
  bool error_creates_candidate{false};           ///< Fatal source errors own no Tag/candidate.
  bool automatic_retry{false};  ///< baseline never retries or rebinds automatically.
  ReadableSourceErrorDisposition fatal_error{ReadableSourceErrorDisposition::kFailRun};
  std::string readiness_witness_identity{"holoscan.readiness.readable-witness.v1"};
  std::string source_error_identity{"holoscan.readiness.readable-source-error.v1"};
  std::string stale_generation_identity{"holoscan.readiness.stale-source-generation.v1"};
};

/// @brief Identify one complete phase-local missing-input aggregation.
enum class MissingSelectionPhase : std::uint8_t {
  kActivation = 0,       ///< Non-predicate trigger input selection.
  kPredicateEvaluation,  ///< Inputs needed before invoking the predicate evaluator.
  kPredicateCompute,     ///< Compute-only inputs selected after a firing decision.
};

/// @brief Preserve one ordered input participating in a missing aggregate.
struct MissingAggregateInputPlanRecord {
  PortId input{};                                         ///< Dense plan-local input identity.
  std::uint32_t ordinal{};                                ///< Stable order within the phase.
  InputSelectorKind selector{InputSelectorKind::kExact};  ///< Frozen selector family.
  TemporalRecordMissingKind missing{TemporalRecordMissingKind::kUnspecified};
};

/// @brief Preserve SafeDefault-dominant aggregation and selection commit semantics.
struct MissingAggregatePlanRecord {
  OperatorRef owner{};                                              ///< Plan-bound owning Operator.
  MissingSelectionPhase phase{MissingSelectionPhase::kActivation};  ///< Reached phase.
  std::vector<MissingAggregateInputPlanRecord> inputs;  ///< Complete canonical phase set.
  bool evaluate_complete_set{true};                     ///< Never short-circuit per-input results.
  bool safe_default_dominates{true};                ///< Any SafeDefault accepts one degraded call.
  bool absent_without_safe_default_no_fires{true};  ///< Skip/exhausted reuse invokes no callback.
  std::uint32_t degraded_callback_count{1U};        ///< Exactly one degraded callback.
  std::uint32_t absent_callback_count{};            ///< Exactly zero callbacks for no-fire.
  std::uint32_t nominal_callback_count{1U};         ///< Exactly one nominal callback.
  bool degraded_commits_destructive_selection{true};  ///< Accepted degraded work commits once.
  bool no_fire_preserves_provisional_selection{
      true};                                   ///< Pure no-fire commits no DrainAll advance.
  bool bracketed_window_all_or_nothing{true};  ///< A failed window exposes no partial batch.
  std::string input_state_identity{"holoscan.input-state.v1"};
  std::string selection_reason_identity{"holoscan.selection-reason.v1"};
  std::string aggregate_identity{"holoscan.temporal.missing-aggregate.v1"};
};

/// @brief Record the admitted bounded state machine for one \c OnWindow trigger.
struct WindowTriggerPlanRecord {
  OperatorRef owner{};                  ///< Plan-bound owning Operator.
  PortId anchor{};                      ///< Exact anchor input.
  ClockId wait_clock{};                 ///< Independently progressing wait domain.
  std::string capability_evidence_id;   ///< Stable proof for read/wait/progress/deadline.
  bool external_driver_used{};          ///< True only when an admitted external driver closes time.
  std::int64_t max_wait_ns{};           ///< Positive finite close wait.
  std::uint32_t max_pending_windows{};  ///< Positive finite pending-window capacity.
  std::vector<PortId> bracketed_inputs;            ///< Inputs requiring a terminal close outcome.
  std::vector<PortId> presence_guaranteed_inputs;  ///< Exact static set, always the anchor.
  bool first_anchor_prime_only{true};              ///< The first anchor creates no window.
  bool advance_on_every_anchor{true};              ///< Every accepted anchor advances boundaries.
  bool reset_on_producer_epoch{true};              ///< Producer epoch change re-primes state.
  bool reset_on_timestamp_regression{true};        ///< Decreasing anchor time re-primes state.
  std::string primed_witness_identity;             ///< Stable first-anchor priming witness token.
  std::string open_witness_identity;           ///< Stable pending-window creation witness token.
  std::string close_witness_identity;          ///< Stable normal-close witness token.
  std::string timeout_witness_identity;        ///< Stable timeout witness token.
  std::string pending_limit_witness_identity;  ///< Stable bounded-capacity terminal token.
  std::string end_of_stream_witness_identity;  ///< Stable end-of-stream terminal token.
  std::string timestamp_regression_witness_identity;  ///< Stable regression/reset token.
  std::string cancelled_witness_identity;      ///< Stable lifecycle/session cancellation token.
  std::string producer_epoch_identity_schema;  ///< Schema for reset/re-prime epoch evidence.
};

/// @brief Record one complete retained-history contract for a bracketed input.
struct BracketedWindowPlanRecord {
  OperatorRef owner{};                             ///< Plan-bound owning Operator.
  PortId input{};                                  ///< Bracketed payload input.
  PortId anchor{};                                 ///< Consecutive-boundary anchor input.
  std::uint32_t lookbehind{};                      ///< Strict predecessor count.
  std::uint32_t lookahead{};                       ///< Strict successor count.
  std::uint32_t max_samples{};                     ///< Positive all-or-nothing selection bound.
  ClockId event_clock{};                           ///< Normalized event-time comparison domain.
  std::uint64_t conversion_id{};                   ///< Candidate-to-anchor conversion, or zero.
  std::uint32_t queue_capacity{};                  ///< Accepted complete-history queue capacity.
  std::uint32_t cache_capacity{};                  ///< Accepted retained-entry cache capacity.
  std::uint32_t producer_max_emits_per_compute{};  ///< Accepted finite producer burst bound.
  std::uint32_t retained_overlap_capacity{};       ///< Entries retained across adjacent windows.
  PayloadIdentityRecord payload_identity{};  ///< Complete pointer-free payload/schema identity.
  TemporalRecordMissingKind missing{TemporalRecordMissingKind::kUnspecified};
  bool strict_outer_samples{true};  ///< Outer predecessor/successor are strict.
  bool inclusive_interior{true};    ///< Interior anchor interval is closed.
  bool retain_overlap{true};        ///< Completed overlap remains retained.
  std::string ordering{"normalized_capture_time_then_source_sequence"};
  std::string close_condition{"qualifying_successor_or_terminal_witness"};
  std::string semantic_reference{"current_anchor_capture_time"};
  std::string activation_time_policy{"window_terminal_acceptance_time"};
  std::string lease_lifetime_evidence{
      "holoscan.sample-lease.activation-retained.v1"};  ///< Required lease-retention contract.
  std::string insufficient_history_reason{"window_insufficient_history"};
  std::string insufficient_history_witness_identity;  ///< Stable incomplete-history token.
  std::string timeout_reason{"window_timeout"};
  std::string timeout_witness_identity;  ///< Stable per-input timeout terminal token.
  std::string sample_limit_reason{"window_sample_limit"};
  std::string sample_limit_witness_identity;  ///< Stable all-or-nothing overflow token.
  std::string producer_epoch_identity_schema{"holoscan.producer-epoch.v1"};
  std::string source_sequence_identity_schema{"holoscan.source-sequence.v1"};
};

/// @brief Identify the event that evaluates one predicate.
enum class PredicateEvaluationSourceKind : std::uint8_t {
  kInputUpdate = 0,  ///< Evaluate for every accepted exact driver sample.
  kClock,            ///< Evaluate for every selected fixed-phase tick.
};

/// @brief Identify the portability scope of a predicate identity.
enum class PredicateIdentityScope : std::uint8_t {
  kOperatorScoped = 0,  ///< Identity is qualified by its owning Operator.
  kIndependentlyKeyed,  ///< Identity and semantic version are independently stable.
};

/// @brief Identify executable replay authority retained by a predicate plan.
enum class PredicateReplayCapability : std::uint8_t {
  kDecisionReplay = 0,          ///< Replay consumes the recorded decision.
  kLocalGraphExecReevaluation,  ///< The same retained local executable may reevaluate.
  kCrossBuildRecomputation,     ///< Matching qualified build evidence permits recomputation.
};

/// @brief Report how callable evidence affected profile admission.
enum class PredicateCallableDisposition : std::uint8_t {
  kAdmitted = 0,         ///< Callable satisfies the effective profile.
  kAdmittedWithWarning,  ///< Callable is allowed but retains a stable warning.
  kRejected,             ///< Callable lacks required capability evidence.
};

/// @brief Identify the exact result carrier validated for a predicate evaluator.
enum class PredicateResultKind : std::uint8_t {
  kBool = 0,           ///< The evaluator returns exactly \c bool.
  kExpectedBoolError,  ///< The evaluator returns \c expected<bool, Error>.
};

/// @brief Preserve pointer-free callable classification and capability evidence.
struct PredicateCallablePlanRecord {
  std::string predicate_id;  ///< Owning plan-local predicate identity.
  PredicateCallableKind kind{PredicateCallableKind::kStaticFunction};
  std::string implementation_fingerprint;    ///< Stable type/build identity; never an address.
  std::string build_evidence;                ///< Immutable package/build evidence identity.
  std::string toolchain_evidence;            ///< Immutable compiler/toolchain evidence identity.
  std::optional<std::string> predicate_key;  ///< Independent qualified key when authored.
  std::optional<SemanticVersion> predicate_key_version;  ///< Version paired with the key.
  std::uint64_t size_bytes{};                            ///< Exact finite owned descriptor size.
  std::uint64_t alignment_bytes{};                       ///< Exact descriptor alignment.
  bool no_dynamic_allocation{};                          ///< Proven no-allocation invocation.
  bool bounded_call{};                                   ///< Proven finite-duration invocation.
  bool is_noexcept{};                                    ///< Proven nonthrowing invocation.
  bool owned_lifetime{};  ///< Receiver/capture lifetime is session-owned.
  bool self_rebind{};     ///< Bound self descriptor rebinds per session.
  PredicateResultKind result_kind{PredicateResultKind::kBool};  ///< Exact validated result.
  std::string argument_contract;   ///< Exact canonical argument-order schema identity.
  std::string dispatch_model;      ///< Direct, inline-owned, rebound-member, or erased dispatch.
  std::string binding_provenance;  ///< Stable receiver/value ownership provenance.
  std::string capture_provenance;  ///< Stable capture classification; never capture bytes.
  std::string receiver_role;       ///< Empty except for the admitted \c self role.
  bool type_erased{};              ///< Whether invocation uses owned type erasure.
  bool possible_dynamic_allocation{};  ///< Whether static evidence cannot rule allocation out.
  bool requires_python_gil{};          ///< Reserved Python adapter GIL requirement.
  std::vector<std::string> criticality_constraints;  ///< Effective region contributors.
  std::vector<std::string> diagnostic_ids;           ///< Stable admission warning/error identities.
  std::vector<std::string> capability_evidence_references;  ///< Qualified immutable evidence.
  std::uint64_t evidence_validity_epoch{};        ///< Plan/build epoch that validated evidence.
  std::uint64_t evidence_revocation_epoch{};      ///< Zero when no revocation is recorded.
  Profile effective_profile{Profile::kStandard};  ///< Resolved admission profile.
  PredicateCallableDisposition disposition{PredicateCallableDisposition::kRejected};
};

/// @brief Record one ordered scalar input to a predicate evaluator.
struct PredicateInputPlanRecord {
  std::string predicate_id;                               ///< Owning plan-local predicate identity.
  PortId input{};                                         ///< Dense plan-local input identity.
  std::uint32_t argument_ordinal{};                       ///< Zero-based evaluator input order.
  bool implicit_driver{};                                 ///< True only for an input-update driver.
  InputSelectorKind selector{InputSelectorKind::kExact};  ///< Scalar selector.
  TemporalRecordMissingKind missing{TemporalRecordMissingKind::kUnspecified};
  ClockId reference_clock{};                 ///< Frozen comparison domain.
  std::uint64_t conversion_id{};             ///< Direct conversion identity, or zero.
  std::uint32_t cache_capacity{};            ///< Positive finite retained-entry capacity.
  PayloadIdentityRecord payload_identity{};  ///< Complete pointer-free payload/schema identity.
  bool capture_time_required{true};  ///< Event-time selection requires genuine capture time.
  std::string selection_state_schema{"holoscan.input-state.v1"};
  std::string selection_reason_schema{"holoscan.selection-reason.v1"};
  std::int64_t max_age_ns{};        ///< Latest/freshness bound when applicable.
  std::int64_t reuse_max_age_ns{};  ///< Reuse bound when applicable.
};

/// @brief Preserve the complete normalized semantics of one predicate trigger.
// Field order follows the semantic record contract and canonical serializer.
// NOLINTNEXTLINE(clang-analyzer-optin.performance.Padding)
struct PredicateTriggerPlanRecord {
  std::string id;                       ///< Stable plan-local predicate identity.
  OperatorRef owner{};                  ///< Plan-bound owning Operator.
  PlanGenerationId graph_generation{};  ///< Frozen generation qualifying every local identity.
  PredicateIdentityScope identity_scope{PredicateIdentityScope::kOperatorScoped};
  std::string identity;                             ///< Authored local or qualified semantic name.
  std::optional<SemanticVersion> semantic_version;  ///< Required for independent identity.
  PredicateEvaluationSourceKind source{PredicateEvaluationSourceKind::kInputUpdate};
  PortId driver{};                 ///< Exact driver, or invalid for clock evaluation.
  ClockId clock{};                 ///< Schedule clock, or invalid for input evaluation.
  std::int64_t period_ns{};        ///< Positive only for clock evaluation.
  std::optional<TimePoint> phase;  ///< Explicit phase when authored.
  PredicateFiring firing{PredicateFiring::kOnFalseToTrue};
  bool initial_truth{};  ///< Always false for a fresh admitted state.
  bool successful_evaluation_commits_truth{true};
  bool missing_or_error_preserves_truth{true};
  bool candidate_state_is_transactional{
      true};  ///< Failed evaluation never mutates committed state.
  bool compute_adopts_evaluator_snapshot{true};  ///< Compute receives the exact evaluator snapshot.
  bool consumes_reserved_candidate_tag{true};    ///< Every source event consumes its reserved tag.
  std::uint64_t state_size_bytes{};              ///< Exact finite state size, or zero.
  std::uint64_t state_alignment_bytes{};         ///< Exact state alignment, or zero.
  std::string state_schema;       ///< Stable state schema, empty exactly when absent.
  std::string state_type_digest;  ///< Canonical state type identity, never object bytes.
  std::string state_initialization{"value_initialized_false_truth"};
  PlanGenerationId state_generation{};            ///< Generation that owns state/truth.
  bool has_config{};                              ///< Whether real config was authored.
  std::uint64_t config_size_bytes{};              ///< Exact finite config size, or zero.
  std::uint64_t config_alignment_bytes{};         ///< Exact config alignment, or zero.
  std::optional<std::string> config_schema;       ///< Absent for configless predicates.
  std::optional<std::string> config_digest;       ///< Absent for configless predicates.
  std::optional<std::string> config_type_digest;  ///< Canonical config type identity when present.
  TemporalPredicateResetPolicy reset_policy{TemporalPredicateResetPolicy::kResetEventsV1};
  std::vector<std::string> reset_reasons;  ///< Exact reset causes; pause/resume is absent.
  PredicateReplayCapability replay{PredicateReplayCapability::kDecisionReplay};
  std::string replay_evidence;                 ///< Required only for cross-build recomputation.
  std::string evaluation_witness_identity;     ///< Stable evaluation witness token.
  std::string transition_witness_identity;     ///< Stable firing-transition witness token.
  std::string error_witness_identity;          ///< Stable evaluator-error witness token.
  std::string reset_witness_identity;          ///< Stable state/truth reset witness token.
  std::string suppression_witness_identity;    ///< Stable false/missing suppression token.
  std::string safe_default_witness_identity;   ///< Stable degraded-path witness token.
  std::string reaction_fire_witness_identity;  ///< Stable compute-reaction witness token.
  std::string missing_precedence{"safe_default_then_skip_or_exhausted_reuse"};
  std::string evaluator_error_policy{
      "discard_candidate_preserve_truth_follow_runtime_failure_policy"};
};

}  // namespace holoscan
