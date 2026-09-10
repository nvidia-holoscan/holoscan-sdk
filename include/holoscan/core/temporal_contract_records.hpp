/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include <holoscan/holoscan_core_export.h>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/source_id.hpp>
#include <holoscan/core/temporal_contract.hpp>

namespace holoscan {

/// @brief Name the schema emitted by the baseline temporal-record serializer.
inline constexpr const char* kTemporalRecordSchema = "holoscan.temporal-records";

/// @brief Name the only accepted baseline temporal-record schema version.
inline constexpr std::uint32_t kTemporalRecordVersion = 1;

/// @brief Identify one normalized closed trigger without retaining an authoring object.
enum class TemporalRecordTriggerKind : std::uint8_t {
  kAnyInput = 0,  ///< Trigger from one accepted input and pin its exact sample identity.
  kEach,          ///< Trigger for every accepted sample on one input.
  kClock,         ///< Trigger from one fixed-phase periodic clock schedule.
  kReadable,      ///< Trigger from one logical level-readable source.
  kNotified,      ///< Trigger from one logical counted-notification source.
  kDoorbell,      ///< Trigger from one logical monotonic doorbell source.
  kSpin,          ///< Trigger from one logical false-to-true spin source.
  kAligned,       ///< Trigger from one deterministic oldest-head alignment.
  kWindow,        ///< Trigger after one bounded bracketed window closes.
  kPredicate,     ///< Trigger after one admitted predicate firing decision.
};

/// @brief Explain whether a normalized contract was authored or materialized by Core.
enum class TemporalRecordOrigin : std::uint8_t {
  kAuthored = 0,     ///< The Operator explicitly authored the trigger.
  kCoreSynthesized,  ///< Core materialized the inputful empty-contract default.
};

/// @brief Identify one normalized input selector.
enum class TemporalRecordSelectorKind : std::uint8_t {
  kExact = 0,        ///< Select one exact scalar sample.
  kLatestWithin,     ///< Select one latest causal scalar sample within a finite age.
  kDrainAll,         ///< Select every retained FIFO survivor as one batch.
  kBracketedWindow,  ///< Select one all-or-nothing bracketed batch.
};

/// @brief Identify the immutable accessor shape of a normalized selector.
enum class TemporalRecordCardinality : std::uint8_t {
  kScalar = 0,  ///< The selector produces at most one sample.
  kBatch,       ///< The selector produces one bounded borrowed batch.
};

/// @brief Identify how an accepted selector affects retained cache entries.
enum class TemporalRecordBatchMode : std::uint8_t {
  kScalar = 0,              ///< Scalar selection; no batch transaction exists.
  kConsumeOnAcceptance,     ///< Commit the complete frozen batch exactly once on acceptance.
  kNondestructiveSnapshot,  ///< Retain the complete batch for overlapping windows.
};

/// @brief Identify one normalized missing-input disposition.
enum class TemporalRecordMissingKind : std::uint8_t {
  kUnspecified = 0,  ///< The trigger or selector shape proves no explicit policy is needed.
  kSkip,             ///< Suppress the candidate when selection completes without a value.
  kSafeDefault,      ///< Enter the fieldless Operator-authored degraded branch.
  kReuseLast,        ///< Reuse one prior scalar selection within a finite age.
};

/// @brief Identify the coordinate used as a selector's frozen reference.
enum class TemporalRecordReferenceKind : std::uint8_t {
  kTriggerSample = 0,  ///< Use the exact readiness-causing input sample coordinate.
  kScheduledTick,      ///< Use the selected fixed-phase clock tick.
  kAlignedHigh,        ///< Use the greatest normalized member-head coordinate.
  kWindowClose,        ///< Use the bounded window close-decision coordinate.
  kReadinessWitness,   ///< Use the accepted logical readiness witness coordinate.
  kPredicateDriver,    ///< Use the input-driven predicate sample coordinate.
  kPredicateTick,      ///< Use the clock-driven predicate tick coordinate.
};

/// @brief Identify the clock relation required before a reference may be compared.
enum class TemporalRecordConversionKind : std::uint8_t {
  kSameClock = 0,  ///< Candidate and reference already share one clock domain.
  kDirectAffine,   ///< Compilation must resolve one declared direct affine conversion.
};

/// @brief Explain how one finite normalized capacity was derived.
enum class TemporalRecordBoundOrigin : std::uint8_t {
  kSingleSample = 0,   ///< The scalar selector requires exactly one retained slot.
  kResolvedRetention,  ///< Graph storage and temporal retention produced the bound.
  kSelectorMaximum,    ///< An explicit selector maximum produced the bound.
};

/// @brief Identify the normalized source shape of one predicate evaluation.
enum class TemporalPredicateSourceKind : std::uint8_t {
  kInputUpdate = 0,  ///< Evaluate for each accepted exact driver sample.
  kClock,            ///< Evaluate for each admitted fixed-phase clock tick.
};

/// @brief Identify whether a predicate identity is local or independently versioned.
enum class TemporalPredicateScope : std::uint8_t {
  kOperatorLocal = 0,  ///< Identity is expanded within the owning Operator scope.
  kQualified,          ///< Identity is globally qualified and has a semantic version.
};

/// @brief Identify the fixed predicate state/truth reset contract.
enum class TemporalPredicateResetPolicy : std::uint8_t {
  /// Reset for a new plan, cold session, restart/rebind, producer epoch, or
  /// relevant clock discontinuity. Pause/resume is explicitly not a reset.
  kResetEventsV1 = 0,
};

/// @brief Identify whether a predicate record can be recomputed from durable evidence.
enum class TemporalPredicateReplayKind : std::uint8_t {
  kDecisionReplay = 0,          ///< Replay consumes the recorded decision without execution.
  kLocalGraphExecReevaluation,  ///< The retained local GraphExec may reevaluate its callable.
  kCrossBuildRecomputation,     ///< Qualified matching evidence permits cross-build execution.
};

/// @brief Carry one copied clock key in a pointer-free normalized record.
///
/// HSDK5-005 records the authoring/default identity. HSDK5-009 resolves that key to a
/// plan-local clock and direct-conversion record without changing this schema.
struct TemporalClockKeyRecord {
  bool is_default{true};  ///< Whether this denotes the graph-default clock.
  std::string name;       ///< Copied named key; empty exactly when \ref is_default is true.

  friend bool operator==(const TemporalClockKeyRecord&, const TemporalClockKeyRecord&) = default;
};

/// @brief Carry stable authoring and canonical-order provenance for one normalized record.
struct TemporalRecordProvenance {
  std::string source;                 ///< Stable bounded source or derivation identity.
  std::uint32_t declaration_order{};  ///< Canonical declaration order within the Operator.

  friend bool operator==(const TemporalRecordProvenance&,
                         const TemporalRecordProvenance&) = default;
};

/// @brief Identify how an input-readiness trigger pins its accepted sample.
enum class TemporalTriggerPinMode : std::uint8_t {
  /// Pin the accepted arrival identity without manufacturing an Exact selector or requiring
  /// capture time.
  kAcceptedArrivalIdentity = 0,
};

/// @brief Preserve one input-readiness pin independently of selector records.
struct TemporalTriggerPinRecord {
  std::string input_name;  ///< Canonical local input name in declaration order.
  TemporalTriggerPinMode mode{TemporalTriggerPinMode::kAcceptedArrivalIdentity};
  /// Positive finite simultaneous retention bound from witness acceptance through activation
  /// retirement; this is neither selected-sample count nor queue depth.
  std::uint32_t retained_pin_capacity{};
  TemporalRecordBoundOrigin capacity_origin{TemporalRecordBoundOrigin::kResolvedRetention};
  TemporalRecordProvenance provenance{};  ///< Canonical input declaration order and source.

  friend bool operator==(const TemporalTriggerPinRecord&,
                         const TemporalTriggerPinRecord&) = default;
};

/// @brief Describe one selector reference coordinate and required clock relation.
struct TemporalReferenceRecord {
  TemporalRecordReferenceKind kind{TemporalRecordReferenceKind::kTriggerSample};
  TemporalClockKeyRecord clock{};  ///< Frozen reference clock key.
  TemporalRecordConversionKind conversion{TemporalRecordConversionKind::kSameClock};
  /// Stable plan-local conversion-record identity resolved by HSDK5-009, or empty for same-clock.
  std::string conversion_record_id;

  friend bool operator==(const TemporalReferenceRecord&, const TemporalReferenceRecord&) = default;
};

/// @brief Preserve the normalized oldest-head alignment contract.
struct TemporalAlignedRecord {
  std::vector<std::string> member_inputs;  ///< Stable declaration-order input names.
  std::int64_t inclusive_window_ns{};      ///< Nonnegative inclusive maximum head spread.
  TemporalReferenceRecord reference{
      .kind = TemporalRecordReferenceKind::kAlignedHigh};  ///< Comparison coordinate.
  std::string algorithm{"holoscan.temporal.aligned-oldest-head/v1"};
  bool no_scan_ahead{true};          ///< Whether only the oldest eligible head is inspected.
  bool consume_accepted_once{true};  ///< Whether one accepted tuple consumes each head once.

  friend bool operator==(const TemporalAlignedRecord&, const TemporalAlignedRecord&) = default;
};

/// @brief Preserve the complete normalized window-trigger state-machine contract.
struct TemporalWindowRecord {
  std::string anchor_input;             ///< Exact anchor input copied from authoring.
  TemporalClockKeyRecord wait_clock{};  ///< Independently progressing finite-wait clock.
  std::int64_t max_wait_ns{};           ///< Positive finite close wait.
  std::uint32_t max_pending_windows{};  ///< Positive finite pending-window capacity.
  TemporalReferenceRecord reference{
      .kind = TemporalRecordReferenceKind::kWindowClose};  ///< Close comparison coordinate.
  bool previous_current_anchor_roles{true};  ///< Whether boundaries are consecutive anchors.
  bool strict_outer_samples{true};           ///< Whether predecessor/successor are strict.
  bool inclusive_interior{true};             ///< Whether the anchor interval is closed.
  bool close_on_timeout{true};               ///< Whether the admitted wait closes a window.
  bool close_on_end_of_stream{true};         ///< Whether EOS proof may close a window.
  bool retain_overlap{true};                 ///< Whether completed overlap remains retained.
  bool reset_on_epoch_change{true};          ///< Whether a new source epoch re-primes state.
  bool reset_on_timestamp_regression{true};  ///< Whether decreasing time re-primes state.

  friend bool operator==(const TemporalWindowRecord&, const TemporalWindowRecord&) = default;
};

/// @brief Preserve one ordered predicate input without storing a C++ type address.
///
/// HSDK5-009 resolves the copied name to a \ref PortId and validates the
/// process-local type token against the canonical port catalog. The token
/// itself is never persisted here or emitted by the serializer.
struct TemporalPredicateInputRecord {
  std::string input_name;  ///< Copied local input name in exact evaluator order.

  friend bool operator==(const TemporalPredicateInputRecord&,
                         const TemporalPredicateInputRecord&) = default;
};

/// @brief Preserve cold callable classification and profile-admission evidence.
struct TemporalPredicateCallableRecord {
  PredicateCallableKind kind{PredicateCallableKind::kStaticFunction};
  std::string implementation_fingerprint;  ///< Pointer-free type/build implementation identity.
  std::string build_evidence;              ///< Immutable package/build evidence reference.
  std::uint64_t size_bytes{};              ///< Positive finite owned descriptor size.
  std::uint64_t alignment_bytes{};         ///< Positive power-of-two descriptor alignment.
  bool no_dynamic_allocation{};            ///< Proven no-allocation execution capability.
  bool bounded_call{};                     ///< Proven finite call-duration capability.
  bool is_noexcept{};                      ///< Proven nonthrowing invocation capability.
  bool owned_lifetime{};                   ///< Whether the session owns/rebinds all receivers.
  bool self_rebind{};  ///< Whether a self-member descriptor rebinds per session.

  friend bool operator==(const TemporalPredicateCallableRecord&,
                         const TemporalPredicateCallableRecord&) = default;
};

/// @brief Preserve bounded predicate state without serializing state bytes.
struct TemporalPredicateStateRecord {
  bool present{};                   ///< Whether the evaluator has mutable state.
  std::uint64_t size_bytes{};       ///< Exact state size, or zero when absent.
  std::uint64_t alignment_bytes{};  ///< Exact state alignment, or zero when absent.
  std::uint64_t max_bytes{};        ///< Admitted finite storage bound, or zero when absent.
  std::string schema;               ///< Stable state schema identity, or empty when absent.
  TemporalPredicateResetPolicy reset_policy{TemporalPredicateResetPolicy::kResetEventsV1};

  friend bool operator==(const TemporalPredicateStateRecord&,
                         const TemporalPredicateStateRecord&) = default;
};

/// @brief Preserve bounded predicate configuration without serializing executable bytes.
struct TemporalPredicateConfigRecord {
  bool present{};                   ///< Whether explicit configuration was authored.
  std::uint64_t size_bytes{};       ///< Exact configuration size, or zero when absent.
  std::uint64_t alignment_bytes{};  ///< Exact alignment, or zero when absent.
  std::uint64_t max_bytes{};        ///< Admitted finite storage bound, or zero when absent.
  std::string schema;               ///< Stable configuration schema, or empty when absent.
  std::string digest;               ///< Canonical semantic digest, or empty when absent.

  friend bool operator==(const TemporalPredicateConfigRecord&,
                         const TemporalPredicateConfigRecord&) = default;
};

/// @brief Preserve the complete pointer-free predicate plan record.
struct TemporalPredicateRecord {
  std::string record_id;  ///< Stable plan-local predicate record identity.
  std::string identity;   ///< Local or qualified semantic identity.
  TemporalPredicateScope scope{TemporalPredicateScope::kOperatorLocal};
  std::optional<SemanticVersion> semantic_version;  ///< Required only for qualified scope.
  TemporalPredicateSourceKind source{TemporalPredicateSourceKind::kInputUpdate};
  std::string driver_input;        ///< Exact driver input, empty only for clock source.
  TemporalClockKeyRecord clock{};  ///< Schedule clock for a clock-driven predicate.
  std::int64_t period_ns{};        ///< Positive period for a clock-driven predicate.
  std::optional<TimePoint> phase;  ///< Optional absolute clock phase.
  std::vector<TemporalPredicateInputRecord> additional_inputs;  ///< Exact evaluator order.
  PredicateFiring firing{PredicateFiring::kOnFalseToTrue};
  TemporalPredicateCallableRecord callable{};  ///< Classification; never callable bytes.
  TemporalPredicateStateRecord state{};        ///< Bounded state description.
  TemporalPredicateConfigRecord config{};      ///< Bounded optional configuration.
  TemporalPredicateReplayKind replay{TemporalPredicateReplayKind::kDecisionReplay};
  std::string replay_evidence;  ///< Required only for cross-build recomputation.

  friend bool operator==(const TemporalPredicateRecord&, const TemporalPredicateRecord&) = default;
};

/// @brief Preserve one normalized per-Operator temporal contract.
///
/// The record contains copied names and logical source identities only. It owns
/// no authoring address, callable bytes, queue, cache, scheduler, clock object,
/// or backend handle. HSDK5-009 may embed a validated record into an immutable
/// \c ExecutionPlan after resolving graph-local identities.
struct ContractRecord {
  std::uint32_t version{kTemporalRecordVersion};  ///< Exact record schema version.
  std::string operator_name;                      ///< Canonical copied Operator identity.
  TemporalRecordOrigin origin{TemporalRecordOrigin::kAuthored};
  TemporalRecordTriggerKind trigger{TemporalRecordTriggerKind::kAnyInput};
  std::string trigger_input;  ///< Input source for each/input-predicate triggers.
  std::vector<TemporalTriggerPinRecord> trigger_pins;  ///< Any/each accepted-arrival pin records.
  SourceId trigger_source{};               ///< Logical source for readiness-trigger forms.
  std::string source_provenance;           ///< Logical source provenance; never a backend handle.
  TemporalClockKeyRecord trigger_clock{};  ///< Clock source for clock-trigger forms.
  std::int64_t period_ns{};                ///< Positive clock period when applicable.
  std::optional<TimePoint> phase;          ///< Optional absolute phase when applicable.
  std::optional<TemporalAlignedRecord> aligned;      ///< Present exactly for \c kAligned.
  std::optional<TemporalWindowRecord> window;        ///< Present exactly for \c kWindow.
  std::optional<TemporalPredicateRecord> predicate;  ///< Present exactly for \c kPredicate.
  std::optional<std::int64_t> deadline_ns;           ///< Positive latest-start budget when present.
  std::optional<OnViolation> violation;   ///< Present exactly when \ref deadline_ns is present.
  TemporalRecordProvenance provenance{};  ///< Stable source and declaration order.

  friend bool operator==(const ContractRecord&, const ContractRecord&) = default;
};

/// @brief Preserve one normalized per-input-edge temporal selection contract.
struct EdgeSelectionRecord {
  std::uint32_t version{kTemporalRecordVersion};  ///< Exact record schema version.
  std::string input_name;                         ///< Canonical copied local input identity.
  TemporalRecordSelectorKind selector{TemporalRecordSelectorKind::kExact};
  TemporalRecordCardinality cardinality{TemporalRecordCardinality::kScalar};
  std::int64_t max_age_ns{};            ///< Positive only for \c kLatestWithin.
  std::uint32_t lookbehind{};           ///< Positive only for \c kBracketedWindow.
  std::uint32_t lookahead{};            ///< Positive only for \c kBracketedWindow.
  TemporalReferenceRecord reference{};  ///< Selector reference and conversion requirement.
  std::uint32_t cache_capacity{};       ///< Positive finite retained-entry bound.
  TemporalRecordBoundOrigin cache_capacity_origin{TemporalRecordBoundOrigin::kSingleSample};
  std::uint32_t max_selected_samples{};  ///< Positive finite scalar/batch result bound.
  TemporalRecordBoundOrigin max_selected_origin{TemporalRecordBoundOrigin::kSingleSample};
  TemporalRecordBatchMode batch_mode{TemporalRecordBatchMode::kScalar};
  TemporalRecordMissingKind missing{TemporalRecordMissingKind::kUnspecified};
  std::int64_t reuse_max_age_ns{};        ///< Positive only for \c kReuseLast.
  TemporalRecordProvenance provenance{};  ///< Stable authored/derived edge provenance.

  friend bool operator==(const EdgeSelectionRecord&, const EdgeSelectionRecord&) = default;
};

/// @brief Validate one normalized temporal contract record without serialization.
/// @param record Record to validate.
/// @return Success only when the version, enum values, source shape, predicate,
/// window, deadline, and finite bounds are internally consistent.
[[nodiscard]] HOLOSCAN_CORE_EXPORT expected<void, Error> validate_temporal_record(
    const ContractRecord& record);

/// @brief Validate one normalized temporal edge-selection record.
/// @param record Record to validate.
/// @return Success only when selector shape, bounds, missing policy, reference,
/// and provenance are internally consistent.
[[nodiscard]] HOLOSCAN_CORE_EXPORT expected<void, Error> validate_temporal_record(
    const EdgeSelectionRecord& record);

/// @brief Serialize one contract and its canonical-order edge records to canonical JSON.
///
/// The serializer validates every record first, rejects duplicate or non-increasing
/// edge declaration order, emits a fixed schema/version and key order, and never
/// writes addresses, callable bytes, enum ordinals, padding, or host-endian data.
///
/// @param contract Normalized per-Operator contract.
/// @param edges Canonical declaration-order input selection records.
/// @return Compact deterministic JSON, or a typed validation/encoding failure.
[[nodiscard]] HOLOSCAN_CORE_EXPORT expected<std::string, Error> serialize_temporal_records(
    const ContractRecord& contract, std::span<const EdgeSelectionRecord> edges);

}  // namespace holoscan
