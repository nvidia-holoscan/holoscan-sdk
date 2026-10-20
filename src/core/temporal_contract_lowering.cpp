/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "temporal_contract_lowering.hpp"

#include <algorithm>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace holoscan::detail {
namespace {

[[nodiscard]] unexpected<Error> invalid(std::string message) {
  return make_unexpected(Error{ErrorCode::kInvalidArgument, std::move(message)});
}

[[nodiscard]] TemporalClockKeyRecord clock_record(const ClockKey& key) {
  return {.is_default = key.is_default(), .name = std::string{key.name()}};
}

[[nodiscard]] TemporalRecordReferenceKind reference_kind(
    const TemporalTriggerRecord& trigger) noexcept {
  switch (trigger.kind) {
    case TemporalTriggerKind::kAnyInput:
    case TemporalTriggerKind::kEach:
      return TemporalRecordReferenceKind::kTriggerSample;
    case TemporalTriggerKind::kClock:
      return TemporalRecordReferenceKind::kScheduledTick;
    case TemporalTriggerKind::kReadable:
    case TemporalTriggerKind::kNotified:
    case TemporalTriggerKind::kDoorbell:
    case TemporalTriggerKind::kSpin:
      return TemporalRecordReferenceKind::kReadinessWitness;
    case TemporalTriggerKind::kAligned:
      return TemporalRecordReferenceKind::kAlignedHigh;
    case TemporalTriggerKind::kWindow:
      return TemporalRecordReferenceKind::kWindowClose;
    case TemporalTriggerKind::kPredicate:
      return trigger.predicate_clock_source ? TemporalRecordReferenceKind::kPredicateTick
                                            : TemporalRecordReferenceKind::kPredicateDriver;
    case TemporalTriggerKind::kNone:
      return TemporalRecordReferenceKind::kTriggerSample;
  }
  return TemporalRecordReferenceKind::kTriggerSample;
}

[[nodiscard]] expected<TemporalRecordTriggerKind, Error> trigger_kind(TemporalTriggerKind kind) {
  switch (kind) {
    case TemporalTriggerKind::kAnyInput:
      return TemporalRecordTriggerKind::kAnyInput;
    case TemporalTriggerKind::kEach:
      return TemporalRecordTriggerKind::kEach;
    case TemporalTriggerKind::kClock:
      return TemporalRecordTriggerKind::kClock;
    case TemporalTriggerKind::kReadable:
      return TemporalRecordTriggerKind::kReadable;
    case TemporalTriggerKind::kNotified:
      return TemporalRecordTriggerKind::kNotified;
    case TemporalTriggerKind::kDoorbell:
      return TemporalRecordTriggerKind::kDoorbell;
    case TemporalTriggerKind::kSpin:
      return TemporalRecordTriggerKind::kSpin;
    case TemporalTriggerKind::kAligned:
      return TemporalRecordTriggerKind::kAligned;
    case TemporalTriggerKind::kWindow:
      return TemporalRecordTriggerKind::kWindow;
    case TemporalTriggerKind::kPredicate:
      return TemporalRecordTriggerKind::kPredicate;
    case TemporalTriggerKind::kNone:
      return invalid("cannot lower a contract without a normalized trigger");
  }
  return invalid("cannot lower an unknown normalized trigger");
}

[[nodiscard]] expected<TemporalRecordMissingKind, Error> missing_kind(TemporalMissingKind kind) {
  switch (kind) {
    case TemporalMissingKind::kUnspecified:
      return TemporalRecordMissingKind::kUnspecified;
    case TemporalMissingKind::kSkip:
      return TemporalRecordMissingKind::kSkip;
    case TemporalMissingKind::kSafeDefault:
      return TemporalRecordMissingKind::kSafeDefault;
    case TemporalMissingKind::kReuseLast:
      return TemporalRecordMissingKind::kReuseLast;
  }
  return invalid("cannot lower an unknown normalized missing policy");
}

[[nodiscard]] expected<TemporalReferenceRecord, Error> make_reference(
    TemporalRecordReferenceKind kind, const TemporalResolvedReferenceFacts& facts) {
  if (!facts.resolved) {
    return invalid("temporal lowering requires an explicit resolved reference fact");
  }
  return TemporalReferenceRecord{
      .kind = kind,
      .clock = facts.clock,
      .conversion = facts.conversion,
      .conversion_record_id = facts.conversion_record_id,
  };
}

[[nodiscard]] expected<EdgeSelectionRecord, Error> lower_edge(
    const TemporalInputRecord& input, const TemporalResolvedEdgeFacts& facts,
    TemporalRecordReferenceKind kind) {
  if (!facts.resolved || facts.input_name.empty() || facts.cache_capacity == 0U) {
    return invalid("temporal lowering requires complete finite per-edge compiler facts");
  }
  auto reference = make_reference(kind, facts.reference);
  if (!reference) {
    return make_unexpected(std::move(reference.error()));
  }

  EdgeSelectionRecord edge{
      .input_name = facts.input_name,
      .reference = std::move(*reference),
      .cache_capacity = facts.cache_capacity,
      .cache_capacity_origin = facts.cache_capacity_origin,
      .provenance = facts.provenance,
  };
  auto missing = missing_kind(input.missing);
  if (!missing) {
    return make_unexpected(std::move(missing.error()));
  }
  edge.missing = *missing;
  edge.reuse_max_age_ns = input.reuse_max_age.count();
  switch (input.selector) {
    case TemporalSelectorKind::kExact:
      edge.selector = TemporalRecordSelectorKind::kExact;
      edge.cardinality = TemporalRecordCardinality::kScalar;
      edge.max_selected_samples = 1U;
      edge.max_selected_origin = TemporalRecordBoundOrigin::kSingleSample;
      edge.batch_mode = TemporalRecordBatchMode::kScalar;
      break;
    case TemporalSelectorKind::kLatestWithin:
      edge.selector = TemporalRecordSelectorKind::kLatestWithin;
      edge.cardinality = TemporalRecordCardinality::kScalar;
      edge.max_age_ns = input.max_age.count();
      edge.max_selected_samples = 1U;
      edge.max_selected_origin = TemporalRecordBoundOrigin::kSingleSample;
      edge.batch_mode = TemporalRecordBatchMode::kScalar;
      break;
    case TemporalSelectorKind::kDrainAll:
      if (facts.drain_all_max_samples == 0U) {
        return invalid("DrainAll lowering requires a resolved finite selected-sample maximum");
      }
      edge.selector = TemporalRecordSelectorKind::kDrainAll;
      edge.cardinality = TemporalRecordCardinality::kBatch;
      edge.max_selected_samples = facts.drain_all_max_samples;
      edge.max_selected_origin = TemporalRecordBoundOrigin::kResolvedRetention;
      edge.batch_mode = TemporalRecordBatchMode::kConsumeOnAcceptance;
      break;
    case TemporalSelectorKind::kBracketedWindow:
      edge.selector = TemporalRecordSelectorKind::kBracketedWindow;
      edge.cardinality = TemporalRecordCardinality::kBatch;
      edge.lookbehind = input.bracket.lookbehind;
      edge.lookahead = input.bracket.lookahead;
      edge.max_selected_samples = input.bracket.max_samples;
      edge.max_selected_origin = TemporalRecordBoundOrigin::kSelectorMaximum;
      edge.batch_mode = TemporalRecordBatchMode::kNondestructiveSnapshot;
      break;
  }
  return edge;
}

[[nodiscard]] expected<TemporalPredicateRecord, Error> lower_predicate(
    const TemporalTriggerRecord& trigger, const TemporalResolvedPredicateFacts& facts) {
  if (!facts.resolved) {
    return invalid("predicate lowering requires explicit compiler identity and capability facts");
  }
  TemporalPredicateCallableRecord callable{
      .kind = trigger.predicate.callable_kind(),
      .implementation_fingerprint = facts.implementation_fingerprint,
      .build_evidence = facts.build_evidence,
      .size_bytes = trigger.predicate_callable_size,
      .alignment_bytes = trigger.predicate_callable_alignment,
      .no_dynamic_allocation = facts.no_dynamic_allocation,
      .bounded_call = facts.bounded_call,
      .is_noexcept = trigger.predicate_callable_noexcept,
      .owned_lifetime = facts.owned_lifetime,
      .self_rebind = facts.self_rebind,
  };
  TemporalPredicateStateRecord state{
      .present = trigger.predicate_state_size != 0U,
      .size_bytes = trigger.predicate_state_size,
      .alignment_bytes = trigger.predicate_state_alignment,
      .max_bytes = facts.state_max_bytes,
      .schema = facts.state_schema,
  };
  TemporalPredicateConfigRecord config{
      .present = trigger.predicate_config_size != 0U,
      .size_bytes = trigger.predicate_config_size,
      .alignment_bytes = trigger.predicate_config_alignment,
      .max_bytes = facts.config_max_bytes,
      .schema = facts.config_schema,
      .digest = facts.config_digest,
  };
  TemporalPredicateRecord predicate{
      .record_id = facts.record_id,
      .identity = std::string{trigger.predicate.name()},
      .scope = trigger.predicate_semantic_version.has_value()
                   ? TemporalPredicateScope::kQualified
                   : TemporalPredicateScope::kOperatorLocal,
      .semantic_version = trigger.predicate_semantic_version,
      .source = trigger.predicate_clock_source ? TemporalPredicateSourceKind::kClock
                                               : TemporalPredicateSourceKind::kInputUpdate,
      .driver_input = trigger.predicate_clock_source ? std::string{} : trigger.port_name,
      .clock =
          trigger.predicate_clock_source ? clock_record(trigger.clock) : TemporalClockKeyRecord{},
      .period_ns = trigger.predicate_clock_source ? trigger.duration.count() : 0,
      .phase = trigger.predicate_clock_source ? trigger.start_at : std::nullopt,
      .firing = trigger.predicate_firing,
      .callable = std::move(callable),
      .state = std::move(state),
      .config = std::move(config),
      .replay = facts.replay,
      .replay_evidence = facts.replay_evidence,
  };
  predicate.additional_inputs.reserve(trigger.predicate_input_names.size());
  for (const auto& name : trigger.predicate_input_names) {
    predicate.additional_inputs.push_back({.input_name = name});
  }
  return predicate;
}

}  // namespace

// The function intentionally validates through the canonical serializer after mapping. That cold
// path exercises cross-record constraints as well as individual shapes and guarantees that no
// compiler consumer can receive a record set the public schema would reject.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
expected<LoweredTemporalRecords, Error> lower_temporal_contract_records(
    const TemporalContractSnapshot& snapshot, const TemporalContractLoweringContext& context) {
  auto mapped_trigger = trigger_kind(snapshot.trigger.kind);
  if (!mapped_trigger) {
    return make_unexpected(std::move(mapped_trigger.error()));
  }

  LoweredTemporalRecords lowered;
  lowered.contract.operator_name = context.operator_name;
  lowered.contract.origin = snapshot.synthesized_default ? TemporalRecordOrigin::kCoreSynthesized
                                                         : TemporalRecordOrigin::kAuthored;
  lowered.contract.trigger = *mapped_trigger;
  lowered.contract.provenance = context.provenance;
  if (snapshot.trigger.kind == TemporalTriggerKind::kAnyInput &&
      (context.connected_input_count == 0U ||
       context.connected_input_count != snapshot.connected_input_count ||
       context.trigger_pins.size() != context.connected_input_count)) {
    return invalid("any-input lowering requires the complete compiler-confirmed connected pin set");
  }
  if (snapshot.trigger.kind == TemporalTriggerKind::kEach &&
      (context.trigger_pins.size() != 1U ||
       context.trigger_pins.front().input_name != snapshot.trigger.port_name)) {
    return invalid("each lowering requires exactly its compiler-resolved trigger pin");
  }
  if (snapshot.trigger.kind != TemporalTriggerKind::kAnyInput &&
      snapshot.trigger.kind != TemporalTriggerKind::kEach && !context.trigger_pins.empty()) {
    return invalid("only any-input and each triggers may carry trigger-pin facts");
  }
  lowered.contract.trigger_pins.reserve(context.trigger_pins.size());
  for (const auto& pin : context.trigger_pins) {
    if (!pin.resolved || pin.input_name.empty() || pin.retained_pin_capacity == 0U) {
      return invalid("trigger-pin lowering requires complete finite compiler facts");
    }
    lowered.contract.trigger_pins.push_back({
        .input_name = pin.input_name,
        .mode = TemporalTriggerPinMode::kAcceptedArrivalIdentity,
        .retained_pin_capacity = pin.retained_pin_capacity,
        .capacity_origin = TemporalRecordBoundOrigin::kResolvedRetention,
        .provenance = pin.provenance,
    });
  }
  if (snapshot.max_start_delay) {
    lowered.contract.max_start_delay_ns = snapshot.max_start_delay->count();
    lowered.contract.violation = snapshot.violation;
  }

  switch (snapshot.trigger.kind) {
    case TemporalTriggerKind::kAnyInput:
      break;
    case TemporalTriggerKind::kEach:
      lowered.contract.trigger_input = snapshot.trigger.port_name;
      break;
    case TemporalTriggerKind::kClock:
      lowered.contract.trigger_clock = clock_record(snapshot.trigger.clock);
      lowered.contract.period_ns = snapshot.trigger.duration.count();
      lowered.contract.phase = snapshot.trigger.start_at;
      break;
    case TemporalTriggerKind::kReadable:
    case TemporalTriggerKind::kNotified:
    case TemporalTriggerKind::kDoorbell:
    case TemporalTriggerKind::kSpin:
      lowered.contract.trigger_source = snapshot.trigger.source_id;
      lowered.contract.source_provenance = context.source_provenance;
      break;
    case TemporalTriggerKind::kAligned: {
      auto reference =
          make_reference(TemporalRecordReferenceKind::kAlignedHigh, context.trigger_reference);
      if (!reference) {
        return make_unexpected(std::move(reference.error()));
      }
      lowered.contract.aligned = TemporalAlignedRecord{
          .member_inputs = snapshot.trigger.ports,
          .inclusive_window_ns = snapshot.trigger.duration.count(),
          .reference = std::move(*reference),
      };
      break;
    }
    case TemporalTriggerKind::kWindow: {
      auto reference =
          make_reference(TemporalRecordReferenceKind::kWindowClose, context.trigger_reference);
      if (!reference) {
        return make_unexpected(std::move(reference.error()));
      }
      lowered.contract.trigger_input = snapshot.trigger.port_name;
      lowered.contract.window = TemporalWindowRecord{
          .anchor_input = snapshot.trigger.port_name,
          .wait_clock = clock_record(snapshot.trigger.clock),
          .max_wait_ns = snapshot.trigger.duration.count(),
          .max_pending_windows = snapshot.trigger.capacity,
          .reference = std::move(*reference),
      };
      break;
    }
    case TemporalTriggerKind::kPredicate:
      if (!context.predicate) {
        return invalid("predicate lowering requires compiler predicate facts");
      } else {
        auto predicate = lower_predicate(snapshot.trigger, *context.predicate);
        if (!predicate) {
          return make_unexpected(std::move(predicate.error()));
        }
        lowered.contract.trigger_input = predicate->driver_input;
        lowered.contract.predicate = std::move(*predicate);
      }
      break;
    case TemporalTriggerKind::kNone:
      return invalid("cannot lower a contract without a normalized trigger");
  }

  const auto kind = reference_kind(snapshot.trigger);
  std::vector<bool> matched_inputs(snapshot.inputs.size(), false);
  lowered.edges.reserve(context.edges.size());
  for (const auto& facts : context.edges) {
    const auto match = std::find_if(
        snapshot.inputs.begin(), snapshot.inputs.end(), [&](const TemporalInputRecord& input) {
          return input.port_name == facts.input_name;
        });
    if (match == snapshot.inputs.end()) {
      return invalid("compiler edge facts do not correspond to a normalized selector");
    }
    const auto index = static_cast<std::size_t>(match - snapshot.inputs.begin());
    if (matched_inputs[index]) {
      return invalid("compiler edge facts duplicate one normalized selector");
    }
    matched_inputs[index] = true;
    auto edge = lower_edge(*match, facts, kind);
    if (!edge) {
      return make_unexpected(std::move(edge.error()));
    }
    lowered.edges.push_back(std::move(*edge));
  }
  if (std::any_of(
          matched_inputs.begin(), matched_inputs.end(), [](bool matched) { return !matched; })) {
    return invalid("temporal lowering is missing compiler facts for a normalized selector");
  }

  auto serialized = serialize_temporal_records(lowered.contract, lowered.edges);
  if (!serialized) {
    return make_unexpected(std::move(serialized.error()));
  }
  return lowered;
}

}  // namespace holoscan::detail
