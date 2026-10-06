/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "temporal_plan_lowering.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iterator>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <holoscan/core/parameter_schema.hpp>
#include "core/canonical_text_encoding.hpp"
#include "core/clock_binding_resolution.hpp"
#include "core/diagnostic_context.hpp"
#include "core/graph_compiler_access.hpp"
#include "core/operator_spec_compiler_access.hpp"
#include "core/readiness_source_internal.hpp"
#include "core/source_id_internal.hpp"
#include "core/temporal_contract_lowering.hpp"
#include "profile_admission.hpp"
#include "snapshot_lookup.hpp"

namespace holoscan::detail {
namespace {

[[nodiscard]] unexpected<Error> replay_error(std::string_view token) {
  return make_unexpected(Error{ErrorCode::kFailure, std::string(token)});
}

[[nodiscard]] unexpected<Error> replay_error(std::string_view token, const Error& cause) {
  std::string message{token};
  if (!cause.message.empty()) {
    message.append(": ");
    message.append(cause.message);
  }
  return make_unexpected(Error{cause.code, std::move(message)});
}

[[nodiscard]] const PortRecord* find_port(const GraphSnapshotData& snapshot, VertexId owner,
                                          PortDirection direction, std::string_view name) noexcept {
  const auto found =
      std::find_if(snapshot.ports.begin(), snapshot.ports.end(), [&](const auto& port) {
        return port.owner == owner && port.direction == direction && port.name == name;
      });
  return found == snapshot.ports.end() ? nullptr : &*found;
}

[[nodiscard]] const EdgeRecord* incoming_edge(const GraphSnapshotData& snapshot,
                                              GraphPortId input) {
  const auto aliases = interface_alias_component(snapshot, input);
  const auto found =
      std::find_if(snapshot.edges.begin(), snapshot.edges.end(), [&](const auto& edge) {
        return std::find(aliases.begin(), aliases.end(), edge.target) != aliases.end();
      });
  return found == snapshot.edges.end() ? nullptr : &*found;
}

[[nodiscard]] std::size_t connected_input_count(const GraphSnapshotData& snapshot, VertexId owner) {
  std::size_t count = 0;
  for (const auto& port : snapshot.ports) {
    if (port.owner == owner && port.direction == PortDirection::kInput &&
        incoming_edge(snapshot, port.id) != nullptr) {
      ++count;
    }
  }
  return count;
}

[[nodiscard]] const ReadinessPlanRecord* find_readiness(
    std::span<const ReadinessPlanRecord> readiness, VertexId owner, std::uint32_t ordinal,
    ReadinessSourceKind kind, std::string_view name) noexcept {
  const auto found = std::find_if(readiness.begin(), readiness.end(), [&](const auto& source) {
    return source.owner == owner && source.declaration_ordinal == ordinal && source.kind == kind &&
           source.name == name;
  });
  return found == readiness.end() ? nullptr : &*found;
}

void bind_source(const OperatorSpecCompilerAccess::SourceMetadata& source, SourceId id,
                 PlanGenerationId generation) {
  void* declaration = const_cast<void*>(source.declaration);
  switch (source.kind) {
    case ReadinessSourceKind::kReadable:
      ReadinessSourceAccess::bind(
          *static_cast<ReadableSource*>(declaration),
          ReadinessSourceAccess::make_ref<ReadinessSourceKind::kReadable>(id, generation));
      break;
    case ReadinessSourceKind::kNotification:
      ReadinessSourceAccess::bind(
          *static_cast<NotificationSource*>(declaration),
          ReadinessSourceAccess::make_ref<ReadinessSourceKind::kNotification>(id, generation));
      break;
    case ReadinessSourceKind::kDoorbell:
      ReadinessSourceAccess::bind(
          *static_cast<DoorbellSource*>(declaration),
          ReadinessSourceAccess::make_ref<ReadinessSourceKind::kDoorbell>(id, generation));
      break;
    case ReadinessSourceKind::kSpin:
      ReadinessSourceAccess::bind(
          *static_cast<SpinSource*>(declaration),
          ReadinessSourceAccess::make_ref<ReadinessSourceKind::kSpin>(id, generation));
      break;
  }
}

[[nodiscard]] ClockId resolve_clock(const GraphSnapshotData& snapshot,
                                    std::span<const ClockPlanRecord> clocks, GraphId graph,
                                    const ClockKey& key) noexcept {
  const AuthoredClockResolution resolved = resolve_authored_clock(snapshot.graphs,
                                                                  snapshot.clocks,
                                                                  snapshot.clock_imports,
                                                                  snapshot.clock_bindings,
                                                                  graph,
                                                                  key);
  if (!resolved.resolved()) {
    return {};
  }
  const auto found = std::find_if(clocks.begin(), clocks.end(), [&](const auto& clock) {
    return clock.graph == resolved.declaration->graph && clock.key == resolved.declaration->key;
  });
  return found == clocks.end() ? ClockId{} : found->id;
}

[[nodiscard]] const ClockPlanRecord* find_clock(std::span<const ClockPlanRecord> clocks,
                                                ClockId id) noexcept {
  const auto found =
      std::find_if(clocks.begin(), clocks.end(), [&](const auto& clock) { return clock.id == id; });
  return found == clocks.end() ? nullptr : &*found;
}

[[nodiscard]] const ClockConversionRecord* find_conversion(
    std::span<const ClockConversionRecord> conversions, ClockId from, ClockId to) noexcept {
  const auto found = std::find_if(conversions.begin(), conversions.end(), [&](const auto& item) {
    return item.from == from && item.to == to;
  });
  return found == conversions.end() ? nullptr : &*found;
}

[[nodiscard]] const ClockConversionRecord* find_conversion(
    std::span<const ClockConversionRecord> conversions, std::uint64_t id) noexcept {
  const auto found = std::find_if(
      conversions.begin(), conversions.end(), [&](const auto& item) { return item.id == id; });
  return found == conversions.end() ? nullptr : &*found;
}

[[nodiscard]] const ExternalTopicRecord* imported_source_topic(const GraphSnapshotData& snapshot,
                                                               const EdgeRecord& edge) noexcept {
  const auto aliases = interface_alias_component(snapshot, edge.source);
  const auto topic = std::find_if(
      snapshot.external_topics.begin(), snapshot.external_topics.end(), [&](const auto& candidate) {
        return candidate.imported &&
               std::find(aliases.begin(), aliases.end(), candidate.graph_port) != aliases.end();
      });
  return topic == snapshot.external_topics.end() ? nullptr : &*topic;
}

[[nodiscard]] std::optional<ClockId> explicit_source_capture_clock(
    const GraphSnapshotData& snapshot, std::span<const ClockPlanRecord> clocks,
    const EdgeRecord& edge) {
  const auto* topic = imported_source_topic(snapshot, edge);
  if (topic == nullptr || !topic->capture_time) {
    return std::nullopt;
  }
  const VertexRecord* boundary = find_vertex(snapshot, topic->vertex);
  return boundary == nullptr
             ? ClockId{}
             : resolve_clock(snapshot, clocks, boundary->graph, topic->capture_time->clock);
}

struct SourceClockCoordinate {
  ClockId source_clock{};
  std::uint64_t conversion_id{};

  friend bool operator==(const SourceClockCoordinate&, const SourceClockCoordinate&) = default;
};

[[nodiscard]] std::optional<SourceClockCoordinate> input_source_coordinate(
    const GraphSnapshotData& snapshot, std::span<const ClockPlanRecord> clocks,
    std::span<const ClockConversionRecord> conversions, PortId input, ClockId reference_clock) {
  const PortRecord* port = find_port(snapshot, GraphPortId{input.value()});
  const EdgeRecord* edge = port == nullptr ? nullptr : incoming_edge(snapshot, port->id);
  if (edge == nullptr || !reference_clock.valid()) {
    return std::nullopt;
  }
  const std::optional<ClockId> explicit_clock =
      explicit_source_capture_clock(snapshot, clocks, *edge);
  if (explicit_clock && !explicit_clock->valid()) {
    return std::nullopt;
  }
  const ClockId source_clock = explicit_clock.value_or(reference_clock);
  if (source_clock == reference_clock) {
    return SourceClockCoordinate{.source_clock = source_clock};
  }
  const ClockConversionRecord* conversion =
      find_conversion(conversions, source_clock, reference_clock);
  return conversion == nullptr
             ? std::nullopt
             : std::optional<SourceClockCoordinate>{SourceClockCoordinate{
                   .source_clock = source_clock, .conversion_id = conversion->id}};
}

[[nodiscard]] constexpr bool requires_capture_time(TemporalSelectorKind selector) noexcept {
  return selector == TemporalSelectorKind::kExact ||
         selector == TemporalSelectorKind::kLatestWithin ||
         selector == TemporalSelectorKind::kBracketedWindow;
}

[[nodiscard]] const GraphCompilerAccess::TemporalReplayPort* replay_port(
    const GraphCompilerAccess::TemporalReplay& replay, std::string_view name) noexcept {
  const auto found = std::find_if(replay.ports.begin(), replay.ports.end(), [&](const auto& port) {
    return port.direction == PortDirection::kInput && port.name == name;
  });
  return found == replay.ports.end() ? nullptr : &*found;
}

[[nodiscard]] std::uint32_t effective_queue_depth(const PortRecord& port,
                                                  const EdgeRecord& edge) noexcept {
  if (edge.graph_override.queue_depth) {
    return *edge.graph_override.queue_depth;
  }
  if (edge.add_flow_options.queue_depth) {
    return *edge.add_flow_options.queue_depth;
  }
  if (port.graph_override.queue_depth) {
    return *port.graph_override.queue_depth;
  }
  return port.setup.queue_depth.value_or(1U);
}

[[nodiscard]] bool has_explicit_queue_depth(const PortRecord& port,
                                            const EdgeRecord& edge) noexcept {
  return edge.graph_override.queue_depth.has_value() ||
         edge.add_flow_options.queue_depth.has_value() ||
         port.graph_override.queue_depth.has_value() || port.setup.queue_depth.has_value();
}

[[nodiscard]] QueuePolicy effective_queue_policy(const EdgeRecord& edge) noexcept {
  return edge.graph_override.queue_policy.value_or(edge.add_flow_options.queue_policy);
}

[[nodiscard]] InputSelectorKind selector_kind(TemporalRecordSelectorKind kind) noexcept {
  switch (kind) {
    case TemporalRecordSelectorKind::kExact:
      return InputSelectorKind::kExact;
    case TemporalRecordSelectorKind::kLatestWithin:
      return InputSelectorKind::kLatestWithin;
    case TemporalRecordSelectorKind::kDrainAll:
      return InputSelectorKind::kDrainAll;
    case TemporalRecordSelectorKind::kBracketedWindow:
      return InputSelectorKind::kBracketedWindow;
  }
  return InputSelectorKind::kExact;
}

[[nodiscard]] InputSelectionCardinality cardinality(TemporalRecordCardinality value) noexcept {
  return value == TemporalRecordCardinality::kScalar ? InputSelectionCardinality::kSingleSample
                                                     : InputSelectionCardinality::kSampleBatch;
}

[[nodiscard]] SelectedSampleBoundOrigin selected_origin(TemporalRecordBoundOrigin value) noexcept {
  switch (value) {
    case TemporalRecordBoundOrigin::kSingleSample:
      return SelectedSampleBoundOrigin::kSingleSample;
    case TemporalRecordBoundOrigin::kResolvedRetention:
      return SelectedSampleBoundOrigin::kResolvedRetainedCapacity;
    case TemporalRecordBoundOrigin::kSelectorMaximum:
      return SelectedSampleBoundOrigin::kSelectorMaxSamples;
  }
  return SelectedSampleBoundOrigin::kSingleSample;
}

[[nodiscard]] BatchSelectionMode batch_mode(TemporalRecordBatchMode value) noexcept {
  switch (value) {
    case TemporalRecordBatchMode::kScalar:
      return BatchSelectionMode::kNotApplicable;
    case TemporalRecordBatchMode::kConsumeOnAcceptance:
      return BatchSelectionMode::kConsumeSelected;
    case TemporalRecordBatchMode::kNondestructiveSnapshot:
      return BatchSelectionMode::kNonDestructiveSnapshot;
  }
  return BatchSelectionMode::kNotApplicable;
}

[[nodiscard]] DeadlineSourceCoordinateKind deadline_source_coordinate(
    const ContractRecord& contract) noexcept {
  switch (contract.trigger) {
    case TemporalRecordTriggerKind::kAnyInput:
    case TemporalRecordTriggerKind::kEach:
      return DeadlineSourceCoordinateKind::kTriggerSample;
    case TemporalRecordTriggerKind::kClock:
      return DeadlineSourceCoordinateKind::kScheduledTick;
    case TemporalRecordTriggerKind::kReadable:
    case TemporalRecordTriggerKind::kNotified:
    case TemporalRecordTriggerKind::kDoorbell:
    case TemporalRecordTriggerKind::kSpin:
      return DeadlineSourceCoordinateKind::kReadinessWitness;
    case TemporalRecordTriggerKind::kAligned:
      return DeadlineSourceCoordinateKind::kAlignedHighOrClosureProof;
    case TemporalRecordTriggerKind::kWindow:
      return DeadlineSourceCoordinateKind::kWindowClose;
    case TemporalRecordTriggerKind::kPredicate:
      return contract.predicate && contract.predicate->source == TemporalPredicateSourceKind::kClock
                 ? DeadlineSourceCoordinateKind::kScheduledTick
                 : DeadlineSourceCoordinateKind::kPredicateDriver;
  }
  return DeadlineSourceCoordinateKind::kTriggerSample;
}

[[nodiscard]] std::uint32_t deadline_callback_count(OnViolation action) noexcept {
  return action == OnViolation::kFailRun ? 0U : 1U;
}

[[nodiscard]] MissingAggregateInputPlanRecord aggregate_input(
    const InputSelectionPlanRecord& selection, std::uint32_t ordinal) noexcept {
  return MissingAggregateInputPlanRecord{
      .input = selection.input,
      .ordinal = ordinal,
      .selector = selection.selector,
      .missing = selection.missing,
  };
}

[[nodiscard]] MissingAggregatePlanRecord make_missing_aggregate(
    OperatorRef owner, MissingSelectionPhase phase,
    std::span<const InputSelectionPlanRecord> selections) {
  MissingAggregatePlanRecord record{.owner = owner, .phase = phase};
  record.inputs.reserve(selections.size());
  std::vector<const InputSelectionPlanRecord*> canonical;
  canonical.reserve(selections.size());
  for (const auto& selection : selections) {
    canonical.push_back(&selection);
  }
  std::sort(canonical.begin(), canonical.end(), [](const auto* left, const auto* right) {
    return left->input.value() < right->input.value();
  });
  for (std::size_t index = 0U; index < canonical.size(); ++index) {
    record.inputs.push_back(aggregate_input(*canonical[index], static_cast<std::uint32_t>(index)));
  }
  return record;
}

[[nodiscard]] std::optional<MissingAggregatePlanRecord> make_missing_aggregate(
    OperatorRef owner, MissingSelectionPhase phase,
    std::span<const InputSelectionPlanRecord> selections, std::span<const PortId> ordered_inputs) {
  MissingAggregatePlanRecord record{.owner = owner, .phase = phase};
  record.inputs.reserve(ordered_inputs.size());
  for (std::size_t index = 0U; index < ordered_inputs.size(); ++index) {
    const auto selection =
        std::find_if(selections.begin(), selections.end(), [&](const auto& item) {
          return item.input == ordered_inputs[index];
        });
    if (selection == selections.end()) {
      return std::nullopt;
    }
    record.inputs.push_back(aggregate_input(*selection, static_cast<std::uint32_t>(index)));
  }
  return record;
}

[[nodiscard]] PredicateReplayCapability replay_capability(
    TemporalPredicateReplayKind value) noexcept {
  switch (value) {
    case TemporalPredicateReplayKind::kDecisionReplay:
      return PredicateReplayCapability::kDecisionReplay;
    case TemporalPredicateReplayKind::kLocalGraphExecReevaluation:
      return PredicateReplayCapability::kLocalGraphExecReevaluation;
    case TemporalPredicateReplayKind::kCrossBuildRecomputation:
      return PredicateReplayCapability::kCrossBuildRecomputation;
  }
  return PredicateReplayCapability::kDecisionReplay;
}

[[nodiscard]] bool predicate_python(PredicateCallableKind kind) noexcept {
  return kind == PredicateCallableKind::kPythonFunction ||
         kind == PredicateCallableKind::kPythonLambda ||
         kind == PredicateCallableKind::kPythonBoundMethod;
}

[[nodiscard]] PredicateCallableDisposition callable_disposition(PredicateCallableKind kind,
                                                                bool /* is_noexcept */,
                                                                Profile profile) noexcept {
  if (predicate_python(kind)) {
    return PredicateCallableDisposition::kRejected;
  }
  if (kind == PredicateCallableKind::kTypeErasedFunction) {
    return profile == Profile::kStandard || profile == Profile::kResearch
               ? PredicateCallableDisposition::kAdmittedWithWarning
               : PredicateCallableDisposition::kRejected;
  }
  if (profile == Profile::kStandard || profile == Profile::kResearch) {
    return PredicateCallableDisposition::kAdmitted;
  }
  if (profile == Profile::kRobotics && (kind == PredicateCallableKind::kStaticFunction ||
                                        kind == PredicateCallableKind::kNonCapturingLambda)) {
    // The maintained matrix admits direct noncritical Robotics callables. A
    // critical-region contributor would require bounded-call and nonthrowing
    // package evidence; no such contributor is present in this baseline slice.
    return PredicateCallableDisposition::kAdmitted;
  }
  // Callable category and noexcept constrain dispatch/exception shape, but do
  // not prove evaluator-body allocation or duration. Constrained profiles stay
  // fail-closed until immutable package capability evidence is available.
  return PredicateCallableDisposition::kRejected;
}

[[nodiscard]] std::string_view predicate_dispatch_model(PredicateCallableKind kind) noexcept {
  switch (kind) {
    case PredicateCallableKind::kStaticFunction:
    case PredicateCallableKind::kNonCapturingLambda:
      return "direct_function";
    case PredicateCallableKind::kInlineFunctor:
    case PredicateCallableKind::kOpaqueClosure:
      return "inline_owned";
    case PredicateCallableKind::kBoundMember:
      return "self_rebound_member";
    case PredicateCallableKind::kTypeErasedFunction:
      return "type_erased_owned";
    case PredicateCallableKind::kPythonFunction:
    case PredicateCallableKind::kPythonLambda:
    case PredicateCallableKind::kPythonBoundMethod:
      return "python_reserved";
  }
  return "invalid";
}

[[nodiscard]] std::string_view predicate_binding_provenance(PredicateCallableKind kind) noexcept {
  switch (kind) {
    case PredicateCallableKind::kStaticFunction:
    case PredicateCallableKind::kNonCapturingLambda:
      return "static_or_stateless";
    case PredicateCallableKind::kInlineFunctor:
      return "owned_inline_functor";
    case PredicateCallableKind::kOpaqueClosure:
      return "owned_opaque_closure";
    case PredicateCallableKind::kBoundMember:
      return "operator_self_rebind";
    case PredicateCallableKind::kTypeErasedFunction:
      return "owned_type_erasure";
    case PredicateCallableKind::kPythonFunction:
    case PredicateCallableKind::kPythonLambda:
    case PredicateCallableKind::kPythonBoundMethod:
      return "python_adapter";
  }
  return "invalid";
}

[[nodiscard]] std::string_view predicate_capture_provenance(PredicateCallableKind kind) noexcept {
  switch (kind) {
    case PredicateCallableKind::kStaticFunction:
    case PredicateCallableKind::kNonCapturingLambda:
      return "none";
    case PredicateCallableKind::kInlineFunctor:
      return "owned_structural_inline";
    case PredicateCallableKind::kOpaqueClosure:
      return "owned_opaque";
    case PredicateCallableKind::kBoundMember:
      return "self_descriptor";
    case PredicateCallableKind::kTypeErasedFunction:
      return "owned_type_erased";
    case PredicateCallableKind::kPythonFunction:
    case PredicateCallableKind::kPythonLambda:
    case PredicateCallableKind::kPythonBoundMethod:
      return "python_owned";
  }
  return "invalid";
}

[[nodiscard]] std::string predicate_type_digest(std::string_view schema, std::uint64_t size_bytes,
                                                std::uint64_t alignment_bytes) {
  StableHashBuilder hash;
  hash.add("holoscan.temporal.predicate-type.v1");
  hash.add(schema);
  hash.add(size_bytes);
  hash.add(alignment_bytes);
  return lowercase_hex_text(hash.finish().value, HexPrefix::kSha256);
}

[[nodiscard]] PredicateCallablePlanRecord make_predicate_callable_record(
    std::string predicate_id, const TemporalTriggerRecord& trigger, std::string fingerprint,
    std::string build_evidence, Profile profile, PredicateCallableDisposition disposition,
    PlanGenerationId plan_generation) {
  const PredicateCallableKind kind = trigger.predicate.callable_kind();
  std::vector<std::string> diagnostics;
  if (disposition == PredicateCallableDisposition::kAdmittedWithWarning) {
    diagnostics.emplace_back("PREDICATE_CALLABLE_TYPE_ERASED");
  } else if (disposition == PredicateCallableDisposition::kRejected) {
    diagnostics.emplace_back("PREDICATE_CALLABLE_PROFILE_UNSUPPORTED");
  }
  return PredicateCallablePlanRecord{
      .predicate_id = std::move(predicate_id),
      .kind = kind,
      .implementation_fingerprint = std::move(fingerprint),
      .build_evidence = std::move(build_evidence),
      .toolchain_evidence = "unqualified",
      .predicate_key = trigger.predicate_semantic_version
                           ? std::optional<std::string>{std::string(trigger.predicate.name())}
                           : std::nullopt,
      .predicate_key_version = trigger.predicate_semantic_version,
      .size_bytes = trigger.predicate_callable_size,
      .alignment_bytes = trigger.predicate_callable_alignment,
      .no_dynamic_allocation = false,
      .bounded_call = false,
      .is_noexcept = trigger.predicate_callable_noexcept,
      .owned_lifetime = !predicate_python(kind),
      .self_rebind = kind == PredicateCallableKind::kBoundMember,
      .result_kind = trigger.predicate_callable_returns_expected
                         ? PredicateResultKind::kExpectedBoolError
                         : PredicateResultKind::kBool,
      .argument_contract = "holoscan.temporal.predicate-arguments.v1",
      .dispatch_model = std::string(predicate_dispatch_model(kind)),
      .binding_provenance = std::string(predicate_binding_provenance(kind)),
      .capture_provenance = std::string(predicate_capture_provenance(kind)),
      .receiver_role = kind == PredicateCallableKind::kBoundMember ? "self" : "",
      .type_erased = kind == PredicateCallableKind::kTypeErasedFunction,
      .possible_dynamic_allocation = true,
      .requires_python_gil = predicate_python(kind),
      .criticality_constraints = {},
      .diagnostic_ids = std::move(diagnostics),
      .capability_evidence_references = {},
      .evidence_validity_epoch = plan_generation,
      .evidence_revocation_epoch = 0U,
      .effective_profile = profile,
      .disposition = disposition,
  };
}

void add_error(TemporalPlanLoweringResult& result, std::string token, VertexId owner = {},
               PortId port = {}) {
  result.errors.push_back(
      {.token = std::move(token), .source = DiagnosticSource{.vertex = owner, .port = port}});
}

void add_replay_error(TemporalPlanLoweringResult& result, const Error& error) {
  const std::string_view message = error.message;
  const std::size_t separator = message.find(": ");
  const std::string_view token =
      separator == std::string_view::npos ? message : message.substr(0U, separator);
  const std::string_view detail =
      separator == std::string_view::npos ? std::string_view{} : message.substr(separator + 2U);
  result.errors.push_back(
      {.token = token.empty() ? "TEMPORAL_CONTRACT_REPLAY_FAILED" : std::string(token),
       .provided = std::string(detail)});
}

void add_warning(TemporalPlanLoweringResult& result, std::string token, VertexId owner = {},
                 PortId port = {}) {
  result.warnings.push_back(
      {.token = std::move(token), .source = DiagnosticSource{.vertex = owner, .port = port}});
}

[[nodiscard]] bool completion_trigger_supported(TemporalTriggerKind kind) noexcept {
  switch (kind) {
    case TemporalTriggerKind::kClock:
    case TemporalTriggerKind::kReadable:
    case TemporalTriggerKind::kNotified:
    case TemporalTriggerKind::kDoorbell:
    case TemporalTriggerKind::kSpin:
      return true;
    case TemporalTriggerKind::kNone:
    case TemporalTriggerKind::kAnyInput:
    case TemporalTriggerKind::kEach:
    case TemporalTriggerKind::kAligned:
    case TemporalTriggerKind::kWindow:
    case TemporalTriggerKind::kPredicate:
      return false;
  }
  return false;
}

[[nodiscard]] bool completion_placement_supported(
    const GraphCompilerAccess::TemporalReplay& replay) noexcept {
  const bool has_input =
      std::any_of(replay.ports.begin(), replay.ports.end(), [](const auto& port) {
        return port.direction == PortDirection::kInput;
      });
  const bool has_output =
      std::any_of(replay.ports.begin(), replay.ports.end(), [](const auto& port) {
        return port.direction == PortDirection::kOutput;
      });
  return !has_input && has_output && completion_trigger_supported(replay.contract.trigger.kind);
}

void add_missing_default_clock_error(TemporalPlanLoweringResult& result,
                                     const GraphSnapshotData& snapshot, GraphId graph) {
  const GraphRecord* record = find_graph(snapshot, graph);
  const std::size_t index = record == nullptr ? 0U : static_cast<std::size_t>(graph.value - 1U);
  const bool embedded = record != nullptr && record->parent.has_value();
  const std::string scope = std::string(embedded ? "embedded graph \"" : "root graph \"") +
                            (record == nullptr ? "_root" : canonical_graph_path(*record)) + "\"";
  const bool has_import = std::any_of(snapshot.clock_imports.begin(),
                                      snapshot.clock_imports.end(),
                                      [&](const auto& item) { return item.graph == graph; });
  const bool has_clock = std::any_of(snapshot.clocks.begin(),
                                     snapshot.clocks.end(),
                                     [&](const auto& item) { return item.graph == graph; });
  std::string required;
  std::string provided = scope + " has no selected default clock";
  if (embedded && !has_import && !has_clock) {
    const GraphRecord* parent = find_graph(snapshot, *record->parent);
    required = "Declare a clock dependency using import_clock() in " + scope +
               ", select it using set_default_clock(), and bind it using bind_clock() in "
               "the immediate parent graph \"" +
               (parent == nullptr ? "_root" : canonical_graph_path(*parent)) + "\"";
    provided += " and has not declared a clock import";
  } else if (has_import || has_clock) {
    required = "Select a declared " + std::string(has_import ? "clock import" : "clock") +
               " using set_default_clock() in " + scope;
  } else {
    required =
        "Declare a clock using add_clock() and select it using set_default_clock() in " + scope;
  }
  result.errors.push_back(TemporalPlanDiagnostic{
      .token = "TEMPORAL_CLOCK_UNDECLARED",
      .source = DiagnosticSource{.graph_generation = snapshot.generation, .graph = graph},
      .json_pointer = "/graphs/" + std::to_string(index) + "/default_clock",
      .required = std::move(required),
      .provided = std::move(provided),
  });
}

void add_undeclared_temporal_clock_error(TemporalPlanLoweringResult& result,
                                         const GraphSnapshotData& snapshot,
                                         const VertexRecord& vertex, PortId port,
                                         std::string json_pointer, std::string required,
                                         std::string provided) {
  result.errors.push_back(TemporalPlanDiagnostic{
      .token = "TEMPORAL_CLOCK_UNDECLARED",
      .source = DiagnosticSource{.graph_generation = snapshot.generation,
                                 .graph = vertex.graph,
                                 .vertex = vertex.id,
                                 .port = port},
      .json_pointer = std::move(json_pointer),
      .required = std::move(required),
      .provided = std::move(provided),
  });
}

[[nodiscard]] std::size_t vertex_index(const GraphSnapshotData& snapshot, VertexId owner) noexcept {
  return find_vertex(snapshot, owner) == nullptr ? 0U : static_cast<std::size_t>(owner.value - 1U);
}

[[nodiscard]] std::uint32_t first_unassigned_source_id(
    std::span<const ReadinessPlanRecord> readiness) noexcept {
  std::uint32_t next = 1U;
  for (const ReadinessPlanRecord& source : readiness) {
    const std::uint32_t assigned = source_id_value(source.source_id);
    if (assigned == std::numeric_limits<std::uint32_t>::max()) {
      return 0U;
    }
    next = std::max(next, assigned + 1U);
  }
  return next;
}

[[nodiscard]] std::optional<SourceId> allocate_source_id(std::uint32_t& next) noexcept {
  if (next == 0U) {
    return std::nullopt;
  }
  const SourceId assigned = make_source_id(next);
  if (next == std::numeric_limits<std::uint32_t>::max()) {
    next = 0U;
  } else {
    ++next;
  }
  return assigned;
}

[[nodiscard]] std::vector<PortId> connected_inputs(const GraphSnapshotData& snapshot,
                                                   VertexId owner) {
  std::vector<PortId> result;
  for (const auto& port : snapshot.ports) {
    if (port.owner == owner && port.direction == PortDirection::kInput &&
        incoming_edge(snapshot, port.id) != nullptr) {
      result.emplace_back(port.id.value);
    }
  }
  return result;
}

[[nodiscard]] std::vector<PortId> presence_set(const GraphCompilerAccess::TemporalReplay& replay,
                                               const TemporalContractSnapshot& contract) {
  std::vector<PortId> result;
  auto append = [&](std::string_view name) {
    const auto* port = replay_port(replay, name);
    if (port != nullptr && std::find(result.begin(), result.end(), port->id) == result.end()) {
      result.push_back(port->id);
    }
  };
  switch (contract.trigger.kind) {
    case TemporalTriggerKind::kAnyInput:
      break;
    case TemporalTriggerKind::kEach:
    case TemporalTriggerKind::kWindow:
      append(contract.trigger.port_name);
      break;
    case TemporalTriggerKind::kAligned:
      for (const auto& input : contract.trigger.ports) {
        append(input);
      }
      break;
    case TemporalTriggerKind::kPredicate:
      if (!contract.trigger.predicate_clock_source) {
        append(contract.trigger.port_name);
      }
      break;
    case TemporalTriggerKind::kClock:
    case TemporalTriggerKind::kReadable:
    case TemporalTriggerKind::kNotified:
    case TemporalTriggerKind::kDoorbell:
    case TemporalTriggerKind::kSpin:
    case TemporalTriggerKind::kNone:
      break;
  }
  return result;
}

[[nodiscard]] bool optional_safe_default_contract(const TemporalContractSnapshot& contract,
                                                  const TemporalInputRecord& input) noexcept {
  if (input.selector != TemporalSelectorKind::kExact ||
      input.missing != TemporalMissingKind::kSafeDefault) {
    return false;
  }

  switch (contract.trigger.kind) {
    case TemporalTriggerKind::kAnyInput:
    case TemporalTriggerKind::kClock:
    case TemporalTriggerKind::kReadable:
    case TemporalTriggerKind::kNotified:
    case TemporalTriggerKind::kDoorbell:
    case TemporalTriggerKind::kSpin:
      return true;
    case TemporalTriggerKind::kEach:
    case TemporalTriggerKind::kWindow:
      return input.port_name != contract.trigger.port_name;
    case TemporalTriggerKind::kAligned:
      return std::find(contract.trigger.ports.begin(),
                       contract.trigger.ports.end(),
                       input.port_name) == contract.trigger.ports.end();
    case TemporalTriggerKind::kPredicate:
      return (contract.trigger.predicate_clock_source ||
              input.port_name != contract.trigger.port_name) &&
             std::find(contract.trigger.predicate_input_names.begin(),
                       contract.trigger.predicate_input_names.end(),
                       input.port_name) == contract.trigger.predicate_input_names.end();
    case TemporalTriggerKind::kNone:
      return false;
  }
  return false;
}

// Retain supplemental text only after the structural row exists. Allocation failure may lose
// explanation, but cannot change the diagnostic's classification or invent an underlying Error.
void attach_optional_input_context(TemporalPlanDiagnostic& diagnostic,
                                   const GraphSnapshotData& snapshot, const PortRecord& port,
                                   std::string_view provided, std::string_view remedy) noexcept {
  auto& detail = diagnostic.detail.emplace(DiagnosticDetail{
      .operation = DiagnosticOperation::kCompile, .error = Error{ErrorCode::kSuccess}});
  try {
    if (const auto* owner = find_vertex(snapshot, port.owner)) {
      copy_diagnostic_context(detail.operator_path,
                              canonical_operator_path(snapshot, *owner).value_or(owner->name),
                              detail,
                              DiagnosticCapture::kTextCapacity / 2U);
    }
    copy_diagnostic_context(detail.port_name, port.name, detail, 255U);
    detail.mismatches.push_back(DiagnosticMismatch{.field = "optional input contract",
                                                   .provided = std::string(provided),
                                                   .required = std::string(remedy)});
  } catch (...) {
    detail.truncated = true;
  }
}

[[nodiscard]] std::string_view optional_contract_violation(const TemporalContractSnapshot* contract,
                                                           std::string_view port_name) noexcept {
  if (contract == nullptr) {
    return "input has no temporal contract";
  }
  const auto input = std::find_if(contract->inputs.begin(),
                                  contract->inputs.end(),
                                  [&](const auto& item) { return item.port_name == port_name; });
  if (input == contract->inputs.end()) {
    return "input is not selected by its temporal contract";
  }
  switch (input->selector) {
    case TemporalSelectorKind::kLatestWithin:
      return "LatestWithin is not eligible for optional absence";
    case TemporalSelectorKind::kDrainAll:
      return "DrainAll is a batch selector";
    case TemporalSelectorKind::kBracketedWindow:
      return "BracketedWindow is a batch selector";
    case TemporalSelectorKind::kExact:
      break;
  }
  switch (input->missing) {
    case TemporalMissingKind::kUnspecified:
      return "missing policy is unspecified";
    case TemporalMissingKind::kSkip:
      return "Skip suppresses activation instead of providing SafeDefault";
    case TemporalMissingKind::kReuseLast:
      return "ReuseLast requires previously observed input data";
    case TemporalMissingKind::kSafeDefault:
      break;
  }
  switch (contract->trigger.kind) {
    case TemporalTriggerKind::kAligned:
      return "input is a required OnAligned member";
    case TemporalTriggerKind::kWindow:
      return "input is the OnWindow anchor";
    case TemporalTriggerKind::kPredicate:
      return "input supplies predicate evaluation or trigger authority";
    case TemporalTriggerKind::kEach:
      return "input is the OnEach trigger driver";
    default:
      return "input has no independent activation authority";
  }
}

[[nodiscard]] bool runtime_parameter_updates_requested(
    const ParameterCompileOptions& options) noexcept {
  return options.runtime_updates == ParamRuntimeUpdatePolicy::kEnabledIfAdmitted ||
         options.runtime_updates == ParamRuntimeUpdatePolicy::kRequirePreallocated;
}

[[nodiscard]] bool resolve_duration_dependency(const GraphCompilerAccess::TemporalReplay& replay,
                                               const ParamDurationDependency& dependency,
                                               ParameterDerivedTermKind kind, PortId input,
                                               std::optional<Profile> profile,
                                               const ParameterCompileOptions& parameter_options,
                                               ParameterDerivedTermPlanRecord& record,
                                               std::int64_t& initial_value_ns) {
  if (!dependency || replay.parameter_type == nullptr || replay.parameter_schema == nullptr ||
      replay.initial_parameter_value == nullptr ||
      dependency.root_type_token != replay.parameter_type->local_type_token ||
      dependency.root_schema() != replay.parameter_schema) {
    return false;
  }
  const ParamFieldDescriptor* field = dependency.final_field();
  const std::string field_path = dependency.canonical_path();
  if (field == nullptr || field_path.empty() || field->type_shape.optional ||
      field->logical_type != ParamLogicalType::kDuration || !field->type_shape.has_stable_shape ||
      (field->type_shape.element_or_rep_type != ParamLogicalType::kSignedInteger &&
       field->type_shape.element_or_rep_type != ParamLogicalType::kUnsignedInteger) ||
      !field->constraints.has_range ||
      !dependency.read_initial_ns(replay.initial_parameter_value, initial_value_ns)) {
    return false;
  }
  std::int64_t minimum_ns{};
  std::int64_t maximum_ns{};
  if (!dependency.constraint_to_ns(field->constraints.range_min, minimum_ns) ||
      !dependency.constraint_to_ns(field->constraints.range_max, maximum_ns) || minimum_ns <= 0 ||
      maximum_ns < minimum_ns || initial_value_ns < minimum_ns || initial_value_ns > maximum_ns) {
    return false;
  }

  ParameterDerivedTermUpdateDisposition update_disposition =
      ParameterDerivedTermUpdateDisposition::kFrozenCompileTime;
  if (field->mutability == ParamMutability::kRuntime &&
      runtime_parameter_updates_requested(parameter_options) && profile &&
      parameter_mutation_profile_disposition(*profile) ==
          ParameterMutationProfileDisposition::kRuntimeAndLifecycle) {
    update_disposition = ParameterDerivedTermUpdateDisposition::kRuntimeAtomic;
  } else if (field->mutability == ParamMutability::kLifecycle ||
             field->mutability == ParamMutability::kRecompile) {
    update_disposition = ParameterDerivedTermUpdateDisposition::kRecompileRequired;
  }

  const std::string term_suffix =
      kind == ParameterDerivedTermKind::kLatestWithinMaxAge
          ? "/input:" + std::to_string(input.value()) + "/latest_within.max_age"
          : "/contract.deadline";
  record = ParameterDerivedTermPlanRecord{
      .term_id = "operator:" + std::to_string(replay.operator_id) + term_suffix,
      .owner = OperatorRef{.id = replay.operator_id},
      .kind = kind,
      .input = input,
      .field = field->id,
      .field_full_id = field->full_id,
      .field_path = field_path,
      .schema_id = std::string(replay.parameter_schema->stable_id),
      .schema_version = replay.parameter_schema->version,
      .mutability = field->mutability,
      .update_disposition = update_disposition,
      .initial_value_ns = initial_value_ns,
      .minimum_value_ns = minimum_ns,
      .maximum_value_ns = maximum_ns,
  };
  return true;
}

}  // namespace

expected<std::vector<GraphCompilerAccess::TemporalReplay>, Error>
// Replay is transactional across each Operator's typed temporal declarations.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
GraphCompilerAccess::replay_temporal_contracts(const ExecutableFactoryInventory& factories,
                                               const GraphSnapshotData& snapshot,
                                               PlanGenerationId plan_generation,
                                               std::span<const ReadinessPlanRecord> readiness,
                                               Diagnostic* failure) {
  // REQ: COMPILE-015 - replay typed declarations into the canonical plan
  // generation before resolving copied temporal identities.
  if (plan_generation == 0U) {
    return replay_error("TEMPORAL_PLAN_GENERATION_INVALID");
  }
  std::vector<TemporalReplay> result;
  result.reserve(factories.operators.size());
  for (const auto& entry : factories.operators) {
    if (!entry.factory || entry.operator_id == 0U) {
      return replay_error("TEMPORAL_CONTRACT_FACTORY_INVALID");
    }
    const VertexId owner = entry.vertex;
    const VertexRecord* vertex = find_vertex(snapshot, owner);
    if (!owner || vertex == nullptr) {
      return replay_error("TEMPORAL_CONTRACT_OWNER_UNRESOLVED");
    }
    DiagnosticCapture capture;
    const auto failed =
        [&](std::string_view token, DiagnosticOperation operation, const Error* cause = nullptr) {
          if (capture.count == 0U && cause != nullptr) {
            capture.append(cause->code, cause->message);
          }
          if (failure != nullptr) {
            failure->source = DiagnosticSource{
                .graph_generation = snapshot.generation, .graph = vertex->graph, .vertex = owner};
            failure->detail = capture.detail(operation);
            failure->detail->operator_path =
                canonical_operator_path(snapshot, *vertex).value_or(vertex->name);
          }
          return replay_error(token);
        };
    auto created = entry.factory->create(&capture);
    if (!created || !created.value()) {
      if (created) {
        capture.append(ErrorCode::kFailure, "factory returned no operator");
      }
      return failed("TEMPORAL_CONTRACT_FACTORY_REPLAY_FAILED",
                    DiagnosticOperation::kConstruct,
                    created ? nullptr : &created.error());
    }

    OperatorSpec spec;
    try {
      created.value()->setup(spec);
    } catch (...) {
      capture.capture_exception();
      return failed("TEMPORAL_CONTRACT_SETUP_FAILED", DiagnosticOperation::kSetup);
    }
    if (OperatorSpecCompilerAccess::declaration_error(spec).code != ErrorCode::kSuccess) {
      return failed("TEMPORAL_CONTRACT_SETUP_FAILED",
                    DiagnosticOperation::kSetup,
                    &OperatorSpecCompilerAccess::declaration_error(spec));
    }

    TemporalReplay replay{
        .owner = owner,
        .operator_id = entry.operator_id,
        .operator_name = vertex->name,
        .parameter_type = &entry.factory->parameter_type(),
        .parameter_schema = entry.factory->parameter_schema(),
        .initial_parameter_value = entry.factory->initial_parameter_value(),
    };
    replay.ports.reserve(OperatorSpecCompilerAccess::port_count(spec));
    for (std::size_t index = 0; index < OperatorSpecCompilerAccess::port_count(spec); ++index) {
      const auto metadata = OperatorSpecCompilerAccess::port(spec, index);
      const PortRecord* canonical = find_port(snapshot, owner, metadata.direction, metadata.name);
      if (canonical == nullptr) {
        return replay_error("TEMPORAL_CONTRACT_PORT_UNRESOLVED");
      }
      replay.ports.push_back({.id = PortId{canonical->id.value},
                              .direction = metadata.direction,
                              .name = std::string(metadata.name),
                              .local_type_token = metadata.local_type_token});
    }
    for (std::size_t index = 0; index < OperatorSpecCompilerAccess::source_count(spec); ++index) {
      const auto metadata = OperatorSpecCompilerAccess::source(spec, index);
      const auto* resolved = find_readiness(
          readiness, owner, static_cast<std::uint32_t>(index), metadata.kind, metadata.name);
      if (resolved == nullptr || !resolved->source_id) {
        return replay_error("TEMPORAL_TRIGGER_SOURCE_UNRESOLVED");
      }
      bind_source(metadata, resolved->source_id, plan_generation);
    }

    Contract contract;
    try {
      contract = created.value()->contract();
    } catch (...) {
      capture.capture_exception();
      return failed("TEMPORAL_CONTRACT_REPLAY_FAILED", DiagnosticOperation::kContract);
    }
    const std::size_t connected_count = connected_input_count(snapshot, owner);
    // Existing operators that have no temporal authoring and no connected input are inert in
    // baseline. Inputful contracts may omit the trigger and normalize to OnAnyInput, including when
    // they customize selectors, missing policy, or deadline policy.
    if (connected_count == 0U && ContractTestAccess::empty(contract)) {
      continue;
    }
    auto normalized = ContractTestAccess::normalize(contract, connected_count);
    if (!normalized) {
      return failed(
          "TEMPORAL_CONTRACT_INVALID", DiagnosticOperation::kContract, &normalized.error());
    }
    replay.contract = std::move(*normalized);
    result.push_back(std::move(replay));
  }
  std::sort(result.begin(), result.end(), [](const auto& left, const auto& right) {
    return left.owner.value < right.owner.value;
  });
  return result;
}

// Keep every temporal record family and cross-family join in one fail-closed pass.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
TemporalPlanLoweringResult lower_temporal_plan(
    std::span<const GraphCompilerAccess::TemporalReplay> normalized_contracts,
    const GraphSnapshotData& snapshot, PlanGenerationId plan_generation,
    std::span<const ClockPlanRecord> clocks, std::span<const ClockConversionRecord> conversions,
    std::vector<ReadinessPlanRecord>& readiness,
    std::span<const std::optional<Profile>> effective_profiles,
    const ParameterCompileOptions& parameter_options,
    const RuntimePackageCapabilityView& package_capabilities) {
  TemporalPlanLoweringResult result;
  // Validate permission even when wired, or when an inert Operator has no temporal replay.
  // A reusable Operator must not advertise optionality that depends on the application's wiring.
  for (const auto& port : snapshot.ports) {
    if (!port.optional) {
      continue;
    }
    const auto replay = std::find_if(normalized_contracts.begin(),
                                     normalized_contracts.end(),
                                     [&](const auto& item) { return item.owner == port.owner; });
    const bool eligible =
        port.owner_kind == PortOwnerKind::kOperator && port.direction == PortDirection::kInput &&
        replay != normalized_contracts.end() &&
        std::any_of(
            replay->contract.inputs.begin(), replay->contract.inputs.end(), [&](const auto& input) {
              return input.port_name == port.name &&
                     optional_safe_default_contract(replay->contract, input);
            });
    if (!eligible) {
      add_error(result, "TEMPORAL_INPUT_OPTIONAL_INVALID", port.owner, PortId{port.id.value});
      attach_optional_input_context(
          result.errors.back(),
          snapshot,
          port,
          optional_contract_violation(
              replay == normalized_contracts.end() ? nullptr : &replay->contract, port.name),
          "use a compute-only single-sample Exact selector with explicit SafeDefault; "
          "otherwise remove .optional() and wire the required input");
    }
  }
  if (!result.accepted()) {
    return result;
  }

  std::uint32_t next_temporal_source = first_unassigned_source_id(readiness);

  for (const auto& replay : normalized_contracts) {
    const VertexRecord* vertex = find_vertex(snapshot, replay.owner);
    if (vertex == nullptr) {
      add_error(result, "TEMPORAL_CONTRACT_OWNER_UNRESOLVED", replay.owner);
      continue;
    }
    const auto profile = effective_profile_for_vertex(effective_profiles, replay.owner);
    if (!profile) {
      add_error(result, "TEMPORAL_PROFILE_UNRESOLVED", replay.owner);
      continue;
    }

    auto contract = replay.contract;
    if (contract.compute_calls && !completion_placement_supported(replay)) {
      add_error(result, "OPERATOR_COMPLETION_PLACEMENT_UNSUPPORTED", replay.owner);
      continue;
    }
    const ClockKey reference_key = (contract.trigger.kind == TemporalTriggerKind::kClock ||
                                    contract.trigger.kind == TemporalTriggerKind::kWindow ||
                                    (contract.trigger.kind == TemporalTriggerKind::kPredicate &&
                                     contract.trigger.predicate_clock_source))
                                       ? contract.trigger.clock
                                       : ClockKey{};
    const ClockId reference_clock = resolve_clock(snapshot, clocks, vertex->graph, reference_key);
    const bool readiness_trigger = contract.trigger.kind == TemporalTriggerKind::kReadable ||
                                   contract.trigger.kind == TemporalTriggerKind::kNotified ||
                                   contract.trigger.kind == TemporalTriggerKind::kDoorbell ||
                                   contract.trigger.kind == TemporalTriggerKind::kSpin;
    auto readiness_source = readiness.end();
    if (readiness_trigger) {
      readiness_source =
          std::find_if(readiness.begin(), readiness.end(), [&](const auto& candidate) {
            return candidate.source_id == contract.trigger.source_id &&
                   candidate.owner == replay.owner;
          });
      if (readiness_source == readiness.end()) {
        add_error(result, "TEMPORAL_TRIGGER_SOURCE_UNRESOLVED", replay.owner);
        continue;
      }
    }
    const ClockId effective_reference_clock =
        readiness_trigger ? readiness_source->resolved_clock : reference_clock;
    const bool readiness_selection_requires_clock =
        readiness_trigger &&
        std::any_of(contract.inputs.begin(), contract.inputs.end(), [](const auto& input) {
          return requires_capture_time(input.selector);
        });
    const bool requires_resolved_clock = contract.trigger.kind == TemporalTriggerKind::kAnyInput ||
                                         contract.trigger.kind == TemporalTriggerKind::kEach ||
                                         contract.trigger.kind == TemporalTriggerKind::kClock ||
                                         contract.trigger.kind == TemporalTriggerKind::kAligned ||
                                         contract.trigger.kind == TemporalTriggerKind::kWindow ||
                                         contract.trigger.kind == TemporalTriggerKind::kPredicate ||
                                         readiness_selection_requires_clock;
    if (requires_resolved_clock && !effective_reference_clock.valid()) {
      // Clock-catalog failures publish no partial catalog. An unbound import (including one
      // farther up an explicit binding chain) is diagnosed at its exact boundary by that pass;
      // do not tell already configured graphs to create another clock as a consequence.
      const auto authored = resolve_authored_clock(snapshot.graphs,
                                                   snapshot.clocks,
                                                   snapshot.clock_imports,
                                                   snapshot.clock_bindings,
                                                   vertex->graph,
                                                   reference_key);
      if (authored.status == AuthoredClockResolutionStatus::kNotFound) {
        if (reference_key.is_default()) {
          add_missing_default_clock_error(result, snapshot, vertex->graph);
        } else {
          const auto* graph = find_graph(snapshot, vertex->graph);
          const std::string scope = graph == nullptr ? "_root" : canonical_graph_path(*graph);
          const std::string clock = "clock \"" + std::string(reference_key.name()) + "\"";
          const std::string required =
              graph != nullptr && graph->parent
                  ? "Declare " + clock + " using import_clock() in graph \"" + scope +
                        "\" and bind it using bind_clock() in the immediate parent"
                  : "Declare " + clock + " using add_clock() in graph \"" + scope + "\"";
          add_undeclared_temporal_clock_error(
              result,
              snapshot,
              *vertex,
              {},
              "/vertices/" + std::to_string(vertex_index(snapshot, replay.owner)) +
                  "/temporal_contract/trigger/clock",
              required,
              "The trigger references undeclared " + clock + " in graph \"" + scope + "\"");
        }
      }
      continue;
    }

    TemporalContractLoweringContext context{
        .operator_name = replay.operator_name,
        .provenance = {.source = "operator.contract", .declaration_order = 0U},
        .connected_input_count =
            static_cast<std::uint32_t>(connected_input_count(snapshot, replay.owner)),
        .trigger_reference = {.resolved = true,
                              .clock = {.is_default = reference_key.is_default(),
                                        .name = std::string(reference_key.name())}},
    };

    if (readiness_trigger) {
      context.source_provenance = "operator.setup.readiness-source";
    }

    const auto connected = connected_inputs(snapshot, replay.owner);
    if (contract.trigger.kind == TemporalTriggerKind::kAnyInput) {
      for (std::size_t index = 0; index < connected.size(); ++index) {
        const PortRecord* port = find_port(snapshot, GraphPortId{connected[index].value()});
        if (port != nullptr) {
          context.trigger_pins.push_back({
              .resolved = true,
              .input_name = port->name,
              .retained_pin_capacity = 1U,
              .provenance = {.source = "trigger.accepted-arrival",
                             .declaration_order = static_cast<std::uint32_t>(index)},
          });
        }
      }
    } else if (contract.trigger.kind == TemporalTriggerKind::kEach) {
      context.trigger_pins.push_back({
          .resolved = true,
          .input_name = contract.trigger.port_name,
          .retained_pin_capacity = 1U,
          .provenance = {.source = "trigger.accepted-arrival", .declaration_order = 0U},
      });
    }

    bool selection_error = false;
    std::vector<std::uint64_t> selection_conversion_ids;
    selection_conversion_ids.reserve(contract.inputs.size());
    for (std::size_t index = 0; index < contract.inputs.size(); ++index) {
      auto& input = contract.inputs[index];
      const auto* replayed_port = replay_port(replay, input.port_name);
      if (replayed_port == nullptr || replayed_port->local_type_token != input.local_type_token) {
        add_error(result,
                  "TEMPORAL_INPUT_PORT_INVALID",
                  replay.owner,
                  replayed_port == nullptr ? PortId{} : replayed_port->id);
        selection_error = true;
        break;
      }
      const PortRecord* port = find_port(snapshot, GraphPortId{replayed_port->id.value()});
      if (port == nullptr) {
        add_error(result, "TEMPORAL_INPUT_PORT_INVALID", replay.owner, replayed_port->id);
        selection_error = true;
        break;
      }
      if (input.max_age_dependency) {
        ParameterDerivedTermPlanRecord term;
        std::int64_t resolved_ns{};
        if (input.selector != TemporalSelectorKind::kLatestWithin ||
            !resolve_duration_dependency(replay,
                                         input.max_age_dependency,
                                         ParameterDerivedTermKind::kLatestWithinMaxAge,
                                         replayed_port->id,
                                         profile,
                                         parameter_options,
                                         term,
                                         resolved_ns)) {
          add_error(result, "PARAMETER_DERIVED_TERM_INVALID", replay.owner, replayed_port->id);
          selection_error = true;
          break;
        }
        term.owner.plan_generation = plan_generation;
        input.max_age = std::chrono::nanoseconds{resolved_ns};
        input.max_age_dependency = {};
        result.parameter_derived_terms.push_back(std::move(term));
      }
      const EdgeRecord* edge = incoming_edge(snapshot, port->id);
      if (edge == nullptr) {
        if (!port->optional || !optional_safe_default_contract(contract, input)) {
          add_error(result, "TEMPORAL_INPUT_UNCONNECTED", replay.owner, replayed_port->id);
          attach_optional_input_context(
              result.errors.back(),
              snapshot,
              *port,
              "required input has no graph connection",
              optional_safe_default_contract(contract, input)
                  ? "connect this input; its operator may instead permit intentional absence "
                    "with .optional() for this compute-only Exact + SafeDefault contract"
                  : "connect this required input; its current contract does not permit "
                    "optional absence");
          selection_error = true;
          break;
        }
        // preserve one ordinary bounded selector record for an explicitly
        // degraded optional input without fabricating a graph edge or transport endpoint.
        context.edges.push_back({
            .resolved = true,
            .input_name = input.port_name,
            .reference = {.resolved = true,
                          .clock = {.is_default = reference_key.is_default(),
                                    .name = std::string(reference_key.name())},
                          .conversion = TemporalRecordConversionKind::kSameClock,
                          .conversion_record_id = {}},
            .cache_capacity = 1U,
            .cache_capacity_origin = TemporalRecordBoundOrigin::kSingleSample,
            .drain_all_max_samples = 0U,
            .provenance = {.source = "contract.unconnected-safe-default",
                           .declaration_order = static_cast<std::uint32_t>(index)},
        });
        selection_conversion_ids.push_back(0U);
        add_warning(
            result, "TEMPORAL_INPUT_UNCONNECTED_SAFE_DEFAULT", replay.owner, replayed_port->id);
        attach_optional_input_context(
            result.warnings.back(),
            snapshot,
            *port,
            "input is intentionally unwired",
            "handle SafeDefault/NotConnected with an operator-owned fallback; "
            "connect the input when its data is required");
        continue;
      }
      // REQ: COMPILE-017 - freeze the final retained geometry before deriving
      // selector cardinality or selected-sample bounds.
      std::uint32_t depth = effective_queue_depth(*port, *edge);
      if (depth == 0U) {
        add_error(result, "TEMPORAL_RETENTION_UNBOUNDED", replay.owner, replayed_port->id);
        selection_error = true;
        break;
      }
      if (input.selector == TemporalSelectorKind::kBracketedWindow) {
        if (effective_queue_policy(*edge) != QueuePolicy::kBlocking) {
          add_error(
              result, "TEMPORAL_WINDOW_QUEUE_POLICY_DROP_OLDEST", replay.owner, replayed_port->id);
          selection_error = true;
          break;
        }
        const std::uint64_t minimum =
            static_cast<std::uint64_t>(input.bracket.lookbehind) + input.bracket.lookahead + 1U;
        const std::uint64_t pending_windows = contract.trigger.capacity;
        const std::uint64_t required_complete_history =
            static_cast<std::uint64_t>(input.bracket.max_samples) * pending_windows;
        const bool invalid_geometry =
            input.bracket.max_samples < minimum || pending_windows == 0U ||
            required_complete_history > std::numeric_limits<std::uint32_t>::max();
        const bool explicit_depth_insufficient = !invalid_geometry &&
                                                 has_explicit_queue_depth(*port, *edge) &&
                                                 depth < required_complete_history;
        if (invalid_geometry || explicit_depth_insufficient) {
          add_error(result, "TEMPORAL_RETENTION_INSUFFICIENT", replay.owner, replayed_port->id);
          selection_error = true;
          break;
        }
        if (!has_explicit_queue_depth(*port, *edge)) {
          depth = static_cast<std::uint32_t>(required_complete_history);
        }
      }
      // An external import whose catalog makes capture time optional cannot
      // satisfy an event-time selector. Internal producers are different:
      // compilation can derive a required output-binding policy for them, but
      // an absent boundary contract cannot be strengthened after peer attach.
      const auto* imported_topic = imported_source_topic(snapshot, *edge);
      if (requires_capture_time(input.selector) && imported_topic != nullptr &&
          !imported_topic->capture_time) {
        add_error(result, "TEMPORAL_CAPTURE_TIME_REQUIRED", replay.owner, replayed_port->id);
        selection_error = true;
        break;
      }
      const std::optional<ClockId> explicit_capture_clock =
          explicit_source_capture_clock(snapshot, clocks, *edge);
      if (explicit_capture_clock && !explicit_capture_clock->valid()) {
        add_undeclared_temporal_clock_error(
            result,
            snapshot,
            *vertex,
            replayed_port->id,
            "/vertices/" + std::to_string(vertex_index(snapshot, replay.owner)) +
                "/temporal_contract/inputs/" + std::to_string(index) + "/capture_clock",
            "one declared capture-time clock visible to the imported endpoint",
            "the authored capture-time clock did not resolve; declare or bind that clock");
        selection_error = true;
        break;
      }
      const ClockId capture_clock = explicit_capture_clock.value_or(effective_reference_clock);
      const ClockConversionRecord* conversion = nullptr;
      if (capture_clock.valid() != effective_reference_clock.valid()) {
        add_error(result, "TEMPORAL_CLOCK_CONVERSION_UNAVAILABLE", replay.owner, replayed_port->id);
        selection_error = true;
        break;
      }
      if (capture_clock.valid() && capture_clock != effective_reference_clock) {
        conversion = find_conversion(conversions, capture_clock, effective_reference_clock);
        if (conversion == nullptr) {
          add_error(
              result, "TEMPORAL_CLOCK_CONVERSION_UNAVAILABLE", replay.owner, replayed_port->id);
          selection_error = true;
          break;
        }
      }
      context.edges.push_back({
          .resolved = true,
          .input_name = input.port_name,
          .reference = {.resolved = true,
                        .clock = {.is_default = reference_key.is_default(),
                                  .name = std::string(reference_key.name())},
                        .conversion = conversion == nullptr
                                          ? TemporalRecordConversionKind::kSameClock
                                          : TemporalRecordConversionKind::kDirectAffine,
                        .conversion_record_id =
                            conversion == nullptr
                                ? std::string{}
                                : "clock-conversion/" + std::to_string(conversion->id)},
          .cache_capacity = depth,
          .cache_capacity_origin = TemporalRecordBoundOrigin::kResolvedRetention,
          .drain_all_max_samples = input.selector == TemporalSelectorKind::kDrainAll ? depth : 0U,
          .provenance = {.source = "graph.retained-geometry",
                         .declaration_order = static_cast<std::uint32_t>(index)},
      });
      selection_conversion_ids.push_back(conversion == nullptr ? 0U : conversion->id);
    }
    if (selection_error) {
      continue;
    }

    if (contract.deadline_dependency) {
      ParameterDerivedTermPlanRecord term;
      std::int64_t resolved_ns{};
      if (!resolve_duration_dependency(replay,
                                       contract.deadline_dependency,
                                       ParameterDerivedTermKind::kContractDeadline,
                                       {},
                                       profile,
                                       parameter_options,
                                       term,
                                       resolved_ns)) {
        add_error(result, "PARAMETER_DERIVED_TERM_INVALID", replay.owner);
        continue;
      }
      term.owner.plan_generation = plan_generation;
      contract.deadline = std::chrono::nanoseconds{resolved_ns};
      contract.deadline_dependency = {};
      result.parameter_derived_terms.push_back(std::move(term));
    }

    const ClockPlanRecord* window_wait_clock = nullptr;
    if (contract.trigger.kind == TemporalTriggerKind::kWindow) {
      window_wait_clock = find_clock(clocks, reference_clock);
      const bool direct_capability =
          window_wait_clock != nullptr && window_wait_clock->can_read_now &&
          window_wait_clock->can_wait_until && window_wait_clock->steady &&
          window_wait_clock->independent_progress && window_wait_clock->finite_deadline &&
          !window_wait_clock->capability_evidence_id.empty();
      const bool driven_capability =
          window_wait_clock != nullptr && window_wait_clock->can_read_now &&
          window_wait_clock->can_wait_until && window_wait_clock->steady &&
          window_wait_clock->external_driver_admitted && window_wait_clock->finite_deadline &&
          !window_wait_clock->capability_evidence_id.empty();
      if (!direct_capability && !driven_capability) {
        add_error(result, "TEMPORAL_WINDOW_WAIT_CLOCK_CAPABILITY_UNAVAILABLE", replay.owner);
        continue;
      }
    }

    std::optional<PredicateCallablePlanRecord> callable_plan;
    if (contract.trigger.kind == TemporalTriggerKind::kPredicate) {
      const auto kind = contract.trigger.predicate.callable_kind();
      // REQ: COMPILE-016 - configured predicates require a qualified canonical semantic codec.
      // Raw object bytes, padding, size, and alignment never authorize plan identity.
      if (contract.trigger.predicate_config_size != 0U &&
          !contract.trigger.predicate_config_identity_available) {
        add_error(result, "PREDICATE_CONFIG_IDENTITY_UNAVAILABLE", replay.owner);
        continue;
      }
      const std::string predicate_id = "predicate/" + std::to_string(replay.operator_id) + "/" +
                                       std::string(contract.trigger.predicate.name());
      StableHashBuilder fingerprint_hash;
      fingerprint_hash.add("holoscan.temporal.predicate-callable.v1");
      if (contract.trigger.predicate_semantic_version) {
        fingerprint_hash.add(contract.trigger.predicate.name());
        fingerprint_hash.add(contract.trigger.predicate_semantic_version->major);
        fingerprint_hash.add(contract.trigger.predicate_semantic_version->minor);
        fingerprint_hash.add(contract.trigger.predicate_semantic_version->patch);
      } else {
        fingerprint_hash.add(predicate_id);
      }
      fingerprint_hash.add(kind);
      fingerprint_hash.add_canonical_size(contract.trigger.predicate_callable_size);
      fingerprint_hash.add_canonical_size(contract.trigger.predicate_callable_alignment);
      fingerprint_hash.add(contract.trigger.predicate_callable_noexcept);
      fingerprint_hash.add(package_capabilities.manifest().package_id);
      fingerprint_hash.add(package_capabilities.manifest().build_id);
      const std::string fingerprint =
          lowercase_hex_text(fingerprint_hash.finish().value, HexPrefix::kSha256);
      const auto disposition =
          callable_disposition(kind, contract.trigger.predicate_callable_noexcept, *profile);
      callable_plan = make_predicate_callable_record(predicate_id,
                                                     contract.trigger,
                                                     fingerprint,
                                                     package_capabilities.manifest().build_id,
                                                     *profile,
                                                     disposition,
                                                     plan_generation);
      if (disposition == PredicateCallableDisposition::kRejected) {
        add_error(result, "PREDICATE_CALLABLE_PROFILE_UNSUPPORTED", replay.owner);
        result.predicate_callables.push_back(std::move(*callable_plan));
        continue;
      }
      if (disposition == PredicateCallableDisposition::kAdmittedWithWarning) {
        add_warning(result, "PREDICATE_CALLABLE_TYPE_ERASED", replay.owner);
      }
      // PredicateKey supplies semantic naming but does not attest executable
      // equivalence. Cross-build recomputation is therefore admitted only when
      // the independently generated target-package manifest names this exact
      // key/version and repeats the compiler-derived build fingerprint.
      const auto* recomputation_offer = contract.trigger.predicate_semantic_version
                                            ? package_capabilities.find_predicate_recomputation(
                                                  contract.trigger.predicate.name(),
                                                  *contract.trigger.predicate_semantic_version,
                                                  fingerprint)
                                            : nullptr;
      context.predicate = TemporalResolvedPredicateFacts{
          .resolved = true,
          .record_id = predicate_id,
          .implementation_fingerprint = fingerprint,
          .build_evidence = callable_plan->build_evidence,
          .no_dynamic_allocation = false,
          .bounded_call = false,
          .owned_lifetime = kind != PredicateCallableKind::kPythonBoundMethod,
          .self_rebind = kind == PredicateCallableKind::kBoundMember,
          .state_max_bytes = contract.trigger.predicate_state_size,
          .state_schema = contract.trigger.predicate_state_size == 0U
                              ? std::string{}
                              : "cpp.predicate-state/size-" +
                                    std::to_string(contract.trigger.predicate_state_size),
          .config_max_bytes = contract.trigger.predicate_config_size,
          .config_schema = contract.trigger.predicate_config_schema,
          .config_digest = contract.trigger.predicate_config_digest,
          .replay = recomputation_offer != nullptr
                        ? TemporalPredicateReplayKind::kCrossBuildRecomputation
                        : TemporalPredicateReplayKind::kDecisionReplay,
          .replay_evidence =
              recomputation_offer != nullptr ? recomputation_offer->evidence_id : std::string{},
      };
    }

    auto logical = lower_temporal_contract_records(contract, context);
    if (!logical) {
      add_error(result, "TEMPORAL_CONTRACT_INVALID", replay.owner);
      continue;
    }

    const bool predicate_clock =
        logical->contract.predicate &&
        logical->contract.predicate->source == TemporalPredicateSourceKind::kClock;
    const std::int64_t trigger_period_ns =
        predicate_clock ? logical->contract.predicate->period_ns : logical->contract.period_ns;
    const std::optional<TimePoint>& trigger_phase =
        predicate_clock ? logical->contract.predicate->phase : logical->contract.phase;

    TemporalTriggerPlanRecord trigger{
        .owner = OperatorRef{.id = replay.operator_id, .plan_generation = plan_generation},
        .kind = logical->contract.trigger,
        .origin = logical->contract.origin,
        .declared_inputs = {},
        .presence_guaranteed_inputs = presence_set(replay, contract),
        .trigger_clock =
            (logical->contract.trigger == TemporalRecordTriggerKind::kClock ||
             logical->contract.trigger == TemporalRecordTriggerKind::kWindow || predicate_clock)
                ? reference_clock
                : ClockId{},
        .reference_clock = reference_clock,
        .period_ns = trigger_period_ns,
        .phase = trigger_phase,
        .bounded_state_capacity = 1U,
        .witness_identity = "holoscan.temporal.witness.v1/" +
                            std::to_string(static_cast<unsigned>(logical->contract.trigger)),
        .replay_identity = "holoscan.temporal.replay.v1/decision",
        .deadline_ns = logical->contract.deadline_ns,
        .violation = logical->contract.violation,
    };
    for (const auto& input : logical->edges) {
      const auto* port = replay_port(replay, input.input_name);
      if (port != nullptr) {
        trigger.declared_inputs.push_back(port->id);
      }
    }
    if (!logical->contract.trigger_input.empty()) {
      const auto* port = replay_port(replay, logical->contract.trigger_input);
      if (port != nullptr) {
        trigger.trigger_input = port->id;
      }
    }
    if (logical->contract.trigger_source) {
      trigger.readiness_source = logical->contract.trigger_source;
      if (readiness_source == readiness.end() ||
          readiness_source->source_id != trigger.readiness_source) {
        add_error(result, "TEMPORAL_TRIGGER_SOURCE_UNRESOLVED", replay.owner);
        continue;
      }
      readiness_source->referenced = true;
      readiness_source->materialized = true;
      trigger.reference_clock = readiness_source->resolved_clock;
      trigger.bounded_state_capacity = readiness_source->accepted_capacity;
    } else {
      // Every accepted trigger owns one logical candidate-source identity, including
      // Channel-backed input triggers. Input sources are materialized from the physical
      // subscriber plan by Runtime and therefore deliberately have no fabricated
      // ReadinessPlanRecord.
      const std::optional<SourceId> source = allocate_source_id(next_temporal_source);
      if (!source) {
        add_error(result, "READINESS_SOURCE_ID_EXHAUSTED", replay.owner);
        continue;
      }
      trigger.readiness_source = *source;
      if (logical->contract.window) {
        trigger.bounded_state_capacity = logical->contract.window->max_pending_windows;
      } else if (!logical->contract.trigger_pins.empty()) {
        trigger.bounded_state_capacity =
            static_cast<std::uint32_t>(logical->contract.trigger_pins.size());
      }
    }

    const std::size_t selection_begin = result.input_selections.size();
    for (std::size_t index = 0; index < logical->edges.size(); ++index) {
      const auto& edge = logical->edges[index];
      const auto* port = replay_port(replay, edge.input_name);
      if (port == nullptr) {
        continue;
      }
      result.input_selections.push_back({
          .input = port->id,
          .selector = selector_kind(edge.selector),
          .cardinality = cardinality(edge.cardinality),
          .max_selected_samples = edge.max_selected_samples,
          .max_selected_samples_origin = selected_origin(edge.max_selected_origin),
          .batch_selection_mode = batch_mode(edge.batch_mode),
          .cache_capacity = edge.cache_capacity,
          .reference_clock = trigger.reference_clock,
          .conversion_id =
              index < selection_conversion_ids.size() ? selection_conversion_ids[index] : 0U,
          .missing = edge.missing,
          .max_age_ns = edge.max_age_ns,
          .reuse_max_age_ns = edge.reuse_max_age_ns,
          .lookbehind = edge.lookbehind,
          .lookahead = edge.lookahead,
          .unwired_safe_default = incoming_edge(snapshot, GraphPortId{port->id.value()}) == nullptr,
      });
      if (edge.selector == TemporalRecordSelectorKind::kBracketedWindow) {
        const PortRecord* selected_port = find_port(snapshot, GraphPortId{port->id.value()});
        const EdgeRecord* selected_edge =
            selected_port == nullptr ? nullptr : incoming_edge(snapshot, selected_port->id);
        const PortRecord* producer =
            selected_edge == nullptr ? nullptr : find_port(snapshot, selected_edge->source);
        const std::uint32_t producer_burst =
            producer == nullptr ? 1U
                                : producer->graph_override.max_emits_per_compute.value_or(
                                      producer->setup.max_emits_per_compute.value_or(1U));
        result.bracketed_windows.push_back({
            .owner = trigger.owner,
            .input = port->id,
            .anchor = trigger.trigger_input,
            .lookbehind = edge.lookbehind,
            .lookahead = edge.lookahead,
            .max_samples = edge.max_selected_samples,
            .event_clock = trigger.reference_clock,
            .conversion_id =
                index < selection_conversion_ids.size() ? selection_conversion_ids[index] : 0U,
            .queue_capacity = edge.cache_capacity,
            .cache_capacity = edge.cache_capacity,
            .producer_max_emits_per_compute = producer_burst,
            .retained_overlap_capacity = edge.cache_capacity,
            .payload_identity = selected_port == nullptr ? PayloadIdentityRecord{}
                                                         : selected_port->payload_identity,
            .missing = edge.missing,
            .insufficient_history_witness_identity =
                "holoscan.temporal.window-insufficient-history.v1",
            .timeout_witness_identity = "holoscan.temporal.window-timeout.v1",
            .sample_limit_witness_identity = "holoscan.temporal.window-sample-limit.v1",
        });
      }
    }

    const auto current_selections =
        std::span<const InputSelectionPlanRecord>{result.input_selections.data() + selection_begin,
                                                  result.input_selections.size() - selection_begin};

    if (logical->contract.aligned) {
      const auto& aligned = *logical->contract.aligned;
      AlignedTriggerPlanRecord planned{
          .owner = trigger.owner,
          .inclusive_window_ns = aligned.inclusive_window_ns,
          .comparison_clock = trigger.reference_clock,
      };
      planned.closure_proofs = {
          AlignedClosureProofKind::kStreamAdvancement,
          AlignedClosureProofKind::kLossWitness,
          AlignedClosureProofKind::kPrivateWatermark,
          AlignedClosureProofKind::kEndOfStream,
          AlignedClosureProofKind::kProducerEpochClosed,
          AlignedClosureProofKind::kTemporalClosure,
      };
      planned.members.reserve(aligned.member_inputs.size());
      bool aligned_invalid = !trigger.reference_clock.valid() || aligned.member_inputs.empty();
      for (std::size_t ordinal = 0U; ordinal < aligned.member_inputs.size(); ++ordinal) {
        const auto* member = replay_port(replay, aligned.member_inputs[ordinal]);
        const auto selection =
            member == nullptr
                ? current_selections.end()
                : std::find_if(current_selections.begin(),
                               current_selections.end(),
                               [&](const auto& item) { return item.input == member->id; });
        if (member == nullptr || selection == current_selections.end() ||
            selection->selector != InputSelectorKind::kExact ||
            selection->cardinality != InputSelectionCardinality::kSingleSample ||
            selection->cache_capacity == 0U) {
          add_error(result,
                    "TEMPORAL_ALIGNED_MEMBER_INVALID",
                    replay.owner,
                    member == nullptr ? PortId{} : member->id);
          aligned_invalid = true;
          continue;
        }
        const ClockConversionRecord* conversion =
            selection->conversion_id == 0U ? nullptr
                                           : find_conversion(conversions, selection->conversion_id);
        if (selection->conversion_id != 0U && conversion == nullptr) {
          add_error(result, "TEMPORAL_CLOCK_CONVERSION_UNAVAILABLE", replay.owner, member->id);
          aligned_invalid = true;
          continue;
        }
        planned.members.push_back(AlignedMemberPlanRecord{
            .input = member->id,
            .member_ordinal = static_cast<std::uint32_t>(ordinal),
            .event_clock = conversion == nullptr ? trigger.reference_clock : conversion->from,
            .conversion_id = selection->conversion_id,
            .queue_capacity = selection->cache_capacity,
            .cache_capacity = selection->cache_capacity,
        });
      }
      if (aligned_invalid || planned.members.size() != aligned.member_inputs.size()) {
        continue;
      }
      result.aligned_triggers.push_back(std::move(planned));
    }

    if (logical->contract.trigger == TemporalRecordTriggerKind::kClock || predicate_clock) {
      const std::int64_t period_ns =
          predicate_clock ? logical->contract.predicate->period_ns : logical->contract.period_ns;
      const std::optional<TimePoint>& phase =
          predicate_clock ? logical->contract.predicate->phase : logical->contract.phase;
      const ClockConversionRecord* phase_conversion = nullptr;
      if (phase.has_value() && phase->clock_id != trigger.trigger_clock) {
        phase_conversion = find_conversion(conversions, phase->clock_id, trigger.trigger_clock);
      }
      if (!trigger.trigger_clock.valid() || period_ns <= 0) {
        add_error(result, "TEMPORAL_CLOCK_TRIGGER_INVALID", replay.owner);
        continue;
      }
      if (phase.has_value() && find_clock(clocks, phase->clock_id) == nullptr) {
        add_undeclared_temporal_clock_error(
            result,
            snapshot,
            *vertex,
            {},
            "/vertices/" + std::to_string(vertex_index(snapshot, replay.owner)) +
                "/temporal_contract/trigger/phase/clock_id",
            "one declared clock for the explicit absolute trigger phase",
            "explicit phase references undeclared ClockId " +
                std::to_string(phase->clock_id.value));
        continue;
      }
      if (phase.has_value() && phase->clock_id != trigger.trigger_clock &&
          phase_conversion == nullptr) {
        add_error(result, "TEMPORAL_CLOCK_CONVERSION_UNAVAILABLE", replay.owner);
        continue;
      }
      result.clock_triggers.push_back(ClockTriggerPlanRecord{
          .owner = trigger.owner,
          .clock = trigger.trigger_clock,
          .period_ns = period_ns,
          .phase_origin = phase.has_value() ? ClockPhaseOrigin::kExplicitAbsolute
                                            : ClockPhaseOrigin::kCapturedAtSuccessfulStart,
          .explicit_phase = phase,
          .phase_conversion_id = phase_conversion == nullptr ? 0U : phase_conversion->id,
      });
    }

    if (logical->contract.deadline_ns && logical->contract.violation) {
      const DeadlineSourceCoordinateKind source_coordinate =
          deadline_source_coordinate(logical->contract);
      SourceClockCoordinate deadline_coordinate{.source_clock = trigger.reference_clock};
      std::vector<PortId> source_inputs;
      if (source_coordinate == DeadlineSourceCoordinateKind::kTriggerSample) {
        if (trigger.kind == TemporalRecordTriggerKind::kAnyInput) {
          source_inputs.reserve(logical->contract.trigger_pins.size());
          for (const auto& pin : logical->contract.trigger_pins) {
            const auto* port = replay_port(replay, pin.input_name);
            if (port != nullptr) {
              source_inputs.push_back(port->id);
            }
          }
        } else if (trigger.trigger_input) {
          source_inputs.push_back(trigger.trigger_input);
        }
      } else if (source_coordinate == DeadlineSourceCoordinateKind::kPredicateDriver &&
                 trigger.trigger_input) {
        source_inputs.push_back(trigger.trigger_input);
      }

      bool deadline_invalid = !deadline_coordinate.source_clock.valid();
      std::optional<SourceClockCoordinate> common_input_coordinate;
      for (const PortId input : source_inputs) {
        const auto coordinate =
            input_source_coordinate(snapshot, clocks, conversions, input, trigger.reference_clock);
        if (!coordinate) {
          add_error(result, "TEMPORAL_CLOCK_CONVERSION_UNAVAILABLE", replay.owner, input);
          deadline_invalid = true;
          break;
        }
        if (common_input_coordinate && *common_input_coordinate != *coordinate) {
          add_error(result, "TEMPORAL_DEADLINE_SOURCE_AMBIGUOUS", replay.owner, input);
          deadline_invalid = true;
          break;
        }
        common_input_coordinate = coordinate;
      }
      if ((source_coordinate == DeadlineSourceCoordinateKind::kTriggerSample ||
           source_coordinate == DeadlineSourceCoordinateKind::kPredicateDriver) &&
          source_inputs.empty()) {
        add_error(result, "TEMPORAL_DEADLINE_SOURCE_UNRESOLVED", replay.owner);
        deadline_invalid = true;
      }
      if (deadline_invalid) {
        continue;
      }
      if (common_input_coordinate) {
        deadline_coordinate = *common_input_coordinate;
      }
      trigger.conversion_id = deadline_coordinate.conversion_id;
      result.deadlines.push_back(DeadlinePlanRecord{
          .owner = trigger.owner,
          .budget_ns = *logical->contract.deadline_ns,
          .action = *logical->contract.violation,
          .source_coordinate = source_coordinate,
          .source_clock = deadline_coordinate.source_clock,
          .conversion_id = deadline_coordinate.conversion_id,
          .callback_count_if_late = deadline_callback_count(*logical->contract.violation),
      });
    }

    if (logical->contract.trigger == TemporalRecordTriggerKind::kReadable) {
      const auto source = std::find_if(readiness.begin(), readiness.end(), [&](const auto& item) {
        return item.owner == replay.owner && item.source_id == trigger.readiness_source &&
               item.kind == ReadinessSourceKind::kReadable;
      });
      if (source == readiness.end() || source->accepted_capacity != 1U ||
          !source->resolved_clock.valid()) {
        add_error(result, "TEMPORAL_READABLE_SOURCE_INVALID", replay.owner);
        continue;
      }
      result.readable_triggers.push_back(ReadableTriggerPlanRecord{
          .owner = trigger.owner,
          .source = source->source_id,
          .plan_generation = plan_generation,
          .witness_clock = source->resolved_clock,
          .initial_source_generation = source->initial_source_generation,
          .initial_source_sequence = source->initial_source_sequence,
          .effective_capacity = source->accepted_capacity,
          .max_outstanding_candidates = source->accepted_capacity,
      });
    }

    if (logical->contract.trigger != TemporalRecordTriggerKind::kPredicate) {
      result.missing_aggregates.push_back(make_missing_aggregate(
          trigger.owner, MissingSelectionPhase::kActivation, current_selections));
    }

    if (logical->contract.window) {
      if (window_wait_clock == nullptr) {
        add_error(result, "TEMPORAL_WINDOW_WAIT_CLOCK_CAPABILITY_UNAVAILABLE", replay.owner);
        continue;
      }
      WindowTriggerPlanRecord window{
          .owner = trigger.owner,
          .anchor = trigger.trigger_input,
          .wait_clock = trigger.trigger_clock,
          .capability_evidence_id = window_wait_clock->capability_evidence_id,
          .external_driver_used = !window_wait_clock->independent_progress,
          .max_wait_ns = logical->contract.window->max_wait_ns,
          .max_pending_windows = logical->contract.window->max_pending_windows,
          .presence_guaranteed_inputs = {trigger.trigger_input},
          .reset_on_producer_epoch = logical->contract.window->reset_on_epoch_change,
          .reset_on_timestamp_regression = logical->contract.window->reset_on_timestamp_regression,
          .primed_witness_identity = "holoscan.temporal.window-primed.v1",
          .open_witness_identity = "holoscan.temporal.window-open.v1",
          .close_witness_identity = "holoscan.temporal.window-close.v1",
          .timeout_witness_identity = "holoscan.temporal.window-timeout.v1",
          .pending_limit_witness_identity = "holoscan.temporal.window-pending-limit.v1",
          .end_of_stream_witness_identity = "holoscan.temporal.window-end-of-stream.v1",
          .timestamp_regression_witness_identity =
              "holoscan.temporal.window-timestamp-regression.v1",
          .cancelled_witness_identity = "holoscan.temporal.window-cancelled.v1",
          .producer_epoch_identity_schema = "holoscan.producer-epoch.v1",
      };
      for (const auto& bracketed : result.bracketed_windows) {
        if (bracketed.owner == trigger.owner) {
          window.bracketed_inputs.push_back(bracketed.input);
        }
      }
      result.window_triggers.push_back(std::move(window));
    }

    if (logical->contract.predicate) {
      const auto& predicate = *logical->contract.predicate;
      PredicateTriggerPlanRecord planned{
          .id = predicate.record_id,
          .owner = trigger.owner,
          .graph_generation = plan_generation,
          .identity_scope = predicate.scope == TemporalPredicateScope::kQualified
                                ? PredicateIdentityScope::kIndependentlyKeyed
                                : PredicateIdentityScope::kOperatorScoped,
          .identity = predicate.identity,
          .semantic_version = predicate.semantic_version,
          .source = predicate.source == TemporalPredicateSourceKind::kClock
                        ? PredicateEvaluationSourceKind::kClock
                        : PredicateEvaluationSourceKind::kInputUpdate,
          .driver = trigger.trigger_input,
          .clock = trigger.trigger_clock,
          .period_ns = predicate.period_ns,
          .phase = predicate.phase,
          .firing = predicate.firing,
          .initial_truth = false,
          .candidate_state_is_transactional = true,
          .compute_adopts_evaluator_snapshot = true,
          .consumes_reserved_candidate_tag = true,
          .state_size_bytes = predicate.state.size_bytes,
          .state_alignment_bytes = predicate.state.alignment_bytes,
          .state_schema = predicate.state.schema,
          .state_type_digest = predicate.state.present
                                   ? predicate_type_digest(predicate.state.schema,
                                                           predicate.state.size_bytes,
                                                           predicate.state.alignment_bytes)
                                   : std::string{},
          .state_generation = plan_generation,
          .has_config = predicate.config.present,
          .config_size_bytes = predicate.config.size_bytes,
          .config_alignment_bytes = predicate.config.alignment_bytes,
          .config_schema = predicate.config.present
                               ? std::optional<std::string>{predicate.config.schema}
                               : std::nullopt,
          .config_digest = predicate.config.present
                               ? std::optional<std::string>{predicate.config.digest}
                               : std::nullopt,
          .config_type_digest = predicate.config.present
                                    ? std::optional<std::string>{predicate_type_digest(
                                          predicate.config.schema,
                                          predicate.config.size_bytes,
                                          predicate.config.alignment_bytes)}
                                    : std::nullopt,
          .reset_policy = predicate.state.reset_policy,
          .reset_reasons = {"new_plan_generation",
                            "cold_run_start",
                            "operator_restart_rebind",
                            "producer_source_epoch_change",
                            "clock_discontinuity_after_restart"},
          .replay = replay_capability(predicate.replay),
          .replay_evidence = predicate.replay_evidence,
          .evaluation_witness_identity = "holoscan.temporal.predicate-evaluation.v1",
          .transition_witness_identity = "holoscan.temporal.predicate-transition.v1",
          .error_witness_identity = "holoscan.temporal.predicate-error.v1",
          .reset_witness_identity = "holoscan.temporal.predicate-reset.v1",
          .suppression_witness_identity = "holoscan.temporal.predicate-suppression.v1",
          .safe_default_witness_identity = "holoscan.temporal.predicate-safe-default.v1",
          .reaction_fire_witness_identity = "holoscan.temporal.predicate-reaction-fire.v1",
      };
      std::vector<PortId> predicate_phase_inputs;
      predicate_phase_inputs.reserve(predicate.additional_inputs.size() + 1U);
      std::uint32_t ordinal = 0U;
      if (planned.source == PredicateEvaluationSourceKind::kInputUpdate) {
        const auto selection = std::find_if(
            result.input_selections.begin(), result.input_selections.end(), [&](const auto& item) {
              return item.input == planned.driver;
            });
        if (selection == result.input_selections.end() ||
            selection->selector != InputSelectorKind::kExact) {
          add_error(result, "PREDICATE_DRIVER_INVALID", replay.owner, planned.driver);
          continue;
        }
        result.predicate_inputs.push_back({
            .predicate_id = planned.id,
            .input = planned.driver,
            .argument_ordinal = ordinal++,
            .implicit_driver = true,
            .selector = selection->selector,
            .missing = selection->missing,
            .reference_clock = selection->reference_clock,
            .conversion_id = selection->conversion_id,
            .cache_capacity = selection->cache_capacity,
            .payload_identity =
                find_port(snapshot, GraphPortId{planned.driver.value()})->payload_identity,
            .max_age_ns = selection->max_age_ns,
            .reuse_max_age_ns = selection->reuse_max_age_ns,
        });
        predicate_phase_inputs.push_back(planned.driver);
      }
      bool input_error = false;
      for (const auto& additional : predicate.additional_inputs) {
        const auto* port = replay_port(replay, additional.input_name);
        const auto selection =
            port == nullptr
                ? result.input_selections.end()
                : std::find_if(result.input_selections.begin(),
                               result.input_selections.end(),
                               [&](const auto& item) { return item.input == port->id; });
        if (port == nullptr || selection == result.input_selections.end() ||
            selection->cardinality != InputSelectionCardinality::kSingleSample) {
          add_error(result,
                    "PREDICATE_INPUT_INVALID",
                    replay.owner,
                    port == nullptr ? PortId{} : port->id);
          input_error = true;
          continue;
        }
        result.predicate_inputs.push_back({
            .predicate_id = planned.id,
            .input = port->id,
            .argument_ordinal = ordinal++,
            .implicit_driver = false,
            .selector = selection->selector,
            .missing = selection->missing,
            .reference_clock = selection->reference_clock,
            .conversion_id = selection->conversion_id,
            .cache_capacity = selection->cache_capacity,
            .payload_identity =
                find_port(snapshot, GraphPortId{port->id.value()})->payload_identity,
            .max_age_ns = selection->max_age_ns,
            .reuse_max_age_ns = selection->reuse_max_age_ns,
        });
        predicate_phase_inputs.push_back(port->id);
      }
      if (input_error) {
        continue;
      }
      if (!callable_plan) {
        add_error(result, "PREDICATE_CALLABLE_PROFILE_UNSUPPORTED", replay.owner);
        continue;
      }

      auto evaluation_aggregate =
          make_missing_aggregate(trigger.owner,
                                 MissingSelectionPhase::kPredicateEvaluation,
                                 current_selections,
                                 predicate_phase_inputs);
      if (!evaluation_aggregate) {
        add_error(result, "PREDICATE_INPUT_INVALID", replay.owner);
        continue;
      }
      result.missing_aggregates.push_back(std::move(*evaluation_aggregate));

      std::vector<InputSelectionPlanRecord> compute_only;
      compute_only.reserve(current_selections.size());
      for (const auto& selection : current_selections) {
        if (std::find(predicate_phase_inputs.begin(),
                      predicate_phase_inputs.end(),
                      selection.input) == predicate_phase_inputs.end()) {
          compute_only.push_back(selection);
        }
      }
      result.missing_aggregates.push_back(make_missing_aggregate(
          trigger.owner, MissingSelectionPhase::kPredicateCompute, compute_only));
      result.predicate_callables.push_back(std::move(*callable_plan));
      result.predicate_triggers.push_back(std::move(planned));
    }

    if (contract.compute_calls) {
      result.operator_completion_rules.push_back(OperatorCompletionPlanRecord{
          .owner = OperatorRef{.id = replay.operator_id, .plan_generation = plan_generation},
          .kind = OperatorCompletionRuleKind::kComputeCalls,
          .limit = contract.compute_calls->limit,
      });
    }
    result.triggers.push_back(std::move(trigger));
  }
  std::sort(
      result.input_selections.begin(),
      result.input_selections.end(),
      [](const auto& left, const auto& right) { return left.input.value() < right.input.value(); });
  std::sort(result.parameter_derived_terms.begin(),
            result.parameter_derived_terms.end(),
            [](const auto& left, const auto& right) { return left.term_id < right.term_id; });
  std::sort(result.operator_completion_rules.begin(),
            result.operator_completion_rules.end(),
            [](const auto& left, const auto& right) { return left.owner.id < right.owner.id; });
  ClockId scheduling_clock{};
  for (const auto& trigger : result.triggers) {
    if (!trigger.reference_clock.valid()) {
      continue;
    }
    if (!scheduling_clock.valid()) {
      scheduling_clock = trigger.reference_clock;
    } else if (trigger.reference_clock != scheduling_clock) {
      add_error(result, "TEMPORAL_SCHEDULING_DOMAIN_UNSUPPORTED", VertexId{trigger.owner.id});
      break;
    }
  }
  return result;
}

}  // namespace holoscan::detail
