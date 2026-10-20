/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "controlled_clock_run.hpp"

#include <algorithm>
#include <chrono>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>

namespace holoscan::testing::detail {
namespace {
using holoscan::detail::FixtureCandidateOutcome;
using holoscan::detail::FixtureObservationKind;
using holoscan::detail::FixtureObservationRecord;
using holoscan::detail::RuntimeEventKind;

[[noreturn]] void reject(std::string_view reason) {
  throw std::invalid_argument{"controlled operator test: " + std::string{reason}};
}

const PortPlanRecord& port(const ExecutionPlan& plan, VertexId owner, std::string_view name,
                           PortDirection direction) {
  const auto ports = plan.ports();
  const auto found = std::find_if(ports.begin(), ports.end(), [&](const auto& row) {
    return row.owner == owner && row.local_name == name && row.direction == direction;
  });
  if (found == ports.end())
    reject("fixture port identity changed");
  return *found;
}

VertexId vertex(const ExecutionPlan& plan, std::string_view name) {
  const auto vertices = plan.graph().vertices();
  const auto found = std::find_if(
      vertices.begin(), vertices.end(), [&](const auto& row) { return row.name == name; });
  if (found == vertices.end())
    reject("fixture operator identity changed");
  return found->id;
}

bool retired(const FixtureObservationRecord& record) {
  return record.kind == FixtureObservationKind::kCandidateFinished && record.admitted &&
         record.outcome == FixtureCandidateOutcome::kRetired;
}

bool ingested(std::span<const FixtureObservationRecord> records, const ClockFixtureEdge& edge,
              std::uint64_t sequence) {
  return std::any_of(records.begin(), records.end(), [&](const auto& record) {
    return record.kind == FixtureObservationKind::kInputIngested && record.owner == edge.target &&
           std::any_of(record.inputs.begin(), record.inputs.end(), [&](const auto& input) {
             return input.input == edge.input && input.has_sample &&
                    input.sample.source_sequence == sequence;
           });
  });
}

bool input_available(std::span<const FixtureObservationRecord> records,
                     const ClockFixtureEdge& edge) {
  for (const auto& record : records) {
    if (!retired(record) || record.owner != edge.source)
      continue;
    for (const auto& publication : record.publications) {
      if (publication.output == edge.output && publication.error == ErrorCode::kSuccess &&
          publication.sequence_consumed && ingested(records, edge, publication.sequence))
        return true;
    }
  }
  return false;
}

bool captured(std::span<const FixtureObservationRecord> records, const ClockFixtureEdge& edge,
              std::uint64_t sequence) {
  return std::any_of(records.begin(), records.end(), [&](const auto& record) {
    return retired(record) && record.owner == edge.target &&
           std::any_of(record.inputs.begin(), record.inputs.end(), [&](const auto& input) {
             return input.input == edge.input && input.has_sample &&
                    input.selection.state == InputState::kSelected &&
                    input.sample.source_sequence == sequence;
           });
  });
}

bool all_outputs_captured(std::span<const FixtureObservationRecord> records, VertexId tested,
                          std::span<const ClockFixtureEdge> edges) {
  for (const auto& record : records) {
    if (record.owner != tested || !retired(record))
      continue;
    for (const auto& publication : record.publications) {
      const auto edge = std::find_if(edges.begin(), edges.end(), [&](const auto& row) {
        return row.output == publication.output;
      });
      if (edge == edges.end() || !captured(records, *edge, publication.sequence))
        return false;
    }
  }
  return true;
}
void validate_attempt_options(const ExecutionPlan& plan, std::size_t input_count,
                              ControlledClockRun options) {
  if (input_count == 0U || options.tick.count() <= 0 || options.observation_capacity == 0U ||
      options.timeout.count() <= 0)
    reject("positive tick, capacity, timeout and at least one input are required");
  const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::time_point::max() - std::chrono::steady_clock::now());
  if (options.timeout >= remaining)
    reject("watchdog timeout cannot be represented by the steady clock");
  if (plan.execution_partitions().size() != 1U || plan.graph().graphs().size() != 1U ||
      plan.lifecycle_transaction_capacity() || !plan.max_start_delays().empty() ||
      !plan.predicate_triggers().empty())
    reject("one partition, one Graph and no managed restart are currently qualified");
}

ClockId validate_fixture_clock(const ExecutionPlan& plan) {
  const auto catalog = plan.clock_catalog();
  if (catalog.clocks.size() != 1U || !catalog.clocks.front().shared ||
      !catalog.clocks.front().is_default || catalog.clocks.front().key.name() != "fixture_clock")
    reject("the fixture-owned controlled clock must remain the sole default clock");
  return catalog.root_default_clock;
}

void validate_clock_contract(const ExecutionPlan& plan, std::string_view tested_name, ClockId clock,
                             std::chrono::nanoseconds tick) {
  const auto parameters = plan.operator_parameters();
  const auto owner = std::find_if(parameters.begin(), parameters.end(), [&](const auto& row) {
    const auto path = OperatorPath::parse(row.operator_path);
    return path && path->segments().size() == 1U && path->segments().front() == tested_name;
  });
  if (owner == parameters.end())
    reject("operator identity missing");
  const auto clocks = plan.clock_triggers();
  const auto trigger = std::find_if(
      clocks.begin(), clocks.end(), [&](const auto& row) { return row.owner == owner->owner; });
  if (trigger == clocks.end() || trigger->clock != clock || trigger->explicit_phase ||
      trigger->period_ns <= 0 || tick.count() % trigger->period_ns != 0)
    reject("OnClock with its ordinary startup phase and an exact period boundary is required");
  for (const auto& row : clocks) {
    if (row.clock != clock || row.explicit_phase)
      reject("helper clocks must use the fixture default and ordinary startup phase");
  }
}

void validate_selection_contracts(const ExecutionPlan& plan, VertexId tested) {
  for (const auto& row : plan.ports()) {
    if (row.payload_identity.kind != PayloadIdentityKind::kScalar)
      reject("controlled observation currently qualifies scalar payloads only");
    if (row.owner == tested && row.direction == PortDirection::kInput) {
      const auto selections = plan.input_selections();
      const auto selected = std::find_if(selections.begin(),
                                         selections.end(),
                                         [&](const auto& value) { return value.input == row.id; });
      if (selected == selections.end() || selected->selector != InputSelectorKind::kLatestWithin ||
          selected->missing != TemporalRecordMissingKind::kSkip ||
          selected->cardinality != InputSelectionCardinality::kSingleSample || row.optional)
        reject("required single-sample LatestWithin inputs with Skip are currently qualified");
    }
  }
}

ClockFixtureEdge resolve_fixture_edge(const ExecutionPlan& plan, VertexId tested,
                                      const ClockFixtureConnection& connection, bool incoming) {
  const auto helper = vertex(plan, connection.helper);
  const auto& source = port(plan,
                            incoming ? helper : tested,
                            incoming ? connection.helper_port : connection.operator_port,
                            PortDirection::kOutput);
  const auto& target = port(plan,
                            incoming ? tested : helper,
                            incoming ? connection.operator_port : connection.helper_port,
                            PortDirection::kInput);
  const auto edges = plan.payload_edges();
  if (std::count_if(edges.begin(), edges.end(), [&](const auto& row) {
        return row.target == target.id;
      }) != 1)
    reject("each observed input requires one exact producer");
  const auto edge = std::find_if(edges.begin(), edges.end(), [&](const auto& row) {
    return row.source == source.id && row.target == target.id;
  });
  if (edge == edges.end() || edge->queue_policy != QueuePolicy::kBlocking ||
      edge->locality != ChannelLocalityKind::kProcessLocal)
    reject("non-dropping local fixture edges are required");
  return ClockFixtureEdge{
      connection.operator_port, target.id, source.id, source.owner, target.owner};
}

void validate_fixture_coverage(const ExecutionPlan& plan, const ControlledClockAttempt& attempt) {
  for (const auto& row : plan.graph().vertices()) {
    if (row.id != attempt.tested &&
        std::none_of(attempt.inputs.begin(),
                     attempt.inputs.end(),
                     [&](const auto& edge) { return edge.source == row.id; }) &&
        std::none_of(attempt.outputs.begin(), attempt.outputs.end(), [&](const auto& edge) {
          return edge.target == row.id;
        }))
      reject("controlled observation only supports the operator and fixture-owned helpers");
  }
  std::size_t input_count{}, output_count{};
  for (const auto& row : plan.ports()) {
    if (row.owner != attempt.tested)
      continue;
    if (row.direction == PortDirection::kInput)
      ++input_count;
    else
      ++output_count;
  }
  if (input_count != attempt.inputs.size() || output_count != attempt.outputs.size() ||
      plan.payload_edges().size() != attempt.inputs.size() + attempt.outputs.size())
    reject("every input needs staging and every output needs one collector");
}

struct CandidateProgress {
  bool failed{};
  bool startup_finished{};
  const FixtureObservationRecord* requested_tick{};
};

CandidateProgress candidate_progress(std::span<const FixtureObservationRecord> records,
                                     VertexId tested, ClockId clock,
                                     std::chrono::nanoseconds requested_tick) {
  CandidateProgress progress;
  for (const auto& record : records) {
    if (record.kind != FixtureObservationKind::kCandidateFinished)
      continue;
    if (record.outcome == FixtureCandidateOutcome::kFailed ||
        std::any_of(record.publications.begin(), record.publications.end(), [](const auto& p) {
          return p.error != ErrorCode::kSuccess || !p.sequence_consumed;
        })) {
      progress.failed = true;
      return progress;
    }
    if (record.owner != tested || record.witness.kind != RuntimeEventKind::kClockTick ||
        record.witness.reference_time.clock_id != clock)
      continue;
    const auto tick = std::chrono::nanoseconds{record.witness.reference_time.timestamp_ns};
    progress.startup_finished = progress.startup_finished || tick == std::chrono::nanoseconds{0};
    if (tick == requested_tick)
      progress.requested_tick = &record;
  }
  return progress;
}
}  // namespace

std::shared_ptr<ControlledClockAttempt> prepare_controlled_clock(
    const ExecutionPlan& plan, std::string_view tested_name,
    std::span<const ClockFixtureConnection> inputs, std::span<const ClockFixtureConnection> outputs,
    ControlledClockRun options) {
  validate_attempt_options(plan, inputs.size(), options);
  auto result = std::make_shared<ControlledClockAttempt>();
  result->tested = vertex(plan, tested_name);
  result->clock = validate_fixture_clock(plan);
  validate_clock_contract(plan, tested_name, result->clock, options.tick);
  validate_selection_contracts(plan, result->tested);
  for (const auto& input : inputs)
    result->inputs.push_back(resolve_fixture_edge(plan, result->tested, input, true));
  for (const auto& output : outputs)
    result->outputs.push_back(resolve_fixture_edge(plan, result->tested, output, false));
  validate_fixture_coverage(plan, *result);
  result->observation = std::make_shared<holoscan::detail::RuntimeFixtureObservation>(
      options.observation_capacity, plan.ports().size(), options.observation_capacity);
  return result;
}

void ControlledClockAttempt::drive(RunSession& session, SyntheticClock& controlled_clock,
                                   ControlledClockRun options) {
  const auto now = std::chrono::steady_clock::now();
  const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::time_point::max() - now);
  const auto deadline = options.timeout < remaining ? now + options.timeout
                                                    : std::chrono::steady_clock::time_point::max();
  bool advanced{};
  for (;;) {
    // Observe terminal state before acquiring the committed record prefix, so the final
    // retirement cannot be missed when the session ends between controller polls.
    const bool running = session.is_running();
    if (observation->invalid()) {
      failure = "controlled clock observation capacity or identity invalidated";
      break;
    }
    const auto records = observation->records();
    const auto progress = candidate_progress(records, tested, clock, options.tick);
    if (progress.failed) {
      failure = "controlled clock observation contains failed retirement or publication";
      break;
    }
    tick_record = progress.requested_tick;
    if (running && !advanced && tick_record == nullptr && progress.startup_finished &&
        std::all_of(inputs.begin(), inputs.end(), [&](const auto& input) {
          return input_available(records, input);
        })) {
      controlled_clock.advance_to(options.tick.count());
      advanced = true;
    }
    if (tick_record != nullptr && all_outputs_captured(records, tested, outputs)) {
      boundary_reached = true;
      break;
    }
    if (!running)
      break;
    if (std::chrono::steady_clock::now() >= deadline) {
      watchdog_expired = true;
      failure = "controlled clock observation watchdog expired before the Runtime barrier";
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds{1});
  }
  if (!boundary_reached && failure.empty())
    failure = "controlled clock observation ended before the requested Runtime barrier";
  session.request_stop();
}

ClockTestOutcome ControlledClockAttempt::outcome() const noexcept {
  if (!boundary_reached || tick_record == nullptr || observation->invalid())
    return ClockTestOutcome::kFailed;
  return tick_record->outcome == FixtureCandidateOutcome::kSuppressed
             ? ClockTestOutcome::kSuppressed
             : ClockTestOutcome::kRetired;
}

InputSelection ControlledClockAttempt::selection(std::string_view name) const {
  if (outcome() == ClockTestOutcome::kFailed)
    throw std::logic_error{"controlled tick did not complete"};
  const auto edge =
      std::find_if(inputs.begin(), inputs.end(), [&](const auto& row) { return row.name == name; });
  if (edge == inputs.end())
    throw std::invalid_argument{"unknown controlled input"};
  const auto selected = std::find_if(tick_record->inputs.begin(),
                                     tick_record->inputs.end(),
                                     [&](const auto& row) { return row.input == edge->input; });
  if (selected == tick_record->inputs.end())
    throw std::logic_error{"native selection is missing from controlled observation"};
  return selected->selection;
}

}  // namespace holoscan::testing::detail
