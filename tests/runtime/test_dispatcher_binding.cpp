/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <utility>
#include <vector>

#include <holoscan/channel/clock_health.hpp>
#include <holoscan/time/synthetic_clock.hpp>

#include "common/allocation_counter.hpp"
#include "core/source_id_internal.hpp"
#include "core/temporal_contract_internal.hpp"
#include "runtime/dispatcher.hpp"
#include "runtime/endpoint_clock_health.hpp"
#include "runtime/predicate_evaluator.hpp"
#include "runtime/runtime_phase_counters.hpp"
#include "runtime/temporal_depth_publication.hpp"

namespace holoscan {
namespace {

constexpr ClockId kRuntimeClock{9U};
constexpr ClockId kMaxStartDelaySourceClock{8U};
constexpr PlanGenerationId kMaxStartDelayPlanGeneration{77U};

// Manufactures a ReadyCandidateView at a given logical timestamp, plan order, and source id.
// Representative test: DispatcherOwnsOnePartitionLocalSerialSelection.
detail::scheduling::ReadyCandidateView ready_candidate(std::int64_t logical_ns, std::uint32_t order,
                                                       std::uint64_t source) {
  return {
      .partition = ExecutionPartitionId{1U},
      .execution_lane = 0U,
      .vertex = VertexId{order + 1U},
      .tag_time = TagTime{logical_ns, 0U, 0U},
      .plan_vertex_order = order,
      .stable_source_id = detail::make_source_id(static_cast<std::uint32_t>(source)),
      .stable_port_id = {},
      .stable_candidate_tie = source,
      .active = true,
  };
}

// Minimal single-lane deterministic-serial scheduler plan shared across Dispatcher tests.
// Representative test: DispatcherOwnsOnePartitionLocalSerialSelection.
SchedulerPlanRecord scheduler_plan() {
  return {
      .ready_policy = AlgorithmKey{"holoscan.ready.deterministic_serial", 1U},
      .execution_lane_count = 1U,
      .max_concurrent_activations = 1U,
      .activation_overlap = false,
      .worker_count = 1U,
      .intra_path_parallelism = 1U,
  };
}

// Constructs a RuntimeSourceBindingRecord for a given event kind and optional queue capacity.
// The vertex and stable_source_id are derived from `index` to keep ordinals consistent.
// Representative test: CoordinatorFreezesEverySourceKindInDeterministicOrder.
detail::RuntimeSourceBindingRecord source_binding(std::uint32_t index,
                                                  detail::RuntimeEventKind kind,
                                                  std::uint32_t capacity = 1U) {
  return {
      .candidate_kind = kind,
      .source = detail::make_source_id(index + 1U),
      .port = kind == detail::RuntimeEventKind::kInputArrival ? PortId{index + 1U} : PortId{},
      .source_clock = kRuntimeClock,
      .reference_clock = kRuntimeClock,
      .source_to_reference_conversion_id = 0U,
      .source_order = index,
      .capacity = capacity,
      .source_generation = 1U,
      .tag_generation = 1U,
      .receiver_index = 0U,
      .partition = ExecutionPartitionId{1U},
      .execution_lane = 0U,
      .vertex = VertexId{index + 1U},
      .plan_vertex_order = index,
      .stable_candidate_tie = index + 1U,
  };
}

// Constructs a RuntimeEvent matching a source binding at a given sequence and reference time.
// Clock IDs and source_generation are copied from the binding so ingestion succeeds without
// modification. Representative test: RestartFencesOldSourceEpochAndPublishesNewTagGeneration.
detail::RuntimeEvent runtime_event(const detail::RuntimeSourceBindingRecord& source,
                                   std::uint64_t sequence, std::int64_t reference_ns) {
  return {
      .kind = source.candidate_kind,
      .source = source.source,
      .port = source.port,
      .source_decision_time = TimePoint{reference_ns, source.reference_clock},
      .reference_time = TimePoint{reference_ns, source.reference_clock},
      .source_generation = source.source_generation,
      .source_sequence = sequence,
  };
}

// Builds a TemporalSampleRecord with the given capture timestamp, source sequence, and arrival.
// producer_epoch is fixed at 1; tests needing epoch transitions modify the record directly.
// Representative test: PlanBackedScalarAndDrainSelectorsCommitOnlyAcceptedCandidates.
detail::TemporalSampleRecord temporal_sample(std::int64_t capture_ns, std::uint64_t sequence,
                                             std::uint64_t arrival) {
  return {
      .capture_time = TimePoint{capture_ns, kRuntimeClock},
      .source_sequence = sequence,
      .producer_epoch = 1U,
      .arrival_sequence = arrival,
  };
}

// Full EndpointClockHealthPlanRecord wired to publisher-serialized monitoring for a given port.
// Policy: kSkip for absent/stale, kDegraded for degraded, kFailRun for unhealthy.
// Representative test: AppliesFrozenClockHealthDispositionOnlyToOtherwiseReadyWork.
EndpointClockHealthPlanRecord monitored_health_plan(PortId endpoint) {
  return {
      .endpoint = endpoint,
      .plan_generation = 1U,
      .capture_clock = kRuntimeClock,
      .producer_epoch_source = EndpointProducerEpochSource::kAttachmentGeneration,
      .capability = ClockHealthCapability::kStatusReports,
      .requirement = ClockHealthRequirement::kFreshHealthyReport,
      .max_report_age = std::chrono::nanoseconds{100},
      .health_age_clock = ClockId{10U},
      .fixed_report_ingress_capacity = 1U,
      .fixed_update_capacity = 16U,
      .ingress_capacity_origin = EndpointClockHealthBoundOrigin::kPortableMinimum,
      .per_dispatch_report_budget = 1U,
      .publisher_operation_sequence_id = "holoscan.clock_health.publisher-operation-sequence.v1",
      .report_sequence_id = "holoscan.clock_health.report-sequence.v1",
      .health_update_sequence_id = "holoscan.clock_health.update-sequence.v1",
      .monitor_path_kind = EndpointClockHealthMonitorPathKind::kPublisherSerialized,
      .no_report_disposition = EndpointClockHealthDisposition::kSkip,
      .degraded_disposition = EndpointClockHealthDisposition::kDegraded,
      .unhealthy_disposition = EndpointClockHealthDisposition::kFailRun,
      .stale_disposition = EndpointClockHealthDisposition::kSkip,
      .clock_health_update_witness_id = "holoscan.event.clock-health-update.v1",
      .trace_loss_disposition = EndpointClockHealthTraceLossDisposition::kFailClosed,
  };
}

// Builds an InputSelectionPlanRecord for the given selector kind, port, and capacity.
// Cardinality and batch_selection_mode are derived from the selector to satisfy plan validation.
// Representative test: PlanBackedScalarAndDrainSelectorsCommitOnlyAcceptedCandidates.
InputSelectionPlanRecord selection_plan(InputSelectorKind selector, PortId input,
                                        std::uint32_t cache_capacity,
                                        std::uint32_t max_selected_samples) {
  InputSelectionPlanRecord plan{
      .input = input,
      .selector = selector,
      .cardinality =
          selector == InputSelectorKind::kExact || selector == InputSelectorKind::kLatestWithin
              ? InputSelectionCardinality::kSingleSample
              : InputSelectionCardinality::kSampleBatch,
      .max_selected_samples = max_selected_samples,
      .max_selected_samples_origin =
          selector == InputSelectorKind::kExact || selector == InputSelectorKind::kLatestWithin
              ? SelectedSampleBoundOrigin::kSingleSample
          : selector == InputSelectorKind::kDrainAll
              ? SelectedSampleBoundOrigin::kResolvedRetainedCapacity
              : SelectedSampleBoundOrigin::kSelectorMaxSamples,
      .batch_selection_mode = selector == InputSelectorKind::kDrainAll
                                  ? BatchSelectionMode::kConsumeSelected
                              : selector == InputSelectorKind::kBracketedWindow
                                  ? BatchSelectionMode::kNonDestructiveSnapshot
                                  : BatchSelectionMode::kNotApplicable,
      .cache_capacity = cache_capacity,
      .reference_clock = kRuntimeClock,
  };
  return plan;
}

// Mirror only the selector inventory fields consumed by Runtime's source-private temporal-depth
// publication algorithm. The focused regression moves selectors through this inventory so it
// exercises the same validate-all/apply-all implementation as RunSession without a Core ABI seam.
struct TemporalDepthPublicationTestState {
  struct InputSelectorSlot {
    detail::RuntimeInputSelectionState selector;
    bool aligned_member{};
  };

  struct AlignedSelectorSlot {
    detail::RuntimeAlignedSelectionState selector;
  };

  std::vector<InputSelectorSlot> input_selectors;
  std::vector<AlignedSelectorSlot> aligned_selectors;
};

[[nodiscard]] bool commit_two_ordinary_selectors(detail::RuntimeInputSelectionState& first,
                                                 detail::RuntimeInputSelectionState& second) {
  TemporalDepthPublicationTestState state;
  state.input_selectors.reserve(2U);
  state.input_selectors.push_back(
      TemporalDepthPublicationTestState::InputSelectorSlot{.selector = std::move(first)});
  state.input_selectors.push_back(
      TemporalDepthPublicationTestState::InputSelectorSlot{.selector = std::move(second)});
  std::span<TemporalDepthPublicationTestState> states{&state, 1U};

  const bool committed = detail::runtime_internal::commit_rate_candidate_temporal_depths(states);
  first = std::move(state.input_selectors[0].selector);
  second = std::move(state.input_selectors[1].selector);
  return committed;
}

// Two-member AlignedTriggerPlanRecord covering ports 1 and 2 with configurable capacity/window.
// Must be paired with aligned_selections() when constructing RuntimeAlignedSelectionState.
// Representative test: PlanBackedAlignedJoinCommitsAcceptedHeadsAndRetiresEveryMinimumMiss.
AlignedTriggerPlanRecord aligned_plan(std::uint32_t capacity = 2U,
                                      std::int64_t inclusive_window_ns = 1) {
  AlignedTriggerPlanRecord plan{
      .owner = OperatorRef{1U, kMaxStartDelayPlanGeneration, 0U},
      .members =
          std::vector<AlignedMemberPlanRecord>{
              AlignedMemberPlanRecord{
                  .input = PortId{1U},
                  .member_ordinal = 0U,
                  .event_clock = kRuntimeClock,
                  .queue_capacity = capacity,
                  .cache_capacity = capacity,
              },
              AlignedMemberPlanRecord{
                  .input = PortId{2U},
                  .member_ordinal = 1U,
                  .event_clock = kRuntimeClock,
                  .queue_capacity = capacity,
                  .cache_capacity = capacity,
              },
          },
      .inclusive_window_ns = inclusive_window_ns,
      .comparison_clock = kRuntimeClock,
  };
  plan.closure_proofs = {
      AlignedClosureProofKind::kStreamAdvancement,
      AlignedClosureProofKind::kLossWitness,
      AlignedClosureProofKind::kPrivateWatermark,
      AlignedClosureProofKind::kEndOfStream,
      AlignedClosureProofKind::kProducerEpochClosed,
      AlignedClosureProofKind::kTemporalClosure,
  };
  return plan;
}

// Paired kExact InputSelectionPlanRecords for ports 1 and 2 at the given cache capacity.
// Representative test: PlanBackedAlignedJoinCommitsAcceptedHeadsAndRetiresEveryMinimumMiss.
std::array<InputSelectionPlanRecord, 2U> aligned_selections(std::uint32_t capacity = 2U) {
  return {
      selection_plan(InputSelectorKind::kExact, PortId{1U}, capacity, 1U),
      selection_plan(InputSelectorKind::kExact, PortId{2U}, capacity, 1U),
  };
}

// Single-operator MaxStartDelayPlanRecord with a 10-ns budget driven by predicate-driver
// coordinate. Representative test: MaxStartDelayGateConsumesFrozenPlanRecordAfterSelection.
MaxStartDelayPlanRecord max_start_delay_plan(OnViolation action) {
  return {
      .owner = OperatorRef{1U, 77U, 0U},
      .budget_ns = 10,
      .action = action,
      .source_coordinate = MaxStartDelaySourceCoordinateKind::kPredicateDriver,
      .source_clock = kRuntimeClock,
      .callback_count_if_late = action == OnViolation::kFailRun ? 0U : 1U,
  };
}

// Deadline plan that reads its start coordinate from an external source clock via conversion_id 1.
// Pairs with cross_clock_max_start_delay_trigger() and cross_clock_max_start_delay_conversion().
// Representative test: CrossClockMaxStartDelayAddsBudgetBeforeExactReferenceComparison.
MaxStartDelayPlanRecord cross_clock_max_start_delay_plan(OnViolation action) {
  auto result = max_start_delay_plan(action);
  result.source_coordinate = MaxStartDelaySourceCoordinateKind::kTriggerSample;
  result.source_clock = kMaxStartDelaySourceClock;
  result.conversion_id = 1U;
  return result;
}

// TemporalTriggerPlanRecord referencing conversion_id 1 for the cross-clock max_start_delay tests.
// The paired conversion carries a 3/2 scale and +5-ns offset.
TemporalTriggerPlanRecord cross_clock_max_start_delay_trigger(OnViolation action) {
  return {
      .owner = OperatorRef{1U, kMaxStartDelayPlanGeneration, 0U},
      .kind = TemporalRecordTriggerKind::kEach,
      .reference_clock = kRuntimeClock,
      .conversion_id = 1U,
      .max_start_delay_ns = 10,
      .violation = action,
  };
}

// Maps kMaxStartDelaySourceClock -> kRuntimeClock via scale 3/2 and +5-ns offset.
// Drives the max_start_delay arithmetic in
// CrossClockMaxStartDelayAddsBudgetBeforeExactReferenceComparison.
ClockConversionRecord cross_clock_max_start_delay_conversion() {
  return {
      .id = 1U,
      .plan_generation = kMaxStartDelayPlanGeneration,
      .graph = GraphId{1U},
      .from = kMaxStartDelaySourceClock,
      .to = kRuntimeClock,
      .scale = {.numerator = 3, .denominator = 2U},
      .offset_ns = 5,
  };
}

// REQ: RUNTIME-002
// REQ: RUNTIME-007
TEST(DispatcherBindingTest, DirectBindingIsBoundedOrderedAndExplicitlyClosed) {
  // Key setup: capacity=2; third push must return kBackpressured; drain preserves arrival order.
  detail::BoundedEventPortBinding binding{2U};
  const detail::RuntimeEvent first{
      .kind = detail::RuntimeEventKind::kReadable,
      .source = detail::make_source_id(1U),
      .source_decision_time = TimePoint{10, kRuntimeClock},
      .reference_time = TimePoint{10, kRuntimeClock},
      .source_generation = 1U,
      .source_sequence = 1U,
  };
  const detail::RuntimeEvent second{
      .kind = detail::RuntimeEventKind::kNotification,
      .source = detail::make_source_id(2U),
      .source_decision_time = TimePoint{11, kRuntimeClock},
      .reference_time = TimePoint{11, kRuntimeClock},
      .source_generation = 1U,
      .source_sequence = 1U,
  };
  EXPECT_TRUE(binding.push(first));
  EXPECT_TRUE(binding.push(second));
  EXPECT_EQ(binding.push(first).error().code, ErrorCode::kBackpressured);

  std::array<detail::RuntimeEvent, 2> drained;
  auto count = binding.drain(drained);
  ASSERT_TRUE(count);
  EXPECT_EQ(count.value(), 2U);
  EXPECT_EQ(drained[0].source, detail::make_source_id(1U));
  EXPECT_EQ(drained[1].source, detail::make_source_id(2U));

  binding.close();
  // Key assertion: push after close() returns kClosed, and closed() reports true.
  EXPECT_EQ(binding.push(first).error().code, ErrorCode::kClosed);
  EXPECT_TRUE(binding.closed());
}

// REQ: RUNTIME-002
// REQ: RUNTIME-007
// One kSpin source -> coordinator [publish_restart] -> stale event rejected -> new-epoch candidate.
TEST(DispatcherBindingTest, RestartFencesOldSourceEpochAndPublishesNewTagGeneration) {
  const auto source = source_binding(0U, detail::RuntimeEventKind::kSpin);
  auto coordinator = detail::RuntimeCandidateCoordinator::create({&source, 1U}, 1U, 1U);
  ASSERT_TRUE(coordinator);

  detail::BoundedEventPortBinding first_binding{1U};
  ASSERT_TRUE(first_binding.push(runtime_event(source, 1U, 10)));
  ASSERT_EQ(coordinator->ingest(first_binding), 1U);
  auto first = coordinator->freeze_next();
  ASSERT_TRUE(first);
  EXPECT_EQ(first->candidate.tag.generation, 1U);
  EXPECT_EQ(first->candidate.tag.sequence, 1U);
  ASSERT_TRUE(coordinator->retire(*first));

  // Key setup: one event ingested and retired; publish_restart(0) advances the generation fence.
  ASSERT_TRUE(coordinator->publish_restart(0U));

  detail::BoundedEventPortBinding stale_binding{1U};
  ASSERT_TRUE(stale_binding.push(runtime_event(source, 2U, 20)));
  const auto stale = coordinator->ingest(stale_binding);
  ASSERT_FALSE(stale);
  EXPECT_EQ(stale.error().code, ErrorCode::kStaleEndpoint);

  auto restarted_event = runtime_event(source, 1U, 20);
  restarted_event.source_generation = 2U;
  detail::BoundedEventPortBinding restarted_binding{1U};
  ASSERT_TRUE(restarted_binding.push(restarted_event));
  ASSERT_EQ(coordinator->ingest(restarted_binding), 1U);
  auto restarted = coordinator->freeze_next();
  ASSERT_TRUE(restarted);
  // Key assertion: post-restart event with source_generation=2 yields tag.generation=2, sequence=1.
  EXPECT_EQ(restarted->candidate.tag.generation, 2U);
  EXPECT_EQ(restarted->candidate.tag.sequence, 1U);
  EXPECT_EQ(restarted->witness.source_generation, 2U);
  EXPECT_EQ(restarted->witness.source_sequence, 1U);
  EXPECT_TRUE(coordinator->retire(*restarted));
}

// REQ: RUNTIME-007
// REQ: RUNTIME-014
TEST(DispatcherBindingTest, RejectsMixedReferenceClocksWithSpecificRuntimeEvidence) {
  // Key setup: two sources where the second uses a different reference_clock than the first.
  const auto first = source_binding(0U, detail::RuntimeEventKind::kClockTick);
  auto second = source_binding(1U, detail::RuntimeEventKind::kClockTick);
  second.source_clock = ClockId{10U};
  second.reference_clock = ClockId{10U};
  const std::array sources{first, second};

  auto coordinator = detail::RuntimeCandidateCoordinator::create(sources, 1U, sources.size());

  // Key assertion: create() fails with RUNTIME_SCHEDULING_DOMAIN_MISMATCH.
  ASSERT_FALSE(coordinator);
  EXPECT_EQ(coordinator.error().code, ErrorCode::kInvalidArgument);
  EXPECT_EQ(coordinator.error().message, "RUNTIME_SCHEDULING_DOMAIN_MISMATCH");
}

// REQ: RUNTIME-007
// kClockTick + kSpin sources -> coordinator; ingest one event -> that source loses credit.
TEST(DispatcherBindingTest, SourceCreditPreflightPreservesPendingSingleCreditIngress) {
  const std::array sources{
      source_binding(0U, detail::RuntimeEventKind::kClockTick),
      source_binding(1U, detail::RuntimeEventKind::kSpin),
  };
  auto coordinator = detail::RuntimeCandidateCoordinator::create(sources, 2U, sources.size());
  ASSERT_TRUE(coordinator);
  const auto first_initial_credit = coordinator->source_credit_remaining(sources[0].source);
  const auto second_initial_credit = coordinator->source_credit_remaining(sources[1].source);
  ASSERT_TRUE(first_initial_credit);
  ASSERT_TRUE(second_initial_credit);
  EXPECT_TRUE(*first_initial_credit);
  EXPECT_TRUE(*second_initial_credit);

  // Key setup: first source ingested but not frozen; its credit is consumed until retire().
  detail::BoundedEventPortBinding binding{1U};
  ASSERT_TRUE(binding.push(runtime_event(sources[0], 1U, 10)));
  ASSERT_EQ(coordinator->ingest(binding), 1U);
  const auto first_pending_credit = coordinator->source_credit_remaining(sources[0].source);
  const auto second_pending_credit = coordinator->source_credit_remaining(sources[1].source);
  ASSERT_TRUE(first_pending_credit);
  ASSERT_TRUE(second_pending_credit);
  EXPECT_FALSE(*first_pending_credit);
  EXPECT_TRUE(*second_pending_credit);

  auto frozen = coordinator->freeze_next();
  ASSERT_TRUE(frozen);
  // Freezing assigns candidate identity but retains the physical source credit through compute.
  const auto first_frozen_credit = coordinator->source_credit_remaining(sources[0].source);
  ASSERT_TRUE(first_frozen_credit);
  EXPECT_FALSE(*first_frozen_credit);
  ASSERT_TRUE(coordinator->retire(*frozen));
  // Key assertion: after retire(), the consumed credit is restored; sibling credit is unaffected.
  const auto first_retired_credit = coordinator->source_credit_remaining(sources[0].source);
  ASSERT_TRUE(first_retired_credit);
  EXPECT_TRUE(*first_retired_credit);
  const auto unknown = coordinator->source_credit_remaining(detail::make_source_id(99U));
  ASSERT_FALSE(unknown);
  EXPECT_EQ(unknown.error().code, ErrorCode::kNotFound);
}

// REQ: RUNTIME-007
TEST(DispatcherBindingTest, EndOfStreamWaitsForAcceptedPrefixWithFutureCaptureTimes) {
  for (const bool constrained : {false, true}) {
    auto input = source_binding(0U, detail::RuntimeEventKind::kInputArrival, 3U);
    auto unrelated = source_binding(1U, detail::RuntimeEventKind::kClockTick);
    unrelated.vertex = VertexId{2U};
    unrelated.receiver_index = 1U;
    auto fenced_input = source_binding(2U, detail::RuntimeEventKind::kInputArrival);
    fenced_input.vertex = VertexId{3U};
    fenced_input.receiver_index = 2U;
    const std::array sources{input, unrelated, fenced_input};
    auto coordinator = detail::RuntimeCandidateCoordinator::create(sources, 3U, 5U);
    ASSERT_TRUE(coordinator);
    detail::BoundedEventPortBinding input_binding{3U, input.source};
    detail::BoundedEventPortBinding other_binding{1U, unrelated.source};
    ASSERT_TRUE(input_binding.push(runtime_event(input, 1U, 100)));
    ASSERT_TRUE(input_binding.push(runtime_event(input, 2U, 5)));
    auto terminal = runtime_event(input, 3U, 0);
    terminal.kind = detail::RuntimeEventKind::kInputEndOfStream;
    ASSERT_TRUE(input_binding.push(terminal));
    ASSERT_EQ(coordinator->ingest(input_binding), 3U);
    const std::array fences{detail::RuntimeCandidateInputFence{
        fenced_input.vertex, fenced_input.source, fenced_input.port}};
    const auto freeze = [&]() {
      return constrained ? coordinator->freeze_next_constrained(fences)
                         : coordinator->freeze_next();
    };

    auto second_data = freeze();
    ASSERT_TRUE(second_data);
    ASSERT_EQ(second_data->witness.kind, detail::RuntimeEventKind::kInputArrival);
    EXPECT_EQ(second_data->witness.source_sequence, 2U);
    EXPECT_EQ(second_data->witness.reference_time.timestamp_ns, 5);
    ASSERT_TRUE(coordinator->retire(*second_data));
    auto first_data = freeze();
    ASSERT_TRUE(first_data);
    EXPECT_EQ(first_data->witness.source_sequence, 1U);
    EXPECT_EQ(first_data->witness.reference_time.timestamp_ns, 100);

    // Freezing is insufficient: the accepted predecessor must actually retire.
    const auto blocked = freeze();
    ASSERT_FALSE(blocked);
    EXPECT_EQ(blocked.error().code, ErrorCode::kNotReady);
    EXPECT_EQ(coordinator->pending_count(), 1U);
    EXPECT_EQ(coordinator->outstanding_count(), 2U);

    // Repeated polling while the predecessor is held cannot spend another source credit, grow
    // terminal storage, or allocate a retry record. Exercise both ordinary and constrained freeze.
    bool bounded_retry = true;
    std::size_t retry_allocations = 0U;
    {
      const test::ScopedAllocationCounter counter;
      for (std::size_t attempt = 0U; attempt < 32U; ++attempt) {
        const auto retry = freeze();
        const auto credit = coordinator->source_credit_remaining(input.source);
        bounded_retry = bounded_retry && !retry && retry.error().code == ErrorCode::kNotReady &&
                        coordinator->pending_count() == 1U &&
                        coordinator->outstanding_count() == 2U && credit && *credit == 1U;
      }
      retry_allocations = counter.count();
    }
    EXPECT_TRUE(bounded_retry);
    EXPECT_EQ(retry_allocations, 0U);

    ASSERT_TRUE(other_binding.push(runtime_event(unrelated, 1U, 200)));
    ASSERT_EQ(coordinator->ingest(other_binding), 1U);
    auto other = freeze();
    ASSERT_TRUE(other);
    EXPECT_EQ(other->witness.source, unrelated.source);
    ASSERT_TRUE(coordinator->retire(*other));

    ASSERT_TRUE(coordinator->retire(*first_data));
    auto eos = freeze();
    ASSERT_TRUE(eos);
    EXPECT_EQ(eos->witness.kind, detail::RuntimeEventKind::kInputEndOfStream);
    EXPECT_EQ(eos->witness.source_sequence, 3U);
    EXPECT_EQ(eos->witness.reference_time.timestamp_ns, 0);
    ASSERT_TRUE(coordinator->rearm_terminal_successor(*eos));
    EXPECT_EQ(eos->witness.source_sequence, 3U);
    EXPECT_EQ(coordinator->outstanding_count(), 1U);
    ASSERT_TRUE(coordinator->retire(*eos));
    EXPECT_EQ(coordinator->outstanding_count(), 0U);
  }
}

// REQ: RUNTIME-007
// REQ: RUNTIME-043
// Two non-trigger freshness prefixes fence only their common owner. The globally earlier witness
// from an unrelated owner must still run first, while each fenced input contributes only its
// oldest source-sequence arrival even when its capture/reference coordinates move backwards.
TEST(DispatcherBindingTest, InputFencesPreserveGlobalOrderAndEachSourcePrefix) {
  auto first_input = source_binding(0U, detail::RuntimeEventKind::kInputArrival, 3U);
  auto second_input = source_binding(1U, detail::RuntimeEventKind::kInputArrival);
  second_input.vertex = first_input.vertex;
  second_input.receiver_index = first_input.receiver_index;
  auto fenced_trigger = source_binding(2U, detail::RuntimeEventKind::kClockTick);
  fenced_trigger.vertex = first_input.vertex;
  fenced_trigger.receiver_index = first_input.receiver_index;
  auto unrelated = source_binding(3U, detail::RuntimeEventKind::kClockTick);
  unrelated.vertex = VertexId{2U};
  unrelated.receiver_index = 1U;
  const std::array sources{first_input, second_input, fenced_trigger, unrelated};
  auto coordinator = detail::RuntimeCandidateCoordinator::create(sources, 2U, 6U);
  ASSERT_TRUE(coordinator);

  detail::BoundedEventPortBinding first_binding{3U, first_input.source};
  detail::BoundedEventPortBinding second_binding{1U, second_input.source};
  detail::BoundedEventPortBinding trigger_binding{1U, fenced_trigger.source};
  detail::BoundedEventPortBinding unrelated_binding{1U, unrelated.source};
  ASSERT_TRUE(first_binding.push(runtime_event(first_input, 1U, 100)));
  ASSERT_TRUE(first_binding.push(runtime_event(first_input, 2U, 5)));
  auto terminal = runtime_event(first_input, 3U, 0);
  terminal.kind = detail::RuntimeEventKind::kInputEndOfStream;
  ASSERT_TRUE(first_binding.push(terminal));
  ASSERT_TRUE(second_binding.push(runtime_event(second_input, 1U, 50)));
  ASSERT_TRUE(trigger_binding.push(runtime_event(fenced_trigger, 1U, 1)));
  ASSERT_TRUE(unrelated_binding.push(runtime_event(unrelated, 1U, 10)));
  ASSERT_EQ(coordinator->ingest(first_binding), 3U);
  ASSERT_EQ(coordinator->ingest(second_binding), 1U);
  ASSERT_EQ(coordinator->ingest(trigger_binding), 1U);
  ASSERT_EQ(coordinator->ingest(unrelated_binding), 1U);

  const std::array fences{
      detail::RuntimeCandidateInputFence{first_input.vertex, first_input.source, first_input.port},
      detail::RuntimeCandidateInputFence{
          second_input.vertex, second_input.source, second_input.port},
  };
  auto freeze_constrained = [&]() { return coordinator->freeze_next_constrained(fences); };

  auto frozen = freeze_constrained();
  ASSERT_TRUE(frozen);
  EXPECT_EQ(frozen->witness.source, unrelated.source);
  EXPECT_EQ(frozen->witness.reference_time.timestamp_ns, 10);
  ASSERT_TRUE(coordinator->retire(*frozen));

  frozen = freeze_constrained();
  ASSERT_TRUE(frozen);
  EXPECT_EQ(frozen->witness.source, second_input.source);
  EXPECT_EQ(frozen->witness.reference_time.timestamp_ns, 50);
  ASSERT_TRUE(coordinator->retire(*frozen));

  frozen = freeze_constrained();
  ASSERT_TRUE(frozen);
  EXPECT_EQ(frozen->witness.source, first_input.source);
  EXPECT_EQ(frozen->witness.source_sequence, 1U);
  EXPECT_EQ(frozen->witness.reference_time.timestamp_ns, 100);
  ASSERT_TRUE(coordinator->retire(*frozen));

  frozen = freeze_constrained();
  ASSERT_TRUE(frozen);
  EXPECT_EQ(frozen->witness.source, first_input.source);
  EXPECT_EQ(frozen->witness.source_sequence, 2U);
  EXPECT_EQ(frozen->witness.reference_time.timestamp_ns, 5);
  ASSERT_TRUE(coordinator->retire(*frozen));

  // EOS shares the input prefix but is not a refill prerequisite. It cannot overtake either
  // arrival, and remains pending with the owner's trigger while the input fence is active.
  const auto fenced_terminal = freeze_constrained();
  ASSERT_FALSE(fenced_terminal);
  EXPECT_EQ(fenced_terminal.error().code, ErrorCode::kNotReady);

  auto end_of_stream = coordinator->freeze_next();
  ASSERT_TRUE(end_of_stream);
  EXPECT_EQ(end_of_stream->witness.kind, detail::RuntimeEventKind::kInputEndOfStream);
  EXPECT_EQ(end_of_stream->witness.source_sequence, 3U);
  EXPECT_EQ(end_of_stream->witness.reference_time.timestamp_ns, 0);
  ASSERT_TRUE(coordinator->retire(*end_of_stream));

  // Key assertion: the fenced owner's trigger remained pending until both input prefixes and EOS
  // were drained. Removing the fences restores its ordinary eligibility without losing identity.
  auto trigger = coordinator->freeze_next();
  ASSERT_TRUE(trigger);
  EXPECT_EQ(trigger->witness.source, fenced_trigger.source);
  EXPECT_EQ(trigger->witness.reference_time.timestamp_ns, 1);
  ASSERT_TRUE(coordinator->retire(*trigger));
}

TEST(DispatcherBindingTest, TriggerWitnessReferenceExcludesTerminalButPreservesData) {
  auto source = source_binding(0U, detail::RuntimeEventKind::kInputArrival, 2U);
  source.port = {};
  const PortId trigger_port{7U};
  auto coordinator = detail::RuntimeCandidateCoordinator::create({&source, 1U}, 1U, 2U);
  ASSERT_TRUE(coordinator);

  detail::BoundedEventPortBinding terminal_only{2U, source.source};
  auto terminal = runtime_event(source, 1U, 20);
  terminal.kind = detail::RuntimeEventKind::kInputEndOfStream;
  terminal.port = trigger_port;
  ASSERT_TRUE(terminal_only.push(terminal));
  ASSERT_EQ(coordinator->ingest(terminal_only), 1U);
  EXPECT_FALSE(coordinator->oldest_pending_nonterminal_trigger_reference(
      source.vertex, source.source, trigger_port));
  EXPECT_TRUE(coordinator->no_pending_trigger_witness(source.vertex, trigger_port));

  detail::BoundedEventPortBinding data{2U, source.source};
  auto arrival = runtime_event(source, 2U, 10);
  arrival.port = trigger_port;
  ASSERT_TRUE(data.push(arrival));
  ASSERT_EQ(coordinator->ingest(data), 1U);
  EXPECT_EQ(coordinator->oldest_pending_nonterminal_trigger_reference(
                source.vertex, source.source, trigger_port),
            (TimePoint{10, source.reference_clock}));
  EXPECT_FALSE(coordinator->no_pending_trigger_witness(source.vertex, trigger_port));
}

// REQ: RUNTIME-007
// One ordinary root followed by one compiler-proven causal successor preserves
// the root logical coordinate while advancing exactly one microstep. The
// successor witness retains its independent physical reference time.
TEST(DispatcherBindingTest, CausalSuccessorUsesTheExactParentLogicalCoordinate) {
  const std::array sources{
      source_binding(0U, detail::RuntimeEventKind::kClockTick),
      source_binding(1U, detail::RuntimeEventKind::kInputArrival),
  };
  auto coordinator = detail::RuntimeCandidateCoordinator::create(sources, 1U, sources.size());
  ASSERT_TRUE(coordinator);

  detail::BoundedEventPortBinding root_binding{1U, sources[0].source};
  ASSERT_TRUE(root_binding.push(runtime_event(sources[0], 1U, 100)));
  ASSERT_EQ(coordinator->ingest(root_binding), 1U);
  auto root = coordinator->freeze_next();
  ASSERT_TRUE(root);
  ASSERT_TRUE(coordinator->retire(*root));

  detail::BoundedEventPortBinding successor_binding{1U, sources[1].source};
  ASSERT_TRUE(successor_binding.push(runtime_event(sources[1], 1U, 250)));
  ASSERT_EQ(coordinator->ingest(successor_binding), 1U);
  const TagTime parent = root->candidate.ready.tag_time;
  auto successor = coordinator->freeze_causal_next(sources[1].source, parent);
  ASSERT_TRUE(successor);
  EXPECT_EQ(successor->candidate.ready.tag_time.logical_ns, parent.logical_ns);
  EXPECT_EQ(successor->candidate.ready.tag_time.microstep, parent.microstep + 1U);
  EXPECT_EQ(successor->witness.reference_time, (TimePoint{250, sources[1].reference_clock}));
  ASSERT_TRUE(coordinator->retire(*successor));
}

// REQ: RUNTIME-007
// The causal path may omit a repeated prefix scan only after its sole-credit proof excludes a
// held predecessor. The same EOS remains pending on rejection and becomes eligible after retire.
TEST(DispatcherBindingTest, CausalEndOfStreamRequiresRetiredSolePredecessor) {
  const auto source = source_binding(0U, detail::RuntimeEventKind::kInputArrival, 2U);
  auto coordinator = detail::RuntimeCandidateCoordinator::create({&source, 1U}, 1U, 2U);
  ASSERT_TRUE(coordinator);
  detail::BoundedEventPortBinding binding{2U, source.source};
  ASSERT_TRUE(binding.push(runtime_event(source, 1U, 100)));
  ASSERT_EQ(coordinator->ingest(binding), 1U);
  auto predecessor = coordinator->freeze_next();
  ASSERT_TRUE(predecessor);
  const TagTime parent = predecessor->candidate.ready.tag_time;

  auto terminal = runtime_event(source, 2U, 0);
  terminal.kind = detail::RuntimeEventKind::kInputEndOfStream;
  ASSERT_TRUE(binding.push(terminal));
  ASSERT_EQ(coordinator->ingest(binding), 1U);
  const auto held = coordinator->freeze_causal_next(source.source, parent);
  ASSERT_FALSE(held);
  EXPECT_EQ(held.error().code, ErrorCode::kProtocolError);
  EXPECT_EQ(coordinator->pending_count(), 1U);
  EXPECT_EQ(coordinator->outstanding_count(), 2U);
  const auto ordinary = coordinator->freeze_next();
  ASSERT_FALSE(ordinary);
  EXPECT_EQ(ordinary.error().code, ErrorCode::kNotReady);

  ASSERT_TRUE(coordinator->retire(*predecessor));
  auto successor = coordinator->freeze_causal_next(source.source, parent);
  ASSERT_TRUE(successor);
  EXPECT_EQ(successor->witness.kind, detail::RuntimeEventKind::kInputEndOfStream);
  EXPECT_EQ(successor->witness.source_sequence, 2U);
  EXPECT_EQ(successor->witness.reference_time, (TimePoint{0, source.reference_clock}));
  EXPECT_EQ(successor->candidate.ready.tag_time.logical_ns, parent.logical_ns);
  EXPECT_EQ(successor->candidate.ready.tag_time.microstep, parent.microstep + 1U);
  EXPECT_EQ(coordinator->pending_count(), 0U);
  EXPECT_EQ(coordinator->outstanding_count(), 1U);
  ASSERT_TRUE(coordinator->retire(*successor));
  EXPECT_EQ(coordinator->outstanding_count(), 0U);
}

// REQ: RUNTIME-007
// Causal cursor and microstep validation fail before consuming the pending
// successor, leaving ordinary terminal cleanup with exact source ownership.
TEST(DispatcherBindingTest, CausalSuccessorRejectsStaleOrExhaustedParentWithoutMutation) {
  const std::array sources{
      source_binding(0U, detail::RuntimeEventKind::kClockTick),
      source_binding(1U, detail::RuntimeEventKind::kInputArrival),
  };
  auto coordinator = detail::RuntimeCandidateCoordinator::create(sources, 1U, sources.size());
  ASSERT_TRUE(coordinator);

  detail::BoundedEventPortBinding root_binding{1U, sources[0].source};
  ASSERT_TRUE(root_binding.push(runtime_event(sources[0], 1U, 100)));
  ASSERT_EQ(coordinator->ingest(root_binding), 1U);
  auto root = coordinator->freeze_next();
  ASSERT_TRUE(root);
  ASSERT_TRUE(coordinator->retire(*root));

  detail::BoundedEventPortBinding successor_binding{1U, sources[1].source};
  ASSERT_TRUE(successor_binding.push(runtime_event(sources[1], 1U, 250)));
  ASSERT_EQ(coordinator->ingest(successor_binding), 1U);
  const TagTime parent = root->candidate.ready.tag_time;
  const auto stale = coordinator->freeze_causal_next(
      sources[1].source,
      TagTime{parent.logical_ns, static_cast<std::uint32_t>(parent.microstep + 1U), 0U});
  ASSERT_FALSE(stale);
  EXPECT_EQ(stale.error().code, ErrorCode::kProtocolError);
  EXPECT_EQ(coordinator->pending_count(), 1U);
  EXPECT_EQ(coordinator->outstanding_count(), 1U);

  const auto exhausted = coordinator->freeze_causal_next(
      sources[1].source, TagTime{parent.logical_ns, std::numeric_limits<std::uint32_t>::max(), 0U});
  ASSERT_FALSE(exhausted);
  EXPECT_EQ(exhausted.error().code, ErrorCode::kResourceExhausted);
  EXPECT_EQ(coordinator->pending_count(), 1U);
  EXPECT_EQ(coordinator->outstanding_count(), 1U);
  coordinator->cancel_all();
}

// REQ: RUNTIME-007
// One source retains an older accepted witness while its binding queues two successors. The
// source-local preflight permits exactly one successor even though the unrelated source leaves
// additional global coordinator capacity.
TEST(DispatcherBindingTest, SourceCreditBoundStopsDestructiveDrainBeforeQueuedSuccessor) {
  const std::array sources{
      source_binding(0U, detail::RuntimeEventKind::kInputArrival, 2U),
      source_binding(1U, detail::RuntimeEventKind::kClockTick, 2U),
  };
  auto coordinator = detail::RuntimeCandidateCoordinator::create(sources, 2U, 4U);
  ASSERT_TRUE(coordinator);

  detail::BoundedEventPortBinding first{1U, sources[0].source};
  ASSERT_TRUE(first.push(runtime_event(sources[0], 1U, 10)));
  ASSERT_EQ(coordinator->ingest(first), 1U);

  detail::BoundedEventPortBinding successors{2U, sources[0].source};
  ASSERT_TRUE(successors.push(runtime_event(sources[0], 2U, 20)));
  ASSERT_TRUE(successors.push(runtime_event(sources[0], 3U, 30)));
  auto credit = coordinator->source_credit_remaining(sources[0].source);
  ASSERT_TRUE(credit);
  ASSERT_EQ(*credit, 1U);
  // Key assertion: event 2 is accepted and event 3 remains binding-owned instead of being
  // destructively drained into a kBackpressured source.
  ASSERT_EQ(coordinator->ingest(successors, sources[0].source, *credit), 1U);

  auto first_frozen = coordinator->freeze_next();
  ASSERT_TRUE(first_frozen);
  EXPECT_EQ(first_frozen->witness.source_sequence, 1U);
  ASSERT_TRUE(coordinator->retire(*first_frozen));

  credit = coordinator->source_credit_remaining(sources[0].source);
  ASSERT_TRUE(credit);
  ASSERT_EQ(*credit, 1U);
  ASSERT_EQ(coordinator->ingest(successors, sources[0].source, *credit), 1U);
  for (std::uint64_t sequence = 2U; sequence <= 3U; ++sequence) {
    auto frozen = coordinator->freeze_next();
    ASSERT_TRUE(frozen);
    EXPECT_EQ(frozen->witness.source_sequence, sequence);
    ASSERT_TRUE(coordinator->retire(*frozen));
  }
  EXPECT_EQ(coordinator->outstanding_count(), 0U);
}

// REQ: RUNTIME-007
// A real no-wrap Tag sequence bound is distinct from late cross-source visibility. The first
// candidate consumes the final valid sequence and the next freeze preserves kResourceExhausted,
// which RunSession maps to the existing RUNTIME_TAG_CAPACITY_EXHAUSTED token.
TEST(DispatcherBindingTest, TrueTagSequenceExhaustionRetainsTheCapacityCause) {
  const auto source = source_binding(0U, detail::RuntimeEventKind::kClockTick);
  detail::TagAllocator tags{1U, std::numeric_limits<std::uint64_t>::max()};
  const std::array<detail::TagAllocator*, 1U> tag_allocators{&tags};
  auto coordinator =
      detail::RuntimeCandidateCoordinator::create_with_tags({&source, 1U}, tag_allocators, 1U);
  ASSERT_TRUE(coordinator);

  detail::BoundedEventPortBinding last_valid{1U, source.source};
  ASSERT_TRUE(last_valid.push(runtime_event(source, 1U, 10)));
  ASSERT_EQ(coordinator->ingest(last_valid), 1U);
  auto final_candidate = coordinator->freeze_next();
  ASSERT_TRUE(final_candidate);
  EXPECT_EQ(final_candidate->candidate.tag.sequence, std::numeric_limits<std::uint64_t>::max());
  ASSERT_TRUE(coordinator->retire(*final_candidate));

  detail::BoundedEventPortBinding exhausted_binding{1U, source.source};
  ASSERT_TRUE(exhausted_binding.push(runtime_event(source, 2U, 20)));
  ASSERT_EQ(coordinator->ingest(exhausted_binding), 1U);
  const auto exhausted = coordinator->freeze_next();
  ASSERT_FALSE(exhausted);
  EXPECT_EQ(exhausted.error().code, ErrorCode::kResourceExhausted);
  // The accepted witness remains coordinator-owned until terminal cleanup; exhaustion does not
  // misclassify or destructively discard it as a late-visibility failure.
  EXPECT_EQ(coordinator->pending_count(), 1U);
  EXPECT_EQ(coordinator->outstanding_count(), 1U);
  coordinator->cancel_all();
}

// REQ: RUNTIME-007
// Three sub-cases: wrong port -> kProtocolError; null port (multiplexed) -> port propagated;
// kFiniteSourceComplete event -> credit=false (not kNotFound).
TEST(DispatcherBindingTest, InputArrivalRequiresTheFrozenPortAndClosedSourcesExposeNoCredit) {
  const auto source = source_binding(0U, detail::RuntimeEventKind::kInputArrival);
  const std::array sources{source};

  auto wrong_port_coordinator =
      detail::RuntimeCandidateCoordinator::create(sources, 1U, sources.size());
  ASSERT_TRUE(wrong_port_coordinator);
  detail::BoundedEventPortBinding wrong_port_binding{1U};
  auto wrong_port = runtime_event(source, 1U, 10);
  wrong_port.port = PortId{source.port.value() + 1U};
  ASSERT_TRUE(wrong_port_binding.push(wrong_port));
  auto rejected = wrong_port_coordinator->ingest(wrong_port_binding);
  ASSERT_FALSE(rejected);
  EXPECT_EQ(rejected.error().code, ErrorCode::kProtocolError);

  auto multiplexed_source = source;
  multiplexed_source.port = {};
  const std::array multiplexed_sources{multiplexed_source};
  auto multiplexed_coordinator = detail::RuntimeCandidateCoordinator::create(
      multiplexed_sources, 1U, multiplexed_sources.size());
  ASSERT_TRUE(multiplexed_coordinator);
  detail::BoundedEventPortBinding multiplexed_binding{1U};
  auto owner_input = runtime_event(multiplexed_source, 1U, 10);
  owner_input.port = PortId{source.port.value() + 1U};
  ASSERT_TRUE(multiplexed_binding.push(owner_input));
  ASSERT_EQ(multiplexed_coordinator->ingest(multiplexed_binding), 1U);
  auto multiplexed_candidate = multiplexed_coordinator->freeze_next();
  ASSERT_TRUE(multiplexed_candidate);
  EXPECT_EQ(multiplexed_candidate->witness.port, owner_input.port);
  EXPECT_EQ(multiplexed_candidate->candidate.ready.stable_port_id, owner_input.port);
  EXPECT_TRUE(multiplexed_coordinator->retire(*multiplexed_candidate));

  auto closed_coordinator =
      detail::RuntimeCandidateCoordinator::create(sources, 1U, sources.size());
  ASSERT_TRUE(closed_coordinator);
  detail::BoundedEventPortBinding completion_binding{1U};
  auto completion = runtime_event(source, 1U, 10);
  completion.kind = detail::RuntimeEventKind::kFiniteSourceComplete;
  ASSERT_TRUE(completion_binding.push(completion));
  ASSERT_EQ(closed_coordinator->ingest(completion_binding), 1U);

  // Key assertion: a source that received kFiniteSourceComplete has credit=false, not kNotFound.
  auto credit = closed_coordinator->source_credit_remaining(source.source);
  ASSERT_TRUE(credit);
  EXPECT_FALSE(*credit);
}

// REQ: RUNTIME-020
// target_source[receiver=0] + unrelated_source[receiver=1] -> coordinator;
// cancel_receiver(0) removes only target events; unrelated survives and retires normally.
TEST(DispatcherBindingTest, RestartCancelsOnlyTargetPendingWitnessesBeforeGenerationAdvance) {
  auto target_source = source_binding(0U, detail::RuntimeEventKind::kInputArrival, 2U);
  auto unrelated_source = source_binding(1U, detail::RuntimeEventKind::kInputArrival, 2U);
  unrelated_source.receiver_index = 1U;
  const std::array sources{target_source, unrelated_source};
  auto coordinator = detail::RuntimeCandidateCoordinator::create(sources, 2U, 4U);
  ASSERT_TRUE(coordinator);

  detail::BoundedEventPortBinding binding{2U};
  ASSERT_TRUE(binding.push(runtime_event(target_source, 1U, 10)));
  ASSERT_TRUE(binding.push(runtime_event(unrelated_source, 1U, 11)));
  ASSERT_EQ(coordinator->ingest(binding), 2U);
  ASSERT_EQ(coordinator->pending_count(), 2U);
  ASSERT_EQ(coordinator->outstanding_count(), 2U);

  // Key setup: cancel_receiver with an undersized output span fails with kResourceExhausted.
  std::array<detail::RuntimeEvent, 0U> insufficient;
  const auto rejected = coordinator->cancel_receiver(0U, insufficient);
  ASSERT_FALSE(rejected);
  EXPECT_EQ(rejected.error().code, ErrorCode::kResourceExhausted);
  EXPECT_EQ(coordinator->pending_count(), 2U);
  EXPECT_EQ(coordinator->outstanding_count(), 2U);

  // Key assertion: exactly one event is canceled (target); unrelated source remains pending.
  std::array<detail::RuntimeEvent, 2U> canceled;
  const auto canceled_count = coordinator->cancel_receiver(0U, canceled);
  ASSERT_TRUE(canceled_count);
  ASSERT_EQ(*canceled_count, 1U);
  EXPECT_EQ(canceled[0].source, target_source.source);
  EXPECT_EQ(canceled[0].source_sequence, 1U);
  EXPECT_EQ(coordinator->pending_count(), 1U);
  EXPECT_EQ(coordinator->outstanding_count(), 1U);
  EXPECT_TRUE(coordinator->publish_restart(0U));

  auto unrelated = coordinator->freeze_next();
  ASSERT_TRUE(unrelated);
  EXPECT_EQ(unrelated->witness.source, unrelated_source.source);
  EXPECT_TRUE(coordinator->retire(*unrelated));
  EXPECT_EQ(coordinator->outstanding_count(), 0U);
}

// REQ: RUNTIME-002
// REQ: RUNTIME-004
// REQ: RUNTIME-007
// REQ: RUNTIME-041
// candidate[plan_order=0,receiver=0] + candidate[plan_order=1,receiver=1] -> dispatcher [serial];
// lower plan_vertex_order wins selection; concurrent select() blocked until complete().
TEST(DispatcherBindingTest, DispatcherOwnsOnePartitionLocalSerialSelection) {
  const SchedulerPlanRecord plan = scheduler_plan();
  auto dispatcher = detail::Dispatcher::create(plan, 2U);
  ASSERT_TRUE(dispatcher);
  const std::array candidates{
      detail::DispatcherCandidate{
          .ready = ready_candidate(20, 1U, 2U), .tag = Tag{2U, 1U, 0U}, .receiver_index = 1U},
      detail::DispatcherCandidate{
          .ready = ready_candidate(20, 0U, 1U), .tag = Tag{1U, 1U, 0U}, .receiver_index = 0U},
  };
  ASSERT_TRUE(dispatcher->freeze(candidates));

  auto first = dispatcher->select();
  ASSERT_TRUE(first);
  // Key assertion: plan_vertex_order=0 (receiver_index=0) is selected first; a second select()
  // while the first is active returns kInvalidArgument (serial constraint).
  EXPECT_EQ(first->receiver_index, 0U);
  EXPECT_EQ(first->tag, (Tag{1U, 1U, 0U}));
  EXPECT_EQ(dispatcher->select().error().code, ErrorCode::kInvalidArgument);
  EXPECT_TRUE(dispatcher->complete(*first));

  auto second = dispatcher->select();
  ASSERT_TRUE(second);
  EXPECT_EQ(second->receiver_index, 1U);
  EXPECT_EQ(second->tag, (Tag{2U, 1U, 0U}));
  EXPECT_TRUE(dispatcher->complete(*second));
  EXPECT_EQ(dispatcher->select().error().code, ErrorCode::kNotReady);
}

// REQ: RUNTIME-002
// REQ: RUNTIME-004
// REQ: RUNTIME-008
// candidate[receiver=0] + candidate[receiver=1] -> dispatcher; cancel_receiver(1) blocked while
// receiver=0 is active; succeeds after complete() and removes exactly one idle candidate.
TEST(DispatcherBindingTest, ReceiverRestartFenceCancelsOnlyIdleTargetCandidates) {
  const SchedulerPlanRecord plan = scheduler_plan();
  auto dispatcher = detail::Dispatcher::create(plan, 2U);
  ASSERT_TRUE(dispatcher);
  const std::array candidates{
      detail::DispatcherCandidate{
          .ready = ready_candidate(20, 0U, 1U), .tag = Tag{1U, 1U, 0U}, .receiver_index = 0U},
      detail::DispatcherCandidate{
          .ready = ready_candidate(20, 1U, 2U), .tag = Tag{1U, 1U, 0U}, .receiver_index = 1U},
  };
  ASSERT_TRUE(dispatcher->freeze(candidates));

  auto requester = dispatcher->select();
  ASSERT_TRUE(requester);
  // Key setup: receiver=0 is active; cancel_receiver(1) is blocked with kBackpressured.
  EXPECT_EQ(requester->receiver_index, 0U);
  EXPECT_EQ(dispatcher->cancel_receiver(1U).error().code, ErrorCode::kBackpressured);
  ASSERT_TRUE(dispatcher->complete(*requester));

  // Key assertion: after complete(), cancel_receiver(1) removes the one idle candidate.
  auto canceled = dispatcher->cancel_receiver(1U);
  ASSERT_TRUE(canceled);
  EXPECT_EQ(*canceled, 1U);
  EXPECT_EQ(dispatcher->active_count(), 0U);
  EXPECT_EQ(dispatcher->select().error().code, ErrorCode::kNotReady);
  auto repeated = dispatcher->cancel_receiver(1U);
  ASSERT_TRUE(repeated);
  EXPECT_EQ(*repeated, 0U);
}

// REQ: RUNTIME-002
// REQ: RUNTIME-004
// REQ: RUNTIME-007
// 6 sources (InputArrival,ClockTick,Readable,Notification,Doorbell,Spin) -> coordinator
//   -> dispatcher [serial]; events pushed in reverse order; freeze yields plan_vertex_order order.
TEST(DispatcherBindingTest, CoordinatorFreezesEverySourceKindInDeterministicOrder) {
  const std::array sources{
      source_binding(0U, detail::RuntimeEventKind::kInputArrival),
      source_binding(1U, detail::RuntimeEventKind::kClockTick),
      source_binding(2U, detail::RuntimeEventKind::kReadable),
      source_binding(3U, detail::RuntimeEventKind::kNotification),
      source_binding(4U, detail::RuntimeEventKind::kDoorbell),
      source_binding(5U, detail::RuntimeEventKind::kSpin),
  };
  detail::RuntimePhaseCounters counters;
  auto coordinator =
      detail::RuntimeCandidateCoordinator::create(sources, 1U, sources.size(), &counters);
  ASSERT_TRUE(coordinator);
  // Key setup: all 6 events pushed in reverse index order to confirm freeze order is by plan
  // ordinal, not arrival order.
  detail::BoundedEventPortBinding binding{sources.size()};
  for (std::size_t index = sources.size(); index-- > 0U;) {
    ASSERT_TRUE(binding.push(runtime_event(sources[index], 1U, 100)));
  }
  auto ingested = coordinator->ingest(binding);
  ASSERT_TRUE(ingested);
  EXPECT_EQ(*ingested, sources.size());

  std::array<detail::FrozenRuntimeCandidate, sources.size()> frozen;
  std::array<detail::DispatcherCandidate, sources.size()> ready;
  for (std::size_t index = 0; index < sources.size(); ++index) {
    auto candidate = coordinator->freeze_next();
    ASSERT_TRUE(candidate);
    frozen[index] = *candidate;
    ready[index] = candidate->candidate;
    EXPECT_EQ(candidate->witness.kind, sources[index].candidate_kind);
    EXPECT_EQ(candidate->candidate.tag, (Tag{index + 1U, 1U, 0U}));
    // Key assertion: tag.sequence==index+1 (monotonic); tag_time.order==index (plan_vertex_order).
    EXPECT_EQ(candidate->candidate.ready.tag_time,
              (TagTime{100, static_cast<std::uint32_t>(index), 0U}));
  }
  EXPECT_EQ(coordinator->freeze_next().error().code, ErrorCode::kNotReady);

  const auto phase_counts = counters.snapshot();
  const auto phase = [&](detail::RuntimePhaseCounter counter) {
    return phase_counts[detail::RuntimePhaseCounters::index(counter)];
  };
  EXPECT_EQ(phase(detail::RuntimePhaseCounter::kCandidateOrderComparisons), 15U);
  EXPECT_EQ(phase(detail::RuntimePhaseCounter::kCandidateOrderMoves), 0U);
  EXPECT_EQ(phase(detail::RuntimePhaseCounter::kCandidateTagTimeReservations), sources.size());

  const SchedulerPlanRecord plan = scheduler_plan();
  auto dispatcher = detail::Dispatcher::create(plan, ready.size());
  ASSERT_TRUE(dispatcher);
  ASSERT_TRUE(dispatcher->freeze(ready));
  for (std::size_t index = 0; index < ready.size(); ++index) {
    auto selected = dispatcher->select();
    ASSERT_TRUE(selected);
    EXPECT_EQ(selected->tag, (Tag{index + 1U, 1U, 0U}));
    ASSERT_TRUE(dispatcher->complete(*selected));
    ASSERT_TRUE(coordinator->retire(frozen[index]));
  }
  EXPECT_EQ(coordinator->outstanding_count(), 0U);

  // A completed READY set can be reused without allocating another set of
  // policy slots. Candidate identity and microstep ordering continue rather
  // than being donated by the previous retirements.
  detail::BoundedEventPortBinding second_binding{1U};
  ASSERT_TRUE(second_binding.push(runtime_event(sources[0], 2U, 101)));
  ASSERT_TRUE(coordinator->ingest(second_binding));
  auto second_candidate = coordinator->freeze_next();
  ASSERT_TRUE(second_candidate);
  EXPECT_EQ(second_candidate->candidate.tag, (Tag{7U, 1U, 0U}));
  EXPECT_EQ(second_candidate->candidate.ready.tag_time, (TagTime{101, 0U, 0U}));
  const std::array second_ready{second_candidate->candidate};
  ASSERT_TRUE(dispatcher->freeze(second_ready));
  auto selected = dispatcher->select();
  ASSERT_TRUE(selected);
  ASSERT_TRUE(dispatcher->complete(*selected));
  EXPECT_TRUE(coordinator->retire(*second_candidate));
}

// Source lookup is repeated across readiness credit, acceptance, and lifecycle queries, so its
// linear work must remain visible to opt-in performance attribution.
// REQ: RUNTIME-007
TEST(DispatcherBindingTest, CoordinatorPhaseCountersAttributeSourceLookupProbes) {
  constexpr std::size_t kSourceCount = 6U;
  std::array<detail::RuntimeSourceBindingRecord, kSourceCount> sources;
  for (std::size_t index = 0U; index < sources.size(); ++index) {
    sources[index] =
        source_binding(static_cast<std::uint32_t>(index), detail::RuntimeEventKind::kClockTick);
  }
  detail::RuntimePhaseCounters counters;
  auto coordinator =
      detail::RuntimeCandidateCoordinator::create(sources, 1U, sources.size(), &counters);
  ASSERT_TRUE(coordinator);

  // Key setup: query the first, last, and one absent source in the frozen inventory.
  EXPECT_TRUE(coordinator->source_credit_remaining(sources.front().source));
  EXPECT_TRUE(coordinator->source_credit_remaining(sources.back().source));
  auto absent = coordinator->source_credit_remaining(detail::make_source_id(100U));
  ASSERT_FALSE(absent);
  EXPECT_EQ(absent.error().code, ErrorCode::kNotFound);

  // Key assertion: attribution reports the exact 1 + N + N linear probes and no candidate work.
  const auto phase_counts = counters.snapshot();
  const auto phase = [&](detail::RuntimePhaseCounter counter) {
    return phase_counts[detail::RuntimePhaseCounters::index(counter)];
  };
  EXPECT_EQ(phase(detail::RuntimePhaseCounter::kCandidateSourceLookupProbes),
            1U + kSourceCount + kSourceCount);
  EXPECT_EQ(phase(detail::RuntimePhaseCounter::kCandidateOrderComparisons), 0U);
  EXPECT_EQ(phase(detail::RuntimePhaseCounter::kCandidateTagTimeReservations), 0U);
}

// Independently accepted sources may become visible in separate scans: a later reference can
// freeze before an earlier notification is drained. The witness retains its semantic coordinate,
// while TagTime advances monotonically at the current logical nanosecond.
// REQ: RUNTIME-007
TEST(DispatcherBindingTest, LateCrossSourceWitnessAdvancesTheTagTimeMicrostep) {
  const std::array sources{
      source_binding(0U, detail::RuntimeEventKind::kInputArrival),
      source_binding(1U, detail::RuntimeEventKind::kNotification),
  };
  auto coordinator = detail::RuntimeCandidateCoordinator::create(sources, 1U, sources.size());
  ASSERT_TRUE(coordinator);

  // Key setup: freeze and retire the input at t=200 before the independently accepted
  // notification at t=100 becomes visible to the coordinator.
  detail::BoundedEventPortBinding input_binding{1U};
  ASSERT_TRUE(input_binding.push(runtime_event(sources[0], 1U, 200)));
  ASSERT_EQ(coordinator->ingest(input_binding), 1U);
  auto input = coordinator->freeze_next();
  ASSERT_TRUE(input);
  EXPECT_EQ(input->candidate.ready.tag_time, (TagTime{200, 0U, 0U}));
  ASSERT_TRUE(coordinator->retire(*input));

  detail::BoundedEventPortBinding notification_binding{1U};
  ASSERT_TRUE(notification_binding.push(runtime_event(sources[1], 1U, 100)));
  ASSERT_EQ(coordinator->ingest(notification_binding), 1U);
  auto notification = coordinator->freeze_next();
  ASSERT_TRUE(notification);

  // Key assertion: the original t=100 reference remains exact, but the activation cannot be
  // backdated and therefore receives the next TagTime microstep at logical time 200.
  EXPECT_EQ(notification->witness.reference_time, (TimePoint{100, kRuntimeClock}));
  EXPECT_EQ(notification->candidate.ready.tag_time, (TagTime{200, 1U, 0U}));
  EXPECT_EQ(notification->candidate.tag, (Tag{2U, 1U, 0U}));
  EXPECT_TRUE(coordinator->retire(*notification));
}

// REQ: RUNTIME-007
// REQ: RUNTIME-018
// kInputArrival[t=100] + kWindowTimeout[t=100] -> coordinator; equal timestamps allowed;
// freeze preserves ingestion order and propagates the timeout's reference_time unchanged.
TEST(DispatcherBindingTest, CoordinatorAcceptsWindowTimeoutAfterEqualTimeInputArrival) {
  auto source = source_binding(0U, detail::RuntimeEventKind::kInputArrival, 2U);
  const std::array sources{source};
  auto coordinator = detail::RuntimeCandidateCoordinator::create(sources, 1U, 2U);
  ASSERT_TRUE(coordinator);
  detail::BoundedEventPortBinding binding{2U};
  // Key setup: both events share reference_time=100; the window timeout is pushed second.
  ASSERT_TRUE(binding.push(runtime_event(source, 1U, 100)));
  ASSERT_TRUE(binding.push(detail::RuntimeEvent{
      .kind = detail::RuntimeEventKind::kWindowTimeout,
      .source = source.source,
      .port = source.port,
      .source_decision_time = TimePoint{100, kRuntimeClock},
      .reference_time = TimePoint{100, kRuntimeClock},
      .source_generation = source.source_generation,
      .source_sequence = 2U,
  }));
  ASSERT_EQ(coordinator->ingest(binding), 2U);

  auto input = coordinator->freeze_next();
  ASSERT_TRUE(input);
  EXPECT_EQ(input->witness.kind, detail::RuntimeEventKind::kInputArrival);
  // Key assertion: second freeze yields kWindowTimeout with reference_time preserved at 100.
  auto timeout = coordinator->freeze_next();
  ASSERT_TRUE(timeout);
  EXPECT_EQ(timeout->witness.kind, detail::RuntimeEventKind::kWindowTimeout);
  EXPECT_EQ(timeout->witness.reference_time, (TimePoint{100, kRuntimeClock}));
  ASSERT_TRUE(coordinator->retire(*input));
  ASSERT_TRUE(coordinator->retire(*timeout));
}

// REQ: CONTRACT-003
// REQ: CONTRACT-006
// REQ: CONTRACT-007
// REQ: RUNTIME-002
// REQ: RUNTIME-004
// REQ: RUNTIME-007
// GoogleTest fatal assertions terminate this transaction test before optional access, but
// clang-tidy does not model the macro's control flow.
// NOLINTBEGIN(bugprone-unchecked-optional-access)
// One kInputArrival source[capacity=4] -> coordinator; 4 candidates exercised across scalar commit,
// no-fire abort, degraded drain, and bracketed-window snapshot scenarios.
TEST(DispatcherBindingTest, ActivationSelectionCommitsOnlyAcceptedTransactionsAndOwnsLifetime) {
  const std::array sources{
      source_binding(0U, detail::RuntimeEventKind::kInputArrival, 4U),
  };
  auto coordinator = detail::RuntimeCandidateCoordinator::create(sources, 1U, 4U);
  ASSERT_TRUE(coordinator);
  detail::BoundedEventPortBinding binding{4U};
  for (std::uint64_t sequence = 1U; sequence <= 4U; ++sequence) {
    ASSERT_TRUE(binding.push(
        runtime_event(sources[0], sequence, 100 + static_cast<std::int64_t>(sequence))));
  }
  ASSERT_TRUE(coordinator->ingest(binding));
  std::array<detail::FrozenRuntimeCandidate, 4> candidates;
  for (auto& candidate : candidates) {
    auto frozen = coordinator->freeze_next();
    ASSERT_TRUE(frozen);
    candidate = *frozen;
  }

  // Key setup: activation sized for 2 inputs and 8 samples; four distinct commit scenarios follow.
  auto activation = detail::ActivationSelectionState::create(2U, 8U);
  ASSERT_TRUE(activation);
  detail::TemporalInputCache cache{4U};
  ASSERT_TRUE(cache.push(temporal_sample(10, 1U, 1U)));
  auto scalar = cache.select_exact_transaction(TimePoint{10, kRuntimeClock});
  ASSERT_TRUE(scalar);
  ASSERT_TRUE(scalar->result().sample);
  const std::array scalar_input{scalar->result().selection};
  const std::array scalar_record{*scalar->result().sample};
  auto scalar_view = activation->begin(candidates[0], scalar_input, scalar_record);
  ASSERT_TRUE(scalar_view);
  EXPECT_EQ(scalar_view->disposition, detail::MissingAggregateDisposition::kNominal);
  ASSERT_TRUE(scalar->commit());
  EXPECT_EQ(cache.size(), 0U);
  // Key assertion: view() after commit() returns the selected sample; valid until retire().
  auto stable_after_commit = activation->view(scalar_view->generation);
  ASSERT_TRUE(stable_after_commit);
  ASSERT_EQ(stable_after_commit->selected.size(), 1U);
  EXPECT_EQ(stable_after_commit->selected[0], scalar_record[0]);
  ASSERT_TRUE(activation->retire(scalar_view->tag));
  EXPECT_EQ(activation->view(scalar_view->generation).error().code, ErrorCode::kStaleEndpoint);
  ASSERT_TRUE(coordinator->retire(candidates[0]));

  ASSERT_TRUE(cache.push(temporal_sample(20, 2U, 2U)));
  ASSERT_TRUE(cache.push(temporal_sample(21, 3U, 3U)));
  {
    auto drain = cache.snapshot_drain_all(3U, 4U);
    ASSERT_TRUE(drain);
    const std::array missing{
        InputSelection{InputState::kAbsent, SelectionReason::kNeverSeen},
    };
    auto no_fire = activation->begin(candidates[1], missing, drain->samples());
    ASSERT_TRUE(no_fire);
    EXPECT_EQ(no_fire->disposition, detail::MissingAggregateDisposition::kNoFire);
    // Deliberately do not commit: destruction aborts the provisional DrainAll.
    ASSERT_TRUE(activation->retire(no_fire->tag));
  }
  EXPECT_EQ(cache.size(), 2U);
  ASSERT_TRUE(coordinator->retire(candidates[1]));

  auto degraded_drain = cache.snapshot_drain_all(3U, 4U);
  ASSERT_TRUE(degraded_drain);
  const std::array degraded_inputs{
      InputSelection{InputState::kAbsent, SelectionReason::kNeverSeen},
      InputSelection{InputState::kSafeDefault, SelectionReason::kTooStale},
  };
  auto degraded = activation->begin(candidates[2], degraded_inputs, degraded_drain->samples());
  ASSERT_TRUE(degraded);
  EXPECT_EQ(degraded->disposition, detail::MissingAggregateDisposition::kDegraded);
  ASSERT_TRUE(degraded_drain->commit());
  EXPECT_EQ(cache.size(), 0U);
  ASSERT_TRUE(activation->retire(degraded->tag));
  ASSERT_TRUE(coordinator->retire(candidates[2]));

  detail::BracketedWindowEngine window{
      BracketedWindow{.lookbehind = 1U, .lookahead = 1U, .max_samples = 8U}, 16U, 2U};
  ASSERT_TRUE(window.push_sample(temporal_sample(5, 1U, 1U)));
  ASSERT_TRUE(window.push_anchor(temporal_sample(10, 1U, 1U)));
  ASSERT_TRUE(window.push_sample(temporal_sample(10, 2U, 2U)));
  ASSERT_TRUE(window.push_sample(temporal_sample(15, 3U, 3U)));
  ASSERT_TRUE(window.push_anchor(temporal_sample(20, 2U, 2U)));
  ASSERT_TRUE(window.push_sample(temporal_sample(20, 4U, 4U)));
  ASSERT_TRUE(window.push_sample(temporal_sample(25, 5U, 5U)));
  const auto* closed = window.front_closed();
  ASSERT_NE(closed, nullptr);
  const std::array window_input{closed->selection};
  auto window_view = activation->begin(candidates[3], window_input, closed->samples);
  ASSERT_TRUE(window_view);
  ASSERT_EQ(window_view->selected.size(), 5U);
  window.pop_closed();
  auto stable_after_pop = activation->view(window_view->generation);
  ASSERT_TRUE(stable_after_pop);
  EXPECT_EQ(stable_after_pop->selected.front().capture_time.timestamp_ns, 5);
  EXPECT_EQ(stable_after_pop->selected.back().capture_time.timestamp_ns, 25);
  ASSERT_TRUE(activation->retire(window_view->tag));
  ASSERT_TRUE(coordinator->retire(candidates[3]));
}

// REQ: CONTRACT-004
// REQ: CONTRACT-006
// REQ: RUNTIME-007
// An independently accepted t=100 witness becomes visible after t=200 was frozen. The activation
// preserves close=100 and semantic-reference=200 while TagTime 200.1 supplies scheduling order.
TEST(DispatcherBindingTest, LateVisibilityPreservesDistinctWitnessAndSemanticReferenceTimes) {
  const auto source = source_binding(0U, detail::RuntimeEventKind::kInputArrival);
  auto coordinator = detail::RuntimeCandidateCoordinator::create({&source, 1U}, 1U, 1U);
  ASSERT_TRUE(coordinator);
  detail::BoundedEventPortBinding binding{1U};

  ASSERT_TRUE(binding.push(runtime_event(source, 1U, 200)));
  ASSERT_TRUE(coordinator->ingest(binding));
  auto newer = coordinator->freeze_next();
  ASSERT_TRUE(newer);
  EXPECT_EQ(newer->witness.reference_time, (TimePoint{200, kRuntimeClock}));
  EXPECT_EQ(newer->candidate.ready.tag_time, (TagTime{200, 0U, 0U}));
  ASSERT_TRUE(coordinator->retire(*newer));

  ASSERT_TRUE(binding.push(runtime_event(source, 2U, 100)));
  ASSERT_TRUE(coordinator->ingest(binding));
  auto late = coordinator->freeze_next();
  ASSERT_TRUE(late);
  EXPECT_EQ(late->witness.reference_time, (TimePoint{100, kRuntimeClock}));
  EXPECT_EQ(late->candidate.ready.tag_time, (TagTime{200, 1U, 0U}));

  auto activation = detail::ActivationSelectionState::create(0U, 0U);
  ASSERT_TRUE(activation);
  // TagTime cannot authorize a semantic coordinate it has not reached.
  EXPECT_EQ(activation->begin(*late, {}, {}, TimePoint{201, kRuntimeClock}).error().code,
            ErrorCode::kInvalidArgument);
  auto selected = activation->begin(*late, {}, {}, TimePoint{200, kRuntimeClock});
  ASSERT_TRUE(selected);
  EXPECT_EQ(selected->close_time, (TimePoint{100, kRuntimeClock}));
  EXPECT_EQ(selected->reference_time, (TimePoint{200, kRuntimeClock}));
  EXPECT_EQ(selected->tag_time, (TagTime{200, 1U, 0U}));
  ASSERT_TRUE(activation->retire(selected->tag));
  ASSERT_TRUE(coordinator->retire(*late));
}

// REQ: RUNTIME-007
// REQ: CONTRACT-017
// NOLINTEND(bugprone-unchecked-optional-access)
// Three selectors exercised sequentially: kExact[reuse_last], kLatestWithin[max_age], kDrainAll.
// finish(false) leaves retained_count unchanged; finish(true) drains it.
TEST(DispatcherBindingTest, PlanBackedScalarAndDrainSelectorsCommitOnlyAcceptedCandidates) {
  auto exact_plan = selection_plan(InputSelectorKind::kExact, PortId{1U}, 4U, 1U);
  exact_plan.missing = TemporalRecordMissingKind::kReuseLast;
  exact_plan.reuse_max_age_ns = 50;
  auto exact = detail::RuntimeInputSelectionState::create(exact_plan);
  ASSERT_TRUE(exact);
  ASSERT_TRUE(exact->push(temporal_sample(10, 1U, 1U)));
  std::array<detail::TemporalSampleRecord, 4U> exact_claims{};
  auto exact_claim_count = exact->copy_payload_claims(exact_claims);
  ASSERT_TRUE(exact_claim_count);
  ASSERT_EQ(*exact_claim_count, 1U);
  EXPECT_EQ(exact_claims[0], temporal_sample(10, 1U, 1U));

  // Key setup: aborted transaction (finish=false) must leave the retained_count at 1.
  auto aborted = exact->begin(TimePoint{10, kRuntimeClock}, 1U, 1U);
  ASSERT_TRUE(aborted);
  EXPECT_EQ(aborted->selection().state, InputState::kSelected);
  ASSERT_EQ(aborted->selected().size(), 1U);
  ASSERT_TRUE(aborted->finish(false));
  EXPECT_EQ(exact->retained_count(), 1U);

  auto accepted = exact->begin(TimePoint{10, kRuntimeClock}, 1U, 1U);
  ASSERT_TRUE(accepted);
  ASSERT_TRUE(accepted->finish(true));
  EXPECT_EQ(exact->retained_count(), 0U);
  exact_claim_count = exact->copy_payload_claims(exact_claims);
  ASSERT_TRUE(exact_claim_count);
  ASSERT_EQ(*exact_claim_count, 1U);
  EXPECT_EQ(exact_claims[0], temporal_sample(10, 1U, 1U));

  // Key assertion: reuse_last makes a later begin() at t=20 return kReused for the t=10 sample.
  auto reused = exact->begin(TimePoint{20, kRuntimeClock}, 1U, 1U);
  ASSERT_TRUE(reused);
  EXPECT_EQ(reused->selection().state, InputState::kReused);
  ASSERT_EQ(reused->selected().size(), 1U);
  EXPECT_EQ(reused->selected().front().source_sequence, 1U);
  ASSERT_TRUE(reused->finish(true));

  ASSERT_TRUE(exact->reset());
  exact_claim_count = exact->copy_payload_claims(exact_claims);
  ASSERT_TRUE(exact_claim_count);
  EXPECT_EQ(*exact_claim_count, 0U);
  auto after_reset = exact->begin(TimePoint{20, kRuntimeClock}, 1U, 2U);
  ASSERT_TRUE(after_reset);
  EXPECT_EQ(after_reset->selection().state, InputState::kAbsent);
  EXPECT_TRUE(after_reset->selected().empty());
  ASSERT_TRUE(after_reset->finish(false));

  auto latest_plan = selection_plan(InputSelectorKind::kLatestWithin, PortId{3U}, 2U, 1U);
  latest_plan.max_age_ns = 10;
  auto latest = detail::RuntimeInputSelectionState::create(latest_plan);
  ASSERT_TRUE(latest);
  ASSERT_TRUE(latest->push(temporal_sample(40, 1U, 1U)));
  ASSERT_TRUE(latest->push(temporal_sample(45, 2U, 2U)));
  std::array<detail::TemporalSampleRecord, 2U> latest_claims{};
  auto latest_claim_count = latest->copy_payload_claims(latest_claims);
  ASSERT_TRUE(latest_claim_count);
  EXPECT_EQ(*latest_claim_count, 2U);
  auto latest_selected = latest->begin(TimePoint{48, kRuntimeClock}, 2U, 1U);
  ASSERT_TRUE(latest_selected);
  ASSERT_TRUE(latest_selected->finish(true));
  latest_claim_count = latest->copy_payload_claims(latest_claims);
  ASSERT_TRUE(latest_claim_count);
  EXPECT_EQ(*latest_claim_count, 1U);
  EXPECT_EQ(latest_claims[0], temporal_sample(40, 1U, 1U));

  // LatestWithin is consume-once: the committed t=45 sample cannot be selected again. The next
  // accepted transaction advances to the still-retained t=40 sample, then the cache is empty.
  auto next_latest = latest->begin(TimePoint{48, kRuntimeClock}, 3U, 1U);
  ASSERT_TRUE(next_latest);
  ASSERT_EQ(next_latest->selected().size(), 1U);
  EXPECT_EQ(next_latest->selected().front(), temporal_sample(40, 1U, 1U));
  ASSERT_TRUE(next_latest->finish(true));
  auto exhausted_latest = latest->begin(TimePoint{48, kRuntimeClock}, 4U, 1U);
  ASSERT_TRUE(exhausted_latest);
  EXPECT_EQ(exhausted_latest->selection().state, InputState::kAbsent);
  EXPECT_TRUE(exhausted_latest->selected().empty());
  ASSERT_TRUE(exhausted_latest->finish(true));

  auto reuse_latest_plan = selection_plan(InputSelectorKind::kLatestWithin, PortId{4U}, 2U, 1U);
  reuse_latest_plan.max_age_ns = 10;
  reuse_latest_plan.missing = TemporalRecordMissingKind::kReuseLast;
  reuse_latest_plan.reuse_max_age_ns = 50;
  auto reuse_latest = detail::RuntimeInputSelectionState::create(reuse_latest_plan);
  ASSERT_TRUE(reuse_latest);
  ASSERT_TRUE(reuse_latest->push(temporal_sample(50, 1U, 1U)));
  auto fresh_latest = reuse_latest->begin(TimePoint{50, kRuntimeClock}, 1U, 1U);
  ASSERT_TRUE(fresh_latest);
  EXPECT_EQ(fresh_latest->selection().state, InputState::kSelected);
  ASSERT_TRUE(fresh_latest->finish(true));
  auto held_latest = reuse_latest->begin(TimePoint{70, kRuntimeClock}, 2U, 1U);
  ASSERT_TRUE(held_latest);
  EXPECT_EQ(held_latest->selection().state, InputState::kReused);
  ASSERT_EQ(held_latest->selected().size(), 1U);
  EXPECT_EQ(held_latest->selected().front(), temporal_sample(50, 1U, 1U));
  ASSERT_TRUE(held_latest->finish(true));
  // A newer cache candidate can be too old for LatestWithin while the independent fallback
  // reaches its own inclusive age boundary. Reusing it must not renew that boundary.
  ASSERT_TRUE(reuse_latest->push(temporal_sample(80, 2U, 2U)));
  auto boundary_reuse = reuse_latest->begin(TimePoint{100, kRuntimeClock}, 3U, 1U);
  ASSERT_TRUE(boundary_reuse);
  EXPECT_EQ(boundary_reuse->selection().state, InputState::kReused);
  ASSERT_TRUE(boundary_reuse->finish(true));
  auto expired_reuse = reuse_latest->begin(TimePoint{101, kRuntimeClock}, 4U, 1U);
  ASSERT_TRUE(expired_reuse);
  EXPECT_EQ(expired_reuse->selection().state, InputState::kAbsent);
  EXPECT_TRUE(expired_reuse->selected().empty());
  ASSERT_TRUE(expired_reuse->finish(false));

  auto drain_plan = selection_plan(InputSelectorKind::kDrainAll, PortId{2U}, 3U, 3U);
  auto drain = detail::RuntimeInputSelectionState::create(drain_plan);
  ASSERT_TRUE(drain);
  ASSERT_TRUE(drain->push(temporal_sample(30, 1U, 1U)));
  ASSERT_TRUE(drain->push(temporal_sample(31, 2U, 2U)));
  std::array<detail::TemporalSampleRecord, 3U> drain_claims{};
  auto drain_claim_count = drain->copy_payload_claims(drain_claims);
  ASSERT_TRUE(drain_claim_count);
  EXPECT_EQ(*drain_claim_count, 2U);

  auto provisional = drain->begin(TimePoint{31, kRuntimeClock}, 2U, 1U);
  ASSERT_TRUE(provisional);
  ASSERT_EQ(provisional->selected().size(), 2U);
  ASSERT_TRUE(provisional->finish(false));
  EXPECT_EQ(drain->retained_count(), 2U);
  drain_claim_count = drain->copy_payload_claims(drain_claims);
  ASSERT_TRUE(drain_claim_count);
  EXPECT_EQ(*drain_claim_count, 2U);

  auto committed = drain->begin(TimePoint{31, kRuntimeClock}, 2U, 1U);
  ASSERT_TRUE(committed);
  ASSERT_EQ(committed->selected().size(), 2U);
  ASSERT_TRUE(committed->finish(true));
  EXPECT_EQ(drain->retained_count(), 0U);
  drain_claim_count = drain->copy_payload_claims(drain_claims);
  ASSERT_TRUE(drain_claim_count);
  EXPECT_EQ(*drain_claim_count, 0U);

  auto empty = drain->begin(TimePoint{32, kRuntimeClock}, 2U, 1U);
  ASSERT_TRUE(empty);
  EXPECT_EQ(empty->selection().state, InputState::kSelected);
  EXPECT_TRUE(empty->selected().empty());
  EXPECT_TRUE(empty->finish(true));
}

// REQ: LIFECYCLE-012
// REQ: CHAN-141
// A lifecycle-selected logical depth is enforced independently from the preallocated selector
// backing. Candidate changes remain private until committed, and changing the active bound never
// changes the payload-claim scratch bound.
TEST(DispatcherBindingTest, RateCandidateDepthSeparatesSelectorBackingFromActiveAdmission) {
  const auto plan = selection_plan(InputSelectorKind::kExact, PortId{1U}, 2U, 1U);
  auto state = detail::RuntimeInputSelectionState::create(
      plan, nullptr, 0U, detail::RuntimeQueueCapacity{.backing = 8U, .active = 2U});
  ASSERT_TRUE(state);
  EXPECT_EQ(state->payload_claim_capacity(), 8U);

  ASSERT_TRUE(state->push(temporal_sample(10, 1U, 1U)));
  ASSERT_TRUE(state->push(temporal_sample(11, 2U, 2U)));
  EXPECT_EQ(state->push(temporal_sample(12, 3U, 3U)).error().code, ErrorCode::kResourceExhausted);

  ASSERT_TRUE(state->prepare_rate_candidate_capacity(4U));
  EXPECT_EQ(state->push(temporal_sample(12, 3U, 3U)).error().code, ErrorCode::kResourceExhausted);
  state->discard_rate_candidate_capacity();
  EXPECT_EQ(state->push(temporal_sample(12, 3U, 3U)).error().code, ErrorCode::kResourceExhausted);

  ASSERT_TRUE(state->prepare_rate_candidate_capacity(4U));
  ASSERT_TRUE(state->commit_rate_candidate_capacity());
  ASSERT_TRUE(state->push(temporal_sample(12, 3U, 3U)));
  ASSERT_TRUE(state->push(temporal_sample(13, 4U, 4U)));
  EXPECT_EQ(state->push(temporal_sample(14, 5U, 5U)).error().code, ErrorCode::kResourceExhausted);
  EXPECT_EQ(state->payload_claim_capacity(), 8U);
}

// REQ: LIFECYCLE-012
// REQ: RUNTIME-007
// A published lifecycle restart may grow the logical depth inside preallocated backing. DrainAll
// must consume that generation's complete active prefix instead of retaining the smaller bound
// frozen into the original selection record.
TEST(DispatcherBindingTest, LifecycleDepthGrowthExpandsAutomaticDrainAllServiceBound) {
  const auto plan = selection_plan(InputSelectorKind::kDrainAll, PortId{1U}, 2U, 2U);
  auto state = detail::RuntimeInputSelectionState::create(
      plan, nullptr, 0U, detail::RuntimeQueueCapacity{.backing = 8U, .active = 2U});
  ASSERT_TRUE(state);

  ASSERT_TRUE(state->push(temporal_sample(10, 1U, 1U)));
  ASSERT_TRUE(state->push(temporal_sample(11, 2U, 2U)));
  auto initial = state->begin(TimePoint{11, kRuntimeClock}, 2U, 1U);
  ASSERT_TRUE(initial);
  ASSERT_EQ(initial->selected().size(), 2U);
  ASSERT_TRUE(initial->finish(true));

  ASSERT_TRUE(state->prepare_rate_candidate_capacity(4U));
  ASSERT_TRUE(state->commit_rate_candidate_capacity());
  ASSERT_TRUE(state->push(temporal_sample(20, 3U, 3U)));
  ASSERT_TRUE(state->push(temporal_sample(21, 4U, 4U)));
  ASSERT_TRUE(state->push(temporal_sample(22, 5U, 5U)));
  ASSERT_TRUE(state->push(temporal_sample(23, 6U, 6U)));

  auto restarted = state->begin(TimePoint{23, kRuntimeClock}, 6U, 2U);
  ASSERT_TRUE(restarted) << restarted.error().message;
  ASSERT_EQ(restarted->selected().size(), 4U);
  EXPECT_EQ(restarted->selected().front().source_sequence, 3U);
  EXPECT_EQ(restarted->selected().back().source_sequence, 6U);
  ASSERT_TRUE(restarted->finish(true));
  EXPECT_EQ(state->retained_count(), 0U);
}

// REQ: LIFECYCLE-012
// REQ: RUNTIME-007
// Selection scratch is sized from the largest batch that Runtime can materialize. Automatic
// DrainAll follows the immutable lifecycle backing ceiling, while explicitly bounded and scalar
// selectors retain their compiler-frozen limits.
TEST(DispatcherBindingTest, MaximumSelectedCapacityDistinguishesAutomaticAndExplicitBounds) {
  const auto automatic_plan = selection_plan(InputSelectorKind::kDrainAll, PortId{1U}, 2U, 2U);
  auto automatic = detail::RuntimeInputSelectionState::create(
      automatic_plan, nullptr, 0U, detail::RuntimeQueueCapacity{.backing = 8U, .active = 2U});
  ASSERT_TRUE(automatic);
  EXPECT_EQ(automatic->maximum_selected_capacity(), 8U);

  auto explicit_plan = selection_plan(InputSelectorKind::kBracketedWindow, PortId{2U}, 8U, 3U);
  explicit_plan.lookbehind = 1U;
  explicit_plan.lookahead = 1U;
  const BracketedWindowPlanRecord window{
      .owner = OperatorRef{1U, 77U, 0U},
      .input = PortId{2U},
      .anchor = PortId{1U},
      .lookbehind = 1U,
      .lookahead = 1U,
      .max_samples = 3U,
      .event_clock = kRuntimeClock,
      .queue_capacity = 8U,
      .cache_capacity = 8U,
      .producer_max_emits_per_compute = 1U,
      .retained_overlap_capacity = 2U,
  };
  auto explicit_bound = detail::RuntimeInputSelectionState::create(explicit_plan, &window, 2U);
  ASSERT_TRUE(explicit_bound);
  EXPECT_EQ(explicit_bound->maximum_selected_capacity(), 3U);

  const auto scalar_plan = selection_plan(InputSelectorKind::kExact, PortId{3U}, 8U, 1U);
  auto scalar = detail::RuntimeInputSelectionState::create(scalar_plan);
  ASSERT_TRUE(scalar);
  EXPECT_EQ(scalar->maximum_selected_capacity(), 1U);
}

// REQ: RUNTIME-007
// An explicit selected-capacity bound remains an overflow boundary even when the retained queue is
// larger. A failed snapshot must neither consume the prefix nor leave a stale transaction active.
TEST(DispatcherBindingTest, SelectedCapacityBoundRejectsOverflowAndPreservesRetainedPrefix) {
  detail::TemporalInputCache cache{4U};
  ASSERT_TRUE(cache.push(temporal_sample(10, 1U, 1U)));
  ASSERT_TRUE(cache.push(temporal_sample(11, 2U, 2U)));
  ASSERT_TRUE(cache.push(temporal_sample(12, 3U, 3U)));

  auto overflow = cache.snapshot_drain_all(3U, 2U);
  ASSERT_FALSE(overflow);
  EXPECT_EQ(overflow.error().code, ErrorCode::kResourceExhausted);
  EXPECT_EQ(cache.size(), 3U);

  ASSERT_TRUE(cache.evict(3U));
  auto exact_bound = cache.snapshot_drain_all(3U, 2U);
  ASSERT_TRUE(exact_bound);
  ASSERT_EQ(exact_bound->samples().size(), 2U);
  ASSERT_TRUE(exact_bound->commit());
  EXPECT_EQ(cache.size(), 0U);
}

// REQ: LIFECYCLE-012
// Restart publication validates every selector before mutating any of them. A later selector that
// cannot shrink below retained history therefore leaves an earlier selector on its old generation.
TEST(DispatcherBindingTest, CandidateCapacityValidationPreventsEarlierPartialPublication) {
  const auto earlier_plan = selection_plan(InputSelectorKind::kExact, PortId{1U}, 1U, 1U);
  auto earlier = detail::RuntimeInputSelectionState::create(
      earlier_plan, nullptr, 0U, detail::RuntimeQueueCapacity{.backing = 2U, .active = 1U});
  ASSERT_TRUE(earlier);
  ASSERT_TRUE(earlier->push(temporal_sample(10, 1U, 1U)));
  ASSERT_TRUE(earlier->prepare_rate_candidate_capacity(2U));

  const auto later_plan = selection_plan(InputSelectorKind::kExact, PortId{2U}, 2U, 1U);
  auto later = detail::RuntimeInputSelectionState::create(later_plan);
  ASSERT_TRUE(later);
  ASSERT_TRUE(later->push(temporal_sample(10, 1U, 1U)));
  ASSERT_TRUE(later->push(temporal_sample(11, 2U, 2U)));
  ASSERT_TRUE(later->prepare_rate_candidate_capacity(1U));

  EXPECT_FALSE(commit_two_ordinary_selectors(*earlier, *later));

  // Key assertion: the production closure performs no apply pass after the later validation
  // fails, so the first state still enforces depth one instead of exposing its staged depth two.
  EXPECT_EQ(earlier->push(temporal_sample(11, 2U, 2U)).error().code, ErrorCode::kResourceExhausted);
  earlier->discard_rate_candidate_capacity();
  later->discard_rate_candidate_capacity();
}

// REQ: RUNTIME-017
// kLatestWithin[capacity=1]: the drop-oldest reconcile releases its superseded candidate through
// evict_retained, which must surrender both the record and its payload claim so the caller can
// retire the transport slot, and must leave the cache able to admit the newer arrival.
TEST(DispatcherBindingTest, EvictRetainedSurrendersTheNamedScalarRecordAndItsPayloadClaim) {
  auto plan = selection_plan(InputSelectorKind::kLatestWithin, PortId{1U}, 1U, 1U);
  plan.max_age_ns = 1000;
  plan.missing = TemporalRecordMissingKind::kSkip;
  auto state = detail::RuntimeInputSelectionState::create(plan);
  ASSERT_TRUE(state);
  ASSERT_TRUE(state->push(temporal_sample(10, 1U, 1U)));

  std::array<detail::TemporalSampleRecord, 2U> claims{};
  auto claim_count = state->copy_payload_claims(claims);
  ASSERT_TRUE(claim_count);
  ASSERT_EQ(*claim_count, 1U);
  EXPECT_EQ(claims.front().arrival_sequence, 1U);

  // Key assertion: the claim disappears with the record, which is what lets the reconcile hand the
  // transport slot back through retire_samples() without stranding an observer.
  ASSERT_TRUE(state->evict_retained(1U));
  EXPECT_EQ(state->retained_count(), 0U);
  claim_count = state->copy_payload_claims(claims);
  ASSERT_TRUE(claim_count);
  EXPECT_EQ(*claim_count, 0U);
  // A depth-one cache that still refused the successor would convert a stale selection into a
  // permanently missing one.
  ASSERT_TRUE(state->push(temporal_sample(30, 2U, 2U)));
  auto selected = state->begin(TimePoint{30, kRuntimeClock}, 2U, 1U);
  ASSERT_TRUE(selected);
  EXPECT_EQ(selected->selection().state, InputState::kSelected);
  ASSERT_EQ(selected->selected().size(), 1U);
  EXPECT_EQ(selected->selected().front().source_sequence, 2U);
  ASSERT_TRUE(selected->finish(true));

  // Evicting an identity the cache never held is refused rather than silently ignored.
  const auto absent = state->evict_retained(99U);
  ASSERT_FALSE(absent);
  EXPECT_EQ(absent.error().code, ErrorCode::kNotFound);
}

// REQ: CHAN-131
// A ReuseLast selector reports its reuse retention through copy_payload_claims, but that record
// lives in retained_ rather than in the cache: commit() erases the cache entry and republishes it
// as the reuse copy. evict_retained only inspects the cache, so the reconcile sees exactly one
// claim, passes its depth-one gate, and then cannot evict the record that claim names.
//
// Key assertion: the two disagree. This is the cheapest possible reproduction of the permanent
// depth-one ReuseLast stall -- claims reports 1 while eviction reports kNotFound -- and it needs no
// graph, no transport, and no timing.
TEST(DispatcherBindingTest, EvictRetainedCannotReachAReuseLastRetentionItsClaimReports) {
  auto plan = selection_plan(InputSelectorKind::kLatestWithin, PortId{1U}, 1U, 1U);
  plan.max_age_ns = 1000;
  plan.missing = TemporalRecordMissingKind::kReuseLast;
  plan.reuse_max_age_ns = 1000000;
  auto state = detail::RuntimeInputSelectionState::create(plan);
  ASSERT_TRUE(state);
  ASSERT_TRUE(state->push(temporal_sample(10, 1U, 1U)));

  // Accept the record, which moves it out of the cache and into the reuse retention.
  auto selected = state->begin(TimePoint{10, kRuntimeClock}, 1U, 1U);
  ASSERT_TRUE(selected);
  ASSERT_EQ(selected->selection().state, InputState::kSelected);
  ASSERT_TRUE(selected->finish(true));
  EXPECT_EQ(state->retained_count(), 0U);

  std::array<detail::TemporalSampleRecord, 2U> claims{};
  const auto claim_count = state->copy_payload_claims(claims);
  ASSERT_TRUE(claim_count);
  ASSERT_EQ(*claim_count, 1U) << "the reuse retention is reported as a live payload claim";
  EXPECT_EQ(claims.front().arrival_sequence, 1U);

  // Key assertion: the claim names a record eviction cannot reach, so any reconcile gated on
  // "exactly one claim" is a no-op here while the transport slot stays pinned by that claim.
  const auto evicted = state->evict_retained(claims.front().arrival_sequence);
  ASSERT_FALSE(evicted) << "eviction unexpectedly reached the reuse retention";
  EXPECT_EQ(evicted.error().code, ErrorCode::kNotFound);
}

// REQ: RUNTIME-050
TEST(DispatcherBindingTest, ReuseReplacementTransactionFencesSelectionAndRestoresOnRollback) {
  auto plan = selection_plan(InputSelectorKind::kLatestWithin, PortId{1U}, 1U, 1U);
  plan.max_age_ns = 1000;
  plan.missing = TemporalRecordMissingKind::kReuseLast;
  plan.reuse_max_age_ns = 1000000;
  auto state = detail::RuntimeInputSelectionState::create(plan);
  ASSERT_TRUE(state);
  const auto retained = temporal_sample(10, 1U, 1U);
  ASSERT_TRUE(state->push(retained));
  auto selected = state->begin(TimePoint{10, kRuntimeClock}, 1U, 1U);
  ASSERT_TRUE(selected);
  ASSERT_TRUE(selected->finish(true));

  {
    auto replacement = state->reserve_reuse_replacement(retained);
    ASSERT_TRUE(replacement);
    auto concurrent = state->begin(TimePoint{11, kRuntimeClock}, 1U, 1U);
    ASSERT_FALSE(concurrent);
    EXPECT_EQ(concurrent.error().code, ErrorCode::kBackpressured);
    auto reset = state->reset();
    ASSERT_FALSE(reset);
    EXPECT_EQ(reset.error().code, ErrorCode::kBackpressured);
  }
  std::array<detail::TemporalSampleRecord, 1U> claims{};
  auto restored = state->copy_payload_claims(claims);
  ASSERT_TRUE(restored);
  ASSERT_EQ(*restored, 1U);
  EXPECT_EQ(claims.front(), retained);

  auto replacement = state->reserve_reuse_replacement(retained);
  ASSERT_TRUE(replacement);
  replacement->commit();
  auto committed = state->copy_payload_claims(claims);
  ASSERT_TRUE(committed);
  EXPECT_EQ(*committed, 0U);
}

// REQ: RUNTIME-050
TEST(DispatcherBindingTest, CacheReplacementTransactionRestoresOrCommitsOneExactRecord) {
  auto plan = selection_plan(InputSelectorKind::kLatestWithin, PortId{1U}, 1U, 1U);
  plan.max_age_ns = 1000;
  auto state = detail::RuntimeInputSelectionState::create(plan);
  ASSERT_TRUE(state);
  const auto retained = temporal_sample(10, 1U, 1U);
  ASSERT_TRUE(state->push(retained));

  {
    auto replacement = state->reserve_cache_replacement(retained);
    ASSERT_TRUE(replacement);
    auto concurrent = state->begin(TimePoint{11, kRuntimeClock}, 1U, 1U);
    ASSERT_FALSE(concurrent);
    EXPECT_EQ(concurrent.error().code, ErrorCode::kBackpressured);
  }
  std::array<detail::TemporalSampleRecord, 1U> claims{};
  auto restored = state->copy_payload_claims(claims);
  ASSERT_TRUE(restored);
  ASSERT_EQ(*restored, 1U);
  EXPECT_EQ(claims.front(), retained);

  auto replacement = state->reserve_cache_replacement(retained);
  ASSERT_TRUE(replacement);
  replacement->commit();
  EXPECT_EQ(state->retained_count(), 0U);
  EXPECT_TRUE(state->push(temporal_sample(20, 2U, 2U)));
}

// REQ: RUNTIME-043
// REQ: RUNTIME-050
// Capture absence is an arrival fact even though it cannot identify the selected payload. A
// rejected activation preserves that evidence; an accepted ReuseLast activation consumes it while
// retaining the independently selected prior payload.
TEST(DispatcherBindingTest, ReuseLastConsumesCaptureAbsentEvidenceOnlyOnAcceptedCommit) {
  auto plan = selection_plan(InputSelectorKind::kLatestWithin, PortId{1U}, 2U, 1U);
  plan.max_age_ns = 1000;
  plan.missing = TemporalRecordMissingKind::kReuseLast;
  plan.reuse_max_age_ns = 1000000;
  auto state = detail::RuntimeInputSelectionState::create(plan);
  ASSERT_TRUE(state);
  const auto retained = temporal_sample(10, 1U, 1U);
  ASSERT_TRUE(state->push(retained));
  auto selected = state->begin(TimePoint{10, kRuntimeClock}, 1U, 1U);
  ASSERT_TRUE(selected);
  ASSERT_TRUE(selected->finish(true));

  auto absent_capture = temporal_sample(20, 2U, 2U);
  absent_capture.capture_time = {};
  ASSERT_TRUE(state->push(absent_capture));
  {
    auto reused = state->begin(TimePoint{20, kRuntimeClock}, 2U, 1U);
    ASSERT_TRUE(reused);
    EXPECT_EQ(reused->selection().state, InputState::kReused);
    EXPECT_EQ(reused->selection().reason, SelectionReason::kCaptureTimeAbsent);
    ASSERT_EQ(reused->selected().size(), 1U);
    EXPECT_EQ(reused->selected().front(), retained);
    ASSERT_TRUE(reused->finish(false));
  }
  std::array<detail::TemporalSampleRecord, 2U> claims{};
  auto rolled_back = state->copy_payload_claims(claims);
  ASSERT_TRUE(rolled_back);
  EXPECT_EQ(*rolled_back, 2U);

  auto reused = state->begin(TimePoint{20, kRuntimeClock}, 2U, 1U);
  ASSERT_TRUE(reused);
  ASSERT_TRUE(reused->finish(true));
  auto committed = state->copy_payload_claims(claims);
  ASSERT_TRUE(committed);
  ASSERT_EQ(*committed, 1U);
  EXPECT_EQ(claims.front(), retained);
}

// REQ: RUNTIME-017
// kBracketedWindow: a window owns its retained set until closure, so the reconcile must be refused
// outright rather than removing a record from underneath a pending or closed window.
TEST(DispatcherBindingTest, EvictRetainedIsRefusedForABracketedSelector) {
  auto selection = selection_plan(InputSelectorKind::kBracketedWindow, PortId{2U}, 8U, 5U);
  selection.lookbehind = 1U;
  selection.lookahead = 1U;
  const BracketedWindowPlanRecord window{
      .owner = OperatorRef{1U, 77U, 0U},
      .input = PortId{2U},
      .anchor = PortId{1U},
      .lookbehind = 1U,
      .lookahead = 1U,
      .max_samples = 5U,
      .event_clock = kRuntimeClock,
      .queue_capacity = 8U,
      .cache_capacity = 8U,
      .producer_max_emits_per_compute = 1U,
      .retained_overlap_capacity = 2U,
  };
  auto state = detail::RuntimeInputSelectionState::create(selection, &window, 2U);
  ASSERT_TRUE(state);
  ASSERT_TRUE(state->push(temporal_sample(5, 1U, 1U)));

  // Key assertion: refused by selector family, not by whether this particular record happens to be
  // reachable from a window right now.
  const auto refused = state->evict_retained(1U);
  ASSERT_FALSE(refused);
  EXPECT_EQ(refused.error().code, ErrorCode::kNotSupported);

  // The window's payload claim must survive the refusal. retained_count() would not show this: a
  // bracketed selector keeps its samples in the window engine, so cache_ -- which is what
  // retained_count() reports -- is empty here whether or not the claim was stripped.
  std::array<detail::TemporalSampleRecord, 4U> claims{};
  const auto claim_count = state->copy_payload_claims(claims);
  ASSERT_TRUE(claim_count);
  EXPECT_EQ(*claim_count, 1U);
  EXPECT_EQ(claims.front().arrival_sequence, 1U);
}

// REQ: RUNTIME-017
// kExact: an open transaction is reading the retained set, so eviction must be refused until it
// resolves; the reconcile treats that refusal as a no-op and retries on a later pass.
TEST(DispatcherBindingTest, EvictRetainedIsRefusedWhileASelectionTransactionIsOpen) {
  auto plan = selection_plan(InputSelectorKind::kExact, PortId{1U}, 2U, 1U);
  plan.missing = TemporalRecordMissingKind::kSkip;
  auto state = detail::RuntimeInputSelectionState::create(plan);
  ASSERT_TRUE(state);
  ASSERT_TRUE(state->push(temporal_sample(10, 1U, 1U)));

  auto open = state->begin(TimePoint{10, kRuntimeClock}, 1U, 1U);
  ASSERT_TRUE(open);
  ASSERT_EQ(open->selected().size(), 1U);

  const auto refused = state->evict_retained(1U);
  ASSERT_FALSE(refused);
  EXPECT_EQ(refused.error().code, ErrorCode::kBackpressured);

  // Key assertion: the refusal leaves the transaction's view intact, so aborting still restores the
  // record for a later reconcile rather than losing it to a half-applied release.
  ASSERT_TRUE(open->finish(false));
  EXPECT_EQ(state->retained_count(), 1U);
  ASSERT_TRUE(state->evict_retained(1U));
  EXPECT_EQ(state->retained_count(), 0U);
}

// REQ: CHAN-136
// Once EOS is final, unobservable scalar cache and ReuseLast history must release their typed
// Channel claims without resetting the generation's arrival-order fence.
TEST(DispatcherBindingTest, ScalarEndOfStreamReleasesUnobservableHistoryWithoutGenerationReset) {
  auto plan = selection_plan(InputSelectorKind::kExact, PortId{1U}, 3U, 1U);
  plan.missing = TemporalRecordMissingKind::kReuseLast;
  plan.reuse_max_age_ns = 100;
  auto state = detail::RuntimeInputSelectionState::create(plan);
  ASSERT_TRUE(state);
  ASSERT_TRUE(state->push(temporal_sample(10, 1U, 1U)));
  ASSERT_TRUE(state->push(temporal_sample(20, 2U, 2U)));
  ASSERT_TRUE(state->push(temporal_sample(30, 3U, 3U)));

  EXPECT_EQ(state->release_terminal_history().error().code, ErrorCode::kInvalidArgument);
  auto middle = state->begin(TimePoint{20, kRuntimeClock}, 3U, 1U);
  ASSERT_TRUE(middle);
  EXPECT_EQ(state->release_terminal_history().error().code, ErrorCode::kBackpressured);
  ASSERT_TRUE(middle->finish(true));
  auto final = state->begin(TimePoint{30, kRuntimeClock}, 3U, 1U);
  ASSERT_TRUE(final);
  ASSERT_TRUE(final->finish(true));

  std::array<detail::TemporalSampleRecord, 4U> claims{};
  auto claim_count = state->copy_payload_claims(claims);
  ASSERT_TRUE(claim_count);
  ASSERT_EQ(*claim_count, 2U);
  EXPECT_EQ(claims[0], temporal_sample(10, 1U, 1U));
  EXPECT_EQ(claims[1], temporal_sample(30, 3U, 3U));

  ASSERT_TRUE(state->close_end_of_stream());
  ASSERT_TRUE(state->release_terminal_history());
  EXPECT_EQ(state->retained_count(), 0U);
  claim_count = state->copy_payload_claims(claims);
  ASSERT_TRUE(claim_count);
  EXPECT_EQ(*claim_count, 0U);
  EXPECT_EQ(state->push(temporal_sample(40, 4U, 3U)).error().code, ErrorCode::kInvalidArgument);
}

// REQ: RUNTIME-007
// kLatestWithin[max_age=5ns]: sample at t=10, query at t=20 -> absent; after extending max_age
// to 10ns at runtime, the same query returns kSelected without re-pushing the sample.
TEST(DispatcherBindingTest, LatestWithinSelectorUsesConfiguredMaxAge) {
  auto plan = selection_plan(InputSelectorKind::kLatestWithin, PortId{1U}, 2U, 1U);
  plan.max_age_ns = 5;
  plan.missing = TemporalRecordMissingKind::kSkip;
  auto state = detail::RuntimeInputSelectionState::create(plan);
  ASSERT_TRUE(state);
  ASSERT_TRUE(state->push(temporal_sample(10, 1U, 1U)));

  // Key setup: initial max_age=5ns; the 10-ns gap makes t=10 sample invisible at t=20.
  auto generation_zero = state->begin(TimePoint{20, kRuntimeClock}, 1U, 1U);
  ASSERT_TRUE(generation_zero);
  EXPECT_EQ(generation_zero->selection().state, InputState::kAbsent);
  EXPECT_TRUE(generation_zero->selected().empty());
  ASSERT_TRUE(generation_zero->finish(false));

  // Key assertion: after set_latest_within_max_age(10ns), the same t=20 query returns kSelected.
  ASSERT_TRUE(state->set_latest_within_max_age(std::chrono::nanoseconds{10}));
  auto successor = state->begin(TimePoint{20, kRuntimeClock}, 1U, 1U);
  ASSERT_TRUE(successor);
  EXPECT_EQ(successor->selection().state, InputState::kSelected);
  ASSERT_EQ(successor->selected().size(), 1U);
  EXPECT_EQ(successor->selected().front().source_sequence, 1U);
  ASSERT_TRUE(successor->finish(true));
}

// REQ: RUNTIME-007
// kExact[reuse_last]: relocating the owner state while a transaction is in-flight orphans the
// transaction's cache pointer; finish(true) must detect and report kFailure without UB.
TEST(DispatcherBindingTest, FailedScalarCommitDoesNotPublishReuseLastRetention) {
  auto plan = selection_plan(InputSelectorKind::kExact, PortId{1U}, 1U, 1U);
  plan.missing = TemporalRecordMissingKind::kReuseLast;
  plan.reuse_max_age_ns = 50;
  auto state = detail::RuntimeInputSelectionState::create(plan);
  ASSERT_TRUE(state);
  ASSERT_TRUE(state->push(temporal_sample(10, 1U, 1U)));

  // Key setup: begin() acquires a transaction while the state is still alive; then state is moved.
  auto selected = state->begin(TimePoint{10, kRuntimeClock}, 1U, 1U);
  ASSERT_TRUE(selected);
  ASSERT_EQ(selected->selected().size(), 1U);

  // Relocating the owner after selection moves the selected cache entry away while leaving the
  // transaction bound to the original cache. This exercises the checked destructive-commit
  // failure path without a test-only mutation hook.
  const detail::RuntimeInputSelectionState relocated{std::move(*state)};
  EXPECT_EQ(relocated.retained_count(), 1U);
  // Key assertion: finish(true) on an orphaned transaction returns kFailure without retaining.
  const auto committed = selected->finish(true);
  ASSERT_FALSE(committed);
  EXPECT_EQ(committed.error().code, ErrorCode::kFailure);

  std::array<detail::TemporalSampleRecord, 2U> claims{};
  const auto claim_count = state->copy_payload_claims(claims);
  ASSERT_TRUE(claim_count);
  EXPECT_EQ(*claim_count, 0U);
}

// REQ: RUNTIME-007
// kDrainAll[capacity=3]: empty begin; single sample with abort then accept; maximum (3 samples).
// selected() is a stable span -- data pointer unchanged across repeated calls on one transaction.
TEST(DispatcherBindingTest, DirectDrainAllCoversEmptyOneAndMaximumTransactionalSnapshots) {
  auto plan = selection_plan(InputSelectorKind::kDrainAll, PortId{2U}, 3U, 3U);
  auto state = detail::RuntimeInputSelectionState::create(plan);
  ASSERT_TRUE(state);

  // Key setup: three phases -- empty queue, single-with-abort, and max-capacity drain.
  auto empty = state->begin(TimePoint{1, kRuntimeClock}, 0U, 1U);
  ASSERT_TRUE(empty);
  EXPECT_TRUE(empty->selected().empty());
  EXPECT_EQ(empty->selection().state, InputState::kSelected);
  EXPECT_EQ(empty->selection().reason, SelectionReason::kNone);
  ASSERT_TRUE(empty->finish(true));

  ASSERT_TRUE(state->push(temporal_sample(10, 1U, 1U)));
  auto one = state->begin(TimePoint{10, kRuntimeClock}, 1U, 1U);
  ASSERT_TRUE(one);
  ASSERT_EQ(one->selected().size(), 1U);
  const auto one_again = one->selected();
  ASSERT_EQ(one_again.size(), 1U);
  EXPECT_EQ(one_again.front().source_sequence, 1U);
  ASSERT_TRUE(one->finish(false));
  EXPECT_EQ(state->retained_count(), 1U);

  // A sibling Skip suppresses the activation and therefore aborts the
  // provisional drain. The same prefix remains available to a later accepted
  // activation and is committed exactly once.
  auto skipped = state->begin(TimePoint{10, kRuntimeClock}, 1U, 1U);
  ASSERT_TRUE(skipped);
  const std::array sibling_inputs{skipped->selection(),
                                  InputSelection{InputState::kAbsent, SelectionReason::kNeverSeen}};
  ASSERT_EQ(detail::aggregate_missing_inputs(sibling_inputs),
            detail::MissingAggregateDisposition::kNoFire);
  ASSERT_TRUE(skipped->finish(false));
  EXPECT_EQ(state->retained_count(), 1U);

  auto accepted_one = state->begin(TimePoint{10, kRuntimeClock}, 1U, 1U);
  ASSERT_TRUE(accepted_one);
  ASSERT_TRUE(accepted_one->finish(true));
  EXPECT_EQ(state->retained_count(), 0U);

  ASSERT_TRUE(state->push(temporal_sample(20, 2U, 2U)));
  ASSERT_TRUE(state->push(temporal_sample(21, 3U, 3U)));
  ASSERT_TRUE(state->push(temporal_sample(22, 4U, 4U)));
  auto maximum = state->begin(TimePoint{22, kRuntimeClock}, 4U, 1U);
  ASSERT_TRUE(maximum);
  ASSERT_EQ(maximum->selected().size(), 3U);
  EXPECT_EQ(maximum->selected()[0].source_sequence, 2U);
  EXPECT_EQ(maximum->selected()[1].source_sequence, 3U);
  EXPECT_EQ(maximum->selected()[2].source_sequence, 4U);
  // Key assertion: repeated calls to selected() return the same data pointer (no extra allocation).
  const auto repeated = maximum->selected();
  ASSERT_EQ(repeated.size(), 3U);
  EXPECT_EQ(repeated.data(), maximum->selected().data());
  ASSERT_TRUE(maximum->finish(true));
  EXPECT_EQ(state->retained_count(), 0U);
}

// REQ: RUNTIME-007
// port1[t=100] + port2[t=101] -> aligned join [inclusive_window=1ns] -> kNominal (both selected);
// then port1[t=200] + port2[t=210] -> kRetireMisaligned for port1; port2 survives in queue.
TEST(DispatcherBindingTest, PlanBackedAlignedJoinCommitsAcceptedHeadsAndRetiresEveryMinimumMiss) {
  const auto plan = aligned_plan(2U, 1);
  const auto selections = aligned_selections(2U);
  auto state = detail::RuntimeAlignedSelectionState::create(plan, selections);
  ASSERT_TRUE(state);
  ASSERT_EQ(state->member_inputs().size(), 2U);
  EXPECT_EQ(state->member_inputs()[0], PortId{1U});
  EXPECT_EQ(state->member_inputs()[1], PortId{2U});

  // Key setup: samples pushed out of port order to confirm begin() takes the maximum head time.
  ASSERT_TRUE(state->push(PortId{2U}, temporal_sample(101, 1U, 1U)));
  ASSERT_TRUE(state->push(PortId{1U}, temporal_sample(100, 1U, 1U)));
  std::array<detail::TemporalSampleRecord, 2U> member_claims{};
  auto member_claim_count = state->copy_payload_claims(PortId{1U}, member_claims);
  ASSERT_TRUE(member_claim_count);
  ASSERT_EQ(*member_claim_count, 1U);
  EXPECT_EQ(member_claims[0], temporal_sample(100, 1U, 1U));
  auto provisional = state->begin();
  ASSERT_TRUE(provisional);
  EXPECT_EQ(provisional->disposition(), detail::AlignedCandidateDisposition::kNominal);
  EXPECT_EQ(provisional->reference_time(), (TimePoint{101, kRuntimeClock}));
  ASSERT_EQ(provisional->selections().size(), 2U);
  EXPECT_EQ(provisional->selections()[0].state, InputState::kSelected);
  EXPECT_EQ(provisional->selections()[1].state, InputState::kSelected);
  ASSERT_EQ(provisional->selected().size(), 2U);
  EXPECT_EQ(provisional->selected()[0].capture_time.timestamp_ns, 100);
  EXPECT_EQ(provisional->selected()[1].capture_time.timestamp_ns, 101);
  EXPECT_EQ(state->reset().error().code, ErrorCode::kBackpressured);
  ASSERT_TRUE(provisional->finish(false));
  EXPECT_EQ(state->queued(PortId{1U}), 1U);
  EXPECT_EQ(state->queued(PortId{2U}), 1U);
  member_claim_count = state->copy_payload_claims(PortId{1U}, member_claims);
  ASSERT_TRUE(member_claim_count);
  EXPECT_EQ(*member_claim_count, 1U);

  auto accepted = state->begin();
  ASSERT_TRUE(accepted);
  ASSERT_TRUE(accepted->finish(true));
  EXPECT_EQ(state->queued(PortId{1U}), 0U);
  EXPECT_EQ(state->queued(PortId{2U}), 0U);
  member_claim_count = state->copy_payload_claims(PortId{1U}, member_claims);
  ASSERT_TRUE(member_claim_count);
  EXPECT_EQ(*member_claim_count, 0U);

  ASSERT_TRUE(state->push(PortId{1U}, temporal_sample(200, 2U, 2U)));
  // Key assertion: gap of 10ns exceeds window=1ns; port1 is retired as kRetireMisaligned.
  ASSERT_TRUE(state->push(PortId{2U}, temporal_sample(210, 2U, 2U)));
  auto miss = state->begin();
  ASSERT_TRUE(miss);
  EXPECT_EQ(miss->disposition(), detail::AlignedCandidateDisposition::kRetireMisaligned);
  EXPECT_EQ(miss->reason(), SelectionReason::kAlignmentWindowMiss);
  ASSERT_EQ(miss->selected().size(), 1U);
  EXPECT_EQ(miss->selected().front().capture_time.timestamp_ns, 200);
  ASSERT_TRUE(miss->finish(false));
  EXPECT_EQ(state->queued(PortId{1U}), 0U);
  EXPECT_EQ(state->queued(PortId{2U}), 1U);
  member_claim_count = state->copy_payload_claims(PortId{1U}, member_claims);
  ASSERT_TRUE(member_claim_count);
  EXPECT_EQ(*member_claim_count, 0U);
  member_claim_count = state->copy_payload_claims(PortId{2U}, member_claims);
  ASSERT_TRUE(member_claim_count);
  EXPECT_EQ(*member_claim_count, 1U);
}

// REQ: LIFECYCLE-012
// REQ: CHAN-141
// Each aligned member retains its maximum physical backing while lifecycle publication changes
// only the member-local active FIFO bound.
TEST(DispatcherBindingTest, RateCandidateDepthSeparatesAlignedBackingFromActiveAdmission) {
  const auto plan = aligned_plan(1U, 1);
  const auto selections = aligned_selections(1U);
  const std::array capacities{
      detail::RuntimeQueueCapacity{.backing = 4U, .active = 1U},
      detail::RuntimeQueueCapacity{.backing = 4U, .active = 1U},
  };
  auto state = detail::RuntimeAlignedSelectionState::create(plan, selections, capacities);
  ASSERT_TRUE(state);
  EXPECT_EQ(state->payload_claim_capacity(PortId{1U}), 4U);

  ASSERT_TRUE(state->push(PortId{1U}, temporal_sample(10, 1U, 1U)));
  EXPECT_EQ(state->push(PortId{1U}, temporal_sample(11, 2U, 2U)).error().code,
            ErrorCode::kResourceExhausted);
  ASSERT_TRUE(state->prepare_rate_candidate_capacity(PortId{1U}, 2U));
  EXPECT_EQ(state->push(PortId{1U}, temporal_sample(11, 2U, 2U)).error().code,
            ErrorCode::kResourceExhausted);
  state->discard_rate_candidate_capacities();

  ASSERT_TRUE(state->prepare_rate_candidate_capacity(PortId{1U}, 2U));
  ASSERT_TRUE(state->prepare_rate_candidate_capacity(PortId{2U}, 2U));
  ASSERT_TRUE(state->commit_rate_candidate_capacities());
  ASSERT_TRUE(state->push(PortId{1U}, temporal_sample(11, 2U, 2U)));
  EXPECT_EQ(state->push(PortId{1U}, temporal_sample(12, 3U, 3U)).error().code,
            ErrorCode::kResourceExhausted);
  EXPECT_EQ(state->payload_claim_capacity(PortId{1U}), 4U);
}

// REQ: RUNTIME-007
// Aligned join [window=0ns, port2=kSafeDefault]: begin() without closure proof -> kNotReady;
// begin(EndOfStream) -> kDegraded; reset() clears both member queues and all generation state.
TEST(DispatcherBindingTest, PlanBackedAlignedJoinRequiresProofAndResetsAllGenerationState) {
  const auto plan = aligned_plan(1U, 0);
  auto selections = aligned_selections(1U);
  selections[1].missing = TemporalRecordMissingKind::kSafeDefault;
  auto state = detail::RuntimeAlignedSelectionState::create(plan, selections);
  ASSERT_TRUE(state);

  // Key setup: only port1 has a sample; begin() without closure proof is blocked (port2 missing).
  ASSERT_TRUE(state->push(PortId{1U}, temporal_sample(300, 1U, 1U)));
  EXPECT_EQ(state->begin().error().code, ErrorCode::kNotReady);
  const std::array closure{detail::AlignmentClosureProof::kOpen,
                           detail::AlignmentClosureProof::kEndOfStream};
  auto degraded = state->begin(closure, TimePoint{300, kRuntimeClock});
  ASSERT_TRUE(degraded);
  EXPECT_EQ(degraded->disposition(), detail::AlignedCandidateDisposition::kDegraded);
  ASSERT_EQ(degraded->selections().size(), 2U);
  EXPECT_EQ(degraded->selections()[0].state, InputState::kSelected);
  EXPECT_EQ(degraded->selections()[1].state, InputState::kSafeDefault);
  ASSERT_EQ(degraded->selected().size(), 1U);
  // Key assertion: EndOfStream closure proof makes port2 resolve as kSafeDefault -> kDegraded.
  ASSERT_TRUE(degraded->finish(true));
  EXPECT_EQ(state->queued(PortId{1U}), 0U);

  auto wrong_clock = temporal_sample(301, 2U, 2U);
  wrong_clock.capture_time.clock_id = ClockId{77U};
  EXPECT_EQ(state->push(PortId{1U}, wrong_clock).error().code, ErrorCode::kInvalidArgument);
  EXPECT_EQ(state->push(PortId{99U}, temporal_sample(301, 2U, 2U)).error().code,
            ErrorCode::kInvalidArgument);
  ASSERT_TRUE(state->push(PortId{1U}, temporal_sample(301, 2U, 2U)));
  EXPECT_EQ(state->push(PortId{1U}, temporal_sample(302, 3U, 3U)).error().code,
            ErrorCode::kResourceExhausted);
  ASSERT_TRUE(state->reset());
  EXPECT_EQ(state->queued(PortId{1U}), 0U);
  EXPECT_EQ(state->queued(PortId{2U}), 0U);

  auto tampered = plan;
  tampered.no_scan_ahead = false;
  EXPECT_EQ(detail::RuntimeAlignedSelectionState::create(tampered, selections).error().code,
            ErrorCode::kInvalidArgument);
  auto mismatched = selections;
  mismatched[1].selector = InputSelectorKind::kLatestWithin;
  mismatched[1].max_age_ns = 1;
  EXPECT_EQ(detail::RuntimeAlignedSelectionState::create(plan, mismatched).error().code,
            ErrorCode::kInvalidArgument);
}

// REQ: CHAN-136
// Aligned join [port1=kReuseLast]: after finish(true) the port1 claim stays visible (reuse buffer)
// until every member closes and terminal history is released; port2 reports zero after commit.
TEST(DispatcherBindingTest, AlignedReusePayloadClaimSurvivesUntilTerminalRelease) {
  const auto plan = aligned_plan(1U, 0);
  auto selections = aligned_selections(1U);
  selections[0].missing = TemporalRecordMissingKind::kReuseLast;
  selections[0].reuse_max_age_ns = 100;
  auto state = detail::RuntimeAlignedSelectionState::create(plan, selections);
  ASSERT_TRUE(state);

  ASSERT_TRUE(state->push(PortId{1U}, temporal_sample(400, 1U, 1U)));
  ASSERT_TRUE(state->push(PortId{2U}, temporal_sample(400, 1U, 1U)));
  auto selected = state->begin();
  ASSERT_TRUE(selected);
  ASSERT_TRUE(selected->finish(true));

  std::array<detail::TemporalSampleRecord, 2U> claims{};
  // Key assertion: port1 claim count stays 1 after commit (reuse buffer); port2 drops to 0.
  auto first_count = state->copy_payload_claims(PortId{1U}, claims);
  ASSERT_TRUE(first_count);
  ASSERT_EQ(*first_count, 1U);
  EXPECT_EQ(claims[0], temporal_sample(400, 1U, 1U));
  auto second_count = state->copy_payload_claims(PortId{2U}, claims);
  ASSERT_TRUE(second_count);
  EXPECT_EQ(*second_count, 0U);

  EXPECT_EQ(state->release_terminal_history().error().code, ErrorCode::kInvalidArgument);
  ASSERT_TRUE(state->close_end_of_stream(PortId{1U}, TimePoint{401, kRuntimeClock}));
  EXPECT_EQ(state->release_terminal_history().error().code, ErrorCode::kInvalidArgument);
  ASSERT_TRUE(state->close_end_of_stream(PortId{2U}, TimePoint{401, kRuntimeClock}));
  ASSERT_TRUE(state->release_terminal_history());
  first_count = state->copy_payload_claims(PortId{1U}, claims);
  ASSERT_TRUE(first_count);
  EXPECT_EQ(*first_count, 0U);

  ASSERT_TRUE(state->reset());
  first_count = state->copy_payload_claims(PortId{1U}, claims);
  ASSERT_TRUE(first_count);
  EXPECT_EQ(*first_count, 0U);
}

// REQ: RUNTIME-007
// anchor(10) -> pending; anchor(20) closes window; samples at t=5,10,15,25 in lookahead range.
// finish(false) retires the close; second begin() -> kNotReady (close consumed exactly once).
// Producer-epoch transition does not retract an already-closed window activation.
TEST(DispatcherBindingTest, PlanBackedBracketedWindowPreservesClosedSnapshotButRetiresItOnce) {
  auto selection = selection_plan(InputSelectorKind::kBracketedWindow, PortId{2U}, 8U, 5U);
  selection.lookbehind = 1U;
  selection.lookahead = 1U;
  const BracketedWindowPlanRecord window{
      .owner = OperatorRef{1U, 77U, 0U},
      .input = PortId{2U},
      .anchor = PortId{1U},
      .lookbehind = 1U,
      .lookahead = 1U,
      .max_samples = 5U,
      .event_clock = kRuntimeClock,
      .queue_capacity = 8U,
      .cache_capacity = 8U,
      .producer_max_emits_per_compute = 1U,
      .retained_overlap_capacity = 2U,
  };
  auto state = detail::RuntimeInputSelectionState::create(selection, &window, 2U);
  ASSERT_TRUE(state);

  ASSERT_TRUE(state->push(temporal_sample(5, 1U, 1U)));
  ASSERT_TRUE(state->push_anchor(temporal_sample(10, 1U, 1U)));
  ASSERT_TRUE(state->push(temporal_sample(10, 2U, 2U)));
  ASSERT_TRUE(state->push(temporal_sample(15, 3U, 3U)));
  ASSERT_TRUE(state->push_anchor(temporal_sample(20, 2U, 2U)));
  ASSERT_TRUE(state->push(temporal_sample(25, 4U, 4U)));

  std::array<detail::TemporalSampleRecord, 32U> claims{};
  auto claim_count = state->copy_payload_claims(claims);
  ASSERT_TRUE(claim_count);
  EXPECT_LE(*claim_count, state->payload_claim_capacity());
  EXPECT_NE(std::find(claims.begin(),
                      claims.begin() + static_cast<std::ptrdiff_t>(*claim_count),
                      temporal_sample(5, 1U, 1U)),
            claims.begin() + static_cast<std::ptrdiff_t>(*claim_count));

  // Key setup: anchor(10) and anchor(20) pushed; begin() at t=20 yields 4 samples in window.
  auto closed = state->begin(TimePoint{20, kRuntimeClock}, 4U, 1U);
  ASSERT_TRUE(closed);
  ASSERT_EQ(closed->selected().size(), 4U);
  // Key assertion: reset() while a window is open returns kBackpressured.
  EXPECT_EQ(state->reset().error().code, ErrorCode::kBackpressured);
  // Sibling missing policy may suppress compute. The close is retired once,
  // while bracket samples remain non-destructive runtime history.
  ASSERT_TRUE(closed->finish(false));
  EXPECT_EQ(state->begin(TimePoint{20, kRuntimeClock}, 4U, 1U).error().code, ErrorCode::kNotReady);
  claim_count = state->copy_payload_claims(claims);
  ASSERT_TRUE(claim_count);
  EXPECT_EQ(std::find(claims.begin(),
                      claims.begin() + static_cast<std::ptrdiff_t>(*claim_count),
                      temporal_sample(5, 1U, 1U)),
            claims.begin() + static_cast<std::ptrdiff_t>(*claim_count));

  ASSERT_TRUE(state->reset());
  claim_count = state->copy_payload_claims(claims);
  ASSERT_TRUE(claim_count);
  EXPECT_EQ(*claim_count, 0U);
  EXPECT_EQ(state->begin(TimePoint{20, kRuntimeClock}, 4U, 2U).error().code, ErrorCode::kNotReady);

  // A producer-epoch transition cancels pending state but does not retract an
  // already closed immutable activation. The prior close remains dispatchable
  // exactly once and is never relabeled as new-epoch state.
  ASSERT_TRUE(state->push(temporal_sample(5, 1U, 1U)));
  ASSERT_TRUE(state->push_anchor(temporal_sample(10, 1U, 1U)));
  ASSERT_TRUE(state->push(temporal_sample(10, 2U, 2U)));
  ASSERT_TRUE(state->push_anchor(temporal_sample(20, 2U, 2U)));
  ASSERT_TRUE(state->push(temporal_sample(25, 3U, 3U)));
  auto next_epoch = temporal_sample(30, 1U, 4U);
  next_epoch.producer_epoch = 2U;
  ASSERT_TRUE(state->push(next_epoch));
  auto pre_transition_close = state->begin(TimePoint{20, kRuntimeClock}, 4U, 1U);
  ASSERT_TRUE(pre_transition_close);
  EXPECT_EQ(pre_transition_close->selection().state, InputState::kSelected);
  ASSERT_TRUE(pre_transition_close->finish(false));
  EXPECT_EQ(state->begin(TimePoint{20, kRuntimeClock}, 4U, 1U).error().code, ErrorCode::kNotReady);
}

// REQ: RUNTIME-007
// anchor(10) + anchor(20) -> pending window, no samples; close_oldest(kWindowTimeout) -> kAbsent;
// then anchor(30) + anchor(40), close_oldest(kEndOfStream) -> kAbsent with kEndOfStream reason.
TEST(DispatcherBindingTest, BracketedWindowExposesBoundedTimeoutAndEndOfStreamClosure) {
  auto selection = selection_plan(InputSelectorKind::kBracketedWindow, PortId{2U}, 4U, 3U);
  selection.lookbehind = 1U;
  selection.lookahead = 1U;
  const BracketedWindowPlanRecord window{
      .owner = OperatorRef{1U, 77U, 0U},
      .input = PortId{2U},
      .anchor = PortId{1U},
      .lookbehind = 1U,
      .lookahead = 1U,
      .max_samples = 3U,
      .event_clock = kRuntimeClock,
      .queue_capacity = 4U,
      .cache_capacity = 4U,
      .producer_max_emits_per_compute = 1U,
      .retained_overlap_capacity = 1U,
  };
  auto state = detail::RuntimeInputSelectionState::create(selection, &window, 1U);
  ASSERT_TRUE(state);

  ASSERT_TRUE(state->push_anchor(temporal_sample(10, 1U, 1U)));
  ASSERT_TRUE(state->push_anchor(temporal_sample(20, 2U, 2U)));
  // Key setup: two anchors but no samples; one window is pending after the second anchor.
  ASSERT_EQ(state->pending_window_count(), 1U);
  const auto oldest_pending_window = state->oldest_pending_window_reference();
  ASSERT_TRUE(oldest_pending_window);
  if (!oldest_pending_window) {
    return;
  }
  EXPECT_EQ(*oldest_pending_window, (TimePoint{20, kRuntimeClock}));
  ASSERT_TRUE(state->close_oldest(SelectionReason::kWindowTimeout));
  EXPECT_EQ(state->pending_window_count(), 0U);
  EXPECT_FALSE(state->oldest_pending_window_reference());
  // Key assertion: begin() after timeout close returns kAbsent with reason=kWindowTimeout.
  auto timed_out = state->begin(TimePoint{20, kRuntimeClock}, 0U, 1U);
  ASSERT_TRUE(timed_out);
  EXPECT_EQ(timed_out->selection().state, InputState::kAbsent);
  EXPECT_EQ(timed_out->selection().reason, SelectionReason::kWindowTimeout);
  ASSERT_TRUE(timed_out->finish(false));

  ASSERT_TRUE(state->reset());
  ASSERT_TRUE(state->push_anchor(temporal_sample(30, 1U, 3U)));
  ASSERT_TRUE(state->push_anchor(temporal_sample(40, 2U, 4U)));
  ASSERT_TRUE(state->close_oldest(SelectionReason::kEndOfStream));
  auto end_of_stream = state->begin(TimePoint{40, kRuntimeClock}, 0U, 2U);
  ASSERT_TRUE(end_of_stream);
  EXPECT_EQ(end_of_stream->selection().state, InputState::kAbsent);
  EXPECT_EQ(end_of_stream->selection().reason, SelectionReason::kEndOfStream);
  ASSERT_TRUE(end_of_stream->finish(false));
}

// REQ: RUNTIME-007
// REQ: CONTRACT-019
// End of stream is unsolicited: a bracketed input can have no pending window when the producer's
// close() lands, because whether the anchor already closed the last window is a race.
// close_end_of_stream() absorbs that race and succeeds, so a well-formed close() cannot terminate
// the run. The dedicated terminal latch is distinct from close_oldest(): directly closing a window
// the engine never observed remains a caller error, which the assertions below pin down (see also
// TEMPORAL_CONTRACT.TemporalInputCache.WindowTimeoutAndEndOfStreamClosePendingInOrder).
TEST(DispatcherBindingTest, BracketedWindowEndOfStreamClosureAcceptsEmptyPendingSet) {
  auto selection = selection_plan(InputSelectorKind::kBracketedWindow, PortId{2U}, 4U, 3U);
  selection.lookbehind = 1U;
  selection.lookahead = 1U;
  const BracketedWindowPlanRecord window{
      .owner = OperatorRef{1U, 77U, 0U},
      .input = PortId{2U},
      .anchor = PortId{1U},
      .lookbehind = 1U,
      .lookahead = 1U,
      .max_samples = 3U,
      .event_clock = kRuntimeClock,
      .queue_capacity = 4U,
      .cache_capacity = 4U,
      .producer_max_emits_per_compute = 1U,
      .retained_overlap_capacity = 1U,
  };

  // No anchors pushed: pending_ is empty from the start.
  auto fresh = detail::RuntimeInputSelectionState::create(selection, &window, 1U);
  ASSERT_TRUE(fresh);
  ASSERT_EQ(fresh->pending_window_count(), 0U);
  // Key assertion: the regression. This returned kInvalidArgument before, which reached
  // run_session.cpp and failed the run with RUNTIME_PORT_BINDING_UNAVAILABLE.
  EXPECT_TRUE(fresh->close_end_of_stream());
  // The engine keeps its strict precondition: a direct close of a window it never observed is a
  // caller error, for both reasons. Only close_end_of_stream() absorbs the race.
  EXPECT_EQ(fresh->close_oldest(SelectionReason::kWindowTimeout).error().code,
            ErrorCode::kInvalidArgument);
  EXPECT_EQ(fresh->close_oldest(SelectionReason::kEndOfStream).error().code,
            ErrorCode::kInvalidArgument);

  // Same guarantee after the pending set has been drained by an earlier close, which is the
  // ordering the camera/IMU reference application actually hits.
  auto drained = detail::RuntimeInputSelectionState::create(selection, &window, 1U);
  ASSERT_TRUE(drained);
  ASSERT_TRUE(drained->push_anchor(temporal_sample(10, 1U, 1U)));
  ASSERT_TRUE(drained->push_anchor(temporal_sample(20, 2U, 2U)));
  ASSERT_EQ(drained->pending_window_count(), 1U);
  ASSERT_TRUE(drained->close_oldest(SelectionReason::kEndOfStream));
  ASSERT_EQ(drained->pending_window_count(), 0U);
  // A second end-of-stream against the now-empty pending set is still accepted.
  EXPECT_TRUE(drained->close_end_of_stream());
}

// REQ: CHAN-136
// One transport EOS must close every window already pending for that producer generation. The
// selector may already hold closed decisions when EOS arrives, so terminal closure is latched and
// continues as result slots are retired; no second transport EOS or timeout is required.
TEST(DispatcherBindingTest, BracketedWindowEndOfStreamDrainsEveryPendingWindow) {
  auto selection = selection_plan(InputSelectorKind::kBracketedWindow, PortId{2U}, 4U, 3U);
  selection.lookbehind = 1U;
  selection.lookahead = 1U;
  const BracketedWindowPlanRecord window{
      .owner = OperatorRef{1U, 77U, 0U},
      .input = PortId{2U},
      .anchor = PortId{1U},
      .lookbehind = 1U,
      .lookahead = 1U,
      .max_samples = 3U,
      .event_clock = kRuntimeClock,
      .queue_capacity = 4U,
      .cache_capacity = 4U,
      .producer_max_emits_per_compute = 1U,
      .retained_overlap_capacity = 1U,
  };
  auto state = detail::RuntimeInputSelectionState::create(selection, &window, 2U);
  ASSERT_TRUE(state);

  // Fill two of the three preallocated result slots, then leave two windows pending. EOS can
  // publish only one terminal decision immediately; retiring the oldest result must publish the
  // remaining latched terminal decision without another external event.
  ASSERT_TRUE(state->push(temporal_sample(5, 1U, 1U)));
  ASSERT_TRUE(state->push_anchor(temporal_sample(10, 1U, 1U)));
  ASSERT_TRUE(state->push_anchor(temporal_sample(20, 2U, 2U)));
  ASSERT_TRUE(state->close_oldest(SelectionReason::kWindowTimeout));
  ASSERT_TRUE(state->push_anchor(temporal_sample(30, 3U, 3U)));
  ASSERT_TRUE(state->close_oldest(SelectionReason::kWindowTimeout));
  ASSERT_TRUE(state->push_anchor(temporal_sample(40, 4U, 4U)));
  ASSERT_TRUE(state->push_anchor(temporal_sample(50, 5U, 5U)));
  ASSERT_EQ(state->pending_window_count(), 2U);

  ASSERT_TRUE(state->close_end_of_stream());
  EXPECT_EQ(state->pending_window_count(), 1U);

  constexpr std::array expected_reasons{SelectionReason::kWindowTimeout,
                                        SelectionReason::kWindowTimeout,
                                        SelectionReason::kEndOfStream,
                                        SelectionReason::kEndOfStream};
  constexpr std::array<std::int64_t, 4U> expected_reference_times{20, 30, 40, 50};
  for (std::size_t index = 0U; index < expected_reasons.size(); ++index) {
    auto closed = state->begin(TimePoint{60, kRuntimeClock}, 0U, 1U);
    ASSERT_TRUE(closed);
    EXPECT_EQ(closed->selection().state, InputState::kAbsent);
    EXPECT_EQ(closed->selection().reason, expected_reasons[index]);
    EXPECT_EQ(closed->semantic_reference_time(),
              (TimePoint{expected_reference_times[index], kRuntimeClock}));
    ASSERT_TRUE(closed->finish(false));
    if (index == 0U) {
      EXPECT_EQ(state->pending_window_count(), 0U);
      std::array<detail::TemporalSampleRecord, 4U> claims{};
      const auto claim_count = state->copy_payload_claims(claims);
      ASSERT_TRUE(claim_count);
      EXPECT_EQ(*claim_count, 0U);
    }
  }
  EXPECT_EQ(state->begin(TimePoint{60, kRuntimeClock}, 0U, 1U).error().code, ErrorCode::kNotReady);
}

// REQ: CHAN-136
// A window that became ordinarily ready before EOS keeps that decision even when a full closed-
// result ring delayed publication. Retiring one older result must publish the ready snapshot
// before EOS closes only the still-unresolved window.
TEST(DispatcherBindingTest, BracketedWindowEndOfStreamPreservesReadyDecisionBehindFullRing) {
  auto selection = selection_plan(InputSelectorKind::kBracketedWindow, PortId{2U}, 4U, 3U);
  selection.lookbehind = 1U;
  selection.lookahead = 1U;
  const BracketedWindowPlanRecord window{
      .owner = OperatorRef{1U, 77U, 0U},
      .input = PortId{2U},
      .anchor = PortId{1U},
      .lookbehind = 1U,
      .lookahead = 1U,
      .max_samples = 3U,
      .event_clock = kRuntimeClock,
      .queue_capacity = 4U,
      .cache_capacity = 4U,
      .producer_max_emits_per_compute = 1U,
      .retained_overlap_capacity = 1U,
  };
  auto state = detail::RuntimeInputSelectionState::create(selection, &window, 2U);
  ASSERT_TRUE(state);

  // Fill all three preallocated result slots, then leave two windows pending. The first pending
  // window has a complete [lookbehind, interior, lookahead] snapshot before EOS arrives; the
  // second has no outer successor and remains unresolved.
  ASSERT_TRUE(state->push_anchor(temporal_sample(10, 1U, 1U)));
  ASSERT_TRUE(state->push_anchor(temporal_sample(20, 2U, 2U)));
  ASSERT_TRUE(state->close_oldest(SelectionReason::kWindowTimeout));
  ASSERT_TRUE(state->push_anchor(temporal_sample(30, 3U, 3U)));
  ASSERT_TRUE(state->close_oldest(SelectionReason::kWindowTimeout));
  ASSERT_TRUE(state->push_anchor(temporal_sample(40, 4U, 4U)));
  ASSERT_TRUE(state->close_oldest(SelectionReason::kWindowTimeout));
  ASSERT_TRUE(state->push_anchor(temporal_sample(50, 5U, 5U)));
  ASSERT_TRUE(state->push_anchor(temporal_sample(60, 6U, 6U)));
  ASSERT_TRUE(state->push(temporal_sample(35, 1U, 1U)));
  ASSERT_TRUE(state->push(temporal_sample(45, 2U, 2U)));
  ASSERT_TRUE(state->push(temporal_sample(55, 3U, 3U)));
  ASSERT_EQ(state->pending_window_count(), 2U);

  ASSERT_TRUE(state->close_end_of_stream());
  EXPECT_EQ(state->pending_window_count(), 2U);

  constexpr std::array<std::int64_t, 3U> timeout_references{20, 30, 40};
  for (const auto reference : timeout_references) {
    auto timed_out = state->begin(TimePoint{70, kRuntimeClock}, 0U, 1U);
    ASSERT_TRUE(timed_out);
    EXPECT_EQ(timed_out->selection().state, InputState::kAbsent);
    EXPECT_EQ(timed_out->selection().reason, SelectionReason::kWindowTimeout);
    EXPECT_EQ(timed_out->semantic_reference_time(), (TimePoint{reference, kRuntimeClock}));
    ASSERT_TRUE(timed_out->finish(false));
  }

  auto ready = state->begin(TimePoint{70, kRuntimeClock}, 3U, 1U);
  ASSERT_TRUE(ready);
  EXPECT_EQ(ready->selection().state, InputState::kSelected);
  EXPECT_EQ(ready->selection().reason, SelectionReason::kNone);
  EXPECT_EQ(ready->semantic_reference_time(), (TimePoint{50, kRuntimeClock}));
  ASSERT_EQ(ready->selected().size(), 3U);
  EXPECT_EQ(ready->selected()[0], temporal_sample(35, 1U, 1U));
  EXPECT_EQ(ready->selected()[1], temporal_sample(45, 2U, 2U));
  EXPECT_EQ(ready->selected()[2], temporal_sample(55, 3U, 3U));
  ASSERT_TRUE(ready->finish(false));

  auto end_of_stream = state->begin(TimePoint{70, kRuntimeClock}, 3U, 1U);
  ASSERT_TRUE(end_of_stream);
  EXPECT_EQ(end_of_stream->selection().state, InputState::kAbsent);
  EXPECT_EQ(end_of_stream->selection().reason, SelectionReason::kEndOfStream);
  EXPECT_EQ(end_of_stream->semantic_reference_time(), (TimePoint{60, kRuntimeClock}));
  ASSERT_TRUE(end_of_stream->finish(false));

  EXPECT_EQ(state->pending_window_count(), 0U);
  EXPECT_EQ(state->begin(TimePoint{70, kRuntimeClock}, 3U, 1U).error().code, ErrorCode::kNotReady);
  std::array<detail::TemporalSampleRecord, 4U> claims{};
  const auto claim_count = state->copy_payload_claims(claims);
  ASSERT_TRUE(claim_count);
  EXPECT_EQ(*claim_count, 0U);
}

// REQ: RUNTIME-007
// BracketedWindow[missing=kSafeDefault]: timeout close produces kSafeDefault with empty selected()
// and a valid semantic_reference_time; no samples are invented to fill the missing batch.
TEST(DispatcherBindingTest, BracketedWindowSafeDefaultMapsTimeoutWithoutInventingBatch) {
  auto selection = selection_plan(InputSelectorKind::kBracketedWindow, PortId{2U}, 4U, 3U);
  selection.lookbehind = 1U;
  selection.lookahead = 1U;
  selection.missing = TemporalRecordMissingKind::kSafeDefault;
  const BracketedWindowPlanRecord window{
      .owner = OperatorRef{1U, 77U, 0U},
      .input = PortId{2U},
      .anchor = PortId{1U},
      .lookbehind = 1U,
      .lookahead = 1U,
      .max_samples = 3U,
      .event_clock = kRuntimeClock,
      .queue_capacity = 4U,
      .cache_capacity = 4U,
      .producer_max_emits_per_compute = 1U,
      .retained_overlap_capacity = 1U,
  };
  auto state = detail::RuntimeInputSelectionState::create(selection, &window, 1U);
  ASSERT_TRUE(state);

  ASSERT_TRUE(state->push_anchor(temporal_sample(10, 1U, 1U)));
  ASSERT_TRUE(state->push_anchor(temporal_sample(20, 2U, 2U)));
  ASSERT_TRUE(state->close_oldest(SelectionReason::kWindowTimeout));
  // Key assertion: kSafeDefault result has empty selected() but semantic_reference_time=anchor(20).
  auto timed_out = state->begin(TimePoint{25, kRuntimeClock}, 0U, 1U);
  ASSERT_TRUE(timed_out);
  EXPECT_EQ(timed_out->selection().state, InputState::kSafeDefault);
  EXPECT_EQ(timed_out->selection().reason, SelectionReason::kWindowTimeout);
  EXPECT_TRUE(timed_out->selected().empty());
  EXPECT_EQ(timed_out->semantic_reference_time(), (TimePoint{20, kRuntimeClock}));
  ASSERT_TRUE(timed_out->finish(true));
}

// REQ: RUNTIME-007
// Samples: t=5  t=10(anchor1)  t=10  t=15  t=20(anchor2)  t=20  t=25
// Window [lookbehind=1, lookahead=1] around anchor2: reference_time=20, close_time=25.
// Overlap window at anchor3=30 reuses t=15..t=35 non-destructively.
TEST(DispatcherBindingTest, BracketedWindowKeepsSemanticAnchorAndLaterCloseTag) {
  auto selection = selection_plan(InputSelectorKind::kBracketedWindow, PortId{2U}, 8U, 5U);
  selection.lookbehind = 1U;
  selection.lookahead = 1U;
  const BracketedWindowPlanRecord window{
      .owner = OperatorRef{1U, 77U, 0U},
      .input = PortId{2U},
      .anchor = PortId{1U},
      .lookbehind = 1U,
      .lookahead = 1U,
      .max_samples = 5U,
      .event_clock = kRuntimeClock,
      .queue_capacity = 8U,
      .cache_capacity = 8U,
      .producer_max_emits_per_compute = 1U,
      .retained_overlap_capacity = 2U,
  };
  auto state = detail::RuntimeInputSelectionState::create(selection, &window, 2U);
  ASSERT_TRUE(state);

  // Key setup: SyntheticClock provides monotonic timestamps; samples span both sides of anchor2.
  SyntheticClock clock;
  clock.advance_to(5);
  ASSERT_TRUE(state->push(temporal_sample(clock.timestamp(), 1U, 1U)));
  clock.advance_to(10);
  auto first_anchor = state->push_anchor(temporal_sample(clock.timestamp(), 1U, 1U));
  ASSERT_TRUE(first_anchor);
  EXPECT_EQ(first_anchor->disposition, detail::WindowAnchorDisposition::kPrimed);
  ASSERT_TRUE(state->push(temporal_sample(clock.timestamp(), 2U, 2U)));
  clock.advance_to(15);
  ASSERT_TRUE(state->push(temporal_sample(clock.timestamp(), 3U, 3U)));
  clock.advance_to(20);
  auto current_anchor = state->push_anchor(temporal_sample(clock.timestamp(), 2U, 2U));
  ASSERT_TRUE(current_anchor);
  EXPECT_EQ(current_anchor->disposition, detail::WindowAnchorDisposition::kPending);
  EXPECT_EQ(state->begin(TimePoint{20, kRuntimeClock}, 3U, 1U).error().code, ErrorCode::kNotReady);
  ASSERT_TRUE(state->push(temporal_sample(clock.timestamp(), 4U, 4U)));

  clock.advance_to(25);
  ASSERT_TRUE(state->push(temporal_sample(clock.timestamp(), 5U, 5U)));
  auto closed = state->begin(TimePoint{20, kRuntimeClock}, 5U, 1U);
  ASSERT_TRUE(closed);
  EXPECT_EQ(closed->semantic_reference_time(), (TimePoint{20, kRuntimeClock}));
  ASSERT_EQ(closed->selected().size(), 5U);
  EXPECT_EQ(closed->selected().front().capture_time.timestamp_ns, 5);
  EXPECT_EQ(closed->selected().back().capture_time.timestamp_ns, 25);

  const auto source = source_binding(0U, detail::RuntimeEventKind::kClockTick);
  auto coordinator = detail::RuntimeCandidateCoordinator::create({&source, 1U}, 1U, 1U);
  ASSERT_TRUE(coordinator);
  detail::BoundedEventPortBinding binding{1U};
  ASSERT_TRUE(binding.push(runtime_event(source, 1U, clock.timestamp())));
  ASSERT_TRUE(coordinator->ingest(binding));
  auto frozen = coordinator->freeze_next();
  ASSERT_TRUE(frozen);
  auto activation = detail::ActivationSelectionState::create(1U, selection.max_selected_samples);
  ASSERT_TRUE(activation);
  const std::array inputs{closed->selection()};
  auto view =
      activation->begin(*frozen, inputs, closed->selected(), closed->semantic_reference_time());
  ASSERT_TRUE(view);
  // Key assertion: reference_time is the semantic anchor (20); close_time is the last sample (25).
  EXPECT_EQ(view->reference_time, (TimePoint{20, kRuntimeClock}));
  EXPECT_EQ(view->close_time, (TimePoint{25, kRuntimeClock}));
  EXPECT_EQ(view->tag_time.logical_ns, 25);
  EXPECT_GT(view->close_time.timestamp_ns, view->reference_time.timestamp_ns);
  ASSERT_TRUE(closed->finish(true));
  ASSERT_TRUE(activation->retire(view->tag));
  ASSERT_TRUE(coordinator->retire(*frozen));

  // The next camera window reuses overlapping history without destructively
  // removing samples selected by the first activation.
  clock.advance_to(30);
  auto next_anchor = state->push_anchor(temporal_sample(clock.timestamp(), 3U, 3U));
  ASSERT_TRUE(next_anchor);
  EXPECT_EQ(next_anchor->disposition, detail::WindowAnchorDisposition::kPending);
  ASSERT_TRUE(state->push(temporal_sample(clock.timestamp(), 6U, 6U)));
  clock.advance_to(35);
  ASSERT_TRUE(state->push(temporal_sample(clock.timestamp(), 7U, 7U)));
  auto overlap = state->begin(TimePoint{30, kRuntimeClock}, 7U, 1U);
  ASSERT_TRUE(overlap);
  ASSERT_EQ(overlap->selected().size(), 5U);
  EXPECT_EQ(overlap->selected().front().capture_time.timestamp_ns, 15);
  EXPECT_EQ(overlap->selected().back().capture_time.timestamp_ns, 35);
  ASSERT_TRUE(overlap->finish(true));
}

// REQ: RUNTIME-007
// BracketedWindow[max_samples=3]: 4 samples arrive between anchor(1) and anchor(2); overflow
// terminates the window as kAbsent(kWindowSampleLimit); second begin() -> kNotReady (close gone).
TEST(DispatcherBindingTest, BracketedWindowOverflowIsTerminalAndNeverPartial) {
  auto selection = selection_plan(InputSelectorKind::kBracketedWindow, PortId{2U}, 8U, 3U);
  selection.lookbehind = 1U;
  selection.lookahead = 1U;
  const BracketedWindowPlanRecord window{
      .owner = OperatorRef{1U, 77U, 0U},
      .input = PortId{2U},
      .anchor = PortId{1U},
      .lookbehind = 1U,
      .lookahead = 1U,
      .max_samples = 3U,
      .event_clock = kRuntimeClock,
      .queue_capacity = 8U,
      .cache_capacity = 8U,
      .producer_max_emits_per_compute = 1U,
      .retained_overlap_capacity = 1U,
  };
  auto state = detail::RuntimeInputSelectionState::create(selection, &window, 1U);
  ASSERT_TRUE(state);
  ASSERT_TRUE(state->push(temporal_sample(0, 1U, 1U)));
  ASSERT_TRUE(state->push_anchor(temporal_sample(1, 1U, 1U)));
  ASSERT_TRUE(state->push(temporal_sample(1, 2U, 2U)));
  ASSERT_TRUE(state->push_anchor(temporal_sample(2, 2U, 2U)));
  ASSERT_TRUE(state->push(temporal_sample(2, 3U, 3U)));
  ASSERT_TRUE(state->push(temporal_sample(3, 4U, 4U)));

  // Key setup: 4 samples pushed; max_samples=3 so the window overflows before close.
  auto overflow = state->begin(TimePoint{2, kRuntimeClock}, 4U, 1U);
  ASSERT_TRUE(overflow);
  // Key assertion: overflow yields kAbsent(kWindowSampleLimit); no partial batch is produced.
  EXPECT_EQ(overflow->selection().state, InputState::kAbsent);
  EXPECT_EQ(overflow->selection().reason, SelectionReason::kWindowSampleLimit);
  EXPECT_TRUE(overflow->selected().empty());
  ASSERT_TRUE(overflow->finish(false));
  EXPECT_EQ(state->begin(TimePoint{2, kRuntimeClock}, 4U, 1U).error().code, ErrorCode::kNotReady);
}

// REQ: RUNTIME-007
// REQ: RUNTIME-028
// REQ: RUNTIME-081
// Deadline[budget=10ns]: arrival at t=110 is exactly on time (110-100==budget);
// t=111 is late; merge rules: degraded-prior + nominal -> degraded; nominal + late -> late.
TEST(DispatcherBindingTest, MaxStartDelayGateConsumesFrozenPlanRecordAfterSelection) {
  const auto emit = max_start_delay_plan(OnViolation::kEmitWitness);
  // Key setup: start=100, arrival=110 exactly equals the 10-ns budget -> nominal.
  auto exact_boundary = detail::evaluate_plan_max_start_delay(
      emit, TimePoint{100, kRuntimeClock}, TimePoint{110, kRuntimeClock});
  ASSERT_TRUE(exact_boundary);
  EXPECT_EQ(exact_boundary->disposition, detail::MaxStartDelayGateDisposition::kInvokeNominal);
  EXPECT_EQ(exact_boundary->status.reason, ActivationReason::kNominal);
  EXPECT_EQ(exact_boundary->latest_start, (TimePoint{110, kRuntimeClock}));
  EXPECT_EQ(exact_boundary->dispatch_start, (TimePoint{110, kRuntimeClock}));
  EXPECT_EQ(exact_boundary->checked_lateness_ns, 0);

  const ActivationStatus degraded_before_max_start_delay{ActivationReason::kInvariantViolated,
                                                         OnViolation::kSafeDefault};
  const ActivationStatus merged_degraded = detail::merge_max_start_delay_activation_status(
      degraded_before_max_start_delay, *exact_boundary);
  EXPECT_EQ(merged_degraded.reason, degraded_before_max_start_delay.reason);
  EXPECT_EQ(merged_degraded.policy, degraded_before_max_start_delay.policy);

  const ActivationStatus merged_nominal =
      detail::merge_max_start_delay_activation_status(ActivationStatus{}, *exact_boundary);
  EXPECT_EQ(merged_nominal.reason, exact_boundary->status.reason);
  EXPECT_EQ(merged_nominal.policy, exact_boundary->status.policy);

  auto late_emit = detail::evaluate_plan_max_start_delay(
      emit, TimePoint{100, kRuntimeClock}, TimePoint{111, kRuntimeClock});
  ASSERT_TRUE(late_emit);
  EXPECT_EQ(late_emit->disposition, detail::MaxStartDelayGateDisposition::kInvokeViolation);
  EXPECT_EQ(late_emit->status.reason, ActivationReason::kMaxStartDelayExceeded);
  EXPECT_EQ(late_emit->status.policy, OnViolation::kEmitWitness);
  EXPECT_EQ(late_emit->latest_start, (TimePoint{110, kRuntimeClock}));
  EXPECT_EQ(late_emit->dispatch_start, (TimePoint{111, kRuntimeClock}));
  EXPECT_EQ(late_emit->checked_lateness_ns, 1);
  // Key assertion: max_start_delay violation overrides a degraded-prior status in the merge.
  const ActivationStatus merged_late =
      detail::merge_max_start_delay_activation_status(degraded_before_max_start_delay, *late_emit);
  EXPECT_EQ(merged_late.reason, late_emit->status.reason);
  EXPECT_EQ(merged_late.policy, late_emit->status.policy);

  auto safe = detail::evaluate_plan_max_start_delay(max_start_delay_plan(OnViolation::kSafeDefault),
                                                    TimePoint{100, kRuntimeClock},
                                                    TimePoint{111, kRuntimeClock});
  ASSERT_TRUE(safe);
  EXPECT_EQ(safe->disposition, detail::MaxStartDelayGateDisposition::kInvokeViolation);
  EXPECT_EQ(safe->status.policy, OnViolation::kSafeDefault);

  auto fail = detail::evaluate_plan_max_start_delay(max_start_delay_plan(OnViolation::kFailRun),
                                                    TimePoint{100, kRuntimeClock},
                                                    TimePoint{111, kRuntimeClock});
  ASSERT_TRUE(fail);
  EXPECT_EQ(fail->disposition, detail::MaxStartDelayGateDisposition::kFailRun);

  auto malformed = emit;
  malformed.final_invocation_gate = false;
  EXPECT_EQ(detail::evaluate_plan_max_start_delay(
                malformed, TimePoint{100, kRuntimeClock}, TimePoint{111, kRuntimeClock})
                .error()
                .code,
            ErrorCode::kInvalidArgument);
}

// REQ: RUNTIME-013
// REQ: RUNTIME-080
// Exhaustive table: each TemporalRecordTriggerKind maps to exactly one coordinate kind.
// A mismatched coordinate in the plan record returns kInvalidArgument from
// bind_runtime_max_start_delay.
TEST(DispatcherBindingTest, MaxStartDelaySourceMappingIsExhaustiveAndClosed) {
  struct SourceCase {
    TemporalRecordTriggerKind trigger;
    MaxStartDelaySourceCoordinateKind coordinate;
    bool predicate;
    PredicateEvaluationSourceKind predicate_source;
  };
  constexpr std::array cases{
      SourceCase{TemporalRecordTriggerKind::kAnyInput,
                 MaxStartDelaySourceCoordinateKind::kTriggerSample,
                 false,
                 PredicateEvaluationSourceKind::kInputUpdate},
      SourceCase{TemporalRecordTriggerKind::kEach,
                 MaxStartDelaySourceCoordinateKind::kTriggerSample,
                 false,
                 PredicateEvaluationSourceKind::kInputUpdate},
      SourceCase{TemporalRecordTriggerKind::kClock,
                 MaxStartDelaySourceCoordinateKind::kScheduledTick,
                 false,
                 PredicateEvaluationSourceKind::kClock},
      SourceCase{TemporalRecordTriggerKind::kReadable,
                 MaxStartDelaySourceCoordinateKind::kReadinessWitness,
                 false,
                 PredicateEvaluationSourceKind::kInputUpdate},
      SourceCase{TemporalRecordTriggerKind::kNotified,
                 MaxStartDelaySourceCoordinateKind::kReadinessWitness,
                 false,
                 PredicateEvaluationSourceKind::kInputUpdate},
      SourceCase{TemporalRecordTriggerKind::kDoorbell,
                 MaxStartDelaySourceCoordinateKind::kReadinessWitness,
                 false,
                 PredicateEvaluationSourceKind::kInputUpdate},
      SourceCase{TemporalRecordTriggerKind::kSpin,
                 MaxStartDelaySourceCoordinateKind::kReadinessWitness,
                 false,
                 PredicateEvaluationSourceKind::kInputUpdate},
      SourceCase{TemporalRecordTriggerKind::kAligned,
                 MaxStartDelaySourceCoordinateKind::kAlignedHighOrClosureProof,
                 false,
                 PredicateEvaluationSourceKind::kInputUpdate},
      SourceCase{TemporalRecordTriggerKind::kWindow,
                 MaxStartDelaySourceCoordinateKind::kWindowClose,
                 false,
                 PredicateEvaluationSourceKind::kInputUpdate},
      SourceCase{TemporalRecordTriggerKind::kPredicate,
                 MaxStartDelaySourceCoordinateKind::kPredicateDriver,
                 true,
                 PredicateEvaluationSourceKind::kInputUpdate},
      SourceCase{TemporalRecordTriggerKind::kPredicate,
                 MaxStartDelaySourceCoordinateKind::kScheduledTick,
                 true,
                 PredicateEvaluationSourceKind::kClock},
  };

  // Key setup: iterate all cases; for each, also try a mismatched coordinate and expect rejection.
  for (const auto& test_case : cases) {
    SCOPED_TRACE(static_cast<unsigned>(test_case.trigger));
    SCOPED_TRACE(static_cast<unsigned>(test_case.coordinate));
    auto trigger = cross_clock_max_start_delay_trigger(OnViolation::kEmitWitness);
    trigger.kind = test_case.trigger;
    trigger.conversion_id = 0U;
    auto max_start_delay = max_start_delay_plan(OnViolation::kEmitWitness);
    max_start_delay.source_coordinate = test_case.coordinate;
    const std::array max_start_delays{max_start_delay};

    PredicateTriggerPlanRecord predicate;
    const PredicateTriggerPlanRecord* predicate_ptr = nullptr;
    if (test_case.predicate) {
      predicate.owner = trigger.owner;
      predicate.graph_generation = kMaxStartDelayPlanGeneration;
      predicate.source = test_case.predicate_source;
      predicate.driver = test_case.predicate_source == PredicateEvaluationSourceKind::kInputUpdate
                             ? PortId{1U}
                             : PortId{};
      predicate_ptr = &predicate;
    }
    const auto binding =
        detail::bind_runtime_max_start_delay(max_start_delays,
                                             trigger,
                                             std::span<const ClockConversionRecord>{},
                                             kMaxStartDelayPlanGeneration,
                                             predicate_ptr);
    ASSERT_TRUE(binding);
    ASSERT_TRUE(binding->has_value());

    auto mismatched = max_start_delay;
    mismatched.source_coordinate =
        test_case.coordinate == MaxStartDelaySourceCoordinateKind::kTriggerSample
            ? MaxStartDelaySourceCoordinateKind::kWindowClose
            : MaxStartDelaySourceCoordinateKind::kTriggerSample;
    const std::array mismatched_max_start_delays{mismatched};
    EXPECT_EQ(detail::bind_runtime_max_start_delay(mismatched_max_start_delays,
                                                   trigger,
                                                   std::span<const ClockConversionRecord>{},
                                                   kMaxStartDelayPlanGeneration,
                                                   predicate_ptr)
                  .error()
                  .code,
              ErrorCode::kInvalidArgument);
  }
}

// REQ: RUNTIME-007
// REQ: RUNTIME-028
// REQ: RUNTIME-081
// REQ: RUNTIME-013
// REQ: RUNTIME-080
// GoogleTest fatal assertions establish optional engagement, and the negative test deliberately
// injects invalid enum representations; clang-tidy does not model either test-only pattern.
// NOLINTBEGIN(bugprone-unchecked-optional-access,clang-analyzer-optin.core.EnumCastOutOfRange)
// Conversion: kMaxStartDelaySourceClock -> kRuntimeClock, scale=3/2, offset=+5ns, budget=10ns.
// source=100 -> reference=155; latest_start=170 (exact); source=101 -> 156.5 rounds to 156,
// latest_start=171.5; reference=157 with source=101 is a contradiction -> kProtocolError.
TEST(DispatcherBindingTest, CrossClockMaxStartDelayAddsBudgetBeforeExactReferenceComparison) {
  const auto conversion = cross_clock_max_start_delay_conversion();
  const std::array conversions{conversion};
  constexpr std::array actions{
      OnViolation::kEmitWitness,
      OnViolation::kSafeDefault,
      OnViolation::kFailRun,
  };

  // Key setup: three violation policies (kEmitWitness, kSafeDefault, kFailRun) all share the same
  // arithmetic path; only the late-dispatch disposition differs.
  for (const OnViolation action : actions) {
    SCOPED_TRACE(static_cast<unsigned>(action));
    const std::array max_start_delays{cross_clock_max_start_delay_plan(action)};
    const auto trigger = cross_clock_max_start_delay_trigger(action);
    auto binding = detail::bind_runtime_max_start_delay(
        max_start_delays, trigger, conversions, kMaxStartDelayPlanGeneration);
    ASSERT_TRUE(binding);
    ASSERT_TRUE(binding->has_value());

    // 100 source ticks convert to 155 reference ticks. Adding the 10-tick
    // source-domain budget first produces the exact reference boundary 170.
    auto exact = detail::evaluate_runtime_max_start_delay(**binding,
                                                          TimePoint{100, kMaxStartDelaySourceClock},
                                                          TimePoint{155, kRuntimeClock},
                                                          TimePoint{170, kRuntimeClock});
    ASSERT_TRUE(exact);
    EXPECT_EQ(exact->disposition, detail::MaxStartDelayGateDisposition::kInvokeNominal);
    EXPECT_EQ(exact->latest_start, (TimePoint{170, kRuntimeClock}));
    EXPECT_EQ(exact->dispatch_start, (TimePoint{170, kRuntimeClock}));
    EXPECT_EQ(exact->checked_lateness_ns, 0);

    auto late = detail::evaluate_runtime_max_start_delay(**binding,
                                                         TimePoint{100, kMaxStartDelaySourceClock},
                                                         TimePoint{155, kRuntimeClock},
                                                         TimePoint{171, kRuntimeClock});
    ASSERT_TRUE(late);
    EXPECT_EQ(late->disposition,
              action == OnViolation::kFailRun
                  ? detail::MaxStartDelayGateDisposition::kFailRun
                  : detail::MaxStartDelayGateDisposition::kInvokeViolation);
    EXPECT_EQ(late->status.reason, ActivationReason::kMaxStartDelayExceeded);
    EXPECT_EQ(late->status.policy, action);
    EXPECT_EQ(late->latest_start, (TimePoint{170, kRuntimeClock}));
    EXPECT_EQ(late->dispatch_start, (TimePoint{171, kRuntimeClock}));
    EXPECT_EQ(late->checked_lateness_ns, 1);
  }

  const std::array fractional_max_start_delays{
      cross_clock_max_start_delay_plan(OnViolation::kEmitWitness)};
  auto fractional_binding = detail::bind_runtime_max_start_delay(
      fractional_max_start_delays,
      cross_clock_max_start_delay_trigger(OnViolation::kEmitWitness),
      conversions,
      kMaxStartDelayPlanGeneration);
  ASSERT_TRUE(fractional_binding);
  ASSERT_TRUE(fractional_binding->has_value());
  // 101 * 3/2 + 5 is 156.5 and materializes to 156 by ties-to-even.
  // The exact latest-start boundary is 171.5, so 171 is on time even though
  // the bounded evidence coordinate materializes to 172.
  auto fractional =
      detail::evaluate_runtime_max_start_delay(**fractional_binding,
                                               TimePoint{101, kMaxStartDelaySourceClock},
                                               TimePoint{156, kRuntimeClock},
                                               TimePoint{171, kRuntimeClock});
  ASSERT_TRUE(fractional);
  EXPECT_EQ(fractional->disposition, detail::MaxStartDelayGateDisposition::kInvokeNominal);
  EXPECT_EQ(fractional->latest_start, (TimePoint{172, kRuntimeClock}));
  EXPECT_EQ(fractional->dispatch_start, (TimePoint{171, kRuntimeClock}));
  EXPECT_EQ(fractional->checked_lateness_ns, 0);
  EXPECT_EQ(detail::evaluate_runtime_max_start_delay(**fractional_binding,
                                                     TimePoint{101, kMaxStartDelaySourceClock},
                                                     TimePoint{157, kRuntimeClock},
                                                     TimePoint{171, kRuntimeClock})
                .error()
                .code,
            ErrorCode::kProtocolError);

  auto source = source_binding(0U, detail::RuntimeEventKind::kInputArrival);
  source.source_clock = kMaxStartDelaySourceClock;
  source.reference_clock = kRuntimeClock;
  source.source_to_reference_conversion_id = conversion.id;
  source.max_start_delay = **fractional_binding;
  const std::array sources{source};
  auto coordinator = detail::RuntimeCandidateCoordinator::create(sources, 1U, 1U);
  ASSERT_TRUE(coordinator);
  detail::BoundedEventPortBinding ingress{1U};
  ASSERT_TRUE(ingress.push({
      .kind = source.candidate_kind,
      .source = source.source,
      .port = source.port,
      .source_decision_time = TimePoint{101, kMaxStartDelaySourceClock},
      .reference_time = TimePoint{156, kRuntimeClock},
      .source_generation = source.source_generation,
      .source_sequence = 1U,
  }));
  ASSERT_EQ(coordinator->ingest(ingress), 1U);
  auto frozen = coordinator->freeze_next();
  ASSERT_TRUE(frozen);
  EXPECT_EQ(frozen->witness.source_decision_time, (TimePoint{101, kMaxStartDelaySourceClock}));
  EXPECT_EQ(frozen->witness.reference_time, (TimePoint{156, kRuntimeClock}));
  EXPECT_EQ(frozen->candidate.ready.tag_time, (TagTime{156, 0U, 0U}));
  ASSERT_TRUE(coordinator->retire(*frozen));

  ASSERT_TRUE(ingress.push({
      .kind = source.candidate_kind,
      .source = source.source,
      .port = source.port,
      .source_decision_time = TimePoint{102, kMaxStartDelaySourceClock},
      .reference_time = TimePoint{159, kRuntimeClock},
      .source_generation = source.source_generation,
      .source_sequence = 2U,
  }));
  // Key assertion: event with reference_time=159 (contradicts source=102 expected ref=158) fails.
  EXPECT_EQ(coordinator->ingest(ingress).error().code, ErrorCode::kProtocolError);
}

// REQ: RUNTIME-007
// REQ: RUNTIME-013
// REQ: RUNTIME-080
// Negative tests for bind_runtime_max_start_delay: each tampered ClockConversionRecord field
// returns kInvalidArgument; arithmetic overflow during conversion returns kResourceExhausted.
TEST(DispatcherBindingTest, CrossClockMaxStartDelayFailsClosedOnTamperedConversionEvidence) {
  const auto max_start_delay = cross_clock_max_start_delay_plan(OnViolation::kEmitWitness);
  const std::array max_start_delays{max_start_delay};
  const auto trigger = cross_clock_max_start_delay_trigger(OnViolation::kEmitWitness);
  const auto valid = cross_clock_max_start_delay_conversion();

  EXPECT_EQ(detail::bind_runtime_max_start_delay(std::span<const MaxStartDelayPlanRecord>{},
                                                 trigger,
                                                 std::array{valid},
                                                 kMaxStartDelayPlanGeneration)
                .error()
                .code,
            ErrorCode::kInvalidArgument);
  const std::array duplicate_max_start_delays{max_start_delay, max_start_delay};
  EXPECT_EQ(
      detail::bind_runtime_max_start_delay(
          duplicate_max_start_delays, trigger, std::array{valid}, kMaxStartDelayPlanGeneration)
          .error()
          .code,
      ErrorCode::kInvalidArgument);

  auto max_start_delay_without_action = trigger;
  max_start_delay_without_action.violation.reset();
  EXPECT_EQ(detail::bind_runtime_max_start_delay(max_start_delays,
                                                 max_start_delay_without_action,
                                                 std::array{valid},
                                                 kMaxStartDelayPlanGeneration)
                .error()
                .code,
            ErrorCode::kInvalidArgument);
  auto action_without_max_start_delay = trigger;
  action_without_max_start_delay.max_start_delay_ns.reset();
  EXPECT_EQ(detail::bind_runtime_max_start_delay(max_start_delays,
                                                 action_without_max_start_delay,
                                                 std::array{valid},
                                                 kMaxStartDelayPlanGeneration)
                .error()
                .code,
            ErrorCode::kInvalidArgument);

  EXPECT_EQ(detail::bind_runtime_max_start_delay(max_start_delays,
                                                 trigger,
                                                 std::span<const ClockConversionRecord>{},
                                                 kMaxStartDelayPlanGeneration)
                .error()
                .code,
            ErrorCode::kInvalidArgument);

  {
    auto mismatched = max_start_delay;
    mismatched.source_coordinate = MaxStartDelaySourceCoordinateKind::kWindowClose;
    const std::array mismatched_max_start_delays{mismatched};
    EXPECT_EQ(
        detail::bind_runtime_max_start_delay(
            mismatched_max_start_delays, trigger, std::array{valid}, kMaxStartDelayPlanGeneration)
            .error()
            .code,
        ErrorCode::kInvalidArgument);
  }
  {
    auto scheduled_trigger = trigger;
    scheduled_trigger.kind = TemporalRecordTriggerKind::kClock;
    scheduled_trigger.trigger_clock = kRuntimeClock;
    auto impossible = max_start_delay;
    impossible.source_coordinate = MaxStartDelaySourceCoordinateKind::kScheduledTick;
    const std::array impossible_max_start_delays{impossible};
    EXPECT_EQ(detail::bind_runtime_max_start_delay(impossible_max_start_delays,
                                                   scheduled_trigger,
                                                   std::array{valid},
                                                   kMaxStartDelayPlanGeneration)
                  .error()
                  .code,
              ErrorCode::kInvalidArgument);
  }
  {
    auto invalid_action = max_start_delay;
    invalid_action.action = static_cast<OnViolation>(255U);
    invalid_action.callback_count_if_late = 1U;
    const std::array invalid_max_start_delays{invalid_action};
    EXPECT_EQ(
        detail::bind_runtime_max_start_delay(
            invalid_max_start_delays, trigger, std::array{valid}, kMaxStartDelayPlanGeneration)
            .error()
            .code,
        ErrorCode::kInvalidArgument);
  }
  {
    auto no_max_start_delay = trigger;
    no_max_start_delay.max_start_delay_ns.reset();
    no_max_start_delay.violation.reset();
    EXPECT_EQ(detail::bind_runtime_max_start_delay(std::span<const MaxStartDelayPlanRecord>{},
                                                   no_max_start_delay,
                                                   std::array{valid},
                                                   kMaxStartDelayPlanGeneration)
                  .error()
                  .code,
              ErrorCode::kInvalidArgument);
    no_max_start_delay.conversion_id = 0U;
    auto unbound = detail::bind_runtime_max_start_delay(std::span<const MaxStartDelayPlanRecord>{},
                                                        no_max_start_delay,
                                                        std::array{valid},
                                                        kMaxStartDelayPlanGeneration);
    ASSERT_TRUE(unbound);
    EXPECT_FALSE(unbound->has_value());
  }
  {
    auto stray = max_start_delay;
    stray.owner.id = 2U;
    const std::array inventory{max_start_delay, stray};
    const std::array triggers{trigger};
    EXPECT_EQ(detail::validate_runtime_max_start_delay_inventory(
                  inventory, triggers, {}, std::array{valid}, kMaxStartDelayPlanGeneration)
                  .error()
                  .code,
              ErrorCode::kInvalidArgument);
  }
  {
    auto predicate_trigger = trigger;
    predicate_trigger.kind = TemporalRecordTriggerKind::kPredicate;
    auto predicate_max_start_delay = max_start_delay;
    predicate_max_start_delay.source_coordinate =
        MaxStartDelaySourceCoordinateKind::kPredicateDriver;
    const std::array predicate_max_start_delays{predicate_max_start_delay};
    PredicateTriggerPlanRecord predicate;
    predicate.owner = trigger.owner;
    predicate.graph_generation = kMaxStartDelayPlanGeneration;
    predicate.source = PredicateEvaluationSourceKind::kInputUpdate;
    predicate.driver = PortId{1U};
    auto predicate_binding = detail::bind_runtime_max_start_delay(predicate_max_start_delays,
                                                                  predicate_trigger,
                                                                  std::array{valid},
                                                                  kMaxStartDelayPlanGeneration,
                                                                  &predicate);
    ASSERT_TRUE(predicate_binding);
    ASSERT_TRUE(predicate_binding->has_value());
    predicate_max_start_delay.source_coordinate = MaxStartDelaySourceCoordinateKind::kScheduledTick;
    const std::array mismatched_predicate_max_start_delays{predicate_max_start_delay};
    EXPECT_EQ(detail::bind_runtime_max_start_delay(mismatched_predicate_max_start_delays,
                                                   predicate_trigger,
                                                   std::array{valid},
                                                   kMaxStartDelayPlanGeneration,
                                                   &predicate)
                  .error()
                  .code,
              ErrorCode::kInvalidArgument);
  }

  // Key setup: lambda verifies that each tampered conversion field is rejected at bind time.
  const auto expect_rejected = [&](ClockConversionRecord tampered) {
    const std::array conversions{tampered};
    const auto result = detail::bind_runtime_max_start_delay(
        max_start_delays, trigger, conversions, kMaxStartDelayPlanGeneration);
    ASSERT_FALSE(result);
    EXPECT_EQ(result.error().code, ErrorCode::kInvalidArgument);
  };
  {
    auto tampered = valid;
    tampered.id = 2U;
    expect_rejected(tampered);
  }
  {
    auto tampered = valid;
    tampered.from = kRuntimeClock;
    expect_rejected(tampered);
  }
  {
    auto tampered = valid;
    tampered.to = kMaxStartDelaySourceClock;
    expect_rejected(tampered);
  }
  {
    auto tampered = valid;
    tampered.plan_generation = kMaxStartDelayPlanGeneration + 1U;
    expect_rejected(tampered);
  }
  {
    auto tampered = valid;
    tampered.rounding = static_cast<ClockRoundingMode>(255U);
    expect_rejected(tampered);
  }
  {
    auto tampered = valid;
    tampered.overflow = static_cast<ClockOverflowPolicy>(255U);
    expect_rejected(tampered);
  }

  auto overflow_conversion = valid;
  overflow_conversion.scale = {.numerator = 1, .denominator = 2U};
  overflow_conversion.offset_ns = 0;
  const std::array overflow_conversions{overflow_conversion};
  auto overflow_binding = detail::bind_runtime_max_start_delay(
      max_start_delays, trigger, overflow_conversions, kMaxStartDelayPlanGeneration);
  ASSERT_TRUE(overflow_binding);
  ASSERT_TRUE(overflow_binding->has_value());
  constexpr std::int64_t kOverflowSource = std::numeric_limits<std::int64_t>::max() - 5;
  // Key assertion: source near INT64_MAX overflows the scaled conversion -> kResourceExhausted.
  EXPECT_EQ(detail::evaluate_runtime_max_start_delay(
                **overflow_binding,
                TimePoint{kOverflowSource, kMaxStartDelaySourceClock},
                TimePoint{kOverflowSource / 2, kRuntimeClock},
                TimePoint{0, kRuntimeClock})
                .error()
                .code,
            ErrorCode::kResourceExhausted);
}

// REQ: CONTRACT-011
// REQ: RUNTIME-002
// REQ: RUNTIME-007
// NOLINTEND(bugprone-unchecked-optional-access,clang-analyzer-optin.core.EnumCastOutOfRange)
// kClockTick source -> coordinator -> frozen candidate; predicate lambda inspects reference_time
// and tag_time from the frozen source; must match the event's reference timestamp exactly.
TEST(DispatcherBindingTest, PredicateConsumesTheExactFrozenSourceIdentity) {
  const std::array sources{
      source_binding(0U, detail::RuntimeEventKind::kClockTick),
  };
  auto coordinator = detail::RuntimeCandidateCoordinator::create(sources, 1U, 1U);
  ASSERT_TRUE(coordinator);
  detail::BoundedEventPortBinding binding{1U};
  ASSERT_TRUE(binding.push(runtime_event(sources[0], 1U, 77)));
  ASSERT_TRUE(coordinator->ingest(binding));
  auto frozen = coordinator->freeze_next();
  ASSERT_TRUE(frozen);

  // Key setup: predicate checks reference_time==77 and tag_time==77 from the frozen candidate.
  auto predicate_binding =
      predicate("dispatcher-source", [](const PredicateEvaluationContext& context) noexcept {
        return context.reference_time().timestamp_ns == 77 && context.tag_time().logical_ns == 77;
      });
  auto storage = detail::ContractTestAccess::clone_predicate_storage(predicate_binding);
  ASSERT_TRUE(storage);
  detail::PredicateEvaluatorEngine<1U, 1U> engine{
      std::move(*storage), false, 0U, PredicateFiring::kOnEveryTrueEvaluation, 1U};
  auto decision = engine.evaluate(detail::PredicateFrozenSource{
      .reference_time = frozen->witness.reference_time,
      .tag = frozen->candidate.tag,
      .tag_time = frozen->candidate.ready.tag_time,
  });
  // Key assertion: evaluate returns kNominalCompute and firing_accepted=true for the matching time.
  ASSERT_TRUE(decision);
  EXPECT_TRUE(decision->evaluator_called);
  EXPECT_TRUE(decision->firing_accepted);
  EXPECT_EQ(decision->disposition, detail::PredicateCandidateDisposition::kNominalCompute);
  EXPECT_TRUE(coordinator->retire(*frozen));
}

// REQ: RUNTIME-007
// Plan-backed PredicateEvaluatorEngine: happy-path bool predicate (sample.data==7) and error
// predicate (returns unexpected<Error>); engine preserves frozen source identity in both decisions.
TEST(DispatcherBindingTest, PredicateEvaluatorConsumesPlanRecordsAndPreservesFrozenContext) {
  auto binding = predicate("plan-backed",
                           [](const PredicateEvaluationContext&,
                              const PredicateInputView<std::uint32_t>& input) noexcept {
                             return input.sample().data == 7U;
                           });
  auto storage = detail::ContractTestAccess::clone_predicate_storage(binding);
  ASSERT_TRUE(storage);

  const PredicateTriggerPlanRecord trigger{
      .id = "operator/plan-backed",
      .owner = OperatorRef{1U, 1U, 0U},
      .graph_generation = 1U,
      .source = PredicateEvaluationSourceKind::kInputUpdate,
      .driver = PortId{1U},
      .firing = PredicateFiring::kOnEveryTrueEvaluation,
      .state_generation = 1U,
  };
  const PredicateCallablePlanRecord callable{
      .predicate_id = trigger.id,
      .result_kind = PredicateResultKind::kBool,
      .disposition = PredicateCallableDisposition::kAdmitted,
  };
  const std::array inputs{PredicateInputPlanRecord{
      .predicate_id = trigger.id,
      .input = trigger.driver,
      .argument_ordinal = 0U,
      .implicit_driver = true,
      .selector = InputSelectorKind::kExact,
      .missing = TemporalRecordMissingKind::kUnspecified,
      .reference_clock = kRuntimeClock,
      .cache_capacity = 1U,
  }};
  auto engine = detail::PredicateEvaluatorEngine<2U, 2U>::create(
      trigger, inputs, callable, std::move(*storage));
  ASSERT_TRUE(engine);

  const std::uint64_t generation = 1U;
  const Sample<std::uint32_t> sample{7U, SampleMetadata{.sequence = 1U}};
  auto view = detail::PredicateBindingAccess::input<std::uint32_t>({
      .sample = &sample,
      .selection = {InputState::kSelected, SelectionReason::kNone},
      .active_generation = &generation,
      .expected_generation = generation,
  });
  ASSERT_TRUE(view);
  const std::array frozen_inputs{detail::PredicateFrozenInput{
      .port = trigger.driver,
      .selection = {InputState::kSelected, SelectionReason::kNone},
      .typed_view = &*view,
  }};
  const detail::PredicateFrozenSource frozen{
      .source = detail::make_source_id(1U),
      .source_sequence = 1U,
      .driver_port = trigger.driver,
      .reference_time = TimePoint{50, kRuntimeClock},
      .tag = Tag{1U, 1U, 0U},
      .tag_time = TagTime{50, 0U, 0U},
  };
  // Key assertion: bool predicate returns true for sample.data==7; frozen source fields propagated.
  auto decision = engine->evaluate(frozen, frozen_inputs);
  ASSERT_TRUE(decision);
  EXPECT_TRUE(decision->firing_accepted);
  EXPECT_EQ(decision->predicate_id, trigger.id);
  EXPECT_EQ(decision->operator_id, trigger.owner.id);
  EXPECT_EQ(decision->source.source, frozen.source);
  EXPECT_EQ(decision->source.source_sequence, frozen.source_sequence);
  EXPECT_EQ(decision->state_generation, trigger.state_generation);
  EXPECT_FALSE(decision->prior_truth);

  auto error_binding =
      predicate("plan-error",
                [](const PredicateEvaluationContext&,
                   const PredicateInputView<std::uint32_t>&) noexcept -> expected<bool, Error> {
                  return make_unexpected(Error{ErrorCode::kFailure, "preserved predicate detail"});
                });
  auto error_storage = detail::ContractTestAccess::clone_predicate_storage(error_binding);
  ASSERT_TRUE(error_storage);
  auto error_trigger = trigger;
  error_trigger.id = "operator/plan-error";
  auto error_callable = callable;
  error_callable.predicate_id = error_trigger.id;
  error_callable.result_kind = PredicateResultKind::kExpectedBoolError;
  auto error_inputs = inputs;
  error_inputs[0].predicate_id = error_trigger.id;
  auto error_engine = detail::PredicateEvaluatorEngine<2U, 2U>::create(
      error_trigger, error_inputs, error_callable, std::move(*error_storage));
  ASSERT_TRUE(error_engine);
  auto evaluator_error = error_engine->evaluate(frozen, frozen_inputs);
  ASSERT_TRUE(evaluator_error);
  EXPECT_EQ(evaluator_error->disposition, detail::PredicateCandidateDisposition::kEvaluatorError);
  EXPECT_EQ(evaluator_error->evaluator_error.code, ErrorCode::kFailure);
  EXPECT_EQ(evaluator_error->evaluator_error.message, "preserved predicate detail");
  EXPECT_EQ(evaluator_error->predicate_id, error_trigger.id);
  EXPECT_EQ(evaluator_error->operator_id, error_trigger.owner.id);
  EXPECT_EQ(evaluator_error->source.source, frozen.source);
  EXPECT_EQ(evaluator_error->source.source_sequence, frozen.source_sequence);
  EXPECT_EQ(evaluator_error->source.tag, frozen.tag);
  EXPECT_EQ(evaluator_error->state_generation, error_trigger.state_generation);

  auto wrong_driver = frozen;
  wrong_driver.driver_port = PortId{2U};
  EXPECT_EQ(engine->evaluate(wrong_driver, frozen_inputs).error().code,
            ErrorCode::kInvalidArgument);

  auto malformed_trigger = trigger;
  malformed_trigger.compute_adopts_evaluator_snapshot = false;
  auto second_storage = detail::ContractTestAccess::clone_predicate_storage(binding);
  ASSERT_TRUE(second_storage);
  const auto malformed_engine = detail::PredicateEvaluatorEngine<2U, 2U>::create(
      malformed_trigger, inputs, callable, std::move(*second_storage));
  ASSERT_FALSE(malformed_engine);
  EXPECT_EQ(malformed_engine.error().code, ErrorCode::kInvalidArgument);
}

// REQ: RUNTIME-007
// PredicateEvaluatorEngine[kOnFalseToTrue]: reset() for each PredicateResetReason advances the
// generation and fires on the first F->T evaluation; missing input / false return / evaluator error
// never change the reset record.
TEST(DispatcherBindingTest, PredicateResetRecordUsesOnlyApprovedGenerationBoundaries) {
  auto true_binding =
      predicate("reset-record", [](const PredicateEvaluationContext&) noexcept { return true; });
  auto true_storage = detail::ContractTestAccess::clone_predicate_storage(true_binding);
  ASSERT_TRUE(true_storage);
  detail::PredicateEvaluatorEngine<1U, 1U> engine{
      std::move(*true_storage), false, 0U, PredicateFiring::kOnFalseToTrue, 1U};
  EXPECT_EQ(engine.last_reset_record(),
            (detail::PredicateResetRecord{detail::PredicateResetReason::kColdRunStart, 1U}));

  constexpr std::array reset_reasons{
      detail::PredicateResetReason::kPlanGeneration,
      detail::PredicateResetReason::kColdRunStart,
      detail::PredicateResetReason::kOperatorRestart,
      detail::PredicateResetReason::kSessionRebind,
      detail::PredicateResetReason::kProducerEpoch,
      detail::PredicateResetReason::kClockDiscontinuity,
  };
  // Key setup: cycle through all reset reasons; each advances the generation and re-arms F->T.
  std::uint64_t generation = 2U;
  for (const auto reason : reset_reasons) {
    engine.reset(reason, generation);
    EXPECT_EQ(engine.last_reset_record(), (detail::PredicateResetRecord{reason, generation}));
    auto first = engine.evaluate(detail::PredicateFrozenSource{
        .reference_time = TimePoint{static_cast<std::int64_t>(generation), kRuntimeClock},
        .tag = Tag{1U, static_cast<std::uint32_t>(generation), 0U},
        .tag_time = TagTime{static_cast<std::int64_t>(generation), 0U, 0U},
    });
    ASSERT_TRUE(first);
    EXPECT_TRUE(first->firing_accepted);
    EXPECT_EQ(first->reset_record, (detail::PredicateResetRecord{reason, generation}));
    ++generation;
  }

  const auto stable_record = engine.last_reset_record();
  auto repeated_true = engine.evaluate(detail::PredicateFrozenSource{
      .reference_time = TimePoint{100, kRuntimeClock},
      .tag = Tag{2U, static_cast<std::uint32_t>(stable_record.generation), 0U},
      .tag_time = TagTime{100, 0U, 0U},
  });
  ASSERT_TRUE(repeated_true);
  EXPECT_FALSE(repeated_true->firing_accepted);
  EXPECT_TRUE(engine.previous_truth());
  EXPECT_EQ(engine.last_reset_record(), stable_record);

  // Missing-input suppression never invokes the evaluator and cannot create a
  // reset. This models pause/no-witness and callback failure as the same
  // absence of an accepted reset operation: only reset(...) changes the record.
  auto missing_storage = detail::ContractTestAccess::clone_predicate_storage(true_binding);
  ASSERT_TRUE(missing_storage);
  detail::PredicateEvaluatorEngine<1U, 1U> missing_engine{
      std::move(*missing_storage), false, 1U, PredicateFiring::kOnFalseToTrue, 11U};
  const std::array missing_inputs{detail::PredicateFrozenInput{
      .port = PortId{1U},
      .selection = {InputState::kAbsent, SelectionReason::kNeverSeen},
  }};
  auto missing = missing_engine.evaluate(
      detail::PredicateFrozenSource{
          .reference_time = TimePoint{11, kRuntimeClock},
          .tag = Tag{1U, 11U, 0U},
          .tag_time = TagTime{11, 0U, 0U},
      },
      missing_inputs);
  ASSERT_TRUE(missing);
  EXPECT_FALSE(missing->evaluator_called);
  EXPECT_EQ(missing->disposition, detail::PredicateCandidateDisposition::kNoFire);
  EXPECT_EQ(missing_engine.last_reset_record(),
            (detail::PredicateResetRecord{detail::PredicateResetReason::kColdRunStart, 11U}));

  auto false_binding =
      predicate("false-no-reset", [](const PredicateEvaluationContext&) noexcept { return false; });
  auto false_storage = detail::ContractTestAccess::clone_predicate_storage(false_binding);
  ASSERT_TRUE(false_storage);
  detail::PredicateEvaluatorEngine<1U, 1U> false_engine{
      std::move(*false_storage), false, 0U, PredicateFiring::kOnFalseToTrue, 12U};
  auto false_result = false_engine.evaluate(detail::PredicateFrozenSource{
      .reference_time = TimePoint{12, kRuntimeClock},
      .tag = Tag{1U, 12U, 0U},
      .tag_time = TagTime{12, 0U, 0U},
  });
  ASSERT_TRUE(false_result);
  EXPECT_FALSE(false_result->value);
  EXPECT_EQ(false_engine.last_reset_record(),
            (detail::PredicateResetRecord{detail::PredicateResetReason::kColdRunStart, 12U}));

  auto error_binding = predicate(
      "error-no-reset", [](const PredicateEvaluationContext&) noexcept -> expected<bool, Error> {
        return make_unexpected(Error{ErrorCode::kFailure});
      });
  auto error_storage = detail::ContractTestAccess::clone_predicate_storage(error_binding);
  ASSERT_TRUE(error_storage);
  detail::PredicateEvaluatorEngine<1U, 1U> error_engine{
      std::move(*error_storage), false, 0U, PredicateFiring::kOnFalseToTrue, 13U};
  auto error_result = error_engine.evaluate(detail::PredicateFrozenSource{
      .reference_time = TimePoint{13, kRuntimeClock},
      .tag = Tag{1U, 13U, 0U},
      .tag_time = TagTime{13, 0U, 0U},
  });
  ASSERT_TRUE(error_result);
  EXPECT_EQ(error_result->disposition, detail::PredicateCandidateDisposition::kEvaluatorError);
  // Key assertion: evaluator error does not alter last_reset_record; it stays at kColdRunStart.
  EXPECT_EQ(error_engine.last_reset_record(),
            (detail::PredicateResetRecord{detail::PredicateResetReason::kColdRunStart, 13U}));
}

// REQ: RUNTIME-007
// Conformance vector: three engine scenarios (strict aligned, misaligned, partial-closed+predicate)
// run twice. Both passes must yield identical observation arrays, verifying no hidden mutable state
// causes non-determinism on re-run.
TEST(DispatcherBindingTest, SyntheticClockProducesCanonicalDirectTemporalConformanceVector) {
  std::array<std::array<std::int64_t, 12U>, 2U> observations{};
  // Key setup: each pass reconstructs all three engines from scratch under the same clock schedule.
  for (std::size_t pass = 0U; pass < observations.size(); ++pass) {
    SyntheticClock clock;
    clock.advance_to(100);
    const std::array missing{detail::AlignedMissingPolicy{}, detail::AlignedMissingPolicy{}};
    const std::array open{detail::AlignmentClosureProof::kOpen,
                          detail::AlignmentClosureProof::kOpen};

    detail::AlignedJoinEngine strict{2U, 2U};
    ASSERT_TRUE(strict.push(0U, temporal_sample(clock.timestamp(), 1U, 1U)));
    clock.advance_to(101);
    ASSERT_TRUE(strict.push(1U, temporal_sample(clock.timestamp(), 1U, 1U)));
    auto strict_selection = strict.select(std::chrono::nanoseconds{1}, missing, open);
    ASSERT_TRUE(strict_selection);
    ASSERT_EQ(strict_selection->members().size(), 2U);

    detail::AlignedJoinEngine misaligned{2U, 2U};
    ASSERT_TRUE(misaligned.push(0U, temporal_sample(100, 1U, 1U)));
    ASSERT_TRUE(misaligned.push(1U, temporal_sample(110, 1U, 1U)));
    auto misaligned_selection = misaligned.select(std::chrono::nanoseconds{1}, missing, open);
    ASSERT_TRUE(misaligned_selection);

    detail::AlignedJoinEngine partial{2U, 2U};
    ASSERT_TRUE(partial.push(0U, temporal_sample(120, 1U, 1U)));
    const std::array partial_missing{
        detail::AlignedMissingPolicy{},
        detail::AlignedMissingPolicy{.kind = detail::AlignedMissingPolicyKind::kSafeDefault},
    };
    const std::array partial_closed{detail::AlignmentClosureProof::kOpen,
                                    detail::AlignmentClosureProof::kEndOfStream};
    auto partial_selection = partial.select(std::chrono::nanoseconds{0},
                                            partial_missing,
                                            partial_closed,
                                            TimePoint{120, kRuntimeClock});
    ASSERT_TRUE(partial_selection);

    auto predicate_binding =
        predicate("synthetic-clock", [](const PredicateEvaluationContext& context) noexcept {
          return context.reference_time().timestamp_ns == 120;
        });
    auto predicate_storage = detail::ContractTestAccess::clone_predicate_storage(predicate_binding);
    ASSERT_TRUE(predicate_storage);
    detail::PredicateEvaluatorEngine<1U, 1U> predicate_engine{
        std::move(*predicate_storage), false, 0U, PredicateFiring::kOnFalseToTrue, 1U};
    auto predicate_decision = predicate_engine.evaluate(detail::PredicateFrozenSource{
        .source = detail::make_source_id(7U),
        .source_sequence = 1U,
        .reference_time = TimePoint{120, kRuntimeClock},
        .tag = Tag{1U, 1U, 0U},
        .tag_time = TagTime{120, 0U, 0U},
    });
    ASSERT_TRUE(predicate_decision);

    observations[pass] = {
        static_cast<std::int64_t>(strict_selection->disposition()),
        strict_selection->reference_time().timestamp_ns,
        static_cast<std::int64_t>(strict_selection->members()[0].source_sequence),
        static_cast<std::int64_t>(strict_selection->members()[1].source_sequence),
        static_cast<std::int64_t>(misaligned_selection->disposition()),
        static_cast<std::int64_t>(misaligned_selection->reason()),
        static_cast<std::int64_t>(partial_selection->disposition()),
        partial_selection->reference_time().timestamp_ns,
        static_cast<std::int64_t>(partial_selection->selections()[1].state),
        static_cast<std::int64_t>(partial_selection->closure_evidence()[1]),
        static_cast<std::int64_t>(predicate_decision->disposition),
        predicate_decision->firing_accepted ? 1 : 0,
    };

    ASSERT_TRUE(strict_selection->commit());
    ASSERT_TRUE(misaligned_selection->commit());
    ASSERT_TRUE(partial_selection->commit());
  }

  // Key assertion: pass 0 and pass 1 observation vectors are bit-identical.
  EXPECT_EQ(observations[0], observations[1]);
  EXPECT_EQ(observations[0][0],
            static_cast<std::int64_t>(detail::AlignedCandidateDisposition::kNominal));
  EXPECT_EQ(observations[0][1], 101);
  EXPECT_EQ(observations[0][4],
            static_cast<std::int64_t>(detail::AlignedCandidateDisposition::kRetireMisaligned));
  EXPECT_EQ(observations[0][5], static_cast<std::int64_t>(SelectionReason::kAlignmentWindowMiss));
  EXPECT_EQ(observations[0][6],
            static_cast<std::int64_t>(detail::AlignedCandidateDisposition::kDegraded));
  EXPECT_EQ(observations[0][8], static_cast<std::int64_t>(InputState::kSafeDefault));
  EXPECT_EQ(observations[0][10],
            static_cast<std::int64_t>(detail::PredicateCandidateDisposition::kNominalCompute));
  EXPECT_EQ(observations[0][11], 1);
}

// REQ: RUNTIME-002
// REQ: RUNTIME-004
// REQ: RUNTIME-009
// Q61-T05: report reasons never become independent READY witnesses.
// GoogleTest fatal assertions terminate this policy test before optional access, but clang-tidy
// does not model the macro's control flow.
// NOLINTBEGIN(bugprone-unchecked-optional-access)
// producer -> EndpointClockHealthReceiver[port=1] -> source -> coordinator -> dispatcher;
// health states exercised in order: absent(skip) -> healthy -> degraded -> stale(skip) ->
// unhealthy(failRun) -> epoch-reset(skip).
TEST(DispatcherBindingTest, AppliesFrozenClockHealthDispositionOnlyToOtherwiseReadyWork) {
  constexpr detail::EndpointClockHealthBindingIdentity health_binding{
      .run_generation = 1U,
      .producer_epoch = 1U,
      .clock_epoch = 1U,
  };
  auto receiver = detail::EndpointClockHealthReceiver::create(monitored_health_plan(PortId{1U}),
                                                              health_binding);
  ASSERT_TRUE(receiver);

  // Key setup: source wired to the health receiver; each submit_report+dispatch_pending advances
  // the health state visible to the next freeze().
  auto source = source_binding(0U, detail::RuntimeEventKind::kInputArrival);
  source.clock_health = &*receiver;
  const std::array sources{source};
  auto coordinator = detail::RuntimeCandidateCoordinator::create(sources, 1U, 1U);
  ASSERT_TRUE(coordinator);
  detail::BoundedEventPortBinding binding{1U};
  const SchedulerPlanRecord plan = scheduler_plan();
  auto dispatcher = detail::Dispatcher::create(plan, 1U);
  ASSERT_TRUE(dispatcher);

  const auto freeze = [&](std::uint64_t sequence, std::int64_t reference_ns) {
    EXPECT_TRUE(binding.push(runtime_event(source, sequence, reference_ns)));
    EXPECT_TRUE(coordinator->ingest(binding));
    return coordinator->freeze_next();
  };

  // Initial/reset absence is observed only after the source is otherwise
  // READY. It consumes a Tag but never enters user compute.
  auto absent = freeze(1U, 10);
  ASSERT_TRUE(absent);
  EXPECT_EQ(absent->candidate.clock_health_disposition, EndpointClockHealthDisposition::kSkip);
  EXPECT_EQ(absent->clock_health_generation, 1U);
  const std::array absent_set{absent->candidate};
  ASSERT_TRUE(dispatcher->freeze(absent_set));
  EXPECT_EQ(dispatcher->active_count(), 0U);
  EXPECT_EQ(dispatcher->select().error().code, ErrorCode::kNotReady);
  ASSERT_TRUE(coordinator->retire(*absent));

  ASSERT_TRUE(
      receiver->submit_report(ClockHealthReport::healthy(), health_binding.producer_epoch, 1U, 1U));
  ASSERT_TRUE(receiver->dispatch_pending(20));
  EXPECT_EQ(coordinator->pending_count(), 0U);
  auto healthy = freeze(2U, 20);
  ASSERT_TRUE(healthy);
  EXPECT_EQ(healthy->candidate.clock_health_disposition,
            EndpointClockHealthDisposition::kNotApplicable);
  const std::array healthy_set{healthy->candidate};
  ASSERT_TRUE(dispatcher->freeze(healthy_set));
  auto healthy_selection = dispatcher->select();
  ASSERT_TRUE(healthy_selection);
  EXPECT_EQ(healthy_selection->clock_health_disposition,
            EndpointClockHealthDisposition::kNotApplicable);
  ASSERT_TRUE(dispatcher->complete(*healthy_selection));
  ASSERT_TRUE(coordinator->retire(*healthy));

  ASSERT_TRUE(receiver->submit_report(ClockHealthReport::degraded(ClockHealthReason::kClockStalled),
                                      health_binding.producer_epoch,
                                      2U,
                                      2U));
  ASSERT_TRUE(receiver->dispatch_pending(30));
  EXPECT_EQ(coordinator->pending_count(), 0U);
  auto degraded = freeze(3U, 30);
  ASSERT_TRUE(degraded);
  EXPECT_EQ(degraded->candidate.clock_health_disposition,
            EndpointClockHealthDisposition::kDegraded);
  const std::array degraded_set{degraded->candidate};
  ASSERT_TRUE(dispatcher->freeze(degraded_set));
  auto degraded_selection = dispatcher->select();
  ASSERT_TRUE(degraded_selection);
  EXPECT_EQ(degraded_selection->clock_health_disposition,
            EndpointClockHealthDisposition::kDegraded);
  auto activation = detail::ActivationSelectionState::create(0U, 0U);
  ASSERT_TRUE(activation);
  auto activation_view = activation->begin(*degraded, {}, {});
  ASSERT_TRUE(activation_view);
  EXPECT_EQ(activation_view->disposition, detail::MissingAggregateDisposition::kDegraded);
  EXPECT_EQ(activation_view->clock_health_disposition, EndpointClockHealthDisposition::kDegraded);
  ASSERT_TRUE(activation->retire(activation_view->tag));
  ASSERT_TRUE(dispatcher->complete(*degraded_selection));
  ASSERT_TRUE(coordinator->retire(*degraded));

  const auto expiry = receiver->expiry_ticket();
  ASSERT_TRUE(expiry.has_value());
  auto expired = receiver->expire_if_due(*expiry, expiry->expiry_deadline_ns);
  ASSERT_TRUE(expired);
  ASSERT_TRUE(*expired);
  auto stale = freeze(4U, 130);
  ASSERT_TRUE(stale);
  EXPECT_EQ(stale->candidate.clock_health_disposition, EndpointClockHealthDisposition::kSkip);
  const std::array stale_set{stale->candidate};
  ASSERT_TRUE(dispatcher->freeze(stale_set));
  EXPECT_EQ(dispatcher->select().error().code, ErrorCode::kNotReady);
  ASSERT_TRUE(coordinator->retire(*stale));

  ASSERT_TRUE(
      receiver->submit_report(ClockHealthReport::unhealthy(ClockHealthReason::kMonitorFault),
                              health_binding.producer_epoch,
                              3U,
                              3U));
  ASSERT_TRUE(receiver->dispatch_pending(140));
  auto unhealthy = freeze(5U, 140);
  ASSERT_TRUE(unhealthy);
  EXPECT_EQ(unhealthy->candidate.clock_health_disposition,
            EndpointClockHealthDisposition::kFailRun);
  const std::array unhealthy_set{unhealthy->candidate};
  // Key assertion: unhealthy report causes dispatcher.freeze() to propagate kFailure immediately.
  EXPECT_EQ(dispatcher->freeze(unhealthy_set).error().code, ErrorCode::kFailure);
  ASSERT_TRUE(coordinator->retire(*unhealthy));

  const detail::EndpointClockHealthBindingIdentity reset_binding{
      .run_generation = health_binding.run_generation,
      .producer_epoch = health_binding.producer_epoch + 1U,
      .clock_epoch = health_binding.clock_epoch,
  };
  ASSERT_TRUE(receiver->reset(
      detail::EndpointClockHealthUpdateOrigin::kProducerEpochReset, reset_binding, 150));
  auto reset = freeze(6U, 150);
  ASSERT_TRUE(reset);
  EXPECT_EQ(reset->candidate.clock_health_disposition, EndpointClockHealthDisposition::kSkip);
  ASSERT_TRUE(coordinator->retire(*reset));
}

// REQ: RUNTIME-002
// REQ: RUNTIME-007
// NOLINTEND(bugprone-unchecked-optional-access)
// kSpin source -> coordinator[max_outstanding=1] + activation[max_inputs=0, max_selected=0];
// full ingest->freeze->begin->retire->retire round-trip must perform zero heap allocations.
TEST(DispatcherBindingTest, WarmedIngressFreezeAndActivationPathsDoNotAllocate) {
  const std::array sources{
      source_binding(0U, detail::RuntimeEventKind::kSpin),
  };
  auto coordinator = detail::RuntimeCandidateCoordinator::create(sources, 1U, 1U);
  auto activation = detail::ActivationSelectionState::create(0U, 0U);
  ASSERT_TRUE(coordinator);
  ASSERT_TRUE(activation);
  EXPECT_EQ(coordinator->max_outstanding(), 1U);
  EXPECT_EQ(activation->max_inputs(), 0U);
  EXPECT_EQ(activation->max_selected(), 0U);

  // Key setup: coordinator and activation pre-warmed at max sizes before the counter starts.
  detail::BoundedEventPortBinding binding{1U};
  ASSERT_TRUE(binding.push(runtime_event(sources[0], 1U, 90)));
  std::size_t allocations = 0U;
  bool succeeded = false;
  {
    const test::ScopedAllocationCounter counter;
    auto ingested = coordinator->ingest(binding);
    auto frozen = coordinator->freeze_next();
    if (ingested && *ingested == 1U && frozen) {
      auto selected = activation->begin(*frozen, {}, {});
      if (selected) {
        const auto retired_activation = activation->retire(selected->tag);
        const auto retired_source = coordinator->retire(*frozen);
        succeeded = retired_activation && retired_source;
      }
    }
    allocations = counter.count();
  }
  EXPECT_TRUE(succeeded);
  // Key assertion: ScopedAllocationCounter reports zero allocations for the hot path.
  EXPECT_EQ(allocations, 0U);
  EXPECT_EQ(coordinator->outstanding_count(), 0U);
}

// REQ: RUNTIME-017
// REQ: CHAN-131
// The Exact half of the reconcile's eligibility rule, stated as its consequence rather than as the
// rule itself. kExact requires capture_time == reference_time, so a retained record is the only
// record that can ever satisfy a witness at its coordinate: no later sample substitutes for it,
// however fresh. Releasing it is therefore an unconditional loss rather than a trade, and no
// ordering of the release can make it safe -- which is why the reconcile refuses kExact outright
// instead of gating it the way it gates kLatestWithin.
//
// Both selectors are exercised against the same eviction so the asymmetry is the assertion. Under
// kLatestWithin the successor satisfies the witness and the trade is sound; under kExact the
// witness that the retained record alone could satisfy is stranded.
TEST(DispatcherBindingTest, EvictingAnExactRetentionStrandsTheWitnessNoSuccessorCanSatisfy) {
  constexpr std::int64_t kRetainedCaptureNs = 10;
  constexpr std::int64_t kSuccessorCaptureNs = 30;

  struct Vector {
    InputSelectorKind selector{};
    // The coordinate each selector's witness is formed at: its own capture for kExact, and the
    // successor's for kLatestWithin, which is where each is meant to be satisfiable.
    std::int64_t witness_reference_ns{};
    bool successor_satisfies{};
  };
  constexpr std::array vectors{
      Vector{InputSelectorKind::kExact, kRetainedCaptureNs, false},
      Vector{InputSelectorKind::kLatestWithin, kSuccessorCaptureNs, true},
  };

  for (const Vector& vector : vectors) {
    SCOPED_TRACE(vector.selector == InputSelectorKind::kExact ? "kExact" : "kLatestWithin");
    const TimePoint witness{vector.witness_reference_ns, kRuntimeClock};

    // Control: the retained record, still held, satisfies this witness. Without this the negative
    // half below cannot distinguish "the eviction stranded the witness" from "this witness was
    // never satisfiable at all".
    // max_age_ns belongs to kLatestWithin alone; kExact rejects a plan that carries one.
    auto control_plan = selection_plan(vector.selector, PortId{1U}, 1U, 1U);
    if (vector.selector == InputSelectorKind::kLatestWithin) {
      control_plan.max_age_ns = 1000;
    }
    control_plan.missing = TemporalRecordMissingKind::kSkip;
    auto control = detail::RuntimeInputSelectionState::create(control_plan);
    ASSERT_TRUE(control);
    ASSERT_TRUE(control->push(temporal_sample(kRetainedCaptureNs, 1U, 1U)));
    auto held = control->begin(witness, 1U, 1U);
    ASSERT_TRUE(held);
    EXPECT_EQ(held->selection().state, InputState::kSelected);
    ASSERT_EQ(held->selected().size(), 1U);
    EXPECT_EQ(held->selected().front().capture_time.timestamp_ns, kRetainedCaptureNs);
    ASSERT_TRUE(held->finish(true));

    // The same witness, after the reconcile's release admitted a strictly newer record in place of
    // the one it gave up.
    auto plan = selection_plan(vector.selector, PortId{1U}, 1U, 1U);
    if (vector.selector == InputSelectorKind::kLatestWithin) {
      plan.max_age_ns = 1000;
    }
    plan.missing = TemporalRecordMissingKind::kSkip;
    auto state = detail::RuntimeInputSelectionState::create(plan);
    ASSERT_TRUE(state);
    ASSERT_TRUE(state->push(temporal_sample(kRetainedCaptureNs, 1U, 1U)));
    ASSERT_TRUE(state->evict_retained(1U));
    ASSERT_TRUE(state->push(temporal_sample(kSuccessorCaptureNs, 2U, 2U)));

    auto decision = state->begin(witness, 2U, 1U);
    ASSERT_TRUE(decision);
    if (vector.successor_satisfies) {
      EXPECT_EQ(decision->selection().state, InputState::kSelected);
      ASSERT_EQ(decision->selected().size(), 1U);
      EXPECT_EQ(decision->selected().front().capture_time.timestamp_ns, kSuccessorCaptureNs);
    } else {
      // Key assertion: a fresher record is present and cannot stand in. The witness the retained
      // record alone could satisfy now selects nothing, so the release cost an activation outright
      // rather than trading staleness for freshness.
      EXPECT_NE(decision->selection().state, InputState::kSelected);
      EXPECT_TRUE(decision->selected().empty());
    }
    ASSERT_TRUE(decision->finish(true));
  }
}

}  // namespace
}  // namespace holoscan
