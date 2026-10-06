/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "artifact_test_support.hpp"
#include "core/diagnostic_registry.hpp"
#include "physical_plan_test_support.hpp"

// GTest fatal assertions establish optional preconditions that clang-tidy does not model.
// NOLINTBEGIN(bugprone-unchecked-optional-access)
namespace holoscan::detail {
namespace {

using namespace physical_plan_test;

template <typename Enum>
[[nodiscard]] Enum unknown_enum() noexcept;

class DescriptorTensorArtifactSource final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.output(output, "output")
        .tensor_field(TensorFieldSelector<schema::ImageT>{4U, 0U})
        .produces_tensor(TensorOutputSpec{
            .representation = {.memory_kind = MemoryKind::kHost,
                               .dtype = DLDataType{kDLUInt, 8U, 1U},
                               .rank = 1U,
                               .strides = TensorStridePolicy::kContiguousOnly},
            .bounds = tensor_bounds(128U),
            .storage = TensorOutputStorage::kRuntimePool,
        });
  }

  expected<void, Error> compute([[maybe_unused]] ExecutionContext&) override { return {}; }

  Output<schema::ImageT> output;
};

TEST(CompilePhysicalPlanArtifact, NormalizesEmptyLargeAndMalformedJsonWithoutOutOfBoundsAccess) {
  EXPECT_TRUE(artifact_test::normalized_artifact_json({}).empty());

  // Key setup: reserve enough output storage to force the same heap-backed path as real plan
  // artifacts before the normalizer sees its first non-whitespace byte.
  const std::string payload(48'027U, 'x');
  const std::string compact = "{\"payload\":\"" + payload + "\"}";
  const std::string expected = "{\n  \"payload\": \"" + payload + "\"\n}\n";

  // Key assertion: empty input is explicit and a large first-token object is formatted exactly.
  EXPECT_EQ(artifact_test::normalized_artifact_json(compact), expected);

  // Key setup: start malformed input with an unmatched closing delimiter before an indentation
  // point that would expose an unsigned depth wrap.
  constexpr std::string_view kUnmatchedClose{"},"};
  // Key assertion: malformed test input remains bounded instead of requesting SIZE_MAX padding.
  EXPECT_EQ(artifact_test::normalized_artifact_json(kUnmatchedClose), "},\n\n");
}

// REQ: COMPILE-096
TEST(CompilePhysicalPlanArtifact, SupportThreadAffinityIsCanonicalAndFailsClosed) {
  const ThreadPlacement affinity{
      .affinity = CPUAffinity{3U, 7U},
      .probe_failure_policy = ProbeFailurePolicy::kFailAdmission,
  };
  auto baseline = physical_artifact_state();
  baseline.support_thread_affinities = {SupportThreadAffinityPlanRecord{
      .role = SupportThreadRole::kController,
      .requested = affinity,
      .accepted = affinity,
      .disposition = PlanDisposition::kAccepted,
  }};
  auto different_role = baseline;
  different_role.support_thread_affinities.front().role = SupportThreadRole::kChannelPump;

  const auto baseline_finalized = finalize_plan_artifact(baseline);
  const auto different_role_finalized = finalize_plan_artifact(different_role);
  ASSERT_TRUE(baseline_finalized) << baseline_finalized.error().message;
  ASSERT_TRUE(different_role_finalized) << different_role_finalized.error().message;
  EXPECT_NE(baseline.digest, different_role.digest);
  EXPECT_NE(baseline.json.find("\"support_thread_affinities\":[{\"accepted\":"), std::string::npos);

  const auto expect_rejection = [&](auto mutate) {
    auto candidate = baseline;
    mutate(candidate);
    candidate.json.clear();
    const auto finalized = finalize_plan_artifact(candidate);
    ASSERT_FALSE(finalized);
    EXPECT_EQ(finalized.error().code, ErrorCode::kInvalidArgument);
    EXPECT_EQ(finalized.error().message, "SUPPORT_THREAD_AFFINITY_PLAN_INVALID");
    EXPECT_TRUE(candidate.json.empty());
  };
  expect_rejection([](auto& state) {
    state.support_thread_affinities.push_back(state.support_thread_affinities.front());
  });
  expect_rejection([](auto& state) {
    state.support_thread_affinities.front().role = unknown_enum<SupportThreadRole>();
  });
  expect_rejection([](auto& state) {
    state.support_thread_affinities.front().requested.affinity = CPUAffinity{};
  });
  expect_rejection([](auto& state) {
    state.support_thread_affinities.front().accepted->affinity = CPUAffinity{9U};
  });
}

template <typename Enum>
[[nodiscard]] Enum unknown_enum() noexcept {
  using Underlying = std::underlying_type_t<Enum>;
  // Deliberately construct a defensive-deserialization case that typed authoring cannot express.
  // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange)
  return static_cast<Enum>(std::numeric_limits<Underlying>::max());
}

[[nodiscard]] OperatorCompletionPlanRecord completion_rule(const OperatorInstanceId owner,
                                                           const PlanGenerationId plan_generation,
                                                           const std::uint64_t limit) {
  return OperatorCompletionPlanRecord{
      .owner = {.id = owner, .plan_generation = plan_generation, .run_generation = 0U},
      .kind = OperatorCompletionRuleKind::kComputeCalls,
      .limit = limit,
  };
}

[[nodiscard]] TemporalTriggerPlanRecord completion_trigger(
    const OperatorInstanceId owner, const PlanGenerationId plan_generation,
    const TemporalRecordTriggerKind kind = TemporalRecordTriggerKind::kNotified) {
  return TemporalTriggerPlanRecord{
      .owner = {.id = owner, .plan_generation = plan_generation, .run_generation = 0U},
      .kind = kind,
      .origin = TemporalRecordOrigin::kAuthored,
      .readiness_source = make_source_id(owner),
      .trigger_clock = kind == TemporalRecordTriggerKind::kClock ? ClockId{1U} : ClockId{},
      .reference_clock = ClockId{1U},
      .period_ns = kind == TemporalRecordTriggerKind::kClock ? 1000 : 0,
      .bounded_state_capacity = 1U,
      .witness_identity = "holoscan.temporal.witness.v1/completion-fixture",
      .replay_identity = "holoscan.temporal.replay.v1/decision",
  };
}

[[nodiscard]] ReadinessPlanRecord completion_readiness(
    const OperatorInstanceId owner, const PlanGenerationId plan_generation,
    const ReadinessSourceKind kind = ReadinessSourceKind::kNotification) {
  return ReadinessPlanRecord{
      .source_id = make_source_id(owner),
      .plan_generation = plan_generation,
      .owner = VertexId{owner},
      .owner_path = "fixture/completion-" + std::to_string(owner),
      .kind = kind,
      .name = "completion-source",
      .resolved_clock = ClockId{1U},
      .requested_capacity = 2U,
      .requested_sender_reference_capacity = 1U,
      .accepted_capacity = 2U,
      .accepted_sender_reference_capacity = 1U,
      .referenced = true,
      .materialized = true,
      .initial_source_generation = 1U,
  };
}

void add_valid_completion_rules(ExecutionPlanState& state, const std::uint64_t first_limit,
                                const std::uint64_t second_limit) {
  state.operator_completion_rules = {
      completion_rule(1U, state.plan_generation, first_limit),
      completion_rule(2U, state.plan_generation, second_limit),
  };
  state.temporal_triggers = {
      completion_trigger(1U, state.plan_generation),
      completion_trigger(2U, state.plan_generation),
  };
  state.readiness = {
      completion_readiness(1U, state.plan_generation),
      completion_readiness(2U, state.plan_generation),
  };
}

void add_valid_rate_records(ExecutionPlanState& state) {
  state.port_rates.clear();
  state.port_rates.reserve(state.ports.size());
  for (const auto& port : state.ports) {
    const bool is_fixed_source = port.id == PortId(1U);
    state.port_rates.push_back(PortRatePlanRecord{
        .port = port.id,
        .kind = is_fixed_source ? PortRateKind::kFixedNominal : PortRateKind::kUnspecified,
        .declared_nominal_rate = is_fixed_source ? std::make_optional(Rate::hz(60U)) : std::nullopt,
        .disposition = PlanDisposition::kAccepted,
    });
  }
  state.accumulation_horizons.clear();
  if (!state.payload_edges.empty()) {
    const auto target =
        std::ranges::find(state.ports, state.payload_edges.front().target, &PortPlanRecord::id);
    ASSERT_NE(target, state.ports.end());
    const auto trigger_owner = static_cast<OperatorInstanceId>(target->owner.value);
    state.temporal_triggers = {TemporalTriggerPlanRecord{
        .owner = {.id = trigger_owner,
                  .plan_generation = state.plan_generation,
                  .run_generation = 0U},
        .kind = TemporalRecordTriggerKind::kClock,
        .origin = TemporalRecordOrigin::kAuthored,
        .trigger_clock = ClockId{1U},
        .reference_clock = ClockId{1U},
        .period_ns = 1'000'000'000,
        .bounded_state_capacity = 1U,
        .witness_identity = "holoscan.temporal.witness.v1/rate-fixture",
        .replay_identity = "holoscan.temporal.replay.v1/decision",
    }};
    state.accumulation_horizons.push_back(AccumulationHorizonPlanRecord{
        .id = HorizonRecordId{1U},
        .edge = state.payload_edges.front().edge,
        .horizon = ExactDuration{.numerator_ns = 1'000'000'000U, .denominator = 1U},
        .origin = AccumulationHorizonOrigin::kClockPeriod,
        .source_record = static_cast<std::uint32_t>(trigger_owner)});
  }
  state.hard_floor_contributors.clear();
  state.queue_sizing_recipes.clear();
  for (std::size_t index = 0U; index < state.payload_edges.size(); ++index) {
    const auto& edge = state.payload_edges[index];
    const std::uint32_t id = static_cast<std::uint32_t>(index + 1U);
    state.hard_floor_contributors.push_back(
        HardFloorContributorPlanRecord{.id = HardFloorContributorId{id},
                                       .edge = edge.edge,
                                       .origin = HardFloorOrigin::kMinimumQueue,
                                       .required_slots = 1U,
                                       .scenario = HardFloorScenarioId{id},
                                       .alternative_group = HardFloorAlternativeGroupId{id}});
    state.queue_sizing_recipes.push_back(QueueSizingRecipePlanRecord{
        .id = QueueSizingRecipeId{id},
        .edge = edge.edge,
        .origin = index == 0U ? QueueDepthOrigin::kRateAndContractDerived
                              : QueueDepthOrigin::kAddFlowOption,
        .explicit_depth =
            index == 0U ? std::nullopt : std::optional<std::uint32_t>{edge.queue_depth},
        .headroom_ratio = RateRatio{2U, 1U},
        .reuse_policy = ProviderReusePolicy::kCompatibleExactOrLarger,
        .disposition = PlanDisposition::kAccepted});
  }
}

void use_valid_window_rate_horizon(ExecutionPlanState& state) {
  ASSERT_EQ(state.temporal_triggers.size(), 1U);
  ASSERT_EQ(state.accumulation_horizons.size(), 1U);
  ASSERT_FALSE(state.payload_edges.empty());
  auto& trigger = state.temporal_triggers.front();
  trigger.kind = TemporalRecordTriggerKind::kWindow;
  trigger.trigger_input = state.payload_edges.front().target;
  trigger.period_ns = 0;
  auto& horizon = state.accumulation_horizons.front();
  ASSERT_TRUE(horizon.horizon.has_value());
  horizon.origin = AccumulationHorizonOrigin::kTemporalWindowBound;
  state.window_triggers = {WindowTriggerPlanRecord{
      .owner = trigger.owner,
      .anchor = trigger.trigger_input,
      .wait_clock = trigger.trigger_clock,
      .capability_evidence_id = "holoscan.clock.realtime.v1",
      .max_wait_ns = static_cast<std::int64_t>(horizon.horizon->numerator_ns),
      .max_pending_windows = 1U,
      .presence_guaranteed_inputs = {trigger.trigger_input},
      .primed_witness_identity = "holoscan.temporal.window-primed.v1",
      .open_witness_identity = "holoscan.temporal.window-open.v1",
      .close_witness_identity = "holoscan.temporal.window-close.v1",
      .timeout_witness_identity = "holoscan.temporal.window-timeout.v1",
      .pending_limit_witness_identity = "holoscan.temporal.window-pending-limit.v1",
      .end_of_stream_witness_identity = "holoscan.temporal.window-end-of-stream.v1",
      .timestamp_regression_witness_identity = "holoscan.temporal.window-timestamp-regression.v1",
      .cancelled_witness_identity = "holoscan.temporal.window-cancelled.v1",
      .producer_epoch_identity_schema = "holoscan.producer-epoch.v1",
  }};
}

void use_valid_predicate_clock_rate_horizon(ExecutionPlanState& state) {
  ASSERT_EQ(state.temporal_triggers.size(), 1U);
  ASSERT_EQ(state.accumulation_horizons.size(), 1U);
  auto& trigger = state.temporal_triggers.front();
  trigger.kind = TemporalRecordTriggerKind::kPredicate;
  auto& horizon = state.accumulation_horizons.front();
  horizon.origin = AccumulationHorizonOrigin::kClockPeriod;
  state.predicate_triggers = {PredicateTriggerPlanRecord{
      .id = "predicate/rate-fixture",
      .owner = trigger.owner,
      .graph_generation = state.plan_generation,
      .identity = "rate-fixture",
      .source = PredicateEvaluationSourceKind::kClock,
      .clock = trigger.trigger_clock,
      .period_ns = trigger.period_ns,
  }};
}

static_assert(std::is_standard_layout_v<OperatorCompletionPlanRecord>);
static_assert(std::is_trivially_copyable_v<OperatorCompletionPlanRecord>);

// The pointer-free value record participates in plan equality by all three semantic fields.
// REQ: CONTRACT-020
TEST(CompilePhysicalPlanArtifact, OperatorCompletionRecordEqualityIncludesOwnerKindAndLimit) {
  const OperatorCompletionPlanRecord baseline = completion_rule(3U, PlanGenerationId{7U}, 11U);
  EXPECT_EQ(baseline, completion_rule(3U, PlanGenerationId{7U}, 11U));

  auto changed_owner = baseline;
  changed_owner.owner.id = OperatorInstanceId{4U};
  EXPECT_NE(baseline, changed_owner);

  auto changed_kind = baseline;
  changed_kind.kind = unknown_enum<OperatorCompletionRuleKind>();
  EXPECT_NE(baseline, changed_kind);

  auto changed_limit = baseline;
  changed_limit.limit = 12U;
  EXPECT_NE(baseline, changed_limit);
}

// Completion policy is a separate pointer-free physical-plan record family. It participates in
// canonical JSON and plan identity without extending the temporal-record v1 schema.
// REQ: CONTRACT-020
TEST(CompilePhysicalPlanArtifact, SerializesOperatorCompletionRulesCanonically) {
  auto rule_free = physical_artifact_state();
  const auto rule_free_finalized = finalize_plan_artifact(rule_free);
  ASSERT_TRUE(rule_free_finalized) << rule_free_finalized.error().message;
  EXPECT_EQ(rule_free.json.find("\"operator_completion_rules\""), std::string::npos)
      << rule_free.json;

  // Clock sources are synthesized by temporal lowering and therefore deliberately own no
  // ReadinessPlanRecord. The graph/factory-free fixture still resolves the logical owner through
  // its retained physical-port projection.
  auto clock = physical_artifact_state();
  clock.operator_completion_rules = {completion_rule(1U, clock.plan_generation, 3U)};
  clock.temporal_triggers = {
      completion_trigger(1U, clock.plan_generation, TemporalRecordTriggerKind::kClock)};
  const auto clock_finalized = finalize_plan_artifact(clock);
  ASSERT_TRUE(clock_finalized) << clock_finalized.error().message;
  EXPECT_TRUE(clock.readiness.empty());

  auto maximum = physical_artifact_state();
  add_valid_completion_rules(maximum, std::numeric_limits<std::uint64_t>::max(), 0U);
  auto changed = maximum;
  changed.operator_completion_rules.front().limit--;

  const auto maximum_finalized = finalize_plan_artifact(maximum);
  const auto changed_finalized = finalize_plan_artifact(changed);
  ASSERT_TRUE(maximum_finalized) << maximum_finalized.error().message;
  ASSERT_TRUE(changed_finalized) << changed_finalized.error().message;

  constexpr std::string_view kCanonicalGolden =
      "\"operator_completion_rules\":[{\"kind\":0,\"limit\":18446744073709551615,"
      "\"owner\":{\"id\":1,\"plan_generation\":71,\"run_generation\":0}},{\"kind\":0,"
      "\"limit\":0,\"owner\":{\"id\":2,\"plan_generation\":71,\"run_generation\":0}}]";
  EXPECT_NE(maximum.json.find(kCanonicalGolden), std::string::npos) << maximum.json;
  EXPECT_EQ(maximum.json.find("\"limit\":-1"), std::string::npos) << maximum.json;
  EXPECT_NE(maximum.digest, changed.digest);
  EXPECT_NE(maximum.json, changed.json);
}

// Malformed completion records are rejected before serialization: the owner must be a current
// plan-bound logical zero-input producer with output, kind must be recognized, records must be
// unique and canonical, and each rule must have exactly one matching authored source trigger.
// REQ: CONTRACT-020
TEST(CompilePhysicalPlanArtifact, RejectsMalformedOperatorCompletionRulesBeforeSerialization) {
  const auto expect_rejection = [](auto mutate) {
    auto candidate = physical_artifact_state();
    add_valid_completion_rules(candidate, 3U, 5U);
    mutate(candidate);

    const auto finalized = finalize_plan_artifact(candidate);
    ASSERT_FALSE(finalized);
    EXPECT_EQ(finalized.error().code, ErrorCode::kInvalidArgument);
    EXPECT_EQ(finalized.error().message, "OPERATOR_COMPLETION_PLAN_INVALID");
    EXPECT_TRUE(candidate.json.empty());
  };

  expect_rejection([](auto& state) {
    state.operator_completion_rules.front().kind = unknown_enum<OperatorCompletionRuleKind>();
  });
  expect_rejection(
      [](auto& state) { state.operator_completion_rules.front().owner = OperatorRef{}; });
  expect_rejection(
      [](auto& state) { state.operator_completion_rules.front().owner.run_generation = 1U; });
  expect_rejection(
      [](auto& state) { --state.operator_completion_rules.front().owner.plan_generation; });
  expect_rejection([](auto& state) { state.operator_completion_rules.front().owner.id = 999U; });
  expect_rejection([](auto& state) {
    state.operator_completion_rules = {state.operator_completion_rules.front(),
                                       state.operator_completion_rules.front()};
  });
  expect_rejection([](auto& state) {
    std::swap(state.operator_completion_rules[0], state.operator_completion_rules[1]);
  });
  expect_rejection([](auto& state) { state.temporal_triggers.pop_back(); });
  expect_rejection(
      [](auto& state) { state.temporal_triggers.push_back(state.temporal_triggers.front()); });
  expect_rejection([](auto& state) {
    state.temporal_triggers.front().kind = TemporalRecordTriggerKind::kAnyInput;
  });
  expect_rejection([](auto& state) {
    state.temporal_triggers.front().origin = TemporalRecordOrigin::kCoreSynthesized;
  });
  expect_rejection(
      [](auto& state) { state.temporal_triggers.front().readiness_source = SourceId{}; });
  expect_rejection([](auto& state) { --state.temporal_triggers.front().owner.plan_generation; });
  expect_rejection([](auto& state) {
    auto input = state.ports[2];
    input.id = PortId{8U};
    input.owner = VertexId{1U};
    state.ports.push_back(std::move(input));
    state.port_rates.push_back(PortRatePlanRecord{
        .port = PortId{8U},
        .kind = PortRateKind::kUnspecified,
        .disposition = PlanDisposition::kAccepted,
    });
  });
  expect_rejection([](auto& state) {
    state.operator_completion_rules = {completion_rule(3U, state.plan_generation, 3U)};
    state.temporal_triggers = {completion_trigger(3U, state.plan_generation)};
  });
}

// Physical completion triggers must resolve exactly one plan-current readiness record. Source
// identity alone is insufficient: the record must belong to the resolved Operator vertex, match
// the trigger family, and be both referenced and materialized.
// REQ: CONTRACT-020
TEST(CompilePhysicalPlanArtifact, RejectsMalformedCompletionReadinessSourceBeforeSerialization) {
  const auto expect_rejection = [](auto mutate) {
    auto candidate = physical_artifact_state();
    add_valid_completion_rules(candidate, 3U, 5U);
    mutate(candidate);

    const auto finalized = finalize_plan_artifact(candidate);
    ASSERT_FALSE(finalized);
    EXPECT_EQ(finalized.error().code, ErrorCode::kInvalidArgument);
    EXPECT_EQ(finalized.error().message, "OPERATOR_COMPLETION_PLAN_INVALID");
    EXPECT_TRUE(candidate.json.empty());
  };

  expect_rejection([](auto& state) { state.readiness.pop_back(); });
  expect_rejection([](auto& state) { state.readiness.push_back(state.readiness.front()); });
  expect_rejection([](auto& state) { state.readiness.front().source_id = make_source_id(999U); });
  expect_rejection([](auto& state) { state.readiness.front().owner = VertexId{2U}; });
  expect_rejection(
      [](auto& state) { state.readiness.front().kind = ReadinessSourceKind::kReadable; });
  expect_rejection([](auto& state) { --state.readiness.front().plan_generation; });
  expect_rejection([](auto& state) { state.readiness.front().referenced = false; });
  expect_rejection([](auto& state) { state.readiness.front().materialized = false; });
}

// The Graph snapshot must deep-copy exact construction evidence rather than infer it from a POD
// layout or generated schema identity. Constructor-only POD publication remains admitted through
// emit(T), but does not fabricate an allocate() capability or a generic-loan pool.
// REQ: COMPILE-009
// REQ: COMPILE-095
// REQ: COMPILE-047
// REQ: CHAN-074
// REQ: CHAN-075
// REQ: CHAN-120
TEST(CompilePhysicalPlanArtifact, FreezesTypedGenericLoanEvidenceWithoutFunctionPointers) {
  Graph graph{"generic-loan-evidence"};
  (void)graph.op<GenericLoanEvidenceOperator>("source");

  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();

  const auto find_port_by_name = [&](std::string_view name) {
    return std::find_if(plan.ports().begin(), plan.ports().end(), [&](const auto& port) {
      return port.local_name == name;
    });
  };
  const auto pod = find_port_by_name("pod");
  const auto schema_port = find_port_by_name("schema");
  const auto constructor_only = find_port_by_name("constructor-only");
  ASSERT_NE(pod, plan.ports().end());
  ASSERT_NE(schema_port, plan.ports().end());
  ASSERT_NE(constructor_only, plan.ports().end());
  EXPECT_TRUE(pod->generic_loan_supported);
  EXPECT_EQ(pod->generic_loan_slot_size_bytes, sizeof(CompileLoanPod));
  EXPECT_TRUE(schema_port->generic_loan_supported);
  EXPECT_EQ(schema_port->generic_loan_slot_size_bytes, sizeof(schema::StringT));
  EXPECT_FALSE(constructor_only->generic_loan_supported);
  EXPECT_EQ(constructor_only->generic_loan_slot_size_bytes, 0U);

  const auto find_loan = [&](PortId output) {
    return std::find_if(plan.generic_output_loans().begin(),
                        plan.generic_output_loans().end(),
                        [&](const auto& loan) { return loan.output == output; });
  };
  const auto pod_loan = find_loan(pod->id);
  const auto schema_loan = find_loan(schema_port->id);
  ASSERT_NE(pod_loan, plan.generic_output_loans().end());
  ASSERT_NE(schema_loan, plan.generic_output_loans().end());
  EXPECT_EQ(pod_loan->trait_evidence, GenericLoanTraitEvidence::kNothrowDefaultConstructiblePOD);
  EXPECT_EQ(pod_loan->slot_size_bytes, sizeof(CompileLoanPod));
  EXPECT_EQ(schema_loan->trait_evidence,
            GenericLoanTraitEvidence::kNothrowDefaultConstructibleSchema);
  EXPECT_EQ(schema_loan->slot_size_bytes, sizeof(schema::StringT));
  EXPECT_EQ(find_loan(constructor_only->id), plan.generic_output_loans().end());
  EXPECT_NE(
      std::find_if(plan.publication_outcomes().begin(),
                   plan.publication_outcomes().end(),
                   [&](const auto& outcome) { return outcome.output == constructor_only->id; }),
      plan.publication_outcomes().end());
  EXPECT_NE(plan.json().find("\"generic_loan_supported\":true"), std::string::npos);
  EXPECT_NE(plan.json().find("\"trait_evidence\":1"), std::string::npos);
  EXPECT_NE(plan.json().find("\"trait_evidence\":2"), std::string::npos);
}

// REQ: COMPILE-002
// REQ: COMPILE-009
// REQ: COMPILE-095
// REQ: COMPILE-040
// REQ: COMPILE-048
// REQ: COMPILE-064
TEST(CompilePhysicalPlanArtifact, SerializesEveryPublicPhysicalFamilyAndSemanticFieldCanonically) {
  auto first = physical_artifact_state();
  auto second = physical_artifact_state();
  ASSERT_FALSE(first.ports.empty());
  ASSERT_FALSE(first.interface_port_bindings.empty());
  ASSERT_FALSE(first.payload_edges.empty());
  ASSERT_FALSE(first.channel_locality.empty());
  ASSERT_FALSE(first.channel_services.empty());
  ASSERT_FALSE(first.channel_publishers.empty());
  ASSERT_FALSE(first.channel_subscribers.empty());
  ASSERT_FALSE(first.process_local_channel_services.empty());
  ASSERT_FALSE(first.process_local_channel_publishers.empty());
  ASSERT_FALSE(first.process_local_channel_subscribers.empty());
  ASSERT_EQ(first.process_local_channel_services.size(),
            second.process_local_channel_services.size());
  ASSERT_EQ(first.process_local_channel_publishers.size(),
            second.process_local_channel_publishers.size());
  ASSERT_EQ(first.process_local_channel_subscribers.size(),
            second.process_local_channel_subscribers.size());
  for (std::size_t index = 0U; index < first.process_local_channel_services.size(); ++index) {
    const auto& left = first.process_local_channel_services[index];
    const auto& right = second.process_local_channel_services[index];
    EXPECT_EQ(left.source, right.source);
    EXPECT_EQ(left.service_id, right.service_id);
    EXPECT_EQ(left.payload_identity.kind, right.payload_identity.kind);
    EXPECT_EQ(left.payload_identity.payload_kind, right.payload_identity.payload_kind);
    ASSERT_EQ(left.payload_identity.scalar.has_value(), right.payload_identity.scalar.has_value());
    if (left.payload_identity.scalar) {
      EXPECT_EQ(left.payload_identity.scalar->kind, right.payload_identity.scalar->kind);
    }
    EXPECT_EQ(left.payload_size_bytes, right.payload_size_bytes);
    EXPECT_EQ(left.tensor_payload, right.tensor_payload);
    EXPECT_EQ(left.tensor_max_byte_span, right.tensor_max_byte_span);
    EXPECT_EQ(left.tensor_pool_id, right.tensor_pool_id);
    EXPECT_EQ(left.tensor_memory_kind, right.tensor_memory_kind);
    EXPECT_EQ(left.tensor_device, right.tensor_device);
    EXPECT_EQ(left.delivery_kind, right.delivery_kind);
    EXPECT_EQ(left.queue_policy, right.queue_policy);
    EXPECT_EQ(left.max_subscribers, right.max_subscribers);
    EXPECT_EQ(left.max_subscriber_queue_depth, right.max_subscriber_queue_depth);
    EXPECT_EQ(left.max_subscriber_borrowed_samples, right.max_subscriber_borrowed_samples);
    EXPECT_EQ(left.publisher_history_depth, right.publisher_history_depth);
    EXPECT_EQ(left.event_wake, right.event_wake);
    EXPECT_EQ(left.event_wake_reason, right.event_wake_reason);
  }
  for (std::size_t index = 0U; index < first.process_local_channel_publishers.size(); ++index) {
    const auto& left = first.process_local_channel_publishers[index];
    const auto& right = second.process_local_channel_publishers[index];
    EXPECT_EQ(left.source, right.source);
    EXPECT_EQ(left.service_id, right.service_id);
    EXPECT_EQ(left.max_loaned_samples, right.max_loaned_samples);
    EXPECT_EQ(left.replacement_attribution_required, right.replacement_attribution_required);
  }
  for (std::size_t index = 0U; index < first.process_local_channel_subscribers.size(); ++index) {
    const auto& left = first.process_local_channel_subscribers[index];
    const auto& right = second.process_local_channel_subscribers[index];
    EXPECT_EQ(left.edge, right.edge);
    EXPECT_EQ(left.source, right.source);
    EXPECT_EQ(left.target, right.target);
    EXPECT_EQ(left.service_id, right.service_id);
    EXPECT_EQ(left.requested_queue_depth, right.requested_queue_depth);
    EXPECT_EQ(left.max_borrowed_samples, right.max_borrowed_samples);
    EXPECT_EQ(left.cache_capacity, right.cache_capacity);
    EXPECT_EQ(left.readiness_capacity, right.readiness_capacity);
    EXPECT_EQ(left.event_wake, right.event_wake);
  }
  ASSERT_FALSE(first.generic_output_loans.empty());
  ASSERT_FALSE(first.tensor_allocations.empty());
  ASSERT_FALSE(first.tensor_interface_pools.empty());
  ASSERT_FALSE(first.tensor_device_resources.empty());
  ASSERT_FALSE(first.tensor_inline_records.empty());
  ASSERT_FALSE(first.imported_generation_capacities.empty());
  ASSERT_FALSE(first.tensor_transfer_plans.empty());
  ASSERT_FALSE(first.pool_capacity_contributors.empty());
  ASSERT_FALSE(first.pool_capacity_dependencies.empty());
  ASSERT_FALSE(first.provider_allocation_plans.empty());
  ASSERT_FALSE(first.publication_outcomes.empty());
  ASSERT_FALSE(first.aligned_triggers.empty());
  ASSERT_FALSE(first.clock_triggers.empty());
  ASSERT_FALSE(first.deadlines.empty());
  ASSERT_FALSE(first.readable_triggers.empty());
  ASSERT_FALSE(first.missing_aggregates.empty());
  EXPECT_EQ(first.generic_output_loans.front().plan_generation, 71U);
  EXPECT_EQ(first.generic_output_loans.front().run_generation, 0U);
  EXPECT_EQ(first.tensor_allocations.front().plan_generation, 71U);
  EXPECT_EQ(first.tensor_allocations.front().run_generation, 0U);
  EXPECT_EQ(first.tensor_interface_pools.front().plan_generation, 71U);
  EXPECT_EQ(first.tensor_interface_pools.front().run_generation, 0U);
  EXPECT_EQ(first.tensor_device_resources.front().plan_generation, 71U);
  EXPECT_EQ(first.tensor_device_resources.front().run_generation, 0U);

  const auto first_finalized = finalize_plan_artifact(first);
  const auto second_finalized = finalize_plan_artifact(second);
  ASSERT_TRUE(first_finalized) << first_finalized.error().message;
  ASSERT_TRUE(second_finalized) << second_finalized.error().message;
  EXPECT_EQ(first.digest, second.digest);
  EXPECT_EQ(first.json, second.json);
  EXPECT_NE(first.json.find("\"role\":\"external_boundary\",\"representation\":"),
            std::string::npos);
  EXPECT_NE(first.json.find("\"memory_conversion\":1"), std::string::npos);
  EXPECT_NE(first.json.find("\"origins\":{\"memory_kind\":\"absent\""), std::string::npos);

  constexpr std::array required_tokens{
      "\"ports\"",
      "\"interface_port_bindings\"",
      "\"payload_edges\"",
      "\"channel_locality\"",
      "\"channel_services\"",
      "\"channel_publishers\"",
      "\"channel_subscribers\"",
      "\"generic_output_loans\"",
      "\"tensor_allocations\"",
      "\"tensor_device_resources\"",
      "\"tensor_inline_records\"",
      "\"tensor_interface_pools\"",
      "\"imported_generation_capacities\"",
      "\"tensor_transfer_plans\"",
      "\"pool_capacity_contributors\"",
      "\"pool_capacity_dependencies\"",
      "\"provider_allocation_plans\"",
      "\"publication_outcomes\"",
      "\"aligned_triggers\"",
      "\"clock_triggers\"",
      "\"deadlines\"",
      "\"readable_triggers\"",
      "\"missing_aggregates\"",
      "\"local_name\"",
      "\"owner_kind\"",
      "\"late_joiner_history_depth\"",
      "\"tensor_layout\"",
      "\"internal_port\"",
      "\"public_port\"",
      "\"max_inflight_samples\"",
      "\"producer_max_emits_per_compute\"",
      "\"fanout\"",
      "\"cache_capacity\"",
      "\"requires_runtime_attach\"",
      "\"max_subscriber_queue_depth\"",
      "\"max_subscriber_borrowed_samples\"",
      "\"publisher_history_depth\"",
      "\"tensor_emit_policy\"",
      "\"max_loaned_samples\"",
      "\"replacement_attribution_required\"",
      "\"requested_queue_depth\"",
      "\"max_borrowed_samples\"",
      "\"trait_evidence\"",
      "\"max_live_loans\"",
      "\"materialize_before\"",
      "\"pre_submit_preserves_same_activation_loan\"",
      "\"plan_generation\"",
      "\"run_generation\"",
      "\"allocator_root_id\"",
      "\"allocator_process_group\"",
      "\"aggregate_byte_capacity\"",
      "\"max_outstanding_loans\"",
      "\"control_capacity\"",
      "\"readiness_capacity\"",
      "\"release_capacity\"",
      "\"retirement_capacity\"",
      "\"quarantine_capacity\"",
      "\"exporter_subpool_id\"",
      "\"logical\"",
      "\"publisher_loan_capacity\"",
      "\"stream_id\"",
      "\"fused_linear_edges\"",
      "\"cuda_compute_streams\"",
      "\"cuda_green_context_assignments\"",
      "\"cuda_green_context_pools\"",
      "\"cuda_green_context_requests\"",
      "\"cuda_green_context_resources\"",
      "\"pool_count\"",
      "\"gpu_resident_backend_selected\"",
      "\"fixed_record_bucket_id\"",
      "\"fixed_record_bucket_bytes\"",
      "\"fixed_metadata_overhead_bytes\"",
      "\"provider_max_inline_logical_bytes\"",
      "\"profile_max_inline_logical_bytes\"",
      "\"effective_max_inline_logical_bytes\"",
      "\"accepted_max_inline_logical_bytes\"",
      "\"record_format_id\"",
      "\"record_format_version\"",
      "\"inline_discriminator\"",
      "\"opaque_descriptor_discriminator\"",
      "\"inline_arm_schema_id\"",
      "\"opaque_descriptor_arm_schema_id\"",
      "\"endpoint_max_generations\"",
      "\"service_max_generations\"",
      "\"runtime_max_generations\"",
      "\"no_wrap_aba_generation\"",
      "\"attach_max_wait_ns\"",
      "\"peer_requirement\"",
      "\"candidate_reservation_capacity\"",
      "\"release_record_capacity\"",
      "\"quarantine_record_capacity\"",
      "\"source_class\"",
      "\"realization\"",
      "\"min_logical_bytes\"",
      "\"max_logical_bytes\"",
      "\"delivered_read_only\"",
      "\"contributor_id\"",
      "\"maximum_distinct_slots\"",
      "\"pre_submit_error\"",
      "\"post_submit_error\"",
      "\"post_submit_partial_error\"",
      "\"post_submit_consumes_authority\"",
      "\"producer_loan\"",
      "\"retirement_service\"",
  };
  for (const std::string_view token : required_tokens) {
    EXPECT_NE(first.json.find(token), std::string::npos) << token;
  }
  EXPECT_NE(first.json.find("\"post_submit_error\":22"), std::string::npos);
  EXPECT_NE(first.json.find("\"post_submit_partial_error\":36"), std::string::npos);
  for (const std::string_view private_token : {"process_local_channel_services",
                                               "process_local_channel_publishers",
                                               "process_local_channel_subscribers"}) {
    EXPECT_EQ(first.json.find(private_token), std::string::npos) << private_token;
  }
}

// REQ: COMPILE-009
// REQ: COMPILE-095
// REQ: COMPILE-040
TEST(CompilePhysicalPlanArtifact, PreservesDescriptorTensorServiceFieldProvenance) {
  constexpr TensorFieldSelector<schema::ImageT> kImageData{4U, 0U};
  DescriptorTensorTopicContract<schema::ImageT> tensors;
  tensors.field(kImageData,
                TensorBoundaryContract{
                    .representation = {.memory_kind = MemoryKind::kHost,
                                       .dtype = DLDataType{kDLUInt, 8U, 1U},
                                       .rank = 1U,
                                       .strides = TensorStridePolicy::kContiguousOnly},
                    .bounds = tensor_bounds(128U),
                    .transfer = TensorTransferPolicy{.foreign_emit_policy =
                                                         InterfaceTensorEmitPolicy::kCopyIfNeeded},
                });

  // Key setup: compile a descriptor-backed external topic whose service record carries authored
  // field origins into the complete diagnostic artifact.
  Graph graph{"descriptor-service-field-provenance"};
  test::add_compile_test_default_clock(graph);
  auto exported = graph.export_topic<schema::ImageT>(
      "holomq://descriptor-service-field-provenance", TopicOptions{}, std::move(tensors));
  const auto source = graph.op<DescriptorTensorArtifactSource>("source");
  graph.add_flow(source->output, exported.input());
  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_FALSE(plan.channel_services().empty());
  ASSERT_FALSE(plan.channel_services().front().tensor_fields.empty());

  // Key assertion: complete-artifact channel service fields retain their contract origins.
  const auto services_begin = plan.json().find("\"channel_services\":");
  const auto services_end = plan.json().find(",\"channel_subscribers\":", services_begin);
  ASSERT_NE(services_begin, std::string::npos);
  ASSERT_NE(services_end, std::string::npos);
  const std::string_view services_json =
      plan.json().substr(services_begin, services_end - services_begin);
  EXPECT_NE(services_json.find("\"origins\":{\"memory_kind\":"), std::string::npos)
      << services_json;

  const auto retained = ExecutionPlanAccess::retain(plan);
  ASSERT_TRUE(retained);
  auto provenance_variant = std::make_shared<ExecutionPlanState>(*retained);
  provenance_variant->channel_services.front().tensor_fields.front().tensor.origins.memory_kind =
      TensorContractValueOrigin::kDefaultedBoundary;
  const auto finalized = finalize_plan_artifact(*provenance_variant);
  ASSERT_TRUE(finalized) << finalized.error().message;

  // Key assertion: changing only diagnostic field provenance changes complete JSON but cannot
  // perturb the semantic execution-plan identity.
  EXPECT_EQ(provenance_variant->digest, plan.digest());
  EXPECT_NE(provenance_variant->json, plan.json());
  EXPECT_NE(provenance_variant->json.find("\"memory_kind\":\"defaulted_boundary\""),
            std::string::npos);
}

// Task041's immutable declaration, canonical horizon/trigger projection, hard-floor, and recipe
// records are all semantic plan authorities. Each valid authority change must perturb the
// private-v7 artifact identity.
// REQ: COMPILE-069
TEST(CompilePhysicalPlanArtifact, RateRecipeFamiliesPerturbSemanticIdentity) {
  auto baseline = physical_artifact_state();
  add_valid_rate_records(baseline);
  const auto baseline_finalized = finalize_plan_artifact(baseline);
  ASSERT_TRUE(baseline_finalized) << baseline_finalized.error().message;

  const auto expect_change = [&](auto mutate) {
    auto candidate = physical_artifact_state();
    add_valid_rate_records(candidate);
    mutate(candidate);
    const auto finalized = finalize_plan_artifact(candidate);
    ASSERT_TRUE(finalized) << finalized.error().message;
    EXPECT_NE(candidate.digest, baseline.digest);
    EXPECT_NE(candidate.json, baseline.json);
  };

  expect_change(
      [](auto& state) { state.port_rates.front().declared_nominal_rate = Rate::hz(61U); });
  expect_change([](auto& state) {
    ASSERT_TRUE(state.accumulation_horizons.front().horizon.has_value());
    state.accumulation_horizons.front().horizon->numerator_ns = 999'999'999U;
    state.temporal_triggers.front().period_ns = 999'999'999;
  });
  expect_change([](auto& state) { ++state.hard_floor_contributors.front().required_slots; });
  expect_change(
      [](auto& state) { state.queue_sizing_recipes.front().headroom_ratio = RateRatio{3U, 2U}; });

  EXPECT_NE(baseline.json.find("\"declared_expected_rate\":null"), std::string::npos);
  EXPECT_NE(baseline.json.find("\"follows_input\":null"), std::string::npos);
  EXPECT_NE(baseline.json.find("\"inverse_rate_port\":null"), std::string::npos);
  EXPECT_NE(baseline.json.find("\"explicit_depth\":null"), std::string::npos);
  EXPECT_NE(baseline.json.find("\"latest_binding_stage\":null"), std::string::npos);
}

// Private-v4 rate records fail closed before malformed values can enter the digest preimage.
// REQ: COMPILE-069
TEST(CompilePhysicalPlanArtifact, RejectsMalformedRateRecipeRecordsBeforeSerialization) {
  const auto expect_rejection = [](auto mutate) {
    auto candidate = physical_artifact_state();
    add_valid_rate_records(candidate);
    mutate(candidate);
    const auto finalized = finalize_plan_artifact(candidate);
    ASSERT_FALSE(finalized);
    EXPECT_EQ(finalized.error().code, ErrorCode::kInvalidArgument);
    EXPECT_EQ(finalized.error().message, "RATE_PLAN_RECORDS_INVALID");
    EXPECT_TRUE(candidate.json.empty());
  };

  expect_rejection([](auto& state) { state.port_rates.front().port = PortId{}; });
  expect_rejection([](auto& state) { state.port_rates.pop_back(); });
  expect_rejection([](auto& state) { state.port_rates.front().declared_nominal_rate.reset(); });
  expect_rejection([](auto& state) {
    state.port_rates[2U] = PortRatePlanRecord{
        .port = state.ports[2U].id,
        .kind = PortRateKind::kFixedNominal,
        .declared_nominal_rate = Rate::hz(60U),
        .disposition = PlanDisposition::kAccepted,
    };
  });
  expect_rejection([](auto& state) {
    ASSERT_TRUE(state.accumulation_horizons.front().horizon.has_value());
    state.accumulation_horizons.front().horizon->denominator = 0U;
  });
  expect_rejection([](auto& state) {
    state.accumulation_horizons.front().inverse_rate_port = state.payload_edges.front().target;
  });
  expect_rejection([](auto& state) {
    state.accumulation_horizons.front().horizon.reset();
    state.accumulation_horizons.front().inverse_rate_port.reset();
  });
  expect_rejection([](auto& state) { state.accumulation_horizons.front().source_record = 0U; });
  expect_rejection([](auto& state) { ++state.accumulation_horizons.front().source_record; });
  expect_rejection([](auto& state) {
    state.accumulation_horizons.front().origin = AccumulationHorizonOrigin::kTemporalWindowBound;
  });
  expect_rejection([](auto& state) {
    state.accumulation_horizons.front().origin = AccumulationHorizonOrigin::kExternalCadence;
  });
  expect_rejection([](auto& state) {
    state.temporal_triggers.front().kind = TemporalRecordTriggerKind::kEach;
    state.temporal_triggers.front().trigger_input = state.payload_edges.front().target;
  });
  expect_rejection([](auto& state) { ++state.temporal_triggers.front().period_ns; });
  expect_rejection([](auto& state) {
    state.accumulation_horizons.front().origin = AccumulationHorizonOrigin::kDriverInterarrival;
  });
  expect_rejection([](auto& state) { state.hard_floor_contributors.front().required_slots = 0U; });
  expect_rejection(
      [](auto& state) { state.queue_sizing_recipes.front().headroom_ratio = RateRatio{1U, 2U}; });
  expect_rejection([](auto& state) { state.queue_sizing_recipes.pop_back(); });
  expect_rejection([](auto& state) { state.hard_floor_contributors.pop_back(); });

  auto retained_drain_all = physical_artifact_state();
  add_valid_rate_records(retained_drain_all);
  ASSERT_FALSE(retained_drain_all.payload_edges.empty());
  const auto& edge = retained_drain_all.payload_edges.front();
  retained_drain_all.input_selections.push_back(InputSelectionPlanRecord{
      .input = edge.target,
      .selector = InputSelectorKind::kDrainAll,
      .cardinality = InputSelectionCardinality::kSampleBatch,
      .max_selected_samples = edge.queue_depth,
      .max_selected_samples_origin = SelectedSampleBoundOrigin::kResolvedRetainedCapacity,
      .batch_selection_mode = BatchSelectionMode::kConsumeSelected,
      .cache_capacity = edge.queue_depth,
  });
  const auto retained_drain_all_finalized = finalize_plan_artifact(retained_drain_all);
  ASSERT_TRUE(retained_drain_all_finalized) << retained_drain_all_finalized.error().message;

  retained_drain_all.hard_floor_contributors.push_back(HardFloorContributorPlanRecord{
      .id = HardFloorContributorId{static_cast<std::uint32_t>(
          retained_drain_all.hard_floor_contributors.size() + 1U)},
      .edge = edge.edge,
      .origin = HardFloorOrigin::kSelectorRetention,
      .required_slots = edge.queue_depth,
      .scenario = HardFloorScenarioId{17U},
      .alternative_group = HardFloorAlternativeGroupId{17U},
  });
  retained_drain_all.json.clear();
  const auto circular_floor_finalized = finalize_plan_artifact(retained_drain_all);
  ASSERT_FALSE(circular_floor_finalized);
  EXPECT_EQ(circular_floor_finalized.error().code, ErrorCode::kInvalidArgument);
  EXPECT_EQ(circular_floor_finalized.error().message, "RATE_PLAN_RECORDS_INVALID");
  EXPECT_TRUE(retained_drain_all.json.empty());
}

// A private-v7 horizon is accepted only when it is an exact projection of the destination
// operator's canonical clock, predicate clock, window, or OnEach trigger.
// REQ: COMPILE-069
TEST(CompilePhysicalPlanArtifact, AcceptsCanonicalRateHorizonTriggerProvenance) {
  auto clock = physical_artifact_state();
  add_valid_rate_records(clock);
  const auto clock_finalized = finalize_plan_artifact(clock);
  ASSERT_TRUE(clock_finalized) << clock_finalized.error().message;

  auto window = physical_artifact_state();
  add_valid_rate_records(window);
  use_valid_window_rate_horizon(window);
  const auto window_finalized = finalize_plan_artifact(window);
  ASSERT_TRUE(window_finalized) << window_finalized.error().message;

  auto predicate_clock = physical_artifact_state();
  add_valid_rate_records(predicate_clock);
  use_valid_predicate_clock_rate_horizon(predicate_clock);
  const auto predicate_clock_finalized = finalize_plan_artifact(predicate_clock);
  ASSERT_TRUE(predicate_clock_finalized) << predicate_clock_finalized.error().message;

  auto inverse = physical_artifact_state();
  add_valid_rate_records(inverse);
  auto& inverse_record = inverse.accumulation_horizons.front();
  inverse_record.horizon.reset();
  inverse_record.inverse_rate_port = inverse.payload_edges.front().target;
  inverse_record.origin = AccumulationHorizonOrigin::kProducerInterarrival;
  auto& inverse_trigger = inverse.temporal_triggers.front();
  inverse_trigger.kind = TemporalRecordTriggerKind::kEach;
  inverse_trigger.trigger_clock = ClockId{};
  inverse_trigger.period_ns = 0;
  inverse_trigger.trigger_input = inverse_record.inverse_rate_port.value();
  const auto inverse_finalized = finalize_plan_artifact(inverse);
  ASSERT_TRUE(inverse_finalized) << inverse_finalized.error().message;
}

// A periodic duration is not attributable without the exact clock identity that owns it.
// REQ: COMPILE-069
TEST(CompilePhysicalPlanArtifact, RejectsClockHorizonWithoutTriggerClock) {
  auto candidate = physical_artifact_state();
  add_valid_rate_records(candidate);
  candidate.temporal_triggers.front().trigger_clock = ClockId{};

  const auto finalized = finalize_plan_artifact(candidate);
  ASSERT_FALSE(finalized);
  EXPECT_EQ(finalized.error().code, ErrorCode::kInvalidArgument);
  EXPECT_EQ(finalized.error().message, "RATE_PLAN_RECORDS_INVALID");
  EXPECT_TRUE(candidate.json.empty());
}

// A window horizon joins the exact WindowTriggerPlanRecord that owns max_wait_ns. A generic
// kWindow trigger alone cannot authorize a duration because its period_ns is intentionally zero.
// REQ: COMPILE-069
TEST(CompilePhysicalPlanArtifact, RejectsMismatchedWindowHorizonTriggerProvenance) {
  const auto expect_rejection = [](auto mutate) {
    auto candidate = physical_artifact_state();
    add_valid_rate_records(candidate);
    use_valid_window_rate_horizon(candidate);
    mutate(candidate);

    const auto finalized = finalize_plan_artifact(candidate);
    ASSERT_FALSE(finalized);
    EXPECT_EQ(finalized.error().code, ErrorCode::kInvalidArgument);
    EXPECT_EQ(finalized.error().message, "RATE_PLAN_RECORDS_INVALID");
    EXPECT_TRUE(candidate.json.empty());
  };

  expect_rejection([](auto& state) { state.window_triggers.clear(); });
  expect_rejection([](auto& state) {
    const auto duplicate = state.window_triggers.front();
    state.window_triggers.push_back(duplicate);
  });
  expect_rejection([](auto& state) { ++state.window_triggers.front().owner.id; });
  expect_rejection([](auto& state) { ++state.window_triggers.front().owner.plan_generation; });
  expect_rejection([](auto& state) { state.window_triggers.front().anchor = PortId{1U}; });
  expect_rejection([](auto& state) {
    state.temporal_triggers.front().trigger_input = PortId{};
    state.window_triggers.front().anchor = PortId{};
  });
  expect_rejection([](auto& state) { ++state.window_triggers.front().max_wait_ns; });
  expect_rejection([](auto& state) { state.window_triggers.front().wait_clock = ClockId{2U}; });
}

// Predicate-clock cadence is finite only when the exact canonical predicate source record agrees
// with the generic temporal trigger. Input-driven predicates remain unclassified.
// REQ: COMPILE-069
TEST(CompilePhysicalPlanArtifact, RejectsMismatchedPredicateClockHorizonProvenance) {
  const auto expect_rejection = [](auto mutate) {
    auto candidate = physical_artifact_state();
    add_valid_rate_records(candidate);
    use_valid_predicate_clock_rate_horizon(candidate);
    mutate(candidate);

    const auto finalized = finalize_plan_artifact(candidate);
    ASSERT_FALSE(finalized);
    EXPECT_EQ(finalized.error().code, ErrorCode::kInvalidArgument);
    EXPECT_EQ(finalized.error().message, "RATE_PLAN_RECORDS_INVALID");
    EXPECT_TRUE(candidate.json.empty());
  };

  expect_rejection([](auto& state) { state.predicate_triggers.clear(); });
  expect_rejection([](auto& state) {
    const auto duplicate = state.predicate_triggers.front();
    state.predicate_triggers.push_back(duplicate);
  });
  expect_rejection([](auto& state) {
    state.predicate_triggers.front().source = PredicateEvaluationSourceKind::kInputUpdate;
  });
  expect_rejection([](auto& state) { ++state.predicate_triggers.front().owner.id; });
  expect_rejection([](auto& state) { ++state.predicate_triggers.front().period_ns; });
  expect_rejection([](auto& state) { state.predicate_triggers.front().clock = ClockId{2U}; });
}

// Inverse horizons fail closed when any source, input, or origin fact diverges from the canonical
// OnEach trigger that owns the destination edge.
// REQ: COMPILE-069
TEST(CompilePhysicalPlanArtifact, RejectsMismatchedInverseHorizonTriggerProvenance) {
  const auto expect_rejection = [](auto mutate) {
    auto candidate = physical_artifact_state();
    add_valid_rate_records(candidate);
    auto& record = candidate.accumulation_horizons.front();
    record.horizon.reset();
    record.inverse_rate_port = candidate.payload_edges.front().target;
    record.origin = AccumulationHorizonOrigin::kProducerInterarrival;
    auto& trigger = candidate.temporal_triggers.front();
    trigger.kind = TemporalRecordTriggerKind::kEach;
    trigger.trigger_clock = ClockId{};
    trigger.period_ns = 0;
    trigger.trigger_input = record.inverse_rate_port.value();
    mutate(candidate);

    const auto finalized = finalize_plan_artifact(candidate);
    ASSERT_FALSE(finalized);
    EXPECT_EQ(finalized.error().code, ErrorCode::kInvalidArgument);
    EXPECT_EQ(finalized.error().message, "RATE_PLAN_RECORDS_INVALID");
    EXPECT_TRUE(candidate.json.empty());
  };

  expect_rejection([](auto& state) { ++state.accumulation_horizons.front().source_record; });
  expect_rejection([](auto& state) { state.temporal_triggers.front().trigger_input = PortId{7U}; });
  expect_rejection(
      [](auto& state) { state.accumulation_horizons.front().inverse_rate_port = PortId{7U}; });
  expect_rejection([](auto& state) {
    state.accumulation_horizons.front().origin = AccumulationHorizonOrigin::kDriverInterarrival;
  });
}

// Task041 pool dependencies and provider recipes are semantic artifact identity, while malformed
// source joins and provider ranges fail before entering the private-v7 digest preimage.
// REQ: COMPILE-069
TEST(CompilePhysicalPlanArtifact, PoolCapacityFamiliesAreCanonicalAndFailClosed) {
  auto baseline = physical_artifact_state();
  ASSERT_FALSE(baseline.pool_capacity_dependencies.empty());
  ASSERT_FALSE(baseline.provider_allocation_plans.empty());
  const auto baseline_finalized = finalize_plan_artifact(baseline);
  ASSERT_TRUE(baseline_finalized) << baseline_finalized.error().message;

  auto changed = physical_artifact_state();
  changed.pool_capacity_dependencies.front().scenario = PoolCapacityScenarioId{2U};
  const auto changed_finalized = finalize_plan_artifact(changed);
  ASSERT_TRUE(changed_finalized) << changed_finalized.error().message;
  EXPECT_NE(changed.digest, baseline.digest);
  EXPECT_NE(changed.json, baseline.json);

  auto changed_provider = physical_artifact_state();
  const std::string old_pool_id = changed_provider.provider_allocation_plans.front().pool_id;
  const std::string changed_pool_id = old_pool_id + ".semantic-change";
  changed_provider.provider_allocation_plans.front().pool_id = changed_pool_id;
  for (auto& dependency : changed_provider.pool_capacity_dependencies) {
    if (dependency.pool_id == old_pool_id) {
      dependency.pool_id = changed_pool_id;
    }
  }
  for (auto& contributor : changed_provider.pool_capacity_contributors) {
    if (contributor.pool_id == old_pool_id) {
      contributor.pool_id = changed_pool_id;
    }
  }
  for (auto& allocation : changed_provider.tensor_allocations) {
    if (allocation.pool_id == old_pool_id) {
      allocation.pool_id = changed_pool_id;
    }
  }
  for (auto& allocation : changed_provider.tensor_field_allocations) {
    if (allocation.pool_id == old_pool_id) {
      allocation.pool_id = changed_pool_id;
    }
  }
  const auto changed_provider_finalized = finalize_plan_artifact(changed_provider);
  ASSERT_TRUE(changed_provider_finalized) << changed_provider_finalized.error().message;
  EXPECT_NE(changed_provider.digest, baseline.digest);
  EXPECT_NE(changed_provider.json, baseline.json);
  EXPECT_NE(baseline.json.find("\"first_sizing_dependency\":"), std::string::npos);
  EXPECT_NE(baseline.json.find("\"mapping_multiplicity\":"), std::string::npos);
  EXPECT_NE(baseline.json.find("\"predecessor_contributor_id\":"), std::string::npos);
  EXPECT_NE(baseline.json.find("\"geometry_mode\":"), std::string::npos);

  const auto expect_rejection = [](auto mutate) {
    auto candidate = physical_artifact_state();
    mutate(candidate);
    const auto finalized = finalize_plan_artifact(candidate);
    ASSERT_FALSE(finalized);
    EXPECT_EQ(finalized.error().code, ErrorCode::kInvalidArgument);
    EXPECT_EQ(finalized.error().message, "POOL_CAPACITY_PLAN_INVALID");
    EXPECT_TRUE(candidate.json.empty());
  };
  expect_rejection([](auto& state) { state.pool_capacity_dependencies.front().id = {}; });
  expect_rejection([](auto& state) { state.pool_capacity_dependencies.clear(); });
  expect_rejection([](auto& state) { state.provider_allocation_plans.clear(); });
  expect_rejection(
      [](auto& state) { state.pool_capacity_dependencies.front().mapping_multiplicity = 0U; });
  expect_rejection([](auto& state) {
    state.pool_capacity_dependencies.front().predecessor_contributor_id = "missing";
  });
  expect_rejection(
      [](auto& state) { state.provider_allocation_plans.front().sizing_dependency_count = 0U; });
  expect_rejection([](auto& state) { state.provider_allocation_plans.front().output.reset(); });
}

TEST(CompilePhysicalPlanArtifact, PredicateCallableProvenanceDoesNotEnterSemanticDigest) {
  auto first = physical_artifact_state();
  const PredicateCallablePlanRecord callable{
      .predicate_id = "predicate",
      .binding_provenance = "first-binding",
      .capture_provenance = "first-capture",
  };
  first.predicate_callables.push_back(callable);
  auto second = first;
  second.predicate_callables.front().binding_provenance = "second-binding";
  second.predicate_callables.front().capture_provenance = "second-capture";

  // Key setup: finalize two otherwise identical plans whose callable provenance differs.
  ASSERT_TRUE(finalize_plan_artifact(first));
  ASSERT_TRUE(finalize_plan_artifact(second));

  // Key assertion: complete JSON retains provenance while the semantic digest excludes it.
  EXPECT_EQ(first.digest, second.digest);
  EXPECT_NE(first.json, second.json);
  EXPECT_NE(first.json.find("\"binding_provenance\":\"first-binding\""), std::string::npos);
  EXPECT_NE(first.json.find("\"capture_provenance\":\"first-capture\""), std::string::npos);
  EXPECT_NE(second.json.find("\"binding_provenance\":\"second-binding\""), std::string::npos);
  EXPECT_NE(second.json.find("\"capture_provenance\":\"second-capture\""), std::string::npos);
}

// REQ: COMPILE-002
// REQ: COMPILE-009
// REQ: COMPILE-095
// REQ: COMPILE-040
// REQ: COMPILE-048
// REQ: COMPILE-064
TEST(CompilePhysicalPlanArtifact, EveryPublicPhysicalFamilyPerturbsSemanticIdentity) {
  auto baseline = physical_artifact_state();
  const auto baseline_finalized = finalize_plan_artifact(baseline);
  ASSERT_TRUE(baseline_finalized) << baseline_finalized.error().message;
  ASSERT_FALSE(baseline.cuda_compute_streams.empty());
  const auto expect_change = [&](auto mutate) {
    auto candidate = physical_artifact_state();
    mutate(candidate);
    const auto finalized = finalize_plan_artifact(candidate);
    ASSERT_TRUE(finalized) << finalized.error().message;
    EXPECT_NE(candidate.digest, baseline.digest);
    EXPECT_NE(candidate.json, baseline.json);
  };

  expect_change([](auto& state) { state.ports.front().local_name = "changed"; });
  expect_change([](auto& state) {
    state.interface_port_bindings.front().direction = PortDirection::kOutput;
  });
  expect_change([](auto& state) { ++state.payload_edges.front().max_inflight_samples; });
  expect_change([](auto& state) {
    // Locality acceptance and the payload-edge locality are one closed
    // compiler relation. Perturb the external edge coherently so this test
    // continues to exercise artifact identity rather than corruption rejection.
    state.payload_edges.back().locality = ChannelLocalityKind::kExternalExport;
    state.channel_locality.back().accepted = ChannelLocalityKind::kExternalExport;
  });
  expect_change([](auto& state) { ++state.channel_services.front().max_subscriber_queue_depth; });
  expect_change([](auto& state) { ++state.channel_publishers.front().max_loaned_samples; });
  expect_change([](auto& state) { ++state.channel_subscribers.front().max_borrowed_samples; });
  expect_change([](auto& state) { ++state.generic_output_loans.front().slot_count; });
  expect_change([](auto& state) { ++state.tensor_allocations.front().bounds.max_rank; });
  expect_change([](auto& state) {
    state.tensor_allocations.front().layout->strides = TensorStridePolicy::kContiguousOnly;
  });
  expect_change([](auto& state) {
    state.tensor_allocations.front().axes.push_back(
        TensorAxisBounds{.max_extent = 16U, .max_abs_stride_bytes = 256U});
  });
  expect_change([](auto& state) { ++state.tensor_device_resources.front().slot_capacity; });
  expect_change([](auto& state) {
    const std::string original = state.cuda_compute_streams.front().stream_id;
    state.cuda_compute_streams.front().stream_id += ".changed";
    for (auto& binding : state.operator_cuda_stream_bindings) {
      if (binding.stream_id == original) {
        binding.stream_id = state.cuda_compute_streams.front().stream_id;
      }
    }
  });
  expect_change([](auto& state) { ++*state.tensor_interface_pools.front().logical.max_byte_span; });
  expect_change([](auto& state) {
    state.tensor_interface_pools.front().logical.memory_conversion =
        MemoryConversionPolicy::kAllowIfSafe;
  });
  expect_change([](auto& state) {
    ASSERT_FALSE(state.tensor_interface_pools.front().logical.axes.empty());
    --state.tensor_interface_pools.front().logical.axes.front().max_extent;
  });
  expect_change(
      [](auto& state) { ++state.tensor_inline_records.front().fixed_record_bucket_bytes; });
  expect_change(
      [](auto& state) { ++state.tensor_inline_records.front().provider_max_inline_logical_bytes; });
  expect_change(
      [](auto& state) { state.tensor_inline_records.front().record_format_version += 1U; });
  expect_change(
      [](auto& state) { ++state.imported_generation_capacities.front().runtime_max_generations; });
  expect_change(
      [](auto& state) { state.imported_generation_capacities.front().attach_max_wait_ns = 1U; });
  expect_change(
      [](auto& state) { state.tensor_transfer_plans.front().delivered_read_only = false; });
  expect_change([](auto& state) { ++state.tensor_transfer_plans.front().min_logical_bytes; });
  expect_change(
      [](auto& state) { ++state.pool_capacity_contributors.front().maximum_distinct_slots; });
  expect_change(
      [](auto& state) { state.publication_outcomes.front().runtime_retry_enabled = true; });
  expect_change([](auto& state) {
    state.publication_outcomes.front().post_submit_error = ErrorCode::kFailure;
  });
  expect_change([](auto& state) {
    state.publication_outcomes.front().post_submit_partial_error = ErrorCode::kFailure;
  });
  expect_change([](auto& state) { ++state.aligned_triggers.front().inclusive_window_ns; });
  expect_change([](auto& state) { ++state.clock_triggers.front().period_ns; });
  expect_change([](auto& state) { ++state.deadlines.front().budget_ns; });
  expect_change([](auto& state) { ++state.readable_triggers.front().effective_capacity; });
  expect_change([](auto& state) { ++state.missing_aggregates.front().degraded_callback_count; });
}

// REQ: COMPILE-063
TEST(CompilePhysicalPlanArtifact, RejectsMalformedCudaExecutionRecordJoinsBeforeSerialization) {
  // Key setup: independently corrupt every identity join in an otherwise valid physical artifact.
  const auto expect_rejection = [](auto mutate) {
    auto candidate = physical_artifact_state();
    ASSERT_FALSE(candidate.tensor_device_resources.empty());
    ASSERT_FALSE(candidate.cuda_compute_streams.empty());
    mutate(candidate);
    const auto finalized = finalize_plan_artifact(candidate);
    ASSERT_FALSE(finalized);
    EXPECT_EQ(finalized.error().message, "CUDA_EXECUTION_PLAN_INVALID");
    EXPECT_EQ(diagnostic_id(finalized.error().message),
              DiagnosticId{
                  static_cast<std::uint32_t>(StructuralDiagnosticId::kCUDAExecutionPlanInvalid)});
  };

  // Key assertion: duplicate resources/streams and broken generation, edge, context, or owner
  // references all fail final validation with the same stable diagnostic.
  expect_rejection([](auto& state) {
    state.tensor_device_resources.push_back(state.tensor_device_resources.front());
  });
  expect_rejection([](auto& state) {
    state.cuda_compute_streams.push_back(state.cuda_compute_streams.front());
  });
  expect_rejection([](auto& state) { ++state.cuda_compute_streams.front().plan_generation; });
  expect_rejection(
      [](auto& state) { state.cuda_compute_streams.front().fused_linear_edges = {EdgeId{999U}}; });
  expect_rejection([](auto& state) {
    state.cuda_compute_streams.front().green_context_id = "missing-green-context";
  });
  expect_rejection([](auto& state) { state.cuda_compute_streams.front().owner_id += ".wrong"; });
  expect_rejection([](auto& state) {
    state.cuda_compute_stream_policy = static_cast<CudaComputeStreamPolicy>(255U);
  });
  expect_rejection([](auto& state) { state.operator_cuda_stream_bindings.clear(); });
  expect_rejection([](auto& state) {
    state.operator_cuda_stream_bindings.push_back(state.operator_cuda_stream_bindings.front());
  });
  expect_rejection([](auto& state) {
    state.operator_cuda_stream_bindings.front().stream_id = "tensor-stream.missing";
  });
  expect_rejection([](auto& state) {
    state.operator_cuda_stream_bindings.front().partition = ExecutionPartitionId{999U};
  });
  expect_rejection([](auto& state) { ++state.tensor_allocations.front().release_stream_capacity; });
}

// REQ: COMPILE-002
// REQ: COMPILE-009
// REQ: COMPILE-095
// REQ: COMPILE-040
// REQ: CHAN-065
TEST(CompilePhysicalPlanArtifact, RejectsProcessLocalConstructionFactDrift) {
  const auto expect_rejection = [&](auto mutate) {
    auto candidate = physical_artifact_state();
    mutate(candidate);
    const auto finalized = finalize_plan_artifact(candidate);
    ASSERT_FALSE(finalized);
    EXPECT_EQ(finalized.error().message, "PROCESS_LOCAL_CHANNEL_PLAN_INVALID");
  };

  expect_rejection(
      [](auto& state) { ++state.process_local_channel_services.front().payload_size_bytes; });
  expect_rejection(
      [](auto& state) { ++state.process_local_channel_publishers.front().max_loaned_samples; });
  expect_rejection(
      [](auto& state) { ++state.process_local_channel_subscribers.front().cache_capacity; });
  expect_rejection([](auto& state) {
    state.process_local_channel_services.front().event_wake_reason =
        ProcessLocalHoloMQEventWakeReason::kReplayAmbiguity;
  });

  auto ineligible_direct = physical_artifact_state();
  ASSERT_EQ(ineligible_direct.process_local_channel_services.front().delivery_kind,
            ProcessLocalChannelDeliveryKind::kDirectRetained);
  ineligible_direct.payload_edges.front().fanout = 2U;
  const auto finalized = finalize_plan_artifact(ineligible_direct);
  ASSERT_FALSE(finalized);
  EXPECT_EQ(finalized.error().message, "PROCESS_LOCAL_CHANNEL_PLAN_INVALID");
}

// REQ: COMPILE-002
// REQ: COMPILE-040
// REQ: CHAN-065
TEST(CompilePhysicalPlanArtifact, RejectsBrokenProcessLocalChannelJoins) {
  {
    auto missing_subscriber = physical_artifact_state();
    ASSERT_FALSE(missing_subscriber.process_local_channel_subscribers.empty());
    missing_subscriber.process_local_channel_subscribers.clear();
    const auto finalized = finalize_plan_artifact(missing_subscriber);
    ASSERT_FALSE(finalized);
    EXPECT_EQ(finalized.error().message, "PROCESS_LOCAL_CHANNEL_PLAN_INVALID");
  }
  {
    auto missing_service_identity = physical_artifact_state();
    ASSERT_FALSE(missing_service_identity.process_local_channel_services.empty());
    missing_service_identity.process_local_channel_services.front().service_id.clear();
    const auto finalized = finalize_plan_artifact(missing_service_identity);
    ASSERT_FALSE(finalized);
    EXPECT_EQ(finalized.error().message, "PROCESS_LOCAL_CHANNEL_PLAN_INVALID");
  }
  {
    auto duplicate_service = physical_artifact_state();
    ASSERT_FALSE(duplicate_service.process_local_channel_services.empty());
    duplicate_service.process_local_channel_services.push_back(
        duplicate_service.process_local_channel_services.front());
    const auto finalized = finalize_plan_artifact(duplicate_service);
    ASSERT_FALSE(finalized);
    EXPECT_EQ(finalized.error().message, "PROCESS_LOCAL_CHANNEL_PLAN_INVALID");
  }
  {
    auto mismatched_payload = physical_artifact_state();
    ASSERT_FALSE(mismatched_payload.process_local_channel_services.empty());
    ASSERT_TRUE(mismatched_payload.process_local_channel_services.front().payload_identity.scalar);
    // Int32 has the same physical width as UInt32. The exact payload identity,
    // not only its byte count, is part of the private construction join.
    mismatched_payload.process_local_channel_services.front().payload_identity.scalar->kind =
        ScalarIdentityKind::kInt32;
    const auto finalized = finalize_plan_artifact(mismatched_payload);
    ASSERT_FALSE(finalized);
    EXPECT_EQ(finalized.error().message, "PROCESS_LOCAL_CHANNEL_PLAN_INVALID");
  }
  {
    auto mismatched_source = physical_artifact_state();
    ASSERT_FALSE(mismatched_source.process_local_channel_publishers.empty());
    mismatched_source.process_local_channel_publishers.front().source = PortId{2U};
    const auto finalized = finalize_plan_artifact(mismatched_source);
    ASSERT_FALSE(finalized);
    EXPECT_EQ(finalized.error().message, "PROCESS_LOCAL_CHANNEL_PLAN_INVALID");
  }
  {
    auto duplicate_publisher = physical_artifact_state();
    ASSERT_FALSE(duplicate_publisher.process_local_channel_publishers.empty());
    duplicate_publisher.process_local_channel_publishers.push_back(
        duplicate_publisher.process_local_channel_publishers.front());
    const auto finalized = finalize_plan_artifact(duplicate_publisher);
    ASSERT_FALSE(finalized);
    EXPECT_EQ(finalized.error().message, "PROCESS_LOCAL_CHANNEL_PLAN_INVALID");
  }
  {
    auto duplicate_subscriber = physical_artifact_state();
    ASSERT_FALSE(duplicate_subscriber.process_local_channel_subscribers.empty());
    duplicate_subscriber.process_local_channel_subscribers.push_back(
        duplicate_subscriber.process_local_channel_subscribers.front());
    const auto finalized = finalize_plan_artifact(duplicate_subscriber);
    ASSERT_FALSE(finalized);
    EXPECT_EQ(finalized.error().message, "PROCESS_LOCAL_CHANNEL_PLAN_INVALID");
  }
  {
    GraphSnapshotData snapshot;
    snapshot.generation = GraphGeneration{1U};
    snapshot.vertices = {vertex(1U, "tensor-source"), vertex(2U, "tensor-sink")};
    snapshot.ports = {
        output_port(1U, 1U, "out", tensor_payload()),
        input_port(2U, 2U, "in", tensor_payload(), 2U),
    };
    snapshot.ports.front().setup.tensor_layout = TensorRepresentation{
        .memory_kind = MemoryKind::kHost,
        .dtype = DLDataType{kDLUInt, 8U, 1U},
        .rank = 1U,
    };
    snapshot.ports.front().setup.tensor_allocation =
        TensorPoolBounds{.max_rank = 1U, .max_byte_span = 64U};
    snapshot.edges = {edge(1U, 1U, 2U, 2U)};
    const auto profiles = standard_profiles(snapshot.vertices.size());
    auto physical = lower_for_test(snapshot, 37U, profiles);
    ASSERT_TRUE(physical.errors.empty());
    ASSERT_EQ(physical.tensor_allocations.size(), 1U);
    ASSERT_EQ(physical.process_local_channel_services.size(), 1U);

    auto ambiguous_tensor_pool = published_physical_plan_state(std::move(physical), 37U);
    add_valid_rate_records(ambiguous_tensor_pool);
    ASSERT_TRUE(valid_process_local_channel_plan(ambiguous_tensor_pool));
    const auto& direct = ambiguous_tensor_pool.tensor_allocations.front();
    TensorInterfacePoolPlanRecord duplicate_pool;
    duplicate_pool.graph_port = direct.output;
    duplicate_pool.pool_id = direct.pool_id + ".ambiguous";
    duplicate_pool.logical.max_byte_span = direct.bounds.max_byte_span;
    duplicate_pool.memory_kind = direct.memory_kind;
    duplicate_pool.device = direct.device;
    ambiguous_tensor_pool.tensor_interface_pools.push_back(std::move(duplicate_pool));
    const auto finalized = finalize_plan_artifact(ambiguous_tensor_pool);
    ASSERT_FALSE(finalized);
    EXPECT_EQ(finalized.error().message, "PROCESS_LOCAL_CHANNEL_PLAN_INVALID");

    auto duplicate_direct =
        published_physical_plan_state(lower_for_test(snapshot, 37U, profiles), 37U);
    add_valid_rate_records(duplicate_direct);
    ASSERT_EQ(duplicate_direct.tensor_allocations.size(), 1U);
    auto second_direct = duplicate_direct.tensor_allocations.front();
    second_direct.pool_id += ".duplicate";
    duplicate_direct.tensor_allocations.push_back(std::move(second_direct));
    EXPECT_FALSE(valid_process_local_channel_plan(duplicate_direct));

    auto duplicate_boundary =
        published_physical_plan_state(lower_for_test(snapshot, 37U, profiles), 37U);
    add_valid_rate_records(duplicate_boundary);
    ASSERT_EQ(duplicate_boundary.tensor_allocations.size(), 1U);
    const auto direct_facts = duplicate_boundary.tensor_allocations.front();
    duplicate_boundary.tensor_allocations.clear();
    TensorInterfacePoolPlanRecord first_boundary;
    first_boundary.graph_port = direct_facts.output;
    first_boundary.pool_id = direct_facts.pool_id + ".boundary";
    first_boundary.logical.max_byte_span = direct_facts.bounds.max_byte_span;
    first_boundary.memory_kind = direct_facts.memory_kind;
    first_boundary.device = direct_facts.device;
    duplicate_boundary.tensor_interface_pools.push_back(first_boundary);
    first_boundary.pool_id += ".duplicate";
    duplicate_boundary.tensor_interface_pools.push_back(std::move(first_boundary));
    EXPECT_FALSE(valid_process_local_channel_plan(duplicate_boundary));

    auto duplicate_candidate =
        published_physical_plan_state(lower_for_test(snapshot, 37U, profiles), 37U);
    add_valid_rate_records(duplicate_candidate);
    ASSERT_EQ(duplicate_candidate.tensor_allocations.size(), 1U);
    const auto candidate_facts = duplicate_candidate.tensor_allocations.front();
    duplicate_candidate.tensor_allocations.clear();
    const auto provider = std::ranges::find(duplicate_candidate.provider_allocation_plans,
                                            candidate_facts.output,
                                            &ProviderAllocationPlanRecord::output);
    ASSERT_NE(provider, duplicate_candidate.provider_allocation_plans.end());
    provider->geometry_mode = AllocationGeometryMode::kLifecycleCandidate;
    provider->tensor_template = TensorProviderTemplatePlanRecord{
        .allocator_root_id = candidate_facts.allocator_root_id,
        .allocator_process_group = candidate_facts.allocator_process_group,
        .bounds = candidate_facts.bounds,
        .axes = candidate_facts.axes,
        .layout = candidate_facts.layout,
        .memory_kind = candidate_facts.memory_kind,
        .device = candidate_facts.device,
        .max_outstanding_loans = candidate_facts.max_outstanding_loans,
        .max_emits_per_compute = candidate_facts.max_emits_per_compute,
        .fallback_rejected = true,
    };
    ASSERT_TRUE(valid_process_local_channel_plan(duplicate_candidate));
    auto second_candidate = *provider;
    second_candidate.id.value += 1U;
    second_candidate.pool_id += ".duplicate";
    duplicate_candidate.provider_allocation_plans.push_back(std::move(second_candidate));
    EXPECT_FALSE(valid_process_local_channel_plan(duplicate_candidate));
  }
}

// REQ: COMPILE-002
// REQ: COMPILE-040
// REQ: CHAN-065
TEST(CompilePhysicalPlanArtifact, RejectsProcessLocalAndExternalLocalityJoinDrift) {
  const auto expect_rejection = [](auto mutate) {
    auto candidate = physical_artifact_state();
    mutate(candidate);
    const auto finalized = finalize_plan_artifact(candidate);
    ASSERT_FALSE(finalized);
    EXPECT_EQ(finalized.error().message, "PROCESS_LOCAL_CHANNEL_PLAN_INVALID");
  };

  expect_rejection([](auto& state) {
    const auto found = std::find_if(
        state.channel_locality.begin(), state.channel_locality.end(), [](const auto& row) {
          return row.accepted == ChannelLocalityKind::kProcessLocal;
        });
    ASSERT_NE(found, state.channel_locality.end());
    found->accepted = ChannelLocalityKind::kExternalExport;
    found->requires_runtime_attach = true;
  });
  expect_rejection([](auto& state) {
    const auto found = std::find_if(
        state.channel_locality.begin(), state.channel_locality.end(), [](const auto& row) {
          return row.accepted != ChannelLocalityKind::kProcessLocal;
        });
    ASSERT_NE(found, state.channel_locality.end());
    found->accepted = ChannelLocalityKind::kProcessLocal;
    found->requires_runtime_attach = false;
  });
}

// REQ: COMPILE-002
// REQ: COMPILE-040
// REQ: COMPILE-048
// REQ: COMPILE-064
TEST(CompilePhysicalPlanArtifact, RebindsProcessLocalServiceIdentityAsOneJoinFamily) {
  auto state = physical_artifact_state();
  ASSERT_FALSE(state.process_local_channel_services.empty());
  ASSERT_FALSE(state.tensor_device_resources.empty());
  ASSERT_FALSE(state.cuda_compute_streams.empty());
  ASSERT_FALSE(state.pool_capacity_dependencies.empty());
  ASSERT_FALSE(state.provider_allocation_plans.empty());
  const std::string original_pool_id = state.provider_allocation_plans.front().pool_id;
  const std::string original_allocator_id = state.tensor_device_resources.front().allocator_root_id;
  const std::string original_stream_id = state.cuda_compute_streams.front().stream_id;
  ASSERT_FALSE(state.operator_cuda_stream_bindings.empty());
  state.profile_manifest.derived_capacities.push_back(ProfileDerivedCapacityRecord{
      .object_id = original_allocator_id,
      .capacity_id = "slot_count",
      .limit = state.tensor_device_resources.front().slot_capacity,
      .unit = "count",
  });
  state.profile_manifest.materialization_obligations.push_back(
      ProfileMaterializationObligationRecord{
          .object_id = original_stream_id,
          .strategy_id = "holoscan.runtime.cuda_compute_stream/v1",
          .complete_before_stage = LifecycleStage::kArm,
          .cleanup_stage_mask = 0xC0U,
          .max_cleanup_actions = 1U,
          .evidence_id = "holoscan.compile.cuda_compute_stream_plan/v1",
      });

  rebind_plan_generation(state, 99U);

  const auto& service = state.process_local_channel_services.front();
  EXPECT_EQ(service.service_id, "process-local-channel.99.source.1.policy.0");
  ASSERT_FALSE(state.process_local_channel_publishers.empty());
  ASSERT_FALSE(state.process_local_channel_subscribers.empty());
  EXPECT_EQ(state.process_local_channel_publishers.front().service_id, service.service_id);
  EXPECT_EQ(state.process_local_channel_subscribers.front().service_id, service.service_id);
  EXPECT_NE(state.tensor_device_resources.front().allocator_root_id, original_allocator_id);
  EXPECT_NE(state.cuda_compute_streams.front().stream_id, original_stream_id);
  EXPECT_EQ(state.operator_cuda_stream_bindings.front().plan_generation, 99U);
  EXPECT_NE(state.operator_cuda_stream_bindings.front().stream_id, original_stream_id);
  EXPECT_EQ(state.operator_cuda_stream_bindings.front().stream_id,
            state.cuda_compute_streams.front().stream_id);
  EXPECT_NE(state.provider_allocation_plans.front().pool_id, original_pool_id);
  EXPECT_EQ(state.pool_capacity_dependencies.front().pool_id,
            state.provider_allocation_plans.front().pool_id);
  if (!state.pool_capacity_dependencies.front().predecessor_contributor_id.empty()) {
    EXPECT_TRUE(state.pool_capacity_dependencies.front().predecessor_contributor_id.starts_with(
        state.provider_allocation_plans.front().pool_id));
  }
  ASSERT_EQ(state.profile_manifest.derived_capacities.size(), 1U);
  EXPECT_EQ(state.profile_manifest.derived_capacities.front().object_id,
            state.tensor_device_resources.front().allocator_root_id);
  ASSERT_EQ(state.profile_manifest.materialization_obligations.size(), 1U);
  EXPECT_EQ(state.profile_manifest.materialization_obligations.front().object_id,
            state.cuda_compute_streams.front().stream_id);
  const auto finalized = finalize_plan_artifact(state);
  ASSERT_TRUE(finalized) << finalized.error().message;
}

// REQ: COMPILE-009
// REQ: COMPILE-095
// REQ: COMPILE-040
TEST(CompilePhysicalPlanManifest, PublishesFiniteUniqueCanonicalPhysicalRequirements) {
  Graph graph{"physical-manifest"};
  const auto source = graph.op<ManifestTensorOperator>("source");
  constexpr std::array axes{TensorAxisBounds{.max_extent = 16U},
                            TensorAxisBounds{.max_extent = 16U}};
  TopicOptions options;
  options.tensor = TensorBoundaryContract{
      .representation = {.memory_kind = MemoryKind::kHost,
                         .dtype = DLDataType{kDLFloat, 32U, 1U},
                         .rank = 2U},
      .bounds = TensorBounds{.max_rank = 2U, .max_byte_span = 4096U, .axes = axes},
      .transfer =
          TensorTransferPolicy{
              .foreign_emit_policy = InterfaceTensorEmitPolicy::kInlineSmallHostThenCopyIfNeeded,
              .max_inline_bytes = 1024U,
          },
  };
  auto exported = graph.export_topic<Tensor>("holomq://manifest-tensor", options);
  graph.add_flow(source->tensor, exported.input());

  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.generic_output_loans().size(), 1U);
  ASSERT_EQ(plan.tensor_allocations().size(), 1U);
  ASSERT_EQ(plan.tensor_interface_pools().size(), 1U);
  EXPECT_TRUE(plan.tensor_device_resources().empty());
  ASSERT_EQ(plan.tensor_inline_records().size(), 1U);
  const auto& manifest = plan.profile_manifest();

  EXPECT_TRUE(std::is_sorted(manifest.required_capabilities.begin(),
                             manifest.required_capabilities.end(),
                             [](const auto& lhs, const auto& rhs) {
                               return std::pair{lhs.capability_id, lhs.version} <
                                      std::pair{rhs.capability_id, rhs.version};
                             }));
  EXPECT_TRUE(std::is_sorted(manifest.derived_capacities.begin(),
                             manifest.derived_capacities.end(),
                             [](const auto& lhs, const auto& rhs) {
                               return std::pair{lhs.object_id, lhs.capacity_id} <
                                      std::pair{rhs.object_id, rhs.capacity_id};
                             }));
  EXPECT_TRUE(std::is_sorted(manifest.materialization_obligations.begin(),
                             manifest.materialization_obligations.end(),
                             [](const auto& lhs, const auto& rhs) {
                               return std::pair{lhs.object_id, lhs.strategy_id} <
                                      std::pair{rhs.object_id, rhs.strategy_id};
                             }));
  EXPECT_EQ(std::adjacent_find(manifest.required_capabilities.begin(),
                               manifest.required_capabilities.end(),
                               [](const auto& lhs, const auto& rhs) {
                                 return lhs.capability_id == rhs.capability_id &&
                                        lhs.version == rhs.version;
                               }),
            manifest.required_capabilities.end());
  EXPECT_EQ(std::adjacent_find(manifest.derived_capacities.begin(),
                               manifest.derived_capacities.end(),
                               [](const auto& lhs, const auto& rhs) {
                                 return lhs.object_id == rhs.object_id &&
                                        lhs.capacity_id == rhs.capacity_id;
                               }),
            manifest.derived_capacities.end());
  EXPECT_TRUE(std::all_of(manifest.derived_capacities.begin(),
                          manifest.derived_capacities.end(),
                          [](const auto& capacity) {
                            return capacity.limit < std::numeric_limits<std::uint64_t>::max();
                          }));
  const auto has_capability = [&](std::string_view id) {
    return std::any_of(manifest.required_capabilities.begin(),
                       manifest.required_capabilities.end(),
                       [&](const auto& capability) { return capability.capability_id == id; });
  };
  EXPECT_TRUE(has_capability("holoscan.runtime.payload_edges"));
  EXPECT_TRUE(has_capability("holoscan.runtime.channel_service_attach"));
  EXPECT_TRUE(has_capability("holoscan.runtime.generic_output_loans"));
  EXPECT_TRUE(has_capability("holoscan.runtime.tensor_pool_materialization"));
  EXPECT_FALSE(has_capability("holoscan.runtime.tensor_device_stream_materialization"));
  EXPECT_TRUE(has_capability("holoscan.runtime.tensor_transfer_matrix"));
  EXPECT_TRUE(has_capability(kTensorInlineRecordCapabilityId));
  const auto& inline_record = plan.tensor_inline_records().front();
  EXPECT_TRUE(std::any_of(manifest.derived_capacities.begin(),
                          manifest.derived_capacities.end(),
                          [&](const auto& capacity) {
                            return capacity.object_id == inline_record.fixed_record_bucket_id &&
                                   capacity.capacity_id == "fixed_record_bytes" &&
                                   capacity.limit == inline_record.fixed_record_bucket_bytes;
                          }));
  EXPECT_TRUE(std::any_of(manifest.materialization_obligations.begin(),
                          manifest.materialization_obligations.end(),
                          [&](const auto& obligation) {
                            return obligation.object_id == inline_record.fixed_record_bucket_id &&
                                   obligation.strategy_id ==
                                       "holoscan.runtime.tensor_inline_record_bucket/v1" &&
                                   obligation.evidence_id == inline_record.capability_evidence_id;
                          }));
}

// REQ: COMPILE-040
TEST(CompilePhysicalPlanManifest, AggregatesMultipleImportedTensorGenerationScopes) {
  Graph graph{"multi-import-generation-manifest"};
  test::add_compile_test_default_clock(graph);
  const auto tensor_options = [] {
    TopicOptions options;
    options.tensor = TensorBoundaryContract{
        .representation = {.memory_kind = MemoryKind::kHost,
                           .dtype = DLDataType{kDLUInt, 8U, 1U},
                           .rank = 1U},
        .bounds = TensorBounds{.max_rank = 1U, .max_byte_span = 16U},
        .transfer = TensorTransferPolicy{.foreign_emit_policy =
                                             InterfaceTensorEmitPolicy::kRequireExportable},
    };
    return options;
  };

  auto first_options = tensor_options();
  first_options.endpoint_name = "shared-first";
  auto second_options = tensor_options();
  second_options.endpoint_name = "shared-second";
  auto third_options = tensor_options();
  third_options.endpoint_name = "distinct-third";
  auto first = graph.import_topic<Tensor>("holomq://shared-import", first_options);
  auto second = graph.import_topic<Tensor>("holomq://shared-import", second_options);
  auto third = graph.import_topic<Tensor>("holomq://distinct-import", third_options);
  const auto first_sink = graph.op<ManifestTensorSinkOperator>("first-sink");
  const auto second_sink = graph.op<ManifestTensorSinkOperator>("second-sink");
  const auto third_sink = graph.op<ManifestTensorSinkOperator>("third-sink");
  graph.add_flow(first.output(), first_sink->input);
  graph.add_flow(second.output(), second_sink->input);
  graph.add_flow(third.output(), third_sink->input);
  graph.partition("first-import").add(first_sink);
  graph.partition("second-import").add(second_sink);
  graph.partition("third-import").add(third_sink);

  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.imported_generation_capacities().size(), 3U);

  std::vector<std::string> services;
  std::vector<std::string> runtime_scopes;
  for (const auto& generation : plan.imported_generation_capacities()) {
    services.push_back(generation.endpoint_or_service_id);
    runtime_scopes.push_back(generation.runtime_scope_id);
    EXPECT_EQ(generation.runtime_max_generations, 3U);
  }
  std::sort(services.begin(), services.end());
  services.erase(std::unique(services.begin(), services.end()), services.end());
  std::sort(runtime_scopes.begin(), runtime_scopes.end());
  runtime_scopes.erase(std::unique(runtime_scopes.begin(), runtime_scopes.end()),
                       runtime_scopes.end());
  EXPECT_EQ(services.size(), 3U);
  ASSERT_EQ(runtime_scopes.size(), 1U);

  const auto& capacities = plan.profile_manifest().derived_capacities;
  const auto capacity_count = [&](std::string_view id) {
    return std::count_if(capacities.begin(), capacities.end(), [&](const auto& capacity) {
      return capacity.capacity_id == id;
    });
  };
  EXPECT_EQ(capacity_count("endpoint_generations"), 3);
  EXPECT_EQ(capacity_count("candidate_generation_reservations"), 3);
  EXPECT_EQ(capacity_count("generation_release_records"), 3);
  EXPECT_EQ(capacity_count("generation_quarantine_records"), 3);
  EXPECT_EQ(capacity_count("service_generations"), 3);
  EXPECT_EQ(capacity_count("runtime_generations"), 1);
  const auto runtime_capacity =
      std::find_if(capacities.begin(), capacities.end(), [&](const auto& capacity) {
        return capacity.object_id == runtime_scopes.front() &&
               capacity.capacity_id == "runtime_generations";
      });
  ASSERT_NE(runtime_capacity, capacities.end());
  EXPECT_EQ(runtime_capacity->limit, 3U);

  const auto& materializations = plan.profile_manifest().materialization_obligations;
  EXPECT_EQ(std::count_if(materializations.begin(),
                          materializations.end(),
                          [](const auto& item) {
                            return item.strategy_id == "holoscan.runtime.channel_service_attach/v1";
                          }),
            3);
}

// REQ: COMPILE-040
TEST(CompilePhysicalPlanManifest, DeduplicatesSharedServiceCapacityProjection) {
  ExecutionPlanState state;
  state.channel_services = {
      ChannelServiceStaticPlanRecord{
          .topic = ExternalTopicId{1U},
          .service_id = "shared-service",
          .max_subscriber_queue_depth = 4U,
          .max_subscriber_borrowed_samples = 3U,
      },
      ChannelServiceStaticPlanRecord{
          .topic = ExternalTopicId{2U},
          .service_id = "shared-service",
          .max_subscriber_queue_depth = 4U,
          .max_subscriber_borrowed_samples = 3U,
      },
  };
  state.imported_generation_capacities = {
      ImportedGenerationCapacityPlanRecord{
          .topic = ExternalTopicId{1U},
          .runtime_scope_id = "runtime-scope",
          .endpoint_or_service_id = "shared-service",
          .endpoint_max_generations = 1U,
          .service_max_generations = 2U,
          .runtime_max_generations = 2U,
          .release_record_capacity = 1U,
          .quarantine_record_capacity = 1U,
      },
      ImportedGenerationCapacityPlanRecord{
          .topic = ExternalTopicId{2U},
          .runtime_scope_id = "runtime-scope",
          .endpoint_or_service_id = "shared-service",
          .endpoint_max_generations = 2U,
          .service_max_generations = 2U,
          .runtime_max_generations = 2U,
          .candidate_reservation_capacity = 1U,
          .release_record_capacity = 2U,
          .quarantine_record_capacity = 2U,
      },
  };
  auto& manifest = state.profile_manifest;
  manifest.disposition = ProfileAdmissionDisposition::kAccepted;
  manifest.requested_profile = Profile::kStandard;
  manifest.effective_graph_profile = Profile::kStandard;
  manifest.profile_policy_id = profile_policy_id_v1();
  manifest.plan_schema_version = "holoscan.execution_plan.private.v8";
  manifest.runtime_abi_version = "holoscan.runtime_abi.v1";
  manifest.target_triple = std::string(compile_target_triple());

  const auto finalized = finalize_profile_manifest(CompileOptions{}, state);
  ASSERT_TRUE(finalized) << finalized.error().message;
  const auto& capacities = state.profile_manifest.derived_capacities;
  const auto matching_capacities = [&](std::string_view object_id, std::string_view capacity_id) {
    return std::count_if(capacities.begin(), capacities.end(), [&](const auto& capacity) {
      return capacity.object_id == object_id && capacity.capacity_id == capacity_id;
    });
  };
  const auto capacity_limit = [&](std::string_view object_id, std::string_view capacity_id) {
    const auto found =
        std::find_if(capacities.begin(), capacities.end(), [&](const auto& capacity) {
          return capacity.object_id == object_id && capacity.capacity_id == capacity_id;
        });
    EXPECT_NE(found, capacities.end());
    return found == capacities.end() ? 0U : found->limit;
  };

  EXPECT_EQ(matching_capacities("shared-service", "max_subscriber_queue_depth"), 1);
  EXPECT_EQ(capacity_limit("shared-service", "max_subscriber_queue_depth"), 4U);
  EXPECT_EQ(capacity_limit("shared-service", "max_subscriber_borrowed_samples"), 3U);
  EXPECT_EQ(capacity_limit("shared-service", "publisher_history_depth"), 0U);
  EXPECT_EQ(matching_capacities("shared-service", "service_generations"), 1);
  EXPECT_EQ(capacity_limit("shared-service", "service_generations"), 2U);
  EXPECT_EQ(matching_capacities("runtime-scope", "runtime_generations"), 1);
  EXPECT_EQ(capacity_limit("runtime-scope", "runtime_generations"), 2U);
  EXPECT_EQ(matching_capacities("imported-generation-endpoint:1", "endpoint_generations"), 1);
  EXPECT_EQ(matching_capacities("imported-generation-endpoint:2", "endpoint_generations"), 1);

  const auto& materializations = state.profile_manifest.materialization_obligations;
  EXPECT_EQ(std::count_if(materializations.begin(),
                          materializations.end(),
                          [](const auto& item) {
                            return item.object_id == "shared-service" &&
                                   item.strategy_id == "holoscan.runtime.channel_service_attach/v1";
                          }),
            1);
}

}  // namespace
}  // namespace holoscan::detail
// NOLINTEND(bugprone-unchecked-optional-access)
