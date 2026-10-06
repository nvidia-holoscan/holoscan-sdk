/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <array>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <limits>
#include <memory>
#include <ranges>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <holoscan/core/predicate.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "artifact_test_support.hpp"
#include "compiler/compile_internal.hpp"
#include "compiler/pool_capacity_plan_lowering.hpp"
#include "compiler/process_placement_internal.hpp"
#include "compiler/rate_plan_lowering.hpp"
#include "core/diagnostic_registry.hpp"
#include "physical_plan_test_support.hpp"

namespace holoscan::detail {
namespace {

using namespace physical_plan_test;
using namespace std::chrono_literals;

class FixedRateSource final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.output(output, "output").nominal_rate(Rate::hz(60U));
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Output<std::uint32_t> output;
};

class ClockRateSink final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override { spec.input(input, "input"); }

  Contract contract() const override {
    ++contract_calls;
    Contract result;
    result.trigger(OnClock{.period = 1s}).input(input, Exact{}).on_missing(input, Skip{});
    return result;
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Input<std::uint32_t> input;
  static inline std::uint32_t contract_calls{};
};

class ClockDrainAllRateSink final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override { spec.input(input, "input"); }

  Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 1s}).input(input, DrainAll{});
    return result;
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Input<std::uint32_t> input;
};

class ClockPredicateRateSink final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override { spec.input(input, "input"); }

  Contract contract() const override {
    auto evaluator = predicate("clock-rate-predicate",
                               [](const PredicateEvaluationContext&) noexcept { return true; });
    Contract result;
    result
        .trigger(OnPredicate{
            .evaluate_on = OnClock{.period = 1s},
            .predicate = std::move(evaluator),
            .firing = PredicateFiring::kOnEveryTrueEvaluation,
        })
        .input(input, Exact{})
        .on_missing(input, Skip{});
    return result;
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Input<std::uint32_t> input;
};

class WindowRateSource final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.output(anchor, "anchor").nominal_rate(Rate::hz(1'000U));
    spec.output(samples, "samples").nominal_rate(Rate::hz(1'000U));
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Output<std::uint32_t> anchor;
  Output<std::uint32_t> samples;
};

class WindowRateSink final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(anchor, "anchor");
    spec.input(samples, "samples");
  }

  Contract contract() const override {
    Contract result;
    result.trigger(OnWindow{.anchor = anchor, .max_wait = 2ms, .max_pending_windows = 2U})
        .input(anchor, Exact{})
        .input(samples, BracketedWindow{.lookbehind = 1U, .lookahead = 1U, .max_samples = 3U})
        .on_missing(samples, Skip{});
    return result;
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Input<std::uint32_t> anchor;
  Input<std::uint32_t> samples;
};

class LifecycleRateTensorSource final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.output(output, "output")
        .lifecycle_nominal_rate(LifecycleStage::kDiscover)
        .max_nominal_rate(Rate::hz(120U))
        .produces_tensor(TensorOutputSpec{
            .representation = {.memory_kind = MemoryKind::kHost,
                               .dtype = DLDataType{kDLUInt, 8U, 1U},
                               .rank = 1U,
                               .strides = TensorStridePolicy::kContiguousOnly},
            .bounds = tensor_bounds(64U),
            .storage = TensorOutputStorage::kRuntimePool,
        });
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Output<Tensor> output;
};

class LifecycleCandidateTensorSource final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.output(output, "output")
        .lifecycle_nominal_rate(LifecycleStage::kDiscover)
        .produces_tensor(TensorOutputSpec{
            .representation = {.memory_kind = MemoryKind::kHost,
                               .dtype = DLDataType{kDLUInt, 8U, 1U},
                               .rank = 1U,
                               .strides = TensorStridePolicy::kContiguousOnly},
            .bounds = tensor_bounds(64U),
            .storage = TensorOutputStorage::kRuntimePool,
        });
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Output<Tensor> output;
};

class LifecycleCandidateScalarSource final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.output(output, "output").lifecycle_nominal_rate(LifecycleStage::kDiscover);
  }

  Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 1ms});
    return result;
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Output<std::uint32_t> output;
};

class EachRateSink final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override { spec.input(input, "input"); }

  Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input});
    return result;
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Input<std::uint32_t> input;
};

class LifecycleOnEachDriverSource final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.output(output, "output")
        .lifecycle_nominal_rate(LifecycleStage::kDiscover)
        .max_nominal_rate(Rate::hz(120U));
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Output<std::uint32_t> output;
};

class FixedRateSecondarySource final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.output(output, "output").nominal_rate(Rate::hz(1000U));
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Output<std::uint32_t> output;
};

class LifecycleOnEachSink final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(driver, "driver");
    spec.input(secondary, "secondary");
  }

  Contract contract() const override {
    Contract result;
    result.trigger(OnEach{driver});
    return result;
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Input<std::uint32_t> driver;
  Input<std::uint32_t> secondary;
};

class ClockTensorRateSink final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override { spec.input(input, "input"); }

  Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 1s}).input(input, Exact{}).on_missing(input, Skip{});
    return result;
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Input<Tensor> input;
};

[[nodiscard]] GraphSnapshotData fixed_rate_graph(std::uint64_t numerator_hz = 60U,
                                                 std::uint64_t denominator_hz = 1U) {
  GraphSnapshotData snapshot;
  snapshot.generation = GraphGeneration{7U};
  snapshot.vertices = {vertex(1U, "source"), vertex(2U, "sink")};
  snapshot.ports = {
      output_port(1U, 1U, "out", scalar_payload()),
      input_port(2U, 2U, "in", scalar_payload()),
  };
  snapshot.ports[0].setup.rate_kind = PortRateKind::kFixedNominal;
  snapshot.ports[0].setup.declared_nominal_rate = Rate::hz(numerator_hz, denominator_hz);
  snapshot.ports[1].setup.queue_depth.reset();
  snapshot.edges = {edge(1U, 1U, 2U)};
  snapshot.edges[0].add_flow_options.queue_depth.reset();
  snapshot.edges[0].add_flow_options.max_inflight_samples.reset();
  return snapshot;
}

[[nodiscard]] TemporalTriggerPlanRecord one_second_clock_trigger() {
  return TemporalTriggerPlanRecord{
      .owner = OperatorRef{.id = 2U, .plan_generation = 41U},
      .kind = TemporalRecordTriggerKind::kClock,
      .trigger_clock = ClockId{1U},
      .period_ns = 1'000'000'000,
  };
}

[[nodiscard]] ProcessPlacementBinding split_rate_source_sink_placement() {
  constexpr std::array source_selector{
      GraphObjectSelector{.graph_path = "source"},
  };
  constexpr std::array sink_selector{
      GraphObjectSelector{.graph_path = "sink"},
  };
  ProcessPlacementBinding placement;
  placement
      .bind_process_role("source-role",
                         ProcessPlacement{
                             .members = source_selector,
                             .launch_group = "source-process",
                         })
      .bind_process_role("sink-role",
                         ProcessPlacement{
                             .members = sink_selector,
                             .launch_group = "sink-process",
                         });
  return placement;
}

// REQ: COMPILE-069
TEST(CompileRatePlan, FixedRateAndClockHorizonDeriveExactDefaultHeadroomDepth) {
  const auto snapshot = fixed_rate_graph(120U, 2U);
  const std::array triggers{one_second_clock_trigger()};
  const auto profiles = standard_profiles(snapshot.vertices.size());

  const auto result = lower_rate_plan(
      snapshot, RateRatio{2U, 1U}, triggers, {}, {}, profiles, PlanGenerationId{41U});

  ASSERT_TRUE(result.errors.empty());
  ASSERT_EQ(result.port_rates.size(), 2U);
  EXPECT_EQ(result.port_rates[0].declared_nominal_rate, Rate::hz(60U));
  ASSERT_EQ(result.queue_sizings.size(), 1U);
  const auto& sizing = result.queue_sizings.front();
  EXPECT_EQ(sizing.edge, EdgeId{1U});
  EXPECT_EQ(sizing.origin, QueueDepthOrigin::kRateAndContractDerived);
  EXPECT_EQ(sizing.nominal_arrivals, 60U);
  EXPECT_EQ(sizing.rate_recommendation, 120U);
  EXPECT_EQ(sizing.effective_logical_depth, 120U);
}

// REQ: COMPILE-069
// REQ: OPERATOR-048
TEST(CompileRatePlan, ExplicitDepthRemainsExactBelowAdvisoryRecommendation) {
  auto snapshot = fixed_rate_graph();
  snapshot.edges[0].graph_override.queue_depth = 7U;
  const std::array triggers{one_second_clock_trigger()};
  const auto profiles = standard_profiles(snapshot.vertices.size());

  const auto result = lower_rate_plan(
      snapshot, RateRatio{2U, 1U}, triggers, {}, {}, profiles, PlanGenerationId{41U});

  ASSERT_TRUE(result.errors.empty());
  ASSERT_EQ(result.queue_sizings.size(), 1U);
  EXPECT_EQ(result.queue_sizings.front().origin, QueueDepthOrigin::kGraphEdgeOverride);
  EXPECT_EQ(result.queue_sizings.front().effective_logical_depth, 7U);
  EXPECT_EQ(result.queue_sizings.front().rate_recommendation, 120U);
  EXPECT_TRUE(std::ranges::any_of(result.warnings, [](const auto& diagnostic) {
    return diagnostic.token == "RATE_QUEUE_DEPTH_BELOW_RECOMMENDATION";
  }));
}

// REQ: COMPILE-069
// Task-041 metadata must not retroactively reject a legacy explicit queue that has no rate
// declaration. Existing Contract lowering remains authoritative for that graph; the additional
// rate-plan hard-floor evidence becomes an admission constraint only when automatic sizing or a
// rate declaration opts the edge into Task-041 behavior.
TEST(CompileRatePlan, LegacyExplicitDepthWithoutRateMetadataPreservesSourceBehavior) {
  auto snapshot = fixed_rate_graph();
  snapshot.ports[0].setup.rate_kind = PortRateKind::kUnspecified;
  snapshot.ports[0].setup.declared_nominal_rate.reset();
  snapshot.edges[0].graph_override.queue_depth = 1U;
  const std::array triggers{one_second_clock_trigger()};
  const std::array selections{InputSelectionPlanRecord{
      .input = PortId{2U},
      .selector = InputSelectorKind::kExact,
      .cardinality = InputSelectionCardinality::kSingleSample,
      .max_selected_samples = 1U,
      .max_selected_samples_origin = SelectedSampleBoundOrigin::kSelectorMaxSamples,
      .batch_selection_mode = BatchSelectionMode::kConsumeSelected,
      .cache_capacity = 1U,
  }};
  const auto profiles = standard_profiles(snapshot.vertices.size());

  const auto result = lower_rate_plan(
      snapshot, RateRatio{2U, 1U}, triggers, selections, {}, profiles, PlanGenerationId{41U});

  ASSERT_TRUE(result.errors.empty());
  ASSERT_EQ(result.queue_sizings.size(), 1U);
  EXPECT_EQ(result.queue_sizings.front().origin, QueueDepthOrigin::kGraphEdgeOverride);
  EXPECT_EQ(result.queue_sizings.front().effective_logical_depth, 1U);
}

// REQ: COMPILE-069
TEST(CompileRatePlan, FollowsResolvesFromConnectedInputWithoutSchedulingSideEffects) {
  GraphSnapshotData snapshot;
  snapshot.generation = GraphGeneration{7U};
  snapshot.vertices = {vertex(1U, "source"), vertex(2U, "relay"), vertex(3U, "sink")};
  snapshot.ports = {
      output_port(1U, 1U, "out", scalar_payload()),
      input_port(2U, 2U, "in", scalar_payload()),
      output_port(3U, 2U, "out", scalar_payload()),
      input_port(4U, 3U, "in", scalar_payload()),
  };
  snapshot.ports[0].setup.rate_kind = PortRateKind::kFixedNominal;
  snapshot.ports[0].setup.declared_nominal_rate = Rate::hz(1'000U);
  snapshot.ports[1].setup.queue_depth.reset();
  snapshot.ports[2].setup.rate_kind = PortRateKind::kFollowsInput;
  snapshot.ports[2].setup.follows_input = GraphPortId{2U};
  snapshot.ports[2].setup.follows_ratio = RateRatio{1U, 4U};
  snapshot.ports[3].setup.queue_depth.reset();
  snapshot.edges = {edge(1U, 1U, 2U), edge(2U, 3U, 4U)};
  for (auto& item : snapshot.edges) {
    item.add_flow_options.queue_depth.reset();
    item.add_flow_options.max_inflight_samples.reset();
  }
  const auto profiles = standard_profiles(snapshot.vertices.size());

  const auto result =
      lower_rate_plan(snapshot, RateRatio{2U, 1U}, {}, {}, {}, profiles, PlanGenerationId{41U});

  ASSERT_TRUE(result.errors.empty());
  const auto relay_output =
      std::ranges::find(result.resolved_rates, PortId{3U}, &ResolvedPortRate::port);
  ASSERT_NE(relay_output, result.resolved_rates.end());
  EXPECT_EQ(relay_output->resolved_rate, Rate::hz(250U));
  EXPECT_EQ(relay_output->origin, ResolvedRateOrigin::kFollowsExpression);
  EXPECT_EQ(relay_output->immediate_provenance_port, PortId{2U});
  EXPECT_EQ(relay_output->root_provenance_port, PortId{1U});
}

// A follows declaration is a cold dataflow relation. A cycle must be rejected instead of being
// treated as an absent upstream and accidentally admitting an input expectation fallback.
// REQ: COMPILE-069
TEST(CompileRatePlan, FollowsDependencyCycleFailsClosedWithoutFallback) {
  GraphSnapshotData snapshot;
  snapshot.generation = GraphGeneration{7U};
  snapshot.vertices = {vertex(1U, "first"), vertex(2U, "second")};
  snapshot.ports = {
      output_port(1U, 1U, "first-out", scalar_payload()),
      input_port(2U, 1U, "first-in", scalar_payload()),
      output_port(3U, 2U, "second-out", scalar_payload()),
      input_port(4U, 2U, "second-in", scalar_payload()),
  };
  snapshot.ports[0].setup.rate_kind = PortRateKind::kFollowsInput;
  snapshot.ports[0].setup.follows_input = GraphPortId{2U};
  snapshot.ports[0].setup.follows_ratio = RateRatio{1U, 1U};
  snapshot.ports[2].setup.rate_kind = PortRateKind::kFollowsInput;
  snapshot.ports[2].setup.follows_input = GraphPortId{4U};
  snapshot.ports[2].setup.follows_ratio = RateRatio{1U, 1U};
  snapshot.ports[1].setup.rate_kind = PortRateKind::kExpectedInput;
  snapshot.ports[1].setup.declared_expected_rate = Rate::hz(60U);
  snapshot.edges = {edge(1U, 1U, 4U), edge(2U, 3U, 2U)};
  const auto profiles = standard_profiles(snapshot.vertices.size());

  const auto result =
      lower_rate_plan(snapshot, RateRatio{2U, 1U}, {}, {}, {}, profiles, PlanGenerationId{41U});

  EXPECT_TRUE(std::ranges::any_of(result.errors, [](const auto& diagnostic) {
    return diagnostic.token == "RATE_DEPENDENCY_CYCLE";
  }));
  ASSERT_EQ(result.resolved_rates.size(), 4U);
  EXPECT_FALSE(result.resolved_rates[0].resolved_rate.has_value());
  EXPECT_FALSE(result.resolved_rates[2].resolved_rate.has_value());
  EXPECT_FALSE(result.resolved_rates[1].resolved_rate.has_value());
  EXPECT_FALSE(result.resolved_rates[1].genuinely_absent);
}

// A fractional chain records the direct predecessor independently from the declaration root.
// Two outputs may therefore follow the same local input without aliasing their provenance.
// REQ: COMPILE-069
TEST(CompileRatePlan, FractionalFollowsChainPreservesImmediateAndRootProvenancePerOutput) {
  GraphSnapshotData snapshot;
  snapshot.generation = GraphGeneration{7U};
  snapshot.vertices = {vertex(1U, "source"), vertex(2U, "first-relay"), vertex(3U, "second-relay")};
  snapshot.ports = {
      output_port(1U, 1U, "source-out", scalar_payload()),
      input_port(2U, 2U, "first-in", scalar_payload()),
      output_port(3U, 2U, "first-out", scalar_payload()),
      input_port(4U, 3U, "second-in", scalar_payload()),
      output_port(5U, 3U, "slow-out", scalar_payload()),
      output_port(6U, 3U, "fast-out", scalar_payload()),
  };
  snapshot.ports[0].setup.rate_kind = PortRateKind::kFixedNominal;
  snapshot.ports[0].setup.declared_nominal_rate = Rate::hz(120U);
  snapshot.ports[2].setup.rate_kind = PortRateKind::kFollowsInput;
  snapshot.ports[2].setup.follows_input = GraphPortId{2U};
  snapshot.ports[2].setup.follows_ratio = RateRatio{1U, 2U};
  snapshot.ports[4].setup.rate_kind = PortRateKind::kFollowsInput;
  snapshot.ports[4].setup.follows_input = GraphPortId{4U};
  snapshot.ports[4].setup.follows_ratio = RateRatio{2U, 3U};
  snapshot.ports[5].setup.rate_kind = PortRateKind::kFollowsInput;
  snapshot.ports[5].setup.follows_input = GraphPortId{4U};
  snapshot.ports[5].setup.follows_ratio = RateRatio{3U, 2U};
  snapshot.edges = {edge(1U, 1U, 2U), edge(2U, 3U, 4U)};
  const auto profiles = standard_profiles(snapshot.vertices.size());

  const auto result =
      lower_rate_plan(snapshot, RateRatio{2U, 1U}, {}, {}, {}, profiles, PlanGenerationId{41U});

  ASSERT_TRUE(result.errors.empty());
  ASSERT_EQ(result.resolved_rates.size(), 6U);
  const auto expect_follow = [&](PortId port, Rate rate, PortId immediate) {
    const auto found = std::ranges::find(result.resolved_rates, port, &ResolvedPortRate::port);
    ASSERT_NE(found, result.resolved_rates.end());
    EXPECT_EQ(found->resolved_rate, rate);
    EXPECT_EQ(found->origin, ResolvedRateOrigin::kFollowsExpression);
    EXPECT_EQ(found->immediate_provenance_port, immediate);
    EXPECT_EQ(found->root_provenance_port, PortId{1U});
  };
  expect_follow(PortId{3U}, Rate::hz(60U), PortId{2U});
  expect_follow(PortId{5U}, Rate::hz(40U), PortId{4U});
  expect_follow(PortId{6U}, Rate::hz(90U), PortId{4U});
}

// A lifecycle-selected root is the restart authority for every owner whose rate is propagated
// from that root. This compiler-private relation widens restart membership without inventing a
// public lifecycle callback-order dependency.
// REQ: COMPILE-069
TEST(CompileRatePlan, LifecycleRootRecordsTransitiveFollowsAndFanoutRestartDependents) {
  GraphSnapshotData snapshot;
  snapshot.generation = GraphGeneration{7U};
  snapshot.vertices = {vertex(1U, "source"),
                       vertex(2U, "relay"),
                       vertex(3U, "left"),
                       vertex(4U, "right"),
                       vertex(5U, "unrelated")};
  snapshot.ports = {
      output_port(1U, 1U, "out", scalar_payload()),
      input_port(2U, 2U, "in", scalar_payload()),
      output_port(3U, 2U, "followed", scalar_payload()),
      input_port(4U, 3U, "in", scalar_payload()),
      input_port(5U, 4U, "in", scalar_payload()),
      output_port(6U, 5U, "fixed", scalar_payload()),
  };
  snapshot.ports[0].setup.rate_kind = PortRateKind::kLifecycleNominal;
  snapshot.ports[0].setup.lifecycle_binding_stage = LifecycleStage::kDiscover;
  snapshot.ports[0].setup.declared_max_nominal_rate = Rate::hz(120U);
  snapshot.ports[2].setup.rate_kind = PortRateKind::kFollowsInput;
  snapshot.ports[2].setup.follows_input = GraphPortId{2U};
  snapshot.ports[2].setup.follows_ratio = RateRatio{1U, 2U};
  snapshot.ports[5].setup.rate_kind = PortRateKind::kFixedNominal;
  snapshot.ports[5].setup.declared_nominal_rate = Rate::hz(30U);
  snapshot.edges = {
      edge(1U, 1U, 2U),
      edge(2U, 3U, 4U),
      edge(3U, 3U, 5U),
  };
  for (auto& item : snapshot.edges) {
    item.add_flow_options.queue_depth.reset();
    item.add_flow_options.max_inflight_samples.reset();
  }
  const auto profiles = standard_profiles(snapshot.vertices.size());

  const auto result =
      lower_rate_plan(snapshot, RateRatio{2U, 1U}, {}, {}, {}, profiles, PlanGenerationId{41U});

  ASSERT_TRUE(result.errors.empty());
  const std::vector expected{
      RateRestartDependency{.source_owner = VertexId{1U}, .dependent_owner = VertexId{2U}},
      RateRestartDependency{.source_owner = VertexId{1U}, .dependent_owner = VertexId{3U}},
      RateRestartDependency{.source_owner = VertexId{1U}, .dependent_owner = VertexId{4U}},
  };
  EXPECT_EQ(result.restart_dependencies, expected);
}

// Process placement is already frozen when rate lowering runs. Lifecycle-selected values cannot
// cross that boundary without a distributed candidate/barrier/restart protocol, even when an
// explicit depth or finite maximum makes the storage geometry compile concrete.
// REQ: COMPILE-071
TEST(CompileRatePlan, LifecycleDirectCrossProcessRejectsEveryDepthAuthority) {
  const std::array process_groups{1U, 2U};
  const std::array triggers{one_second_clock_trigger()};

  const auto expect_rejected = [&](GraphSnapshotData snapshot, std::string_view scenario) {
    SCOPED_TRACE(scenario);
    snapshot.ports[0].setup.rate_kind = PortRateKind::kLifecycleNominal;
    snapshot.ports[0].setup.declared_nominal_rate.reset();
    snapshot.ports[0].setup.lifecycle_binding_stage = LifecycleStage::kDiscover;
    const auto profiles = standard_profiles(snapshot.vertices.size());
    const auto result = lower_rate_plan(snapshot,
                                        RateRatio{2U, 1U},
                                        triggers,
                                        {},
                                        {},
                                        profiles,
                                        {},
                                        PlanGenerationId{41U},
                                        process_groups);
    EXPECT_TRUE(std::ranges::any_of(result.errors, [](const auto& diagnostic) {
      return diagnostic.token == "RATE_LIFECYCLE_CROSS_PROCESS_UNSUPPORTED";
    }));
  };

  expect_rejected(fixed_rate_graph(), "omitted depth and no maximum");

  auto explicit_depth = fixed_rate_graph();
  explicit_depth.edges[0].graph_override.queue_depth = 8U;
  expect_rejected(std::move(explicit_depth), "explicit depth");

  auto finite_maximum = fixed_rate_graph();
  finite_maximum.ports[0].setup.declared_max_nominal_rate = Rate::hz(120U);
  expect_rejected(std::move(finite_maximum), "finite maximum");
}

// A local source-to-relay hop does not make a later follows-derived cross-process hop local.
// REQ: COMPILE-071
TEST(CompileRatePlan, LifecycleFollowsCrossProcessDependencyRejects) {
  GraphSnapshotData snapshot;
  snapshot.generation = GraphGeneration{7U};
  snapshot.vertices = {vertex(1U, "source"), vertex(2U, "relay"), vertex(3U, "sink")};
  snapshot.ports = {
      output_port(1U, 1U, "source-out", scalar_payload()),
      input_port(2U, 2U, "relay-in", scalar_payload()),
      output_port(3U, 2U, "relay-out", scalar_payload()),
      input_port(4U, 3U, "sink-in", scalar_payload()),
  };
  snapshot.ports[0].setup.rate_kind = PortRateKind::kLifecycleNominal;
  snapshot.ports[0].setup.lifecycle_binding_stage = LifecycleStage::kDiscover;
  snapshot.ports[0].setup.declared_max_nominal_rate = Rate::hz(120U);
  snapshot.ports[2].setup.rate_kind = PortRateKind::kFollowsInput;
  snapshot.ports[2].setup.follows_input = GraphPortId{2U};
  snapshot.ports[2].setup.follows_ratio = RateRatio{1U, 2U};
  snapshot.edges = {edge(1U, 1U, 2U), edge(2U, 3U, 4U)};
  const auto profiles = standard_profiles(snapshot.vertices.size());
  const std::array process_groups{1U, 1U, 2U};

  const auto result = lower_rate_plan(
      snapshot, RateRatio{2U, 1U}, {}, {}, {}, profiles, {}, PlanGenerationId{41U}, process_groups);

  EXPECT_TRUE(std::ranges::any_of(result.errors, [](const auto& diagnostic) {
    return diagnostic.token == "RATE_LIFECYCLE_CROSS_PROCESS_UNSUPPORTED";
  }));
}

// The lifecycle driver and consumer are local, but the fixed secondary producer's queue horizon
// still depends on that lifecycle root. That inverse-rate dependency must not cross groups.
// REQ: COMPILE-071
TEST(CompileRatePlan, LifecycleInverseRateDriverCrossProcessDependencyRejects) {
  GraphSnapshotData snapshot;
  snapshot.generation = GraphGeneration{7U};
  snapshot.vertices = {vertex(1U, "camera-source"), vertex(2U, "imu-source"), vertex(3U, "fusion")};
  snapshot.ports = {
      output_port(1U, 1U, "camera-out", scalar_payload()),
      output_port(2U, 2U, "imu-out", scalar_payload()),
      input_port(3U, 3U, "camera", scalar_payload()),
      input_port(4U, 3U, "imu", scalar_payload()),
  };
  snapshot.ports[0].setup.rate_kind = PortRateKind::kLifecycleNominal;
  snapshot.ports[0].setup.lifecycle_binding_stage = LifecycleStage::kDiscover;
  snapshot.ports[0].setup.declared_max_nominal_rate = Rate::hz(120U);
  snapshot.ports[1].setup.rate_kind = PortRateKind::kFixedNominal;
  snapshot.ports[1].setup.declared_nominal_rate = Rate::hz(1000U);
  snapshot.edges = {edge(1U, 1U, 3U), edge(2U, 2U, 4U)};
  const std::array triggers{TemporalTriggerPlanRecord{
      .owner = OperatorRef{.id = 3U, .plan_generation = 41U},
      .kind = TemporalRecordTriggerKind::kEach,
      .trigger_input = PortId{3U},
  }};
  const auto profiles = standard_profiles(snapshot.vertices.size());
  const std::array process_groups{1U, 2U, 1U};

  const auto result = lower_rate_plan(snapshot,
                                      RateRatio{2U, 1U},
                                      triggers,
                                      {},
                                      {},
                                      profiles,
                                      {},
                                      PlanGenerationId{41U},
                                      process_groups);

  EXPECT_TRUE(std::ranges::any_of(result.errors, [](const auto& diagnostic) {
    return diagnostic.token == "RATE_LIFECYCLE_CROSS_PROCESS_UNSUPPORTED";
  }));
}

// Fixed rates need no distributed candidate exchange, and a disconnected lifecycle root in a
// remote group is not part of the local fixed-rate edge's dependency closure.
// REQ: COMPILE-071
TEST(CompileRatePlan, FixedCrossProcessAndDisconnectedLifecycleRootRemainAdmitted) {
  auto snapshot = fixed_rate_graph();
  snapshot.vertices.push_back(vertex(3U, "remote-lifecycle-source"));
  snapshot.ports.push_back(output_port(3U, 3U, "remote-out", scalar_payload()));
  snapshot.ports.back().setup.rate_kind = PortRateKind::kLifecycleNominal;
  snapshot.ports.back().setup.lifecycle_binding_stage = LifecycleStage::kDiscover;
  const std::array triggers{one_second_clock_trigger()};
  const auto profiles = standard_profiles(snapshot.vertices.size());
  const std::array process_groups{1U, 2U, 3U};

  const auto result = lower_rate_plan(snapshot,
                                      RateRatio{2U, 1U},
                                      triggers,
                                      {},
                                      {},
                                      profiles,
                                      {},
                                      PlanGenerationId{41U},
                                      process_groups);

  EXPECT_TRUE(result.errors.empty());
  EXPECT_TRUE(result.restart_dependencies.empty());
  ASSERT_EQ(result.queue_sizings.size(), 1U);
  EXPECT_EQ(result.queue_sizings.front().effective_logical_depth, 120U);
}

// The production compiler must feed its already-frozen process cover into rate admission rather
// than relying on tests or worker reconstruction to rediscover the boundary.
// REQ: COMPILE-071
TEST(CompileRatePlan, ProductionCompileRejectsLifecycleDerivedCrossProcessDependency) {
  Graph graph{"lifecycle-rate-cross-process"};
  test::add_compile_test_default_clock(graph);
  auto source = graph.op<LifecycleCandidateScalarSource>("source");
  auto sink = graph.op<EachRateSink>("sink");
  graph.add_flow(source->output, sink->input);

  const ExecutionPlan plan =
      compile_with_process_placement(graph, CompileOptions{}, split_rate_source_sink_placement());

  EXPECT_FALSE(plan.ok());
  EXPECT_TRUE(std::ranges::any_of(plan.diagnostics(), [](const auto& diagnostic) {
    return diagnostic.token == "RATE_LIFECYCLE_CROSS_PROCESS_UNSUPPORTED" &&
           static_cast<bool>(diagnostic.source.vertex) &&
           diagnostic.required == "lifecycle-derived rate dependency within one process group" &&
           diagnostic.provided.find("source_group=") != std::string::npos &&
           diagnostic.provided.find(",dependent_group=") != std::string::npos;
  })) << plan.json();
}

// A fixed rate is immutable plan evidence and needs no distributed lifecycle transaction.
// REQ: COMPILE-071
TEST(CompileRatePlan, ProductionCompilePreservesFixedRateCrossProcessEdge) {
  Graph graph{"fixed-rate-cross-process"};
  test::add_compile_test_default_clock(graph);
  auto source = graph.op<FixedRateSource>("source");
  auto sink = graph.op<EachRateSink>("sink");
  graph.add_flow(source->output, sink->input);

  const ExecutionPlan plan =
      compile_with_process_placement(graph, CompileOptions{}, split_rate_source_sink_placement());

  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.payload_edges().size(), 1U);
  EXPECT_EQ(plan.payload_edges().front().locality, ChannelLocalityKind::kCrossProcess);
  EXPECT_FALSE(std::ranges::any_of(plan.diagnostics(), [](const auto& diagnostic) {
    return diagnostic.token == "RATE_LIFECYCLE_CROSS_PROCESS_UNSUPPORTED";
  }));
}

// REQ: COMPILE-069
TEST(CompileRatePlan, InvalidCompileHeadroomFailsClosed) {
  const auto snapshot = fixed_rate_graph();
  const auto profiles = standard_profiles(snapshot.vertices.size());

  const auto zero =
      lower_rate_plan(snapshot, RateRatio{0U, 1U}, {}, {}, {}, profiles, PlanGenerationId{41U});
  const auto below_one =
      lower_rate_plan(snapshot, RateRatio{1U, 2U}, {}, {}, {}, profiles, PlanGenerationId{41U});

  EXPECT_TRUE(std::ranges::any_of(zero.errors, [](const auto& diagnostic) {
    return diagnostic.token == "RATE_HEADROOM_INVALID";
  }));
  EXPECT_TRUE(std::ranges::any_of(below_one.errors, [](const auto& diagnostic) {
    return diagnostic.token == "RATE_HEADROOM_INVALID";
  }));
  for (const auto* rejected : {&zero, &below_one}) {
    ASSERT_EQ(rejected->port_rates.size(), snapshot.ports.size());
    ASSERT_EQ(rejected->hard_floor_contributors.size(), 2U);
    ASSERT_EQ(rejected->queue_sizing_recipes.size(), snapshot.edges.size());
    EXPECT_EQ(rejected->queue_sizing_recipes.front().headroom_ratio, (RateRatio{1U, 1U}));
    EXPECT_EQ(rejected->queue_sizing_recipes.front().disposition, PlanDisposition::kRejected);
    EXPECT_EQ(rejected->queue_sizing_recipes.front().reason,
              diagnostic_id("RATE_HEADROOM_INVALID"));
  }
}

// Invalid compile-wide options still produce a complete deterministic rejected artifact. The
// canonical 1/1 recipe ratio is rejection-only evidence; it must never turn the invalid request
// into an accepted sizing policy or weaken private-artifact validation.
// REQ: COMPILE-069
TEST(CompileRatePlan, ProductionCompileReturnsRejectedArtifactForInvalidHeadroom) {
  Graph graph{"rate-aware-invalid-headroom"};
  test::add_compile_test_default_clock(graph);
  auto source = graph.op<FixedRateSource>("source");
  auto sink = graph.op<ClockRateSink>("sink");
  graph.add_flow(source->output, sink->input);
  CompileOptions options;
  options.automatic_queue_sizing.headroom_ratio = RateRatio{1U, 2U};

  const ExecutionPlan plan = compile(graph, options);

  EXPECT_FALSE(plan.ok());
  EXPECT_TRUE(std::ranges::any_of(plan.diagnostics(), [](const auto& diagnostic) {
    return diagnostic.token == "RATE_HEADROOM_INVALID";
  })) << plan.json();
  ASSERT_EQ(plan.port_rates().size(), 2U);
  ASSERT_EQ(plan.queue_sizing_recipes().size(), 1U);
  EXPECT_EQ(plan.queue_sizing_recipes().front().headroom_ratio, (RateRatio{1U, 1U}));
  EXPECT_EQ(plan.queue_sizing_recipes().front().disposition, PlanDisposition::kRejected);
  EXPECT_NE(plan.json().find("\"schema\":\"holoscan.execution_plan.private.v8\""),
            std::string_view::npos);
}

// REQ: COMPILE-069
TEST(CompileRatePlan, ExpectedInputFallbackSizesOnlyAGenuinelyAbsentUpstream) {
  auto snapshot = fixed_rate_graph();
  snapshot.ports[0].setup.rate_kind = PortRateKind::kUnspecified;
  snapshot.ports[0].setup.declared_nominal_rate.reset();
  snapshot.ports[1].setup.rate_kind = PortRateKind::kExpectedInput;
  snapshot.ports[1].setup.declared_expected_rate = Rate::hz(60U);
  const std::array triggers{one_second_clock_trigger()};
  const auto profiles = standard_profiles(snapshot.vertices.size());

  const auto result = lower_rate_plan(
      snapshot, RateRatio{2U, 1U}, triggers, {}, {}, profiles, PlanGenerationId{41U});

  ASSERT_TRUE(result.errors.empty());
  ASSERT_EQ(result.resolved_rates.size(), 2U);
  EXPECT_EQ(result.resolved_rates[1].resolved_rate, Rate::hz(60U));
  EXPECT_EQ(result.resolved_rates[1].origin, ResolvedRateOrigin::kInputExpectationFallback);
  ASSERT_EQ(result.queue_sizings.size(), 1U);
  EXPECT_EQ(result.queue_sizings.front().rate_recommendation, 120U);
  EXPECT_EQ(result.queue_sizings.front().effective_logical_depth, 120U);
}

// An OnEach consumer has a finite horizon for its secondary inputs once the trigger input's
// nominal rate is known. The expected-rate fallback is allowed only for the genuinely absent
// secondary producer and must use that driver-derived horizon.
// REQ: COMPILE-069
TEST(CompileRatePlan, ExpectedSecondaryInputUsesResolvedOnEachDriverHorizon) {
  GraphSnapshotData snapshot;
  snapshot.generation = GraphGeneration{7U};
  snapshot.vertices = {
      vertex(1U, "driver-source"), vertex(2U, "secondary-source"), vertex(3U, "sink")};
  snapshot.ports = {
      output_port(1U, 1U, "driver-out", scalar_payload()),
      output_port(2U, 2U, "secondary-out", scalar_payload()),
      input_port(3U, 3U, "driver", scalar_payload()),
      input_port(4U, 3U, "secondary", scalar_payload()),
  };
  snapshot.ports[0].setup.rate_kind = PortRateKind::kFixedNominal;
  snapshot.ports[0].setup.declared_nominal_rate = Rate::hz(30U);
  snapshot.ports[1].setup.rate_kind = PortRateKind::kUnspecified;
  snapshot.ports[2].setup.queue_depth.reset();
  snapshot.ports[3].setup.rate_kind = PortRateKind::kExpectedInput;
  snapshot.ports[3].setup.declared_expected_rate = Rate::hz(120U);
  snapshot.ports[3].setup.queue_depth.reset();
  snapshot.edges = {edge(1U, 1U, 3U), edge(2U, 2U, 4U)};
  for (auto& item : snapshot.edges) {
    item.add_flow_options.queue_depth.reset();
    item.add_flow_options.max_inflight_samples.reset();
  }
  const std::array triggers{TemporalTriggerPlanRecord{
      .owner = OperatorRef{.id = 3U, .plan_generation = 41U},
      .kind = TemporalRecordTriggerKind::kEach,
      .trigger_input = PortId{3U},
  }};
  const auto profiles = standard_profiles(snapshot.vertices.size());

  const auto result = lower_rate_plan(
      snapshot, RateRatio{2U, 1U}, triggers, {}, {}, profiles, PlanGenerationId{41U});

  ASSERT_TRUE(result.errors.empty());
  ASSERT_EQ(result.resolved_rates.size(), 4U);
  EXPECT_EQ(result.resolved_rates[2].resolved_rate, Rate::hz(30U));
  EXPECT_EQ(result.resolved_rates[3].resolved_rate, Rate::hz(120U));
  EXPECT_EQ(result.resolved_rates[3].origin, ResolvedRateOrigin::kInputExpectationFallback);
  ASSERT_EQ(result.queue_sizings.size(), 2U);
  const auto secondary =
      std::ranges::find(result.queue_sizings, EdgeId{2U}, &ResolvedQueueSizing::edge);
  ASSERT_NE(secondary, result.queue_sizings.end());
  EXPECT_EQ(secondary->nominal_arrivals, 4U);
  EXPECT_EQ(secondary->rate_recommendation, 8U);
  EXPECT_EQ(secondary->effective_logical_depth, 8U);
}

// A lifecycle-selected OnEach driver controls how long a fixed-rate secondary input accumulates.
// The immutable plan must preserve that inverse-rate expression even though its exact duration is
// unavailable until the Discover barrier closes.
// REQ: COMPILE-069
TEST(CompileRatePlan, LifecycleOnEachDriverOwnsSecondaryHorizonAndRestartDependency) {
  GraphSnapshotData snapshot;
  snapshot.generation = GraphGeneration{7U};
  snapshot.vertices = {vertex(1U, "camera-source"), vertex(2U, "imu-source"), vertex(3U, "fusion")};
  snapshot.ports = {
      output_port(1U, 1U, "camera-out", scalar_payload()),
      output_port(2U, 2U, "imu-out", scalar_payload()),
      input_port(3U, 3U, "camera", scalar_payload()),
      input_port(4U, 3U, "imu", scalar_payload()),
  };
  snapshot.ports[0].setup.rate_kind = PortRateKind::kLifecycleNominal;
  snapshot.ports[0].setup.lifecycle_binding_stage = LifecycleStage::kDiscover;
  snapshot.ports[0].setup.declared_max_nominal_rate = Rate::hz(120U);
  snapshot.ports[1].setup.rate_kind = PortRateKind::kFixedNominal;
  snapshot.ports[1].setup.declared_nominal_rate = Rate::hz(1000U);
  snapshot.ports[2].setup.queue_depth.reset();
  snapshot.ports[3].setup.queue_depth.reset();
  snapshot.edges = {edge(1U, 1U, 3U), edge(2U, 2U, 4U)};
  for (auto& item : snapshot.edges) {
    item.add_flow_options.queue_depth.reset();
    item.add_flow_options.max_inflight_samples.reset();
  }
  const std::array triggers{TemporalTriggerPlanRecord{
      .owner = OperatorRef{.id = 3U, .plan_generation = 41U},
      .kind = TemporalRecordTriggerKind::kEach,
      .trigger_input = PortId{3U},
  }};
  const auto profiles = standard_profiles(snapshot.vertices.size());

  const auto result = lower_rate_plan(
      snapshot, RateRatio{2U, 1U}, triggers, {}, {}, profiles, PlanGenerationId{41U});

  ASSERT_TRUE(result.errors.empty());
  const auto horizon = std::ranges::find(
      result.accumulation_horizons, EdgeId{2U}, &AccumulationHorizonPlanRecord::edge);
  ASSERT_NE(horizon, result.accumulation_horizons.end());
  EXPECT_EQ(horizon->origin, AccumulationHorizonOrigin::kDriverInterarrival);
  EXPECT_FALSE(horizon->horizon.has_value());
  EXPECT_EQ(horizon->inverse_rate_port, PortId{3U});
  const auto recipe = std::ranges::find(
      result.queue_sizing_recipes, EdgeId{2U}, &QueueSizingRecipePlanRecord::edge);
  ASSERT_NE(recipe, result.queue_sizing_recipes.end());
  EXPECT_EQ(recipe->latest_binding_stage, LifecycleStage::kDiscover);
  EXPECT_EQ(recipe->geometry_mode, AllocationGeometryMode::kLifecycleCandidate);
  const auto sizing =
      std::ranges::find(result.queue_sizings, EdgeId{2U}, &ResolvedQueueSizing::edge);
  ASSERT_NE(sizing, result.queue_sizings.end());
  EXPECT_FALSE(sizing->rate_recommendation.has_value());
  EXPECT_FALSE(sizing->admitted_depth_ceiling.has_value());
  EXPECT_TRUE(std::ranges::find(result.restart_dependencies,
                                RateRestartDependency{.source_owner = VertexId{1U},
                                                      .dependent_owner = VertexId{3U}}) !=
              result.restart_dependencies.end());
  EXPECT_TRUE(std::ranges::find(result.restart_dependencies,
                                RateRestartDependency{.source_owner = VertexId{1U},
                                                      .dependent_owner = VertexId{2U}}) !=
              result.restart_dependencies.end());

  std::vector strict(snapshot.vertices.size(), RateBoundedEvidenceRequirement::kNotRequired);
  strict[2] = RateBoundedEvidenceRequirement::kRequired;
  const auto rejected = lower_rate_plan(
      snapshot, RateRatio{2U, 1U}, triggers, {}, {}, profiles, strict, PlanGenerationId{41U});
  EXPECT_TRUE(std::ranges::any_of(rejected.errors, [](const auto& diagnostic) {
    return diagnostic.token == "RATE_BOUNDED_EVIDENCE_REQUIRED";
  }));
}

// The private-v7 artifact must preserve an unresolved lifecycle driver as a stable inverse-rate
// port reference rather than silently replacing it with a guessed duration. The small golden is
// a valid JSON object containing the complete canonical accumulation-horizons artifact section
// generated by production compile and lowering, not a hand-authored ExecutionPlanState fixture.
// REQ: COMPILE-069
TEST(CompileRatePlan, LifecycleOnEachInverseRateArtifactMatchesNormalizedBaseline) {
  Graph graph{"lifecycle-on-each-inverse-rate-artifact"};
  test::add_compile_test_default_clock(graph);
  auto driver = graph.op<LifecycleOnEachDriverSource>("driver");
  auto secondary = graph.op<FixedRateSecondarySource>("secondary");
  auto sink = graph.op<LifecycleOnEachSink>("sink");
  graph.add_flow(driver->output, sink->driver);
  graph.add_flow(secondary->output, sink->secondary);

  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.accumulation_horizons().size(), 2U);
  const auto driver_horizon =
      std::ranges::find_if(plan.accumulation_horizons(), [](const auto& record) {
        return record.origin == AccumulationHorizonOrigin::kDriverInterarrival;
      });
  ASSERT_NE(driver_horizon, plan.accumulation_horizons().end());
  EXPECT_FALSE(driver_horizon->horizon.has_value());
  EXPECT_EQ(driver_horizon->inverse_rate_port, PortId{3U});

  const auto accepted = ExecutionPlanAccess::retain(plan);
  ASSERT_TRUE(accepted);
  auto normalized = std::make_shared<ExecutionPlanState>(*accepted);
  rebind_plan_generation(*normalized, PlanGenerationId{1U});
  normalized->profile_manifest.target_triple = "normalized-target-triple";
  const auto finalized = finalize_plan_artifact(*normalized);
  ASSERT_TRUE(finalized) << runtime_error_message(finalized.error());
  const ExecutionPlan normalized_plan = ExecutionPlanAccess::make(std::move(normalized));

  const std::string golden_path = std::string(HOLOSCAN_COMPILER_TEST_DATA_DIR) +
                                  "/lifecycle_on_each_inverse_rate_horizons.v8.normalized.json";
  std::ifstream golden_stream{golden_path, std::ios::binary};
  ASSERT_TRUE(golden_stream.is_open()) << golden_path;
  const std::string golden{std::istreambuf_iterator<char>{golden_stream},
                           std::istreambuf_iterator<char>{}};
  ASSERT_TRUE(golden_stream.good() || golden_stream.eof()) << golden_path;

  const std::string normalized_json =
      artifact_test::normalized_artifact_json(normalized_plan.json());
  constexpr std::string_view kHorizonSection{"  \"accumulation_horizons\": ["};
  constexpr std::string_view kNextSection{"  \"hard_floor_contributors\":"};
  const std::size_t begin = normalized_json.find(kHorizonSection);
  ASSERT_NE(begin, std::string::npos) << normalized_json;
  const std::size_t end = normalized_json.find(kNextSection, begin);
  ASSERT_NE(end, std::string::npos) << normalized_json;

  std::string normalized_horizon_json{"{\n"};
  normalized_horizon_json.append(normalized_json, begin, end - begin);
  ASSERT_TRUE(normalized_horizon_json.ends_with(",\n")) << normalized_horizon_json;
  normalized_horizon_json.erase(normalized_horizon_json.size() - 2U, 1U);
  normalized_horizon_json.append("}\n");

  EXPECT_EQ(normalized_horizon_json, golden)
      << "review the complete private-v7 accumulation-horizons diff before updating "
      << golden_path;

  // The inverse-rate port and its origin are semantic facts. A coherent alternate OnEach driver
  // must therefore produce a different private-v7 identity rather than merely different JSON.
  auto alternate_driver = *accepted;
  ASSERT_EQ(alternate_driver.temporal_triggers.size(), 1U);
  alternate_driver.temporal_triggers.front().trigger_input = PortId{4U};
  for (auto& record : alternate_driver.accumulation_horizons) {
    record.inverse_rate_port = PortId{4U};
    record.origin = record.edge == EdgeId(1U) ? AccumulationHorizonOrigin::kDriverInterarrival
                                              : AccumulationHorizonOrigin::kProducerInterarrival;
  }
  const auto alternate_finalized = finalize_plan_artifact(alternate_driver);
  ASSERT_TRUE(alternate_finalized) << runtime_error_message(alternate_finalized.error());
  EXPECT_NE(alternate_driver.digest, accepted->digest);

  // Exactly one horizon representation is required: neither form and both forms fail closed
  // before a malformed record can enter the semantic preimage.
  const auto expect_malformed_horizon = [&](const auto mutate) {
    auto malformed = *accepted;
    mutate(malformed.accumulation_horizons.front());
    const auto malformed_finalized = finalize_plan_artifact(malformed);
    ASSERT_FALSE(malformed_finalized);
    EXPECT_EQ(malformed_finalized.error().code, ErrorCode::kInvalidArgument);
    EXPECT_EQ(malformed_finalized.error().message, "RATE_PLAN_RECORDS_INVALID");
  };
  expect_malformed_horizon([](auto& record) {
    record.horizon = ExactDuration{.numerator_ns = 1'000'000'000U, .denominator = 1U};
  });
  expect_malformed_horizon([](auto& record) { record.inverse_rate_port.reset(); });
}

// REQ: COMPILE-069
TEST(CompileRatePlan, PendingLifecycleRateIsNeverReclassifiedAsAbsentFallback) {
  auto snapshot = fixed_rate_graph();
  snapshot.ports[0].setup.rate_kind = PortRateKind::kLifecycleNominal;
  snapshot.ports[0].setup.declared_nominal_rate.reset();
  snapshot.ports[0].setup.lifecycle_binding_stage = LifecycleStage::kDiscover;
  snapshot.ports[0].setup.declared_max_nominal_rate = Rate::hz(120U);
  snapshot.ports[1].setup.rate_kind = PortRateKind::kExpectedInput;
  snapshot.ports[1].setup.declared_expected_rate = Rate::hz(60U);
  const std::array triggers{one_second_clock_trigger()};
  const auto profiles = standard_profiles(snapshot.vertices.size());

  const auto result = lower_rate_plan(
      snapshot, RateRatio{2U, 1U}, triggers, {}, {}, profiles, PlanGenerationId{41U});

  ASSERT_TRUE(result.errors.empty());
  ASSERT_EQ(result.resolved_rates.size(), 2U);
  EXPECT_FALSE(result.resolved_rates[1].resolved_rate.has_value());
  EXPECT_FALSE(result.resolved_rates[1].genuinely_absent);
  EXPECT_EQ(result.resolved_rates[1].max_nominal_rate, Rate::hz(120U));
  ASSERT_EQ(result.queue_sizings.size(), 1U);
  EXPECT_FALSE(result.queue_sizings.front().rate_recommendation.has_value());
  EXPECT_EQ(result.queue_sizings.front().admitted_depth_ceiling, 240U);
}

// The rational multiply must fail before a wrapped rate can look like an absent producer and
// authorize the input's weaker expected-rate fallback.
// REQ: COMPILE-069
TEST(CompileRatePlan, FollowsRationalOverflowFailsClosedWithoutExpectedFallback) {
  GraphSnapshotData snapshot;
  snapshot.generation = GraphGeneration{7U};
  snapshot.vertices = {vertex(1U, "source"), vertex(2U, "relay"), vertex(3U, "sink")};
  snapshot.ports = {
      output_port(1U, 1U, "source-out", scalar_payload()),
      input_port(2U, 2U, "relay-in", scalar_payload()),
      output_port(3U, 2U, "relay-out", scalar_payload()),
      input_port(4U, 3U, "sink-in", scalar_payload()),
  };
  snapshot.ports[0].setup.rate_kind = PortRateKind::kFixedNominal;
  snapshot.ports[0].setup.declared_nominal_rate =
      Rate::hz(std::numeric_limits<std::uint64_t>::max());
  snapshot.ports[2].setup.rate_kind = PortRateKind::kFollowsInput;
  snapshot.ports[2].setup.follows_input = GraphPortId{2U};
  snapshot.ports[2].setup.follows_ratio = RateRatio{2U, 1U};
  snapshot.ports[3].setup.rate_kind = PortRateKind::kExpectedInput;
  snapshot.ports[3].setup.declared_expected_rate = Rate::hz(60U);
  snapshot.edges = {edge(1U, 1U, 2U), edge(2U, 3U, 4U)};
  const auto profiles = standard_profiles(snapshot.vertices.size());

  const auto result =
      lower_rate_plan(snapshot, RateRatio{2U, 1U}, {}, {}, {}, profiles, PlanGenerationId{41U});

  EXPECT_TRUE(std::ranges::any_of(result.errors, [](const auto& diagnostic) {
    return diagnostic.token == "RATE_ARITHMETIC_OVERFLOW";
  }));
  ASSERT_EQ(result.resolved_rates.size(), 4U);
  EXPECT_EQ(result.resolved_rates[0].resolved_rate,
            Rate::hz(std::numeric_limits<std::uint64_t>::max()));
  EXPECT_FALSE(result.resolved_rates[2].resolved_rate.has_value());
  EXPECT_FALSE(result.resolved_rates[3].resolved_rate.has_value());
  EXPECT_FALSE(result.resolved_rates[3].genuinely_absent);
}

// REQ: COMPILE-069
TEST(CompileRatePlan, OnEachInverseRateCrossCancelsBeforeMultiplication) {
  auto snapshot = fixed_rate_graph(1'000'000'000U, 1'000'000'000'001U);
  const std::array triggers{TemporalTriggerPlanRecord{
      .owner = OperatorRef{.id = 2U, .plan_generation = 41U},
      .kind = TemporalRecordTriggerKind::kEach,
      .trigger_input = PortId{2U},
  }};
  const auto profiles = standard_profiles(snapshot.vertices.size());

  const auto result = lower_rate_plan(
      snapshot, RateRatio{2U, 1U}, triggers, {}, {}, profiles, PlanGenerationId{41U});

  ASSERT_TRUE(result.errors.empty());
  ASSERT_EQ(result.accumulation_horizons.size(), 1U);
  EXPECT_EQ(result.accumulation_horizons.front().inverse_rate_port, PortId{2U});
  EXPECT_FALSE(result.accumulation_horizons.front().horizon.has_value());
  ASSERT_EQ(result.queue_sizings.size(), 1U);
  EXPECT_EQ(result.queue_sizings.front().nominal_arrivals, 1U);
  EXPECT_EQ(result.queue_sizings.front().effective_logical_depth, 2U);
}

// REQ: COMPILE-069
TEST(CompileRatePlan, HardFloorCompositionSumsDistinctGroupsAndMaxesAlternatives) {
  const std::array contributors{
      HardFloorContributorPlanRecord{.id = HardFloorContributorId{1U},
                                     .edge = EdgeId{1U},
                                     .required_slots = 8U,
                                     .scenario = HardFloorScenarioId{1U},
                                     .alternative_group = HardFloorAlternativeGroupId{1U}},
      HardFloorContributorPlanRecord{.id = HardFloorContributorId{2U},
                                     .edge = EdgeId{1U},
                                     .required_slots = 6U,
                                     .scenario = HardFloorScenarioId{1U},
                                     .alternative_group = HardFloorAlternativeGroupId{1U}},
      HardFloorContributorPlanRecord{.id = HardFloorContributorId{3U},
                                     .edge = EdgeId{1U},
                                     .required_slots = 3U,
                                     .scenario = HardFloorScenarioId{1U},
                                     .alternative_group = HardFloorAlternativeGroupId{2U}},
      HardFloorContributorPlanRecord{.id = HardFloorContributorId{4U},
                                     .edge = EdgeId{1U},
                                     .required_slots = 9U,
                                     .scenario = HardFloorScenarioId{2U},
                                     .alternative_group = HardFloorAlternativeGroupId{1U}},
  };

  EXPECT_EQ(compose_hard_floor(contributors, EdgeId{1U}), 11U);

  auto overflow = contributors;
  overflow[0].required_slots = std::numeric_limits<std::uint32_t>::max();
  overflow[1].alternative_group = HardFloorAlternativeGroupId{2U};
  EXPECT_FALSE(compose_hard_floor(overflow, EdgeId{1U}).has_value());
}

// REQ: COMPILE-069
TEST(CompileRatePlan, LoweringSeparatesSimultaneousBurstAndSelectorRetentionLifetimes) {
  auto snapshot = fixed_rate_graph();
  snapshot.ports[0].setup.max_emits_per_compute = 3U;
  const std::array selections{InputSelectionPlanRecord{
      .input = PortId{2U},
      .selector = InputSelectorKind::kDrainAll,
      .cardinality = InputSelectionCardinality::kSampleBatch,
      .max_selected_samples = 8U,
      .max_selected_samples_origin = SelectedSampleBoundOrigin::kSelectorMaxSamples,
      .batch_selection_mode = BatchSelectionMode::kConsumeSelected,
      .cache_capacity = 8U,
  }};
  const auto profiles = standard_profiles(snapshot.vertices.size());

  const auto result = lower_rate_plan(
      snapshot, RateRatio{2U, 1U}, {}, selections, {}, profiles, PlanGenerationId{41U});

  ASSERT_TRUE(result.errors.empty());
  ASSERT_EQ(result.queue_sizings.size(), 1U);
  EXPECT_EQ(result.queue_sizings.front().hard_floor, 11U);
  EXPECT_EQ(result.queue_sizings.front().effective_logical_depth, 11U);
  ASSERT_EQ(result.hard_floor_contributors.size(), 3U);
  EXPECT_EQ(result.hard_floor_contributors[0].alternative_group,
            result.hard_floor_contributors[1].alternative_group);
  EXPECT_NE(result.hard_floor_contributors[1].alternative_group,
            result.hard_floor_contributors[2].alternative_group);
}

// A resolved-retained DrainAll maximum is the queue depth viewed through the selector record. It
// is not a second retained lifetime that can be summed back into the queue's own hard floor.
// REQ: COMPILE-069
// REQ: OPERATOR-048
TEST(CompileRatePlan, ProductionDrainAllKeepsExplicitDepthWithoutCircularRetentionFloor) {
  for (const std::uint32_t depth : {2U, 200U}) {
    SCOPED_TRACE(depth);
    Graph graph{"rate-aware-explicit-drain-all-" + std::to_string(depth)};
    test::add_compile_test_default_clock(graph);
    auto source = graph.op<FixedRateSource>("source");
    auto sink = graph.op<ClockDrainAllRateSink>("sink");
    graph.add_flow(source->output, sink->input, ConnectionOptions{.queue_depth = depth});

    const ExecutionPlan plan = compile(graph);

    ASSERT_TRUE(plan.ok()) << plan.json();
    ASSERT_EQ(plan.payload_edges().size(), 1U);
    EXPECT_EQ(plan.payload_edges().front().queue_depth, depth);
    ASSERT_EQ(plan.input_selections().size(), 1U);
    const auto& selection = plan.input_selections().front();
    EXPECT_EQ(selection.selector, InputSelectorKind::kDrainAll);
    EXPECT_EQ(selection.max_selected_samples_origin,
              SelectedSampleBoundOrigin::kResolvedRetainedCapacity);
    EXPECT_EQ(selection.max_selected_samples, depth);
    EXPECT_EQ(selection.cache_capacity, depth);
    EXPECT_TRUE(std::ranges::none_of(plan.hard_floor_contributors(), [](const auto& contributor) {
      return contributor.origin == HardFloorOrigin::kSelectorRetention;
    }));
  }
}

// Automatic depth, cache capacity, and the resolved-retained DrainAll bound are three projections
// of the same evaluated queue authority and must remain byte-for-byte equal in the final plan.
// REQ: COMPILE-069
TEST(CompileRatePlan, ProductionDrainAllFinalizesOneAutomaticDepthAuthority) {
  Graph graph{"rate-aware-automatic-drain-all"};
  test::add_compile_test_default_clock(graph);
  auto source = graph.op<FixedRateSource>("source");
  auto sink = graph.op<ClockDrainAllRateSink>("sink");
  graph.add_flow(source->output, sink->input);

  const ExecutionPlan plan = compile(graph);

  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.payload_edges().size(), 1U);
  ASSERT_EQ(plan.input_selections().size(), 1U);
  const std::uint32_t depth = plan.payload_edges().front().queue_depth;
  EXPECT_EQ(depth, 120U);
  EXPECT_EQ(plan.input_selections().front().max_selected_samples, depth);
  EXPECT_EQ(plan.input_selections().front().cache_capacity, depth);
  EXPECT_TRUE(std::ranges::none_of(plan.hard_floor_contributors(), [](const auto& contributor) {
    return contributor.origin == HardFloorOrigin::kSelectorRetention;
  }));
}

// REQ: COMPILE-069
// DrainAll consumes the full active retained prefix, so automatic queue sizing must update both
// its cache and selected-sample bounds from the same resolved edge authority.
TEST(CompileRatePlan, TemporalFinalizationKeepsDrainAllSelectionBoundWithQueueDepth) {
  auto snapshot = fixed_rate_graph();
  const std::array sizings{ResolvedQueueSizing{
      .edge = EdgeId{1U},
      .effective_logical_depth = 11U,
  }};
  std::array selections{InputSelectionPlanRecord{
      .input = PortId{2U},
      .selector = InputSelectorKind::kDrainAll,
      .cardinality = InputSelectionCardinality::kSampleBatch,
      .max_selected_samples = 8U,
      .max_selected_samples_origin = SelectedSampleBoundOrigin::kResolvedRetainedCapacity,
      .batch_selection_mode = BatchSelectionMode::kConsumeSelected,
      .cache_capacity = 8U,
  }};

  finalize_temporal_queue_capacities(snapshot, sizings, selections, {}, {}, {});

  EXPECT_EQ(selections.front().cache_capacity, 11U);
  EXPECT_EQ(selections.front().max_selected_samples, 11U);
}

// REQ: COMPILE-069
// Automatic DrainAll advances the complete retained prefix. Stability classification therefore
// uses the same derived logical depth that temporal finalization publishes, not the provisional
// selector bound that exists while rate lowering is still in progress.
TEST(CompileRatePlan, AutomaticDrainAllUsesDerivedDepthForStabilityClassification) {
  const std::array triggers{one_second_clock_trigger()};
  const std::array selections{InputSelectionPlanRecord{
      .input = PortId{2U},
      .selector = InputSelectorKind::kDrainAll,
      .cardinality = InputSelectionCardinality::kSampleBatch,
      .max_selected_samples = 1U,
      .max_selected_samples_origin = SelectedSampleBoundOrigin::kResolvedRetainedCapacity,
      .batch_selection_mode = BatchSelectionMode::kConsumeSelected,
      .cache_capacity = 1U,
  }};
  const auto snapshot = fixed_rate_graph(60U);
  const auto profiles = standard_profiles(snapshot.vertices.size());

  const auto result = lower_rate_plan(
      snapshot, RateRatio{2U, 1U}, triggers, selections, {}, profiles, PlanGenerationId{41U});

  ASSERT_TRUE(result.errors.empty());
  ASSERT_EQ(result.queue_sizings.size(), 1U);
  EXPECT_EQ(result.queue_sizings.front().nominal_arrivals, 60U);
  EXPECT_EQ(result.queue_sizings.front().effective_logical_depth, 120U);
  EXPECT_EQ(result.queue_sizings.front().stability, StabilityClass::kStableNominalEstimate);
  EXPECT_FALSE(std::ranges::any_of(result.warnings, [](const auto& diagnostic) {
    return diagnostic.token == "RATE_SUSTAINED_BACKPRESSURE" ||
           diagnostic.token == "RATE_PREDICTED_REPLACEMENT";
  }));
}

// REQ: COMPILE-069
TEST(CompileRatePlan, StabilityUsesExactActivationRateTimesAdvancementCapacity) {
  const std::array triggers{one_second_clock_trigger()};
  const std::array selections{InputSelectionPlanRecord{
      .input = PortId{2U},
      .selector = InputSelectorKind::kDrainAll,
      .cardinality = InputSelectionCardinality::kSampleBatch,
      .max_selected_samples = 60U,
      .max_selected_samples_origin = SelectedSampleBoundOrigin::kSelectorMaxSamples,
      .batch_selection_mode = BatchSelectionMode::kConsumeSelected,
      .cache_capacity = 60U,
  }};

  const auto stable_snapshot = fixed_rate_graph(60U);
  const auto stable_profiles = standard_profiles(stable_snapshot.vertices.size());
  const auto stable = lower_rate_plan(stable_snapshot,
                                      RateRatio{2U, 1U},
                                      triggers,
                                      selections,
                                      {},
                                      stable_profiles,
                                      PlanGenerationId{41U});
  ASSERT_TRUE(stable.errors.empty());
  ASSERT_EQ(stable.queue_sizings.size(), 1U);
  EXPECT_EQ(stable.queue_sizings.front().stability, StabilityClass::kStableNominalEstimate);

  const auto unstable_snapshot = fixed_rate_graph(61U);
  const auto unstable_profiles = standard_profiles(unstable_snapshot.vertices.size());
  const auto unstable = lower_rate_plan(unstable_snapshot,
                                        RateRatio{2U, 1U},
                                        triggers,
                                        selections,
                                        {},
                                        unstable_profiles,
                                        PlanGenerationId{41U});
  ASSERT_TRUE(unstable.errors.empty());
  ASSERT_EQ(unstable.queue_sizings.size(), 1U);
  EXPECT_EQ(unstable.queue_sizings.front().stability, StabilityClass::kPredictedSustainedBlocking);

  auto lossy_snapshot = stable_snapshot;
  lossy_snapshot.edges.front().add_flow_options.queue_policy = QueuePolicy::kDropOldest;
  const auto lossy = lower_rate_plan(lossy_snapshot,
                                     RateRatio{2U, 1U},
                                     triggers,
                                     selections,
                                     {},
                                     stable_profiles,
                                     PlanGenerationId{41U});
  ASSERT_TRUE(lossy.errors.empty());
  ASSERT_EQ(lossy.queue_sizings.size(), 1U);
  EXPECT_EQ(lossy.queue_sizings.front().stability, StabilityClass::kStableNominalEstimate);

  auto overloaded_lossy_snapshot = unstable_snapshot;
  overloaded_lossy_snapshot.edges.front().add_flow_options.queue_policy = QueuePolicy::kDropOldest;
  const auto overloaded_lossy = lower_rate_plan(overloaded_lossy_snapshot,
                                                RateRatio{2U, 1U},
                                                triggers,
                                                selections,
                                                {},
                                                unstable_profiles,
                                                PlanGenerationId{41U});
  ASSERT_TRUE(overloaded_lossy.errors.empty());
  ASSERT_EQ(overloaded_lossy.queue_sizings.size(), 1U);
  EXPECT_EQ(overloaded_lossy.queue_sizings.front().stability,
            StabilityClass::kPredictedReplacementOrDrop);
}

// Event/readiness trigger families do not supply a finite cadence, and a bracketed window does
// not have a bounded destructive advancement count. Keep both cases explicitly conservative
// instead of manufacturing a backpressure verdict from unrelated capacity fields.
// REQ: COMPILE-069
TEST(CompileRatePlan, UnboundedCadenceAndAdvancementRemainUnclassified) {
  const auto snapshot = fixed_rate_graph();
  const auto profiles = standard_profiles(snapshot.vertices.size());
  const std::array exact_selection{InputSelectionPlanRecord{
      .input = PortId{2U},
      .selector = InputSelectorKind::kExact,
      .cardinality = InputSelectionCardinality::kSingleSample,
      .max_selected_samples = 1U,
      .max_selected_samples_origin = SelectedSampleBoundOrigin::kSingleSample,
      .batch_selection_mode = BatchSelectionMode::kNotApplicable,
      .cache_capacity = 1U,
  }};
  for (const auto kind : {TemporalRecordTriggerKind::kAnyInput,
                          TemporalRecordTriggerKind::kAligned,
                          TemporalRecordTriggerKind::kReadable,
                          TemporalRecordTriggerKind::kNotified,
                          TemporalRecordTriggerKind::kDoorbell,
                          TemporalRecordTriggerKind::kSpin}) {
    SCOPED_TRACE(static_cast<std::uint32_t>(kind));
    const std::array triggers{TemporalTriggerPlanRecord{
        .owner = OperatorRef{.id = 2U, .plan_generation = 41U},
        .kind = kind,
        .period_ns = 1'000'000'000,
    }};
    const auto result = lower_rate_plan(snapshot,
                                        RateRatio{2U, 1U},
                                        triggers,
                                        exact_selection,
                                        {},
                                        profiles,
                                        PlanGenerationId{41U});
    ASSERT_TRUE(result.errors.empty());
    ASSERT_EQ(result.queue_sizings.size(), 1U);
    EXPECT_TRUE(result.accumulation_horizons.empty());
    EXPECT_EQ(result.queue_sizings.front().stability, StabilityClass::kUnresolvedBoundedEvidence);
  }

  auto missing_clock_trigger = one_second_clock_trigger();
  missing_clock_trigger.trigger_clock = ClockId{};
  const std::array missing_clock_triggers{missing_clock_trigger};
  auto missing_clock_snapshot = snapshot;
  missing_clock_snapshot.ports.front().setup.rate_kind = PortRateKind::kUnspecified;
  missing_clock_snapshot.ports.front().setup.declared_nominal_rate.reset();
  missing_clock_snapshot.ports.back().setup.rate_kind = PortRateKind::kExpectedInput;
  missing_clock_snapshot.ports.back().setup.declared_expected_rate = Rate::hz(60U);
  const auto missing_clock = lower_rate_plan(missing_clock_snapshot,
                                             RateRatio{2U, 1U},
                                             missing_clock_triggers,
                                             exact_selection,
                                             {},
                                             profiles,
                                             PlanGenerationId{41U});
  ASSERT_TRUE(missing_clock.errors.empty());
  EXPECT_TRUE(missing_clock.accumulation_horizons.empty());
  ASSERT_EQ(missing_clock.resolved_rates.size(), 2U);
  EXPECT_FALSE(missing_clock.resolved_rates.back().resolved_rate.has_value());

  const std::array predicate_triggers{TemporalTriggerPlanRecord{
      .owner = OperatorRef{.id = 2U, .plan_generation = 41U},
      .kind = TemporalRecordTriggerKind::kPredicate,
      .trigger_input = PortId{2U},
  }};
  const std::array input_predicates{PredicateTriggerPlanRecord{
      .id = "input-predicate",
      .owner = OperatorRef{.id = 2U, .plan_generation = 41U},
      .graph_generation = PlanGenerationId{41U},
      .source = PredicateEvaluationSourceKind::kInputUpdate,
      .driver = PortId{2U},
  }};
  const auto input_predicate = lower_rate_plan(snapshot,
                                               RateRatio{2U, 1U},
                                               predicate_triggers,
                                               exact_selection,
                                               {},
                                               profiles,
                                               {},
                                               PlanGenerationId{41U},
                                               {},
                                               {},
                                               input_predicates);
  ASSERT_TRUE(input_predicate.errors.empty());
  ASSERT_EQ(input_predicate.queue_sizings.size(), 1U);
  EXPECT_TRUE(input_predicate.accumulation_horizons.empty());
  EXPECT_EQ(input_predicate.queue_sizings.front().stability,
            StabilityClass::kUnresolvedBoundedEvidence);

  const std::array window_triggers{TemporalTriggerPlanRecord{
      .owner = OperatorRef{.id = 2U, .plan_generation = 41U},
      .kind = TemporalRecordTriggerKind::kWindow,
      .trigger_input = PortId{2U},
      .trigger_clock = ClockId{1U},
      .bounded_state_capacity = 2U,
  }};
  const std::array windows{WindowTriggerPlanRecord{
      .owner = OperatorRef{.id = 2U, .plan_generation = 41U},
      .anchor = PortId{2U},
      .wait_clock = ClockId{1U},
      .max_wait_ns = 2'000'000,
      .max_pending_windows = 2U,
  }};
  const std::array bracketed_selection{InputSelectionPlanRecord{
      .input = PortId{2U},
      .selector = InputSelectorKind::kBracketedWindow,
      .cardinality = InputSelectionCardinality::kSampleBatch,
      .max_selected_samples = 3U,
      .max_selected_samples_origin = SelectedSampleBoundOrigin::kSelectorMaxSamples,
      .batch_selection_mode = BatchSelectionMode::kNonDestructiveSnapshot,
      .cache_capacity = 3U,
  }};
  auto window_snapshot = snapshot;
  window_snapshot.ports.front().setup.rate_kind = PortRateKind::kUnspecified;
  window_snapshot.ports.front().setup.declared_nominal_rate.reset();
  window_snapshot.ports.back().setup.rate_kind = PortRateKind::kExpectedInput;
  window_snapshot.ports.back().setup.declared_expected_rate = Rate::hz(60U);
  const auto bracketed = lower_rate_plan(window_snapshot,
                                         RateRatio{2U, 1U},
                                         window_triggers,
                                         bracketed_selection,
                                         {},
                                         profiles,
                                         {},
                                         PlanGenerationId{41U},
                                         {},
                                         windows);
  ASSERT_TRUE(bracketed.errors.empty());
  ASSERT_EQ(bracketed.queue_sizings.size(), 1U);
  ASSERT_EQ(bracketed.accumulation_horizons.size(), 1U);
  ASSERT_EQ(bracketed.resolved_rates.size(), 2U);
  EXPECT_EQ(bracketed.resolved_rates.back().origin, ResolvedRateOrigin::kInputExpectationFallback);
  EXPECT_EQ(bracketed.queue_sizings.front().stability, StabilityClass::kUnresolvedBoundedEvidence);

  auto mismatched_windows = windows;
  mismatched_windows.front().wait_clock = ClockId{2U};
  const auto mismatched_window = lower_rate_plan(window_snapshot,
                                                 RateRatio{2U, 1U},
                                                 window_triggers,
                                                 bracketed_selection,
                                                 {},
                                                 profiles,
                                                 {},
                                                 PlanGenerationId{41U},
                                                 {},
                                                 mismatched_windows);
  ASSERT_TRUE(mismatched_window.errors.empty());
  EXPECT_TRUE(mismatched_window.accumulation_horizons.empty());

  auto mismatched_anchor_windows = windows;
  mismatched_anchor_windows.front().anchor = PortId{1U};
  const auto mismatched_anchor = lower_rate_plan(window_snapshot,
                                                 RateRatio{2U, 1U},
                                                 window_triggers,
                                                 bracketed_selection,
                                                 {},
                                                 profiles,
                                                 {},
                                                 PlanGenerationId{41U},
                                                 {},
                                                 mismatched_anchor_windows);
  ASSERT_TRUE(mismatched_anchor.errors.empty());
  EXPECT_TRUE(mismatched_anchor.accumulation_horizons.empty());
  ASSERT_EQ(mismatched_anchor.resolved_rates.size(), 2U);
  EXPECT_FALSE(mismatched_anchor.resolved_rates.back().resolved_rate.has_value());

  auto zero_anchor_triggers = window_triggers;
  zero_anchor_triggers.front().trigger_input = PortId{};
  auto zero_anchor_windows = windows;
  zero_anchor_windows.front().anchor = PortId{};
  const auto zero_anchor = lower_rate_plan(window_snapshot,
                                           RateRatio{2U, 1U},
                                           zero_anchor_triggers,
                                           bracketed_selection,
                                           {},
                                           profiles,
                                           {},
                                           PlanGenerationId{41U},
                                           {},
                                           zero_anchor_windows);
  ASSERT_TRUE(zero_anchor.errors.empty());
  EXPECT_TRUE(zero_anchor.accumulation_horizons.empty());
  ASSERT_EQ(zero_anchor.resolved_rates.size(), 2U);
  EXPECT_FALSE(zero_anchor.resolved_rates.back().resolved_rate.has_value());

  const std::array clock_predicate_triggers{TemporalTriggerPlanRecord{
      .owner = OperatorRef{.id = 2U, .plan_generation = 41U},
      .kind = TemporalRecordTriggerKind::kPredicate,
      .trigger_clock = ClockId{1U},
      .period_ns = 1'000'000'000,
  }};
  const std::array mismatched_clock_predicates{PredicateTriggerPlanRecord{
      .id = "mismatched-clock-predicate",
      .owner = OperatorRef{.id = 2U, .plan_generation = 41U},
      .graph_generation = PlanGenerationId{41U},
      .source = PredicateEvaluationSourceKind::kClock,
      .clock = ClockId{2U},
      .period_ns = 1'000'000'000,
  }};
  const auto mismatched_predicate = lower_rate_plan(snapshot,
                                                    RateRatio{2U, 1U},
                                                    clock_predicate_triggers,
                                                    exact_selection,
                                                    {},
                                                    profiles,
                                                    {},
                                                    PlanGenerationId{41U},
                                                    {},
                                                    {},
                                                    mismatched_clock_predicates);
  ASSERT_TRUE(mismatched_predicate.errors.empty());
  EXPECT_TRUE(mismatched_predicate.accumulation_horizons.empty());
}

// REQ: COMPILE-069
TEST(CompileRatePlan, ConstrainedProfilesRejectNominalOnlySizingEvenWithExplicitDepth) {
  auto snapshot = fixed_rate_graph();
  snapshot.edges.front().graph_override.queue_depth = 256U;
  const std::array triggers{one_second_clock_trigger()};
  const auto profiles = standard_profiles(snapshot.vertices.size());
  const std::array requirements{RateBoundedEvidenceRequirement::kRequired,
                                RateBoundedEvidenceRequirement::kRequired};

  const auto constrained = lower_rate_plan(
      snapshot, RateRatio{2U, 1U}, triggers, {}, {}, profiles, requirements, PlanGenerationId{41U});
  EXPECT_TRUE(std::ranges::any_of(constrained.errors, [](const auto& diagnostic) {
    return diagnostic.token == "RATE_BOUNDED_EVIDENCE_REQUIRED";
  }));
  ASSERT_EQ(constrained.queue_sizing_recipes.size(), 1U);
  EXPECT_EQ(constrained.queue_sizing_recipes.front().disposition, PlanDisposition::kRejected);
  EXPECT_EQ(constrained.queue_sizing_recipes.front().reason, DiagnosticId{317U});

  snapshot.ports.front().setup.rate_kind = PortRateKind::kUnspecified;
  snapshot.ports.front().setup.declared_nominal_rate.reset();
  const auto legacy_explicit = lower_rate_plan(
      snapshot, RateRatio{2U, 1U}, triggers, {}, {}, profiles, requirements, PlanGenerationId{41U});
  EXPECT_TRUE(std::ranges::none_of(legacy_explicit.errors, [](const auto& diagnostic) {
    return diagnostic.token == "RATE_BOUNDED_EVIDENCE_REQUIRED";
  }));
}

// Bounded-arrival evidence is a per-vertex profile obligation. Attribute each failure to the
// endpoint that owns the requirement instead of reporting every source-side failure on the sink.
// REQ: COMPILE-069
TEST(CompileRatePlan, BoundedEvidenceDiagnosticNamesEveryRequiringEndpoint) {
  auto snapshot = fixed_rate_graph();
  snapshot.edges.front().graph_override.queue_depth = 256U;
  const std::array triggers{one_second_clock_trigger()};
  const auto profiles = standard_profiles(snapshot.vertices.size());

  const auto diagnose = [&](RateBoundedEvidenceRequirement source_requirement,
                            RateBoundedEvidenceRequirement target_requirement) {
    const std::array requirements{source_requirement, target_requirement};
    const auto result = lower_rate_plan(snapshot,
                                        RateRatio{2U, 1U},
                                        triggers,
                                        {},
                                        {},
                                        profiles,
                                        requirements,
                                        PlanGenerationId{41U});
    std::vector<DiagnosticSource> sources;
    for (const auto& diagnostic : result.errors) {
      if (diagnostic.token == "RATE_BOUNDED_EVIDENCE_REQUIRED") {
        sources.push_back(diagnostic.source);
      }
    }
    return sources;
  };

  const auto source_only = diagnose(RateBoundedEvidenceRequirement::kRequired,
                                    RateBoundedEvidenceRequirement::kNotRequired);
  ASSERT_EQ(source_only.size(), 1U);
  EXPECT_EQ(source_only.front().vertex, VertexId{1U});
  EXPECT_EQ(source_only.front().port, PortId{1U});

  const auto target_only = diagnose(RateBoundedEvidenceRequirement::kNotRequired,
                                    RateBoundedEvidenceRequirement::kRequired);
  ASSERT_EQ(target_only.size(), 1U);
  EXPECT_EQ(target_only.front().vertex, VertexId{2U});
  EXPECT_EQ(target_only.front().port, PortId{2U});

  const auto both = diagnose(RateBoundedEvidenceRequirement::kRequired,
                             RateBoundedEvidenceRequirement::kRequired);
  ASSERT_EQ(both.size(), 2U);
  EXPECT_EQ(both[0].vertex, VertexId{1U});
  EXPECT_EQ(both[0].port, PortId{1U});
  EXPECT_EQ(both[1].vertex, VertexId{2U});
  EXPECT_EQ(both[1].port, PortId{2U});

  // A source-owned obligation applies independently to every outgoing edge. The diagnostic is
  // canonicalized by source, but deduplication must not accidentally admit later fan-out recipes.
  auto fanout = fixed_rate_graph();
  fanout.edges.front().graph_override.queue_depth = 256U;
  fanout.vertices.push_back(vertex(3U, "second-sink"));
  auto second_input = input_port(3U, 3U, "in", scalar_payload());
  second_input.setup.queue_depth.reset();
  fanout.ports.push_back(std::move(second_input));
  fanout.edges.push_back(edge(2U, 1U, 3U));
  fanout.edges.back().graph_override.queue_depth = 256U;
  fanout.edges.back().add_flow_options.queue_depth.reset();
  fanout.edges.back().add_flow_options.max_inflight_samples.reset();

  const std::array fanout_triggers{
      one_second_clock_trigger(),
      TemporalTriggerPlanRecord{
          .owner = OperatorRef{.id = 3U, .plan_generation = 41U},
          .kind = TemporalRecordTriggerKind::kClock,
          .trigger_clock = ClockId{1U},
          .period_ns = 1'000'000'000,
      },
  };
  const std::array fanout_requirements{RateBoundedEvidenceRequirement::kRequired,
                                       RateBoundedEvidenceRequirement::kNotRequired,
                                       RateBoundedEvidenceRequirement::kNotRequired};
  const auto fanout_result = lower_rate_plan(fanout,
                                             RateRatio{2U, 1U},
                                             fanout_triggers,
                                             {},
                                             {},
                                             standard_profiles(fanout.vertices.size()),
                                             fanout_requirements,
                                             PlanGenerationId{41U});
  std::vector<DiagnosticSource> fanout_sources;
  for (const auto& diagnostic : fanout_result.errors) {
    if (diagnostic.token == "RATE_BOUNDED_EVIDENCE_REQUIRED") {
      fanout_sources.push_back(diagnostic.source);
    }
  }
  ASSERT_EQ(fanout_sources.size(), 1U);
  EXPECT_EQ(fanout_sources.front().vertex, VertexId{1U});
  EXPECT_EQ(fanout_sources.front().port, PortId{1U});
  ASSERT_EQ(fanout_result.queue_sizing_recipes.size(), 2U);
  EXPECT_TRUE(std::ranges::all_of(fanout_result.queue_sizing_recipes, [](const auto& recipe) {
    return recipe.disposition == PlanDisposition::kRejected &&
           recipe.reason == diagnostic_id("RATE_BOUNDED_EVIDENCE_REQUIRED");
  }));
}

// A maximum selectable nominal average and an explicit depth are allocation authorities, not a
// maximum-arrival, jitter, or burst envelope. Derive the rate policy from the actual profile pass
// so this test covers the production profile-to-rate join rather than injecting private booleans.
// REQ: COMPILE-069
TEST(CompileRatePlan, PublicProfileMatrixKeepsBoundedArrivalEvidenceIndependentOfCapacity) {
  struct Expectation {
    Profile profile;
    bool requires_bounded_evidence;
  };
  constexpr std::array expectations{
      Expectation{Profile::kStandard, false},
      Expectation{Profile::kResearch, false},
      Expectation{Profile::kRobotics, false},
      Expectation{Profile::kRealTime, true},
      Expectation{Profile::kRegulated, true},
  };

  for (const auto expectation : expectations) {
    SCOPED_TRACE(static_cast<std::uint32_t>(expectation.profile));
    auto snapshot = fixed_rate_graph();
    snapshot.ports.front().setup.rate_kind = PortRateKind::kLifecycleNominal;
    snapshot.ports.front().setup.declared_nominal_rate.reset();
    snapshot.ports.front().setup.lifecycle_binding_stage = LifecycleStage::kDiscover;
    snapshot.ports.front().setup.declared_max_nominal_rate = Rate::hz(120U);
    snapshot.edges.front().graph_override.queue_depth = 256U;
    const auto admission = resolve_profile_admission(snapshot, expectation.profile);
    ASSERT_EQ(admission.rate_evidence_requirements().size(), snapshot.vertices.size());
    const std::array triggers{one_second_clock_trigger()};

    const auto result = lower_rate_plan(snapshot,
                                        RateRatio{2U, 1U},
                                        triggers,
                                        {},
                                        {},
                                        admission.effective_profiles(),
                                        admission.rate_evidence_requirements(),
                                        PlanGenerationId{41U});
    const bool rejected = std::ranges::any_of(result.errors, [](const auto& diagnostic) {
      return diagnostic.token == "RATE_BOUNDED_EVIDENCE_REQUIRED";
    });
    EXPECT_EQ(rejected, expectation.requires_bounded_evidence);
    ASSERT_EQ(result.queue_sizing_recipes.size(), 1U);
    EXPECT_EQ(result.queue_sizing_recipes.front().disposition,
              expectation.requires_bounded_evidence ? PlanDisposition::kRejected
                                                    : PlanDisposition::kAccepted);
    ASSERT_EQ(result.queue_sizings.size(), 1U);
    EXPECT_EQ(result.queue_sizings.front().explicit_depth, 256U);
    EXPECT_EQ(result.queue_sizings.front().admitted_depth_ceiling, 240U);
  }

  auto critical = fixed_rate_graph();
  critical.ports.front().setup.rate_kind = PortRateKind::kLifecycleNominal;
  critical.ports.front().setup.declared_nominal_rate.reset();
  critical.ports.front().setup.lifecycle_binding_stage = LifecycleStage::kDiscover;
  critical.ports.front().setup.declared_max_nominal_rate = Rate::hz(120U);
  critical.edges.front().graph_override.queue_depth = 256U;
  critical.regions.push_back(RegionRecord{
      .id = RegionId{1U},
      .graph = GraphId{1U},
      .name = "critical",
      .members = {VertexId{1U}, VertexId{2U}},
      .constraints = {RegionConstraint::bounded_memory()},
      .profile = Profile::kRobotics,
  });
  const auto critical_admission = resolve_profile_admission(critical, Profile::kStandard);
  EXPECT_TRUE(std::ranges::all_of(critical_admission.rate_evidence_requirements(),
                                  [](const auto requirement) {
                                    return requirement == RateBoundedEvidenceRequirement::kRequired;
                                  }));
  const std::array triggers{one_second_clock_trigger()};
  const auto critical_result = lower_rate_plan(critical,
                                               RateRatio{2U, 1U},
                                               triggers,
                                               {},
                                               {},
                                               critical_admission.effective_profiles(),
                                               critical_admission.rate_evidence_requirements(),
                                               PlanGenerationId{41U});
  EXPECT_TRUE(std::ranges::any_of(critical_result.errors, [](const auto& diagnostic) {
    return diagnostic.token == "RATE_BOUNDED_EVIDENCE_REQUIRED";
  }));
}

// REQ: COMPILE-069
TEST(CompileRatePlan, RateDerivedPoolDependenciesComposeEachLifetimeExactlyOnce) {
  constexpr std::string_view kPool = "tensor-output-pool";
  const std::array<PoolCapacityContributorRecord, 2U> predecessors{
      PoolCapacityContributorRecord{.pool_id = std::string(kPool),
                                    .contributor_id = std::string(kPool) + ".producer",
                                    .maximum_distinct_slots = 1U,
                                    .reasons = {TensorSlotClaimReason::kProducerLoan}},
      PoolCapacityContributorRecord{.pool_id = std::string(kPool),
                                    .contributor_id = std::string(kPool) + ".retirement",
                                    .maximum_distinct_slots = 1U,
                                    .reasons = {TensorSlotClaimReason::kRetirementService}},
  };
  const std::array<PoolCapacityDependencyPlanRecord, 4U> dependencies{
      PoolCapacityDependencyPlanRecord{
          .id = PoolCapacityDependencyId{1U},
          .pool_id = std::string(kPool),
          .origin = PoolClaimOrigin::kProducerLoan,
          .source = PoolCapacityDependencySource::kPredecessorContributor,
          .predecessor_contributor_id = std::string(kPool) + ".producer",
          .scenario = PoolCapacityScenarioId{1U},
          .alternative_group = PoolCapacityAlternativeGroupId{1U}},
      PoolCapacityDependencyPlanRecord{
          .id = PoolCapacityDependencyId{2U},
          .pool_id = std::string(kPool),
          .origin = PoolClaimOrigin::kEdgeDelivery,
          .source = PoolCapacityDependencySource::kQueueSizingRecipe,
          .queue_sizing_recipe = QueueSizingRecipeId{1U},
          .scenario = PoolCapacityScenarioId{1U},
          .alternative_group = PoolCapacityAlternativeGroupId{2U},
          .reasons = {TensorSlotClaimReason::kAcceptedPublication,
                      TensorSlotClaimReason::kHolomqTransportOrFanout}},
      PoolCapacityDependencyPlanRecord{.id = PoolCapacityDependencyId{3U},
                                       .pool_id = std::string(kPool),
                                       .origin = PoolClaimOrigin::kTemporalRetention,
                                       .source = PoolCapacityDependencySource::kQueueSizingRecipe,
                                       .queue_sizing_recipe = QueueSizingRecipeId{1U},
                                       .scenario = PoolCapacityScenarioId{1U},
                                       .alternative_group = PoolCapacityAlternativeGroupId{3U},
                                       .reasons = {TensorSlotClaimReason::kTemporalCache}},
      PoolCapacityDependencyPlanRecord{
          .id = PoolCapacityDependencyId{4U},
          .pool_id = std::string(kPool),
          .origin = PoolClaimOrigin::kRetirement,
          .source = PoolCapacityDependencySource::kPredecessorContributor,
          .predecessor_contributor_id = std::string(kPool) + ".retirement",
          .scenario = PoolCapacityScenarioId{1U},
          .alternative_group = PoolCapacityAlternativeGroupId{4U}},
  };
  const std::array<ResolvedQueueSizing, 1U> sizings{
      ResolvedQueueSizing{.edge = EdgeId{1U},
                          .recipe = QueueSizingRecipeId{1U},
                          .admitted_depth_ceiling = 6U,
                          .effective_logical_depth = 6U}};

  const auto capacity = evaluate_pool_capacity_plan(kPool, dependencies, predecessors, sizings);
  ASSERT_TRUE(capacity) << capacity.error().message;
  EXPECT_EQ(capacity->required_entries, 14U);
  EXPECT_EQ(capacity->admitted_entry_ceiling, 14U);
}

// REQ: COMPILE-069
TEST(CompileRatePlan, TensorPoolUsesRateRecipeClaimsInsteadOfCountingAggregateClaimsTwice) {
  auto snapshot = fixed_rate_graph(3U);
  snapshot.ports[0].payload_identity = tensor_payload();
  snapshot.ports[1].payload_identity = tensor_payload();
  snapshot.ports[0].generic_loan_supported = false;
  snapshot.ports[0].generic_loan_slot_size_bytes = 0U;
  snapshot.ports[0].setup.tensor_layout = TensorRepresentation{
      .memory_kind = MemoryKind::kHost,
      .dtype = DLDataType{kDLUInt, 8U, 1U},
      .rank = 1U,
  };
  snapshot.ports[0].setup.tensor_allocation =
      TensorPoolBounds{.max_rank = 1U, .max_byte_span = 64U};
  const std::array triggers{one_second_clock_trigger()};
  const std::array selections{InputSelectionPlanRecord{
      .input = PortId{2U},
      .selector = InputSelectorKind::kExact,
      .cardinality = InputSelectionCardinality::kSingleSample,
      .max_selected_samples = 1U,
      .cache_capacity = 6U,
  }};
  const auto profiles = standard_profiles(snapshot.vertices.size());
  const auto rate = lower_rate_plan(
      snapshot, RateRatio{2U, 1U}, triggers, selections, {}, profiles, PlanGenerationId{41U});
  ASSERT_TRUE(rate.errors.empty());
  ASSERT_EQ(rate.queue_sizings.size(), 1U);
  ASSERT_EQ(rate.queue_sizings.front().effective_logical_depth, 6U);

  const auto physical =
      lower_for_test(snapshot,
                     PlanGenerationId{41U},
                     profiles,
                     selections,
                     {},
                     {},
                     nullptr,
                     {},
                     {},
                     LowerForTestOptions{.resolved_queue_sizings = rate.queue_sizings,
                                         .queue_sizing_recipes = rate.queue_sizing_recipes});
  ASSERT_TRUE(physical.errors.empty())
      << (physical.errors.empty() ? std::string{} : physical.errors.front().token);
  ASSERT_EQ(physical.tensor_allocations.size(), 1U);
  EXPECT_EQ(physical.tensor_allocations.front().slot_count, 14U);
  EXPECT_EQ(physical.pool_capacity_contributors.size(), 2U);
  EXPECT_EQ(physical.pool_capacity_dependencies.size(), 4U);
  ASSERT_EQ(physical.provider_allocation_plans.size(), 1U);
  EXPECT_EQ(physical.provider_allocation_plans.front().sizing_dependency_count, 4U);
  EXPECT_EQ(std::ranges::count_if(physical.pool_capacity_dependencies,
                                  [](const auto& dependency) {
                                    return dependency.source ==
                                           PoolCapacityDependencySource::kQueueSizingRecipe;
                                  }),
            2U);
}

// REQ: COMPILE-069
TEST(CompileRatePlan, FiniteLifecycleMaximumPreallocatesCompileConcreteProviderGeometry) {
  auto snapshot = fixed_rate_graph();
  snapshot.ports[0].payload_identity = tensor_payload();
  snapshot.ports[1].payload_identity = tensor_payload();
  snapshot.ports[0].generic_loan_supported = false;
  snapshot.ports[0].generic_loan_slot_size_bytes = 0U;
  snapshot.ports[0].setup.tensor_layout = TensorRepresentation{
      .memory_kind = MemoryKind::kHost,
      .dtype = DLDataType{kDLUInt, 8U, 1U},
      .rank = 1U,
  };
  snapshot.ports[0].setup.tensor_allocation =
      TensorPoolBounds{.max_rank = 1U, .max_byte_span = 64U};
  snapshot.ports[0].setup.rate_kind = PortRateKind::kLifecycleNominal;
  snapshot.ports[0].setup.declared_nominal_rate.reset();
  snapshot.ports[0].setup.lifecycle_binding_stage = LifecycleStage::kDiscover;
  snapshot.ports[0].setup.declared_max_nominal_rate = Rate::hz(120U);
  const std::array triggers{one_second_clock_trigger()};
  const auto profiles = standard_profiles(snapshot.vertices.size());
  const auto rate = lower_rate_plan(
      snapshot, RateRatio{2U, 1U}, triggers, {}, {}, profiles, PlanGenerationId{41U});
  ASSERT_TRUE(rate.errors.empty());
  ASSERT_EQ(rate.queue_sizings.size(), 1U);
  ASSERT_EQ(rate.queue_sizings.front().admitted_depth_ceiling, 240U);

  const auto physical =
      lower_for_test(snapshot,
                     PlanGenerationId{41U},
                     profiles,
                     {},
                     {},
                     {},
                     nullptr,
                     {},
                     {},
                     LowerForTestOptions{.resolved_queue_sizings = rate.queue_sizings,
                                         .queue_sizing_recipes = rate.queue_sizing_recipes});

  ASSERT_TRUE(physical.errors.empty())
      << (physical.errors.empty() ? std::string{} : physical.errors.front().token);
  ASSERT_EQ(physical.payload_edges.size(), 1U);
  EXPECT_EQ(physical.payload_edges.front().queue_depth, 240U);
  ASSERT_EQ(physical.process_local_channel_subscribers.size(), 1U);
  EXPECT_EQ(physical.process_local_channel_subscribers.front().requested_queue_depth, 240U);
  ASSERT_EQ(physical.provider_allocation_plans.size(), 1U);
  EXPECT_EQ(physical.provider_allocation_plans.front().geometry_mode,
            AllocationGeometryMode::kCompileConcrete);
  ASSERT_EQ(physical.tensor_allocations.size(), 1U);
  EXPECT_EQ(physical.tensor_allocations.front().slot_count, 242U);
}

// Descriptor-backed multi-field pools have one compile-cold allocation contract per canonical
// field. A finite lifecycle maximum therefore keeps every field provider compile concrete.
// REQ: COMPILE-069
TEST(CompileRatePlan, FiniteLifecycleMaximumKeepsMultiFieldProvidersCompileConcrete) {
  const RuntimePortTypeOps* const type_ops = runtime_port_type_ops<schema::ImageT>();
  ASSERT_NE(type_ops, nullptr);
  auto fixture = make_multi_tensor_snapshot(type_ops->wire_capacity);
  auto& snapshot = fixture.graph;
  snapshot.ports[0].setup.rate_kind = PortRateKind::kLifecycleNominal;
  snapshot.ports[0].setup.lifecycle_binding_stage = LifecycleStage::kDiscover;
  snapshot.ports[0].setup.declared_max_nominal_rate = Rate::hz(120U);
  snapshot.ports[1].setup.queue_depth.reset();
  snapshot.edges[0].add_flow_options.queue_depth.reset();
  snapshot.edges[0].add_flow_options.max_inflight_samples.reset();
  const std::array triggers{one_second_clock_trigger()};

  const auto rate = lower_rate_plan(
      snapshot, RateRatio{2U, 1U}, triggers, {}, {}, fixture.profiles, PlanGenerationId{41U});
  ASSERT_TRUE(rate.errors.empty());
  ASSERT_EQ(rate.queue_sizing_recipes.size(), 1U);
  EXPECT_EQ(rate.queue_sizing_recipes.front().geometry_mode,
            AllocationGeometryMode::kCompileConcrete);

  const auto physical =
      lower_for_test(snapshot,
                     PlanGenerationId{41U},
                     fixture.profiles,
                     {},
                     {},
                     {},
                     nullptr,
                     {},
                     fixture.partitions,
                     LowerForTestOptions{.resolved_queue_sizings = rate.queue_sizings,
                                         .queue_sizing_recipes = rate.queue_sizing_recipes});

  ASSERT_TRUE(physical.errors.empty())
      << (physical.errors.empty() ? std::string{} : physical.errors.front().token);
  ASSERT_EQ(physical.tensor_field_allocations.size(), 3U);
  ASSERT_EQ(physical.provider_allocation_plans.size(), 3U);
  EXPECT_TRUE(std::ranges::all_of(physical.provider_allocation_plans, [](const auto& provider) {
    return provider.allocation_kind == ProviderAllocationKind::kTensorFieldOutput &&
           provider.geometry_mode == AllocationGeometryMode::kCompileConcrete &&
           !provider.tensor_template.has_value();
  }));
}

// One fanout edge is automatically sized from a fixed nominal rate; the other keeps an explicit
// depth. Each edge recipe must join every field pool exactly once, and the explicit edge must not
// inherit the rate-derived recommendation as a storage obligation. The adjacent lifecycle test
// covers finite-maximum geometry without weakening HoloMQ's active-depth fail-closed boundary.
// REQ: COMPILE-069
TEST(CompileRatePlan, MultiFieldFanoutKeepsPoolDependencyRangesLocalAndAvoidsDoubleCounting) {
  const RuntimePortTypeOps* const type_ops = runtime_port_type_ops<schema::ImageT>();
  ASSERT_NE(type_ops, nullptr);
  auto fixture = make_multi_tensor_snapshot(type_ops->wire_capacity);
  auto& snapshot = fixture.graph;

  snapshot.vertices.push_back(vertex(3U, "multi-sink-explicit"));
  auto second_input = input_port(3U, 3U, "frames", image_payload());
  second_input.serialized_payload_capacity_bytes = type_ops->wire_capacity;
  second_input.descriptor_tensor_field_count = 3U;
  second_input.setup.tensor_fields = snapshot.ports[0].setup.tensor_fields;
  second_input.setup.queue_depth.reset();
  snapshot.ports.push_back(std::move(second_input));

  snapshot.ports[0].setup.rate_kind = PortRateKind::kFixedNominal;
  snapshot.ports[0].setup.declared_nominal_rate = Rate::hz(120U);
  snapshot.ports[1].setup.queue_depth.reset();
  snapshot.edges[0].add_flow_options.queue_depth.reset();
  snapshot.edges[0].add_flow_options.max_inflight_samples.reset();
  snapshot.edges.push_back(edge(2U, 1U, 3U, 4U));
  snapshot.edges[1].add_flow_options.max_inflight_samples.reset();
  fixture.profiles = standard_profiles(snapshot.vertices.size());
  fixture.partitions = default_execution_partitions(snapshot);

  const std::array triggers{
      one_second_clock_trigger(),
      TemporalTriggerPlanRecord{
          .owner = OperatorRef{.id = 3U, .plan_generation = 41U},
          .kind = TemporalRecordTriggerKind::kClock,
          .trigger_clock = ClockId{1U},
          .period_ns = 1'000'000'000,
      },
  };
  const auto rate = lower_rate_plan(
      snapshot, RateRatio{2U, 1U}, triggers, {}, {}, fixture.profiles, PlanGenerationId{41U});
  ASSERT_TRUE(rate.errors.empty());
  ASSERT_EQ(rate.queue_sizing_recipes.size(), 2U);
  ASSERT_EQ(rate.queue_sizings.size(), 2U);
  const auto* const automatic = find_resolved_queue_sizing(rate.queue_sizings, EdgeId{1U});
  const auto* const explicit_depth = find_resolved_queue_sizing(rate.queue_sizings, EdgeId{2U});
  ASSERT_NE(automatic, nullptr);
  ASSERT_NE(explicit_depth, nullptr);
  EXPECT_FALSE(automatic->explicit_depth);
  EXPECT_EQ(automatic->effective_logical_depth, 240U);
  EXPECT_EQ(automatic->admitted_depth_ceiling, 240U);
  EXPECT_EQ(explicit_depth->explicit_depth, 4U);
  EXPECT_EQ(explicit_depth->effective_logical_depth, 4U);

  const auto physical =
      lower_for_test(snapshot,
                     PlanGenerationId{41U},
                     fixture.profiles,
                     {},
                     {},
                     {},
                     nullptr,
                     {},
                     fixture.partitions,
                     LowerForTestOptions{.resolved_queue_sizings = rate.queue_sizings,
                                         .queue_sizing_recipes = rate.queue_sizing_recipes});
  ASSERT_TRUE(physical.errors.empty())
      << (physical.errors.empty() ? std::string{} : physical.errors.front().token);
  ASSERT_EQ(physical.tensor_field_allocations.size(), 3U);
  ASSERT_EQ(physical.provider_allocation_plans.size(), 3U);
  ASSERT_EQ(physical.pool_capacity_contributors.size(), 6U);
  ASSERT_EQ(physical.pool_capacity_dependencies.size(), 18U);

  for (std::size_t provider_index = 0U; provider_index < physical.provider_allocation_plans.size();
       ++provider_index) {
    const auto& provider = physical.provider_allocation_plans[provider_index];
    EXPECT_EQ(provider.allocation_kind, ProviderAllocationKind::kTensorFieldOutput);
    EXPECT_EQ(provider.geometry_mode, AllocationGeometryMode::kCompileConcrete);
    EXPECT_FALSE(provider.tensor_template);
    EXPECT_EQ(provider.first_sizing_dependency.value,
              static_cast<std::uint32_t>(provider_index * 6U + 1U));
    ASSERT_EQ(provider.sizing_dependency_count, 6U);

    const std::size_t first = provider.first_sizing_dependency.value - 1U;
    const auto dependencies = std::span{physical.pool_capacity_dependencies}.subspan(first, 6U);
    EXPECT_TRUE(std::ranges::all_of(dependencies, [&](const auto& dependency) {
      return dependency.pool_id == provider.pool_id && dependency.mapping_multiplicity == 1U;
    }));
    EXPECT_EQ(std::ranges::count_if(dependencies,
                                    [](const auto& dependency) {
                                      return dependency.source ==
                                             PoolCapacityDependencySource::kPredecessorContributor;
                                    }),
              2U);
    EXPECT_EQ(std::ranges::count_if(dependencies,
                                    [](const auto& dependency) {
                                      return dependency.queue_sizing_recipe ==
                                             QueueSizingRecipeId{1U};
                                    }),
              2U);
    EXPECT_EQ(std::ranges::count_if(dependencies,
                                    [](const auto& dependency) {
                                      return dependency.queue_sizing_recipe ==
                                             QueueSizingRecipeId{2U};
                                    }),
              2U);

    const auto evaluated = evaluate_pool_capacity_plan(provider.pool_id,
                                                       physical.pool_capacity_dependencies,
                                                       physical.pool_capacity_contributors,
                                                       rate.queue_sizings);
    ASSERT_TRUE(evaluated) << evaluated.error().message;
    EXPECT_EQ(evaluated->required_entries, 492U);
    EXPECT_EQ(evaluated->admitted_entry_ceiling, 492U);

    const auto allocation = std::ranges::find(physical.tensor_field_allocations,
                                              provider.pool_id,
                                              &TensorFieldAllocationPlanRecord::pool_id);
    ASSERT_NE(allocation, physical.tensor_field_allocations.end());
    EXPECT_EQ(allocation->slot_count, 492U);
    EXPECT_EQ(allocation->aggregate_byte_capacity, 492U * allocation->bounds.max_byte_span);
  }

  for (const QueueSizingRecipeId recipe : {QueueSizingRecipeId{1U}, QueueSizingRecipeId{2U}}) {
    EXPECT_EQ(std::ranges::count_if(
                  physical.pool_capacity_dependencies,
                  [&](const auto& dependency) { return dependency.queue_sizing_recipe == recipe; }),
              6U);
  }
}

// Task 041 deliberately does not infer descriptor-backed multi-field pool geometry from a
// no-maximum lifecycle candidate. The complete field family rejects without publishing a partial
// provider or allocation inventory.
// REQ: COMPILE-069
TEST(CompileRatePlan, NoMaximumLifecycleMultiFieldCandidateFailsClosedWithoutPartialGeometry) {
  const RuntimePortTypeOps* const type_ops = runtime_port_type_ops<schema::ImageT>();
  ASSERT_NE(type_ops, nullptr);
  auto fixture = make_multi_tensor_snapshot(type_ops->wire_capacity);
  auto& snapshot = fixture.graph;
  snapshot.ports[0].setup.rate_kind = PortRateKind::kLifecycleNominal;
  snapshot.ports[0].setup.lifecycle_binding_stage = LifecycleStage::kDiscover;
  snapshot.ports[1].setup.queue_depth.reset();
  snapshot.edges[0].add_flow_options.queue_depth.reset();
  snapshot.edges[0].add_flow_options.max_inflight_samples.reset();
  const std::array triggers{one_second_clock_trigger()};

  const auto rate = lower_rate_plan(
      snapshot, RateRatio{2U, 1U}, triggers, {}, {}, fixture.profiles, PlanGenerationId{41U});
  ASSERT_TRUE(rate.errors.empty());
  ASSERT_EQ(rate.queue_sizing_recipes.size(), 1U);
  EXPECT_EQ(rate.queue_sizing_recipes.front().geometry_mode,
            AllocationGeometryMode::kLifecycleCandidate);

  const auto physical =
      lower_for_test(snapshot,
                     PlanGenerationId{41U},
                     fixture.profiles,
                     {},
                     {},
                     {},
                     nullptr,
                     {},
                     fixture.partitions,
                     LowerForTestOptions{.resolved_queue_sizings = rate.queue_sizings,
                                         .queue_sizing_recipes = rate.queue_sizing_recipes});

  const auto diagnostic = std::ranges::find(
      physical.errors, "PHYSICAL_TENSOR_PROVIDER_UNSUPPORTED", &PlanLoweringDiagnostic::token);
  ASSERT_NE(diagnostic, physical.errors.end());
  EXPECT_TRUE(diagnostic->json_pointer.ends_with("/tensor_fields/2/rate_geometry"));
  EXPECT_TRUE(physical.tensor_field_allocations.empty());
  EXPECT_TRUE(std::ranges::none_of(physical.provider_allocation_plans, [](const auto& provider) {
    return provider.allocation_kind == ProviderAllocationKind::kTensorFieldOutput;
  }));
}

// REQ: COMPILE-069
TEST(CompileRatePlan, ProductionCompileAcceptsFiniteLifecycleCeilingAsConcreteArtifactGeometry) {
  Graph graph{"rate-aware-lifecycle-ceiling"};
  test::add_compile_test_default_clock(graph);
  auto source = graph.op<LifecycleRateTensorSource>("source");
  auto sink = graph.op<ClockTensorRateSink>("sink");
  graph.add_flow(source->output, sink->input);

  const ExecutionPlan plan = compile(graph);

  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.provider_allocation_plans().size(), 1U);
  EXPECT_EQ(plan.provider_allocation_plans().front().geometry_mode,
            AllocationGeometryMode::kCompileConcrete);
  ASSERT_EQ(plan.tensor_allocations().size(), 1U);
  EXPECT_GT(plan.tensor_allocations().front().slot_count, 0U);
  EXPECT_NE(plan.json().find("\"schema\":\"holoscan.execution_plan.private.v8\""),
            std::string::npos);
}

// A permissive Discover-selected rate with no admitted maximum freezes only provider/template
// facts. It must not encode the compile-time hard floor as a concrete Tensor allocation.
// REQ: COMPILE-069
TEST(CompileRatePlan, ProductionCompileEmitsLifecycleTensorTemplateWithoutAllocationPlaceholder) {
  Graph graph{"rate-aware-lifecycle-candidate"};
  test::add_compile_test_default_clock(graph);
  auto source = graph.op<LifecycleCandidateTensorSource>("source");
  auto sink = graph.op<ClockTensorRateSink>("sink");
  graph.add_flow(source->output, sink->input);

  const ExecutionPlan plan = compile(graph);

  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.provider_allocation_plans().size(), 1U);
  const auto& provider = plan.provider_allocation_plans().front();
  EXPECT_EQ(provider.geometry_mode, AllocationGeometryMode::kLifecycleCandidate);
  ASSERT_TRUE(provider.tensor_template.has_value());
  EXPECT_EQ(provider.tensor_template->bounds.max_byte_span, 64U);
  EXPECT_EQ(provider.tensor_template->memory_kind, MemoryKind::kHost);
  EXPECT_TRUE(plan.tensor_allocations().empty());
  EXPECT_TRUE(std::ranges::any_of(
      plan.profile_manifest().required_capabilities, [](const auto& capability) {
        return capability.capability_id == "holoscan.runtime.tensor_pool_materialization" &&
               capability.version == 1U;
      }));
  EXPECT_TRUE(std::ranges::any_of(
      plan.profile_manifest().materialization_obligations, [&](const auto& obligation) {
        return obligation.object_id == provider.pool_id &&
               obligation.strategy_id == "holoscan.runtime.tensor_output_pool/v1" &&
               obligation.complete_before_stage == LifecycleStage::kAllocate;
      }));
}

// A generic value loan is compile-cold producer storage, not retained subscriber queue backing.
// A Discover-selected rate therefore defers only queue geometry and must not invent a second
// provider/allocation family for the generic output.
// REQ: COMPILE-069
TEST(CompileRatePlan, ProductionCompileDefersGenericQueueWithoutRegeneratingOutputLoans) {
  Graph graph{"rate-aware-generic-lifecycle-candidate"};
  test::add_compile_test_default_clock(graph);
  auto source = graph.op<LifecycleCandidateScalarSource>("source");
  auto sink = graph.op<ClockRateSink>("sink");
  graph.add_flow(source->output, sink->input);

  const ExecutionPlan plan = compile(graph);

  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.queue_sizing_recipes().size(), 1U);
  EXPECT_EQ(plan.queue_sizing_recipes().front().geometry_mode,
            AllocationGeometryMode::kLifecycleCandidate);
  ASSERT_EQ(plan.generic_output_loans().size(), 1U);
  EXPECT_EQ(plan.generic_output_loans().front().output, plan.payload_edges().front().source);
  EXPECT_TRUE(plan.provider_allocation_plans().empty());
  EXPECT_TRUE(plan.tensor_allocations().empty());
}

// Lifecycle-selected queues with no compile-time ceiling can only be materialized through a
// provider that accepts a session-local depth before user kAllocate. HoloMQ fixes queue depth at
// endpoint construction and therefore must be rejected at compile time rather than fail on start.
// REQ: COMPILE-069
TEST(CompileRatePlan, LifecycleCandidateQueueRejectsHoloMQPhysicalRealization) {
  auto snapshot = fixed_rate_graph();
  snapshot.ports[0].setup.rate_kind = PortRateKind::kLifecycleNominal;
  snapshot.ports[0].setup.declared_nominal_rate.reset();
  snapshot.ports[0].setup.lifecycle_binding_stage = LifecycleStage::kDiscover;
  const std::array triggers{one_second_clock_trigger()};
  const auto profiles = standard_profiles(snapshot.vertices.size());
  const auto rate = lower_rate_plan(
      snapshot, RateRatio{2U, 1U}, triggers, {}, {}, profiles, PlanGenerationId{41U});
  ASSERT_TRUE(rate.errors.empty());
  ASSERT_EQ(rate.queue_sizing_recipes.size(), 1U);
  ASSERT_EQ(rate.queue_sizing_recipes.front().geometry_mode,
            AllocationGeometryMode::kLifecycleCandidate);

  const auto physical =
      lower_for_test(snapshot,
                     PlanGenerationId{41U},
                     profiles,
                     {},
                     {},
                     {},
                     nullptr,
                     {},
                     {},
                     LowerForTestOptions{.force_holomq_process_local_inline = true,
                                         .resolved_queue_sizings = rate.queue_sizings,
                                         .queue_sizing_recipes = rate.queue_sizing_recipes});

  EXPECT_TRUE(has_error(physical, "QUEUE_ACTIVE_DEPTH_NOT_ENFORCEABLE"));
}

// A finite maximum preallocates backing, but the selected active depth can still be smaller.
// HoloMQ has no independent active-limit authority, so it is not a valid automatic-rate backend.
// REQ: COMPILE-069
TEST(CompileRatePlan, FiniteMaximumAutomaticQueueRejectsHoloMQWithoutActiveLimit) {
  auto snapshot = fixed_rate_graph();
  snapshot.ports[0].setup.rate_kind = PortRateKind::kLifecycleNominal;
  snapshot.ports[0].setup.declared_nominal_rate.reset();
  snapshot.ports[0].setup.lifecycle_binding_stage = LifecycleStage::kDiscover;
  snapshot.ports[0].setup.declared_max_nominal_rate = Rate::hz(120U);
  const std::array triggers{one_second_clock_trigger()};
  const auto profiles = standard_profiles(snapshot.vertices.size());
  const auto rate = lower_rate_plan(
      snapshot, RateRatio{2U, 1U}, triggers, {}, {}, profiles, PlanGenerationId{41U});
  ASSERT_TRUE(rate.errors.empty());
  ASSERT_EQ(rate.queue_sizing_recipes.size(), 1U);
  ASSERT_EQ(rate.queue_sizing_recipes.front().geometry_mode,
            AllocationGeometryMode::kCompileConcrete);
  ASSERT_FALSE(rate.queue_sizing_recipes.front().explicit_depth.has_value());

  const auto physical =
      lower_for_test(snapshot,
                     PlanGenerationId{41U},
                     profiles,
                     {},
                     {},
                     {},
                     nullptr,
                     {},
                     {},
                     LowerForTestOptions{.force_holomq_process_local_inline = true,
                                         .resolved_queue_sizings = rate.queue_sizings,
                                         .queue_sizing_recipes = rate.queue_sizing_recipes});

  EXPECT_TRUE(has_error(physical, "QUEUE_ACTIVE_DEPTH_NOT_ENFORCEABLE"));
}

// WarmUp and Arm occur after Core allocation. Without an explicit depth or finite maximum they
// cannot change physical queue geometry and must fail during compile rather than during start.
// REQ: COMPILE-069
TEST(CompileRatePlan, LateLifecycleBindingWithoutPreallocationBoundFailsClosed) {
  for (const LifecycleStage stage : {LifecycleStage::kWarmUp, LifecycleStage::kArm}) {
    SCOPED_TRACE(static_cast<std::uint32_t>(stage));
    auto snapshot = fixed_rate_graph();
    snapshot.ports[0].setup.rate_kind = PortRateKind::kLifecycleNominal;
    snapshot.ports[0].setup.declared_nominal_rate.reset();
    snapshot.ports[0].setup.lifecycle_binding_stage = stage;
    const std::array triggers{one_second_clock_trigger()};
    const auto profiles = standard_profiles(snapshot.vertices.size());

    const auto unbounded = lower_rate_plan(
        snapshot, RateRatio{2U, 1U}, triggers, {}, {}, profiles, PlanGenerationId{41U});
    EXPECT_TRUE(std::ranges::any_of(unbounded.errors, [](const auto& diagnostic) {
      return diagnostic.token == "QUEUE_PREALLOCATION_BOUND_REQUIRED";
    }));

    snapshot.ports[0].setup.declared_max_nominal_rate = Rate::hz(120U);
    const auto bounded = lower_rate_plan(
        snapshot, RateRatio{2U, 1U}, triggers, {}, {}, profiles, PlanGenerationId{41U});
    EXPECT_TRUE(bounded.errors.empty());

    const std::array producer_driven_triggers{TemporalTriggerPlanRecord{
        .owner = OperatorRef{.id = 2U, .plan_generation = 41U},
        .kind = TemporalRecordTriggerKind::kEach,
        .trigger_input = PortId{2U},
    }};
    const auto producer_driven = lower_rate_plan(snapshot,
                                                 RateRatio{2U, 1U},
                                                 producer_driven_triggers,
                                                 {},
                                                 {},
                                                 profiles,
                                                 PlanGenerationId{41U});
    ASSERT_TRUE(producer_driven.errors.empty());
    ASSERT_EQ(producer_driven.queue_sizings.size(), 1U);
    EXPECT_EQ(producer_driven.queue_sizings.front().admitted_depth_ceiling, 2U);

    snapshot.ports[0].setup.declared_max_nominal_rate.reset();
    snapshot.ports[1].setup.queue_depth = 256U;
    const auto explicit_depth = lower_rate_plan(
        snapshot, RateRatio{2U, 1U}, triggers, {}, {}, profiles, PlanGenerationId{41U});
    EXPECT_TRUE(explicit_depth.errors.empty());
  }
}

// The immutable hard floor is not the candidate generation's active depth. A candidate edge must
// therefore stay on the ordinary direct-retained path even when its compile-cold records happen to
// satisfy the depth-one closed-island shape.
// REQ: COMPILE-069
TEST(CompileRatePlan, LifecycleCandidateQueueIsExcludedFromDepthOneClosedIslandSpecialization) {
  Graph graph{"rate-aware-candidate-closed-island"};
  test::add_compile_test_default_clock(graph);
  auto source = graph.op<LifecycleCandidateScalarSource>("source");
  auto sink = graph.op<EachRateSink>("sink");
  graph.add_flow(source->output, sink->input);

  const ExecutionPlan plan = compile(graph);

  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.queue_sizing_recipes().size(), 1U);
  ASSERT_EQ(plan.queue_sizing_recipes().front().geometry_mode,
            AllocationGeometryMode::kLifecycleCandidate);
  const auto state = ExecutionPlanAccess::retain(plan);
  ASSERT_NE(state, nullptr);
  EXPECT_TRUE(state->closed_island_activations.empty());
}

// OnWindow owns a finite close horizon in WindowTriggerPlanRecord::max_wait_ns. The generic
// temporal trigger deliberately keeps period_ns zero because that field is clock-only.
// REQ: COMPILE-069
TEST(CompileRatePlan, ProductionOnWindowUsesCanonicalMaximumWaitHorizon) {
  Graph graph{"rate-aware-window-horizon"};
  const ClockKey clock = graph.add_clock<RealtimeClock>("window-clock");
  graph.set_default_clock(clock);
  auto source = graph.op<WindowRateSource>("source");
  auto sink = graph.op<WindowRateSink>("sink");
  graph.add_flow(source->anchor, sink->anchor);
  graph.add_flow(source->samples, sink->samples);

  const ExecutionPlan plan = compile(graph);

  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.window_triggers().size(), 1U);
  EXPECT_EQ(plan.window_triggers().front().max_wait_ns, 2'000'000);
  ASSERT_EQ(plan.accumulation_horizons().size(), 2U);
  for (const auto& horizon : plan.accumulation_horizons()) {
    EXPECT_EQ(horizon.origin, AccumulationHorizonOrigin::kTemporalWindowBound);
    EXPECT_EQ(horizon.horizon, (ExactDuration{.numerator_ns = 2'000'000U, .denominator = 1U}));
    EXPECT_FALSE(horizon.inverse_rate_port.has_value());
    EXPECT_EQ(horizon.source_record,
              static_cast<std::uint32_t>(plan.window_triggers().front().owner.id));
  }
  ASSERT_EQ(plan.payload_edges().size(), 2U);
  const auto anchor_edge = std::ranges::find(
      plan.payload_edges(), plan.window_triggers().front().anchor, &PayloadEdgePlanRecord::target);
  ASSERT_NE(anchor_edge, plan.payload_edges().end());
  EXPECT_EQ(anchor_edge->queue_depth, 4U);
  ASSERT_EQ(plan.window_triggers().front().bracketed_inputs.size(), 1U);
  const auto samples = plan.window_triggers().front().bracketed_inputs.front();
  const auto samples_edge =
      std::ranges::find(plan.payload_edges(), samples, &PayloadEdgePlanRecord::target);
  ASSERT_NE(samples_edge, plan.payload_edges().end());
  EXPECT_EQ(samples_edge->queue_depth, 7U);
}

// A predicate evaluated by an admitted periodic clock has exact cadence only when the generic
// trigger and the PredicateTriggerPlanRecord agree on owner, source, and period.
// REQ: COMPILE-069
TEST(CompileRatePlan, ProductionClockPredicateUsesExactCanonicalCadence) {
  Graph graph{"rate-aware-clock-predicate"};
  test::add_compile_test_default_clock(graph);
  auto source = graph.op<FixedRateSource>("source");
  auto sink = graph.op<ClockPredicateRateSink>("sink");
  graph.add_flow(source->output, sink->input);

  const ExecutionPlan plan = compile(graph);

  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.predicate_triggers().size(), 1U);
  const auto& predicate = plan.predicate_triggers().front();
  EXPECT_EQ(predicate.source, PredicateEvaluationSourceKind::kClock);
  EXPECT_EQ(predicate.period_ns, 1'000'000'000);
  ASSERT_EQ(plan.accumulation_horizons().size(), 1U);
  const auto& horizon = plan.accumulation_horizons().front();
  EXPECT_EQ(horizon.origin, AccumulationHorizonOrigin::kClockPeriod);
  EXPECT_EQ(horizon.horizon, (ExactDuration{.numerator_ns = 1'000'000'000U, .denominator = 1U}));
  ASSERT_EQ(plan.payload_edges().size(), 1U);
  EXPECT_EQ(plan.payload_edges().front().queue_depth, 120U);
}

// REQ: COMPILE-069
TEST(CompileRatePlan, PhysicalLoweringConsumesTheSingleEvaluatedDepthAuthority) {
  const auto snapshot = fixed_rate_graph();
  const std::array triggers{one_second_clock_trigger()};
  const auto profiles = standard_profiles(snapshot.vertices.size());
  const auto rate = lower_rate_plan(
      snapshot, RateRatio{2U, 1U}, triggers, {}, {}, profiles, PlanGenerationId{41U});
  ASSERT_TRUE(rate.errors.empty());

  auto physical =
      lower_for_test(snapshot,
                     PlanGenerationId{41U},
                     profiles,
                     {},
                     {},
                     {},
                     nullptr,
                     {},
                     {},
                     LowerForTestOptions{.resolved_queue_sizings = rate.queue_sizings,
                                         .queue_sizing_recipes = rate.queue_sizing_recipes});

  ASSERT_TRUE(physical.errors.empty());
  ASSERT_EQ(physical.payload_edges.size(), 1U);
  EXPECT_EQ(physical.payload_edges.front().queue_depth, 120U);
  ASSERT_EQ(physical.ports.size(), 2U);
  EXPECT_EQ(physical.ports[1].queue_depth, 120U);
  ASSERT_EQ(physical.process_local_channel_services.size(), 1U);
  EXPECT_EQ(physical.process_local_channel_services.front().max_subscriber_queue_depth, 120U);
  ASSERT_EQ(physical.process_local_channel_subscribers.size(), 1U);
  EXPECT_EQ(physical.process_local_channel_subscribers.front().requested_queue_depth, 120U);

  const auto state =
      published_physical_plan_state(std::move(physical), PlanGenerationId{41U}, {}, &rate);
  ASSERT_EQ(state.port_rates.size(), rate.port_rates.size());
  EXPECT_EQ(state.port_rates.front().kind, PortRateKind::kFixedNominal);
  EXPECT_EQ(state.port_rates.front().declared_nominal_rate, Rate::hz(60U));
  ASSERT_EQ(state.accumulation_horizons.size(), rate.accumulation_horizons.size());
  EXPECT_EQ(state.accumulation_horizons.front().origin, AccumulationHorizonOrigin::kClockPeriod);
  ASSERT_EQ(state.hard_floor_contributors.size(), rate.hard_floor_contributors.size());
  EXPECT_EQ(state.hard_floor_contributors.front().origin, HardFloorOrigin::kMinimumQueue);
  ASSERT_EQ(state.queue_sizing_recipes.size(), rate.queue_sizing_recipes.size());
  EXPECT_EQ(state.queue_sizing_recipes.front().origin, QueueDepthOrigin::kRateAndContractDerived);
  EXPECT_FALSE(state.queue_sizing_recipes.front().explicit_depth.has_value());
}

// REQ: COMPILE-069
TEST(CompileRatePlan, ProductionCompilePublishesRateRecipesAndDerivedPhysicalDepth) {
  Graph graph{"rate-aware-compile"};
  test::add_compile_test_default_clock(graph);
  auto source = graph.op<FixedRateSource>("source");
  auto sink = graph.op<ClockRateSink>("sink");
  graph.add_flow(source->output, sink->input);
  ClockRateSink::contract_calls = 0U;

  const ExecutionPlan plan = compile(graph);

  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.payload_edges().size(), 1U);
  EXPECT_EQ(plan.payload_edges().front().queue_depth, 120U);
  ASSERT_EQ(plan.input_selections().size(), 1U);
  EXPECT_EQ(plan.input_selections().front().cache_capacity, 120U);
  EXPECT_EQ(ClockRateSink::contract_calls, 1U);
  ASSERT_EQ(plan.port_rates().size(), 2U);
  ASSERT_EQ(plan.queue_sizing_recipes().size(), 1U);
  EXPECT_EQ(plan.queue_sizing_recipes().front().origin, QueueDepthOrigin::kRateAndContractDerived);
  const std::string_view artifact = plan.json();
  EXPECT_NE(artifact.find("\"schema\":\"holoscan.execution_plan.private.v8\""),
            std::string_view::npos);
  EXPECT_NE(artifact.find("\"port_rates\":["), std::string_view::npos);
  EXPECT_NE(artifact.find("\"accumulation_horizons\":["), std::string_view::npos);
  EXPECT_NE(artifact.find("\"hard_floor_contributors\":["), std::string_view::npos);
  EXPECT_NE(artifact.find("\"queue_sizing_recipes\":["), std::string_view::npos);
}

// REQ: COMPILE-069
TEST(CompileRatePlan, ProductionCompilePreservesExplicitDepthAndPublishesAdvisoryWarning) {
  Graph graph{"rate-aware-explicit"};
  test::add_compile_test_default_clock(graph);
  auto source = graph.op<FixedRateSource>("source");
  auto sink = graph.op<ClockRateSink>("sink");
  graph.add_flow(source->output, sink->input, ConnectionOptions{.queue_depth = 7U});

  const ExecutionPlan plan = compile(graph);

  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.payload_edges().size(), 1U);
  EXPECT_EQ(plan.payload_edges().front().queue_depth, 7U);
  EXPECT_TRUE(std::ranges::any_of(plan.diagnostics(), [](const auto& diagnostic) {
    return diagnostic.token == "RATE_QUEUE_DEPTH_BELOW_RECOMMENDATION" &&
           diagnostic.severity == holoscan::DiagnosticSeverity::kWarning;
  })) << plan.json();
}

}  // namespace
}  // namespace holoscan::detail
