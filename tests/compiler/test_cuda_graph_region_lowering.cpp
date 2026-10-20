/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <initializer_list>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <holoscan/core/parameter_schema.hpp>
#include <holoscan/time/manual_clock.hpp>

#include "../support/optional_assertions.hpp"
#include "compile_test_graph.hpp"
#include "compiler/cuda_graph_region_lowering.hpp"
#include "compiler/execution_plan_state.hpp"
#include "compiler/plan_artifact_serialization.hpp"
#include "core/canonical_text_encoding.hpp"

namespace holoscan {

struct CaptureDeadlineParams {
  std::chrono::milliseconds budget{1};
};

HOLOSCAN_PARAM_SCHEMA(CaptureDeadlineParams, param_schema("holoscan.test.CaptureDeadlineParams", 1),
                      field<&CaptureDeadlineParams::budget>("budget").recompile_required().range(
                          std::chrono::milliseconds{1}, std::chrono::milliseconds{10}));

namespace {

using namespace std::chrono_literals;

enum class CaptureTrigger : std::uint8_t {
  kDefault,
  kEach,
  kClock,
  kReadable,
  kNotified,
  kDoorbell,
  kSpin,
  kClockWithMaxStartDelay,
  kEachSideInput,
  kAnyInputWithSelector,
  kEachWithSelector,
  kEachWithMaxStartDelay,
};

class CaptureTensorOperator final : public Operator<> {
 public:
  explicit CaptureTensorOperator(CaptureTrigger trigger = CaptureTrigger::kDefault,
                                 bool eligible = true)
      : trigger_(trigger), eligible_(eligible) {}

  void setup(OperatorSpec& spec) override {
    spec.input(input, "input");
    spec.input(side_input, "side_input");
    spec.output(output, "output");
    spec.readable_source(readable, "readable");
    spec.notification_source(notified, "notified").capacity(1).sender_reference_capacity(1);
    spec.doorbell_source(doorbell, "doorbell").capacity(1);
    spec.spin_source(spin, "spin");
    spec.cuda_graph_capture_eligible(eligible_);
  }

  Contract contract() const override {
    Contract result;
    switch (trigger_) {
      case CaptureTrigger::kDefault:
        break;
      case CaptureTrigger::kEach:
        result.trigger(OnEach{input});
        break;
      case CaptureTrigger::kClock:
        result.trigger(OnClock{.period = 1ms});
        break;
      case CaptureTrigger::kReadable:
        result.trigger(OnReadable{readable});
        break;
      case CaptureTrigger::kNotified:
        result.trigger(OnNotified{notified});
        break;
      case CaptureTrigger::kDoorbell:
        result.trigger(OnDoorbell{doorbell});
        break;
      case CaptureTrigger::kSpin:
        result.trigger(OnSpin{spin});
        break;
      case CaptureTrigger::kClockWithMaxStartDelay:
        result.trigger(OnClock{.period = 1ms})
            .max_start_delay(1ms)
            .on_violation(OnViolation::kFailRun);
        break;
      case CaptureTrigger::kEachSideInput:
        result.trigger(OnEach{side_input});
        break;
      case CaptureTrigger::kAnyInputWithSelector:
        result.trigger(OnAnyInput{}).input(input, DrainAll{});
        break;
      case CaptureTrigger::kEachWithSelector:
        result.trigger(OnEach{input}).input(input, Exact{});
        break;
      case CaptureTrigger::kEachWithMaxStartDelay:
        result.trigger(OnEach{input}).max_start_delay(1ms).on_violation(OnViolation::kFailRun);
        break;
    }
    return result;
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Input<Tensor> input;
  Input<Tensor> side_input;
  Output<Tensor> output;
  ReadableSource readable;
  NotificationSource notified;
  DoorbellSource doorbell;
  SpinSource spin;

 private:
  CaptureTrigger trigger_;
  bool eligible_;
};

class CaptureScalarOperator final : public Operator<> {
 public:
  explicit CaptureScalarOperator(CaptureTrigger trigger = CaptureTrigger::kDefault)
      : trigger_(trigger) {}

  void setup(OperatorSpec& spec) override {
    spec.input(input, "input");
    spec.output(output, "output");
    spec.cuda_graph_capture_eligible();
  }

  Contract contract() const override {
    Contract result;
    if (trigger_ == CaptureTrigger::kClock) {
      result.trigger(OnClock{.period = 1ms});
    }
    return result;
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Input<std::uint32_t> input;
  Output<std::uint32_t> output;

 private:
  CaptureTrigger trigger_;
};

class CaptureCompletionSourceOperator final : public Operator<> {
 public:
  explicit CaptureCompletionSourceOperator(CaptureTrigger trigger, std::uint64_t limit = 3U)
      : trigger_(trigger), limit_(limit) {}

  void setup(OperatorSpec& spec) override {
    spec.output(output, "output");
    spec.readable_source(readable, "readable");
    spec.notification_source(notified, "notified").capacity(1).sender_reference_capacity(1);
    spec.doorbell_source(doorbell, "doorbell").capacity(1);
    spec.spin_source(spin, "spin");
    spec.cuda_graph_capture_eligible();
  }

  Contract contract() const override {
    Contract result;
    switch (trigger_) {
      case CaptureTrigger::kClock:
        result.trigger(OnClock{.period = 1ms});
        break;
      case CaptureTrigger::kReadable:
        result.trigger(OnReadable{readable});
        break;
      case CaptureTrigger::kNotified:
        result.trigger(OnNotified{notified});
        break;
      case CaptureTrigger::kDoorbell:
        result.trigger(OnDoorbell{doorbell});
        break;
      case CaptureTrigger::kSpin:
        result.trigger(OnSpin{spin});
        break;
      default:
        break;
    }
    result.complete_after(ComputeCalls{.limit = limit_});
    return result;
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Output<Tensor> output;
  ReadableSource readable;
  NotificationSource notified;
  DoorbellSource doorbell;
  SpinSource spin;

 private:
  CaptureTrigger trigger_;
  std::uint64_t limit_;
};

class CaptureCompletionInputOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(input, "input");
    spec.output(output, "output");
    spec.cuda_graph_capture_eligible();
  }

  Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input}).complete_after(ComputeCalls{.limit = 3U});
    return result;
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Input<Tensor> input;
  Output<Tensor> output;
};

class CaptureParameterDeadlineOperator final : public Operator<CaptureDeadlineParams> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(input, "input");
    spec.cuda_graph_capture_eligible();
  }

  Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input})
        .max_start_delay(param_ref<&CaptureDeadlineParams::budget>())
        .on_violation(OnViolation::kFailRun);
    return result;
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Input<Tensor> input;
};

class CaptureChainGraph final : public Graph {
 public:
  Input<Tensor> input{"input"};
  Output<Tensor> output{"output"};

 protected:
  void compose() override {
    const ClockKey clock = add_clock<ManualClock>("default", std::int64_t{0});
    set_default_clock(clock);
    auto source = op<CaptureTensorOperator>("source", CaptureTrigger::kEach);
    auto sink = op<CaptureTensorOperator>("sink");
    add_flow(source->output, sink->input);
    export_input(input, source->input);
    export_output(output, sink->output);
  }
};

class EmbeddedCaptureRegionGraph final : public Graph {
 protected:
  void compose() override {
    const ClockKey clock = add_clock<ManualClock>("default", std::int64_t{0});
    set_default_clock(clock);
    auto captured = op<CaptureTensorOperator>("captured", CaptureTrigger::kSpin);
    region("nested-capture").add(captured).constrain(RegionConstraint::cuda_graph());
  }
};

void add_default_clock(Graph& graph) {
  const ClockKey clock = graph.add_clock<ManualClock>("default", std::int64_t{0});
  graph.set_default_clock(clock);
}

std::span<const detail::CUDAGraphCapturePlanRecord> cuda_graph_capture_plans(
    const ExecutionPlan& plan) {
  const auto* state = detail::ExecutionPlanAccess::state(plan);
  if (state == nullptr) {
    return {};
  }
  return state->cuda_graph_capture_plans;
}

struct ExpectedRegionRejection {
  std::string_view rule;
  std::string_view region_name;
  std::string_view vertex_name;
  std::string_view port_name;
};

std::string_view vertex_name(const ExecutionPlan& plan, VertexId id) {
  if (!id) {
    return {};
  }
  const auto found = std::find_if(plan.graph().vertices().begin(),
                                  plan.graph().vertices().end(),
                                  [&](const auto& vertex) { return vertex.id == id; });
  return found == plan.graph().vertices().end() ? std::string_view{"<unresolved>"} : found->name;
}

std::string_view port_name(const ExecutionPlan& plan, PortId id) {
  if (!id) {
    return {};
  }
  const auto found = std::find_if(plan.graph().ports().begin(),
                                  plan.graph().ports().end(),
                                  [&](const auto& port) { return port.id.value == id.value(); });
  return found == plan.graph().ports().end() ? std::string_view{"<unresolved>"} : found->name;
}

void expect_region_rejections(const ExecutionPlan& plan,
                              std::initializer_list<ExpectedRegionRejection> expected_rejections) {
  using RejectionEvidence = std::tuple<std::string, std::string, std::string, std::string>;
  std::vector<RejectionEvidence> actual;
  for (const Diagnostic& diagnostic : plan.diagnostics()) {
    if (diagnostic.token == "CUDA_GRAPH_REGION_INVALID") {
      constexpr std::string_view kActualMarker{",actual="};
      const std::size_t marker = diagnostic.provided.find(kActualMarker);
      EXPECT_NE(marker, std::string::npos) << diagnostic.provided;
      if (marker != std::string::npos) {
        EXPECT_LT(marker + kActualMarker.size(), diagnostic.provided.size()) << diagnostic.provided;
      }
      const std::string region = marker == std::string::npos
                                     ? diagnostic.provided
                                     : diagnostic.provided.substr(0U, marker);
      actual.emplace_back(diagnostic.required,
                          region,
                          vertex_name(plan, diagnostic.source.vertex),
                          port_name(plan, diagnostic.source.port));
    }
  }
  std::vector<RejectionEvidence> expected;
  expected.reserve(expected_rejections.size());
  for (const auto& rejection : expected_rejections) {
    const auto region = std::find_if(
        plan.graph().regions().begin(), plan.graph().regions().end(), [&](const auto& candidate) {
          return candidate.name == rejection.region_name;
        });
    if (region == plan.graph().regions().end()) {
      ADD_FAILURE() << "Expected region is absent: " << rejection.region_name;
      continue;
    }
    expected.emplace_back(std::string(rejection.rule),
                          "region_id=" + std::to_string(region->id.value) + ",region_name=" +
                              detail::percent_encode_uri_component(rejection.region_name),
                          std::string(rejection.vertex_name),
                          std::string(rejection.port_name));
  }
  std::sort(actual.begin(), actual.end());
  std::sort(expected.begin(), expected.end());
  EXPECT_EQ(actual, expected) << plan.json();
}

void expect_rejection_actual_contains(const ExecutionPlan& plan, std::string_view rule,
                                      std::string_view expected) {
  const auto diagnostic =
      std::find_if(plan.diagnostics().begin(), plan.diagnostics().end(), [&](const auto& item) {
        return item.token == "CUDA_GRAPH_REGION_INVALID" && item.required == rule;
      });
  ASSERT_NE(diagnostic, plan.diagnostics().end()) << plan.json();
  EXPECT_NE(diagnostic->provided.find(expected), std::string::npos) << diagnostic->provided;
}

// REQ: GRAPH-041
// REQ: COMPILE-079
TEST(CUDAGraphRegionLowering, ProducesSourceToSinkCapturePlanForLinearTensorChain) {
  Graph graph{"linear-capture"};
  add_default_clock(graph);
  auto source = graph.op<CaptureTensorOperator>("source", CaptureTrigger::kClock);
  auto middle = graph.op<CaptureTensorOperator>("middle", CaptureTrigger::kEach);
  auto sink = graph.op<CaptureTensorOperator>("sink");
  graph.add_flow(source->output, middle->input);
  graph.add_flow(middle->output, sink->input);
  graph.region("capture").add(source).add(middle).add(sink).constrain(
      RegionConstraint::cuda_graph());

  const ExecutionPlan plan = compile(graph);
  ASSERT_EQ(cuda_graph_capture_plans(plan).size(), 1U);
  const auto& capture_plan = cuda_graph_capture_plans(plan).front();
  ASSERT_EQ(capture_plan.members.size(), 3U);
  EXPECT_EQ(vertex_name(plan, capture_plan.members[0].vertex), "source");
  EXPECT_EQ(vertex_name(plan, capture_plan.members[1].vertex), "middle");
  EXPECT_EQ(vertex_name(plan, capture_plan.members[2].vertex), "sink");
  EXPECT_FALSE(capture_plan.ingress.has_value());
  EXPECT_FALSE(capture_plan.egress.has_value());
  EXPECT_EQ(capture_plan.contracts.at(0).trigger, TemporalRecordTriggerKind::kClock);
  EXPECT_EQ(capture_plan.contracts.at(1).trigger, TemporalRecordTriggerKind::kEach);
  EXPECT_EQ(capture_plan.contracts.at(2).trigger, TemporalRecordTriggerKind::kAnyInput);
  EXPECT_TRUE(capture_plan.contracts.at(2).synthesized_default);
}

// REQ: GRAPH-041
// REQ: COMPILE-079
TEST(CUDAGraphRegionLowering, PreservesOneTensorIngressAndEgress) {
  Graph graph{"capture-boundary"};
  add_default_clock(graph);
  auto producer = graph.op<CaptureTensorOperator>("producer", CaptureTrigger::kClock, false);
  auto source = graph.op<CaptureTensorOperator>("source");
  auto sink = graph.op<CaptureTensorOperator>("sink");
  auto consumer = graph.op<CaptureTensorOperator>("consumer", CaptureTrigger::kDefault, false);
  graph.add_flow(producer->output, source->input);
  graph.add_flow(source->output, sink->input);
  graph.add_flow(sink->output, consumer->input);
  graph.region("capture").add(source).add(sink).constrain(RegionConstraint::cuda_graph());

  const ExecutionPlan plan = compile(graph);
  ASSERT_EQ(cuda_graph_capture_plans(plan).size(), 1U);
  const auto& capture_plan = cuda_graph_capture_plans(plan).front();
  const auto& ingress = capture_plan.ingress;
  const auto& egress = capture_plan.egress;
  HOLOSCAN_ASSERT_HAS_VALUE(ingress);
  HOLOSCAN_ASSERT_HAS_VALUE(egress);
  EXPECT_EQ(vertex_name(plan, ingress->source), "producer");
  EXPECT_EQ(vertex_name(plan, egress->target), "consumer");
  EXPECT_EQ(capture_plan.contracts.front().trigger, TemporalRecordTriggerKind::kAnyInput);
  EXPECT_TRUE(capture_plan.contracts.front().synthesized_default);
}

// REQ: GRAPH-041
// REQ: COMPILE-079
TEST(CUDAGraphRegionLowering, AcceptsIngressOnlyAndEgressOnlyTensorBoundaries) {
  const auto make_plan = [](bool ingress) {
    Graph graph{ingress ? "tensor-ingress-only" : "tensor-egress-only"};
    add_default_clock(graph);
    auto captured = graph.op<CaptureTensorOperator>(
        "captured", ingress ? CaptureTrigger::kDefault : CaptureTrigger::kClock);
    if (ingress) {
      auto outside = graph.op<CaptureTensorOperator>("producer", CaptureTrigger::kClock, false);
      graph.add_flow(outside->output, captured->input);
    } else {
      auto outside = graph.op<CaptureTensorOperator>("consumer", CaptureTrigger::kDefault, false);
      graph.add_flow(captured->output, outside->input);
    }
    graph.region("capture").add(captured).constrain(RegionConstraint::cuda_graph());
    return compile(graph);
  };

  const ExecutionPlan ingress_only = make_plan(true);
  ASSERT_EQ(cuda_graph_capture_plans(ingress_only).size(), 1U);
  EXPECT_TRUE(cuda_graph_capture_plans(ingress_only).front().ingress.has_value());
  EXPECT_FALSE(cuda_graph_capture_plans(ingress_only).front().egress.has_value());

  const ExecutionPlan egress_only = make_plan(false);
  ASSERT_EQ(cuda_graph_capture_plans(egress_only).size(), 1U);
  EXPECT_FALSE(cuda_graph_capture_plans(egress_only).front().ingress.has_value());
  EXPECT_TRUE(cuda_graph_capture_plans(egress_only).front().egress.has_value());
}

// REQ: GRAPH-040
TEST(CUDAGraphRegionLowering, RejectsEveryOtherRegionConstraintButAllowsSeparateProfile) {
  const std::array combinations{
      std::pair{RegionConstraint::static_topology(), std::string_view{"static_topology"}},
      std::pair{RegionConstraint::bounded_memory(), std::string_view{"bounded_memory"}},
      std::pair{RegionConstraint::device_compatible_ops(),
                std::string_view{"device_compatible_operators"}},
      std::pair{RegionConstraint::latency_budget(1ms), std::string_view{"latency_budget"}},
  };
  for (const auto& [other_constraint, name] : combinations) {
    SCOPED_TRACE(name);
    Graph graph{"combined-cuda-graph-constraint"};
    add_default_clock(graph);
    auto source = graph.op<CaptureTensorOperator>("source", CaptureTrigger::kClock);
    graph.region("capture")
        .add(source)
        .constrain(RegionConstraint::cuda_graph())
        .constrain(other_constraint);

    const ExecutionPlan plan = compile(graph);
    EXPECT_TRUE(cuda_graph_capture_plans(plan).empty()) << plan.json();
    expect_region_rejections(
        plan, {{detail::cuda_graph_region_rule::kExclusiveCUDAGraphConstraint, "capture", {}, {}}});
    expect_rejection_actual_contains(plan,
                                     detail::cuda_graph_region_rule::kExclusiveCUDAGraphConstraint,
                                     "constraint_count=2;other_constraints=" + std::string{name});
  }

  Graph multiple{"multiple-cuda-graph-constraints"};
  add_default_clock(multiple);
  auto multiple_source = multiple.op<CaptureTensorOperator>("source", CaptureTrigger::kClock);
  multiple.region("capture")
      .add(multiple_source)
      .constrain(RegionConstraint::cuda_graph())
      .constrain(RegionConstraint::static_topology())
      .constrain(RegionConstraint::bounded_memory());
  const ExecutionPlan multiple_plan = compile(multiple);
  EXPECT_TRUE(cuda_graph_capture_plans(multiple_plan).empty()) << multiple_plan.json();
  expect_rejection_actual_contains(multiple_plan,
                                   detail::cuda_graph_region_rule::kExclusiveCUDAGraphConstraint,
                                   "constraint_count=3;other_constraints="
                                   "static_topology,bounded_memory");

  Graph profiled{"profiled-cuda-graph-region"};
  add_default_clock(profiled);
  auto source = profiled.op<CaptureTensorOperator>("source", CaptureTrigger::kClock);
  profiled.region("capture")
      .add(source)
      .constrain(RegionConstraint::cuda_graph())
      .profile(Profile::kStandard);
  const ExecutionPlan plan = compile(profiled);
  ASSERT_EQ(cuda_graph_capture_plans(plan).size(), 1U) << plan.json();
  EXPECT_FALSE(test::has_diagnostic(plan, "CUDA_GRAPH_REGION_INVALID"));
}

struct IndependentTriggerCase {
  CaptureTrigger requested;
  TemporalRecordTriggerKind recorded;
};

class CUDAGraphIndependentTriggerTest : public testing::TestWithParam<IndependentTriggerCase> {};

// REQ: GRAPH-041
TEST_P(CUDAGraphIndependentTriggerTest, AcceptsSupportedSourceTrigger) {
  Graph graph{"independent-source"};
  add_default_clock(graph);
  auto source = graph.op<CaptureTensorOperator>("source", GetParam().requested);
  graph.region("capture").add(source).constrain(RegionConstraint::cuda_graph());

  const ExecutionPlan plan = compile(graph);
  ASSERT_EQ(cuda_graph_capture_plans(plan).size(), 1U);
  EXPECT_EQ(cuda_graph_capture_plans(plan).front().contracts.front().trigger, GetParam().recorded);
  EXPECT_FALSE(test::has_diagnostic(plan, "CUDA_GRAPH_REGION_INVALID"));
}

INSTANTIATE_TEST_SUITE_P(
    SupportedTriggers, CUDAGraphIndependentTriggerTest,
    testing::Values(
        IndependentTriggerCase{CaptureTrigger::kClock, TemporalRecordTriggerKind::kClock},
        IndependentTriggerCase{CaptureTrigger::kReadable, TemporalRecordTriggerKind::kReadable},
        IndependentTriggerCase{CaptureTrigger::kNotified, TemporalRecordTriggerKind::kNotified},
        IndependentTriggerCase{CaptureTrigger::kDoorbell, TemporalRecordTriggerKind::kDoorbell},
        IndependentTriggerCase{CaptureTrigger::kSpin, TemporalRecordTriggerKind::kSpin}));

// REQ: CONTRACT-020
// REQ: CONTRACT-024
// REQ: GRAPH-038
// REQ: GRAPH-041
// REQ: COMPILE-079
TEST(CUDAGraphRegionLowering, PreservesSourceCompletionForEverySupportedTrigger) {
  constexpr std::uint64_t kCompletionLimit = 7U;
  constexpr std::array<IndependentTriggerCase, 5U> kCases{{
      {CaptureTrigger::kClock, TemporalRecordTriggerKind::kClock},
      {CaptureTrigger::kReadable, TemporalRecordTriggerKind::kReadable},
      {CaptureTrigger::kNotified, TemporalRecordTriggerKind::kNotified},
      {CaptureTrigger::kDoorbell, TemporalRecordTriggerKind::kDoorbell},
      {CaptureTrigger::kSpin, TemporalRecordTriggerKind::kSpin},
  }};
  for (const auto& trigger : kCases) {
    Graph graph{"independent-source-completion"};
    add_default_clock(graph);
    auto source =
        graph.op<CaptureCompletionSourceOperator>("source", trigger.requested, kCompletionLimit);
    graph.region("capture").add(source).constrain(RegionConstraint::cuda_graph());

    const ExecutionPlan plan = compile(graph);
    ASSERT_EQ(cuda_graph_capture_plans(plan).size(), 1U) << plan.json();
    const auto& capture_plan = cuda_graph_capture_plans(plan).front();
    ASSERT_EQ(capture_plan.contracts.size(), 1U);
    EXPECT_EQ(capture_plan.contracts.front().trigger, trigger.recorded);
    ASSERT_EQ(plan.operator_completion_rules().size(), 1U);
    const auto& completion = plan.operator_completion_rules().front();
    EXPECT_EQ(completion.owner, capture_plan.source);
    EXPECT_EQ(completion.owner, capture_plan.contracts.front().owner);
    EXPECT_EQ(completion.kind, OperatorCompletionRuleKind::kComputeCalls);
    EXPECT_EQ(completion.limit, kCompletionLimit);
    const auto* state = detail::ExecutionPlanAccess::state(plan);
    ASSERT_NE(state, nullptr);
    EXPECT_TRUE(detail::valid_cuda_graph_capture_plans(*state));
    EXPECT_FALSE(test::has_diagnostic(plan, "CUDA_GRAPH_REGION_INVALID"));
  }
}

// REQ: CONTRACT-020
// REQ: CONTRACT-024
// REQ: GRAPH-041
TEST(CUDAGraphRegionLowering, RejectsCompletionOnRegionSourceWithIngress) {
  Graph graph{"input-driven-source-completion"};
  add_default_clock(graph);
  auto producer = graph.op<CaptureTensorOperator>("producer", CaptureTrigger::kClock, false);
  auto source = graph.op<CaptureCompletionInputOperator>("source");
  graph.add_flow(producer->output, source->input);
  graph.region("capture").add(source).constrain(RegionConstraint::cuda_graph());

  const ExecutionPlan plan = compile(graph);
  EXPECT_TRUE(cuda_graph_capture_plans(plan).empty());
  EXPECT_TRUE(test::has_diagnostic(plan, "OPERATOR_COMPLETION_PLACEMENT_UNSUPPORTED"));
  expect_region_rejections(
      plan, {{detail::cuda_graph_region_rule::kSupportedMemberContract, "capture", "source", {}}});
  expect_rejection_actual_contains(plan,
                                   detail::cuda_graph_region_rule::kSupportedMemberContract,
                                   "completion=compute_calls;placement=input_driven_source");
}

// REQ: CONTRACT-020
// REQ: CONTRACT-024
// REQ: GRAPH-041
TEST(CUDAGraphRegionLowering, RejectsCompletionOnCapturedNonSourceMember) {
  Graph graph{"captured-non-source-completion"};
  add_default_clock(graph);
  auto source = graph.op<CaptureTensorOperator>("source", CaptureTrigger::kClock);
  auto sink = graph.op<CaptureCompletionInputOperator>("sink");
  graph.add_flow(source->output, sink->input);
  graph.region("capture").add(source).add(sink).constrain(RegionConstraint::cuda_graph());

  const ExecutionPlan plan = compile(graph);
  EXPECT_TRUE(cuda_graph_capture_plans(plan).empty());
  EXPECT_TRUE(test::has_diagnostic(plan, "OPERATOR_COMPLETION_PLACEMENT_UNSUPPORTED"));
  expect_region_rejections(
      plan, {{detail::cuda_graph_region_rule::kSupportedMemberContract, "capture", "sink", {}}});
  expect_rejection_actual_contains(plan,
                                   detail::cuda_graph_region_rule::kSupportedMemberContract,
                                   "completion=compute_calls;placement=captured_non_source");
}

// REQ: CONTRACT-020
// REQ: CONTRACT-024
// REQ: GRAPH-038
// REQ: COMPILE-079
TEST(CUDAGraphRegionLowering, RejectsCompletionRuleReboundToCapturedNonSource) {
  Graph graph{"malformed-capture-completion"};
  add_default_clock(graph);
  auto source = graph.op<CaptureCompletionSourceOperator>("source", CaptureTrigger::kClock, 7U);
  auto sink = graph.op<CaptureTensorOperator>("sink");
  graph.add_flow(source->output, sink->input);
  graph.region("capture").add(source).add(sink).constrain(RegionConstraint::cuda_graph());

  const ExecutionPlan plan = compile(graph);
  const auto* compiled = detail::ExecutionPlanAccess::state(plan);
  ASSERT_NE(compiled, nullptr);
  ASSERT_EQ(compiled->cuda_graph_capture_plans.size(), 1U) << plan.json();
  ASSERT_EQ(compiled->operator_completion_rules.size(), 1U);
  EXPECT_TRUE(detail::valid_cuda_graph_capture_plans(*compiled));

  detail::ExecutionPlanState malformed = *compiled;
  malformed.operator_completion_rules.front().owner =
      malformed.cuda_graph_capture_plans.front().sink;
  EXPECT_FALSE(detail::valid_cuda_graph_capture_plans(malformed));
}

// REQ: GRAPH-041
TEST(CUDAGraphRegionLowering, RejectsEmptyIneligibleAndOverlappingRegions) {
  Graph graph{"membership-errors"};
  add_default_clock(graph);
  auto ineligible = graph.op<CaptureTensorOperator>("ineligible", CaptureTrigger::kClock, false);
  auto shared = graph.op<CaptureTensorOperator>("shared", CaptureTrigger::kClock);
  graph.region("empty").constrain(RegionConstraint::cuda_graph());
  graph.region("ineligible").add(ineligible).constrain(RegionConstraint::cuda_graph());
  graph.region("first").add(shared).constrain(RegionConstraint::cuda_graph());
  graph.region("second").add(shared).constrain(RegionConstraint::cuda_graph());

  const ExecutionPlan plan = compile(graph);
  EXPECT_TRUE(cuda_graph_capture_plans(plan).empty());
  expect_region_rejections(
      plan,
      {{detail::cuda_graph_region_rule::kNonemptyResolvedOperatorMembership, "empty", {}, {}},
       {detail::cuda_graph_region_rule::kCaptureEligibleOperator, "ineligible", "ineligible", {}},
       {detail::cuda_graph_region_rule::kDisjointCUDAGraphRegions, "first", "shared", {}},
       {detail::cuda_graph_region_rule::kDisjointCUDAGraphRegions, "second", "shared", {}}});
}

// REQ: GRAPH-041
TEST(CUDAGraphRegionLowering, EncodesRegionNameInRejectionEvidence) {
  Graph graph{"encoded-region-name"};
  add_default_clock(graph);
  graph.region("stage,mode=capture").constrain(RegionConstraint::cuda_graph());

  const ExecutionPlan plan = compile(graph);
  expect_region_rejections(plan,
                           {{detail::cuda_graph_region_rule::kNonemptyResolvedOperatorMembership,
                             "stage,mode=capture",
                             {},
                             {}}});
}

// REQ: GRAPH-041
TEST(CUDAGraphRegionLowering, RejectsFanOutTopology) {
  Graph graph{"topology-errors"};
  add_default_clock(graph);
  auto source = graph.op<CaptureTensorOperator>("source", CaptureTrigger::kClock);
  auto left = graph.op<CaptureTensorOperator>("left");
  auto right = graph.op<CaptureTensorOperator>("right");
  graph.add_flow(source->output, left->input);
  graph.add_flow(source->output, right->input);
  graph.region("capture").add(source).add(left).add(right).constrain(
      RegionConstraint::cuda_graph());

  const ExecutionPlan plan = compile(graph);
  EXPECT_TRUE(cuda_graph_capture_plans(plan).empty());
  expect_region_rejections(
      plan,
      {{detail::cuda_graph_region_rule::kSingleLinearOperatorChain, "capture", "source", {}}});
  expect_rejection_actual_contains(plan,
                                   detail::cuda_graph_region_rule::kSingleLinearOperatorChain,
                                   "outgoing_internal_edges=2");
}

// REQ: GRAPH-041
TEST(CUDAGraphRegionLowering, RejectsDisconnectedJoinAndParallelEdges) {
  const auto rejects = [](const std::string& name,
                          auto connect,
                          std::string_view expected_rule,
                          std::string_view expected_vertex = {},
                          std::string_view expected_port = {}) {
    Graph graph{name};
    add_default_clock(graph);
    auto first = graph.op<CaptureTensorOperator>("first", CaptureTrigger::kClock);
    auto second = graph.op<CaptureTensorOperator>("second", CaptureTrigger::kClock);
    auto third = graph.op<CaptureTensorOperator>("third");
    connect(graph, first, second, third);
    graph.region("capture").add(first).add(second).add(third).constrain(
        RegionConstraint::cuda_graph());
    const ExecutionPlan plan = compile(graph);
    EXPECT_TRUE(cuda_graph_capture_plans(plan).empty()) << plan.json();
    expect_region_rejections(plan, {{expected_rule, "capture", expected_vertex, expected_port}});
  };

  rejects(
      "disconnected",
      [](Graph&, const auto&, const auto&, const auto&) {},
      detail::cuda_graph_region_rule::kSingleLinearOperatorChain);
  rejects(
      "join",
      [](Graph& graph, const auto& first, const auto& second, const auto& third) {
        graph.add_flow(first->output, third->input);
        graph.add_flow(second->output, third->side_input);
      },
      detail::cuda_graph_region_rule::kSingleLinearOperatorChain,
      "third");
  rejects(
      "parallel",
      [](Graph& graph, const auto& first, const auto& second, const auto& third) {
        graph.add_flow(first->output, second->input);
        graph.add_flow(first->output, second->side_input);
        graph.add_flow(second->output, third->input);
      },
      detail::cuda_graph_region_rule::kSingleEdgePerOperatorHop,
      "first",
      "output");
}

// REQ: GRAPH-006
TEST(CUDAGraphRegionLowering, GraphRejectsCycleDuringFlowConstruction) {
  Graph graph{"cycle"};
  auto first = graph.op<CaptureTensorOperator>("first", CaptureTrigger::kClock);
  auto second = graph.op<CaptureTensorOperator>("second");
  graph.add_flow(first->output, second->input);
  EXPECT_THROW(graph.add_flow(second->output, first->input), RuntimeError);
}

// REQ: GRAPH-041
TEST(CUDAGraphRegionLowering, RejectsBoundaryEdgeFromIntermediateOperator) {
  Graph graph{"side-output"};
  add_default_clock(graph);
  auto source = graph.op<CaptureTensorOperator>("source", CaptureTrigger::kClock);
  auto middle = graph.op<CaptureTensorOperator>("middle");
  auto sink = graph.op<CaptureTensorOperator>("sink");
  auto outside = graph.op<CaptureTensorOperator>("outside", CaptureTrigger::kDefault, false);
  graph.add_flow(source->output, middle->input);
  graph.add_flow(middle->output, sink->input);
  graph.add_flow(middle->output, outside->input);
  graph.region("capture").add(source).add(middle).add(sink).constrain(
      RegionConstraint::cuda_graph());

  const ExecutionPlan plan = compile(graph);
  EXPECT_TRUE(cuda_graph_capture_plans(plan).empty());
  expect_region_rejections(plan,
                           {{detail::cuda_graph_region_rule::kRegionEdgesOnlyAtSourceAndSink,
                             "capture",
                             "middle",
                             "output"}});
}

// REQ: GRAPH-041
TEST(CUDAGraphRegionLowering, RejectsMultipleIngressAndMultipleEgressIndependently) {
  const auto rejects = [](bool multiple_ingress) {
    Graph graph{multiple_ingress ? "multiple-ingress" : "multiple-egress"};
    add_default_clock(graph);
    auto captured = graph.op<CaptureTensorOperator>(
        "captured", multiple_ingress ? CaptureTrigger::kEach : CaptureTrigger::kClock);
    if (multiple_ingress) {
      auto first = graph.op<CaptureTensorOperator>("first-producer", CaptureTrigger::kClock, false);
      auto second =
          graph.op<CaptureTensorOperator>("second-producer", CaptureTrigger::kClock, false);
      graph.add_flow(first->output, captured->input);
      graph.add_flow(second->output, captured->side_input);
    } else {
      auto first =
          graph.op<CaptureTensorOperator>("first-consumer", CaptureTrigger::kDefault, false);
      auto second =
          graph.op<CaptureTensorOperator>("second-consumer", CaptureTrigger::kDefault, false);
      graph.add_flow(captured->output, first->input);
      graph.add_flow(captured->output, second->input);
    }
    graph.region("capture").add(captured).constrain(RegionConstraint::cuda_graph());
    const ExecutionPlan plan = compile(graph);
    EXPECT_TRUE(cuda_graph_capture_plans(plan).empty());
    expect_region_rejections(
        plan,
        {{detail::cuda_graph_region_rule::kAtMostOneRegionIngressAndEgress, "capture", {}, {}}});
  };

  rejects(true);
  rejects(false);
}

// REQ: GRAPH-041
TEST(CUDAGraphRegionLowering, RejectsIngressToIntermediateOperator) {
  Graph graph{"side-input"};
  add_default_clock(graph);
  auto source = graph.op<CaptureTensorOperator>("source", CaptureTrigger::kClock);
  auto middle = graph.op<CaptureTensorOperator>("middle");
  auto sink = graph.op<CaptureTensorOperator>("sink");
  auto outside = graph.op<CaptureTensorOperator>("outside", CaptureTrigger::kClock, false);
  graph.add_flow(source->output, middle->input);
  graph.add_flow(middle->output, sink->input);
  graph.add_flow(outside->output, middle->side_input);
  graph.region("capture").add(source).add(middle).add(sink).constrain(
      RegionConstraint::cuda_graph());

  const ExecutionPlan plan = compile(graph);
  EXPECT_TRUE(cuda_graph_capture_plans(plan).empty());
  expect_region_rejections(plan,
                           {{detail::cuda_graph_region_rule::kRegionEdgesOnlyAtSourceAndSink,
                             "capture",
                             "middle",
                             "side_input"}});
}

// REQ: GRAPH-041
TEST(CUDAGraphRegionLowering, RejectsEmbeddedGraphMember) {
  Graph graph{"embedded-capture"};
  add_default_clock(graph);
  auto producer = graph.op<CaptureTensorOperator>("producer", CaptureTrigger::kClock, false);
  auto captured = graph.embed<CaptureChainGraph>("captured");
  auto consumer = graph.op<CaptureTensorOperator>("consumer", CaptureTrigger::kDefault, false);
  graph.add_flow(producer->output, captured->input);
  graph.add_flow(captured->output, consumer->input);
  graph.region("capture").add(captured).constrain(RegionConstraint::cuda_graph());

  const ExecutionPlan plan = compile(graph);
  EXPECT_TRUE(cuda_graph_capture_plans(plan).empty());
  expect_region_rejections(
      plan,
      {{detail::cuda_graph_region_rule::kDirectOperatorMembersInRegionGraph, "capture", {}, {}}});
}

// REQ: GRAPH-041
TEST(CUDAGraphRegionLowering, RejectsRegionDeclaredInsideEmbeddedGraph) {
  Graph graph{"embedded-cuda-graph-region"};
  add_default_clock(graph);
  graph.embed<EmbeddedCaptureRegionGraph>("nested");

  const ExecutionPlan plan = compile(graph);
  EXPECT_TRUE(cuda_graph_capture_plans(plan).empty());
  expect_region_rejections(
      plan, {{detail::cuda_graph_region_rule::kRootGraphRegion, "nested-capture", {}, {}}});
}

// REQ: GRAPH-041
TEST(CUDAGraphRegionLowering, RejectsDirectOperatorMemberOwnedByEmbeddedGraph) {
  Graph graph{"cross-graph-operator-member"};
  auto captured = graph.embed<CaptureChainGraph>("captured");
  graph.region("capture").add(captured).constrain(RegionConstraint::cuda_graph());

  auto snapshot = *detail::GraphCompilerAccess::canonical_snapshot(graph);
  ASSERT_EQ(snapshot.regions.size(), 1U);
  auto& region = snapshot.regions.front();
  const auto nested_operator =
      std::find_if(snapshot.vertices.begin(), snapshot.vertices.end(), [&](const auto& vertex) {
        return vertex.kind == VertexKind::kOperator && vertex.graph != region.graph;
      });
  ASSERT_NE(nested_operator, snapshot.vertices.end());

  // Exercise the same-Graph check independently of the public embedded-Graph member declaration.
  region.graph_members.clear();
  region.members = {nested_operator->id};
  const auto analysis = detail::analyze_cuda_graph_regions(snapshot);
  ASSERT_EQ(analysis.errors.size(), 1U);
  EXPECT_EQ(analysis.errors.front().required,
            detail::cuda_graph_region_rule::kDirectOperatorMembersInRegionGraph);
  EXPECT_FALSE(analysis.errors.front().source.vertex);
  EXPECT_TRUE(analysis.regions.empty());
}

// REQ: GRAPH-038
// REQ: GRAPH-041
TEST(CUDAGraphRegionLowering, RejectsOnEachNamingInputOutsideRegionTopology) {
  Graph graph{"wrong-trigger-input"};
  add_default_clock(graph);
  auto producer = graph.op<CaptureTensorOperator>("producer", CaptureTrigger::kClock, false);
  auto captured = graph.op<CaptureTensorOperator>("captured", CaptureTrigger::kEachSideInput);
  graph.add_flow(producer->output, captured->input);
  graph.region("capture").add(captured).constrain(RegionConstraint::cuda_graph());

  const ExecutionPlan plan = compile(graph);
  EXPECT_TRUE(cuda_graph_capture_plans(plan).empty());
  expect_region_rejections(plan,
                           {{detail::cuda_graph_region_rule::kSupportedMemberContract,
                             "capture",
                             "captured",
                             "side_input"}});
  expect_rejection_actual_contains(plan,
                                   detail::cuda_graph_region_rule::kSupportedMemberContract,
                                   "each_input=side_input;selected_port_id=");
  expect_rejection_actual_contains(
      plan, detail::cuda_graph_region_rule::kSupportedMemberContract, ";expected_port_id=");
}

// REQ: GRAPH-038
// REQ: GRAPH-041
TEST(CUDAGraphRegionLowering, FallsBackToTopologyInputForUnresolvedOnEachName) {
  Graph graph{"unresolved-trigger-input"};
  add_default_clock(graph);
  auto producer = graph.op<CaptureTensorOperator>("producer", CaptureTrigger::kClock, false);
  auto captured = graph.op<CaptureTensorOperator>("captured", CaptureTrigger::kEach);
  graph.add_flow(producer->output, captured->input);
  graph.region("capture").add(captured).constrain(RegionConstraint::cuda_graph());

  const ExecutionPlan plan = compile(graph);
  const auto* compiled = detail::ExecutionPlanAccess::state(plan);
  ASSERT_NE(compiled, nullptr);
  const auto snapshot = detail::GraphCompilerAccess::retain_snapshot(compiled->graph);
  ASSERT_NE(snapshot, nullptr);
  auto contracts = detail::GraphCompilerAccess::replay_temporal_contracts(
      compiled->executable_factories, *snapshot, compiled->plan_generation, compiled->readiness);
  ASSERT_TRUE(contracts);
  const auto replay = std::find_if(contracts->begin(), contracts->end(), [](const auto& item) {
    return item.operator_name == "captured";
  });
  ASSERT_NE(replay, contracts->end());
  const auto expected_input =
      std::find_if(replay->ports.begin(), replay->ports.end(), [](const auto& port) {
        return port.direction == PortDirection::kInput && port.name == "input";
      });
  ASSERT_NE(expected_input, replay->ports.end());
  replay->contract.trigger.port_name = "missing_input";

  const auto analysis = detail::analyze_cuda_graph_regions(*snapshot);
  ASSERT_TRUE(analysis.errors.empty());
  ASSERT_EQ(analysis.regions.size(), 1U);
  const auto lowering =
      detail::lower_cuda_graph_regions(analysis.regions, compiled->plan_generation, *contracts);
  ASSERT_EQ(lowering.errors.size(), 1U);
  EXPECT_EQ(lowering.errors.front().required,
            detail::cuda_graph_region_rule::kSupportedMemberContract);
  EXPECT_EQ(lowering.errors.front().source.port, expected_input->id);
  EXPECT_NE(lowering.errors.front().provided.find(
                "each_input=missing_input;selected_port_id=unresolved;expected_port_id="),
            std::string::npos)
      << lowering.errors.front().provided;
}

// REQ: GRAPH-041
TEST(CUDAGraphRegionLowering, RejectsExplicitInputSelector) {
  Graph graph{"explicit-input-selector"};
  add_default_clock(graph);
  auto producer = graph.op<CaptureTensorOperator>("producer", CaptureTrigger::kClock, false);
  auto captured =
      graph.op<CaptureTensorOperator>("captured", CaptureTrigger::kAnyInputWithSelector);
  graph.add_flow(producer->output, captured->input);
  graph.region("capture").add(captured).constrain(RegionConstraint::cuda_graph());

  const ExecutionPlan plan = compile(graph);
  EXPECT_TRUE(cuda_graph_capture_plans(plan).empty());
  expect_region_rejections(
      plan,
      {{detail::cuda_graph_region_rule::kSupportedMemberContract, "capture", "captured", "input"}});
}

// REQ: GRAPH-041
TEST(CUDAGraphRegionLowering, RejectsInputDrivenMemberDeadline) {
  Graph graph{"contract-errors"};
  add_default_clock(graph);
  auto producer = graph.op<CaptureTensorOperator>("producer", CaptureTrigger::kClock, false);
  auto source = graph.op<CaptureTensorOperator>("source", CaptureTrigger::kEachWithMaxStartDelay);
  graph.add_flow(producer->output, source->input);
  graph.region("capture").add(source).constrain(RegionConstraint::cuda_graph());

  const ExecutionPlan plan = compile(graph);
  EXPECT_TRUE(cuda_graph_capture_plans(plan).empty());
  expect_region_rejections(
      plan, {{detail::cuda_graph_region_rule::kNoMemberMaxStartDelay, "capture", "source", ""}});
}

// REQ: GRAPH-041
TEST(CUDAGraphRegionLowering, RejectsIndependentSourceDeadline) {
  Graph graph{"source-deadline"};
  add_default_clock(graph);
  auto source = graph.op<CaptureTensorOperator>("source", CaptureTrigger::kClockWithMaxStartDelay);
  graph.region("capture").add(source).constrain(RegionConstraint::cuda_graph());

  const ExecutionPlan plan = compile(graph);
  EXPECT_TRUE(cuda_graph_capture_plans(plan).empty());
  expect_region_rejections(
      plan, {{detail::cuda_graph_region_rule::kNoMemberMaxStartDelay, "capture", "source", ""}});
}

// REQ: GRAPH-041
TEST(CUDAGraphRegionLowering, RejectsParameterDerivedMemberDeadline) {
  Graph graph{"parameter-deadline"};
  add_default_clock(graph);
  auto producer = graph.op<CaptureTensorOperator>("producer", CaptureTrigger::kClock, false);
  auto captured = graph.op<CaptureParameterDeadlineOperator>("captured");
  graph.add_flow(producer->output, captured->input);
  graph.region("capture").add(captured).constrain(RegionConstraint::cuda_graph());

  const ExecutionPlan plan = compile(graph);
  EXPECT_TRUE(cuda_graph_capture_plans(plan).empty());
  expect_region_rejections(
      plan, {{detail::cuda_graph_region_rule::kNoMemberMaxStartDelay, "capture", "captured", ""}});
}

// REQ: GRAPH-041
TEST(CUDAGraphRegionLowering, RejectsIndependentTriggerInsideChain) {
  Graph graph{"independent-internal-trigger"};
  add_default_clock(graph);
  auto source = graph.op<CaptureTensorOperator>("source", CaptureTrigger::kClock);
  auto sink = graph.op<CaptureTensorOperator>("sink", CaptureTrigger::kClock);
  graph.add_flow(source->output, sink->input);
  graph.region("capture").add(source).add(sink).constrain(RegionConstraint::cuda_graph());

  const ExecutionPlan plan = compile(graph);
  EXPECT_TRUE(cuda_graph_capture_plans(plan).empty()) << plan.json();
  expect_region_rejections(
      plan,
      {{detail::cuda_graph_region_rule::kSupportedMemberContract, "capture", "sink", "input"}});
  expect_rejection_actual_contains(plan,
                                   detail::cuda_graph_region_rule::kSupportedMemberContract,
                                   "input_driven_member_trigger=clock");
}

// REQ: GRAPH-041
TEST(CUDAGraphRegionLowering, RejectsMemberWithoutNormalizedContract) {
  Graph graph{"missing-normalized-contract"};
  add_default_clock(graph);
  auto source = graph.op<CaptureTensorOperator>("source");
  auto sink = graph.op<CaptureTensorOperator>("sink");
  graph.add_flow(source->output, sink->input);
  graph.region("capture").add(source).add(sink).constrain(RegionConstraint::cuda_graph());

  const ExecutionPlan plan = compile(graph);
  EXPECT_TRUE(cuda_graph_capture_plans(plan).empty()) << plan.json();
  expect_region_rejections(
      plan, {{detail::cuda_graph_region_rule::kNormalizedMemberContract, "capture", "source", {}}});
}

// REQ: GRAPH-041
TEST(CUDAGraphRegionLowering, PublishesStructuralErrorsWhenTemporalReplayFails) {
  Graph graph{"invalid-temporal-contract"};
  add_default_clock(graph);
  auto source = graph.op<CaptureTensorOperator>("source", CaptureTrigger::kClock);
  auto sink = graph.op<CaptureTensorOperator>("sink", CaptureTrigger::kEachWithSelector);
  graph.add_flow(source->output, sink->input);
  graph.region("empty").constrain(RegionConstraint::cuda_graph());
  graph.region("capture").add(source).add(sink).constrain(RegionConstraint::cuda_graph());

  const ExecutionPlan plan = compile(graph);
  EXPECT_TRUE(cuda_graph_capture_plans(plan).empty()) << plan.json();
  EXPECT_TRUE(test::has_diagnostic(plan, "TEMPORAL_CONTRACT_INVALID")) << plan.json();
  expect_region_rejections(
      plan,
      {{detail::cuda_graph_region_rule::kNonemptyResolvedOperatorMembership, "empty", {}, {}}});
}

// REQ: GRAPH-041
TEST(CUDAGraphRegionLowering, RejectsNonTensorInternalEdge) {
  Graph graph{"non-tensor-edge"};
  add_default_clock(graph);
  auto source = graph.op<CaptureScalarOperator>("source", CaptureTrigger::kClock);
  auto sink = graph.op<CaptureScalarOperator>("sink");
  graph.add_flow(source->output, sink->input);
  graph.region("capture").add(source).add(sink).constrain(RegionConstraint::cuda_graph());

  const ExecutionPlan plan = compile(graph);
  EXPECT_TRUE(cuda_graph_capture_plans(plan).empty());
  expect_region_rejections(
      plan,
      {{detail::cuda_graph_region_rule::kTensorInternalEdges, "capture", "source", "output"}});
  expect_rejection_actual_contains(
      plan, detail::cuda_graph_region_rule::kTensorInternalEdges, "source_payload=scalar");
}

// REQ: GRAPH-041
TEST(CUDAGraphRegionLowering, RejectsNonTensorIngressAndEgressIndependently) {
  const auto rejects = [](bool ingress) {
    Graph graph{ingress ? "non-tensor-ingress" : "non-tensor-egress"};
    add_default_clock(graph);
    auto captured = graph.op<CaptureScalarOperator>(
        "captured", ingress ? CaptureTrigger::kDefault : CaptureTrigger::kClock);
    if (ingress) {
      auto outside = graph.op<CaptureScalarOperator>("producer", CaptureTrigger::kClock);
      graph.add_flow(outside->output, captured->input);
    } else {
      auto outside = graph.op<CaptureScalarOperator>("consumer");
      graph.add_flow(captured->output, outside->input);
    }
    graph.region("capture").add(captured).constrain(RegionConstraint::cuda_graph());
    const ExecutionPlan plan = compile(graph);
    EXPECT_TRUE(cuda_graph_capture_plans(plan).empty());
    expect_region_rejections(plan,
                             {{detail::cuda_graph_region_rule::kTensorRegionBoundaryEdges,
                               "capture",
                               ingress ? "producer" : "captured",
                               "output"}});
  };

  rejects(true);
  rejects(false);
}

// REQ: COMPILE-002
// REQ: COMPILE-079
TEST(CUDAGraphRegionLowering, CapturePlanAndDigestDoNotDependOnRegionMemberInsertionOrder) {
  const auto make_plan = [](bool reverse_members) {
    Graph graph{"deterministic-capture"};
    add_default_clock(graph);
    auto source = graph.op<CaptureTensorOperator>("source", CaptureTrigger::kClock);
    auto middle = graph.op<CaptureTensorOperator>("middle");
    auto sink = graph.op<CaptureTensorOperator>("sink");
    graph.add_flow(source->output, middle->input);
    graph.add_flow(middle->output, sink->input);
    auto region = graph.region("capture");
    if (reverse_members) {
      region.add(sink).add(middle).add(source);
    } else {
      region.add(source).add(middle).add(sink);
    }
    region.constrain(RegionConstraint::cuda_graph());
    return compile(graph);
  };

  const ExecutionPlan forward = make_plan(false);
  const ExecutionPlan reverse = make_plan(true);
  ASSERT_EQ(cuda_graph_capture_plans(forward).size(), 1U);
  ASSERT_EQ(cuda_graph_capture_plans(reverse).size(), 1U);
  EXPECT_EQ(forward.digest(), reverse.digest());
  EXPECT_EQ(forward.json(), reverse.json());
}

// REQ: COMPILE-079
TEST(CUDAGraphRegionLowering, ArtifactValidationRejectsMalformedCapturePlan) {
  Graph graph{"malformed-capture-plan"};
  add_default_clock(graph);
  auto source = graph.op<CaptureTensorOperator>("source", CaptureTrigger::kClock);
  auto middle = graph.op<CaptureTensorOperator>("middle", CaptureTrigger::kEach);
  auto sink = graph.op<CaptureTensorOperator>("sink");
  graph.add_flow(source->output, middle->input);
  graph.add_flow(middle->output, sink->input);
  graph.region("capture").add(source).add(middle).add(sink).constrain(
      RegionConstraint::cuda_graph());
  const ExecutionPlan plan = compile(graph);
  const auto* compiled = detail::ExecutionPlanAccess::state(plan);
  ASSERT_NE(compiled, nullptr);
  ASSERT_EQ(compiled->cuda_graph_capture_plans.size(), 1U);
  ASSERT_EQ(compiled->cuda_graph_capture_plans.front().contracts.size(), 3U);

  const auto expect_rejected = [](detail::ExecutionPlanState& malformed) {
    EXPECT_FALSE(detail::valid_cuda_graph_capture_plans(malformed));
    auto artifact = detail::finalize_plan_artifact(malformed);
    ASSERT_FALSE(artifact);
    EXPECT_EQ(artifact.error().message, "CUDA_GRAPH_CAPTURE_PLAN_INVALID");
  };

  detail::ExecutionPlanState malformed = *compiled;
  malformed.cuda_graph_capture_plans.front().members.clear();
  expect_rejected(malformed);

  malformed = *compiled;
  auto& member = malformed.cuda_graph_capture_plans.front().members.front();
  member.vertex = VertexId{member.vertex.value + 1U};
  expect_rejected(malformed);

  // Another supported source trigger is still invalid when it differs from the temporal plan.
  malformed = *compiled;
  malformed.cuda_graph_capture_plans.front().contracts.front().trigger =
      TemporalRecordTriggerKind::kSpin;
  expect_rejected(malformed);

  // The reduced record must also preserve whether Core synthesized the default contract.
  malformed = *compiled;
  malformed.cuda_graph_capture_plans.front().contracts.back().synthesized_default = false;
  expect_rejected(malformed);

  // The stored trigger input must agree with both the region topology and the temporal plan.
  malformed = *compiled;
  const OperatorRef middle_owner = malformed.cuda_graph_capture_plans.front().contracts[1].owner;
  const auto middle_trigger =
      std::find_if(malformed.temporal_triggers.begin(),
                   malformed.temporal_triggers.end(),
                   [middle_owner](const auto& trigger) { return trigger.owner == middle_owner; });
  ASSERT_NE(middle_trigger, malformed.temporal_triggers.end());
  middle_trigger->trigger_input = {};
  EXPECT_FALSE(detail::valid_cuda_graph_capture_plans(malformed));
}

// REQ: GRAPH-038
// REQ: COMPILE-079
TEST(CUDAGraphRegionLowering, ArtifactValidationRejectsMissingAndSubstitutedBoundaries) {
  Graph graph{"capture-boundary-completeness"};
  add_default_clock(graph);

  auto first_producer =
      graph.op<CaptureTensorOperator>("first-producer", CaptureTrigger::kClock, false);
  auto first_captured = graph.op<CaptureTensorOperator>("first-captured");
  auto first_consumer =
      graph.op<CaptureTensorOperator>("first-consumer", CaptureTrigger::kDefault, false);
  graph.add_flow(first_producer->output, first_captured->input);
  graph.add_flow(first_captured->output, first_consumer->input);
  graph.region("first-capture").add(first_captured).constrain(RegionConstraint::cuda_graph());

  auto second_producer =
      graph.op<CaptureTensorOperator>("second-producer", CaptureTrigger::kClock, false);
  auto second_captured = graph.op<CaptureTensorOperator>("second-captured");
  auto second_consumer =
      graph.op<CaptureTensorOperator>("second-consumer", CaptureTrigger::kDefault, false);
  graph.add_flow(second_producer->output, second_captured->input);
  graph.add_flow(second_captured->output, second_consumer->input);
  graph.region("second-capture").add(second_captured).constrain(RegionConstraint::cuda_graph());

  const ExecutionPlan plan = compile(graph);
  const auto* compiled = detail::ExecutionPlanAccess::state(plan);
  ASSERT_NE(compiled, nullptr);
  ASSERT_EQ(compiled->cuda_graph_capture_plans.size(), 2U) << plan.json();
  ASSERT_TRUE(compiled->cuda_graph_capture_plans[0].ingress);
  ASSERT_TRUE(compiled->cuda_graph_capture_plans[0].egress);
  ASSERT_TRUE(compiled->cuda_graph_capture_plans[1].ingress);
  ASSERT_TRUE(compiled->cuda_graph_capture_plans[1].egress);
  EXPECT_TRUE(detail::valid_cuda_graph_capture_plans(*compiled));

  const auto expect_rejected = [&](auto mutate) {
    detail::ExecutionPlanState malformed = *compiled;
    mutate(malformed.cuda_graph_capture_plans);
    EXPECT_FALSE(detail::valid_cuda_graph_capture_plans(malformed));
    auto artifact = detail::finalize_plan_artifact(malformed);
    ASSERT_FALSE(artifact);
    EXPECT_EQ(artifact.error().message, "CUDA_GRAPH_CAPTURE_PLAN_INVALID");
  };

  expect_rejected([](auto& plans) { plans[0].ingress.reset(); });
  expect_rejected([](auto& plans) { plans[0].egress.reset(); });
  expect_rejected([](auto& plans) { plans[0].ingress = plans[1].ingress; });
  expect_rejected([](auto& plans) { plans[0].egress = plans[1].egress; });
}

// REQ: COMPILE-079
TEST(CUDAGraphRegionLowering, ArtifactValidationRejectsUnorderedAndDuplicateCapturePlans) {
  Graph graph{"capture-plan-order"};
  add_default_clock(graph);
  auto first = graph.op<CaptureTensorOperator>("first", CaptureTrigger::kClock);
  auto second = graph.op<CaptureTensorOperator>("second", CaptureTrigger::kClock);
  graph.region("first-region").add(first).constrain(RegionConstraint::cuda_graph());
  graph.region("second-region").add(second).constrain(RegionConstraint::cuda_graph());
  const ExecutionPlan plan = compile(graph);
  const auto* compiled = detail::ExecutionPlanAccess::state(plan);
  ASSERT_NE(compiled, nullptr);
  ASSERT_EQ(compiled->cuda_graph_capture_plans.size(), 2U);

  detail::ExecutionPlanState malformed = *compiled;
  std::swap(malformed.cuda_graph_capture_plans[0], malformed.cuda_graph_capture_plans[1]);
  EXPECT_FALSE(detail::valid_cuda_graph_capture_plans(malformed));

  malformed = *compiled;
  malformed.cuda_graph_capture_plans[1] = malformed.cuda_graph_capture_plans[0];
  EXPECT_FALSE(detail::valid_cuda_graph_capture_plans(malformed));
}

// REQ: GRAPH-041
TEST(CUDAGraphRegionLowering, ReportsUnresolvedApplicationEdgeAtGraphScopeOnce) {
  Graph graph{"unresolved-application-edge"};
  add_default_clock(graph);
  auto capture = graph.op<CaptureTensorOperator>("capture", CaptureTrigger::kClock);
  auto outside_source = graph.op<CaptureTensorOperator>("outside-source", CaptureTrigger::kClock);
  auto outside_sink = graph.op<CaptureTensorOperator>("outside-sink");
  graph.add_flow(outside_source->output, outside_sink->input);
  graph.region("capture").add(capture).constrain(RegionConstraint::cuda_graph());

  auto snapshot = *detail::GraphCompilerAccess::canonical_snapshot(graph);
  ASSERT_EQ(snapshot.edges.size(), 1U);
  snapshot.edges.front().source = GraphPortId{snapshot.ports.size() + 1U};

  const auto analysis = detail::analyze_cuda_graph_regions(snapshot);
  ASSERT_EQ(analysis.errors.size(), 1U);
  EXPECT_EQ(analysis.errors.front().token, "CUDA_GRAPH_REGION_ANALYSIS_UNAVAILABLE");
  EXPECT_EQ(analysis.errors.front().required,
            detail::cuda_graph_analysis_rule::kResolvedGraphEdges);
  EXPECT_FALSE(analysis.errors.front().source.vertex);
  EXPECT_TRUE(analysis.regions.empty());
}

}  // namespace
}  // namespace holoscan
