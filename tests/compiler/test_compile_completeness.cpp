/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>

#include <holoscan/time/manual_clock.hpp>

#include "compile_test_graph.hpp"
#include "compiler/execution_plan_state.hpp"
#include "compiler/process_local_channel_plan_validation.hpp"

namespace holoscan {
namespace {

// Composable Graph that chains two CompileTestOperators internally and exports its
// boundary ports as public Input/Output. Used in
// EmbeddedPlanRefsOutliveParentAndChildAuthoringObjects.
//
//   [input] --> first --> second --> [output]
class EmbeddedCompileGraph final : public Graph {
 public:
  [[nodiscard]] ClockKey clock_import() const noexcept { return clock_import_; }

  Input<std::uint32_t> input{"input"};
  Output<std::uint32_t> output{"output"};

 protected:
  void compose() override {
    clock_import_ = import_clock("timeline");
    set_default_clock(clock_import_);
    const auto first = op<test::CompileTestOperator>("first");
    const auto second = op<test::CompileTestOperator>("second");
    add_flow(first->output, second->input);
    export_input(input, first->input);
    export_output(output, second->output);
    lifecycle().before(first, second, LifecycleStage::kStart);
  }

 private:
  ClockKey clock_import_;
};

// "M1 hero" operator: queue_depth(4) input, max_emits_per_compute(2) output, and a
// DrainAll input selector declared via Contract. Used in compile_hero_graph() and
// all CompileCompleteness M1Hero tests.
class HeroCompileOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(input, "input").queue_depth(4);
    spec.output(output, "output").max_emits_per_compute(2);
  }

  Contract contract() const override {
    Contract result;
    result.trigger(OnAnyInput{}).input(input, DrainAll{});
    return result;
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Input<std::uint32_t> input;
  Output<std::uint32_t> output;
};

// REQ: COMPILE-002
// REQ: RUNTIME-015
TEST(CompileCompleteness, EmbeddedPlanRefsOutliveParentAndChildAuthoringObjects) {
  ExecutionPlan retained;
  std::string retained_json;
  {
    // Key setup: parent and child authoring objects are destroyed at end of this block;
    // the compiled plan must outlive them.
    //
    //   source --> [child: first --> second] --> target
    //
    auto parent = std::make_unique<Graph>("embedded-plan-lifetime");
    const ClockKey parent_clock = test::add_compile_test_default_clock(*parent);
    const auto source = parent->op<test::CompileTestOperator>("source");
    const auto target = parent->op<test::CompileTestOperator>("target");
    const auto child = parent->embed<EmbeddedCompileGraph>("child");
    parent->bind_clock(child, child->clock_import(), parent_clock);
    parent->add_flow(source->output, child->input);
    parent->add_flow(child->output, target->input);

    retained = compile(*parent);
    ASSERT_TRUE(retained.ok()) << retained.json();
    retained_json = retained.json();
    ASSERT_EQ(retained.graph().graphs().size(), 2U);
    ASSERT_EQ(retained.graph().interface_bindings().size(), 2U);
    const auto* state = detail::ExecutionPlanAccess::state(retained);
    ASSERT_NE(state, nullptr);
    ASSERT_TRUE(detail::valid_process_local_channel_plan(*state));
    ASSERT_EQ(state->payload_edges.size(), retained.graph().edges().size());
    ASSERT_EQ(state->process_local_channel_subscribers.size(), state->payload_edges.size());
    const auto operator_owned_port = [&](PortId id) {
      return id && id.value() <= state->ports.size() &&
             state->ports[static_cast<std::size_t>(id.value() - 1U)].owner_kind ==
                 PortOwnerKind::kOperator;
    };
    for (const auto& edge : state->payload_edges) {
      EXPECT_TRUE(operator_owned_port(edge.source));
      EXPECT_TRUE(operator_owned_port(edge.target));
    }
    for (const auto& binding : state->interface_port_bindings) {
      EXPECT_TRUE(std::none_of(
          state->payload_edges.begin(), state->payload_edges.end(), [&](const auto& edge) {
            return edge.source == binding.public_port || edge.target == binding.public_port;
          }));
    }
    ASSERT_FALSE(retained.lifecycle_dependencies().empty());
    for (const auto& dependency : retained.lifecycle_dependencies()) {
      EXPECT_TRUE(dependency.before.plan_bound());
      EXPECT_TRUE(dependency.after.plan_bound());
      EXPECT_EQ(dependency.before.run_generation, 0U);
      EXPECT_EQ(dependency.after.run_generation, 0U);
    }
  }

  // Key assertion: plan remains valid and byte-identical after parent/child objects
  // are destroyed.
  EXPECT_TRUE(retained.ok());
  EXPECT_EQ(retained.json(), retained_json);
  ASSERT_EQ(retained.graph().graphs().size(), 2U);
  EXPECT_EQ(retained.graph().interface_bindings().size(), 2U);
  EXPECT_FALSE(retained.graph().vertices().empty());
  EXPECT_FALSE(retained.graph().ports().empty());
}

// Builds the full "M1 hero" integration graph: two ManualClocks with an affine
// conversion, an imported topic, two HeroCompileOperators, an exported topic, a
// bounded-memory region, and lifecycle ordering.
//
//   imported --[depth=4]--> first --[depth=4]--> second --[depth=4]--> exported
//
// Shared by M1HeroIsByteDeterministicAndRejectedPlansRemainInspectable and
// M1HeroTraversesEveryCompileOwnedPipelineFamily.
[[nodiscard]] ExecutionPlan compile_hero_graph() {
  Graph graph{"holoscan-m1-hero"};
  const ClockKey capture_clock = graph.add_clock<ManualClock>("capture", std::int64_t{0});
  const ClockKey plan_clock = graph.add_clock<ManualClock>("plan", std::int64_t{0});
  graph.set_default_clock(plan_clock);
  graph.add_clock_conversion(
      capture_clock,
      plan_clock,
      AffineClockConversion{.scale = ClockRateRatio{.numerator = 1, .denominator = 1},
                            .offset_ns = 7});

  TopicOptions imported_options;
  imported_options.endpoint_name = "hero_input";
  imported_options.capture_time = RequireCaptureTime{.clock = capture_clock};
  TopicOptions exported_options;
  exported_options.endpoint_name = "hero_output";
  const auto imported =
      graph.import_topic<std::uint32_t>("holomq://holoscan-m1-hero/input", imported_options);
  const auto exported =
      graph.export_topic<std::uint32_t>("holomq://holoscan-m1-hero/output", exported_options);
  const auto first = graph.op<HeroCompileOperator>("first");
  const auto second = graph.op<HeroCompileOperator>("second");
  graph.add_flow(imported.output(), first->input, ConnectionOptions{.queue_depth = 4U});
  graph.add_flow(first->output, second->input, ConnectionOptions{.queue_depth = 4U});
  graph.add_flow(second->output, exported.input(), ConnectionOptions{.queue_depth = 4U});
  graph.region("bounded").add(first).add(second).constrain(RegionConstraint::bounded_memory());
  graph.lifecycle().before(first, second, LifecycleStage::kStart).allow_self_restart(first);
  return compile(graph);
}

// REQ: COMPILE-001
// REQ: COMPILE-002
TEST(CompileCompleteness, M1HeroIsByteDeterministicAndRejectedPlansRemainInspectable) {
  // Key setup: two independent compile_hero_graph() calls with the same input state.
  const ExecutionPlan first = compile_hero_graph();
  const ExecutionPlan second = compile_hero_graph();
  ASSERT_TRUE(first.ok()) << first.json();
  ASSERT_TRUE(second.ok()) << second.json();
  // Key assertion: both JSON and digest are byte-identical across independent compiles.
  EXPECT_EQ(first.json(), second.json());
  EXPECT_EQ(first.digest(), second.digest());
  EXPECT_FALSE(first.graph().external_topics().empty());
  EXPECT_FALSE(first.clock_catalog().clocks.empty());
  EXPECT_FALSE(first.lifecycle_checkpoints().empty());
  EXPECT_FALSE(first.profile_manifest().scope_resolutions.empty());

  Graph rejected_graph{"holoscan-m1-hero-rejected"};
  test::add_linear_graph(rejected_graph);
  CompileOptions invalid;
  invalid.profile = static_cast<Profile>(255U);
  const ExecutionPlan rejected = compile(rejected_graph, invalid);
  EXPECT_FALSE(rejected.ok());
  EXPECT_FALSE(rejected.json().empty());
  EXPECT_FALSE(rejected.diagnostics().empty());
  EXPECT_FALSE(rejected.graph().vertices().empty());
  EXPECT_FALSE(rejected.profile_manifest().requested_profile.has_value());
}

// REQ: COMPILE-002
// REQ: COMPILE-007
// REQ: COMPILE-015
// REQ: COMPILE-060
// REQ: COMPILE-019
// REQ: COMPILE-040
TEST(CompileCompleteness, M1HeroTraversesEveryCompileOwnedPipelineFamily) {
  const ExecutionPlan plan = compile_hero_graph();
  ASSERT_TRUE(plan.ok()) << plan.json();

  EXPECT_FALSE(plan.graph().ports().empty());
  EXPECT_FALSE(plan.graph().edges().empty());
  EXPECT_FALSE(plan.graph().regions().empty());
  EXPECT_FALSE(plan.graph().clocks().empty());
  EXPECT_FALSE(plan.graph().clock_conversions().empty());
  EXPECT_FALSE(plan.graph().lifecycle_dependencies().empty());
  EXPECT_FALSE(plan.graph().lifecycle_restart_permissions().empty());

  EXPECT_FALSE(plan.partition_decisions().empty());
  EXPECT_FALSE(plan.execution_partitions().empty());
  EXPECT_FALSE(plan.ports().empty());
  EXPECT_FALSE(plan.payload_edges().empty());
  EXPECT_FALSE(plan.channel_locality().empty());
  EXPECT_FALSE(plan.channel_services().empty());
  EXPECT_FALSE(plan.channel_publishers().empty());
  EXPECT_FALSE(plan.channel_subscribers().empty());
  EXPECT_FALSE(plan.generic_output_loans().empty());

  const detail::ExecutionPlanState* const state = detail::ExecutionPlanAccess::state(plan);
  ASSERT_NE(state, nullptr);
  EXPECT_FALSE(state->vertex_process_placements.empty());
  EXPECT_FALSE(state->process_local_channel_services.empty());
  EXPECT_FALSE(state->process_local_channel_publishers.empty());
  EXPECT_FALSE(state->process_local_channel_subscribers.empty());
  EXPECT_EQ(plan.json().find("process_local_channel_services"), std::string::npos);
  EXPECT_EQ(plan.json().find("process_local_channel_publishers"), std::string::npos);
  EXPECT_EQ(plan.json().find("process_local_channel_subscribers"), std::string::npos);

  EXPECT_FALSE(plan.clock_catalog().clocks.empty());
  EXPECT_FALSE(plan.clock_conversions().empty());
  EXPECT_FALSE(plan.temporal_triggers().empty());
  EXPECT_FALSE(plan.input_selections().empty());
  EXPECT_FALSE(plan.lifecycle_checkpoints().empty());
  EXPECT_FALSE(plan.lifecycle_dependencies().empty());
  EXPECT_FALSE(plan.lifecycle_restart_targets().empty());
  EXPECT_FALSE(plan.operator_parameters().empty());
  EXPECT_FALSE(plan.profile_manifest().scope_resolutions.empty());

  EXPECT_TRUE(plan.endpoint_catalog().endpoint("hero_input"));
  EXPECT_TRUE(plan.endpoint_catalog().endpoint("hero_output"));
}

}  // namespace
}  // namespace holoscan
