/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <holoscan/core/compile.hpp>

#include "compile_test_graph.hpp"
#include "compiler/closed_island_activation_plan.hpp"
#include "compiler/compile_internal.hpp"
#include "compiler/execution_plan_state.hpp"
#include "compiler/process_placement_internal.hpp"

namespace holoscan {

struct ClosedIslandPartitionPod {
  std::uint32_t value{};
};

template <>
struct ChannelTypeIdentity<ClosedIslandPartitionPod> {
  static constexpr std::string_view name = "nvidia.holoscan.test.ClosedIslandPartitionPod";
  static constexpr std::uint32_t version = 1U;
};

template <>
struct ChannelPodLayout<ClosedIslandPartitionPod> {
  static constexpr auto fields = pod_fields(pod_field("value", &ClosedIslandPartitionPod::value,
                                                      offsetof(ClosedIslandPartitionPod, value)));
};

namespace {

class PartitionJoinOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(first, "first").queue_depth(2U);
    spec.input(second, "second").queue_depth(2U);
    spec.output(output, "output").max_emits_per_compute(1U);
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Input<std::uint32_t> first;
  Input<std::uint32_t> second;
  Output<std::uint32_t> output;
};

class ClosedIslandSourceOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override { spec.output(output, "output"); }

  Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = std::chrono::milliseconds{1}});
    return result;
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Output<ClosedIslandPartitionPod> output;
};

class ClosedIslandSinkOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override { spec.input(input, "input").queue_depth(1U); }

  Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input});
    return result;
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Input<ClosedIslandPartitionPod> input;
};

[[nodiscard]] detail::ProcessPlacementBinding process_placement(std::string_view source_group,
                                                                std::string_view target_group) {
  constexpr std::array source{detail::GraphObjectSelector{.graph_path = "source"}};
  constexpr std::array target{detail::GraphObjectSelector{.graph_path = "target"}};
  detail::ProcessPlacementBinding placement;
  placement
      .bind_process_role("source",
                         detail::ProcessPlacement{.members = source, .launch_group = source_group})
      .bind_process_role("target",
                         detail::ProcessPlacement{.members = target, .launch_group = target_group});
  return placement;
}

[[nodiscard]] ExecutionPlan compile_manual_process_partition(std::string_view source_group,
                                                             std::string_view target_group) {
  Graph graph{"manual-process-partition"};
  test::add_compile_test_default_clock(graph);
  const auto source = graph.op<test::CompileTestOperator>("source");
  const auto target = graph.op<test::CompileTestOperator>("target");
  graph.add_flow(source->output, target->input);
  graph.partition("pipeline").add(source).add(target);
  return detail::compile_with_process_placement(
      graph, CompileOptions{}, process_placement(source_group, target_group));
}

struct CanonicalPartitionEvidence {
  std::vector<std::string> partitions;
  std::vector<std::string> schedule_edges;
  std::vector<std::string> placements;
  std::vector<std::string> probes;
  std::vector<std::string> lanes;

  friend bool operator==(const CanonicalPartitionEvidence&,
                         const CanonicalPartitionEvidence&) = default;
};

[[nodiscard]] CanonicalPartitionEvidence partition_evidence(const ExecutionPlan& plan) {
  CanonicalPartitionEvidence result;
  for (const auto& partition : plan.execution_partitions()) {
    std::string record = std::to_string(partition.id.value) + ":" +
                         partition.authored_name.value_or("<implicit>") + ":" +
                         std::to_string(partition.execution_lane);
    for (const VertexId member : partition.members) {
      record += ":" + std::to_string(member.value);
    }
    result.partitions.push_back(std::move(record));
  }
  for (const auto& edge : plan.schedule_edges()) {
    result.schedule_edges.push_back(std::to_string(edge.edge.value) + ":" +
                                    std::to_string(edge.source.value) + ":" +
                                    std::to_string(edge.destination.value));
  }
  for (const auto& placement : plan.partition_placements()) {
    result.placements.push_back(std::to_string(placement.partition.value) + ":" +
                                placement.authored_name + ":" +
                                std::to_string(placement.execution_lane) + ":" +
                                std::to_string(static_cast<std::uint32_t>(placement.disposition)));
  }
  for (const auto& probe : plan.placement_probes()) {
    result.probes.push_back(std::to_string(probe.partition.value) + ":" + probe.probe_id + ":" +
                            std::to_string(static_cast<std::uint32_t>(probe.status)) + ":" +
                            probe.evidence_id + ":" + std::to_string(probe.reason.value));
  }
  for (const auto& lane : plan.execution_lanes()) {
    result.lanes.push_back(std::to_string(lane.execution_lane) + ":" +
                           std::to_string(lane.partition.value));
  }
  return result;
}

[[nodiscard]] ExecutionPlan compile_canonical_manual_partitions(bool reverse) {
  Graph graph{"canonical-manual-partitions"};
  test::add_compile_test_default_clock(graph);
  const auto first = graph.op<test::CompileTestOperator>("first");
  const auto second = graph.op<test::CompileTestOperator>("second");
  const auto third = graph.op<test::CompileTestOperator>("third");
  const auto fourth = graph.op<test::CompileTestOperator>("fourth");
  graph.add_flow(first->output, second->input);
  graph.add_flow(second->output, third->input);
  graph.add_flow(third->output, fourth->input);

  CompileOptions options;
  if (reverse) {
    graph.partition("beta").add(fourth).add(third);
    graph.partition("alpha").add(second).add(first);
    options.deployment.bind_thread_placement("beta", ThreadPlacement{});
    options.deployment.bind_thread_placement("alpha", ThreadPlacement{});
  } else {
    graph.partition("alpha").add(first).add(second);
    graph.partition("beta").add(third).add(fourth);
    options.deployment.bind_thread_placement("alpha", ThreadPlacement{});
    options.deployment.bind_thread_placement("beta", ThreadPlacement{});
  }
  return compile(graph, options);
}

[[nodiscard]] ExecutionPlan compile_manual_fusion_partitions(bool reverse) {
  Graph graph{"manual-fusion-groups"};
  test::add_compile_test_default_clock(graph);
  const auto camera = graph.op<test::CompileTestOperator>("camera");
  const auto camera_pre = graph.op<test::CompileTestOperator>("camera-pre");
  const auto imu = graph.op<test::CompileTestOperator>("imu");
  const auto imu_pre = graph.op<test::CompileTestOperator>("imu-pre");
  const auto fusion = graph.op<PartitionJoinOperator>("fusion");
  const auto display = graph.op<test::CompileTestOperator>("display");
  const auto visualization = graph.op<test::CompileTestOperator>("visualization");
  const auto inference = graph.op<test::CompileTestOperator>("inference");
  const auto storage = graph.op<test::CompileTestOperator>("storage");
  graph.add_flow(camera->output, camera_pre->input);
  graph.add_flow(camera_pre->output, fusion->first);
  graph.add_flow(imu->output, imu_pre->input);
  graph.add_flow(imu_pre->output, fusion->second);
  graph.add_flow(fusion->output, display->input);
  graph.add_flow(display->output, visualization->input);
  graph.add_flow(fusion->output, inference->input);
  graph.add_flow(inference->output, storage->input);

  if (reverse) {
    graph.partition("inference-storage").add(storage).add(inference);
    graph.partition("fusion-display").add(visualization).add(display).add(fusion);
    graph.partition("imu").add(imu_pre).add(imu);
    graph.partition("camera").add(camera_pre).add(camera);
  } else {
    graph.partition("camera").add(camera).add(camera_pre);
    graph.partition("imu").add(imu).add(imu_pre);
    graph.partition("fusion-display").add(fusion).add(display).add(visualization);
    graph.partition("inference-storage").add(inference).add(storage);
  }
  return compile(graph);
}

[[nodiscard]] std::vector<std::string> profile_capacity_evidence(const ExecutionPlan& plan) {
  std::vector<std::string> result;
  for (const auto& capacity : plan.profile_manifest().derived_capacities) {
    result.push_back(capacity.object_id + ":" + capacity.capacity_id + ":" +
                     std::to_string(capacity.limit) + ":" + capacity.unit);
  }
  return result;
}

// REQ: COMPILE-058
TEST(CompilePartitionAuthoring, RejectsManualPartitionAcrossPhysicalProcessGroups) {
  const ExecutionPlan split = compile_manual_process_partition("source-process", "target-process");
  EXPECT_FALSE(split.ok());
  EXPECT_TRUE(test::has_diagnostic(split, "PARTITION_COVER_INVALID"));

  const ExecutionPlan colocated = compile_manual_process_partition("workers", "workers");
  ASSERT_TRUE(colocated.ok()) << colocated.json();
  ASSERT_EQ(colocated.execution_partitions().size(), 1U);
  EXPECT_EQ(colocated.execution_partitions().front().authored_name,
            std::optional<std::string>{"pipeline"});
  const auto process_placements =
      detail::ProcessPlacementPlanAccess::vertex_process_placements(colocated);
  ASSERT_EQ(process_placements.size(), 2U);
  EXPECT_EQ(process_placements[0].process_group, process_placements[1].process_group);
}

// REQ: COMPILE-058
// REQ: COMPILE-059
TEST(CompilePartitionAuthoring, CanonicalizesManualDeclarationMemberAndPlacementOrder) {
  const ExecutionPlan forward = compile_canonical_manual_partitions(false);
  const ExecutionPlan reverse = compile_canonical_manual_partitions(true);
  ASSERT_TRUE(forward.ok()) << forward.json();
  ASSERT_TRUE(reverse.ok()) << reverse.json();

  EXPECT_EQ(forward.digest(), reverse.digest());
  EXPECT_EQ(forward.json(), reverse.json());
  EXPECT_EQ(partition_evidence(forward), partition_evidence(reverse));
  ASSERT_EQ(forward.execution_partitions().size(), 2U);
  EXPECT_EQ(forward.execution_partitions()[0].id, ExecutionPartitionId{1U});
  EXPECT_EQ(forward.execution_partitions()[0].authored_name, std::optional<std::string>{"alpha"});
  EXPECT_EQ(forward.execution_partitions()[1].id, ExecutionPartitionId{2U});
  EXPECT_EQ(forward.execution_partitions()[1].authored_name, std::optional<std::string>{"beta"});
  ASSERT_EQ(forward.schedule_edges().size(), 1U);
  EXPECT_EQ(forward.schedule_edges().front().source, ExecutionPartitionId{1U});
  EXPECT_EQ(forward.schedule_edges().front().destination, ExecutionPartitionId{2U});
}

// Mirrors Jay's two-source fusion/fanout execution shape:
//
//   camera -> camera-pre --\
//                           fusion -> display -> visualization
//   imu    -> imu-pre    --/      \-> inference -> storage
//
// REQ: COMPILE-058
TEST(CompilePartitionAuthoring, LowersFourManualFusionExecutionGroups) {
  const ExecutionPlan plan = compile_manual_fusion_partitions(false);
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.execution_partitions().size(), 4U);
  EXPECT_EQ(plan.execution_partitions()[0].authored_name, std::optional<std::string>{"camera"});
  EXPECT_EQ(plan.execution_partitions()[1].authored_name,
            std::optional<std::string>{"fusion-display"});
  EXPECT_EQ(plan.execution_partitions()[2].authored_name, std::optional<std::string>{"imu"});
  EXPECT_EQ(plan.execution_partitions()[3].authored_name,
            std::optional<std::string>{"inference-storage"});
  EXPECT_EQ(plan.schedule_edges().size(), 3U);
  EXPECT_EQ(plan.partitioning_plan().selection_source, PlanSelectionSource::kCoreDefault);
  ASSERT_TRUE(plan.partitioning_plan().accepted_algorithm.has_value());
  // The assertion above establishes that the optional is engaged.
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
  EXPECT_EQ(plan.partitioning_plan().accepted_algorithm->id, "holoscan.partition.serial_baseline");
  EXPECT_EQ(plan.scheduler_plan().execution_lane_count, 4U);
  EXPECT_EQ(plan.scheduler_plan().max_concurrent_activations, 4U);
  EXPECT_TRUE(plan.scheduler_plan().activation_overlap);
  EXPECT_EQ(plan.scheduler_plan().worker_count, 4U);
  EXPECT_EQ(plan.scheduler_plan().intra_path_parallelism, 1U);

  const auto& capacities = plan.profile_manifest().derived_capacities;
  const auto scheduler_capacity =
      std::find_if(capacities.begin(), capacities.end(), [](const auto& capacity) {
        return capacity.object_id == "scheduler" &&
               capacity.capacity_id == "max_concurrent_activations";
      });
  ASSERT_NE(scheduler_capacity, capacities.end());
  EXPECT_EQ(scheduler_capacity->limit, 4U);

  const auto* const state = detail::ExecutionPlanAccess::state(plan);
  ASSERT_NE(state, nullptr);
  ASSERT_EQ(state->partition_runners.size(), 4U);
  for (std::size_t index = 0U; index < state->partition_runners.size(); ++index) {
    const auto& runner = state->partition_runners[index];
    EXPECT_EQ(runner.ordinal, static_cast<std::uint32_t>(index));
    EXPECT_EQ(runner.partition, plan.execution_partitions()[index].id);
    EXPECT_EQ(runner.execution_lane, plan.execution_partitions()[index].execution_lane);
    ASSERT_FALSE(runner.operators.empty());
    EXPECT_TRUE(runner.operators.front().topology_root);
    EXPECT_EQ(std::count_if(runner.operators.begin(),
                            runner.operators.end(),
                            [](const auto& member) { return member.topology_root; }),
              1);
  }
  EXPECT_EQ(state->partition_runners[0].operators.size(), 2U);
  EXPECT_EQ(state->partition_runners[0].internal_edges.size(), 1U);
  EXPECT_TRUE(state->partition_runners[0].incoming_boundary_edges.empty());
  EXPECT_EQ(state->partition_runners[0].outgoing_boundary_edges.size(), 1U);
  EXPECT_EQ(state->partition_runners[1].operators.size(), 3U);
  EXPECT_EQ(state->partition_runners[1].internal_edges.size(), 2U);
  EXPECT_EQ(state->partition_runners[1].incoming_boundary_edges.size(), 2U);
  EXPECT_EQ(state->partition_runners[1].outgoing_boundary_edges.size(), 1U);
  EXPECT_EQ(state->partition_runners[2].operators.size(), 2U);
  EXPECT_EQ(state->partition_runners[2].internal_edges.size(), 1U);
  EXPECT_TRUE(state->partition_runners[2].incoming_boundary_edges.empty());
  EXPECT_EQ(state->partition_runners[2].outgoing_boundary_edges.size(), 1U);
  EXPECT_EQ(state->partition_runners[3].operators.size(), 2U);
  EXPECT_EQ(state->partition_runners[3].internal_edges.size(), 1U);
  EXPECT_EQ(state->partition_runners[3].incoming_boundary_edges.size(), 1U);
  EXPECT_TRUE(state->partition_runners[3].outgoing_boundary_edges.empty());
}

// REQ: COMPILE-058
TEST(CompilePartitionAuthoring, CanonicalizesFourManualPartitionPublicProjections) {
  const ExecutionPlan forward = compile_manual_fusion_partitions(false);
  const ExecutionPlan reverse = compile_manual_fusion_partitions(true);
  ASSERT_TRUE(forward.ok()) << forward.json();
  ASSERT_TRUE(reverse.ok()) << reverse.json();

  // Key assertion: private partition-runner reconstruction authority does not leak into the
  // canonical public artifact. Authoring-order differences preserve public JSON, digest, and
  // profile capacity evidence for the same four-partition cover.
  EXPECT_EQ(forward.json(), reverse.json());
  EXPECT_EQ(forward.digest(), reverse.digest());
  EXPECT_EQ(partition_evidence(forward), partition_evidence(reverse));
  EXPECT_EQ(profile_capacity_evidence(forward), profile_capacity_evidence(reverse));
  EXPECT_EQ(forward.json().find("partition_runners"), std::string::npos);
  EXPECT_EQ(reverse.json().find("partition_runners"), std::string::npos);

  const auto scheduler_capacity =
      std::find_if(forward.profile_manifest().derived_capacities.begin(),
                   forward.profile_manifest().derived_capacities.end(),
                   [](const auto& capacity) {
                     return capacity.object_id == "scheduler" &&
                            capacity.capacity_id == "max_concurrent_activations";
                   });
  ASSERT_NE(scheduler_capacity, forward.profile_manifest().derived_capacities.end());
  EXPECT_EQ(scheduler_capacity->limit, 4U);
}

// A closed-island composite is owned by one partition runner. Even if two
// partitions share a physical lane, the compiler must leave their boundary
// edge on the ordinary controller path rather than mint cross-owner authority.
TEST(CompilePartitionAuthoring, ClosedIslandCompositeDoesNotCrossPartitionIdentity) {
  Graph graph{"closed-island-partition-boundary"};
  test::add_compile_test_default_clock(graph);
  const auto source = graph.op<ClosedIslandSourceOperator>("source");
  const auto sink = graph.op<ClosedIslandSinkOperator>("sink");
  graph.add_flow(source->output, sink->input, ConnectionOptions{.queue_depth = 1U});
  graph.partition("pipeline").add(source).add(sink);

  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto* const compiled = detail::ExecutionPlanAccess::state(plan);
  ASSERT_NE(compiled, nullptr);
  const auto baseline = detail::derive_closed_island_activation_plan(*compiled);
  ASSERT_EQ(baseline.size(), 1U);
  ASSERT_EQ(baseline.front().source_partition, baseline.front().successor_partition);

  detail::ExecutionPlanState split = *compiled;
  split.execution_partitions = {
      ExecutionPartitionRecord{
          .id = ExecutionPartitionId{1U},
          .authored_name = std::string{"source"},
          .members = {baseline.front().source},
          .execution_lane = baseline.front().execution_lane,
      },
      ExecutionPartitionRecord{
          .id = ExecutionPartitionId{2U},
          .authored_name = std::string{"sink"},
          .members = {baseline.front().successor},
          .execution_lane = baseline.front().execution_lane,
      },
  };

  EXPECT_TRUE(detail::derive_closed_island_activation_plan(split).empty());
}

}  // namespace
}  // namespace holoscan
