/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/execution_plan.hpp>

#include "compile_test_graph.hpp"
#include "compiler/compile_internal.hpp"
#include "compiler/process_placement_internal.hpp"

namespace holoscan::test {
namespace {

// Minimal nested graph with one imported uint32 input; creates a namespaced operator
// reachable at graph path "/child/op" for delimiter-safe process-group identity tests
// (SeparateGroupIdentityIsDelimiterSafeAndCanonical).
class ProcessPlacementChildGraph final : public Graph {
 public:
  [[nodiscard]] ClockKey clock_import() const noexcept { return clock_import_; }

  Input<std::uint32_t> input{"input"};

 protected:
  void compose() override {
    clock_import_ = import_clock("timeline");
    set_default_clock(clock_import_);
    const auto member = op<CompileTestOperator>("op");
    export_input(input, member->input);
  }

 private:
  ClockKey clock_import_;
};

// Builds a source-private process binding with "producer" bound to "source" and
// "consumer" bound to "target", each using the supplied launch-group string.
[[nodiscard]] detail::ProcessPlacementBinding two_role_placement(std::string_view first_group,
                                                                 std::string_view second_group) {
  constexpr std::array first{detail::GraphObjectSelector{.graph_path = "source"}};
  constexpr std::array second{detail::GraphObjectSelector{.graph_path = "target"}};
  detail::ProcessPlacementBinding placement;
  placement
      .bind_process_role("producer",
                         detail::ProcessPlacement{
                             .members = first,
                             .launch_group = first_group,
                         })
      .bind_process_role("consumer",
                         detail::ProcessPlacement{
                             .members = second,
                             .launch_group = second_group,
                         });
  return placement;
}

// Compiles a linear source --> target graph with one process role covering one selector.
// Used in rejection tests for invalid, unknown, and incompatible selector authority.
[[nodiscard]] ExecutionPlan compile_one_role(
    detail::GraphObjectSelector selector,
    detail::ProcessIsolation isolation = detail::ProcessIsolation::kSameProcess,
    std::string_view launch_group = "workers") {
  Graph graph{"process-placement-rejection"};
  add_linear_graph(graph);
  const std::array selectors{selector};
  detail::ProcessPlacementBinding placement;
  placement.bind_process_role("role",
                              detail::ProcessPlacement{
                                  .isolation = isolation,
                                  .members = selectors,
                                  .launch_group = launch_group,
                              });
  return detail::compile_with_process_placement(graph, CompileOptions{}, std::move(placement));
}

// REQ: COMPILE-018
// REQ: COMPILE-061
TEST(CompileInternalProcessPlacement, BindingDeepCopiesEveryBorrowedSelectorAndIdentity) {
  std::string role{"producer"};
  std::string graph_path{"source"};
  std::string launch_group{"workers"};
  std::string partition_name{"control"};
  std::array selectors{detail::GraphObjectSelector{.graph_path = graph_path}};

  DeploymentBinding deployment;
  detail::ProcessPlacementBinding process_placement;
  deployment.bind_thread_placement(partition_name, ThreadPlacement{});
  process_placement.bind_process_role(role,
                                      detail::ProcessPlacement{
                                          .members = selectors,
                                          .launch_group = launch_group,
                                      });
  role.assign("changed");
  graph_path.assign("changed");
  launch_group.assign("changed");
  partition_name.assign("changed");
  selectors.front().graph_path = "changed";

  ASSERT_EQ(process_placement.process_roles().size(), 1U);
  const auto& copied = process_placement.process_roles().front();
  // Key assertion: all strings are deep-copied; mutating originals after bind must not
  //               affect the recorded role, group, or selector path.
  EXPECT_EQ(copied.role, "producer");
  EXPECT_EQ(copied.placement.launch_group, "workers");
  ASSERT_EQ(copied.placement.members.size(), 1U);
  EXPECT_EQ(copied.placement.members.front().graph_path, "source");
  ASSERT_EQ(deployment.thread_placements().size(), 1U);
  EXPECT_EQ(deployment.thread_placements().front().partition_name, "control");
}

// REQ: COMPILE-061
TEST(CompileInternalProcessPlacement, PublicCompileLowersExactlyOneDefaultProcessGroup) {
  Graph graph{"public-default-process-group"};
  add_linear_graph(graph);

  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();

  const auto roles = detail::ProcessPlacementPlanAccess::process_roles(plan);
  const auto selectors = detail::ProcessPlacementPlanAccess::process_selectors(plan);
  const auto placements = detail::ProcessPlacementPlanAccess::vertex_process_placements(plan);
  EXPECT_TRUE(roles.empty());
  EXPECT_TRUE(selectors.empty());
  ASSERT_EQ(placements.size(), 2U);
  EXPECT_TRUE(std::all_of(placements.begin(), placements.end(), [](const auto& placement) {
    return placement.process_group.value == 1U && !placement.process_role &&
           placement.isolation == detail::ProcessIsolation::kSameProcess &&
           placement.launch_group == "holoscan.default_process" && !placement.runtime_seal_required;
  }));
  ASSERT_EQ(plan.payload_edges().size(), 1U);
  EXPECT_EQ(plan.payload_edges().front().locality, ChannelLocalityKind::kProcessLocal);
}

// REQ: COMPILE-061
//   source --[cross-process channel]--> target
//   "producer-process" launch group --> "consumer-process" launch group (separate boundary)
TEST(CompileInternalProcessPlacement,
     FreezesGroupsAndDerivesCrossProcessPhysicalAndScheduleEvidence) {
  Graph graph{"cross-process"};
  add_linear_graph(graph);

  const ExecutionPlan plan = detail::compile_with_process_placement(
      graph, CompileOptions{}, two_role_placement("producer-process", "consumer-process"));
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto roles = detail::ProcessPlacementPlanAccess::process_roles(plan);
  const auto selectors = detail::ProcessPlacementPlanAccess::process_selectors(plan);
  const auto placements = detail::ProcessPlacementPlanAccess::vertex_process_placements(plan);
  ASSERT_EQ(roles.size(), 2U);
  ASSERT_EQ(selectors.size(), 2U);
  ASSERT_EQ(placements.size(), 2U);
  EXPECT_TRUE(std::all_of(roles.begin(), roles.end(), [](const auto& role) {
    return role.disposition == PlanDisposition::kAccepted && role.accepted_isolation.has_value() &&
           role.runtime_seal_required;
  }));
  EXPECT_NE(placements[0].process_group, placements[1].process_group);

  ASSERT_EQ(plan.payload_edges().size(), 1U);
  // Key assertion: the single edge crosses a process boundary and requires runtime attach.
  EXPECT_EQ(plan.payload_edges().front().locality, ChannelLocalityKind::kCrossProcess);
  ASSERT_EQ(plan.channel_locality().size(), 1U);
  EXPECT_EQ(plan.channel_locality().front().accepted, ChannelLocalityKind::kCrossProcess);
  EXPECT_TRUE(plan.channel_locality().front().requires_runtime_attach);
  EXPECT_TRUE(std::any_of(plan.partition_decisions().begin(),
                          plan.partition_decisions().end(),
                          [](const auto& decision) {
                            return std::find(decision.reasons.begin(),
                                             decision.reasons.end(),
                                             PartitionDecisionReason::kProcessBoundary) !=
                                   decision.reasons.end();
                          }));
  EXPECT_NE(plan.json().find("\"process_roles\""), std::string_view::npos);
  EXPECT_NE(plan.json().find("\"process_selectors\""), std::string_view::npos);
  EXPECT_NE(plan.json().find("\"vertex_process_placements\""), std::string_view::npos);
}

// REQ: COMPILE-061
//   same_graph:     source --[process-local]---> target  (both in launch_group "workers")
//   separate_graph: source --[cross-process]---> target  (kSeparateProcess, same group name)
TEST(CompileInternalProcessPlacement, CoalescesSameLaunchGroupAndSeparatesEveryRequestedMember) {
  Graph same_graph{"same-process"};
  add_linear_graph(same_graph);
  const ExecutionPlan same = detail::compile_with_process_placement(
      same_graph, CompileOptions{}, two_role_placement("workers", "workers"));
  ASSERT_TRUE(same.ok()) << same.json();
  const auto same_placements = detail::ProcessPlacementPlanAccess::vertex_process_placements(same);
  ASSERT_EQ(same_placements.size(), 2U);
  // Key assertion: same launch group coalesces into one process_group; edge is process-local.
  EXPECT_EQ(same_placements[0].process_group, same_placements[1].process_group);
  ASSERT_EQ(same.payload_edges().size(), 1U);
  EXPECT_EQ(same.payload_edges().front().locality, ChannelLocalityKind::kProcessLocal);

  Graph separate_graph{"separate-process"};
  add_linear_graph(separate_graph);
  constexpr std::array both{detail::GraphObjectSelector{.graph_path = "source"},
                            detail::GraphObjectSelector{.graph_path = "target"}};
  detail::ProcessPlacementBinding separate_placement;
  separate_placement.bind_process_role("isolated-workers",
                                       detail::ProcessPlacement{
                                           .isolation = detail::ProcessIsolation::kSeparateProcess,
                                           .members = both,
                                           .launch_group = "workers",
                                           .require_locality_probe_success = false,
                                       });
  const ExecutionPlan separate = detail::compile_with_process_placement(
      separate_graph, CompileOptions{}, std::move(separate_placement));
  ASSERT_TRUE(separate.ok()) << separate.json();
  const auto separate_placements =
      detail::ProcessPlacementPlanAccess::vertex_process_placements(separate);
  ASSERT_EQ(separate_placements.size(), 2U);
  EXPECT_NE(separate_placements[0].process_group, separate_placements[1].process_group);
  EXPECT_FALSE(separate_placements[0].runtime_seal_required);
  EXPECT_FALSE(separate_placements[1].runtime_seal_required);
}

// REQ: COMPILE-061
//   region_graph:   source --> target  (source in region "ingest"; region selector resolves)
//   external_graph: [holomq://camera] --[kExternalImport]--> sink
TEST(CompileInternalProcessPlacement,
     ResolvesRegionAndExternalTopicSelectorsWithoutOverridingBoundaryLocality) {
  Graph region_graph{"region-selection"};
  add_compile_test_default_clock(region_graph);
  const auto source = region_graph.op<CompileTestOperator>("source");
  const auto target = region_graph.op<CompileTestOperator>("target");
  region_graph.add_flow(source->output, target->input);
  region_graph.region("ingest").add(source);
  constexpr std::array region_selector{detail::GraphObjectSelector{.region_id = "ingest"}};
  detail::ProcessPlacementBinding region_placement;
  region_placement.bind_process_role(
      "ingest-process",
      detail::ProcessPlacement{.members = region_selector, .launch_group = "ingest"});
  const ExecutionPlan region_plan = detail::compile_with_process_placement(
      region_graph, CompileOptions{}, std::move(region_placement));
  ASSERT_TRUE(region_plan.ok()) << region_plan.json();
  const auto region_selectors = detail::ProcessPlacementPlanAccess::process_selectors(region_plan);
  ASSERT_EQ(region_selectors.size(), 1U);
  EXPECT_EQ(region_selectors.front().kind, detail::ProcessSelectorKind::kRegion);
  EXPECT_TRUE(region_selectors.front().region.has_value());
  EXPECT_EQ(region_selectors.front().members.size(), 1U);

  Graph external_graph{"external-selection"};
  add_compile_test_default_clock(external_graph);
  auto imported = external_graph.import_topic<std::uint32_t>("holomq://camera");
  auto sink = external_graph.op<CompileTestOperator>("sink");
  external_graph.add_flow(imported.output(), sink->input);
  constexpr std::array external_selector{
      detail::GraphObjectSelector{.external_topic = "holomq://camera"}};
  detail::ProcessPlacementBinding external_placement;
  external_placement.bind_process_role("camera-client",
                                       detail::ProcessPlacement{
                                           .isolation = detail::ProcessIsolation::kExternalProcess,
                                           .members = external_selector,
                                           .launch_group = "camera-client",
                                       });
  const ExecutionPlan external_plan = detail::compile_with_process_placement(
      external_graph, CompileOptions{}, std::move(external_placement));
  ASSERT_TRUE(external_plan.ok()) << external_plan.json();
  const auto external_selectors =
      detail::ProcessPlacementPlanAccess::process_selectors(external_plan);
  ASSERT_EQ(external_selectors.size(), 1U);
  EXPECT_EQ(external_selectors.front().kind, detail::ProcessSelectorKind::kExternalTopic);
  ASSERT_EQ(external_plan.payload_edges().size(), 1U);
  // Key assertion: external topic selector does not override the imported boundary locality.
  EXPECT_EQ(external_plan.payload_edges().front().locality, ChannelLocalityKind::kExternalImport);
}

// REQ: COMPILE-061
TEST(CompileInternalProcessPlacement, RejectsInvalidUnknownAndIncompatibleSelectorAuthority) {
  const ExecutionPlan multiple_families =
      compile_one_role(detail::GraphObjectSelector{.graph_path = "source", .region_id = "region"});
  EXPECT_FALSE(multiple_families.ok());
  EXPECT_TRUE(has_diagnostic(multiple_families, "PROCESS_PLACEMENT_REJECTED"));

  const ExecutionPlan unknown =
      compile_one_role(detail::GraphObjectSelector{.graph_path = "unknown-vertex"});
  EXPECT_FALSE(unknown.ok());
  EXPECT_TRUE(has_diagnostic(unknown, "PROCESS_PLACEMENT_REJECTED"));

  const ExecutionPlan incompatible =
      compile_one_role(detail::GraphObjectSelector{.graph_path = "source"},
                       detail::ProcessIsolation::kExternalProcess);
  EXPECT_FALSE(incompatible.ok());
  EXPECT_TRUE(has_diagnostic(incompatible, "PROCESS_PLACEMENT_REJECTED"));
}

// REQ: COMPILE-061
// Four source --> target rejection scenarios in sequence: duplicate role names, duplicate
// selectors within one role, overlapping selectors across two roles, and incompatible
// isolation modes sharing the same launch group name.
TEST(CompileInternalProcessPlacement, RejectsDuplicateRolesSelectorsOverlapsAndConflictingGroups) {
  Graph duplicate_role_graph{"duplicate-role"};
  add_linear_graph(duplicate_role_graph);
  constexpr std::array source_only{detail::GraphObjectSelector{.graph_path = "source"}};
  constexpr std::array target_only{detail::GraphObjectSelector{.graph_path = "target"}};
  detail::ProcessPlacementBinding duplicate_roles;
  duplicate_roles
      .bind_process_role("workers",
                         detail::ProcessPlacement{.members = source_only, .launch_group = "first"})
      .bind_process_role(
          "workers", detail::ProcessPlacement{.members = target_only, .launch_group = "second"});
  const ExecutionPlan duplicate_role_plan = detail::compile_with_process_placement(
      duplicate_role_graph, CompileOptions{}, std::move(duplicate_roles));
  EXPECT_FALSE(duplicate_role_plan.ok());
  EXPECT_TRUE(has_diagnostic(duplicate_role_plan, "PROCESS_PLACEMENT_REJECTED"));

  Graph duplicate_selector_graph{"duplicate-selector"};
  add_linear_graph(duplicate_selector_graph);
  constexpr std::array duplicate_selectors{detail::GraphObjectSelector{.graph_path = "source"},
                                           detail::GraphObjectSelector{.graph_path = "source"}};
  detail::ProcessPlacementBinding duplicate_selector_placement;
  duplicate_selector_placement.bind_process_role(
      "workers",
      detail::ProcessPlacement{.members = duplicate_selectors, .launch_group = "workers"});
  const ExecutionPlan duplicate_selector_plan = detail::compile_with_process_placement(
      duplicate_selector_graph, CompileOptions{}, std::move(duplicate_selector_placement));
  EXPECT_FALSE(duplicate_selector_plan.ok());
  EXPECT_TRUE(has_diagnostic(duplicate_selector_plan, "PROCESS_PLACEMENT_REJECTED"));

  Graph overlap_graph{"overlap"};
  add_linear_graph(overlap_graph);
  detail::ProcessPlacementBinding overlap_placement;
  overlap_placement
      .bind_process_role("first",
                         detail::ProcessPlacement{.members = source_only, .launch_group = "first"})
      .bind_process_role(
          "second", detail::ProcessPlacement{.members = source_only, .launch_group = "second"});
  const ExecutionPlan overlap_plan = detail::compile_with_process_placement(
      overlap_graph, CompileOptions{}, std::move(overlap_placement));
  EXPECT_FALSE(overlap_plan.ok());
  EXPECT_TRUE(has_diagnostic(overlap_plan, "PROCESS_PLACEMENT_REJECTED"));

  Graph conflict_graph{"group-conflict"};
  add_linear_graph(conflict_graph);
  detail::ProcessPlacementBinding conflict_placement;
  // Key setup: same launch group "workers" assigned to kSameProcess and kSeparateProcess
  //            simultaneously; incompatible isolation modes must be rejected.
  conflict_placement
      .bind_process_role("first",
                         detail::ProcessPlacement{
                             .isolation = detail::ProcessIsolation::kSameProcess,
                             .members = source_only,
                             .launch_group = "workers",
                         })
      .bind_process_role("second",
                         detail::ProcessPlacement{
                             .isolation = detail::ProcessIsolation::kSeparateProcess,
                             .members = target_only,
                             .launch_group = "workers",
                         });
  const ExecutionPlan conflict_plan = detail::compile_with_process_placement(
      conflict_graph, CompileOptions{}, std::move(conflict_placement));
  EXPECT_FALSE(conflict_plan.ok());
  EXPECT_TRUE(has_diagnostic(conflict_plan, "PROCESS_PLACEMENT_REJECTED"));
}

// REQ: COMPILE-061
//   source ---> target  (compiled twice; producer/consumer role and selector order reversed)
// Both compilations must produce the same plan_generation and JSON serialization.
TEST(CompileInternalProcessPlacement,
     CanonicalizesRoleAndSelectorOrderIntoGenerationAndArtifactIdentity) {
  Graph first_graph{"canonical-process-placement"};
  add_linear_graph(first_graph);
  const ExecutionPlan first = detail::compile_with_process_placement(
      first_graph, CompileOptions{}, two_role_placement("producer", "consumer"));
  ASSERT_TRUE(first.ok()) << first.json();

  Graph second_graph{"canonical-process-placement"};
  add_linear_graph(second_graph, true);
  constexpr std::array target{detail::GraphObjectSelector{.graph_path = "target"}};
  constexpr std::array source{detail::GraphObjectSelector{.graph_path = "source"}};
  detail::ProcessPlacementBinding second_placement;
  second_placement
      .bind_process_role("consumer",
                         detail::ProcessPlacement{.members = target, .launch_group = "consumer"})
      .bind_process_role("producer",
                         detail::ProcessPlacement{.members = source, .launch_group = "producer"});
  const ExecutionPlan second = detail::compile_with_process_placement(
      second_graph, CompileOptions{}, std::move(second_placement));
  ASSERT_TRUE(second.ok()) << second.json();

  // Key assertion: role and selector order must not affect plan_generation or JSON output.
  EXPECT_EQ(first.clock_catalog().plan_generation, second.clock_catalog().plan_generation);
  EXPECT_EQ(first.json(), second.json());

  Graph baseline_graph{"canonical-process-placement"};
  add_linear_graph(baseline_graph);
  const ExecutionPlan baseline = compile(baseline_graph);
  ASSERT_TRUE(baseline.ok()) << baseline.json();
  EXPECT_NE(first.clock_catalog().plan_generation, baseline.clock_catalog().plan_generation);
}

// REQ: COMPILE-061
//   op --[cross-process]--> /child/op
//   root launch_group="x//child"; child launch_group="x"
TEST(CompileInternalProcessPlacement, SeparateGroupIdentityIsDelimiterSafeAndCanonical) {
  const auto compile_collision = [](bool reverse) {
    Graph graph{"delimiter-safe-process-placement"};
    const ClockKey root_clock = add_compile_test_default_clock(graph);
    const auto root = graph.op<CompileTestOperator>("op");
    const auto child = graph.embed<ProcessPlacementChildGraph>("child");
    graph.bind_clock(child, child->clock_import(), root_clock);
    graph.add_flow(root->output, child->input);

    constexpr std::array root_selector{detail::GraphObjectSelector{.graph_path = "op"}};
    constexpr std::array child_selector{detail::GraphObjectSelector{.graph_path = "/child/op"}};
    // Key setup: "x//child" and "x" share a common prefix; the compiler must assign
    //            distinct process groups regardless of binding order.
    const detail::ProcessPlacement root_placement{
        .isolation = detail::ProcessIsolation::kSeparateProcess,
        .members = root_selector,
        .launch_group = "x//child",
    };
    const detail::ProcessPlacement child_placement{
        .isolation = detail::ProcessIsolation::kSeparateProcess,
        .members = child_selector,
        .launch_group = "x",
    };
    detail::ProcessPlacementBinding placement;
    if (reverse) {
      placement.bind_process_role("child", child_placement);
      placement.bind_process_role("root", root_placement);
    } else {
      placement.bind_process_role("root", root_placement);
      placement.bind_process_role("child", child_placement);
    }
    return detail::compile_with_process_placement(graph, CompileOptions{}, std::move(placement));
  };

  const ExecutionPlan forward = compile_collision(false);
  const ExecutionPlan reverse = compile_collision(true);
  ASSERT_TRUE(forward.ok()) << forward.json();
  ASSERT_TRUE(reverse.ok()) << reverse.json();
  const auto forward_placements =
      detail::ProcessPlacementPlanAccess::vertex_process_placements(forward);
  const auto root =
      std::find_if(forward_placements.begin(), forward_placements.end(), [](const auto& placement) {
        return placement.launch_group == "x//child";
      });
  const auto child =
      std::find_if(forward_placements.begin(), forward_placements.end(), [](const auto& placement) {
        return placement.launch_group == "x";
      });
  ASSERT_NE(root, forward_placements.end());
  ASSERT_NE(child, forward_placements.end());
  EXPECT_NE(root->process_group, child->process_group);
  ASSERT_EQ(forward.payload_edges().size(), 1U);
  EXPECT_EQ(forward.payload_edges().front().locality, ChannelLocalityKind::kCrossProcess);
  // Key assertion: binding order must not alter the plan digest or JSON output.
  EXPECT_EQ(forward.digest(), reverse.digest());
  EXPECT_EQ(forward.json(), reverse.json());
}

}  // namespace
}  // namespace holoscan::test
