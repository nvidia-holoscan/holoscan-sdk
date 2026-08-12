/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "compile_test_graph.hpp"
#include "compiler/deployment_plan_lowering.hpp"
#include "compiler/profile_admission.hpp"
#include "core/graph_compiler_access.hpp"

// GTest fatal assertions establish optional preconditions that clang-tidy does not model.
// NOLINTBEGIN(bugprone-unchecked-optional-access)
namespace holoscan {
namespace {

// REQ: COMPILE-014
// REQ: COMPILE-059

// Stub placement probe that returns a predetermined result on every call; accumulates
// call count and placement requests so tests can verify seam behavior without OS state.
class FixedPlacementProbe final : public detail::PlacementPlatformProbe {
 public:
  explicit FixedPlacementProbe(detail::PlacementPlatformProbeResult result)
      : result_(std::move(result)) {}

  [[nodiscard]] detail::PlacementPlatformProbeResult inspect(
      const ThreadPlacement& placement) const override {
    ++calls;
    requests.push_back(placement);
    return result_;
  }

  mutable std::size_t calls{};
  mutable std::vector<ThreadPlacement> requests;

 private:
  detail::PlacementPlatformProbeResult result_;
};

// Returns the PlacementProbePlanRecord whose probe_id matches, or nullptr if absent.
// Used in ProbeSeamFreezes... to assert the probe status for each outcome variant.
[[nodiscard]] const PlacementProbePlanRecord* find_probe(
    const detail::DeploymentPlanLowering& lowering, std::string_view probe_id) {
  const auto found = std::find_if(lowering.placement_probes.begin(),
                                  lowering.placement_probes.end(),
                                  [&](const auto& probe) { return probe.probe_id == probe_id; });
  return found == lowering.placement_probes.end() ? nullptr : &*found;
}

// Serializes partition placement, probe, and lane records into strings so
// tests can compare canonical deployment evidence independent of declaration order.
[[nodiscard]] std::vector<std::string> canonical_deployment_evidence(
    const detail::DeploymentPlanLowering& lowering) {
  std::vector<std::string> result;
  result.reserve(lowering.placements.size() + lowering.placement_probes.size() +
                 lowering.lanes.size());
  for (const auto& placement : lowering.placements) {
    result.push_back("placement:" + placement.authored_name + ":" +
                     std::to_string(placement.partition.value) + ":" +
                     std::to_string(static_cast<unsigned>(placement.disposition)));
  }
  for (const auto& probe : lowering.placement_probes) {
    result.push_back("probe:" + std::to_string(probe.partition.value) + ":" + probe.probe_id + ":" +
                     std::to_string(static_cast<unsigned>(probe.status)) + ":" + probe.evidence_id);
  }
  for (const auto& lane : lowering.lanes) {
    result.push_back("lane:" + std::to_string(lane.execution_lane) + ":" +
                     std::to_string(lane.partition.value));
  }
  return result;
}

// REQ: COMPILE-002
// REQ: COMPILE-009
TEST(CompileDeployment, ResolvesExactManualPartitionPlacement) {
  Graph graph{"deployment-partition-placement"};
  const auto op = graph.op<test::CompileTestOperator>("operator");
  graph.partition("control").add(op);
  CompileOptions options;
  options.deployment.bind_thread_placement("control", ThreadPlacement{});

  const ExecutionPlan plan = compile(graph, options);
  ASSERT_TRUE(plan.ok());
  ASSERT_EQ(plan.partition_placements().size(), 1U);
  const auto& placement = plan.partition_placements().front();
  EXPECT_EQ(placement.authored_name, "control");
  EXPECT_TRUE(static_cast<bool>(placement.partition));
  EXPECT_EQ(placement.disposition, PlanDisposition::kAccepted);
  ASSERT_EQ(plan.execution_lanes().size(), 1U);
  EXPECT_EQ(plan.execution_lanes().front().partition, placement.partition);
}

// REQ: COMPILE-002
// REQ: COMPILE-009
TEST(CompileDeployment, RejectsUnknownAndDuplicatePartitionPlacementAuthority) {
  Graph invalid_graph{"deployment-invalid-partition"};
  const auto invalid_op = invalid_graph.op<test::CompileTestOperator>("operator");
  invalid_graph.partition("known").add(invalid_op);
  CompileOptions invalid_options;
  invalid_options.deployment.bind_thread_placement("unknown", ThreadPlacement{});
  const ExecutionPlan invalid = compile(invalid_graph, invalid_options);
  EXPECT_FALSE(invalid.ok());
  EXPECT_TRUE(test::has_diagnostic(invalid, "PARTITION_PLACEMENT_UNKNOWN_OR_DUPLICATE"));

  Graph duplicate_graph{"deployment-duplicate-partition"};
  const auto duplicate_op = duplicate_graph.op<test::CompileTestOperator>("operator");
  duplicate_graph.partition("worker").add(duplicate_op);
  CompileOptions duplicate_options;
  duplicate_options.deployment.bind_thread_placement("worker", ThreadPlacement{});
  duplicate_options.deployment.bind_thread_placement("worker", ThreadPlacement{});
  const ExecutionPlan duplicate = compile(duplicate_graph, duplicate_options);
  EXPECT_FALSE(duplicate.ok());
  EXPECT_TRUE(test::has_diagnostic(duplicate, "PARTITION_PLACEMENT_UNKNOWN_OR_DUPLICATE"));
  ASSERT_EQ(duplicate.partition_placements().size(), 2U);
  EXPECT_TRUE(std::all_of(
      duplicate.partition_placements().begin(),
      duplicate.partition_placements().end(),
      [](const auto& record) { return record.disposition == PlanDisposition::kRejected; }));
}

// REQ: COMPILE-009
TEST(CompileDeployment, ProbeSeamFreezesPassedFallbackAndRejectedOutcomesWithoutApplyingOsState) {
  Graph graph{"placement-probe-seam"};
  (void)graph.op<test::CompileTestOperator>("operator");
  const auto snapshot = detail::GraphCompilerAccess::canonical_snapshot(graph);
  const auto standard = detail::resolve_profile_admission(*snapshot, Profile::kStandard);
  std::vector partitions{ExecutionPartitionRecord{
      .id = ExecutionPartitionId{1U},
      .authored_name = std::string{"worker"},
      .members = {snapshot->vertices.front().id},
      .execution_lane = 0U,
  }};

  // Key setup: FixedPlacementProbe stubs OS calls; four outcomes are verified in
  //            sequence: passed, fallback, rejected (real-time), and unavailable.
  DeploymentBinding proven_binding;
  proven_binding.bind_thread_placement(
      "worker",
      ThreadPlacement{.irq_isolation = IRQIsolation::kRequireIsolated,
                      .probe_failure_policy = ProbeFailurePolicy::kFailAdmission});
  const FixedPlacementProbe proven_probe(detail::PlacementPlatformProbeResult{
      .disposition = detail::PlacementPlatformProbeDisposition::kPassed,
      .irq_isolation_proven = true,
      .evidence_id = "test.placement.proven/v1",
  });
  const auto proven = detail::lower_deployment_plan(
      *snapshot, proven_binding, standard.effective_profiles(), partitions, proven_probe);
  ASSERT_TRUE(proven.error_tokens.empty());
  ASSERT_EQ(proven.placements.size(), 1U);
  ASSERT_TRUE(proven.placements.front().accepted.has_value());
  EXPECT_EQ(proven.placements.front().accepted->irq_isolation, IRQIsolation::kRequireIsolated);
  ASSERT_NE(find_probe(proven, "holoscan.placement.irq_isolation/v1"), nullptr);
  EXPECT_EQ(find_probe(proven, "holoscan.placement.irq_isolation/v1")->status,
            PlacementProbeStatus::kPassed);
  EXPECT_EQ(proven_probe.calls, 1U);

  DeploymentBinding fallback_binding;
  fallback_binding.bind_thread_placement(
      "worker",
      ThreadPlacement{.irq_isolation = IRQIsolation::kPreferIsolated,
                      .probe_failure_policy = ProbeFailurePolicy::kWarnAndRecordFallback});
  const FixedPlacementProbe unproven_probe(detail::PlacementPlatformProbeResult{
      .disposition = detail::PlacementPlatformProbeDisposition::kPassed,
      .irq_isolation_proven = false,
      .evidence_id = "test.placement.irq_unproven/v1",
  });
  const auto fallback = detail::lower_deployment_plan(
      *snapshot, fallback_binding, standard.effective_profiles(), partitions, unproven_probe);
  ASSERT_TRUE(fallback.error_tokens.empty());
  ASSERT_EQ(fallback.placements.size(), 1U);
  ASSERT_TRUE(fallback.placements.front().accepted.has_value());
  EXPECT_EQ(fallback.placements.front().requested.irq_isolation, IRQIsolation::kPreferIsolated);
  EXPECT_EQ(fallback.placements.front().accepted->irq_isolation, IRQIsolation::kNotRequested);
  ASSERT_NE(find_probe(fallback, "holoscan.placement.irq_isolation/v1"), nullptr);
  EXPECT_EQ(find_probe(fallback, "holoscan.placement.irq_isolation/v1")->status,
            PlacementProbeStatus::kFallbackRecorded);

  std::vector<std::optional<Profile>> real_time(snapshot->vertices.size(), Profile::kRealTime);
  const auto rejected = detail::lower_deployment_plan(
      *snapshot, fallback_binding, real_time, partitions, unproven_probe);
  EXPECT_EQ(rejected.error_tokens, (std::vector<std::string>{"THREAD_PLACEMENT_REJECTED"}));
  ASSERT_EQ(rejected.placements.size(), 1U);
  EXPECT_FALSE(rejected.placements.front().accepted.has_value());
  EXPECT_EQ(rejected.placements.front().disposition, PlanDisposition::kRejected);
  EXPECT_EQ(rejected.lanes.size(), 1U);

  const FixedPlacementProbe unavailable_probe(detail::PlacementPlatformProbeResult{
      .disposition = detail::PlacementPlatformProbeDisposition::kUnavailable,
      .irq_isolation_proven = false,
      .evidence_id = "test.placement.unavailable/v1",
  });
  const auto unavailable = detail::lower_deployment_plan(
      *snapshot, fallback_binding, standard.effective_profiles(), partitions, unavailable_probe);
  EXPECT_EQ(unavailable.error_tokens, (std::vector<std::string>{"THREAD_PLACEMENT_REJECTED"}));
  ASSERT_NE(find_probe(unavailable, "holoscan.placement.platform/v1"), nullptr);
  EXPECT_EQ(find_probe(unavailable, "holoscan.placement.platform/v1")->status,
            PlacementProbeStatus::kRejected);
}

// REQ: COMPILE-009
TEST(CompileDeployment, ReordersBindingsAndDeclarationsWithoutChangingCanonicalEvidence) {
  const auto lower = [](bool reverse) {
    Graph graph{"canonical-deployment"};
    (void)graph.op<test::CompileTestOperator>("alpha-operator");
    (void)graph.op<test::CompileTestOperator>("beta-operator");
    const auto snapshot = detail::GraphCompilerAccess::canonical_snapshot(graph);
    const auto admission = detail::resolve_profile_admission(*snapshot, Profile::kStandard);
    std::vector partitions{
        ExecutionPartitionRecord{
            .id = ExecutionPartitionId{1U},
            .authored_name = std::string{"alpha"},
            .members = {snapshot->vertices[0].id},
            .execution_lane = 0U,
        },
        ExecutionPartitionRecord{
            .id = ExecutionPartitionId{2U},
            .authored_name = std::string{"beta"},
            .members = {snapshot->vertices[1].id},
            .execution_lane = 1U,
        },
    };
    DeploymentBinding deployment;
    if (reverse) {
      deployment.bind_thread_placement("alpha", ThreadPlacement{});
      deployment.bind_thread_placement("beta", ThreadPlacement{});
    } else {
      deployment.bind_thread_placement("beta", ThreadPlacement{});
      deployment.bind_thread_placement("alpha", ThreadPlacement{});
    }
    const FixedPlacementProbe probe(detail::PlacementPlatformProbeResult{
        .disposition = detail::PlacementPlatformProbeDisposition::kPassed,
        .irq_isolation_proven = false,
        .evidence_id = "test.placement.canonical/v1",
    });
    return detail::lower_deployment_plan(
        *snapshot, deployment, admission.effective_profiles(), partitions, probe);
  };

  const auto forward = lower(false);
  const auto reverse = lower(true);
  EXPECT_TRUE(forward.error_tokens.empty());
  EXPECT_TRUE(reverse.error_tokens.empty());
  // Key assertion: declaration and binding order must not affect canonical evidence
  //               or the lane-to-vertex assignment.
  EXPECT_EQ(canonical_deployment_evidence(forward), canonical_deployment_evidence(reverse));
  EXPECT_EQ(forward.lane_by_vertex, reverse.lane_by_vertex);
}

// REQ: COMPILE-002
TEST(CompileDeployment, CanonicalizesAbsentAndEmptyAffinityInRejectedDuplicates) {
  const auto compile_duplicates = [](bool reverse) {
    Graph graph{"canonical-invalid-deployment"};
    const auto op = graph.op<test::CompileTestOperator>("operator");
    graph.partition("worker").add(op);
    CompileOptions options;
    const ThreadPlacement absent_affinity;
    const ThreadPlacement empty_affinity{.affinity = CPUAffinity{}};
    if (reverse) {
      options.deployment.bind_thread_placement("worker", empty_affinity);
      options.deployment.bind_thread_placement("worker", absent_affinity);
    } else {
      options.deployment.bind_thread_placement("worker", absent_affinity);
      options.deployment.bind_thread_placement("worker", empty_affinity);
    }
    return compile(graph, options);
  };

  const ExecutionPlan forward = compile_duplicates(false);
  const ExecutionPlan reverse = compile_duplicates(true);
  ASSERT_FALSE(forward.ok());
  ASSERT_FALSE(reverse.ok());
  // Key assertion: absent affinity and empty CPUAffinity{} are canonically equivalent;
  //               reversed binding order must produce the same digest and JSON.
  EXPECT_EQ(forward.digest(), reverse.digest());
  EXPECT_EQ(forward.json(), reverse.json());
}

}  // namespace
}  // namespace holoscan
// NOLINTEND(bugprone-unchecked-optional-access)
