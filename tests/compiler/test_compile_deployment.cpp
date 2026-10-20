/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <cerrno>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iterator>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <sched.h>
#endif

#include "compile_test_graph.hpp"
#include "compiler/deployment_plan_lowering.hpp"
#include "compiler/profile_admission.hpp"
#include "core/graph_compiler_access.hpp"
#include "threading/linux_thread_controller.hpp"

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

[[nodiscard]] std::vector<std::string> diagnostic_tokens(
    const std::vector<detail::PlanLoweringDiagnostic>& diagnostics) {
  std::vector<std::string> result;
  result.reserve(diagnostics.size());
  std::ranges::transform(diagnostics, std::back_inserter(result), [](const auto& diagnostic) {
    return diagnostic.token;
  });
  return result;
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
  for (const auto& placement : lowering.support_thread_affinities) {
    result.push_back("support:" + std::to_string(static_cast<unsigned>(placement.role)) + ":" +
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

// REQ: COMPILE-101
TEST(CompileDeployment, AdmitsPermittedCpuOutsideCallerAffinity) {
#if defined(__linux__)
  // A disposable thread prevents affinity setup from contaminating other tests.
  std::thread([] {
    cpu_set_t original;
    CPU_ZERO(&original);
    ASSERT_EQ(sched_getaffinity(0, sizeof(original), &original), 0);
    struct RestoreAffinity {
      cpu_set_t mask;
      ~RestoreAffinity() { EXPECT_EQ(sched_setaffinity(0, sizeof(mask), &mask), 0); }
    };
    const RestoreAffinity restore{original};
    std::vector<std::uint32_t> cpus;
    for (unsigned cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
      if (CPU_ISSET(cpu, &original)) {
        cpus.push_back(cpu);
      }
    }
    if (cpus.size() < 2U) {
      GTEST_SKIP() << "Requires two permitted CPUs";
    }
    cpu_set_t caller;
    CPU_ZERO(&caller);
    CPU_SET(cpus[0], &caller);
    cpu_set_t worker;
    CPU_ZERO(&worker);
    CPU_SET(cpus[1], &worker);
    ASSERT_EQ(sched_setaffinity(0, sizeof(caller), &caller), 0);
    ASSERT_EQ(sched_setaffinity(0, sizeof(worker), &worker), 0);
    cpu_set_t observed;
    ASSERT_EQ(sched_getaffinity(0, sizeof(observed), &observed), 0);
    ASSERT_TRUE(CPU_EQUAL(&observed, &worker));
    ASSERT_EQ(sched_setaffinity(0, sizeof(caller), &caller), 0);

    Graph graph{"permitted-worker-affinity"};
    const auto first = graph.op<test::CompileTestOperator>("first");
    const auto second = graph.op<test::CompileTestOperator>("second");
    graph.partition("first").add(first);
    graph.partition("second").add(second);
    CompileOptions options;
    options.deployment.bind_thread_placement("first",
                                             ThreadPlacement{.affinity = CPUAffinity{cpus[1]}});
    options.deployment.bind_support_thread_affinity(SupportThreadRole::kController,
                                                    CPUAffinity{cpus[0]});
    const ExecutionPlan plan = compile(graph, options);
    EXPECT_TRUE(plan.ok()) << plan.json();
    ASSERT_EQ(sched_getaffinity(0, sizeof(observed), &observed), 0);
    EXPECT_TRUE(CPU_EQUAL(&observed, &caller));
    ASSERT_EQ(plan.partition_placements().size(), 1U);
    const auto& placement = plan.partition_placements().front();
    ASSERT_TRUE(placement.accepted.has_value());
    EXPECT_EQ(placement.requested.affinity->cores, (std::vector<std::uint32_t>{cpus[1]}));
    EXPECT_EQ(placement.accepted->affinity->cores, placement.requested.affinity->cores);
  }).join();
#else
  GTEST_SKIP() << "Linux CPU affinity is required";
#endif
}

// REQ: COMPILE-101
TEST(CompileDeployment, RejectsForbiddenCpuAndPartialMaskInRestrictedCgroup) {
#if defined(__linux__)
  const char* forbidden_text = std::getenv("HOLOSCAN_TEST_FORBIDDEN_CPU");
  if (forbidden_text == nullptr) {
    GTEST_SKIP() << "Requires a restricted cgroup and HOLOSCAN_TEST_FORBIDDEN_CPU";
  }
  const std::string_view text{forbidden_text};
  std::uint32_t forbidden{};
  const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), forbidden);
  ASSERT_EQ(error, std::errc());
  ASSERT_EQ(end, text.data() + text.size());
  ASSERT_LT(forbidden, CPU_SETSIZE);
  std::thread([forbidden] {
    cpu_set_t original;
    ASSERT_EQ(sched_getaffinity(0, sizeof(original), &original), 0);
    struct RestoreAffinity {
      cpu_set_t mask;
      ~RestoreAffinity() { EXPECT_EQ(sched_setaffinity(0, sizeof(mask), &mask), 0); }
    };
    const RestoreAffinity restore{original};
    const auto snapshot = detail::LinuxThreadController::snapshot();
    ASSERT_TRUE(snapshot);
    ASSERT_FALSE(snapshot->affinity.cores.empty());
    cpu_set_t denied;
    CPU_ZERO(&denied);
    CPU_SET(forbidden, &denied);
    ASSERT_EQ(sched_setaffinity(0, sizeof(denied), &denied), -1);
    ASSERT_EQ(errno, EINVAL);
    for (const auto& affinity :
         {CPUAffinity{forbidden}, CPUAffinity{snapshot->affinity.cores.front(), forbidden}}) {
      const auto observation =
          detail::LinuxThreadController::probe(ThreadPlacement{.affinity = affinity});
      ASSERT_TRUE(observation) << observation.error().message;
      EXPECT_EQ(observation->unavailable_cores, (std::vector<std::uint32_t>{forbidden}));
      Graph graph{"forbidden-cpuset"};
      const auto op = graph.op<test::CompileTestOperator>("operator");
      graph.partition("worker").add(op);
      CompileOptions options;
      options.deployment.bind_thread_placement("worker", ThreadPlacement{.affinity = affinity});
      const auto plan = compile(graph, options);
      EXPECT_FALSE(plan.ok());
      EXPECT_TRUE(test::has_diagnostic(plan, "THREAD_PLACEMENT_REJECTED"));
      ASSERT_EQ(plan.partition_placements().size(), 1U);
      EXPECT_FALSE(plan.partition_placements().front().accepted.has_value());
    }
  }).join();
#else
  GTEST_SKIP() << "Linux cgroup integration is required";
#endif
}

// REQ: COMPILE-096
// REQ: COMPILE-099
// REQ: COMPILE-100
TEST(CompileDeployment, LowersSupportThreadAffinityWithOneCanonicalAffinityOnlyContract) {
  Graph graph{"support-thread-affinity"};
  (void)graph.op<test::CompileTestOperator>("operator");
  const auto snapshot = detail::GraphCompilerAccess::canonical_snapshot(graph);
  const auto admission = detail::resolve_profile_admission(*snapshot, Profile::kStandard);
  std::vector partitions{ExecutionPartitionRecord{
      .id = ExecutionPartitionId{1U},
      .members = {snapshot->vertices.front().id},
      .execution_lane = 0U,
  }};
  DeploymentBinding deployment;
  deployment.bind_support_thread_affinity(SupportThreadRole::kChannelPump, CPUAffinity{7U, 3U, 7U})
      .bind_support_thread_affinity(SupportThreadRole::kController, CPUAffinity{5U});
  const FixedPlacementProbe probe(detail::PlacementPlatformProbeResult{
      .disposition = detail::PlacementPlatformProbeDisposition::kPassed,
      .evidence_id = "test.support-thread-affinity.available/v1",
  });

  const auto lowered = detail::lower_deployment_plan(
      *snapshot, deployment, admission.effective_profiles(), partitions, probe);

  EXPECT_TRUE(lowered.errors.empty());
  ASSERT_EQ(lowered.support_thread_affinities.size(), 2U);
  EXPECT_EQ(lowered.support_thread_affinities[0].role, SupportThreadRole::kController);
  EXPECT_EQ(lowered.support_thread_affinities[1].role, SupportThreadRole::kChannelPump);
  ASSERT_TRUE(lowered.support_thread_affinities[0].accepted.has_value());
  ASSERT_TRUE(lowered.support_thread_affinities[1].accepted.has_value());
  EXPECT_EQ(lowered.support_thread_affinities[0].accepted->affinity->cores,
            (std::vector<std::uint32_t>{5U}));
  EXPECT_EQ(lowered.support_thread_affinities[1].accepted->affinity->cores,
            (std::vector<std::uint32_t>{3U, 7U}));
  for (const auto& record : lowered.support_thread_affinities) {
    EXPECT_EQ(record.requested.scheduling.policy, SchedulingPolicy::kOther);
    EXPECT_EQ(record.requested.wake_mode, WakeMode::kSleepingWaitset);
    EXPECT_EQ(record.requested.irq_isolation, IRQIsolation::kNotRequested);
    EXPECT_EQ(record.requested.probe_failure_policy, ProbeFailurePolicy::kFailAdmission);
  }
  EXPECT_EQ(probe.calls, 2U);
}

// REQ: COMPILE-096
// REQ: COMPILE-099
TEST(CompileDeployment, PublicSupportThreadAffinityRetainsAcceptedPlanEvidenceAndIdentity) {
#if defined(__linux__)
  const auto compiling_thread = detail::LinuxThreadController::snapshot();
  ASSERT_TRUE(compiling_thread.has_value());
  ASSERT_FALSE(compiling_thread->affinity.cores.empty());
  CPUAffinity authored{compiling_thread->affinity.cores.front()};
  CompileOptions controller_options;
  controller_options.deployment.bind_support_thread_affinity(SupportThreadRole::kController,
                                                             authored);
  authored.cores.clear();
  ASSERT_EQ(controller_options.deployment.support_thread_affinities().size(), 1U);
  EXPECT_FALSE(controller_options.deployment.support_thread_affinities()
                   .front()
                   .placement.affinity->empty());

  Graph controller_graph{"controller-support-thread-affinity"};
  (void)controller_graph.op<test::CompileTestOperator>("operator");
  const ExecutionPlan controller = compile(controller_graph, controller_options);
  ASSERT_TRUE(controller.ok()) << controller.json();
  ASSERT_EQ(controller.support_thread_affinities().size(), 1U);
  const auto& record = controller.support_thread_affinities().front();
  EXPECT_EQ(record.role, SupportThreadRole::kController);
  EXPECT_EQ(record.disposition, PlanDisposition::kAccepted);
  ASSERT_TRUE(record.accepted.has_value());
  ASSERT_TRUE(record.accepted->affinity.has_value());
  EXPECT_EQ(record.accepted->affinity->cores,
            (std::vector<std::uint32_t>{compiling_thread->affinity.cores.front()}));
  EXPECT_NE(controller.json().find("\"support_thread_affinities\":[{"), std::string::npos);
  EXPECT_TRUE(std::ranges::any_of(
      controller.profile_manifest().required_capabilities, [](const auto& requirement) {
        return requirement.capability_id == "holoscan.runtime.thread_placement.apply_readback" &&
               requirement.version == 1U &&
               requirement.evidence_id == "holoscan.runtime.thread_placement.apply_readback/v1";
      }));

  CompileOptions pump_options;
  pump_options.deployment.bind_support_thread_affinity(
      SupportThreadRole::kChannelPump, CPUAffinity{compiling_thread->affinity.cores.front()});
  Graph pump_graph{"controller-support-thread-affinity"};
  (void)pump_graph.op<test::CompileTestOperator>("operator");
  const ExecutionPlan pump = compile(pump_graph, pump_options);
  ASSERT_TRUE(pump.ok()) << pump.json();
  EXPECT_NE(controller.digest(), pump.digest());
#else
  GTEST_SKIP() << "Support-thread affinity probing is supported only on Linux";
#endif
}

// REQ: COMPILE-096
// REQ: COMPILE-099
TEST(CompileDeployment, RejectsConflictingControllerAndSolePartitionAffinity) {
#if defined(__linux__)
  const auto compiling_thread = detail::LinuxThreadController::snapshot();
  ASSERT_TRUE(compiling_thread.has_value());
  if (compiling_thread->affinity.cores.size() < 2U) {
    GTEST_SKIP() << "The effective compiling-thread affinity contains fewer than two CPUs";
  }
  const std::uint32_t controller_cpu = compiling_thread->affinity.cores[0];
  const std::uint32_t partition_cpu = compiling_thread->affinity.cores[1];
  const auto compile_placement = [&](std::optional<std::uint32_t> worker_cpu) {
    Graph graph{"inline-controller-affinity"};
    const auto op = graph.op<test::CompileTestOperator>("operator");
    graph.partition("worker").add(op);
    CompileOptions options;
    options.deployment.bind_support_thread_affinity(SupportThreadRole::kController,
                                                    CPUAffinity{controller_cpu});
    if (worker_cpu) {
      options.deployment.bind_thread_placement(
          "worker", ThreadPlacement{.affinity = CPUAffinity{*worker_cpu}});
    }
    return compile(graph, std::move(options));
  };

  const ExecutionPlan conflicting = compile_placement(partition_cpu);
  ASSERT_FALSE(conflicting.ok()) << conflicting.json();
  const auto conflict = std::ranges::find(
      conflicting.diagnostics(), std::string{"THREAD_PLACEMENT_REJECTED"}, &Diagnostic::token);
  ASSERT_NE(conflict, conflicting.diagnostics().end());
  EXPECT_EQ(conflict->json_pointer, "/deployment/support_thread_affinities/controller");
  EXPECT_NE(conflict->required.find("share one Runtime thread"), std::string::npos);
  EXPECT_NE(conflict->provided.find("controller=[" + std::to_string(controller_cpu) + "]"),
            std::string::npos);
  EXPECT_NE(conflict->provided.find("partition_affinity=[" + std::to_string(partition_cpu) + "]"),
            std::string::npos);

  const ExecutionPlan matching = compile_placement(controller_cpu);
  EXPECT_TRUE(matching.ok()) << matching.json();
  const ExecutionPlan inherited = compile_placement(std::nullopt);
  EXPECT_TRUE(inherited.ok()) << inherited.json();
#else
  GTEST_SKIP() << "Support-thread affinity probing is supported only on Linux";
#endif
}

// REQ: COMPILE-096
// REQ: COMPILE-099
TEST(CompileDeployment, RejectsInvalidDuplicateAndUnavailableSupportThreadAffinity) {
  Graph graph{"invalid-support-thread-affinity"};
  (void)graph.op<test::CompileTestOperator>("operator");
  const auto snapshot = detail::GraphCompilerAccess::canonical_snapshot(graph);
  const auto admission = detail::resolve_profile_admission(*snapshot, Profile::kStandard);
  std::vector partitions{ExecutionPartitionRecord{
      .id = ExecutionPartitionId{1U},
      .members = {snapshot->vertices.front().id},
      .execution_lane = 0U,
  }};

  DeploymentBinding duplicate;
  duplicate.bind_support_thread_affinity(SupportThreadRole::kController, CPUAffinity{1U})
      .bind_support_thread_affinity(SupportThreadRole::kController, CPUAffinity{2U});
  const FixedPlacementProbe available(detail::PlacementPlatformProbeResult{
      .disposition = detail::PlacementPlatformProbeDisposition::kPassed,
      .evidence_id = "test.support-thread-affinity.available/v1",
  });
  const auto duplicate_result = detail::lower_deployment_plan(
      *snapshot, duplicate, admission.effective_profiles(), partitions, available);
  EXPECT_EQ(diagnostic_tokens(duplicate_result.errors),
            (std::vector<std::string>{"SUPPORT_THREAD_PLACEMENT_UNKNOWN_OR_DUPLICATE"}));
  ASSERT_TRUE(duplicate_result.errors.front().detail.has_value());
  EXPECT_EQ(duplicate_result.errors.front().detail->operator_path, "support-thread/controller");
  ASSERT_EQ(duplicate_result.errors.front().detail->mismatches.size(), 1U);
  EXPECT_EQ(duplicate_result.errors.front().detail->mismatches.front().provided,
            "role=controller,bindings=2");
  EXPECT_EQ(available.calls, 0U);

  DeploymentBinding malformed;
  // Supply an unknown role to exercise deployment rejection.
  // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange)
  malformed.bind_support_thread_affinity(static_cast<SupportThreadRole>(255U), CPUAffinity{1U});
  malformed.bind_support_thread_affinity(SupportThreadRole::kChannelPump, CPUAffinity{});
  const auto malformed_result = detail::lower_deployment_plan(
      *snapshot, malformed, admission.effective_profiles(), partitions, available);
  EXPECT_EQ(diagnostic_tokens(malformed_result.errors),
            (std::vector<std::string>{"SUPPORT_THREAD_PLACEMENT_UNKNOWN_OR_DUPLICATE",
                                      "THREAD_PLACEMENT_REJECTED"}));
  ASSERT_TRUE(malformed_result.errors.back().detail.has_value());
  EXPECT_EQ(malformed_result.errors.back().detail->operation, DiagnosticOperation::kCompile);
  EXPECT_NE(malformed_result.errors.back().detail->mismatches.front().provided.find(
                "role=channel_pump,requested=[]"),
            std::string::npos);

  DeploymentBinding unavailable_binding;
  unavailable_binding.bind_support_thread_affinity(SupportThreadRole::kChannelPump,
                                                   CPUAffinity{3U});
  const FixedPlacementProbe unavailable(detail::PlacementPlatformProbeResult{
      .disposition = detail::PlacementPlatformProbeDisposition::kUnavailable,
      .evidence_id = "test.support-thread-affinity.unavailable/v1",
      .effective_affinity = CPUAffinity{1U, 2U},
  });
  const auto unavailable_result = detail::lower_deployment_plan(
      *snapshot, unavailable_binding, admission.effective_profiles(), partitions, unavailable);
  EXPECT_EQ(diagnostic_tokens(unavailable_result.errors),
            (std::vector<std::string>{"THREAD_PLACEMENT_REJECTED"}));
  ASSERT_EQ(unavailable_result.support_thread_affinities.size(), 1U);
  EXPECT_EQ(unavailable_result.support_thread_affinities.front().disposition,
            PlanDisposition::kRejected);
  ASSERT_TRUE(unavailable_result.errors.front().detail.has_value());
  const auto& unavailable_detail = *unavailable_result.errors.front().detail;
  EXPECT_EQ(unavailable_detail.operator_path, "support-thread/channel_pump");
  ASSERT_EQ(unavailable_detail.mismatches.size(), 1U);
  EXPECT_NE(
      unavailable_detail.mismatches.front().provided.find("requested=[3],current_affinity=[1,2]"),
      std::string::npos);
  EXPECT_NE(unavailable_detail.mismatches.front().required.find("proven permitted by Linux"),
            std::string::npos);
  EXPECT_EQ(unavailable.calls, 1U);
}

// REQ: COMPILE-096
// REQ: COMPILE-099
TEST(CompileDeployment, RetainsDistinctSupportThreadRoleDiagnostics) {
  Graph empty_graph{"distinct-empty-support-thread-affinity"};
  (void)empty_graph.op<test::CompileTestOperator>("operator");
  CompileOptions empty_options;
  empty_options.deployment
      .bind_support_thread_affinity(SupportThreadRole::kController, CPUAffinity{})
      .bind_support_thread_affinity(SupportThreadRole::kChannelPump, CPUAffinity{});

  const ExecutionPlan empty_plan = compile(empty_graph, empty_options);

  ASSERT_FALSE(empty_plan.ok());
  EXPECT_EQ(
      std::ranges::count(
          empty_plan.diagnostics(), std::string{"THREAD_PLACEMENT_REJECTED"}, &Diagnostic::token),
      2U);
  for (const auto [role, pointer] :
       {std::pair{std::string_view{"controller"},
                  std::string_view{"/deployment/support_thread_affinities/controller"}},
        std::pair{std::string_view{"channel_pump"},
                  std::string_view{"/deployment/support_thread_affinities/channel_pump"}}}) {
    const auto diagnostic =
        std::ranges::find(empty_plan.diagnostics(), pointer, &Diagnostic::json_pointer);
    ASSERT_NE(diagnostic, empty_plan.diagnostics().end());
    ASSERT_TRUE(diagnostic->detail.has_value());
    EXPECT_EQ(diagnostic->detail->operator_path, "support-thread/" + std::string(role));
    EXPECT_NE(format_diagnostic(*diagnostic, empty_plan.graph()).find(role), std::string::npos);
  }

  Graph unknown_graph{"distinct-unknown-support-thread-affinity"};
  (void)unknown_graph.op<test::CompileTestOperator>("operator");
  // Supply unknown roles to exercise deployment rejection.
  // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange)
  const auto unknown_role_254 = static_cast<SupportThreadRole>(254U);
  // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange)
  const auto unknown_role_255 = static_cast<SupportThreadRole>(255U);
  CompileOptions unknown_options;
  unknown_options.deployment.bind_support_thread_affinity(unknown_role_254, CPUAffinity{1U})
      .bind_support_thread_affinity(unknown_role_255, CPUAffinity{1U});

  const ExecutionPlan unknown_plan = compile(unknown_graph, unknown_options);

  ASSERT_FALSE(unknown_plan.ok());
  EXPECT_EQ(std::ranges::count(unknown_plan.diagnostics(),
                               std::string{"SUPPORT_THREAD_PLACEMENT_UNKNOWN_OR_DUPLICATE"},
                               &Diagnostic::token),
            2U);
  for (const std::uint32_t role : {254U, 255U}) {
    const std::string pointer =
        "/deployment/support_thread_affinities/unknown-" + std::to_string(role);
    const auto diagnostic =
        std::ranges::find(unknown_plan.diagnostics(), pointer, &Diagnostic::json_pointer);
    ASSERT_NE(diagnostic, unknown_plan.diagnostics().end());
    ASSERT_TRUE(diagnostic->detail.has_value());
    EXPECT_EQ(diagnostic->detail->mismatches.front().provided, "value=" + std::to_string(role));
  }
}

// REQ: COMPILE-096
// REQ: COMPILE-099
TEST(CompileDeployment, PublishesActionableSupportThreadAffinityFailureWithoutChangingToken) {
  Graph graph{"invalid-support-thread-affinity-detail"};
  (void)graph.op<test::CompileTestOperator>("operator");
  CompileOptions options;
  // Supply an unknown role to exercise deployment rejection.
  // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange)
  options.deployment.bind_support_thread_affinity(static_cast<SupportThreadRole>(255U),
                                                  CPUAffinity{1U});

  const ExecutionPlan plan = compile(graph, options);

  ASSERT_FALSE(plan.ok());
  const auto diagnostic =
      std::ranges::find(plan.diagnostics(),
                        std::string{"SUPPORT_THREAD_PLACEMENT_UNKNOWN_OR_DUPLICATE"},
                        &Diagnostic::token);
  ASSERT_NE(diagnostic, plan.diagnostics().end());
  ASSERT_TRUE(diagnostic->detail.has_value());
  EXPECT_EQ(diagnostic->detail->operation, DiagnosticOperation::kCompile);
  EXPECT_EQ(diagnostic->detail->operator_path, "support-thread/unknown");
  ASSERT_EQ(diagnostic->detail->mismatches.size(), 1U);
  EXPECT_EQ(diagnostic->detail->mismatches.front().provided, "value=255");
  const std::string formatted = format_diagnostic(*diagnostic, plan.graph());
  EXPECT_NE(formatted.find("support_thread_role"), std::string::npos);
  EXPECT_NE(formatted.find("value=255"), std::string::npos);
}

// REQ: COMPILE-002
// REQ: COMPILE-009
// REQ: COMPILE-095
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
// REQ: COMPILE-095
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
// REQ: COMPILE-095
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
  ASSERT_TRUE(proven.errors.empty());
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
  ASSERT_TRUE(fallback.errors.empty());
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
  EXPECT_EQ(diagnostic_tokens(rejected.errors),
            (std::vector<std::string>{"THREAD_PLACEMENT_REJECTED"}));
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
  EXPECT_EQ(diagnostic_tokens(unavailable.errors),
            (std::vector<std::string>{"THREAD_PLACEMENT_REJECTED"}));
  ASSERT_NE(find_probe(unavailable, "holoscan.placement.platform/v1"), nullptr);
  EXPECT_EQ(find_probe(unavailable, "holoscan.placement.platform/v1")->status,
            PlacementProbeStatus::kRejected);
}

// REQ: COMPILE-009
// REQ: COMPILE-095
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
  EXPECT_TRUE(forward.errors.empty());
  EXPECT_TRUE(reverse.errors.empty());
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
