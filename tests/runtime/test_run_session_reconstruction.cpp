/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <ranges>
#include <string>
#include <string_view>
#include <utility>

#include <gtest/gtest.h>

#include <holoscan/core/diagnostic.hpp>
#include <holoscan/time/synthetic_clock.hpp>

#include "compiler/compile_internal.hpp"
#include "compiler/execution_plan_state.hpp"
#include "compiler/process_local_channel_plan_validation.hpp"
#include "run_session_test_support.hpp"
#include "runtime/execution_lane.hpp"
#include "runtime/graph_exec.hpp"
#include "runtime/parameter_update_coordinator.hpp"
#include "runtime/physical_readiness_binding.hpp"
#include "runtime/runtime_partition_test_plan_access.hpp"
#include "runtime/runtime_phase_counters.hpp"
#include "runtime/runtime_test_access.hpp"
#include "runtime/runtime_work_inventory.hpp"
#include "runtime/source_wait.hpp"
#include "runtime_channel_binding_test_support.hpp"
#include "runtime_test_graph.hpp"
#include "support/direct_port_binding.hpp"

namespace holoscan {

namespace {

using namespace std::chrono_literals;

using namespace test::run_session;

class OptionalLifecycleReplayProbe final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    if (!omit_callbacks.load(std::memory_order_relaxed)) {
      spec.lifecycle()
          .optional(LifecycleStage::kStart, &OptionalLifecycleReplayProbe::start_stage)
          .optional(LifecycleStage::kStop, &OptionalLifecycleReplayProbe::stop_stage);
    }
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 1ms});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override { return {}; }
  LifecycleStatus start_stage(LifecycleContext&) noexcept { return LifecycleStatus::kOk; }
  LifecycleStatus stop_stage(LifecycleContext&) noexcept { return LifecycleStatus::kOk; }

  static inline std::atomic_bool omit_callbacks{};
};

class CudaGraphEligibilityReplayProbe final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.cuda_graph_capture_eligible(eligible.load(std::memory_order_relaxed));
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 1ms});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override { return {}; }

  static inline std::atomic_bool eligible{true};
};

// Starts the connected index fixture from a clock and exposes its first output. Used by
// GraphExecIndexTest.ResolvesDenseVertexIndexWithAbsentNonOperatorSlots.
class GraphExecIndexProbeOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override { spec.output(output, "output"); }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 1ms});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Output<std::uint32_t> output;
};

// Extends the index fixture with one input-triggered callback and output. Used by
// GraphExecIndexTest.ResolvesDenseVertexIndexWithAbsentNonOperatorSlots.
class GraphExecIndexRelayOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(input, "input");
    spec.output(output, "output");
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnAnyInput{}).input(input, DrainAll{});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Input<std::uint32_t> input;
  Output<std::uint32_t> output;
};

// REQ: RUNTIME-007
// REQ: RUNTIME-013
// REQ: RUNTIME-080
TEST(RunSessionTest, RejectsStrayDeadlineInventoryInOperatorFreePlan) {
  Graph graph;
  const ExecutionPlan baseline = compile(graph);
  ASSERT_TRUE(baseline.ok()) << baseline.json();
  const ExecutionPlan malformed =
      detail::RuntimeTestPlanAccess::with_stray_max_start_delay(baseline);
  ASSERT_TRUE(malformed.ok()) << malformed.json();

  RunSession session = run_async(malformed);
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kFailed);
  ASSERT_EQ(session.diagnostics().size(), 1U);
  EXPECT_EQ(session.diagnostics().front().token, "RUNTIME_OPERATOR_REPLAY_MISMATCH");
  EXPECT_TRUE(session.diagnostics().front().required.empty());
  EXPECT_TRUE(session.diagnostics().front().provided.empty());
}

// REQ: RUNTIME-001
// REQ: RUNTIME-011
TEST(RunSessionTest, RunsCompilerAcceptedEmptyGraphAsNoOp) {
  // Graph topology:
  //   (empty)
  Graph graph{"empty-runtime-plan"};
  // Key setup: compile an empty graph with a diagnostic sink attached to its accepted plan.
  const ExecutionPlan compiled = compile(graph);
  const auto sink = std::make_shared<RecordingDiagnosticSink>();
  const ExecutionPlan plan = detail::RuntimeTestPlanAccess::with_diagnostic_sink(compiled, sink);
  ASSERT_TRUE(plan.ok());
  EXPECT_TRUE(plan.graph().vertices().empty());
  EXPECT_TRUE(plan.execution_partitions().empty());
  EXPECT_TRUE(plan.partition_decisions().empty());
  EXPECT_TRUE(plan.execution_lanes().empty());
  EXPECT_TRUE(plan.partition_placements().empty());
  EXPECT_EQ(plan.scheduler_plan().execution_lane_count, 0U);
  EXPECT_EQ(plan.scheduler_plan().worker_count, 0U);
  EXPECT_TRUE(plan.profile_manifest().materialization_obligations.empty());

  const auto accepted = detail::ExecutionPlanAccess::retain(plan);
  ASSERT_NE(accepted, nullptr);
  EXPECT_FALSE(detail::has_structural_runtime_topology(*accepted));
  EXPECT_FALSE(detail::has_plan_callback_owners(*accepted));
  auto reconstructed = detail::GraphExec::reconstruct(accepted, RunGenerationId{17U});
  ASSERT_TRUE(reconstructed) << reconstructed.error().message;
  EXPECT_FALSE((*reconstructed)->has_process_local_callback_owners());
  EXPECT_TRUE((*reconstructed)->parameter_updates().catalog().operators().empty());
  EXPECT_EQ((*reconstructed)->parameter_updates().max_event_cursors(), 0U);

  RunSession session = run_async(plan);
  session.wait();

  // Key assertion: empty topology authenticates and completes without inventing callback owners.
  EXPECT_EQ(session.termination(), RunTermination::kCompleted);
  EXPECT_TRUE(session.diagnostics().empty());
  const auto lane_evidence = sink->snapshot_lane_evidence();
  ASSERT_EQ(lane_evidence.size(), 1U);
  EXPECT_TRUE(lane_evidence.front().empty());
  EXPECT_EQ(run(plan, RunSignalPolicy::kCallerManaged), 0);
}

TEST(GraphExecIndexTest, ResolvesDenseVertexIndexWithAbsentNonOperatorSlots) {
  constexpr std::size_t kOperatorCount = 32U;
  // Graph:
  //   operator-000 -> operator-001 -> ... -> operator-031 -> exported topic
  Graph graph{"graph-exec-dense-index"};
  const auto first = graph.op<GraphExecIndexProbeOperator>("operator-000");
  auto exported = graph.export_topic<std::uint32_t>("holomq://graph-exec-index-topic");
  auto previous = graph.op<GraphExecIndexRelayOperator>("operator-001");
  graph.add_flow(first->output, previous->input);
  for (std::size_t index = 2U; index < kOperatorCount; ++index) {
    const auto next = graph.op<GraphExecIndexRelayOperator>("operator-" + std::to_string(index));
    graph.add_flow(previous->output, next->input);
    previous = next;
  }
  graph.add_flow(previous->output, exported.input());
  graph.set_default_clock(graph.add_clock<test::RuntimeProbeClock>("graph-exec-index-clock"));
  const ExecutionPlan plan = compile(graph);
  std::string diagnostic_tokens;
  for (const Diagnostic& diagnostic : plan.diagnostics()) {
    diagnostic_tokens += diagnostic.token + "\n";
  }
  ASSERT_TRUE(plan.ok()) << diagnostic_tokens;
  const auto accepted = detail::ExecutionPlanAccess::retain(plan);
  ASSERT_NE(accepted, nullptr);

  // Key setup: one non-Operator exported-topic vertex creates an absent slot within the otherwise
  // contiguous accepted VertexId range.
  ASSERT_EQ(accepted->graph.vertices().size(), kOperatorCount + 1U);
  EXPECT_TRUE(detail::has_structural_runtime_topology(*accepted));
  EXPECT_TRUE(detail::has_plan_callback_owners(*accepted));
  const auto non_operator = std::ranges::find_if(
      accepted->graph.vertices(),
      [](const VertexRecord& vertex) { return vertex.kind != VertexKind::kOperator; });
  ASSERT_NE(non_operator, accepted->graph.vertices().end());
  auto reconstructed = detail::GraphExec::reconstruct(accepted, RunGenerationId{17U});
  ASSERT_TRUE(reconstructed) << reconstructed.error().message;
  detail::GraphExec& exec = **reconstructed;
  EXPECT_TRUE(exec.has_process_local_callback_owners());
  ASSERT_EQ(exec.operators().size(), kOperatorCount);

  // Key assertion: first/last identities resolve in O(1), the non-Operator and invalid slots remain
  // absent, and both const and mutable accessors identify the exact reconstructed object.
  const VertexId first_vertex = exec.operators().front().vertex;
  const VertexId last_vertex = exec.operators().back().vertex;
  EXPECT_EQ(exec.receiver_index(first_vertex), 0U);
  EXPECT_EQ(exec.receiver_index(last_vertex), kOperatorCount - 1U);
  EXPECT_EQ(exec.receiver_index(exec.operators().front().operator_id), 0U);
  EXPECT_EQ(exec.receiver_index(exec.operators().back().operator_id), kOperatorCount - 1U);
  EXPECT_FALSE(exec.receiver_index(non_operator->id));
  EXPECT_FALSE(exec.receiver_index(VertexId{}));
  EXPECT_FALSE(exec.receiver_index(VertexId{accepted->graph.vertices().size() + 1U}));
  EXPECT_FALSE(exec.receiver_index(OperatorInstanceId{}));
  EXPECT_FALSE(exec.receiver_index(static_cast<OperatorInstanceId>(kOperatorCount + 1U)));
  EXPECT_EQ(exec.operator_by_vertex(first_vertex), &exec.operators().front());
  EXPECT_EQ(exec.operator_by_id(exec.operators().back().operator_id), &exec.operators().back());
  const detail::GraphExec& const_exec = exec;
  EXPECT_EQ(const_exec.operator_by_vertex(last_vertex), &const_exec.operators().back());
  EXPECT_EQ(const_exec.operator_by_id(const_exec.operators().front().operator_id),
            &const_exec.operators().front());
}

TEST(GraphExecIndexTest, RejectsDuplicateReconstructedVertexIdentityBeforePublication) {
  const ExecutionPlan baseline = test::make_runtime_multi_lane_test_plan();
  ASSERT_TRUE(baseline.ok()) << baseline.json();
  const auto accepted = detail::ExecutionPlanAccess::retain(baseline);
  ASSERT_NE(accepted, nullptr);
  ASSERT_GE(accepted->executable_factories.operators.size(), 2U);
  auto malformed = std::make_shared<detail::ExecutionPlanState>(*accepted);

  // Key setup: two executable factories claim the exact same accepted graph vertex.
  malformed->executable_factories.operators[1].vertex =
      malformed->executable_factories.operators.front().vertex;
  const auto reconstructed =
      detail::GraphExec::reconstruct(std::move(malformed), RunGenerationId{17U});

  // Key assertion: the shared accepted-cover authority now rejects the factory-derived partition
  // lookup mismatch before a GraphExec with an overwritten index can escape.
  ASSERT_FALSE(reconstructed);
  EXPECT_EQ(reconstructed.error().code, ErrorCode::kNotSupported);
  EXPECT_EQ(reconstructed.error().message, "RUNTIME_PLACEMENT_BINDING_UNAVAILABLE");
}

// REQ: OPERATOR-050
TEST(RunSessionTest, RejectsEmptyOperatorPathBeforeReconstruction) {
  const ExecutionPlan baseline = test::make_runtime_test_plan();
  ASSERT_TRUE(baseline.ok()) << baseline.json();
  const auto accepted = detail::ExecutionPlanAccess::retain(baseline);
  ASSERT_NE(accepted, nullptr);
  auto malformed = std::make_shared<detail::ExecutionPlanState>(*accepted);
  ASSERT_FALSE(malformed->operator_parameters.empty());

  malformed->operator_parameters.front().operator_path.clear();
  test::RuntimeProbeState::reset();
  const auto reconstructed =
      detail::GraphExec::reconstruct(std::move(malformed), RunGenerationId{17U});

  ASSERT_FALSE(reconstructed);
  EXPECT_EQ(reconstructed.error().code, ErrorCode::kFailure);
  EXPECT_EQ(reconstructed.error().message, "RUNTIME_PARAMETER_INVENTORY_MISMATCH");
  EXPECT_EQ(test::RuntimeProbeState::constructed.load(std::memory_order_acquire), 0U);
  EXPECT_FALSE(test::RuntimeProbeState::setup_entered.load(std::memory_order_acquire));
}

// REQ: PARAM-017
TEST(RunSessionTest, RejectsNoncontiguousParameterOwnerInventoriesBeforeReconstruction) {
  using Mutation = detail::RuntimeTestPlanAccess::ParameterInventoryMutation;
  const std::array cases{
      std::pair{test::make_runtime_two_param_test_plan(), Mutation::kInterleavedFields},
      std::pair{test::make_runtime_two_deadline_param_test_plan(),
                Mutation::kInterleavedDerivedTerms},
  };
  for (const auto& [baseline, mutation] : cases) {
    ASSERT_TRUE(baseline.ok()) << baseline.json();
    const ExecutionPlan malformed =
        detail::RuntimeTestPlanAccess::mutate_parameter_inventory(baseline, mutation);
    ASSERT_TRUE(malformed.ok()) << malformed.json();
    const auto accepted = detail::ExecutionPlanAccess::retain(malformed);
    ASSERT_NE(accepted, nullptr);
    const auto reconstructed = detail::GraphExec::reconstruct(accepted, RunGenerationId{17U});
    ASSERT_FALSE(reconstructed);
    EXPECT_EQ(reconstructed.error().code, ErrorCode::kFailure);
    EXPECT_EQ(reconstructed.error().message, "RUNTIME_PARAMETER_INVENTORY_MISMATCH");
  }
}

// REQ: PARAM-017
TEST(RunSessionTest, RejectsZeroParameterOwnersBeforeReconstruction) {
  using Mutation = detail::RuntimeTestPlanAccess::ParameterInventoryMutation;
  const std::array cases{
      std::pair{test::make_runtime_two_param_test_plan(), Mutation::kZeroFieldOwner},
      std::pair{test::make_runtime_two_deadline_param_test_plan(), Mutation::kZeroDerivedTermOwner},
  };
  for (const auto& [baseline, mutation] : cases) {
    ASSERT_TRUE(baseline.ok()) << baseline.json();
    // Key setup: corrupt one otherwise valid owner reference with the reserved zero identity.
    const ExecutionPlan malformed =
        detail::RuntimeTestPlanAccess::mutate_parameter_inventory(baseline, mutation);
    ASSERT_TRUE(malformed.ok()) << malformed.json();
    const auto accepted = detail::ExecutionPlanAccess::retain(malformed);
    ASSERT_NE(accepted, nullptr);

    const auto reconstructed = detail::GraphExec::reconstruct(accepted, RunGenerationId{17U});

    // Key assertion: startup rejects zero before using it to index owner-group state.
    ASSERT_FALSE(reconstructed);
    EXPECT_EQ(reconstructed.error().code, ErrorCode::kFailure);
    EXPECT_EQ(reconstructed.error().message, "RUNTIME_PARAMETER_INVENTORY_MISMATCH");
  }
}

// REQ: RUNTIME-010
// REQ: RUNTIME-011
TEST(RunSessionTest, EmptyGraphTreatsAuthoredClockCatalogAsInertAuthoringEvidence) {
  Graph graph{"empty-runtime-plan-with-clocks"};
  const ClockKey default_clock = graph.add_clock<test::RuntimeProbeClock>("default-clock");
  const ClockKey comparison_clock = graph.add_clock<test::RuntimeProbeClock>("comparison-clock");
  graph.set_default_clock(default_clock);
  graph.add_clock_conversion(default_clock, comparison_clock, AffineClockConversion{});
  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.clock_catalog().clocks.size(), 2U);
  ASSERT_EQ(plan.clock_conversions().size(), 1U);
  EXPECT_TRUE(plan.profile_manifest().materialization_obligations.empty());

  test::RuntimeProbeState::reset();
  RunSession session = run_async(plan);
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kCompleted);
  EXPECT_TRUE(session.diagnostics().empty());
  EXPECT_EQ(test::RuntimeProbeState::clocks_constructed.load(std::memory_order_relaxed), 0U);
}

// REQ: RUNTIME-010
TEST(RunSessionTest, EmptyGraphRejectsTamperedInertClockInventoryWithoutConstruction) {
  using InertClockMutation = detail::RuntimeTestPlanAccess::InertClockMutation;
  constexpr std::array mutations{
      InertClockMutation::kPlanClockKey,
      InertClockMutation::kMissingFactory,
      InertClockMutation::kFactoryMode,
      InertClockMutation::kRootDefault,
      InertClockMutation::kConversionGeneration,
      InertClockMutation::kConversionEndpoint,
  };
  Graph graph{"empty-runtime-plan-with-corrupted-clocks"};
  const ClockKey default_clock = graph.add_clock<test::RuntimeProbeClock>("default-clock");
  const ClockKey comparison_clock = graph.add_clock<test::RuntimeProbeClock>("comparison-clock");
  graph.set_default_clock(default_clock);
  graph.add_clock_conversion(default_clock, comparison_clock, AffineClockConversion{});
  const ExecutionPlan baseline = compile(graph);
  ASSERT_TRUE(baseline.ok()) << baseline.json();

  for (const InertClockMutation mutation : mutations) {
    SCOPED_TRACE(static_cast<unsigned>(mutation));
    test::RuntimeProbeState::reset();
    const ExecutionPlan plan =
        detail::RuntimeTestPlanAccess::mutate_inert_clock(baseline, mutation);
    ASSERT_TRUE(plan.ok());

    RunSession session = run_async(plan);
    session.wait();

    EXPECT_EQ(session.termination(), RunTermination::kFailed);
    EXPECT_TRUE(test::has_runtime_diagnostic(session, "RUNTIME_CLOCK_INVENTORY_MISMATCH"));
    EXPECT_EQ(test::RuntimeProbeState::clocks_constructed.load(std::memory_order_relaxed), 0U);
  }
}

// REQ: RUNTIME-011
TEST(RunSessionTest, RejectsMalformedZeroWorkSchedulerAndPartitionFacts) {
  using ZeroWorkMutation = detail::RuntimePartitionTestPlanAccess::ZeroWorkMutation;
  constexpr std::array mutations{
      ZeroWorkMutation::kWorkerCount,
      ZeroWorkMutation::kExecutionLaneCount,
      ZeroWorkMutation::kMaxConcurrentActivations,
      ZeroWorkMutation::kIntraPathParallelism,
      ZeroWorkMutation::kExecutionLane,
      ZeroWorkMutation::kOrphanPartition,
      ZeroWorkMutation::kPartitionDecision,
      ZeroWorkMutation::kScheduleEdge,
      ZeroWorkMutation::kPlacementProbe,
      ZeroWorkMutation::kPartitionPlacement,
  };
  Graph graph{"malformed-empty-runtime-plan"};
  const ExecutionPlan baseline = compile(graph);
  ASSERT_TRUE(baseline.ok());

  for (const ZeroWorkMutation mutation : mutations) {
    SCOPED_TRACE(static_cast<unsigned>(mutation));
    const ExecutionPlan plan =
        detail::RuntimePartitionTestPlanAccess::mutate_zero_work(baseline, mutation);
    RunSession session = run_async(plan);
    session.wait();

    EXPECT_EQ(session.termination(), RunTermination::kFailed);
    EXPECT_TRUE(test::has_runtime_diagnostic(session, "RUNTIME_PLACEMENT_BINDING_UNAVAILABLE"));
  }
}

// REQ: RUNTIME-010
TEST(RunSessionTest, RejectsMalformedZeroWorkRuntimeOwnedInventoriesBeforeReconstruction) {
  using ZeroWorkMutation = detail::RuntimePartitionTestPlanAccess::ZeroWorkMutation;
  struct Case {
    ZeroWorkMutation mutation;
    std::string_view token;
  };
  constexpr std::array cases{
      Case{ZeroWorkMutation::kOperatorFactory, "RUNTIME_OPERATOR_INVENTORY_MISMATCH"},
      Case{ZeroWorkMutation::kMaterializationObligation, "PROFILE_MATERIALIZATION_FAILED"},
  };
  Graph graph{"malformed-empty-runtime-inventory"};
  const ExecutionPlan baseline = compile(graph);
  ASSERT_TRUE(baseline.ok());

  for (const auto& test_case : cases) {
    SCOPED_TRACE(test_case.token);
    const ExecutionPlan plan =
        detail::RuntimePartitionTestPlanAccess::mutate_zero_work(baseline, test_case.mutation);
    RunSession session = run_async(plan);
    session.wait();

    EXPECT_EQ(session.termination(), RunTermination::kFailed);
    EXPECT_TRUE(test::has_runtime_diagnostic(session, test_case.token));
  }
}

// REQ: RUNTIME-010
TEST(RunSessionTest, EmptyGraphStillRejectsEveryPackageMismatch) {
  using ProfileMutation = detail::RuntimeTestPlanAccess::ProfileMutation;
  struct Case {
    ProfileMutation mutation;
    std::string_view token;
  };
  constexpr std::array cases{
      Case{ProfileMutation::kPlanSchema, "PROFILE_PLAN_SCHEMA_UNSUPPORTED"},
      Case{ProfileMutation::kProfilePolicy, "PROFILE_POLICY_UNSUPPORTED"},
      Case{ProfileMutation::kRuntimeAbi, "PROFILE_RUNTIME_ABI_INCOMPATIBLE"},
      Case{ProfileMutation::kTarget, "PROFILE_TARGET_INCOMPATIBLE"},
      Case{ProfileMutation::kCapability, "PROFILE_CAPABILITY_UNSATISFIED"},
      Case{ProfileMutation::kEvidence, "PROFILE_EVIDENCE_UNSATISFIED"},
  };
  Graph graph{"empty-runtime-profile-plan"};
  const ExecutionPlan baseline = compile(graph);
  ASSERT_TRUE(baseline.ok());

  for (const auto& test_case : cases) {
    SCOPED_TRACE(test_case.token);
    const ExecutionPlan plan =
        detail::RuntimeTestPlanAccess::mutate_profile(baseline, test_case.mutation);
    ASSERT_TRUE(plan.ok());

    RunSession session = run_async(plan);
    session.wait();

    EXPECT_EQ(session.termination(), RunTermination::kFailed);
    EXPECT_TRUE(test::has_runtime_diagnostic(session, test_case.token));
  }
}

// REQ: RUNTIME-002
TEST(RunSessionTest, ReconstructsEveryPlannedClockCapability) {
  const auto expect_capabilities = [](const ExecutionPlan& source) {
    ASSERT_EQ(source.clock_catalog().clocks.size(), 1U);
    const ClockPlanRecord planned = source.clock_catalog().clocks.front();
    const ExecutionPlan plan = test::enable_direct_runtime(source);
    test::RuntimeProbeState::reset();

    RunSession session = run_async(plan);
    session.wait();

    EXPECT_EQ(session.termination(), RunTermination::kCompleted);
    EXPECT_EQ(test::RuntimeProbeState::observed_clock_can_read_now.load(std::memory_order_relaxed),
              planned.can_read_now);
    EXPECT_EQ(
        test::RuntimeProbeState::observed_clock_can_wait_until.load(std::memory_order_relaxed),
        planned.can_wait_until);
    EXPECT_EQ(test::RuntimeProbeState::observed_clock_steady.load(std::memory_order_relaxed),
              planned.steady);
  };

  expect_capabilities(test::make_runtime_test_plan_with_clock<RealtimeClock>(false));
  expect_capabilities(test::make_runtime_test_plan_with_clock<ManualClock>(false, std::int64_t{0}));
  expect_capabilities(
      test::make_runtime_test_plan_with_clock<SyntheticClock>(false, std::int64_t{0}));
  expect_capabilities(test::make_runtime_test_plan());
  EXPECT_FALSE(test::RuntimeProbeState::observed_clock_steady.load(std::memory_order_relaxed));
}

// REQ: RUNTIME-002
// REQ: RUNTIME-004
// REQ: RUNTIME-027
TEST(RunSessionTest, RejectsReplayedContractMismatchBeforeLifecycleCallbacks) {
  struct CallbackProbeReset {
    CallbackProbeReset() {
      test::RuntimeParamProbeState::reset();
      test::RuntimeDeadlineParamProbeState::reset();
    }
    ~CallbackProbeReset() {
      test::RuntimeParamProbeState::reset();
      test::RuntimeDeadlineParamProbeState::reset();
    }
  };
  const CallbackProbeReset callback_probe_reset{};
  test::RuntimeProbeState::reset();
  const ExecutionPlan plan = test::enable_direct_runtime(test::make_runtime_test_plan());
  test::RuntimeProbeState::reset(test::RuntimeProbeMode::kMismatchedContract);

  RunSession session = run_async(plan);
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kFailed);
  ASSERT_EQ(session.diagnostics().size(), 1U);
  EXPECT_EQ(session.diagnostics().front().token, "RUNTIME_OPERATOR_REPLAY_FINGERPRINT_MISMATCH");
  {
    const std::lock_guard lock(test::RuntimeParamProbeState::mutex);
    EXPECT_FALSE(test::RuntimeParamProbeState::start_entered);
    EXPECT_FALSE(test::RuntimeParamProbeState::stop_entered);
  }
  EXPECT_EQ(test::RuntimeDeadlineParamProbeState::compute_count.load(std::memory_order_acquire),
            0U);
}

// REQ: RUNTIME-004
// REQ: LIFECYCLE-002
TEST(RunSessionTest, RejectsMissingAdvancedOptionalLifecycleCallbacksDuringReplay) {
  struct ReplayReset {
    ~ReplayReset() {
      OptionalLifecycleReplayProbe::omit_callbacks.store(false, std::memory_order_relaxed);
    }
  };
  const ReplayReset reset{};

  Graph graph{"optional-lifecycle-replay"};
  static_cast<void>(graph.op<OptionalLifecycleReplayProbe>("probe"));
  const ClockKey default_clock = graph.add_clock<RealtimeClock>("runtime-test-clock");
  graph.set_default_clock(default_clock);
  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();

  OptionalLifecycleReplayProbe::omit_callbacks.store(true, std::memory_order_relaxed);
  auto reconstructed = detail::GraphExec::reconstruct(detail::ExecutionPlanAccess::retain(plan),
                                                      RunGenerationId{29U});

  ASSERT_FALSE(reconstructed);
  EXPECT_EQ(reconstructed.error().message, "RUNTIME_OPERATOR_REPLAY_MISMATCH");
}

// REQ: OPERATOR-051
TEST(RunSessionTest, RejectsCudaGraphEligibilityReplayMismatchBeforeCallbacks) {
  struct EligibilityReset {
    ~EligibilityReset() {
      CudaGraphEligibilityReplayProbe::eligible.store(true, std::memory_order_relaxed);
    }
  };
  const EligibilityReset reset{};

  CudaGraphEligibilityReplayProbe::eligible.store(true, std::memory_order_relaxed);
  Graph graph{"cuda-graph-eligibility-runtime-replay"};
  (void)graph.op<CudaGraphEligibilityReplayProbe>("probe");
  graph.set_default_clock(graph.add_clock<RealtimeClock>("runtime-test-clock"));
  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();

  CudaGraphEligibilityReplayProbe::eligible.store(false, std::memory_order_relaxed);
  detail::RuntimeReplayMismatchDetail mismatch;
  auto reconstructed = detail::GraphExec::reconstruct(
      detail::ExecutionPlanAccess::retain(plan), RunGenerationId{30U}, {}, &mismatch);

  ASSERT_FALSE(reconstructed);
  EXPECT_EQ(reconstructed.error().message, "RUNTIME_OPERATOR_REPLAY_MISMATCH");
  ASSERT_TRUE(mismatch.present);
  EXPECT_EQ(mismatch.category, detail::RuntimeReplayMismatchCategory::kDeclaration);
}

// A runtime-mutable deadline retains its independently frozen violation action. Replaying the
// same Parameter dependency must not allow a different action to bypass cold reconstruction.
// REQ: RUNTIME-004
// REQ: RUNTIME-027
// REQ: PARAM-017
TEST(RunSessionTest, RejectsParameterDerivedDeadlineViolationReplayMismatchBeforeCallbacks) {
  struct ReplayViolationReset {
    ~ReplayViolationReset() { test::RuntimeDeadlineParamProbeState::reset(); }
  };
  const ReplayViolationReset reset{};

  test::RuntimeProbeState::reset();
  test::RuntimeParamProbeState::reset();
  test::RuntimeDeadlineParamProbeState::reset();
  const ExecutionPlan compiled = test::make_runtime_deadline_param_test_plan();
  ASSERT_TRUE(compiled.ok()) << compiled.json();
  const ExecutionPlan plan = detail::RuntimeTestPlanAccess::enable_direct_once(compiled);

  test::RuntimeDeadlineParamProbeState::replay_violation.store(OnViolation::kSafeDefault,
                                                               std::memory_order_relaxed);
  RunSession session = run_async(plan);
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kFailed);
  ASSERT_EQ(session.diagnostics().size(), 1U);
  EXPECT_EQ(session.diagnostics().front().token, "RUNTIME_OPERATOR_REPLAY_FINGERPRINT_MISMATCH");
  EXPECT_TRUE(test::RuntimeProbeState::snapshot().empty());
  EXPECT_FALSE(test::RuntimeProbeState::compute_entered.load(std::memory_order_acquire));
}

// REQ: RUNTIME-004
// REQ: RUNTIME-016
TEST(RunSessionTest, ReconstructsPhaseSeparatedInputPredicateWithoutFakeReadinessSource) {
  const ExecutionPlan plan = test::make_runtime_input_predicate_replay_plan();
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.predicate_triggers().size(), 1U);
  ASSERT_EQ(plan.temporal_triggers().size(), 2U);
  const auto predicate_trigger = std::find_if(
      plan.temporal_triggers().begin(), plan.temporal_triggers().end(), [](const auto& trigger) {
        return trigger.kind == TemporalRecordTriggerKind::kPredicate;
      });
  ASSERT_NE(predicate_trigger, plan.temporal_triggers().end());
  ASSERT_TRUE(static_cast<bool>(predicate_trigger->readiness_source));
  EXPECT_TRUE(std::none_of(plan.readiness().begin(), plan.readiness().end(), [&](const auto& row) {
    return row.source_id == predicate_trigger->readiness_source;
  }));

  auto reconstructed = detail::GraphExec::reconstruct(detail::ExecutionPlanAccess::retain(plan),
                                                      RunGenerationId{17U});
  ASSERT_TRUE(reconstructed) << reconstructed.error().message;
  EXPECT_EQ((*reconstructed)->operators().size(), 2U);
}

// REQ: RUNTIME-004
// REQ: RUNTIME-015
TEST(RunSessionTest, ReconstructsEveryInputBackedTriggerWithoutFakeReadinessRecords) {
  constexpr std::array trigger_kinds{
      test::RuntimeInputTriggerReplayKind::kAny,
      test::RuntimeInputTriggerReplayKind::kEach,
      test::RuntimeInputTriggerReplayKind::kAligned,
      test::RuntimeInputTriggerReplayKind::kWindow,
  };
  for (const auto kind : trigger_kinds) {
    SCOPED_TRACE(static_cast<unsigned>(kind));
    const ExecutionPlan plan = test::make_runtime_input_trigger_replay_plan(kind);
    ASSERT_TRUE(plan.ok()) << plan.json();
    ASSERT_EQ(plan.temporal_triggers().size(), 2U);
    const auto trigger = std::find_if(
        plan.temporal_triggers().begin(), plan.temporal_triggers().end(), [](const auto& row) {
          return row.kind != TemporalRecordTriggerKind::kClock;
        });
    ASSERT_NE(trigger, plan.temporal_triggers().end());
    ASSERT_TRUE(static_cast<bool>(trigger->readiness_source));
    EXPECT_TRUE(
        std::none_of(plan.readiness().begin(), plan.readiness().end(), [&](const auto& row) {
          return row.source_id == trigger->readiness_source;
        }));

    auto reconstructed = detail::GraphExec::reconstruct(detail::ExecutionPlanAccess::retain(plan),
                                                        RunGenerationId{18U});
    ASSERT_TRUE(reconstructed) << reconstructed.error().message;
    EXPECT_EQ((*reconstructed)->operators().size(), 2U);
  }
}

// Clock scheduling is independent of input arrival, but its callback still consumes the complete
// frozen selector, missing-policy, and deadline plan. Replay must preserve both dimensions.
// REQ: RUNTIME-004
// REQ: RUNTIME-015
// REQ: RUNTIME-016
TEST(RunSessionTest, ReconstructsClockDrivenOperatorsWithConnectedInputs) {
  constexpr std::array kinds{
      test::RuntimeClockInputReplayKind::kClock,
      test::RuntimeClockInputReplayKind::kPredicate,
  };
  constexpr std::array profiles{Profile::kStandard, Profile::kResearch};
  for (const auto profile : profiles) {
    for (const auto kind : kinds) {
      SCOPED_TRACE(static_cast<unsigned>(profile));
      SCOPED_TRACE(static_cast<unsigned>(kind));
      const ExecutionPlan plan = test::make_runtime_clock_input_replay_plan(kind, profile);
      ASSERT_TRUE(plan.ok()) << plan.json();
      ASSERT_EQ(plan.input_selections().size(), 2U);
      ASSERT_EQ(plan.max_start_delays().size(), 1U);
      ASSERT_EQ(plan.clock_triggers().size(), 2U);

      auto reconstructed = detail::GraphExec::reconstruct(detail::ExecutionPlanAccess::retain(plan),
                                                          RunGenerationId{19U});
      ASSERT_TRUE(reconstructed) << reconstructed.error().message;
      EXPECT_EQ((*reconstructed)->operators().size(), 2U);
    }
  }
}

// REQ: RUNTIME-002
// REQ: RUNTIME-004
// REQ: RUNTIME-016
// REQ: RUNTIME-027
TEST(RunSessionTest, ClockReplayMismatchCarriesStableOperatorAndFingerprintContext) {
  struct ReplayModeReset {
    ~ReplayModeReset() {
      test::RuntimeClockInputReplayOperator::mismatch_ordinal.store(0U, std::memory_order_relaxed);
    }
  };
  const ReplayModeReset reset{};

  const ExecutionPlan baseline = test::make_runtime_dual_clock_input_replay_plan();
  ASSERT_TRUE(baseline.ok()) << baseline.json();

  const auto reconstruct_mismatch = [&](std::uint32_t ordinal) {
    test::RuntimeClockInputReplayOperator::mismatch_ordinal.store(ordinal,
                                                                  std::memory_order_relaxed);
    detail::RuntimeReplayMismatchDetail mismatch;
    auto reconstructed = detail::GraphExec::reconstruct(
        detail::ExecutionPlanAccess::retain(baseline), RunGenerationId{19U}, {}, &mismatch);
    EXPECT_FALSE(reconstructed);
    if (!reconstructed) {
      EXPECT_EQ(reconstructed.error().message, "RUNTIME_OPERATOR_REPLAY_MISMATCH");
    }
    return mismatch;
  };

  const detail::RuntimeReplayMismatchDetail ordinary = reconstruct_mismatch(1U);
  const detail::RuntimeReplayMismatchDetail predicate = reconstruct_mismatch(2U);
  ASSERT_TRUE(ordinary.present);
  ASSERT_TRUE(predicate.present);
  EXPECT_EQ(ordinary.category, detail::RuntimeReplayMismatchCategory::kTemporalContract);
  EXPECT_EQ(predicate.category, detail::RuntimeReplayMismatchCategory::kTemporalContract);
  EXPECT_EQ(ordinary.factory_slot, 1U);
  EXPECT_EQ(predicate.factory_slot, 2U);
  EXPECT_EQ(ordinary.operator_id, 1U);
  EXPECT_EQ(predicate.operator_id, 2U);
  EXPECT_NE(ordinary.vertex, predicate.vertex);
  EXPECT_NE(ordinary.expected, predicate.expected);
  EXPECT_NE(ordinary.observed, predicate.observed);

  const auto sink = std::make_shared<RecordingDiagnosticSink>();
  const ExecutionPlan plan = detail::RuntimeTestPlanAccess::with_diagnostic_sink(baseline, sink);
  test::RuntimeClockInputReplayOperator::mismatch_ordinal.store(1U, std::memory_order_relaxed);
  RunSession session = run_async(plan);
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kFailed);
  ASSERT_EQ(session.diagnostics().size(), 1U);
  const Diagnostic& diagnostic = session.diagnostics().front();
  EXPECT_EQ(diagnostic.token, "RUNTIME_OPERATOR_REPLAY_FINGERPRINT_MISMATCH");
  EXPECT_EQ(diagnostic.source.vertex, ordinary.vertex);
  EXPECT_EQ(diagnostic.required.size(), 71U);
  EXPECT_EQ(diagnostic.provided.size(), 71U);
  EXPECT_TRUE(diagnostic.required.starts_with("sha256:"));
  EXPECT_TRUE(diagnostic.provided.starts_with("sha256:"));
  EXPECT_NE(diagnostic.required, diagnostic.provided);

  const auto published = sink->snapshot_details();
  ASSERT_EQ(published.size(), 1U);
  ASSERT_EQ(published.front().size(), 1U);
  const detail::RuntimeDiagnosticDetail& retained = published.front().front();
  EXPECT_TRUE(retained.present);
  EXPECT_EQ(retained.error.code, ErrorCode::kFailure);
  EXPECT_EQ(retained.operator_id, ordinary.operator_id);
  EXPECT_EQ(retained.replay_mismatch.vertex, ordinary.vertex);
  EXPECT_EQ(retained.replay_mismatch.operator_id, ordinary.operator_id);
  EXPECT_EQ(retained.replay_mismatch.factory_slot, ordinary.factory_slot);
  EXPECT_EQ(retained.replay_mismatch.category, ordinary.category);
  EXPECT_EQ(retained.replay_mismatch.expected, ordinary.expected);
  EXPECT_EQ(retained.replay_mismatch.observed, ordinary.observed);
}

// REQ: RUNTIME-002
// REQ: RUNTIME-004
TEST(RunSessionTest, RejectsUninjectedReadinessBeforeLifecycleCallbacks) {
  test::RuntimeProbeState::reset();
  const ExecutionPlan plan = test::make_runtime_test_plan();
  test::RuntimeProbeState::reset();

  RunSession session = run_async(plan);
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kFailed);
  EXPECT_TRUE(test::has_runtime_diagnostic(session, "RUNTIME_READINESS_BINDING_UNAVAILABLE"));
  EXPECT_TRUE(test::RuntimeProbeState::snapshot().empty());
  EXPECT_FALSE(test::RuntimeProbeState::compute_entered.load(std::memory_order_acquire));
  EXPECT_EQ(test::RuntimeProbeState::active.load(std::memory_order_acquire), 0U);
}

// REQ: RUNTIME-010
TEST(RunSessionTest, ReportsMaterializationFailureBeforeDeadlineCallback) {
  test::RuntimeProbeState::reset();
  const ExecutionPlan baseline = test::enable_direct_runtime(test::make_runtime_test_plan());
  const ExecutionPlan plan = detail::RuntimeTestPlanAccess::mutate_materialization_strategy(
      baseline, "holoscan.runtime.unknown/v1");
  test::RuntimeProbeState::reset();

  RunSession session = run_async(plan);
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kFailed);
  EXPECT_TRUE(test::has_runtime_diagnostic(session, "PROFILE_MATERIALIZATION_FAILED"));
  EXPECT_TRUE(test::RuntimeProbeState::snapshot().empty());
  EXPECT_FALSE(test::RuntimeProbeState::compute_entered.load(std::memory_order_acquire));
}

// REQ: RUNTIME-010
TEST(RunSessionTest, RejectsEveryPackageMismatchBeforeOperatorReconstruction) {
  using ProfileMutation = detail::RuntimeTestPlanAccess::ProfileMutation;
  struct Case {
    ProfileMutation mutation;
    std::string_view token;
  };
  constexpr std::array cases{
      Case{ProfileMutation::kPlanSchema, "PROFILE_PLAN_SCHEMA_UNSUPPORTED"},
      Case{ProfileMutation::kProfilePolicy, "PROFILE_POLICY_UNSUPPORTED"},
      Case{ProfileMutation::kRuntimeAbi, "PROFILE_RUNTIME_ABI_INCOMPATIBLE"},
      Case{ProfileMutation::kTarget, "PROFILE_TARGET_INCOMPATIBLE"},
      Case{ProfileMutation::kCapability, "PROFILE_CAPABILITY_UNSATISFIED"},
      Case{ProfileMutation::kEvidence, "PROFILE_EVIDENCE_UNSATISFIED"},
  };

  for (const auto& test_case : cases) {
    SCOPED_TRACE(test_case.token);
    test::RuntimeProbeState::reset();
    const ExecutionPlan baseline = test::enable_direct_runtime(test::make_runtime_test_plan());
    const ExecutionPlan plan =
        detail::RuntimeTestPlanAccess::mutate_profile(baseline, test_case.mutation);
    ASSERT_TRUE(plan.ok());
    test::RuntimeProbeState::reset();

    RunSession session = run_async(plan);
    session.wait();

    EXPECT_EQ(session.termination(), RunTermination::kFailed);
    EXPECT_TRUE(test::has_runtime_diagnostic(session, test_case.token));
    EXPECT_EQ(test::RuntimeProbeState::constructed.load(std::memory_order_acquire), 0U);
    EXPECT_FALSE(test::RuntimeProbeState::setup_entered.load(std::memory_order_acquire));
    EXPECT_TRUE(test::RuntimeProbeState::snapshot().empty());
  }
}

// REQ: RUNTIME-002
// REQ: RUNTIME-004
// REQ: RUNTIME-008
TEST(RunSessionTest, RejectsMalformedLifecyclePlanBeforeCallbacks) {
  using LifecycleMutation = detail::RuntimeTestPlanAccess::LifecycleMutation;
  constexpr std::array mutations{
      LifecycleMutation::kCheckpointMissing,
      LifecycleMutation::kCheckpointRequired,
      LifecycleMutation::kCheckpointLevel,
      LifecycleMutation::kCheckpointOrder,
      LifecycleMutation::kDependency,
      LifecycleMutation::kRestartPermission,
      LifecycleMutation::kMutableField,
      LifecycleMutation::kTransactionCapacity,
      LifecycleMutation::kRestartTargetMissing,
      LifecycleMutation::kRestartTargetIdentity,
      LifecycleMutation::kRestartTargetAuthority,
      LifecycleMutation::kRestartClosureMissing,
  };

  for (const LifecycleMutation mutation : mutations) {
    SCOPED_TRACE(static_cast<unsigned>(mutation));
    test::RuntimeProbeState::reset();
    const ExecutionPlan baseline = test::enable_direct_runtime(test::make_runtime_test_plan());
    const ExecutionPlan plan = detail::RuntimeTestPlanAccess::mutate_lifecycle(baseline, mutation);
    test::RuntimeProbeState::reset();

    RunSession session = run_async(plan);
    session.wait();

    EXPECT_EQ(session.termination(), RunTermination::kFailed);
    EXPECT_TRUE(test::has_runtime_diagnostic(session, "RUNTIME_LIFECYCLE_PLAN_INVALID"));
    EXPECT_TRUE(test::RuntimeProbeState::snapshot().empty());
    EXPECT_FALSE(test::RuntimeProbeState::compute_entered.load(std::memory_order_acquire));
  }
}

}  // namespace
}  // namespace holoscan
