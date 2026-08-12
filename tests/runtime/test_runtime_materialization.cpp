/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "compiler/execution_plan_state.hpp"
#include "runtime/execution_lane.hpp"
#include "runtime/graph_exec.hpp"
#include "runtime/lifecycle_executor.hpp"
#include "runtime/runtime_materialization.hpp"
#include "runtime/runtime_test_access.hpp"
#include "runtime_test_graph.hpp"

namespace holoscan::detail {

// White-box accessor that exposes RuntimeLifecycleExecutor::materialization_ for test
// inspection without requiring a public API. Used in LifecycleGraphFails... to inspect
// per-obligation records after a failed startup.
class RuntimeLifecycleExecutorTestAccess {
 public:
  [[nodiscard]] static const RuntimeMaterializationState* materialization(
      const RuntimeLifecycleExecutor& executor) noexcept {
    return executor.materialization_ ? &*executor.materialization_ : nullptr;
  }
};

namespace {

// Minimal operator that records start/stop invocations via static atomics. Used in the lifecycle
// graph test to verify that operator
// callbacks are not called when materialization fails before reaching kStart.
class LifecycleMaterializationProbeOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override { spec.spin_source(spin_, "activation").capacity(1U); }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnSpin{.event = spin_});
    return result;
  }

  void start() override { start_count.fetch_add(1U, std::memory_order_relaxed); }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override { return {}; }

  void stop() override { stop_count.fetch_add(1U, std::memory_order_relaxed); }

  static inline std::atomic<std::uint32_t> start_count{};
  static inline std::atomic<std::uint32_t> stop_count{};

 private:
  SpinSource spin_;
};

// Compiles a three-operator plan with distinct execution partitions, all bound to BusyPoll.
// Operators a, b, c are independent spin-sources with no flow edges:
//   a   b   c   (each a spin-source; no downstream consumer)
// Used in LifecycleGraphFailsInCanonicalOrder... to supply a realistic ExecutionPlan.
[[nodiscard]] ExecutionPlan three_lane_materialization_plan() {
  Graph graph{"runtime-materialization-order"};
  const auto first = graph.op<LifecycleMaterializationProbeOperator>("a");
  const auto second = graph.op<LifecycleMaterializationProbeOperator>("b");
  const auto third = graph.op<LifecycleMaterializationProbeOperator>("c");
  graph.partition("materialization-partition-a").add(first);
  graph.partition("materialization-partition-b").add(second);
  graph.partition("materialization-partition-c").add(third);
  const ClockKey default_clock = graph.add_clock<test::RuntimeProbeClock>("runtime-test-clock");
  graph.set_default_clock(default_clock);
  CompileOptions options;
  options.deployment
      .bind_thread_placement("materialization-partition-a",
                             ThreadPlacement{.wake_mode = WakeMode::kBusyPoll})
      .bind_thread_placement("materialization-partition-b",
                             ThreadPlacement{.wake_mode = WakeMode::kBusyPoll})
      .bind_thread_placement("materialization-partition-c",
                             ThreadPlacement{.wake_mode = WakeMode::kBusyPoll});
  return compile(graph, std::move(options));
}

// Re-instantiates each operator from its factory and assigns it to the correct partition.
// Mirrors what the runtime does before startup; used in the lifecycle graph test to
// build a realistic RuntimeOperatorInstance list from an ExecutionPlanState.
[[nodiscard]] expected<std::vector<RuntimeOperatorInstance>, Error> reconstruct_operators(
    const ExecutionPlanState& plan) {
  std::vector<RuntimeOperatorInstance> operators;
  operators.reserve(plan.executable_factories.operators.size());
  for (const auto& record : plan.executable_factories.operators) {
    if (record.factory == nullptr) {
      return make_unexpected(Error{ErrorCode::kFailure, "RUNTIME_OPERATOR_FACTORY_INVALID"});
    }
    auto created = record.factory->create();
    if (!created || created.value() == nullptr) {
      return make_unexpected(Error{ErrorCode::kFailure, "RUNTIME_OPERATOR_RECONSTRUCTION_FAILED"});
    }
    auto replay = std::make_unique<OperatorSpec>();
    created.value()->setup(*replay);
    const auto partition = std::find_if(
        plan.execution_partitions.begin(),
        plan.execution_partitions.end(),
        [&](const auto& candidate) {
          return std::find(candidate.members.begin(), candidate.members.end(), record.vertex) !=
                 candidate.members.end();
        });
    if (partition == plan.execution_partitions.end()) {
      return make_unexpected(Error{ErrorCode::kFailure, "RUNTIME_OPERATOR_INVENTORY_MISMATCH"});
    }
    operators.push_back(RuntimeOperatorInstance{
        .vertex = record.vertex,
        .operator_id = record.operator_id,
        .partition = partition->id,
        .execution_lane = partition->execution_lane,
        .op = std::move(created).value(),
        .replay_spec = std::move(replay),
        .tags = std::make_unique<TagAllocator>(),
    });
  }
  return operators;
}

// Returns a SchedulerPlanRecord for deterministic serial execution across `lane_count`
// lanes, defaulting to 1. Used to build ExecutionPlanState fixtures in this file.
[[nodiscard]] SchedulerPlanRecord serial_scheduler(std::uint32_t lane_count = 1U) {
  return SchedulerPlanRecord{
      .ready_policy = AlgorithmKey{"holoscan.ready.deterministic_serial", 1U},
      .execution_lane_count = lane_count,
      .max_concurrent_activations = 1U,
      .activation_overlap = false,
      .worker_count = lane_count,
      .intra_path_parallelism = 1U,
  };
}

// Returns an ExecutionLanePlanRecord for the given lane index and partition with no explicit
// OS placement. Used alongside default_lane() in multi-lane materialization tests.
[[nodiscard]] ExecutionLanePlanRecord explicit_lane(std::uint32_t lane,
                                                    ExecutionPartitionId partition) {
  return ExecutionLanePlanRecord{
      .execution_lane = lane,
      .partition = partition,
      .accepted_placement = ThreadPlacement{},
  };
}

// Returns the first partition lane with no explicit OS placement.
[[nodiscard]] ExecutionLanePlanRecord default_lane() {
  return ExecutionLanePlanRecord{
      .execution_lane = 0U,
      .partition = ExecutionPartitionId{1U},
      .accepted_placement = ThreadPlacement{},
  };
}

// Returns a RuntimePackageCapabilityMatchReport representing a successful capability
// match. Used as the reference package across all RuntimeMaterializationState tests.
[[nodiscard]] RuntimePackageCapabilityMatchReport matching_package() {
  return RuntimePackageCapabilityMatchReport{
      .context = {.startup_attempt = 3U, .plan_generation = 5U, .run_generation = 7U},
      .package_id = "holoscan-sdk",
      .package_build_id = "test-build",
      .package_manifest_digest =
          "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
      .result = RuntimePackageCapabilityMatchResult::kMatched,
  };
}

// Returns a ProfileMaterializationObligationRecord for the execution_lane strategy on
// object "lane:0". The most common obligation fixture in this file.
[[nodiscard]] ProfileMaterializationObligationRecord lane_obligation() {
  return ProfileMaterializationObligationRecord{
      .object_id = "lane:0",
      .strategy_id = "holoscan.runtime.execution_lane/v1",
      .complete_before_stage = LifecycleStage::kStart,
      .cleanup_stage_mask = 0xC0U,
      .max_cleanup_actions = 1U,
      .evidence_id = "holoscan.compile.thread_placement.validation/v1",
  };
}

// Returns a ProfileMaterializationObligationRecord for the tensor_device_stream strategy,
// targeting the stream resource defined by tensor_device_resource(). Due at kArm.
[[nodiscard]] ProfileMaterializationObligationRecord tensor_device_stream_obligation() {
  return ProfileMaterializationObligationRecord{
      .object_id = "tensor-stream.5.process.1.device.0",
      .strategy_id = "holoscan.runtime.tensor_device_stream/v1",
      .complete_before_stage = LifecycleStage::kArm,
      .cleanup_stage_mask = 0xC0U,
      .max_cleanup_actions = 1U,
      .evidence_id = "holoscan.compile.tensor_device_resource_plan/v1",
  };
}

// Returns a TensorDeviceResourcePlanRecord with plan_generation=5 and a stream_id that
// matches tensor_device_stream_obligation(). Used in ValidatesTensorDeviceStreamObligationAtArm.
[[nodiscard]] TensorDeviceResourcePlanRecord tensor_device_resource() {
  return TensorDeviceResourcePlanRecord{
      .plan_generation = 5U,
      .run_generation = 0U,
      .process_group = ProcessGroupId{1U},
      .device = DeviceId{0},
      .allocator_root_id = "tensor-allocator.5.process.1.device.0",
      .stream_id = "tensor-stream.5.process.1.device.0",
      .pool_count = 1U,
      .slot_capacity = 2U,
      .loan_credit_capacity = 1U,
      .control_capacity = 2U,
      .readiness_capacity = 2U,
      .release_capacity = 2U,
      .retirement_capacity = 2U,
      .quarantine_capacity = 2U,
      .materialize_before = LifecycleStage::kArm,
      .per_partition_stream = false,
      .per_partition_event = false,
      .per_partition_green_context = false,
      .gpu_resident_backend_selected = false,
      .fallback_rejected = true,
  };
}

// Holds a pre-built ExecutionPlanState and a started ExecutionLaneSet together.
// Populated by started_lane() before returning to each test.
struct StartedLane {
  ExecutionPlanState plan;
  std::unique_ptr<ExecutionLaneSet> lanes;
};

// Creates a single-lane serial ExecutionPlanState, starts its ExecutionLaneSet,
// and returns the pair. Used as the base fixture for most materialization tests.
[[nodiscard]] StartedLane started_lane() {
  StartedLane result;
  result.plan.scheduler_plan = serial_scheduler();
  result.plan.execution_lanes = {default_lane()};
  auto lanes = ExecutionLaneSet::create(result.plan.scheduler_plan, result.plan.execution_lanes);
  EXPECT_TRUE(lanes) << runtime_error_message(lanes.error());
  if (!lanes) {
    return result;
  }
  result.lanes = std::move(lanes).value();
  EXPECT_TRUE(result.lanes->start());
  return result;
}

// REQ: RUNTIME-010
TEST(RuntimeMaterializationTest, CompletesAtDeadlineAndCleansExactlyOnce) {
  auto fixture = started_lane();
  ASSERT_NE(fixture.lanes, nullptr);
  fixture.plan.profile_manifest.materialization_obligations = {lane_obligation()};

  auto created =
      RuntimeMaterializationState::create(fixture.plan, matching_package(), fixture.lanes.get());
  ASSERT_TRUE(created) << created.error().message;
  auto state = std::move(created).value();
  ASSERT_EQ(state.size(), 1U);

  EXPECT_EQ(state.transition(0U, LifecycleStage::kStart, LifecycleTransitionPhase::kStartupForward),
            LifecycleStatus::kOk);
  EXPECT_EQ(state.record(0U).result, RuntimeMaterializationResult::kCompleted);
  EXPECT_EQ(state.record(0U).context.plan_generation, 5U);
  EXPECT_EQ(state.record(0U).package_id, "holoscan-sdk");

  EXPECT_EQ(
      state.transition(0U, LifecycleStage::kStop, LifecycleTransitionPhase::kShutdownTeardown),
      LifecycleStatus::kSkipped);
  EXPECT_EQ(
      state.transition(0U, LifecycleStage::kRelease, LifecycleTransitionPhase::kShutdownTeardown),
      LifecycleStatus::kOk);
  EXPECT_EQ(state.record(0U).result, RuntimeMaterializationResult::kCleaned);
  EXPECT_EQ(state.record(0U).cleanup_actions, 1U);
  EXPECT_EQ(
      state.transition(0U, LifecycleStage::kRelease, LifecycleTransitionPhase::kShutdownTeardown),
      LifecycleStatus::kSkipped);
  EXPECT_EQ(state.record(0U).cleanup_actions, 1U);
  EXPECT_TRUE(fixture.lanes->join());
}

// REQ: CHAN-103
TEST(RuntimeMaterializationTest, HostOnlyReconstructionKeepsCudaSelectionDisabled) {
  const ExecutionPlan plan =
      RuntimeTestPlanAccess::enable_direct_once(three_lane_materialization_plan());
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto accepted = ExecutionPlanAccess::retain(plan);
  ASSERT_TRUE(accepted);

  auto reconstructed = GraphExec::reconstruct(accepted, RunGenerationId{17U});
  ASSERT_TRUE(reconstructed) << reconstructed.error().message;
  ASSERT_EQ((*reconstructed)->operators().size(), 3U);
  for (const RuntimeOperatorInstance& instance : (*reconstructed)->operators()) {
    EXPECT_FALSE(instance.compute_device.valid());
    EXPECT_EQ(instance.compute_stream, nullptr);
  }
}

// REQ: RUNTIME-010
TEST(RuntimeMaterializationTest, CleansCompletedObligationAfterPartialStartup) {
  auto fixture = started_lane();
  ASSERT_NE(fixture.lanes, nullptr);
  fixture.plan.profile_manifest.materialization_obligations = {lane_obligation()};

  auto created =
      RuntimeMaterializationState::create(fixture.plan, matching_package(), fixture.lanes.get());
  ASSERT_TRUE(created) << created.error().message;
  auto state = std::move(created).value();

  EXPECT_EQ(state.transition(0U, LifecycleStage::kStart, LifecycleTransitionPhase::kStartupForward),
            LifecycleStatus::kOk);
  state.cleanup_remaining();
  EXPECT_EQ(state.record(0U).result, RuntimeMaterializationResult::kCleaned);
  EXPECT_EQ(state.record(0U).cleanup_actions, 1U);
  state.cleanup_remaining();
  EXPECT_EQ(state.record(0U).cleanup_actions, 1U);
  EXPECT_TRUE(fixture.lanes->join());
}

// REQ: COMPILE-048
// REQ: OPERATOR-025
TEST(RuntimeMaterializationTest, ValidatesTensorDeviceStreamObligationAtArm) {
  auto fixture = started_lane();
  ASSERT_NE(fixture.lanes, nullptr);
  // Key setup: plan_generation=5 links the tensor resource to the obligation via stream_id.
  fixture.plan.plan_generation = 5U;
  fixture.plan.tensor_device_resources = {tensor_device_resource()};
  fixture.plan.profile_manifest.materialization_obligations = {tensor_device_stream_obligation()};

  auto created =
      RuntimeMaterializationState::create(fixture.plan, matching_package(), fixture.lanes.get());
  ASSERT_TRUE(created) << created.error().message;
  auto state = std::move(created).value();
  EXPECT_EQ(state.transition(0U, LifecycleStage::kArm, LifecycleTransitionPhase::kStartupForward),
            LifecycleStatus::kOk);

  // Key assertion: stream_id mismatch between obligation and resource causes kFatalFailure.
  fixture.plan.tensor_device_resources.front().stream_id = "tensor-stream.mismatch";
  created =
      RuntimeMaterializationState::create(fixture.plan, matching_package(), fixture.lanes.get());
  ASSERT_TRUE(created) << created.error().message;
  state = std::move(created).value();
  EXPECT_EQ(state.transition(0U, LifecycleStage::kArm, LifecycleTransitionPhase::kStartupForward),
            LifecycleStatus::kFatalFailure);
  EXPECT_TRUE(state.failed());
  EXPECT_TRUE(fixture.lanes->join());
}

// REQ: RUNTIME-010
TEST(RuntimeMaterializationTest, CleansMultipleCompletedObligationsInReverseCanonicalOrder) {
  StartedLane fixture;
  fixture.plan.scheduler_plan = serial_scheduler(2U);
  fixture.plan.execution_lanes = {default_lane(), explicit_lane(1U, ExecutionPartitionId{2U})};
  auto lanes = ExecutionLaneSet::create(fixture.plan.scheduler_plan, fixture.plan.execution_lanes);
  ASSERT_TRUE(lanes) << runtime_error_message(lanes.error());
  fixture.lanes = std::move(lanes).value();
  ASSERT_TRUE(fixture.lanes->start());

  auto first = lane_obligation();
  auto second = lane_obligation();
  second.object_id = "lane:1";
  fixture.plan.profile_manifest.materialization_obligations = {first, second};
  auto created =
      RuntimeMaterializationState::create(fixture.plan, matching_package(), fixture.lanes.get());
  ASSERT_TRUE(created) << created.error().message;
  auto state = std::move(created).value();

  // Key setup: obligations for lane:0 and lane:1 both complete at kStart in order.
  ASSERT_EQ(state.transition(0U, LifecycleStage::kStart, LifecycleTransitionPhase::kStartupForward),
            LifecycleStatus::kOk);
  ASSERT_EQ(state.transition(1U, LifecycleStage::kStart, LifecycleTransitionPhase::kStartupForward),
            LifecycleStatus::kOk);
  state.cleanup_remaining();

  // Key assertion: cleanup proceeds in reverse canonical order — lane:1 before lane:0.
  ASSERT_EQ(state.cleanup_order().size(), 2U);
  EXPECT_EQ(state.cleanup_order()[0], 1U);
  EXPECT_EQ(state.cleanup_order()[1], 0U);
  EXPECT_EQ(state.record(0U).result, RuntimeMaterializationResult::kCleaned);
  EXPECT_EQ(state.record(1U).result, RuntimeMaterializationResult::kCleaned);
  EXPECT_TRUE(fixture.lanes->join());
}

// REQ: RUNTIME-010
TEST(RuntimeMaterializationTest,
     LifecycleGraphFailsInCanonicalOrderAndCleansCompletedBarriersInReverseExactlyOnce) {
  LifecycleMaterializationProbeOperator::start_count.store(0U, std::memory_order_relaxed);
  LifecycleMaterializationProbeOperator::stop_count.store(0U, std::memory_order_relaxed);
  const ExecutionPlan baseline = three_lane_materialization_plan();
  ASSERT_TRUE(baseline.ok());
  const auto accepted = ExecutionPlanAccess::retain(baseline);
  ASSERT_NE(accepted, nullptr);
  ASSERT_EQ(accepted->scheduler_plan.execution_lane_count, 3U);

  auto plan = std::make_shared<ExecutionPlanState>(*accepted);
  // Key setup: obligations 0 and 1 use the valid lane strategy; obligation 2 uses an
  //   invalid strategy (sorted before the valid lane:2 entry); obligation 3 is unreachable.
  auto lane_zero = lane_obligation();
  auto lane_one = lane_obligation();
  lane_one.object_id = "lane:1";
  auto failing = lane_obligation();
  failing.object_id = "lane:2";
  failing.strategy_id = "aaa.invalid-before-valid-lane-strategy";
  auto never_reached = lane_obligation();
  never_reached.object_id = "lane:2";
  plan->profile_manifest.materialization_obligations = {
      lane_zero,
      lane_one,
      failing,
      never_reached,
  };

  auto reconstructed = reconstruct_operators(*plan);
  ASSERT_TRUE(reconstructed) << reconstructed.error().message;
  auto operators = std::move(reconstructed).value();
  auto lanes = ExecutionLaneSet::create(plan->scheduler_plan, plan->execution_lanes);
  ASSERT_TRUE(lanes) << runtime_error_message(lanes.error());
  auto lane_set = std::move(lanes).value();
  const auto lanes_started = lane_set->start();
  ASSERT_TRUE(lanes_started);

  const std::atomic_bool cancellation{};
  constexpr RunGenerationId kRunGeneration = 17U;
  auto package = matching_package();
  package.context.plan_generation = plan->plan_generation;
  package.context.run_generation = kRunGeneration;
  auto created = RuntimeLifecycleExecutor::create(
      *plan, operators, kRunGeneration, &cancellation, nullptr, lane_set.get(), package);
  ASSERT_TRUE(created) << created.error().message;
  auto lifecycle = std::move(created).value();

  EXPECT_FALSE(lifecycle->startup());
  EXPECT_EQ(lifecycle->failure_token(), "PROFILE_MATERIALIZATION_FAILED");
  const RuntimeMaterializationState* const materialization =
      RuntimeLifecycleExecutorTestAccess::materialization(*lifecycle);
  ASSERT_NE(materialization, nullptr);
  ASSERT_EQ(materialization->size(), 4U);
  // Key assertion: failure at index 2 stops forward progress; 0 and 1 are cleaned in
  //   reverse order; operator start/stop never fire since kStart was not reached.
  EXPECT_EQ(materialization->record(0U).result, RuntimeMaterializationResult::kCleaned);
  EXPECT_EQ(materialization->record(1U).result, RuntimeMaterializationResult::kCleaned);
  EXPECT_EQ(materialization->record(2U).result, RuntimeMaterializationResult::kFailed);
  EXPECT_EQ(materialization->record(3U).result, RuntimeMaterializationResult::kPending);
  EXPECT_EQ(materialization->record(0U).cleanup_actions, 1U);
  EXPECT_EQ(materialization->record(1U).cleanup_actions, 1U);
  ASSERT_EQ(materialization->cleanup_order().size(), 2U);
  EXPECT_EQ(materialization->cleanup_order()[0], 1U);
  EXPECT_EQ(materialization->cleanup_order()[1], 0U);
  EXPECT_EQ(LifecycleMaterializationProbeOperator::start_count.load(std::memory_order_relaxed), 0U);
  EXPECT_EQ(LifecycleMaterializationProbeOperator::stop_count.load(std::memory_order_relaxed), 0U);

  (void)lifecycle->shutdown();
  EXPECT_EQ(materialization->record(0U).cleanup_actions, 1U);
  EXPECT_EQ(materialization->record(1U).cleanup_actions, 1U);
  EXPECT_EQ(materialization->cleanup_order().size(), 2U);
  EXPECT_EQ(LifecycleMaterializationProbeOperator::start_count.load(std::memory_order_relaxed), 0U);
  EXPECT_EQ(LifecycleMaterializationProbeOperator::stop_count.load(std::memory_order_relaxed), 0U);
  const auto lanes_joined = lane_set->join();
  EXPECT_TRUE(lanes_joined) << runtime_error_message(lanes_joined.error());
}

// REQ: RUNTIME-010
TEST(RuntimeMaterializationTest, FailsUnknownStrategyWithoutWideningThePlan) {
  auto fixture = started_lane();
  ASSERT_NE(fixture.lanes, nullptr);
  auto obligation = lane_obligation();
  obligation.strategy_id = "holoscan.runtime.unknown/v1";
  fixture.plan.profile_manifest.materialization_obligations = {obligation};

  auto created =
      RuntimeMaterializationState::create(fixture.plan, matching_package(), fixture.lanes.get());
  ASSERT_TRUE(created) << created.error().message;
  auto state = std::move(created).value();
  EXPECT_EQ(state.transition(0U, LifecycleStage::kStart, LifecycleTransitionPhase::kStartupForward),
            LifecycleStatus::kFatalFailure);
  EXPECT_TRUE(state.failed());
  EXPECT_EQ(state.record(0U).result, RuntimeMaterializationResult::kFailed);
  EXPECT_EQ(fixture.plan.profile_manifest.materialization_obligations.front().strategy_id,
            "holoscan.runtime.unknown/v1");
  EXPECT_TRUE(fixture.lanes->join());
}

// REQ: RUNTIME-010
TEST(RuntimeMaterializationTest, RejectsNoncanonicalOrUnboundedObligationsBeforeCallbacks) {
  auto fixture = started_lane();
  ASSERT_NE(fixture.lanes, nullptr);
  const auto package = matching_package();

  // Key setup: four invalid obligation configurations exercised in sequence.
  // Key assertion: all four produce PROFILE_MATERIALIZATION_FAILED before any callback.
  auto incompatible_package = package;
  incompatible_package.result = RuntimePackageCapabilityMatchResult::kCapabilityUnsatisfied;
  fixture.plan.profile_manifest.materialization_obligations = {lane_obligation()};
  auto created =
      RuntimeMaterializationState::create(fixture.plan, incompatible_package, fixture.lanes.get());
  ASSERT_FALSE(created);
  EXPECT_EQ(created.error().message, "PROFILE_MATERIALIZATION_FAILED");

  auto first = lane_obligation();
  first.object_id = "z";
  auto second = lane_obligation();
  second.object_id = "a";
  fixture.plan.profile_manifest.materialization_obligations = {first, second};
  created = RuntimeMaterializationState::create(fixture.plan, package, fixture.lanes.get());
  ASSERT_FALSE(created);
  EXPECT_EQ(created.error().message, "PROFILE_MATERIALIZATION_FAILED");

  auto invalid = lane_obligation();
  invalid.cleanup_stage_mask = 0U;
  fixture.plan.profile_manifest.materialization_obligations = {invalid};
  created = RuntimeMaterializationState::create(fixture.plan, package, fixture.lanes.get());
  ASSERT_FALSE(created);
  EXPECT_EQ(created.error().message, "PROFILE_MATERIALIZATION_FAILED");

  invalid = lane_obligation();
  invalid.object_id.assign(kRuntimeMaterializationMaximumTokenBytes + 1U, 'x');
  fixture.plan.profile_manifest.materialization_obligations = {invalid};
  created = RuntimeMaterializationState::create(fixture.plan, package, fixture.lanes.get());
  ASSERT_FALSE(created);
  EXPECT_EQ(created.error().message, "PROFILE_MATERIALIZATION_FAILED");
  EXPECT_TRUE(fixture.lanes->join());
}

}  // namespace
}  // namespace holoscan::detail
