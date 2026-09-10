/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "compile_test_graph.hpp"
#include "compiler/lifecycle_plan_lowering.hpp"
#include "core/graph_compiler_access.hpp"
#include "core/lifecycle_kernel.hpp"
#include "core/lifecycle_transaction_kernel.hpp"

// GTest fatal assertions establish optional preconditions that clang-tidy does not model.
// NOLINTBEGIN(bugprone-unchecked-optional-access)
namespace holoscan {
namespace {

// Returns LifecycleTransactionOptions with generous fixed capacities so that
// tests can focus on plan logic rather than hitting capacity limits.
// Shared by most CompileLifecycle tests; contrast lifecycle_options() which
// re-compiles at the exact compiler-derived minimum.
[[nodiscard]] LifecycleTransactionOptions ample_lifecycle_capacity() {
  return LifecycleTransactionOptions{
      .max_pending_requests = 64U,
      .max_pending_transactions = 64U,
      .max_receipts = 128U,
      .max_fenced_operators = 64U,
      .max_candidate_generations = 64U,
      .max_events = 1024U,
      .max_audit_records = 256U,
      .max_quiesce_records = 256U,
      .max_rollback_records = 256U,
  };
}

// Converts the compiler-derived LifecycleTransactionCapacityValues minimum
// back into LifecycleTransactionOptions for re-compilation at the exact
// required minimum. Used by ExactMinimumAdmitsActiveAndSuccessorInTransactionKernel.
[[nodiscard]] LifecycleTransactionOptions lifecycle_options(
    const LifecycleTransactionCapacityValues& values) noexcept {
  return LifecycleTransactionOptions{
      .max_pending_requests = values.max_pending_requests,
      .max_pending_transactions = values.max_pending_transactions,
      .max_receipts = values.max_receipts,
      .max_fenced_operators = values.max_fenced_operators,
      .max_candidate_generations = values.max_candidate_generations,
      .max_events = values.max_events,
      .max_audit_records = values.max_audit_records,
      .max_quiesce_records = values.max_quiesce_records,
      .max_rollback_records = values.max_rollback_records,
  };
}

// Converts LifecycleTransactionCapacityValues into the internal
// LifecycleTransactionKernelCapacity type so tests can drive
// LifecycleTransactionKernel directly with compiler-derived bounds.
[[nodiscard]] detail::LifecycleTransactionKernelCapacity kernel_capacity(
    const LifecycleTransactionCapacityValues& values) noexcept {
  return detail::LifecycleTransactionKernelCapacity{
      .max_pending_requests = values.max_pending_requests,
      .max_pending_transactions = values.max_pending_transactions,
      .max_receipts = values.max_receipts,
      .max_fenced_operators = values.max_fenced_operators,
      .max_candidate_generations = values.max_candidate_generations,
      .max_events = values.max_events,
      .max_audit_records = values.max_audit_records,
      .max_quiesce_records = values.max_quiesce_records,
      .max_rollback_records = values.max_rollback_records,
  };
}

inline constexpr LifecycleDependencyKey kAllStageReady{"test.lifecycle.all-stage-ready"};

// Operator that registers callbacks for all eight lifecycle stages and can
// publish or consume a named dependency key at kArm. The `callbacks=false`
// flag creates a sparse operator (no callbacks registered). Used by
// FreezesAllEightStageDagsSparseCheckpointsAndBothEdgeKinds.
class AllStageLifecycleOperator final : public Operator<> {
 public:
  enum class Facts : std::uint8_t { kNone, kProducer, kConsumer };

  explicit AllStageLifecycleOperator(bool callbacks = true, Facts facts = Facts::kNone)
      : callbacks_(callbacks), facts_(facts) {}

  void setup(OperatorSpec& spec) override {
    if (callbacks_) {
      spec.lifecycle()
          .stage(LifecycleStage::kConfigure, &AllStageLifecycleOperator::checkpoint)
          .stage(LifecycleStage::kDiscover, &AllStageLifecycleOperator::checkpoint)
          .stage(LifecycleStage::kAllocate, &AllStageLifecycleOperator::checkpoint)
          .stage(LifecycleStage::kWarmUp, &AllStageLifecycleOperator::checkpoint)
          .stage(LifecycleStage::kArm, &AllStageLifecycleOperator::checkpoint)
          .stage(LifecycleStage::kStart, &AllStageLifecycleOperator::checkpoint)
          .stage(LifecycleStage::kStop, &AllStageLifecycleOperator::checkpoint)
          .stage(LifecycleStage::kRelease, &AllStageLifecycleOperator::checkpoint);
    }
    if (facts_ == Facts::kProducer) {
      spec.lifecycle().produces(LifecycleStage::kArm, kAllStageReady);
    } else if (facts_ == Facts::kConsumer) {
      spec.lifecycle().requires_dependency(LifecycleStage::kArm, kAllStageReady);
    }
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }
  LifecycleStatus checkpoint(LifecycleContext&) noexcept { return LifecycleStatus::kOk; }

 private:
  bool callbacks_;
  Facts facts_;
};

// Populates a two-operator graph with a data edge and lifecycle constraints:
// before(first, second, kStart), allow_self_restart(first),
// allow_external_restart(second), and allow_restart_request(first, second).
// Shared setup for capacity, policy, and permission lowering tests.
void author_lifecycle_graph(Graph& graph) {
  test::add_compile_test_default_clock(graph);
  auto first = graph.op<test::CompileTestOperator>("first");
  auto second = graph.op<test::CompileTestOperator>("second");
  graph.add_flow(first->output, second->input);
  graph.lifecycle()
      .before(first, second, LifecycleStage::kStart)
      .allow_self_restart(first)
      .allow_external_restart(second)
      .allow_restart_request(first, second);
}

// Returns true if any diagnostic in the lowering result carries the given
// token string. Avoids full-structure inspection in snapshot-mutation tests.
// Used by DistinguishesEveryDefensiveStageGraphRejection.
[[nodiscard]] bool has_lifecycle_diagnostic(const detail::LifecyclePlanLowering& lowering,
                                            std::string_view token) {
  return std::any_of(lowering.diagnostics.begin(),
                     lowering.diagnostics.end(),
                     [token](const auto& diagnostic) { return diagnostic.token == token; });
}

// Wraps lower_lifecycle_plan with standard-profile defaults and a fixed plan
// generation (id=1), reducing boilerplate in snapshot-mutation tests.
// Used by DistinguishesEveryDefensiveStageGraphRejection and
// RejectsInvalidRestartStageMaskWithBoundedPayload.
[[nodiscard]] detail::LifecyclePlanLowering lower_lifecycle_snapshot(
    const detail::GraphSnapshotData& snapshot,
    std::span<const LifecycleMutableFieldPlanRecord> mutable_fields = {}) {
  const std::vector<std::optional<Profile>> profiles(snapshot.vertices.size(), Profile::kStandard);
  return detail::lower_lifecycle_plan(
      snapshot, profiles, mutable_fields, std::nullopt, std::nullopt, PlanGenerationId{1U});
}

// Eight direct before-edges between first and sparse; one kDependencyKey fact
// at kArm between sparse and consumer; restart_request from first to consumer.
//
//   first --[before, all 8 stages]---> sparse(producer, no callbacks)
//   sparse --[kAllStageReady, kArm, kDependencyKey]--> consumer
//   first --[allow_restart_request]---> consumer
//
// REQ: COMPILE-008
TEST(CompileLifecycle, FreezesAllEightStageDagsSparseCheckpointsAndBothEdgeKinds) {
  Graph graph{"lifecycle-all-stages"};
  auto first =
      graph.op<AllStageLifecycleOperator>("first", true, AllStageLifecycleOperator::Facts::kNone);
  auto sparse = graph.op<AllStageLifecycleOperator>(
      "sparse", false, AllStageLifecycleOperator::Facts::kProducer);
  auto consumer = graph.op<AllStageLifecycleOperator>(
      "consumer", true, AllStageLifecycleOperator::Facts::kConsumer);

  constexpr std::array stages{
      LifecycleStage::kConfigure,
      LifecycleStage::kDiscover,
      LifecycleStage::kAllocate,
      LifecycleStage::kWarmUp,
      LifecycleStage::kArm,
      LifecycleStage::kStart,
      LifecycleStage::kStop,
      LifecycleStage::kRelease,
  };
  for (const auto stage : stages) {
    graph.lifecycle().before(first, sparse, stage);
  }
  graph.lifecycle().allow_self_restart(first).allow_external_restart(sparse).allow_restart_request(
      first, consumer);
  graph.partition("first").add(first);
  graph.partition("sparse").add(sparse);
  graph.partition("consumer").add(consumer);

  // Key setup: sparse has callbacks_=false so checkpoints are recorded without
  //            a registered callback; consumer declares a dependency on kAllStageReady.
  CompileOptions options;
  options.lifecycle_transactions = ample_lifecycle_capacity();
  const ExecutionPlan plan = compile(graph, options);
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.lifecycle_checkpoints().size(), stages.size() * 3U);
  ASSERT_EQ(plan.lifecycle_dependencies().size(), stages.size() + 1U);

  for (const auto stage : stages) {
    const auto stage_checkpoints =
        std::count_if(plan.lifecycle_checkpoints().begin(),
                      plan.lifecycle_checkpoints().end(),
                      [stage](const auto& checkpoint) { return checkpoint.stage == stage; });
    EXPECT_EQ(stage_checkpoints, 3);
    const auto sparse_checkpoint =
        std::find_if(plan.lifecycle_checkpoints().begin(),
                     plan.lifecycle_checkpoints().end(),
                     [stage](const auto& checkpoint) {
                       return checkpoint.stage == stage && !checkpoint.callback_present;
                     });
    ASSERT_NE(sparse_checkpoint, plan.lifecycle_checkpoints().end());
    // Key assertion: the sparse operator's checkpoint has callback_present=false
    //               and required=false (no registered callback → not mandatory).
    EXPECT_FALSE(sparse_checkpoint->required);

    const auto direct = std::find_if(plan.lifecycle_dependencies().begin(),
                                     plan.lifecycle_dependencies().end(),
                                     [stage](const auto& dependency) {
                                       return dependency.stage == stage &&
                                              dependency.kind == LifecycleEdgeKind::kBefore;
                                     });
    ASSERT_NE(direct, plan.lifecycle_dependencies().end());
    EXPECT_TRUE(direct->dependency_key.empty());
    EXPECT_LT(direct->before.id, direct->after.id);
  }

  const auto keyed = std::find_if(
      plan.lifecycle_dependencies().begin(),
      plan.lifecycle_dependencies().end(),
      [](const auto& dependency) { return dependency.kind == LifecycleEdgeKind::kDependencyKey; });
  ASSERT_NE(keyed, plan.lifecycle_dependencies().end());
  EXPECT_EQ(keyed->stage, LifecycleStage::kArm);
  EXPECT_EQ(keyed->dependency_key, kAllStageReady.stable_name);
  EXPECT_TRUE(std::all_of(plan.lifecycle_checkpoints().begin(),
                          plan.lifecycle_checkpoints().end(),
                          [](const auto& checkpoint) {
                            return checkpoint.operator_instance.plan_bound() &&
                                   checkpoint.operator_instance.run_generation == 0U;
                          }));
}

// Canonical graph: producer(kProducer) --[before, kConfigure]--> consumer(kConsumer)
// Each mutation of the canonical snapshot is expected to trigger a distinct
// diagnostic token; exercises every defensive check inside lower_lifecycle_plan.
//
// REQ: COMPILE-008
TEST(CompileLifecycle, DistinguishesEveryDefensiveStageGraphRejection) {
  Graph graph{"lifecycle-stage-graph-diagnostics"};
  auto producer = graph.op<AllStageLifecycleOperator>(
      "producer", true, AllStageLifecycleOperator::Facts::kProducer);
  auto consumer = graph.op<AllStageLifecycleOperator>(
      "consumer", true, AllStageLifecycleOperator::Facts::kConsumer);
  graph.lifecycle().before(producer, consumer, LifecycleStage::kConfigure);
  // Key setup: each block below mutates exactly one invariant in a fresh copy
  //            of the canonical snapshot; the expected diagnostic token differs.
  const auto canonical = detail::GraphCompilerAccess::canonical_snapshot(graph);
  ASSERT_TRUE(canonical);

  auto missing = *canonical;
  missing.lifecycle_facts.erase(
      std::remove_if(
          missing.lifecycle_facts.begin(),
          missing.lifecycle_facts.end(),
          [](const auto& fact) { return fact.kind == detail::LifecycleFactKind::kProduces; }),
      missing.lifecycle_facts.end());
  EXPECT_TRUE(
      has_lifecycle_diagnostic(lower_lifecycle_snapshot(missing), "LIFECYCLE_DEPENDENCY_MISSING"));

  auto duplicate = *canonical;
  auto duplicated_producer = *std::find_if(
      duplicate.lifecycle_facts.begin(), duplicate.lifecycle_facts.end(), [](const auto& fact) {
        return fact.kind == detail::LifecycleFactKind::kProduces;
      });
  const auto consumer_vertex =
      std::find_if(duplicate.vertices.begin(), duplicate.vertices.end(), [](const auto& vertex) {
        return vertex.kind == VertexKind::kOperator && vertex.name == "consumer";
      });
  ASSERT_NE(consumer_vertex, duplicate.vertices.end());
  duplicated_producer.owner = consumer_vertex->id;
  duplicate.lifecycle_facts.push_back(std::move(duplicated_producer));
  EXPECT_TRUE(has_lifecycle_diagnostic(lower_lifecycle_snapshot(duplicate),
                                       "LIFECYCLE_DEPENDENCY_DUPLICATE"));

  // Canonical Graph facts carry one stage, so they cannot represent a
  // disagreement between the private dependency declaration and key stages.
  // Verify the defensive kernel-to-compiler mapping directly instead of
  // fabricating a different, valid same-stage Graph fact.
  EXPECT_EQ(detail::lifecycle_plan_diagnostic_token(
                detail::LifecycleKernelErrorCode::kCrossStageDependency),
            "LIFECYCLE_DEPENDENCY_STAGE_INVALID");

  auto cycle = *canonical;
  const auto direct = cycle.lifecycle_dependencies.front();
  cycle.lifecycle_dependencies.push_back({
      .graph = direct.graph,
      .before = direct.after,
      .after = direct.before,
      .stage = direct.stage,
  });
  // Key assertion: a back-edge in the dependency graph triggers LIFECYCLE_DEPENDENCY_CYCLE.
  EXPECT_TRUE(
      has_lifecycle_diagnostic(lower_lifecycle_snapshot(cycle), "LIFECYCLE_DEPENDENCY_CYCLE"));

  auto duplicate_callback = *canonical;
  duplicate_callback.lifecycle_callbacks.push_back(duplicate_callback.lifecycle_callbacks.front());
  EXPECT_TRUE(has_lifecycle_diagnostic(lower_lifecycle_snapshot(duplicate_callback),
                                       "LIFECYCLE_CALLBACK_DUPLICATE"));

  auto ambiguous = *canonical;
  ASSERT_GE(ambiguous.vertices.size(), 2U);
  ambiguous.vertices[1].name = ambiguous.vertices[0].name;
  EXPECT_TRUE(has_lifecycle_diagnostic(lower_lifecycle_snapshot(ambiguous),
                                       "LIFECYCLE_NODE_IDENTITY_AMBIGUOUS"));
}

// Single-operator graph; the lifecycle mutable-field record carries an invalid
// from_stage encoded as bit_cast<LifecycleStage>(0xFF), which lies outside
// the valid 6-bit mask range (0x3F).
//
// REQ: COMPILE-008
TEST(CompileLifecycle, RejectsInvalidRestartStageMaskWithBoundedPayload) {
  Graph graph{"lifecycle-invalid-stage-mask"};
  (void)graph.op<test::CompileTestOperator>("operator");
  const auto snapshot = detail::GraphCompilerAccess::canonical_snapshot(graph);
  ASSERT_TRUE(snapshot);
  // Key setup: from_stage carries 0xFF, which has bits set above the valid 0x3F mask.
  const std::array fields{LifecycleMutableFieldPlanRecord{
      .owner = OperatorRef{.id = 1U, .plan_generation = 1U, .run_generation = 0U},
      .field = ParamFieldId{1U},
      .from_stage = std::bit_cast<LifecycleStage>(std::uint8_t{255U}),
  }};

  const auto lowered = lower_lifecycle_snapshot(*snapshot, fields);
  ASSERT_FALSE(lowered.accepted);
  const auto diagnostic =
      std::find_if(lowered.diagnostics.begin(), lowered.diagnostics.end(), [](const auto& item) {
        return item.token == "LIFECYCLE_RESTART_STAGE_MASK_INVALID";
      });
  ASSERT_NE(diagnostic, lowered.diagnostics.end());
  // Key assertion: the diagnostic json_pointer names the invalid field and
  //               required/provided expose the valid mask and the supplied value.
  EXPECT_EQ(diagnostic->json_pointer, "/lifecycle_mutable_fields/from_stage");
  EXPECT_EQ(diagnostic->required, "0x3f");
  EXPECT_EQ(diagnostic->provided, "255");
}

// Two-operator graph via author_lifecycle_graph; compiled twice — once without
// lifecycle_transactions (disabled path) and once with ample capacity.
//
//   first --[output->input]--> second
//   lifecycle: before(kStart), self_restart(first), external_restart(second),
//              restart_request(first -> second)
//
// REQ: COMPILE-008
TEST(CompileLifecycle, LowersTotalRestartDispositionAndExactCapacity) {
  Graph disabled_graph{"lifecycle-disabled"};
  author_lifecycle_graph(disabled_graph);
  const ExecutionPlan disabled = compile(disabled_graph);
  ASSERT_TRUE(disabled.ok());
  ASSERT_EQ(disabled.lifecycle_restart_targets().size(), 2U);
  EXPECT_TRUE(std::all_of(disabled.lifecycle_restart_targets().begin(),
                          disabled.lifecycle_restart_targets().end(),
                          [](const auto& target) {
                            return target.disposition ==
                                   LifecycleRestartTargetDisposition::kDisabled;
                          }));
  // Key assertion: omitting lifecycle_transactions yields kDisabled disposition
  //               and no capacity evidence in the plan.
  EXPECT_FALSE(disabled.lifecycle_transaction_capacity().has_value());

  Graph enabled_graph{"lifecycle-enabled"};
  author_lifecycle_graph(enabled_graph);
  CompileOptions options;
  options.lifecycle_transactions = ample_lifecycle_capacity();
  const ExecutionPlan enabled = compile(enabled_graph, options);
  ASSERT_TRUE(enabled.ok());
  const auto capacity = enabled.lifecycle_transaction_capacity();
  if (!capacity.has_value()) {
    ADD_FAILURE() << "expected lifecycle transaction capacity evidence";
    return;
  }
  EXPECT_EQ(capacity->configured.max_pending_requests, 64U);
  EXPECT_EQ(capacity->configured.max_pending_transactions, 64U);
  EXPECT_EQ(capacity->configured.max_receipts, 128U);
  EXPECT_EQ(capacity->configured.max_fenced_operators, 64U);
  EXPECT_EQ(capacity->configured.max_candidate_generations, 64U);
  EXPECT_EQ(capacity->configured.max_events, 1024U);
  EXPECT_EQ(capacity->configured.max_audit_records, 256U);
  EXPECT_EQ(capacity->configured.max_quiesce_records, 256U);
  EXPECT_EQ(capacity->configured.max_rollback_records, 256U);
  EXPECT_LE(capacity->required_minimum.max_pending_requests,
            capacity->configured.max_pending_requests);
  EXPECT_LE(capacity->required_minimum.max_fenced_operators,
            capacity->configured.max_fenced_operators);
  ASSERT_EQ(enabled.lifecycle_restart_targets().size(), 2U);
  EXPECT_TRUE(
      std::all_of(enabled.lifecycle_restart_targets().begin(),
                  enabled.lifecycle_restart_targets().end(),
                  [](const auto& target) {
                    return target.disposition == LifecycleRestartTargetDisposition::kAdmitted ||
                           target.disposition == LifecycleRestartTargetDisposition::kNotRequested;
                  }));
}

// Graph via author_lifecycle_graph; compiled 6 times: 3 policies × 2 lifecycle
// states (enabled/disabled). When enabled the resolved policy must match the
// requested one; restart targets must all be admitted and not parameter-origin.
//
//   first --[output->input]--> second   (lifecycle: same as LowersTotalRestart...)
//
// REQ: COMPILE-008
// REQ: COMPILE-010
TEST(CompileLifecycle, PreservesTheClosedParameterPolicyByLifecycleOptionMatrix) {
  struct PolicyCase {
    ParamRuntimeUpdatePolicy policy;
    ParamStorageMode storage;
  };
  constexpr std::array cases{
      PolicyCase{ParamRuntimeUpdatePolicy::kDisabled, ParamStorageMode::kDynamicColdPath},
      PolicyCase{ParamRuntimeUpdatePolicy::kEnabledIfAdmitted, ParamStorageMode::kProfileDefault},
      PolicyCase{ParamRuntimeUpdatePolicy::kRequirePreallocated,
                 ParamStorageMode::kPreallocatedBounded},
  };

  for (const auto& policy : cases) {
    for (const bool lifecycle_enabled : {false, true}) {
      Graph graph{"lifecycle-parameter-matrix"};
      author_lifecycle_graph(graph);
      CompileOptions options;
      options.parameters.runtime_updates = policy.policy;
      options.parameters.storage = policy.storage;
      if (lifecycle_enabled) {
        options.lifecycle_transactions = ample_lifecycle_capacity();
      }

      const ExecutionPlan plan = compile(graph, options);
      ASSERT_TRUE(plan.ok()) << plan.json();
      ASSERT_EQ(plan.lifecycle_restart_targets().size(), 2U);
      if (!lifecycle_enabled) {
        // Key setup: without lifecycle_transactions the plan must record kDisabled
        //            and must not emit capacity evidence regardless of policy.
        EXPECT_FALSE(plan.lifecycle_transaction_capacity().has_value());
        EXPECT_TRUE(std::all_of(plan.lifecycle_restart_targets().begin(),
                                plan.lifecycle_restart_targets().end(),
                                [](const auto& target) {
                                  return target.disposition ==
                                         LifecycleRestartTargetDisposition::kDisabled;
                                }));
        continue;
      }

      ASSERT_TRUE(plan.lifecycle_transaction_capacity().has_value());
      // Key assertion: resolved_parameter_update_policy echoes the requested
      //               policy and no target carries parameter_origin_allowed.
      EXPECT_EQ(plan.lifecycle_transaction_capacity()->resolved_parameter_update_policy,
                policy.policy);
      EXPECT_TRUE(std::all_of(plan.lifecycle_restart_targets().begin(),
                              plan.lifecycle_restart_targets().end(),
                              [](const auto& target) {
                                return target.disposition ==
                                       LifecycleRestartTargetDisposition::kAdmitted;
                              }));
      EXPECT_TRUE(std::none_of(plan.lifecycle_restart_targets().begin(),
                               plan.lifecycle_restart_targets().end(),
                               [](const auto& target) { return target.parameter_origin_allowed; }));
    }
  }
}

// Graph via author_lifecycle_graph; verifies two distinct failure modes:
// (1) all-zero capacity → LIFECYCLE_TRANSACTION_CAPACITY_INVALID,
// (2) ample capacity with max_fenced_operators=1 → INSUFFICIENT.
//
//   first --[output->input]--> second   (lifecycle: same as LowersTotalRestart...)
//
// REQ: COMPILE-008
TEST(CompileLifecycle, RejectsInvalidOrInsufficientTransactionCapacity) {
  Graph graph{"lifecycle-insufficient"};
  author_lifecycle_graph(graph);
  CompileOptions options;
  // Key setup: all-zero LifecycleTransactionOptions must be rejected before any
  //            runtime execution; no compute or lifecycle calls should occur.
  options.lifecycle_transactions = LifecycleTransactionOptions{};

  const ExecutionPlan plan = compile(graph, options);
  EXPECT_FALSE(plan.ok());
  EXPECT_FALSE(plan.diagnostics().empty());
  const auto invalid_capacity = plan.lifecycle_transaction_capacity();
  if (!invalid_capacity.has_value()) {
    ADD_FAILURE() << "expected invalid lifecycle capacity evidence";
    return;
  }
  EXPECT_EQ(invalid_capacity->configured.max_pending_requests, 0U);
  // Key assertion: the compiler derives a minimum of 2 pending_requests for
  //               the two-operator graph and surfaces it as capacity evidence.
  EXPECT_EQ(invalid_capacity->required_minimum.max_pending_requests, 2U);
  EXPECT_TRUE(test::has_diagnostic(plan, "LIFECYCLE_TRANSACTION_CAPACITY_INVALID"));
  EXPECT_EQ(test::CompileTestOperator::compute_calls, 0U);
  EXPECT_EQ(test::CompileTestOperator::lifecycle_calls, 0U);

  CompileOptions insufficient;
  insufficient.lifecycle_transactions = ample_lifecycle_capacity();
  insufficient.lifecycle_transactions->max_fenced_operators = 1U;
  const ExecutionPlan insufficient_plan = compile(graph, insufficient);
  ASSERT_FALSE(insufficient_plan.ok());
  const auto insufficient_capacity = insufficient_plan.lifecycle_transaction_capacity();
  if (!insufficient_capacity.has_value()) {
    ADD_FAILURE() << "expected insufficient lifecycle capacity evidence";
    return;
  }
  EXPECT_EQ(insufficient_capacity->configured.max_fenced_operators, 1U);
  EXPECT_GT(insufficient_capacity->required_minimum.max_fenced_operators,
            insufficient_capacity->configured.max_fenced_operators);
  EXPECT_TRUE(
      test::has_diagnostic(insufficient_plan, "LIFECYCLE_TRANSACTION_CAPACITY_INSUFFICIENT"));
  const auto diagnostic =
      std::find_if(insufficient_plan.diagnostics().begin(),
                   insufficient_plan.diagnostics().end(),
                   [](const auto& item) {
                     return item.token == "LIFECYCLE_TRANSACTION_CAPACITY_INSUFFICIENT" &&
                            item.json_pointer == "/lifecycle_transactions/max_fenced_operators";
                   });
  ASSERT_NE(diagnostic, insufficient_plan.diagnostics().end());
  EXPECT_EQ(diagnostic->required,
            std::to_string(insufficient_capacity->required_minimum.max_fenced_operators));
  EXPECT_EQ(diagnostic->provided, "1");
}

// Graph via author_lifecycle_graph; four capacity fields set to UINT32_MAX so
// the compiler's derived minima (which multiply or add the configured values)
// overflow uint32 and cannot be represented as a required_minimum record.
//
//   first --[output->input]--> second   (lifecycle: same as LowersTotalRestart...)
//
// REQ: COMPILE-042
// REQ: COMPILE-045
TEST(CompileLifecycle, RejectsUnrepresentableDerivedTransactionCapacity) {
  Graph graph{"lifecycle-capacity-overflow"};
  author_lifecycle_graph(graph);
  CompileOptions options;
  auto capacity = ample_lifecycle_capacity();
  // Key setup: four fields at UINT32_MAX cause derived-minimum calculations to
  //            overflow; the plan must report OVERFLOW rather than INSUFFICIENT.
  capacity.max_pending_requests = std::numeric_limits<std::uint32_t>::max();
  capacity.max_pending_transactions = std::numeric_limits<std::uint32_t>::max();
  capacity.max_candidate_generations = std::numeric_limits<std::uint32_t>::max();
  capacity.max_audit_records = std::numeric_limits<std::uint32_t>::max();
  options.lifecycle_transactions = capacity;

  const ExecutionPlan plan = compile(graph, options);

  ASSERT_FALSE(plan.ok());
  const auto diagnostic =
      std::find_if(plan.diagnostics().begin(), plan.diagnostics().end(), [](const auto& item) {
        return item.token == "LIFECYCLE_TRANSACTION_CAPACITY_OVERFLOW";
      });
  ASSERT_NE(diagnostic, plan.diagnostics().end());
  EXPECT_EQ(diagnostic->json_pointer, "/lifecycle_transactions");
  EXPECT_EQ(diagnostic->required, "representable uint32 derived minima");
  EXPECT_EQ(diagnostic->provided, "overflow");
  // Key assertion: overflow suppresses INSUFFICIENT; the saturated fields are
  //               preserved in required_minimum for diagnostics visibility.
  EXPECT_FALSE(test::has_diagnostic(plan, "LIFECYCLE_TRANSACTION_CAPACITY_INSUFFICIENT"));
  ASSERT_TRUE(plan.lifecycle_transaction_capacity().has_value());
  EXPECT_EQ(plan.lifecycle_transaction_capacity()->required_minimum.max_receipts,
            std::numeric_limits<std::uint32_t>::max());
  EXPECT_EQ(plan.lifecycle_transaction_capacity()->required_minimum.max_events,
            std::numeric_limits<std::uint32_t>::max());
}

// Graph via author_lifecycle_graph; iterated over all 9 capacity fields. Each
// iteration zeros exactly one field while the other eight stay ample; verifies
// that every per-field diagnostic carries the correct json_pointer.
//
//   first --[output->input]--> second   (lifecycle: same as LowersTotalRestart...)
//
// REQ: COMPILE-008
TEST(CompileLifecycle, RetainsEveryInvalidTransactionCapacityField) {
  struct CapacityFieldCase {
    std::string_view name;
    std::uint32_t LifecycleTransactionOptions::* option;
    std::uint32_t LifecycleTransactionCapacityValues::* record;
  };
  constexpr std::array cases{
      CapacityFieldCase{"max_pending_requests",
                        &LifecycleTransactionOptions::max_pending_requests,
                        &LifecycleTransactionCapacityValues::max_pending_requests},
      CapacityFieldCase{"max_pending_transactions",
                        &LifecycleTransactionOptions::max_pending_transactions,
                        &LifecycleTransactionCapacityValues::max_pending_transactions},
      CapacityFieldCase{"max_receipts",
                        &LifecycleTransactionOptions::max_receipts,
                        &LifecycleTransactionCapacityValues::max_receipts},
      CapacityFieldCase{"max_fenced_operators",
                        &LifecycleTransactionOptions::max_fenced_operators,
                        &LifecycleTransactionCapacityValues::max_fenced_operators},
      CapacityFieldCase{"max_candidate_generations",
                        &LifecycleTransactionOptions::max_candidate_generations,
                        &LifecycleTransactionCapacityValues::max_candidate_generations},
      CapacityFieldCase{"max_events",
                        &LifecycleTransactionOptions::max_events,
                        &LifecycleTransactionCapacityValues::max_events},
      CapacityFieldCase{"max_audit_records",
                        &LifecycleTransactionOptions::max_audit_records,
                        &LifecycleTransactionCapacityValues::max_audit_records},
      CapacityFieldCase{"max_quiesce_records",
                        &LifecycleTransactionOptions::max_quiesce_records,
                        &LifecycleTransactionCapacityValues::max_quiesce_records},
      CapacityFieldCase{"max_rollback_records",
                        &LifecycleTransactionOptions::max_rollback_records,
                        &LifecycleTransactionCapacityValues::max_rollback_records},
  };

  Graph graph{"lifecycle-invalid-fields"};
  author_lifecycle_graph(graph);
  for (const auto& field : cases) {
    LifecycleTransactionOptions candidate = ample_lifecycle_capacity();
    // Key setup: zero exactly one field per iteration; all others remain ample
    //            so that only the targeted field produces a diagnostic.
    candidate.*field.option = 0U;
    CompileOptions options;
    options.lifecycle_transactions = candidate;
    const ExecutionPlan plan = compile(graph, options);
    ASSERT_FALSE(plan.ok());
    const auto capacity = plan.lifecycle_transaction_capacity();
    if (!capacity.has_value()) {
      ADD_FAILURE() << "expected invalid lifecycle capacity evidence";
      return;
    }
    EXPECT_EQ(capacity->configured.*field.record, 0U);
    EXPECT_GT(capacity->required_minimum.*field.record, 0U);
    // Key assertion: the per-field diagnostic json_pointer names the zeroed
    //               field; required="positive" and provided="0".
    EXPECT_TRUE(test::has_diagnostic(plan, "LIFECYCLE_TRANSACTION_CAPACITY_INVALID"));
    const auto diagnostic =
        std::find_if(plan.diagnostics().begin(), plan.diagnostics().end(), [&](const auto& item) {
          return item.token == "LIFECYCLE_TRANSACTION_CAPACITY_INVALID" &&
                 item.json_pointer == "/lifecycle_transactions/" + std::string(field.name);
        });
    ASSERT_NE(diagnostic, plan.diagnostics().end());
    EXPECT_EQ(diagnostic->required, "positive");
    EXPECT_EQ(diagnostic->provided, "0");
  }
}

// Graph via author_lifecycle_graph; verifies that all plan-bound references
// (checkpoints, dependencies, permissions, targets, closures) carry resolved
// OperatorRef values rather than raw VertexId authoring-time identifiers.
//
//   first --[output->input]--> second   (lifecycle: same as LowersTotalRestart...)
//
// REQ: COMPILE-008
TEST(CompileLifecycle, PreservesRequestedEffectiveStagesAndPermissionEdges) {
  Graph graph{"lifecycle-records"};
  author_lifecycle_graph(graph);
  CompileOptions options;
  options.lifecycle_transactions = ample_lifecycle_capacity();
  const ExecutionPlan plan = compile(graph, options);
  ASSERT_TRUE(plan.ok());

  ASSERT_FALSE(plan.lifecycle_checkpoints().empty());
  EXPECT_TRUE(std::all_of(
      plan.lifecycle_checkpoints().begin(),
      plan.lifecycle_checkpoints().end(),
      [](const auto& checkpoint) { return checkpoint.operator_instance.plan_bound(); }));
  const auto dependency =
      std::find_if(plan.lifecycle_dependencies().begin(),
                   plan.lifecycle_dependencies().end(),
                   [](const auto& item) { return item.stage == LifecycleStage::kStart; });
  ASSERT_NE(dependency, plan.lifecycle_dependencies().end());
  EXPECT_TRUE(dependency->before.plan_bound());
  EXPECT_TRUE(dependency->after.plan_bound());

  // Key assertion: exactly one restart_permission record with distinct plan-bound
  //               requester and target OperatorRefs confirms the authoring edge
  //               survived lowering without conflation.
  ASSERT_EQ(plan.lifecycle_restart_permissions().size(), 1U);
  EXPECT_TRUE(plan.lifecycle_restart_permissions()[0].requester.plan_bound());
  EXPECT_TRUE(plan.lifecycle_restart_permissions()[0].target.plan_bound());
  EXPECT_NE(plan.lifecycle_restart_permissions()[0].requester,
            plan.lifecycle_restart_permissions()[0].target);
  for (const auto& target : plan.lifecycle_restart_targets()) {
    EXPECT_TRUE(target.target.plan_bound());
    EXPECT_EQ(target.target.run_generation, 0U);
    EXPECT_NE(target.allowed_from_stage_mask, 0U);
  }
  ASSERT_EQ(plan.lifecycle_restart_closures().size(), 2U);
  const auto first_closure =
      std::find_if(plan.lifecycle_restart_closures().begin(),
                   plan.lifecycle_restart_closures().end(),
                   [](const auto& record) { return record.closure.size() == 2U; });
  ASSERT_NE(first_closure, plan.lifecycle_restart_closures().end());
  EXPECT_EQ(first_closure->valid_requested_stage_mask, 0x3FU);
  EXPECT_EQ(first_closure->required_fenced_operators, 2U);
  EXPECT_EQ(first_closure->required_quiesce_records, 2U);
  EXPECT_EQ(first_closure->required_rollback_records, 2U);
  EXPECT_TRUE(std::is_sorted(
      first_closure->closure.begin(),
      first_closure->closure.end(),
      [](const OperatorRef& left, const OperatorRef& right) { return left.id < right.id; }));
  for (std::size_t index = 0U; index < first_closure->stage_resolutions.size(); ++index) {
    EXPECT_EQ(static_cast<std::size_t>(first_closure->stage_resolutions[index].requested_stage),
              index);
    EXPECT_EQ(first_closure->stage_resolutions[index].required_structural_event_records, 8U);
  }
}

// Transitive before-chain with two different widening stages. The effective
// restart stage for each target must be the earliest in its full closure.
//
//   first --[before, kAllocate]--> second --[before, kWarmUp]--> third
//   all three: allow_self_restart
//
// REQ: COMPILE-008
TEST(CompileLifecycle, WidensEachTargetsStageFromItsOwnTransitiveDependencyClosure) {
  Graph graph{"lifecycle-closure-widening"};
  const auto first = graph.op<test::CompileTestOperator>("first");
  const auto second = graph.op<test::CompileTestOperator>("second");
  const auto third = graph.op<test::CompileTestOperator>("third");
  graph.lifecycle()
      .before(first, second, LifecycleStage::kAllocate)
      .before(second, third, LifecycleStage::kWarmUp)
      .allow_self_restart(first)
      .allow_self_restart(second)
      .allow_self_restart(third);
  graph.partition("first").add(first);
  graph.partition("second").add(second);
  graph.partition("third").add(third);

  CompileOptions options;
  options.lifecycle_transactions = ample_lifecycle_capacity();
  const ExecutionPlan plan = compile(graph, options);
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.lifecycle_restart_closures().size(), 3U);

  const auto& first_record = plan.lifecycle_restart_closures()[0];
  const auto& second_record = plan.lifecycle_restart_closures()[1];
  const auto& third_record = plan.lifecycle_restart_closures()[2];
  EXPECT_EQ(first_record.closure.size(), 3U);
  EXPECT_EQ(second_record.closure.size(), 2U);
  EXPECT_EQ(third_record.closure.size(), 1U);
  const auto start_index = static_cast<std::size_t>(LifecycleStage::kStart);
  // Key assertion: first's kStart is widened to kAllocate (earliest before-stage
  //               in its 3-operator closure); third's kStart stays kStart.
  EXPECT_EQ(first_record.stage_resolutions[start_index].effective_stage, LifecycleStage::kAllocate);
  EXPECT_EQ(first_record.stage_resolutions[start_index].reason,
            LifecycleRestartWideningReason::kDependencyStage);
  EXPECT_EQ(first_record.stage_resolutions[start_index].required_quiesce_records, 0U);
  EXPECT_EQ(first_record.stage_resolutions[start_index].required_rollback_records, 0U);
  EXPECT_EQ(second_record.stage_resolutions[start_index].effective_stage, LifecycleStage::kWarmUp);
  EXPECT_EQ(third_record.stage_resolutions[start_index].effective_stage, LifecycleStage::kStart);
  EXPECT_EQ(third_record.stage_resolutions[start_index].reason,
            LifecycleRestartWideningReason::kNone);
  ASSERT_TRUE(plan.lifecycle_transaction_capacity());
  EXPECT_EQ(plan.lifecycle_transaction_capacity()->required_minimum.max_fenced_operators, 3U);
  EXPECT_EQ(plan.lifecycle_transaction_capacity()->required_minimum.max_quiesce_records, 6U);
  EXPECT_EQ(plan.lifecycle_transaction_capacity()->required_minimum.max_rollback_records, 6U);
}

// Two independent self-restart targets with no lifecycle edges between them;
// the required capacity must be the union of both closures, not the max of one.
//
//   first (allow_self_restart)    second (allow_self_restart)    [no edges]
//
// REQ: COMPILE-008
TEST(CompileLifecycle, SizesCapacityForCompatibleMultiTargetClosureUnion) {
  Graph graph{"lifecycle-multi-target-union"};
  const auto first = graph.op<test::CompileTestOperator>("first");
  const auto second = graph.op<test::CompileTestOperator>("second");
  graph.lifecycle().allow_self_restart(first).allow_self_restart(second);

  CompileOptions options;
  options.lifecycle_transactions = ample_lifecycle_capacity();
  // Key setup: tighten exactly the three union-sensitive fields to 1 so the
  //            required minimum of 2/4/4 causes INSUFFICIENT.
  options.lifecycle_transactions->max_fenced_operators = 1U;
  options.lifecycle_transactions->max_quiesce_records = 1U;
  options.lifecycle_transactions->max_rollback_records = 1U;
  const ExecutionPlan plan = compile(graph, options);

  ASSERT_FALSE(plan.ok());
  const auto capacity = plan.lifecycle_transaction_capacity();
  ASSERT_TRUE(capacity);
  const auto& required = capacity->required_minimum;
  // Key assertion: required fields are 2/4/4 — the union of the two single-
  //               operator closures, not just the max of one closure.
  EXPECT_EQ(required.max_fenced_operators, 2U);
  EXPECT_EQ(required.max_quiesce_records, 4U);
  EXPECT_EQ(required.max_rollback_records, 4U);
  EXPECT_TRUE(test::has_diagnostic(plan, "LIFECYCLE_TRANSACTION_CAPACITY_INSUFFICIENT"));
  ASSERT_EQ(plan.lifecycle_restart_closures().size(), 2U);
  EXPECT_TRUE(std::all_of(plan.lifecycle_restart_closures().begin(),
                          plan.lifecycle_restart_closures().end(),
                          [](const auto& record) { return record.closure.size() == 1U; }));
}

// One external-restart target and one disconnected operator. Compiles twice:
// first with generous capacity to read the required_minimum, then at exactly
// that minimum; then exercises the kernel directly for all stage resolutions.
//
//   restartable (allow_external_restart)
//   disconnected  [no lifecycle edges; stresses per-operator capacity sizing]
//
// REQ: COMPILE-008
// REQ: COMPILE-045
TEST(CompileLifecycle, ExactMinimumAdmitsActiveAndSuccessorInTransactionKernel) {
  Graph graph{"lifecycle-kernel-capacity-conformance"};
  const auto restartable = graph.op<test::CompileTestOperator>("restartable");
  const auto disconnected = graph.op<test::CompileTestOperator>("disconnected");
  graph.lifecycle().allow_external_restart(restartable);
  graph.partition("restartable").add(restartable);
  graph.partition("disconnected").add(disconnected);

  CompileOptions options;
  options.lifecycle_transactions = LifecycleTransactionOptions{
      .max_pending_requests = 2U,
      .max_pending_transactions = 2U,
      .max_receipts = 4U,
      .max_fenced_operators = 8U,
      .max_candidate_generations = 2U,
      .max_events = 32U,
      .max_audit_records = 4U,
      .max_quiesce_records = 8U,
      .max_rollback_records = 8U,
  };
  // Key setup: compile with a deliberately generous but finite capacity to
  //            obtain the exact required_minimum; re-compile at that minimum.
  const ExecutionPlan provisional = compile(graph, options);
  ASSERT_TRUE(provisional.ok()) << provisional.json();
  ASSERT_TRUE(provisional.lifecycle_transaction_capacity());
  const auto required = provisional.lifecycle_transaction_capacity()->required_minimum;
  EXPECT_EQ(required.max_pending_requests, 2U);
  EXPECT_EQ(required.max_pending_transactions, 2U);
  EXPECT_EQ(required.max_receipts, 4U);
  EXPECT_EQ(required.max_fenced_operators, 1U);
  EXPECT_EQ(required.max_candidate_generations, 2U);
  EXPECT_EQ(required.max_events, 32U);
  EXPECT_EQ(required.max_audit_records, 4U);
  EXPECT_EQ(required.max_quiesce_records, 2U);
  EXPECT_EQ(required.max_rollback_records, 2U);

  options.lifecycle_transactions = lifecycle_options(required);
  const ExecutionPlan exact = compile(graph, options);
  ASSERT_TRUE(exact.ok()) << exact.json();
  // Key assertion: exact-minimum re-compilation admits the restartable target;
  //               the kernel then accepts concurrent active + successor requests.
  const auto admitted_target =
      std::find_if(exact.lifecycle_restart_targets().begin(),
                   exact.lifecycle_restart_targets().end(),
                   [](const auto& target) {
                     return target.disposition == LifecycleRestartTargetDisposition::kAdmitted;
                   });
  ASSERT_NE(admitted_target, exact.lifecycle_restart_targets().end());
  const auto closure = std::find_if(
      exact.lifecycle_restart_closures().begin(),
      exact.lifecycle_restart_closures().end(),
      [&](const auto& candidate) { return candidate.target == admitted_target->target; });
  ASSERT_NE(closure, exact.lifecycle_restart_closures().end());
  ASSERT_EQ(closure->closure.size(), 1U);

  std::vector<OperatorInstanceId> closure_members;
  closure_members.reserve(closure->closure.size());
  for (const auto& member : closure->closure) {
    closure_members.push_back(member.id);
  }
  OperatorRef run_target = admitted_target->target;
  run_target.run_generation = 1U;

  for (std::size_t index = 0U; index < closure->stage_resolutions.size(); ++index) {
    const auto& stage = closure->stage_resolutions[index];
    detail::LifecycleTransactionKernel kernel(kernel_capacity(required));
    const auto request = [&](std::uint64_t compatibility) {
      return detail::LifecycleTransactionRequest{
          .origin = detail::LifecycleTransactionRequestOrigin::kExternalControl,
          .target = run_target,
          .requested_stage = stage.requested_stage,
          .effective_stage = stage.effective_stage,
          .compatibility =
              detail::LifecycleTransactionCompatibilityKey{
                  .authorization = compatibility,
                  .profile = 1U,
                  .rollback = 1U,
                  .deadline = 1U,
                  .closure = 1U,
              },
          .closure = closure_members,
          .target_state = detail::LifecycleTransactionTargetState::kIdle,
          .structural_event_delta = stage.required_structural_event_records,
      };
    };

    const auto first = kernel.submit(request(1U));
    ASSERT_TRUE(first.has_value()) << index;
    const auto first_view = kernel.inspect(*first);
    ASSERT_TRUE(first_view.has_value()) << index;
    ASSERT_TRUE(kernel.acknowledge_gate(first_view->transaction_id,
                                        run_target.id,
                                        detail::LifecycleTransactionGate::kControlSafePoint))
        << index;
    ASSERT_TRUE(kernel.begin_quiescing(first_view->transaction_id)) << index;

    const auto successor = kernel.submit(request(2U));
    EXPECT_TRUE(successor.has_value()) << index;
    EXPECT_EQ(kernel.pending_transaction_count(), 2U) << index;
    EXPECT_EQ(kernel.reserved_event_count(), 16U) << index;
    EXPECT_TRUE(kernel.is_fenced(run_target.id)) << index;

    // Retain one superseded predecessor for each live active/successor request.
    detail::LifecycleTransactionKernel retained(kernel_capacity(required));
    const auto candidate_request = [&](ParamGeneration generation, bool equivalent = false) {
      auto candidate = request(1U);
      candidate.origin = detail::LifecycleTransactionRequestOrigin::kParameter;
      candidate.candidate_generation = generation;
      candidate.parameter_candidate_equivalent = equivalent;
      return candidate;
    };
    const auto first_candidate = retained.submit(candidate_request(1U));
    ASSERT_TRUE(first_candidate) << index;
    const auto second_candidate = retained.submit(candidate_request(2U));
    ASSERT_TRUE(second_candidate) << index;
    const auto retained_first_view = retained.inspect(*first_candidate);
    const auto retained_second_view = retained.inspect(*second_candidate);
    ASSERT_TRUE(retained_first_view) << index;
    ASSERT_TRUE(retained_second_view) << index;
    EXPECT_EQ(retained_first_view->state, LifecycleTransactionState::kSuperseded) << index;
    ASSERT_TRUE(retained.acknowledge_gate(retained_second_view->transaction_id,
                                          run_target.id,
                                          detail::LifecycleTransactionGate::kControlSafePoint))
        << index;
    ASSERT_TRUE(retained.begin_quiescing(retained_second_view->transaction_id)) << index;

    const auto third_candidate = retained.submit(candidate_request(3U));
    ASSERT_TRUE(third_candidate) << index;
    const auto fourth_candidate = retained.submit(candidate_request(4U));
    ASSERT_TRUE(fourth_candidate) << index;
    const auto retained_third_view = retained.inspect(*third_candidate);
    ASSERT_TRUE(retained_third_view) << index;
    EXPECT_EQ(retained_third_view->state, LifecycleTransactionState::kSuperseded) << index;
    EXPECT_EQ(retained.pending_request_count(), 2U) << index;
    EXPECT_EQ(retained.pending_transaction_count(), 2U) << index;
    EXPECT_EQ(retained.reserved_event_count(), 32U) << index;

    const auto receipt_full = retained.submit(candidate_request(5U));
    ASSERT_FALSE(receipt_full) << index;
    EXPECT_EQ(receipt_full.error().capacity, detail::LifecycleTransactionCapacityKind::kReceipts)
        << index;
    ASSERT_TRUE(retained.retire_receipt(*first_candidate)) << index;
    EXPECT_EQ(retained.reserved_event_count(), 24U) << index;
    EXPECT_TRUE(retained.submit(candidate_request(5U))) << index;
    EXPECT_EQ(retained.reserved_event_count(), 32U) << index;

    // Give the earlier checks headroom so audit exhaustion is observed directly.
    auto audit_capacity = kernel_capacity(required);
    ++audit_capacity.max_receipts;
    audit_capacity.max_events += stage.required_structural_event_records;
    detail::LifecycleTransactionKernel audit_limited(audit_capacity);
    for (ParamGeneration generation = 1U; generation <= 4U; ++generation) {
      EXPECT_TRUE(audit_limited.submit(candidate_request(generation, true))) << index;
    }
    const auto audit_full = audit_limited.submit(candidate_request(5U, true));
    ASSERT_FALSE(audit_full) << index;
    EXPECT_EQ(audit_full.error().capacity, detail::LifecycleTransactionCapacityKind::kAuditRecords)
        << index;
  }
}

// Two-operator graph with a restart_request edge; a second permission record
// with a non-existent requester vertex is injected into the canonical snapshot.
// The valid permission must survive alongside the rejection diagnostic.
//
//   first --[allow_restart_request]--> second
//   (+ injected: invalid_requester --[allow_restart_request]--> second)
//
// REQ: COMPILE-008
// REQ: COMPILE-013
TEST(CompileLifecycle, RejectedPermissionsRetainResolvedTargetAndValidSiblingEvidence) {
  Graph graph{"lifecycle-rejected-permission-evidence"};
  const auto first = graph.op<test::CompileTestOperator>("first");
  const auto second = graph.op<test::CompileTestOperator>("second");
  graph.lifecycle().allow_restart_request(first, second);

  auto snapshot = *detail::GraphCompilerAccess::canonical_snapshot(graph);
  ASSERT_EQ(snapshot.lifecycle_restart_permissions.size(), 1U);
  const auto valid = snapshot.lifecycle_restart_permissions.front();
  // Key setup: inject a permission whose requester vertex id is beyond the
  //            snapshot's vertex table, triggering LIFECYCLE_RESTART_TARGET_UNSUPPORTED.
  snapshot.lifecycle_restart_permissions.push_back(LifecycleRestartPermissionAuthoringRecord{
      .origin = LifecycleRestartOrigin::kOperatorRequest,
      .requester = VertexId{static_cast<std::uint64_t>(snapshot.vertices.size() + 100U)},
      .target = valid.target,
  });
  const std::vector<std::optional<Profile>> profiles(snapshot.vertices.size(), Profile::kStandard);
  const auto lowered = detail::lower_lifecycle_plan(
      snapshot, profiles, {}, std::nullopt, ample_lifecycle_capacity(), PlanGenerationId{1U});

  // Key assertion: lowering is rejected but the valid permission is preserved in
  //               lowered.restart_permissions with distinct requester/target refs.
  EXPECT_FALSE(lowered.accepted);
  const auto diagnostic =
      std::find_if(lowered.diagnostics.begin(), lowered.diagnostics.end(), [](const auto& item) {
        return item.token == "LIFECYCLE_RESTART_TARGET_UNSUPPORTED";
      });
  ASSERT_NE(diagnostic, lowered.diagnostics.end());
  EXPECT_EQ(diagnostic->json_pointer, "/lifecycle/restart_permissions");
  EXPECT_EQ(diagnostic->required, "valid_origin_requester_target_authority");
  EXPECT_EQ(diagnostic->provided, "operator_request:requester_invalid");
  ASSERT_EQ(lowered.restart_permissions.size(), 1U);
  EXPECT_NE(lowered.restart_permissions.front().requester,
            lowered.restart_permissions.front().target);
  const auto target = std::find_if(
      lowered.restart_targets.begin(), lowered.restart_targets.end(), [&](const auto& record) {
        return record.target == lowered.restart_permissions.front().target;
      });
  ASSERT_NE(target, lowered.restart_targets.end());
  EXPECT_EQ(target->disposition, LifecycleRestartTargetDisposition::kRejected);
}

// Two self-restart targets in different region profiles; the compiler must
// resolve each target's profile independently and record it in the plan.
//
//   standard (allow_self_restart, default region: kStandard)
//   robotics (allow_self_restart, region: kRobotics)
//
// REQ: COMPILE-003
// REQ: COMPILE-008
TEST(CompileLifecycle, UsesEachTargetsEffectiveRegionProfile) {
  Graph graph{"lifecycle-effective-profiles"};
  const auto standard = graph.op<test::CompileTestOperator>("standard");
  const auto robotics = graph.op<test::CompileTestOperator>("robotics");
  graph.lifecycle().allow_self_restart(standard).allow_self_restart(robotics);
  graph.region("robotics-region").add(robotics).profile(Profile::kRobotics);
  graph.partition("standard").add(standard);
  graph.partition("robotics").add(robotics);

  CompileOptions options;
  options.lifecycle_transactions = ample_lifecycle_capacity();
  const ExecutionPlan plan = compile(graph, options);
  ASSERT_TRUE(plan.ok());
  ASSERT_EQ(plan.lifecycle_restart_targets().size(), 2U);
  // Key assertion: exactly one target with kStandard and one with kRobotics;
  //               both are admitted because kRobotics supports restart.
  EXPECT_EQ(std::count_if(
                plan.lifecycle_restart_targets().begin(),
                plan.lifecycle_restart_targets().end(),
                [](const auto& target) { return target.effective_profile == Profile::kStandard; }),
            1);
  EXPECT_EQ(std::count_if(
                plan.lifecycle_restart_targets().begin(),
                plan.lifecycle_restart_targets().end(),
                [](const auto& target) { return target.effective_profile == Profile::kRobotics; }),
            1);
  EXPECT_TRUE(std::all_of(plan.lifecycle_restart_targets().begin(),
                          plan.lifecycle_restart_targets().end(),
                          [](const auto& target) {
                            return target.disposition ==
                                   LifecycleRestartTargetDisposition::kAdmitted;
                          }));
}

// Two self-restart targets; the real-time operator's region profile does not
// support lifecycle transactions, so its disposition must be kRejected while
// the standard operator remains kAdmitted.
//
//   standard (allow_self_restart, region: kStandard)
//   real-time (allow_self_restart, region: kRealTime — restart unsupported)
//
// REQ: COMPILE-003
// REQ: COMPILE-008
TEST(CompileLifecycle, RejectsRequestedTargetInUnsupportedEffectiveRegionProfile) {
  Graph graph{"lifecycle-unsupported-effective-profile"};
  const auto standard = graph.op<test::CompileTestOperator>("standard");
  const auto real_time = graph.op<test::CompileTestOperator>("real-time");
  graph.lifecycle().allow_self_restart(standard).allow_self_restart(real_time);
  graph.region("real-time-region").add(real_time).profile(Profile::kRealTime);

  CompileOptions options;
  options.lifecycle_transactions = ample_lifecycle_capacity();
  // Key setup: kRealTime profile is present in the restart targets; the compile
  //            must reject while preserving evidence for both targets.
  const ExecutionPlan plan = compile(graph, options);
  ASSERT_FALSE(plan.ok());
  EXPECT_TRUE(test::has_diagnostic(plan, "LIFECYCLE_TRANSACTION_UNSUPPORTED_PROFILE"));
  const auto profile_diagnostic =
      std::find_if(plan.diagnostics().begin(), plan.diagnostics().end(), [](const auto& item) {
        return item.token == "LIFECYCLE_TRANSACTION_UNSUPPORTED_PROFILE" &&
               item.provided == "real_time";
      });
  ASSERT_NE(profile_diagnostic, plan.diagnostics().end());
  EXPECT_TRUE(profile_diagnostic->source.vertex);
  EXPECT_EQ(profile_diagnostic->json_pointer, "/lifecycle_transactions/profile");
  EXPECT_EQ(profile_diagnostic->required, "active_restart_admitted");
  // Key assertion: kRealTime target is kRejected; kStandard target is kAdmitted.
  const auto rejected = std::find_if(
      plan.lifecycle_restart_targets().begin(),
      plan.lifecycle_restart_targets().end(),
      [](const auto& target) { return target.effective_profile == Profile::kRealTime; });
  ASSERT_NE(rejected, plan.lifecycle_restart_targets().end());
  EXPECT_EQ(rejected->disposition, LifecycleRestartTargetDisposition::kRejected);
  EXPECT_FALSE(rejected->self_allowed);
  const auto admitted = std::find_if(
      plan.lifecycle_restart_targets().begin(),
      plan.lifecycle_restart_targets().end(),
      [](const auto& target) { return target.effective_profile == Profile::kStandard; });
  ASSERT_NE(admitted, plan.lifecycle_restart_targets().end());
  EXPECT_EQ(admitted->disposition, LifecycleRestartTargetDisposition::kAdmitted);
  EXPECT_TRUE(admitted->self_allowed);
}

}  // namespace
}  // namespace holoscan
// NOLINTEND(bugprone-unchecked-optional-access)
