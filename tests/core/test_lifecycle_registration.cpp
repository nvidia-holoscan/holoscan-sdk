/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <map>
#include <string>
#include <string_view>

#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/parameter_schema.hpp>

#include <core/graph_compiler_access.hpp>
#include <core/lifecycle_context_internal.hpp>
#include <core/lifecycle_context_test_access.hpp>
#include <core/operator_spec_test_access.hpp>

namespace holoscan {
namespace {

inline constexpr LifecycleDependencyKey kPoolReady{"test.pool.ready"};

// Canonical operator that exercises the full spec.lifecycle() DSL:
//   - required kAllocate callback (method index 3 in the virtual table)
//   - optional kWarmUp callback (may return kSkipped without error)
//   - produces kPoolReady fact at kAllocate
//   - requires a named dependency at kArm
// Representative test: CallbacksAreOwnedAndEveryMissingStageIsSkipped
class LifecycleRegistrationOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.lifecycle()
        .stage(LifecycleStage::kAllocate, &LifecycleRegistrationOperator::allocate)
        .optional(LifecycleStage::kWarmUp, &LifecycleRegistrationOperator::warm_up)
        .produces(LifecycleStage::kAllocate, kPoolReady)
        .requires_dependency(LifecycleStage::kArm, LifecycleDependencyKey{dependency_name_});
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  LifecycleStatus allocate(LifecycleContext&) noexcept {
    ++calls_;
    return LifecycleStatus::kOk;
  }

  LifecycleStatus warm_up(LifecycleContext&) noexcept {
    ++calls_;
    return LifecycleStatus::kSkipped;
  }

  std::uint32_t calls() const noexcept { return calls_; }

 private:
  std::string dependency_name_{"test.arm.ready"};
  std::uint32_t calls_{};
};

// Operator that registers kArm both as required and optional in the same setup call.
// Used by RejectsDuplicateOrInvalidCallbacks to confirm that the spec deduplicates
// to one entry but Graph::op<> still throws RuntimeError for the duplicated stage.
class DuplicateLifecycleOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.lifecycle()
        .stage(LifecycleStage::kArm, &DuplicateLifecycleOperator::arm)
        .optional(LifecycleStage::kArm, &DuplicateLifecycleOperator::arm);
  }
  expected<void, Error> compute(ExecutionContext&) override { return {}; }
  LifecycleStatus arm(LifecycleContext&) noexcept { return LifecycleStatus::kOk; }
};

// Operator that produces a lifecycle fact with an empty LifecycleDependencyKey.
// Used by RejectsDuplicateOrInvalidCallbacks (empty key must be rejected at graph time).
class InvalidLifecycleFactOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.lifecycle().produces(LifecycleStage::kAllocate, LifecycleDependencyKey{});
  }
  expected<void, Error> compute(ExecutionContext&) override { return {}; }
};

// Operators that use test-access helpers to inject out-of-range enum bytes into the
// spec after a valid setup call.  They verify that Graph::op<> validates enum values
// and rejects them with GRAPH_LIFECYCLE_STAGE_INVALID or GRAPH_LIFECYCLE_FACT_KIND_INVALID.
class InvalidLifecycleCallbackStageOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.lifecycle().stage(LifecycleStage::kConfigure,
                           &InvalidLifecycleCallbackStageOperator::configure);
    detail::OperatorSpecTestAccess::set_lifecycle_callback_stage(spec, 0U, 0xFFU);
  }
  expected<void, Error> compute(ExecutionContext&) override { return {}; }
  LifecycleStatus configure(LifecycleContext&) noexcept { return LifecycleStatus::kOk; }
};

class InvalidLifecycleFactStageOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.lifecycle().produces(LifecycleStage::kConfigure, kPoolReady);
    detail::OperatorSpecTestAccess::set_lifecycle_fact_stage(spec, 0U, 0xFFU);
  }
  expected<void, Error> compute(ExecutionContext&) override { return {}; }
};

class InvalidLifecycleFactKindOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.lifecycle().produces(LifecycleStage::kConfigure, kPoolReady);
    detail::OperatorSpecTestAccess::set_lifecycle_fact_kind(spec, 0U, 0xFFU);
  }
  expected<void, Error> compute(ExecutionContext&) override { return {}; }
};

// Template helper: constructs a Graph, inserts one operator, and asserts that a
// RuntimeError is thrown containing the given diagnostic token string.
// Reduces repetition in RejectsFabricatedLifecycleEnumsBeforeRegistration.
template <typename Op>
void expect_fabricated_lifecycle_metadata_rejected(std::string_view name, std::string_view token) {
  Graph graph;
  try {
    (void)graph.op<Op>(name);
    FAIL() << "fabricated lifecycle metadata was accepted";
  } catch (const RuntimeError& error) {
    EXPECT_EQ(error.code(), ErrorCode::kInvalidArgument);
    EXPECT_NE(std::string_view{error.what()}.find(token), std::string_view::npos);
  }
}

// Operators used by LegacyOverridesLowerAndConflictsReject to verify the legacy
// start()/stop() detection and conflict checking logic:
//   LegacyLifecycleOperator          — both start() and stop() overridden
//   NoexceptLegacyLifecycleOperator   — noexcept qualifiers are transparent to detection
//   OverloadedLegacyLifecycleOperator — overloaded start(int) must not block detection
//   IntermediateLifecycleOperator /
//   InheritedLifecycleOperator        — inherited override is still detected
//   NoLegacyLifecycleOperator         — baseline: no phantom callbacks registered
//   AdvancedStartOperator             — uses the advanced spec.lifecycle() path for kStart
//   ConflictingStartOperator          — both legacy start() and advanced kStart → kDuplicateName
class LegacyLifecycleOperator final : public Operator<> {
 public:
  void start() override { ++start_calls; }
  void stop() override { ++stop_calls; }
  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  std::uint32_t start_calls{};
  std::uint32_t stop_calls{};
};

class NoexceptLegacyLifecycleOperator final : public Operator<> {
 public:
  void start() noexcept override {}
  void stop() noexcept override {}
  expected<void, Error> compute(ExecutionContext&) override { return {}; }
};

class OverloadedLegacyLifecycleOperator final : public Operator<> {
 public:
  void start() override {}
  void start(int) {}
  expected<void, Error> compute(ExecutionContext&) override { return {}; }
};

class IntermediateLifecycleOperator : public Operator<> {
 public:
  void start() override {}
};

class InheritedLifecycleOperator final : public IntermediateLifecycleOperator {
 public:
  expected<void, Error> compute(ExecutionContext&) override { return {}; }
};

class NoLegacyLifecycleOperator final : public Operator<> {
 public:
  expected<void, Error> compute(ExecutionContext&) override { return {}; }
};

class AdvancedStartOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.lifecycle().stage(LifecycleStage::kStart, &AdvancedStartOperator::on_start);
  }
  expected<void, Error> compute(ExecutionContext&) override { return {}; }
  LifecycleStatus on_start(LifecycleContext&) noexcept { return LifecycleStatus::kOk; }
};

class ConflictingStartOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.lifecycle().stage(LifecycleStage::kStart, &ConflictingStartOperator::on_start);
  }
  void start() override {}
  expected<void, Error> compute(ExecutionContext&) override { return {}; }
  LifecycleStatus on_start(LifecycleContext&) noexcept { return LifecycleStatus::kOk; }
};

struct LifecycleOwnerParams {
  bool use_base_callback{};
};

HOLOSCAN_PARAM_SCHEMA(
    LifecycleOwnerParams, holoscan::param_schema("nvidia.holoscan.test.LifecycleOwnerParams", 1),
    holoscan::field<&LifecycleOwnerParams::use_base_callback>("use_base_callback"));

// Base class that holds a lifecycle callback; used to test that a derived operator
// may not pass a base-class method pointer as its own lifecycle callback.
// The graph compiler enforces that the callback owner type matches the operator type.
class LifecycleOwnerBase : public Operator<LifecycleOwnerParams> {
 public:
  LifecycleStatus base_callback(LifecycleContext&) noexcept { return LifecycleStatus::kOk; }
  expected<void, Error> compute(ExecutionContext&) override { return {}; }
};

// Attempts to register a base-class method pointer during initial setup; expects
// GRAPH_LIFECYCLE_CALLBACK_OWNER_MISMATCH at graph.op<>() time.
class InitialOwnerMismatchOperator final : public LifecycleOwnerBase {
 public:
  void setup(OperatorSpec& spec) override {
    spec.lifecycle().stage(LifecycleStage::kConfigure, &LifecycleOwnerBase::base_callback);
  }
};

// Passes correct callback initially but switches to a base-class pointer on parameter
// replay; expects GRAPH_LIFECYCLE_CALLBACK_OWNER_MISMATCH during op.set_params().
class ReplayOwnerMismatchOperator final : public LifecycleOwnerBase {
 public:
  void setup(OperatorSpec& spec) override {
    if (params().use_base_callback) {
      spec.lifecycle().stage(LifecycleStage::kConfigure, &LifecycleOwnerBase::base_callback);
    } else {
      spec.lifecycle().stage(LifecycleStage::kConfigure,
                             &ReplayOwnerMismatchOperator::exact_callback);
    }
  }

  LifecycleStatus exact_callback(LifecycleContext&) noexcept { return LifecycleStatus::kOk; }
};

template <typename View>
concept ExposesPrivateLifecycleRecords = requires(const View& view) {
  view.lifecycle_callbacks();
  view.lifecycle_facts();
};

static_assert(!ExposesPrivateLifecycleRecords<GraphView>);

// REQ: LIFECYCLE-002
TEST(LifecycleRegistration, CallbacksAreOwnedAndEveryMissingStageIsSkipped) {
  // Key setup: verify the spec-level metadata first (callback count, required flags,
  // fact names), then confirm that the callback dispatches to the receiver object,
  // not the authoring declaration object.
  LifecycleRegistrationOperator declaration;
  OperatorSpec spec;
  declaration.setup(spec);

  ASSERT_EQ(detail::OperatorSpecTestAccess::lifecycle_callback_count(spec), 2U);
  const auto allocate = detail::OperatorSpecTestAccess::lifecycle_callback(spec, 0U);
  EXPECT_EQ(allocate.stage, LifecycleStage::kAllocate);
  EXPECT_TRUE(allocate.required);
  EXPECT_EQ(allocate.method, 3U);
  const auto warm_up = detail::OperatorSpecTestAccess::lifecycle_callback(spec, 1U);
  EXPECT_FALSE(warm_up.required);

  ASSERT_EQ(detail::OperatorSpecTestAccess::lifecycle_fact_count(spec), 2U);
  EXPECT_EQ(detail::OperatorSpecTestAccess::lifecycle_fact(spec, 0U).stable_name,
            "test.pool.ready");
  EXPECT_EQ(detail::OperatorSpecTestAccess::lifecycle_fact(spec, 1U).stable_name, "test.arm.ready");

  LifecycleRegistrationOperator fresh_receiver;
  detail::LifecycleContextState state{.phase = LifecycleTransitionPhase::kStartupForward};
  LifecycleContext context = detail::LifecycleContextTestAccess::context(state);
  EXPECT_EQ(
      detail::OperatorSpecTestAccess::invoke_lifecycle_callback(spec, 0U, &fresh_receiver, context),
      LifecycleStatus::kOk);
  EXPECT_EQ(fresh_receiver.calls(), 1U);
  EXPECT_EQ(declaration.calls(), 0U);

  // Topology: single operator, no edges
  //   [source: LifecycleRegistrationOperator]
  //
  // Key assertion: canonical_snapshot must carry both callbacks and both facts
  // exactly as declared in setup(), including the dependency fact name.
  Graph graph;
  (void)graph.op<LifecycleRegistrationOperator>("source");
  const auto snapshot = detail::GraphCompilerAccess::canonical_snapshot(graph);
  ASSERT_TRUE(snapshot);
  EXPECT_EQ(snapshot->lifecycle_callbacks.size(), 2U);
  EXPECT_EQ(snapshot->lifecycle_facts.size(), 2U);
  EXPECT_EQ(snapshot->lifecycle_facts[1].stable_name, "test.arm.ready");
}

// REQ: LIFECYCLE-002
TEST(LifecycleRegistration, RejectsDuplicateOrInvalidCallbacks) {
  DuplicateLifecycleOperator duplicate;
  OperatorSpec spec;
  duplicate.setup(spec);
  EXPECT_EQ(detail::OperatorSpecTestAccess::lifecycle_callback_count(spec), 1U);
  Graph graph;
  EXPECT_THROW((void)graph.op<DuplicateLifecycleOperator>("duplicate"), RuntimeError);
  EXPECT_THROW((void)graph.op<InvalidLifecycleFactOperator>("invalid-fact"), RuntimeError);
}

// REQ: LIFECYCLE-002
TEST(LifecycleRegistration, RejectsFabricatedLifecycleEnumsBeforeRegistration) {
  expect_fabricated_lifecycle_metadata_rejected<InvalidLifecycleCallbackStageOperator>(
      "invalid-callback-stage", "GRAPH_LIFECYCLE_STAGE_INVALID");
  expect_fabricated_lifecycle_metadata_rejected<InvalidLifecycleFactStageOperator>(
      "invalid-fact-stage", "GRAPH_LIFECYCLE_STAGE_INVALID");
  expect_fabricated_lifecycle_metadata_rejected<InvalidLifecycleFactKindOperator>(
      "invalid-fact-kind", "GRAPH_LIFECYCLE_FACT_KIND_INVALID");
}

// REQ: LIFECYCLE-002
TEST(LifecycleRegistration, RejectsCallbackOwnerMismatchAtRegistrationAndParameterReplay) {
  // Key setup: two separate Graph instances exercise the two mismatch paths:
  //   initial — mismatch caught at graph.op<>() during first registration
  //   replay  — mismatch caught at op.set_params() during parameter replay
  // Key assertion: both must throw RuntimeError with kIncompatible and the diagnostic
  // token; the graph generation must not advance on the replay failure.
  Graph initial;
  try {
    (void)initial.op<InitialOwnerMismatchOperator>("initial-mismatch");
    FAIL() << "base-class lifecycle callback was accepted during initial registration";
  } catch (const RuntimeError& error) {
    EXPECT_EQ(error.code(), ErrorCode::kIncompatible);
    EXPECT_NE(std::string_view{error.what()}.find("GRAPH_LIFECYCLE_CALLBACK_OWNER_MISMATCH"),
              std::string_view::npos);
  }

  Graph replay;
  auto op = replay.op<ReplayOwnerMismatchOperator>("replay-mismatch");
  const GraphView before = replay.view();
  try {
    op.set_params(LifecycleOwnerParams{.use_base_callback = true});
    FAIL() << "base-class lifecycle callback was accepted during parameter replay";
  } catch (const RuntimeError& error) {
    EXPECT_EQ(error.code(), ErrorCode::kIncompatible);
    EXPECT_NE(std::string_view{error.what()}.find("GRAPH_LIFECYCLE_CALLBACK_OWNER_MISMATCH"),
              std::string_view::npos);
  }
  EXPECT_EQ(replay.view().generation(), before.generation());
  EXPECT_FALSE(op->params().use_base_callback);
}

// REQ: LIFECYCLE-002
TEST(LifecycleRegistration, LegacyOverridesLowerAndConflictsReject) {
  // Topology: 6 independent operators, no edges
  //   [legacy]  [noexcept]  [overloaded]  [inherited]  [base-noop]  [advanced]
  //
  // Key setup: each operator tests a different detection path for legacy start/stop
  // callbacks and the advanced spec.lifecycle() path.
  Graph graph;
  (void)graph.op<LegacyLifecycleOperator>("legacy");
  (void)graph.op<NoexceptLegacyLifecycleOperator>("noexcept");
  (void)graph.op<OverloadedLegacyLifecycleOperator>("overloaded");
  (void)graph.op<InheritedLifecycleOperator>("inherited");
  (void)graph.op<NoLegacyLifecycleOperator>("base-noop");
  (void)graph.op<AdvancedStartOperator>("advanced");

  // Key assertion: legacy kLegacyStart/kLegacyStop callbacks appear for operators that
  // override start()/stop(); kAdvancedMember appears for AdvancedStartOperator; no
  // phantom callbacks appear for NoLegacyLifecycleOperator.
  const auto snapshot = detail::GraphCompilerAccess::canonical_snapshot(graph);
  ASSERT_TRUE(snapshot);
  std::map<std::string, VertexId> vertices;
  for (const auto& vertex : snapshot->vertices) {
    if (vertex.kind == VertexKind::kOperator)
      vertices.emplace(vertex.name, vertex.id);
  }
  const auto count = [&](std::string_view name, detail::LifecycleCallbackOrigin origin) {
    return std::count_if(snapshot->lifecycle_callbacks.begin(),
                         snapshot->lifecycle_callbacks.end(),
                         [&](const auto& callback) {
                           return callback.owner == vertices.at(std::string(name)) &&
                                  callback.origin == origin;
                         });
  };
  EXPECT_EQ(count("legacy", detail::LifecycleCallbackOrigin::kLegacyStart), 1);
  EXPECT_EQ(count("legacy", detail::LifecycleCallbackOrigin::kLegacyStop), 1);
  EXPECT_EQ(count("noexcept", detail::LifecycleCallbackOrigin::kLegacyStart), 1);
  EXPECT_EQ(count("noexcept", detail::LifecycleCallbackOrigin::kLegacyStop), 1);
  EXPECT_EQ(count("overloaded", detail::LifecycleCallbackOrigin::kLegacyStart), 1);
  EXPECT_EQ(count("inherited", detail::LifecycleCallbackOrigin::kLegacyStart), 1);
  EXPECT_EQ(count("base-noop", detail::LifecycleCallbackOrigin::kLegacyStart), 0);
  EXPECT_EQ(count("base-noop", detail::LifecycleCallbackOrigin::kLegacyStop), 0);
  EXPECT_EQ(count("advanced", detail::LifecycleCallbackOrigin::kAdvancedMember), 1);

  Graph conflict;
  try {
    (void)conflict.op<ConflictingStartOperator>("conflict");
    FAIL() << "legacy/advanced stage conflict was accepted";
  } catch (const RuntimeError& error) {
    EXPECT_EQ(error.code(), ErrorCode::kDuplicateName);
    EXPECT_NE(std::string_view{error.what()}.find("GRAPH_LIFECYCLE_CALLBACK_DUPLICATE"),
              std::string_view::npos);
  }
}

}  // namespace
}  // namespace holoscan
