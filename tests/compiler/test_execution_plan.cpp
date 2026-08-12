/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <utility>

#include <holoscan/time/manual_clock.hpp>

#include "compile_test_graph.hpp"
#include "compiler/artifact_emitter_test_access.hpp"
#include "compiler/execution_plan_state.hpp"
#include "core/parameter_override_set_internal.hpp"

// GTest fatal assertions establish optional preconditions that clang-tidy does not model.
// NOLINTBEGIN(bugprone-unchecked-optional-access)
namespace holoscan {

struct LifecycleCapacityParams {
  std::uint32_t value{};
};

HOLOSCAN_PARAM_SCHEMA(LifecycleCapacityParams,
                      holoscan::param_schema("nvidia.holoscan.test.LifecycleCapacityParams", 1),
                      holoscan::field<&LifecycleCapacityParams::value>("value").lifecycle_mutable(
                          LifecycleStage::kConfigure));

namespace {

// Minimal operator with a lifecycle-mutable uint32 "value" parameter.
// Drives parameter-capacity validation and dependency-expansion tests
// (e.g. CrossValidatesAllLifecycleResourcesForRepeatedParameterSubmissions,
// ExpandsParameterCapacityAcrossNonMutableLifecycleDependencies).
class LifecycleCapacityOperator final : public Operator<LifecycleCapacityParams> {
 public:
  expected<void, Error> compute(ExecutionContext&) override { return {}; }
};

// Reusable Graph that declares one exact clock dependency and uses it as the default clock.
// ExecutionPlan public-view tests use this to prove that compile preserves the authoring records.
class ExecutionClockImportGraph final : public Graph {
 public:
  [[nodiscard]] ClockKey clock_import() const noexcept { return clock_import_; }

 protected:
  void compose() override {
    clock_import_ = import_clock("timeline");
    set_default_clock(clock_import_);
    (void)op<test::CompileTestOperator>("worker");
  }

 private:
  ClockKey clock_import_;
};

// Minimal embedded Graph used to prove that EA1 manual partitions remain a
// flat-root-only authoring surface.
class ManualPartitionEmbeddedGraph final : public Graph {
 protected:
  void compose() override { (void)op<test::CompileTestOperator>("nested"); }
};

enum class InputTriggerMode : std::uint8_t {
  kEach,
  kAnyInput,
};

// Ordinary input-driven successors are not independent execution roots. This
// explicit-contract fixture keeps that distinction visible when partition
// admission classifies temporal trigger kinds.
class InputTriggeredCompileOperator final : public Operator<> {
 public:
  explicit InputTriggeredCompileOperator(InputTriggerMode mode) : mode_(mode) {}

  void setup(OperatorSpec& spec) override {
    spec.input(input, "input").queue_depth(2U);
    spec.output(output, "output").max_emits_per_compute(1U);
  }

  Contract contract() const override {
    Contract result;
    if (mode_ == InputTriggerMode::kEach) {
      result.trigger(OnEach{input});
    } else {
      result.trigger(OnAnyInput{});
    }
    return result;
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Input<std::uint32_t> input;
  Output<std::uint32_t> output;

 private:
  InputTriggerMode mode_;
};

// Returns lifecycle transaction options with ample capacity for all resource dimensions.
// Tests reduce one field below the computed minimum to trigger capacity rejection.
[[nodiscard]] LifecycleTransactionOptions ample_parameter_lifecycle_capacity() {
  return LifecycleTransactionOptions{
      .max_pending_requests = 64U,
      .max_pending_transactions = 64U,
      .max_receipts = 128U,
      .max_fenced_operators = 64U,
      .max_candidate_generations = 64U,
      .max_events = 1024U,
      .max_audit_records = 128U,
      .max_quiesce_records = 64U,
      .max_rollback_records = 64U,
  };
}

// RAII wrapper that creates a unique temporary artifact root directory per test run.
// The guard directory and its contents are removed on destruction; every
// ExecutionPlanArtifacts test uses this to isolate its filesystem writes.
class ScopedArtifactRoot {
 public:
  ScopedArtifactRoot() {
    static std::atomic<std::uint64_t> next{1U};
    const auto* test = ::testing::UnitTest::GetInstance()->current_test_info();
    const std::string test_identity =
        test != nullptr ? std::string(test->test_suite_name()) + "-" + test->name() : "unknown";
    const auto tick = std::chrono::steady_clock::now().time_since_epoch().count();
    for (std::size_t attempt = 0U; attempt < 128U; ++attempt) {
      std::filesystem::path candidate_guard =
          std::filesystem::temp_directory_path() /
          ("holoscan-ea1-artifacts-" + test_identity + "-" + std::to_string(tick) + "-" +
           std::to_string(next.fetch_add(1U, std::memory_order_relaxed)));
      std::filesystem::path candidate_root = candidate_guard / "root";
      std::error_code error;
      if (std::filesystem::create_directory(candidate_guard, error)) {
        guard_.swap(candidate_guard);
        root_.swap(candidate_root);
        return;
      }
      if (error) {
        throw std::filesystem::filesystem_error{
            "create artifact test root", candidate_guard, error};
      }
    }
    throw std::runtime_error{"unable to reserve a unique artifact test root"};
  }

  ~ScopedArtifactRoot() {
    std::error_code error;
    std::filesystem::remove_all(guard_, error);
  }

  ScopedArtifactRoot(const ScopedArtifactRoot&) = delete;
  ScopedArtifactRoot& operator=(const ScopedArtifactRoot&) = delete;

  [[nodiscard]] const std::filesystem::path& path() const noexcept { return root_; }

 private:
  std::filesystem::path guard_;
  std::filesystem::path root_;
};

// Factory that returns a ScopedArtifactRoot tied to the current GTest test case.
[[nodiscard]] ScopedArtifactRoot unique_artifact_root() {
  return {};
}

// Converts a PlanDigest byte array to a lowercase hex string, used to construct
// the expected on-disk directory path in symlink-containment tests.
[[nodiscard]] std::string digest_hex(const PlanDigest& digest) {
  constexpr std::string_view kHex{"0123456789abcdef"};
  std::string result;
  result.reserve(digest.value.size() * 2U);
  for (const std::byte byte : digest.value) {
    const auto value = std::to_integer<std::uint8_t>(byte);
    result.push_back(kHex[value >> 4U]);
    result.push_back(kHex[value & 0x0FU]);
  }
  return result;
}

// Reads an artifact file entirely into a std::string for in-test content comparison.
[[nodiscard]] std::string read_artifact(const std::filesystem::path& path) {
  std::ifstream stream(path, std::ios::binary);
  return {std::istreambuf_iterator<char>{stream}, std::istreambuf_iterator<char>{}};
}

// Appends one '!' byte to a file, placing it at exactly expected_size+1.
// Used to trigger the oversized-destination repair path in artifact write tests.
void append_artifact_sentinel(const std::filesystem::path& path) {
  std::ofstream stream(path, std::ios::binary | std::ios::app);
  ASSERT_TRUE(stream);
  stream.put('!');
  stream.close();
  ASSERT_FALSE(stream.fail());
}

// Compile-time guards: Graph intentionally omits compile() and run() members, and
// ExecutionGraphView is a distinct type from the lower-level GraphView.
template <typename T>
concept HasCompileMember = requires(T& value) { value.compile(); };

template <typename T>
concept HasRunMember = requires(T& value) { value.run(); };

static_assert(!HasCompileMember<Graph>);
static_assert(!HasRunMember<Graph>);
static_assert(!std::is_same_v<ExecutionGraphView, GraphView>);

//   source --> sink  (via test::add_linear_graph; all plan inspection surfaces populated)
// REQ: COMPILE-001
TEST(CompileEntryPoint, GraphOnlyCompileReturnsTotalInspectablePlan) {
  Graph graph{"compile-entry"};
  test::add_linear_graph(graph);

  const ExecutionPlan plan = compile(graph);
  EXPECT_TRUE(plan.ok());
  EXPECT_NO_THROW(plan.require_no_errors());
  EXPECT_FALSE(plan.json().empty());
  EXPECT_EQ(plan.generation(), plan.graph_identity().version);
  EXPECT_EQ(plan.graph().vertices().size(), 2U);
  EXPECT_TRUE(test::has_nonzero_byte(plan.graph_hash().value));
  EXPECT_TRUE(test::has_nonzero_byte(plan.digest().value));
  EXPECT_FALSE(plan.execution_partitions().empty());
}

//   source --> left
//          \-> right
// One compiler-proven root remains one implicit serial execution partition.
// REQ: COMPILE-058
TEST(CompilePartitionAuthoring, OneRootForkUsesOneImplicitPartition) {
  Graph graph{"implicit-one-root-fork"};
  test::add_compile_test_default_clock(graph);
  const auto source = graph.op<test::CompileTestOperator>("source");
  const auto left = graph.op<test::CompileTestOperator>("left");
  const auto right = graph.op<test::CompileTestOperator>("right");
  graph.add_flow(source->output, left->input);
  graph.add_flow(source->output, right->input);

  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.execution_partitions().size(), 1U);
  EXPECT_FALSE(plan.execution_partitions().front().authored_name.has_value());
  EXPECT_EQ(plan.execution_partitions().front().members.size(), 3U);
  EXPECT_EQ(plan.partitioning_plan().selection_source, PlanSelectionSource::kCoreDefault);
  ASSERT_TRUE(plan.partitioning_plan().accepted_algorithm.has_value());
  EXPECT_EQ(plan.partitioning_plan().accepted_algorithm->id, "holoscan.partition.serial_baseline");
  EXPECT_EQ(plan.scheduler_plan().execution_lane_count, 1U);
  EXPECT_EQ(plan.scheduler_plan().max_concurrent_activations, 1U);
  EXPECT_FALSE(plan.scheduler_plan().activation_overlap);
  EXPECT_EQ(plan.scheduler_plan().worker_count, 1U);
  EXPECT_EQ(plan.scheduler_plan().intra_path_parallelism, 1U);

  const auto* const state = detail::ExecutionPlanAccess::state(plan);
  ASSERT_NE(state, nullptr);
  ASSERT_EQ(state->partition_runners.size(), 1U);
  const auto& runner = state->partition_runners.front();
  EXPECT_EQ(runner.ordinal, 0U);
  EXPECT_EQ(runner.partition, plan.execution_partitions().front().id);
  EXPECT_EQ(runner.execution_lane, plan.execution_partitions().front().execution_lane);
  ASSERT_EQ(runner.operators.size(), 3U);
  EXPECT_EQ(runner.operators[0].vertex, VertexId{3U});
  EXPECT_EQ(runner.operators[1].vertex, VertexId{1U});
  EXPECT_EQ(runner.operators[2].vertex, VertexId{2U});
  EXPECT_TRUE(runner.operators[0].topology_root);
  EXPECT_FALSE(runner.operators[1].topology_root);
  EXPECT_FALSE(runner.operators[2].topology_root);
  EXPECT_EQ(runner.internal_edges.size(), 2U);
  EXPECT_TRUE(runner.incoming_boundary_edges.empty());
  EXPECT_TRUE(runner.outgoing_boundary_edges.empty());
}

//   source --> each
//          \-> any-input
// Explicit ordinary input triggers remain successors of one root rather than
// manufacturing additional independent temporal roots.
// REQ: COMPILE-058
TEST(CompilePartitionAuthoring, OrdinaryInputTriggersRemainInOneImplicitPartition) {
  Graph graph{"implicit-ordinary-input-triggers"};
  test::add_compile_test_default_clock(graph);
  const auto source = graph.op<test::CompileTestOperator>("source");
  const auto each = graph.op<InputTriggeredCompileOperator>("each", InputTriggerMode::kEach);
  const auto any =
      graph.op<InputTriggeredCompileOperator>("any-input", InputTriggerMode::kAnyInput);
  graph.add_flow(source->output, each->input);
  graph.add_flow(source->output, any->input);

  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.execution_partitions().size(), 1U);
  EXPECT_EQ(plan.execution_partitions().front().members.size(), 3U);

  const auto* const state = detail::ExecutionPlanAccess::state(plan);
  ASSERT_NE(state, nullptr);
  ASSERT_EQ(state->partition_runners.size(), 1U);
  const auto& operators = state->partition_runners.front().operators;
  ASSERT_EQ(operators.size(), 3U);
  EXPECT_EQ(std::count_if(operators.begin(),
                          operators.end(),
                          [](const auto& member) { return member.topology_root; }),
            1);
}

//   camera-source      imu-source
// Two independent roots require an explicit manual cover.
TEST(CompilePartitionAuthoring, UnpartitionedMultiRootRequiresManualAuthoring) {
  Graph graph{"implicit-multi-root-rejected"};
  test::add_compile_test_default_clock(graph);
  (void)graph.op<test::CompileTestOperator>("camera-source");
  (void)graph.op<test::CompileTestOperator>("imu-source");

  const ExecutionPlan plan = compile(graph);
  EXPECT_FALSE(plan.ok());
  EXPECT_TRUE(test::has_diagnostic(plan, "PARTITION_MANUAL_AUTHORING_REQUIRED"));
  EXPECT_FALSE(plan.partitioning_plan().accepted_algorithm.has_value());
}

//   camera-source      imu-source
//       [camera]          [imu]
// REQ: COMPILE-058
TEST(CompilePartitionAuthoring, ExactManualCoverLowersNamedPartitions) {
  Graph graph{"manual-two-root"};
  test::add_compile_test_default_clock(graph);
  const auto camera = graph.op<test::CompileTestOperator>("camera-source");
  const auto camera_sink = graph.op<test::CompileTestOperator>("camera-sink");
  const auto imu = graph.op<test::CompileTestOperator>("imu-source");
  const auto imu_sink = graph.op<test::CompileTestOperator>("imu-sink");
  graph.add_flow(camera->output, camera_sink->input);
  graph.add_flow(imu->output, imu_sink->input);
  graph.partition("camera").add(camera).add(camera_sink);
  graph.partition("imu").add(imu).add(imu_sink);

  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.execution_partitions().size(), 2U);
  ASSERT_TRUE(plan.execution_partitions()[0].authored_name.has_value());
  ASSERT_TRUE(plan.execution_partitions()[1].authored_name.has_value());
  EXPECT_EQ(*plan.execution_partitions()[0].authored_name, "camera");
  EXPECT_EQ(*plan.execution_partitions()[1].authored_name, "imu");
  EXPECT_EQ(plan.execution_partitions()[0].members.size(), 2U);
  EXPECT_EQ(plan.execution_partitions()[1].members.size(), 2U);
  EXPECT_EQ(plan.partitioning_plan().selection_source, PlanSelectionSource::kCoreDefault);
  ASSERT_TRUE(plan.partitioning_plan().accepted_algorithm.has_value());
  EXPECT_EQ(plan.partitioning_plan().accepted_algorithm->id, "holoscan.partition.serial_baseline");
}

// REQ: COMPILE-058
TEST(CompilePartitionAuthoring, RejectsEmptyPartialAndMultiRootPartitionCovers) {
  {
    Graph graph{"manual-empty"};
    test::add_compile_test_default_clock(graph);
    (void)graph.op<test::CompileTestOperator>("source");
    (void)graph.partition("empty");
    const ExecutionPlan plan = compile(graph);
    EXPECT_FALSE(plan.ok());
    EXPECT_TRUE(test::has_diagnostic(plan, "PARTITION_COVER_INVALID"));
  }
  {
    Graph graph{"manual-partial"};
    test::add_compile_test_default_clock(graph);
    const auto source = graph.op<test::CompileTestOperator>("source");
    (void)graph.op<test::CompileTestOperator>("uncovered");
    graph.partition("only-source").add(source);
    const ExecutionPlan plan = compile(graph);
    EXPECT_FALSE(plan.ok());
    EXPECT_TRUE(test::has_diagnostic(plan, "PARTITION_COVER_INVALID"));
  }
  {
    Graph graph{"manual-two-roots-one-partition"};
    test::add_compile_test_default_clock(graph);
    const auto first = graph.op<test::CompileTestOperator>("first");
    const auto second = graph.op<test::CompileTestOperator>("second");
    graph.partition("invalid").add(first).add(second);
    const ExecutionPlan plan = compile(graph);
    EXPECT_FALSE(plan.ok());
    EXPECT_TRUE(test::has_diagnostic(plan, "PARTITION_COVER_INVALID"));
  }
}

// REQ: COMPILE-058
TEST(CompilePartitionAuthoring, RejectsManualPartitionsWithEmbeddedGraphs) {
  Graph graph{"manual-embedded"};
  test::add_compile_test_default_clock(graph);
  const auto root = graph.op<test::CompileTestOperator>("root");
  graph.partition("root").add(root);
  (void)graph.embed<ManualPartitionEmbeddedGraph>("child");

  const ExecutionPlan plan = compile(graph);
  EXPECT_FALSE(plan.ok());
  EXPECT_TRUE(test::has_diagnostic(plan, "PARTITION_MANUAL_EMBEDDED_GRAPH_UNSUPPORTED"));
}

TEST(CompilePartitionAuthoring, AttachesOnlyUnambiguousSyntheticBoundaryVertices) {
  {
    Graph graph{"manual-import"};
    test::add_compile_test_default_clock(graph);
    auto imported = graph.import_topic<std::uint32_t>("holomq://manual-import");
    const auto sink = graph.op<test::CompileTestOperator>("sink");
    graph.add_flow(imported.output(), sink->input);
    graph.partition("sink").add(sink);

    const ExecutionPlan plan = compile(graph);
    ASSERT_TRUE(plan.ok()) << plan.json();
    ASSERT_EQ(plan.execution_partitions().size(), 1U);
    // Key assertion: the Operator-only authored cover deterministically owns
    // its adjacent compiler-synthetic imported-topic vertex.
    EXPECT_EQ(plan.execution_partitions().front().members.size(), 2U);
  }
  {
    Graph graph{"manual-ambiguous-import"};
    test::add_compile_test_default_clock(graph);
    auto imported = graph.import_topic<std::uint32_t>("holomq://manual-ambiguous-import");
    const auto left = graph.op<test::CompileTestOperator>("left");
    const auto right = graph.op<test::CompileTestOperator>("right");
    graph.add_flow(imported.output(), left->input);
    graph.add_flow(imported.output(), right->input);
    graph.partition("left").add(left);
    graph.partition("right").add(right);

    const ExecutionPlan plan = compile(graph);
    EXPECT_FALSE(plan.ok());
    EXPECT_TRUE(test::has_diagnostic(plan, "PARTITION_COVER_INVALID"));
  }
  {
    Graph graph{"manual-two-import-roots"};
    test::add_compile_test_default_clock(graph);
    auto first_import = graph.import_topic<std::uint32_t>("holomq://manual-first-import");
    auto second_import = graph.import_topic<std::uint32_t>("holomq://manual-second-import");
    const auto first_sink = graph.op<test::CompileTestOperator>("first-sink");
    const auto second_sink = graph.op<test::CompileTestOperator>("second-sink");
    graph.add_flow(first_import.output(), first_sink->input);
    graph.add_flow(second_import.output(), second_sink->input);
    graph.partition("invalid-two-roots").add(first_sink).add(second_sink);

    const ExecutionPlan plan = compile(graph);
    EXPECT_FALSE(plan.ok());
    EXPECT_TRUE(test::has_diagnostic(plan, "PARTITION_COVER_INVALID"));
  }
}

//   [operator]  (invalid Profile 0xFF; plan returned but ok()==false)
// REQ: COMPILE-001
TEST(CompileEntryPoint, RejectedCompileStillReturnsInspectableNonRunnablePlan) {
  Graph graph{"compile-rejected"};
  (void)graph.op<test::CompileTestOperator>("operator");
  // Key setup: deliberately invalid Profile value forces compile rejection before lowering.
  CompileOptions options;
  options.profile =
      static_cast<Profile>(0xFF);  // NOLINT(clang-analyzer-optin.core.EnumCastOutOfRange)

  const ExecutionPlan plan = compile(graph, options);
  // Key assertion: rejected plan still exposes json(), diagnostics(), graph(), and partition data.
  EXPECT_FALSE(plan.ok());
  EXPECT_FALSE(plan.json().empty());
  EXPECT_FALSE(plan.diagnostics().empty());
  EXPECT_EQ(plan.graph().vertices().size(), 1U);
  EXPECT_THROW(plan.require_no_errors(), CompileError);
  EXPECT_FALSE(plan.execution_partitions().empty());
  const auto& requested_algorithm = plan.partitioning_plan().requested_algorithm;
  if (!requested_algorithm.has_value()) {
    ADD_FAILURE() << "expected the rejected plan to retain its requested algorithm";
    return;
  }
  EXPECT_EQ(requested_algorithm->id, "holoscan.partition.serial_baseline");
  EXPECT_TRUE(plan.partitioning_plan().accepted_algorithm.has_value());
  EXPECT_NE(plan.clock_catalog().plan_generation, 0U);
  ASSERT_EQ(plan.lifecycle_restart_targets().size(), 1U);
  EXPECT_FALSE(plan.lifecycle_restart_targets().front().effective_profile.has_value());
}

//   [operator]  (inline-only storage mode; opaque override contribution rejected)
// REQ: COMPILE-001
TEST(CompileEntryPoint, AppliesInertParameterPolicyAndRejectsOpaqueOverrideContribution) {
  Graph graph{"unsupported-parameter-inputs"};
  (void)graph.op<test::CompileTestOperator>("operator");

  // Key setup: kInlineOnly disables runtime updates; opaque ParamOverrideSet triggers rejection.
  CompileOptions configured;
  configured.parameters.storage = ParamStorageMode::kInlineOnly;
  const ExecutionPlan configured_plan = compile(graph, configured);
  EXPECT_TRUE(configured_plan.ok()) << configured_plan.json();
  ASSERT_TRUE(configured_plan.parameter_runtime_capacity().has_value());
  EXPECT_FALSE(configured_plan.parameter_runtime_capacity()->runtime_updates_enabled);

  CompileOptions overridden;
  overridden.parameter_overrides = detail::ParamOverrideSetAccess::make();
  const ExecutionPlan overridden_plan = compile(graph, std::move(overridden));
  // Key assertion: opaque override plan fails with PARAMETER_OVERRIDE_RECONSTRUCTION_UNAVAILABLE.
  EXPECT_FALSE(overridden_plan.ok());
  EXPECT_TRUE(
      test::has_diagnostic(overridden_plan, "PARAMETER_OVERRIDE_RECONSTRUCTION_UNAVAILABLE"));
}

//   [LifecycleCapacityOperator]  (each lifecycle resource dimension validated independently)
// REQ: COMPILE-010
TEST(CompileEntryPoint, CrossValidatesAllLifecycleResourcesForRepeatedParameterSubmissions) {
  Graph graph{"parameter-lifecycle-capacity"};
  (void)graph.op<LifecycleCapacityOperator>("operator");

  // Key setup: ample capacity produces required_lifecycle_capacity values for the operator.
  CompileOptions accepted_options;
  accepted_options.lifecycle_transactions = ample_parameter_lifecycle_capacity();
  const ExecutionPlan accepted = compile(graph, accepted_options);
  ASSERT_TRUE(accepted.ok()) << accepted.json();
  ASSERT_TRUE(accepted.parameter_runtime_capacity());
  const auto required = accepted.parameter_runtime_capacity()->required_lifecycle_capacity;
  EXPECT_EQ(required.max_pending_requests, 4U);
  EXPECT_EQ(required.max_pending_transactions, 4U);
  EXPECT_EQ(required.max_receipts, 8U);
  EXPECT_EQ(required.max_fenced_operators, 1U);
  EXPECT_EQ(required.max_candidate_generations, 4U);
  EXPECT_EQ(required.max_events, 64U);
  EXPECT_EQ(required.max_audit_records, 8U);
  EXPECT_EQ(required.max_quiesce_records, 2U);
  EXPECT_EQ(required.max_rollback_records, 2U);

  struct CapacityFieldCase {
    std::uint32_t LifecycleTransactionOptions::* option;
    std::uint32_t LifecycleTransactionCapacityValues::* required;
  };
  constexpr std::array cases{
      CapacityFieldCase{&LifecycleTransactionOptions::max_pending_requests,
                        &LifecycleTransactionCapacityValues::max_pending_requests},
      CapacityFieldCase{&LifecycleTransactionOptions::max_pending_transactions,
                        &LifecycleTransactionCapacityValues::max_pending_transactions},
      CapacityFieldCase{&LifecycleTransactionOptions::max_receipts,
                        &LifecycleTransactionCapacityValues::max_receipts},
      CapacityFieldCase{&LifecycleTransactionOptions::max_fenced_operators,
                        &LifecycleTransactionCapacityValues::max_fenced_operators},
      CapacityFieldCase{&LifecycleTransactionOptions::max_candidate_generations,
                        &LifecycleTransactionCapacityValues::max_candidate_generations},
      CapacityFieldCase{&LifecycleTransactionOptions::max_events,
                        &LifecycleTransactionCapacityValues::max_events},
      CapacityFieldCase{&LifecycleTransactionOptions::max_audit_records,
                        &LifecycleTransactionCapacityValues::max_audit_records},
      CapacityFieldCase{&LifecycleTransactionOptions::max_quiesce_records,
                        &LifecycleTransactionCapacityValues::max_quiesce_records},
      CapacityFieldCase{&LifecycleTransactionOptions::max_rollback_records,
                        &LifecycleTransactionCapacityValues::max_rollback_records},
  };
  // Key assertion: reducing any single resource field below its computed minimum rejects the plan.
  for (const auto& field : cases) {
    CompileOptions insufficient;
    insufficient.lifecycle_transactions = ample_parameter_lifecycle_capacity();
    insufficient.lifecycle_transactions.value().*field.option = required.*field.required - 1U;
    const ExecutionPlan rejected = compile(graph, insufficient);
    EXPECT_FALSE(rejected.ok());
    EXPECT_TRUE(test::has_diagnostic(rejected, "PARAMETER_LIFECYCLE_CAPACITY_INSUFFICIENT"));
  }
}

//   mutable-target <--[lifecycle: before configure]-- non-mutable-dependency
// REQ: COMPILE-010
TEST(CompileEntryPoint, ExpandsParameterCapacityAcrossNonMutableLifecycleDependencies) {
  Graph graph{"parameter-lifecycle-dependency-capacity"};
  const auto mutable_target = graph.op<LifecycleCapacityOperator>("mutable-target");
  const auto dependency = graph.op<test::CompileTestOperator>("non-mutable-dependency");
  // Key setup: before-configure dependency on a non-mutable op expands the restart closure.
  graph.lifecycle().before(mutable_target, dependency, LifecycleStage::kConfigure);
  graph.partition("mutable-target").add(mutable_target);
  graph.partition("non-mutable-dependency").add(dependency);

  CompileOptions options;
  options.parameters.max_pending_updates_per_operator = 1U;
  options.lifecycle_transactions = ample_parameter_lifecycle_capacity();
  const ExecutionPlan plan = compile(graph, options);
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_TRUE(plan.parameter_runtime_capacity());
  const auto required = plan.parameter_runtime_capacity()->required_lifecycle_capacity;
  EXPECT_EQ(required.max_fenced_operators, 2U);
  EXPECT_EQ(required.max_candidate_generations, 2U);
  EXPECT_EQ(required.max_quiesce_records, 4U);
  EXPECT_EQ(required.max_rollback_records, 4U);

  const auto target = std::find_if(plan.lifecycle_restart_closures().begin(),
                                   plan.lifecycle_restart_closures().end(),
                                   [](const auto& closure) { return closure.target.id == 1U; });
  ASSERT_NE(target, plan.lifecycle_restart_closures().end());
  // Key assertion: restart closure for mutable-target includes both operators (size==2).
  EXPECT_EQ(target->closure.size(), 2U);

  CompileOptions insufficient = options;
  insufficient.lifecycle_transactions->max_candidate_generations = 1U;
  const ExecutionPlan rejected = compile(graph, insufficient);
  ASSERT_FALSE(rejected.ok());
  EXPECT_TRUE(test::has_diagnostic(rejected, "PARAMETER_LIFECYCLE_CAPACITY_INSUFFICIENT"));
  ASSERT_TRUE(rejected.lifecycle_transaction_capacity());
  EXPECT_FALSE(rejected.lifecycle_transaction_capacity()->parameter_origin_enabled);
  EXPECT_TRUE(std::none_of(rejected.lifecycle_restart_targets().begin(),
                           rejected.lifecycle_restart_targets().end(),
                           [](const auto& item) {
                             return item.parameter_origin_allowed &&
                                    item.disposition ==
                                        LifecycleRestartTargetDisposition::kAdmitted;
                           }));
}

// Phase 1: [first]   Phase 2: [first] [second]
// Retained plan from Phase 1 must remain valid after Phase 2 compile and graph destruction.
// REQ: COMPILE-002
TEST(ExecutionPlanIdentity, OwnsStateAfterGraphMutationAndDestruction) {
  ExecutionPlan retained;
  std::string retained_json;
  PlanDigest retained_digest;
  {
    Graph graph{"owned-state"};
    const auto first = graph.op<test::CompileTestOperator>("first");
    graph.partition("first").add(first);
    // Key setup: compile while graph has one op; add a second op and compile again.
    retained = compile(graph);
    ASSERT_TRUE(retained.ok());
    retained_json = std::string(retained.json());
    retained_digest = retained.digest();

    const auto second = graph.op<test::CompileTestOperator>("second");
    graph.partition("second").add(second);
    const ExecutionPlan next_generation = compile(graph);
    ASSERT_TRUE(next_generation.ok());
    EXPECT_GT(next_generation.generation(), retained.generation());
    EXPECT_NE(next_generation.digest(), retained_digest);
    EXPECT_EQ(retained.json(), retained_json);
  }

  // Key assertion: retained plan preserves json, digest, and graph after destruction.
  EXPECT_TRUE(retained.ok());
  EXPECT_EQ(retained.json(), retained_json);
  EXPECT_EQ(retained.digest(), retained_digest);
  EXPECT_EQ(retained.graph().vertices().size(), 1U);

  const ExecutionPlan shared_copy = retained;
  EXPECT_EQ(shared_copy.json().data(), retained.json().data());
  EXPECT_EQ(shared_copy.digest(), retained.digest());
}

// [operator] + ManualClock "clock"; plan destroyed while graph and catalog views are retained.
// REQ: COMPILE-002
// REQ: COMPILE-007
// REQ: COMPILE-011
TEST(ExecutionPlanIdentity, ReturnedViewsRetainCanonicalStateAfterPlanDestruction) {
  ExecutionGraphView retained_graph;
  ClockCatalogView retained_catalog;
  {
    Graph graph{"retained-plan-views"};
    (void)graph.op<test::CompileTestOperator>("operator");
    (void)graph.add_clock<ManualClock>("clock", std::int64_t{0});
    const ExecutionPlan plan = compile(graph);
    ASSERT_TRUE(plan.ok());
    // Key setup: capture graph and catalog views before the ExecutionPlan goes out of scope.
    retained_graph = plan.graph();
    retained_catalog = plan.clock_catalog();
  }

  // Key assertion: graph view and clock catalog remain accessible after plan is destroyed.
  ASSERT_EQ(retained_graph.vertices().size(), 1U);
  ASSERT_EQ(retained_catalog.clocks.size(), 1U);
  EXPECT_NE(retained_catalog.plan_generation, 0U);
  EXPECT_EQ(retained_catalog.clocks.front().id, ClockId{1U});
}

// root_clock --> child.import_clock("timeline", default) --> child/worker
// REQ: GRAPH-026
// REQ: COMPILE-053
TEST(ExecutionPlanIdentity, ExecutionGraphViewExposesFrozenClockImportsAndBindings) {
  ExecutionGraphView retained_graph;
  {
    Graph graph{"execution-clock-boundary"};
    const ClockKey root_clock = graph.add_clock<ManualClock>("root_clock", std::int64_t{17});
    auto child = graph.embed<ExecutionClockImportGraph>("child");
    graph.bind_clock(child, child->clock_import(), root_clock);

    const ExecutionPlan plan = compile(graph);
    ASSERT_TRUE(plan.ok()) << plan.json();
    retained_graph = plan.graph();
  }

  ASSERT_EQ(retained_graph.clock_imports().size(), 1U);
  const ClockImportAuthoringRecord& imported = retained_graph.clock_imports().front();
  EXPECT_TRUE(imported.graph);
  EXPECT_EQ(imported.key, ClockKey::named("timeline"));
  EXPECT_TRUE(imported.is_default);

  ASSERT_EQ(retained_graph.clock_bindings().size(), 1U);
  const EmbeddedClockBindingAuthoringRecord& binding = retained_graph.clock_bindings().front();
  EXPECT_EQ(binding.child_graph, imported.graph);
  EXPECT_EQ(binding.child_clock, imported.key);
  EXPECT_TRUE(binding.parent_graph);
  EXPECT_NE(binding.parent_graph, binding.child_graph);
  EXPECT_EQ(binding.parent_clock, ClockKey::named("root_clock"));
}

//   forward:  source --> sink  (ops added in declaration order)
//   reverse:  source --> sink  (ops added in reverse order via add_linear_graph flag)
// REQ: COMPILE-002
TEST(ExecutionPlanIdentity, IsCanonicalAcrossEquivalentAuthoringOrder) {
  Graph forward{"canonical"};
  test::add_linear_graph(forward, false);
  Graph reverse{"canonical"};
  test::add_linear_graph(reverse, true);

  const ExecutionPlan first = compile(forward);
  const ExecutionPlan second = compile(reverse);
  ASSERT_TRUE(first.ok());
  ASSERT_TRUE(second.ok());
  // Key assertion: same graph_hash and digest regardless of op-addition order.
  EXPECT_EQ(first.graph_hash(), second.graph_hash());
  EXPECT_EQ(first.digest(), second.digest());
  ASSERT_EQ(first.execution_partitions().size(), second.execution_partitions().size());
  EXPECT_EQ(first.partition_decisions().size(), second.partition_decisions().size());
}

// [LifecycleCapacityOperator]: derived capacity options produce the same digest as
// explicit options that supply identical generation_slots and audit_capacity values.
// REQ: COMPILE-002
// REQ: COMPILE-019
TEST(ExecutionPlanIdentity, CanonicalizesDerivedAndExplicitEquivalentParameterCapacities) {
  Graph graph{"canonical-parameter-capacity"};
  (void)graph.op<LifecycleCapacityOperator>("operator");

  CompileOptions derived_options;
  derived_options.lifecycle_transactions = ample_parameter_lifecycle_capacity();
  const ExecutionPlan derived = compile(graph, derived_options);
  ASSERT_TRUE(derived.ok()) << derived.json();
  ASSERT_TRUE(derived.parameter_runtime_capacity().has_value());

  // Key setup: explicit options feed generation_slots and audit_capacity from the derived plan.
  CompileOptions explicit_options;
  explicit_options.lifecycle_transactions = ample_parameter_lifecycle_capacity();
  explicit_options.parameters.generation_slots_per_operator =
      derived.parameter_runtime_capacity()->generation_slots_per_operator;
  explicit_options.parameters.audit_capacity = derived.parameter_runtime_capacity()->audit_capacity;
  const ExecutionPlan explicit_equivalent = compile(graph, explicit_options);
  ASSERT_TRUE(explicit_equivalent.ok()) << explicit_equivalent.json();

  // Key assertion: same digest and JSON between derived and explicitly-equivalent options.
  EXPECT_EQ(derived.clock_catalog().plan_generation,
            explicit_equivalent.clock_catalog().plan_generation);
  EXPECT_EQ(derived.digest(), explicit_equivalent.digest());
  EXPECT_EQ(derived.json(), explicit_equivalent.json());
}

// [operator] with an unused generated-endpoint pin; digest must be identical to baseline.
// REQ: COMPILE-002
// REQ: COMPILE-012
TEST(ExecutionPlanIdentity, ExcludesUnusedValidGeneratedEndpointPins) {
  Graph graph{"canonical-unused-endpoint-pin"};
  (void)graph.op<test::CompileTestOperator>("operator");

  const ExecutionPlan baseline = compile(graph);
  ASSERT_TRUE(baseline.ok()) << baseline.json();

  CompileOptions options;
  options.deployment.allow_generated_endpoint_name(
      GeneratedEndpointNamePin{.final_name = "unused-generated-endpoint"});
  const ExecutionPlan with_unused_pin = compile(graph, options);
  ASSERT_TRUE(with_unused_pin.ok()) << with_unused_pin.json();

  // Key assertion: unused generated endpoint pin does not change digest or JSON.
  EXPECT_EQ(baseline.clock_catalog().plan_generation,
            with_unused_pin.clock_catalog().plan_generation);
  EXPECT_EQ(baseline.digest(), with_unused_pin.digest());
  EXPECT_EQ(baseline.json(), with_unused_pin.json());
}

//   [a-first]  [z-last]  (isolated ops; alphabetic name order determines canonical vertex IDs)
// REQ: COMPILE-002
// REQ: COMPILE-007
TEST(ExecutionPlanIdentity, ExposesTheSameCanonicalIdsUsedByPlanRecords) {
  Graph graph{"canonical-plan-view"};
  const auto last = graph.op<test::CompileTestOperator>("z-last");
  const auto first = graph.op<test::CompileTestOperator>("a-first");
  graph.partition("a-first").add(first);
  graph.partition("z-last").add(last);

  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok());
  ASSERT_EQ(plan.graph().vertices().size(), 2U);
  ASSERT_EQ(plan.partition_decisions().size(), 2U);
  // Key assertion: alphabetical name order assigns vertex IDs; partition_decisions mirror them.
  EXPECT_EQ(plan.graph().vertices()[0].name, "a-first");
  EXPECT_EQ(plan.graph().vertices()[0].id, VertexId{1U});
  EXPECT_EQ(plan.graph().vertices()[1].name, "z-last");
  EXPECT_EQ(plan.graph().vertices()[1].id, VertexId{2U});
  EXPECT_EQ(plan.partition_decisions()[0].vertex, plan.graph().vertices()[0].id);
  EXPECT_EQ(plan.partition_decisions()[1].vertex, plan.graph().vertices()[1].id);
}

// REQ: COMPILE-007
// REQ: COMPILE-025
TEST(ExecutionPlanArtifacts, PublishesManifestLastAndPreservesIdenticalFileTimes) {
  Graph graph{"artifact-publication"};
  (void)graph.op<test::CompileTestOperator>("operator");
  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();

  const ScopedArtifactRoot root_scope = unique_artifact_root();
  const std::filesystem::path& root = root_scope.path();
  const auto first =
      emit_artifacts(plan, ArtifactEmitOptions{.root = root, .deployment_id = "test-deployment"});
  ASSERT_TRUE(first) << first.error().message;
  EXPECT_TRUE(first->changed);
  EXPECT_TRUE(std::filesystem::exists(first->catalog_path));
  EXPECT_TRUE(std::filesystem::exists(first->manifest_path));
  const auto directory = first->manifest_path.parent_path();
  EXPECT_TRUE(std::filesystem::exists(directory / "execution-plan.json"));
  EXPECT_TRUE(std::filesystem::exists(directory / "profile-manifest.json"));
  const auto catalog_time = std::filesystem::last_write_time(first->catalog_path);
  const auto plan_time = std::filesystem::last_write_time(directory / "execution-plan.json");
  const auto profile_time = std::filesystem::last_write_time(directory / "profile-manifest.json");
  const auto manifest_time = std::filesystem::last_write_time(first->manifest_path);

  // Key setup: emit a second time to the same root; files must have identical timestamps.
  const auto second =
      emit_artifacts(plan, ArtifactEmitOptions{.root = root, .deployment_id = "test-deployment"});
  ASSERT_TRUE(second) << second.error().message;
  // Key assertion: second emit is not changed and all file timestamps are preserved.
  EXPECT_FALSE(second->changed);
  EXPECT_EQ(std::filesystem::last_write_time(second->catalog_path), catalog_time);
  EXPECT_EQ(std::filesystem::last_write_time(directory / "execution-plan.json"), plan_time);
  EXPECT_EQ(std::filesystem::last_write_time(directory / "profile-manifest.json"), profile_time);
  EXPECT_EQ(std::filesystem::last_write_time(second->manifest_path), manifest_time);
  EXPECT_EQ(second->catalog_digest, first->catalog_digest);

  std::error_code cleanup_error;
  std::filesystem::remove_all(root, cleanup_error);
}

// REQ: COMPILE-007
// REQ: COMPILE-025
TEST(ExecutionPlanArtifacts, RejectsBeforePublishingIntoAnInvalidRoot) {
  Graph graph{"artifact-failure"};
  (void)graph.op<test::CompileTestOperator>("operator");
  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();

  const ScopedArtifactRoot root_scope = unique_artifact_root();
  const std::filesystem::path& root = root_scope.path();
  // Key setup: a regular file at the root path blocks directory creation.
  {
    std::ofstream blocker(root, std::ios::binary);
    ASSERT_TRUE(blocker);
    blocker << "not-a-directory";
  }
  const auto emitted =
      emit_artifacts(plan, ArtifactEmitOptions{.root = root, .deployment_id = "blocked"});
  // Key assertion: emit returns kFailure without creating any artifact files.
  EXPECT_FALSE(emitted);
  EXPECT_EQ(emitted.error().code, ErrorCode::kFailure);

  std::error_code cleanup_error;
  std::filesystem::remove(root, cleanup_error);
}

// REQ: COMPILE-041
TEST(ExecutionPlanArtifacts, RejectsSymlinkedDigestAndDeploymentComponents) {
  Graph graph{"artifact-symlink-containment"};
  (void)graph.op<test::CompileTestOperator>("operator");
  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();

  const ScopedArtifactRoot root_scope = unique_artifact_root();
  const std::filesystem::path& root = root_scope.path();
  const std::filesystem::path outside = root.parent_path() / "outside";
  const std::filesystem::path sentinel = outside / "sentinel";
  ASSERT_TRUE(std::filesystem::create_directories(outside));
  {
    std::ofstream stream(sentinel, std::ios::binary);
    ASSERT_TRUE(stream);
    stream << "unchanged";
  }
  ASSERT_TRUE(std::filesystem::create_directories(root));

  const std::filesystem::path digest_directory = root / digest_hex(plan.digest());
  std::error_code error;
  std::filesystem::create_directory_symlink(outside, digest_directory, error);
#if defined(_WIN32)
  if (error) {
    GTEST_SKIP() << "directory symlink creation is not permitted: " << error.message();
  }
#else
  ASSERT_FALSE(error) << error.message();
#endif
  // Key setup: digest directory is a symlink to an external path; emit must refuse to follow it.
  const auto digest_link =
      emit_artifacts(plan, ArtifactEmitOptions{.root = root, .deployment_id = "symlinked-digest"});
  // Key assertion: emit fails with ARTIFACT_DIRECTORY_CREATE_FAILED; sentinel is unchanged.
  ASSERT_FALSE(digest_link);
  EXPECT_EQ(digest_link.error().code, ErrorCode::kFailure);
  EXPECT_EQ(digest_link.error().message, "ARTIFACT_DIRECTORY_CREATE_FAILED");
  EXPECT_EQ(read_artifact(sentinel), "unchanged");
  EXPECT_EQ(std::distance(std::filesystem::directory_iterator(outside),
                          std::filesystem::directory_iterator{}),
            1);

  ASSERT_TRUE(std::filesystem::remove(digest_directory));
  ASSERT_TRUE(std::filesystem::create_directory(digest_directory));
  const std::filesystem::path deployment_directory = digest_directory / "symlinked-deployment";
  std::filesystem::create_directory_symlink(outside, deployment_directory, error);
  ASSERT_FALSE(error) << error.message();
  const auto deployment_link = emit_artifacts(
      plan, ArtifactEmitOptions{.root = root, .deployment_id = "symlinked-deployment"});
  ASSERT_FALSE(deployment_link);
  EXPECT_EQ(deployment_link.error().code, ErrorCode::kFailure);
  EXPECT_EQ(deployment_link.error().message, "ARTIFACT_DIRECTORY_CREATE_FAILED");
  EXPECT_EQ(read_artifact(sentinel), "unchanged");
  EXPECT_EQ(std::distance(std::filesystem::directory_iterator(outside),
                          std::filesystem::directory_iterator{}),
            1);

  for (const auto& entry : std::filesystem::recursive_directory_iterator(root)) {
    const std::string name = entry.path().filename().generic_string();
    EXPECT_EQ(name.find("artifact-manifest.json"), std::string::npos);
    EXPECT_EQ(name.find(".tmp-"), std::string::npos);
  }
}

// REQ: COMPILE-025
TEST(ExecutionPlanArtifacts, RejectsNonportableDeploymentComponentsBeforeFilesystemMutation) {
  Graph graph{"artifact-deployment-id"};
  (void)graph.op<test::CompileTestOperator>("operator");
  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();

  const ScopedArtifactRoot root_scope = unique_artifact_root();
  const std::filesystem::path& root = root_scope.path();
  // Key setup: 14 portable-name violations covering reserved names, paths, and hidden files.
  const std::array<std::string, 14> invalid_ids{
      "",
      ".",
      "..",
      "D:",
      "D:relative",
      "CON",
      "nul.txt",
      "COM1",
      "lpt9.log",
      "/absolute",
      "nested/path",
      "nested\\path",
      ".hidden",
      "trailing.",
  };
  for (const auto& deployment_id : invalid_ids) {
    const auto emitted =
        emit_artifacts(plan, ArtifactEmitOptions{.root = root, .deployment_id = deployment_id});
    ASSERT_FALSE(emitted) << deployment_id;
    EXPECT_EQ(emitted.error().code, ErrorCode::kInvalidArgument) << deployment_id;
    EXPECT_EQ(emitted.error().message, "ARTIFACT_PLAN_OR_DEPLOYMENT_INVALID") << deployment_id;
  }
  // Key assertion: no filesystem mutation occurs for any invalid deployment ID.
  EXPECT_FALSE(std::filesystem::exists(root));
}

// REQ: COMPILE-007
// REQ: COMPILE-025
TEST(ExecutionPlanArtifacts, RepairsCorruptManifestWithoutRewritingValidContent) {
  Graph graph{"artifact-manifest-repair"};
  (void)graph.op<test::CompileTestOperator>("operator");
  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();

  const ScopedArtifactRoot root_scope = unique_artifact_root();
  const std::filesystem::path& root = root_scope.path();
  const auto first =
      emit_artifacts(plan, ArtifactEmitOptions{.root = root, .deployment_id = "repair"});
  ASSERT_TRUE(first) << first.error().message;
  const auto directory = first->manifest_path.parent_path();
  const auto catalog_time = std::filesystem::last_write_time(first->catalog_path);
  const auto plan_time = std::filesystem::last_write_time(directory / "execution-plan.json");
  const auto profile_time = std::filesystem::last_write_time(directory / "profile-manifest.json");
  // Key setup: overwrite the manifest with a syntactically valid but wrong-schema JSON object.
  {
    std::ofstream corrupt(first->manifest_path, std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(corrupt);
    corrupt << "{\"schema\":\"corrupt\"}";
  }

  // Key assertion: repaired emit rewrites the manifest but preserves unchanged file timestamps.
  const auto repaired =
      emit_artifacts(plan, ArtifactEmitOptions{.root = root, .deployment_id = "repair"});
  ASSERT_TRUE(repaired) << repaired.error().message;
  EXPECT_TRUE(repaired->changed);
  EXPECT_EQ(std::filesystem::last_write_time(repaired->catalog_path), catalog_time);
  EXPECT_EQ(std::filesystem::last_write_time(directory / "execution-plan.json"), plan_time);
  EXPECT_EQ(std::filesystem::last_write_time(directory / "profile-manifest.json"), profile_time);
  const std::string manifest = read_artifact(repaired->manifest_path);
  EXPECT_NE(manifest.find("holoscan.artifact_manifest.v1"), std::string::npos);
  EXPECT_NE(manifest.find(repaired->catalog_digest), std::string::npos);

  const auto unchanged =
      emit_artifacts(plan, ArtifactEmitOptions{.root = root, .deployment_id = "repair"});
  ASSERT_TRUE(unchanged) << unchanged.error().message;
  EXPECT_FALSE(unchanged->changed);

  std::error_code cleanup_error;
  std::filesystem::remove_all(root, cleanup_error);
}

// REQ: COMPILE-025
TEST(ExecutionPlanArtifacts, AcceptsCanonicalSignedInt64TimestampBoundaries) {
  Graph graph{"artifact-negative-timestamp"};
  (void)graph.op<test::CompileTestOperator>("operator");
  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();

  const ScopedArtifactRoot root_scope = unique_artifact_root();
  const std::filesystem::path& root = root_scope.path();
  const auto first = emit_artifacts(
      plan, ArtifactEmitOptions{.root = root, .deployment_id = "negative-timestamp"});
  ASSERT_TRUE(first) << first.error().message;
  constexpr std::string_view kTimestampKey{"\"content_updated_at_ns\":"};
  constexpr std::array<std::string_view, 2> kBoundaryTimestamps{"-9223372036854775808",
                                                                "9223372036854775807"};
  // Key setup: replace content_updated_at_ns with each INT64 boundary value in the manifest.
  for (const std::string_view timestamp : kBoundaryTimestamps) {
    std::string manifest = read_artifact(first->manifest_path);
    const std::size_t timestamp_begin = manifest.find(kTimestampKey);
    ASSERT_NE(timestamp_begin, std::string::npos);
    const std::size_t value_begin = timestamp_begin + kTimestampKey.size();
    const std::size_t value_end = manifest.find(',', value_begin);
    ASSERT_NE(value_end, std::string::npos);
    manifest.replace(value_begin, value_end - value_begin, timestamp);
    {
      std::ofstream stream(first->manifest_path, std::ios::binary | std::ios::trunc);
      ASSERT_TRUE(stream);
      stream << manifest;
    }

    const auto unchanged = emit_artifacts(
        plan, ArtifactEmitOptions{.root = root, .deployment_id = "negative-timestamp"});
    ASSERT_TRUE(unchanged) << unchanged.error().message;
    // Key assertion: boundary timestamp value is treated as valid; emit reports unchanged.
    EXPECT_FALSE(unchanged->changed);
    EXPECT_EQ(read_artifact(unchanged->manifest_path), manifest);
  }
}

// REQ: COMPILE-025
TEST(ExecutionPlanArtifacts, RepairsNoncanonicalAndOutOfRangeSignedTimestamps) {
  Graph graph{"artifact-invalid-timestamps"};
  (void)graph.op<test::CompileTestOperator>("operator");
  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();

  const ScopedArtifactRoot root_scope = unique_artifact_root();
  const std::filesystem::path& root = root_scope.path();
  const auto first = emit_artifacts(
      plan, ArtifactEmitOptions{.root = root, .deployment_id = "invalid-timestamps"});
  ASSERT_TRUE(first) << first.error().message;
  constexpr std::string_view kTimestampKey{"\"content_updated_at_ns\":"};
  constexpr std::array<std::string_view, 5> kInvalidTimestamps{
      "+1", "01", "-0", "9223372036854775808", "-9223372036854775809"};
  // Key setup: replace content_updated_at_ns with non-canonical or out-of-range values.
  for (const std::string_view timestamp : kInvalidTimestamps) {
    std::string manifest = read_artifact(first->manifest_path);
    const std::size_t timestamp_begin = manifest.find(kTimestampKey);
    ASSERT_NE(timestamp_begin, std::string::npos);
    const std::size_t value_begin = timestamp_begin + kTimestampKey.size();
    const std::size_t value_end = manifest.find(',', value_begin);
    ASSERT_NE(value_end, std::string::npos);
    manifest.replace(value_begin, value_end - value_begin, timestamp);
    {
      std::ofstream stream(first->manifest_path, std::ios::binary | std::ios::trunc);
      ASSERT_TRUE(stream);
      stream << manifest;
    }

    const auto repaired = emit_artifacts(
        plan, ArtifactEmitOptions{.root = root, .deployment_id = "invalid-timestamps"});
    ASSERT_TRUE(repaired) << repaired.error().message;
    // Key assertion: each non-canonical or out-of-range timestamp triggers manifest repair.
    EXPECT_TRUE(repaired->changed);
    EXPECT_NE(read_artifact(repaired->manifest_path), manifest);
  }
}

// REQ: COMPILE-007
// REQ: COMPILE-025
TEST(ExecutionPlanArtifacts, RepairsDestinationWithExpectedBytesPlusOneSentinel) {
  Graph graph{"artifact-oversized-destination"};
  (void)graph.op<test::CompileTestOperator>("operator");
  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();

  const ScopedArtifactRoot root_scope = unique_artifact_root();
  const std::filesystem::path& root = root_scope.path();
  const auto first = emit_artifacts(
      plan, ArtifactEmitOptions{.root = root, .deployment_id = "oversized-destination"});
  ASSERT_TRUE(first) << first.error().message;
  const std::filesystem::path plan_path =
      first->manifest_path.parent_path() / "execution-plan.json";
  const std::string expected = read_artifact(plan_path);
  // Key setup: append one '!' sentinel byte to the plan JSON, making it exactly size+1.
  append_artifact_sentinel(plan_path);
  ASSERT_EQ(std::filesystem::file_size(plan_path), expected.size() + 1U);

  const auto repaired = emit_artifacts(
      plan, ArtifactEmitOptions{.root = root, .deployment_id = "oversized-destination"});
  ASSERT_TRUE(repaired) << repaired.error().message;
  // Key assertion: repaired emit corrects the oversized file; unchanged emit is then a no-op.
  EXPECT_TRUE(repaired->changed);
  EXPECT_EQ(read_artifact(plan_path), expected);

  const auto unchanged = emit_artifacts(
      plan, ArtifactEmitOptions{.root = root, .deployment_id = "oversized-destination"});
  ASSERT_TRUE(unchanged) << unchanged.error().message;
  EXPECT_FALSE(unchanged->changed);

  std::error_code cleanup_error;
  std::filesystem::remove_all(root, cleanup_error);
}

// REQ: COMPILE-007
// REQ: COMPILE-025
TEST(ExecutionPlanArtifacts, RepairsManifestAtFixedSchemaMaximumPlusOne) {
  Graph graph{"artifact-oversized-manifest"};
  (void)graph.op<test::CompileTestOperator>("operator");
  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();

  const ScopedArtifactRoot root_scope = unique_artifact_root();
  const std::filesystem::path& root = root_scope.path();
  const auto first = emit_artifacts(
      plan, ArtifactEmitOptions{.root = root, .deployment_id = "oversized-manifest"});
  ASSERT_TRUE(first) << first.error().message;
  const std::string expected = read_artifact(first->manifest_path);
  const std::size_t maximum_manifest_bytes =
      detail::ArtifactEmitterTestAccess::maximum_manifest_bytes();
  ASSERT_LE(expected.size(), maximum_manifest_bytes);
  // Key setup: pad the manifest to maximum_manifest_bytes+1 with '!' bytes.
  {
    std::ofstream stream(first->manifest_path, std::ios::binary | std::ios::app);
    ASSERT_TRUE(stream);
    for (std::size_t index = expected.size(); index <= maximum_manifest_bytes; ++index) {
      stream.put('!');
    }
  }
  ASSERT_EQ(std::filesystem::file_size(first->manifest_path), maximum_manifest_bytes + 1U);

  const auto repaired = emit_artifacts(
      plan, ArtifactEmitOptions{.root = root, .deployment_id = "oversized-manifest"});
  ASSERT_TRUE(repaired) << repaired.error().message;
  // Key assertion: repaired emit removes all padding and restores a valid manifest JSON.
  EXPECT_TRUE(repaired->changed);
  const std::string repaired_manifest = read_artifact(repaired->manifest_path);
  EXPECT_EQ(repaired_manifest.find('!'), std::string::npos);
  EXPECT_NE(repaired_manifest.find("holoscan.artifact_manifest.v1"), std::string::npos);
  EXPECT_NE(repaired_manifest.find(repaired->catalog_digest), std::string::npos);

  const auto unchanged = emit_artifacts(
      plan, ArtifactEmitOptions{.root = root, .deployment_id = "oversized-manifest"});
  ASSERT_TRUE(unchanged) << unchanged.error().message;
  EXPECT_FALSE(unchanged->changed);

  std::error_code cleanup_error;
  std::filesystem::remove_all(root, cleanup_error);
}

// REQ: COMPILE-025
TEST(ExecutionPlanArtifacts, RejectsOversizedTemporaryAndRemovesOwnedFiles) {
  Graph graph{"artifact-oversized-temporary"};
  (void)graph.op<test::CompileTestOperator>("operator");
  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();

  const ScopedArtifactRoot root_scope = unique_artifact_root();
  const std::filesystem::path& root = root_scope.path();
  // Key setup: oversize_next_temporary() makes the next temporary exceed its expected byte count.
  // Exercise the one-byte oversize sentinel at the temporary verification boundary.
  detail::ArtifactEmitterTestAccess::oversize_next_temporary();
  const auto failed = emit_artifacts(
      plan, ArtifactEmitOptions{.root = root, .deployment_id = "oversized-temporary"});
  detail::ArtifactEmitterTestAccess::reset_temporary_injection();
  // Key assertion: emit fails with ARTIFACT_TEMPORARY_VERIFY_FAILED; no temps or manifests left.
  ASSERT_FALSE(failed);
  EXPECT_EQ(failed.error().code, ErrorCode::kFailure);
  EXPECT_EQ(failed.error().message, "ARTIFACT_TEMPORARY_VERIFY_FAILED");

  std::size_t manifest_count = 0U;
  std::size_t temporary_count = 0U;
  for (const auto& entry : std::filesystem::recursive_directory_iterator(root)) {
    const std::string name = entry.path().filename().generic_string();
    manifest_count += name == "artifact-manifest.json" ? 1U : 0U;
    temporary_count += name.find(".tmp-") == std::string::npos ? 0U : 1U;
  }
  EXPECT_EQ(manifest_count, 0U);
  EXPECT_EQ(temporary_count, 0U);

  std::error_code cleanup_error;
  std::filesystem::remove_all(root, cleanup_error);
}

// REQ: COMPILE-007
// REQ: COMPILE-025
TEST(ExecutionPlanArtifacts, ConcurrentWritersLeaveOneValidBundleAndNoTemporaryFiles) {
  Graph graph{"artifact-concurrent"};
  (void)graph.op<test::CompileTestOperator>("operator");
  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();

  const ScopedArtifactRoot root_scope = unique_artifact_root();
  const std::filesystem::path& root = root_scope.path();
  std::atomic<std::uint32_t> successes{};
  auto writer = [&]() {
    const auto result =
        emit_artifacts(plan, ArtifactEmitOptions{.root = root, .deployment_id = "concurrent"});
    if (result) {
      successes.fetch_add(1U);
    }
  };
  // Key setup: two threads both call emit_artifacts for the same plan and deployment ID.
  std::thread first{writer};
  std::thread second{writer};
  first.join();
  second.join();
  // Key assertion: both writers succeed and no .tmp- files remain after joining.
  EXPECT_EQ(successes.load(), 2U);

  const auto verified =
      emit_artifacts(plan, ArtifactEmitOptions{.root = root, .deployment_id = "concurrent"});
  ASSERT_TRUE(verified) << verified.error().message;
  EXPECT_FALSE(verified->changed);
  for (const auto& entry :
       std::filesystem::directory_iterator(verified->manifest_path.parent_path())) {
    EXPECT_EQ(entry.path().filename().generic_string().find(".tmp-"), std::string::npos);
  }

  std::error_code cleanup_error;
  std::filesystem::remove_all(root, cleanup_error);
}

// REQ: COMPILE-025
TEST(ExecutionPlanArtifacts, ReplaceFailureRemovesOwnedTemporaryAndDoesNotCommitManifest) {
  Graph graph{"artifact-replace-failure"};
  (void)graph.op<test::CompileTestOperator>("operator");
  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();

  const ScopedArtifactRoot root_scope = unique_artifact_root();
  const std::filesystem::path& root = root_scope.path();
  // Key setup: fail_next_replace() causes the first atomic file replacement to fail.
  detail::ArtifactEmitterTestAccess::fail_next_replace();
  const auto failed =
      emit_artifacts(plan, ArtifactEmitOptions{.root = root, .deployment_id = "replace-failure"});
  // Key assertion: emit fails with ARTIFACT_REPLACE_FAILED; no manifest or temp files written.
  ASSERT_FALSE(failed);
  EXPECT_EQ(failed.error().message, "ARTIFACT_REPLACE_FAILED");

  std::size_t manifest_count = 0U;
  std::size_t temporary_count = 0U;
  for (const auto& entry : std::filesystem::recursive_directory_iterator(root)) {
    const std::string name = entry.path().filename().generic_string();
    manifest_count += name == "artifact-manifest.json" ? 1U : 0U;
    temporary_count += name.find(".tmp-") == std::string::npos ? 0U : 1U;
  }
  EXPECT_EQ(manifest_count, 0U);
  EXPECT_EQ(temporary_count, 0U);

  std::error_code cleanup_error;
  std::filesystem::remove_all(root, cleanup_error);
}

// REQ: COMPILE-025
TEST(ExecutionPlanArtifacts, LaterReplaceFailureLeavesNoCommitMarkerAndRetryRepairsBundle) {
  Graph graph{"artifact-later-replace-failure"};
  (void)graph.op<test::CompileTestOperator>("operator");
  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();

  const ScopedArtifactRoot root_scope = unique_artifact_root();
  const std::filesystem::path& root = root_scope.path();
  // Key setup: allow 2 successful file replacements then inject a failure on the third.
  detail::ArtifactEmitterTestAccess::fail_replace_after_successes(2U);
  const auto failed = emit_artifacts(
      plan, ArtifactEmitOptions{.root = root, .deployment_id = "later-replace-failure"});
  ASSERT_FALSE(failed);
  EXPECT_EQ(failed.error().message, "ARTIFACT_REPLACE_FAILED");

  std::size_t manifest_count = 0U;
  std::size_t temporary_count = 0U;
  for (const auto& entry : std::filesystem::recursive_directory_iterator(root)) {
    const std::string name = entry.path().filename().generic_string();
    manifest_count += name == "artifact-manifest.json" ? 1U : 0U;
    temporary_count += name.find(".tmp-") == std::string::npos ? 0U : 1U;
  }
  EXPECT_EQ(manifest_count, 0U);
  EXPECT_EQ(temporary_count, 0U);

  // Key assertion: a retry after partial failure repairs the bundle and completes successfully.
  const auto repaired = emit_artifacts(
      plan, ArtifactEmitOptions{.root = root, .deployment_id = "later-replace-failure"});
  ASSERT_TRUE(repaired) << repaired.error().message;
  EXPECT_TRUE(repaired->changed);
  EXPECT_TRUE(std::filesystem::exists(repaired->manifest_path));
  const auto unchanged = emit_artifacts(
      plan, ArtifactEmitOptions{.root = root, .deployment_id = "later-replace-failure"});
  ASSERT_TRUE(unchanged) << unchanged.error().message;
  EXPECT_FALSE(unchanged->changed);

  std::error_code cleanup_error;
  std::filesystem::remove_all(root, cleanup_error);
}

// REQ: COMPILE-025
TEST(ExecutionPlanArtifacts, MapsFilesystemExceptionsToTypedFailure) {
  Graph graph{"artifact-filesystem-exception"};
  (void)graph.op<test::CompileTestOperator>("operator");
  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();

  const ScopedArtifactRoot root_scope = unique_artifact_root();
  const std::filesystem::path& root = root_scope.path();
  // Key setup: inject a filesystem_error throw for the next filesystem operation.
  detail::ArtifactEmitterTestAccess::throw_next_filesystem_error();
  const auto failed = emit_artifacts(
      plan, ArtifactEmitOptions{.root = root, .deployment_id = "filesystem-exception"});
  // Key assertion: filesystem_error is caught and mapped to ARTIFACT_FILESYSTEM_EXCEPTION.
  ASSERT_FALSE(failed);
  EXPECT_EQ(failed.error().code, ErrorCode::kFailure);
  EXPECT_EQ(failed.error().message, "ARTIFACT_FILESYSTEM_EXCEPTION");
  EXPECT_FALSE(std::filesystem::exists(root));
}

}  // namespace
}  // namespace holoscan
// NOLINTEND(bugprone-unchecked-optional-access)
