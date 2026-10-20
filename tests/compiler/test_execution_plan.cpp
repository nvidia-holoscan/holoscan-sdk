/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <holoscan/core/diagnostic.hpp>

#include <tuple>

#include <algorithm>
#include <array>
#include <atomic>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <numeric>
#include <optional>
#include <ranges>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include <holoscan/time/manual_clock.hpp>

#include "../common/allocation_counter.hpp"
#include "../support/host_unique_test_name.hpp"
#include "compile_test_graph.hpp"
#include "compiler/artifact_emitter_test_access.hpp"
#include "compiler/execution_plan_state.hpp"
#include "compiler/plan_artifact_serialization.hpp"

// GTest fatal assertions establish optional preconditions that clang-tidy does not model.
// NOLINTBEGIN(bugprone-unchecked-optional-access)
namespace holoscan {

struct LifecycleCapacityParams {
  std::uint32_t value{};
};

HOLOSCAN_PARAM_SCHEMA(LifecycleCapacityParams,
                      holoscan::param_schema("holoscan.test.LifecycleCapacityParams", 1),
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

// Minimal embedded Graph used to prove that baseline manual partitions remain a
// flat-root-only authoring surface.
class ManualPartitionEmbeddedGraph final : public Graph {
 protected:
  void compose() override { (void)op<test::CompileTestOperator>("nested"); }
};

// Minimal composed Graph whose public output resolves to one nested logical Operator. The parent
// fans that output out so schedule-edge validation must cross the Graph-interface binding before
// comparing the every-and-only crossing set.
class PartitionTopologyEmbeddedSourceGraph final : public Graph {
 public:
  Output<std::uint32_t> output{"output"};

 protected:
  void compose() override {
    const auto source = op<test::CompileTestOperator>("source");
    export_output(output, source->output);
  }
};

// Reusable Graph whose decoded Operator name is supplied by the parent. Partition-inspection
// regressions compose instance and Operator segments that would collide under raw slash joining.
class PartitionInspectionNamedOperatorGraph final : public Graph {
 public:
  explicit PartitionInspectionNamedOperatorGraph(std::string operator_name)
      : operator_name_(std::move(operator_name)) {}

 protected:
  void compose() override { (void)op<test::CompileTestOperator>(operator_name_); }

 private:
  std::string operator_name_;
};

// A root Graph can retain a handle produced by its lazy compose() callback. Public partition
// inspection must bind that handle to the generation frozen after successful composition.
class PartitionInspectionComposedHandleGraph final : public Graph {
 public:
  using Graph::Graph;

  [[nodiscard]] const std::optional<OperatorHandle<test::CompileTestOperator>>& source() const {
    return source_;
  }

 protected:
  void compose() override {
    test::add_compile_test_default_clock(*this);
    source_ = op<test::CompileTestOperator>("source");
  }

 private:
  std::optional<OperatorHandle<test::CompileTestOperator>> source_;
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

// Two-input join used by public automatic-partition topology tests. Distinct input ports let the
// tests distinguish a structural fan-in from parallel edges between the same ordered Operator
// pair without depending on Runtime execution.
class PartitionTopologyJoinOperator final : public Operator<> {
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

[[nodiscard]] const ExecutionPartitionRecord* partition_for_path(const ExecutionPlan& plan,
                                                                 std::string_view path) {
  const auto found = std::find_if(plan.execution_partitions().begin(),
                                  plan.execution_partitions().end(),
                                  [&](const auto& partition) {
                                    return std::find(partition.operator_paths.begin(),
                                                     partition.operator_paths.end(),
                                                     path) != partition.operator_paths.end();
                                  });
  return found == plan.execution_partitions().end() ? nullptr : &*found;
}

[[nodiscard]] const PartitionDecisionRecord* partition_decision_for_path(const ExecutionPlan& plan,
                                                                         std::string_view path) {
  const auto vertex = std::find_if(
      plan.graph().vertices().begin(), plan.graph().vertices().end(), [&](const auto& record) {
        return record.kind == VertexKind::kOperator && record.name == path;
      });
  if (vertex == plan.graph().vertices().end()) {
    return nullptr;
  }
  const auto decision =
      std::find_if(plan.partition_decisions().begin(),
                   plan.partition_decisions().end(),
                   [&](const auto& record) { return record.vertex == vertex->id; });
  return decision == plan.partition_decisions().end() ? nullptr : &*decision;
}

[[nodiscard]] bool has_partition_reason(const PartitionDecisionRecord& decision,
                                        PartitionDecisionReason reason) {
  return std::find(decision.reasons.begin(), decision.reasons.end(), reason) !=
         decision.reasons.end();
}

// Check the bounded structural contract through only public ExecutionPlan evidence. Parallel
// edges are collapsed for degree/connectivity classification but remain distinct schedule edges.
void expect_linear_automatic_partitions_and_crossing_edges(const ExecutionPlan& plan) {
  const auto owner_of_port = [&](GraphPortId port) {
    const auto found = std::find_if(plan.graph().ports().begin(),
                                    plan.graph().ports().end(),
                                    [&](const auto& record) { return record.id == port; });
    return found == plan.graph().ports().end() ? VertexId{} : found->owner;
  };
  const auto partition_of_vertex = [&](VertexId vertex) {
    const auto found = std::find_if(plan.execution_partitions().begin(),
                                    plan.execution_partitions().end(),
                                    [&](const auto& partition) {
                                      return std::find(partition.members.begin(),
                                                       partition.members.end(),
                                                       vertex) != partition.members.end();
                                    });
    return found == plan.execution_partitions().end() ? ExecutionPartitionId{} : found->id;
  };

  for (const auto& partition : plan.execution_partitions()) {
    if (partition.authored_name.has_value()) {
      continue;
    }
    // Imported/exported endpoints may retain their own automatic endpoint-only boundary.
    if (partition.operator_paths.empty()) {
      continue;
    }
    std::vector<VertexId> operators;
    for (const VertexId member : partition.members) {
      const auto vertex = std::find_if(plan.graph().vertices().begin(),
                                       plan.graph().vertices().end(),
                                       [&](const auto& record) { return record.id == member; });
      if (vertex != plan.graph().vertices().end() && vertex->kind == VertexKind::kOperator) {
        operators.push_back(member);
      }
    }
    ASSERT_EQ(operators.size(), partition.operator_paths.size());

    std::vector<std::pair<VertexId, VertexId>> simple_edges;
    for (const auto& edge : plan.graph().edges()) {
      const VertexId source = owner_of_port(edge.source);
      const VertexId target = owner_of_port(edge.target);
      if (partition_of_vertex(source) == partition.id &&
          partition_of_vertex(target) == partition.id) {
        simple_edges.emplace_back(source, target);
      }
    }
    std::sort(simple_edges.begin(), simple_edges.end(), [](const auto& lhs, const auto& rhs) {
      return lhs.first.value != rhs.first.value ? lhs.first.value < rhs.first.value
                                                : lhs.second.value < rhs.second.value;
    });
    simple_edges.erase(std::unique(simple_edges.begin(), simple_edges.end()), simple_edges.end());
    EXPECT_EQ(simple_edges.size() + 1U, operators.size());
    for (const VertexId member : operators) {
      EXPECT_LE(std::count_if(simple_edges.begin(),
                              simple_edges.end(),
                              [&](const auto& edge) { return edge.second == member; }),
                1);
      EXPECT_LE(std::count_if(simple_edges.begin(),
                              simple_edges.end(),
                              [&](const auto& edge) { return edge.first == member; }),
                1);
    }
  }

  std::vector<ScheduleEdgeRecord> expected_schedule_edges;
  for (const auto& edge : plan.graph().edges()) {
    const ExecutionPartitionId source = partition_of_vertex(owner_of_port(edge.source));
    const ExecutionPartitionId destination = partition_of_vertex(owner_of_port(edge.target));
    ASSERT_TRUE(source);
    ASSERT_TRUE(destination);
    const auto matching = std::count_if(
        plan.schedule_edges().begin(), plan.schedule_edges().end(), [&](const auto& schedule_edge) {
          return schedule_edge.edge == edge.id && schedule_edge.source == source &&
                 schedule_edge.destination == destination;
        });
    EXPECT_EQ(matching, source == destination ? 0 : 1);
    if (source != destination) {
      expected_schedule_edges.push_back(ScheduleEdgeRecord{edge.id, source, destination});
    }
  }
  ASSERT_EQ(plan.schedule_edges().size(), expected_schedule_edges.size());
  for (std::size_t index = 0U; index < expected_schedule_edges.size(); ++index) {
    EXPECT_EQ(plan.schedule_edges()[index].edge, expected_schedule_edges[index].edge);
    EXPECT_EQ(plan.schedule_edges()[index].source, expected_schedule_edges[index].source);
    EXPECT_EQ(plan.schedule_edges()[index].destination, expected_schedule_edges[index].destination);
  }
}

[[nodiscard]] ExecutionPlan compile_automatic_diamond(bool reverse_declarations,
                                                      bool reverse_edges) {
  Graph graph{"automatic-diamond"};
  test::add_compile_test_default_clock(graph);
  const auto add_edges =
      [&](const auto& source, const auto& left, const auto& right, const auto& join) {
        if (reverse_edges) {
          graph.add_flow(right->output, join->second);
          graph.add_flow(left->output, join->first);
          graph.add_flow(source->output, right->input);
          graph.add_flow(source->output, left->input);
        } else {
          graph.add_flow(source->output, left->input);
          graph.add_flow(source->output, right->input);
          graph.add_flow(left->output, join->first);
          graph.add_flow(right->output, join->second);
        }
      };
  if (reverse_declarations) {
    const auto join = graph.op<PartitionTopologyJoinOperator>("join");
    const auto right = graph.op<test::CompileTestOperator>("right");
    const auto left = graph.op<test::CompileTestOperator>("left");
    const auto source = graph.op<test::CompileTestOperator>("source");
    add_edges(source, left, right, join);
  } else {
    const auto source = graph.op<test::CompileTestOperator>("source");
    const auto left = graph.op<test::CompileTestOperator>("left");
    const auto right = graph.op<test::CompileTestOperator>("right");
    const auto join = graph.op<PartitionTopologyJoinOperator>("join");
    add_edges(source, left, right, join);
  }
  return compile(graph);
}

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
    for (std::size_t attempt = 0U; attempt < 128U; ++attempt) {
      std::filesystem::path candidate_guard =
          std::filesystem::temp_directory_path() /
          test::make_host_unique_test_name("holoscan-holoscan-artifacts");
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

template <typename T>
concept HasArrowOperator = requires(const T& value) { value.operator->(); };

static_assert(!std::default_initializable<ExecutionPartitionView>);
static_assert(!std::default_initializable<ExecutionPartitionsView>);
static_assert(std::forward_iterator<ExecutionPartitionsView::const_iterator>);
static_assert(std::is_same_v<ExecutionPartitionsView::const_iterator::iterator_category,
                             std::input_iterator_tag>);
static_assert(std::ranges::forward_range<ExecutionPartitionsView>);
static_assert(!std::ranges::borrowed_range<ExecutionPartitionsView>);
static_assert(!HasArrowOperator<ExecutionPartitionsView::const_iterator>);

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

// REQ: COMPILE-068
TEST(ExecutionPlanPartitionsView, ExposesModeCountsPathsRangeAndEveryLookupKey) {
  Graph graph{"partition-inspection"};
  test::add_compile_test_default_clock(graph);
  const auto source = graph.op<test::CompileTestOperator>("source");
  const auto target = graph.op<test::CompileTestOperator>("target");
  graph.add_flow(source->output, target->input);

  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto partitions = plan.partitions();
  EXPECT_EQ(partitions.mode(), PartitioningMode::kAutomatic);
  EXPECT_EQ(partitions.authored_count(), 0U);
  EXPECT_EQ(partitions.automatic_count(), 1U);
  ASSERT_EQ(partitions.size(), 1U);
  EXPECT_FALSE(partitions.empty());

  const ExecutionPartitionView iterated = *partitions.begin();
  EXPECT_EQ(iterated.origin(), ExecutionPartitionOrigin::kAutomatic);
  EXPECT_FALSE(iterated.authored_name().has_value());
  EXPECT_TRUE(std::ranges::equal(iterated.members(), plan.execution_partitions().front().members));
  ASSERT_EQ(iterated.operator_paths().size(), 2U);
  EXPECT_EQ(iterated.operator_paths()[0], "source");
  EXPECT_EQ(iterated.operator_paths()[1], "target");
  EXPECT_EQ(iterated.execution_lane(), plan.execution_partitions().front().execution_lane);

  const auto by_id = partitions.lookup(iterated.id());
  ASSERT_TRUE(by_id);
  EXPECT_EQ(by_id->id(), iterated.id());
  const auto by_path = partitions.lookup_for_path("target");
  ASSERT_TRUE(by_path);
  EXPECT_EQ(by_path->id(), iterated.id());
  EXPECT_EQ(partitions.of(source).id(), iterated.id());

  const auto operator_record = std::ranges::find(
      plan.operator_parameters(), "target", &OperatorParameterPlanRecord::operator_path);
  ASSERT_NE(operator_record, plan.operator_parameters().end());
  EXPECT_EQ(partitions.of(operator_record->owner).id(), iterated.id());

  EXPECT_EQ(partitions.lookup(ExecutionPartitionId{}).error().code, ErrorCode::kInvalidArgument);
  EXPECT_EQ(partitions.lookup(ExecutionPartitionId{42U}).error().code, ErrorCode::kNotFound);
  EXPECT_EQ(partitions.lookup_named("").error().code, ErrorCode::kInvalidArgument);
  EXPECT_EQ(partitions.lookup_named("automatic#1").error().code, ErrorCode::kNotFound);
  EXPECT_EQ(partitions.lookup_for_path("").error().code, ErrorCode::kInvalidArgument);
  EXPECT_EQ(partitions.lookup_for_path("missing").error().code, ErrorCode::kNotFound);
}

// REQ: COMPILE-068
TEST(ExecutionPlanPartitionsView, ResolvesAuthoredNamesAndEmbeddedCanonicalPaths) {
  Graph graph{"partition-inspection-authored"};
  test::add_compile_test_default_clock(graph);
  const auto source = graph.op<test::CompileTestOperator>("source");
  const auto target = graph.op<test::CompileTestOperator>("target");
  graph.add_flow(source->output, target->input);
  graph.partition("application").include_all();

  const ExecutionPlan manual = compile(graph);
  ASSERT_TRUE(manual.ok()) << manual.json();
  const auto partitions = manual.partitions();
  EXPECT_EQ(partitions.mode(), PartitioningMode::kManual);
  EXPECT_EQ(partitions.authored_count(), 1U);
  EXPECT_EQ(partitions.automatic_count(), 0U);
  const auto named = partitions.lookup_named("application");
  ASSERT_TRUE(named);
  EXPECT_EQ(named->origin(), ExecutionPartitionOrigin::kAuthored);
  ASSERT_TRUE(named->authored_name());
  EXPECT_EQ(*named->authored_name(), "application");

  Graph embedded{"partition-inspection-embedded"};
  test::add_compile_test_default_clock(embedded);
  (void)embedded.embed<ManualPartitionEmbeddedGraph>("pipeline");
  const ExecutionPlan embedded_plan = compile(embedded);
  ASSERT_TRUE(embedded_plan.ok()) << embedded_plan.json();
  const auto nested = embedded_plan.partitions().lookup_for_path("pipeline/nested");
  ASSERT_TRUE(nested);
  EXPECT_EQ(nested->operator_paths().front(), "pipeline/nested");
}

// REQ: COMPILE-068
TEST(ExecutionPlanPartitionsView, EncodesDecodedPathSegmentsLosslesslyAndRoundTripsLookups) {
  Graph graph{"partition-inspection-encoded-paths"};
  test::add_compile_test_default_clock(graph);
  (void)graph.op<test::CompileTestOperator>("camera/left%\xC3\xA9");
  (void)graph.embed<PartitionInspectionNamedOperatorGraph>("a", "b/c");
  (void)graph.embed<PartitionInspectionNamedOperatorGraph>("a/b", "c");
  (void)graph.embed<PartitionInspectionNamedOperatorGraph>("\xE8\x9E\x8D\xE5\x90\x88/%",
                                                           "op/\xE8\x9E\x8D\xE5\x90\x88%");

  const ExecutionPlan plan = compile(graph);

  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto partitions = plan.partitions();
  constexpr std::string_view kRoot{"camera%2Fleft%25%C3%A9"};
  constexpr std::string_view kFirst{"a/b%2Fc"};
  constexpr std::string_view kSecond{"a%2Fb/c"};
  constexpr std::string_view kUtf8{"%E8%9E%8D%E5%90%88%2F%25/op%2F%E8%9E%8D%E5%90%88%25"};
  const auto root = partitions.lookup_for_path(kRoot);
  const auto first = partitions.lookup_for_path(kFirst);
  const auto second = partitions.lookup_for_path(kSecond);
  const auto utf8 = partitions.lookup_for_path(kUtf8);
  ASSERT_TRUE(root);
  ASSERT_TRUE(first);
  ASSERT_TRUE(second);
  ASSERT_TRUE(utf8);
  EXPECT_NE(first->id(), second->id());
  EXPECT_EQ(first->operator_paths().front(), kFirst);
  EXPECT_EQ(second->operator_paths().front(), kSecond);
  EXPECT_EQ(utf8->operator_paths().front(), kUtf8);
  EXPECT_EQ(partitions.lookup_for_path("a/b/c").error().code, ErrorCode::kNotFound);

  const auto retained = detail::ExecutionPlanAccess::retain(plan);
  ASSERT_TRUE(retained);
  auto accepted = std::make_shared<detail::ExecutionPlanState>(*retained);
  const auto finalized = detail::finalize_plan_artifact(*accepted);
  ASSERT_TRUE(finalized) << finalized.error().message;
  EXPECT_NE(accepted->json.find(std::string{kFirst}), std::string::npos);

  auto malformed = std::make_shared<detail::ExecutionPlanState>(*retained);
  const auto encoded = std::ranges::find(
      malformed->execution_partitions, kFirst, [](const auto& partition) -> std::string_view {
        return partition.operator_paths.empty() ? std::string_view{} : partition.operator_paths[0];
      });
  ASSERT_NE(encoded, malformed->execution_partitions.end());
  encoded->operator_paths.front() = "a/b/c";
  const auto rejected = detail::finalize_plan_artifact(*malformed);
  ASSERT_FALSE(rejected);
  EXPECT_EQ(rejected.error().message, "PARTITION_INSPECTION_PLAN_INVALID");
}

// REQ: COMPILE-068
TEST(ExecutionPlanPartitionsView, ResolvesRootHandleCapturedDuringLazyComposition) {
  PartitionInspectionComposedHandleGraph graph{"partition-inspection-composed-handle"};

  const ExecutionPlan plan = compile(graph);

  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_TRUE(graph.source().has_value());
  const auto partition = plan.partitions().of(*graph.source());
  ASSERT_EQ(partition.operator_paths().size(), 1U);
  EXPECT_EQ(partition.operator_paths().front(), "source");
}

// REQ: COMPILE-068
TEST(ExecutionPlanPartitionsView, RetainsIteratorsElementsAndAccessorSpansAcrossAssignment) {
  Graph graph{"partition-inspection-lifetime"};
  test::add_compile_test_default_clock(graph);
  const auto source = graph.op<test::CompileTestOperator>("source");
  const auto target = graph.op<test::CompileTestOperator>("target");
  graph.add_flow(source->output, target->input);

  ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  auto partitions = plan.partitions();
  auto retained_iterator = partitions.begin();
  const auto retained_end = partitions.end();
  const ExecutionPartitionView retained_element = *retained_iterator;
  const auto retained_members = retained_element.members();
  const auto retained_paths = retained_element.operator_paths();

  plan = ExecutionPlan{};
  ASSERT_EQ(retained_members.size(), 2U);
  ASSERT_EQ(retained_paths.size(), 2U);
  EXPECT_EQ(retained_paths.front(), "source");
  EXPECT_EQ(partitions.size(), 1U);
  EXPECT_EQ((*retained_iterator).id(), retained_element.id());
  ++retained_iterator;
  EXPECT_EQ(retained_iterator, retained_end);
}

// REQ: COMPILE-068
TEST(ExecutionPlanPartitionsView, DistinguishesInvalidForeignStaleAndUnknownOperatorIdentity) {
  Graph graph{"partition-inspection-errors"};
  test::add_compile_test_default_clock(graph);
  const auto source = graph.op<test::CompileTestOperator>("source");
  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto partitions = plan.partitions();

  Graph foreign_graph{"partition-inspection-foreign"};
  test::add_compile_test_default_clock(foreign_graph);
  const auto foreign = foreign_graph.op<test::CompileTestOperator>("foreign");
  try {
    (void)partitions.of(foreign);
    FAIL() << "foreign Graph handle unexpectedly resolved";
  } catch (const RuntimeError& error) {
    EXPECT_EQ(error.code(), ErrorCode::kIncompatible);
  }

  const auto operator_record = plan.operator_parameters().front();
  OperatorRef unknown = operator_record.owner;
  unknown.id = 42U;
  try {
    (void)partitions.of(unknown);
    FAIL() << "unknown OperatorRef unexpectedly resolved";
  } catch (const RuntimeError& error) {
    EXPECT_EQ(error.code(), ErrorCode::kNotFound);
  }

  OperatorRef stale = operator_record.owner;
  ++stale.plan_generation;
  try {
    (void)partitions.of(stale);
    FAIL() << "stale OperatorRef unexpectedly resolved";
  } catch (const RuntimeError& error) {
    EXPECT_EQ(error.code(), ErrorCode::kStaleOperatorRef);
  }

  try {
    (void)ExecutionPlan{}.partitions();
    FAIL() << "empty plan unexpectedly published a partition view";
  } catch (const RuntimeError& error) {
    EXPECT_EQ(error.code(), ErrorCode::kInvalidArgument);
  }

  OperatorRef invalid = operator_record.owner;
  invalid.run_generation = 1U;
  try {
    (void)partitions.of(invalid);
    FAIL() << "run-bound OperatorRef unexpectedly resolved";
  } catch (const RuntimeError& error) {
    EXPECT_EQ(error.code(), ErrorCode::kInvalidArgument);
  }

  (void)graph.op<test::CompileTestOperator>("later");
  try {
    (void)partitions.of(source);
    FAIL() << "stale root Graph handle unexpectedly resolved";
  } catch (const RuntimeError& error) {
    EXPECT_EQ(error.code(), ErrorCode::kStaleOperatorRef);
  }
}

// REQ: COMPILE-068
TEST(ExecutionPlanPartitionsView, CopyIterationAndLookupsAllocateNoMemory) {
  Graph graph{"partition-inspection-allocation"};
  test::add_compile_test_default_clock(graph);
  const auto source = graph.op<test::CompileTestOperator>("source");
  graph.partition("application").include_all();
  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();

  const auto warm = plan.partitions();
  ASSERT_TRUE(warm.lookup_named("application"));
  ASSERT_TRUE(warm.lookup_for_path("source"));
  std::size_t allocation_count{};
  bool observed{};
  {
    const test::ScopedAllocationCounter allocations;
    const auto partitions = plan.partitions();
    // Exercise copying the non-owning view without allocating.
    // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
    const auto copy = partitions;
    auto iterator = copy.begin();
    const auto element = *iterator;
    const auto by_id = copy.lookup(element.id());
    const auto by_name = copy.lookup_named("application");
    const auto by_path = copy.lookup_for_path("source");
    const auto by_ref = copy.of(plan.operator_parameters().front().owner);
    const auto by_handle = copy.of(source);
    const auto missing = copy.lookup_for_path("missing");
    observed = by_id && by_name && by_path && by_ref.id() == element.id() &&
               by_handle.id() == element.id() && !missing && copy.end() != iterator;
    allocation_count = allocations.count();
  }
  EXPECT_TRUE(observed);
  EXPECT_EQ(allocation_count, 0U);
}

// REQ: COMPILE-068
TEST(ExecutionPlanPartitionsView, AcceptedZeroWorkReturnsValidEmptyView) {
  Graph graph{"partition-inspection-zero-work"};

  const ExecutionPlan plan = compile(graph);

  ASSERT_TRUE(plan.ok()) << plan.json();
  EXPECT_EQ(plan.partitioning_plan().mode, PartitioningMode::kAutomatic);
  EXPECT_EQ(plan.partitioning_plan().authored_partition_count, 0U);
  EXPECT_EQ(plan.partitioning_plan().automatic_partition_count, 0U);
  EXPECT_TRUE(plan.execution_partitions().empty());
  EXPECT_TRUE(plan.partition_decisions().empty());
  EXPECT_TRUE(plan.schedule_edges().empty());
  const auto partitions = plan.partitions();
  EXPECT_EQ(partitions.mode(), PartitioningMode::kAutomatic);
  EXPECT_EQ(partitions.authored_count(), 0U);
  EXPECT_EQ(partitions.automatic_count(), 0U);
  EXPECT_EQ(partitions.size(), 0U);
  EXPECT_TRUE(partitions.empty());
  EXPECT_EQ(partitions.begin(), partitions.end());
}

// REQ: COMPILE-068
TEST(ExecutionPlanPartitionsView, ArtifactValidationRejectsMalformedModeCountsPathsAndDecisions) {
  Graph graph{"partition-inspection-artifact-validation"};
  test::add_compile_test_default_clock(graph);
  const auto source = graph.op<test::CompileTestOperator>("source");
  const auto sink = graph.op<test::CompileTestOperator>("sink");
  graph.add_flow(source->output, sink->input);
  graph.partition("anchor").add(source, sink);
  graph.auto_partition_remaining();
  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.partitioning_plan().mode, PartitioningMode::kConstrainedAutomatic);
  ASSERT_EQ(plan.partitioning_plan().automatic_partition_count, 0U);
  const auto accepted = detail::ExecutionPlanAccess::retain(plan);
  ASSERT_TRUE(accepted);

  const auto expect_rejected = [&](auto mutate) {
    auto malformed = std::make_shared<detail::ExecutionPlanState>(*accepted);
    malformed->digest = {};
    malformed->json.clear();
    mutate(*malformed);
    const auto finalized = detail::finalize_plan_artifact(*malformed);
    ASSERT_FALSE(finalized);
    EXPECT_EQ(finalized.error().code, ErrorCode::kInvalidArgument);
    EXPECT_EQ(finalized.error().message, "PARTITION_INSPECTION_PLAN_INVALID");
    EXPECT_EQ(malformed->digest, PlanDigest{});
    EXPECT_TRUE(malformed->json.empty());
  };
  expect_rejected([](auto& state) {
    // The out-of-range value exercises fail-closed artifact validation.
    // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange)
    state.partitioning_plan.mode = static_cast<PartitioningMode>(255U);
  });
  expect_rejected([](auto& state) { ++state.partitioning_plan.authored_partition_count; });
  expect_rejected(
      [](auto& state) { state.execution_partitions.front().operator_paths.front() = "missing"; });
  expect_rejected([](auto& state) {
    // The out-of-range value exercises fail-closed artifact validation.
    // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange)
    state.partition_decisions.front().decision = static_cast<PartitionDecisionKind>(255U);
  });
  expect_rejected([](auto& state) {
    std::reverse(state.execution_partitions.front().members.begin(),
                 state.execution_partitions.front().members.end());
  });
  expect_rejected([](auto& state) {
    std::reverse(state.execution_partitions.front().operator_paths.begin(),
                 state.execution_partitions.front().operator_paths.end());
  });
  expect_rejected(
      [](auto& state) { std::swap(state.partition_decisions[0], state.partition_decisions[1]); });
  expect_rejected([](auto& state) { state.partitioning_plan.mode = PartitioningMode::kManual; });
  expect_rejected([](auto& state) {
    state.execution_partition_index_by_operator_path.front().table_index = 0U;
  });
}

// Imported and exported topics remain explicit automatic endpoint-only boundaries. Their concrete
// Graph edges are the exact crossing-edge evidence exposed through the accepted public view.
// REQ: COMPILE-068
TEST(CompilePartitionAuthoring, ExternalEndpointsRetainEndpointOnlyAutomaticPartitions) {
  const auto expect_endpoint_boundary = [](const ExecutionPlan& plan,
                                           VertexKind endpoint_kind,
                                           std::string_view operator_path,
                                           bool endpoint_is_source) {
    ASSERT_TRUE(plan.ok()) << plan.json();
    ASSERT_EQ(plan.execution_partitions().size(), 2U);
    const auto partitions = plan.partitions();
    EXPECT_EQ(partitions.mode(), PartitioningMode::kAutomatic);
    EXPECT_EQ(partitions.authored_count(), 0U);
    EXPECT_EQ(partitions.automatic_count(), 2U);

    const auto endpoint_record = std::ranges::find_if(
        plan.execution_partitions(),
        [](const auto& partition) { return partition.operator_paths.empty(); });
    ASSERT_NE(endpoint_record, plan.execution_partitions().end());
    ASSERT_EQ(endpoint_record->members.size(), 1U);
    const auto endpoint_vertex = std::ranges::find(
        plan.graph().vertices(), endpoint_record->members.front(), &VertexRecord::id);
    ASSERT_NE(endpoint_vertex, plan.graph().vertices().end());
    EXPECT_EQ(endpoint_vertex->kind, endpoint_kind);

    const auto endpoint = partitions.lookup(endpoint_record->id);
    ASSERT_TRUE(endpoint);
    EXPECT_EQ(endpoint->origin(), ExecutionPartitionOrigin::kAutomatic);
    EXPECT_FALSE(endpoint->authored_name().has_value());
    EXPECT_TRUE(endpoint->operator_paths().empty());
    EXPECT_TRUE(std::ranges::equal(endpoint->members(), endpoint_record->members));

    const auto operator_partition = partitions.lookup_for_path(operator_path);
    ASSERT_TRUE(operator_partition);
    EXPECT_EQ(operator_partition->origin(), ExecutionPartitionOrigin::kAutomatic);
    EXPECT_FALSE(operator_partition->authored_name().has_value());
    ASSERT_EQ(operator_partition->operator_paths().size(), 1U);
    EXPECT_EQ(operator_partition->operator_paths().front(), operator_path);
    ASSERT_EQ(operator_partition->members().size(), 1U);
    const auto operator_vertex =
        std::ranges::find_if(plan.graph().vertices(), [&](const auto& vertex) {
          return vertex.kind == VertexKind::kOperator && vertex.name == operator_path;
        });
    ASSERT_NE(operator_vertex, plan.graph().vertices().end());
    EXPECT_EQ(operator_partition->members().front(), operator_vertex->id);

    ASSERT_EQ(plan.schedule_edges().size(), 1U);
    ASSERT_EQ(plan.graph().edges().size(), 1U);
    EXPECT_EQ(plan.schedule_edges().front().edge, plan.graph().edges().front().id);
    EXPECT_EQ(plan.schedule_edges().front().source,
              endpoint_is_source ? endpoint->id() : operator_partition->id());
    EXPECT_EQ(plan.schedule_edges().front().destination,
              endpoint_is_source ? operator_partition->id() : endpoint->id());
  };

  {
    Graph graph{"automatic-import-endpoint"};
    test::add_compile_test_default_clock(graph);
    auto imported = graph.import_topic<std::uint32_t>("holomq://automatic/import");
    const auto sink = graph.op<test::CompileTestOperator>("sink");
    graph.add_flow(imported.output(), sink->input);
    expect_endpoint_boundary(compile(graph), VertexKind::kImportedTopic, "sink", true);
  }
  {
    Graph graph{"automatic-export-endpoint"};
    test::add_compile_test_default_clock(graph);
    const auto source = graph.op<test::CompileTestOperator>("source");
    auto exported = graph.export_topic<std::uint32_t>("holomq://automatic/export");
    graph.add_flow(source->output, exported.input());
    expect_endpoint_boundary(compile(graph), VertexKind::kExportedTopic, "source", false);
  }
}

// Same-stage lifecycle ordering is retained independently of data-path partition topology.
// REQ: COMPILE-068
TEST(CompilePartitionAuthoring, LifecycleOrderDoesNotInventAutomaticPartitionBoundary) {
  Graph graph{"automatic-lifecycle-order"};
  test::add_compile_test_default_clock(graph);
  const auto source = graph.op<test::CompileTestOperator>("source");
  const auto sink = graph.op<test::CompileTestOperator>("sink");
  graph.add_flow(source->output, sink->input);
  graph.lifecycle().before(sink, source, LifecycleStage::kStart);

  const ExecutionPlan plan = compile(graph);

  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.execution_partitions().size(), 1U);
  EXPECT_EQ(plan.execution_partitions().front().operator_paths,
            (std::vector<std::string>{"sink", "source"}));
  EXPECT_TRUE(plan.schedule_edges().empty());
  ASSERT_EQ(plan.lifecycle_dependencies().size(), 1U);
  const auto source_record = std::ranges::find(
      plan.operator_parameters(), "source", &OperatorParameterPlanRecord::operator_path);
  const auto sink_record = std::ranges::find(
      plan.operator_parameters(), "sink", &OperatorParameterPlanRecord::operator_path);
  ASSERT_NE(source_record, plan.operator_parameters().end());
  ASSERT_NE(sink_record, plan.operator_parameters().end());
  const auto& dependency = plan.lifecycle_dependencies().front();
  EXPECT_EQ(dependency.before, sink_record->owner);
  EXPECT_EQ(dependency.after, source_record->owner);
  EXPECT_EQ(dependency.stage, LifecycleStage::kStart);
  EXPECT_EQ(dependency.kind, LifecycleEdgeKind::kBefore);
  EXPECT_TRUE(dependency.dependency_key.empty());
  ASSERT_EQ(plan.partition_decisions().size(), 2U);
  const auto* source_decision = partition_decision_for_path(plan, "source");
  const auto* sink_decision = partition_decision_for_path(plan, "sink");
  ASSERT_NE(source_decision, nullptr);
  ASSERT_NE(sink_decision, nullptr);
  EXPECT_EQ(source_decision->decision, PartitionDecisionKind::kStart);
  EXPECT_EQ(source_decision->reasons,
            std::vector<PartitionDecisionReason>{PartitionDecisionReason::kSourceRoot});
  EXPECT_EQ(sink_decision->decision, PartitionDecisionKind::kJoin);
  EXPECT_EQ(sink_decision->reasons,
            std::vector<PartitionDecisionReason>{PartitionDecisionReason::kLinearChainJoin});
  EXPECT_TRUE(std::none_of(plan.partition_decisions().begin(),
                           plan.partition_decisions().end(),
                           [](const auto& decision) {
                             return std::find(decision.reasons.begin(),
                                              decision.reasons.end(),
                                              PartitionDecisionReason::kLifecycleIsolation) !=
                                    decision.reasons.end();
                           }));
}

//   source --> left
//          \-> right
// Production source-chain planning isolates a fan-out vertex and both successor chains.
// REQ: COMPILE-068
TEST(CompilePartitionAuthoring, OneRootForkUsesAutomaticSourceChainCover) {
  Graph graph{"implicit-one-root-fork"};
  test::add_compile_test_default_clock(graph);
  const auto source = graph.op<test::CompileTestOperator>("source");
  const auto left = graph.op<test::CompileTestOperator>("left");
  const auto right = graph.op<test::CompileTestOperator>("right");
  graph.add_flow(source->output, left->input);
  graph.add_flow(source->output, right->input);

  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.execution_partitions().size(), 3U);
  EXPECT_TRUE(std::all_of(plan.execution_partitions().begin(),
                          plan.execution_partitions().end(),
                          [](const auto& partition) {
                            return !partition.authored_name.has_value() &&
                                   partition.members.size() == 1U;
                          }));
  EXPECT_EQ(plan.partitioning_plan().selection_source, PlanSelectionSource::kCoreDefault);
  ASSERT_TRUE(plan.partitioning_plan().accepted_algorithm.has_value());
  EXPECT_EQ(plan.partitioning_plan().accepted_algorithm->id, "holoscan.partition.source_chain");
  EXPECT_EQ(plan.scheduler_plan().execution_lane_count, 3U);
  EXPECT_EQ(plan.scheduler_plan().max_concurrent_activations, 3U);
  EXPECT_TRUE(plan.scheduler_plan().activation_overlap);
  EXPECT_EQ(plan.scheduler_plan().worker_count, 3U);
  EXPECT_EQ(plan.scheduler_plan().intra_path_parallelism, 1U);

  const auto* const state = detail::ExecutionPlanAccess::state(plan);
  ASSERT_NE(state, nullptr);
  ASSERT_EQ(state->partition_runners.size(), 3U);
  EXPECT_TRUE(std::all_of(
      state->partition_runners.begin(), state->partition_runners.end(), [](const auto& runner) {
        return runner.operators.size() == 1U && runner.operators.front().topology_root &&
               runner.internal_edges.empty();
      }));
  ASSERT_EQ(plan.schedule_edges().size(), 2U);
  const auto* const source_partition = partition_for_path(plan, "source");
  const auto* const left_partition = partition_for_path(plan, "left");
  const auto* const right_partition = partition_for_path(plan, "right");
  ASSERT_NE(source_partition, nullptr);
  ASSERT_NE(left_partition, nullptr);
  ASSERT_NE(right_partition, nullptr);
  EXPECT_EQ(source_partition->operator_paths, std::vector<std::string>{"source"});
  EXPECT_EQ(left_partition->operator_paths, std::vector<std::string>{"left"});
  EXPECT_EQ(right_partition->operator_paths, std::vector<std::string>{"right"});
  const auto* const left_decision = partition_decision_for_path(plan, "left");
  const auto* const right_decision = partition_decision_for_path(plan, "right");
  ASSERT_NE(left_decision, nullptr);
  ASSERT_NE(right_decision, nullptr);
  EXPECT_EQ(left_decision->decision, PartitionDecisionKind::kFanOutLegSplit);
  EXPECT_EQ(right_decision->decision, PartitionDecisionKind::kFanOutLegSplit);
  EXPECT_TRUE(has_partition_reason(*left_decision, PartitionDecisionReason::kFanOutLeg));
  EXPECT_TRUE(has_partition_reason(*right_decision, PartitionDecisionReason::kFanOutLeg));
  expect_linear_automatic_partitions_and_crossing_edges(plan);
}

//   left-source  --> join <-- right-source
// Production source-chain planning gives both roots and the fan-in join distinct partitions.
// REQ: COMPILE-068
TEST(CompilePartitionAuthoring, TwoRootsJoiningUseThreeAutomaticLinearPartitions) {
  Graph graph{"automatic-two-root-join"};
  test::add_compile_test_default_clock(graph);
  const auto left = graph.op<test::CompileTestOperator>("left-source");
  const auto right = graph.op<test::CompileTestOperator>("right-source");
  const auto join = graph.op<PartitionTopologyJoinOperator>("join");
  graph.add_flow(left->output, join->first);
  graph.add_flow(right->output, join->second);

  const ExecutionPlan plan = compile(graph);

  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.execution_partitions().size(), 3U);
  ASSERT_EQ(plan.schedule_edges().size(), 2U);
  const auto* const join_partition = partition_for_path(plan, "join");
  ASSERT_NE(join_partition, nullptr);
  EXPECT_EQ(join_partition->operator_paths, std::vector<std::string>{"join"});
  const auto* const join_decision = partition_decision_for_path(plan, "join");
  ASSERT_NE(join_decision, nullptr);
  EXPECT_EQ(join_decision->decision, PartitionDecisionKind::kFanInSplit);
  EXPECT_TRUE(has_partition_reason(*join_decision, PartitionDecisionReason::kFanIn));
  expect_linear_automatic_partitions_and_crossing_edges(plan);
}

//   source --> left  --> join
//          \-> right --/
// The public plan and digest are canonical across declaration and edge insertion permutations.
// REQ: COMPILE-068
TEST(CompilePartitionAuthoring, DiamondUsesFourDeterministicAutomaticLinearPartitions) {
  const ExecutionPlan forward = compile_automatic_diamond(false, false);
  const ExecutionPlan reverse_declarations = compile_automatic_diamond(true, false);
  const ExecutionPlan reverse_edges = compile_automatic_diamond(false, true);
  const ExecutionPlan reverse_both = compile_automatic_diamond(true, true);

  for (const ExecutionPlan* const plan :
       {&forward, &reverse_declarations, &reverse_edges, &reverse_both}) {
    ASSERT_TRUE(plan->ok()) << plan->json();
    ASSERT_EQ(plan->execution_partitions().size(), 4U);
    ASSERT_EQ(plan->schedule_edges().size(), 4U);
    const auto* const join_decision = partition_decision_for_path(*plan, "join");
    ASSERT_NE(join_decision, nullptr);
    EXPECT_EQ(join_decision->decision, PartitionDecisionKind::kFanInSplit);
    EXPECT_TRUE(has_partition_reason(*join_decision, PartitionDecisionReason::kFanIn));
    expect_linear_automatic_partitions_and_crossing_edges(*plan);
  }
  EXPECT_EQ(forward.json(), reverse_declarations.json());
  EXPECT_EQ(forward.digest(), reverse_declarations.digest());
  EXPECT_EQ(forward.json(), reverse_edges.json());
  EXPECT_EQ(forward.digest(), reverse_edges.digest());
  EXPECT_EQ(forward.json(), reverse_both.json());
  EXPECT_EQ(forward.digest(), reverse_both.digest());
}

// Two concrete port edges between one ordered Operator pair remain one structural successor.
// REQ: COMPILE-068
TEST(CompilePartitionAuthoring, ParallelEdgesDoNotCreateStructuralBranch) {
  Graph graph{"automatic-parallel-edges"};
  test::add_compile_test_default_clock(graph);
  const auto source = graph.op<test::CompileTestOperator>("source");
  const auto join = graph.op<PartitionTopologyJoinOperator>("join");
  graph.add_flow(source->output, join->first);
  graph.add_flow(source->output, join->second);

  const ExecutionPlan plan = compile(graph);

  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.execution_partitions().size(), 1U);
  EXPECT_EQ(plan.execution_partitions().front().operator_paths,
            (std::vector<std::string>{"join", "source"}));
  EXPECT_TRUE(plan.schedule_edges().empty());
  const auto* const join_decision = partition_decision_for_path(plan, "join");
  ASSERT_NE(join_decision, nullptr);
  EXPECT_EQ(join_decision->decision, PartitionDecisionKind::kJoin);
  EXPECT_TRUE(has_partition_reason(*join_decision, PartitionDecisionReason::kLinearChainJoin));
  EXPECT_FALSE(has_partition_reason(*join_decision, PartitionDecisionReason::kFanIn));
  expect_linear_automatic_partitions_and_crossing_edges(plan);
}

// The exact left anchor is adjacent to both structural branch points in this diamond. It must not
// grow, while the automatic source, right leg, and fan-in join remain distinct linear chains.
// REQ: COMPILE-068
TEST(CompilePartitionAuthoring, ExactAnchorBesideFanOutAndFanInKeepsLinearRemainder) {
  Graph graph{"automatic-diamond-anchor"};
  test::add_compile_test_default_clock(graph);
  const auto source = graph.op<test::CompileTestOperator>("source");
  const auto left = graph.op<test::CompileTestOperator>("left");
  const auto right = graph.op<test::CompileTestOperator>("right");
  const auto join = graph.op<PartitionTopologyJoinOperator>("join");
  graph.add_flow(source->output, left->input);
  graph.add_flow(source->output, right->input);
  graph.add_flow(left->output, join->first);
  graph.add_flow(right->output, join->second);
  graph.partition("left-anchor").add(left);
  graph.auto_partition_remaining();

  const ExecutionPlan plan = compile(graph);

  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.execution_partitions().size(), 4U);
  EXPECT_EQ(plan.partitioning_plan().mode, PartitioningMode::kConstrainedAutomatic);
  EXPECT_EQ(plan.partitioning_plan().authored_partition_count, 1U);
  EXPECT_EQ(plan.partitioning_plan().automatic_partition_count, 3U);
  const auto* const anchor = partition_for_path(plan, "left");
  ASSERT_NE(anchor, nullptr);
  EXPECT_EQ(anchor->authored_name, std::optional<std::string>{"left-anchor"});
  EXPECT_EQ(anchor->operator_paths, std::vector<std::string>{"left"});
  const auto* const right_decision = partition_decision_for_path(plan, "right");
  const auto* const join_decision = partition_decision_for_path(plan, "join");
  ASSERT_NE(right_decision, nullptr);
  ASSERT_NE(join_decision, nullptr);
  EXPECT_EQ(right_decision->decision, PartitionDecisionKind::kFanOutLegSplit);
  EXPECT_TRUE(has_partition_reason(*right_decision, PartitionDecisionReason::kFanOutLeg));
  EXPECT_EQ(join_decision->decision, PartitionDecisionKind::kFanInSplit);
  EXPECT_TRUE(has_partition_reason(*join_decision, PartitionDecisionReason::kFanIn));
  ASSERT_EQ(plan.schedule_edges().size(), 4U);
  expect_linear_automatic_partitions_and_crossing_edges(plan);
}

//   source --> each
//          \-> any-input
// Ordinary input triggers remain successors rather than manufacturing independent temporal roots,
// while structural fan-out still produces three automatic source-chain partitions.
// REQ: COMPILE-068
TEST(CompilePartitionAuthoring, OrdinaryInputTriggersRetainOneIndependentReadinessRoot) {
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
  ASSERT_EQ(plan.execution_partitions().size(), 3U);

  const auto* const state = detail::ExecutionPlanAccess::state(plan);
  ASSERT_NE(state, nullptr);
  const auto topology_roots = std::accumulate(
      state->partition_runners.begin(),
      state->partition_runners.end(),
      std::size_t{0U},
      [](std::size_t count, const auto& runner) {
        return count + static_cast<std::size_t>(
                           std::count_if(runner.operators.begin(),
                                         runner.operators.end(),
                                         [](const auto& member) { return member.topology_root; }));
      });
  EXPECT_EQ(topology_roots, 3U);
  EXPECT_EQ(std::count_if(plan.partition_decisions().begin(),
                          plan.partition_decisions().end(),
                          [](const auto& decision) {
                            return std::find(decision.reasons.begin(),
                                             decision.reasons.end(),
                                             PartitionDecisionReason::kSourceRoot) !=
                                   decision.reasons.end();
                          }),
            1);
  EXPECT_TRUE(std::none_of(plan.partition_decisions().begin(),
                           plan.partition_decisions().end(),
                           [](const auto& decision) {
                             return std::find(decision.reasons.begin(),
                                              decision.reasons.end(),
                                              PartitionDecisionReason::kReadinessIsolation) !=
                                        decision.reasons.end() ||
                                    std::find(decision.reasons.begin(),
                                              decision.reasons.end(),
                                              PartitionDecisionReason::kIndependentReadinessRoot) !=
                                        decision.reasons.end();
                           }));
}

//   camera-source      imu-source
// Two independent roots are automatically covered by two production source chains.
// REQ: COMPILE-068
TEST(CompilePartitionAuthoring, UnpartitionedMultiRootUsesAutomaticSourceChains) {
  Graph graph{"implicit-multi-root-rejected"};
  test::add_compile_test_default_clock(graph);
  (void)graph.op<test::CompileTestOperator>("camera-source");
  (void)graph.op<test::CompileTestOperator>("imu-source");

  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.execution_partitions().size(), 2U);
  ASSERT_TRUE(plan.partitioning_plan().accepted_algorithm.has_value());
  EXPECT_EQ(plan.partitioning_plan().accepted_algorithm->id, "holoscan.partition.source_chain");
  EXPECT_EQ(plan.partitioning_plan().selection_source, PlanSelectionSource::kCoreDefault);
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
  ASSERT_TRUE(plan.partitioning_plan().requested_algorithm.has_value());
  ASSERT_TRUE(plan.partitioning_plan().accepted_algorithm.has_value());
  EXPECT_EQ(plan.partitioning_plan().requested_algorithm->id, "holoscan.partition.serial_baseline");
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

// REQ: COMPILE-068
TEST(ExecutionPlanPartitionsView,
     ArtifactValidationRejectsMissingExtraAndWrongCrossingScheduleEdges) {
  const ExecutionPlan plan = compile_automatic_diamond(false, false);
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_GE(plan.schedule_edges().size(), 2U);
  const auto accepted = detail::ExecutionPlanAccess::retain(plan);
  ASSERT_TRUE(accepted);

  const auto expect_rejected = [&](auto mutate) {
    auto malformed = std::make_shared<detail::ExecutionPlanState>(*accepted);
    mutate(*malformed);
    const auto finalized = detail::finalize_plan_artifact(*malformed);
    ASSERT_FALSE(finalized);
    EXPECT_EQ(finalized.error().code, ErrorCode::kInvalidArgument);
    EXPECT_EQ(finalized.error().message, "PARTITION_INSPECTION_PLAN_INVALID");
  };
  expect_rejected([](auto& state) { state.schedule_edges.pop_back(); });
  expect_rejected([](auto& state) { state.schedule_edges.push_back(state.schedule_edges.back()); });
  expect_rejected([](auto& state) {
    state.schedule_edges.front().destination = state.schedule_edges.front().source;
  });
  expect_rejected([](auto& state) { std::swap(state.schedule_edges[0], state.schedule_edges[1]); });
}

// REQ: COMPILE-092
TEST(ExecutionPlanOperatorHosts, ArtifactValidationRejectsMalformedHostInventory) {
  Graph graph{"operator-host-artifact-validation"};
  test::add_compile_test_default_clock(graph);
  static_cast<void>(graph.op<test::CompileTestOperator>("source"));
  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto accepted = detail::ExecutionPlanAccess::retain(plan);
  ASSERT_TRUE(accepted);
  ASSERT_EQ(accepted->operator_hosts.size(), 1U);

  const auto expect_rejected = [&](auto mutate) {
    auto malformed = std::make_shared<detail::ExecutionPlanState>(*accepted);
    mutate(*malformed);
    const auto finalized = detail::finalize_plan_artifact(*malformed);
    ASSERT_FALSE(finalized);
    EXPECT_EQ(finalized.error().code, ErrorCode::kInvalidArgument);
    EXPECT_EQ(finalized.error().message, "OPERATOR_HOST_PLAN_INVALID");
  };

  // Key assertion: missing, unknown-kind, and stale-generation rows all fail finalization.
  expect_rejected([](auto& state) { state.operator_hosts.clear(); });
  expect_rejected([](auto& state) {
    // The malformed artifact test intentionally constructs an out-of-domain serialized enum.
    // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange)
    state.operator_hosts.front().kind = static_cast<OperatorHostKind>(255U);
  });
  expect_rejected([](auto& state) { state.operator_hosts.front().owner.plan_generation = 0U; });
}

// REQ: COMPILE-068
TEST(ExecutionPlanPartitionsView,
     EmbeddedInterfaceCrossingScheduleEdgesAreExactAndArtifactValidated) {
  Graph graph{"embedded-interface-partition-crossing"};
  test::add_compile_test_default_clock(graph);
  const auto child = graph.embed<PartitionTopologyEmbeddedSourceGraph>("child");
  const auto left = graph.op<test::CompileTestOperator>("left");
  const auto right = graph.op<test::CompileTestOperator>("right");
  graph.add_flow(child->output, left->input);
  graph.add_flow(child->output, right->input);

  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.graph().interface_bindings().size(), 1U);
  ASSERT_EQ(plan.graph().edges().size(), 2U);
  ASSERT_EQ(plan.schedule_edges().size(), 2U);

  const auto& binding = plan.graph().interface_bindings().front();
  const auto public_port =
      std::find_if(plan.graph().ports().begin(), plan.graph().ports().end(), [&](const auto& port) {
        return port.id == binding.public_port;
      });
  const auto internal_port =
      std::find_if(plan.graph().ports().begin(), plan.graph().ports().end(), [&](const auto& port) {
        return port.id == binding.internal_port;
      });
  ASSERT_NE(public_port, plan.graph().ports().end());
  ASSERT_NE(internal_port, plan.graph().ports().end());
  EXPECT_EQ(public_port->owner_kind, PortOwnerKind::kGraphInterface);
  EXPECT_EQ(internal_port->owner_kind, PortOwnerKind::kOperator);

  const auto* source_partition = partition_for_path(plan, "child/source");
  const auto* left_partition = partition_for_path(plan, "left");
  const auto* right_partition = partition_for_path(plan, "right");
  ASSERT_NE(source_partition, nullptr);
  ASSERT_NE(left_partition, nullptr);
  ASSERT_NE(right_partition, nullptr);
  EXPECT_EQ(plan.graph().edges()[0].source, binding.public_port);
  EXPECT_EQ(plan.graph().edges()[1].source, binding.public_port);
  EXPECT_EQ(plan.schedule_edges()[0].edge, plan.graph().edges()[0].id);
  EXPECT_EQ(plan.schedule_edges()[0].source, source_partition->id);
  EXPECT_EQ(plan.schedule_edges()[0].destination, left_partition->id);
  EXPECT_EQ(plan.schedule_edges()[1].edge, plan.graph().edges()[1].id);
  EXPECT_EQ(plan.schedule_edges()[1].source, source_partition->id);
  EXPECT_EQ(plan.schedule_edges()[1].destination, right_partition->id);

  const auto accepted = detail::ExecutionPlanAccess::retain(plan);
  ASSERT_TRUE(accepted);
  const auto expect_rejected = [&](auto mutate) {
    auto malformed = std::make_shared<detail::ExecutionPlanState>(*accepted);
    malformed->digest = {};
    malformed->json.clear();
    mutate(*malformed);
    const auto finalized = detail::finalize_plan_artifact(*malformed);
    ASSERT_FALSE(finalized);
    EXPECT_EQ(finalized.error().code, ErrorCode::kInvalidArgument);
    EXPECT_EQ(finalized.error().message, "PARTITION_INSPECTION_PLAN_INVALID");
    EXPECT_EQ(malformed->digest, PlanDigest{});
    EXPECT_TRUE(malformed->json.empty());
  };
  expect_rejected([](auto& state) {
    state.schedule_edges.front().source = state.schedule_edges.front().destination;
  });
  expect_rejected([](auto& state) {
    const auto graph_interface =
        std::ranges::find(state.graph.vertices(), VertexKind::kGraphInterface, &VertexRecord::kind);
    ASSERT_NE(graph_interface, state.graph.vertices().end());
    const auto source_partition =
        std::ranges::find_if(state.execution_partitions, [](const auto& partition) {
          return std::ranges::find(partition.operator_paths, "child/source") !=
                 partition.operator_paths.end();
        });
    ASSERT_NE(source_partition, state.execution_partitions.end());
    source_partition->members.insert(
        std::ranges::lower_bound(
            source_partition->members, graph_interface->id.value, {}, &VertexId::value),
        graph_interface->id);
    state.partition_decisions.push_back(PartitionDecisionRecord{
        .vertex = graph_interface->id,
        .partition = source_partition->id,
        .decision = PartitionDecisionKind::kJoin,
    });
    std::ranges::sort(state.partition_decisions, [](const auto& left, const auto& right) {
      return left.vertex.value < right.vertex.value;
    });
  });
}

// REQ: COMPILE-058
TEST(CompilePartitionAuthoring, RejectsManualPartitionsWithUncoveredEmbeddedGraphs) {
  Graph graph{"manual-embedded"};
  test::add_compile_test_default_clock(graph);
  const auto root = graph.op<test::CompileTestOperator>("root");
  graph.partition("root").add(root);
  (void)graph.embed<ManualPartitionEmbeddedGraph>("child");

  const ExecutionPlan plan = compile(graph);
  EXPECT_FALSE(plan.ok());
  EXPECT_TRUE(test::has_diagnostic(plan, "PARTITION_EMBEDDED_INSTANCE_UNASSIGNED"));
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
// REQ: COMPILE-068
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
  EXPECT_EQ(requested_algorithm->id, "holoscan.partition.source_chain");
  EXPECT_TRUE(plan.partitioning_plan().accepted_algorithm.has_value());
  try {
    (void)plan.partitions();
    FAIL() << "rejected plan unexpectedly published a partition view";
  } catch (const RuntimeError& error) {
    EXPECT_EQ(error.code(), ErrorCode::kInvalidArgument);
  }
  EXPECT_NE(plan.clock_catalog().plan_generation, 0U);
  ASSERT_EQ(plan.lifecycle_restart_targets().size(), 1U);
  EXPECT_FALSE(plan.lifecycle_restart_targets().front().effective_profile.has_value());
}

// REQ: COMPILE-001
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
  const ClockPlanRecord* retained_clock = nullptr;
  {
    Graph graph{"retained-plan-views"};
    (void)graph.op<test::CompileTestOperator>("operator");
    (void)graph.add_clock<ManualClock>("clock", std::int64_t{0});
    const ExecutionPlan plan = compile(graph);
    ASSERT_TRUE(plan.ok());
    // Key setup: capture graph and catalog views before the ExecutionPlan goes out of scope.
    retained_graph = plan.graph();
    retained_catalog = plan.clock_catalog();
    retained_clock = retained_catalog.find(ClockId{1U});
    ASSERT_NE(retained_clock, nullptr);
  }

  // Key assertion: graph view and clock catalog remain accessible after plan is destroyed.
  ASSERT_EQ(retained_graph.vertices().size(), 1U);
  ASSERT_EQ(retained_catalog.clocks.size(), 1U);
  EXPECT_NE(retained_catalog.plan_generation, 0U);
  EXPECT_EQ(retained_catalog.clocks.front().id, ClockId{1U});
  // Leave the catalog as the only owner before reading the borrowed lookup result.
  retained_graph = {};
  EXPECT_EQ(retained_catalog.find(ClockId{1U}), retained_clock);
  EXPECT_EQ(retained_clock->id, ClockId{1U});
}

// REQ: COMPILE-007
TEST(ExecutionPlanIdentity, ClockCatalogLookupRejectsInvalidIds) {
  const ClockCatalogView empty;
  EXPECT_EQ(empty.find(ClockId{}), nullptr);
  EXPECT_EQ(empty.find(ClockId{1U}), nullptr);

  Graph graph{"clock-catalog-lookup"};
  (void)graph.op<test::CompileTestOperator>("operator");
  (void)graph.add_clock<ManualClock>("first", std::int64_t{0});
  (void)graph.add_clock<ManualClock>("second", std::int64_t{0});
  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  const ClockCatalogView catalog = plan.clock_catalog();
  ASSERT_EQ(catalog.clocks.size(), 2U);
  EXPECT_EQ(catalog.find(ClockId{}), nullptr);
  EXPECT_EQ(catalog.find(ClockId{1U}), &catalog.clocks[0]);
  EXPECT_EQ(catalog.find(ClockId{2U}), &catalog.clocks[1]);
  EXPECT_EQ(catalog.find(ClockId{3U}), nullptr);
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

// REQ: COMPILE-007
// REQ: COMPILE-025
// REQ: COMPILE-078
// REQ: EXPECTED-030
TEST(ExecutionPlanArtifacts, AdmittedWarningDetailDoesNotChangeAnyPortableFile) {
  ExecutionPlan plan;
  {
    // This is the real admitted warning fixture from
    // CompileLowering.CanonicalizesTopologyReadinessAndClockDomains. No rejected plan
    // is promoted and no severity, structural diagnostic, or admission field is changed.
    Graph graph{"warning-detail-artifacts"};
    (void)graph.op<test::CompileReadinessOperator>("ready");
    const ClockKey notification =
        graph.add_clock<ManualClock>("notification_clock", std::int64_t{0});
    const ClockKey doorbell = graph.add_clock<ManualClock>("doorbell_clock", std::int64_t{0});
    graph.set_default_clock(notification);
    graph.add_clock_conversion(notification, doorbell, AffineClockConversion{.offset_ns = 3});
    plan = compile(graph);
  }
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto rows = plan.diagnostics();
  ASSERT_FALSE(rows.empty());
  EXPECT_TRUE(std::ranges::all_of(
      rows, [](const auto& row) { return row.severity == DiagnosticSeverity::kWarning; }));
  const auto warning = std::ranges::find(rows, "READINESS_SOURCE_UNUSED", &Diagnostic::token);
  ASSERT_NE(warning, rows.end());
  EXPECT_EQ(warning->severity, DiagnosticSeverity::kWarning);
  const auto index = static_cast<std::size_t>(warning - rows.begin());
  const std::string original_warning_report = format_diagnostic(*warning, {});
  const auto* state = detail::ExecutionPlanAccess::state(plan);
  ASSERT_NE(state, nullptr);
  ASSERT_TRUE(state->admitted);

  const auto structural_key = [](const Diagnostic& row) {
    return std::tuple{row.id.value,
                      row.severity,
                      row.token,
                      row.source.graph_generation,
                      row.source.graph.value,
                      row.source.vertex.value,
                      row.source.port.value(),
                      row.service,
                      row.endpoint,
                      row.json_pointer,
                      row.required,
                      row.provided,
                      row.cause_id};
  };
  const ScopedArtifactRoot baseline_root = unique_artifact_root();
  const auto baseline = emit_artifacts(
      plan, ArtifactEmitOptions{.root = baseline_root.path(), .deployment_id = "detail-only"});
  ASSERT_TRUE(baseline) << baseline.error().message;
  ASSERT_TRUE(std::filesystem::is_regular_file(baseline->manifest_path));
  const auto baseline_directory = baseline->manifest_path.parent_path();
  constexpr std::array<std::string_view, 3U> kPortableFiles{
      "execution-plan.json", "endpoint-catalog.json", "profile-manifest.json"};
  std::array<std::string, 3U> baseline_bytes;
  for (std::size_t file = 0U; file < kPortableFiles.size(); ++file) {
    const auto path = baseline_directory / kPortableFiles[file];
    ASSERT_TRUE(std::filesystem::is_regular_file(path));
    baseline_bytes[file] = read_artifact(path);
    ASSERT_FALSE(baseline_bytes[file].empty());
    EXPECT_EQ(baseline_bytes[file].size(), std::filesystem::file_size(path));
  }
  EXPECT_EQ(baseline_bytes.front(), plan.json());

  // Both controls remain genuinely admitted and retain exactly the same structural warning.
  // These context-only public detail values are test mutations, not claims about a producer.
  for (const std::string marker : {"nonsemantic-detail-alpha", "nonsemantic-detail-beta"}) {
    SCOPED_TRACE(marker);
    auto changed = std::make_shared<detail::ExecutionPlanState>(*state);
    changed->diagnostics[index].detail = DiagnosticDetail{
        .operation = DiagnosticOperation::kCompile,
        .error = Error{ErrorCode::kSuccess},
        .operator_path = "ready",
        .mismatches = {DiagnosticMismatch{.field = "schema_path", .provided = marker}},
        .truncated = marker == "nonsemantic-detail-beta"};
    ASSERT_EQ(changed->diagnostics.size(), rows.size());
    for (std::size_t row = 0U; row < rows.size(); ++row) {
      EXPECT_EQ(structural_key(changed->diagnostics[row]), structural_key(rows[row]));
    }
    const auto serialized = detail::finalize_plan_artifact(*changed);
    ASSERT_TRUE(serialized) << serialized.error().message;
    EXPECT_TRUE(changed->admitted);
    EXPECT_EQ(changed->json, plan.json());
    EXPECT_EQ(changed->digest, plan.digest());
    const ExecutionPlan changed_plan = detail::ExecutionPlanAccess::make(changed);
    ASSERT_TRUE(changed_plan.ok());
    EXPECT_NE(format_diagnostic(changed_plan.diagnostics()[index], {}).find(marker),
              std::string::npos);
    EXPECT_EQ(format_diagnostic(plan.diagnostics()[index], {}), original_warning_report);

    const ScopedArtifactRoot changed_root = unique_artifact_root();
    const auto emitted = emit_artifacts(
        changed_plan,
        ArtifactEmitOptions{.root = changed_root.path(), .deployment_id = "detail-only"});
    ASSERT_TRUE(emitted) << emitted.error().message;
    EXPECT_TRUE(emitted->changed);
    ASSERT_TRUE(std::filesystem::is_regular_file(emitted->manifest_path));
    EXPECT_EQ(emitted->catalog_digest, baseline->catalog_digest);
    const auto directory = emitted->manifest_path.parent_path();
    for (std::size_t file = 0U; file < kPortableFiles.size(); ++file) {
      SCOPED_TRACE(kPortableFiles[file]);
      const auto path = directory / kPortableFiles[file];
      ASSERT_TRUE(std::filesystem::is_regular_file(path));
      const std::string actual = read_artifact(path);
      ASSERT_FALSE(actual.empty());
      EXPECT_EQ(actual.size(), std::filesystem::file_size(path));
      // Compare actual file contents. No JSON normalization, field removal, regenerated
      // expected fixture, or comparison of the path names substitutes for these bytes.
      EXPECT_EQ(actual, baseline_bytes[file]);
      EXPECT_EQ(actual.find("nonsemantic-detail-alpha"), std::string::npos);
      EXPECT_EQ(actual.find("nonsemantic-detail-beta"), std::string::npos);
      EXPECT_EQ(actual.find("private projector implementation detail"), std::string::npos);
      EXPECT_EQ(actual.find("implementation_token"), std::string::npos);
    }
    // artifact-manifest.json is the publication commit marker and carries per-file times.
    // It must exist, but its timestamp-bearing bytes are not a portable-plan identity claim.
  }
}

}  // namespace
}  // namespace holoscan
// NOLINTEND(bugprone-unchecked-optional-access)
