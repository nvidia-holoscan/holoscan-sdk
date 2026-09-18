/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <holoscan/core/compile.hpp>

#include "compile_test_graph.hpp"
#include "compiler/authored_partition_membership.hpp"
#include "compiler/execution_plan_state.hpp"
#include "compiler/partition_authority_validation.hpp"
#include "compiler/partition_problem_lowering.hpp"
#include "compiler/plan_artifact_serialization.hpp"
#include "core/graph_compiler_access.hpp"
#include "physical_plan_test_support.hpp"

namespace holoscan {
namespace {

class AtomicCompileLeaf final : public Graph {
 protected:
  void compose() override { (void)op<test::CompileTestOperator>("second"); }
};

class AtomicCompileTree final : public Graph {
 protected:
  void compose() override {
    (void)op<test::CompileTestOperator>("first");
    (void)embed<AtomicCompileLeaf>("nested");
  }
};

class AtomicCompileInput final : public Graph {
 public:
  Input<std::uint32_t> input{"input"};

 protected:
  void compose() override {
    test::add_compile_test_default_clock(*this);
    const auto sink = op<test::CompileTestOperator>("sink");
    export_input(input, sink->input);
  }
};

class AtomicTensorBranches final : public Graph {
 public:
  explicit AtomicTensorBranches(bool reverse = false) : reverse_(reverse) {}

  [[nodiscard]] ClockKey clock_import() const noexcept { return clock_import_; }

 protected:
  void compose() override {
    clock_import_ = import_clock("timeline");
    set_default_clock(clock_import_);
    const auto add_branch = [&](const char* name, const char* sink_name) {
      const auto source = op<detail::physical_plan_test::DeviceTensorOperator>(name);
      const auto sink = op<detail::physical_plan_test::ManifestTensorSinkOperator>(sink_name);
      add_flow(source->tensor, sink->input);
    };
    if (reverse_) {
      add_branch("second", "second-sink");
      add_branch("first", "first-sink");
    } else {
      add_branch("first", "first-sink");
      add_branch("second", "second-sink");
    }
  }

 private:
  bool reverse_{};
  ClockKey clock_import_{};
};

class AtomicTensorInput final : public Graph {
 public:
  Input<Tensor> input{"input"};

 protected:
  void compose() override {
    test::add_compile_test_default_clock(*this);
    const auto sink = op<detail::physical_plan_test::ManifestTensorSinkOperator>("sink");
    export_input(input, sink->input);
  }
};

class AtomicNestedTensorInput final : public Graph {
 public:
  Input<Tensor> input{"input"};

 protected:
  void compose() override {
    const auto nested = embed<AtomicTensorInput>("nested");
    export_input(input, nested->input);
  }
};

[[nodiscard]] auto atomic_green_context_capabilities() {
  return std::array{
      detail::CudaGreenContextCapabilityEvidence{
          .device = DeviceId{0},
          .deployment_device_id = "gpu.test.uuid.atomic.0",
          .total_sm_count = 80U,
          .min_sm_partition_size = 8U,
          .sm_coscheduled_alignment = 8U,
          .max_structured_context_count = 10U,
          .evidence_id = "test.cuda.green_context.capability.atomic.0/v1",
          .probe_status = PlacementProbeStatus::kPassed,
      },
      detail::CudaGreenContextCapabilityEvidence{
          .device = DeviceId{1},
          .deployment_device_id = "gpu.test.uuid.atomic.1",
          .total_sm_count = 80U,
          .min_sm_partition_size = 8U,
          .sm_coscheduled_alignment = 8U,
          .max_structured_context_count = 10U,
          .evidence_id = "test.cuda.green_context.capability.atomic.1/v1",
          .probe_status = PlacementProbeStatus::kPassed,
      },
  };
}

// REQ: COMPILE-075
TEST(CompilePartitionInstances, DeviceDomainsAgreeForManualAndAutomaticRemainder) {
  for (const bool automatic : {false, true}) {
    for (const bool conflict : {false, true}) {
      SCOPED_TRACE(::testing::Message() << "automatic=" << automatic << " conflict=" << conflict);
      const auto make_plan = [&](bool reverse) {
        Graph graph{"root"};
        const auto clock = test::add_compile_test_default_clock(graph);
        const auto child = graph.embed<AtomicTensorBranches>("child", reverse);
        graph.bind_clock(child, child->clock_import(), clock);
        graph.partition("child-owner").add(child);
        if (automatic) {
          (void)graph.op<test::CompileTestOperator>("peer");
          graph.auto_partition_remaining();
        }
        CompileOptions options;
        const TensorOutputDevicePlacement first{"child/first", "tensor", DeviceId{0}};
        const TensorOutputDevicePlacement second{
            "child/second", "tensor", DeviceId{conflict ? 1 : 0}};
        options.deployment.bind_tensor_output_device(reverse ? second : first)
            .bind_tensor_output_device(reverse ? first : second);
        return compile(graph, std::move(options));
      };
      const auto forward = make_plan(false);
      const auto reverse = make_plan(true);
      EXPECT_EQ(forward.digest(), reverse.digest());
      EXPECT_EQ(forward.json(), reverse.json());
      if (conflict) {
        EXPECT_FALSE(forward.ok());
        EXPECT_TRUE(test::has_diagnostic(forward, "PARTITION_ANCHOR_DOMAIN_INCOMPATIBLE"));
        EXPECT_FALSE(forward.partitioning_plan().accepted_algorithm.has_value());
        EXPECT_TRUE(forward.execution_partitions().empty());
        EXPECT_TRUE(forward.execution_lanes().empty());
      } else {
        ASSERT_TRUE(forward.ok()) << forward.json();
        const auto selected = forward.partitions().lookup_named("child-owner");
        ASSERT_TRUE(selected);
        EXPECT_EQ(selected->operator_paths().size(), 4U);
        EXPECT_EQ(selected->graph_members().size(), 1U);
      }
    }
  }
}

// REQ: COMPILE-075
TEST(CompilePartitionInstances, IncludeAllRetainsIndependentDeviceDomains) {
  Graph graph{"root"};
  const auto clock = test::add_compile_test_default_clock(graph);
  const auto child = graph.embed<AtomicTensorBranches>("child");
  graph.bind_clock(child, child->clock_import(), clock);
  graph.partition("all").include_all();
  CompileOptions options;
  options.deployment.bind_tensor_output_device({"child/first", "tensor", DeviceId{0}})
      .bind_tensor_output_device({"child/second", "tensor", DeviceId{1}});
  const auto plan = compile(graph, std::move(options));
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.execution_partitions().size(), 1U);
  EXPECT_EQ(plan.execution_partitions().front().members.size(), 4U);
}

// REQ: COMPILE-075
TEST(CompilePartitionInstances, MixedOwnerFoldsSiblingAndDirectDeviceDomains) {
  for (const bool automatic : {false, true}) {
    for (const int conflict : {0, 1, 2}) {
      SCOPED_TRACE(::testing::Message() << "automatic=" << automatic << " conflict=" << conflict);
      Graph graph{"root"};
      const auto clock = test::add_compile_test_default_clock(graph);
      const auto first = graph.embed<AtomicTensorBranches>("first");
      const auto second = graph.embed<AtomicTensorBranches>("second");
      graph.bind_clock(first, first->clock_import(), clock);
      graph.bind_clock(second, second->clock_import(), clock);
      const auto direct = graph.op<detail::physical_plan_test::DeviceTensorOperator>("direct");
      graph.partition("owner").add(first).add(second).add(direct);
      if (automatic) {
        (void)graph.op<test::CompileTestOperator>("peer");
        graph.auto_partition_remaining();
      }
      CompileOptions options;
      options.deployment.bind_tensor_output_device({"first/first", "tensor", DeviceId{0}})
          .bind_tensor_output_device({"first/second", "tensor", DeviceId{0}})
          .bind_tensor_output_device({"second/first", "tensor", DeviceId{conflict == 1 ? 1 : 0}})
          .bind_tensor_output_device({"second/second", "tensor", DeviceId{conflict == 1 ? 1 : 0}})
          .bind_tensor_output_device({"direct", "tensor", DeviceId{conflict == 2 ? 1 : 0}});
      const auto plan = compile(graph, std::move(options));
      if (conflict != 0) {
        EXPECT_FALSE(plan.ok());
        EXPECT_TRUE(test::has_diagnostic(plan, "PARTITION_ANCHOR_DOMAIN_INCOMPATIBLE"));
        EXPECT_TRUE(plan.execution_partitions().empty());
      } else {
        ASSERT_TRUE(plan.ok()) << plan.json();
        const auto owner = plan.partitions().lookup_named("owner");
        ASSERT_TRUE(owner);
        EXPECT_EQ(owner->graph_members().size(), 2U);
        EXPECT_EQ(owner->operator_paths().size(), 9U);
      }
    }
  }
}

// REQ: COMPILE-075
TEST(CompilePartitionInstances, IndependentOwnersKeepDistinctCompatibleDeviceDomains) {
  for (const bool automatic : {false, true}) {
    Graph graph{"root"};
    const auto clock = test::add_compile_test_default_clock(graph);
    const auto first = graph.embed<AtomicTensorBranches>("first");
    const auto second = graph.embed<AtomicTensorBranches>("second");
    graph.bind_clock(first, first->clock_import(), clock);
    graph.bind_clock(second, second->clock_import(), clock);
    graph.partition("first-owner").add(first);
    graph.partition("second-owner").add(second);
    if (automatic) {
      (void)graph.op<test::CompileTestOperator>("peer");
      graph.auto_partition_remaining();
    }
    CompileOptions options;
    options.deployment.bind_tensor_output_device({"first/first", "tensor", DeviceId{0}})
        .bind_tensor_output_device({"first/second", "tensor", DeviceId{0}})
        .bind_tensor_output_device({"second/first", "tensor", DeviceId{1}})
        .bind_tensor_output_device({"second/second", "tensor", DeviceId{1}});
    const auto plan = compile(graph, std::move(options));
    ASSERT_TRUE(plan.ok()) << plan.json();
    EXPECT_TRUE(plan.partitions().lookup_named("first-owner"));
    EXPECT_TRUE(plan.partitions().lookup_named("second-owner"));
  }
}

// REQ: COMPILE-075
TEST(CompilePartitionInstances, NestedConsumerAndGreenContextDomainsAgreeAcrossCoverModes) {
  const auto capabilities = atomic_green_context_capabilities();
  for (const bool automatic : {false, true}) {
    for (const bool conflict : {false, true}) {
      SCOPED_TRACE(::testing::Message() << "automatic=" << automatic << " conflict=" << conflict);
      Graph graph{"root"};
      test::add_compile_test_default_clock(graph);
      const auto producer = graph.op<detail::physical_plan_test::DeviceTensorOperator>("producer");
      const auto child = graph.embed<AtomicNestedTensorInput>("child");
      graph.add_flow(producer->tensor, child->input);
      graph.partition("child-owner").add(child);
      if (automatic) {
        graph.auto_partition_remaining();
      } else {
        graph.partition("producer-owner").add(producer);
      }
      CompileOptions options;
      options.deployment.bind_tensor_output_device({"producer", "tensor", DeviceId{0}})
          .bind_cuda_green_context("child-owner",
                                   {.device = DeviceId{conflict ? 1 : 0}, .sm_count = 16U});
      auto evidence = detail::runtime_compile_evidence_view();
      evidence.cuda_green_context_capabilities = capabilities;
      const auto plan = detail::compile_with_runtime_evidence(graph, std::move(options), evidence);
      if (conflict) {
        EXPECT_FALSE(plan.ok());
        EXPECT_TRUE(test::has_diagnostic(plan, "PARTITION_ANCHOR_DOMAIN_INCOMPATIBLE"));
        EXPECT_FALSE(plan.partitioning_plan().accepted_algorithm.has_value());
        EXPECT_TRUE(plan.execution_partitions().empty());
        EXPECT_TRUE(plan.execution_lanes().empty());
      } else {
        ASSERT_TRUE(plan.ok()) << plan.json();
        const auto selected = plan.partitions().lookup_named("child-owner");
        ASSERT_TRUE(selected);
        EXPECT_EQ(std::vector<std::string>(selected->operator_paths().begin(),
                                           selected->operator_paths().end()),
                  (std::vector<std::string>{"child/nested/sink"}));
      }
    }
  }
}

// REQ: COMPILE-075
TEST(CompilePartitionInstances, UnassignedEmbeddedDescendantHasCanonicalDiagnosticSource) {
  const auto make_plan = [](bool reverse, bool automatic) {
    Graph graph{"root"};
    test::add_compile_test_default_clock(graph);
    (void)graph.embed<AtomicCompileTree>(reverse ? "right" : "left");
    (void)graph.embed<AtomicCompileTree>(reverse ? "left" : "right");
    graph.partition("owner").add(graph.op<test::CompileTestOperator>("peer"));
    if (automatic) {
      graph.auto_partition_remaining();
    }
    return compile(graph);
  };
  const auto plan = make_plan(false, false);
  const auto reversed = make_plan(true, false);
  EXPECT_FALSE(plan.ok());
  EXPECT_EQ(plan.json(), reversed.json());
  EXPECT_EQ(plan.digest(), reversed.digest());
  const auto diagnostic = std::ranges::find(
      plan.diagnostics(), "PARTITION_EMBEDDED_INSTANCE_UNASSIGNED", &Diagnostic::token);
  ASSERT_NE(diagnostic, plan.diagnostics().end()) << plan.json();
  EXPECT_EQ(plan.partitioning_plan().reason, diagnostic->id);
  EXPECT_FALSE(test::has_diagnostic(plan, "PARTITION_COVER_INVALID"));
  EXPECT_FALSE(test::has_diagnostic(plan, "COMPILER_DIAGNOSTIC_TOKEN_UNREGISTERED"));
  const auto vertex =
      std::ranges::find(plan.graph().vertices(), diagnostic->source.vertex, &VertexRecord::id);
  ASSERT_NE(vertex, plan.graph().vertices().end());
  EXPECT_EQ(vertex->name, "first");
  EXPECT_EQ(vertex->graph, diagnostic->source.graph);
  const auto owner =
      std::ranges::find(plan.graph().graphs(), diagnostic->source.graph, &GraphRecord::id);
  ASSERT_NE(owner, plan.graph().graphs().end());
  EXPECT_EQ(owner->name, "left");
  EXPECT_TRUE(plan.execution_partitions().empty());
  EXPECT_TRUE(plan.execution_lanes().empty());
  const auto automatic = make_plan(false, true);
  EXPECT_TRUE(automatic.ok()) << automatic.json();

  Graph direct{"root"};
  test::add_compile_test_default_clock(direct);
  direct.partition("owner").add(direct.op<test::CompileTestOperator>("first"));
  (void)direct.op<test::CompileTestOperator>("second");
  const auto uncovered_root = compile(direct);
  EXPECT_TRUE(test::has_diagnostic(uncovered_root, "PARTITION_COVER_INVALID"));
  EXPECT_FALSE(test::has_diagnostic(uncovered_root, "PARTITION_EMBEDDED_INSTANCE_UNASSIGNED"));
}

// REQ: COMPILE-075
TEST(CompilePartitionInstances, SelectedInstancePreservesAttachedExternalEndpoint) {
  Graph graph{"root"};
  test::add_compile_test_default_clock(graph);
  const auto child = graph.embed<AtomicCompileInput>("child");
  auto imported = graph.import_topic<std::uint32_t>("holomq://atomic-instance-input");
  graph.add_flow(imported.output(), child->input);
  (void)graph.op<test::CompileTestOperator>("peer");
  graph.partition("owner").add(child);
  graph.auto_partition_remaining();

  const auto plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto selected = plan.partitions().lookup_named("owner");
  ASSERT_TRUE(selected);
  ASSERT_EQ(selected->graph_members().size(), 1U);
  EXPECT_EQ(selected->members().size(), 2U);
  EXPECT_EQ(std::vector<std::string>(selected->operator_paths().begin(),
                                     selected->operator_paths().end()),
            (std::vector<std::string>{"child/sink"}));
  const auto vertices = plan.graph().vertices();
  EXPECT_EQ(
      std::count_if(selected->members().begin(),
                    selected->members().end(),
                    [&](VertexId member) {
                      const auto found = std::ranges::find(vertices, member, &VertexRecord::id);
                      return found != vertices.end() && found->kind == VertexKind::kImportedTopic;
                    }),
      1);
  const auto state = detail::ExecutionPlanAccess::retain(plan);
  EXPECT_TRUE(detail::valid_accepted_partition_authority(*state));
}

// REQ: COMPILE-075
TEST(CompilePartitionInstances, RecursiveIndependentRootsSupportManualAndAutomaticRemainder) {
  for (const bool automatic : {false, true}) {
    Graph graph{"root"};
    test::add_compile_test_default_clock(graph);
    const auto child = graph.embed<AtomicCompileTree>("child");
    const auto peer = graph.op<test::CompileTestOperator>("peer");
    graph.partition("child-owner").add(child);
    if (automatic) {
      graph.auto_partition_remaining();
    } else {
      graph.partition("peer-owner").add(peer);
    }
    const auto plan = compile(graph);
    ASSERT_TRUE(plan.ok()) << plan.json();
    ASSERT_EQ(plan.execution_partitions().size(), 2U);
    const auto selected = plan.partitions().lookup_named("child-owner");
    ASSERT_TRUE(selected);
    EXPECT_EQ(selected->members().size(), 2U);
    ASSERT_EQ(selected->graph_members().size(), 1U);
    EXPECT_EQ(std::vector<std::string>(selected->operator_paths().begin(),
                                       selected->operator_paths().end()),
              (std::vector<std::string>{"child/first", "child/nested/second"}));
    EXPECT_EQ(plan.partitioning_plan().mode,
              automatic ? PartitioningMode::kConstrainedAutomatic : PartitioningMode::kManual);
  }
}

// REQ: COMPILE-075
TEST(CompilePartitionInstances, RetainedViewsPreserveSiblingAndMixedSelections) {
  std::optional<ExecutionPartitionView> retained;
  {
    Graph graph{"root"};
    test::add_compile_test_default_clock(graph);
    const auto left = graph.embed<AtomicCompileTree>("left");
    const auto right = graph.embed<AtomicCompileTree>("right");
    const auto direct = graph.op<test::CompileTestOperator>("direct");
    graph.partition("bundle").add(right).add(direct).add(left);
    const auto plan = compile(graph);
    ASSERT_TRUE(plan.ok()) << plan.json();
    const auto partition = plan.partitions().lookup_named("bundle");
    ASSERT_TRUE(partition);
    retained = *partition;
  }
  ASSERT_TRUE(retained);
  ASSERT_EQ(retained->graph_members().size(), 2U);
  EXPECT_LT(retained->graph_members()[0].value, retained->graph_members()[1].value);
  EXPECT_EQ(retained->members().size(), 5U);
  EXPECT_EQ(retained->operator_paths().size(), 5U);
}

// REQ: COMPILE-075
TEST(CompilePartitionInstances, SelectionChangesConfigDigestButAuthoringOrderDoesNot) {
  const auto make_plan = [](bool select_left, bool reverse) {
    Graph graph{"root"};
    test::add_compile_test_default_clock(graph);
    const auto first = graph.embed<AtomicCompileTree>(reverse ? "right" : "left");
    const auto second = graph.embed<AtomicCompileTree>(reverse ? "left" : "right");
    graph.partition("owner").add(select_left != reverse ? first : second);
    graph.auto_partition_remaining();
    return compile(graph);
  };
  const auto left = make_plan(true, false);
  const auto right = make_plan(false, false);
  const auto reordered = make_plan(true, true);
  ASSERT_TRUE(left.ok()) << left.json();
  ASSERT_TRUE(right.ok()) << right.json();
  ASSERT_TRUE(reordered.ok()) << reordered.json();
  EXPECT_NE(left.partitioning_plan().config_digest, right.partitioning_plan().config_digest);
  EXPECT_EQ(left.partitioning_plan().config_digest, reordered.partitioning_plan().config_digest);
  EXPECT_EQ(left.json(), reordered.json());
}

// REQ: COMPILE-075
TEST(CompilePartitionInstances, GreenContextDeviceObligationsReachSelectedDescendants) {
  Graph graph{"root"};
  const auto child = graph.embed<AtomicCompileTree>("child");
  (void)graph.op<test::CompileTestOperator>("peer");
  graph.partition("owner").add(child);
  graph.auto_partition_remaining();
  const auto snapshot = detail::GraphCompilerAccess::canonical_snapshot(graph);
  const std::vector<std::uint32_t> groups(snapshot->vertices.size(), 1U);
  DeploymentBinding deployment;
  deployment.bind_cuda_green_context("owner", {.device = DeviceId{7}, .sm_count = 1U});
  const detail::ExecutionPlanState plan;
  const auto storage =
      detail::make_partition_problem_storage(*snapshot, groups, deployment, {}, plan);
  ASSERT_TRUE(storage);
  ASSERT_EQ(storage->vertices.size(), 3U);
  for (const auto& vertex : snapshot->vertices) {
    if (vertex.kind != VertexKind::kOperator)
      continue;
    const auto& view = storage->vertices.at(storage->executable_index.at(vertex.id.value));
    const std::vector<std::int32_t> actual(view.device_obligations.begin(),
                                           view.device_obligations.end());
    const std::vector<std::int32_t> expected =
        vertex.name == "peer" ? std::vector<std::int32_t>{} : std::vector<std::int32_t>{7};
    EXPECT_EQ(actual, expected) << vertex.name;
  }
}

// REQ: COMPILE-075
TEST(CompilePartitionInstances, CompilerResolverRejectsMalformedSelections) {
  Graph graph{"root"};
  const auto child = graph.embed<AtomicCompileTree>("child");
  (void)graph.embed<Graph>("empty");
  graph.partition("owner").add(child);
  const auto canonical = detail::GraphCompilerAccess::canonical_snapshot(graph);
  const auto valid =
      detail::resolve_authored_partition_members(*canonical, canonical->partitions[0]);
  ASSERT_TRUE(valid);
  ASSERT_EQ(valid->size(), 2U);
  const auto selected = canonical->partitions[0].graph_members[0];
  const auto check = [&](detail::GraphSnapshotData snapshot, const char* token) {
    const auto result =
        detail::resolve_authored_partition_members(snapshot, snapshot.partitions[0]);
    ASSERT_FALSE(result);
    EXPECT_EQ(result.error().message, token);
  };
  for (const GraphId invalid : {GraphId{}, GraphId{9999U}, canonical->graphs[0].id}) {
    auto snapshot = *canonical;
    snapshot.partitions[0].graph_members = {invalid};
    check(std::move(snapshot), "GRAPH_PARTITION_GRAPH_INVALID");
  }
  for (const auto& record : canonical->graphs) {
    if (record.parent == selected) {
      auto snapshot = *canonical;
      snapshot.partitions[0].graph_members = {record.id};
      check(std::move(snapshot), "GRAPH_PARTITION_GRAPH_INVALID");
    }
  }
  auto duplicate = *canonical;
  duplicate.partitions[0].graph_members.push_back(selected);
  check(std::move(duplicate), "GRAPH_PARTITION_GRAPH_INVALID");
  auto reach_in = *canonical;
  reach_in.partitions[0].members = {*valid->begin()};
  check(std::move(reach_in), "PARTITION_COVER_INVALID");
  for (const auto& record : canonical->graphs) {
    if (record.id != selected && record.parent != selected)
      continue;
    const auto vertex = std::ranges::find(canonical->vertices, record.id, &VertexRecord::graph);
    ASSERT_NE(vertex, canonical->vertices.end());
    for (const bool nonempty : {false, true}) {
      auto internal = *canonical;
      internal.partitions.push_back(PartitionRecord{
          .id = PartitionId{2U},
          .graph = record.id,
          .name = "internal",
          .members = nonempty ? std::vector<VertexId>{vertex->id} : std::vector<VertexId>{}});
      check(std::move(internal), "GRAPH_PARTITION_INTERNAL_DECLARATION");
    }
  }
  auto empty = *canonical;
  const auto empty_graph =
      std::ranges::find(empty.graphs, std::string{"empty"}, &GraphRecord::name);
  ASSERT_NE(empty_graph, empty.graphs.end());
  empty.partitions[0].graph_members = {empty_graph->id};
  check(std::move(empty), "GRAPH_PARTITION_GRAPH_EMPTY");
}

// REQ: COMPILE-075
TEST(CompilePartitionInstances, ArtifactAuthorityRejectsForgedOrOmittedInstanceProvenance) {
  Graph graph{"root"};
  test::add_compile_test_default_clock(graph);
  const auto child = graph.embed<AtomicCompileTree>("child");
  (void)graph.op<test::CompileTestOperator>("peer");
  graph.partition("owner").add(child);
  graph.auto_partition_remaining();
  const auto plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  EXPECT_NE(plan.json().find("\"graph_members\":["), std::string::npos);
  EXPECT_NE(plan.json().find("holoscan.execution_plan.private.v6"), std::string::npos);
  const auto baseline = detail::ExecutionPlanAccess::retain(plan);
  ASSERT_TRUE(detail::valid_accepted_partition_authority(*baseline));
  for (const int corruption : {0, 1, 2, 3, 4, 5}) {
    auto state = *baseline;
    auto selected = std::ranges::find_if(state.execution_partitions, [](const auto& partition) {
      return partition.authored_name == "owner";
    });
    ASSERT_NE(selected, state.execution_partitions.end());
    if (corruption == 0) {
      selected->graph_members.clear();
    } else if (corruption == 1) {
      selected->graph_members[0] = GraphId{9999U};
    } else if (corruption == 2) {
      selected->graph_members.push_back(selected->graph_members[0]);
    } else if (corruption == 3) {
      selected->members.pop_back();
    } else if (corruption == 5) {
      selected->graph_members.clear();
      selected->authored_name = "renamed";
      for (auto& lookup : state.execution_partition_index_by_authored_name) {
        if (lookup.key == "owner")
          lookup.key = "renamed";
      }
    } else {
      auto automatic = std::ranges::find_if(state.execution_partitions, [](const auto& partition) {
        return !partition.authored_name;
      });
      ASSERT_NE(automatic, state.execution_partitions.end());
      automatic->graph_members = selected->graph_members;
    }
    EXPECT_FALSE(detail::valid_accepted_partition_authority(state));
    EXPECT_FALSE(detail::finalize_plan_artifact(state));
  }
}

}  // namespace
}  // namespace holoscan
