/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <iomanip>
#include <memory>
#include <new>
#include <ostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <holoscan/core/graph.hpp>

#include "core/graph_compiler_access.hpp"
#include "graph_test_support.hpp"

namespace holoscan {
namespace {

using test::graph_support::graph_error_from;
using test::graph_support::PassOp;

class PartitionLeaf final : public Graph {
 protected:
  void compose() override { (void)op<PassOp>("leaf"); }
};

class PartitionTree final : public Graph {
 protected:
  void compose() override {
    (void)op<PassOp>("first");
    (void)embed<PartitionLeaf>("nested");
  }
};

class InternallyPartitionedTree final : public Graph {
 protected:
  void compose() override {
    (void)embed<PartitionTree>("nested");
    (void)partition("empty-internal-declaration");
  }
};

template <bool Empty>
class InternallyPartitionedLeaf final : public Graph {
 protected:
  void compose() override {
    const auto leaf = op<PassOp>("leaf");
    auto internal = partition("internal");
    if constexpr (!Empty)
      internal.add(leaf);
  }
};

template <bool Empty>
class NestedInternallyPartitionedTree final : public Graph {
 protected:
  void compose() override { (void)embed<InternallyPartitionedLeaf<Empty>>("nested"); }
};

class NamedInternalPartitions final : public Graph {
 public:
  explicit NamedInternalPartitions(bool reverse) : reverse_(reverse) {}

 protected:
  void compose() override {
    (void)op<PassOp>("leaf");
    (void)partition(reverse_ ? "z" : "a \"quoted\"");
    (void)partition(reverse_ ? "a \"quoted\"" : "z");
  }

 private:
  bool reverse_{};
};

class OrderedInternalPartitionTree final : public Graph {
 public:
  explicit OrderedInternalPartitionTree(bool reverse) : reverse_(reverse) {}

 protected:
  void compose() override {
    (void)embed<NamedInternalPartitions>(reverse_ ? "z" : "a/nested", reverse_);
    (void)embed<NamedInternalPartitions>(reverse_ ? "a/nested" : "z", reverse_);
  }

 private:
  bool reverse_{};
};

template <unsigned Depth, bool Encoded = false>
class LongInternalPartitionTree final : public Graph {
 protected:
  void compose() override {
    if constexpr (Depth != 0U) {
      (void)embed<LongInternalPartitionTree<Depth - 1U, Encoded>>(
          Encoded ? "x" + std::string(100U, '%') : std::string(255U, 'x'));
    } else {
      (void)op<PassOp>("leaf");
      (void)partition(Encoded ? "x" + std::string(100U, '%') : std::string(255U, 'p'));
    }
  }
};

class InvalidAtomicSelectionGraph final : public Graph {
 protected:
  void compose() override {
    const auto child = embed<NestedInternallyPartitionedTree<false>>("child");
    partition("owner").add(child);
  }
};

// Kept identical to the public reusable-Graphs troubleshooting recipe.
void print_partition_declarations(const holoscan::GraphView& view, std::ostream& out) {
  for (const auto& declaration : view.partitions()) {
    const auto owner =
        std::ranges::find(view.graphs(), declaration.graph, &holoscan::GraphRecord::id);
    if (owner == view.graphs().end())
      continue;
    out << "owner=[";
    const auto& path = owner->path;
    if (path) {
      for (const auto& segment : path->segments())
        out << std::quoted(segment) << ' ';
    }
    out << "] partition=" << std::quoted(declaration.name) << '\n';
  }
}

// REQ: GRAPH-037
TEST(GraphComposition, PartitionInstancesGraphViewRecipeLocatesRetainedInternalDeclarations) {
  const auto retained = [] {
    Graph graph{"root"};
    (void)graph.embed<OrderedInternalPartitionTree>("selected", false);
    return graph.view();
  }();
  std::ostringstream output;
  print_partition_declarations(retained, output);
  EXPECT_NE(output.str().find("owner=[\"selected\" \"a/nested\" ] partition=\"a \\\"quoted\\\"\""),
            std::string::npos);
  EXPECT_NE(output.str().find("owner=[\"selected\" \"z\" ] partition=\"z\""), std::string::npos);
}

// REQ: GRAPH-037
// REQ: GRAPH-039
TEST(GraphComposition, PartitionInstancesPreserveInternalContextAsStickyComposeCause) {
  InvalidAtomicSelectionGraph graph;
  const Error first = graph_error_from([&] { (void)graph.view(); });
  const Error repeated = graph_error_from([&] { (void)graph.view(); });
  EXPECT_TRUE(std::string_view{first.message}.starts_with("GRAPH_COMPOSE_FAILED"));
  EXPECT_NE(first.message.find("GRAPH_PARTITION_INTERNAL_DECLARATION"), std::string::npos);
  EXPECT_NE(first.message.find("owner=child/nested, partition=internal"), std::string::npos);
  EXPECT_EQ(first.code, repeated.code);
  EXPECT_EQ(first.message, repeated.message);
  const auto diagnostic = detail::GraphCompilerAccess::composition_diagnostic(graph);
  if (!diagnostic) {
    FAIL() << "composition diagnostic was not retained";
    return;
  }
  const auto& cause = diagnostic->cause;
  if (!cause) {
    FAIL() << "internal partition declaration cause was not retained";
    return;
  }
  EXPECT_EQ(diagnostic->primary.message, first.message);
  EXPECT_EQ(cause->message,
            "GRAPH_PARTITION_INTERNAL_DECLARATION (owner=child/nested, partition=internal)");
}

// REQ: GRAPH-037
TEST(GraphComposition, PartitionInstancesReportCanonicalOwnedInternalContext) {
  Error retained;
  for (const bool reverse : {false, true}) {
    Graph graph{"root"};
    const auto child = graph.embed<OrderedInternalPartitionTree>("selected", reverse);
    (void)graph.embed<OrderedInternalPartitionTree>("other", !reverse);
    auto selected = graph.partition("owner");
    const auto before = graph.view();
    const auto error = graph_error_from([&] { selected.add(child); });
    EXPECT_EQ(error.code, ErrorCode::kInvalidArgument);
    EXPECT_EQ(error.message,
              "GRAPH_PARTITION_INTERNAL_DECLARATION (owner=selected/a%2Fnested, "
              "partition=a%20%22quoted%22)");
    EXPECT_EQ(graph.view().identity(), before.identity());
    EXPECT_TRUE(graph.view().partitions().back().graph_members.empty());
    if (reverse) {
      EXPECT_EQ(error.message, retained.message);
    }
    retained = error;
  }
  // Exception/value conversion and Graph/snapshot retirement must not borrow diagnostic text.
  EXPECT_EQ(make_error(RuntimeError{retained}).message, retained.message);
  EXPECT_NE(retained.message.find("selected/a%2Fnested"), std::string::npos);
}

// REQ: GRAPH-037
TEST(GraphComposition, PartitionInstancesBoundDeepInternalContextWithoutMutation) {
  Graph graph{"root"};
  const auto child = graph.embed<LongInternalPartitionTree<3U>>("selected");
  auto selected = graph.partition("owner");
  const auto before = graph.view();
  const auto error = graph_error_from([&] { selected.add(child); });
  EXPECT_TRUE(error.message.starts_with("GRAPH_PARTITION_INTERNAL_DECLARATION (owner=selected/"));
  EXPECT_NE(error.message.find("..., partition="), std::string::npos);
  EXPECT_TRUE(error.message.ends_with("...)"));
  EXPECT_LE(error.message.size(), 512U);
  EXPECT_EQ(graph.view().identity(), before.identity());
}

// REQ: GRAPH-037
TEST(GraphComposition, PartitionInstancesTruncateOnlyCompleteEncodedBytes) {
  Graph graph{"root"};
  const auto child = graph.embed<LongInternalPartitionTree<1U, true>>("selected");
  auto selected = graph.partition("owner");
  const auto before = graph.view();
  const auto error = graph_error_from([&] { selected.add(child); });
  EXPECT_TRUE(
      error.message.starts_with("GRAPH_PARTITION_INTERNAL_DECLARATION (owner=selected/x%25%25"));
  EXPECT_NE(error.message.find("..., partition=x%25%25"), std::string::npos);
  EXPECT_TRUE(error.message.ends_with("%25...)"));
  EXPECT_EQ(error.message.find("%..."), std::string::npos);
  EXPECT_EQ(error.message.find("%2..."), std::string::npos);
  EXPECT_LE(error.message.size(), 512U);
  EXPECT_EQ(graph.view().identity(), before.identity());
}

// REQ: GRAPH-037
TEST(GraphComposition, PartitionInstancesInternalContextFailurePreservesRetry) {
  Graph graph{"root"};
  const auto child = graph.embed<NestedInternallyPartitionedTree<true>>("child");
  auto selected = graph.partition("owner");
  const auto before = graph.view();
  detail::GraphCompilerAccess::fail_mutation_after(graph, 0U);
  EXPECT_THROW(selected.add(child), std::bad_alloc);
  EXPECT_EQ(graph.view().identity(), before.identity());
  EXPECT_EQ(graph_error_from([&] { selected.add(child); }).message,
            "GRAPH_PARTITION_INTERNAL_DECLARATION (owner=child/nested, partition=internal)");
  EXPECT_EQ(graph.view().identity(), before.identity());
}

// REQ: GRAPH-037
TEST(GraphComposition, PartitionInstancesRetainCanonicalSelectionAndRollback) {
  Graph graph{"root"};
  const auto second = graph.embed<PartitionTree>("second");
  const auto first = graph.embed<PartitionTree>("first");
  auto partition = graph.partition("owner");
  const auto before = graph.view();
  detail::GraphCompilerAccess::fail_mutation_after(graph, 0U);
  EXPECT_THROW(partition.add(first), std::bad_alloc);
  EXPECT_EQ(graph.view().identity(), before.identity());
  EXPECT_TRUE(graph.view().partitions()[0].graph_members.empty());
  EXPECT_NO_THROW(partition.add(second).add(first));
  const auto selected = graph.view();
  ASSERT_EQ(selected.partitions()[0].graph_members.size(), 2U);
  EXPECT_TRUE(selected.partitions()[0].members.empty());
  EXPECT_EQ(graph_error_from([&] { partition.add(first); }).message,
            "GRAPH_PARTITION_MEMBER_DUPLICATE");
  EXPECT_EQ(graph.view().identity(), selected.identity());

  Graph reordered{"root"};
  const auto other_first = reordered.embed<PartitionTree>("first");
  const auto other_second = reordered.embed<PartitionTree>("second");
  reordered.partition("owner").add(other_first).add(other_second);
  EXPECT_EQ(reordered.view().identity().hash, selected.identity().hash);
  const auto canonical = detail::GraphCompilerAccess::canonical_snapshot(graph);
  const auto other = detail::GraphCompilerAccess::canonical_snapshot(reordered);
  EXPECT_EQ(canonical->partitions[0].graph_members, other->partitions[0].graph_members);
}

// REQ: GRAPH-037
TEST(GraphComposition, PartitionInstancesRejectInvalidSelectionsWithoutMutation) {
  Graph graph{"root"};
  const auto child = graph.embed<PartitionTree>("child");
  const auto internal = graph.embed<InternallyPartitionedTree>("internal");
  const auto nested_empty = graph.embed<NestedInternallyPartitionedTree<true>>("nested-empty");
  const auto nested_nonempty =
      graph.embed<NestedInternallyPartitionedTree<false>>("nested-nonempty");
  const auto empty = graph.embed<Graph>("empty");
  Graph foreign{"foreign"};
  const auto foreign_child = foreign.embed<PartitionTree>("child");
  const auto expired = [] {
    Graph temporary{"temporary"};
    return temporary.embed<PartitionTree>("child");
  }();
  auto selected = graph.partition("selected");
  auto other = graph.partition("other");
  selected.add(child);
  const auto before = graph.view();
  EXPECT_EQ(graph_error_from([&] { other.add(child); }).message,
            "GRAPH_PARTITION_MEMBER_DUPLICATE");
  EXPECT_EQ(graph_error_from([&] { other.add(foreign_child); }).message,
            "GRAPH_PARTITION_MEMBER_FOREIGN");
  EXPECT_EQ(graph_error_from([&] { other.add(expired); }).message,
            "GRAPH_PARTITION_HANDLE_INVALID");
  EXPECT_EQ(graph_error_from([&] { other.add(empty); }).message, "GRAPH_PARTITION_GRAPH_EMPTY");
  EXPECT_EQ(graph_error_from([&] { other.add(internal); }).message,
            "GRAPH_PARTITION_INTERNAL_DECLARATION (owner=internal, "
            "partition=empty-internal-declaration)");
  EXPECT_EQ(graph_error_from([&] { other.add(nested_empty); }).message,
            "GRAPH_PARTITION_INTERNAL_DECLARATION (owner=nested-empty/nested, partition=internal)");
  EXPECT_EQ(
      graph_error_from([&] { other.add(nested_nonempty); }).message,
      "GRAPH_PARTITION_INTERNAL_DECLARATION (owner=nested-nonempty/nested, partition=internal)");
  EXPECT_EQ(graph.view().identity(), before.identity());

  Graph all{"all"};
  const auto all_child = all.embed<PartitionTree>("child");
  auto all_partition = all.partition("owner");
  all_partition.include_all();
  const auto all_before = all.view();
  EXPECT_EQ(graph_error_from([&] { all_partition.add(all_child); }).message,
            "GRAPH_PARTITION_INCLUDE_ALL_CONFLICT");
  EXPECT_EQ(all.view().identity(), all_before.identity());

  Graph explicit_graph{"explicit"};
  const auto explicit_child = explicit_graph.embed<PartitionTree>("child");
  auto explicit_partition = explicit_graph.partition("owner");
  explicit_partition.add(explicit_child);
  const auto explicit_before = explicit_graph.view();
  EXPECT_EQ(graph_error_from([&] { explicit_partition.include_all(); }).message,
            "GRAPH_PARTITION_INCLUDE_ALL_CONFLICT");
  EXPECT_EQ(explicit_graph.view().identity(), explicit_before.identity());
}

}  // namespace
}  // namespace holoscan
