/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <set>
#include <string>

#include <holoscan/core/graph.hpp>

#include "runtime/operator_profile_labels.hpp"

namespace holoscan::detail {
namespace {

class NestedProfileGraph final : public Graph {
 protected:
  void compose() override { (void)embed<Graph>("stage"); }
};

TEST(OperatorProfileLabelsTest, RootLabelsOmitGraphNameAndKeepPortSeparator) {
  Graph graph{"root-name"};
  ExecutionPlanState plan;
  plan.graph = graph.view();
  ASSERT_EQ(plan.graph.graphs().size(), 1U);
  const GraphId root = plan.graph.graphs().front().id;

  EXPECT_EQ(operator_profile_operator_label(plan, root, "source"), "source");
  EXPECT_EQ(operator_profile_port_label(plan, root, "source", "out"), "source.out");
}

TEST(OperatorProfileLabelsTest, NestedInstancesKeepFullDistinctPaths) {
  Graph graph{"root-name"};
  (void)graph.embed<NestedProfileGraph>("left");
  (void)graph.embed<NestedProfileGraph>("right");
  ExecutionPlanState plan;
  plan.graph = graph.view();

  std::set<std::string> operator_labels;
  std::set<std::string> port_labels;
  for (const auto& row : plan.graph.graphs()) {
    if (row.name == "stage") {
      operator_labels.insert(operator_profile_operator_label(plan, row.id, "scale"));
      port_labels.insert(operator_profile_port_label(plan, row.id, "scale", "out"));
    }
  }

  EXPECT_EQ(operator_labels, (std::set<std::string>{"left/stage/scale", "right/stage/scale"}));
  EXPECT_EQ(port_labels, (std::set<std::string>{"left/stage/scale.out", "right/stage/scale.out"}));
}

TEST(OperatorProfileLabelsTest, LiteralGraphSeparatorDoesNotAliasNestedPath) {
  Graph graph;
  (void)graph.embed<Graph>("left/stage");
  (void)graph.embed<NestedProfileGraph>("left");
  ExecutionPlanState plan;
  plan.graph = graph.view();

  std::set<std::string> labels;
  for (const auto& row : plan.graph.graphs()) {
    if (row.name == "left/stage" || row.name == "stage") {
      labels.insert(operator_profile_operator_label(plan, row.id, "scale"));
    }
  }

  EXPECT_EQ(labels, (std::set<std::string>{R"(left\/stage/scale)", "left/stage/scale"}));
}

TEST(OperatorProfileLabelsTest, EscapesReservedCharactersInEveryNameComponent) {
  Graph graph;
  (void)graph.embed<Graph>(R"(graph.part:0\left/right)");
  ExecutionPlanState plan;
  plan.graph = graph.view();
  ASSERT_EQ(plan.graph.graphs().size(), 2U);
  const GraphId child = plan.graph.graphs().back().id;

  EXPECT_EQ(operator_profile_operator_label(plan, child, R"(camera.front:0\left/right)"),
            R"(graph\.part\:0\\left\/right/camera\.front\:0\\left\/right)");
  EXPECT_EQ(operator_profile_port_label(plan, child, "camera", R"(rgb.port:0\left/right)"),
            R"(graph\.part\:0\\left\/right/camera.rgb\.port\:0\\left\/right)");
}

}  // namespace
}  // namespace holoscan::detail
