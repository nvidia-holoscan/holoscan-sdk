/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <vector>

#include <holoscan/holoscan.hpp>
#include "holoscan/core/graphs/flow_graph.hpp"

namespace holoscan {

// Utility class for testing - simple operator with configurable inputs/outputs
class TestOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(TestOp)

  TestOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.input<int>("in");
    spec.output<int>("out");
  }

  void compute(InputContext&, OutputContext&, ExecutionContext&) override {}
};

// Test operator with multiple input ports
class MultiInputOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(MultiInputOp)

  MultiInputOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.input<int>("in1");
    spec.input<int>("in2");
    spec.input<int>("in3");
    spec.output<int>("out");
  }

  void compute(InputContext&, OutputContext&, ExecutionContext&) override {}
};

// Test operator with multiple output ports
class MultiOutputOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(MultiOutputOp)

  MultiOutputOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.input<int>("in");
    spec.output<int>("out1");
    spec.output<int>("out2");
    spec.output<int>("out3");
  }

  void compute(InputContext&, OutputContext&, ExecutionContext&) override {}
};

// Test fixture for FlowGraph tests
class FlowGraphTest : public ::testing::Test {
 protected:
  void SetUp() override {}
};

TEST_F(FlowGraphTest, TestCycleDetectionNoCycle) {
  auto app = make_application<Application>();

  auto op1 = app->make_operator<TestOp>("op1");
  auto op2 = app->make_operator<TestOp>("op2");

  app->add_flow(op1, op2);

  auto& graph = static_cast<OperatorFlowGraph&>(app->graph());
  auto cyclic_roots = graph.has_cycle();

  EXPECT_EQ(cyclic_roots.size(), 0);
}

TEST_F(FlowGraphTest, TestCycleDetectionWithCycle) {
  auto app = make_application<Application>();

  auto op1 = app->make_operator<TestOp>("op1");
  auto op2 = app->make_operator<TestOp>("op2");

  // Create a cycle: op1 -> op2 -> op1
  app->add_flow(op1, op2);
  app->add_flow(op2, op1);

  auto& graph = static_cast<OperatorFlowGraph&>(app->graph());
  auto cyclic_roots = graph.has_cycle();

  EXPECT_GT(cyclic_roots.size(), 0);
}

TEST_F(FlowGraphTest, TestCacheCycleInvalidationOnAddFlow) {
  auto app = make_application<Application>();

  // Step 1: Add two graph nodes (n1, n2)
  auto n1 = app->make_operator<TestOp>("n1");
  auto n2 = app->make_operator<TestOp>("n2");

  auto& graph = static_cast<OperatorFlowGraph&>(app->graph());

  // Step 2: add_flow(n1, n2)
  app->add_flow(n1, n2);

  // Step 3: check has_cycle - cached_cyclic_roots_ caches cyclic_roots which is empty
  auto cyclic_roots_before = graph.has_cycle();
  EXPECT_EQ(cyclic_roots_before.size(), 0) << "Graph should have no cycles after first add_flow";

  // Step 4: add_flow(n2, n1) - this creates a cycle
  app->add_flow(n2, n1);

  // Step 5: check has_cycle - cached_cyclic_roots_ should be invalidated and return new result
  auto cyclic_roots_after = graph.has_cycle();
  EXPECT_GT(cyclic_roots_after.size(), 0)
      << "Graph should have a cycle after second add_flow, cache must be invalidated";
}

TEST_F(FlowGraphTest, TestCacheInvalidationOnAddNode) {
  auto app = make_application<Application>();

  auto op1 = app->make_operator<TestOp>("op1");
  auto op2 = app->make_operator<TestOp>("op2");

  app->add_flow(op1, op2);
  app->add_flow(op2, op1);

  auto& graph = static_cast<OperatorFlowGraph&>(app->graph());

  // First call to has_cycle should detect the cycle and cache it
  auto cyclic_roots_before = graph.has_cycle();
  EXPECT_GT(cyclic_roots_before.size(), 0);

  // Add a new node connected to the cycle to ensure cache invalidation works
  auto op3 = app->make_operator<TestOp>("op3");
  app->add_flow(op2, op3);  // Connect op3 to the existing cycle

  // The cache should be invalidated by add_flow, so has_cycle should recalculate
  // The cycle should still exist (op1 -> op2 -> op1)
  auto cyclic_roots_after = graph.has_cycle();
  EXPECT_GT(cyclic_roots_after.size(), 0) << "Cycle should still be detected after adding a node";
}

TEST_F(FlowGraphTest, TestCacheInvalidationOnRemoveNode) {
  auto app = make_application<Application>();

  auto op1 = app->make_operator<TestOp>("op1");
  auto op2 = app->make_operator<TestOp>("op2");
  auto op3 = app->make_operator<TestOp>("op3");

  // Create: op1 -> op2 -> op3 -> op1 (cycle)
  app->add_flow(op1, op2);
  app->add_flow(op2, op3);
  app->add_flow(op3, op1);

  auto& graph = static_cast<OperatorFlowGraph&>(app->graph());

  // First call to has_cycle should detect the cycle
  auto cyclic_roots_before = graph.has_cycle();
  EXPECT_GT(cyclic_roots_before.size(), 0) << "Graph should have a cycle";

  // Remove a node from the cycle
  graph.remove_node(op2);

  // Cache should be invalidated and cycle should no longer be detected
  auto cyclic_roots_after = graph.has_cycle();
  EXPECT_EQ(cyclic_roots_after.size(), 0)
      << "Cycle should be broken after removing a node from the cycle";
}

TEST_F(FlowGraphTest, TestMultipleCycleDetectionCalls) {
  auto app = make_application<Application>();

  auto op1 = app->make_operator<TestOp>("op1");
  auto op2 = app->make_operator<TestOp>("op2");

  auto& graph = static_cast<OperatorFlowGraph&>(app->graph());

  app->add_flow(op1, op2);

  // Multiple calls to has_cycle without graph modification should return same result
  auto result1 = graph.has_cycle();
  auto result2 = graph.has_cycle();
  auto result3 = graph.has_cycle();

  EXPECT_EQ(result1.size(), 0);
  EXPECT_EQ(result2.size(), 0);
  EXPECT_EQ(result3.size(), 0);

  // Now create a cycle
  app->add_flow(op2, op1);

  // After modification, has_cycle should return different result
  auto result4 = graph.has_cycle();
  EXPECT_GT(result4.size(), 0);

  // Multiple calls should still return the same (cached) result
  auto result5 = graph.has_cycle();
  auto result6 = graph.has_cycle();

  EXPECT_EQ(result4.size(), result5.size());
  EXPECT_EQ(result5.size(), result6.size());
}

TEST_F(FlowGraphTest, TestIsUserDefinedRootNoCycle) {
  auto app = make_application<Application>();

  auto op1 = app->make_operator<TestOp>("op1");
  auto op2 = app->make_operator<TestOp>("op2");

  // op1 -> op2 (no cycle)
  app->add_flow(op1, op2);

  auto& graph = static_cast<OperatorFlowGraph&>(app->graph());

  // op1 is the first node added, but there's no cycle, so it's not a user-defined root
  EXPECT_FALSE(graph.is_user_defined_root(op1));
  EXPECT_FALSE(graph.is_user_defined_root(op2));
}

TEST_F(FlowGraphTest, TestIsUserDefinedRootWithCycle) {
  auto app = make_application<Application>();

  auto op1 = app->make_operator<TestOp>("op1");
  auto op2 = app->make_operator<TestOp>("op2");

  // Create a cycle: op1 -> op2 -> op1
  app->add_flow(op1, op2);
  app->add_flow(op2, op1);

  auto& graph = static_cast<OperatorFlowGraph&>(app->graph());

  // op1 is the first node added AND there's a cycle, so it's a user-defined root
  EXPECT_TRUE(graph.is_user_defined_root(op1));
  // op2 is not the first node, so it's not a user-defined root
  EXPECT_FALSE(graph.is_user_defined_root(op2));
}

TEST_F(FlowGraphTest, TestIsUserDefinedRootWithNullptr) {
  auto app = make_application<Application>();
  auto& graph = static_cast<OperatorFlowGraph&>(app->graph());

  // Capture warnings
  testing::internal::CaptureStderr();

  EXPECT_FALSE(graph.is_user_defined_root(nullptr));

  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("Calling is_user_defined_root() with nullptr") != std::string::npos);
}

TEST_F(FlowGraphTest, TestFragmentFlowGraphCycleDetection) {
  auto app = make_application<Application>();

  auto frag1 = app->make_fragment<Fragment>("frag1");
  auto frag2 = app->make_fragment<Fragment>("frag2");

  // Create a cycle in fragment graph
  app->add_flow(frag1, frag2, {{"out", "in"}});
  app->add_flow(frag2, frag1, {{"out", "in"}});

  auto& graph = static_cast<FragmentFlowGraph&>(app->fragment_graph());
  auto cyclic_roots = graph.has_cycle();

  EXPECT_GT(cyclic_roots.size(), 0) << "Fragment graph should detect cycles";
}

// ==================== get_indegree tests ====================

TEST_F(FlowGraphTest, TestGetIndegreeSimpleConnection) {
  auto app = make_application<Application>();

  auto tx = app->make_operator<TestOp>("tx");
  auto rx = app->make_operator<TestOp>("rx");

  // tx.out -> rx.in
  app->add_flow(tx, rx, {{"out", "in"}});

  auto& graph = static_cast<OperatorFlowGraph&>(app->graph());

  // rx.in should have indegree of 1
  EXPECT_EQ(graph.get_indegree(rx, "in"), 1) << "rx.in should have indegree of 1";

  // tx.in should have indegree of 0 (no incoming connections)
  EXPECT_EQ(graph.get_indegree(tx, "in"), 0) << "tx.in should have indegree of 0";

  // tx.out is an output port, should have indegree of 0
  EXPECT_EQ(graph.get_indegree(tx, "out"), 0)
      << "tx.out is an output port, should have indegree of 0";
}

TEST_F(FlowGraphTest, TestGetIndegreeMultipleConnectionsToSamePort) {
  auto app = make_application<Application>();

  auto tx1 = app->make_operator<MultiOutputOp>("tx1");
  auto tx2 = app->make_operator<MultiOutputOp>("tx2");
  auto rx = app->make_operator<MultiInputOp>("rx");

  // Connect multiple outputs from different sources to the same input port
  // tx1.out1 -> rx.in1
  app->add_flow(tx1, rx, {{"out1", "in1"}});
  // tx2.out1 -> rx.in1 (same input port)
  app->add_flow(tx2, rx, {{"out1", "in1"}});

  auto& graph = static_cast<OperatorFlowGraph&>(app->graph());

  // rx.in1 should have indegree of 2 (two incoming connections)
  EXPECT_EQ(graph.get_indegree(rx, "in1"), 2)
      << "rx.in1 should have indegree of 2 (connections from tx1 and tx2)";

  // rx.in2 should have indegree of 0 (no connections)
  EXPECT_EQ(graph.get_indegree(rx, "in2"), 0) << "rx.in2 should have indegree of 0";
}

TEST_F(FlowGraphTest, TestGetIndegreeSamePredecessorMultiplePorts) {
  auto app = make_application<Application>();

  auto tx = app->make_operator<MultiOutputOp>("tx");
  auto rx = app->make_operator<MultiInputOp>("rx");

  // Connect multiple output ports from the same source to different input ports
  // tx.out1 -> rx.in1
  app->add_flow(tx, rx, {{"out1", "in1"}});
  // tx.out2 -> rx.in2
  app->add_flow(tx, rx, {{"out2", "in2"}});
  // tx.out3 -> rx.in3
  app->add_flow(tx, rx, {{"out3", "in3"}});

  auto& graph = static_cast<OperatorFlowGraph&>(app->graph());

  // Each input port should have indegree of 1
  EXPECT_EQ(graph.get_indegree(rx, "in1"), 1) << "rx.in1 should have indegree of 1";
  EXPECT_EQ(graph.get_indegree(rx, "in2"), 1) << "rx.in2 should have indegree of 1";
  EXPECT_EQ(graph.get_indegree(rx, "in3"), 1) << "rx.in3 should have indegree of 1";
}

TEST_F(FlowGraphTest, TestGetIndegreeNonexistentPort) {
  auto app = make_application<Application>();

  auto tx = app->make_operator<TestOp>("tx");
  auto rx = app->make_operator<TestOp>("rx");

  // tx.out -> rx.in
  app->add_flow(tx, rx, {{"out", "in"}});

  auto& graph = static_cast<OperatorFlowGraph&>(app->graph());

  // Query a port that doesn't exist
  EXPECT_EQ(graph.get_indegree(rx, "nonexistent_port"), 0)
      << "Nonexistent port should have indegree of 0";
}

TEST_F(FlowGraphTest, TestGetIndegreeNodeNotInGraph) {
  auto app = make_application<Application>();

  auto tx = app->make_operator<TestOp>("tx");
  auto rx = app->make_operator<TestOp>("rx");
  auto isolated = app->make_operator<TestOp>("isolated");

  // Only connect tx and rx, isolated is not in the graph
  app->add_flow(tx, rx, {{"out", "in"}});

  auto& graph = static_cast<OperatorFlowGraph&>(app->graph());

  // Query a node that's not in the graph's edges (but might be added as a node)
  EXPECT_EQ(graph.get_indegree(isolated, "in"), 0) << "Node not in graph should have indegree of 0";
}

TEST_F(FlowGraphTest, TestGetIndegreeDifferentPortNames) {
  auto app = make_application<Application>();

  auto tx = app->make_operator<MultiOutputOp>("tx");
  auto rx = app->make_operator<MultiInputOp>("rx");

  // Connect output port "out1" to input port "in1" (different base names)
  app->add_flow(tx, rx, {{"out1", "in1"}});
  app->add_flow(tx, rx, {{"out2", "in2"}});

  auto& graph = static_cast<OperatorFlowGraph&>(app->graph());

  // When searching for indegree of "in1", we should find it even though
  // the predecessor's port is named "out1"
  EXPECT_EQ(graph.get_indegree(rx, "in1"), 1)
      << "rx.in1 should have indegree of 1 (connected from tx.out1)";
  EXPECT_EQ(graph.get_indegree(rx, "in2"), 1)
      << "rx.in2 should have indegree of 1 (connected from tx.out2)";
  EXPECT_EQ(graph.get_indegree(rx, "in3"), 0)
      << "rx.in3 should have indegree of 0 (no connections)";

  // Make sure we're not accidentally finding the wrong port
  // If the implementation incorrectly searches in the keys (output port names),
  // querying "out1" would incorrectly return 1 instead of 0
  EXPECT_EQ(graph.get_indegree(rx, "out1"), 0)
      << "rx doesn't have a port named 'out1', should have indegree of 0";
}

TEST_F(FlowGraphTest, TestGetIndegreeComplex) {
  auto app = make_application<Application>();

  auto tx1 = app->make_operator<MultiOutputOp>("tx1");
  auto tx2 = app->make_operator<MultiOutputOp>("tx2");
  auto rx = app->make_operator<MultiInputOp>("rx");

  // Create a complex connection pattern:
  // - tx1.out1 -> rx.in1
  // - tx1.out2 -> rx.in1 (second connection to same input)
  // - tx2.out1 -> rx.in1 (third connection to same input)
  // - tx2.out2 -> rx.in2
  app->add_flow(tx1, rx, {{"out1", "in1"}});
  app->add_flow(tx1, rx, {{"out2", "in1"}});
  app->add_flow(tx2, rx, {{"out1", "in1"}});
  app->add_flow(tx2, rx, {{"out2", "in2"}});

  auto& graph = static_cast<OperatorFlowGraph&>(app->graph());

  // rx.in1 has 3 incoming connections
  EXPECT_EQ(graph.get_indegree(rx, "in1"), 3)
      << "rx.in1 should have indegree of 3 (from tx1.out1, tx1.out2, tx2.out1)";

  // rx.in2 has 1 incoming connection
  EXPECT_EQ(graph.get_indegree(rx, "in2"), 1) << "rx.in2 should have indegree of 1 (from tx2.out2)";

  // rx.in3 has no incoming connections
  EXPECT_EQ(graph.get_indegree(rx, "in3"), 0) << "rx.in3 should have indegree of 0";
}

TEST_F(FlowGraphTest, TestGetIndegreeOutdegreeConsistency) {
  auto app = make_application<Application>();

  auto tx = app->make_operator<TestOp>("tx");
  auto rx1 = app->make_operator<TestOp>("rx1");
  auto rx2 = app->make_operator<TestOp>("rx2");

  // tx broadcasts to two receivers:
  // tx.out -> rx1.in
  // tx.out -> rx2.in
  app->add_flow(tx, rx1, {{"out", "in"}});
  app->add_flow(tx, rx2, {{"out", "in"}});

  auto& graph = static_cast<OperatorFlowGraph&>(app->graph());

  // The outdegree of tx.out should be 2
  EXPECT_EQ(graph.get_outdegree(tx, "out"), 2)
      << "tx.out should have outdegree of 2 (broadcasting to rx1 and rx2)";

  // The indegree of each receiver should be 1
  EXPECT_EQ(graph.get_indegree(rx1, "in"), 1) << "rx1.in should have indegree of 1";
  EXPECT_EQ(graph.get_indegree(rx2, "in"), 1) << "rx2.in should have indegree of 1";

  // Sum of indegrees should equal the outdegree
  size_t total_indegree = graph.get_indegree(rx1, "in") + graph.get_indegree(rx2, "in");
  EXPECT_EQ(total_indegree, graph.get_outdegree(tx, "out"))
      << "Sum of indegrees should equal outdegree";
}

}  // namespace holoscan
