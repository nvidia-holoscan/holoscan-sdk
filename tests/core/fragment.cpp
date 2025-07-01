/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "holoscan/core/fragment.hpp"

#include <gtest/gtest.h>
#include <gxf/core/gxf.h>

#include <string>
#include <utility>
#include <vector>

#include "../config.hpp"
#include "common/assert.hpp"
#include "holoscan/core/application.hpp"
#include "holoscan/core/arg.hpp"
#include "holoscan/core/component.hpp"
#include "holoscan/core/component_spec.hpp"
#include "holoscan/core/condition.hpp"
#include "holoscan/core/conditions/gxf/boolean.hpp"
#include "holoscan/core/conditions/gxf/count.hpp"
#include "holoscan/core/conditions/gxf/downstream_affordable.hpp"
#include "holoscan/core/conditions/gxf/message_available.hpp"
#include "holoscan/core/config.hpp"
#include "holoscan/core/executor.hpp"
#include "holoscan/core/graph.hpp"
#include "holoscan/core/graphs/flow_graph.hpp"
#include "holoscan/core/operator.hpp"
#include "holoscan/core/operator_spec.hpp"
#include "holoscan/core/resource.hpp"
#include "holoscan/core/resources/gxf/unbounded_allocator.hpp"
#include "holoscan/operators/ping_rx/ping_rx.hpp"
#include "holoscan/operators/ping_tx/ping_tx.hpp"
#include "holoscan/operators/video_stream_recorder/video_stream_recorder.hpp"

using namespace std::string_literals;

static HoloscanTestConfig test_config;

namespace holoscan {

TEST(Fragment, TestFragmentName) {
  Fragment F;
  EXPECT_EQ(F.name(), "");

  std::string name1{"fragment-1"};
  F.name(name1);
  EXPECT_EQ(F.name(), name1);

  F.name("fragment-renamed"s);
  EXPECT_EQ(F.name(), "fragment-renamed"s);
}

TEST(Fragment, TestFragmentAssignApplication) {
  Application* A;
  A = new Application;
  A->name("my application");
  Fragment F;

  F.application(A);
  delete A;
}

// TODO(unknown): how to properly specify path to the config file here
//       ? maybe define a path in the CMake config that we can reference here?

TEST(Fragment, TestFragmentConfig) {
  Fragment F;

  const std::string config_file = test_config.get_test_data_file("app_config.yaml");
  F.config(config_file);

  Config& C = F.config();
  ASSERT_TRUE(C.config_file() == config_file);

  ArgList args = F.from_config("format_converter_replayer"s);
  EXPECT_EQ(args.size(), 4);

  args = F.from_config("format_converter_replayer.out_tensor_name");
  Arg& arg = *args.begin();
  EXPECT_EQ(arg.arg_type().element_type(), ArgElementType::kYAMLNode);
  EXPECT_EQ(arg.arg_type().container_type(), ArgContainerType::kNative);
  EXPECT_EQ(arg.name(), "out_tensor_name");
  EXPECT_EQ(args.as<std::string>(), "source_video");

  // verify that second call to config raises a warning
  testing::internal::CaptureStderr();
  F.config(config_file);
  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("warning") != std::string::npos) << "=== LOG ===\n"
                                                               << log_output << "\n===========\n";
  EXPECT_TRUE(log_output.find("Config object was already created. Overwriting...") !=
              std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
}

TEST(Fragment, TestFragmentConfigNestedArgs) {
  Fragment F;

  const std::string config_file = test_config.get_test_data_file("app_config.yaml");
  F.config(config_file);

  Config& C = F.config();
  ASSERT_TRUE(C.config_file() == config_file);

  // can directly access a specific argument under the "replayer" section
  ArgList arglist = F.from_config("replayer.frame_rate"s);
  EXPECT_EQ(arglist.size(), 1);
  Arg width = arglist.args()[0];
  EXPECT_EQ(width.arg_type().element_type(), ArgElementType::kYAMLNode);
  EXPECT_EQ(width.arg_type().container_type(), ArgContainerType::kNative);
}

TEST(Fragment, TestConfigUninitializedWarning) {
  Fragment F;

  Config& C = F.config();
  EXPECT_EQ(C.config_file(), "");
}

TEST(Fragment, TestFragmentFromConfigNonexistentKey) {
  Fragment F;

  testing::internal::CaptureStderr();

  const std::string config_file = test_config.get_test_data_file("app_config.yaml");
  F.config(config_file);

  Config& C = F.config();
  ASSERT_TRUE(C.config_file() == config_file);
  ArgList args = F.from_config("non-existent"s);
  EXPECT_EQ(args.size(), 0);

  // verify that an error is logged when the key is not in the YAML file
  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("error") != std::string::npos) << "=== LOG ===\n"
                                                             << log_output << "\n===========\n";
  EXPECT_TRUE(log_output.find("Unable to find the parameter item/map with key 'non-existent'") !=
              std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
}

TEST(Fragment, TestFragmentConfigNonexistentFile) {
  Fragment F;

  const std::string config_file = "nonexistent.yaml";

  // capture stderr
  testing::internal::CaptureStderr();
  F.config(config_file);

  // verify that an error is logged when the YAML file doesn't exist
  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("warning") != std::string::npos) << "=== LOG ===\n"
                                                               << log_output << "\n===========\n";
  EXPECT_TRUE(log_output.find("Config file 'nonexistent.yaml' doesn't exist") != std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
}

TEST(Fragment, TestFragmentGraph) {
  Fragment F;

  // First call to graph creates a FlowGraph object
  // F.graph() returns a reference to the abstract Graph base class so use
  // static_cast here
  OperatorFlowGraph& G = static_cast<OperatorFlowGraph&>(F.graph());
}

TEST(Fragment, TestAddOperator) {
  Fragment F;
  auto op = F.make_operator<Operator>("op");

  F.add_operator(op);

  // First call to graph creates a FlowGraph object
  // F.graph() returns a reference to the abstract Graph base class so use
  // static_cast here
  OperatorFlowGraph& G = static_cast<OperatorFlowGraph&>(F.graph());

  // verify that the operator was added to the graph
  auto nodes = G.get_nodes();
  EXPECT_EQ(nodes.size(), 1);
  EXPECT_EQ(nodes[0], op);
}

TEST(Fragment, TestMakeThreadPool) {
  Fragment F;
  auto op1 = F.make_operator<Operator>("op1");
  auto op2 = F.make_operator<Operator>("op2");
  auto op3 = F.make_operator<Operator>("op3");

  // create a pool of size 1
  auto pool1 = F.make_thread_pool("pool1", 1);
  // add an individual operator without thread pinning
  pool1->add(op1, false);

  // create a pool of size 2
  auto pool2 = F.make_thread_pool("pool2", 2);
  // add multiple operators, each with thread pinning
  pool2->add({std::move(op2), std::move(op3)}, true);

  EXPECT_EQ(pool1->name(), std::string{"pool1"});
  EXPECT_EQ(pool2->name(), std::string{"pool2"});

  // check that the associated operators are as expected
  auto pool1_ops = pool1->operators();
  EXPECT_EQ(pool1_ops.size(), 1);
  EXPECT_EQ(pool1_ops[0]->name(), std::string{"op1"});
  auto pool2_ops = pool2->operators();
  EXPECT_EQ(pool2_ops.size(), 2);
  EXPECT_EQ(pool2_ops[0]->name(), std::string{"op2"});
  EXPECT_EQ(pool2_ops[1]->name(), std::string{"op3"});

  // description contains the GXF typename and info on operators in the pool
  auto description1 = pool1->description();
  ASSERT_TRUE(description1.find("gxf_typename: nvidia::gxf::ThreadPool") != std::string::npos);
  ASSERT_TRUE(description1.find("operators in pool") != std::string::npos);
}

TEST(Application, TestAddOperatorsWithSameName) {
  Fragment F;

  auto tx = F.make_operator<ops::PingTxOp>("op");
  auto rx = F.make_operator<ops::PingRxOp>("op");

  F.add_operator(tx);

  EXPECT_THROW(F.add_operator(rx), holoscan::RuntimeError);
  EXPECT_THROW(F.add_flow(tx, rx, {{"out", "in"}}), holoscan::RuntimeError);
}

TEST(Fragment, TestAddFlow) {
  Fragment F;

  auto tx = F.make_operator<ops::PingTxOp>("tx");
  auto rx = F.make_operator<ops::PingRxOp>("rx");

  F.add_flow(tx, rx, {{"out", "in"}});

  // First call to graph creates a FlowGraph object
  // F.graph() returns a reference to the abstract Graph base class so use
  // static_cast here
  OperatorFlowGraph& G = static_cast<OperatorFlowGraph&>(F.graph());

  // verify that the operators and edges were added to the graph
  auto nodes = G.get_nodes();
  EXPECT_EQ(nodes.size(), 2);
  auto find_tx = G.find_node([&tx](const holoscan::OperatorNodeType& node) { return node == tx; });
  auto find_rx = G.find_node([&rx](const holoscan::OperatorNodeType& node) { return node == rx; });
  EXPECT_TRUE(find_tx);
  EXPECT_TRUE(find_rx);
  auto port_map = G.get_port_map(tx, rx);
  EXPECT_TRUE(port_map);
  EXPECT_EQ(std::begin(*port_map.value())->first, "out");
  auto& input_port_set = std::begin(*port_map.value())->second;
  EXPECT_EQ(input_port_set.size(), 1);
  EXPECT_EQ(*(input_port_set.begin()), "in");
}

TEST(Fragment, TestOperatorOrder) {
  Fragment F;

  auto rx2 = F.make_operator<ops::PingRxOp>("rx2");
  auto tx = F.make_operator<ops::PingTxOp>("tx");
  auto rx = F.make_operator<ops::PingRxOp>("rx");
  auto tx2 = F.make_operator<ops::PingTxOp>("tx2");

  F.add_operator(tx2);
  F.add_flow(tx, rx, {{"out", "in"}});
  F.add_flow(tx2, rx2, {{"out", "in"}});

  OperatorFlowGraph& G = static_cast<OperatorFlowGraph&>(F.graph());
  auto order = G.get_nodes();

  const std::vector<std::string> expected_order = {"tx2", "tx", "rx", "rx2"};
  // verify that the operators were added to the graph in the correct order
  EXPECT_EQ(order.size(), expected_order.size());
  for (size_t i = 0; i < order.size(); i++) { EXPECT_EQ(order[i]->name(), expected_order[i]); }
}

TEST(Fragment, TestFragmentExecutor) {
  Fragment F;

  // First call to executor generates an Executor object
  Executor& E = F.executor();

  // this fragment is associated with the executor that was created
  EXPECT_EQ(E.fragment(), &F);
}

// Fragment::make_condition is tested elsewhere in condition_classes.cpp
// Fragment::make_resource is tested elsewhere in resource_classes.cpp
// Fragment::make_operator is tested elsewhere in operator_classes.cpp

}  // namespace holoscan
