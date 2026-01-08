/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <fmt/ranges.h>

#include <any>
#include <memory>
#include <set>
#include <string>
#include <utility>

#include <holoscan/core/gxf/entity.hpp>
#include <holoscan/holoscan.hpp>
#include <holoscan/operators/ping_rx/ping_rx.hpp>
#include <holoscan/operators/ping_tx/ping_tx.hpp>

#include "env_wrapper.hpp"

namespace holoscan {

namespace ops {

class ForwardingOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(ForwardingOp)

  ForwardingOp() = default;

  void setup(OperatorSpec& spec) override;

  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override;
};

void ForwardingOp::setup(OperatorSpec& spec) {
  spec.input<std::any>("in");
  spec.output<std::any>("out");
}

void ForwardingOp::compute(InputContext& op_input, OutputContext& op_output,
                           [[maybe_unused]] ExecutionContext& context) {
  auto in_message = op_input.receive<std::any>("in");
  if (in_message) {
    auto value = in_message.value();
    if (value.type() == typeid(holoscan::gxf::Entity)) {
      // emit as entity
      auto entity = std::any_cast<holoscan::gxf::Entity>(value);
      op_output.emit(entity, "out");
    } else {
      // emit as std::any
      op_output.emit(value, "out");
    }
  }
}

// Custom PingRxOp that can receive from multiple sources
class MultiPingRxOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(MultiPingRxOp)

  MultiPingRxOp() = default;

  void setup(OperatorSpec& spec) override {
    // Define multi-receiver input port that can accept any number of connections
    spec.input<std::vector<int>>("receivers", IOSpec::kAnySize);
  }

  void compute(InputContext& op_input, OutputContext&, ExecutionContext&) override {
    auto value_vector = op_input.receive<std::vector<int>>("receivers").value();

    HOLOSCAN_LOG_INFO("Rx message received (count: {}, size: {})", count_++, value_vector.size());

    for (size_t i = 0; i < value_vector.size(); i++) {
      HOLOSCAN_LOG_INFO("Rx message value[{}]: {}", i, value_vector[i]);
    }
  }

 private:
  int count_ = 1;
};

}  // namespace ops
}  // namespace holoscan

/**
 * @brief Subgraph containing a single PingTxOp transmitter
 */
class PingTxSubgraph : public holoscan::Subgraph {
 public:
  PingTxSubgraph(holoscan::Fragment* fragment, const std::string& name)
      : holoscan::Subgraph(fragment, name) {}

  void compose() override {
    using namespace holoscan;

    // Create a PingTxOp with a count condition
    auto tx_op = make_operator<ops::PingTxOp>(
        "transmitter",
        make_condition<CountCondition>(8));  // Send 8 messages per transmitter

    auto forwarding_op = make_operator<ops::ForwardingOp>("forwarding");

    // Add the operator to this subgraph
    // add_operator(tx_op);
    // add_operator(forwarding_op);
    add_flow(tx_op, forwarding_op);

    // Expose the "out" port so external operators can connect to it
    add_output_interface_port("data_out", forwarding_op, "out");
  }
};

/**
 * @brief Subgraph containing a multi-receiver MultiPingRxOp
 */
class MultiPingRxSubgraph : public holoscan::Subgraph {
 public:
  MultiPingRxSubgraph(holoscan::Fragment* fragment, const std::string& name)
      : holoscan::Subgraph(fragment, name) {}

  void compose() override {
    using namespace holoscan;

    // Create a multi-receiver MultiPingRxOp
    auto rx_op = make_operator<ops::MultiPingRxOp>("multi_receiver");

    // Add the operator to this subgraph
    add_operator(rx_op);

    // Expose the "receivers" port so multiple external operators can connect to it
    add_input_interface_port("data_in", rx_op, "receivers");
  }
};

/**
 * @brief Subgraph containing a single-receiver PingRxOp
 */
class PingRxSubgraph : public holoscan::Subgraph {
 public:
  PingRxSubgraph(holoscan::Fragment* fragment, const std::string& name)
      : holoscan::Subgraph(fragment, name) {}

  void compose() override {
    using namespace holoscan;

    // Create a PingRxOp
    auto rx_op = make_operator<ops::PingRxOp>("receiver");

    // Add the operator to this subgraph
    add_operator(rx_op);

    // Expose the "in" port so external operators can connect to it
    add_input_interface_port("data_in", rx_op, "in");
  }
};

/**
 * @brief Nested Subgraph that contains PingTxSubgraph connected to ForwardingOp
 */
class NestedTxSubgraph : public holoscan::Subgraph {
 public:
  NestedTxSubgraph(holoscan::Fragment* fragment, const std::string& name)
      : holoscan::Subgraph(fragment, name) {}

  void compose() override {
    using namespace holoscan;

    // Create a nested PingTxSubgraph
    auto ping_tx_subgraph = make_subgraph<PingTxSubgraph>("ping_tx");

    // Create a ForwardingOp
    auto forwarding_op = make_operator<ops::ForwardingOp>("forwarding");

    // not necessary, but shouldn't hurt if add_operator was called explicitly before add_flow
    add_operator(forwarding_op);

    // Connect the nested subgraph to the forwarding operator
    add_flow(ping_tx_subgraph, forwarding_op, {{"data_out", "in"}});

    // Expose the forwarding operator's output as our interface
    add_output_interface_port("data_out", forwarding_op, "out");
  }
};

/**
 * @brief Nested Subgraph that contains ForwardingOp connected to PingRxSubgraph
 */
class NestedRxSubgraph : public holoscan::Subgraph {
 public:
  NestedRxSubgraph(holoscan::Fragment* fragment, const std::string& name)
      : holoscan::Subgraph(fragment, name) {}

  void compose() override {
    using namespace holoscan;

    // Create a ForwardingOp
    auto forwarding_op = make_operator<ops::ForwardingOp>("forwarding");

    // Create a nested PingRxSubgraph
    auto ping_rx_subgraph = make_subgraph<PingRxSubgraph>("ping_rx");

    // Connect the forwarding operator to the nested subgraph
    add_flow(forwarding_op, ping_rx_subgraph, {{"out", "data_in"}});

    // Expose the forwarding operator's input as our interface
    add_input_interface_port("data_in", forwarding_op, "in");
  }
};

/**
 * @brief Double nested Subgraph that contains only NestedRxSubgraph
 *
 * This demonstrates exposing a nested subgraph's interface port directly as the parent's
 * interface port using the add_input_interface_port overload that accepts Subgraph.
 */
class DoubleNestedRxSubgraph : public holoscan::Subgraph {
 public:
  DoubleNestedRxSubgraph(holoscan::Fragment* fragment, const std::string& name)
      : holoscan::Subgraph(fragment, name) {}

  void compose() override {
    using namespace holoscan;

    // Create a nested NestedRxSubgraph
    auto nested_rx_subgraph = make_subgraph<NestedRxSubgraph>("nested_rx");

    // Expose the nested subgraph's interface port as our own interface port
    // This uses the overload that takes std::shared_ptr<Subgraph>
    add_input_interface_port("data_receiver", nested_rx_subgraph, "data_in");
  }
};

/**
 * @brief Application demonstrating Subgraph reusability with multiple instances
 *
 * This application creates:
 * - 3 instances of PingTxSubgraph (each will create operators named tx1_transmitter,
 *   tx3_transmitter, tx4_transmitter)
 * - 1 instance of PingTxOp operator "tx2"
 * - 1 instance of MultiPingRxSubgraph (creates operator named multi_rx_multi_receiver)
 * - All transmitters connect to the single multi-receiver via exposed ports
 *
 * The underscore separator ensures no conflicts with Holoscan's reserved '.' character.
 */
class SubgraphPingApplication : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;

    // Create 3 transmitter subgraphs and one transmitter operator
    auto tx_instance1 = make_subgraph<PingTxSubgraph>("tx1");
    auto tx2 = make_operator<ops::PingTxOp>("tx2", make_condition<CountCondition>(8));
    auto tx_instance3 = make_subgraph<PingTxSubgraph>("tx3");
    auto tx_instance4 = make_subgraph<PingTxSubgraph>("tx4");

    // Create 3 receiver subgraphs and one receiver operator
    auto rx_instance1 = make_subgraph<PingRxSubgraph>("rx1");
    auto rx_instance2 = make_subgraph<PingRxSubgraph>("rx2");
    auto rx3 = make_operator<ops::PingRxOp>("rx3");
    auto rx_instance4 = make_subgraph<PingRxSubgraph>("rx4");

    // Make various types of 1:1 connections
    add_flow(tx_instance1, rx_instance1, {{"data_out", "data_in"}});
    add_flow(tx2, rx_instance2, {{"out", "data_in"}});
    add_flow(tx_instance3, rx3, {{"data_out", "in"}});
    add_flow(tx_instance4, rx_instance4, {{"data_out", "data_in"}});
  }
};

/**
 * @brief Application demonstrating Subgraph reusability with multiple instances
 *
 * This application creates:
 * - 4 instances of PingTxSubgraph (each will create operators named tx1_transmitter,
 *   tx2_transmitter, tx3_transmitter, tx4_transmitter)
 * - 1 instance of MultiPingRxSubgraph (creates operator named multi_rx_multi_receiver)
 * - All transmitters connect to the single multi-receiver via exposed ports
 *
 * The underscore separator ensures no conflicts with Holoscan's reserved '.' character.
 */
class SubgraphPingApplicationWithoutPortMap : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;

    // Create 3 transmitter subgraphs and one transmitter operator
    auto tx_instance1 = make_subgraph<PingTxSubgraph>("tx1");
    auto tx2 = make_operator<ops::PingTxOp>("tx2", make_condition<CountCondition>(8));
    auto tx_instance3 = make_subgraph<PingTxSubgraph>("tx3");
    auto tx_instance4 = make_subgraph<PingTxSubgraph>("tx4");

    // Create 3 receiver subgraphs and one receiver operator
    auto rx_instance1 = make_subgraph<PingRxSubgraph>("rx1");
    auto rx_instance2 = make_subgraph<PingRxSubgraph>("rx2");
    auto rx3 = make_operator<ops::PingRxOp>("rx3");
    auto rx_instance4 = make_subgraph<PingRxSubgraph>("rx4");

    // Make various types of 1:1 connections
    add_flow(tx_instance1, rx_instance1);
    add_flow(tx2, rx_instance2);
    add_flow(tx_instance3, rx3);
    add_flow(tx_instance4, rx_instance4);
  }
};

/**
 * @brief Application demonstrating Subgraph reusability with multiple instances
 *
 * This application creates:
 * - 4 instances of PingTxSubgraph (each will create operators named tx1_transmitter,
 *   tx2_transmitter, tx3_transmitter, tx4_transmitter)
 * - 1 instance of MultiPingRxSubgraph (creates operator named multi_rx_multi_receiver)
 * - All transmitters connect to the single multi-receiver via exposed ports
 *
 * The underscore separator ensures no conflicts with Holoscan's reserved '.' character.
 */
class MultiPingApplication : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;

    // Create 3 instances of the transmitter subgraph and one transmitter operator
    auto tx_instance1 = make_subgraph<PingTxSubgraph>("tx1");
    auto tx2 = make_operator<ops::PingTxOp>("tx2", make_condition<CountCondition>(8));
    auto tx_instance3 = make_subgraph<PingTxSubgraph>("tx3");
    auto tx_instance4 = make_subgraph<PingTxSubgraph>("tx4");

    auto rx_instance = make_subgraph<MultiPingRxSubgraph>("multi_rx");

    // Create N:1 connections from operators and subgraphs to the multi_rx subgraph
    add_flow(tx_instance1, rx_instance, {{"data_out", "data_in"}});
    add_flow(tx2, rx_instance, {{"out", "data_in"}});
    add_flow(tx_instance3, rx_instance, {{"data_out", "data_in"}});
    add_flow(tx_instance4, rx_instance, {{"data_out", "data_in"}});
  }
};

/**
 * @brief Application demonstrating nested Subgraphs
 *
 * Architecture: NestedTxSubgraph -> ForwardingOp -> NestedRxSubgraph
 * Where:
 * - NestedTxSubgraph contains: PingTxSubgraph -> ForwardingOp
 * - NestedRxSubgraph contains: ForwardingOp -> PingRxSubgraph
 */
class NestedSubgraphApplication : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;

    // Create nested subgraphs
    auto nested_tx = make_subgraph<NestedTxSubgraph>("nested_tx");
    auto nested_rx = make_subgraph<NestedRxSubgraph>("nested_rx");

    // Create a middle ForwardingOp to connect the nested subgraphs
    auto middle_forwarding = make_operator<ops::ForwardingOp>("middle_forwarding");

    // Connect: NestedTxSubgraph -> ForwardingOp -> NestedRxSubgraph
    add_flow(nested_tx, middle_forwarding, {{"data_out", "in"}});
    add_flow(middle_forwarding, nested_rx, {{"out", "data_in"}});
  }
};

// application that should fail during initialization due to duplicate subgraph names
class InvalidSubgraphApplication : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;

    // Create both tx subgraphs with same instance name (should throw an error)
    auto tx1 = make_subgraph<PingTxSubgraph>("tx");
    auto tx2 = make_subgraph<PingTxSubgraph>("tx");

    // Create two receiver subgraphs
    auto rx1 = make_subgraph<PingRxSubgraph>("rx");
    auto rx2 = make_subgraph<PingRxSubgraph>("rx");

    // Make various types of 1:1 connections
    add_flow(tx1, rx1, {{"data_out", "data_in"}});
    add_flow(tx2, rx2, {{"data_out", "data_in"}});
  }
};

// Test fixture for parameterized testing
class SubgraphOneToOneTest : public ::testing::TestWithParam<bool> {
 protected:
  // GetParam() returns true for SubgraphPingApplication, false for
  // SubgraphPingApplicationWithoutPortMap
  bool use_port_map() const { return GetParam(); }
};

TEST_P(SubgraphOneToOneTest, TestSubgraphsAndOperators_OneToOne) {
  using namespace holoscan;

  EnvVarWrapper wrapper({
      std::make_pair("HOLOSCAN_LOG_LEVEL", "DEBUG"),
  });

  std::shared_ptr<holoscan::Application> app;
  if (use_port_map()) {
    app = holoscan::make_application<SubgraphPingApplication>();
  } else {
    app = holoscan::make_application<SubgraphPingApplicationWithoutPortMap>();
  }

  // capture output to check that the expected messages were logged
  testing::internal::CaptureStderr();

  // test ability to get port map description
  app->compose_graph();
  std::string port_map_yaml = app->graph().port_map_description();

  app->run();

  std::string log_output = testing::internal::GetCapturedStderr();

  // Check that the node names are as expected
  std::set<std::string> node_names;
  for (const auto& node : app->graph().get_nodes()) {
    node_names.insert(node->name());
  }

  std::set<std::string> expected_names{"tx1_transmitter",
                                       "tx1_forwarding",
                                       "tx2",
                                       "tx3_transmitter",
                                       "tx3_forwarding",
                                       "tx4_transmitter",
                                       "tx4_forwarding",
                                       "rx1_receiver",
                                       "rx2_receiver",
                                       "rx3",
                                       "rx4_receiver"};

  // Check that both sets contain exactly the same names
  EXPECT_EQ(node_names, expected_names) << fmt::format(
      "Node names don't match expected names.\n"
      "Actual nodes: {}\n"
      "Expected nodes: {}\n",
      node_names,
      expected_names);

  // Check that messages were received
  std::string stream_msg = "Rx message value: 8";

  // Count occurrences of the message
  size_t count = 0;
  size_t pos = 0;
  while ((pos = log_output.find(stream_msg, pos)) != std::string::npos) {
    count++;
    pos += stream_msg.length();
  }

  EXPECT_EQ(count, 4) << "Expected exactly 4 occurrences of '" << stream_msg << "', but found "
                      << count << "\n=== LOG ===\n"
                      << log_output << "\n===========\n";
}

// Instantiate the test suite with both parameter values
INSTANTIATE_TEST_SUITE_P(WithAndWithoutPortMap, SubgraphOneToOneTest,
                         ::testing::Values(true, false),
                         [](const ::testing::TestParamInfo<bool>& info) {
                           return info.param ? "WithPortMap" : "WithoutPortMap";
                         });

TEST(SubgraphTests, TestSubgraphsAndOperators_MultiToOne) {
  using namespace holoscan;

  EnvVarWrapper wrapper({
      std::make_pair("HOLOSCAN_LOG_LEVEL", "DEBUG"),
  });

  auto app = holoscan::make_application<MultiPingApplication>();

  // capture output to check that the expected messages were logged
  testing::internal::CaptureStderr();

  // test ability to get port map description
  app->compose_graph();
  std::string port_map_yaml = app->graph().port_map_description();

  app->run();

  std::string log_output = testing::internal::GetCapturedStderr();

  std::set<std::string> node_names;
  for (const auto& node : app->graph().get_nodes()) {
    node_names.insert(node->name());
  }

  std::set<std::string> expected_names{"tx1_transmitter",
                                       "tx1_forwarding",
                                       "tx2",
                                       "tx3_transmitter",
                                       "tx3_forwarding",
                                       "tx4_transmitter",
                                       "tx4_forwarding",
                                       "multi_rx_multi_receiver"};

  // Check that both sets contain exactly the same names
  EXPECT_EQ(node_names, expected_names) << fmt::format(
      "Node names don't match expected names.\n"
      "Actual nodes: {}\n"
      "Expected nodes: {}\n",
      node_names,
      expected_names);
  // Check that messages were received
  std::string stream_msg = "Rx message received (count: 8, size: 4)";
  EXPECT_TRUE(log_output.find(stream_msg) != std::string::npos) << "=== LOG ===\n"
                                                                << log_output << "\n===========\n";
  stream_msg = "Rx message value[3]: 8";
  EXPECT_TRUE(log_output.find(stream_msg) != std::string::npos) << "=== LOG ===\n"
                                                                << log_output << "\n===========\n";
}

TEST(SubgraphTests, TestNestedSubgraphs) {
  using namespace holoscan;

  EnvVarWrapper wrapper({
      std::make_pair("HOLOSCAN_LOG_LEVEL", "DEBUG"),
  });

  auto app = holoscan::make_application<NestedSubgraphApplication>();

  // capture output to check that the expected messages were logged
  testing::internal::CaptureStderr();

  // test ability to get port map description
  app->compose_graph();
  std::string port_map_yaml = app->graph().port_map_description();

  app->run();

  std::string log_output = testing::internal::GetCapturedStderr();

  std::set<std::string> node_names;
  for (const auto& node : app->graph().get_nodes()) {
    node_names.insert(node->name());
  }

  std::set<std::string> expected_names{"nested_tx_ping_tx_transmitter",
                                       "nested_tx_ping_tx_forwarding",
                                       "nested_tx_forwarding",
                                       "middle_forwarding",
                                       "nested_rx_forwarding",
                                       "nested_rx_ping_rx_receiver"};

  // Check that both sets contain exactly the same names
  EXPECT_EQ(node_names, expected_names) << fmt::format(
      "Node names don't match expected names.\n"
      "Actual nodes: {}\n"
      "Expected nodes: {}\n",
      node_names,
      expected_names);

  // Check that messages were received through the nested subgraph chain
  // The message should pass through: PingTxSubgraph -> ForwardingOp -> ForwardingOp -> ForwardingOp
  // -> PingRxSubgraph
  std::string stream_msg = "Rx message value: 8";
  EXPECT_TRUE(log_output.find(stream_msg) != std::string::npos) << "=== LOG ===\n"
                                                                << log_output << "\n===========\n";
}

TEST(SubgraphTests, TestInvalidDuplicateSubgraphNames) {
  using namespace holoscan;

  auto app = holoscan::make_application<InvalidSubgraphApplication>();

  // Expect compose_graph() to throw an exception due to duplicate subgraph names
  // The exception should be thrown when make_subgraph is called with a duplicate name
  EXPECT_THROW(app->compose_graph(), std::runtime_error);
}

/**
 * @brief Minimal application demonstrating double-nested subgraph interface port exposure
 *
 * This application creates:
 * - 1 PingTxOp transmitter
 * - 1 DoubleNestedRxSubgraph that exposes its nested subgraph's interface port
 *
 * Architecture:
 * PingTxOp -> DoubleNestedRxSubgraph (contains NestedRxSubgraph (contains ForwardingOp ->
 * PingRxSubgraph))
 */
class DoubleNestedSubgraphApplication : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;

    // Create a simple transmitter
    auto tx = make_operator<ops::PingTxOp>("tx", make_condition<CountCondition>(5));

    // Create the double-nested receiver subgraph
    auto double_nested_rx = make_subgraph<DoubleNestedRxSubgraph>("double_nested_rx");

    // Connect transmitter to double-nested receiver using exposed interface port
    add_flow(tx, double_nested_rx, {{"out", "data_receiver"}});
  }
};

TEST(SubgraphTests, TestDoubleNestedSubgraphInterfacePort) {
  using namespace holoscan;
  auto app = holoscan::make_application<DoubleNestedSubgraphApplication>();

  // capture output to check that the expected messages were logged
  testing::internal::CaptureStderr();

  // test ability to get port map description
  app->compose_graph();
  std::string port_map_yaml = app->graph().port_map_description();

  app->run();

  std::string log_output = testing::internal::GetCapturedStderr();

  std::set<std::string> node_names;
  for (const auto& node : app->graph().get_nodes()) {
    node_names.insert(node->name());
  }

  // The hierarchical naming should produce:
  // - tx (the transmitter)
  // - double_nested_rx_nested_rx_forwarding (ForwardingOp in NestedRxSubgraph)
  // - double_nested_rx_nested_rx_ping_rx_receiver (PingRxOp in PingRxSubgraph)
  std::set<std::string> expected_names{
      "tx", "double_nested_rx_nested_rx_forwarding", "double_nested_rx_nested_rx_ping_rx_receiver"};

  // Check that both sets contain exactly the same names
  EXPECT_EQ(node_names, expected_names) << fmt::format(
      "Node names don't match expected names.\n"
      "Actual nodes: {}\n"
      "Expected nodes: {}\n",
      node_names,
      expected_names);

  // Check that messages were received through the double-nested subgraph chain
  // The message should pass through: PingTxOp -> NestedRxSubgraph.ForwardingOp ->
  // PingRxSubgraph.PingRxOp
  std::string stream_msg = "Rx message value: 5";
  EXPECT_TRUE(log_output.find(stream_msg) != std::string::npos)
      << "Expected to find '" << stream_msg << "' in log output\n=== LOG ===\n"
      << log_output << "\n===========\n";
}

// =======================================================================================
// Control Flow Tests with Subgraphs
// =======================================================================================

/**
 * @brief Simple operator that logs when it executes (for control flow testing)
 */
class SimpleExecOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(SimpleExecOp)

  SimpleExecOp() = default;

  void compute([[maybe_unused]] holoscan::InputContext& op_input,
               [[maybe_unused]] holoscan::OutputContext& op_output,
               [[maybe_unused]] holoscan::ExecutionContext& context) override {
    HOLOSCAN_LOG_INFO("SimpleExecOp: {} executed", name());
  }
};

/**
 * @brief Subgraph containing sequential execution flow (node2 -> node3)
 */
class SequentialExecSubgraph : public holoscan::Subgraph {
 public:
  SequentialExecSubgraph(holoscan::Fragment* frag, const std::string& name)
      : Subgraph(frag, name) {}

  void compose() override {
    using namespace holoscan;

    // Define the operators inside the subgraph
    auto node2 = make_operator<SimpleExecOp>("node2");
    auto node3 = make_operator<SimpleExecOp>("node3");

    // Define the sequential workflow inside the subgraph
    add_flow(node2, node3);

    // Expose node2's input execution port as the subgraph's input execution interface port
    add_input_exec_interface_port("exec_in", node2);

    // Expose node3's output execution port as the subgraph's output execution interface port
    add_output_exec_interface_port("exec_out", node3);
  }
};

/**
 * @brief Application demonstrating control flow with Subgraph
 *
 * Tests sequential execution: node1 -> SequentialExecSubgraph (node2 -> node3) -> node4
 */
class SequentialExecWithSubgraphApp : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;

    // Define operators outside the subgraph
    auto node1 = make_operator<SimpleExecOp>("node1");
    auto node4 = make_operator<SimpleExecOp>("node4");

    // Create the subgraph containing node2 and node3
    auto sequential_sg = make_subgraph<SequentialExecSubgraph>("sequential_sg");

    // Define the sequential workflow using control flow
    // start_op() -> node1 -> sequential_sg (node2 -> node3) -> node4
    add_flow(start_op(), node1);
    add_flow(node1, sequential_sg);  // Auto-resolves to exec_in
    add_flow(sequential_sg, node4);  // Auto-resolves to exec_out
  }
};

/**
 * @brief Subgraph with only input execution interface port
 */
class InputExecSubgraph : public holoscan::Subgraph {
 public:
  InputExecSubgraph(holoscan::Fragment* frag, const std::string& name) : Subgraph(frag, name) {}

  void compose() override {
    using namespace holoscan;

    auto node = make_operator<SimpleExecOp>("node");
    add_input_exec_interface_port("exec_in", node);
  }
};

/**
 * @brief Subgraph with both input and output execution interface ports (passthrough)
 */
class OutputExecSubgraph : public holoscan::Subgraph {
 public:
  OutputExecSubgraph(holoscan::Fragment* frag, const std::string& name) : Subgraph(frag, name) {}

  void compose() override {
    using namespace holoscan;

    auto node = make_operator<SimpleExecOp>("node");
    // Need both input and output ports for the operator to be triggered and to trigger downstream
    add_input_exec_interface_port("exec_in", node);
    add_output_exec_interface_port("exec_out", node);
  }
};

/**
 * @brief Application testing input execution interface port
 */
class InputExecSubgraphApp : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;

    auto node1 = make_operator<SimpleExecOp>("node1");
    auto input_sg = make_subgraph<InputExecSubgraph>("input_sg");

    add_flow(start_op(), node1);
    add_flow(node1, input_sg);  // Auto-resolves to exec_in
  }
};

/**
 * @brief Application testing output execution interface port
 */
class OutputExecSubgraphApp : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;

    auto output_sg = make_subgraph<OutputExecSubgraph>("output_sg");
    auto node2 = make_operator<SimpleExecOp>("node2");

    add_flow(start_op(), output_sg);
    add_flow(output_sg, node2);  // Auto-resolves from exec_out
  }
};

/**
 * @brief Nested Subgraph with execution interface ports
 */
class NestedExecSubgraph : public holoscan::Subgraph {
 public:
  NestedExecSubgraph(holoscan::Fragment* frag, const std::string& name) : Subgraph(frag, name) {}

  void compose() override {
    using namespace holoscan;

    // Create a nested sequential subgraph
    auto sequential_sg = make_subgraph<SequentialExecSubgraph>("sequential_sg");

    // Expose the nested subgraph's execution interface ports
    add_input_exec_interface_port("exec_in", sequential_sg, "exec_in");
    add_output_exec_interface_port("exec_out", sequential_sg, "exec_out");
  }
};

/**
 * @brief Application testing nested Subgraph with execution interface ports
 */
class NestedExecSubgraphApp : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;

    auto node1 = make_operator<SimpleExecOp>("node1");
    auto nested_exec_sg = make_subgraph<NestedExecSubgraph>("nested_exec_sg");
    auto node4 = make_operator<SimpleExecOp>("node4");

    add_flow(start_op(), node1);
    add_flow(node1, nested_exec_sg);
    add_flow(nested_exec_sg, node4);
  }
};

TEST(SubgraphControlFlowTests, TestSequentialExecWithSubgraph) {
  using namespace holoscan;

  auto app = holoscan::make_application<SequentialExecWithSubgraphApp>();

  // Capture output to check execution order
  testing::internal::CaptureStderr();

  app->compose_graph();
  app->run();

  std::string log_output = testing::internal::GetCapturedStderr();

  // Check that all nodes are present in the graph
  std::set<std::string> node_names;
  for (const auto& node : app->graph().get_nodes()) {
    node_names.insert(node->name());
  }

  std::set<std::string> expected_names{
      "<|start|>", "node1", "sequential_sg_node2", "sequential_sg_node3", "node4"};

  EXPECT_EQ(node_names, expected_names) << fmt::format(
      "Node names don't match expected names.\n"
      "Actual nodes: {}\n"
      "Expected nodes: {}\n",
      node_names,
      expected_names);

  // Check that all operators executed
  EXPECT_TRUE(log_output.find("SimpleExecOp: node1 executed") != std::string::npos)
      << "node1 did not execute\n=== LOG ===\n"
      << log_output << "\n===========\n";
  EXPECT_TRUE(log_output.find("SimpleExecOp: sequential_sg_node2 executed") != std::string::npos)
      << "sequential_sg_node2 did not execute\n=== LOG ===\n"
      << log_output << "\n===========\n";
  EXPECT_TRUE(log_output.find("SimpleExecOp: sequential_sg_node3 executed") != std::string::npos)
      << "sequential_sg_node3 did not execute\n=== LOG ===\n"
      << log_output << "\n===========\n";
  EXPECT_TRUE(log_output.find("SimpleExecOp: node4 executed") != std::string::npos)
      << "node4 did not execute\n=== LOG ===\n"
      << log_output << "\n===========\n";
}

TEST(SubgraphControlFlowTests, TestInputExecInterfacePort) {
  using namespace holoscan;

  auto app = holoscan::make_application<InputExecSubgraphApp>();

  // Capture output to check execution
  testing::internal::CaptureStderr();

  app->compose_graph();
  app->run();

  std::string log_output = testing::internal::GetCapturedStderr();

  // Check that nodes are present
  std::set<std::string> node_names;
  for (const auto& node : app->graph().get_nodes()) {
    node_names.insert(node->name());
  }

  std::set<std::string> expected_names{"<|start|>", "node1", "input_sg_node"};

  EXPECT_EQ(node_names, expected_names);

  // Check that both operators executed
  EXPECT_TRUE(log_output.find("SimpleExecOp: node1 executed") != std::string::npos);
  EXPECT_TRUE(log_output.find("SimpleExecOp: input_sg_node executed") != std::string::npos);
}

TEST(SubgraphControlFlowTests, TestOutputExecInterfacePort) {
  using namespace holoscan;

  auto app = holoscan::make_application<OutputExecSubgraphApp>();

  // Capture output to check execution
  testing::internal::CaptureStderr();

  app->compose_graph();
  app->run();

  std::string log_output = testing::internal::GetCapturedStderr();

  // Check that nodes are present
  std::set<std::string> node_names;
  for (const auto& node : app->graph().get_nodes()) {
    node_names.insert(node->name());
  }

  std::set<std::string> expected_names{"<|start|>", "output_sg_node", "node2"};

  EXPECT_EQ(node_names, expected_names);

  // Check that both operators executed
  EXPECT_TRUE(log_output.find("SimpleExecOp: output_sg_node executed") != std::string::npos);
  EXPECT_TRUE(log_output.find("SimpleExecOp: node2 executed") != std::string::npos);
}

TEST(SubgraphControlFlowTests, TestNestedExecSubgraph) {
  using namespace holoscan;

  auto app = holoscan::make_application<NestedExecSubgraphApp>();

  // Capture output to check execution
  testing::internal::CaptureStderr();

  app->compose_graph();
  app->run();

  std::string log_output = testing::internal::GetCapturedStderr();

  // Check that all nested nodes are present with hierarchical naming
  std::set<std::string> node_names;
  for (const auto& node : app->graph().get_nodes()) {
    node_names.insert(node->name());
  }

  std::set<std::string> expected_names{"<|start|>",
                                       "node1",
                                       "nested_exec_sg_sequential_sg_node2",
                                       "nested_exec_sg_sequential_sg_node3",
                                       "node4"};

  EXPECT_EQ(node_names, expected_names) << fmt::format(
      "Node names don't match expected names.\n"
      "Actual nodes: {}\n"
      "Expected nodes: {}\n",
      node_names,
      expected_names);

  // Check that all operators executed in the correct order
  EXPECT_TRUE(log_output.find("SimpleExecOp: node1 executed") != std::string::npos);
  EXPECT_TRUE(log_output.find("SimpleExecOp: nested_exec_sg_sequential_sg_node2 executed") !=
              std::string::npos);
  EXPECT_TRUE(log_output.find("SimpleExecOp: nested_exec_sg_sequential_sg_node3 executed") !=
              std::string::npos);
  EXPECT_TRUE(log_output.find("SimpleExecOp: node4 executed") != std::string::npos);
}
