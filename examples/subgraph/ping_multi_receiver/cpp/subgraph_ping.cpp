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

#include <any>
#include <string>
#include <vector>

#include <holoscan/core/gxf/entity.hpp>
#include <holoscan/holoscan.hpp>
#include <holoscan/operators/ping_rx/ping_rx.hpp>
#include <holoscan/operators/ping_tx/ping_tx.hpp>

namespace holoscan::ops {

class ForwardingOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(ForwardingOp)

  ForwardingOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.input<std::any>("in");
    spec.output<std::any>("out");
  }

  void compute(InputContext& op_input, OutputContext& op_output,
               [[maybe_unused]] ExecutionContext& context) override {
    auto in_message = op_input.receive<std::any>("in");
    // emit as std::any
    op_output.emit(in_message.value(), "out");
  }
};

// Custom PingRxOp that can receive from multiple sources
class MultiPingRxOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(MultiPingRxOp)

  MultiPingRxOp() = default;

  void setup(OperatorSpec& spec) override {
    // Define multi-receiver input port that can accept any number of connections
    spec.input<std::vector<int>>("receivers", IOSpec::kAnySize);
  }

  void compute(InputContext& op_input, [[maybe_unused]] OutputContext& op_output,
               [[maybe_unused]] ExecutionContext& context) override {
    auto value_vector = op_input.receive<std::vector<int>>("receivers").value();

    HOLOSCAN_LOG_INFO("Rx message received (count: {}, size: {})", count_++, value_vector.size());

    for (size_t i = 0; i < value_vector.size(); i++) {
      HOLOSCAN_LOG_INFO("Rx message value[{}]: {}", i, value_vector[i]);
    }
  }

 private:
  int count_ = 1;
};

}  // namespace holoscan::ops

/**
 * @brief Subgraph containing a single PingTxOp transmitter
 */
class PingTxSubgraph : public holoscan::Subgraph {
 public:
  PingTxSubgraph(holoscan::Fragment* fragment, const std::string& instance_name)
      : holoscan::Subgraph(fragment, instance_name) {}

  void compose() override {
    using namespace holoscan;

    // Create a PingTxOp with a count condition (send 8 messages total)
    auto tx_op = make_operator<ops::PingTxOp>("transmitter", make_condition<CountCondition>(8));
    auto forwarding_op = make_operator<ops::ForwardingOp>("forwarding");

    // Add the operator to this subgraph
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
  MultiPingRxSubgraph(holoscan::Fragment* fragment, const std::string& instance_name)
      : holoscan::Subgraph(fragment, instance_name) {}

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
  PingRxSubgraph(holoscan::Fragment* fragment, const std::string& instance_name)
      : holoscan::Subgraph(fragment, instance_name) {}

  void compose() override {
    using namespace holoscan;

    // Create a single-receiver PingRxOp
    auto rx_op = make_operator<ops::PingRxOp>("receiver");

    // Add the operator to this subgraph
    add_operator(rx_op);

    // Expose the "in" port so external operators can connect to it
    add_input_interface_port("data_in", rx_op, "in");
  }
};

/**
 * @brief Application demonstrating Subgraph reusability with multiple instances
 *
 * This application creates:
 * - 3 instances of PingTxSubgraph (will create operators named "tx1_transmitter",
 *   "tx2_transmitter", "tx4_transmitter", etc.)
 * - 1 instance of PingTxOp operator "tx2"
 * - 1 instance of MultiPingRxSubgraph (creates operator named multi_rx_multi_receiver)
 * - All transmitters connect to the single multi-receiver via exposed ports
 */
class MultiPingApplication : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;

    // Create 3 transmitter subgraphs and one standalone transmitter operator
    auto tx_instance1 = make_subgraph<PingTxSubgraph>("tx1");
    auto tx2 = make_operator<ops::PingTxOp>("tx2", make_condition<CountCondition>(8));
    auto tx_instance3 = make_subgraph<PingTxSubgraph>("tx3");
    auto tx_instance4 = make_subgraph<PingTxSubgraph>("tx4");

    // Create one instance of the multi-receiver subgraph
    // This will create internal operator: multi_rx_multi_receiver
    auto rx_instance = make_subgraph<MultiPingRxSubgraph>("multi_rx");

    // Subgraphs are automatically added to the Fragment when created with make_subgraph

    // Connect all transmitters to the multi-receiver using exposed port names
    // Each PingTxSubgraph's "data_out" port (maps to internal "out" port) connects
    // to the receiver's "data_in" port (maps to internal "receivers" port)
    add_flow(tx_instance1, rx_instance, {{"data_out", "data_in"}});
    add_flow(tx2, rx_instance, {{"out", "data_in"}});
    add_flow(tx_instance3, rx_instance, {{"data_out", "data_in"}});
    add_flow(tx_instance4, rx_instance, {{"data_out", "data_in"}});

    HOLOSCAN_LOG_INFO("Application composed with Subgraph instances:");
  }
};

int main() {
  auto app = holoscan::make_application<MultiPingApplication>();

  // optional code to visualize the flattened graph and port mapping
  app->compose_graph();
  std::string port_map_yaml = app->graph().port_map_description();
  HOLOSCAN_LOG_INFO("====== PORT MAPPING =======\n{}", port_map_yaml);

  // run the application
  app->run();

  return 0;
}
