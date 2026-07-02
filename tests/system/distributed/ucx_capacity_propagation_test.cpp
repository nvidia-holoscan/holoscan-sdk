/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <string>

#include <holoscan/holoscan.hpp>

#include "distributed_app_fixture.hpp"

namespace holoscan {

namespace {

// Number of messages emitted in a single compute() call. Must be > 1 so that the bug
// (capacity reverting to 1 on cross-fragment UCX ports) causes the second emit to fail.
constexpr int kNumMessages = 8;

// Transmitter that emits multiple messages in a single compute() on a UCX output port
// explicitly configured with capacity = kNumMessages. This mirrors the CLARAHOLOS-2896
// repro: `spec.output(...).connector(IOSpec.ConnectorType.UCX, capacity=N)`.
//
// With the capacity correctly propagated across the fragment boundary, all emits succeed.
// Without it, the transmitter queue capacity is 1 and the second emit fails to publish
// (default 'fault' policy), which logs an error and throws before the completion log below.
class MultiEmitTxOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(MultiEmitTxOp)

  MultiEmitTxOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.output<gxf::Entity>("out").connector(IOSpec::ConnectorType::kUCX,
                                              Arg("capacity", static_cast<uint64_t>(kNumMessages)));
  }

  void compute([[maybe_unused]] InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override {
    for (int i = 0; i < kNumMessages; ++i) {
      auto out_message = gxf::Entity::New(&context);
      op_output.emit(out_message, "out");
    }
    HOLOSCAN_LOG_INFO("MultiEmitTxOp emitted {} messages in one compute", kNumMessages);
  }
};

class CountRxOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(CountRxOp)

  CountRxOp() = default;

  void setup(OperatorSpec& spec) override { spec.input<gxf::Entity>("in"); }

  void compute(InputContext& op_input, [[maybe_unused]] OutputContext& op_output,
               [[maybe_unused]] ExecutionContext& context) override {
    auto in_message = op_input.receive<gxf::Entity>("in");
    if (in_message) {
      HOLOSCAN_LOG_INFO("CountRxOp received message {}", ++count_);
    }
  }

 private:
  int count_ = 0;
};

class TxFragment : public Fragment {
 public:
  void compose() override {
    auto tx = make_operator<MultiEmitTxOp>("tx", make_condition<CountCondition>(1));
    add_operator(tx);
  }
};

class RxFragment : public Fragment {
 public:
  void compose() override {
    auto rx = make_operator<CountRxOp>("rx");
    add_operator(rx);
  }
};

class UcxCapacityApp : public Application {
 public:
  using Application::Application;

  void compose() override {
    auto tx_fragment = make_fragment<TxFragment>("fragment1");
    auto rx_fragment = make_fragment<RxFragment>("fragment2");
    add_flow(tx_fragment, rx_fragment, {{"tx.out", "rx.in"}});
  }
};

}  // namespace

// Regression test for CLARAHOLOS-2896: a UCX output port configured with capacity > 1 on a
// cross-fragment connection must retain that capacity at runtime instead of silently reverting
// to 1.
TEST_F(DistributedApp, TestUcxOutputCapacityPropagatedAcrossFragments) {
  auto app = make_application<UcxCapacityApp>();

  testing::internal::CaptureStderr();

  try {
    app->run();
  } catch (const std::exception& e) {
    HOLOSCAN_LOG_ERROR("Exception: {}", e.what());
  }

  std::string log_output = testing::internal::GetCapturedStderr();

  // Without the fix the cross-fragment UCX transmitter capacity reverts to 1, so the second emit
  // in compute() fails to publish (gxf_io_context.cpp), which logs this error and throws.
  EXPECT_EQ(log_output.find("failed to publish output message"), std::string::npos)
      << "Cross-fragment UCX output capacity was not honored (capacity > 1 discarded).\n"
      << "=== LOG ===\n"
      << log_output << "\n===========\n";

  // The transmitter only reaches this log if all kNumMessages emits in a single compute succeeded.
  const std::string all_emitted_log =
      "MultiEmitTxOp emitted " + std::to_string(kNumMessages) + " messages";
  EXPECT_TRUE(log_output.find(all_emitted_log) != std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
}

}  // namespace holoscan
