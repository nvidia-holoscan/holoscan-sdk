/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <gxf/core/gxf.h>

#include <string>

#include <holoscan/holoscan.hpp>
#include "../config.hpp"
#include "common/assert.hpp"

#include "ping_rx_op.hpp"
#include "ping_tx_op.hpp"

using namespace std::string_literals;

static HoloscanTestConfig test_config;

namespace holoscan {

class NativeMultiBroadcastsApp : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;
    auto tx = make_operator<ops::PingTxOp>("tx", make_condition<CountCondition>(10));
    auto rx11 = make_operator<ops::PingRxOp>("rx11");
    auto rx12 = make_operator<ops::PingRxOp>("rx12");
    auto rx21 = make_operator<ops::PingRxOp>("rx21");
    auto rx22 = make_operator<ops::PingRxOp>("rx22");

    add_flow(tx, rx11, {{"out1", "receivers"}});
    add_flow(tx, rx12, {{"out1", "receivers"}});

    add_flow(tx, rx21, {{"out2", "receivers"}});
    add_flow(tx, rx22, {{"out2", "receivers"}});
  }
};

TEST(NativeOperatorMultiBroadcastsApp, TestNativeOperatorMultiBroadcastsApp) {
  load_env_log_level();

  auto app = make_application<NativeMultiBroadcastsApp>();

  const std::string config_file = test_config.get_test_data_file("minimal.yaml");
  app->config(config_file);

  // capture output so that we can check that the expected value is present
  testing::internal::CaptureStderr();

  app->run();

  std::string log_output = testing::internal::GetCapturedStderr();

  // Check if 'log_output' has 'Rx message received (count: 10, size: 1)' four times in it.
  // (from rx11, rx12, rx21, rx22)
  int count = 0;
  std::string recv_string{"Rx message received (count: 10, size: 1)"};
  auto pos = log_output.find(recv_string);
  while (pos != std::string::npos) {
    count++;
    pos = log_output.find(recv_string, pos + recv_string.size());
  }
  EXPECT_EQ(count, 4);
}

}  // namespace holoscan
