/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

class NativeOpApp : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;
    auto tx = make_operator<ops::PingTxOp>("tx", make_condition<CountCondition>(10));
    auto rx = make_operator<ops::PingRxOp>("rx");

    add_flow(tx, rx, {{"out1", "receivers"}, {"out2", "receivers"}});
  }
};

TEST(NativeOperatorPingApp, TestNativeOperatorPingApp) {
  load_env_log_level();

  auto app = make_application<NativeOpApp>();

  const std::string config_file = test_config.get_test_data_file("minimal.yaml");
  app->config(config_file);

  // capture output so that we can check that the expected value is present
  testing::internal::CaptureStderr();

  app->run();

  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("value1: 1") != std::string::npos);
  EXPECT_TRUE(log_output.find("value2: 100") != std::string::npos);
}

}  // namespace holoscan
