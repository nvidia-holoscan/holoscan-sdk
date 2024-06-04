/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <chrono>
#include <string>
#include <thread>
#include <vector>

#include <holoscan/holoscan.hpp>

namespace holoscan {

namespace {

class DummyOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(DummyOp)

  DummyOp() = default;

  void setup(OperatorSpec& spec) override {}

  void compute(InputContext&, OutputContext& op_output, ExecutionContext&) override {
    HOLOSCAN_LOG_INFO("Operator: {}, Index: {}", name(), index_);
    // Sleep for 100ms to simulate some work
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    index_++;
  }

  int index() const { return index_; }

 private:
  int index_ = 1;
};

class Fragment1 : public holoscan::Fragment {
 public:
  Fragment1() = default;

  void compose() override {
    using namespace holoscan;
    auto tx = make_operator<DummyOp>("tx", make_condition<CountCondition>(10));
    add_operator(tx);
  }
};

class Fragment2 : public holoscan::Fragment {
 public:
  Fragment2() = default;

  void compose() override {
    using namespace holoscan;
    auto rx = make_operator<DummyOp>("rx", make_condition<CountCondition>(5));
    add_operator(rx);
  }
};

class StandaloneFragmentApp : public holoscan::Application {
 public:
  // Inherit the constructor
  using Application::Application;

  void compose() override {
    using namespace holoscan;
    auto fragment1 = make_fragment<Fragment1>("fragment1");
    auto fragment2 = make_fragment<Fragment2>("fragment2");

    add_fragment(fragment1);
    add_fragment(fragment2);
  }
};

}  // namespace

///////////////////////////////////////////////////////////////////////////////
// Tests
///////////////////////////////////////////////////////////////////////////////

TEST(DistributedApp, TestStandaloneFragments) {
  // Test that two fragments can be run independently in a distributed app (issue 4616519).
  const std::vector<std::string> args{"test_app", "--driver", "--worker", "--fragments", "all"};
  auto app = make_application<StandaloneFragmentApp>(args);

  // capture output so that we can check that the expected value is present
  testing::internal::CaptureStderr();

  app->run();

  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("Operator: tx, Index: 10") != std::string::npos);
  EXPECT_TRUE(log_output.find("Operator: rx, Index: 5") != std::string::npos);
}

}  // namespace holoscan
