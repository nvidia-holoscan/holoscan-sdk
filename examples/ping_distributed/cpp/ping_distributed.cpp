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

#include <holoscan/holoscan.hpp>
#include <holoscan/operators/ping_rx/ping_rx.hpp>
#include <holoscan/operators/ping_tx/ping_tx.hpp>

class Fragment1 : public holoscan::Fragment {
 public:
  void compose() override {
    using namespace holoscan;
    auto tx = make_operator<ops::PingTxOp>("tx", make_condition<CountCondition>(10));

    add_operator(tx);
  }
};

class Fragment2 : public holoscan::Fragment {
 public:
  void compose() override {
    using namespace holoscan;
    auto rx = make_operator<ops::PingRxOp>("rx");

    add_operator(rx);
  }
};

class App : public holoscan::Application {
 public:
  // Inherit the constructor
  using Application::Application;

  void compose() override {
    using namespace holoscan;
    auto fragment1 = make_fragment<Fragment1>("fragment1");
    auto fragment2 = make_fragment<Fragment2>("fragment2");

    // Connect the two fragments (tx.out -> rx.in)
    // We can skip the "out" and "in" suffixes, as they are the default
    add_flow(fragment1, fragment2, {{"tx", "rx"}});

    // resource(from_config("resources.fragments"));
  }
};

int main(int argc, char** argv) {
  auto app = holoscan::make_application<App>();
  app->run();

  return 0;
}
