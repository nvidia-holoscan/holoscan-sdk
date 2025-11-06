/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "holoscan/core/executors/gxf/gxf_executor.hpp"
#include "holoscan/core/graphs/flow_graph.hpp"

namespace holoscan {

TEST(Application, TestAppDescription) {
  auto app = make_application<Application>();
  EXPECT_EQ(app->description(), "");
  app->description("test description");
  EXPECT_EQ(app->description(), "test description");
}

TEST(Application, TestAppVersion) {
  auto app = make_application<Application>();
  EXPECT_EQ(app->version(), "0.0.0");
  app->version("1.2.3");
  EXPECT_EQ(app->version(), "1.2.3");
}

TEST(Application, TestAppArgv) {
  auto app = make_application<Application>();
  EXPECT_GT(app->argv().size(), 0);
}

TEST(Application, TestAppEmptyOptions) {
  auto app = make_application<Application>();
  auto& options = app->options();
  EXPECT_FALSE(options.run_driver);
  EXPECT_FALSE(options.run_worker);
  EXPECT_EQ(options.driver_address, "");
  EXPECT_EQ(options.worker_address, "");
  EXPECT_EQ(options.worker_targets.size(), 0);
  EXPECT_EQ(options.config_path, "");
}

TEST(Application, TestAppCustomArguments) {
  std::vector<std::string> args{"my_app",
                                "--driver",
                                "--worker",
                                "--address",
                                "10.0.0.1:9999",
                                "--worker-address=:9999",
                                "--dummy_option",
                                "dummy_value",
                                "--fragments",
                                "fragment1,fragment2,fragment3",
                                "--config",
                                "app_config.yaml",
                                "dummy_positional_arg"};
  auto app = make_application<Application>(args);
  auto& argv = app->argv();
  EXPECT_EQ(argv.size(), 4);
  EXPECT_EQ(argv[0], "my_app");
  EXPECT_EQ(argv[1], "--dummy_option");
  EXPECT_EQ(argv[2], "dummy_value");
  EXPECT_EQ(argv[3], "dummy_positional_arg");

  auto& options = app->options();
  EXPECT_TRUE(options.run_driver);
  EXPECT_TRUE(options.run_worker);
  EXPECT_EQ(options.driver_address, "10.0.0.1:9999");
  EXPECT_EQ(options.worker_address, ":9999");
  EXPECT_EQ(options.worker_targets.size(), 3);
  EXPECT_EQ(options.worker_targets[0], "fragment1");
  EXPECT_EQ(options.worker_targets[1], "fragment2");
  EXPECT_EQ(options.worker_targets[2], "fragment3");
  EXPECT_EQ(options.config_path, "app_config.yaml");
}

TEST(Application, TestAppPrintOptions) {
  std::vector<std::string> args{"my_app",
                                "--driver",
                                "--dummy_option",
                                "dummy_value",
                                "--worker",
                                "--address=10.0.0.1:9999",
                                "--worker-address=0.0.0.0:8888",
                                "--fragments=fragment1,fragment2,fragment3",
                                "--config=app_config.yaml",
                                "dummy_positional_arg"};
  auto app = make_application<Application>(args);
  auto& options = app->options();

  // capture output so that we can check that the expected value is present
  testing::internal::CaptureStderr();

  options.print();

  std::string log_output = testing::internal::GetCapturedStderr();

  EXPECT_TRUE(log_output.find("run_driver: true") != std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
  EXPECT_TRUE(log_output.find("run_worker: true") != std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
  EXPECT_TRUE(log_output.find("driver_address: 10.0.0.1:9999") != std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
  EXPECT_TRUE(log_output.find("worker_address: 0.0.0.0:8888") != std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
  EXPECT_TRUE(log_output.find("worker_targets: fragment1, fragment2, fragment3") !=
              std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
  EXPECT_TRUE(log_output.find("config_path: app_config.yaml") != std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
}

TEST(Application, TestAppHelpOption) {
  // capture output so that we can check that the expected value is present
  testing::internal::CaptureStdout();

  std::vector<std::string> args{"my_app", "--help"};
  EXPECT_EXIT(make_application<Application>(args), ::testing::ExitedWithCode(0), ".*");

  std::string log_output = testing::internal::GetCapturedStdout();
  EXPECT_TRUE(log_output.find("Usage: ") != std::string::npos) << "=== LOG ===\n"
                                                               << log_output << "\n===========\n";
}

TEST(Application, TestAppVersionOption) {
  // capture output so that we can check that the expected value is present
  testing::internal::CaptureStdout();

  std::vector<std::string> args{"my_app", "--version"};
  EXPECT_EXIT(make_application<Application>(args), ::testing::ExitedWithCode(0), ".*");

  std::string log_output = testing::internal::GetCapturedStdout();
  EXPECT_TRUE(log_output.find("0.0.0") != std::string::npos) << "=== LOG ===\n"
                                                             << log_output << "\n===========\n";
}

TEST(Application, TestAddFragment) {
  auto app = make_application<Application>();
  auto fragment = app->make_fragment<Fragment>("fragment");

  app->add_fragment(fragment);

  // First call to graph creates a FlowGraph object
  // F.graph() returns a reference to the abstract Graph base class so use
  // static_cast here
  FragmentFlowGraph& G = static_cast<FragmentFlowGraph&>(app->fragment_graph());

  // verify that the operator was added to the graph
  auto nodes = G.get_nodes();
  EXPECT_EQ(nodes.size(), 1);
  EXPECT_EQ(nodes[0], fragment);
}

TEST(Application, TestAddFlow) {
  auto app = make_application<Application>();

  auto fragment1 = app->make_fragment<Fragment>("fragment1");
  auto fragment2 = app->make_fragment<Fragment>("fragment2");

  app->add_flow(fragment1, fragment2, {{"blur_image", "sharpen_image"}});

  // First call to graph creates a FlowGraph object
  // F.graph() returns a reference to the abstract Graph base class so use
  // static_cast here
  FragmentFlowGraph& G = static_cast<FragmentFlowGraph&>(app->fragment_graph());

  // verify that the fragments and edges were added to the graph
  auto nodes = G.get_nodes();
  EXPECT_EQ(nodes.size(), 2);
  auto find_fragment1 = G.find_node(
      [&fragment1](const holoscan::FragmentNodeType& node) { return node == fragment1; });
  auto find_fragment2 = G.find_node(
      [&fragment2](const holoscan::FragmentNodeType& node) { return node == fragment2; });
  EXPECT_TRUE(find_fragment1);
  EXPECT_TRUE(find_fragment2);
  auto port_map = G.get_port_map(fragment1, fragment2);
  EXPECT_TRUE(port_map);
  EXPECT_EQ(std::begin(*port_map.value())->first, "blur_image");
  auto& input_port_set = std::begin(*port_map.value())->second;
  EXPECT_EQ(input_port_set.size(), 1);
  EXPECT_EQ(*(input_port_set.begin()), "sharpen_image");
}

TEST(Application, TestAddFragmentsWithSameName) {
  auto app = make_application<Application>();

  auto fragment1 = app->make_fragment<Fragment>("fragment");
  auto fragment2 = app->make_fragment<Fragment>("fragment");

  app->add_fragment(fragment1);

  EXPECT_THROW(app->add_fragment(fragment2), holoscan::RuntimeError);
  EXPECT_THROW(app->add_flow(fragment1, fragment2, {{"blur_image", "sharpen_image"}}),
               holoscan::RuntimeError);
}

TEST(Application, TestReservedFragmentName) {
  auto app = make_application<Application>();

  testing::internal::CaptureStderr();

  auto fragment = app->make_fragment<Fragment>("all");

  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("Fragment name 'all' is reserved") != std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
}

TEST(Application, TestAddFlowWithDifferentExecutors) {
  auto app = make_application<Application>();

  auto fragment1 = app->make_fragment<Fragment>("fragment1");
  auto fragment2 = app->make_fragment<Fragment>("fragment2");

  // Assign different executors to fragments
  fragment1->executor(std::make_shared<gxf::GXFExecutor>(fragment1.get()));
  fragment2->executor(std::make_shared<Executor>(fragment2.get()));

  // the port pair parameter does not matter, as it should throw error even
  // before checking whether the port pairs are valid.
  EXPECT_THROW(app->add_flow(fragment1, fragment2, {{"op1", "op2"}}), std::runtime_error);
}

}  // namespace holoscan
