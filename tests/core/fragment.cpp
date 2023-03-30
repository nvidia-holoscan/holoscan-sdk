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

#include "holoscan/core/fragment.hpp"

#include <gtest/gtest.h>
#include <gxf/core/gxf.h>

#include <string>

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

// TODO: how to properly specify path to the config file here
//       ? maybe define a path in the CMake config that we can reference here?

TEST(Fragment, TestFragmentConfig) {
  Fragment F;

  const std::string config_file = test_config.get_test_data_file("app_config.yaml");
  F.config(config_file);

  Config C = F.config();
  ASSERT_TRUE(C.config_file() == config_file);

  ArgList args = F.from_config("format_converter_replayer"s);
  EXPECT_EQ(args.size(), 4);

  // verify that second call to config raises a warning
  testing::internal::CaptureStderr();
  F.config(config_file);
  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("warning") != std::string::npos);
  EXPECT_TRUE(log_output.find("Config object was already created. Overwriting...") !=
              std::string::npos);
}

TEST(Fragment, TestFragmentConfigNestedArgs) {
  Fragment F;

  const std::string config_file = test_config.get_test_data_file("app_config.yaml");
  F.config(config_file);

  Config C = F.config();
  ASSERT_TRUE(C.config_file() == config_file);

  // can directly access a specific argument under the "aja" section
  ArgList arglist = F.from_config("aja.width"s);
  EXPECT_EQ(arglist.size(), 1);
  Arg width = arglist.args()[0];
  EXPECT_EQ(width.arg_type().element_type(), ArgElementType::kYAMLNode);
  EXPECT_EQ(width.arg_type().container_type(), ArgContainerType::kNative);
}

TEST(Fragment, TestConfigUninitializedWarning) {
  Fragment F;

  Config C = F.config();
  EXPECT_EQ(C.config_file(), "");
}

TEST(Fragment, TestFragmentFromConfigNonexistentKey) {
  Fragment F;

  testing::internal::CaptureStderr();

  const std::string config_file = test_config.get_test_data_file("app_config.yaml");
  F.config(config_file);

  Config C = F.config();
  ASSERT_TRUE(C.config_file() == config_file);
  ArgList args = F.from_config("non-existent"s);
  EXPECT_EQ(args.size(), 0);

  // verify that an error is logged when the key is not in the YAML file
  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("error") != std::string::npos);
  EXPECT_TRUE(log_output.find("Unable to find the parameter item/map with key 'non-existent'") !=
              std::string::npos);
}

TEST(Fragment, TestFragmentConfigNonexistentFile) {
  Fragment F;

  const std::string config_file = "nonexistent.yaml";

  // capture stderr
  testing::internal::CaptureStderr();
  F.config(config_file);

  // verify that an error is logged when the YAML file doesn't exist
  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("warning") != std::string::npos);
  EXPECT_TRUE(log_output.find("Config file 'nonexistent.yaml' doesn't exist") != std::string::npos);
}

TEST(Fragment, TestFragmentGraph) {
  Fragment F;

  // First call to graph creates a FlowGraph object
  // F.graph() returns a pointer to the abstract Graph base class so use
  // static_cast here
  FlowGraph G = static_cast<FlowGraph&>(F.graph());
}

TEST(Fragment, TestFragmentExecutor) {
  Fragment F;

  // First call to executor generates an Executor object
  Executor E = F.executor();

  // this fragment is associated with the executor that was created
  EXPECT_EQ(E.fragment(), &F);
}

TEST(Fragment, TestFragmentMoveAssignment) {
  Fragment G;
  G.name("G");
  Fragment F;
  F.name("F");

  // can only move assign (copy assignment operator has been deleted)
  F = std::move(G);
  EXPECT_EQ(F.name(), "G");
  EXPECT_EQ(G.name(), "");
}

// Fragment::make_condition is tested elsewhere in condition_classes.cpp
// Fragment::make_resource is tested elsewhere in resource_classes.cpp
// Fragment::make_operator is tested elsewhere in operator_classes.cpp

}  // namespace holoscan
