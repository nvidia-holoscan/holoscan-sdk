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

#include <string>

#include <holoscan/holoscan.hpp>
#include <holoscan/operators/gxf_codelet/gxf_codelet.hpp>

#include "../config.hpp"

static HoloscanTestConfig test_config;

namespace holoscan {

// Do not pollute holoscan namespace with utility classes
namespace {

///////////////////////////////////////////////////////////////////////////////
// Utility Resources/Operators
///////////////////////////////////////////////////////////////////////////////

HOLOSCAN_WRAP_GXF_COMPONENT_AS_RESOURCE(MyCudaStreamPool, "nvidia::gxf::CudaStreamPool")
HOLOSCAN_WRAP_GXF_CODELET_AS_OPERATOR(HelloWorldOp, "nvidia::gxf::HelloWorld")

///////////////////////////////////////////////////////////////////////////////
// Utility Applications
///////////////////////////////////////////////////////////////////////////////

class LoadInsideComposeApp : public holoscan::Application {
 public:
  using Application::Application;

  void compose() override {
    auto extension_manager = executor().extension_manager();
    extension_manager->load_extension("libgxf_cuda.so");
    extension_manager->load_extension("libgxf_sample.so");

    auto pool = make_resource<MyCudaStreamPool>("pool");
    auto hello = make_operator<HelloWorldOp>("hello", make_condition<CountCondition>(10));
    add_operator(hello);
  }
};

class DummyApp : public holoscan::Application {
 public:
  using Application::Application;

  void compose() override {
    auto pool = make_resource<MyCudaStreamPool>("pool");
    auto hello = make_operator<HelloWorldOp>("hello", make_condition<CountCondition>(10));
    add_operator(hello);
  }
};

}  // namespace

///////////////////////////////////////////////////////////////////////////////
// Tests
///////////////////////////////////////////////////////////////////////////////

TEST(Extensions, LoadInsideComposeMethod) {
  auto app = make_application<LoadInsideComposeApp>();

  // Capture stderr output to check for specific error messages
  testing::internal::CaptureStderr();

  app->run();

  std::string log_output = testing::internal::GetCapturedStderr();
  // Check that log_output has 10 instances of "Hello world"
  auto pos = log_output.find("Hello world");
  int count = 0;
  while (pos != std::string::npos) {
    count++;
    pos = log_output.find("Hello world", pos + 1);
  }
  EXPECT_EQ(count, 10) << "Expected to find 10 instances of 'Hello world' in log output, but found "
                       << count << "\n=== LOG ===\n"
                       << log_output << "\n===========\n";
}

TEST(Extensions, LoadOutsideApp) {
  auto app = make_application<DummyApp>();

  // Load the extensions outside of the application before calling run() method
  auto& executor = app->executor();
  auto extension_manager = executor.extension_manager();
  extension_manager->load_extension("libgxf_cuda.so");
  extension_manager->load_extension("libgxf_sample.so");

  // Capture stderr output to check for specific error messages
  testing::internal::CaptureStderr();

  app->run();

  std::string log_output = testing::internal::GetCapturedStderr();
  // Check that log_output has 10 instances of "Hello world"
  auto pos = log_output.find("Hello world");
  int count = 0;
  while (pos != std::string::npos) {
    count++;
    pos = log_output.find("Hello world", pos + 1);
  }
  EXPECT_EQ(count, 10) << "Expected to find 10 instances of 'Hello world' in log output, but found "
                       << count << "\n=== LOG ===\n"
                       << log_output << "\n===========\n";
}

TEST(Extensions, LoadFromConfigFile) {
  auto app = make_application<DummyApp>();

  const std::string config_file = test_config.get_test_data_file("loading_gxf_extension.yaml");
  app->config(config_file);

  // Capture stderr output to check for specific error messages
  testing::internal::CaptureStderr();

  app->run();

  std::string log_output = testing::internal::GetCapturedStderr();
  // Check that log_output has 10 instances of "Hello world"
  auto pos = log_output.find("Hello world");
  int count = 0;
  while (pos != std::string::npos) {
    count++;
    pos = log_output.find("Hello world", pos + 1);
  }
  EXPECT_EQ(count, 10) << "Expected to find 10 instances of 'Hello world' in log output, but found "
                       << count << "\n=== LOG ===\n"
                       << log_output << "\n===========\n";
}

TEST(Extensions, LoadFromConfigFileAfterAccessingExecutor) {
  auto app = make_application<DummyApp>();
  // Access executor before calling config() or run() method to see if it works
  app->executor();

  const std::string config_file = test_config.get_test_data_file("loading_gxf_extension.yaml");
  app->config(config_file);

  // Capture stderr output to check for specific error messages
  testing::internal::CaptureStderr();

  app->run();

  std::string log_output = testing::internal::GetCapturedStderr();
  // Check that log_output has 10 instances of "Hello world"
  auto pos = log_output.find("Hello world");
  int count = 0;
  while (pos != std::string::npos) {
    count++;
    pos = log_output.find("Hello world", pos + 1);
  }
  EXPECT_EQ(count, 10) << "Expected to find 10 instances of 'Hello world' in log output, but found "
                       << count << "\n=== LOG ===\n"
                       << log_output << "\n===========\n";
}

}  // namespace holoscan
