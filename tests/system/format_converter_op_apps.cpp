/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <utility>

#include "../config.hpp"

#include "holoscan/holoscan.hpp"
#include "holoscan/operators/format_converter/format_converter.hpp"
#include "holoscan/operators/ping_tensor_rx/ping_tensor_rx.hpp"
#include "holoscan/operators/ping_tensor_tx/ping_tensor_tx.hpp"

using namespace holoscan;

static HoloscanTestConfig test_config;

class FormatConverterApp : public holoscan::Application {
 public:
  explicit FormatConverterApp(const std::string& storage_type)
      : Application(), storage_type_(storage_type) {}

  void compose() override {
    const int32_t width = 32, height = 64;
    auto count = make_condition<CountCondition>(10);
    auto shape_args = ArgList({Arg("rows", height), Arg("columns", width), Arg("channels", 4)});
    auto source = make_operator<ops::PingTensorTxOp>(
        "ping_source", shape_args, Arg("storage_type", storage_type_), count);
    auto pool = Arg("pool", make_resource<holoscan::UnboundedAllocator>("pool"));
    auto in_dtype = Arg("in_dtype", std::string("rgba8888"));
    auto out_dtype = Arg("out_dtype", std::string("rgb888"));
    std::string in_tensor_name = "tensor";
    std::string out_tensor_name = "rgb";
    auto converter = make_operator<ops::FormatConverterOp>("converter",
                                                           in_dtype,
                                                           out_dtype,
                                                           pool,
                                                           Arg("in_tensor_name", in_tensor_name),
                                                           Arg("out_tensor_name", out_tensor_name));
    auto rx = make_operator<ops::PingTensorRxOp>("rx");

    add_flow(source, converter, {{"out", "source_video"}});
    add_flow(converter, rx, {{"tensor", "in"}});
  }

  void set_storage_type(const std::string& storage_type) { storage_type_ = storage_type; }

 private:
  std::string storage_type_ = std::string("device");

  FormatConverterApp() = delete;
};

void run_app(const std::string& failure_str = "", const std::string& storage_type = "device") {
  auto app = make_application<FormatConverterApp>(storage_type);

  // capture output to check that the expected messages were logged
  testing::internal::CaptureStderr();
  try {
    app->run();
  } catch (const std::exception& ex) {
    GTEST_FATAL_FAILURE_(
        fmt::format("{}{}", testing::internal::GetCapturedStderr(), ex.what()).c_str());
  }
  std::string log_output = testing::internal::GetCapturedStderr();
  if (failure_str.empty()) {
    EXPECT_TRUE(log_output.find("error") == std::string::npos) << "=== LOG ===\n"
                                                               << log_output << "\n===========\n";
  } else {
    EXPECT_TRUE(log_output.find(failure_str) != std::string::npos)
        << "=== LOG ===\n"
        << log_output << "\n===========\n";
  }
}

class FormatConverterStorageParameterizedTestFixture
    : public ::testing::TestWithParam<std::string> {};

INSTANTIATE_TEST_CASE_P(FormatConverterOpAppTests, FormatConverterStorageParameterizedTestFixture,
                        ::testing::Values(std::string("device"), std::string("host"),
                                          std::string("system")));

// run this case with various tensor memory storage types
TEST_P(FormatConverterStorageParameterizedTestFixture, TestFormatConverterStorageTypes) {
  std::string storage_type = GetParam();
  run_app("", storage_type);
}
