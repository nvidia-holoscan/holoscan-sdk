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

#include <memory>
#include <string>
#include <utility>

#include "../config.hpp"
#include "ping_tensor_tx_op.hpp"
#include "tensor_compare_op.hpp"

#include "holoscan/holoscan.hpp"
#include "holoscan/operators/format_converter/format_converter.hpp"
#include "holoscan/operators/holoviz/holoviz.hpp"

using namespace holoscan;

static HoloscanTestConfig test_config;

using StringOrArg = std::variant<std::string, Arg>;

class HolovizToHolovizApp : public holoscan::Application {
 public:
  explicit HolovizToHolovizApp(StringOrArg enable_arg) : Application(), enable_arg_(enable_arg) {}

  ArgList get_enable_arg() {
    if (std::holds_alternative<Arg>(enable_arg_)) { return ArgList({std::get<Arg>(enable_arg_)}); }
    if (std::holds_alternative<std::string>(enable_arg_)) {
      return from_config(std::get<std::string>(enable_arg_));
    }
    return ArgList();
  }

  void compose() override {
    const int32_t width = 471, height = 177;
    auto allocator = Arg("allocator", make_resource<holoscan::UnboundedAllocator>("allocator"));
    auto count = make_condition<CountCondition>(10);
    auto headless = Arg("headless", true);
    auto shape = ArgList({Arg("rows", height), Arg("columns", width), Arg("channels", 3)});
    auto source = make_operator<ops::PingTensorTxOp>("ping_source", shape);
    auto renderer = make_operator<ops::HolovizOp>("renderer",
                                                  count,
                                                  headless,
                                                  from_config("holoviz_tensor_input"),
                                                  allocator,
                                                  Arg("width", uint32_t(width)),
                                                  Arg("height", uint32_t(height)),
                                                  get_enable_arg());

    auto pool = Arg("pool", make_resource<holoscan::UnboundedAllocator>("pool"));
    auto in_dtype = Arg("in_dtype", std::string("rgba8888"));
    auto out_dtype = Arg("out_dtype", std::string("rgb888"));
    auto video_to_tensor =
        make_operator<ops::FormatConverterOp>("converter", in_dtype, out_dtype, pool);

    auto comparator = make_operator<ops::TensorCompareOp>("comparator");

    add_flow(source, renderer, {{"out", "receivers"}});
    add_flow(source, comparator, {{"out", "input1"}});
    add_flow(renderer, video_to_tensor, {{"render_buffer_output", "source_video"}});
    add_flow(video_to_tensor, comparator, {{"tensor", "input2"}});
  }

 private:
  StringOrArg enable_arg_;

  HolovizToHolovizApp() = delete;
};

void run_app(StringOrArg enable_arg, std::string failure_str = "") {
  auto app = make_application<HolovizToHolovizApp>(enable_arg);

  const std::string config_file = test_config.get_test_data_file("app_config.yaml");
  app->config(config_file);

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
    EXPECT_TRUE(log_output.find("error") == std::string::npos) << log_output;
  } else {
    EXPECT_TRUE(log_output.find(failure_str) != std::string::npos) << log_output;
  }
}

TEST(HolovizApps, TestEnableRenderBufferOutputYAML) {
  run_app("holoviz_enable_ports");
}

TEST(HolovizApps, TestDisableRenderBufferOutputYAML) {
  run_app("holoviz_disable_ports");
}

TEST(HolovizApps, TestInvalidRenderBufferOutputYAML) {
  run_app("holoviz_invalid_ports", "Could not parse YAML parameter");
}

TEST(HolovizApps, TestEnableRenderBufferOutputArg) {
  run_app(Arg("enable_render_buffer_output", true));
}

TEST(HolovizApps, TestDisableRenderBufferOutputArg) {
  run_app(Arg("enable_render_buffer_output", false));
}

TEST(HolovizApps, TestInvalidRenderBufferOutputArg) {
  run_app(Arg("enable_render_buffer_output", 2), "Could not cast parameter");
}
