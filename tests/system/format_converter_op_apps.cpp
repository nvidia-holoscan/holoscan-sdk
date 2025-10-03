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

#include <array>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "../config.hpp"

#include "tensor_compare_op.hpp"

#include "holoscan/operators/format_converter/format_converter.hpp"
#include "holoscan/operators/ping_tensor_rx/ping_tensor_rx.hpp"
#include "holoscan/operators/ping_tensor_tx/ping_tensor_tx.hpp"

using namespace holoscan;

static HoloscanTestConfig test_config;

class FormatConverterApp : public holoscan::Application {
 public:
  FormatConverterApp() = default;

  void compose() override {
    auto count = make_condition<CountCondition>(10);

    auto [data_type, channels] = get_data_type_and_channels(in_dtype_);
    auto source = make_operator<ops::PingTensorTxOp>("ping_source",
                                                     Arg("rows", height_),
                                                     Arg("columns", width_),
                                                     Arg("channels", channels),
                                                     Arg("data_type", data_type),
                                                     Arg("storage_type", storage_type_),
                                                     Arg("data", data_in_),
                                                     count);
    auto pool = Arg("pool", make_resource<holoscan::UnboundedAllocator>("pool"));
    auto in_dtype = Arg("in_dtype", in_dtype_);
    auto out_dtype = Arg("out_dtype", out_dtype_);
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

    if (data_out_.size() > 0) {
      auto comparator = make_operator<ops::TensorCompareOp>("comparator");
      add_flow(converter, comparator, {{"tensor", "input1"}});
      auto [data_type, channels] = get_data_type_and_channels(out_dtype_);
      auto source_expected = make_operator<ops::PingTensorTxOp>("ping_source_expected",
                                                                Arg("rows", height_),
                                                                Arg("columns", width_),
                                                                Arg("channels", channels),
                                                                Arg("data_type", data_type),
                                                                Arg("storage_type", storage_type_),
                                                                Arg("data", data_out_));

      add_flow(source_expected, comparator, {{"out", "input2"}});
    }
  }

  void set_width(const int32_t width) { width_ = width; }
  void set_height(const int32_t height) { height_ = height; }
  void set_storage_type(const std::string& storage_type) { storage_type_ = storage_type; }
  void set_in_dtype(const std::string& in_dtype) { in_dtype_ = in_dtype; }
  void set_out_dtype(const std::string& out_dtype) { out_dtype_ = out_dtype; }
  void set_data_in(const std::vector<uint8_t>& data_in) { data_in_ = data_in; }
  void set_data_out(const std::vector<uint8_t>& data_out) { data_out_ = data_out; }

 private:
  int32_t width_ = 32;
  int32_t height_ = 64;
  std::string storage_type_ = "device";
  std::string in_dtype_ = "rgba8888";
  std::string out_dtype_ = "rgb888";
  std::vector<uint8_t> data_in_ = {};
  std::vector<uint8_t> data_out_ = {};

  std::pair<std::string, int32_t> get_data_type_and_channels(const std::string& in_dtype) {
    if (in_dtype == "rgb161616") {
      return std::make_pair("uint16_t", 3);
    } else if (in_dtype == "rgba16161616") {
      return std::make_pair("uint16_t", 4);
    } else if (in_dtype == "rgb888") {
      return std::make_pair("uint8_t", 3);
    } else if (in_dtype == "rgba8888") {
      return std::make_pair("uint8_t", 4);
    } else {
      EXPECT_TRUE(false) << "Unsupported input data type: " << in_dtype;
      return std::make_pair("invalid", -1);
    }
  }
};

void run_app(const std::shared_ptr<FormatConverterApp>& app, const std::string& failure_str = "") {
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
                                          std::string("system"), std::string("cuda_managed")));

// run this case with various tensor memory storage types
TEST_P(FormatConverterStorageParameterizedTestFixture, TestFormatConverterStorageTypes) {
  std::string storage_type = GetParam();
  auto app = make_application<FormatConverterApp>();
  app->set_storage_type(storage_type);
  run_app(app, "");
}

class FormatConverterFormatParameterizedTestFixture
    : public ::testing::TestWithParam<std::pair<std::string, std::string>> {};

INSTANTIATE_TEST_CASE_P(FormatConverterOpAppTests, FormatConverterFormatParameterizedTestFixture,
                        ::testing::Values(
                            // FormatConversionType::kRGB888ToRGBA8888
                            std::pair<std::string, std::string>("rgb888", "rgba8888"),
                            // FormatConversionType::kRGBA8888ToRGB888
                            std::pair<std::string, std::string>("rgba8888", "rgb888"),
                            // FormatConversionType::kRGBA16161616ToRGB888
                            std::pair<std::string, std::string>("rgba16161616", "rgb888"),
                            // FormatConversionType::kRGB161616ToRGB888
                            std::pair<std::string, std::string>("rgb161616", "rgb888")));

// run this case with various tensor memory storage types
TEST_P(FormatConverterFormatParameterizedTestFixture, TestFormatConverterFormatTypes) {
  std::pair<std::string, std::string> format_type = GetParam();

  std::vector<uint8_t> data_in, data_out;
  if ((format_type.first == "rgb161616") && (format_type.second == "rgb888")) {
    std::array<uint16_t, 3> data = {0x1234, 0x5678, 0x9abc};
    // NPP is using a scale factor which gives different results than just left shifting
    const float scale_factor = 255.f / 65535.f;
    for (auto& value : data) {
      data_in.push_back(value & 0xff);
      data_in.push_back(value >> 8);
      data_out.push_back(static_cast<uint8_t>(value * scale_factor));
    }
  } else if ((format_type.first == "rgba16161616") && (format_type.second == "rgb888")) {
    std::array<uint16_t, 4> data = {0x1234, 0x5678, 0x9abc, 0xdef0};
    // NPP is using a scale factor which gives different results than just left shifting
    const float scale_factor = 255.f / 65535.f;
    for (int i = 0; i < data.size(); i++) {
      auto value = data[i];
      data_in.push_back(value & 0xff);
      data_in.push_back(value >> 8);
      if (i < 3) {
        data_out.push_back(static_cast<uint8_t>(value * scale_factor));
      }
    }
  }
  auto app = make_application<FormatConverterApp>();
  app->set_in_dtype(format_type.first);
  app->set_out_dtype(format_type.second);
  app->set_data_in(data_in);
  app->set_data_out(data_out);
  app->set_width(1);
  app->set_height(1);
  run_app(app, "");
}
