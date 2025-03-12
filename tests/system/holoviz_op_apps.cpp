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
#include <utility>
#include <vector>

#include "../config.hpp"
#include "tensor_compare_op.hpp"

#include "holoscan/holoscan.hpp"
#include "holoscan/operators/format_converter/format_converter.hpp"
#include "holoscan/operators/holoviz/holoviz.hpp"
#include "holoscan/operators/ping_tensor_tx/ping_tensor_tx.hpp"

using namespace holoscan;

static HoloscanTestConfig test_config;

using StringOrArg = std::variant<std::string, Arg>;

class HolovizToHolovizApp : public holoscan::Application {
 public:
  explicit HolovizToHolovizApp(StringOrArg enable_arg, ArgList source_args)
      : Application(), enable_arg_(enable_arg), source_args_(source_args) {
    if (source_args_.size() == 0) {
      source_args_ = ArgList({Arg("rows", height_), Arg("columns", width_), Arg("channels", 3)});
    }
  }

  ArgList get_enable_arg() {
    if (std::holds_alternative<Arg>(enable_arg_)) { return ArgList({std::get<Arg>(enable_arg_)}); }
    if (std::holds_alternative<std::string>(enable_arg_)) {
      std::string arg_name = std::get<std::string>(enable_arg_);
      if (arg_name != "") { return from_config(arg_name); }
    }
    return ArgList();
  }

  void compose() override {
    auto allocator = Arg("allocator", make_resource<holoscan::UnboundedAllocator>("allocator"));
    auto count = make_condition<CountCondition>(10);
    auto headless = Arg("headless", true);
    auto source = make_operator<ops::PingTensorTxOp>(
        "ping_source", source_args_, Arg("storage_type", storage_type_));
    auto renderer = make_operator<ops::HolovizOp>("renderer",
                                                  count,
                                                  headless,
                                                  from_config("holoviz_tensor_input"),
                                                  allocator,
                                                  Arg("width", uint32_t(width_)),
                                                  Arg("height", uint32_t(height_)),
                                                  get_enable_arg());

    auto pool = Arg("pool", make_resource<holoscan::UnboundedAllocator>("pool"));
    auto in_dtype = Arg("in_dtype", std::string("rgba8888"));
    auto out_dtype = Arg("out_dtype", std::string("rgb888"));

    add_flow(source, renderer, {{"out", "receivers"}});

    // TensorCompareOp only works for device tensors
    if (storage_type_ == "device") {
      auto comparator = make_operator<ops::TensorCompareOp>("comparator");
      auto video_to_tensor =
          make_operator<ops::FormatConverterOp>("converter", in_dtype, out_dtype, pool);

      add_flow(renderer, video_to_tensor, {{"render_buffer_output", "source_video"}});
      add_flow(source, comparator, {{"out", "input1"}});
      add_flow(video_to_tensor, comparator, {{"tensor", "input2"}});
    }
  }

  void set_storage_type(const std::string& storage_type) { storage_type_ = storage_type; }

 private:
  const int32_t width_ = 471, height_ = 177;
  StringOrArg enable_arg_;
  ArgList source_args_;
  std::string storage_type_ = std::string("device");

  HolovizToHolovizApp() = delete;
};

void run_app(StringOrArg enable_arg, const std::string& failure_str = "",
             const std::string& storage_type = "device", ArgList source_args = {}) {
  auto app = make_application<HolovizToHolovizApp>(enable_arg, source_args);
  app->set_storage_type(storage_type);

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
    EXPECT_TRUE(log_output.find("error") == std::string::npos) << "=== LOG ===\n"
                                                               << log_output << "\n===========\n";
  } else {
    EXPECT_TRUE(log_output.find(failure_str) != std::string::npos)
        << "=== LOG ===\n"
        << log_output << "\n===========\n";
  }
}

class HolovizStorageParameterizedTestFixture : public ::testing::TestWithParam<std::string> {};

INSTANTIATE_TEST_CASE_P(HolovizOpAppTests, HolovizStorageParameterizedTestFixture,
                        ::testing::Values(std::string("device"), std::string("host"),
                                          std::string("system")));

// run this case with various tensor memory storage types
TEST_P(HolovizStorageParameterizedTestFixture, TestHolovizStorageTypes) {
  std::string storage_type = GetParam();
  // run without any extra arguments, just to verify HolovizOp support various memory types
  run_app("", "", storage_type);
}

// run this case with various tensor memory storage types
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

// Test the layer callback. The other callbacks are tested in Python, but the layer callback
// is available in C++ only.
TEST(HolovizApps, TestLayerCallback) {
  std::vector<std::size_t> input_sizes;
  run_app(Arg("layer_callback",
              ops::HolovizOp::LayerCallbackFunction(
                  [&input_sizes](const std::vector<holoscan::gxf::Entity>& inputs) -> void {
                    input_sizes.push_back(inputs.size());
                  })));
  EXPECT_EQ(input_sizes.size(), 1);
  EXPECT_EQ(input_sizes[0], 1);
}

enum class InputType {
  FAIL_TENSOR_INPUT_TYPE_DETECT,
  FAIL_VIDEO_BUFFER_INPUT_TYPE_DETECT,
  UNSUPPORTED_TENSOR_FORMAT,
  UNSUPPORTED_VIDEO_BUFFER_FORMAT,
  DETECT_CROSSES,
  DETECT_COLOR_LUT
};

class SourceFormatOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(SourceFormatOp)

  explicit SourceFormatOp(InputType input_type) : input_type_(input_type) {}
  SourceFormatOp() = delete;

  template <nvidia::gxf::VideoFormat VIDEO_FORMAT>
  void add_video_buffer(nvidia::gxf::Expected<nvidia::gxf::Entity>& entity, uint32_t width = 16,
                        uint32_t height = 8) {
    nvidia::gxf::VideoFormatSize<VIDEO_FORMAT> video_format_size;
    data_.resize(video_format_size.size(width, height, false));

    nvidia::gxf::VideoBufferInfo video_buffer_info;
    video_buffer_info.width = width;
    video_buffer_info.height = height;
    video_buffer_info.color_format = VIDEO_FORMAT;
    video_buffer_info.surface_layout = nvidia::gxf::SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR;
    video_buffer_info.color_planes = video_format_size.getDefaultColorPlanes(
        video_buffer_info.width, video_buffer_info.height, false);

    auto video_buffer = entity.value().add<nvidia::gxf::VideoBuffer>("video_buffer");
    video_buffer.value()->wrapMemory(video_buffer_info,
                                     data_.size(),
                                     nvidia::gxf::MemoryStorageType::kSystem,
                                     data_.data(),
                                     [](void*) mutable { return nvidia::gxf::Success; });
  }

  void setup(OperatorSpec& spec) override { spec.output<nvidia::gxf::Entity>("out"); }
  void compute([[maybe_unused]] InputContext& op_input, OutputContext& op_output,
               [[maybe_unused]] ExecutionContext& context) override {
    auto entity = nvidia::gxf::Entity::New(context.context());

    switch (input_type_) {
      case InputType::FAIL_TENSOR_INPUT_TYPE_DETECT:
        // two component tensor does not map to any supported input type
        add_tensor(entity, 2, nvidia::gxf::PrimitiveType::kUnsigned8);
        break;
      case InputType::UNSUPPORTED_TENSOR_FORMAT:
        // `kFloat16` is not supported
        add_tensor(entity, 3, nvidia::gxf::PrimitiveType::kFloat16);
        break;
      case InputType::FAIL_VIDEO_BUFFER_INPUT_TYPE_DETECT:
      case InputType::UNSUPPORTED_VIDEO_BUFFER_FORMAT:
        // rgbd8 is not supported
        add_video_buffer<nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_RGBD8>(entity);
        break;
      case InputType::DETECT_CROSSES:
        add_tensor(entity, 2, nvidia::gxf::PrimitiveType::kFloat32, 1);
        break;
      case InputType::DETECT_COLOR_LUT:
        add_tensor(entity, 1, nvidia::gxf::PrimitiveType::kUnsigned8);
        break;
      default:
        EXPECT_TRUE(false) << "Unsupported input type";
        break;
    }

    op_output.emit(entity.value());
  }

  void add_tensor(nvidia::gxf::Expected<nvidia::gxf::Entity>& entity, uint32_t components,
                  nvidia::gxf::PrimitiveType prmitive_type, uint32_t width = 16,
                  uint32_t height = 8) {
    const uint32_t primitive_type_size = PrimitiveTypeSize(prmitive_type);
    data_.resize(width * height * components * primitive_type_size);

    auto tensor = entity.value().add<nvidia::gxf::Tensor>("tensor");
    tensor.value()->wrapMemory({int32_t(height), int32_t(width), int32_t(components)},
                               prmitive_type,
                               primitive_type_size,
                               nvidia::gxf::Unexpected{GXF_UNINITIALIZED_VALUE},
                               nvidia::gxf::MemoryStorageType::kSystem,
                               data_.data(),
                               [](void*) mutable { return nvidia::gxf::Success; });
  }

  InputType input_type_;
  std::vector<uint8_t> data_;
};

class FormatApp : public holoscan::Application {
 public:
  explicit FormatApp(InputType input_type) : Application(), input_type_(input_type) {}

  void compose() override {
    auto source = make_operator<SourceFormatOp>(input_type_);

    ArgList args;
    switch (input_type_) {
      case InputType::UNSUPPORTED_TENSOR_FORMAT: {
        std::vector<ops::HolovizOp::InputSpec> input_specs;
        input_specs.emplace_back("tensor", ops::HolovizOp::InputType::COLOR);
        args = ArgList({Arg("tensors", input_specs)});
      } break;
      case InputType::UNSUPPORTED_VIDEO_BUFFER_FORMAT: {
        std::vector<ops::HolovizOp::InputSpec> input_specs;
        input_specs.emplace_back("video_buffer", ops::HolovizOp::InputType::COLOR);
        args = ArgList({Arg("tensors", input_specs)});
      } break;
      case InputType::DETECT_COLOR_LUT: {
        std::vector<std::vector<float>> color_lut;
        color_lut.emplace_back(std::vector<float>({1.f, 1.f, 1.f, 1.f}));
        args = ArgList({Arg("color_lut", color_lut)});
      } break;
      default:
        // do nothing for FAIL_VIDEO_BUFFER_INPUT_TYPE_DETECT, DETECT_CROSSES
        break;
    }

    auto renderer = make_operator<ops::HolovizOp>(
        "renderer", make_condition<CountCondition>(1), Arg("headless", true), args);

    add_flow(source, renderer, {{"out", "receivers"}});
  }

  InputType input_type_;
};

class HolovizInputTestFixture : public ::testing::TestWithParam<InputType> {};

INSTANTIATE_TEST_CASE_P(HolovizOpAppTests, HolovizInputTestFixture,
                        ::testing::Values(InputType::FAIL_TENSOR_INPUT_TYPE_DETECT,
                                          InputType::FAIL_VIDEO_BUFFER_INPUT_TYPE_DETECT,
                                          InputType::UNSUPPORTED_TENSOR_FORMAT,
                                          InputType::UNSUPPORTED_VIDEO_BUFFER_FORMAT,
                                          InputType::DETECT_CROSSES, InputType::DETECT_COLOR_LUT));

// run this case with various tensor memory storage types
TEST_P(HolovizInputTestFixture, TestHolovizInputs) {
  auto input_type = GetParam();
  auto app = make_application<FormatApp>(input_type);

  // capture output to check that the expected messages were logged
  testing::internal::CaptureStderr();
  std::string exception;
  try {
    app->run();
  } catch (const std::exception& ex) { exception = ex.what(); }
  std::string log_output = testing::internal::GetCapturedStderr();
  std::string expected_log;
  std::string expected_exception;
  switch (input_type) {
    case InputType::FAIL_TENSOR_INPUT_TYPE_DETECT:
      expected_log = "Ignoring Tensor 'tensor'";
      break;
    case InputType::FAIL_VIDEO_BUFFER_INPUT_TYPE_DETECT:
      expected_log = "Ignoring VideoBuffer 'video_buffer'";
      break;
    case InputType::UNSUPPORTED_TENSOR_FORMAT:
      expected_log = "Format is not supported:";
      expected_exception = "Unsupported input `tensor`";
      break;
    case InputType::UNSUPPORTED_VIDEO_BUFFER_FORMAT:
      expected_log = "Format is not supported:";
      expected_exception = "Unsupported input `video_buffer`";
      break;
    case InputType::DETECT_CROSSES:
      expected_log = "- type: crosses";
      break;
    case InputType::DETECT_COLOR_LUT:
      expected_log = "- type: color_lut";
      break;
    default:
      EXPECT_TRUE(false) << "Unsupported input type";
      break;
  }
  EXPECT_TRUE(log_output.find(expected_log) != std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
  if (!expected_exception.empty()) {
    EXPECT_TRUE(exception.find(expected_exception) != std::string::npos)
        << "=== Exception ===\n"
        << exception << "\n===========\n";
  }
}
