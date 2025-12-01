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

#include "holoscan/operators/format_converter/format_converter.hpp"
#include "holoscan/operators/holoviz/holoviz.hpp"
#include "holoscan/operators/ping_tensor_tx/ping_tensor_tx.hpp"

using namespace holoscan;

static HoloscanTestConfig test_config;

using StringOrArg = std::variant<std::string, Arg>;

class HolovizToHolovizApp : public holoscan::Application {
 public:
  explicit HolovizToHolovizApp(StringOrArg& enable_arg, ArgList source_args)
      : Application(), enable_arg_(enable_arg), source_args_(source_args) {
    if (source_args_.size() == 0) {
      source_args_ = ArgList({Arg("rows", height_), Arg("columns", width_), Arg("channels", 3)});
    }
  }

  ArgList get_enable_arg() {
    if (std::holds_alternative<Arg>(enable_arg_)) {
      return ArgList({std::get<Arg>(enable_arg_)});
    }
    if (std::holds_alternative<std::string>(enable_arg_)) {
      std::string arg_name = std::get<std::string>(enable_arg_);
      if (arg_name != "") {
        return from_config(arg_name);
      }
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

    // TensorCompareOp works for device, cuda host and cuda managed tensors
    if (storage_type_ == "device" || storage_type_ == "cuda_managed" || storage_type_ == "host") {
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
                                          std::string("system"), std::string("cuda_managed")));

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

// Smoke test app for window_close_callback wiring in C++ (headless)
class HolovizCloseCallbackApp : public holoscan::Application {
 public:
  explicit HolovizCloseCallbackApp(std::shared_ptr<std::atomic<int>> cb_calls,
                                   ArgList source_args = {})
      : Application(), cb_calls_(std::move(cb_calls)), source_args_(std::move(source_args)) {
    if (source_args_.size() == 0) {
      source_args_ = ArgList({Arg("rows", height_), Arg("columns", width_), Arg("channels", 3)});
    }
  }

  void compose() override {
    auto allocator = Arg("allocator", make_resource<holoscan::UnboundedAllocator>("allocator"));
    auto count = make_condition<CountCondition>(1);
    auto headless = Arg("headless", true);

    auto source = make_operator<ops::PingTensorTxOp>("ping_source", source_args_);

    // Provide a no-arg callback that increments the atomic count; it should not be called
    // during a normal run (no window close event is triggered).
    auto cb = Arg("window_close_callback",
                  ops::HolovizOp::WindowCloseCallbackFunction(
                      [calls = cb_calls_]() { calls->fetch_add(1); }));

    auto renderer = make_operator<ops::HolovizOp>("renderer",
                                                  count,
                                                  headless,
                                                  from_config("holoviz_tensor_input"),
                                                  allocator,
                                                  Arg("width", uint32_t(width_)),
                                                  Arg("height", uint32_t(height_)),
                                                  cb);

    add_flow(source, renderer, {{"out", "receivers"}});
  }

 private:
  const int32_t width_ = 32, height_ = 32;
  std::shared_ptr<std::atomic<int>> cb_calls_;
  ArgList source_args_{};
};

TEST(HolovizApps, TestWindowCloseCallbackSmokeCpp) {
  auto cb_calls = std::make_shared<std::atomic<int>>(0);
  auto app = make_application<HolovizCloseCallbackApp>(cb_calls);

  const std::string config_file = test_config.get_test_data_file("app_config.yaml");
  app->config(config_file);

  testing::internal::CaptureStderr();
  try {
    app->run();
  } catch (const std::exception& ex) {
    GTEST_FATAL_FAILURE_(
        fmt::format("{}{}", testing::internal::GetCapturedStderr(), ex.what()).c_str());
  }
  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("error") == std::string::npos) << "=== LOG ===\n"
                                                             << log_output << "\n===========\n";
  // Ensure the callback was not called during normal run
  EXPECT_EQ(cb_calls->load(), 0);
}

// Holoviz operator that explicitly triggers the window close path once.
class HolovizCloseCallbackTriggerOp : public ops::HolovizOp {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS_SUPER(HolovizCloseCallbackTriggerOp, ops::HolovizOp)

  HolovizCloseCallbackTriggerOp() = default;

  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override {
    // Run the regular Holoviz compute path first.
    ops::HolovizOp::compute(op_input, op_output, context);
    if (!triggered_) {
      // Simulate a window close event by invoking the same helper Holoviz uses internally.
      disable_via_window_close();
      triggered_ = true;
    }
  }

 private:
  bool triggered_ = false;
};

// App that wires a callback and uses HolovizCloseCallbackTriggerOp to force invocation.
class HolovizCloseCallbackInvokeApp : public holoscan::Application {
 public:
  explicit HolovizCloseCallbackInvokeApp(std::shared_ptr<std::atomic<int>> cb_calls,
                                         ArgList source_args = {})
      : Application(), cb_calls_(std::move(cb_calls)), source_args_(std::move(source_args)) {
    if (source_args_.size() == 0) {
      source_args_ = ArgList({Arg("rows", height_), Arg("columns", width_), Arg("channels", 3)});
    }
  }

  void compose() override {
    auto allocator = Arg("allocator", make_resource<holoscan::UnboundedAllocator>("allocator"));
    auto count = make_condition<CountCondition>(1);
    auto headless = Arg("headless", true);

    auto source = make_operator<ops::PingTensorTxOp>("ping_source", source_args_);

    auto cb = Arg("window_close_callback",
                  ops::HolovizOp::WindowCloseCallbackFunction(
                      [calls = cb_calls_]() { calls->fetch_add(1); }));

    auto renderer =
        make_operator<HolovizCloseCallbackTriggerOp>("renderer",
                                                     count,
                                                     headless,
                                                     from_config("holoviz_tensor_input"),
                                                     allocator,
                                                     Arg("width", uint32_t(width_)),
                                                     Arg("height", uint32_t(height_)),
                                                     cb);

    add_flow(source, renderer, {{"out", "receivers"}});
  }

 private:
  const int32_t width_ = 32, height_ = 32;
  std::shared_ptr<std::atomic<int>> cb_calls_;
  ArgList source_args_{};
};

TEST(HolovizApps, TestWindowCloseCallbackInvokedCpp) {
  auto cb_calls = std::make_shared<std::atomic<int>>(0);
  auto app = make_application<HolovizCloseCallbackInvokeApp>(cb_calls);

  const std::string config_file = test_config.get_test_data_file("app_config.yaml");
  app->config(config_file);

  testing::internal::CaptureStderr();
  try {
    app->run();
  } catch (const std::exception& ex) {
    GTEST_FATAL_FAILURE_(
        fmt::format("{}{}", testing::internal::GetCapturedStderr(), ex.what()).c_str());
  }
  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("error") == std::string::npos) << "=== LOG ===\n"
                                                             << log_output << "\n===========\n";
  // The trigger op should have called disable_via_window_close(), which in turn invokes the
  // configured window_close_callback exactly once.
  EXPECT_EQ(cb_calls->load(), 1);
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
  } catch (const std::exception& ex) {
    exception = ex.what();
  }
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

// This operator is used to test the render buffer input, it creates a video buffer and emits it as
// an output. It also creates a simple cube geometry and specs and emits it.
class RenderBufferSourceOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(RenderBufferSourceOp)

  RenderBufferSourceOp() = default;

  void initialize() override {
    allocator_ = fragment()->make_resource<UnboundedAllocator>("pool");
    add_arg(allocator_);
    Operator::initialize();
  }

  void setup(OperatorSpec& spec) override {
    spec.output<holoscan::gxf::Entity>("outputs");
    spec.output<std::vector<ops::HolovizOp::InputSpec>>("output_specs");
    spec.output<holoscan::gxf::Entity>("color_buffer");
    spec.output<holoscan::gxf::Entity>("depth_buffer");
  }

  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override {
    // Create a new entity for geometries
    auto entity = holoscan::gxf::Entity::New(&context);
    auto specs = std::vector<ops::HolovizOp::InputSpec>();

    // Create a simple cube geometry
    add_data<24, 3>(entity,
                    "cube",
                    {{{-0.2f, -0.2f, -0.2f}, {-0.2f, 0.2f, -0.2f},  // Front face
                      {-0.2f, 0.2f, -0.2f},  {0.2f, 0.2f, -0.2f},  {0.2f, 0.2f, -0.2f},
                      {0.2f, -0.2f, -0.2f},  {0.2f, -0.2f, -0.2f}, {-0.2f, -0.2f, -0.2f},
                      {-0.2f, -0.2f, 0.2f},  {-0.2f, 0.2f, 0.2f},  // Back face
                      {-0.2f, 0.2f, 0.2f},   {0.2f, 0.2f, 0.2f},   {0.2f, 0.2f, 0.2f},
                      {0.2f, -0.2f, 0.2f},   {0.2f, -0.2f, 0.2f},  {-0.2f, -0.2f, 0.2f},
                      {-0.2f, -0.2f, -0.2f}, {-0.2f, -0.2f, 0.2f},  // Connecting lines
                      {-0.2f, 0.2f, -0.2f},  {-0.2f, 0.2f, 0.2f},  {0.2f, 0.2f, -0.2f},
                      {0.2f, 0.2f, 0.2f},    {0.2f, -0.2f, -0.2f}, {0.2f, -0.2f, 0.2f}}},
                    context);

    op_output.emit(entity, "outputs");

    // Create input spec for the cube
    ops::HolovizOp::InputSpec cube_spec;
    cube_spec.tensor_name_ = "cube";
    cube_spec.type_ = ops::HolovizOp::InputType::LINES_3D;
    cube_spec.color_ = {1.0f, 0.0f, 0.0f, 1.0f};  // Red color
    specs.push_back(cube_spec);

    op_output.emit(specs, "output_specs");

    // Create a render buffer
    auto render_entity = nvidia::gxf::Entity::New(context.context());
    auto color_buffer = render_entity.value().add<nvidia::gxf::VideoBuffer>("color_buffer");

    // Create allocator handle
    auto allocator = nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(context.context(),
                                                                         allocator_->gxf_cid());

    // Allocate buffer
    color_buffer.value()->resize<nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_RGBA>(
        640U,
        480U,
        nvidia::gxf::SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR,
        nvidia::gxf::MemoryStorageType::kDevice,
        allocator.value());

    // Emit output
    auto result = holoscan::gxf::Entity(std::move(render_entity.value()));
    op_output.emit(result, "color_buffer");

    // Create a depth buffer
    auto depth_entity = nvidia::gxf::Entity::New(context.context());
    auto depth_buffer = depth_entity.value().add<nvidia::gxf::VideoBuffer>("depth_buffer");

    // Allocate buffer
    depth_buffer.value()->resize<nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_D32F>(
        640U,
        480U,
        nvidia::gxf::SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR,
        nvidia::gxf::MemoryStorageType::kDevice,
        allocator.value());

    // Emit output
    result = holoscan::gxf::Entity(std::move(depth_entity.value()));
    op_output.emit(result, "depth_buffer");
  }

 private:
  template <std::size_t N, std::size_t C>
  void add_data(holoscan::gxf::Entity& entity, const char* name,
                const std::array<std::array<float, C>, N>& data, ExecutionContext& context) {
    auto allocator = nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(context.context(),
                                                                         allocator_->gxf_cid());
    auto tensor = static_cast<nvidia::gxf::Entity&>(entity).add<nvidia::gxf::Tensor>(name).value();
    tensor->reshape<float>(
        nvidia::gxf::Shape({N, C}), nvidia::gxf::MemoryStorageType::kHost, allocator.value());
    std::memcpy(tensor->pointer(), data.data(), N * C * sizeof(float));
  }

  std::shared_ptr<UnboundedAllocator> allocator_;
};

// This operator is used to test the render buffer output, it receives a render buffer and logs it
class RenderBufferSinkOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(RenderBufferSinkOp)

  RenderBufferSinkOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.input<nvidia::gxf::Entity>("render_buffer_input");
    spec.input<nvidia::gxf::Entity>("depth_buffer_input");
  }

  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override {
    auto maybe_render_buffer = op_input.receive<nvidia::gxf::Entity>("render_buffer_input");
    ASSERT_TRUE(maybe_render_buffer && !maybe_render_buffer.value().is_null());
    auto video_buffer = maybe_render_buffer.value().get<nvidia::gxf::VideoBuffer>();
    ASSERT_TRUE(video_buffer && !video_buffer.value().is_null());
    auto render_buffer_info = video_buffer.value()->video_frame_info();
    ASSERT_EQ(render_buffer_info.width, 640U);
    ASSERT_EQ(render_buffer_info.height, 480U);
    ASSERT_EQ(render_buffer_info.color_format, nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_RGBA);
    holoscan::log_info("Received render buffer");

    auto maybe_depth_buffer = op_input.receive<nvidia::gxf::Entity>("depth_buffer_input");
    ASSERT_TRUE(maybe_depth_buffer && !maybe_depth_buffer.value().is_null());
    video_buffer = maybe_depth_buffer.value().get<nvidia::gxf::VideoBuffer>();
    ASSERT_TRUE(video_buffer && !video_buffer.value().is_null());
    auto depth_buffer_info = video_buffer.value()->video_frame_info();
    ASSERT_EQ(depth_buffer_info.width, 640U);
    ASSERT_EQ(depth_buffer_info.height, 480U);
    ASSERT_EQ(depth_buffer_info.color_format, nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_D32F);
    holoscan::log_info("Received depth buffer");
  }
};

// This application is used to test the render buffer input and output
class RenderBufferInputApp : public holoscan::Application {
 public:
  void compose() override {
    auto source = make_operator<RenderBufferSourceOp>("source");
    auto renderer = make_operator<ops::HolovizOp>("renderer",
                                                  make_condition<CountCondition>(2),
                                                  Arg("headless", true),
                                                  Arg("enable_render_buffer_input", true),
                                                  Arg("enable_render_buffer_output", true),
                                                  Arg("enable_depth_buffer_input", true),
                                                  Arg("enable_depth_buffer_output", true),
                                                  Arg("width", 640U),
                                                  Arg("height", 480U));
    auto sink = make_operator<RenderBufferSinkOp>("sink");

    add_flow(source, renderer, {{"outputs", "receivers"}});
    add_flow(source, renderer, {{"output_specs", "input_specs"}});
    add_flow(source, renderer, {{"color_buffer", "render_buffer_input"}});
    add_flow(renderer, sink, {{"render_buffer_output", "render_buffer_input"}});
    add_flow(source, renderer, {{"depth_buffer", "depth_buffer_input"}});
    add_flow(renderer, sink, {{"depth_buffer_output", "depth_buffer_input"}});
  }

  InputType input_type_;
};

TEST(HolovizApps, TestRenderBufferInput) {
  auto app = make_application<RenderBufferInputApp>();

  // capture output to check that the expected messages were logged
  testing::internal::CaptureStderr();
  EXPECT_NO_THROW(app->run());
  std::string log_output = testing::internal::GetCapturedStderr();

  EXPECT_TRUE((log_output.find("Received render buffer") != std::string::npos) &&
              (log_output.find("Received depth buffer") != std::string::npos))
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
}
