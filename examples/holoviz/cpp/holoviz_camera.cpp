/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
// NOLINTFILE(concurrency-mt-unsafe)

#include <getopt.h>

#include <chrono>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <holoscan/holoscan.hpp>
#include <holoscan/operators/holoviz/holoviz.hpp>

#include <gxf/multimedia/camera.hpp>
#include <gxf/std/tensor.hpp>

namespace holoscan::ops {
/**
 * This operatore receives camera pose information and prints to the console (but only once every
 * second).
 */
class CameraPoseRxOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(CameraPoseRxOp)

  CameraPoseRxOp() = default;

  void setup(OperatorSpec& spec) override { spec.input<nvidia::gxf::Pose3D>("input"); }

  void start() override { start_time_ = std::chrono::steady_clock::now(); }

  void compute(InputContext& op_input, [[maybe_unused]] OutputContext& op_output,
               [[maybe_unused]] ExecutionContext& context) override {
    auto value = op_input.receive<std::shared_ptr<nvidia::gxf::Pose3D>>("input").value();

    // print once every second
    if (std::chrono::steady_clock::now() - start_time_ > std::chrono::seconds(1)) {
      HOLOSCAN_LOG_INFO("Received camera pose:\nrotation {}\ntranslation {}",
                        value->rotation,
                        value->translation);

      start_time_ = std::chrono::steady_clock::now();
    }
  }

 private:
  std::chrono::steady_clock::time_point start_time_;
};

/**
 * The operator generates a 3D cube, each side is output as a separate tensor. It also randomly
 * switches between camera positions each second.
 */
class GeometrySourceOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(GeometrySourceOp)

  GeometrySourceOp() = default;

  void initialize() override {
    // Create an allocator for the operator
    allocator_ = fragment()->make_resource<UnboundedAllocator>("pool");
    // Add the allocator to the operator so that it is initialized
    add_arg(allocator_);

    // Call the base class initialize function
    Operator::initialize();
  }

  void setup(OperatorSpec& spec) override {
    spec.output<gxf::Entity>("geometry_output");
    spec.output<std::array<float, 3>>("camera_eye_output");
    spec.output<std::array<float, 3>>("camera_look_at_output");
    spec.output<std::array<float, 3>>("camera_up_output");
  }

  void start() override { start_time_ = std::chrono::steady_clock::now(); }

  /**
   * Helper function to add a tensor with data to an entity.
   */
  template <std::size_t N, std::size_t C>
  void add_data(gxf::Entity& entity, const char* name,
                const std::array<std::array<float, C>, N>& data, ExecutionContext& context) {
    // get Handle to underlying nvidia::gxf::Allocator from std::shared_ptr<holoscan::Allocator>
    auto allocator = nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(context.context(),
                                                                         allocator_->gxf_cid());
    // add a tensor
    auto tensor = static_cast<nvidia::gxf::Entity&>(entity).add<nvidia::gxf::Tensor>(name).value();
    // reshape the tensor to the size of the data
    tensor->reshape<float>(
        nvidia::gxf::Shape({N, C}), nvidia::gxf::MemoryStorageType::kHost, allocator.value());
    // copy the data to the tensor
    std::memcpy(tensor->pointer(), data.data(), N * C * sizeof(float));
  }

  void compute([[maybe_unused]] InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override {
    auto entity = gxf::Entity::New(&context);

    // Create a colored box
    // Each triangle is defined by a set of 3 (x, y, z) coordinate pairs.
    add_data<6, 3>(entity,
                   "back",
                   {{{-1.F, -1.F, -1.F},
                     {1.F, -1.F, -1.F},
                     {1.F, 1.F, -1.F},
                     {1.F, 1.F, -1.F},
                     {-1.F, 1.F, -1.F},
                     {-1.F, -1.F, -1.F}}},
                   context);
    add_data<6, 3>(entity,
                   "front",
                   {{{-1.F, -1.F, 1.F},
                     {1.F, -1.F, 1.F},
                     {1.F, 1.F, 1.F},
                     {1.F, 1.F, 1.F},
                     {-1.F, 1.F, 1.F},
                     {-1.F, -1.F, 1.F}}},
                   context);
    add_data<6, 3>(entity,
                   "right",
                   {{{1.F, -1.F, -1.F},
                     {1.F, -1.F, 1.F},
                     {1.F, 1.F, 1.F},
                     {1.F, 1.F, 1.F},
                     {1.F, 1.F, -1.F},
                     {1.F, -1.F, -1.F}}},
                   context);
    add_data<6, 3>(entity,
                   "left",
                   {{{-1.F, -1.F, -1.F},
                     {-1.F, -1.F, 1.F},
                     {-1.F, 1.F, 1.F},
                     {-1.F, 1.F, 1.F},
                     {-1.F, 1.F, -1.F},
                     {-1.F, -1.F, -1.F}}},
                   context);
    add_data<6, 3>(entity,
                   "top",
                   {{{-1.F, 1.F, -1.F},
                     {-1.F, 1.F, 1.F},
                     {1.F, 1.F, 1.F},
                     {1.F, 1.F, 1.F},
                     {1.F, 1.F, -1.F},
                     {-1.F, 1.F, -1.F}}},
                   context);
    add_data<6, 3>(entity,
                   "bottom",
                   {{{-1.F, -1.F, -1.F},
                     {-1.F, -1.F, 1.F},
                     {1.F, -1.F, 1.F},
                     {1.F, -1.F, 1.F},
                     {1.F, -1.F, -1.F},
                     {-1.F, -1.F, -1.F}}},
                   context);

    // emit the tensors
    op_output.emit(entity, "geometry_output");

    // every second, switch camera
    if (std::chrono::steady_clock::now() - start_time_ > std::chrono::seconds(1)) {
      // NOLINTBEGIN(cert-msc30-c,cert-msc50-cpp,concurrency-mt-unsafe)
      // NOLINTBEGIN(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
      const int camera = std::rand() % sizeof(cameras_) / sizeof(cameras_[0]);
      // NOLINTEND(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
      // NOLINTEND(cert-msc30-c,cert-msc50-cpp,concurrency-mt-unsafe)

      // NOLINTBEGIN(cppcoreguidelines-pro-bounds-constant-array-index)
      camera_eye_ = cameras_[camera][0];
      camera_look_at_ = cameras_[camera][1];
      camera_up_ = cameras_[camera][2];
      // NOLINTEND(cppcoreguidelines-pro-bounds-constant-array-index)

      op_output.emit(camera_eye_, "camera_eye_output");
      op_output.emit(camera_look_at_, "camera_look_at_output");
      op_output.emit(camera_up_, "camera_up_output");

      start_time_ = std::chrono::steady_clock::now();
    }
  }

  const std::array<float, 3>& camera_eye() const { return camera_eye_; }
  const std::array<float, 3>& camera_look_at() const { return camera_look_at_; }
  const std::array<float, 3>& camera_up() const { return camera_up_; }

 private:
  std::shared_ptr<UnboundedAllocator> allocator_;

  std::chrono::steady_clock::time_point start_time_;

  // NOLINTBEGIN(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays)

  // define some cameras we switch between
  static constexpr std::array<float, 3> cameras_[4][3]{
      {{0.F, 0.F, 5.F}, {1.F, 1.F, 0.F}, {0.F, 1.F, 0.F}},
      {{1.F, 1.F, -3.F}, {0.F, 0.F, 0.F}, {0.F, 1.F, 0.F}},
      {{3.F, -4.F, 0.F}, {0.F, 1.F, 1.F}, {1.F, 0.F, 0.F}},
      {{-2.F, 0.F, -3.F}, {-1.F, 0.F, -1.F}, {0.F, 0.F, 1.F}}};

  // NOLINTEND(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays)
  std::array<float, 3> camera_eye_ = cameras_[0][0];
  std::array<float, 3> camera_look_at_ = cameras_[0][1];
  std::array<float, 3> camera_up_ = cameras_[0][2];
};

}  // namespace holoscan::ops

/**
 * Example of an application that uses the operators defined above.
 *
 * This application has the following operators:
 *
 * - GeometrySourceOp
 * - HolovizOp
 * - CameraPoseRxOp
 *
 * The GeometrySourceOp creates geometric primitives and camera properties and sends it to the
 * HolovizOp. It runs at 60 Hz.
 * The HolovizOp displays the geometry and is using the camera properties.
 * The CameraPoseRxOp receives camera pose information from the HolovizOp and prints it on the
 * console.
 */
class HolovizCameraApp : public holoscan::Application {
 public:
  /**
   * @brief Construct a new HolovizCameraApp object
   *
   * @param count Limits the number of frames to show before the application ends.
   *   Set to -1 by default. Any positive integer will limit on the number of frames displayed.
   */
  explicit HolovizCameraApp(int64_t count) : count_(count) {}

  void compose() override {
    using namespace holoscan;

    auto source = make_operator<ops::GeometrySourceOp>(
        "source",
        // run at 60 Hz
        make_condition<PeriodicCondition>("frame_limiter",
                                          Arg("recess_period", std::string("60Hz"))));
    // Limit the total number of frames (if count_ is positive)
    if (count_ >= 0) {
      source->add_arg(make_condition<CountCondition>("frame_limit", count_));
    }

    // build the input spec list
    std::vector<ops::HolovizOp::InputSpec> input_spec;

    // Parameters defining the triangle primitives
    const std::array<const char*, 6> spec_names{"back", "front", "left", "right", "top", "bottom"};
    unsigned int index = 0;
    for (const auto* spec_name : spec_names) {
      auto& spec = input_spec.emplace_back(
          ops::HolovizOp::InputSpec(spec_name, ops::HolovizOp::InputType::TRIANGLES_3D));
      spec.color_ = {static_cast<float>((index + 1) & 1U),
                     static_cast<float>(((index + 1) / 2) & 1U),
                     static_cast<float>(((index + 1) / 4) & 1U),
                     1.0F};
      index++;
    }

    auto visualizer = make_operator<ops::HolovizOp>(
        "holoviz",
        Arg("width", 1024U),
        Arg("height", 1024U),
        Arg("tensors", input_spec),
        Arg("enable_camera_pose_output", true),
        Arg("camera_pose_output_type", std::string("extrinsics_model")),
        // pass the initial camera properties to HolovizOp
        Arg("camera_eye", source->camera_eye()),
        Arg("camera_look_at", source->camera_look_at()),
        Arg("camera_up", source->camera_up()));

    auto camera_pose_rx = make_operator<ops::CameraPoseRxOp>("camera_pose_rx");

    // Define the workflow: source -> holoviz
    add_flow(source, visualizer, {{"geometry_output", "receivers"}});
    add_flow(source, visualizer, {{"camera_eye_output", "camera_eye_input"}});
    add_flow(source, visualizer, {{"camera_look_at_output", "camera_look_at_input"}});
    add_flow(source, visualizer, {{"camera_up_output", "camera_up_input"}});
    add_flow(visualizer, camera_pose_rx, {{"camera_pose_output", "input"}});
  }

 private:
  int64_t count_ = -1;
};

int main(int argc, char** argv) {
  // Parse args
  // NOLINTBEGIN(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays)
  struct option long_options[] = {{"help", no_argument, nullptr, 'h'},
                                  {"count", required_argument, nullptr, 'c'},
                                  {nullptr, 0, nullptr, 0}};
  // NOLINTEND(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays)
  int64_t count = -1;
  while (true) {
    int option_index = 0;
    // NOLINTBEGIN(concurrency-mt-unsafe)
    const int c = getopt_long(argc, argv, "hc:", static_cast<option*>(long_options), &option_index);
    // NOLINTEND(concurrency-mt-unsafe)

    if (c == -1) {
      break;
    }

    const std::string argument(optarg != nullptr ? optarg : "");
    switch (c) {
      case 'h':
      case '?':
        std::cout
            << "Usage: " << argv[0] << " [options]" << std::endl
            << "Options:" << std::endl
            << "  -h, --help     display this information" << std::endl
            << "  -c, --count    limits the number of frames to show before the application "
               "ends. Set to `"
            << count
            << "` by default. Any positive integer will limit on the number of frames displayed."
            << std::endl;
        return EXIT_SUCCESS;

      case 'c':
        count = std::stoll(argument);
        break;
      default:
        throw std::runtime_error(fmt::format("Unhandled option `{}`", static_cast<char>(c)));
    }
  }

  auto app = holoscan::make_application<HolovizCameraApp>(count);
  app->run();

  return 0;
}
