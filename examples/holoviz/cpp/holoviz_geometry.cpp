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
#include <getopt.h>

#include <memory>
#include <string>
#include <vector>

#include <holoscan/holoscan.hpp>
#include <holoscan/operators/holoviz/holoviz.hpp>
#include <holoscan/operators/video_stream_replayer/video_stream_replayer.hpp>

#include <gxf/std/tensor.hpp>

namespace holoscan::ops {

/**
 * Example of an operator generating geometric primitives to be displayed by the HolovizOp
 *
 *  This operator has:
 *       outputs: "output_tensor"
 *       output_specs: "output_specs"
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
    spec.output<gxf::Entity>("outputs");
    spec.output<std::vector<HolovizOp::InputSpec>>("output_specs");
  }

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

  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override {
    auto entity = gxf::Entity::New(&context);
    auto specs = std::vector<HolovizOp::InputSpec>();

    // Now draw various different types of geometric primitives.
    // In all cases, x and y are normalized coordinates in the range [0, 1].
    // x runs from left to right and y from bottom to top.

    ///////////////////////////////////////////
    // Create a tensor defining four rectnagles
    ///////////////////////////////////////////
    // For rectangles (bounding boxes), they are defined by a pair of
    // 2-tuples defining the upper-left and lower-right coordinates of a
    // box: (x1, y1), (x2, y2).
    add_data<8, 2>(entity,
                   "boxes",
                   {{{0.1f, 0.2f},
                     {0.8f, 0.5f},
                     {0.2f, 0.4f},
                     {0.3f, 0.6f},
                     {0.3f, 0.5f},
                     {0.4f, 0.7f},
                     {0.5f, 0.7f},
                     {0.6f, 0.9f}}},
                   context);

    /////////////////////////////////////////
    // Create a tensor defining two triangles
    /////////////////////////////////////////
    // Each triangle is defined by a set of 3 (x, y) coordinate pairs.
    add_data<6, 2>(entity,
                   "triangles",
                   {{{0.1f, 0.8f},
                     {0.18f, 0.75f},
                     {0.14f, 0.66f},
                     {0.3f, 0.8f},
                     {0.38f, 0.75f},
                     {0.34f, 0.56f}}},
                   context);

    ///////////////////////////////////////
    // Create a tensor defining two crosses
    ///////////////////////////////////////
    // Each cross is defined by an (x, y, size) 3-tuple
    add_data<2, 3>(entity, "crosses", {{{0.25f, 0.25f, 0.05f}, {0.75f, 0.25f, 0.10f}}}, context);

    ///////////////////////////////////////
    // Create a tensor defining three ovals
    ///////////////////////////////////////
    // Each oval is defined by an (x, y, size_x, size_y) 4-tuple
    add_data<3, 4>(entity,
                   "ovals",
                   {{{0.25f, 0.65f, 0.10f, 0.05f},
                     {0.25f, 0.65f, 0.10f, 0.05f},
                     {0.75f, 0.65f, 0.05f, 0.10f}}},
                   context);

    ////////////////////////////////////////
    // Create a time-varying "points" tensor
    ////////////////////////////////////////
    // Set of (x, y) points with 50 points equally spaced along x whose y
    // coordinate varies sinusoidally over time.
    constexpr uint32_t POINTS = 50;
    constexpr float PI = 3.14f;
    std::array<std::array<float, 2>, POINTS> point_coords;
    for (uint32_t i = 0; i < POINTS; ++i) {
      point_coords[i][0] = (1.f / POINTS) * i;
      point_coords[i][1] =
          0.8f + 0.1f * std::sin(8.f * PI * point_coords[i][0] + count_ / 60.f * 2.f * PI);
    }

    add_data(entity, "points", point_coords, context);

    /////////////////////////////////////
    // Create a tensor for "label_coords"
    /////////////////////////////////////
    // Set of two (x, y) points marking the location of text labels
    add_data<2, 2>(entity, "label_coords", {{{0.10f, 0.1f}, {0.70f, 0.1f}}}, context);

    /////////////////////////////////////
    // Create a tensor for "dynamic_text"
    /////////////////////////////////////
    // Set of two (x, y) points marking the location of text labels
    add_data<2, 2>(entity, "dynamic_text", {{{0.f, 0.f}}}, context);

    // emit the tensors
    op_output.emit(entity, "outputs");

    /////////////////////////////////////////
    // Create a input spec for "dynamic_text"
    /////////////////////////////////////////
    // To dynamically change the input spec create a list of HolovizOp.InputSpec objects
    // and pass it to Holoviz.
    // All properties of the input spec (type, color, text, line width, ...) can be changed
    // dynamically.
    HolovizOp::InputSpec spec;
    spec.tensor_name_ = "dynamic_text";
    spec.type_ = HolovizOp::InputType::TEXT;
    spec.text_.push_back(std::string("Frame ") + std::to_string(count_));
    specs.push_back(spec);

    // emit the output specs
    op_output.emit(specs, "output_specs");

    count_++;
  }

 private:
  std::shared_ptr<UnboundedAllocator> allocator_;
  uint32_t count_ = 0;
};

}  // namespace holoscan::ops

/**
 * Example of an application that uses the operators defined above.
 *
 * This application has the following operators:
 *
 * - VideoStreamReplayerOp
 * - GeometrySourceOp
 * - HolovizOp
 *
 * The VideoStreamReplayerOp reads a video file and sends the frames to the ImageProcessingOp.
 * The GeometrySourceOp creates geometric primitives and sends it to the HolovizOp.
 * The HolovizOp displays the processed frames and geometry.
 */
class HolovizGeometryApp : public holoscan::Application {
 public:
  /**
   * @brief Construct a new HolovizGeometryApp object
   *
   * @param count Limits the number of frames to show before the application ends.
   *   Set to 0 by default. The video stream will not automatically stop.
   *   Any positive integer will limit on the number of frames displayed.
   */
  explicit HolovizGeometryApp(uint64_t count) : count_(count) {}

  void compose() override {
    using namespace holoscan;

    ArgList args;
    auto data_directory = std::getenv("HOLOSCAN_INPUT_PATH");
    if (data_directory != nullptr && data_directory[0] != '\0') {
      auto video_directory = std::filesystem::path(data_directory);
      video_directory /= "racerx";
      args.add(Arg("directory", video_directory.string()));
      HOLOSCAN_LOG_INFO("Using video from {}", video_directory.string());
    }

    // Define the replayer, geometry source and holoviz operators
    auto replayer =
        make_operator<ops::VideoStreamReplayerOp>("replayer",
                                                  Arg("directory", std::string("../data/racerx")),
                                                  Arg("basename", std::string("racerx")),
                                                  Arg("frame_rate", 0.f),
                                                  Arg("repeat", true),
                                                  Arg("realtime", true),
                                                  Arg("count", count_),
                                                  args);

    auto source = make_operator<ops::GeometrySourceOp>("source");

    // build the input spec list
    std::vector<ops::HolovizOp::InputSpec> input_spec;
    int32_t priority = 0;

    auto& video_spec =
        input_spec.emplace_back(ops::HolovizOp::InputSpec("", ops::HolovizOp::InputType::COLOR));
    video_spec.line_width_ = 2.f;
    video_spec.opacity_ = 0.5f;
    video_spec.priority_ = priority++;

    // Parameters defining the rectangle primitives
    auto& boxes_spec = input_spec.emplace_back(
        ops::HolovizOp::InputSpec("boxes", ops::HolovizOp::InputType::RECTANGLES));
    boxes_spec.line_width_ = 2.f;
    boxes_spec.color_ = {1.0f, 0.0f, 1.0f, 0.5f};
    boxes_spec.priority_ = priority++;

    // line strip reuses the rectangle coordinates. This will make
    // a connected set of line segments through the diagonals of
    // each box.
    auto& line_strip_spec = input_spec.emplace_back(
        ops::HolovizOp::InputSpec("boxes", ops::HolovizOp::InputType::LINE_STRIP));
    line_strip_spec.line_width_ = 3.f;
    line_strip_spec.color_ = {0.4f, 0.4f, 1.0f, 0.7f};
    line_strip_spec.priority_ = priority++;

    // Lines also reuses the boxes coordinates so will plot a set of
    // disconnected line segments along the box diagonals.
    auto& lines_spec = input_spec.emplace_back(
        ops::HolovizOp::InputSpec("boxes", ops::HolovizOp::InputType::LINES));
    lines_spec.line_width_ = 3.f;
    lines_spec.color_ = {0.4f, 1.0f, 0.4f, 0.7f};
    lines_spec.priority_ = priority++;

    // Parameters defining the triangle primitives
    auto& triangles_spec = input_spec.emplace_back(
        ops::HolovizOp::InputSpec("triangles", ops::HolovizOp::InputType::TRIANGLES));
    triangles_spec.color_ = {1.0f, 0.0f, 0.0f, 0.5f};
    triangles_spec.priority_ = priority++;

    // Parameters defining the crosses primitives
    auto& crosses_spec = input_spec.emplace_back(
        ops::HolovizOp::InputSpec("crosses", ops::HolovizOp::InputType::CROSSES));
    crosses_spec.line_width_ = 3.f;
    crosses_spec.color_ = {0.0f, 1.0f, 0.0f, 1.0f};
    crosses_spec.priority_ = priority++;

    // Parameters defining the ovals primitives
    auto& ovals_spec = input_spec.emplace_back(
        ops::HolovizOp::InputSpec("ovals", ops::HolovizOp::InputType::OVALS));
    ovals_spec.opacity_ = 0.5f;
    ovals_spec.line_width_ = 2.f;
    ovals_spec.color_ = {1.0f, 1.0f, 1.0f, 1.0f};
    ovals_spec.priority_ = priority++;

    // Parameters defining the points primitives
    auto& points_spec = input_spec.emplace_back(
        ops::HolovizOp::InputSpec("points", ops::HolovizOp::InputType::POINTS));
    points_spec.point_size_ = 4.f;
    points_spec.color_ = {1.0f, 1.0f, 1.0f, 1.0f};
    points_spec.priority_ = priority++;

    // Parameters defining the label_coords primitives
    auto& label_coords_spec = input_spec.emplace_back(
        ops::HolovizOp::InputSpec("label_coords", ops::HolovizOp::InputType::TEXT));
    label_coords_spec.color_ = {1.0f, 1.0f, 1.0f, 1.0f};
    label_coords_spec.text_ = {"label_1", "label_2"};
    label_coords_spec.priority_ = priority++;

    auto visualizer = make_operator<ops::HolovizOp>("holoviz",
                                                    Arg("width", 854u),
                                                    Arg("height", 480u),
                                                    Arg("tensors", input_spec));

    // Define the workflow: source -> holoviz
    add_flow(source, visualizer, {{"outputs", "receivers"}});
    add_flow(source, visualizer, {{"output_specs", "input_specs"}});
    add_flow(replayer, visualizer, {{"output", "receivers"}});
  }

 private:
  uint64_t count_ = 0;
};

int main(int argc, char** argv) {
  // Parse args
  struct option long_options[] = {
      {"help", no_argument, 0, 'h'}, {"count", required_argument, 0, 'c'}, {0, 0, 0, 0}};
  uint64_t count;
  while (true) {
    int option_index = 0;
    const int c = getopt_long(argc, argv, "hc:", long_options, &option_index);

    if (c == -1) { break; }

    const std::string argument(optarg ? optarg : "");
    switch (c) {
      case 'h':
        std::cout << "Usage: " << argv[0] << " [options]" << std::endl
                  << "Options:" << std::endl
                  << "  -h, --help     display this information" << std::endl
                  << "  -c, --count    Set the number of frames to display the video" << std::endl
                  << std::endl;
        return EXIT_SUCCESS;

      case 'c':
        count = std::stoull(argument);
        break;
      default:
        throw std::runtime_error("Unhandled option ");
    }
  }

  auto app = holoscan::make_application<HolovizGeometryApp>(count);
  app->run();

  return 0;
}
