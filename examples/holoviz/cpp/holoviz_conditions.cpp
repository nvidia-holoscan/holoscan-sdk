/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

/**
 * @file holoviz_conditions.cpp
 * @brief Example demonstrating the use of FirstPixelOutCondition and PresentDoneCondition with
 * HolovizOp
 *
 * This example shows how to synchronize operator execution with the display using two different
 * conditions:
 * - FirstPixelOutCondition: Waits for the first pixel of the next display refresh cycle to leave
 * the display engine for the display
 * - PresentDoneCondition: Waits for the presentation to complete
 *
 * Both conditions ensure that the source operator runs in sync with the display refresh rate,
 * avoiding generating frames faster than they can be displayed.
 *
 * Key concepts demonstrated:
 * - Switching between FirstPixelOutCondition and PresentDoneCondition via command line
 * - Creating and emitting HolovizOp InputSpecs dynamically
 * - Calculating and displaying frame rate information
 */

#include <getopt.h>

#include <chrono>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <holoscan/holoscan.hpp>
#include <holoscan/operators/holoviz/conditions/first_pixel_out.hpp>
#include <holoscan/operators/holoviz/conditions/present_done.hpp>
#include <holoscan/operators/holoviz/holoviz.hpp>

#include <gxf/std/tensor.hpp>
namespace holoscan::ops {

/**
 * @brief Source operator that generates text data for visualization
 *
 * This operator generates dynamic text showing the current frame number and frame rate.
 */
class SourceOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(SourceOp)

  SourceOp() = default;

  void initialize() override {
    // Create an allocator for the operator
    allocator_ = fragment()->make_resource<UnboundedAllocator>("pool");
    // Add the allocator to the operator so that it is initialized
    add_arg(allocator_);

    // Call the base class initialize function
    Operator::initialize();
  }

  void setup(OperatorSpec& spec) override {
    spec.output<gxf::Entity>("output");
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
        nvidia::gxf::Shape({N, C}), nvidia::gxf::MemoryStorageType::kSystem, allocator.value());
    // copy the data to the tensor
    std::memcpy(tensor->pointer(), data.data(), N * C * sizeof(float));
  }

  /**
   * @brief Compute method that generates data for each frame
   *
   * This method is called on each execution and:
   * 1. Calculates the current frame rate
   * 2. Creates a new entity with a tensor containing placeholder data
   * 3. Creates an InputSpec that tells Holoviz to render text with the frame number and rate
   * 4. Emits both the entity and the specs to the visualizer
   */
  void compute([[maybe_unused]] InputContext& op_input, [[maybe_unused]] OutputContext& op_output,
               [[maybe_unused]] ExecutionContext& context) override {
    // Calculate frame rate
    float frame_rate = 0.F;
    if (start_time_.time_since_epoch().count() == 0) {
      // First frame: record the start time
      start_time_ = std::chrono::steady_clock::now();
    } else {
      // Calculate frames per second
      frame_rate = static_cast<float>(frame_index_) /
                   static_cast<float>(std::chrono::duration_cast<std::chrono::milliseconds>(
                                          std::chrono::steady_clock::now() - start_time_)
                                          .count()) *
                   1000.F;
    }

    // Create a new entity to hold the tensor data
    auto entity = gxf::Entity::New(&context);

    // Add a dummy tensor (the actual text content comes from the InputSpec)
    add_data<2, 2>(entity, "dynamic_text", {{{0.F, 0.F}}}, context);
    op_output.emit(entity, "output");

    // Create the input specification that describes how to visualize the data
    std::vector<HolovizOp::InputSpec> specs;
    HolovizOp::InputSpec spec;
    spec.tensor_name_ = "dynamic_text";
    spec.type_ = HolovizOp::InputType::TEXT;
    // The text to display, showing frame number and frame rate
    spec.text_.push_back(fmt::format("Frame {} rate {}", frame_index_, frame_rate));
    specs.push_back(spec);
    op_output.emit(specs, "output_specs");

    // Increment frame counter for next iteration
    ++frame_index_;
  }

 private:
  std::shared_ptr<UnboundedAllocator> allocator_;     ///< Allocator for tensor memory
  uint64_t frame_index_ = 0;                          ///< Current frame number
  std::chrono::steady_clock::time_point start_time_;  ///< Time when first frame was rendered
};

}  // namespace holoscan::ops

/**
 * @brief Enum to specify which condition type to use for synchronization
 */
enum class ConditionType {
  FIRST_PIXEL_OUT,  ///< Use FirstPixelOutCondition to wait for first pixel out signal
  PRESENT_DONE      ///< Use PresentDoneCondition to wait for presentation completion
};

/**
 * @brief Application demonstrating the use of display synchronization conditions with Holoviz
 *
 * This example shows how to use either FirstPixelOutCondition or PresentDoneCondition to
 * synchronize an operator's execution with the display. This ensures that the source operator runs
 * in sync with the display refresh rate, avoiding generating frames faster than they can be
 * displayed.
 *
 * The application creates a pipeline with:
 * - A source operator that generates text data
 * - A visualizer (HolovizOp) that renders the text
 * - A condition (FirstPixelOut or PresentDone) that signals when to generate the next frame
 */
class HolovizConditionsApp : public holoscan::Application {
 public:
  /**
   * @brief Construct a new HolovizConditionsApp object
   *
   * @param count Limits the number of frames to show before the application ends.
   *   Set to -1 by default. Any positive integer will limit on the number of frames displayed.
   * @param condition_type The type of condition to use for synchronization (FirstPixelOut or
   * PresentDone)
   * @param vsync Enable or disable vsync for the HolovizOp operator
   */
  explicit HolovizConditionsApp(int64_t count, ConditionType condition_type, bool vsync)
      : count_(count), condition_type_(condition_type), vsync_(vsync) {}

  /**
   * @brief Compose the application by creating and connecting operators
   *
   * This method sets up the data flow pipeline:
   * 1. Creates a HolovizOp for visualization
   * 2. Creates a SourceOp with two conditions:
   *    - CountCondition: Limits the total number of frames
   *    - FirstPixelOutCondition or PresentDoneCondition: Synchronizes with display
   * 3. Connects the source outputs to the visualizer inputs
   *
   * The choice between FirstPixelOutCondition and PresentDoneCondition is determined by
   * the condition_type_ member variable set during construction.
   */
  void compose() override {
    using namespace holoscan;

    // Create the visualizer operator with vsync setting
    auto visualizer = make_operator<ops::HolovizOp>("holoviz", Arg("vsync") = vsync_);

    std::shared_ptr<Condition> condition;
    if (condition_type_ == ::ConditionType::FIRST_PIXEL_OUT) {
      condition = make_condition<FirstPixelOutCondition>("first_pixel_out_limiter", visualizer);
    } else {
      condition = make_condition<PresentDoneCondition>("present_done_limiter", visualizer);
    }

    // Create the source operator with conditions based on the selected type
    auto source = make_operator<ops::SourceOp>("source", condition);
    // Limit the total number of frames (if count_ is positive)
    if (count_ >= 0) {
      source->add_arg(make_condition<CountCondition>("frame_limit", count_));
    }

    // Connect source outputs to visualizer inputs
    // - "output" (entity with tensor) -> "receivers" (input port of HolovizOp)
    // - "output_specs" (visualization specs) -> "input_specs" (specs port of HolovizOp)
    add_flow(source, visualizer, {{"output", "receivers"}, {"output_specs", "input_specs"}});
  }

 private:
  int64_t count_ = -1;            ///< Maximum number of frames to display (-1 means unlimited)
  ConditionType condition_type_;  ///< Type of condition to use for display synchronization
  bool vsync_ = false;            ///< Enable or disable vsync for the visualizer
};

/**
 * @brief Main entry point for the holoviz conditions example
 *
 * Parses command-line arguments to optionally limit the number of frames, select
 * the condition type, and enable vsync. Creates the application and runs it.
 *
 * @param argc Number of command-line arguments
 * @param argv Array of command-line argument strings
 * @return EXIT_SUCCESS on successful completion
 */
int main(int argc, char** argv) {
  // Parse command-line arguments
  // NOLINTBEGIN(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays)
  struct option long_options[] = {{"help", no_argument, nullptr, 'h'},
                                  {"count", required_argument, nullptr, 'c'},
                                  {"type", required_argument, nullptr, 't'},
                                  {"vsync", no_argument, nullptr, 'v'},
                                  {nullptr, 0, nullptr, 0}};
  // NOLINTEND(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays)
  int64_t count = -1;                                          // Default: unlimited frames
  ConditionType condition_type = ConditionType::PRESENT_DONE;  // Default: PresentDoneCondition
  bool vsync = false;                                          // Default: vsync disabled
  // Process command-line options
  while (true) {
    int option_index = 0;
    // NOLINTBEGIN(concurrency-mt-unsafe)
    const int c =
        getopt_long(argc, argv, "hc:t:v", static_cast<option*>(long_options), &option_index);
    // NOLINTEND(concurrency-mt-unsafe)

    if (c == -1) {
      break;  // No more options to process
    }

    const std::string argument(optarg != nullptr ? optarg : "");
    switch (c) {
      case 'h':
      case '?':
        // Display help message
        std::cout
            << "Usage: " << argv[0] << " [options]" << std::endl
            << "Options:" << std::endl
            << "  -h, --help        display this information" << std::endl
            << "  -c, --count       limits the number of frames to show before the application "
               "ends. Set to -1 by default (unlimited). Any positive integer will limit the "
               "number of frames displayed."
            << std::endl
            << "  -t, --type        condition type to use: 'first_pixel_out' or 'present_done' "
               "(default: present_done)"
            << std::endl
            << "  -v, --vsync       enable vsync for the visualizer (default: disabled)"
            << std::endl
            << std::endl
            << "Condition types:" << std::endl
            << "  first_pixel_out   FirstPixelOutCondition - waits for first pixel out signal"
            << std::endl
            << "  present_done      PresentDoneCondition - waits for presentation completion"
            << std::endl;
        return EXIT_SUCCESS;

      case 'c':
        // Parse the frame count limit
        count = std::stoll(argument);
        break;

      case 't':
        // Parse the condition type
        if (argument == "first_pixel_out") {
          condition_type = ConditionType::FIRST_PIXEL_OUT;
        } else if (argument == "present_done") {
          condition_type = ConditionType::PRESENT_DONE;
        } else {
          std::cerr << "Error: Invalid condition type '" << argument << "'. "
                    << "Valid options are: 'first_pixel_out', 'present_done'" << std::endl;
          return EXIT_FAILURE;
        }
        break;

      case 'v':
        // Enable vsync
        vsync = true;
        break;

      default:
        throw std::runtime_error(fmt::format("Unhandled option `{}`", static_cast<char>(c)));
    }
  }

  // Create and run the application
  auto app = holoscan::make_application<HolovizConditionsApp>(count, condition_type, vsync);
  app->run();

  return 0;
}
