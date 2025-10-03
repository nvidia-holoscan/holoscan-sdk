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

#include <getopt.h>

#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <holoscan/holoscan.hpp>
#include <holoscan/operators/format_converter/format_converter.hpp>
#include <holoscan/operators/holoviz/holoviz.hpp>
#include <holoscan/operators/inference_processor/inference_processor.hpp>
#include <holoscan/operators/video_stream_replayer/video_stream_replayer.hpp>

class App : public holoscan::Application {
 public:
  explicit App(bool enable_green_context) : enable_green_context_(enable_green_context) {}

  void compose() override {
    using namespace holoscan;

    // Sets the data directory to use from the environment variable if it is set
    ArgList args;
    auto* data_directory = std::getenv("HOLOSCAN_INPUT_PATH");  // NOLINT(*)
    if (data_directory != nullptr && data_directory[0] != '\0') {
      auto video_directory = std::filesystem::path(data_directory);
      video_directory /= "racerx";
      args.add(Arg("directory", video_directory.string()));
    }

    std::shared_ptr<Operator> source;
    std::shared_ptr<Resource> pool_resource = make_resource<UnboundedAllocator>("pool");
    std::shared_ptr<Resource> pool_resource1 = make_resource<UnboundedAllocator>("pool");

    source = make_operator<ops::VideoStreamReplayerOp>("replayer", from_config("replayer"), args);

    auto in_dtype = std::string("rgb888");
    std::shared_ptr<Operator> format_converter;
    std::shared_ptr<Operator> format_converter2;
    std::shared_ptr<Operator> processor;

    if (enable_green_context_) {
      // Create a global CUDA Green context pool
      // Use the default min_sm_count=2, create partitions with 3 green contexts
      std::vector<uint32_t> partitions = {4, 4, 8};
      auto cuda_green_context_pool = make_resource<CudaGreenContextPool>(
          "cuda_green_context_pool", 0, 0, partitions.size(), partitions);
      auto cuda_green_context1 =
          make_resource<CudaGreenContext>("cuda_green_context", cuda_green_context_pool, 0);
      auto cuda_stream_pool1 =
          make_resource<CudaStreamPool>("cuda_stream_pool", 0, 0, 0, 1, 5, cuda_green_context1);
      auto cuda_green_context2 =
          make_resource<CudaGreenContext>("cuda_green_context", cuda_green_context_pool, 1);
      auto cuda_stream_pool2 =
          make_resource<CudaStreamPool>("cuda_stream_pool", 0, 0, 0, 1, 5, cuda_green_context2);
      auto cuda_green_context3 =
          make_resource<CudaGreenContext>("cuda_green_context", cuda_green_context_pool, 2);
      auto cuda_stream_pool3 =
          make_resource<CudaStreamPool>("cuda_stream_pool", 0, 0, 0, 1, 5, cuda_green_context3);

      format_converter = make_operator<ops::FormatConverterOp>("format_converter",
                                                               from_config("format_converter"),
                                                               Arg("in_dtype") = in_dtype,
                                                               Arg("pool") = pool_resource,
                                                               cuda_stream_pool1);

      format_converter2 = make_operator<ops::FormatConverterOp>("format_converter2",
                                                                from_config("format_converter2"),
                                                                Arg("in_dtype") = in_dtype,
                                                                Arg("pool") = pool_resource,
                                                                cuda_stream_pool2);

      processor = make_operator<ops::InferenceProcessorOp>("processor",
                                                           from_config("processor"),
                                                           Arg("allocator") = pool_resource,
                                                           cuda_stream_pool3);
    } else {
      format_converter = make_operator<ops::FormatConverterOp>("format_converter",
                                                               from_config("format_converter"),
                                                               Arg("in_dtype") = in_dtype,
                                                               Arg("pool") = pool_resource);

      format_converter2 = make_operator<ops::FormatConverterOp>("format_converter2",
                                                                from_config("format_converter2"),
                                                                Arg("in_dtype") = in_dtype,
                                                                Arg("pool") = pool_resource);

      processor = make_operator<ops::InferenceProcessorOp>(
          "processor", from_config("processor"), Arg("allocator") = pool_resource);
    }

    std::vector<ops::HolovizOp::InputSpec> input_object_specs(3);
    input_object_specs[0].tensor_name_ = "input_formatted";
    input_object_specs[0].priority_ = 0;
    input_object_specs[0].type_ = ops::HolovizOp::InputType::COLOR;
    ops::HolovizOp::InputSpec::View object_view;
    object_view.width_ = 0.5;
    object_view.height_ = 1.0;
    input_object_specs[0].views_.push_back(object_view);

    input_object_specs[1].tensor_name_ = "input_processed";
    input_object_specs[1].priority_ = 0;
    input_object_specs[1].type_ = ops::HolovizOp::InputType::COLOR;
    ops::HolovizOp::InputSpec::View grayscale_view;
    grayscale_view.width_ = 0.5;
    grayscale_view.height_ = 1.0;
    grayscale_view.offset_x_ = 0.33;
    input_object_specs[1].views_.push_back(grayscale_view);

    input_object_specs[2].tensor_name_ = "input_processed2";
    input_object_specs[2].priority_ = 0;
    input_object_specs[2].type_ = ops::HolovizOp::InputType::COLOR;
    ops::HolovizOp::InputSpec::View edge_view;
    edge_view.width_ = 0.5;
    edge_view.height_ = 1.0;
    edge_view.offset_x_ = 0.67;
    input_object_specs[2].views_.push_back(edge_view);

    auto holoviz = make_operator<ops::HolovizOp>(
        "holoviz", from_config("holoviz"), Arg("tensors") = input_object_specs);

    // Flow definition

    add_flow(source, format_converter);
    add_flow(source, format_converter2);
    add_flow(format_converter, processor, {{"", "receivers"}});
    add_flow(format_converter2, processor, {{"", "receivers"}});
    add_flow(format_converter, holoviz, {{"", "receivers"}});
    add_flow(processor, holoviz, {{"transmitter", "receivers"}});
  }

 private:
  bool enable_green_context_ = false;
};

int main(int argc, char** argv) {
  auto config_path = std::filesystem::canonical(argv[0]).parent_path();
  config_path += "/custom_cuda_kernel_multi_sample.yaml";
  bool green_context = false;

  // NOLINTBEGIN(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays)
  struct option long_options[] = {{"help", no_argument, nullptr, 'h'},
                                  {"config_path", optional_argument, nullptr, 'c'},
                                  {"green_context", no_argument, nullptr, 'g'},
                                  {nullptr, 0, nullptr, 0}};

  // Parse command line options for config_path and green_context
  while (true) {
    int option_index = 0;
    // NOLINTBEGIN(concurrency-mt-unsafe)
    int c = getopt_long(argc, argv, "hc:g", static_cast<option*>(long_options), &option_index);
    // NOLINTEND(concurrency-mt-unsafe)
    if (c == -1) {
      break;
    }
    const std::string argument(optarg != nullptr ? optarg : "");
    switch (c) {
      case 'h':
      case '?':
        std::cout << "Usage: " << argv[0] << " [options]" << std::endl
                  << "Options:" << std::endl
                  << "  -h, --help           display this information" << std::endl
                  << "  -c, --config_path    path to config yaml file" << std::endl
                  << "  -g, --green_context  flag to enable green context" << std::endl;
        return EXIT_SUCCESS;
      case 'c':
        if (!argument.empty()) {
          config_path = argument;
        }
        break;
      case 'g':
        green_context = true;
        break;
      default:
        throw std::runtime_error(fmt::format("Unhandled option `{}`", static_cast<char>(c)));
    }
  }
  // NOLINTEND(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays)

  auto app = holoscan::make_application<App>(green_context);

  // EventBasedScheduler is required for CUDA context switching to run the application
  // on multiple worker threads
  auto scheduler = app->make_scheduler<holoscan::EventBasedScheduler>(
      "event-based-scheduler", holoscan::Arg("worker_thread_number", 2L));
  app->scheduler(scheduler);

  app->config(config_path);

  app->run();

  return 0;
}
