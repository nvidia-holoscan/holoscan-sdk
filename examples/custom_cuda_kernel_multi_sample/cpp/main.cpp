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
    auto format_converter = make_operator<ops::FormatConverterOp>("format_converter",
                                                                  from_config("format_converter"),
                                                                  Arg("in_dtype") = in_dtype,
                                                                  Arg("pool") = pool_resource);

    auto format_converter2 = make_operator<ops::FormatConverterOp>("format_converter2",
                                                                   from_config("format_converter2"),
                                                                   Arg("in_dtype") = in_dtype,
                                                                   Arg("pool") = pool_resource);

    auto processor = make_operator<ops::InferenceProcessorOp>(
        "processor", from_config("processor"), Arg("allocator") = pool_resource);

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
};

int main(int argc, char** argv) {
  auto app = holoscan::make_application<App>();

  auto config_path = std::filesystem::canonical(argv[0]).parent_path();
  config_path += "/custom_cuda_kernel_multi_sample.yaml";
  if (argc >= 2) { config_path = argv[1]; }

  app->config(config_path);

  app->run();

  return 0;
}
