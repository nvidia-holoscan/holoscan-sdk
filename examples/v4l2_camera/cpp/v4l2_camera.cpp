/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <memory>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <vector>

#include <holoscan/holoscan.hpp>
#include <holoscan/operators/format_converter/format_converter.hpp>
#include <holoscan/operators/holoviz/holoviz.hpp>
#include <holoscan/operators/v4l2_video_capture/v4l2_video_capture.hpp>
#include <gxf/rmm/rmm_allocator.hpp>

#include <v4l2_camera_passthrough_op.hpp>

static bool key_exists(const holoscan::ArgList& config, const std::string& key) {
  return (std::find_if(config.begin(), config.end(), [&key](const holoscan::Arg& arg) {
            return arg.name() == key;
          }) != config.end());
}

class App : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;

    auto cuda_stream_pool = make_resource<CudaStreamPool>("cuda_stream", 0, 0, 0, 1, 1);

    auto source_args = from_config("source");
    auto source =
        make_operator<ops::V4L2VideoCaptureOp>("source", Arg("pass_through", true), source_args);

    // Create allocator for format converter
    auto pool = make_resource<RMMAllocator>("rmm_allocator", from_config("rmm_allocator"));

    // Create format converter for YUYV to RGB conversion
    auto format_converter =
        make_operator<ops::FormatConverterOp>("format_converter",
                                              Arg("in_dtype") = std::string("yuyv"),
                                              Arg("out_dtype") = std::string("rgb888"),
                                              Arg("pool") = pool,
                                              from_config("format_converter"));

    auto passthrough = make_operator<ops::V4L2CameraPassthroughOp>("passthrough");

    auto viz_args = from_config("visualizer");
    if (key_exists(source_args, "width") && key_exists(source_args, "height")) {
      // Set Holoviz width and height from source resolution
      for (auto& arg : source_args) {
        if (arg.name() == "width" || arg.name() == "height") {
          viz_args.add(arg);
        }
      }
    }

    auto visualizer = make_operator<ops::HolovizOp>(
        "visualizer", viz_args, Arg("cuda_stream_pool", cuda_stream_pool));

    // 1. Flow for YUYV format, not supported directly by Holoviz in display drivers >=R550
    // source -> format_converter -> passthrough -> visualizer
    add_flow(source, format_converter, {{"signal", "source_video"}});
    add_flow(format_converter, passthrough, {{"tensor", "input"}});

    // 2. Flow for other VideoBuffer formats (NV12, RGB24, etc.) directly compatible with Holoviz
    // source -> passthrough -> visualizer
    add_flow(source, passthrough, {{"signal", "input"}});

    // Use a basic passthrough operator to circumvent the Holoviz "receivers" multi-port.
    // Directly connecting multiple operators to the visualizer "receivers" multi-port
    // creates multiple receiver ports and causes a deadlock.
    add_flow(passthrough, visualizer, {{"output", "receivers"}});

    // Define conditional routing based on pixel format metadata
    set_dynamic_flows(source, [format_converter, passthrough](const std::shared_ptr<Operator>& op) {
      std::string pixel_format = op->metadata()->get<std::string>("V4L2_pixel_format", "");

      // Route based on pixel format
      if (!pixel_format.empty() && (pixel_format.find("YUYV") != std::string::npos ||
                                    pixel_format.find("yuyv") != std::string::npos)) {
        op->add_dynamic_flow(format_converter);
      } else {
        op->add_dynamic_flow(passthrough);
      }
    });
  }
};

int main(int argc, char** argv) {
  App app;

  // Get the configuration
  auto config_path = std::filesystem::canonical(argv[0]).parent_path();
  config_path += "/v4l2_camera.yaml";
  if (argc >= 2) {
    config_path = argv[1];
  }

  app.config(config_path);
  app.run();

  HOLOSCAN_LOG_INFO("Application has finished running.");

  return 0;
}
