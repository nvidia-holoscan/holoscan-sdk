/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <string>
#include <typeinfo>
#include <holoscan/holoscan.hpp>
#include <holoscan/operators/v4l2_video_capture/v4l2_video_capture.hpp>
#include <holoscan/operators/holoviz/holoviz.hpp>

bool key_exists(const holoscan::ArgList& config, const std::string& key) {
  bool exists = false;
  for (auto& arg : config) {
    if (arg.name() == key) {
      exists = true;
      break;
    }
  }
  return exists;
}

class App : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;

    std::shared_ptr<ops::V4L2VideoCaptureOp> source;
    std::shared_ptr<ops::HolovizOp> visualizer;

    if (key_exists(from_config("source"), "width") && key_exists(from_config("source"), "height")) {
      // width and height given, use BlockMemoryPool (better latency)
      const int width = from_config("source.width").as<int>();
      const int height = from_config("source.height").as<int>();
      const int n_channels = 4;
      uint64_t block_size = width * height * n_channels;
      auto allocator = make_resource<BlockMemoryPool>("pool", 0, block_size, 1);

      source = make_operator<ops::V4L2VideoCaptureOp>(
          "source", from_config("source"), Arg("allocator") = allocator);

      // Set Holoviz width and height from source resolution
      auto viz_args = from_config("visualizer");
      for (auto& arg : from_config("source")) {
        if (arg.name() == "width") viz_args.add(arg);
        else if (arg.name() == "height")
          viz_args.add(arg);
      }
      visualizer =
          make_operator<ops::HolovizOp>("visualizer", viz_args, Arg("allocator") = allocator);
    } else {
      // width and height not given, use UnboundedAllocator (worse latency)
      source = make_operator<ops::V4L2VideoCaptureOp>(
          "source",
          from_config("source"),
          Arg("allocator") = make_resource<UnboundedAllocator>("pool"));
      visualizer = make_operator<ops::HolovizOp>("visualizer", from_config("visualizer"));
    }

    // Flow definition
    add_flow(source, visualizer, {{"signal", "receivers"}});
  }
};

int main(int argc, char** argv) {
  App app;

  // Get the configuration
  auto config_path = std::filesystem::canonical(argv[0]).parent_path();
  config_path += "/v4l2_camera.yaml";
  if (argc >= 2) { config_path = argv[1]; }

  app.config(config_path);
  app.run();

  HOLOSCAN_LOG_INFO("Application has finished running.");

  return 0;
}
