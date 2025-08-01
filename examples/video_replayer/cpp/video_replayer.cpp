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
#include <string>
#include <vector>

#include <holoscan/holoscan.hpp>
#include "holoscan/data_loggers/async_console_logger/async_console_logger.hpp"
#include <holoscan/operators/video_stream_replayer/video_stream_replayer.hpp>
#include <holoscan/operators/holoviz/holoviz.hpp>

class VideoReplayerApp : public holoscan::Application {
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
    // create an allocator supporting both host and device memory pools
    // (The video stream is copied to an intermediate host buffer before being copied to the GPU)
    args.add(Arg("allocator",
                 make_resource<RMMAllocator>("rmm_allocator", from_config("rmm_allocator"))));

    // Define the replayer and holoviz operators and configure using yaml configuration
    auto replayer =
        make_operator<ops::VideoStreamReplayerOp>("replayer", from_config("replayer"), args);
    auto visualizer = make_operator<ops::HolovizOp>("holoviz", from_config("holoviz"));

    // Define the workflow: replayer -> holoviz
    add_flow(replayer, visualizer, {{"output", "receivers"}});

    // Check if the YAML dual_window parameter is set and add a second visualizer in that case
    auto dual_window = from_config("dual_window").as<bool>();
    if (dual_window) {
      auto visualizer2 = make_operator<ops::HolovizOp>("holoviz2", from_config("holoviz"));
      add_flow(replayer, visualizer2, {{"output", "receivers"}});
    }
  }
};

int main(int argc, char** argv) {
  // Get the yaml configuration file
  auto config_path = std::filesystem::canonical(argv[0]).parent_path();
  config_path /= std::filesystem::path("video_replayer.yaml");
  if (argc >= 2) {
    config_path = argv[1];
  }

  auto app = holoscan::make_application<VideoReplayerApp>();
  app->config(config_path);

  if (app->from_config("dual_window").as<bool>()) {
    // use event-based scheduler to allow multiple operators to run in parallel
    app->scheduler(app->make_scheduler<holoscan::EventBasedScheduler>(
        "event-based-scheduler", app->from_config("event_based_scheduler")));
  }

  // enable logging of message contents to console if requested
  auto enable_data_logging = app->from_config("data_logging").as<bool>();
  if (enable_data_logging) {
    // custom text serializer to limit the number of data elements printed for each tensor
    auto text_serializer = app->make_resource<holoscan::data_loggers::SimpleTextSerializer>(
        "simple_text_serializer", app->from_config("simple_text_serializer"));

    // configure the console logger to use the custom text serializer
    auto console_logger = app->make_resource<holoscan::data_loggers::AsyncConsoleLogger>(
        "console_logger",
        holoscan::Arg("serializer", text_serializer),
        app->from_config("basic_console_logger"));

    // add the console logger to the application
    app->add_data_logger(console_logger);
  }

  app->run();

  return 0;
}
