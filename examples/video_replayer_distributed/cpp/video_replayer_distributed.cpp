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

#include <fmt/format.h>

#include <filesystem>
#include <iostream>
#include <string>

#include <holoscan/core/arg.hpp>
#include <holoscan/holoscan.hpp>
#include <holoscan/logger/logger.hpp>
#include <holoscan/operators/holoviz/holoviz.hpp>
#include <holoscan/operators/video_stream_replayer/video_stream_replayer.hpp>

class ReplayerFragment : public holoscan::Fragment {
 public:
  void compose() override {
    using namespace holoscan;

    ArgList args;
    auto* data_directory = std::getenv("HOLOSCAN_INPUT_PATH");  // NOLINT(*)
    if (data_directory != nullptr && data_directory[0] != '\0') {
      auto video_directory = std::filesystem::path(data_directory);
      video_directory /= "racerx";
      args.add(Arg("directory", video_directory.string()));
      HOLOSCAN_LOG_INFO("Using video from {}", video_directory.string());
    }

    auto replayer =
        make_operator<ops::VideoStreamReplayerOp>("replayer", from_config("replayer"), args);
    add_operator(replayer);
  }
};

class VisualizerFragment : public holoscan::Fragment {
 public:
  void compose() override {
    using namespace holoscan;

    auto visualizer = make_operator<ops::HolovizOp>("holoviz", from_config("holoviz"));
    add_operator(visualizer);
  }
};

class HolovizOpAutoClose : public holoscan::ops::HolovizOp {
  // A version of HolovizOp that simulates the behavior of the user closing the window
  // (or pressing Esc) after a fixed number of fromes.
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS_SUPER(HolovizOpAutoClose, HolovizOp)

  HolovizOpAutoClose() = default;

  void compute(holoscan::InputContext& op_input, holoscan::OutputContext& op_output,
               holoscan::ExecutionContext& context) override {
    holoscan::ops::HolovizOp::compute(op_input, op_output, context);
    HOLOSCAN_LOG_INFO("HolovizOpAutoClose: compute called {} times", compute_count_ + 1);
    compute_count_++;
    if (compute_count_ >= 30) {
      // In HolovizOp this code runs only if the display window was closed
      //   if (viz::WindowShouldClose()) { disable_via_window_close(); }
      // Here we instead trigger the behavior after a fixed number of frames
      disable_via_window_close();
    }
  }

  void set_compute_count(uint64_t compute_count) { compute_count_ = compute_count; }

 private:
  uint64_t compute_count_ = 0;
};

class VisualizerFragment2 : public holoscan::Fragment {
 public:
  void compose() override {
    using namespace holoscan;

    auto visualizer =
        make_operator<HolovizOpAutoClose>("holoviz_auto_close", from_config("holoviz"));
    add_operator(visualizer);
  }
};

class DistributedVideoReplayerApp : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;

    auto fragment1 = make_fragment<ReplayerFragment>("fragment1");
    auto fragment2 = make_fragment<VisualizerFragment>("fragment2");

    // Define the workflow: replayer -> holoviz
    add_flow(fragment1, fragment2, {{"replayer.output", "holoviz.receivers"}});

    // Check if the YAML dual_window parameter is set and add a third fragment with
    // a second visualizer in that case
    auto dual_window = from_config("dual_window").as<bool>();
    if (dual_window) {
      auto fragment3 = make_fragment<VisualizerFragment2>("fragment3");
      add_flow(fragment1, fragment3, {{"replayer.output", "holoviz_auto_close.receivers"}});
    }
  }
};

int main([[maybe_unused]] int argc, char** argv) {
  // Get the yaml configuration file
  auto config_path = std::filesystem::canonical(argv[0]).parent_path();
  config_path /= std::filesystem::path("video_replayer_distributed.yaml");

  auto app = holoscan::make_application<DistributedVideoReplayerApp>();
  app->config(config_path);
  app->run();

  // If desired, input/output port mapping can be printed in human readable (YAML) format
  auto& fragment_graph = app->fragment_graph();
  std::cout << "====== APPLICATION PORT MAPPING =======\n";
  std::cout << fmt::format("{}", fragment_graph.port_map_description());
  for (const auto& fragment : fragment_graph.get_nodes()) {
    std::cout << fmt::format("\n\n====== FRAGMENT '{}' PORT MAPPING =======\n", fragment->name());
    std::cout << fmt::format("{}", fragment->graph().port_map_description());
  }

  return 0;
}
