/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <vector>

#include "../config.hpp"
#include "tensor_compare_op.hpp"

#include <holoscan/holoscan.hpp>
#include <holoscan/operators/video_stream_replayer/video_stream_replayer.hpp>
#include <holoscan/operators/holoviz/holoviz.hpp>

using namespace holoscan;

static HoloscanTestConfig test_config;

class VideoReplayerApp : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;

    // Sets the data directory to use from the environment variable if it is set
    ArgList args;
    auto data_directory = std::getenv("HOLOSCAN_INPUT_PATH");
    if (data_directory != nullptr && data_directory[0] != '\0') {
      auto video_directory = std::filesystem::path(data_directory);
      video_directory /= "racerx";
      args.add(Arg("directory", video_directory.string()));
    }

    if (use_allocator_args_ || use_entity_serializer_arg_) {
      if (use_entity_serializer_arg_) {
        auto entity_serializer =
            make_resource<holoscan::StdEntitySerializer>("video_entity_serializer");
        args.add(Arg("entity_serializer", entity_serializer));
      } else {
        // the video data has a header that indicates device memory
        args.add(Arg("allocator", make_resource<UnboundedAllocator>("video_replayer_allocator")));
      }
    }

    // Define the replayer and holoviz operators and configure using yaml configuration
    auto replayer =
        make_operator<ops::VideoStreamReplayerOp>("replayer", from_config("replayer"), args);
    auto visualizer = make_operator<ops::HolovizOp>("holoviz", from_config("holoviz"));

    // Define the workflow: replayer -> holoviz
    add_flow(replayer, visualizer, {{"output", "receivers"}});
  }

  void set_use_entity_serializer(bool use_entity_serializer_arg) {
    use_entity_serializer_arg_ = use_entity_serializer_arg;
  }
  void set_use_allocators(bool use_allocator_args) { use_allocator_args_ = use_allocator_args; }

 private:
  bool use_entity_serializer_arg_ = false;
  bool use_allocator_args_ = false;
};

void run_app(bool use_allocator_args = false, bool use_entity_serializer_arg = false,
             const std::string& failure_str = "") {
  auto app = make_application<VideoReplayerApp>();

  const std::string config_file = test_config.get_test_data_file("video_replayer_apps.yaml");
  app->config(config_file);

  app->set_use_allocators(use_allocator_args);
  app->set_use_entity_serializer(use_entity_serializer_arg);

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
    EXPECT_TRUE(log_output.find("error") == std::string::npos) << log_output;
  } else {
    EXPECT_TRUE(log_output.find(failure_str) != std::string::npos) << log_output;
  }
}

// run app without providing entity_serializer or allocaor argument
TEST(VideoStreamReplayerApps, TestDefaultEntitySerializer) {
  run_app(false, false);
}

// run app providing allocator argument
TEST(VideoStreamReplayerApps, TestUserProvidedAllocator) {
  run_app(true, false);
}

// run app providing entity_serializer argument
TEST(VideoStreamReplayerApps, TestUserProvidedEntitySerializer) {
  run_app(false, true);
}
