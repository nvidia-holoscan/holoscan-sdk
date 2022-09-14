/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <holoscan/holoscan.hpp>
#include <holoscan/std_ops.hpp>

class Fragment1 : public holoscan::Fragment {
 public:
  void compose() override {
    using namespace holoscan;
    auto replayer = make_operator<ops::VideoStreamReplayerOp>("replayer", from_config("replayer"));

    auto recorder = make_operator<ops::VideoStreamRecorderOp>("recorder", from_config("recorder"));

    auto format_converter = make_operator<ops::FormatConverterOp>(
        "format_converter",
        from_config("format_converter_replayer"),
        Arg("pool") = make_resource<BlockMemoryPool>("pool", 1, 4919041, 2));

    auto lstm_inferer = make_operator<ops::LSTMTensorRTInferenceOp>(
        "lstm_inferer",
        from_config("lstm_inference"),
        Arg("pool") = make_resource<UnboundedAllocator>("pool"),
        Arg("cuda_stream_pool") = make_resource<CudaStreamPool>("cuda_stream", 0, 0, 0, 1, 5));

    auto visualizer_format_converter = make_operator<ops::FormatConverterOp>(
        "visualizer_format_converter",
        from_config("visualizer_format_converter_replayer"),
        Arg("pool") = make_resource<BlockMemoryPool>("pool", 1, 6558720, 2));

    // Flow definition
    add_flow(replayer, format_converter);
    add_flow(format_converter, lstm_inferer);
    add_flow(replayer, visualizer_format_converter);
    add_flow(replayer, recorder);
  }
};

class Fragment2 : public holoscan::Fragment {
 public:
  void compose() override {
    using namespace holoscan;
    auto video_composer =
        make_operator<ops::VideoComposerOp>("video_composer", from_config("video_composer"));

    auto video_recorder =
        make_operator<ops::VideoStreamRecorderOp>("video_recorder", from_config("video_recorder"));

    // Flow definition
    add_flow(video_composer, video_recorder, {{"video", ""}});
  }
};

class App : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;

    auto fragment1 = make_fragment<Fragment1>("fragment1");
    auto fragment2 = make_fragment<Fragment2>("fragment2");

    add_flow(fragment1, fragment2, {{"lstm_inferer.out_tensor", "video_composer.tensor"}});
    add_flow(fragment1, fragment2, {{"visualizer_format_converter.source_video", "video_composer.source_video"}});
  }
};

int main() {
  App app;
  app.config("apps/endoscopy_tool_tracking/app_config.yaml");
  app.run();
  return 0;
}
