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

class App : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;

    auto replayer = make_operator<ops::VideoStreamReplayerOp>("replayer", from_config("replayer"));

    auto recorder = make_operator<ops::VideoStreamRecorderOp>("recorder", from_config("recorder"));

    auto format_converter = make_operator<ops::FormatConverterOp>(
        "format_converter",
        from_config("format_converter_replayer"),
        Arg("pool") = make_resource<BlockMemoryPool>("pool", 1, 854 * 480 * 3 * 4, 2));

    auto lstm_inferer = make_operator<ops::LSTMTensorRTInferenceOp>(
        "lstm_inferer",
        from_config("lstm_inference"),
        Arg("pool") = make_resource<UnboundedAllocator>("pool"),
        Arg("cuda_stream_pool") = make_resource<CudaStreamPool>("cuda_stream", 0, 0, 0, 1, 5));


    // A known issue using the BlockMemoryPool when converting the ONNX model to TRT from
    // the LSTMTensorRTInferenceOp causes the inference to produce wrong values the first
    // time the conversion is done inline.
    // Switching from BlockMemoryPool to UnboundedAllocator or increasing the block size of the
    // BlockMemoryPool seems to fix the issue.
    // Using TensorRT 8.4.1 and above seems also to be fixing the issue.
    auto tool_tracking_postprocessor = make_operator<ops::ToolTrackingPostprocessorOp>(
        "tool_tracking_postprocessor",
        from_config("tool_tracking_postprocessor"),
        // Arg("device_allocator") =
        //    make_resource<BlockMemoryPool>("device_allocator", 1, 107 * 60 * 7 * 4, 2),
        Arg("device_allocator") = make_resource<UnboundedAllocator>("device_allocator"),
        Arg("host_allocator") = make_resource<UnboundedAllocator>("host_allocator"));

    auto visualizer = make_operator<ops::HolovizOp>("holoviz", from_config("holoviz"));

    // Flow definition
    add_flow(replayer, visualizer, {{"output", "receivers"}});

    add_flow(replayer, format_converter);
    add_flow(format_converter, lstm_inferer);
    add_flow(lstm_inferer, tool_tracking_postprocessor, {{"tensor", "in"}});
    add_flow(tool_tracking_postprocessor, visualizer, {{"out", "receivers"}});

    add_flow(replayer, recorder);
  }
};

int main(int argc, char** argv) {
  App app;

  // Get the configuration
  auto config_path = std::filesystem::canonical(argv[0]).parent_path();
  config_path += "/app_config.yaml";
  app.config(config_path);

  app.run();

  return 0;
}
