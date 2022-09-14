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
  void set_source(const std::string& source) {
    if (source == "aja") { is_aja_source_ = true; }
  }
  void set_record(bool record) { do_record_ = record; }

  void compose() override {
    using namespace holoscan;

    std::shared_ptr<Operator> source;
    std::shared_ptr<Operator> recorder;
    std::shared_ptr<Operator> recorder_format_converter;

    bool is_rdma = from_config("aja.rdma").as<bool>();
    uint64_t format_converter_block_size = 0;
    uint64_t format_converter_num_blocks = 0;
    uint64_t visualizer_format_converter_block_size = 0;
    uint64_t visualizer_format_converter_num_blocks = 0;

    if (is_aja_source_) {
      uint32_t width = from_config("aja.width").as<uint32_t>();
      uint32_t height = from_config("aja.height").as<uint32_t>();
      source = make_operator<ops::AJASourceOp>("aja", from_config("aja"));
      if (do_record_) {
        recorder_format_converter = make_operator<ops::FormatConverterOp>(
            "recorder_format_converter",
            from_config("recorder_format_converter"),
            Arg("pool") =
                make_resource<BlockMemoryPool>("pool", 1, width * height * 4 * 4, is_rdma ? 3 : 4));
      }
      format_converter_block_size = width * height * 4 * 4;
      format_converter_num_blocks = is_rdma ? 3 : 4;
      visualizer_format_converter_block_size = width * height * 4 * 4;
      visualizer_format_converter_num_blocks = is_rdma ? 2 : 3;
    } else {
      source = make_operator<ops::VideoStreamReplayerOp>("replayer", from_config("replayer"));
      format_converter_block_size = 854 * 480 * 3 * 4;
      format_converter_num_blocks = 2;
      visualizer_format_converter_block_size = 854 * 480 * 4 * 4;
      visualizer_format_converter_num_blocks = 2;
    }

    if (do_record_) {
      recorder = make_operator<ops::VideoStreamRecorderOp>("recorder", from_config("recorder"));
    }

    auto format_converter = make_operator<ops::FormatConverterOp>(
        "format_converter",
        from_config(is_aja_source_ ? "format_converter_aja" : "format_converter_replayer"),
        Arg("pool") = make_resource<BlockMemoryPool>(
            "pool", 1, format_converter_block_size, format_converter_num_blocks));

    auto lstm_inferer = make_operator<ops::LSTMTensorRTInferenceOp>(
        "lstm_inferer",
        from_config("lstm_inference"),
        Arg("pool") = make_resource<UnboundedAllocator>("pool"),
        Arg("cuda_stream_pool") = make_resource<CudaStreamPool>("cuda_stream", 0, 0, 0, 1, 5));

    auto visualizer_format_converter = make_operator<ops::FormatConverterOp>(
        "visualizer_format_converter",
        from_config(is_aja_source_ ? "visualizer_format_converter_aja"
                                   : "visualizer_format_converter_replayer"),
        Arg("pool") = make_resource<BlockMemoryPool>("pool",
                                                     1,
                                                     visualizer_format_converter_block_size,
                                                     visualizer_format_converter_num_blocks));

    auto visualizer = make_operator<ops::ToolTrackingVizOp>(
        "visualizer",
        from_config("visualizer"),
        Arg("pool") = make_resource<UnboundedAllocator>("pool"));

    // Flow definition
    add_flow(source, visualizer_format_converter);
    add_flow(visualizer_format_converter, visualizer, {{"", "source_video"}});

    add_flow(source, format_converter);
    add_flow(format_converter, lstm_inferer);
    add_flow(lstm_inferer, visualizer, {{"", "tensor"}});

    if (do_record_) {
      if (is_aja_source_) {
        add_flow(source, recorder_format_converter);
        add_flow(recorder_format_converter, recorder);
      } else {
        add_flow(source, recorder);
      }
    }
  }

 private:
  bool is_aja_source_ = false;
  bool do_record_ = false;
};

int main(int argc, char** argv) {
  holoscan::load_env_log_level();

  auto app = holoscan::make_application<App>();

  if (argc == 2) {
    app->config(argv[1]);
  } else {
    app->config("apps/endoscopy_tool_tracking/app_config.yaml");
  }
  auto source = app->from_config("source").as<std::string>();
  app->set_source(source);
  auto do_record = app->from_config("do_record").as<bool>();
  app->set_record(do_record);

  app->run();

  return 0;
}
