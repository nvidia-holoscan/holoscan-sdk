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

static bool key_exists(const holoscan::ArgList& config, const std::string& key) {
  return (std::find_if(config.begin(), config.end(), [&key](const holoscan::Arg& arg) {
            return arg.name() == key;
          }) != config.end());
}

namespace holoscan::ops {

/**
 * The V4L2VideoCaptureOp outputs a VideoBuffer if the V4L2 pixel format has equivalent
 * nvidia::gxf::VideoFormat enum, see `v4l2_to_gxf_format`. If this is not the case then
 * V4L2VideoCaptureOp outputs a tensor. This operator checks for that tensor and uses the metadata
 * provided by the V4L2VideoCaptureOp and translates it to a HolovizOp::InputSpec so HolovizOp can
 * display the video data. It also sets the YCbCr encoding model and quantization range.
 */
class V4L2FormatTranslateOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(V4L2FormatTranslateOp)

  V4L2FormatTranslateOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.input<std::shared_ptr<holoscan::gxf::Entity>>("input");
    spec.output<std::vector<HolovizOp::InputSpec>>("output_specs");
  }

  void compute(InputContext& op_input, OutputContext& op_output,
               [[maybe_unused]] ExecutionContext& context) override {
    if (!is_metadata_enabled()) {
      throw std::runtime_error("Metadata needs to be enabled for this operator");
    }
    auto entity = op_input.receive<holoscan::gxf::Entity>("input");

    auto maybe_tensor = entity->nvidia::gxf::Entity::get<nvidia::gxf::Tensor>();
    if (maybe_tensor) {
      // use the metadata provided by the V4L2VideoCaptureOp to build the input spec for the
      // HolovizOp
      auto meta = metadata();

      std::vector<HolovizOp::InputSpec> spec;
      auto& video_spec = spec.emplace_back(HolovizOp::InputSpec("", HolovizOp::InputType::COLOR));

      const auto v4l2_pixel_format = meta->get<std::string>("V4L2_pixel_format");
      if (v4l2_pixel_format == "YUYV") {
        video_spec.image_format_ = HolovizOp::ImageFormat::Y8U8Y8V8_422_UNORM;

        // also set the encoding and quantization
        const auto v4l2_ycbcr_encoding = meta->get<std::string>("V4L2_ycbcr_encoding");
        if (v4l2_ycbcr_encoding == "V4L2_YCBCR_ENC_601") {
          video_spec.yuv_model_conversion_ = HolovizOp::YuvModelConversion::YUV_601;
        } else if (v4l2_ycbcr_encoding == "V4L2_YCBCR_ENC_709") {
          video_spec.yuv_model_conversion_ = HolovizOp::YuvModelConversion::YUV_709;
        } else if (v4l2_ycbcr_encoding == "V4L2_YCBCR_ENC_2020") {
          video_spec.yuv_model_conversion_ = HolovizOp::YuvModelConversion::YUV_2020;
        }

        const auto v4l2_quantization = meta->get<std::string>("V4L2_quantization");
        if (v4l2_quantization == "V4L2_QUANTIZATION_FULL_RANGE") {
          video_spec.yuv_range_ = HolovizOp::YuvRange::ITU_FULL;
        } else if (v4l2_ycbcr_encoding == "V4L2_QUANTIZATION_LIM_RANGE") {
          video_spec.yuv_range_ = HolovizOp::YuvRange::ITU_NARROW;
        }
      } else {
        throw std::runtime_error(fmt::format("Unhandled V4L2 pixel format {}", v4l2_pixel_format));
      }

      // don't pass the meta data along to avoid errors when MetadataPolicy is `kRaise`
      meta->clear();

      // emit the output spec
      op_output.emit(spec, "output_specs");
    }
  }
};

}  // namespace holoscan::ops

class App : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;

    auto cuda_stream_pool = make_resource<CudaStreamPool>("cuda_stream", 0, 0, 0, 1, 1);

    auto source_args = from_config("source");
    auto source =
        make_operator<ops::V4L2VideoCaptureOp>("source", Arg("pass_through", true), source_args);

    auto format_translate = make_operator<ops::V4L2FormatTranslateOp>("format_translate");

    auto viz_args = from_config("visualizer");
    if (key_exists(source_args, "width") && key_exists(source_args, "height")) {
      // Set Holoviz width and height from source resolution
      for (auto& arg : source_args) {
        if (arg.name() == "width" || arg.name() == "height") {
          viz_args.add(arg);
        }
      }
    }

    auto pool = make_resource<UnboundedAllocator>("pool");
    auto format_converter = make_operator<ops::FormatConverterOp>(
        "format_converter", from_config("format_converter"), Arg("pool") = pool);

    auto visualizer = make_operator<ops::HolovizOp>(
        "visualizer", viz_args, Arg("cuda_stream_pool", cuda_stream_pool));

    // Flow definition
    add_flow(source, format_converter, {{"signal", "source_video"}});
    add_flow(format_converter, visualizer, {{"", "receivers"}});

    // need metadata so V4L2FormatTranslateOp can translate the format

    // As of Holoscan 3.0, metadata is enabled by default, but if we wanted to explicitly
    // disable it we could call `enable_metadata(false);` here
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
