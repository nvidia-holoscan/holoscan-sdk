/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cuda_runtime.h>

#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include <gxf/std/tensor.hpp>
#include <holoscan/holoscan.hpp>
#include <holoscan/operators/bayer_demosaic/bayer_demosaic.hpp>
#include <holoscan/operators/holoviz/holoviz.hpp>
#include <holoscan/operators/raw_image_processor/raw_image_processor.hpp>
#include <holoscan/operators/v4l2_video_capture/v4l2_video_capture.hpp>

namespace holoscan::ops {

/**
 * Reinterprets a raw Bayer blob from V4L2 as a uint16 tensor shaped (height, width, 1).
 *
 * The V4L2 operator with pass_through=true emits a 1-D uint8 memory blob for raw Bayer
 * formats (RG10, RG12, …).  This operator:
 *   1. Reads the actual capture dimensions and pixel format from V4L2 metadata.
 *   2. Reinterprets the blob as 16-bit samples (2 bytes per Bayer pixel).
 *   3. Crops away any row-stride padding so the output shape is exactly
 *      (height, width, 1) with element type uint16.
 *
 * Requires pass_through=true on the upstream V4L2VideoCaptureOp.
 */
class RawFrameConverterOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(RawFrameConverterOp)

  RawFrameConverterOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.input<holoscan::gxf::Entity>("input");
    spec.output<holoscan::gxf::Entity>("output");
    spec.param(allocator_, "allocator", "Allocator", "Memory allocator for output tensors.");
  }

  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override {
    auto in_entity = op_input.receive<holoscan::gxf::Entity>("input").value();

    // Read V4L2 metadata propagated by the upstream capture operator.
    const auto pixel_format = metadata()->get<std::string>("V4L2_pixel_format", "");
    const auto width = metadata()->get<int32_t>("V4L2_width", 0);
    const auto height = metadata()->get<int32_t>("V4L2_height", 0);

    if (pixel_format.empty() || width == 0 || height == 0) {
      throw std::runtime_error(
          "RawFrameConverterOp: missing V4L2 metadata (pixel_format/width/height). "
          "Ensure pass_through=true on the V4L2 capture operator.");
    }

    // V4L2 with pass_through emits a single unnamed uint8 tensor (1-D byte blob).
    auto maybe_in_tensor = static_cast<nvidia::gxf::Entity&>(in_entity).get<nvidia::gxf::Tensor>();
    if (!maybe_in_tensor) {
      throw std::runtime_error("RawFrameConverterOp: no tensor found in input entity.");
    }
    auto& in_tensor = maybe_in_tensor.value();

    // element_count() == total bytes for a uint8 1-D tensor.
    const size_t total_bytes = in_tensor->element_count();
    // Each Bayer pixel occupies 2 bytes (uint16).  The driver may pad rows to a larger stride.
    const size_t stride_u16 = (total_bytes / sizeof(uint16_t)) / static_cast<size_t>(height);

    // Create output entity and allocate a (height, width, 1) uint16 tensor.
    auto out_entity_gxf = nvidia::gxf::Entity::New(context.context()).value();
    auto out_tensor = out_entity_gxf.add<nvidia::gxf::Tensor>(in_tensor.name()).value();

    auto allocator_handle = nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(
                                context.context(), allocator_->gxf_cid())
                                .value();

    out_tensor->reshape<uint16_t>(nvidia::gxf::Shape({height, width, 1}),
                                  nvidia::gxf::MemoryStorageType::kDevice,
                                  allocator_handle);

    // Copy rows from the (possibly wider) V4L2 buffer, cropping stride padding to active width.
    const cudaError_t err =
        cudaMemcpy2D(out_tensor->pointer(),
                     static_cast<size_t>(width) * sizeof(uint16_t),  // dst pitch (bytes per row)
                     in_tensor->pointer(),
                     stride_u16 * sizeof(uint16_t),                  // src pitch (bytes per row)
                     static_cast<size_t>(width) * sizeof(uint16_t),  // copy width in bytes
                     static_cast<size_t>(height),                    // number of rows
                     cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) {
      throw std::runtime_error(std::string("RawFrameConverterOp: cudaMemcpy2D failed: ") +
                               cudaGetErrorString(err));
    }

    auto out_entity = holoscan::gxf::Entity(std::move(out_entity_gxf));
    op_output.emit(out_entity, "output");
  }

 private:
  Parameter<std::shared_ptr<Allocator>> allocator_;
};

}  // namespace holoscan::ops

class App : public holoscan::Application {
 public:
  explicit App(int64_t frame_limit = 0, std::string pixel_format = "RG10", int bayer_grid = -1,
               int raw_depth = -1, bool headless = false, bool fullscreen = false,
               bool use_exclusive_display = false)
      : frame_limit_(frame_limit),
        pixel_format_(std::move(pixel_format)),
        bayer_grid_(bayer_grid),
        raw_depth_(raw_depth),
        headless_(headless),
        fullscreen_(fullscreen),
        use_exclusive_display_(use_exclusive_display) {}

  void compose() override {
    using namespace holoscan;

    // Resolve raw_image_processor config: CLI > YAML > default.
    // bayer_grid: NppiBayerGridPosition (0=BGGR, 1=RGGB, 2=GRBG, 3=GBRG). Default: 1 (RGGB/RG10)
    // raw_depth:  holoscan::csi::PixelFormat (1=RAW_10, 2=RAW_12).        Default: 1 (RAW_10)
    // optical_black: sensor black level offset.                           Default: 50
    int bayer_grid = bayer_grid_;
    int raw_depth = raw_depth_;
    int optical_black = 50;

    // Check YAML if values were not set in CLI
    for (const auto& yaml_node : config().yaml_nodes()) {
      const auto& ip = yaml_node["raw_image_processor"];
      if (bayer_grid < 0)
        bayer_grid = ip["bayer_grid"].as<int>(-1);
      if (raw_depth < 0)
        raw_depth = ip["raw_depth"].as<int>(-1);
      optical_black = ip["optical_black"].as<int>(optical_black);
    }

    // Set to defaults if values were not specified in CLI nor YAML
    if (bayer_grid < 0)
      bayer_grid = 1;
    if (raw_depth < 0)
      raw_depth = 1;

    // V4L2 capture with pass-through so raw Bayer data is forwarded as-is.
    // pixel_format_ is passed last so a CLI --pixel-format overrides the YAML default.
    auto receiver =
        (frame_limit_ > 0)
            ? make_operator<ops::V4L2VideoCaptureOp>(
                  "receiver",
                  make_condition<CountCondition>("frame-limit", frame_limit_),
                  Arg("pass_through", true),
                  from_config("receiver"),
                  Arg("pixel_format", pixel_format_))
            : make_operator<ops::V4L2VideoCaptureOp>("receiver",
                                                     Arg("pass_through", true),
                                                     from_config("receiver"),
                                                     Arg("pixel_format", pixel_format_));

    // Reinterpret the raw uint8 blob as a (height, width, 1) uint16 Bayer tensor.
    auto raw_frame_converter = make_operator<ops::RawFrameConverterOp>(
        "raw_frame_converter",
        Arg("allocator") = make_resource<UnboundedAllocator>("raw_frame_allocator"));

    // Optical black correction and white-balance preprocessing.
    auto raw_image_processor =
        make_operator<ops::RawImageProcessorOp>("raw_image_processor",
                                                Arg("pixel_format", raw_depth),
                                                Arg("bayer_format", bayer_grid),
                                                Arg("optical_black", optical_black));

    // Pool sized for the maximum expected sensor resolution (4K, RGBA, 16-bit).
    constexpr int64_t kMaxWidth = 3840;
    constexpr int64_t kMaxHeight = 2160;
    constexpr int64_t kRgbaChannels = 4;
    constexpr int64_t kBytesPerUint16 = 2;
    auto bayer_pool = make_resource<BlockMemoryPool>(
        "pool",
        Arg("storage_type", 1),  // device memory
        Arg("block_size", kMaxWidth * kMaxHeight * kRgbaChannels * kBytesPerUint16),
        Arg("num_blocks", 4));

    // Bayer demosaic — generate RGBA output.
    auto demosaic = make_operator<ops::BayerDemosaicOp>("demosaic",
                                                        Arg("pool") = bayer_pool,
                                                        Arg("generate_alpha", true),
                                                        Arg("alpha_value", 65535),
                                                        Arg("bayer_grid_pos", bayer_grid),
                                                        Arg("interpolation_mode", 0));

    auto visualizer =
        make_operator<ops::HolovizOp>("holoviz",
                                      Arg("headless", headless_),
                                      Arg("fullscreen", fullscreen_),
                                      Arg("use_exclusive_display", use_exclusive_display_),
                                      from_config("holoviz"));

    add_flow(receiver, raw_frame_converter, {{"signal", "input"}});
    add_flow(raw_frame_converter, raw_image_processor, {{"output", "input"}});
    add_flow(raw_image_processor, demosaic, {{"output", "receiver"}});
    add_flow(demosaic, visualizer, {{"transmitter", "receivers"}});
  }

 private:
  int64_t frame_limit_ = 0;
  std::string pixel_format_ = "RG10";
  int bayer_grid_ = -1;
  int raw_depth_ = -1;
  bool headless_ = false;
  bool fullscreen_ = false;
  bool use_exclusive_display_ = false;
};

int main(int argc, char** argv) {
  // Parse arguments: [config_path] [--frame-limit N] [--pixel-format FMT]
  //                  [--bayer-grid N] [--raw-depth N]
  //                  [--headless] [--fullscreen] [--use-exclusive-display]
  std::string config_path_str;
  int64_t frame_limit = 0;
  std::string pixel_format = "RG10";  // 'RG10' (10-bit Bayer RGRG/GBGB)
  int bayer_grid = -1;                // -1 = not set; resolved from YAML or default in compose()
  int raw_depth = -1;                 // -1 = not set; resolved from YAML or default in compose()
  bool headless = false;
  bool fullscreen = false;
  bool use_exclusive_display = false;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--frame-limit" && i + 1 < argc) {
      frame_limit = std::stoll(argv[++i]);
    } else if (arg == "--pixel-format" && i + 1 < argc) {
      pixel_format = argv[++i];
    } else if (arg == "--bayer-grid" && i + 1 < argc) {
      bayer_grid = std::stoi(argv[++i]);
    } else if (arg == "--raw-depth" && i + 1 < argc) {
      raw_depth = std::stoi(argv[++i]);
    } else if (arg == "--headless") {
      headless = true;
    } else if (arg == "--fullscreen") {
      fullscreen = true;
    } else if (arg == "--use-exclusive-display") {
      use_exclusive_display = true;
    } else if (config_path_str.empty()) {
      config_path_str = arg;
    }
  }

  App app(frame_limit,
          pixel_format,
          bayer_grid,
          raw_depth,
          headless,
          fullscreen,
          use_exclusive_display);

  auto config_path = std::filesystem::canonical(argv[0]).parent_path();
  config_path /= "v4l2_imx274_player.yaml";
  if (!config_path_str.empty()) {
    config_path = config_path_str;
  }

  app.config(config_path);
  app.run();

  HOLOSCAN_LOG_INFO("Application has finished running.");

  return 0;
}
