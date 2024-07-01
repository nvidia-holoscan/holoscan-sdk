/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef HOLOSCAN_OPERATORS_V4L2_VIDEO_CAPTURE_HPP
#define HOLOSCAN_OPERATORS_V4L2_VIDEO_CAPTURE_HPP

#include <linux/videodev2.h>
#include <memory>
#include <string>

#include "holoscan/core/operator.hpp"

namespace holoscan::ops {

/**
 * @brief Operator class to get the video stream from V4L2.
 *
 * https://www.kernel.org/doc/html/v4.9/media/uapi/v4l/v4l2.html
 *
 * Inputs a video stream from a V4L2 node, including USB cameras and HDMI IN.
 * - Input stream is on host. If no pixel format is specified in the yaml configuration file, the
 *   pixel format will be automatically selected. However, only `AB24`, `YUYV`, and MJPG are then
 *   supported.
 *   If a pixel format is specified in the yaml file, then this format will be used. However, note
 *   that the operator then expects that this format can be encoded as RGBA32. If not, the behavior
 *   is undefined.
 * - Output stream is on host. Always RGBA32 at this time.
 *
 * Use `holoscan::ops::FormatConverterOp` to move data from the host to a GPU device.
 *
 * ==Named Outputs==
 *
 * - **signal** : `nvidia::gxf::VideoBuffer`
 *   - A message containing a video buffer on the host with format
 *     GXF_VIDEO_FORMAT_RGBA.
 *
 * ==Parameters==
 *
 * - **allocator**: Memory allocator to use for the output.
 * - **device**: The device to target (e.g. "/dev/video0" for device 0).
 *   Default value is `"/dev/video0"`.
 * - **width**: Width of the video stream. Optional (default: `0`).
 * - **height**: Height of the video stream. Optional (default: `0`).
 * - **num_buffers**: Number of V4L2 buffers to use. Optional (default: `4`).
 * - **pixel_format**: Video stream pixel format (little endian four character code (fourcc)).
 *   Default value is `"auto"`.
 * - **exposure_time**: Exposure time of the camera sensor in multiples of 100 μs (e.g. setting
 *   exposure_time to 100 is 10 ms). Optional (default: auto exposure, or camera sensor default).
 *   Use `v4l2-ctl -d /dev/<your_device> -L` for a range of values supported by your device.
 *   - When not set by the user, V4L2_CID_EXPOSURE_AUTO is set to V4L2_EXPOSURE_AUTO, or to
 *     V4L2_EXPOSURE_APERTURE_PRIORITY if the former is not supported.
 *   - When set by the user, V4L2_CID_EXPOSURE_AUTO is set to V4L2_EXPOSURE_SHUTTER_PRIORITY, or to
 *     V4L2_EXPOSURE_MANUAL if the former is not supported. The provided value is then used to set
 *     V4L2_CID_EXPOSURE_ABSOLUTE.
 * - **gain**: Gain of the camera sensor. Optional (default: auto gain, or camera sensor default).
 *   Use `v4l2-ctl -d /dev/<your_device> -L` for a range of values supported by your device.
 *   - When not set by the user, V4L2_CID_AUTOGAIN is set to false (if supported).
 *   - When set by the user, V4L2_CID_AUTOGAIN is set to true (if supported). The provided value is
 *     then used to set V4L2_CID_GAIN.
 *
 * ==Device Memory Requirements==
 *
 * When using this operator with a `BlockMemoryPool`, a single device memory block is needed
 * (`storage_type` = 1). The size of this memory block can be determined by rounding the width and
 * height up to the nearest even size and then padding the rows as needed so that the row stride is
 * a multiple of 256 bytes. C++ code to calculate the block size is as follows:
 *
 * ```cpp
 * #include <cstdint>
 *
 * int64_t get_block_size(int32_t height, int32_t width) {
 *   int32_t height_even = height + (height & 1);
 *   int32_t width_even = width + (width & 1);
 *   int64_t row_bytes = width_even * 4;  // 4 bytes per pixel for 8-bit RGBA
 *   int64_t row_stride = (row_bytes % 256 == 0) ? row_bytes : ((row_bytes / 256 + 1) * 256);
 *   return height_even * row_stride;
 * }
 * ```
 */
class V4L2VideoCaptureOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(V4L2VideoCaptureOp)

  V4L2VideoCaptureOp() = default;

  void setup(OperatorSpec& spec) override;
  void start() override;
  void initialize() override;
  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override;
  void stop() override;

 private:
  Parameter<holoscan::IOSpec*> signal_;

  Parameter<std::shared_ptr<Allocator>> allocator_;
  Parameter<std::string> device_;
  Parameter<uint32_t> width_;
  Parameter<uint32_t> height_;
  Parameter<uint32_t> num_buffers_;
  Parameter<std::string> pixel_format_;
  Parameter<uint32_t> exposure_time_;
  Parameter<uint32_t> gain_;

  void v4l2_initialize();
  void v4l2_requestbuffers();
  void v4l2_check_formats();
  void v4l2_set_mode();
  void v4l2_set_formats();
  bool v4l2_camera_supports_control(int cid, const char* control_name);
  void v4l2_set_camera_control(v4l2_control control, const char* control_name, bool warn);
  void v4l2_set_camera_settings();
  void v4l2_start();
  void v4l2_read_buffer(v4l2_buffer& buf);

  void YUYVToRGBA(const void* yuyv, void* rgba, size_t width, size_t height);
  void MJPEGToRGBA(const void* mjpg, void* rgba, size_t width, size_t height);

  struct Buffer {
    void* ptr;
    size_t length;
  };
  Buffer* buffers_ = nullptr;
  int fd_ = -1;

  uint32_t width_use_{0};
  uint32_t height_use_{0};
  uint32_t pixel_format_use_{V4L2_PIX_FMT_RGBA32};
};

}  // namespace holoscan::ops

#endif /* HOLOSCAN_OPERATORS_V4L2_VIDEO_CAPTURE_HPP */
