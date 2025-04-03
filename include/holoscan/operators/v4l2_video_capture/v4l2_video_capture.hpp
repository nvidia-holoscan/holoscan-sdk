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

#ifndef INCLUDE_HOLOSCAN_OPERATORS_V4L2_VIDEO_CAPTURE_V4L2_VIDEO_CAPTURE_HPP
#define INCLUDE_HOLOSCAN_OPERATORS_V4L2_VIDEO_CAPTURE_V4L2_VIDEO_CAPTURE_HPP

#include <linux/videodev2.h>

#include <list>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "holoscan/core/operator.hpp"

namespace holoscan::ops {

class GxfFormat;

/**
 * @brief Operator class to get a video stream from a V4L2 source.
 *
 * https://www.kernel.org/doc/html/latest/userspace-api/media/v4l/v4l2.html
 *
 * Inputs a video stream from a V4L2 node, including USB cameras and HDMI IN.
 * If no pixel format is specified in the yaml configuration file, the pixel format will be
 * automatically selected.
 * If a pixel format is specified in the yaml file, then this format will be used.
 *
 * ==Named Outputs==
 *
 * - **signal** : `nvidia::gxf::VideoBuffer` or `nvidia::gxf::Tensor`
 *   - A message containing a video buffer if the V4L2 pixel format has equivalent
 *     `nvidia::gxf::VideoFormat`, else a tensor.
 *
 * ==Parameters==
 *
 * - **allocator**: Deprecated, do not use.
 * - **device**: The device to target (e.g. "/dev/video0" for device 0).
 *   Default value is `"/dev/video0"`.
 * - **width**: Width of the video stream. If set to `0`, use the default width of the device.
 *   Optional (default: `0`).
 * - **height**: Height of the video stream. If set to `0`, use the default height of the device.
 *   Optional (default: `0`).
 * - **frame_rate**: Frame rate of the video stream. If the device does not support the exact frame
 *   rate, the nearest match is used instead. If set to `0.0`, use the default width of the device.
 *   Optional (default: `0.0`).
 * - **num_buffers**: Number of V4L2 buffers to use. Optional (default: `4`).
 * - **pixel_format**: Video stream pixel format (little endian four character code (fourcc)).
 *   Default value is `"auto"`.
 * - **pass_through**: Deprecated, do not use.
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
 * ==Metadata==
 *
 * - **V4L2_pixel_format** : std::string
 *   - V4L2 pixel format
 * - **V4L2_ycbcr_encoding** : std::string
 *   - V4L2 YCbCr encoding (`enum v4l2_ycbcr_encoding` value as string)
 * - **V4L2_quantization** : std::string
 *   - V4L2 quantization (`enum v4l2_quantization` value as string)
 *
 * ==Device Memory Requirements==
 *
 * Deprecated, only needed if `pass_through` is `false`.
 * When using this operator with a `BlockMemoryPool`, a single device memory block is needed
 * (`storage_type` = 0 (kHostnvidia::gxf::MemoryStorageType::kHost)). The size of this memory block
 * can be determined by rounding the width and height up to the nearest even size and then padding
 * the rows as needed so that the row stride is a multiple of 256 bytes. C++ code to calculate the
 * block size is as follows:
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
  void compute([[maybe_unused]] InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override;
  void stop() override;

  using FormatListItem = std::pair<uint32_t, std::shared_ptr<GxfFormat>>;
  using FormatList = std::list<FormatListItem>;
  typedef void (*ConverterFunc)(const void* in, void* rgba, size_t width, size_t height);

 private:
  Parameter<std::shared_ptr<Allocator>> allocator_;
  Parameter<std::string> device_;
  Parameter<uint32_t> width_;
  Parameter<uint32_t> height_;
  Parameter<float> frame_rate_;
  Parameter<uint32_t> num_buffers_;
  Parameter<std::string> pixel_format_;
  Parameter<bool> pass_through_;
  Parameter<uint32_t> exposure_time_;
  Parameter<uint32_t> gain_;

  void v4l2_initialize();
  void v4l2_request_buffers();
  void v4l2_check_formats();
  void v4l2_get_format();
  void v4l2_set_format();
  bool v4l2_camera_supports_control(int cid, const char* control_name);
  void v4l2_set_camera_control(v4l2_control control, const char* control_name, bool warn);
  void v4l2_set_camera_settings();
  void v4l2_start();
  void v4l2_read_buffer(v4l2_buffer& buf);

  /**
   * Output memory type, the operator automatically selects the memory type supported by the system
   * which is best suited for CUDA operations.
   **/
  nvidia::gxf::MemoryStorageType memory_storage_type_;
  /**
   * Capture memory method, the operator automatically uses `V4L2_MEMORY_USERPTR` if supported, else
   * `V4L2_MEMORY_MMAP`.
   */
  enum v4l2_memory capture_memory_method_;

  /// capture buffer properties
  struct Buffer {
    void* ptr;
    size_t length;
  };
  /// allocated capture buffers
  std::vector<Buffer> buffers_;
  /// device file descriptor
  int fd_ = -1;

  /// selected format
  v4l2_fmtdesc format_desc_{};
  /// selected stream data format
  v4l2_format format_{};
  /// if set the the device supports setting the frame rate
  bool supports_frame_rate_ = false;
  /// selected frame rate
  uint32_t frame_rate_denominator_use_{1};
  uint32_t frame_rate_numerator_use_{1};

  /// the corresponding entry in the format conversion table translating from the V4L2 pixel format
  /// to the GXF video buffer format.
  FormatList::const_iterator v4l2_to_gxf_format_;

  /// function doing the conversion to RGBA, deprecated.
  ConverterFunc converter_{nullptr};
};

}  // namespace holoscan::ops

#endif /* INCLUDE_HOLOSCAN_OPERATORS_V4L2_VIDEO_CAPTURE_V4L2_VIDEO_CAPTURE_HPP */
