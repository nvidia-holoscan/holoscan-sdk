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
#ifndef NVIDIA_CLARA_HOLOSCAN_GXF_EXTENSIONS_AJA_SOURCE_HPP_
#define NVIDIA_CLARA_HOLOSCAN_GXF_EXTENSIONS_AJA_SOURCE_HPP_

#include <ajantv2/includes/ntv2card.h>
#include <ajantv2/includes/ntv2devicescanner.h>
#include <ajantv2/includes/ntv2enums.h>

#include <string>
#include <vector>

#include "gxf/std/codelet.hpp"
#include "gxf/std/receiver.hpp"
#include "gxf/std/transmitter.hpp"

namespace nvidia {
namespace holoscan {

constexpr uint32_t kNumBuffers = 2;

constexpr char kDefaultDevice[] = "0";
constexpr NTV2Channel kDefaultChannel = NTV2_CHANNEL1;
constexpr uint32_t kDefaultWidth = 1920;
constexpr uint32_t kDefaultHeight = 1080;
constexpr uint32_t kDefaultFramerate = 60;
constexpr bool kDefaultRDMA = false;
constexpr NTV2PixelFormat kDefaultPixelFormat = NTV2_FBF_ABGR;
constexpr size_t kWarmupFrames = 5;
constexpr bool kDefaultEnableOverlay = false;
constexpr bool kDefaultOverlayRDMA = false;
constexpr NTV2Channel kDefaultOverlayChannel = NTV2_CHANNEL2;

/*
 * TODO
 * - Make pixel format configurable (non-ABGR must be supported downstream)
 */

/// @brief Video input codelet for use with AJA capture cards.
///
/// Provides a codelet for supporting AJA capture card as a source.
/// It offers support for GPUDirect-RDMA on Quadro/NVIDIA RTX GPUs.
/// The output is a VideoBuffer object.
class AJASource : public gxf::Codelet {
 public:
  AJASource();

  gxf_result_t registerInterface(gxf::Registrar* registrar) override;

  gxf_result_t start() override;
  gxf_result_t tick() override;
  gxf_result_t stop() override;

 private:
  AJAStatus DetermineVideoFormat();
  AJAStatus OpenDevice();
  AJAStatus SetupVideo();
  AJAStatus SetupBuffers();
  AJAStatus StartAutoCirculate();
  bool AllocateBuffers(std::vector<void*>& buffers, size_t num_buffers,
                       size_t buffer_size, bool rdma);
  void FreeBuffers(std::vector<void*>& buffers, bool rdma);
  bool GetNTV2VideoFormatTSI(NTV2VideoFormat* format);

  gxf::Parameter<gxf::Handle<gxf::Transmitter>> video_buffer_output_;
  gxf::Parameter<std::string> device_specifier_;
  gxf::Parameter<NTV2Channel> channel_;
  gxf::Parameter<uint32_t> width_;
  gxf::Parameter<uint32_t> height_;
  gxf::Parameter<uint32_t> framerate_;
  gxf::Parameter<bool> use_rdma_;

  gxf::Parameter<bool> enable_overlay_;
  gxf::Parameter<NTV2Channel> overlay_channel_;
  gxf::Parameter<bool> overlay_rdma_;
  gxf::Parameter<gxf::Handle<gxf::Transmitter>> overlay_buffer_output_;
  gxf::Parameter<gxf::Handle<gxf::Receiver>> overlay_buffer_input_;

  CNTV2Card device_;
  NTV2DeviceID device_id_;
  NTV2VideoFormat video_format_;
  NTV2PixelFormat pixel_format_;
  bool is_kona_hdmi_;
  bool use_tsi_;

  std::vector<void*> buffers_;
  std::vector<void*> overlay_buffers_;
  uint8_t current_buffer_;
  uint8_t current_hw_frame_;
  uint8_t current_overlay_hw_frame_;
};

}  // namespace holoscan
}  // namespace nvidia

#endif  // NVIDIA_CLARA_HOLOSCAN_GXF_EXTENSIONS_AJA_SOURCE_HPP_
