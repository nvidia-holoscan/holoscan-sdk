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
#include "aja_source.hpp"

#include <cuda.h>
#include <cuda_runtime.h>

#include <sstream>
#include <string>
#include <utility>

#include "gxf/multimedia/video.hpp"

template <>
struct YAML::convert<NTV2Channel> {
  static Node encode(const NTV2Channel& rhs) {
    Node node;
    int channel = static_cast<int>(rhs);
    std::stringstream ss;
    ss << "NTV2_CHANNEL";
    ss << channel;
    node.push_back(ss.str());
    YAML::Node value_node = node[0];
    return value_node;
  }

  static bool decode(const Node& node, NTV2Channel& rhs) {
    if (!node.IsScalar()) return false;

    const std::string prefix("NTV2_CHANNEL");
    auto value = node.Scalar();
    if (value.find(prefix) != 0) return false;
    value = value.substr(prefix.length());

    try {
      size_t len;
      const auto index = std::stoi(value, &len);
      if (index < 1 || index > NTV2_MAX_NUM_CHANNELS || len != value.length()) { return false; }
      rhs = static_cast<NTV2Channel>(index - 1);
      return true;
    } catch (...) { return false; }
  }
};

namespace nvidia {
namespace holoscan {

AJASource::AJASource() : pixel_format_(kDefaultPixelFormat), current_buffer_(0),
                         current_hw_frame_(0), current_overlay_hw_frame_(0),
                         use_tsi_(false), is_kona_hdmi_(false) {}

gxf_result_t AJASource::registerInterface(gxf::Registrar* registrar) {
  gxf::Expected<void> result;
  result &= registrar->parameter(video_buffer_output_, "video_buffer_output", "VideoBufferOutput",
                                 "Output for the video buffer.");
  result &= registrar->parameter(device_specifier_, "device", "Device", "Device specifier.",
                                 std::string(kDefaultDevice));
  result &=
      registrar->parameter(channel_, "channel", "Channel", "NTV2Channel to use.", kDefaultChannel);
  result &= registrar->parameter(width_, "width", "Width", "Width of the stream.", kDefaultWidth);
  result &=
      registrar->parameter(height_, "height", "Height", "Height of the stream.", kDefaultHeight);
  result &= registrar->parameter(framerate_, "framerate", "Framerate", "Framerate of the stream.",
                                 kDefaultFramerate);
  result &= registrar->parameter(use_rdma_, "rdma", "RDMA", "Enable RDMA.", kDefaultRDMA);

  result &= registrar->parameter(enable_overlay_, "enable_overlay", "EnableOverlay",
                                 "Enable overlay.", kDefaultEnableOverlay);
  result &= registrar->parameter(overlay_channel_, "overlay_channel",
                                 "OverlayChannel", "NTV2Channel to use for overlay output.",
                                 kDefaultOverlayChannel);
  result &= registrar->parameter(overlay_rdma_, "overlay_rdma", "OverlayRDMA",
                                 "Enable Overlay RDMA.", kDefaultOverlayRDMA);
  result &= registrar->parameter(overlay_buffer_output_, "overlay_buffer_output",
                                 "OverlayBufferOutput", "Output for an empty overlay buffer.",
                                 gxf::Registrar::NoDefaultParameter(),
                                 GXF_PARAMETER_FLAGS_OPTIONAL);
  result &= registrar->parameter(overlay_buffer_input_, "overlay_buffer_input",
                                 "OverlayBufferInput", "Input for a filled overlay buffer.",
                                gxf::Registrar::NoDefaultParameter(), GXF_PARAMETER_FLAGS_OPTIONAL);

  return gxf::ToResultCode(result);
}

AJAStatus AJASource::DetermineVideoFormat() {
  if (width_ == 1920 && height_ == 1080 && framerate_ == 60) {
    video_format_ = NTV2_FORMAT_1080p_6000_A;
  } else if  (width_ == 3840 && height_ == 2160 && framerate_ == 60) {
    video_format_ = NTV2_FORMAT_3840x2160p_6000;
  } else {
    return AJA_STATUS_UNSUPPORTED;
  }

  return AJA_STATUS_SUCCESS;
}

AJAStatus AJASource::OpenDevice() {
  // Get the requested device.
  if (!CNTV2DeviceScanner::GetFirstDeviceFromArgument(device_specifier_, device_)) {
    GXF_LOG_ERROR("Device %s not found.", device_specifier_.get().c_str());
    return AJA_STATUS_OPEN;
  }

  // Check if the device is ready.
  if (!device_.IsDeviceReady(false)) {
    GXF_LOG_ERROR("Device %s not ready.", device_specifier_.get().c_str());
    return AJA_STATUS_INITIALIZE;
  }

  // Get the device ID.
  device_id_ = device_.GetDeviceID();

  // Detect Kona HDMI device.
  is_kona_hdmi_ = NTV2DeviceGetNumHDMIVideoInputs(device_id_) > 1;

  // Check if a TSI 4x format is needed.
  if (is_kona_hdmi_) { use_tsi_ = GetNTV2VideoFormatTSI(&video_format_); }

  // Check device capabilities.
  if (!NTV2DeviceCanDoVideoFormat(device_id_, video_format_)) {
    GXF_LOG_ERROR("AJA device does not support requested video format.");
    return AJA_STATUS_UNSUPPORTED;
  }
  if (!NTV2DeviceCanDoFrameBufferFormat(device_id_, pixel_format_)) {
    GXF_LOG_ERROR("AJA device does not support requested pixel format.");
    return AJA_STATUS_UNSUPPORTED;
  }
  if (!NTV2DeviceCanDoCapture(device_id_)) {
    GXF_LOG_ERROR("AJA device cannot capture video.");
    return AJA_STATUS_UNSUPPORTED;
  }
  if (!NTV2_IS_VALID_CHANNEL(channel_)) {
    GXF_LOG_ERROR("Invalid AJA channel: %d", channel_);
    return AJA_STATUS_UNSUPPORTED;
  }

  // Check overlay capabilities.
  if (enable_overlay_) {
    if (!NTV2_IS_VALID_CHANNEL(overlay_channel_)) {
      GXF_LOG_ERROR("Invalid overlay channel: %d", overlay_channel_);
      return AJA_STATUS_UNSUPPORTED;
    }

    if (NTV2DeviceGetNumVideoChannels(device_id_) < 2) {
      GXF_LOG_ERROR("Insufficient number of video channels");
      return AJA_STATUS_UNSUPPORTED;
    }

    if (NTV2DeviceGetNumFrameStores(device_id_) < 2) {
      GXF_LOG_ERROR("Insufficient number of frame stores");
      return AJA_STATUS_UNSUPPORTED;
    }

    if (NTV2DeviceGetNumMixers(device_id_) < 1) {
      GXF_LOG_ERROR("Hardware mixing not supported");
      return AJA_STATUS_UNSUPPORTED;
    }

    if (!NTV2DeviceHasBiDirectionalSDI(device_id_)) {
      GXF_LOG_ERROR("BiDirectional SDI not supported");
      return AJA_STATUS_UNSUPPORTED;
    }
  }

  return AJA_STATUS_SUCCESS;
}

AJAStatus AJASource::SetupVideo() {
  NTV2InputSourceKinds input_kind = is_kona_hdmi_ ? NTV2_INPUTSOURCES_HDMI : NTV2_INPUTSOURCES_SDI;
  NTV2InputSource input_src = ::NTV2ChannelToInputSource(channel_, input_kind);
  NTV2Channel tsi_channel = static_cast<NTV2Channel>(channel_ + 1);

  if (!IsRGBFormat(pixel_format_)) {
    GXF_LOG_ERROR("YUV formats not yet supported");
    return AJA_STATUS_UNSUPPORTED;
  }

  // Detect if the source is YUV or RGB (i.e. if CSC is required or not).
  bool is_input_rgb(false);
  if (input_kind == NTV2_INPUTSOURCES_HDMI) {
    NTV2LHIHDMIColorSpace input_color;
    device_.GetHDMIInputColor(input_color, channel_);
    is_input_rgb = (input_color == NTV2_LHIHDMIColorSpaceRGB);
  }

  // Setup the input routing.
  device_.ClearRouting();
  device_.EnableChannel(channel_);
  if (use_tsi_) {
    device_.SetTsiFrameEnable(true, channel_);
    device_.EnableChannel(tsi_channel);
  }
  device_.SetMode(channel_, NTV2_MODE_CAPTURE);
  if (NTV2DeviceHasBiDirectionalSDI(device_id_) && NTV2_INPUT_SOURCE_IS_SDI(input_src)) {
    device_.SetSDITransmitEnable(channel_, false);
  }
  device_.SetVideoFormat(video_format_, false, false, channel_);
  device_.SetFrameBufferFormat(channel_, pixel_format_);
  if (use_tsi_) { device_.SetFrameBufferFormat(tsi_channel, pixel_format_); }
  device_.EnableInputInterrupt(channel_);
  device_.SubscribeInputVerticalEvent(channel_);

  NTV2OutputXptID input_output_xpt =
      GetInputSourceOutputXpt(input_src, /*DS2*/ false, is_input_rgb, /*Quadrant*/ 0);
  NTV2InputXptID fb_input_xpt(GetFrameBufferInputXptFromChannel(channel_));
  if (use_tsi_) {
    if (!is_input_rgb) {
      if (NTV2DeviceGetNumCSCs(device_id_) < 4) {
        GXF_LOG_ERROR("CSCs not available for TSI input.");
        return AJA_STATUS_UNSUPPORTED;
      }
      device_.Connect(NTV2_XptFrameBuffer1Input, NTV2_Xpt425Mux1ARGB);
      device_.Connect(NTV2_XptFrameBuffer1BInput, NTV2_Xpt425Mux1BRGB);
      device_.Connect(NTV2_XptFrameBuffer2Input, NTV2_Xpt425Mux2ARGB);
      device_.Connect(NTV2_XptFrameBuffer2BInput, NTV2_Xpt425Mux2BRGB);
      device_.Connect(NTV2_Xpt425Mux1AInput, NTV2_XptCSC1VidRGB);
      device_.Connect(NTV2_Xpt425Mux1BInput, NTV2_XptCSC2VidRGB);
      device_.Connect(NTV2_Xpt425Mux2AInput, NTV2_XptCSC3VidRGB);
      device_.Connect(NTV2_Xpt425Mux2BInput, NTV2_XptCSC4VidRGB);
      device_.Connect(NTV2_XptCSC1VidInput, NTV2_XptHDMIIn1);
      device_.Connect(NTV2_XptCSC2VidInput, NTV2_XptHDMIIn1Q2);
      device_.Connect(NTV2_XptCSC3VidInput, NTV2_XptHDMIIn1Q3);
      device_.Connect(NTV2_XptCSC4VidInput, NTV2_XptHDMIIn1Q4);
    } else {
      device_.Connect(NTV2_XptFrameBuffer1Input, NTV2_Xpt425Mux1ARGB);
      device_.Connect(NTV2_XptFrameBuffer1BInput, NTV2_Xpt425Mux1BRGB);
      device_.Connect(NTV2_XptFrameBuffer2Input, NTV2_Xpt425Mux2ARGB);
      device_.Connect(NTV2_XptFrameBuffer2BInput, NTV2_Xpt425Mux2BRGB);
      device_.Connect(NTV2_Xpt425Mux1AInput, NTV2_XptHDMIIn1RGB);
      device_.Connect(NTV2_Xpt425Mux1BInput, NTV2_XptHDMIIn1Q2RGB);
      device_.Connect(NTV2_Xpt425Mux2AInput, NTV2_XptHDMIIn1Q3RGB);
      device_.Connect(NTV2_Xpt425Mux2BInput, NTV2_XptHDMIIn1Q4RGB);
    }
  } else if (!is_input_rgb) {
    if (NTV2DeviceGetNumCSCs(device_id_) <= static_cast<int>(channel_)) {
      GXF_LOG_ERROR("No CSC available for NTV2_CHANNEL%d", channel_ + 1);
      return AJA_STATUS_UNSUPPORTED;
    }
    NTV2InputXptID csc_input = GetCSCInputXptFromChannel(channel_);
    NTV2OutputXptID csc_output =
        GetCSCOutputXptFromChannel(channel_, /*inIsKey*/ false, /*inIsRGB*/ true);
    device_.Connect(fb_input_xpt, csc_output);
    device_.Connect(csc_input, input_output_xpt);
  } else {
    device_.Connect(fb_input_xpt, input_output_xpt);
  }

  if (enable_overlay_) {
    // Setup output channel.
    device_.SetReference(NTV2_REFERENCE_INPUT1);
    device_.SetMode(overlay_channel_, NTV2_MODE_DISPLAY);
    device_.SetSDITransmitEnable(overlay_channel_, true);
    device_.SetVideoFormat(video_format_, false, false, overlay_channel_);
    device_.SetFrameBufferFormat(overlay_channel_, NTV2_FBF_ABGR);

    // Setup mixer controls.
    device_.SetMixerFGInputControl(0, NTV2MIXERINPUTCONTROL_SHAPED);
    device_.SetMixerBGInputControl(0, NTV2MIXERINPUTCONTROL_FULLRASTER);
    device_.SetMixerCoefficient(0, 0x10000);
    device_.SetMixerFGMatteEnabled(0, false);
    device_.SetMixerBGMatteEnabled(0, false);

    // Setup routing (overlay frame to CSC, CSC and SDI input to mixer, mixer to SDI output).
    NTV2OutputDestination output_dst = ::NTV2ChannelToOutputDestination(overlay_channel_);
    device_.Connect(GetCSCInputXptFromChannel(overlay_channel_),
                    GetFrameBufferOutputXptFromChannel(overlay_channel_, true /*RGB*/));
    device_.Connect(NTV2_XptMixer1FGVidInput,
                    GetCSCOutputXptFromChannel(overlay_channel_, false /*Key*/));
    device_.Connect(NTV2_XptMixer1FGKeyInput,
                    GetCSCOutputXptFromChannel(overlay_channel_, true /*Key*/));
    device_.Connect(NTV2_XptMixer1BGVidInput, input_output_xpt);
    device_.Connect(GetOutputDestInputXpt(output_dst), NTV2_XptMixer1VidYUV);

    // Set initial output frame (overlay uses HW frames 2 and 3).
    current_overlay_hw_frame_ = 2;
    device_.SetOutputFrame(overlay_channel_, current_overlay_hw_frame_);
  }

  // Wait for a number of frames to acquire video signal.
  current_hw_frame_ = 0;
  device_.SetInputFrame(channel_, current_hw_frame_);
  device_.WaitForInputVerticalInterrupt(channel_, kWarmupFrames);

  return AJA_STATUS_SUCCESS;
}

bool AJASource::AllocateBuffers(std::vector<void*>& buffers, size_t num_buffers,
                                size_t buffer_size, bool rdma) {
  buffers.resize(num_buffers);
  for (auto& buf : buffers) {
    if (rdma) {
      cudaMalloc(&buf, buffer_size);
      unsigned int syncFlag = 1;
      if (cuPointerSetAttribute(&syncFlag, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS,
                                reinterpret_cast<CUdeviceptr>(buf))) {
        GXF_LOG_ERROR("Failed to set SYNC_MEMOPS CUDA attribute for RDMA");
        return false;
      }
    } else {
      buf = malloc(buffer_size);
    }

    if (!buf) {
      GXF_LOG_ERROR("Failed to allocate buffer memory");
      return false;
    }

    if (!device_.DMABufferLock(static_cast<const ULWord*>(buf), buffer_size, true, rdma)) {
      GXF_LOG_ERROR("Failed to map buffer for DMA");
      return false;
    }
  }

  return true;
}

void AJASource::FreeBuffers(std::vector<void*>& buffers, bool rdma) {
  for (auto& buf : buffers) {
    if (rdma) {
      cudaFree(buf);
    } else {
      free(buf);
    }
  }
  buffers.clear();
}

AJAStatus AJASource::SetupBuffers() {
  auto size = GetVideoWriteSize(video_format_, pixel_format_);

  if (!AllocateBuffers(buffers_, kNumBuffers, size, use_rdma_)) {
    return AJA_STATUS_INITIALIZE;
  }

  if (enable_overlay_) {
    if (!AllocateBuffers(overlay_buffers_, kNumBuffers, size, overlay_rdma_)) {
      return AJA_STATUS_INITIALIZE;
    }
  }

  return AJA_STATUS_SUCCESS;
}

gxf_result_t AJASource::start() {
  GXF_LOG_INFO("AJA Source: Capturing from NTV2_CHANNEL%d", (channel_.get() + 1));
  GXF_LOG_INFO("AJA Source: RDMA is %s", use_rdma_ ? "enabled" : "disabled");
  if (enable_overlay_) {
    GXF_LOG_INFO("AJA Source: Outputting overlay to NTV2_CHANNEL%d", (overlay_channel_.get() + 1));
    GXF_LOG_INFO("AJA Source: Overlay RDMA is %s", overlay_rdma_ ? "enabled" : "disabled");
  } else {
    GXF_LOG_INFO("AJA Source: Overlay output is disabled");
  }

  AJAStatus status = DetermineVideoFormat();
  if (AJA_FAILURE(status)) {
    GXF_LOG_ERROR("Video format could not be determined based on parameters.");
    return GXF_FAILURE;
  }

  status = OpenDevice();
  if (AJA_FAILURE(status)) {
    GXF_LOG_ERROR("Failed to open device %s", device_specifier_.get().c_str());
    return GXF_FAILURE;
  }

  status = SetupVideo();
  if (AJA_FAILURE(status)) {
    GXF_LOG_ERROR("Failed to setup device %s", device_specifier_.get().c_str());
    return GXF_FAILURE;
  }

  status = SetupBuffers();
  if (AJA_FAILURE(status)) {
    GXF_LOG_ERROR("Failed to setup AJA buffers.");
    return GXF_FAILURE;
  }

  return GXF_SUCCESS;
}

gxf_result_t AJASource::stop() {
  device_.UnsubscribeInputVerticalEvent(channel_);
  device_.DMABufferUnlockAll();

  if (enable_overlay_) {
    device_.SetMixerMode(0, NTV2MIXERMODE_FOREGROUND_OFF);
  }

  FreeBuffers(buffers_, use_rdma_);
  FreeBuffers(overlay_buffers_, overlay_rdma_);

  return GXF_SUCCESS;
}

gxf_result_t AJASource::tick() {
  // Update the overlay frame.
  if (enable_overlay_ && overlay_buffer_input_.try_get()) {
    const auto& overlay_buffer_input = overlay_buffer_input_.try_get().value()->receive();
    if (overlay_buffer_input) {
      const auto& overlay_buffer = overlay_buffer_input.value().get<gxf::VideoBuffer>();
      if (overlay_buffer) {
        // Overlay uses HW frames 2 and 3.
        current_overlay_hw_frame_ = ((current_overlay_hw_frame_ + 1) % 2) + 2;

        const auto& buffer = overlay_buffer.value();
        ULWord* ptr = reinterpret_cast<ULWord*>(buffer->pointer());
        device_.DMAWriteFrame(current_overlay_hw_frame_, ptr, buffer->size());
        device_.SetOutputFrame(overlay_channel_, current_overlay_hw_frame_);
        device_.SetMixerMode(0, NTV2MIXERMODE_MIX);
      }
    }
  }

  // Update the next input frame and wait until it starts.
  uint32_t next_hw_frame = (current_hw_frame_ + 1) % 2;
  device_.SetInputFrame(channel_, next_hw_frame);
  device_.WaitForInputVerticalInterrupt(channel_);

  // Read the last completed frame.
  auto size = GetVideoWriteSize(video_format_, pixel_format_);
  auto ptr = static_cast<ULWord*>(buffers_[current_buffer_]);
  device_.DMAReadFrame(current_hw_frame_, ptr, size);

  // Set the frame to read for the next tick.
  current_hw_frame_ = next_hw_frame;

  // Common (output and overlay) buffer info
  gxf::VideoTypeTraits<gxf::VideoFormat::GXF_VIDEO_FORMAT_RGBA> video_type;
  gxf::VideoFormatSize<gxf::VideoFormat::GXF_VIDEO_FORMAT_RGBA> color_format;
  auto color_planes = color_format.getDefaultColorPlanes(width_, height_);
  gxf::VideoBufferInfo info{width_, height_, video_type.value, color_planes,
                            gxf::SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR};

  // Pass an empty overlay buffer downstream.
  if (enable_overlay_ && overlay_buffer_output_.try_get()) {
    auto overlay_output = gxf::Entity::New(context());
    if (!overlay_output) {
      GXF_LOG_ERROR("Failed to allocate overlay output; terminating.");
      return GXF_FAILURE;
    }

    auto overlay_buffer = overlay_output.value().add<gxf::VideoBuffer>();
    if (!overlay_buffer) {
      GXF_LOG_ERROR("Failed to allocate overlay buffer; terminating.");
      return GXF_FAILURE;
    }

    auto overlay_storage_type =
        overlay_rdma_ ? gxf::MemoryStorageType::kDevice : gxf::MemoryStorageType::kHost;
    overlay_buffer.value()->wrapMemory(info, size, overlay_storage_type,
                                       overlay_buffers_[current_buffer_], nullptr);

    auto overlay_result = overlay_buffer_output_.try_get().value()->publish(
        std::move(overlay_output.value()));
    if (GXF_SUCCESS != gxf::ToResultCode(overlay_result)) {
      GXF_LOG_ERROR("Failed to publish overlay buffer; terminating.");
      return GXF_FAILURE;
    }
  }

  // Pass the video output buffer downstream.
  auto video_output = gxf::Entity::New(context());
  if (!video_output) {
    GXF_LOG_ERROR("Failed to allocate video output; terminating.");
    return GXF_FAILURE;
  }

  auto video_buffer = video_output.value().add<gxf::VideoBuffer>();
  if (!video_buffer) {
    GXF_LOG_ERROR("Failed to allocate video buffer; terminating.");
    return GXF_FAILURE;
  }

  auto storage_type = use_rdma_ ? gxf::MemoryStorageType::kDevice : gxf::MemoryStorageType::kHost;
  video_buffer.value()->wrapMemory(info, size, storage_type, buffers_[current_buffer_], nullptr);

  auto result = video_buffer_output_->publish(std::move(video_output.value()));
  if (GXF_SUCCESS != gxf::ToResultCode(result)) {
    GXF_LOG_ERROR("Failed to publish video buffer; terminating.");
    return GXF_FAILURE;
  }

  // Update the current buffer (index shared between video and overlay)
  current_buffer_ = (current_buffer_ + 1) % kNumBuffers;

  return GXF_SUCCESS;
}

bool AJASource::GetNTV2VideoFormatTSI(NTV2VideoFormat* format) {
  switch (*format) {
    case NTV2_FORMAT_3840x2160p_2400:
      *format = NTV2_FORMAT_4x1920x1080p_2400;
      return true;
    case NTV2_FORMAT_3840x2160p_6000:
      *format = NTV2_FORMAT_4x1920x1080p_6000;
      return true;
    case NTV2_FORMAT_4096x2160p_2400:
      *format = NTV2_FORMAT_4x2048x1080p_2400;
      return true;
    case NTV2_FORMAT_4096x2160p_6000:
      *format = NTV2_FORMAT_4x2048x1080p_6000;
      return true;
    default:
      return false;
  }
}

}  // namespace holoscan
}  // namespace nvidia
