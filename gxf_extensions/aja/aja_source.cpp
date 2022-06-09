/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
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

#include <string>
#include <utility>

#include "gxf/multimedia/video.hpp"

template <>
struct YAML::convert<NTV2Channel> {
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

AJASource::AJASource() : pixel_format_(kDefaultPixelFormat), current_buffer_(0) {}

gxf_result_t AJASource::registerInterface(gxf::Registrar* registrar) {
  gxf::Expected<void> result;
  result &= registrar->parameter(signal_, "signal", "Output", "Output signal.");
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
  return gxf::ToResultCode(result);
}

AJAStatus AJASource::DetermineVideoFormat() {
  if (width_ == 1920 && height_ == 1080 && framerate_ == 60) {
    video_format_ = NTV2_FORMAT_1080p_6000_A;
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

  // Wait for a number of frames to acquire video signal.
  current_hw_frame_ = 0;
  device_.SetInputFrame(channel_, current_hw_frame_);
  device_.WaitForInputVerticalInterrupt(channel_, kWarmupFrames);

  return AJA_STATUS_SUCCESS;
}

AJAStatus AJASource::SetupBuffers() {
  auto size = GetVideoWriteSize(video_format_, pixel_format_);

  buffers_.resize(kNumBuffers);
  for (auto& buf : buffers_) {
    if (use_rdma_) {
      cudaMalloc(&buf, size);
      unsigned int syncFlag = 1;
      if (cuPointerSetAttribute(&syncFlag, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS,
                                reinterpret_cast<CUdeviceptr>(buf))) {
        GXF_LOG_ERROR("Failed to set SYNC_MEMOPS CUDA attribute for RDMA");
        return AJA_STATUS_INITIALIZE;
      }
    } else {
      buf = malloc(size);
    }

    if (!buf) {
      GXF_LOG_ERROR("Failed to allocate buffer memory");
      return AJA_STATUS_INITIALIZE;
    }

    if (!device_.DMABufferLock(static_cast<const ULWord*>(buf), size, true, use_rdma_)) {
      GXF_LOG_ERROR("Failed to map buffer for DMA");
      return AJA_STATUS_INITIALIZE;
    }
  }

  return AJA_STATUS_SUCCESS;
}

gxf_result_t AJASource::start() {
  GXF_LOG_INFO("AJA Source: Using NTV2_CHANNEL%d", (channel_.get() + 1));
  GXF_LOG_INFO("AJA Source: RDMA is %s", use_rdma_ ? "enabled" : "disabled");

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

  for (auto& buf : buffers_) {
    if (use_rdma_) {
      cudaFree(buf);
    } else {
      free(buf);
    }
  }
  buffers_.clear();

  return GXF_SUCCESS;
}

gxf_result_t AJASource::tick() {
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

  // Pass the frame downstream.
  auto message = gxf::Entity::New(context());
  if (!message) {
    GXF_LOG_ERROR("Failed to allocate message; terminating.");
    return GXF_FAILURE;
  }

  auto buffer = message.value().add<gxf::VideoBuffer>();
  if (!buffer) {
    GXF_LOG_ERROR("Failed to allocate video buffer; terminating.");
    return GXF_FAILURE;
  }

  gxf::VideoTypeTraits<gxf::VideoFormat::GXF_VIDEO_FORMAT_RGBA> video_type;
  gxf::VideoFormatSize<gxf::VideoFormat::GXF_VIDEO_FORMAT_RGBA> color_format;
  auto color_planes = color_format.getDefaultColorPlanes(width_, height_);
  gxf::VideoBufferInfo info{width_, height_, video_type.value, color_planes,
                            gxf::SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR};
  auto storage_type = use_rdma_ ? gxf::MemoryStorageType::kDevice : gxf::MemoryStorageType::kHost;
  buffer.value()->wrapMemory(info, size, storage_type, buffers_[current_buffer_], nullptr);

  const auto result = signal_->publish(std::move(message.value()));

  current_buffer_ = (current_buffer_ + 1) % kNumBuffers;

  return gxf::ToResultCode(message);
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
