/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef HOLOSCAN_OPERATORS_VIDEO_IO_VIDEO_IO_CAPABILITIES_HPP
#define HOLOSCAN_OPERATORS_VIDEO_IO_VIDEO_IO_CAPABILITIES_HPP

#include <cstdint>
#include <string>
#include <vector>

namespace holoscan::ops::video_io {

/** Maximum number of I/O ports a base video operator may allocate.
 *
 *  Each port consumes two GXF components (a Transmitter/Receiver plus
 *  its scheduling condition), and GXF limits an entity to 1024
 *  components, so 128 leaves ample headroom for other components. */
inline constexpr std::uint32_t kVideoIoMaxStreams = 128;

/**
 * @brief Physical or logical transport for a video channel.
 */
enum class VideoTransport {
  kUnknown = 0,
  kSdi,
  kHdmi,
  kEthernet,
  kUsb,
  kV4l2,
  kFile,
  kOther,
};

/**
 * @brief Declared color space / range semantics for a channel.
 */
enum class VideoColorSpaceKind {
  kUnknown = 0,
  kBt601,
  kBt709,
  kBt2020,
  kSrgb,
  kAuto,
};

/**
 * @brief Synchronization / clocking modes exposed by hardware or stack.
 */
enum class SyncClockMode {
  kUnknown = 0,
  kFreeRun,
  kGenlock,
  kPtp,
  kHouseSync,
};

/**
 * @brief Minimum and maximum frame rate for a mode, in frames per second.
 */
struct FramerateRange {
  double min_fps = 0.;
  double max_fps = 0.;
};

struct Resolution {
  uint32_t width = 0;
  uint32_t height = 0;
};

/**
 * @brief Human- or vendor-readable pixel format label (e.g. fourcc, vendor codec name).
 */
struct PixelFormatDesc {
  std::string name;
  std::string description;
};

/**
 * @brief Qualitative / quantitative latency hints.
 */
struct VideoLatencyHint {
  /// Opaque vendor or stack-specific latency class ("ultra_low", "standard", etc.).
  std::string qualitative;
  /// End-to-end estimate in milliseconds when known; negative if unavailable.
  double estimated_ms = -1.;
};

/**
 * @brief Shared fields for a single input or output channel on a device.
 */
struct ChannelCapabilities {
  uint32_t channel_index = 0;
  VideoTransport transport = VideoTransport::kUnknown;
  std::vector<Resolution> resolutions;
  std::vector<FramerateRange> framerates;
  std::vector<PixelFormatDesc> pixel_formats;
  std::vector<VideoColorSpaceKind> color_spaces;
  std::vector<SyncClockMode> sync_modes;
  bool gpudirect_rdma_supported = false;
  bool safety_bypass_supported = false;
  VideoLatencyHint latency;
};

/**
 * @brief Capture-oriented channel capabilities (concrete input role).
 */
struct VideoCaptureChannelCapabilities : ChannelCapabilities {
  /// Human-readable connector or stream label (e.g. "SDI In 1", "HDMI A").
  std::string interface_label;
  bool progressive_capture_supported = true;
  bool interlaced_capture_supported = false;
  bool hardware_timestamp_supported = false;
};

/**
 * @brief Transmission-oriented channel capabilities (concrete output role).
 */
struct VideoTransmitChannelCapabilities : ChannelCapabilities {
  std::string interface_label;
  bool progressive_output_supported = true;
  bool interlaced_output_supported = false;
  bool embedded_audio_output_supported = false;
};

/**
 * @brief Capability snapshot for one capture device (boards, NIC ingest, files, etc.).
 */
struct VideoCaptureCapabilities {
  std::string backend_id;
  std::string device_id;
  std::string device_uri;
  uint32_t max_concurrent_inputs = 0;
  std::vector<VideoTransport> transports;
  std::vector<VideoCaptureChannelCapabilities> input_channels;
  /// URI schemes this device accepts for @c device_uri (e.g. "file", "rtp", "sdi"), when known.
  std::vector<std::string> connection_uri_schemes;
};

/**
 * @brief Capability snapshot for one transmit / playout device.
 */
struct VideoTransmitCapabilities {
  std::string backend_id;
  std::string device_id;
  std::string device_uri;
  uint32_t max_concurrent_outputs = 0;
  std::vector<VideoTransport> transports;
  std::vector<VideoTransmitChannelCapabilities> output_channels;
  std::vector<std::string> connection_uri_schemes;
};

/**
 * @brief Capability snapshot for one logical video device (board, NIC, URI namespace, etc.).
 *
 * Combines input and output limits for tools that need a single summary. Prefer
 * `VideoCaptureCapabilities` / `VideoTransmitCapabilities` for typed capture vs transmit views.
 */
struct VideoDeviceCapabilities {
  std::string backend_id;
  std::string device_id;
  std::string device_uri;
  uint32_t max_concurrent_inputs = 0;
  uint32_t max_concurrent_outputs = 0;
  std::vector<VideoTransport> transports;
  std::vector<ChannelCapabilities> channels;
};

/**
 * @brief Flatten capture capabilities into the direction-agnostic device summary.
 *
 * @warning Direction-specific fields (`interface_label`, `progressive_capture_supported`,
 * `interlaced_capture_supported`, `hardware_timestamp_supported`) are lost during the
 * conversion because the output uses the base `ChannelCapabilities` type.
 */
VideoDeviceCapabilities to_video_device_capabilities(const VideoCaptureCapabilities& c);

/**
 * @brief Flatten transmit capabilities into the direction-agnostic device summary.
 *
 * @warning Direction-specific fields (`interface_label`, `progressive_output_supported`,
 * `interlaced_output_supported`, `embedded_audio_output_supported`) are lost during the
 * conversion because the output uses the base `ChannelCapabilities` type.
 */
VideoDeviceCapabilities to_video_device_capabilities(const VideoTransmitCapabilities& c);

}  // namespace holoscan::ops::video_io

#endif  // HOLOSCAN_OPERATORS_VIDEO_IO_VIDEO_IO_CAPABILITIES_HPP
