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

#include "holoscan/operators/video_io/video_transmission_operator.hpp"

#include "holoscan/core/io_context.hpp"
#include "holoscan/operators/video_io/video_io_capabilities.hpp"
#include "video_io_parse_helpers.hpp"

#include <fmt/format.h>

#include <algorithm>
#include <cctype>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace holoscan::ops {

namespace {

void append_uri_scheme(video_io::VideoTransmitCapabilities& cap, const std::string& uri) {
  const auto pos = uri.find("://");
  if (pos != std::string::npos && pos > 0) {
    cap.connection_uri_schemes.push_back(uri.substr(0, pos));
  }
}

void fill_transmit_channel(video_io::VideoTransmitChannelCapabilities& ch, uint32_t channel_index,
                           const std::string& transport_str, uint32_t width, uint32_t height,
                           float frame_rate, const std::string& pixel_format,
                           const std::string& color_space_str) {
  ch.channel_index = channel_index;
  ch.transport = video_io::parse_transport_token(transport_str);
  ch.interface_label = "output " + std::to_string(channel_index);
  if (width > 0 && height > 0) {
    ch.resolutions.push_back({width, height});
  }
  if (frame_rate > 0.f) {
    ch.framerates.push_back({frame_rate, frame_rate});
  }
  std::string pf_lower = pixel_format;
  std::transform(pf_lower.begin(), pf_lower.end(), pf_lower.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  if (pf_lower != "auto" && !pf_lower.empty()) {
    ch.pixel_formats.push_back({pixel_format, std::string{}});
  }
  ch.color_spaces.push_back(video_io::parse_color_space(color_space_str));
}

}  // namespace

VideoTransmissionOperator::VideoTransmissionOperator(uint32_t num_streams)
    : num_streams_(std::max(1u, std::min(num_streams, video_io::kVideoIoMaxStreams))) {
  init_port_names();
}

void VideoTransmissionOperator::setup(OperatorSpec& spec) {
  for (uint32_t i = 0; i < num_streams_; ++i) {
    spec.input<std::shared_ptr<holoscan::gxf::Entity>>(port_names_[i]);
  }

  static constexpr char kDefaultBackend[] = "generic";
  static constexpr uint32_t kDefaultChannel = 0;
  static constexpr char kDefaultUri[] = "";
  static constexpr uint32_t kDefaultSize = 0;
  static constexpr float kDefaultFps = 0.f;
  static constexpr char kDefaultPixel[] = "auto";
  static constexpr char kDefaultColor[] = "auto";
  static constexpr char kDefaultTransport[] = "auto";

  spec.param(backend_id_,
             "backend_id",
             "Backend ID",
             "Logical name for this transmission integration "
             "(capability reporting).",
             std::string(kDefaultBackend));
  spec.param(channel_index_,
             "channel_index",
             "Channel index",
             "Zero-based output channel index for this "
             "operator instance.",
             kDefaultChannel);
  spec.param(channel_indices_,
             "channel_indices",
             "Channel indices",
             "Optional multi-channel output index list for "
             "singleton vendor SDK fallback.",
             std::vector<uint32_t>{});
  spec.param(
      uri_, "uri", "URI / device", "Device path, id, or stream URI.", std::string(kDefaultUri));
  spec.param(width_, "width", "Width", "Requested frame width (0 = default).", kDefaultSize);
  spec.param(height_, "height", "Height", "Requested frame height (0 = default).", kDefaultSize);
  spec.param(
      frame_rate_, "frame_rate", "Frame rate", "Requested frame rate (0 = default).", kDefaultFps);
  spec.param(pixel_format_,
             "pixel_format",
             "Pixel format",
             "Requested pixel format label or fourcc.",
             std::string(kDefaultPixel));
  spec.param(color_space_,
             "color_space",
             "Color space",
             "Requested color space (auto, bt709, ...).",
             std::string(kDefaultColor));
  spec.param(transport_,
             "transport",
             "Transport",
             "Transport hint: auto, sdi, hdmi, ethernet, "
             "v4l2, ...",
             std::string(kDefaultTransport));
  spec.param(vendor_extensions_,
             "vendor_extensions",
             "Vendor extensions",
             "YAML map of vendor.<name>.* keys.",
             YAML::Node(YAML::NodeType::Map));
}

void VideoTransmissionOperator::initialize() {
  Operator::initialize();
}

bool VideoTransmissionOperator::is_transmit_stream_enabled(uint32_t stream_index) const {
  return stream_index < num_streams_;
}

std::string VideoTransmissionOperator::transmit_input_port_name(uint32_t stream_index) {
  if (stream_index >= video_io::kVideoIoMaxStreams) {
    throw std::out_of_range(
        "VideoTransmissionOperator::transmit_input_port_name: "
        "stream_index");
  }
  return stream_index == 0 ? std::string("signal")
                           : std::string("signal_") + std::to_string(stream_index);
}

std::shared_ptr<holoscan::gxf::Entity> VideoTransmissionOperator::receive_transmit_stream(
    InputContext& op_input, uint32_t stream_index) {
  if (stream_index >= num_streams_) {
    throw std::out_of_range(
        fmt::format("VideoTransmissionOperator::receive_transmit_stream: "
                    "stream_index {} >= num_streams ({})",
                    stream_index,
                    num_streams_));
  }
  return op_input.receive<std::shared_ptr<holoscan::gxf::Entity>>(port_names_[stream_index].c_str())
      .value();
}

video_io::VideoTransmitCapabilities VideoTransmissionOperator::query_transmit_capabilities() const {
  return build_transmit_capabilities_from_parameters();
}

video_io::VideoDeviceCapabilities VideoTransmissionOperator::query_capabilities() const {
  return video_io::to_video_device_capabilities(query_transmit_capabilities());
}

video_io::VideoTransmitCapabilities
VideoTransmissionOperator::build_transmit_capabilities_from_parameters() const {
  video_io::VideoTransmitCapabilities cap;
  cap.backend_id = backend_id_.get();
  cap.device_uri = uri_.get();
  cap.device_id = cap.device_uri;
  append_uri_scheme(cap, cap.device_uri);

  const std::string transport_str = transport_.get();
  const uint32_t w = width_.get();
  const uint32_t h = height_.get();
  const float fps = frame_rate_.get();
  const std::string pf = pixel_format_.get();
  const std::string cs = color_space_.get();

  const auto& channels = channel_indices_.get();
  if (!channels.empty()) {
    cap.max_concurrent_outputs = static_cast<uint32_t>(channels.size());
    for (uint32_t idx : channels) {
      video_io::VideoTransmitChannelCapabilities ch{};
      fill_transmit_channel(ch, idx, transport_str, w, h, fps, pf, cs);
      cap.output_channels.push_back(std::move(ch));
    }
  } else {
    cap.max_concurrent_outputs = 1;
    video_io::VideoTransmitChannelCapabilities ch{};
    fill_transmit_channel(ch, channel_index_.get(), transport_str, w, h, fps, pf, cs);
    cap.output_channels.push_back(std::move(ch));
  }

  video_io::VideoTransport tr = video_io::parse_transport_token(transport_str);
  if (tr != video_io::VideoTransport::kUnknown) {
    cap.transports.push_back(tr);
  }

  return cap;
}

}  // namespace holoscan::ops
