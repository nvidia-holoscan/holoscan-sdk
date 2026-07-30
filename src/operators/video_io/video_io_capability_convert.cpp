/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "holoscan/operators/video_io/video_io_capabilities.hpp"

namespace holoscan::ops::video_io {

VideoDeviceCapabilities to_video_device_capabilities(const VideoCaptureCapabilities& c) {
  VideoDeviceCapabilities out;
  out.backend_id = c.backend_id;
  out.device_id = c.device_id;
  out.device_uri = c.device_uri;
  out.max_concurrent_inputs = c.max_concurrent_inputs;
  out.max_concurrent_outputs = 0;
  out.transports = c.transports;
  out.channels.reserve(c.input_channels.size());
  for (const auto& ic : c.input_channels) {
    out.channels.push_back(static_cast<const ChannelCapabilities&>(ic));
  }
  return out;
}

VideoDeviceCapabilities to_video_device_capabilities(const VideoTransmitCapabilities& c) {
  VideoDeviceCapabilities out;
  out.backend_id = c.backend_id;
  out.device_id = c.device_id;
  out.device_uri = c.device_uri;
  out.max_concurrent_inputs = 0;
  out.max_concurrent_outputs = c.max_concurrent_outputs;
  out.transports = c.transports;
  out.channels.reserve(c.output_channels.size());
  for (const auto& oc : c.output_channels) {
    out.channels.push_back(static_cast<const ChannelCapabilities&>(oc));
  }
  return out;
}

}  // namespace holoscan::ops::video_io
