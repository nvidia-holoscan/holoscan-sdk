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
