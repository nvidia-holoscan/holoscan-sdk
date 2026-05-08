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

#ifndef SRC_OPERATORS_VIDEO_IO_VIDEO_IO_PARSE_HELPERS_HPP
#define SRC_OPERATORS_VIDEO_IO_VIDEO_IO_PARSE_HELPERS_HPP

#include "holoscan/operators/video_io/video_io_capabilities.hpp"

#include <algorithm>
#include <cctype>
#include <string>

namespace holoscan::ops::video_io {

inline VideoColorSpaceKind parse_color_space(const std::string& s) {
  std::string lower = s;
  std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  if (lower == "bt601") {
    return VideoColorSpaceKind::kBt601;
  }
  if (lower == "bt709") {
    return VideoColorSpaceKind::kBt709;
  }
  if (lower == "bt2020") {
    return VideoColorSpaceKind::kBt2020;
  }
  if (lower == "srgb") {
    return VideoColorSpaceKind::kSrgb;
  }
  if (lower == "auto" || lower.empty()) {
    return VideoColorSpaceKind::kAuto;
  }
  return VideoColorSpaceKind::kUnknown;
}

inline VideoTransport parse_transport_token(const std::string& s) {
  std::string lower = s;
  std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  if (lower == "sdi") {
    return VideoTransport::kSdi;
  }
  if (lower == "hdmi") {
    return VideoTransport::kHdmi;
  }
  if (lower == "ethernet" || lower == "eth") {
    return VideoTransport::kEthernet;
  }
  if (lower == "usb") {
    return VideoTransport::kUsb;
  }
  if (lower == "v4l2") {
    return VideoTransport::kV4l2;
  }
  if (lower == "file") {
    return VideoTransport::kFile;
  }
  if (lower == "auto" || lower.empty()) {
    return VideoTransport::kUnknown;
  }
  return VideoTransport::kOther;
}

}  // namespace holoscan::ops::video_io

#endif  // SRC_OPERATORS_VIDEO_IO_VIDEO_IO_PARSE_HELPERS_HPP
