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

#ifndef HOLOSCAN_OPERATORS_VIDEO_IO_VIDEO_IO_REGISTRY_HPP
#define HOLOSCAN_OPERATORS_VIDEO_IO_VIDEO_IO_REGISTRY_HPP

#include <functional>
#include <string>
#include <vector>

#include "./video_io_capabilities.hpp"

namespace holoscan::ops::video_io {

using VideoCaptureCapabilityEnumerator = std::function<std::vector<VideoCaptureCapabilities>()>;
using VideoTransmitCapabilityEnumerator = std::function<std::vector<VideoTransmitCapabilities>()>;

/**
 * @brief Register a capability enumerator for video acquisition.
 *
 * Multiple enumerators may be registered per @p backend_id; results are concatenated in
 * registration order. Vendor extensions should use a distinct backend_id (e.g. "vendor.example").
 *
 * @param backend_id Non-empty identifier for the stack or vendor integration.
 * @param enumerator Callable that returns `VideoCaptureCapabilities` without starting a pipeline.
 */
void register_video_acquisition_enumerator(const std::string& backend_id,
                                           VideoCaptureCapabilityEnumerator enumerator);

/**
 * @brief Register a capability enumerator for video transmission.
 */
void register_video_transmission_enumerator(const std::string& backend_id,
                                            VideoTransmitCapabilityEnumerator enumerator);

/**
 * @brief Enumerate acquisition capabilities (concrete capture view per device).
 *
 * If @p backend_id is empty, invokes every registered acquisition enumerator and concatenates
 * results. Otherwise only enumerators for that backend are run.
 */
std::vector<VideoCaptureCapabilities> enumerate_video_acquisition_devices(
    const std::string& backend_id = "");

/**
 * @brief Enumerate transmission capabilities (concrete transmit view per device).
 */
std::vector<VideoTransmitCapabilities> enumerate_video_transmission_devices(
    const std::string& backend_id = "");

}  // namespace holoscan::ops::video_io

#endif  // HOLOSCAN_OPERATORS_VIDEO_IO_VIDEO_IO_REGISTRY_HPP
