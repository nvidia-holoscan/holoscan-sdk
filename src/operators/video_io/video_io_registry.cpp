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

#include "holoscan/operators/video_io/video_io_registry.hpp"

#include <holoscan/logger/logger.hpp>

#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace holoscan::ops::video_io {

namespace {

std::mutex g_registry_mutex;
std::unordered_map<std::string, std::vector<VideoCaptureCapabilityEnumerator>> g_acquisition;
std::unordered_map<std::string, std::vector<VideoTransmitCapabilityEnumerator>> g_transmission;

template <typename CapT, typename EnumeratorT>
std::vector<CapT> run_enumerators(std::vector<EnumeratorT> to_run) {
  std::vector<CapT> out;
  for (const auto& fn : to_run) {
    auto chunk = fn();
    out.insert(out.end(), chunk.begin(), chunk.end());
  }
  return out;
}

template <typename EnumeratorT>
std::vector<EnumeratorT> collect_enumerators_locked(
    const std::unordered_map<std::string, std::vector<EnumeratorT>>& table,
    const std::string& backend_id) {
  std::vector<EnumeratorT> to_run;
  {
    std::lock_guard<std::mutex> lock(g_registry_mutex);
    if (backend_id.empty()) {
      for (const auto& entry : table) {
        for (const auto& fn : entry.second) {
          if (fn) {
            to_run.push_back(fn);
          }
        }
      }
    } else {
      auto it = table.find(backend_id);
      if (it != table.end()) {
        for (const auto& fn : it->second) {
          if (fn) {
            to_run.push_back(fn);
          }
        }
      }
    }
  }
  return to_run;
}

}  // namespace

void register_video_acquisition_enumerator(const std::string& backend_id,
                                           VideoCaptureCapabilityEnumerator enumerator) {
  if (backend_id.empty()) {
    HOLOSCAN_LOG_WARN(
        "register_video_acquisition_enumerator called with empty backend_id; ignoring");
    return;
  }
  std::lock_guard<std::mutex> lock(g_registry_mutex);
  g_acquisition[backend_id].push_back(std::move(enumerator));
}

void register_video_transmission_enumerator(const std::string& backend_id,
                                            VideoTransmitCapabilityEnumerator enumerator) {
  if (backend_id.empty()) {
    HOLOSCAN_LOG_WARN(
        "register_video_transmission_enumerator called with empty backend_id; ignoring");
    return;
  }
  std::lock_guard<std::mutex> lock(g_registry_mutex);
  g_transmission[backend_id].push_back(std::move(enumerator));
}

std::vector<VideoCaptureCapabilities> enumerate_video_acquisition_devices(
    const std::string& backend_id) {
  return run_enumerators<VideoCaptureCapabilities>(
      collect_enumerators_locked(g_acquisition, backend_id));
}

std::vector<VideoTransmitCapabilities> enumerate_video_transmission_devices(
    const std::string& backend_id) {
  return run_enumerators<VideoTransmitCapabilities>(
      collect_enumerators_locked(g_transmission, backend_id));
}

}  // namespace holoscan::ops::video_io
