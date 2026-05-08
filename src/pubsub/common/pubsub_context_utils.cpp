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

#include <holoscan/pubsub/common/pubsub_context_utils.hpp>

#include <cstdlib>
#include <fstream>
#include <string>

#include <cuda_runtime.h>

#include <holoscan/logger/logger.hpp>

#include <gxf/pubsub/cuda_ipc_descriptor.hpp>
#include <gxf/pubsub/cuda_ipc_eligibility.hpp>

namespace holoscan {

std::string load_stable_host_id() {
  if (const char* env = std::getenv("HOLOSCAN_HOST_ID")) {
    if (env[0] != '\0') {
      return std::string(env);
    }
  }
  for (const char* path : {"/etc/machine-id", "/var/lib/dbus/machine-id"}) {
    std::ifstream file(path);
    if (!file)
      continue;
    std::string line;
    if (!std::getline(file, line))
      continue;
    while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) {
      line.pop_back();
    }
    if (!line.empty())
      return line;
  }
  return {};
}

GpuDeviceInfo query_gpu_device_info() {
  GpuDeviceInfo info;
  int device_id = 0;
  cudaError_t err = cudaGetDevice(&device_id);
  if (err == cudaSuccess) {
    info.device_id = device_id;
    auto gpu_info = nvidia::gxf::CudaDeviceIpcInfo::query(device_id);
    info.device_uuid = gpu_info.uuid;
    info.cuda_ipc_supported = gpu_info.cuda_ipc_supported;
  }
  return info;
}

nvidia::gxf::NativeBufferPolicy parse_native_buffer_policy(const std::string& policy_str) {
  if (policy_str == "disabled")
    return nvidia::gxf::NativeBufferPolicy::kDisabled;
  if (policy_str == "required")
    return nvidia::gxf::NativeBufferPolicy::kRequired;
  if (policy_str != "preferred") {
    HOLOSCAN_LOG_WARN(
        "parse_native_buffer_policy: unrecognized policy '{}', defaulting to 'preferred'",
        policy_str);
  }
  return nvidia::gxf::NativeBufferPolicy::kPreferred;
}

nvidia::gxf::NativeBufferCapability build_native_buffer_capability(
    const std::string& host_id, const std::string& gpu_device_uuid, bool cuda_ipc_supported,
    nvidia::gxf::NativeBufferPolicy policy) {
  nvidia::gxf::NativeBufferCapability cap;
  cap.host_id = host_id;
  if (policy == nvidia::gxf::NativeBufferPolicy::kDisabled)
    return cap;
  if (cuda_ipc_supported) {
    cap.native_buffer_protocols = {"cuda_ipc"};
    cap.memory_domain = "same_host_gpu";
    cap.gpu_device_uuid = gpu_device_uuid;
    cap.native_buffer_profile = "cuda_ipc_same_gpu_v1";
    cap.descriptor_format_version = nvidia::gxf::kCudaIpcFormatVersion;
  }
  return cap;
}

}  // namespace holoscan
