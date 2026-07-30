/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef PUBSUB_COMMON_INCLUDE_PUBSUB_PUBSUB_CONTEXT_UTILS_HPP
#define PUBSUB_COMMON_INCLUDE_PUBSUB_PUBSUB_CONTEXT_UTILS_HPP

#include <string>

#include <gxf/pubsub/endpoint_info.hpp>
#include <gxf/pubsub/pubsub_context.hpp>

namespace holoscan {

/// Load a stable host identifier for native-buffer discovery advertisement.
/// Checks HOLOSCAN_HOST_ID env var first, then /etc/machine-id, then
/// /var/lib/dbus/machine-id. Returns empty string if none found.
std::string load_stable_host_id();

/// GPU device info queried for CUDA IPC eligibility.
struct GpuDeviceInfo {
  int32_t device_id{0};
  std::string device_uuid;
  bool cuda_ipc_supported{false};
};

/// Query the current CUDA device for IPC eligibility.
/// Returns default (device_id=0, empty UUID, unsupported) if CUDA is unavailable.
GpuDeviceInfo query_gpu_device_info();

/// Parse a native buffer policy string ("disabled", "preferred", "required")
/// to the GXF enum. Defaults to kPreferred for unrecognized values.
nvidia::gxf::NativeBufferPolicy parse_native_buffer_policy(const std::string& policy_str);

/// Build a NativeBufferCapability from the current device info and policy.
/// Returns a capability with cuda_ipc protocol if supported, or an empty
/// capability if policy is disabled or CUDA IPC is not available.
nvidia::gxf::NativeBufferCapability build_native_buffer_capability(
    const std::string& host_id, const std::string& gpu_device_uuid, bool cuda_ipc_supported,
    nvidia::gxf::NativeBufferPolicy policy);

}  // namespace holoscan

#endif /* PUBSUB_COMMON_INCLUDE_PUBSUB_PUBSUB_CONTEXT_UTILS_HPP */
