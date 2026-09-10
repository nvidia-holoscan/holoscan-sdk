/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cu_device.hpp"
#include <cstring>
#include "cu_errors.hpp"

namespace holoviz::rhi::cuda {

/// cuInit(0) exactly once per process, thread-safe (magic static: concurrent first
/// callers block until the initializer finishes). A failure is sticky — cuInit does
/// not self-heal, so every later caller re-reports the stored CUresult.
static expected<void> ensure_cuda_initialized() {
  static const CUresult init_result = cuInit(0);
  VIZ_CU_FORWARD_ERROR(init_result);
  return {};
}

expected<int> get_cuda_ordinal(const std::array<uint8_t, 16>& uuid) {
  VIZ_FORWARD_ERROR(ensure_cuda_initialized());

  int device_count = 0;
  VIZ_CU_FORWARD_ERROR(cuDeviceGetCount(&device_count));

  for (int i = 0; i < device_count; ++i) {
    CUdevice device;
    VIZ_CU_FORWARD_ERROR(cuDeviceGet(&device, i));

    CUuuid device_uuid;
    VIZ_CU_FORWARD_ERROR(cuDeviceGetUuid(&device_uuid, device));

    if (std::memcmp(std::begin(device_uuid.bytes), uuid.data(), sizeof(device_uuid.bytes)) == 0) {
      return i;  // Found the matching CUDA device ordinal
    }
  }

  return make_unexpected(
      Error{.code = ErrorCode::kNotFound,
            .message = "[holoviz] No CUDA device found with the specified UUID"});
}

}  // namespace holoviz::rhi::cuda
