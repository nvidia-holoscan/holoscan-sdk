/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cuda/current_device.hpp"

#include <cuda.h>

namespace holomq {
namespace cuda {

std::optional<int32_t> current_device_ordinal() noexcept {
  CUdevice device{};
  if (cuCtxGetDevice(&device) != CUDA_SUCCESS) {
    return std::nullopt;
  }
  return static_cast<int32_t>(device);
}

}  // namespace cuda
}  // namespace holomq
