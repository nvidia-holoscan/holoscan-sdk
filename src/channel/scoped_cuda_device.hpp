/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <optional>

#include <cuda.h>
#include <cuda_runtime_api.h>

namespace holoscan::channel_internal {

// Device restoration alone selects a primary context and loses a caller's Green/custom
// context. Capture the exact context before switching; the absent-device host path stays cheap.
class ScopedCUDADevice final {
 public:
  explicit ScopedCUDADevice(const std::optional<std::int32_t> device) noexcept {
    if (!device) {
      active_ = true;
      return;
    }
    if (*device < 0 || cudaGetDevice(&previous_device_) != cudaSuccess) {
      return;
    }
    if (previous_device_ != *device) {
      if (cuCtxGetCurrent(&previous_context_) != CUDA_SUCCESS) {
        return;
      }
      changed_ = true;
      if (cudaSetDevice(*device) != cudaSuccess) {
        return;
      }
    }
    active_ = true;
  }
  ScopedCUDADevice(const ScopedCUDADevice&) = delete;
  ScopedCUDADevice& operator=(const ScopedCUDADevice&) = delete;
  ~ScopedCUDADevice() {
    if (changed_) {
      static_cast<void>(cuCtxSetCurrent(previous_context_));
    }
  }
  [[nodiscard]] bool active() const noexcept { return active_; }

 private:
  CUcontext previous_context_{};
  int previous_device_{};
  bool changed_{};
  bool active_{};
};

}  // namespace holoscan::channel_internal
