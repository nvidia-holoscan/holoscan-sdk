/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Selects one planned CUDA device for a bounded Runtime operation and restores the prior device.

#pragma once

#include <cuda_runtime_api.h>

#include <holoscan/core/memory_kind.hpp>

namespace holoscan::detail {

class ScopedCudaDevice final {
 public:
  explicit ScopedCudaDevice(DeviceId requested) noexcept {
    if (!requested.valid() || cudaGetDevice(&previous_) != cudaSuccess) {
      return;
    }
    if (previous_ != requested.value) {
      if (cudaSetDevice(requested.value) != cudaSuccess) {
        return;
      }
      changed_ = true;
    }
    active_ = true;
  }

  ~ScopedCudaDevice() {
    if (changed_) {
      static_cast<void>(cudaSetDevice(previous_));
    }
  }

  [[nodiscard]] bool active() const noexcept { return active_; }

 private:
  int previous_{};
  bool active_{};
  bool changed_{};
};

}  // namespace holoscan::detail
