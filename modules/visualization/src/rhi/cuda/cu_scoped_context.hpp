/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda.h>

namespace holoviz::rhi {

/// Push a CUDA context for the current scope, so the enclosing code's CUDA calls never
/// depend on the caller having made a context current. A null context is a no-op, as is
/// a failed push (the CUDA call that follows will report the real error).
class ScopedCudaContext {
 public:
  explicit ScopedCudaContext(CUcontext context) {
    pushed_ = context != nullptr && cuCtxPushCurrent(context) == CUDA_SUCCESS;
  }
  ~ScopedCudaContext() {
    if (pushed_) {
      CUcontext popped = nullptr;
      cuCtxPopCurrent(&popped);
    }
  }
  ScopedCudaContext(const ScopedCudaContext&) = delete;
  ScopedCudaContext& operator=(const ScopedCudaContext&) = delete;
  ScopedCudaContext(ScopedCudaContext&&) = delete;
  ScopedCudaContext& operator=(ScopedCudaContext&&) = delete;

 private:
  bool pushed_ = false;
};

}  // namespace holoviz::rhi
