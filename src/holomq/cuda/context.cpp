/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cuda/context.hpp"

#include <holomq/logger.hpp>

#include <cuda.h>

#include <utility>

namespace holomq {
namespace cuda {

Context::Context(CUdevice device, unsigned int flags) noexcept {
  CUcontext ctx{};
#if defined(CUDA_VERSION) && (CUDA_VERSION >= 13000)
  if (cuCtxCreate(&ctx, nullptr, flags, device) != CUDA_SUCCESS) {
#else
  if (cuCtxCreate(&ctx, flags, device) != CUDA_SUCCESS) {
#endif
    holomq::log_literal(holomq::LogLevel::Error,
                        "cuCtxCreate failed during holomq::cuda::Context construction");
    return;
  }
  ctx_ = ctx;
}

Context::Context(Context&& other) noexcept : ctx_(other.ctx_) {
  other.ctx_ = nullptr;
}

Context& Context::operator=(Context&& other) noexcept {
  if (this != &other) {
    if (static_cast<bool>(*this)) {
      (void)cuCtxDestroy(ctx_);
    }
    ctx_ = other.ctx_;
    other.ctx_ = nullptr;
  }
  return *this;
}

Context::~Context() noexcept {
  if (!static_cast<bool>(*this)) {
    return;
  }
  (void)cuCtxDestroy(ctx_);
}

}  // namespace cuda
}  // namespace holomq
