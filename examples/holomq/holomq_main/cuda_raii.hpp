/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda.h>

namespace holomq_main {

/// Example-local CUDA driver context RAII (public HoloMQ API does not expose this yet).
class DemoCudaContext {
 public:
  DemoCudaContext() noexcept = default;

  explicit DemoCudaContext(CUdevice device, unsigned int flags = CU_CTX_SCHED_AUTO) noexcept {
    CUcontext ctx{};
#if defined(CUDA_VERSION) && (CUDA_VERSION >= 13000)
    if (cuCtxCreate(&ctx, nullptr, flags, device) != CUDA_SUCCESS) {
#else
    if (cuCtxCreate(&ctx, flags, device) != CUDA_SUCCESS) {
#endif
      return;
    }
    ctx_ = ctx;
  }

  DemoCudaContext(const DemoCudaContext&) = delete;
  DemoCudaContext& operator=(const DemoCudaContext&) = delete;

  DemoCudaContext(DemoCudaContext&& other) noexcept : ctx_(other.ctx_) { other.ctx_ = nullptr; }

  DemoCudaContext& operator=(DemoCudaContext&& other) noexcept {
    if (this != &other) {
      destroy();
      ctx_ = other.ctx_;
      other.ctx_ = nullptr;
    }
    return *this;
  }

  ~DemoCudaContext() noexcept { destroy(); }

  [[nodiscard]] explicit operator bool() const noexcept { return ctx_ != nullptr; }

  [[nodiscard]] CUcontext get() const noexcept { return ctx_; }

 private:
  void destroy() noexcept {
    if (ctx_ == nullptr) {
      return;
    }
    (void)cuCtxDestroy(ctx_);
    ctx_ = nullptr;
  }

  CUcontext ctx_{nullptr};
};

/// Example-local stream RAII. Synchronizes before destroy so HoloMQ deferred ref releases
/// enqueued on the stream complete (same contract as the library's internal helper).
class DemoCudaStream {
 public:
  DemoCudaStream() noexcept = default;

  explicit DemoCudaStream(unsigned int flags) noexcept {
    CUstream stream{};
    if (cuStreamCreate(&stream, flags) != CUDA_SUCCESS) {
      return;
    }
    stream_ = stream;
  }

  DemoCudaStream(const DemoCudaStream&) = delete;
  DemoCudaStream& operator=(const DemoCudaStream&) = delete;

  DemoCudaStream(DemoCudaStream&& other) noexcept : stream_(other.stream_) {
    other.stream_ = nullptr;
  }

  DemoCudaStream& operator=(DemoCudaStream&& other) noexcept {
    if (this != &other) {
      drain_and_destroy();
      stream_ = other.stream_;
      other.stream_ = nullptr;
    }
    return *this;
  }

  ~DemoCudaStream() noexcept { drain_and_destroy(); }

  [[nodiscard]] explicit operator bool() const noexcept { return stream_ != nullptr; }

  [[nodiscard]] CUstream get() const noexcept { return stream_; }

 private:
  void drain_and_destroy() noexcept {
    if (stream_ == nullptr) {
      return;
    }
    (void)cuStreamSynchronize(stream_);
    (void)cuStreamDestroy(stream_);
    stream_ = nullptr;
  }

  CUstream stream_{};
};

}  // namespace holomq_main
