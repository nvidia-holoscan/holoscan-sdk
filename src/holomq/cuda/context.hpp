/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda.h>

namespace holomq {
namespace cuda {

/// Owns a CUDA driver context from \c cuCtxCreate; calls \c cuCtxDestroy on destruction. If
/// creation fails, the object is invalid (\c operator bool() is \c false) and the destructor is a
/// no-op.
///
/// The caller must call \c cuInit and \c cuDeviceGet (or otherwise obtain a \c CUdevice) before
/// construction. This type does not call \c cuCtxSetCurrent; the application must set the context
/// current when required.
class Context {
 public:
  Context() noexcept = default;

  /// Calls \c cuCtxCreate for \p device (CUDA 13+: \c cuCtxCreate(&ctx, nullptr, flags, device);
  /// earlier:
  /// \c cuCtxCreate(&ctx, flags, device)). On failure the object remains invalid.
  explicit Context(CUdevice device, unsigned int flags = CU_CTX_SCHED_AUTO) noexcept;

  Context(const Context&) = delete;
  Context& operator=(const Context&) = delete;

  Context(Context&& other) noexcept;
  Context& operator=(Context&& other) noexcept;

  ~Context() noexcept;

  [[nodiscard]] explicit operator bool() const noexcept { return ctx_ != nullptr; }

  [[nodiscard]] CUcontext get() const noexcept { return ctx_; }

 private:
  CUcontext ctx_{nullptr};
};

}  // namespace cuda
}  // namespace holomq
