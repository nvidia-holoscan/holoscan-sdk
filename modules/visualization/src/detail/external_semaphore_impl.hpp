/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <chrono>

#include <cuda.h>

#include <holoviz/external_semaphore.hpp>

namespace holoviz::detail {

/// Backend interface for ExternalSemaphore; see the handle for API docs.
class ExternalSemaphoreImpl {
 public:
  ExternalSemaphoreImpl(const ExternalSemaphoreImpl&) = delete;
  ExternalSemaphoreImpl& operator=(const ExternalSemaphoreImpl&) = delete;
  ExternalSemaphoreImpl(ExternalSemaphoreImpl&&) = delete;
  ExternalSemaphoreImpl& operator=(ExternalSemaphoreImpl&&) = delete;
  virtual ~ExternalSemaphoreImpl() = default;

  /// Release all owned backend resources now. \p timeout bounds how long to
  /// wait for the object to stop being in use: zero (the default) fails
  /// immediately with ErrorCode::kBackpressured if the object is in use, and a
  /// non-zero timeout waits up to that duration before returning
  /// ErrorCode::kBackpressured. In use means pending signal() work on the CUDA
  /// stream or an incomplete Device::submit() that waited on one of the
  /// semaphore's tokens. Idempotent: repeated calls are no-ops
  /// returning success. Called by the destructor if not already destroyed.
  [[nodiscard]] virtual expected<void> destroy(std::chrono::nanoseconds timeout) = 0;

  [[nodiscard]] virtual expected<void> wait(const SubmitSyncToken& sync_token) = 0;
  [[nodiscard]] virtual expected<ExternalSyncToken> signal() = 0;
  [[nodiscard]] virtual expected<CUstream> stream() const = 0;

 protected:
  ExternalSemaphoreImpl() = default;
};

}  // namespace holoviz::detail
