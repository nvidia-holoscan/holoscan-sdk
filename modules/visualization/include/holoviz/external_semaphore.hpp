/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// ExternalSemaphore: the CUDA edge of the timeline synchronization model,
// created from a CUstream (Device::create_external_semaphore). wait() makes
// the stream wait for a submit's SubmitSyncToken; signal() enqueues a signal
// on the stream and returns the ExternalSyncToken a later submit can wait on
// (SubmitDesc::wait_external_sync_tokens). One timeline vocabulary covers
// GPU→GPU, GPU→host, and GPU↔CUDA ordering — no fences, no binary semaphores.

#pragma once

#include <chrono>
#include <memory>

#include <cuda.h>

#include <holoviz/holoviz_export.h>

#include "common.hpp"

namespace holoviz {

namespace detail {
class ExternalSemaphoreImpl;
}  // namespace detail

class ExternalSemaphore {
 public:
  HOLOVIZ_EXPORT explicit ExternalSemaphore(std::unique_ptr<detail::ExternalSemaphoreImpl> impl);
  HOLOVIZ_EXPORT ExternalSemaphore();
  HOLOVIZ_EXPORT ExternalSemaphore(ExternalSemaphore&&) noexcept;
  HOLOVIZ_EXPORT ExternalSemaphore& operator=(ExternalSemaphore&&) noexcept;
  ExternalSemaphore(const ExternalSemaphore&) = delete;
  ExternalSemaphore& operator=(const ExternalSemaphore&) = delete;
  HOLOVIZ_EXPORT ~ExternalSemaphore();

  explicit operator bool() const { return impl_ != nullptr; }

  /// Release the underlying backend resources now instead of at destruction.
  /// \p timeout bounds how long to wait for the object to stop being in use:
  /// zero (the default) fails immediately with ErrorCode::kBackpressured if
  /// the object is in use; a non-zero timeout waits up to that duration before
  /// returning ErrorCode::kBackpressured. The semaphore is in use until every
  /// signal() enqueued on its CUDA stream has completed and every
  /// Device::submit() that waited on one of its tokens has completed on the
  /// GPU. On success (and on any
  /// error other than kBackpressured) the handle becomes empty. Safe to call
  /// on an empty handle (no-op success). The destructor calls this
  /// automatically and logs any error.
  [[nodiscard]] HOLOVIZ_EXPORT expected<void> destroy(
      std::chrono::nanoseconds timeout = std::chrono::nanoseconds::zero());

  [[nodiscard]] HOLOVIZ_EXPORT expected<void> wait(const SubmitSyncToken& sync_token);
  [[nodiscard]] HOLOVIZ_EXPORT expected<ExternalSyncToken> signal();

  [[nodiscard]] HOLOVIZ_EXPORT expected<CUstream> stream() const;

  /// Backend-internal access; not part of the stable API surface.
  /// Precondition: the handle is non-empty.
  detail::ExternalSemaphoreImpl& impl() { return *impl_; }
  const detail::ExternalSemaphoreImpl& impl() const { return *impl_; }

  /// Backend-internal; not part of the stable API surface. Abandon the handle WITHOUT
  /// destroying the underlying resource: unretired GPU/CUDA work may still reference it,
  /// so a deliberate leak beats a use-after-free. Allocation-free and no-throw by
  /// construction — safe inside terminal error paths. The caller must poison the owning
  /// device (DeviceImpl::report_leaked_resources()) so it never destroys the leaked
  /// child's native parents.
  void leak() noexcept { [[maybe_unused]] auto* abandoned = impl_.release(); }

 private:
  std::unique_ptr<detail::ExternalSemaphoreImpl> impl_;
};

}  // namespace holoviz
