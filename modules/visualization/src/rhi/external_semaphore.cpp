/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "holoviz/rhi/external_semaphore.hpp"

#include <memory>
#include <utility>

#include "detail/external_semaphore_impl.hpp"
#include "detail/handle_errors.hpp"

namespace holoviz::rhi {

ExternalSemaphore::ExternalSemaphore(std::unique_ptr<detail::ExternalSemaphoreImpl> impl)
    : impl_(std::move(impl)) {}
ExternalSemaphore::ExternalSemaphore() = default;
ExternalSemaphore::ExternalSemaphore(ExternalSemaphore&&) noexcept = default;
ExternalSemaphore& ExternalSemaphore::operator=(ExternalSemaphore&&) noexcept = default;
ExternalSemaphore::~ExternalSemaphore() = default;

expected<void> ExternalSemaphore::destroy(std::chrono::nanoseconds timeout) {
  if (!impl_) {
    return {};  // empty handle: nothing to free
  }
  expected<void> result = impl_->destroy(timeout);
  if (result || result.error().code != ErrorCode::kBackpressured) {
    impl_.reset();
  }
  return result;
}

expected<void> ExternalSemaphore::wait(const SubmitSyncToken& sync_token) {
  if (!impl_) {
    return detail::null_impl_error("ExternalSemaphore::wait");
  }
  return impl_->wait(sync_token);
}

expected<ExternalSyncToken> ExternalSemaphore::signal() {
  if (!impl_) {
    return detail::null_impl_error("ExternalSemaphore::signal");
  }
  return impl_->signal();
}

expected<CUstream> ExternalSemaphore::stream() const {
  if (!impl_) {
    return detail::null_impl_error("ExternalSemaphore::stream");
  }
  return impl_->stream();
}

}  // namespace holoviz::rhi
