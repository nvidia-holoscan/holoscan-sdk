/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holomq/readiness_handle.hpp>

#include <memory>

namespace holomq {

class ReadinessHandle::Impl {};

ReadinessHandle::ReadinessHandle(ReadinessHandle&& other) noexcept = default;
ReadinessHandle& ReadinessHandle::operator=(ReadinessHandle&& other) noexcept = default;
ReadinessHandle::~ReadinessHandle() = default;

expected<ReadinessHandle, Error> ReadinessHandle::create(
    std::shared_ptr<void> owner,  // NOLINT(performance-unnecessary-value-param)
    const int source_descriptor, [[maybe_unused]] const bool initially_ready,
    const AcknowledgeFn acknowledge, const DetachFn detach) {
  if (owner == nullptr || source_descriptor < 0 || acknowledge == nullptr || detach == nullptr) {
    if (owner != nullptr && detach != nullptr) {
      detach(owner.get());
    }
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }

  // A caller that acquired a backend claim before discovering the platform boundary must not leave
  // the subscriber owned after this checked failure.
  detach(owner.get());
  return make_unexpected(Error{ErrorCode::kNotSupported});
}

int ReadinessHandle::descriptor() const noexcept {
  return -1;
}

expected<void, Error> ReadinessHandle::acknowledge() noexcept {
  return make_unexpected(Error{ErrorCode::kClosed});
}

void ReadinessHandle::detach() noexcept {
  impl_.reset();
}

}  // namespace holomq
