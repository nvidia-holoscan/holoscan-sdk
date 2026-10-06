/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "channel_readiness_handle.hpp"

#include <memory>
#include <vector>

namespace holoscan {

class ChannelReadinessHandle::Impl {};

ChannelReadinessHandle::ChannelReadinessHandle(ChannelReadinessHandle&& other) noexcept = default;
ChannelReadinessHandle& ChannelReadinessHandle::operator=(ChannelReadinessHandle&& other) noexcept =
    default;
ChannelReadinessHandle::~ChannelReadinessHandle() = default;

int ChannelReadinessHandle::descriptor() const noexcept {
  return -1;
}

expected<void, Error> ChannelReadinessHandle::acknowledge() noexcept {
  return make_unexpected(Error{ErrorCode::kClosed});
}

void ChannelReadinessHandle::detach() noexcept {
  impl_.reset();
}

expected<ChannelReadinessHandle, Error> channel_internal::ChannelReadinessHandleAccess::create(
    std::vector<holomq::ReadinessHandle>
        holomq_handles,  // NOLINT(performance-unnecessary-value-param)
    std::vector<ChannelReadinessHandle>
        channel_handles,  // NOLINT(performance-unnecessary-value-param)
    const bool initially_ready) {
  static_cast<void>(initially_ready);
  if (holomq_handles.empty() && channel_handles.empty()) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  // Destroying the retained source handles on return releases any claims acquired before this
  // platform capability was checked.
  return make_unexpected(Error{ErrorCode::kNotSupported});
}

}  // namespace holoscan
