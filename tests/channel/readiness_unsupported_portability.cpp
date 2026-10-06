/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holomq/readiness_handle.hpp>
#include <holoscan/channel/readiness_handle.hpp>

#include <utility>

namespace {

// REQ: CHAN-148
[[nodiscard]] holomq::ReadinessHandle move_holomq_handle(holomq::ReadinessHandle&& handle) {
  return std::move(handle);
}

void use_holomq_handle(holomq::ReadinessHandle* const handle) {
  static_cast<void>(handle->descriptor());
  static_cast<void>(handle->acknowledge());
  handle->detach();
  handle->~ReadinessHandle();
}

[[nodiscard]] holoscan::ChannelReadinessHandle move_channel_handle(
    holoscan::ChannelReadinessHandle&& handle) {
  return std::move(handle);
}

void use_channel_handle(holoscan::ChannelReadinessHandle* const handle) {
  static_cast<void>(handle->descriptor());
  static_cast<void>(handle->acknowledge());
  handle->detach();
  handle->~ChannelReadinessHandle();
}

}  // namespace

int main() {
  auto* volatile holomq_move = &move_holomq_handle;
  auto* volatile holomq_use = &use_holomq_handle;
  auto* volatile channel_move = &move_channel_handle;
  auto* volatile channel_use = &use_channel_handle;
  return holomq_move == nullptr || holomq_use == nullptr || channel_move == nullptr ||
                 channel_use == nullptr
             ? 1
             : 0;
}
