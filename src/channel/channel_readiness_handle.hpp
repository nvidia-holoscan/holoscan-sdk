/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <holomq/readiness_handle.hpp>
#include <holoscan/channel/readiness_handle.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>

#include <vector>

namespace holoscan::channel_internal {

/// @brief Construct a public readiness handle from source-private lane handles.
class ChannelReadinessHandleAccess {
 public:
  /// @brief Aggregate HoloMQ and nested Channel readiness descriptors without a pump thread.
  /// @param holomq_handles Direct transport-lane handles to retain.
  /// @param channel_handles Composite Channel handles to retain.
  /// @param initially_ready Whether source-private staged work must prime the descriptor.
  /// @return A public readiness handle, or a checked descriptor setup error.
  [[nodiscard]] static expected<ChannelReadinessHandle, Error> create(
      std::vector<holomq::ReadinessHandle> holomq_handles,
      std::vector<ChannelReadinessHandle> channel_handles = {}, bool initially_ready = false);
};

}  // namespace holoscan::channel_internal
