/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <chrono>
#include <thread>
#include <utility>

#include <holoscan/channel/channel.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>

#include "endpoint_catalog_state.hpp"

namespace holoscan::channel_internal {

/// @brief Resolve one caller override against a catalog-frozen attachment bound.
[[nodiscard]] inline expected<std::chrono::nanoseconds, Error> catalog_attach_bound(
    const detail::ChannelEndpointRecord& endpoint, const ChannelAttachOptions& options) noexcept {
  if (!endpoint.attach.max_wait || endpoint.attach.max_wait->count() <= 0) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument, "ENDPOINT_ATTACH_BOUND_INVALID"});
  }
  const std::chrono::nanoseconds maximum = *endpoint.attach.max_wait;
  if (!options.timeout) {
    return maximum;
  }
  if (options.timeout->count() <= 0 || *options.timeout > maximum) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument, "ENDPOINT_ATTACH_BOUND_INVALID"});
  }
  return *options.timeout;
}

/// @brief Poll one endpoint-specific peer observation under its catalog-frozen bound.
template <typename ObservePeer>
[[nodiscard]] expected<void, Error> wait_for_catalog_required_peer(
    const detail::ChannelEndpointRecord& endpoint, const ChannelAttachOptions& options,
    ObservePeer observe_peer) {
  if (endpoint.attach.peer == PeerRequirement::kOptional) {
    return {};
  }

  constexpr std::chrono::nanoseconds kPollInterval = std::chrono::milliseconds{1};
  auto bound = catalog_attach_bound(endpoint, options);
  if (!bound) {
    return make_unexpected(std::move(bound).error());
  }
  const auto started = std::chrono::steady_clock::now();
  while (true) {
    auto observed = observe_peer();
    if (!observed) {
      return make_unexpected(std::move(observed).error());
    }
    if (*observed) {
      return {};
    }

    const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - started);
    if (elapsed >= *bound) {
      return make_unexpected(Error{ErrorCode::kTimeout, "ENDPOINT_REQUIRED_PEER_TIMEOUT"});
    }
    std::this_thread::sleep_for(std::min(kPollInterval, *bound - elapsed));
  }
}

}  // namespace holoscan::channel_internal
