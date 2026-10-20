/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <chrono>
#include <cstdint>
#include <optional>

#include <holoscan/time/clock_registry.hpp>

namespace holoscan {

/// @brief Require capture timestamps in one declared clock domain.
struct RequireCaptureTime {
  ClockKey clock;  ///< Cold named/default clock key copied into Graph state.
};

/// @brief Select whether absence of an external peer is admissible.
enum class PeerRequirement : std::uint8_t {
  kOptional = 0,  ///< Graph start never waits for or fails on an absent peer.
  kRequired = 1,  ///< Explicit attach/wait may report timeout at the accepted bound.
};
/// @brief Carry the endpoint peer requirement and finite attach/wait bound.
///
/// The default optional peer never makes Graph composition or Graph start wait
/// for an external process. A positive \ref max_wait is retained only as a bound
/// for later explicit attach/service verification and required-peer waiting.
struct TopicAttachPolicy {
  PeerRequirement peer{PeerRequirement::kOptional};  ///< Whether an absent external peer is valid.
  std::optional<std::chrono::nanoseconds> max_wait;  ///< Positive finite attach bound when present.
};
/// @brief Bound publisher-local external endpoint claims.
struct PublisherEndpointBounds {
  std::optional<std::uint32_t> max_loaned_samples;  ///< Positive finite external publisher claims.
};
/// @brief Bound subscriber-local external endpoint queues and claims.
struct SubscriberEndpointBounds {
  std::optional<std::uint32_t> queue_depth;  ///< Positive finite external subscriber queue depth.
  std::optional<std::uint32_t> max_borrowed_samples;  ///< Positive finite read claims.
};
/// @brief Carry the direction-specific external endpoint bounds.
struct EndpointBounds {
  std::optional<PublisherEndpointBounds> publisher;    ///< External publisher bounds for imports.
  std::optional<SubscriberEndpointBounds> subscriber;  ///< External subscriber bounds for exports.
};

}  // namespace holoscan
