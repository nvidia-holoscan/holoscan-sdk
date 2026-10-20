/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>

namespace holomq {

/// Default maximum subscriber count used by the iceoryx2 pub/sub service configuration.
///
/// HoloMQ leaves the backend setting unchanged when \ref PubSubServiceOptions::max_subscribers is
/// zero. Keep setup-side storage sized for that resolved default rather than treating zero as a
/// single-subscriber service.
inline constexpr std::uint32_t k_default_max_subscribers = 8U;

/// Per-subscriber sample queue depth wired into the iceoryx2 pub/sub backend (for example
/// \c subscriber_max_buffer_size and subscriber \c buffer_size).
inline constexpr std::uint32_t k_subscriber_queue_depth = 8U;

/// Default borrowed-sample allowance for each subscriber.
///
/// \c 0 means "derive from the configured subscriber queue depth" so callers can receive and hold
/// queued samples unless they explicitly tighten the borrowed-sample allowance. Borrowed capacity
/// remains independent of whether a drop-oldest queue replaces unread samples.
inline constexpr std::uint32_t k_subscriber_max_borrowed_samples = 0U;

/// Publisher history retained for late-joining subscribers (for example iceoryx2 \c history_size).
/// \c 0 disables publisher-side history.
inline constexpr std::uint32_t k_publisher_history_depth = 0U;

/// When \c true, the iceoryx2 backend disables sample overflow and retries until delivery
/// succeeds instead of dropping.
inline constexpr bool k_lossless_flow = true;

/// Default derived iceoryx2 \c max_loaned_samples under the current \ref k_lossless_flow setting.
///
/// This publisher-loan limit is not a general \ref BufferPool slot-count recommendation. An
/// external pool may also need backing for publisher history and for unread or borrowed samples
/// retained by every subscriber.
inline constexpr std::uint32_t k_max_in_flight =
    k_lossless_flow
        ? (k_subscriber_queue_depth > k_publisher_history_depth ? k_subscriber_queue_depth
                                                                : k_publisher_history_depth)
        : k_subscriber_queue_depth;

}  // namespace holomq
