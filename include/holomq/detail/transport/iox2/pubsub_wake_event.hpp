/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <iox2/event_id.hpp>

#include <cstdint>

namespace holomq::detail::transport {

enum class PubSubWakeEvent : std::uint8_t {
  SentSample = 0,
};

inline auto make_sent_sample_event_id() -> ::iox2::EventId {
  return ::iox2::EventId(static_cast<std::size_t>(PubSubWakeEvent::SentSample));
}

}  // namespace holomq::detail::transport
