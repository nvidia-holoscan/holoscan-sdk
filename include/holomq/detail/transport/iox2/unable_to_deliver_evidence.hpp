/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <holomq/delivery_attribution.hpp>

#include <algorithm>
#include <array>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

namespace holomq::detail::transport {

[[nodiscard]] inline std::optional<DeliveryRecipientId>&
unable_to_deliver_recipient_storage() noexcept {
  // The callback executes synchronously inside send(), so thread-local storage correlates evidence
  // without serializing independent publishers or adding state to their pressure-free path.
  static thread_local std::optional<DeliveryRecipientId> recipient;
  return recipient;
}

inline void record_unable_to_deliver_recipient(const DeliveryRecipientId& recipient) noexcept {
  unable_to_deliver_recipient_storage() = recipient;
}

[[nodiscard]] inline std::optional<DeliveryRecipientId>
take_unable_to_deliver_recipient() noexcept {
  return std::exchange(unable_to_deliver_recipient_storage(), std::nullopt);
}

/// Consume receiver attribution only for a publisher that installed the synchronous handler.
///
/// A handler-free publisher can otherwise inherit stale thread-local evidence left by a bounded
/// retry that ultimately succeeded on the same thread.
[[nodiscard]] inline std::optional<DeliveryRecipientId> take_unable_to_deliver_recipient_if(
    const bool handler_installed) noexcept {
  return handler_installed ? take_unable_to_deliver_recipient() : std::nullopt;
}

[[nodiscard]] inline std::string unable_to_deliver_recipient_diagnostic(
    const DeliveryRecipientId& recipient) {
  constexpr std::string_view kPrefix{"HOLOMQ_BACKPRESSURED recipient="};
  constexpr std::array kHexDigits{
      '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f'};
  std::array<char, kPrefix.size() + DeliveryRecipientId::kSize * 2U> formatted{};
  std::copy(kPrefix.begin(), kPrefix.end(), formatted.begin());
  std::size_t offset = kPrefix.size();
  for (const std::uint8_t byte : recipient.bytes) {
    formatted[offset++] = kHexDigits[byte >> 4U];
    formatted[offset++] = kHexDigits[byte & 0x0FU];
  }
  return std::string(formatted.data(), formatted.size());
}

}  // namespace holomq::detail::transport
