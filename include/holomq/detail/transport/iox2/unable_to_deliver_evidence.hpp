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

/// What the synchronous handler recorded about the recipients it rejected during one send.
struct UnableToDeliverEvidence {
  std::optional<DeliveryRecipientId> recipient;  ///< The last recipient that rejected the sample.
  bool cancelled{};  ///< A bounded-retry cancellation flag rejected a recipient within budget.
  bool rejected{};   ///< An expired budget or a reject-when-full policy rejected a recipient.

  /// Whether cancellation, and nothing else, rejected the send's recipients.
  [[nodiscard]] bool cancelled_only() const noexcept { return cancelled && !rejected; }
};

[[nodiscard]] inline UnableToDeliverEvidence& unable_to_deliver_evidence_storage() noexcept {
  // The callback executes synchronously inside send(), so thread-local storage correlates evidence
  // without serializing independent publishers or adding state to their pressure-free path. The
  // recipient and its causes are one record, taken together, so neither outlives the other.
  static thread_local UnableToDeliverEvidence evidence;
  return evidence;
}

inline void record_unable_to_deliver_recipient(const DeliveryRecipientId& recipient,
                                               const bool cancelled = false) noexcept {
  UnableToDeliverEvidence& evidence = unable_to_deliver_evidence_storage();
  evidence.recipient = recipient;
  (cancelled ? evidence.cancelled : evidence.rejected) = true;
}

/// Consume the evidence only for a publisher that installed the synchronous handler.
///
/// A handler-free publisher can otherwise inherit stale thread-local evidence left by a bounded
/// retry that ultimately succeeded on the same thread.
[[nodiscard]] inline UnableToDeliverEvidence take_unable_to_deliver_evidence_if(
    const bool handler_installed) noexcept {
  return handler_installed ? std::exchange(unable_to_deliver_evidence_storage(), {})
                           : UnableToDeliverEvidence{};
}

/// Consume the evidence, as above, and return only its recipient.
[[nodiscard]] inline std::optional<DeliveryRecipientId> take_unable_to_deliver_recipient_if(
    const bool handler_installed) noexcept {
  return take_unable_to_deliver_evidence_if(handler_installed).recipient;
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
