/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "channel_publisher_pressure.hpp"

#include <holoscan/channel/endpoint_interfaces.hpp>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>

namespace holoscan::channel_internal {

/// Source-private auxiliary interface for a logical subscriber with a protected terminal lane.
class ProtectedTerminalWaitSetSubscriber {
 public:
  virtual ~ProtectedTerminalWaitSetSubscriber() = default;
  [[nodiscard]] virtual std::optional<std::uint64_t> protected_terminal_wait_handle()
      const noexcept = 0;
};

/// Source-private auxiliary interface for protected terminal publication.
class ProtectedTerminalInlinePublisher {
 public:
  virtual ~ProtectedTerminalInlinePublisher() = default;
  [[nodiscard]] virtual InlinePublicationResult emit_protected_terminal(
      std::span<const std::byte> payload, const SampleMetadata& metadata) = 0;
};

[[nodiscard]] inline std::optional<std::uint64_t> protected_terminal_wait_handle(
    WaitSetSubscriber& subscriber) noexcept {
  const auto* const protected_subscriber =
      dynamic_cast<const ProtectedTerminalWaitSetSubscriber*>(&subscriber);
  return protected_subscriber != nullptr ? protected_subscriber->protected_terminal_wait_handle()
                                         : std::nullopt;
}

[[nodiscard]] inline InlinePublicationResult emit_protected_terminal(
    InlinePayloadPublisher& publisher, const std::span<const std::byte> payload,
    const SampleMetadata& metadata) {
  auto* const protected_publisher = dynamic_cast<ProtectedTerminalInlinePublisher*>(&publisher);
  if (protected_publisher == nullptr) {
    return InlinePublicationResult{
        .phase = InlinePublicationPhase::kPreSubmitRejected,
        .error =
            Error{ErrorCode::kProtocolError, "PROTECTED_TERMINAL_PUBLISHER_CAPABILITY_MISSING"},
    };
  }
  return protected_publisher->emit_protected_terminal(payload, metadata);
}

[[nodiscard]] constexpr bool terminal_publication_error_consumes_authority(
    const ErrorCode code) noexcept {
  return publication_error_consumes_authority(code) || code == ErrorCode::kProtocolError ||
         code == ErrorCode::kInvalidArgument;
}

}  // namespace holoscan::channel_internal
