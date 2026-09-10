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
#include <utility>

namespace holoscan::channel_internal {

struct StagedDataState {
  std::uint64_t sequence{};
  TimePoint capture_time{};
};

/// Source-private capability for a subscriber that can retain one validated data head.
class StagedDataWaitSetSubscriber {
 public:
  virtual ~StagedDataWaitSetSubscriber() = default;
  [[nodiscard]] virtual std::optional<StagedDataState> staged_data_state() const noexcept = 0;
  [[nodiscard]] virtual std::optional<std::uint64_t> staged_terminal_sequence() const noexcept = 0;
};

/// Source-private auxiliary interface for a logical subscriber with a protected terminal lane.
class ProtectedTerminalWaitSetSubscriber {
 public:
  virtual ~ProtectedTerminalWaitSetSubscriber() = default;
  [[nodiscard]] virtual std::optional<std::uint64_t> protected_terminal_wait_handle()
      const noexcept = 0;
};

[[nodiscard]] inline std::optional<StagedDataState> staged_data_state(
    WaitSetSubscriber& subscriber) noexcept {
  const auto* const staged_subscriber =
      dynamic_cast<const StagedDataWaitSetSubscriber*>(&subscriber);
  return staged_subscriber != nullptr ? staged_subscriber->staged_data_state() : std::nullopt;
}

[[nodiscard]] inline std::optional<std::uint64_t> staged_terminal_sequence(
    WaitSetSubscriber& subscriber) noexcept {
  const auto* const staged_subscriber =
      dynamic_cast<const StagedDataWaitSetSubscriber*>(&subscriber);
  return staged_subscriber != nullptr ? staged_subscriber->staged_terminal_sequence()
                                      : std::nullopt;
}

[[nodiscard]] inline bool supports_staged_data(WaitSetSubscriber& subscriber) noexcept {
  return dynamic_cast<const StagedDataWaitSetSubscriber*>(&subscriber) != nullptr;
}

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

struct ProtectedTerminalStaging {
  bool data{};
  bool terminal{};
};

/// Stage one data/terminal transaction across independent lossy transport lanes.
///
/// A protected terminal can become visible after the first data-lane probe. Stage data, run the
/// source-private inter-probe seam, stage terminal, and re-stage data exactly once only in that
/// race window. Catalog generation, health, and sequence arbitration remain the caller's concern.
template <typename StageData, typename StageTerminal, typename AfterEmptyDataProbe>
// These callables remain stable lvalues because the data probe may execute twice around the
// terminal race window; forwarding either invocation could consume callback state.
// NOLINTBEGIN(cppcoreguidelines-missing-std-forward)
[[nodiscard]] expected<ProtectedTerminalStaging, Error> stage_protected_terminal_transaction(
    StageData&& stage_data, StageTerminal&& stage_terminal,
    AfterEmptyDataProbe&& after_empty_data_probe) noexcept {
  auto data_staged = stage_data();
  if (!data_staged) {
    return make_unexpected(std::move(data_staged).error());
  }
  if (!*data_staged) {
    after_empty_data_probe();
  }
  auto terminal_staged = stage_terminal();
  if (!terminal_staged) {
    return make_unexpected(std::move(terminal_staged).error());
  }
  if (!*data_staged && *terminal_staged) {
    data_staged = stage_data();
    if (!data_staged) {
      return make_unexpected(std::move(data_staged).error());
    }
  }
  return ProtectedTerminalStaging{.data = *data_staged, .terminal = *terminal_staged};
}
// NOLINTEND(cppcoreguidelines-missing-std-forward)

}  // namespace holoscan::channel_internal
