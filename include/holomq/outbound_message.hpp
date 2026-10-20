/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <holomq/byte_slice.hpp>
#include <holomq/detail/transport/backend.hpp>

#include <utility>

namespace holomq {

template <typename Payload>
class Publisher;

/// Move-only outbound message loan from \ref Publisher::loan.
///
/// Fill via \ref payload(), then pass to \ref Publisher::send. Obtain only from
/// \ref Publisher::loan (not default-constructible). For \ref ByteSlice, \ref payload() is the
/// loaned frame of the length requested from \ref Publisher::loan.
///
/// \note Destroying without \ref Publisher::send returns the loan to the transport; the sample is
/// not published.
template <typename Payload>
class OutboundMessage {
 public:
  OutboundMessage(const OutboundMessage&) = delete;
  OutboundMessage& operator=(const OutboundMessage&) = delete;

  OutboundMessage(OutboundMessage&&) noexcept = default;
  OutboundMessage& operator=(OutboundMessage&&) noexcept = default;
  ~OutboundMessage() = default;

  [[nodiscard]] MutablePayloadView<Payload> payload() noexcept { return loan_.payload(); }

  [[nodiscard]] ConstPayloadView<Payload> payload() const noexcept { return loan_.payload(); }

 private:
  friend class Publisher<Payload>;

  OutboundMessage() = delete;

  explicit OutboundMessage(detail::transport::OutboundLoan<Payload> loan) noexcept
      : loan_(std::move(loan)) {}

  [[nodiscard]] detail::transport::OutboundLoan<Payload> release_loan() && noexcept {
    return std::move(loan_);
  }

  [[nodiscard]] const void* release_key() const noexcept { return loan_.release_key(); }

  detail::transport::OutboundLoan<Payload> loan_;
};

}  // namespace holomq
