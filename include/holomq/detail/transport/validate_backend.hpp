/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <holomq/detail/transport/concepts.hpp>

#include <cstdint>

namespace holomq::detail::transport::contract {

/// Trivial payload used only for compile-time backend contract checks.
struct Payload {
  std::uint64_t field{0};
};

template <typename PayloadT, typename Loan, typename InMsg, typename PubPort, typename SubPort,
          typename RuntimeT>
consteval bool validate_backend() {
  static_assert(concepts::OutboundLoanLike<Loan, PayloadT>);
  static_assert(concepts::InboundMessageLike<InMsg, PayloadT>);
  static_assert(concepts::PublisherPortLike<PubPort, PayloadT, Loan>);
  static_assert(concepts::SubscriberPortLike<SubPort, InMsg>);
  static_assert(concepts::WaitableSubscriberPortLike<SubPort, InMsg>);
  static_assert(concepts::RuntimeLike<RuntimeT, PayloadT, PubPort, SubPort>);
  return true;
}

}  // namespace holomq::detail::transport::contract
